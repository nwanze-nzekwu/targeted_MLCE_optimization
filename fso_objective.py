import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb
import os
import warnings
import json

# --- Dependency Imports ---
# NOTE: These custom dependencies must be available in the environment.
from my_pyt_lms import my_pyt_lms
from rytov_vs_latency import rytov_vs_latency

warnings.filterwarnings('ignore')

# ============================================================================
# DEFAULT CONFIGURATION
# These can be overridden by arguments passed to the objective_function.
# ============================================================================
DEFAULT_CONFIG = {
    'DATA_DIR': 'data/',
    'DATASET_FILE': 'lin_wan5_weak_turb_samps.mat',
    'DATASET_VAR': 'lin_wan5_w_dat',
    'DATASET_NAME': 'Weak Turbulence',
    'FS_MEAS': 1e4,
    'FS': 1e4 / 1,
    'N_TRAIN': 100000,
    'USE_DIFFERENTIAL': True,
    'RF_N_ESTIMATORS': 100,
    'XGB_N_ESTIMATORS': 100,
    'CB_ITERATIONS': 100,
    'OUTPUT_DIR': './output/',
    'RESULTS_FILENAME_BASE': 'fso_metrics_run',
    # Adaptive training-size selection controls
    'N_TRAIN_MIN': 5000,                 # minimum absolute training samples to consider
    'N_TRAIN_VAL_FRACTION': 0.1,         # fraction of candidate train size used as validation
    'N_TRAIN_VAL_MAX': 10000,            # maximum validation size
    'N_TRAIN_TOL': 0.02,                 # tolerance over best validation RMSE to accept smaller train size (2%)
    'N_TRAIN_CANDIDATE_FRACS': [0.1, 0.25, 0.5, 0.75, 1.0],  # fractions of cap to probe
}

# ============================================================================
# HELPER FUNCTIONS (Minimal set required for execution)
# ============================================================================

pow2db = lambda x: 10 * np.log10(x)
db2pow = lambda x: 10 ** (x / 10)

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_fso_data(data_dir, filename, var_name, fs_meas, fs):
    """Load FSO signal data from .mat file and preprocess."""
    filepath = os.path.join(data_dir, filename)
    try:
        mat_data = scipy.io.loadmat(filepath)
        data = mat_data[var_name].flatten()
        
        # Downsample and convert to dB/center
        X = data[::int(fs_meas/fs)]
        wa = pow2db(X) - np.mean(pow2db(X))
        
        return wa
    except FileNotFoundError:
        raise FileNotFoundError(f"Data files not found in {data_dir}.")
    except KeyError:
        raise KeyError(f"Variable '{var_name}' not found in {filename}.")

def create_lagged_features(signal, latency, n_taps, use_differential):
    """Create enhanced lagged feature matrix with rolling statistics and derived features."""
    df = pd.DataFrame({'OptPow': signal})
    
    # Basic differential features
    df['OptPow_diff'] = df['OptPow'].diff() if use_differential else df['OptPow']
    
    # Second-order differences (acceleration/rate of change)
    df['OptPow_diff2'] = df['OptPow_diff'].diff()
    
    # === Core Lagged Features ===
    for lag in range(latency, latency + n_taps):
        df[f'OptPow_diff_lag{lag}'] = df['OptPow_diff'].shift(lag)
        
        # Add squared terms for non-linearity
        df[f'OptPow_diff_lag{lag}_sq'] = df[f'OptPow_diff_lag{lag}'] ** 2
    
    # Add second-order difference lags (capture acceleration patterns)
    for lag in range(latency, latency + min(n_taps, 5)):
        df[f'OptPow_diff2_lag{lag}'] = df['OptPow_diff2'].shift(lag)
    
    # === Rolling Window Statistics ===
    # Multiple window sizes to capture patterns at different time scales
    rolling_windows = [3, 5, 10, 20]
    
    for window in rolling_windows:
        # Only compute if window is reasonable relative to latency
        if window < latency:
            # Calculate rolling statistics on the original signal
            rolling = df['OptPow'].shift(latency).rolling(window=window, min_periods=1)
            df[f'OptPow_rolling_mean_{window}'] = rolling.mean()
            df[f'OptPow_rolling_std_{window}'] = rolling.std()
            df[f'OptPow_rolling_min_{window}'] = rolling.min()
            df[f'OptPow_rolling_max_{window}'] = rolling.max()
            
            # Rolling range (max - min) as a volatility measure
            df[f'OptPow_rolling_range_{window}'] = (
                df[f'OptPow_rolling_max_{window}'] - df[f'OptPow_rolling_min_{window}']
            )
    
    # === Exponentially Weighted Moving Averages ===
    # EWMA gives more weight to recent observations
    ewma_spans = [5, 10, 20]
    
    for span in ewma_spans:
        if span < latency:
            # Calculate EWMA and shift appropriately to avoid leakage
            df[f'OptPow_ewma_{span}'] = (
                df['OptPow'].shift(latency).ewm(span=span, adjust=False).mean()
            )
            
            # EWMA of differential signal
            df[f'OptPow_diff_ewma_{span}'] = (
                df['OptPow_diff'].shift(latency).ewm(span=span, adjust=False).mean()
            )
    
    # === Interaction Features ===
    # Create a few strategic interaction terms between recent lags
    if n_taps >= 2:
        df[f'OptPow_diff_lag{latency}_lag{latency+1}_interact'] = (
            df[f'OptPow_diff_lag{latency}'] * df[f'OptPow_diff_lag{latency+1}']
        )
    
    # === Reference lag and target ===
    df[f'OptPow_lag{latency}'] = df['OptPow'].shift(latency)
    
    if use_differential:
        df[f'OptPow_{latency}stepdiff_target'] = df['OptPow'] - df[f'OptPow_lag{latency}']
    else:
        df[f'OptPow_{latency}stepdiff_target'] = df['OptPow']
    
    return df.dropna()

def find_min_effective_train_size(df, latency, n_taps, config):
    """
    Determine the smallest training size that achieves near-best validation RMSE,
    using Linear Regression as a fast proxy model.
    """
    max_cap = min(int(config.get('N_TRAIN', 100000)), len(df) - 1000)
    if max_cap <= 0:
        return 0

    min_size = int(config.get('N_TRAIN_MIN', 5000))
    cand_fracs = config.get('N_TRAIN_CANDIDATE_FRACS', [0.1, 0.25, 0.5, 0.75, 1.0])
    # Build candidate sizes bounded by min_size and max_cap
    candidates = sorted(set([max(min_size, int(max_cap * f)) for f in cand_fracs] + [max_cap]))
    candidates = [n for n in candidates if n <= max_cap]

    # Build feature list mirroring training
    def build_feature_columns(df_ref):
        cols = [f'OptPow_diff_lag{i}' for i in range(latency, latency + n_taps)]
        cols.extend([f'OptPow_diff_lag{i}_sq' for i in range(latency, latency + n_taps)])
        cols.extend([f'OptPow_diff2_lag{i}' for i in range(latency, latency + min(n_taps, 5))])

        for window in [3, 5, 10, 20]:
            if window < latency:
                cols.extend([
                    f'OptPow_rolling_mean_{window}',
                    f'OptPow_rolling_std_{window}',
                    f'OptPow_rolling_min_{window}',
                    f'OptPow_rolling_max_{window}',
                    f'OptPow_rolling_range_{window}'
                ])

        for span in [5, 10, 20]:
            if span < latency:
                cols.extend([
                    f'OptPow_ewma_{span}',
                    f'OptPow_diff_ewma_{span}'
                ])

        if n_taps >= 2:
            cols.append(f'OptPow_diff_lag{latency}_lag{latency+1}_interact')

        return [c for c in cols if c in df_ref.columns]

    feature_columns = build_feature_columns(df)
    target_column = f'OptPow_{latency}stepdiff_target'
    if len(feature_columns) == 0 or target_column not in df.columns:
        return max_cap

    tol = float(config.get('N_TRAIN_TOL', 0.02))
    val_frac = float(config.get('N_TRAIN_VAL_FRACTION', 0.1))
    val_max = int(config.get('N_TRAIN_VAL_MAX', 10000))

    results = []
    for n in candidates:
        n_val = min(val_max, max(1000, int(n * val_frac)))
        if n_val >= n:
            continue
        n_fit = n - n_val

        df_tr = df.iloc[:n_fit]
        df_val = df.iloc[n_fit:n]

        try:
            X_tr = df_tr[feature_columns].values
            y_tr = df_tr[target_column].values
            X_val = df_val[feature_columns].values
            y_val_true = df_val['OptPow'].values

            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_val_diff = model.predict(X_val)
            if config.get('USE_DIFFERENTIAL', True):
                y_val_pred = y_val_diff + df_val[f'OptPow_lag{latency}'].values
            else:
                y_val_pred = y_val_diff

            rmse = calculate_rmse(y_val_true, y_val_pred)
            results.append((n, rmse))
        except Exception:
            # If any candidate fails due to insufficient features/rows, skip it
            continue

    if not results:
        return max_cap

    best_rmse = min(r for _, r in results)
    threshold = best_rmse * (1.0 + tol)

    # Select the smallest candidate within tolerance of best RMSE
    for n, r in sorted(results, key=lambda t: (t[0], t[1])):
        if r <= threshold:
            return n

    return max_cap

def train_and_evaluate_models(df_train, df_test, latency, n_taps, config):
    """Train all models and calculate RMSE."""
    
    # Build comprehensive feature list including all enhanced features
    feature_columns = [f'OptPow_diff_lag{i}' for i in range(latency, latency + n_taps)]
    
    # Add squared lag features
    feature_columns.extend([f'OptPow_diff_lag{i}_sq' for i in range(latency, latency + n_taps)])
    
    # Add second-order difference lags
    feature_columns.extend([f'OptPow_diff2_lag{i}' for i in range(latency, latency + min(n_taps, 5))])
    
    # Add rolling statistics features
    rolling_windows = [3, 5, 10, 20]
    for window in rolling_windows:
        if window < latency:
            feature_columns.extend([
                f'OptPow_rolling_mean_{window}',
                f'OptPow_rolling_std_{window}',
                f'OptPow_rolling_min_{window}',
                f'OptPow_rolling_max_{window}',
                f'OptPow_rolling_range_{window}'
            ])
    
    # Add EWMA features
    ewma_spans = [5, 10, 20]
    for span in ewma_spans:
        if span < latency:
            feature_columns.extend([
                f'OptPow_ewma_{span}',
                f'OptPow_diff_ewma_{span}'
            ])
    
    # Add interaction features
    if n_taps >= 2:
        feature_columns.append(f'OptPow_diff_lag{latency}_lag{latency+1}_interact')
    
    # Filter to only include columns that exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df_train.columns]
    
    target_column = f'OptPow_{latency}stepdiff_target'
    
    X_train = df_train[feature_columns].values
    y_train = df_train[target_column].values
    X_test = df_test[feature_columns].values
    y_true = df_test['OptPow'].values
    
    results = {}
    
    # --- Least Mean Square (only use basic lagged features for compatibility) ---
    basic_features = [f'OptPow_diff_lag{i}' for i in range(latency, latency + n_taps)]
    X_train_lms = df_train[basic_features].values
    X_test_lms = df_test[basic_features].values
    train_and_test_lms_model(df_test, latency, n_taps, config, X_train_lms, y_train, X_test_lms, results)
    
    # --- Linear Regression ---
    train_and_test_lr_model(df_test, latency, config, X_train, y_train, X_test, results)
    
    # --- Random Forest ---
    train_and_test_rf_model(df_test, latency, config, X_train, y_train, X_test, results)
    
    # --- XGBoost ---
    train_and_test_xgb_model(df_test, latency, config, X_train, y_train, X_test, results)
    
    # --- CatBoost ---
    train_and_test_cb_model(df_test, latency, config, X_train, y_train, X_test, results)
    
    # --- Zero Order Hold ---
    pred_zoh = df_test[f'OptPow_lag{latency}'].values
    results['zoh'] = pred_zoh
    
    # --- CALCULATE Root Mean Square Error (RMSE) ---
    rmse_results = {model: calculate_rmse(y_true, results[model]) for model in results if model != 'y_true'}
    
    results['rmse'] = rmse_results
    results['y_true'] = y_true 
    
    return results

# Train and test Least Mean Square model
def train_and_test_lms_model(df_test, latency, n_taps, config, X_train, y_train, X_test, results):
    y_tr, err_tr, wts_tr = my_pyt_lms(X_train, y_train, n_taps, None, False)
    wts = wts_tr[-1, :]
    yt, e, wts_final = my_pyt_lms(X_test, None, n_taps, wts, True)
    pred_lms = yt + df_test[f'OptPow_lag{latency}'].values if config['USE_DIFFERENTIAL'] else yt
    results['lms'] = pred_lms

# Train and test Linear Regression model
def train_and_test_lr_model(df_test, latency, config, X_train, y_train, X_test, results):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    pred_lr_diff = model_lr.predict(X_test)
    pred_lr = pred_lr_diff + df_test[f'OptPow_lag{latency}'].values if config['USE_DIFFERENTIAL'] else pred_lr_diff
    results['lr'] = pred_lr

# Train and test Random Forest model
def train_and_test_rf_model(df_test, latency, config, X_train, y_train, X_test, results):
    model_rf = RandomForestRegressor(n_estimators=config['RF_N_ESTIMATORS'], random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    pred_rf_diff = model_rf.predict(X_test)
    pred_rf = pred_rf_diff + df_test[f'OptPow_lag{latency}'].values if config['USE_DIFFERENTIAL'] else pred_rf_diff
    results['rf'] = pred_rf

# Train and test XGBoost model
def train_and_test_xgb_model(df_test, latency, config, X_train, y_train, X_test, results):
    model_xgb = xgb.XGBRegressor(n_estimators=config['XGB_N_ESTIMATORS'], random_state=42, n_jobs=-1)
    model_xgb.fit(X_train, y_train)
    pred_xgb_diff = model_xgb.predict(X_test)
    pred_xgb = pred_xgb_diff + df_test[f'OptPow_lag{latency}'].values if config['USE_DIFFERENTIAL'] else pred_xgb_diff
    results['xgb'] = pred_xgb

# Train and test CatBoost model
def train_and_test_cb_model(df_test, latency, config, X_train, y_train, X_test, results):
    model_cb = cb.CatBoostRegressor(iterations=config['CB_ITERATIONS'], verbose=False, random_state=42)
    model_cb.fit(X_train, y_train)
    pred_cb_diff = model_cb.predict(X_test)
    pred_cb = pred_cb_diff + df_test[f'OptPow_lag{latency}'].values if config['USE_DIFFERENTIAL'] else pred_cb_diff
    results['cb'] = pred_cb




# --- CALCULATE Rytov Variance  ---
def calculate_rytov_metrics(results):
    """Calculate Rytov variance for precompensated signals."""
    y_true = results['y_true']
    rytov_results = {}
    
    models = ['lms', 'lr', 'rf', 'xgb', 'cb', 'zoh']
    for model in models:
        precom_error = y_true - results[model]
        # rytov_vs_latency returns a tuple (variance, mean, ...) - we take the tuple
        rytov_results[model] = rytov_vs_latency(db2pow(precom_error)) 
    
    rytov_results['input'] = rytov_vs_latency(db2pow(y_true))
    
    return rytov_results


def save_json_results(all_results, output_dir, filename, config, latency_values):
    """Saves the final results to a JSON file."""
    
    log_data = {
        'metadata': {
            'dataset_name': config['DATASET_NAME'],
            'taps_n': config['N_TAPS'],
            'n_train': config['N_TRAIN'],
            'latency_samples_tested': latency_values,
        },
        'results_by_latency': {}
    }

    for lat in latency_values:
        if lat in all_results:
            lat_results = all_results[lat]
            
            # Rytov variance is the first element of the tuple from rytov_vs_latency
            rytov_clean = {
                model: lat_results['rytov'][model][0] 
                for model in lat_results['rytov']
            }

            log_data['results_by_latency'][str(lat)] = {
                'rmse': lat_results['rmse'],
                'rytov_variance': rytov_clean
            }
        
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"File saved successfully: {filepath}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective_function(
    n_taps, 
    n_train, 
    latency_list, 
    rf_estimators=DEFAULT_CONFIG['RF_N_ESTIMATORS'], 
    xgb_estimators=DEFAULT_CONFIG['XGB_N_ESTIMATORS'], 
    cb_iterations=DEFAULT_CONFIG['CB_ITERATIONS']
):
    """
    Executes the FSO channel estimation benchmark for optimization.

    This function runs the full benchmark, saves the results to a JSON file, 
    and returns a structured dictionary of metrics.

    Args:
        n_taps (int): Filter memory length / number of lagged features.
        n_train (int): Number of samples for training.
        latency_list (list): List of prediction horizons in samples.
        rf_estimators (int): Number of estimators for Random Forest.
        xgb_estimators (int): Number of estimators for XGBoost.
        cb_iterations (int): Number of iterations for CatBoost.

    Returns:
        dict: A nested dictionary containing RMSE and Rytov Variance for all 
              models across all tested latencies.
    """
    
    # 1. SETUP CONFIGURATION
    config = DEFAULT_CONFIG.copy()
    config['N_TAPS'] = int(n_taps)
    config['N_TRAIN'] = int(n_train)
    config['RF_N_ESTIMATORS'] = int(rf_estimators)
    config['XGB_N_ESTIMATORS'] = int(xgb_estimators)
    config['CB_ITERATIONS'] = int(cb_iterations)
    
    latency_values = [latency_list] if isinstance(latency_list, int) else latency_list
    
    # Generate unique filename based on key parameters
    run_id = f"t{n_taps}_tr{int(n_train/1000)}k_rf{rf_estimators}_xgb{xgb_estimators}_cb{cb_iterations}"
    results_filename = f"{config['RESULTS_FILENAME_BASE']}_{run_id}.json"

    print(f"Starting FSO Benchmark Run: {run_id}")
    
    # 2. LOAD DATA
    try:
        wa = load_fso_data(
            config['DATA_DIR'], config['DATASET_FILE'], config['DATASET_VAR'],
            config['FS_MEAS'], config['FS']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return {} # Return empty dict on critical error

    # 3. PROCESS AND EVALUATE
    all_results = {}
    
    for lat in latency_values:
        # 3.1 Create features
        df = create_lagged_features(wa, lat, config['N_TAPS'], config['USE_DIFFERENTIAL'])
        
        # 3.2 Choose minimal effective training size and split data
        n_train_cap = min(config['N_TRAIN'], len(df) - 1000)
        n_train_eff = find_min_effective_train_size(df, lat, config['N_TAPS'], config)
        if n_train_eff is None or n_train_eff <= 0:
            n_train_eff = n_train_cap
        df_train = df.iloc[:n_train_eff]
        df_test = df.iloc[n_train_eff:]
        print(f"Latency {lat}: using effective training size {n_train_eff} (cap {n_train_cap})")
        
        # 3.3 Train and evaluate
        results = train_and_evaluate_models(df_train, df_test, lat, config['N_TAPS'], config)
        
        # 3.4 Calculate Rytov metrics
        rytov_results = calculate_rytov_metrics(results)
        
        # Store only the necessary metrics
        all_results[lat] = {
            'rmse': results['rmse'],
            'rytov': rytov_results
        }
    
    # 4. SAVE RESULTS TO JSON FILE
    save_json_results(all_results, config['OUTPUT_DIR'], results_filename, config, latency_values)

    # 5. RETURN SIMPLIFIED METRICS DICTIONARY
    
    # Final structure must be simple {latency: {model: {metric: value}}}
    final_metrics = {}
    
    for lat, result_data in all_results.items():
        lat_key = str(lat)
        final_metrics[lat_key] = {}
        
        # Combine RMSE and Rytov Variance (taking index 0 of the Rytov tuple)
        for model in result_data['rmse'].keys():
            final_metrics[lat_key][model] = {
                'rmse': result_data['rmse'][model],
                'rytov_variance': result_data['rytov'][model][0]
            }
            
        # Add input Rytov variance for reference
        final_metrics[lat_key]['input_rytov_variance'] = result_data['rytov']['input'][0]

    return final_metrics