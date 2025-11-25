"""
FSO Channel Estimation with Multiple Machine Learning Models

This script analyzes Free Space Optical (FSO) communication signal data to predict
signal attenuation across different time horizons. It loads FSO power measurements,
applies feature engineering with time-lagged differencing, trains multiple machine
learning models (LMS, Linear Regression, Random Forest, XGBoost, CatBoost, and
Zero-Order Hold baseline), and evaluates their performance using RMSE and Rytov
variance metrics.

Key Features:
- Loads .mat files containing FSO signal measurements at 10 kHz sampling
- Creates stationary features using first-order differencing
- Builds time-lagged feature matrices for prediction
- Trains 6 different models for attenuation prediction
- Evaluates performance across different latency horizons
- Generates comparison plots of RMSE vs. latency
- Calculates turbulence metrics using Rytov variance

Configuration parameters are set at the top of the script for easy modification.
"""

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb
import os
import warnings

# Import custom functions (dependencies from previous tasks)
from my_pyt_lms import my_pyt_lms
from rytov_vs_latency import rytov_vs_latency

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data parameters

# datapath = r'C:\Users\wanzay\OneDrive - Universidade de Aveiro\Desktop\Course module Materials\Marco NN\NN-FSO\NN-FSO REP\ML_Channel_Estimation\data'
DATA_DIR = '../data/'
# Options: lin_wan5_mod_turb_samps.mat, 
#          lin_wan5_strong_turb_samps.mat,
#          lin_wan5_weak_turb_samps.mat

DATASET_FILE = 'lin_wan5_strong_turb_samps.mat'                                                
DATASET_VAR = 'lin_wan5_s_dat'  # Variable name in .mat file
DATASET_NAME = 'Strong Turbulence'

# DATASET_FILE = 'lin_wan5_mod_turb_samps.mat'                                                
# DATASET_VAR = 'lin_wan5_m_dat'  # Variable name in .mat file
# DATASET_NAME = 'Moderate Turbulence'

# DATASET_FILE = 'lin_wan5_weak_turb_samps.mat'                                                
# DATASET_VAR = 'lin_wan5_w_dat'  # Variable name in .mat file
# DATASET_NAME = 'Weak Turbulence'

# Signal parameters
FS_MEAS = 1e4  # Measurement sampling frequency (10 kHz)
FS = FS_MEAS / 1  # Processing sampling frequency

# Model parameters
LATENCY = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100, 150]#20  # Prediction horizon in samples (can be list: [1, 5, 10, 15, 20, 25, 30, 35, 40, 50])
N_TAPS = 10   # Filter memory length / number of lagged features

# Training parameters
N_TRAIN = 100000  # Number of samples for training
USE_DIFFERENTIAL = True  # Use first-order differencing for stationarity

# Model hyperparameters
RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
CB_ITERATIONS = 100

# Visualization
VISUAL_DEBUG = True
OUTPUT_DIR = './output/'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Power/dB conversion functions
pow2db = lambda x: 10 * np.log10(x)
db2pow = lambda x: 10 ** (x / 10)

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    # return mean_squared_error(y_true, y_pred, squared=False) # deprecated 
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_fso_data(data_dir, filename, var_name):
    """
    Load FSO signal data from .mat file
    
    Args:
        data_dir: Directory containing data files
        filename: Name of .mat file
        var_name: Variable name in .mat file
        
    Returns:
        Loaded data as numpy array
        
    Raises:
        FileNotFoundError: If data file not found
    """
    filepath = os.path.join(data_dir, filename)
    
    try:
        mat_data = scipy.io.loadmat(filepath)
        data = mat_data[var_name].flatten()
        print(f"Successfully loaded {filename}")
        print(f"Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data files not found in {data_dir}. "
            "Please provide .mat files with FSO signal measurements."
        )
    except KeyError:
        raise KeyError(
            f"Variable '{var_name}' not found in {filename}. "
            "Please check the variable name in the .mat file."
        )

def create_lagged_features(signal, latency, n_taps, use_differential=True):
    """
    Create lagged feature matrix from signal using first-order differencing
    
    Args:
        signal: Input signal (power in dB)
        latency: Prediction horizon
        n_taps: Number of lagged features
        use_differential: Whether to use first-order differencing
        
    Returns:
        df: DataFrame with features and target
    """
    # Create DataFrame with original signal
    df = pd.DataFrame({'OptPow': signal})
    
    # Apply first-order differencing if enabled
    if use_differential:
        df['OptPow_diff'] = df['OptPow'].diff()
    else:
        df['OptPow_diff'] = df['OptPow']
    
    # Create lagged features
    for lag in range(latency, latency + n_taps):
        df[f'OptPow_diff_lag{lag}'] = df['OptPow_diff'].shift(lag)
    
    # Create target (lagged original signal for ZOH reference)
    df[f'OptPow_lag{latency}'] = df['OptPow'].shift(latency)
    
    # Create target variable (difference to predict)
    if use_differential:
        df[f'OptPow_{latency}stepdiff_target'] = df['OptPow'] - df[f'OptPow_lag{latency}']
    else:
        df[f'OptPow_{latency}stepdiff_target'] = df['OptPow']
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def train_and_evaluate_models(df_train, df_test, latency, n_taps, use_differential=True):
    """
    Train all models and evaluate on test set
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        latency: Prediction horizon
        n_taps: Number of features
        use_differential: Whether differential mode is used
        
    Returns:
        results: Dictionary containing predictions and metrics
    """
    # Define feature columns
    feature_columns = [f'OptPow_diff_lag{i}' for i in range(latency, latency + n_taps)]
    target_column = f'OptPow_{latency}stepdiff_target'
    
    # Extract features and targets
    X_train = df_train[feature_columns].values
    y_train = df_train[target_column].values
    X_test = df_test[feature_columns].values
    y_test = df_test[target_column].values
    
    results = {}
    
    print(f"\nTraining models for latency = {latency} samples, nTaps = {n_taps}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # ========================================================================
    # 1. LMS ADAPTIVE FILTER
    # ========================================================================
    print("  Training LMS Adaptive Filter...")
    
    # Train LMS
    lock_coeff = False
    weights = None
    # y_tr, err_tr, wts_tr = my_pyt_lms(X_train, y_train, latency, n_taps, weights, lock_coeff)
    y_tr, err_tr, wts_tr = my_pyt_lms(X_train, y_train, n_taps, weights, lock_coeff)
    
    # Test LMS with locked coefficients
    lock_coeff = True
    wts = wts_tr[-1, :]  # Use the last, most updated weights
    # yt, e, wts_final = my_pyt_lms(X_test, None, latency, n_taps, wts, lock_coeff)
    yt, e, wts_final = my_pyt_lms(X_test, None, n_taps, wts, lock_coeff)
    
    # Convert predictions to original scale
    if use_differential:
        predictions_lms = yt + df_test[f'OptPow_lag{latency}'].values
    else:
        predictions_lms = yt
    
    results['lms'] = predictions_lms
    
    # ========================================================================
    # 2. LINEAR REGRESSION
    # ========================================================================
    print("  Training Linear Regression...")
    
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    predictions_lr_diff = model_lr.predict(X_test)
    
    # Convert predictions to original scale
    if use_differential:
        predictions_lr = predictions_lr_diff + df_test[f'OptPow_lag{latency}'].values
    else:
        predictions_lr = predictions_lr_diff
    
    results['lr'] = predictions_lr
    
    # ========================================================================
    # 3. RANDOM FOREST
    # ========================================================================
    print("  Training Random Forest...")
    
    model_rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    predictions_rf_diff = model_rf.predict(X_test)
    
    # Convert predictions to original scale
    if use_differential:
        predictions_rf = predictions_rf_diff + df_test[f'OptPow_lag{latency}'].values
    else:
        predictions_rf = predictions_rf_diff
    
    results['rf'] = predictions_rf
    
    # ========================================================================
    # 4. XGBOOST
    # ========================================================================
    print("  Training XGBoost...")
    
    model_xgb = xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATORS, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train, y_train)
    predictions_xgb_diff = model_xgb.predict(X_test)
    
    # Convert predictions to original scale
    if use_differential:
        predictions_xgb = predictions_xgb_diff + df_test[f'OptPow_lag{latency}'].values
    else:
        predictions_xgb = predictions_xgb_diff
    
    results['xgb'] = predictions_xgb
    
    # ========================================================================
    # 5. CATBOOST
    # ========================================================================
    print("  Training CatBoost...")
    
    model_cb = cb.CatBoostRegressor(iterations=CB_ITERATIONS, verbose=False, random_state=42)
    model_cb.fit(X_train, y_train)
    predictions_cb_diff = model_cb.predict(X_test)
    
    # Convert predictions to original scale
    if use_differential:
        predictions_cb = predictions_cb_diff + df_test[f'OptPow_lag{latency}'].values
    else:
        predictions_cb = predictions_cb_diff
    
    results['cb'] = predictions_cb
    
    # ========================================================================
    # 6. ZERO-ORDER HOLD (ZOH) BASELINE
    # ========================================================================
    print("  Computing Zero-Order Hold baseline...")
    
    # ZOH simply uses the lagged value as prediction
    predictions_zoh = df_test[f'OptPow_lag{latency}'].values
    
    results['zoh'] = predictions_zoh
    
    # ========================================================================
    # CALCULATE RMSE FOR ALL MODELS
    # ========================================================================
    
    y_true = df_test['OptPow'].values
    
    rmse_results = {
        'lms': calculate_rmse(y_true, predictions_lms),
        'lr': calculate_rmse(y_true, predictions_lr),
        'rf': calculate_rmse(y_true, predictions_rf),
        'xgb': calculate_rmse(y_true, predictions_xgb),
        'cb': calculate_rmse(y_true, predictions_cb),
        'zoh': calculate_rmse(y_true, predictions_zoh)
    }
    
    print(f"\n  RMSE Results:")
    for model_name, rmse_val in rmse_results.items():
        print(f"    {model_name.upper():6s}: {rmse_val:.6f}")
    
    results['rmse'] = rmse_results
    results['y_true'] = y_true
    
    return results

def calculate_rytov_metrics(results, latency):
    """
    Calculate Rytov variance for precompensated signals
    
    Args:
        results: Dictionary with predictions and true values
        latency: Prediction horizon
        
    Returns:
        rytov_results: Dictionary with Rytov variance values
    """
    print(f"\n  Calculating Rytov variance metrics...")
    
    y_true = results['y_true']
    rytov_results = {}
    
    # Calculate precompensation error (residual) for each model
    precom_lms = y_true - results['lms']
    precom_lr = y_true - results['lr']
    precom_rf = y_true - results['rf']
    precom_xgb = y_true - results['xgb']
    precom_cb = y_true - results['cb']
    precom_zoh = y_true - results['zoh']
    
    # Convert to power and calculate Rytov variance
    rytov_results['lms'] = rytov_vs_latency(db2pow(precom_lms))
    rytov_results['lr'] = rytov_vs_latency(db2pow(precom_lr))
    rytov_results['rf'] = rytov_vs_latency(db2pow(precom_rf))
    rytov_results['xgb'] = rytov_vs_latency(db2pow(precom_xgb))
    rytov_results['cb'] = rytov_vs_latency(db2pow(precom_cb))
    rytov_results['zoh'] = rytov_vs_latency(db2pow(precom_zoh))
    rytov_results['input'] = rytov_vs_latency(db2pow(y_true))
    
    # print(f"    Input Rytov variance: {rytov_results['input']:.6f}")
    print(f"    Input Rytov variance: {rytov_results['input'][0]:.6f}")

    for model_name in ['lms', 'lr', 'rf', 'xgb', 'cb', 'zoh']:
        # print(f"    {model_name.upper():6s} Rytov variance: {rytov_results[model_name]:.6f}")
        print(f"    {model_name.upper():6s} Rytov variance: {rytov_results[model_name][0]:.6f}")
    
    return rytov_results

def plot_results(all_results, latency_values, dataset_name):
    """
    Generate plots comparing model performance across latency values
    
    Args:
        all_results: Dictionary containing results for each latency value
        latency_values: List of latency values tested
        dataset_name: Name of dataset for plot title
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract RMSE values for each model
    models = ['lms', 'lr', 'rf', 'xgb', 'cb', 'zoh']
    model_labels = {
        'lms': 'LMS',
        'lr': 'Linear Regression',
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'cb': 'CatBoost',
        'zoh': 'ZOH'
    }
    
    rmse_data = {model: [] for model in models}
    rytov_data = {model: [] for model in models}
    rytov_input = []
    
    for lat in latency_values:
        if lat in all_results:
            for model in models:
                rmse_data[model].append(all_results[lat]['rmse'][model])
                rytov_data[model].append(all_results[lat]['rytov'][model])
            # rytov_input.append(all_results[lat]['rytov']['input'])
            rytov_input.append(all_results[lat]['rytov']['input'][0])  # Adjusted for use rytov of original input signal... before lagging.
    
    # ========================================================================
    # PLOT 1: RMSE vs. Latency
    # ========================================================================
    plt.figure(figsize=(10, 6))
    
    markers = {
        'lms': '^-',
        'lr': '*-',
        'rf': 's-',
        'xgb': 'o-',
        'cb': 'd-',
        'zoh': '+-'
    }
    
    for model in models:
        plt.plot(latency_values, rmse_data[model], markers[model], 
                linewidth=1.5, markersize=6, label=model_labels[model])
    
    plt.xlabel('Estimation Latency [samples]', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'RMSE vs. Latency [{dataset_name}]', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plot_filename = os.path.join(OUTPUT_DIR, f'rmse_vs_latency_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\nSaved RMSE plot to: {plot_filename}")
    
    # ========================================================================
    # PLOT 2: Rytov Variance vs. Latency
    # ========================================================================
    plt.figure(figsize=(10, 6))
    
    # Plot input variance as reference
    latency_ms = np.array(latency_values) / (FS / 1000)  # Convert to milliseconds
    plt.plot(latency_ms, rytov_input, '--', linewidth=2, label='Input Variance')
    
    # Plot model Rytov variances (only selected models)
    # selected_models = ['lms', 'lr', 'zoh']  # Match MATLAB script
    # Dont match MATLAB script... instead plot all models for completeness
    for model in models:
        y_values = [a for a, b in rytov_data[model]]  # use lists comprehension instead to extract rytov variance,that is the first element of each tuple from the lists in the dictionary.
        plt.plot(latency_ms, y_values, markers[model],
                linewidth=1.5, markersize=6, label=model_labels[model])
        
        # plt.plot(latency_ms, rytov_data[model], markers[model],
        #         linewidth=1.5, markersize=6, label=model_labels[model])
        # Note: rytov_data[model] is a list of lists, we take the first element for plotting.
    
    plt.xlabel('Estimation Latency [ms]', fontsize=12)
    plt.ylabel('Rytov Variance', fontsize=12)
    plt.title(f'Rytov Variance [{dataset_name}]', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plot_filename = os.path.join(OUTPUT_DIR, f'rytov_vs_latency_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"Saved Rytov variance plot to: {plot_filename}")
    
    if VISUAL_DEBUG:
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("FSO Channel Estimation with Machine Learning Models")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Latency: {LATENCY} samples")
    print(f"Number of taps: {N_TAPS}")
    print(f"Training samples: {N_TRAIN}")
    print(f"Use differential: {USE_DIFFERENTIAL}")
    print("="*80)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n[1/5] Loading FSO signal data...")
    
    try:
        X = load_fso_data(DATA_DIR, DATASET_FILE, DATASET_VAR)
    except Exception as e:
        print(f"\nError loading data: {e}")
        return
    
    # Downsample if needed
    X = X[::int(FS_MEAS/FS)]
    
    # Convert to dB and center (remove mean)
    wa = pow2db(X) - np.mean(pow2db(X))
    
    print(f"Signal length: {len(wa)} samples")
    print(f"Signal statistics: mean={np.mean(wa):.6f}, std={np.std(wa):.6f}")
    
    # ========================================================================
    # PROCESS MULTIPLE LATENCY VALUES
    # ========================================================================
    
    # Convert LATENCY to list if it's a single value
    latency_values = [LATENCY] if isinstance(LATENCY, int) else LATENCY
    
    all_results = {}
    
    for lat in latency_values:
        print(f"\n{'='*80}")
        print(f"[2/5] Processing latency = {lat} samples")
        print(f"{'='*80}")
        
        # ====================================================================
        # CREATE FEATURES
        # ====================================================================
        print("\n[3/5] Creating lagged features...")
        
        df = create_lagged_features(wa, lat, N_TAPS, USE_DIFFERENTIAL)
        
        print(f"Feature matrix shape: {df.shape}")
        print(f"Available samples after lagging: {len(df)}")
        
        # ====================================================================
        # SPLIT DATA
        # ====================================================================
        print("\n[4/5] Splitting into train/test sets...")
        
        n_train = min(N_TRAIN, len(df) - 1000)  # Ensure we have test samples
        df_train = df.iloc[:n_train]
        df_test = df.iloc[n_train:]
        
        print(f"Training set: {len(df_train)} samples")
        print(f"Test set: {len(df_test)} samples")
        
        # ====================================================================
        # TRAIN AND EVALUATE MODELS
        # ====================================================================
        print("\n[5/5] Training and evaluating models...")
        
        results = train_and_evaluate_models(df_train, df_test, lat, N_TAPS, USE_DIFFERENTIAL)
        
        # Calculate Rytov metrics
        rytov_results = calculate_rytov_metrics(results, lat)
        results['rytov'] = rytov_results
        
        all_results[lat] = results
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    
    if len(latency_values) > 1:
        plot_results(all_results, latency_values, DATASET_NAME)
    else:
        print("\nNote: Multiple latency values needed for comparison plots.")
        print("To generate plots, set LATENCY to a list like [1, 5, 10, 15, 20, 25, 30]")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nProcessed {len(latency_values)} latency value(s)")
    print(f"Trained 6 models per latency")
    print(f"Total evaluations: {len(latency_values) * 6}")
    
    # Print final RMSE summary
    print("\nFinal RMSE Summary:")
    for lat in latency_values:
        print(f"\n  Latency = {lat} samples:")
        for model in ['lms', 'lr', 'rf', 'xgb', 'cb', 'zoh']:
            rmse = all_results[lat]['rmse'][model]
            print(f"    {model.upper():6s}: {rmse:.6f}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()





# To Do:
# in the next iteration by Artermis optimizer...
# Include the option to specify the number of datapoints to load from the .mat file
# Include the option to specify train/test split ratio, as well as validity set.
# Enable inclusion or exclusion of features, e.g., moving average, moving std, spectral features, etc.