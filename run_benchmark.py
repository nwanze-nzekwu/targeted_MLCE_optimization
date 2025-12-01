import argparse
import json
import sys
import os

# Command line usage example: python run_benchmark.py --taps 15 --latencies 10


# Ensure the directory of this script is in the path to find fso_objective
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the objective function and its default config for use as argument defaults
from fso_objective import objective_function, DEFAULT_CONFIG

def main():
    """
    Main function to parse command-line arguments and run the FSO objective function.
    """
    parser = argparse.ArgumentParser(
        description="Run the FSO Channel Estimation benchmark and log results. Requires external files: fso_objective.py, my_pyt_lms.py, and rytov_vs_latency.py."
    )

    # Required Arguments now made optional with defaults from fso_objective.py
    # NOTE: We use a default list [5, 20, 50] for latencies, which is a common test case.
    parser.add_argument(
        '--taps', 
        type=int, 
        default=10, # Hardcoded default for standalone run
        required=False, 
        help='Filter memory length / number of lagged features (n_taps). Default: 10'
    )
    parser.add_argument(
        '--train_samples', 
        type=int, 
        default=DEFAULT_CONFIG['N_TRAIN'],
        required=False, 
        help='Number of samples for training (n_train). Default: 100000'
    )
    
    # Latency argument - accepts multiple values separated by commas
    # Making this optional, defaulting to a common test set like "5,20,50"
    parser.add_argument(
        '--latencies', 
        type=str, 
        default="5,20,50", 
        required=False, 
        help='Comma-separated list of prediction horizons in samples (e.g., "5,20,50"). Default: "5,20,50"'
    )

    # Optional Hyperparameters (kept the same, but specifying defaults for clarity)
    parser.add_argument(
        '--rf_estimators', 
        type=int, 
        default=DEFAULT_CONFIG['RF_N_ESTIMATORS'], 
        help='Number of estimators for Random Forest (default: 100).'
    )
    parser.add_argument(
        '--xgb_estimators', 
        type=int, 
        default=DEFAULT_CONFIG['XGB_N_ESTIMATORS'], 
        help='Number of estimators for XGBoost (default: 100).'
    )
    parser.add_argument(
        '--cb_iterations', 
        type=int, 
        default=DEFAULT_CONFIG['CB_ITERATIONS'], 
        help='Number of iterations for CatBoost (default: 100).'
    )

    args = parser.parse_args()

    # Process latency list from string
    try:
        latency_list = [int(x.strip()) for x in args.latencies.split(',')]
    except ValueError:
        print("Error: Latencies must be a comma-separated list of integers.")
        sys.exit(1)

    print("--- FSO Benchmark: Starting Objective Function Execution ---")
    print(f"Parameters: Taps={args.taps}, Training Samples={args.train_samples}, Latencies={latency_list}")
    
    # Call the objective function, which handles the internal logging to JSON file.
    results = objective_function(
        n_taps=args.taps,
        n_train=args.train_samples,
        latency_list=latency_list,
        rf_estimators=args.rf_estimators,
        xgb_estimators=args.xgb_estimators,
        cb_iterations=args.cb_iterations
    )

    # Print the returned results to stdout
    if results:
        print("\n--- RESULTS JSON OUTPUT (Printed to Screen) ---")
        print(json.dumps(results, indent=4))
        print("-----------------------------------------------")
        print("Final metrics successfully computed, logged to the output directory, and printed.")
    else:
        print("\n--- EXECUTION ERROR ---")
        print("Benchmark failed or returned no metrics. Check data file paths and dependencies.")
        sys.exit(1)


if __name__ == '__main__':
    main()