"""
Rytov Variance Estimation from Optical Power Measurements

This module estimates atmospheric turbulence strength (Rytov variance) by fitting
theoretical probability distributions to observed irradiance data from Free-Space
Optical (FSO) communications.

Author: Converted from MATLAB implementation
"""

import numpy as np
from scipy.optimize import fmin
from typing import Tuple

# Import dependencies (assumed to exist from previous tasks)
from power2irradiance import power2irradiance
from pdf_gamma_gamma_wrapped import pdf_gamma_gamma_wrapped
from pdf_log_normal import pdf_log_normal


def rytov_vs_latency(power_data: np.ndarray, aperture_diameter: float = 7e-3, 
                     latency_steps: int = 1) -> Tuple[float, float]:
    """
    Estimate Rytov variance from optical power measurements.
    
    This function converts optical power to irradiance, normalizes the data,
    generates an empirical probability distribution, and fits it against
    theoretical distributions (Gamma-Gamma for strong turbulence, Log-Normal
    for weak turbulence) to estimate the Rytov variance parameter.
    
    Args:
        power_data: Array of optical power measurements (Watts). Can be 1D or 2D.
                   For 2D arrays, each column is processed separately.
        aperture_diameter: Receiver aperture diameter (meters). Default is 7e-3
                          (7mm, corresponding to F810APC).
        latency_steps: Number of time steps for latency analysis. Currently
                      processes entire dataset as single block.
    
    Returns:
        Tuple of (rytov_variance, scintillation_index):
            - rytov_variance: Estimated Rytov variance parameter
            - scintillation_index: Normalized intensity variance (sigma_I^2 / mean_I^2)
    
    Algorithm:
        1. Convert optical power to irradiance
        2. Normalize by moving average to remove slow fading
        3. Generate empirical PDF from histogram
        4. Fit Gamma-Gamma distribution using MSE minimization
        5. If estimated Rytov < 0.03, switch to Log-Normal distribution
        6. Return optimized Rytov variance and scintillation index
    
    References:
        - Gamma-Gamma model: Andrews & Phillips, "Laser Beam Propagation
          through Random Media" (2005)
        - Log-Normal model: Valid for weak turbulence (Rytov variance < 0.3)
    """
    
    # Ensure power_data is at least 2D for consistent processing
    power_w = np.atleast_2d(power_data)
    if power_w.shape[0] == 1:
        power_w = power_w.T
    
    n_samples, n_columns = power_w.shape
    
    # Convert Received Optical Power to Irradiance
    I_meas = power2irradiance(power_w, aperture_diameter)
    
    # Normalize by moving average to remove slow fading (atmospheric trends)
    # Using 1000-sample moving average as in MATLAB implementation
    window_size = 1000
    I_fastFading = np.zeros_like(I_meas)
    
    for col in range(n_columns):
        # Calculate moving mean using convolution (equivalent to MATLAB's movmean)
        moving_avg = np.convolve(I_meas[:, col], np.ones(window_size)/window_size, mode='same')
        # Avoid division by zero
        moving_avg = np.where(moving_avg == 0, 1e-10, moving_avg)
        I_fastFading[:, col] = I_meas[:, col] / moving_avg
    
    # Histogram parameters (from MATLAB implementation)
    n_bins = int(1.5e5)  # 150,000 bins for high resolution
    y_threshold = 1e-6   # Minimum PDF value to consider
    
    # Process entire dataset as single block (as per MATLAB default behavior)
    n_samples_block = n_samples
    n_blocks = 1
    
    # Storage for results
    rytov_var = np.zeros((n_blocks, n_columns))
    mse_val = np.zeros((n_blocks, n_columns))
    scintillation_index = np.zeros(n_columns)
    
    # Mean Squared Error cost function
    def mse_cost(x, y):
        """Calculate MSE between two arrays, ignoring NaN values."""
        return np.nanmean(np.abs(x - y) ** 2)
    
    # Process each column (latency step)
    for col in range(n_columns):
        # Calculate scintillation index for this column
        I_mean = np.mean(I_fastFading[:, col])
        I_var = np.var(I_fastFading[:, col])
        scintillation_index[col] = I_var / (I_mean ** 2)
        
        for block in range(n_blocks):
            # Extract data for this block
            start_idx = block * n_samples_block
            end_idx = (block + 1) * n_samples_block
            I_fit = I_fastFading[start_idx:end_idx, col]
            
            # Generate histogram
            counts, edges = np.histogram(I_fit, bins=n_bins)
            xx = edges[:-1]  # Bin left edges
            yy = counts / np.sum(counts)  # Normalize to probability
            
            # Filter out bins with very small probabilities
            mask = yy > y_threshold
            xx1 = xx[mask]
            yy1 = yy[mask]
            
            # Normalize to unit integral area using trapezoidal integration
            integral_area = np.trapz(yy1, xx1)
            if integral_area > 0:
                y_normalized = yy1 / integral_area
                x_normalized = xx1
            else:
                # Fallback if integration fails
                y_normalized = yy1
                x_normalized = xx1
            
            # Initial guess for Rytov variance
            RV_init = 0.08
            
            # Define cost function for Gamma-Gamma fitting
            def cost_gamma_gamma(RV):
                try:
                    pdf_theoretical = pdf_gamma_gamma_wrapped(RV, x_normalized)
                    return mse_cost(pdf_theoretical, y_normalized)
                except Exception as e:
                    # Return large penalty if PDF calculation fails
                    return 1e10
            
            # Define cost function for Log-Normal fitting
            def cost_log_normal(RV):
                try:
                    pdf_theoretical = pdf_log_normal(RV, x_normalized)
                    return mse_cost(pdf_theoretical, y_normalized)
                except Exception as e:
                    return 1e10
            
            # Optimization options (matching MATLAB settings)
            # fmin parameters: disp=0 suppresses output
            try:
                # First attempt: Fit with Gamma-Gamma distribution
                rytov_var_gamma_gamma = fmin(
                    cost_gamma_gamma,
                    RV_init,
                    maxfun=int(1e5),
                    xtol=1e-10,
                    ftol=1e-10,
                    disp=False
                )
                
                if np.isscalar(rytov_var_gamma_gamma):
                    rytov_var_gamma_gamma_scalar = rytov_var_gamma_gamma
                else:
                    rytov_var_gamma_gamma_scalar = rytov_var_gamma_gamma[0]
                
                mse_val_gamma_gamma = cost_gamma_gamma(rytov_var_gamma_gamma_scalar)
                
                # Check if we should switch to Log-Normal (weak turbulence)
                if rytov_var_gamma_gamma_scalar < 0.03:
                    print(f"Switching to Log-Normal distribution (Rytov = {rytov_var_gamma_gamma_scalar:.4f})")
                    
                    # Refit with Log-Normal distribution
                    rytov_var_result = fmin(
                        cost_log_normal,
                        RV_init,
                        maxfun=int(1e5),
                        xtol=1e-10,
                        ftol=1e-10,
                        disp=False
                    )
                    
                    if np.isscalar(rytov_var_result):
                        rytov_var[block, col] = rytov_var_result
                    else:
                        rytov_var[block, col] = rytov_var_result[0]
                    
                    mse_val[block, col] = cost_log_normal(rytov_var[block, col])
                else:
                    # Use Gamma-Gamma result
                    rytov_var[block, col] = rytov_var_gamma_gamma_scalar
                    mse_val[block, col] = mse_val_gamma_gamma
                    
            except Exception as e:
                print(f"Optimization failed for column {col}, block {block}: {e}")
                rytov_var[block, col] = np.nan
                mse_val[block, col] = np.nan
    
    # Return results for first block and first column (most common use case)
    # If multiple columns exist, return average
    if n_columns == 1:
        return float(rytov_var[0, 0]), float(scintillation_index[0])
    else:
        # Return average across columns
        return float(np.nanmean(rytov_var)), float(np.nanmean(scintillation_index))
