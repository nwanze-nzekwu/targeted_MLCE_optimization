import numpy as np


def get_number_eddies(rytov_var):
    """
    Derive turbulence parameters (alpha, beta) from Rytov variance.
    
    This function calculates the effective number of large-scale (alpha) and 
    small-scale (beta) turbulent eddies based on the Rytov variance using the 
    Gamma-Gamma atmospheric turbulence model. These parameters characterize the 
    statistics of optical wave propagation through atmospheric turbulence.
    
    The formulas are derived from atmospheric turbulence theory and are based on:
    - Andrews & Phillips, "Laser Beam Propagation through Random Media" (2005)
    - The Gamma-Gamma distribution model for moderate-to-strong turbulence
    
    Physical interpretation:
    - alpha: Effective number of large-scale turbulent eddies (related to 
             large-scale atmospheric variations)
    - beta: Effective number of small-scale turbulent eddies (related to 
            small-scale atmospheric scattering)
    - SI: Scintillation Index - normalized variance of irradiance fluctuations
    
    Parameters
    ----------
    rytov_var : float or array-like
        Rytov variance (sigma_R²), a dimensionless parameter characterizing 
        the strength of optical turbulence. Typical ranges:
        - < 0.3: Weak turbulence (log-normal regime)
        - 0.3-1.0: Moderate turbulence
        - > 1.0: Strong turbulence (fully developed)
    
    Returns
    -------
    alpha : float or ndarray
        Alpha parameter - effective number of large-scale turbulent eddies
    beta : float or ndarray
        Beta parameter - effective number of small-scale turbulent eddies
    si : float or ndarray
        Scintillation Index - normalized irradiance variance
    
    Notes
    -----
    - The function handles both scalar and array inputs
    - For very small Rytov variance (< ~0.03), consider using log-normal model
    - The formulas use empirical coefficients (0.49, 0.51, 1.11, 0.69) derived 
      from turbulence theory
    
    Examples
    --------
    >>> alpha, beta, si = get_number_eddies(0.5)
    >>> print(f"Alpha: {alpha:.4f}, Beta: {beta:.4f}, SI: {si:.4f}")
    
    References
    ----------
    Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through 
    random media (2nd ed.). SPIE Press.
    """
    # Convert input to numpy array for vectorization support
    rytov_var = np.asarray(rytov_var)
    
    # Input validation
    if np.any(rytov_var < 0):
        raise ValueError("Rytov variance must be non-negative")
    
    # Calculate square root of Rytov variance
    rrv = np.sqrt(rytov_var)
    
    # Calculate alpha parameter (large-scale turbulence)
    # Formula: alpha = 1 / (exp(term1) - 1)
    # where term1 = (0.49 * RRV^2) / (1 + 1.11 * RRV^(12/5))^(7/6)
    term1 = (0.49 * rrv**2) / ((1 + 1.11 * rrv**(12/5))**(7/6))
    
    # Handle case where exp(term1) - 1 might be very small
    exp_term1 = np.exp(term1)
    with np.errstate(divide='warn', invalid='warn'):
        alpha = 1.0 / (exp_term1 - 1)
    
    # Calculate beta parameter (small-scale turbulence)
    # Formula: beta = 1 / (exp(term2) - 1)
    # where term2 = (0.51 * RRV^2) / (1 + 0.69 * RRV^(12/5))^(5/6)
    term2 = (0.51 * rrv**2) / ((1 + 0.69 * rrv**(12/5))**(5/6))
    
    # Handle case where exp(term2) - 1 might be very small
    exp_term2 = np.exp(term2)
    with np.errstate(divide='warn', invalid='warn'):
        beta = 1.0 / (exp_term2 - 1)
    
    # Calculate Scintillation Index (SI)
    # Formula: SI = exp(term1 + term2) - 1
    si = np.exp(term1 + term2) - 1
    
    # Handle edge cases for very small Rytov variance
    # When Rytov variance approaches 0, alpha and beta should approach infinity
    # and SI should approach 0
    if np.isscalar(rytov_var):
        if rytov_var == 0:
            alpha = np.inf
            beta = np.inf
            si = 0.0
    else:
        zero_mask = (rytov_var == 0)
        if np.any(zero_mask):
            alpha = np.where(zero_mask, np.inf, alpha)
            beta = np.where(zero_mask, np.inf, beta)
            si = np.where(zero_mask, 0.0, si)
    
    return alpha, beta, si
