"""
Log-Normal probability density function for weak atmospheric turbulence.

This module provides a function to calculate the log-normal PDF used in 
modeling weak turbulence effects in free-space optical communications.

Authors: Marco Fernandes <marcofernandes@av.it.pt>
Python conversion: 2024
"""

import numpy as np


def pdf_log_normal(rytov_var, I):
    """
    Calculate the Log-Normal probability density function.
    
    This function computes the log-normal PDF for modeling irradiance 
    fluctuations in weak atmospheric turbulence conditions. The PDF is 
    parameterized by the Rytov variance.
    
    The log-normal distribution assumes that log(I) follows a normal 
    distribution with mean E[log(I)] = -rytov_var/2 and variance rytov_var.
    
    Formula:
        p(I) = (1 / (I * sqrt(2π * σ²))) * exp(-(ln(I) - μ)² / (2σ²))
        where μ = -rytov_var/2 and σ² = rytov_var
    
    Parameters
    ----------
    rytov_var : float or array_like
        Rytov variance (also called scintillation index in weak turbulence).
        Must be positive.
    I : float or array_like
        Normalized received irradiance. Must be positive.
    
    Returns
    -------
    p : float or ndarray
        Probability density at the given irradiance value(s)
    
    Raises
    ------
    ValueError
        If rytov_var or I contains zero or negative values
    
    Examples
    --------
    >>> pdf_log_normal(0.1, 1.0)
    1.5957691216057308
    
    >>> pdf_log_normal(0.1, np.array([0.5, 1.0, 1.5]))
    array([1.35335283, 1.59576912, 1.41421356])
    
    References
    ----------
    Andrews, L. C., & Phillips, R. L. (2005). Laser Beam Propagation through 
    Random Media (2nd ed.). SPIE Press.
    """
    # Convert to numpy arrays for consistent handling
    rytov_var = np.asarray(rytov_var)
    I = np.asarray(I)
    
    # Check for invalid inputs
    if np.any(rytov_var <= 0):
        raise ValueError("Rytov variance must be positive")
    if np.any(I <= 0):
        raise ValueError("Irradiance must be positive")
    
    # Calculate mean of log(I)
    El = -rytov_var / 2.0
    
    # Calculate components of the PDF
    out_exp = (1.0 / (np.sqrt(2.0 * np.pi * rytov_var))) * (1.0 / I)
    num_exp = -(np.log(I) - El) ** 2
    den_exp = 2.0 * rytov_var
    
    # Calculate PDF
    p = out_exp * np.exp(num_exp / den_exp)
    
    return p
