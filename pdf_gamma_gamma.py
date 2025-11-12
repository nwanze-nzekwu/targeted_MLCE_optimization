"""
Gamma-Gamma probability density function for atmospheric turbulence.

This module provides a function to calculate the gamma-gamma PDF used in 
modeling moderate to strong turbulence effects in free-space optical 
communications.

Authors: Marco Fernandes <marcofernandes@av.it.pt>
Python conversion: 2024
"""

import numpy as np
from scipy.special import gamma, kv


def pdf_gamma_gamma(alpha, beta, I):
    """
    Calculate the Gamma-Gamma probability density function.
    
    This function computes the gamma-gamma PDF for modeling irradiance 
    fluctuations in moderate to strong atmospheric turbulence conditions.
    The model was proposed by M. A. Al-Habash et al. in "Mathematical model 
    for the irradiance probability density function of a laser beam propagating
    through turbulent media".
    
    The gamma-gamma model assumes that the received irradiance is the product 
    of two independent gamma-distributed random variables representing large-scale
    and small-scale turbulent eddies.
    
    Formula:
        p(I) = (2(αβ)^((α+β)/2) / (Γ(α)Γ(β))) * I^((α+β)/2 - 1) * K_(α-β)(2√(αβI))
        
    where K_ν is the modified Bessel function of the second kind of order ν.
    
    Parameters
    ----------
    alpha : float or array_like
        Effective number of large-scale turbulent eddies. Must be positive.
    beta : float or array_like
        Effective number of small-scale turbulent eddies. Must be positive.
    I : float or array_like
        Normalized received irradiance. Must be positive.
    
    Returns
    -------
    p : float or ndarray
        Probability density at the given irradiance value(s)
    
    Raises
    ------
    ValueError
        If alpha, beta, or I contains zero or negative values
    
    Examples
    --------
    >>> pdf_gamma_gamma(2.0, 3.0, 1.0)
    0.8788007830714049
    
    >>> pdf_gamma_gamma(2.0, 3.0, np.array([0.5, 1.0, 1.5]))
    array([0.73575888, 0.87880078, 0.68040039])
    
    References
    ----------
    Al-Habash, M. A., Andrews, L. C., & Phillips, R. L. (2001). Mathematical 
    model for the irradiance probability density function of a laser beam 
    propagating through turbulent media. Optical Engineering, 40(8), 1554-1562.
    """
    # Convert to numpy arrays for consistent handling
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    I = np.asarray(I)
    
    # Check for invalid inputs
    if np.any(alpha <= 0):
        raise ValueError("Alpha must be positive")
    if np.any(beta <= 0):
        raise ValueError("Beta must be positive")
    if np.any(I <= 0):
        raise ValueError("Irradiance must be positive")
    
    # Calculate intermediate variables
    k = (alpha + beta) / 2.0
    k1 = alpha * beta
    
    # Calculate normalization constant
    K = 2.0 * (k1 ** k) / (gamma(alpha) * gamma(beta))
    
    # Calculate I-dependent term
    K1 = I ** (k - 1.0)
    
    # Calculate argument for Bessel function
    Z = 2.0 * np.sqrt(k1 * I)
    
    # Calculate PDF using modified Bessel function of the second kind
    # kv(nu, z) is the modified Bessel function K_nu(z)
    p = K * K1 * kv(alpha - beta, Z)
    
    return p
