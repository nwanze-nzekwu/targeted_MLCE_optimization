"""
Convert optical power to irradiance.

This module provides a function to convert optical power (in Watts) to 
irradiance (in W/m²) given an aperture diameter.
"""

import numpy as np


def power2irradiance(P, d):
    """
    Convert optical power to irradiance.
    
    Calculates the irradiance (intensity) from optical power and aperture 
    diameter using the formula:
        I = P / (π × (d/2)²)
    
    Parameters
    ----------
    P : float or array_like
        Optical power in Watts
    d : float or array_like
        Aperture diameter in meters
    
    Returns
    -------
    I : float or ndarray
        Irradiance in W/m²
    
    Raises
    ------
    ValueError
        If diameter d is zero or negative
    
    Examples
    --------
    >>> power2irradiance(1.0, 0.1)
    127.32395447351628
    
    >>> power2irradiance(np.array([1.0, 2.0]), 0.1)
    array([127.32395447, 254.64790895])
    """
    # Convert to numpy arrays for consistent handling
    P = np.asarray(P)
    d = np.asarray(d)
    
    # Check for zero or negative diameter
    if np.any(d <= 0):
        raise ValueError("Aperture diameter must be positive")
    
    # Calculate area of circular aperture
    area = np.pi * (d / 2.0) ** 2
    
    # Calculate irradiance
    I = P / area
    
    return I
