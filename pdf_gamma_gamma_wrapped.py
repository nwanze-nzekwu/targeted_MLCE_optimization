import numpy as np
from get_number_eddies import get_number_eddies
from pdf_gamma_gamma import pdf_gamma_gamma


def pdf_gamma_gamma_wrapped(RytovVar: float, I: np.ndarray) -> np.ndarray:
    """
    Calculate Gamma-Gamma PDF using Rytov variance.
    
    This function serves as a convenience wrapper that combines turbulence 
    parameter estimation with PDF calculation. It first estimates the alpha 
    and beta parameters from the Rytov variance, then uses them to compute 
    the Gamma-Gamma probability density function.
    
    Args:
        RytovVar: Rytov variance (dimensionless)
        I: Irradiance values (numpy array)
    
    Returns:
        PDF values corresponding to input irradiance
    """
    alpha, beta, _ = get_number_eddies(RytovVar)
    p = pdf_gamma_gamma(alpha, beta, I)
    return p
