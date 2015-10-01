h = 6.62606896e-27  # erg * s  (planck constant)
c = 29979245800.0   # cm s-1   (speed of light)
k = 1.3806504e-16 # erg K-1  (Boltzmann constant)
from numpy import exp

def blackbody( wave, T, waveunit='Angstrom' ):
    """ returns the Planck function for radiation from a blackbody 
    at temperature T (K) at wavelength(s) wave, given in Angstrom
    Returns radiance in cgs units.
    """
   
    if waveunit=='Angstrom':
        # convert wavelength from angstroms to cm
        wave = wave / 1e10 * 100.
    elif waveunit=='nm':
        # convert wavelength from angstroms to cm
        wave = wave / 1e9 * 100.

    return( ((2 * h * c* c)/wave**5 ) / (exp(h*c/(wave*k*T))-1) )
