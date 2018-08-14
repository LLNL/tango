"""
File with some helper functions for tango_kinetic_gene_run

Specifying initial profiles and source
"""

from __future__ import division, absolute_import
import numpy as np

from tango.utilities.gene import write_profiles

def read_seed_turb_flux(filenameFluxSeed):
    """Read in a file that contains a turbulent flux profile (heat flux for either species, or particle flux) to use as the EWMA seed.
    
    The seed should probably come from a long run of GENE (long compared to a single iteration here) so that the flux is averaged over
    many cycles.  The flux here is <Q dot grad psi> or <Gamma dot grad psi>, not V' * <Q dot grad psi>, because that is what Tango uses.
    
    Inputs:
      filenameFluxSeed      path of file containing a flux seed (string)
    Outputs:
      fluxProfile           turbulent flux profile as a function of radius (1D array)
    """
    fluxProfile = np.loadtxt(filenameFluxSeed, unpack=True)
    return fluxProfile

# ============================================================================================================== #
# INITIAL CONDITIONS
def density_initial_condition(rho):
    """Initial ion density profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      n         density profile in SI (array)
    """
    minorRadius = 0.594  # a
    majorRadius = 1.65  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    # Density profile
    kappa_n = 2.4
    delta_n = 0.5
    Delta_n = 0.1
    rho0_n = 0.5
    n0 = 3e19
    n = write_profiles.base_profile_shape(rho, kappa_n, delta_n, Delta_n, rho0_n, n0, inverseAspectRatio)
    return n
    
def ion_temperature_initial_condition(rho):
    """Initial ion temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      Ti        ion temperature profile in SI (array)
    """
    minorRadius = 0.594  # a
    majorRadius = 1.65  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    e = 1.60217662e-19          # electron charge
    # Ion temperature profile
    kappa_i = 6.9
    delta_i = 0.9
    Delta_i = 0.1
    rho0_i = 0.5
    Ti0 = 1
    Ti = write_profiles.base_profile_shape(rho, kappa_i, delta_i, Delta_i, rho0_i, Ti0, inverseAspectRatio)
    Ti *= 1000 * e      # convert from keV to SI
    return Ti
    
def electron_temperature_initial_condition(rho):
    """Initial electron temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      T         temperature profile in SI (array)
    """
    minorRadius = 0.594  # a
    majorRadius = 1.65  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    e = 1.60217662e-19          # electron charge
    # Electron temperature profile
    kappa_e = 7.3
    delta_e = 0.9
    Delta_e = 0.1
    rho0_e = 0.5
    Te0 = 1
    Te = write_profiles.base_profile_shape(rho, kappa_e, delta_e, Delta_e, rho0_e, Te0, inverseAspectRatio)
    Te *= 1000 * e      # convert from keV to SI
    return Te

# ============================================================================================================== #
# SOURCES
def fn_hat(rho):
    """Return the density source V' * S_n."""
    An = 4 * 2.8628e22  # amplitude... with the factor of 4, 1.2e22 particles/second of input fueling
    fn = np.zeros_like(rho)
    rho_a = 0.2
    rho_b = 0.6
    rho_0 = 0.4
    w = 0.1
    ind = (rho > rho_a) & (rho < rho_b)
    fn[ind] = An * np.exp( -(rho[ind] - rho_0)**2 / w**2 )
    return fn
        
def fi_hat(rho):
    """Return the ion heat source V' S_i."""
    Ai = 6 * 2.0409e7 # amplitude... with the factor of 6, 18 MW of input power
    fi = np.zeros_like(rho)
    rho_a = 0.2
    rho_b = 0.8
    rho_0 = 0.4
    
    ind1 = (rho > rho_a) & (rho < rho_0)
    w1 = 0.1
    fi[ind1] = Ai * np.exp( -(rho[ind1] - rho_0)**2 / w1**2)
    
    ind2 = (rho >= rho_0) & (rho < rho_b)
    w2 = 0.18
    fi[ind2] = Ai * np.exp( -(rho[ind2] - rho_0)**2 / w2**2)
    return fi
    
def fe_hat(rho):
    """Return the electron heat source V' S_e."""
    Ae = 6 * 1.9004e7 # amplitude... with the factor of 6, 12 MW of input power
    fe = np.zeros_like(rho)
    rho_a = 0.3
    rho_b = 0.8
    rho_0 = 0.55
    
    ind = (rho > rho_a) & (rho < rho_b)
    w = 0.1
    fe[ind] = Ae * np.exp( -(rho[ind] - rho_0)**2 / w**2)
    return fe
    
# ============================================================================================================== #
# Other physics
def convert_Te_to_eV(n, pe):
    """Return the electron temperature in eV.
    Inputs:
      n     density in m^-3 (array)
      pe    electron pressure in SI (array)
    Outputs:
      Te    electron temperature in eV (array)
    """
    e = 1.60217662e-19
    Te_SI = pe / n
    Te = Te_SI / e
    return Te

def nu_E(n, Te):
    """Return the collisional energy exchange frequency [assuming ni=ne].
    Inputs:
      n     density in m^-3 (array)
      Te    electron temperature in eV (array)
    Outputs:
      nu    collisional energy exchange frequency in s^-1 (array)
    """
    # Z = 1
    # loglambda = 10
    # mu = 2
    # nu = 3.2e-15 * Z**2 * loglambda / (mu * Te**(3/2)) * n
    ### For now, neglect collisional energy exchange and set nu=0
    nu = np.zeros_like(n)
    return nu    