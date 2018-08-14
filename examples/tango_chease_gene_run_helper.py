"""
File with some helper functions for tango_chease_gene_run

Specifying density profile, initial temperature profile, and source
"""

import numpy as np

from tango.utilities.gene import write_profiles

P_in = 3e6 # total input power in Watts

def read_seed_turb_flux(filenameFluxSeed):
    """Read in a file that contains a turbulent flux profile to use as the EWMA seed.
    
    The seed should probably come from a long run of GENE (long compared to a single iteration here) so that the flux is averaged over
    many cycles.  The flux here is <Q dot grad psi> or <Gamma dot grad psi>, not V' * <Q dot grad psi>, because <Q dot grad psi> is what Tango uses.
    
    Inputs:
      filenameFluxSeed      path of file containing a flux seed (string)
    Outputs:
      fluxProfile           turbulent flux profile as a function of radius (1D array)
    """
    fluxProfile = np.loadtxt(filenameFluxSeed, unpack=True)
    return fluxProfile


def density_profile(rho):
    """density profile, fixed in time.  Use scenario 3.
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      n         density profile in SI (array)
    """
    n = 1e19 * write_profiles.scenario3_densityprofile(rho) # convert to SI by multiplying by 1e19
    return n

def temperature_initial_condition(rho):
    """Initial temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      T         temperature profile in SI (array)
    """
    e = 1.60217662e-19          # electron charge
    T = write_profiles.scenario3_temperatureprofile(rho)  # get T in keV
    T *= 1000 * e  # convert from keV to SI
    return T

def fi_hat(rho):
    """Return the ion heat source V' S_i."""
    # Compute
    #A = 6 * 2.0409e7 # amplitude... with the factor of 6, 18 MW of input power
    fi = np.zeros_like(rho)
    rho_a = 0.15
    rho_b = 0.55
    rho_0 = 0.35
    w = 0.1
    ind = (rho > rho_a) & (rho < rho_b)
    fi[ind] = A * np.exp( -(rho[ind] - rho_0)**2 / w**2 )
    return fi

# Calculate A corresponding to P_in specified at the top of the module, to be used in fi_hat
A = 1
rho = np.linspace(0, 1, 5000)
f = fi_hat(rho)
minorRadius = 0.741206  # a, in m
A = P_in / (np.trapz(f, rho) * minorRadius)