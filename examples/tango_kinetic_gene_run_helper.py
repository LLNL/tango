"""
File with some helper functions for tango_kinetic_gene_run

Specifying initial profiles and source
"""

from __future__ import division, absolute_import
import numpy as np

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
    rho0 = 0.5
    
    # density profile
    n0 = 3.3e19;     # in SI, m^-3
    kappa_n = 2.22;  # R0 / Ln    
    deltar = 0.5
    rhominus = rho - rho0 + deltar/2
    deltan = 0.1
    n = n0 * np.exp( -kappa_n * inverseAspectRatio * (rho - rho0 - deltan * (np.tanh(rhominus/deltan) - np.tanh(deltar/2/deltan))))
    
    # set n to a constant for rho < rho0-deltar/2
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2))
    n[ind2] = n[ind];
    return n
    
def ion_temperature_initial_condition(rho):
    """Initial ion temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      T         temperature profile in SI (array)
    """
    e = 1.60217662e-19          # electron charge
    
    kappa_T = 6.96
    deltar = 0.9
    rho0 = 0.5
    rhominus = rho - rho0 + deltar/2
    deltaT = 0.1

    e = 1.60217662e-19
    T0 = 1000*e
    invasp = 0.36
    T = T0 * np.exp( -kappa_T * invasp * (rho - rho0 - deltaT * (np.tanh(rhominus/deltaT) - np.tanh(deltar/2/deltaT))));
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2));
    T[ind2] = T[ind];
    return T
    
def electron_temperature_initial_condition(rho):
    """Initial electron temperature profile
    Inputs:
      rho       normalized radial coordinate rho=r/a (array)
    Outputs:
      T         temperature profile in SI (array)
    """
    # return T
    pass


def density_source():
    pass

def ion_heat_source():
    pass

def electron_heat_source():
    pass

def collisional_energy_exchange():
    pass
# define sources for density, ion pressure, and electron pressure
# define collisional energy exchange term    


## aaa: p
### test lol
#### hi wow
#### Below: source for Tango 1.0

def source_fhat(rho):
    """Provide the function fhat in the source term V'S = A fhat(rho)
    
    Inputs:
      rho           radial coordinate rho=r/a (array)
    Outputs:
      fhat          output (array)
    """
    rho_a = 0.2
    rho_b = 0.4
    rho0 = 0.3
    w = 0.05
    
    fhat = np.zeros_like(rho)
    ind = (rho > rho_a) & (rho < rho_b)
    fhat[ind] = np.exp( -(rho[ind] - rho0)**2 / w**2)
    return fhat

def source_H7(r, minorRadius, majorRadius, A):
    """Provide the source term V'S to the transport equation.  The contribution to H7 is V'S.
    
    The source is written as V'S = A * f(r) = A * fhat(r/a)
    
    Inputs:
      r             radial coordinate r, in m (array)
      minorRadius   minor radius a, in m (scalar)
      majorRadius   major radius R0, in m (scalar)
      A             amplitude in SI of V'S, in SI (scalar)
    Outputs:
      H7contrib     contribution to H7 (=V'S) (array)
    """
    rho = r / minorRadius
    fhat = source_fhat(rho)
    H7contrib = A * fhat
    return H7contrib
        