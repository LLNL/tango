"""
**!!!!!!!!!!!!!!!!!!!! NOT READY YET.  NEEDS TO BE EDITED.  !!!!!!!!!!!!!!!!!!**
File with some helper functions for tango_chease_kinetic_run

Specifying initial profiles and source
"""

import numpy as np
import scipy.interpolate

from tango.utilities.gene import read_profiles

def interpolate_1d_qty(x, y, xNew):
    """Interpolate a 1D quantity from given radial grid to new radial grid.
             
    Inputs:
        x           x grid as given (1D array)
        y           quantity evaluated on given grid x (1D array)
        xNew        new x grid on which to interpolate the quantity (1D array)
        
    
    Outputs:
        yNew        quantity interpolated onto xNew grid (1D array)
    """
    interpolator = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    yNew = interpolator(xNew)
    return yNew

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
# INITIAL CONDITIONS.  Read in from a file, then interpolate onto
def initial_conditions(rho):
    """Set up initial conditions for tango run.
    
    Here, read the initial conditions from a file, interpolate onto a given rho grid for Tango,
    then convert to SI units.
    """
    # read density, Ti, Te from file in what units
    # interpolate onto rho grid
    # load profiles
    profileIons = read_profiles.read_profiles('IC_profiles_ions')
    profileElectrons = read_profiles.read_profiles('IC_profiles_electrons')
    
    # interpolate and convert to SI units
    n = interpolate_1d_qty(profileIons.rho, profileIons.n, rho) * 1e19
    Ti_keV = interpolate_1d_qty(profileIons.rho, profileIons.T, rho)
    Te_keV = interpolate_1d_qty(profileElectrons.rho, profileElectrons.T, rho)
    
    e = 1.60217662e-19
    pi = Ti_keV * 1e3 * e * n
    pe = Te_keV * 1e3 * e * n
    return (n, pi, pe)    
    
# ============================================================================================================== #
# SOURCES
    
def Sn_func(rho):
    """Particle source Sn.  To be later multiplied by V'
    
    note: total input # particles/second is a * integral(V' Sn, [rho, 0, 0.85]) = a * np.trapz(Vprime*Sn, rho)
    """
    pnfit = np.array([8.03e18, 1.44e19, 2.80e18])
    Sn = np.polyval(pnfit, rho)
    Sn *= 4 # manual adjustment
    return Sn

def Si_func(rho):
    """Ion heat source Si.  To be later multiplied by V'
    """
    pifit = np.array([-9.9e5, 1.18e6, -2.36e5, 5.51e4])
    Si = np.polyval(pifit, rho)
    Si *= 15 # manual adjustment
    return Si


def Se_func(rho):
    """Electron heat source Se.  To be later multiplied by V'
    """
    pefit = np.array([-3.54e6, 5.81e6, -2.71e6, 4.56e5, 8.14e2])
    Se = np.polyval(pefit, rho)
    Se *= 15 # manual adjustment
    return Se

    
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

def calc_nuE(n, Te):
    """Calculate the collisional energy exchange frequency.
    
    Valid for array input n, Te.
    
    Inputs:
        n     density in m^-3 (array)
        Te    electron temperature in SI units (array)
    Outputs:
        nuE_ie    collisional energy exchange frequency in s^-1 (array)
    
    """
    e = 1.60217662e-19   # electron charge
    # from nrl formulary
    ionMass = 2 # measured in proton masses
    logLambda = 10
    ionCharge = 1
    electronDensity = n
    Te_eV = Te / e
    nuE_ie = 3.2e-15 * electronDensity * ionCharge**2 * logLambda / (ionMass * Te_eV**(3/2))
    return nuE_ie

# ============================================================================================================== #
# Custom ftheta for LoDestro Method.  Can be different for density, ion pressure, electron pressure
def n_custom_ftheta(DHat, dndx, thetaParams):
    """custom ftheta for density and particle flux."""
    #Dmin = 1e-2  # scalar
    #Dmax = 50  # scalar
    #dndxThreshold = 1e18
    
    # theta = 1/2 for regions of particle positive flux
    theta = 0.5 * np.ones_like(DHat)
    
    # for regions of negative particle flux (where Dhat < 0), give those regions theta = -1
    ind = DHat < 0
    theta[ind] = -1

    # Anti-diffusion-lock [antilock brakes]: for regions where D = theta*Dhat would be too large, reduce theta to cap D at Dmax
    Dmax = 15
    ind = (theta * DHat) > Dmax
    theta[ind] = Dmax / DHat[ind]

    return theta
    
def const_ftheta(DHat, dndx, thetaParams):
    """custom ftheta for heat flux (both ions and electrons)."""
    # use a constant value of theta everywhere in the domain.
    theta0 = 0.95
    theta = theta0 * np.ones_like(DHat)
    
    # for regions of negative heat flux (where Dhat < 0), give those regions theta = -0.3
    ind = DHat < 0
    theta[ind] = -0.3

    # Anti-diffusion-lock: for regions where D = theta*Dhat would be too large, reduce theta to cap D at Dmax
    Dmax = 30
    ind = (theta * DHat) > Dmax
    theta[ind] = Dmax / DHat[ind]

    return theta
    