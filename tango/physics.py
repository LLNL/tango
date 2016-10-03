"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np


"""
physics

Module for dealing with other (non-turbulent) transport physics that contribute to the transport equation
"""




class transportPhysics(object):
    def __init__(self, profiles_all):
        self.profiles_all = profiles_all
    def neoclassical_chi(self):
        """to be filled in
        Uses a simple model for neoclassical ion thermal diffusivity.  Chi is computed from local
        parameters.
        
        Inputs:
          ()
        Outputs:
          chi       ion thermal diffusivity (array)
        """
        pass          
    

class Profiles_All(object):
    """Interface to the profiles.  Also contains some basic information
    relating to the profiles -- species mass, charge number, etc.
    
    For now, assume the species is singly charged...
    
    Internally, store the plasma density (same for ion & electron), the ion
    pressure, and the electron temperature.  Ion temperature can be derived
    
    Assume SI units...
    """
    def __init__(self, mu, n, Ti, Te, psi, a, R0, I, Bpi2, Vprime, gradpsisq):
        """Initialize the profiles object.
        
        Note, ion *temperature*, not *pressure*, is used to initialize.
        Note, the electron temperature is assumed fixed and unchanging
        
        Inputs:
          mu:       ion mass (in proton masses) (scalar)
          n:        plasma density (array)
          Ti:       ion temperature (array)
          Te:       electron temperature (array)
          psi:      psi grid (array)
          a:        minor radius (scalar)
          R0:       major radius (scalar)
          I:        magnetic flux function I(psi) = R B_phi (array)
          Bpi2:     magnetic field strength at theta=pi/2 on each flux surface (array)
          Vprime:   V' = dV/dpsi (array)
          gradpsisq: <|grad psi|^2> (array)
        """
        self.proton_mass = 1.6726219e-27 # in kg
        self.e = 1.60217662e-19 # electron charge
        self.mu = mu
        self.m = self.proton_mass * mu
        self.a = a
        self.R0 = R0
        self.aR0 = a/R0
        self.Z = 1
        
        self.n = n
        self.P = n*Ti
        self.Te = Te
        self.I = I
        self.Bpi2 = Bpi2
        self.psi = psi
        self.Vprime = Vprime
        self.gradpsisq = gradpsisq
        
    @property
    def Ti(self):
        return self.P / self.n
    def IonTemperatureIneV(self):
        return self.Ti/self.e
    def AsDict(self):
        """
        Return profile data as a dict
        """
        profiles_all = {'mu': self.mu, 'a': self.a, 'R0': self.R0,
                        'n': self.n, 'Te': self.Te,
                        'Vprime': self.Vprime, 'gradpsisq': self.gradpsisq}
        return profiles_all

def Mockup_trapezoidal_chi(psi1, psi2, chi_max, psi):
    """Create a diffusivity chi in the same of a trapezoid on the grid psi.
  
chi(psi) looks like:  
    
.... ------psi grid ------>|
           ________________      
          /.                
         / .               
chi=0   /  .    chi=chi_max     
______ /   .                    
      psi1 |
           |
          psi2
    
    Inputs:
      psi1          value of psi at which chi begins to grow linearly from 0 (scalar)
      psi2          value of psi at which chi reaches chi_max (scalar)
      chi_max       peak of chi (scalar)
      psi           grid (array) [it is assumed that psi increases monotonically]
    
    Outputs:
      chi           ion thermal diffusivity (array)
    """
    assert min(psi) <= psi1 <= psi2 <= max(psi)
    chi = np.zeros_like(psi)
    # region 2: the linearly increasing region
    if psi1 != psi2:
        ind_region2 = (psi > psi1) & (psi < psi2)
        slope = chi_max / (psi2 - psi1)
        chi[ind_region2] = slope * (psi[ind_region2] - psi1)
    
    # region 3: chi=chi_max
    ind_region3 = psi >= psi2
    chi[ind_region3] = chi_max
    pass  


def _dxCenteredDifference(u, dx):
    """Compute du/dx.
      du/dx is computed using centered differences on the same grid as u.  For the edge points, one-point differences are used.
    
    Inputs:
      u         profile (array)
      dx        grid spacing (scalar)
    
    Outputs:
      dudx      (array, same length as u)
    """
    dudx = np.zeros_like(u, dtype=float)
    dudx[0] = (u[1] - u[0]) / dx
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    dudx[-1] = (u[-1] - u[-2]) / dx
    return dudx


# need to change the input argument
def CollisionalEnergyExchange(Profiles):
    """Contribution to linear equation for new iteration from the collisional energy exchange term in the ion pressure equation.
          n * nuE_ie * (Te-Ti)  =  n*nuE_ie*Te - nuE_ie*pi
        currently not normalized (given in SI units...)
    """
    mu = Profiles.mu
    Z = Profiles.Z
    Te = Profiles.GetElectronTemperature()
    
    n = Profiles.GetIonDensity()       # in SI units.  assume ni = ne
    ne = n    
    
    logLambda = 10
    e = 1.60217662e-19 # electron charge
    Te_ineV = Te/e      # convert from J to eV
    
    # Reference: NRL formulary.  Gives result in s^-1
    nuE_ie = 3.2e-15 * ne * Z**2 * logLambda / (mu * Te_ineV)
        
    # Contributions to finite difference arrays
    B_contrib = nuE_ie
    H_contrib = -n * nuE_ie * Te
    
    A_contrib = np.zeros_like(B_contrib)
    C_contrib = np.zeros_like(B_contrib)
    
    # right boundary: Dirichlet
    B_contrib[-1] = 0
    H_contrib[-1] = 0    
    
    return (A_contrib, B_contrib, C_contrib, H_contrib)
    
    
def nu_Braginskii(n, T, mu, Z):
    """Calculate the Braginskii ion-ion collision frequency.  This is used in
    the ion neoclassical heat flux
    
     Inputs:
        n: ion density
        T: ion temperature
        mu: ion mass (measured in proton masses)
        Z: ion charge number
        
     For now, assume log Lambda = 10
     
     For now, everything is assumed to be in SI units.
     
      Outputs:
       nuB_ii: the computed collision frequency
     
     Vectorized: If n & T are arrays corresponding to different positions, then
      the output is also an array
    """
    
    # constants
    logLambda = 10
    e = 1.60217662e-19 # electron charge
    mp = 1.6726219e-27 # protos mass
    m = mu*mp
    
    nuB_ii = 4 * np.pi * Z**4 * e**4 * n * logLambda / (3 * np.sqrt(m) * T**(3/2))
    return nuB_ii
    
    
def MagneticGeometryCircular(psi):
    """Set up the model magnetic geometry used in various GENE studies.  This
     involves concentric circular flux surfaces.
     
     Inputs:
         psi    1d array of containing grid points psi = r/a
     Outputs: [stored in dict geom]
         I          magnetic I(psi) = R B_phi
         Bpi2       magnetic field strength at theta=pi/2 on each flux surface
         a          minor radius
         R0         major radius on axis
         Vprime     dV/dpsi, where V = volume within flux surface psi
         gradpsisq  |grad psi|^2
         
    
    Reference: X. Lapillonne et al. (2009) - Clarification to limitations of
               the s-alpha equilibrium model
               
    The Lapillonne paper uses psi = r/a = r_n
    B = R0 B0 / R * (phi_hat + r/(R0*qbar) theta_hat)
    
    safety factor q related to qbar:
        q = qbar / sqrt(1- r^2 / R0^2)
        McMillan / Lapillonne uses q(rn) = (0.854 + 2.184 rn^2) / sqrt(1 - (rn a / R0)^2)
          i.e., qbar = 0.854 + 2.184 rn^2
    """
    
    # set some initial parameters from which other quantities are derived
    R0 = 1
    B0 = 1
    a = 0.5
    
    I = R0*B0 * np.ones_like(psi) # I= R B_phi is independent ofpsi
    
    qbar = 0.854 + 2.184 * psi**2
    Bpi2 = B0 * np.sqrt(1 + (psi*a/(R0*qbar))**2)
    
    Vprime = (2 * np.pi)**2 * R0 * a * psi
    gradpsisq = 1 
    
    # set up a dictionary to store outputs
    geom = {}
    geom['R0'] = R0
    geom['B0'] = B0
    geom['a'] = a
    geom['I'] = I
    geom['Bpi2'] = Bpi2
    geom['Vprime'] = Vprime
    geom['gradpsisq'] = gradpsisq

    return geom