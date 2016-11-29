"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
from . import parameters


"""
physics

Module for dealing with other (non-turbulent) transport physics that contribute to the transport equation
"""

# physical constants with module-wide scope.  Given in SI units
e = 1.60217662e-19          # electron charge
mp = 1.6726219e-27          # proton mass
eps0 = 8.85418782e-12       # permittivity of free space

class TransportPhysics(object):
    def __init__(self, profilesAll):
        self.profilesAll = profilesAll
    def neoclassical_chi(self, P):
        """to be filled in
        Uses a simple model for neoclassical ion thermal diffusivity.  Chi is computed from local
        parameters.
        
        Inputs:
          P         new ion pressure profile (array) [SIDE EFFECT: stores P internally as the new pressure]
        Outputs:
          chi       ion thermal diffusivity due to neoclassical effects on radial grid (array)
        """
        #n, T, mu, Z = (1,1,1,1)  # fill in based on internal representation of profiles
        #B_pi2, qbar, aR0, psi = (1,1 1, 1)      # fill in
        self.update_pressure(P)
        
        n = self.profilesAll.n
        T = self.profilesAll.Ti
        mu = self.profilesAll.mu
        Z = self.profilesAll.Z
        B_pi2 = self.profilesAll.B_pi2
        qbar = self.profilesAll.qbar
        aR0 = self.profilesAll.aR0
        psi = self.profilesAll.psi
        gradpsisq = self.profilesAll.gradpsisq
        
        # compute intermediate parameters
        epsilon = psi * aR0     # epsilon = r/R0 = r/a * a/R0 (inverse aspect ratio)
        m = mu * mp
        Omega_pi2 = Z * e * B_pi2 / m
        nuB = nu_Braginskii(n, T, mu, Z)
        
        # compute sigma
        sigma = neoclassical_sigma(mu, T, nuB, Omega_pi2, qbar, epsilon)
        
        # compute diffusivity chi from sigma
        chi = sigma_to_chi(sigma, gradpsisq)
        return chi
        
    def update_pressure(self, PNew):
        """Update the internal profiles with a new pressure
        """
        self.profilesAll.P = PNew
        
    
def neoclassical_sigma(mu, T, nuB, Omega_pi2, qbar, epsilon):
    """Compute the neoclassical heat transport coefficient sigma.
    
    Here, sigma is calculated for the *banana regime* only, in a formula applicable to concentric circular surfaces
    where psi=r.  The present formula accounts for ion-ion collisions of a single species.  This formula is not
    invariant to a coordinate transform to arbitrary flux coordinate.
    
    First, we define sigma_p through
            
            (1)     <q dot grad psi_p> = -sigma_p * n * dT/dpsi_p
            
    where psi_p is the poloidal flux divided by 2*pi.  sigma_p is given by [See Parker 2012]
            
            (2)     sigma_p = I^2 T nu_B g(eps) / (m_i Omega_pi2^2)
            
    where g(epsilon) is a dimensionless geometric coefficient in terms of epsilon=r/R0.  However, Equation (1) is not 
    invariant to coordinate transform for psi_p to an arbitrary flux coordinate psi.  One can quickly see that
    for a different coordinate psi, one obtains
            
            (3)     <q dot grad psi> = -sigma_p * n * (dpsi / dpsi_p)^2 * dT/dpsi
    
    Finally, we define sigma in a particular flux coordinate system through
    
            (4)     <q dot grad psi> = -sigma * n * dT/dpsi
            
    Hence, we have
    
            (5)     sigma = sigma_p * (dpsi / dpsi_p)^2
    
    For the special case of concentric circular magnetic geometry, using the flux coordinate psi = r, and using the
    geometry in the form written in Lapillone (2009), one finds dpsi / dpsi_p = qbar / (r*B0).  Then one can cast
    Equation (5) into
    
            (6)     sigma = -1/epsilon^2 * g(eps) * T * nuB * qbar^2 / (m_i * Omega_pi2^2)
          
    Note, the formula for sigma blows up near r=0 or epsilon=0 due to the factor of 1/epsilon^2.  This is fine; the
    total heat flux <q dot grad psi> goes to zero at r=0, as it must, because dT/dr -> 0.
          
    Inputs: [IN SI UNITS]
      mu            ion mass measured in proton masses (scalar)
      T             ion temperature (measured in energy) on radial grid (array)
      nuB           Braginskii ion-ion collision frequency on radial grid (array)
      Omega_pi2     ion cyclotron frequency at theta=pi/2 on radial grid (array)
      qbar          coefficient related to safety factor in specification of magnetic geometry on radial grid (array)
      epsilon       radial position r/R0 on radial grid (array)
    
    Outputs: [IN SI UNITS]
      sigma         neoclassical heat transport coefficient in radial coordinates
    
    See Parker and Catto, Plasma Phys. Control. Fusion, Variational calculation of neoclassical ion heat flux and
        poloidal flow in the banana regime for axisymmetric magnetic geometry. (2014)
    """
    m = mu * mp
    
    g = 1.34*epsilon**(1/2) + 2.60*epsilon - 2.13*epsilon**(3/2) + 3.18*epsilon**2  # Ref: Parker and Catto (2012) - Variational calculation of neoclassical ion heat flux...
    # g = 2*epsilon**(1/2) * (0.66 + 1.88*epsilon**(1/2) - 1.54*epsilon) * (1 + 3/2*epsilon**2) # Ref: Chang-Hinton formula simplified to banana regime
    #  the Parker & Chang-Hinton formulae are nearly identical for 0 <= epsilon <= 1
    sigma = (1/epsilon**2) * g * T * nuB * qbar**2 / (m * Omega_pi2**2)
    return sigma

def sigma_to_chi(sigma, gradpsisq):
    """convert neoclassical sigma to chi", where chi is defined as
    
            <q dot grad psi> = -n * chi * dT/dpsi * <|grad psi|^2>
    Hence, the relationship is 
    
            chi = sigma / <|grad psi|^2>
    
    Inputs:
      sigma         neoclassical heat transport coefficient (array)
      gradpsisq     <|grad psi|^2> (scalar or array)
    Outputs:
      chi           thermal diffusivity coefficient (array)
    """
    chi = sigma / gradpsisq
    return chi


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
    
     Inputs: [IN SI UNITS]
        n: ion density
        T: ion temperature
        mu: ion mass (measured in proton masses)
        Z: ion charge number
        
     For now, assume log Lambda = 10.  For now, everything is assumed to be in SI units.
     
      Outputs: [IN SI UNITS]
       nuB_ii: the computed collision frequency
     
     Vectorized: If n & T are arrays corresponding to different positions, then
      the output is also an array
    """
    
    # constants
    logLambda = 10
    m = mu*mp
    nuB_ii = 4 * np.pi * Z**4 * e**4 / (4*np.pi*eps0)**2 * n * logLambda / (3 * np.sqrt(m) * T**(3/2))
    return nuB_ii


class ProfilesAll(object):
    """Interface to the profiles.  Also contains some basic information
    relating to the profiles -- species mass, charge number, etc.
    
    For now, assume the species is singly charged...
    
    Internally, store the plasma density (same for ion & electron), the ion
    pressure, and the electron temperature.  Ion temperature can be derived
    
    Assume SI units...
    """
    def __init__(self, mu, n, psi, a, R0, B_pi2, qbar, Vprime, gradpsisq):
        """Initialize the profiles object.
        
        Note, ion *temperature*, not *pressure*, is used to initialize.
        Note, the electron temperature is assumed to be equal to ion temperature.
        
        Inputs:
          mu:       ion mass (in proton masses) (scalar)
          n:        plasma density (array)
          psi:      psi grid (array)
          a:        minor radius (scalar)
          R0:       major radius (scalar)
          B_pi2:    magnetic field strength at theta=pi/2 on each flux surface (array)
          qbar:     coefficient related to safety factor in specification of magnetic geometry (array)
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
        self.P = None # gets updated
        self.B_pi2 = B_pi2
        self.qbar = qbar
        self.psi = psi
        self.Vprime = Vprime
        self.gradpsisq = gradpsisq
        
    @property
    def Ti(self):
        return self.P / self.n
    def IonTemperatureIneV(self):
        return self.Ti/self.e
#    def AsDict(self):
#        """
#        Return profile data as a dict
#        """
#        profilesAll = {'mu': self.mu, 'a': self.a, 'R0': self.R0,
#                        'n': self.n, 'Te': self.Te,
#                        'Vprime': self.Vprime, 'gradpsisq': self.gradpsisq}
#        return profilesAll
        

def initialize_profile_defaults(ionMass, density, r, minorRadius, majorRadius, B0, Vprime, gradPsiSq):
    """Create and return an instance of ProfilesAll.
    
    Inputs:
      ionMass       ion mass measured in proton masses (scalar)
      density       plasma density on radial grid (array)
      r             radial grid in coordinate r, measured in m (array)
      minorRadius   minor radius a, measured in m (scalar)
      majorRadius   major radius R0, measured in m (scalar)
      B0            magnetic field parameter B0 in specification of analytic geometry.  Measured in Tesla (scalar)
      Vprime        geometric coefficient V' = dV/dr (array)
      gradPsiSq     geometric coefficient <|grad psi|^2>, where psi=r (array)
    """
    B_pi2 = magnetic_geometry_circular(r, minorRadius, majorRadius, B0)
    qbar = parameters.analytic_safety_factor_qbar(r, minorRadius)
    profilesAll = ProfilesAll(ionMass, density, r, minorRadius, majorRadius, B_pi2, qbar, Vprime, gradPsiSq)
    return profilesAll
    
    
def magnetic_geometry_circular(r, a, R0, B0):
    """Set up the model magnetic geometry used in various GENE studies.  This
     involves concentric circular flux surfaces.
     
     Inputs:
         r          radial grid (array)
         a          minor radius  (scalar)
         R0         major radius (scalar)
         B0         magnetic field parameter B0 in specification of analytic geometry.  Measured in Tesla (scalar)
     Outputs:
         B_pi2      magnetic field strength |B| at theta=pi/2 on radial grid.  Measured in Tesla (array)
    
    Reference: X. Lapillonne et al. (2009) - Clarification to limitations of the s-alpha equilibrium model
               
    The Lapillonne paper uses r_n = r/a = r_n
    B = R0 B0 / R * (phi_hat + r/(R0*qbar) theta_hat)
    
    safety factor q related to qbar:
        q = qbar / sqrt(1- r^2 / R0^2)
        McMillan / Lapillonne uses q(rn) = (0.854 + 2.184 rn^2) / sqrt(1 - (rn a / R0)^2)
          i.e., qbar = 0.854 + 2.184 rn^2
    """    
    qbar = parameters.analytic_safety_factor_qbar(r, a)
    B_pi2 = B0 * np.sqrt(1 + (r/(R0*qbar))**2)
    return B_pi2