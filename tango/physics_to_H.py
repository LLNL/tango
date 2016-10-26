"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
from . import physics
from . import derivatives
"""
PhysicsToH

Module for converting between physics-based transport coefficients and the H coefficients specifying the 
transport equation.  This conversion typically involves magnetic geometry coefficients.

The transport equation in toroidal geometry in flux coordinates takes the form [where d/dt = partial time derivative
and d/dpsi = partial psi derivative]

       (1)  3/2 dp/dt + 1/V' d/dpsi [ V' <qturb dot grad psi> ] + ... = S
       
After multiplying through by V', it becomes
      
       (2)  V' 3/2 dp/dt + d/dpsi [ V' <q dot grad psi> ] + ... = V' S

The transport equation is specified in the form  [where d_t = partial time derivative, d_x = partial space derivative] 
       (3)  H_1 d_t U - d_x( H_2 d_x U + H_3 U + H_4) - H_6 U - H_7 = 0     [note, H_5 is not implemented in this package]
       

"""

class Hcontrib_TransportPhysics(object):
    """Provide an interface to computing the contributions to the H coefficients from
    physical effects in the transport equation other than turbulence.
    """
    def __init__(self, profilesAll):
        self.transportPhysics = physics.TransportPhysics(profilesAll)
        self.profilesAll = profilesAll
        self.Vprime = profilesAll.Vprime
        self.gradpsisq = profilesAll.gradpsisq
    def Hcontrib_thermal_diffusivity(self, chi):
        """Compute the contributions to the H coefficients in the transport equation due to a specified
        ion thermal diffusivity.
        
        Inputs:
           chi      ion thermal diffusivity (array)
        Outputs:
           H2contrib    array
           H3contrib    array
        """
        n = self.profilesAll.n
        psi = self.profilesAll.psi
        Vprime = self.Vprime
        gradpsisq = self.gradpsisq
        
        dpsi = psi[1] - psi[0]
        dndpsi = derivatives.dx_centered_difference(n, dpsi)
        (H2contrib, H3contrib) = thermal_diffusivity_to_H(chi, Vprime, gradpsisq, n, dndpsi)
        return (H2contrib, H3contrib)
        
    def Hcontrib_neoclassical_thermal_diffusivity(self, P):
        """Compute the contributions to the H coefficients in the transport equation due to neoclassical
        ion thermal diffusivity.
        
        Inputs:
           P            new ion pressure profile (array) [SIDE EFFECT: stores P internally as the new pressure]
        Outputs:
           H2contrib    array
           H3contrib    array
        """
        chi = self.transportPhysics.neoclassical_chi(P)
        (H2contrib, H3contrib) = self.Hcontrib_thermal_diffusivity(chi)
        return (H2contrib, H3contrib)
        
    def update_pressure(self, PNew):
        self.profilesAll.P = PNew
        self.transportPhysics.update_pressure(PNew)
        
    def time_derivative_to_H(self, timeDerivativeCoeff):
        """Convert the time derivative coefficient to the H1 coefficient
        """
        H1contrib = time_derivative_to_H(timeDerivativeCoeff, self.Vprime)
        return H1contrib
    
    def source_to_H(self, S):
        """Convert an input source term (on the RHS) to the H7 coefficient.
        Inputs:
          S             Source term (array)
          Vprime        Geometric coefficient dV/dpsi (scalar or array)
        Outputs:
          H7contrib     contribution to H7 (array)    
        """
        H7contrib = source_to_H(S, self.Vprime)
        return H7contrib



def source_to_H(S, Vprime):
    """Convert an input source term (on the RHS) to the H7 coefficient.
    Inputs:
      S             Source term (array)
      Vprime        Geometric coefficient dV/dpsi (scalar or array)
    Outputs:
      H7contrib     contribution to H7 (array)    
    """
    
    H7contrib = S * Vprime
    return H7contrib

def PDep_heating_to_H(Pheating, Vprime):
    """Convert a P-dependent heating source, e.g., a term Pheating*P on the RHS of (1) to the H6 coefficient
    
    Inputs:
      Pheating      Heating proportionality coefficient (array)
      Vprime        Geometric coefficient dV/dpsi (scalar or array)
    Outputs:
      H6contrib     contribution to H6 (array)
    """
    H6contrib = Pheating * Vprime
    return H6contrib
def time_derivative_to_H(timeDerivativeCoeff, Vprime):
    """Convert the time derivative coefficient to the H1 coefficient
    """
    H1contrib = timeDerivativeCoeff * Vprime
    return H1contrib

def thermal_diffusivity_to_H(chi, Vprime, gradpsisq, n, dndpsi):
    """Convert a thermal diffusivity chi to the H coefficients.
    
    Add diffusivity in the form of a diffusive heat flux,
         q = -n * chi * grad T
    or more precisely,
         <q dot grad psi> = -n * chi * <grad T dot grad psi>
                          = -n * chi * dT/dpsi * <|grad psi|^2>
    Since our dependent variable is pressure, not temperature, this becomes
                          = -chi <|grad psi|^2> dp/dpsi + chi <|grad psi|^2> * 1/n * dn/dpsi * p
    The result is returned in the form of the H2, H3 coefficients to specify the transport equation.  This requires
    multiplying by V'(psi), another geometric coefficient.
    
    n and dn/dpsi are separate inputs to allow for dn/dpsi to be computed using whatever method desired.  chi, Vprime,
    and gradpsisq may be scalars or arrays --- if scalars they will be expanded.
    
    Inputs:
      chi           Diffusivity chi(psi) (scalar or array)
      Vprime        Geometric coefficient dV/dpsi (scalar or array)
      gradpsisq     Geometric coefficient <|grad psi|^2> (scalar or array)
      n             density n(psi) (array)
      dndpsi        dn/dpsi (array)
    Outputs:
      H2contrib     contribution to H2 from this diffusivity
      H3contrib     contribution to H3 from this diffusivity
    """
    assert np.all(n > 0)
    assert np.all(chi > 0)
    H2contrib = Vprime * chi * gradpsisq * np.ones_like(n)
    H3contrib = -Vprime * chi * gradpsisq / n * dndpsi
    return (H2contrib, H3contrib)
    
def geometrized_diffusion_coeff_to_H(D, Vprime):
    """Convert a "geometrized diffusion coefficient" into an H coefficient.
    
    A geometrized diffusion coefficient D is defined as follows:
    
             flux = -D dp/dpsi, where p is the dependent variable and psi is the independent variable
    
    The geometric coefficient <|grad psi|^2> has been absorbed into D and does not appear.  The transport equation looks like
             V' dp/dt + d/dpsi [ V' * flux] + ...
    
    so the flux contributes V' * D to H2
    """
    H2contrib = Vprime * D
    return H2contrib

def geometrized_convection_coeff_to_H(c, Vprime):
    """Convert a "geometrized convection coefficient" into an H coefficient.
    
    A geometrized convection coefficient c is defined as follows:
    
             flux = c*p, where p is the dependent variable
    
    The geometric coefficient <grad psi> has been absorbed into c and does not appear.  The transport equation looks like
             V' dp/dt + d/dpsi [ V' * flux] + ...
    
    so the flux contributes -V' * c to H3
    """
    H3contrib = -Vprime * c
    return H3contrib

###############################################################
#  End functions for converting from physics-based coefficients to H coefficients

#  Begin functions for converting from H coefficients to physics-based coefficients
###############################################################
    
def H_to_geometrized_diffusion_coeff(H2, Vprime):
    """Convert the H2 coefficient into a "geometrized" diffusion coefficient.
    """
    D = H2 / Vprime
    return D

def H_to_geometrized_convection_coeff(H3, Vprime):
    """Convert the H3 coefficient into a "geometrized convection coefficient.
    """
    c = -H3 / Vprime
    return c
    
def geometrized_diffusion_coeff_to_diffusivity(D, gradpsisq):
    """Convert a "geometrized diffusion coefficient" into a diffusivity that has dimensions of
    Length^2/Time.
    
    A geometrized diffusion coefficient is
        D = chi * <|grad psi|^2>
    where chi is the diffusivity
    """
    chi = D / gradpsisq
    return chi
    
def geometrized_convection_coeff_to_convection_coeff(c, gradpsisq):
    """Convert a "geometrized convection coefficient" into a convection coefficient that has dimensions
    of Length/Time.
    
    The convection coefficient v is given by the relation
             q = v p,
    or
             <q dot grad psi> = <v dot grad psi> p
    Assume vector v is in direction of grad psi, such that
             v = u grad psi
    or
             <v dot grad psi> = u <|grad psi|^2>
    Then,
             <q dot grad psi> = u <|grad psi|^2> p
    The geometrized convection coefficient c is given by
             <q dot grad psi> = c * p
    Hence, we see
            c = u <|grad psi|^2>
    And
            v = c grad psi / <|grad psi|^2>
    
    There is no unambiguous way to compute v on the flux surface because it may vary over the surface.  One 
    quantity we can compute is vbar = sqrt( <|v|^2> ), or
            vbar = c / sqrt <|grad psi|^2 >
    
    """
    vbar = c / np.sqrt(gradpsisq)
    return vbar
    
def H_to_diffusivity(H2, Vprime, gradpsisq):
    """Convert H2 coefficient to thermal diffusivity chi.
    """
    D = H_to_geometrized_diffusion_coeff(H2, Vprime)
    chi = geometrized_diffusion_coeff_to_diffusivity(D, gradpsisq)
    return chi

def HToConvectionCoeff(H3, Vprime, gradpsisq):
    """Convert H3 coefficient to effective convection coefficient.
    """
    c = H_to_geometrized_convection_coeff(H3, Vprime)
    vbar = geometrized_convection_coeff_to_convection_coeff(c, gradpsisq)
    return vbar