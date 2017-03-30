"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np

"""Shestakov Test Module"""
class AnalyticFluxModel(object):
    """Class-based interface to the flux model in the Shestakov analytic
    test problem.
    
    Slight modification of boundary conditions so that n(1) = nL rather than n(1)=0,
    in order to make the computation of the flux at the boundary much easier by
    avoiding a divide by zero condition.
    """
    def __init__(self, dx):
        self.dx = dx
    
    def get_flux(self, n):
        return get_flux(n, self.dx)

class shestakov_analytic_fluxmodel(object):
    """Class-based interface to the flux model in the Shestakov analytic
    test problem.
    
    Slight modification of boundary conditions so that n(1) = nL rather than n(1)=0,
    in order to make the computation of the flux at the boundary much easier by
    avoiding a divide by zero condition.
    
    Alias to AnalyticFluxModel which conforms to style guidleines.  This class is kept around for backwards compatibility reasons.
    """
    def __init__(self, dx):
        self.dx = dx
    
    def get_flux(self, n):
        return get_flux(n, self.dx)
    
    
    
#==============================================================================
#   Functions specifying the Shestakov analytic test problem start here
#==============================================================================
def H7contrib_Source(x):
    S = GetSource(x)
    H7 = S
    return H7


def get_flux(n, dx):
    """Test problem from Shestakov et al. (2003)
    Return the flux Gamma, which depends on the density profile n as follows:
       Gamma[n] = -(dn/dx)^3 / n^2
    """
    Gamma = np.zeros_like(n)
    
    # Return flux Gamma on the same grid as n
    dndx = _dxCenteredDifference(n, dx)
    Gamma = - dndx**3 / n**2    
    return Gamma
    
def GetSource(x, S0=1, delta=0.1):
    """Test problem from Shestakov et al. (2003).
    Return the source S."""
    S = np.zeros_like(x)
    S[x < delta] = S0
    return S
    
def _dxCenteredDifference(u, dx):
    """Compute du/dx.
      du/dx is computed using centered differences on the same grid as u.  For the edge points, one-point differences are used.
    
    Inputs:
      u         profile (array)
      dx        grid spacing (scalar)
    
    Outputs:
      dudx      (array, same length as u)
    """
    dudx = np.zeros_like(u)
    dudx[0] = (u[1] - u[0]) / dx
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    dudx[-1] = (u[-1] - u[-2]) / dx
    return dudx
    
def steady_state_solution(x, nL, S0=1, delta=0.1):
    """Return the exast steady state solution for the Shestakov test problem
    
    Inputs:
      x             Spatial coordinate grid (array)
      nL            boundary condition n(L) (scalar)
      S0            parameter in source term --- amplitude (scalar)
      delta         parameter in source term --- location where it turns off (scalar)
    Outputs:
    """
    nright = ( nL**(1/3) + 1/3 * (S0 * delta)**(1/3) *(1-x) )**3
    # nleft = (L - delta + 0.75*(delta - x**(4/3) / delta**(1/3)))**3-
    nleft = ( nL**(1/3) + 1/3 * (S0 * delta)**(1/3) * (1 - delta + (3/4) * (delta - x**(4/3) / delta**(1/3))))**3
    nss = nright
    nss[x < delta] = nleft[x < delta]
    return nss
    
def GetSteadyStateSolution(x, nL, S0=1, delta=0.1):
    """alias to steady_state_solution which conforms to style guidelines, but this function is kept around for backwards compatibility."""
    return steady_state_solution(x, nL, S0=S0, delta=delta)