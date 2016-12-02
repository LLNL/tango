"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np

"""Shestakov Test Module"""

class shestakov_analytic_fluxmodel(object):
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
    
def GetSource(x):
    """Test problem from Shestakov et al. (2003).
    Return the source S."""
    S = np.zeros_like(x)
    S0 = 1
    delta = 0.1
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
    
def GetSteadyStateSolution(x, nL):
    
    S0 = 1
    delta = 0.1
    #D0 = 1
    #L = 1
    nright = ( nL**(1/3) + 1/3 * (S0 * delta)**(1/3) *(1-x) )**3
    # nleft = (L - delta + 0.75*(delta - x**(4/3) / delta**(1/3)))**3-
    nleft = ( nL**(1/3) + 1/3 * (S0 * delta)**(1/3) * (1 - delta + (3/4) * (delta - x**(4/3) / delta**(1/3))))**3
    nss = nright
    nss[x < delta] = nleft[x < delta]
    #nss[x < delta] = S0 * delta / (27*D0) * nleft[x < delta]
    return nss