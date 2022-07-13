"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np

class AnalyticFluxModel:
    """Flux model from Shestakov et al. (2003)
        Return the flux Gamma, which depends on the density profile n as follows:
        Gamma[n] = -(dn/dx)^p * n^q
        """
    def __init__(self, dx, p = 3, q = -2, field='n'):
        """
        # Inputs
        dx : Grid spacing, used to calculate gradients
        p : int
            Flux depends on gradient to this power
        """
        self.dx = dx
        self.p = p
        self.q = q
        self.field = field

    def get_flux(self, profiles):
        """Return flux as a dictionary, using profiles dictionary
        """
        n = profiles[self.field]
        flux = {}
        flux[self.field] = self.calc_flux(n, self.dx)
        return flux

    def calc_flux(self, n, dx):
        """Return flux Gamma on the same grid as n
        """
        dndx = self._dxCenteredDifference(n, dx)
        return - dndx**self.p * n**self.q

    def calc_flux_gradient(self, n, dx):
        """
        Return divergence of flux
        """
        flux = self.get_flux(n, dx)
        return self._dxCenteredDifference(flux, dx)

    def _dxCenteredDifference(self, u, dx):
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

class ShestakovTestProblem(AnalyticFluxModel):
    """Test problem from Shestakov et al. (2003)
        Return the flux Gamma, which depends on the density profile n as follows:
        Gamma[n] = -(dn/dx)^p * n^q
        """
    def __init__(self, dx, p = 3, q = -2, S0 = 1, delta = 0.1):
        """
        # Inputs
        dx            Grid spacing
        S0            parameter in source term --- amplitude (scalar)
        delta         parameter in source term --- location where it turns off (scalar)
        """
        super(ShestakovTestProblem, self).__init__(dx, p=p, q=q, field='n')
        self.S0 = S0
        self.delta = delta

    def H7contrib_Source(self, x):
        return self.GetSource(x)

    def GetSource(self, x):
        """Test problem from Shestakov et al. (2003).
        Return the source S."""
        S = np.zeros_like(x)
        S[x < self.delta] = self.S0
        return S

    def steady_state_solution(self, x, nL):
        """Return the exact steady state solution for the Shestakov test problem
        Generalised for arbitrary p, q

        Inputs:
        x             Spatial coordinate grid (array)
        nL            boundary condition n(L) (scalar)
        Outputs:
        """

        coef = self.q / self.p + 1.  # This is 1/3 for p=3,q=-2 standard case
        L = 1.0

        # Solution in region delta < x < L
        nright = (nL**coef + coef * (self.S0 * self.delta)**(1./self.p) * (L - x))**(1./coef)
        nleft = (nL**coef + coef * (self.S0 * self.delta)**(1./self.p) * (L - self.delta + (self.p/(self.p + 1)) * (self.delta - x * (x / self.delta)**(1./self.p))))**(1./coef)

        nss = nright
        nss[x < self.delta] = nleft[x < self.delta]
        return nss
    
    def GetSteadyStateSolution(self, x, nL):
        """alias to steady_state_solution which conforms to style guidelines, but this function is kept around for backwards compatibility."""
        return self.steady_state_solution(x, nL)
