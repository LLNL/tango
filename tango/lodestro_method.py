"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
from . import derivatives

"""
lodestro_method

Module for implementing a crucial part of the LoDestro Method:  transform turbulent fluxes into
  transport coefficients, using iterate-averaging for smoothing and stabilization.
  
The turbulent flux is transformed into effective diffusive and convective transport coefficients.
  There is a lot of freedom in how this split into diffusive and convective contributions is
  performed, which is handled by the "ftheta" function  .

Iterate averaging is achieved through an exponentially weighted moving average (EWMA), which is a
  form of relaxation.  For a quantity y_l, where l is the iteration index, the EWMA is denoted by
  yEWMA_l.  Each iterate of the EWMA is computed from
  
          yEWMA_l = A*y_l  +  (1-A)*yEWMA_{l-1}

  Both the turbulent fluxes and profiles will be relaxed in this form.  Different relaxation
  parameters A_turbflux and A_profile are allowed.
"""

class TurbulenceHandler(object):
    """This class acts as an interface to underlying packages for the Lodestro Method, a model 
    for turbulent flux, and moving between transport and turbulence grids (if necessary).

    The class is designed to be flexible and accept as given a model to compute the turbulent
    flux.  In principle, this model could be analytic or based on a turbulence code.  The
    requirement is that it be given as the input FluxModel and have a GetFlux() method.
    
    Since the class provides the H coefficients, and not simply diffusion coefficients, it must
    be aware of the coordinate geometric factor V' = dV/dpsi.
    """
    def __init__(self, dx, lmparams, FluxModel, grids=None, Vprime=None):
        """A geometric factor Vprime may be optionally provided to the constructor.  This 
            geometric factor is essentially a Jacoabian and appears in transport equations
            in non-Cartesian geometry.  It typically appears in the form
                        dn/dt = 1/V' * d/dpsi(V' Gamma) + ...  
            For example, in polar coordinates where r is the flux label, V' is proportional
            to r.  If Vprime is not provided, Cartesian geometry is assumed, where V' = 1.
    
            Here, Vprime = dV/psi.  The flux coordinate psi does not have to be the poloidal
            flux function; any flux label is valid.
            
            Inputs:
              dx          grid spacing for independent variable x [sometimes referred to in
                            the documentation as psi] (scalar)
              lmparams    parameters to be used by the lodestro method (dict)
                            --> EWMA_param_turbflux, EWMA_param_profile, thetaparams
              FluxModel   object with a GetFlux() method that provides turbulent flux
                             GetFlux() accepts the profile on the turbulence grid and returns
                             the turbulent flux on the turbulence grid
              grids       (optional) object for transforming the profiles and transport coefficients
                            between the transport grid and the grid used to compute turbulent fluxes
                                default: None [same grid for transport and turbulence]
              Vprime      (optional) geometric coefficients dV/dpsi depending on coordinate system (array)
                                default: None [V'=1]
        """
        self.dx = dx
        self.LoDestroMethod = lm(lmparams['EWMA_param_turbflux'], lmparams['EWMA_param_profile'], lmparams['thetaparams'])
        
        self.FluxModel = FluxModel
        assert hasattr(FluxModel, 'GetFlux') and callable(getattr(FluxModel, 'GetFlux'))
        
        if grids is not None:
            assert hasattr(grids, 'MapProfileOntoTurbGrid') and callable(getattr(grids, 'MapProfileOntoTurbGrid'))
            assert hasattr(grids, 'MapTransportCoeffsOntoTransportGrid') and callable(getattr(grids, 'MapTransportCoeffsOntoTransportGrid'))
            self.grids = grids
        else:
            self.grids = grids_DoNothing()
        
        if Vprime is not None:
            self.Vprime = Vprime
            self.isNonCartesian = True
        else:
            self.isNonCartesian = False
        
    def Hcontrib_TurbulentFlux(self, profile):
        """Given the lth iterate of a profile (e.g., n), perform all the steps necessary to
        calculate the contributions to the transport equation.  This involves using the
        LoDestro Method to average over iterates, the FluxModel to compute a turbulent flux,
        and the LoDestro Method to transform the flux into transport coefficients.  Additionally,
        any necessary transformations between the quantities on the transport grid and the
        quantities on the turbulence grid are also handled.
         
        Inputs:
          profile           (array)
        
        Outputs:
          H2contrib         array
          H3contrib         array
          data              other data
        """
        profile_turbgrid = self.grids.MapProfileOntoTurbGrid(profile)
        profileEWMA_turbgrid = self.LoDestroMethod.EWMA_Profile(profile_turbgrid) # could also reverse the order of this and the previous step...
        
        flux_turbgrid = self.FluxModel.GetFlux(profileEWMA_turbgrid)
        fluxEWMA_turbgrid = self.LoDestroMethod.EWMA_TurbFlux(flux_turbgrid)
        (D_turbgrid, c_turbgrid, Dcdata_turbgrid) = self.LoDestroMethod.FluxToTransportCoeffs(fluxEWMA_turbgrid, profileEWMA_turbgrid, self.dx)
        (D, c) = self.grids.MapTransportCoeffsOntoTransportGrid(D_turbgrid, c_turbgrid)
        (H2contrib, H3contrib) = self.DcToHcontrib(D, c)
        
        # Other data that may be useful for debugging or data analysis purposes
        data = {'D': D, 'c': c,
                'profile_turbgrid': profile_turbgrid, 'profileEWMA_turbgrid': profileEWMA_turbgrid,
                'flux_turbgrid': flux_turbgrid, 'fluxEWMA_turbgrid': fluxEWMA_turbgrid,
                'D_turbgrid': D_turbgrid, 'c_turbgrid': c_turbgrid,
                'Dhat_turbgrid': Dcdata_turbgrid['D_hat'], 'chat_turbgrid': Dcdata_turbgrid['c_hat'], 'theta_turbgrid': Dcdata_turbgrid['theta']}
        return (H2contrib, H3contrib, data)
    
    def get_EWMA_params(self):
        """Return the EWMA parameter for turbulent flux and the profile, respectively.

        Outputs:
          EWMAparam_turbflux    (scalar)
          EWMAparam_profile     (scalar)
        """
        (EWMAparam_turbflux, EWMAparam_profile) = self.LoDestroMethod.get_EWMA_params()
        return (EWMAparam_turbflux, EWMAparam_profile)
        
        
        
        
    def DcToHcontrib(self, D, c):
        """Transform the effective diffusion coefficient D and effective convective velocity c
        into the contributions to the H coefficients for the iteration-update solver for the
        transport equation.  The form of the transport equation for ion pressure is 
                3/2 V' dp/dt - d/dpsi[ V' D dp/dpsi - V'c p] 
        Hence, H2 = V'*D  and  H3 = -V'*c.
        """
        #  Should change this to use the physics_to_H module
        #     H2contrib = physics_to_H.GeometrizedDiffusionCoeffToH(D, Vprime)
        #     H3contrib = physics_to_H.GeometrizedConvectionCoeffToH(c, Vprime):
        if self.isNonCartesian == True:
            H2contrib = self.Vprime * D
            H3contrib = -self.Vprime * c
        elif self.isNonCartesian == False:
            H2contrib = D
            H3contrib = -c
        else:
            raise RuntimeError('Invalid value for isNonCartesian.')
        return (H2contrib, H3contrib)
    
        
        
class lm(object):
    """High level class for the LoDestro method: Handle two separate things:
        1) The exponentially weighted moving average (EWMA) of both the turbulent flux and the profiles
        2) Transform (averaged) turbulent fluxes into effective transport coefficients
    These two functions are handled with two separate classes    
    """
    def __init__(self, EWMA_param_turbflux, EWMA_param_profile, thetaparams):
        # create instances for handling EWMAs
        self._EWMAturbflux = EWMA(EWMA_param_turbflux)
        self._EWMAprofile = EWMA(EWMA_param_profile)
        
        # Create instance of FluxSplit
        self._FluxSplitter = FluxSplit(thetaparams) 
    
    # Provide an interface to the EWMA and FluxSplitter methods
    def EWMA_TurbFlux(self, turbflux_l):
        """Return the next iterate of the exponentially weighted moving average of the turbulent flux.
        See EWMA.NextEWMAIterate() for more detail.
        Inputs:  
          turbflux_l             current value (iterate l) of turbflux; array
          
        Outputs:
          turbfluxEWMA_l         current value (iterate l) of turbfluxEWMA_l; array        
        """
        return self._EWMAturbflux.NextEWMAIterate(turbflux_l)
        
    def EWMA_Profile(self, profile_l):
        """Return the next iterate of the exponentially weighted moving average of the profile.
        See EWMA.NextEWMAIterate() for more detail.
        Inputs:  
          profile_l             current value (iterate l) of profile; array
          
        Outputs:
          profileEWMA_l         current value (iterate l) of profileEWMA_l; array        
        """
        return self._EWMAprofile.NextEWMAIterate(profile_l)
        
    def FluxToTransportCoeffs(self, flux, p, dx):
        """Transform a flux into effective transport coefficients.
        See FluxSplit.FluxToTransportCoeffs() for more detail.
        
        Inputs:
          flux          (averaged) flux given on integer grid points (array)
          p             (averaged) pressure profile p given on integer grid points (array)
          dx            grid spacing (scalar)
        
        Outputs:
          D             Effective diffusion coefficient (array)
          c             Effective convective coefficient (array)
          data          Other data useful for debugging (dict)        
        """
        (D, c, data) = self._FluxSplitter.FluxToTransportCoeffs(flux, p, dx)
        return (D, c, data)
        
    def get_EWMA_params(self):
        """Return the EWMA parameter for turbulent flux and the profile, respectively.

        Outputs:
          EWMAparam_turbflux    (scalar)
          EWMAparam_profile     (scalar)
        """
        EWMAparam_turbflux = self._EWMAturbflux.EWMA_param 
        EWMAparam_profile = self._EWMAprofile.EWMA_param
        return (EWMAparam_turbflux, EWMAparam_profile)
        
        
class FluxSplit(object):
    """Class for splitting a flux into diffusive and convective contributions.  Any averaging to be applied to the flux or
    profiles is assumed to be applied externally.  This, for a given profile p, determines D and c such that
        Gamma = -D*dp/dx + c*p
    """
    def __init__(self, thetaparams):
        """Class constructor
            Inputs:
               thetaparams              dict containing parameters to be used in the ftheta function.
        """
        # define/initialize internal varibales
        self.thetaparams = thetaparams
    
    def FluxToTransportCoeffs(self, flux, p, dx):
        """Given the current iterate of the (averaged/relaxed) flux, use the LoDestro Method to compute effective transport coefficients
            to be used in Picard Iteration.
        
        For all purposes here, the flux is assumed to be already averaged using the EWMA relaxation or other method.  The flux is split
          into effective diffusive and convective contributions; what is returned are the transport coefficients D and c.  If the inputs
          are the fluxes and the profile p, then the flux is computed in the form
        
            Flux = -D * (dp/dx)  +  c * p
        
        where p is the pressure (the dependent variable), D is the diffusion coefficient, c is the convective velocity, and x is the
          independent variable (which may be some flux coordinate psi).
          
        Inputs:
          flux          (averaged) flux given on integer grid points (array)
          p             (averaged) pressure profile p given on integer grid points (array)
          dx            grid spacing (scalar)
        
        Outputs:
          D             Effective diffusion coefficient (array)
          c             Effective convective coefficient (array)
          data          Other data useful for debugging (dict)
        
          *************************
        
        This function first calculates D_hat and c_hat, which are the diffusion and convective coefficients that would result if the
          turbulent flux were represented as purely diffusive (D_hat) or purely convective (c_hat).  That is,
          
            D_hat = -Flux / (dp/dx),
            c_hat = Flux / p
        
        Then, a coefficient theta is computed that determines the split between diffusive and convective contributions.
        
            D = theta * D_hat
            c = (1 - theta) * c_hat
            
            0 <= theta <= 1
            
        The coefficient theta may vary throughout space.  There is a lot of freedom in how theta is chosen; various schemes may work.
        
          *************************
        Note that in general, a *vector* flux will have geometric coefficients appear (in particular, |grad psi|^2).  But we need not
          worry about these geometric coefficients.  For example, consider diffusive effects.  A vector flux Q results in the quantity
          of interest 
            
            qturb = Q dot grad psi.
        
        A Fick's Law diffusive assumption would involve writing, for some D2,
        
            Q = -D2 * grad p = -D2 * (dp/dpsi) * grad psi,
        
        which leads to
        
            qturb = Q dot grad psi = -D2 |grad psi|^2 dp/dpsi.
            
        Defining D = D2 |grad psi|^2, this leads to
        
            qturb = -D dp/dpsi.
            
        Assume the input here is the number qturb, which is the radial flux (already dotted with grad psi).  qturb is what will be returned
          by a turbulence code; the vector flux Q will not be returned.  Hence, we do not have to worry about the geometric coefficient
          |grad psi|^2.  It is absorbed into the effective diffusive coefficient D returned by this function.  Note the dimensions of D
          are different than the dimensions of D2 if |grad psi|^2 is not dimensionless.       
        """
        dpdx = derivatives.dx_centered_difference(p, dx)
        D_hat = -flux / dpdx
        D_hat[dpdx==0] = 0     # get rid of infinities resulting from divide by zero
        c_hat = flux / p
        
        theta = self._ftheta(D_hat, dpdx, self.thetaparams)
        # uncomment the following line to turn off convective terms and use only diffusive terms
        # theta[:] = 1        
        
        D = theta * D_hat
        c = (1 - theta) * c_hat
        
        # "data" contains other data that may be useful for debugging purposes
        data = {'D_hat': D_hat, 'c_hat': c_hat, 'theta': theta}
        return (D, c, data)
        
    @staticmethod
    def _ftheta(D_hat, dpdx, thetaparams):
        """Scheme to calculate theta, the parameter that determines the split between diffusive and convective pieces in representations
          of the flux.
        
        Modification of Shestakov's default algorithm.  Here, when Dhat is large, we only add a convective part if dp/dx is also SMALL.
          In other words, if Flux and D_hat are large because dp/dx is large, then representing the flux purely as diffusive is fine.
          The convective split for large D_hat is really to protect against spurious large D_hat resulting from finite flux at small
          gradients.
          
            if D_hat < Dmin, set theta to 0 (all convective)
            if D_hat >= Dmin AND dp/dx is small, use the Shestakov formula
            otherwise, set theta = 1 (all diffusive)
            
        What to use for Dmin, Dmax, and dpdx_thresh will depend on the problem.  The numerical values will further depend on what
          units are used to represent the dependent and independent variables.
        """
        Dmin = thetaparams['Dmin']  # scalar
        Dmax = thetaparams['Dmax']  # scalar
        dpdx_thresh = thetaparams['dpdx_thresh'] # scalar
        
        ind1 = D_hat < Dmin
        ind2 = np.logical_and.reduce((abs(dpdx) < dpdx_thresh, D_hat >= Dmin, D_hat <= Dmax))
        ind3 = np.logical_and(abs(dpdx) < dpdx_thresh, D_hat > Dmax)
        
        theta = np.ones_like(D_hat)
        theta[ind1] = 0
        theta[ind2] = (Dmax - D_hat[ind2]) / (Dmax - Dmin)
        theta[ind3] = 0
        
        assert np.count_nonzero(np.logical_and(theta>=0, theta<=1)) == np.size(theta), 'some theta is not between 0 and 1'
        return theta
    
        
class EWMA(object):
    """Class for handling the exponentially weighted moving average.  Each instance stores a previous iterate and the relaxation
    parameter for a single profile.
    """
    def __init__(self, EWMA_param):
        # define/initialize internal varibales
        self.EWMA_param = EWMA_param
        
        # memory of previous relaxed iterates 
        self._yEWMA_lminus1 = None
    
    def NextEWMAIterate(self, y_l):
        """Return the next iterate of the exponentially weighted moving average (EWMA).
        Inputs:  
          y_l             current value (iterate l) of y; array
          
        Outputs:
          yEWMA_l         current value (iterate l) of relaxed y; array
          
                  yEWMA_l = A*y_l  +  (1-A)*yEWMA_{l-1}, where A is the EWMA_param
        
        Side effect: save the returned value yEWMA_l as the "old" _yEWMA_lminus1, so that when this function is called again,
          it will be used for the next relaxation iterate.
        """
        if self._yEWMA_lminus1 is None:               # initialization for the first time this is called
            self._yEWMA_lminus1 = y_l
        yEWMA_l = self._ComputeNextEWMA(y_l, self._yEWMA_lminus1, self.EWMA_param)
        
        self._yEWMA_lminus1 = yEWMA_l
        return yEWMA_l
    
    @staticmethod
    def _ComputeNextEWMA(y_l, yEWMA_lminus1, EWMA_param):
        """Compute the next iterate of the exponentially weighted moving average (EWMA) of a quantity y.
        This can be done using a recursive formula that is a form of relaxation.
        
        yEWMA_l = EWMA_param*y_l  +  (1-EWMA_param) * yEWMA_l-1
        Note: EWMA_param = 1  means no relaxation (use current value), while 
              EWMA_param = .001 means a lot of relaxation (use very little of current value)
        """
        return EWMA_param * y_l  +  (1-EWMA_param) * yEWMA_lminus1
        
    def ResetEWMAIterate(self):
        self._EWMA_lminus1 = None
        
class grids_DoNothing(object):
    """Placeholder class for moving between grids when the turbulence grid will be the same as the transport grid.
    No interpolation of quantities between grids will be performed, as there is only one grid.
    """
    def MapProfileOntoTurbGrid(self, profile):
        return profile
    def MapTransportCoeffsOntoTransportGrid(self, D, c):
        return (D, c)    