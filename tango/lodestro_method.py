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
  
See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np

from . import derivatives
from . import multifield


class TurbulenceHandler(object):
    """    
    This class acts as a high-level interface to effect the LoDestro Method.

    The class is designed to be flexible and accept as given a model to compute the turbulent
    flux.  In principle, this model could be analytic or based on a turbulence code.  The
    requirement is that it be given as the input fluxModel and have a get_flux() method.  The
    get_flux() method must take the input profiles as a dict (arrays accessed by label) and return
    the turbulent fluxes as a dict (accessed by the same labels, corresponding to their associated
    conserved quantity).
    
    Since the class provides the H coefficients, and not simply diffusion coefficients, it must
    be aware of the coordinate geometric factor V' = dV/dpsi.
    """
    def __init__(self, dxTurbGrid, xTango, fluxModel, VprimeTango=None, fluxSmoother=None):
        """A geometric factor Vprime may be optionally provided to the constructor.  This 
            geometric factor is essentially a Jacobian and appears in transport equations
            in non-Cartesian geometry.  It typically appears in the form
                        dn/dt = 1/V' * d/dx (V' Gamma) + ...  
            For example, in polar coordinates where r is the flux label, V' is proportional
            to r.  If Vprime is not provided, Cartesian geometry is assumed, where V' = 1.
    
            Here, Vprime = dV/dx.
            
            Inputs:
              dxTurbGrid                grid spacing for independent variable x on the turbulence grid (scalar)
              xTango                    independent variable x on the transport grid (array)
              fluxModel                 object with a get_flux() method that provides turbulent flux
                                            get_flux() accepts the profiles (as a dict) on the turbulence grid and returns
                                            the turbulent fluxes (as a dict) on the turbulence grid
              VprimeTango (optional)    geometric coefficients dV/dpsi depending on coordinate system, on the transport grid (array)
                                            default: None [V'=1 everywhere]
              fluxSmoother (optional)   object with a smooth() method to spatially smooth the turbulent flux. 
        """
        self.dxTurbGrid = dxTurbGrid
        self.xTango = xTango        
        self.fluxModel = fluxModel
        assert hasattr(fluxModel, 'get_flux') and callable(getattr(fluxModel, 'get_flux'))
        
        if VprimeTango is not None:
            self.VprimeTango = VprimeTango
            self.isNonCartesian = True
        else:
            self.isNonCartesian = False
            
        if fluxSmoother is not None:
            assert hasattr(fluxSmoother, 'smooth') and callable(getattr(fluxSmoother, 'smooth'))
            self.fluxSmoother = fluxSmoother
            self.doSmoothing = True
        else:
            self.doSmoothing = False
            
    def turbflux_to_Hcoeffs_multifield(self, fields, profiles):
        """Compute the turbulent fluxes and transform to effective transport (H) coefficients.
        
        
        Inputs:
          fields                    collection of fields (list)
          profiles                  collection of profiles, accessed by label (dict)
        Outputs:
          HCoeffsTurbAllFields      collection of HCoefficients, accessed by label, containing the contributions from turbulence (dict)
          extradataAllFields        extra data that might be useful for debugging or data analysis (dict)
        """
        
        # map profiles to turbulence grid, and compute next EWMA iterate of profiles
        profilesTurbGrid = {}
        profilesEWMATurbGrid = {}
        for field in fields:
            label = field.label
            profileTurbGrid = field.gridMapper.map_profile_onto_turb_grid(profiles[label])
            profileEWMATurbGrid = field.lodestroMethod.ewma_profile(profileTurbGrid)
            profilesEWMATurbGrid[label] = profileEWMATurbGrid
            profilesTurbGrid[label] = profileTurbGrid
            
        # get next turbulent flux
        fluxesTurbGrid = self.fluxModel.get_flux(profilesEWMATurbGrid)
        
        # Loop over fields and transform flux into effective transport coefficients
        #  initialize dicts
        HCoeffsTurbAllFields = {}
        extradataAllFields = {}
        for field in fields:
            label = field.label
            # spatially smooth the flux, if specified
            if self.doSmoothing:
                smoothedFluxTurbGrid = self.fluxSmoother.smooth(fluxesTurbGrid[label])
            else:
                smoothedFluxTurbGrid = fluxesTurbGrid[label]
                        
            # calculate the next iterate of relaxed flux using EWMA
            fluxEWMATurbGrid = field.lodestroMethod.ewma_turb_flux(smoothedFluxTurbGrid)                
            
            # Convert the flux into effective transport coefficients
            (DTurbGrid, cTurbGrid, DcDataTurbGrid) = field.lodestroMethod.flux_to_transport_coeffs(fluxEWMATurbGrid, profilesEWMATurbGrid[label], self.dxTurbGrid)
            
            # Map the transport coefficients from the turbulence grid back to the transport grid
            (D, c) = field.gridMapper.map_transport_coeffs_onto_transport_grid(DTurbGrid, cTurbGrid)
            (H2contrib, H3contrib) = self.Dc_to_Hcontrib(D, c)
            HCoeffsTurb = multifield.HCoefficients(H2=H2contrib, H3=H3contrib)
            HCoeffsTurbAllFields[label] = HCoeffsTurb
            # Other data that may be useful for debugging or data analysis purposes
            extradataAllFields[label] = {
                'D': D, 'c': c,
                'profileTurbGrid': profilesTurbGrid[label], 'profileEWMATurbGrid': profilesEWMATurbGrid[label],
                'fluxTurbGrid': fluxesTurbGrid[label], 'smoothedFluxTurbGrid': smoothedFluxTurbGrid, 'fluxEWMATurbGrid': fluxEWMATurbGrid,
                'DTurbGrid': DTurbGrid, 'cTurbGrid': cTurbGrid,
                'DHatTurbGrid': DcDataTurbGrid['DHat'], 'cHatTurbGrid': DcDataTurbGrid['cHat'], 'thetaTurbGrid': DcDataTurbGrid['theta']}
        
            # other data to save??
#           x = self.gridMapper.get_x_transport_grid()        
#           xTurbGrid = self.gridMapper.get_x_turbulence_grid()
            
        return (HCoeffsTurbAllFields, extradataAllFields)
            
    def Dc_to_Hcontrib(self, D, c):
        """Transform the effective diffusion coefficient D and effective convective velocity c
        into the contributions to the H coefficients for the iteration-update solver for the
        transport equation.  The form of the transport equation for ion pressure is 
                3/2 V' dp/dt - d/dpsi[ V' D dp/dpsi - V'c p] + ...
        Hence, H2 = V'*D  and  H3 = -V'*c.
        """
        #  Could change this to use the physics_to_H module
        #     H2contrib = physics_to_H.GeometrizedDiffusionCoeffToH(D, Vprime)
        #     H3contrib = physics_to_H.GeometrizedConvectionCoeffToH(c, Vprime):
        if self.isNonCartesian == True:
            H2contrib = self.VprimeTango * D
            H3contrib = -self.VprimeTango * c
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
    def __init__(self, EWMAParamTurbFlux, EWMAParamProfile, thetaParams):
        # create instances for handling EWMAs
        self._EWMATurbFlux = EWMA(EWMAParamTurbFlux)
        self._EWMAProfile = EWMA(EWMAParamProfile)
        
        # Create instance of FluxSplit
        self._fluxSplitter = FluxSplit(thetaParams) 
    
    # Provide an interface to the EWMA and fluxSplitter methods
    def ewma_turb_flux(self, turbFlux_l):
        """Return the next iterate of the exponentially weighted moving average of the turbulent flux.
        See EWMA.next_ewma_iterate() for more detail.
        Inputs:  
          turbFlux_l             current value (iterate l) of turbflux; array
          
        Outputs:
          turbFluxEWMA_l         current value (iterate l) of turbfluxEWMA_l; array        
        """
        return self._EWMATurbFlux.next_ewma_iterate(turbFlux_l)
        
    def ewma_profile(self, profile_l):
        """Return the next iterate of the exponentially weighted moving average of the profile.
        See EWMA.next_ewma_iterate() for more detail.
        Inputs:  
          profile_l             current value (iterate l) of profile; array
          
        Outputs:
          profileEWMA_l         current value (iterate l) of profileEWMA_l; array        
        """
        return self._EWMAProfile.next_ewma_iterate(profile_l)
        
    def flux_to_transport_coeffs(self, flux, p, dx):
        """Transform a flux into effective transport coefficients.
        See FluxSplit.flux_to_transport_coeffs() for more detail.
        
        Inputs:
          flux          (averaged) flux given on integer grid points (array)
          p             (averaged) pressure profile p given on integer grid points (array)
          dx            grid spacing (scalar)
        
        Outputs:
          D             Effective diffusion coefficient (array)
          c             Effective convective coefficient (array)
          data          Other data useful for debugging (dict)        
        """
        (D, c, data) = self._fluxSplitter.flux_to_transport_coeffs(flux, p, dx)
        return (D, c, data)
        
    def get_ewma_params(self):
        """Return the EWMA parameter for turbulent flux and the profile, respectively.

        Outputs:
          EWMAParamTurbFlux    (scalar)
          EWMAParamProfile     (scalar)
        """
        EWMAParamTurbFlux = self._EWMATurbFlux.EWMAParam 
        EWMAParamProfile = self._EWMAProfile.EWMAParam
        return (EWMAParamTurbFlux, EWMAParamProfile)
        
    def set_ewma_params(self, EWMAParamTurbFlux, EWMAParamProfile):
        """Set the EWMA parameter for turbulent flux and the profile.
        
        Inputs:
          EWMAParamTurbFlux     (scalar)
          EWMAParamProfile      (scalar)
        """
        self._EWMATurbFlux.EWMAParam = EWMAParamTurbFlux
        self._EWMAProfile.EWMAParam = EWMAParamProfile
    
    def set_ewma_iterates(self, profileEWMA, turbFluxEWMA):
        """Set the EWMA iterates for both the profile and turbulent flux.
        
        Inputs:
          profileEWMA   New EWMA iterate for the profile (array)
          turbFluxEWMA  New EWMA iterate for the turbulent flux (array)
        """
        self.set_ewma_profile(profileEWMA)
        self.set_ewma_turb_flux(turbFluxEWMA)
    
    def set_ewma_profile(self, profileEWMA):
        """Set the EWMA iterate for the profile"""
        self._EWMAProfile.set_ewma_iterate(profileEWMA)
    
    def set_ewma_turb_flux(self, turbFluxEWMA):
        """Set the EWMA iterate for the turbulent flux"""
        self._EWMATurbFlux.set_ewma_iterate(turbFluxEWMA)
    
        
class FluxSplit(object):
    """Class for splitting a flux into diffusive and convective contributions.  Any averaging to be applied to the flux or
    profiles is assumed to be applied externally.  This, for a given profile p, determines D and c such that
        Gamma = -D*dp/dx + c*p
    """
    def __init__(self, thetaParams):
        """Class constructor
        Inputs:
          thetaParams              dict containing parameters to be used in the ftheta function.
        """
        # define/initialize internal varibales
        self.thetaParams = thetaParams
    
    def flux_to_transport_coeffs(self, flux, p, dx):
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
        
        This function first calculates DHat and cHat, which are the diffusion and convective coefficients that would result if the
          turbulent flux were represented as purely diffusive (DHat) or purely convective (cHat).  That is,
          
            DHat = -Flux / (dp/dx),
            cHat = Flux / p
        
        Then, a coefficient theta is computed that determines the split between diffusive and convective contributions.
        
            D = theta * DHat
            c = (1 - theta) * cHat
            
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
        dpdx = derivatives.dx_centered_difference_edge_first_order(p, dx)
        DHat = -flux / dpdx
        DHat[dpdx==0] = 0     # get rid of infinities resulting from divide by zero
        cHat = flux / p
        
        theta = self._ftheta(DHat, dpdx, self.thetaParams)
        # uncomment the following line to turn off convective terms and use only diffusive terms
        # theta[:] = 1        
        
        D = theta * DHat
        c = (1 - theta) * cHat
        
        # "data" contains other data that may be useful for debugging purposes
        data = {'DHat': DHat, 'cHat': cHat, 'theta': theta}
        return (D, c, data)
        
    @staticmethod
    def _ftheta(DHat, dpdx, thetaParams):
        """Scheme to calculate theta, the parameter that determines the split between diffusive and convective pieces in representations
          of the flux.
        
        Modification of Shestakov's default algorithm.  Here, when Dhat is large, we only add a convective part if dp/dx is also SMALL.
          In other words, if Flux and DHat are large because dp/dx is large, then representing the flux purely as diffusive is fine.
          The convective split for large DHat is really to protect against spurious large DHat resulting from finite flux at small
          gradients.
          
            if DHat < Dmin, set theta to 0 (all convective)
            if DHat >= Dmin AND dp/dx is small, use the Shestakov formula
            otherwise, set theta = 1 (all diffusive)
            
        What to use for Dmin, Dmax, and dpdxThreshold will depend on the problem.  The numerical values will further depend on what
          units are used to represent the dependent and independent variables.
        """
        Dmin = thetaParams['Dmin']  # scalar
        Dmax = thetaParams['Dmax']  # scalar
        dpdxThreshold = thetaParams['dpdxThreshold'] # scalar
        theta = np.ones_like(DHat)
        
        ind1 = DHat < Dmin
        theta[ind1] = 0
        
        ind2 = (abs(dpdx) < dpdxThreshold) & (DHat >= Dmin) & (DHat <= Dmax)
        theta[ind2] = (Dmax - DHat[ind2]) / (Dmax - Dmin)
        
        ind3 = (abs(dpdx) < dpdxThreshold) & (DHat > Dmax)
        theta[ind3] = 0
        
        assert np.count_nonzero((theta >= 0) & (theta <= 1)) == np.size(theta), 'some theta is not between 0 and 1'
        return theta
    
        
class EWMA(object):
    """Class for handling the exponentially weighted moving average.  Each instance stores a previous iterate and the relaxation
    parameter for a single profile.
    """
    def __init__(self, EWMAParam):
        # define/initialize internal variables
        self.EWMAParam = EWMAParam
        
        # memory of previous relaxed iterates 
        self._yEWMA_lminus1 = None
    
    def next_ewma_iterate(self, y_l):
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
        yEWMA_l = self._compute_next_ewma(y_l, self._yEWMA_lminus1, self.EWMAParam)
        
        self._yEWMA_lminus1 = yEWMA_l
        return yEWMA_l
    
    @staticmethod
    def _compute_next_ewma(y_l, yEWMA_lminus1, EWMAParam):
        """Compute the next iterate of the exponentially weighted moving average (EWMA) of a quantity y.
        This can be done using a recursive formula that is a form of relaxation.
        
        yEWMA_l = EWMA_param*y_l  +  (1-EWMA_param) * yEWMA_l-1
        Note: EWMA_param = 1  means no relaxation (use current value), while 
              EWMA_param = .001 means a lot of relaxation (use very little of current value)
        """
        return EWMAParam * y_l  +  (1-EWMAParam) * yEWMA_lminus1
        
    def reset_ewma_iterate(self):
        self._yEWMA_lminus1 = None
    
    def set_ewma_iterate(self, yEWMA):
        self._yEWMA_lminus1 = yEWMA
        
class GridsNull(object):
    """Null class for moving between grids when the turbulence grid will be the same as the transport grid.
    No interpolation of quantities between grids will be performed, as there is only one grid.
    """
    def __init__(self, x):
        self.x = x
    def map_profile_onto_turb_grid(self, profile):
        return profile
    def map_transport_coeffs_onto_transport_grid(self, D, c):
        return (D, c)
    def get_x_transport_grid(self):
        return self.x
    def get_x_turbulence_grid(self):
        return self.x