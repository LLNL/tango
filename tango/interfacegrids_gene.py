"""
interfacegrids_gene

Module for handling the interfacing of Tango's radial grid to another radial
grid.  For example, GENE performs computations on its own radial grid.  This
radial grid may have a different domain than Tango, either larger or smaller,
in which case to connect one grid to the other requires extrapolation or
truncation.  Furthermore, the GENE radial grid might be finer than Tango's
radial grid, in which case interfacing requires coarse graining or
interpolation.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
import scipy.interpolate
import scipy

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

class GridInterfaceTangoOutside(object):
    """Class for interacing Tango's grid and GENE's grid where at the outer boundary, Tango's grid
      extends radially outward farther than GENE's.  At the inner bounadry, Tango's grid extends
      radially inward farther than GENE's.
    
    Mapping a profile from the transport grid onto GENE's grid: since Tango's domain is larger in
    both directions, we can use a simple interpolating spline
    
    Mapping transport coefficients from GENE's grid to Tango's grid: we first resample the transport
    coefficients on the part of Tango's grid that overlaps with GENE's grid, then add zeros where
    necessary.
    """
    def __init__(self, psiTango, psiGene):
        assert psiTango[-1] >= psiGene[-1]
        self.psiTango = psiTango  # values of flux coordinate psi on tango's grid
        self.psiGene = psiGene    # values of psi on gene's grid
    def map_profile_onto_turb_grid(self, profileTango):
        """Since Tango's domain is larger than GENE's in both directions, we can use a simple interpolating spline to 
        resample the profile on GENE's grid.
        """
        interpolate = scipy.interpolate.InterpolatedUnivariateSpline(self.psiTango, profileTango)
        profileGene = interpolate(self.psiGene)
        return profileGene
    def map_transport_coeffs_onto_transport_grid(self, DGeneGrid, cGeneGrid):
        DTango = self.map_to_transport_grid(DGeneGrid, enforcePositive=True)
        cTango = self.map_to_transport_grid(cGeneGrid)
        return (DTango, cTango)
    def map_to_transport_grid(self, fGene, enforcePositive=False):
        """Map a quantity f (typically a transport coefficient) from GENE's grid to tango's grid.
        
        Here, tango's grid extends further than GENE's (which occurs at both the inner boundary and the
        outer boundary), so the transport coefficient is set to zero in the nonoverlapping region.

        GENE's turbulence goes to zero in its buffer zone at the boundaries of its domain.  Therefore, the
        transport coefficients returned by GENE should go to zero.
        
        Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
        Inputs:
          fGene               f(psi) on the gene grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          fTango              f(psi) on the Tango grid (array)
        """
        fTango = extend_with_zeros_both_sides(self.psiGene, fGene, self.psiTango, enforcePositive=enforcePositive)
        return fTango
    def get_x_transport_grid(self):
        return self.psiTango
    def get_x_turbulence_grid(self):
        return self.psiGene
        
        
class TangoOutsideExtrapCoeffs(object):
    """Class for interacing Tango's grid and GENE's grid where at the outer boundary, Tango's grid
      extends radially outward farther than GENE's.  At the inner bounadry, Tango's grid extends
      radially inward farther than GENE's.
      
    Similar to TangoOutside, but this class will extrapolate the transport coefficients on the outer
    region when mapping from the turbulence grid to the transport grid rather than just adding zeros.
    
    Mapping a profile from the transport grid onto GENE's grid: since Tango's domain is larger in
    both directions, we can use a simple interpolating spline
    
    Mapping transport coefficients from GENE's grid to Tango's grid: we first resample the transport
    coefficients on the part of Tango's grid that overlaps with GENE's grid, then extrapolate for the
    outer region.
    
    The point of this extrapolation is to eliminate the effects of the buffer zone.  Introduce the
    definitions:
        L                   size of domain and the right-hand boundary
        xTurbBnd            x coordinate of the end of the turbulence grid (should equal the end of the buffer zone)
        xRBufferBegin       x coordinate of the beginning of the right buffer
        xRBufferEnd         x coordinate of the end of the right buffer
        xEL                 x coordinate of the left side of the extrapolation zone
        xER                 x xoordinate of the right side of the extrapolation zone
    
    We will have xEL < xER < xRBufferBegin < xRBufferEnd == xTurbBnd < L.
    
    The turbulent flux (or rather here, we speak of transport coefficients) will nominally be zero for
    xTurbBnd < x < L.  But also, the transport coefficients cannot be trusted in the buffer zone either,
    i.e. xRBufferBegin < x < xTurnBnd.  [And probably, the transport coefficient cannot be trusted a
    little to the left of xRBufferBegin.]  So, the solution here will be to throw out that data and pretend
    that an extrapolation of the transport coefficient from data that can be trusted to these regions is
    sufficient.  We will use data from the region xEL <= x <= xER and extrapolate this to the region
    xER < x < L, REPLACING whatever the turbulent flux provided in the region XER < x < L previously.  The
    extrapolation here is done simply, using a low order polynomial.
    
    The purpose here is solely to generate an overall transport coefficient that is smooth, so the solution
    is smooth (and can be easily converged to numerically).  The physics cannot be trusted where the
    turbulent flux is inaccurate, so we are not concerned with getting an accurate solution in that region.
    Instead the concern is making sure we can get a smooth solution at all.
    
    
    """
    def __init__(self, xTango, xTurb, xExtrapZoneLeft, xExtrapZoneRight, polynomialDegree):
        assert xTango[-1] >= xTurb[-1]
        self.xTango = xTango  # values of coordinate x on tango's grid
        self.xTurb = xTurb    # values of x on turbulent grid
        self.xExtrapZoneLeft = xExtrapZoneLeft          # x coordinate of left side of the extrapolation zone
        self.xExtrapZoneRight = xExtrapZoneRight        # x coordinate of the right side of the extrapolation zone
        self.polynomialDegree = polynomialDegree        # degree of polynomial to use when extrapolating
    def map_profile_onto_turb_grid(self, profileTango):
        """Since Tango's domain is larger than GENE's in both directions, we can use a simple interpolating spline to 
        resample the profile on GENE's grid.
        """
        interpolate = scipy.interpolate.InterpolatedUnivariateSpline(self.xTango, profileTango)
        profileTurb = interpolate(self.xTurb)
        return profileTurb
    def map_transport_coeffs_onto_transport_grid(self, DTurbGrid, cTurbGrid):
        DTango = self.map_to_transport_grid(DTurbGrid, enforcePositive=True)
        cTango = self.map_to_transport_grid(cTurbGrid)
        return (DTango, cTango)
    def map_to_transport_grid(self, fTurb, enforcePositive=False):
        """Map a quantity f (typically a transport coefficient) from GENE's grid to tango's grid.
        
        Here, tango's grid extends further than GENE's (which occurs at both the inner boundary and the
        outer boundary), so the transport coefficient is set to zero in the nonoverlapping region on the
        left.  On the right, the transport coefficient is extrapolated using a polynomial

        GENE's turbulence goes to zero in its buffer zone at the boundaries of its domain.  Therefore, the
        transport coefficients returned by GENE should go to zero.
        
        Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
        Inputs:
          fTurb               f(x) on the turbulence grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          fTango              f(x) on the Tango grid (array)
        """
        fTango = zeropad_on_left_extrap_on_right(self.xTurb, fTurb, self.xTango,
                                                 self.xExtrapZoneLeft, self.xExtrapZoneRight,
                                                 self.polynomialDegree, enforcePositive=enforcePositive) 
        return fTango
    def get_x_transport_grid(self):
        return self.xTango
    def get_x_turbulence_grid(self):
        return self.xTurb        


class TangoOutsideExtrapCoeffsBothSides(object):
    """Almost identical to class TangoOutsideExtrapCoeffs.  But in that class, extrapolation of transport
    coefficients occurs only at the outer boundary.  Here, extrapolation of transport coefficients occurs
    at both inner and outer boundaries (assuming Tango's grid extends radially farther inward and outer
    than the turbulence grid).  See class TangoOutsideExtrapCoeffs for more details and explanation.
    """
    def __init__(self, xTango, xTurb, 
                 xInnerExtrapZoneLeft, xInnerExtrapZoneRight,
                 xOuterExtrapZoneLeft, xOuterExtrapZoneRight,
                 polynomialDegree):
        assert xTango[-1] >= xTurb[-1]
        self.xTango = xTango  # values of coordinate x on tango's grid
        self.xTurb = xTurb    # values of x on turbulent grid
        self.xInnerExtrapZoneLeft = xInnerExtrapZoneLeft          # x coordinate of left side of the extrapolation zone at inner boundary
        self.xInnerExtrapZoneRight = xInnerExtrapZoneRight        # x coordinate of the right side of the extrapolation zone at inner boundary
        self.xOuterExtrapZoneLeft = xOuterExtrapZoneLeft          # x coordinate of left side of the extrapolation zone at outer boundary
        self.xOuterExtrapZoneRight = xOuterExtrapZoneRight        # x coordinate of the right side of the extrapolation zone at outer boundary
        self.polynomialDegree = polynomialDegree        # degree of polynomial to use when extrapolating
    
    def map_profile_onto_turb_grid(self, profileTango):
        """Since Tango's domain is larger than GENE's in both directions, we can use a simple interpolating spline to 
        resample the profile on GENE's grid.
        """
        interpolate = scipy.interpolate.InterpolatedUnivariateSpline(self.xTango, profileTango)
        profileTurb = interpolate(self.xTurb)
        return profileTurb
    
    def map_transport_coeffs_onto_transport_grid(self, DTurbGrid, cTurbGrid):
        DTango = self.map_to_transport_grid(DTurbGrid, enforcePositive=True)
        cTango = self.map_to_transport_grid(cTurbGrid)
        return (DTango, cTango)
    
    def map_to_transport_grid(self, fTurb, enforcePositive=False):
        """Map a quantity f (typically a transport coefficient) from GENE's grid to tango's grid.
        
        Here, tango's grid extends further than GENE's (which occurs at both the inner boundary and the
        outer boundary).  On both the left and right, the transport coefficient is extrapolated using a
        polynomial.

        GENE's turbulence goes to zero in its buffer zone at the boundaries of its domain.  Therefore, the
        transport coefficients returned by GENE should go to zero.  These zeros are removed by the
        extrapolation.
        
        Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
        Inputs:
          fTurb               f(x) on the turbulence grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          fTango              f(x) on the Tango grid (array)
        """
        fTango = extrap_on_left_and_right(self.xTurb, fTurb, self.xTango,
                                          self.xInnerExtrapZoneLeft, self.xInnerExtrapZoneRight,
                                          self.xOuterExtrapZoneLeft, self.xOuterExtrapZoneRight,
                                          self.polynomialDegree, enforcePositive=enforcePositive) 
        return fTango
    
    def get_x_transport_grid(self):
        return self.xTango
    
    def get_x_turbulence_grid(self):
        return self.xTurb
        

def zeropad_on_left_extrap_on_right(xIn, fIn, xOut, xExtrapZoneLeft, xExtrapZoneRight, polynomialDegree, enforcePositive=False):
    """Extending a function to another domain.  Where the function is not originally defined, use
    zeros on the left and extrapolation on the right to provide new values.
    
    The domains xIn and xOut should satsify xOut[0] < xIn[0] and xOut[-1] > xIn[0].  The output
    domain extends farther than the input domain on both sides.
    
    This function operates by resampling within the overlapping region, and then extending.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      xIn                   independent variable on the input domain (array)
      fIn                   dependent variable on the input domain (array)
      xOut                  independent variable on the new domain (array)
      xExtrapZoneLeft       x coordinate of left side of extrapolation zone (scalar)
      xExtrapZoneRight      x coordinate of right side of extrapolation zone (scalar)
      polynomialDegree      degree of polynomial for extrapolation (integer)
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      fOut                  dependent variable on the new domain (array)
    """
    assert xOut[0] <= xIn[0] and xOut[-1] >= xIn[-1]
    fOut = np.zeros_like(xOut)  # initialize with zeros  
    
    # left side region ... zeros
    
    # =============== interpolation region: xIn[0] <= x <= xExtrapZoneRight ===============
    interpolatorInterior = scipy.interpolate.InterpolatedUnivariateSpline(xIn, fIn)
    ind1 = (xOut >= xIn[0]) & (xOut <= xExtrapZoneRight)
    fOut[ind1] = interpolatorInterior(xOut[ind1])
    
    # =============== extrapolation region xExtrapZoneLeft < x ===============
    # first, fit the polynomial model using data within xExtrapZoneLeft <= x <= xExtrapZoneRight
    ind2 = (xIn >= xExtrapZoneLeft) & (xIn <= xExtrapZoneRight)
    xPoly = xIn[ind2]
    fPoly = fIn[ind2]
    p = np.polyfit(xPoly, fPoly, polynomialDegree)
    
    # second, use the model to predict f in the region xExtrapZoneRight < x
    ind3 = xOut > xExtrapZoneRight 
    fOut[ind3] = np.polyval(p, xOut[ind3])
    
    if enforcePositive == True:
        ind = fOut < 0
        fOut[ind] = 0  
    return fOut        
        

def extrap_on_left_and_right(xIn, fIn, xOut,
                             xInnerExtrapZoneLeft, xInnerExtrapZoneRight,
                             xOuterExtrapZoneLeft, xOuterExtrapZoneRight,
                             polynomialDegree, enforcePositive=False):
    """Extending a function to another domain.  Where the function is not originally defined, use
    zeros on the left and extrapolation on the right to provide new values.
    
    The domains xIn and xOut should satsify xOut[0] < xIn[0] and xOut[-1] > xIn[0].  The output
    domain extends farther than the input domain on both sides.
    
    This function operates by resampling within the overlapping region, and then extending.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      xIn                       independent variable on the input domain (array)
      fIn                       dependent variable on the input domain (array)
      xOut                      independent variable on the new domain (array)
      xInnerExtrapZoneLeft      x coordinate of left side of extrapolation zone at inner bounadry (scalar)
      xInnerExtrapZoneRight     x coordinate of right side of extrapolation zone at inner boundary (scalar)
      xOuterExtrapZoneLeft      x coordinate of left side of extrapolation zone at outer boundary (scalar)
      xOuterExtrapZoneRight     x coordinate of right side of extrapolation zone at outer boundary (scalar)
      polynomialDegree          degree of polynomial for extrapolation (integer)
      enforcePositive           (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      fOut                  dependent variable on the new domain (array)
    """
    assert xOut[0] <= xIn[0] and xOut[-1] >= xIn[-1]
    fOut = np.zeros_like(xOut)  # initialize with zeros  
    
    
    # =============== interpolation region: xIn[0] <= x <= xOuterExtrapZoneRight ===============
    interpolatorInterior = scipy.interpolate.InterpolatedUnivariateSpline(xIn, fIn)
    ind1 = (xOut >= xIn[0]) & (xOut <= xOuterExtrapZoneRight)
    fOut[ind1] = interpolatorInterior(xOut[ind1])
    
    # ======================= inner extrapolation region x < xInnerExtrapZoneRight ======================
    #   first, fit the polynomial model using data with xInnerExtrapZoneLeft <= x <= xInnerExtrapZoneRight
    ind2a = (xIn >= xInnerExtrapZoneLeft) & (xIn <= xInnerExtrapZoneRight)
    xPoly = xIn[ind2a]
    fPoly = fIn[ind2a]
    p = np.polyfit(xPoly, fPoly, polynomialDegree)
    
    #   second, use the model to predict f in the region x < xInnerExtrapZoneLeft
    ind3a = xOut < xInnerExtrapZoneLeft
    fOut[ind3a] = np.polyval(p, xOut[ind3a])
    
    # ======================= outer extrapolation region xOuterExtrapZoneLeft < x ======================
    #   first, fit the polynomial model using data within xOuterExtrapZoneLeft <= x <= xOuterExtrapZoneRight
    ind2b = (xIn >= xOuterExtrapZoneLeft) & (xIn <= xOuterExtrapZoneRight)
    xPoly = xIn[ind2b]
    fPoly = fIn[ind2b]
    p = np.polyfit(xPoly, fPoly, polynomialDegree)
    
    #   second, use the model to predict f in the region xOuterExtrapZoneRight < x
    ind3b = xOut > xOuterExtrapZoneRight 
    fOut[ind3b] = np.polyval(p, xOut[ind3b])
    
    if enforcePositive == True:
        ind = fOut < 0
        fOut[ind] = 0  
    return fOut

        
class GridInterfaceTangoInside(object):
    """Class for interacing Tango's grid and GENE's grid where at the outer boundary, GENE's grid
      extends radially outward farther than GENE's.  At the inner bounadry, Tango's grid extends
      radially inward farther than GENE's.
    
    Mapping a profile from the transport grid onto GENE's grid: On the inward side where Tango's domain
    is larger, we can use a simple interpolating spline.  On the outward side, Tango's profile is
    extrapolated.
    
    Mapping transport coefficients from GENE's grid to Tango's grid: we first resample the transport
    coefficients on the part of Tango's grid that overlaps with GENE's grid, then add zeros where
    necessary.
    """
    def __init__(self, psiTango, psiGene):
        assert psiGene[-1] >= psiTango[-1]
        self.psiTango = psiTango  # values of flux coordinate psi on tango's grid
        self.psiGene = psiGene    # values of psi on gene's grid
    def map_profile_onto_turb_grid(self, profileTango):
        profileGene = truncate_on_left_extrapolate_on_right(self.psiTango, profileTango, self.psiGene, enforcePositive=True)
        return profileGene
    def map_transport_coeffs_onto_transport_grid(self, DGeneGrid, cGeneGrid):
        DTango = self.map_to_transport_grid(DGeneGrid, enforcePositive=True)
        cTango = self.map_to_transport_grid(cGeneGrid)
        return (DTango, cTango)
    def map_to_transport_grid(self, fGene, enforcePositive=False):
        """Map a quantity f (typically a transport coefficient) from GENE's grid to tango's grid.
        
        Here, tango's grid extends farther radially inward than GENE's at the inner boundary, so the
        transport coefficient is set to zero in the nonoverlapping region.  At the outer boundary, GENE's
        grid extends farther outward than Tango's, so the transport coefficients are simply truncated.

        GENE's turbulence goes to zero in its buffer zone at the boundaries of its domain.  Therefore, the
        transport coefficients returned by GENE should go to zero.
        
        Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
        Inputs:
          fGene               f(psi) on the gene grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          fTango             f(psi) on the Tango grid (array)
        """
        fTango = extend_with_zeros_left_side(self.psiGene, fGene, self.psiTango, enforcePositive=enforcePositive)
        return fTango
    def get_x_transport_grid(self):
        return self.psiTango
    def get_x_turbulence_grid(self):
        return self.psiGene
        
class GridInterfaceTangoInsideFixedOutside(object):
    """Similar to GridInterface_TangoInside.  But instead of linearly extrapolating Tango's profile on the outward side,
    a *fixed* slope in the non-overlapping region is used.  In the region where GENE's radial domain exists but Tango's
    does not, the profile is fixed for all time (where a Dirichlet boundary condition for Tango is assumed)
    
    """
    def __init__(self, psiTango, psiGene, outwardSlopeGene):
        self.psiTango = psiTango  # values of flux coordinate psi on tango's grid
        self.psiGene = psiGene    # values of psi on gene's grid
        self.outwardSlopeGene = outwardSlopeGene       # imposed value of dp/dpsi in gene's outer region
        assert outwardSlopeGene < 0, "You probably meant to impose a negative, not positive, slope on the outward side."
    def map_profile_onto_turb_grid(self, profileTango):
        profileGene = truncate_on_left_fixed_slope_on_right(self.psiTango, profileTango, self.psiGene, self.outwardSlopeGene, enforcePositive=True)
        return profileGene
    def map_transport_coeffs_onto_transport_grid(self, DGeneGrid, cGeneGrid):
        DTango = self.map_to_transport_grid(DGeneGrid, enforcePositive=True)
        cTango = self.map_to_transport_grid(cGeneGrid)
        return (DTango, cTango)
    def map_to_transport_grid(self, fGene, enforcePositive=False):
        fTango = extend_with_zeros_left_side(self.psiGene, fGene, self.psiTango, enforcePositive=enforcePositive)
        return fTango
    def get_x_transport_grid(self):
        return self.psiTango
    def get_x_turbulence_grid(self):
        return self.psiGene
        

def extend_with_zeros_both_sides(xSmall, fSmall, xLarge, enforcePositive=False):
    """Extending a function to a larger domain, with zeros where it was not originally defined.
    
    The domain xSmall should be fully contained within xLarge.  That is, xLarge extends farther outward
    on both sides of the domain.
    
    This function operates by resampling within the overlapping region xSmall, and then extending with zeros.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      xSmall                independent variable on the smaller domain (array)
      fSmall                dependent variable on the smaller domain (array)
      xLarge                independent variable on the larger domain (array)
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      fLarge                dependent variable on the larger domain (array)
    """
    assert xLarge[0] <= xSmall[0] and xLarge[-1] >= xSmall[-1]
    # resample within the overlapping region
    fLarge = np.zeros_like(xLarge)  # initialize with zeros
    ind = np.where(xLarge > xSmall[0])
    indstart = ind[0][0]
    ind = np.where(xLarge < xSmall[-1])
    indfinal = ind[0][-1]
    xLargeTemp = xLarge[indstart : indfinal + 1]
    
    interpolate = scipy.interpolate.InterpolatedUnivariateSpline(xSmall, fSmall)
    fLarge[indstart : indfinal+1] = interpolate(xLargeTemp)    
    
    # extend with zeros -- automatically performed because fLarge was initialized with zeros
    if enforcePositive == True:
        ind = fLarge < 0
        fLarge[ind] = 0  
        
    return fLarge

def extend_with_zeros_left_side(xIn, fIn, xOut, enforcePositive=False):
    """Extending a function to another domain, with zeros where it was not originally defined.
    
    The domains xIn and xOut should satsify xOut[0] < xIn[0] and xOut[-1] < xIn[0].  The output
    domain is "to the left" of the input domain.
    
    This function operates by resampling within the overlapping region, and then extending with zeros.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      xIn                   independent variable on the input domain (array)
      fIn                   dependent variable on the input domain (array)
      xOut                  independent variable on the new domain (array)
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      fOut                  dependent variable on the new domain (array)
    """
    assert xOut[0] <= xIn[0] and xOut[-1] <= xIn[-1]
    fOut = np.zeros_like(xOut)  # initialize with zeros    
    # resample within the overlapping region
    ind = np.where(xOut > xIn[0])
    indstart = ind[0][0]
    xOutTemp = xOut[indstart:]
    
    interpolate = scipy.interpolate.InterpolatedUnivariateSpline(xIn, fIn)
    fOut[indstart:] = interpolate(xOutTemp)    
    
    # extend with zeros -- automatically performed because fOut was initialized with zeros
    if enforcePositive == True:
        ind = fOut < 0
        fOut[ind] = 0  
        
    return fOut
        
        
###################################################
#### Functions for extrapolation ####
        
def least_squares_slope(x, y, x0, y0):
    """Compute the slope m of the line passing through the point (x0, y0) that gives the least squares error 
    fit to the data x,y

    Elementary calculus shows that        (x-x0) dot (y-y0)
                                     m  = -----------------
                                          (x-x0) dot (x-x0)
    
    Inputs:
      x, y          independent and dependent variables (1d arrays)
      x0, y0        point the line is constrained to pass through
    Outputs:
      m             slope that minimizes the least squares error
    """
    m = np.dot(x-x0, y-y0) / np.dot(x-x0, x-x0)
    return m        
    
def extrap1d_constrained_linear_regression(x, y, xEval, side='right', numPts=10):
    """Perform extrapolation using constrained linear regression on part of the data (x,y).  Use numPts
    from either the left or right side of the data (specified by input variable side) as the input data.
    The linear regression is constrained to pass through the final point (x0, y0) (rightmost point if
    side=='right', leftmost if side=='left').  Data MUST be sorted.
    
    Inputs:
      x                 independent variable on the smaller domain (array)
      y                 dependent variable on the smaller domain (array)
      xEval             values of x at which to evaluate the linear regression model
      side              side of the data from which to perform linear regression ('left' or 'right')
      numPts            number of points to use in the linear regression (scalar)
    Outputs:
      yEval             values of dependent variable in the linear regression model evaluated at x_eval (array)
    """
    assert side=='left' or side=='right'
    if side=='left':
        xSide = x[:numPts]
        ySide = y[:numPts]
        x0 = x[0]
        y0 = y[0]
    elif side=='right':
        xSide = x[-numPts:]
        ySide = y[-numPts:]
        x0 = x[-1]
        y0 = y[-1]
        
    a = least_squares_slope(xSide, ySide, x0, y0)  # determine model (a, x0, y0)
    b = y0 - a*x0
    #y_eval = scipy.polyval([a,b], x_eval)
    yEval = a*(xEval - x0) + y0 # evaluate model on desired points
    return yEval    
    
    
def make_extrapolator(xSmall, fSmall, side, numPts):
    """Create an extrapolator that uses cubic interpolation within the given domain xSmall, and linear
    regression for extrapolation outside the given domain xSmall.  Linear regression is based upon the
    Npts left- or right-most points.  This function does not use linear regression for both sides of the
    given data --- only one side.  Data MUST be sorted.
    
    Inputs:
      xSmall           independent variable on the smaller domain (array)
      fSmall           dependent variable on the smaller domain (array)
      side              side of the data from which to perform linear regression ('left' or 'right')
      numPts            number of points to use in the linear regression (scalar)
    Outputs:
      extrapolator      function that can be evaluated on a domain, like interpolators
    """
    def extrapolator(xLarge):
        interpolatorInterior = scipy.interpolate.InterpolatedUnivariateSpline(xSmall, fSmall, k=3) # cubic 
        
        # extrapolated points: linear regression
        fLarge = extrap1d_constrained_linear_regression(xSmall, fSmall, xLarge, side, numPts=numPts)
        
        # interpolated points in the interior using cubic interpolation
        ind = (xLarge > xSmall[0]) & (xLarge < xSmall[-1])
        fLarge[ind] = interpolatorInterior(xLarge[ind])
    
        return fLarge
    return extrapolator    

def truncate_on_left_extrapolate_on_right(xIn, fIn, xOut, numPts=10, enforcePositive=False):
    """Map fIn from a 1D domain xIn to another domain xOut.  To be used when xOut[0] > xIn[0]
      and xOut[-1] > xIn[-1].  On the left side of the domain, where xOut is contained within
      xIn, fIn is truncated.  On the right side of the domain, where xOut is not contained within
      xIn, fIn is extrapolated.
    
    The output, fOut, is defined on the xOut grid.  In the region of overlap, fOut is just fIn 
    resampled using cubic interpolation.  Outside the region of overlap, on the right boundary,
    fOut is determined by linear extrapolation of the last two points of fIn.
    
    Inputs:
      xIn                  independent variable on the input domain (array)
      fIn                  dependent variable on the input domain (array)
      xOut                 independent variable on the new domain (array)
      numPts
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
    
    Outputs:
      fOut                 dependent variable on the new domain (array)
    """
    assert xOut[0] >= xIn[0] and xOut[-1] >= xIn[-1]
    extrapolator = make_extrapolator(xIn, fIn, side='right', numPts=numPts)
    fOut = extrapolator(xOut)
    
    if enforcePositive == True:
        ind = fOut < 0
        fOut[ind] = 0
    
    return fOut

def make_extrapolator_fixed_slope(xSmall, fSmall, outwardSlope):
    """Create an extrapolator that uses cubic interpolation within the given domain xSmall, and an
    imposed linear fit with imposed slope outside the given domain xSmall.  Data must be sorted.
    
    Inputs:
      xSmall           independent variable on the smaller domain (array)
      fSmall           dependent variable on the smaller domain (array)
      outwardSlope      imposed slope outside the domain xSmall
    Outputs:
      extrapolator      function that can be evaluated on a domain, like interpolators
    """
    def extrapolator(xLarge):
        fLarge = np.zeros_like(xLarge, dtype=np.float)
        # exterior region: left side
        indLeftExterior = xLarge < xSmall[0]
        fLarge[indLeftExterior] = outwardSlope * (xLarge[indLeftExterior] - xSmall[0]) + fSmall[0]
        
        #exterior region: right side
        indRightExterior = xLarge > xSmall[-1]
        fLarge[indRightExterior] = outwardSlope * (xLarge[indRightExterior] - xSmall[-1]) + fSmall[-1]
        
        # interpolated points in the interior using cubic interpolation
        interpolatorInterior = scipy.interpolate.InterpolatedUnivariateSpline(xSmall, fSmall, k=3) # cubic 
        indInterior = (xLarge >= xSmall[0]) & (xLarge <= xSmall[-1])
        fLarge[indInterior] = interpolatorInterior(xLarge[indInterior])
        return fLarge
    return extrapolator   

def truncate_on_left_fixed_slope_on_right(xIn, fIn, xOut, outwardSlope, enforcePositive=False):
    """Map fIn from a 1D domain xIn to another domain xOut.  To be used when xOut[0] > xIn[0]
      and xOut[-1] > xIn[-1].  On the left side of the domain, where xOut is contained within
      xIn, fIn is truncated.  On the right side of the domain, where xOut is not contained within
      xIn, fOut is set to a fixed profile --- linear in this case.
    
    The output, fOut, is defined on the xOut grid.  In the region of overlap, fOut is just fIn 
    resampled using cubic interpolation.  Outside the region of overlap, on the right boundary,
    fOut is determined by the last point of fIn, and an imposed slope.
    
    Inputs:
      xIn                   independent variable on the input domain (array)
      fIn                   dependent variable on the input domain (array)
      xOut                  independent variable on the new domain (array)
      outwardSlope          imposed value of the slope determining fOut on the right side
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
    
    Outputs:
      fOut                 dependent variable on the new domain (array)
    """
    assert xOut[0] >= xIn[0] and xOut[-1] >= xIn[-1]
    extrapolator = make_extrapolator_fixed_slope(xIn, fIn, outwardSlope)
    fOut = extrapolator(xOut)
    
    if enforcePositive == True:
        ind = fOut < 0
        fOut[ind] = 0
    
    return fOut