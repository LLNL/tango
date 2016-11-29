"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import scipy.interpolate
import scipy

"""
module interfacegrids_gene

Module for handling the interfacing of Tango's radial grid to another radial
  grid.  For example, GENE performs computations on its own radial grid.  This
  radial grid may have a different domain than Tango, either larger or smaller,
  in which case to connect one grid to the other requires extrapolation or
  truncation.  Furthermore, the GENE radial grid might be finer than Tango's
  radial grid, in which case interfacing requires coarse graining or
  interpolation.
"""

class GridInterface_TangoOutside(object):
    """Class for interacing Tango's grid and GENE's grid where at the outer boundary, Tango's grid
      extends radially outward farther than GENE's.  At the inner bounadry, Tango's grid extends
      radially inward farther than GENE's.
    
    Mapping a profile from the transport grid onto GENE's grid: since Tango's domain is larger in
    both directions, we can use a simple interpolating spline
    
    Mapping transport coefficients from GENE's grid to Tango's grid: we first resample the transport
    coefficients on the part of Tango's grid that overlaps with GENE's grid, then add zeros where
    necessary.
    """
    def __init__(self, psi_tango, psi_gene):
        assert psi_tango[-1] >= psi_gene[-1]
        self.psi_tango = psi_tango  # values of flux coordinate psi on tango's grid
        self.psi_gene = psi_gene    # values of psi on gene's grid
    def MapProfileOntoTurbGrid(self, profile_tango):
        """Since Tango's domain is larger than GENE's in both directions, we can use a simple interpolating spline to 
        resample the profile on GENE's grid.
        """
        interpolate = scipy.interpolate.InterpolatedUnivariateSpline(self.psi_tango, profile_tango)
        profile_gene = interpolate(self.psi_gene)
        return profile_gene
    def MapTransportCoeffsOntoTransportGrid(self, D_genegrid, c_genegrid):
        D_tango = self.MapToTransportGrid(D_genegrid, enforcePositive=True)
        c_tango = self.MapToTransportGrid(c_genegrid)
        return (D_tango, c_tango)
    def MapToTransportGrid(self, f_gene, enforcePositive=False):
        """Map a quantity f (typically a transport coefficient) from GENE's grid to tango's grid.
        
        Here, tango's grid extends further than GENE's (which occurs at both the inner boundary and the
        outer boundary), so the transport coefficient is set to zero in the nonoverlapping region.

        GENE's turbulence goes to zero in its buffer zone at the boundaries of its domain.  Therefore, the
        transport coefficients returned by GENE should go to zero.
        
        Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
        Inputs:
          f_gene              f(psi) on the gene grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          f_tango             f(psi) on the Tango grid (array)
        """
        f_tango = ExtendWithZeros_BothSides(self.psi_gene, f_gene, self.psi_tango, enforcePositive=enforcePositive)
        return f_tango
    def get_x_transport_grid(self):
        return self.psi_tango
    def get_x_turbulence_grid(self):
        return self.psi_gene

class GridInterface_TangoInside(object):
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
    def __init__(self, psi_tango, psi_gene):
        assert psi_gene[-1] >= psi_tango[-1]
        self.psi_tango = psi_tango  # values of flux coordinate psi on tango's grid
        self.psi_gene = psi_gene    # values of psi on gene's grid
    def MapProfileOntoTurbGrid(self, profile_tango):
        profile_gene = TruncateOnLeft_ExtrapolateOnRight(self.psi_tango, profile_tango, self.psi_gene, enforcePositive=True)
        return profile_gene
    def MapTransportCoeffsOntoTransportGrid(self, D_genegrid, c_genegrid):
        D_tango = self.MapToTransportGrid(D_genegrid, enforcePositive=True)
        c_tango = self.MapToTransportGrid(c_genegrid)
        return (D_tango, c_tango)
    def MapToTransportGrid(self, f_gene, enforcePositive=False):
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
          f_gene              f(psi) on the gene grid (array)
          enforcePositive     (optional). If True, set any negative values to zero before returning (boolean)
        
        Outputs:
          f_tango             f(psi) on the Tango grid (array)
        """
        f_tango = ExtendWithZeros_LeftSide(self.psi_gene, f_gene, self.psi_tango, enforcePositive=enforcePositive)
        return f_tango
    def get_x_transport_grid(self):
        return self.psi_tango
    def get_x_turbulence_grid(self):
        return self.psi_gene
        
class GridInterface_TangoInside_fixedoutside(object):
    """Similar to GridInterface_TangoInside.  But instead of linearly extrapolating Tango's profile on the outward side,
    a *fixed* slope in the non-overlapping region is used.  In the region where GENE's radial domain exists but Tango's
    does not, the profile is fixed for all time (where a Dirichlet boundary condition for Tango is assumed)
    
    """
    def __init__(self, psi_tango, psi_gene, outwardslope_gene):
        self.psi_tango = psi_tango  # values of flux coordinate psi on tango's grid
        self.psi_gene = psi_gene    # values of psi on gene's grid
        self.outwardslope_gene = outwardslope_gene       # imposed value of dp/dpsi in gene's outer region
        assert outwardslope_gene < 0, "You probably meant to impose a negative, not positive, slope on the outward side."
    def MapProfileOntoTurbGrid(self, profile_tango):
        profile_gene = TruncateOnLeft_ExtrapolateOnRight(self.psi_tango, profile_tango, self.psi_gene, enforcePositive=True)
        return profile_gene
    def MapTransportCoeffsOntoTransportGrid(self, D_genegrid, c_genegrid):
        D_tango = self.MapToTransportGrid(D_genegrid, enforcePositive=True)
        c_tango = self.MapToTransportGrid(c_genegrid)
        return (D_tango, c_tango)
    def MapToTransportGrid(self, f_gene, enforcePositive=False):
        f_tango = ExtendWithZeros_LeftSide(self.psi_gene, f_gene, self.psi_tango, enforcePositive=enforcePositive)
        return f_tango
    def get_x_transport_grid(self):
        return self.psi_tango
    def get_x_turbulence_grid(self):
        return self.psi_gene
        

def ExtendWithZeros_BothSides(x_small, f_small, x_large, enforcePositive=False):
    """Extending a function to a larger domain, with zeros where it was not originally defined.
    
    The domain x_small should be fully contained within x_large.  That is, x_large extends farther outward
    on both sides of the domain.
    
    This function operates by resampling within the overlapping region x_small, and then extending with zeros.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      x_small               independent variable on the smaller domain (array)
      f_small               dependent variable on the smaller domain (array)
      x_large               independent variable on the larger domain (array)
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      f_large               dependent variable on the larger domain (array)
    """
    assert x_large[0] <= x_small[0] and x_large[-1] >= x_small[-1]
    # resample within the overlapping region
    f_large = np.zeros_like(x_large)  # initialize with zeros
    ind = np.where(x_large > x_small[0])
    indstart = ind[0][0]
    ind = np.where(x_large < x_small[-1])
    indfinal = ind[0][-1]
    x_large_temp = x_large[indstart : indfinal + 1]
    
    interpolate = scipy.interpolate.InterpolatedUnivariateSpline(x_small, f_small)
    f_large[indstart : indfinal+1] = interpolate(x_large_temp)    
    
    # extend with zeros -- automatically performed because f_large was initialized with zeros
    if enforcePositive == True:
        ind = f_large < 0
        f_large[ind] = 0  
        
    return f_large

def ExtendWithZeros_LeftSide(x_in, f_in, x_out, enforcePositive=False):
    """Extending a function to another domain, with zeros where it was not originally defined.
    
    The domains x_in and x_out should satsify x_out[0] < x_in[0] and x_out[-1] < x_in[0].  The output
    domain is "to the left" of the input domain.
    
    This function operates by resampling within the overlapping region, and then extending with zeros.
    
    Sometimes, interpolation might produce negative values when zero is the minimum for physical reasons.
        The diffusion coefficient is one example where one wants to maintain positivity.  In this case, one
        can optionally enforce positivity of the returned value by zeroing out negative values.
        
    Inputs:
      x_in                  independent variable on the input domain (array)
      f_in                  dependent variable on the input domain (array)
      x_out                 independent variable on the new domain (array)
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
        
    Outputs:
      f_out                 dependent variable on the new domain (array)
    """
    assert x_out[0] <= x_in[0] and x_out[-1] <= x_in[-1]
    f_out = np.zeros_like(x_out)  # initialize with zeros    
    # resample within the overlapping region
    ind = np.where(x_out > x_in[0])
    indstart = ind[0][0]
    x_out_temp = x_out[indstart:]
    
    interpolate = scipy.interpolate.InterpolatedUnivariateSpline(x_in, f_in)
    f_out[indstart:] = interpolate(x_out_temp)    
    
    # extend with zeros -- automatically performed because f_out was initialized with zeros
    if enforcePositive == True:
        ind = f_out < 0
        f_out[ind] = 0  
        
    return f_out
        
        
###################################################
#### Functions for extrapolation ####
        
def LeastSquaresSlope(x, y, x0, y0):
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
    
def Extrap1d_ConstrainedLinReg(x, y, x_eval, side='right', numPts=10):
    """Perform extrapolation using constrained linear regression on part of the data (x,y).  Use numPts
    from either the left or right side of the data (specified by input variable side) as the input data.
    The linear regression is constrained to pass through the final point (x0, y0) (rightmost point if
    side=='right', leftmost if side=='left').  Data MUST be sorted.
    
    Inputs:
      x                 independent variable on the smaller domain (array)
      y                 dependent variable on the smaller domain (array)
      x_eval            values of x at which to evaluate the linear regression model
      side              side of the data from which to perform linear regression ('left' or 'right')
      numPts            number of points to use in the linear regression (scalar)
    Outputs:
      y_eval            values of dependent variable in the linear regression model evaluated at x_eval (array)
    """
    assert side=='left' or side=='right'
    if side=='left':
        xside = x[:numPts]
        yside = y[:numPts]
        x0 = x[0]
        y0 = y[0]
    elif side=='right':
        xside = x[-numPts:]
        yside = y[-numPts:]
        x0 = x[-1]
        y0 = y[-1]
        
    a = LeastSquaresSlope(xside, yside, x0, y0)  # determine model (a, x0, y0)
    b = y0 - a*x0
    #y_eval = scipy.polyval([a,b], x_eval)
    y_eval = a*(x_eval - x0) + y0 # evaluate model on desired points
    return y_eval    
    
    
def MakeExtrapolator(x_small, f_small, side, numPts):
    """Create an extrapolator that uses cubic interpolation within the given domain x_small, and linear
    regression for extrapolation outside the given domain x_small.  Linear regression is based upon the
    Npts left- or right-most points.  This function does not use linear regression for both sides of the
    given data --- only one side.  Data MUST be sorted.
    
    Inputs:
      x_small           independent variable on the smaller domain (array)
      f_small           dependent variable on the smaller domain (array)
      side              side of the data from which to perform linear regression ('left' or 'right')
      numPts            number of points to use in the linear regression (scalar)
    Outputs:
      extrapolator      function that can be evaluated on a domain, like interpolators
    """
    def extrapolator(x_large):
        ip_interior = scipy.interpolate.InterpolatedUnivariateSpline(x_small, f_small, k=3) # cubic 
        
        # extrapolated points: linear regression
        f_large = Extrap1d_ConstrainedLinReg(x_small, f_small, x_large, side, numPts=numPts)
        
        # interpolated points in the interior using cubic interpolation
        ind = (x_large > x_small[0]) & (x_large < x_small[-1])
        f_large[ind] = ip_interior(x_large[ind])
    
        return f_large
    return extrapolator    

def TruncateOnLeft_ExtrapolateOnRight(x_in, f_in, x_out, numPts=10, enforcePositive=False):
    """Map f_in from a 1D domain x_in to another domain x_out.  To be used when x_out[0] > x_in[0]
      and x_out[-1] > x_in[-1].  On the left side of the domain, where x_out is contained within
      x_in, f_in is truncated.  On the right side of the domain, where x_out is not contained within
      x_in, f_in is extrapolated.
    
    The output, f_out, is defined on the x_out grid.  In the region of overlap, f_out is just f_in 
    resampled using cubic interpolation.  Outside the region of overlap, on the right boundary,
    f_out is determined by linear extrapolation of the last two points of f_in.
    
    Inputs:
      x_in                  independent variable on the input domain (array)
      f_in                  dependent variable on the input domain (array)
      x_out                 independent variable on the new domain (array)
      numPts
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
    
    Outputs:
      f_out                 dependent variable on the new domain (array)
    """
    assert x_out[0] >= x_in[0] and x_out[-1] >= x_in[-1]
    extrapolator = MakeExtrapolator(x_in, f_in, side='right', numPts=numPts)
    f_out = extrapolator(x_out)
    
    if enforcePositive == True:
        ind = f_out < 0
        f_out[ind] = 0
    
    return f_out

def MakeExtrapolator_fixedslope(x_small, f_small, outwardSlope):
    """Create an extrapolator that uses cubic interpolation within the given domain x_small, and an
    imposed linear fit with imposed slope outside the given domain x_small.  Data must be sorted.
    
    Inputs:
      x_small           independent variable on the smaller domain (array)
      f_small           dependent variable on the smaller domain (array)
      outwardSlope      imposed slope outside the domain x_small
    Outputs:
      extrapolator      function that can be evaluated on a domain, like interpolators
    """
    def extrapolator(x_large):
        f_large = np.zeros_like(x_large, dtype=np.float)
        # exterior region: left side
        ind_leftexterior = x_large < x_small[0]
        f_large[ind_leftexterior] = outwardSlope * (x_large[ind_leftexterior] - x_small[0]) + f_small[0]
        
        #exterior region: right side
        ind_rightexterior = x_large > x_small[-1]
        f_large[ind_rightexterior] = outwardSlope * (x_large[ind_rightexterior] - x_small[-1]) + f_small[-1]
        
        # interpolated points in the interior using cubic interpolation
        ip_interior = scipy.interpolate.InterpolatedUnivariateSpline(x_small, f_small, k=3) # cubic 
        ind_interior = (x_large >= x_small[0]) & (x_large <= x_small[-1])
        f_large[ind_interior] = ip_interior(x_large[ind_interior])
        return f_large
    return extrapolator   

def TruncateOnLeft_FixedSlopeOnRight(x_in, f_in, x_out, outwardSlope, enforcePositive=False):
    """Map f_in from a 1D domain x_in to another domain x_out.  To be used when x_out[0] > x_in[0]
      and x_out[-1] > x_in[-1].  On the left side of the domain, where x_out is contained within
      x_in, f_in is truncated.  On the right side of the domain, where x_out is not contained within
      x_in, f_out is set to a fixed profile --- linear in this case.
    
    The output, f_out, is defined on the x_out grid.  In the region of overlap, f_out is just f_in 
    resampled using cubic interpolation.  Outside the region of overlap, on the right boundary,
    f_out is determined by the last point of f_in, and an imposed slope.
    
    Inputs:
      x_in                  independent variable on the input domain (array)
      f_in                  dependent variable on the input domain (array)
      x_out                 independent variable on the new domain (array)
      outwardSlope          imposed value of the slope determining f_out on the right side
      enforcePositive       (optional) If True, set any negative values to zero before returning (boolean)
    
    Outputs:
      f_out                 dependent variable on the new domain (array)
    """
    assert x_out[0] >= x_in[0] and x_out[-1] >= x_in[-1]
    extrapolator = MakeExtrapolator_fixedslope(x_in, f_in, outwardSlope)
    f_out = extrapolator(x_out)
    
    if enforcePositive == True:
        ind = f_out < 0
        f_out[ind] = 0
    
    return f_out