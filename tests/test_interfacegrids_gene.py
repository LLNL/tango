# test_interfacegrids_gene

from __future__ import division
import numpy as np
import tango.interfacegrids_gene 
import scipy.interpolate

def test_ExtendWithZeros_BothSides_zeros():
    # Test that the extended function is actually zero in the extended region when
    #  using ExtendWithZeros_BothSides
    N = 50
    x_small = np.linspace(0, 2, N)
    f = lambda x: 1 - (x-1)**2
    f_small = f(x_small)
    N2 = 221
    x_large = np.linspace(-0.3, 2.2, N2)
    
    f_large = tango.interfacegrids_gene.ExtendWithZeros_BothSides(x_small, f_small, x_large, enforcePositive=True)
    
    indleft = x_large < x_small[0]
    indright = x_large > x_large[-1]
    assert np.all(f_large[indleft]==0) and np.all(f_large[indright]==0)
    
def test_ExtendWithZeros_BothSides_interior():
    # Test that the extended function, when resampled on the original domain, is very
    #  close to the original function on the original domain
    N = 50
    x_small = np.linspace(0, 2, N)
    f = lambda x: 1 - (x-1)**2
    f_small = f(x_small)
    N2 = 221
    x_large = np.linspace(-0.3, 2.2, N2)
    
    f_large = tango.interfacegrids_gene.ExtendWithZeros_BothSides(x_small, f_small, x_large, enforcePositive=True)
    ip = scipy.interpolate.InterpolatedUnivariateSpline(x_large, f_large)
    f_small_resampled = ip(x_small)
    # remove the first and last points because these can be disturbed by the interpolation
    f_check = f_small[1:-1]
    f_check_resampled = f_small_resampled[1:-1]
    assert np.allclose(f_check, f_check_resampled, atol=1e-4)
    
def test_ExtendWithZeros_LeftSide_zeros():
    # Test that the extended function is actually zero in the extended region.
    N = 50
    x_in = np.linspace(0, 2, N)
    f = lambda x: 1 - (x-1)**2
    f_in = f(x_in)
    N2 = 221
    x_out = np.linspace(-0.3, 1.7, N2)
    
    f_out = tango.interfacegrids_gene.ExtendWithZeros_LeftSide(x_in, f_in, x_out, enforcePositive=True)
    indleft = x_out < x_in[0]
    assert np.all(f_out[indleft]==0)

def test_ExtendWithZeros_LeftSide_interior():
    # Test that the extended function, when resampled on the original domain, is very
    #  close to the original function on the original domain
    N = 50
    x_in = np.linspace(0, 2, N)
    f = lambda x: 1 - (x-1)**2
    f_in = f(x_in)
    N2 = 221
    x_out = np.linspace(-0.3, 1.7, N2)
    
    f_out = tango.interfacegrids_gene.ExtendWithZeros_LeftSide(x_in, f_in, x_out, enforcePositive=True)
    ip = scipy.interpolate.InterpolatedUnivariateSpline(x_out, f_out)
    f_in_resampled = ip(x_in)
    # remove the first and last points because these can be disturbed by the interpolation
    f_check = f_in[1:-1]
    f_check_resampled = f_in_resampled[1:-1]
    assert np.allclose(f_check, f_check_resampled, atol=1e-4)
    
### test extrapolation methods
def test_LeastSquaresSlope():
    # test 1
    x = np.array([-1, 0, 1, 2])
    y = np.array((7, 5, 3, 1))
    x0 = 1
    y0 = 3
    obs = tango.interfacegrids_gene.LeastSquaresSlope(x, y, x0, y0)
    exp = -2
    assert np.isclose(obs, exp, rtol=0, atol=1e-13)
    
    # test 2
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    y = np.array([11.1, 9.7, 10.1, 8.1, 11, 8.5, 7.5])
    x0 = 6
    y0 = 7.5
    obs = tango.interfacegrids_gene.LeastSquaresSlope(x, y, x0, y0)
    exp = -0.58021978021978016
    assert np.isclose(obs, exp, rtol=0, atol=1e-13)
    
def test_Extrap1d_ConstrainedLinReg():
    x = np.array([3.0, 3.1, 3.5, 3.9, 4.1, 5, 7])
    y = np.array([0, 0, 0.1, 0.5, 1, 1.3, 2])
    x_eval = np.array([4.3, 5, 6.3, 7, 7.3, 8])
    y_eval = tango.interfacegrids_gene.Extrap1d_ConstrainedLinReg(x, y, x_eval, side='right', numPts=4)
    obs = y_eval
    exp = np.array([0.90259, 1.18710, 1.71549, 2.0, 2.12193, 2.40644])
    assert np.allclose(obs, exp, rtol=0, atol=1e-5)
    
def test_TruncateOnLeft_ExtrapolateOnRight():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y = np.array([5, 4.9, 4.75, 4.5, 4.2, 3.87, 3.6, 3.4, 3.25, 3.11, 2.97, 2.87])
    x_eval = np.linspace(3.5, 14, 10)
    y_eval = tango.interfacegrids_gene.TruncateOnLeft_ExtrapolateOnRight(x, y, x_eval, numPts=4)
    obs = y_eval
    exp = np.array([4.35546444, 3.9775479, 3.63970781, 3.4, 3.22681231, 3.06202911, 2.91182052, 2.78809524, 2.6447619, 2.50142857])
    assert np.allclose(obs, exp, rtol=0, atol=1e-6)

def test_TruncateOnLeft_FixedSlopeOnRight():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 4.5, 6.0])
    x_eval = np.array([1.5, 2.5, 3.5, 4.0, 4.5, 5.5])
    outwardSlope = -0.2
    y_eval = tango.interfacegrids_gene.TruncateOnLeft_FixedSlopeOnRight(x, y, x_eval, outwardSlope)
    obs = y_eval
    exp = np.array([2.09375, 3.78125, 5.21875,  6.,  5.9, 5.7])
    assert np.allclose(obs, exp, rtol=0, atol=1e-4)