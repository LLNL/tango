from __future__ import division
import numpy as np
from tango import smoother

def test_moving_average():
    x = np.array([  1.,   2.,   9.,   4.,  10.,   3.,   1.,   7.,   5.])
    windowSize = 5
    obs = smoother.moving_average(x, windowSize)
    exp = np.array([1., 4., 5.2, 5.6, 5.4, 5., 5.2, 13./3, 5])
    assert np.allclose(obs, exp, rtol=1e-14)