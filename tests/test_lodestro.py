# test_lodestro

from __future__ import division
import numpy as np
from tango import lodestro_method

def test_dxCenteredDifference():
    dx = 0.4
    u = np.array([2, 3, 7, 5, -3, 2, 8])
    dudx = lodestro_method.FluxSplit._dxCenteredDifference(u, dx)
    obs = dudx
    exp = np.array([5/2, 25/4, 5/2, -25/2, -15/4, 55/4, 15])
    assert(np.allclose(obs, exp, rtol=0, atol=1e-13))

def test_ComputeNextEWMA():
    EWMA_param = 0.12
    yEWMA_lminus1 = 7.1
    y_l = 37
    yEWMA_l = lodestro_method.EWMA._ComputeNextEWMA(y_l, yEWMA_lminus1, EWMA_param)
    obs = yEWMA_l
    exp = 10.688
    assert(np.isclose(obs, exp, rtol=0, atol=1e-13))
    
def test_NextEWMAiterate():
    # assumes that in class EWMA, yEWMA gets initialized to the first y_l iterate -- that yEWMA1 = 1.2
    EWMA_param = 0.12
    ewma = lodestro_method.EWMA(EWMA_param)
    yEWMA1 = ewma.NextEWMAIterate(1.2)
    yEWMA2 = ewma.NextEWMAIterate(2.2)
    yEWMA3 = ewma.NextEWMAIterate(-0.3)
    yEWMA4 = ewma.NextEWMAIterate(3.4)
    obs = yEWMA4
    exp = 1.398528
    assert(np.isclose(obs, exp, rtol=0, atol=1e-13))