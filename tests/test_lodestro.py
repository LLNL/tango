# test_lodestro

from __future__ import division
import numpy as np
from tango import lodestro_method

def test_compute_next_ewma():
    EWMA_param = 0.12
    yEWMA_lminus1 = 7.1
    y_l = 37
    yEWMA_l = lodestro_method.EWMA._compute_next_ewma(y_l, yEWMA_lminus1, EWMA_param)
    obs = yEWMA_l
    exp = 10.688
    assert(np.isclose(obs, exp, rtol=0, atol=1e-13))
    
def test_NextEWMAiterate():
    # assumes that in class EWMA, yEWMA gets initialized to the first y_l iterate -- that yEWMA1 = 1.2
    EWMA_param = 0.12
    ewma = lodestro_method.EWMA(EWMA_param)
    yEWMA1 = ewma.next_ewma_iterate(1.2)
    yEWMA2 = ewma.next_ewma_iterate(2.2)
    yEWMA3 = ewma.next_ewma_iterate(-0.3)
    yEWMA4 = ewma.next_ewma_iterate(3.4)
    obs = yEWMA4
    exp = 1.398528
    assert(np.isclose(obs, exp, rtol=0, atol=1e-13))
    
# need to do a LOT more testing...