from __future__ import division
import numpy as np
from tango import derivatives

def test_dx_centered_difference_edge_first_order():
    dx = 0.4
    u = np.array([2, 3, 7, 5, -3, 2, 8])
    dudx = derivatives.dx_centered_difference_edge_first_order(u, dx)
    obs = dudx
    exp = np.array([5/2, 25/4, 5/2, -25/2, -15/4, 55/4, 15])
    assert(np.allclose(obs, exp, rtol=0, atol=1e-13))
    
def test_dx_centered_difference():
    dx = 0.4
    u = np.array([2, 3, 7, 5, -3, 2, 8])
    dudx = derivatives.dx_centered_difference(u, dx)
    obs = dudx
    exp = np.array([5/2, 25/4, 5/2, -25/2, -15/4, 55/4, 15])
    exp = np.array([-5/4,   25/4, 5/2, -25/2, -15/4, 55/4, 65/4])
    assert(np.allclose(obs, exp, rtol=0, atol=1e-13))