# test_tridiag

from __future__ import division
import numpy as np
from tango import tridiag

def test_solve():
    N = 12
    B = -np.ones(N)
    A = 0.1 * np.ones(N)
    C = 0.1 * np.ones(N)
    A[-1] = 0
    C[0] = 0
    D = np.linspace(0, 0.22, N)
    u1 = tridiag.solve(A, B, C, D)
    
    ## direct matrix solve with numpy
    M = np.diag(B) + np.diag(A[:-1], 1) + np.diag(C[1:], -1)
    u2 = np.linalg.solve(M, D)  # assume this is the correct answer
    
    err = u1 - u2
    obs = err
    exp = 0
    tol = 1e-14
    assert np.linalg.norm(obs - exp) < tol