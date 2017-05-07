"""test_JK_to_matrix_fd.py

Test the module to solve couple-ODEs with two fields.
"""

from __future__ import division
import numpy as np
from tango import JK_to_matrix_fd

def test_solver_one():
    """test J2, J3, J6, J7, J8, K2, K3, K6, K7, K8
    
    Use method of manufactured solutions to generate a problem with known analytic solution.
    Not checking order of accuracy of convergence as dx is decreased, only that we get an
    accurate solution.
    """
    # polar-like coordinate system, but calling it 'x'
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N - 0.5)  # spatial grid size
    x = np.linspace(dx/2, L, N)   # location corresponding to grid points j=0, ..., N-1
    
    DU = 1.6
    DW = 2.4
    cU = 0.3
    cW = 9
    nu = 1.3
    alpha = 1.9
    beta = 0.09
    
    UL = 0.75
    WL = 3.33
    
    
    J2 = DU * x
    K2 = DW * x
    J3 = -cU * x**2
    K3 = -cW * x**2
    J6 = x
    K6 = x**2
    J8 = nu * x
    K8 = -nu * x
    
    pi = np.pi
    sin = np.sin
    cos = np.cos
    J7 = (9*alpha*DU*x**2  +  2*cU*x*UL  +  cU*alpha*(2*x-5*x**4) 
        - x*UL  - alpha*(x-x**4)  -  nu*x*WL  -  nu*beta*x*cos(pi * x / 2) )
    
    K7 = (pi/2 * DW * beta * (sin(pi*x/2) + pi*x/2 * cos(pi*x/2)) + 2*cW*x*WL
        + cW*beta*(2*x*cos(pi*x/2) - pi*x**2/2 * sin(pi*x/2)) - x**2 * WL
        - beta*x**2 * cos(pi*x/2)  +  nu*x*UL + nu*alpha*(x-x**4) )
    
    dt = 1e9
    rightBC = (UL, WL)
    U_initial = np.zeros_like(x)
    W_initial = np.zeros_like(x)
    psi_mminus1 = (U_initial, W_initial)
    
    # irrelevant values but have to be specified
    J1 = np.ones_like(x)
    K1 = np.ones_like(x)
    
    (UEqnCoeffs, WEqnCoeffs) = JK_to_matrix_fd.JK_to_matrix(dt, dx, rightBC, psi_mminus1,
            J1=J1, J2=J2, J3=J3, J6=J6, J7=J7, J8=J8,
            K1=K1, K2=K2, K3=K3, K6=K6, K7=K7, K8=K8)
    
    # solve for numerical solutions
    (Uobs, Wobs) = JK_to_matrix_fd.solve(UEqnCoeffs, WEqnCoeffs)
    
    # exact solutions
    Uexp = UL + alpha * (1 - x**3)
    Wexp = WL + beta * cos(pi * x / 2)
    
    assert np.allclose(Uobs, Uexp, rtol=1e-4, atol=1e-4)
    assert np.allclose(Wobs, Wexp, rtol=1e-4, atol=1e-4)
    