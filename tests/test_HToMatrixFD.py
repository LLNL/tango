"""test_HToMatrixFD.py

Simple diffusion equation with constant diffusion coefficient -- solve in "one iteration" with large timestep to steady state
  tests H1, H2, H3, and H7
"""

from __future__ import division
import numpy as np
from tango import HToMatrixFD

def test_diffusion_cartesian():
    # test H2 and H7
    dt = 1e5        # test only steady state
    
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    
    nL = 0.2
    # initial conditions
    n_initial = np.sin(np.pi * x) + nL * x
    
    # diffusion problem coefficients
    D = 0.8
    H1 = np.ones_like(x)
    H2 = D * np.ones_like(x)
    H7 = 1 - x**2
    
    # solve it for steady state: only one iteration required for linear problem
    n_final = HToMatrixFD.H_to_matrix_and_solve(dt, dx, nL, n_initial, H1=H1, H2=H2, H7=H7)
    
    # compare with analytic and plots
    n_ss_analytic = nL + 1/(12*D) * (1-x**2) * (5 - x**2)
    tol = 1e-4
    mean_square_error = np.linalg.norm(n_final - n_ss_analytic) / N
    obs = mean_square_error
    exp = 0
    assert abs(obs - exp) < tol
    
def test_diffusion_polar():
    # test H2 and H7 for spatially-dependent H2.  Diffusion in polar coordinates
    dt = 1e6
    
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dr = L / (N - 0.5)  # spatial grid size
    r = np.linspace(dr/2, L, N)   # location corresponding to grid points j=0, ..., N-1
    
    nL = 0.03
    # initial conditions
    n_initial = 0.3*np.sin(np.pi * r) + nL * r
    
    # diffusion problem coefficients
    D = 1.3
    H1 = r
    H2 = D*r
    H7 = r**2
    
    n_final = HToMatrixFD.H_to_matrix_and_solve(dt, dr, nL, n_initial, H1=H1, H2=H2, H7=H7 )
    
    # compare with analytic and plots
    n_ss_analytic = nL + 1/(9*D) * (1-r**3)
    tol = 1e-8 # for N=500    
    mean_square_error = np.linalg.norm(n_final - n_ss_analytic) / N
    obs = mean_square_error
    exp = 0
    assert abs(obs - exp) < tol
    
def test_diffusion_convection_time_polar():
    # test H1, H2, H3, H7 -- diffusion-convection problem in polar coordinates, including time dependence
    #   use a manufactured solution to generate a problem with analytic solution
    L = 1           # size of domain
    N = 300         # number of spatial grid points
    dr = L / (N - 0.5)  # spatial grid size
    r = np.linspace(dr/2, L, N)   # location corresponding to grid points j=0, ..., N-1
    
    nL = 0.5
    # initial conditions
    n_initial = nL * np.ones_like(r)
    n = n_initial

    # specify diffusion problem coefficients
    D = 1.3
    c0 = 0.7
    gamma = 0.9
    H1 = r
    H2 = D*r
    H3 = -c0 * r**2
    fH7 = lambda r, t: r/D * gamma * np.exp(-gamma*t) * (1-r**5) + \
                      25 * r**4 * (1 - np.exp(-gamma*t)) + \
                      c0 * (2*r*nL + 1/D * (1 - np.exp(-gamma*t)) * (2*r - 7*r**6))
    
    # set up time loop
    dt = 1e-2
    tf = 0.1
    tvec = np.arange(0, tf + 1e-10, dt)
    
    for m in range(1, len(tvec)):
        t = tvec[m]
        H7 = fH7(r, t)
        n = HToMatrixFD.H_to_matrix_and_solve(dt, dr, nL, n, H1=H1, H2=H2, H3=H3, H7=H7 )
        
    n_final = n
    # compare with analytic and plots
    #n_ss_analytic = nL + 1/D * (1-r**5)
    n_timedep_analytic = nL + 1/D * (1 - np.exp(-gamma*t)) * (1 - r**5)

    tol = 2e-5 # for N=300    
    mean_square_timedep_error = np.linalg.norm(n_final - n_timedep_analytic) / N
    obs = mean_square_timedep_error
    exp = 0
    assert abs(obs - exp) < tol
    
def test_H4():
    # test H4 in a problem with an analytic solution
    dt = 1e9        # test only steady state
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    
    UL = 2.2
    U_initial = np.sin(np.pi * x) + UL * x
    
    H1 = np.ones_like(x)
    H2 = np.ones_like(x)
    H4 = -np.sin(x)
    H7 = 1 - x**2
    
    U_final = HToMatrixFD.H_to_matrix_and_solve(dt, dx, UL, U_initial, H1=H1, H2=H2, H4=H4, H7=H7)
    U_ss_analytic = UL - (np.cos(x) - np.cos(L)) - (x**2 - L**2) / 2 + (x**4 - L**4) / 12
    
    assert np.allclose(U_final, U_ss_analytic, rtol=1e-3, atol=1e-3)