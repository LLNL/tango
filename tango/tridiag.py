"""
tridiag

Solver for tridiagonal systems.

See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import scipy.linalg

def solve(A, B, C, D):
    """Solve a tridiagonal system Mu = D, where the matrix M is tridiagonal.
    [B0  A0                               ] [u0]     [D0]
    [C1  B1   A1                          ] [u1]     [D1]
    [    C2   B2  A2                      ] [u2]     [D2]
    [            . . .                    ]  .     =  .
    [               . . .                 ]  .        .
    [                   CN-1   BN-1   AN-1]  .        .
    [                          CN     BN  ] [uN]     [DN]
    
    The arrays A, and C must have the same length as B.  A[-1] must be 0 and C[0] must be 0.
      We can also write the matrix equation as
    C_j u_{j-1}  +  B_j u_j  +  A_j u_{j+1} = D_j      j = 0, ..., N
    
    Wrapper function for the actual solvers --- solve_python, solve_scipy
    
    Inputs:  1D arrays A, B, C, D, each of the same length.
    Output:  Solution u
    """
    return solve_with_scipy(A, B, C, D)

def solve_python(A, B, C, D):
    """Solve the tridiagonal system in solve() in pure Python using the Thomas Algorithm
    
    For details, see any standard methods textbooks, e.g., Jardin -- Computational
    Methods in Plasma Physics, Section 3.2.2.
    
    Inputs:  1D arrays A, B, C, D, each of the same length.
    Output:  Solution u
    """
    
    B = -B # algorithm in Jardin's Computational Plasma Physics textbook used negative B on the diagonal
    assert len(B)==len(A) and len(C)==len(A) and len(D)==len(A), 'lengths must be equal'
    assert A[-1]==0, 'last element of A must be zero'
    assert C[0]==0, 'first element of C must be zero'
    
    E = np.zeros_like(A)
    F = np.zeros_like(A)
    E[0] = A[0] / B[0]
    F[0] = -D[0] / B[0]
    for j in np.arange(1,len(A)):
        temp = B[j] - C[j]*E[j-1]
        E[j] = A[j] / temp
        F[j] = (C[j]*F[j-1] - D[j]) / temp
    
    u = np.zeros_like(A)
    u[-2] = (F[-2]*B[-1] - E[-2]*D[-1]) / (B[-1] - E[-2]*C[-1])
    u[-1] = -(D[-1] - C[-1]*u[-2]) / B[-1]
    for j in np.arange(len(A) - 3, -1, -1):
        u[j] = E[j]*u[j+1] + F[j]
    
    return u
    
    """ Sample code to test solve()
    N = 12
    B = -np.ones(N)
    A = 0.1 * np.ones(N)
    C = 0.1 * np.ones(N)
    A[-1] = 0
    C[0] = 0
    D = np.linspace(0, 0.22, N)
    u1 = solve(A, B, C, D)
    
    ## direct matrix solve
    # Construct the matrix
    M = np.diag(B) + np.diag(A[:-1], 1) + np.diag(C[1:], -1)
    u2 = np.linalg.solve(M, D)

    err = u1 - u2
    obs = err
    exp = 0
    tol = 1e-15
    assert np.linalg.norm(obs - exp) < tol
    
    """

def solve_with_scipy(A, B, C, D):
    """Solve the tridiagonal system in solve(), using scipy's banded matrix solver.
    
    Scipy calls a Fortran routine and so should be faster than pure python.
    """
    # construct the ab matrix needed by the scipy solver
    ab = np.array((np.roll(A, 1), B, np.roll(C, -1)))   # roll the A and C vectors so that the zero appears in the proper place for the scipy solver
    l_and_u = (1, 1)  # number of lower and upper diagonals
    u = scipy.linalg.solve_banded(l_and_u, ab, D)
    return u
