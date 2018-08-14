"""
HToMatrixFD

Module for constructing and solving the linear matrix system corresponding to the iteration equation within
an implicit timestep advance of a transport equation.
  
This module assumes we are solving a 1D PDE.  It is designed to be used in the solution of transport equations.
For example, imagine we are solving an equation of the form [where d_t === partial_t]

    H_1 d_t U - d_x( H_2 d_x U + H_3 U + H_4) - H_6 U - H_7 = 0     [note, H_5 is not implemented in this module]

where U(x,t) is the dependent variable.  The H_i are coefficients, possibly depending on space, time, and on the
unknown solution U.  Thus the equation may be nonlinear.  To solve this transport equation, we assume we will
advance in time using implicit steps, leading to a nonlinear equation for the time-advanced U.  Introduce a time index m
and an iteration index l.  Iteration is required to solve the implicit equation for the advanced U_m in terms of the
previous time U_{m-1}.  For convenience, we suppress the time index m everywhere except when it is m-1.

Picard iteration is used to solve the implicit equation.  The iteration equation takes the form

  (1)   H_1^l d_t U^{l+1} - d_x( H_2^l d_x U^{l+1} + H_3^l U^{l+1} + H_4^l ) - H_6^l U^{l+1} - H_7^l = 0.
 
Equation (1) is the equation that this module helps to solve.  Here, d_t U^{l+1} is understood to mean a backward Euler step,

    d_t U^{l+1} = (U^{l+1} - U^{m-1, infinity}) / delta t

Because the H_i coefficients are evaluated at the previous iterate l, Equation (1) is linear in the unknown U^{l+1}.  The H_i
coefficients are specified by the user as arrays, not as a user-supplied function.  Hence the user computes the H_i externally
and supplies them to this module for each iteration.

The H_i are specified on a grid, e.g., H_1;j for j=0, 1, ..., N-1.  The U^{l+1} is also specified on this grid.

A note on boundary conditions:  Boundary conditions in this module are not completely general.  For now, the left boundary
is assumed to be a "no flux" boundary.  The right boundary is assumed to be a Dirichlet boundary, with U(L) = UL, where UL
is some given number.
 
Since Equation (1) does not have time as a parameter, it can be considered an ODE BVP.  Various methods can be used
for its solution.  This module applies a low-order finite difference scheme.  It is conservative with the flux terms
and it also uses an adaptive upwinding to preserve the M-Matrix property from the convective flux terms.  With a low-order
finite difference scheme, the matrix system is tridiagonal, which can be solved efficiently.

 Note: Corisca uses finite elements (linear or cubic).  A spectral method could also be considered;
 using a Chebyshev basis is probably going to be extremely accurate and fast.  E.g., dedalus could be used, or chebfun
 
References: Corsica Users' Manual, p39, p65.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
from . import tridiag

def H_to_matrix(dt, dx, UL, U_mminus1, H1, H2=None, H3=None, H4=None, H6=None, H7=None):
    """
    Given the H coefficients for Equation (1), use a low-order finite difference scheme to construct the tridiagonal
      matrix.  Adaptive upwinding is applied to the H3 term to preserve the M-Matrix property.  Flux terms are treated
      in a manner consistent with global conservation.
      
    Boundary conditions: no flux on the left.  Dirichlet boundary condition on the right, U(L) = UL
    
    Inputs:
     H1, H2, H3, H4, H6, H7    the H_i evaluated on the grid (array)
     U_mminus1                 the solution U_{m-1} at the previous timestep on the grid (array)
     dt                        timestep (scalar)
     dx                        grid spacing (scalar)
     UL                        boundary condition at right edge (scalar)
    
    Outputs:
     A, B, C, D                arrays to specify a matrix equation in tridiagonal form
     
    The tridiagonal matrix equation is specified as
    
                    C_j u_{j-1}  +  B_j u_j  +  A_j u_{j+1} = D_j      j = 0, ..., N-1

    Each coefficient H_i accumulates terms into the A, B, C, g arrays.  A separate function performs the accumulation
      for the different H_i coefficients.  
      
    Purely as a matter of convenience, these functions will return their contributions to the 4 arrays A, B, C, D in 
      a single 4xN array, rather than four arrays of length N.  This allows the convenient one-liner for incrementing:
              ABCg += _H1Contribution()
      rather than
              (Ac, Bc, Cc, gc) = _H1Contribution() 
              A += Ac, B += Bc, C += Cc, g += gc
    
    H1 is required.  H2, ..., H7 are optional.  The default value for the optional H's is 'None'.
    """
    N = len(U_mminus1)
    assert len(H1) == N
    ABCg = np.zeros((4, N))
    
    ABCg += _H1_contribution(H1, U_mminus1, dt)
    if H2 is not None:
        assert len(H2) == N
        ABCg += _H2_contribution(H2, dx)
    if H3 is not None:
        assert len(H3) == N
        if H2 is None:          
            H2 = np.zeros(N)    # initialize a default value for H2; needed for the adaptive upwinding
        ABCg += _H3_contribution(H2, H3, dx) # requires H2 for determining adaptive upwinding
    if H4 is not None:
        assert len(H4) == N
        ABCg += _H4_contribution(H4, dx)
    if H6 is not None:
        assert len(H6) ==  N
        ABCg += _H6_contribution(H6)
    if H7 is not None:
        assert len(H7) == N
        ABCg += _H7_contribution(H7)
    ABCg += _right_boundary_contribution(UL, N)
    
    # retrieve arrays from ABCg to return separately
    A = ABCg[0, :]
    B = ABCg[1, :]
    C = ABCg[2, :]
    g = ABCg[3, :]
    D = -g  # move to right hand side for "standard" tridiagonal format.
    return (A, B, C, D)

def solve(A, B, C, D):
    return tridiag.solve(A, B, C, D)
        
def H_to_matrix_and_solve(dt, dx, UL, U_mminus1, H1, H2=None, H3=None, H4=None, H6=None, H7=None):
    (A, B, C, D) = H_to_matrix(dt, dx, UL, U_mminus1, H1, H2, H3, H4, H6, H7)
    return solve(A, B, C, D)    

def _H1_contribution(H1, U_mminus1, dt):
    N = len(H1)
    ABCgContribution = np.zeros((4, N))
    ABCgContribution[1, :] = H1 / dt               # B_j += H1/dt
    ABCgContribution[3, :] = -H1 * U_mminus1 / dt  # g_j += -H1 U_mminus1 / dt
    ABCgContribution[:, N-1] = 0                   # right boundary: no contribution (Dirichlet BC)
    return ABCgContribution
    
def _H2_contribution(H2, dx):
    N = len(H2)
    ABCgContribution = np.zeros((4, N))
    
    H2Half = _return_average_on_half_integer_grid(H2)
    # interior points: j = 1, ..., N-2
    ABCgContribution[0, 1:-1] = -_get_interior_at_j_plus_half(H2Half) / dx**2      # A_j += -H2;j+1/2 / dx^2
    ABCgContribution[1, 1:-1] = ( _get_interior_at_j_minus_half(H2Half) + _get_interior_at_j_plus_half(H2Half) ) / dx**2   # B_j += (H2;j-1/2 + H2;j+1/2) / dx^2
    ABCgContribution[2, 1:-1] = -_get_interior_at_j_minus_half(H2Half) / dx**2     # C_j += -H2;j-1/2 / dx^2
    
    # left boundary: j = 0
    ABCgContribution[0, 0] = -_get_left_bndry_at_j_plus_half(H2Half) / dx**2   # A_0
    ABCgContribution[1, 0] = _get_left_bndry_at_j_plus_half(H2Half) / dx**2    # B_0
    
    return ABCgContribution

def _H3_contribution(H2, H3, dx):
    N = len(H3)
    ABCgContribution = np.zeros((4, N))
    
    H2Half = _return_average_on_half_integer_grid(H2)
    H3Half = _return_average_on_half_integer_grid(H3)
    betaHalf = _fbeta(H2Half, H3Half, dx)
    
    # interior points: j = 1, ..., N-2
    ABCgContribution[0, 1:-1] = -_get_interior_at_j_plus_half(H3Half) * (1 - _get_interior_at_j_plus_half(betaHalf)) / dx     # A_j += -H3;j+1/2 * (1-beta_j+1/2) / dx
    ABCgContribution[1, 1:-1] = (-_get_interior_at_j_plus_half(H3Half) * _get_interior_at_j_plus_half(betaHalf) + 
                                  _get_interior_at_j_minus_half(H3Half) * (1 - _get_interior_at_j_minus_half(betaHalf)) ) / dx # B_j += (-H3;j+1/2 beta_j+1/2  +  H3;j-1/2 (1 - beta_j-1/2) ) / dx
    ABCgContribution[2, 1:-1] = _get_interior_at_j_minus_half(H3Half) * _get_interior_at_j_minus_half(betaHalf) / dx          # C_j += H3;j-1/2 beta_j-1/2 / dx
    
    # left boundary: j = 0
    ABCgContribution[0, 0] = -_get_left_bndry_at_j_plus_half(H3Half) * (1 - _get_left_bndry_at_j_plus_half(betaHalf)) / dx          # A_0 += -H3;1/2 (1 - beta_1/2) / dx
    ABCgContribution[1, 0] = -_get_left_bndry_at_j_plus_half(H3Half) * _get_left_bndry_at_j_plus_half(betaHalf) / dx                # B_0 += -H3;1/2 beta_1/2 / dx
    return ABCgContribution

def _H4_contribution(H4, dx):
    N = len(H4)
    ABCgContribution = np.zeros((4, N))
    # interior points
    ABCgContribution[3, 1:-1] = -(H4[2:] - H4[:-2]) / (2 * dx)  # g_j += -(H4;j+1 - H4;j-1) / (2*dx)
    
    # left boundary: no flux from the left
    ABCgContribution[3, 0] = -(H4[0] + H4[1]) / (2 * dx)        # g_0 += -(H4;0 + H4;1) / (2*dx)
      # one could also conceivably write this at g_0 += -(H4;1 - H4;0) / (2*dx).  This is different from the previous line by the sign of H4;0.
      # but Neumann, no-flux boundary conditions have been assumed anyway, so H4;0 should be equal to zero.  If H4;0 is not 0 there may be problems.
    
    # right boundary: no contribution (Dirichlet BC)
    return ABCgContribution
    
def _H6_contribution(H6):
    N = len(H6)
    ABCgContribution = np.zeros((4, N))
    ABCgContribution[1, :] = -H6               # B_j += -H6;j
    
    ABCgContribution[:, N-1] = 0            # right boundary: no contribution (Dirichlet BC)
    return ABCgContribution
    
def _H7_contribution(H7):
    N = len(H7)
    ABCgContribution = np.zeros((4, N))
    ABCgContribution[3, :] = -H7           # g_j += -H7;j
    
    
    ABCgContribution[:, N-1] = 0            # right boundary: no contribution (Dirichlet BC)
    return ABCgContribution
    
def _right_boundary_contribution(UL, N):
    ABCgContribution = np.zeros((4, N))
    ABCgContribution[1, N-1] = 1               # B_N-1 = 1
    ABCgContribution[3, N-1] = -UL             # g_N-1 = -UL
    return ABCgContribution
    
# functions for dealing with conservative differencing for fluxes using half-integer grid points
def _return_average_on_half_integer_grid(u):
    """For an array u indexed at integer grid points, return an array corresponding to half-integer grid points, computed as the average of adjacent gridpoints.
    Input:  u                   array of length N, accessed by u[j] = u_j
    
    Output: uHalfIntGrid       array of length N
         Accessing: uHalfIntGrid[j] gives u_{j+1/2}, defined as (u_j + u_{j+1})/2
         Note: the final element, uHalfIntGrid[N-1], does not exist because uHalfIntGrid[N-2] -> u_{N-3/2} is the final element.
           Rather than make uHalfIntGrid a length N-1 vector, uHalfIntGrid[N-1] is unused and is left as zero
        uHalfIntGrid --->  [u_1/2  u_3/2 ... u_N-3/2 0]
    """
    uHalfIntGrid = np.zeros_like(u)
    uHalfIntGrid[:-1] = (u[:-1] + u[1:]) / 2
    return uHalfIntGrid
    
def _get_interior(u):
    """Return the interior points for an array u that is given on integer gridpoints
        Input: u            array of length N
        Output: u_interior  array of length N-2
    """
    uInterior = u[1:-1]
    return uInterior   

def _get_interior_at_j_plus_half(uHalfIntGrid):
    """At the interior points, return the j+1/2 array for u.
    Assume input array is length N, j=0, 1, ..., N-1.  The interior points are j=1, ..., N-2
    Assume x is given on an integer grid, and we want u_{j+1/2}, given from an array uHalfIntGrid given on a half-integer grid 
      To get all elements corresponding to interior points of x, access using x[1:-1]
      To get all elements corresponding to interior points of u_{j+1/2}, access using uHalfIntGrid[1:-1]
          --> for j=1, we want u_3/2.  This corresponds to uHalfIntGrid[1]
          --> for j=N-2, we want u_{N-3/2}.  This corresponds to uHalfIntGrid[-2]
        
    Inputs:
      uHalfIntGrid            array of length N.  uHalfIntGrid[0] --> u_1/2,  uHalfIntGrid[1] --> u_3/2, ..., uHalfIntGrid[-2] = u_{N-3/2}
    
    Outputs:
      uJPlusHalfInterior     array of length N-2.  uJPlusHalfInterior[0] --> u-->3/2, ..., uJPlusHalfInterior[-1] --> u_{N-3/2}
    """
    uJPlusHalfInterior = uHalfIntGrid[1:-1]
    return uJPlusHalfInterior
    
def _get_interior_at_j_minus_half(uHalfIntGrid):
    """At the interior points, return the j-1/2 array for u.
    Assume input array is length N, j=0, 1, ..., N-1.  The interior points are j=1, ..., N-2
    Assume x is given on an integer grid, and we want u_{j-1/2}, given from an array uHalfIntGrid given on a half-integer grid 
      To get all elements corresponding to interior points of x, access using x[1:-1]
      To get all elements corresponding to interior points of u_{j-1/2}, access using uHalfIntGrid[:-2]
          --> for j=1, we want u_1/2.  This corresponds to uHalfIntGrid[0]
          --> for j=N-2, we want u_{N-5/2}.  This corresponds to uHalfIntGrid[-3], which is the last element of uHalfIntGrid[:-2]
        
        Input:  uHalfIntGrid            array of length N.  uHalfIntGrid[0] --> u_1/2,  uHalfIntGrid[1] --> u_3/2, ..., uHalfIntGrid[-2] = u_{N-3/2}
    
        Output: uJMinusHalfInterior    array of length N-2.  uJMinusHalfInterior[0] --> u-->1/2, ..., uJMinusHalfInterior[-1] --> u_{N-5/2}
    """
    uJMinusHalfInterior = uHalfIntGrid[:-2]
    return uJMinusHalfInterior
    
def _get_left_bndry_at_j_plus_half(uHalfIntGrid):
    """At the left boundary j=0, get the u_j+1/2 term, or u_1/2
    """
    return uHalfIntGrid[0]
    
def _fbeta(H2Half, H3Half, dx):
    """Beta is a term that provides adaptive upwinding; i.e., controls the amount of upwinding of the convective flux term.  The adaptiveness
       arises to preserve the M-matrix property.  A centered difference of the derivative is more accurate; but an upwind difference is
       1) more stable, and 2) preserves the M-matrix property.  This function sets beta to use more "centered difference" and less "upwind"
       as long as the M-matrix property is preserved.
       
    This algorithm evaluates beta on the half-integer grid [but beta does *not* correspond to an average of values from adjacent
    integer gridpoints].  Each beta_j+1/2 is determined locally from H2_j+1/2 and H3_j+1/2.
    """
    gamma = 0.9
    cHalf = -H3Half    
    betaHalf = np.zeros_like(H3Half)
    betaStar = np.zeros_like(betaHalf)
    
    
    # positive c
    ind1 = cHalf > 0
    betaStar[ind1] = 1 - H2Half[ind1] / (cHalf[ind1] * dx)
    betaHalf[ind1 & (betaStar <= 1/2)] = 1/2
    betaHalf[ind1 & (betaStar > 1/2)] = gamma * betaStar[ind1 & (betaStar > 1/2)] + (1-gamma)
    
    # zero c
    ind2 = cHalf == 0  # c_hat must be 0 (meaning gammabar must be 0)
    betaHalf[ind2] = 0      # does not matter what this value is when c=0
    
    # negative c
    ind3 = cHalf < 0
    betaStar[ind3] = -H2Half[ind3] / (cHalf[ind3] * dx)
    betaHalf[ind3 & (betaStar >= 1/2)] = 1/2
    betaHalf[ind3 & (betaStar < 1/2)] = gamma * betaStar[ind3 & (betaStar < 1/2)]

    return betaHalf
