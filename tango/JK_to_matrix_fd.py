"""
JK_to_matrix_fd

Generalization of HToMatrixFD for a slightly more general situation of two fields coupled together in a specific way.

Module for constructing and solving the linear system corresponding to the iteration equation within an implicit
timestep advance of a transport equation.  In HToMatrixFD, we assumed we are solving an equation of the form

  (1)   H_1^l d_t U^{l+1} - d_x( H_2^l d_x U^{l+1} + H_3^l U^{l+1} + H_4^l ) - H_6^l U^{l+1} - H_7^l = 0.
 
If multiple fields were present, they were not coupled together in the implicit update; instead, any fields other than the
current one were evaluated at the previous iteration l and the dependence occurred in the H_i coefficients.

Now, we allow for an iteration update for two coupled fields of the form

  (2)  J_1^l d_t U^{l+1} - d_x( J_2^l d_x U^{l+1} + J_3^l U^{l+1} + J_4^l ) - J_6^l U^{l+1} - J_7^l - J8^l W^{l+1} = 0
       K_1^l d_t W^{l+1} - d_x( K_2^l d_x W^{l+1} + K_3^l W^{l+1} + K_4^l ) - K_6^l W^{l+1} - K_7^l - K8^l U^{l+1} = 0.
  
The pair of fields {U, W} will be collectively denoted as the state vector psi.

The coupling between the fields here is quite specific, and motivated by the collisional coupling between ion and electron
temperatures, which may be on a faster timescale than other processes.  Hence the ion and electron temperature (or pressure)
may need to be updated together implicitly to achieve stable iteration.  This does not allow other, more general forms of
coupling between the fields, such as coupling within the conservative flux terms.  This type of coupling is also used in
CORSICA.
    
Because the J_i and K_i coefficients are evaluted at the previous iterate l, Equation (2) is linear in the unknown U^{l+1}, 
W^{l+1}.  The J_i and K_i coefficients are specified by the user as arrays, not as a user-supplied function.  Hence the user
computes the J_i and K_i externally and supplies them to this module for each iteration.

The J_i and K_i are specified on a grid, e.g., J_1;j for j=0, 1, ..., N-1.  The U^{l+1}, W^{l+1} is also specified on this grid.

A note on boundary conditions:  Boundary conditions in this module are not completely general.  For now, the left boundary
is assumed to be a "no flux" boundary.  The right boundary is assumed to be a Dirichlet boundary, with U(L) = UL and W(L) = WL,
where UL and WL are some given numbers.
 
References: Corsica Users' Manual, p39, p65.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division, absolute_import
import numpy as np
import scipy.linalg
from collections import namedtuple

from . import HToMatrixFD

MatEqnCoeffs = namedtuple('MatEqnCoeffs',
               ('A', 'B', 'C', 'D', 'G'))

def JK_to_matrix(dt, dx, rightBC, psi_mminus1,
        J1=None, J2=None, J3=None, J4=None, J6=None, J7=None, J8=None,
        K1=None, K2=None, K3=None, K4=None, K6=None, K7=None, K8=None):
    """
    Given the J,K coefficients for Equation (2), use a low-order finite difference scheme to construct the discretized
      equation.  The results is a banded matrix, with 5 nonzero bands: 2 upper diagonals, the main diagonal, and 2 lower
      diagonals.
      
    Much of the infrastructure from HToMatrixFD is reused.  See there for more details.  The new pieces this module
    computes relate to the coupling between the fields U,W, but the within-field coefficients are handled by reusing
    HToMatrixFD.
      
    Boundary conditions: no flux on the left.  Dirichlet boundary condition on the right, U(L) = UL, W(L) = WL
    
    Inputs:
     dt                             timestep (scalar)
     dx                             grid spacing (scalar)
     rightBC                        boundary condition at right edge (UL, WL) (tuple of 2 scalars)
     psi_mminus1                    the solution (U_{m-1}, W_{m-1}) at the previous timestep on the grid (tuple of 2 arrays)
     J1, J2, J3, J4, J6, J7, J8     the J_i evaluated on the grid (array)
     K1, K2, K3, K4, K6, K7, K8     the K_i evaluated on the grid (array)
    
    Outputs:
     A, B, C, G, D                  arrays to specify the matrix equation in tridiagonal form
                                      each is a tuple of 2 length-N arrays.  A = (AU, AW), etc.
     
    The discretized equations are specified as
    
                C_j^U U_{j-1}  +  B_j^U U_j  +  A_j^U U_{j+1} + G_j^W W_j = D_j^U      j = 0, ..., N-1
                
                C_j^W W_{j-1}  +  B_j^W W_j  +  A_j^W W_{j+1} + G_j^U U_j = D_j^W      j = 0, ..., N-1
                
    Each coefficient J_i, K_i accumulates terms into the A, B, C, G, D arrays.  A separate function performs the accumulation
      for the different coefficients.  
    
    J1, K1 are required.  The rest are optional.  The default value for the optional J's, K's is 'None'.
    """
    U_mminus1, W_mminus1 = psi_mminus1[0], psi_mminus1[1]
    UL, WL = rightBC[0], rightBC[1]
    N = len(U_mminus1)
    assert len(W_mminus1) == N
    assert J1 is not None, 'Must supply J1'
    assert K1 is not None, 'Must supply K1'
    
    # map the J_i to A^U, B^U, C^U, g^U coefficients, i = 1, ..., 7
    AU, BU, CU, DU = HToMatrixFD.H_to_matrix(dt, dx, UL, U_mminus1,
                         H1=J1, H2=J2, H3=J3, H4=J4, H6=J6, H7=J7)
    
    # map the K_i to A^W, B^W, C^W, g^W coefficients, i = 1, ..., 7
    AW, BW, CW, DW = HToMatrixFD.H_to_matrix(dt, dx, WL, W_mminus1,
                         H1=K1, H2=K2, H3=K3, H4=K4, H6=K6, H7=K7)
    
    # coupling term: map J_8 to G^W and K_8 to G^U 
    GW = _H8_contribution(J8)
    GU = _H8_contribution(K8)
    
    UEqnCoeffs = MatEqnCoeffs(A=AU, B=BU, C=CU, D=DU, G=GW) # GW is not a typo here
    WEqnCoeffs = MatEqnCoeffs(A=AW, B=BW, C=CW, D=DW, G=GU)
    return (UEqnCoeffs, WEqnCoeffs)

        
def _H8_contribution(H8):
    """Map H8 to G contribution (coupling term between fields)
    Inputs:
      H8                (1D array)
    Outputs:
      GContribution     (1D array)
    """
    N = len(H8)
    GContribution = np.zeros(N)
    GContribution[:N-1] = -H8[:N-1]         # G_j += -H8;j
    GContribution[N-1] = 0                  # right boundary: no contribution (Dirichlet BC)
    return GContribution
    
def solve(UEqnCoeffs, WEqnCoeffs):
    """Solve the 2-field coupled system.
    
    Use scipy's banded matrix solver.
    
    Inputs:
      UEqnCoeffs        The AU, BU, CU, DU, GW coefficients (namedtuple)
      WEqnCoeffs        The AW, BW, CW, DW, GU coefficients (namedtuple)
    Outputs:
      U                 Solution U (1D array)
      W                 Solution W (1D array)
    """
    AU, BU, CU, DU, GW = UEqnCoeffs.A, UEqnCoeffs.B, UEqnCoeffs.C, UEqnCoeffs.D, UEqnCoeffs.G
    AW, BW, CW, DW, GU = WEqnCoeffs.A, WEqnCoeffs.B, WEqnCoeffs.C, WEqnCoeffs.D, WEqnCoeffs.G
    # the way HToMatrix works, the A and C arrays are length N, with
    # A[-1] = 0, and C[0] = 0, but these elements do not really exist
    
    # construct the diagonals and RHS
    # np.roll is necessary to move the zeros in the proper place for scipy's matrix solver.
    secondUpperDiagonal = np.roll(_interleave(AU, AW), 2)
    firstUpperDiagonal = np.roll(_interleave(GW, np.zeros_like(GW)), 1)
    mainDiagonal = _interleave(BU, BW)
    firstLowerDiagonal = _interleave(GU, np.zeros_like(GU))
    secondLowerDiagonal = np.roll(_interleave(CU, CW), -2)
    
    # construct input matrix in the form for scipy's matrix solver
    ab = np.array((secondUpperDiagonal, firstUpperDiagonal, mainDiagonal, firstLowerDiagonal, secondLowerDiagonal))
    # construct the RHS
    D = _interleave(DU, DW)
    
    # solve it
    l_and_u = (2, 2)
    psi = scipy.linalg.solve_banded(l_and_u, ab, D)
    U, W = _deinterleave(psi)
    return U, W
    
def _interleave(a, b):
    """Interlave two 1D arrays of the same length.
    
    If the inputs are a = [a0, a1, ..., a_N-1] and b = [b0, b1, ..., b_N-1], then the output is
    c = [a0, b0, a1, b1, ..., a_N-1, b_N-1]
    
    Inputs:
      a         first input array (1D array)
      b         first input array (1D array)
    Outputs:
      c         interleaved output data (1D array)
    """
    N = len(a)
    assert len(b) == N  # eh?
    c = np.zeros(2*N)
    c[0::2] = a
    c[1::2] = b
    return c

def _deinterleave(c):
    """Perform the opposite operation from _interleave()
    """
    a = c[0::2]
    b = c[1::2]
    return a, b