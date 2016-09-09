# Tango is a transport equation solver intended for coupling with codes that calculate turbulent fluxes.
# Copyright (C) 2016 Lawrence Livermore National Security, LLC
# Authored by Jeff Parker 

# This file is part of Tango.
#
# Tango is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tango is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Tango.  If not, see <http://www.gnu.org/licenses/>

from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from collections import namedtuple
import time # for tic/toc functionality

# reload(transport_solvery_py)  # use this to reload interactively, NOT import transport_solver_py

def mainsolve():
    """Test problem of nonlinear diffusion in Shestakov et al. (2003)
    Trying out algorithm to split flux into diffusive and convective piece.
    Diffision coefficient is given by
        D = [(dn/dx) / n]^2
    And the source term is given by
        S = {S0     for x < delta
             0      otherwise }
          with S0=1, delta=0.1
    
    Domain:  0 <= x <= 1
    Boundary conditions:
      Neumann conditions at x=0:  dn/dx(0) = 0
      Dirichlet conditions at x=1: n(1) = * (try various values)
    Initial conditions:
        n0(x) = 1, or n0(x) = 1 - 0.5x
    """
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    
    Na = 100        # averaging window for iteration scheme
    lmax = 3000     # max number of iterations
    dt = 1e-5       # timestep
    
    Flag_CalculateFluxWithAvgN = False
        # if true, use nbar to calculate flux Gamma: Gamma[nbar] rather than Gamma[n]
    thetaparams = paramstruct()
    betaparams = paramstruct()
    thetaparams.Dmin = 1e-5
    thetaparams.Dmax = 1e13
    thetaparams.dndx_thresh = 10
    betaparams.gamma = 0.8
    
    # set up arrays, or at least indicate their length
    n = np.ones_like(x)
    n[-1] = 0.1     # initial condition
    # n = 1 - 0.5*x
    
    Gamma_ml = np.zeros_like(x)
    Gammabar_ml = np.zeros_like(x)
    nbar = np.zeros_like(x)
    Gammabar_lminus1 = np.zeros_like(x)
    nbar_lminus1 = np.zeros_like(x)     # it's okay this is left as zeros initially
    GammaInitialNa = np.zeros((Na, N))    # history Gamma^{m,k} for first Na iterations at a given timestep m
    nInitialNa = np.zeros_like(GammaInitialNa)        # history n^{m,k} for first Na iterations, 1 <= k <= Na
    f = np.zeros_like(x)        # RHS of matrix equation
    
    tol = 1e-8      # tol for convergence... reached when a certain error < tol
    errhistory = np.zeros(lmax)      # error history vs. iteration at a giventimetsp
    
    tf = dt
    t = np.arange(0, tf + 1e-10, dt)    # add 1e-10 so that tf is the last element
    
    # alternate way: manually specify a few times
    # t = np.array([0, 1e-5, 1e-4, 3e-4, 6e-4, 9e-4, 1e-3, 2e-3, 4e-3, 1e-2, 1e-1, 1e0, 1e3])
    
    thetacount = 0     # diagnostic counter to see the number of times that theta is not 1
    
    # initialize "m minus 1" variables for the first timestep
    nbar_mminus1 = n
    n_mminus1 = n
    Gammabar_mminus1 = GetFlux(nbar_mminus1, dx)
    
    
    tic()
    for m in range(1, len(t)):
        # Time advance: iterate to solve the nonlinear equation!
        # dt = t[m] - t[m-1]   # use this if using non-constant timesteps
        GammaInitialNa[:] = 0       # reset elements to zero for new timestep
        nInitialNa[:] = 0
        converged = False
        
        # compute matrix for first iterate.  Use converged Gammabar from previous timestep (m-1) as Gammabar^{m,0}
        Gammabar_m0 = Gammabar_mminus1
        nbar_m0 = nbar_mminus1
        n_m0 = n_mminus1
        
        # Calculate Dhat, chat, theta, beta, D, c for first iterate
        if Flag_CalculateFluxWithAvgN == True:
            (Dhat, chat, theta, beta, D, c) = GetCoeffs(Gammabar_m0, nbar_m0, dx, thetaparams, betaparams)
        else:
            (Dhat, chat, theta, beta, D, c) = GetCoeffs(Gammabar_m0, n_m0, dx, thetaparams, betaparams)
        
        S = GetS(x)
        (A, B, C, f) = GetTridiagMatAndRHS(D, c, beta, n_mminus1, S, dt, dx)
        
        l=1
        while not converged:
            # 1. Solve the tridiagonal matrix for n^{m,l} with matrix M from previous l
            n = solve_tridiag(A, B, C, f)
            if l <= Na:
                nInitialNa[l-1,] = n
            
            # 2. Compute Gamma^{m,l}
            if Flag_CalculateFluxWithAvgN == True:
                nbar = GetAvgQty(n, nbar_lminus1, nInitialNa, nbar_mminus1, l, Na)
                Gamma_ml = GetFlux(nbar, dx)
            else:
                Gamma_ml = GetFlux(n, dx)
                
            if l <= Na:
                GammaInitialNa[l-1,] = Gamma_ml
            
            # 3. Compute the weighted average Gammabar^{m,l}, and Dhat, chat, theta, beta, D, c
            Gammabar_ml = GetAvgQty(Gamma_ml, Gammabar_lminus1, GammaInitialNa, Gammabar_mminus1, l, Na)
            
            if Flag_CalculateFluxWithAvgN == True:
                (Dhat, chat, theta, beta, D, c) = GetCoeffs(Gammabar_ml, nbar, dx, thetaparams, betaparams)
            else:
                (Dhat, chat, theta, beta, D, c) = GetCoeffs(Gammabar_ml, n, dx, thetaparams, betaparams)
            
            # 4. With D, c, beta, construct the tridiagonal matrix M^{m,l}.  Also construct the source S and RHS term f
            S = GetS(x)
            (A, B, C, f) = GetTridiagMatAndRHS(D, c, beta, n_mminus1, S, dt, dx)
            
            # 5. Check for convergence and for too many iterations
            #   convergence check: is || M(m,l) n(m,l) - f(m,l) || < tol
            #    could add more convergence checks here
            resid = A*np.concatenate((n[1:], np.zeros(1))) + B*n + C*np.concatenate((np.zeros(1), n[:-1])) - f
            err = np.linalg.norm(resid) / len(resid)    # average error of each point.  Divide by N to get rid of scaling with N
            errhistory[l-1] = err
            if err < tol:
                converged = True
            
            # about to loop to next iteration l
            l += 1
            if l >= lmax:
                raise RuntimeError('Too many iterations on timestep %d.  Error is %f.' % (m, err))
            Gammabar_lminus1 = Gammabar_ml
            nbar_lminus1 = nbar
            
            # Check for NaNs or infs
            if np.all(np.isfinite(n)) == False:
                raise RuntimeError('NaN or Inf detected at l=%d.  Exiting...' % (l-1))
            
            thetacount += np.count_nonzero(1 - theta[:-1] > 1e-2)   # count how many theta are less than 1
            # end while loop for iteration convergence
        
        # Converged.  Before advancing to next timestep m, save some stuff
        Gammabar_mminus1 = Gammabar_ml
        n_mminus1 = n
        if Flag_CalculateFluxWithAvgN == True:
            nbar_mminus1 = nbar
        
        print('Number of iterations is %d' % l)
        # end for loop for time advancement
    
    toc()
    # plot the result
    # steady state solution
    S0 = 1; delta = 0.1; D0=1
    nright = (L-x)**3
    nleft = (L - delta + 0.75*(delta - x**(4/3) / delta**(1/3)))**3
    nss = S0 * delta / (27*D0) * nright
    nss[x < delta] = S0 * delta / (27*D0) * nleft[x < delta]
    
    #fig = plt.figure()
    plt.plot(x, n, 'b-')
    #plt.plot(x, nss, 'r-')
    #plt.show()
    
    
    return (x, n, Gammabar_ml)


def GetAvgQty(yl, ybar_lminus1, yInitialNa, ybar_mminus1, l, Na):
    """ For a quantity y (e.g., Gamma or n), compute the average over iterates ybar.
    Use the weighted averaging method; use the relaxation formula for faster performance.
    
    Inputs:
      yl            value of y on current iteration l (size N array)
      ybar_lminus1  ybar at previous iteration (size N array)
      yInitialNa    history of all y for the initial Na iterations (Na x N array) [used only for l <= Na]
      ybar_mminus1  converged ybar of previous timestep (size N array)
      l             current iteration
      Na            window size for averaging over iterates
    
    See Shestakov et al. (2003) for details on the averaging method.
    """
    
    if l < Na:
        ybar = np.sum(yInitialNa[:l,], axis=0) / Na + (Na - l)/Na * ybar_mminus1
    elif l == Na:
        # the "direct" calculation of the weighted average.  Need to do it once before using the relaxation method
        a = 1 - 1/Na
        Al = (1-a) / (1-a**(l+1))
        avec = a ** np.arange(l-1, -1, -1)  # avec = [a^(l-1),  a^(l-2), ..., a^2, a, 1]
        avec = np.reshape(avec, (l,1))
        # we want to multiply the kth *row* of yInitialNa by the kth element of avec: Broadcasting multiplication
        ybar = Al * np.sum(avec * yInitialNa[:l,], axis=0)
    else: # l > Na -- use relaxation method
        a = 1 - 1/Na
        Al = (1-a) / (1-a**(l+1))
        ybar = Al*yl + (1-Al) * ybar_lminus1
    return ybar
    
    
def GetCoeffs(Gammabar, nbar, dx, thetaparams, betaparams):
    """Calculate Dhat, chat, theta, beta, D, c, which are all evaluated at half-integer gridpoints"""
    # Formulas are slightly different if Gamma is given at integer or half-integer gridpoints.
    #  Here, assume Gamma is given at half-integer gridpoints.  If instead it is given at integer gridpoints,
    #  one estimates Gamma at the half-integer points with an average
    #
    # Let N denote the number of gridpoints.  Terms at half-integer gridpoints have only (N-1) elements that
    #  are used.  But they are represented in the code by a length N array, and the last element is NOT USED.
    
    # 1. Calculate Dhat, chat
    Dhat = np.zeros_like(nbar)
    chat = np.zeros_like(nbar)
    
    dntemp = (nbar[1:] - nbar[:-1]) / dx    # (partial_x n) on the half-integer grid points
    thetaparams.dndx = np.concatenate((dntemp, np.array([0.0])))
    
    Dhat[:-1] = -Gammabar[:-1] / dntemp
    Dhat[dntemp==0] = 0     # get rid of infinities resulting from divide by zero
    
    chat[:-1] = Gammabar[:-1] * 2 / (nbar[1:] + nbar[:-1])
    
    # 2. Calculate theta
    theta = ftheta(Dhat, thetaparams)
    # uncomment the following line to turn off convective terms anduse only diffusive terms
    # theta[:] = 1
    
    # 3. Calculate D, c
    D = theta * Dhat
    c = (1 - theta) * chat
    
    # 4. Calculate beta
    beta = fbeta(theta, c, D, dx, betaparams)
    
    return (Dhat, chat, theta, beta, D, c)

def ftheta(Dhat, thetaparams):
    """Scheme to calculate theta, the parameter that determines the split between diffusive and convective
    pieces in representations of the flux."""
    theta = np.zeros_like(Dhat)
    
    # Shestakov's default algorithm
    Dmin = thetaparams.Dmin
    Dmax = thetaparams.Dmax
    
    ind = np.logical_and(Dmin <= Dhat, Dhat <= Dmax)
    theta[ind] = (Dmax - Dhat[ind]) / (Dmax - Dmin)
    
    assert np.count_nonzero(np.logical_and(theta>=0, theta<=1)) == np.size(theta), 'some theta is not between 0 and 1'
    return theta
    
def ftheta2(Dhat, thetaparams):
    """Alternate scheme to calculate theta, the parameter that determines the split between diffusive and convective
    pieces in representations of the flux.
    
    Modification of Shestakov's default algorithm.  Here, when Dhat is large, we only add a convective
    part if dn/dx is also SMALL.  In other words, if Gamma and Dhat are large because dn/dx is large,
    then representing the flux purely as diffusive is fine.  The convective split for large Dhat is really
    to protect against spurious large Dhat resulting from finite flux at small gradients.
        if Dhat < Dmin, set theta to 0 (all convective)
        if Dhat >= Dmin AND dn/dx is small, use the Shestakov formula
        otherwise, set theta = 1 (all diffusive)
    """
    Dmin = thetaparams.Dmin
    Dmax = thetaparams.Dmax
    dndx = thetaparams.dndx
    dndx_thresh = thetaparams.dndx_thresh
    
    theta = np.ones_like(Dhat)
    ind = Dhat < Dmin
    theta[ind] = 0
    ind2 = np.logical_and.reduce((abs(dndx) < dndx_thresh, Dhat >= Dmin, Dhat <= Dmax))
    ind3 = np.logical_and(abs(dndx) < dndx_thresh, Dhat > Dmax)
    theta[ind2] = (Dmax - Dhat[ind2]) / (Dmax - Dmin)
    theta[ind3] = 0
    
    theta[-1] = 0   # not using the last element for an array at half-integer gridpoints
    assert np.count_nonzero(np.logical_and(theta>=0, theta<=1)) == np.size(theta), 'some theta is not between 0 and 1'
    return theta
    
def fbeta(theta, c, D, dx, betaparams):
    """Scheme to calculate beta, the parameter that determines the amount of upwinding when representing
    the convective part of the flux."""
    gamma = betaparams.gamma
    beta = np.zeros_like(theta)
    
    # Uncomment the following line to use pure upwind
    # beta = heaviside(c)
    
    # theta = 0 points
    beta[theta==0] = heaviside(c[theta==0])  # pure upwind
    
    # theta = 1 points
    beta[theta==1] = -9999999   # if theta=1, should not matter what this value is
    
    # 0 < theta < 1 points
    #   positive c
    # note, numpy logical_and only take 2 arguments max; use reduce to allow many arguments
    ind1 = np.logical_and.reduce((theta > 0, theta < 1, c > 0))
    betastar = np.zeros_like(beta)
    betastar[ind1] = 1 - D[ind1] / (c[ind1] * dx)
    beta[np.logical_and(ind1, betastar <= 1/2)] = 1/2
    beta[np.logical_and(ind1, betastar > 1/2)] = gamma * betastar[np.logical_and(ind1, betastar > 1/2)] + (1-gamma)
    
    #   zero c
    ind2 = np.logical_and.reduce((theta > 0, theta < 1, c == 0))  # chat must be 0 (meaning gammabar must be 0)
    beta[ind2] = 0      # does not matter what this value is when c=0
    
    #   negative c
    ind3 = np.logical_and.reduce((theta > 0, theta < 1, c < 0))
    betastar[ind3] = -D[ind3] / (c[ind3] * dx)
    beta[np.logical_and(ind3, betastar >= 1/2)] = 1/2
    beta[np.logical_and(ind3, betastar < 1/2)] = gamma * betastar[np.logical_and(ind3, betastar < 1/2)]
    
    return beta
    
def GetFlux(n, dx):
    """Test problem from Shestakov et al. (2003)
    Return the flux Gamma, which depends on the density profile n as follows:
       Gamma[n] = -(dn/dx)^3 / n^2
    """
    Gamma = np.zeros_like(n)
    
    # Gamma[0] corresponds to Gamma_{1/2} --- i.e., Gamma is expressed on a half-integer gridpoint
    Gamma[:-1] = -((n[1:] - n[:-1])/dx)**3/ ((n[:-1] + n[1:])/2)**2
    return Gamma
    
def GetS(x):
    """Test problem from Shestakov et al. (2003).
    Return the source S."""
    S = np.zeros_like(x)
    S0 = 1
    delta = 0.1
    S[x < delta] = S0
    return S
    
def GetTridiagMatAndRHS(D, c, beta, n_mminus1, S, dt, dx):
    """Construct the terms needed for the tridiagonal matrix equation to solve for the
    next iteration of n.
    
    Inputs: 
      D, c, beta at half-integer grid points, at iteration l-1
      n_mminus1 (converged density from last timestep), at integer grid points
      S (source term) at iteration l, at integer grid points
      
    For arrays at integer grid points, n[0] corresponds to the j=0 point, and n[-1] corresponds to the j=N-1 point
    For arrays at half-integer grid points, D[0] corresponds to j=1/2.  D[-2] corresponds to j = N - 3/2
    """
    (A, B, C) = GetTridiagMat(D, c, beta, dt, dx)
    f = GetMatRHS(n_mminus1, S, dt)
    
    # divide through the matrix equation by dt so that errors in steady state are not sensitive to dt
    A /= dt
    B /= dt
    C /= dt
    f /= dt
    return (A, B, C, f)
    
def GetTridiagMat(D, c, beta, dt, dx):
    """Used in GetTridiagMatAndRHS.  Construct the three diagonals of the matrix"""
    s = dt / dx**2
    s2 = dt / dx
    A = np.zeros_like(D)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    A[-1] = 0  # by definition
    C[0] = 0   # by definition
    
    #### Interior Points ####
    #  Accessing the interior points j=1, ..., N-2
    # for all elements corresponding to A_j:  A[1:-1]
    # for all elements corresponding to D_{j+1/2}:  D[1:-1]
    #   --because for j=1, we want D_{3/2},  and D[1] gives D_(3/2).
    #   --and for j=N-2, we want j-3/2, and D[-2] gives the N-3/2 element
    # for all elements corresponding to D_{j-1/2}:  D[:-2]
    #   --because for j=1, we want D_{1/2}, and D[0] gives D_{1/2}
    #   --and for j=N-2, we want j-5/2, and D[-3] gives the N-5/2 element, which will be the last element of D[:-2]
    
    A[1:-1] = -s * D[1:-1] + s2 * c[1:-1] * (1 - beta[1:-1])
    B[1:-1] = 1 + s*(D[1:-1] + D[:-2]) + s2*(c[1:-1]*beta[1:-1] - c[:-2]*(1-beta[:-2]))
    C[1:-1] = -s * D[:-2] - s2 * c[:-2] * beta[:-2]
    
    #### Boundary Points ####
    ### Apply Neumann conditions at left boundary
    A[0] = -s * D[0] + s2*c[0]*(1-beta[0])
    B[0] = 1 + s * D[0] + s2*c[0]*beta[0]
    
    ### Apply Dirichlet conditions at right boundary
    B[-1] = 1
    C[-1] = 0
    
    return (A, B, C)

def GetMatRHS(n_mminus1, S, dt):
    """Used in GetTridiagMatAndRHS.  Construct the RHS of the matrix equation."""
    #### Interior points ####    
    f = n_mminus1 + S*dt
    
    #### Modify f to satisfy boundary conditions ####
    ### Apply Neumann conditions at left boundary
    # f[0] = n_mminus1[0] + S[0]*dt    # same form as above, comment out because unnecessary
    
    ### Apply Dirichlet conditions at right boundary
    #f[-1] = 0.0
    f[-1] = 0.5
    return f

### A few other utilities
    
class paramstruct(object):
    """functions like a struct, but is mutable, unlike a namedtuple"""
    pass

def heaviside(x):
    """ out = heaviside(x).
    Heaviside step function.  Returns 0 if x<0, 1 if x>1, and 0.5 if x=0.  Vectorized."""
    x = np.asarray(x)    # convert scalars or empty sets into numpy array
    y = np.asarray(x>0).astype(float)
    y[x==0] = 0.5
    return y
    
def solve_tridiag(A,B,C,D):
    """Solve a tridiagonal system Mu = D, where the matrix M is tridiagonal.
 [B0  A0                               ] [u0]     [D0]
 [C1  B1   A1                          ] [u1]     [D1]
 [    C2   B2  A2                      ] [u2]     [D2]
 [            . . .                    ]  .     =  .
 [               . . .                 ]  .        .
 [                   CN-1   BN-1   AN-1]  .        .
 [                          CN     BN  ] [uN]     [DN]
    
    The arrays A, and C must have the same length as B.  A[-1] must be 0 and C[0] must be 0
    """
    B = -B # Jardin's algorithm used negative B on the diagonal
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
    
    """ Sample code to test solve_tridiag
    N = 12
    B = -np.ones(N)
    A = 0.1 * np.ones(N)
    C = 0.1 * np.ones(N)
    A[-1] = 0
    C[0] = 0
    D = 0.2 * np.random.rand(N)
    u1 = solve_tridiag(A, B, C, D)
    
    ## direct matrix solve
    # Construct the matrix
    M = np.diag(B) + np.diag(A[:-1], 1) + np.diag(C[1:], -1)
    u2 = np.linalg.solve(M, D)
    err = u1 - u2
    """
    
def tic():
    # Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
    
    
    