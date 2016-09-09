# Copyright (c) 2016, Lawrence Livermore National Security, LLC.  Produced at
# the Lawrence Livermore National Laboratory.  LLNL-CODE-702341.  All Rights
# Reserved.
#
# This file is part of Tango, a transport equation solver intended for coupling
# with codes that calculate turbulent fluxes.
#
# Tango is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.


from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import transport_solver_py

# transport solve script
L = 1           # size of domain
N = 500         # number of spatial grid points
dx = L / (N-1)  # spatial grid size
x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1

Na = 40        # averaging window for iteration scheme
lmax = 5000     # max number of iterations
dt = 1e3       # timestep

Flag_CalculateFluxWithAvgN = True
    # if true, use nbar to calculate flux Gamma: Gamma[nbar] rather than Gamma[n]
thetaparams = transport_solver_py.paramstruct()
betaparams = transport_solver_py.paramstruct()
thetaparams.Dmin = 1e-5
thetaparams.Dmax = 1e13
thetaparams.dndx_thresh = 10
betaparams.gamma = 0.8

# set up arrays, or at least indicate their length
n = np.ones_like(x)
#n[-1] = 0.1     # initial condition
n = 1 - 0.5*x

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
#t = np.array([0, 1e-5, 1e-4, 3e-4, 6e-4, 9e-4, 1e-3, 2e-3, 4e-3, 1e-2, 1e-1, 1e0, 1e3, 1e4])

thetacount = 0     # diagnostic counter to see the number of times that theta is not 1

# initialize "m minus 1" variables for the first timestep
nbar_mminus1 = n
n_mminus1 = n
Gammabar_mminus1 = transport_solver_py.GetFlux(nbar_mminus1, dx)


transport_solver_py.tic()
for m in range(1, len(t)):
    # Time advance: iterate to solve the nonlinear equation!
    #dt = t[m] - t[m-1]   # use this if using non-constant timesteps
    GammaInitialNa[:] = 0       # reset elements to zero for new timestep
    nInitialNa[:] = 0
    converged = False
    
    # compute matrix for first iterate.  Use converged Gammabar from previous timestep (m-1) as Gammabar^{m,0}
    Gammabar_m0 = Gammabar_mminus1
    nbar_m0 = nbar_mminus1
    n_m0 = n_mminus1
    
    # Calculate Dhat, chat, theta, beta, D, c for first iterate
    if Flag_CalculateFluxWithAvgN == True:
        (Dhat, chat, theta, beta, D, c) = transport_solver_py.GetCoeffs(Gammabar_m0, nbar_m0, dx, thetaparams, betaparams)
    else:
        (Dhat, chat, theta, beta, D, c) = transport_solver_py.GetCoeffs(Gammabar_m0, n_m0, dx, thetaparams, betaparams)
    
    S = transport_solver_py.GetS(x)
    (A, B, C, f) = transport_solver_py.GetTridiagMatAndRHS(D, c, beta, n_mminus1, S, dt, dx)
    
    l=1
    while not converged:
        # 1. Solve the tridiagonal matrix for n^{m,l} with matrix M from previous l
        n = transport_solver_py.solve_tridiag(A, B, C, f)
        if l <= Na:
            nInitialNa[l-1,] = n
        
        #### DEBUG POINT
        #if l==6:
            #raise SystemExit(0)
        
        # 2. Compute Gamma^{m,l}
        if Flag_CalculateFluxWithAvgN == True:
            nbar = transport_solver_py.GetAvgQty(n, nbar_lminus1, nInitialNa, nbar_mminus1, l, Na)
            Gamma_ml = transport_solver_py.GetFlux(nbar, dx)
        else:
            Gamma_ml = transport_solver_py.GetFlux(n, dx)
            
        if l <= Na:
            GammaInitialNa[l-1,] = Gamma_ml
        
        # 3. Compute the weighted average Gammabar^{m,l}, and Dhat, chat, theta, beta, D, c
        Gammabar_ml = transport_solver_py.GetAvgQty(Gamma_ml, Gammabar_lminus1, GammaInitialNa, Gammabar_mminus1, l, Na)
        
        if Flag_CalculateFluxWithAvgN == True:
            (Dhat, chat, theta, beta, D, c) = transport_solver_py.GetCoeffs(Gammabar_ml, nbar, dx, thetaparams, betaparams)
        else:
            (Dhat, chat, theta, beta, D, c) = transport_solver_py.GetCoeffs(Gammabar_ml, n, dx, thetaparams, betaparams)
        
        # 4. With D, c, beta, construct the tridiagonal matrix M^{m,l}.  Also construct the source S and RHS term f
        S = transport_solver_py.GetS(x)
        (A, B, C, f) = transport_solver_py.GetTridiagMatAndRHS(D, c, beta, n_mminus1, S, dt, dx)
        
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

transport_solver_py.toc()
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

