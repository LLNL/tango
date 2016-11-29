"""Example for how to use tango to solve a turbulence and transport problem.

Here, the "turbulent flux" is specified analytically, using the example in the Shestakov et al. (2003) paper.
This example is a nonlinear diffusion equation with specified diffusion coefficient and source.  There is a
closed form answer for the steady state solution which can be compared with the numerically found solution.
"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tango.extras import shestakov_nonlinear_diffusion
import tango as tng

def initialize_shestakov_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    nL = 1e-2           # right boundary condition
    n_initialcondition = 1 - 0.5*x
    return (L, N, dx, x, nL, n_initialcondition)

def initialize_parameters():
    MaxIterations = 2000
    thetaparams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdx_thresh': 10}
    EWMA_param_turbflux = 0.30
    EWMA_param_profile = 0.30
    lmparams = {'EWMA_param_turbflux': EWMA_param_turbflux,
            'EWMA_param_profile': EWMA_param_profile,
            'thetaparams': thetaparams}
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (MaxIterations, lmparams, tol)

def Compute_Hs(x, n, turbhandler):
    # Define the contributions to the H coefficients for the Shestakov Problem
    H1 = np.ones_like(x)
    H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
    (H2, H3, data) = turbhandler.Hcontrib_TurbulentFlux(n)
    H4 = None
    H6 = None
    return (H1, H2, H3, H4, H6, H7)
    
def Get_dt(t, m):
    # assuming the user specifies an array t of times, and we are on the mth step, return the time
    # increment dt = t[m] - t[m-1]
    return t[m] - t[m-1] # particularly important if using non-constant timestesp
    
def CheckConvergence(A, B, C, f, n, tol):
    # convergence check: is || ( M[n^l] n^l - f[n^l] ) / max(abs(f)) || < tol
    # could add more convergence checks
    resid = A*np.concatenate((n[1:], np.zeros(1))) + B*n + C*np.concatenate((np.zeros(1), n[:-1])) - f
    resid = resid / np.max(np.abs(f))  # normalize residuals
    rms_error = np.sqrt( 1/len(resid) * np.sum(resid**2))  
    converged = False
    if rms_error < tol:
        converged = True
    return (converged, rms_error, resid)

#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
L, N, dx, x, nL, n = initialize_shestakov_problem()
MaxIterations, lmparams, tol = initialize_parameters()

           
FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
turbhandler = tng.TurbulenceHandler(dx, x, lmparams, FluxModel)
errhistory = np.zeros(MaxIterations-1)      # error history vs. iteration at a given timestep
t = np.array([0, 1e4])  # specify the timesteps to be used.
n_mminus1 = n           # initialize "m minus 1" variables for the first timestep

for m in range(1, len(t)):
    # Implicit time advance: iterate to solve the nonlinear equation!
    converged = False
    dt = Get_dt(t, m)
    
    l = 1   # reset iteration counter
    errhistory[:] = 0
    while not converged:
        # compute H's from current iterate n
        (H1, H2, H3, H4, H6, H7) = Compute_Hs(x, n, turbhandler)
        
        # compute matrix system (A, B, C, f)
        (A, B, C, f) = tng.HToMatrix(dt, dx, nL, n_mminus1, H1, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7)

        converged, rms_error, resid = CheckConvergence(A, B, C, f, n, tol)
        errhistory[l-1] = rms_error
        
        # compute new iterate n
        n = tng.solve(A, B, C, f)
               
        # Check for NaNs or infs
        if np.all(np.isfinite(n)) == False:
            raise RuntimeError('NaN or Inf detected at l=%d.  Exiting...' % (l))
        
        # about to loop to next iteration l
        l += 1
        if l >= MaxIterations:
            raise RuntimeError('Too many iterations on timestep %d.  Error is %f.' % (m, rms_error))
        
        # end of while loop for iteration convergence
    
    # Converged.  Before advancing to next timestep m, save some stuff
    n_mminus1 = n
    
    print('Number of iterations is %d' % l)
    # end for loop for time advancement

# Plot result and compare with analytic steady state solution
nss = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL)

fig = plt.figure()
plt.plot(x, n, 'b-')
plt.plot(x, nss, 'r-')

solution_residual = (n - nss) / np.max(np.abs(nss))
solution_rms_error = np.sqrt( 1/len(n) * np.sum(solution_residual**2))
print('Error compared to analytic steady state solution is %f' % (solution_rms_error))


#plt.figure()
#plt.semilogy(errhistory)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)