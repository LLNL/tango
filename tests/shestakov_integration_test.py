# integration test ---  test the LoDestro Method and solvers on the analytic Shestakov problem
#  NOT using the solver class in tango

from __future__ import division
import numpy as np
from tango import HToMatrixFD
from tango import lodestro_method
from tango.extras import shestakov_nonlinear_diffusion

def test_lodestro_shestakovproblem():
    # test the LoDestro Method on the analytic shestakov problem

    # setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    nL = 1e-2           # right boundary condition
    
    lmax = 2000     # max number of iterations
    dt = 1e4       # timestep
    
    thetaparams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdx_thresh': 10}
    
    lmparams = {'EWMA_param_turbflux': 0.30,
                'EWMA_param_profile': 0.30,
                'thetaparams': thetaparams}
                
    FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
    turbhandler = lodestro_method.TurbulenceHandler(dx, lmparams, FluxModel)
    
    # initial condition
    n = 1 - 0.5*x
    
    tol = 1e-11      # tol for convergence
    t = np.array([0, 1e6])
    
    # initialize "m minus 1" variables for the first timestep
    n_mminus1 = n
    
    for m in range(1, len(t)):
        # Implicit time advance: iterate to solve the nonlinear equation!
        dt = t[m] - t[m-1]   # use this if using non-constant timesteps
        converged = False
        
        l = 1
        while not converged:
            # compute H's from current iterate n
            H1 = np.ones_like(x)
            H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
            
            # this needs to be packaged... give n, get out H2, H3.
            (H2, H3, data) = turbhandler.Hcontrib_TurbulentFlux(n)
                    
            # compute matrix system (A, B, C, f)
            (A, B, C, f) = HToMatrixFD.HToMatrix(dt, dx, nL, n_mminus1, H1, H2=H2, H3=H3, H7=H7)
            
            # check convergence
            #    convergence check: is || ( M[n^l] n^l - f[n^l] ) / max(abs(f)) || < tol
            #    could add more convergence checks here
            resid = A*np.concatenate((n[1:], np.zeros(1))) + B*n + C*np.concatenate((np.zeros(1), n[:-1])) - f
            resid = resid / np.max(np.abs(f))  # normalize residuals
            rms_error = np.sqrt( 1/len(resid) * np.sum(resid**2))  
            
            if rms_error < tol:
                converged = True
            
            # compute new iterate n
            n = HToMatrixFD.solve(A, B, C, f)
            
            # Check for NaNs or infs
            if np.all(np.isfinite(n)) == False:
                raise RuntimeError('NaN or Inf detected at l=%d.  Exiting...' % (l))
            
            # about to loop to next iteration l
            l += 1
            if l >= lmax:
                raise RuntimeError('Too many iterations on timestep %d.  Error is %f.' % (m, rms_error))
            
            
                
            # end of while loop for iteration convergence
        
        # Converged.  Before advancing to next timestep m, save some stuff
        n_mminus1 = n    
    
    nss_analytic = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL) 
    solution_residual = (n - nss_analytic) / np.max(np.abs(nss_analytic))
    solution_rms_error = np.sqrt( 1/len(n) * np.sum(solution_residual**2))
    
    obs = solution_rms_error
    exp = 0
    testtol = 1e-3
    assert abs(obs - exp) < testtol