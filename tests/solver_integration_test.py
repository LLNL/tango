# integration test ---  test the solver, datasaver modules in tango

from __future__ import division
import numpy as np
from tango.extras import shestakov_nonlinear_diffusion
import tango as tng
import os

def test_solver_basic():
    # test the use of solver class
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    solver = tng.solver.solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    
    # set up data logger
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.TakeTimestep()
        
    n = solver.profile  # finished solution
    # compare with analytic steady state solution
    nss = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL)
    solution_residual = (n - nss) / np.max(np.abs(nss))
    solution_rms_error = np.sqrt( 1/len(n) * np.sum(solution_residual**2))
    
    obs = solution_rms_error
    exp = 0
    testtol = 1e-3
    assert abs(obs - exp) < testtol

def test_solver_single_timestep():
    # test the use of solver class with data logger --- single timestep
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    solver = tng.solver.solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    
    # set up data logger
    arrays_to_save = ['H2', 'H3', 'profile']
    databasename = 'test_integration_data'
    solver.DataSaverHandler.initialize_datasaver(databasename, MaxIterations, arrays_to_save)
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.TakeTimestep()
        
    n = solver.profile
    datasavename_timestep = databasename + "1_timestep.npz"
    datasavename_iterations = databasename + "1_iterations.npz"
    with np.load(datasavename_timestep) as npzfile:
        n_current = npzfile['profile_m']
    with np.load(datasavename_iterations) as npzfile:
        n_loaded = npzfile['profile'][-1,:]   # last row corresponds to last iteration
    assert(np.allclose(n, n_loaded, rtol=0, atol=1e-15))
    assert(np.allclose(n, n_current, rtol=0, atol=1e-15))
    os.remove(datasavename_timestep)
    os.remove(datasavename_iterations)

def test_solver_multiple_files():
    # test the use of solver class with data logger --- multiple files from multiple timesteps
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    t_array = np.array([0, 1.0, 1e4])
    solver = tng.solver.solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    
    # set up data logger
    arrays_to_save = ['H2', 'H3', 'profile']
    databasename = 'test_integration_data'
    solver.DataSaverHandler.initialize_datasaver(databasename, MaxIterations, arrays_to_save)
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.TakeTimestep()
        
    n = solver.profile  # finished solution
    
    datasavename1_iterations = databasename + "1_iterations.npz"
    datasavename2_timestep = databasename + "2_timestep.npz"
    with np.load(datasavename1_iterations) as npzfile:
        H2 = npzfile['H2']
        (temp, H2N) = np.shape(H2)
    with np.load(datasavename2_timestep) as npzfile:
        n_loaded = npzfile['profile_m']
    assert N == H2N
    assert(np.allclose(n, n_loaded, rtol=0, atol=1e-15))
    os.remove(datasavename1_iterations)
    os.remove(databasename + "1_timestep.npz")
    os.remove(datasavename2_timestep)
    os.remove(databasename + "2_iterations.npz")

def test_solver_not_converging():
    # test that data is stored even when solution does not converge within MaxIterations
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    MaxIterations = 100  # takes 170 iterations to converge for these parameters
    solver = tng.solver.solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    
    # set up data logger
    arrays_to_save = ['H2', 'H3', 'profile']
    databasename = 'test_integration_data'
    solver.DataSaverHandler.initialize_datasaver(databasename, MaxIterations, arrays_to_save)
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.TakeTimestep()
        
    n = solver.profile
    datasavename_timestep = databasename + "1_timestep.npz"
    with np.load(datasavename_timestep) as npzfile:
        n_loaded = npzfile['profile_m']
        iteration_number = npzfile['iteration_number']
    assert(np.allclose(n, n_loaded, rtol=0, atol=1e-15))
    assert len(iteration_number) == MaxIterations
    os.remove(datasavename_timestep)
    os.remove(databasename + "1_iterations.npz")
    
#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================

def problem_setup():
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    MaxIterations, lmparams, tol = initialize_parameters()
    FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
    turbhandler = tng.TurbulenceHandler(dx, lmparams, FluxModel)
    compute_all_H = ComputeAllH(turbhandler)
    t_array = np.array([0, 1e4])  # specify the timesteps to be used.
    return (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array)
    

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
    MaxIterations = 1000
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

class ComputeAllH(object):
    def __init__(self, turbhandler):
        self.turbhandler = turbhandler
    def __call__(self, t, x, n):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        (H2, H3, extradata) = self.turbhandler.Hcontrib_TurbulentFlux(n)
        H4 = None
        H6 = None
        return (H1, H2, H3, H4, H6, H7, extradata)
