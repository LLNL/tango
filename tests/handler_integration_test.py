# integration test ---  test the solver, datasaver modules in tango

from __future__ import division
import numpy as np
from tango.extras import shestakov_nonlinear_diffusion
import tango as tng
import os

def test_handler_single_timestep():
    # test the use of solver class with data logger --- single timestep
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    (tangoCheckpointHandler, tangoHistoryHandler) = setup_handlers()
    solver = tng.solver.Solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    solver.fileHandlerExecutor.add_handler(tangoCheckpointHandler)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()
        
    checkpointProfName = 'checkpoint_test_prof.txt'    
    checkpointEWMAName = 'checkpoint_test_ewma.txt'
    historyTimestepName = 'history_test_timestep.npz'
    historyIterationsName = 'history_test_iterations.npz'
    
    assert os.path.exists(checkpointProfName)
    assert os.path.exists(checkpointEWMAName)
    assert os.path.exists(historyTimestepName)
    assert os.path.exists(historyIterationsName)
    
    # teardown
    os.remove(checkpointProfName)
    os.remove(checkpointEWMAName)
    os.remove(historyTimestepName)
    os.remove(historyIterationsName)

def test_handler_multiple_files():
    """test the use of handlers when there are multiple timesteps"""
    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
    (tangoCheckpointHandler, tangoHistoryHandler) = setup_handlers()
    t_array = np.array([0, 1.0, 1e4])
    solver = tng.solver.Solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
    solver.fileHandlerExecutor.add_handler(tangoCheckpointHandler)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()
    
    checkpointProfName = 'checkpoint_test_prof.txt'    
    checkpointEWMAName = 'checkpoint_test_ewma.txt'
    historyTimestepName = 'history_test_timestep.npz'
    historyIterationsName = 'history_test_iterations.npz'
    
    assert os.path.exists(checkpointProfName)
    assert os.path.exists(checkpointEWMAName)
    assert os.path.exists(historyTimestepName)
    assert os.path.exists(historyIterationsName)
    
    # teardown
    os.remove(checkpointProfName)
    os.remove(checkpointEWMAName)
    os.remove(historyTimestepName)
    os.remove(historyIterationsName)
    
#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================

def problem_setup():
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    MaxIterations, lmparams, tol = initialize_parameters()
    FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
    turbhandler = tng.TurbulenceHandler(dx, x, lmparams, FluxModel)
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
    maxIterations = 1000
    thetaParams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdxThreshold': 10}
    EWMAParamTurbFlux = 0.30
    EWMAParamProfile = 0.30
    lmParams = {'EWMAParamTurbFlux': EWMAParamTurbFlux,
            'EWMAParamProfile': EWMAParamProfile,
            'thetaParams': thetaParams}
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)

def setup_handlers():
    checkpointInterval = 11
    tangoCheckpointHandler = tng.handlers.TangoCheckpointHandler(iterationInterval=checkpointInterval, basename='checkpoint_test')
    historyInterval = 12
    tangoHistoryHandler = tng.handlers.TangoHistoryHandler(iterationInterval=historyInterval, basename='history_test', maxIterations=1000)
    return (tangoCheckpointHandler, tangoHistoryHandler)

class ComputeAllH(object):
    def __init__(self, turbhandler):
        self.turbhandler = turbhandler
    def __call__(self, t, x, n):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        (H2, H3, extradata) = self.turbhandler.Hcontrib_turbulent_flux(n)
        H4 = None
        H6 = None
        return (H1, H2, H3, H4, H6, H7, extradata)