"""integration test ---  test the solver + handler modules in tango

Note --- these tests can take a relatively long time because using h5py to open & close hdf5 files every iteration can take a while.
For ~170 iterations, it might take a few seconds for a test.
"""

from __future__ import division
import numpy as np
import os
import h5py

from tango.extras import shestakov_nonlinear_diffusion
import tango

def test_history_handler_single_timestep():
    # test the use of solver class with history handler --- single timestep
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    
    setNumber = 0
    xTango = x
    xTurb = x
    t = tArray[1]
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    historyInterval = 12
    basename = 'tangodata'
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=historyInterval, basename='tangodata', maxIterations=1000, initialData=initialData)
    filename = basename + '_s{}'.format(setNumber) + '.hdf5'
    
    solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()
    
    assert os.path.exists(filename)
    
    # teardown
    os.remove(filename)
    
def test_history_handler_two_fields():
    """test the history handler with two fields, in a single timestep"""
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup_twofields()
    setNumber = 0
    xTango = x
    xTurb = x
    t = tArray[1]
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    basename = 'tangodata'
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=1000, initialData=initialData)
    filename = basename + '_s{}'.format(setNumber) + '.hdf5'
    
    solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()
    
    n_end = solver.profiles['n1']
    # compute analytic solution
    nss = shestakov_nonlinear_diffusion.steady_state_solution(x, nL)
    
    # read data and run checks
    with h5py.File(filename, 'r') as f:
        iterationNumberArray = f['iterationNumber']
        finalIterNumber = iterationNumberArray[-1]
        numWrites = f.attrs['writes']
        assert numWrites == finalIterNumber + 1
        # numerical solution
        n_read = f['n1/profile'][-1, :]
        solution_residual = (n_read - nss) / np.max(np.abs(nss))
        solution_rms_error = np.sqrt( 1/len(n_read) * np.sum(solution_residual**2))
        
        obs = solution_rms_error
        exp = 0
        testtol = 1e-3
        assert abs(obs - exp) < testtol
        assert np.allclose(n_read, n_end, rtol=0, atol=1e-15)
        
    # teardown
    os.remove(filename)
    
#def test_handler_multiple_files():
#    """test the use of handlers when there are multiple timesteps"""
#    (L, N, dx, x, nL, n, MaxIterations, tol, turbhandler, compute_all_H, t_array) = problem_setup()
#    (tangoCheckpointHandler, tangoHistoryHandler) = setup_handlers()
#    t_array = np.array([0, 1.0, 1e4])
#    solver = tng.solver.Solver(L, x, n, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)
#    solver.fileHandlerExecutor.add_handler(tangoCheckpointHandler)
#    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
#
#    while solver.ok:
#        # Implicit time advance: iterate to solve the nonlinear equation!
#        solver.take_timestep()
#    
#    checkpointProfName = 'checkpoint_test_prof.txt'    
#    checkpointEWMAName = 'checkpoint_test_ewma.txt'
#    historyTimestepName = 'history_test_timestep.npz'
#    historyIterationsName = 'history_test_iterations.npz'
#    
#    assert os.path.exists(checkpointProfName)
#    assert os.path.exists(checkpointEWMAName)
#    assert os.path.exists(historyTimestepName)
#    assert os.path.exists(historyIterationsName)
#    
#    # teardown
#    os.remove(checkpointProfName)
#    os.remove(checkpointEWMAName)
#    os.remove(historyTimestepName)
#    os.remove(historyIterationsName)
    
#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================

def problem_setup():    
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    maxIterations, lmParams, tol = initialize_parameters()
    compute_all_H = ComputeAllH()
    lm = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
    field = tango.multifield.Field(label='n', rightBC=nL, profile_mminus1=n, compute_all_H=compute_all_H, lodestroMethod=lm)
    fields = [field]
    tango.multifield.check_fields_initialize(fields)
    fluxModel = shestakov_nonlinear_diffusion.AnalyticFluxModel(dx)
    turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)
    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)
    tArray = np.array([0, 1e4])  # specify the timesteps to be used.    
    return (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray)

def problem_setup_twofields():    
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    maxIterations, lmParams, tol = initialize_parameters()
    compute_all_H1 = ComputeAllH()
    lm1 = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
    field1 = tango.multifield.Field(label='n0', rightBC=nL, profile_mminus1=n, compute_all_H=compute_all_H1, lodestroMethod=lm1)
    
    compute_all_H2 = ComputeAllH()
    lm2 = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
    field2 = tango.multifield.Field(label='n1', rightBC=nL, profile_mminus1=n, compute_all_H=compute_all_H2, lodestroMethod=lm2)
    
    fields = [field1, field2]
    tango.multifield.check_fields_initialize(fields)
    fluxModel = ShestakovTwoFluxModel(dx)
    turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)
    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)
    tArray = np.array([0, 1e4])  # specify the timesteps to be used.    
    return (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray)    

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

class ComputeAllH(object):
    def __init__(self):
        pass
    def __call__(self, t, x, profiles, HCoeffsTurb):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs

class ShestakovTwoFluxModel(object):
    def __init__(self, dx):
        self.dx = dx
    def get_flux(self, profiles):
        n0 = profiles['n0']
        n1 = profiles['n1']
        
        fluxes = {}
        fluxes['n0'] = shestakov_nonlinear_diffusion.get_flux(n0, self.dx)
        fluxes['n1'] = shestakov_nonlinear_diffusion.get_flux(n1, self.dx)
        return fluxes        
        
def setup_history_handler():
    historyInterval = 12
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=historyInterval, basename='history_test', maxIterations=1000)
    return tangoHistoryHandler
        
def setup_handlers():
    checkpointInterval = 11
    tangoCheckpointHandler = tango.handlers.TangoCheckpointHandler(iterationInterval=checkpointInterval, basename='checkpoint_test')
    historyInterval = 12
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=historyInterval, basename='history_test', maxIterations=1000)
    return (tangoCheckpointHandler, tangoHistoryHandler)