"""Test handlers.py"""

from __future__ import division
import numpy as np
import os
import h5py

from tango.extras import shestakov_nonlinear_diffusion
import tango
#from tango import handlers, restart

def test_restart():
    # first run, should abort at iterationNumber 100 (101 writes)
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    
    # set up the handler
    (setNumber, xTango, xTurb, t) = (0, x, x, tArray[1])
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, maxIterationsPerSet=101)
    basename = 'tangodata'
    filename0 = basename + '_s0.hdf5'
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
    
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()

    # check the filename0, last iteration number is 100
    with h5py.File(filename0, 'r') as f:
        assert f['iterationNumber'][-1] == 100
        
    # ******************************* Begin Restart ******************************
    # set up fields as normal to fill in the EWMA params, labels, profiles_mminus1, etc.
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    # restartfile = tango.restart.check_if_should_restart(basename)
    restartfile = 'tangodata_s0.hdf5'  # use this to short circuit errors possibly resulting from the above...
    # xTango, xTurb, basename set above
    
    if restartfile:
        (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turbFluxesEWMA) = tango.restart.read_metadata_from_previousfile(restartfile)
        tango.restart.set_ewma_iterates(fields, old_profilesEWMA, old_turbFluxesEWMA)
        initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    else: # shouldn't be run here, but this is the strategy to use in general code
        (setNumber, startIterationNumber, t, timestepNumber) = (0, 0, tArray[1], 1)
        initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    # instantiate the handler
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
    if restartfile:
        solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, startIterationNumber=startIterationNumber, profiles=old_profiles)
    else:
        solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
    filename1 = basename + '_s1.hdf5'

    # run solver a second time
    while solver.ok:
        solver.take_timestep()
    
    # check
    with h5py.File(filename1, 'r') as f:
        assert f['iterationNumber'][0] == 101
        assert f['iterationNumber'][-1] == 170
    
    # teardown
    os.remove(filename0)
    os.remove(filename1)
    
    
    
def test_combine_savefiles():
    # first run, should abort at iterationNumber 100 (101 writes)
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    
    # set up the handler
    (setNumber, xTango, xTurb, t) = (0, x, x, tArray[1])
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, maxIterationsPerSet=101)
    basename = 'tangodata'
    filename0 = basename + '_s0.hdf5'
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
    
    while solver.ok:
        # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()

    # check the filename0, last iteration number is 100
    with h5py.File(filename0, 'r') as f:
        assert f['iterationNumber'][-1] == 100
        
    # ******************************* Begin First Restart ******************************
    # second run, should at abort at iteration number 110 (10 writes)
    # set up fields as normal to fill in the EWMA params, labels, profiles_mminus1, etc.
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    # restartfile = tango.restart.check_if_should_restart(basename)
    restartfile = 'tangodata_s0.hdf5'  # use this to short circuit errors possibly resulting from the above...
    # xTango, xTurb, basename set above
    
    if restartfile:
        (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turbFluxesEWMA) = tango.restart.read_metadata_from_previousfile(restartfile)
        tango.restart.set_ewma_iterates(fields, old_profilesEWMA, old_turbFluxesEWMA)
        initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    # instantiate the handler
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
    if restartfile:
        solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, startIterationNumber=startIterationNumber, maxIterationsPerSet=10, profiles=old_profiles)

    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
    filename1 = basename + '_s1.hdf5'

    # run solver a second time
    while solver.ok:
        solver.take_timestep()
    
    # ******************************* Begin Second Restart ******************************
        # set up fields as normal to fill in the EWMA params, labels, profiles_mminus1, etc.
    (L, N, dx, x, nL, n, maxIterations, tol, fields, compute_all_H_all_fields, tArray) = problem_setup()
    # restartfile = tango.restart.check_if_should_restart(basename)
    restartfile = 'tangodata_s1.hdf5'  # use this to short circuit errors possibly resulting from the above...
    # xTango, xTurb, basename set above
    if restartfile:
        (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turbFluxesEWMA) = tango.restart.read_metadata_from_previousfile(restartfile)
        tango.restart.set_ewma_iterates(fields, old_profilesEWMA, old_turbFluxesEWMA)
        initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
    
    # instantiate the handler
    tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
    if restartfile:
        solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, startIterationNumber=startIterationNumber, profiles=old_profiles)

    solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
    filename2 = basename + '_s2.hdf5'
    
    # run solver a third time
    while solver.ok:
        solver.take_timestep()    
        
    # Combine the three savefiles
    tango.restart.combine_savefiles()
        
    filename = 'tangodata_combined.hdf5'
        
    # check
    with h5py.File(filename0, 'r') as f0, h5py.File(filename1, 'r') as f1, h5py.File(filename2, 'r') as f2, h5py.File(filename, 'r') as f:
        assert f0['iterationNumber'][-1] == 100
        assert f1['iterationNumber'][0] == 101
        assert f1['iterationNumber'][-1] == 110
        assert f2['iterationNumber'][0] == 111
        assert np.allclose(f['n/profile'][111:, :], f2['n/profile'][:, :], rtol=0, atol=1e-15)
        
    # teardown
    os.remove(filename0)
    os.remove(filename1)
    os.remove(filename2)
    os.remove(filename)

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
