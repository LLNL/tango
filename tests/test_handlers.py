"""Test handlers.py"""

from __future__ import division
import numpy as np
import os
from tango import handlers

def test_f1history_handler_execute():
    """Test the Savef1HistoryHandler execute() method --- copying the f1 checkpoint file"""
    basename = 'f1_iteration_history'
    f1Handler = handlers.Savef1HistoryHandler(iterationInterval=1, basename=basename)
    # setup -- create an empty file 
    genefile = 'checkpoint_000'
    open(genefile, 'w').close()
    
    data=None
    iterationNumber = 12
    f1Handler.execute(data, iterationNumber)
    
    dest = basename + '_' + str(iterationNumber)
    assert os.path.isfile(dest) == True
    
    # teardown
    os.remove(genefile)
    os.remove(dest)
    
def test_tango_checkpoint_execute():
    """Test the TangoCheckpointHandler execute() method --- write out ASCII files"""
    # setup
    basename='tango_checkpoint'
    tangoCheckpointHandler = handlers.TangoCheckpointHandler(iterationInterval=1, basename=basename)
    data = data_setup()
    iterationNumber = 14
    
    # run the execute() method
    tangoCheckpointHandler.execute(data, iterationNumber)

    # check it    
    filename_prof = basename + '_prof.txt'
    filename_ewma = basename + '_ewma.txt'
        
    # read filename_prof    
    # get the header:
    with open(filename_prof, 'r') as f:
        first_line = f.readline().strip().lstrip('# ') # remove trailing newline and beginning comment character
    iterNumber_read = int(first_line)
    
    # read the data
    (x, profile) = np.loadtxt(filename_prof, unpack=True)
    
    # read filename_ewma
    (x_turbgrid, profileEWMA_turbgrid, fluxEWMA_turbgrid) = np.loadtxt(filename_ewma, unpack=True)
    
    assert iterNumber_read == iterationNumber
    assert np.all(x == data['x'])
    assert np.all(profile == data['profile'])
    assert np.all(x_turbgrid == data['xTurbGrid'])
    assert np.all(profileEWMA_turbgrid == data['profileEWMATurbGrid'])
    assert np.all(fluxEWMA_turbgrid == data['fluxEWMATurbGrid'])
    
    # teardown
    os.remove(filename_prof)
    os.remove(filename_ewma)

def test_tango_history_add_data():
    """Test the TangoHistoryHandler add_data() method"""
    # setup
    basename='tango_history'
    maxIterations=20
    iterationInterval=5
    tangoHistoryHandler = handlers.TangoHistoryHandler(iterationInterval=iterationInterval, basename=basename, maxIterations=maxIterations)
    data = data_setup()
    
    iterationNumber = 14
    
    # run add_data()
    tangoHistoryHandler.add_data(data, iterationNumber)
    tangoHistoryHandler.add_data(data, iterationNumber+1)
    
    assert tangoHistoryHandler.countStoredIterations == 2
    assert tangoHistoryHandler.data0D['rmsError'][0] == 0.02
    assert tangoHistoryHandler.data0D['rmsError'][1] == 0.02
    assert np.all(tangoHistoryHandler.data1D['x'][1, :] == data['x'])
    assert tangoHistoryHandler.iterationNumber[1] == 15
    
def test_tango_history_prepare_to_write_data():
    """Test the TangoHistoryHandler prepare_to_write_data() method"""    
    # setup
    basename='tango_history'
    maxIterations=20
    iterationInterval=5
    tangoHistoryHandler = handlers.TangoHistoryHandler(iterationInterval=iterationInterval, basename=basename, maxIterations=maxIterations)
    data = data_setup()
    iterationNumber = 14
    tangoHistoryHandler.add_data(data, iterationNumber)
    tangoHistoryHandler.add_data(data, iterationNumber+1)
    
    # run it
    (data0D, data1D) = tangoHistoryHandler.prepare_to_write_data()
    
    # check
    assert len(data0D['rmsError']) == 2
    nrows, ncols = data1D['profile'].shape
    assert nrows == 2
    assert ncols == len(data['profile'])
    
    
def test_tango_history_save_to_file():
    """Test the TangoHistoryHandler save_to_file() method"""
    # setup
    basename='tango_history'
    maxIterations=20
    iterationInterval=5
    tangoHistoryHandler = handlers.TangoHistoryHandler(iterationInterval=iterationInterval, basename=basename, maxIterations=maxIterations)
    data = data_setup()
    iterationNumber = 14
    tangoHistoryHandler.add_data(data, iterationNumber)
    tangoHistoryHandler.add_data(data, iterationNumber+1)
    (data0D, data1D) = tangoHistoryHandler.prepare_to_write_data()
    
    # run it
    tangoHistoryHandler.save_to_file(data0D, data1D)
    
    # check
    filename_timestep = basename + "_timestep.npz"
    filename_iterations = basename + "_iterations.npz"
        
    with np.load(filename_timestep) as npzfile:
        assert npzfile['rmsError'][1] == data['rmsError']
        assert npzfile['iterationNumber'][0] == iterationNumber
    with np.load(filename_iterations) as npzfile:
        assert np.all(npzfile['profile'][1,:] == data['profile'])
        assert np.all(npzfile['fluxEWMATurbGrid'][0,:] == data['fluxEWMATurbGrid'])
    
    # teardown
    os.remove(filename_timestep)
    os.remove(filename_iterations)
    
def test_tango_history_reset():
    """Test the TangoHistoryHandler reset_for_next_timestep() method"""
    # setup
    basename='tango_history'
    maxIterations=20
    iterationInterval=5
    tangoHistoryHandler = handlers.TangoHistoryHandler(iterationInterval=iterationInterval, basename=basename, maxIterations=maxIterations)
    data = data_setup()
    iterationNumber = 14
    tangoHistoryHandler.add_data(data, iterationNumber)
    tangoHistoryHandler.add_data(data, iterationNumber+1)
    
    assert tangoHistoryHandler.countStoredIterations == 2
    assert tangoHistoryHandler.iterationNumber[0] == 14
    
    tangoHistoryHandler.reset_for_next_timestep()
    assert tangoHistoryHandler.countStoredIterations == 0
    assert tangoHistoryHandler.iterationNumber[0] == 0
    assert tangoHistoryHandler.data0D == {}
    assert tangoHistoryHandler.data1D == {}    

def test_executor_add_handler():
    """test the Executor add_handler() method"""
    # setup
    executor = handlers.Executor()
    checkpointInterval = 10
    checkpointHandler = handlers.TangoCheckpointHandler(iterationInterval=checkpointInterval)
    historyInterval = 50
    maxIterations = 500
    historyHandler = handlers.TangoHistoryHandler(iterationInterval=historyInterval, maxIterations=maxIterations)
    executor.add_handler(checkpointHandler)
    executor.add_handler(historyHandler)
    
    # check
    assert len(executor.handlers) == 2
    assert checkpointHandler in executor.handlers
    assert historyHandler in executor.handlers
    
def test_executor_scheduled_handlers():
    """test the Executor scheduled_handlers() method"""
    # setup
    executor = handlers.Executor()
    checkpointInterval = 10
    historyInterval = 12
    checkpointHandler = handlers.TangoCheckpointHandler(iterationInterval=checkpointInterval)    
    maxIterations = 500
    historyHandler = handlers.TangoHistoryHandler(iterationInterval=historyInterval, maxIterations=maxIterations)
    executor.add_handler(checkpointHandler)
    executor.add_handler(historyHandler)
    
    iterationNumber = 9
    scheduled = executor.scheduled_handlers(iterationNumber)
    assert checkpointHandler not in scheduled and historyHandler not in scheduled
    
    iterationNumber = 10
    scheduled = executor.scheduled_handlers(iterationNumber)
    assert checkpointHandler in scheduled and historyHandler not in scheduled
    
    # run with same iterationNumber, checkPoint should not be scheduled
    scheduled = executor.scheduled_handlers(iterationNumber)
    assert checkpointHandler not in scheduled and historyHandler not in scheduled
    
    iterationNumber = 20
    scheduled = executor.scheduled_handlers(iterationNumber)
    assert checkpointHandler in scheduled and historyHandler in scheduled


#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================

def data_setup():
    data = {}
    N_tangogrid = 9
    N_turbgrid = 7
    data['x'] = np.linspace(0, 3, N_tangogrid)
    data['profile'] = np.linspace(2, 3, N_tangogrid)
    
    data['xTurbGrid'] =np.linspace(0, 3, N_turbgrid)
    data['profileEWMATurbGrid'] = np.linspace(4, 6, N_turbgrid)
    data['fluxEWMATurbGrid'] = np.linspace(9, 9.4, N_turbgrid)
    
    data['rmsError'] = 0.02
    return data



#def test_AddData():
#    MaxIterations=15
#    NumPts = 20
#    arrays_to_save = ['H2', 'H3']
#    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
#    H2 = np.linspace(1, 2, NumPts)
#    H3 = np.linspace(4, 5, NumPts)
#    H4 = np.linspace(4, 5, 30)
#    data = {'H2': H2, 'H3': H3, 'H4': H4}
#    DataSaver.AddData(data, 1)
#    DataSaver.AddData(data, 2)
#    
#    assert DataSaver.counter == 2
#    assert DataSaver.data_all_iterations['H3'][1, 19] == 5
#    assert DataSaver.iteration_number[0] == 1
#   
#def test_SaveToFile():
#    MaxIterations=15
#    NumPts = 20
#    arrays_to_save = ['H2', 'H3']
#    psi = np.linspace(0, 1, NumPts)
#    Vprime = np.ones(NumPts)
#    one_off_data = {'psi': psi, 'Vprime': Vprime}
#    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
#    DataSaver.AddOneOffData(one_off_data)
#    H2 = np.linspace(1, 2, NumPts)
#    H3 = np.linspace(4, 5, NumPts)
#    data = {'H2': H2, 'H3': H3}
#    DataSaver.AddData(data, 1)
#    DataSaver.AddData(data, 2)
#    
#    DataSaver.SaveToFile('testsave')
#    
#    with np.load('testsave_timestep.npz') as npzfile:
#    # check the one_off_data got saved and then check H2 H3
#        assert np.all(npzfile['psi'] == psi)
#        assert np.all(npzfile['Vprime'] == Vprime)
#    with np.load('testsave_iterations.npz') as npzfile:
#        assert np.all(npzfile['H2'][1,:] == H2)
#        assert np.all(npzfile['H3'][0,:] == H3)
#    
#    # teardown
#    os.remove('testsave_timestep.npz')
#    os.remove('testsave_iterations.npz')
#
#def test_ResetData():
#    MaxIterations=15
#    NumPts = 20
#    arrays_to_save = ['H2', 'H3']
#    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
#    H2 = np.linspace(1, 2, NumPts)
#    H3 = np.linspace(4, 5, NumPts)
#    data = {'H2': H2, 'H3': H3}
#    DataSaver.AddData(data, 1)
#    DataSaver.AddData(data, 2)
#    
#    assert np.all(DataSaver.data_all_iterations['H2'][1,:] == H2)
#    DataSaver.ResetForNextTimestep()
#    assert DataSaver.counter == 0
#    assert DataSaver.finalized == False
#    assert DataSaver.one_off_data == {}
#    assert np.all(DataSaver.iteration_number == np.zeros(MaxIterations))
#    assert DataSaver.data_all_iterations == {}
