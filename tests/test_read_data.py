"""Test read_data.py"""

from __future__ import division
import numpy as np
import os
from tango import handlers
from tango import read_data

def test_read_tango_checkpoint():
    """Test read_data.read_tango_checkpoint() for reading in of tango checkpoint files"""
     # setup
    basename='tango_checkpoint'
    tangoCheckpointHandler = handlers.TangoCheckpointHandler(iterationInterval=1, basename=basename)
    data = data_setup()
    iterationNumber = 14
    tangoCheckpointHandler.execute(data, iterationNumber)

    # run it and check
    (iterationNumberRead, xTango, pressureProfile, xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid) = read_data.read_tango_checkpoint()
    
    assert iterationNumberRead == iterationNumber
    assert np.all(xTango == data['x'])
    assert np.all(pressureProfile == data['profile'])
    assert np.all(xTurbGrid == data['xTurbGrid'])
    assert np.all(profileEWMATurbGrid == data['profileEWMATurbGrid'])
    assert np.all(fluxEWMATurbGrid == data['fluxEWMATurbGrid'])
    
    # teardown
    os.remove(basename + '_prof.txt')
    os.remove(basename + '_ewma.txt')

def test_read_tango_history():
    """test read_data.read_tango_history() for reading in of tango history flies."""
    # setup
    basename='tango_history'
    maxIterations=20
    iterationInterval=5
    tangoHistoryHandler = handlers.TangoHistoryHandler(iterationInterval=iterationInterval, basename=basename, maxIterations=maxIterations)
    data = data_setup()
    iterationNumber = 14
    tangoHistoryHandler.add_data(data, iterationNumber)
    tangoHistoryHandler.add_data(data, iterationNumber+1)
    (data0DToSave, data1DToSave) = tangoHistoryHandler.prepare_to_write_data()
    tangoHistoryHandler.save_to_file(data0DToSave, data1DToSave)
    
    # run it
    (data0D, data1D) = read_data.read_tango_history(basename=basename)
    
    # check    
    assert data0D['rmsError'][1] == data['rmsError']
    assert data0D['iterationNumber'][0] == iterationNumber
    assert np.all(data1D['profile'][1,:] == data['profile'])
    assert np.all(data1D['fluxEWMATurbGrid'][0,:] == data['fluxEWMATurbGrid'])
    
    # teardown
    filename_timestep = basename + "_timestep.npz"
    filename_iterations = basename + "_iterations.npz"
    os.remove(filename_timestep)
    os.remove(filename_iterations)
    pass


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