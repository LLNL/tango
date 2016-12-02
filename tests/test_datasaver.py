# test_datasaver

from __future__ import division
import numpy as np
import os
from tango import datasaver

def test_AddData():
    maxIterations=15
    numPts = 20
    arraysToSave = ['H2', 'H3']
    dataSaver = datasaver.DataSaver(maxIterations, arraysToSave)
    H2 = np.linspace(1, 2, numPts)
    H3 = np.linspace(4, 5, numPts)
    H4 = np.linspace(4, 5, 30)
    data = {'H2': H2, 'H3': H3, 'H4': H4}
    dataSaver.add_data(data, 1)
    dataSaver.add_data(data, 2)
    
    assert dataSaver.countStoredIterations == 2
    assert dataSaver.dataAllIterations['H3'][1, 19] == 5
    assert dataSaver.iterationNumber[0] == 1

def test_FinalizeData():
    maxIterations=15
    numPts = 20
    arraysToSave = ['H2', 'H3']
    dataSaver = datasaver.DataSaver(maxIterations, arraysToSave)
    H2 = np.linspace(1, 2, numPts)
    H3 = np.linspace(4, 5, numPts)
    H4 = np.linspace(4, 5, numPts)
    data = {'H2': H2, 'H3': H3, 'H4': H4}
    dataSaver.add_data(data, 1)
    dataSaver.add_data(data, 2)
    
    assert dataSaver.dataAllIterations['H2'].shape == (maxIterations, numPts)
    dataSaver._finalize_data()
    assert dataSaver.dataAllIterations['H2'].shape == (2, numPts)
    assert len(dataSaver.iterationNumber) == 2

    dataSaver.finalized == True
    
def test_SaveToFile():
    maxIterations=15
    numPts = 20
    arraysToSave = ['H2', 'H3']
    psi = np.linspace(0, 1, numPts)
    Vprime = np.ones(numPts)
    oneOffData = {'psi': psi, 'Vprime': Vprime}
    dataSaver = datasaver.DataSaver(maxIterations, arraysToSave)
    dataSaver.add_one_off_data(oneOffData)
    H2 = np.linspace(1, 2, numPts)
    H3 = np.linspace(4, 5, numPts)
    data = {'H2': H2, 'H3': H3}
    dataSaver.add_data(data, 1)
    dataSaver.add_data(data, 2)
    
    dataSaver.save_to_file('testsave')
    
    with np.load('testsave_timestep.npz') as npzfile:
    # check the one_off_data got saved and then check H2 H3
        assert np.all(npzfile['psi'] == psi)
        assert np.all(npzfile['Vprime'] == Vprime)
    with np.load('testsave_iterations.npz') as npzfile:
        assert np.all(npzfile['H2'][1,:] == H2)
        assert np.all(npzfile['H3'][0,:] == H3)
    
    # teardown
    os.remove('testsave_timestep.npz')
    os.remove('testsave_iterations.npz')

def test_ResetData():
    maxIterations=15
    numPts = 20
    arraysToSave = ['H2', 'H3']
    dataSaver = datasaver.DataSaver(maxIterations, arraysToSave)
    H2 = np.linspace(1, 2, numPts)
    H3 = np.linspace(4, 5, numPts)
    data = {'H2': H2, 'H3': H3}
    dataSaver.add_data(data, 1)
    dataSaver.add_data(data, 2)
    
    assert np.all(dataSaver.dataAllIterations['H2'][1,:] == H2)
    dataSaver.reset_for_next_timestep()
    assert dataSaver.countStoredIterations == 0
    assert dataSaver.finalized == False
    assert dataSaver.oneOffData == {}
    assert np.all(dataSaver.iterationNumber == np.zeros(maxIterations))
    assert dataSaver.dataAllIterations == {}
