# test_datasaver

from __future__ import division
import numpy as np
import os
from tango import datasaver

def test_AddData():
    MaxIterations=15
    NumPts = 20
    arrays_to_save = ['H2', 'H3']
    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
    H2 = np.linspace(1, 2, NumPts)
    H3 = np.linspace(4, 5, NumPts)
    H4 = np.linspace(4, 5, 30)
    data = {'H2': H2, 'H3': H3, 'H4': H4}
    DataSaver.AddData(data, 1)
    DataSaver.AddData(data, 2)
    
    assert DataSaver.counter == 2
    assert DataSaver.data_all_iterations['H3'][1, 19] == 5
    assert DataSaver.iteration_number[0] == 1

def test_FinalizeData():
    MaxIterations=15
    NumPts = 20
    arrays_to_save = ['H2', 'H3']
    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
    H2 = np.linspace(1, 2, NumPts)
    H3 = np.linspace(4, 5, NumPts)
    H4 = np.linspace(4, 5, NumPts)
    data = {'H2': H2, 'H3': H3, 'H4': H4}
    DataSaver.AddData(data, 1)
    DataSaver.AddData(data, 2)
    
    assert DataSaver.data_all_iterations['H2'].shape == (MaxIterations, NumPts)
    DataSaver._FinalizeData()
    assert DataSaver.data_all_iterations['H2'].shape == (2, NumPts)
    assert len(DataSaver.iteration_number) == 2

    DataSaver.finalized == True
    
def test_SaveToFile():
    MaxIterations=15
    NumPts = 20
    arrays_to_save = ['H2', 'H3']
    psi = np.linspace(0, 1, NumPts)
    Vprime = np.ones(NumPts)
    one_off_data = {'psi': psi, 'Vprime': Vprime}
    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
    DataSaver.AddOneOffData(one_off_data)
    H2 = np.linspace(1, 2, NumPts)
    H3 = np.linspace(4, 5, NumPts)
    data = {'H2': H2, 'H3': H3}
    DataSaver.AddData(data, 1)
    DataSaver.AddData(data, 2)
    
    DataSaver.SaveToFile('testsave')
    
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
    MaxIterations=15
    NumPts = 20
    arrays_to_save = ['H2', 'H3']
    DataSaver = datasaver.dataSaver(MaxIterations, arrays_to_save)
    H2 = np.linspace(1, 2, NumPts)
    H3 = np.linspace(4, 5, NumPts)
    data = {'H2': H2, 'H3': H3}
    DataSaver.AddData(data, 1)
    DataSaver.AddData(data, 2)
    
    assert np.all(DataSaver.data_all_iterations['H2'][1,:] == H2)
    DataSaver.ResetForNextTimestep()
    assert DataSaver.counter == 0
    assert DataSaver.finalized == False
    assert DataSaver.one_off_data == {}
    assert np.all(DataSaver.iteration_number == np.zeros(MaxIterations))
    assert DataSaver.data_all_iterations == {}
