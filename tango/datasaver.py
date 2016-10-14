"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import logging

class dataSaverHandler(object):
    """convenient interface to a dataSavers."""
    def __init__(self):
        self.DataSaver = None
    def initialize_datasaver(self, basename, MaxIterations, arrays_to_save):
        """create a dataSaver and add to collection"""
        self.basename = basename
        self.DataSaver = dataSaver(MaxIterations, arrays_to_save)
        #DataSaver = dataSaver(**kw)
        #self.datasavers.append(DataSaver)
        return self.DataSaver
    def add_data(self, input_data, iteration_num):
        if self.DataSaver is not None:
            self.DataSaver.AddData(input_data, iteration_num)
    def add_one_off_data(self, one_off_data):
        if self.DataSaver is not None:
            self.DataSaver.AddOneOffData(one_off_data)
    def save_to_file(self, m):
        if self.DataSaver is not None:
            filename = self.basename + str(m)
            self.DataSaver.SaveToFile(filename)
    def reset_for_next_timestep(self):
        if self.DataSaver is not None:
            self.DataSaver.ResetForNextTimestep()
        
    

class dataSaver(object):
    def __init__(self, MaxIterations, arrays_to_save):
        """
        Inputs:
          
          MaxIterations         Max # of iterations allowed per timestep
          arrays_to_save        list containing strings of data to save on each iteration -- acts as keys for a dict
        """
        self.MaxIterations = MaxIterations
        self.arrays_to_save = arrays_to_save
        
        
        # initialize data storage
        self.one_off_data = {}
        self.finalized = False
        self.counter = 0
        self.iteration_number = np.zeros(self.MaxIterations)
        self.data_all_iterations = {}
        #self.ResetForNextTimestep()
        
        
    def AddData(self, input_data, iteration_num):
        """
        Inputs:
            data            dict with a bunch of 1D arrays
            iteration_num   iteration number l (scalar)
        """
        
        # add a check if counter >= MaxIterations
        if self.finalized == False:        
            
            self.iteration_number[self.counter] = iteration_num
            for array_name in self.arrays_to_save:
                # initialize storage on first use
                if self.counter==0:  
                    Npts = len(input_data[array_name])
                    self.data_all_iterations[array_name] = np.zeros((self.MaxIterations, Npts))
                
                # save the data into the container
                self.data_all_iterations[array_name][self.counter, :] = input_data[array_name]
            
            self.counter += 1
        else:
            print("Object is in finalized state.  Cannot add data.")
        
    def _FinalizeData(self):
        """Perform final operations before writing the file.
        E.g., remove unused elements of the data (preallocated memory)
        """
        self.iteration_number = self.iteration_number[0:self.counter]
        for array_name in self.arrays_to_save:
            self.data_all_iterations[array_name] = self.data_all_iterations[array_name][0:self.counter, :]
        self.finalized = True
    
    def AddOneOffData(self, one_off_data):
        """Add one-off data to get saved (e.g., psi, Vprime) (dict)."""
        self.one_off_data = self._merge_two_dicts(self.one_off_data, one_off_data)
        #self.one_off_data = one_off_data

    @staticmethod            
    def _merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = x.copy()
        z.update(y)
        return z
    
    def SaveToFile(self, filename):
        """Save the requested data to files as individual arrays.
        
        Save two files: 
          The first file (ending in _timestep) is for the data that is constant throughout the timestep
          (e.g., x or psi) or for 0D data that changes on iterations (e.g., rms error)
          
          The second file (ending in _iterations) is for the 1D data that changes each iteration
          (e.g., H2, profile)
        """
        if self.finalized == False:
            self._FinalizeData()
            # save data whole-timestep data
            filename_timestep = filename + "_timestep"
            self.one_off_data['iteration_number'] = self.iteration_number
            
            logging.info("Saving timestep data to {}.npz".format(filename_timestep))
            np.savez(filename_timestep, **self.one_off_data)
            logging.info("... Saved!")
            
            # save 1D data that changes each iteration
            if self.data_all_iterations != {}:
                filename_iterations = filename + "_iterations"
                logging.info("Saving iterations data to {}.npz".format(filename_iterations))
                np.savez(filename_iterations, **self.data_all_iterations)
                logging.info("... Saved!")
            else:
                logging.info("Not saving any iterations data for this timestep.")
        else:
            print("Object is in finalized state.  Cannot save.")
        
    def ResetForNextTimestep(self):
        """Reset all of the data to zeros and the counter so that the object is fresh for the next
        timestep (but still assumed to save the next data)"""
        self.counter = 0
        self.finalized = False
        self.one_off_data = {}
        self.iteration_number = np.zeros(self.MaxIterations)
        self.data_all_iterations = {}
#        for array_name in self.arrays_to_save:
#            temp, Npts = np.shape(self.data_all_iterations[array_name])
#            self.data_all_iterations[array_name] = np.zeros((self.MaxIterations, Npts))
        