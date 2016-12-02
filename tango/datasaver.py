"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import logging

class DataSaverHandler(object):
    """convenient interface to a dataSavers."""
    def __init__(self):
        self.dataSaver = None
        self.parallelEnvironment = False
        self.MPIrank = None
        
    def initialize_datasaver(self, basename, maxIterations, arraysToSave):
        """create a dataSaver and add to collection"""
        self.basename = basename
        self.dataSaver = DataSaver(maxIterations, arraysToSave)
        #DataSaver = dataSaver(**kw)
        #self.datasavers.append(DataSaver)
        
    def add_data(self, inputData, iterationNum):
        if self.dataSaver is not None and self.serial_or_rank0():
                self.dataSaver.add_data(inputData, iterationNum)
                
    def add_one_off_data(self, oneOffData):
        if self.dataSaver is not None and self.serial_or_rank0():
            self.dataSaver.add_one_off_data(oneOffData)
            
    def save_to_file(self, m):
        if self.dataSaver is not None and self.serial_or_rank0():
            filename = self.basename + str(m)
            self.dataSaver.save_to_file(filename)
            
    def reset_for_next_timestep(self):
        if self.dataSaver is not None and self.serial_or_rank0():
            self.dataSaver.reset_for_next_timestep()
            
    def set_parallel_environment(self, parallelEnvironment, MPIrank):
        self.parallelEnvironment = parallelEnvironment
        self.MPIrank = MPIrank
        
    def serial_or_rank0(self):
        out = False        
        if self.parallelEnvironment == False:
            out = True
        else:
            if self.MPIrank == 0:
                out = True
        return out
        
        
    

class DataSaver(object):
    def __init__(self, maxIterations, arraysToSave):
        """
        Inputs:
          
          maxIterations         Max # of iterations allowed per timestep
          arraysToSave          list containing strings of data to save on each iteration -- acts as keys for a dict
        """
        self.maxIterations = maxIterations
        self.arraysToSave = arraysToSave
        
        
        # initialize data storage
        self.oneOffData = {}
        self.finalized = False
        self.countStoredIterations = 0  # how many iterations have been stored so far
        self.iterationNumber = np.zeros(self.maxIterations)
        self.dataAllIterations = {}
        #self.ResetForNextTimestep()
        
        
    def add_data(self, inputData, iterationNum):
        """
        Inputs:
            data            dict with a bunch of 1D arrays
            iterationNum   iteration number l (scalar)
        """
        
        # add a check if countStoredIterations >= maxIterations
        if self.finalized == False:        
            
            self.iterationNumber[self.countStoredIterations] = iterationNum
            for arrayName in self.arraysToSave:
                # initialize storage on first use
                if self.countStoredIterations==0:  
                    Npts = len(inputData[arrayName])
                    self.dataAllIterations[arrayName] = np.zeros((self.maxIterations, Npts))
                
                # save the data into the container
                self.dataAllIterations[arrayName][self.countStoredIterations, :] = inputData[arrayName]
            
            self.countStoredIterations += 1
        else:
            print("Object is in finalized state.  Cannot add data.")
        
    def _finalize_data(self):
        """Perform final operations before writing the file.
        E.g., remove unused elements of the data (preallocated memory)
        """
        self.iterationNumber = self.iterationNumber[0:self.countStoredIterations]
        for arrayName in self.arraysToSave:
            self.dataAllIterations[arrayName] = self.dataAllIterations[arrayName][0:self.countStoredIterations, :]
        self.finalized = True
    
    def add_one_off_data(self, oneOffData):
        """Add one-off data to get saved (e.g., psi, Vprime) (dict)."""
        self.oneOffData = self._merge_two_dicts(self.oneOffData, oneOffData)

    @staticmethod            
    def _merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = x.copy()
        z.update(y)
        return z
    
    def save_to_file(self, filename):
        """Save the requested data to files as individual arrays.
        
        Save two files: 
          The first file (ending in _timestep) is for the data that is constant throughout the timestep
          (e.g., x or psi) or for 0D data that changes on iterations (e.g., rms error)
          
          The second file (ending in _iterations) is for the 1D data that changes each iteration
          (e.g., H2, profile)
        """
        if self.finalized == False:
            self._finalize_data()
            # save data whole-timestep data
            filenameTimestep = filename + "_timestep"
            self.oneOffData['iterationNumber'] = self.iterationNumber
            
            logging.info("Saving timestep data to {}.npz".format(filenameTimestep))
            np.savez(filenameTimestep, **self.oneOffData)
            logging.info("... Saved!")
            
            # save 1D data that changes each iteration
            if self.dataAllIterations != {}:
                filenameIterations = filename + "_iterations"
                logging.info("Saving iterations data to {}.npz".format(filenameIterations))
                np.savez(filenameIterations, **self.dataAllIterations)
                logging.info("... Saved!")
            else:
                logging.info("Not saving any iterations data for this timestep.")
        else:
            print("Object is in finalized state.  Cannot save.")
        
    def reset_for_next_timestep(self):
        """Reset all of the data to zeros and the counter so that the object is fresh for the next
        timestep (but still assumed to save the next data)"""
        self.countStoredIterations = 0
        self.finalized = False
        self.oneOffData = {}
        self.iterationNumber = np.zeros(self.maxIterations)
        self.dataAllIterations = {}
        