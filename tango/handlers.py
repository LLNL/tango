"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import shutil

class Executor(object):
    """Coordinates execution of tasks of Handlers, whose primary purpose is dealing with
    files on disk and saving data.
    
    Also works in an MPI environment.  Only a single process should execute the tasks to save
    files to disk, in order to prevent multiple processes from trying to access the same file.
    This is accomplished through the use of an MPI rank, and only the process with MPIrank == 0
    will carry out the tasks.
    """    
    def __init__(self):
        self.handlers = []
        self.parallelEnvironment = False
        self.MPIrank = None
    
    def add_handler(self, handler):
        self.handlers.append(handler)
    
    def scheduled_handlers(self, iterationNumber):
        """Return a list of handlers scheduled to execute on this iteration number."""
        scheduledHandlers = []
        for handler in self.handlers:
            # Get divisor of current iteration
            iterationDivisor = iterationNumber // handler.iterationInterval
            # Check if handler should run 
            if iterationDivisor > handler.lastIterationDivisor:
                scheduledHandlers.append(handler)
                # update divisor
                handler.lastIterationDivisor = iterationDivisor
        return scheduledHandlers
    
    def execute_scheduled(self, data, iterationNumber):
        """Wrapper for handler execution that determines what to do in a parallel environment"""
        if self.serial_or_rank0():
            self.execute_scheduled_do(data, iterationNumber)
        
    def execute_scheduled_do(self, data, iterationNumber):
        """Execute the handlers scheduled for this iteration number.
        
        Inputs:
          data              complete data for this iteration (dict)
          iterationNumber   current iteration number within a timestep (integer)
        """
        scheduledHandlers = self.scheduled_handlers(iterationNumber)
        for handler in scheduledHandlers:
            handler.execute(data, iterationNumber)
    
    def reset_handlers_for_next_timestep(self):
        """Perform any cleanup of handlers for the next timestep."""
        for handler in self.handlers:
            handler.reset_for_next_timestep()
            
    def set_parallel_environment(self, parallelEnvironment, MPIrank):
        """Set the parameters in a parallel environment.
        
        Inputs:
          parallelEnvironment   True when in a parallel environment (Boolean)
          MPIrank               rank returned from GENE (integer)
        """
        self.parallelEnvironment = parallelEnvironment
        self.MPIrank = MPIrank
        
    def serial_or_rank0(self):
        """Return true if in a nonparallel environment or has MPIrank==0, otherwise False"""
        if self.parallelEnvironment == False:
            return True
        else:
            if self.MPIrank==0:
                return True
        return False
        

class Handler(object):
    """Abstract superclass for Handlers.

    iterationInterval is the interval for executing the Handler's task.  If iterationalInterval=50,
        then the Handler will execute every 50 iterations (on iteration 50, 100, 150, etc.)
    
    lastIterationDivisor tracks when the Handler should execute.  The Executor computes
        iterationNumber // iterationInterval (note the integer division).  When iterationNumber is
        large enough such that the result exceeds lastIterationDivisor, then it executes the Handler's
        task and increments lastIterationDivisor.
    """
    def __init__(self, iterationInterval=np.inf):
        self.iterationInterval = iterationInterval
        self.lastIterationDivisor = 0
        
    def execute(self, data, iterationNumber):
        """Subclasses must implement this.
        
        Inputs:
          data              complete dataset from current interation (dict)
          iterationNumber   current iteration number (integer)
        """
        
        pass
    
    def reset_for_next_timestep(self):
        """If cleanup necessary, subclasses should implement this."""
        pass


class Savef1HistoryHandler(Handler):
    """Handler for saving the f1 checkpoint history at a desired interval.
    
    Under usual conditions, Tango runs GENE to save a checkpoint file for f1 called checkpoint_000 at the end
    of every GENE run.  It gets overwritten on each run.  To preserve the checkpoint history, we make a copy
    of this file, leaving the original intact so that GENE uses it on its next startup as its initial
    condition.
    """
    def __init__(self, iterationInterval=np.inf, basename='f1_iteration_history', genefile='checkpoint_000'):
        """blah"""
        Handler.__init__(self, iterationInterval)
        self.basename = basename
        self.genefile = genefile
        
    def execute(self, data, iterationNumber):
        """Copy the gene file to a new file"""
        destination = self.basename + '_' + str(iterationNumber)
        shutil.copyfile(self.genefile, destination)
        

class TangoCheckpointHandler(Handler):
    """Handler for writing out a checkpoint for Tango.
    
    Print two files, both in ASCII: Using the default basename=tango_checkpoint, these are
      tango_checkpoint_prof.txt and tango_checkpoint_ewma.txt.  Will overwrite the previous
      checkpoint file whenever a new checkpoint is written.
        
        tango_checkpoint_prof.txt consists of:
          Header: current iteration number
          1st column: tango radial grid
          2nd column: pressure profile
         
        tango_checkpoint_ewma.txt consists of:
          1st column: turbulence radial grid
          2nd column: relaxed pressure profile (EWMA) on the turbulence grid
          3rd column: relaxed turbulent heat flux profile (EWMA) on the turbulence grid
    """
    def __init__(self, iterationInterval=np.inf, basename='tango_checkpoint'):
        """blah"""
        Handler.__init__(self, iterationInterval)
        
        # add checking of filename?
        self.basename = basename
        
    def execute(self, data, iterationNumber):
        """Print out the essential data to restart a Tango run."""
        filename_prof = self.basename + '_prof.txt'
        xTango = data['x']
        pressureProfile = data['profile']
        np.savetxt(filename_prof, np.transpose([xTango, pressureProfile]), header=str(iterationNumber))
        
        filename_ewma = self.basename + '_ewma.txt'
        xTurbGrid = data['xTurbGrid']
        profileEWMATurbGrid = data['profileEWMATurbGrid']
        fluxEWMATurbGrid = data['fluxEWMATurbGrid']
        np.savetxt(filename_ewma, np.transpose([xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid]))
    
class TangoHistoryHandler(Handler):
    """Handler for incrementally saving the iteration history, to preserve data in case the simulation crashes.
    
    Each time the Handler is executed, add the complete data dict into internal memory.  Save to <basename>.npz,
    overwriting any previous <basename>.npz.  At the first interval, one iteration's worth of data is saved.  At
    the second interval, two iterations' worth of data is saved, and so on.
    
    Note that this is intended to save the iterations within a timestep, not across multiple timesteps.
    
    Similar to DataSaver in datasaver.py.  But DataSaver does not write data to file until the simulation is over,
    and DataSaver allows the user to specify a subset of data to save.  DataSaver also by default saves the data
    from every iteration rather than every N iterations.
    """
    def __init__(self, iterationInterval=np.inf, basename='tango_history', maxIterations=9000):
        """
        Inputs:
          iterationInterval     interval for executing the Handler's task (integer)
          basename              base of filename to save data to (string)
          maxIterations         maximum number of iterations Tango will use per timestep (integer)
        """
        Handler.__init__(self, iterationInterval)
        
        # add checking of filename?
        self.basename = basename
        self.maxCount = 1 + maxIterations // iterationInterval   # maximum number of iterations to store
        
        # initialize data storage
        self.countStoredIterations = 0    # how many iterations have been stored so far
        self.iterationNumber = np.zeros(self.maxCount)
        self.data1D = {}
        self.data0D = {}
    
    def execute(self, data, iterationNumber):
        # add the data to internal storage
        self.add_data(data, iterationNumber)
        
        # save data to disk
        (data0D, data1D) = self.prepare_to_write_data()
        self.save_to_file(data0D, data1D)
    
    def add_data(self, data, iterationNumber):
        """Add the data to the internal storage
        
        Inputs:
          data             dict with a bunch of data (1D arrays and scalars)
          iterationNumber  iteration number l within a timestep (scalar)
        """
        self.iterationNumber[self.countStoredIterations] = iterationNumber
        for varName in data: # loop through keys
            if np.size(data[varName]) > 1:   # 1D arrays... assume nothing 2D or higher
                # initialize storage on first use
                if self.countStoredIterations==0:  
                    Npts = len(data[varName])
                    self.data1D[varName] = np.zeros((self.maxCount, Npts))
                
                # save the data into the container
                self.data1D[varName][self.countStoredIterations, :] = data[varName]
            elif np.size(data[varName]) == 1:   # scalars
                # initialize storage on first use
                if self.countStoredIterations==0:
                    self.data0D[varName] = np.zeros(self.maxCount)
                
                # save the data into the container
                self.data0D[varName][self.countStoredIterations] = data[varName]
        self.countStoredIterations += 1
    
    def prepare_to_write_data(self):
        """Copy data to a separate dict and remove unused elements of the data (the preallocated memory)"""
        iterationNumberReduced = self.iterationNumber[0:self.countStoredIterations]
        data1DReduced = {}
        data0DReduced = {}
        for arrayName in self.data1D:
            data1DReduced[arrayName] = self.data1D[arrayName][0:self.countStoredIterations, :]
        for varName in self.data0D:
            data0DReduced[varName] = self.data0D[varName][0:self.countStoredIterations]
        data0DReduced['iterationNumber'] = iterationNumberReduced
        return (data0DReduced, data1DReduced)
    
    def save_to_file(self, data0D, data1D):
        """Save the data to disk.
        
        Save two files: 
          The first file (ending in _timestep) is for the data that is for 0D data that changes on iterations (e.g., rms error)
          The second file (ending in _iterations) is for the 1D data that changes each iteration (e.g., H2, profile)
        """
        # save data whole-timestep data
        filename_timestep = self.basename + "_timestep"
        np.savez(filename_timestep, **data0D)
        
        # save 1D data that changes each iteration
        filename_iterations = self.basename + "_iterations"
        np.savez(filename_iterations, **data1D)
        
    
    def reset_for_next_timestep(self):
        """Reset to pristine status, for use in next timestep."""
        self.countStoredIterations = 0
        self.iterationNumber = np.zeros(self.maxCount)
        self.data1D = {}
        self.data0D = {}