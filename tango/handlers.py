"""
handlers

Flexible approach to dealing with data and files on disk during a Tango run.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
import h5py
import shutil
import os

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
            
    def set_parallel_environment(self, parallel=False, MPIrank=0):
        """Set the parameters in a parallel environment.
        
        Inputs:
          parallel              True when in a parallel environment (Boolean)
          MPIrank               MPI rank of the process (integer)
        """
        self.parallelEnvironment = parallel
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
        self.lastIterationDivisor = -1  # -1 ensures the first iteration (iterationNumber==0) is saved.
        
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


class SaveGeneOutputHandler(Handler):
    """Handler for saving the f1 checkpoint history at a desired interval.
    
    Under usual conditions, Tango runs GENE to save a checkpoint file for f1 called checkpoint_000 at the end
    of every GENE run.  It gets overwritten on each run.  To preserve the checkpoint history, we make a copy
    of this file, leaving the original intact so that GENE uses it on its next startup as its initial
    condition.
    
    Files printed out by GENE (ignoring the suffix _###)
        autopar*            (autoparallelization) * = ignore here
        checkpoint          distribution function f1.  Binary
        circular*           (magnetic geometry data)
        codemods*           (differences in code compared to repository)
        field               contains phi1 (and A1parallel, B1parallel in EM simulation).  Binary
        mom_ions            contains velocity-space moments of f1 including perturbed density, temperature.  Binary.
        nrg                 contains volume-averaged quantities.  ASCII
        parameters*         (output parameters file)
        profile_ions        contains the time-dependent output profiles (T,n) and flux (Gamma, Q), saved
                                instantaneously every istep_prof.  ASCII
        profiles_ions       contains the profiles (T, n) actually used for the GENE run.  ASCII
        s_checkpoint*       (safety checkpoint)
        srcmom_ions         contains the Krook source term used to maintain f0.  Binary
        vsp                 velocity-space data dependent on the z coordinate
    
    When saving a file like checkpoint_000, this will be renamed to tg_checkpoint_<iterationNumber>
    """
    def __init__(self, *geneFilenames, **kwargs):
        """Constructor.
        
        Inputs:
          geneFilenames         Gene output files to be saved (strings).  Should be from the names above,
                                    e.g., 'checkpoint_000', 'nrg_000', 'profiles_ions_000', ...
          kwargs
            iterationInterval
            diagdir             directory where GENE's output files are saved (string)
        """
        iterationInterval = kwargs.get('iterationInterval', np.inf)
        Handler.__init__(self, iterationInterval)
        self.diagdir = kwargs.get('diagdir', '')
        self.geneFilenames = geneFilenames
        self.prefix = 'tg_'  # prefix to be appended to files
        
        
    def execute(self, data, iterationNumber):
        """Copy the gene file to a new file"""
        for geneFilename in self.geneFilenames:
            destination = os.path.join(self.diagdir, self.prefix + self.strip_suffix(geneFilename) + '_' + str(iterationNumber))
            geneFilepath = os.path.join(self.diagdir, geneFilename)
            shutil.copyfile(geneFilepath, destination)
        
    def strip_suffix(self, geneFilename):
        """Strip the suffix _<###> from a filename like checkpoint_000.
        
        E.g.,
          strip_suffix('checkpoint_000') returns 'checkpoint'
          strip_suffix('profiles_ions_000') returns 'profiles_ions'
        
        Inputs:
          geneFilename      input filename (string)
        """
        return geneFilename.rsplit('_', 1)[0]


class TangoHistoryHandler(Handler):
    """Handler for incrementally saving the iteration history.
    
    Each time the Handler is executed, the data is added to the output h5 file.  Default basename=tango_history.
    
    This Handler is intended to save the iterations within a timestep, not across multiple timesteps.
    
    Similar to DataSaver in datasaver.py.  But DataSaver does not write data to file until the simulation is over,
    and DataSaver allows the user to specify a subset of data to save.  DataSaver also by default saves the data
    from every iteration rather than every N iterations.
    """
    def __init__(self, iterationInterval=np.inf, basename='tangodata', maxIterations=9000, initialData=None):
        """
        Inputs:
          iterationInterval     interval for executing the Handler's task (integer)
          basename              base of filename to save data to (string)
          maxIterations         maximum number of iterations Tango will use per timestep (integer)
          initialData           initial data to save at the beginning... dict with same hierarchical structure as the hdf5 file. (dict)
                                   Must have the attribute 'setNumber', referring to which set in a given Tango run this is (int)
        """
        Handler.__init__(self, iterationInterval)
        
        self.basename = basename
        self.maxCount = 1 + maxIterations // iterationInterval   # maximum number of iterations to store
        self.initialData = initialData
        
        self.setNumber = initialData['setNumber']
        self.filename = basename + '_s{}'.format(self.setNumber) + '.hdf5'
        
        # initialize data storage
        self.countStoredIterations = 0    # how many iterations have been stored so far
        # create a new hdf5 file
        self._create_file(initialData)
    
    def execute(self, data, iterationNumber):
        # open the hdf5 file and store the data
        with self._get_file() as f:
            if self.countStoredIterations == 0:
                self._initialize_datasets_on_first_use(f, data)
            # set the root data & metadata that updates each iteration
            index = self.countStoredIterations
            
            # save the 0D field data that updates each iteration
            varNames = (label for label in data if not isinstance(data[label], dict))
            for varName in varNames:
                dset = f[varName]
                dset.resize(index + 1, axis=0)
                dset[index] = data[varName]
            
            # save the 1D field data that updates each iteration.
            fieldlabels = (label for label in data if isinstance(data[label], dict))
            for label in fieldlabels:
                fielddata = data[label]
                for varName in fielddata:  # loop through keys of the data for a given field
                    dsetName = '/'.join((label, varName))
                    dset = f[dsetName]
                    dset.resize(index + 1, axis=0)
                    dset[index, :] = fielddata[varName]
                    
            # finish up
            self.countStoredIterations += 1
            f.attrs['writes'] = self.countStoredIterations
    
        
    def _get_file(self):
        """return the opened hdf5 file.  assume file is already created."""
        return h5py.File(self.filename, 'a')
        
    def _create_file(self, initialData):
        """Create the hdf5, create the groups, and store some initial metadata.
        
        Inputs:
          initialdata       dict with items to save, in the same structure as to be saved in the hdf5 file (dict)
        """
        # create file
        with h5py.File(self.filename, "w") as f:   # use mode w to overwrite and mode w- to fail if file exists
            # store initial metadata
            varNamesAtRoot = (varName for varName in initialData if not isinstance(initialData[varName], dict))
            for varName in varNamesAtRoot:
                f.attrs[varName] = initialData[varName]
            
            fieldlabels = (label for label in initialData if isinstance(initialData[label], dict))
            for label in fieldlabels:
                # create the group for each field
                grp = f.create_group(label)
                fieldData = initialData[label]
                for varName in fieldData:  # loop through keys of the data for a given field
                    grp.attrs[varName] = fieldData[varName]
        
    def _initialize_datasets_on_first_use(self, f, data):
        """Called on the first run of execute() to create the datasets.
        
        Inputs:
          f         an open hdf5 file
          data      dict containing all the data
        """
        # create the datasets for the 0D field data that updates each iteration
        varNamesAtRoot = (varName for varName in data if not isinstance(data[varName], dict))
        for varName in varNamesAtRoot:
            if isinstance(data[varName], np.int):
                f.create_dataset(varName, (0,), maxshape=(self.maxCount,), dtype=np.int)
            else: # assume float
                f.create_dataset(varName, (0,), maxshape=(self.maxCount,), dtype=np.float64)
        
        # create the datasets for the 1D field data that updates each iteration... these exist inside dicts for each field.
        fieldlabels = (label for label in data if isinstance(data[label], dict))
        for label in fieldlabels:
            fielddata = data[label]
            for varName in fielddata:  # loop through keys of the data for a given field
                dsetName = '/'.join((label, varName))
                Npts = len(fielddata[varName])
                # assume float type
                f.create_dataset(dsetName, (0, Npts), maxshape=(self.maxCount, Npts), dtype=np.float64)
    
    @staticmethod
    def set_up_initialdata(setNumber, xTango, xTurb, t, fields):
        """Helper function to package up the initial data (used in the constructor of this handler)."""
        initialData = {
                'setNumber': setNumber,
                'xTango': xTango,
                'xTurb': xTurb,
                't': t,
                'timestepNumber': 1}
        for field in fields:
            (EWMAParamTurbFlux, EWMAParamProfile) = field.lodestroMethod.get_ewma_params()
            initialData[field.label] = {
                    'EWMAParamProfile': EWMAParamProfile,
			'EWMAParamTurbFlux': EWMAParamTurbFlux,
			'profile_mminus1': field.profile_mminus1}
        return initialData
    
#    def reset_for_next_timestep(self):
#        """Reset to pristine status, for use in next timestep.
#        
#        ****NOT READY FOR USE."""
#        self.countStoredIterations = 0
#        self.iterationNumber = np.zeros(self.maxCount)