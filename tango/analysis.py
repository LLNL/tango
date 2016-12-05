"""
analysis

Module for performing data analysis on saved files.

See https://github.com/LLNL/tango for copyright and license information
"""



from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('qt5agg')  # workaround for a bug in recent Anaconda build of matplotlib
import matplotlib.pyplot as plt
from . import physics_to_H



        
class TimestepData(object):
    def __init__(self, basename):
        dataTimestep = load_NPZ_file(basename + "_timestep")
        dataIterations = load_NPZ_file(basename + "_iterations")
        
        # data for the whole timestep: save dict into member variables
        for varname in dataTimestep:
            setattr(self, varname, dataTimestep[varname])

        # also, store the dicts
        self.dataTimestep = dataTimestep     
        self.dataIterations = dataIterations
    def get_first_iteration(self):
        return IterationData(0, self.dataTimestep, self.dataIterations)
    def get_last_iteration(self):
        Niters = len(self.iterationNumber)
        return IterationData(Niters-1, self.dataTimestep, self.dataIterations)

    def get_nth_iteration(self, N):
        return IterationData(N, self.dataTimestep, self.dataIterations)
        
    def profile_nth_iteration(self, N):
        """Provide a shortcut to returning the Nth iteration of the profile without having to instantiate an Iteration object
        """
        return self.dataIterations['profile'][N, :]
        
        
    
    ################################################################
    ## Accessors to retrieve available data
    def available_timestep_fields(self):
        """return a list of all data for the whole timestep"""
        return self.dataTimestep.keys()
    
    def available_solution_fields(self):
        """Return a list of all attributes in solution data that is available (data that changes each iteration)"""
        return self.dataIterations.keys()
    
    @property
    def psi(self):
        """Alias to accessing x"""
        return self.x
    
    ### Plotting Routines
    def plot_err_history(self, savename=None):
        """Plot the self-consistency error vs. iteration number for this timestep.
        Optionally, save the figure to a file.
        """
        fig = plt.figure()
        plt.semilogy(self.iterationNumber, self.errHistory)
        plt.xlabel('Iteration Number')
        plt.ylabel('self-consistency error (rms)')
        if savename is not None:
            fig.savefig(savename, bbox_inches='tight')
        return fig
        

class IterationData(object):
    def __init__(self, iterationCounter, dataTimestep, dataIterations):
        """Create an object to represent the data from a single iteration.
        
        Inputs:
          iterationCounter      iteration to retrieve data from (integer)
          dataTimestep          constant or 0D data that may change each iteration (dict)
          dataIterations        1D array data that changes each iteration (dict)
        """
        self.iterationCounter = iterationCounter 
        self.dataTimestep = dataTimestep
        # data for the whole timestep--should always be there
            # scalars
        for varname in dataTimestep:
            # special handling for iterationNumber and errHistory where we want a particular iteration from the history
            if varname == 'iterationNumber':
                self.l = dataTimestep['iterationNumber'][iterationCounter]
            elif varname == 'errHistory':
                self.err = dataTimestep['errHistory'][iterationCounter]
            else:  # save everything else
                setattr(self, varname, dataTimestep[varname])
       
        
        self.data = {}
        # store all 1D data from the specified iteration 
        for varname in dataIterations:
            self.data[varname] = dataIterations[varname][iterationCounter, :]
            # also save into self as an attribute for direct access
            setattr(self, varname, self.data[varname])

        
    def available_timestep_fields(self):
        """return a list of all data for the whole timestep"""
        timestepFields = self.dataTimestep.keys()
        
        # rename a couple of fields
        if 'iterationNumber' in timestepFields:
            timestepFields.remove('iterationNumber')
            timestepFields.append('l')
        if 'errHistory' in timestepFields:
            timestepFields.remove('errHistory')
            timestepFields.append('err')
        return timestepFields
        
    def available_solution_fields(self):
        """Return a list of all attributes in solution data that is available (data that changes each iteration)"""
        return self.data.keys()
    
    def all_available_fields(self):
        """return a list of all attributes, excluding internal attributes, dicts, and methods.
        
        This list is almost identical to the union of AvailableTimestepFields and AvailableSolutionFields."""
        return [key for key, value in self.__dict__.items() if not key.startswith("__") and not type(value) is dict and not callable(value)]
    
    ## Accessors to retrieve available data
    @property
    def psi(self):
        """Alias to accessing x"""
        return self.x
    
    ################################################################
    ### Data Analysis Routines
    def geometrized_diffusion_coefficient(self):
        D = physics_to_H.H_to_geometrized_diffusion_coeff(self.H2, self.Vprime)
        return D
    
    def diffusion_coefficient(self):
        chi = physics_to_H.H_to_diffusivity(self.H2, self.Vprime, self.gradpsisq)
        return chi
        
    def geometrized_convection_coefficient(self):
        c = physics_to_H.H_to_geometrized_convection_coeff(self.H3, self.Vprime)
        return c
        
    def convection_coefficient(self):
        vbar = physics_to_H.H_to_convection_coeff(self.H3, self.Vprime, self.gradpsisq)
        return vbar
    
    ################################################################
    ### Analysis routines for data on the GENE grid
    ####
    
    ################################################################
    ### Plotting Routines
    def plot_profile(self, savename=None):
        """Plt the profile at this iteration.  Optionally, save the figure to a file."""
        fig=plt.figure()
        plt.plot(self.psi, self.profile)
        plt.xlabel(r'$\psi$')
        plt.ylabel('Profile')
        if savename is not None:
            fig.savefig(savename, bbox_inches='tight')
        return fig
    def plot_profile_and_starting_profile(self, savename=None):
        """Plot the profile at this iteration along with the profile at the beginning of the timestep.
        Optionally, save the figure to a file."""
        fig=plt.figure()
        plt.plot(self.psi, self.profile_mminus1, 'b-', label='At beginning of timestep')
        plt.plot(self.psi, self.profile, 'r-', label='After iteration {}'.format(self.l))
        plt.xlabel(r'$\psi$')
        plt.ylabel('Profile')
        plt.legend(loc='best')
        if savename is not None:
            fig.savefig(savename, bbox_inches='tight')
        return fig
 
def load_NPZ_file(input_filename):
    """Open a file (adding a .npz extension if not present), retrieve the data, then close the file
    
    Note that when numpy loads an npz file, it uses a lazy form of loading, where the storage for a specific
    variable is not retrieved until that variable is accessed.  For very large files, this can save time.
    Here, for simplicity, and because the files for Tango are not expected to be that large (less than ~1 GB),
    this routine loads all variables into memory in a new dict.
    
    Inputs:
      input_filename    name of .npz file with output from tango (string)
    Outputs:
      data              dict containing all saved data
    """
    # Add a .npz extension if not present
    filename = input_filename
    if not input_filename.lower().endswith('.npz'):
        filename = filename + '.npz'
        
    # load
    data = {}
    with np.load(filename) as npzfile:
        for key in npzfile.files:
            data[key] = npzfile[key]
    return data