"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import physics_to_H

"""
analysis

Module for performing data analysis on saved files
"""

        
class TimestepData(object):
    def __init__(self, basename):
        data_timestep = LoadNPZFile(basename + "_timestep")
        data_iterations = LoadNPZFile(basename + "_iterations")
        
        # data for the whole timestep
        for varname in data_timestep:
            setattr(self, varname, data_timestep[varname])

        # also, store the dicts
        self.data_timestep = data_timestep     
        self.data_iterations = data_iterations
    def GetFirstIteration(self):
        return IterationData(0, self.data_timestep, self.data_iterations)
        pass
    def GetLastIteration(self):
        Niters = len(self.iteration_number)
        return IterationData(Niters-1, self.data_timestep, self.data_iterations)

    def GetNthIteration(self, N):
        return IterationData(N, self.data_timestep, self.data_iterations)
        
    def Profile_NthIteration(self, N):
        """Provide a shortcut to returning the Nth iteration of the profile without having to instantiate an Iteration object
        """
        return self.data_iterations['profile'][N, :]
        
        
    
    ################################################################
    ## Accessors to retrieve available data
    def AvailableTimestepFields(self):
        """return a list of all data for the whole timestep"""
        return self.data_timestep.keys()
    
    def AvailableSolutionFields(self):
        """Return a list of all attributes in solution data that is available (data that changes each iteration)"""
        return self.data_iterations.keys()
    
    @property
    def psi(self):
        """Alias to accessing x"""
        return self.x
    
    ### Plotting Routines
    def PlotErrhistory(self, savename=None):
        """Plot the self-consistency error vs. iteration number for this timestep.
        Optionally, save the figure to a file.
        """
        fig = plt.figure()
        plt.semilogy(self.iteration_number, self.errhistory)
        plt.xlabel('Iteration Number')
        plt.ylabel('self-consistency error (rms)')
        if savename is not None:
            fig.savefig(savename, bbox_inches='tight')
        return fig
        

class IterationData(object):
    def __init__(self, iteration_counter, data_timestep, data_iterations):
        """Create an object to represent the data from a single iteration.
        
        Inputs:
          iteration_counter     iteration to retrieve data from (integer)
          data_timestep         constant or 0D data that may change each iteration (dict)
          data_iterations       1D array data that changes each iteration (dict)
        """
        self.iteration_counter = iteration_counter 
        self.data_timestep = data_timestep
        # data for the whole timestep--should always be there
            # scalars
        for varname in data_timestep:
            # special handling for iteration_number and errhistory where we want a particular iteration from the history
            if varname == 'iteration_number':
                self.l = data_timestep['iteration_number'][iteration_counter]
            elif varname == 'errhistory':
                self.err = data_timestep['errhistory'][iteration_counter]
            else:  # save everything else
                setattr(self, varname, data_timestep[varname])
       
        
        self.data = {}
        # store all 1D data from the specified iteration 
        for varname in data_iterations:
            self.data[varname] = data_iterations[varname][iteration_counter, :]
            # also save into self as an attribute for direct access
            setattr(self, varname, self.data[varname])

        
    def AvailableTimestepFields(self):
        """return a list of all data for the whole timestep"""
        timestep_fields = self.data_timestep.keys()
        if 'iteration_number' in timestep_fields:
            timestep_fields.remove('iteration_number')
            timestep_fields.append('l')
        if 'errhistory' in timestep_fields:
            timestep_fields.remove('errhistory')
            timestep_fields.append('err')
        return timestep_fields
        
    def AvailableSolutionFields(self):
        """Return a list of all attributes in solution data that is available (data that changes each iteration)"""
        return self.data.keys()
    
    def AllAvailableFields(self):
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
    def GeometrizedDiffusionCoefficient(self):
        D = physics_to_H.HToGeometrizedDiffusionCoeff(self.H2, self.Vprime)
        return D
    
    def DiffusionCoefficient(self):
        chi = physics_to_H.HToDiffusivity(self.H2, self.Vprime, self.gradpsisq)
        return chi
        
    def GeometrizedConvectionCoefficient(self):
        c = physics_to_H.HToGeometrizedConvectionCoeff(self.H3, self.Vprime)
        return c
        
    def ConvectionCoefficient(self):
        vbar = physics_to_H.HToConvectionCoeff(self.H3, self.Vprime, self.gradpsisq)
        return vbar
    
    ################################################################
    ### Analysis routines for data on the GENE grid
    ####
    
    ################################################################
    ### Plotting Routines
    def PlotProfile(self, savename=None):
        """Plt the profile at this iteration.  Optionally, save the figure to a file."""
        fig=plt.figure()
        plt.plot(self.psi, self.profile)
        plt.xlabel(r'$\psi$')
        plt.ylabel('Profile')
        if savename is not None:
            fig.savefig(savename, bbox_inches='tight')
        return fig
    def PlotProfileAndStartingProfile(self, savename=None):
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
 
def LoadNPZFile(input_filename):
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