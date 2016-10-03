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
    def __init__(self, filename):
        data = LoadNPZFile(filename)
        
        # data for the whole timestep--should always be there
            # scalars (saved as 0 dimensional arrays)
        self.m = data['m']
        self.t = data['t']
            # arrays
        self.iteration_number = data['iteration_number']   # array
        self.errhistory = data['errhistory']
        self.x = data['x']
        self.profile_m = data['profile_m']
        self.profile_mminus1 = data['profile_mminus1']
        
        self.whole_timestep_keys = ['m', 't', 'iteration_number', 'errhistory', 'x', 'profile_m', 'profile_mminus1']

        # store all data from all iterations        
        self.data = data
    def GetFirstIteration(self):
        return IterationData(0, self.data, self.whole_timestep_keys)
        pass
    def GetLastIteration(self):
        Niters = len(self.iteration_number)
        return IterationData(Niters-1, self.data, self.whole_timestep_keys)

    def GetNthIteration(self, N):
        return IterationData(N, self.data, self.whole_timestep_keys)
    
    ################################################################
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
    def __init__(self, iteration_counter, data, whole_timestep_keys):
        """Create an object to represent the data from a single iteration.
        
        Inputs:
          iteration_counter     iteration to retrieve data from (integer)
          data                  data that is output and saved by tango (dict)
          whole_timestep_keys   list of strings of keys to dict data that represent information about the timestep
                                  as a whole rather than a single iteration
        """
        self.iteration_counter = iteration_counter 
        # data for the whole timestep--should always be there
            # scalars
        self.m = data['m']
        self.t = data['t']
        self.l = data['iteration_number'][iteration_counter]   #
        self.err = data['errhistory'][iteration_counter]
            # arrays
        self.x = data['x']
        self.profile_mminus1 = data['profile_mminus1']
        
        self.data = {}
        
        #data for the whole timestep -- that might be there
        
        
        # store all data from the specified iteration
        for varname in data:
            if varname not in whole_timestep_keys: # in this loop, only store variables that change per iteration
                self.data[varname] = data[varname][iteration_counter, :]

    def _SaveInternallyIfPresent(self, data, key):
        
        
    def AvailableSolutionFields(self):
        return self.data.keys()
    
    ## Accessors to retrieve available data
    @property
    def psi(self):
        """Alias to accessing x"""
        return self.x
    @property
    def profile(self):
        key = 'profile'
        return self.ReturnDataFromKey(key)
    @property
    def H1(self):
        key = 'H1'
        return self.ReturnDataFromKey(key)
    @property
    def H2(self):
        key = 'H2'
        return self.ReturnDataFromKey(key)
    @property
    def H3(self):
        key = 'H3'
        return self.ReturnDataFromKey(key)
    @property
    def H4(self):
        key = 'H4'
        return self.ReturnDataFromKey(key)
    @property
    def H6(self):
        key = 'H6'
        return self.ReturnDataFromKey(key)
    @property
    def H7(self):
        key = 'H7'
        return self.ReturnDataFromKey(key)
    @property
    def A(self):
        key = 'A'
        return self.ReturnDataFromKey(key)
    @property
    def B(self):
        key = 'B'
        return self.ReturnDataFromKey(key)
    @property
    def C(self):
        key = 'C'
        return self.ReturnDataFromKey(key)
    @property
    def f(self):
        key = 'f'
        return self.ReturnDataFromKey(key)
    @property
    def D(self):
        key = 'D'
        return self.ReturnDataFromKey(key)
    @property
    def c(self):
        key = 'c'
        return self.ReturnDataFromKey(key)
    @property
    def profile_turbgrid(self):
        key = 'profile_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def profileEWMA_turbgrid(self):
        key = 'profileEWMA_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def flux_turbgrid(self):
        key = 'flux_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def fluxEWMA_turbgrid(self):
        key = 'fluxEWMA_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def D_turbgrid(self):
        key = 'D_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def c_turbgrid(self):
        key = 'c_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def Dhat_turbgrid(self):
        key = 'Dhat_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def chat_turbgrid(self):
        key = 'chat_turbgrid'
        return self.ReturnDataFromKey(key)
    @property
    def theta_turbgrid(self):
        key = 'theta_turbgrid'
        return self.ReturnDataFromKey(key)
    
    def ReturnDataFromKey(self, key):
        if key in self.data:
            return self.data[key]
        else:
            print "variable " + key + " is not available"
            raise KeyError
    
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
    For simplicity, and because the files for Tango are not expected to be that large (less than ~1 GB), this
    routine loads all variables into memory in a new dict.
    
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