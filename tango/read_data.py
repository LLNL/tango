"""
read_data

read in data from files.  For example, files saved by the Handlers or by the DataSaver.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division, absolute_import
import numpy as np
import os
from . import tango_logging

def read_tango_checkpoint(**kwargs):
    """Read in the files saved by TangoCheckpointHandler
    
    Two files to read: 
        <basename>_prof.txt, containing the current iteration number, the Tango radial grid, and the current profile data
        <basename>_ewma,txt, containing the turbulence radial grid, the EWMA profile, and the EWMA turbulent flux

    If no basename is provided and in the current working directory only one file ends in _prof.txt and only one file ends
    in _ewma.txt, those two files are read in.
    
    Data is returned as a collection of arrays
    
    Inputs:
      basename      Name specifying checkpoint files (optional keyword argument) (string)
        
    Outputs:
      iterationNumber           iteration number of the checkpoint (integer)
      xTango                    radial grid for Tango (array)
      pressureProfile           pressure profile at this iteration on Tango radial grid (array)
      xTurbGrid                 radial grid on turbulence grid (array)
      profileEWMATurbGrid       EWMA pressure profile for this iteration on turbulence radial grid (array)
      fluxEWMATurbGrid          EWMA turbulent flux for this iteration on turbulence radial grid (array)
    """
    if 'basename' not in kwargs:   # no basename specified for checkpoint files
        cwd = os.getcwd()
        files = os.listdir(cwd)
        # look to see how many files in the current directory end in _prof.txt and _ewma.txt
        filesProf = [f for f in files if f.endswith('_prof.txt')]
        filesEWMA = [f for f in files if f.endswith('_ewma.txt')]
        if len(filesProf) == 1 and len(filesEWMA) == 1:
            # read in these files
            fileProf = filesProf[0]
            fileEWMA = filesEWMA[0]
            (iterationNumber, xTango, pressureProfile, xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid) = read_tango_checkpoint_files(fileProf, fileEWMA)
        else:
            tango_logging.log('No basename provided to read_tango_checkpoint and unique checkpoint files in the current directory {} cannot be found.'.format(cwd))
            raise ValueError
    else:  # basename is specified
        fileProf = kwargs['basename'] + '_prof.txt'
        fileEWMA = kwargs['basename'] + '_ewma.txt'
        (iterationNumber, xTango, pressureProfile, xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid) = read_tango_checkpoint_files(fileProf, fileEWMA)
    return (iterationNumber, xTango, pressureProfile, xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid)
            
            
def read_tango_checkpoint_files(fileProf, fileEWMA):
    """Helper file for read_tango_checkpoint()
    
    The profile file consists of [see handlers.py]:
        Header: current iteration number preceded by a # symbol
        1st column: tango radial grid
        2nd column: pressure profile
        
    The EWMA file consists of:
        1st column: turbulence radial grid
        2nd column: relaxed pressure profile (EWMA) on the turbulence grid
        3rd column: relaxed turbulent heat flux profile (EWMA) on the turbulence grid
        
    Read all of these and output them.
    
    Inputs:
        fileProf      path to Profile file (string)
        fileEWMA      path to EWMA file (string)
    """
    # read fileProf    
    # get the header:
    with open(fileProf, 'r') as f:
        firstLine = f.readline().strip().lstrip('# ') # remove trailing newline and beginning comment character
    iterationNumber = int(firstLine)
    
    # read the data from fileProf
    (xTango, pressureProfile) = np.loadtxt(fileProf, unpack=True)
    
    # read fileEWMA
    (xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid) = np.loadtxt(fileEWMA, unpack=True)

    return (iterationNumber, xTango, pressureProfile, xTurbGrid, profileEWMATurbGrid, fluxEWMATurbGrid)
    

def read_tango_history(basename='tango_history'):
    """Read in the files saved by TangoHistoryHandler
    
    TangoHistoryHandler saves two files,
        <basename>_timestep.npz             0D data
        <basename>_iterations.npz           1D data that changes each iteration
        
    By default, basename=tango_history
    
    Data is returned as two dicts, one for the 0D data and one for the 1D data.
    
    Inputs:
      basename      base filename for Tango history files (string)
    Outputs:
      data0D        saved data that is 0D (dict)
      data1D        saved data that is 1D (dict)
    """
    data0D = load_npz_file(basename + "_timestep")
    data1D = load_npz_file(basename + "_iterations")
    return (data0D, data1D)

def load_npz_file(input_filename):
    """Open a numpy data file (adding a .npz extension if not present), retrieve the data, then close the file
    
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