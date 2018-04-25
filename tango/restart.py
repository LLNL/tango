"""
restart

Module for providing restart capability to Tango.  So far, can restart *within* a timestep, but not for separate timesteps.

The module will read in a restart file (which is the .hdf5 file saved during the previous Tango run), load the pertinent
data, and provide the necessary startup to continue with the next Tango iteration.
"""

from __future__ import division, absolute_import
import numpy as np
import os
import h5py
import shutil

# given a restart file, read in the pertinent data
def read_metadata_from_previousfile(filename):
    """for now, assume the timestep is not changing upon a restart."""
    with h5py.File(filename, 'r') as f:
        # read some metadata
        old_setNumber = f.attrs['setNumber']
        old_last_iterationNumber = f['iterationNumber'][-1] # need to add one when starting the next one
        old_t = f.attrs['t']
        old_timestepNumber = f.attrs['timestepNumber']
        # read old profile_mminus1 for each field
        labels = get_labels_from_hdf5file(f)
        
        old_profiles_mminus1 = {}
        old_profiles = {}
        old_profilesEWMA = {}
        old_turbFluxesEWMA = {}
        for label in labels:
            grp = f[label]
            old_profiles_mminus1[label] = grp.attrs['profile_mminus1']
            old_profiles[label] = grp['profile'][-1]
            old_profilesEWMA[label] = grp['profileEWMATurbGrid'][-1]
            old_turbFluxesEWMA[label] = grp['fluxEWMATurbGrid'][-1]
            
    # new values for the restart
    setNumber = old_setNumber + 1
    startIterationNumber = old_last_iterationNumber + 1
    t = old_t
    timestepNumber = old_timestepNumber
    return (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turbFluxesEWMA)

def set_ewma_iterates(fields, old_profilesEWMA, old_turbFluxesEWMA):
    """Set the EWMA iterates in the fields for a restart.
    """
    for field in fields:
        label = field.label
        # set the EWMA iterates for each field
        field.lodestroMethod.set_ewma_iterates(old_profilesEWMA[label], old_turbFluxesEWMA[label])
        # set the mminus1 variable for each field (only important if more than one timestep is used)

#initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)

#set_up_initialdata_on_restart(setNumber, xTango, xTurb, t, fields)

# instantiate history handler with new setNumber (new initialdata)

# initialize solver
#solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, startIterationNumber=startIterationNumber, profiles=old_profiles)
# add the handler to solver

# return solver... or just run

# check if restart
def check_if_should_restart(basename=None):
    """Check if an output file is found that Tango can restart from.
    
    If one is found, return the filename.  If not, return None.
    
    Look for files in the current directory that have a name in the form <basename>_s<number>.hdf5.
    Of all the ones that exist, return the one with the largest number; this is the one that will be
    restarted from.
    
    If no basename is provided, and in the current working directory all .hdf5 files have the same
    basename, then that basename is used.
    """
    if basename is None:
        cwd = os.getcwd()
        files = os.listdir(cwd)
        # get a list of the hdf5 files in the current directory
        h5files = [f for f in files if f.endswith('.hdf5')]
        if not h5files:
            filename = None
        else:
            h5filestarts = [f[:f.rfind('_')] for f in h5files]
            if len(set(h5filestarts)) == 1:  # there is a unique startname
                basename = h5filestarts[0]
                filename = find_latest_savefile(basename)
            else:
                raise ValueError('No basename provided, and multiple basenames for hdf5 files exist in the current directory {}.'.format(cwd))
    else:
        filename = find_latest_savefile(basename)
    return filename
    
def find_latest_savefile(basename):
    """Find in the current directory the latest savefile with name beginning with basename.
    
    The savefiles have name in the form <basename>_s<number>.hdf5.  Return the one with
    the largest number.  If no hdf5 savefiles exist with this basename, return None
    """
    cwd = os.getcwd()
    files = os.listdir(cwd)
    start = basename + '_s'
    h5files = [f for f in files if f.endswith('.hdf5') and f.startswith(start)]
    if h5files:
        h5filesNoExtension = [f[:f.rfind('.')] for f in h5files]
        numberOnly = [int(f[f.rfind('_') + 2:]) for f in h5filesNoExtension]
        setNumber = max(numberOnly)
        filename = basename + '_s{}'.format(setNumber) + '.hdf5'
    else:
        filename = None
    return filename
    
def combine_savefiles(basename='tangodata'):
    """Combine several savefiles in the current directory of form <basename>_s<number>.hdf5 into one, named
    <basename>_combined.hdf5
    """
    outputFilename = basename + '_combined.hdf5'
    # get a list of the savefiles to combine
    cwd = os.getcwd()
    files = os.listdir(cwd)
    start = basename + '_s'
    h5files = [fname for fname in files if fname.endswith('.hdf5') and fname.startswith(start)]
    if not h5files:
        raise ValueError('no files found.')
    sortedh5files = sorted(h5files) # sorted list of filenames, in increasing number    
    
    # copy the first file, first
    firstFilename = sortedh5files[0]
    shutil.copyfile(firstFilename, outputFilename)
    
    with h5py.File(outputFilename) as f:
        # get list of datasets at root
        dsetlabels = get_datasets_in_group(f)
        # get list of field labels
        labels = get_labels_from_hdf5file(f)
        
        # delete some metadata that doesn't make sense for the combined file
        del f.attrs['setNumber']
        del f.attrs['writes']
        # loop over the rest of the save files
        for savefilename in sortedh5files[1:]:
            with h5py.File(savefilename, 'r') as fread:
                currentSize = len(f['iterationNumber'])
                writes = fread.attrs['writes']
                # at root: resize datasets and copy in the data from current save file
                for dsetlabel in dsetlabels:
                    dset = f[dsetlabel]
                    dset.resize(currentSize + writes, axis=0)
                    dset[currentSize:] = fread[dsetlabel]
                
                # now, loop over each field
                for label in labels:
                    # loop over each dataset in the field
                    grp = f[label]
                    grpread = fread[label]
                    datalabels = get_datasets_in_group(grp)
                    # resize datasets and copy in the data from the current save file
                    for datalabel in datalabels:
                        dset = grp[datalabel]
                        dset.resize(currentSize + writes, axis=0)
                        dset[currentSize:, :] = grpread[datalabel][:, :]
    
def get_labels_from_hdf5file(f):
    """return a list of the field labels in an hdf5 file.
    
    The list of field labels is determined by the groups that exist at the root level.
    
    Inputs:
      f         An open hdf5 file 
    """
    labels = [key for key in f.keys() if isinstance(f[key], h5py._hl.group.Group)]
    return labels
    
def get_datasets_in_group(grp):
    """return a list of the dataset labels (not groups) from a given group in an hdf5 file.
    
    Inputs:
      grp           group of an open hdf5 file
    Outputs:
      dsetlabels    dataset names at group (list)
    """
    dsetlabels = [key for key in grp.keys() if isinstance(grp[key], h5py._hl.dataset.Dataset)]
    return dsetlabels
        