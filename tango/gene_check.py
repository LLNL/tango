"""
gene_check

Purposes:
1. Very quick run of GENE to make sure that it works
2. Return an MPI rank from GENE into Tango.  This is because due to some technical issues in the Python-Fortran
     coupling on the NERSC machines, we do not have MPI-awareness within the Python code.  Yet when the Python
     code is initialized in a parallel job, each processor runs independently runs the code.  In order to ensure
     that some actions, in particular writing to file, are handled by only a single process, GENE returns an
     integer that effectively functions as an MPI  rank.  The Python code can then ensure that some actions only
     occur on the process with rank==0.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
from . import genecomm_lowlevel
import os
import glob
import time

def gene_check():
    """Perform checking to make sure GENE works.
    
    Outputs:
      status        Status code (not currently being used)
      MPIrank       An MPI rank, unique for each process, returned from GENE (integer)
    """
    
    N = 64
    rho = np.linspace(0.1, 0.9, N)
    simulationTime = 0.3

    # set a few profiles    
    rho0 = 0.5
    kappa_T = 6.96
    kappa_n = 2.23
    aspr_in=0.36     #renormalization factor from minor to major radius
    temperatureHat = np.exp(-kappa_T * aspr_in * (rho-rho0))
    densityHat = np.exp(-kappa_n*aspr_in*(rho-rho0))
    safetyFactor = 0.85 + 2.2 * rho**2
    Lref = 1.65
    rhoStar = 1/150
    Bref = 2.5
    Tref = 1
    nref = 1
    
    # choose a suffix number unlikely to be used in pratice, then check that the checkpoint file does not exist already 
    checkpointSuffix = 999  # choose a checkpoint number unlikely to be used in practice
    assert not checkpoint_exists(checkpointSuffix), "Error in gene_check().  Checkpoint file with suffix {} aready exists".format(checkpoint_suffix_string(checkpointSuffix))
    
    # Perform a very short GENE run
    (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = genecomm_lowlevel.call_gene_low_level(
                simulationTime=simulationTime, rho=rho,
                temperatureHat=temperatureHat, densityHat=densityHat, safetyFactor=safetyFactor,
                Lref=Lref, Bref=Bref, rhoStar=rhoStar, Tref=Tref, nref=nref, checkpointSuffix=checkpointSuffix)
    
    time.sleep(0.1) # pause to allow processes to catch up
    
    # Check that necessary checkpoint files are created, then remove them
    if MPIrank==0:
        assert checkpoint_exists(checkpointSuffix), "Error in gene_check().  Checkpoint file not created as expected!"
        clean_files(checkpointSuffix)
    
    status = 0 # could add some error checking here?
    return (status, MPIrank)
    
def clean_files(checkpointSuffix):
    """Delete files used in the run.
    
    For instance, if checkpointSuffix==999, then this function deletes all files that end in _999.
    
    Inputs:
      checkpointSuffix      (integer)
    """
    globStr = '*_{}'.format(checkpoint_suffix_string(checkpointSuffix))
    filelist = glob.glob(globStr)
    for f in filelist:
        os.remove(f)
    
def checkpoint_exists(checkpointSuffix):
    """Return True if a checkpoint file 'checkpoint_<suffix>' exists in the current directory.
    
    For example, if checkpointSuffix==0, this function returns True if the file checkpoint_000 exists in the
    current directory.
    
    Inputs:
      checkpointSuffix      (integer)
    Outputs:
      exists                True if the checkpoint file exists, False if not (boolean)
    """
    filename = 'checkpoint_' + checkpoint_suffix_string(checkpointSuffix)
    exists = os.path.exists(filename)
    return exists
    
def checkpoint_suffix_string(checkpointSuffix):
    """Return a string representing the checkpoint file suffix for a given input integer.
    
    GENE uses a 3 digit integer for the suffix.  E.g., if checkpointSuffix=0, then files end in _000.
    
    Inputs:
      checkpointSuffix      (integer)
    Outputs:
      checkpointSuffix_str  checkpoint suffix with necessary padding (string)
    """
    checkpointSuffix_str = '{:03d}'.format(checkpointSuffix)
    return checkpointSuffix_str