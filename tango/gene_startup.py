from __future__ import division
import numpy as np
from . import gene_check
from . import genecomm
from . import parameters


"""
Handle the initialization of GENE for a Tango-GENE run.
"""
    
def setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityTangoGrid, pressureTangoGrid, Bref, Lref, 
                   grids, fromCheckpoint=True):
    """Do all the necessary setup for a tango run using GENE
    
    This function works with either a clean run from no checkpoint, or starting from an initial condition.
    
    Inputs:
      psiTango              Tango's grid for radial coordinate, psi = r (array)
      psiGene               GENE's grid for radial coordinate, psi = x = r (array)
      minorRadius           minor radius a (scalar)
      majorRadius           major radius R0 (scalar)
      B0                    magnetic field parameter for analytic circular geometry, in Tesla (array)
      ionMass               ion mass, measured in proton mass (scalar)
      ionCharge             ion charge, measured in electron charge (scalar)
      densityTangoGrid      density profile on Tango radial grid in m^-3 (array)
      pressureTangoGrid     ion pressure profile (initial condition) on Tango radial grid in J/m^3 (array)
      Bref                  GENE reference magnetic field in Tesla (scalar)
      Lref                  GENE reference length in m (scalar)
      grids                 object for interfacing between Tango and GENE grids [see interfacegrids_gene.py]
      fromCheckpoint        True if restarting GENE from a checkpoint (Boolean)
    """
    # check that GENE works and get MPI rank
    (status, MPIrank) = gene_check.gene_check()
    
    # compute safety Factor on GENE grid
    safetyFactorGeneGrid = parameters.analytic_safety_factor(psiGene, minorRadius, majorRadius)
    
    # if doing a clean run from no checkpoint, create the initial checkpoint...
    if fromCheckpoint==False:
        geneFluxModelTemp = genecomm.geneComm(Bref=Bref, Lref=Lref, B0=B0, a=minorRadius, R0=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                                      psiTangoGrid=psiTango, psiGeneGrid=psiGene, densityTangoGrid=densityTangoGrid, ionMass=ionMass, ionCharge=ionCharge, grids=grids)
        simulationTimeInitialRun = 30
        pressureGeneGrid = grids.MapProfileOntoTurbGrid(pressureTangoGrid)
        initial_gene_run(geneFluxModelTemp, pressureGeneGrid, simulationTimeInitialRun)
    
    # create a GENE Fluxmodel    
    geneFluxModel = genecomm.geneComm(Bref=Bref, Lref=Lref, B0=B0, a=minorRadius, R0=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                                      psiTangoGrid=psiTango, psiGeneGrid=psiGene, densityTangoGrid=densityTangoGrid, ionMass=ionMass, ionCharge=ionCharge, grids=grids)
    
    # set the simulation time per GENE call
    simulationTime = 10 # measured in Lref/cref
    geneFluxModel.set_simulation_time(simulationTime)
    return (geneFluxModel, MPIrank)
    

def pseudo_setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityProfile, pressureProfile, Bref, Lref, 
                   grids, fromCheckpoint=True):
    """Pseudo version of setup_gene_run to return values for testing purposes without actually running GENE
    """
    MPIrank = 0
    
    class GeneFluxModel(object):
        def GetFlux(self):  # turbhandler requires that a FluxModel has a GetFlux() method
            pass
        def set_simulation_time(self, simulationTime):  # this method  is also required
            pass
    
    geneFluxModel = GeneFluxModel()
    return (geneFluxModel, MPIrank)
    
def initial_gene_run(geneFluxModel, pressureGeneGrid, simulationTime):
    """Perform the initial GENE run.  
    
    Tango uses a GetFlux() method which runs a GENE simulation for a short amount of time with slightly updated
    profiles and receives a slightly updated turbulent flux.  For these numbers to be sensical, (in other words,
    for the returned turbulent flux to be reflective of the input density profile), an initial GENE run must be
    performed so that the turbulence has had time to develop.  This function performs that initial run to create
    a checkpoint with somewhat developed or mostly developed turbulence from which Tango can then begin its normal 
    iteration.
    
    Inputs:
      geneFluxModel     object to control simulations of GENE
      pressureGeneGrid  ion pressure profile on GENE radial grid, measured in J/m^3 (array)
      simulationTime:   Time for GENE simulation.  Measured in units of Lref/cref (scalar)
    
    Outputs:
    """
    checkpointSuffix = 0
    
    # Check that a checkpoint _000 does not exist.  If it does, abort.    
    assert not gene_check.checkpoint_exists(checkpointSuffix), "Error in gene_startup.initial_gene_run().  Checkpoint file with suffix {} aready exists".format(gene_check.checkpoint_suffix_string(checkpointSuffix))
    
    # Run.
    geneFluxModel.set_simulation_time(simulationTime)
    heatFlux_notUsed = geneFluxModel.GetFlux(pressureGeneGrid)
    
    # Verify checkpoint
    assert gene_check.checkpoint_exists(checkpointSuffix), "Error in gene_startup.initial_gene_run().  Checkpoint file not created as expected!"