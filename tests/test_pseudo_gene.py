"""Test the Python interface code to run GENE in various modules

But to avoid actually running GENE, use a pseudo GENE call that takes the same input arguments and returns
the same type of values.  The values returned are not correct, but these tests do ensure code syntax is
proper.
"""

from __future__ import division, absolute_import
import numpy as np

import tango.parameters
import tango.genecomm_lowlevel
import tango.gene_check
import tango.genecomm
import tango.gene_startup

def test_genecomm_lowlevel_call():
    """Test the basic call to GENE in genecomm_lowlevel."""
    (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, Tref, nref, R0, a, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = tango.genecomm_lowlevel.pseudo_call_gene_low_level(
                                                                 simulationTime=simulationTime, rho=rho, temperatureHat=temperatureHat,
                                                                 densityHat = densityHat, safetyFactor=safetyFactor, Lref=Lref, Bref=Bref,
                                                                 rhoStar=rhoStar, Tref=Tref, nref=nref, checkpointSuffix=checkpointSuffix)
    
    assert dVdxHat is not None
    
def test_genecomm_lowlevel_interface():
    """Test the class-based interface to calling GENE in genecomm_lowlevel."""
    (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    pseudoGene = True
    geneInterface = tango.genecomm_lowlevel.GeneInterface(rho=rho, densityHat=densityHat, safetyFactor=safetyFactor,
                                                          Lref=Lref, Bref=Bref, rhoStar=rhoStar, Tref=Tref, nref=nref, checkpointSuffix=checkpointSuffix,
                                                           pseudoGene=pseudoGene)
    
    # test the call to GENE
    (dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = geneInterface.call_gene(simulationTime, temperatureHat)
    
    assert dVdxHat is not None
        
def test_genecomm():
    """Test the class-based interface to calling pseudo-GENE in genecomm."""
    (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    densityGeneGrid = densityHat * 1e19    # assume Tango and GENE use the same radial grids here.
    e = 1.60217662e-19          # electron charge
    temperatureGeneGrid = temperatureHat * 1000 * e # convert to SI
    pseudoGene = True
    geneComm = tango.genecomm.GeneComm(Bref=Bref, Lref=Lref, Tref=Tref, nref=nref, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactor,
                                       psiTangoGrid=r, psiGeneGrid=r, densityTangoGrid=densityGeneGrid, gridMapper=gridMapper,
                                       pseudoGene=pseudoGene)
    geneComm.set_simulation_time(simulationTime)
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    
    # test the call
    avgHeatFluxGeneGrid = geneComm.get_flux(pressureGeneGrid)
    
    assert avgHeatFluxGeneGrid is not None

def test_genestartup():
    """Test the GENE startup script with pseudoGENE"""
    (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    psiTango = r
    psiGene = r
    B0 = Bref
    ionMass = 1
    ionCharge = 1
    densityTangoGrid = densityHat * 1e19    # assume Tango and GENE use the same radial grids here.
    e = 1.60217662e-19          # electron charge
    temperatureTangoGrid = temperatureHat * 1000 * e # convert to SI
    pressureTangoGrid = temperatureTangoGrid * densityTangoGrid
    
    pseudoGene = True
    (geneFluxModel, MPIrank) = tango.gene_startup.setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityTangoGrid, pressureTangoGrid, Bref, Lref, Tref, nref,
                   gridMapper, fromCheckpoint=True, pseudoGene=pseudoGene)
    
    assert MPIrank is not None
    assert hasattr(geneFluxModel, 'get_flux')
    

def test_genecomm_with_different_grids_tango_inside():
    """Test the entire python-GENE interface where different grids for Tango and GENE are used.
    
    Here, Tango's outer radial boundary is radially inward that of GENE.
    """
    (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_different_grids_tango_inside()
 
    pseudoGene = True
    geneComm = tango.genecomm.GeneComm(Bref=Bref, Lref=Lref, Tref=1, nref=1, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                                       psiTangoGrid=rTango, psiGeneGrid=rGene, densityTangoGrid=densityTangoGrid, gridMapper=gridMapper,
                                       pseudoGene=pseudoGene)
    geneComm.set_simulation_time(simulationTime)
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    avgHeatFluxGeneGrid = geneComm.get_flux(pressureGeneGrid)
    assert avgHeatFluxGeneGrid is not None
    
    

def test_genecomm_with_different_grids_tango_outside():
    """Test the entire python-GENE interface where different grids for Tango and GENE are used.
    
    Here, Tango's outer radial boundary is farther radially outward than that of GENE.
    """
    (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_different_grids_tango_outside()
 
    pseudoGene = True
    geneComm = tango.genecomm.GeneComm(Bref=Bref, Lref=Lref, Tref=1, nref=1, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                                       psiTangoGrid=rTango, psiGeneGrid=rGene, densityTangoGrid=densityTangoGrid, gridMapper=gridMapper,
                                       pseudoGene=pseudoGene)
    geneComm.set_simulation_time(simulationTime)
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    avgHeatFluxGeneGrid = geneComm.get_flux(pressureGeneGrid)
    assert avgHeatFluxGeneGrid is not None


########################### Helper Functions ##############################
def setup_parameters():
    # some parameters for a call to GENE
    simulationTime = 0.4
    rho = np.linspace(0.1, 0.9, 50)
    temperatureHat = 0.4 * np.ones_like(rho)
    densityHat = np.ones_like(rho)
    Lref = 1.65
    Bref = 2.5
    Tref = 0.4
    nref = 1
    majorRadius = 1.65
    minorRadius = 0.594
    rhoStar = 1/140
    r = rho * minorRadius
    safetyFactor = tango.parameters.analytic_safety_factor(r, minorRadius, majorRadius)
    gridMapper = tango.interfacegrids_gene.GridsNull(r)
    checkpointSuffix = 999
    return (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix)
    
def setup_parameters_different_grids_tango_inside():
    # set up radial grids with Tango's outer radial boundary radially inward that of GENE.
    simulationTime = 0.4
    Lref = 1.65
    Bref = 2.5
    majorRadius = 1.65
    minorRadius = 0.594
    rhoStar = 1/140
    checkpointSuffix = 999
    
    numRadialPtsTango = 100
    numRadialPtsGene = 80
    rhoTango = np.linspace(0.1, 0.8, numRadialPtsTango)      # rho = r/a
    rhoGene = np.linspace(0.2, 0.9, numRadialPtsGene)
    
    rTango = rhoTango * minorRadius    # physical radius r
    rGene = rhoGene * minorRadius
    safetyFactorGeneGrid = tango.parameters.analytic_safety_factor(rGene, minorRadius, majorRadius)
    
    e = 1.60217662e-19          # electron charge
    temperatureGeneGrid = 1000 * e * np.ones_like(rGene)
    densityTangoGrid = 1e19 * np.ones_like(rTango)
    densityGeneGrid = 1e19 * np.ones_like(rGene)
    gridMapper = tango.interfacegrids_gene.GridInterfaceTangoInside(rTango, rGene)
    return (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix)
    
def setup_parameters_different_grids_tango_outside():
    # set up radial grids with Tango's outer radial boundary radially inward that of GENE.
    simulationTime = 0.4
    Lref = 1.65
    Bref = 2.5
    majorRadius = 1.65
    minorRadius = 0.594
    rhoStar = 1/140
    checkpointSuffix = 999
    
    numRadialPtsTango = 100
    numRadialPtsGene = 80
    rhoTango = np.linspace(0.1, 0.9, numRadialPtsTango)      # rho = r/a
    rhoGene = np.linspace(0.2, 0.7, numRadialPtsGene)
    
    rTango = rhoTango * minorRadius    # physical radius r
    rGene = rhoGene * minorRadius
    safetyFactorGeneGrid = tango.parameters.analytic_safety_factor(rGene, minorRadius, majorRadius)
    
    e = 1.60217662e-19          # electron charge
    temperatureGeneGrid = 1000 * e * np.ones_like(rGene)
    densityTangoGrid = 1e19 * np.ones_like(rTango)
    densityGeneGrid = 1e19 * np.ones_like(rGene)
    gridMapper = tango.interfacegrids_gene.GridInterfaceTangoOutside(rTango, rGene)
    return (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix)