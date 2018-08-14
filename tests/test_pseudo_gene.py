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
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, R0, a, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies) = tango.genecomm_lowlevel.pseudo_call_gene_low_level(
                simulationTime=simulationTime, rho=rho, mass=mass, charge=charge, temperatureHatAllSpecies=temperatureHatAllSpecies,
                densityHatAllSpecies=densityHatAllSpecies, safetyFactor=safetyFactor, Lref=Lref, Bref=Bref,
                rhoStar=rhoStar, Tref=Tref, nref=nref, checkpointSuffix=checkpointSuffix)
    
    assert dVdxHat is not None
    
def test_genecomm_lowlevel_interface():
    """Test the class-based interface to calling GENE in genecomm_lowlevel."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    pseudoGene = True
    geneInterface = tango.genecomm_lowlevel.GeneInterface(
                rho=rho, mass=mass, charge=charge, safetyFactor=safetyFactor, Lref=Lref, Bref=Bref, rhoStar=rhoStar, Tref=Tref, nref=nref,
                checkpointSuffix=checkpointSuffix, pseudoGene=pseudoGene)
    
    # test the call to GENE
    (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies) = geneInterface.call_gene(
                simulationTime, densityHatAllSpecies, temperatureHatAllSpecies)
    
    assert dVdxHat is not None
        
def test_genecomm_adiabatic_electrons():
    """Test the class-based interface to calling pseudo-GENE with genecomm for adiabatic electrons."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    densityHat = densityHatAllSpecies[0, :]
    densityGeneGrid = densityHat * 1e19    # assume Tango and GENE use the same radial grids here.
    
    e = 1.60217662e-19          # electron charge
    temperatureHat = temperatureHatAllSpecies[0, :]
    temperatureGeneGrid = temperatureHat * 1000 * e # convert to SI
    pseudoGene = True
    
    geneComm = tango.genecomm.GeneComm_SingleIonAdiabaticElectrons(
                Bref=Bref, Lref=Lref, Tref=Tref, nref=nref, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactor,
                psiTangoGrid=r, psiGeneGrid=r, densityTangoGrid=densityGeneGrid, mass=1, charge=1, gridMapper=gridMapper, pseudoGene=pseudoGene)
    
    geneComm.set_simulation_time(simulationTime)
    
    
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    profiles = {}
    profiles['pi'] = pressureGeneGrid
    
    
    # test the call
    fluxes = geneComm.get_flux(profiles)
    
    assert fluxes is not None

def test_genecomm_kinetic_electrons():
    """Test the class-based interface to calling pseudo-GENE with genecomm for kinetic electrons."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_kinetic()
    
    e = 1.60217662e-19          # electron charge
    pseudoGene = True
    geneComm = tango.genecomm.GeneComm_SingleIonKineticElectrons(
                Bref=Bref, Lref=Lref, Tref=Tref, nref=nref, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactor,
                psiTangoGrid=r, psiGeneGrid=r, mass=mass, charge=charge, pseudoGene=pseudoGene)
    
    geneComm.set_simulation_time(simulationTime)
    
    densityGeneGrid = densityHatAllSpecies[0, :] * 1e19
    
    profiles = {}
    profiles['n'] = densityGeneGrid
    profiles['pi'] = temperatureHatAllSpecies[0, :] * 1000 * e * densityGeneGrid
    profiles['pe'] = temperatureHatAllSpecies[1, :] * 1000 * e * densityGeneGrid
    
    # test the call
    fluxes = geneComm.get_flux(profiles)
    assert fluxes is not None

def test_genecomm_chease_adiabatic_electrons():
    """Test the class-based interface to calling pseudo-GENE with genecomm for adiabatic electrons with CHEASE geometry."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    densityHat = densityHatAllSpecies[0, :]
    densityGeneGrid = densityHat * 1e19    # assume Tango and GENE use the same radial grids here.

    e = 1.60217662e-19          # electron charge
    temperatureHat = temperatureHatAllSpecies[0, :]
    temperatureGeneGrid = temperatureHat * 1000 * e  # convert to SI
    pseudoGene = True

    cheaseTangoData = Empty()
    cheaseTangoData.Bref = 4
    cheaseTangoData.Lref = 3.2
    cheaseTangoData.minorRadius = 1
    x = r

    geneComm = tango.genecomm.GeneComm_CheaseSingleIonAdiabaticElectrons(
        cheaseTangoData=cheaseTangoData, Tref=Tref, nref=nref,
        xTangoGrid=x, xGeneGrid=x, densityTangoGrid=densityGeneGrid, mass=1, charge=1, gridMapper=gridMapper,
        pseudoGene=pseudoGene)

    geneComm.set_simulation_time(simulationTime)

    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    profiles = {}
    profiles['pi'] = pressureGeneGrid

    # test the call
    fluxes = geneComm.get_flux(profiles)
    assert fluxes is not None

def test_genestartup_adiabatic_electrons():
    """Test the GENE startup script for adiabatic electrons."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    psiTango = r
    psiGene = r
    B0 = Bref
    ionMass = 1
    ionCharge = 1
    densityTangoGrid = densityHatAllSpecies[0, :] * 1e19    # assume Tango and GENE use the same radial grids here.
    e = 1.60217662e-19          # electron charge
    temperatureTangoGrid = temperatureHatAllSpecies[0, :] * 1000 * e # convert to SI
    pressureTangoGrid = temperatureTangoGrid * densityTangoGrid

    pseudoGene = True

    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_adiabaticelectrons(
                psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityTangoGrid, pressureTangoGrid, safetyFactor,
                Bref, Lref, Tref, nref, gridMapper, fromCheckpoint=True, pseudoGene=pseudoGene)
    
    assert hasattr(geneFluxModel, 'get_flux')


def test_genestartup_kinetic_electrons():
    """Test the GENE startup script for kinetic electrons."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_kinetic()
    psiTango = r
    psiGene = r
    B0 = Bref

    pseudoGene = True

    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_kineticelectrons(
                psiTango, psiGene, minorRadius, majorRadius, B0, mass, charge, safetyFactor,
                Bref, Lref, Tref, nref, fromCheckpoint=True, pseudoGene=pseudoGene)

    assert hasattr(geneFluxModel, 'get_flux')


def test_genestartup_chease_adiabatic_electrons():
    """Test the GENE startup script for adiabatic electrons and CHEASE geometry."""
    (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref, majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix) = setup_parameters()
    cheaseTangoData = Empty()
    cheaseTangoData.Bref = 4
    cheaseTangoData.Lref = 3.2
    cheaseTangoData.minorRadius = 1

    xTango = r
    xGene = r
    ionMass = 1
    ionCharge = 1
    densityTangoGrid = densityHatAllSpecies[0, :] * 1e19    # assume Tango and GENE use the same radial grids here.
    pseudoGene = True

    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_chease_adiabaticelectrons(
        cheaseTangoData, xTango, xGene, ionMass, ionCharge, densityTangoGrid,
        Tref, nref, gridMapper, fromCheckpoint=True, pseudoGene=pseudoGene)

    assert hasattr(geneFluxModel, 'get_flux')

def test_genecomm_with_different_grids_tango_inside():
    """Test the python-GENE interface where different grids for Tango and GENE are used (and adiabatic electrons)
    
    Here, Tango's outer radial boundary is radially inward that of GENE.
    """
    (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_different_grids_tango_inside()
 
    pseudoGene = True    
    geneComm = tango.genecomm.GeneComm_SingleIonAdiabaticElectrons(
                Bref=Bref, Lref=Lref, Tref=1, nref=1, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                psiTangoGrid=rTango, psiGeneGrid=rGene, densityTangoGrid=densityTangoGrid, mass=1, charge=1, gridMapper=gridMapper,
                pseudoGene=pseudoGene)
    geneComm.set_simulation_time(simulationTime)
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    profiles = {}
    profiles['pi'] = pressureGeneGrid
    
    fluxes = geneComm.get_flux(profiles)
    assert fluxes is not None
    
def test_genecomm_with_different_grids_tango_outside():
    """Test the python-GENE interface where different grids for Tango and GENE are used (and adiabatic electrons)
    
    Here, Tango's outer radial boundary is farther radially outward than that of GENE.
    """
    (simulationTime, rTango, rGene, temperatureGeneGrid, densityTangoGrid, densityGeneGrid, safetyFactorGeneGrid, Lref, Bref, majorRadius, minorRadius, rhoStar, gridMapper, checkpointSuffix) = setup_parameters_different_grids_tango_outside()
 
    pseudoGene = True
    geneComm = tango.genecomm.GeneComm_SingleIonAdiabaticElectrons(
                Bref=Bref, Lref=Lref, Tref=1, nref=1, B0=Bref, minorRadius=minorRadius, majorRadius=majorRadius, safetyFactorGeneGrid=safetyFactorGeneGrid,
                psiTangoGrid=rTango, psiGeneGrid=rGene, densityTangoGrid=densityTangoGrid, mass=1, charge=1, gridMapper=gridMapper,
                pseudoGene=pseudoGene)
    geneComm.set_simulation_time(simulationTime)
    pressureGeneGrid = temperatureGeneGrid * densityGeneGrid
    profiles = {}
    profiles['pi'] = pressureGeneGrid
    
    fluxes = geneComm.get_flux(profiles)
    assert fluxes is not None


########################### Helper Functions ##############################
def setup_parameters():
    # some parameters for a call to GENE
    simulationTime = 0.4
    Npts = 50
    rho = np.linspace(0.1, 0.9, Npts)
    temperatureHat = 0.4 * np.ones_like(rho)
    densityHat = np.ones_like(rho)
    temperatureHatAllSpecies = np.zeros((1, Npts), dtype=np.float64)
    densityHatAllSpecies = np.zeros((1, Npts), dtype=np.float64)
    temperatureHatAllSpecies[0, :] = temperatureHat
    densityHatAllSpecies[0, :] = densityHat
    
    mass = np.array([1])
    charge = np.array([1])
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
    return (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref,
            majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix)
    
def setup_parameters_kinetic():
    # some parameters for a call to GENE with kinetic electrons (2 species)
    simulationTime = 0.4
    Npts = 50
    rho = np.linspace(0.1, 0.9, Npts)
    temperatureHat = 0.4 * np.ones_like(rho)
    densityHat = np.ones_like(rho)
    temperatureHatAllSpecies = np.zeros((2, Npts), dtype=np.float64)
    densityHatAllSpecies = np.zeros((2, Npts), dtype=np.float64)
    temperatureHatAllSpecies[0, :] = 1.0 * temperatureHat
    temperatureHatAllSpecies[1, :] = 1.0 * temperatureHat
    densityHatAllSpecies[0, :] = 1.0 * densityHat
    densityHatAllSpecies[1, :] = 1.0 * densityHat
    
    mass = np.array([2.0, 1.0/1836])
    charge = np.array([1, -1])
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
    return (simulationTime, rho, mass, charge, temperatureHatAllSpecies, densityHatAllSpecies, safetyFactor, Lref, Bref, Tref, nref,
            majorRadius, minorRadius, r, rhoStar, gridMapper, checkpointSuffix)
    
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

class Empty(object):
    pass