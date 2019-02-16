"""
genecomm

High-level communcation between GENE and the rest of Tango using a class-based interface

Various incarnations exist for different situations
    1. Single ion species with adiabatic electrons
    2. Single ion species with kinetic electrons
    3. (Not yet written) Multiple ion species with adiabatic electrons

Responsibilities of the class:
1. Unit conversion   Tango speaks in SI, GENE speaks in normalized units.  This module converts between the two.
2. Encapsulation     Internally maintain everything that stays fixed; provide an interface that can be called by Tango
                       with only the quantities that change (e.g., density and/or pressure profile)
3. Provide interface to modify certain parameters, e.g., time step simulation time

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
from . import genecomm_lowlevel
from . import genecomm_unitconversion


class GeneComm_SingleIonAdiabaticElectrons(object):
    """Class-based interface for Tango to call GENE, when GENE is run with a single ion species and adiabatic electrons, and
    using analytic circular magnetic geometry.
    
    Tango requires an object with a get_flux() method, which this class provides.  Except where specifically noted otherwise,
    all quantities are stored in SI units.
    
    With a single ion species and adiabatic electrons, the density profile does not evolve in time.  Hence, in this class, the
    initial density profile is stored and used for every call, and the particle flux output is ignored.
    """
    def __init__(self, Bref=None, Lref=None, Tref=None, nref=None, B0=None, minorRadius=None, majorRadius=None, safetyFactorGeneGrid=None,
                 psiTangoGrid=None, psiGeneGrid=None, densityTangoGrid=None, mass=None, charge=None, gridMapper=None,
                 pseudoGene=False):
        """Constructor.
            
        Inputs:
          Bref                      GENE reference magnetic field, measured in Tesla (scalar)
          Lref                      GENE reference length, measured in meters (scalar)
          Tref                      GENE reference temperature, measured in keV (scalar)
          nref                      GENE reference density, measured in 10^19 m^-3 (scalar)
          B0                        parameter specifying the analytic magnetic field for circular geometry.  Measured in Tesla.  
                                      Typically equal to Bref (scalar)
          minorRadius               minor radius a, measured in meters (scalar)
          majorRadius               major radius R0 on axis, measured in meters (scalar)
          safetyFactorGeneGrid      safety factor evaluated on the GENE grid psiGeneGrid (array)
          psiTangoGrid              radial grid used by Tango (array)
          psiGeneGrid               radial grid used by GENE, with coordinate psi=r (array)
          densityTangoGrid          density profile, measured in m^-3, on the Tango grid (array)
          mass                      mass of ion species, measured in proton masses (scalar)
          charge                    charge of ion species, measured in electron charge (scalar)
          gridMapper                object for transforming the profiles and transport coefficients between the Tango grid and
                                      the GENE grid.  See interfacegrids_gene.py (object)
          pseudoGene                False for normal GENE run, True for a pseudo call that does not run GENE but is used to test code (Boolean)
        """
        self.numSpecies = 1
        self.Bref = Bref
        self.Lref = Lref
        self.Tref = Tref
        self.nref = nref
        self.mref = mass       # measured in proton masses
        self.B0 = B0
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        self.safetyFactorGeneGrid = safetyFactorGeneGrid
        self.psiTangoGrid = psiTangoGrid  # psi = r
        self.psiGeneGrid = psiGeneGrid # psi = x = r
        self.densityTangoGrid = np.squeeze(densityTangoGrid) # squeeze in case it is an array with singleton dimension
        self.ionMass = mass # measured in (proton masses)
        self.ionCharge = charge # measured in electron charge
        self.simulationTime = None  # measured in Lref/cref
        
        self.numRadialGridPointsGENE = len(psiGeneGrid)
        assert hasattr(gridMapper, 'map_profile_onto_turb_grid') and callable(getattr(gridMapper, 'map_profile_onto_turb_grid'))
        assert hasattr(gridMapper, 'map_transport_coeffs_onto_transport_grid') and callable(getattr(gridMapper, 'map_transport_coeffs_onto_transport_grid'))
        self.gridMapper = gridMapper
        self.pseudoGene = pseudoGene
        
        self.densityGeneGrid = self.gridMapper.map_profile_onto_turb_grid(densityTangoGrid)
        
        # place density into a 2D array, species x space
        densityHatGeneGrid = genecomm_unitconversion.density_SI_to_gene(self.densityGeneGrid)
        self.densityHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        self.densityHatGeneGridAllSpecies[0, :] = densityHatGeneGrid
        
        # create the interface for calling gene
        self.geneInterface = self._create_gene_interface()
    
    
    #### Interface for Tango  ####
    def get_flux(self, profiles):
        """Run GENE with a given pressure profile.  Return heat flux.
        
        According to the documentation for the Turbulence Handler, get_flux() must take as input the profile on the turbulence grid and return
        the turbulent flux on the turbulence grid (see lodestro_method.py)
        
        Inputs:
          profiles      dict containing:
                           'pi': pressure profile, measured in SI units, on the GENE radial grid (1D array)
        Outputs:
          fluxes        dict containing:
                           'pi': heat flux, measured in SI units, on the GENE radial grid (1D array)
          
        Side Effects: genecomm_lowlevel is currently set such that the same checkpoint suffix (_000) is used every run.  This means that all of
        the GENE output files, including the checkpoint file, are overwritten each time.  GENE writes out two checkpoint files when it completes
        a simulation: checkpoint and s_checkpoint [with the suffix, this becomes checkpoint_000 and s_checkpoint_000].  These checkpoint files
        are contain the same information.  They each contain the perturbed distribution function f1.
        
        When GENE starts a new simulation, it will look for the file checkpoint_000.  If that file does not exist GENE will look for
        s_checkpoint_000.  If that does not exist, GENE looks for a checkpoint from the previous iteration.  If that doesn't exist,
        GENE will run without a checkpoint, and start from scratch.
        """
        pressureGeneGrid = profiles['pi']  # ion pressure on GENE grid
        
        # convert pressure profile into a temperature
        temperatureGeneGrid = pressure_to_temperature(pressureGeneGrid, self.densityGeneGrid)
        
        # convert the temperature from SI to GENE's normalized units
        temperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(temperatureGeneGrid)
        
        # place temperature into a 2D array, species x space
        temperatureHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        temperatureHatGeneGridAllSpecies[0, :] = temperatureHatGeneGrid
        
        # run GENE and get heat flux on GENE's grid
        (dVdxHat, sqrt_gxx, avgParticleFluxHatGeneGridAllSpecies, avgHeatFluxHatGeneGridAllSpecies) = self.geneInterface.call_gene(
                    self.simulationTime, self.densityHatGeneGridAllSpecies, temperatureHatGeneGridAllSpecies)

        # extract individual heat flux
        avgHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[0, :]
                
        # convert GENE's normalized heat flux into SI units
        avgHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        
        fluxes = {}
        fluxes['pi'] = avgHeatFluxGeneGrid
        return fluxes
        
    def set_simulation_time(self, simulationTime):
        """Adjust how long GENE will run for when called to perform turbulence simulations.
        
        Inputs:
          simulationTime    GENE parameter that controls how long a simulation is run.  Measured in Lref/cref (scalar)
        """
        self.simulationTime = simulationTime
    
    def _create_gene_interface(self):
        """Instantiate an object that provides an interface to GENE.
        
        genecomm_lowlevel.GeneInterface expects quantities to be given in GENE normalized units.  Additionally, quantities must exist
        on GENE's radial grid, psiGene.
        
        Outputs:
          geneInterface     Instance of genecomm_lowlevel.GeneInterface, enables running GENE
        """
        # convert SI quantities to what GENE expects
        rhoGeneInterface = genecomm_unitconversion.radius_SI_to_libgenetango_input(self.psiGeneGrid, self.minorRadius)  # convert to rho = r/a
        
        rhoStar = genecomm_unitconversion.calculate_consistent_rhostar(self.Tref, self.Bref, self.mref, self.minorRadius)
        
        # create a few arrays that geneInterface expects
        mass = np.array([self.ionMass])
        charge = np.array([self.ionCharge])
        
        geneInterface = genecomm_lowlevel.GeneInterface(
                rho=rhoGeneInterface, mass=mass, charge=charge, safetyFactor=self.safetyFactorGeneGrid,
                Lref=self.Lref, Bref=self.Bref, rhoStar=rhoStar, Tref=self.Tref, nref=self.nref,
                pseudoGene=self.pseudoGene)
        return geneInterface
        

class GeneComm_SingleIonKineticElectrons(object):
    """Class-based interface for Tango to call GENE, when GENE is run with a single ion species and adiabatic electrons.
    
    Tango requires an object with a get_flux() method, which this class provides.  Except where specifically noted otherwise,
    all quantities are stored in SI units.
    
    With a kinetic ions and kinetic electrons, the density profile can evolve in time.  Hence, in this class, there is no
    initial density profile that is stored.  Rather, the density profile is provided on every call to GENE.  Furthermore, the
    particle flux is returned, not ignored as in the case with adiabatic electrons.
    """
    def __init__(self, Bref=None, Lref=None, Tref=None, nref=None, B0=None, minorRadius=None, majorRadius=None, safetyFactorGeneGrid=None,
                 psiTangoGrid=None, psiGeneGrid=None, mass=None, charge=None, pseudoGene=False):
        """Constructor.
            
        Inputs:
          Bref                      GENE reference magnetic field, measured in Tesla (scalar)
          Lref                      GENE reference length, measured in meters (scalar)
          Tref                      GENE reference temperature, measured in keV (scalar)
          nref                      GENE reference density, measured in 10^19 m^-3 (scalar)
          B0                        parameter specifying the analytic magnetic field for circular geometry.  Measured in Tesla.  
                                      Typically equal to Bref (scalar)
          minorRadius               minor radius a, measured in meters (scalar)
          majorRadius               major radius R0 on axis, measured in meters (scalar)
          safetyFactorGeneGrid      safety factor evaluated on the GENE grid psiGeneGrid (array)
          psiTangoGrid              radial grid used by Tango (array)
          psiGeneGrid               radial grid used by GENE, with coordinate psi=r (array)
          mass                      species masses, in proton masses (1D array, by species)
          charge                    species charges, in elementary charges (1D array, by species)
          pseudoGene                False for normal GENE run, True for a pseudo call that does not run GENE but is used to test code (Boolean)
        """
        self.numSpecies = 2
        self.Bref = Bref
        self.Lref = Lref
        self.Tref = Tref
        self.nref = nref
        self.mref = mass[0]       # measured in proton masses
        self.B0 = B0
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        self.safetyFactorGeneGrid = safetyFactorGeneGrid
        self.psiTangoGrid = psiTangoGrid  # psi = r
        self.psiGeneGrid = psiGeneGrid # psi = x = r
        self.mass = mass
        self.charge = charge
        self.simulationTime = None  # measured in Lref/cref
        
        self.numRadialGridPointsGENE = len(psiGeneGrid)
        self.pseudoGene = pseudoGene
                
        # create the interface for calling gene
        self.geneInterface = self._create_gene_interface()
        
    #### Interface for Tango  ####
    def get_flux(self, profiles):
        """Run GENE with given profiles.  Return ion particle flux and ion & electron heat fluxes.
        
        According to the documentation for the Turbulence Handler, get_flux() must take as input the profile on the turbulence grid and return
        the turbulent flux on the turbulence grid (see lodestro_method.py)
        
        Inputs:
          profiles      dict containing:
                           'n': density profile, measured in SI units, on the GENE radial grid (1D array)
                           'pi': ion pressure profile, measured in SI units, on the GENE radial grid (1D array)
                           'pe': electron pressure profile, measured in SI units, on the GENE radial grid (1D array)
        Outputs:
          fluxes        dict containing:
                           'n': particle flux, measured in SI units, on the GENE radial grid (1D array)
                           'pi': ion heat flux, measured in SI units, on the GENE radial grid (1D array)
                           'pe': electron heat flux, measured in SI units, on the GENE radial grid (1D array)
        
        Here, density 'n' represents the ion density, and particle flux represents the ion particle flux, which is a relevant distinction
        if the ion charge is not equal to 1.       

        It is also assumed that GENE is run where the ions are "species 1" and the electrons are "species 2".                    
                           
        Side Effects: genecomm_lowlevel is currently set such that the same checkpoint suffix (_000) is used every run.  This means that all of
        the GENE output files, including the checkpoint file, are overwritten each time.  GENE writes out two checkpoint files when it completes
        a simulation: checkpoint and s_checkpoint [with the suffix, this becomes checkpoint_000 and s_checkpoint_000].  These checkpoint files
        are contain the same information.  They each contain the perturbed distribution function f1.
        
        When GENE starts a new simulation, it will look for the file checkpoint_000.  If that file does not exist GENE will look for
        s_checkpoint_000.  If that does not exist, GENE looks for a checkpoint from the previous iteration.  If that doesn't exist,
        GENE will run without a checkpoint, and start from scratch.
        """
        densityGeneGrid = profiles['n']
        ionPressureGeneGrid = profiles['pi']
        electronPressureGeneGrid = profiles['pe']
        
        # convert pressure profile into a temperature, as GENE takes temperature as input
        ionTemperatureGeneGrid = pressure_to_temperature(ionPressureGeneGrid, densityGeneGrid)
        electronTemperatureGeneGrid = pressure_to_temperature(electronPressureGeneGrid, densityGeneGrid)
        
        # convert density and temperature from SI to GENE's normalized units
        densityHatGeneGrid = genecomm_unitconversion.density_SI_to_gene(densityGeneGrid)
        ionTemperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(ionTemperatureGeneGrid)
        electronTemperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(electronTemperatureGeneGrid)
        
        # place density into a 2D array, species x space... 
        # density represents ION density, so electron density = ion charge * ion density
        densityHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        densityHatGeneGridAllSpecies[0, :] = densityHatGeneGrid         # ion density
        densityHatGeneGridAllSpecies[1, :] = self.charge[0] * densityHatGeneGrid # electron density
        
        # place temperature into a 2D array, species x space
        temperatureHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        temperatureHatGeneGridAllSpecies[0, :] = ionTemperatureHatGeneGrid
        temperatureHatGeneGridAllSpecies[1, :] = electronTemperatureHatGeneGrid
        
        # run GENE and get heat flux on GENE's grid
        (dVdxHat, sqrt_gxx, avgParticleFluxHatGeneGridAllSpecies, avgHeatFluxHatGeneGridAllSpecies) = self.geneInterface.call_gene(
                    self.simulationTime, densityHatGeneGridAllSpecies, temperatureHatGeneGridAllSpecies)
        
        # extract individual particle and heat flux
        avgIonParticleFluxHatGeneGrid = avgParticleFluxHatGeneGridAllSpecies[0, :]
        avgIonHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[0, :]
        avgElectronHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[1, :]
        
        # convert GENE's normalized particle and heat flux into SI units
        avgIonParticleFluxGeneGrid = genecomm_unitconversion.particleflux_gene_to_SI(avgIonParticleFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        avgIonHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgIonHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        avgElectronHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgElectronHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        
        fluxes = {}
        fluxes['n'] = avgIonParticleFluxGeneGrid
        fluxes['pi'] = avgIonHeatFluxGeneGrid
        fluxes['pe'] = avgElectronHeatFluxGeneGrid
        return fluxes
        
    def set_simulation_time(self, simulationTime):
        """Adjust how long GENE will run for when called to perform turbulence simulations.
        
        Inputs:
          simulationTime    GENE parameter that controls how long a simulation is run.  Measured in Lref/cref (scalar)
        """
        self.simulationTime = simulationTime
        
    def _create_gene_interface(self):
        """Instantiate an object that provides an interface to GENE.
        
        genecomm_lowlevel.GeneInterface expects quantities to be given in GENE normalized units.  Additionally, quantities must exist
        on GENE's radial grid, psiGene.
        
        Outputs:
          geneInterface     Instance of genecomm_lowlevel.GeneInterface, enables running GENE
        """
        # convert SI quantities to what GENE expects
        rhoGeneInterface = genecomm_unitconversion.radius_SI_to_libgenetango_input(self.psiGeneGrid, self.minorRadius)  # convert to rho = r/a
        
        rhoStar = genecomm_unitconversion.calculate_consistent_rhostar(self.Tref, self.Bref, self.mref, self.minorRadius)
        
        geneInterface = genecomm_lowlevel.GeneInterface(
                rho=rhoGeneInterface, mass=self.mass, charge=self.charge, safetyFactor=self.safetyFactorGeneGrid,
                Lref=self.Lref, Bref=self.Bref, rhoStar=rhoStar, Tref=self.Tref, nref=self.nref,
                pseudoGene=self.pseudoGene)
        return geneInterface
    

class GeneComm_CheaseSingleIonAdiabaticElectrons:
    """Class-based interface for Tango to call GENE, when GENE is run with a single ion species and adiabatic electrons, and
    using magnetic geometry from a CHEASE h5 file.

    Tango requires an object with a get_flux() method, which this class provides.  Except where specifically noted otherwise,
    all quantities are stored in SI units.

    With a single ion species and adiabatic electrons, the density profile does not evolve in time.  Hence, in this class, the
    initial density profile is stored and used for every call, and the particle flux output from GENE is ignored.
    """
    def __init__(self, cheaseTangoData=None, Tref=None, nref=None,
                 xTangoGrid=None, xGeneGrid=None, densityTangoGrid=None, mass=None, charge=None, gridMapper=None,
                 pseudoGene=False):
        """Constructor.

        Inputs:
          cheaseTangoData           container with Chease data (instance of CheaseTangoData)
                                      From the chease data, get Bref, Lref, minor radius
          Tref                      GENE reference temperature, measured in keV (scalar)
          nref                      GENE reference density, measured in 10^19 m^-3 (scalar)
          xTangoGrid                radial grid used by Tango (array)
          xGeneGrid                 radial grid used by GENE, with coordinate x = rho_tor (array)
          densityTangoGrid          density profile, measured in m^-3, on the Tango grid (array)
          mass                      mass of ion species, measured in proton masses (scalar)
          charge                    charge of ion species, measured in electron charge (scalar)
          gridMapper                object for transforming the profiles and transport coefficients between the Tango grid and
                                        the GENE grid.  See interfacegrids_gene.py (object).  Used here for mapping the density
                                        profile onto the Tango grid
          pseudoGene                False for normal GENE run, True for a pseudo call that does not run GENE but is used to test code (Boolean)
        """
        self.numSpecies = 1
        self.Bref = cheaseTangoData.Bref
        self.Lref = cheaseTangoData.Lref
        self.minorRadius = cheaseTangoData.minorRadius

        self.Tref = Tref
        self.nref = nref
        self.mref = mass       # measured in proton masses

        # when using CHEASE, GENE ignores the safety factor from the python interface.  Use junk here
        self.safetyFactorGeneGrid = -1 * xGeneGrid

        self.xTangoGrid = xTangoGrid  # x = rho_tor when using CHEASE geometry
        self.xGeneGrid = xGeneGrid
        self.densityTangoGrid = np.squeeze(densityTangoGrid)  # squeeze in case it is an array with singleton dimension
        self.ionMass = mass  # measured in (proton masses)
        self.ionCharge = charge  # measured in electron charge
        self.simulationTime = None  # measured in Lref/cref

        self.numRadialGridPointsGENE = len(xGeneGrid)
        assert hasattr(gridMapper, 'map_profile_onto_turb_grid') and callable(getattr(gridMapper, 'map_profile_onto_turb_grid'))
        assert hasattr(gridMapper, 'map_transport_coeffs_onto_transport_grid') and callable(getattr(gridMapper, 'map_transport_coeffs_onto_transport_grid'))
        self.gridMapper = gridMapper
        self.pseudoGene = pseudoGene

        self.densityGeneGrid = self.gridMapper.map_profile_onto_turb_grid(densityTangoGrid)

        # place density into a 2D array, species x space
        densityHatGeneGrid = genecomm_unitconversion.density_SI_to_gene(self.densityGeneGrid)
        self.densityHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        self.densityHatGeneGridAllSpecies[0, :] = densityHatGeneGrid

        # create the interface for calling gene
        self.geneInterface = self._create_gene_interface()

    #### Interface for Tango  ####
    def get_flux(self, profiles):
        """Run GENE with a given pressure profile.  Return heat flux.

        According to the documentation for the Turbulence Handler, get_flux() must take as input the profile on the turbulence grid and return
        the turbulent flux on the turbulence grid (see lodestro_method.py)

        Inputs:
          profiles      dict containing:
                           'pi': pressure profile, measured in SI units, on the GENE radial grid (1D array)
        Outputs:
          fluxes        dict containing:
                           'pi': heat flux, measured in SI units, on the GENE radial grid (1D array)

        Side Effects: genecomm_lowlevel is currently set such that the same checkpoint suffix (_000) is used every run.  This means that all of
        the GENE output files, including the checkpoint file, are overwritten each time.  GENE writes out two checkpoint files when it completes
        a simulation: checkpoint and s_checkpoint [with the suffix, this becomes checkpoint_000 and s_checkpoint_000].  These checkpoint files
        are contain the same information.  They each contain the perturbed distribution function f1.

        When GENE starts a new simulation, it will look for the file checkpoint_000.  If that file does not exist GENE will look for
        s_checkpoint_000.  If that does not exist, GENE looks for a checkpoint from the previous iteration.  If that doesn't exist,
        GENE will run without a checkpoint, and start from scratch.
        """
        pressureGeneGrid = profiles['pi']  # ion pressure on GENE grid

        # convert pressure profile into a temperature
        temperatureGeneGrid = pressure_to_temperature(pressureGeneGrid, self.densityGeneGrid)

        # convert the temperature from SI to GENE's normalized units
        temperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(temperatureGeneGrid)

        # place temperature into a 2D array, species x space
        temperatureHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        temperatureHatGeneGridAllSpecies[0, :] = temperatureHatGeneGrid

        # run GENE and get heat flux on GENE's grid
        (dVdxHat, sqrt_gxx, avgParticleFluxHatGeneGridAllSpecies, avgHeatFluxHatGeneGridAllSpecies) = self.geneInterface.call_gene(
            self.simulationTime, self.densityHatGeneGridAllSpecies, temperatureHatGeneGridAllSpecies)

        # extract individual heat flux
        avgHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[0, :]

        # convert GENE's normalized heat flux into SI units
        avgHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)

        fluxes = {}
        fluxes['pi'] = avgHeatFluxGeneGrid
        return fluxes

    def set_simulation_time(self, simulationTime):
        """Adjust how long GENE will run for when called to perform turbulence simulations.

        Inputs:
          simulationTime    GENE parameter that controls how long a simulation is run.  Measured in Lref/cref (scalar)
        """
        self.simulationTime = simulationTime
    
    def _create_gene_interface(self):
        """Instantiate an object that provides an interface to GENE.

        genecomm_lowlevel.GeneInterface expects quantities to be given in GENE normalized units.  Additionally, quantities must exist
        on GENE's radial grid, xGene.

        Outputs:
          geneInterface     Instance of genecomm_lowlevel.GeneInterface, enables running GENE
        """
        # convert SI quantities to what GENE expects
        rhoGeneInterface = genecomm_unitconversion.radius_SI_to_libgenetango_input(self.xGeneGrid, self.minorRadius)  # convert to rho = r/a

        rhoStar = genecomm_unitconversion.calculate_consistent_rhostar(self.Tref, self.Bref, self.mref, self.minorRadius)

        # create a few arrays that geneInterface expects
        mass = np.array([self.ionMass])
        charge = np.array([self.ionCharge])

        geneInterface = genecomm_lowlevel.GeneInterface(
            rho=rhoGeneInterface, mass=mass, charge=charge, safetyFactor=self.safetyFactorGeneGrid,
            Lref=self.Lref, Bref=self.Bref, rhoStar=rhoStar, Tref=self.Tref, nref=self.nref,
            pseudoGene=self.pseudoGene)
        return geneInterface


class GeneComm_CheaseSingleIonKineticElectrons:
    """Class-based interface for Tango to call GENE, when GENE is run with a single ion species and kinetic electrons, and
    using magnetic geometry from a CHEASE h5 file.

    Tango requires an object with a get_flux() method, which this class provides.  Except where specifically noted otherwise,
    all quantities are stored in SI units.

    With a kinetic ions and kinetic electrons, the density profile can evolve in time.  Hence, in this class, there is no
    initial density profile that is stored.  Rather, the density profile is provided on every call to GENE.  Furthermore, the
    particle flux is returned, not ignored as in the case with adiabatic electrons.
    
    **** CAUTION.  Unlike the previous GeneComm for kinetic electrons, due to an implementation detail, this class
    assumes that ELECTRONS are the FIRST species, and ions are the second.  This is something that comes from the order
    of how species are specified in the GENE parameters file.
    """
    def __init__(self, cheaseTangoData=None, Tref=None, nref=None,
                 xTangoGrid=None, xGeneGrid=None, mass=None, charge=None,
                 pseudoGene=False):
        """Constructor.

        Inputs:
          cheaseTangoData           container with Chease data (instance of CheaseTangoData)
                                      From the chease data, get Bref, Lref, minor radius
          Tref                      GENE reference temperature, measured in keV (scalar)
          nref                      GENE reference density, measured in 10^19 m^-3 (scalar)
          xTangoGrid                radial grid used by Tango (array)
          xGeneGrid                 radial grid used by GENE, with coordinate x = rho_tor (array)
          mass                      species masses, in proton masses (1D array, by species)
          charge                    species charges, in elementary charges (1D array, by species)
          pseudoGene                False for normal GENE run, True for a pseudo call that does not run GENE but is used to test code (Boolean)
        """
        self.numSpecies = 2
        self.Bref = cheaseTangoData.Bref
        self.Lref = cheaseTangoData.Lref
        self.minorRadius = cheaseTangoData.minorRadius

        self.Tref = Tref
        self.nref = nref
        self.mref = mass[0]       # measured in proton masses.  Ions must be the 0 species.

        # when using CHEASE, GENE ignores the safety factor from the python interface.  Use junk here
        self.safetyFactorGeneGrid = -1 * xGeneGrid

        self.xTangoGrid = xTangoGrid  # x = rho_tor when using CHEASE geometry
        self.xGeneGrid = xGeneGrid
        self.mass = mass  # measured in (proton masses)
        self.charge = charge  # measured in electron charge
        self.simulationTime = None  # measured in Lref/cref

        self.numRadialGridPointsGENE = len(xGeneGrid)
        self.pseudoGene = pseudoGene

        # create the interface for calling gene
        self.geneInterface = self._create_gene_interface()

    #### Interface for Tango  ####
    def get_flux(self, profiles):
        """Run GENE with a given profiles.  Return ion particle flux and ion & electron heat fluxes.

        According to the documentation for the Turbulence Handler, get_flux() must take as input the profiles on the turbulence grid and return
        the turbulent fluxes on the turbulence grid (see lodestro_method.py)

        Inputs:
            profiles    dict containing:
                            'n': density profile, measured in SI units, on the GENE radial grid (1D array)
                            'pi': ion pressure profile, measured in SI units, on the GENE radial grid (1D array)
                            'pe': electron pressure profile, measured in SI units, on the GENE radial grid (1D array)
        Outputs:
            fluxes      dict containing:
                            'n': particle flux, measured in SI units, on the GENE radial grid (1D array)
                            'pi': ion heat flux, measured in SI units, on the GENE radial grid (1D array)
                            'pe': electron heat flux, measured in SI units, on the GENE radial grid (1D array)

        Here, density 'n' represents the ion density, and particle flux represents the ion particle flux, which is a relevant distinction
        if the ion charge is not equal to 1.
        
        IMPORTANT: It is also assumed that GENE is run where the ions are "species 1" and the electrons are "species 2".

        Side Effects: genecomm_lowlevel is currently set such that the same checkpoint suffix (_000) is used every run.  This means that all of
        the GENE output files, including the checkpoint file, are overwritten each time.  GENE writes out two checkpoint files when it completes
        a simulation: checkpoint and s_checkpoint [with the suffix, this becomes checkpoint_000 and s_checkpoint_000].  These checkpoint files
        contain the same information.  They each contain the perturbed distribution function f1.

        When GENE starts a new simulation, it will look for the file checkpoint_000.  If that file does not exist GENE will look for
        s_checkpoint_000.  If that does not exist, GENE looks for a checkpoint from the previous iteration.  If that doesn't exist,
        GENE will run without a checkpoint, and start from scratch.
        """
        densityGeneGrid = profiles['n']
        ionPressureGeneGrid = profiles['pi']
        electronPressureGeneGrid = profiles['pe']

        # convert pressure profile into a temperature, as GENE takes temperature as input
        ionTemperatureGeneGrid = pressure_to_temperature(ionPressureGeneGrid, densityGeneGrid)
        electronTemperatureGeneGrid = pressure_to_temperature(electronPressureGeneGrid, densityGeneGrid)

        # convert density and temperature from SI to GENE's normalized units
        densityHatGeneGrid = genecomm_unitconversion.density_SI_to_gene(densityGeneGrid)
        ionTemperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(ionTemperatureGeneGrid)
        electronTemperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(electronTemperatureGeneGrid)

        # place density into a 2D array, species x space... 
        # density represents ION density, so electron density = ion charge * ion density
        densityHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        densityHatGeneGridAllSpecies[1, :] = densityHatGeneGrid         # ion density
        densityHatGeneGridAllSpecies[0, :] = self.charge[0] * densityHatGeneGrid # electron density

        # place temperature into a 2D array, species x space
        temperatureHatGeneGridAllSpecies = np.zeros((self.numSpecies, self.numRadialGridPointsGENE), dtype=np.float64)
        temperatureHatGeneGridAllSpecies[1, :] = ionTemperatureHatGeneGrid
        temperatureHatGeneGridAllSpecies[0, :] = electronTemperatureHatGeneGrid

        # run GENE and get heat flux on GENE's grid
        (dVdxHat, sqrt_gxx, avgParticleFluxHatGeneGridAllSpecies, avgHeatFluxHatGeneGridAllSpecies) = self.geneInterface.call_gene(
                    self.simulationTime, densityHatGeneGridAllSpecies, temperatureHatGeneGridAllSpecies)

        # extract individual particle and heat flux
        avgIonParticleFluxHatGeneGrid = avgParticleFluxHatGeneGridAllSpecies[1, :]
        avgIonHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[1, :]
        avgElectronHeatFluxHatGeneGrid = avgHeatFluxHatGeneGridAllSpecies[0, :]

        # convert GENE's normalized particle and heat flux into SI units
        avgIonParticleFluxGeneGrid = genecomm_unitconversion.particleflux_gene_to_SI(avgIonParticleFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        avgIonHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgIonHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        avgElectronHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgElectronHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        
        fluxes = {}
        fluxes['n'] = avgIonParticleFluxGeneGrid
        fluxes['pi'] = avgIonHeatFluxGeneGrid
        fluxes['pe'] = avgElectronHeatFluxGeneGrid
        return fluxes

    def set_simulation_time(self, simulationTime):
        """Adjust how long GENE will run for when called to perform turbulence simulations.

        Inputs:
          simulationTime    GENE parameter that controls how long a simulation is run.  Measured in Lref/cref (scalar)
        """
        self.simulationTime = simulationTime
    
    def _create_gene_interface(self):
        """Instantiate an object that provides an interface to GENE.

        genecomm_lowlevel.GeneInterface expects quantities to be given in GENE normalized units.  Additionally, quantities must exist
        on GENE's radial grid, xGene.

        Outputs:
          geneInterface     Instance of genecomm_lowlevel.GeneInterface, enables running GENE
        """
        # convert SI quantities to what GENE expects
        rhoGeneInterface = genecomm_unitconversion.radius_SI_to_libgenetango_input(self.xGeneGrid, self.minorRadius)  # convert to rho = r/a

        rhoStar = genecomm_unitconversion.calculate_consistent_rhostar(self.Tref, self.Bref, self.mref, self.minorRadius)

        geneInterface = genecomm_lowlevel.GeneInterface(
                rho=rhoGeneInterface, mass=self.mass, charge=self.charge, safetyFactor=self.safetyFactorGeneGrid,
                Lref=self.Lref, Bref=self.Bref, rhoStar=rhoStar, Tref=self.Tref, nref=self.nref,
                pseudoGene=self.pseudoGene)
        return geneInterface

        
def pressure_to_temperature(pressureProfile, densityProfile):
    """Convert pressure to temperature using p=nT, or T = p/n.  Nominally, everything is expected to be in SI units.
      Inputs:
        pressureProfile     (array)
        densityProfile      (array)
      Outputs:
        temperatureProfile  (array)
    """
    assert np.all(densityProfile > 0), "Must have positive densities!"
    temperatureProfile = pressureProfile / densityProfile
    return temperatureProfile
