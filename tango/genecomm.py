"""
genecomm

High-level communcation between GENE and the rest of Tango.

Responsibilities:
1. Unit conversion   Tango speaks in SI, GENE speaks in normalized units.  This module converts between the two.
2. Encapsulation     Internally maintain everything that stays fixed; provide an interface that can be called by Tango
                       with only the quantity that changes (pressure profile)
3. Provide interface to change certain quantities, e.g., time step simulation time

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
from . import genecomm_lowlevel
from . import genecomm_unitconversion



class GeneComm(object):
    """Class-based interface for Tango to call GENE.
    
    Tango requires an object with a get_flux() method, which this class provides.  Except where specifically noted otherwise,
    all quantities are stored in SI units.
    """
    def __init__(self, Bref=None, Lref=None, Tref=1, nref=1, B0=None, minorRadius=None, majorRadius=None, safetyFactorGeneGrid=None,
                 psiTangoGrid=None, psiGeneGrid=None, densityTangoGrid=None, ionMass=1, ionCharge=1, gridMapper=None,
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
          ionMass                   mass of ion species, measured in proton masses (scalar)
          ionCharge                 charge of ion species, measured in electron charge (scalar)
          gridMapper                object for transforming the profiles and transport coefficients between the Tango grid and
                                      the GENE grid.  See interfacegrids_gene.py (object)
          pseudoGene                False for normal GENE run, True for a pseudo call that does not run GENE but is used to test code (Boolean)
        """
        self.Bref = Bref
        self.Lref = Lref
        self.Tref = Tref
        self.nref = nref
        self.mref = 1       # measured in proton masses
        self.B0 = B0
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        self.safetyFactorGeneGrid = safetyFactorGeneGrid
        self.psiTangoGrid = psiTangoGrid  # psi = r
        self.psiGeneGrid = psiGeneGrid # psi = x = r
        self.densityTangoGrid = densityTangoGrid
        self.ionMass = ionMass # measured in mref (proton masses)
        self.ionCharge = ionCharge # measured in electron charge
        self.simulationTime = None  # measured in Lref/cref
        
        assert hasattr(gridMapper, 'map_profile_onto_turb_grid') and callable(getattr(gridMapper, 'map_profile_onto_turb_grid'))
        assert hasattr(gridMapper, 'map_transport_coeffs_onto_transport_grid') and callable(getattr(gridMapper, 'map_transport_coeffs_onto_transport_grid'))
        self.gridMapper = gridMapper
        self.pseudoGene = pseudoGene
        
        self.densityGeneGrid = self.gridMapper.map_profile_onto_turb_grid(densityTangoGrid)
        
        self.geneInterface = self._create_gene_interface()
    
    
    #### Interface for Tango  ####
    def get_flux(self, pressureGeneGrid):
        """Run GENE with a given pressure profile.  Return heat flux.
        
        According to the documentation for the Turbulence Handler, GetFlux() must take as input the profile on the turbulence grid and return
        the turbulent flux on the turbulence grid (see lodestro_method.py)
        
        Inputs:
          pressureGeneGrid      pressure profile, measured in SI units, on the GENE radial grid.
        Outputs:
          avgHeatFluxGeneGrid   heat flux, measured in SI units, on the GENE radial grid.
          
        Side Effects: genecomm_lowlevel is currently set such that the same checkpoint suffix (_000) is used every run.  This means that all of
        the GENE output files, including the checkpoint file, are overwritten each time.  GENE writes out two checkpoint files when it completes
        a simulation: checkpoint and s_checkpoint [with the suffix, this becomes checkpoint_000 and s_checkpoint_000].  These checkpoint files
        are contain the same information.  They each contain the perturbed distribution function f1.
        
        When GENE starts a new simulation, it will look for the file checkpoint_000.  If that file does not exist GENE will look for
        s_checkpoint_000.  If that does not exist, GENE looks for a checkpoint from the previous iteration.  If that doesn't exist,
        GENE will run without a checkpoint, and start from scratch.
        """
        
        # convert pressure profile into a temperature
        temperatureGeneGrid = pressure_to_temperature(pressureGeneGrid, self.densityGeneGrid)
        
        # convert the temperature from SI to GENE's normalized units
        temperatureHatGeneGrid = genecomm_unitconversion.temperature_SI_to_gene(temperatureGeneGrid, self.Tref)
        
        # run GENE and get heat flux on GENE's grid
        (dVdxHat, sqrt_gxx, avgParticleFluxHatGeneGrid, avgHeatFluxHatGeneGrid, temperatureOutput, densityOutput) = self.geneInterface.call_gene(self.simulationTime, temperatureHatGeneGrid)
                
        # convert GENE's normalized heat flux into SI units
        avgHeatFluxGeneGrid = genecomm_unitconversion.heatflux_gene_to_SI(avgHeatFluxHatGeneGrid, self.nref, self.Tref, self.mref, self.Bref, self.Lref)
        return avgHeatFluxGeneGrid
        
    
    def set_simulation_time(self, simulationTime):
        """Adjust how long GENE will run for when called to perform turbulence simulations.
        
        Inputs:
          simulationTime    GENE parameter that controls how long a simulation is run.  Measured in Lref/cref (scalar)
        """
        self.simulationTime = simulationTime
    
    #### methods for talking to libgene_tango ####
    def _create_gene_interface(self):
        """Instantiate an object that provides an interface to GENE.
        
        genecomm_lowlevel.GeneInterface expects quantities to be given in GENE normalized units.  Additionally, quantities must exist
        on GENE's radial grid, psi_gene.
        
        Outputs:
          geneInterface     Instance of genecomm_lowlevel.GeneInterface, enables running GENE
        """
        # convert SI quantities to what GENE expects
        rhoGeneInterface = genecomm_unitconversion.radius_SI_to_libgenetango_input(self.psiGeneGrid, self.minorRadius)  # convert to rho = r/a
        densityHatTangoGrid = genecomm_unitconversion.density_SI_to_gene(self.densityTangoGrid)
        densityHatGeneGrid = self.gridMapper.map_profile_onto_turb_grid(densityHatTangoGrid)
                
        rhoStar = genecomm_unitconversion.calculate_consistent_rhostar(self.Tref, self.Bref, self.mref, self.minorRadius)
        geneInterface = genecomm_lowlevel.GeneInterface(rho=rhoGeneInterface, densityHat=densityHatGeneGrid,
                                                        safetyFactor=self.safetyFactorGeneGrid, ionMass=self.ionMass, ionCharge=self.ionCharge,
                                                        Lref=self.Lref, Bref=self.Bref, rhoStar=rhoStar,
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