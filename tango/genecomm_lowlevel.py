"""
genecomm_lowlevel

Low level access to libgene_tango, which provides a Python interface to calling GENE using f2py bindings.

A note about quasineutrality.  When the number of species is more than one, the libgene_tango intercface enforces
quasineutrality by modifying the density of the final species.  [Unclear what happens with two ion species and
adiabatic electrons]

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
import scipy.interpolate

import tango.tango_logging as tlog

try:
    import gene_tango
except ImportError:
    print("Warning: gene_tango unable to be imported.  Running GENE will not work.")
        
inttype=np.int
fltype=np.float64

class GeneInterface(object):
    """
    Class-based interface to GENE using the libgene_tango interface.  Everything here uses units that GENE expects and uses
    GENE's radial grid.  Unit conversion into/out of GENE's units and conversion to/from GENE's radial grid are performed elsewhere.
    """
    def __init__(self, rho=None, mass=None, charge=None, safetyFactor=None, Lref=None, Bref=None,
                 rhoStar=None, Tref=None, nref=None, checkpointSuffix=0, pseudoGene=False):
        """
        Set up and store the things that do not change (everything except the pressure/temperature profile).  See
        call_gene_low_level for a description of the inputs.
        
        Inputs:
          (...)                 see call_gene_low_level
          pseudoGene            False for normal GENE run, True for a pseudo call that does not run GENE but is used only to test code (Boolean)
        """
        self.fixedParams = {
                'rho':rho,  'mass':mass,  'charge':charge,  'safetyFactor':safetyFactor,
                'Lref':Lref,  'Bref':Bref,  'rhoStar':rhoStar, 'Tref':Tref,  'nref':nref,  'checkpointSuffix':checkpointSuffix}
        self.pseudoGene = pseudoGene

    def call_gene(self, simulationTime, densityHatAllSpecies, temperatureHatAllSpecies):
        """Run gene.
        
        Inputs:
          simulationTime                    amount of time for which GENE will run.  Measured in Lref/cref (scalar)
          densityHatAllSpecies              species density profiles, measured in 10^19 m^-3 on grid rho (2D array, species x space)
          temperatureHatAllSpecies          species temperature profiles, measured in keV on grid rho (2D array, species x space)
        Outputs:
          dVdxHat                           dVhat/dxhat on grid rho, in GENE's normalized units  (array)
          sqrt_gxx                          sqrt(g_xx) on grid rho (array)
          avgParticleFluxHatAllSpecies      time & flux-surface-averaged particle flux on grid rho, in GENE's normalized units (2D array, species x space)
          avgHeatFluxHatAllSpecies          time & flux-surface-averaged heat flux on grid rho (2D array, species x space)
        """
    
        if self.pseudoGene==False:
            (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies) = call_gene_low_level(
                simulationTime=simulationTime, densityHatAllSpecies=densityHatAllSpecies, temperatureHatAllSpecies=temperatureHatAllSpecies, **self.fixedParams)
        else:
            (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies) = pseudo_call_gene_low_level(
                simulationTime=simulationTime, densityHatAllSpecies=densityHatAllSpecies, temperatureHatAllSpecies=temperatureHatAllSpecies, **self.fixedParams)
        return (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies)
                                                                                                                        

def call_gene_low_level(
        simulationTime=None, rho=None, mass=None, charge=None, densityHatAllSpecies=None, temperatureHatAllSpecies=None, safetyFactor=None, 
        Lref=None, Bref=None, rhoStar=None, Tref=None, nref=None, checkpointSuffix=0):
    """
    Call GENE using the libgene_tango interface.
    
    Note, for analytic concentric circular geometry, the libgene_tango interface takes as input a radial coordinate r/a, where a
    is the minor radius.  However, the rest of GENE uses x=r as the radial coordinate (whatever x is, it has dimensions of length).
    However, for numerical purposes, GENE normalizes things to a "hat" variable, e.g., xhat = x/Lref, That = T/Tref, etc.
    
    Note that some arrays within the function are created with order='F', meaning Fortran-style contiguous.  For arrays that are
    only 1D, this is not strictly necessary because they are already both C-contiguous and F-contiguous.  So, for example, while
    temperatureHat is placed into an F-contiguous array, the 1D arrays safetyFactor and magneticShear are not explicitly placed
    into F-contiguous arrays.
    
    Inputs:
      simulationTime            amount of time for GENE to simulate in this call.  Measured in Lref/cref (scalar).  labeled simtimelim in libgene_tango.f90
      rho                       radial grid, rho = r/a for input and output profiles.  Dimensionless. (array)
      mass                      species masses, in proton masses (1D array, by species)
      charge                    species charges, in elementary charges (1D array, by species)
      densityHatAllSpecies      species density profiles in 10^19 m^-3 on grid rho (2D array, species x space)
      temperatureHatAllSpecies  species temperature profiles in keV on grid rho (2D array, species x space)
      safetyFactor              safety factor q on grid rho (array)
      Lref                      reference length Lref.  Measured in meters.  (scalar)
      Bref                      reference magnetic field, equal to B0 for analytic geometry.  Measured in Tesla. (scalar)
      rhoStar                   Parameter equal to rhoref/a = (cref/Omegaref)/a.  Dimensionless. (scalar)
      Tref                      reference temperature Tref.  Measured in keV.  Used by GENE to determine velocity gridding so Tref should be roughly equal to plasma temperature for best gridding. (scalar)
      nref                      reference density nref.  Measured in 10^19 m^-3.  nref be roughly equal to plasma density. (scalar)
      checkpointSuffix          (optional) What the libgene_tango interface refers to as the "iteration number".  This is the number that gets appended to checkpoint
                                  filenames.  E.g., with the default value of 0, filenames end in _000 added. (integer).  Default = 0.  Labeled 'it' in libgene_tango.f90
      
                              
    Assumptions (input parameters to gene_tango.gene_tango that I set as fixed here):
      electrostatic         = True
      toroidal flow velocity = 0
      
    Outputs:
      dVdxHat                       dVhat/dxhat on grid rho, in GENE's normalized units  (1D array)
      sqrt_gxx                      sqrt(g_xx) on grid rho (1D array)
      avgParticleFluxHatAllSpecies  time & flux-surface-averaged particle flux on grid rho, in GENE's normalized units (2D array, species x space)
      avgHeatFluxHatAllSpecies      time & flux-surface-averaged heat flux on grid rho (2D array, species x space)
    """
    # check inputs have been provided
    for var in (simulationTime, rho, mass, charge, densityHatAllSpecies, temperatureHatAllSpecies, safetyFactor, Lref, Bref, rhoStar, Tref, nref):
        if var is None:
            #logging.error("Input variables must be provided in call_gene_low_level.")
            raise ValueError
    
    ##################### Boilerplate #####################
    electrostatic=True       # Electrostatic simulation
    numSpecies = mass.shape[0]
    numRadialGridPts = len(rho)   # labeled px in libgene_tango.f90
    
    # Set up input arrays for passing to GENE
    massGENE = np.array(mass, dtype=fltype, order='F')
    chargeGENE = np.array(charge, dtype=fltype, order='F')
    # for 2D arrays: make a new copy.  Transpose changes C-order to Fortran-order.
    temperatureHatAllSpeciesGENE = np.copy(temperatureHatAllSpecies).T      # labeled temp_io in libgene_tango.f90
    densityHatAllSpeciesGENE = np.copy(densityHatAllSpecies).T              # labeled dens_io in libgene_tango.f90
    
    toroidalVelocityGENE = np.zeros(numRadialGridPts, dtype=fltype, order='F')    # labeled vrot_in in libgene_tango.f90
    inverseAspectRatio = -9999.99   # used only for local simulations so irrelevant
    
    # Set up output arrays for GENE...
    dVdxHat = np.empty(numRadialGridPts, dtype=fltype, order='F')
    sqrt_gxx = np.empty(numRadialGridPts, dtype=fltype, order='F')
    avgParticleFluxHatAllSpecies = np.empty((numRadialGridPts, numSpecies), dtype=fltype, order='F')
    avgHeatFluxHatAllSpecies = np.empty((numRadialGridPts, numSpecies), dtype=fltype, order='F')
    temperatureOutput = np.empty((numRadialGridPts, numSpecies), dtype=fltype, order='F')
    densityOutput = np.empty((numRadialGridPts, numSpecies), dtype=fltype, order='F')
    
    # perform whatever calculations are required
    magneticShear = calculate_magnetic_shear(safetyFactor, rho)
    ####################### End Boilerplate ######################
    
    tlog.info('Running GENE...')
    (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies, temperatureOutput, densityOutput) = gene_tango.gene_tango(
               checkpointSuffix, electrostatic, simulationTime, rho, temperatureHatAllSpeciesGENE, densityHatAllSpeciesGENE, massGENE, chargeGENE, toroidalVelocityGENE,
               rhoStar, Tref, nref, safetyFactor, magneticShear, inverseAspectRatio, Lref, Bref,
               numRadialGridPts, numSpecies)
    tlog.info('GENE finished!')
    
    # convert from Fortran-contiguous to C-contiguous arrays for rest of Python code
    dVdxHat = np.ascontiguousarray(dVdxHat) # for a 1D array, doesn't actually do anything...
    sqrt_gxx = np.ascontiguousarray(sqrt_gxx)
    # for 2D arrays, transpose: this reshapes dimensions from (space x species) to (species x space) and also ensures arrays are C-contiguous
    avgParticleFluxHatAllSpecies = avgParticleFluxHatAllSpecies.T
    avgHeatFluxHatAllSpecies = avgHeatFluxHatAllSpecies.T
    
    return (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies)
    
def calculate_magnetic_shear(safetyFactor, psi):
    """compute the magnetic shear s from the safety factor q, given by the formula
              s = (psi/q) dq/dpsi
    Inputs:
      safetyFactor      safety factor q on grid psi (array)
      psi               radial coordinate grid (array)
    Outputs:
      magneticShear     magnetic shear s on grid psi (array)
    """
    # construct a spline from the safety factor in order to take its derivative
    q_spline = scipy.interpolate.InterpolatedUnivariateSpline(psi, safetyFactor)
    dq_dpsi_spline = q_spline.derivative()
    dq_dpsi = dq_dpsi_spline(psi)
    magneticShear = dq_dpsi * psi/safetyFactor
    return magneticShear

    
def pseudo_call_gene_low_level(
        simulationTime=None, rho=None, mass=None, charge=None, densityHatAllSpecies=None, temperatureHatAllSpecies=None, safetyFactor=None, 
        Lref=None, Bref=None, rhoStar=None, Tref=None, nref=None, checkpointSuffix=0):
    """Function to emulate a call to GENE with the same input arguments and return values.
    
    Used for testing other code when the overhead of an actual startup of GENE is not needed.  Of course, this can only
    be used to test correctness of syntax, not correctness of values.
    """
    
    # check inputs have been provided
    for var in (simulationTime, rho, mass, charge, densityHatAllSpecies, temperatureHatAllSpecies, safetyFactor, Lref, Bref, rhoStar, Tref, nref):
        if var is None:
            #logging.error("Input variables must be provided in call_gene_low_level.")
            raise ValueError
            
    dVdxHat = np.ones_like(rho)
    sqrt_gxx = np.ones_like(rho)
    avgParticleFluxHatAllSpecies = np.ones_like(temperatureHatAllSpecies)
    avgHeatFluxHatAllSpecies = np.ones_like(temperatureHatAllSpecies)
    return (dVdxHat, sqrt_gxx, avgParticleFluxHatAllSpecies, avgHeatFluxHatAllSpecies)
