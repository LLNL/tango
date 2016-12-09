"""
genecomm_lowlevel

Low level access to libgene_tango, which provides a Python interface to calling GENE using f2py bindings.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
import scipy.interpolate
try:
    import gene_tango
except ImportError:
    print("Warning: gene_tango unable to be imported.  Running GENE will not work.")
        
#import logging
inttype=np.int
fltype=np.float64

class GeneInterface(object):
    """
    Class-based interface to GENE using the libgene_tango interface.  Everything here uses units that GENE expects and uses
    GENE's radial grid.  Unit conversion into/out of GENE's units and conversion to/from GENE's radial grid are performed elsewhere.
    """
    def __init__(self, rho=None, densityHat=None, safetyFactor=None,
                 ionMass=1, ionCharge=1, Lref=None, Bref=None, rhoStar=None, checkpointSuffix=0,
                 pseudoGene=False):
        """
        Set up and store the things that do not change (everything except the pressure/temperature profile).  See
        call_gene_low_level for a description of the inputs.
        
        Inputs:
          (...)                 see call_gene_low_level
          pseudoGene            False for normal GENE run, True for a pseudo call that does not run GENE but is used only to test code (Boolean)
        """
        self.fixedParams = {'rho':rho,  'densityHat':densityHat,  'safetyFactor':safetyFactor,
                            'ionMass':ionMass,  'ionCharge':ionCharge,  'Lref':Lref,  'Bref':Bref,
                            'rhoStar':rhoStar, 'checkpointSuffix':checkpointSuffix}
        self.pseudoGene = pseudoGene

    def call_gene(self, simulationTime, temperatureHat):
        """Run gene.
        
        Inputs:
          simulationTime        amount of time for which GENE will run.  Measured in Lref/cref (scalar)
          temperatureHat        temperature profile in keV (array)
        Outputs:
          dVdxHat               dVhat/dxhat on grid psi_out, in GENE's normalized units  (array)
          sqrt_gxx              sqrt(g_xx) on grid psi_out (array)
          avgParticleFluxHat    time & flux-surface-averaged particle flux on grid psi_out, in GENE's normalized units (array)
          avgHeatFluxHat        time & flux-surface-averaged heat flux on grid psi_out (array)
          temperatureOutput     ? not sure
          densityOutput         ? not sure
        """
    
        if self.pseudoGene==False:
            (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = call_gene_low_level(simulationTime=simulationTime, temperatureHat=temperatureHat, **self.fixedParams)
        else:
            (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = pseudo_call_gene_low_level(simulationTime=simulationTime, temperatureHat=temperatureHat, **self.fixedParams)
        return (dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput)
                                                                                                                        

def call_gene_low_level(simulationTime=None, rho=None, temperatureHat=None, densityHat=None, safetyFactor=None, 
                        ionMass=1, ionCharge=1, Lref=None, Bref=None, rhoStar=None, checkpointSuffix=0):
    """
    Call GENE using the libgene_tango interface.
    
    Note, for analytic concentric circular geometry, the libgene_tango interface takes as input a radial coordinate r/a, where a
    is the minor radius.  However, the rest of GENE uses x=r as the radial coordinate (whatever x is, it has dimensions of length).
    However, for numerical purposes, GENE normalizes things to a "hat" variable, e.g., xhat = x/Lref, That = T/Tref, etc.
    
    Note that some arrays within the function are created with order='F', meaning Fortran-style contiguous.  For arrays that are
    only 1D, this is not strictly necessary because they are already both C-contiguous and F-contiguous.  So, for example, while
    temperatureHat is placed into an F-contiguous array, the 1D arrays psiIn, psiOut, safetyFactor, and magneticShear are not
    explicitly placed into F-contiguous arrays.
    
    Inputs:
      simulationTime        amount of time for GENE to simulate in this call.  Measured in Lref/cref (scalar).  labeled simtimelim in libgene_tango.f90
      rho                   radial grid, rho = r/a for input and output profiles (array)
      temperatureHat        temperature profile in keV on grid rho (array)
      densityHat            density profile in 10^19 m^-3 on grid rho (array)
      safetyFactor          safety factor q on grid rho (array)
      ionMass               ion mass in proton masses (scalar).  Default = 1
      ionCharge             ion charge in elementary charges (scalar).  Default = 1
      Lref                  reference length Lref.  Measured in meters.  (scalar)
      Bref                  reference magnetic field, equal to B0 for analytic geometry.  Measured in Tesla. (scalar)
      rhoStar               Parameter equal to rhoref/a = (cref/Omegaref)/a.  Dimensionless. (scalar)
      checkpointSuffix      What the libgene_tango interface refers to as the "iteration number".  This is the number that gets appended to checkpoint
                              filenames.  E.g., with the default value of 0, filenames end in _000 added. (integer).  Default = 0.  Labeled 'it' in libgene_tango.f90
    
    Assumptions (input parameters to gene_tango.gene_tango that I set as fixed here):
      electrostatic         = True
      ionSpeciesCount       = 1   total number of ion species
      toroidal flow velocity = 0
      
    Outputs:
      MPIrank               Rank from the MPI environment within GENE
      dVdxHat               dVhat/dxhat on grid rho, in GENE's normalized units  (array)
      sqrt_gxx              sqrt(g_xx) on grid rho (array)
      avgParticleFluxHat    time & flux-surface-averaged particle flux on grid rho, in GENE's normalized units (array)
      avgHeatFluxHat        time & flux-surface-averaged heat flux on grid rho (array)
      temperatureOutput     ? not sure
      densityOutput         ? not sure
    """
    # check inputs have been provided
    for var in (simulationTime, rho, temperatureHat, densityHat, safetyFactor, Lref, Bref, rhoStar):
        if var is None:
            #logging.error("Input variables must be provided in call_gene_low_level.")
            raise ValueError
    
    ##################### Boilerplate #####################
    electrostatic=True       # Electrostatic simulation
    ionSpeciesCount = 1      # labeled n_spec_in in libgene_tango.f90
    numRadialGridPts = len(rho)   # labeled px in libgene_tango.f90
    
    # Set up input arrays for GENE...
    temperatureHatGENE = np.ones((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')    # labeled temp_io in libgene_tango.f90
    temperatureHatGENE[:, 0] = temperatureHat
    densityHatGENE = np.ones((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')        # labeled dens_io in libgene_tango.f90
    densityHatGENE[:, 0] = densityHat
    ionMassGENE = np.array([ionMass], dtype=fltype, order='F')      # labeled mass_in in libgene_tango.f90
    toroidalVelocityGENE = np.zeros(numRadialGridPts, dtype=fltype, order='F')    # labeled vrot_in in libgene_tango.f90
    ionChargeGENE = np.array([ionCharge], dtype=fltype, order='F')  # labeled charge_in in libgene_tango.f90
    inverseAspectRatio = -9999.99   # used only for local simulations so irrelevant
    
    # Set up output arrays for GENE...
    dVdxHat = np.empty(numRadialGridPts, dtype=fltype, order='F')
    sqrt_gxx = np.empty(numRadialGridPts, dtype=fltype, order='F')
    avgParticleFluxHat = np.empty((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')
    avgHeatFluxHat = np.empty((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')
    temperatureOutput = np.empty((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')
    densityOutput = np.empty((numRadialGridPts, ionSpeciesCount), dtype=fltype, order='F')
    
    # perform whatever calculations are required
    magneticShear = calculate_magnetic_shear(safetyFactor, rho)
    ####################### End Boilerplate ######################
    
    #logging.info('Running GENE...')
    (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput) = gene_tango.gene_tango(
               checkpointSuffix, electrostatic, simulationTime, rho, temperatureHatGENE, densityHatGENE, ionMassGENE, ionChargeGENE, toroidalVelocityGENE,
               rhoStar, safetyFactor, magneticShear, inverseAspectRatio, Lref, Bref,
               numRadialGridPts, ionSpeciesCount)
    #logging.info('GENE finished!')
    
    # convert from Fortran-contiguous to C-contiguous arrays for rest of Python code
    dVdxHat = np.ascontiguousarray(dVdxHat)
    sqrt_gxx = np.ascontiguousarray(sqrt_gxx)
    avgParticleFluxHat = np.ascontiguousarray(avgParticleFluxHat)
    avgHeatFluxHat = np.ascontiguousarray(avgHeatFluxHat)
    temperatureOutput = np.ascontiguousarray(temperatureOutput)
    densityOutput = np.ascontiguousarray(densityOutput)
    
    return (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput)
    
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

    
def pseudo_call_gene_low_level(simulationTime=None, rho=None, temperatureHat=None, densityHat=None, safetyFactor=None, 
                        ionMass=1, ionCharge=1, Lref=None, Bref=None, rhoStar=None, checkpointSuffix=0):
    """Function  to emulate a call to GENE with the same input arguments and return values.
    
    Used for testing other code when the overhead of an actual startup of GENE is not needed.  Of course, this can only
    be used to test correctness of syntax, not correctness of values.
    """
    MPIrank = 1
    dVdxHat = np.ones_like(rho)
    sqrt_gxx = np.ones_like(rho)
    avgParticleFluxHat = np.ones_like(rho)
    avgHeatFluxHat = np.ones_like(rho)
    temperatureOutput = np.ones_like(rho)
    densityOutput = np.ones_like(rho)
    return (MPIrank, dVdxHat, sqrt_gxx, avgParticleFluxHat, avgHeatFluxHat, temperatureOutput, densityOutput)