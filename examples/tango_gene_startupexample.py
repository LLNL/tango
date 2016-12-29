"""Test run for Tango/GENE on NERSC Machines
"""

from __future__ import division, absolute_import
import numpy as np
import tango
import tango.gene_startup

def initialize_iteration_parameters():
    maxIterations = 1000
    thetaParams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdxThreshold': 10}
    EWMAParamTurbFlux = 0.30
    EWMAParamProfile = 0.30
    lmParams = {'EWMAParamTurbFlux': EWMAParamTurbFlux,
            'EWMAParamProfile': EWMAParamProfile,
            'thetaParams': thetaParams}
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)
    
class ComputeAllH(object):
    def __init__(self, turbhandler, HcontribTransportPhysics):
        self.turbhandler = turbhandler
        self.HcontribTransportPhysics = HcontribTransportPhysics
        #self.Vprime = HcontribTransportPhysics.profilesAll.Vprime
        #self.gradpsisq = HcontribTransportPhysics.profilesAll.gradpsisq
    def __call__(self, t, psi, pressure):
        """Define the contributions to the H coefficients
        
        Inputs:
          t         time (scalar)
          psi       radial coordinate in SI (array)
          pressure  pressure profile in SI (array)
        """
        timeDerivCoeff = 1.5 * np.ones_like(psi)
        H1 = self.HcontribTransportPhysics.time_derivative_to_H(timeDerivCoeff)
        
        # turbulent flux
        (H2_turb, H3_turb, extradata) = self.turbhandler.Hcontrib_turbulent_flux(pressure)
        
        # add some manually input diffusivity chi
        chi = 0.2 * np.ones_like(psi)   # 0.2 m^2/s
        (H2_adhoc, H3_adhoc) = self.HcontribTransportPhysics.Hcontrib_thermal_diffusivity(chi)
        
        # could also add neoclassical diffusivity chi here.
        # add the neoclassical diffusivity chi (for banana regime)
        #(H2_b, H3_b) = self.HcontribTransportPhysics.Hcontrib_neoclassical_thermal_diffusivity(p)

        # sum up the various neoclassical + turbulent + adhoc transport terms
        H2 = H2_turb + H2_adhoc
        H3 = H3_turb + H3_adhoc
        
        H4 = None
        H6 = None
        
        # Source
        S = 0.2 * np.ones_like(psi) # heat source
        H7 = self.HcontribTransportPhysics.source_to_H(S)
        
        return (H1, H2, H3, H4, H6, H7, extradata)

def problem_setup():
    """
    Still to be done: need to choose good profiles for
        ion temperature initial condition
        plasma density
        heat source
        tango and GENE radial grids
    """

    # Get most of the problem parameters
    (psiTango, psiGene, minorRadius, majorRadius, Vprime, gradPsiSq, B0, ionMass, ionCharge, densityProfile, Bref, Lref) = tango.parameters.get_default_parameters()
    Tref = 0.4
    nref = 1
    
    L = psiTango[-1] - psiTango[0]
    # create object for interfacing tango and GENE radial grids
       # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    gridMapper = tango.interfacegrids_gene.GridInterfaceTangoOutside(psiTango, psiGene)
    
    
    # specify a boundary condition for pressure at the outward radial boundary
    e = 1.60217662e-19          # electron charge
    temperatureRightBC = 400 * e  # temperature in J, 
    pressureRightBC = temperature_to_pressure(temperatureRightBC, densityProfile[-1])
    
    # specify a pressure initial condition
    temperatureIC_inkeV = 0.4 * np.ones_like(psiTango)
    temperatureIC = temperatureIC_inkeV * 1000 * e  # in SI units
    pressureIC = temperature_to_pressure(temperatureIC, densityProfile)
    
    # GENE setup
    fromCheckpoint = False    # true if restarting a simulation from an already-saved checkpoint
    #(geneFluxModel, MPIrank) = tango.gene_startup.setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityProfile, pressureIC, Bref, Lref, grids, fromCheckpoint)
    
         # use this version to test, if in an environment without GENE
    pseudoGene = True
    (geneFluxModel, MPIrank) = tango.gene_startup.setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityProfile, pressureIC, Bref, Lref, Tref, nref, gridMapper, fromCheckpoint, pseudoGene)
    # (geneFluxModel, MPIrank) = tango.gene_startup.pseudo_setup_gene_run(psiTango, psiGene, minorRadius, majorRadius, B0, ionMass, ionCharge, densityProfile, pressureIC, Bref, Lref, grids, fromCheckpoint)
    
    
    # other transport physics / physicsToH object creation
    profilesAll = tango.physics.initialize_profile_defaults(ionMass, densityProfile, psiTango, minorRadius, majorRadius, B0, Vprime, gradPsiSq)
    HcontribTransportPhysics = tango.physics_to_H.Hcontrib_TransportPhysics(profilesAll)
    
    # iteration parameters setup
    maxIterations, lmParams, tol = initialize_iteration_parameters()    
    
    # creation of turbulence handler
    dpsi = psiTango[1] - psiTango[0]    
    turbhandler = tango.lodestro_method.TurbulenceHandler(dpsi, psiTango, lmParams, geneFluxModel, grids, Vprime)
    
    # specify a source function?
    
    # initialize the compute all H object
    compute_all_H = ComputeAllH(turbhandler, HcontribTransportPhysics)
    t_array = np.array([0, 1e4])  # specify the timesteps to be used.
    return (MPIrank, L, psiTango, pressureRightBC, pressureIC, maxIterations, tol, geneFluxModel, turbhandler, compute_all_H, t_array)

def temperature_to_pressure(temperature, density):
    """Convert temperature to pressure (given a density).
    
    Inputs:
      temperature           temperature in J (scalar or array)
      density               density in m^-3 (scalar or array)
    Outputs
      pressure              pressure in J/m^3 (scalar or array)
    """
    pressure = density * temperature
    return pressure

def setup_f1_history_handler():
    pass

def setup_tango_checkpoint_handler():
    pass

def setup_tango_history_handler():
    pass
    
# ************************************************** #
####              START OF MAIN PROGRAM           ####
# ************************************************** #

(MPIrank, L, psi, pressureRightBC, pressureIC, maxIterations, tol, geneFluxModel, turbhandler, compute_all_H, t_array) = problem_setup()



# set up FileHandlers
f1HistoryHandler = tango.handlers.Savef1HistoryHandler(iterationInterval=50, basename='f1_iteration_history', genefile='checkpoint_000')
tangoCheckpointHandler = tango.handlers.TangoCheckpointHandler(iterationInterval=20, basename='tango_checkpoint')
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=50, basename='tango_history', maxIterations=maxIterations)

# specify how long GENE runs between Tango iterations.  Specified in 
geneFluxModel.set_simulation_time(500)

solver = tango.solver.Solver(L, psi, pressureIC, pressureRightBC, t_array, maxIterations, tol, compute_all_H, turbhandler)

## Set up the file handling 
parallelEnvironment = True
solver.fileHandlerExecutor.set_parallel_environment(parallelEnvironment, MPIrank)
solver.fileHandlerExecutor.add_handler(f1HistoryHandler)
solver.fileHandlerExecutor.add_handler(tangoCheckpointHandler)
solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)

# create parameters for dataSaverHandler
arrays_to_save = ['H2', 'H3', 'profile'] # see solver.py for list
databasename = 'datasaver'
solver.dataSaverHandler.initialize_datasaver(databasename, maxIterations, arrays_to_save)
solver.dataSaverHandler.set_parallel_environment(parallelEnvironment, MPIrank)

# logging ??
#while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    # solver.TakeTimestep()
    
print("Using pseudo-GENE, python-GENE interface code initialized OK!")



#if solver.reached_end == True:
#    if MPIrank==0:
#        print("The solution has been reached successfully.")
#        print("Took {} iterations".format(solver.l))
#else:
#    if MPIrank==0:
#        print("The solver failed for some reason.")
        
# other logging??