"""
Checklist for a Gene-Tango run:
GENE parameters file
    diagdir
    read_checkpoint [and copy and rename a start file to checkpoint_000]
    parallelization / number of processors [perform auto-tuning ahead of time; do not include any -1 in a Tango run]

submit.cmd file
    time limit
    number of processors

Tango python file
    maximum number of iterations
    EWMA parameters
    diagdir
    GENE simulation time per iteration
    density profile
    user_control_func
    etc.
    
Rule of thumb for theta parameters (which determine the diffusive/convective split) in LoDestro Method:
    --for field U.  set dpdxthreshold = Geometric_Avg[U] * 80    (where the geometric average is over the estimated min and max values)
    --Dmin = 1e-5, Dmax = 1e3?  
    --these aren't heavily tested and may well need to be modified.
    --probably a logarithmic derivative should be used for thresholding, but has not been implemented yet
"""

from __future__ import division, absolute_import
import numpy as np
import time

import gene_tango # This must come before other tango imports, else it crashes on quartz!

# Tango imports... should simplify these so I'm not importing them all separately!
import tango
import tango.gene_startup
import tango.smoother
import tango.genecomm_unitconversion
import tango.tango_logging as tlog
import tango.utilities.util # for duration_as_hms
import tango_kinetic_gene_run_helper as helper  # helper module needs to be kept in same directory

# constants
MAXITERS = 50
ITERATIONS_PER_SET = 10
# DIAGDIR = '/scratch2/scratchdirs/jbparker/genedata/prob##/'   # scratch on Edison
# DIAGDIR = '/global/cscratch1/sd/jbparker/genecori/prob##/'  # scratch on cori
DIAGDIR = '/p/lscratchh/parker68/q_gene/prob##/' # scratch on quartz
SIMTIME = 50 # GENE simulation time per iteration, in Lref/cref

thetaParamsPi = {'Dmin': 1e-5,  'Dmax': 1e3,  'dpdxThreshold': 4e5}
thetaParamsPe = {'Dmin': 1e-5,  'Dmax': 1e3,  'dpdxThreshold': 4e5}
thetaParamsN = {'Dmin': 1e-5,  'Dmax': 1e3,  'dpdxThreshold': 2.4e20}

def initialize_iteration_parameters():
    maxIterations = MAXITERS
    thetaParams = {'Dmin': 1e-5,
                   'Dmax': 1e3,
                   'dpdxThreshold': 400000}
    EWMAParamTurbFlux = 0.3
    EWMAParamProfile = 0.3
    lmParams = {'EWMAParamTurbFlux': EWMAParamTurbFlux,
            'EWMAParamProfile': EWMAParamProfile,
            'thetaParams': thetaParams}
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)


# define collisional energy exchange term
    
# ============================================================================== #
# Define the terms in the Transport Equations:  density, ion pressure, elcetron pressure

# density transport equation
class ComputeAllH_n(object):
    def __init__(self, Vprime, minorRadius):
        self.Vprime = Vprime
        self.minorRadius = minorRadius
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['n']
        #pi = profiles['pi']
        #pe = profiles['pe']
        #n = profiles['field0']
        # Define the contributions to the H coefficients
        H1 = 1.0 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.05  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Particle source
        H7 = helper.fn_hat(x / self.minorRadius)
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
    

# ion pressure transport equation
class ComputeAllH_pi(object):
    def __init__(self, Vprime, minorRadius):
        self.Vprime = Vprime
        self.minorRadius = minorRadius
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        #pi = profiles['pi']
        pe = profiles['pe']
        TeInEV = helper.convert_Te_to_eV(n, pe)
        nu = helper.nu_E(n, TeInEV)
        
        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.1  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Ion heat source
        H7 = helper.fi_hat(x / self.minorRadius)

        # collisional energy exchage
        H6 = -nu * self.Vprime
        H8 = nu * self.Vprime
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H6=H6, H7=H7, H8=H8)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        

# electron pressure transport equation
class ComputeAllH_pe(object):
    def __init__(self, Vprime, minorRadius):
        self.Vprime = Vprime
        self.minorRadius = minorRadius
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        #pi = profiles['pi']
        pe = profiles['pe']
        TeInEV = helper.convert_Te_to_eV(n, pe)
        nu = helper.nu_E(n, TeInEV)
        
        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.1  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Electron heat source
        H7 = helper.fe_hat(x / self.minorRadius)
        
        H6 = -nu * self.Vprime
        H8 = nu * self.Vprime
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H6=H6, H7=H7, H8=H8)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
def problem_setup():
    """
    Take care of a lot of initialization and boilerplate.
    """
    # Set up problem parameters
    minorRadius = 0.594     # minor radius a, in meters
    majorRadius = 1.65      # major radius R0, in meters
    
    # Set up the turbulence (GENE) grid
    rhoLeftBndyGene = 0.1
    rhoRightBndy = 0.9
    numRadialPtsGene = 360  # from parameters file... needs to be consistent
    rhoGene = np.linspace(rhoLeftBndyGene, rhoRightBndy, numRadialPtsGene)
    rGene = rhoGene * minorRadius
    drGene = rGene[1] - rGene[0]
    
    # Set up the transport (Tango) grid
    numRadialPtsTango = 53
    drho = rhoRightBndy / (numRadialPtsTango - 0.5)  # spatial grid size
    rhoTango = np.linspace(drho/2, rhoRightBndy, numRadialPtsTango)  # Tango inner-most point is set at delta rho/2, not exactly zero.
    rTango = rhoTango * minorRadius # physical radius r, measured in meters, used as the independent coordinate in Tango
    # drTango = rTango[1] - rTango[0]
    L = rTango[-1]  # size of domain
    
    #VprimeGene = 4 * np.pi**2 * majorRadius * rGene
    VprimeTango = 4 * np.pi**2 * majorRadius * rTango
    #gradPsiSqTango = np.ones_like(rTango) # |grad r|^2 = 1
    
    Bref = 1.14
    B0 = Bref
    Lref = 1.65
    Tref = 1
    nref = 3.3
    mref = 2    # in proton masses
    
    # create object for interfacing tango and GENE radial grids
    # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    rExtrapZoneLeft = 0.75 * minorRadius
    rExtrapZoneRight = 0.80 * minorRadius
    polynomialDegree = 1
    gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(rTango, rGene, rExtrapZoneLeft, rExtrapZoneRight, polynomialDegree)
    
    #============================================================================#
    # Boundary Conditions at the outer radial boundary
    #  density BC
    densityRightBC = 2e19
    
    #  pressure BC for ion and electrons.  Set temperature BC is 0.5 keV for both.
    temperatureRightBCInkeV = 0.44
    e = 1.60217662e-19          # electron charge
    temperatureRightBC = temperatureRightBCInkeV * 1000 * e  # temperature in SI
    ionPressureRightBC = temperatureRightBC * densityRightBC
    electronPressureRightBC = ionPressureRightBC
    
    #============================================================================#
    # Initial Conditions for the density and pressure profiles
    densityICTango = helper.density_initial_condition(rhoTango)
    ionTemperatureICTango = helper.ion_temperature_initial_condition(rhoTango)
    electronTemperatureICTango = helper.electron_temperature_initial_condition(rhoTango)
    
    ionPressureICTango = densityICTango * ionTemperatureICTango
    electronPressureICTango = densityICTango * electronTemperatureICTango
    
    # specify species masses and charges
    mass = np.array([2.0, 2.0/100])
    charge = np.array([1, -1])
    
    # specify safety factor
    safetyFactorGeneGrid = 0.868 + 2.2 * rhoGene**2
    
    # GENE setup
    fromCheckpoint = True    # true if restarting a simulation from an already-saved checkpoint
    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_kineticelectrons(
                rTango, rGene, minorRadius, majorRadius, B0, mass, charge, safetyFactorGeneGrid,
                Bref, Lref, Tref, nref, fromCheckpoint)

    # iteration parameters setup
    #### NEW: need to do separate lmparams for each field??
    (maxIterations, lmParams, tol) = initialize_iteration_parameters()    
    
    # create flux smoother for spatial averaging of flux
    windowSizeInGyroradii = 10
    rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
    windowSizeInPoints = int( np.round(windowSizeInGyroradii * rhoref / drGene) )
    fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)
    
    tArray = np.array([0, 1e4])  # specify the timesteps to be used.
    
    # creation of turbulence handler
    turbHandler = tango.lodestro_method.TurbulenceHandler(drGene, rTango, geneFluxModel, VprimeTango=VprimeTango, fluxSmoother=fluxSmoother)
    
    ## *************************** ##
    # set up for density equation
    compute_all_H_n = ComputeAllH_n(VprimeTango, minorRadius)
    lm_n = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsN)
    # seed the EWMA for the turbulent particle flux
    particleFluxSeed = helper.read_seed_turb_flux('particle_flux_seed_ions')
    lm_n.set_ewma_turb_flux(particleFluxSeed)
    # create the field
    field_n = tango.multifield.Field(
            label='n', rightBC=densityRightBC, profile_mminus1=densityICTango, compute_all_H=compute_all_H_n,
            lodestroMethod=lm_n, gridMapper=gridMapper)
    
    ## *************************** ##
    # set up for ion pressure equation
    compute_all_H_pi = ComputeAllH_pi(VprimeTango, minorRadius)
    lm_pi = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsPi)
    # seed the EWMA for the turbulent ion heat flux
    ionHeatFluxSeed = helper.read_seed_turb_flux('heat_flux_seed_ions')
    lm_pi.set_ewma_turb_flux(ionHeatFluxSeed)
    # create the field, coupled to pe
    field_pi = tango.multifield.Field(
            label='pi', rightBC=ionPressureRightBC, profile_mminus1=ionPressureICTango, compute_all_H=compute_all_H_pi,
            lodestroMethod=lm_pi, gridMapper=gridMapper, coupledTo='pe')
    
    ## *************************** ##
    # set up for electron pressure equation
    compute_all_H_pe = ComputeAllH_pe(VprimeTango, minorRadius)
    lm_pe = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsPe)
    # seed the EWMA for the turbulent electron heat flux
    electronHeatFluxSeed = helper.read_seed_turb_flux('heat_flux_seed_electrons')
    lm_pe.set_ewma_turb_flux(electronHeatFluxSeed)
    # create the field, coupled to pe
    field_pe = tango.multifield.Field(
            label='pe', rightBC=electronPressureRightBC, profile_mminus1=electronPressureICTango, compute_all_H=compute_all_H_pe,
            lodestroMethod=lm_pe, gridMapper=gridMapper, coupledTo='pi')

    # combine fields and do checking
    fields = [field_n, field_pi, field_pe]
    tango.multifield.check_fields_initialize(fields)
    # initialize the function that computes transport equation for all fields
    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)
    
    return (L, rTango, rGene, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields)

    
class UserControlFunc(object):
    def __init__(self, densityICTango):
        self.densityICTango = densityICTango
    def __call__(self, solver):
        """
        User Control Function for the solver.
        
        Here, reset the density profile to its initial condition.  This has the effect of preventing
        the density from evolving, so Tango should act as if density is not being evolved at all through
        a transport equation.
        
        Inputs:
          solver            tango Solver (object)
        """
        iterationNumber = solver.iterationNumber
        if iterationNumber < 50:
            solver.profiles['n'] = self.densityICTango
    
# ************************************************** #
####              START OF MAIN PROGRAM           ####
# ************************************************** #
MPIrank = gene_tango.init_mpi()
tlog.setup(True, MPIrank, tlog.DEBUG)
(L, rTango, rGene, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields) = problem_setup()

iterationsPerSet = ITERATIONS_PER_SET


# set up FileHandlers
#  GENE output to save periodically.  For list of available, see handlers.py
diagdir = DIAGDIR
f1HistoryHandler = tango.handlers.SaveGeneOutputHandler('checkpoint_000', iterationInterval=10, diagdir=diagdir)

### save the GENE data analysis output
geneFilesToSave = [name + '_000' for name in ['field', 'mom_ions', 'nrg', 'profile_ions', 'profiles_ions', 'srcmom_ions', 'vsp']]
geneOutputHandler = tango.handlers.SaveGeneOutputHandler(*geneFilesToSave, iterationInterval=1, diagdir=diagdir)

### set up Tango History handler
basename = 'tangodata'
restartfile = tango.restart.check_if_should_restart(basename)   # returns path of restartfile as string; returns None if no restartfile found

if restartfile: # Restart file exists
    (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turbFluxesEWMA) = tango.restart.read_metadata_from_previousfile(restartfile)
    tango.restart.set_ewma_iterates(fields, old_profilesEWMA, old_turbFluxesEWMA)
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, rTango, rGene, t, fields)
else: # No restartfile exists.  Set up for first Tango run
    (setNumber, startIterationNumber, t, timestepNumber) = (0, 0, tArray[1], 1)
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, rTango, rGene, t, fields)

# instantiate the handler
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
filenameTangoHistory = basename + '_s{}'.format(setNumber) + '.hdf5'  # filename that data is saved to.

#  specify how long GENE runs between Tango iterations.  Specified in Lref/cref
geneFluxModel.set_simulation_time(SIMTIME)

# initialize the user control function, if applicable.  If using it, then it needs to be passed as a parameter when intializaing solver
# densityICTango = fields[0].profile_mminus1
# user_control_func = UserControlFunc(densityICTango)

# set up solver.  See solver.py for optional arguments
if restartfile:
    solver = tango.solver.Solver(L, rTango, tArray, maxIterations, tol, compute_all_H_all_fields, fields,
                                 startIterationNumber=startIterationNumber, profiles=old_profiles,
                                 maxIterationsPerSet=iterationsPerSet)
else:
    solver = tango.solver.Solver(L, rTango, tArray, maxIterations, tol, compute_all_H_all_fields, fields)


# Add the file handlers
solver.fileHandlerExecutor.set_parallel_environment(parallel=True, MPIrank=MPIrank)
solver.fileHandlerExecutor.add_handler(f1HistoryHandler)
solver.fileHandlerExecutor.add_handler(geneOutputHandler)
solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
tlog.info("Tango history handler setup, saving to {}.".format(filenameTangoHistory))

tlog.info("Entering main time loop...")

startTime = time.time()
while solver.ok:
     # Implicit time advance: iterate to solve the nonlinear equation!
     solver.take_timestep()

if solver.reachedEnd == True:
    tlog.info("The solution has been reached successfully.")
    tlog.info("Took {} iterations".format(solver.l))
else:
    tlog.info("The solver failed for some reason.")
    tlog.info("The residual at the end is {}".format(solver.errHistoryFinal[-1]))
    
pi = solver.profiles['pi'] # finished solution
tlog.info("The ion pressure profile at the end is:")

tlog.info("{}".format(pi))

endTime = time.time()
durationHMS = tango.utilities.util.duration_as_hms(endTime - startTime)
tlog.info("Total wall-time spent for Tango run: {}".format(durationHMS))

if MPIrank == 0:
    tango.restart.combine_savefiles(basename)
tlog.info('Created a combined tango save file.')

gene_tango.finalize_mpi()


