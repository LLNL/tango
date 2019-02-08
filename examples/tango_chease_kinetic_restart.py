"""
**!!!!!!!!!!!!!!!!!!!! NOT READY YET.  NEEDS TO BE EDITED.  !!!!!!!!!!!!!!!!!!**
Checklist for a Gene-Tango run:
GENE parameters file
    diagdir
    read_checkpoint [and copy and rename a start file to checkpoint_000]
    parallelization / number of processors [perform auto-tuning ahead of time; do not include any -1 in a Tango run]

submit.cmd file
    time limit
    number of processors

Tango python file
    pseudoGene (toggle)
    maximum number of iterations
    EWMA parameters
    diagdir
    GENE simulation time per iteration
    density profile
    user_control_func
    species masses (does it come from Python, rather than GENE)
    etc.
    
Rule of thumb for theta parameters (which determine the diffusive/convective split) in LoDestro Method:
    --for field U.  set dpdxthreshold = Geometric_Avg[U] * 80    (where the geometric average is over the estimated min and max values)
    --Dmin = 1e-5, Dmax = 1e3?  
    --these aren't heavily tested and may well need to be modified.
    --probably a logarithmic derivative should be used for thresholding, but has not been implemented yet
"""

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
import tango_kinetic_gene_run2_helper as helper  # helper module needs to be kept in same directory

# constants
MAXITERS = 100
#DIAGDIR = '/scratch2/scratchdirs/jbparker/genedata/kin3/prob3_cont2/'   # scratch on Edison
# DIAGDIR = '/global/cscratch1/sd/jbparker/genecori/prob##/'  # scratch on cori
# DIAGDIR = '/p/lscratchh/parker68/q_gene/prob##/' # scratch on quartz
SIMTIME = 50 # GENE simulation time per iteration, in Lref/cref
ITERATIONS_PER_SET = 50

thetaParamsN = {'custom_ftheta': helper.n_custom_ftheta}
thetaParamsPi = {'custom_ftheta': helper.const_ftheta}
thetaParamsPe = {'custom_ftheta': helper.const_ftheta}

def initialize_iteration_parameters():
    maxIterations = MAXITERS
    thetaParams = {'Dmin': 1e-2,
                   'Dmax': 1e3,
                   'dpdxThreshold': 400000}
    EWMAParamTurbFlux = 0.1
    EWMAParamProfile = 1
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
        D_adhoc = 0.10  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Particle source
        H7 = self.Vprime * helper.Sn_func(x / self.minorRadius)
        
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
        
        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.2  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Ion heat source
        H7 = self.Vprime * helper.Si_func(x / self.minorRadius)

        # collisional energy exchage
        Te = pe / n
        nuE = helper.calc_nuE(n, Te)
        H6 = -nuE * self.Vprime
        H8 = nuE * self.Vprime
        
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
        

        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.15  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime
        
        # Electron heat source
        H7 = self.Vprime * helper.Se_func(x / self.minorRadius)
        
        # collisional energy exchage
        Te = pe / n
        nuE = helper.calc_nuE(n, Te)
        H6 = -nuE * self.Vprime
        H8 = nuE * self.Vprime

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
    rhoLeftBndyGene = 0.15
    rhoRightBndy = 0.85
    numRadialPtsGene = 360  # from parameters file... needs to be consistent
    rhoGene = np.linspace(rhoLeftBndyGene, rhoRightBndy, numRadialPtsGene)
    xGene = rhoGene * minorRadius
    dxGene = xGene[1] - xGene[0]
    
    # Set up the transport (Tango) grid
    numRadialPtsTango = 53
    drho = rhoRightBndy / (numRadialPtsTango - 0.5)  # spatial grid size
    rhoTango = np.linspace(drho/2, rhoRightBndy, numRadialPtsTango)  # Tango inner-most point is set at delta rho/2, not exactly zero.
    xTango = rhoTango * minorRadius # physical radius r, measured in meters, used as the independent coordinate in Tango
    # drTango = rTango[1] - rTango[0]
    L = xTango[-1]  # size of domain
    
    VprimeTango = 4 * np.pi**2 * majorRadius * xTango
    #gradPsiSqTango = np.ones_like(rTango) # |grad r|^2 = 1
    
    Bref = 2.5
    B0 = Bref
    Lref = 1.65
    Tref = 3.5
    nref = 5
    mref = 2    # in proton masses
    
    # create object for interfacing tango and GENE radial grids
    # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    rExtrapZoneLeft = 0.75 * minorRadius
    rExtrapZoneRight = 0.80 * minorRadius
    polynomialDegree = 0
    gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(xTango, xGene, rExtrapZoneLeft, rExtrapZoneRight, polynomialDegree)
    
    #============================================================================#
    # Boundary Conditions at the outer radial boundary.
    n_BC = 5e19
    Ti_keV_BC = 1.08
    Te_keV_BC = 1.08
    
    # Convert to pressure boundary condition
    e = 1.60217662e-19          # electron charge
    pi_BC = n_BC * Ti_keV_BC * 1000 * e
    pe_BC = n_BC * Te_keV_BC * 1000 * e
    
    #============================================================================#
    # Initial Conditions for the density and pressure profiles
    (densityICTango, ionPressureICTango, electronPressureICTango) = helper.initial_conditions(rhoTango)
    
    # specify species masses and charges
    mass = np.array([2.0, 2.0/400])
    charge = np.array([1, -1])
    
    # specify safety factor
    safetyFactorGeneGrid = 0.868 + 2.2 * rhoGene**2
    
    # GENE setup
    fromCheckpoint = True    # true if restarting a simulation from an already-saved checkpoint
    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_kineticelectrons(
                xTango, xGene, minorRadius, majorRadius, B0, mass, charge, safetyFactorGeneGrid,
                Bref, Lref, Tref, nref, fromCheckpoint)

    # iteration parameters setup
    #### NEW: need to do separate lmparams for each field??
    (maxIterations, lmParams, tol) = initialize_iteration_parameters()    
    
    # create flux smoother for spatial averaging of flux
    windowSizeInGyroradii = 10
    rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
    windowSizeInPoints = int( np.round(windowSizeInGyroradii * rhoref / dxGene) )
    fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)
    
    tArray = np.array([0, 1e4])  # specify the timesteps to be used.
    
    ### NEW
    
    # creation of turbulence handler
    turbHandler = tango.lodestro_method.TurbulenceHandler(dxGene, xTango, geneFluxModel, VprimeTango=VprimeTango, fluxSmoother=fluxSmoother)
    
    ## *************************** ##
    # no need to use seed EWMA --- this is taken from the restart file
    # set up for density equation
    compute_all_H_n = ComputeAllH_n(VprimeTango, minorRadius)
    
    lm_n = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsN)
    # create the field
    field_n = tango.multifield.Field(
            label='n', rightBC=n_BC, profile_mminus1=densityICTango, compute_all_H=compute_all_H_n,
            lodestroMethod=lm_n, gridMapper=gridMapper)
    
    ## *************************** ##
    # set up for ion pressure equation
    compute_all_H_pi = ComputeAllH_pi(VprimeTango, minorRadius)
    lm_pi = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsPi)
    # create the field, coupled to pe
    field_pi = tango.multifield.Field(
            label='pi', rightBC=pi_BC, profile_mminus1=ionPressureICTango, compute_all_H=compute_all_H_pi,
            lodestroMethod=lm_pi, gridMapper=gridMapper, coupledTo='pe')
    
    ## *************************** ##
    # set up for electron pressure equation
    compute_all_H_pe = ComputeAllH_pe(VprimeTango, minorRadius)
    lm_pe = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], thetaParamsPe)
    # create the field, coupled to pe
    field_pe = tango.multifield.Field(
            label='pe', rightBC=pe_BC, profile_mminus1=electronPressureICTango, compute_all_H=compute_all_H_pe,
            lodestroMethod=lm_pe, gridMapper=gridMapper, coupledTo='pi')

    # combine fields and do checking
    fields = [field_n, field_pi, field_pe]
    tango.multifield.check_fields_initialize(fields)
    # initialize the function that computes transport equation for all fields
    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)
    
    return (L, xTango, xGene, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields)

    
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
(L, xTango, xGene, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields) = problem_setup()

# set up FileHandlers
#  GENE output to save periodically.  For list of available, see handlers.py
#diagdir = DIAGDIR
#f1HistoryHandler = tango.handlers.SaveGeneOutputHandler('checkpoint_000', iterationInterval=5, diagdir=diagdir)

### save the GENE data analysis output
#geneFilesToSave = [name + '_000' for name in ['field', 'mom_ions', 'nrg', 'profile_ions', 'profiles_ions', 'srcmom_ions', 'vsp']]
#geneOutputHandler = tango.handlers.SaveGeneOutputHandler(*geneFilesToSave, iterationInterval=1, diagdir=diagdir)

### set up Tango History handler
basename = 'tangodata'
restartfile = tango.restart.check_if_should_restart(basename)   # returns path of restartfile as string; returns None if no restartfile found

if restartfile: # Restart file exists
    (setNumber, startIterationNumber, t, timestepNumber, old_profiles, old_profilesEWMA, old_turb_D_EWMA, old_turb_c_EWMA) = tango.restart.read_metadata_from_previousfile(restartfile)
    # if the density was artificially controlled in the last run, then the density saved to Tango will be incorrected.
    # Fix that here if necessary.  Otherwise keep line commented.
    # old_profiles['n'] = densityICTango
    
    tango.restart.set_ewma_iterates(fields, old_profilesEWMA, old_turb_D_EWMA, old_turb_c_EWMA)
    initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xGene, t, fields)
else: # No restartfile exists.  Set up for first Tango run
    tlog.info('Error.  Should not be here.  Stopping')
    sys.exit(1)

tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
filenameTangoHistory = basename + '_s{}'.format(setNumber) + '.hdf5'

#  specify how long GENE runs between Tango iterations.  Specified in Lref/cref
geneFluxModel.set_simulation_time(SIMTIME)

# initialize the user control function, if applicable.  If using it, then it needs to be passed as a parameter when intializaing solver
#densityICTango = fields[0].profile_mminus1
#user_control_func = UserControlFunc(densityICTango)

if restartfile:
    solver = tango.solver.Solver(L, xTango, tArray, maxIterations, tol, compute_all_H_all_fields, fields,
                                 startIterationNumber=startIterationNumber, profiles=old_profiles,
                                 maxIterationsPerSet=ITERATIONS_PER_SET)


# Add the file handlers
solver.fileHandlerExecutor.set_parallel_environment(parallel=True, MPIrank=MPIrank)
#solver.fileHandlerExecutor.add_handler(f1HistoryHandler)
#solver.fileHandlerExecutor.add_handler(geneOutputHandler)
solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
tlog.info("Tango history handler setup, saving to {}.".format(filenameTangoHistory))

tlog.info("Entering main time loop...")

startTime = time.time()
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()

tlog.info("Tango iterations have completed.")
tlog.info("The residual at the end is {}".format(solver.errHistoryFinal[-1]))
    
e = 1.60217662e-19
pi = solver.profiles['pi'] # finished 
Ti0 = pi[0] / solver.profiles['n'][0] / (1000 * e)
tlog.info(f'The ion temperature at r=0 is:  {Ti0} keV.')

endTime = time.time()
durationHMS = tango.utilities.util.duration_as_hms(endTime - startTime)
tlog.info("Total wall-time spent for Tango run: {}".format(durationHMS))

gene_tango.finalize_mpi()