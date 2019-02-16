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
#import tango.smoother
#import tango.genecomm_unitconversion
import tango.tango_logging as tlog
import tango.utilities.util # for duration_as_hms
import tango.utilities.gene.read_chease_file as read_chease_file
import tango_chease_kinetic_run_helper as helper  # helper module needs to be kept in same directory

# constants
MAXITERS = 50
SIMTIME = 30 # GENE simulation time per iteration, in Lref/cref
CHEASEFILE = 'xyz.h5'  # 'TCV28008_t11.h5' for negative triangularity,  '28014_t12.h5' for positive triangularity
EWMA_PARAM_TURB_FLUX = 0.1

thetaParamsN = {'custom_ftheta': helper.n_custom_ftheta}
thetaParamsPi = {'custom_ftheta': helper.const_ftheta}
thetaParamsPe = {'custom_ftheta': helper.const_ftheta}

def initialize_iteration_parameters():
    maxIterations = MAXITERS
    EWMAParamTurbFlux = EWMA_PARAM_TURB_FLUX
    EWMAParamProfile = 1
    tol = 1e-11  # tol for convergence... reached when a certain error < tol
    return (maxIterations, EWMAParamTurbFlux, EWMAParamProfile, tol)


# define collisional energy exchange term
    
# ============================================================================== #
# Define the terms in the Transport Equations:  density, ion pressure, electron pressure

# density transport equation
class ComputeAllH_n(object):
    def __init__(self, Vprime, gxxAvg, minorRadius):
        self.Vprime = Vprime
        self.gxxAvg = gxxAvg  # <grad x dot grad x>, used for adhoc diffusion coefficients in geometry
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
        H2_adhoc = D_adhoc * self.Vprime * self.gxxAvg
        
        # Particle source
        H7 = self.Vprime * helper.Sn_func(x / self.minorRadius)
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
    

# ion pressure transport equation
class ComputeAllH_pi(object):
    def __init__(self, Vprime, gxxAvg, minorRadius):
        self.Vprime = Vprime
        self.gxxAvg = gxxAvg
        self.minorRadius = minorRadius
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        #pi = profiles['pi']
        pe = profiles['pe']
        
        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.2  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime * self.gxxAvg
        
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
    def __init__(self, Vprime, gxxAvg, minorRadius):
        self.Vprime = Vprime
        self.gxxAvg = gxxAvg
        self.minorRadius = minorRadius
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        #pi = profiles['pi']
        pe = profiles['pe']
        

        # Define the contributions to the H coefficients
        H1 = 3/2 * self.Vprime
        
        # add some manually input diffusivity
        D_adhoc = 0.15  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime * self.gxxAvg
        
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
    ########### NEW
    
    # Set up the turbulence (GENE) grid
    rhoLeftBndyGene = 0.55
    rhoRightBndy = 0.95
    numRadialPtsGene = 180  # from parameters file... needs to be consistent
    rhoGene = np.linspace(rhoLeftBndyGene, rhoRightBndy, numRadialPtsGene)
    
    # Set up the transport (Tango) grid
    numRadialPtsTango = 54
    drho = rhoRightBndy / (numRadialPtsTango - 0.5)  # spatial grid size
    rhoTango = np.linspace(drho / 2, rhoRightBndy, numRadialPtsTango)  # Tango inner-most point is set at delta rho/2, not exactly zero.
    
    # Get CHEASE data
    cheaseTangoData = read_chease_file.get_chease_data_on_Tango_grid(CHEASEFILE, rhoTango)
    minorRadius = cheaseTangoData.minorRadius
    Lref = cheaseTangoData.Lref
    majorRadius = Lref
    Bref = cheaseTangoData.Bref
    VprimeTango = cheaseTangoData.dVdx
    gxxAvgTango = cheaseTangoData.gxxAvg  # average <g^xx> = <grad x dot grad x>
    gradxAvgTango = cheaseTangoData.gradxAvg
    
    
    xGene = rhoGene * minorRadius
    dxGene = xGene[1] - xGene[0]
    xTango = rhoTango * minorRadius  # rho_tor, measured in meters, used as the independent coordinate in Tango
    # drTango = rTango[1] - rTango[0]
    L = xTango[-1]  # size of domain

    Tref = 1
    nref = 1
    ionMass = 2  # in proton masses
    mref = ionMass    
    
    
    # create object for interfacing tango and GENE radial grids
    # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    rExtrapZoneLeft = 0.80 * minorRadius
    rExtrapZoneRight = 0.85 * minorRadius
    polynomialDegree = 0
    gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(xTango, xGene, rExtrapZoneLeft, rExtrapZoneRight, polynomialDegree)
    
    # map <|grad x|^2> and <|grad x|> to turbulence grid
    gxxAvgTurb = gridMapper.map_profile_onto_turb_grid(gxxAvgTango)
    gradxAvgTurb = gridMapper.map_profile_onto_turb_grid(gradxAvgTango)
    
    
    #============================================================================#
    # Boundary Conditions at the outer radial boundary.
    n_BC = 0.55e19
    Ti_keV_BC = 0.25
    Te_keV_BC = 0.25
    
    # Convert to pressure boundary condition
    e = 1.60217662e-19          # electron charge
    pi_BC = n_BC * Ti_keV_BC * 1000 * e
    pe_BC = n_BC * Te_keV_BC * 1000 * e
    
    
    
    #============================================================================#
    # Initial Conditions for the density and pressure profiles
    (densityICTango, ionPressureICTango, electronPressureICTango) = helper.initial_conditions(rhoTango)
    
    # specify species masses and charges
    mass = np.array([2.0, 2.0/200])
    charge = np.array([1, -1])
    
    
    # GENE setup
    fromCheckpoint = True    # true if restarting a simulation from an already-saved checkpoint
    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_chease_kineticelectrons(
                cheaseTangoData, xTango, xGene, mass, charge, Tref, nref, fromCheckpoint)

    # iteration parameters setup
    (maxIterations, EWMAParamTurbFlux, EWMAParamProfile, tol) = initialize_iteration_parameters()

    # create flux smoother for spatial averaging of flux
    windowSizeInGyroradii = 10
    rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
    windowSizeInPoints = int( np.round(windowSizeInGyroradii * rhoref / dxGene) )
    fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)
    
    tArray = np.array([0, 1e4])  # specify the timesteps to be used.
    
    # creation of turbulence handler
    turbHandler = tango.lodestro_method.TurbulenceHandler(dxGene, xTango, geneFluxModel, VprimeTango=VprimeTango, fluxSmoother=fluxSmoother,
                                                          gxxAvgTango=gxxAvgTango, gradxAvgTango=gradxAvgTango)
    
    ## *************************** ##
    # set up for density equation
    compute_all_H_n = ComputeAllH_n(VprimeTango, gxxAvgTango, minorRadius)
    
    lm_n = tango.lodestro_method.lm(EWMAParamTurbFlux, EWMAParamProfile, thetaParamsN,
                                    gxxAvgTurb=gxxAvgTurb, gradxAvgTurb=gradxAvgTurb)
    # seed the EWMA D,c for the turbulent particle flux
    particleFluxSeed = helper.read_seed_turb_flux('particle_flux_seed_ions')
    smoothedFluxTurbGrid = fluxSmoother.smooth(particleFluxSeed)
    densityTurbGrid = gridMapper.map_profile_onto_turb_grid(densityICTango)
    #   convert flux to D,c
    (n_DTurbGrid, n_cTurbGrid, _) = lm_n.flux_to_transport_coeffs(smoothedFluxTurbGrid, densityTurbGrid, dxGene)
    lm_n.set_ewma_turb_D(n_DTurbGrid)
    lm_n.set_ewma_turb_c(n_cTurbGrid)
    # create the field
    field_n = tango.multifield.Field(
            label='n', rightBC=n_BC, profile_mminus1=densityICTango, compute_all_H=compute_all_H_n,
            lodestroMethod=lm_n, gridMapper=gridMapper)
    
    ## *************************** ##
    # set up for ion pressure equation
    compute_all_H_pi = ComputeAllH_pi(VprimeTango, gxxAvgTango, minorRadius)
    lm_pi = tango.lodestro_method.lm(EWMAParamTurbFlux, EWMAParamProfile, thetaParamsPi,
                                     gxxAvgTurb=gxxAvgTurb, gradxAvgTurb=gradxAvgTurb)
    # seed the EWMA D,c for the turbulent ion heat flux
    ionHeatFluxSeed = helper.read_seed_turb_flux('heat_flux_seed_ions')
    smoothedFluxTurbGrid = fluxSmoother.smooth(ionHeatFluxSeed)
    ionPressureTurbGrid = gridMapper.map_profile_onto_turb_grid(ionPressureICTango)
    #   convert flux to D,c
    (pi_DTurbGrid, pi_cTurbGrid, _) = lm_pi.flux_to_transport_coeffs(smoothedFluxTurbGrid, ionPressureTurbGrid, dxGene)
    lm_pi.set_ewma_turb_D(pi_DTurbGrid)
    lm_pi.set_ewma_turb_c(pi_cTurbGrid)
    # create the field, coupled to pe
    field_pi = tango.multifield.Field(
            label='pi', rightBC=pi_BC, profile_mminus1=ionPressureICTango, compute_all_H=compute_all_H_pi,
            lodestroMethod=lm_pi, gridMapper=gridMapper, coupledTo='pe')
    
    ## *************************** ##
    # set up for electron pressure equation
    compute_all_H_pe = ComputeAllH_pe(VprimeTango, gxxAvgTango, minorRadius)
    lm_pe = tango.lodestro_method.lm(EWMAParamTurbFlux, EWMAParamProfile, thetaParamsPe,
                                     gxxAvgTurb=gxxAvgTurb, gradxAvgTurb=gradxAvgTurb)
    # seed the EWMA D,c for the turbulent electron heat flux
    electronHeatFluxSeed = helper.read_seed_turb_flux('heat_flux_seed_electrons')
    smoothedFluxTurbGrid = fluxSmoother.smooth(electronHeatFluxSeed)
    electronPressureTurbGrid = gridMapper.map_profile_onto_turb_grid(electronPressureICTango)
    #   convert flux to D,c
    (pe_DTurbGrid, pe_cTurbGrid, _) = lm_pe.flux_to_transport_coeffs(smoothedFluxTurbGrid, electronPressureTurbGrid, dxGene)
    lm_pe.set_ewma_turb_D(pe_DTurbGrid)
    lm_pe.set_ewma_turb_c(pe_cTurbGrid)
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
setNumber = 0
t = tArray[1]
initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xGene, t, fields)
basename = 'tangodata'   # without restart, saves into tangodata_s0.hdf5
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
filenameTangoHistory = basename + '_s{}'.format(setNumber) + '.hdf5'

#  specify how long GENE runs between Tango iterations.  Specified in Lref/cref
geneFluxModel.set_simulation_time(SIMTIME)

# initialize the user control function, if applicable.  If using it, then it needs to be passed as a parameter when initializing solver
#densityICTango = fields[0].profile_mminus1
#user_control_func = UserControlFunc(densityICTango)

solver = tango.solver.Solver(L, xTango, tArray, maxIterations, tol, compute_all_H_all_fields, fields)


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