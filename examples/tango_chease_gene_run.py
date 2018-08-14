"""
Checklist for a Gene-CHEASE-Tango run:
GENE parameters file
    diagdir
    read_checkpoint [and copy and rename a start file to checkpoint_000]
    parallelization / number of processors
    Chease file specification: which file,

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
    reference values
    etc.

Rule of thumb for theta parameters (which determine the diffusive/convective split) in LoDestro Method:
    --for field U.  set dpdxthreshold = Geometric_Avg[U] * 4e4    (where the geometric average is over the estimated min and max values)
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
import tango.utilities.util  # for duration_as_hms
import tango.utilities.gene.read_chease_file as read_chease_file
import tango_chease_gene_run_helper as helper  # helper module needs to be kept in same directory

# constants
MAXITERS = 50
DIAGDIR = '/scratch2/scratchdirs/jbparker/genedata/prob##/'
# DIAGDIR = '/scratch2/scratchdirs/jbparker/genedata/prob##/'   # scratch on Edison
# DIAGDIR = '/global/cscratch1/sd/jbparker/genecori/prob##/'  # scratch on cori
# DIAGDIR = '/p/lscratchh/parker68/q_gene/prob##/' # scratch on quartz
SIMTIME = 50  # GENE simulation time per iteration, in Lref/cref
CHEASEFILE = 'xyz.h5'



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


class ComputeAllH(object):
    def __init__(self, Vprime, gxxAvg, minorRadius):
        self.Vprime = Vprime
        self.gxxAvg = gxxAvg  # <grad x dot grad x>, used for adhoc diffusion coefficients in geometry
        self.minorRadius = minorRadius

    def __call__(self, t, x, profiles, HCoeffsTurb):
        """Define the contributions to the H coefficients

        Inputs:
          t             time (scalar)
          x             radial coordinate in SI (array)
          pressure      pressure profile in SI (array)
          HCoeffsTurb   coefficients from turbulence
        """
        # pressure = profiles['pi']
        H1 = 1.5 * self.Vprime

        # add some manually input diffusivity
        D_adhoc = 0.15  # in SI, m^2/s
        H2_adhoc = D_adhoc * self.Vprime * self.gxxAvg

        # Heat Source
        H7 = helper.fi_hat(x / self.minorRadius)

        HCoeffs = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs


def problem_setup():
    """
    Still to be done: need to choose good profiles for
        ion temperature initial condition
        plasma density
        heat source
        tango and GENE radial grids
    """
    # Set up problem parameters

    # Set up the turbulence (GENE) grid
    rhoLeftBndyGene = 0.1
    rhoRightBndy = 0.9
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

    xGene = rhoGene * minorRadius
    dxGene = xGene[1] - xGene[0]
    xTango = rhoTango * minorRadius  # physical radius r, measured in meters, used as the independent coordinate in Tango
    # drTango = rTango[1] - rTango[0]
    L = xTango[-1]  # size of domain

    Tref = 3
    nref = 3.3
    ionMass = 2  # in proton masses
    mref = ionMass
    ionCharge = 1

    densityTango = helper.density_profile(rhoTango)

    # create object for interfacing tango and GENE radial grids
    # must be consistent with whether Tango's or Gene's radial domain extends farther radially outward
    xExtrapZoneLeft = 0.75 * minorRadius
    xExtrapZoneRight = 0.80 * minorRadius
    polynomialDegree = 1
    gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(xTango, xGene, xExtrapZoneLeft, xExtrapZoneRight, polynomialDegree)

    # specify a boundary condition for pressure at the outward radial boundary
    temperatureRightBCInkeV = 1
    e = 1.60217662e-19          # electron charge
    temperatureRightBC = temperatureRightBCInkeV * 1000 * e  # temperature in SI
    pressureRightBC = temperatureRightBC * densityTango[-1]

    # specify a temperature and pressure initial condition
    temperatureICTango = helper.temperature_initial_condition(rhoTango)  # in SI
    pressureICTango = temperatureICTango * densityTango

    # GENE setup
    fromCheckpoint = True    # true if restarting a simulation from an already-saved checkpoint
    geneFluxModel = tango.gene_startup.setup_gene_run_singleion_chease_adiabaticelectrons(
        cheaseTangoData, xTango, xGene, ionMass, ionCharge, densityTango,
        Tref, nref, gridMapper, fromCheckpoint)

    # other transport physics / physicsToH object creation
    # profilesAll = tango.physics.initialize_profile_defaults(ionMass, densityProfile, psiTango, minorRadius, majorRadius, B0, Vprime, gradPsiSq)
    # HcontribTransportPhysics = tango.physics_to_H.Hcontrib_TransportPhysics(profilesAll)

    # iteration parameters setup
    (maxIterations, lmParams, tol) = initialize_iteration_parameters()

    # create flux smoother for spatial averaging of flux
    windowSizeInGyroradii = 10
    rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
    windowSizeInPoints = int(np.round(windowSizeInGyroradii * rhoref / dxGene))
    fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)

    tArray = np.array([0, 1e4])  # specify the timesteps to be used.

    # creation of turbulence handler
    turbHandler = tango.lodestro_method.TurbulenceHandler(dxGene, xTango, geneFluxModel, VprimeTango=VprimeTango, fluxSmoother=fluxSmoother)

    compute_all_H_pressure = ComputeAllH(VprimeTango, gxxAvgTango, minorRadius)
    lodestroMethod = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])

    # seed the EWMA for the turbulent heat flux
    heatFluxSeed = helper.read_seed_turb_flux('heat_flux_seed')
    lodestroMethod.set_ewma_turb_flux(heatFluxSeed)

    field0 = tango.multifield.Field(
        label='pi', rightBC=pressureRightBC, profile_mminus1=pressureICTango, compute_all_H=compute_all_H_pressure,
        lodestroMethod=lodestroMethod, gridMapper=gridMapper)
    fields = [field0]
    tango.multifield.check_fields_initialize(fields)
    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)

    return (L, xTango, xGene, pressureRightBC, pressureICTango, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields)


# ************************************************** #
####              START OF MAIN PROGRAM           ####
# ************************************************** #
MPIrank = gene_tango.init_mpi()
tlog.setup(True, MPIrank, tlog.DEBUG)
(L, xTango, xGene, pressureRightBC, pressureICTango, maxIterations, tol, geneFluxModel, turbHandler, compute_all_H_all_fields, tArray, fields) = problem_setup()


# set up FileHandlers
###  GENE output to save periodically.  For list of available, see handlers.py
diagdir = DIAGDIR

### save the GENE checkpoint file --- distribution function f1
# WARNING: on Quartz, this won't work, so don't use it
# f1HistoryHandler = tango.handlers.SaveGeneOutputHandler('checkpoint_000', iterationInterval=10, diagdir=diagdir)

### save the GENE data analysis output
geneFilesToSave = [name + '_000' for name in ['field', 'mom_ions', 'nrg', 'profile_ions', 'profiles_ions', 'srcmom_ions', 'vsp']]
geneOutputHandler = tango.handlers.SaveGeneOutputHandler(*geneFilesToSave, iterationInterval=1, diagdir=diagdir)

### set up Tango History handler
setNumber = 0
t = tArray[1]
initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xGene, t, fields)
basename = 'tangodata'   # without restart, saves into tangodata_s0.hdf5
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
filenameTangoHistory = basename + '_s{}'.format(setNumber) + '.hdf5'


#  specify how long GENE runs between Tango iterations.  Specified in Lref/cref
geneFluxModel.set_simulation_time(SIMTIME)

solver = tango.solver.Solver(L, xTango, tArray, maxIterations, tol, compute_all_H_all_fields, fields)

# Add the file handlers
solver.fileHandlerExecutor.set_parallel_environment(parallel=True, MPIrank=MPIrank)
#solver.fileHandlerExecutor.add_handler(f1HistoryHandler)
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

pi = solver.profiles['pi']  # finished solution
tlog.info("The profile at the end is:")

tlog.info("{}".format(pi))

endTime = time.time()
durationHMS = tango.utilities.util.duration_as_hms(endTime - startTime)
tlog.info("Total wall-time spent for Tango run: {}".format(durationHMS))

gene_tango.finalize_mpi()
