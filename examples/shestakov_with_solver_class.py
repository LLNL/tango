"""Example for how to use tango to solve a turbulence and transport problem.

Using a solver class, saving to a file, and using the analysis package to load the data and save a plot

Here, the "turbulent flux" is specified analytically, using the example in the Shestakov et al. (2003) paper.
This example is a nonlinear diffusion equation with specified diffusion coefficient and source.  There is a
closed form answer for the steady state solution which can be compared with the numerically found solution.
"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging


from tango.extras import shestakov_nonlinear_diffusion
import tango as tng
import tango.analysis

def initialize_shestakov_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    nL = 1e-2           # right boundary condition
    nInitialCondition = 1 - 0.5*x
    return (L, N, dx, x, nL, nInitialCondition)

def initialize_parameters():
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
    def __init__(self, turbhandler):
        self.turbhandler = turbhandler
    def __call__(self, t, x, n):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        (H2, H3, extradata) = self.turbhandler.Hcontrib_turbulent_flux(n)
        H4 = None
        H6 = None
        return (H1, H2, H3, H4, H6, H7, extradata)
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
logfile = 'example_class.log'
logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
#logging.basicConfig(level=logging.INFO)


logging.info("Initializing...")
L, N, dx, x, nL, n = initialize_shestakov_problem()
maxIterations, lmParams, tol = initialize_parameters()
fluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
turbHandler = tng.lodestro_method.TurbulenceHandler(dx, x, lmParams, fluxModel)

tArray = np.array([0, 1e4])  # specify the timesteps to be used.

compute_all_H = ComputeAllH(turbHandler)
solver = tng.solver.Solver(L, x, n, nL, tArray, maxIterations, tol, compute_all_H, turbHandler)

# set up data logger
arraysToSave = ['H2', 'H3', 'profile']  # for list of possible arrays, see solver._pkgdata()
dataBasename = 'shestakov_solution_data'
solver.dataSaverHandler.initialize_datasaver(dataBasename, maxIterations, arraysToSave)
logging.info("Preparing DataSaver to save files with prefix {}.".format(dataBasename))

logging.info("Initialization complete.")

logging.info("Beginning time integration...")
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()


    
    
n = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
nss = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL)

fig = plt.figure()
line1, = plt.plot(x, n, 'b-', label='numerical solution')
line2, = plt.plot(x, nss, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(handles=[line1, line2])

solutionResidual = (n - nss) / np.max(np.abs(nss))
solutionRmsError = np.sqrt( 1/len(n) * np.sum(solutionResidual**2))

if solver.reachedEnd == True:
    print("The solution has been reached successfully.")
    print('Error compared to analytic steady state solution is %f' % (solutionRmsError))
else:
    print("The solver failed for some reason.  See log file {}".format(logfile))
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))


#plt.figure()
#plt.semilogy(errhistory)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
filename = dataBasename + "1"
Timestep = tango.analysis.TimestepData(filename)
lastiter = Timestep.get_last_iteration()
lastiter.plot_profile_and_starting_profile(savename='solution.png')