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

def initialize_shestakov_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    nL = 1e-2           # right boundary condition
    n_initialcondition = 1 - 0.5*x
    return (L, N, dx, x, nL, n_initialcondition)

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
maxIterations, lmparams, tol = initialize_parameters()
FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
turbhandler = tng.TurbulenceHandler(dx, x, lmparams, FluxModel)

t_array = np.array([0, 1e4])  # specify the timesteps to be used.

compute_all_H = ComputeAllH(turbhandler)
solver = tng.solver.solver(L, x, n, nL, t_array, maxIterations, tol, compute_all_H, turbhandler)

# set up data logger
arrays_to_save = ['H2', 'H3', 'profile']  # for list of possible arrays, see solver._pkgdata()
data_basename = 'shestakov_solution_data'
solver.dataSaverHandler.initialize_datasaver(data_basename, maxIterations, arrays_to_save)
logging.info("Preparing DataSaver to save files with prefix {}.".format(data_basename))

logging.info("Initialization complete.")

logging.info("Beginning time integration...")
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()


    
    
n = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
nss = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL)

fig = plt.figure()
plt.plot(x, n, 'b-')
plt.plot(x, nss, 'r-')

solution_residual = (n - nss) / np.max(np.abs(nss))
solution_rms_error = np.sqrt( 1/len(n) * np.sum(solution_residual**2))

if solver.reachedEnd == True:
    print("The solution has been reached successfully.")
    print('Error compared to analytic steady state solution is %f' % (solution_rms_error))
else:
    print("The solver failed for some reason.  See log file {}".format(logfile))
    print('Error at end compared to analytic steady state solution is %f' % (solution_rms_error))


#plt.figure()
#plt.semilogy(errhistory)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
filename = data_basename + "1"
Timestep = tng.analysis.TimestepData(filename)
lastiter = Timestep.get_last_iteration()
lastiter.plot_profile_and_starting_profile(savename='solution.png')