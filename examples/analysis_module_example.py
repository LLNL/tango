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
    EWMAParamTurbFlux = 0.3
    EWMAParamProfile = 0.3
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

L, N, dx, x, nL, n = initialize_shestakov_problem()
maxIterations, lmParams, tol = initialize_parameters()
fluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
turbHandler = tng.lodestro_method.TurbulenceHandler(dx, x, lmParams, fluxModel)

tArray = np.array([0, 1e4])  # specify the timesteps to be used.

compute_all_H = ComputeAllH(turbHandler)
solver = tng.solver.Solver(L, x, n, nL, tArray, maxIterations, tol, compute_all_H, turbHandler)

# set up data logger
arraysToSave = ['H2', 'H3', 'profile', 
                'D', 'c', 
                'profileEWMATurbGrid',
                'fluxTurbGrid', 'fluxEWMATurbGrid',
                'DHatTurbGrid', 'cHatTurbGrid', 'thetaTurbGrid']  # for list of possible arrays, see solver._pkgdata()
dataBasename = 'shestakov_solution_data'
solver.dataSaverHandler.initialize_datasaver(dataBasename, maxIterations, arraysToSave)

while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()

# Postprocessing and Analysis
plt.figure()
residualHistory = solver.errHistoryFinal
plt.semilogy(residualHistory)
plt.xlabel('iteration number')
plt.ylabel('Residual')

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
    print("The solver failed for some reason.")
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))

    
#==============================================================================
#  Use the Analysis Module and data on disk
#==============================================================================
filename = dataBasename + "1"  # load the data from timestep 1
Timestep = tango.analysis.TimestepData(filename)
# some things that are possible:
    
# naming convention isn't great for what 'timestep' vs. 'solution' means
#   Roughly, 'solution' means something that changes every iteration and is an array, not a single number
print(Timestep.available_timestep_fields())
# prints out ['iterationNumber', 'm', 'EWMAParamProfile', 'errhistory', 't', 
#             'EWMAParamTurbFlux', 'x', 'profile_m', 'profile_mminus1']
#  access these as follows:
profileInitialCondition = Timestep.dataTimestep['profile_mminus1']
profileAtEndOfTimestep = Timestep.dataTimestep['profile_m']
x = Timestep.dataTimestep['x']


print(Timestep.available_solution_fields())
# prints out ['profile', 'c', 'fluxTurbGrid', 'D', 'H2', 'H3', 'DHatTurbGrid', 'cHatTurbGrid',
#             'thetaTurbGrid', 'fluxEWMATurbGrid', 'profileEWMATurbGrid']
#  These are all 2D arrays, first dimension=iteration number, second dimension=spatial index
#  access these as follows:
DHat = Timestep.dataIterations['DHatTurbGrid']

###############################
#For a single iteration

lastiter = Timestep.get_last_iteration()
lastiter.plot_profile_and_starting_profile(savename='solution.png')
iteration5 = Timestep.get_nth_iteration(5)

# note, the initial condition is NOT iteration 0.  Get the initial condition using 'profile_mminus1' as above
print('--------------------------------')
print('For an iteration:')
print(iteration5.available_timestep_fields())
# prints out ['m', 'EWMAParamProfile', 'errhistory', 't', 'EWMAParamTurbFlux', 'x', 
#             'profile_m', 'profile_mminus1', 'l']

# Access these as follows
iterationNumber = iteration5.dataTimestep['iterationNumber'] # get 5

print(iteration5.available_solution_fields())
# prints out ['profile', 'c', 'fluxTurbGrid', 'D', 'H2', 'H3', 'DHatTurbGrid', 
#             'cHatTurbGrid', 'thetaTurbGrid', 'fluxEWMATurbGrid', 'profileEWMATurbGrid']

# Access these using the dictionary data
# These are all 1D arrays, dimension = space
theta = iteration5.data['thetaTurbGrid']
DHat = iteration5.data['DHatTurbGrid']

plt.figure()
plt.plot(x, DHat)
plt.xlabel('x')
plt.title(r'$\hat{D}$ at iteration 5')
