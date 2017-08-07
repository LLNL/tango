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
import h5py

import tango.tango_logging as tlog
from tango.extras import shestakov_nonlinear_diffusion
import tango
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
    def __init__(self):
        pass
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['default']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)

        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        
        return HCoeffs
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
tlog.setup()


tlog.info("Initializing...")
L, N, dx, x, nL, n = initialize_shestakov_problem()
maxIterations, lmParams, tol = initialize_parameters()
fluxModel = shestakov_nonlinear_diffusion.AnalyticFluxModel(dx)

label = 'n'


turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)

compute_all_H_density = ComputeAllH()
lodestroMethod = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field0 = tango.multifield.Field(label=label, rightBC=nL, profile_mminus1=n, compute_all_H=compute_all_H_density, lodestroMethod=lodestroMethod)
fields = [field0]
tango.multifield.check_fields_initialize(fields)

compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)

tArray = np.array([0, 1e4])  # specify the timesteps to be used.

# set up handler
setNumber = 0
xTango = x
xTurb = x
t = tArray[1]
initialData = tango.handlers.TangoHistoryHandler.set_up_initialdata(setNumber, xTango, xTurb, t, fields)
basename = 'tangodata'
tangoHistoryHandler = tango.handlers.TangoHistoryHandler(iterationInterval=1, basename=basename, maxIterations=maxIterations, initialData=initialData)
filename = basename + '_s{}'.format(setNumber) + '.hdf5'

# create solver
solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)
solver.fileHandlerExecutor.add_handler(tangoHistoryHandler)
#  note: running a handler causes the solver to take much longer when using an analytic flux, because the limiting step becomes the opening & closing
#  of the file on each iteration.


tlog.info("Initialization complete.")


tlog.info("Entering main time loop...")
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()

n = solver.profiles[label]

f = h5py.File(filename, 'r')
    
    
#n = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
nss = shestakov_nonlinear_diffusion.steady_state_solution(x, nL)

fig = plt.figure()
line1, = plt.plot(x, n, 'b-', label='numerical solution')
line2, = plt.plot(x, nss, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(handles=[line1, line2])

solutionResidual = (n - nss) / np.max(np.abs(nss))
solutionRmsError = np.sqrt( 1/len(n) * np.sum(solutionResidual**2))

if solver.reachedEnd == True:
    print('The solution has been reached successfully.')
    print('Error compared to analytic steady state solution is %f' % (solutionRmsError))
else:
    print('The solver failed for some reason and did not reach the specified convergence tolerance.')
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))


#plt.figure()
#plt.semilogy(errhistory)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
# filename = dataBasename + "1"
# Timestep = tango.analysis.TimestepData(filename)
# lastiter = Timestep.get_last_iteration()
# lastiter.plot_profile_and_starting_profile(savename='solution.png')