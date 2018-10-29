"""Example for how to use tango to solve a turbulence and transport problem.

Using a solver class, saving to a file, and using the analysis package to load the data and save a plot

Here, the "turbulent flux" is specified analytically, using the example in the Shestakov et al. (2003) paper.
This example is a nonlinear diffusion equation with specified diffusion coefficient and source.  There is a
closed form answer for the steady state solution which can be compared with the numerically found solution.
"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib.pyplot as plt

import tango.tango_logging as tlog
from tango.extras import shestakov_nonlinear_diffusion
import tango

def initialize_shestakov_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    #nL = 1e-2           # right boundary condition
    #nInitialCondition = 1 - 0.5*x
    #nL = 1e-2           # right boundary condition
    #nInitialCondition = -0.02 * (x - 1) + 0.014
    nL = 1           # right boundary condition
    nInitialCondition = -0.2 * (x - 1) + 1.2
    return (L, N, dx, x, nL, nInitialCondition)

def initialize_parameters():
    maxIterations = 1000
    thetaParams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdxThreshold': 10}
    EWMAParamTurbFlux = 0.1
    EWMAParamProfile = 1
    lmParams = {'EWMAParamTurbFlux': EWMAParamTurbFlux,
            'EWMAParamProfile': EWMAParamProfile,
            'thetaParams': thetaParams}
    tol = 1e-10  # tol for convergence... reached when a certain error < tol
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
L, N, dx, x, nL, nIC = initialize_shestakov_problem()
n = nIC
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

treadInitiallyParams = {'numInitialSteps': 6,
                        'dtau': 0.3}
solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields,
                             useTreadInitially=True, treadInitiallyParams=treadInitiallyParams)

tlog.info("Initialization complete.")
tlog.info("Entering main time loop...")

while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()

n = solver.profiles[label] # finished solution
    
# Plot result and compare with analytic steady state solution
nss = shestakov_nonlinear_diffusion.steady_state_solution(x, nL)

#fig = plt.figure()
#line1, = plt.plot(x, n, 'b-', label='numerical solution')
#line2, = plt.plot(x, nss, 'r-', label='analytic solution')
#plt.xlabel('x')
#plt.ylabel('n')
#plt.legend(handles=[line1, line2])

solutionResidual = (n - nss) / np.max(np.abs(nss))
solutionRmsError = np.sqrt( 1/len(n) * np.sum(solutionResidual**2))

if solver.reachedEnd == True:
    print('The solution has been reached successfully.')
    print('Error compared to analytic steady state solution is %f' % (solutionRmsError))
else:
    print('The solver failed for some reason.')
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))

# profile vs. iterations
profileAllIterations = solver.profilesAllIterations[label]

plt.figure()
plt.plot(x, nIC, 'k--')
jmin = 0
#jmax = len(profileAllIterations)
jmax = 40
for j in np.arange(0, jmax):
    r = 1 - (j-jmin)/(jmax-jmin)
    g = 0
    b = (j-jmin)/(jmax-jmin)
    plt.plot(x, profileAllIterations[j], color=(r, g, b), alpha=1)
plt.xlabel('x')

#plt.figure()
#plt.semilogy(solver.errHistoryFinal)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')

plt.figure()
plt.plot(x, nIC, 'k--')
plt.plot(x, profileAllIterations[0], color='b')
plt.plot(x, profileAllIterations[1], color='r')
plt.plot(x, profileAllIterations[2], color='g')
plt.plot(x, profileAllIterations[3], color='b', linestyle=':')
plt.plot(x, profileAllIterations[4], color='r', linestyle=':')
plt.plot(x, profileAllIterations[5], color='g', linestyle=':')
plt.plot(x, profileAllIterations[6], color='b', linestyle='-.')
plt.plot(x, profileAllIterations[7], color='r', linestyle='-.')
plt.plot(x, profileAllIterations[8], color='g', linestyle='-.')
plt.plot(x, profileAllIterations[15], color='k', linestyle='-')
#plt.plot(x, profileAllIterations[9], color='b', linestyle='--')
#plt.plot(x, profileAllIterations[10], color='r', linestyle='--')
#plt.plot(x, profileAllIterations[11], color='g', linestyle='--')
