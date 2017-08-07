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

import tango.tango_logging as tlog
from tango.extras import shestakov_nonlinear_diffusion
import tango
import tango.analysis
import tango.lodestro_method
import tango.solver
import tango.multifield
from tango import derivatives

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

class ComputeAllH_field0(object):
    def __init__(self):
        pass
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['default']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        S0 = 1
        delta = 0.1
        H7 = shestakov_nonlinear_diffusion.GetSource(x, S0=S0, delta=delta)
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
class ComputeAllH_field1(object):
    def __init__(self):
        pass
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['default']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        S0 = 1000
        delta = 0.2
        H7 = shestakov_nonlinear_diffusion.GetSource(x, S0=S0, delta=delta)
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
class ShestakovTwoFluxModel(object):
    def __init__(self, dx):
        self.dx = dx
    def get_flux(self, profiles):
        n0 = profiles['n0']
        n1 = profiles['n1']
        
        fluxes = {}
        fluxes['n0'] = shestakov_nonlinear_diffusion.get_flux(n0, dx)
        fluxes['n1'] = shestakov_nonlinear_diffusion.get_flux(n1, dx)
        return fluxes
        
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
tlog.setup()


tlog.info("Initializing...")
L, N, dx, x, nL, n = initialize_shestakov_problem()

n1 = 1.0 * n
nL0 = 0.01
nL1 = 0.01
maxIterations, lmParams, tol = initialize_parameters()
fluxModel = ShestakovTwoFluxModel(dx)

label0 = 'n0'
label1 = 'n1'

turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)

# set up for field0
compute_all_H_field0 = ComputeAllH_field0()
lm0 = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field0 = tango.multifield.Field(label=label0, rightBC=nL0, profile_mminus1=n, compute_all_H=compute_all_H_field0, lodestroMethod=lm0)

# set up for field1
compute_all_H_field1 = ComputeAllH_field1()
lm1 = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field1 = tango.multifield.Field(label=label1, rightBC=nL1, profile_mminus1=n1, compute_all_H=compute_all_H_field1, lodestroMethod=lm1)

# combine fields and do checking
fields = [field0, field1]
tango.multifield.check_fields_initialize(fields)

compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)

tArray = np.array([0, 1e4])  # specify the timesteps to be used.

solver = tango.solver.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)


# set up data logger
#arraysToSave = ['H2', 'H3', 'profile']  # for list of possible arrays, see solver._pkgdata()
#dataBasename = 'shestakov_solution_data'
#solver.dataSaverHandler.initialize_datasaver(dataBasename, maxIterations, arraysToSave)
#tlog.info("Preparing DataSaver to save files with prefix {}.".format(dataBasename))

tlog.info("Initialization complete.")


tlog.info("Entering main time loop...")
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()

n0 = solver.profiles[label0]
n1 = solver.profiles[label1]
    
    
#n = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
nss0 = shestakov_nonlinear_diffusion.steady_state_solution(x, nL0, S0=1, delta=0.1)
nss1 = shestakov_nonlinear_diffusion.steady_state_solution(x, nL1, S0=1000, delta=0.2)

fig = plt.figure()
line1, = plt.plot(x, n0, 'b-', label='numerical solution')
line2, = plt.plot(x, nss0, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.title('Field 0')
plt.legend(handles=[line1, line2])

fig = plt.figure()
line1, = plt.plot(x, n1, 'b-', label='numerical solution')
line2, = plt.plot(x, nss1, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.title('Field 1')
plt.legend(handles=[line1, line2])

solutionResidual = (n0 - nss0) / np.max(np.abs(nss0))
solutionRmsError = np.sqrt( 1/len(n0) * np.sum(solutionResidual**2))

if solver.reachedEnd == True:
    print('The solution has been reached successfully.')
    print('Error compared to analytic steady state solution is %f' % (solutionRmsError))
else:
    print('The solver failed for some reason.')
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))

#  Plot the residual
#plt.figure()
#plt.semilogy(solver.errHistoryFinal)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')


#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
# filename = dataBasename + "1"
# Timestep = tango.analysis.TimestepData(filename)
# lastiter = Timestep.get_last_iteration()
# lastiter.plot_profile_and_starting_profile(savename='solution.png')

# ANALYSIS of BUG
def integrated_source(x, S0=1, delta=0.1):
    """compute the integrated source, \int_0^x dx' S(x')."""
    intSource = np.zeros_like(x)
    intSource[x < delta] = S0 * x
    intSource[x >= delta] = S0 * delta
    return intSource
    
Gamma0 = shestakov_nonlinear_diffusion.get_flux(n0, dx)
Gamma0ss = shestakov_nonlinear_diffusion.get_flux(nss0, dx)
plt.figure()
line1, = plt.plot(x, Gamma0, 'b-', label='numerical Gamma0')
line2, = plt.plot(x, Gamma0ss, 'r-', label='Gamma0ss')
plt.legend(handles=[line1, line2])

Gamma1 = shestakov_nonlinear_diffusion.get_flux(n1, dx)
Gamma1ss = shestakov_nonlinear_diffusion.get_flux(nss1, dx)
plt.figure()
line1, = plt.plot(x, Gamma1, 'b-', label='numerical Gamma0')
line2, = plt.plot(x, Gamma1ss, 'r-', label='Gamma1ss')
plt.legend(handles=[line1, line2])