"""Analytic example.  Use Tango to find the steady-state solution of:

dp0/dt + dG0/dx = S0(x)
dp1/dt + dG1/dx = S1(x)

where
    G0(x) = -(dp0/dx)^3 / p0^2
    G1(x) = -(dp0/dx)^2 / p0^2 * dp1/dx

Here, p0 is independent as in the Shestakov example, and p1 is coupled to p0.

"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tango.tango_logging as tlog
from tango.extras import shestakov_nonlinear_diffusion
import tango
import tango.analysis
import tango.lodestro_method_multifield
import tango.solver_multifield
import tango.multifield
import tango.derivatives

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
    tol = 1e-9  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)

class ShestakovTwoFieldFluxModel(object):
    def __init__(self, dx):
        self.dx = dx
        self.fluxmodel0 = shestakov_nonlinear_diffusion.AnalyticFluxModel(dx)
    def get_flux(self, profileArray):
        p0 = profileArray[0, :]
        p1 = profileArray[1, :]
        fluxes = np.zeros_like(profileArray)
    
        # Return flux Gamma on the same grid as n
        dp0dx = tango.derivatives.dx_centered_difference_edge_first_order(p0, dx)
        dp1dx = tango.derivatives.dx_centered_difference_edge_first_order(p1, dx)
        D = dp0dx**2 / p0**2
        Gamma0 = -D * dp0dx
        Gamma1 = -D * dp1dx
        
        fluxes[0, :] = Gamma0
        fluxes[1, :] = Gamma1
        return fluxes
        
def source0(x, S0=1, delta=0.1):
    """Return the source S for field0."""
    S = np.zeros_like(x)
    S[x < delta] = S0
    return S      
    
def source1(x, S0=3, delta=0.4):
    """Return the source S for field1."""
    S = np.zeros_like(x)
    ind = x < delta
    S[ind] = S0 * x[ind]
    return S
    
def integrated_source0(x, S0=1, delta=0.1):
    """compute the integrated source, \int_0^x dx' S(x')."""
    intSource = np.zeros_like(x)
    ind = x < delta
    intSource[ind] = S0 * x[ind]
    intSource[x >= delta] = S0 * delta
    return intSource
        
def integrated_source1(x, S0=3, delta=0.4):
    """compute the integrated source, \int_0^x dx' S(x')."""
    intSource = np.zeros_like(x)
    ind = x < delta
    intSource[ind] = S0 * x[ind]**2 / 2
    intSource[x >= delta] = S0 * delta**2 / 2
    return intSource
    
class ComputeAllH_field0(object):
    def __init__(self):
        pass
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['field0']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = source0(x)
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
class ComputeAllH_field1(object):
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #n = profiles['field1']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = source1(x)
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
tlog.setup()


tlog.info("Initializing...")
L, N, dx, x, nL, n = initialize_shestakov_problem()

n1 = 1.0 * n
n2 = 1.0 * n
nL0 = 0.01
nL1 = 0.05

maxIterations, lmParams, tol = initialize_parameters()


label0 = 'pi'
label1 = 'pe'
labels = [label0, label1]


# set up for field0
compute_all_H_field0 = ComputeAllH_field0()
lm0 = tango.lodestro_method_multifield.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field0 = tango.multifield.Field(label=label0, rightBC=nL0, profile_mminus1=n, compute_all_H=compute_all_H_field0, lodestroMethod=lm0)
#field0 = tango.multifield.Field(label=label0, rightBC=nL0, profile_mminus1=n, compute_all_H=compute_all_H_field0, lodestroMethod=lm0, coupledTo='pe')

# set up for field1
compute_all_H_field1 = ComputeAllH_field1()
lm1 = tango.lodestro_method_multifield.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field1 = tango.multifield.Field(label=label1, rightBC=nL1, profile_mminus1=n1, compute_all_H=compute_all_H_field1, lodestroMethod=lm1)
#field1 = tango.multifield.Field(label=label1, rightBC=nL1, profile_mminus1=n1, compute_all_H=compute_all_H_field1, lodestroMethod=lm1, coupledTo='pi')

# combine fields and do checking
fields = [field0, field1]
tango.multifield.check_fields_initialize(fields)

# create the flux model and the turbulence handler
fluxModel = ShestakovTwoFieldFluxModel(dx)
turbHandler = tango.lodestro_method_multifield.TurbulenceHandlerMultifield(dx, x, fluxModel, labels, labels)
compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)


tArray = np.array([0, 1e6])  # specify the timesteps to be used.

# initialize the solver
solver = tango.solver_multifield.Solver(L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields)


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
nss0 = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL0)

fig = plt.figure()
line1, = plt.plot(x, n0, 'b-', label='numerical solution')
line2, = plt.plot(x, nss0, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.title('Field 0')
plt.legend(handles=[line1, line2])

fig = plt.figure()
line1, = plt.plot(x, n1, 'b-', label='numerical solution')
#line2, = plt.plot(x, nss1, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
plt.title('Field 1')
plt.legend(handles=[line1])

solutionResidual = (n0 - nss0) / np.max(np.abs(nss0))
solutionRmsError = np.sqrt( 1/len(n0) * np.sum(solutionResidual**2))

if solver.reachedEnd == True:
    print('The solution has been reached successfully.')
    print('Error compared to analytic steady state solution is %f' % (solutionRmsError))
else:
    print('The solver failed for some reason.')
    print('Error at end compared to analytic steady state solution is %f' % (solutionRmsError))

Gamma = fluxModel.get_flux(np.array((n0, n1)))
Gamma0 = Gamma[0, :]
Gamma1 = Gamma[1, :]
Gamma0_analytic = integrated_source0(x)
Gamma1_analytic = integrated_source1(x)

plt.figure()
line1, = plt.plot(x, Gamma0, 'b-', label='numerical flux')
line2, = plt.plot(x, Gamma0_analytic, 'r-', label='analytic flux')
plt.title('field 0')
plt.legend(handles=[line1, line2])

plt.figure()
line1, = plt.plot(x, Gamma1, 'b-', label='numerical flux')
line2, = plt.plot(x, Gamma1_analytic, 'r-', label='analytic flux')
plt.title('field 1')
plt.legend(handles=[line1, line2])
    
    
plt.figure()
plt.semilogy(solver.errHistoryFinal)
plt.xlabel('iteration number')
plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
# filename = dataBasename + "1"
# Timestep = tango.analysis.TimestepData(filename)
# lastiter = Timestep.get_last_iteration()
# lastiter.plot_profile_and_starting_profile(savename='solution.png')