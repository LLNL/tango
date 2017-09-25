"""Analytic example.  Use Tango to find the steady-state solution of:

dn/dt + dG/dx = Sn(x) 
dpi/dt + dQi/dx = Si(x) - nu * (pi - pe)
dpe/dt + dQe/dx = Se(x) - nu * (pe - pi)

where
    G = -D dn/dx
    Qi = -D dpi/dx
    Qe = -D dpe/dx
and
    nu = nu0 / ( pi/n + pe/n)^(3/2) 
    D = (dpi/dx)^2 / pi^2

Here, p0 is independent as in the Shestakov example, and p1 is coupled to p0.

"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

import tango.tango_logging as tlog
import tango
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

class ShestakovThreeFieldFluxModel(object):
    def __init__(self, dx):
        self.dx = dx
    def get_flux(self, profiles):
        n = profiles['n']
        pi = profiles['pi']
        pe = profiles['pe']
    
        # Return flux Gamma on the same grid as n
        dndx = tango.derivatives.dx_centered_difference_edge_first_order(n, dx)
        dpidx = tango.derivatives.dx_centered_difference_edge_first_order(pi, dx)
        dpedx = tango.derivatives.dx_centered_difference_edge_first_order(pe, dx)
        D = dpidx**2 / pi**2
        Gamma = -D * dndx
        Qi = -D * dpidx
        Qe = -D * dpedx
        
        fluxes = {}
        fluxes['n'] = Gamma
        fluxes['pi'] = Qi
        fluxes['pe'] = Qe
        return fluxes

def source_n(x, S0=8, delta=0.3):
    """Return the source S_n."""
    S = np.zeros_like(x)
    S[x < delta] = S0
    return S
        
def source_i(x, S0=1, delta=0.1):
    """Return the source S_i."""
    S = np.zeros_like(x)
    S[x < delta] = S0
    return S
    
def source_e(x, S0=3, delta=0.4):
    """Return the source S_e."""
    S = np.zeros_like(x)
    ind = x < delta
    S[ind] = S0 * x[ind]
    return S
    
    
class ComputeAllH_n(object):
    def __call__(self, t, x, profiles, HCoeffsTurb):
        #pi = profiles['pi']
        #pe = profiles['pe']
        #n = profiles['field0']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = source_n(x)
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
    
def calc_nu(nu0, n, pi, pe):
    nu = nu0 / (( pi/n + pe/n) ** (3/2) )
    return nu
        
class ComputeAllH_pi(object):
    def __init__(self, nu0):
        self.nu0 = nu0
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        pi = profiles['pi']
        pe = profiles['pe']
        #n = profiles['field0']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = source_i(x)
        
        nu = calc_nu(self.nu0, n, pi, pe)
        H6 = -nu
        H8 = nu
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H6=H6, H7=H7, H8=H8)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
class ComputeAllH_pe(object):
    def __init__(self, nu0):
        self.nu0 = nu0
    def __call__(self, t, x, profiles, HCoeffsTurb):
        n = profiles['n']
        pi = profiles['pi']
        pe = profiles['pe']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = source_e(x)
        
        nu = calc_nu(self.nu0, n, pi, pe)
        H6 = -nu
        H8 = nu
        
        HCoeffs = tango.multifield.HCoefficients(H1=H1, H6=H6, H7=H7, H8=H8)
        HCoeffs = HCoeffs + HCoeffsTurb
        return HCoeffs
        
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
tlog.setup()


tlog.info("Initializing...")
L, N, dx, x, nL, n = initialize_shestakov_problem()

n_IC = 1.0 * n
pi_IC = 1.0 * n
pe_IC = 1.0 * n

n_L = 2
pi_L = 0.1
pe_L = 0.3

nu0 = 2.2

maxIterations, lmParams, tol = initialize_parameters()


label0 = 'n'
label1 = 'pi'
label2 = 'pe'
labels = [label0, label1, label2]


# set up for n
compute_all_H_n = ComputeAllH_n()
lm_n = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field0 = tango.multifield.Field(label=label0, rightBC=n_L, profile_mminus1=n_IC, compute_all_H=compute_all_H_n, lodestroMethod=lm_n)

# set up for pi
compute_all_H_pi = ComputeAllH_pi(nu0)
lm_pi = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field1 = tango.multifield.Field(label=label1, rightBC=pi_L, profile_mminus1=pi_IC, compute_all_H=compute_all_H_pi, lodestroMethod=lm_pi, coupledTo='pe')

# set up for pe
compute_all_H_pe = ComputeAllH_pe(nu0)
lm_pe = tango.lodestro_method.lm(lmParams['EWMAParamTurbFlux'], lmParams['EWMAParamProfile'], lmParams['thetaParams'])
field2 = tango.multifield.Field(label=label2, rightBC=pe_L, profile_mminus1=pe_IC, compute_all_H=compute_all_H_pe, lodestroMethod=lm_pe, coupledTo='pi')

# combine fields and do checking
fields = [field0, field1, field2]
tango.multifield.check_fields_initialize(fields)

# create the flux model and the turbulence handler
fluxModel = ShestakovThreeFieldFluxModel(dx)
turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)
compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(fields, turbHandler)


tArray = np.array([0, 1e6])  # specify the timesteps to be used.

# initialize the solver
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

if solver.reachedEnd == True:
    print('The solution has been reached successfully.')
else:
    print('The solver failed for some reason.')    
    
n = solver.profiles[label0]    
pi = solver.profiles[label1]
pe = solver.profiles[label2]
    
#n = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
#nss0 = shestakov_nonlinear_diffusion.GetSteadyStateSolution(x, nL0)

fig = plt.figure()
line1, = plt.plot(x, n, 'b-', label='numerical solution n')
#line2, = plt.plot(x, nss0, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('n')
#plt.title('Field 0')
plt.legend(handles=[line1])

fig = plt.figure()
line1, = plt.plot(x, pi, 'b-', label='numerical solution pi')
#line2, = plt.plot(x, nss0, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('pi')
#plt.title('Field 0')
plt.legend(handles=[line1])

fig = plt.figure()
line1, = plt.plot(x, pe, 'b-', label='numerical solution pe')
#line2, = plt.plot(x, nss1, 'r-', label='analytic solution')
plt.xlabel('x')
plt.ylabel('pe')
#plt.title('Field 1')
plt.legend(handles=[line1])


fluxes = fluxModel.get_flux(solver.profiles)
Gamma = fluxes['n']
Qi = fluxes['pi']
Qe = fluxes['pe']

RHSn = source_n(x)
RHSi = source_i(x) - calc_nu(nu0, n, pi, pe) * (pi - pe)
RHSe = source_e(x) - calc_nu(nu0, n, pi, pe) * (pe - pi)

RHSn_integrated = scipy.integrate.cumtrapz(RHSn, x=x, initial=0)
RHSi_integrated = scipy.integrate.cumtrapz(RHSi, x=x, initial=0)
RHSe_integrated = scipy.integrate.cumtrapz(RHSe, x=x, initial=0)


plt.figure()
line1, = plt.plot(x, Gamma, 'b-', label='numerical flux for n')
line2, = plt.plot(x, RHSn_integrated, 'r-', label='integrated sources')
plt.title('n')
plt.legend(handles=[line1, line2])

plt.figure()
line1, = plt.plot(x, Qi, 'b-', label='numerical flux for pi')
line2, = plt.plot(x, RHSi_integrated, 'r-', label='integrated sources')
plt.title('pi')
plt.legend(handles=[line1, line2])

plt.figure()
line1, = plt.plot(x, Qe, 'b-', label='numerical flux for pe')
line2, = plt.plot(x, RHSe_integrated, 'r-', label='integrated sources')
plt.title('pe')
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