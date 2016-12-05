"""Example for how to use tango to solve a turbulence and transport problem with "additional" transport physics.

The physics package can add in thermal diffusivities for general chi(psi).  It can also provide the neoclassical
chi specifically.
"""

from __future__ import division, absolute_import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging


from tango.extras import shestakov_nonlinear_diffusion
import tango as tng

def initialize_diffusion_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dr = L / (N - 0.5)  # spatial grid size
    r = np.linspace(dr/2, L, N)   # location corresponding to grid points j=0, ..., N-1
    pL = 1
    # initial conditions
    p_initialcondition = 0.3*np.sin(np.pi * r) + pL * r
    
    # diffusion problem coefficients
    D = 1.3

    return (L, N, dr, r, pL, p_initialcondition, D)

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
    tol = 1e-7  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)

class ComputeAllH(object):
    def __init__(self, HcontribTransportPhysics):
        self.HcontribTransportPhysics = HcontribTransportPhysics
        self.Vprime = HcontribTransportPhysics.profilesAll.Vprime
        self.gradpsisq = HcontribTransportPhysics.profilesAll.gradpsisq
    def __call__(self, t, r, p):
        # Define the contributions to the H coefficients for this test problem
        D = 0.05
        timeDerivCoeff = np.ones_like(r)
        chi = D * np.ones_like(r)
        S = r   # mockup source
        
        H1 = self.HcontribTransportPhysics.time_derivative_to_H(timeDerivCoeff)
        # add some manually input diffusivity chi
        (H2_a, H3_a) = self.HcontribTransportPhysics.Hcontrib_thermal_diffusivity(chi)
        
        # add the neoclassical diffusivity chi (for banana regime)
        (H2_b, H3_b) = self.HcontribTransportPhysics.Hcontrib_neoclassical_thermal_diffusivity(p)
        H7 = self.HcontribTransportPhysics.source_to_H(S)
        
        H2 = H2_a + H2_b
        H3 = H3_a + H3_b
        
        extradata = {} # because there is no turbulent flux.    
        H4 = None
        H6 = None
        
        return (H1, H2, H3, H4, H6, H7, extradata)
        
    
#==============================================================================
#  MAIN STARTS HERE
#==============================================================================
logfile = 'example_class.log'
#logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
logging.basicConfig(level=logging.INFO) 


logging.info("Initializing...")
L, N, dr, r, nL, p, D = initialize_diffusion_problem()



MaxIterations, lmparams, tol = initialize_parameters()
FluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dr)
turbhandler = tng.TurbulenceHandler(dr, r, lmparams, FluxModel)

# initialize transport module: profiles (with a slight modification to set Vprime to psi=r/a)
mu = 1
n = 1e19 * np.ones_like(r) # density
psi = r     # dr to 1

minorRadius = 1
majorRadius = 1
B0 = 1
Vprime = r
gradPsiSq = np.ones_like(r)
profilesAll = tng.physics.initialize_profile_defaults(mu, n, psi, minorRadius, majorRadius, B0, Vprime, gradPsiSq)
HcontribTransportPhysics = tng.physics_to_H.Hcontrib_TransportPhysics(profilesAll)


t_array = np.array([0, 1e4])  # specify the timesteps to be used.
compute_all_H = ComputeAllH(HcontribTransportPhysics)
solver = tng.solver.Solver(L, r, p, nL, t_array, MaxIterations, tol, compute_all_H, turbhandler)

# set up data logger
#arrays_to_save = ['H2', 'H3', 'profile']  # for list of possible arrays, see solver._pkgdata()
#data_basename = 'test_transport_physics_module_data'
#solver.DataSaverHandler.initialize_datasaver(data_basename, MaxIterations, arrays_to_save)
#logging.info("Preparing DataSaver to save files with prefix {}.".format(data_basename))

logging.info("Initialization complete.")

logging.info("Beginning time integration...")
while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()


    
    
p = solver.profile  # finished solution
# Plot result and compare with analytic steady state solution
# pss = nL + 1/(9*D) * (1-r**3) # analytic solution when neoclassical transport = 0

fig = plt.figure()
plt.plot(r, p, 'b-')
#plt.plot(r, pss, 'r-')

#solution_residual = (p - pss) / np.max(np.abs(pss))
#solution_rms_error = np.sqrt( 1/len(p) * np.sum(solution_residual**2))

if solver.reachedEnd == True:
    print("The solution has been reached successfully.")
    print("Took {} iterations".format(solver.l))
else:
    print("The solver failed for some reason.  See log file {}".format(logfile))

#plt.figure()
#plt.semilogy(errhistory)
#plt.xlabel('iteration number')
#plt.ylabel('rms error')
#plt.plot(x, n-nss)
#plt.ylim(ymin=0)
#filename = data_basename + "1"
#Timestep = tng.analysis.TimestepData(filename)
#lastiter = Timestep.GetLastIteration()
#lastiter.PlotProfileAndStartingProfile(savename='solution.png')