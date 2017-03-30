"""Demonstrate the transport solver in a GENE-like situation where the transport grid and the turbulence grid are
not the same.  In this example, the turbulence grid resides *inside* the transport grid.  Furthermore, as in GENE,
the turbulent flux model in this example will have a 'buffer zone' applied to it that damps out the flux near both
boundaries.

In this example:
    On the right side of the domain, the transport coefficients are *extrapolated* from a region to the left of the
    buffer zone, all the way to the edge of the transport grid.
    
    On the left side of the domain, one could in principle perform similar extrapolation, but that is not done here.
    Instead, the heating source is only nonzero in the region where the turbulent flux model is active.  So in both
    the left buffer zone and the region near the left boundary which the turbulence grid does not cover, there is
    no heating source --- this tends to make the profile approximately flat.  A small amount of diffusivity is then
    added everywhere for regularization.
"""

from __future__ import division, absolute_import
import numpy as np

from tango.extras import shestakov_nonlinear_diffusion, bufferzone
import tango as tng
import matplotlib.pyplot as plt

from tango import analysis
import tango.tango_logging as tlog
tlog.setup(False, 0, tlog.INFO)        # when in a serial environment

def initialize_shestakov_problem():
    # Problem Setup
    L = 1           # size of domain
    N = 300         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    nL = 0.6           # right boundary condition
    n_initialcondition = 1 - 0.5*x
    return (L, N, dx, x, nL, n_initialcondition)

def initialize_parameters():
    maxIterations = 2000
    thetaparams = {'Dmin': 1e-5,
                   'Dmax': 1e13,
                   'dpdxThreshold': 10}
    EWMA_param_turbflux = 0.1
    EWMA_param_profile = 0.1
    lmparams = {'EWMAParamTurbFlux': EWMA_param_turbflux,
            'EWMAParamProfile': EWMA_param_profile,
            'thetaParams': thetaparams}
    tol = 1e-9  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmparams, tol)

def source(x, S0=1):
    """Source that is nonzero for xa <= x <= xb.
    Return the source S."""
    S = np.zeros_like(x)
    xa = 0.25
    xb = 0.45
    S[(x >= xa) & (x <= xb)] = S0
    return S
    
    
class ComputeAllHNoBuffer(object):
    def __init__(self, turbhandler):
        self.turbhandler = turbhandler
    def __call__(self, t, x, n):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        #H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        H7 = source(x)
        (H2turb, H3, extradata) = self.turbhandler.Hcontrib_turbulent_flux(n)
        H4 = None
        H6 = None
        # add "other" diffusive contributions by specifying a diffusivity, H2 = V'D [but V' = 1 here]
        H2constdiff = 0.03
        
        H2 = H2turb + H2constdiff
        return (H1, H2, H3, H4, H6, H7, extradata)
        
class ComputeAllHWithBuffer(object):
    def __init__(self, turbhandler):
        self.turbhandler = turbhandler
    def __call__(self, t, x, n):
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        #H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)
        H7 = source(x)
        (H2turb, H3, extradata) = self.turbhandler.Hcontrib_turbulent_flux(n)
        H4 = None
        H6 = None
        # add "other" diffusive contributions by specifying a diffusivity, H2 = V'D [but V' = 1 here]
        H2constdiff = 0.03
        
        def diffusivity_right(x):
            diffusivity = np.zeros_like(x)
            xr = 0.85
            D0 = 7
            diffusivity[x > xr] = D0
            return diffusivity
        
        H2 = H2turb + H2constdiff    
        #H2 = H2turb + H2constdiff + diffusivity_right(x)   # if adding const to right edge
        return (H1, H2, H3, H4, H6, H7, extradata)
        
def regular_solution():
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    maxIterations, lmparams, tol = initialize_parameters()
    fluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dx)
    turbhandler = tng.lodestro_method.TurbulenceHandler(dx, x, lmparams, fluxModel)
    t_array = np.array([0, 1e4])  # specify the timesteps to be used.
    compute_all_H = ComputeAllHNoBuffer(turbhandler)
    solver = tng.solver.Solver(L, x, n, nL, t_array, maxIterations, tol, compute_all_H, turbhandler)
    arraysToSave = ['H2', 'H3', 'profile', 'fluxTurbGrid', 'xTurbGrid', 'DTurbGrid']  # for list of possible arrays, see solver._pkgdata()
    dataBasename = 'nobuffer'
    solver.dataSaverHandler.initialize_datasaver(dataBasename, maxIterations, arraysToSave)
    while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
        solver.take_timestep()
    nSteadyState = solver.profile
    return nSteadyState
    
L, N, dxTangoGrid, x, nL, n = initialize_shestakov_problem()
xTango = x
maxIterations, lmparams, tol = initialize_parameters()

#==============================================================================
#  MAIN STARTS HERE :  Do a bunch of set up
#==============================================================================

# ========== Set up the Turbulence grid ===========================
NTurb = 400
x1 = 0.1
x2 = 0.9
xTurb = np.linspace(x1, x2, NTurb)
dxTurbGrid = xTurb[1] - xTurb[0]

# create gridMapper, fluxModel
xExtrapZoneLeft = 0.7
xExtrapZoneRight = 0.77
polynomialDegree = 2
gridMapper = tng.interfacegrids_gene.TangoOutsideExtrapCoeffs(xTango, xTurb, xExtrapZoneLeft, xExtrapZoneRight, polynomialDegree)

# model for turbulent flux: Shestakov model
fluxModel = shestakov_nonlinear_diffusion.shestakov_analytic_fluxmodel(dxTurbGrid)
# apply decorator to create a 'buffer zone'
fluxModel = bufferzone.BufferZone(fluxModel, taperwidth=0.125)

turbhandler = tng.lodestro_method.TurbulenceHandler(dxTurbGrid, x, lmparams, fluxModel, gridMapper=gridMapper)
t_array = np.array([0, 1e4])  # specify the timesteps to be used.

compute_all_H = ComputeAllHWithBuffer(turbhandler)
solver = tng.solver.Solver(L, x, n, nL, t_array, maxIterations, tol, compute_all_H, turbhandler)

# set up data saver
arraysToSave = ['H2', 'H3', 'profile', 'fluxTurbGrid', 'xTurbGrid', 'DTurbGrid']  # for list of possible arrays, see solver._pkgdata()
dataBasename = 'withbuffer'
solver.dataSaverHandler.initialize_datasaver(dataBasename, maxIterations, arraysToSave)
tlog.info("Preparing DataSaver to save files with prefix {}.".format(dataBasename))

while solver.ok:
    # Implicit time advance: iterate to solve the nonlinear equation!
    solver.take_timestep()
 
#plt.figure()
nSteadyState = solver.profile
#plt.plot(x, nSteadyState)
#plt.title("With Buffer: n_final")
#
nSteadyState2 = regular_solution()
#plt.figure()
#plt.plot(x, nSteadyState2)
#plt.title("No Buffer: n_final")

filenameNoBuffer = "nobuffer1"
filenameBuffer = dataBasename + "1"
timestepNoBuffer = analysis.TimestepData(filenameNoBuffer)
timestepBuffer = analysis.TimestepData(filenameBuffer)
lastiterNoBuffer = timestepNoBuffer.get_last_iteration()
lastiterBuffer = timestepBuffer.get_last_iteration()


plt.figure()
plt.plot(x, nSteadyState2, 'r-', label='No Buffer')
plt.plot(x, nSteadyState, 'b-', label='With Buffer')
plt.legend(loc='best')
plt.title('n_final')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.figure()
plt.plot(lastiterNoBuffer.x, lastiterNoBuffer.H2, 'r-', label='No Buffer')
plt.plot(lastiterBuffer.x, lastiterBuffer.H2, 'b-', label='With Buffer')
plt.legend(loc='best')
plt.title('H2_final')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()