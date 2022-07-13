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
from tango.extras.fluxrelaxation import FluxRelaxation, FluxDoubleRelaxation
from tango.extras.noisyflux import NoisyFlux, NoisyFluxSpaceTime
import tango

def initialize_shestakov_problem():
    # Problem Setup
    L = 1  # size of domain
    N = 500  # number of spatial grid points
    dx = L / (N - 1)  # spatial grid size
    x = np.arange(N) * dx  # location corresponding to grid points j=0, ..., N-1
    nL = 1e-2  # right boundary condition
    nInitialCondition = 1 - 0.5 * x
    return (L, N, dx, x, nL, nInitialCondition)

class ComputeAllH(object):
    def __init__(self, test_problem):
        self.test_problem = test_problem

    def __call__(self, t, x, profiles, HCoeffsTurb):
        # n = profiles['default']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = self.test_problem.H7contrib_Source(x)

        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb

        return HCoeffs

# ==============================================================================
#  Solve function
# ==============================================================================

def solve_system(noise_timescale = 1.0,    # AR time of random noise
                 turb_timescale = 1.0,     # Flux relaxation time
                 damping_timescale = 1.0,  # Flux damping timescale
                 noise_amplitude = 0.0,    # Amplitude of AR(1) noise
                 noise_scalelength = 10,   # Spatial scale length (cells)
                 EWMAParamTurbFlux = 0.01,
                 EWMAParamProfile = 1.0,
                 thetaParams = {"Dmin": 1e-5, "Dmax": 1e13, "dpdxThreshold": 10},
                 tol = 1e-2,
                 maxIterations = 1000,
                 plot_convergence = False,
                 check_tol = 0.1
                 ):
    tlog.info("Initializing...")
    L, N, dx, x, nL, n = initialize_shestakov_problem()
    test_problem = shestakov_nonlinear_diffusion.ShestakovTestProblem(dx, p=15, q=-14)

    fluxModel = NoisyFluxSpaceTime(
        FluxDoubleRelaxation(
            test_problem,
            turb_timescale,
            damping_timescale,
        ),
        noise_amplitude,
        noise_scalelength,
        1.0, # dx
        noise_timescale, # noise timescale
        1.0 # dt
    )

    label = "n"
    turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)

    lodestroMethod = tango.lodestro_method.lm(
        EWMAParamTurbFlux,
        EWMAParamProfile,
        thetaParams,
    )
    field0 = tango.multifield.Field(
        label=label,
        rightBC=nL,
        profile_mminus1=n,
        compute_all_H=ComputeAllH(test_problem),
        lodestroMethod=lodestroMethod,
    )
    fields = [field0]
    tango.multifield.check_fields_initialize(fields)

    compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(
        fields, turbHandler
    )

    tArray = np.array([0, 1e4])  # specify the timesteps to be used.

    solver = tango.solver.Solver(
        L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields,
        saveFluxesInMemory = plot_convergence
    )

    tlog.info("Initialization complete.")
    tlog.info("Entering main time loop...")

    err_history = []
    profile_history = []
    flux_history = []
    num_iterations = 0
    restarts = []
    while num_iterations < maxIterations:
        try:
            while solver.ok:
                # Implicit time advance: iterate to solve the nonlinear equation!
                solver.take_timestep()
        except Exception as e:
            # Failed
            print("Solver failed with error: {}".format(e))
            solver.reachedEnd = False
            solutionResidual = 1e10
            solutionRmsError = 1e10
            num_iterations = maxIterations
            break

        n = solver.profiles[label]  # finished solution

        source = test_problem.GetSource(x)
        nss = test_problem.steady_state_solution(x, nL)

        solutionResidual = (n - nss) / np.max(np.abs(nss))
        solutionRmsError = np.sqrt(1 / len(n) * np.sum(solutionResidual ** 2))

        num_iterations += len(solver.errHistoryFinal)
        err_history.append(solver.errHistoryFinal.copy())
        profile_history.append(solver.profilesAllIterations[label].copy())
        flux_history.append(solver.fluxesAllIterations[label].copy())

        if solver.reachedEnd == True:
            print("The solution has been reached successfully.")
            print(
                "Error compared to analytic steady state solution is %f"
                % (solutionRmsError)
            )

            if solutionRmsError > check_tol:
                print("False convergence!")
                solver.m = 0 # Reset timestep index
                solver.t = 0.0

                restarts.append(num_iterations)
            else:
                break
        else:
            print("The solver failed for some reason.")
            print(
                "Error at end compared to analytic steady state solution is %f"
                % (solutionRmsError)
            )
            if solutionRmsError < check_tol:
                # Succeeded anyway
                restarts.append(num_iterations)
                solver.reachedEnd = True
            else:
                num_iterations = maxIterations
            break # Give up

    if err_history != []:
        err_history = np.concatenate(err_history)
        profiles = np.concatenate(profile_history)
        fluxes = np.concatenate(flux_history)

    if plot_convergence:
        for j in range(num_iterations):
            fig, [ax1, ax2, ax3] = plt.subplots(1,3)

            # Profile and flux history in grey
            for i in range(num_iterations):
                ax1.plot(profiles[i,:], color='0.6')
                ax2.plot(fluxes[i,:], color='0.6')

            # Plot the converged profiles and fluxes in black
            ax1.plot(profiles[-1,:], color='k')
            ax2.plot(fluxes[-1,:], color='k')

            # Current step in blue
            ax1.plot(profiles[j,:], color='b')
            ax2.plot(fluxes[j,:], color='b')

            ax3.plot(err_history, color='k')
            ax3.plot([j], [err_history[j]], 'bo')
            ax3.set_yscale('log')
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Residual")
            plt.show()

        import sys
        sys.exit(0)

    return {"converged": solver.reachedEnd,
            "num_iterations": num_iterations,
            "residual": solutionResidual,
            "rms_error": solutionRmsError,
            "err_history": err_history}

def collect_statistics(inputs, num_samples=20):
    iterations = []
    for i in range(num_samples):
        result = solve_system(**inputs)
        iterations.append(result["num_iterations"])

    result["max_iterations"] = max(iterations)
    result["min_iterations"] = min(iterations)
    result["mean_iterations"] = np.mean(np.array(iterations))
    return result

# ==============================================================================
#  Settings
# ==============================================================================

timescales = [1.0] #1e-3, 0.1, 0.2, 0.5, 1, 2, 5, 10]
damping_multipliers = [] #[1.0, 2.0, 5.0, 10]
noise_multipliers = [1e-2]#, 1e-1, 1.0]

tolerance = 1e-2
alpha_profile = 0.0
alpha_flux = 0.0

noise_amplitude = 0.01  # Amplitude of noise to add
noise_scalelength = 10.0  # Spatial scale length (number of cells)

num_samples = 1  # Number of repeated solves, to gather statistics

# ==============================================================================
#  MAIN STARTS HERE
# ==============================================================================
tlog.setup()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for damping_multiplier in damping_multipliers:
    for noise_multiplier in noise_multipliers:
        mean_iterations = []
        max_iterations = []
        min_iterations = []
        for turb_timescale in timescales:
            result = collect_statistics({"noise_timescale": turb_timescale * noise_multiplier,    # AR time of random noise
                                         "turb_timescale": turb_timescale,     # Flux relaxation time
                                         "damping_timescale": turb_timescale * damping_multiplier,  # Flux damping timescale
                                         "noise_amplitude": noise_amplitude,    # Amplitude of AR(1) noise
                                         "noise_scalelength": noise_scalelength,   # Spatial scale length (cells)
                                         "EWMAParamTurbFlux": alpha_flux,
                                         "EWMAParamProfile": alpha_profile,
                                         "thetaParams": {"Dmin": 1e-5, "Dmax": 1e13, "dpdxThreshold": 10},
                                         "tol": tolerance,
                                         "maxIterations": 1000,
                                         "plot_convergence": False},
                                        num_samples = num_samples)

            # Plot error history on figure 1 (for the last sample)
            ax1.plot(result["err_history"], label=str(turb_timescale))

            mean_iterations.append(result["mean_iterations"])
            max_iterations.append(result["max_iterations"])
            min_iterations.append(result["min_iterations"])

        timescales = np.array(timescales)

        # Plot iterations vs timescale on figure 2
        ax2.plot(
            1./timescales,
            mean_iterations,
            "-o",
            label=r"$\tau_{{damp}} / \tau_{{turb}} = {}, \tau_{{noise}} / \tau_{{turb}} = {}$".format(damping_multiplier, noise_multiplier),
        )

        if num_samples > 1:
            print(mean_iterations, max_iterations, min_iterations)
            ax2.fill_between(1./timescales, min_iterations, max_iterations, alpha=0.5)

        #ax2.axvline(1.0 / damping_multiplier, linestyle="--", color="k")
        #ax2.axvline(1.0 / noise_multiplier, linestyle=":", color='k')

        # Plot total turbulence run time on figure 3
        ax3.plot(
            1. / timescales,
            np.array(mean_iterations) / timescales,
            "-o",
            label=r"$\tau_{{damp}} / \tau_{{turb}} = {}, \tau_{{noise}} / \tau_{{turb}} = {}$".format(damping_multiplier, noise_multiplier),
        )

        if num_samples > 1:
            ax3.fill_between(1. / timescales,
                             np.array(min_iterations) / timescales,
                             np.array(max_iterations) / timescales, alpha=0.5)

# Plot of error against iteration
ax1.set_yscale("log")
ax1.set_xlabel("iteration number")
ax1.set_ylabel("rms error")
ax1.legend()

fig1.savefig("residual_history.pdf")
fig1.savefig("residual_history.png")

# Plot of iteration count against timescale
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Simulation run time / $\tau_{{turb}}$")
ax2.set_ylabel("Iterations required")
ax2.legend()

fig2.savefig("iteration_count.pdf")
fig2.savefig("iteration_count.png")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"Simulation run time / $\tau_{{turb}}$")
ax3.set_ylabel("Total run time required")
ax3.legend()

fig3.savefig("run_time.pdf")
fig3.savefig("run_time.png")

plt.show()


# ==============================================================================
#  Settings
# ==============================================================================

turb_timescale = 1.0
damping_multiplier = 1.0
noise_multiplier = 1.0

tolerance = 1e-2
alpha_ps = np.logspace(-2, 0, num=10)
alpha_ds = np.logspace(-2, 0, num=10)

noise_amplitude = 0.01  # Amplitude of noise to add
noise_scalelength = 10.0  # Spatial scale length (number of cells)

num_samples = 1  # Number of repeated solves, to gather statistics

# ==============================================================================
#  MAIN STARTS HERE
# ==============================================================================
tlog.setup()

iterations = np.zeros((len(alpha_ps), len(alpha_ds)))

for i, alpha_p in enumerate(alpha_ps):
    for j, alpha_d in enumerate(alpha_ds):
        result = collect_statistics({"noise_timescale": turb_timescale * noise_multiplier,    # AR time of random noise
                                         "turb_timescale": turb_timescale,     # Flux relaxation time
                                         "damping_timescale": turb_timescale * damping_multiplier,  # Flux damping timescale
                                         "noise_amplitude": noise_amplitude,    # Amplitude of AR(1) noise
                                         "noise_scalelength": noise_scalelength,   # Spatial scale length (cells)
                                         "EWMAParamTurbFlux": alpha_d,
                                         "EWMAParamProfile": alpha_p,
                                         "thetaParams": {"Dmin": 1e-5, "Dmax": 1e13, "dpdxThreshold": 10},
                                         "tol": tolerance,
                                         "maxIterations": 1000,
                                         "plot_convergence": False},
                                        num_samples = num_samples)
        iterations[i,j] = result["mean_iterations"]

plt.contourf(np.log10(alpha_ds), np.log10(alpha_ps), np.log10(iterations), 50)
plt.xlabel(r'Diffusion smoothing $\log_{10} \alpha_D$')
plt.ylabel(r'Profile smoothing $\log_{10} \alpha_p$')
plt.title("$\log_{10}$ iterations")
plt.colorbar()
plt.savefig("iterations_contour.png")
plt.show()
