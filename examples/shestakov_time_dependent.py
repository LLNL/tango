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
from tango.extras.noisyflux import NoisyFlux
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


def initialize_parameters():
    maxIterations = 5000
    thetaParams = {"Dmin": 1e-5, "Dmax": 1e13, "dpdxThreshold": 10}
    EWMAParamTurbFlux = 0.3
    EWMAParamProfile = 1
    lmParams = {
        "EWMAParamTurbFlux": EWMAParamTurbFlux,
        "EWMAParamProfile": EWMAParamProfile,
        "thetaParams": thetaParams,
    }
    tol = 1e-2  # tol for convergence... reached when a certain error < tol
    return (maxIterations, lmParams, tol)


class ComputeAllH(object):
    def __init__(self):
        pass

    def __call__(self, t, x, profiles, HCoeffsTurb):
        # n = profiles['default']
        # Define the contributions to the H coefficients for the Shestakov Problem
        H1 = np.ones_like(x)
        H7 = shestakov_nonlinear_diffusion.H7contrib_Source(x)

        HCoeffs = tango.multifield.HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb

        return HCoeffs


# ==============================================================================
#  Settings
# ==============================================================================

timescales = [1e-3, 0.1, 0.2, 0.5, 1, 2, 5, 10]
damping_multipliers = [1.0, 2.0, 5.0]

noise_amplitude = 1e-3  # Amplitude of noise to add
noise_scalelength = 10.0  # Spatial scale length (number of cells)

num_samples = 20  # Number of repeated solves, to gather statistics

# ==============================================================================
#  MAIN STARTS HERE
# ==============================================================================
tlog.setup()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for damping_multiplier in damping_multipliers:
    mean_iterations = []
    max_iterations = []
    min_iterations = []
    for turb_timescale in timescales:
        its = []  # Iterations for these settings
        for sample in range(num_samples):
            tlog.info("Initializing...")
            L, N, dx, x, nL, n = initialize_shestakov_problem()
            maxIterations, lmParams, tol = initialize_parameters()
            fluxModel = NoisyFlux(
                FluxDoubleRelaxation(
                    shestakov_nonlinear_diffusion.AnalyticFluxModel(dx),
                    turb_timescale,
                    damping_multiplier * turb_timescale,
                ),
                noise_amplitude,
                noise_scalelength,
                1.0,
            )

            label = "n"
            turbHandler = tango.lodestro_method.TurbulenceHandler(dx, x, fluxModel)

            compute_all_H_density = ComputeAllH()
            lodestroMethod = tango.lodestro_method.lm(
                lmParams["EWMAParamTurbFlux"],
                lmParams["EWMAParamProfile"],
                lmParams["thetaParams"],
            )
            field0 = tango.multifield.Field(
                label=label,
                rightBC=nL,
                profile_mminus1=n,
                compute_all_H=compute_all_H_density,
                lodestroMethod=lodestroMethod,
            )
            fields = [field0]
            tango.multifield.check_fields_initialize(fields)

            compute_all_H_all_fields = tango.multifield.ComputeAllHAllFields(
                fields, turbHandler
            )

            tArray = np.array([0, 1e4])  # specify the timesteps to be used.

            solver = tango.solver.Solver(
                L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields
            )

            tlog.info("Initialization complete.")
            tlog.info("Entering main time loop...")

            while solver.ok:
                # Implicit time advance: iterate to solve the nonlinear equation!
                solver.take_timestep()

            n = solver.profiles[label]  # finished solution
            nss = shestakov_nonlinear_diffusion.steady_state_solution(x, nL)

            solutionResidual = (n - nss) / np.max(np.abs(nss))
            solutionRmsError = np.sqrt(1 / len(n) * np.sum(solutionResidual ** 2))

            if solver.reachedEnd == True:
                print("The solution has been reached successfully.")
                print(
                    "Error compared to analytic steady state solution is %f"
                    % (solutionRmsError)
                )
            else:
                print("The solver failed for some reason.")
                print(
                    "Error at end compared to analytic steady state solution is %f"
                    % (solutionRmsError)
                )

            its.append(len(solver.errHistoryFinal))

        # Plot error history on figure 1 (for the last sample)
        ax1.plot(solver.errHistoryFinal, label=str(turb_timescale))

        its = np.array(its)
        mean_iterations.append(np.mean(its))
        max_iterations.append(np.amax(its))
        min_iterations.append(np.amin(its))
        print(its)

    # Plot iterations vs timescale on figure 2
    ax2.plot(
        timescales,
        mean_iterations,
        "-o",
        label=r"$\tau_{{damp}} / \tau_{{turb}} = {}$".format(damping_multiplier),
    )

    if num_samples > 1:
        print(mean_iterations, max_iterations, min_iterations)
        ax2.fill_between(timescales, min_iterations, max_iterations, alpha=0.5)

    ax2.axvline(1.0 / damping_multiplier, linestyle="--", color="k")

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
ax2.set_xlabel("Turbulence relaxation time")
ax2.set_ylabel("Iterations required")
ax2.legend()

fig2.savefig("iteration_count.pdf")
fig2.savefig("iteration_count.png")

plt.show()
