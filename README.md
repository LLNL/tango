# tango
Tango is a transport solver intended for coupling with codes that return turbulent fluxes.  In many physics problems, a separation of timescales exists between the timescale of turbulence and the timescale for evolution of mean, macroscopic quantities such as density and temperature in response to turbulent fluxes.  In the numerical simulation of such scenarios, it is desirable to efficiently exploit the separation of timescales.  Furthermore, because in general the turbulent fluxes depend nonlinearly on the macroscopic quantities and their derivatives, certain numerical instabilities arise and special methods are required to handle them.

For the kind of situation just described, Tango solves a one-dimensional transport equation.  That is, the macroscopic quantities are assumed to depend on a single spatial variable.  Tango is designed to accept fluxes from a separate turbulence simulation code in order to evolve the transport equation on a longer timescale.  Tango will communicate the updated macroscopic quantities back to the turbulence simulation code and proceed with solution.  The fundamental ideas are laid out in [Shestakov et al., J. Comp. Phys. (2003)](http://www.sciencedirect.com/science/article/pii/S0021999102000633).

## Coupling with gyrokinetic turbulence
Tango is designed to couple with the gyrokinetic turbulence simulation code [GENE](http://genecode.org/) through a Python-Fortran interface.

## Saving data to files
Tango has a few ways for saving data to file.

1. Tango saves data to a file at the end of the program (this may change in the future to save to a file occasionally during operation in case of program failure).  For each timestep, two files are saved: one containing arrays relevant to the entire timestep,  and another containing the data for all iterations of the 1D arrays that changes on each iteration within the timestep.

The user specifies a basename and the files are saved with a suffix `#_timestep` and `#_iterations`, where `#` corresponds to the timestep number.  In other words, if the basename were `data`, then the two saved data files for the second timestep would be `data2_timestep.npz` and `data2_iterations.npz`.   The files are saved in the `.npz` format, which is numpy's default format for saving multiple arrays.

Instead of saving all possible arrays, the user specifies which arrays are saved to file.  This allows to save disk space while still allowing flexibility in case the full storage of all arrays is desired.

2. Checkpointing.  (For now, see source code for description)

3. Intermittent history saving on the fly.  In case of a fatal crash in the middle of operation which might occur in the turbulence code, Tango also can save data to disk in the course of operation, using the `TangoHistoryHandler`.  (For now, see source code for description)


## Source Code Style Convention
Capitalization and underscores:
* `ClassName`: Camel Case with each word capitalized
* `function_name` or `method_name`: lower case, with underscore between words
* `variableName` or `objectName`: Camel case with each word after the first capitalized
At times, there are exceptions to this general rule.  Typically this will occur in variables closely related to mathematical notation, where capitalization to conform to mathematical convention is appropriate.  Rarely, underscores are used outside of function names.


## History
* v0.20: GENE interface code included
* v0.15: More modular design with examples and tests
* v0.1: A minimal testcase is working for an analytically prescribed flux
