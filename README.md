# tango
Tango is a transport solver intended for coupling with codes that return turbulent fluxes.  In many physics problems, a separation of timescales exists between the timescale of turbulence and the timescale for evolution of mean, macroscopic quantities such as density and temperature in response to turbulent fluxes.  In the numerical simulation of such scenarios, it is desirable to efficiently exploit the separation of timescales.  Furthermore, because in general the turbulent fluxes depend nonlinearly on the macroscopic quantities and their derivatives, certain numerical instabilities arise and special methods are required to handle them.

For the kind of situation just described, Tango solves a one-dimensional transport equation.  That is, the macroscopic quantities are assumed to depend on a single spatial variable.  Tango is designed to accept fluxes from a separate turbulence simulation code in order to evolve the transport equation on a longer timescale.  Tango will communicate the updated macroscopic quantities back to the turbulence simulation code and proceed with solution.  The fundamental ideas are laid out in [Shestakov et al., J. Comp. Phys. (2003)](http://www.sciencedirect.com/science/article/pii/S0021999102000633).

## History
* v0.1: A minimal testcase is working for an analytically prescribed flux
