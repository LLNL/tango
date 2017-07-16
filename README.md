# Tango
Tango is a transport solver intended for coupling with codes that return turbulent fluxes.  In many physics problems, including magnetically confined fusion plasmas such as a tokamak, a separation of timescales exists between the timescale of turbulence and the timescale for evolution of mean, macroscopic quantities such as density and temperature in response to turbulent fluxes.  In the numerical simulation of such scenarios, it is desirable to efficiently exploit the separation of timescales.  Furthermore, because in general the turbulent fluxes depend nonlinearly on the macroscopic quantities and their derivatives, certain numerical instabilities arise and special methods are required to handle them.

For the kind of situation just described, Tango solves a one-dimensional transport equation.  That is, the macroscopic quantities are assumed to depend on a single spatial variable.  Tango is designed to accept fluxes from a separate turbulence simulation code in order to evolve the transport equation on a longer timescale.  Tango will communicate the updated macroscopic quantities back to the turbulence simulation code and proceed with solution.  The fundamental ideas are laid out in [Shestakov et al., J. Comp. Phys. (2003)](http://www.sciencedirect.com/science/article/pii/S0021999102000633).

**Note**: Tango is under active development.  New features and capabilities will be coming at some point in the future and these will most likely be incompatible with the current code.

## Coupling with gyrokinetic turbulence
Tango is designed to couple with the gyrokinetic turbulence simulation code [GENE](http://genecode.org/) through a Python-Fortran interface.  In principle, Tango could couple to other simulation codes if suitable interfaces are available.

More details will be presented in a forthcoming journal article.

## Contact
For questions and inquiries, including regarding possible collaborations, please contact the author, [Jeff Parker](https://pls.llnl.gov/people/staff-bios/physics/parker-j), at parker68@llnl.gov.