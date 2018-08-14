"""example_create_flux_seed
from a fluxprof2D file, output by GENE diagnostics
"""

from __future__ import division
import numpy as np

from tango.utilities.gene import read_fluxprof2D
from tango import genecomm_unitconversion

filename = '/path/to/fluxprof2Dions.dat'
speciesName = 'ions'

fluxData = read_fluxprof2D.ProfileFileData(filename, speciesName)
# if one wants to read electromagnetic
#  fluxData = read_fluxprof2D.ProfileFileData(filename, speciesName, readElectromagnetic=True)

rho = fluxData.rho  # radial grid, rho = r/a
time = fluxData.time    # time grid
qhat = fluxData.heatFluxTurb  # heat flux qhat = <Qhat dot grad psi>, 2D array, first dim = time, second dim = space
ghat = fluxData.particleFluxTurb # particle flux ghat = <Gammahat dot grad psi>, 2D array, time x space

# set some constants of the simulation: must be updated properly!
a = 1
R0 = 3
rhoStar = 1/292.4
Vprime = 4 * np.pi**2 * R0 * a * rho

Bref = 2.5
Lref = 3
mref = 2
Tref = 3.5
nref = 3.3

# convert to SI and take the time average
qhatSI = genecomm_unitconversion.heatflux_gene_to_SI(qhat, nref, Tref, mref, Bref, Lref)
qhat_timeavg = np.trapz(qhatSI, time, axis=0) / (time[-1] - time[0])

# save the file.
#  Print only the flux value, not the radial coordinates.
#  Transpose is to print only one value per line.
np.savetxt('heat_flux_seed_ions', np.transpose([qhat_timeavg]))

# if particle flux is also desired:
# ghatSI = genecomm_unitconversion.particleflux_gene_to_SI(ghat, nref, Tref, mref, Bref, Lref)
# ghat_timeavg = np.trapz(ghatSI, time, axis=0) / (time[-1] - time[0])
# np.savetxt('particle_flux_seed_ions', np.transpose([ghat_timeavg]))

# if reading electron file is desired:
# filename = '/path/to/fluxprof2Delectrons.dat'
# speciesName = 'elec'
