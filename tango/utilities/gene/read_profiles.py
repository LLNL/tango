"""
read_profiles

Python interface to read profiles in the GENE format, e.g., for a profiles_ions file.  GENE's format for this file
is that it should be named 'profiles_<spec>', where <spec> might be 'ions' or 'elec' with two header lines, followed by data in 4 or 6 columns.  E.g.,


#  x/a      x/rhoref     T     n     omt     omn
#
   data.....................................
   data....................................

Temperature (T) is in keV.  Density (n) is in 10^19 m^-3.

The normalized temperature gradient omt = -(R0/T) dT/dx  is dimensionless, as is the
normalized density gradient omn = -(R0/n) dn/dx.
"""


from __future__ import division, absolute_import
from collections import namedtuple
import numpy as np

ProfileData = namedtuple('ProfileData', 'rho T n')
ProfileDataOmt = namedtuple('ProfileDataOmt', 'rho T n omt omn')

def read_profiles(filename):
    """Read in GENE's profile data.
    
    rho = x/a.  Typically, for EFIT/CHEASE geometry, GENE uses x = rhotor
    """
    (rho, xOverRhoRef, T, n) = np.loadtxt(filename, unpack=True)
    profile = ProfileData(rho, T, n)
    return profile

def read_profiles_omt(filename):
    """Read in GENE's profile data, including normalized gradients omt, omn.
    
    rho = x/a.  Typically, for EFIT/CHEASE geometry, GENE uses x = rhotor
    """
    (rho, xOverRhoRef, T, n, omt, omn) = np.loadtxt(filename, unpack=True)
    profile = ProfileDataOmt(rho, T, n, omt, omn)
    return profile
