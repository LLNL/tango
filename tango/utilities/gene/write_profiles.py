"""
write_profiles

Python interface to write out profiles in the format GENE expects for an input profiles_ions file.  GENE's format for this file
is that it should be named 'profiles_ions', with two header lines, followed by data in 4 columns.  E.g.,

#  x/a      x/rhoref     T (in keV)     n (in 10^19 m^-3)
#
   data.....................................
   data....................................
   
"""

from __future__ import division, absolute_import
import numpy as np

def write_default1():
    (xOvera, xOverRhoRef, T, n) = default1()
    write('profiles_ions', xOvera, xOverRhoRef, T, n)

def write(filename, xOvera, xOverRhoRef, T, n):
    """Write the profiles_ions file given the input arrays
    
    Inputs:
      filename          path for output file (string)
      xOvera            radial grid in x/a, dimensionless (array)
      xOverRhoRef       radial grid in x/rhoref, dimensionless (array)
      T                 temperature profile in keV (array)
      n                 density profile in keV (array)
        
    Outputs:        
    """
    header = 'x/a      x/rhoref     T     n' + '\n' + ''  # hash signs are automatically added by default to all header lines
    np.savetxt(filename, np.transpose([xOvera, xOverRhoRef, T, n]), header=header)
    
def default1():
    """Some defaults for x/a, x/rhoref, T, n
    
    Note: x/rhoref = x/a * a/rhoRef = (x/a) / rhostar
    """
    numRadialPts = 144
    minorRadius = 0.28  # a
    majorRadius = 0.71  # R0
    
    rhoMin = 0.1
    rhoMax = 0.9
    xOvera = np.linspace(rhoMin, rhoMax, numRadialPts)
    
    rhoStar = 0.00565316008562
    xOverRhoRef = xOvera / rhoStar

    rmid = 0.6;

    Tmax = 2;       # in keV
    Tmin = 1;
    rWidth = 0.13;
    T = (Tmax + Tmin)/2 - (Tmax - Tmin)/2 * np.tanh((xOvera - rmid)/ rWidth);
    
    n0 = 1;     # in 10^19 m^-3
    kappa_n = 2.2;  # R0 / Ln
    invAspectRatio = minorRadius / majorRadius
    n = n0 * np.exp( -kappa_n * invAspectRatio * (xOvera - rmid));
    
    return (xOvera, xOverRhoRef, T, n)