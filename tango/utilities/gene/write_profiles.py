"""
write_profiles

Python interface to write out profiles in the format GENE expects for an input profiles_ions file.  GENE's format for this file
is that it should be named 'profiles_ions', with two header lines, followed by data in 4 columns.  E.g.,


#  x/a      x/rhoref     T     n
#
   data.....................................
   data....................................

Temperature data should be in keV and density data should be in 10^19 m^-3
"""

from __future__ import division, absolute_import
import numpy as np

def write_scenario1():
    (xOvera, xOverRhoRef, T, n) = scenario1()
    write('profiles_ions', xOvera, xOverRhoRef, T, n)
    
def write_scenario2():
    (xOvera, xOverRhoRef, T, n) = scenario2()
    write('profiles_ions', xOvera, xOverRhoRef, T, n)

def write(filename, xOvera, xOverRhoRef, T, n):
    """Write the profiles_ions file given the input arrays
    
    Inputs:
      filename          path for output file (string)
      xOvera            radial grid in x/a, dimensionless (array)
      xOverRhoRef       radial grid in x/rhoref, dimensionless (array)
      T                 temperature profile in keV (array)
      n                 density profile in 10^19 m^-3 (array)
        
    Outputs:        
    """
    header = 'x/a      x/rhoref     T     n' + '\n' + ''  # hash signs are automatically added by default to all header lines
    np.savetxt(filename, np.transpose([xOvera, xOverRhoRef, T, n]), header=header)
    
    
def scenario1():
    """Scenario 1 for x/a, x/rhoref, T, n"""
    numRadialPts = 120
    minorRadius = 0.594  # a
    majorRadius = 1.65  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    rhoMin = 0.1
    rhoMax = 0.9
    rho0 = 0.5
    rho = np.linspace(rhoMin, rhoMax, numRadialPts)   # rho = x/a = r/a
    xOvera = rho
    rhoStar = 0.0067478807018899684
    xOverRhoRef = xOvera / rhoStar
    
    # density profile
    n0 = 3.3;     # in 10^19 m^-3
    kappa_n = 2.2;  # R0 / Ln
    #n = n0 * np.exp( -kappa_n * inverseAspectRatio * (xOvera - rho0));
    
    deltar = 0.5
    rhominus = rho - rho0 + deltar/2
    deltan=0.1
    n = n0 * np.exp( -kappa_n * inverseAspectRatio * (rho - rho0 - deltan * (np.tanh(rhominus/deltan) - np.tanh(deltar/2/deltan))))
    
    # set n to a constant for rho < rho0-deltar/2
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2))
    n[ind2] = n[ind];
    
    
    # temperature profile
    kappa_T = 6.96
    deltar = 0.9
    rhominus = rho - rho0 + deltar/2
    deltaT = 0.1
    
    T0 = 1  # 1 keV
    T = T0 * np.exp( -kappa_T * inverseAspectRatio * (rho - rho0 - deltaT * (np.tanh(rhominus/deltaT) - np.tanh(deltar/2/deltaT))));
    
    # set T to a constant for rho < rho0-deltar/2
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2))
    T[ind2] = T[ind];
    
    return (xOvera, xOverRhoRef, T, n)
    
    
def scenario2():
    """Scenario 1 for x/a, x/rhoref, T, n
    
    Note: x/rhoref = x/a * a/rhoRef = (x/a) / rhostar
    """
    numRadialPts = 240
    minorRadius = 1  # a
    majorRadius = 3  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    rhoMin = 0.1
    rhoMax = 0.9
    rho0 = 0.5
    rho = np.linspace(rhoMin, rhoMax, numRadialPts)   # rho = x/a = r/a
    xOvera = rho
    rhoStar = 0.0034194220332098143
    xOverRhoRef = xOvera / rhoStar
    
    # density profile
    n0 = 3.3;     # in 10^19 m^-3
    kappa_n = 2.3;  # R0 / Ln
    #n = n0 * np.exp( -kappa_n * inverseAspectRatio * (xOvera - rho0));
    
    deltar = 0.5
    rhominus = rho - rho0 + deltar/2
    deltan=0.1
    n = n0 * np.exp( -kappa_n * inverseAspectRatio * (rho - rho0 - deltan * (np.tanh(rhominus/deltan) - np.tanh(deltar/2/deltan))))
    
    # set n to a constant for rho < rho0-deltar/2
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2))
    n[ind2] = n[ind];
    
    
    # temperature profile
    kappa_T = 6.96
    deltar = 0.9
    rhominus = rho - rho0 + deltar/2
    deltaT = 0.1
    
    T0 = 2.8  # 1 keV
    T = T0 * np.exp( -kappa_T * inverseAspectRatio * (rho - rho0 - deltaT * (np.tanh(rhominus/deltaT) - np.tanh(deltar/2/deltaT))));
    
    # set T to a constant for rho < rho0-deltar/2
    ind = int(np.abs(rho - (rho0 - deltar/2)).argmin())
    ind2 = (rho < (rho0-deltar/2))
    T[ind2] = T[ind];
    
    return (xOvera, xOverRhoRef, T, n)    