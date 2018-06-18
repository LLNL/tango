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
    
def write_scenario3():
    (xOvera, xOverRhoRef, T, n) = scenario3()
    write('profiles_ions', xOvera, xOverRhoRef, T, n)    

def write_ke_scenario1():
    (xOvera, xOverRhoRef, Ti, ni, Te, ne) = ke_scenario1()
    write('profiles_ions', xOvera, xOverRhoRef, Ti, ni)
    write('profiles_electrons', xOvera, xOverRhoRef, Te, ne)
    
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

def compute_x_over_rhoref(xOvera, rhoStar):
    """
    x/rhoref = x/a * a/rhoRef = (x/a) / rhostar
    """
    xOverRhoRef = xOvera / rhoStar
    return xOverRhoRef
    
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
    """Scenario 2 for x/a, x/rhoref, T, n"""
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

def scenario3():
    """Scenario 3 for x/a, x/rhoref, T, n.  CHEASE, DIII-D-like run"""
    numRadialPts = 180
    #minorRadius = 0.741206  # a, in m
    #majorRadius = 1.68  # R0, in m
    #inverseAspectRatio = minorRadius / majorRadius
    
    rhoMin = 0.1
    rhoMax = 0.9
    rho = np.linspace(rhoMin, rhoMax, numRadialPts)   # rho = x/a = r/a
    xOvera = rho
    rhoStar = 0.0034194220332098143
    xOverRhoRef = xOvera / rhoStar
    
    # Density profile in 10^19 m^-3
    n = scenario3_densityprofile(rho)
    
    # Ion temperature in keV
    T = scenario3_temperatureprofile(rho)
    
    return (xOvera, xOverRhoRef, T, n)

def scenario3_densityprofile(rho):
    """Density profile, in 10^19 m^-3, for scenario 3.
    Input: rho = x/a
    """
    # Machine parameters
    minorRadius = 0.741206  # a, in m
    majorRadius = 1.68  # R0, in m
    inverseAspectRatio = minorRadius / majorRadius
    
    # Density profile
    kappa_n = 2.22
    delta_n = 0.5
    Delta_n = 0.1
    rho0_n = 0.5
    n0 = 3.3  # in 10^19 m^-3
    n = base_profile_shape(rho, kappa_n, delta_n, Delta_n, rho0_n, n0, inverseAspectRatio)
    return n

def scenario3_temperatureprofile(rho):
    """Temperature profile, in keV, for scenario 3.
    Input: rho = x/a
    """
    # Machine parameters
    minorRadius = 0.741206  # a, in m
    majorRadius = 1.68  # R0, in m
    inverseAspectRatio = minorRadius / majorRadius
    
    # Ion temperature in keV
    kappa_T = 6.96
    delta_T = 0.8
    Delta_T = 0.1
    rho0_T = 0.5
    T0 = 2.3
    T = base_profile_shape(rho, kappa_T, delta_T, Delta_T, rho0_T, T0, inverseAspectRatio)
    return T
    
def ke_scenario1():
    """Kinetic Electrons: Scenario 1 for x/a, x/rhoref, Ti, ni, Te, ne"""
    numRadialPts = 360
    minorRadius = 0.594  # a
    majorRadius = 1.65  # R0
    inverseAspectRatio = minorRadius / majorRadius
    
    rhoMin = 0.1
    rhoMax = 0.9
    rho = np.linspace(rhoMin, rhoMax, numRadialPts)   # rho = x/a = r/a
    xOvera = rho
    rhoStar = 0.0067478807018899684
    xOverRhoRef = xOvera / rhoStar
    
    # Density profile
    kappa_n = 2.4
    delta_n = 0.5
    Delta_n = 0.1
    rho0_n = 0.5
    n0 = 3  # in 10^19 m^-3
    ni = base_profile_shape(rho, kappa_n, delta_n, Delta_n, rho0_n, n0, inverseAspectRatio)
    ne = ni
    
    # Ion temperature
    kappa_i = 6.9
    delta_i = 0.9
    Delta_i = 0.1
    rho0_i = 0.5
    Ti0 = 1
    Ti = base_profile_shape(rho, kappa_i, delta_i, Delta_i, rho0_i, Ti0, inverseAspectRatio)
    
    # Electron temperature
    kappa_e = 7.3
    delta_e = 0.9
    Delta_e = 0.1
    rho0_e = 0.5
    Te0 = 1.
    Te = base_profile_shape(rho, kappa_e, delta_e, Delta_e, rho0_e, Te0, inverseAspectRatio)
    
    return (xOvera, xOverRhoRef, Ti, ni, Te, ne)

def base_profile_shape(rho, kappa, delta, Delta, rho0, Y0, inverseAspectRatio):
    """Base profile shape that is flat near rho=0, then decreases
      
    Inputs:
      rho                   normalized radial coordinate rho=r/a (array)
      kappa                 constant that sets the characteristic gradient scale length (scalar) 
      delta                 constant that affects the shape of the profile (scalar)
      Delta                 constant that affects the shape of the profile (scalar)
      rho0                  constant that affects the location of the profile (scalar)
      Y0                    value that determines Y(rho0), the value of the profile at rho0 (scalar)
      inverseAspectRatio    constant representing the inverse aspect ratio a/R0 (scalar)
    Outputs:
      Y                     profile (in same units As Y0 is given) (array)
    """
    rhominus = rho - rho0 + delta/2
    Y = Y0 * np.exp( -kappa * inverseAspectRatio * (rho - rho0 - Delta * (np.tanh(rhominus/Delta) - np.tanh(delta/2/Delta))));
    
    # find the index closest to the location rho_a = rho0 - delta/2
    ind = int(np.abs(rho - (rho0 - delta/2)).argmin())
    # set Y equal to a constant for Y < rho_a
    ind2 = (rho < (rho0-delta/2));
    Y[ind2] = Y[ind];
    return Y