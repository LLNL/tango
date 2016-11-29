"""Parameters module

Unified place to set defaults and provide input parameters and profiles
"""

from __future__ import division, absolute_import
import numpy as np

def get_default_parameters():
    """Default parameters for GENE/Tango runs
    
    Outputs:
      rTango            Tango radial grid, physical radius r, measured in m (array)
      rGene             GENE radial grid, physical radius r, measured in m (array)
      minorRadius       minor radius a.  Measured in m (scalar)
      majorRadius       major radius R0. Measured in m (scalar)
      Vprime            dV/dr, where V = volume inside flux surface r.  measured in m^2 (array)
      gradPsiSq         <|grad psi|^2>.  Equal to 1 for circular geometry.  (array)
      B0                magnetic field parameter [see Lapillone 2009].  Measured in Tesla (scalar)
      ionMass           ion mass, measured in proton mass (scalar)
      ionCharge         ion charge, measured in electron charge (scalar)
      densityProfile    plasma density profile, measured in m^-3 (array)
      Bref              GENE reference magnetic field, measured in Tesla.  Typically equal to B0 (scalar)
      Lref              GENE reference length, measured in m (scalar)
    """
    numRadialPtsTango = 100
    numRadialPtsGene = 80
    rhoTango = np.linspace(0.1, 0.9, numRadialPtsTango)      # rho = r/a
    rhoGene = np.linspace(0.2, 0.8, numRadialPtsGene)
    minorRadius = 0.65
    majorRadius = 1.65
    rTango = rhoTango * minorRadius    # physical radius r
    rGene = rhoGene * minorRadius
    Vprime = calculate_analytic_Vprime(rTango, majorRadius)
    gradPsiSq = np.ones_like(rTango)
    B0 = 2.5
    ionMass = 1
    ionCharge = 1
    densityProfile = 1e19 * np.ones_like(rTango)    
    Bref = B0
    Lref = 1.65
    return (rTango, rGene, minorRadius, majorRadius, Vprime, gradPsiSq, B0, ionMass, ionCharge, densityProfile, Bref, Lref)

def calculate_analytic_Vprime(r, R0):
    """Calculate Vprime = dV/dr for analytic magnetic geometry with circular surfaces.
    
      Volume within a flux surface labeled by r:
              V(r) = pi * r^2  * 2 * pi * R0
      Hence,
              dV/dr = 4 * pi^2 * R0 * r
    """
    Vprime = (2 * np.pi)**2 * R0 * r
    return Vprime

def analytic_safety_factor(r, a, R0):
    """Compute the analytic safety factor for standard GENE runs in the analytic magnetic geometry
    with circular surfaces.
    
       q(r) = qbar(r) / sqrt(1 - r^2/R0^2),
    where
       qbar(r) = 0.854 + 2.184 * (r/a)^2
    
    Inputs:
      r         radial coordinate equal to physical radius (array)
      a         minor radius.  measured in same units as r (scalar)
      R0        major radius.  measured in same units as r (scalar)
    
    Outputs:
      q         safety factor as a function of r (array)
      
    References: Lapillone (2009)
    """
    assert np.all(r < R0), "r must be smaller than R0 everywhere"
    qbar = analytic_safety_factor_qbar(r, a)
    q = qbar / np.sqrt(1 - (r/R0)**2)
    return q
    
def analytic_safety_factor_qbar(r, a):
    """Compute the qbar profile for standard GENE runs in the analytic magnetic geometry with
    circular surfaces.
            qbar(r) = 0.854 + 2.184 * (r/a)^2
    
    Inputs:
      r         radial coordinate (array)
      a         minor radius.  Measured in same units as r (scalar)
    Outputs:
      qbar
    """
    qbar = 0.854 + 2.184 * (r/a)**2
    return qbar