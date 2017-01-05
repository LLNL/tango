"""
genecomm_unitconversion

Provides conversions between GENE's normalized units and SI units.

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np

# physical constants with module-wide scope.  Given in SI units
e = 1.60217662e-19          # electron charge
mp = 1.6726219e-27          # proton mass

##### Functions converting quantities in SI to GENE's normalized units #####

def density_SI_to_gene(n_SI):
    """Convert a density in SI to a density in GENE's normalized units for input to GENE in libgene_tango
    
    Inputs:
      n_SI         density in SI units (array)
    Outputs:
      nHatGene     density in GENE normalized units of 10^19 m^-3 (array)
    """
    nrefGeneFactor = 1e19
    nHatGene = n_SI / nrefGeneFactor
    return nHatGene
    
def temperature_SI_to_gene(T_SI):
    """Convert a temperature in SI to a temperature in GENE's normalized units for input to GENE in libgene_tango
    
    Inputs:
      T_SI         tepmerature in SI energy units (array)
    Outputs:
      THatGene     temperature in GENE normalized units of keV (array)
    """
    TrefGeneFactor = 1000 * e
    THatGene = T_SI / TrefGeneFactor
    return THatGene

def radius_SI_to_libgenetango_input(psi, a):
    """Convert radius in SI to the input length that libgene_tango expects for analytic circular geoemtry.
    libgene_tango expects radius to be measured in r/a, where a is the minor radius.
    
    Inputs:
      psi               input radial coordinate psi=r (array)
      a                 minor radius (scalar)
    Outputs:
      normalizedRadius  r/a (array)
    """
    normalizedRadius = psi/a
    return normalizedRadius

def dVdx_gene_to_SI(dVdxHat, Lref):
    """Convert GENE's output dVdxHat into SI units.
    GENE outputs dVdx_hat = dV_hat / dx_hat, where V_hat = V/Lref^3 and x_hat = r/Lref, where x=r.  Hence, one has
      dV_hat / dx_hat = (1/Lref^2) dV/dr
    
    Note that for analytic circular geometry, one can calculate V(r) and dV/dr analytically:
       V(r) = 2 * pi^2 * R0 * r^2
       dV/dr = 4 * pi^2 * R0 * r
    
    Inputs:
      dVdxHat        dVdx in GENE normalized units (array)
      Lref           GENE's reference length Lref, measured in meters (scalar)
    Outputs:
      dVdxSI         dVdx in SI units (array)
    """
    dVdxSI = dVdxHat * Lref**2
    return dVdxSI

def heatflux_gene_to_SI(heatFluxHat, nref, Tref, mref, Bref, Lref):
    """Convert GENE's output radial heat flux into SI units.
    GENE outputs avgHeatFluxHat = <Qhat dot gradhat xhat> = <Q/Qref dot Lref grad x/Lref> = <Q dot grad x> / Qref.    
    For circular geometry, x=r
    
    Inputs:
      heatFluxHat       GENE's output, <Qhat dot gradhat xhat> (array)
      nref              measured in 10^19 m^-3
      Tref              measured in keV (scalar)
      mref              measured in proton masses (scalar)
      Bref              measured in Tesla (scalar)
      Lref              measured in m (scalar)
    Outputs:
      heatFluxSI       <Q dot grad x> (array)
    """
    Qref = Q_ref(nref, Tref, mref, Bref, Lref)
    heatFluxSI = heatFluxHat * Qref
    return heatFluxSI


##### Functions calculating derived reference values for GENE #####
def omega_ref(Bref, mref):
    """Compute reference gyrofrequency Omegaref = e*Bref/mref
    Inputs:
      Bref    measured in Tesla (scalar)
      mref    measured in proton masses (scalar)
    Outputs:
      Omegaref measured in s^-1 (scalar)
    """
    mref *= mp  # convert to SI units
    Omegaref = e * Bref / mref
    return Omegaref
    
def c_ref(Tref, mref):
    """Compute reference velocity cref = sqrt(Tref/mref)
    Inputs:
      Tref      measured in keV (scalar)
      mref      measured in proton masses (scalar)
    Outputs:
      cref      measured in m/s (scalar)
    """
    # convert to SI units
    mref *= mp
    TrefGeneFactor = 1000 * e
    Tref *= TrefGeneFactor
    
    cref = np.sqrt(Tref / mref)
    return cref    

def rho_ref(Tref, mref, Bref):
    """Compute reference gyroradius rhoref = cref / Omegaref
    Inputs:
      Tref      measured in keV (scalar)
      mref      measured in proton masses (scalar)
      Bref      measured in Tesla (scalar)
    Outputs:
      rhoref    measured in m (scalar)
    """
    cref = c_ref(Tref, mref)
    Omegaref = omega_ref(Bref, mref)
    rhoref = cref / Omegaref
    return rhoref

def Q_ref(nref, Tref, mref, Bref, Lref):
    """Compute reference heat flux Qref = nref * Tref * cref * rhoref^2 / Lref^2.  
    
    The GENE Documentation, Appendix A, p66 calls this Qgb.
    
    Inputs:
      nref      measured in 10^19 m^-3 (scalar)
      Tref      measured in keV (scalar)
      mref      measured in proton masses (scalar)
      Bref      measured in Tesla (scalar)
      Lref      measured in m (scalar)
    Outputs:
      Qref      measured in Joules / (m^2 s)
    """
    cref = c_ref(Tref, mref)
    rhoref = rho_ref(Tref, mref, Bref)
    
    # convert to SI
    nrefGeneFactor = 1e19
    nref_SI = nrefGeneFactor * nref
    Tref_SI = 1000 * e * Tref
    
    Qref = nref_SI * Tref_SI * cref * rhoref**2 / Lref**2
    return Qref
    
def calculate_consistent_rhostar(Tref, Bref, mref, minorRadius):
    """Compute a self-consistent rhostar, given Tref, mref, Bref, and minor radius.
    
       rhostar = rhoref/a
       rhoref = cref/Omegaref
       cref = sqrt(Tref/mref)
       Omegaref = electron charge * Bref / mref
    
    Inputs:
      Tref          measured in keV (scalar)
      Bref          measured in Tesla (scalar)
      mref          measured in proton masses (scalar)
      minorRadius   minor radius a, measured in m (scalar)
    Outputs:
      rhostar       (cref/Omegaref)/a, dimensionless (scalar)
    """
    rhostar = rho_ref(Tref, mref, Bref) / minorRadius
    return rhostar