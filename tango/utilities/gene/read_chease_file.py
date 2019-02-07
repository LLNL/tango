"""
read_chease_file

Module for reading the output file that the magnetic equilibrium code CHEASE produces for GENE.  This output file is
in hdf5 format.  The CHEASE file, usually with a .h5 extension, provides the data to specify the magnetic geometry for
a tokamak.  CHEASE provides the data on a (psi, chi) grid, where psi is the poloidal flux divided by 2*pi, and chi is
a straight-field-line poloidal angle

Some of the data CHEASE provides include
    --metric coefficients g11, g12, g21, g22 (grad psi dot grad psi, etc)
    --rho_toroidal (another flux quantity, normalized toroidal flux, that is often used as the radial coordinate within GENE)
    --safety factor
    --profiles such as ion temperature, electron temperature, density, and their derivatives

This module reads the CHEASE file and provides an interface to providing the data that Tango needs.  Tango needs
    --radial coordinate grids psi and rho_tor.  In this module, rho_tor is just labeled 'x', since rho_tor is the default
          radial coordinate used by GENE for realistic magnetic geometry.  In the future this module could be made more
          general to allow for an arbitrary radial coordinate x, and specifically distinguish rho_tor
    --differential volume dV/dx
    --safety factor
    --differential conversion between psi and x, dpsi/dx
    --flux surface average of the metric coefficient <g^xx> = <grad x dot grad x>
    --References value for magnetic field and length.  Bref = magnetic field on axis, Lref = major radius ???

This module can also read the profiles such as Ti, Te, ni, ne.  These are not required by Tango, but they may be useful.
"""

from __future__ import division
import numpy as np
import h5py
import scipy.interpolate

class CheaseTangoData(object):
    """Output container to store the data read from a CHEASE file, then interpolated, and passed to Tango."""
    def __init__(self, psi, x, dVdx, safetyFactor, dpsidx, gxxAvg, gradxAvg, Bref, Lref, minorRadius):
        self.psi = psi
        self.x = x
        self.dVdx = dVdx
        self.safetyFactor = safetyFactor
        self.dpsidx = dpsidx
        self.gxxAvg = gxxAvg
        self.gradxAvg = gradxAvg
        self.Bref = Bref
        self.Lref = Lref
        self.minorRadius = minorRadius

class CheaseTangoProfiles(object):
    """Output container to store the temperature and density profiles read from a CHEASE file, then interpolated,
    and passed to Tango.
    
    Electron profiles may be optional.
    """
    def __init__(self, x, ionTemperature, ionDensity, electronTemperature=None, electronDensity=None):
        self.x = x
        self.ionTemperature = ionTemperature
        self.ionDensity = ionDensity
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity

#################################################################################################################
# ------------------------ Functions for reading directly from the CHEASE hdf5 file --------------------------- #


def get_reference_vals(f):
    """Return the magnetic field and major radius reference values.

    Inputs:
      f             An open hdf5 chease file

    Outputs:
      Bref          magnetic field on axis to be used as reference magnetic field (scalar)
      majorRadius   major radius to be used as reference length (scalar)
      minorRadius   minor radius value, equal to rho_tor at the LCFS, which is the final value of rho_tor
    """
    Bref = f['data'].attrs['B0EXP']
    majorRadius = f['data'].attrs['R0EXP']
    rhoTor = get_rhotor(f)
    minorRadius = rhoTor[-1]
    return (Bref, majorRadius, minorRadius)


def get_rhotor(f):
    """Return the toroidal flux (1d array) as provided by CHEASE. One possible radial coordinate"""
    rhoTor = f['data/var1d/rho_tor'][:]
    return rhoTor


def get_psi(f):
    """Return the poloidal flux divided by 2*pi (1D array) as provided by CHEASE.  Another possible radial
    coordinate.
    """
    psi = f['data/grid/PSI'][:]
    return psi


def get_chi(f):
    """Return the straight-field-line poloidal angle (1D array) as provided by CHEASE.  The poloidal coordinate."""
    chi = f['data/grid/CHI'][:]
    return chi


def get_g11(f):
    """Return the metric coefficient g11 = grad psi dot grad psi on the CHEASE grid.  2D array of size nchi x npsi."""
    g11 = f['data/var2d/g11'][:]
    return g11


def get_safety_factor(f):
    """Return the safety factor on the CHEASE grid.  1D array of size npsi."""
    safetyFactor = f['data/var1d/q'][:]
    return safetyFactor


def get_jacobian(f):
    """Return the jacobian of the coordinate system on the CHEASE grid.  2D array of size nchi x npsi"""
    jacobian = f['data/var2d/Jacobian'][:]
    return jacobian


def get_dVdpsi(f):
    """Return the differential volume dV/dpsi on the CHEASE grid.  1D array of size npsi."""
    dVdpsi = f['data/var1d/dVdpsi'][:]
    return dVdpsi


def get_dpsidrhotor(f):
    """Return dpsi / drhotor = dpsi/dx on the CHEASE grid.  1D array of size npsi."""
    dpsidrhotor = f['data/var1d/dpsidrhotor'][:]
    return dpsidrhotor


def get_Ti(f):
    """Return the ion temperature on the CHEASE grid.  1D array of size npsi."""
    Ti = f['data/var1d/Ti'][:]
    return Ti


def get_Te(f):
    """Return the electron temperature on the CHEASE grid.  1D array of size npsi."""
    Te = f['data/var1d/Te'][:]
    return Te


def get_ni(f):
    """Return the ion density on the CHEASE grid.  1D array of size npsi."""
    ni = f['data/var1d/ni'][:]
    return ni


def get_ne(f):
    """Return the electron density on the CHEASE grid.  1D array of size npsi."""
    ne = f['data/var1d/ne'][:]
    return ne

#################################################################################################################
# --------------------------- Functions for performing computations on CHEASE data ---------------------------- #    
def integrate_2d_qty_in_chi(chi, qty):
    """Integrate a 2D quantity in chi (i.e., poloidal angle) on chease grid.  The first dimension is assumed to be
    the coordinate chi and the second dimension psi, as that is how the CHEASE file provides the data.
    
    A cross check for this integration is posible using the definition of dV/dpsi:
        
            dV/dpsi on the CHEASE grid should be equal to 2pi * integral[J(psi, chi, {chi, 0, 2*pi})],
    
    where J is the Jacobian.  This is indeed satisfied, to very good approximation.  
    
    Inputs:
      chi               grid for poloidal coordinate (1D array)
      qty               quantity to be integrated (2D array, nchi x npsi)
    
    Outputs:
      integratedQty     quantity as a function of psi, after integrating in chi (1D array, size npsi)
    """
    dchi = chi[1] - chi[0]
    # integrate in chi by summing along the first dimension
    integratedQty = dchi * np.sum(qty, axis=0)
    return integratedQty

def integrate_chi_2d_qty_weighted_by_jacobian(chi, jacobian, qty):
    """Integrate a 2D quantity in chi (i.e., poloidal angle theta) on chease grid, weighted by the Jacobian.
    
    One potential problem is the singularity at the magnetic axis, psi=0.  The Jacobian is set to zero here.
    Because nothing is really going to be evaluated at psi=0, there wouldn't be a problem, *except* for the
    fact that we are constructing splines in order to interpolate onto the Tango grid.  Therefore having a
    quantity with a value of zero at psi=0 will cause a bad interpolation onto the small psi values.  A
    simple but perhaps non-optimal fix is to just set the quantity to be the value at the first nonzero
    gridpoint, rather than 0.  We do this here.
    
    Inputs:
      chi                       grid for poloidal coordinate (1D array)
      jacobian                  jacobian of coordinate system (2D array, nchi x npsi)
      qty                       quantity to be integrated (2D array, nchi x npsi)
    
    Outputs:
      integratedWeightedQty     quantity as a function of psi, after integrating in chi (1D array, size npsi)
    """
    integratedWeightedQty = integrate_2d_qty_in_chi(chi, jacobian*qty)
    integratedWeightedQty[0] = integratedWeightedQty[1]
    return integratedWeightedQty
    
def flux_surface_average(chi, jacobian, qty):
    """Compute a flux surface average of a 2D quantity (psi, chi) on the CHEASE grid.
    
    Flux surface average is defined for an axisymmetric quantity Q as:
        
            <Q> = integral[Jacobian * Q, theta] / integral[Jacobian, theta]
        
    Inputs:
      chi                       grid for poloidal coordinate (1D array)
      jacobian                  jacobian of coordinate system (2D array, nchi x npsi)
      qty                       quantity to be flux-surface averaged (2D array, nchi x npsi)
    
    Outputs:
      qtyFluxSurfAvg            flux-surface averaged quantity as a function of psi (1D array, size npsi)
    """
    numerator = integrate_chi_2d_qty_weighted_by_jacobian(chi, jacobian, qty)
    denominator = integrate_chi_2d_qty_weighted_by_jacobian(chi, jacobian, 1)
    
    qtyFluxSurfAvg = numerator / denominator
    return qtyFluxSurfAvg    
    
def compute_avg_gxx_cheasegrid(f):
    """Compute the flux surface average of the metric coefficient g^xx = grad x dot grad x on the CHEASE grid,
    where x = rho_tor.
    
    CHEASE provides g^11 = grad psi dot grad psi.  This function first computes <g^11>, then performs a change
    of coordinates to g^xx.  This is justified because grad x = grad psi * dx/dpsi, and thus
        
            grad x dot grad x = grad psi dot grad psi * (dx/dpsi)^2
            
    Then, since dx/dpsi is a function of flux coordinate only, it passes through a flux surface average, yielding
            
            <grad x dot grad x> = <grad psi dot grad psi> * (dx/dpsi)^2
    or
            <g^xx> = <g^11> * (dx/dpsi)^2
            
    Inputs:
      f         An open hdf5 chease file
    
    Outputs:
      gxxAvg    <g^xx> = <grad x dot grad x> on the CHEASE grid (1D array, size npsi)
    """
    chi = get_chi(f)
    jacobian = get_jacobian(f)
    g11 = get_g11(f)
    g11Avg = flux_surface_average(chi, jacobian, g11)
    
    # convert <g^11> to <g^xx> through a change of coordinates
    dpsidx = get_dpsidrhotor(f)
    dpsidx[0] = dpsidx[1]  # remove what could be a divide by zero error
    dxdpsi = 1 / dpsidx
    gxxAvg = dxdpsi**2 * g11Avg
    return gxxAvg

def compute_avg_gradx_cheasegrid(f):
    """Compute the flux surface average of sqrt(g^xx) = |grad x| on the CHEASE grid, where x = rho_tor.
    
    CHEASE provides g^11 = grad psi dot grad psi.  Here, we compute <sqrt(g^11)> = <|grad psi|>, then performs
    a change of coordinates to <|grad x|>.  This is justified because grad x = grad psi * dx/dpsi, and thus
    
            <|grad x|> = <|grad psi|> * dx/dpsi
    
    Inputs:
        f       An open hdf5 chease file
    
    Outputs:
        gradxAvg    <|grad x|> on the CHEASE grid (1D array, size npsi)
    """
    chi = get_chi(f)
    jacobian = get_jacobian(f)
    g11 = get_g11(f)        # |grad psi|^2
    gradpsi = np.sqrt(g11)  # |grad psi|
    gradpsiAvg = flux_surface_average(chi, jacobian, gradpsi)    # <|grad psi|>
    
    # convert <|grad psi|> to <|grad x|> through a change of coordinates
    dpsidx = get_dpsidrhotor(f)
    dpsidx[0] = dpsidx[1]   # remove what could be a divide by zero error
    dxdpsi = 1 / dpsidx
    gradxAvg = dxdpsi * gradpsiAvg      # <|grad x|>
    return gradxAvg
    
#################################################################################################################
# ---------------- Functions for collecting CHEASE data and interpolating to the Tango grid ------------------- #
def create_tango_x_grid(xmax, numRadialPts):
    """Create a uniformly spaced grid of the radial coordinate x = rho_toroidal for Tango.
    
    Assume the Tango grid goes down to x=0 (or rather, dx/2 as the lowest point.  Not sure if this function will
    be necessary, or if the interface will assume xTango is passed in externally.
    
    Inputs:
      xmax              The value of the final gridpoint of rho_tor (scalar)
      numRadialPts      Number of grid points for the radial grid (intege)
    
    Outputs:
      xTango            Uniformly spaced grid that goes from [dx/2, xmax]
    """
    dx = xmax / (numRadialPts - 0.5)
    xTango = np.linspace(dx/2, xmax, numRadialPts)
    return xTango

def interpolate_1d_qty(xChease, xTango, qtyChease):
    """Interpolate a 1D quantity from Chease radial grid to Tango radial grid.
    
    The interpolation is necessary for two reasons:
        1) Tango requires a uniformly spaced grid.  The grid that CHEASE provides is not uniformly spaced in the
               coordinate x = rho_tor
        2) Tango can use a different domain and different number of grid points as CHEASE.  In particular the
               radial boundary for CHEASE is at the last closed flux surface, whereas Tango probably wants to
               end its domain before that.
             
    Inputs:
      xChease           rho_tor grid as provided by CHEASE (1D array)
      xTango            rho_tor grid for Tango on which to interpolate the quantity (1D array)
      qtyChease         quantity evaluated on CHEASE grid points xChease (1D array)
    
    Outputs:
      qtyTango          quantity interpolated onto xTango grid points (1D array)
    """
    interpolator = scipy.interpolate.InterpolatedUnivariateSpline(xChease, qtyChease)
    qtyTango = interpolator(xTango)
    return qtyTango

def interpolate_psi(f, xTango):
    """Interpolate psi = poloidal flux / 2pi onto the Tango grid.
    
    Inputs:
      f             An open hdf5 chease file
      xTango        Tango rho_tor grid (1D array)
    
    Outputs:
      psiTango      psi interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    psiChease = get_psi(f)
    psiTango = interpolate_1d_qty(xChease, xTango, psiChease)
    return psiTango
    
def interpolate_dVdx(f, xTango):
    """Interpolate dV/dx onto the Tango grid.
    
    Since CHEASE provides dV/dpsi, first dV/dx is computed on the CHEASE grid from 
    
            dV/dx = dV/dpsi * dpsi/dx

    Inputs:
      f             An open hdf5 chease file
      xTango        Tango rho_tor grid (1D array)
    
    Outputs:
      dVdxTango     dV/dx interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    dVdpsiChease = get_dVdpsi(f)
    dpsidxChease = get_dpsidrhotor(f)
    dVdxChease = dVdpsiChease * dpsidxChease  # by the chain rule
    dVdxTango = interpolate_1d_qty(xChease, xTango, dVdxChease)
    return dVdxTango        
    
def interpolate_safety_factor(f, xTango):
    """Interpolate the safety factor q onto the Tango grid.
    
    Inputs:
      f                     An open hdf5 chease file
      xTango                Tango rho_tor grid (1D array)
    
    Outputs:
      safetyFactorTango     safety factor q interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    safetyFactorChease = get_safety_factor(f)
    safetyFactorTango = interpolate_1d_qty(xChease, xTango, safetyFactorChease)    
    return safetyFactorTango
    
def interpolate_dpsidx(f, xTango):
    """Interpolate dpsi/dx onto the Tango grid.
    
    Caution: Depending on what is doing with this output, one needs to be careful, because the first element
    of dpsi/dx on the CHEASE grid (i.e., dpsi/dx at psi=0) can be zero.
    
    Inputs:
      f             An open hdf5 chease file
      xTango        Tango rho_tor grid (1D array)
    
    Outputs:
      dpsidxTango   dpsi/dx interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    dpsidxChease = get_dpsidrhotor(f)
    dpsidxTango = interpolate_1d_qty(xChease, xTango, dpsidxChease)
    return dpsidxTango
    
def interpolate_gxxAvg(f, xTango):
    """Compute <g^xx> = <|grad x|^2> on Chease grid, then interpolate onto the Tango grid.
    
    Inputs:
      f             An open hdf5 chease file
      xTango        Tango rho_tor grid (1D array)
    
    Outputs:
      gxxAvgTango   <|grad x|^2> interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    gxxAvgChease = compute_avg_gxx_cheasegrid(f)
    gxxAvgTango = interpolate_1d_qty(xChease, xTango, gxxAvgChease)
    return gxxAvgTango

def interpolate_gradxAvg(f, xTango):
    """Compute <|grad x|> on Chease grid, then interpolate onto the Tango grid.
    
    Inputs:
        f               An open hdf5 chease file
        xTango          Tango rho_tor grid (1D array)
        
    Outputs:
        gradxAvgTango   <|grad x|> interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    gradxAvgChease = compute_avg_gradx_cheasegrid(f)
    gradxAvgTango = interpolate_1d_qty(xChease, xTango, gradxAvgChease)
    return gradxAvgTango
    
def gather_1d_interpolations(f, xTango):
    """Gather 1D interpolations to get the quantities needed by Tango onto the Tango grid.

    Note, psiTango provides psi(x) evaluated at xTango.  In other words, this array holds the values of psi (the
    poloidal flux divided by 2 pi) at the radial locations corresponding to the entries of xTango.

    Inputs:
      f                     An open hdf5 chease file
      xTango                Tango rho_tor grid (1D array)

    Outputs:
      psiTango              poloidal flux / 2pi interpolated onto the Tango grid (1D array)
      dVdxTango             dV/dx interpolated onto the Tango grid (1D array)
      safetyFactorTango     safety factor q interpolated onto the Tango grid (1D array)
      dpsidxTango           dpsi/dx interpolated onto the Tango grid (1D array)
      gxxAvgTango           <g^xx> interpolated onto the Tango grid (1D array)
    """
    psiTango = interpolate_psi(f, xTango)
    dVdxTango = interpolate_dVdx(f, xTango)
    safetyFactorTango = interpolate_safety_factor(f, xTango)
    dpsidxTango = interpolate_dpsidx(f, xTango)
    gxxAvgTango = interpolate_gxxAvg(f, xTango)
    gradxAvgTango = interpolate_gradxAvg(f, xTango)
    return (psiTango, dVdxTango, safetyFactorTango, dpsidxTango, gxxAvgTango, gradxAvgTango)
    
def interpolate_profiles(f, xTango):
    """Collect the temperature and density profiles from the CHEASE file and interpolate to Tango grid.
    
    Note that CHEASE provides both ion and electron density, as they can differ if impurities are present.  CHEASE
    also can provide zeff, the effective charge number of the ions.  However, currently this module does not use
    zeff

    Inputs:
      f                         An open hdf5 chease file
      xTango                    Tango rho_tor grid (1D array)

    Outputs:
      ionTemperatureTango       ion temperature interpolated onto the Tango grid (1D array)
      ionDensityTango           ion density interpolated onto the Tango grid (1D array)
      electronTemperatureTango  electron temperature interpolated onto the Tango grid (1D array)
      electronDensityTango      electron density interpolated onto the Tango grid (1D array)
    """
    xChease = get_rhotor(f)
    ionTemperatureChease = get_Ti(f)
    ionDensityChease = get_ni(f)
    electronTemperatureChease = get_Te(f)
    electronDensityChease = get_ne(f)

    ionTemperatureTango = interpolate_1d_qty(xChease, xTango, ionTemperatureChease)
    ionDensityTango = interpolate_1d_qty(xChease, xTango, ionDensityChease)
    electronTemperatureTango = interpolate_1d_qty(xChease, xTango, electronTemperatureChease)
    electronDensityTango = interpolate_1d_qty(xChease, xTango, electronDensityChease)

    return (ionTemperatureTango, ionDensityTango, electronTemperatureTango, electronDensityTango)

#################################################################################################################
# -------------------------------- Functions for interface used by Tango -------------------------------------- #


def get_chease_data_on_Tango_grid(filename, rhoTango):
    """Given a Tango grid, get CHEASE data needed by Tango, interpolated to the Tango grid.

    This function does not return the temperature and density profiles.

    Inputs:
      filename              path to CHEASE hdf5 file (string)
      rhoTango              Tango normalized rho_tor grid = rho_tor / rho_tor_max.  0 < rho < 1 (1D array)

    Outputs:
      cheaseTangoData       container for CHEASE data interpolated onto the Tango grid (CheaseTangoData)
    """
    with h5py.File(filename, 'r') as f:
        (Bref, majorRadius, minorRadius) = get_reference_vals(f)
        xTango = minorRadius * rhoTango
        (psi, dVdx, safetyFactor, dpsidx, gxxAvg, gradxAvg) = gather_1d_interpolations(f, xTango)

    # in GENE, when using CHEASE, Lref is always set to the major radius from the CHEASE file.  We follow this here.
    Lref = majorRadius
    cheaseTangoData = CheaseTangoData(psi, xTango, dVdx, safetyFactor, dpsidx, gxxAvg, gradxAvg, Bref, Lref, minorRadius)
    return cheaseTangoData


def get_chease_profiles_on_Tango_grid(filename, rhoTango):
    """Given a Tango grid, get CHEASE temperature and density profiles, interpolated on the Tango grid.

    Inputs:
      filename              path to CHEASE hdf5 file (string)
      xTango                Tango rho_tor grid (1D array)

    Outputs:
      cheaseTangoProfiles   container for CHEASE profiles interpolated onto the Tango grid (CheaseTangoProfiles)
    """
    with h5py.File(filename, 'r') as f:
        (_, _, minorRadius) = get_reference_vals(f)
        xTango = minorRadius * rhoTango
        (ionTemperature, ionDensity, electronTemperature, electronDensity) = interpolate_profiles(f, xTango)
        cheaseTangoProfiles = CheaseTangoProfiles(xTango, ionTemperature, ionDensity, electronTemperature, electronDensity)
        return cheaseTangoProfiles
