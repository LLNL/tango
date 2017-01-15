"""
safety_factor_fit

Magnetic safety factor fit to a 5th order polynomial

GENE allows 5th order polynomial in its parameters file, but there is no simple way to input an arbitrary safety factor
profile.  But the ability to use an arbitrary profile is needed because through the libgene_tango interface, an arbitrary
safety factor can be input.  So I want a way to use the same arbitrary safety factor when running GENE standalone, without
the Python interface.

Therefore, this function takes an arbitrary profile and fit it to an nth order polynomial (max 5), which should be
sufficient for any smooth function.
"""

from __future__ import division
import numpy as np

def write_default1():
    filename = 'q_coeffs'
    qCoeffs = default1()
    write(filename, qCoeffs)

def write(filename, qCoeffs):
    """write qCoeffs to file
    
    Inputs:
      qCoeffs     polynomial coefficients for safety factor.  First element is coefficient of (r/a)^0, second is coefficient
                    of (r/a)^1, etc.  Length is arbitrary.  (array)
    """
    qCoeffs = qCoeffs.reshape((1, len(qCoeffs))) # reshape into a 1 x n array so numpy saves onto a single line
    np.savetxt(filename, qCoeffs, fmt='%.4f', delimiter=', ')


def default1():
    numRadialPts = 144;
    rhoMin = 0.1
    rhoMax = 0.9
    rho = np.linspace(rhoMin, rhoMax, numRadialPts);
    qbar = 0.854 + 2.184 * rho**2;

    minorRadius = 0.28;  # a
    majorRadius = 0.71;  # R0
    r = rho * minorRadius;
    safetyFactor = qbar / np.sqrt(1 - (r/majorRadius)**2);

    # fit
    polynomialDegree = 3;
    p = np.polyfit(rho, safetyFactor, polynomialDegree)
    qCoeffs = p[::-1]
    return qCoeffs