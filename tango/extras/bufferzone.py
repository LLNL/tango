"""
bufferzone

This module provides functionality to apply a GENE-like "buffer-zone" to a turbulent model of flux.  It mimics the effect of
the buffer zone in GENE by damping out the flux returned by a FluxModel at the boundaries of the domain.

The code uses a decorator pattern for a FluxModel.

See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import scipy.signal

class BufferZone(object):
    """Decorator that damps out the flux at the boundaries of the domain.  The width over which to apply damping on each boundary
    as a fraction of domain size is controlled by the epsilon parameter.
    """
    def __init__(self, fluxModel, taperwidth):
        """
        Inputs:
          fluxModel         fluxmodel to be decorated (object)
          taperwidth        width over which to apply damping at each boundary, as a fraction of domain size (scalar)
        """
        self.fluxModel = fluxModel
        self.taperwidth = taperwidth
        
    def get_flux(self, profiles):
        fluxes = self.fluxModel.get_flux(profiles)
        dampedFluxes = self._damp_fluxes(fluxes)
        return dampedFluxes
        
    def _damp_fluxes(self, fluxes):
        dampedFluxes = {}
        for label in fluxes:
            dampedFluxes[label] = _damp_flux(fluxes[label], self.taperwidth)
        return dampedFluxes
        
def _damp_flux(flux, taperwidth):
    """Apply damping to the sides of the input flux.  The damping applies to a width of
    size taperwidth on each side of the domain.
    
    Currently, uses a Tukey window.
    
    Inputs:
      flux              input signal on which to damp sides (array)
      taperwidth        width over which to apply damping at each boundary, as a fraction of domain size (scalar)
    Outputs:
      dampedFlux        signal with edges damped (array)
    """
    alpha = taperwidth * 2      # alpha sets the width of the damping for the entire signal, accounting for both left and right boundaries
    tukeyWindow = scipy.signal.tukey(len(flux), alpha)
    dampedFlux = flux * tukeyWindow
    return dampedFlux