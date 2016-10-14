"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division
import numpy as np
import scipy.signal

"""
Noisy Flux:  Decorator pattern for a FluxModel.

Decorator that adds random noise to the flux returned by a FluxModel.
"""

class noisyFlux(object):
    def __init__(self, FluxModel, amplitude, tac_x, dx):
        """
        Inputs:
          FluxModel         fluxmodel to be decorated
          amplitude         amplitude of noise (scalar)
          tac_x             autocorrelation time/length measured in units of x
          dx                grid spacing (scalar)
        """
        self.FluxModel = FluxModel
        self.amplitude = amplitude   # scalar
        self.tac = tac_x / dx        # autocorrelation time measured in discrete units (does not have to be integer)
        
    def GetFlux(self, profile):
        flux = self.FluxModel.GetFlux(profile)
        noisy_flux = self._AddNoise(flux, self.amplitude, self.tac)
        return noisy_flux
        
    @staticmethod
    def _AddNoise(v, amplitude, tac):
        """Add noise to an array v in the following way:.
        
                    noisy_v = (1+h) * v
           
        where h is a random noise with specified standard deviation and autocorrelation time.  The
        noise h is trimmed to be zero close to both boundaries.
        
        Inputs:
          v              input to add noise to (array)           
          ampltitude     specified standard deviation of noise (scalar)
          tac            autocorrelation time measured in discrete samples (scalar)
        Outputs:
          noisy_v        v with noise
        """
        numSamples = len(v)
        h = amplitude * ar1noise(numSamples, tac)
        h = noisyFlux._dampenSides(h)
        noisy_v = (1 + h) * v
        return noisy_v
    
    @staticmethod
    def _dampenSides(v, numPts=None):
        """
        Gradually dampen (linearly) the amplitude of v to zero on both sides.
        
        Inputs:
          v           input to trim (array)
          numPts    (optional) number of points to dampen on each side.  Default = 3% of len(v)
        Outputs:
          v_dampened  (array)
        
        """
        N = len(v)
        if numPts is None:
            numPts = int(0.03 * N)    # default number of points
        assert len(v) > 2 * numPts, 'numPts is too large'
        
        v_dampened = v.copy()
        v_dampened[0:numPts] = v_dampened[0:numPts] * np.linspace(0, 1, numPts)
        v_dampened[-numPts:] = v_dampened[-numPts:] * np.linspace(1, 0, numPts)
        return v_dampened
        
    

def ar1noise(numSamples, tac):
    """Generate an AR(1) process.
    
    Generate samples with a user-specified autocorrelation time and unit variance from the AR(1) process
    
                                y_n+1  =  lambda * y_n  +  a_n
    
    where a_n is a white noise process.  lambda is specified in terms of the autocorrelation time tac.
    
    Here, the memory of the initial condition y_0 is removed by generating some extra samples and
    discard them.  The memory is forgotten by the mth iterate, as long as m satisfies lambda^(2*m) >> 1.
    For example, we can set m = -log(1000) / (2*log(lambda)) and then discard the first m iterates.
    
    Inputs:
      numSamples        desired number of samples (scalar)
      tac               2-sided autocorrelation time (essentially, double the e-folding time of the
                        autocorrelation function), measured in discrete samples (scalar)
    Outputs:
      noise             noise with unit variance and autocorrelation time tac.  (array of length numSamples)
    """
    lamb = (tac - 1) / (tac + 1)
    cc = 1000
    m = int( -np.log(cc) / (2*np.log(lamb)) ) + 1
    numSamplesPlusm = numSamples + m
    
    white_noise_variance = 1 - lamb**2  # variance to make output noise unit variance
    white_noise = np.random.normal(scale=np.sqrt(white_noise_variance), size=numSamplesPlusm)
    
    phi = np.array([1, -lamb])
    b = np.array([1])
    # run the noise through the AR process
    noise = scipy.signal.lfilter(b, phi, white_noise)
    
    # discard memory of initial conditions    
    noise = noise[m:]               
    return noise