"""Spatial smoothing"""

from __future__ import division
import numpy as np

class Smoother(object):
    """Class-based interface to the moving_average function below."""
    def __init__(self, windowSize):
        self.windowSize = windowSize
    def smooth(self, x):
        return moving_average(x, self.windowSize)

def moving_average(x, windowSize):
    """Computes the centered moving average of x
    
    End points are handled in a special way.  At the end point, a centered moving average is not possible.  This occurs 
    for (windowSize-1)/2 points on each end.  What we do here is to simply not average these end points; the output of
    this function just uses the input data at that point.
    
    This is done only because it is simple and shouldn't matter due to particulars of GENE's boundary conditions.  GENE
    uses a buffer zone at both radial ends that damp out fluctuations, so the values of the flux being handled should be
    close to zero anyway.  A more sophisticated approach would continue to average over a few points.
    
    This function requires that windowSize be an odd integer, so that a centered moving average makes sense.
    
    Inputs:
      x                 input data (array)
      windowSize        size of moving average window in points (odd integer)
    Outputs:
      xAvg              output data (array)
    """
    
    assert windowSize % 2 == 1, 'windowSize must be an odd integer.'
    boundarySize = int(windowSize-1) // 2
    xAvg = np.zeros_like(x)
    # moving average 
    cumsum = np.cumsum(np.insert(x, 0, 0))
    movingAvg = (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
    xAvg[boundarySize:-boundarySize] = movingAvg
    # end points
    xAvg[:boundarySize] = x[:boundarySize]
    xAvg[-boundarySize:] = x[-boundarySize:]
    return xAvg