"""Spatial smoothing"""

from __future__ import division
import numpy as np

class Smoother(object):
    """Class-based interface to the moving_average function below."""
    def __init__(self, windowSize):
        self.windowSize = add_one_if_even(windowSize)
    def smooth(self, x):
        return moving_average(x, self.windowSize)

def moving_average(x, windowSize):
    """Computes the centered moving average of x.
    
    End points are handled in a special way.  At the end points, a centered moving average with the same number of grid
    points is not possible.  This occurs for (windowSize-1)/2 points on each end.  For these points, what we do here is
    average these points by the number of grid points left until you hit the boundary.
    
    This function requires that windowSize be an odd integer, so that a centered moving average makes sense.
    
    Inputs:
      x                 input data (1D array)
      windowSize        size of moving average window in points (odd integer)
    Outputs:
      xAvg              output data (1D array)
    """
    if windowSize == 1:  # no averaging
        return x
    
    assert windowSize % 2 == 1, 'windowSize must be an odd integer.'
    boundarySize = int(windowSize-1) // 2
    xAvg = np.zeros_like(x)
    # moving average 
    cumsum = np.cumsum(np.insert(x, 0, 0))
    movingAvg = (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
    xAvg[boundarySize:-boundarySize] = movingAvg
    
    # end points
    for j in np.arange(boundarySize):
        windowSizeForBndy = 2*j + 1
        # left side
        xAvg[j] = np.mean(x[:windowSizeForBndy])
        
        # right side
        k = j + 1  # because the last element is indexed by -1, not -0
        xAvg[-k] = np.mean(x[-windowSizeForBndy:])
    return xAvg
    
def add_one_if_even(n):
    """if n is an even integer, add one and return an odd integer.  If n is an odd integer, return n"""
    if n % 2 == 0:
        return n + 1
    elif n % 2 == 1:
        return n
    else:
        raise ValueError('n does not appear to be an integer.')