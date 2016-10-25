from __future__ import division
import numpy as np

def dx_centered_difference_edge_first_order(u, dx):
    """Compute du/dx.
      du/dx is computed using centered differences on the same grid as u.  For the edge points, one-point differences are used.
    
    Inputs:
      u         profile (array)
      dx        grid spacing (scalar)
    
    Outputs:
      dudx      (array, same length as u)
    """
    dudx = np.zeros_like(u, dtype=float)
    dudx[0] = (u[1] - u[0]) / dx
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    dudx[-1] = (u[-1] - u[-2]) / dx
    return dudx
    
def dx_centered_difference(u, dx):
    """Compute du/dx.
      du/dx is computed using centered differences on the same grid as u.  For the edge points, two-point differences are used.
    
    Inputs:
      u         profile (array)
      dx        grid spacing (scalar)
    
    Outputs:
      dudx      (array, same length as u)
    """
    dudx = np.zeros_like(u, dtype=float)
    dudx[0] = (-3/2*u[0] + 2*u[1] - 1/2*u[2]) / dx
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    dudx[-1] = (3/2*u[-1] - 2*u[-2] + 1/2*u[-3]) / dx    
    return dudx