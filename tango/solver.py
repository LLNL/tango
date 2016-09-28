"""See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division, absolute_import
import numpy as np
import logging
import tango as tng
from . import datasaver

"""
solver
"""

class solver(object):
    def __init__(self, L, x, profile_IC, profile_rightBC, t_array, MaxIterations, tol, ComputeAllH, turbhandler):
        self.L = L        # size of domain
        self.x = x        # grid (array)
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.profile = profile_IC               # initial condition for the profile (array)
        self.profile_mminus1 = profile_IC       # initialize "m minus 1" variable
        self.profile_rightBC = profile_rightBC  # dirichlet boundary condition
        self.t_array = t_array                   # times t at which the solution is desired (array).  First value is initial condition
        self.MaxIterations = MaxIterations      # maximum number of iterations per timestep (scalar)
        self.tol = tol                          # tolerance for iteration convergence (scalar)
        assert callable(ComputeAllH), "ComputeAllH must be a function"
        self.ComputeAllH = ComputeAllH
        assert hasattr(turbhandler, 'Hcontrib_TurbulentFlux') and callable(getattr(turbhandler, 'Hcontrib_TurbulentFlux'))
        self.turbhandler = turbhandler
        
        # initialize other instance data
        self.DataSaverHandler = datasaver.dataSaverHandler()
        self.solution_error = False
        self.t = 0
        self.dt = None        
        self.l = None                                         # Iteration index
        self.converged = None
        self.errhistory = np.zeros(self.MaxIterations)      # error history vs. iteration at a given timestep
        self.t_final = t_array[-1]
        self.m = 0                                            # Timestep index
        self.reached_end = False
    
    def TakeTimestep(self):
        # Implicit time advance: iterate to solve the nonlinear equation!
        self.m += 1
        self.converged = False
        (self.t, self.dt) = self._get_tdt(self.t_array, self.m)
        
        self.l = 0   # reset iteration counter
        self.errhistory[:] = 0
        logging.info("Timestep m={}:  Beginning iteration loop ...".format(self.m))
        
        # compute next iteration
        
        while self.l < self.MaxIterations and not self.converged and not self.solution_error:
            self.ComputeNextIteration()

        if self.converged:            
            logging.info("Timestep m={}:  Converged!  Successfully found the solution for t={}.  Rms error={}.  Took {} iterations.".format(self.m, self.t, self.errhistory[self.l-1], self.l))
            if self.m >= len(self.t_array) - 1:
                logging.info("Reached the final timestep m={} at t={}.  Simulation ending...".format(self.m, self.t))
                self.reached_end = True
        
        
        # DATA SAVER STUFF        
        
        # Save some stuff   
        errhistory_final = self.errhistory[0:self.l]
        one_off_data = {'x': self.x, 'profile_m': self.profile, 'profile_mminus1': self.profile_mminus1, 'errhistory': errhistory_final, 't': self.t, 'm': self.m}
        self.DataSaverHandler.add_one_off_data(one_off_data)
        self.DataSaverHandler.save_to_file(self.m)
        self.DataSaverHandler.reset_for_next_timestep()
        self.profile_mminus1 = self.profile

    def ComputeNextIteration(self):
        # compute H's from current iterate of profile
        (H1, H2, H3, H4, H6, H7, extraturbdata) = self.ComputeAllH(self.t, self.x, self.profile, self.turbhandler)
        
        # compute matrix system (A, B, C, f)
        (A, B, C, f) = tng.HToMatrix(self.dt, self.dx, self.profile_rightBC, self.profile_mminus1, H1, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7)

        self.converged, rms_error, resid = self.CheckConvergence(A, B, C, f, self.profile, self.tol)
        self.errhistory[self.l] = rms_error
        
        # compute new iterate of profile
        self.profile = tng.solve(A, B, C, f)
        
        logging.info("Timestep m={}: after iteration number l={}, first 4 entries of profile={};  last 4 entries={}".format(
            self.m, self.l, self.profile[:4], self.profile[-4:]))
        
        # save data if desired
        datadict = self._pkgdata(H1=H2, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7, A=A, B=B, C=C, f=f, profile=self.profile, extradata=extraturbdata)
        self.DataSaverHandler.add_data(datadict, self.l)
        
        # Check for NaNs or infs or negative values
        self.CheckProfileIsValid(self.profile)
        
        # about to loop to next iteration l
        self.l += 1
        if self.l >= self.MaxIterations:
            logging.warning("Timestep m={} and time t={}:  MaxIterations ({}) reached.  Error={} while tol={}".format(self.m, self.t, self.MaxIterations, rms_error, self.tol))
    
    @property
    def ok(self):
        """True unless a stop condition is reached."""
        if self.solution_error == True:
            return False
        elif self.m >= len(self.t_array) - 1:
            return False
        elif self.t >= self.t_final:
            return False
        elif self.l >= self.MaxIterations:
            return False
        else:
            return True
    
    def CheckProfileIsValid(self, profile):
        """Check the profile for validity: check for NaNs, infinities, negative numbers
        """
        if np.all(np.isfinite(profile)) == False:
            logging.error("NaN or Inf detected in profile at l={}.  Aborting...".format(self.l))
            self.solution_error = True
        if np.any(profile < 0) == True:
            logging.error("Negative value detected in profile at l={}.  Aborting...".format(self.l))
            self.solution_error = True
            

    def CheckConvergence(self, A, B, C, f, profile, tol):
        # convergence check: is || ( M[n^l] n^l - f[n^l] ) / max(abs(f)) || < tol
        # could add more convergence checks
        resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
        resid = resid / np.max(np.abs(f))  # normalize residuals
        rms_error = np.sqrt( 1/len(resid) * np.sum(resid**2))  
        converged = False
        if rms_error < tol:
            converged = True
        return (converged, rms_error, resid)
    
    @staticmethod
    def _get_tdt(t_array, timestep_number):
        """Compute the next time and timestep for an already-defined time array.
        
        For example, if t_array = [0 2.5 10], then for timestep_number = 1, we are looking for the solution at
        t[1]=2.5, so t_new = 2.5 and dt = 2.5.  If timestep_number = 2, we are looking for the solution at t[2],
        so t_new = 10 and dt = 7.5.
        """
        t_new = t_array[timestep_number]
        t_old = t_array[timestep_number - 1]
        dt = t_new - t_old
        return t_new, dt
        
    def _pkgdata(self, H1=None, H2=None, H3=None, H4=None, H6=None, H7=None, A=None, B=None, C=None, f=None, profile=None, extradata=None):
        """input dict extradata contains the following [see Hcontrib_TurbulentFlux in lodestro_method.py]:
        'D': data['D'], 'c': data['c'],
        'profile_turbgrid': data['profile_turbgrid'], 'profileEWMA_turbgrid': data['profileEWMA_turbgrid'],
        'flux_turbgrid': data['flux_turbgrid'], 'fluxEWMA_turbgrid': data['fluxEWMA_turbgrid'],
        'D_turbgrid': data['D_turbgrid'], 'c_turbgrid': data['c_turbgrid'],
        'Dhat_turbgrid': data['Dhat_turbgrid'], 'chat_turbgrid': data['chat_turbgrid'], 'theta_turbgrid': data['theta_turbgrid']}
        """
        data1 = {'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'H6': H6, 'H7': H7, 'A': A, 'B': B, 'C': C, 'f': f, 'profile': profile}
        pkg_data = self._merge_two_dicts(extradata, data1)
        return pkg_data

    @staticmethod            
    def _merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = x.copy()
        z.update(y)
        return z