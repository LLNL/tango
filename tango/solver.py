"""
solver

Module that handles the meat of solving a timestep in a Tango transport equation, including the inner
loop of iterating to solve a nonlinear equation.

Convergence of the iteration loop is determined through comparing the rms residual of the nonlinear transport
equation to a relative tolerance value tol.

When all of the timesteps are successfully completed, solver.ok changes from True to False.

Failure modes:
    --convergence tolerance not reached within the maximum number of iterations (maxIterations)
    --solution becomes unphysical (negative, infinite, or NaN vlaues)

If one of the failure modes is detected, the solver attempts to fail gracefully by setting solver.ok = False with
control passing back to the user.  Additionally, if the solution is unphysical, solver.solutionError = True is set.

See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division, absolute_import
import numpy as np
import time
from . import tango_logging
from . import HToMatrixFD
from . import datasaver
from . import handlers
from .utilities import util



class Solver(object):
    def __init__(self, L, x, profileIC, profileRightBC, tArray, maxIterations, tol, compute_all_H, turbhandler):
        self.L = L        # size of domain
        self.x = x        # grid (array)
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.profile = profileIC                # initial condition for the profile (array)
        self.profile_mminus1 = profileIC        # initialize "m minus 1" variable
        self.profileRightBC = profileRightBC    # dirichlet boundary condition
        self.tArray = tArray                    # times t at which the solution is desired (array).  First value is initial condition
        self.maxIterations = maxIterations      # maximum number of iterations per timestep (scalar)
        self.tol = tol                          # tolerance for iteration convergence (scalar)
        assert callable(compute_all_H), "compute_all_H must be callable (e.g., a function)"
        self.compute_all_H = compute_all_H
        self.turbhandler = turbhandler          # why do I need turbhandler as input?  I only use it here to get_EWMA_params.  I don't like this; need better design.  What if there is no turbhandler?
        
        # initialize other instance data
        # DataSaverHandler and fileHandlerExecutor are two different ways of dealing with saving files on disk (and have different goals)
        self.dataSaverHandler = datasaver.DataSaverHandler()
        self.fileHandlerExecutor = handlers.Executor()
        
        self.solutionError = False
        self.t = 0
        self.dt = None        
        self.l = None                                         # Iteration index
        self.converged = None
        self.errHistory = np.zeros(self.maxIterations)      # error history vs. iteration at a given timestep
        self.errHistoryFinal = None
        self.tFinal = tArray[-1]
        self.m = 0                                            # Timestep index
        self.reachedEnd = False
    
    def take_timestep(self):
        # Implicit time advance: iterate to solve the nonlinear equation!
        self.m += 1
        self.converged = False
        (self.t, self.dt) = self._get_tdt(self.tArray, self.m)
        
        self.l = 0   # reset iteration counter
        self.errHistory[:] = 0
        tango_logging.info("Timestep m={}:  Beginning iteration loop ...".format(self.m))
        
        # compute next iteration
        
        while self.l < self.maxIterations and not self.converged and not self.solutionError:
            self.compute_next_iteration()

        if self.converged:            
            tango_logging.info("Timestep m={}:  Converged!  Successfully found the solution for t={}.  Rms error={}.  Took {} iterations.".format(self.m, self.t, self.errHistory[self.l-1], self.l))
            if self.m >= len(self.tArray) - 1:
                tango_logging.info("Reached the final timestep m={} at t={}.  Simulation ending...".format(self.m, self.t))
                self.reachedEnd = True
        
        
        ##### Section for saving data ####
        self.errHistoryFinal = self.errHistory[0:self.l]
        (EWMAParamTurbFlux, EWMAParamProfile) = self.turbhandler.get_ewma_params()
        
        timestepData = {'x': self.x, 'profile_m': self.profile, 'profile_mminus1': self.profile_mminus1, 'errhistory': self.errHistoryFinal, 't': self.t, 'm': self.m,
                        'EWMAParamTurbFlux':EWMAParamTurbFlux,  'EWMAParamProfile':EWMAParamProfile}
        self.dataSaverHandler.add_one_off_data(timestepData)
        self.dataSaverHandler.save_to_file(self.m)
        
        # Reset if another timestep is about to come.  If not, data is preserved for access at end.
        if self.ok:
            self.dataSaverHandler.reset_for_next_timestep()
            self.fileHandlerExecutor.reset_handlers_for_next_timestep()
        ##### End of section for saving data ##### 
        
        self.profile_mminus1 = self.profile

    def compute_next_iteration(self):
        startTime = time.time()
        
        # compute H's from current iterate of profile
        (H1, H2, H3, H4, H6, H7, extraturbdata) = self.compute_all_H(self.t, self.x, self.profile)
        
        # compute matrix system (A, B, C, f)
        (A, B, C, f) = HToMatrixFD.H_to_matrix(self.dt, self.dx, self.profileRightBC, self.profile_mminus1, H1, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7)

        (self.converged, rmsError, resid) = self.check_convergence(A, B, C, f, self.profile, self.tol)
        self.errHistory[self.l] = rmsError
        
        # compute new iterate of profile
        self.profile = HToMatrixFD.solve(A, B, C, f)
        
        # some output information
        endTime = time.time()
        durationHMS = util.duration_as_hms(endTime - startTime)
        tango_logging.debug("...iteration {} took a wall time of {}".format(self.l, durationHMS))
        tango_logging.debug("Timestep m={}: after iteration number l={}, first 4 entries of profile={};  last 4 entries={}".format(
            self.m, self.l, self.profile[:4], self.profile[-4:]))  # debug
        
        # save data if desired
        datadict = self._pkgdata(H1=H1, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7, A=A, B=B, C=C, f=f, profile=self.profile, rmsError=rmsError, extradata=extraturbdata)
        
        self.dataSaverHandler.add_data(datadict, self.l)
        self.fileHandlerExecutor.execute_scheduled(datadict, self.l)
        
        # Check for NaNs or infs or negative values
        self.check_profile_is_valid(self.profile)
        
        # about to loop to next iteration l
        self.l += 1
        if self.l >= self.maxIterations:
            tango_logging.info("Timestep m={} and time t={}:  maxIterations ({}) reached.  Error={} while tol={}".format(self.m, self.t, self.maxIterations, rmsError, self.tol))  # warning
    
    @property
    def ok(self):
        """True unless a stop condition is reached."""
        if self.solutionError == True:
            return False
        elif self.m >= len(self.tArray) - 1:
            return False
        elif self.t >= self.tFinal:
            return False
        elif self.l >= self.maxIterations:
            return False
        else:
            return True
    
    def check_profile_is_valid(self, profile):
        """Check the profile for validity: check for NaNs, infinities, negative numbers
        """
        if np.all(np.isfinite(profile)) == False:
            tango_logging.error("NaN or Inf detected in profile at l={}.  Aborting...".format(self.l))
            self.solutionError = True
        if np.any(profile < 0) == True:
            tango_logging.error("Negative value detected in profile at l={}.  Aborting...".format(self.l))
            self.solutionError = True

    def check_convergence(self, A, B, C, f, profile, tol):
        # convergence check: is || ( M[n^l] n^l - f[n^l] ) / max(abs(f)) || < tol
        #   where n^l is solved from M[n^{l-1}] n^l = f[n^{l-1}]
        # could add more convergence checks
        resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
        resid = resid / np.max(np.abs(f))  # normalize residuals
        rmsError = np.sqrt( 1/len(resid) * np.sum(resid**2))  
        converged = False
        if rmsError < tol:
            converged = True
        return (converged, rmsError, resid)
    
    @staticmethod
    def _get_tdt(tArray, timestepNumber):
        """Compute the next time and timestep for an already-defined time array.
        
        For example, if tArray = [0 2.5 10], then for timestep_number = 1, we are looking for the solution at
        t[1]=2.5, so t_new = 2.5 and dt = 2.5.  If timestep_number = 2, we are looking for the solution at t[2],
        so t_new = 10 and dt = 7.5.
        
        Inputs:
          tArray                times at which solution is desired.  Initial time should be first element (array)
          timestepNumber        timestep number corresponding to the desired solution time (integer)
        Outputs:
          tNew                  Desired solution time (scalar)
          dt                    time difference between desired solution time and previous time (scalar)
        """
        tNew = tArray[timestepNumber]
        tOld = tArray[timestepNumber - 1]
        dt = tNew - tOld
        return tNew, dt
        
    def _pkgdata(self, H1=None, H2=None, H3=None, H4=None, H6=None, H7=None, A=None, B=None, C=None, f=None, profile=None, rmsError=None, extradata=None):
        """input dict extradata contains the following [see Hcontrib_turbulent_flux in lodestro_method.py]:
        'x': data['x'], 'xTurbGrid': data['xTurbGrid'],
        'D': data['D'], 'c': data['c'],
        'profileTurbGrid': data['profileTurbGrid'], 'profileEWMATurbGrid': data['profileEWMATurbGrid'],
        'fluxTurbGrid': data['fluxTurbGrid'], 'smoothedFluxTurbGrid': data['smoothedFluxTurbGrid'], 'fluxEWMATurbGrid': data['fluxEWMATurbGrid'],
        'DTurbGrid': data['DTurbGrid'], 'cTurbGrid': data['cTurbGrid'],
        'DHatTurbGrid': data['DHatTurbGrid'], 'cHatTurbGrid': data['cHatTurbGrid'], 'thetaTurbGrid': data['thetaTurbGrid']}
        """
        data1 = {'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'H6': H6, 'H7': H7, 'A': A, 'B': B, 'C': C, 'f': f, 'profile': profile, 'rmsError': rmsError}
        pkgData = self._merge_two_dicts(extradata, data1)
        return pkgData

    @staticmethod            
    def _merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict using a shallow copy."""
        z = x.copy()
        z.update(y)
        return z