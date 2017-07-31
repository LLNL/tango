"""
solver_multifield

Module that handles the meat of solving a timestep in a Tango transport equation, including the inner
loop of iterating to solve a nonlinear equation.  Modification of solver.py to allow for multiple fields.

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
from . import datasaver
from . import handlers
from . import fieldgroups
from .utilities import util


#   turbhandler to get EWMA Params?  but there is one for each field.
class Solver(object):
    def __init__(self, L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, user_control_func=None):
        """Constructor
        
        Inputs:
          L                         size of domain (scalar)
          x                         independent coordinate grid (array)                    
          tArray                    times t at which the solution is desired (array).  First value is the initial condition
          maxIterations             maximum number of iterations per timestep (scalar)
          tol                       tolerance for iteration convergence (scalar)
          compute_all_H_all_fields  computes the H coefficients defining the linear transport equation for iteration (callable)
          fields                    collection of fields [list]
                                        Each field has properties:
                                            label               identifier for each field (string)
                                            rightBC             dirichlet boundary condition for the profile at the right (scalar)
                                            profile_mminus1     initial condition of the profile, used to initialize "m minus 1" variable (array)
                                            coupledTo           identifier for field that a field is coupled to (string or None)
                                            compute_all_H       callable that provides the H coefficients defining the transport equation (callable)
          user_control_func         [optional] user-supplied function for customizing control (function).  Run once per iteration, at the beginning.
                                        The function must take one argument, and that is the Solver object.
        """
        self.L = L
        self.x = x
        self.N = len(x)
        self.dx = x[1] - x[0]
#        self.profile = profileIC
 #       self.profile_mminus1 = profileIC

        self.tArray = tArray
        self.maxIterations = maxIterations
        self.tol = tol
        self.compute_all_H_all_fields = compute_all_H_all_fields
        self.fields = fields
#        self.compute_all_H = compute_all_H
#        self.turbhandler = turbhandler
        if user_control_func is not None:
            assert callable(user_control_func), "user_control_func must be callable (e.g., a function)"
        self.user_control_func = user_control_func
        
        # set self.profiles to the profile_mminus1 variables
        self.profiles = {}
        for field in fields:
            self.profiles[field.label] = field.profile_mminus1
        
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
        
        # TODO:
        ##### Section for saving data ####
        self.errHistoryFinal = self.errHistory[0:self.l]
#        (EWMAParamTurbFlux, EWMAParamProfile) = self.turbhandler.get_ewma_params()
#        
#        timestepData = {'x': self.x, 'profile_m': self.profile, 'profile_mminus1': self.profile_mminus1, 'errhistory': self.errHistoryFinal, 't': self.t, 'm': self.m,
#                        'EWMAParamTurbFlux':EWMAParamTurbFlux,  'EWMAParamProfile':EWMAParamProfile}
#        self.dataSaverHandler.add_one_off_data(timestepData)
#        self.dataSaverHandler.save_to_file(self.m)
#        
#        # Reset if another timestep is about to come.  If not, data is preserved for access at end.
#        if self.ok:
#            self.dataSaverHandler.reset_for_next_timestep()
#            self.fileHandlerExecutor.reset_handlers_for_next_timestep()
        ##### End of section for saving data ##### 
        
        # set previous timestep profiles in fields
        for field in self.fields:
            field.profile_mminus1 = self.profiles[field.label]
    
    def compute_next_iteration(self):
        startTime = time.time()
        
        # Run the customized control function if passed in
        if self.user_control_func is not None:
            self.user_control_func(self)
        
        # iterate through all the fields, compute all the H's
        (HCoeffsAllFields, extradataAllFields) = self.compute_all_H_all_fields(self.t, self.x, self.profiles)
    
        
        # create fieldGroups from fields as prelude to the iteration step
        fieldGroups = fieldgroups.fields_to_fieldgroups(self.fields, HCoeffsAllFields)
   
        
        # discretize and compute matrix system [iterating over groups]
        for fieldGroup in fieldGroups:
            fieldGroup.matrixEqn = fieldGroup.Hcoeffs_to_matrix_eqn(self.dt, self.dx, fieldGroup.rightBC, fieldGroup.psi_mminus1, fieldGroup.HCoeffs)
            
        # check convergence
        (self.converged, rmsError, normalizedResids) = self.check_convergence(fieldGroups, self.profiles, self.tol)
        self.errHistory[self.l] = rmsError
        self.normalizedResids = normalizedResids # debugging
        
        
        # compute new iterate of profiles [iterating over groups]
        for fieldGroup in fieldGroups:
            fieldGroup.profileSolution = fieldGroup.solve_matrix_eqn(fieldGroup.matrixEqn)
            
        # get the profiles for the fields out of the fieldGroups, put into a dict of profiles
        self.profiles = fieldgroups.fieldgroups_to_profiles(fieldGroups)
        
        # some output information
        endTime = time.time()
        durationHMS = util.duration_as_hms(endTime - startTime)
        tango_logging.debug("...iteration {} took a wall time of {}".format(self.l, durationHMS))
        #TODO:
        #tango_logging.debug("Timestep m={}: after iteration number l={}, first 4 entries of profile={};  last 4 entries={}".format(
        #    self.m, self.l, self.profile[:4], self.profile[-4:]))
        
        #TODO:
        ## save data if desired
        # datadict = self._pkgdata(H1=H1, H2=H2, H3=H3, H4=H4, H6=H6, H7=H7, A=A, B=B, C=C, f=f, profile=self.profile, rmsError=rmsError, extradata=extraturbdata)
        
        # self.dataSaverHandler.add_data(datadict, self.l)
        # self.fileHandlerExecutor.execute_scheduled(datadict, self.l)
        
        # Check for NaNs or infs or negative values
        self.check_profiles_are_valid(self.profiles)
        
        # about to loop to next iteration l
        self.l += 1
        if self.l >= self.maxIterations:
            tango_logging.info("Timestep m={} and time t={}:  maxIterations ({}) reached.  Error={} while tol={}".format(self.m, self.t, self.maxIterations, rmsError, self.tol))
            
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
    
    def check_profiles_are_valid(self, profiles):
        """Check thes profile for validity: check for NaNs, infinities, negative numbers
        
        Assumes that profiles should never be negative.  This is valid for profiles like a density or a pressure,
        but is not a valid assumption for other types of profiles like angular momentum.  If or when angular
        velocity is added, this check will need to be more nuanced, more fine-grained.
        
        Inputs:
          profiles          collection of profiles, accessed by label (dict)
        """
        for label in profiles:
            profile = profiles[label]
            if np.all(np.isfinite(profile)) == False:
                tango_logging.error("field {}: NaN or Inf detected in profile at l={}.  Aborting...".format(label, self.l))
                self.solutionError = True
                break
            if np.any(profile < 0) == True:
                tango_logging.error("field {}: Negative value detected in profile at l={}.  Aborting...".format(label, self.l))
                self.solutionError = True
                break

    def check_convergence(self, fieldGroups, profiles, tol):
        """Compute the global residual over all fields.
        
        Inputs:
          fieldGroups           fieldgroup collection; each fieldgroup contains the matrix equation needed to compute the residual (list)
          profiles              collection of profiles, accessed by label, needed to compute the residual (dict)
          tol                   relative tolerance for convergence (scalar)
        
        Outputs:
          converged             True if error is smaller than the tolerance (boolean)
          rmsError              global residual of the iteration (scalar)
          normalizedResids      normalized residual for each field, accessed by label (dict)
        """
        # get the residuals from each field
        normalizedResids = {}
        
        for fieldGroup in fieldGroups:
            normalizedResid = fieldGroup.calculate_residual(fieldGroup.matrixEqn, profiles)
            for label in normalizedResid:
                # store the residual of each field
                normalizedResids[label] = normalizedResid[label]
        
        # compute a combined residual
        totalLength = 0
        sumOfSquaresResid = 0
        for label in normalizedResids:
            resid = normalizedResids[label]
            totalLength += len(resid)
            sumOfSquaresResid += np.sum( resid**2)
        rmsError = np.sqrt(1/totalLength * sumOfSquaresResid)
        
        converged = False
        if rmsError < tol:
            converged = True
        return (converged, rmsError, normalizedResids)
        
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