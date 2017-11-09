"""
solver

Module that handles the meat of solving a timestep in a Tango transport equation, including the inner
loop of iterating to solve a nonlinear equation.  Modification of solver.py to allow for multiple fields.

Convergence of the iteration loop is determined through comparing the rms residual of the nonlinear transport
equation to a relative tolerance value tol.

When all of the timesteps are successfully completed, solver.ok changes from True to False.

Failure modes:
    --convergence tolerance not reached within the maximum number of iterations (maxIterations)
    --maximum number of iterations per "set" reached (shut down gracefully before a restart)
    --solution becomes unphysical (negative, infinite, or NaN vlaues)

If one of the failure modes is detected, the solver attempts to fail gracefully by setting solver.ok = False with
control passing back to the user.  Additionally, if the solution is unphysical, solver.solutionError = True is set.

See https://github.com/LLNL/tango for copyright and license information"""

from __future__ import division, absolute_import
import numpy as np
import time
from . import tango_logging
from . import handlers
from . import fieldgroups
from .utilities import util


#   turbhandler to get EWMA Params?  but there is one for each field.
class Solver(object):
    def __init__(self, L, x, tArray, maxIterations, tol, compute_all_H_all_fields, fields, 
                 startIterationNumber=0, profiles=None, maxIterationsPerSet=np.inf,
                 useInnerIteration=False, innerIterationMaxCount=20,
                 user_control_func=None):
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
          startIteration            [optional] Used for restarting   Set the starting iteration number, default 0 (int)
          profiles                  [optional] Used for restarting.  Set the initial profiles, default to field.profile_mminus1 (dict of arrays, accessed by label)
          maxIterationsPerSet       [optional] Used for shutting down gracefully for restarts.  Set the number of iterations that can be performed on this set. (int)
          useInnerIteration         [optional] If True, perform an inner iteration loop [need to specify loop parameters somewhere...] where the turbulent
                                       coefficients are held fixed but other nonlinear terms can be converged. (boolean)
          innerIterationMaxCount    [optional, default=20] Maximum number of iterations to use on the inner iteration loop (int)
          user_control_func         [optional] user-supplied function for customizing control (function).  Runs once per iteration, at the end.
                                        The function must take one argument, and that is the Solver object.
        """
        self.L = L
        self.x = x
        self.N = len(x)
        self.dx = x[1] - x[0]

        self.tArray = tArray
        self.maxIterations = maxIterations
        self.tol = tol
        self.compute_all_H_all_fields = compute_all_H_all_fields
        self.fields = fields
        self.useInnerIteration = useInnerIteration
        self.innerIterationMaxCount = innerIterationMaxCount
        if user_control_func is not None:
            assert callable(user_control_func), "user_control_func must be callable (e.g., a function)"
        self.user_control_func = user_control_func
        
        
        if profiles is None:
            # set self.profiles to the profile_mminus1 variables
            self.profiles = {}
            for field in fields:
                self.profiles[field.label] = field.profile_mminus1
        else:
            self.profiles = profiles
        # initialize other instance data
        self.fileHandlerExecutor = handlers.Executor()
        
        self.solutionError = False
        self.t = 0
        self.dt = None
        self.countStoredIterations = 0
        self.iterationNumber = startIterationNumber
        self.maxIterationsPerSet = maxIterationsPerSet
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
        
        self.errHistory[:] = 0
        tango_logging.info("Timestep m={}:  Beginning iteration loop ...".format(self.m))
        
        # compute next iteration
        while (self.iterationNumber < self.maxIterations
               and self.countStoredIterations < self.maxIterationsPerSet
               and not self.converged
               and not self.solutionError):
            self.compute_next_iteration()

        if self.converged:            
            tango_logging.info("Timestep m={}:  Converged!  Successfully found the solution for t={}.  Rms error={}.  Took {} iterations.".format(
                    self.m, self.t, self.errHistory[self.iterationNumber-1], self.iterationNumber))
            if self.m >= len(self.tArray) - 1:
                tango_logging.info("Reached the final timestep m={} at t={}.  Simulation ending...".format(self.m, self.t))
                self.reachedEnd = True
                
        self.errHistoryFinal = self.errHistory[0:self.countStoredIterations]
#        
#        # Reset if another timestep is about to come.  If not, data is preserved for access at end.
#        if self.ok:
#            self.fileHandlerExecutor.reset_handlers_for_next_timestep()
        ##### End of section for saving data ##### 
        
        # set previous timestep profiles in fields
        for field in self.fields:
            field.profile_mminus1 = self.profiles[field.label]
        self.iterationNumber = 0  # reset iteration counter for next timestep
        self.countStoredIterations = 0    
        
    def compute_next_iteration(self):
        startTime = time.time()
        
        # Run the customized control function if passed in
        if self.user_control_func is not None:
            self.user_control_func(self)
        
        index = self.countStoredIterations
            
        # iterate through all the fields, compute all the H's [runs the turbulence calculation]
        (HCoeffsAllFields, HCoeffsTurbAllFields, extradataAllFields) = self.compute_all_H_all_fields(self.t, self.x, self.profiles)
    
        # create fieldGroups from fields as prelude to the iteration step
        fieldGroups = fieldgroups.fields_to_fieldgroups(self.fields, HCoeffsAllFields)
        
        # discretize and compute matrix system [iterating over groups]
        for fieldGroup in fieldGroups:
            fieldGroup.matrixEqn = fieldGroup.Hcoeffs_to_matrix_eqn(self.dt, self.dx, fieldGroup.rightBC, fieldGroup.psi_mminus1, fieldGroup.HCoeffs)
            
        # check convergence
        (self.converged, rmsError, normalizedResids) = self.check_convergence(fieldGroups, self.profiles, self.tol)
        self.errHistory[index] = rmsError
        self.normalizedResids = normalizedResids
        
        
        # compute new iterate of profiles [iterating over groups]
        for fieldGroup in fieldGroups:
            fieldGroup.profileSolution = fieldGroup.solve_matrix_eqn(fieldGroup.matrixEqn)
            
        # get the profiles for the fields out of the fieldGroups, put into a dict of profiles
        self.profiles = fieldgroups.fieldgroups_to_profiles(fieldGroups)
        
        
        # ADD INNER ITERATION LOOP HERE
        if self.useInnerIteration:
            # self.profiles is set internally
            (HCoeffsAllFields, ignore, ignore) = self.perform_inner_iteration(HCoeffsTurbAllFields)
        
        # some output information
        endTime = time.time()
        durationHMS = util.duration_as_hms(endTime - startTime)
        tango_logging.debug("...iteration {} took a wall time of {}".format(self.iterationNumber, durationHMS))

        #tango_logging.debug("Timestep m={}: after iteration number l={}, first 4 entries of profile={};  last 4 entries={}".format(
        #    self.m, self.l, self.profile[:4], self.profile[-4:]))
        
        # save data if desired
        datadict = self._pkgdata(
                HCoeffsAllFields=HCoeffsAllFields, extradataAllFields=extradataAllFields, profiles=self.profiles, normalizedResids=normalizedResids,
                rmsError=rmsError, iterationNumber=self.iterationNumber)
        self.datadict=datadict
        self.fileHandlerExecutor.execute_scheduled(datadict, self.iterationNumber)
        
        # Check for NaNs or infs or negative values
        self.check_profiles_are_valid(self.profiles)
        
        # about to loop to next iteration
        self.countStoredIterations += 1
        self.iterationNumber += 1
        if self.iterationNumber >= self.maxIterations:
            tango_logging.info("Timestep m={} and time t={}:  maxIterations ({}) reached.  Error={} while tol={}".format(self.m, self.t, self.maxIterations, rmsError, self.tol))
        if self.countStoredIterations >= self.maxIterationsPerSet:
            tango_logging.info("Timestep m={} and time t={}:  maxIterationsPerSet ({}) reached.  Error={} while tol={}.  Shutting down...".format(self.m, self.t, self.maxIterationsPerSet, rmsError, self.tol))

    def perform_inner_iteration(self, HCoeffsTurbAllFields):
        """Todo: Need to set some inner iteration parameters.
        
        Finish condition: fixed number of iterations.  (Have not implemented a tolerance, yet)
        """
        innerIterationCounter = 0
        innerIterationDone = False
        while not innerIterationDone:
            # iterate through all the fields, compute all the H's
            (HCoeffsAllFields, ignore, ignore) = self.compute_all_H_all_fields(self.t, self.x, self.profiles, computeTurbulence=False, HCoeffsTurbAllFields=HCoeffsTurbAllFields)
            
            # create fieldGroups from fields as prelude to the iteration step
            fieldGroups = fieldgroups.fields_to_fieldgroups(self.fields, HCoeffsAllFields)
            
            # discretize and compute matrix system [iterating over groups]
            for fieldGroup in fieldGroups:
                fieldGroup.matrixEqn = fieldGroup.Hcoeffs_to_matrix_eqn(self.dt, self.dx, fieldGroup.rightBC, fieldGroup.psi_mminus1, fieldGroup.HCoeffs)
            
            # check convergence
            (ignore, rmsError, normalizedResids) = self.check_convergence(fieldGroups, self.profiles, self.tol)
            
            # compute new iterate of profiles [iterating over groups]
            for fieldGroup in fieldGroups:
                fieldGroup.profileSolution = fieldGroup.solve_matrix_eqn(fieldGroup.matrixEqn)
            
            # get the profiles for the fields out of the fieldGroups, put into a dict of profiles
            self.profiles = fieldgroups.fieldgroups_to_profiles(fieldGroups)
            
            # finish up
            innerIterationCounter += 1
            # check if done
            if innerIterationCounter >= self.innerIterationMaxCount:
                innerIterationDone = True
            # tolerance?
            
        return (HCoeffsAllFields, normalizedResids, rmsError)
    @property
    def ok(self):
        """True unless a stop condition is reached."""
        if self.solutionError == True:
            return False
        elif self.m >= len(self.tArray) - 1:
            return False
        elif self.t >= self.tFinal:
            return False
        elif self.iterationNumber >= self.maxIterations:
            return False
        elif self.countStoredIterations >= self.maxIterationsPerSet:
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
                tango_logging.error("field {}: NaN or Inf detected in profile at l={}.  Aborting...".format(label, self.iterationNumber))
                self.solutionError = True
                break
            if np.any(profile < 0) == True:
                tango_logging.error("field {}: Negative value detected in profile at l={}.  Aborting...".format(label, self.iterationNumber))
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
            sumOfSquaresResid += np.sum(resid**2)
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
    
    def _pkgdata(self, HCoeffsAllFields=None, extradataAllFields=None, profiles=None, normalizedResids=None, rmsError=None, iterationNumber=None):
        """Package all the data into the desired data structure for saving.
        
        Several inputs are individual dicts, accessed by label.  The desired output has each of the field-specific data in a
        single sub-dictionary of the overall datadict.
        
        The input dict extradataAllFields contains a dictionary for each field, accessed by label.  Each subdictionary contains the
        following [see turbflux_to_Hcoeffs_multifield in lodestro_method.py]:
        
          'D': data['D'], 'c': data['c'],
          'profileTurbGrid': data['profileTurbGrid'], 'profileEWMATurbGrid': data['profileEWMATurbGrid'],
          'fluxTurbGrid': data['fluxTurbGrid'], 'smoothedFluxTurbGrid': data['smoothedFluxTurbGrid'], 'fluxEWMATurbGrid': data['fluxEWMATurbGrid'],
          'DTurbGrid': data['DTurbGrid'], 'cTurbGrid': data['cTurbGrid'],
          'DHatTurbGrid': data['DHatTurbGrid'], 'cHatTurbGrid': data['cHatTurbGrid'], 'thetaTurbGrid': data['thetaTurbGrid']}
            
        Inputs:
          HCoeffsAllFields          HCoeffs for each field, accessed by label (dict)
          extradataAllFields        turbulence data for each field, accessed by label (dict)
          profiles                  current profile for each field, accessed by label (dict)
          normalizedResids          residual array for each field, accessed by label (dict)
          rmsError                  current residual, calculated over all fields (scalar)
          iterationNumber           current iteration number (int)
        
        Outputs:
          data                      data for saving (dict)
        """
        data = {'errHistory': rmsError, 'iterationNumber': iterationNumber}
        for label in HCoeffsAllFields:
            data[label] = {}
            data[label]['profile'] = profiles[label]
            data[label]['normalizedResid'] = normalizedResids[label]
            if extradataAllFields is not None:
                data[label] = self._merge_two_dicts(data[label], extradataAllFields[label])
            # store Hdata into data
            self._pkgdata_HCoeffs_helper(data, HCoeffsAllFields, label)            
        return data
        
    def _pkgdata_HCoeffs_helper(self, data, HCoeffsAllFields, label):
        """Add the HCoeffs into the data dict (mutates it in place).
        
        If an Hj is None, it is stored as an array of zeros.
        
        Inputs:
          data                  data (dict)
          HCoeffsAllFields      HCoeffs for each field, accessed by label (dict)
          label                 current label (string)
        """
        HList = ['H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8']
        for HjName in HList:
            Hj = getattr(HCoeffsAllFields[label], HjName)
            if Hj is not None:
                data[label][HjName] = Hj
            else:
                data[label][HjName] = np.zeros_like(HCoeffsAllFields[label].H1)

    @staticmethod            
    def _merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict using a shallow copy.
        
        Note: when moving to Python 3, this can be replaced by ChainMap.
        """
        z = x.copy()
        z.update(y)
        return z