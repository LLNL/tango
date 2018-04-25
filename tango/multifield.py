"""
multifield

Define class that is used as the base representation for a "field" in Tango.      
"""
from __future__ import division
import numpy as np


class Field(object):
    """representation of data for a "field".  Tango solves a transport equation for each field.    
    """
    def __init__(self, label=None, rightBC=None, profile_mminus1=None, coupledTo=None, compute_all_H=None, gridMapper=None, lodestroMethod=None):
        if label is None:
            raise ValueError('label is a required input when creating a Field.')
        self.label = label
        self.rightBC = rightBC
        self.profile_mminus1 = profile_mminus1
        self.coupledTo = coupledTo
        self.compute_all_H = compute_all_H
        
        if gridMapper is not None:
            assert hasattr(gridMapper, 'map_profile_onto_turb_grid') and callable(getattr(gridMapper, 'map_profile_onto_turb_grid'))
            assert hasattr(gridMapper, 'map_transport_coeffs_onto_transport_grid') and callable(getattr(gridMapper, 'map_transport_coeffs_onto_transport_grid'))
            self.gridMapper = gridMapper
        else:
            self.gridMapper = GridsNull()
        
        self.lodestroMethod = lodestroMethod

def get_field_by_label(fields, label):
    """Given a list of fields and a label, return the field corresponding to that label.
    
    There is no checking for duplicate labels.  In the case of a duplicate, the first match is returned.
    
    Inputs:
      fields        collection of fields (list)
      label         field identifier (string)
      
    Outputs:
      field         the field in input that has field.label == label
    """
    field = next((f for f in fields if f.label == label), None)
    if field is None:
        raise ValueError('field corresponding to the given label is not found.')
    return field
        
def check_fields_label(fields):
    """check that labels on fields are unique.  If unique, do nothing.  If not, raise exception."""
    labels = [field.label for field in fields]
    if len(labels) != len(set(labels)):
        return False
    return True

def check_fields_rightBC(fields):
    """check that rightBC is a scalar for each field.  If so, do nothing.  If not, raise exception."""
    for field in fields:
        if not np.isscalar(field.rightBC):
            return False
    return True

def check_fields_profile_mminus1(fields):
    """check that profile_mminus1 is an array of the same length for each field."""
    N = len(fields[0].profile_mminus1)
    for field in fields:
        if field.profile_mminus1 is None or len(field.profile_mminus1) != N:
            return False
    return True

def check_fields_coupled_to(fields):
    """check that all fields have coupledTo equal to None, or to a partner that has them coupled back."""
    for field in fields:
        if field.coupledTo is not None:  # has a partner
            partnerFieldLabel = field.coupledTo
            if partnerFieldLabel == field.label:
                return False  # a field cannot be "coupled" to itself
            partnerField = next((f for f in fields if f.label == partnerFieldLabel), None)
            if partnerField is None:  # partner field doesn't exist
                return False
            else:   # partner field exists; check that it is coupled to field
                if partnerField.coupledTo != field.label:
                    return False
    return True

def check_fields_compute_all_H(fields):
    """check that all fields have a callable compute_all_H."""
    for field in fields:
        if not callable(getattr(field, 'compute_all_H')):
            return False
    return True
    
def check_fields_initialize(fields):
    """Run checks on fields and raise exception if a check does not pass.  If all checks pass, do nothing.
    
    Meant to be run once.
    """
    if not check_fields_label(fields):
        raise ValueError('labels of fields are not unique.')
    if not check_fields_rightBC(fields):
        raise ValueError('rightBC must be a scalar for each field.')
    if not check_fields_profile_mminus1(fields):
        raise ValueError('profile_mminus1 must have the same length for all fields.')
    if not check_fields_coupled_to(fields):
        raise ValueError('all fields must either be coupledTo None or a partner.')
    if not check_fields_compute_all_H(fields):
        raise ValueError('all fields must have a callable compute_all_H.')
    
class HCoefficients(object):
    """data container class for H coefficients.
    
    If entries are not specified, they are left as None (which represents zero, but does not require computation.
    """
    def __init__(self, H1=None, H2=None, H3=None, H4=None, H6=None, H7=None, H8=None):
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.H4 = H4
        self.H6 = H6
        self.H7 = H7
        self.H8 = H8
        
    def __add__(self, other):
        HList = ['H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8']
        Hdict = {}
        for Hj in HList:
            Hdict[Hj] = self._add_H(getattr(self, Hj), getattr(other, Hj))
        return HCoefficients(**Hdict)
            
    @staticmethod
    def _add_H(Ha, Hb):
        """Helper for the __add__ method
        
        Add a single H coefficient, returning None if both inputs are None.  If only one input is None, treat that
        input as zero.
        """
        if Ha is None:
            H = Hb
        else:
            if Hb is None:
                H = Ha
            else:
                H = Ha + Hb
        return H

    def __repr__(self):
        s = 'HCoefficients(\n'
        HList = ['H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8']
        for Hj in HList:
            if getattr(self, Hj) is not None:
                s += '{}={},\n'.format(Hj, getattr(self, Hj))
        s +=')'
        return s
                
        

# ComputeAllHAllFields that uses the individual compute_all_H_single_field (which is an attribute of the field)
class ComputeAllHAllFields(object):
    def __init__(self, fields, turbhandler=None):
        """Constructor.
        
        The default turbhandler=None is there as an option to allow the Tango machinery to be used even if the LoDestro method is not applied (that is,
        turbulent fluxes are not transformed into effective diffusive & convective contributions).  While this should be uncommon, because the primary
        purpose of Tango is to implement the LoDestro method, one may want to use Tango to solve linear diffusion equations, possibly for debugging or
        verification purposes.
        """
        self.fields = fields
        self.turbhandler = turbhandler
        
    def __call__(self, t, x, profiles, computeTurbulence=True, HCoeffsTurbAllFields=None):
        """
        The default computeTurbulence=True will call the fluxModel and compute new turbulent fluxes and transform these into new effective diffusive
        and convective contributions (and HCoefficients).  However, by setting computeTurbulence=False, one can provide HCoefficients corresponding
        to the turbulent contributions.  This is useful if one wants to implement an inner iteration loop, where an expensive turbulence simulation is
        performed seldomly at the outer iteration loop, and other nonlinear terms that are faster to compute are converged in the inner iteration loop.
        
        Inputs:
          t                         value of time at which to evaluate the H (scalar)
          x                         independent coordinate grid (array)
          profiles                  container of profiles, accessed by label (dict)
          computeTurbulence         If True, use the fluxModel to compute new turbulent fluxes.  If False, use the input HCoeffsTurbAllFields. (boolean)
          HCoeffsTurbAllFields      used if computeTurbulence==True.  container of HCoeffs for the turbulent contribution, accessed by label (dict)
        
        Outputs:
          HCoeffsAllFields          container of HCoeffs, accessed by label (dict)
          HCoeffsTurbAllFieldsOut   container of HCoeffs for the turbulent contribution, accessed by label (dict)
          extradataAllFields        container for extra data that may be useful for debugging (dict)
        """
        if computeTurbulence:
            # get the turbulent flux and transform into H coefficients
            if self.turbhandler is not None:
                (HCoeffsTurbAllFields, extradataAllFields) = self.turbhandler.turbflux_to_Hcoeffs_multifield(self.fields, profiles)
            else:
                extradataAllFields = None
        else:
            assert HCoeffsTurbAllFields is not None, 'must provide HCoeffsTurbAllFields when using computeTurbulence==False'
            extradataAllFields = None
            
        # iterate through individual fields to compute the other H coefficients of each field and build up a dict
        HCoeffsAllFields = {}
        for field in self.fields:
            if self.turbhandler is not None:
                HCoeffsAllFields[field.label] = field.compute_all_H(t, x, profiles, HCoeffsTurbAllFields[field.label])
            else:
                HCoeffsAllFields[field.label] = field.compute_all_H(t, x, profiles)
            
        return (HCoeffsAllFields, HCoeffsTurbAllFields, extradataAllFields)

        


class GridsNull(object):
    """Null class for moving between grids when the turbulence grid will be the same as the transport grid.
    No interpolation of quantities between grids will be performed, as there is only one grid.
    """
    def __init__(self):
        pass
#        self.x = x
    def map_profile_onto_turb_grid(self, profile):
        return profile
    def map_transport_coeffs_onto_transport_grid(self, D, c):
        return (D, c)
#    def get_x_transport_grid(self):
#        return self.x
#    def get_x_turbulence_grid(self):
#        return self.x