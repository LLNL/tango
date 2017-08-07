"""
multifield

Define class that is used as the base representation for a "field" in Tango.      
"""
from __future__ import division
import numpy as np
from collections import namedtuple

# miscellaneous functions that will / could be useful
#  this function refers to GENE, and should go in to GENEcomm
#  [Actually, there is no explicit reference to GENE.  Nevertheless, this doesn't seem like the appropriate place]
# Perhaps this should be a Function ?  Nah
#   --*** Might as well put this into GENEcomm ??? Nah...  the labels only exist in Tango core, not the other flux computation.
#   and for analytically specified functions, make it part of the Tango spec that fluxes come back in a dict, accessed by label.
def flux_array_to_dict(fluxArray, fluxOrderedLabels):
    """
    Take the m x N array of output of flux from GENE (more precisely, GENEcomm) and turn it into a dict for access by label    
    (where m is the number of flux outputs, N is the number of radial points)
    
    If m==1, the array is allowed to be a 1D array.
    
    The label should refer to the conserved quantity rather than the flux (i.e., 'ionpressure' rather than 'heat flux')
    
    For example, if the three rows of fluxArray are [particleflux; ionheatflux; electronheatflux], and the labels are 'density',
        'ionpressure, 'electronpressure', then from the output, one accesses fluxDict['ionpressure'] to retrieve ionheatflux.
    
    Inputs:
      fluxArray             Fluxes in array form, m x N (1D or 2D array)
      fluxOrderedLabels     ordered list with labels for fluxes (list)
                              --order must match fluxArray
    Outputs:
      fluxDict              Fluxes in dict form, accessed by label
    """
    fluxDict = {}
    if len(fluxOrderedLabels) != 1:
        for (j, label) in enumerate(fluxOrderedLabels):
            fluxDict[label] = fluxArray[j, :]
    else:
        label = fluxOrderedLabels[0]
        fluxDict[label] = fluxArray
    return fluxDict
    
def profile_dict_to_array(profileDict, profileOrderedLabels):
    """
    Take the dict of profiles, accessed by label, and turn it into an m x N array.
    (where m is the number of species, N is the number of radial points).
    
    If m==1, the output is a 1D array rather than a 2D array.
    
    For example, if profileDict is
        {'density': density,
         'electronpressure': electronpressure,
         'ionpressure': ionpressure}
    and orderedLabels is ['density', 'ionpressure', 'electronpressure'], then the output should be
    
    profileArray = 
       np.array([density, ionpressure, electronpressure])
    
    Inputs:
      profileDict               profiles in dict form, accessed by label (dict)
      profileOrderedLabels      ordered list with labels for profiles (list)
    Outputs:
      profileArray              profiles in array form, m x N (1D or 2D array)
    """
    m = len(profileOrderedLabels)
    N = len(profileDict[profileOrderedLabels[0]])
    if m != 1:
        profileArray = np.zeros((m, N))
        for (j, label) in enumerate(profileOrderedLabels):
            profileArray[j, :] = profileDict[label]
    else:  # only 1 field; use a 1D array
        label = profileOrderedLabels[0]
        profileArray = profileDict[label]
    return profileArray

class Field(object):
    """representation of data for a "field".  Tango solves a transport equation for each field.    
    """
    def __init__(self, label=None, rightBC=None, profile_mminus1=None, coupledTo=None, compute_all_H=None, gridMapper=None, lodestroMethod=None):
        if label is None:
            raise ValueError, 'label is a required input when creating a Field.'
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
        raise ValueError, 'field corresponding to the given label is not found.'
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
        raise ValueError, 'labels of fields are not unique.'
    if not check_fields_rightBC(fields):
        raise ValueError, 'rightBC must be a scalar for each field.'
    if not check_fields_profile_mminus1(fields):
        raise ValueError, 'profile_mminus1 must have the same length for all fields.'
    if not check_fields_coupled_to(fields):
        raise ValueError, 'all fields must either be coupledTo None or a partner.'
    if not check_fields_compute_all_H(fields):
        raise ValueError, 'all fields must have a callable compute_all_H.'
    
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
#  ew... do I want to assume that the LoDestro Method gets used?  This is not assumed by the 1-field version.
#    Related question:  should this go into the solver or the user file?  Or a multifield / computeH file?
# probably can go in multifield.py... or a compute_H.py file
class ComputeAllHAllFields(object):
    def __init__(self, fields, turbhandler=None):
        self.fields = fields
        self.turbhandler = turbhandler
        
    def __call__(self, t, x, profiles):
        """
        Inputs:
          t                     value of time at which to evaluate the H (scalar)
          x                     independent coordinate grid (array)
          profiles              container of profiles, accessed by label (dict)
        Outputs:
          HCoeffsAllFields      container of HCoeffs, accessed by label (dict)
          extradataAllFields    container for extra data that may be useful for debugging (dict)
        """
        # get the turbulent flux and transform into H coefficients
        if self.turbhandler is not None:
            (HCoeffsTurbAllFields, extradataAllFields) = self.turbhandler.turbflux_to_Hcoeffs_multifield(self.fields, profiles)
            
        # iterate through individual fields to compute the other H coefficients of each field and build up a dict
        HCoeffsAllFields = {}
        for field in self.fields:
            if self.turbhandler is not None:
                HCoeffsAllFields[field.label] = field.compute_all_H(t, x, profiles, HCoeffsTurbAllFields[field.label])
            else:
                HCoeffsAllFields[field.label] = field.compute_all_H(t, x, profiles)
            
        return (HCoeffsAllFields, extradataAllFields)
        

# example compute_all_H_single_field that would go in a driver script
class ComputeAllHIonPressure(object):
     def __init__(self, turbhandler, Vprime, minorRadius, majorRadius, A):
        self.turbhandler = turbhandler
        self.Vprime = Vprime
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        self.A = A
        
     def __call__(self, t, r, profiles, HCoeffsTurb):
        """Define the contributions to the H coefficients
        
        Inputs:
          t                 time (scalar)
          r                 radial coordinate in SI (array)
          profiles          container for profiles, accessed by label (dict)
          HCoeffsTurb       turbulent contribution to ion heat flux
        """
        pressure = profiles['pressure']
        H1 = 1.5 * self.Vprime
        
        # turbulent flux --- turbhandler now takes the flux itself as INPUT, rather than calling to compute it
        #(H2_turb, H3_turb, extradata) = self.turbhandler.Hcontrib_turbulent_flux(pressure, ionHeatFluxTurb)
        #other stuff
        
        # package it up in the data class and return
        HCoeffs = HCoefficients(H1=H1, H7=H7)
        HCoeffs = HCoeffs + HCoeffsTurb
        return (HCoeffs, extradata)
        
        # return statement
    
        #return (H1, H2, H3, H4, H6, H7, extradata)

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