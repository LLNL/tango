"""
fieldGroups

Define an abstract class fieldGroup describing how the Tango transport solver will solve the linear iteration equation.  This class
is used when the iteration equation is solved by sequential iterations through multiple fields, or in this case, fieldGroups.  A
fieldGroup may consist of one field, two coupled fields, or more fields.

Implementations for one field and two coupled fields are provided.


Each fieldGroup has several methods:
    --H coeffs to matrix equation
    --solve matrix equation
    

See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import numpy as np
from collections import namedtuple
#from enum import Enum

from . import HToMatrixFD
from . import JK_to_matrix_fd
from . import multifield

#kind = Enum('kind', ['uncoupled', 'paircoupled'])

class FieldGroup(object):
    """Abstract superclass for FieldGroups.  Solve 1D ODE BVPs.
    
    Attributes:
      label         identifier to label the meaning of the field
                      (string) if uncoupled
                      (tuple of 2 strings) if paircoupled
    """
    def __init__(self, label):
        # define attributes
        self.matrixEqn = None
        self.HCoeffs = None
        self.profileSolution = None
        self.rightBC = None
        self.psi_mminus1 = None
        self.fieldsolution = None
        
    def Hcoeffs_to_matrix_eqn(self, dt, dx, rightBC, psi_mminus1, HCoeffs):
        """Subclasses must implement this."""
        pass
    
    def solve_matrix_eqn(self, matrixEqn):
        """Subclasses must implement this."""
        pass
    
    def calculate_residual(self, matrixEqn, profile):
        """Subclasses must implement this.
        
        Each fieldgroup solves the linear equation (in abstract, undiscretized form)
            H^{l-1} [U^l] = f^{l-1}
        for the new profile iterate U^l, where H here is some linear operators on the unknown U^l.  The H^{l-1} may be computed from U^l.
        
        One can define a residual by computing the new H^l and f^l using U^l:
            r^l = H^l [U^l] - f^l
        and a normalized residual
            rnorm^l = r^l / max(abs(f^l))
        """
    
UncoupledMatrixEqn = namedtuple('UncoupledMatrixEquation',
               ('A', 'B', 'C', 'f'))

class UncoupledFieldGroup(FieldGroup):
    """Class for uncoupled fields.  Solve the transport equation for a single field.
    
    The transport equation H coefficients are represented by H1, ..., H7 [with H8=0]
    
    The matrix equation is represented by (A, B, C, D) 1D arrays --- this is determined by HToMatrixFD.py
    """
    def __init__(self, label):
        self.label = label
    
    def Hcoeffs_to_matrix_eqn(self, dt, dx, rightBC, psi_mminus1, HCoeffs):
        """Convert H coefficients to marix equation for uncoupled fields.
        
        Use second-order finite difference.
        
        Inputs:
          dt                timestep (scalar)
          dx                grid spacing (scalar)
          rightBC           boundary condition at right edge (scalar)
          psi_mminus1       solution U_{m-1} at the previous timestep (array)
          HCoeffs           container for the H coefficients (HCoefficients)
        
        Outputs:
          matrixEqn         object representing the matrix equation (UncoupledMatrixEqn)
        """
        (A, B, C, f) = HToMatrixFD.H_to_matrix(dt, dx, rightBC, psi_mminus1, 
                                               H1=HCoeffs.H1, H2=HCoeffs.H2, H3=HCoeffs.H3, H4=HCoeffs.H4, H6=HCoeffs.H6, H7=HCoeffs.H7)
        matrixEqn = UncoupledMatrixEqn(A, B, C, f)
        return matrixEqn
    
    def solve_matrix_eqn(self, matrixEqn):
        """
        Solve the matrix equation for an uncoupled field group
        
        Inputs:
          matrixEqn             object representing the discretized equations (UncoupledMatrixEqn)
            
        Outputs:
          profileSolution       solutions for the new profiles in a dict, with label as key (dict)
        """
        A = matrixEqn.A
        B = matrixEqn.B
        C = matrixEqn.C
        f = matrixEqn.f
        profile = HToMatrixFD.solve(A, B, C, f)
        
        # create dict container to return solution
        profileSolution = {}
        profileSolution[self.label] = profile
        return profileSolution
        
    def calculate_residual(self, matrixEqn, profiles):
        """Calculate the residual for an uncoupled field group.
        
        The residual is defined as
            r^l = H^l [U^l] - f^l
        
        For the discretization in terms of the A, B, C, f representation of the matrix equation, see HToMatrixFD.py            
            
        Inputs:
          matrixEqn             object representing the discretized equations (UncoupledMatrixEqn)
          profiles              collection of field profiles, accessed by label (dict)
        Outputs:
          normalizedResid       container (dict?)
        """
        A, B, C, f = matrixEqn.A, matrixEqn.B, matrixEqn.C, matrixEqn.f
        profile = profiles[self.label]
        
        resid = A*np.concatenate((profile[1:], np.zeros(1))) + B*profile + C*np.concatenate((np.zeros(1), profile[:-1])) - f
        resid = resid / np.max(np.abs(f))  # normalize residuals
        
        normalizedResid = {}
        normalizedResid[self.label] = resid
        return normalizedResid
    

class JKCoefficients(object):
    """data container class for J,K coefficients.
    
    The J,K coefficients are simply the H coefficients (relabeled) when two fields are coupled together.
    """
    def __init__(self, J1=None, J2=None, J3=None, J4=None, J6=None, J7=None, J8=None,
                       K1=None, K2=None, K3=None, K4=None, K6=None, K7=None, K8=None):
        self.J1 = J1
        self.J2 = J2
        self.J3 = J3
        self.J4 = J4
        self.J6 = J6
        self.J7 = J7
        self.J8 = J8
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.K6 = K6
        self.K7 = K7
        self.K8 = K8

PairCoupledMatrixEqn = namedtuple('PairCoupledMatrixEqn',
               ('UEqnCoeffs', 'WEqnCoeffs'))
    
class PairCoupledFieldGroup(FieldGroup):
    """Class for pair-coupled fields.
    
    The transport equation H coefficients are represented by J1, ..., J8, K1, ..., K8
    
    The matrix equation is represented by (UEqnCoeffs, WEqnCoeffs) --- determined by JK_to_matrix_fd.py
    """
    def __init__(self, label0, label1):
        """ do something here"""
        self.label0 = label0
        self.label1 = label1
    
    def Hcoeffs_to_matrix_eqn(self, dt, dx, rightBC, psi_mminus1, JKCoeffs):
        """Convert H coefficients to matrix equation for pair-coupled fields.
        
        Use second-order finite difference.
        
        Inputs:
          dt                        timestep (scalar)
          dx                        grid spacing (scalar)
          rightBC                   boundary condition at right edge (UL, WL) (tuple of 2 scalars)
          psi_mminus1               the solution (U_{m-1}, W_{m-1}) at the previous timestep on the grid (tuple of 2 arrays)
          JKCoeffs                  container for the J,K coefficients (JKCoefficients)
            
        Outputs:
          matrixEqn                 object representing the matrix equation (PairCoupledMatrixEqn)
        """        
        (UEqnCoeffs, WEqnCoeffs) = JK_to_matrix_fd.JK_to_matrix(dt, dx, rightBC, psi_mminus1,
                                    J1=JKCoeffs.J1, J2=JKCoeffs.J2, J3=JKCoeffs.J3, J4=JKCoeffs.J4, J6=JKCoeffs.J6, J7=JKCoeffs.J7, J8=JKCoeffs.J8,
                                    K1=JKCoeffs.K1, K2=JKCoeffs.K2, K3=JKCoeffs.K3, K4=JKCoeffs.K4, K6=JKCoeffs.K6, K7=JKCoeffs.K7, K8=JKCoeffs.K8)
        matrixEqn = PairCoupledMatrixEqn(UEqnCoeffs, WEqnCoeffs)
        return matrixEqn
    
    def solve_matrix_eqn(self, matrixEqn):
        """
        Solve the matrix equation for a pair-coupled field group
        
        Inputs:
          matrixEqn             object representing the discretized equations (PairCoupledMatrixEqn)
            
        Outputs:
          profileSolution       solutions for the new profiles in a dict, with labels as keys (dict)
            
        """
        UEqnCoeffs = matrixEqn.UEqnCoeffs
        WEqnCoeffs = matrixEqn.WEqnCoeffs
        (U, W) = JK_to_matrix_fd.solve(UEqnCoeffs, WEqnCoeffs)
        
        # create dict container to return solution
        profileSolution = {}
        profileSolution[self.label0] = U
        profileSolution[self.label1] = W
        return profileSolution
        
    def calculate_residual(self, matrixEqn, profiles):
        """Calculate the residual for an pair-coupled field group.
        
        The residual is defined as
            r^l = H^l [U^l] - f^l
        
        For the discretization and explanation of the representation of the matrix equation, see JK_to_matrix_fd.py            
            
        Inputs:
          matrixEqn             object representing the discretized equations (UncoupledMatrixEqn)
          profile               collection of field profiles, accessed by label (dict)
        Outputs:
          normalizedResid       dict containing arrays of normalized residual for each field, accessed by label [dict]
        """
        UEqnCoeffs, WEqnCoeffs = matrixEqn.UEqnCoeffs, matrixEqn.WEqnCoeffs
        AU, BU, CU, DU, GW = UEqnCoeffs.A, UEqnCoeffs.B, UEqnCoeffs.C, UEqnCoeffs.D, UEqnCoeffs.G
        AW, BW, CW, DW, GU = WEqnCoeffs.A, WEqnCoeffs.B, WEqnCoeffs.C, WEqnCoeffs.D, WEqnCoeffs.G
        
        # C_j^U U_{j-1}  +  B_j^U U_j  +  A_j^U U_{j+1} + G_j^W W_j = D_j^U
        U = profiles[self.label0]
        W = profiles[self.label1]
        residU = AU*np.concatenate((U[1:], np.zeros(1))) + BU*U + CU*np.concatenate((np.zeros(1), U[:-1])) + GW*W - DU
        residU = residU / np.max(np.abs(DU))  # normalize residuals
        
        residW = AW*np.concatenate((W[1:], np.zeros(1))) + BW*W + CW*np.concatenate((np.zeros(1), W[:-1])) + GU*U - DW
        residW = residW / np.max(np.abs(DW))  # normalize residuals
        
        normalizedResid = {}
        normalizedResid[self.label0] = residU
        normalizedResid[self.label1] = residW
        return normalizedResid
        
    

def Hcoeffs_to_JKcoeffs(field1HCoeffs, field2HCoeffs):
    """For pair-coupled fields, package up the HCoeffs for the 2 fields separately into a combined container (JKcoeffs)
    
    Inputs:
        field1Label     identifier for field 1 (string)
        field1HCoeffs   HCoeffs for field 1
        field2Label     identifier for field 2 (string)
        field2HCoeffs   HCoeffs for field 2
        
    Outputs:
        JKCoeffs        JKCoeffs for field 1,2
    """
    return JKCoefficients(J1=field1HCoeffs.H1, J2=field1HCoeffs.H2, J3=field1HCoeffs.H3, J4=field1HCoeffs.H4, J6=field1HCoeffs.H6, J7=field1HCoeffs.H7, J8=field1HCoeffs.H8,
                          K1=field2HCoeffs.H1, K2=field2HCoeffs.H2, K3=field2HCoeffs.H3, K4=field2HCoeffs.H4, K6=field2HCoeffs.H6, K7=field2HCoeffs.H7, K8=field2HCoeffs.H8)        

def fields_to_fieldgroups(fields, HCoeffsAllFields):
    """Convert the list of fields & their HCoeffs into a list of fieldGroups.
    
    Some fieldGroups consist of multiple fields, if a field has been designated to be solved while coupled to another field.
    
    After creation, each fieldGroup in fieldGroups contains
        --the label(s) (string) of the field(s) the group represents
        --The HCoeffs container with the coefficients of the transport equation
        --right boundary condition of each field (scalar or 2-tuple)
        --psi_mminus1: the value at the previous timestep for each field (scalar or 2-tuple)
        
    
    Inputs:
      fields            fields that Tango is solving for [list of field]
      HCoeffsAllFields  data container for H Coeffs [dict; each item is object of type HCoefficients]
      
    Outputs:
      fieldGroups       groups of fields [list of fieldGroup]
    """
    fieldGroups = []
    fieldsAccountedFor = set([]) # set to keep track of which fields have been accounted for
    for field in fields:
        # 3 possibilities
        if field.coupledTo is None:    # 1. field is uncoupled
            # create the field group
            fg = UncoupledFieldGroup(field.label)
            # attach the Hcoeffs
            fg.HCoeffs = HCoeffsAllFields[field.label]
            # attach the boundary condition on the right side of the domain
            fg.rightBC = field.rightBC
            # attach the profile from the previous timestep
            fg.psi_mminus1 = field.profile_mminus1
            # add to the list
            fieldGroups.append(fg)
            fieldsAccountedFor.add(field.label)
        else:  # field is paircoupled
            if field.label not in fieldsAccountedFor and field.coupledTo not in fieldsAccountedFor:
                # 2. neither partner has yet been accounted for: add both
                partnerField = multifield.get_field_by_label(fields, field.coupledTo)
                # create the field group
                fg = PairCoupledFieldGroup(field.label, partnerField.label)
                # attach the HCoeffs
                fg.HCoeffs = Hcoeffs_to_JKcoeffs(HCoeffsAllFields[field.label], HCoeffsAllFields[partnerField.label])
                # attach the boundary conditions on the right side of the domain
                fg.rightBC = (field.rightBC, partnerField.rightBC)
                # attach the profiles from the previous timestep
                fg.psi_mminus1 = (field.profile_mminus1, partnerField.profile_mminus1)
                # add to the list              
                fieldGroups.append(fg)
                fieldsAccountedFor.add(field.label)
                fieldsAccountedFor.add(field.coupledTo)
            else:
                # 3. field and its partner have already been accounted for.  Do nothing.
                pass
        
    # need to write a unit test to check that this works!!!
    return fieldGroups
    
def fieldgroups_to_profiles(fieldGroups):
    """Extract the profiles from the fieldgroups after the solve for the new iterate has occurred.
    
    Inputs:
      fieldGroups        fieldGroup collection (list)
      
    Outputs:
      profiles           container with profile solutions (dict, accessed by label)
    """
    profiles = {}
    for fieldGroup in fieldGroups:
        for label in fieldGroup.profileSolution:
            profiles[label] = fieldGroup.profileSolution[label]
    return profiles