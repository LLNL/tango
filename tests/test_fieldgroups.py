"""Test fieldgroups.py"""

from __future__ import division
import numpy as np
from tango import HToMatrixFD, JK_to_matrix_fd
from tango import multifield
from tango import fieldgroups


def test_uncoupled_Hcoeffs_to_matrix():
    """Test Hcoeffs_to_matrix_eqn() method of the UncoupledFieldGroup class."""
    # create the UncoupledFieldGroup
    ufg = fieldgroups.UncoupledFieldGroup('default')
    
    # create the Hcoeffs, rightBC, psi_mminus1
    (dt, dx, rightBC, psi_mminus1, H1, H2, H7, n_ss_analytic, rtol) = one_field_diffusion_setup()
    Hcoeffs = multifield.HCoefficients(H1=H1, H2=H2, H7=H7)
    
    # discretize into matrix equation
    matrixEqn = ufg.Hcoeffs_to_matrix_eqn(dt, dx, rightBC, psi_mminus1, Hcoeffs)
    
    # matrix solve (using the underlying internals, not the object's method)
    n_solution = HToMatrixFD.solve(matrixEqn.A, matrixEqn.B, matrixEqn.C, matrixEqn.f)
    
    # check
    assert np.allclose(n_solution, n_ss_analytic, rtol=rtol)
    
def test_uncoupled_solve_matrix_eqn():
    """Test solve_matrix_eqn() method of the UncoupledFieldGroup class when solving a problem in polar coordinates."""
    # create the UncoupledFieldGroup
    label ='default'
    ufg = fieldgroups.UncoupledFieldGroup(label)
    
    # create the Hcoeffs, rightBC, psi_mminus1
    (dt, dx, rightBC, psi_mminus1, H1, H2, H7, n_ss_analytic, rtol) = one_field_diffusion_setup()
    HCoeffs = multifield.HCoefficients(H1=H1, H2=H2, H7=H7)
    
    # discretize into matrix equation
    matrixEqn = ufg.Hcoeffs_to_matrix_eqn(dt, dx, rightBC, psi_mminus1, HCoeffs)
    
    # matrix solve
    profileSolution = ufg.solve_matrix_eqn(matrixEqn)
    n_solution = profileSolution[label]
    
    # check
    assert np.allclose(n_solution, n_ss_analytic, rtol=rtol)
    
def test_uncoupled_polar_solve_matrix_eqn():
    """Test solve_matrix_eqn() method of the UncoupledFieldGroup class."""
    # create the UncoupledFieldGroup
    label ='default'
    ufg = fieldgroups.UncoupledFieldGroup(label)
    
    # create the Hcoeffs, rightBC, psi_mminus1
    (dt, dx, rightBC, psi_mminus1, H1, H2, H7, n_ss_analytic, rtol) = one_field_polar_setup()
    HCoeffs = multifield.HCoefficients(H1=H1, H2=H2, H7=H7)
    
    # discretize into matrix equation
    matrixEqn = ufg.Hcoeffs_to_matrix_eqn(dt, dx, rightBC, psi_mminus1, HCoeffs)
    
    # matrix solve
    profileSolution = ufg.solve_matrix_eqn(matrixEqn)
    n_solution = profileSolution[label]
    
    # check
    assert np.allclose(n_solution, n_ss_analytic, rtol=rtol)
    
def test_paircoupled_Hcoeffs_to_matrix():
    """Test Hcoeffs_to_matrix_eqn() method of the PairCoupledFieldGroup class."""
    # create the PairCoupledFieldGroup
    label0 = 'field0'
    label1 = 'field1'
    pfg = fieldgroups.PairCoupledFieldGroup(label0, label1)
    
    # create the Hcoeffs, rightBC, psi_mminus1
    (dt, dx, rightBC, psi_mminus1, J1, J2, J3, J6, J7, J8, K1, K2, K3, K6, K7, K8, psiAnalytic, rtol) = two_field_equation_setup()
    JKCoeffs = fieldgroups.JKCoefficients(J1=J1, J2=J2, J3=J3, J6=J6, J7=J7, J8=J8,
                                          K1=K1, K2=K2, K3=K3, K6=K6, K7=K7, K8=K8)
    
    # discretize into matrix equation
    matrixEqn = pfg.Hcoeffs_to_matrix_eqn(dt, dx, rightBC, psi_mminus1, JKCoeffs)
    
    # matrix solve (using the underlying internals, not the object's method)
    (field0Solution, field1Solution) = JK_to_matrix_fd.solve(matrixEqn.UEqnCoeffs, matrixEqn.WEqnCoeffs)
    
    # check
    assert np.allclose(field0Solution, psiAnalytic[0], rtol=rtol)    
    assert np.allclose(field1Solution, psiAnalytic[1], rtol=rtol)
    
    
def test_paircoupled_solve_matrix_eqn():
    """Test Hcoeffs_to_matrix_eqn() method of the PairCoupledFieldGroup class."""
    # create the UncoupledFieldGroup
    label0 = 'field0'
    label1 = 'field1'
    pfg = fieldgroups.PairCoupledFieldGroup(label0, label1)
    
    # create the Hcoeffs, rightBC, psi_mminus1
    (dt, dx, rightBC, psi_mminus1, J1, J2, J3, J6, J7, J8, K1, K2, K3, K6, K7, K8, psiAnalytic, rtol) = two_field_equation_setup()
    JKCoeffs = fieldgroups.JKCoefficients(J1=J1, J2=J2, J3=J3, J6=J6, J7=J7, J8=J8,
                                          K1=K1, K2=K2, K3=K3, K6=K6, K7=K7, K8=K8)
    
    # discretize into matrix equation
    matrixEqn = pfg.Hcoeffs_to_matrix_eqn(dt, dx, rightBC, psi_mminus1, JKCoeffs)
    
    # matrix solve
    profileSolution = pfg.solve_matrix_eqn(matrixEqn)
    
    # check
    field0Solution = profileSolution[label0]
    field1Solution = profileSolution[label1]
    
    assert np.allclose(field0Solution, psiAnalytic[0], rtol=rtol)    
    assert np.allclose(field1Solution, psiAnalytic[1], rtol=rtol)
    
def test_fields_to_fieldgroups():
    """Test the fields_to_fieldgroups() function."""
    # setup
    (dt, dx, nL, n_initial, H1, H2, H7, n_ss_analytic, rtol) = one_field_polar_setup()
    label0 = 'field0'
    
    (dt, dx, rightBC, psi_mminus1, J1, J2, J3, J6, J7, J8, K1, K2, K3, K6, K7, K8, psiAnalytic, rtol) = two_field_equation_setup()
    label1 = 'field1'
    label2 = 'field2'
    
    # create the fields
    field0 = multifield.Field(label0, nL, n_initial, coupledTo=None)
    field1 = multifield.Field(label1, rightBC[0], psi_mminus1[0], coupledTo='field2')
    field2 = multifield.Field(label2, rightBC[1], psi_mminus1[1], coupledTo='field1')
    fields = [field0, field1, field2]
    
    # create the HCoeffsAllFields
    HCoeffsAllFields = {}
    HCoeffs0 = multifield.HCoefficients(H1=H1, H2=H2, H7=H7)
    HCoeffs1 = multifield.HCoefficients(H1=J1, H2=J2, H3=J3, H6=J6, H7=J7, H8=J8)
    HCoeffs2 = multifield.HCoefficients(H1=K1, H2=K2, H3=K3, H6=K6, H7=K7, H8=K8)
    HCoeffsAllFields[field0.label] = HCoeffs0
    HCoeffsAllFields[field1.label] = HCoeffs1
    HCoeffsAllFields[field2.label] = HCoeffs2
    
    # run the function
    fieldGroups = fieldgroups.fields_to_fieldgroups(fields, HCoeffsAllFields)
    
    # check that fieldGroups was created properly
    assert len(fieldGroups) == 2
    fg0 = fieldGroups[0]
    # label, Hcoeffs, rightBC, psi_mminus1
    assert fg0.label == label0
    assert fg0.HCoeffs == HCoeffs0
    assert fg0.rightBC == nL
    assert np.allclose(fg0.psi_mminus1, n_initial)
    
    fg1 = fieldGroups[1]
    assert fg1.label0 == label1
    assert fg1.label1 == label2
    assert np.allclose(fg1.HCoeffs.J1, HCoeffs1.H1) and np.allclose(fg1.HCoeffs.J2, HCoeffs1.H2) and np.allclose(fg1.HCoeffs.J3, HCoeffs1.H3) and np.allclose(fg1.HCoeffs.J6, HCoeffs1.H6)
    assert np.allclose(fg1.HCoeffs.J7, HCoeffs1.H7) and np.allclose(fg1.HCoeffs.J8, HCoeffs1.H8)
    assert np.allclose(fg1.HCoeffs.K1, HCoeffs2.H1) and np.allclose(fg1.HCoeffs.K2, HCoeffs2.H2) and np.allclose(fg1.HCoeffs.K3, HCoeffs2.H3) and np.allclose(fg1.HCoeffs.K6, HCoeffs2.H6)
    assert np.allclose(fg1.HCoeffs.K7, HCoeffs2.H7) and np.allclose(fg1.HCoeffs.K8, HCoeffs2.H8)
    assert np.allclose(fg1.rightBC, rightBC)
    assert np.allclose(fg1.psi_mminus1[0], psi_mminus1[0])
    assert np.allclose(fg1.psi_mminus1[1], psi_mminus1[1])
    
def test_fieldgroups_to_profiles():
    """Test fieldsgroups_to_profiles() function."""
    # Setup
    label0 = 'f0'
    fg0 = fieldgroups.UncoupledFieldGroup(label0)
    fg0.profileSolution = {}
    prof0 = np.ones(10)
    fg0.profileSolution[label0] = prof0
    
    label1 = 'f1'
    label2 = 'f2'
    fg1 = fieldgroups.PairCoupledFieldGroup(label1, label2)
    fg1.profileSolution = {}
    prof1 = 2 * prof0
    prof2 = 3 * prof0
    fg1.profileSolution[label1] = prof1
    fg1.profileSolution[label2] = prof2
    fieldGroups = [fg0, fg1]
    
    # run it
    profiles = fieldgroups.fieldgroups_to_profiles(fieldGroups)
    
    # check
    assert np.allclose(profiles[label0], prof0)
    assert np.allclose(profiles[label1], prof1)
    assert np.allclose(profiles[label2], prof2)
    
    
def test_solve_multiple_fieldgroups():
    """Test the chain of functions for solving multiple fieldgroups."""
    # ----Setup----
    (dt, dx, nL, n_initial, H1, H2, H7, n_ss_analytic, rtol) = one_field_polar_setup()
    label0 = 'field0'
    
    (dt, dx, rightBC, psi_mminus1, J1, J2, J3, J6, J7, J8, K1, K2, K3, K6, K7, K8, psiAnalytic, rtol) = two_field_equation_setup()
    label1 = 'field1'
    label2 = 'field2'
    
    # create the fields
    field0 = multifield.Field(label0, nL, n_initial, coupledTo=None)
    field1 = multifield.Field(label1, rightBC[0], psi_mminus1[0], coupledTo='field2')
    field2 = multifield.Field(label2, rightBC[1], psi_mminus1[1], coupledTo='field1')
    fields = [field0, field1, field2]
    
     # create the HCoeffsAllFields
    HCoeffsAllFields = {}
    HCoeffs0 = multifield.HCoefficients(H1=H1, H2=H2, H7=H7)
    HCoeffs1 = multifield.HCoefficients(H1=J1, H2=J2, H3=J3, H6=J6, H7=J7, H8=J8)
    HCoeffs2 = multifield.HCoefficients(H1=K1, H2=K2, H3=K3, H6=K6, H7=K7, H8=K8)
    HCoeffsAllFields[field0.label] = HCoeffs0
    HCoeffsAllFields[field1.label] = HCoeffs1
    HCoeffsAllFields[field2.label] = HCoeffs2
    
    # ----Run through the solve steps----
    # create fieldGroups from fields
    fieldGroups = fieldgroups.fields_to_fieldgroups(fields, HCoeffsAllFields)
    
    # discretize to create matrix equation
    for fieldGroup in fieldGroups:
        fieldGroup.matrixEqn = fieldGroup.Hcoeffs_to_matrix_eqn(dt, dx, fieldGroup.rightBC, fieldGroup.psi_mminus1, fieldGroup.HCoeffs)
        
    # solve the matrix equations [iterating over groups]
    for fieldGroup in fieldGroups:
        fieldGroup.profileSolution = fieldGroup.solve_matrix_eqn(fieldGroup.matrixEqn)
        
    # get the profiles for the fields out of the fieldGroups, put into a dict of profiles
    profiles = fieldgroups.fieldgroups_to_profiles(fieldGroups)
    
    # ----Check----
    assert np.allclose(profiles[label0], n_ss_analytic, rtol)
    assert np.allclose(profiles[label1], psiAnalytic[0], rtol)
    assert np.allclose(profiles[label2], psiAnalytic[1], rtol)
    
    
    
#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================
def one_field_diffusion_setup():
    dt = 1e5        # test only steady state
    
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N-1)  # spatial grid size
    x = np.arange(N)*dx # location corresponding to grid points j=0, ..., N-1
    
    nL = 0.2
    # initial conditions
    n_initial = np.sin(np.pi * x) + nL * x
    
    # diffusion problem coefficients
    D = 0.8
    H1 = np.ones_like(x)
    H2 = D * np.ones_like(x)
    H7 = 1 - x**2
    
    # analytic steady-state solution
    n_ss_analytic = nL + 1/(12*D) * (1-x**2) * (5 - x**2)
    rtol = 1e-2
    
    return (dt, dx, nL, n_initial, H1, H2, H7, n_ss_analytic, rtol)

def one_field_polar_setup():
    dt = 1e9
    
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N - 0.5)  # spatial grid size
    x = np.linspace(dx/2, L, N)   # location corresponding to grid points j=0, ..., N-1... x=radius
    
    nL = 0.03
    # initial conditions
    n_initial = 0.3*np.sin(np.pi * x) + nL * x
    
    # diffusion problem coefficients
    D = 1.3
    H1 = x
    H2 = D*x
    H7 = x**2
    
    # analytic steady-state solution
    n_ss_analytic = nL + 1/(9*D) * (1-x**3)
    rtol = 1e-4
    return (dt, dx, nL, n_initial, H1, H2, H7, n_ss_analytic, rtol)
    
    
def two_field_equation_setup():
    L = 1           # size of domain
    N = 500         # number of spatial grid points
    dx = L / (N - 0.5)  # spatial grid size
    x = np.linspace(dx/2, L, N)   # location corresponding to grid points j=0, ..., N-1
    
    DU = 1.6
    DW = 2.4
    cU = 0.3
    cW = 9
    nu = 1.3
    alpha = 1.9
    beta = 0.09
    
    UL = 0.75
    WL = 3.33
    
    
    J2 = DU * x
    K2 = DW * x
    J3 = -cU * x**2
    K3 = -cW * x**2
    J6 = x
    K6 = x**2
    J8 = nu * x
    K8 = -nu * x
    
    pi = np.pi
    sin = np.sin
    cos = np.cos
    J7 = (9*alpha*DU*x**2  +  2*cU*x*UL  +  cU*alpha*(2*x-5*x**4) 
        - x*UL  - alpha*(x-x**4)  -  nu*x*WL  -  nu*beta*x*cos(pi * x / 2) )
    
    K7 = (pi/2 * DW * beta * (sin(pi*x/2) + pi*x/2 * cos(pi*x/2)) + 2*cW*x*WL
        + cW*beta*(2*x*cos(pi*x/2) - pi*x**2/2 * sin(pi*x/2)) - x**2 * WL
        - beta*x**2 * cos(pi*x/2)  +  nu*x*UL + nu*alpha*(x-x**4) )
    
    dt = 1e9
    rightBC = (UL, WL)
    U_initial = np.zeros_like(x)
    W_initial = np.zeros_like(x)
    psi_mminus1 = (U_initial, W_initial)
    
    # irrelevant values but have to be specified
    J1 = np.ones_like(x)
    K1 = np.ones_like(x)    
    
    # analytic solutions
    UAnalytic = UL + alpha * (1 - x**3)
    WAnalytic = WL + beta * cos(pi * x / 2)
    psiAnalytic = (UAnalytic, WAnalytic)
    
    rtol=1e-4
    
    return (dt, dx, rightBC, psi_mminus1, J1, J2, J3, J6, J7, J8, K1, K2, K3, K6, K7, K8, psiAnalytic, rtol)
    