"""Test multifield.py"""

from __future__ import division
import numpy as np
from tango import multifield


def test_get_field_by_label():
    """Test get_field_by_label() function."""
    # create a list of fields with labels
    f1 = multifield.Field(label='ab')
    f2 = multifield.Field(label='de')
    f3 = multifield.Field(label='ions')
    f4 = multifield.Field(label='electrons')
    fields = [f1, f2, f3, f4]
    
    # check
    assert multifield.get_field_by_label(fields, 'ab') is f1
    assert multifield.get_field_by_label(fields, 'de') is f2
    assert multifield.get_field_by_label(fields, 'ions') is f3
    assert multifield.get_field_by_label(fields, 'electrons') is f4

def test_check_fields_label():
    # setup
    f1 = multifield.Field(label='ab', rightBC=1.3)
    f2 = multifield.Field(label='de', rightBC=1.4)
    f3 = multifield.Field(label='ions', rightBC=np.ones(2))
    f4 = multifield.Field(label='ions', rightBC=1.5)
    fields = [f1, f2, f3]
    fields2 = [f1, f2, f3, f4]
    
    # check
    assert multifield.check_fields_label(fields)
    assert not multifield.check_fields_label(fields2)
    
#
def test_check_fields_rightBC():
    # setup
    f1 = multifield.Field(label='ab', rightBC=1.3)
    f2 = multifield.Field(label='de', rightBC=1.4)
    f3 = multifield.Field(label='ions', rightBC=np.ones(2))
    f4 = multifield.Field(label='ions')
    fields = [f1, f2]
    fields2 = [f1, f2, f3]
    fields3 = [f1, f2, f4]
    
    # check
    assert multifield.check_fields_rightBC(fields)
    assert not multifield.check_fields_rightBC(fields2)
    assert not multifield.check_fields_rightBC(fields3)

def test_check_fields_profile_mminus1():
    # setup
    f1 = multifield.Field(label='ab', profile_mminus1=np.zeros(3))
    f2 = multifield.Field(label='de', profile_mminus1=np.zeros(3))
    f3 = multifield.Field(label='ions', profile_mminus1=np.ones(4))
    f4 = multifield.Field(label='elecs')
    fields = [f1, f2]
    fields2 = [f1, f2, f3]
    fields3 = [f1, f2, f4]
    
    # check
    assert multifield.check_fields_profile_mminus1(fields)
    assert not multifield.check_fields_profile_mminus1(fields2)
    assert not multifield.check_fields_profile_mminus1(fields3)

def test_check_fields_coupled_to():
    # setup
    f1 = multifield.Field(label='ab')
    f2 = multifield.Field(label='de')
    f3 = multifield.Field(label='ions', coupledTo='elecs')
    f4 = multifield.Field(label='elecs', coupledTo='ions')
    f5 = multifield.Field(label='elecs', coupledTo='aa')
    f6 = multifield.Field(label='elecs')
    
    # check
    assert multifield.check_fields_coupled_to([f1, f2, f3, f4])
    assert not multifield.check_fields_coupled_to([f1, f2, f3, f5])
    assert not multifield.check_fields_coupled_to([f1, f2, f3, f6])
    # check with a different order
    assert multifield.check_fields_coupled_to([f1, f2, f4, f3])
    assert not multifield.check_fields_coupled_to([f1, f2, f5, f3])
    assert not multifield.check_fields_coupled_to([f1, f2, f6, f3])
    
    
#==============================================================================
#    End of tests.  Below are helper functions used by the tests
#==============================================================================
def create_field_a():
    pass