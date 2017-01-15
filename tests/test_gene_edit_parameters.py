"""Test the code that edits a GENE parameters file to modify variable values.

Goal: use an IOstream so that a string buffer can be used as a file-like object, to avoid reading and writing files on disk...

"""

from __future__ import division, absolute_import
import numpy as np
import os
import tango.utilities.gene.parameters as parameters

def test_extract_line_with_variable():
    # test extracting lines from parameters string (four possible datatypes)
    parametersString = setup()
    line1 = parameters.extract_line_with_variable(parametersString, 'n_procs_w')
    line2 = parameters.extract_line_with_variable(parametersString, 'adapt_ly')
    line3 = parameters.extract_line_with_variable(parametersString, 'mu_grid_type')
    line4 = parameters.extract_line_with_variable(parametersString, 'major_R')
    
    assert line1 == 'n_procs_w =  24'
    assert line2 == 'adapt_ly = .T.'
    assert line3 == "mu_grid_type = 'gau_lag'"
    assert line4 == 'major_R = 1.0000000'

def test_extract_value_from_line():
    # test extracting value as a string from a line
    parametersString = setup()
    line1 = parameters.extract_line_with_variable(parametersString, 'n_procs_w')
    line2 = parameters.extract_line_with_variable(parametersString, 'adapt_ly')
    line3 = parameters.extract_line_with_variable(parametersString, 'mu_grid_type')
    line4 = parameters.extract_line_with_variable(parametersString, 'major_R')
    
    value1Str = parameters.extract_value_from_line(line1)
    value2Str = parameters.extract_value_from_line(line2)
    value3Str = parameters.extract_value_from_line(line3)
    value4Str = parameters.extract_value_from_line(line4)
    
    assert value1Str == '24'
    assert value2Str == '.T.'
    assert value3Str == "'gau_lag'"
    assert value4Str == '1.0000000'
    
def test_convert_to_string():
    # test converting value to a string appropriate for GENE parameters file
    assert parameters.convert_to_string(24) == '24'
    assert parameters.convert_to_string(True) == '.T.'
    assert parameters.convert_to_string('True') == '.T.'
    assert parameters.convert_to_string('.T.') == '.T.'
    assert parameters.convert_to_string(False) == '.F.'
    assert parameters.convert_to_string('False') == '.F.'
    assert parameters.convert_to_string('.F.') == '.F.'
    assert parameters.convert_to_string('gau_lag') == "'gau_lag'"
    assert parameters.convert_to_string(1.0003) == '1.0003'
    
def test_extract_current_value():
    # test extracting value as a string from a parameters file
    parametersString = setup()
    value1Str = parameters.extract_current_value(parametersString, 'mu_grid_type')
    assert value1Str == "'gau_lag'"
    
def test_modify_parameters_string():
    # test modifying parameters string with lower level function
    parametersString = setup()
    parametersString = parameters.modify_parameters_string(parametersString, 'n_procs_w', 26)
    parametersString = parameters.modify_parameters_string(parametersString, 'adapt_ly', False)
    parametersString = parameters.modify_parameters_string(parametersString, 'mu_grid_type', 'no_lag')
    parametersString = parameters.modify_parameters_string(parametersString, 'major_R', 3.245)
    
    expectedParametersString = setup2()
    assert parametersString == expectedParametersString

def test_modify_parameters_file():
    # test modifying parameters file on disk
    
    # setup by writing a file to disk
    parametersStringToWrite = setup()
    tempPath = 'tempParamFile.txt'
    parameters.write_parameters_file(tempPath, parametersStringToWrite)
        
    # read it and modify it on disk
    parameters.modify_parameters_file(tempPath, n_procs_w=26,  adapt_ly=False,  mu_grid_type='no_lag',  major_R=3.245)
    
    # check
    expectedParametersString = setup2()
    readParametersString = parameters.read_parameters_file(tempPath)
    assert readParametersString == expectedParametersString
    
    # teardown
    os.remove(tempPath)
    
########################### Helper Functions ##############################
def setup():
    # Create a snippet of a GENE parameters file and return it as a string
    parametersString = '''&parallelization
    n_procs_s =   1
    n_procs_v =   2
    n_procs_w =  24
    
    &box
    n_spec = 1
    nx0 = 64

    adapt_ly = .T.
    mu_grid_type = 'gau_lag'
    /
    
    &in_out
    diagdir = '/scratch2/scratchdirs/jbparker/genedata/prob13'
    read_checkpoint = .T.
    major_R = 1.0000000
    '''
    return parametersString
    
def setup2():
    # Create an alernate snippet of a GENE parameters file, slightly modified from setup()
    parametersString = '''&parallelization
    n_procs_s =   1
    n_procs_v =   2
    n_procs_w =  26
    
    &box
    n_spec = 1
    nx0 = 64

    adapt_ly = .F.
    mu_grid_type = 'no_lag'
    /
    
    &in_out
    diagdir = '/scratch2/scratchdirs/jbparker/genedata/prob13'
    read_checkpoint = .T.
    major_R = 3.245
    '''
    return parametersString