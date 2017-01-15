"""
parameters

Provide a Python interface for reading and modifying the GENE parameters file.  This module allows one to edit various
parameters that do not exist within the libgene_tango interface.

The parameters file has many lines, each consisting of at most one <varname> = <value> statement.  A very short sample
of a GENE parameters file looks like.

x0 = 0.5000
adapt_ly = .T.
mu_grid_type = 'gau_lag'

In modifying the parameters file on disk, this module allows one to find a specific variable and replace its value with
another, user-specified value.

Larger sample of a GENE parameters file:
    
&parallelization
n_procs_s =   1
n_procs_v =   2
n_procs_w =  24
n_procs_x =   1
n_procs_y =   1
n_procs_z =   4
n_procs_sim =    192
/

&box
n_spec = 1
nx0 = 64
nky0 = 16
nz0 = 16
nv0 = 64
nw0 = 24

kymin = 0.08
lv = 4
lw = 20
x0 = 0.5000
adapt_ly = .T.
mu_grid_type = 'gau_lag'
lx_a = 0.9
/

&in_out
diagdir = '/scratch2/scratchdirs/jbparker/genedata/prob13'
read_checkpoint = .T.
istep_nrg = 10
istep_field = 100
istep_mom = 100
istep_prof = 100
istep_vsp = 500
istep_schpt = 5000
istep_g1 = 0
/

(... omitted code... )

&nonlocal_x
rad_bc_type = 1
ck_heat = 0.1000

(... omitted code... )

minor_r = 0.35000000
major_R = 1.0000000    
    
-----------------------
See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division

def read_parameters_file(parametersFilename):
    """Read parameters file into memory as a string.
    
    Inputs:
      parametersFilename    path to parameters file (string)
    Outputs:
      parametersString      parameters file as a single string (string)
    """
    with open(parametersFilename, 'r') as f:
        parametersString = f.read()
    return parametersString    

def write_parameters_file(parametersFilename, parametersString):
    """Write out the modified parameters string to file.  Overwrites the parameters file if it exists.
    
    Inputs:
      parametersFilename   path to file to write (string)
      parametersString     contents of parameters file to write (string)
    """
    with open(parametersFilename, 'w') as f:
        f.write(parametersString)

def modify_parameters_file(parametersFilename, **kwargs):
    """Modify the GENE parameters file, allowing to change multiple values in the form <varName> = <value>.
    
    Specify each name and new value as a keyword pair, i.e., 
        modify_parameters_file(parametersFilename, varName1=newValue1, varName2=newValue2, ...)
    E.g., 
        modify_parameters_file(parametersFilename, diagdir='/newpath/', lx=45.45, read_checkpoint=True)
        
    Inputs:
      parametersFilename    path to parameters file to overwrite (string)
      kwargs                <varName> = <newVarValue> pairs
    """
    modifiedParametersString = read_parameters_file(parametersFilename)
    for varName in kwargs:
        newVarValue = kwargs[varName]
        modifiedParametersString = modify_parameters_string(modifiedParametersString, varName, newVarValue)
    write_parameters_file(parametersFilename, modifiedParametersString)        
        
def modify_parameters_string(parametersString, varName, newVarValue):
    """Wrapper for do_modify_parameters_string().  Converts the newVarValue to a string then passes."""   
    newVarValueStr = convert_to_string(newVarValue)
    modifiedParametersString = do_modify_parameters_string(parametersString, varName, newVarValueStr)
    return modifiedParametersString
    
def do_modify_parameters_string(parametersString, varName, newVarValueStr):
    """Modify the in-memory string corresponding to the GENE parameters file.
    newVarValue is assumed to be a string and in a form suitable for Fortran to read.  E.g., not True or 'True', but '.T.'
    
    Inputs:
      parametersString      entire parameters file (string)
      varName               name of variable whose value will be modified (string)
      newVarValueStr        new value for variable (string)
    """    
    # Extract the line with <varName> = <value>, then extract the value as a string
    lineOfInterest = extract_line_with_variable(parametersString, varName)
    currentVarValueStr = extract_value_from_line(lineOfInterest)
    
    # leave the whitespace and replace currentVarValueStr with the NewVarValueStr within the line of interest
    modifiedLineOfInterest = lineOfInterest.replace(currentVarValueStr, newVarValueStr)
    
    # replace the old line with the new line in the parameters-file string
    modifiedParametersString = parametersString.replace(lineOfInterest, modifiedLineOfInterest)    
    return modifiedParametersString
    
def extract_line_with_variable(parametersString, varName):
    """Extract a substring of the form <varName> = <value> from a file.  The file is assumed to contain multiple lines, 
    consisting of at most one <varName> = <value> statement, with nothing else on the line.
    
    Warning: If there is more than one line with the same varName, then currently the *first* line is found and modified.  This
    can be a problem if, for example, a line with the same variable name is commented out.  E.g.,
            ! n_procs_x = 4
            n_procs_x = 12
    For now, since this function does not check for comments, one must ensure that there are no commented lines with duplicates
    of variables that might be changed.
    
    Inputs:
      parametersString      entire parameters file (string)
      varName               name of variable on left side of equal sign (string)
    
    Outputs:
      line                  substring containing <varName> = <value>.  Excludes the trailing newline.  (string)
    """
    # Find the (first) location of the variable
    ind = parametersString.find(varName)
    if ind == -1:
        raise ValueError('variable {} not found within parameters file'.format(varName))
    
    # Find the location of the newline at the end of the line
    indLineEnd = parametersString.find('\n', ind)
    
    # Extract the substring containing the specified variable, excluding the newline
    line = parametersString[ind:indLineEnd]
    return line
    
def extract_value_from_line(line):
    """Extract the value from a <varname> = <value> string.
    Inputs:
      line          string consisting of <varName> = <value> (string)
    Outputs:
      varValue      string after the equals sign in <varName> = <value>, eliminating any whitespace (string)
    """
    # Find the equals sign
    indEqualSign = line.find('=')
    
    # extract the value (as a string) as what is to the right of the equals sign, then eliminate any whitespace
    varValue = line[indEqualSign + 1:].strip()
    return varValue
    
def convert_to_string(value):
    """Convert input value to a string suitable for inserting into GENE parameters file.
    4 types supported: bool, int, float, string
    
    Inputs:
      value        (supported types are bool, int, float, string)
    Outputs:
      geneString   (string)
    """
    if isinstance(value, bool) or value in ('True', 'False', '.T.', '.F.'):
        geneString = boolean_to_string(value)
    elif isinstance(value, int): # must be done in this order because a bool is a type of int
        geneString = int_to_string(value)
    elif isinstance(value, float):
        geneString = float_to_string(value)
    elif isinstance(value, str):
        geneString = string_to_string(value)
    else:
        raise TypeError
    return geneString        
    
def boolean_to_string(inBool):
    """Convert Boolean to either .T. or .F. for inserting into GENE parameters file
    
    Inputs:
      inBool        (bool [or 'True', 'False', '.T.', or '.F.'])
    Outputs:
      geneString    (string)
    """
    if inBool == True or inBool == 'True' or inBool == '.T.':
        geneString = '.T.'
    if inBool == False or inBool == 'False' or inBool == '.F.':
        geneString = '.F.'
    return geneString

def int_to_string(inInt):
    """Convert integer to string for inserting into GENE parameters file
    
    Inputs:
      inInt        (int)
    Outputs:
      geneString   (string)
    """
    geneString = '{}'.format(inInt)
    return geneString

def float_to_string(inFloat):
    """Convert float to string for inserting into GENE parameters file
    """
    geneString = '{}'.format(inFloat)
    return geneString

def string_to_string(inString):
    """Convert string to string for inserting into GENE parameters file.  Add ' ' around the string.  For
    instance, if inString=='/path/to/file', then geneString=="'/path/to/file'".
    """
    geneString = '\'' + inString + '\''
    return geneString
    
def extract_current_value(parametersString, varName):
    """Get the current value of <varName> in the parameters file.
    
    Inputs:
      parametersString      entire parameters file (string)
      varName               name of variable whose value is to be returned (string)
    Outputs:
      varValue              value of varName (string)
    """
    line = extract_line_with_variable(parametersString, varName)
    varValue = extract_value_from_line(line)
    return varValue