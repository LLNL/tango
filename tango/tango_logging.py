"""tango_logging: encapsulate the mechanism for handling messages to log.

Usage: The usage is modeled after the Python standard logging module.  The user should call setup() only once, after
MPIrank has been obtained from somewhere.  Further calls to setup() do not do anything.  Within any other Python module
one wants to use logging, one should import tango_logging and simply call tango_logging.log(message).  

For now, the interface to tango_logging uses only the log() call, unlike the Python logging module which allows debug(),
info(), warning(), error(), and critical(), as shorthand for log() calls.

Currently supports only logging to stdout, mainly because stdout when running a batch job is automatically sent to a
job output file.  In the future, the option to add files as an additional optional target may be added.

Example:  To initialize, the user code should (when in a parallel environment) call
    tango_logging.setup(True, MPIrank)
    
See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import datetime

parallelEnvironment = None
MPIrank = None
initialized = False

def setup(_parallelEnvironment, _MPIrank):
    """Initial setup.  Run this only once.
    
    In order for logging to occur, either parallelEnvironment must be False, or a process must have an MPIrank of 0.
    
    Inputs:
      _parallelEnvironment      True if in a parallel environment (Boolean)
      _MPIrank                  unique rank for each process (integer)
    """
    global parallelEnvironment, MPIrank, initialized
    if initialized == False:
        parallelEnvironment = _parallelEnvironment
        MPIrank = _MPIrank
        initialized = True

def log(message, level=None):
    """Wrapper handling logic around the actual logging"""
    if initialized == True and serial_or_rank0():
        _log(message, level)
    
def _log(message, level=None):
    """Log the message, adding a timestamp.
    
    Adding optional targets would go here.
    """
    message = add_time_stamp(message)
    _log_to_stdout(message)

def _log_to_stdout(message):
    """Log the message to stdout."""
    print(message)

def serial_or_rank0():
    """Return true if in a nonparallel environment or if MPIrank==0, otherwise False"""
    if parallelEnvironment == False:
        return True
    else:
        if MPIrank == 0:
            return True
    return False
    
def time_stamp():
    """Produce a time stamp appropriate for logging messages.
    
    Timestamp uses UTC, which is Pacific Time + 8 hours.  The timestamp has the time only and does not include the date.
    """
    # timestampStr = datetime.datetime.utcnow().strftime("%H:%M:%S") # equivalent to line below
    timestampStr = '{:%H:%M:%S}'.format(datetime.datetime.utcnow())
    return timestampStr
    
def add_time_stamp(message):
    """Add a time stamp in parentheses to the end of a message."""
    timeStamp = '  ({})'.format(time_stamp())
    return message + timeStamp