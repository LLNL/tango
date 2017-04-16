"""tango_logging: encapsulate the mechanism for handling messages to log.

Usage: The usage is modeled after the Python standard logging module.  The user should call setup() only once, after
MPIrank has been obtained from somewhere.  Further calls to setup() do not do anything.  Within any other Python module
one wants to use logging, one should import tango_logging and simply call log(message).  If setup() is not called, then
nothing will happen.

Like logging, the interface to tango_logging uses debug(), info(), warning(), error(), and critical(), as shorthand for log() calls.

Currently supports only logging to stdout, mainly because stdout when running a batch job is automatically sent to a
job output file.  In the future, the option to add files as an additional optional target may be added.

Example.   Run setup() only once.  Then call one of debug(), info(), warning(), error(), critical()
import tango.tango_logging as tlog
tlog.setup(True, MPIrank, tlog.DEBUG)     # when in a parallel environment
tlog.setup(False, 0, tlog.WARNING)        # when in a serial environment
tlog.info("message goes here")
tlog.debug("message goes here")    


Example:  To initialize, the user code should (when in a parallel environment) call
    tango_logging.setup(True, MPIrank)
For serial code, call
    tango_logging.setup(False, 0)
    
See https://github.com/LLNL/tango for copyright and license information
"""

from __future__ import division
import datetime

# globals
parallelEnvironment = None
MPIrank = None
threshold = None
initialized = False

# constants taken from logging module
CRITICAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DEBUG = 10


def setup(parallelEnvironmentIn, MPIrankIn, level=INFO):
    """Initial setup.  Run this only once.
    
    In order for logging to occur, either parallelEnvironment must be False, or a process must have an MPIrank of 0.
    
    Inputs:
      parallelEnvironmentIn     True if in a parallel environment (Boolean)
      MPIrankIn                 unique rank for each process (integer)
      level                     Sets the threshold for this logger. Logging messages which are less severe than thresholdIn will be ignored.
    """
    global parallelEnvironment, MPIrank, threshold, initialized
    if initialized == False:
        parallelEnvironment = parallelEnvironmentIn
        MPIrank = MPIrankIn
        threshold = level
        initialized = True

def debug(message):
    return log(message, DEBUG)

def info(message):
    return log(message, INFO)

def warning(message):
    return log(message, WARNING)

def error(message):
    return log(message, ERROR)

def critical(message):
    return log(message, CRITICAL)
        
def log(message, level=INFO):
    """Wrapper handling logic around the actual logging.
    
    Only log if:
        1) logger is initialized
        2) this is the correct process to perform logging
        3) level >= currently set threshold 
    """
    if initialized and serial_or_rank0() and level >= threshold:
        _log(message, level)
    
def _log(message, level):
    """Log the message, adding a timestamp.
    
    Adding optional targets would go here.  Input argument level is currently unused, but it could also be used to tag messages.
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

def set_level(level):
    global threshold
    threshold = level
    
def get_level():
    return threshold    
    
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