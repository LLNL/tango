from __future__ import division, print_function, absolute_import

def duration_as_hms(durationInSeconds):
    """Convert a duration in seconds into days, hours, minutes, seconds
    
    Rounds to the nearest second.
    
    Inputs:
      durationInSeconds     a time duration, measured in seconds (scalar)
    Outputs:
      durationAsHMS         time duration in days (if nonzero), hours:minutes:seconds (string)
    """
    minutes, seconds = divmod(round(durationInSeconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days == 0:
        durationAsHMS = "%02d:%02d:%02d" % (hours, minutes, seconds)
    else:
        durationAsHMS = "%d days, %d:%02d:%02d" % (days, hours, minutes, seconds)
    return durationAsHMS