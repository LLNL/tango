""" read_fluxprof2D.py

Read the fluxprof2D files that contain particle flux and heat flux as functions of 
time and radius that are output by the GENE IDL diagnostic 'flux profile 2D'
e.g., fluxprof2Dions_0001.dat

Note: GENE saves these fluxes every istep_mom.  The values saved are the instantaneous values in the
GENE simulation at that time.

Format of the flux_prof2Dions_0001.dat file:   [notes in <> are added by me, and are not in the file itself]

<multiple lines of comments that start with #>
#x/a  <radial grid>
x1
x2
...
xN


# t  <time grid>
t1
t2
...
tM


# G_es,ions
y11 y12 .. y1M    .....  M entries in the line representing the time dependence at x1
y21 y22 .. y2M    .....  M entries in the line representing the time dependence at x2
y31 y32 .. y3M    .....  Data is N x M, whitespace delimited.   Meaning 
..
yN1 yN2 .. yNM    .....  M entries in the line representing the time dependence at xN


# Q_es,ions
y11 y12 .. y1M    .....  M entries in the line representing the time dependence at x1
y21 y22 .. y2M    .....  M entries in the line representing the time dependence at x2
y31 y32 .. y3M    .....  Data is N x M, whitespace delimited.   Meaning 
..
yN1 yN2 .. yNM    .....  M entries in the line representing the time dependence at xN

# G_em,ions [[only if EM fluxes are included in output]]
....
...

# Q_em,ions [[only if EM fluxes are included in output]]
.....
....
.....


------
Also note that when the species name is electrons, the comment lines say G_es,elec instea of G_es,electrons.


-----------------------------------------------
Example usage:
filename = 'fluxprof2D.dat'
fluxes = ProfileFileData(filename)
    
    
"""

from __future__ import division
import numpy as np

class ProfileFileData(object):
    """Class to handle the profile data for a single profile_{spec}{fext}

    """

    def __init__(self, filename, speciesName, readElectromagnetic=False):
        """Class constructor.  Read in the file
        Inputs:
          filename                          name of GENE fluxprof2D file to read the turbulent flux data from (string)
          speciesName                   name of species as given to GENE, e.g. 'ions' or 'electrons' (string)
          readElectromagnetic       (optional) if True, read electromagnetic fluxes in addition to the electrostatic fluxes (boolean)
        
        Outputs:
          None
        
        Side effects:
          self.particleFluxTurb is set (2D array, time x space) 
          self.heatFluxTurb is set (2D array, time x space)
          self.rho is set (1D array)
          self.time is set (1D array)
        """
        self.filename = filename
        self.speciesName = speciesName
        self.readElectromagnetic = readElectromagnetic
        # declaring member variables
        self.numRadialPts = None
        self.numTimePts = None
        self.time = None
        self.rho = None  # x/a
        self.particleFluxTurb = None
        self.heatFluxTurb = None
        
        self.read_profile_data(filename)

#    def get_start_end_times(self):
#        return self.timeField[0], self.timeField[-1]

    def read_profile_data(self, filename):
        """ Read radial grid, time grid, and fluxes 
        
        Outputs:
          none
          
        Side Effects:
          self.particleFluxTurb is set (2D array, time x space) 
          self.heatFluxTurb is set (2D array, time x space)
          self.rho is set (1D array)
          self.time is set (1D array)
        """
        #print('Reading  {}\n'.format(self.filename))
        with open(self.filename) as f:
            lines = f.readlines()
            
        # for debug
        #self.lines = lines    
        
        # read radial coordinate array
        self.rho = self._read_rho(lines)
        numRadialPts = len(self.rho)
        
        # read time array
        self.time = self._read_time(lines)
        numTimePts = len(self.time)
        
        # read particle and heat fluxes
        (particleFluxElectrostatic, heatFluxElectrostatic) = self._read_electrostatic_fluxes(lines, numRadialPts, numTimePts)
        self.particleFluxTurb = particleFluxElectrostatic
        self.heatFluxTurb = heatFluxElectrostatic
        
        if self.readElectromagnetic:
            (particleFluxElectromagnetic, heatFluxElectromagnetic) = self._read_electromagnetic_fluxes(lines, numRadialPts, numTimePts)
            self.particleFluxTurb += particleFluxElectromagnetic
            self.heatFluxTurb += heatFluxElectromagnetic
        
    @staticmethod
    def _read_rho(lines):
        """Read in the rho array from lines.
        
        Inputs:
          lines             fluxprof2D file read in by readlines() (list)
        Outputs:
          rho               normalized radial coordinate, rho=x/a (array)
        """
        rho = _read_1d_array_at_comment(lines, 'x/a')
        return rho
        
    @staticmethod
    def _read_time(lines):
        """Read in the rho array from lines.
        
        Inputs:
          lines             fluxprof2D file read in by readlines() (list)
        Outputs:
          time              time grid, in GENE normalized units of Lref/cref (array)
        """
        time = _read_1d_array_at_comment(lines, 't')
        return time
        
        
    def _read_electrostatic_fluxes(self, lines, numRadialPts, numTimePts):
        """Read particle flux and heat flux data, electrostatic part.
        
        Inputs:
          lines                                     lines of the fluxprof2D file in format given by readlines() (list)
          numRadialPts                      number of radial points in the file (integer)
          numTimePts                        number of time points in the file (integer)
        Outputs:
          particleFluxElectrostatic     GENE's turbulent particle flux (2D array, time x space)
          heatFluxElectrostatic         GENE's turbulent heat flux (2D array, time x space)
        """
        particleFluxLabel = 'G_es,{}'.format(self.speciesName)
        particleFluxElectrostatic = _read_flux2D_array_at_comment(lines, particleFluxLabel, numTimePts, numRadialPts)
        heatFluxLabel = 'Q_es,{}'.format(self.speciesName)
        heatFluxElectrostatic = _read_flux2D_array_at_comment(lines, heatFluxLabel, numTimePts, numRadialPts)
            
              #### JP: SOMEONE NEEDS TO EXPLAIN THIS???  Need to be clear about what is the convective flux, what is GENE's output, etc.
        #self.heatFluxNeo -= 2.5 * self.particleFluxNeo * self.temperature
        #self.heatFluxTurb -= 2.5 * self.particleFluxTurb * self.temperature
        return (particleFluxElectrostatic, heatFluxElectrostatic)
            
    def _read_electromagnetic_fluxes(self, lines, numRadialPts, numTimePts):
        """Read particle flux and heat flux data, electromagnetic part.
        
        Inputs:
          lines                                             lines of the fluxprof2D file in format given by readlines() (list)
          numRadialPts                              number of radial points in the file (integer)
          numTimePts                                number of time points in the file (integer)
        Outputs:
          particleFluxElectromagnetic       GENE's turbulent particle flux (2D array, time x space)
          heatFluxElectromagnetic           GENE's turbulent heat flux (2D array, time x space)
        """
        particleFluxLabel = 'G_em,{}'.format(self.speciesName)
        particleFluxElectromagnetic = _read_flux2D_array_at_comment(lines, particleFluxLabel, numTimePts, numRadialPts)
        heatFluxLabel = 'Q_em,{}'.format(self.speciesName)
        heatFluxElectromagnetic = _read_flux2D_array_at_comment(lines, heatFluxLabel, numTimePts, numRadialPts)
        return (particleFluxElectromagnetic, heatFluxElectromagnetic)
    
    @staticmethod
    def _get_next_data_line(lineGenerator):
        """Helper function for _read_profile().
        
        Get the next data line by skipping over whitespace or comment lines.
        
        Inputs:
          lineGenerator     generator object which generates lines of the profile file upon caling .next()
        Outputs:
          line              data line which is not a whitespace or comment line (string)
        """
        gotDataLine = False
        while not gotDataLine:
            line = lineGenerator.next()
            if not line.strip() or line.startswith('#'):  # ignore lines with only whitespace or comment lines
                gotDataLine = False
            else:
                gotDataLine = True
        return line
            
    
def _whitespace_or_comment(line):
    """Returns True if line has only whitespace or is a comment line (starts with #), False otherwise.
    Inputs:
      line                       (string)
    Outputs:
      isWhitespaceOrComment      (boolean)
    """
    if not line.strip() or line.startswith('#'):
        isWhitespaceOrComment = True
    else:
        isWhitespaceOrComment = False
    return isWhitespaceOrComment
    
def _line_gen_at_comment(lines, substring):
    """Return a line generator that reads up to a comment line starting with '#' and consisting of substring.
    
    In other words, the first use of next() method returns the line after the comment line.  There may be whitespace
    around substring.
    
    Inputs:
      lines             list of lines, e.g., from readlines() (list)
      substring         substring to look for in comment line (string)
    Outputs:
      lineGenerator     generator starting at desired line
    """
    lineGenerator = (line for line in lines)
    # eat lines until line starts with # and contains substring
    gotLine = False
    while not gotLine:
        line = lineGenerator.next()
        if line.startswith('#') and line[1:].strip()==substring:
            gotLine = True
    return lineGenerator
    
def _read_1d_array_at_comment(lines, substring):
    """Read in a 1D array that occurs in a list of lines with a # identifier.
    
    The array occurs beginning after a line that is in the form:
       #<whitespace><substring><whitespace>
    Then the data is listed, one entry on each line, with the end of the data identified by a whitespace line or comment line
    
    Inputs:
      lines             file read in by readlines() (list)
      substring         identifier in the line preceding the data (string)
    Outputs:
      data              1d data array (array)
    """
    # get a line generator starting at line with radial grid
    lineGenerator = _line_gen_at_comment(lines, substring)
    
    endOfData = False
    dataList = []
    while not endOfData:
        line = lineGenerator.next()
        if not _whitespace_or_comment(line):
            dataList.append(np.fromstring(line, sep=' ')[0])
        else:
            endOfData = True
    data = np.asarray(dataList)            
    return data

def _read_flux2D_array_at_comment(lines, fluxLabel, numTimePts, numRadialPts):
    """Read in a 2D array that occurs in a list of lines starting after the comment with the fluxLabel.
   
    E.g., fluxLabel = 'Q_es,ion' or 'G_em,electrons'
   
    Inputs:
      lines                      lines of the fluxprof2D file in format given by readlines() (list)
      fluxLabel              string in comment of GENE's fluxprof2D file that identifies the flux (string)
      numTimePts         number of time points in the file (integer)
      numRadialPts       number of radial points in the file (integer)
    Outputs:
      flux                       flux read in from lines (2D array, time x space)
    """
    flux = np.zeros((numTimePts, numRadialPts))
    lineGenerator = _line_gen_at_comment(lines, fluxLabel)
    for radialIndex in np.arange(numRadialPts):
        line = lineGenerator.next()
        data = np.fromstring(line, sep=' ')
        flux[:, radialIndex] = data
    return flux