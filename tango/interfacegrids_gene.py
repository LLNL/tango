"""module interfacegrids_gene
"""

class MoveBetweenGrids(object):
    """
    """
    def __init__(self):
        # initialize variables
        pass
    def MapProfileOntoTurbGrid(self, profile):
        return profile
    def MapTransportCoeffsOntoTransportGrid(self, D_genegrid, c_genegrid):
        D = self.MapToTransportGrid(D_genegrid)
        c = self.MapToTransportGrid(c_genegrid)
        return (D, c)
    def MapToTransportGrid(self, z_genegrid):
        """Map a quantity z (typically a transport coefficient) from gene's grid to tango's grid.
        
        If tango's grid extends further than gene's (which occurs at the inner boundary and possibly
        the outer boundary), then the transport coefficient is set to zero in the nonoverlapping region.
        
        If gene's grid extends further than tango's (which possibly occurs at the outer boundary), then
        the transport coefficient is merely truncated.
        
        Interpolation is also applied in case the grids do not exactly overlap.
        """
        z = 0
        return z