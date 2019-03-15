"""
Tango analysis helper.  

Helper class that provides conveniences for offline analysis to test out Tango steps.  Particularly
to see the effects of a negative particle flux on the first iteration

Take in profiles and flux as inputs

Allow the possiblity of trying many different options for stepping.  Don't change the state at the end
    --allow treadlightly (or not)
    --allow change to thetaparams / thatfunc (diffusive convective split).  Dmax, Dmin.
    --explicit step?  put all or some of flux into H4
"""

import numpy as np
import scipy.interpolate
import scipy.integrate

import tango
import tango.fieldgroups
import tango.utilities.gene.read_chease_file as read_chease_file

# physical constants with module-wide scope.  Given in SI units
e = 1.60217662e-19          # electron charge
mp = 1.6726219e-27          # proton mass

def interpolate_1d_qty(x, y, xNew):
    """Interpolate a 1D quantity from given radial grid to new radial grid.
             
    Inputs:
        x           x grid as given (1D array)
        y           quantity evaluated on given grid x (1D array)
        xNew        new x grid on which to interpolate the quantity (1D array)
        
    
    Outputs:
        yNew        quantity interpolated onto xNew grid (1D array)
    """
    interpolator = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    yNew = interpolator(xNew)
    return yNew


class analysis:
    def __init__(self, rhoTango, rhoTurb, profiles, fluxes,
                 cheaseFilename, 
                 n_BC, Ti_keV_BC, Te_keV_BC,
                 Tref, mref, nref,
                 n_D_adhoc, pi_D_adhoc, pe_D_adhoc,
                 rhoExtrapZoneLeft=0.80, rhoExtrapZoneRight=0.85):
        self.rhoTango = rhoTango
        self.rhoTurb = rhoTurb
        
        
        self.profiles = profiles
        self.fluxes = fluxes
        
        cheaseTangoData = read_chease_file.get_chease_data_on_Tango_grid(cheaseFilename, rhoTango)
        minorRadius = cheaseTangoData.minorRadius
        Lref = cheaseTangoData.Lref
        majorRadius = Lref
        Bref = cheaseTangoData.Bref
        VprimeTango = cheaseTangoData.dVdx
        gxxAvgTango = cheaseTangoData.gxxAvg  # average <g^xx> = <grad x dot grad x>
        gradxAvgTango = cheaseTangoData.gradxAvg
        
        self.VprimeTango = VprimeTango
        self.gxxAvgTango = gxxAvgTango
        self.gradxAvgTango = gradxAvgTango
        self.gxxAvgTurb = interpolate_1d_qty(rhoTango, gxxAvgTango, rhoTurb)
        self.gradxAvgTurb = interpolate_1d_qty(rhoTango, gradxAvgTango, rhoTurb)
        
        xTango = rhoTango * minorRadius
        xTurb = rhoTurb * minorRadius
        
        self.dxTango = xTango[1] - xTango[0]
        self.dxTurb = xTurb[1] - xTurb[0]
    
        self.minorRadius = minorRadius
        self.majorRadius = majorRadius
        
        self.Bref = Bref
        self.mref = mref
        self.Tref = Tref
        self.nref = nref
        self.Lref = majorRadius
        
        self.n_D_adhoc = n_D_adhoc
        self.pi_D_adhoc = pi_D_adhoc
        self.pe_D_adhoc = pe_D_adhoc
        
        
        self.rhoTango = xTango / minorRadius
        self.rhoTurb = xTurb / minorRadius
        
        self.n = profiles['n']
        self.pi = profiles['pi']
        self.pe = profiles['pe']
        self.particleFluxRaw = fluxes['n']
        self.ionHeatFluxRaw = fluxes['pi']
        self.electronHeatFluxRaw = fluxes['pe']
        
        # boundary ... get in SI
        self.n_BC = n_BC
        self.Ti_keV_BC = Ti_keV_BC
        self.Te_keV_BC = Te_keV_BC
        self.pi_BC = Ti_keV_BC * 1000 * e * n_BC
        self.pe_BC = Te_keV_BC * 1000 * e * n_BC
        
        # Set up a grid mapper
        xExtrapZoneLeft = rhoExtrapZoneLeft * minorRadius
        xExtrapZoneRight = rhoExtrapZoneRight * minorRadius
        polynomialDegree = 0
        
        # for testing extrapolation at inner boundary
        rhoInnerExtrapZoneLeft = 0.55
        rhoInnerExtrapZoneRight = 0.60
        xInnerExtrapZoneLeft = rhoInnerExtrapZoneLeft * minorRadius
        xInnerExtrapZoneRight = rhoInnerExtrapZoneRight * minorRadius
        
        
        
#        self.gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(
#                xTango, xTurb, xExtrapZoneLeft, xExtrapZoneRight, polynomialDegree)
        
        self.gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffsBothSides(
                xTango, xTurb,
                xInnerExtrapZoneLeft, xInnerExtrapZoneRight,
                xExtrapZoneLeft, xExtrapZoneRight,
                polynomialDegree)
    
        # set up flux smoother for spatial averaging of flux
        windowSizeInGyroradii = 5
        rhoref = tango.genecomm_unitconversion.rho_ref(Tref, mref, Bref)
        windowSizeInPoints = int(np.round(windowSizeInGyroradii * rhoref / self.dxTurb))
        self.fluxSmoother = tango.smoother.Smoother(windowSizeInPoints)
        
        # apply smoothing
        self.particleFlux = self.fluxSmoother.smooth(self.particleFluxRaw)
        self.ionHeatFlux = self.fluxSmoother.smooth(self.ionHeatFluxRaw)
        self.electronHeatFlux = self.fluxSmoother.smooth(self.electronHeatFluxRaw)
        
        # run set-up analysis
        self.initial_analysis()
        
    def initial_analysis(self):
        """Compute Dhat, cHat for each field."""
        # get profiles on the turbulent grid
        self.nTurb = self.gridMapper.map_profile_onto_turb_grid(self.n)
        self.piTurb = self.gridMapper.map_profile_onto_turb_grid(self.pi)
        self.peTurb = self.gridMapper.map_profile_onto_turb_grid(self.pe)
        
        dndx = tango.derivatives.dx_centered_difference_edge_first_order(self.nTurb, self.dxTurb)
        dpidx = tango.derivatives.dx_centered_difference_edge_first_order(self.piTurb, self.dxTurb)
        dpedx = tango.derivatives.dx_centered_difference_edge_first_order(self.peTurb, self.dxTurb)
        
        # get fluxes on the turbulent grid.  assume <|nabla x|> = 1
        self.n_DhatTurb = -self.particleFlux / (self.gxxAvgTurb * dndx)
        self.n_cHatTurb = self.particleFlux / (self.gradxAvgTurb * self.nTurb)
        
        self.pi_DhatTurb = -self.ionHeatFlux / (self.gxxAvgTurb * dpidx)
        self.pi_cHatTurb = self.ionHeatFlux / (self.gradxAvgTurb * self.piTurb)

        self.pe_DhatTurb = -self.electronHeatFlux / (self.gxxAvgTurb * dpedx)
        self.pe_cHatTurb = self.electronHeatFlux / (self.gradxAvgTurb * self.peTurb)
        
        
        
    def solve_for_all_profiles(self):
        pass
    
    def solve_for_new_n(self, densityHCoeffs, dt=1e6):
        """Functionality to easily solve for the new iterate of the profile given the density for H1, ..., H8.  Test it out."""
        # create a fieldgroup
        fg = tango.fieldgroups.UncoupledFieldGroup('n')
        
        # set some parameters        
        matrixEqn = fg.Hcoeffs_to_matrix_eqn(dt, self.dxTango, self.n_BC, self.n, densityHCoeffs)
        
        # solve for new density
        profileSolution = fg.solve_matrix_eqn(matrixEqn)
        new_n = profileSolution['n']
        return new_n
    
    def solve_for_new_pi(self, piHCoeffs, dt):
        # create a fieldgroup
        fg = tango.fieldgroups.UncoupledFieldGroup('pi')
        
        # set some parameters        
        matrixEqn = fg.Hcoeffs_to_matrix_eqn(dt, self.dxTango, self.pi_BC, self.pi, piHCoeffs)
        
        # solve for new pi
        profileSolution = fg.solve_matrix_eqn(matrixEqn)
        new_pi = profileSolution['pi']
        return new_pi
        
    
    def solve_for_new_pi_pe(self, piHCoeffs, peHCoeffs, dt=1e6):
        """Functionality to easily solve for the new iterate of the profile given the density for H1, ..., H8.  Test it out."""
        # create a fieldgroup
        fg = tango.fieldgroups.PairCoupledFieldGroup('pi', 'pe')
        # attach the HCoeffs
        fg.HCoeffs = tango.fieldgroups.Hcoeffs_to_JKcoeffs(piHCoeffs, peHCoeffs)
        # create the boundary conditions on the right side of the domain
        BC = (self.pi_BC, self.pe_BC)
        # create the profiles from the previous timestep
        psi_mminus1 = (self.pi, self.pe)
        
        # create matrix equation
        matrixEqn = fg.Hcoeffs_to_matrix_eqn(dt, self.dxTango, BC, psi_mminus1, fg.HCoeffs)
        
        # solve for new profiles
        profileSolution = fg.solve_matrix_eqn(matrixEqn)
        new_pi = profileSolution['pi']
        new_pe = profileSolution['pe']
        return (new_pi, new_pe)
    
    def compute_densityHCoeffs(self, turboption=1):
        # H1 (Vprime)
        # adhoc diffusivity -> H2
        # H7 (source * Vprime)
        
        # handling of turbulent flux:
        #  --> H2, H3 using ftheta
        #  --> H4 (all explicit)
        # ---> some combination of H2, H3, H4 ???
        H1 = self.VprimeTango
        H2_adhoc = self.n_D_adhoc * self.VprimeTango * self.gxxAvgTango

        H7 = self.VprimeTango * Sn_func(self.rhoTango)
        HCoeffs_NoTurbulence = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H7=H7)
        
        # Turbulent flux handling
        HCoeffs_Turb = self.HCoeff_turbflux(self.nTurb, self.particleFlux, self.n_DhatTurb, self.n_cHatTurb, turboption)
        
        # Combine the terms
        HCoeffs = HCoeffs_NoTurbulence + HCoeffs_Turb
        return HCoeffs
    
    def compute_piHCoeffs(self, turboption=1):
        n = self.n
        pe = self.pe
        # H1 (Vprime)
        # adhoc diffusivity -> H2
        # H7 (source * Vprime)
        
        # handling of turbulent flux:
        #  --> H2, H3 using ftheta
        #  --> H4 (all explicit)
        # ---> some combination of H2, H3, H4 ???
        H1 = 3/2 * self.VprimeTango
        H2_adhoc = self.pi_D_adhoc * self.VprimeTango * self.gxxAvgTango
        
        H7 = self.VprimeTango * Si_func(self.rhoTango)
        
        # ion-electron collisional temperature equilibration: H6, H8
        nu0 = calc_nu0(n, pe/n)
        H6 = -nu0 * self.VprimeTango
        H8 = nu0 * self.VprimeTango

        HCoeffs_NoTurbulence = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H6=H6, H7=H7, H8=H8)
        
        # Turbulent flux handling
        HCoeffs_Turb = self.HCoeff_turbflux(self.nTurb, self.ionHeatFlux, self.pi_DhatTurb, self.pi_cHatTurb, turboption)
        
        # Combine the terms
        HCoeffs = HCoeffs_NoTurbulence + HCoeffs_Turb
        return HCoeffs
    
    def compute_peHCoeffs(self, turboption=1):
        n = self.n
        pe = self.pe
        # H1 (Vprime)
        # adhoc diffusivity -> H2
        # H7 (source * Vprime)
        
        # handling of turbulent flux:
        #  --> H2, H3 using ftheta
        #  --> H4 (all explicit)
        # ---> some combination of H2, H3, H4 ???
        H1 = 3/2 * self.VprimeTango
        H2_adhoc = self.pe_D_adhoc * self.VprimeTango * self.gxxAvgTango
        
        H7 = self.VprimeTango * Se_func(self.rhoTango)
        
        # ion-electron collisional temperature equilibration
        nu0 = calc_nu0(n, pe/n)
        H6 = -nu0 * self.VprimeTango
        H8 = nu0 * self.VprimeTango
        
        HCoeffs_NoTurbulence = tango.multifield.HCoefficients(H1=H1, H2=H2_adhoc, H6=H6, H7=H7, H8=H8)
        
        # Turbulent flux handling
        HCoeffs_Turb = self.HCoeff_turbflux(self.nTurb, self.electronHeatFlux, self.pe_DhatTurb, self.pe_cHatTurb, turboption)
        
        # Combine the terms
        HCoeffs = HCoeffs_NoTurbulence + HCoeffs_Turb
        return HCoeffs
    
    def HCoeff_turbflux(self, profileTurb, fluxTurb, DHatTurb, cHatTurb, turboption=1):
        if turboption == 1:
            HCoeffs_Turb = self.Hcoeff_turbflux_ftheta(profileTurb, fluxTurb, DHatTurb, cHatTurb)
        elif turboption == 2:
            HCoeffs_Turb = self.Hcoeff_turbflux_H4(profileTurb, fluxTurb)
        return HCoeffs_Turb
    
#    def ftheta(self, Dhat, cHat, profile):
#        thetaparams = self.thetaparams
#        dprofdx = tango.derivatives.dx_centered_difference_edge_first_order(profile, self.dxTurb)
        
        
        
    def Hcoeff_turbflux_ftheta(self, profileTurb, fluxTurb, DHatTurb, cHatTurb):
        """represent turbulent flux with H2, H3, using some function to split into convective and diffusive parts"""
        # compute theta with callable custom_ftheta
        #theta = ftheta(DHatTurb, cHatTurb, profileTurb)
        #dprofdxTurb = tango.derivatives.dx_centered_difference_edge_first_order(profileTurb, self.dxTurb)
        theta = self.ftheta(DHatTurb, cHatTurb, profileTurb, self.dxTurb)
        
        # Compute D and c from theta DHat, cHat
        DTurb = theta * DHatTurb
        cTurb = (1 - theta) * cHatTurb
        
        # Map the transport coefficients from the turbulence grid back to the transport grid
        (D, c) = self.gridMapper.map_transport_coeffs_onto_transport_grid(DTurb, cTurb)
        
        #### kill the extrapolation
        #c[47:] = 0
        
        # Compute H2contrib, H3contrib from D and c
        H2contrib = self.VprimeTango * D * self.gxxAvgTango
        H3contrib = -self.VprimeTango * c * self.gradxAvgTango
        HCoeffsTurb = tango.multifield.HCoefficients(H2=H2contrib, H3=H3contrib)
        return HCoeffsTurb
    
    def Hcoeff_turbflux_H4(self, profileTurb, fluxTurb):
        """represent turbulent flux with H4 only. (i.e., explicit)"""
        flux = self.gridMapper.map_to_transport_grid(fluxTurb)
        H4contrib = -self.VprimeTango * flux
        HCoeffsTurb = tango.multifield.HCoefficients(H4=H4contrib)
        return HCoeffsTurb
    
    def Hcoeff_turbflux_combo(self):
        """represent turbulent flux with some combination of H2, H3 H4"""
        pass
    
    
    def custom_ftheta(self, DHat, dndx, thetaParamsDensity):
        pass
        


# source functions
def Sn_func(rho):
    """Particle source.  To be later multiplied by V'
    
    note: total input # particles/second is a * integral(V' Sn, [rho, 0, 0.85]) = a * np.trapz(Vprime*Sn, rho)
    """
    # Sn = 9.06177465714e20 * np.exp(-(rho - 0.80)**2 / 0.05**2) # for positive triangularity
    Sn = 8.80176086347e20 * np.exp(-(rho - 0.80)**2 / 0.05**2) # for negative triangularity.  totalN = 2e20
    return Sn

def Si_func(rho):
    """Ion heat source.  To be later multiplied by V'
    """
    #Si = 1.42372145293e5 * np.exp(-(rho - 0.60)**2 / 0.10**2)  # positive triangularity
    Si = 1.4428908097e5 * np.exp(-(rho - 0.60)**2 / 0.10**2) # negative triangularity
    Si *= 4  # to get to 0.2 MW
    return Si


def Se_func(rho):
    """Electron heat source.  To be later multiplied by V'
    """
    # Se = 4.2711643588e6 * np.exp(-(rho - 0.60)**2 / 0.10**2) # positive triangularity
    Se = 4.328672429e6 * np.exp(-(rho - 0.60)**2 / 0.10**2)  # negative triangularity
    Se *= 2 # to get to 3 MW
    return Se

# plotting conveniences
    
def calc_nu0(n, Te):
    """given n and Te in SI units, calculate nu0 in SI units.
    
    nu0 is the collisional energy exchange frequency.
    
    Valid for array input n, Te"""
    e = 1.60217662e-19   # electron charge
    # from nrl formulary
    ionMass = 2 # measured in proton masses
    logLambda = 10
    ionCharge = 1
    electronDensity = n
    Te_eV = Te / e
    nuE_ie = 3.2e-15 * electronDensity * ionCharge**2 * logLambda / (ionMass * Te_eV**(3/2))
    
    return nuE_ie