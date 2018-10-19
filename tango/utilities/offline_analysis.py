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

# physical constants with module-wide scope.  Given in SI units
e = 1.60217662e-19          # electron charge
mp = 1.6726219e-27          # proton mass

class analysis:
    def __init__(self, xTango, xTurb, profiles, fluxes, 
                 minorRadius, majorRadius, n_BC, Ti_keV_BC, Te_keV_BC,
                 Tref, mref, Bref, nref,
                 n_D_adhoc, pi_D_adhoc, pe_D_adhoc):
        self.xTango = xTango
        self.xTurb = xTurb
        self.dxTango = xTango[1] - xTango[0]
        self.dxTurb = xTurb[1] - xTurb[0]
        
        self.profiles = profiles
        self.fluxes = fluxes
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
        
        # for circular geometry
        self.VprimeTango = 4 * np.pi**2 * majorRadius * xTango
        self.VprimeTurb = 4 * np.pi**2 * majorRadius * xTurb
        
        # boundary ... get in SI
        self.n_BC = n_BC
        self.Ti_keV_BC = Ti_keV_BC
        self.Te_keV_BC = Te_keV_BC
        self.pi_BC = Ti_keV_BC * 1000 * e * n_BC
        self.pe_BC = Te_keV_BC * 1000 * e * n_BC
        
        # Set up a grid mapper
        rExtrapZoneLeft = 0.70 * minorRadius
        rExtrapZoneRight = 0.75 * minorRadius
        polynomialDegree = 1
        self.gridMapper = tango.interfacegrids_gene.TangoOutsideExtrapCoeffs(
                xTango, xTurb, rExtrapZoneLeft, rExtrapZoneRight, polynomialDegree)
        
        # set up flux smoother for spatial averaging of flux
        windowSizeInGyroradii = 10
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
        self.n_DhatTurb = -self.particleFlux / dndx
        self.n_cHatTurb = self.particleFlux / self.nTurb
        
        self.pi_DhatTurb = -self.ionHeatFlux / dpidx
        self.pi_cHatTurb = self.ionHeatFlux / self.piTurb

        self.pe_DhatTurb = -self.electronHeatFlux / dpedx
        self.pe_cHatTurb = self.electronHeatFlux / self.peTurb
        
        
        
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
        H2_adhoc = self.n_D_adhoc * self.VprimeTango
        
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
        H2_adhoc = self.pi_D_adhoc * self.VprimeTango
        
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
        H2_adhoc = self.pe_D_adhoc * self.VprimeTango
        
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
            HCoeffs_Turb = self.HCoeff_turbflux_H4()
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
        H2contrib = self.VprimeTango * D
        H3contrib = -self.VprimeTango * c
        HCoeffsTurb = tango.multifield.HCoefficients(H2=H2contrib, H3=H3contrib)
        return HCoeffsTurb
    
    def Hcoeff_turbflux_H4(self):
        """represent turbulent flux with H4 only. (i.e., explicit)"""
        pass
    
    def Hcoeff_turbflux_combo(self):
        """represent turbulent flux with some combination of H2, H3 H4"""
        pass
    
    
    def custom_ftheta(self, DHat, dndx, thetaParamsDensity):
        pass
    
    # build for 
    
    def n_H1(self):
        H1 = self.VprimeTango
        return tango.multifield.HCoefficients(H1=H1)
    
    def pi_H1(self):
        H1 = 3/2 * self.VprimeTango
        return tango.multifield.HCoefficients(H1=H1)
        
    def pe_H1(self):
        H1 = 3/2 * self.VprimeTango
        return tango.multifield.HCoefficients(H1=H1)
    
    def adhoc_H2(self, D_adhoc):
        """Caution!  geometric coefficients for |nabla x| != 1 are not included here"""
        H2_adhoc = D_adhoc * self.Vprime
        return tango.multifield.HCoefficients(H2=H2_adhoc)
    
    


# source functions
def Sn_func(rho):
    """Particle source.  To be later multiplied by V'
    
    note: total input # particles/second is a * integral(V' Sn, [rho, 0, 0.85]) = a * np.trapz(Vprime*Sn, rho)
    """
    pnfit = np.array([8.03e18, 1.44e19, 2.80e18])
    Sn = np.polyval(pnfit, rho)
    Sn *= 5 # manual adjustment
    return Sn

def Si_func(rho):
    """Ion heat source.  To be later multiplied by V'
    """
    pifit = np.array([-9.9e5, 1.18e6, -2.36e5, 5.51e4])
    Si = np.polyval(pifit, rho)
    Si *= 20 # manual adjustment
    return Si


def Se_func(rho):
    """Electron heat source.  To be later multiplied by V'
    """
    pefit = np.array([-3.54e6, 5.81e6, -2.71e6, 4.56e5, 8.14e2])
    Se = np.polyval(pefit, rho)
    Se *= 20 # manual adjustment
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