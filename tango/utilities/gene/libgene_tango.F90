!#include "epik_user.inc"
#include "redef.h"
#include "intrinsic_sizes.h"

subroutine init_mpi(rank)
  Use mpi
  implicit none
  integer, intent(out):: rank
  integer :: ierr
  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
end subroutine
  
subroutine finalize_mpi()
  Use mpi
  integer :: ierr
  call mpi_finalize(ierr)
end subroutine
 
!>GENE-Tango interface (a modified version of the GENE/TRINITY interface by T. Goerler)
!!
!!\TODO in GENE:
!!* momentum flux?
!!* global ExB
!!
!! \param n_transp_it_in Transport code iteration number
!! \param electrostatic Switch to enforce electrostatic runs
!! \param rank Return MPI rank to Tango to enable writing on only one process despite not having initialized MPI in Python
!! \param px0_in number of radial grid points used for profiles and simulation
!! \param n_spec_in number of species (has to be <= number of species namelist in parameter file)
!! \param temp_io Array (rad. grid, n_spec_in) containing the temperature in keV
!! \param dens_io Array (rad. grid, n_spec_in) containing the densities in 10^19 m^-3
!! \param mass_in Array containing species masses
!! \param vrot_in global code: vrot in rad/s
!! \param charge_in Array containing species charges
!! \param rhostar rho_ref/a, set by Tango
!! \param q_in Safety factor
!! \param shat_in Magnetic shear
!! \param aspr_in aspect ratio to renormalize to major R for s_alpha or circular model
!! \param Lref_in reference length only required for analytical models else used for consistency check
!! \param Bref_in reference magnetic field only required for analytical models else used for consistency check
!! \param o_dVdx Radial derivative of volume in units of [Lref^2] defined on rho_grid
!! \param o_sqrtgxx_FS Flux surface avg. of sqrt(gxx) defined on rho_grid
!! \param o_avgpflx Time avg'd.particle flux in gyro-Bohm units defined on rho_grid
!! \param o_avgqflx Time avg'd heat flux in gyro-Bohm units defined on rho_grid
!! \param o_temp_bin Time avg'd temperature profile defined on rho_grid
!! \param o_dens_bin Time avg'd density profile defined on rho_grid
subroutine gene_tango (n_transp_it_in, electrostatic, rank, simtimelim_in, & !administrative stuff
     &px0_in,rho_grid,& !grids
     &n_spec_in, temp_io, dens_io, mass_in, charge_in, vrot_in, & !profiles
     &rhostar_in, & !allow Tango to set correct rhostar
     &Tref_in, nref_in, & !use initial temperature and density at x0 as Tref and nref
     &q_in, shat_in, aspr_in, Lref_in, Bref_in,& !only required for analytical geometries
     &o_dVdx, o_sqrtgxx_FS, o_avgpflx, o_avgqflx, o_temp_bin, o_dens_bin) ! return values
  ! Note.  MPI must already be initialized for this subroutine to work: mpi_init() must have already been called
  Use mpi
  Use gene_subroutine
  Use parameters_IO
  Use communications
  use profiles_mod, only: set_spec_profiles, unset_spec_profiles, set_ref_vals
  use lagrange_interpolation

  Implicit None

  integer, intent (in) :: n_transp_it_in ! for GENE output files index
  integer, intent (in) :: px0_in ! number of radial points used for input profiles (1 for local)
  logical, intent (in) :: electrostatic
  integer, intent(out):: rank

  !Note that upon reading a checkpoint, the simulation time will always be reset to zero when using this library!
  real(kind=8), intent(in) :: simtimelim_in !time limit in simulation units
  real(kind=8), intent(in) :: rhostar_in !rho_ref/a


  !Note the explicit kind specification, which is required for f2py to recognize that we're dealing with double
  !precision here. When compiling GENE with single precision, this could lead to problems.
  real(kind=8), dimension (px0_in), intent (in) :: rho_grid   !grid for input profiles (r/a for analytical geom., rho_tor else)
  integer, intent (in) :: n_spec_in !number of species
  real(kind=8), dimension (px0_in,0:n_spec_in-1), intent (inout) :: temp_io, dens_io !normalized temperature/density profiles
  real(kind=8), dimension (0:n_spec_in-1), intent (in) :: mass_in, charge_in !mass in proton mass, charge in elementary charge
  real(kind=8), dimension (px0_in), intent (in) :: vrot_in !tor. angular vel. vrot (global) or -ExBrate (local)
  real(kind=8), dimension (px0_in), intent (in) :: q_in, shat_in !for analytical geometries *only*

  real(kind=8), intent(in) :: aspr_in, Lref_in, Bref_in !for analytical geometries and consistency checks *only*
  real(kind=8), intent(in) :: Tref_in, nref_in
  ! following are binned (averaged) mean quantities and gradients from global GENE sim on rad_out_gene grid
  real(kind=8), dimension (px0_in), intent (out) :: o_dVdx, o_sqrtgxx_FS
  real(kind=8), dimension (px0_in,0:n_spec_in-1), intent (out) :: o_avgpflx, o_avgqflx
  real(kind=8), dimension (px0_in,0:n_spec_in-1), intent (out) :: o_temp_bin, o_dens_bin

  !local variables
  character(len=FILENAME_MAX):: tango_par_in_dir='',tango_checkpoint_in=''
  character(len=FILEEXT_MAX):: tango_file_extension
  integer :: file_unit = -1, is, ierr
  logical :: flag

  !Call mpi_init(ierr)
  call mpi_initialized(flag, ierr)  ! flag is set to true if MPI has been initialized
  if (.NOT. flag)  Stop 'mpi must be initialized before calling gene_tango()!'
  

  Call mpi_comm_size (MPI_COMM_WORLD, n_procs_sim, ierr)
  If (ierr /= 0) Stop 'mpi_comm_size failed!'
  call mpi_comm_rank (MPI_COMM_WORLD, rank, ierr)

  if (n_spec_in.gt.1) then
     !enforce quasineutrality
     dens_io(:,n_spec_in-1) = 0.0
     do is = 0, n_spec_in-2
        dens_io(:,n_spec_in-1) = dens_io(:,n_spec_in-1)-&
             &(charge_in(is)/charge_in(n_spec_in-1))*dens_io(:,is)
     enddo
  endif

  !change file extension .dat to numbers
  WRITE(tango_file_extension,"(A,I3.3)") "_",n_transp_it_in

  !set all n_procs to 1 to "fool" the check_parameters
  !subroutine; the actual values will be read later in the program
  n_procs_s = 1; n_procs_v = 1; n_procs_w = 1
  n_procs_x = 1; n_procs_y = 1; n_procs_z = 1

  tango_par_in_dir = ''
  print_ini_msg = .false.

  x0 = 0.5 ! --- need to look for a better solution (tbg)
  call read_parameters('')
  if (magn_geometry.eq.'tracer') then
     file_unit = -1
     call read_tracer_namelist(file_unit,.true.,.false.)
  endif

  !check compatibility with input parameters
  if (n_spec.ne.n_spec_in) STOP &
       'Number of species in parameters file and interface do not match!'

  call Modify_parameters

  !IMPORTANT: The following part ensures that any modified parameters will not get
  !overwritten again by entries present in the GENE parameter file
  tango_par_in_dir = 'skip_parfile'
  call check_params(tango_par_in_dir)

  call read_parall_nml('')

  call rungene(MPI_COMM_WORLD,tango_par_in_dir,tango_file_extension,&
       &tango_checkpoint_in)

  call Prepare_Output
  if (allocated(in_profiles)) deallocate(in_profiles)
  if (allocated(in_qprof)) deallocate(in_qprof)
  if (allocated(spec)) deallocate(spec)
  if (allocated(dVdx)) deallocate(dVdx)
  if (allocated(sqrtgxx_FS)) deallocate(sqrtgxx_FS)
  if (allocated(area_FS)) deallocate(area_FS)

  !Call mpi_finalize(ierr)

  !*************************************************************************!
  !************************* End of main program  **************************!
  !*************************************************************************!

Contains

  !>Replace parameters read from the parameters input file
  !!by values passed from Tango
  Subroutine Modify_parameters
    IMPLICIT NONE
    Integer :: n
    real :: mp_o_e = 1.043968E-8 !kg/C

    x_local = .false.
    nx0=px0_in

    x0 = 0.5
    allocate(in_profiles(1:px0_in,0:n_spec-1,0:4))
    mag_prof = .true.
    if (init_cond.eq.'ppj') init_cond='db'

    if ((trim(magn_geometry).ne.'tracer').and.&
         &(trim(magn_geometry).ne.'tracer_efit')) then
       allocate(in_qprof(1:px0_in))
       in_qprof = q_in
    endif

    !basic species settings
    mref = -1.
    do n=0, n_spec-1
       spec(n)%dens = 1.0
       spec(n)%temp = 1.0
       spec(n)%omt = 0.0
       spec(n)%omn = 0.0
       spec(n)%prof_type = 10
       in_profiles(:,n,0) = rho_grid
       in_profiles(:,n,1) = temp_io(:,n)
       in_profiles(:,n,3) = dens_io(:,n)
       spec(n)%charge = charge_in(n)
       if (spec(n)%charge .ge. 0. .and. mref .lt. 0) mref = mass_in(n)
       spec(n)%mass = mass_in(n) / mref
    enddo

    !DTOLD, 12-9-16, use Tref and nref as input parameters from Tango.
    !Expect temperature profiles to be given in keV, density in 10^19 cm^-3.
    !Tref and nref are set to initial values at x0.
    Tref=Tref_in
    nref=nref_in

    !DTOLD, 12-1-16, use rhostar provided by Tango
    rhostar=rhostar_in

    Lref = Lref_in !will be overwritten for non-analytic equil.
    Bref = abs(Bref_in) !will be overwritten for non-analytic equil.
    norm_index = 0

    if (electrostatic) then
		if (n_spec_in.eq.1) then
          beta = 0.  ! assuming adiabatic electrons
       else
          beta = 1E-4  ! assuming kinetic electrons
       endif
    else
       beta = -1
    endif
!commented out, dtold 08/19
!    coll = -1
!    debye2 = -1
!    rhostar = (SQRT(mref*1000*Tref*mp_o_e) / Bref) / (minor_r*Lref)

    !depending on simulation length for Tango, istep_prof may have to be decreased
    lx = (rho_grid(px0_in)-rho_grid(1))/rhostar
    if (istep_prof.le.0) istep_prof=50
    if (ck_heat.le.0.0) ck_heat = 0.05
    if ((n_spec.gt.1).and.(ck_part.le.0.0)) ck_part = 0.05

    !switch on calculation of time averaged fluxes if not set
    !in parameters
    if (avgflux_stime.lt.0.0) avgflux_stime = 0.0
    if (avgprof_stime.lt.0.0) avgprof_stime = 0.0 !needed for global runs

    !set simulation time limit according to input
    simtimelim = simtimelim_in

    !check for checkpoints: first, try to take those with the same file extension
    !(i.e. continue a GENE run)
    !if not present, try to take those from the previous iteration
    !(i.e. resume with slightly modified gradients -> avoid big overshoots??)
    if (read_checkpoint) then
       reset_chpt_time = .true.
       tango_checkpoint_in = TRIM(chptdir)//'/checkpoint'//tango_file_extension
       if (.not.valid_chpt(MPI_COMM_WORLD,tango_checkpoint_in)) then
          !try secure checkpoint
          tango_checkpoint_in = TRIM(chptdir)//'/s_checkpoint'//tango_file_extension
          if (.not.valid_chpt(MPI_COMM_WORLD,tango_checkpoint_in)) then
             !try checkpoint from previous run
             WRITE(tango_checkpoint_in,"(2A,(I2.2))") TRIM(chptdir),'/checkpoint_',(n_transp_it_in-1)
             if (.not.valid_chpt(MPI_COMM_WORLD,tango_checkpoint_in)) then
                tango_checkpoint_in = 'no' !i.e. switch off checkpoint read
             endif
          endif
       endif
    else
       tango_checkpoint_in='no'
    endif

    print_ini_msg = .false.
    multiple_tracer_files = .true.

  End Subroutine Modify_parameters

  !>Prepare return values/arrays for Tango
  Subroutine Prepare_Output
    IMPLICIT NONE

    integer :: n, ix
    Real, dimension (px0_in,0:n_spec-1) :: o_omn_bin, o_omt_bin

    o_avgpflx = 0.0
    o_avgqflx = 0.0

    if ((time-avgprof_stime).ge.0.0) then
       do n=0,n_spec-1
          avgprof(:,n,2) = avgprof(:,n,2)*spec(n)%temp
          avgprof(:,n,1) = avgprof(:,n,1)*spec(n)%dens

!DTOLD 10/24/16: this should not matter for Tango, as it doesn't require profile feedback from GENE
!tbg: return current values instead of trinity values?
!maybe, in the future - for now, keep input values
!             call lag3interp(avgprof(:,n,2),avgprof(:,n,0),&
!                  &size(avgprof(:,n,0)),temp_io(:,n),rho_grid,&
!                  &px0_in)
!             call lag3interp(avgprof(:,n,1),avgprof(:,n,0),&
!                  &size(avgprof(:,n,0)),dens_io(:,n),rho_grid,&
!                  &px0_in)
!add units (see below)


!\Todo get rid of duplicate output arrays for background profiles and gradients
          o_dens_bin(:,n)=avgprof(:,n,1)
          o_temp_bin(:,n)=avgprof(:,n,2)
          if (comp_type.ne.'NC') then
             o_avgpflx(:,n)=avgprof(:,n,5)
             o_avgqflx(:,n)=avgprof(:,n,6)
          else
             o_avgpflx(:,n)=neoflux(:,n,1)
             o_avgqflx(:,n)=neoflux(:,n,2)
          endif

          !COMMENTED OUT DTOLD 10/24/16
          !change normalization from center value to 'local'
          !normalization in each bin (always w.r.t. to first species)
          !o_avgpflx(:,n) = o_avgpflx(:,n)/&
          !     &(o_dens_bin(:,0)*(o_temp_bin(:,0))**1.5)
          !o_avgqflx(:,n) = o_avgqflx(:,n)/&
          !     &(o_dens_bin(:,0)*(o_temp_bin(:,0))**2.5)
       enddo
    endif

    o_dVdx=dVdx
    o_sqrtgxx_FS=sqrtgxx_FS

    avgflux_stime = -1.0
    avgprof_stime = -1.0

    !DTOLD 10/24/16 leave units to Tango
    !o_temp_bin = o_temp_bin * Tref
    !o_dens_bin = o_dens_bin * nref
    !dens_io = dens_io

End Subroutine Prepare_Output

  !>Calculates radial averages in bins being
  !!centered between neighbouring output grid points.
  !!This routine fails if nx0<n_in -- ensure the radial resolution is sufficient!
  Subroutine calc_avg_in_bins(inarr,rad_in,n_in,outarr,rad_out,n_out)

    Implicit None

    Integer, Intent(in) :: n_in, n_out

    Real, dimension(n_in) :: inarr, rad_in
    Real, dimension(n_out) :: outarr, rad_out

    Integer :: i,j,count
    Real :: xstart, xend

    outarr = 0.0
    Do i=1, n_out
       if (i.eq.1) then
          xstart = rad_in(1)
       else
          xstart = 0.5*(rad_out(i-1)+rad_out(i))
       endif
       if (i.eq.n_out) then
          xend = rad_in(n_in)
       else
          xend = 0.5*(rad_out(i)+rad_out(i+1))
       endif
       count = 0
       Do j=1,n_in
          if ((rad_in(j).ge.xstart).and.(rad_in(j).le.xend)) then
             count = count+1
             outarr(i) = outarr(i) + inarr(j)
          else
             cycle
          endif
       enddo
       if (count.eq.0) stop 'binning failed - use more input grid points'
       outarr(i) = outarr(i)/count
    Enddo

  End Subroutine calc_avg_in_bins


  logical function valid_chpt(mpi_comm_in,chpt_str)
    Implicit None
    integer,intent(in) :: mpi_comm_in
    character(len=*),intent(in) :: chpt_str
    integer :: handle=MPI_FILE_NULL, ierr
    integer(MPI_OFFSET_KIND) :: offset

    inquire(file=TRIM(chpt_str),exist=valid_chpt)

    if ((.not.chpt_read_h5).and.(valid_chpt)) then !check for file size
       call mpi_file_open(mpi_comm_in,TRIM(chpt_str),&
            &MPI_MODE_RDONLY,MPI_INFO_NULL,handle,ierr)
       !check file size
       call mpi_file_get_size(handle,offset,ierr)
       call mpi_file_close(handle,ierr)
       if (abs(offset).le.6) valid_chpt = .false.
       !using abs(offset) for now as OpenMPI has an issue with files sizes >2GB
       !and returns neg. numbers: https://svn.open-mpi.org/trac/ompi/ticket/2145
    endif

  end function valid_chpt



End Subroutine gene_tango
