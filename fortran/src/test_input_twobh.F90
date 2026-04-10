! ============================================================================
! test_input_twobh.F90 - Test input module for two black holes.
! Reads runtime configuration from a namelist file (config.nml).
! Production code (input_twobh.F90) is left untouched.
! ============================================================================

module user_input

    implicit none

    ! --- Physical system ---
    character(len=64), parameter :: system_name = "Two Black Holes (Test)"
    character(len=64), parameter :: coord_sys   = "Brill-Lindquist"

    ! --- Number of black holes ---
    integer, parameter :: n_blackholes = 2

    ! --- BH names ---
    character(len=64), parameter :: bh_names(2) = [character(len=64) :: "BH1", "BH2"]

    ! --- Physical parameters (fixed) ---
    real(8), parameter :: M1 = 1.0d0
    real(8), parameter :: M2 = 1.0d0
    logical, parameter :: find_indiv = .false.

    ! --- Runtime-configurable variables (read from config.nml) ---
    real(8)  :: z0            = 0.5d0
    integer  :: Ntheta_common = 64
    integer  :: Nphi_common   = 64
    character(len=64)  :: ksp_type    = "gmres"
    
    ! --- Derived grid variables for individual horizons (fixed for tests) ---
    integer, parameter :: Ntheta_indiv = 64
    integer, parameter :: Nphi_indiv   = 64

    ! --- Solver tolerances (fixed) ---
    character(len=64), parameter :: pc_type     = "ilu"
    real(8),  parameter :: ksp_rtol             = 1.0d-6
    real(8),  parameter :: ksp_atol             = 1.0d-6
    real(8),  parameter :: ksp_stol             = 1.0d4
    integer,  parameter :: ksp_max_iter         = 5000
    integer,  parameter :: max_iter             = 5000
    real(8),  parameter :: omega                = 0.3d0

    ! --- Output directory ---
    character(len=256), parameter :: output_dir = "./data"

    ! --- Namelist declaration ---
    namelist /twobh_config/ z0, Ntheta_common, Nphi_common, ksp_type

contains

    ! ------------------------------------------------------------------------
    ! read_config: read namelist from config.nml
    ! ------------------------------------------------------------------------
    subroutine read_config(config_file)
        character(len=*), intent(in) :: config_file

        integer :: unit_num = 10
        integer :: ios

        open(unit=unit_num, file=trim(config_file), status='old', iostat=ios)
        if (ios /= 0) then
            write(*,*) "ERROR: could not open config file: ", trim(config_file)
            stop
        end if

        read(unit_num, nml=twobh_config, iostat=ios)
        if (ios /= 0) then
            write(*,*) "ERROR: failed to read namelist, iostat=", ios, " file=", trim(config_file)
            stop
        end if

        close(unit_num)

        write(*,'("Config loaded:")')
        write(*,'("  z0            = ", f8.4)') z0
        write(*,'("  Ntheta_common = ", i4)')   Ntheta_common
        write(*,'("  Nphi_common   = ", i4)')   Nphi_common
        write(*,'("  ksp_type      = ", A)')    ksp_type
    end subroutine read_config


    ! ------------------------------------------------------------------------
    ! psi, psi_bh, Kij, hguess, hguess_bh — identical to input_twobh.F90
    ! ------------------------------------------------------------------------
    function psi(r, theta, phi) result(val)
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val

        real(8), parameter :: epsilon = 1.0d-12
        real(8) :: rr, x, y, z, r1, r2

        rr = max(r, epsilon)
        x  = rr * sin(theta) * cos(phi)
        y  = rr * sin(theta) * sin(phi)
        z  = rr * cos(theta)

        r1 = max(sqrt(x**2 + y**2 + (z + z0)**2), epsilon)
        r2 = max(sqrt(x**2 + y**2 + (z - z0)**2), epsilon)

        val = 1.0d0 + M1/(2.0d0*r1) + M2/(2.0d0*r2)
    end function psi


    function psi_bh(bh_idx, r, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val

        real(8), parameter :: epsilon = 1.0d-12
        real(8) :: rr, x, y, z_global, r1, r2, z_shift

        rr = max(r, epsilon)
        z_shift = merge(-z0, z0, bh_idx == 1)

        x        = rr * sin(theta) * cos(phi)
        y        = rr * sin(theta) * sin(phi)
        z_global = z_shift + rr * cos(theta)

        r1 = max(sqrt(x**2 + y**2 + (z_global + z0)**2), epsilon)
        r2 = max(sqrt(x**2 + y**2 + (z_global - z0)**2), epsilon)

        if (bh_idx == 1) then
            val = 1.0d0 + M1/(2.0d0*r1) + M2/(2.0d0*r2)
        else
            val = 1.0d0 + M2/(2.0d0*r2) + M1/(2.0d0*r1)
        end if
    end function psi_bh


    subroutine Kij(r, theta, phi, K)
        real(8), intent(in)  :: r, theta, phi
        real(8), intent(out) :: K(3,3)
        K = 0.0d0
    end subroutine Kij


    function hguess(theta, phi) result(val)
        real(8), intent(in) :: theta, phi
        real(8)             :: val
        val = 2.0d0
    end function hguess


    function hguess_bh(bh_idx, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: theta, phi
        real(8)             :: val
        val = merge(0.5d0*M1, 0.5d0*M2, bh_idx == 1)
    end function hguess_bh

end module user_input