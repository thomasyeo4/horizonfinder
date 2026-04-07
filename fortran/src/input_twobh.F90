! ============================================================================
! input_twobh.F90 - Input module for two black holes in Brill-Lindquist data
! ============================================================================
! Run with:
!   ./bin/twobh
!
! Notes:
! - find_indiv = .true. because the solver will first look for the two
!   individual horizons and then the common horizon.
! - The common horizon finder may fail for sufficiently large z0.
! ============================================================================

module user_input

    implicit none

    ! --- Physical system ---
    character(len=64), parameter :: system_name = "Two Black Holes"
    character(len=64), parameter :: coord_sys   = "Brill-Lindquist"

    ! --- Find individual horizons? ---
    logical, parameter :: find_indiv = .true.

    ! --- Number of black holes ---
    integer, parameter :: n_blackholes = 2

    ! --- BH names ---
    character(len=64), parameter :: bh_names(2) = [character(len=64) :: "BH1", "BH2"]

    ! --- Physical parameters ---
    real(8), parameter :: M1 = 1.0d0
    real(8), parameter :: M2 = 1.0d0
    real(8), parameter :: z0 = 0.5d0

    ! --- Grid resolution ---
    integer, parameter :: Ntheta_indiv  = 64
    integer, parameter :: Nphi_indiv    = 64
    integer, parameter :: Ntheta_common = 64
    integer, parameter :: Nphi_common   = 64

    ! --- Solver tolerances ---
    character(len=64), parameter :: ksp_type = "gmres"
    character(len=64), parameter :: pc_type  = "ilu"
    real(8),  parameter :: ksp_rtol   = 1.0d-6
    real(8),  parameter :: ksp_atol   = 1.0d-6
    real(8),  parameter :: ksp_stol   = 1.0d4
    integer,  parameter :: ksp_max_iter = 5000

    ! --- Outer iteration parameters ---
    integer, parameter :: max_iter = 5000
    real(8),  parameter :: omega    = 0.3d0

    ! --- Output directory ---
    character(len=256), parameter :: output_dir = "./data"

contains

    ! ------------------------------------------------------------------------
    ! psi: conformal factor for two black holes in Brill-Lindquist data
    ! ------------------------------------------------------------------------
    function psi(r, theta, phi) result(val)
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val

        real(8), parameter :: epsilon = 1.0d-12
        real(8) :: rr, x, y, z, r1, r2

        rr = max(r, epsilon)

        x = rr * sin(theta) * cos(phi)
        y = rr * sin(theta) * sin(phi)
        z = rr * cos(theta)

        r1 = sqrt(x**2 + y**2 + (z + z0)**2)
        r2 = sqrt(x**2 + y**2 + (z - z0)**2)

        r1 = max(r1, epsilon)
        r2 = max(r2, epsilon)

        val = 1.0d0 + M1 / (2.0d0 * r1) + M2 / (2.0d0 * r2)
    end function psi


    ! ------------------------------------------------------------------------
    ! psi_bh: conformal factor centered on the requested black hole
    ! ------------------------------------------------------------------------
    function psi_bh(bh_idx, r, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val

        real(8), parameter :: epsilon = 1.0d-12
        real(8) :: rr, x, y, z_global, r1, r2, z_shift

        rr = max(r, epsilon)

        if (bh_idx == 1) then
            z_shift = -z0
        else
            z_shift = z0
        end if

        x = rr * sin(theta) * cos(phi)
        y = rr * sin(theta) * sin(phi)
        z_global = z_shift + rr * cos(theta)

        r1 = sqrt(x**2 + y**2 + (z_global + z0)**2)
        r2 = sqrt(x**2 + y**2 + (z_global - z0)**2)

        r1 = max(r1, epsilon)
        r2 = max(r2, epsilon)

        if (bh_idx == 1) then
            val = 1.0d0 + M1 / (2.0d0 * r1) + M2 / (2.0d0 * r2)
        else
            val = 1.0d0 + M2 / (2.0d0 * r2) + M1 / (2.0d0 * r1)
        end if
    end function psi_bh


    ! ------------------------------------------------------------------------
    ! Kij: extrinsic curvature - zero for a time-symmetric slice
    ! ------------------------------------------------------------------------
    subroutine Kij(r, theta, phi, K)
        real(8), intent(in)  :: r, theta, phi
        real(8), intent(out) :: K(3,3)

        K = 0.0d0
    end subroutine Kij


    ! ------------------------------------------------------------------------
    ! hguess: initial guess for the common horizon shape
    ! ------------------------------------------------------------------------
    function hguess(theta, phi) result(val)
        real(8), intent(in) :: theta, phi
        real(8)             :: val

        val = 2.0d0
    end function hguess


    ! ------------------------------------------------------------------------
    ! hguess_bh: initial guess for the individual horizon shape
    ! ------------------------------------------------------------------------
    function hguess_bh(bh_idx, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: theta, phi
        real(8)             :: val

        if (bh_idx == 1) then
            val = 0.5d0 * M1
        else
            val = 0.5d0 * M2
        end if
    end function hguess_bh

end module user_input