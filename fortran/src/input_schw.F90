! ==============================================================================
! input_schw.F90 — Input module for Schwarzschild in Isotropic coordinates
! ==============================================================================
! Run with:
!   ./bin/schw
!
! Note: find_indiv = .false. for this case — no individual horizons needed.
!       psi_bh, hguess_bh, bh_names are stubs required by the solver interface.
! ==============================================================================

module user_input

    implicit none

    ! --- Physical system ---
    character(len=64), parameter :: system_name = "Schwarzschild"
    character(len=64), parameter :: coord_sys   = "Isotropic"

    ! --- Find individual horizons? ---
    logical, parameter :: find_indiv  = .false.

    ! --- Number of black holes (0 since find_indiv = .false.) ---
    integer, parameter :: n_blackholes = 0

    ! --- BH names (stub — unused when find_indiv = .false.) ---
    character(len=64) :: bh_names(1) = ["BH1"]

    ! --- Physical parameters ---
    real(8), parameter :: M = 1.0d0

    ! --- Grid resolution ---
    integer, parameter :: Ntheta_common = 100
    integer, parameter :: Nphi_common   = 100

    ! --- Individual horizon grid (stubs — unused when find_indiv = .false.) ---
    integer, parameter :: Ntheta_indiv  = 64
    integer, parameter :: Nphi_indiv    = 64

    ! --- Solver tolerances ---
    character(len=64), parameter :: ksp_type     = "gmres"
    character(len=64), parameter :: pc_type      = "ilu"
    real(8),  parameter :: ksp_rtol     = 1.0d-6
    real(8),  parameter :: ksp_atol     = 1.0d-6
    real(8),  parameter :: ksp_stol     = 1.0d6
    integer,  parameter :: ksp_max_iter = 1000

    ! --- Outer iteration parameters ---
    integer,  parameter :: max_iter     = 500
    real(8),  parameter :: omega        = 0.3d0

    ! --- Output directory ---
    character(len=256), parameter :: output_dir = "./data"

contains

    ! --------------------------------------------------------------------------
    ! psi: conformal factor for Schwarzschild in isotropic coordinates
    !   psi = 1 + M / (2r)
    ! Horizon is at r = M/2 = 0.5 for M=1
    ! --------------------------------------------------------------------------
    function psi(r, theta, phi) result(val)
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val
        val = 1.0d0 + M / (2.0d0 * r)
    end function psi


    ! --------------------------------------------------------------------------
    ! Kij: extrinsic curvature — zero for time-symmetric slice
    ! --------------------------------------------------------------------------
    subroutine Kij(r, theta, phi, K)
        real(8), intent(in)  :: r, theta, phi
        real(8), intent(out) :: K(3,3)
        K = 0.0d0
    end subroutine Kij


    ! --------------------------------------------------------------------------
    ! hguess: initial guess for horizon shape
    ! Slightly non-spherical to avoid degenerate starting point
    ! --------------------------------------------------------------------------
    function hguess(theta, phi) result(val)
        real(8), intent(in) :: theta, phi
        real(8)             :: val
        val = 0.6d0
    end function hguess


    ! --------------------------------------------------------------------------
    ! psi_bh: stub — not used when find_indiv = .false.
    ! --------------------------------------------------------------------------
    function psi_bh(bh_idx, r, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: r, theta, phi
        real(8)             :: val
        val = psi(r, theta, phi)
    end function psi_bh


    ! --------------------------------------------------------------------------
    ! hguess_bh: stub — not used when find_indiv = .false.
    ! --------------------------------------------------------------------------
    function hguess_bh(bh_idx, theta, phi) result(val)
        integer, intent(in) :: bh_idx
        real(8), intent(in) :: theta, phi
        real(8)             :: val
        val = hguess(theta, phi)
    end function hguess_bh

end module user_input
