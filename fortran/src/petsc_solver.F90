#include "petsc/finclude/petsc.h"

module petsc_solver

    use petsc
    use user_input
    use hdf5
    use ieee_arithmetic

    implicit none

contains

    ! --------------------------------------------------------------------------
    ! build_A: construct the PETSc sparse LHS matrix
    ! L[h] = h_tt + cot(t)*h_t + h_pp/sin^2(t) - 2h
    ! Pole BCs from Shibata (2000) Eqs. 2.7-2.8
    ! --------------------------------------------------------------------------
    subroutine build_A(Ntheta, Nphi, dtheta, dphi, A)

        integer,  intent(in)  :: Ntheta, Nphi
        real(8),  intent(in)  :: dtheta, dphi
        Mat,      intent(out) :: A

        PetscErrorCode :: ierr
        PetscInt       :: n_tot, row, half
        PetscInt       :: i, j
        PetscInt       :: row_idx(1), col_idx(5)
        PetscScalar    :: mat_vals(5)

        real(8) :: theta, s, c, inv_dt2, inv_dp2
        real(8) :: c_phi, cot_t, c_theta_plus, c_theta_minus, c_diag

        n_tot = Ntheta * Nphi
        half  = Nphi / 2

        inv_dt2 = 1.0d0 / (dtheta**2)
        inv_dp2 = 1.0d0 / (dphi**2)

        call MatCreate(PETSC_COMM_SELF, A, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call MatSetType(A, MATSEQAIJ, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call MatSetSizes(A, n_tot, n_tot, n_tot, n_tot, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call MatSetUp(A, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

        do j = 1, Ntheta
            theta         = (j - 0.5d0) * dtheta
            s             = sin(theta)
            c             = cos(theta)
            cot_t         = c / s
            c_phi         = inv_dp2 / (s * s)
            c_theta_plus  = inv_dt2 + 0.5d0 * cot_t / dtheta
            c_theta_minus = inv_dt2 - 0.5d0 * cot_t / dtheta
            c_diag        = -2.0d0 * inv_dt2 - 2.0d0 * c_phi - 2.0d0

            do i = 1, Nphi

                row = (j-1)*Nphi + (i-1)
                row_idx(1) = row

                ! --- phi neighbours (periodic) ---
                col_idx(1) = (j-1)*Nphi + mod(i, Nphi)
                col_idx(2) = (j-1)*Nphi + mod(i-2+Nphi, Nphi)
                mat_vals(1) = c_phi
                mat_vals(2) = c_phi

                ! --- theta neighbours with pole BCs ---
                if (j == 1) then
                    col_idx(3) = 1*Nphi + (i-1)
                    col_idx(4) = 0*Nphi + mod(i-1+half, Nphi)
                    mat_vals(3) = c_theta_plus
                    mat_vals(4) = c_theta_minus

                else if (j == Ntheta) then
                    col_idx(3) = (j-2)*Nphi + (i-1)
                    col_idx(4) = (j-1)*Nphi + mod(i-1+half, Nphi)
                    mat_vals(3) = c_theta_minus
                    mat_vals(4) = c_theta_plus

                else
                    col_idx(3) = j*Nphi + (i-1)
                    col_idx(4) = (j-2)*Nphi + (i-1)
                    mat_vals(3) = c_theta_plus
                    mat_vals(4) = c_theta_minus

                end if

                ! --- Diagonal ---
                col_idx(5) = row
                mat_vals(5) = c_diag

                call MatSetValues(A, 1, row_idx, 5, col_idx, mat_vals, ADD_VALUES, ierr)
                if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

            end do
        end do

        call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

    end subroutine build_A


    ! --------------------------------------------------------------------------
    ! h_derivatives: finite difference derivatives of h on the grid
    ! --------------------------------------------------------------------------
    subroutine h_derivatives(h, Ntheta, Nphi, dtheta, dphi, &
                              ht, hp, htt, hpp, htp)

        integer, intent(in)  :: Ntheta, Nphi
        real(8), intent(in)  :: h(Ntheta,Nphi), dtheta, dphi
        real(8), intent(out) :: ht(Ntheta,Nphi), hp(Ntheta,Nphi)
        real(8), intent(out) :: htt(Ntheta,Nphi), hpp(Ntheta,Nphi)
        real(8), intent(out) :: htp(Ntheta,Nphi)

        integer :: i, j, ip, im, half
        real(8) :: h_shift_n(Nphi), h_shift_s(Nphi)

        half = Nphi / 2

        ! --- phi derivatives (periodic) ---
        do i = 1, Nphi
            ip = mod(i, Nphi) + 1
            im = mod(i-2+Nphi, Nphi) + 1
            do j = 1, Ntheta
                hp(j,i)  = (h(j,ip) - h(j,im)) / (2.0d0*dphi)
                hpp(j,i) = (h(j,ip) - 2.0d0*h(j,i) + h(j,im)) / (dphi**2)
            end do
        end do

        ! --- theta derivatives interior ---
        do i = 1, Nphi
            do j = 2, Ntheta-1
                ht(j,i)  = (h(j+1,i) - h(j-1,i)) / (2.0d0*dtheta)
                htt(j,i) = (h(j+1,i) - 2.0d0*h(j,i) + h(j-1,i)) / (dtheta**2)
            end do
        end do

        ! --- North pole (j=1) ---
        do i = 1, Nphi
            h_shift_n(i) = h(1, mod(i-1+half, Nphi) + 1)
        end do
        do i = 1, Nphi
            ht(1,i)  = (h(2,i) - h_shift_n(i)) / (2.0d0*dtheta)
            htt(1,i) = (h(2,i) - 2.0d0*h(1,i) + h_shift_n(i)) / (dtheta**2)
        end do

        ! --- South pole (j=Ntheta) ---
        do i = 1, Nphi
            h_shift_s(i) = h(Ntheta, mod(i-1+half, Nphi) + 1)
        end do
        do i = 1, Nphi
            ht(Ntheta,i)  = (h_shift_s(i) - h(Ntheta-1,i)) / (2.0d0*dtheta)
            htt(Ntheta,i) = (h_shift_s(i) - 2.0d0*h(Ntheta,i) &
                             + h(Ntheta-1,i)) / (dtheta**2)
        end do

        ! --- Mixed derivative htp interior ---
        do i = 1, Nphi
            ip = mod(i, Nphi) + 1
            im = mod(i-2+Nphi, Nphi) + 1
            do j = 2, Ntheta-1
                htp(j,i) = (h(j+1,ip) - h(j+1,im) &
                            - h(j-1,ip) + h(j-1,im)) / (4.0d0*dtheta*dphi)
            end do
        end do

        ! --- Mixed derivative north pole ---
        do i = 1, Nphi
            ip = mod(i, Nphi) + 1
            im = mod(i-2+Nphi, Nphi) + 1
            htp(1,i) = (h(2,ip) - h(2,im) &
                        - h_shift_n(ip) + h_shift_n(im)) / (4.0d0*dtheta*dphi)
        end do

        ! --- Mixed derivative south pole ---
        do i = 1, Nphi
            ip = mod(i, Nphi) + 1
            im = mod(i-2+Nphi, Nphi) + 1
            htp(Ntheta,i) = (h_shift_s(ip) - h_shift_s(im) &
                             - h(Ntheta-1,ip) + h(Ntheta-1,im)) &
                             / (4.0d0*dtheta*dphi)
        end do

    end subroutine h_derivatives


    ! --------------------------------------------------------------------------
    ! psi_derivatives: evaluate psi and partial derivatives at h(j,i)
    ! --------------------------------------------------------------------------
    subroutine psi_derivatives(h, theta, phi, Ntheta, Nphi, dtheta, dphi, &
                                bh_idx, is_indiv, &
                                psi_v, psi_r, psi_t, psi_p)

        integer, intent(in)  :: Ntheta, Nphi, bh_idx
        logical, intent(in)  :: is_indiv
        real(8), intent(in)  :: h(Ntheta,Nphi), theta(Ntheta), phi(Nphi)
        real(8), intent(in)  :: dtheta, dphi
        real(8), intent(out) :: psi_v(Ntheta,Nphi)
        real(8), intent(out) :: psi_r(Ntheta,Nphi)
        real(8), intent(out) :: psi_t(Ntheta,Nphi)
        real(8), intent(out) :: psi_p(Ntheta,Nphi)

        real(8), parameter :: dr = 1.0d-6
        integer :: i, j

        do i = 1, Nphi
            do j = 1, Ntheta
                if (is_indiv) then
                    psi_v(j,i) = psi_bh(bh_idx, h(j,i), theta(j), phi(i))
                    psi_r(j,i) = ( psi_bh(bh_idx, h(j,i)+dr, theta(j), phi(i)) &
                                 - psi_bh(bh_idx, h(j,i)-dr, theta(j), phi(i)) ) &
                                 / (2.0d0*dr)
                    psi_t(j,i) = ( psi_bh(bh_idx, h(j,i), theta(j)+dtheta, phi(i)) &
                                 - psi_bh(bh_idx, h(j,i), theta(j)-dtheta, phi(i)) ) &
                                 / (2.0d0*dtheta)
                    psi_p(j,i) = ( psi_bh(bh_idx, h(j,i), theta(j), phi(i)+dphi) &
                                 - psi_bh(bh_idx, h(j,i), theta(j), phi(i)-dphi) ) &
                                 / (2.0d0*dphi)
                else
                    psi_v(j,i) = psi(h(j,i), theta(j), phi(i))
                    psi_r(j,i) = ( psi(h(j,i)+dr, theta(j), phi(i)) &
                                 - psi(h(j,i)-dr, theta(j), phi(i)) ) &
                                 / (2.0d0*dr)
                    psi_t(j,i) = ( psi(h(j,i), theta(j)+dtheta, phi(i)) &
                                 - psi(h(j,i), theta(j)-dtheta, phi(i)) ) &
                                 / (2.0d0*dtheta)
                    psi_p(j,i) = ( psi(h(j,i), theta(j), phi(i)+dphi) &
                                 - psi(h(j,i), theta(j), phi(i)-dphi) ) &
                                 / (2.0d0*dphi)
                end if
            end do
        end do

    end subroutine psi_derivatives


    ! --------------------------------------------------------------------------
    ! source_term: compute nonlinear RHS S(h, psi, Kij)
    ! Shibata (1997) Eq. 2.6 / Shibata (2000) Eq. 2.1
    ! --------------------------------------------------------------------------
    subroutine source_term(h, Ntheta, Nphi, dtheta, dphi, &
                            theta, phi, sin_t, cos_t, cot_t, &
                            bh_idx, is_indiv, &
                            ht, hp, htt, hpp, htp, &
                            psi_v, psi_r, psi_t, psi_p, S)

        integer, intent(in)  :: Ntheta, Nphi, bh_idx
        logical, intent(in)  :: is_indiv
        real(8), intent(in)  :: h(Ntheta,Nphi), dtheta, dphi
        real(8), intent(in)  :: theta(Ntheta), phi(Nphi)
        real(8), intent(in)  :: sin_t(Ntheta,Nphi), cos_t(Ntheta,Nphi)
        real(8), intent(in)  :: cot_t(Ntheta,Nphi)
        real(8), intent(inout) :: ht(Ntheta,Nphi), hp(Ntheta,Nphi)
        real(8), intent(inout) :: htt(Ntheta,Nphi), hpp(Ntheta,Nphi)
        real(8), intent(inout) :: htp(Ntheta,Nphi)
        real(8), intent(inout) :: psi_v(Ntheta,Nphi), psi_r(Ntheta,Nphi)
        real(8), intent(inout) :: psi_t(Ntheta,Nphi), psi_p(Ntheta,Nphi)
        real(8), intent(out) :: S(Ntheta,Nphi)

        real(8) :: sin2, C_val, s_r, s_t, s_p
        real(8) :: Kij_val(3,3), Kss, psi4
        real(8) :: gamma_rr, gamma_tt, gamma_pp, Ktrace
        real(8) :: termK, term2, term3, term4, term5, term6
        integer :: i, j

        call h_derivatives(h, Ntheta, Nphi, dtheta, dphi, &
                            ht, hp, htt, hpp, htp)
        call psi_derivatives(h, theta, phi, Ntheta, Nphi, dtheta, dphi, &
                              bh_idx, is_indiv, psi_v, psi_r, psi_t, psi_p)

        do i = 1, Nphi
            do j = 1, Ntheta

                sin2  = sin_t(j,i)**2
                C_val = sqrt(h(j,i)**2 + ht(j,i)**2 + hp(j,i)**2 / sin2)
                s_r   =  h(j,i)  / C_val
                s_t   = -ht(j,i) / C_val
                s_p   = -hp(j,i) / (C_val * sin2)

                call Kij(h(j,i), theta(j), phi(i), Kij_val)

                Kss = Kij_val(1,1)*s_r*s_r &
                    + 2.0d0*Kij_val(1,2)*s_r*s_t &
                    + 2.0d0*Kij_val(1,3)*s_r*s_p &
                    + Kij_val(2,2)*s_t*s_t &
                    + 2.0d0*Kij_val(2,3)*s_t*s_p &
                    + Kij_val(3,3)*s_p*s_p

                psi4     = psi_v(j,i)**4
                gamma_rr = 1.0d0 / psi4
                gamma_tt = 1.0d0 / (psi4 * h(j,i)**2)
                gamma_pp = 1.0d0 / (psi4 * h(j,i)**2 * sin2)
                Ktrace   = gamma_rr*Kij_val(1,1) &
                         + gamma_tt*Kij_val(2,2) &
                         + gamma_pp*Kij_val(3,3)

                termK = (psi_v(j,i)**2 * h(j,i)**2 / C_val**3) &
                      * (Kss - Ktrace)

                term2 = (4.0d0/psi_v(j,i)) &
                      * ( psi_r(j,i) &
                        - psi_t(j,i)*ht(j,i)/h(j,i)**2 &
                        - psi_p(j,i)*hp(j,i)/(h(j,i)**2 * sin2) ) &
                      * (h(j,i)**2 + ht(j,i)**2 + hp(j,i)**2/sin2)

                term3 = (3.0d0/h(j,i)) &
                      * (ht(j,i)**2 + hp(j,i)**2/sin2)

                term4 = (1.0d0/(h(j,i)**2 * sin2)) &
                      * ( 2.0d0*ht(j,i)*hp(j,i)*htp(j,i) &
                        - cot_t(j,i)*hp(j,i)**2*ht(j,i) )

                term5 = -(ht(j,i)**2 / (h(j,i)**2 * sin2)) &
                       * (sin_t(j,i)*cos_t(j,i)*ht(j,i) + hpp(j,i))

                term6 = -(hp(j,i)**2 / (h(j,i)**2 * sin2)) &
                       * (htt(j,i) + cot_t(j,i)*ht(j,i))

                S(j,i) = termK + term2 + term3 + term4 + term5 + term6

            end do
        end do

    end subroutine source_term


    ! --------------------------------------------------------------------------
    ! solve_horizon: fixed-point KSP iteration to find h(theta,phi)
    ! --------------------------------------------------------------------------
    subroutine solve_horizon(bh_idx, is_indiv, Ntheta, Nphi, omega, max_iter, &
                              h_out, theta_out, phi_out)

        integer,  intent(in)  :: bh_idx, Ntheta, Nphi, max_iter
        logical,  intent(in)  :: is_indiv
        real(8),  intent(in)  :: omega

        real(8), allocatable, intent(out) :: h_out(:,:)
        real(8), allocatable, intent(out) :: theta_out(:), phi_out(:)

        PetscErrorCode :: ierr
        Mat            :: A_loc
        Vec            :: B_loc, x_loc
        KSP            :: ksp_loc
        PC             :: pc_loc

        real(8), allocatable :: theta(:), phi(:)
        real(8), allocatable :: sin_t(:,:), cos_t(:,:), cot_t(:,:)
        real(8), allocatable :: h_old(:,:), h_new(:,:), S(:,:)
        real(8), allocatable :: ht(:,:), hp(:,:), htt(:,:), hpp(:,:), htp(:,:)
        real(8), allocatable :: psi_v(:,:), psi_r(:,:), psi_t(:,:), psi_p(:,:)
        PetscScalar, pointer :: b_arr(:)
        PetscScalar, pointer :: x_arr(:)

        real(8) :: dtheta, dphi, diff, tol
        integer :: n_tot, it, i, j

        tol    = 1.0d-8
        dtheta = acos(-1.0d0) / Ntheta
        dphi   = 2.0d0 * acos(-1.0d0) / Nphi

        allocate(theta(Ntheta), phi(Nphi))
        allocate(sin_t(Ntheta,Nphi), cos_t(Ntheta,Nphi), cot_t(Ntheta,Nphi))

        do j = 1, Ntheta
            theta(j) = (j - 0.5d0) * dtheta
        end do
        do i = 1, Nphi
            phi(i) = (i - 0.5d0) * dphi
        end do

        do i = 1, Nphi
            do j = 1, Ntheta
                sin_t(j,i) = sin(theta(j))
                cos_t(j,i) = cos(theta(j))
                cot_t(j,i) = cos_t(j,i) / sin_t(j,i)
            end do
        end do

        n_tot = Ntheta * Nphi
        allocate(h_old(Ntheta,Nphi), h_new(Ntheta,Nphi), S(Ntheta,Nphi))
        allocate(ht(Ntheta,Nphi), hp(Ntheta,Nphi), htt(Ntheta,Nphi))
        allocate(hpp(Ntheta,Nphi), htp(Ntheta,Nphi))
        allocate(psi_v(Ntheta,Nphi), psi_r(Ntheta,Nphi), psi_t(Ntheta,Nphi))
        allocate(psi_p(Ntheta,Nphi))


        do i = 1, Nphi
            do j = 1, Ntheta
                if (is_indiv) then
                    h_old(j,i) = hguess_bh(bh_idx, theta(j), phi(i))
                else
                    h_old(j,i) = hguess(theta(j), phi(i))
                end if
            end do
        end do

        call build_A(Ntheta, Nphi, dtheta, dphi, A_loc)

        call KSPCreate(PETSC_COMM_SELF, ksp_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call KSPSetOperators(ksp_loc, A_loc, A_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call KSPSetType(ksp_loc, trim(ksp_type), ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call KSPGetPC(ksp_loc, pc_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call PCSetType(pc_loc, trim(pc_type), ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call KSPSetTolerances(ksp_loc, ksp_rtol, ksp_atol, &
                              ksp_stol, ksp_max_iter, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call KSPSetFromOptions(ksp_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

        call VecCreateSeq(PETSC_COMM_SELF, n_tot, B_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
        call VecCreateSeq(PETSC_COMM_SELF, n_tot, x_loc, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

        diff = 1.0d0
        it   = 0

        do while (diff > tol .and. it < max_iter)

            call source_term(h_old, Ntheta, Nphi, dtheta, dphi, &
                             theta, phi, sin_t, cos_t, cot_t, &
                             bh_idx, is_indiv, &
                             ht, hp, htt, hpp, htp, psi_v, psi_r, psi_t, psi_p, S)

            call VecGetArray(B_loc, b_arr, ierr)
            if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
            do i = 1, Nphi
                do j = 1, Ntheta
                    b_arr((j-1)*Nphi + i) = S(j,i)
                end do
            end do
            call VecRestoreArray(B_loc, b_arr, ierr)
            if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

            call KSPSolve(ksp_loc, B_loc, x_loc, ierr)
            if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

            call check_ksp(ksp_loc, converged=diff)

            if (diff < 0.0d0) then
                call PetscPrintf(PETSC_COMM_WORLD, &
                                 "  KSP failed to converge.\n", ierr)
                deallocate(theta, phi, sin_t, cos_t, cot_t)
                deallocate(ht, hp, htt, hpp, htp)
                deallocate(psi_v, psi_r, psi_t, psi_p)
                deallocate(h_old, h_new, S)
                call MatDestroy(A_loc, ierr)
                call VecDestroy(B_loc, ierr)
                call VecDestroy(x_loc, ierr)
                call KSPDestroy(ksp_loc, ierr)
                return
            end if

            call VecGetArrayRead(x_loc, x_arr, ierr)
            if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if
            do i = 1, Nphi
                do j = 1, Ntheta
                    h_new(j,i) = x_arr((j-1)*Nphi + i)
                end do
            end do
            call VecRestoreArrayRead(x_loc, x_arr, ierr)
            if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

            diff = 0.0d0
            do i = 1, Nphi
                do j = 1, Ntheta
                    diff = diff + (h_new(j,i) - h_old(j,i))**2
                end do
            end do
            diff = sqrt(diff)

            write(*,'("  Iter ",i5,"   diff = ",es12.4)') it, diff

            if (.not. ieee_is_finite(diff) .or. diff > 1.0d6) then
                call PetscPrintf(PETSC_COMM_WORLD, &
                                 "  Solver diverged - horizon not found.\n", ierr)
                deallocate(theta, phi, sin_t, cos_t, cot_t)
                deallocate(ht, hp, htt, hpp, htp)
                deallocate(psi_v, psi_r, psi_t, psi_p)
                deallocate(h_old, h_new, S)
                call MatDestroy(A_loc, ierr)
                call VecDestroy(B_loc, ierr)
                call VecDestroy(x_loc, ierr)
                call KSPDestroy(ksp_loc, ierr)
                return
            end if

            h_old = (1.0d0 - omega) * h_old + omega * h_new
            it = it + 1

        end do

        call PetscPrintf(PETSC_COMM_WORLD, "  Solver finished.\n", ierr)

        allocate(h_out(Ntheta,Nphi))
        allocate(theta_out(Ntheta), phi_out(Nphi))
        h_out     = h_new
        theta_out = theta
        phi_out   = phi

        deallocate(theta, phi, sin_t, cos_t, cot_t)
        deallocate(ht, hp, htt, hpp, htp)
        deallocate(psi_v, psi_r, psi_t, psi_p)
        deallocate(h_old, h_new, S)
        call MatDestroy(A_loc, ierr)
        call VecDestroy(B_loc, ierr)
        call VecDestroy(x_loc, ierr)
        call KSPDestroy(ksp_loc, ierr)

    end subroutine solve_horizon


    ! --------------------------------------------------------------------------
    ! check_ksp: returns -1.0 if KSP diverged, 0.0 if converged
    ! --------------------------------------------------------------------------
    subroutine check_ksp(ksp_loc, converged)

        KSP,     intent(in)  :: ksp_loc
        real(8), intent(out) :: converged

        PetscErrorCode      :: ierr
        KSPConvergedReason  :: reason

        call KSPGetConvergedReason(ksp_loc, reason, ierr)
        if (ierr /= 0) then; write(*,*) "PETSc error: ", ierr; stop; end if

        if (reason%v < 0) then
            write(*,'("  KSP diverged, reason code = ",i0)') reason%v
            converged = -1.0d0
        else
            converged = 0.0d0
        end if

    end subroutine check_ksp


    ! --------------------------------------------------------------------------
    ! save_hdf5: write h, theta, phi to HDF5 file
    ! --------------------------------------------------------------------------
    subroutine save_hdf5(filename, h, theta, phi)

        character(len=*), intent(in) :: filename
        real(8),          intent(in) :: h(:,:), theta(:), phi(:)

        integer(HID_T)   :: file_id, dset_id, dspace_id
        integer(HSIZE_T) :: dims2(2), dims1(1)
        integer          :: hdferr, Ntheta, Nphi

        Ntheta = size(h, 1)
        Nphi   = size(h, 2)

        call h5open_f(hdferr)
        call h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, hdferr)

        ! Write h
        dims2 = [Ntheta, Nphi]
        call h5screate_simple_f(2, dims2, dspace_id, hdferr)
        call h5dcreate_f(file_id, "h", H5T_NATIVE_DOUBLE, &
                         dspace_id, dset_id, hdferr)
        call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, h, dims2, hdferr)
        call h5dclose_f(dset_id, hdferr)
        call h5sclose_f(dspace_id, hdferr)

        ! Write theta
        dims1 = [Ntheta]
        call h5screate_simple_f(1, dims1, dspace_id, hdferr)
        call h5dcreate_f(file_id, "theta", H5T_NATIVE_DOUBLE, &
                         dspace_id, dset_id, hdferr)
        call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, theta, dims1, hdferr)
        call h5dclose_f(dset_id, hdferr)
        call h5sclose_f(dspace_id, hdferr)

        ! Write phi
        dims1 = [Nphi]
        call h5screate_simple_f(1, dims1, dspace_id, hdferr)
        call h5dcreate_f(file_id, "phi", H5T_NATIVE_DOUBLE, &
                         dspace_id, dset_id, hdferr)
        call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, phi, dims1, hdferr)
        call h5dclose_f(dset_id, hdferr)
        call h5sclose_f(dspace_id, hdferr)

        call h5fclose_f(file_id, hdferr)
        call h5close_f(hdferr)

    end subroutine save_hdf5

end module petsc_solver
