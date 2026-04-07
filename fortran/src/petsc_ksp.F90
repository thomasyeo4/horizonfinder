#include "petsc/finclude/petsc.h"

program petsc_ksp
    
    use petsc
    use user_input      ! provided by input_schw.F90 or input_twobh.F90
    use petsc_solver

    implicit none
    
    ! Local variables
    PetscErrorCode       :: ierr
    real(8), allocatable :: h_sol(:,:), theta_grid(:), phi_grid(:)
    real(8)              :: t_start, t_end
    integer              :: bh_idx 
    character(len=256)   :: outfile


    ! Initialise PETSc
    call PetscInitialize(PETSC_NULL_CHARACTER, ierr)

    call PetscPrintf(PETSC_COMM_WORLD, &
        "====================================================\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, &
        "Apparent Horizon Finder - nosym_ksp (Fortran/PETSc)\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, &
        "====================================================\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, "System     : "//trim(system_name)//"\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, "Coord sys  : "//trim(coord_sys)//"\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, &
        "====================================================\n", ierr)

    ! Ensure output directory exists (best-effort, no fatal error)
    call system("mkdir -p "//trim(output_dir))


    ! Individual horizons (if requested)
    if (find_indiv) then
        call PetscPrintf(PETSC_COMM_WORLD, "Finding individual horizons...\n", ierr)

        do bh_idx = 1, n_blackholes
            call PetscPrintf(PETSC_COMM_WORLD, &
                "  Solving for "//trim(bh_names(bh_idx))//"\n", ierr)

            call cpu_time(t_start)
            call solve_horizon( &
                bh_idx, .true.,           &
                Ntheta_indiv, Nphi_indiv, &
                omega, max_iter,          &
                h_sol, theta_grid, phi_grid)
            call cpu_time(t_end)

            if (.not. allocated(h_sol)) then
                call PetscPrintf(PETSC_COMM_WORLD, &
                "  WARNING: horizon not found for "//trim(bh_names(bh_idx))//"\n", ierr)
                cycle
            end if

            write(outfile,'(a,"/",a,"_horizon.h5")') trim(output_dir), trim(bh_names(bh_idx))
            call save_hdf5(outfile, h_sol, theta_grid, phi_grid)

            call PetscPrintf(PETSC_COMM_WORLD, "  Saved -> "//trim(outfile)//"\n", ierr)
            write(*,'("  CPU time: ",f10.4," s")') t_end - t_start

            deallocate(h_sol, theta_grid, phi_grid)
        end do

    else
        call PetscPrintf(PETSC_COMM_WORLD, &
        "Skipping individual horizons (find_indiv = .false.)\n", ierr)
    end if

    ! Common / origin-centred horizon
    call PetscPrintf(PETSC_COMM_WORLD, &
        "====================================================\n", ierr)
    call PetscPrintf(PETSC_COMM_WORLD, "Finding common horizon...\n", ierr)

    call cpu_time(t_start)
    call solve_horizon( &
        0, .false.,                  &
        Ntheta_common, Nphi_common,  &
        omega, max_iter,             &
        h_sol, theta_grid, phi_grid)
    call cpu_time(t_end)

    if (.not. allocated(h_sol)) then
        call PetscPrintf(PETSC_COMM_WORLD, &
        "WARNING: common horizon not found.\n", ierr)
    else
        write(outfile,'(a,"/horizon_common.h5")') trim(output_dir)
        call save_hdf5(outfile, h_sol, theta_grid, phi_grid)
        call PetscPrintf(PETSC_COMM_WORLD, "Saved -> "//trim(outfile)//"\n", ierr)
        write(*,'("CPU time: ",f10.4," s")') t_end - t_start
        deallocate(h_sol, theta_grid, phi_grid)
    end if

    call PetscPrintf(PETSC_COMM_WORLD, &
        "====================================================\n", ierr)

    call PetscFinalize(ierr)

end program petsc_ksp
