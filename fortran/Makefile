# ==============================================================================
# Makefile for nosym_ksp Fortran horizon finder
# Uses PETSc's own make infrastructure for correct flags
# ==============================================================================

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# --- Compiler and flags ---
FC       = mpif90
BUILD    ?= release
FFLAGS_DEBUG   = -O2 -Wall -fcheck=all -ffree-line-length-none -Wno-unused-dummy-argument -J bin/
FFLAGS_RELEASE = -O3 -ffast-math -Wall -ffree-line-length-none -Wno-unused-dummy-argument -J bin/

ifeq ($(BUILD),debug)
FFLAGS = $(FFLAGS_DEBUG)
else
FFLAGS = $(FFLAGS_RELEASE)
endif

# --- HDF5 (OpenMPI flavour, matches PETSc build) ---
HDF5_DIR     = /usr/lib/x86_64-linux-gnu/hdf5/openmpi
HDF5_INC     = -I/usr/include/hdf5/openmpi
HDF5_LIBS    = -L$(HDF5_DIR) -lhdf5_fortran -lhdf5hl_fortran -lhdf5 -lz

# --- PETSc flags (provided by the included variables above) ---
PETSC_INC 	 = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
PETSC_LIBS 	 = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

# --- All includes ---
INCLUDES     = $(HDF5_INC) $(PETSC_INC)

# ==============================================================================
# Targets
# Compilation order matters:
#   1. input_*.F90      (defines module user_input)
#   2. petsc_solver.F90 (defines module petsc_solver, uses user_input)
#   3. petsc_ksp.F90    (program driver, uses both)
# ==============================================================================

# --- Schwarzschild ---
schw: bin src/input_schw.F90 src/petsc_solver.F90 src/petsc_ksp.F90
	$(FC) $(FFLAGS) $(INCLUDES) -o bin/schw \
	    src/input_schw.F90 \
	    src/petsc_solver.F90 \
	    src/petsc_ksp.F90 \
	    $(PETSC_LIBS) $(HDF5_LIBS)
	@echo "Built: bin/schw"

# --- Two Black Holes ---
twobh: bin src/input_twobh.F90 src/petsc_solver.F90 src/petsc_ksp.F90
	$(FC) $(FFLAGS) $(INCLUDES) -o bin/twobh \
	    src/input_twobh.F90 \
	    src/petsc_solver.F90 \
	    src/petsc_ksp.F90 \
	    $(PETSC_LIBS) $(HDF5_LIBS)
	@echo "Built: bin/twobh"

# --- All cases ---
all: schw twobh

# --- Create bin dir if missing ---
bin:
	mkdir -p bin

# --- Clean (:: matches PETSc rules file which also uses ::) ---
clean::
	rm -f bin/* *.o *.mod

.PHONY: all schw twobh bin
