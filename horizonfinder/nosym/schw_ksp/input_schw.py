"""
Type: SCHWARSCHILD in ISOTROPIC

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
python ../../src/nosym_ksp.py input_schw.py

Note: nosym_snes.py is slow but more robust, nosym_ksp.py is faster but less robust.
"""

import numpy as np

# -- Physical system --
system_name = "Schwarzschild"
coord_sys   = "Isotropic"

# -- Find individual BHs? --
find_indiv  = False

# --- Parameters ---
M = 1.0

# Conformal factor for Schwarzschild in isotropic coordinates
def psi(r, theta, phi):
    return 1 + M/(2*r)

# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta, phi):
    return np.zeros((3,3))

# Initial guess for horizon shape r = h(theta), slow-varying! 
# DON'T CHANGE !!!!
def hguess(theta, phi):
    return 0.6 + 0.01*np.sin(theta)*np.cos(phi)
    #return 0.6

# Number of theta grid points (Even numbers only!)
# DON'T CHANGE !!!!
Ntheta_common = 100
Nphi_common   = 100

# Solver options
# --- Solver type ---
ksp_type       =  "gmres" # This is faster than "newtonls" when using source_NewtonSNES
pc_type        =  "ilu" # Preconditioner type for KSP solver

# --- Tolerances ---
ksp_rtol       = 1e-6         # relative residual tolerance
ksp_atol       = 1e-6         # absolute residual tolerance
ksp_stol       = 1e-14        # stagnation tolerance

# --- Misc ---
max_iter                = 500         # Maximum number of iterations
omega                   = 0.3         # Relaxation parameter for KSP solver (if applicable)   
ksp_monitor             = None        # Prints the iterations.
ksp_converged_reason    = None

# Print every iteration? 
save_iterations = False

# Output file
output_dir = "./data"
