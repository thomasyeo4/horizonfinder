"""
Type: SCHWARSCHILD in ISOTROPIC

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
python ../../src/source_v3_1.py input_schwarzschild.py

Note: source_v3_1.py is slow but more robust, source_v3_2.py is faster but less robust.
"""

import numpy as np

# -- Physical system --
system_name = "Schwarzschild"
coord_sys   = "Isotropic"

# --- Symmetry type ---
symmetry = "nosym"

# -- Find individual BHs? --
find_indiv  = False

# --- Parameters ---
M = 1.0

# Spatial metric gamma_ij in spherical coordinates (no phi dependance)
def gammaij(r, theta, phi):
    psi = 1 + M/(2*r)
    return np.array([
        [psi**4, 0, 0],
        [0, psi**4 * r**2, 0],
        [0, 0, psi**4 * r**2 * np.sin(theta)**2]
    ])

# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta, phi):
    return np.zeros((3,3))

# Initial guess for horizon shape r = h(theta), slow-varying! 
# DON'T CHANGE !!!!
def hguess(theta, phi):
    return 0.6

# Number of theta grid points (Even numbers only!)
# DON'T CHANGE !!!!
Ntheta_common = 50
Nphi_common   = 32

# Solver options
# --- Solver type ---
snes_type       =  "newtontr" # This is faster than "newtonls" when using source_NewtonSNES

# --- Tolerances ---
snes_rtol       = 1e-6         # relative residual tolerance
snes_atol       = 1e-6         # absolute residual tolerance
snes_stol       = 1e-14        # stagnation tolerance
snes_max_it     = 200          # maximum number of SNES iterations

# --- Line search options ---
snes_linesearch_type    = "basic"   # backtracking line search
snes_linesearch_maxstep = 1.0       # allow full Newton step
snes_linesearch_damping = 1.0       # starting fraction of step
snes_linesearch_monitor = True      # prints info for each line search step

# --- Misc ---
snes_monitor    = True         # Prints the iterations.
snes_mf         = False

# Print every iteration? 
save_iterations = False

# Output file
output_file = "./data.npz"
