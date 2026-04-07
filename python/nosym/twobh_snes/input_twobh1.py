"""

Type: TWO BLACK HOLES in BRILL-LINDQUIST 

Note: Horizon finder fails to converge at z0 >=0.767, because
      it couldn't find the common horizon anymore. 

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
python ../../src/nosym_snes input_twobh.py

"""

import numpy as np

# -- Physical system --
system_name = "Two Black Holes"
coord_sys   = "Brill-Lindquist"
number_bh   = 2

# -- Find individual BHs? --
find_indiv  = False

# ----- Parameters ----
M1  = 1.0
M2  = 1.0
sep = 0.2                   # separation factor (critical = 0.767)
z0  = sep*(M1 + M2)         # distance from origin

#%%
# Spatial metric gamma_ij in spherical coordinates (no phi dependance)
def gammaij(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    r1 = np.sqrt(x**2 + y**2 + (z + z0)**2)
    r2 = np.sqrt(x**2 + y**2 + (z - z0)**2)
    
    psi = 1 + M1/(2*r1) + M2/(2*r2)
    return np.array([
        [psi**4, np.zeros_like(psi), np.zeros_like(psi)],
        [np.zeros_like(psi), psi**4 * r**2, np.zeros_like(psi)],
        [np.zeros_like(psi), np.zeros_like(psi), psi**4 * r**2 * np.sin(theta)**2]
    ])

# gamma_ij for BH1 (for individual horizon, centered at z = -z0)
def gammaij1(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z_global = -z0 + r * np.cos(theta)
    
    r1 = np.sqrt(x**2 + y**2 + (z_global + z0)**2)
    r2 = np.sqrt(x**2 + y**2 + (z_global - z0)**2)
    
    psi = 1.0 + M1/(2*r1) + M2/(2*r2)
    return np.array([
        [psi**4, np.zeros_like(psi), np.zeros_like(psi)],
        [np.zeros_like(psi), psi**4 * r**2, np.zeros_like(psi)],
        [np.zeros_like(psi), np.zeros_like(psi), psi**4 * r**2 * np.sin(theta)**2]
    ])

# gamma_ij for BH2 (for individual horizon, centered at z = +z0)
def gammaij2(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z_global = +z0 + r * np.cos(theta)
    
    r1 = np.sqrt(x**2 + y**2 + (z_global + z0)**2)
    r2 = np.sqrt(x**2 + y**2 + (z_global - z0)**2)
    
    psi = 1.0 + M1/(2*r1) + M2/(2*r2)
    return np.array([
        [psi**4, np.zeros_like(psi), np.zeros_like(psi)],
        [np.zeros_like(psi), psi**4 * r**2, np.zeros_like(psi)],
        [np.zeros_like(psi), np.zeros_like(psi), psi**4 * r**2 * np.sin(theta)**2]
    ])

# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta=None, phi=None):
    return np.zeros((3,3))

# Individual guess shapes
def hguess1(theta, phi):
    return 0.5*M1

def hguess2(theta, phi):
    return 0.5*M2

# Initial guess for horizon shape r = h(theta, phi), slow-varying! 
# This form is robust, a constant radius initial guess isn't good.
def hguess(theta, phi):
    h0 = 1.0   # ~ m = M/2 = 1.0
    h1 = 0.3   # small prolate deformation along z
    return h0 - h1 * np.cos(theta)



# The solver will loop over this if find_indiv = True
blackholes = [
    {"name": "BH1", "mass": M1, "center": [0, 0, -z0], "gammaij": gammaij1, "hguess": hguess1},
    {"name": "BH2", "mass": M2, "center": [0, 0, +z0], "gammaij": gammaij2, "hguess": hguess2}]


#%%
# Solver options

# Number of theta grid points
Ntheta_indiv  = 32
Ntheta_common = 32

# Number of phi grid points
Nphi_indiv    = 32
Nphi_common   = 32

# --- Solver type ---
snes_type       = "newtonls"

# --- Tolerances ---
snes_rtol       = 1e-8         # relative residual tolerance
snes_atol       = 1e-8         # absolute residual tolerance
snes_stol       = 1e-14        # stagnation tolerance
snes_max_it     = 100          # maximum number of SNES iterations

# --- Line search options ---
snes_linesearch_type    = "bt"      # backtracking line search
snes_linesearch_damping = 0.1       # starting fraction of step
snes_linesearch_monitor = True      # prints info for each line search step

# --- Misc ---
snes_monitor    = True         # Prints the iterations.
snes_mf         = True

# Print every iteration? 
save_iterations = False

# Output file
output_dir = "./data"
