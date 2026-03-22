"""

Type: TWO BLACK HOLES in BRILL-LINDQUIST 

Note: Horizon finder fails to converge at z0 >=0.767, because
      it couldn't find the common horizon anymore. 

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
python ../../src/nosym_ksp.py input_twobh.py

"""

import numpy as np

# -- Physical system --
system_name = "Two Black Holes"
coord_sys   = "Brill-Lindquist"
number_bh   = 2

# -- Find individual BHs? --
find_indiv  = True

# ----- Parameters ----
M1  = 1.0
M2  = 1.0
z0  = 0.5       # distance from origin

#%%
# Spatial psi_ij in spherical coordinates
def psi(r, theta, phi):
    epsilon = 1e-12
    r = np.maximum(r, epsilon)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    r1 = np.sqrt(x**2 + y**2 + (z + z0)**2)
    r2 = np.sqrt(x**2 + y**2 + (z - z0)**2)

    r1 = np.maximum(r1, epsilon)
    r2 = np.maximum(r2, epsilon)

    return 1 + M1/(2*r1) + M2/(2*r2)


# psi_ij for BH1 (for individual horizon, centered at z = -z0)
def psi1(r, theta, phi):
    epsilon = 1e-12
    r = np.maximum(r, epsilon)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z_global = -z0 + r * np.cos(theta)

    r1 = np.sqrt(x**2 + y**2 + (z_global + z0)**2)
    r1 = np.maximum(r1, epsilon)
    r2 = np.sqrt(x**2 + y**2 + (z_global - z0)**2)
    r2 = np.maximum(r2, epsilon)
    
    return 1 + M1/(2*r1) + M2/(2*r2)


# psi_ij for BH2 (for individual horizon, centered at z = +z0)
def psi2(r, theta, phi):
    epsilon = 1e-12
    r = np.maximum(r, epsilon)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z_global = z0 + r * np.cos(theta)

    r1 = np.sqrt(x**2 + y**2 + (z_global + z0)**2)
    r1 = np.maximum(r1, epsilon)
    r2 = np.sqrt(x**2 + y**2 + (z_global - z0)**2)
    r2 = np.maximum(r2, epsilon)

    return 1 + M2/(2*r2) + M1/(2*r1)


# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta=None, phi=None):
    return np.zeros((3,3))

# Initial guess for horizon shape r = h(theta, phi) for individual horizons, slow-varying!
def hguess1(theta, phi):
    R = 0.5 * M1
    return R * np.ones_like(theta)

def hguess2(theta, phi):
    R = 0.5 * M2
    return R * np.ones_like(theta)

# Initial guess for horizon shape r = h(theta, phi), slow-varying! 
# This form is robust, a constant radius initial guess isn't good.
def hguess(theta, phi):
    return 2.0 * np.ones_like(theta)


# The solver will loop over this if find_indiv = True
blackholes = [{"name": "BH1", "psi": psi1, "hguess": hguess1},{"name": "BH2", "psi": psi2, "hguess": hguess2}]

#%%
# Solver options
Ntheta_indiv = 64
Nphi_indiv   = 64

Ntheta_common = 64
Nphi_common   = 64

# --- Solver type ---
ksp_type = "gmres"
pc_type  = "ilu"

ksp_rtol = 1e-6
ksp_atol = 1e-6
ksp_stol = 1e-14

max_iter = 5000
omega    = 0.3

# Print every iteration? 
save_iterations = False

# Output file
output_dir = "./data"
