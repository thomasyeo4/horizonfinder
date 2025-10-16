"""

Type: TWO BLACK HOLES in BRILL-LINDQUIST 

Note: Horizon finder fails to converge at z0 >=0.767, because
      it couldn't find the common horizon anymore. 

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
mpirun -n 1 python ../../HorizonFinder.py input_TwoBH.py

"""

import numpy as np

# -- Physical system --
system_name = "Two Black Holes"
coord_sys   = "Brill-Lindquist"

# --- Symmetry type ---
symmetry = "axisym"

# ----- Parameters ----
M1  = 1.0
M2  = 1.0
sep = 0.767                 # separation factor (critical = 0.767)
z0  = sep*(M1 + M2)         # distance from origin
phi = 0

#%%
# Spatial metric gamma_ij in spherical coordinates (no phi dependance)
def gammaij(r, theta):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    r1 = np.sqrt(x**2 + y**2 + (z + z0)**2)
    r2 = np.sqrt(x**2 + y**2 + (z - z0)**2)
    psi = 1 + M1/(2*r1) + M2/(2*r2)
    return np.array([
        [psi**4, 0, 0],
        [0, psi**4 * r**2, 0],
        [0, 0, psi**4 * r**2 * np.sin(theta)**2]
    ])

# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta=None, phi=None):
    return np.zeros((3,3))

# Initial guess for horizon shape r = h(theta), slow-varying! 
# Thsi form is robust, a constant radius initial guess isn't good.
def hguess(theta):
    # base radius ~ combined mass scale
    R0 = 1.5 * (M1 + M2)
    # small oblateness (helps convergence)
    eps = 0.2
    # deformation term (Legendre P2)
    P2 = 0.5 * (3*np.cos(theta)**2 - 1)
    # final shape
    r = R0 * (1 + eps * P2)
    # avoid touching punctures
    r = np.clip(r, 0.5, None)
    return r

#%%
# Number of theta grid points
Ntheta = 100

# Solver options
snes_type = "newtonls"
snes_monitor = True     # Prints the iterations.
snes_mf = True
use_multigrid = False

# Output file
output_file = "./data.csv"
