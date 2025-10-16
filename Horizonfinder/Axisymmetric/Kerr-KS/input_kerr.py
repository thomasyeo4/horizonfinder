"""

Type: KERR in KERR-SCHILD

==================================================================
This is the input file. Don't change anything that is compulsory
to run the code!!!
==================================================================
Run the code with this line in bash:
mpirun -n 1 python ../../HorizonFinder.py input_kerr.py

"""

import numpy as np

# -- Physical system --
system_name = "Kerr"
coord_sys   = "Kerr-Schild"

# --- Symmetry type ---
symmetry = "axisym"

# --- Parameters ---
a = 0.95    # Spin parameter 0 =< a < 1
M = 1.0     # Mass M
phi = 0     # phi


# Jacobian 
def jacobian(r, theta, phi, a):
    R = r**2 + a**2
    
    # Jacobian matrix
    J = np.zeros((3, 3))
    J[0, 0] = (r*np.sin(theta)*np.cos(phi))/np.sqrt(R)
    J[0, 1] = np.sqrt(R)*np.cos(theta)*np.cos(phi)
    J[0, 2] = -np.sqrt(R)*np.sin(theta)*np.sin(phi)
    J[1, 0] = (r*np.sin(theta)*np.sin(phi))/np.sqrt(R)
    J[1, 1] = np.sqrt(R)*np.cos(theta)*np.sin(phi)
    J[1, 2] = np.sqrt(R)*np.sin(theta)*np.cos(phi)
    J[2, 0] = np.cos(theta)
    J[2, 1] = -r*np.sin(theta)
    J[2, 2] = 0
    
    return J


# l^i = l_i vectors in spheroidal Kerr-Schild
def lvec(r, theta, phi, a):
    R = r**2 + a**2
    
    # Relationship between Cartesian and Spheroidal
    x = np.sqrt(R)*np.sin(theta)*np.cos(phi)
    y = np.sqrt(R)*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    # l^i vectors
    lx = (r*x + a*y)/R
    ly = (r*y - a*x)/R
    lz = z/r
    
    return lx, ly, lz


# H function 
def h(theta, r, M, a):
    return M*r/(r**2 + a**2*np.cos(theta)**2)


#%%
# Spatial metric Cartesian Kerr-Schild coordinates
def gammac(r, theta, phi, a, M):
    # l^i vectors
    lx, ly, lz = lvec(r, theta, phi, a)
    
    # H function
    H = h(theta, r, M, a)
    
    # Cartesian gamma_ij
    gamma_cart = np.zeros((3, 3))
    gamma_cart[0, 0] = 2*H*lx**2 + 1
    gamma_cart[0, 1] = 2*H*lx*ly 
    gamma_cart[1, 0] = gamma_cart[0, 1]
    gamma_cart[0, 2] = 2*H*lx*lz
    gamma_cart[2, 0] = gamma_cart[0, 2]
    gamma_cart[1, 1] = 2*H*ly**2 + 1
    gamma_cart[1, 2] = 2*H*ly*lz
    gamma_cart[2, 1] = gamma_cart[1, 2]
    gamma_cart[2, 2] = 2*H*lz**2 + 1
    
    return gamma_cart


# Spatial Metric in Spheroidal Kerr-Schild coordinates
def gammaij(r, theta):
    # Cartesian gamma_ij
    gamma_cart = gammac(r, theta, phi, a, M)
    
    # Jacobian
    J = jacobian(r, theta, phi, a)
    
    # Spheroidal gamma_ij
    gamma = J.T @ gamma_cart @ J
    
    return gamma


#%%
# Extrinsic curvature in Cartesian 
def Kijc(r, theta, phi, a, M):
    
    # l^i vectors
    lx, ly, lz = lvec(r, theta, phi, a)

    # H function
    H = h(theta, r, M, a)
    
    # Other stuff
    z = r*np.cos(theta)
    R = r**2 + a**2
    alpha = np.sqrt(1/(1 + 2*H))
    
    # Cartesian K_ij 
    K_cart = np.zeros((3, 3))
    K_cart[0, 0] = ((2*alpha*r*H)/R + 2*alpha*H**2*(2*r*lx**2/R + 2*a*lx*ly/R)
                    - (4*a**2*z**2*alpha*H**3*lx**2)/(M*r**3))
    K_cart[0, 1] = (2*alpha*H**2*((a/R)*(ly**2 - lx**2) + 2*r*lx*ly/R) 
                    - (4*a**2*z**2*alpha*H**3*lx*ly)/(M*r**3))
    K_cart[1, 0] = K_cart[0, 1]
    K_cart[0, 2] = (-(2*alpha*M*a**2*r**3*z*lx)/((r**4 + a**2*z**2)**2)
                    + 2*alpha*H**2*((r/R + 1/r)*lx*lz + (a/R)*ly*lz) 
                    - (4*a**2*z**2*alpha*H**3)/(M*r**3)*lx*lz)
    K_cart[2, 0] = K_cart[0, 2]
    K_cart[1, 1] = ((2*alpha*r*H)/R + 2*alpha*H**2*(-2*a*lx*ly/R + 2*r*ly**2/R) 
                    - (4*a**2*z**2*alpha*H**3)/(M*r**3)*ly**2)
    K_cart[1, 2] = (-(2*alpha*M*a**2*r**3*z)/((r**4 + a**2*z**2)**2)*ly 
                    + 2*alpha*H**2*((r/R + 1/r)*ly*lz - (a/R)*lx*lz) 
                    - (4*a**2*z**2*alpha*H**3)/(M*r**3)*ly*lz)
    K_cart[2, 1] = K_cart[1, 2]
    K_cart[2, 2] = (2*alpha*H/r 
                    - (4*alpha*M*a**2*r**3*z)/((r**4 + a**2*z**2)**2)*lz 
                    + (4*alpha*H**2*lz**2)/2
                    - (4*a**2*z**2*alpha*H**3)/(M*r**3)*lz**2)
    
    return K_cart


# Extrinsic curvature in Spheroidal Kerr-Schild coordinates
def Kij(r, theta):
    # Cartesian gamma_ij
    Kc = Kijc(r, theta, phi, a, M)
    
    # Jacobian
    J = jacobian(r, theta, phi, a)
    
    # Spheroidal gamma_ij
    K = J.T @ Kc @ J
    
    return K



#%%
# Initial guess for horizon shape r = h(theta), slow-varying! 
def hguess(theta):
    # base radius ~ combined mass scale
    R0 = M + np.sqrt(M*M - a*a)
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

