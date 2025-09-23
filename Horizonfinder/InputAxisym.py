#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:34:47 2025

@author: thomasyeo
"""

"""
This is the input file. Don't change anything that is compulsory
to run the code!!!


Run the code with this line in bash:
    
mpirun -n 1 python HorizonFinder.py InputAxisym.py

"""

import numpy as np

# --- Symmetry type ---
symmetry = "axisym"

# Spatial metric gamma_ij in spherical coordinates (no phi dependance)
def gammaij(r, theta):
    M = 1.0
    psi = 1 + M/(2*r)
    return np.array([
        [psi**4, 0, 0],
        [0, psi**4 * r**2, 0],
        [0, 0, psi**4 * r**2 * np.sin(theta)**2]
    ])

# Extrinsic curvature (zero in time-symmetric slice)
def Kij(r, theta=None, phi=None):
    return np.zeros((3,3))

# Initial guess for horizon shape r = h(theta)
def hguess(theta):
    return 0.5*(1 + 0.3*np.sin(4*theta))

# Number of theta grid points
Ntheta = 64

# Solver options
snes_type = "newtonls"
snes_mf = True
use_multigrid = False

# Output file
output_file = "../Data/AxisymHorizon.csv"
