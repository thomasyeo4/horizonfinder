#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 19:04:43 2025

@author: thomasyeo
"""


"""
This is the input file. Don't change anything that is compulsory
to run the code!!!


Run the code with this line in bash:
    
mpirun -n 1 python HorizonFinder.py Input.py

"""
import numpy as np

# --- Symmetry type ---
# Options: "spheresym", "axisym", "nosym"
symmetry = "spheresym"

# Conformal factor for Schwarzschild in isotropic coordinates
def psi(r):
    M = 1.0
    return 1 + M/(2*r)

# For axisymmetry / no symmetry:
# gammaij = 3x3 matrix function returning spatial metric at given coordinates
# def gammaij(x, y, z):
#     return np.eye(3)  # placeholder

# Extrinsic curvature, here zero
def Kij(r):
    return np.zeros((3,3))

# Solver options
rguess = 3
snes_type = "newtonls"       # PETSc SNES type
use_multigrid = False         # Multigrid options