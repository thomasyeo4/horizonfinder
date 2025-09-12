#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:51:34 2025

@author: thomasyeo
"""

import importlib.util
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np


#%%

class spheresym:
    PETSc.Sys.Print("Spherical Symmetry spacetime")
    #------ Only provide psi (conformal factor) and Kij (extrinsic curvature)
    #------ and the code will do the rest!
    
    def __init__(self, psi, Kij):
        self.psi = psi
        self.Kij = Kij
        
        #---------- Christoffel Symbols for spherical symmetric spacetimes ----------
    def Christoffels(self, r, theta=np.pi/2, dr=1e-6):
        psi = self.psi(r)
        psifront = self.psi(r+dr)
        psiback = self.psi(r-dr)
        psiprime = (psifront - psiback)/(2*dr)
        
        Gamma = np.zeros((3, 3, 3))
        
        Gamma[0, 0, 0] = 2*(psiprime/psi)
        Gamma[0, 1, 1] = -r - 2*(r**2)*(psiprime/psi)
        Gamma[0, 2, 2] = -r*(np.sin(theta))**2 - 2*r**2*(np.sin(theta))**2*(psiprime/psi)
        Gamma[1, 1, 0] = (1/r) + 2*(psiprime/psi)
        Gamma[1, 0, 1] = Gamma[1, 1, 0]
        Gamma[1, 2, 2] = -np.sin(theta)*np.cos(theta)
        Gamma[2, 0, 2] = (1/r) + 2*(psiprime/psi)
        Gamma[2, 2, 0] = Gamma[2, 0, 2]
        Gamma[2, 2, 1] = 1/np.tan(theta)
        Gamma[2, 1, 2] = Gamma[2, 2, 1]
        
        return Gamma
    
        #----------- The Horizon function (Theta) for Spherical Symmetric case -------
    def Horizon(self, r):
        Gamma = self.Christoffels(r)
        Kij = self.Kij(r)
        psi = self.psi(r)
        
        sup = np.array([1/psi**2, 0, 0])
        sdown = np.array([psi**2, 0, 0])
        invgamma = np.diag([1/psi**4, 1/(psi**4)*r**2, 1/((psi**4)*(r**2)*(np.sin(np.pi/2))**2)])
        m = invgamma - np.outer(sup, sup)
        
        value = 0
        for i in range(3):
            for j in range(3):
                bracket = 0
                for k in range(3):
                    bracket += sdown[k]*Gamma[k, i, j]
                bracket += Kij[i, j]
                value += m[i, j]*bracket
        
        return value
        
        #---------- The SNES solver by PETSc -------
    def Solver(self, rguess):
        
        def F(snes, x, f):
            rval = x[0]
            f[0] = self.Horizon(rval)
            
        r = PETSc.Vec().createSeq(1)
        r.setValue(0, rguess)
        r.assemble()
        
        snes = PETSc.SNES().create()
        snes.setFunction(F)
        snes.setFromOptions()
        snes.solve(None, r)
            
        return r.getArray()[0]
    




#%%

#class axisym:

    


#%%

#class nosym:
    
    

#%%

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: mpirun -n 1 python HorizonFinder.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    spec = importlib.util.spec_from_file_location("user_input", input_file)
    user_input = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_input)
    
    # ---- Symmetry type ----
    symmetry = getattr(user_input, "symmetry", None)
    if symmetry is None:
        raise ValueError("Input file must define symmetry type!")
    
    # ---- Mandatory input ----
    Kij = getattr(user_input, "Kij", None)
    if Kij is None:
        raise ValueError("Kij missing in input file.")
    
    # ---- Symmetry type ----
    if symmetry == "spheresym":
        psi = getattr(user_input, "psi", None)
        if psi is None: 
            raise ValueError("psi missing in input file.")
        solver = spheresym(psi, Kij)
    elif symmetry in ["axisym", "nosym"]:
        gammaij = getattr(user_input, "gammaij", None)
        if gammaij is None:
            raise ValueError("gammaij missing in input file.")
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")
    
    
    # --- Load solver options from input ---
    rguess = getattr(user_input, "rguess", None)
    snes_type = getattr(user_input, "snes_type", None)
    use_multigrid = getattr(user_input, "use_multigrid", None)
    
    
    # --- Set PETSc options ---
    opts = PETSc.Options()
    opts["snes_type"] = snes_type
    if use_multigrid:
        opts["ksp_type"] = "gmres"
        opts["pc_type"] = "mg"
        # Future multigrid-specific options can also be added here

    # --- Solve ---
    rhorizon = solver.Solver(rguess=rguess)
    PETSc.Sys.Print("Horizon radius:", rhorizon)