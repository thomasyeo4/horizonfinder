#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:58:31 2025

@author: thomasyeo
"""

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np

#%%
"""
A 2 x 2 linear system of equation solver using PETSc

2x - y = 1
3x + 4y = 3

Uses KSP (Krylov subspace method)
"""


def linearsys():
    PETSc.Sys.Print("2 x 2 Linear System Example")
    
    A = PETSc.Mat().createAIJ([2,2])
    A.setValues([0,1], [0,1], [[2.0, -1.0], [3.0, 4.0]])
    A.assemble()
    
    b = PETSc.Vec().createSeq(2)
    b.setValues([0,1], [1.0, 3.0])
    b.assemble()
    
    x = PETSc.Vec().createSeq(2)
    
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    
    print("Solution:", x.getArray())
    

#%%
"""
A 2 x 2 nonlinear system of equation solver using PETSc

x^2 + y - 3 = 0
x + y^2 - 4 = 0

=> F(x) = 0

Uses Scalable Nonlinear Equations Solvers (SNES)
"""


def nonlinearsys():
    PETSc.Sys.Print("2 x 2 non-Linear System Example")
    
    def F(snes, x, f):
        xx = x.getArray(readonly=True)
        f.setValue(0, xx[0]**2 + xx[1] - 3)
        f.setValue(1, xx[0] + xx[1]**2 - 4)
        f.assemble()
        
    snes = PETSc.SNES().create()
    snes.setFromOptions()
    
    x = PETSc.Vec().createSeq(2)
    x.setValues([0,1], [1, 1])
    
    residual = PETSc.Vec().createSeq(2)
    
    snes.setFunction(F, residual)
    snes.setJacobian(None, None)

    snes.solve(None, x)
    
    print("Solution:", x.getArray())
    


#%%

if __name__ == "__main__":
    opts = PETSc.Options()
    which = opts.getString("system", None)
    
    if which is None:
        PETSc.Sys.Print("Error: You must specify a system with -system <linear|nonlinear>")
        PETSc.Sys.Exit(1) 
    
    if which == "linear":
        linearsys()
    
    elif which == "nonlinear":
        nonlinearsys()
    