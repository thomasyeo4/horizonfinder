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
###############################################################################
class spheresym:
    #------ Only provide psi (conformal factor) and Kij (extrinsic curvature)
    #------ and the code will do the rest!
    
    def __init__(self, psi, Kij):
        PETSc.Sys.Print("Spherical Symmetry spacetime")
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
        Gamma[0, 2, 2] = (-r*(np.sin(theta))**2 
                          - 2*r**2*(np.sin(theta))**2*(psiprime/psi))
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
###############################################################################
class axisym:
    #------ Only provide gammaij (spatial metric) and Kij (extrinsic curvature)
    #------ and the code will do the rest!
    
    def __init__(self, gammaij, Kij):
        PETSc.Sys.Print("Axisymmetric spacetime")
        self.gammaij = gammaij
        self.Kij = Kij
        
        
        #---------- Inverse gamma_ij ----------
    def gammainverse(self, r, theta):
        gamma = self.gammaij(r, theta)
        gammainv = np.linalg.inv(gamma)
        
        return gammainv
    
        
        #---------- Finite-differenced gamma_ij for Christoffel ---------
    def metricderiv(self, r, theta, dr=1e-6, dtheta=1e-6):
        gammar = np.zeros((3, 3))
        gammatheta = np.zeros((3, 3))
        
        gamma_rfront = self.gammaij(r+dr, theta)
        gamma_rback = self.gammaij(r-dr, theta)
        gamma_thetafront = self.gammaij(r, theta+dtheta)
        gamma_thetaback = self.gammaij(r, theta-dtheta)
        
        for i in range(3):
            for j in range(3):
                gammar[i,j] = (gamma_rfront[i,j] - gamma_rback[i,j]) / (2*dr)
                gammatheta[i,j] = (gamma_thetafront[i,j] - gamma_thetaback[i,j]) / (2*dtheta)
        
        return gammar, gammatheta
    
        
        #---------- Christoffel for axisymmetric spacetimes ----------
    def Christoffels(self, r, theta):
        gammainv = self.gammainverse(r, theta)
        gammar, gammatheta = self.metricderiv(r, theta)
        
        Gamma = np.zeros((3, 3, 3))
        
        Gamma[0, 0, 0] = (0.5*gammainv[0,0]*gammar[0,0] 
                            + gammainv[0,1]*(gammar[1,0] - 0.5*gammatheta[0,0]) 
                            + gammainv[0,2]*gammar[2,0])
        Gamma[0, 0, 1] = 0.5*(gammainv[0,1]*gammar[1,1] 
                              + gammainv[0,0]*gammatheta[0,0]
                              + gammainv[0,2]*(gammar[2,1] + gammatheta[2,0]))
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = (gammainv[0,0]*(gammatheta[0,1] - 0.5*gammar[1,1]) 
                          + gammainv[0,2]*gammatheta[2,1] 
                          + 0.5*gammainv[0,1]*gammatheta[1,1])
        Gamma[0, 2, 2] = -0.5*(gammainv[0,0]*gammar[2,2] 
                                + gammainv[0,1]*gammatheta[2,2])
        Gamma[1, 0, 0] = (0.5*gammainv[1,0]*gammar[0,0] 
                          + gammainv[1,1]*(gammar[1,0] + gammatheta[0,0])
                          + gammainv[1,2]*gammar[2,0])
        Gamma[1, 0, 1] = 0.5*(gammainv[1,0]*gammatheta[0,0] 
                              + gammainv[1,1]*gammar[1,1]
                              + gammainv[1,2]*(gammar[2,1] + gammatheta[2,0]))
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = (gammainv[1,0]*(gammatheta[0,1] - 0.5*gammar[1,1])
                          + 0.5*gammainv[1,1]*gammatheta[1,1]
                          + gammainv[1,2]*gammatheta[2,1])
        Gamma[1, 2, 2] = -0.5*(gammainv[1,0]*gammar[2,2]
                               + gammainv[1,1]*gammatheta[2,2])

        return Gamma
    
    
        #----------- The Horizon function (Theta) for Axisymmetric case (Residual) -------
    def Horizon(self, h, theta, dtheta):
        
        # Number of points on grid -> Defines number of h values on grid 
        N = len(h)
        Residual = np.zeros(N)
        
        # Extend grid of h to Hext then impose BCs at poles 
        Hext = np.empty(N + 4)
        Hext[2:N + 2] = h[:]
        Hext[1] = h[1]
        Hext[0] = h[2]
        Hext[N + 2] = h[N - 2]
        Hext[N + 3] = h[N - 3]        
        
        # 4th order derivatives
        hfirst = np.zeros(N)
        hsecond = np.zeros(N)
        for i in range(N):
            j = i + 2
            hfirst[i] = (Hext[j - 2] - 8*Hext[j - 1] 
                         + 8*Hext[j + 1] - Hext[j + 2]) / (12*dtheta)
            hsecond[i] = (-Hext[j - 2] + 16*Hext[j - 1] - 30*Hext[j] 
                         + 16*Hext[j + 1] - Hext[j + 2]) / (12*dtheta**2)
        
        # Evaluating the residual in the interior points
        for i in range(N):
            # Setting r_i = h_i(theta) and theta_i for all coefficients
            r = h[i]
            t = theta[i]
            gamma = self.gammaij(r, t)
            gammainv = self.gammainverse(r, t)
            Gamma = self.Christoffels(r, t)
            Kij = self.Kij(r, t)
        
            # Evaluating (ds/dtheta)^2 and gamma^(2)
            dsdt2 = ((hfirst[i]**2)*gamma[0, 0] + gamma[1,  1] 
                     + 2*hfirst[i]*gamma[0, 1])
            gamma2 = gamma[0, 0]*gamma[1, 1] - gamma[0, 1]**2

            # Residual terms
            first = hsecond[i]
            
            second = (Gamma[1, 0, 0]*hfirst[i]**3 
                      + (Gamma[0, 0, 0] + 2*Gamma[1, 0, 1])*hfirst[i]**2
                      + (Gamma[1, 1, 1] + 2*Gamma[0, 1, 0])*hfirst[i] 
                      + Gamma[0, 1, 1])
            
            third = dsdt2*gammainv[2, 2]*(Gamma[0, 2, 2] 
                           - Gamma[1, 2, 2]*hfirst[i])

            fourth = np.sqrt(dsdt2 / gamma2)*(Kij[0, 0]*hfirst[i]**2 
                                                       + 2*hfirst[i]*Kij[0, 1]
                                                       + Kij[1, 1])
        
            fifth = (np.sqrt(1 / gamma2)*
                     ((np.sqrt(dsdt2))**3)*gammainv[2, 2]*Kij[2, 2])
            
            # The residual term aka the Horizon function 
            Residual[i] = first + second + third + fourth + fifth 
            
        return Residual
    
    
    
        #---------- The SNES solver by PETSc -------
    def Solver(self, hguess, Ntheta):
        
        # Theta grid
        epsilon = 1e-8  # small offset to avoid poles
        theta = np.linspace(epsilon, np.pi - epsilon, Ntheta)
        dtheta = theta[1] - theta[0]
        
        # Initial guess horizon function
        h0 = np.array([hguess(t) for t in theta])
        
        # Create PETSc vector for horizon function
        hvec = PETSc.Vec().createSeq(Ntheta)
        for i in range(Ntheta):
            hvec.setValue(i, h0[i])
        hvec.assemble()
        
        # Residual function for PETSc
        def F(snes, x, f):
            h = x.getArray(readonly=True).copy()
            Residual = self.Horizon(h, theta, dtheta)
            f[:] = Residual
        
        # Creating SNES object
        snes = PETSc.SNES().create()
        snes.setFunction(F)
        
        # Take options from input.py or command line
        snes.setFromOptions()
        
        # Solves the system
        snes.solve(None, hvec)
        hsol = hvec.getArray()
            
        return hsol, theta






    


#%%

#class nosym:
    
    

#%%
###############################################################################
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
    
    
    # --- Load solver options from input ---
    snes_type = getattr(user_input, "snes_type", None)
    snes_mf = getattr(user_input, "snes_mf", None)
    use_multigrid = getattr(user_input, "use_multigrid", None)
    rguess = getattr(user_input, "rguess", None)
    hguess = getattr(user_input, "hguess", None) 
    Ntheta = getattr(user_input, "Ntheta", None)
    outfile = getattr(user_input, "output_file", "data.csv")
    
    
    # --- Set PETSc options ---
    opts = PETSc.Options()
    opts["snes_type"] = snes_type
    if use_multigrid:
        opts["ksp_type"] = "gmres"
        opts["pc_type"] = "mg"
        # Future multigrid-specific options can also be added here
        
    
    # ---- Symmetry type ----
    if symmetry == "spheresym":
        psi = getattr(user_input, "psi", None)
        if psi is None: 
            raise ValueError("psi missing in input file.")
            
        solver = spheresym(psi, Kij)
        rhorizon = solver.Solver(rguess=rguess)
        PETSc.Sys.Print("Done! Horizon radius:", rhorizon)
        
    elif symmetry in ["axisym", "nosym"]:
        gammaij = getattr(user_input, "gammaij", None)
        if gammaij is None:
            raise ValueError("gammaij missing in input file.")
        if hguess is None or Ntheta is None:
            raise ValueError("Axisym input file must define hguess(theta) and Ntheta.") 
            
        solver = axisym(gammaij, Kij)
        hsol, theta = solver.Solver(hguess, Ntheta)
        
        data = np.column_stack((theta, hsol))  # shape (Ntheta, 2)
        np.savetxt(outfile, data, delimiter=",", header="theta,h", comments="")
        #PETSc.Sys.Print("Horizon shape h(theta):", hsol)
        PETSc.Sys.Print(f"Done! Data saved to {outfile}")
        
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")
    
    