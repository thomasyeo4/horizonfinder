"""
This is the source code of the Horizon Solver
The __init__.py is in the same folder to make the classes in this file
callable for diagnostics etc. 

Diagnostics source code... trying to figure out why PETSc SNES solver is not converging...

"""

import importlib.util
import petsc4py
import h5py # type: ignore
import sys
import os
import time 

petsc4py.init(sys.argv)

from petsc4py import PETSc 
import numpy as np


#%%   SPHERICAL SYMMETRY
###############################################################################
class spheresym:
    #------ Only provide psi (conformal factor) and Kij (extrinsic curvature)
    #------ and the code will do the rest!
    
    def __init__(self, psi, Kij):
        PETSc.Sys.Print("Symmetry type: Spherical")
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Solving for horizon ...")
        self.psi = psi
        self.Kij = Kij
        
        
        #------ Christoffel Symbols for spherical symmetric spacetimes --------
    def Christoffels(self, r, theta=np.pi/2, dr=1e-6):
        psi      = self.psi(r)
        psifront = self.psi(r+dr)
        psiback  = self.psi(r-dr)
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
    
    
        #----- The Horizon function (Theta) for Spherical Symmetric case ------
    def Horizon(self, r):
        Gamma = self.Christoffels(r)
        Kij   = self.Kij(r)
        psi   = self.psi(r)
        
        sup      = np.array([1/psi**2, 0, 0])
        sdown    = np.array([psi**2, 0, 0])
        invgamma = (np.diag([1/psi**4, 1/(psi**4)*r**2, 
                             1/((psi**4)*(r**2)*(np.sin(np.pi/2))**2)]))
        m = invgamma - np.outer(sup, sup)
        
        value = 0
        for i in range(3):
            for j in range(3):
                bracket = 0
                for k in range(3):
                    bracket += sdown[k]*Gamma[k, i, j]
                bracket += Kij[i, j]
                value   += m[i, j]*bracket
        
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
    


#%%   AXISYMMETRY
###############################################################################
class axisym:
    # Only provide gammaij (spatial metric) and Kij (extrinsic curvature)
    # and the code will do the rest!
    
    def __init__(self, gammaij, Kij):
        PETSc.Sys.Print("Solving for horizon ...")
        self.gammaij = gammaij
        self.Kij     = Kij
        
        
        # ----- Inverse gamma_ij -----
    def gammainverse(self, r, theta):
        gamma    = self.gammaij(r, theta)
        gammainv = np.linalg.inv(gamma)
        
        return gammainv
    
        
        # ----- Finite-differenced gamma_ij for Christoffel -----
    def metricderiv(self, r, theta, dr=1e-6, dtheta=1e-6):
        gammar      = np.zeros((3, 3))
        gammatheta  = np.zeros((3, 3))
        
        gamma_rfront        = self.gammaij(r+dr, theta)
        gamma_rback         = self.gammaij(r-dr, theta)
        gamma_thetafront    = self.gammaij(r, theta+dtheta)
        gamma_thetaback     = self.gammaij(r, theta-dtheta)
        
        for i in range(3):
            for j in range(3):
                gammar[i,j]     = (gamma_rfront[i,j] - gamma_rback[i,j]) / (2*dr)
                gammatheta[i,j] = (gamma_thetafront[i,j] - gamma_thetaback[i,j]) / (2*dtheta)
        
        return gammar, gammatheta
    
        
        # ------ Christoffel for axisymmetric spacetimes ------
    def Christoffels(self, r, theta):
        gammainv            = self.gammainverse(r, theta)
        gammar, gammatheta  = self.metricderiv(r, theta)
        
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
        
        # Print convergence reason
        reason = snes.getConvergedReason()
        PETSc.Sys.Print(f"SNES convergence reason: {reason}")
            
        return hsol, theta

#%%     NOSYMMETRY
###############################################################################
class nosym:
    # Only provide gammaij (spatial metric) and Kij (extrinsic curvature)
    # and the code will do the rest!
    def __init__(self, gammaij, Kij):
        PETSc.Sys.Print("Solving for horizon ...")
        self.gammaij = gammaij
        self.Kij     = Kij

    # ----- Inverse gamma_ij -----
    def gammainverse(self, r, theta, phi):
        gamma    = self.gammaij(r, theta, phi)
        gammainv = np.linalg.inv(gamma)
        
        return gammainv
    
    # ----- Finite-differenced gamma_ij for Christoffel -----
    def metricderiv(self, r, theta, phi, dr=1e-6, dtheta=1e-6, dphi=1e-6):
        gamma_rfront        = self.gammaij(r+dr, theta, phi)
        gamma_rback         = self.gammaij(r-dr, theta, phi)
        gamma_thetafront    = self.gammaij(r, theta+dtheta, phi)
        gamma_thetaback     = self.gammaij(r, theta-dtheta, phi)
        gamma_phifront      = self.gammaij(r, theta, phi+dphi)
        gamma_phiback       = self.gammaij(r, theta, phi-dphi)

        # Vectorize the finite differences instead of looping
        gammar     = (gamma_rfront - gamma_rback) / (2*dr)
        gammatheta = (gamma_thetafront - gamma_thetaback) / (2*dtheta)
        gammaphi   = (gamma_phifront - gamma_phiback) / (2*dphi)
        
        return gammar, gammatheta, gammaphi

    # ------ Christoffel for general spacetimes ------
    def Christoffels(self, r, theta, phi, gammainv=None):
        if gammainv is None:
            gammainv = self.gammainverse(r, theta, phi)
        gammar, gammatheta, gammaphi  = self.metricderiv(r, theta, phi)
        
        dgamma        = np.zeros((3,3,3))
        dgamma[0,:,:] = gammar
        dgamma[1,:,:] = gammatheta
        dgamma[2,:,:] = gammaphi

        # Use einsum to replace nested tensor loops
        # Gamma[i,j,k] = 0.5 * sum_l gammainv[i,l] * (dgamma[j,k,l] + dgamma[k,j,l] - dgamma[l,j,k])
        Gamma = 0.5 * (np.einsum('il,jkl->ijk', gammainv, dgamma) + 
                       np.einsum('il,kjl->ijk', gammainv, dgamma) - 
                       np.einsum('il,ljk->ijk', gammainv, dgamma))

        return Gamma

    # ------ Buffer manager to reuse scratch arrays and avoid repeat allocations ------
    def _ensure_buffers(self, Ntheta, Nphi):
        key = (Ntheta, Nphi)
        buf = getattr(self, "_buffers", None)
        if buf is None or buf.get("shape") != key:
            self._buffers = {
                "shape": key,
                "Hext": np.empty((Ntheta + 4, Nphi + 4)),
                "h_theta1": np.empty((Ntheta, Nphi)),
                "h_theta2": np.empty((Ntheta, Nphi)),
                "h_phi1": np.empty((Ntheta, Nphi)),
                "h_phi2": np.empty((Ntheta, Nphi)),
                "h_thetaphi": np.empty((Ntheta, Nphi)),
                "Residual": np.empty((Ntheta, Nphi)),
            }
        return self._buffers

    # ------ Surface Normal vector ------
    def normalvector(self, r, theta, phi, dhdtheta, dhdphi, gammainv=None):
        if gammainv is None:
            gammainv = self.gammainverse(r, theta, phi)

        m = np.array([1.0, -dhdtheta, -dhdphi])

        norm = 0.0
        for i in range(3):
            for j in range(3):
                norm += gammainv[i, j] * m[i] * m[j]    

        lam = 1.0 / np.sqrt(norm)

        s_down = lam * m
        s_up   = gammainv @ s_down

        return s_up, s_down, lam

    # ------ Inverse induced metric on horizon surface ------
    def minverse(self, r, theta, phi, dhdtheta, dhdphi, gammainv=None, s_up=None):
        if gammainv is None:
            gammainv = self.gammainverse(r, theta, phi)
        if s_up is None:
            s_up, _, _ = self.normalvector(r, theta, phi, dhdtheta, dhdphi, gammainv)

        minv = gammainv - np.outer(s_up, s_up)

        return minv

    # ------ The Horizon function (Theta) for general case (Residual) ------
    def Horizon(self, h_flat, theta, phi, dtheta, dphi):
        
        # Number of points on grid -> Defines number of h values on grid 
        Ntheta = len(theta)
        Nphi = len(phi)

        buffers = self._ensure_buffers(Ntheta, Nphi)
        Hext       = buffers["Hext"]
        h_theta1   = buffers["h_theta1"]
        h_theta2   = buffers["h_theta2"]
        h_phi1     = buffers["h_phi1"]
        h_phi2     = buffers["h_phi2"]
        h_thetaphi = buffers["h_thetaphi"]
        Residual   = buffers["Residual"]

        h = h_flat.reshape((Ntheta, Nphi))

        # Extend grid of h to Hext then impose BCs at poles and periodicity in phi
        Hext[2:Ntheta + 2, 2:Nphi + 2] = h[:, :]

        # Theta BCs (poles) for cell-centered theta grid using Shibata mapping
        # (theta, phi) -> (reflected theta, phi + pi), implemented as phi roll.
        half_shift = Nphi // 2
        # North pole side ghosts (rows 1 and 0 in Hext)
        Hext[1,  2:Nphi + 2] = np.roll(h[0, :], half_shift)
        Hext[0,  2:Nphi + 2] = np.roll(h[1, :], half_shift)
        # South pole side ghosts (rows Ntheta+2 and Ntheta+3 in Hext)
        Hext[Ntheta + 2, 2:Nphi + 2] = np.roll(h[Ntheta - 1, :], half_shift)
        Hext[Ntheta + 3, 2:Nphi + 2] = np.roll(h[Ntheta - 2, :], half_shift)

        # Phi BCs (periodic) for every theta row (including pole ghost rows).
        Hext[:, 1] = Hext[:, Nphi + 1]
        Hext[:, 0] = Hext[:, Nphi]
        Hext[:, Nphi + 2] = Hext[:, 2]
        Hext[:, Nphi + 3] = Hext[:, 3]

        # Angular derivatives (2nd order central differences)
        h_theta1[:, :] = (Hext[3:Ntheta+3, 2:Nphi+2] - Hext[1:Ntheta+1, 2:Nphi+2]) / (2*dtheta)
        h_theta2[:, :] = (Hext[3:Ntheta+3, 2:Nphi+2] - 2*Hext[2:Ntheta+2, 2:Nphi+2] + Hext[1:Ntheta+1, 2:Nphi+2]) / (dtheta**2)
        h_phi1[:, :] = (Hext[2:Ntheta+2, 3:Nphi+3] - Hext[2:Ntheta+2, 1:Nphi+1]) / (2*dphi)
        h_phi2[:, :] = (Hext[2:Ntheta+2, 3:Nphi+3] - 2*Hext[2:Ntheta+2, 2:Nphi+2] + Hext[2:Ntheta+2, 1:Nphi+1]) / (dphi**2)
        h_thetaphi[:, :] = (
            Hext[3:Ntheta+3, 3:Nphi+3] - Hext[3:Ntheta+3, 1:Nphi+1]
            - Hext[1:Ntheta+1, 3:Nphi+3] + Hext[1:Ntheta+1, 1:Nphi+1]
        ) / (4*dtheta*dphi)

        # Evaluating the residual in the interior points
        for i in range(Ntheta):
            for j in range(Nphi):
                # Setting r_ij = h_ij(theta, phi) and theta_i, phi_j for all coefficients
                r = h[i, j]
                t = theta[i]
                p = phi[j]
                gamma             = self.gammaij(r, t, p)
                gammainv          = self.gammainverse(r, t, p)
                Gamma             = self.Christoffels(r, t, p, gammainv)
                s_up, s_down, lam = self.normalvector(r, t, p, h_theta1[i, j], h_phi1[i, j], gammainv)
                minv              = self.minverse(r, t, p, h_theta1[i, j], h_phi1[i, j], gammainv, s_up)
                Kij               = self.Kij(r, t, p)

                # Residual terms
                first = lam*(minv[1,1]*h_theta2[i, j] + minv[2,2]*h_phi2[i, j] + minv[1,2]*h_thetaphi[i, j]
                             + minv[2,1]*h_thetaphi[i, j])

                # Use einsum to replace nested tensor loops
                second = np.einsum('ab,cab,c->', minv, Gamma, s_down)
                third = np.einsum('ab,ab->', minv, Kij)

                # The residual term aka the Horizon function 
                Residual[i, j] = first - second - third
        
        return Residual.reshape(-1)

    #---------- The SNES solver by PETSc -------
    def Solver(self, hguess, Ntheta, Nphi):

        # Cell-centered theta and phi grids
        dtheta = np.pi / Ntheta
        dphi   = 2*np.pi / Nphi
        theta  = (np.arange(Ntheta) + 0.5) * dtheta
        phi    = (np.arange(Nphi) + 0.5) * dphi

        # Initial guess horizon function
        h0_flat = np.array([hguess(theta[i], phi[j]) for i in range(Ntheta) for j in range(Nphi)])

        # Create PETSc vector for horizon function
        hvec = PETSc.Vec().createSeq(Ntheta * Nphi)
        for i in range(Ntheta * Nphi):
            hvec.setValue(i, h0_flat[i])
        hvec.assemble()

        # Residual function for PETSc
        def F(snes, x, f):
            h_flat        = x.getArray(readonly=True).copy()
            Residual_flat = self.Horizon(h_flat, theta, phi, dtheta, dphi)
            f[:]          = Residual_flat 

        # Benchmark: time a single Horizon call
        horizont_start = time.time()
        _ = self.Horizon(h0_flat, theta, phi, dtheta, dphi)
        horizont_end = time.time() - horizont_start
        PETSc.Sys.Print(f"Single Horizon call time: {horizont_end:.6f} seconds")

        # Creating SNES object
        snes = PETSc.SNES().create()
        snes.setFunction(F)

        # Take options from input.py or command line
        snes.setFromOptions()

        # Solves the system
        snes.solve(None, hvec)
        hsol_flat = hvec.getArray()
        hsol = hsol_flat.reshape((Ntheta, Nphi))

        # Print convergence reason
        reason = snes.getConvergedReason()
        PETSc.Sys.Print(f"SNES convergence reason: {reason}")

        return hsol, theta, phi
        
#%%     DRIVER
###############################################################################
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: python <path_to>/source_NewtonSNES_diag.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    spec       = importlib.util.spec_from_file_location("user_input", input_file)
    user_input = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_input)
    
    PETSc.Sys.Print("====================================================")
    
    # --- Get system name from input file ---
    system_name = getattr(user_input, "system_name", "Unknown")
    PETSc.Sys.Print(f"System: {system_name}")
    
    # --- Get spacetime name from input file ---
    coord_sys = getattr(user_input, "coord_sys", "Unknown")
    PETSc.Sys.Print(f"Coordinate system: {coord_sys}")
    
    # ---- Symmetry type ----
    symmetry = getattr(user_input, "symmetry", None)
    if symmetry is None:
        raise ValueError("Input file must define symmetry type!")
    
    
    # ---- Mandatory input ----
    Kij = getattr(user_input, "Kij", None)
    if Kij is None:
        raise ValueError("Kij missing in input file.")
    
    
    # --- Load solver options from input ---
    snes_type                = getattr(user_input, "snes_type", None)
    snes_linesearch_type     = getattr(user_input, "snes_linesearch_type", None)
    snes_linesearch_maxstep  = getattr(user_input, "snes_linesearch_maxstep", None)
    snes_linesearch_damping  = getattr(user_input, "snes_linesearch_damping", None)
    snes_linesearch_monitor  = getattr(user_input, "snes_linesearch_monitor", False)
    snes_rtol                = getattr(user_input, "snes_rtol", None)
    snes_atol                = getattr(user_input, "snes_atol", None)
    snes_stol                = getattr(user_input, "snes_stol", None)
    snes_max_it              = getattr(user_input, "snes_max_it", None)
    snes_mf                  = getattr(user_input, "snes_mf", None)
    snes_monitor             = getattr(user_input, "snes_monitor", False)
    rguess                   = getattr(user_input, "rguess", None)
    hguess                   = getattr(user_input, "hguess", None) 
    Ntheta                   = getattr(user_input, "Ntheta", None)
    save_iterations          = getattr(user_input, "save_iterations", False)
    outfile                  = getattr(user_input, "output_file", "data.csv")
    
    
    # --- Set PETSc options ---
    opts = PETSc.Options()
    if snes_type:
        opts["snes_type"] = snes_type
    if snes_mf:
        opts["snes_mf_operator"] = None
    if snes_monitor:
        opts["snes_monitor"] = None        
    if snes_rtol:
        opts["snes_rtol"] = snes_rtol
    if snes_atol:
        opts["snes_atol"] = snes_atol
    if snes_stol:
        opts["snes_stol"] = snes_stol
    if snes_max_it:
        opts["snes_max_it"] = snes_max_it
    
        
    
    # ---- Symmetry type ----
    if symmetry == "spheresym":
        psi = getattr(user_input, "psi", None)
        if psi is None: 
            raise ValueError("psi missing in input file.")
            
        solver   = spheresym(psi, Kij)
        rhorizon = solver.Solver(rguess=rguess)
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Job done! Horizon radius:", rhorizon)
        
    
    elif symmetry in ["axisym"]:
        PETSc.Sys.Print("Symmetry type: Axisymmetry")
        gammaij_global = getattr(user_input, "gammaij", None)
        hguess_global  = getattr(user_input, "hguess", None)
        blackholes     = getattr(user_input, "blackholes", [])
        find_indiv     = getattr(user_input, "find_indiv", False)
        output_dir     = getattr(user_input, "output_dir", "./data")
        Ntheta_indiv   = getattr(user_input, "Ntheta_indiv", 150)
        Ntheta_common  = getattr(user_input, "Ntheta_common", 400)
        save_iterations= getattr(user_input, "save_iterations", False)
        Kij            = getattr(user_input, "Kij", None)
        
        if gammaij_global is None or hguess_global is None:
            raise ValueError("Global metric (gammaij_global) and hguess_global must be defined in input file.")
        if Kij is None:
            raise ValueError("Kij missing in input file.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper functions: enclosure test for origin-centered horizon
        def inside(h_theta, theta_grid, point_xyz):
            x, y, z = point_xyz
            r_p = np.sqrt(x*x + y*y + z*z)
            if r_p > 0.0:
                theta_p = np.arccos(np.clip(z / r_p, -1.0, 1.0))
            else:
                theta_p = 0.0
            h_interp = np.interp(theta_p, theta_grid, h_theta)
            return r_p < h_interp

        def horizon_encloses_both(h_theta, theta_grid, z0_val):
            inside1 = inside(h_theta, theta_grid, (0.0, 0.0, -z0_val))
            inside2 = inside(h_theta, theta_grid, (0.0, 0.0, +z0_val))
            return inside1, inside2
        
        results = {}
        
        PETSc.Sys.Print("====================================================")
        # --- Individual horizons ---
        if find_indiv:
            PETSc.Sys.Print("Finding individual horizons:")
            for bh in blackholes:
                PETSc.Sys.Print(f"Solving for {bh['name']}.")
        
                # Create a solver instance for this BH
                solver_bh = axisym(bh["gammaij"], Kij)
        
                # Call solver with timing
                t_start = time.time()
                hsol, theta_grid = solver_bh.Solver(bh["hguess"], Ntheta_indiv)
                t_elapsed = time.time() - t_start
        
                # Save the data
                outfile = os.path.join(output_dir, f"{bh['name']}_horizon.h5")
                with h5py.File(outfile, 'w') as f:
                    f.create_dataset('h', data=hsol, compression='gzip', compression_opts=9)
                    f.create_dataset('theta', data=theta_grid, compression='gzip')
                PETSc.Sys.Print(f"Individual horizon for {bh['name']} saved to {outfile}")
                PETSc.Sys.Print(f"  Computation time: {t_elapsed:.4f} seconds")
        else:
            PETSc.Sys.Print("Skipping individual horizon search (find_indiv = False)...")
        
        # --- Common horizon ---
        PETSc.Sys.Print("====================================================")
        sep_val = getattr(user_input, "sep", None)
        skip_common = False
        if sep_val is not None and sep_val > 1.0:
            skip_common = True
            PETSc.Sys.Print(f"Skipping common horizon search: separation = {sep_val} > 1.0")
        
        if not skip_common:
            PETSc.Sys.Print("Finding common/origin-centered horizon.")
            solver_common = axisym(gammaij_global, Kij)
            try:
                # Call solver with timing
                t_start = time.time()
                hsol_common, theta_common = solver_common.Solver(hguess_global, Ntheta_common)
                t_elapsed = time.time() - t_start
        
                fname_common = os.path.join(output_dir, "horizon_common.h5")
                with h5py.File(fname_common, 'w') as f:
                    f.create_dataset('h', data=hsol_common, compression='gzip', compression_opts=9)
                    f.create_dataset('theta', data=theta_common, compression='gzip')
                PETSc.Sys.Print(f"Saved common horizon -> {fname_common}")
                PETSc.Sys.Print(f"  Computation time: {t_elapsed:.4f} seconds")
                results["common"] = (hsol_common, theta_common)
        
                z0_val = getattr(user_input, "z0", None)
                if z0_val is not None:

                    enc1, enc2 = horizon_encloses_both(hsol_common, theta_common, z0_val)
                    PETSc.Sys.Print(f"Common horizon encloses punctures? BH1: {enc1}, BH2: {enc2}")
        
            except Exception as e:
                PETSc.Sys.Print(f"  ERROR: Common solver raised exception: {e}")
        
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Use 'AxisymmetricPlotting.ipynb' for visualisation.")
    

    elif symmetry in ["nosym"]:
        PETSc.Sys.Print("Symmetry type: General symmetry")
        gammaij_global = getattr(user_input, "gammaij", None)
        hguess_global  = getattr(user_input, "hguess", None)
        blackholes     = getattr(user_input, "blackholes", [])
        find_indiv     = getattr(user_input, "find_indiv", False)
        output_dir     = getattr(user_input, "output_dir", "./data")
        Ntheta_indiv   = getattr(user_input, "Ntheta_indiv", 150)
        Nphi_indiv     = getattr(user_input, "Nphi_indiv", 150)
        Ntheta_common  = getattr(user_input, "Ntheta_common", 400)
        Nphi_common    = getattr(user_input, "Nphi_common", 400)
        save_iterations= getattr(user_input, "save_iterations", False)
        Kij            = getattr(user_input, "Kij", None)
        
        if gammaij_global is None or hguess_global is None:
            raise ValueError("Global metric (gammaij_global) and hguess_global must be defined in input file.")
        if Kij is None:
            raise ValueError("Kij missing in input file.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper functions: enclosure test for origin-centered horizon
        def bilinear_interp(h, theta, phi, theta_p, phi_p):
            """Bilinear interpolation of h(theta, phi) at arbitrary point."""
            Nphi = len(phi)
            phi_p = phi_p % (2*np.pi)  # enforce periodicity

            # Find phi indices surrounding phi_p
            j1 = np.searchsorted(phi, phi_p)
            j0 = j1 - 1
            if j1 >= Nphi:
                j1 = 0  # wrap around for periodicity
                j0 = Nphi - 1
            u = (phi_p - phi[j0]) / (phi[j1] - phi[j0])

            # Interpolate along theta for both phi indices
            h_theta0 = np.interp(theta_p, theta, h[:, j0])
            h_theta1 = np.interp(theta_p, theta, h[:, j1])

            # Interpolate along phi
            h_interp = (1 - u) * h_theta0 + u * h_theta1
            return h_interp

        def inside(h, theta, phi, point_xyz):
            x, y, z = point_xyz
            r_p = np.sqrt(x*x + y*y + z*z)
            if r_p > 0.0:
                theta_p = np.arccos(np.clip(z / r_p, -1.0, 1.0))
                phi_p   = np.arctan2(y, x) % (2*np.pi)
            else:
                theta_p = 0.0
                phi_p   = 0.0

            h_interp = bilinear_interp(h, theta, phi, theta_p, phi_p)
            return r_p < h_interp


        def horizon_encloses_both(h, theta, phi, points_xyz):
            enc_results = [inside(h, theta, phi, pt) for pt in points_xyz]
            return enc_results
        
        PETSc.Sys.Print("====================================================")
        # --- Individual horizons ---
        if find_indiv:
            PETSc.Sys.Print("Finding individual horizons:")
            for bh in blackholes:
                PETSc.Sys.Print(f"Solving for {bh['name']}.")
        
                # Create a solver instance for this BH
                solver_bh = nosym(bh["gammaij"], Kij)
        
                # Call solver with timing
                t_start = time.time()
                hsol, theta_grid, phi_grid = solver_bh.Solver(bh["hguess"], Ntheta_indiv, Nphi_indiv)
                t_elapsed = time.time() - t_start
        
                # Save the data
                outfile = os.path.join(output_dir, f"{bh['name']}_horizon.h5")
                with h5py.File(outfile, 'w') as f:
                    f.create_dataset('h', data=hsol, compression='gzip', compression_opts=9)
                    f.create_dataset('theta', data=theta_grid, compression='gzip')
                    f.create_dataset('phi', data=phi_grid, compression='gzip')
                PETSc.Sys.Print(f"Individual horizon for {bh['name']} saved to {outfile}")
                PETSc.Sys.Print(f"  Computation time: {t_elapsed:.4f} seconds")
        else:
            PETSc.Sys.Print("Skipping individual horizon search (find_indiv = False)...")
        
        # --- Common horizon ---
        PETSc.Sys.Print("====================================================")
        sep_val = getattr(user_input, "sep", None)
        skip_common = False
        if sep_val is not None and sep_val > 1.0:
            skip_common = True
            PETSc.Sys.Print(f"Skipping common horizon search: separation = {sep_val} > 1.0")
        
        if not skip_common:
            PETSc.Sys.Print("Finding common/origin-centered horizon.")
            solver_common = nosym(gammaij_global, Kij)
            try:
                # Solve with timing
                t_start = time.time()
                hsol_common, theta_common, phi_common = solver_common.Solver(hguess_global, Ntheta_common, Nphi_common)
                t_elapsed = time.time() - t_start
                
                # Save results
                fname_common = os.path.join(output_dir, "horizon_common.h5")
                with h5py.File(fname_common, 'w') as f:
                    f.create_dataset('h', data=hsol_common, compression='gzip', compression_opts=9)
                    f.create_dataset('theta', data=theta_common, compression='gzip')
                    f.create_dataset('phi', data=phi_common, compression='gzip')
                PETSc.Sys.Print(f"Saved common horizon -> {fname_common}")
                PETSc.Sys.Print(f"  Computation time: {t_elapsed:.4f} seconds")

                # Check enclosure
                points = [(0,0,-getattr(user_input,"z0",0)), (0,0,+getattr(user_input,"z0",0))]
                enc_results = horizon_encloses_both(hsol_common, theta_common, phi_common, points)
                PETSc.Sys.Print(f"Common horizon encloses punctures? {enc_results}")

            except Exception as e:
                PETSc.Sys.Print(f"ERROR: Common solver raised exception: {e}")
        
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Use 'GenSymmetricPlotting.ipynb' for visualisation.")

    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")
            
    
