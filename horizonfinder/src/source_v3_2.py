"""
This is the source code of the Horizon Solver
The __init__.py is in the same folder to make the classes in this file
callable for diagnostics etc. 

A code working in progress...

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
    # Only provide psi (conformal factor) and Kij (extrinsic curvature)
    # and the code will do the rest!
    def __init__(self, psi, Kij):
        PETSc.Sys.Print("Solving for horizon ...")
        self.psi = psi
        self.Kij = Kij

    # ------ Build LHS matrix A from Eq. 2.1 (Shibata 2000) ------
    # Full domain: theta in [0, pi], phi in [0, 2pi]
    # L[h] = h_tt + cot(t)*h_t + h_pp/sin^2(t) - 2h
    # BCs:
    #   phi: 2pi-periodic  ->  h(i-1) wraps, h(i+1) wraps
    #   theta poles: half-period phi shift (Shibata 2000, Eqs. 2.7-2.8)
    #     h(i, j=-1)       = h((i + Nphi//2) % Nphi, j=0)
    #     h(i, j=Ntheta)   = h((i + Nphi//2) % Nphi, j=Ntheta-1)
    def build_A(self, Ntheta, Nphi):
        # Grid spacings for cell-centered grid
        dtheta  = np.pi / Ntheta          # theta in [0, pi]
        dphi    = 2.0 * np.pi / Nphi      # phi   in [0, 2pi]
        inv_dt2 = 1.0 / (dtheta**2)
        inv_dp2 = 1.0 / (dphi**2)
        half    = Nphi // 2               # half-period phi shift for pole BC

        n = Ntheta * Nphi
        # 5 nonzeros per row: center + phi+/- + theta+/-
        A = PETSc.Mat().createAIJ([n, n], nnz=5, comm=PETSc.COMM_SELF)
        A.setUp()

        # Global flat index: p = j * Nphi + i
        #   j = theta index (0..Ntheta-1), i = phi index (0..Nphi-1)
        for j in range(Ntheta):
            theta = (j + 0.5) * dtheta
            s     = np.sin(theta)
            c     = np.cos(theta)
            phi_c = inv_dp2 / (s * s)        # coeff for phi neighbors
            cot_t = c / s
            a_tp  = inv_dt2 + 0.5 * cot_t / dtheta   # theta+ stencil coeff
            a_tm  = inv_dt2 - 0.5 * cot_t / dtheta   # theta- stencil coeff
            a_0   = -2.0 * inv_dt2 - 2.0 * phi_c - 2.0  # center coeff

            for i in range(Nphi):
                p  = j * Nphi + i
                ip = (i + 1) % Nphi
                im = (i - 1 + Nphi) % Nphi

                cols = []
                vals = []

                # phi neighbors: always periodic 
                cols += [j * Nphi + ip, j * Nphi + im]
                vals += [phi_c, phi_c]

                # theta neighbors with Shibata 2000 pole BCs 
                if j == 0:
                    # Real theta+ neighbor
                    cols.append(1 * Nphi + i)
                    vals.append(a_tp)
                    # Ghost theta- maps to half-shifted phi on same row j=0
                    # h(i, -1) = h((i + half) % Nphi, 0)
                    cols.append(0 * Nphi + (i + half) % Nphi)
                    vals.append(a_tm)
                elif j == Ntheta - 1:
                    # Real theta- neighbor
                    cols.append((j - 1) * Nphi + i)
                    vals.append(a_tm)
                    # Ghost theta+ maps to half-shifted phi on same row j=Ntheta-1
                    # h(i, Ntheta) = h((i + half) % Nphi, Ntheta-1)
                    cols.append(j * Nphi + (i + half) % Nphi)
                    vals.append(a_tp)
                else:
                    cols.append((j + 1) * Nphi + i)
                    vals.append(a_tp)
                    cols.append((j - 1) * Nphi + i)
                    vals.append(a_tm)

                # center diagonal 
                cols.append(p)
                vals.append(a_0)

                A.setValues(p, cols, vals)

        A.assemblyBegin()
        A.assemblyEnd()

        return A, dtheta, dphi


    # ----- Build grid and precompute trig factors for source term evaluation ------
    def grid(self, Ntheta, Nphi):

        self.Ntheta = Ntheta
        self.Nphi   = Nphi

        self.dtheta = np.pi / Ntheta
        self.dphi   = 2.0 * np.pi / Nphi

        # cell-centered coordinates
        theta = (np.arange(Ntheta) + 0.5) * self.dtheta
        phi   = (np.arange(Nphi) + 0.5) * self.dphi

        # meshgrid
        self.t, self.p = np.meshgrid(theta, phi, indexing="ij")

        # useful trig factors
        self.sin_t = np.sin(self.t)
        self.cos_t = np.cos(self.t)
        self.cot_t = self.cos_t / self.sin_t

        return self.dtheta, self.dphi


    # ----- Compute finite-difference derivatives of h on the grid for source term evaluation ------
    def h_derivatives(self, h):

        Ntheta = self.Ntheta
        Nphi   = self.Nphi
        dtheta = self.dtheta
        dphi   = self.dphi
        half   = Nphi // 2

        h2 = h.reshape(Ntheta, Nphi)
        ht  = np.zeros_like(h2)
        hp  = np.zeros_like(h2)
        htt = np.zeros_like(h2)
        hpp = np.zeros_like(h2)
        htp = np.zeros_like(h2)

        # phi derivatives (periodic) 
        hp  = (np.roll(h2,-1,axis=1) - np.roll(h2,1,axis=1))/(2*dphi)
        hpp = (np.roll(h2,-1,axis=1) - 2*h2 + np.roll(h2,1,axis=1))/(dphi**2)
        # theta derivatives interior 
        ht[1:-1,:]  = (h2[2:,:] - h2[:-2,:])/(2*dtheta)
        htt[1:-1,:] = (h2[2:,:] - 2*h2[1:-1,:] + h2[:-2,:])/(dtheta**2)
        # north pole 
        h_shift = np.roll(h2[0,:], half)
        ht[0,:]  = (h2[1,:] - h_shift)/(2*dtheta)
        htt[0,:] = (h2[1,:] - 2*h2[0,:] + h_shift)/(dtheta**2)
        # south pole 
        h_shift = np.roll(h2[-1,:], half)
        ht[-1,:]  = (h_shift - h2[-2,:])/(2*dtheta)
        htt[-1,:] = (h_shift - 2*h2[-1,:] + h2[-2,:])/(dtheta**2)
        # mixed derivative interior 
        htp[1:-1,:] = (np.roll(h2[2:,:],-1,axis=1) - np.roll(h2[2:,:],1,axis=1)
                        - np.roll(h2[:-2,:],-1,axis=1) + np.roll(h2[:-2,:],1,axis=1))/(4*dtheta*dphi)
        # mixed derivative north pole
        h_jp = h2[1,:]
        h_jm = np.roll(h2[0,:], half)
        htp[0,:] = (np.roll(h_jp,-1) - np.roll(h_jp,1) - np.roll(h_jm,-1) + np.roll(h_jm,1))/(4*dtheta*dphi)
        # mixed derivative south pole 
        h_jp = np.roll(h2[-1,:], half)
        h_jm = h2[-2,:]
        htp[-1,:] = (np.roll(h_jp,-1) - np.roll(h_jp,1) - np.roll(h_jm,-1) + np.roll(h_jm,1))/(4*dtheta*dphi)

        return ht, hp, htt, hpp, htp
    

    # ----- Compute psi and its derivatives on the grid for source term evaluation ------
    def psi_derivatives(self, h):

        r = h.reshape(self.Ntheta, self.Nphi)
        t = self.t
        p = self.p
        dtheta = self.dtheta
        dphi   = self.dphi
        # evaluate psi on horizon
        psi = self.psi(r, t, p)
        # radial derivative
        dr = 1e-6
        psi_r = (self.psi(r+dr, t, p) - self.psi(r-dr, t, p))/(2*dr)
        # theta derivative
        psi_t = (self.psi(r, t+dtheta, p) - self.psi(r, t-dtheta, p))/(2*dtheta)
        # phi derivative
        psi_p = (self.psi(r, t, p+dphi) - self.psi(r, t, p-dphi))/(2*dphi)

        return psi, psi_r, psi_t, psi_p


    # ------ Compute the nonlinear source term S(h, psi, Kij) for the horizon equation ------
    def source_term(self, h):

        # reshape h to grid
        h2 = h.reshape(self.Ntheta, self.Nphi)

        sin_t = self.sin_t
        cos_t = self.cos_t
        cot_t = self.cot_t
        sin2  = sin_t**2

        # derivatives of h 
        ht, hp, htt, hpp, htp = self.h_derivatives(h)
        # psi and derivatives 
        psi, psi_r, psi_t, psi_p = self.psi_derivatives(h)
        # normalization factor C
        C = np.sqrt(h2**2 + ht**2 + hp**2 / sin2)
        # unit normal components
        s_r = h2 / C
        s_t = -ht / C
        s_p = -hp / (C * sin2)

        # extrinsic curvature
        Kij = self.Kij(h2, self.t, self.p)
        Kss = (Kij[0,0]*s_r*s_r + 2*Kij[0,1]*s_r*s_t + 2*Kij[0,2]*s_r*s_p 
                + Kij[1,1]*s_t*s_t + 2*Kij[1,2]*s_t*s_p + Kij[2,2]*s_p*s_p)

        psi4 = psi**4
        gamma_rr = 1/psi4
        gamma_tt = 1/(psi4 * h2**2)
        gamma_pp = 1/(psi4 * h2**2 * sin2)

        Ktrace = (gamma_rr * Kij[0,0] + gamma_tt * Kij[1,1] + gamma_pp * Kij[2,2])

        # term 1 
        termK = (psi**2 * h2**2 / C**3) * (Kss - Ktrace)
        # term 2 
        term2 = (4/psi) * (psi_r - psi_t * ht / h2**2
                    - psi_p * hp / (h2**2 * sin2)) * (h2**2 + ht**2 + hp**2 / sin2)
        # term 3 
        term3 = (3/h2) * (ht**2 + hp**2 / sin2)
        # term 4 
        term4 = (1/(h2**2 * sin2)) * (2*ht*hp*htp - cot_t * hp**2 * ht)
        # term 5
        term5 = -(ht**2/(h2**2 * sin2)) * (sin_t*cos_t*ht + hpp)
        # term 6
        term6 = -(hp**2/(h2**2 * sin2)) * (htt + cot_t*ht)
        # total source 
        S = termK + term2 + term3 + term4 + term5 + term6

        return S


    # ----- Build RHS vector B from source term S(h) for given h ------
    def build_B(self, h_old):

        S = self.source_term(h_old)

        B = PETSc.Vec().createSeq(self.Ntheta * self.Nphi)
        B.setArray(S.flatten())

        return B
    

    # ----- The KSP solver by PETSc -------
    def Solver(self, hguess, Ntheta, Nphi, omega, max_iter):
        
        # build grid
        dtheta, dphi = self.grid(Ntheta, Nphi)

        # build linear operator A
        A, _, _ = self.build_A(Ntheta, Nphi)

        # create linear solver 
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setFromOptions()

        # initial guess 
        h_old = np.array(hguess(self.t, self.p))

        tol = 1e-8
        diff = 1.0
        it = 0

        h_new_vec = PETSc.Vec().createSeq(Ntheta * Nphi)
        B = PETSc.Vec().createSeq(Ntheta * Nphi)

        while diff > tol and it < max_iter:

            B = self.build_B(h_old)

            # solve A h_new = B
            ksp.solve(B, h_new_vec)
            if ksp.getConvergedReason() < 0:
                raise RuntimeError("KSP failed to converge")

            h_new = h_new_vec.getArray().reshape(Ntheta, Nphi)

            # convergence check
            diff = np.linalg.norm(h_new - h_old)

            PETSc.Sys.Print(f"Iter {it}   diff = {diff:e}")

            if not np.isfinite(diff) or diff > 1e6:
                PETSc.Sys.Print("Solver diverged — horizon not found.")
                return None, self.t[:,0], self.p[0,:]

            if diff < tol:
                break

            h_old = (1-omega)*h_old + omega*h_new
            it += 1

        PETSc.Sys.Print("Solver finished.")

        return h_new, self.t[:,0], self.p[0,:]
    
    

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
    ksp_type                 = getattr(user_input, "ksp_type", None)
    pc_type                  = getattr(user_input, "pc_type", None)
    ksp_rtol                 = getattr(user_input, "ksp_rtol", None)
    ksp_atol                 = getattr(user_input, "ksp_atol", None)
    ksp_max_it               = getattr(user_input, "ksp_max_it", None)
    ksp_monitor              = getattr(user_input, "ksp_monitor", False)
    ksp_converged_reason     = getattr(user_input, "ksp_converged_reason", False)
    rguess                   = getattr(user_input, "rguess", None)
    hguess                   = getattr(user_input, "hguess", None) 
    Ntheta                   = getattr(user_input, "Ntheta", None)
    max_iter                 = getattr(user_input, "max_iter", 500)
    omega                    = getattr(user_input, "omega", 0.3)
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
    if ksp_type:
        opts["ksp_type"] = ksp_type
    if pc_type:
        opts["pc_type"] = pc_type
    if ksp_rtol:
        opts["ksp_rtol"] = ksp_rtol
    if ksp_atol:
        opts["ksp_atol"] = ksp_atol
    if ksp_max_it:
        opts["ksp_max_it"] = ksp_max_it
    if ksp_monitor:
        opts["ksp_monitor"] = ksp_monitor
    if ksp_converged_reason:
        opts["ksp_converged_reason"] = ksp_converged_reason

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
        psi            = getattr(user_input, "psi", None)
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
        
        if psi is None or hguess_global is None:
            raise ValueError("Global conformal factor (psi) and hguess_global must be defined in input file.")
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
                solver_bh = nosym(bh["psi"], Kij)
        
                # Call solver with timing
                t_start = time.time()
                hsol, theta_grid, phi_grid = solver_bh.Solver(bh["hguess"], Ntheta_indiv, Nphi_indiv, omega, max_iter)
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
            solver_common = nosym(psi, Kij)
            try:
                # Solve with timing
                t_start = time.time()
                hsol_common, theta_common, phi_common = solver_common.Solver(hguess_global, 
                                                                            Ntheta_common, Nphi_common, omega, max_iter)
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
            
    
