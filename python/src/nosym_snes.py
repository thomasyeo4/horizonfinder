"""
Horizon Solver — No Symmetry (Nonlinear SNES)
===============================================
Solves the full nonlinear apparent horizon equation using PETSc SNES.
Provide gammaij (spatial metric) and Kij (extrinsic curvature)
and the solver will find the apparent horizon shape h(theta, phi).

Supports:
  - Single / common horizon search
  - Individual horizon search per black hole (set find_indiv = True)

Usage:
    python nosym_snes.py <input_file>
"""

import importlib.util
import petsc4py
import h5py
import sys
import os
import time

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np


#%%     NOSYMMETRY — SNES (Nonlinear solver)
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
        gamma_rfront     = self.gammaij(r+dr, theta, phi)
        gamma_rback      = self.gammaij(r-dr, theta, phi)
        gamma_thetafront = self.gammaij(r, theta+dtheta, phi)
        gamma_thetaback  = self.gammaij(r, theta-dtheta, phi)
        gamma_phifront   = self.gammaij(r, theta, phi+dphi)
        gamma_phiback    = self.gammaij(r, theta, phi-dphi)

        # Vectorize the finite differences instead of looping
        gammar     = (gamma_rfront - gamma_rback) / (2*dr)
        gammatheta = (gamma_thetafront - gamma_thetaback) / (2*dtheta)
        gammaphi   = (gamma_phifront - gamma_phiback) / (2*dphi)

        return gammar, gammatheta, gammaphi

    # ------ Christoffel for general spacetimes ------
    def Christoffels(self, r, theta, phi, gammainv=None):
        if gammainv is None:
            gammainv = self.gammainverse(r, theta, phi)
        gammar, gammatheta, gammaphi = self.metricderiv(r, theta, phi)

        dgamma        = np.zeros((3,3,3))
        dgamma[0,:,:] = gammar
        dgamma[1,:,:] = gammatheta
        dgamma[2,:,:] = gammaphi

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

        Ntheta = len(theta)
        Nphi   = len(phi)

        buffers    = self._ensure_buffers(Ntheta, Nphi)
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

        # Theta BCs (poles): Shibata 2000 half-period phi shift
        half_shift = Nphi // 2
        Hext[1,          2:Nphi + 2] = np.roll(h[0, :],          half_shift)
        Hext[0,          2:Nphi + 2] = np.roll(h[1, :],          half_shift)
        Hext[Ntheta + 2, 2:Nphi + 2] = np.roll(h[Ntheta - 1, :], half_shift)
        Hext[Ntheta + 3, 2:Nphi + 2] = np.roll(h[Ntheta - 2, :], half_shift)

        # Phi BCs (periodic)
        Hext[:, 1]        = Hext[:, Nphi + 1]
        Hext[:, 0]        = Hext[:, Nphi]
        Hext[:, Nphi + 2] = Hext[:, 2]
        Hext[:, Nphi + 3] = Hext[:, 3]

        # Angular derivatives (2nd order central differences)
        h_theta1[:, :]   = (Hext[3:Ntheta+3, 2:Nphi+2] - Hext[1:Ntheta+1, 2:Nphi+2]) / (2*dtheta)
        h_theta2[:, :]   = (Hext[3:Ntheta+3, 2:Nphi+2] - 2*Hext[2:Ntheta+2, 2:Nphi+2] + Hext[1:Ntheta+1, 2:Nphi+2]) / (dtheta**2)
        h_phi1[:, :]     = (Hext[2:Ntheta+2, 3:Nphi+3] - Hext[2:Ntheta+2, 1:Nphi+1]) / (2*dphi)
        h_phi2[:, :]     = (Hext[2:Ntheta+2, 3:Nphi+3] - 2*Hext[2:Ntheta+2, 2:Nphi+2] + Hext[2:Ntheta+2, 1:Nphi+1]) / (dphi**2)
        h_thetaphi[:, :] = (
            Hext[3:Ntheta+3, 3:Nphi+3] - Hext[3:Ntheta+3, 1:Nphi+1]
            - Hext[1:Ntheta+1, 3:Nphi+3] + Hext[1:Ntheta+1, 1:Nphi+1]
        ) / (4*dtheta*dphi)

        # Evaluating the residual at each grid point
        for i in range(Ntheta):
            for j in range(Nphi):
                r = h[i, j]
                t = theta[i]
                p = phi[j]
                gamma             = self.gammaij(r, t, p)
                gammainv          = self.gammainverse(r, t, p)
                Gamma             = self.Christoffels(r, t, p, gammainv)
                s_up, s_down, lam = self.normalvector(r, t, p, h_theta1[i, j], h_phi1[i, j], gammainv)
                minv              = self.minverse(r, t, p, h_theta1[i, j], h_phi1[i, j], gammainv, s_up)
                Kij               = self.Kij(r, t, p)

                first  = lam*(minv[1,1]*h_theta2[i, j] + minv[2,2]*h_phi2[i, j]
                              + minv[1,2]*h_thetaphi[i, j] + minv[2,1]*h_thetaphi[i, j])
                second = np.einsum('ab,cab,c->', minv, Gamma, s_down)
                third  = np.einsum('ab,ab->', minv, Kij)

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
        h0_flat = np.array([hguess(theta[i], phi[j])
                            for i in range(Ntheta) for j in range(Nphi)])

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
        t_start = time.time()
        _ = self.Horizon(h0_flat, theta, phi, dtheta, dphi)
        PETSc.Sys.Print(f"Single Horizon call time: {time.time() - t_start:.6f} seconds")

        # Creating SNES object
        snes = PETSc.SNES().create()
        snes.setFunction(F)
        snes.setFromOptions()
        snes.solve(None, hvec)

        hsol_flat = hvec.getArray()
        hsol      = hsol_flat.reshape((Ntheta, Nphi))

        reason = snes.getConvergedReason()
        PETSc.Sys.Print(f"SNES convergence reason: {reason}")
        if reason <= 0:
            raise RuntimeError(f"SNES failed to converge (reason={reason}) in nosym solver")

        return hsol, theta, phi


#%%     DRIVER
###############################################################################
if __name__ == "__main__":

    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: python nosym_snes.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    spec       = importlib.util.spec_from_file_location("user_input", input_file)
    user_input = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_input)

    PETSc.Sys.Print("====================================================")

    system_name = getattr(user_input, "system_name", "Unknown")
    PETSc.Sys.Print(f"System: {system_name}")

    coord_sys = getattr(user_input, "coord_sys", "Unknown")
    PETSc.Sys.Print(f"Coordinate system: {coord_sys}")
    PETSc.Sys.Print("Symmetry type: General (no symmetry) — SNES nonlinear solver")

    # --- Mandatory input ---
    Kij = getattr(user_input, "Kij", None)
    if Kij is None:
        raise ValueError("Kij missing in input file.")

    gammaij_global = getattr(user_input, "gammaij", None)
    if gammaij_global is None:
        raise ValueError("gammaij missing in input file.")

    hguess_global = getattr(user_input, "hguess", None)
    if hguess_global is None:
        raise ValueError("hguess missing in input file.")

    # --- Optional input ---
    find_indiv    = getattr(user_input, "find_indiv", False)
    blackholes    = getattr(user_input, "blackholes", [])
    output_dir    = getattr(user_input, "output_dir", "./data")
    Ntheta_indiv  = getattr(user_input, "Ntheta_indiv", 150)
    Nphi_indiv    = getattr(user_input, "Nphi_indiv", 150)
    Ntheta_common = getattr(user_input, "Ntheta_common", 400)
    Nphi_common   = getattr(user_input, "Nphi_common", 400)
    sep_val       = getattr(user_input, "sep", None)
    z0_val        = getattr(user_input, "z0", None)

    os.makedirs(output_dir, exist_ok=True)

    # --- PETSc SNES options ---
    opts = PETSc.Options()
    snes_type               = getattr(user_input, "snes_type", None)
    snes_mf                 = getattr(user_input, "snes_mf", None)
    snes_monitor            = getattr(user_input, "snes_monitor", False)
    snes_rtol               = getattr(user_input, "snes_rtol", None)
    snes_atol               = getattr(user_input, "snes_atol", None)
    snes_stol               = getattr(user_input, "snes_stol", None)
    snes_max_it             = getattr(user_input, "snes_max_it", None)
    snes_linesearch_type    = getattr(user_input, "snes_linesearch_type", None)
    snes_linesearch_maxstep = getattr(user_input, "snes_linesearch_maxstep", None)
    snes_linesearch_damping = getattr(user_input, "snes_linesearch_damping", None)
    snes_linesearch_monitor = getattr(user_input, "snes_linesearch_monitor", False)

    if snes_type:               opts["snes_type"]               = snes_type
    if snes_mf:                 opts["snes_mf_operator"]         = None
    if snes_monitor:            opts["snes_monitor"]             = None
    if snes_rtol:               opts["snes_rtol"]                = snes_rtol
    if snes_atol:               opts["snes_atol"]                = snes_atol
    if snes_stol:               opts["snes_stol"]                = snes_stol
    if snes_max_it:             opts["snes_max_it"]              = snes_max_it
    if snes_linesearch_type:    opts["snes_linesearch_type"]     = snes_linesearch_type
    if snes_linesearch_maxstep: opts["snes_linesearch_maxstep"]  = snes_linesearch_maxstep
    if snes_linesearch_damping: opts["snes_linesearch_damping"]  = snes_linesearch_damping
    if snes_linesearch_monitor: opts["snes_linesearch_monitor"]  = None

    # --- Helper functions: enclosure test ---
    def bilinear_interp(h, theta, phi, theta_p, phi_p):
        Nphi  = len(phi)
        phi_p = phi_p % (2*np.pi)

        j1 = np.searchsorted(phi, phi_p)
        j0 = j1 - 1
        if j1 >= Nphi:
            j1 = 0
            j0 = Nphi - 1
        u = (phi_p - phi[j0]) / (phi[j1] - phi[j0])

        h_theta0 = np.interp(theta_p, theta, h[:, j0])
        h_theta1 = np.interp(theta_p, theta, h[:, j1])
        return (1 - u) * h_theta0 + u * h_theta1

    def inside(h, theta, phi, point_xyz):
        x, y, z = point_xyz
        r_p = np.sqrt(x*x + y*y + z*z)
        if r_p > 0.0:
            theta_p = np.arccos(np.clip(z / r_p, -1.0, 1.0))
            phi_p   = np.arctan2(y, x) % (2*np.pi)
        else:
            theta_p = 0.0
            phi_p   = 0.0
        return r_p < bilinear_interp(h, theta, phi, theta_p, phi_p)

    def horizon_encloses_both(h, theta, phi, points_xyz):
        return [inside(h, theta, phi, pt) for pt in points_xyz]

    PETSc.Sys.Print("====================================================")

    # --- Individual horizons ---
    if find_indiv:
        if not blackholes:
            raise ValueError("find_indiv = True but no blackholes list found in input file.")
        PETSc.Sys.Print("Finding individual horizons:")
        for bh in blackholes:
            PETSc.Sys.Print(f"Solving for {bh['name']}.")

            solver_bh = nosym(bh["gammaij"], Kij)

            t_start = time.time()
            hsol, theta_grid, phi_grid = solver_bh.Solver(bh["hguess"], Ntheta_indiv, Nphi_indiv)
            t_elapsed = time.time() - t_start

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
    skip_common = False
    if sep_val is not None and sep_val > 1.0:
        skip_common = True
        PETSc.Sys.Print(f"Skipping common horizon search: separation = {sep_val} > 1.0")

    if not skip_common:
        PETSc.Sys.Print("Finding common/origin-centered horizon.")
        solver_common = nosym(gammaij_global, Kij)
        try:
            t_start = time.time()
            hsol_common, theta_common, phi_common = solver_common.Solver(
                hguess_global, Ntheta_common, Nphi_common)
            t_elapsed = time.time() - t_start

            fname_common = os.path.join(output_dir, "horizon_common.h5")
            with h5py.File(fname_common, 'w') as f:
                f.create_dataset('h', data=hsol_common, compression='gzip', compression_opts=9)
                f.create_dataset('theta', data=theta_common, compression='gzip')
                f.create_dataset('phi', data=phi_common, compression='gzip')
            PETSc.Sys.Print(f"Saved common horizon -> {fname_common}")
            PETSc.Sys.Print(f"  Computation time: {t_elapsed:.4f} seconds")

            if z0_val is not None:
                points      = [(0, 0, -z0_val), (0, 0, +z0_val)]
                enc_results = horizon_encloses_both(hsol_common, theta_common, phi_common, points)
                PETSc.Sys.Print(f"Common horizon encloses punctures? {enc_results}")

        except Exception as e:
            PETSc.Sys.Print(f"ERROR: Common solver raised exception: {e}")

    PETSc.Sys.Print("====================================================")
    PETSc.Sys.Print("Use 'GenSymmetricPlotting.ipynb' for visualisation.")
