"""
Horizon Solver — No Symmetry (Linearized KSP)
===============================================
Solves the linearized apparent horizon equation using a fixed-point
iteration with PETSc KSP as the inner linear solver.

Based on Shibata (2000), Eq. 2.1:
    L[h] = h_tt + cot(t)*h_t + h_pp/sin^2(t) - 2h = S(h, psi, Kij)

IMPORTANT: This solver assumes a conformally flat metric.
           The input must provide psi (conformal factor), NOT gammaij.

Supports:
  - Single / common horizon search
  - Individual horizon search per black hole (set find_indiv = True)

Usage:
    python nosym_ksp.py <input_file>
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


#%%     NOSYMMETRY — KSP (Linearized solver)
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
        dtheta  = np.pi / Ntheta
        dphi    = 2.0 * np.pi / Nphi
        inv_dt2 = 1.0 / (dtheta**2)
        inv_dp2 = 1.0 / (dphi**2)
        half    = Nphi // 2

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
            phi_c = inv_dp2 / (s * s)
            cot_t = c / s
            a_tp  = inv_dt2 + 0.5 * cot_t / dtheta
            a_tm  = inv_dt2 - 0.5 * cot_t / dtheta
            a_0   = -2.0 * inv_dt2 - 2.0 * phi_c - 2.0

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
                    cols.append(1 * Nphi + i)
                    vals.append(a_tp)
                    cols.append(0 * Nphi + (i + half) % Nphi)
                    vals.append(a_tm)
                elif j == Ntheta - 1:
                    cols.append((j - 1) * Nphi + i)
                    vals.append(a_tm)
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

    # ----- Build grid and precompute trig factors ------
    def grid(self, Ntheta, Nphi):

        self.Ntheta = Ntheta
        self.Nphi   = Nphi
        self.dtheta = np.pi / Ntheta
        self.dphi   = 2.0 * np.pi / Nphi

        theta = (np.arange(Ntheta) + 0.5) * self.dtheta
        phi   = (np.arange(Nphi) + 0.5) * self.dphi

        self.t, self.p = np.meshgrid(theta, phi, indexing="ij")

        self.sin_t = np.sin(self.t)
        self.cos_t = np.cos(self.t)
        self.cot_t = self.cos_t / self.sin_t

        return self.dtheta, self.dphi

    # ----- Finite-difference derivatives of h on the grid ------
    def h_derivatives(self, h):

        Ntheta = self.Ntheta
        Nphi   = self.Nphi
        dtheta = self.dtheta
        dphi   = self.dphi
        half   = Nphi // 2

        h2  = h.reshape(Ntheta, Nphi)
        ht  = np.zeros_like(h2)
        hp  = np.zeros_like(h2)
        htt = np.zeros_like(h2)
        hpp = np.zeros_like(h2)
        htp = np.zeros_like(h2)

        # phi derivatives (periodic)
        hp  = (np.roll(h2, -1, axis=1) - np.roll(h2, 1, axis=1)) / (2*dphi)
        hpp = (np.roll(h2, -1, axis=1) - 2*h2 + np.roll(h2, 1, axis=1)) / (dphi**2)

        # theta derivatives interior
        ht[1:-1,:]  = (h2[2:,:] - h2[:-2,:]) / (2*dtheta)
        htt[1:-1,:] = (h2[2:,:] - 2*h2[1:-1,:] + h2[:-2,:]) / (dtheta**2)

        # north pole
        h_shift    = np.roll(h2[0,:], half)
        ht[0,:]    = (h2[1,:] - h_shift) / (2*dtheta)
        htt[0,:]   = (h2[1,:] - 2*h2[0,:] + h_shift) / (dtheta**2)

        # south pole
        h_shift    = np.roll(h2[-1,:], half)
        ht[-1,:]   = (h_shift - h2[-2,:]) / (2*dtheta)
        htt[-1,:]  = (h_shift - 2*h2[-1,:] + h2[-2,:]) / (dtheta**2)

        # mixed derivative interior
        htp[1:-1,:] = (np.roll(h2[2:,:], -1, axis=1) - np.roll(h2[2:,:], 1, axis=1)
                       - np.roll(h2[:-2,:], -1, axis=1) + np.roll(h2[:-2,:], 1, axis=1)) / (4*dtheta*dphi)

        # mixed derivative north pole
        h_jp     = h2[1,:]
        h_jm     = np.roll(h2[0,:], half)
        htp[0,:] = (np.roll(h_jp, -1) - np.roll(h_jp, 1)
                    - np.roll(h_jm, -1) + np.roll(h_jm, 1)) / (4*dtheta*dphi)

        # mixed derivative south pole
        h_jp      = np.roll(h2[-1,:], half)
        h_jm      = h2[-2,:]
        htp[-1,:] = (np.roll(h_jp, -1) - np.roll(h_jp, 1)
                     - np.roll(h_jm, -1) + np.roll(h_jm, 1)) / (4*dtheta*dphi)

        return ht, hp, htt, hpp, htp

    # ----- Compute psi and its derivatives on the grid ------
    def psi_derivatives(self, h):

        r      = h.reshape(self.Ntheta, self.Nphi)
        t      = self.t
        p      = self.p
        dtheta = self.dtheta
        dphi   = self.dphi
        dr     = 1e-6

        psi   = self.psi(r, t, p)
        psi_r = (self.psi(r+dr, t, p) - self.psi(r-dr, t, p)) / (2*dr)
        psi_t = (self.psi(r, t+dtheta, p) - self.psi(r, t-dtheta, p)) / (2*dtheta)
        psi_p = (self.psi(r, t, p+dphi) - self.psi(r, t, p-dphi)) / (2*dphi)

        return psi, psi_r, psi_t, psi_p

    # ------ Compute the nonlinear source term S(h, psi, Kij) ------
    def source_term(self, h):

        h2    = h.reshape(self.Ntheta, self.Nphi)
        sin_t = self.sin_t
        cos_t = self.cos_t
        cot_t = self.cot_t
        sin2  = sin_t**2

        ht, hp, htt, hpp, htp       = self.h_derivatives(h)
        psi, psi_r, psi_t, psi_p   = self.psi_derivatives(h)

        C   = np.sqrt(h2**2 + ht**2 + hp**2 / sin2)
        s_r = h2 / C
        s_t = -ht / C
        s_p = -hp / (C * sin2)

        Kij  = self.Kij(h2, self.t, self.p)
        Kss  = (Kij[0,0]*s_r*s_r + 2*Kij[0,1]*s_r*s_t + 2*Kij[0,2]*s_r*s_p
                + Kij[1,1]*s_t*s_t + 2*Kij[1,2]*s_t*s_p + Kij[2,2]*s_p*s_p)

        psi4     = psi**4
        gamma_rr = 1/psi4
        gamma_tt = 1/(psi4 * h2**2)
        gamma_pp = 1/(psi4 * h2**2 * sin2)
        Ktrace   = gamma_rr*Kij[0,0] + gamma_tt*Kij[1,1] + gamma_pp*Kij[2,2]

        termK  = (psi**2 * h2**2 / C**3) * (Kss - Ktrace)
        term2  = (4/psi) * (psi_r - psi_t * ht / h2**2
                            - psi_p * hp / (h2**2 * sin2)) * (h2**2 + ht**2 + hp**2 / sin2)
        term3  = (3/h2) * (ht**2 + hp**2 / sin2)
        term4  = (1/(h2**2 * sin2)) * (2*ht*hp*htp - cot_t * hp**2 * ht)
        term5  = -(ht**2 / (h2**2 * sin2)) * (sin_t*cos_t*ht + hpp)
        term6  = -(hp**2 / (h2**2 * sin2)) * (htt + cot_t*ht)

        return termK + term2 + term3 + term4 + term5 + term6

    # ----- Build RHS vector B from source term S(h) ------
    def build_B(self, h_old):
        S = self.source_term(h_old)
        B = PETSc.Vec().createSeq(self.Ntheta * self.Nphi)
        B.setArray(S.flatten())
        return B

    # ----- The KSP solver by PETSc -------
    def Solver(self, hguess, Ntheta, Nphi, omega, max_iter):

        dtheta, dphi = self.grid(Ntheta, Nphi)

        A, _, _ = self.build_A(Ntheta, Nphi)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setFromOptions()

        h_old = np.array(hguess(self.t, self.p))

        tol       = 1e-8
        diff      = 1.0
        it        = 0
        h_new_vec = PETSc.Vec().createSeq(Ntheta * Nphi)

        while diff > tol and it < max_iter:

            B = self.build_B(h_old)

            ksp.solve(B, h_new_vec)
            if ksp.getConvergedReason() < 0:
                raise RuntimeError("KSP failed to converge")

            h_new = h_new_vec.getArray().reshape(Ntheta, Nphi)
            diff  = np.linalg.norm(h_new - h_old)

            PETSc.Sys.Print(f"Iter {it}   diff = {diff:e}")

            if not np.isfinite(diff) or diff > 1e6:
                PETSc.Sys.Print("Solver diverged — horizon not found.")
                return None, self.t[:,0], self.p[0,:]

            if diff < tol:
                break

            h_old = (1 - omega)*h_old + omega*h_new
            it += 1

        PETSc.Sys.Print("Solver finished.")
        return h_new, self.t[:,0], self.p[0,:]


#%%     DRIVER
###############################################################################
if __name__ == "__main__":

    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: python nosym_ksp.py <input_file>")
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
    PETSc.Sys.Print("Symmetry type: General (no symmetry) — KSP linearized solver")

    # --- Mandatory input ---
    psi = getattr(user_input, "psi", None)
    if psi is None:
        raise ValueError("psi (conformal factor) missing in input file. "
                         "nosym_ksp requires psi, not gammaij.")

    Kij = getattr(user_input, "Kij", None)
    if Kij is None:
        raise ValueError("Kij missing in input file.")

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
    omega         = getattr(user_input, "omega", 0.3)
    max_iter      = getattr(user_input, "max_iter", 500)

    os.makedirs(output_dir, exist_ok=True)

    # --- PETSc KSP options ---
    opts                 = PETSc.Options()
    ksp_type             = getattr(user_input, "ksp_type", None)
    pc_type              = getattr(user_input, "pc_type", None)
    ksp_rtol             = getattr(user_input, "ksp_rtol", None)
    ksp_atol             = getattr(user_input, "ksp_atol", None)
    ksp_max_it           = getattr(user_input, "ksp_max_it", None)
    ksp_monitor          = getattr(user_input, "ksp_monitor", False)
    ksp_converged_reason = getattr(user_input, "ksp_converged_reason", False)

    if ksp_type:             opts["ksp_type"]              = ksp_type
    if pc_type:              opts["pc_type"]               = pc_type
    if ksp_rtol:             opts["ksp_rtol"]              = ksp_rtol
    if ksp_atol:             opts["ksp_atol"]              = ksp_atol
    if ksp_max_it:           opts["ksp_max_it"]            = ksp_max_it
    if ksp_monitor:          opts["ksp_monitor"]           = ksp_monitor
    if ksp_converged_reason: opts["ksp_converged_reason"]  = ksp_converged_reason

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

            solver_bh = nosym(bh["psi"], Kij)

            t_start = time.time()
            hsol, theta_grid, phi_grid = solver_bh.Solver(
                bh["hguess"], Ntheta_indiv, Nphi_indiv, omega, max_iter)
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
        solver_common = nosym(psi, Kij)
        try:
            t_start = time.time()
            hsol_common, theta_common, phi_common = solver_common.Solver(
                hguess_global, Ntheta_common, Nphi_common, omega, max_iter)
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
