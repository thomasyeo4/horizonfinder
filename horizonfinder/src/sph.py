"""
Horizon Solver — Spherical Symmetry
====================================
Provide psi (conformal factor) and Kij (extrinsic curvature)
and the solver will find the apparent horizon radius.

Usage:
    python sph.py <input_file>
"""

import importlib.util
import petsc4py
import sys

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


#%%     DRIVER
###############################################################################
if __name__ == "__main__":

    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: python sph.py <input_file>")
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

    # --- Mandatory input ---
    psi = getattr(user_input, "psi", None)
    if psi is None:
        raise ValueError("psi missing in input file.")

    Kij = getattr(user_input, "Kij", None)
    if Kij is None:
        raise ValueError("Kij missing in input file.")

    rguess = getattr(user_input, "rguess", None)
    if rguess is None:
        raise ValueError("rguess missing in input file.")

    # --- PETSc SNES options ---
    opts = PETSc.Options()
    snes_type    = getattr(user_input, "snes_type", None)
    snes_mf      = getattr(user_input, "snes_mf", None)
    snes_monitor = getattr(user_input, "snes_monitor", False)
    snes_rtol    = getattr(user_input, "snes_rtol", None)
    snes_atol    = getattr(user_input, "snes_atol", None)
    snes_stol    = getattr(user_input, "snes_stol", None)
    snes_max_it  = getattr(user_input, "snes_max_it", None)

    if snes_type:    opts["snes_type"]         = snes_type
    if snes_mf:      opts["snes_mf_operator"]  = None
    if snes_monitor: opts["snes_monitor"]       = None
    if snes_rtol:    opts["snes_rtol"]          = snes_rtol
    if snes_atol:    opts["snes_atol"]          = snes_atol
    if snes_stol:    opts["snes_stol"]          = snes_stol
    if snes_max_it:  opts["snes_max_it"]        = snes_max_it

    # --- Solve ---
    solver   = spheresym(psi, Kij)
    rhorizon = solver.Solver(rguess=rguess)
    PETSc.Sys.Print("====================================================")
    PETSc.Sys.Print("Job done! Horizon radius:", rhorizon)
