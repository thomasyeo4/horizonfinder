"""
This is the source code of the Horizon Solver
The __init__.py is in the same folder to make the classes in this file
callable for diagnostics etc. 

"""

import importlib.util
import petsc4py
import sys
import os 

petsc4py.init(sys.argv)

from petsc4py import PETSc
from mpi4py import MPI
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
        Kij = self.Kij(r)
        psi = self.psi(r)
        
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


#%%     DRIVER
###############################################################################
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        PETSc.Sys.Print("Usage: mpirun -n 1 python <path_to>/HorizonFinder.py <input_file>")
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
    use_multigrid            = getattr(user_input, "use_multigrid", None)
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
    if use_multigrid:
        opts["ksp_type"] = "gmres"
        opts["pc_type"] = "mg"
        # Future multigrid-specific options can also be added here

    
        
    
    # ---- Symmetry type ----
    if symmetry == "spheresym":
        psi = getattr(user_input, "psi", None)
        if psi is None: 
            raise ValueError("psi missing in input file.")
            
        solver   = spheresym(psi, Kij)
        rhorizon = solver.Solver(rguess=rguess)
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Job done! Horizon radius:", rhorizon)
        
    
    elif symmetry in ["axisym", "nosym"]:
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
        
                # Call solver
                hsol, theta_grid = solver_bh.Solver(bh["hguess"], Ntheta_indiv)
        
                # Save the data
                outfile = os.path.join(output_dir, f"{bh['name']}_horizon.csv")
                np.savetxt(outfile, np.column_stack((theta_grid, hsol)),
                           delimiter=",", header="theta,h", comments="")
                PETSc.Sys.Print(f"Individual horizon for {bh['name']} saved to {outfile}")
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
                # Call solver (serial)
                hsol_common, theta_common = solver_common.Solver(hguess_global, Ntheta_common)
        
                fname_common = os.path.join(output_dir, "horizon_common.csv")
                np.savetxt(fname_common, np.column_stack((theta_common, hsol_common)),
                           delimiter=",", header="theta,h", comments="")
                PETSc.Sys.Print(f"Saved common horizon -> {fname_common}")
                results["common"] = (hsol_common, theta_common)
        
                z0_val = getattr(user_input, "z0", None)
                if z0_val is not None:
                    enc1, enc2 = horizon_encloses_both(hsol_common, theta_common, z0_val)
                    PETSc.Sys.Print(f"Common horizon encloses punctures? BH1: {enc1}, BH2: {enc2}")
        
            except Exception as e:
                PETSc.Sys.Print(f"  ERROR: Common solver raised exception: {e}")
        
        PETSc.Sys.Print("====================================================")
        PETSc.Sys.Print("Use 'AxisymmetricPlotting.ipynb' for visualisation.")
        
                
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry}")
            
    
