##################################################################
# This is a pytest file to test the solver for finding apparent 
# horizons in a BBH system (nosym_ksp.py). 
# List of tests:
# 1. test_solver_converges: Check if the solver converges for a
#    simple case of two black holes at z = +/- 0.5.
# 2. test_ksp_type_variations: Benchmark different KSP types (gmres, bcgs, etc.)
#    for the same problem and record their performance.
# 3. test_table_I_area: For each (rBH, grid) combination in Shibata (1997) Table I, 
#    run the solver and compute A/(16piM^2). Compare to Shibata's value at that grid 
#    resolution and check if the error is within the specified tolerance.

# To run the tests, use the command:
# pytest test_ksp_twobh.py -v
##################################################################

import os
import sys
sys.path.insert(0, "./../src")

from petsc4py import PETSc
import petsc4py
petsc4py.init(sys.argv)

from nosym_ksp import nosym
import numpy as np
import pytest 


### Physical parameters ###
M1 = 1.0
M2 = 1.0
M_total = M1 + M2


### 2BH system parameters ###
def make_psi(z0):
    def psi(r, theta, phi):
        epsilon = 1e-12
        r = np.maximum(r, epsilon)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        r1 = np.sqrt(x**2 + y**2 + (z + z0)**2)
        r2 = np.sqrt(x**2 + y**2 + (z - z0)**2)

        r1 = np.maximum(r1, epsilon)
        r2 = np.maximum(r2, epsilon)

        return 1 + M1/(2*r1) + M2/(2*r2)
    return psi

def Kij(r, theta=None, phi=None):
    return np.zeros((3, 3))

def hguess(theta, phi):
    return 2.0 * np.ones_like(theta)


### Diagnostic function to compute the area of the apparent horizon ###
def compute_area_norm(h_c, theta_c, phi_c, z0):
    """
    Returns A / (16 π M_total²).
    h_c      : (Ntheta, Nphi) horizon surface
    theta_c  : (Ntheta,) grid
    phi_c    : (Nphi,)   grid
    z0       : distance of each BH from origin  (= rBH_shibata / 2)
    """
    T, P = np.meshgrid(theta_c, phi_c, indexing='ij')
 
    X = h_c * np.sin(T) * np.cos(P)
    Y = h_c * np.sin(T) * np.sin(P)
    Z = h_c * np.cos(T)
 
    eps = 1e-12
    r1 = np.maximum(np.sqrt(X**2 + Y**2 + (Z + z0)**2), eps)
    r2 = np.maximum(np.sqrt(X**2 + Y**2 + (Z - z0)**2), eps)
    psi_h = 1.0 + M1 / (2 * r1) + M2 / (2 * r2)
 
    dtheta = theta_c[1] - theta_c[0]
    dphi   = phi_c[1]   - phi_c[0]
    half   = len(phi_c) // 2
 
    h_t = np.zeros_like(h_c)
    h_p = np.zeros_like(h_c)
 
    # phi derivative (periodic)
    h_p = (np.roll(h_c, -1, axis=1) - np.roll(h_c, 1, axis=1)) / (2 * dphi)
 
    # theta derivative (interior)
    h_t[1:-1, :] = (h_c[2:, :] - h_c[:-2, :]) / (2 * dtheta)
    # pole BCs (Shibata 2000, Eq. 2.7–2.8)
    h_t[0,  :] = (h_c[1, :]                 - np.roll(h_c[0,  :], half)) / (2 * dtheta)
    h_t[-1, :] = (np.roll(h_c[-1, :], half) - h_c[-2, :])                / (2 * dtheta)
 
    sin_T = np.sin(T)
    integrand = (psi_h**4 * h_c
                 * np.sqrt(h_c**2 * sin_T**2
                            + h_t**2 * sin_T**2
                            + h_p**2))
 
    A_AH = np.sum(integrand) * dtheta * dphi
    return A_AH / (16 * np.pi * M_total**2)


### Shared storage for results.txt ###
_area_results = []   # populated by test_table_I_area
_bench_results = []

@pytest.fixture(scope="session", autouse=True)
def write_results_file():
    """Writes results.txt after the full test session finishes."""
    yield  # all tests run here
 
    out_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(out_path, "w") as f:
 
        # Benchmark table ─
        f.write("=" * 68 + "\n")
        f.write("KSP BENCHMARK  (grid 32x32, z0=0.5)\n")
        f.write("=" * 68 + "\n")

        if _bench_results:
            f.write(f"{'KSP type':<12}  {'mean (s)':>10}  {'std (s)':>10}  {'rounds':>7}\n")
            f.write("-" * 48 + "\n")
            for row in _bench_results:
                f.write(
                    f"{row['ksp_type']:<12}  "
                    f"{row['mean']:>10.4f}  "
                    f"{row['stddev']:>10.4f}  "
                    f"{row['rounds']:>7}\n"
                )
        else:
            f.write("  (no benchmark data)\n")
 
        # Area table 
        f.write("=" * 68 + "\n")
        f.write("AREA COMPARISON  A/(16piM^2)  vs  Shibata (1997) Table I\n")
        f.write("=" * 68 + "\n")
        f.write(f"{'rBH':>5}  {'z0':>6}  {'grid':<10}  "
                f"{'computed':>10}  {'Shibata':>10}  {'err (%)':>8}  {'tol (%)':>8}  {'pass?':>6}\n")
        f.write("-" * 68 + "\n")
        for row in _area_results:
            flag = "PASS" if row["pass"] else "FAIL"
            f.write(
                f"{row['rBH']:>5.2f}  {row['z0']:>6.3f}  {row['grid']:<10}  "
                f"{row['computed']:>10.5f}  {row['shibata']:>10.5f}  "
                f"{row['pct_err']:>8.3f}  {row['tol']:>8.1f}  {flag:>6}\n"
            )
 
    print(f"\nResults written to {out_path}")


### Table I reference values (Shibata 1997) ###
# Shibata's rBH/m is the full separation between the two BHs.
# Our z0 = rBH_shibata / 2  (distance of each BH from the origin).
#
# Layout: { rBH_shibata: { grid_label: A/(16πM²) } }
SHIBATA_TABLE_I = {
    1.53: {"32x32": 0.97540, "48x48": 0.97637, "64x64": 0.97670, "100x100": 0.97693},
    1.52: {"32x32": 0.97659, "48x48": 0.97738, "64x64": 0.97766, "100x100": 0.97786},
    1.50: {"32x32": 0.97840, "48x48": 0.97906, "64x64": 0.97930, "100x100": 0.97947},
    1.40: {"32x32": 0.98478, "48x48": 0.98518, "64x64": 0.98532, "100x100": 0.98543},
    1.20: {"32x32": 0.99230, "48x48": 0.99247, "64x64": 0.99253, "100x100": 0.99257},
    1.00: {"32x32": 0.99641, "48x48": 0.99646, "64x64": 0.99648, "100x100": 0.99649},
}
 
# Map label -> (Ntheta, Nphi)
GRID_SIZES = {
    "32x32":   (32,  32),
    "48x48":   (48,  48),
    "64x64":   (64,  64),
    "100x100": (100, 100),
}
 
# Tolerance per grid column (% relative error allowed vs Shibata's value)
TOLERANCE_PCT = {
    "32x32":   0.5,
    "48x48":   0.3,
    "64x64":   0.2,
    "100x100": 0.1,
}
 
# Build the full parametrize list: (rBH_shibata, grid_label)
TABLE_I_CASES = [
    (rBH, grid_label)
    for rBH in sorted(SHIBATA_TABLE_I.keys())
    for grid_label in GRID_SIZES
]



#%%
#####################################
### TESTS ###
#####################################


### Convergence test ###
def test_solver_converges():
    solver = nosym(make_psi(0.5), Kij)
    h, theta, phi = solver.Solver(hguess, 32, 32, omega=0.3, max_iter=5000)
    assert h is not None, "Solver did not converge for z0=0.5"


### Different KSP types and speed test ###
@pytest.mark.parametrize("ksp_type", ["gmres", "bcgs", "bcgsl", "tfqmr", "fgmres"])
def test_ksp_type_variations(benchmark, ksp_type):
    opts = PETSc.Options()
    opts["ksp_type"] = ksp_type
    opts["pc_type"]  = "ilu"

    solver = nosym(make_psi(0.5), Kij)

    def run_solver():
        return solver.Solver(hguess, 32, 32, omega=0.3, max_iter=5000)

    h, theta, phi = benchmark(run_solver)
    assert h is not None, f"Solver did not converge with ksp_type={ksp_type}"

    _bench_results.append({
        "ksp_type": ksp_type,
        "mean":     benchmark.stats["mean"],
        "stddev":   benchmark.stats["stddev"],
        "rounds":   benchmark.stats["rounds"],
    })


### Area calculation test against Shibata (1997) Table I ###
@pytest.mark.parametrize("rBH,grid_label", TABLE_I_CASES)
def test_table_I_area(rBH, grid_label):
    """
    For every (rBH, grid_size) combination in Shibata (1997) Table I,
    run the solver and compare A/(16πM²) to the published value at that
    exact grid resolution.
 
    Shibata's rBH/m is the full BH separation; our z0 = rBH / 2.
    """
    z0 = rBH / 2.0
    Ntheta, Nphi = GRID_SIZES[grid_label]
    ref          = SHIBATA_TABLE_I[rBH][grid_label]
    tol_pct      = TOLERANCE_PCT[grid_label]
 
    solver = nosym(make_psi(z0), Kij)
    h, theta, phi = solver.Solver(hguess, Ntheta, Nphi, omega=0.3, max_iter=5000)
 
    assert h is not None, f"Solver did not converge for rBH={rBH}, grid={grid_label}"
 
    A_norm  = compute_area_norm(h, theta, phi, z0)
    pct_err = abs(A_norm - ref) / ref * 100.0

    _area_results.append({
        "rBH":      rBH,
        "z0":       z0,
        "grid":     grid_label,
        "computed": A_norm,
        "shibata":  ref,
        "pct_err":  pct_err,
        "tol":      tol_pct,
        "pass":     pct_err < tol_pct,
    })
 
    print(
        f"\nrBH={rBH:.2f} (z0={z0:.3f}) | grid={grid_label} | "
        f"computed={A_norm:.5f} | Shibata={ref:.5f} | err={pct_err:.3f}%"
    )
 
    assert pct_err < tol_pct, (
        f"rBH={rBH}, grid={grid_label}: area error = {pct_err:.3f}% "
        f"(tolerance {tol_pct:.1f}%)"
    )