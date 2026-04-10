##################################################################
# This is a pytest file to test the solver for finding apparent 
#   horizons in a BBH system written in Fortran. 
# List of tests:
# 1. test_twobh_runs: Check if the Fortran binary runs without errors using a default config.
# 2. test_common_horizon_exists: Check if the expected HDF5 output file for the common horizon is produced.
# 3. test_table_I_area: For each (rBH, grid) combination in Shibata (1997) Table I, 
#       run the solver and compute A/(16piM^2). Compare to Shibata's value at that grid resolution 
#       and check if the error is within the specified tolerance.
# 4. 


# To run the tests, use the command:
# pytest test_ksp_twobh.py -vs
##################################################################

import subprocess
import numpy as np
import h5py
import pytest
from pathlib import Path

_area_results = []
_ksp_results = []

# --- Paths ---
TEST_DIR  = Path(__file__).parent
BIN       = TEST_DIR / "../bin/test_twobh"
DATA_DIR  = TEST_DIR / "data"
H5_COMMON = DATA_DIR / "horizon_common.h5"
CONFIG    = TEST_DIR / "config.nml"

M1, M2   = 1.0, 1.0
M_total  = M1 + M2


# --- Helper: write config.nml ---
def write_config(z0, Ntheta, Nphi, ksp_type="gmres"):
    with open(CONFIG, "w") as f:
        f.write("&twobh_config\n")
        f.write(f"    z0            = {z0}\n")
        f.write(f"    Ntheta_common = {Ntheta}\n")
        f.write(f"    Nphi_common   = {Nphi}\n")
        f.write(f"    ksp_type      = \"{ksp_type}\"\n")
        f.write("/\n")
        f.write("\n")


# --- Helper: compute normalised area from HDF5 ---
def compute_area(h5_path, z0):
    with h5py.File(h5_path, "r") as f:
        h_raw = f["h"][:]
        theta = f["theta"][:]
        phi   = f["phi"][:]

    # Fix transposition if needed
    h = np.array(h_raw, dtype=float)
    if h.shape == (len(phi), len(theta)):
        h = h.T
    phi_var   = np.mean(np.std(h, axis=1))
    theta_var = np.mean(np.std(h, axis=0))
    if phi_var > theta_var:
        h = h.T

    dtheta = theta[1] - theta[0]
    dphi   = phi[1]   - phi[0]
    half   = len(phi) // 2

    T, P = np.meshgrid(theta, phi, indexing='ij')

    X = h * np.sin(T) * np.cos(P)
    Y = h * np.sin(T) * np.sin(P)
    Z = h * np.cos(T)

    eps   = 1e-12
    r1    = np.maximum(np.sqrt(X**2 + Y**2 + (Z + z0)**2), eps)
    r2    = np.maximum(np.sqrt(X**2 + Y**2 + (Z - z0)**2), eps)
    psi_h = 1.0 + M1/(2*r1) + M2/(2*r2)

    h_t = np.zeros_like(h)
    h_p = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2*dphi)
    h_t[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2*dtheta)
    h_t[0,  :]   = (h[1, :]                 - np.roll(h[0,  :], half)) / (2*dtheta)
    h_t[-1, :]   = (np.roll(h[-1, :], half) - h[-2, :])                / (2*dtheta)

    sin_T     = np.sin(T)
    integrand = psi_h**4 * h * np.sqrt(h**2*sin_T**2 + h_t**2*sin_T**2 + h_p**2)

    A_AH = np.sum(integrand) * dtheta * dphi
    return A_AH / (16.0 * np.pi * M_total**2)


# --- Fixture to run tests and write results ---
@pytest.fixture(scope="session", autouse=True)
def write_results_file():
    yield

    out_path = TEST_DIR / "results.txt"
    with open(out_path, "w") as f:

        # KSP benchmark table
        f.write("=" * 68 + "\n")
        f.write("KSP BENCHMARK  (z0=0.5, grid=32x32, 5 rounds)\n")
        f.write("=" * 68 + "\n")
        if _ksp_results:
            f.write(f"{'ksp_type':<12}  {'mean (s)':>10}  {'std (s)':>10}  {'rounds':>7}\n")
            f.write("-" * 48 + "\n")
            for row in _ksp_results:
                f.write(
                    f"{row['ksp_type']:<12}  "
                    f"{row['mean']:>10.4f}  "
                    f"{row['std']:>10.4f}  "
                    f"{row['rounds']:>7}\n"
                )
        else:
            f.write("  (no benchmark data)\n")

        # Area table
        f.write("\n")
        f.write("=" * 68 + "\n")
        f.write("AREA COMPARISON  A/(16piM^2)  vs  Shibata (1997) Table I\n")
        f.write("=" * 68 + "\n")
        f.write(f"{'rBH':>5}  {'z0':>6}  {'grid':<10}  "
                f"{'computed':>10}  {'Shibata':>10}  {'err (%)':>8}  "
                f"{'tol (%)':>8}  {'pass?':>6}\n")
        f.write("-" * 68 + "\n")
        for row in _area_results:
            flag = "PASS" if row["pass"] else "FAIL"
            f.write(
                f"{row['rBH']:>5.2f}  {row['z0']:>6.3f}  {row['grid']:<10}  "
                f"{row['computed']:>10.5f}  {row['shibata']:>10.5f}  "
                f"{row['pct_err']:>8.3f}  {row['tol']:>8.1f}  {flag:>6}\n"
            )

    print(f"\nResults written to {out_path}")


# --- Table I reference values (Shibata 1997) ---
SHIBATA_TABLE_I = {
    1.00: {"32x32": 0.99641, "48x48": 0.99646, "64x64": 0.99648, "100x100": 0.99649},
    1.20: {"32x32": 0.99230, "48x48": 0.99247, "64x64": 0.99253, "100x100": 0.99257},
    1.40: {"32x32": 0.98478, "48x48": 0.98518, "64x64": 0.98532, "100x100": 0.98543},
    1.50: {"32x32": 0.97840, "48x48": 0.97906, "64x64": 0.97930, "100x100": 0.97947},
    1.52: {"32x32": 0.97659, "48x48": 0.97738, "64x64": 0.97766, "100x100": 0.97786},
    1.53: {"32x32": 0.97540, "48x48": 0.97637, "64x64": 0.97670, "100x100": 0.97693},
}

GRID_SIZES = {
    "32x32":   (32,  32),
    "48x48":   (48,  48),
    "64x64":   (64,  64),
    "100x100": (100, 100),
}

TOLERANCE_PCT = {
    "32x32":   0.5,
    "48x48":   0.3,
    "64x64":   0.2,
    "100x100": 0.1,
}

TABLE_I_CASES = [
    (rBH, grid_label)
    for rBH in sorted(SHIBATA_TABLE_I.keys())
    for grid_label in GRID_SIZES
]

#%%
# ===========================================================================
# Tests
# ===========================================================================

def test_twobh_runs():
    """Fortran binary exits cleanly with default config."""
    write_config(z0=0.5, Ntheta=64, Nphi=64)
    result = subprocess.run(
        [str(BIN.resolve()),"./config.nml"],
        cwd=str(TEST_DIR),
        capture_output=True, text=True
    )
    assert result.returncode == 0, \
        f"Binary failed:\n{result.stdout}\n{result.stderr}"


def test_common_horizon_exists():
    """HDF5 output file for common horizon is produced."""
    assert H5_COMMON.exists(), \
        f"Expected output not found: {H5_COMMON}"


#def test_common_horizon_area_z0_0p5():
    """
    Area of common horizon matches Shibata 1997 Table I.
    z0=0.5 -> rBH/m=1.0 -> A/16piM^2=0.99648 (64x64 grid).
    Tolerance: 1%.
    """
#    write_config(z0=0.5, Ntheta=64, Nphi=64)
#    subprocess.run(
#        [str(BIN.resolve()), "./config.nml"],
#        cwd=str(TEST_DIR), check=True,
#        capture_output=True, text=True
#    )

#    A_norm      = compute_area(H5_COMMON, z0=0.5)
#    shibata_ref = 0.99648
#    tol         = 0.01

#    print(f"\nA/(16pi M^2) = {A_norm:.6f}")
#    print(f"Shibata ref  = {shibata_ref}")
#    print(f"Difference   = {abs(A_norm - shibata_ref):.6f}")

#    assert abs(A_norm - shibata_ref) < tol, \
#        f"Area mismatch: got {A_norm:.6f}, expected {shibata_ref} ± {tol}"


@pytest.mark.parametrize("ksp_type", ["gmres", "bcgs", "bcgsl", "tfqmr", "fgmres"])
def test_ksp_type_variations(benchmark, ksp_type):
    write_config(z0=0.5, Ntheta=32, Nphi=32, ksp_type=ksp_type)

    def run_fortran():
        return subprocess.run(
            [str(BIN.resolve()), "./config.nml"],
            cwd=str(TEST_DIR),
            capture_output=True, text=True
        )

    result = benchmark(run_fortran)

    assert result.returncode == 0, \
        f"ksp_type={ksp_type} failed:\n{result.stdout}"

    area    = compute_area(H5_COMMON, z0=0.5)
    ref     = 0.99641
    tol_pct = 0.5
    pct_err = abs(area - ref) / ref * 100.0

    _ksp_results.append({
        "ksp_type": ksp_type,
        "mean":     benchmark.stats["mean"],
        "std":      benchmark.stats["stddev"],
        "rounds":   benchmark.stats["rounds"],
    })

    assert pct_err < tol_pct, \
        f"ksp_type={ksp_type}: area error={pct_err:.3f}% (tolerance {tol_pct}%)"


@pytest.mark.parametrize("rBH,grid_label", TABLE_I_CASES)
def test_table_I_area(rBH, grid_label):
    """
    For every (rBH, grid) in Shibata (1997) Table I, run the solver
    and compare A/(16piM^2) to the published value at that resolution.
    rBH/m is full separation; z0 = rBH/2.
    """
    z0             = rBH / 2.0
    Ntheta, Nphi   = GRID_SIZES[grid_label]
    shibata_ref    = SHIBATA_TABLE_I[rBH][grid_label]
    tol_pct        = TOLERANCE_PCT[grid_label]

    write_config(z0=z0, Ntheta=Ntheta, Nphi=Nphi)
    result = subprocess.run(
        [str(BIN.resolve()), "./config.nml"],
        cwd=str(TEST_DIR),
        capture_output=True, text=True
    )

    assert result.returncode == 0, \
        f"Binary failed for rBH={rBH}, grid={grid_label}:\n{result.stdout}"

    assert H5_COMMON.exists(), \
        f"No output file for rBH={rBH}, grid={grid_label}"

    A_norm  = compute_area(H5_COMMON, z0=z0)
    pct_err = abs(A_norm - shibata_ref) / shibata_ref * 100.0

    #print(
    #    f"\nrBH={rBH:.2f} (z0={z0:.3f}) | grid={grid_label} | "
    #    f"computed={A_norm:.5f} | Shibata={shibata_ref:.5f} | "
    #    f"err={pct_err:.3f}%"
    #)

    _area_results.append({
                            "rBH":      rBH,
                            "z0":       z0,
                            "grid":     grid_label,
                            "computed": A_norm,
                            "shibata":  shibata_ref,
                            "pct_err":  pct_err,
                            "tol":      tol_pct,
                            "pass":     pct_err < tol_pct,
                        })

    assert pct_err < tol_pct, (
        f"rBH={rBH}, grid={grid_label}: area error={pct_err:.3f}% "
        f"(tolerance {tol_pct:.1f}%)"
    )