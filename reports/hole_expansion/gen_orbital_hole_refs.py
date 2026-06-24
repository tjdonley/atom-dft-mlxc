"""Generate exact orbital-exchange-hole reference data for He and Be (s-only closed shells).

Runs an all-electron EXX SCF, reconstructs the exact spherically-averaged exchange hole from
the occupied orbitals (atom.xc.orbital_hole), verifies the integrated E_x against the solver's
``oep_exchange``, and saves orbitals + reference hole coefficients to tests/simple/data/.

Run:  python3 reports/hole_expansion/gen_orbital_hole_refs.py
"""
import os
import numpy as np

from atom import AtomicDFTSolver
from atom.xc import orbital_hole as oh

np.seterr(all="ignore")

R_C = 6.0
N_CHAN = 16
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "simple", "data")


def run(Z, name):
    s = AtomicDFTSolver(atomic_number=Z, xc_functional="EXX", all_electron_flag=True,
                        domain_size=14.0, max_scf_iterations=80)
    res = s.solve()
    assert res["converged"], f"{name}: SCF did not converge"
    ex_ref = float(res["energy_components"].oep_exchange)

    r_sorted, g_sorted, occ = oh.extract_s_orbitals(res)
    Ex_hole = oh.exact_Ex(res, n_u=160, n_mu=80, r0_stride=1)
    print(f"{name} (Z={Z}): solver oep_exchange = {ex_ref:.6f}  | orbital-hole E_x = {Ex_hole:.6f}"
          f"  | diff = {1e3*abs(Ex_hole-ex_ref):.3f} mHa")

    # reference hole coefficients at a spread of r0 (skip the outermost low-density tail)
    r = np.asarray(res["quadrature_nodes"], float)
    rho = np.asarray(res["rho"], float)
    mask = rho > 1e-6 * rho.max()
    r0_grid = np.unique(np.clip(np.linspace(r[mask].min() + 1e-3, min(r[mask].max(), 8.0), 40),
                                1e-3, None))
    coeffs = np.array([oh.project_exact_hole(r0, r_sorted, g_sorted, occ, R_C, N_CHAN) for r0 in r0_grid])
    eps_exact = np.array([oh.exact_eps_x(r0, r_sorted, g_sorted, occ, n_u=160, n_mu=80) for r0 in r0_grid])
    rho_r0 = np.array([oh.on_top_density(r0, r_sorted, g_sorted, occ) for r0 in r0_grid])

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"orbital_hole_{name}.npz")
    np.savez(out, Z=Z, r_sorted=r_sorted, g_sorted=g_sorted, occ=occ,
             oep_exchange=ex_ref, Ex_hole=Ex_hole, r0_grid=r0_grid, rho_r0=rho_r0,
             rhotilde_exact=coeffs, eps_exact=eps_exact, r_c=R_C, n_channels=N_CHAN)
    print(f"  saved {out}")
    return ex_ref, Ex_hole


if __name__ == "__main__":
    for Z, name in [(2, "He"), (4, "Be")]:
        run(Z, name)
