"""Reduced gradient s(r), reduced Laplacian q(r), and the re-summed SIMPLE exchange
enhancement F_x(r) = eps_x/eps_x^unif along a few representative atoms (He, Be, N,
Ne), stored to data/atom_sqfx.npz. Each atom is run self-consistently with the
(exchange-only) bare SIMPLE hole to get a converged smooth pseudopotential density;
the spectral operators [Eq. (sq)] give s, q, and SIMPLE_HOLE_GEA gives the re-summed
F_x (forward energy density only -- no adjoint needed for the trajectory). A
Fermi-Amaldi flag marks the compact-core points where Q_S(zeta_min)<=2 (deformation
inactive); the F_x(s) overlay uses the normal (non-FA) regime.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from atom.solver import AtomicDFTSolver  # noqa: E402
from atom.descriptors.simple.derivatives import (  # noqa: E402
    reduced_gradient_from_grad,
    reduced_laplacian_from_grad,
)
from atom.xc.simple_hole import SIMPLE_HOLE_GEA, SIMPLEHOLEGEAParameters  # noqa: E402

_DATA = Path(__file__).resolve().parent / "data" / "atom_sqfx.npz"
ATOMS = [("He", 2), ("Be", 4), ("N", 7), ("Ne", 10)]


def trajectory(Z):
    res = AtomicDFTSolver(atomic_number=Z, xc_functional="SIMPLE_HOLE",
                          all_electron_flag=False, verbose=False,
                          use_pulay_mixing=True, max_scf_iterations=400).solve()
    if not res["converged"]:
        return None
    o = np.argsort(np.asarray(res["quadrature_nodes"]))
    r = np.asarray(res["quadrature_nodes"])[o]
    w = np.asarray(res["quadrature_weights"])[o]
    rho = np.maximum(np.asarray(res["rho"])[o], 1e-12)
    gea = SIMPLE_HOLE_GEA(derivative_matrix=np.zeros((1, r.size, 1)), r_quad=r,
                          quadrature_weights=w, params=SIMPLEHOLEGEAParameters(gauge_fix=False))
    C = np.array([op @ rho for op in gea._ops])
    g = gea._grad_op @ rho
    lap = gea._lap_op @ rho
    s = np.abs(reduced_gradient_from_grad(g, rho))
    q = reduced_laplacian_from_grad(lap, rho)
    c, _, _, _ = gea._amplitude(rho, g, lap)
    eps = gea._eps_def(C, c)                                  # forward energy density only
    Fx = eps / (-0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0))
    fa = (gea._alpha @ C)[0] <= 2.0                           # Fermi-Amaldi (compact-core) flag
    return dict(r=r, rho=rho, s=s, q=q, Fx=Fx, fa=fa)


def main():
    out = {}
    for sym, Z in ATOMS:
        t = trajectory(Z)
        if t is None:
            print(f"{sym}: SCF did not converge -- skipped")
            continue
        for k, v in t.items():
            out[f"{sym}_{k}"] = v
        m = (t["rho"] > 1e-2 * t["rho"].max()) & (~t["fa"])
        print(f"{sym}: normal-regime s in [{t['s'][m].min():.2f},{t['s'][m].max():.2f}], "
              f"F_x in [{t['Fx'][m].min():.2f},{t['Fx'][m].max():.2f}]")
    _DATA.parent.mkdir(exist_ok=True)
    np.savez(_DATA, atoms=np.array([s for s, _ in ATOMS]), **out)
    print(f"stored {_DATA.name}")


if __name__ == "__main__":
    main()
