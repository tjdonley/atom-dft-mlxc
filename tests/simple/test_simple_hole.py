"""Self-consistent convolutional exchange-hole functional (atom/xc/simple_hole.py).

  * the exchange potential (operator-transpose / Sahoo adjoint over the fixed
    monopole operators) equals the finite-difference derivative of the discrete
    exchange energy -- the core correctness test, the same check the ladder
    functionals pass in test_simple_xc.py;
  * the functional runs self-consistently: a full SCF with SIMPLE_HOLE converges
    on a pseudopotential atom (the production setting), and its converged exchange
    energy matches the independent direct-integral proof of concept on the same
    density (so the convolutional SCF functional reproduces the hole model).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atom.scf.density import DensityData  # noqa: E402
from atom.xc.simple_hole import (  # noqa: E402
    SIMPLE_HOLE,
    SIMPLEHOLEParameters,
)

_DATA = _REPO_ROOT / "tests" / "simple" / "data" / "n_atom_Z7_pbe_psp8.npz"


def test_hole_potential_matches_fd():
    """Operator-transpose (Sahoo) v_x == d E_x / d rho (finite difference)."""
    r = np.linspace(0.05, 11.0, 130)
    w = np.full(r.size, r[1] - r[0])
    # gauge_fix=False: the gauge/damping is a separate energy-neutral post-step;
    # the pure discrete adjoint [Eq. (adjoint-discrete)] is what equals dE_x/drho.
    params = SIMPLEHOLEParameters(r_c=8.0, n_channels=12, n_zeta=40, n_window=80,
                                  n_angle=32, gauge_fix=False)
    ev = SIMPLE_HOLE(derivative_matrix=np.zeros((1, r.size, 1)), r_quad=r,
                     quadrature_weights=w, params=params)
    ew = ev.energy_weights

    def E_x(rho):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.sum(ew * rho * ev.compute_xc(DensityData(rho=rho)).e_x))

    rho = np.exp(-(r / 3.0) ** 2) + 0.2 * np.exp(-(r / 6.0) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ev.compute_xc(DensityData(rho=rho))
    eps = 1e-6
    probe = np.arange(8, r.size - 8, 11)
    fd = np.array([
        (E_x(rho + eps * (np.arange(r.size) == j)) - E_x(rho - eps * (np.arange(r.size) == j))) / (2 * eps)
        for j in probe
    ])
    target = (ew * out.v_x)[probe]
    resid = np.abs(fd - target).max() / np.abs(target).max()
    assert resid < 5e-6, resid
    assert np.all(np.isfinite(out.e_x)) and np.all(np.isfinite(out.v_x))


def _scf(z, params, **kw):
    from atom.solver import AtomicDFTSolver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AtomicDFTSolver(atomic_number=z, xc_functional="SIMPLE_HOLE", all_electron_flag=False,
                               verbose=False, xc_params=params, use_pulay_mixing=True,
                               linear_mixing_alpha1=0.3, max_scf_iterations=300, **kw).solve(
                                   save_energy_density=True)


def test_simple_hole_scf_converges_psp():
    """SIMPLE_HOLE runs self-consistently: a full SCF on PSP Be converges, and the
    converged exchange energy matches the direct-integral hole on the same density
    (the convolutional SCF functional reproduces the proof-of-concept hole)."""
    from atom.xc.simple_hole_explicit import hole_solve
    from atom.descriptors.simple.derivatives import axial_multipole_profile
    from scipy.interpolate import CubicSpline

    params = SIMPLEHOLEParameters(r_c=8.0, n_channels=20, n_zeta=48, n_window=100, n_angle=40)
    res = _scf(4, params)
    assert res["converged"], "SIMPLE_HOLE SCF did not converge for PSP Be"

    o = np.argsort(np.asarray(res["quadrature_nodes"]))
    r = np.asarray(res["quadrature_nodes"])[o]
    w = np.asarray(res["quadrature_weights"])[o]
    rho = np.maximum(np.asarray(res["rho"])[o], 1e-12)
    ew = 4.0 * np.pi * r ** 2 * w
    # the solver stores e_x_local as the per-volume density rho*eps_x, so the
    # exchange energy is sum(ew * e_x_local) (no extra rho), matching the OEP demo
    Ex_scf = float(np.sum(ew * np.asarray(res["e_x_local"])[o]))

    # direct-integral hole on the SAME (SCF) density
    spl = CubicSpline(r, rho); lo, hi = r[0], r[-1]
    def rf(R):
        R = np.asarray(R, float)
        v = np.where(R > hi, 0.0, spl(np.clip(R, lo, hi)))
        return np.maximum(np.where(R < lo, rho[0], v), 0.0)
    bulk = np.where((rho > 1e-4) & (r > 0.05))[0]
    idx = bulk[np.linspace(0, bulk.size - 1, 40).astype(int)]
    ex = np.array([hole_solve(lambda u, r0=float(r[i]): axial_multipole_profile(rf, r0, u, 0), 8.0)[0]
                   for i in idx])
    ok = np.isfinite(ex)
    Ex_poc = float(np.sum(ew * rho * np.interp(r, r[idx][ok], ex[ok])))
    assert abs(Ex_scf - Ex_poc) * 1000.0 < 25.0, (Ex_scf, Ex_poc)
