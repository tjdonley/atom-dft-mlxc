"""Production vs explicit exchange hole, fast (no SCF) check.

The *production* convolutional functional ``atom.xc.simple_hole.SIMPLE_HOLE``
(fixed monopole operators C_n + precomputed alpha/beta tables + on-top table
inversion) must reproduce the *explicit* direct-integral reference
``atom.xc.simple_hole_explicit.hole_solve`` [Eq. (eps-x)] on the same smooth
density. The only difference is operator/table evaluation vs per-point quadrature;
they compute the same hole self-energy.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atom.descriptors.simple.derivatives import axial_multipole_profile  # noqa: E402
from atom.xc.evaluator import DensityData  # noqa: E402
from atom.xc.simple_hole import SIMPLE_HOLE, SIMPLEHOLEParameters  # noqa: E402
from atom.xc.simple_hole_explicit import hole_solve  # noqa: E402


def test_production_matches_explicit_on_smooth_density():
    """SIMPLE_HOLE (convolutional) eps_x == hole_solve (direct integral) in the bulk."""
    r = np.linspace(0.05, 14.0, 160)
    rho = np.exp(-(r / 3.0) ** 2)                       # smooth, no cusp
    r_c = 8.0

    # explicit reference: direct integral on the l=0 axial profile at each center
    def rho_func(R):                                    # same linear model the operator sees
        return np.interp(np.asarray(R, dtype=float), r, rho, right=0.0)
    idx = np.linspace(20, 140, 16).astype(int)          # interior points
    ex_explicit = np.array([
        hole_solve(lambda u, r0=float(r[i]): axial_multipole_profile(rho_func, r0, u, 0), r_c)[0]
        for i in idx
    ])

    # production: the convolutional functional (gauge off -> raw self-energy)
    w = np.full(r.size, r[1] - r[0])
    params = SIMPLEHOLEParameters(r_c=r_c, n_channels=24, n_zeta=96,
                                  n_window=120, n_angle=48, gauge_fix=False)
    ev = SIMPLE_HOLE(derivative_matrix=np.zeros((1, r.size, 1)), r_quad=r,
                     quadrature_weights=w, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex_prod = ev.compute_xc(DensityData(rho=rho)).e_x[idx]

    ok = np.isfinite(ex_prod) & np.isfinite(ex_explicit)
    assert ok.sum() >= 12
    assert np.corrcoef(ex_prod[ok], ex_explicit[ok])[0, 1] > 0.999
    assert np.max(np.abs(ex_prod[ok] - ex_explicit[ok])) < 5e-3
