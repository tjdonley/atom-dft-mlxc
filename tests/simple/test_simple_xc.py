"""SIMPLE on Jacob's ladder: PBE / r2SCAN fed only SIMPLE-reconstructed ingredients.

  * the XC potential (exchange + correlation, through the gradient and Laplacian
    operators) equals the finite-difference derivative of the discrete XC energy
    -- the exact discrete adjoint, the core correctness test;
  * full SCF with SIMPLE_GGA reproduces PBE for light atoms (H, He) and smooth
    pseudopotential valence atoms (C, Si), tightening with reconstruction order;
  * full SCF with SIMPLE_SCAN reproduces the orbital meta-GGA (rSCAN) for Si.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atom.scf.density import DensityData  # noqa: E402
from atom.xc.simple_xc import (  # noqa: E402
    SIMPLE_GGA,
    SIMPLE_SCAN,
    SIMPLEGGAParameters,
    SIMPLESCANParameters,
)


def _evaluator(cls, params, r):
    """Build a SIMPLE ladder evaluator on a uniform grid (dummy derivative
    matrix: compute_xc uses only the generic energy + SIMPLE operators)."""
    w = np.full(r.size, r[1] - r[0])
    return cls(derivative_matrix=np.zeros((1, r.size, 1)), r_quad=r,
               quadrature_weights=w, params=params)


def _fd_potential_check(cls, params):
    r = np.linspace(0.1, 5.0, 150)
    ev = _evaluator(cls, params, r)
    ew = ev.energy_weights

    def E_xc(rho):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ev.compute_xc(DensityData(rho=rho))
        return float(np.sum(ew * rho * (out.e_x + out.e_c)))

    rho = np.exp(-1.7 * r) + 0.3 * np.exp(-0.6 * r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ev.compute_xc(DensityData(rho=rho))
    v = out.v_x + out.v_c
    eps = 1e-6
    probe = np.arange(0, r.size, 8)
    fd = np.array([
        (E_xc(rho + eps * (np.arange(r.size) == j)) - E_xc(rho - eps * (np.arange(r.size) == j))) / (2 * eps)
        for j in probe
    ])
    scale = np.abs((ew * v)[probe]).max()
    return np.abs(fd - (ew * v)[probe]).max() / scale


def test_gga_potential_matches_fd():
    """SIMPLE_GGA: analytic v_xc == d E_xc / d rho (finite difference)."""
    assert _fd_potential_check(SIMPLE_GGA, SIMPLEGGAParameters()) < 1e-6


def test_scan_potential_matches_fd():
    """SIMPLE_SCAN: analytic v_xc (gradient + Laplacian adjoints) == FD."""
    assert _fd_potential_check(SIMPLE_SCAN, SIMPLESCANParameters()) < 1e-6


def _scf_energy(xc, z, all_electron=True, params=None, **kw):
    from atom.solver import AtomicDFTSolver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = AtomicDFTSolver(atomic_number=z, xc_functional=xc, all_electron_flag=all_electron,
                              verbose=False, xc_params=params, **kw).solve()
    return res


def test_simple_gga_reproduces_pbe_light_atoms():
    """SIMPLE_GGA reproduces PBE total energy for H, He (all-electron)."""
    for z, tol_mha in ((1, 0.2), (2, 2.0)):
        e_pbe = _scf_energy("GGA_PBE", z)["energy"]
        sg = _scf_energy("SIMPLE_GGA", z)
        assert sg["converged"]
        assert abs(sg["energy"] - e_pbe) * 1000.0 < tol_mha, (z, sg["energy"] - e_pbe)


def test_simple_gga_pseudopotential_valence():
    """With pseudopotentials (smooth valence) SIMPLE_GGA reproduces PBE to sub-mHa."""
    for z in (6, 14):  # C, Si
        e_pbe = _scf_energy("GGA_PBE", z, all_electron=False)["energy"]
        sg = _scf_energy("SIMPLE_GGA", z, all_electron=False)
        assert sg["converged"]
        assert abs(sg["energy"] - e_pbe) * 1000.0 < 1.0, (z, sg["energy"] - e_pbe)


def test_simple_gga_tightens_with_resolution():
    """SIMPLE_GGA-vs-PBE discrepancy shrinks as the gradient reconstruction order
    (channel/moment count) grows."""
    e_pbe = _scf_energy("GGA_PBE", 2)["energy"]
    errs = []
    for n_ch, n_mom in ((6, 3), (10, 5), (14, 7)):
        sg = _scf_energy("SIMPLE_GGA", 2, params=SIMPLEGGAParameters(n_channels=n_ch, n_moments=n_mom))
        errs.append(abs(sg["energy"] - e_pbe))
    assert errs[0] > errs[-1]
    assert errs[-1] * 1000.0 < 0.5


def test_simple_scan_si_reproduces_rscan():
    """SIMPLE_SCAN (rSCAN base, fully deorbitalized X+C) converges for Si and
    reproduces orbital rSCAN total energy to ~15 mHa -- the bare-GEA
    deorbitalization error (dominated by the functional, not SIMPLE; see
    docs/SIMPLE 11). Damped mixing for the stiff Laplacian-dependent potential."""
    ref = _scf_energy("RSCAN", 14, all_electron=False)
    sc = _scf_energy("SIMPLE_SCAN", 14, all_electron=False,
                     params=SIMPLESCANParameters(base="RSCAN"),
                     use_pulay_mixing=True, linear_mixing_alpha1=0.4, max_scf_iterations=400)
    assert sc["converged"]
    assert abs(sc["energy"] - ref["energy"]) * 1000.0 < 25.0, sc["energy"] - ref["energy"]
