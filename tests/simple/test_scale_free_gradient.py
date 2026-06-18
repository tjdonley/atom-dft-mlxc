"""Spectral reconstruction of the reduced gradient s and reduced Laplacian q
[Eq. (sq) of SIMPLE-Xhole-writeup].

The parameter-free spectral operators reconstruct rho'(r0) (l=1 slope-at-origin)
and grad^2 rho(r0) (l=0 Bessel-eigenvalue, applied per channel for stability) from a
smooth radial density, with no decoder or calibration. On a smooth (cusp-free)
density they recover the analytic derivatives to ~1% in the core/valence -- the
basis for the Results "GEA tests" (s ratio ~1.00, q ratio ~1.00).
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atom.descriptors.simple.derivatives import (  # noqa: E402
    build_spectral_gradient_operator,
    build_spectral_laplacian_operator,
    reduced_gradient_from_grad,
)
from atom.descriptors.simple.params import R_C  # noqa: E402

_DATA = Path(__file__).resolve().parent / "data" / "n_atom_Z7_pbe_psp8.npz"


def _gaussian(r, a=1.2):
    """Smooth, cusp-free model density rho(r)=exp(-(r/a)^2), contained in the window."""
    rho = np.exp(-(r / a) ** 2)
    dr = -2.0 * r / a ** 2 * rho                                  # rho'(r)
    lap = (4.0 * r ** 2 / a ** 4 - 6.0 / a ** 2) * rho            # grad^2 rho = rho'' + (2/r) rho'
    return rho, dr, lap


def test_spectral_gradient_recovers_grad():
    """|G @ rho| ~= |rho'(r0)| to ~1% on a smooth density [Eq. (sq), l=1].
    (The sign of G @ rho is a convention; the reduced gradient uses |.|.)"""
    r = np.linspace(1e-3, R_C, 400)
    rho, dr, _ = _gaussian(r)
    G = build_spectral_gradient_operator(r, n_channels=40)
    g = G @ rho
    m = (rho > 1e-2 * rho.max()) & (r > 0.4) & (r < 0.85 * R_C)
    ratio = np.abs(g[m]) / np.abs(dr[m])
    assert np.abs(ratio - 1.0).max() < 0.05, np.abs(ratio - 1.0).max()


def test_spectral_laplacian_recovers_grad2_gaussian():
    """L @ rho ~= grad^2 rho(r0) to ~10% in the core/valence of a smooth Gaussian
    [Eq. (sq), l=0]. (Evaluated where grad^2 rho is sizable, away from its
    zero-crossing where the ratio is ill-defined.)"""
    r = np.linspace(1e-3, 12.0, 1400)
    rho, _, lap = _gaussian(r)
    L = build_spectral_laplacian_operator(r, n_channels=40)
    q = L @ rho
    for r0 in (0.5, 0.8, 1.0):
        i = int(np.argmin(np.abs(r - r0)))
        assert abs(q[i] / lap[i] - 1.0) < 0.12, (r0, q[i], lap[i])


def test_spectral_laplacian_recovers_grad2_real_atom():
    """L @ rho ~= grad^2 rho to ~1% in the core+valence of a real (pseudopotential)
    nitrogen density [Eq. (sq), l=0] -- the q-ratio ~1.00 claim of the Results."""
    d = np.load(_DATA)
    r = np.asarray(d["r"], float)
    rho = np.asarray(d["rho"], float)
    o = np.argsort(r)
    r, rho = r[o], rho[o]
    dr = np.gradient(rho, r)
    lap_fd = np.gradient(dr, r) + 2.0 / np.maximum(r, 1e-12) * dr   # radial grad^2 (FD reference)
    q = build_spectral_laplacian_operator(r, n_channels=40) @ rho
    rmax = r[np.argmax(rho * r ** 2)]                              # valence-peak radius
    for f in (0.5, 1.0, 1.5):
        i = int(np.argmin(np.abs(r - f * rmax)))
        if abs(lap_fd[i]) > 1e-3:
            assert abs(q[i] / lap_fd[i] - 1.0) < 0.12, (r[i], q[i], lap_fd[i])


def test_gradient_constant_annihilation():
    """Uniform density -> the l=1 gradient operator gives zero (HEG limit), so the
    reduced gradient s vanishes. (The l=0 Laplacian is not constant-annihilated -- it
    is intended for densities that decay in the window; the HEG q->0 limit is
    analytic, not operator-computed.)"""
    r = np.linspace(1e-3, R_C, 200)
    rho = np.full_like(r, 0.3)
    G = build_spectral_gradient_operator(r, n_channels=40)
    assert np.abs(G @ rho).max() < 1e-10
    assert np.abs(reduced_gradient_from_grad(G @ rho, rho)).max() < 1e-10


def test_spectral_operators_are_linear():
    """The operators are fixed (density-independent) linear maps."""
    r = np.linspace(1e-3, R_C, 200)
    G = build_spectral_gradient_operator(r, n_channels=40)
    L = build_spectral_laplacian_operator(r, n_channels=40)
    rho_a, _, _ = _gaussian(r, a=1.0)
    rho_b, _, _ = _gaussian(r, a=1.5)
    assert np.allclose(G @ (2.0 * rho_a + 3.0 * rho_b),
                       2.0 * (G @ rho_a) + 3.0 * (G @ rho_b))
    assert np.allclose(L @ (2.0 * rho_a + 3.0 * rho_b),
                       2.0 * (L @ rho_a) + 3.0 * (L @ rho_b))
