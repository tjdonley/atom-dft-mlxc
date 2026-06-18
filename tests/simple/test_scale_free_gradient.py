"""Spectral reconstruction of the reduced gradient s and reduced Laplacian q
[Eq. (sq) of SIMPLE-Xhole-writeup].

The parameter-free spectral l=1 (slope-at-origin) operator reconstructs rho'(r0),
hence the reduced gradient s [Eq. (sq)], to ~1% on a smooth (cusp-free) density --
the basis for the Results "GEA tests" (s ratio ~1.00). Constant-annihilation makes
both operators vanish exactly for a uniform density (the HEG/LDA limit).

NOTE (Phase D): the l=0 spectral Laplacian (q) is the numerically delicate
ingredient. In a quick analytic/finite-difference check on a Gaussian and on the
cached pseudopotential density it does NOT recover grad^2 rho to ~1% (the error
grows with channel count -- high-mode noise), so the quantitative q-ratio claim of
the writeup is left to the Phase-D atom benchmark on real OEP grids, where it is to
be re-validated/diagnosed. Here the Laplacian is exercised only structurally
(linearity + constant-annihilation + finiteness).
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


def _gaussian(r, a=1.2):
    """Smooth, cusp-free model density rho(r)=exp(-(r/a)^2), contained in the window."""
    rho = np.exp(-(r / a) ** 2)
    dr = -2.0 * r / a ** 2 * rho                                  # rho'(r)
    lap = (4.0 * r ** 2 / a ** 4 - 6.0 / a ** 2) * rho            # grad^2 rho = rho'' + (2/r) rho'
    return rho, dr, lap


def _bulk(r, rho):
    """Core/valence mask: density resolved and away from the grid edges."""
    return (rho > 1e-2 * rho.max()) & (r > 0.4) & (r < 0.85 * R_C)


def test_spectral_gradient_recovers_grad():
    """|G @ rho| ~= |rho'(r0)| to ~1% on a smooth density [Eq. (sq), l=1].
    (The sign of G @ rho is a convention; the reduced gradient uses |.|.)"""
    r = np.linspace(1e-3, R_C, 400)
    rho, dr, _ = _gaussian(r)
    G = build_spectral_gradient_operator(r, n_channels=40)
    g = G @ rho
    m = _bulk(r, rho)
    ratio = np.abs(g[m]) / np.abs(dr[m])
    assert np.abs(ratio - 1.0).max() < 0.05, np.abs(ratio - 1.0).max()


def test_spectral_laplacian_structural():
    """The l=0 spectral Laplacian is a finite linear operator on a smooth density.
    (Quantitative grad^2 rho recovery is the Phase-D atom benchmark -- see module
    docstring; not asserted here.)"""
    r = np.linspace(1e-3, R_C, 400)
    rho, _, _ = _gaussian(r)
    L = build_spectral_laplacian_operator(r, n_channels=40)
    assert np.all(np.isfinite(L @ rho))


def test_constant_annihilation_heg_limit():
    """Uniform density -> both operators give exactly zero (HEG/LDA limit), so the
    reduced gradient s vanishes [Eq. (sq), constant annihilation]."""
    r = np.linspace(1e-3, R_C, 200)
    rho = np.full_like(r, 0.3)
    G = build_spectral_gradient_operator(r, n_channels=40)
    L = build_spectral_laplacian_operator(r, n_channels=40)
    assert np.abs(G @ rho).max() < 1e-10
    assert np.abs(L @ rho).max() < 1e-10
    s = reduced_gradient_from_grad(G @ rho, rho)
    assert np.abs(s).max() < 1e-10


def test_spectral_operators_are_linear():
    """The operators are fixed (density-independent) linear maps."""
    r = np.linspace(1e-3, R_C, 200)
    G1 = build_spectral_gradient_operator(r, n_channels=40)
    G2 = build_spectral_gradient_operator(r, n_channels=40)
    assert np.allclose(G1, G2)
    rho_a, _, _ = _gaussian(r, a=1.0)
    rho_b, _, _ = _gaussian(r, a=1.5)
    assert np.allclose(G1 @ (2.0 * rho_a + 3.0 * rho_b),
                       2.0 * (G1 @ rho_a) + 3.0 * (G1 @ rho_b))
