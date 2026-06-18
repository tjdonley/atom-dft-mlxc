"""Fast unit tests for the SIMPLE descriptors (atom/descriptors/simple/).

These run in well under a minute and cover the structural identities; the
full validation suite (figures plus the numbers quoted in
docs/SIMPLE/SIMPLE.tex) is the pair of scripts radial_validation.py and
cartesian_validation.py in this directory.
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atom.descriptors.simple import (  # noqa: E402
    L_MAX,
    LatticeStencil,
    N_OUT,
    R_C,
    bispectrum_components,
    flatten_bispectrum,
    grid_descriptors,
    mirror_matrix,
    n_in_for,
    parity,
    radial_gauss_grid,
    random_rotation_matrix,
    real_sh_rotation_matrix,
    real_wigner_d,
    self_check_couplings,
    simple_descriptors,
    simple_from_window,
    window_basis,
)

HEG_REFERENCE = (-1.0) ** np.arange(N_OUT) / (np.arange(N_OUT) + 1)


def _heg_profiles(rho0):
    def profiles(r, l):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return np.full_like(r, rho0) if l == 0 else np.zeros_like(r)

    return profiles


def test_coupling_self_checks():
    """CG orthogonality, dual coupling-tensor constructions, anchors,
    synthetic invariance and parity -- all at machine precision."""
    self_check_couplings()


def test_wigner_general_l():
    """Sampled Wigner-D blocks: closed-form agreement at l <= 2,
    orthogonality and the representation property at l = 3."""
    rng = np.random.default_rng(20260612)
    r1, r2 = random_rotation_matrix(rng), random_rotation_matrix(rng)
    for l in (1, 2):
        err = np.abs(real_wigner_d(l, r1) - real_sh_rotation_matrix(l, r1)).max()
        assert err < 1e-12
    d3a, d3b = real_wigner_d(3, r1), real_wigner_d(3, r2)
    assert np.abs(d3a @ d3a.T - np.eye(7)).max() < 1e-12
    assert np.abs(real_wigner_d(3, r1 @ r2) - d3a @ d3b).max() < 1e-12


def test_heg_continuum():
    """The HEG limit is exact (to root-find tolerance) in the continuum."""
    for r_s in (1.0, 5.0):
        rho0 = 3.0 / (4.0 * np.pi * r_s**3) / 2.0
        res = simple_descriptors(_heg_profiles(rho0), 32, l_max=0)
        assert np.abs(res[0] - HEG_REFERENCE).max() < 1e-9


def test_heg_lattice():
    """A uniform density through the full gridded pipeline reproduces the
    analytic HEG monopole values to the kernel-quadrature level, with
    machine-zero leakage into l > 0."""
    h = 0.4  # coarse grid keeps this test fast
    stencil = LatticeStencil(h, n_in_for(h))
    rho0 = 0.01
    c_window = stencil.window_coefficients(np.full(len(stencil.points), rho0))
    res = simple_from_window(c_window, lambda radius: rho0, stencil.n_in)
    assert np.abs(res[0][:, 0] - HEG_REFERENCE).max() < 5e-3
    assert max(np.abs(res[l]).max() for l in range(1, L_MAX + 1)) < 1e-12


def test_grid_rotational_invariance_coarse():
    """Power-spectrum invariants of a two-atom density on a coarse lattice
    are orientation-independent at the discretization level."""
    from cartesian_validation import molecule
    from atom.descriptors.simple import power_spectrum

    h = 0.4
    stencil = LatticeStencil(h, n_in_for(h))
    mol = molecule("H", "offaxis")
    rng = np.random.default_rng(7)
    spectra = np.array([
        power_spectrum(grid_descriptors(mol.rotated(random_rotation_matrix(rng)),
                                        stencil))
        for _ in range(3)
    ])
    spread = np.abs(spectra - spectra.mean(axis=0)).max()
    assert spread < 5e-3 * np.abs(spectra.mean(axis=0)).max()


def _radial_gram(l, n_in, h=None):
    basis = window_basis(l, n_in)
    if h is None:
        quad = radial_gauss_grid(R_C, 1024)
        nodes, weights = quad.nodes, quad.weights
    else:
        from atom.descriptors.simple import grid_nodes_weights

        nodes, weights, _ = grid_nodes_weights(h)
    values = basis.evaluate(l, nodes)
    return np.einsum("ni,mi,i->nm", values, values, weights * nodes**2)


def test_channel_orthogonality():
    """The window basis is orthonormal by construction (continuum Gram is
    the identity), and remains well-conditioned -- so each channel adds new
    information -- up to the grid's resolution bound n_in ~ R_c/h, beyond
    which the sampled channels become linearly dependent."""
    for l in range(L_MAX + 1):
        gram = _radial_gram(l, 32)
        assert np.abs(gram - np.eye(32)).max() < 1e-9          # orthonormal
    g16 = _radial_gram(0, 16, h=0.2)
    assert np.linalg.cond(g16) < 1.1                            # independent
    assert np.abs(g16 - np.eye(16)).max() < 2e-2               # ~orthonormal
    # past the node count (~R_c/h = 28 at h = 0.2) the channels collapse
    assert np.linalg.cond(_radial_gram(0, 40, h=0.2)) > 1e6


def test_adaptive_radius_matches_root_find():
    """The closed-form enclosed-moment inversion reproduces the radius that a
    Brent root find on R k_F(rho_bar(R)) = xi* would return."""
    from scipy.optimize import brentq
    from atom.descriptors.simple import R_C, XI_TARGET, adaptive_radius
    from atom.descriptors.simple.pipeline import ball_average, grid_nodes_weights

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from radial_validation import environment_profiles

    _, widths, starts = grid_nodes_weights(0.2)
    shell_edges = starts + widths
    for lam in (0.5, 1.0, 2.0, 4.0):
        pr = environment_profiles(lam=lam)
        rho_bar = lambda R, pr=pr: ball_average(lambda r: pr(r, 0), R, h=0.2)
        r_closed, clamped = adaptive_radius(rho_bar, shell_edges)
        r_root = brentq(
            lambda R: R * (6.0 * np.pi**2 * max(rho_bar(R), 0.0)) ** (1 / 3)
            - XI_TARGET,
            1e-3, R_C, rtol=1e-13,
        )
        assert not clamped and abs(r_closed - r_root) < 1e-9

    # uniform density: cube-root inversion is exact for any sampling
    r_heg, clamped = adaptive_radius(lambda R: 0.01)
    assert not clamped
    assert abs(0.01 * r_heg**3 / 3.0 - XI_TARGET**3 / (18 * np.pi**2)) < 1e-14


def test_bispectrum_parity_synthetic():
    """Scalar components are mirror-even, pseudoscalar components are
    mirror-odd, on synthetic multiplets (pure coupling algebra)."""
    rng = np.random.default_rng(11)
    multiplets = {l: rng.normal(size=(3, 2 * l + 1)) for l in range(L_MAX + 1)}
    mirror = mirror_matrix(rng.normal(size=3))
    mirrored = {
        l: multiplets[l] @ real_wigner_d(l, mirror).T for l in range(L_MAX + 1)
    }
    base, par = flatten_bispectrum(bispectrum_components(multiplets))
    moved, _ = flatten_bispectrum(bispectrum_components(mirrored))
    signs = np.where(par == 0, 1.0, -1.0)
    assert np.abs(moved - signs * base).max() < 1e-12
    assert parity((1, 1, 1)) == 1 and parity((1, 1, 2)) == 0
