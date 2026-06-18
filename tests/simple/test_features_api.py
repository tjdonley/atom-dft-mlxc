"""Smoke tests for the canonical SIMPLE feature API (features.py):
get_simple_features (descriptors d) and get_simple_invariants (invariants I(d))."""
import numpy as np

from atom.descriptors.simple import (
    N_OUT,
    get_simple_features,
    get_simple_invariants,
    simple_descriptors,
)

HEG_REFERENCE = (-1.0) ** np.arange(N_OUT) / (np.arange(N_OUT) + 1)


def _heg_profiles(rho0):
    def profiles(r, l):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return np.full_like(r, rho0) if l == 0 else np.zeros_like(r)
    return profiles


def _gaussian_profiles(rho0, width):
    """A smoothly varying (inhomogeneous) axial environment: l=0 Gaussian bump,
    a small l=1 dipole, no higher multipoles -- just to exercise the API."""
    def profiles(r, l):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        g = rho0 * np.exp(-(r / width) ** 2)
        if l == 0:
            return g
        if l == 1:
            return 0.1 * g * (r / width)
        return np.zeros_like(r)
    return profiles


def test_features_layout_and_heg_values():
    """d = get_simple_features flattens the descriptors with an (n,l,m) layout;
    for a HEG it reduces to d^HEG_{n00} = (-1)^n/(n+1) with zero higher channels."""
    res = simple_descriptors(_heg_profiles(0.1), 32, l_max=2)   # axial: m=0 only
    d, layout = get_simple_features(res)

    expected_len = N_OUT * 3                                    # l=0,1,2 m=0 blocks
    assert d.shape == (expected_len,)
    assert len(layout) == expected_len
    assert layout[0] == (0, 0, 0) and layout[N_OUT - 1] == (N_OUT - 1, 0, 0)
    assert layout[N_OUT] == (0, 1, 0)                           # l=1 block starts

    # l=0 block is the monopole HEG reference; l>=1 channels vanish for a HEG.
    assert np.abs(d[:N_OUT] - HEG_REFERENCE).max() < 1e-6
    assert np.abs(d[N_OUT:]).max() < 1e-6


def test_invariants_heg_limit():
    """I = get_simple_invariants: the power spectrum reduces to the HEG monopole
    values and the l>=1 invariants vanish for a HEG. (The reduced gradient s and
    Laplacian q come from the spectral operators [Eq. (sq)] applied to the density,
    not from this descriptor dict -- see test_scale_free_gradient.)"""
    res = simple_descriptors(_heg_profiles(0.1), 32, l_max=2)
    I, names = get_simple_invariants(res)

    assert I.shape[0] == len(names)
    # power spectrum: l=0 block = (1/(n+1))^2, l>=1 blocks ~ 0
    P_l0 = I[:N_OUT]
    assert np.abs(P_l0 - HEG_REFERENCE ** 2).max() < 1e-6
    assert np.abs(I[N_OUT:3 * N_OUT]).max() < 1e-6        # l=1,2 power blocks


def test_features_inhomogeneous_runs():
    """The API runs on an inhomogeneous environment and turns on l=1 content."""
    res = simple_descriptors(_gaussian_profiles(0.1, 1.5), 32, l_max=2)
    d, layout = get_simple_features(res)
    I, names = get_simple_invariants(res)
    assert np.isfinite(d).all() and np.isfinite(I).all()
    # the dipole profile gives nonzero l=1 descriptors
    assert np.abs(d[N_OUT:2 * N_OUT]).max() > 0.0     # l=1 (m=0) block
