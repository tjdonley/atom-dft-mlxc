"""Canonical SIMPLE feature API (docs/SIMPLE/SIMPLE.tex, Sec. "Pipeline summary").

Two *layered* entry points turn a SIMPLE descriptor ``result`` into the vectors a
functional consumes, with one consistent symbol for each:

    d = get_simple_features(result)     ->  the dimensionless descriptors  (symbol: d)
    I = get_simple_invariants(result)   ->  the rotation-invariant inputs   (symbol: I(d))

``d`` is the covariant primitive (it transforms under rotations); ``I(d)`` are the
rotationally invariant scalars that a functional actually reads. The mapping a
functional sees is therefore

    density  -->  d = get_simple_features(.)  -->  I(d) = get_simple_invariants(.)  -->  E_xc .

``result`` is the dict produced by the pipeline for one evaluation centre,
``{l: array (n_out, 2l+1)}`` plus the scalars ``r_ad``, ``rho_bar_safe`` (from
``pipeline.simple_descriptors`` on a radial environment, or ``grid_descriptors``
on a 3D density). These functions add *no new physics* -- they only flatten and
label the existing descriptor/invariant machinery into a single feature vector so
the code and the writeup share one notation.
"""
from __future__ import annotations

import numpy as np

from .invariants import bispectrum_components, flatten_bispectrum, power_spectrum
from .params import L_MAX


def _l_max_of(result, l_max):
    """Highest integer angular-momentum key present (or the requested cap)."""
    present = max(l for l in result if isinstance(l, (int, np.integer)))
    return present if l_max is None else min(l_max, present)


def get_simple_features(result, l_max=None):
    """The SIMPLE feature vector ``d``: the dimensionless descriptors d_{nlm},
    flattened to one 1D array with an explicit ``(n, l, m)`` layout.

    Parameters
    ----------
    result : dict
        SIMPLE descriptor dict for one centre, ``{l: array (n_out, 2l+1)}`` (+
        scalars), from ``pipeline.simple_descriptors`` or ``grid_descriptors``.
    l_max : int, optional
        Cap on angular momentum (default: all channels in ``result``).

    Returns
    -------
    d : ndarray (n_features,)
        The descriptors, ordered l = 0..l_max, then n = 0..n_out-1, then
        m = -l..+l.
    layout : list[tuple[int, int, int]]
        The ``(n, l, m)`` label of each entry of ``d`` (same length as ``d``).
    """
    lmax = _l_max_of(result, l_max)
    blocks, layout = [], []
    for l in range(lmax + 1):
        arr = np.asarray(result[l], dtype=float)
        if arr.ndim == 1:                          # axial (m=0 only)
            blocks.append(arr)
            layout += [(n, l, 0) for n in range(arr.shape[0])]
        else:                                      # full m-resolved (n_out, 2l+1)
            n_out, ncol = arr.shape
            blocks.append(arr.ravel())
            ms = range(-l, l + 1) if ncol == 2 * l + 1 else range(ncol)
            layout += [(n, l, m) for n in range(n_out) for m in ms]
    return np.concatenate(blocks), layout


def get_simple_invariants(result, l_max=None, include_bispectrum=False):
    """The rotation-invariant feature vector ``I(d)`` a functional consumes.

    Includes the power spectrum ``P_{nl} = sum_m d_{nlm}^2`` (ordered
    l = 0..l_max, n = 0..n_out-1); with ``include_bispectrum`` it appends the
    Clebsch-Gordan bispectrum (large).

    The reduced gradient ``s`` and reduced Laplacian ``q`` [Eq. (sq)] are NOT
    derived from this (lossy) descriptor dict: they are obtained from the
    parameter-free spectral operators ``build_spectral_gradient_operator`` /
    ``build_spectral_laplacian_operator`` applied to the density on its grid, in
    the functional/SCF context where that grid is available (see
    ``atom/xc/simple_xc.py``). The single *dimensional* feature -- the local scale
    ``k_bar_F`` -- is likewise left out of ``I`` (which is scale-free) and is
    available as ``(6 * pi**2 * result['rho_bar_safe'])**(1/3)``.

    Returns
    -------
    I : ndarray (n_invariants,)
    names : list[str]
        A human-readable label per entry of ``I``.
    """
    lmax = _l_max_of(result, l_max)
    # Power spectrum P_{nl}=sum_m d_{nlm}^2: promote axial 1D blocks to (n_out,1)
    # so the existing axis=1 reduction yields d^2 (the only nonzero m for axial).
    pdict = {}
    for l in range(lmax + 1):
        arr = np.asarray(result[l], dtype=float)
        pdict[l] = arr[:, None] if arr.ndim == 1 else arr
    n_out = pdict[0].shape[0]
    P = np.asarray(power_spectrum(pdict, lmax), dtype=float)
    vals = [P]
    names = [f"P[n={n},l={l}]" for l in range(lmax + 1) for n in range(n_out)]

    if include_bispectrum:
        full = {l: np.asarray(result[l]) for l in range(lmax + 1)}
        if any(full[l].ndim != 2 or full[l].shape[1] != 2 * l + 1 for l in range(lmax + 1)):
            raise ValueError(
                "include_bispectrum needs full m-resolved descriptors "
                "(grid_descriptors / simple_from_window), not the axial "
                "(m=0) blocks from simple_descriptors.")
        b, _ = flatten_bispectrum(bispectrum_components(full, lmax))
        vals.append(np.asarray(b, dtype=float))
        names += [f"B[{i}]" for i in range(b.size)]

    return np.concatenate(vals), names
