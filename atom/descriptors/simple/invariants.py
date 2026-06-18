"""Rotationally invariant contractions of the SIMPLE descriptors:
power spectrum and real-basis Clebsch-Gordan (CG) bispectrum.

The SIMPLE descriptors d_nlm are expressed in *real* spherical harmonics, so
invariant contractions need the real-basis analog of the CG coefficients:
for each angular-momentum triple (l1, l2, l3) satisfying the triangle
inequality, the space of 3-tensors C_{m1 m2 m3} invariant under the
simultaneous real rotation D^(l1) x D^(l2) x D^(l3) is exactly
one-dimensional. This module constructs that tensor in two INDEPENDENT ways
and verifies that they agree (self_check_couplings):

  1. coupling_tensor(...)          - sampled null space of the invariance
       constraint (D1 x D2 x D3 - 1) C = 0 stacked over a few random
       rotations; the constraint is linear and exact, so SVD recovers the
       tensor to machine precision.
  2. coupling_tensor_from_cg(...)  - the standard complex-basis CG
       contraction <l1 m1, l2 m2 | l3 m3> (Racah closed form) with a
       conjugated third slot, transformed to the real basis with the
       unitary real-to-complex matrices U^(l).

Both are normalized to unit Frobenius norm with a deterministic sign, and
agree to ~1e-15 for every triple with l <= 3, together with the analytic
anchors C^(l,l,0) = delta / sqrt(2l+1) and C^(1,1,1) = Levi-Civita/sqrt(6).

Parity classification: couplings with even l1+l2+l3 are true scalars under
the full orthogonal group O(3); couplings with odd l1+l2+l3 are
PSEUDOSCALARS - invariant under proper rotations but changing sign under
improper ones (reflections, inversion). Pseudoscalar bispectrum components
vanish identically for achiral (e.g. planar) environments and are the
natural chirality-sensitive features.

The bispectrum of a SIMPLE descriptor set {d_l} is

    B_{n1 n2 n3; l1 l2 l3} = sum_{m1 m2 m3} C^(l1 l2 l3)_{m1 m2 m3}
                             d_{n1 l1 m1} d_{n2 l2 m2} d_{n3 l3 m3} ,

computed by bispectrum_components for all ordered triples l1 <= l2 <= l3.
See Sec. 6.2 and Sec. 9.5 of docs/SIMPLE/SIMPLE.tex.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from .params import L_MAX
from .rotations import (
    mirror_matrix,
    random_rotation_matrix,
    real_sh_rotation_matrix,
    real_wigner_d,
)

_COUPLING_SEED = 1357924680
_COUPLING_CACHE = {}


# =============================================================================
# Second-order invariants
# =============================================================================
def power_spectrum(result, l_max=L_MAX):
    """P_nl = sum_m d_nlm^2, flattened over (l, n) in concatenated order.

    ``result`` is a SIMPLE descriptor dict {l: array (n_out, 2l+1)}."""
    return np.concatenate(
        [np.sum(result[l] ** 2, axis=1) for l in range(l_max + 1)]
    )


def channel_magnitudes(result, l_max=L_MAX):
    """Per-(n, l) channel magnitudes ||d_{nl.}|| (norm over m), ordered as
    power_spectrum (so channel_magnitudes**2 == power_spectrum)."""
    return np.concatenate(
        [np.linalg.norm(result[l], axis=1) for l in range(l_max + 1)]
    )


# =============================================================================
# Clebsch-Gordan coefficients (complex convention, Racah closed form)
# =============================================================================
def clebsch_gordan(l1, m1, l2, m2, l3, m3):
    """<l1 m1, l2 m2 | l3 m3> for integer angular momenta (Racah formula)."""
    if m3 != m1 + m2:
        return 0.0
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    pref = (
        (2 * l3 + 1)
        * factorial(l1 + l2 - l3)
        * factorial(l1 - l2 + l3)
        * factorial(-l1 + l2 + l3)
        / factorial(l1 + l2 + l3 + 1)
    )
    pref *= (
        factorial(l3 + m3) * factorial(l3 - m3)
        * factorial(l1 - m1) * factorial(l1 + m1)
        * factorial(l2 - m2) * factorial(l2 + m2)
    )
    total = 0.0
    k_min = max(0, l2 - l3 - m1, l1 - l3 + m2)
    k_max = min(l1 + l2 - l3, l1 - m1, l2 + m2)
    for k in range(k_min, k_max + 1):
        total += (-1.0) ** k / (
            factorial(k)
            * factorial(l1 + l2 - l3 - k)
            * factorial(l1 - m1 - k)
            * factorial(l2 + m2 - k)
            * factorial(l3 - l2 + m1 + k)
            * factorial(l3 - l1 - m2 + k)
        )
    return np.sqrt(pref) * total


def u_real_to_complex(l):
    """Unitary U^(l) with c^complex = U @ c^real.

    Matches the real_sph_harm convention (Condon-Shortley phase absorbed):
        Y^R_0    = Y_0
        Y^R_{+u} = ((-1)^u Y_{+u} + Y_{-u}) / sqrt(2)        (u > 0)
        Y^R_{-u} = ((-1)^u Y_{+u} - Y_{-u}) / (i sqrt(2))
    """
    dim = 2 * l + 1
    u_mat = np.zeros((dim, dim), dtype=complex)

    def idx(m):
        return m + l

    u_mat[idx(0), idx(0)] = 1.0
    for mu in range(1, l + 1):
        sign = (-1.0) ** mu
        u_mat[idx(mu), idx(mu)] = sign / np.sqrt(2.0)
        u_mat[idx(-mu), idx(mu)] = 1.0 / np.sqrt(2.0)
        u_mat[idx(mu), idx(-mu)] = -1j * sign / np.sqrt(2.0)
        u_mat[idx(-mu), idx(-mu)] = 1j / np.sqrt(2.0)
    return u_mat


# =============================================================================
# Real-basis coupling tensors
# =============================================================================
def _fix_norm_and_sign(tensor):
    tensor = tensor / np.linalg.norm(tensor)
    flat = tensor.ravel()
    pivot = int(np.argmax(np.abs(flat)))
    return tensor if flat[pivot] > 0 else -tensor


def coupling_tensor(l1, l2, l3):
    """The unique invariant coupling tensor C^(l1 l2 l3) of the real
    representations (unit Frobenius norm, deterministic sign), from the
    sampled null space of the simultaneous-rotation constraint."""
    key = (l1, l2, l3)
    if key in _COUPLING_CACHE:
        return _COUPLING_CACHE[key]
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        raise ValueError(f"triangle inequality violated for {key}")
    if (l1, l2, l3) == (0, 0, 0):
        _COUPLING_CACHE[key] = np.ones((1, 1, 1))
        return _COUPLING_CACHE[key]
    rng = np.random.default_rng(_COUPLING_SEED)
    dim = (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)
    rows = []
    for _ in range(3):
        rot = random_rotation_matrix(rng)
        kron = np.kron(
            np.kron(real_wigner_d(l1, rot), real_wigner_d(l2, rot)),
            real_wigner_d(l3, rot),
        )
        rows.append(kron - np.eye(dim))
    _, sing, vt = np.linalg.svd(np.vstack(rows))
    assert sing[-1] < 1e-10 and sing[-2] > 0.05, (
        f"invariant subspace of {key} is not one-dimensional: "
        f"smallest singular values {sing[-2:].tolist()}"
    )
    tensor = _fix_norm_and_sign(
        vt[-1].reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    )
    _COUPLING_CACHE[key] = tensor
    return tensor


def coupling_tensor_from_cg(l1, l2, l3):
    """The same coupling tensor built from the complex-basis CG contraction
    sum <l1 m1, l2 m2 | l3 m3> c1 c2 conj(c3), transformed to the real basis
    with U^(l). The transformed tensor is purely real for even l1+l2+l3 and
    purely imaginary for odd; the surviving part is returned (unit norm,
    deterministic sign)."""
    cg = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = m1 + m2
            if abs(m3) <= l3:
                cg[m1 + l1, m2 + l2, m3 + l3] = clebsch_gordan(
                    l1, m1, l2, m2, l3, m3
                )
    tensor = np.einsum(
        "abc,ai,bj,ck->ijk",
        cg.astype(complex),
        u_real_to_complex(l1),
        u_real_to_complex(l2),
        np.conj(u_real_to_complex(l3)),
    )
    real_part, imag_part = np.real(tensor), np.imag(tensor)
    surviving = real_part if (
        np.linalg.norm(real_part) > np.linalg.norm(imag_part)
    ) else imag_part
    return _fix_norm_and_sign(surviving)


def triangle_triples(l_max=L_MAX):
    """All ordered triples l1 <= l2 <= l3 <= l_max satisfying the triangle
    inequality."""
    return [
        (l1, l2, l3)
        for l1 in range(l_max + 1)
        for l2 in range(l1, l_max + 1)
        for l3 in range(l2, l_max + 1)
        if l3 <= l1 + l2
    ]


def parity(triple):
    """0 for scalar couplings (even l1+l2+l3), 1 for pseudoscalar (odd)."""
    return sum(triple) % 2


# =============================================================================
# Bispectrum
# =============================================================================
def bispectrum_components(result, l_max=L_MAX):
    """B_{n1 n2 n3; l1 l2 l3} for all ordered triples l1 <= l2 <= l3.

    ``result`` is a SIMPLE descriptor dict {l: array (n_out, 2l+1)}. Returns
    {(l1, l2, l3): array (n_out, n_out, n_out)}. Components related by
    permutation of equal-l slots are kept (they are equal up to the exchange
    symmetry of the coupling tensor); diagonal components of antisymmetric
    couplings are exact zeros."""
    return {
        (l1, l2, l3): np.einsum(
            "abc,ia,jb,kc->ijk",
            coupling_tensor(l1, l2, l3),
            result[l1],
            result[l2],
            result[l3],
        )
        for (l1, l2, l3) in triangle_triples(l_max)
    }


def flatten_bispectrum(components):
    """(values, parities) flattened in deterministic triple order."""
    triples = sorted(components.keys())
    values = np.concatenate([components[t].ravel() for t in triples])
    parities = np.concatenate(
        [np.full(components[t].size, parity(t), dtype=int) for t in triples]
    )
    return values, parities


# =============================================================================
# Self-checks
# =============================================================================
def self_check_couplings(l_max=L_MAX):
    """Validate the coupling machinery end to end; used by the validation
    suite at startup and by the unit tests."""
    # Racah formula spot values and orthogonality
    assert abs(clebsch_gordan(1, 0, 1, 0, 2, 0) - np.sqrt(2.0 / 3.0)) < 1e-14
    assert abs(clebsch_gordan(1, 1, 1, -1, 0, 0) - np.sqrt(1.0 / 3.0)) < 1e-14
    assert abs(clebsch_gordan(1, 0, 1, 0, 0, 0) + np.sqrt(1.0 / 3.0)) < 1e-14
    err_orth = 0.0
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            rows = []
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                for m3 in range(-l3, l3 + 1):
                    rows.append([
                        clebsch_gordan(l1, m1, l2, m2, l3, m3)
                        for m1 in range(-l1, l1 + 1)
                        for m2 in range(-l2, l2 + 1)
                    ])
            gram = np.array(rows) @ np.array(rows).T
            err_orth = max(err_orth, np.abs(gram - np.eye(len(rows))).max())
    # unitarity of the real-to-complex transform
    err_unitary = max(
        np.abs(
            u_real_to_complex(l) @ np.conj(u_real_to_complex(l)).T
            - np.eye(2 * l + 1)
        ).max()
        for l in range(l_max + 1)
    )
    # the two constructions agree for every triple (up to overall sign)
    err_cross = 0.0
    for triple in triangle_triples(l_max):
        a = coupling_tensor(*triple).ravel()
        b = coupling_tensor_from_cg(*triple).ravel()
        err_cross = max(err_cross, 1.0 - abs(float(np.dot(a, b))))
    # analytic anchors: (l, l, 0) is delta/sqrt(2l+1); (1, 1, 1) is the
    # Levi-Civita tensor / sqrt(6)
    err_anchor = 0.0
    for l in range(1, l_max + 1):
        expected = np.eye(2 * l + 1)[:, :, None] / np.sqrt(2 * l + 1)
        err_anchor = max(
            err_anchor,
            np.abs(np.abs(coupling_tensor(l, l, 0)) - np.abs(expected)).max(),
        )
    eps = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, k], eps[i, k, j] = 1.0, -1.0
    err_anchor = max(
        err_anchor,
        np.abs(np.abs(coupling_tensor(1, 1, 1)) - np.abs(eps) / np.sqrt(6)).max(),
    )
    # invariance and parity on synthetic multiplets
    rng = np.random.default_rng(_COUPLING_SEED + 1)
    multiplets = {
        l: rng.normal(size=(4, 2 * l + 1)) for l in range(l_max + 1)
    }
    base = bispectrum_components(multiplets, l_max)
    rot = random_rotation_matrix(rng)
    mirror = mirror_matrix(rng.normal(size=3))
    err_inv, err_par = 0.0, 0.0
    for transform, kind in ((rot, "rot"), (mirror, "mirror")):
        moved = {
            l: multiplets[l] @ real_wigner_d(l, transform).T
            for l in range(l_max + 1)
        }
        comps = bispectrum_components(moved, l_max)
        for triple, values in comps.items():
            sign = 1.0 if (kind == "rot" or parity(triple) == 0) else -1.0
            err = np.abs(values - sign * base[triple]).max()
            if kind == "rot":
                err_inv = max(err_inv, err)
            else:
                err_par = max(err_par, err)
    # consistency with the l <= 2 closed-form rotation blocks
    err_wigner = max(
        np.abs(real_wigner_d(l, rot) - real_sh_rotation_matrix(l, rot)).max()
        for l in (1, 2)
    )
    print(
        "self-check couplings: CG orthogonality "
        f"{err_orth:.1e}, U unitary {err_unitary:.1e}, null-space vs "
        f"CG-transform {err_cross:.1e}, anchors {err_anchor:.1e}, "
        f"invariance {err_inv:.1e}, parity {err_par:.1e}, "
        f"Wigner-D l<=2 {err_wigner:.1e}"
    )
    for err in (err_orth, err_unitary, err_cross, err_anchor, err_inv,
                err_par, err_wigner):
        assert err < 1e-12, "coupling self-check failed"


if __name__ == "__main__":
    self_check_couplings()
