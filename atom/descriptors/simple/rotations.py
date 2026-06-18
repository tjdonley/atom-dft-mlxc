"""Real-spherical-harmonic rotation machinery for the SIMPLE descriptors.

Conventions: real spherical harmonics ordered m = -l..l, built from the
complex Y_lm with the Condon-Shortley phase absorbed, so that for l = 1 the
basis functions are proportional to (y, z, x). The block D^(l)(R) satisfies
Y_lm(R u) = sum_m' D_mm' Y_lm'(u); it is orthogonal and a genuine
representation. Covariant coefficient vectors rotate as d' = D d.

``real_wigner_d`` constructs D^(l) for ANY l and ANY orthogonal matrix
(including improper ones, det = -1) by sampled least squares: the defining
relation is linear and exact, so the solve recovers D to machine precision.
``real_sh_rotation_matrix`` provides independent closed-form / sampled-exact
blocks for l <= 2, used by the self-checks.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm


def rotation_matrix_3d(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle`` (radians)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("axis must be nonzero.")
    x, y, z = axis / norm
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(angle)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer([x, y, z], [x, y, z])
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random axis, uniform angle in [0, 2 pi) (proper rotation)."""
    axis = rng.normal(size=3)
    while np.linalg.norm(axis) < 1e-12:
        axis = rng.normal(size=3)
    return rotation_matrix_3d(axis, rng.uniform(0.0, 2.0 * np.pi))


def mirror_matrix(normal) -> np.ndarray:
    """Reflection through the plane with the given normal (improper)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def rotation_z_to(direction) -> np.ndarray:
    """A rotation matrix taking the z axis to ``direction``."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, u))
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    angle = np.arctan2(s, c)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) * np.cos(angle) + np.sin(angle) * cross + (
        1 - np.cos(angle)
    ) * np.outer(axis, axis)


def real_sph_harm(l: int, m: int, unit_vectors: np.ndarray) -> np.ndarray:
    """Real spherical harmonics on an (N, 3) array of unit vectors."""
    v = np.asarray(unit_vectors, dtype=float)
    theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))  # polar
    phi = np.arctan2(v[:, 1], v[:, 0])  # azimuthal
    # scipy: sph_harm(m, l, azimuth, polar)
    if m == 0:
        return np.real(sph_harm(0, l, phi, theta))
    y_abs_m = sph_harm(abs(m), l, phi, theta)
    if m > 0:
        return np.sqrt(2.0) * (-1.0) ** m * np.real(y_abs_m)
    return np.sqrt(2.0) * (-1.0) ** m * np.imag(y_abs_m)


def real_wigner_d(l: int, rot: np.ndarray) -> np.ndarray:
    """Real-spherical-harmonic rotation block D^(l) for arbitrary l.

    The relation Y_lm(R u) = sum_m' D_mm' Y_lm'(u) is linear and exact, so a
    least-squares solve on random sample directions recovers D to machine
    precision (verified against the closed-form l <= 2 blocks and the l = 3
    representation property by the validation suite). ``rot`` may be ANY
    orthogonal matrix; for improper ones this returns the O(3)
    representation block, which is what the parity tests use."""
    if l == 0:
        return np.eye(1)
    rng = np.random.default_rng(987654321 + l)  # fixed: sampling is exact
    directions = rng.normal(size=(16 * (2 * l + 1), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    y_ref = np.column_stack(
        [real_sph_harm(l, m, directions) for m in range(-l, l + 1)]
    )
    y_rot = np.column_stack(
        [real_sph_harm(l, m, directions @ np.asarray(rot).T) for m in range(-l, l + 1)]
    )
    d_matrix, *_ = np.linalg.lstsq(y_ref, y_rot, rcond=None)
    return d_matrix.T


def real_sh_rotation_matrix(l: int, rot: np.ndarray) -> np.ndarray:
    """Independent rotation block D^(l) for l <= 2 (self-check reference).

    l = 0: trivial. l = 1: closed-form permutation conjugation of the 3x3
    rotation matrix (real Y_1m ~ (y, z, x)). l = 2: solved exactly from
    sampled harmonics with a fixed sample set.
    """
    rot = np.asarray(rot, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError("rot must be a 3x3 rotation matrix.")
    if l == 0:
        return np.eye(1)
    if l == 1:
        perm = [1, 2, 0]  # (y, z, x) ordering of real Y_1m
        return rot[np.ix_(perm, perm)]
    if l == 2:
        rng = np.random.default_rng(12345)  # fixed: sampling is exact anyway
        directions = rng.normal(size=(64, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        y_ref = np.column_stack(
            [real_sph_harm(2, m, directions) for m in range(-2, 3)]
        )
        y_rot = np.column_stack(
            [real_sph_harm(2, m, directions @ rot.T) for m in range(-2, 3)]
        )
        d_matrix, *_ = np.linalg.lstsq(y_ref, y_rot, rcond=None)
        return d_matrix.T
    raise NotImplementedError("use real_wigner_d for l > 2.")
