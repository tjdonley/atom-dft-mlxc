"""Spherical Bessel radial basis on a ball of radius ``r_c`` (Dirichlet
boundary), plus the radial quadrature used by the continuum reference.

Conventions (Sec. 2 of docs/SIMPLE/SIMPLE.tex):

    R_nl(r) = N_nl * j_l(k_nl r),    k_nl = z_nl / r_c,    j_l(z_nl) = 0
    N_nl   = [ (r_c^3 / 2) * j_{l+1}(z_nl)^2 ]^{-1/2}
    =>  int_0^{r_c} r^2 R_nl(r) R_n'l(r) dr = delta_nn'

For l = 0 the zeros are exact: z_n0 = (n+1) * pi. Monopole integrals of the
full kernel K_n00 = R_n0 * Y_00:

    A_n(R) = int_{|r| <= R} K_n00(r) d^3r
           = (-1)^n * sqrt(8 pi) * R^{3/2} / ((n+1) * pi)

Note the alternating sign: the homogeneous-electron-gas reference values
``d_n00 = A_n / A_0 = (-1)^n / (n+1)`` are signed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import spherical_jn


def spherical_jn_zeros(l: int, n_zeros: int) -> np.ndarray:
    """First ``n_zeros`` positive zeros of the spherical Bessel function j_l.

    For l = 0 the zeros are exactly ``(n+1) * pi``. For l > 0 the zeros are
    bracketed by the interlacing property ``z_{l-1,n} < z_{l,n} < z_{l-1,n+1}``
    and refined with Brent's method, building up level by level from l = 0.
    """
    if l < 0:
        raise ValueError("l must be non-negative.")
    if n_zeros < 1:
        raise ValueError("n_zeros must be at least 1.")
    zeros = np.pi * np.arange(1, n_zeros + l + 1)
    for level in range(1, l + 1):
        f = lambda x, lev=level: spherical_jn(lev, x)  # noqa: E731
        zeros = np.array(
            [brentq(f, zeros[n], zeros[n + 1]) for n in range(len(zeros) - 1)]
        )
    return zeros


def a_n_closed_form(n_max: int, r_c: float) -> np.ndarray:
    """Closed-form monopole integrals A_n(r_c) for n = 0..n_max."""
    n = np.arange(n_max + 1)
    return (-1.0) ** n * np.sqrt(8.0 * np.pi) * r_c ** 1.5 / ((n + 1) * np.pi)


class RadialBesselBasis:
    """Orthonormal spherical Bessel basis ``R_nl`` on ``[0, r_c]``.

    Parameters
    ----------
    n_max : int
        Highest radial index; the basis holds ``n_max + 1`` functions per l.
    l_max : int
        Highest angular momentum channel.
    r_c : float
        Cutoff radius (Dirichlet boundary: every R_nl vanishes at r_c).
    """

    def __init__(self, n_max: int, l_max: int, r_c: float):
        if n_max < 0:
            raise ValueError("n_max must be non-negative.")
        if l_max < 0:
            raise ValueError("l_max must be non-negative.")
        if r_c <= 0.0:
            raise ValueError("r_c must be positive.")

        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.r_c = float(r_c)

        self.zeros = np.empty((self.l_max + 1, self.n_max + 1))
        for l in range(self.l_max + 1):
            self.zeros[l] = spherical_jn_zeros(l, self.n_max + 1)
        self.k = self.zeros / self.r_c

        self.norms = np.empty_like(self.zeros)
        for l in range(self.l_max + 1):
            j_next = spherical_jn(l + 1, self.zeros[l])
            self.norms[l] = 1.0 / np.sqrt(0.5 * self.r_c**3 * j_next**2)

    def evaluate(self, l: int, r: np.ndarray) -> np.ndarray:
        """Values R_nl(r) for all n; shape ``(n_max + 1, r.size)``.

        Zero outside the ball (r > r_c).
        """
        self._check_l(l)
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty((self.n_max + 1, r.size))
        inside = r <= self.r_c
        for n in range(self.n_max + 1):
            out[n] = np.where(
                inside,
                self.norms[l, n] * spherical_jn(l, self.k[l, n] * r),
                0.0,
            )
        return out

    def derivative(self, l: int, r: np.ndarray) -> np.ndarray:
        """Radial derivatives dR_nl/dr for all n; shape ``(n_max + 1, r.size)``."""
        self._check_l(l)
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty((self.n_max + 1, r.size))
        inside = r <= self.r_c
        for n in range(self.n_max + 1):
            k = self.k[l, n]
            out[n] = np.where(
                inside,
                self.norms[l, n] * k * spherical_jn(l, k * r, derivative=True),
                0.0,
            )
        return out

    def _check_l(self, l: int) -> None:
        if not 0 <= l <= self.l_max:
            raise ValueError(f"l must be in [0, {self.l_max}], got {l}.")


@dataclass(frozen=True)
class RadialQuadrature:
    """Nodes and weights for ``int_0^{r_c} f(r) dr ~ sum_i w_i f(r_i)``."""

    nodes: np.ndarray
    weights: np.ndarray
    r_c: float


def radial_gauss_grid(
    r_c: float,
    n_points: int = 64,
    n_panels: int = 4,
) -> RadialQuadrature:
    """Composite Gauss-Legendre rule on ``[0, r_c]``.

    The total point count is split evenly across ``n_panels`` uniform
    panels; panel boundaries are interior to the smooth integrands so the
    composite rule retains spectral accuracy for the oscillatory
    ``j_l(k_n r)`` integrands.
    """
    if r_c <= 0.0:
        raise ValueError("r_c must be positive.")
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")
    if n_panels < 1:
        raise ValueError("n_panels must be at least 1.")
    per_panel = max(1, int(np.ceil(n_points / n_panels)))
    ref_nodes, ref_weights = leggauss(per_panel)
    boundaries = np.linspace(0.0, r_c, n_panels + 1)
    nodes, weights = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        half = 0.5 * (right - left)
        nodes.append(0.5 * (left + right) + half * ref_nodes)
        weights.append(half * ref_weights)
    return RadialQuadrature(
        nodes=np.concatenate(nodes),
        weights=np.concatenate(weights),
        r_c=float(r_c),
    )
