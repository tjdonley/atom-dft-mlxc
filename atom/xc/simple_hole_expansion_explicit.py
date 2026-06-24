"""Explicit (operator-free) reference for the *direct-expansion* SIMPLE exchange hole.

This is the reference implementation of the exchange-hole functional sketched in
``SIMPLE_hole_expansion.txt``: rather than fixing the spherically-averaged hole to one
universal HEG envelope ``S(zeta u)`` scaled by a single on-top scale ``zeta`` (as in
``atom.xc.simple_hole.SIMPLE_HOLE``), the hole *monopole* is expanded **directly** in the
scale-free SIMPLE Dirichlet--Bessel basis,

    n_x(r0, u) = ntilde(u; r0) = sum_n rhotilde_{n00}(r0) R_{n0}(u),                 (expand)

    R_{n0}(u) = k_n sqrt(2/R_c) j_0(k_n u),   k_n = (n+1) pi / R_c,                  (basis)

so the hole is a *vector* of coefficients rhotilde_{n00} (the "hole" descriptors, tilde to
distinguish them from the density's SIMPLE descriptors rho_{nlm}).

Because the exchange energy needs only the spherical (l=0) average of the hole weighted by
1/u, only the monopole channels of the hole reach the energy, and the u-integral collapses
to a **sum** over closed-form per-basis-function constants:

    a_n = int_0^{R_c} R_{n0}(u) u^2 du     (enclosed charge of basis function n)       (a_n)
    b_n = int_0^{R_c} R_{n0}(u) u   du     (its self-Coulomb)                          (b_n)

    eps_x(r0) = 1/2 int ntilde(u)/u d^3u = 1/2 * 4 pi * sum_n rhotilde_n b_n           (eps)

with the two exact constraints

    sum rule:  int ntilde d^3u = 4 pi sum_n rhotilde_n a_n = -1                        (sum)
    on-top:    ntilde(0) = sum_n rhotilde_n R_{n0}(0) = -rho(r0)/2 .                    (ontop)

Closed forms (l=0 zeros z_{n0} = (n+1) pi are exact):

    a_n = A_n(R_c)/sqrt(4 pi),   A_n = (-1)^n sqrt(8 pi) R_c^{3/2}/((n+1) pi)   [bessel.a_n_closed_form]
        = (-1)^n sqrt(2) R_c^{3/2}/((n+1) pi)
    b_n = sqrt(2/R_c) [1 - cos((n+1) pi)]/k_n = sqrt(2/R_c) (1 + (-1)^n)/k_n   (odd n vanish)
    R_{n0}(0) = k_n sqrt(2/R_c)   (since j_0(0)=1)

This module maps line-for-line to the equations above and is the reference against which the
production convolutional implementation (``atom.xc.simple_hole_expansion.SIMPLE_HOLE_EXPANSION``)
is validated. The parameter-free map rhotilde_n <- M({rho_{nlm}}) lives in that module; here we
expose the representation primitives and the limit checks (HEG -> LDA).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn

from ..descriptors.simple.bessel import a_n_closed_form, radial_gauss_grid


# --------------------------------------------------------------------------- #
# Monopole basis and its closed-form Coulomb/charge moments
# --------------------------------------------------------------------------- #
def radial_basis(n, u, r_c):
    """Orthonormal SIMPLE monopole basis R_{n0}(u) = k_n sqrt(2/R_c) j_0(k_n u) [Eq. (basis)].

    Identical to ``atom.xc.simple_hole._radial_basis``; orthonormal under int_0^{R_c} . u^2 du.
    """
    k_n = (n + 1) * np.pi / r_c
    return k_n * np.sqrt(2.0 / r_c) * spherical_jn(0, k_n * u)


def radial_basis_at_origin(n_channels, r_c):
    """R_{n0}(0) = k_n sqrt(2/R_c) for n = 0..n_channels-1 [Eq. (ontop)]."""
    k = (np.arange(n_channels) + 1) * np.pi / r_c
    return k * np.sqrt(2.0 / r_c)


def charge_moments(n_channels, r_c):
    """a_n = int_0^{R_c} R_{n0} u^2 du for n = 0..n_channels-1 [Eq. (a_n)], closed form."""
    return a_n_closed_form(n_channels - 1, r_c) / np.sqrt(4.0 * np.pi)


def coulomb_moments(n_channels, r_c):
    """b_n = int_0^{R_c} R_{n0} u du for n = 0..n_channels-1 [Eq. (b_n)], closed form.

    b_n = sqrt(2/R_c) (1 + (-1)^n) / k_n; odd n vanish exactly.
    """
    n = np.arange(n_channels)
    k = (n + 1) * np.pi / r_c
    return np.sqrt(2.0 / r_c) * (1.0 + (-1.0) ** n) / k


def coulomb_moments_quad(n_channels, r_c, nu=512):
    """b_n by Gauss--Legendre quadrature (composite grid); reference for the closed form."""
    quad = radial_gauss_grid(r_c, n_points=nu, n_panels=4)
    u, w = quad.nodes, quad.weights
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Rb = np.array([radial_basis(n, u, r_c) for n in range(n_channels)])  # (n_channels, nu)
        return Rb @ (u * w)


def charge_moments_quad(n_channels, r_c, nu=512):
    """a_n by quadrature; reference for the closed form."""
    quad = radial_gauss_grid(r_c, n_points=nu, n_panels=4)
    u, w = quad.nodes, quad.weights
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Rb = np.array([radial_basis(n, u, r_c) for n in range(n_channels)])
        return Rb @ (u ** 2 * w)


# --------------------------------------------------------------------------- #
# Projection of an arbitrary spherical hole profile onto the monopole basis
# --------------------------------------------------------------------------- #
def project_hole(hole_profile, r_c, n_channels, nu=512):
    """Project a spherical hole profile ntilde(u) onto R_{n0}: rhotilde_n = int ntilde R_{n0} u^2 du.

    Parameters
    ----------
    hole_profile : callable(u) -> ntilde(u)
        The spherically-averaged exchange hole as a function of displacement magnitude u.
    r_c, n_channels, nu
        Window radius, number of monopole channels, quadrature nodes.

    Returns the coefficient vector rhotilde (shape (n_channels,)). Because the basis is
    orthonormal under u^2 du, this is the least-squares (and exact, in the limit n_channels
    -> inf) representation of the hole.
    """
    quad = radial_gauss_grid(r_c, n_points=nu, n_panels=4)
    u, w = quad.nodes, quad.weights
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Rb = np.array([radial_basis(n, u, r_c) for n in range(n_channels)])  # (n_channels, nu)
        profile = np.asarray(hole_profile(u), dtype=float)
        return Rb @ (profile * u ** 2 * w)


# --------------------------------------------------------------------------- #
# Energy / constraints from coefficients
# --------------------------------------------------------------------------- #
def eps_from_coeffs(coeffs, b):
    """Exchange energy per particle eps_x = 1/2 * 4 pi * sum_n rhotilde_n b_n [Eq. (eps)]."""
    return 0.5 * 4.0 * np.pi * float(np.dot(coeffs, b))


def enclosed_charge(coeffs, a):
    """Enclosed charge int ntilde d^3u = 4 pi sum_n rhotilde_n a_n [Eq. (sum)] (should -> -1)."""
    return 4.0 * np.pi * float(np.dot(coeffs, a))


def on_top(coeffs, r0_vals):
    """On-top value ntilde(0) = sum_n rhotilde_n R_{n0}(0) [Eq. (ontop)] (should -> -rho/2)."""
    return float(np.dot(coeffs, r0_vals))


# --------------------------------------------------------------------------- #
# HEG reference hole and LDA exchange
# --------------------------------------------------------------------------- #
def heg_envelope(x):
    """HEG exchange-hole envelope S(x) = [3 j_1(x)/x]^2 (-> 1 as x -> 0)."""
    x = np.maximum(np.asarray(x, float), 1e-12)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return (3.0 * spherical_jn(1, x) / x) ** 2


def heg_hole(rho):
    """Exact spin-summed HEG exchange hole n_x(u) = -(rho/2) S(k_F u), k_F = (3 pi^2 rho)^{1/3}.

    On-top n_x(0) = -rho/2 and (in full space) int n_x d^3u = -1."""
    rho = float(rho)
    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    return lambda u: -0.5 * rho * heg_envelope(k_f * u)


def lda_exchange_per_particle(rho):
    """eps_x^unif = -3/(4 pi) (3 pi^2 rho)^{1/3}."""
    rho = np.maximum(np.asarray(rho, float), 1e-12)
    return -3.0 / (4.0 * np.pi) * (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)


# =========================================================================== #
# Phase B: the parameter-free map  M: {density monopole} -> {hole coeffs}
#
# Two anchors, blended by a smooth switch on the PER-SPIN enclosed charge, then
# projected onto the two exact constraints (sum rule, on-top):
#
#   HEG anchor  rhotilde^HEG = project(-(rho/2) S(k_F u))            (lambda -> 0, bulk)
#   FA  anchor  rhotilde^FA  = -C_n / Q   (density-following)        (lambda -> 1, few e/spin)
#   rhotilde = (1-lambda) rhotilde^HEG + lambda rhotilde^FA
#
# THE SWITCH IS PER SPIN. Exchange is a same-spin interaction, so the one-electron
# (self-interaction-free) limit is one electron PER SPIN: Q_sigma = Q/2 <= 1, i.e.
# Q_total <= 2. This is why spin-paired He (Q_total=2 -> Q_sigma=1) is in the SIC/
# density-following limit and is reproduced essentially exactly -- keying the switch on
# the *total* charge instead wrongly calls He "bulk" and collapses it to LDA.
#
# The Fermi-Amaldi anchor -C_n/Q is the universal density-following hole: int = -1 by
# construction, and its on-top -rho0/Q gives -rho0 for one electron (Q=1, fully
# polarized) and -rho0/2 for spin-paired He (Q=2) -- the spin factor falls out of /Q.
# On-top target blends (1-lambda)(-rho0/2) [HEG pair] + lambda(-rho0/Q) [FA].
# =========================================================================== #
def density_coeffs(density_profile, r_c, n_channels, nu=512):
    """Monopole coefficients C_n = int rho_avg(u) R_{n0}(u) u^2 du of the local
    spherically-averaged density profile (the production C_n = P_n @ rho)."""
    return project_hole(density_profile, r_c, n_channels, nu=nu)


def heg_anchor(rho_on_top, r_c, n_channels, nu=512):
    """HEG hole coefficients at the local scale set by the on-top density rho(r0):
    project -(rho/2) S(k_F u), k_F = (3 pi^2 rho)^{1/3}. (lambda -> 0 limit.)"""
    return project_hole(heg_hole(rho_on_top), r_c, n_channels, nu=nu)


def _quintic_smoothstep(t):
    """C^2 smoothstep 6t^5 - 15t^4 + 10t^3 on [0,1] (0 outside via clip)."""
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10.0 + t * (-15.0 + 6.0 * t))


def enclosed_charge_switch(q_spin, q_lo=1.0, q_hi=2.0):
    """Smooth (C^2) monotone switch lambda(Q_sigma) on the PER-SPIN enclosed charge: 1 for
    Q_sigma <= q_lo (one electron per spin / SIC), 0 for Q_sigma >= q_hi (bulk, HEG), quintic
    Hermite in between. Exchange is same-spin, so the relevant charge is Q_sigma = Q_total/2
    (call this with Q/2): spin-paired He (Q_total=2) -> Q_sigma=1 -> SIC limit."""
    return 1.0 - _quintic_smoothstep((q_spin - q_lo) / (q_hi - q_lo))


def constraint_project(coeffs, a, r0_vals, sum_target=-1.0, ontop_target=None):
    """Least-norm correction so coeffs satisfy the two exact linear constraints:
        sum rule  4 pi (a . coeffs) = sum_target          (= -1)
        on-top    (r0_vals . coeffs) = ontop_target
    via  coeffs <- coeffs + A^T (A A^T)^{-1} (c - A coeffs),  A the 2xN constraint matrix.
    The min-norm shift spreads across channels and stays tiny when the anchors nearly
    satisfy the constraints already."""
    A = np.vstack([4.0 * np.pi * np.asarray(a), np.asarray(r0_vals)])  # (2, N)
    c = np.array([sum_target, ontop_target], dtype=float)
    resid = c - A @ coeffs
    gram = A @ A.T  # (2,2)
    return coeffs + A.T @ np.linalg.solve(gram, resid)


def map_coeffs(density_profile, r_c, n_channels, nu=512, return_diagnostics=False):
    """Parameter-free map: local density profile -> hole monopole coefficients rhotilde.

    rho_on_top = rho_avg(0); Q_window = 4 pi sum_n C_n a_n; the switch acts on the PER-SPIN
    charge Q/2 (exchange is same-spin); blend the HEG and Fermi-Amaldi (density-following)
    anchors; project onto the sum-rule and (lambda-interpolated) on-top constraints."""
    a = charge_moments(n_channels, r_c)
    r0 = radial_basis_at_origin(n_channels, r_c)
    C = density_coeffs(density_profile, r_c, n_channels, nu=nu)
    rho0 = float(np.atleast_1d(density_profile(np.array([0.0])))[0])
    q_window = 4.0 * np.pi * float(np.dot(C, a))
    q_safe = max(q_window, 1e-12)
    lam = float(enclosed_charge_switch(0.5 * q_window))      # per-spin switch (Q/2)

    coeffs_heg = heg_anchor(rho0, r_c, n_channels, nu=nu)
    coeffs_fa = -C / q_safe                                  # Fermi-Amaldi: density-following, int -> -1
    coeffs = (1.0 - lam) * coeffs_heg + lam * coeffs_fa

    ontop = (1.0 - lam) * (-0.5 * rho0) + lam * (-rho0 / q_safe)   # HEG pair -rho/2; FA -rho/Q
    coeffs = constraint_project(coeffs, a, r0, sum_target=-1.0, ontop_target=ontop)
    if return_diagnostics:
        return coeffs, {"lambda": lam, "Q_window": q_window, "Q_spin": 0.5 * q_window,
                        "rho0": rho0, "ontop": ontop}
    return coeffs


def eps_x_map(density_profile, r_c, n_channels, nu=512):
    """Exchange energy per particle from the parameter-free map."""
    b = coulomb_moments(n_channels, r_c)
    return eps_from_coeffs(map_coeffs(density_profile, r_c, n_channels, nu=nu), b)


# =========================================================================== #
# Phase F: learnable residual layer with the exact limits enforced by construction
#
# A data-driven correction to the parameter-free hole coefficients that CANNOT break
# the HEG (lambda=0) or one-electron (lambda=1) limits, regardless of the fitted weights:
#
#   rhotilde = rhotilde_paramfree + lambda(1-lambda) * (W_mat @ features) ; then constraint-project.
#
# The gate g = lambda(1-lambda) vanishes at both anchors (lambda in {0,1}), so the learned
# term is active only in the intermediate (inhomogeneous, partially-enclosed) regime -- exactly
# where the parameter-free map is weakest. The final 2-constraint projection keeps the sum rule
# and on-top exact. Features are rotation-invariant scalars (here: the enclosed charge Q and the
# reduced gradient s; in production, the SIMPLE power spectrum / bispectrum slot in unchanged).
# =========================================================================== #
def learnable_residual(features, weights, lam, a, r0_vals, n_channels):
    """Constrained learned correction to the hole coefficients.

    features : (n_feat,) rotation-invariant scalars at the point.
    weights  : (n_channels, n_feat) fitted matrix.
    lam      : enclosed-charge switch value at the point.
    Returns the gated, charge-/on-top-neutral coefficient correction (n_channels,)."""
    gate = lam * (1.0 - lam)
    delta = gate * (np.asarray(weights) @ np.asarray(features))      # (n_channels,)
    # project the correction itself to be charge- and on-top-NEUTRAL so it cannot move the
    # sum rule or the on-top value (limits stay exact); leaves the energy channel free.
    A = np.vstack([4.0 * np.pi * np.asarray(a), np.asarray(r0_vals)])   # (2, n_channels)
    Ginv = np.linalg.inv(A @ A.T)
    return delta - A.T @ (Ginv @ (A @ delta))
