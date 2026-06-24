"""Exact orbital-based exchange-hole reference (ground truth for the direct-expansion hole).

The exact (Kohn--Sham / Hartree--Fock) exchange hole around r0 is built from the occupied
radial orbitals via the spin-resolved one-particle density matrix:

    n_x(r0, u) = -(1/rho(r0)) sum_sigma < |rho1_sigma(r0, r0+u)|^2 >_Omega ,
    rho1_sigma(r, r') = sum_{i in occ,sigma} psi_i(r) psi_i(r') ,

and the exchange energy per particle is eps_x(r0) = 1/2 int n_x(r0,u)/u d^3u. This module
computes the *spherically-averaged* hole n_x(r0,u) (the monopole that the exchange energy
sees) and projects it onto the SIMPLE monopole basis to give the **target** coefficients
rhotilde^exact_{n00}(r0) for testing / fitting the direct-expansion map.

Conventions (verified against ``AtomicDFTSolver`` EXX, all-electron):
- The solver's ``orbitals`` array is the reduced radial wavefunction phi_i(r) = r R_i(r);
  define g_i(r) = phi_i(r)/r, so the density is rho(r) = sum_i occ_i g_i(r)^2 / (4 pi)
  (the Y_00^2 = 1/4pi is folded in). Verified to 1e-15.
- Spin-restricted convention (matching ``atom.xc.hf`` and the solver): occ_{i,sigma} = occ_i/2,
  summed over the two spins. For closed shells this is exact; it reproduces the solver's
  ``oep_exchange`` (He -1.02577, validated).

**Scope.** This implementation covers s-only occupied manifolds (l = 0: H, He, Be, ...),
where the orbital angular part is constant and the spherical average reduces to a 1D radial
integral. Manifolds with l > 0 (e.g. Ne 2p^6) require the spherical-harmonic addition theorem
(Legendre/Wigner machinery already present in ``atom.xc.hf``); that generalization is a
documented extension (see ``exchange_hole_s`` raises for l>0).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from .simple_hole_expansion_explicit import (
    charge_moments, coulomb_moments, radial_basis, radial_basis_at_origin,
)


# --------------------------------------------------------------------------- #
# Orbital extraction
# --------------------------------------------------------------------------- #
def extract_s_orbitals(result):
    """From an AtomicDFTSolver result dict, return (r_sorted, g_sorted, occupations).

    g_i(r) = phi_i(r)/r are the radial orbitals (Y_00 folded into the density via 1/4pi);
    rho = sum_i occ_i g_i^2 / (4 pi). Raises if any occupied orbital has l > 0 (out of scope).
    """
    r = np.asarray(result["quadrature_nodes"], float)
    orb = np.asarray(result["orbitals"], float)           # (n_grid, n_orb) = phi_i = r R_i
    occ = np.asarray(result["occupation_info"].occupations, float)
    l_values = np.asarray(result["occupation_info"].l_values, int)
    if np.any(l_values != 0):
        raise NotImplementedError(
            f"orbital_hole covers s-only manifolds; got l_values={l_values.tolist()}. "
            "l>0 needs the spherical-harmonic addition theorem (see module docstring)."
        )
    g = orb / r[:, None]                                   # (n_grid, n_orb)
    idx = np.argsort(r)
    return r[idx], g[idx], occ


def extract_orbitals(result):
    """All occupied orbitals (any l): return (r_sorted, g_sorted, occ, l_values).

    Same g_i(r) = phi_i(r)/r as ``extract_s_orbitals`` but keeps l>0 manifolds; the
    spherically-averaged density rho = sum_i occ_i g_i^2/(4 pi) holds for filled subshells
    of any l (the sum over m and the (2l+1) degeneracy cancel)."""
    r = np.asarray(result["quadrature_nodes"], float)
    orb = np.asarray(result["orbitals"], float)           # (n_grid, n_orb) = phi_i = r R_i
    occ = np.asarray(result["occupation_info"].occupations, float)
    l_values = np.asarray(result["occupation_info"].l_values, int)
    g = orb / r[:, None]
    idx = np.argsort(r)
    return r[idx], g[idx], occ, l_values


def _g_interp(r_sorted, g_sorted, x):
    """Interpolate the radial orbitals g_i onto points x; shape (..., n_orb). Zero past r_max."""
    x = np.asarray(x, float)
    out = np.empty(x.shape + (g_sorted.shape[1],))
    for i in range(g_sorted.shape[1]):
        out[..., i] = np.interp(np.clip(x, 0.0, r_sorted[-1]), r_sorted, g_sorted[:, i],
                                left=g_sorted[0, i], right=0.0)
    return out


# --------------------------------------------------------------------------- #
# Exact spherically-averaged exchange hole (s-only)
# --------------------------------------------------------------------------- #
def exchange_hole_s(r0, u, r_sorted, g_sorted, occ, n_mu=80):
    """Spherically-averaged exact exchange hole n_x(r0, u) for s-only orbitals.

    n_x(r0,u) = -(2/rho(r0)) < |rho1_sigma(r0, r0+u)|^2 >_Omega  (restricted: occ/2 per spin,
    x2 spins), with rho1_sigma(r0, r0+u) = sum_i (occ_i/2) g_i(r0) g_i(r') / (4 pi) and
    r' = |r0 + u u_hat|. The angular average over u_hat is (1/2) int_{-1}^1 . dmu with
    r'(mu) = sqrt(r0^2 + u^2 + 2 r0 u mu).

    r0 : scalar; u : array. Returns n_x(r0, u) (same shape as u).
    """
    u = np.atleast_1d(np.asarray(u, float))
    mu, wm = leggauss(n_mu)
    rp = np.sqrt(np.maximum(r0 ** 2 + u[:, None] ** 2 + 2.0 * r0 * u[:, None] * mu[None, :], 0.0))
    g_rp = _g_interp(r_sorted, g_sorted, rp)               # (nu, nmu, n_orb)
    g_r0 = _g_interp(r_sorted, g_sorted, np.array([r0]))[0]  # (n_orb,)
    rho1 = np.sum((occ / 2.0)[None, None, :] * g_r0[None, None, :] * g_rp, axis=-1) / (4.0 * np.pi)
    rho1_sq_ang = 0.5 * np.sum(wm[None, :] * rho1 ** 2, axis=1)   # < |rho1|^2 >_Omega  (nu,)
    rho0 = float(np.sum(occ * g_r0 ** 2) / (4.0 * np.pi))
    return -2.0 * rho1_sq_ang / max(rho0, 1e-30)


def exchange_hole(r0, u, r_sorted, g_sorted, occ, l_values, n_mu=80):
    """Spherically-averaged exact exchange hole n_x(r0, u) for *general* l, via the
    spherical-harmonic addition theorem.

    For a spherically-averaged atom the per-spin 1-RDM is
        rho1_sigma(r0, r') = (1/4 pi) sum_i (occ_i/2) g_i(r0) g_i(r') P_{l_i}(cos gamma),
    with r' = |r0 + u u_hat| = sqrt(r0^2 + u^2 + 2 r0 u mu) and the angle gamma between the
    vectors r0 and r' given by cos gamma = (r0 + u mu)/r'  (mu = cos<r0,u_hat>). The hole is
    n_x(r0,u) = -(2/rho0) <|rho1_sigma|^2>_Omega (restricted: occ/2 per spin, x2 spins). For
    l_i = 0 all P_l = 1 and this reduces to ``exchange_hole_s``.

    r0 : scalar; u : array. Returns n_x(r0, u) (same shape as u)."""
    u = np.atleast_1d(np.asarray(u, float))
    mu, wm = leggauss(n_mu)
    rp = np.sqrt(np.maximum(r0 ** 2 + u[:, None] ** 2 + 2.0 * r0 * u[:, None] * mu[None, :], 0.0))
    cg = np.clip((r0 + u[:, None] * mu[None, :]) / np.maximum(rp, 1e-30), -1.0, 1.0)  # cos gamma
    g_rp = _g_interp(r_sorted, g_sorted, rp)               # (nu, nmu, n_orb)
    g_r0 = _g_interp(r_sorted, g_sorted, np.array([r0]))[0]  # (n_orb,)
    rho1 = np.zeros(rp.shape)
    for i in range(len(occ)):
        Pl = eval_legendre(int(l_values[i]), cg) if l_values[i] > 0 else 1.0
        rho1 = rho1 + (occ[i] / 2.0) * g_r0[i] * g_rp[:, :, i] * Pl
    rho1 = rho1 / (4.0 * np.pi)
    rho1_sq_ang = 0.5 * np.sum(wm[None, :] * rho1 ** 2, axis=1)   # < |rho1|^2 >_Omega
    rho0 = float(np.sum(occ * g_r0 ** 2) / (4.0 * np.pi))
    return -2.0 * rho1_sq_ang / max(rho0, 1e-30)


def exact_eps_x_l(r0, r_sorted, g_sorted, occ, l_values, n_u=128, n_mu=80, u_max=None):
    """Exact eps_x(r0) = 1/2 * 4 pi int_0^{u_max} n_x(r0,u) u du via the general-l hole."""
    u_max = (r_sorted[-1] - 1e-6) if u_max is None else u_max
    xu, wu = leggauss(n_u)
    u = 0.5 * u_max * (xu + 1.0)
    wq = 0.5 * u_max * wu
    nx = exchange_hole(r0, u, r_sorted, g_sorted, occ, l_values, n_mu=n_mu)
    return 0.5 * 4.0 * np.pi * float(np.sum(wq * nx * u))


def exact_Ex_l(result, n_u=128, n_mu=64, r0_stride=1):
    """Total exact E_x = int rho(r0) eps_x(r0) d^3r0 from the general-l orbital hole
    (handles p/d manifolds: Ne, Na, Mg, ...)."""
    r = np.asarray(result["quadrature_nodes"], float)
    w = np.asarray(result["quadrature_weights"], float)
    rho = np.asarray(result["rho"], float)
    r_sorted, g_sorted, occ, l_values = extract_orbitals(result)
    sel = np.arange(0, len(r), r0_stride)
    ew = 4.0 * np.pi * r[sel] ** 2 * w[sel]
    if r0_stride > 1:
        ew = ew * (np.sum(4.0 * np.pi * r ** 2 * w) / np.sum(ew))
    eps = np.array([exact_eps_x_l(r[i], r_sorted, g_sorted, occ, l_values, n_u=n_u, n_mu=n_mu)
                    for i in sel])
    return float(np.sum(ew * rho[sel] * eps))


def project_exact_hole_l(r0, r_sorted, g_sorted, occ, l_values, r_c, n_channels, n_u=256, n_mu=80):
    """rhotilde^exact_{n00}(r0) = int_0^{r_c} n_x(r0,u) R_{n0}(u) u^2 du for general-l holes
    (the true monopole-hole coefficients; target for diagnosing/correcting the map)."""
    xu, wu = leggauss(n_u)
    u = 0.5 * r_c * (xu + 1.0)
    wq = 0.5 * r_c * wu
    nx = exchange_hole(r0, u, r_sorted, g_sorted, occ, l_values, n_mu=n_mu)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Rb = np.array([radial_basis(n, u, r_c) for n in range(n_channels)])
        return Rb @ (nx * u ** 2 * wq)


def on_top_density(r0, r_sorted, g_sorted, occ):
    """rho(r0) from the radial orbitals."""
    g_r0 = _g_interp(r_sorted, g_sorted, np.array([r0]))[0]
    return float(np.sum(occ * g_r0 ** 2) / (4.0 * np.pi))


def exact_eps_x(r0, r_sorted, g_sorted, occ, n_u=128, n_mu=80, u_max=None):
    """Exact exchange energy per particle eps_x(r0) = 1/2 int n_x(r0,u)/u d^3u
    = 1/2 * 4 pi int_0^{u_max} n_x(r0,u) u du, via the s-only hole."""
    u_max = (r_sorted[-1] - 1e-6) if u_max is None else u_max
    xu, wu = leggauss(n_u)
    u = 0.5 * u_max * (xu + 1.0)
    wq = 0.5 * u_max * wu
    nx = exchange_hole_s(r0, u, r_sorted, g_sorted, occ, n_mu=n_mu)
    return 0.5 * 4.0 * np.pi * float(np.sum(wq * nx * u))


def exact_Ex(result, n_u=128, n_mu=64, r0_stride=1):
    """Total exact exchange energy E_x = int rho(r0) eps_x(r0) d^3r0 from the orbital hole.

    r0_stride>1 subsamples the quadrature grid for speed (energy uses the native weights on
    the subsample, renormalized); stride=1 uses the full grid (most accurate)."""
    r = np.asarray(result["quadrature_nodes"], float)
    w = np.asarray(result["quadrature_weights"], float)
    rho = np.asarray(result["rho"], float)
    r_sorted, g_sorted, occ = extract_s_orbitals(result)
    sel = np.arange(0, len(r), r0_stride)
    ew = 4.0 * np.pi * r[sel] ** 2 * w[sel]
    if r0_stride > 1:
        ew = ew * (np.sum(4.0 * np.pi * r ** 2 * w) / np.sum(ew))  # renormalize subsample
    eps = np.array([exact_eps_x(r[i], r_sorted, g_sorted, occ, n_u=n_u, n_mu=n_mu) for i in sel])
    return float(np.sum(ew * rho[sel] * eps))


# --------------------------------------------------------------------------- #
# Project the exact hole onto the SIMPLE monopole basis -> target coefficients
# --------------------------------------------------------------------------- #
def project_exact_hole(r0, r_sorted, g_sorted, occ, r_c, n_channels, n_u=256, n_mu=80):
    """rhotilde^exact_{n00}(r0) = int_0^{r_c} n_x(r0,u) R_{n0}(u) u^2 du -- the exact hole's
    coefficients in the SIMPLE monopole basis (target for the map)."""
    xu, wu = leggauss(n_u)
    u = 0.5 * r_c * (xu + 1.0)
    wq = 0.5 * r_c * wu
    nx = exchange_hole_s(r0, u, r_sorted, g_sorted, occ, n_mu=n_mu)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Rb = np.array([radial_basis(n, u, r_c) for n in range(n_channels)])
        return Rb @ (nx * u ** 2 * wq)


# --------------------------------------------------------------------------- #
# Analytic hydrogenic 1s reference (validates the angular-average machinery)
# --------------------------------------------------------------------------- #
def hydrogenic_1s_density(Z):
    """rho(r) = (Z^3/pi) exp(-2 Z r) for a hydrogenic 1s orbital (1 electron)."""
    return lambda r: (Z ** 3 / np.pi) * np.exp(-2.0 * Z * np.atleast_1d(r))


def spherical_avg_hydrogenic_1s(Z, r0, u):
    """Closed-form < rho(r0+u) >_Omega for the hydrogenic 1s density, used to validate the
    numerical angular average. < rho >_Omega = (1/(2 r0 u)) int_{|r0-u|}^{r0+u} rho(r') r' dr',
    with int r' e^{-a r'} dr' = -e^{-a r'}(r'/a + 1/a^2),  a = 2 Z."""
    u = np.atleast_1d(np.asarray(u, float))
    a = 2.0 * Z
    lo = np.abs(r0 - u)
    hi = r0 + u

    def _prim(x):  # antiderivative of r' e^{-a r'}
        return -np.exp(-a * x) * (x / a + 1.0 / a ** 2)

    integral = (Z ** 3 / np.pi) * (_prim(hi) - _prim(lo))
    return integral / np.maximum(2.0 * r0 * u, 1e-30)


def spherical_avg_radial(r_sorted, f_sorted, r0, u):
    """Numerical < f(r0+u) >_Omega for a radial function f given on a sorted grid, via
    the cumulative M(r) = int_0^r f(r') r' dr' (trapezoid):  < f > = (M(hi)-M(lo))/(2 r0 u)."""
    u = np.atleast_1d(np.asarray(u, float))
    integrand = f_sorted * r_sorted
    M = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_sorted))])
    Mlo = np.interp(np.clip(np.abs(r0 - u), 0, r_sorted[-1]), r_sorted, M)
    Mhi = np.interp(np.clip(r0 + u, 0, r_sorted[-1]), r_sorted, M)
    return (Mhi - Mlo) / np.maximum(2.0 * r0 * u, 1e-30)
