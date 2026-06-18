"""Explicit (direct-integral) reference for the SIMPLE exchange-hole functional.

This is the *explicit version* of the hole functional: it evaluates the exchange
energy per electron straight from the exchange-hole integral [Eq. (exact-hole),
Eq. (eps-x) of SIMPLE-Xhole-writeup], with no convolutional/operator machinery. It
maps line-for-line to the equations and is the reference against which the
*production* convolutional implementation ``atom.xc.simple_hole.SIMPLE_HOLE``
(fixed monopole operators + discrete adjoint) is validated
(``tests/simple/test_simple_hole.py``).

Model hole around r0 (spin-unpolarized): the local density profile modulated by the
HEG envelope S(x) = [3 j_1(x)/x]^2, interacting through 1/u,

    n_x(r0, u) = -W(r0) rho(r0+u) S(zeta(r0) u) .

The two local scalars are fixed by exact sum rules, parameter-free:

    on-top  n_x(r0,0) = -rho(r0)/2     => W = 1/2           (S(0)=1)
    on-top sum rule    Q_S(zeta) = 2   => zeta              [Eq. (eps-x)]

and the exchange energy per electron is the operator self-energy [Eq. (eps-x)]

    eps_x(r0) = -1/2 Phi_S(zeta)/Q_S(zeta),
    Q_S(zeta)   = int rho(r0+u) S(zeta u) d^3u    (electrons enclosed),
    Phi_S(zeta) = int rho(r0+u) S(zeta u)/u d^3u  (its self-Coulomb).

Limits, by construction:
  * HEG (uniform rho): zeta -> k_F -> the exact HEG hole -> eps_x = eps_x^unif (LDA).
  * Fermi-Amaldi: where the window holds < one pair, Q_S(zeta_min) <= 2, the hole
    follows the full density and eps_x -> the self-interaction correction
    (exact for one electron per spin); accuracy improves as the window R_c grows.

Only the spherical (l=0) average of the density enters [Eq. (coulomb)]; here it is
supplied directly as ``prof0(u)`` (e.g. the l=0 axial multipole profile).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import spherical_jn


def lda_exchange_per_particle(rho):
    """Spin-unpolarized uniform-gas exchange energy per particle,
    eps_x^unif = -3/(4 pi) (3 pi^2 rho)^{1/3} [Eq. (ex-gga)]."""
    rho = np.maximum(np.asarray(rho, float), 1e-12)
    return -3.0 / (4.0 * np.pi) * (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)


def envelope(x):
    """HEG exchange-hole envelope S(x) = [3 j_1(x)/x]^2 = [j_0(x)+j_2(x)]^2 (-> 1 as x->0)."""
    x = np.maximum(np.asarray(x, float), 1e-12)
    return (3.0 * spherical_jn(1, x) / x) ** 2


def hole_solve(prof0, rc, nu=400):
    """Explicit hole self-energy at one center [Eq. (eps-x)].

    Parameters
    ----------
    prof0 : callable
        Spherical (l=0) density profile rho_avg(u; r0): the angular average of
        rho(r0+u) at displacement magnitude u.
    rc : float
        Window radius for the u-integral.
    nu : int
        Gauss-Legendre nodes on [0, rc].

    Returns ``(eps_x, zeta, W)``: the exchange energy per electron, the on-top scale
    zeta from Q_S(zeta)=2 (Fermi-Amaldi limit zeta->zeta_min when the window cannot
    hold a full hole), and the normalization W = 1/Q_S.
    """
    xu, wu = leggauss(nu)
    u = 0.5 * rc * (xu + 1.0)
    wq = 0.5 * rc * wu                                    # u in [0, rc]
    rho_ang = 2.0 * prof0(u)                              # int_{-1}^1 rho dmu = 2 rho_avg

    def Q_S(zeta):                                        # int rho(r0+u) S(zeta u) d^3u
        return 2.0 * np.pi * np.sum(wq * u ** 2 * rho_ang * envelope(zeta * u))

    def Phi_S(zeta):                                      # int rho(r0+u) S(zeta u)/u d^3u
        return 2.0 * np.pi * np.sum(wq * u * rho_ang * envelope(zeta * u))

    if Q_S(1e-3) <= 2.0:                                  # window < one pair -> Fermi-Amaldi
        zeta = 1e-3
    else:
        zeta = brentq(lambda z: Q_S(z) - 2.0, 1e-3, 100.0, xtol=1e-6)
    qe = Q_S(zeta)
    eps_x = -0.5 * Phi_S(zeta) / qe if qe > 1e-30 else np.nan
    return eps_x, zeta, (1.0 / qe if qe > 1e-30 else np.nan)


def hole_ex(prof0, rc, nu=400):
    """Exchange energy per particle eps_x(r0) [Eq. (eps-x)]; see ``hole_solve``."""
    return hole_solve(prof0, rc, nu)[0]
