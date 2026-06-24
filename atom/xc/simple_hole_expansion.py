"""Production direct-expansion SIMPLE exchange-hole functional (self-consistent).

The spherically-averaged exchange hole is expanded directly in the SIMPLE monopole basis,
n_x(r0,u) = sum_n rhotilde_{n00}(r0) R_{n0}(u), with the coefficients produced by a
parameter-free map from the local density monopole projections C_n = P_n @ rho. See
``simple_hole_expansion_explicit`` for the operator-free reference and the equations, and the
Phase-A/B/C reports under ``reports/hole_expansion/``.

This subclasses ``SIMPLE_HOLE`` to reuse, unchanged:
  * the fixed monopole window operators P_n (``build_window_operator``),
  * the discrete-adjoint potential and gauge fix (``compute_xc``, ``_apply_gauge``).
Only the energy-density kernel ``_eps_from_coeffs`` differs: instead of a single on-top scale
zeta fixed by Q_S(zeta)=2, it runs the direct-expansion map.

Map (a function of C and the local on-top density rho0 = rho(r0)):
  Cm   = C / (4 pi)                          # explicit (project_hole) convention
  Q    = 4 pi sum_n Cm_n a_n                 # total enclosed density charge
  lam  = switch(Q/2)                         # PER-SPIN: 1 (Q<=2, <=1 e/spin) -> 0 (Q>=4, HEG)
  rhotilde = (1-lam) rhotilde^HEG(rho0) + lam (-Cm/Q)      # HEG <-> Fermi-Amaldi anchors
  rhotilde <- project to {sum rule = -1, on-top = (1-lam)(-rho0/2) + lam(-rho0/Q)}
  eps_x    = 1/2 * 4 pi * sum_n rhotilde_n b_n

Exchange is a same-spin interaction, so the one-electron (self-interaction-free) limit is one
electron PER SPIN (Q/2 <= 1). Keying the switch on Q/2 puts spin-paired He (Q=2, Q/2=1) in the
density-following limit -> He exchange is reproduced essentially exactly. The Fermi-Amaldi
anchor -Cm/Q integrates to -1 and its on-top -rho0/Q gives -rho0 for one electron (Q=1) and
-rho0/2 for spin-paired He (Q=2): the spin factor falls out of /Q.

rho0 is the *actual* density rho(r0), not reconstructed from C: the on-top reconstruction
sum_n Cm_n R_{n0}(0) is an ill-conditioned alternating series (Gibbs-like). Because eps_x then
depends on rho explicitly (through rho0), ``compute_xc`` is overridden to add the explicit
rho-derivative term to the discrete-adjoint potential:
  v_x = eps + rho * d eps/d rho0 + sum_n P_n^T[ew rho d eps/d C_n] / ew .
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.special import spherical_jn

from ..descriptors.simple.bessel import RadialBesselBasis
from ..descriptors.simple.derivatives import build_spectral_gradient_operator
from ..descriptors.simple.params import R_C as _PIPELINE_R_C
from ..descriptors.simple.pipeline import transfer_matrix
from .evaluator import DensityData, XCParameters, XCPotentialData
from .simple_hole import SIMPLE_HOLE, SIMPLEHOLEParameters, _bound
from .simple_hole_expansion_explicit import enclosed_charge_switch

_SIX2_3 = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
_GEA2 = 10.0 / 81.0   # second-order gradient-expansion coefficient F_x -> 1 + (10/81) s^2
_X_WINDOW = 8.0       # dimensionless hole window X = k_F R_ad (the implicit scale lock)


def _envelope(x):
    """HEG exchange-hole envelope S(x) = [3 j_1(x)/x]^2."""
    x = np.maximum(np.asarray(x, float), 1e-12)
    return (3.0 * spherical_jn(1, x) / x) ** 2


@dataclass
class SIMPLEHOLEEXPParameters(SIMPLEHOLEParameters):
    """Scale-free hole settings (SIMPLE adaptive-radius frame; prior-SF normalization).

    ``n_channels`` = n_in: the fixed-R_c window resolution used to project the density.
    ``n_out`` = the adaptive-radius basis (the exposed SIMPLE feature count). The implicit
    adaptive radius R_ad = min(x_window/k_F(rho), R_c) makes the hole scale-free, so n_out=10
    resolves it. The SIMPLE ``transfer_matrix`` re-expresses the n_in window coefficients on the
    n_out adaptive basis in closed form. The sum rule is enforced by the on-top scale (the
    Q_S=2 enclosed-charge inversion), and c_ad is used only through the damped envelope
    contractions g(eta).c_ad / h(eta).c_ad (regularization-by-contraction, differentiable)."""
    functional_name: str = "SIMPLE_HOLE_EXPANSION"
    # r_c MUST equal the pipeline R_C: transfer_matrix builds the input (window) basis with the
    # global R_C, so the window operators and the transfer are consistent only when r_c == R_C.
    r_c: float = _PIPELINE_R_C      # = 3 Angstrom in bohr (~5.669); the canonical SIMPLE window
    n_channels: int = 20            # n_in: fixed-R_c projection resolution
    n_out: int = 10                 # adaptive-radius hole basis (exposed feature count)
    x_window: float = _X_WINDOW     # dimensionless hole window X = k_F R_ad
    n_rad: int = 48                 # R_ad-grid for the precomputed transfer matrices
    n_eta: int = 600                # eta-grid for the universal envelope tables g(eta), h(eta)
    eta_max: float = 60.0


class SIMPLE_HOLE_EXPANSION(SIMPLE_HOLE):
    """Exchange-only scale-free SIMPLE hole, self-consistent and parameter-free.

    Project the density to n_in fixed-R_c window coefficients C; set the implicit adaptive
    radius R_ad = min(X/k_F(rho0), R_c); transfer c_ad = T(R_ad) @ C onto the n_out adaptive
    basis. The exchange self-energy follows the prior SF normalization -- the hole self-energy is
    a universal contraction of c_ad with the envelope-projection tables g(eta), h(eta), and the
    on-top scale eta* is fixed by the enclosed-charge sum rule Q_S(eta*) = 2 (Fermi-Amaldi branch
    where the window holds < one pair). This (i) enforces int n_x = -1 through the SCALE (not a
    min-norm coefficient projection -> no overshoot), and (ii) touches c_ad ONLY through the
    damped contractions g.c_ad, h.c_ad (so the transfer's high-frequency content is regularized
    by contraction -- differentiable, stable at R_c >= 8, no SVD). The direct-expansion
    coefficient flexibility (gradient/iso-orbital corrections) layers on top of this base."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        p = self.params
        self._n_in = p.n_channels
        self._n_out = int(getattr(p, "n_out", 10))
        self._X = float(getattr(p, "x_window", _X_WINDOW))
        # universal envelope-projection tables on the unit-window adaptive basis R_m^(1) [0,1]:
        #   g_m(eta) = int_0^1 R_m^(1)(t) S(eta t) t^2 dt   (enclosed charge per channel)
        #   h_m(eta) = int_0^1 R_m^(1)(t) S(eta t) t   dt   (self-Coulomb per channel)
        basis = RadialBesselBasis(self._n_out - 1, 0, 1.0)
        xu, wu = np.polynomial.legendre.leggauss(600)
        t = 0.5 * (xu + 1.0); wt = 0.5 * wu
        Rb1 = basis.evaluate(0, t)                               # (n_out, nt)
        self._etas = np.linspace(1e-3, float(getattr(p, "eta_max", 60.0)),
                                 int(getattr(p, "n_eta", 600)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            S = _envelope(self._etas[:, None] * t[None, :])      # (n_eta, nt)
            self._G = (S * (wt * t ** 2)[None, :]) @ Rb1.T       # (n_eta, n_out)  g_m(eta)
            self._H = (S * (wt * t)[None, :]) @ Rb1.T            # (n_eta, n_out)  h_m(eta)
        # transfer matrices T(R_ad) = transfer_matrix(0, R_ad, n_out, n_in), precomputed + interp
        n_rad = int(getattr(p, "n_rad", 48))
        self._rad_grid = np.linspace(p.r_c / n_rad, p.r_c, n_rad)
        self._T_grid = np.stack([transfer_matrix(0, float(ra), self._n_out, self._n_in)
                                 for ra in self._rad_grid])      # (n_rad, n_out, n_in)

    def _R_ad(self, rho0):
        """Implicit adaptive radius R_ad = min(X/k_F(rho0), R_c) and the unclamped mask
        (R_ad < R_c); explicit and differentiable, dR_ad/drho = -R_ad/(3 rho) where unclamped."""
        kF = (3.0 * np.pi ** 2 * np.maximum(rho0, 1e-12)) ** (1.0 / 3.0)
        raw = self._X / np.maximum(kF, 1e-12)
        return np.minimum(raw, self.params.r_c), raw < self.params.r_c

    def _c_ad(self, C, R_ad):
        """Adaptive-radius density features c_ad (N, n_out) = T(R_ad) @ C, via interpolation.
        C carries the angular 4pi (the 3D window projection); it is kept here and the universal
        tables g, h provide the matching dimensionless factors (prior-SF convention)."""
        rg = self._rad_grid
        k = np.clip(np.searchsorted(rg, R_ad), 1, rg.size - 1)
        f = np.clip((R_ad - rg[k - 1]) / (rg[k] - rg[k - 1]), 0.0, 1.0)
        Tb = (1.0 - f)[:, None, None] * self._T_grid[k - 1] + f[:, None, None] * self._T_grid[k]
        return np.einsum('Noi,iN->No', Tb, C)                    # (N, n_out)

    def _ontop(self, Qc, Pc):
        """Vectorized on-top inversion: per point solve Q_S(eta)=2 (Q_S decreasing in eta;
        Fermi-Amaldi branch when the window holds < one pair), eps = -1/2 Phi_S/Q_S. Qc, Pc are
        (N, n_eta) on the shared eta grid. (Prior-SF normalization.)"""
        etas = self._etas; N = Qc.shape[0]; ar = np.arange(N)
        below = Qc <= 2.0
        fa = below[:, 0]                                          # max Q_S (eta->0) <= 2 -> FA
        crosses = below.any(axis=1)
        idx = np.clip(np.argmax(below, axis=1), 1, etas.size - 1)
        Qlo = Qc[ar, idx - 1]; Qhi = Qc[ar, idx]
        frac = np.clip((Qlo - 2.0) / (Qlo - Qhi + 1e-300), 0.0, 1.0)
        eps = -0.25 * (Pc[ar, idx - 1] + frac * (Pc[ar, idx] - Pc[ar, idx - 1]))   # Q_S=2 -> -Phi/4
        eps = np.where(fa, np.where(Qc[:, 0] > 1e-30, -0.5 * Pc[:, 0] / Qc[:, 0], 0.0), eps)
        eps = np.where(~crosses & ~fa, -0.25 * Pc[:, -1], eps)    # never crosses: clamp to grid end
        return eps

    def _eps_sf(self, C, R_ad):
        """Scale-free eps_x: c_ad = T(R_ad).C; Q_S = R_ad^{3/2} g.c_ad, Phi_S = R_ad^{1/2} h.c_ad;
        on-top Q_S(eta)=2 -> eps. c_ad enters ONLY through the damped contractions g.c_ad, h.c_ad."""
        c_ad = self._c_ad(C, R_ad)                               # (N, n_out)
        Qc = (R_ad ** 1.5)[:, None] * (c_ad @ self._G.T)         # (N, n_eta)
        Pc = (R_ad ** 0.5)[:, None] * (c_ad @ self._H.T)
        return self._ontop(Qc, Pc)

    def _eps_from_coeffs(self, C, rho0):
        """eps_x per point from window coeffs C (n_in, N) and on-top density rho0 (N,)."""
        R_ad, _ = self._R_ad(np.maximum(np.asarray(rho0, float), 1e-12))
        return self._eps_sf(C, R_ad)

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        """Self-consistent exchange. eps_x = f(C, R_ad(rho)); the discrete-adjoint potential is
        the C-channel (operator transpose, R_ad frozen) plus the explicit R_ad(rho) channel."""
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights; ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])             # (n_in, N)
        R_ad, unclamped = self._R_ad(rho)
        eps = self._eps_sf(C, R_ad)

        acc = np.zeros_like(rho)                                  # C-channel adjoint (R_ad fixed)
        for n in range(len(self._ops)):
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._eps_sf(Cp, R_ad) - self._eps_sf(Cm, R_ad)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        hr = 1e-6 * (R_ad + 1e-8)                                 # local R_ad(rho) channel
        deps_dRad = (self._eps_sf(C, R_ad + hr) - self._eps_sf(C, R_ad - hr)) / (2.0 * hr)
        dRad_drho = np.where(unclamped, -(1.0 / 3.0) * R_ad / rho, 0.0)

        v_x = eps + acc / ew + rho * deps_dRad * dRad_drho
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEEXPParameters:
        return SIMPLEHOLEEXPParameters()


@dataclass
class SIMPLEHOLEEXPGGAParameters(SIMPLEHOLEEXPParameters):
    functional_name: str = "SIMPLE_HOLE_EXPANSION_GGA"
    alpha_lda: float = 1.0           # HEG-gate strength: g = exp(-alpha_lda * D_HEG); larger =>
                                     # GEA needs the density closer to HEG before turning on.
                                     # (named alpha_lda; the symbol c is the projected coeffs C_n.)
    gea_mu: float = 10.0 / 81.0      # GEA enhancement coefficient (F_x -> 1 + g*mu*s^2). The
                                     # s^2 slope is exactly mu in the s->0 limit (g->1); tune mu
                                     # only to set the *effective* enhancement at finite gradient
                                     # (the gate adds an O(s^4) self-saturation, see report).


class SIMPLE_HOLE_EXPANSION_GGA(SIMPLE_HOLE_EXPANSION):
    """Direct-expansion hole with the parameter-free second-order gradient correction.

    The charge- and on-top-neutral gradient deformation of the hole enhances the energy by the
    GEA2 factor, gated by how HEG-like the local density is:
        eps_x = eps_x^map * (1 + g(C) * mu * s^2_bounded),   s = |grad rho|/(2 k_F rho),
        g(C)  = exp(-alpha_lda * D_HEG(C)),   D_HEG = sum_n (C_n/C_0 - (-1)^n/(n+1))^2 .
    s comes from the proven-stable l=1 spectral gradient operator (k_n^1, no stiff Laplacian);
    s^2 is smoothly saturated (``_bound``). mu = ``gea_mu`` (default 10/81), alpha_lda = the
    HEG-gate strength ``alpha_lda`` (default 1; the symbol c is reserved for the coeffs C_n).

    THE GATE IS THE L2 DISTANCE FROM HEG IN SIMPLE FEATURE SPACE. The non-dimensional SIMPLE
    monopole features vanish at the homogeneous-gas limit; D_HEG is their squared L2 norm (the
    monopole channel: C_n/C_0 vs the HEG ratios (-1)^n/(n+1)), so D_HEG = 0 for any uniform
    density and grows with inhomogeneity. The gate turns GEA *on* (g -> 1) only when the whole
    local density is HEG-like, and *off* (g -> 0) in strongly inhomogeneous regions (atomic
    cores, one-electron tails) -- so it leaves already-exact results (e.g. spin-paired He, which
    is far from HEG) untouched. This is the natural scale-free inhomogeneity detector of the
    SIMPLE framework, intrinsic rather than the ad-hoc enclosed-charge switch.

    Slope: because D_HEG ~ s^2 across the window, g = 1 - c*k*s^2 + ..., so the enhancement is
    mu*s^2*(1 - c*k*s^2) = mu*s^2 - O(s^4): the s^2 coefficient is exactly mu in the s->0 limit
    (GEA2 recovered without tuning), and the gate's density-derivative contributes only an
    O(s^4) self-saturation. Tune ``gea_mu`` only to set the effective enhancement at finite s.

    Because the gate depends on C (through D_HEG), the self-consistent potential is the full
    discrete adjoint of eps_x, taken by finite difference in all three channels (C, on-top rho,
    gradient g); the gradient channel uses the spectral-operator transpose."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        self._grad_op = build_spectral_gradient_operator(self._r_grid)

    def _s2_bounded(self, rho, g):
        """Smoothly-saturated reduced gradient squared s^2_b, s = |g|/(2 k_F rho)."""
        rho = np.maximum(rho, 1e-12)
        d8 = 4.0 * _SIX2_3 * rho ** (8.0 / 3.0)
        s2b, _ = _bound(g * g / d8)
        return s2b

    def _l2_distance_from_heg(self, C):
        """Squared L2 distance of the monopole SIMPLE features from the HEG limit:
        D_HEG = sum_n (C_n/C_0 - (-1)^n/(n+1))^2. Scale-free (ratio C_n/C_0) and exactly 0 for
        any uniform density. C (nch, N) -> D (N,)."""
        n = np.arange(C.shape[0])
        heg_ratio = ((-1.0) ** n) / (n + 1.0)                       # C_n/C_0 at HEG
        c0 = C[0]
        c0 = np.where(np.abs(c0) > 1e-30, c0, 1e-30)
        ratio = C / c0[None, :]                                     # (nch, N)
        return np.sum((ratio - heg_ratio[:, None]) ** 2, axis=0)    # (N,)

    def _eps_full(self, C, rho0, g):
        """eps_x = eps_map(C, rho0) * (1 + g(C) * mu * s^2_b(g, rho0)), g = exp(-alpha_lda D_HEG(C))."""
        p = self.params
        eps0 = self._eps_from_coeffs(C, rho0)
        gate = np.exp(-p.alpha_lda * self._l2_distance_from_heg(C))
        f = 1.0 + gate * p.gea_mu * self._s2_bounded(rho0, g)
        return eps0 * f

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])
        g = self._grad_op @ rho
        eps = self._eps_full(C, rho, g)

        # full discrete adjoint by finite difference in each channel (the gate's C-dependence
        # is captured by FD-ing the complete eps_full, not just eps_map).
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):                              # C-channel
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._eps_full(Cp, rho, g) - self._eps_full(Cm, rho, g)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        hr = 1e-6 * (rho + 1e-8)                                     # on-top-density channel
        deps_drho0 = (self._eps_full(C, rho + hr, g) - self._eps_full(C, rho - hr, g)) / (2.0 * hr)
        hg = 1e-6 * (np.abs(g) + 1e-8)                               # gradient channel
        deps_dg = (self._eps_full(C, rho, g + hg) - self._eps_full(C, rho, g - hg)) / (2.0 * hg)

        v_x = (eps
               + rho * deps_drho0
               + acc / ew
               + self._grad_op.T @ (ewrho * deps_dg) / ew)          # spectral gradient transpose
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEEXPGGAParameters:
        return SIMPLEHOLEEXPGGAParameters()
