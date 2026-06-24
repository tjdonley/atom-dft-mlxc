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
    """Scale-free direct-expansion hole settings (SIMPLE adaptive-radius frame).

    ``n_channels`` = n_in: the fixed-R_c window resolution used to project the density.
    ``n_out`` = the adaptive-radius hole/feature basis (the exposed SIMPLE feature count).
    The implicit adaptive radius R_ad = min(x_window/k_F(rho), R_c) makes the hole scale-free,
    so n_out=10 resolves it (the dense core gets a small window). The SIMPLE ``transfer_matrix``
    re-expresses the n_in window coefficients on the n_out adaptive basis in closed form."""
    functional_name: str = "SIMPLE_HOLE_EXPANSION"
    r_c: float = 6.0
    n_channels: int = 20            # n_in: fixed-R_c projection resolution
    n_out: int = 10                 # adaptive-radius hole basis (exposed feature count)
    x_window: float = _X_WINDOW     # dimensionless hole window X = k_F R_ad
    n_rad: int = 48                 # R_ad-grid for the precomputed transfer matrices
    n_eta: int = 400                # eta-grid for the universal HEG-anchor shape sigma(eta)


class SIMPLE_HOLE_EXPANSION(SIMPLE_HOLE):
    """Exchange-only scale-free direct-expansion exchange hole, self-consistent. Parameter-free.

    The hole is expanded on the IMPLICIT adaptive-radius basis: project the density to n_in
    fixed-R_c window coefficients C, set R_ad = min(X/k_F(rho0), R_c), transfer
    c_ad = T(R_ad) @ C onto the n_out adaptive basis, and expand the hole there. Because
    k_F R_ad = X is locked (where unclamped), the hole is represented at a density-independent
    dimensionless scale -> scale invariance, and n_out=10 suffices (the dense core gets a small
    window). The HEG anchor is then a UNIVERSAL fixed shape sigma_m = int_0^1 R_m^(1)(t) S(X t)
    t^2 dt; the moments scale as R_ad^{3/2} (charge) and R_ad^{1/2} (self-Coulomb)."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        p = self.params
        self._n_in = p.n_channels
        self._n_out = int(getattr(p, "n_out", 10))
        self._X = float(getattr(p, "x_window", _X_WINDOW))
        # unit-window adaptive monopole basis R_m^(1) on [0,1]: moments + on-top
        basis = RadialBesselBasis(self._n_out - 1, 0, 1.0)
        xu, wu = np.polynomial.legendre.leggauss(600)
        t = 0.5 * (xu + 1.0); wt = 0.5 * wu
        Rb1 = basis.evaluate(0, t)                               # (n_out, nt)
        self._a1 = Rb1 @ (wt * t ** 2)                           # unit-window charge moments
        self._b1 = Rb1 @ (wt * t)                                # unit-window self-Coulomb moments
        self._r0_1 = basis.evaluate(0, np.array([1e-9]))[:, 0]   # R_m^(1)(0)
        # universal HEG-anchor shape sigma(eta) = int R_m^(1)(t) S(eta t) t^2 dt; eta = k_F R_ad <= X
        self._etas = np.linspace(1e-3, self._X, int(getattr(p, "n_eta", 400)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._sigma = np.array([Rb1 @ (wt * t ** 2 * _envelope(eta * t)) for eta in self._etas])
        # transfer matrices T(R_ad) = transfer_matrix(0, R_ad, n_out, n_in), precomputed + interp
        n_rad = int(getattr(p, "n_rad", 48))
        self._rad_grid = np.linspace(p.r_c / n_rad, p.r_c, n_rad)
        self._T_grid = np.stack([transfer_matrix(0, float(ra), self._n_out, self._n_in)
                                 for ra in self._rad_grid])      # (n_rad, n_out, n_in)

    def _R_ad(self, rho0):
        """Implicit adaptive radius R_ad = min(X/k_F(rho0), R_c) (explicit, differentiable)."""
        kF = (3.0 * np.pi ** 2 * np.maximum(rho0, 1e-12)) ** (1.0 / 3.0)
        return np.minimum(self._X / np.maximum(kF, 1e-12), self.params.r_c)

    def _c_ad(self, C, R_ad):
        """Adaptive-radius density features c_ad (N, n_out) = T(R_ad) @ C, via interpolation."""
        rg = self._rad_grid
        k = np.clip(np.searchsorted(rg, R_ad), 1, rg.size - 1)
        f = np.clip((R_ad - rg[k - 1]) / (rg[k] - rg[k - 1]), 0.0, 1.0)
        Tb = (1.0 - f)[:, None, None] * self._T_grid[k - 1] + f[:, None, None] * self._T_grid[k]
        return np.einsum('Noi,iN->No', Tb, C)                    # (N, n_out)

    def _sigma_at(self, eta):
        """Universal HEG-anchor shape sigma(eta) (N, n_out), interpolated on the eta grid."""
        et = self._etas
        k = np.clip(np.searchsorted(et, eta), 1, et.size - 1)
        f = np.clip((eta - et[k - 1]) / (et[k] - et[k - 1]), 0.0, 1.0)
        return (1.0 - f)[:, None] * self._sigma[k - 1] + f[:, None] * self._sigma[k]

    def _map_coeffs(self, C, rho0):
        """Scale-free map: n_in window coeffs C (n_in, N) + on-top density rho0 (N,) -> adaptive
        hole coeffs rhotilde (N, n_out). HEG <-> Fermi-Amaldi anchors blended by the per-spin
        enclosed-charge switch, projected onto the (R_ad-scaled) sum-rule and on-top constraints."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        R_ad = self._R_ad(rho0)
        # C from the window operators carries the angular 4pi; divide it out so c_ad are the
        # bare adaptive coefficients (same convention as the HEG anchor sigma) -- otherwise the
        # enclosed charge Q below is inflated by 4pi and every atom is wrongly flagged HEG.
        c_ad = self._c_ad(C / (4.0 * np.pi), R_ad)               # (N, n_out)
        eta = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0) * R_ad    # = X where unclamped
        Q = 4.0 * np.pi * (R_ad ** 1.5) * (c_ad @ self._a1)      # (N,) enclosed charge
        Qsafe = np.maximum(Q, 1e-12)
        lam = enclosed_charge_switch(0.5 * Q)                    # per-spin switch (exchange is same-spin)
        heg = -(0.5 * rho0)[:, None] * (R_ad ** 1.5)[:, None] * self._sigma_at(eta)   # HEG anchor
        fa = -c_ad / Qsafe[:, None]                              # Fermi-Amaldi anchor
        coeffs = (1.0 - lam)[:, None] * heg + lam[:, None] * fa  # (N, n_out)
        ontop = (1.0 - lam) * (-0.5 * rho0) + lam * (-rho0 / Qsafe)
        # 2-constraint least-norm projection with R_ad-dependent rows (vectorized 2x2 solve):
        A0 = 4.0 * np.pi * (R_ad ** 1.5)[:, None] * self._a1[None, :]    # sum-rule row (N, n_out)
        A1 = (R_ad ** -1.5)[:, None] * self._r0_1[None, :]               # on-top row    (N, n_out)
        g00 = np.sum(A0 * A0, axis=1); g01 = np.sum(A0 * A1, axis=1); g11 = np.sum(A1 * A1, axis=1)
        r0 = -1.0 - np.sum(A0 * coeffs, axis=1); r1 = ontop - np.sum(A1 * coeffs, axis=1)
        det = np.maximum(g00 * g11 - g01 ** 2, 1e-300)
        x0 = (g11 * r0 - g01 * r1) / det; x1 = (g00 * r1 - g01 * r0) / det
        return coeffs + x0[:, None] * A0 + x1[:, None] * A1      # (N, n_out)

    def _eps_from_coeffs(self, C, rho0):
        """eps_x per point: 1/2 * 4pi * R_ad^{1/2} * sum_m rhotilde_m b_m^(1) [Eq. (eps)]."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        R_ad = self._R_ad(rho0)
        coeffs = self._map_coeffs(C, rho0)                       # (N, n_out)
        return 0.5 * 4.0 * np.pi * (R_ad ** 0.5) * (coeffs @ self._b1)   # (N,)

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        """Self-consistent exchange via the direct-expansion hole. The potential is the
        discrete adjoint through both C = P @ rho and the explicit on-top density rho0."""
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        C = np.array([op @ rho for op in self._ops])            # (nch, N)
        eps = self._eps_from_coeffs(C, rho)

        ewrho = ew * rho
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):                          # C-channel adjoint (Sahoo)
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._eps_from_coeffs(Cp, rho) - self._eps_from_coeffs(Cm, rho)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        # explicit on-top-density derivative (rho0 = rho): local term in the variational derivative
        hr = 1e-6 * (rho + 1e-8)
        deps_drho0 = (self._eps_from_coeffs(C, rho + hr) - self._eps_from_coeffs(C, rho - hr)) / (2.0 * hr)

        v_x = eps + rho * deps_drho0 + acc / ew
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
