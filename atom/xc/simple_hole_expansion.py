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
    # Two additive gated corrections of OPPOSITE sign (see report Update 12). The gradient/GEA
    # term (m_g) and the H1s anti-binding term (m_h) cancel for He -> the He/FA limit is
    # preserved; their ratio m_h/m_g is what enforces that cancellation. The overall magnitude
    # (m_g, with m_h = ratio * m_g) is the single calibrated DOF (fit on Be/Na/Mg). Defaults
    # from the self-consistent c_ad-gate calibration (reports/hole_expansion, Update 13).
    m_g: float = -3.0                # HEG-gated s^2 term magnitude (the overall DOF)
    m_h: float = 0.0088              # H1s-gated s^2 term magnitude (= (m_h/m_g) * m_g, ratio
                                     # -0.003 fixed by He cancellation)
    alpha_heg: float = 2.0           # HEG-gate strength: g_HEG = exp(-alpha_heg D_HEG)
    alpha_h1s: float = 8.0           # H1s-gate strength: g_H1s = 1 - exp(-alpha_h1s D_H1s)
    enh_floor: float = 0.02          # LB94-style tail floor: F is soft-bounded below by this
                                     # (>0) so the enhancement cannot flip eps' sign in the tail.
    enh_floor_k: float = 15.0        # soft-floor sharpness; large -> identity in the bulk, only
                                     # lifts F where 1+e dips toward enh_floor (the tail).
    frozen_potential: bool = True    # freeze the enhancement factor F when forming the SCF
                                     # potential (drop its unstable density-response); the energy
                                     # is still the full GGA energy. False = exact (variational)
                                     # adjoint -- correct but unstable in the low-density tail.


class SIMPLE_HOLE_EXPANSION_GGA(SIMPLE_HOLE_EXPANSION):
    """Direct-expansion hole with two additive, opposite-sign gated corrections.

        eps_x = eps_base * (1 + m_g * g_HEG * s^2_b  +  m_h * g_H1s),
        g_HEG = exp(-alpha_heg * D_HEG),   g_H1s = 1 - exp(-alpha_h1s * D_H1s),
        s = |grad rho| / (2 k_F rho)  (proven-stable l=1 spectral gradient; s^2 smoothly bounded).

    The gradient term (m_g, on in HEG-like regions) adds binding; the H1s term (m_h, opposite
    sign, on *away* from the single-orbital limit) is anti-binding and vanishes at the
    one-electron/H1s limit. For He the two integrate to opposite values and cancel, so the
    Fermi-Amaldi limit is preserved (He stays exact); H is exact automatically (D_H1s = 0 on the
    hydrogenic manifold). The cancellation fixes the ratio m_h/m_g, leaving the overall magnitude
    m_g as the single DOF, calibrated on the heavier closed-/near-closed-shell atoms (Be/Na/Mg).

    D_HEG and D_H1s are squared L2 distances of the SCALE-FREE adaptive-radius monopole features
    c_ad = T(R_ad) @ C (the same features the base energy uses) from the HEG signature and from a
    sampled hydrogenic-1s manifold, respectively. Using c_ad (not raw fixed-R_c C) makes the
    signatures Z-independent, so any 1s maps to the manifold and any uniform density to the HEG
    point. Both gates are smooth functions of (C, R_ad), so the self-consistent potential is the
    full discrete adjoint, taken by finite difference in the C, local-rho (R_ad + s^2), and
    gradient channels (the latter via the spectral-operator transpose)."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        self._grad_op = build_spectral_gradient_operator(self._r_grid)
        self._heg_sig, self._h1s_sig = self._build_signatures()

    def _build_signatures(self):
        """Precompute the two scale-free c_ad reference signatures, each a single point-wise
        vector (Z-independent): HEG (uniform density) and a representative hydrogenic-1s point
        (taken at the 1s radial-probability peak). Both gates are then pure point-wise distances
        to a fixed reference -- the natural local form that carries over to 3D unchanged. A single
        1s point cannot match the 1s environment at every radius; that residual is absorbed by the
        magnitude calibration (the gate only needs to separate single-orbital from HEG character)."""
        r = self._r_grid
        rho_u = np.ones_like(r)                                      # uniform -> HEG signature
        Cu = np.array([op @ rho_u for op in self._ops])
        Rad_u, _ = self._R_ad(rho_u)
        cu = self._scalefree(self._c_ad(Cu, Rad_u))
        heg_sig = cu[np.argsort(r)[len(r) // 3]]
        # hydrogenic-1s reference at its radial-probability peak r ~ 1/Z (Z-independent in c_ad)
        Z = 1.5
        rho_1s = (Z ** 3 / np.pi) * np.exp(-2.0 * Z * r)
        C1 = np.array([op @ rho_1s for op in self._ops])
        Rad_1, _ = self._R_ad(np.maximum(rho_1s, 1e-12))
        c1 = self._scalefree(self._c_ad(C1, Rad_1))
        peak = int(np.argmax(rho_1s * r ** 2))                      # radial-probability peak
        return heg_sig, c1[peak]

    @staticmethod
    def _scalefree(c_ad):
        """Scale-free monopole signature: c_ad normalized by its n=0 component (Z-independent)."""
        c0 = c_ad[:, 0:1]
        c0 = np.where(np.abs(c0) > 1e-30, c0, 1e-30)
        return c_ad / c0

    def _gates(self, C, R_ad):
        """D_HEG, D_H1s: pure point-wise squared distances of the scale-free adaptive-radius
        features c_ad = T(R_ad) @ C from the HEG and hydrogenic-1s reference signatures. Both are
        smooth local functions of (C, R_ad) -- no manifold, no min -- so they carry to 3D and the
        self-consistent potential stays smooth."""
        cn = self._scalefree(self._c_ad(C, R_ad))                   # (N, n_out)
        d_heg = np.sum((cn - self._heg_sig[None, :]) ** 2, axis=1)  # (N,)
        d_h1s = np.sum((cn - self._h1s_sig[None, :]) ** 2, axis=1)  # (N,)
        return d_heg, d_h1s

    def _s2_bounded(self, rho, g):
        """Smoothly-saturated reduced gradient squared s^2_b, s = |g|/(2 k_F rho)."""
        rho = np.maximum(rho, 1e-12)
        d8 = 4.0 * _SIX2_3 * rho ** (8.0 / 3.0)
        s2b, _ = _bound(g * g / d8)
        return s2b

    def _enhancement(self, C, rho0, g):
        """The gated gradient enhancement factor F and eps0.

            e     = s^2 (m_g g_HEG + m_h g_H1s)              # raw two-term gated enhancement
            F     = enh_floor + softplus_k(1 + e - enh_floor) # LB94-style tail floor

        Both gated terms carry s^2, so the correction vanishes at the HEG limit (uniform density,
        s=0 -> LDA preserved). g_HEG = exp(-alpha_heg D_HEG) attenuates in non-HEG regions;
        g_H1s = 1 - exp(-alpha_h1s D_H1s) attenuates near the single-orbital limit; opposite-sign
        m_g, m_h make the two cancel for He (FA limit preserved). The raw factor 1+e diverges in
        the low-density tail (s^2 large) and would go negative -> sign-flipped eps and an SCF
        blow-up. The soft floor (softplus with sharpness enh_floor_k) is the IDENTITY in the bulk
        (where 1+e is comfortably above enh_floor, so the good correction is untouched) and lifts
        F smoothly to enh_floor only where 1+e dips toward/below it (the tail) -- keeping F
        strictly positive without disturbing the energy-relevant region."""
        p = self.params
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        R_ad, unclamped = self._R_ad(rho0)
        eps0 = self._eps_sf(C, R_ad)
        d_heg, d_h1s = self._gates(C, R_ad)
        g_heg = np.exp(-p.alpha_heg * d_heg)
        g_h1s = 1.0 - np.exp(-p.alpha_h1s * d_h1s)
        e = self._s2_bounded(rho0, g) * (p.m_g * g_heg + p.m_h * g_h1s)
        k = p.enh_floor_k
        F = p.enh_floor + np.logaddexp(0.0, k * (1.0 + e - p.enh_floor)) / k   # soft lower bound
        return F, eps0, R_ad, unclamped

    def _eps_full(self, C, rho0, g):
        """eps_x = eps_base * F, with the tail-damped gated enhancement F (see ``_enhancement``)."""
        F, eps0, _, _ = self._enhancement(C, rho0, g)
        return eps0 * F

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])
        g = self._grad_op @ rho

        if getattr(self.params, "frozen_potential", True):
            # FROZEN-SCF potential: the gated enhancement factor F is held fixed when forming the
            # potential (its density-response -- the gradient/gate adjoint terms -- is the unstable
            # part, blowing up in the low-density tail). The SCF is then driven by the well-behaved
            # base potential weighted by F; the energy at convergence is the full GGA energy
            # E = sum ew rho eps0 F. v_x is the exact adjoint of E at *fixed* F:
            #   v_x = eps0 F + (1/ew) sum_n P_n^T(ew rho F d eps0/dC_n) + rho F d eps0/drho0 .
            F, eps0, R_ad, unclamped = self._enhancement(C, rho, g)
            eps = eps0 * F
            wF = ewrho * F
            acc = np.zeros_like(rho)
            for n in range(len(self._ops)):                          # C-channel (eps0 only)
                h = 1e-6 * (np.abs(C[n]) + 1e-8)
                Cp = C.copy(); Cp[n] += h
                Cm = C.copy(); Cm[n] -= h
                deps0 = (self._eps_sf(Cp, R_ad) - self._eps_sf(Cm, R_ad)) / (2.0 * h)
                acc += self._ops[n].T @ (wF * deps0)
            hr = 1e-6 * (R_ad + 1e-8)                                # R_ad channel (eps0 only)
            deps0_dRad = (self._eps_sf(C, R_ad + hr) - self._eps_sf(C, R_ad - hr)) / (2.0 * hr)
            dRad_drho = np.where(unclamped, -(1.0 / 3.0) * R_ad / rho, 0.0)
            v_x = eps + acc / ew + rho * F * deps0_dRad * dRad_drho
        else:
            # exact discrete adjoint (variational; used by the adjoint test). Both gates depend on
            # rho through C and R_ad, captured by the C-channel and the local-rho channel; the
            # gradient channel uses the spectral-operator transpose.
            eps = self._eps_full(C, rho, g)
            acc = np.zeros_like(rho)
            for n in range(len(self._ops)):
                h = 1e-6 * (np.abs(C[n]) + 1e-8)
                Cp = C.copy(); Cp[n] += h
                Cm = C.copy(); Cm[n] -= h
                deps_dCn = (self._eps_full(Cp, rho, g) - self._eps_full(Cm, rho, g)) / (2.0 * h)
                acc += self._ops[n].T @ (ewrho * deps_dCn)
            hr = 1e-6 * (rho + 1e-8)                                 # local-rho channel (R_ad + s^2)
            deps_drho0 = (self._eps_full(C, rho + hr, g) - self._eps_full(C, rho - hr, g)) / (2.0 * hr)
            hg = 1e-6 * (np.abs(g) + 1e-8)                           # gradient channel
            deps_dg = (self._eps_full(C, rho, g + hg) - self._eps_full(C, rho, g - hg)) / (2.0 * hg)
            v_x = (eps + rho * deps_drho0 + acc / ew
                   + self._grad_op.T @ (ewrho * deps_dg) / ew)
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEEXPGGAParameters:
        return SIMPLEHOLEEXPGGAParameters()


_LO_FX = 1.804      # Lieb-Oxford enhancement ceiling
_C_LDA = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)   # LDA exchange: eps_x = _C_LDA rho^{1/3}


@dataclass
class SIMPLEHOLEEXPKERNELParameters(SIMPLEHOLEEXPParameters):
    functional_name: str = "SIMPLE_HOLE_EXPANSION_KERNEL"
    # Kernel/fixed-point map -- PARAMETER-FREE. LDA via the HEG anchor; GEA via the feature-
    # distance lever (slope 10/81 by construction, calibrated by the one-time HEG response R);
    # Fermi-Amaldi via the per-spin enclosed-charge gate. No free magnitude.


class SIMPLE_HOLE_EXPANSION_KERNEL(SIMPLE_HOLE_EXPANSION):
    """Kernel / fixed-point exchange-hole map on the scale-free adaptive-radius frame.

    The spherically-averaged hole monopole is produced directly by a kernel interpolation over
    fixed points, on the adaptive unit frame (k_F R_ad = X):
      bulk = HEG anchor  rhotilde_HEG = -(rho0/2) g(X)   (LDA)
           + GEA mode    chi * delta_GEA,  chi = (10/81)/R s^2 (LO-capped)   (gradient, mu=10/81)
      FA   = -d/Q       (density-following, d = unit-basis density coeffs = c_ad/(4 pi R_ad^3/2))
      rhotilde = (1-W_FA(Q/2)) bulk + W_FA(Q/2) FA ;  then 2-constraint projection (sum rule, on-top)
      eps_x = 2 pi R_ad^2 (rhotilde . beta1)

    The l=1 part of the feature distance from HEG IS s^2, so the GEA enters with no separate term
    (parameter-free, calibrated by R). FA is a charge gate (Q = 4 pi R_ad^3 (d.alpha1)); He
    (Q/2=1) -> pure FA -> exact. LDA and GEA are exact by construction; the construction is
    weakest only in the FA<->bulk transition. SCF potential = exact variational adjoint (FD
    through the C, local-rho and gradient channels)."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        self._grad_op = build_spectral_gradient_operator(self._r_grid)
        n = self._n_out
        self._alpha1 = self._G[0].copy()                     # int_0^1 R_m^(1) t^2 dt
        self._beta1 = self._H[0].copy()                      # int_0^1 R_m^(1) t   dt
        self._r1_0 = (np.arange(n) + 1) * np.pi * np.sqrt(2.0)   # R_m^(1)(0), unit window
        self._gX = np.array([np.interp(self._X, self._etas, self._G[:, m]) for m in range(n)])
        # GEA deformation mode delta_GEA = proj(g0(Xt) phi(Xt)), charge/on-top-neutral, + response R
        xu, wu = np.polynomial.legendre.leggauss(400); t = 0.5 * (xu + 1.0); wt = 0.5 * wu
        Rb1 = RadialBesselBasis(n - 1, 0, 1.0).evaluate(0, t)
        xX = np.maximum(self._X * t, 1e-12)
        g0 = 3.0 * spherical_jn(1, xX) / xX; phi = spherical_jn(1, xX)
        d1 = Rb1 @ (g0 * phi * wt * t ** 2)
        A0 = np.vstack([self._alpha1, self._r1_0])
        self._dgea = d1 - A0.T @ np.linalg.solve(A0 @ A0.T, A0 @ d1)
        kF1 = (3.0 * np.pi ** 2) ** (1.0 / 3.0); Rad1 = min(self._X / kF1, self.params.r_c)
        c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
        self._gea_R = 2.0 * np.pi * Rad1 ** 2 * (-(self._dgea @ self._beta1)) / c_lda

    def _kernel_eps(self, C, rho0, g):
        """eps_x from the kernel map; C (n_in,N), rho0 (N,), g (N,) the three rho-channels."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        R_ad, _ = self._R_ad(rho0)
        c_ad = self._c_ad(C, R_ad)                                   # (N, n_out)
        d = c_ad / (4.0 * np.pi * R_ad ** 1.5)[:, None]              # unit-basis density coeffs
        Q = 4.0 * np.pi * R_ad ** 3 * (d @ self._alpha1); Qs = np.maximum(Q, 1e-12)
        W = enclosed_charge_switch(0.5 * Q)                          # per-spin FA gate
        kF = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
        s2, _ = _bound((g / (2.0 * kF * rho0)) ** 2)
        chi = (_GEA2 / self._gea_R) * s2; chi_max = (_LO_FX - 1.0) / self._gea_R
        chi = chi_max * np.tanh(chi / chi_max)
        al, r0v, be = self._alpha1, self._r1_0, self._beta1
        # HEG anchor MOMENT-MATCHED to the exact LDA hole on the n_out basis: pin the three
        # low-order moments {charge int u^2 = -1, on-top, Coulomb int u = exact LDA energy}.
        # The projected HEG hole alone gives 0.984*LDA (the basis truncates the [3j1/x]^2 tail);
        # matching the Coulomb (=energy) moment to C_LDA rho^{1/3} deforms it (~6%) to hit LDA
        # exactly at n_out=10. Principled: the energy IS the hole's Coulomb moment.
        heg = -0.5 * rho0[:, None] * self._gX[None, :]               # (N, n_out) projected HEG
        a_row = 4.0 * np.pi * (R_ad ** 3)[:, None] * al[None, :]     # charge-moment row
        e_row = 2.0 * np.pi * (R_ad ** 2)[:, None] * be[None, :]     # Coulomb (energy) row
        A3 = np.stack([a_row, np.broadcast_to(r0v, a_row.shape), e_row], axis=1)   # (N,3,n_out)
        rhs3 = np.stack([-1.0 - np.sum(a_row * heg, axis=1),
                         -0.5 * rho0 - heg @ r0v,
                         _C_LDA * rho0 ** (1.0 / 3.0) - np.sum(e_row * heg, axis=1)], axis=1)
        lam3 = np.linalg.solve(A3 @ np.transpose(A3, (0, 2, 1)), rhs3[..., None])[..., 0]   # (N,3)
        heg = heg + np.einsum('nk,nkm->nm', lam3, A3)               # exact-LDA HEG hole on basis
        bulk = heg + (chi * -rho0)[:, None] * self._dgea[None, :]
        fa = -d / Qs[:, None]
        coeffs = (1.0 - W)[:, None] * bulk + W[:, None] * fa
        ontop = (1.0 - W) * (-0.5 * rho0) + W * (-rho0 / Qs)
        # vectorized 2-constraint least-norm projection (sum rule = -1, on-top = ontop)
        al, r0v = self._alpha1, self._r1_0
        a_row = 4.0 * np.pi * (R_ad ** 3)[:, None] * al[None, :]
        row0 = np.sum(a_row * coeffs, axis=1); row1 = coeffs @ r0v
        g00 = np.sum(a_row * a_row, axis=1); g01 = a_row @ r0v; g11 = float(r0v @ r0v)
        res0 = -1.0 - row0; res1 = ontop - row1
        det = g00 * g11 - g01 ** 2
        lam0 = (g11 * res0 - g01 * res1) / det; lam1 = (-g01 * res0 + g00 * res1) / det
        coeffs = coeffs + lam0[:, None] * a_row + lam1[:, None] * r0v[None, :]
        return 2.0 * np.pi * R_ad ** 2 * (coeffs @ self._beta1)

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights; ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])
        g = self._grad_op @ rho
        eps = self._kernel_eps(C, rho, g)
        # exact discrete adjoint by FD in each rho-channel (C, local-rho, gradient)
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._kernel_eps(Cp, rho, g) - self._kernel_eps(Cm, rho, g)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        hr = 1e-6 * (rho + 1e-8)
        deps_drho0 = (self._kernel_eps(C, rho + hr, g) - self._kernel_eps(C, rho - hr, g)) / (2.0 * hr)
        hg = 1e-6 * (np.abs(g) + 1e-8)
        deps_dg = (self._kernel_eps(C, rho, g + hg) - self._kernel_eps(C, rho, g - hg)) / (2.0 * hg)
        v_x = (eps + rho * deps_drho0 + acc / ew
               + self._grad_op.T @ (ewrho * deps_dg) / ew)
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEEXPKERNELParameters:
        return SIMPLEHOLEEXPKERNELParameters()
