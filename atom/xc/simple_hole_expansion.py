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

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.optimize import brentq
from scipy.special import spherical_jn

from ..descriptors.simple.bessel import RadialBesselBasis
from ..descriptors.simple.derivatives import build_spectral_gradient_operator, build_spectral_l2_operator
from ..descriptors.simple.params import R_C as _PIPELINE_R_C
from ..descriptors.simple.pipeline import transfer_matrix
from .evaluator import DensityData, XCParameters, XCPotentialData
from .simple_hole import SIMPLE_HOLE, SIMPLEHOLEParameters
from .simple_hole_expansion_explicit import enclosed_charge_switch

_SIX2_3 = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
_GEA2 = 10.0 / 81.0   # second-order gradient-expansion coefficient F_x -> 1 + (10/81) s^2
_LO_FX = 1.804        # Lieb-Oxford ceiling on the spin-unpolarized exchange enhancement F_x
_X_WINDOW = 8.0       # dimensionless hole window X = k_F R_ad (the implicit scale lock)
_INV_BOUND = 4.0      # smooth saturation scale for the reduced gradient (tail safety)


def _bound(v):
    """Smooth saturation v -> v/(1+|v|/M); returns (value, d/dv)."""
    den = 1.0 + np.abs(v) / _INV_BOUND
    return v / den, 1.0 / den ** 2


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



_LO_FX = 1.804      # Lieb-Oxford enhancement ceiling
_C_LDA = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)   # LDA exchange: eps_x = _C_LDA rho^{1/3}

# optional bulk reference holes (exact atomic holes as kernel fixed points); absent -> reference-free
_KERNEL_FP_REFS = os.path.join(os.path.dirname(__file__), "data", "kernel_fp_refs.npz")


@dataclass
class SIMPLEHOLEKERNELFPParameters(SIMPLEHOLEEXPParameters):
    functional_name: str = "SIMPLE_HOLE_KERNEL_FP"
    # Clean kernel-mapped fixed-point hole: interpolate the scale-free hole SHAPE deviation; LDA
    # exact (HEG node), GEA2 from the kernel's l=1 slope via a calibrated GEA node (no additive
    # GEA term, no enhancement factor), Fermi-Amaldi via the charge gate; optional bulk reference holes.
    # Kernel hyperparameters (TUNABLE). The GEA2 slope (10/81) is always enforced via the l=1 node
    # amplitude c_G; the l=1 RBF WIDTH fp_l1 is left free (NOT pinned to the Lieb-Oxford ceiling) so it
    # can be tuned. fp_l0 = l=0 power-vector width; fp_DG = GEA-node placement; fp_ridge = kernel ridge.
    fp_l0: float = 0.5
    fp_l1: float = 0.5
    fp_DG: float = 0.3
    fp_ridge: float = 1e-8
    use_l2: bool = False       # add the l=2 (quadrupole) axial feature as an extra kernel coordinate
    fp_l2: float = 0.5         # its RBF length scale (only used when use_l2)
    fp_ref_ridge: Optional[float] = 1e-2  # ridge on the REFERENCE block only (kernel ridge regression);
                                          # default 1e-2 makes loaded references SCF-stable out of the box
                                          # (exact interp ill-conditions v_x -> SCF spikes). Set to fp_ridge
                                          # for exact interpolation. Backbone stays un-ridged so the HEG
                                          # (LDA) and GEA-slope limits remain exact.
    refs_path: Optional[str] = None   # kernel reference-node .npz (X, DELTA); None -> module _KERNEL_FP_REFS


class SIMPLE_HOLE_KERNEL_FP(SIMPLE_HOLE_EXPANSION):
    """Kernel-mapped fixed-point exchange hole on the scale-free adaptive-radius frame
    (writeup App.~\\ref{app:kernel}).

    The spherically-averaged hole monopole is a coefficient vector rhotilde whose scale-free SHAPE
    sigma = rhotilde / (-rho/2) is produced by interpolating the deviation from the LDA hole over
    fixed points with an RBF on per-l SIMPLE distances [l=0: the monopole power-spectrum vector
    cn[1:]; l=1: s^2]:
      sigma(x) = sigma_LDA + sum_k K(x, x_k) c_k .
    The HEG node (x=0) pins LDA exact at s^2=0 (sigma_LDA = moment-matched [3j1/x]^2 hole); a single
    GEA node along the l=1 axis has its amplitude calibrated so the kernel's l=1 slope reproduces the
    GEA2 limit F_x-1 = (10/81) s^2 exactly (no additive GEA term, no enhancement factor; parity
    automatic since the coordinate is s^2; 4th-order dropped). Fermi-Amaldi (-varrho_0/Q,
    density-following) blends in via the per-spin charge gate W_FA(Q/2). Optional bulk reference holes
    (W_FA-gated to genuine bulk content) shape the strongly-inhomogeneous interior. Energy is the
    direct hole integral eps_x = -pi rho R_ad^2 (rhotilde . C) [writeup Eq. eps-direct]; the SCF
    potential is the exact variational discrete adjoint (FD through the cprime, local-rho and
    gradient channels)."""

    def __init__(self, derivative_matrix=None, r_quad=None, quadrature_weights=None,
                 params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        # --- precompute on the adaptive unit frame ---
        self._grad_op = build_spectral_gradient_operator(self._r_grid)
        self._use_l2 = bool(getattr(self.params, "use_l2", False))
        self._fp_l2 = float(getattr(self.params, "fp_l2", 0.5))
        self._l2_op = build_spectral_l2_operator(self._r_grid) if self._use_l2 else None
        n = self._n_out
        # Notation tracks the writeup: B = charge moments, C = Coulomb moments, R0 = on-top (basis at
        # origin), rhotilde = dimensionless hole shape, cprime = raw monopole coefficient op@rho
        # (writeup c'_{n00}); the dimensionless feature is varrho_0 = cprime / (A0 rho_safe).
        self._Bmom = self._G[0].copy()                       # B: charge moments int_0^1 R(t) t^2 dt (sum rule)
        self._Cmom = self._H[0].copy()                       # C: Coulomb moments int_0^1 R(t) t   dt (energy)
        self._R0 = (np.arange(n) + 1) * np.pi * np.sqrt(2.0) # R0: basis at origin (on-top constraint)
        self._gX = np.array([np.interp(self._X, self._etas, self._G[:, m]) for m in range(n)])
        # GEA deformation mode delta_GEA = proj(g0(Xt) phi(Xt)), charge/on-top-neutral, + response R
        xu, wu = np.polynomial.legendre.leggauss(400); t = 0.5 * (xu + 1.0); wt = 0.5 * wu
        Rb1 = RadialBesselBasis(n - 1, 0, 1.0).evaluate(0, t)
        xX = np.maximum(self._X * t, 1e-12)
        g0 = 3.0 * spherical_jn(1, xX) / xX; phi = spherical_jn(1, xX)
        d1 = Rb1 @ (g0 * phi * wt * t ** 2)
        A0 = np.vstack([self._Bmom, self._R0])
        self._dgea = d1 - A0.T @ np.linalg.solve(A0 @ A0.T, A0 @ d1)
        kF1 = (3.0 * np.pi ** 2) ** (1.0 / 3.0); Rad1 = min(self._X / kF1, self.params.r_c)
        self._gea_R = 2.0 * np.pi * Rad1 ** 2 * (-(self._dgea @ self._Cmom)) / _C_LDA
        # --- fixed-point kernel: HEG node + calibrated GEA node + optional bulk references ---
        # Kernel hyperparameters from params (TUNABLE). The 10/81 GEA slope is enforced by c_G in
        # _build_fp_nodes; the l=1 width _fp_l1 is a free tunable parameter (Lieb-Oxford cap dropped).
        p = self.params
        self._fp_l0, self._fp_l1 = float(p.fp_l0), float(p.fp_l1)
        self._fp_DG, self._fp_ridge = float(p.fp_DG), float(p.fp_ridge)
        self._fp_ell = None                              # optional ARD-SE per-dim length scales (n_out,)
        A = self._X ** 2 / (3.0 * np.pi ** 2) ** (2.0 / 3.0)
        self._fp_kappa = np.pi * A / abs(_C_LDA)              # F_x-1 = kappa (delta_rhotilde . C)
        self._fp_dgb = float(self._dgea @ self._Cmom)
        rho1 = np.array([1.0]); Rad1b, _ = self._R_ad(rho1)
        self._rhotilde_lda = (self._heg_mm(rho1, Rad1b) / (-0.5 * rho1[:, None]))[0]   # dimensionless LDA hole shape
        rhou = np.full(self._r_grid.shape, 1.0)              # HEG monopole signature cn_HEG
        Cu = np.array([op @ rhou for op in self._ops]); Ru, _ = self._R_ad(rhou)
        cau = self._c_ad(Cu, Ru); cnu = cau / np.where(np.abs(cau[:, :1]) > 1e-30, cau[:, :1], 1e-30)
        self._cnH = cnu[len(cnu) // 2]
        self._build_fp_nodes()

    def _fx_gea_axis(self, svals):
        """Realized enhancement F_x(s) on the pure-GEA feature axis (l=0 = HEG signature, l=1 = s^2);
        F_x - 1 = kappa (delta_sigma . C). Used to pin _fp_l1 and for the writeup figure."""
        svals = np.atleast_1d(np.asarray(svals, float))
        cn = np.tile(self._cnH, (len(svals), 1))
        x = self._xfeat(cn, svals ** 2)
        return 1.0 + self._fp_kappa * ((self._Kmat(x, self._fp_Xnodes) @ self._fp_coef) @ self._Cmom)

    def _calibrate_l1_to_LO(self):
        """Solve the l=1 RBF width so the realized F_x peaks at the Lieb-Oxford ceiling _LO_FX.
        c_G is re-solved for each trial width (in _build_fp_nodes), so the 10/81 small-s slope is
        held exact; only the width -- the otherwise-undetermined length scale -- is set here.
        The width is a property of the LDA+GEA BACKBONE's LO-compliance, so it is calibrated with
        reference nodes EXCLUDED (they are localized corrections that must not move the LO width);
        the final node build then re-includes the references at the fixed width."""
        s = np.linspace(0.0, 30.0, 6000)

        def peak_minus_LO(l1):
            self._fp_l1 = float(l1)
            self._build_fp_nodes(include_refs=False)
            return float(self._fx_gea_axis(s).max()) - _LO_FX

        self._fp_l1 = brentq(peak_minus_LO, 1.0, 60.0, xtol=1e-6)
        self._build_fp_nodes(include_refs=True)

    def _heg_mm(self, rho0, R_ad):
        """Moment-matched exact-LDA HEG hole on the n_out basis (N, n_out): pin the three low-order
        moments {charge int u^2 = -1, on-top = -rho0/2, Coulomb int u = C_LDA rho0^{1/3}}. The
        projected HEG hole alone gives 0.984*LDA (the basis truncates the [3j1/x]^2 tail); matching
        the Coulomb (=energy) moment deforms it (~1.6%) to hit LDA exactly at n_out=10."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        Bmom, R0, Cmom = self._Bmom, self._R0, self._Cmom
        heg = -0.5 * rho0[:, None] * self._gX[None, :]
        a_row = 4.0 * np.pi * (R_ad ** 3)[:, None] * Bmom[None, :]
        e_row = 2.0 * np.pi * (R_ad ** 2)[:, None] * Cmom[None, :]
        A3 = np.stack([a_row, np.broadcast_to(R0, a_row.shape), e_row], axis=1)
        rhs3 = np.stack([-1.0 - np.sum(a_row * heg, axis=1),
                         -0.5 * rho0 - heg @ R0,
                         _C_LDA * rho0 ** (1.0 / 3.0) - np.sum(e_row * heg, axis=1)], axis=1)
        lam3 = np.linalg.solve(A3 @ np.transpose(A3, (0, 2, 1)), rhs3[..., None])[..., 0]
        return heg + np.einsum('nk,nkm->nm', lam3, A3)

    def _xfeat(self, cn, s2, t2=None):
        cn = np.atleast_2d(np.asarray(cn, float)); s2 = np.atleast_1d(np.asarray(s2, float))
        cols = [cn[:, 1:], s2]                                   # [l=0 power vector cn[1:], l=1 s^2]
        if getattr(self, "_use_l2", False):                      # optional l=2 (quadrupole) coordinate
            t2 = np.zeros_like(s2) if t2 is None else np.atleast_1d(np.asarray(t2, float))
            cols.append(t2)
        return np.column_stack(cols)

    def _inv_ell(self):
        """Per-dimension inverse length scales (n_out,): ARD `_fp_ell` if set, else isotropic
        [1/l0]*(n_out-1) on the l=0 block + [1/l1] on s^2. Cached until the scales change."""
        ell = getattr(self, "_fp_ell", None)
        l2 = self._fp_l2 if getattr(self, "_use_l2", False) else None
        key = ("ard", id(ell)) if ell is not None else ("iso", self._fp_l0, self._fp_l1, l2)
        if getattr(self, "_inv_ell_key", None) != key:
            nl0 = self._n_out - 1
            if ell is not None:
                w = 1.0 / np.asarray(ell, float)
            else:
                w = np.concatenate([np.full(nl0, 1.0 / self._fp_l0), [1.0 / self._fp_l1]])
                if l2 is not None:                               # extra l=2 coordinate
                    w = np.concatenate([w, [1.0 / l2]])
            self._inv_ell_cache = w; self._inv_ell_key = key
        return self._inv_ell_cache

    def _Kmat(self, Xa, Xb):
        # squared-exp kernel via the BLAS distance identity ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
        # (avoids the (Na, Nb, n_out) broadcast; dominant cost with many nodes).
        w = self._inv_ell()
        A = np.asarray(Xa) * w; B = np.asarray(Xb) * w
        d2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
        return np.exp(-0.5 * np.maximum(d2, 0.0))

    def _build_fp_nodes(self, include_refs=True):
        x_heg = self._xfeat(self._cnH[None, :], np.array([0.0]))          # HEG node (LDA, Delta=0)
        x_gea = self._xfeat(self._cnH[None, :], np.array([self._fp_DG]))  # GEA node (l=1 axis)
        refs_path = getattr(self.params, "refs_path", None) or _KERNEL_FP_REFS
        if include_refs and os.path.exists(refs_path):
            z = np.load(refs_path); Xb = z["X"]; Db = z["DELTA"]
        else:
            Xb = np.zeros((0, x_heg.shape[1])); Db = np.zeros((0, self._n_out))   # Xb: feature dim; Db: n_out
        Xnodes = np.vstack([x_heg, x_gea, Xb])
        mu = np.concatenate([[0.0, 0.0], Db @ self._Cmom])               # node energy-moments (GEA via c_G)
        # kernel ridge regression on the REFERENCE block only (Tikhonov for the ill-conditioned Gram
        # matrix); backbone (HEG+GEA) diagonal stays at fp_ridge so its exact limits are untouched.
        ref_ridge = getattr(self.params, "fp_ref_ridge", None)
        ridge_diag = np.full(len(Xnodes), self._fp_ridge)
        if ref_ridge is not None:
            ridge_diag[2:] = float(ref_ridge)
        K = self._Kmat(Xnodes, Xnodes) + np.diag(ridge_diag)
        Kinv = np.linalg.solve(K, np.eye(len(K)))
        nl0 = self._n_out - 1
        dK1 = self._Kmat(x_heg, Xnodes)[0] * Xnodes[:, nl0] / self._fp_l1 ** 2   # dK/d(s^2) at HEG
        row = self._fp_kappa * (dK1 @ Kinv)                              # a1 = row . mu (linear in c_G)
        c_G = (_GEA2 - float(row @ mu)) / float(row[1] * self._fp_dgb)    # solve l=1 slope = 10/81
        Delta = np.vstack([np.zeros(self._n_out), c_G * self._dgea, Db])
        self._fp_Xnodes = Xnodes; self._fp_coef = np.linalg.solve(K, Delta); self._fp_cG = c_G

    def _kernel_eps(self, cprime, rho0, g):
        """eps_x from the kernel map; cprime (n_in,N) raw monopole coefficients (op@rho), rho0 (N,) density,
        g (N,) gradient -- the three rho-channels the adjoint differentiates through."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        R_ad, _ = self._R_ad(rho0)
        c_ad = self._c_ad(cprime, R_ad)
        d = c_ad / (4.0 * np.pi * R_ad ** 1.5)[:, None]
        Q = 4.0 * np.pi * R_ad ** 3 * (d @ self._Bmom); Qs = np.maximum(Q, 1e-12)
        W = enclosed_charge_switch(0.5 * Q)
        kF = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
        s2, _ = _bound((g / (2.0 * kF * rho0)) ** 2)
        cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
        t2 = None
        if self._use_l2:                                         # reduced l=2 (quadrupole) feature
            t2, _ = _bound((self._l2_op @ rho0 / (4.0 * kF ** 2 * rho0)) ** 2)
        # dimensionless hole shape rhotilde = rhotilde_LDA + kernel deviation; hole = -rho/2 * rhotilde
        rhotilde = self._rhotilde_lda[None, :] + self._Kmat(self._xfeat(cn, s2, t2), self._fp_Xnodes) @ self._fp_coef
        bulk = (-0.5 * rho0)[:, None] * rhotilde
        fa = -d / Qs[:, None]
        coeffs = (1.0 - W)[:, None] * bulk + W[:, None] * fa
        ontop = (1.0 - W) * (-0.5 * rho0) + W * (-rho0 / Qs)
        Bmom, R0, Cmom = self._Bmom, self._R0, self._Cmom
        a_row = 4.0 * np.pi * (R_ad ** 3)[:, None] * Bmom[None, :]
        row0 = np.sum(a_row * coeffs, axis=1); row1 = coeffs @ R0
        g00 = np.sum(a_row * a_row, axis=1); g01 = a_row @ R0; g11 = float(R0 @ R0)
        res0 = -1.0 - row0; res1 = ontop - row1; det = g00 * g11 - g01 ** 2
        lam0 = (g11 * res0 - g01 * res1) / det; lam1 = (-g01 * res0 + g00 * res1) / det
        coeffs = coeffs + lam0[:, None] * a_row + lam1[:, None] * R0[None, :]
        return 2.0 * np.pi * R_ad ** 2 * (coeffs @ Cmom)

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights; ewrho = ew * rho
        cprime = np.array([op @ rho for op in self._ops])   # raw monopole coefficient (writeup c'_{n00})
        g = self._grad_op @ rho
        eps = self._kernel_eps(cprime, rho, g)
        # exact variational discrete adjoint by FD in each rho-channel (cprime, local-rho, gradient)
        acc = np.zeros_like(rho)
        for nn in range(len(self._ops)):
            h = 1e-6 * (np.abs(cprime[nn]) + 1e-8)
            vp = cprime.copy(); vp[nn] += h
            vm = cprime.copy(); vm[nn] -= h
            deps_dvn = (self._kernel_eps(vp, rho, g) - self._kernel_eps(vm, rho, g)) / (2.0 * h)
            acc += self._ops[nn].T @ (ewrho * deps_dvn)
        hr = 1e-6 * (rho + 1e-8)
        deps_drho0 = (self._kernel_eps(cprime, rho + hr, g) - self._kernel_eps(cprime, rho - hr, g)) / (2.0 * hr)
        hg = 1e-6 * (np.abs(g) + 1e-8)
        deps_dg = (self._kernel_eps(cprime, rho, g + hg) - self._kernel_eps(cprime, rho, g - hg)) / (2.0 * hg)
        v_x = (eps + rho * deps_drho0 + acc / ew
               + self._grad_op.T @ (ewrho * deps_dg) / ew)
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEKERNELFPParameters:
        return SIMPLEHOLEKERNELFPParameters()