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

from ..descriptors.simple.derivatives import build_spectral_gradient_operator
from .evaluator import DensityData, XCParameters, XCPotentialData
from .simple_hole import SIMPLE_HOLE, SIMPLEHOLEParameters, _bound, _radial_basis
from .simple_hole_expansion_explicit import (
    charge_moments, coulomb_moments, enclosed_charge_switch, heg_hole,
    project_hole, radial_basis_at_origin,
)

_SIX2_3 = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
_GEA2 = 10.0 / 81.0   # second-order gradient-expansion coefficient F_x -> 1 + (10/81) s^2


@dataclass
class SIMPLEHOLEEXPParameters(SIMPLEHOLEParameters):
    """Direct-expansion hole settings. ``n_channels`` monopole channels resolve the hole
    (need n_channels >~ k_F R_c/pi where the density is high; see Phase-A report)."""
    functional_name: str = "SIMPLE_HOLE_EXPANSION"
    r_c: float = 8.0
    n_channels: int = 24
    n_rho_table: int = 96           # HEG-anchor table resolution in log(rho)
    rho_table_min: float = 1.0e-4
    rho_table_max: float = 1.0e2


class SIMPLE_HOLE_EXPANSION(SIMPLE_HOLE):
    """Exchange-only direct-expansion exchange hole, self-consistent. Parameter-free."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        p = self.params
        nch, rc = p.n_channels, p.r_c
        # per-basis-function moments and on-top values (closed form)
        self._a = charge_moments(nch, rc)               # enclosed charge a_n
        self._b = coulomb_moments(nch, rc)              # self-Coulomb b_n
        self._r0n = radial_basis_at_origin(nch, rc)     # R_{n0}(0)
        # constraint matrix A (sum rule row, on-top row) and its 2x2 Gram inverse
        self._A = np.vstack([4.0 * np.pi * self._a, self._r0n])      # (2, nch)
        self._Ginv = np.linalg.inv(self._A @ self._A.T)              # (2, 2)
        # HEG-anchor table rhotilde^HEG(rho0): project -(rho/2) S(k_F u) on a log-rho grid
        self._rho_tab = np.logspace(np.log10(p.rho_table_min), np.log10(p.rho_table_max),
                                    p.n_rho_table)
        self._heg_tab = np.array([project_hole(heg_hole(rho), rc, nch, nu=1024)
                                  for rho in self._rho_tab])          # (n_rho, nch)

    def _heg_anchor(self, rho0):
        """Interpolate the HEG-anchor coefficients at on-top densities rho0 (vectorized)."""
        lr = np.log10(np.clip(rho0, self._rho_tab[0], self._rho_tab[-1]))
        lt = np.log10(self._rho_tab)
        # column-wise linear interpolation -> (nch, N)
        return np.array([np.interp(lr, lt, self._heg_tab[:, n]) for n in range(self._a.size)])

    def _map_coeffs(self, C, rho0):
        """Parameter-free map: production monopole coeffs C (nch, N) and the local on-top
        density rho0 (N,) -> hole coeffs rhotilde (nch, N). The switch acts on the PER-SPIN
        enclosed charge Q/2 (exchange is same-spin); the SIC/few-electron anchor is the
        Fermi-Amaldi density-following hole -Cm/Q (int -> -1, on-top -rho0/Q)."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        Cm = C / (4.0 * np.pi)                                   # explicit convention
        Q = 4.0 * np.pi * (self._a @ Cm)                        # (N,) total enclosed charge
        Qsafe = np.maximum(Q, 1e-12)
        lam = enclosed_charge_switch(0.5 * Q)                   # per-spin switch (Q/2)
        coeffs = (1.0 - lam)[None, :] * self._heg_anchor(rho0) + lam[None, :] * (-Cm / Qsafe[None, :])
        # on-top target: HEG pair -rho/2 -> FA -rho/Q; sum rule always -1
        ontop = (1.0 - lam) * (-0.5 * rho0) + lam * (-rho0 / Qsafe)
        c = np.vstack([np.full_like(rho0, -1.0), ontop])        # (2, N) targets
        resid = c - self._A @ coeffs                            # (2, N)
        return coeffs + self._A.T @ (self._Ginv @ resid)        # (nch, N)

    def _eps_from_coeffs(self, C, rho0):
        """eps_x per point from monopole coeffs C (nch, N) and on-top density rho0 (N,)."""
        coeffs = self._map_coeffs(C, rho0)
        return 0.5 * 4.0 * np.pi * (self._b @ coeffs)           # (N,)

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
    gea_gate_c: float = 1.0          # gate strength: g = exp(-c * D_HEG); larger c => GEA needs
                                     # the density to be closer to HEG before turning on.
    gea_mu: float = 10.0 / 81.0      # GEA enhancement coefficient (F_x -> 1 + g*mu*s^2). The
                                     # s^2 slope is exactly mu in the s->0 limit (g->1); tune mu
                                     # only to set the *effective* enhancement at finite gradient
                                     # (the gate adds an O(s^4) self-saturation, see report).


class SIMPLE_HOLE_EXPANSION_GGA(SIMPLE_HOLE_EXPANSION):
    """Direct-expansion hole with the parameter-free second-order gradient correction.

    The charge- and on-top-neutral gradient deformation of the hole enhances the energy by the
    GEA2 factor, gated by how HEG-like the local density is:
        eps_x = eps_x^map * (1 + g(C) * mu * s^2_bounded),   s = |grad rho|/(2 k_F rho),
        g(C)  = exp(-c * D_HEG(C)),   D_HEG = sum_n (C_n/C_0 - (-1)^n/(n+1))^2 .
    s comes from the proven-stable l=1 spectral gradient operator (k_n^1, no stiff Laplacian);
    s^2 is smoothly saturated (``_bound``). mu = ``gea_mu`` (default 10/81), c = ``gea_gate_c``.

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
        """eps_x = eps_map(C, rho0) * (1 + g(C) * mu * s^2_b(g, rho0)), g = exp(-c D_HEG(C))."""
        p = self.params
        eps0 = self._eps_from_coeffs(C, rho0)
        gate = np.exp(-p.gea_gate_c * self._l2_distance_from_heg(C))
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
