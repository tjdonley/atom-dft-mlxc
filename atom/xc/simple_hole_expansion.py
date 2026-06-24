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
  Q    = 4 pi sum_n Cm_n a_n                 # enclosed density charge
  lam  = switch(Q)                           # 1 (Q<=1, one electron) -> 0 (Q>=2, HEG)
  rhotilde = (1-lam) rhotilde^HEG(rho0) + lam (-Cm)        # blend the two anchors
  rhotilde <- project to {sum rule = -1, on-top = -W rho0},  W = (1+lam)/2
  eps_x    = 1/2 * 4 pi * sum_n rhotilde_n b_n

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
        density rho0 (N,) -> hole coeffs rhotilde (nch, N)."""
        rho0 = np.maximum(np.asarray(rho0, float), 1e-12)
        Cm = C / (4.0 * np.pi)                                   # explicit convention
        Q = 4.0 * np.pi * (self._a @ Cm)                        # (N,) enclosed charge
        lam = enclosed_charge_switch(Q)                         # (N,)
        coeffs = (1.0 - lam)[None, :] * self._heg_anchor(rho0) + lam[None, :] * (-Cm)
        # project onto the two exact constraints (vectorized over columns)
        W = 0.5 * (1.0 + lam)
        c = np.vstack([np.full_like(rho0, -1.0), -W * rho0])    # (2, N) targets
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


class SIMPLE_HOLE_EXPANSION_GGA(SIMPLE_HOLE_EXPANSION):
    """Direct-expansion hole with the parameter-free second-order gradient correction.

    The charge- and on-top-neutral gradient deformation of the hole enhances the energy by
    the exact GEA2 factor in the slowly-varying limit:
        eps_x = eps_x^map * (1 + (10/81) s^2_bounded),   s = |grad rho| / (2 k_F rho).
    s comes from the proven-stable l=1 spectral gradient operator (k_n^1 growth, no stiff
    Laplacian). s^2 is smoothly saturated (Lieb-Oxford tail safety; reuses ``_bound``). The
    self-consistent potential adds the gradient channel via the spectral-operator transpose."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        self._grad_op = build_spectral_gradient_operator(self._r_grid)

    def _enhancement(self, rho, g):
        """f = 1 + (10/81) s^2_b and its partials ds2b/drho, ds2b/dg (chain rule helpers)."""
        rho = np.maximum(rho, 1e-12)
        d8 = 4.0 * _SIX2_3 * rho ** (8.0 / 3.0)
        s2 = g * g / d8
        s2b, db = _bound(s2)                      # smooth saturation + derivative
        f = 1.0 + _GEA2 * s2b
        ds2_drho = -(8.0 / 3.0) * s2 / rho        # at fixed g
        ds2_dg = 2.0 * g / d8
        return f, _GEA2 * db * ds2_drho, _GEA2 * db * ds2_dg

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])
        g = self._grad_op @ rho
        eps0 = self._eps_from_coeffs(C, rho)
        f, df_drho, df_dg = self._enhancement(rho, g)
        eps = eps0 * f

        # C-channel adjoint, f-weighted: sum_n P_n^T[ew rho f deps0/dC_n]
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps0_dCn = (self._eps_from_coeffs(Cp, rho) - self._eps_from_coeffs(Cm, rho)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * f * deps0_dCn)
        # explicit on-top-density derivative of eps0, f-weighted
        hr = 1e-6 * (rho + 1e-8)
        deps0_drho0 = (self._eps_from_coeffs(C, rho + hr) - self._eps_from_coeffs(C, rho - hr)) / (2.0 * hr)
        # gradient-channel: eps0 * df, split into the local rho part and the spectral g part
        v_x = (eps
               + rho * f * deps0_drho0
               + acc / ew
               + rho * eps0 * df_drho                             # local d f/d rho at fixed g
               + self._grad_op.T @ (ewrho * eps0 * df_dg) / ew)   # spectral gradient transpose
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEEXPGGAParameters:
        return SIMPLEHOLEEXPGGAParameters()
