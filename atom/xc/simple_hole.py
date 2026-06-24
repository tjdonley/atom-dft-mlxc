"""Self-consistent SIMPLE exchange-hole functional (exchange-only).

Reference implementation of the parameter-free exchange-hole functional of
``SIMPLE-Xhole-writeup`` (Sec. "Exchange-hole functional"). Notation and equation
labels below refer to that document (``main.tex`` / ``appendix.tex``).

Model hole [Eq. (exact-hole) ansatz]:

    n_x(r0, u) = -rho(r0+u) S(zeta(r0) u) / N(r0),   S(x) = [3 j_1(x)/x]^2 .

The 1/u Coulomb weight selects the monopole average, so the self-energy density is
a closed contraction of the SIMPLE monopole features C_n [Eq. (coulomb)], and the
hole self-energy is [Eq. (eps-x)]:

    C_n(r0) = (P_n rho)(r0)                      # fixed l=0 window operators
    Q_S(zeta)   = alpha(zeta) . C                # electrons the hole encloses
    Phi_S(zeta) = beta(zeta)  . C                # its self-Coulomb
    eps_x  = -1/2 Phi_S(zeta) / Q_S(zeta) ,      # exchange energy per electron

with the universal envelope tables (projections of S onto the radial basis)

    alpha_n(zeta) = int_0^Rc R_n0(u) S(zeta u) u^2 du
    beta_n(zeta)  = int_0^Rc R_n0(u) S(zeta u) u   du .

On-top scale [Sec. "Exchange-hole functional", App. "On-top scale by enclosed-charge
inversion"]: zeta(r0) is fixed by the on-top sum rule Q_S(zeta)=2. This is NOT an
inner self-consistency loop: Q_S(zeta)=alpha(zeta).C is monotonically decreasing in
zeta, so zeta is recovered by 1D inversion of the precomputed monotonic table -- the
same fixed-stencil enclosed-charge machinery used for the adaptive radius R_ad. The
Fermi-Amaldi branch is Q_S(zeta_min)<=2 (the window holds less than one pair).

Everything is convolutional, so the self-consistent exchange potential is the exact
discrete adjoint (variational derivative) of the energy [Eq. (adjoint),
Eq. (adjoint-discrete)]:

    v_x = eps_x + sum_n P_n^T[ ew rho d eps_x/d C_n ] / ew ,   ew = 4 pi r^2 w ,

with d eps_x/d C_n by finite difference. The -1/r tail is recovered by an
energy-neutral gauge fix + low-density damping [App. "Self-consistent potential and
numerical details"]; gauge_fix=False gives the pure adjoint (used by the FD test).
Exchange-only (e_c = v_c = 0) for direct comparison to OEP/EXX.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import spherical_jn

from ..descriptors.simple.derivatives import (
    build_spectral_gradient_operator,
    build_spectral_laplacian_operator,
    build_window_operator,
)
from .evaluator import DensityData, GenericXCResult, XCEvaluator, XCParameters, XCPotentialData


def _envelope(x):
    """HEG exchange-hole envelope S(x) = [3 j_1(x)/x]^2 = [j_0(x)+j_2(x)]^2.

    This is the homogeneous-gas hole shape; with it the on-top rule gives the exact
    LDA limit [Eq. (eps-x), limit (i)]."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    return (3.0 * spherical_jn(1, x) / x) ** 2


def _radial_basis(n, u, r_c):
    """Orthonormal SIMPLE monopole basis R_{n0}(u) = k_n sqrt(2/R_c) j_0(k_n u)."""
    k_n = (n + 1) * np.pi / r_c
    return k_n * np.sqrt(2.0 / r_c) * spherical_jn(0, k_n * u)


@dataclass
class SIMPLEHOLEParameters(XCParameters):
    """Settings for the convolutional exchange hole. ``n_channels`` is the number of
    monopole channels n_in (resolves the hole to the channel-count bound
    n_in >~ zeta R_c/pi); ``n_zeta`` is the zeta-grid used for the on-top table
    inversion Q_S(zeta)=2."""
    functional_name: str = 'SIMPLE_HOLE'
    r_c: float = 8.0
    n_channels: int = 24
    n_zeta: int = 48
    zeta_min: float = 1.0e-3
    zeta_max: float = 1.0e2
    n_window: int = 120
    n_angle: int = 48
    gauge_fix: bool = True   # subtract the asymptotic constant so v_x -> 0 at infinity (-1/r tail)


class SIMPLE_HOLE(XCEvaluator):
    """Exchange-only convolutional exchange-hole functional, self-consistent.

    The forward map (energy) and the discrete-adjoint potential both run through the
    fixed monopole operators {P_n}; the spherical-Bessel envelope enters only through
    the precomputed weight tables alpha(zeta), beta(zeta)."""

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        if r_quad is None or quadrature_weights is None:
            raise ValueError("SIMPLE_HOLE requires r_quad and quadrature_weights.")
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
        p = self.params
        r = np.asarray(r_quad, dtype=float)
        self._r_grid = r
        self.quadrature_weights = np.asarray(quadrature_weights, dtype=float)
        self.energy_weights = 4.0 * np.pi * r ** 2 * self.quadrature_weights
        self.zetas = np.logspace(np.log10(p.zeta_min), np.log10(p.zeta_max), p.n_zeta)

        # fixed monopole operators C_n = P_n @ rho  (l=0 windowed projections) [Eq. (coulomb)]
        self._ops = [
            build_window_operator(r, p.r_c, p=2,
                                  envelope=lambda u, n=n: _radial_basis(n, u, p.r_c),
                                  n_window=p.n_window, n_angle=p.n_angle)
            for n in range(p.n_channels)
        ]
        # universal envelope-projection weight tables alpha[j,n], beta[j,n] [Eq. (eps-x)]
        xu, wu = np.polynomial.legendre.leggauss(max(400, p.n_window * 3))
        u = 0.5 * p.r_c * (xu + 1.0)
        wq = 0.5 * p.r_c * wu
        Rb = np.array([_radial_basis(n, u, p.r_c) for n in range(p.n_channels)])  # (n_in, nu)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Sq = _envelope(self.zetas[:, None] * u[None, :])                     # (n_zeta, nu)
            self._alpha = (Sq * u ** 2) @ (Rb * wq).T   # (n_zeta, n_in)  alpha_n(zeta)=int R_n S u^2
            self._beta = (Sq * u) @ (Rb * wq).T         # (n_zeta, n_in)  beta_n(zeta) =int R_n S u

    # -- energy density from the monopole coefficients [Eq. (eps-x)] -------------- #
    def _eps_from_coeffs(self, C):
        """eps_x per point from C=(n_in,N): Q_S=alpha.C, Phi_S=beta.C, fix zeta from the
        on-top rule Q_S(zeta)=2 by 1D inversion of the monotonic table (NOT an inner
        loop; FA fallback zeta->zetas[0]), eps=-1/2 Phi_S/Q_S [Eq. (eps-x)]."""
        Q = self._alpha @ C                       # (n_zeta, N)
        Phi = self._beta @ C                      # (n_zeta, N)
        zetas = self.zetas
        N = C.shape[1]
        eps = np.empty(N)
        for i in range(N):
            Qi, Phii = Q[:, i], Phi[:, i]
            if Qi[0] <= 2.0:                      # window holds < one pair -> Fermi-Amaldi (ii)
                eps[i] = -0.5 * Phii[0] / Qi[0] if Qi[0] > 1e-30 else 0.0
            else:
                # enclosed-charge inversion: Q_S(zeta) decreases in zeta, solve Q_S=2
                zeta_sol = np.interp(2.0, Qi[::-1], zetas[::-1])
                eps[i] = -0.5 * np.interp(zeta_sol, zetas, Phii) / 2.0
        return eps

    _RHO_FLOOR = 1.0e-7   # rho/rho_max below which the discrete adjoint is numerically unreliable
    _RHO_DAMP = 1.0e-8    # rho_c/rho_max of the low-density adjoint damping f=rho^2/(rho^2+rho_c^2)

    def _gauge_offset(self, v_x, rho):
        """Spurious additive constant C in v_x -> C - Z/r, from a 2-param fit over the outer
        RESOLVED shell (above the low-density noise floor). Subtracting C sets v_x(inf)=0."""
        r = self._r_grid; rmax = rho.max()
        mask = (rho < 1e-3 * rmax) & (rho > self._RHO_FLOOR * rmax) & (r > 0)
        if mask.sum() >= 4:
            A = np.vstack([np.ones(mask.sum()), -1.0 / r[mask]]).T
            return float(np.linalg.lstsq(A, v_x[mask], rcond=None)[0][0])
        idx = np.where(rho > self._RHO_FLOOR * rmax)[0]
        return float(v_x[idx[np.argmax(r[idx])]]) if idx.size else 0.0

    def _apply_gauge(self, v_x, eps, rho):
        """Gauge-fix + low-density damping [App. "Self-consistent potential ..."]. Subtract the
        asymptotic constant C so v_x -> 0 at infinity, and smoothly damp the (numerically
        unreliable) adjoint correction to zero as rho -> 0:  v_x = eps + f(rho) (v_x - eps - C),
        f = rho^2/(rho^2 + rho_c^2). At bulk density f->1 (gauge-fixed full potential); in the
        density tail f->0 so v_x -> eps. Energy-neutral (does not change e_x)."""
        C = self._gauge_offset(v_x, rho)
        rho_c = self._RHO_DAMP * rho.max()
        f = rho ** 2 / (rho ** 2 + rho_c ** 2)
        return eps + f * (v_x - eps - C)

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        C = np.array([op @ rho for op in self._ops])        # (n_in, N) monopole coeffs [Eq. (coulomb)]
        eps = self._eps_from_coeffs(C)

        # discrete-adjoint (Sahoo) potential [Eq. (adjoint-discrete)]:
        #   v = eps + sum_n P_n^T[ew rho deps/dC_n]/ew
        ewrho = ew * rho
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._eps_from_coeffs(Cp) - self._eps_from_coeffs(Cm)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        v_x = eps + acc / ew
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)

        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEParameters:
        return SIMPLEHOLEParameters()

    # compute_xc is overridden directly; the generic hooks are unused.
    def compute_exchange_generic(self, density_data: DensityData) -> GenericXCResult:
        raise NotImplementedError("SIMPLE_HOLE overrides compute_xc directly.")

    def compute_correlation_generic(self, density_data: DensityData) -> GenericXCResult:
        raise NotImplementedError("SIMPLE_HOLE overrides compute_xc directly.")


# =====================================================================
# Deterministic fourth-order GEA deformation of the hole [Eq. (fx), (ex-gga)]
# =====================================================================
_SIX2_3 = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
# GEA enhancement-factor coefficients [Eq. (fx)]: F_x = 1 + a s^2 + b q^2 + c s^2 q
_GEA_A, _GEA_B, _GEA_C = 10.0 / 81.0, 146.0 / 2025.0, -73.0 / 405.0
_INV_BOUND = 4.0          # smooth saturation of the reduced gradient/Laplacian (tail safety)


def _g0(x):
    """Bare HEG amplitude g0(x) = 3 j_1(x)/x = j_0(x)+j_2(x) (g0(0)=1)."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    return 3.0 * spherical_jn(1, x) / x


def _bound(v):
    """Smooth saturation v -> v/(1+|v|/M); returns (value, d/dv)."""
    den = 1.0 + np.abs(v) / _INV_BOUND
    return v / den, 1.0 / den ** 2


@dataclass
class SIMPLEHOLEGEAParameters(SIMPLEHOLEParameters):
    """Settings for the deterministic GEA-deformed hole [Eq. (fx)]. ``mode`` is the
    single envelope deformation mode (l, kappa) with l>=1 (phi(0)=0 keeps the on-top
    value); its amplitude carries the GEA combination of invariants, with coefficients
    fixed (not fit) by the gradient expansion."""
    functional_name: str = "SIMPLE_HOLE_GEA"
    mode: tuple = (1, 1.0)


class SIMPLE_HOLE_GEA(SIMPLE_HOLE):
    """DEPRECATED / experimental: exchange hole with the deterministic fourth-order GEA
    deformation, carrying (10/81) s^2 + (146/2025) q^2 - (73/405) s^2 q via the envelope
    amplitude. This variant requires the reduced Laplacian q (l=0 spectral operator),
    which is numerically unstable (writeup App. "Practical limits") and is NOT carried in
    the production functional. The production hole is the second-order, gradient-only
    SIMPLE_HOLE_GGA below; SIMPLE_HOLE_GEA is retained only for reference/diagnostics.

    Construction (for reference): the HEG envelope is deformed, S(x;c) = [g0(x) + c
    phi(x)]^2 with phi = j_l(kappa x) (l>=1); the prefactor rho_resp = dF_x/dc is a
    one-time HEG calibration, so the small-gradient expansion of F_x = eps_x/eps_x^unif
    reproduces the GEA coefficients by construction (imposed, not fit). The HEG limit
    (gradients -> 0 => bare hole) and Fermi-Amaldi limit (phi(0)=0 => on-top untouched)
    are preserved structurally. The q (l=0) channel does not converge robustly in SCF."""

    _USE_LAP = True   # SIMPLE_HOLE_GGA sets False (gradient-only; no stiff l=0 channel)

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad,
                         quadrature_weights=quadrature_weights, params=params)
        p = self.params
        ell, kappa = p.mode
        # linearized envelope-deformation tables: dS(x) = 2 g0(x) phi(x), projected onto
        # the monopole basis with (dalpha, u^2 du) and without (dbeta, u du) the 1/u weight.
        xu, wu = np.polynomial.legendre.leggauss(max(400, p.n_window * 3))
        u = 0.5 * p.r_c * (xu + 1.0)
        wq = 0.5 * p.r_c * wu
        Rb = np.array([_radial_basis(n, u, p.r_c) for n in range(p.n_channels)])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dS = 2.0 * _g0(self.zetas[:, None] * u[None, :]) * \
                spherical_jn(int(ell), kappa * self.zetas[:, None] * u[None, :])
            self._dalpha = (dS * u ** 2) @ (Rb * wq).T          # (n_zeta, n_in)
            self._dbeta = (dS * u) @ (Rb * wq).T
        # gradient / Laplacian operators for the invariants [Eq. (sq)]. The Laplacian
        # (q) is built only for the full GEA; the gradient-only fallback omits it.
        self._grad_op = build_spectral_gradient_operator(self._r_grid)
        self._lap_op = build_spectral_laplacian_operator(self._r_grid) if self._USE_LAP else None
        # GEA enhancement coefficients [Eq. (fx)]; gradient-only drops the q^2, s^2 q terms.
        self._gea = (_GEA_A, _GEA_B, _GEA_C) if self._USE_LAP else (_GEA_A, 0.0, 0.0)
        self._rho_resp = self._calibrate_response()             # dF_x/dc at HEG

    # -- deformed self-energy ------------------------------------------------ #
    def _eps_def(self, C, cfield):
        """eps_x per point with the deformed envelope: Q_S=alpha.C + c (dalpha.C),
        Phi_S likewise; resolve zeta from Q_S(zeta)=2 (FA fallback), eps=-1/2 Phi_S/Q_S."""
        Q = self._alpha @ C + cfield[None, :] * (self._dalpha @ C)   # (n_zeta, N)
        Phi = self._beta @ C + cfield[None, :] * (self._dbeta @ C)
        zetas = self.zetas
        eps = np.empty(C.shape[1])
        for i in range(C.shape[1]):
            Qi, Phii = Q[:, i], Phi[:, i]
            if Qi[0] <= 2.0:
                eps[i] = -0.5 * Phii[0] / Qi[0] if Qi[0] > 1e-30 else 0.0
            else:
                z = np.interp(2.0, Qi[::-1], zetas[::-1])
                eps[i] = -0.5 * np.interp(z, zetas, Phii) / 2.0
        return eps

    def _calibrate_response(self):
        """One-time HEG calibration of rho_resp = dF_x/dc (the envelope response), so that
        c = (1/rho_resp)(GEA combination) gives F_x matching Eq. (fx) to linear order."""
        rho0 = np.full(self._r_grid.size, 0.1)
        C = np.array([op @ rho0 for op in self._ops])
        i = self._r_grid.size // 2                              # interior reference point
        eps_unif = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho0[i] ** (1.0 / 3.0)
        h = 1e-4
        cp = np.zeros(self._r_grid.size); cp[i] = h
        cm = np.zeros(self._r_grid.size); cm[i] = -h
        d_eps = (self._eps_def(C, cp)[i] - self._eps_def(C, cm)[i]) / (2.0 * h)
        return d_eps / eps_unif

    def _amplitude(self, rho, g, lap):
        """Deformation amplitude c(r0) and its partials (dc/drho, dc/dg, dc/dlap),
        carrying the GEA combination (10/81)s^2 + (146/2025)q^2 - (73/405)s^2 q."""
        rho = np.maximum(rho, 1e-12)
        d8 = 4.0 * _SIX2_3 * rho ** (8.0 / 3.0)
        d5 = 4.0 * _SIX2_3 * rho ** (5.0 / 3.0)
        s2, q = g * g / d8, lap / d5                            # reduced gradient^2, reduced Laplacian
        s2b, ds2 = _bound(s2)
        qb, dq = _bound(q)
        # partials of the raw invariants
        s2_rho, s2_g = -(8.0 / 3.0) * s2 / rho, 2.0 * g / d8
        q_rho, q_lap = -(5.0 / 3.0) * q / rho, 1.0 / d5
        A, B, C = self._gea
        inv = 1.0 / self._rho_resp
        # c = inv (A s2b + B qb^2 + C s2b qb)
        c = inv * (A * s2b + B * qb ** 2 + C * s2b * qb)
        dc_ds2b = inv * (A + C * qb)
        dc_dqb = inv * (2.0 * B * qb + C * s2b)
        dc_drho = dc_ds2b * ds2 * s2_rho + dc_dqb * dq * q_rho
        dc_dg = dc_ds2b * ds2 * s2_g
        dc_dlap = dc_dqb * dq * q_lap
        return c, dc_drho, dc_dg, dc_dlap

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.maximum(np.asarray(density_data.rho, dtype=float), 1e-12)
        ew = self.energy_weights
        ewrho = ew * rho
        C = np.array([op @ rho for op in self._ops])
        g = self._grad_op @ rho
        lap = self._lap_op @ rho if self._lap_op is not None else np.zeros_like(rho)
        c, dc_drho, dc_dg, dc_dlap = self._amplitude(rho, g, lap)
        eps = self._eps_def(C, c)

        # monopole-channel adjoint (stable): deps/dC_n by FD [Eq. (adjoint-discrete)]
        acc = np.zeros_like(rho)
        for n in range(len(self._ops)):
            h = 1e-6 * (np.abs(C[n]) + 1e-8)
            Cp = C.copy(); Cp[n] += h
            Cm = C.copy(); Cm[n] -= h
            deps_dCn = (self._eps_def(Cp, c) - self._eps_def(Cm, c)) / (2.0 * h)
            acc += self._ops[n].T @ (ewrho * deps_dCn)
        # deformation-channel adjoint: deps/dc by FD, chained through c(rho,g,lap).
        # The g/lap pieces (the GEA gradient terms) are the stiff channel -- in the SCF
        # they run under the frozen-gradient dual loop (writeup App.; Phase D).
        hc = 1e-6 * (np.abs(c) + 1e-8)
        deps_dc = (self._eps_def(C, c + hc) - self._eps_def(C, c - hc)) / (2.0 * hc)
        v_x = (eps + acc / ew
               + deps_dc * dc_drho
               + self._grad_op.T @ (ewrho * deps_dc * dc_dg) / ew)
        if self._lap_op is not None:                            # the stiff l=0 channel
            v_x = v_x + self._lap_op.T @ (ewrho * deps_dc * dc_dlap) / ew
        if getattr(self.params, "gauge_fix", True):
            v_x = self._apply_gauge(v_x, eps, rho)
        zero = np.zeros_like(rho)
        return XCPotentialData(v_x=v_x, v_c=zero, e_x=eps, e_c=zero,
                               de_x_dtau=None, de_c_dtau=None)

    def _default_params(self) -> SIMPLEHOLEGEAParameters:
        return SIMPLEHOLEGEAParameters()


@dataclass
class SIMPLEHOLEGGAParameters(SIMPLEHOLEGEAParameters):
    functional_name: str = "SIMPLE_HOLE_GGA"


class SIMPLE_HOLE_GGA(SIMPLE_HOLE_GEA):
    """PRODUCTION exchange hole (the functional of the writeup): the second-order,
    scale-free GEA-deformed hole matching $F_x\\to1+(10/81)s^2$ [Eq. (fx)] via the
    $\\ell=1$ gradient only, with the reduced Laplacian $q$ excluded (the $q^2$ and
    $s^2q$ terms dropped). Because there is no $\\ell=0$ Laplacian channel, the stiff,
    unstable $q$ adjoint is absent entirely and the functional is self-consistent and
    stable; its $\\ell=1$ adjoint grows only as $k_n^1$ and is well-conditioned. This is
    the hole carried throughout the paper; SIMPLE_HOLE_GEA (4th-order) is deprecated."""
    _USE_LAP = False

    def _default_params(self) -> SIMPLEHOLEGGAParameters:
        return SIMPLEHOLEGGAParameters()
