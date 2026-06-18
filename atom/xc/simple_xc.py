"""SIMPLE on Jacob's ladder: standard functionals fed only SIMPLE descriptors.

This module implements one clean idea. A standard density functional needs, at
each point, a few *local ingredients*:

    LDA   : rho
    GGA   : rho, |grad rho|                 (PBE)
    mGGA  : rho, |grad rho|, tau            (r2SCAN; tau = kinetic energy density)

SIMPLE reconstructs the gradient and Laplacian of the density as fixed *linear
operators* on the radial density (the l=1 and l=0 channels of the local
multipole expansion -- see ``descriptors/simple/derivatives.py``):

    g   = G @ rho   ~ d rho/dr        (l = 1 gradient operator)
    lap = L @ rho   ~ grad^2 rho      (l = 0 Laplacian operator)

and the kinetic energy density is *deorbitalized* by the second-order gradient
expansion, tau ~= tau_unif + |grad rho|^2/(72 rho) + grad^2 rho/6, so that a
meta-GGA needs no orbitals. Feeding (rho, |g|, tau[g, lap]) -- and ONLY these --
to the standard functional gives a direct map

    SIMPLE descriptors  -->  LDA / GGA / meta-GGA

up Jacob's ladder. Because g and lap are linear in rho, the exchange-correlation
potential is the exact discrete adjoint

    v = v_local + G^T[ew rho de/dg]/ew + L^T[ew rho de/dlap]/ew,   ew = 4 pi r^2 w,

with the per-node derivatives of the (standard) energy density taken by finite
difference. Building G and L explicitly and using their literal transposes makes
v equal the finite-difference derivative of the discrete energy
E = sum_i ew_i rho_i (e_x + e_c)_i (verified in tests/simple/test_simple_xc.py).

Two rungs are exposed:
  * ``SIMPLE_GGA``  -- PBE   from (rho, g)            [GGA]
  * ``SIMPLE_SCAN`` -- r2SCAN from (rho, g, lap)      [meta-GGA, deorbitalized]
The LDA rung is just the standard LDA of rho (no reconstruction needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..descriptors.simple.derivatives import (
    build_spectral_gradient_operator,
    build_spectral_laplacian_operator,
)
from ..descriptors.simple.params import R_C
from .evaluator import DensityData, GenericXCResult, XCEvaluator, XCParameters, XCPotentialData
from .gga_pbe import GGA_PBE
from .meta_scan import rSCAN, r2SCAN

_SIX3_2_3 = (3.0 * np.pi**2) ** (2.0 / 3.0)
_BASE = {'GGA_PBE': GGA_PBE, 'r2SCAN': r2SCAN, 'RSCAN': rSCAN}


def _model_tau(rho, g, lap):
    """Deorbitalized kinetic energy density (2nd-order gradient expansion):
    tau ~= tau_unif + |grad rho|^2/(72 rho) + grad^2 rho / 6."""
    rho = np.maximum(rho, 1e-12)
    return 0.3 * _SIX3_2_3 * rho ** (5.0 / 3.0) + g**2 / (72.0 * rho) + lap / 6.0


@dataclass
class SIMPLEXCParameters(XCParameters):
    """Reconstruction-operator settings shared by the SIMPLE ladder functionals.
    ``base`` names the standard functional being fed; the SIMPLE block controls
    the l=1 gradient and l=0 Laplacian SPECTRAL operators [Eq. (sq)], which use the
    full window r_c and ~40 channels (no decoder)."""
    functional_name: str = 'SIMPLE_XC'
    base: str = 'GGA_PBE'
    r_c: float = R_C
    n_channels: int = 40
    n_window: int = 256
    n_angle: int = 64


class _SIMPLEFunctional(XCEvaluator):
    """Base class: reconstruct the ladder ingredients from SIMPLE operators, feed
    them to a standard base functional (exchange AND correlation), and return the
    discrete-adjoint XC potential. Subclasses set ``_BASE_NAME`` and whether the
    Laplacian (meta-GGA) is used."""

    _BASE_NAME = 'GGA_PBE'
    _USE_LAPLACIAN = False

    def __init__(self, derivative_matrix=None, r_quad=None,
                 quadrature_weights=None, params: Optional[XCParameters] = None):
        if derivative_matrix is None or r_quad is None:
            raise ValueError(f"{type(self).__name__} requires derivative_matrix and r_quad.")
        if quadrature_weights is None:
            raise ValueError(f"{type(self).__name__} requires quadrature_weights.")
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
        self.quadrature_weights = np.asarray(quadrature_weights, dtype=float)
        self.energy_weights = 4.0 * np.pi * np.asarray(r_quad, dtype=float) ** 2 * self.quadrature_weights
        base_name = getattr(self.params, 'base', self._BASE_NAME)
        self._base = _BASE[base_name](derivative_matrix=derivative_matrix, r_quad=r_quad)
        p = self.params
        self.gradient_operator = build_spectral_gradient_operator(
            np.asarray(r_quad, dtype=float), n_channels=p.n_channels,
            r_c=p.r_c, n_window=p.n_window, n_angle=p.n_angle)
        self.laplacian_operator = None
        if self._USE_LAPLACIAN:
            self.laplacian_operator = build_spectral_laplacian_operator(
                np.asarray(r_quad, dtype=float), n_channels=p.n_channels,
                r_c=p.r_c, n_window=p.n_window, n_angle=p.n_angle)

    # -- SIMPLE ingredients ------------------------------------------------- #
    def _ingredients(self, rho):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            g = self.gradient_operator @ rho
            lap = self.laplacian_operator @ rho if self._USE_LAPLACIAN else np.zeros_like(rho)
        return g, lap

    def _density_data(self, rho, g, lap):
        tau = _model_tau(rho, g, lap) if self._USE_LAPLACIAN else None
        return DensityData(rho=np.maximum(rho, 1e-12), grad_rho=np.abs(g), tau=tau)

    def _e_x(self, rho, g, lap):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return np.asarray(self._base.compute_exchange_generic(
                self._density_data(rho, g, lap)).e_generic, dtype=float)

    def _e_c(self, rho, g, lap):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return np.asarray(self._base.compute_correlation_generic(
                self._density_data(rho, g, lap)).e_generic, dtype=float)

    # -- discrete-adjoint potential of a SIMPLE-fed energy density ----------- #
    def _adjoint_potential(self, rho, g, lap, e_fn):
        """v = d(rho e)/drho|_local + G^T[ew rho de/dg]/ew + L^T[ew rho de/dlap]/ew,
        with de/drho|_{g,lap}, de/dg, de/dlap by central finite difference."""
        ew = self.energy_weights
        e = e_fn(rho, g, lap)
        hr = 1e-6 * np.maximum(np.abs(rho), 1e-8)
        hg = 1e-6 * (np.abs(g) + 1e-8)
        de_drho = (e_fn(rho + hr, g, lap) - e_fn(rho - hr, g, lap)) / (2 * hr)
        de_dg = (e_fn(rho, g + hg, lap) - e_fn(rho, g - hg, lap)) / (2 * hg)
        v = e + rho * de_drho + (self.gradient_operator.T @ (ew * rho * de_dg)) / ew
        if self._USE_LAPLACIAN:
            hl = 1e-6 * (np.abs(lap) + 1e-8)
            de_dlap = (e_fn(rho, g, lap + hl) - e_fn(rho, g, lap - hl)) / (2 * hl)
            v = v + (self.laplacian_operator.T @ (ew * rho * de_dlap)) / ew
        return v

    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        rho = np.asarray(density_data.rho, dtype=float)
        g, lap = self._ingredients(rho)
        e_x = self._e_x(rho, g, lap)
        e_c = self._e_c(rho, g, lap)
        v_x = self._adjoint_potential(rho, g, lap, self._e_x)
        v_c = self._adjoint_potential(rho, g, lap, self._e_c)
        return XCPotentialData(v_x=v_x, v_c=v_c, e_x=e_x, e_c=e_c,
                               de_x_dtau=None, de_c_dtau=None)

    # abstract methods: unused (compute_xc is overridden)
    def compute_exchange_generic(self, density_data: DensityData) -> GenericXCResult:
        raise NotImplementedError(f"{type(self).__name__} overrides compute_xc directly.")

    def compute_correlation_generic(self, density_data: DensityData) -> GenericXCResult:
        raise NotImplementedError(f"{type(self).__name__} overrides compute_xc directly.")


@dataclass
class SIMPLEGGAParameters(SIMPLEXCParameters):
    """SIMPLE -> PBE (GGA rung)."""
    functional_name: str = 'SIMPLE_GGA'
    base: str = 'GGA_PBE'


@dataclass
class SIMPLESCANParameters(SIMPLEXCParameters):
    """SIMPLE -> rSCAN (meta-GGA rung), deorbitalized (exchange AND correlation).
    Both rSCAN and r2SCAN regularize alpha so the deorbitalized form is bounded;
    in this 1D code rSCAN's regularization converges the more reliably, so it is
    the default. Set ``base='r2SCAN'`` for the r2SCAN form."""
    functional_name: str = 'SIMPLE_SCAN'
    base: str = 'RSCAN'


class SIMPLE_GGA(_SIMPLEFunctional):
    """GGA rung: PBE fed (rho, |grad rho|) from SIMPLE (gradient only)."""
    _BASE_NAME = 'GGA_PBE'
    _USE_LAPLACIAN = False

    def _default_params(self) -> SIMPLEGGAParameters:
        return SIMPLEGGAParameters()


class SIMPLE_SCAN(_SIMPLEFunctional):
    """meta-GGA rung: r2SCAN fed (rho, |grad rho|, tau) from SIMPLE (gradient +
    Laplacian, tau deorbitalized)."""
    _BASE_NAME = 'RSCAN'
    _USE_LAPLACIAN = True

    def _default_params(self) -> SIMPLESCANParameters:
        return SIMPLESCANParameters()
