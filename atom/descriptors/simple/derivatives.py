"""Density-to-ingredient operators for the SIMPLE functionals.

Two families, both fixed linear maps on the radial density (assembled once):

  * the parameter-free SPECTRAL gradient / Laplacian operators that reconstruct the
    reduced gradient s and reduced Laplacian q [Eq. (sq) of SIMPLE-Xhole-writeup];
  * the windowed-convolution / Coulomb operator that produces the SIMPLE monopole
    descriptors C_n and the Coulomb contraction [Eq. (coulomb)].

These are the "written" (spectral) operators; the legacy moment/calibrated decoders
of earlier versions are dropped. All operators are fixed convolutions, so their
functional derivative is the literal transpose (Sahoo et al.) [Eq. (adjoint)].
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .bessel import radial_gauss_grid
from .params import R_C
from .pipeline import window_basis

_SIXPI2_1_3 = (6.0 * np.pi ** 2) ** (1.0 / 3.0)
_FOUR_3PI2_2_3 = 4.0 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)


# =============================================================================
# Reduced gradient / Laplacian from the reconstructed derivatives [Eq. (sq)]
# =============================================================================
def reduced_gradient_from_grad(grad_rho, rho):
    """Reduced density gradient s = |grad rho| / (2 k_F rho) [Eq. (sq)], in the
    spin-unpolarized PBE-exchange convention (rho_up = rho_down = rho/2):
    s = |grad rho| / (2 (6 pi^2)^{1/3} (rho/2)^{4/3})."""
    rho = np.asarray(rho, dtype=float)
    rho_updn = np.maximum(rho / 2.0, 1e-300)
    return np.asarray(grad_rho, dtype=float) / (2.0 * _SIXPI2_1_3 * rho_updn ** (4.0 / 3.0))


def reduced_laplacian_from_grad(lap_rho, rho):
    """Reduced Laplacian q = grad^2 rho / (4 k_F^2 rho) = grad^2 rho /
    (4 (3 pi^2)^{2/3} rho^{5/3}) [Eq. (sq)] (total-density convention)."""
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-300)
    return np.asarray(lap_rho, dtype=float) / (_FOUR_3PI2_2_3 * rho ** (5.0 / 3.0))


# =============================================================================
# Axial multipole profiles (the l=0, l=1 angular averages of the local density)
# =============================================================================
def axial_multipole_profile(rho, r0, r, l, n_angle=96):
    """l-th axial multipole rho_l(r; r0) of a radial density ``rho`` (callable)
    about the off-center point at radius r0:
        rho_l(r;r0) = (2l+1)/2 int_{-1}^1 P_l(u) rho(sqrt(r0^2+r^2-2 r r0 u)) du.
    Only l=0 (P_0=1, the spherical average of Eq. (coulomb)) and l=1 (P_1=u, the
    dipole used for the gradient) are needed."""
    if l not in (0, 1):
        raise ValueError("axial_multipole_profile supports l in {0, 1}.")
    u, wu = leggauss(n_angle)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    dist = np.sqrt(np.maximum(
        r0 ** 2 + r[:, None] ** 2 - 2.0 * r[:, None] * r0 * u[None, :], 0.0))
    pl = u if l == 1 else np.ones_like(u)
    return (2 * l + 1) / 2.0 * np.sum(wu * pl[None, :] * rho(dist), axis=1)


# =============================================================================
# Spectral gradient / Laplacian operators [Eq. (sq)]
# =============================================================================
def build_spectral_laplacian_operator(r_grid, n_channels=40, r_c=R_C,
                                      n_window=256, n_angle=64):
    """Fixed linear operator L (N x N) for grad^2 rho via the SPECTRAL
    (Bessel-eigenvalue) sum [Eq. (sq), l=0]: the window functions are Laplacian
    eigenfunctions, grad^2[j_0(k_n u)] = -k_n^2 j_0(k_n u), so for the monopole
    expansion rho_0(u;r0) = sum_n alpha_n R_{n0}(u),

        grad^2 rho(r0) = - sum_n alpha_n k_n^2 R_{n0}(0),   k_n = (n+1) pi / r_c .

    Parameter-free (no decoder), full window. Recovers grad^2 rho to ~1% in the
    core/valence of smooth (pseudopotential) atoms. Returns ``L`` (N x N).

    NOTE: requires a SMOOTH density; a bare nuclear cusp breaks the expansion
    (this is an exchange-only, pseudopotential-density model)."""
    basis = window_basis(0, n_channels)
    wq = radial_gauss_grid(r_c, n_window)
    radial = basis.evaluate(0, wq.nodes)                  # (n_channels, n_window)
    k = (np.arange(n_channels) + 1) * np.pi / r_c
    R0 = basis.evaluate(0, np.array([1.0e-7]))[:, 0]      # R_{n0}(0)
    spec_w = -k ** 2 * R0                                  # spectral eigenvalue weights
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        proj = spec_w @ radial                            # fold weights + basis -> (n_window,)
    u, wu = leggauss(n_angle)
    r_grid = np.asarray(r_grid, dtype=float)
    n = r_grid.size
    operator = np.zeros((n, n))
    # rho_0(u;r0) = (1/2) int rho(dist) dmu (l=0 axial profile, P_0=1); orthonormal alpha_n.
    kernel = 0.5 * (wq.weights * wq.nodes ** 2 * proj)[:, None] * wu[None, :]
    kflat = kernel.ravel()
    lo, hi = r_grid[0], r_grid[-1]
    for i, r0 in enumerate(r_grid):
        dist = np.sqrt(np.maximum(
            r0 ** 2 + wq.nodes[:, None] ** 2 - 2.0 * wq.nodes[:, None] * r0 * u[None, :], 0.0
        )).ravel()
        dist = np.clip(dist, lo, hi)
        idx = np.clip(np.searchsorted(r_grid, dist) - 1, 0, n - 2)
        left, right = r_grid[idx], r_grid[idx + 1]
        frac = np.where(right > left, (dist - left) / np.where(right > left, right - left, 1.0), 0.0)
        np.add.at(operator[i], idx, kflat * (1.0 - frac))
        np.add.at(operator[i], idx + 1, kflat * frac)
    # Constant annihilation [Eq. (sq) / Numerical implementation]: each row sums to
    # zero so grad^2(const)=0 (HEG/LDA limit) and the DC truncation residual is removed.
    operator -= np.diag(operator.sum(axis=1))
    return operator


def build_spectral_gradient_operator(r_grid, n_channels=40, r_c=R_C,
                                     n_window=256, n_angle=64):
    """Fixed linear operator G (N x N) for the signed radial gradient rho'(r0) via the
    SPECTRAL slope-at-origin sum [Eq. (sq), l=1]: for the l=1 axial expansion
    rho_1 = sum_n alpha_n R_{n1}(u),

        rho'(r0) = rho_1'(0) = sum_n alpha_n R_{n1}'(0),   R_{n1}'(0) = norm_n k_n / 3 .

    Parameter-free, full window, stable for smooth densities. The reduced gradient is
    then s = |G @ rho| / (2 k_F rho) [reduced_gradient_from_grad]. Returns ``G`` (N x N)
    (sign is convention; use |.|)."""
    basis = window_basis(1, n_channels)
    wq = radial_gauss_grid(r_c, n_window)
    radial = basis.evaluate(1, wq.nodes)                  # (n_channels, n_window)
    slope0 = basis.evaluate(1, np.array([1.0e-6]))[:, 0] / 1.0e-6   # R_{n1}'(0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        proj = slope0 @ radial                            # fold slope weights + basis
    u, wu = leggauss(n_angle)
    r_grid = np.asarray(r_grid, dtype=float)
    n = r_grid.size
    operator = np.zeros((n, n))
    # rho_1(u;r0) = (3/2) int mu rho(dist) dmu (l=1 axial profile, P_1=mu); orthonormal alpha_n.
    kernel = 1.5 * (wq.weights * wq.nodes ** 2 * proj)[:, None] * (wu * u)[None, :]
    kflat = kernel.ravel()
    lo, hi = r_grid[0], r_grid[-1]
    for i, r0 in enumerate(r_grid):
        dist = np.sqrt(np.maximum(
            r0 ** 2 + wq.nodes[:, None] ** 2 - 2.0 * wq.nodes[:, None] * r0 * u[None, :], 0.0
        )).ravel()
        dist = np.clip(dist, lo, hi)
        idx = np.clip(np.searchsorted(r_grid, dist) - 1, 0, n - 2)
        left, right = r_grid[idx], r_grid[idx + 1]
        frac = np.where(right > left, (dist - left) / np.where(right > left, right - left, 1.0), 0.0)
        np.add.at(operator[i], idx, kflat * (1.0 - frac))
        np.add.at(operator[i], idx + 1, kflat * frac)
    # Constant annihilation [Eq. (sq)]: each row sums to zero so rho'(const)=0 (HEG limit).
    operator -= np.diag(operator.sum(axis=1))
    return operator


# =============================================================================
# Windowed-convolution / Coulomb operator [Eq. (coulomb)]
# =============================================================================
def build_window_operator(r_grid, r_c, p=1, alpha=None, envelope=None,
                          n_window=120, n_angle=48):
    """Fixed linear windowed-convolution operator W with kernel K(u)=u^{p-2} e^{-alpha u} g(u):

        (W @ rho)(r0) = int_{|u|<r_c} K(|u|) rho(r0 + u) d^3u
                      = 2 pi int_0^{r_c} u^p e^{-alpha u} g(u) du int_{-1}^1 dmu rho(dist).

    p = 1 is the Coulomb potential (K = 1/u): the radial Coulomb contraction of
    Eq. (coulomb), Sum_n w_n C_n with the u-measure reduced from u^2 du to u du.
    p = 2 is the windowed charge (K = 1): int rho over the ball. ``envelope`` g(u)
    is an optional radial weight -- e.g. the monopole basis R_{n0}(u) (giving the
    SIMPLE monopole descriptors C_n = P_n @ rho of the exchange hole), or the HEG
    hole shape S(zeta u) (giving Phi_S / Q_S). ``alpha`` adds Yukawa screening.
    Fixed convolution => adjoint = W^T [Eq. (adjoint)]. Returns the (N, N) operator."""
    wq = radial_gauss_grid(r_c, n_window)
    mu, wmu = leggauss(n_angle)
    r_grid = np.asarray(r_grid, dtype=float)
    n = r_grid.size
    screen = np.exp(-alpha * wq.nodes) if alpha is not None else 1.0
    env = envelope(wq.nodes) if envelope is not None else 1.0
    radial_w = wq.weights * wq.nodes ** p * screen * env
    kernel = (2.0 * np.pi) * radial_w[:, None] * wmu[None, :]
    kflat = kernel.ravel()
    operator = np.zeros((n, n))
    lo, hi = r_grid[0], r_grid[-1]
    for i, r0 in enumerate(r_grid):
        dist = np.sqrt(np.maximum(
            r0 ** 2 + wq.nodes[:, None] ** 2 - 2.0 * wq.nodes[:, None] * r0 * mu[None, :], 0.0
        )).ravel()
        dist = np.clip(dist, lo, hi)
        idx = np.clip(np.searchsorted(r_grid, dist) - 1, 0, n - 2)
        left, right = r_grid[idx], r_grid[idx + 1]
        frac = np.where(right > left, (dist - left) / np.where(right > left, right - left, 1.0), 0.0)
        np.add.at(operator[i], idx, kflat * (1.0 - frac))
        np.add.at(operator[i], idx + 1, kflat * frac)
    return operator
