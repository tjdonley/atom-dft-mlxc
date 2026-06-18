"""The SIMPLE descriptor pipeline (reference implementation).

Pipeline (Secs. 2-7 of docs/SIMPLE/SIMPLE.tex): fixed-window coefficients
c_nlm (n_in radial channels) -> adaptive radius R_ad as the radius
enclosing the fixed moment M* = xi*^3 / (18 pi^2) (clamped at R_c) ->
mean-split linear transfer
c' = rho_bar_W A^(R_ad) delta_l0 + M^(l)(R_ad) (c - rho_bar_W A^(W) delta_l0)
-> d = c' / (A_0 rho_bar_safe). The mean split makes the HEG limit exact by
construction (in exact arithmetic); an l-dependent k_F^l factor is NOT
applied (it breaks scale invariance for l > 0; see the
non-dimensionalization section of the document).

The condition R k_F(rho_bar(R)) = xi* that fixes R_ad reduces to a fixed
enclosed moment: with M(R) = int_0^R r^2 rho_00 dr = rho_bar(R) R^3/3, one
has [R k_F]^3 = 18 pi^2 M(R), so R_ad solves M(R_ad) = xi*^3/(18 pi^2). M
is monotone, so this is a direct inversion of the cumulative radial charge
(``adaptive_radius`` / ``invert_enclosed_moment``) -- no iterative root
finder.

Two evaluation paths share the same algebra (``simple_from_window``):

  * radial / profile path: the environment is given as multipole profiles
    rho_l(r) (``simple_descriptors``), integrated either with Gauss-Legendre
    quadrature (continuum reference) or a finite-difference-like uniform
    radial grid with partial-cell weights and a one-term Euler-Maclaurin
    boundary correction (``project_window``, ``ball_average``);
  * 3D Cartesian path: the environment is a per-spin density sampled on a
    lattice (``LatticeStencil`` + ``grid_descriptors``), with kernel sums
    for the window coefficients and partial-cell ramp shell sums for the
    ball average.

Transfer matrices M^(l): closed form for every l via the Sturm-Liouville
cross-product identity; identity embedding when the adaptive radius is
clamped at the window.

Working parameters (R_C, XI_TARGET, N_OUT, RHO_MIN, L_MAX) are pinned in
``params.py`` to the values validated in the document.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

from .bessel import (
    RadialBesselBasis,
    a_n_closed_form,
    radial_gauss_grid,
    spherical_jn_zeros,
)
from .params import L_MAX, N_OUT, R_C, RHO_MIN, XI_TARGET
from .rotations import real_sph_harm

_BASIS_CACHE = {}

# Enclosed-moment target: R_ad is the radius with M(R_ad) = M_TARGET, where
# M(R) = int_0^R r^2 rho_00 dr, equivalent to the fixed dimensionless cutoff
# R k_F = xi* since [R k_F]^3 = 18 pi^2 M(R).
M_TARGET = XI_TARGET**3 / (18.0 * np.pi**2)

# Default radii at which the cumulative moment is sampled to locate R_ad when
# a path supplies no native shell structure (continuum reference paths, whose
# rho_bar is a cheap quadrature). Fine enough that the closed-form crossing
# matches the exact radius to <1e-5; exact for any uniform density (HEG)
# regardless of sampling.
_DEFAULT_RADIUS_SAMPLES = np.linspace(R_C / 1024.0, R_C, 1024)


def window_basis(l, n_in):
    """Cached fixed-window radial basis with n_in channels for channel l."""
    key = (l, n_in)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = RadialBesselBasis(n_in - 1, l, R_C)
    return _BASIS_CACHE[key]


def n_in_for(h):
    """Grid-matched channel count, n_in(h) = 2 ceil(R_c / 4h).

    Stays at or above the kernel-resolution bound n_in ~ R_c / 2h and
    reproduces the document's pairing n_in = 16 at h = 0.2 bohr."""
    return max(N_OUT, 2 * int(np.ceil(R_C / (4.0 * h))))


# =============================================================================
# Transfer matrices (closed form, every l)
# =============================================================================
def transfer_matrix(l, r_ad, n_out=N_OUT, n_in=32):
    """M_mn = int_0^{r_ad} r^2 R_m^(r_ad,l) R_n^(R_c,l) dr.

    Sturm-Liouville cross-product identity with j_l(z_m) = 0:
        int_0^R r^2 j_l(ar) j_l(br) dr
          = R^2 a j_l'(aR) j_l(bR) / (b^2 - a^2),  a = z_m/r_ad, b = z_n/R_c.
    Identity embedding when the adaptive radius is clamped at the window.
    """
    if abs(r_ad - R_C) < 1e-12 * R_C:
        return np.eye(n_out, n_in)
    z_out = spherical_jn_zeros(l, n_out)
    z_in = spherical_jn_zeros(l, n_in)
    a = z_out / r_ad
    b = z_in / R_C
    norm_out = 1.0 / np.sqrt(0.5 * r_ad**3 * spherical_jn(l + 1, z_out) ** 2)
    norm_in = 1.0 / np.sqrt(0.5 * R_C**3 * spherical_jn(l + 1, z_in) ** 2)
    A, B = a[:, None], b[None, :]
    denom = B**2 - A**2
    degenerate = np.abs(denom) < 1e-10 * A**2
    denom = np.where(degenerate, 1.0, denom)
    matrix = (
        r_ad**2
        * A
        * spherical_jn(l, z_out, derivative=True)[:, None]
        * spherical_jn(l, B * r_ad)
        / denom
    )
    if np.any(degenerate):  # rare near-coincident wavevectors: quadrature
        basis_out = RadialBesselBasis(n_out - 1, l, r_ad)
        basis_in = window_basis(l, n_in)
        quad = radial_gauss_grid(r_ad, 512)
        fallback = np.einsum(
            "mj,nj,j->mn",
            basis_out.evaluate(l, quad.nodes),
            basis_in.evaluate(l, quad.nodes),
            quad.weights * quad.nodes**2,
        ) / (norm_out[:, None] * norm_in[None, :])
        matrix = np.where(degenerate, fallback, matrix)
    return norm_out[:, None] * norm_in[None, :] * matrix


# =============================================================================
# Adaptive radius: fixed enclosed moment, by direct inversion (no root finder)
# =============================================================================
def invert_enclosed_moment(r_samples, m_cumulative):
    """Adaptive radius R_ad from the monotone cumulative moment.

    ``r_samples`` is increasing and ends at R_c; ``m_cumulative`` is
    M(r) = int_0^r r'^2 rho_00 dr' at those radii. R_ad solves
    M(R_ad) = M* = xi*^3/(18 pi^2): the interval bracketing M* is inverted
    in closed form by taking the density constant across it
    (dM = rho r^2 dr, so M(R) = M_lo + rho (R^3 - r_lo^3)/3, a single cube
    root). When the whole window encloses less than M* the radius clamps at
    R_c. Returns ``(r_ad, clamped)``."""
    r = np.asarray(r_samples, dtype=float)
    m = np.asarray(m_cumulative, dtype=float)
    if m[-1] <= M_TARGET:  # window holds less than the target moment
        return R_C, True
    r = np.concatenate(([0.0], r))
    m = np.concatenate(([0.0], m))
    k = int(np.searchsorted(m, M_TARGET))  # m[k-1] < M* <= m[k]
    r_lo, r_hi, m_lo, m_hi = r[k - 1], r[k], m[k - 1], m[k]
    rho_cell = (m_hi - m_lo) / ((r_hi**3 - r_lo**3) / 3.0)
    r_ad = (r_lo**3 + 3.0 * (M_TARGET - m_lo) / rho_cell) ** (1.0 / 3.0)
    return float(r_ad), False


def adaptive_radius(rho_bar, radius_samples=None):
    """Adaptive radius R_ad, the radius enclosing the fixed moment
    M* = xi*^3/(18 pi^2), with M(R) = int_0^R r^2 rho_00 dr = rho_bar(R) R^3/3.

    M is monotone in R, so R_ad is obtained by inverting it directly rather
    than by solving R k_F(rho_bar(R)) = xi* iteratively
    (``invert_enclosed_moment``). The ball-averaged density rho_bar(R) is
    sampled at ``radius_samples`` -- a path's native shell radii, where the
    cube-root inverse is exact for the cell-constant model, or a default
    fine grid. Returns ``(r_ad, clamped)``."""
    r = _DEFAULT_RADIUS_SAMPLES if radius_samples is None else np.asarray(
        radius_samples, dtype=float
    )
    m = np.array([max(rho_bar(ri), 0.0) * ri**3 / 3.0 for ri in r])
    return invert_enclosed_moment(r, m)


# =============================================================================
# Shared pipeline algebra (window coefficients + ball average -> descriptors)
# =============================================================================
def simple_from_window(c_window, rho_bar, n_in, l_max=L_MAX, n_out=N_OUT,
                       radius_samples=None, radius=None):
    """SIMPLE descriptors from window coefficients c_window[l] (n_in, 2l+1)
    and a ball-average callable rho_bar(R). Identical algebra for the
    radial and 3D Cartesian paths; only the two inputs differ.

    A path that has already located the radius (e.g. from a vectorized
    cumulative moment) may pass it as ``radius=(r_ad, clamped)``; otherwise
    it is found from ``rho_bar`` sampled at ``radius_samples``.

    Returns a dict with per-l descriptor arrays (n_out, 2l+1) under integer
    keys plus the scalars ``r_ad``, ``clamped``, ``rho_bar_safe``."""

    r_ad, clamped = radius if radius is not None else adaptive_radius(
        rho_bar, radius_samples
    )
    rho_bar_window = rho_bar(R_C)
    rho_bar_safe = np.sqrt(rho_bar(r_ad) ** 2 + RHO_MIN**2)
    a_0 = a_n_closed_form(0, r_ad)[0]

    result = {"r_ad": r_ad, "clamped": clamped, "rho_bar_safe": rho_bar_safe}
    for l in range(l_max + 1):
        if l == 0:
            c_fluct = c_window[0].copy()
            c_fluct[:, 0] -= rho_bar_window * a_n_closed_form(n_in - 1, R_C)
            c_ad = np.zeros((n_out, 1))
            c_ad[:, 0] = rho_bar_window * a_n_closed_form(n_out - 1, r_ad)
        else:
            c_fluct = c_window[l]
            c_ad = np.zeros((n_out, 2 * l + 1))
        c_ad = c_ad + transfer_matrix(l, r_ad, n_out, n_in) @ c_fluct
        result[l] = c_ad / (a_0 * rho_bar_safe)
    return result


# =============================================================================
# Radial / profile path (axially symmetric environments, m = 0 only)
# =============================================================================
def grid_nodes_weights(h):
    """Cell-centered radial grid with a partial cell at the sphere boundary.

    Returns (nodes, widths, starts); the quadrature weight of each cell is
    its width, and sub-radius shell sums clip against the cell extents."""
    n_full = int(np.floor(R_C / h))
    delta = R_C - n_full * h
    nodes = np.append((np.arange(n_full) + 0.5) * h, n_full * h + 0.5 * delta)
    widths = np.append(np.full(n_full, h), delta)
    starts = np.append(np.arange(n_full) * h, n_full * h)
    return nodes, widths, starts


def project_window(profile_l, l, n_in, h=None):
    """c_nl0 = sqrt(4pi/(2l+1)) int_0^{R_c} r^2 R_nl rho_l dr.

    h = None: Gauss-Legendre (continuum reference). Otherwise the grid rule:
    partial-cell midpoint sums + Euler-Maclaurin boundary term."""
    basis = window_basis(l, n_in)
    angular = np.sqrt(4.0 * np.pi / (2 * l + 1))
    if h is None:
        quad = radial_gauss_grid(R_C, 512)
        values = basis.evaluate(l, quad.nodes)
        return angular * np.einsum(
            "nj,j->n", values, quad.weights * quad.nodes**2 * profile_l(quad.nodes)
        )
    nodes, widths, _ = grid_nodes_weights(h)
    values = basis.evaluate(l, nodes)
    c = angular * np.einsum("nj,j->n", values, widths * nodes**2 * profile_l(nodes))
    # f(r) = r^2 R_nl rho_l vanishes at R_c (Dirichlet); f'(R_c) does not.
    boundary_slope = basis.derivative(l, np.array([R_C]))[:, 0]
    c = c + angular * h**2 / 24.0 * R_C**2 * boundary_slope * float(
        profile_l(np.array([R_C]))[0]
    )
    return c


def ball_average(profile_0, radius, h=None):
    """rho_bar(R) = (3/R^3) int_0^R r^2 rho_0 dr.

    Grid version: cell-constant density (cell-center values) integrated
    against the exact shell-volume moment of each cell's overlap with [0, R]
    -- the partial-cell-volume weighting used by real-space codes. Exact for
    uniform densities at any resolution."""
    if h is None:
        quad = radial_gauss_grid(radius, 192)
        return (
            3.0
            / radius**3
            * float(np.sum(quad.weights * quad.nodes**2 * profile_0(quad.nodes)))
        )
    nodes, widths, starts = grid_nodes_weights(h)
    r_lo = np.minimum(starts, radius)
    r_hi = np.minimum(starts + widths, radius)
    moments = (r_hi**3 - r_lo**3) / 3.0
    return 3.0 / radius**3 * float(np.sum(moments * profile_0(nodes)))


def simple_descriptors(profiles, n_in, h=None, l_max=L_MAX, n_out=N_OUT):
    """SIMPLE descriptors of an axially symmetric environment.

    ``profiles(r, l)`` returns the multipole profile rho_l(r); the result
    dict stores per-l m = 0 coefficient vectors (length n_out) under
    integer keys plus the pipeline scalars."""

    def rho_bar(radius):
        return ball_average(lambda r: profiles(r, 0), radius, h=h)

    if h is None:
        radius_samples = None  # continuum: default fine sampling
    else:  # grid: the shell right-edges, on which the cube-root inverse of
        _, widths, starts = grid_nodes_weights(h)  # the cell-constant model
        radius_samples = starts + widths            # is exact
    c_window = {
        l: project_window(lambda r, ll=l: profiles(r, ll), l, n_in, h=h)[:, None]
        for l in range(l_max + 1)
    }
    result = simple_from_window(c_window, rho_bar, n_in, l_max, n_out,
                                radius_samples=radius_samples)
    for l in range(l_max + 1):
        result[l] = result[l][:, 0]
    return result


def fixed_window_descriptors(profiles, n_out=N_OUT, l_max=L_MAX, h=None):
    """Baseline without the scale transform: raw fixed-window descriptors."""
    rho_bar_safe = np.sqrt(
        ball_average(lambda r: profiles(r, 0), R_C, h=h) ** 2 + RHO_MIN**2
    )
    a_0 = a_n_closed_form(0, R_C)[0]
    return {
        l: project_window(lambda r, ll=l: profiles(r, ll), l, n_out, h=h)
        / (a_0 * rho_bar_safe)
        for l in range(l_max + 1)
    }


# =============================================================================
# 3D Cartesian path
# =============================================================================
class LatticeStencil:
    """All lattice points within the window of an evaluation center.

    The lattice is x = (i + f) h with integer i and fixed fractional
    registration f in [0, 1)^3; the evaluation center is the origin of the
    relative coordinates. Everything geometric -- the kernel factors (radial
    basis values and real harmonics) and the adaptive-radius weight matrix --
    is independent of the density and is cached when the stencil is small
    enough, so a single stencil is reused for every density, orientation,
    scale, and self-consistent iteration at a given (h, n_in, registration).
    On a uniform grid the window stencil is the same at every evaluation
    point, so in production this geometry is built once for the whole grid."""

    CACHE_LIMIT = 4 * 10**5  # points; ~60 MB of cached factors at n_in = 16

    def __init__(self, h, n_in, frac=(0.0, 0.0, 0.0), l_max=L_MAX):
        self.h, self.n_in, self.l_max = float(h), int(n_in), int(l_max)
        reach = int(np.ceil((R_C + h) / h)) + 1
        idx = np.arange(-reach, reach + 1)
        ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
        pts = (np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
               + np.asarray(frac, dtype=float)) * h
        dist = np.linalg.norm(pts, axis=1)
        keep = dist <= R_C + 0.5 * h  # ramp support of the shell sums
        self.points = pts[keep]
        self.dist = dist[keep]
        units = self.points.copy()
        safe = np.maximum(self.dist, 1e-300)
        units /= safe[:, None]
        units[self.dist < 1e-14] = (0.0, 0.0, 1.0)  # R_nl(0) = 0 for l > 0
        self.units = units
        self._inside = self.dist <= R_C
        self._cache = {} if len(self.points) <= self.CACHE_LIMIT else None
        self._radius_geom = None

    def _factors(self, l):
        if self._cache is not None and l in self._cache:
            return self._cache[l]
        radial = window_basis(l, self.n_in).evaluate(l, self.dist)
        radial[:, ~self._inside] = 0.0
        harmonics = np.stack(
            [real_sph_harm(l, m, self.units) for m in range(-l, l + 1)]
        )
        if self._cache is not None:
            self._cache[l] = (radial, harmonics)
        return radial, harmonics

    def window_coefficients(self, rho):
        c_window = {}
        for l in range(self.l_max + 1):
            radial, harmonics = self._factors(l)
            c_window[l] = (radial * rho[None, :]) @ harmonics.T * self.h**3
        return c_window

    def shell_weights(self, radius):
        """Partial-cell ramp weights; exact for uniform densities after
        normalization by their own sum (preserves the exact HEG limit of
        the ball average)."""
        return np.clip((radius - self.dist) / self.h + 0.5, 0.0, 1.0)

    def _radius_geometry(self):
        """Precomputed, density-independent geometry of the adaptive-radius
        inversion: the trial cutoff radii R_j (step h/4), the partial-cell
        weight matrix W[j,i] = clip((R_j - d_i)/h + 1/2, 0, 1), and its row
        sums Sigma_i W[j,i]. These depend only on the lattice, so they are
        built once and cached on the stencil (W densely only when the stencil
        is small enough to hold it; otherwise it is applied chunked on the
        fly). Returns (R_j, W or None, row_sums)."""
        if self._radius_geom is None:
            n_grid = max(64, int(np.ceil(4.0 * R_C / self.h)))
            r_grid = np.linspace(R_C / n_grid, R_C, n_grid)
            if self._cache is not None:  # small enough to store W densely
                weights = np.clip(
                    (r_grid[:, None] - self.dist[None, :]) / self.h + 0.5,
                    0.0, 1.0,
                )
                self._radius_geom = (r_grid, weights, weights.sum(axis=1))
            else:  # large stencil: keep only the tiny row sums, apply chunked
                row_sums = np.zeros(n_grid)
                for s in range(0, self.dist.size, 16384):
                    w = np.clip(
                        (r_grid[:, None] - self.dist[None, s:s + 16384]) / self.h
                        + 0.5, 0.0, 1.0,
                    )
                    row_sums += w.sum(axis=1)
                self._radius_geom = (r_grid, None, row_sums)
        return self._radius_geom

    def enclosed_moment_curve(self, rho):
        """Cumulative enclosed moment M(R_j) = rho_bar(R_j) R_j^3/3 at the
        precomputed trial radii. The geometry (weights, row sums, radii) is
        cached; the only density-dependent work is the contraction W @ rho
        (a fixed-stencil convolution). Returns (R_j, M(R_j))."""
        r_grid, weights, row_sums = self._radius_geometry()
        if weights is not None:
            w_rho = weights @ rho
        else:
            w_rho = np.zeros(r_grid.size)
            for s in range(0, self.dist.size, 16384):
                w = np.clip(
                    (r_grid[:, None] - self.dist[None, s:s + 16384]) / self.h
                    + 0.5, 0.0, 1.0,
                )
                w_rho += w @ rho[s:s + 16384]
        nearest = int(np.argmin(self.dist))
        rho_bar = np.where(row_sums > 0.0, w_rho / np.maximum(row_sums, 1e-300),
                           rho[nearest])
        return r_grid, rho_bar * r_grid**3 / 3.0


def grid_descriptors(density, stencil, l_max=L_MAX, n_out=N_OUT):
    """SIMPLE descriptors of a gridded environment.

    ``density`` is a callable mapping an (N, 3) array of coordinates
    relative to the evaluation center to the per-spin density at those
    points."""
    rho = density(stencil.points)
    c_window = stencil.window_coefficients(rho)
    nearest = int(np.argmin(stencil.dist))

    def rho_bar(radius):
        w = stencil.shell_weights(radius)
        w_sum = np.sum(w)
        if w_sum == 0.0:  # radius below the nearest sample (off-lattice
            return float(rho[nearest])  # center): nearest-cell value
        return float(np.sum(w * rho) / w_sum)

    # Locate R_ad by inverting the cumulative enclosed moment. The
    # partial-cell ball average is read off at the precomputed trial radii
    # (step h/4) via the stencil's cached weight matrix -- pure geometry,
    # built once and reused for every density and self-consistent iteration
    # -- so the only density-dependent work here is the contraction W @ rho
    # (Sec. 3.2). The crossing M = M* is then closed-form
    # (invert_enclosed_moment). This is O(N) per evaluation point, the same
    # order as the window kernel sums and sub-dominant to them, with no
    # iterative solve. The partial-cell weighting keeps R_ad a smooth
    # functional of the density; the residual O(h^2) radius error is part of
    # the grid discretization error (Sec. 9.1).
    radius = invert_enclosed_moment(*stencil.enclosed_moment_curve(rho))
    return simple_from_window(c_window, rho_bar, stencil.n_in, l_max, n_out,
                              radius=radius)
