#!/usr/bin/env python3
"""Validation figures for SIMPLE (scale-invariant multipole local expansion).

Generates the three figures of SIMPLE.tex, each validating a key property of
the expansion numerically, up to l = 3 and including a finite-difference-like
uniform radial grid with spacing h = 0.2 bohr:

    simple_heg.pdf     - homogeneous electron gas limit d_n00 = (-1)^n/(n+1)
    simple_vacuum.pdf  - stability and decay laws as rho -> 0 (all l <= 3)
    simple_scale.pdf   - scale invariance and its two breakdown mechanisms
                         (window clamp at low density, channel/grid
                         resolution at high contraction)

Implementation notes (matching SIMPLE.tex):
  * SIMPLE pipeline: fixed-window coefficients c_nlm (n_in channels) ->
    adaptive radius R_ad from R k_F(rho_bar(R)) = xi* (clamped at R_c) ->
    mean-split linear transfer c' = rho_bar_W A^(R_ad) delta_l0 +
    M^(l)(R_ad) (c - rho_bar_W A^(W) delta_l0) -> d = c'/(A_0 rho_bar_safe).
    The mean split makes the HEG limit exact by construction; the k_F^l
    factor of NOLE Eq. (d_def) is NOT applied (it breaks scale invariance
    for l > 0).
  * Transfer matrices M^(l): closed form for every l via the Sturm-Liouville
    cross-product identity; identity embedding when clamped.
  * Grid quadrature ("FD" curves): cell-centered midpoint sums with
    partial-cell weights at the sphere boundary plus a one-term
    Euler-Maclaurin correction h^2/24 R_c^2 R'_nl(R_c) rho_l(R_c)
    (the integrand itself vanishes at R_c by the Dirichlet condition).
  * Test environment for l > 0: central + off-center atom (b = 1.5 bohr),
    multipole profiles rho_l(r) by Legendre quadrature; under uniform scaling
    about the evaluation point, rho_l -> lambda^3 rho_l(lambda r).

Requires the atom-dft-mlxc package (located automatically next to this
repository, or set ATOM_DFT_MLXC). Run from the repository root:

    python3 scripts/simple_validation_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import eval_legendre, spherical_jn

_DEFAULT = Path(__file__).resolve().parent.parent.parent
_PKG_ROOT = Path(os.environ.get("ATOM_DFT_MLXC", _DEFAULT))
if not (_PKG_ROOT / "atom").is_dir():
    sys.exit(f"atom-dft-mlxc not found at {_PKG_ROOT}; set ATOM_DFT_MLXC.")
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from atom.descriptors.simple import (  # noqa: E402
    RadialBesselBasis,
    a_n_closed_form,
    radial_gauss_grid,
    spherical_jn_zeros,
)

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
R_C = 3.0 * BOHR_PER_ANGSTROM
XI_TARGET = 2.0
N_OUT = 8
N_IN_EXACT = 32  # continuum-quadrature channel count
N_IN_GRID = 16  # grid-matched channel count for h = 0.2 (n_in ~ R_c / 2h)
H_GRID = 0.2
L_MAX = 3
RHO_MIN = 1e-10
SQRT_4PI = np.sqrt(4.0 * np.pi)
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

_BASIS_CACHE = {}


def window_basis(l, n_in):
    key = (l, n_in)
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = RadialBesselBasis(n_in - 1, l, R_C)
    return _BASIS_CACHE[key]


# --- transfer matrices (closed form, every l) --------------------------------
def transfer_matrix(l, r_ad, n_out=N_OUT, n_in=N_IN_EXACT):
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


# --- quadratures -------------------------------------------------------------
def grid_nodes_weights(h):
    """Cell-centered grid with a partial cell at the sphere boundary.

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
    — the partial-cell-volume weighting used by real-space codes. Exact for
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


# --- SIMPLE pipeline ----------------------------------------------------------
def simple_descriptors(profiles, n_in, h=None, l_max=L_MAX, n_out=N_OUT):
    """profiles(r, l) -> rho_l(r). Returns per-l d vectors + scalars."""

    def rho_bar(radius):
        return ball_average(lambda r: profiles(r, 0), radius, h=h)

    def g(radius):
        rb_safe = np.sqrt(max(rho_bar(radius), 0.0) ** 2 + RHO_MIN**2)
        return radius * (6.0 * np.pi**2 * rb_safe) ** (1.0 / 3.0)

    clamped = g(R_C) <= XI_TARGET
    r_ad = R_C if clamped else brentq(
        lambda radius: g(radius) - XI_TARGET, 1e-2, R_C, rtol=1e-13
    )
    rho_bar_window = rho_bar(R_C)
    rho_bar_safe = np.sqrt(rho_bar(r_ad) ** 2 + RHO_MIN**2)
    a_0 = a_n_closed_form(0, r_ad)[0]

    result = {
        "r_ad": r_ad,
        "clamped": clamped,
        "xi_loc": g(R_C) if clamped else XI_TARGET,
        "rho_bar_safe": rho_bar_safe,
    }
    for l in range(l_max + 1):
        c_window = project_window(lambda r, ll=l: profiles(r, ll), l, n_in, h=h)
        if l == 0:  # mean split: window average maps analytically
            c_fluct = c_window - rho_bar_window * a_n_closed_form(n_in - 1, R_C)
            c_ad = rho_bar_window * a_n_closed_form(n_out - 1, r_ad)
        else:
            c_fluct = c_window
            c_ad = np.zeros(n_out)
        c_ad = c_ad + transfer_matrix(l, r_ad, n_out, n_in) @ c_fluct
        result[l] = c_ad / (a_0 * rho_bar_safe)
    return result


def fixed_window_descriptors(profiles, n_out=N_OUT, l_max=L_MAX, h=None):
    """Baseline without the scale transform: NOLE at the fixed cutoff."""
    rho_bar_safe = np.sqrt(
        ball_average(lambda r: profiles(r, 0), R_C, h=h) ** 2 + RHO_MIN**2
    )
    a_0 = a_n_closed_form(0, R_C)[0]
    return {
        l: project_window(lambda r, ll=l: profiles(r, ll), l, n_out, h=h)
        / (a_0 * rho_bar_safe)
        for l in range(l_max + 1)
    }


# --- test densities -----------------------------------------------------------
B_OFF = 1.5
U_NODES, U_WEIGHTS = leggauss(96)


def atom_density(x):
    return 0.5 / np.pi * np.exp(-2.0 * np.asarray(x, dtype=float))


def environment_profiles(lam=1.0, eps=1.0):
    """Central + off-center atom; lambda scales coordinates about the
    evaluation point, eps scales the amplitude only (vacuum sweeps)."""

    def profiles(r, l):
        r_scaled = np.atleast_1d(np.asarray(r, dtype=float)) * lam
        dist = np.sqrt(
            r_scaled[:, None] ** 2
            + B_OFF**2
            - 2.0 * r_scaled[:, None] * B_OFF * U_NODES[None, :]
        )
        profile = (
            (2 * l + 1)
            / 2.0
            * np.sum(U_WEIGHTS * eval_legendre(l, U_NODES) * atom_density(dist), axis=1)
        )
        if l == 0:
            profile = profile + atom_density(r_scaled)
        return eps * lam**3 * profile

    return profiles


def heg_profiles(rho0):
    def profiles(r, l):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return np.full_like(r, rho0) if l == 0 else np.zeros_like(r)

    return profiles


def deviation(value, reference):
    return np.linalg.norm(value - reference) / np.linalg.norm(reference)


# --- figures -------------------------------------------------------------------
def figure_heg():
    r_s_values = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    reference = (-1.0) ** np.arange(N_OUT) / (np.arange(N_OUT) + 1)
    d_grid, err_grid, err_exact, clamped = [], [], [], []
    for r_s in r_s_values:
        rho0 = 3.0 / (4.0 * np.pi * r_s**3) / 2.0  # per spin
        res_g = simple_descriptors(heg_profiles(rho0), N_IN_GRID, h=H_GRID, l_max=0)
        res_e = simple_descriptors(heg_profiles(rho0), N_IN_EXACT, l_max=0)
        d_grid.append(res_g[0])
        err_grid.append(np.max(np.abs(res_g[0] - reference)))
        err_exact.append(np.max(np.abs(res_e[0] - reference)))
        clamped.append(res_g["clamped"])
    d_grid = np.array(d_grid)
    clamp_start = r_s_values[np.argmax(clamped)] if any(clamped) else None

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, N_OUT))
    for n in range(N_OUT):
        axes[0].axhline(reference[n], color=colors[n], lw=0.8, ls="--")
        axes[0].semilogx(r_s_values, d_grid[:, n], "o", ms=4, color=colors[n],
                         label=f"$n={n}$" if n < 4 else None)
    if clamp_start is not None:
        axes[0].axvspan(clamp_start, r_s_values[-1], color="0.92")
        axes[0].text(clamp_start * 1.4, 0.75, "clamped\n($R_{ad}=R_c$)", fontsize=8)
    axes[0].set(xlabel=r"$r_s$ (bohr)", ylabel=r"$\varrho_{n00}$",
                title="(a) HEG values on the $h=0.2$ bohr grid")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].loglog(r_s_values, err_exact, "s-", color="k",
                   label=f"continuum quadrature ($n_{{in}}={N_IN_EXACT}$)")
    axes[1].loglog(r_s_values, err_grid, "o-", color="tab:blue",
                   label=f"$h=0.2$ bohr grid ($n_{{in}}={N_IN_GRID}$)")
    axes[1].set(xlabel=r"$r_s$ (bohr)",
                ylabel=r"$\max_n |\varrho_{n00} - (-1)^n/(n{+}1)|$",
                title="(b) deviation from the analytic HEG values")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_heg.pdf")
    plt.close(fig)
    print(f"HEG: max error continuum {max(err_exact):.1e}, grid {max(err_grid):.1e}")


def figure_vacuum():
    epsilons = np.logspace(-13.0, 0.0, 27)
    norms = {l: [] for l in range(L_MAX + 1)}
    powers = {l: [] for l in range(L_MAX + 1)}
    for eps in epsilons:
        res = simple_descriptors(
            environment_profiles(eps=eps), N_IN_GRID, h=H_GRID
        )
        for l in range(L_MAX + 1):
            norms[l].append(np.linalg.norm(res[l]))
            powers[l].append(np.sum(res[l] ** 2))
    rho_bar_unit = ball_average(
        lambda r: environment_profiles()(r, 0), R_C, h=H_GRID
    )
    eps_star = RHO_MIN / rho_bar_unit

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = plt.cm.plasma(np.linspace(0.0, 0.75, L_MAX + 1))
    for l in range(L_MAX + 1):
        axes[0].loglog(epsilons, norms[l], color=colors[l], label=rf"$\ell={l}$")
        axes[1].loglog(epsilons, powers[l], color=colors[l], label=rf"$\ell={l}$")
    for ax, slope in ((axes[0], 1), (axes[1], 2)):
        ax.axvline(eps_star, color="k", lw=0.8, ls="--")
        guide = epsilons[:8]
        anchor = (norms if slope == 1 else powers)[0][4] / (epsilons[4] ** slope)
        ax.loglog(guide, anchor * guide**slope, "k:", lw=1.0)
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set(ylabel=r"$\Vert \varrho_{\ell}\Vert$",
                title=r"(a) coefficient decay, $\mathcal{O}(\epsilon)$ guide dotted")
    axes[1].set(ylabel=r"$\sum_n \varrho_{n\ell 0}^2$",
                title=r"(b) power spectrum decay, $\mathcal{O}(\epsilon^2)$ guide")
    axes[0].legend(fontsize=8)
    axes[0].text(eps_star * 2.0, 1e-4, r"$\epsilon^*$", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_vacuum.pdf")
    plt.close(fig)
    deep = epsilons <= 1e-10
    slope = np.polyfit(np.log(epsilons[deep]), np.log(np.array(powers[1])[deep]), 1)[0]
    print(f"vacuum: fitted P_1 slope in deep regime = {slope:.3f}")


def figure_scale():
    lambdas = np.exp(np.linspace(np.log(1.0 / 8.0), np.log(8.0), 21))
    mask = np.abs(lambdas - 1.0) > 1e-12

    def sweep(n_in, h):
        refs = simple_descriptors(environment_profiles(), n_in, h=h)
        rows = {l: [] for l in range(L_MAX + 1)}
        for lam in lambdas:
            res = simple_descriptors(environment_profiles(lam=lam), n_in, h=h)
            for l in range(L_MAX + 1):
                rows[l].append(deviation(res[l], refs[l]))
        return {l: np.array(v) for l, v in rows.items()}

    dev_exact = sweep(N_IN_EXACT, None)
    dev_grid = sweep(N_IN_GRID, H_GRID)

    baseline_ref = fixed_window_descriptors(environment_profiles())
    baseline = np.array(
        [
            deviation(
                fixed_window_descriptors(environment_profiles(lam=lam))[0],
                baseline_ref[0],
            )
            for lam in lambdas
        ]
    )

    # clamp onset: R_ad(lambda) = R_ad(1)/lambda reaches R_c
    lam_clamp = simple_descriptors(environment_profiles(), N_IN_EXACT, l_max=0)[
        "r_ad"
    ] / R_C

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = plt.cm.plasma(np.linspace(0.0, 0.75, L_MAX + 1))
    floor = 1e-16
    for l in range(L_MAX + 1):
        axes[0].loglog(lambdas[mask], np.maximum(dev_exact[l][mask], floor),
                       color=colors[l], label=rf"$\ell={l}$")
        axes[0].loglog(lambdas[mask], np.maximum(dev_grid[l][mask], floor),
                       "o", ms=3.5, color=colors[l])
    axes[0].loglog(lambdas[mask], baseline[mask], color="0.55", lw=2.0,
                   ls="--", label="no transform ($\\ell=0$)")
    axes[0].axvspan(lambdas[0], lam_clamp, color="0.92")
    axes[0].text(lambdas[0] * 1.2, 2.0, "clamped", fontsize=8)
    axes[0].set(xlabel=r"$\lambda$", ylabel=r"$D(\lambda)$",
                title=f"(a) lines: continuum, $n_{{in}}={N_IN_EXACT}$;  "
                      f"markers: $h=0.2$, $n_{{in}}={N_IN_GRID}$")
    axes[0].legend(fontsize=8, loc="lower right")

    for n_in, color in ((8, "tab:red"), (16, "tab:orange"),
                        (32, "tab:blue"), (64, "k")):
        refs = simple_descriptors(environment_profiles(), n_in, l_max=0)
        dev = [
            deviation(
                simple_descriptors(environment_profiles(lam=lam), n_in, l_max=0)[0],
                refs[0],
            )
            for lam in lambdas
        ]
        axes[1].loglog(lambdas[mask], np.maximum(np.array(dev)[mask], floor),
                       color=color, label=rf"$n_{{in}}={n_in}$")
    axes[1].axvspan(lambdas[0], lam_clamp, color="0.92")
    axes[1].set(xlabel=r"$\lambda$", ylabel=r"$D(\lambda)$, $\ell=0$",
                title="(b) resolution bound vs channel count (continuum)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_scale.pdf")
    plt.close(fig)
    for l in range(L_MAX + 1):
        i2 = int(np.argmin(np.abs(lambdas - 2.0)))
        print(f"scale l={l}: D(2) exact {dev_exact[l][i2]:.1e}, "
              f"grid {dev_grid[l][i2]:.1e}, baseline(l=0) {baseline[i2]:.1e}")
    print(f"clamp onset lambda = {lam_clamp:.3f}")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    figure_heg()
    figure_vacuum()
    figure_scale()
    print(f"Wrote simple_heg.pdf, simple_vacuum.pdf, simple_scale.pdf to {FIG_DIR}/")
