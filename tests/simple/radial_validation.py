#!/usr/bin/env python3
"""Radially resolved validation of the SIMPLE descriptors.

Generates the radially resolved figures of the "Numerical evaluation I"
section of docs/SIMPLE/SIMPLE.tex, each validating a key property of the
expansion numerically, up to l = 3 and including a finite-difference-like
uniform radial grid with spacing h = 0.2 bohr:

    simple_heg.pdf           - homogeneous electron gas limit
                               d_n00 = (-1)^n/(n+1)
    simple_vacuum.pdf        - stability and decay laws as rho -> 0
                               (all l <= 3)
    simple_scale.pdf         - scale invariance and its two breakdown
                               mechanisms (window clamp at low density,
                               channel/grid resolution at high contraction)
    simple_orthogonality.pdf - orthogonality of the radial channels: the
                               discrete Gram matrix and its condition number
                               vs channel count, showing each channel adds
                               independent information up to the resolution
                               bound n_in ~ R_c / 2h

The SIMPLE pipeline itself lives in atom/descriptors/simple/ (this script
only provides the test environments and figures). Test environment for
l > 0: central + off-center atom (b = 1.5 bohr), each a hydrogen-like
exponential density e^{-2r}/(2 pi) per spin; multipole profiles rho_l(r)
by Legendre quadrature. Under uniform scaling about the evaluation point,
rho_l -> lambda^3 rho_l(lambda r).

Run from the repository root:

    python3 tests/simple/radial_validation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atom.descriptors.simple import (  # noqa: E402
    L_MAX,
    N_OUT,
    R_C,
    ball_average,
    fixed_window_descriptors,
    grid_nodes_weights,
    radial_gauss_grid,
    simple_descriptors,
    window_basis,
)

N_IN_EXACT = 32  # continuum-quadrature channel count
N_IN_GRID = 16  # grid-matched channel count for h = 0.2 (n_in ~ R_c / 2h)
H_GRID = 0.2
RHO_MIN = 1e-10
FIG_DIR = _REPO_ROOT / "docs" / "SIMPLE" / "figures"


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
    axes[0].set(xlabel=r"$r_s$ (bohr)", ylabel=r"$d_{n00}$",
                title="(a) HEG values on the $h=0.2$ bohr grid")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].loglog(r_s_values, err_exact, "s-", color="k",
                   label=f"continuum quadrature ($n_{{in}}={N_IN_EXACT}$)")
    axes[1].loglog(r_s_values, err_grid, "o-", color="tab:blue",
                   label=f"$h=0.2$ bohr grid ($n_{{in}}={N_IN_GRID}$)")
    axes[1].set(xlabel=r"$r_s$ (bohr)",
                ylabel=r"$\max_n |d_{n00} - (-1)^n/(n{+}1)|$",
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
    axes[0].set(ylabel=r"$\Vert d_{\ell}\Vert$",
                title=r"(a) coefficient decay, $\mathcal{O}(\epsilon)$ guide dotted")
    axes[1].set(ylabel=r"$\sum_n d_{n\ell 0}^2$",
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


# --- orthogonality / non-redundancy of the radial channels --------------------
N_IN_ORTHO = (8, 12, 16, 20, 24, 32, 40, 48, 56, 64)


def radial_gram(l, n_in, h=None):
    """Discrete Gram matrix G_{nn'} = <R_nl, R_n'l> of the window basis under
    the inner product the projection actually uses: Gauss-Legendre for the
    continuum (h = None), the cell-width radial quadrature for the grid. The
    continuum Gram equals the identity by construction (orthonormality); its
    condition number measures how many channels remain linearly independent
    once the kernels are sampled."""
    basis = window_basis(l, n_in)
    if h is None:
        quad = radial_gauss_grid(R_C, 1024)
        nodes, weights = quad.nodes, quad.weights
    else:
        nodes, widths, _ = grid_nodes_weights(h)
        weights = widths
    values = basis.evaluate(l, nodes)  # (n_in, n_nodes)
    return np.einsum("ni,mi,i->nm", values, values, weights * nodes**2)


def figure_orthogonality():
    n_heat = 24
    gram_heat = radial_gram(0, n_heat, h=H_GRID)
    dev_identity = np.abs(gram_heat - np.eye(n_heat))

    resolutions = (None, 0.4, 0.2, 0.1)
    colors = {None: "k", 0.4: "tab:red", 0.2: "tab:orange", 0.1: "tab:blue"}
    cond = {
        h: [np.linalg.cond(radial_gram(0, n, h=h)) for n in N_IN_ORTHO]
        for h in resolutions
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    im = axes[0].imshow(
        np.log10(dev_identity + 1e-16), origin="lower", cmap="viridis",
        vmin=-12, vmax=0, aspect="equal",
    )
    axes[0].set(xlabel=r"$n'$", ylabel=r"$n$",
                title=f"(a) $\\log_{{10}}|G^{{(0)}}_{{nn'}}-\\delta_{{nn'}}|$ "
                      f"($h=0.2$, $n_{{in}}={n_heat}$)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for h in resolutions:
        label = "continuum" if h is None else rf"$h={h}$ grid"
        axes[1].semilogy(N_IN_ORTHO, cond[h], "o-", ms=4, color=colors[h],
                         label=label)
        if h is not None:
            axes[1].axvline(R_C / h, color=colors[h], ls=":", lw=1.0)
    axes[1].set(xlabel=r"$n_{in}$", ylabel=r"$\kappa(G^{(0)})$",
                title=r"(b) channel conditioning; dotted: $n_{in}=R_c/h$")
    axes[1].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_orthogonality.pdf")
    plt.close(fig)

    cont = np.abs(radial_gram(0, N_IN_EXACT, h=None) - np.eye(N_IN_EXACT)).max()
    g16 = radial_gram(0, N_IN_GRID, h=H_GRID)
    print(f"orthogonality: continuum Gram dev (n_in={N_IN_EXACT}) {cont:.1e}; "
          f"grid h=0.2 n_in=16 max|G-I| {np.abs(g16 - np.eye(N_IN_GRID)).max():.1e}, "
          f"kappa {np.linalg.cond(g16):.2f}; "
          f"grid h=0.2 kappa at n_in=24 "
          f"{np.linalg.cond(radial_gram(0, 24, h=H_GRID)):.0f}, n_in=32 "
          f"{np.linalg.cond(radial_gram(0, 32, h=H_GRID)):.0e}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_heg()
    figure_vacuum()
    figure_scale()
    figure_orthogonality()
    print(f"Wrote simple_heg.pdf, simple_vacuum.pdf, simple_scale.pdf, "
          f"simple_orthogonality.pdf to {FIG_DIR}/")
