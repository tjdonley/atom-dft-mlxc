#!/usr/bin/env python3
"""Practical parameter selection for SIMPLE: grid spacing h, cutoff R_c, the
scale-invariance range Lambda, the exposed channel count n_out, and the cost.

Two analyses, both at R_c handled as an explicit argument (no global override needed):

  (A) Energy scale of (R_c, n_out): the Hartree self-energy of the hydrogen 1s density
      (rho = e^{-2r}/pi, exact J = 5/16 Ha) evaluated through the SIMPLE windowed
      Coulomb operator projection [Eq. (coulomb)],
          V_H(r0) = int_{|u|<R_c} rho(r0+u)/u d^3u = sum_n w_n C_n,
      with C_n the monopole window coefficients in an n_out-channel basis and
      w_n = 4pi int_0^{R_c} R_n(u) u du. J converges to the exact value as R_c grows
      (Coulomb-tail truncation) and as n_out grows (radial-basis truncation); we locate
      the 1 mHa contour. This sets R_c = 6 bohr, n_out = 12 (-> 0.43 mHa).

  (B) Scale-invariance range vs the inner-channel count. With n_in = R_c/h grid-resolved
      inner channels, the transfer reconstructs scale-invariant features only up to a
      dilation Lambda set by the resolution bound k* Lambda <~ n_in pi/R_c, i.e.
          Lambda_max ~ n_in / n_out = R_c / (h n_out).
      The number of fixed-stencil convolutions is N_conv = n_in (l_max+1)^2 =
      (R_c/h)(l_max+1)^2, so achieving a scale range Lambda with n_out channels through
      l_max costs N_conv >= n_out Lambda (l_max+1)^2 convolutions (h <= R_c/(n_out Lambda)).

Writes data/parameter_selection.json and figures/hartree_convergence.pdf,
figures/scale_vs_lambda.pdf. Run from the repository root:
    python3 writeup/scripts/parameter_selection.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atom.descriptors.simple.bessel import RadialBesselBasis, radial_gauss_grid  # noqa: E402

_FIG = Path(__file__).resolve().parent.parent / "figures"
_DATA = Path(__file__).resolve().parent / "data"
_OUT = _DATA / "parameter_selection.json"

J_EXACT = 5.0 / 16.0          # H 1s Hartree self-energy (Ha)
L_MAX = 3
N_ANG = (L_MAX + 1) ** 2      # angular kernels per radial channel = 16
RC_STAR, NOUT_STAR = 6.0, 10  # the recommended production setting


# =============================================================================
# (A) Hartree self-energy of H through the SIMPLE windowed Coulomb projection
# =============================================================================
def _h_density(r):
    return np.exp(-2.0 * r) / np.pi


def hartree_simple(R_c, n_out, n_r0=600, n_u=400, n_mu=256, r0_max=18.0):
    """J[rho] for H via the windowed Coulomb operator projection (n_out monopole
    channels on the ball of radius R_c)."""
    xr, wr = leggauss(n_r0)
    r0 = 0.5 * r0_max * (xr + 1.0)
    wr0 = 0.5 * r0_max * wr
    qu = radial_gauss_grid(R_c, n_u)
    u, wu = qu.nodes, qu.weights
    mu, wmu = leggauss(n_mu)
    R_nu = RadialBesselBasis(n_out - 1, 0, R_c).evaluate(0, u)        # (n_out, n_u)
    w_n = 4.0 * np.pi * np.sum(R_nu * (wu * u)[None, :], axis=1)      # 4pi int R_n u du
    V = np.empty(r0.size)
    for i, r0i in enumerate(r0):
        d = np.sqrt(np.maximum(r0i ** 2 + u[:, None] ** 2
                               - 2.0 * r0i * u[:, None] * mu[None, :], 0.0))
        rho0 = 0.5 * np.sum(wmu[None, :] * _h_density(d), axis=1)     # monopole profile
        C = R_nu @ (wu * u ** 2 * rho0)                              # n_out coefficients
        V[i] = float(np.dot(w_n, C))
    return 0.5 * float(np.sum(wr0 * _h_density(r0) * V * 4.0 * np.pi * r0 ** 2))


def run_hartree():
    print(f"(A) Hartree self-energy of H (exact J = 5/16 = {J_EXACT:.5f} Ha)")
    rcs = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    nouts = [2, 4, 6, 8, 10, 12, 16, 20]
    floor = {rc: hartree_simple(rc, 40) for rc in rcs}                # n_out-converged
    grid = {rc: {n: hartree_simple(rc, n) for n in nouts} for rc in [4.0, 6.0, 8.0]}
    print("  R_c tail floor (n_out=40):  "
          + ", ".join(f"R_c={rc:.0f}:{1e3*(floor[rc]-J_EXACT):+.2f}mHa" for rc in rcs))
    print(f"  R_c=6, n_out={NOUT_STAR} -> {1e3*(grid[6.0][NOUT_STAR]-J_EXACT):+.2f} mHa")
    return {"exact": J_EXACT, "rcs": rcs, "nouts": nouts,
            "tail_floor": {str(rc): floor[rc] for rc in rcs},
            "grid": {str(rc): {str(n): grid[rc][n] for n in nouts} for rc in grid}}


def plot_hartree(h):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    nouts = h["nouts"]
    for rc in ("4.0", "6.0", "8.0"):
        err = [abs(h["grid"][rc][str(n)] - J_EXACT) * 1e3 for n in nouts]
        axes[0].semilogy(nouts, err, "o-", ms=4, label=fr"$R_c={float(rc):.0f}$ bohr")
    axes[0].axhline(1.0, color="0.5", ls="--", lw=0.9)
    axes[0].text(nouts[-1], 1.1, "1 mHa", fontsize=7, color="0.4", ha="right")
    axes[0].scatter([NOUT_STAR], [abs(h["grid"]["6.0"][str(NOUT_STAR)] - J_EXACT) * 1e3],
                    s=80, facecolors="none", edgecolors="k", zorder=5)
    axes[0].set(xlabel=r"exposed channels $n_{\rm out}$",
                ylabel=r"$|J_{\rm SIMPLE}-J_{\rm exact}|$ (mHa)",
                title="(a) radial-basis convergence")
    axes[0].legend(fontsize=8)
    rcs = h["rcs"]
    errf = [abs(h["tail_floor"][str(rc)] - J_EXACT) * 1e3 for rc in rcs]
    axes[1].semilogy(rcs, np.maximum(errf, 1e-3), "s-", color="tab:purple")
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.9)
    axes[1].axvline(RC_STAR, color="0.6", ls=":", lw=0.9)
    axes[1].text(RC_STAR + 0.1, 30, r"$R_c=6$", fontsize=8, color="0.4")
    axes[1].set(xlabel=r"window cutoff $R_c$ (bohr)",
                ylabel=r"$|J-J_{\rm exact}|$ (mHa), $n_{\rm out}{=}40$",
                title="(b) Coulomb-tail (window) floor")
    fig.suptitle(r"Hartree energy of H through the SIMPLE Coulomb projection: "
                 r"$R_c=6$, $n_{\rm out}=12 \Rightarrow 0.4$ mHa")
    fig.tight_layout()
    fig.savefig(_FIG / "hartree_convergence.pdf")
    plt.close(fig)


# =============================================================================
# (A') Practical-grid check: the same Hartree energy on a 3D Cartesian grid via
#      FFT convolution of the n_out-folded windowed-Coulomb kernel, vs grid
#      spacing h and sub-grid translation. This is what the 1 mHa continuum limit
#      becomes on a real grid before committing to (R_c, n_out).
# =============================================================================
_OFFSETS = {"on-site": (0., 0., 0.), "face": (.5, 0., 0.),
            "body-center": (.5, .5, .5), "generic": (.37, .21, .5)}
_HSWEEP = [0.1, 0.15, 0.2, 0.25, 0.3]


def hartree_cartesian(R_c, n_out, h, offset, box=12.0):
    """J[rho_H] on a cubic grid (spacing h, atom at fractional offset*h) via the
    SIMPLE windowed Coulomb projection, folded to a single n_out kernel
    K(u)=sum_{n<n_out} (int_0^{R_c} R_n u' du') R_n(u) on |u|<=R_c, then V_H=K*rho."""
    from scipy.signal import fftconvolve
    n = int(np.ceil(2 * box / h)) // 2 * 2 + 1
    ax = (np.arange(n) - n // 2) * h
    X, Y, Z = np.meshgrid(ax - offset[0] * h, ax - offset[1] * h, ax - offset[2] * h,
                          indexing="ij")
    rho = _h_density(np.sqrt(X * X + Y * Y + Z * Z))
    m = int(np.ceil(R_c / h))
    cax = np.arange(-m, m + 1) * h
    KX, KY, KZ = np.meshgrid(cax, cax, cax, indexing="ij")
    kd = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    basis = RadialBesselBasis(n_out - 1, 0, R_c)
    qg = radial_gauss_grid(R_c, 400)
    omega = np.sum(basis.evaluate(0, qg.nodes) * (qg.weights * qg.nodes)[None, :], axis=1)
    Rk = basis.evaluate(0, np.clip(kd.ravel(), 0, R_c)).reshape(n_out, *kd.shape)
    K = np.tensordot(omega, Rk, axes=(0, 0))
    K[kd > R_c] = 0.0
    V = fftconvolve(rho, K, mode="same") * h ** 3
    return 0.5 * float(np.sum(rho * V)) * h ** 3


def run_hartree_cartesian():
    print(f"\n(A') practical-grid Hartree (R_c={RC_STAR}, n_out={NOUT_STAR}); err vs exact (mHa)")
    out = {"h": _HSWEEP, "offsets": list(_OFFSETS), "err_mHa": {}, "spread_mHa": {}}
    for h in _HSWEEP:
        errs = {k: 1e3 * (hartree_cartesian(RC_STAR, NOUT_STAR, h, off) - J_EXACT)
                for k, off in _OFFSETS.items()}
        out["err_mHa"][str(h)] = errs
        out["spread_mHa"][str(h)] = max(errs.values()) - min(errs.values())
        print(f"  h={h}: " + ", ".join(f"{k}={errs[k]:+.2f}" for k in _OFFSETS)
              + f"  | spread={out['spread_mHa'][str(h)]:.2f}")
    return out


def plot_hartree_cartesian(c):
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    hs = np.array(_HSWEEP)
    mk = {"on-site": "o", "face": "s", "body-center": "^", "generic": "D"}
    for k in _OFFSETS:
        ax.plot(hs, [c["err_mHa"][str(h)][k] for h in _HSWEEP], mk[k] + "-", ms=5, label=k)
    ax.axhspan(-1.0, 1.0, color="0.85", zorder=0)
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.text(0.1, 1.02, "1 mHa band", fontsize=7, color="0.4")
    ax.set(xlabel=r"grid spacing $h$ (bohr)",
           ylabel=r"$J_{\rm SIMPLE}-J_{\rm exact}$ (mHa)",
           title=r"Practical-grid Hartree error ($R_c=6$, $n_{\rm out}=10$)"
                 "\nvs spacing and sub-grid translation")
    ax.legend(fontsize=8, title="atom registration", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(_FIG / "hartree_cartesian.pdf")
    plt.close(fig)


# =============================================================================
# (B) Scale-invariance range vs inner-channel count (continuum; isolates the
#     transfer-resolution mechanism), at the fixed production n_out
# =============================================================================
def run_scale_vs_lambda():
    import invariant_stress_test as I
    from atom.descriptors.simple.pipeline import simple_from_window

    cl = I.Cluster(I.SYSTEMS["pseudo-N2"]["offaxis"]())

    def cont_d(n_in, lam):
        c = cl.scaled(lam)
        cw = I.continuum_window(c, n_in, L_MAX)

        def rb(R):
            sub = radial_gauss_grid(R, 192)
            rho0 = sum(I._axial_profile(d, float(np.linalg.norm(p / c.lam)), c.lam, 0, sub.nodes)
                       for d, p in c.sites)
            return 3.0 / R ** 3 * float(np.sum(sub.weights * sub.nodes ** 2 * rho0))

        d = simple_from_window(cw, rb, n_in, L_MAX, NOUT_STAR)
        return np.concatenate([np.asarray(d[l]).ravel() for l in range(L_MAX + 1)])

    lams = [1.0, 1.1, 1.25, 1.5, 1.67, 2.0, 2.5, 3.0, 4.0, 5.0]
    hs = [0.3, 0.2, 0.1]
    print(f"\n(B) scale invariance vs Lambda at n_out={NOUT_STAR}, R_c={RC_STAR}")
    out = {"n_out": NOUT_STAR, "R_c": RC_STAR, "lambda": lams, "h": hs,
           "n_in": {}, "lambda_max": {}, "n_conv": {}, "D": {}}
    for h in hs:
        n_in = int(round(RC_STAR / h))
        d1 = cont_d(n_in, 1.0)
        D = [float(np.linalg.norm(cont_d(n_in, lam) - d1) / np.linalg.norm(d1)) for lam in lams]
        out["n_in"][str(h)] = n_in
        out["lambda_max"][str(h)] = n_in / NOUT_STAR
        out["n_conv"][str(h)] = n_in * N_ANG
        out["D"][str(h)] = D
        print(f"  h={h}: n_in={n_in}, Lambda_max~{n_in/NOUT_STAR:.1f}, "
              f"N_conv={n_in*N_ANG}; D(2.0)={D[lams.index(2.0)]:.4f}")
    return out


def plot_scale_vs_lambda(s):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    lams = np.array(s["lambda"])
    colors = {"0.3": "tab:red", "0.2": "tab:green", "0.1": "tab:blue"}
    for h in ("0.3", "0.2", "0.1"):
        n_in = s["n_in"][h]
        lmax = s["lambda_max"][h]
        axes[0].semilogy(lams, np.maximum(s["D"][h], 1e-6), "o-", ms=4, color=colors[h],
                         label=fr"$h={float(h)}$ ($n_{{\rm in}}={n_in}$)")
        axes[0].axvline(lmax, color=colors[h], ls=":", lw=0.9)
    axes[0].axhline(0.01, color="0.5", ls="--", lw=0.8)
    axes[0].text(lams[-1], 0.011, "1%", fontsize=7, color="0.4", ha="right")
    axes[0].set(xlabel=r"scaling $\lambda$",
                ylabel=r"$D(\lambda)=\Vert \varrho(\lambda)-\varrho(1)\Vert/\Vert \varrho(1)\Vert$")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title(r"(a) dotted: $\Lambda_{\max}=n_{\rm in}/n_{\rm out}$")

    # design map: Lambda_max and N_conv vs h at fixed n_out, R_c
    hgrid = np.linspace(0.05, 0.4, 60)
    lam_max = RC_STAR / (hgrid * NOUT_STAR)
    n_conv = (RC_STAR / hgrid) * N_ANG
    ax = axes[1]
    ax.plot(hgrid, lam_max, "b-", label=r"$\Lambda_{\max}=R_c/(h\,n_{\rm out})$")
    ax.set(xlabel=r"grid spacing $h$ (bohr)", ylabel=r"scale range $\Lambda_{\max}$",
           title=r"(b) design map ($R_c=6$, $n_{\rm out}=10$, $\ell_{\max}=3$)")
    ax.axvline(0.3, color="0.6", ls=":", lw=0.9)
    ax2 = ax.twinx()
    ax2.plot(hgrid, n_conv, "r--", label=r"$N_{\rm conv}=(R_c/h)(\ell_{\max}+1)^2$")
    ax2.set_ylabel(r"convolutions $N_{\rm conv}$", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(_FIG / "scale_vs_lambda.pdf")
    plt.close(fig)


def plot_params_main(hartree, hartree_cart):
    """Condensed 3-panel main-text figure: the Hartree-operator calibration of R_c and
    n_out. (a) radial-basis convergence vs n_out for several windows; (b) the
    n_out-converged window (Coulomb-tail) floor vs R_c; (c) the practical-grid error vs
    spacing h and sub-grid registration. Together: R_c=6, n_out=10 -> ~0.5 mHa, robust
    to h<=0.2-0.25 bohr."""
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
    # (a) radial-basis convergence
    nouts = hartree["nouts"]
    for rc in ("4.0", "6.0", "8.0"):
        err = [abs(hartree["grid"][rc][str(n)] - J_EXACT) * 1e3 for n in nouts]
        ax[0].semilogy(nouts, err, "o-", ms=4, label=fr"$R_c={float(rc):.0f}$")
    ax[0].axhline(1.0, color="0.5", ls="--", lw=0.9)
    ax[0].scatter([NOUT_STAR], [abs(hartree["grid"]["6.0"][str(NOUT_STAR)] - J_EXACT) * 1e3],
                  s=80, facecolors="none", edgecolors="k", zorder=5)
    ax[0].set(xlabel=r"$n_{\rm out}$", ylabel=r"$|J-J_{\rm exact}|$ (mHa)",
              title="(a) radial-basis convergence")
    ax[0].legend(fontsize=8)
    # (b) window (tail) floor
    rcs = hartree["rcs"]
    errf = [abs(hartree["tail_floor"][str(rc)] - J_EXACT) * 1e3 for rc in rcs]
    ax[1].semilogy(rcs, np.maximum(errf, 1e-3), "s-", color="tab:purple")
    ax[1].axhline(1.0, color="0.5", ls="--", lw=0.9)
    ax[1].axvline(RC_STAR, color="0.6", ls=":", lw=0.9)
    ax[1].set(xlabel=r"$R_c$ (bohr)", ylabel=r"$|J-J_{\rm exact}|$ (mHa)",
              title="(b) window (tail) floor")
    # (c) practical grid: error vs h and registration
    hs = hartree_cart["h"]
    mk = {"on-site": "o", "face": "s", "body-center": "^", "generic": "D"}
    for k in hartree_cart["offsets"]:
        ax[2].plot(hs, [hartree_cart["err_mHa"][str(h)][k] for h in hs], mk.get(k, "o") + "-",
                   ms=4, label=k)
    ax[2].axhspan(-1.0, 1.0, color="0.85", zorder=0)
    ax[2].axhline(0.0, color="0.5", lw=0.6)
    ax[2].set(xlabel=r"$h$ (bohr)", ylabel=r"$J_{\rm SIMPLE}-J_{\rm exact}$ (mHa)",
              title="(c) practical grid + registration")
    ax[2].legend(fontsize=7, title="registration", title_fontsize=7)
    fig.suptitle(r"Hartree energy of H via the SIMPLE Coulomb projection: "
                 r"$R_c=6$ bohr, $n_{\rm out}=10 \Rightarrow {\sim}0.5$ mHa")
    fig.tight_layout()
    fig.savefig(_FIG / "params_main.pdf")
    plt.close(fig)


def regen_from_json():
    """Regenerate all parameter-selection figures from the cached JSON (no recompute)."""
    d = json.loads(_OUT.read_text())
    plot_hartree(d["hartree"])
    plot_hartree_cartesian(d["hartree_cartesian"])
    plot_scale_vs_lambda(d["scale_vs_lambda"])
    plot_params_main(d["hartree"], d["hartree_cartesian"])
    print(f"regenerated hartree_convergence/_cartesian, scale_vs_lambda, params_main from {_OUT.name}")


def run_fixed_nin():
    """Confirm the convolution count is set by n_in (a design choice = n_out*Lambda),
    not by h: with n_in fixed at 2*n_out=20 (-> N_conv=320 at l_max=3), the scale range
    Lambda_max=n_in/n_out=2 is h-independent and finer h only improves accuracy."""
    import invariant_stress_test as I
    from atom.descriptors.simple.pipeline import (LatticeStencil, grid_descriptors,
                                                  simple_from_window)

    n_in = 2 * NOUT_STAR
    cl = I.Cluster(I.SYSTEMS["pseudo-N2"]["offaxis"]())

    def stack(d):
        return np.concatenate([np.asarray(d[l]).ravel() for l in range(L_MAX + 1)])

    def cont(c):
        cw = I.continuum_window(c, n_in, L_MAX)

        def rb(R):
            sub = radial_gauss_grid(R, 192)
            r0 = sum(I._axial_profile(dd, float(np.linalg.norm(p / c.lam)), c.lam, 0, sub.nodes)
                     for dd, p in c.sites)
            return 3.0 / R ** 3 * float(np.sum(sub.weights * sub.nodes ** 2 * r0))
        return simple_from_window(cw, rb, n_in, L_MAX, NOUT_STAR)

    dcont = stack(cont(cl))
    hs = [0.3, 0.2, 0.1, 0.05]
    lams = [1.25, 1.5, 2.0, 2.5]
    print(f"\n(C) fixed n_in={n_in} (N_conv={n_in*N_ANG} at l_max={L_MAX}); "
          f"Lambda_max=n_in/n_out={n_in/NOUT_STAR:.0f}, h-independent")
    out = {"n_in": n_in, "n_out": NOUT_STAR, "n_conv": n_in * N_ANG,
           "lambda_max": n_in / NOUT_STAR, "h": hs, "lambda": lams, "D": {}, "acc": {}}
    for h in hs:
        st = LatticeStencil(h, n_in)
        d1 = grid_descriptors(cl.density, st, n_out=NOUT_STAR)
        base = np.linalg.norm(stack(d1))
        D = [float(np.linalg.norm(stack(grid_descriptors(cl.scaled(lam).density, st, n_out=NOUT_STAR))
                                  - stack(d1)) / base) for lam in lams]
        acc = float(np.linalg.norm(stack(d1) - dcont) / np.linalg.norm(dcont))
        out["D"][str(h)] = D
        out["acc"][str(h)] = acc
        print(f"  h={h} (n_in<=R_c/h: {n_in <= RC_STAR/h}): D(lam=2)={D[2]:.4f}, "
              f"accuracy vs continuum={acc:.4f}")
    return out


if __name__ == "__main__":
    _FIG.mkdir(exist_ok=True)
    _DATA.mkdir(exist_ok=True)
    hartree = run_hartree()
    hartree_cart = run_hartree_cartesian()
    fixed_nin = run_fixed_nin()
    scale = run_scale_vs_lambda()
    _OUT.write_text(json.dumps({"hartree": hartree, "hartree_cartesian": hartree_cart,
                                "fixed_nin": fixed_nin, "scale_vs_lambda": scale},
                               indent=2, default=float))
    plot_hartree(hartree)
    plot_hartree_cartesian(hartree_cart)
    plot_scale_vs_lambda(scale)
    plot_params_main(hartree, hartree_cart)
    print(f"\nwrote {_OUT}")
    print(f"wrote hartree_convergence.pdf, hartree_cartesian.pdf, scale_vs_lambda.pdf, "
          f"params_main.pdf to {_FIG}/")
