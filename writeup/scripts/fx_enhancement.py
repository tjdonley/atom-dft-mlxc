"""Exchange enhancement factor of the SIMPLE kernel-mapped hole. Produces figures/fx_enhancement.pdf.

Two curves:
  (1) BACKBONE (reference-free): the realized enhancement F_x(s)=eps_x/eps_x^LDA on the homogeneous gas
      at reduced gradient s, evaluated at the pure-GEA-axis feature point (l=0 monopole = HEG signature
      cn_HEG, l=1 coordinate = s^2). The single l=1 kernel node fixes the small-s slope to the exact
      second-order value mu=10/81 [Eq. (fx)]; the bounded deformation peaks just above LDA and relaxes
      back -- the reference-free hole is essentially LDA+Fermi-Amaldi.
  (2) REFERENCED (on atoms): the realized enhancement F_x=eps_x/eps_x^LDA of the FINAL referenced
      functional (exact holes covering the closed-shell density manifold, rho>=1e-2), evaluated
      pointwise on real closed-shell atomic densities (Ne, Ar, Kr; energy-relevant region rho>0.05).
      The references lift the enhancement from the reference-free ~1.05 to PBE-like values (~1.4) at the
      gradients atoms actually carry -- the source of the atomic binding (Table tab:ref-energies).
Raw pointwise second-order GEA (diverges) and PBE are shown for reference; dotted line = Lieb-Oxford.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from atom.xc.simple_hole_expansion import (  # noqa: E402
    SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters)
from cache.refs.loader import load_hf  # noqa: E402

A = 10.0 / 81.0                                         # second-order GEA coefficient [Eq. (fx)]
KAPPA_PBE, MU = 0.804, 0.2195149727645171               # PBE exchange
LO_FX = 1.804                                           # Lieb-Oxford bound on F_x
REF = os.path.join(str(_REPO), "atom", "xc", "data", "kernel_fp_refs_closed_rf001.npz")


def F_pbe(s):
    return 1.0 + KAPPA_PBE - KAPPA_PBE / (1.0 + MU * s ** 2 / KAPPA_PBE)


def main():
    r = np.linspace(1e-3, 14.0, 800); w = np.gradient(r)
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)   # reference-free backbone

    # (1) reference-free GEA-axis enhancement (the GEA deformation mode is charge/on-top neutral, so
    #     this energy moment is exact; the l=1 node amplitude fixes the slope to 10/81)
    def Fx_backbone(svals):
        cn = np.tile(F._cnH, (len(svals), 1))
        x = F._xfeat(cn, np.asarray(svals) ** 2)
        return 1.0 + F._fp_kappa * ((F._Kmat(x, F._fp_Xnodes) @ F._fp_coef) @ F._Cmom)

    s = np.linspace(0.0, 5.0, 400); Fb = Fx_backbone(s)
    sm = (s > 1e-6) & (s < 0.2); slope = np.polyfit(s[sm] ** 2, Fb[sm] - 1.0, 1)[0]
    print(f"backbone small-s slope dF/d(s^2) = {slope:+.5f} (exact 10/81 = {A:.5f}); peak {Fb.max():.3f}")

    # (2) referenced functional, realized enhancement on real atoms
    p = SIMPLEHOLEKERNELFPParameters(fp_l0=0.7, fp_l1=0.5, fp_ref_ridge=1e-8, refs_path=REF)
    clda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    ss, ff = [], []
    for Z in (10, 18, 36):
        hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
        rr = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); ww = np.asarray(hf["w"])[o]
        G = SIMPLE_HOLE_KERNEL_FP(r_quad=rr, quadrature_weights=ww, params=p)
        cp = np.array([op @ rho for op in G._ops]); g = G._grad_op @ rho
        Fx = G._kernel_eps(cp, rho, g) / (clda * rho ** (1.0 / 3.0))
        kF = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0); sval = np.abs(g) / (2.0 * kF * rho)
        m = (rho > 0.05) & (sval < 5.0); ss.append(sval[m]); ff.append(Fx[m])
    ss = np.concatenate(ss); ff = np.concatenate(ff)
    print(f"referenced realized F_x on atoms (rho>0.05): range [{ff.min():.2f},{ff.max():.2f}], median {np.median(ff):.2f}")

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axhline(LO_FX, color="0.4", lw=0.9, ls=":", zorder=0)
    ax.text(0.05, LO_FX + 0.03, "Lieb-Oxford", fontsize=6, color="0.4")
    ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)                              # LDA
    ax.plot(s, 1.0 + A * s ** 2, color="0.55", lw=1.3, ls="--", label=r"GEA (raw)")
    ax.plot(s, F_pbe(s), "k-", lw=1.2, label="PBE")
    ax.scatter(ss, ff, s=4, alpha=0.10, color="steelblue", edgecolors="none", zorder=1,
               label="SIMPLE referenced (atoms)")
    ax.plot(s, Fb, color="crimson", lw=1.8, zorder=4, label="SIMPLE backbone (ref-free)")
    ax.set_xlabel(r"reduced gradient $s$"); ax.set_ylabel(r"exchange enhancement $F_x$")
    ax.set_xlim(0, 5); ax.set_ylim(0.95, 2.0)
    ax.legend(fontsize=5.8, loc="lower right", frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    out = _REPO / "writeup" / "figures" / "fx_enhancement.pdf"
    fig.savefig(out); print(f"wrote {out}")


if __name__ == "__main__":
    main()
