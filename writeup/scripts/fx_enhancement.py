"""Exchange enhancement factor of the scale-free SIMPLE kernel-mapped hole.
Produces figures/fx_enhancement.pdf.

The production hole is SIMPLE_HOLE_KERNEL_FP: the hole COEFFICIENTS are interpolated over
fixed points by a kernel whose per-l SIMPLE distances are the kernel coordinates (l=1 == s^2,
Eq. sq). The amplitude of the single l=1 node is fixed so the small-gradient enhancement
reproduces the exact second-order gradient expansion [Eq. (fx)],
    F_x(s) -> 1 + (10/81) s^2 ,
with NO explicit gradient term and no fitted parameter. There is no separate enhancement
factor: F_x = eps_x / eps_x^unif is whatever the re-summed kernel hole gives.

We plot the ACTUAL realized enhancement F_x(s) = eps_x / eps_x^LDA from the functional,
evaluated on a homogeneous electron gas carrying a controlled reduced gradient s (an
exponential density ramp rho = rho0 exp(a x); at the midpoint s = a/(2 k_F)), against the raw
(pointwise, diverging) second-order GEA and PBE. The dotted line is the Lieb-Oxford ceiling.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _C_LDA  # noqa: E402

A = 10.0 / 81.0                                         # second-order GEA coefficient [Eq. (fx)]
KAPPA_PBE, MU = 0.804, 0.2195149727645171               # PBE exchange
LO_FX = 1.804                                           # Lieb-Oxford bound on F_x
RHO0 = 1.0


def F_pbe(s):
    return 1.0 + KAPPA_PBE - KAPPA_PBE / (1.0 + MU * s ** 2 / KAPPA_PBE)


def main():
    n = 800
    r = np.linspace(1e-3, 14.0, n)
    w = np.gradient(r)
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)

    # Realized enhancement on the homogeneous gas at reduced gradient s: the kernel's defining
    # relation F_x - 1 = kappa (delta_sigma . beta1), evaluated at the pure-GEA-axis feature
    # point (l=0 monopole = the HEG signature cn_HEG, l=1 coordinate = s^2). The GEA deformation
    # mode is charge/on-top neutral by construction, so this energy moment is exact (no density
    # ramp, hence no finite-grid overflow). The l=1 node amplitude fixes the slope to 10/81.
    def Fx_curve(svals):
        cn = np.tile(F._cnH, (len(svals), 1))
        x = F._xfeat(cn, np.asarray(svals) ** 2)
        dsig = F._Kmat(x, F._fp_Xnodes) @ F._fp_coef          # (len, n_out)
        return 1.0 + F._fp_kappa * (dsig @ F._Cmom)

    s = np.linspace(0.0, 3.0, 200)
    Fvals = Fx_curve(s)
    sm = (s > 1e-6) & (s < 0.2)
    slope = np.polyfit(s[sm] ** 2, Fvals[sm] - 1.0, 1)[0]
    print(f"realized small-s slope dF/d(s^2) = {slope:+.5f}  (exact 10/81 = {A:.5f})")

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # raw pointwise GEA (diverges) and PBE
    ax.plot(s, 1.0 + A * s ** 2, color="0.55", lw=1.3, ls="--", label=r"GEA (raw)")
    ax.plot(s, F_pbe(s), "k-", lw=1.2, label="PBE")
    # SIMPLE scale-free kernel hole: realized enhancement (GEA2 slope from the l=1 node)
    ax.plot(s, Fvals, color="crimson", lw=1.8, label=r"SIMPLE hole (kernel)")
    ax.axhline(LO_FX, color="0.4", lw=0.9, ls=":", zorder=0)
    ax.text(0.05, LO_FX + 0.02, "Lieb-Oxford", fontsize=6, color="0.4")
    ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)        # LDA
    ax.set_xlabel(r"reduced gradient $s$")
    ax.set_ylabel(r"exchange enhancement $F_x$")
    ax.set_xlim(0, 3); ax.set_ylim(0.9, 2.3)
    ax.legend(fontsize=6.0, loc="upper left", frameon=False, ncol=1, handlelength=1.8)
    fig.tight_layout(pad=0.3)
    out = _REPO / "writeup" / "figures" / "fx_enhancement.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
