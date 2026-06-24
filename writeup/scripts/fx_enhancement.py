"""Exchange enhancement factor of the scale-free SIMPLE hole with the PBE-style,
Lieb-Oxford-set saturation. Produces figures/fx_enhancement.pdf.

The hole reproduces the second-order gradient expansion at small s [Eq. (fx)],
    F_x(s) = 1 + (10/81) s^2 ,
which diverges as s -> inf. The functional carries this through a deformation amplitude
c that is smoothly saturated, c = kappa * tanh(c_raw/kappa) with c_raw the GEA2 amplitude,
so the enhancement saturates instead of diverging. The ceiling kappa is fixed by the
Lieb-Oxford bound (max F_x <= 1.804 relative to LDA) -- no free parameter. The on-top value
S(0)=1 is preserved for any c (Fermi-Amaldi intact).

We plot the ACTUAL re-summed enhancement F_x(s) = eps_x(c(s))/eps_x^LDA computed from the
reference hole (simple_hole_explicit.hole_solve_def) on the homogeneous gas at the adaptive
window R_ad = X/k_F (the GEA2 second-order hole), against the raw (pointwise, diverging) GEA
and PBE. The horizontal line is the Lieb-Oxford ceiling.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from atom.xc.simple_hole_explicit import hole_solve, hole_solve_def  # noqa: E402

A = 10.0 / 81.0                                         # second-order GEA coefficient [Eq. (fx)]
KAPPA_PBE, MU = 0.804, 0.2195149727645171               # PBE exchange
LO_FX = 1.804                                           # Lieb-Oxford bound on F_x
X = 8.0                                                 # dimensionless hole window k_F R_ad
RHO0 = 1.0


def F_pbe(s):
    return 1.0 + KAPPA_PBE - KAPPA_PBE / (1.0 + MU * s ** 2 / KAPPA_PBE)


def main():
    kF = (3.0 * np.pi ** 2 * RHO0) ** (1.0 / 3.0)
    rc = X / kF
    eps_lda = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * RHO0 ** (1.0 / 3.0)
    unif = lambda u: np.full_like(np.asarray(u, float), RHO0)
    eps_bare = hole_solve(unif, rc)[0]
    F_uniform = eps_bare / eps_lda                       # finite-window LDA-limit offset
    h = 1e-4
    R_sf = (hole_solve_def(unif, +h, rc)[0] - hole_solve_def(unif, -h, rc)[0]) / (2 * h) / eps_lda
    kappa = (LO_FX / F_uniform - 1.0) / R_sf             # LO-set saturation ceiling (amplitude)
    coeff = A / R_sf                                     # scale-free amplitude A = target/R_sf
    print(f"R_sf={R_sf:+.4f} F_uniform={F_uniform:.4f} kappa_LO={kappa:.3f}")

    def Fx_curve(s):
        out = np.empty_like(s)
        for i, si in enumerate(s):
            craw = coeff * si ** 2                        # second-order GEA amplitude
            c = kappa * np.tanh(craw / kappa)            # PBE-style smooth saturation
            out[i] = hole_solve_def(unif, float(c), rc)[0] / eps_lda
        return out

    s = np.linspace(0.0, 3.0, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # raw pointwise GEA (diverges) and PBE
    ax.plot(s, 1.0 + A * s ** 2, color="0.55", lw=1.3, ls="--", label=r"GEA (raw)")
    ax.plot(s, F_pbe(s), "k-", lw=1.2, label="PBE")
    # SIMPLE scale-free hole, LO-saturated second-order (GEA2) enhancement
    ax.plot(s, Fx_curve(s), color="crimson", lw=1.8, label=r"SIMPLE hole (GEA2)")
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
