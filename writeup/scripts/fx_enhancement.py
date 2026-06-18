"""Exchange enhancement factor: the fourth-order gradient expansion (GEA) form that
the SIMPLE hole reproduces [Eq. (fx)], compared to PBE. Produces
figures/fx_enhancement.pdf.

GEA (Eq. (fx)):
    F_x(s,q) = 1 + (10/81) s^2 + (146/2025) q^2 - (73/405) s^2 q ,
plotted for several reduced Laplacians q (typical physical range), against the PBE
enhancement factor F_x^PBE(s) = 1 + kappa - kappa/(1 + mu s^2/kappa). These are the
pointwise GEA curves; they EXCLUDE the re-summation over the density that the hole
functional performs (which bounds F_x at large s), so they grow quadratically.

Overlaid (points) are the actual re-summed F_x of representative atoms (He, Be, N,
Ne; from atom_trajectories.py), sampled in the gradient regime (s > 0.6, |q| < 1.5,
normal/non-Fermi-Amaldi) -- the bounded F_x the functional actually produces. They
are plotted as points, not curves: F_x is not single-valued in s for a real atom
(it also depends on q, and shell structure makes s(r) non-monotonic).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ATOMS_DATA = Path(__file__).resolve().parent / "data" / "atom_sqfx.npz"
_ATOM_COLORS = {"He": "crimson", "Be": "darkorange", "N": "forestgreen", "Ne": "purple"}
_Q_CAP = 1.5   # physical reduced-Laplacian window for the atom overlay (excludes the
               # low-density, high-q pseudopotential central spikes, outside the GEA range)

# GEA coefficients [Eq. (fx)]
A, B, C = 10.0 / 81.0, 146.0 / 2025.0, -73.0 / 405.0
# PBE exchange (Perdew-Burke-Ernzerhof 1996)
KAPPA, MU = 0.804, 0.2195149727645171


def F_simple(s, q):
    return 1.0 + A * s ** 2 + B * q ** 2 + C * s ** 2 * q


def F_pbe(s):
    return 1.0 + KAPPA - KAPPA / (1.0 + MU * s ** 2 / KAPPA)


def main():
    s = np.linspace(0.0, 3.0, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # GEA model reference (pointwise, dashed): q=0 and q=1 bracket the slope tuning.
    for q, col in ((0.0, "0.55"), (1.0, "0.2")):
        ax.plot(s, F_simple(s, q), color=col, lw=1.4, ls="--", label=fr"GEA, $q={q:.0f}$")
    ax.plot(s, F_pbe(s), "k-", lw=1.4, label="PBE")
    # Representative atoms: the actual functional's (re-summed) F_x sampled along each
    # atom, normal (non-Fermi-Amaldi) regime, physical q window. Plotted as POINTS, not
    # a curve: F_x is not single-valued in s for a real atom (it also depends on q, and
    # shell structure makes s(r) non-monotonic), so a connected line would be
    # misleading -- the markers show where each atom lives in the (s, F_x) plane.
    if _ATOMS_DATA.exists():
        d = np.load(_ATOMS_DATA)
        for sym in [a.decode() if isinstance(a, bytes) else str(a) for a in d["atoms"]]:
            rho, sa, qa, Fxa, fa = d[f"{sym}_rho"], d[f"{sym}_s"], d[f"{sym}_q"], d[f"{sym}_Fx"], d[f"{sym}_fa"]
            # s > 0.6: the gradient regime, past the inner-shell shoulder (the inner
            # valence, r < r_peak, also crosses s ~ 0.5 with q > 0 -- a shell-structure
            # branch, not a uniform-gas point, so excluded from the F_x(s) overlay).
            m = ((rho > 1e-2 * rho.max()) & (sa > 0.6) & (sa < 3.0) & (~fa)
                 & (np.abs(qa) < _Q_CAP))
            ax.scatter(sa[m][::10], Fxa[m][::10], s=7, alpha=0.7, edgecolors="none",
                       color=_ATOM_COLORS.get(sym, "gray"), label=sym)
    ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)        # LDA
    ax.set_xlabel(r"reduced gradient $s$")
    ax.set_ylabel(r"exchange enhancement $F_x$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0.9, 2.2)
    ax.legend(fontsize=6.0, loc="upper left", frameon=False, ncol=2,
              columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/fx_enhancement.pdf")
    print("wrote figures/fx_enhancement.pdf")


if __name__ == "__main__":
    main()
