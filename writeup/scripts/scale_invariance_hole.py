"""Scale invariance of the exchange hole on atoms. Produces figures/scale_invariance_hole.pdf.

Uniform density scaling (exact constraint X3): E_x[gamma^3 rho(gamma r)] = gamma E_x[rho]. For
the hydrogenic density rho_Z(r) = (Z^3/pi) e^{-2Zr}, the map Z -> gamma Z IS the uniform
scaling, so the exact exchange has E_x proportional to Z, i.e. E_x/Z is Z-INDEPENDENT.

The scale-free hole evaluates the self-energy on the implicit adaptive radius R_ad = X/k_F
(reached from the fixed-R_c SIMPLE features by the transfer), so E_x/Z is flat (X3 holds). The
fixed-R_c hole (no transfer, window pinned at R_c) breaks scaling: E_x/Z drifts as the density
scale moves relative to the fixed window. Reproduced here by x_window -> inf (R_ad = R_c always).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from atom.xc.simple_hole import scale_free_hole_energy  # noqa: E402
from atom.descriptors.simple.params import R_C  # noqa: E402

X = 8.0


def main():
    r = np.logspace(-3, np.log10(40.0), 320)
    Zs = np.geomspace(0.5, 12.0, 7)
    sf, fx = [], []
    for Z in Zs:
        rho = (Z ** 3 / np.pi) * np.exp(-2.0 * Z * r)
        wq = 4.0 * np.pi * r ** 2 * rho
        e_sf = scale_free_hole_energy(r, rho, 60.0, x_window=X, nu=120)        # adaptive (uncapped)
        e_fx = scale_free_hole_energy(r, rho, R_C, x_window=1e9, nu=120)        # fixed R_c
        sf.append(np.trapz(wq * e_sf, r) / Z)
        fx.append(np.trapz(wq * e_fx, r) / Z)
        print(f"Z={Z:6.3f}  E_x/Z: scale-free={sf[-1]:+.4f}  fixed-R_c={fx[-1]:+.4f}", flush=True)
    sf, fx = np.array(sf), np.array(fx)
    ref = sf[np.argmin(np.abs(Zs - 1.0))]                                       # normalize to Z=1

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(Zs, fx / ref, "s--", color="0.5", lw=1.3, ms=4, label=r"fixed $R_c$ (no transfer)")
    ax.plot(Zs, sf / ref, "o-", color="crimson", lw=1.8, ms=4, label="scale-free (adaptive $R_{ad}$)")
    ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"hydrogenic charge $Z$  (density scale $\gamma$)")
    ax.set_ylabel(r"$E_x/Z$  (normalized; flat $\Leftrightarrow$ scale-free)")
    ax.set_ylim(0.90, 1.20)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.8)
    fig.tight_layout(pad=0.3)
    out = _REPO / "writeup" / "figures" / "scale_invariance_hole.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
