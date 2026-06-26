"""Re-reference a kernel-reference file to a SATURATING-gradient backbone (sat_gradient mode).

The reference DELTAs are stored as sigma_ref - sigma_LDA. When the functional runs with sat_gradient
(a PBE-like saturating gradient backbone added to rhotilde_LDA), those LDA-referenced DELTAs
DOUBLE-COUNT the gradient enhancement -- the refs then HURT (verified: 120 -> 181 mHa in-domain).

Fix: subtract the backbone's contribution at each node so the kernel reproduces the exact hole on top
of the saturating backbone instead of on top of LDA:

  DELTA_new = DELTA_old - A_max h(s^2_raw) _dgea
    A_max = kappa_lo / (kappa * _dgb)      h(u) = a u/(1+a u),  a = mu/kappa_lo
    s^2_raw recovered from the stored bounded s^2 via _bound^{-1}: v = b/(1 - b/4)  (M=_INV_BOUND=4)

At a node the kernel adds DELTA_new and the saturating term adds back A_max h _dgea, so rhotilde ->
sigma_ref exactly (mu cancels at the nodes -- the backbone only matters BETWEEN nodes / out of domain).

NOTE (finding, 2026-06-26): re-referencing fixes the double-counting (refs help again: 120 -> 88 mHa
in-domain, exact interp), but the in-domain closed-shell ceiling is ~88 mHa and mu-INDEPENDENT (the
backbone cancels at nodes) -- the exact-hole-reproduction residual (R_c=6 leakage + kernel
interpolation under sparse nodes). That ceiling is above PBE (49); breaking it needs more FEATURES
(tau / higher multipoles), not a better backbone. See reports/hole_expansion/resaturate_refs.txt.

Usage:
  python3 cache/refs/regen/build_refs_resaturate.py --src kernel_fp_refs_closed_n512.npz \
          --mu 0.2195 --kappa-lo 0.804 [--dst <name>]
Default --dst inserts a _sat tag before .npz. Writes into atom/xc/data/. Does not touch the source.
"""
import argparse
import os
import sys

import numpy as np

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import (  # noqa: E402
    SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters, _INV_BOUND)

DATA = os.path.join(_REPO, "atom", "xc", "data")


def resaturate(src, dst, mu, kappa_lo):
    z = np.load(os.path.join(DATA, src))
    X, DELTA = z["X"].copy(), z["DELTA"].copy()
    b = X[:, -1]                                       # stored BOUNDED s^2 (the kernel feature)
    raw_s2 = b / (1.0 - b / _INV_BOUND)                # invert _bound (v = b/(1-b/M), M=4)
    r = np.linspace(1e-3, 14.0, 400)
    Fb = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=np.gradient(r),
                               params=SIMPLEHOLEKERNELFPParameters(
                                   sat_gradient=True, fp_mu=mu, fp_kappa_lo=kappa_lo))
    A_max = kappa_lo / (Fb._fp_kappa * Fb._fp_dgb)
    a = mu / kappa_lo
    h = a * raw_s2 / (1.0 + a * raw_s2)
    DELTA_new = DELTA - (A_max * h)[:, None] * Fb._dgea[None, :]
    out = {k: z[k] for k in z.files}
    out["DELTA"] = DELTA_new
    out["sat_backbone_mu"] = np.array(mu)
    out["sat_backbone_kappa_lo"] = np.array(kappa_lo)
    np.savez(os.path.join(DATA, dst), **out)
    return np.median(np.linalg.norm(DELTA, axis=1)), np.median(np.linalg.norm(DELTA_new, axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source npz in atom/xc/data/")
    ap.add_argument("--mu", type=float, default=0.2195, help="backbone gradient slope (PBE 0.2195)")
    ap.add_argument("--kappa-lo", type=float, default=0.804, help="Lieb-Oxford ceiling")
    ap.add_argument("--dst", default=None, help="output npz name (default: <src>_sat.npz)")
    args = ap.parse_args()
    dst = args.dst or args.src.replace(".npz", "_sat.npz")
    n0, n1 = resaturate(args.src, dst, args.mu, args.kappa_lo)
    print(f"{args.src} -> {dst}  (mu={args.mu}, kappa_lo={args.kappa_lo})  "
          f"||DELTA|| med {n0:.4f} -> {n1:.4f}")


if __name__ == "__main__":
    main()
