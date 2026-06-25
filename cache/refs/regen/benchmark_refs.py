"""Non-SCF exchange-energy benchmark of the referenced kernel functionals vs reference-free.

For each atom we evaluate E_x = integral rho eps_x on the CACHED HF density (non-SCF -- isolates
hole/energy quality from SCF density drift) and compare to Ehf (exact exchange of the HF density).
We compare the reference-free baseline against functionals carrying n16/n64/n256 reference nodes
(atom/xc/data/kernel_fp_refs_n{N}.npz, built by build_kernel_refs.py).

in-domain  = the exact closed-subshell atoms that contributed references (their features are sampled);
out-of-domain = open-(sub)shell atoms NOT in the reference pool (transferability test).

Speed: the LO width calibrates on the reference-free BACKBONE (refs excluded), so we calibrate once
per atom and just rebuild nodes per reference set.
"""
import argparse
import os
import sys

import numpy as np

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
import atom.xc.simple_hole_expansion as She  # noqa: E402
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP  # noqa: E402
from atom.utils.periodic import atomic_number_to_name as nm  # noqa: E402
from cache.refs.loader import load_hf  # noqa: E402

DATA = os.path.join(_REPO, "atom", "xc", "data")
IN_DOMAIN = [2, 10, 12, 18, 20]            # He Ne Mg Ar Ca  (exact closed-subshell, in pool)
OUT_DOMAIN = [3, 7, 8, 9, 29]              # Li N O F Cu     (open-shell, not in pool; Cu = d-block)
NONE = "/nonexistent.npz"


def ex_nonscf(F, rho, cprime, g):
    return float(np.sum(F.energy_weights * rho * F._kernel_eps(cprime, rho, g)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256, 512])
    ap.add_argument("--l0", type=float, default=0.7, help="l=0 width (balanced optimum ~0.7)")
    ap.add_argument("--l1", type=float, default=0.5, help="l=1/s^2 width (balanced optimum ~0.5)")
    args = ap.parse_args()
    variants = [("reffree", NONE)] + [(f"n{s}", os.path.join(DATA, f"kernel_fp_refs_n{s}.npz"))
                                      for s in args.sizes]
    cols = [v[0] for v in variants]
    rows = []
    for dom, Zs in [("in", IN_DOMAIN), ("out", OUT_DOMAIN)]:
        for Z in Zs:
            hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
            r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12)
            w = np.asarray(hf["w"])[o]; Ehf = float(hf["Ehf"])
            She._KERNEL_FP_REFS = NONE
            F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)   # calibrate once (backbone)
            cprime = np.array([op @ rho for op in F._ops]); g = F._grad_op @ rho
            errs = {}
            for name, path in variants:
                She._KERNEL_FP_REFS = path
                F._fp_l0, F._fp_l1 = args.l0, args.l1
                F._build_fp_nodes(include_refs=True)                    # cheap
                errs[name] = 1e3 * (ex_nonscf(F, rho, cprime, g) - Ehf)
            rows.append((dom, nm(Z), Ehf, errs))

    w0 = 7
    print(f"non-SCF E_x error vs HF (mHa).  in-domain = ref-contributing atoms\n")
    print(f"{'dom':>3} {'atom':>4} {'Ehf':>9} " + " ".join(f"{c:>{w0}}" for c in cols))
    for dom, sym, Ehf, errs in rows:
        print(f"{dom:>3} {sym:>4} {Ehf:>9.4f} " + " ".join(f"{errs[c]:>{w0}.0f}" for c in cols))
    print()
    for dom in ("in", "out"):
        sel = [e for d, _, _, e in rows if d == dom]
        line = f"MAE {dom:>3}: " + " ".join(
            f"{c}={np.mean([abs(e[c]) for e in sel]):.0f}" for c in cols)
        print(line)

    out = os.path.join(_REPO, "reports", "hole_expansion", "benchmark_refs.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"{'dom':>3} {'atom':>4} {'Ehf':>9} " + " ".join(f"{c:>{w0}}" for c in cols) + "\n")
        for dom, sym, Ehf, errs in rows:
            f.write(f"{dom:>3} {sym:>4} {Ehf:>9.4f} " + " ".join(f"{errs[c]:>{w0}.0f}" for c in cols) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
