"""Append the enclosed charge Q (bounded) as a kernel coordinate to the closed-shell l2power references
-> *_l2power_Q.npz (use_Q=True). Q is recomputed per atom EXACTLY as _kernel_eps does
(Q = 4 pi R_ad^3 (d . Bmom), d = c_ad/(4 pi R_ad^1.5), bounded) so node and query coordinates match.
This lets the kernel LEARN the few-electron / tail hole shape (Q spans ~2..20) instead of the hand-built
Fermi-Amaldi shape + W(Q) switch. Usage: python3 cache/refs/regen/build_refs_add_Q.py"""
import os, sys
import numpy as np
np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
DATA = os.path.join(_REPO, "atom", "xc", "data")

def Q_for_atom(Z):
    hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); w = np.asarray(hf["w"])[o]
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)
    cprime = np.array([op @ rho for op in F._ops]); R_ad, _ = F._R_ad(rho); c_ad = F._c_ad(cprime, R_ad)
    d = c_ad / (4.0 * np.pi * R_ad[:, None] ** 1.5)
    return r, _bound(4.0 * np.pi * R_ad ** 3 * (d @ F._Bmom))[0]

def main(src="kernel_fp_refs_closed_rf001_l2power.npz", dst="kernel_fp_refs_closed_rf001_l2power_Q.npz"):
    z = np.load(os.path.join(DATA, src)); X = z["X"]; DELTA = z["DELTA"]; idx = z["idx"]
    pool = load_hole_refs_full(); Zr = pool["Z"][idx]; r0r = pool["r0"][idx]; cache = {}; Qb = np.zeros(len(idx))
    for Z in sorted(set(Zr.tolist())):
        if Z not in cache: cache[Z] = Q_for_atom(Z)
        m = Zr == Z; rA, QA = cache[Z]; Qb[m] = np.interp(r0r[m], rA, QA)
        print(f"  Z={Z:2d}: {int(m.sum())} refs, Qbounded range [{Qb[m].min():.2f},{Qb[m].max():.2f}]", flush=True)
    XQ = np.hstack([X, Qb[:, None]])
    out = os.path.join(DATA, dst)
    np.savez(out, X=XQ, DELTA=DELTA, idx=idx, fp_l0=float(z["fp_l0"]), fp_l1=float(z["fp_l1"]),
             closed_only=True, use_l2_power=True, use_Q=True)
    print(f"built {out}: X {X.shape[1]}->{XQ.shape[1]} dims, {len(idx)} refs")

if __name__ == "__main__":
    main()
