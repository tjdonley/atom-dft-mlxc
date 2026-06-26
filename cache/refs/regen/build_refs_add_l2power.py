"""Add the FULL l=2 POWER SPECTRUM channel p2 = sum_n d_{n,2}^2 (use_l2_power=True) to the closed-shell
rf001 kernel references -> rf001_l2power.npz. The functional's _l2_power_feat produces p2 at eval time;
this script precomputes the matching p2 column for each reference point (per-atom adaptive-radius l=2
power, monopole-normalized, bounded) so the kernel sees the same coordinate at nodes and queries.

The cross-atom-completeness coordinate: the single reduced-l2 scalar t^2 (use_l2) fixed Be's potential
(OEP 14->4.5) but not Mg's; the feature-space floor analysis says the full l=2 power spectrum should
resolve all closed-shell environments. Usage: python3 cache/refs/regen/build_refs_add_l2power.py
"""
import os, sys
import numpy as np
np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from cache.refs.loader import load_hf, load_hole_refs_full
DATA = os.path.join(_REPO, "atom", "xc", "data")

def p2_for_atom(Z):
    hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); w = np.asarray(hf["w"])[o]
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w, params=P(use_l2_power=True))
    cprime = np.array([op @ rho for op in F._ops]); R_ad, _ = F._R_ad(rho); c_ad = F._c_ad(cprime, R_ad)
    return r, F._l2_power_feat(rho, R_ad, c_ad[:, 0])

def main(src="kernel_fp_refs_closed_rf001.npz", dst="kernel_fp_refs_closed_rf001_l2power.npz"):
    z = np.load(os.path.join(DATA, src)); X = z["X"]; DELTA = z["DELTA"]; idx = z["idx"]
    pool = load_hole_refs_full(); Zr = pool["Z"][idx]; r0r = pool["r0"][idx]
    cache = {}; p2 = np.zeros(len(idx))
    for Z in sorted(set(Zr.tolist())):
        if Z not in cache: cache[Z] = p2_for_atom(Z)
        m = Zr == Z; rA, p2A = cache[Z]; p2[m] = np.interp(r0r[m], rA, p2A)
        print(f"  Z={Z:2d}: {int(m.sum())} refs, p2 range [{p2[m].min():.3f},{p2[m].max():.3f}]", flush=True)
    Xp2 = np.hstack([X, p2[:, None]])
    out = os.path.join(DATA, dst)
    np.savez(out, X=Xp2, DELTA=DELTA, idx=idx, fp_l0=float(z["fp_l0"]), fp_l1=float(z["fp_l1"]),
             closed_only=True, use_l2_power=True)
    print(f"built {out}: X {X.shape[1]}->{Xp2.shape[1]} dims, {len(idx)} refs")

if __name__ == "__main__":
    main()
