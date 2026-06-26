"""Add the l=2 power channel (t^2) to closed-shell kernel references -> _l2.npz (use_l2=True).

The power-spectrum-l<=2 representation is [cn[1:] (l=0 vector), s^2 (l=1), t^2 (l=2)]. The functional's
_xfeat already produces this when use_l2=True; this script appends the t^2 column to existing
references (DELTA unchanged -- only the feature vector grows). The l=1 channel stays s^2 (the reduced
gradient), and the backbone HEG/GEA nodes have t^2=0, so the GEA limit is RETAINED (verified: backbone
GEA slope identical with/without l=2; refs perturb it ~5% with or without l=2).

t^2 = _bound((l2_op @ rho / (4 kF^2 rho))^2), matching _kernel_eps. Usage:
  python3 cache/refs/regen/build_refs_add_l2.py
"""
import os, sys
import numpy as np
np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
DATA = os.path.join(_REPO, "atom", "xc", "data")
f = load_hole_refs_full(); Zf = np.asarray(f["Z"]); r0f = np.asarray(f["r0"])
_t2cache = {}
def t2_for_atom(Z):
    if Z in _t2cache: return _t2cache[Z]
    hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); w = np.asarray(hf["w"])[o]
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w, params=P(use_l2=True))
    kF = (3 * np.pi ** 2 * rho) ** (1 / 3); t2, _ = _bound((F._l2_op @ rho / (4.0 * kF ** 2 * rho)) ** 2)
    _t2cache[Z] = (r, t2); return _t2cache[Z]
def add_l2(src, dst):
    z = np.load(os.path.join(DATA, src)); X = z["X"]; idx = z["idx"]
    Zr = Zf[idx]; r0r = r0f[idx]; t2col = np.zeros(len(idx))
    for Z in np.unique(Zr):
        r, t2 = t2_for_atom(int(Z)); m = Zr == Z; t2col[m] = np.interp(r0r[m], r, t2)
    out = {k: z[k] for k in z.files}; out["X"] = np.column_stack([X, t2col])
    np.savez(os.path.join(DATA, dst), **out)
    print(f"{src} -> {dst}: X {X.shape} -> {out['X'].shape}")
if __name__ == "__main__":
    for N in (64, 512): add_l2(f"kernel_fp_refs_closed_n{N}.npz", f"kernel_fp_refs_closed_n{N}_l2.npz")
