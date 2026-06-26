"""Pin the best CLOSED-SHELL-ONLY referenced functional and save it for benchmarking.

Builds kernel reference nodes from the closed-subshell points only (the 14 exact atoms), at the
balanced-optimum settings found by tuning: N=512 farthest-point refs, isotropic widths l0=0.7, l1=0.5
(non-SCF benchmark: in 64 / out 50 mHa MAE vs HF). Open-shell points are excluded here on purpose --
this is the reference functional whose transfer we benchmark against the spin-resolved version later.

Writes atom/xc/data/kernel_fp_refs_closed_n{N}.npz (X, DELTA + provenance incl fp_l0/fp_l1).
Instantiate the functional cleanly via SIMPLEHOLEKERNELFPParameters(fp_l0=0.7, fp_l1=0.5,
refs_path=<that file>) -- no globals, baseline stays reference-free.
"""
import argparse, os, sys
import numpy as np
np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _bound
from cache.refs.loader import load_hole_refs_full
from cache.refs.regen.build_training_sets import valid_filter, to_kernel_feat, kernel_dist, seed_index, fps_order
DATA = os.path.join(_REPO, "atom", "xc", "data")
FP_L0, FP_L1 = 0.7, 0.5            # balanced-optimum widths (pinned for the saved functional)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=512); a = ap.parse_args()
    refs = load_hole_refs_full()
    keep, kept, rho_floor = valid_filter(refs)                      # leak<=10% & rho>=0.1
    closed_kept = kept[refs["closed"][kept]]                        # closed-subshell subset
    Xk = to_kernel_feat(refs["X"][closed_kept])
    r = np.linspace(1e-3, 14.0, 400); w = np.gradient(r)
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)
    l0m, l1m = float(F._fp_l0), float(F._fp_l1)                     # FPS coverage metric (default 0.5/0.5)
    heg = np.concatenate([F._cnH[1:], [0.0]]); sig_lda = F._rhotilde_lda
    D = kernel_dist(Xk, Xk, l0m, l1m)
    order = closed_kept[fps_order(D, seed_index(Xk, l0m, l1m, heg))]  # GLOBAL idx, FPS order
    N = min(a.N, len(order)); idx = order[:N]
    X = refs["X"][idx].copy(); X[:, -1] = _bound(X[:, -1])[0]
    DELTA = refs["rt"][idx] / (-0.5 * refs["rho"][idx])[:, None] - sig_lda[None, :]
    out = os.path.join(DATA, f"kernel_fp_refs_closed_n{N}.npz")
    np.savez(out, X=X, DELTA=DELTA, idx=idx, fp_l0=FP_L0, fp_l1=FP_L1,
             closed_only=True, leakage_cutoff=0.10, rho_floor=float(rho_floor))
    nrm = np.linalg.norm(DELTA, axis=1)
    import collections
    cnt = collections.Counter(refs["sym"][refs["Z"][idx].argsort()*0 + 0]) if False else None
    atoms = sorted(set(refs["Z"][idx].tolist()))
    print(f"closed-valid pool: {len(closed_kept)} pts; took N={N} FPS refs from {len(atoms)} atoms")
    print(f"  ||DELTA|| median {np.median(nrm):.4f} max {nrm.max():.4f}")
    print(f"  saved {out}")
    print(f"  -> SIMPLEHOLEKERNELFPParameters(fp_l0={FP_L0}, fp_l1={FP_L1}, refs_path='{out}')")


if __name__ == "__main__":
    main()
