"""Convert a downsampled training set of exact holes into kernel reference nodes for
SIMPLE_HOLE_KERNEL_FP -> atom/xc/data/kernel_fp_refs_n{N}.npz.

Each node is a (feature X, deviation DELTA) pair that `_build_fp_nodes` consumes:
  X     = [cn[1:], bounded(s^2)]   -- the kernel feature; s^2 BOUNDED via _bound to match what
          `_kernel_eps` queries with (hole_refs_full stores RAW s^2, so we re-bound here).
  DELTA = sigma_ref - sigma_LDA    -- the dimensionless hole-shape deviation added to rhotilde_lda.
          sigma_ref = rt / (-rho/2). NO orthogonalization: the functional re-pins charge/on-top, so
          at a node coeffs_bulk = -0.5 rho sigma_ref = rt (already moment-matched to eps_full) and the
          energy is reproduced exactly. (Orthogonalizing against [Bmom,R0] is a no-op for energy.)

The training set is already filtered to uncapped density (rho >= rho_floor, ~0.08), where sigma is
scale-free and DELTA is bounded and density-consistent across nodes -- below that the R_c cap breaks
scale-freeness and DELTA blows up / interpolates inconsistently (see build_training_sets.py).

Writes NAMED files (kernel_fp_refs_n16/64/256.npz); does NOT touch the canonical kernel_fp_refs.npz,
so the committed baseline stays reference-free. To activate one, point the module global
simple_hole_expansion._KERNEL_FP_REFS at it (the benchmark does this), or copy it over the canonical
name to make it the default.
"""
import argparse
import os
import sys

import numpy as np

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _bound  # noqa: E402
from cache.refs.loader import load_hole_refs_full, load_training_set  # noqa: E402

DATA = os.path.join(_REPO, "atom", "xc", "data")


def build(size, sig_lda, full, ts):
    idx = ts["order"][:size] if size <= len(ts["order"]) else ts["order"]
    X = full["X"][idx].copy()
    X[:, -1] = _bound(X[:, -1])[0]                       # bounded s^2 to match _kernel_eps
    rho = full["rho"][idx]
    DELTA = full["rt"][idx] / (-0.5 * rho)[:, None] - sig_lda[None, :]
    return idx, X, DELTA


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    full = load_hole_refs_full()
    ts = load_training_set()                             # full FPS ordering + provenance
    r = np.linspace(1e-3, 14.0, 400); w = np.gradient(r)
    sig_lda = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)._rhotilde_lda

    print(f"training pool: {len(ts['order'])} refs (leak<= {float(ts['leakage_cutoff']):.0%}, "
          f"rho>= {float(ts['rho_floor']):.4f}).  sigma_LDA norm {np.linalg.norm(sig_lda):.4f}")
    print(f"{'size':>5} {'nodes':>6} {'||DELTA|| med':>13} {'max':>7}  -> file")
    for size in args.sizes:
        idx, X, DELTA = build(size, sig_lda, full, ts)
        out = os.path.join(DATA, f"kernel_fp_refs_n{size}.npz")
        np.savez(out, X=X, DELTA=DELTA, idx=idx,
                 leakage_cutoff=ts["leakage_cutoff"], rho_floor=ts["rho_floor"],
                 fp_l0=ts["fp_l0"], fp_l1=ts["fp_l1"])
        nrm = np.linalg.norm(DELTA, axis=1)
        print(f"{size:>5} {len(idx):>6} {np.median(nrm):>13.4f} {nrm.max():>7.3f}  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
