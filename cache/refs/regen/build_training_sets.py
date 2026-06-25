"""Filter + uniformly downsample the exact-hole reference pool into NESTED training sets.

Pipeline (see plan): start from cache/refs/holes/hole_refs_full.npz (840 pts x 14 atoms), then
  1. drop references whose leakage exceeds a cutoff (default 10%): high-leakage points are
     Fermi-Amaldi-owned tails where the windowed hole is mostly moment-match artifact -- shape
     contamination, [[simple-hole-references]];
  2. farthest-point sample (greedy k-center / Kennard-Stone) the survivors in the KERNEL's own
     distance, so coverage is uniform as the functional resolves the space.

Greedy FPS is NESTED: prefixes of one ordering are themselves uniform sets, so "several training
sets of increasing uniform density" = increasing-length prefixes of a single ordering.

Distance mirrors atom/xc/simple_hole_expansion.py:_Kmat with TUNABLE length scales (default to the
kernel's own l0=0.5 and the LO-calibrated l1~=10.585; reduce --l1 to weight the gradient s^2 more):
    K(a,b) = exp(-0.5[ sum_{l=0}(d cn)^2/l0^2 + (d s^2_bounded)^2/l1^2 ]);  D = sqrt(max(1-K,0)).
s^2 is BOUNDED via _bound (as _kernel_eps does) so the heavy-tailed raw s^2 cannot hijack FPS.

Emits cache/refs/holes/training_sets.npz: the FPS ordering (global indices into hole_refs_full),
the nested size schedule, the leakage keep-mask, provenance, and a fill-distance coverage curve.
Converting rt -> kernel DELTA and writing kernel_fp_refs.npz is a SEPARATE later step.
"""
import argparse
import os
import sys

import numpy as np

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, _bound, _INV_BOUND  # noqa: E402
from atom.utils.periodic import atomic_number_to_name as nm  # noqa: E402
from cache.refs.loader import load_hole_refs_full  # noqa: E402

OUT = os.path.join(_HERE, "..", "holes", "training_sets.npz")
SIZES = [16, 32, 64, 128, 256, 512]
N_OUT = 10


def bounded_s2(s2):
    """Saturated reduced-gradient feature, identical to what _kernel_eps feeds the kernel."""
    return _bound(np.asarray(s2, float))[0]


def to_kernel_feat(X):
    """Replace the raw s^2 column (X[:,-1]) with its bounded value; l=0 block untouched."""
    Xk = X.copy()
    Xk[:, -1] = bounded_s2(X[:, -1])
    return Xk


def kmat(Xa, Xb, l0, l1, n_out=N_OUT):
    """Anisotropic squared-exponential kernel == simple_hole_expansion._Kmat (pure numpy).
    Assumes the s^2 column is ALREADY bounded (use to_kernel_feat first)."""
    nl0 = n_out - 1
    d0 = np.sum((Xa[:, None, :nl0] - Xb[None, :, :nl0]) ** 2, axis=2) / l0 ** 2
    d1 = (Xa[:, None, nl0] - Xb[None, :, nl0]) ** 2 / l1 ** 2
    return np.exp(-0.5 * (d0 + d1))


def kernel_dist(Xa, Xb, l0, l1):
    """D = sqrt(max(1-K,0)) in [0,1]; bounded + saturating in every direction."""
    return np.sqrt(np.maximum(1.0 - kmat(Xa, Xb, l0, l1), 0.0))


def uncapping_density(refs):
    """Density below which R_ad = X/k_F hits the R_c cap: rho_uncap = (X/R_c)^3 / (3 pi^2).
    Below it the scale-free identity R_ad^3 rho = const breaks, so sigma = hole/(-rho/2) is no
    longer density-consistent and a reference's DELTA is not transferable across the kernel."""
    X = float(refs["X_window"]); Rc = float(refs["R_c"])
    return (X / Rc) ** 3 / (3.0 * np.pi ** 2)


def valid_filter(refs, cutoff=0.10, mode="max", rho_floor=None):
    """Return (keep_mask[Npts], kept_global_idx[M]). A reference is VALID for the kernel iff
       (i) leakage <= cutoff (hole captured in the window, not moment-match artifact), AND
       (ii) rho >= rho_floor so R_ad is uncapped and sigma = hole/(-rho/2) is scale-free.
    rho_floor=None uses 0.1 -- safely above the uncapping density (~0.08 at X=8, R_c=6)."""
    lq = np.abs(refs["leakQ"]); le = np.abs(refs["leakE"])
    leak = {"max": np.maximum(lq, le), "leakQ": lq, "leakE": le}[mode]
    if rho_floor is None:
        rho_floor = max(0.1, uncapping_density(refs))
    keep = (leak <= cutoff) & (refs["rho"] >= rho_floor)
    return keep, np.where(keep)[0], rho_floor


def seed_index(Xk, l0, l1, heg_feat):
    """Kept-pool index nearest the HEG node [cnH[1:], s^2=0] -- the kernel's LDA anchor.
    Deterministic; argmin returns the lowest index on ties."""
    D = kernel_dist(Xk, heg_feat[None, :], l0, l1)[:, 0]
    return int(np.argmin(D))


def fps_order(D, seed):
    """Greedy k-center (farthest-point) ordering from a precomputed M x M distance matrix.
    Returns a length-M permutation of local indices; deterministic (argmax -> lowest index)."""
    M = D.shape[0]
    sel = [seed]; mind = D[seed].copy()
    for _ in range(M - 1):
        mind[sel] = -1.0                       # never reselect
        j = int(np.argmax(mind))               # ties -> lowest index
        sel.append(j); mind = np.minimum(mind, D[j])
    return np.array(sel, dtype=int)


def coverage(D, order, sizes):
    """Per-size fill distance f(k)=max_i min_{j in prefix} D[i,j] and min in-set separation."""
    fill = np.zeros(len(sizes)); sep = np.zeros(len(sizes))
    for s, k in enumerate(sizes):
        pre = order[:k]
        fill[s] = float(D[:, pre].min(axis=1).max())
        sub = D[np.ix_(pre, pre)].copy(); np.fill_diagonal(sub, np.inf)
        sep[s] = float(sub.min())
    return fill, sep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leakage-cutoff", type=float, default=0.10)
    ap.add_argument("--mode", choices=["max", "leakQ", "leakE"], default="max")
    ap.add_argument("--rho-floor", type=float, default=None,
                    help="min density (default: uncapping density ~0.08 where R_ad uncaps)")
    ap.add_argument("--l0", type=float, default=None, help="l=0 RBF width (default: kernel _fp_l0)")
    ap.add_argument("--l1", type=float, default=None, help="l=1/s^2 RBF width (default: LO-calibrated _fp_l1)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    refs = load_hole_refs_full()
    X = refs["X"]; Z = refs["Z"]; Npts = len(X)

    # default length scales = the kernel's own (l0=0.5, l1 = LO-calibrated _fp_l1)
    r = np.linspace(1e-3, 14.0, 400); w = np.gradient(r)
    K = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w)
    l0 = args.l0 if args.l0 is not None else float(K._fp_l0)
    l1 = args.l1 if args.l1 is not None else float(K._fp_l1)
    heg_feat = np.concatenate([K._cnH[1:], [0.0]])     # HEG node [cnH[1:], s^2=0]

    keep, kept, rho_floor = valid_filter(refs, args.leakage_cutoff, args.mode, args.rho_floor)
    M = len(kept)
    Xk = to_kernel_feat(X[kept])
    D = np.empty((M, M))                                # blocked build (avoid the (M,M,n_out) tmp)
    for s in range(0, M, 512):
        D[s:s + 512] = kernel_dist(Xk[s:s + 512], Xk, l0, l1)
    seed_local = seed_index(Xk, l0, l1, heg_feat)
    perm = fps_order(D, seed_local)                    # local (into kept pool)
    order = kept[perm]                                 # GLOBAL indices into hole_refs_full
    seed_global = int(kept[seed_local])

    sizes = [k for k in SIZES if k < M] + [M]
    fill, sep = coverage(D, perm, sizes)

    print(f"refs: {Npts} pts; leakage {args.mode}<= {args.leakage_cutoff:.0%} AND rho >= "
          f"{rho_floor:.4f} (uncapped) -> {M} kept (dropped {Npts - M}).  "
          f"metric l0={l0:.4f} l1={l1:.4f} (s^2 bounded to [0,{_INV_BOUND:.0f}))")
    print(f"seed = global idx {seed_global} (Z={int(Z[seed_global])} {nm(int(Z[seed_global]))}, "
          f"nearest HEG node)\n")
    print(f"{'size':>6} {'fill_dist':>10} {'min_sep':>9}  per-atom counts")
    for s, k in enumerate(sizes):
        cnt = np.bincount(Z[order[:k]].astype(int)); present = np.nonzero(cnt)[0]
        spread = " ".join(f"{nm(int(z))}:{cnt[z]}" for z in present)
        print(f"{k:>6} {fill[s]:>10.4f} {sep[s]:>9.4f}  {spread}")

    np.savez(os.path.abspath(OUT),
             order=order, sizes=np.array(sizes, dtype=int), keep_mask=keep,
             seed=seed_global, leakage_cutoff=float(args.leakage_cutoff),
             leakage_mode=args.mode, rho_floor=float(rho_floor), fp_l0=l0, fp_l1=l1, fill_dist=fill)
    print(f"\nwrote {os.path.abspath(OUT)}: FPS ordering over {M} kept refs, "
          f"nested sizes {sizes}")

    if args.plot:
        _plot(Xk, perm, sizes, refs, kept)


def _plot(Xk, perm, sizes, refs, kept):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Xc = Xk[:, :N_OUT - 1] - Xk[:, :N_OUT - 1].mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T                                  # 2-comp PCA of the l=0 block
    s2b = Xk[:, -1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, k in zip(axes, [64, 128]):
        k = min(k, len(perm))
        ax.scatter(P[:, 0], P[:, 1], c=s2b, s=8, cmap="viridis", alpha=0.4, lw=0)
        sub = perm[:k]
        ax.scatter(P[sub, 0], P[sub, 1], s=28, facecolors="none", edgecolors="crimson", lw=1.0)
        ax.set_title(f"FPS size {k} (of {len(perm)})"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Training-set coverage: l=0 PCA (color = bounded s^2)")
    fig.tight_layout()
    out = os.path.join(_REPO, "validation_figures", "training_set_coverage.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
