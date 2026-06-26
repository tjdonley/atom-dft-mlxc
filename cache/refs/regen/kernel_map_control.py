"""Isolate the kernel feature->hole MAP error, then add the machinery back layer by layer.

For each atom we use ALL of its own exact-hole reference points as kernel nodes (dense, in-domain),
fit the kernel map, and evaluate the exchange energy on the FULL radial grid. We peel the functional
into layers so we can see exactly where the error enters:

  L0  kernel-only      sigma = K c           (target = full hole shape sigma_ref; NO LDA/GEA/FA/proj)
  L1  + LDA            sigma = sigma_LDA + K c                (target = sigma_ref - sigma_LDA)
  L2  + GEA (sat)      sigma = sigma_LDA + backbone + K c     (target = sigma_ref - sigma_LDA - backbone)
  L3  + FA + proj      L2 hole through the FA blend + 2-constraint (sum-rule/on-top) projection (= full)

Energy is the direct hole integral eps_x = 2 pi R_ad^2 (-rho/2 sigma . Cmom); error vs Ehf in mHa.

The FEATURE SET is pluggable (feats=...) so we can ask whether richer rotational invariants
(power spectrum at higher l, the CG bispectrum) shrink the L0/L1 kernel-map error -- the quantity that
the reconstruction control (reconstruction_control.py) pinned as the real ~88 mHa source. Baseline
feats = the functional's current [cn[1:] (l=0 monopole power vec), bounded s^2 (l=1 norm)].
"""
import os
import sys

import numpy as np

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import (  # noqa: E402
    SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, enclosed_charge_switch, _bound)
from cache.refs.loader import load_hf, load_hole_refs_full  # noqa: E402

ATOMS = [(2, "He"), (4, "Be"), (10, "Ne"), (12, "Mg"), (18, "Ar")]
MU, KLO, RIDGE = 0.2195, 0.804, 1e-10


def baseline_feats(F, cn, s2_bounded, s2_raw, rho, g, kF):
    """Current functional features: 9 l=0 monopole power-vector dims + 1 bounded-s^2 (l=1 norm).
    Returns (X (N,nf), inv_ell (nf,))."""
    X = np.column_stack([cn[:, 1:], s2_bounded])
    inv_ell = np.concatenate([np.full(cn.shape[1] - 1, 1.0 / 0.7), [1.0 / 0.5]])
    return X, inv_ell


def kmat(A, B, inv_ell):
    Aw = A * inv_ell[None, :]; Bw = B * inv_ell[None, :]
    d2 = (np.sum(Aw * Aw, 1)[:, None] + np.sum(Bw * Bw, 1)[None, :] - 2.0 * Aw @ Bw.T)
    return np.exp(-0.5 * np.maximum(d2, 0.0))


def grid_quantities(F, rho, g):
    """Per-grid-point quantities the layers need."""
    R_ad, _ = F._R_ad(rho); kF = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    cprime = np.array([op @ rho for op in F._ops]); c_ad = F._c_ad(cprime, R_ad)
    cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
    s2_raw = (g / (2.0 * kF * rho)) ** 2; s2_b, _ = _bound(s2_raw)
    a = MU / KLO; h = a * s2_raw / (1.0 + a * s2_raw)
    A_max = KLO / (F._fp_kappa * F._fp_dgb)
    backbone = (A_max * h)[:, None] * F._dgea[None, :]            # sat-GEA deviation, per point
    return R_ad, kF, cn, s2_b, s2_raw, c_ad, cprime, backbone


def energy_from_sigma(F, rho, sigma, R_ad, c_ad, fa_proj):
    bulk = (-0.5 * rho)[:, None] * sigma
    if not fa_proj:
        return 2.0 * np.pi * R_ad ** 2 * (bulk @ F._Cmom)
    Bmom, R0, Cmom = F._Bmom, F._R0, F._Cmom
    d = c_ad / (4.0 * np.pi * R_ad ** 1.5)[:, None]
    Q = 4.0 * np.pi * R_ad ** 3 * (d @ Bmom); Qs = np.maximum(Q, 1e-12); W = enclosed_charge_switch(0.5 * Q)
    fa = -d / Qs[:, None]; coeffs = (1.0 - W)[:, None] * bulk + W[:, None] * fa
    ontop = (1.0 - W) * (-0.5 * rho) + W * (-rho / Qs)
    a_row = 4.0 * np.pi * (R_ad ** 3)[:, None] * Bmom[None, :]
    row0 = np.sum(a_row * coeffs, 1); row1 = coeffs @ R0
    g00 = np.sum(a_row * a_row, 1); g01 = a_row @ R0; g11 = float(R0 @ R0)
    res0 = -1.0 - row0; res1 = ontop - row1; det = g00 * g11 - g01 ** 2
    lam0 = (g11 * res0 - g01 * res1) / det; lam1 = (-g01 * res0 + g00 * res1) / det
    coeffs = coeffs + lam0[:, None] * a_row + lam1[:, None] * R0[None, :]
    return 2.0 * np.pi * R_ad ** 2 * (coeffs @ Cmom)


def run_atom(Z, ref, feats):
    hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12)
    w = np.asarray(hf["w"])[o]; Ehf = float(hf["Ehf"])
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w,
                              params=P(sat_gradient=True, fp_mu=MU, fp_kappa_lo=KLO))
    ew = F.energy_weights; g = F._grad_op @ rho
    R_ad, kF, cn, s2_b, s2_raw, c_ad, cprime, backbone = grid_quantities(F, rho, g)
    sig_lda = F._rhotilde_lda
    # node = the atom's stored ref points, matched onto the full grid by radius
    sel = np.array([int(np.argmin(np.abs(r - r0))) for r0 in np.sort(ref["r0"])])
    rt = ref["rt"][np.argsort(ref["r0"])]
    sigma_ref_nodes = rt / (-0.5 * rho[sel])[:, None]
    Xall, inv_ell = feats(F, cn, s2_b, s2_raw, rho, g, kF)
    Xn = Xall[sel]
    Knn = kmat(Xn, Xn, inv_ell) + RIDGE * np.eye(len(sel))
    Kqn = kmat(Xall, Xn, inv_ell)
    def predict(target_nodes, base_grid):
        coef = np.linalg.solve(Knn, target_nodes)
        return base_grid + Kqn @ coef
    zero = np.zeros_like(sig_lda)[None, :]
    out = {}
    # L0 kernel-only
    sig = predict(sigma_ref_nodes, 0.0)
    out["L0_kernel"] = 1e3 * (float(np.sum(ew * rho * energy_from_sigma(F, rho, sig, R_ad, c_ad, False))) - Ehf)
    # L1 + LDA
    sig = predict(sigma_ref_nodes - sig_lda[None, :], sig_lda[None, :])
    out["L1_+LDA"] = 1e3 * (float(np.sum(ew * rho * energy_from_sigma(F, rho, sig, R_ad, c_ad, False))) - Ehf)
    # L2 + GEA(sat)
    base2 = sig_lda[None, :] + backbone
    sig2 = predict(sigma_ref_nodes - base2[sel], base2)
    out["L2_+GEA"] = 1e3 * (float(np.sum(ew * rho * energy_from_sigma(F, rho, sig2, R_ad, c_ad, False))) - Ehf)
    # L3 + FA + projection (full)
    out["L3_+FA/proj"] = 1e3 * (float(np.sum(ew * rho * energy_from_sigma(F, rho, sig2, R_ad, c_ad, True))) - Ehf)
    return out


def main(feats=baseline_feats, label="baseline [cn[1:], s^2]"):
    f = load_hole_refs_full(); Zf = np.asarray(f["atom_Z"]); off = np.asarray(f["atom_offset"]); npt = np.asarray(f["atom_npts"])
    print(f"kernel-map control, feats = {label}. ALL own-atom refs, exact interp. PBE: He23 Be44 Ne78 Mg105 Ar62.")
    print(f"{'atom':>4} {'Nref':>5} {'L0 kernel':>10} {'L1 +LDA':>9} {'L2 +GEA':>9} {'L3 +FA/proj':>12}")
    rows = []
    for Z, sym in ATOMS:
        ai = int(np.where(Zf == Z)[0][0]); s, e = off[ai], off[ai] + npt[ai]
        ref = {"r0": np.asarray(f["r0"])[s:e], "rt": np.asarray(f["rt"])[s:e]}
        o = run_atom(Z, ref, feats); rows.append(o)
        print(f"{sym:>4} {e - s:>5} {o['L0_kernel']:>10.0f} {o['L1_+LDA']:>9.0f} {o['L2_+GEA']:>9.0f} {o['L3_+FA/proj']:>12.0f}")
    mae = {k: np.mean([abs(rr[k]) for rr in rows]) for k in rows[0]}
    print("MAE  " + "  ".join(f"{k} {v:.0f}" for k, v in mae.items()))
    return rows


if __name__ == "__main__":
    main()
