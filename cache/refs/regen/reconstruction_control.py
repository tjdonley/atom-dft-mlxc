"""Decompose the closed-shell exchange error of SIMPLE_HOLE_KERNEL_FP into its sources, to answer:
is the ~88 mHa in-domain error a HOLE-RECONSTRUCTION error (n_out basis) or a KERNEL feature->hole
INTERPOLATION error?

Controls (closed-shell atoms, valence-only int(rho*eps_x) vs Ehf, mHa):
  (1) reconstruction      = atom_Emm vs atom_Ehf, stored in hole_refs_full. The exact moment-matched
                            hole rt, integrated directly. ~0.15 mHa -- the energy is ONE linear moment
                            of the hole, so the n_out=10 representation reproduces it exactly.
  (3a) exact-hole bulk     = the EXACT hole rt fed through the direct bulk integral (no kernel, no FA,
                            no projection). ~1-3 mHa -- confirms the functional COULD be near-exact.
  (3b) exact-hole + machinery = exact hole through the functional's FA blend + 2-constraint projection.
                            ~ -6..-1 for most, but Be +126 -- the FA blend (built for the tail/SIC)
                            POISONS the closed-shell bulk for Be.
  (D) kernel bulk (no FA)  = the KERNEL-reconstructed bulk hole, direct integral. 110-297 mHa -- the
                            feature->hole map is grossly off; the kernel hole violates the sum rule.
  (full) kernel + machinery= the full functional. ~88 mHa: the 2-constraint projection (sum rule +
                            on-top) rescues the kernel hole from 110-297 down to ~88; FA adjusts tail.

CONCLUSION: the reconstruction error is ~0.15 mHa (negligible). The 88 mHa is the kernel feature->hole
INTERPOLATION -- the SIMPLE features (l=0 monopole cn + l=1 s^2) under-determine the energy-relevant
hole shape (even after the projection pins charge + on-top, the Coulomb/energy moment is off ~88).
Plus the FA blend specifically hurts Be. Fix: more FEATURES (tau, higher multipoles l>=2) to determine
the hole; revisit the FA blend for closed-shell. See reports/hole_expansion/reconstruction_control.txt.
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

DATA = os.path.join(_REPO, "atom", "xc", "data")
CLOSED = os.path.join(DATA, "kernel_fp_refs_closed_n512_sat.npz")
ATOMS = [(2, "He"), (4, "Be"), (10, "Ne"), (12, "Mg"), (18, "Ar")]
MU, KLO = 0.2195, 0.804


def _load(Z):
    hf = load_hf(Z); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12)
    w = np.asarray(hf["w"])[o]
    return r, rho, w, float(hf["Ehf"])


def _eps_tail(F, rho, bulk, R_ad, cprime, fa_blend):
    """The functional's _kernel_eps energy tail given a bulk hole; fa_blend toggles FA + projection."""
    Bmom, R0, Cmom = F._Bmom, F._R0, F._Cmom
    if not fa_blend:
        return 2.0 * np.pi * R_ad ** 2 * (bulk @ Cmom)
    c_ad = F._c_ad(cprime, R_ad); d = c_ad / (4.0 * np.pi * R_ad ** 1.5)[:, None]
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


def main():
    f = load_hole_refs_full()
    Zf = np.asarray(f["atom_Z"]); off = np.asarray(f["atom_offset"]); npt = np.asarray(f["atom_npts"])
    Ehf_s = np.asarray(f["atom_Ehf"]); Emm_s = np.asarray(f["atom_Emm"])
    rtA = np.asarray(f["rt"]); r0A = np.asarray(f["r0"])
    print("closed-shell exchange error vs Ehf (mHa). PBE: He23 Be44 Ne78 Mg105 Ar62; rSCAN ~16.")
    print(f"{'atom':>4} {'(1)recon':>9} {'(3a)exactBulk':>13} {'(3b)exact+mach':>14} "
          f"{'(D)kernBulk':>12} {'(full)':>8}")
    for Z, sym in ATOMS:
        ai = int(np.where(Zf == Z)[0][0]); recon = 1e3 * (Emm_s[ai] - Ehf_s[ai])
        r, rho, w, Ehf = _load(Z)
        F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w, params=P(
            sat_gradient=True, fp_mu=MU, fp_kappa_lo=KLO, fp_l0=0.7, fp_l1=0.5,
            fp_ref_ridge=1e-8, refs_path=os.path.abspath(CLOSED)))
        ew = F.energy_weights; cp = np.array([op @ rho for op in F._ops]); g = F._grad_op @ rho
        R_ad, _ = F._R_ad(rho)
        # exact hole interpolated to the grid
        s, e = off[ai], off[ai] + npt[ai]; oo = np.argsort(r0A[s:e]); r0 = r0A[s:e][oo]; rt = rtA[s:e][oo]
        bulk_ex = np.array([np.interp(r, r0, rt[:, j]) for j in range(rt.shape[1])]).T
        # kernel-reconstructed hole
        kF = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0); s2, _ = _bound((g / (2.0 * kF * rho)) ** 2)
        c_ad = F._c_ad(cp, R_ad); cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
        rt_k = F._rhotilde_lda[None, :] + F._Kmat(F._xfeat(cn, s2), F._fp_Xnodes) @ F._fp_coef
        s2raw = (g / (2.0 * kF * rho)) ** 2; a = MU / KLO; h = a * s2raw / (1.0 + a * s2raw)
        A = KLO / (F._fp_kappa * F._fp_dgb); rt_k = rt_k + (A * h)[:, None] * F._dgea[None, :]
        bulk_k = (-0.5 * rho)[:, None] * rt_k
        E = lambda eps: 1e3 * (float(np.sum(ew * rho * eps)) - Ehf)
        print(f"{sym:>4} {recon:>9.1f} "
              f"{E(_eps_tail(F, rho, bulk_ex, R_ad, cp, False)):>13.0f} "
              f"{E(_eps_tail(F, rho, bulk_ex, R_ad, cp, True)):>14.0f} "
              f"{E(_eps_tail(F, rho, bulk_k, R_ad, cp, False)):>12.0f} "
              f"{E(_eps_tail(F, rho, bulk_k, R_ad, cp, True)):>8.0f}")


if __name__ == "__main__":
    main()
