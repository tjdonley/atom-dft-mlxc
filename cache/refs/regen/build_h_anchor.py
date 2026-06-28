"""Add a ONE-ELECTRON (H) anchor to the production closed-shell references [[simple-hole-kernel-map]].

With the Fermi-Amaldi blend removed (fa_ontop/fa_coeff=False) and the enclosed charge Q a learned
kernel coordinate, the references span only the closed-atom local-Q band (bounded Q ~1.3-3.3). H has
one electron -> local bounded-Q ~0.8, BELOW that band, so H is a pure extrapolation and its exchange is
badly off (-53 mHa vs the HF reference). The clean fix (analogous to the HEG and GEA backbone anchors)
is to add the one-electron limit as its own fixed-point node cluster: the exact RESTRICTED H exchange
hole (occ 0.5/0.5 per spin -- the convention that matches the project's HF reference, on-top -rho/2;
eps integrates to -0.128 = Ehf), moment-matched and embedded at H's features. Because the energy is the
single pinned Coulomb moment of the hole, anchoring H's eps_full makes its exchange APPROXIMATELY correct
(windowing-limited ~8 mHa, in-family with every other atom -- NOT exact). The anchor is LOCALIZED at
Q~0.8, far from many-electron cores (Q>=2) at fp_lQ=0.3, so closed shells are left unchanged -- unlike the
global FA blend that was removed as spurious.

Reads the production refs (rf001_l2power_Q_gf06), appends ~40 H nodes (idx=-1 marks them), writes
kernel_fp_refs_closed_rf001_l2power_Q_gf06_Hanchor.npz -- the production reference set.
Usage: python3 cache/refs/regen/build_h_anchor.py
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import (  # noqa: E402
    SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound)
from atom.xc import orbital_hole as oh  # noqa: E402
from atom.descriptors.simple.bessel import RadialBesselBasis  # noqa: E402
from cache.refs.loader import load_hf  # noqa: E402

DATA = os.path.join(_REPO, "atom", "xc", "data")
SRC = "kernel_fp_refs_closed_rf001_l2power_Q_gf06.npz"
DST = "kernel_fp_refs_closed_rf001_l2power_Q_gf06_Hanchor.npz"
NOUT = 10
N_NODES = 40            # H anchor nodes spanning H's significant-density radial shells


def build_h_nodes():
    """Exact restricted one-electron (H) hole -> (X features, DELTA) at H's grid points, using the SAME
    machinery and grad_filter=0.6 operator as the production functional so node/query coordinates match."""
    tu, wu = leggauss(120); t = 0.5 * (tu + 1.0); wt = 0.5 * wu
    Rb1 = RadialBesselBasis(NOUT - 1, 0, 1.0).evaluate(0, t)
    R0u = (np.arange(NOUT) + 1) * np.pi * np.sqrt(2.0)
    hf = load_hf(1); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); w = np.asarray(hf["w"])[o]
    rs = np.asarray(hf["r_sorted"]); gs = np.asarray(hf["g_sorted"]); lval = np.asarray(hf["l_values"])
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w, params=P(
        fp_l0=0.7, fp_l1=0.5, fp_l2pow=0.02, fp_lQ=0.3, fp_ref_ridge=1e-8,
        refs_path=os.path.join(DATA, SRC), use_l2_power=True, use_Q=True,
        fa_ontop=False, fa_coeff=False, grad_filter=0.6))
    # features at every grid point, assembled exactly as _kernel_eps does
    cprime = np.array([op @ rho for op in F._ops]); R_ad, _ = F._R_ad(rho); c_ad = F._c_ad(cprime, R_ad)
    d = c_ad / (4 * np.pi * R_ad ** 1.5)[:, None]; Q = 4 * np.pi * R_ad ** 3 * (d @ F._Bmom)
    kF = (3 * np.pi ** 2 * rho) ** (1 / 3); s2, _ = _bound((F._grad_op @ rho / (2 * kF * rho)) ** 2)
    cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
    p2 = F._l2_power_feat(rho, R_ad, c_ad[:, 0]); qf = _bound(Q)[0]
    X_all = F._xfeat(cn, s2, None, p2, qf)
    al = F._G[0]; be = F._H[0]; sig_lda = F._rhotilde_lda
    idx = np.where(rho > 1e-2 * rho.max())[0]
    sel = idx[np.linspace(0, len(idx) - 1, min(N_NODES, len(idx))).astype(int)]
    rt = np.zeros((len(sel), NOUT)); epsf = np.zeros(len(sel))
    for k, i in enumerate(sel):
        u = R_ad[i] * t
        nx = oh.exchange_hole_spin(float(r[i]), u, rs, gs, np.array([0.5]), np.array([0.5]), lval, n_mu=64)
        rt0 = Rb1 @ (nx * t ** 2 * wt)
        epsf[k] = oh.exact_eps_x_l_spin(float(r[i]), rs, gs, np.array([0.5]), np.array([0.5]), lval, n_u=128, n_mu=64)
        a_row = 4 * np.pi * R_ad[i] ** 3 * al; e_row = 2 * np.pi * R_ad[i] ** 2 * be
        A3 = np.stack([a_row, R0u, e_row]); tgt = np.array([-1.0, -0.5 * rho[i], epsf[k]])
        lam = np.linalg.solve(A3 @ A3.T, tgt - A3 @ rt0); rt[k] = rt0 + A3.T @ lam
    DELTA = rt / (-0.5 * rho[sel])[:, None] - sig_lda[None, :]
    eps_mm = 2 * np.pi * R_ad[sel] ** 2 * (rt @ be); ew = 4 * np.pi * r ** 2 * w
    Emm = float(np.sum(ew * rho * np.interp(r, r[sel], eps_mm)))
    return X_all[sel], DELTA, Emm, (X_all[sel][:, 11].min(), X_all[sel][:, 11].max())


def main():
    z = np.load(os.path.join(DATA, SRC))
    XH, DH, Emm, qrange = build_h_nodes()
    print(f"  H anchor: {len(XH)} nodes, bounded-Q [{qrange[0]:.2f},{qrange[1]:.2f}], "
          f"Emm={Emm:.5f} (Ehf=-0.12820, ~{1e3*(Emm+0.12820):+.1f} mHa -- windowing floor)")
    Xaug = np.vstack([z["X"], XH]); Daug = np.vstack([z["DELTA"], DH])
    idxaug = np.concatenate([z["idx"], -np.ones(len(XH), dtype=z["idx"].dtype)])  # -1 = H anchor node
    out = os.path.join(DATA, DST)
    np.savez(out, X=Xaug, DELTA=Daug, idx=idxaug,
             closed_only=True, use_l2_power=True, use_Q=True, h_anchor=True)
    print(f"built {out}: {z['X'].shape[0]} closed + {len(XH)} H-anchor = {Xaug.shape[0]} nodes, "
          f"{Xaug.shape[1]} dims")


if __name__ == "__main__":
    main()
