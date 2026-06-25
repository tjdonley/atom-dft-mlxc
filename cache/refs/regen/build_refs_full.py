"""Build the FULL exact-hole reference set for the kernel-mapped fixed-point functional
[[simple-hole-kernel-map]], stored as a flat (atom x r0) table for later down-sampling to a
training set that evenly spans the SIMPLE feature subspace.

For every atom with cached HF orbitals whose occupied subshells are ALL FULL
(occ_i == 2(2 l_i+1)), the spin-restricted addition-theorem hole `orbital_hole.exchange_hole`
is EXACT. Open-(sub)shell atoms need the spin-resolved 1-RDM (not yet implemented) and are
excluded here -- a reference must be an exact point on the universal manifold, not an
approximation.

Per reference point r0 we:
  1. project the exact spherically-averaged hole n_x(r0,u) onto the adaptive unit frame
     u = R_ad t, t in [0,1], R_ad = min(X/k_F, R_c=6)  (scale-free; [[simple-always-scale-free]]);
  2. moment-match the windowed coefficients to charge=-1, on-top=-rho/2, Coulomb=exact eps_x
     (only the low-order/long-range modes move) -> rt;
  3. record LEAKAGE = the charge and energy of the exact hole lying beyond the window
     (diagnostic of where the hard R_c=6 cap bites; for future adaptive-R_c / tail work).

Writes cache/refs/holes/hole_refs_full.npz. Hard cutoff R_c=6, n_out=10, X=8.
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss

np.seterr(all="ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION, SIMPLEHOLEEXPParameters
from atom.xc import orbital_hole as oh
from atom.descriptors.simple.bessel import RadialBesselBasis
from atom.descriptors.simple.derivatives import build_spectral_gradient_operator
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf, available_hf

OUT = os.path.join(_HERE, "..", "holes", "hole_refs_full.npz")
NOUT, NPTS, RHO_FLOOR_FRAC = 10, 150, 1e-5   # 150 r0/atom -> ample valid (rho>=0.1) pool for N=512
tu, wu = leggauss(120); t = 0.5 * (tu + 1.0); wt = 0.5 * wu
Rb1 = RadialBesselBasis(NOUT - 1, 0, 1.0).evaluate(0, t)
R0u = (np.arange(NOUT) + 1) * np.pi * np.sqrt(2.0)


def is_exact_atom(occ, lval):
    """Restricted addition-theorem hole is exact iff every occupied subshell is full."""
    if occ.size == 0:
        return False
    m = occ > 1e-8
    return bool(np.all(np.isclose(occ[m], 2.0 * (2.0 * lval[m] + 1.0))))


def build_atom(Z):
    hf = load_hf(Z)
    r = np.asarray(hf["r"]); rho = np.maximum(np.asarray(hf["rho"]), 1e-12); w = np.asarray(hf["w"])
    o = np.argsort(r); r, rho, w = r[o], rho[o], w[o]; ew = 4 * np.pi * r ** 2 * w
    occ = np.asarray(hf["occ"]); lval = np.asarray(hf["l_values"])
    if not bool(hf["converged"]) or np.asarray(hf["g_sorted"]).size == 0 or not is_exact_atom(occ, lval):
        return None
    Ehf = float(hf["Ehf"]); rs = np.asarray(hf["r_sorted"]); gs = np.asarray(hf["g_sorted"])
    F = SIMPLE_HOLE_EXPANSION(r_quad=r, quadrature_weights=w, params=SIMPLEHOLEEXPParameters())
    C = np.array([op @ rho for op in F._ops]); Rad, _ = F._R_ad(rho)
    c_ad = F._c_ad(C, Rad); cn = c_ad / np.where(np.abs(c_ad[:, :1]) > 1e-30, c_ad[:, :1], 1e-30)
    al = F._G[0]; be = F._H[0]
    d = c_ad / (4 * np.pi * Rad[:, None] ** 1.5); Q = 4 * np.pi * Rad ** 3 * (d @ al)
    gop = build_spectral_gradient_operator(F._r_grid); kF = (3 * np.pi ** 2 * rho) ** (1 / 3)
    s = np.abs(gop @ rho) / (2 * kF * rho)
    idx = np.where(rho > RHO_FLOOR_FRAC * rho.max())[0]
    sel = idx[np.linspace(0, len(idx) - 1, min(NPTS, len(idx))).astype(int)]
    rt = np.zeros((len(sel), NOUT)); leakQ = np.zeros(len(sel)); leakE = np.zeros(len(sel))
    eps_win = np.zeros(len(sel)); eps_full = np.zeros(len(sel))
    for k, i in enumerate(sel):
        u = Rad[i] * t
        nx = oh.exchange_hole(float(r[i]), u, rs, gs, occ, lval, n_mu=64)
        rt0 = Rb1 @ (nx * t ** 2 * wt)
        Qwin = 4 * np.pi * Rad[i] ** 3 * np.sum(wt * nx * t ** 2)   # windowed hole charge
        eps_win[k] = 2 * np.pi * Rad[i] ** 2 * np.sum(wt * nx * t)   # windowed energy density
        eps_full[k] = oh.exact_eps_x_l(float(r[i]), rs, gs, occ, lval, n_u=128, n_mu=64)
        leakQ[k] = 1.0 + Qwin                                        # charge beyond window (total = -1)
        leakE[k] = (eps_full[k] - eps_win[k]) / eps_full[k] if eps_full[k] != 0 else 0.0
        a_row = 4 * np.pi * Rad[i] ** 3 * al; e_row = 2 * np.pi * Rad[i] ** 2 * be
        A3 = np.stack([a_row, R0u, e_row]); tgt = np.array([-1.0, -0.5 * rho[i], eps_full[k]])
        lam = np.linalg.solve(A3 @ A3.T, tgt - A3 @ rt0); rt[k] = rt0 + A3.T @ lam
    eps_mm = 2 * np.pi * Rad[sel] ** 2 * (rt @ be)
    Emm = float(np.sum(ew * rho * np.interp(r, r[sel], eps_mm)))
    return dict(Z=Z, sym=nm(Z), r0=r[sel], rho=rho[sel], Rad=Rad[sel], Q=Q[sel], s=s[sel],
                cn=cn[sel], rt=rt, eps_win=eps_win, eps_full=eps_full, eps_mm=eps_mm,
                leakQ=leakQ, leakE=leakE, Ehf=Ehf, Emm=Emm, npts=len(sel))


def main():
    cols = ["Z", "r0", "rho", "Rad", "Q", "s", "eps_win", "eps_full", "eps_mm", "leakQ", "leakE"]
    flat = {c: [] for c in cols}; cn_all = []; rt_all = []
    atoms_Z = []; atoms_sym = []; atoms_Ehf = []; atoms_Emm = []; atoms_off = []; atoms_n = []
    print(f"{'atom':>4} {'E_x(HF)':>10} {'E_mm':>10} {'err(mHa)':>9} "
          f"{'leakQ% md/mx':>13} {'leakE% md/mx':>13} {'npts':>5}")
    for Z in available_hf():
        b = build_atom(Z)
        if b is None:
            continue
        n = b["npts"]
        flat["Z"].append(np.full(n, Z)); flat["r0"].append(b["r0"]); flat["rho"].append(b["rho"])
        flat["Rad"].append(b["Rad"]); flat["Q"].append(b["Q"]); flat["s"].append(b["s"])
        flat["eps_win"].append(b["eps_win"]); flat["eps_full"].append(b["eps_full"])
        flat["eps_mm"].append(b["eps_mm"]); flat["leakQ"].append(b["leakQ"]); flat["leakE"].append(b["leakE"])
        cn_all.append(b["cn"]); rt_all.append(b["rt"])
        atoms_Z.append(Z); atoms_sym.append(b["sym"]); atoms_Ehf.append(b["Ehf"])
        atoms_Emm.append(b["Emm"]); atoms_off.append(sum(atoms_n)); atoms_n.append(n)
        print(f"{b['sym']:>4} {b['Ehf']:10.4f} {b['Emm']:10.4f} {1e3*(b['Emm']-b['Ehf']):9.1f} "
              f"{1e2*np.median(np.abs(b['leakQ'])):6.2f}/{1e2*np.max(np.abs(b['leakQ'])):5.1f} "
              f"{1e2*np.median(b['leakE']):6.2f}/{1e2*np.max(b['leakE']):5.1f} {n:5d}", flush=True)
    flat = {c: np.concatenate(v) for c, v in flat.items()}
    cn = np.vstack(cn_all); rt = np.vstack(rt_all)
    X = np.column_stack([cn[:, 1:], flat["s"] ** 2])             # kernel feature [l=0 power vec, l=1 s^2]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(os.path.abspath(OUT),
             cn=cn, rt=rt, X=X, **flat,
             atom_Z=np.array(atoms_Z), atom_sym=np.array(atoms_sym),
             atom_Ehf=np.array(atoms_Ehf), atom_Emm=np.array(atoms_Emm),
             atom_offset=np.array(atoms_off), atom_npts=np.array(atoms_n),
             R_c=6.0, n_out=NOUT, X_window=8.0)
    npts = len(flat["Z"])
    mae = 1e3 * np.mean(np.abs(np.array(atoms_Emm) - np.array(atoms_Ehf)))
    print(f"\nwrote {os.path.abspath(OUT)}")
    print(f"  {len(atoms_Z)} exact atoms, {npts} reference points; energy-reconstruction MAE {mae:.2f} mHa")
    print(f"  feature X: ({npts}, {X.shape[1]}) = [cn[1:] (l=0 power vec), s^2]")


if __name__ == "__main__":
    main()
