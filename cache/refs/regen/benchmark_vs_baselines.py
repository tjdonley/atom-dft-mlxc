"""Categorized exchange-energy benchmark: SIMPLE (reference-free + n512 refs) vs PBE & rSCAN, all
referenced to the exact exchange of the HF density (Ehf). Broken down by block (s/p/d) and by
closed- vs open-shell.

!! INVALID AS-IS (found 2026-06-25) !! The cached PBE/rSCAN exchange and Ehf are NOT on a consistent
footing: |pbe_Ex - Ehf| is 0.7-13 Ha across the table (only He/Ne/Na/Mg agree), far too large for real
PBE-vs-HF exchange. Ehf is the spin-RESTRICTED exact exchange (orbital-hole reconstruction matches it
to <3 mHa); the spin-resolved correction explains only part of the open-shell gap and NONE of the
closed-shell Ar gap -- the dominant cause is a PSP/setup inconsistency between the HF and PBE/rSCAN
solves. DO NOT trust these numbers until PBE/rSCAN are recomputed NON-SCF on the cached HF densities
(PBE: rho,grad; rSCAN: + tau). See memory simple-hole-baseline-cache.
"""
import argparse, os, sys
import numpy as np
np.seterr(all="ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_REPO)
import atom.xc.simple_hole_expansion as She
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf, load_baseline
DATA=os.path.join(_REPO,"atom","xc","data"); NONE="/nonexistent.npz"

def block(Z):
    if Z in (21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,57,72,73,74,75,76,77,78,79,80): return "d"
    if Z in (5,6,7,8,9,10,13,14,15,16,17,18,31,32,33,34,35,36,49,50,51,52,53,54,81,82,83): return "p"
    return "s"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--l0",type=float,default=0.7); ap.add_argument("--l1",type=float,default=0.5)
    ap.add_argument("--ref",default=os.path.join(DATA,"kernel_fp_refs_n512.npz")); a=ap.parse_args()
    rows=[]
    from cache.refs.loader import available_hf
    for Z in available_hf():
        try: hf=load_hf(Z)
        except Exception: continue
        occ=np.asarray(hf["occ"]); l=np.asarray(hf["l_values"])
        if occ.size==0: continue
        closed = bool(np.all(np.isclose(occ[occ>1e-8], 2*(2*l[occ>1e-8]+1))))
        o=np.argsort(np.asarray(hf["r"])); r=np.asarray(hf["r"])[o]
        rho=np.maximum(np.asarray(hf["rho"])[o],1e-12); w=np.asarray(hf["w"])[o]; Ehf=float(hf["Ehf"])
        pbe,rsc=load_baseline(Z)
        She._KERNEL_FP_REFS=NONE; F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w)
        F._fp_l0=a.l0; F._fp_l1=a.l1
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        F._build_fp_nodes(include_refs=True); e_free=float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))
        She._KERNEL_FP_REFS=a.ref; F._build_fp_nodes(include_refs=True)
        e_ref=float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))
        rows.append(dict(Z=Z,sym=nm(Z),block=block(Z),closed=closed,Ehf=Ehf,
                         reffree=1e3*(e_free-Ehf), simple=1e3*(e_ref-Ehf),
                         pbe=1e3*(pbe-Ehf) if np.isfinite(pbe) else np.nan,
                         rscan=1e3*(rsc-Ehf) if np.isfinite(rsc) else np.nan))
    cols=["reffree","simple","pbe","rscan"]
    def mae(rs,c): v=[abs(x[c]) for x in rs if np.isfinite(x[c])]; return np.mean(v) if v else np.nan
    def line(name,rs): print(f"{name:>22} {len(rs):>3}  "+"  ".join(f"{c}={mae(rs,c):>5.0f}" for c in cols))
    print("MAE of exchange energy vs Ehf (mHa). simple = SIMPLE hole + n512 refs (l0=%.1f l1=%.1f, non-SCF)."%(a.l0,a.l1))
    print("  PBE/rSCAN = self-consistent (cached). [caveat: SC vs non-SCF]\n")
    print(f"{'category':>22} {'N':>3}  {'  '.join(cols)}")
    line("ALL",rows)
    for b in ("s","p","d"): line(f"block {b}",[x for x in rows if x["block"]==b])
    line("closed-shell (in-dom)",[x for x in rows if x["closed"]])
    line("open-shell (out-dom)",[x for x in rows if not x["closed"]])
    out=os.path.join(_REPO,"reports","hole_expansion","benchmark_vs_baselines.txt")
    with open(out,"w") as f:
        f.write(f"{'sym':>4} {'blk':>3} {'shell':>6} {'Ehf':>9} {'reffree':>8} {'simple':>7} {'pbe':>6} {'rscan':>6}\n")
        for x in sorted(rows,key=lambda z:z["Z"]):
            f.write(f"{x['sym']:>4} {x['block']:>3} {'closed' if x['closed'] else 'open':>6} {x['Ehf']:>9.4f} "
                    f"{x['reffree']:>8.0f} {x['simple']:>7.0f} {x['pbe']:>6.0f} {x['rscan']:>6.0f}\n")
    print(f"\nwrote {out}")

if __name__=="__main__": main()
