"""Held-out transfer benchmark: closed-shell-only refs vs spin-resolved (all-atom) refs, both with the
TEST atoms EXCLUDED. Eval = valence-only int(rho*eps_x) (NLCC-consistent) vs the spin-resolved exact
exchange (atom_Emm). Reference .npz are written to a TEMPDIR -- production atom/xc/data/*.npz untouched.

Result (see reports/hole_expansion/benchmark_heldout.txt): closed-shell (spin-consistent) references
transfer BEST on both closed and open held-out atoms; spin-resolved references DEGRADE transfer and
adding more does not help (not sparsity) -- the spin-UNPOLARIZED functional (input = total rho only)
gets contradictory targets from spin-resolved holes. Production stays on closed-shell references.
"""
import os, sys, tempfile, numpy as np
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hole_refs_full, load_hf
from cache.refs.regen.build_training_sets import valid_filter, to_kernel_feat, kernel_dist, seed_index, fps_order

TEST=[10,18,7,8,26,29]   # Ne,Ar (closed); N,O (open-p); Fe,Cu (open-d)
N=256

def fps_pool(full, pool, sig_lda, heg, path, l0=0.5, l1=0.5):
    Xk=to_kernel_feat(full["X"][pool]); M=len(Xk); D=np.empty((M,M))
    for s in range(0,M,512): D[s:s+512]=kernel_dist(Xk[s:s+512],Xk,l0,l1)
    idx=pool[fps_order(D,seed_index(Xk,l0,l1,heg))][:min(N,M)]
    X=full["X"][idx].copy(); X[:,-1]=_bound(X[:,-1])[0]
    np.savez(path, X=X, DELTA=full["rt"][idx]/(-0.5*full["rho"][idx])[:,None]-sig_lda[None,:])
    return len(idx)

def valEx(Z, refs):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf["r"])); rr=np.asarray(hf["r"])[o]
    rho=np.maximum(np.asarray(hf["rho"])[o],1e-12); w=np.asarray(hf["w"])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,refs_path=refs))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    return float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))

def main():
    full=load_hole_refs_full(); keep,kept,_=valid_filter(full)
    r=np.linspace(1e-3,14,400); F0=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r))
    sig_lda=F0._rhotilde_lda; heg=np.concatenate([F0._cnH[1:],[0.0]])
    nt=~np.isin(full["Z"][kept],TEST)
    with tempfile.TemporaryDirectory() as td:
        cpath=os.path.join(td,"closed.npz"); spath=os.path.join(td,"spin.npz")
        nc=fps_pool(full, kept[nt & full["closed"][kept]], sig_lda, heg, cpath)
        ns=fps_pool(full, kept[nt], sig_lda, heg, spath)
        print("held-out refs: closed-pool ->%d, spin-pool ->%d (test atoms excluded)"%(nc,ns))
        print("%4s %6s %9s %8s %8s %8s   (err vs spin-resolved exact, mHa)"%("atom","shell","Ex_exact","reffree","closed","spin"))
        agg={}
        for Z in TEST:
            i=list(full["atom_Z"]).index(Z); tgt=float(full["atom_Emm"][i])
            cl="closed" if full["closed"][kept][full["Z"][kept]==Z][0] else "open"
            e0,ec,es=valEx(Z,None),valEx(Z,cpath),valEx(Z,spath)
            print("%4s %6s %9.4f %+8.0f %+8.0f %+8.0f"%(nm(Z),cl,tgt,1e3*(e0-tgt),1e3*(ec-tgt),1e3*(es-tgt)))
            agg.setdefault(cl,[]).append((abs(e0-tgt),abs(ec-tgt),abs(es-tgt)))
        for cl,v in agg.items():
            a=np.array(v)*1e3; print("MAE %5s: reffree %.0f  closed %.0f  spin %.0f"%(cl,a[:,0].mean(),a[:,1].mean(),a[:,2].mean()))

if __name__=="__main__": main()
