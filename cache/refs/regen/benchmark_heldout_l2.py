"""Held-out benchmark: does adding the l=2 (quadrupole) axial feature improve closed-shell references?
Same closed-shell refs and SAME FPS-selected points (test atoms excluded); the only difference is the
kernel feature: 10-dim [l0 power vector, s^2] vs 11-dim [..., t^2] (t = reduced l=2 axial). Eval =
valence-only int(rho*eps_x) (NLCC-consistent) vs the spin-resolved exact exchange (atom_Emm).

Result (reports/hole_expansion/benchmark_heldout_l2.txt): l=2 gives a SMALL improvement on closed
held-out atoms (MAE 99->89 mHa at ell2=0.5) and negligible change on open (118->120) -- l0+l1 already
resolve the holes smoothly, so l=2 is a minor refinement. Wired into the functional as use_l2 (default
OFF); refs written to a TEMPDIR, production atom/xc/data/*.npz untouched.
"""
import os, sys, tempfile, numpy as np
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.descriptors.simple.derivatives import build_spectral_l2_operator
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hole_refs_full, load_hf
from cache.refs.regen.build_training_sets import valid_filter, to_kernel_feat, kernel_dist, seed_index, fps_order
TEST=[10,18,7,8,26,29]; N=256

def t2_for(full, indices):
    """Bounded reduced l=2 feature t^2 at each (atom,r0) reference point (built per atom)."""
    out=np.zeros(len(indices))
    for Z in np.unique(full["Z"][indices]):
        hf=load_hf(int(Z)); o=np.argsort(np.asarray(hf["r"])); rr=np.asarray(hf["r"])[o]
        rho=np.maximum(np.asarray(hf["rho"])[o],1e-12); L2=build_spectral_l2_operator(rr)
        kF=(3*np.pi**2*rho)**(1/3); t2b=_bound((L2@rho/(4*kF**2*rho))**2)[0]
        m=full["Z"][indices]==Z; out[m]=np.interp(full["r0"][indices][m], rr, t2b)
    return out

def valEx(Z, refs, use_l2, fp_l2=0.5):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf["r"])); rr=np.asarray(hf["r"])[o]
    rho=np.maximum(np.asarray(hf["rho"])[o],1e-12); w=np.asarray(hf["w"])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=w,
                            params=P(fp_l0=0.7,fp_l1=0.5,use_l2=use_l2,fp_l2=fp_l2,refs_path=refs))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    return float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))

def main():
    full=load_hole_refs_full(); keep,kept,_=valid_filter(full)
    r=np.linspace(1e-3,14,400); F0=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r))
    sig_lda=F0._rhotilde_lda; heg=np.concatenate([F0._cnH[1:],[0.0]])
    pool=kept[(~np.isin(full["Z"][kept],TEST)) & full["closed"][kept]]
    Xk=to_kernel_feat(full["X"][pool]); idx=pool[fps_order(kernel_dist(Xk,Xk,0.5,0.5),seed_index(Xk,0.5,0.5,heg))][:N]
    DELTA=full["rt"][idx]/(-0.5*full["rho"][idx])[:,None]-sig_lda[None,:]
    X10=full["X"][idx].copy(); X10[:,-1]=_bound(X10[:,-1])[0]
    with tempfile.TemporaryDirectory() as td:
        c10=os.path.join(td,"c10.npz"); c11=os.path.join(td,"c11.npz")
        np.savez(c10,X=X10,DELTA=DELTA); np.savez(c11,X=np.column_stack([X10,t2_for(full,idx)]),DELTA=DELTA)
        print("closed refs: %d pts; 10-dim [l0,l1] vs 11-dim [l0,l1,l2]"%len(idx))
        print("%4s %6s %9s %8s %12s %12s"%("atom","shell","Ex_exact","closed","closed+l2(.5)","closed+l2(1)"))
        agg={}
        for Z in TEST:
            i=list(full["atom_Z"]).index(Z); tgt=float(full["atom_Emm"][i])
            cl="closed" if full["closed"][kept][full["Z"][kept]==Z][0] else "open"
            ec,e5,e1=valEx(Z,c10,False),valEx(Z,c11,True,0.5),valEx(Z,c11,True,1.0)
            print("%4s %6s %9.4f %+8.0f %+12.0f %+12.0f"%(nm(Z),cl,tgt,1e3*(ec-tgt),1e3*(e5-tgt),1e3*(e1-tgt)))
            agg.setdefault(cl,[]).append((abs(ec-tgt),abs(e5-tgt),abs(e1-tgt)))
        for cl,v in agg.items():
            a=np.array(v)*1e3; print("MAE %5s: closed %.0f  +l2(.5) %.0f  +l2(1) %.0f"%(cl,a[:,0].mean(),a[:,1].mean(),a[:,2].mean()))

if __name__=="__main__": main()
