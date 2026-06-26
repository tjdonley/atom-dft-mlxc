"""Stage 2: fit reference-node hole AMPLITUDES so the functional also reproduces the exact OEP
exchange potential. v_x is affine in the node hole deviations (verified), so with DELTA_k = a_k *
DELTA_exact_k the fit is a linear least-squares:
    min_a  w_E ||a-1||^2  +  gamma * sum_{OEP atoms} || v_x[a] - v_x^OEP ||^2_{rho-weighted}
 => (w_E I + gamma Psi^T Psi) a = w_E 1 + gamma Psi^T (v_oep - v_x[a=0]),   Psi[:,k] = dv_x/da_k.
a=1 recovers the pure hole fit (Stage 1); gamma re-shapes the hole so its potential matches OEP.
Writes atom/xc/data/kernel_fp_refs_closed_oep_nN.npz (X, DELTA=a*DELTA_exact); backbone limits +
Stage-1 ref-ridge intact. Load with the same params, just point refs_path here.
"""
import argparse, os, sys, glob
import numpy as np
np.seterr(all="ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_REPO)
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _GEA2
from atom.xc.evaluator import DensityData
DATA=os.path.join(_REPO,"atom","xc","data"); OEP=os.path.join(_HERE,"..","oep")

def atom_machinery(base, ridge, N):
    """Per-OEP-atom: a closure v_x(Db) using the functional's ridged kernel solve (affine in Db)."""
    def make(npz):
        z=np.load(npz); r=z["r"]; o=np.argsort(r); r=r[o]; rho=np.maximum(z["rho"][o],1e-12); vo=z["vx_oep"][o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),
                                params=P(fp_l0=0.7,fp_l1=0.5,refs_path=base,fp_ref_ridge=ridge))
        Xn=F._fp_Xnodes; nl0=F._n_out-1; g=F._grad_op@rho
        rd=np.full(len(Xn),F._fp_ridge); rd[2:]=ridge; K=F._Kmat(Xn,Xn)+np.diag(rd); Kinv=np.linalg.solve(K,np.eye(len(K)))
        dK1=F._Kmat(Xn[:1],Xn)[0]*Xn[:,nl0]/F._fp_l1**2; row=F._fp_kappa*(dK1@Kinv); Cmom=F._Cmom
        def vx(Db):
            mu=np.concatenate([[0,0],Db@Cmom]); cG=(_GEA2-float(row@mu))/float(row[1]*F._fp_dgb)
            Delta=np.vstack([np.zeros(F._n_out),cG*F._dgea,Db]); F._fp_coef=Kinv@Delta; F._fp_cG=cG
            return F.compute_xc(DensityData(rho=rho,grad_rho=g)).v_x
        wt=np.sqrt(4*np.pi*r**2*np.gradient(r)*rho)*(rho>1e-4)   # rho-weighted potential norm
        return dict(r=r,rho=rho,vo=vo,vx=vx,wt=wt,Db=z if False else None)
    return make

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--N",type=int,default=64)
    ap.add_argument("--ridge",type=float,default=1e-2); ap.add_argument("--gamma",type=float,default=1.0)
    ap.add_argument("--wE",type=float,default=1.0); a=ap.parse_args()
    base=os.path.join(DATA,f"kernel_fp_refs_closed_n{a.N}.npz")
    Db_ex=np.load(base)["DELTA"]; X=np.load(base)["X"]; nref=len(Db_ex)
    oep_files=sorted(glob.glob(os.path.join(OEP,"oep_Z*.npz")))
    print(f"base=closed_n{a.N} ({nref} refs), ridge={a.ridge}, OEP atoms={[os.path.basename(f) for f in oep_files]}, gamma={a.gamma}")
    PSI=[]; RES=[]
    for f in oep_files:
        m=atom_machinery(base,a.ridge,a.N)(f)
        v0=m["vx"](np.zeros_like(Db_ex))
        psi=np.empty((len(m["r"]),nref))
        for k in range(nref):
            Dk=np.zeros_like(Db_ex); Dk[k]=Db_ex[k]; psi[:,k]=m["vx"](Dk)-v0
        PSI.append(m["wt"][:,None]*psi); RES.append(m["wt"]*(m["vo"]-v0))
    Psi=np.vstack(PSI); res=np.concatenate(RES)
    A=a.wE*np.eye(nref)+a.gamma*(Psi.T@Psi); b=a.wE*np.ones(nref)+a.gamma*(Psi.T@res)
    amp=np.linalg.solve(A,b)
    out=os.path.join(DATA,f"kernel_fp_refs_closed_oep_n{a.N}.npz")
    np.savez(out, X=X, DELTA=amp[:,None]*Db_ex, amp=amp, fp_l0=0.7, fp_l1=0.5, fp_ref_ridge=a.ridge, gamma=a.gamma)
    print(f"  amplitudes a: mean {amp.mean():.2f} std {amp.std():.2f} range [{amp.min():.2f},{amp.max():.2f}]")
    print(f"  saved {out}")

if __name__=="__main__": main()
