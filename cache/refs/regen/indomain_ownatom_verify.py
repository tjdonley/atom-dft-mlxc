import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
f=load_hole_refs_full(); Zf=np.asarray(f['Z']); r0f=np.asarray(f['r0']); rtf=np.asarray(f['rt']); rhof=np.asarray(f['rho']); Xf=np.asarray(f['X'])
DATA="atom/xc/data"
def own_refs(Z):
    m=np.where(Zf==Z)[0]; sig_lda=SIMPLE_HOLE_KERNEL_FP(r_quad=np.linspace(1e-3,14,400),quadrature_weights=np.gradient(np.linspace(1e-3,14,400)))._rhotilde_lda
    X=Xf[m].copy(); X[:,-1]=_bound(X[:,-1])[0]
    DELTA=rtf[m]/(-0.5*rhof[m])[:,None]-sig_lda[None,:]
    p=os.path.abspath(os.path.join(DATA,"_own_%d.npz"%Z)); np.savez(p,X=X,DELTA=DELTA,idx=m); return p,len(m)
def ex(Z,refs,ridge,use_l2=False):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=ridge,refs_path=refs))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    return 1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-Ehf)
print("ACTUAL functional (full machinery: FA blend + 2-constraint projection), own-atom dense refs (all 150).")
print("Q: does the full functional reach ~1 mHa in-domain? err vs Ehf (mHa).")
print("%4s %6s %12s %12s %16s"%("atom","Nown","own ridge1e-8","own ridge1e-2","prod n1098 1e-8"))
for Z in [2,10,18,36]:
    p,n=own_refs(Z)
    print("%4d %6d %12.0f %12.0f %16.0f"%(Z,n,ex(Z,p,1e-8),ex(Z,p,1e-2),ex(Z,os.path.abspath(DATA+"/kernel_fp_refs_closed_n1098.npz"),1e-8)),flush=True)
    os.remove(p)
