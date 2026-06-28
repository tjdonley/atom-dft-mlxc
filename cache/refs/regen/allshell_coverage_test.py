import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hole_refs_full, load_hf
from cache.refs.regen.build_training_sets import to_kernel_feat, kernel_dist, seed_index, fps_order
SCR=os.environ["SCR"]
refs=load_hole_refs_full()
lk=np.maximum(np.abs(refs["leakQ"]),np.abs(refs["leakE"])); keep=(lk<=0.10)&(refs["rho"]>=0.01)   # ALL shells (no closed filter)
kept=np.where(keep)[0]
rr=np.linspace(1e-3,14,400); F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr))
sig=F._rhotilde_lda; heg=np.concatenate([F._cnH[1:],[0.0]])
Xk=to_kernel_feat(refs["X"][kept]); l0m,l1m=0.5,0.5
order=kept[fps_order(kernel_dist(Xk,Xk,l0m,l1m),seed_index(Xk,l0m,l1m,heg))]
N=1529; idx=order[:N]
X=refs["X"][idx].copy(); X[:,-1]=_bound(X[:,-1])[0]
DELTA=refs["rt"][idx]/(-0.5*refs["rho"][idx])[:,None]-sig[None,:]
out=os.path.join(SCR,"allshell_n1529.npz"); np.savez(out,X=X,DELTA=DELTA,idx=idx,fp_l0=0.7,fp_l1=0.5)
nopen=int((~refs["closed"][idx]).sum()); print("all-shell refs: %d pts (%d open, %d closed) from %d atoms"%(N,nopen,N-nopen,len(set(refs["Z"][idx].tolist()))),flush=True)
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def emae(refp,Z):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
    G=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=refp))
    cp=np.array([op@rho for op in G._ops]); g=G._grad_op@rho
    return 1e3*(float(np.sum(G.energy_weights*rho*G._kernel_eps(cp,rho,g)))-float(hf['Ehf']))
print("%4s %8s %12s %12s"%("Z","shell","closed-only","ALL-shell"))
for Z,nm,sh in [(4,'Be','clo'),(10,'Ne','clo'),(12,'Mg','clo'),(18,'Ar','clo'),(6,'C','OPEN'),(7,'N','OPEN'),(8,'O','OPEN'),(14,'Si','OPEN'),(15,'P','OPEN'),(21,'Sc','OPEN'),(22,'Ti','OPEN')]:
    print("%4s %8s %12.0f %12.0f"%(nm,sh,emae(REF,Z),emae(out,Z)),flush=True)
