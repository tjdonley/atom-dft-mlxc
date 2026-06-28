import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
SCR=os.environ["SCR"]
pool=load_hole_refs_full(); lk=np.maximum(np.abs(pool["leakQ"]),np.abs(pool["leakE"]))
keep=(lk<=0.10)&(pool["rho"]>=0.01); kept=np.where(keep)[0]
rr=np.linspace(1e-3,14,400); F0=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr)); sig=F0._rhotilde_lda
# FPS order over the FULL pool (closed+open) in (cn,s^2) kernel metric, running-min (no NxN matrix)
Xk=pool["X"][kept].copy(); Xk[:,9]=_bound(Xk[:,9])[0]
w=np.concatenate([np.full(9,1/0.7),[1/0.5]]); Xw=Xk*w; sq=np.sum(Xw*Xw,1)
def dists(j): return np.sqrt(np.maximum(1-np.exp(-0.5*np.maximum(sq+sq[j]-2*Xw@Xw[j],0)),0))
Nmax=min(2048,len(kept)); sel=[0]; mind=dists(0)
for _ in range(Nmax-1):
    mind[sel]=-1.0; j=int(np.argmax(mind)); sel.append(j); mind=np.minimum(mind,dists(j))
idx=kept[np.array(sel)]                                   # GLOBAL pool indices, FPS order
Zr=pool["Z"][idx]; r0r=pool["r0"][idx]
Xbase=np.column_stack([pool["X"][idx][:,:9], _bound(pool["X"][idx][:,9])[0]])
DELTA=pool["rt"][idx]/(-0.5*pool["rho"][idx])[:,None]-sig[None,:]
# p2 (l2power) per atom on its HF grid, interp at r0
p2=np.zeros(len(idx)); cache={}
for Z in sorted(set(Zr.tolist())):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ww=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=ww,params=P(use_l2_power=True))
    cp=np.array([op@rho for op in F._ops]); R_ad,_=F._R_ad(rho); c_ad=F._c_ad(cp,R_ad)
    m=Zr==Z; p2[m]=np.interp(r0r[m],r,F._l2_power_feat(rho,R_ad,c_ad[:,0]))
np.savez(os.path.join(SCR,"allshell_base.npz"),X=Xbase,DELTA=DELTA)
np.savez(os.path.join(SCR,"allshell_l2.npz"),X=np.column_stack([Xbase,p2]),DELTA=DELTA)
nopen=int((~pool["closed"][idx]).sum())
print("all-shell FPS refs: N=%d (%d open, %d closed) from %d atoms"%(len(idx),nopen,len(idx)-nopen,len(set(Zr.tolist()))),flush=True)
