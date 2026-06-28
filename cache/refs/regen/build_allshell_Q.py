import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
SCR=os.environ["SCR"]
pool=load_hole_refs_full(); lk=np.maximum(np.abs(pool["leakQ"]),np.abs(pool["leakE"]))
keep=(lk<=0.10)&(pool["rho"]>=0.01); kept=np.where(keep)[0]
Xk=pool["X"][kept].copy(); Xk[:,9]=_bound(Xk[:,9])[0]
w=np.concatenate([np.full(9,1/0.7),[1/0.5]]); Xw=Xk*w; sq=np.sum(Xw*Xw,1)
def dists(j): return np.sqrt(np.maximum(1-np.exp(-0.5*np.maximum(sq+sq[j]-2*Xw@Xw[j],0)),0))
Nmax=min(2048,len(kept)); sel=[0]; mind=dists(0)
for _ in range(Nmax-1):
    mind[sel]=-1.0; j=int(np.argmax(mind)); sel.append(j); mind=np.minimum(mind,dists(j))
idx=kept[np.array(sel)]; Zr=pool["Z"][idx]; r0r=pool["r0"][idx]
X2=np.load(SCR+"/allshell_l2.npz")["X"]; D2=np.load(SCR+"/allshell_l2.npz")["DELTA"]   # same FPS order
Qb=np.zeros(len(idx))
for Z in sorted(set(Zr.tolist())):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ww=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=ww)
    cp=np.array([op@rho for op in F._ops]); R_ad,_=F._R_ad(rho); c_ad=F._c_ad(cp,R_ad); d=c_ad/(4*np.pi*R_ad[:,None]**1.5)
    m=Zr==Z; Qb[m]=np.interp(r0r[m],r,_bound(4*np.pi*R_ad**3*(d@F._Bmom))[0])
out="atom/xc/data/kernel_fp_refs_allshell_l2power_Q.npz"; np.savez(out,X=np.column_stack([X2,Qb]),DELTA=D2,use_l2_power=True,use_Q=True)
print("built %s : %d refs, %d dims"%(out,len(idx),X2.shape[1]+1),flush=True)
