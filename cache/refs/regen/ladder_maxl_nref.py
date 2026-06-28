import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from cache.refs.loader import load_hf
SCR=os.environ["SCR"]
Xb=np.load("atom/xc/data/kernel_fp_refs_closed_rf001.npz")["X"]; Db=np.load("atom/xc/data/kernel_fp_refs_closed_rf001.npz")["DELTA"]
z2=np.load("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz"); X2=z2["X"]; D2=z2["DELTA"]
# FPS order on (cn,s^2) kernel features -> nested space-filling prefixes
w=np.concatenate([np.full(9,1/0.7),[1/0.5]]); A=Xb*w; d2=np.sum(A*A,1)[:,None]+np.sum(A*A,1)[None,:]-2*A@A.T
Dist=np.sqrt(np.maximum(1-np.exp(-0.5*np.maximum(d2,0)),0)); N=len(Xb)
sel=[0]; mind=Dist[0].copy()
for _ in range(N-1):
    mind[sel]=-1; j=int(np.argmax(mind)); sel.append(j); mind=np.minimum(mind,Dist[j])
perm=np.array(sel)
CLOSED=[2,4,10,12,18,36,54,80]
HF={Z:load_hf(Z) for Z in CLOSED}
def mae(maxl,n):
    idx=perm[:n]
    if maxl<=1: X,D,l2=Xb[idx],Db[idx],False
    else: X,D,l2=X2[idx],D2[idx],True
    rp=os.path.join(SCR,"lad_%d_%d.npz"%(maxl,n)); np.savez(rp,X=X,DELTA=D)
    l1=100.0 if maxl==0 else 0.5; mu=0.0 if maxl==0 else 10/81
    es=[]
    for Z in CLOSED:
        hf=HF[Z]; o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ww=np.asarray(hf['w'])[o]
        try:
            F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=ww,params=P(fp_l0=0.7,fp_l1=l1,fp_mu=mu,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path=rp,use_l2_power=l2,fa_ontop=False,fa_coeff=False))
            cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
            es.append(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf'])))
        except Exception as e: es.append(np.nan)
    return np.nanmean(np.abs(es))
NS=[64,128,256,512,1024,1529]
print("CLOSED-shell exchange MAE vs exact Ehf [mHa] (principled target). FA-free, no-Q.")
print("%-8s"%"max_l \\ N"+"".join("%7d"%n for n in NS))
for maxl in (0,1,2):
    print("l<=%d    "%maxl+"".join("%7.0f"%mae(maxl,n) for n in NS),flush=True)
