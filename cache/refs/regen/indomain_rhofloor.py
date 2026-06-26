import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
DATA="atom/xc/data"; f=load_hole_refs_full()
clo=np.asarray(f['closed']); leakQ=np.asarray(f['leakQ']); leakE=np.asarray(f['leakE']); rhoA=np.asarray(f['rho']); rtA=np.asarray(f['rt']); XA=np.asarray(f['X'])
sig_lda=SIMPLE_HOLE_KERNEL_FP(r_quad=np.linspace(1e-3,14,400),quadrature_weights=np.gradient(np.linspace(1e-3,14,400)))._rhotilde_lda
leakok=np.maximum(np.abs(leakQ),np.abs(leakE))<=0.10
def build(rho_floor):
    m=clo&leakok&(rhoA>=rho_floor); X=XA[m].copy(); X[:,-1]=_bound(X[:,-1])[0]; DELTA=rtA[m]/(-0.5*rhoA[m])[:,None]-sig_lda[None,:]
    p=DATA+"/_rf.npz"; np.savez(p,X=X,DELTA=DELTA,idx=np.where(m)[0]); return p,int(m.sum())
IN=[2,4,10,12,18,20,30,36,48,54]; atoms={}
for Z in IN:
    h=load_hf(Z); o=np.argsort(np.asarray(h['r'])); r=np.asarray(h['r'])[o]; rho=np.maximum(np.asarray(h['rho'])[o],1e-12); w=np.asarray(h['w'])[o]; atoms[Z]=(r,rho,w,float(h['Ehf']))
def mae(refs,ridge):
    es=[]
    for Z,(r,rho,w,Ehf) in atoms.items():
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=ridge,refs_path=os.path.abspath(refs)))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-Ehf)))
    return np.mean(es),np.max(es)
print("closed-shell IN-DOMAIN MAE / MAX (mHa) vs reference rho-floor (leak<=10%% kept). PBE 49, rSCAN 16.")
print("%-14s %6s %16s %16s"%("rho_floor","Npts","exact1e-8 MAE/max","ridge1e-2 MAE/max"))
for rf in (0.1,0.05,0.03,0.01):
    p,n=build(rf); me,mx=mae(p,1e-8); me2,mx2=mae(p,1e-2)
    print("%-14.2f %6d %16s %16s"%(rf,n,"%.0f / %.0f"%(me,mx),"%.0f / %.0f"%(me2,mx2)),flush=True)
os.remove(DATA+"/_rf.npz")
