import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_oep
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def vx(Z,refs,ridge):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    par=dict(fp_l0=0.7,fp_l1=0.5) if refs is None else dict(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=ridge,refs_path=refs)
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(**par))
    v=F.compute_xc(DensityData(rho=rho)).v_x
    return r,rho,vxo,v
print("v_x vs OEP. report over rho>1e-3 (SCF-relevant): max|v_x|, and roughness = mean|d2 v_x/dr2| (oscillation proxy).")
print("%4s %22s %14s %14s"%("atom","config","max|v_x|","roughness"))
for Z,nm in [(4,'Be'),(12,'Mg'),(10,'Ne')]:
    try:
        r,rho,vxo,_=vx(Z,None,0)
    except Exception as e: print("%4s skip (%s)"%(nm,e)); continue
    m=rho>1e-3
    def rough(v): return np.mean(np.abs(np.gradient(np.gradient(v[m],r[m]),r[m])))
    print("%4s %22s %14.2f %14.1f"%(nm,"OEP (exact-x target)",np.max(np.abs(vxo[m])),rough(vxo)))
    for refs,ridge,lbl in [(None,0,"backbone (ref-free)"),(REF,1e-2,"rf001 ridge1e-2"),(REF,1e-8,"rf001 exact1e-8")]:
        r2,rho2,_,v=vx(Z,refs,ridge); m2=rho2>1e-3
        rr=np.mean(np.abs(np.gradient(np.gradient(v[m2],r2[m2]),r2[m2])))
        print("%4s %22s %14.2f %14.1f"%("",lbl,np.max(np.abs(v[m2])),rr))
