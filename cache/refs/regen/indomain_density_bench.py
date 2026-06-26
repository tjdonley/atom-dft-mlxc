import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from cache.refs.loader import load_hf
DATA="atom/xc/data"; IN=[2,4,10,12,18,20,30,36,48,54]; RIDGES=[1e-2,1e-4,1e-8]
atoms={}
for Z in IN:
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; atoms[Z]=(r,rho,w,float(hf['Ehf']))
def bench(refsfile):
    res={ridge:[] for ridge in RIDGES}
    for Z,(r,rho,w,Ehf) in atoms.items():
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,refs_path=os.path.abspath(os.path.join(DATA,refsfile))))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        for ridge in RIDGES:
            F.params.fp_ref_ridge=ridge; F._build_fp_nodes(include_refs=True)
            res[ridge].append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-Ehf)))
    return {ridge:np.mean(v) for ridge,v in res.items()}
print("closed-shell IN-DOMAIN MAE (mHa) vs reference density. PBE 49, rSCAN 16.")
print("%-22s %10s %10s %10s"%("ref set (pts/atom)","ridge1e-2","ridge1e-4","exact1e-8"))
for fn,lbl in [("kernel_fp_refs_closed_n512.npz","n512 (~36/atom)"),("kernel_fp_refs_closed_n1024.npz","n1024 (~73/atom)"),("kernel_fp_refs_closed_n1098.npz","n1098 (full,~78)")]:
    m=bench(fn); print("%-22s %10.0f %10.0f %10.0f"%(lbl,m[1e-2],m[1e-4],m[1e-8]),flush=True)
