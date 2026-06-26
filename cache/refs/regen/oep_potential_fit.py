import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_oep
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def mismatch(Z,refs,ridge):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    par=dict(fp_l0=0.7,fp_l1=0.5,gauge_fix=False) if refs is None else dict(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=ridge,refs_path=refs,gauge_fix=False)
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(**par))
    v=F.compute_xc(DensityData(rho=rho)).v_x
    m=rho>1e-3; w=rho[m]*r[m]**2   # density-weighted; remove gauge constant (weighted mean diff)
    dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w); return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w))
print("v_x vs OEP shape mismatch (density-weighted RMS, gauge-aligned). Lower = better potential.")
print("%4s %12s %10s %10s %10s %10s %10s"%("atom","backbone","r=1e-2","r=1e-1","r=1.0","r=10","r=100"))
for Z,nm in [(2,'He'),(4,'Be'),(12,'Mg')]:
    bb=mismatch(Z,None,0)
    row=[mismatch(Z,REF,rg) for rg in (1e-2,1e-1,1.0,10.0,100.0)]
    print("%4s %12.3f %10.3f %10.3f %10.3f %10.3f %10.3f"%(nm,bb,*row),flush=True)
