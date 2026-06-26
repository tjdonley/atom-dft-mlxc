import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf, load_oep
L2P=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz")
def oepmis(Z,l2pow_ls,l0=0.7):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=l0,fp_l1=0.5,fp_l2pow=l2pow_ls,fp_ref_ridge=1e-8,refs_path=L2P,use_l2_power=True,gauge_fix=False))
    v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; w=rho[m]*r[m]**2; dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w)
    return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w))
def emae(l2pow_ls,l0=0.7,atoms=[2,4,10,18,12,20,30,36]):
    es=[]
    for Z in atoms:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=l0,fp_l1=0.5,fp_l2pow=l2pow_ls,fp_ref_ridge=1e-8,refs_path=L2P,use_l2_power=True))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf']))))
    return np.mean(es)
print("Tune l=2-power length scale (separates atoms). in-domain Be .27 Mg .93")
print("%8s %8s %10s %10s %10s"%("l2pow_ls","l0","OEP-Be","OEP-Mg","energyMAE"))
for ls in (0.5,0.2,0.1,0.05,0.02):
    print("%8.2f %8.1f %10.3f %10.3f %10.0f"%(ls,0.7,oepmis(4,ls),oepmis(12,ls),emae(ls)),flush=True)
