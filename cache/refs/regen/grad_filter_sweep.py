import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf
R=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
def mk(gf,r,w,gauge=False):
    return SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=R,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False,grad_filter=gf,gauge_fix=gauge))
def vxrange(gf,Z):  # v_x spike on the HF density + the GRADIENT-term range
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ew=4*np.pi*r**2*np.asarray(hf['w'])[o]
    F=mk(gf,r,np.asarray(hf['w'])[o]); v=F.compute_xc(DensityData(rho=rho)).v_x
    # isolate the gradient term grad_op^T(ew rho deps_dg)/ew
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho; hg=1e-6*(np.abs(g)+1e-8)
    dg=(F._kernel_eps(cp,rho,g+hg)-F._kernel_eps(cp,rho,g-hg))/(2*hg); Tg=F._grad_op.T@(F.energy_weights*rho*dg)/F.energy_weights
    return v.min(),v.max(),np.abs(Tg).max()
def emae(gf,atoms=[2,4,10,12,18]):
    es=[]
    for Z in atoms:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
        F=mk(gf,r,w); cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf']))))
    return np.mean(es)
print("grad_filter sweep. Mg v_x spike baseline [-30,51], grad-term |max|. energy MAE(He,Be,Ne,Mg,Ar).")
print("%10s %16s %14s %10s"%("grad_filter","Mg v_x[min,max]","Mg gradterm","energyMAE"))
for gf in (0.0,0.6,0.4,0.25,0.15,0.1):
    mn,mx,tg=vxrange(gf,12)
    print("%10s [%7.1f,%6.1f] %14.1f %10.0f"%(gf,mn,mx,tg,emae(gf)),flush=True)
