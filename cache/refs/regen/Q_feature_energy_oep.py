import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf, load_oep
L2P=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz")
QP=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
CLOSED=[2,4,10,12,18,20,30,36,38,46,48,54,56,80]
def emae(useQ,lQ,refp):
    es=[]
    for Z in CLOSED:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=lQ,fp_ref_ridge=1e-8,refs_path=refp,use_l2_power=True,use_Q=useQ,fa_ontop=False,fa_coeff=False))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf'])))
    return np.array(es)
def oep(useQ,lQ,refp,Z):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=lQ,fp_ref_ridge=1e-8,refs_path=refp,use_l2_power=True,use_Q=useQ,fa_ontop=False,fa_coeff=False,gauge_fix=False))
    v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; w=rho[m]*r[m]**2; dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w)
    return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w)),v.min(),v.max()
e=emae(False,0.5,L2P); ob=oep(False,0.5,L2P,4); om=oep(False,0.5,L2P,12)
print("C no-FA, no-Q:        MAE=%.1f signed=%+.1f Be=%+.0f Sr=%+.0f Ba=%+.0f | OEP-Be %.2f[%.0f,%.0f] OEP-Mg %.2f"%(np.abs(e).mean(),e.mean(),e[1],e[8],e[12],ob[0],ob[1],ob[2],om[0]))
for lQ in (0.1,0.3,0.5,1.0):
    e=emae(True,lQ,QP); ob=oep(True,lQ,QP,4); om=oep(True,lQ,QP,12)
    print("D no-FA, +Q lQ=%-4s: MAE=%.1f signed=%+.1f Be=%+.0f Sr=%+.0f Ba=%+.0f | OEP-Be %.2f[%.0f,%.0f] OEP-Mg %.2f"%(lQ,np.abs(e).mean(),e.mean(),e[1],e[8],e[12],ob[0],ob[1],ob[2],om[0]),flush=True)
