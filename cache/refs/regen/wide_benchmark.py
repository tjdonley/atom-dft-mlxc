import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.gga_pbe import GGA_PBE
from atom.xc.meta_scan import SCAN, rSCANParameters
from atom.xc.evaluator import DensityData
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf, available_hf
QP=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
CLOSED={2,4,10,12,18,20,30,36,38,46,48,54,56,80}
def period(Z): return 1 if Z<=2 else 2 if Z<=10 else 3 if Z<=18 else 4 if Z<=36 else 5 if Z<=54 else 6
def tau_hf(hf,r):
    g=np.asarray(hf['g_sorted']); occ=np.asarray(hf['occ']); l=np.asarray(hf['l_values']); rs=np.asarray(hf['r_sorted']); o=np.argsort(rs); rs=rs[o]; g=g[o]
    gi=np.array([np.interp(r,rs,g[:,i]) for i in range(g.shape[1])]).T; dg=np.gradient(gi,r,axis=0)
    return (1/(8*np.pi))*np.sum(occ*(dg**2+(l*(l+1))*(gi/r[:,None])**2),axis=1)
rows=[]
for Z in available_hf():
    try:
        hf=load_hf(Z)
    except Exception: continue
    o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    grad=np.gradient(rho,r); ew=4*np.pi*r**2*w; D=np.eye(len(r))
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=QP,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    es=float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))
    ep=float(np.sum(ew*rho*GGA_PBE(derivative_matrix=D,r_quad=r).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad)).e_generic))
    er=float(np.sum(ew*rho*SCAN(derivative_matrix=D,r_quad=r,params=rSCANParameters()).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad,tau=tau_hf(hf,r))).e_generic))
    rows.append((Z,nm(Z),period(Z),Z in CLOSED,Ehf,1e3*(es-Ehf),1e3*(ep-Ehf),1e3*(er-Ehf)))
    print("%3d %3s P%d %-6s Ehf=%9.3f  SIMPLE %+7.0f  PBE %+7.0f  rSCAN %+7.0f"%(
        Z,nm(Z),period(Z),"closed" if Z in CLOSED else "open",Ehf,rows[-1][5],rows[-1][6],rows[-1][7]),flush=True)
np.save(os.path.join(os.path.dirname(__file__),"wide_benchmark_rows.npy"),np.array([(r[0],r[2],r[3],r[5],r[6],r[7]) for r in rows],dtype=float))
print("\nsaved wide_benchmark_rows.npy (%d atoms)"%len(rows))
