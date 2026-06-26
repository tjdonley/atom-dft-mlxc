import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.gga_pbe import GGA_PBE
from atom.xc.meta_scan import SCAN, rSCANParameters
from atom.xc.evaluator import DensityData
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
CLOSED=[2,4,10,12,18,20,30,36,38,46,48,54,56,80]
def tau_hf(hf,r):
    g=np.asarray(hf['g_sorted']); occ=np.asarray(hf['occ']); l=np.asarray(hf['l_values']); rs=np.asarray(hf['r_sorted']); o=np.argsort(rs); rs=rs[o]; g=g[o]
    gi=np.array([np.interp(r,rs,g[:,i]) for i in range(g.shape[1])]).T; dg=np.gradient(gi,r,axis=0)
    return (1/(8*np.pi))*np.sum(occ*(dg**2+(l*(l+1))*(gi/r[:,None])**2),axis=1)
print("FINAL closed-shell exchange benchmark (valence-only int rho*eps_x vs Ehf, mHa). SIMPLE = rf001 (rho>=0.01 coverage), exact interp.")
print("%4s %10s %8s %8s %8s"%("atom","Ehf","SIMPLE","PBE","rSCAN"))
S=[];Pp=[];R=[]
for Z in CLOSED:
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    grad=np.gradient(rho,r); ew=4*np.pi*r**2*w; D=np.eye(len(r))
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    es=float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))
    ep=float(np.sum(ew*rho*GGA_PBE(derivative_matrix=D,r_quad=r).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad)).e_generic))
    er=float(np.sum(ew*rho*SCAN(derivative_matrix=D,r_quad=r,params=rSCANParameters()).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad,tau=tau_hf(hf,r))).e_generic))
    S.append(1e3*(es-Ehf)); Pp.append(1e3*(ep-Ehf)); R.append(1e3*(er-Ehf))
    print("%4s %10.4f %8.0f %8.0f %8.0f"%(nm(Z),Ehf,S[-1],Pp[-1],R[-1]),flush=True)
print("MAE  %25.0f %8.0f %8.0f"%(np.mean(np.abs(S)),np.mean(np.abs(Pp)),np.mean(np.abs(R))))
print("max|err| SIMPLE %.0f (%s)"%(np.max(np.abs(S)),nm(CLOSED[int(np.argmax(np.abs(S)))])))
