"""SIMPLE vs PBE vs rSCAN exchange benchmark (valence-only int(rho*eps_x) on HF density vs Ehf,
non-SCF, spin-unpolarized, NLCC-free). See reports/hole_expansion/benchmark_vs_pbe.txt. Verdict:
NOT beating PBE on the clean closed-shell test (SIMPLE 92/116 vs PBE 49 vs rSCAN 16 mHa)."""
import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.gga_pbe import GGA_PBE
from atom.xc.meta_scan import SCAN, rSCANParameters
from atom.xc.evaluator import DensityData
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf
CLOSED=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_n512.npz")
IN=[2,4,10,12,18,20,30,36,48,54]                 # closed-subshell = SIMPLE in-domain (in refs)
OUT=[3,5,6,7,8,9,14,15,16,17,26,29]              # open-shell = out-of-domain (transfer)
def tau_hf(hf,r):
    g=np.asarray(hf['g_sorted']); occ=np.asarray(hf['occ']); l=np.asarray(hf['l_values']); rs=np.asarray(hf['r_sorted']); o=np.argsort(rs); rs=rs[o]; g=g[o]
    gi=np.array([np.interp(r,rs,g[:,i]) for i in range(g.shape[1])]).T; dg=np.gradient(gi,r,axis=0)
    return (1/(8*np.pi))*np.sum(occ*(dg**2+(l*(l+1))*(gi/r[:,None])**2),axis=1)
def row(Z):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    grad=np.gradient(rho,r); ew=4*np.pi*r**2*w; D=np.eye(len(r))
    Fs=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,refs_path=CLOSED))
    cp=np.array([op@rho for op in Fs._ops]); g=Fs._grad_op@rho
    es=float(np.sum(Fs.energy_weights*rho*Fs._kernel_eps(cp,rho,g)))
    ep=float(np.sum(ew*rho*GGA_PBE(derivative_matrix=D,r_quad=r).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad)).e_generic))
    er=float(np.sum(ew*rho*SCAN(derivative_matrix=D,r_quad=r,params=rSCANParameters()).compute_exchange_generic(DensityData(rho=rho,grad_rho=grad,tau=tau_hf(hf,r))).e_generic))
    return Ehf,1e3*(es-Ehf),1e3*(ep-Ehf),1e3*(er-Ehf)
print("valence-only int(rho*eps_x) vs Ehf (mHa). SIMPLE = closed-n512 refs (ridge 1e-2, l0=.7 l1=.5).")
print("%4s %4s %9s %8s %8s %8s"%("atom","dom","Ehf","SIMPLE","PBE","rSCAN"))
agg={"in":[],"out":[]}
for dom,Zs in [("in",IN),("out",OUT)]:
    for Z in Zs:
        Ehf,es,ep,er=row(Z); print("%4s %4s %9.4f %+8.0f %+8.0f %+8.0f"%(nm(Z),dom,Ehf,es,ep,er),flush=True); agg[dom].append((abs(es),abs(ep),abs(er)))
for dom in ("in","out"):
    a=np.array(agg[dom]); print("MAE %3s (n=%d): SIMPLE %.0f  PBE %.0f  rSCAN %.0f"%(dom,len(a),a[:,0].mean(),a[:,1].mean(),a[:,2].mean()))
allv=np.array(agg["in"]+agg["out"]); print("MAE ALL (n=%d): SIMPLE %.0f  PBE %.0f  rSCAN %.0f"%(len(allv),allv[:,0].mean(),allv[:,1].mean(),allv[:,2].mean()))
