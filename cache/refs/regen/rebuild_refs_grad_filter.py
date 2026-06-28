import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from cache.refs.loader import load_hf, load_hole_refs_full
SCR=os.environ["SCR"]; GF=0.6
z=np.load("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz"); X=z["X"].copy(); D=z["DELTA"]; idx=z["idx"]
pool=load_hole_refs_full(); Zr=pool["Z"][idx]; r0r=pool["r0"][idx]
# recompute the s^2 column (index 9) with the FILTERED gradient operator
for Z in sorted(set(Zr.tolist())):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(grad_filter=GF))
    kF=(3*np.pi**2*rho)**(1/3); s2=_bound((F._grad_op@rho/(2*kF*rho))**2)[0]
    m=Zr==Z; X[m,9]=np.interp(r0r[m],r,s2)
out=os.path.join(SCR,"rf001_l2pQ_gf06.npz"); np.savez(out,X=X,DELTA=D,idx=idx); print("rebuilt refs with gf=%.2f s^2"%GF,flush=True)
def emae(refp,gf,atoms=[2,4,10,12,18,36,54,80]):
    es=[]
    for Z in atoms:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=refp,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False,grad_filter=gf))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf']))))
    return np.mean(es)
R0=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
print("energy MAE (8 closed):")
print("  gf=0  stale refs (baseline best): %.0f"%emae(R0,0.0))
print("  gf=0.6 STALE refs:                %.0f"%emae(R0,GF))
print("  gf=0.6 REBUILT refs (consistent): %.0f"%emae(out,GF),flush=True)
