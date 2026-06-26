import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf, load_hole_refs_full, load_oep
SCR=os.environ["SCR"]
# build rf001 + l=2 (t^2) reference set
z=np.load("atom/xc/data/kernel_fp_refs_closed_rf001.npz"); X=z["X"]; DELTA=z["DELTA"]; idx=z["idx"]
pool=load_hole_refs_full(); Zr=pool["Z"][idx]; r0r=pool["r0"][idx]
_c={}
def t2atom(Z):
    if Z in _c: return _c[Z]
    hf=load_hf(Z); o=np.argsort(np.asarray(hf["r"])); r=np.asarray(hf["r"])[o]; rho=np.maximum(np.asarray(hf["rho"])[o],1e-12); w=np.asarray(hf["w"])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(use_l2=True))
    kF=(3*np.pi**2*rho)**(1/3); t2,_=_bound((F._l2_op@rho/(4.0*kF**2*rho))**2); _c[Z]=(r,t2); return _c[Z]
t2=np.array([np.interp(r0r[i],*t2atom(int(Zr[i]))) for i in range(len(idx))])
Xl2=np.hstack([X,t2[:,None]]); outp=os.path.join(SCR,"rf001_l2.npz")
np.savez(outp,X=Xl2,DELTA=DELTA,fp_l0=0.7,fp_l1=0.5); print("built rf001+l2 (t^2), X dims %d->%d"%(X.shape[1],Xl2.shape[1]))
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def oepmis(Z,refp,use_l2):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=refp,use_l2=use_l2,gauge_fix=False))
    v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; w=rho[m]*r[m]**2; dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w)
    return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w))
def emae(refp,use_l2,atoms=[4,10,18,12]):
    es=[]
    for Z in atoms:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=refp,use_l2=use_l2))
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf']))))
    return np.mean(es)
print("Does the richer invariant (l=2 t^2) lift the contaminated cross-atom potential? backbone Be .091 Mg .075; in-domain Be .27")
print("  cn+s^2  (current rf001): OEP Be %.3f Mg %.3f | energyMAE %.0f"%(oepmis(4,REF,False),oepmis(12,REF,False),emae(REF,False)))
print("  cn+s^2+l2 (rf001+t^2):   OEP Be %.3f Mg %.3f | energyMAE %.0f"%(oepmis(4,outp,True),oepmis(12,outp,True),emae(outp,True)),flush=True)
