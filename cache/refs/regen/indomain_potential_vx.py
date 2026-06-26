import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hole_refs_full, load_oep, load_hf
SCR=os.environ["SCR"]
def build_own(Z,rho_floor=0.01):
    r=load_hole_refs_full(); m=(r["Z"]==Z)&(np.maximum(np.abs(r["leakQ"]),np.abs(r["leakE"]))<=0.10)&(r["rho"]>=rho_floor)
    rr=np.linspace(1e-3,14,400); F=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr))
    sig=F._rhotilde_lda; X=r["X"][m].copy(); X[:,-1]=_bound(X[:,-1])[0]
    DELTA=r["rt"][m]/(-0.5*r["rho"][m])[:,None]-sig[None,:]
    out=os.path.join(SCR,f"own_Z{Z}.npz"); np.savez(out,X=X,DELTA=DELTA,fp_l0=0.7,fp_l1=0.5); return out,int(m.sum())
def oepmis(Z,refp,ds):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=refp,gauge_fix=False,deriv_smooth=ds))
    v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; w=rho[m]*r[m]**2; dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w)
    return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w))
def emae(Z,refp,ds):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=refp,deriv_smooth=ds))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    return abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-float(hf['Ehf'])))
print("IN-DOMAIN test: fit an atom's OWN exact holes (no cross-atom mix). Does the potential reach OEP?")
print("backbone OEP: Be 0.091, Mg 0.075")
for Z,nm in [(4,'Be'),(12,'Mg')]:
    refp,npts=build_own(Z)
    print("%s own-refs(%d pts): OEP exact-interp %.3f | OEP deriv_smooth=100 %.3f | energy(self) exact %.0f / ds %.0f mHa"
          %(nm,npts,oepmis(Z,refp,0.0),oepmis(Z,refp,100.0),emae(Z,refp,0.0),emae(Z,refp,100.0)),flush=True)
