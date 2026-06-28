import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf, load_hole_refs_full
SCR=os.environ["SCR"]
base=np.load("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz"); Xb=base["X"]; Db=base["DELTA"]
pool=load_hole_refs_full(); lk=np.maximum(np.abs(pool["leakQ"]),np.abs(pool["leakE"]))
rr=np.linspace(1e-3,14,400); F0=SIMPLE_HOLE_KERNEL_FP(r_quad=rr,quadrature_weights=np.gradient(rr)); sig=F0._rhotilde_lda
def block(Z):   # build that atom's own leak-filtered holes as 12-dim (cn[1:],s2,p2,Qb) reference rows
    m=(pool["Z"]==Z)&(lk<=0.10)&(pool["rho"]>=0.01); r0=pool["r0"][m]
    Xcn=pool["X"][m][:,:9]; s2=_bound(pool["X"][m][:,9])[0]
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(use_l2_power=True))
    cp=np.array([op@rho for op in F._ops]); R_ad,_=F._R_ad(rho); c_ad=F._c_ad(cp,R_ad)
    p2=np.interp(r0,r,F._l2_power_feat(rho,R_ad,c_ad[:,0])); d=c_ad/(4*np.pi*R_ad[:,None]**1.5)
    Qb=np.interp(r0,r,_bound(4*np.pi*R_ad**3*(d@F._Bmom))[0])
    X=np.column_stack([Xcn,s2,p2,Qb]); DELTA=pool["rt"][m]/(-0.5*pool["rho"][m])[:,None]-sig[None,:]
    return X,DELTA,int(m.sum())
Xn,Dn,nN=block(7); Xc,Dc,nC=block(20); print("added blocks: N %d pts, Ca %d pts"%(nN,nC),flush=True)
def mkref(tag,X,D):
    p=os.path.join(SCR,"ref_%s.npz"%tag); np.savez(p,X=X,DELTA=D); return p
refs={"base":mkref("base",Xb,Db),"+N":mkref("N",np.vstack([Xb,Xn]),np.vstack([Db,Dn])),
      "+Ca":mkref("Ca",np.vstack([Xb,Xc]),np.vstack([Db,Dc])),"+N+Ca":mkref("NCa",np.vstack([Xb,Xn,Xc]),np.vstack([Db,Dn,Dc]))}
def err(refp,Z):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]
    G=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=refp,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False))
    cp=np.array([op@rho for op in G._ops]); g=G._grad_op@rho
    return 1e3*(float(np.sum(G.energy_weights*rho*G._kernel_eps(cp,rho,g)))-float(hf['Ehf']))
ATOMS=[(7,'N(open)'),(20,'Ca(clo)'),(10,'Ne'),(12,'Mg'),(18,'Ar')]
print("%-10s %8s %8s %8s %8s"%("atom","base","+N","+Ca","+N+Ca"))
for Z,nm in ATOMS:
    print("%-10s %8.0f %8.0f %8.0f %8.0f"%(nm,err(refs["base"],Z),err(refs["+N"],Z),err(refs["+Ca"],Z),err(refs["+N+Ca"],Z)),flush=True)
