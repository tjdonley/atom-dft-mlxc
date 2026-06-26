import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_hf, load_oep
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def minrough_coef(F,eps):
    Xn=F._fp_Xnodes; K=F._Kmat(Xn,Xn); w=F._inv_ell(); N=len(Xn); Delta=K@F._fp_coef
    G=np.zeros((N,N))
    for d in range(Xn.shape[1]):
        Dd=-(w[d]**2)*(Xn[:,d][:,None]-Xn[:,d][None,:])*K; G+=Dd.T@Dd
    M=np.linalg.solve(G+eps*np.eye(N),np.eye(N))          # (G+eps I)^-1
    MK=M@K; KMK=K@MK+1e-10*np.eye(N)                       # constrained: c = M K (K M K)^-1 Delta
    return MK@np.linalg.solve(KMK,Delta)
def run(eps):
    es=[]
    for Z in [4,10,18,12]:
        hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ww=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=ww,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF))
        if eps is not None: F._fp_coef=minrough_coef(F,eps)
        cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
        es.append(abs(1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-Ehf)))
    om=[]
    for Z in (4,12):
        d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
        F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF,gauge_fix=False))
        if eps is not None: F._fp_coef=minrough_coef(F,eps)
        v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; wt=rho[m]*r[m]**2; dc=np.sum(wt*(v[m]-vxo[m]))/np.sum(wt)
        om.append(np.sqrt(np.sum(wt*(v[m]-vxo[m]-dc)**2)/np.sum(wt)))
    return np.mean(es),om
print("CONSTRAINED min-roughness fit (match refs exactly, smooth dsigma/dx). exact=baseline. backbone OEP Be.091 Mg.075")
print("%12s %10s %10s %10s"%("G-reg eps","energyMAE","OEP-Be","OEP-Mg"))
e,om=run(None); print("%12s %10.0f %10.3f %10.3f"%("exact",e,om[0],om[1]),flush=True)
for eps in (1e2,1.0,1e-2,1e-4):
    e,om=run(eps); print("%12s %10.0f %10.3f %10.3f"%(eps,e,om[0],om[1]),flush=True)
