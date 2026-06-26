import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, enclosed_charge_switch, _bound
from cache.refs.loader import load_hf
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_n1098.npz")
def project_energy(F,rho,bulk,R_ad,cp):
    Bmom,R0,Cmom=F._Bmom,F._R0,F._Cmom; c_ad=F._c_ad(cp,R_ad); d=c_ad/(4*np.pi*R_ad**1.5)[:,None]
    Q=4*np.pi*R_ad**3*(d@Bmom); Qs=np.maximum(Q,1e-12); W=enclosed_charge_switch(0.5*Q)
    fah=-d/Qs[:,None]; coeffs=(1-W)[:,None]*bulk+W[:,None]*fah; ontop=(1-W)*(-0.5*rho)+W*(-rho/Qs)
    a_row=4*np.pi*(R_ad**3)[:,None]*Bmom[None,:]; row0=np.sum(a_row*coeffs,1); row1=coeffs@R0
    g00=np.sum(a_row*a_row,1); g01=a_row@R0; g11=float(R0@R0); res0=-1-row0; res1=ontop-row1; det=g00*g11-g01**2
    lam0=(g11*res0-g01*res1)/det; lam1=(-g01*res0+g00*res1)/det; coeffs=coeffs+lam0[:,None]*a_row+lam1[:,None]*R0[None,:]
    return 2*np.pi*R_ad**2*(coeffs@Cmom)
def run(Z,K=40):
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho; R_ad,_=F._R_ad(rho); kF=(3*np.pi**2*rho)**(1/3)
    s2,_=_bound((g/(2*kF*rho))**2); c_ad=F._c_ad(cp,R_ad); cn=c_ad/np.where(np.abs(c_ad[:,:1])>1e-30,c_ad[:,:1],1e-30)
    Xq=F._xfeat(cn,s2); Xn=F._fp_Xnodes
    rt_glob=F._rhotilde_lda[None,:]+F._Kmat(Xq,Xn)@F._fp_coef
    eg=1e3*(float(np.sum(F.energy_weights*rho*project_energy(F,rho,(-0.5*rho)[:,None]*rt_glob,R_ad,cp)))-Ehf)
    DELTA_nodes=F._Kmat(Xn,Xn)@F._fp_coef
    w_ell=F._inv_ell(); Xnw=Xn*w_ell; rt_loc=np.zeros_like(rt_glob)
    for i in range(len(rho)):
        d2=np.sum((Xnw-Xq[i]*w_ell)**2,1); loc=np.concatenate([[0,1],2+np.argsort(d2[2:])[:K]])
        Kl=F._Kmat(Xn[loc],Xn[loc])+1e-8*np.eye(len(loc)); cl=np.linalg.solve(Kl,DELTA_nodes[loc])
        rt_loc[i]=F._rhotilde_lda+F._Kmat(Xq[i][None,:],Xn[loc])[0]@cl
    el=1e3*(float(np.sum(F.energy_weights*rho*project_energy(F,rho,(-0.5*rho)[:,None]*rt_loc,R_ad,cp)))-Ehf)
    return eg,el
print("global vs LOCAL(k=40) in-domain err vs Ehf (mHa), n1098. PBE 49.")
print("%4s %12s %12s"%("atom","global","local-k40")); gs=[];ls=[]
for Z in [10,18,36,54]:
    eg,el=run(Z); gs.append(abs(eg)); ls.append(abs(el)); print("%4d %12.0f %12.0f"%(Z,eg,el),flush=True)
print("MAE  %11.0f %12.0f"%(np.mean(gs),np.mean(ls)))
