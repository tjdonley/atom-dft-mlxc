import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, enclosed_charge_switch, _bound
from cache.refs.loader import load_hf
DATA="atom/xc/data"; IN=[2,4,10,12,18,20,30,36,48,54]
REF=os.path.abspath(DATA+"/kernel_fp_refs_closed_n1098.npz")
def eps_tail(F,rho,bulk,R_ad,cp,fa,proj):
    Bmom,R0,Cmom=F._Bmom,F._R0,F._Cmom
    if not proj and not fa: return 2*np.pi*R_ad**2*(bulk@Cmom)
    c_ad=F._c_ad(cp,R_ad); d=c_ad/(4*np.pi*R_ad**1.5)[:,None]; Q=4*np.pi*R_ad**3*(d@Bmom); Qs=np.maximum(Q,1e-12)
    W=enclosed_charge_switch(0.5*Q) if fa else np.zeros_like(Q)
    fah=-d/Qs[:,None]; coeffs=(1-W)[:,None]*bulk+W[:,None]*fah; ontop=(1-W)*(-0.5*rho)+W*(-rho/Qs)
    a_row=4*np.pi*(R_ad**3)[:,None]*Bmom[None,:]; row0=np.sum(a_row*coeffs,1); row1=coeffs@R0
    g00=np.sum(a_row*a_row,1); g01=a_row@R0; g11=float(R0@R0); res0=-1-row0; res1=ontop-row1; det=g00*g11-g01**2
    lam0=(g11*res0-g01*res1)/det; lam1=(-g01*res0+g00*res1)/det; coeffs=coeffs+lam0[:,None]*a_row+lam1[:,None]*R0[None,:]
    return 2*np.pi*R_ad**2*(coeffs@Cmom)
print("Decompose in-domain error (n1098, exact interp). err vs Ehf (mHa). cols: kernel-only | +proj(noFA) | +proj+FA(full)")
print("%4s %12s %14s %16s"%("atom","kernel-only","+proj noFA","FULL(+FA)"))
agg={'k':[],'p':[],'f':[]}
for Z in IN:
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); w=np.asarray(hf['w'])[o]; Ehf=float(hf['Ehf'])
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=w,params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho; R_ad,_=F._R_ad(rho); kF=(3*np.pi**2*rho)**(1/3)
    s2,_=_bound((g/(2*kF*rho))**2); c_ad=F._c_ad(cp,R_ad); cn=c_ad/np.where(np.abs(c_ad[:,:1])>1e-30,c_ad[:,:1],1e-30)
    rt=F._rhotilde_lda[None,:]+F._Kmat(F._xfeat(cn,s2),F._fp_Xnodes)@F._fp_coef; bulk=(-0.5*rho)[:,None]*rt
    ek=1e3*(float(np.sum(F.energy_weights*rho*eps_tail(F,rho,bulk,R_ad,cp,False,False)))-Ehf)
    ep=1e3*(float(np.sum(F.energy_weights*rho*eps_tail(F,rho,bulk,R_ad,cp,False,True)))-Ehf)
    ef=1e3*(float(np.sum(F.energy_weights*rho*eps_tail(F,rho,bulk,R_ad,cp,True,True)))-Ehf)
    agg['k'].append(abs(ek)); agg['p'].append(abs(ep)); agg['f'].append(abs(ef))
    print("%4d %12.0f %14.0f %16.0f"%(Z,ek,ep,ef),flush=True)
print("MAE  %11.0f %14.0f %16.0f"%(np.mean(agg['k']),np.mean(agg['p']),np.mean(agg['f'])))
