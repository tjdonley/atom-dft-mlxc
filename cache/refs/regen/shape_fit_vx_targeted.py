import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P, _bound
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_oep
REF=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001.npz")
def oepmis(Z,coef_override=None):
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12); vxo=np.asarray(d['vx_oep'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF,gauge_fix=False))
    if coef_override is not None: F._fp_coef=coef_override
    v=F.compute_xc(DensityData(rho=rho)).v_x; m=rho>1e-3; w=rho[m]*r[m]**2; dc=np.sum(w*(v[m]-vxo[m]))/np.sum(w)
    return np.sqrt(np.sum(w*(v[m]-vxo[m]-dc)**2)/np.sum(w))
def smoothed_coef(Z,eps):
    # build F on Z's grid; H = spatial-roughness of d sigma/ds2 . Cmom (the grad_op^T-amplified v_x channel)
    d=load_oep(Z); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12)
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_ref_ridge=1e-8,refs_path=REF))
    Xn=F._fp_Xnodes; Cmom=F._Cmom; coef=F._fp_coef; cc=coef@Cmom; Kp=F._Kmat(Xn,Xn); dd=Kp@cc; w=F._inv_ell(); nl0=F._n_out-1
    R_ad,_=F._R_ad(rho); kF=(3*np.pi**2*rho)**(1/3); g=F._grad_op@rho; s2,_=_bound((g/(2*kF*rho))**2)
    c_ad=F._c_ad(np.array([op@rho for op in F._ops]),R_ad); cn=c_ad/np.where(np.abs(c_ad[:,:1])>1e-30,c_ad[:,:1],1e-30)
    Xq=F._xfeat(cn,s2); Kq=F._Kmat(Xq,Xn)                       # (grid, Nnodes)
    Pg=-(w[nl0]**2)*(Xq[:,nl0][:,None]-Xn[:,nl0][None,:])*Kq    # d K / d s^2 at grid points
    L=F._grad_op                                                # spatial derivative
    LP=L@Pg                                                     # d/dr ( dsigma/ds2 basis )
    H=LP.T@LP                                                   # roughness penalty on cc
    M=np.linalg.solve(H+eps*np.eye(len(Xn)),np.eye(len(Xn))); MK=M@Kp
    cc_s=MK@np.linalg.solve(Kp@MK+1e-10*np.eye(len(Xn)),dd)     # min-roughness cc matching energy
    coef2=coef+(cc_s-cc)[:,None]*Cmom[None,:]/float(Cmom@Cmom)  # adjust only the Cmom projection
    return coef2
print("v_x-targeted (spatial roughness of dsigma/ds2.Cmom) min-roughness. OEP mismatch. backbone Be.091 Mg.075")
print("exact-interp: Be %.3f  Mg %.3f"%(oepmis(4),oepmis(12)))
for eps in (1e2,1.0,1e-2,1e-4,1e-6):
    print("  eps=%-7s Be %.3f  Mg %.3f"%(eps,oepmis(4,smoothed_coef(4,eps)),oepmis(12,smoothed_coef(12,eps))),flush=True)
