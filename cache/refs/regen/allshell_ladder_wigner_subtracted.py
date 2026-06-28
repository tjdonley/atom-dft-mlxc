import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from cache.refs.loader import load_hf, load_hole_refs_full, available_hf
SCR=os.environ["SCR"]
Xb=np.load(SCR+"/allshell_base.npz")["X"]; Db=np.load(SCR+"/allshell_base.npz")["DELTA"]
X2=np.load(SCR+"/allshell_l2.npz")["X"]; D2=np.load(SCR+"/allshell_l2.npz")["DELTA"]
pool=load_hole_refs_full(); CLOSED={2,4,10,12,18,20,30,36,38,46,48,54,56,80}
ZS=[z for z in available_hf() if z>=2]
rd=np.linspace(1e-3,14,200); wd=np.gradient(rd)
def coefs(maxl,n):     # solve ONCE on a dummy grid (grid-independent kernel solve)
    X,D = (X2[:n],D2[:n]) if maxl==2 else (Xb[:n],Db[:n]); l2=(maxl==2)
    rp=os.path.join(SCR,"as_%d_%d.npz"%(maxl,n)); np.savez(rp,X=X,DELTA=D)
    l1=100.0 if maxl==0 else 0.5; mu=0.0 if maxl==0 else 10/81
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=rd,quadrature_weights=wd,params=P(fp_l0=0.7,fp_l1=l1,fp_mu=mu,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path=rp,use_l2_power=l2,fa_ontop=False,fa_coeff=False))
    return F._fp_Xnodes.copy(),F._fp_coef.copy()
# cache one functional + per-atom data ONCE (use_l2_power=True so the l2 machinery exists)
AT={}; tgt={}
for Z in ZS:
    hf=load_hf(Z); o=np.argsort(np.asarray(hf['r'])); r=np.asarray(hf['r'])[o]; rho=np.maximum(np.asarray(hf['rho'])[o],1e-12); ww=np.asarray(hf['w'])[o]
    F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=ww,params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path="/none.npz",use_l2_power=True,fa_ontop=False,fa_coeff=False))
    cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
    AT[Z]=(F,cp,g,rho)
    m=pool["Z"]==Z; r0=pool["r0"][m]; ef=pool["eps_full"][m]
    tgt[Z]=float(np.sum(4*np.pi*r**2*ww*rho*np.interp(r,np.sort(r0),ef[np.argsort(r0)])))
def evalrow(maxl,n):
    Xn,cf=coefs(maxl,n); l1=100.0 if maxl==0 else 0.5; l2=(maxl==2); ec=[];eo=[]
    for Z in ZS:
        F,cp,g,rho=AT[Z]; F._fp_Xnodes=Xn; F._fp_coef=cf; F._use_l2pow=l2; F._fp_l1=l1; F._inv_ell_key=None
        e=1e3*(float(np.sum(F.energy_weights*rho*F._kernel_eps(cp,rho,g)))-tgt[Z])
        (ec if Z in CLOSED else eo).append(e)
    return np.nanmean(np.abs(ec)),np.nanmean(np.abs(eo))
NS=[256,512,1024,2048]
print("ALL %d atoms, MAE vs WIGNER-SUBTRACTED target (int rho*eps_full) [mHa].  closed|OPEN"%len(ZS))
print("%-7s"%"l\\N"+"".join("%14d"%n for n in NS))
for maxl in (0,1,2):
    cells=[evalrow(maxl,n) for n in NS]; print("l<=%d  "%maxl+"".join("%6.0f|%-7.0f"%(c,o) for c,o in cells),flush=True)
