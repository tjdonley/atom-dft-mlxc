import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from cache.refs.loader import load_oep
L2P=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz")
d=load_oep(12); r=np.asarray(d['r']); o=np.argsort(r); r=r[o]; rho=np.maximum(np.asarray(d['rho'])[o],1e-12)
F=SIMPLE_HOLE_KERNEL_FP(r_quad=r,quadrature_weights=np.gradient(r),params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path=L2P,use_l2_power=True,gauge_fix=False))
ew=F.energy_weights; ewrho=ew*rho; cprime=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
eps=F._kernel_eps(cprime,rho,g)
acc=np.zeros_like(rho)
for nn in range(len(F._ops)):
    h=1e-6*(np.abs(cprime[nn])+1e-8); vp=cprime.copy(); vp[nn]+=h; vm=cprime.copy(); vm[nn]-=h
    acc+=F._ops[nn].T@(ewrho*(F._kernel_eps(vp,rho,g)-F._kernel_eps(vm,rho,g))/(2*h))
hr=1e-6*(rho+1e-8); deps_drho0=(F._kernel_eps(cprime,rho+hr,g)-F._kernel_eps(cprime,rho-hr,g))/(2*hr)
hg=1e-6*(np.abs(g)+1e-8); deps_dg=(F._kernel_eps(cprime,rho,g+hg)-F._kernel_eps(cprime,rho,g-hg))/(2*hg)
T_eps=eps; T_rho=rho*deps_drho0; T_mono=acc/ew; T_grad=F._grad_op.T@(ewrho*deps_dg)/ew
def rng(x): return "[%.1f,%.1f] |max|=%.1f"%(x.min(),x.max(),np.abs(x).max())
print("Mg v_x term decomposition (range over grid):")
print("  eps           :",rng(T_eps))
print("  rho*deps_drho0:",rng(T_rho))
print("  monopole acc  :",rng(T_mono))
print("  gradient grad :",rng(T_grad))
print("  TOTAL v_x     :",rng(T_eps+T_rho+T_mono+T_grad))
# where is the spike?
tot=T_eps+T_rho+T_mono+T_grad; i=np.argmax(np.abs(tot))
print("  spike at r=%.3f rho=%.2e: eps=%.1f rho_t=%.1f mono=%.1f grad=%.1f"%(r[i],rho[i],T_eps[i],T_rho[i],T_mono[i],T_grad[i]))
print("  grad term: nnz spikes |T_grad|>5 at %d pts, r="%np.sum(np.abs(T_grad)>5),np.round(r[np.abs(T_grad)>5][:8],2))
