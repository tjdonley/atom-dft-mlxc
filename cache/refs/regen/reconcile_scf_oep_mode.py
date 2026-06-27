import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
from atom.xc.evaluator import DensityData
from cache.refs.loader import load_oep
L2P=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz")
cap={}
orig=SIMPLE_HOLE_KERNEL_FP.compute_xc
def traced(self,dd):
    cap['rho']=np.asarray(dd.rho).copy(); cap['r']=self._r_grid.copy(); cap['F']=self; return orig(self,dd)
SIMPLE_HOLE_KERNEL_FP.compute_xc=traced
AtomicDFTSolver(atomic_number=12,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=60,
    xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path=L2P,use_l2_power=True)).solve()
F=cap['F']; rho=np.maximum(cap['rho'],1e-12); r=cap['r']
print("SCF(Mg, all_elec=False) density: r[0]=%.4f rho(r->0)=%.2f rho@0.05=%.2f Nelec=%.2f"%(r[0],rho[np.argmin(r)],np.interp(0.05,np.sort(r),rho[np.argsort(r)]),4*np.pi*np.trapz((rho*r**2)[np.argsort(r)],np.sort(r))))
o=load_oep(12); print("OEP(Mg)     density: rho(r->0)=%.2f rho@0.05=%.2f Nelec=%.2f"%(np.asarray(o['rho'])[np.argmin(o['r'])],np.interp(0.05,np.sort(o['r']),np.asarray(o['rho'])[np.argsort(o['r'])]),4*np.pi*np.trapz((np.asarray(o['rho'])*np.asarray(o['r'])**2)[np.argsort(o['r'])],np.sort(o['r']))))
# decompose v_x on the SCF's OWN density
ew=F.energy_weights; ewrho=ew*rho; cp=np.array([op@rho for op in F._ops]); g=F._grad_op@rho
eps=F._kernel_eps(cp,rho,g); acc=np.zeros_like(rho)
for nn in range(len(F._ops)):
    h=1e-6*(np.abs(cp[nn])+1e-8); vp=cp.copy(); vp[nn]+=h; vm=cp.copy(); vm[nn]-=h
    acc+=F._ops[nn].T@(ewrho*(F._kernel_eps(vp,rho,g)-F._kernel_eps(vm,rho,g))/(2*h))
hr=1e-6*(rho+1e-8); dr0=(F._kernel_eps(cp,rho+hr,g)-F._kernel_eps(cp,rho-hr,g))/(2*hr)
hg=1e-6*(np.abs(g)+1e-8); dg=(F._kernel_eps(cp,rho,g+hg)-F._kernel_eps(cp,rho,g-hg))/(2*hg)
Tg=F._grad_op.T@(ewrho*dg)/ew; Tm=acc/ew; Tr=rho*dr0
def rg(x): return "[%.1f,%.1f]"%(x.min(),x.max())
print("SCF v_x terms: eps%s rho_t%s mono%s GRAD%s"%(rg(eps),rg(Tr),rg(Tm),rg(Tg)))
tot=eps+Tr+Tm+Tg; i=np.argmax(np.abs(tot)); rs=np.sort(r)
print("spike at r=%.3f rho=%.2f: grad=%.1f (frac of |spike| from grad: %.0f%%)"%(r[i],rho[i],Tg[i],100*abs(Tg[i])/abs(tot[i])))
