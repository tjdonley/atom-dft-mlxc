import os,sys,numpy as np,time; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P
R=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
def step(Z,lam,rho0):
    s=AtomicDFTSolver(atomic_number=Z,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=120,scf_tolerance=1e-6,
        xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=R,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False,ref_scale=lam))
    return s.solve(rho_initial=rho0)
LAM=[0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0]
for Z,nm in [(10,'Ne'),(12,'Mg')]:
    rho=None
    print("%s fine ramp:"%nm,flush=True)
    for lam in LAM:
        try:
            r=step(Z,lam,rho); cv=bool(r['converged']); rr=r.get('rho_residual',np.nan)
            if np.isfinite(rr) and rr<1e2: rho=np.asarray(r['rho'])   # only warm-start from a sane density
            print("   l=%.2f conv=%s rho_res=%.1e"%(lam,cv,rr),flush=True)
            if (not cv) and rr>1e2: print("   -> diverged at l=%.2f; stop"%lam,flush=True); break
        except Exception as e: print("   l=%.2f EXC %s"%(lam,str(e)[:40]),flush=True); break
