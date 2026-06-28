import os,sys,numpy as np,time; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P
R=os.path.join(os.environ["SCR"],"rf001_l2pQ_gf06.npz"); GF=0.6
def step(Z,lam,rho0):
    s=AtomicDFTSolver(atomic_number=Z,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=150,scf_tolerance=1e-6,
        xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=R,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False,grad_filter=GF,ref_scale=lam))
    return s.solve(rho_initial=rho0)
print("COMBINED: grad_filter=0.6 (smoother operator) + homotopy (ref_scale ramp, warm-started). scf_tol=1e-6")
for Z,nm in [(10,'Ne'),(12,'Mg'),(18,'Ar')]:
    rho=None; traj=[]; t=time.time()
    for lam in [0.0,0.3,0.6,1.0]:
        try:
            r=step(Z,lam,rho); rr=r.get('rho_residual',np.nan)
            if np.isfinite(rr) and rr<1e2: rho=np.asarray(r['rho'])
            traj.append((lam,bool(r['converged']),rr))
        except Exception as e: traj.append((lam,'EXC',str(e)[:25])); break
    print("  %-4s %s (%.0fs)"%(nm,"  ".join("l=%.1f:%s(%.0e)"%(l,c,v) if c!='EXC' else "l=%.1f:EXC"%l for l,c,v in traj),time.time()-t),flush=True)
