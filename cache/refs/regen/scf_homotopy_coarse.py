import os,sys,numpy as np,time; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P
R=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
LAM=[0.0,0.5,1.0]
def step(Z,lam,rho0):
    s=AtomicDFTSolver(atomic_number=Z,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=200,scf_tolerance=1e-6,
        xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=R,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False,ref_scale=lam))
    return s.solve(rho_initial=rho0)
def homotopy(Z,nm):
    t=time.time(); rho=None; traj=[]
    for lam in LAM:
        try:
            r=step(Z,lam,rho); rho=np.asarray(r['rho']); traj.append((lam,bool(r['converged']),r.get('rho_residual',np.nan)))
        except Exception as e: traj.append((lam,'EXC',str(e)[:30])); break
    s="  ".join("l=%.2f:%s(%.0e)"%(l,c,rr) if c!='EXC' else "l=%.2f:EXC"%l for l,c,rr in traj)
    print("%-9s %s   (%.0fs)"%(nm,s,time.time()-t),flush=True)
print("HOMOTOPY SCF (ramp ref_scale 0->1, warm-started). Q-functional. (Mg/Ar/Ne failed cold-start before)")
for Z,nm in [(4,'Be'),(10,'Ne'),(12,'Mg'),(18,'Ar')]:
    homotopy(Z,nm)
