import os,sys,numpy as np,time; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P
R=os.path.abspath("atom/xc/data/kernel_fp_refs_allshell_l2power_Q.npz")
def scf(Z,nm):
    t=time.time()
    try:
        s=AtomicDFTSolver(atomic_number=Z,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=200,scf_tolerance=1e-6,
            xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=R,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False)).solve()
        print("  %-9s conv=%s iters=%s rho_res=%.1e E=%.4f (%.0fs)"%(nm,s["converged"],s.get("iterations"),s.get("rho_residual",np.nan),s.get("energy",np.nan),time.time()-t),flush=True)
    except Exception as e: print("  %-9s EXC %s"%(nm,str(e)[:50]),flush=True)
print("Most-complex functional: l2power + Q + all-shell refs (2048) + no FA, scf_tol=1e-6")
for Z,nm in [(4,'Be(clo)'),(10,'Ne(clo)'),(12,'Mg(clo)'),(6,'C(open)')]:
    scf(Z,nm)
