import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
QP=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power_Q.npz")
orig=SIMPLE_HOLE_KERNEL_FP.compute_xc; hist=[]
def traced(self,dd):
    out=orig(self,dd); v=out.v_x; hist.append((float(v.min()),float(v.max()))); return out
SIMPLE_HOLE_KERNEL_FP.compute_xc=traced
s=AtomicDFTSolver(atomic_number=4,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=200,
    xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_lQ=0.3,fp_ref_ridge=1e-8,refs_path=QP,use_l2_power=True,use_Q=True,fa_ontop=False,fa_coeff=False)).solve()
print("conv=%s iters=%s rho_residual=%.2e E=%.4f"%(s["converged"],s.get("iterations"),s.get("rho_residual",np.nan),s.get("energy",np.nan)))
h=np.array(hist); print("compute_xc calls=%d"%len(h))
print("v_x range over LAST 20 calls: min in [%.2f,%.2f], max in [%.2f,%.2f]"%(h[-20:,0].min(),h[-20:,0].max(),h[-20:,1].min(),h[-20:,1].max()))
print("v_x[max] last 12 calls:",np.round(h[-12:,1],2))
