import os,sys,numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
import atom.xc.simple_hole_expansion as M
from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters as P
L2P=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_rf001_l2power.npz")
orig=SIMPLE_HOLE_KERNEL_FP.compute_xc; cnt=[0]
def traced(self,dd):
    rho=np.asarray(dd.rho); out=orig(self,dd); v=out.v_x; cnt[0]+=1
    if cnt[0]<=14:
        print("  call %2d: rho[min,max]=[%.1e,%.2f] nanRHO=%d | vx[min,max]=[%.2f,%.2f] nanVX=%d ex=%.4f"
              %(cnt[0],rho.min(),rho.max(),np.isnan(rho).sum(),np.nanmin(v),np.nanmax(v),np.isnan(v).sum(),
                float(np.sum(self.energy_weights*np.maximum(rho,1e-12)*out.e_x))),flush=True)
    return out
SIMPLE_HOLE_KERNEL_FP.compute_xc=traced
try:
    AtomicDFTSolver(atomic_number=4,xc_functional="SIMPLE_HOLE_KERNEL_FP",all_electron_flag=False,max_scf_iterations=12,
        xc_params=P(fp_l0=0.7,fp_l1=0.5,fp_l2pow=0.02,fp_ref_ridge=1e-8,refs_path=L2P,use_l2_power=True,lb94_tail=0.1)).solve()
except Exception as e: print("EXC",str(e)[:60])
