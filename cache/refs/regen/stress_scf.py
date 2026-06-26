"""SCF convergence stress test of the locked ridge-stabilized functional across p/d/heavy atoms.
Writes a convergence + energy table; energies are exchange-only-SCF-drift-dominated for heavy atoms
(present in reference-free too) so the SIGNAL is robustness of SCF CONVERGENCE, not accuracy."""
import warnings,os,time,numpy as np; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import sys; sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P
from atom.utils.periodic import atomic_number_to_name as nm
from cache.refs.loader import load_hf
RES="/private/tmp/claude-501/-Users-ajm-Library-CloudStorage-Dropbox-GaTech-Andrew-Medford-amedford6-admin-admin-coding-SIMPLE-hole-functional/12298e99-707e-4af4-aa27-716af07a7159/scratchpad/stress.txt"; CL=os.path.abspath("atom/xc/data/kernel_fp_refs_closed_n64.npz")
open(RES,"w").write("Stress test: LOCKED functional (closed_n64, ridge 1e-2, l0=0.7 l1=0.5) SCF. err vs Ehf(restricted)\n")
open(RES,"a").write("%4s %5s %6s %14s %14s\n"%("atom","block","dom","reffree(c,err)","locked(c,err)"))
# (Z, block, in/out-domain)
ATOMS=[(14,"p","out"),(17,"p","out"),(18,"p","in"),(26,"d","out"),(29,"d","out"),
       (30,"d","in"),(36,"p","in"),(48,"d","in"),(54,"p","in"),(80,"d","in")]
def scf(Z,refs):
    p=P(fp_l0=0.7,fp_l1=0.5,refs_path=refs); t=time.time()
    try:
        r=AtomicDFTSolver(atomic_number=Z,xc_functional="SIMPLE_HOLE_KERNEL_FP",xc_params=p,all_electron_flag=False,max_scf_iterations=250).solve()
        return bool(r["converged"]), float(r["energy_components"].exchange), time.time()-t
    except Exception as e: return False, float("nan"), time.time()-t
for Z,blk,dom in ATOMS:
    Ehf=float(load_hf(Z)["Ehf"])
    c0,e0,t0=scf(Z,None); c1,e1,t1=scf(Z,CL)
    with open(RES,"a") as f:
        f.write("%4s %5s %6s  %1s%+6.0f(%4.0fs)  %1s%+6.0f(%4.0fs)\n"%(nm(Z),blk,dom,
                "Y" if c0 else "N",1e3*(e0-Ehf),t0, "Y" if c1 else "N",1e3*(e1-Ehf),t1))
open(RES,"a").write("DONE\n")
