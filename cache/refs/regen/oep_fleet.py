"""Cache EXX-OEP exact-exchange potentials (the v_x targets for Stage-2 potential fitting) for the
closed-shell atoms where OEP converges. -> cache/refs/oep/oep_Z{NN}.npz {r, rho, vx_oep, Ex, converged}.
EXX-OEP does not converge for every PSP atom (e.g. Ne); we cache whatever does."""
import os, sys, time
import numpy as np
np.seterr(all="ignore")
_HERE=os.path.dirname(os.path.abspath(__file__)); _REPO=os.path.abspath(os.path.join(_HERE,"..","..",".."))
sys.path.insert(0,_REPO)
from atom import AtomicDFTSolver
from atom.utils.periodic import atomic_number_to_name as nm
OUT=os.path.join(_HERE,"..","oep"); os.makedirs(OUT,exist_ok=True)
CLOSED=[2,4,10,12,18,20,30]   # He Be Ne Mg Ar Ca Zn (closed-shell; try, cache converged)
for Z in CLOSED:
    p=os.path.join(OUT,f"oep_Z{Z:02d}.npz")
    if os.path.exists(p): print(f"{nm(Z)} cached",flush=True); continue
    t=time.time()
    try:
        r=AtomicDFTSolver(atomic_number=Z,xc_functional="EXX",all_electron_flag=False,max_scf_iterations=150).solve()
        conv=bool(r["converged"]); rr=np.asarray(r["quadrature_nodes"]); o=np.argsort(rr)
        np.savez(p, Z=Z, r=rr[o], rho=np.asarray(r["rho"])[o], vx_oep=np.asarray(r["v_x_local"])[o],
                 Ex=float(r["energy_components"].exchange), converged=conv)
        print(f"{nm(Z)} converged={conv} t={time.time()-t:.0f}s",flush=True)
    except Exception as e:
        print(f"{nm(Z)} FAILED {type(e).__name__} t={time.time()-t:.0f}s",flush=True)
print("OEP FLEET DONE",flush=True)
