import os,sys,time; import numpy as np; import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore"); sys.path.insert(0,".")
from atom import AtomicDFTSolver
from atom.utils.periodic import atomic_number_to_name as nm
OUT="cache/refs/oep"
def dom(Z): return 24.0 if Z in (20,38,56) else 18.0 if Z<=18 else 20.0 if Z<=54 else 22.0
# settings ladder (try in order; cache first converged): vary oep_basis_number, outer iters, mixing, grid
def settings(Z):
    D=dom(Z)
    return [
      dict(oep_basis_number=8,  max_scf_iterations_outer=80, max_scf_iterations=300, domain_size=D),
      dict(oep_basis_number=6,  max_scf_iterations_outer=120,max_scf_iterations=400, domain_size=D, linear_mixing_alpha1=0.4),
      dict(oep_basis_number=12, max_scf_iterations_outer=120,max_scf_iterations=400, domain_size=D, polynomial_order=37),
      dict(oep_basis_number=5,  max_scf_iterations_outer=150,max_scf_iterations=500, domain_size=D+4, linear_mixing_alpha1=0.3),
      dict(oep_basis_number=10, max_scf_iterations_outer=150,max_scf_iterations=500, domain_size=D, polynomial_order=41, linear_mixing_alpha1=0.4),
    ]
def already(Z):
    p=os.path.join(OUT,f"oep_Z{Z:02d}.npz")
    return os.path.exists(p) and bool(np.load(p)["converged"])
ATOMS=[10,18,20,30,36,54]   # Ne Ar Ca Zn Kr Xe
for Z in ATOMS:
    if already(Z): print(f"{nm(Z)} already converged",flush=True); continue
    done=False
    for i,kw in enumerate(settings(Z)):
        t=time.time()
        try:
            r=AtomicDFTSolver(atomic_number=Z,xc_functional="EXX",all_electron_flag=False,**kw).solve()
            cv=bool(r["converged"])
            print(f"  {nm(Z)} set#{i} basis={kw['oep_basis_number']} conv={cv} ({time.time()-t:.0f}s)",flush=True)
            if cv:
                rr=np.asarray(r["quadrature_nodes"]); o=np.argsort(rr)
                np.savez(os.path.join(OUT,f"oep_Z{Z:02d}.npz"),Z=Z,r=rr[o],rho=np.asarray(r["rho"])[o],
                         vx_oep=np.asarray(r["v_x_local"])[o],Ex=float(r["energy_components"].exchange),converged=True)
                print(f"  -> CACHED {nm(Z)}",flush=True); done=True; break
        except Exception as e: print(f"  {nm(Z)} set#{i} EXC {type(e).__name__}:{str(e)[:40]} ({time.time()-t:.0f}s)",flush=True)
    if not done: print(f"  {nm(Z)} NOT converged by any setting",flush=True)
print("SWEEP DONE",flush=True)
