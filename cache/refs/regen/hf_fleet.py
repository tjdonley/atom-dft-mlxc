"""Phase A: HF reference fleet over all PSP atoms (Z=1-57, 72-83).
Caches one npz per atom under refs/hf_Z{NN}.npz with density, energies, convergence, and the
exact orbitals (for later exact-hole reconstruction). Idempotent: skips atoms already converged.
Sequential (heavy HF is CPU-bound; concurrency caused 10x slowdowns). Retry ladder for stragglers."""
import os, time, numpy as np
np.seterr(all="ignore")
from atom import AtomicDFTSolver
from atom.xc import orbital_hole as oh
from atom.utils.periodic import atomic_number_to_name as nm

SCR = "/private/tmp/claude-501/-Users-ajm-Library-CloudStorage-Dropbox-GaTech-Andrew-Medford-amedford6-admin-admin-coding-SIMPLE-hole-functional/a2aa6175-98ec-455a-9307-c93f60e2052e/scratchpad"
REF = SCR + "/refs"

ALL = list(range(1, 58)) + list(range(72, 84))
CLOSED = [2, 4, 10, 12, 18, 20, 30, 36, 38, 48, 54, 56, 80]      # do first: refs + fast wins
ORDER = CLOSED + [z for z in ALL if z not in CLOSED]

def dom_for(Z):
    if Z in (3, 11, 19, 37, 55):      return 26.0   # alkalis: diffuse valence
    if Z in (4, 12, 20, 38, 56):      return 24.0   # alkaline earths
    if Z <= 18:                       return 18.0
    if Z <= 54:                       return 20.0
    return 22.0

def is_conv(r):
    oc = r.get("outer_converged", None); ic = r.get("converged", None)
    v = oc if oc is not None else ic
    return True if v is None else bool(v)

def attempts(Z):
    D = dom_for(Z)
    return [
        dict(domain_size=D,     max_scf_iterations=400, max_scf_iterations_outer=80),
        dict(domain_size=D,     max_scf_iterations=700, max_scf_iterations_outer=150),
        dict(domain_size=D+4,   max_scf_iterations=700, max_scf_iterations_outer=150,
             use_pulay_mixing=True, linear_mixing_alpha1=0.4),
        dict(domain_size=max(D-3, 12.0), max_scf_iterations=900, max_scf_iterations_outer=200,
             linear_mixing_alpha1=0.3),
    ]

def cached_ok(Z):
    p = f"{REF}/hf_Z{Z:02d}.npz"
    if not os.path.exists(p): return False
    try:
        d = np.load(p, allow_pickle=True)
        return bool(d["converged"])
    except Exception:
        return False

print(f"HF fleet over {len(ORDER)} PSP atoms. Order: closed-shell first.", flush=True)
for Z in ORDER:
    if cached_ok(Z):
        print(f"{Z:>3} {nm(Z):>3}  cached, skip", flush=True); continue
    t0 = time.time(); done = False
    for ai, kw in enumerate(attempts(Z)):
        try:
            r = AtomicDFTSolver(atomic_number=Z, xc_functional="HF", all_electron_flag=False, **kw).solve()
            conv = is_conv(r)
            ev = r["energy_components"]
            Ehf = float(ev.hf_exchange); Etot = float(ev.total)
            rr = np.asarray(r["quadrature_nodes"]); rho = np.asarray(r["rho"]); w = np.asarray(r["quadrature_weights"])
            try:
                r_s, g_s, occ, lval = oh.extract_orbitals(r)
            except Exception as e:
                r_s = g_s = occ = lval = None
            np.savez(f"{REF}/hf_Z{Z:02d}.npz", Z=Z, converged=conv, Ehf=Ehf, Etot=Etot,
                     domain=kw["domain_size"], attempt=ai,
                     r=rr, rho=rho, w=w,
                     r_sorted=(r_s if r_s is not None else np.array([])),
                     g_sorted=(g_s if g_s is not None else np.array([])),
                     occ=(occ if occ is not None else np.array([])),
                     l_values=(lval if lval is not None else np.array([])))
            print(f"{Z:>3} {nm(Z):>3}  conv={str(conv):>5} Ex={Ehf:11.4f} dom={kw['domain_size']:.0f} "
                  f"try={ai} t={time.time()-t0:7.1f}", flush=True)
            if conv:
                done = True; break
        except Exception as e:
            print(f"{Z:>3} {nm(Z):>3}  try={ai} ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
    if not done:
        print(f"{Z:>3} {nm(Z):>3}  *** did not converge after {len(attempts(Z))} attempts ***", flush=True)
print("FLEET DONE", flush=True)
