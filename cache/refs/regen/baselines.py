"""Phase B: PBE + rSCAN exchange baselines over all PSP atoms (Z=1-57, 72-83).
Cheap (LDA/GGA cost). Caches refs/base_Z{NN}.npz = {pbe_Ex, rscan_Ex, converged}. Idempotent."""
import os, time, numpy as np
np.seterr(all="ignore")
from atom import AtomicDFTSolver
from atom.utils.periodic import atomic_number_to_name as nm

SCR = "/private/tmp/claude-501/-Users-ajm-Library-CloudStorage-Dropbox-GaTech-Andrew-Medford-amedford6-admin-admin-coding-SIMPLE-hole-functional/a2aa6175-98ec-455a-9307-c93f60e2052e/scratchpad"
REF = SCR + "/refs"
ALL = list(range(1, 58)) + list(range(72, 84))

def dom_for(Z):
    if Z in (3, 11, 19, 37, 55): return 26.0
    if Z in (4, 12, 20, 38, 56): return 24.0
    if Z <= 18: return 18.0
    if Z <= 54: return 20.0
    return 22.0

def is_conv(r):
    v = r.get("converged", None)
    return True if v is None else bool(v)

def run(Z, xc):
    for kw in (dict(domain_size=dom_for(Z), max_scf_iterations=500),
               dict(domain_size=dom_for(Z)+4, max_scf_iterations=800, linear_mixing_alpha1=0.4)):
        try:
            r = AtomicDFTSolver(atomic_number=Z, xc_functional=xc, all_electron_flag=False, **kw).solve()
            return float(r["energy_components"].exchange), is_conv(r)
        except Exception:
            continue
    return float("nan"), False

print(f"PBE+rSCAN baselines over {len(ALL)} PSP atoms.", flush=True)
for Z in ALL:
    p = f"{REF}/base_Z{Z:02d}.npz"
    if os.path.exists(p):
        try:
            d = np.load(p)
            if bool(d["converged"]): print(f"{Z:>3} {nm(Z):>3} cached", flush=True); continue
        except Exception: pass
    t0 = time.time()
    pbe, cp = run(Z, "GGA_PBE")
    rsc, cr = run(Z, "RSCAN")
    np.savez(p, Z=Z, pbe_Ex=pbe, rscan_Ex=rsc, converged=(cp and cr))
    print(f"{Z:>3} {nm(Z):>3} PBE_Ex={pbe:11.4f}({cp}) rSCAN_Ex={rsc:11.4f}({cr}) t={time.time()-t0:6.1f}", flush=True)
print("BASELINES DONE", flush=True)
