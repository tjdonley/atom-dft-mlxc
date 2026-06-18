"""Phase-D exchange benchmark: the SIMPLE exchange holes (bare, GGA, GEA) vs OEP/HF
references and PBE / rSCAN / r2SCAN, on single atoms (exchange comparison).

References (OEP, HF, PBE, rSCAN, r2SCAN) and the bare hole are single SCF solves.
The GGA- and GEA-deformed holes use a TWO-STAGE (frozen-correction) SCF: stage 1
converges the stable bare hole; the outer loop then freezes the deformation
correction dv = v_deformed[rho] - v_bare[rho] on the stable bare-hole potential and
re-converges the inner SCF, iterating dv to self-consistency. This keeps the stiff
deformation kernel out of the inner SCF feedback (writeup App.). The frozen
correction is computed in the solver's native grid order.

Per (atom, functional): exchange energy E_x (energy_components), total energy,
occupied eigenvalues, convergence; v_x(r) is kept for He/Be/Ne. Results are pickled
incrementally; each run is wrapped so one failure does not abort the sweep.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from atom.solver import AtomicDFTSolver  # noqa: E402
from atom.xc.evaluator import DensityData  # noqa: E402
from atom.xc.simple_hole import (  # noqa: E402
    SIMPLE_HOLE_GEA, SIMPLE_HOLE_GGA, SIMPLEHOLEGEAParameters, SIMPLEHOLEGGAParameters)

_OUT = Path(__file__).resolve().parent / "data" / "phase_d_results.pkl"
_VX = Path(__file__).resolve().parent / "data" / "phase_d_vx.npz"
# bump when the deformed-hole calibration changes, so stale GGA/GEA entries auto-recompute
CALIB_VERSION = "v2-netmu"

ATOMS = [("He", 2), ("Be", 4), ("N", 7), ("Ne", 10), ("Mg", 12), ("P", 15), ("Ar", 18)]
VX_ATOMS = {"He", "Be", "Ne"}
REFS = [
    ("OEP",    dict(xc_functional="EXX", use_oep=True, hybrid_mixing_parameter=1.0)),
    ("HF",     dict(xc_functional="HF")),
    ("PBE",    dict(xc_functional="GGA_PBE")),
    ("rSCAN",  dict(xc_functional="RSCAN")),
    ("r2SCAN", dict(xc_functional="R2SCAN")),
]
DEFORMED = {"SIMPLE_HOLE_GGA": (SIMPLE_HOLE_GGA, SIMPLEHOLEGGAParameters),
            "SIMPLE_HOLE_GEA": (SIMPLE_HOLE_GEA, SIMPLEHOLEGEAParameters)}


def _solve(Z, kw):
    res = AtomicDFTSolver(atomic_number=Z, all_electron_flag=False, verbose=False,
                          use_pulay_mixing=True, max_scf_iterations=300, **kw).solve(
                              save_full_spectrum=True, save_energy_density=True)
    ec = res["energy_components"]
    return dict(
        converged=bool(res["converged"]),
        E_total=float(res["energy"]),
        E_x=float(ec.exchange) + float(ec.oep_exchange) + float(ec.hf_exchange),
        E_c=float(ec.correlation),
        eigen=np.asarray(res["eigen_energies"], dtype=float),
        r=np.asarray(res["quadrature_nodes"], dtype=float),     # native (functional) order
        w=np.asarray(res["quadrature_weights"], dtype=float),
        rho=np.maximum(np.asarray(res["rho"], dtype=float), 1e-12),
        v_x=np.asarray(res["v_x_local"], dtype=float),
    )


def _two_stage(Z, name, n_outer=40, tol=2e-3, beta=0.12):
    """Two-stage frozen-correction SCF for the deformed holes [writeup App.].

    Stage 1 converges the stable bare hole. The outer loop then freezes the
    deformation correction dv = v_full[rho] - v_bare[rho] (the pure deformation
    channel, ungauged) on the stable bare-hole potential and re-converges the inner
    SCF. The outer density is UNDER-RELAXED, rho <- (1-beta) rho + beta rho_inner,
    which is required: the raw deformation channel is stiff (large in the low-density
    tail) and the un-relaxed outer iteration diverges. The energy stabilizes to
    <1e-3 within a couple of outer steps; we report the iterate of MINIMUM density
    change (where the frozen dv and the inner density are most mutually consistent),
    which is also robust to the occasional transient excursion of the outer map.
    """
    base = _solve(Z, dict(xc_functional="SIMPLE_HOLE"))         # stage 1: stable bare hole
    if not base["converged"]:
        return {**base, "note": "stage-1 bare hole not converged"}
    Cls, Par = DEFORMED[name]
    r, w, rho = base["r"], base["w"], base["rho"]
    best = None
    for outer in range(n_outer):
        # ungauged full eval; dv = (full - bare) is the PURE deformation channel.
        # The single gauge fix is applied by the inner functional to bare+dv (=full).
        ev = Cls(derivative_matrix=np.zeros((1, r.size, 1)), r_quad=r,
                 quadrature_weights=w, params=Par(gauge_fix=False))
        C = np.array([op @ rho for op in ev._ops])
        dv = ev.compute_xc(DensityData(rho=rho)).v_x - ev._bare_v_x(rho, C)   # frozen correction
        sol = _solve(Z, dict(xc_functional=name, xc_params=Par(external_v=dv)))
        if sol["r"].shape != rho.shape or not sol["converged"]:
            note = "inner SCF not converged" if not sol["converged"] else "grid mismatch"
            return {**(best or sol), "note": note, "outer_iters": outer + 1}
        rho_in = np.maximum(sol["rho"], 1e-12)
        rho_new = (1.0 - beta) * rho + beta * rho_in            # outer under-relaxation
        drho = float(np.max(np.abs(rho_new - rho)) / max(rho.max(), 1e-30))
        rho = rho_new
        sol["outer_iters"] = outer + 1
        sol["outer_drho"] = drho
        if outer > 0 and (best is None or drho < best["outer_drho"]):  # skip the iter-0 transient
            best = sol
        if drho < tol:
            break
    best = best or sol
    if best.get("outer_drho", 1.0) >= tol:
        best["note"] = f"outer loop not fully converged (best drho={best.get('outer_drho'):.1e})"
    return best


def _record(d, keep_vx):
    keys = ("converged", "E_total", "E_x", "E_c", "eigen", "note", "outer_iters", "outer_drho")
    out = {k: d[k] for k in keys if k in d}
    if keep_vx and "v_x" in d:
        out.update(r=d["r"], rho=d["rho"], v_x=d["v_x"])
    return out


def main():
    results = pickle.loads(_OUT.read_bytes()) if _OUT.exists() else {}
    for sym in results:                                         # drop stale deformed entries
        for k in list(DEFORMED):
            if k in results[sym] and results[sym][k].get("calib") != CALIB_VERSION:
                del results[sym][k]
    for sym, Z in ATOMS:
        results.setdefault(sym, {})
        runs = ([(n, lambda kw=kw: _solve(Z, kw)) for n, kw in REFS]
                + [("SIMPLE_HOLE", lambda: _solve(Z, dict(xc_functional="SIMPLE_HOLE")))]
                + [(n, lambda n=n: _two_stage(Z, n)) for n in DEFORMED])
        for name, fn in runs:
            if name in results[sym]:
                continue
            try:
                d = fn()
                rec = _record(d, sym in VX_ATOMS)
                if name in DEFORMED:
                    rec["calib"] = CALIB_VERSION
                results[sym][name] = rec
                tag = "ok" if d.get("converged") else "NOT CONVERGED"
                extra = f" [outer={d['outer_iters']}]" if "outer_iters" in d else ""
                print(f"{sym:>3} {name:<16} E_x={d['E_x']:+.4f}  {tag}{extra}", flush=True)
            except Exception as e:
                results[sym][name] = {"error": repr(e)}
                print(f"{sym:>3} {name:<16} ERROR: {e!r}", flush=True)
            _OUT.parent.mkdir(exist_ok=True)
            _OUT.write_bytes(pickle.dumps(results))
    vx = {}
    for sym in VX_ATOMS:
        for name in ("OEP", "SIMPLE_HOLE", "SIMPLE_HOLE_GGA", "SIMPLE_HOLE_GEA"):
            d = results.get(sym, {}).get(name, {})
            if "v_x" in d:
                o = np.argsort(d["r"])
                vx[f"{sym}_{name}_r"] = d["r"][o]
                vx[f"{sym}_{name}_vx"] = d["v_x"][o]
                vx[f"{sym}_{name}_rho"] = d["rho"][o]
    if vx:
        np.savez(_VX, **vx)
    print("DONE; results ->", _OUT.name, "and", _VX.name, flush=True)


if __name__ == "__main__":
    main()
