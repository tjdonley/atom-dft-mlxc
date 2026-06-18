"""Summarize the Phase-D benchmark pickle: exchange energies E_x (Ha) per functional
per atom, HOMO eigenvalue, and two-stage convergence status. Used for the Phase-D
gate review (which hole / which meta-GGA reference converges, and the accuracy
pattern vs OEP/HF)."""
import pickle
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent / "data" / "phase_d_results.pkl"
COLS = ["OEP", "HF", "PBE", "rSCAN", "r2SCAN",
        "SIMPLE_HOLE", "SIMPLE_HOLE_GGA", "SIMPLE_HOLE_GEA"]
SHORT = {"OEP": "OEP", "HF": "HF", "PBE": "PBE", "rSCAN": "rSCAN", "r2SCAN": "r2SCAN",
         "SIMPLE_HOLE": "bare", "SIMPLE_HOLE_GGA": "GGA", "SIMPLE_HOLE_GEA": "GEA"}


def _fmt(d):
    if not d or "error" in d:
        return "  err  "
    s = f"{d['E_x']:+.4f}"
    if not d.get("converged", True):
        s += "*"          # SCF not converged
    elif "note" in d:
        s += "~"          # outer loop note
    return s


def main():
    r = pickle.loads(_OUT.read_bytes())
    hdr = "atom " + " ".join(f"{SHORT[c]:>9}" for c in COLS)
    print(hdr); print("-" * len(hdr))
    for sym in r:
        row = f"{sym:>4} " + " ".join(f"{_fmt(r[sym].get(c)):>9}" for c in COLS)
        print(row)
    print("\n* = SCF not converged   ~ = outer-loop note")
    print("\nHOMO eigenvalue (Ha), bare/GEA hole vs OEP:")
    for sym in r:
        def homo(name):
            d = r[sym].get(name, {})
            e = d.get("eigen")
            return f"{float(np.max(e)):+.4f}" if e is not None and len(e) else "  --  "
        print(f"  {sym:>3}  OEP {homo('OEP')}   bare {homo('SIMPLE_HOLE')}   "
              f"GEA {homo('SIMPLE_HOLE_GEA')}")
    print("\nTwo-stage convergence (outer iters / drho / note):")
    for sym in r:
        for c in ("SIMPLE_HOLE_GGA", "SIMPLE_HOLE_GEA"):
            d = r[sym].get(c, {})
            if d and "error" not in d:
                print(f"  {sym:>3} {SHORT[c]:>3}: iters={d.get('outer_iters','?')} "
                      f"drho={d.get('outer_drho', float('nan')):.1e} "
                      f"{d.get('note','')}")


if __name__ == "__main__":
    main()
