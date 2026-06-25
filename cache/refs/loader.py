"""Reusable loader for the SIMPLE exchange-hole reference cache.

The cache (see README.md) holds, for the pseudopotential set Z in {1..57, 72..83}:
  - hf/hf_Z{NN}.npz       : converged HF/PSP density + energies + EXACT ORBITALS
  - baselines/base_Z{NN}.npz : self-consistent PBE & rSCAN exchange energies
  - holes/hole_refs.npz   : moment-matched exact-hole references (13 closed-shell atoms)

Usage:
    from cache.refs.loader import load_hf, load_baseline, load_hole_ref, available_hf
    hf = load_hf(10)                 # Ne: dict with r, rho, w, Ehf, orbitals ...
    pbe, rscan = load_baseline(18)   # Ar PBE & rSCAN exchange energies
    ref = load_hole_ref("Ne")        # dict with rt, cn, s, Q, Rad, rho, eps_sel, Ehf
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(_HERE, "hf")
BASE_DIR = os.path.join(_HERE, "baselines")
HOLES = os.path.join(_HERE, "holes", "hole_refs.npz")


def available_hf():
    """Sorted list of Z with a cached HF solve."""
    zs = []
    for f in os.listdir(HF_DIR):
        if f.startswith("hf_Z") and f.endswith(".npz"):
            zs.append(int(f[4:6]))
    return sorted(zs)


def load_hf(Z, require_converged=True):
    """HF/PSP reference for atomic number Z. Returns a dict:
       r, rho, w (grid+density+weights), Ehf (=hf_exchange), Etot, converged, domain,
       and the exact orbitals r_sorted, g_sorted (nr,n_orb), occ, l_values (for hole reconstruction).
       Raises if missing, or (when require_converged) if the solve did not converge."""
    p = os.path.join(HF_DIR, f"hf_Z{int(Z):02d}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no cached HF for Z={Z} ({p})")
    d = np.load(p, allow_pickle=True)
    if require_converged and not bool(d["converged"]):
        raise ValueError(f"cached HF for Z={Z} did not converge")
    return {k: d[k] for k in d.files}


def load_baseline(Z):
    """(pbe_Ex, rscan_Ex) self-consistent exchange energies for Z, or (nan, nan) if absent."""
    p = os.path.join(BASE_DIR, f"base_Z{int(Z):02d}.npz")
    if not os.path.exists(p):
        return float("nan"), float("nan")
    d = np.load(p)
    return float(d["pbe_Ex"]), float(d["rscan_Ex"])


def _hole_atoms():
    d = np.load(HOLES)
    return sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_rt")})


def load_hole_ref(name):
    """Moment-matched exact-hole reference for an atom symbol (e.g. 'Ne'). Returns a dict with
       rt (npts,n_out) [moment-matched hole coeffs], cn (npts,n_out) [scale-free monopole feats],
       s, Q, Rad, rho, eps_sel (npts,), Ehf (scalar). KeyError if the atom is not in the set."""
    d = np.load(HOLES)
    keys = [k for k in d.files if k.startswith(name + "_")]
    if not keys:
        raise KeyError(f"no hole reference for {name}; available: {_hole_atoms()}")
    return {k[len(name) + 1:]: d[k] for k in keys}


def hole_atoms():
    """List of atom symbols with a moment-matched hole reference."""
    return _hole_atoms()
