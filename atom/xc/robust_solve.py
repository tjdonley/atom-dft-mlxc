"""Robust self-consistent solve for SIMPLE_HOLE_KERNEL_FP -- converges all 69 reference atoms with a
single default procedure (no per-atom tuning) [[simple-hole-kernel-map]].

A single fixed SCF mixer CANNOT converge every atom: the reference-mapped exchange potential makes
some atoms need Pulay/DIIS ACCELERATION (e.g. F, Nb, W -- linear mixing is too slow / oscillates),
while others need pure DAMPING (e.g. K, Ca, Mn -- any Pulay extrapolation diverges them). These
regimes are mutually exclusive for fixed mixing parameters. The robust default is therefore an
ESCALATING-DAMPING ladder: try the accelerated mixer first and, only if it fails to converge, retry
with progressively more damping. Each atom converges on whichever rung fits it; nothing is tuned per
atom. Verified 69/69 (Z=1-57,72-83): 49 via Pulay, 19 via linear-0.2, 1 (Se) via damped-Pulay.

The exchange functional itself is unchanged -- this only governs how the SCF density iteration is
mixed. Production functional defaults are used when xc_params is None.
"""
import numpy as np

from atom import AtomicDFTSolver
from atom.xc.simple_hole_expansion import SIMPLEHOLEKERNELFPParameters as P

# production functional configuration (the H-anchored reference set, single adaptive SCF loop)
import os as _os
_REFS = _os.path.join(_os.path.dirname(__file__), "data",
                      "kernel_fp_refs_closed_rf001_l2power_Q_gf06_Hanchor.npz")


def production_params():
    """Default SIMPLE_HOLE_KERNEL_FP parameters used across the all-atom sweep."""
    return P(fp_l0=0.7, fp_l1=0.5, fp_l2pow=0.02, fp_lQ=0.3, fp_ref_ridge=1e-8, refs_path=_REFS,
             use_l2_power=True, use_Q=True, fa_ontop=False, fa_coeff=False,
             grad_filter=0.6, deriv_smooth=1.0, deriv_smooth_adaptive=True, auto_continuation=True)


# escalating-damping ladder: (name, solver mixing kwargs), increasing damping down the list.
MIXING_LADDER = [
    ("pulay",   {}),                                                                  # DIIS acceleration (default)
    ("lin0.2",  dict(use_pulay_mixing=False, linear_mixing_alpha1=0.2, linear_mixing_alpha2=0.3)),  # pure damping
    ("dpulayA", dict(use_pulay_mixing=True,  linear_mixing_alpha1=0.4, linear_mixing_alpha2=0.5)),  # damped Pulay
]


def robust_scf_solve(atomic_number, xc_params=None, scf_tolerance=3e-4,
                     max_scf_iterations=350, verbose=False, **solver_kw):
    """Solve the atom self-consistently, escalating SCF damping until convergence.

    Returns the solver result dict of the first rung that converges (with an added 'mixer' key naming
    the rung); if none converge, returns the last attempt's result. xc_params defaults to
    production_params(). Extra solver_kw are forwarded to every AtomicDFTSolver attempt."""
    if xc_params is None:
        xc_params = production_params()
    last = None
    for name, mix in MIXING_LADDER:
        s = AtomicDFTSolver(atomic_number=atomic_number, xc_functional="SIMPLE_HOLE_KERNEL_FP",
                            all_electron_flag=False, max_scf_iterations=max_scf_iterations,
                            scf_tolerance=scf_tolerance, xc_params=xc_params, **mix, **solver_kw)
        r = s.solve()
        r["mixer"] = name
        last = r
        if verbose:
            print(f"  Z={atomic_number} mixer={name} converged={r['converged']} "
                  f"res={r.get('rho_residual', np.nan):.2e}", flush=True)
        if r["converged"]:
            return r
    return last
