#!/usr/bin/env python3
"""Reduced-gradient (s) accuracy of the SIMPLE features, and its (non)effect on
non-self-consistent PBE exchange energies.

Two parts, both parameter-free and non-self-consistent (no SCF; the density is
fixed and the exchange energy/ingredient is evaluated once):

  Part A -- s on a genuine 3D Cartesian grid (pseudo-diatomics).
    On the pseudo-H2 (two analytic 1s atoms, rho=exp(-2r)/pi, b=1.40 bohr; CUSPED)
    and pseudo-N2 (two N pseudopotential valence densities, b=2.074 bohr; SMOOTH)
    densities of the 3D validation harness, the reduced gradient
        s = |grad rho| / (2 k_F rho),   k_F = (3 pi^2 rho)^{1/3}
    is reconstructed from the SIMPLE l=1 features and compared to a direct
    finite-difference gradient at three evaluation centers (bond midpoint, atom
    site, off-axis). The l=1 window coefficients c_{n1m} give the gradient with NO
    free constant: rho_{1m}'(0) = sum_n c_{n1m} R'_{n1}(0) and
    |grad rho| = sqrt(3/4pi) sqrt(sum_m rho_{1m}'(0)^2) [Eq. (sq), l=1]. The SIMPLE
    gradient tracks the finite difference in the slowly-varying valence (smooth
    pseudo-N2); it degrades only at the bare pseudo-H2 nuclear cusp, where the
    spectral l=1 expansion is not valid.

  Part B -- non-SCF PBE exchange energy (atomic densities).
    On the radial atomic densities (H 1s; N pseudopotential valence -- the building
    blocks of the pseudo-diatomics), the SIMPLE reduced gradient (radial spectral
    operator) is fed to the PBE exchange enhancement and the exchange energy
        E_x = int rho eps_x^LDA(rho) F_x^PBE(s) d^3r
    is compared, on the SAME fixed density, between the SIMPLE-reconstructed s and a
    direct finite-difference s. The difference is a tiny fraction of the LDA->PBE
    gradient correction: the SIMPLE gradient reproduces the PBE exchange energy, so
    the gradient reconstruction has no significant effect on atomic-density energies.

Writes scripts/data/cartesian_s_pbe.json. Run from the repository root:
    python3 writeup/scripts/cartesian_s_pbe.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atom.descriptors.simple.derivatives import build_spectral_gradient_operator  # noqa: E402
from simple_3d_cartesian_tests import (  # noqa: E402
    BOND_N2,
    LatticeStencil,
    hydrogen_density,
    molecule,
    n_in_for,
    nitrogen_density_table,
)
from simple_validation_figures import R_C, window_basis  # noqa: E402

_OUT = Path(__file__).resolve().parent / "data" / "cartesian_s_pbe.json"
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# --- exchange constants ------------------------------------------------------
C_LDA = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)             # eps_x^LDA = C_LDA rho^{1/3}
KAPPA, MU = 0.804, 0.2195149727645171                   # PBE exchange
_TWO_3PI2_1_3 = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)  # textbook s denominator factor


def reduced_gradient(grad_rho, rho):
    """Textbook reduced gradient s = |grad rho| / (2 (3 pi^2)^{1/3} rho^{4/3})."""
    rho = np.maximum(np.asarray(rho, float), 1e-300)
    return np.abs(grad_rho) / (_TWO_3PI2_1_3 * rho ** (4.0 / 3.0))


def F_pbe(s):
    """PBE exchange enhancement F_x^PBE(s) = 1 + kappa - kappa/(1 + mu s^2/kappa)."""
    return 1.0 + KAPPA - KAPPA / (1.0 + MU * s ** 2 / KAPPA)


# =============================================================================
# Part A -- reduced gradient on the 3D Cartesian pseudo-diatomic grids
# =============================================================================
def simple_gradient_magnitude(stencil, rho_pts, n_in):
    """|grad rho| at the stencil center from the SIMPLE l=1 window coefficients.

    rho_{1m}'(0) = sum_n c_{n1m} R'_{n1}(0); the real-l=1-harmonic -> Cartesian
    factor is sqrt(3/4pi), so |grad rho| = sqrt(3/4pi) ||(rho_{1m}'(0))_m|| with no
    fitted constant (verified consistent with the radial spectral operator)."""
    c_window = stencil.window_coefficients(rho_pts)        # {l: (n_in, 2l+1)}
    eps = 1.0e-6
    slope0 = window_basis(1, n_in).evaluate(1, np.array([eps]))[:, 0] / eps  # R'_{n1}(0)
    g_m = slope0 @ c_window[1]                              # (3,) = rho_{1m}'(0)
    return np.sqrt(3.0 / (4.0 * np.pi)) * np.linalg.norm(g_m)


def fd_gradient_magnitude(mol, h=1.0e-3):
    """|grad rho| at the evaluation center (origin) by central finite differences."""
    grad = np.empty(3)
    for axis in range(3):
        step = np.zeros((2, 3))
        step[0, axis], step[1, axis] = +h, -h
        rho_pm = mol.density(step)                         # per-spin
        grad[axis] = (rho_pm[0] - rho_pm[1]) / (2.0 * h)
    return 2.0 * np.linalg.norm(grad)                      # x2: per-spin -> total


def part_a():
    h = 0.2
    n_in = n_in_for(h)
    stencil = LatticeStencil(h, n_in)
    centers = ("midpoint", "atom", "offaxis")
    out = {}
    for kind, label in (("H", "pseudo-H2"), ("N", "pseudo-N2")):
        rows = {}
        for center in centers:
            mol = molecule(kind, center)
            rho_tot_pts = 2.0 * mol.density(stencil.points)            # total density
            rho_center = 2.0 * float(mol.density(np.zeros((1, 3)))[0])
            grad_simple = simple_gradient_magnitude(stencil, rho_tot_pts, n_in)
            grad_fd = fd_gradient_magnitude(mol)
            s_simple = float(reduced_gradient(grad_simple, rho_center))
            s_fd = float(reduced_gradient(grad_fd, rho_center))
            ratio = s_simple / s_fd if s_fd > 1e-12 else float("nan")
            rows[center] = {"s_simple": s_simple, "s_fd": s_fd, "ratio": ratio,
                            "rho_center": rho_center}
            print(f"  {label:10s} {center:9s}: s_SIMPLE={s_simple:.4f} "
                  f"s_FD={s_fd:.4f} ratio={ratio:.3f}")
        out[label] = rows
    del stencil
    return out


# =============================================================================
# Part B -- non-SCF PBE exchange energy on radial atomic densities
# =============================================================================
def _radial_density(kind):
    """(r_grid, rho_total) for one atom on a window-sized radial grid (bohr)."""
    r = np.linspace(1.0e-3, R_C, 700)
    if kind == "H":
        return r, hydrogen_density(r)                      # exp(-2r)/pi
    r_tab, rho_tab = nitrogen_density_table()              # N valence (5 e), bohr
    rho = np.interp(r, r_tab, rho_tab, left=rho_tab[0], right=0.0)
    return r, rho


def part_b():
    out = {}
    for kind, label in (("H", "H (1s, cusped)"), ("N", "N (valence, smooth)")):
        r, rho = _radial_density(kind)
        n_elec = 4.0 * np.pi * trapezoid(rho * r ** 2, r)
        G = build_spectral_gradient_operator(r, n_channels=40)
        s_simple = reduced_gradient(G @ rho, rho)
        s_fd = reduced_gradient(np.gradient(rho, r), rho)

        eps_lda = C_LDA * rho ** (1.0 / 3.0)
        w = 4.0 * np.pi * r ** 2
        e_lda = float(trapezoid(w * rho * eps_lda, r))
        e_pbe_fd = float(trapezoid(w * rho * eps_lda * F_pbe(s_fd), r))
        e_pbe_simple = float(trapezoid(w * rho * eps_lda * F_pbe(s_simple), r))
        corr_fd = e_pbe_fd - e_lda                          # LDA -> PBE correction
        delta = e_pbe_simple - e_pbe_fd                     # SIMPLE-s vs FD-s
        # s accuracy in the slowly-varying valence (where the density is sizable)
        m = (rho > 1e-2 * rho.max()) & (r > 0.4) & (r < 0.85 * R_C)
        sr = (s_simple[m] / np.maximum(s_fd[m], 1e-12))
        out[label] = {
            "n_elec": n_elec, "E_x_LDA": e_lda, "E_x_PBE_FD": e_pbe_fd,
            "E_x_PBE_SIMPLE": e_pbe_simple, "LDA_to_PBE_corr": corr_fd,
            "deltaE_SIMPLE_minus_FD": delta,
            "deltaE_frac_of_corr_pct": 100.0 * abs(delta) / abs(corr_fd),
            "deltaE_frac_of_Ex_pct": 100.0 * abs(delta) / abs(e_pbe_fd),
            "s_ratio_valence_mean": float(sr.mean()),
            "s_ratio_valence_max_dev": float(np.abs(sr - 1.0).max()),
        }
        print(f"  {label:22s}: E_x LDA={e_lda:.4f} PBE(FD)={e_pbe_fd:.4f} "
              f"PBE(SIMPLE)={e_pbe_simple:.4f} Ha")
        print(f"  {'':22s}  dE(SIMPLE-FD)={1e3*delta:+.3f} mHa "
              f"({out[label]['deltaE_frac_of_corr_pct']:.2f}% of the "
              f"{1e3*corr_fd:+.1f} mHa GGA correction); "
              f"s valence ratio {sr.mean():.3f}+-{np.abs(sr-1).max():.3f}")
    return out


if __name__ == "__main__":
    print("Part A: reduced gradient s on 3D Cartesian pseudo-diatomic grids")
    a = part_a()
    print("\nPart B: non-SCF PBE exchange energy on radial atomic densities")
    b = part_b()
    _OUT.parent.mkdir(exist_ok=True)
    _OUT.write_text(json.dumps({"part_a_cartesian_s": a, "part_b_pbe_energy": b}, indent=2))
    print(f"\nwrote {_OUT}")
