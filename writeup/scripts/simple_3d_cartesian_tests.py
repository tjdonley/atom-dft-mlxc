#!/usr/bin/env python3
"""Three-dimensional Cartesian-grid validation of SIMPLE.

Where the 1D round (simple_validation_figures.py) tested the SIMPLE pipeline
against radially resolved multipole profiles, this script runs the *entire*
pipeline on a genuine 3D Cartesian lattice: window coefficients by direct
kernel sums over lattice points, ball averages by partial-cell shell sums,
adaptive radius, mean-split transfer, and non-dimensionalization. The lattice
itself breaks rotational and translational symmetry, so these tests quantify
the grid-induced anisotropy that the 1D round could not see.

Test densities are built from single-atom densities placed at two centers:

    pseudo-H2 : two analytic hydrogen 1s atoms (rho = exp(-2r)/pi) at the
                H2 bond length b = 1.40 bohr. Cusped at the nuclei -- the
                hard case for a uniform grid.
    pseudo-N2 : two nitrogen pseudopotential valence densities (5 electrons
                each, from the atom-dft-mlxc radial DFT solver, GGA_PBE,
                psp8) at the N2 bond length b = 2.074 bohr. Smooth at the
                nuclei -- representative of real-space pseudopotential codes.

Neither is a self-consistent molecule; the superposition breaks radial
symmetry in a realistic way at realistic density scales, which is all the
descriptor tests require. The N atom density is cached in
scripts/data/n_atom_Z7_pbe_psp8.npz (regenerated automatically by running
the radial solver if the file is missing).

The continuum reference is quasi-exact for arbitrary two-atom geometry: each
atom's environment is axially symmetric about its own direction, so its
multipole profiles are computed by Gauss-Legendre quadrature in its own
frame (m = 0 only) and rotated into the lab frame with real Wigner-D
matrices. General-l real Wigner-D blocks are built by sampled least squares
(the linear relation Y_lm(R u) = sum_m' D_mm' Y_lm'(u) is exact, so the
solve recovers D to machine precision); they are verified at startup against
the closed-form l <= 2 blocks and the representation property at l = 3.

Tests and figures (all l <= 3, R_c = 3 A, xi* = 2, n_out = 8; the grid
channel count follows n_in(h) = 2 ceil(R_c / 4h), giving the 1D round's
pairing n_in = 16 at h = 0.2 bohr):

    simple_3d_convergence.pdf  - deviation of the gridded descriptors from
                                 the continuum reference (same n_in) vs grid
                                 spacing h, at three evaluation centers (bond
                                 midpoint, atom site, off-axis), both
                                 molecules; plus symmetry-forbidden odd-l
                                 leakage at the bond midpoint.
    simple_3d_rotation.pdf     - random rigid rotations of the molecule about
                                 the evaluation center on a fixed lattice:
                                 per-rotation deviation from the continuum
                                 covariant prediction, and the relative
                                 spread of the power-spectrum invariants
                                 across orientations.
    simple_3d_registration.pdf - (a) sub-grid translation sweep: molecule and
                                 evaluation center shifted together by
                                 fractions of h along a low-symmetry
                                 direction; (b) scale invariance D(lambda) of
                                 the full 3D pipeline against the continuum.

Run from the repository root (about 10 minutes):

    python3 scripts/simple_3d_cartesian_tests.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import eval_legendre

# Shared SIMPLE machinery (constants, transfer matrices, window basis) and
# the atom-dft-mlxc path setup come from the 1D validation script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simple_validation_figures import (  # noqa: E402
    _PKG_ROOT,
    L_MAX,
    N_OUT,
    R_C,
    RHO_MIN,
    XI_TARGET,
    transfer_matrix,
    window_basis,
)

from atom.descriptors.simple import (  # noqa: E402
    a_n_closed_form,
    radial_gauss_grid,
    random_rotation_matrix,
    real_sh_rotation_matrix,
    real_sph_harm,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent / "data"
H_REF = 0.2  # lattice spacing for the rotation/registration/scale tests
H_SWEEP = (0.4, 0.3, 0.2, 0.15, 0.1)
N_IN_CONT = 32  # continuum channel count for the scale sweep
SEED = 20260612
N_ROTATIONS = 24

BOND_H2 = 1.40  # bohr
BOND_N2 = 2.074  # bohr (1.0977 A)
OFFAXIS = 1.2  # bohr, perpendicular offset of the off-axis center
# Generic (low-symmetry) directions, fixed for reproducibility.
U_BOND = np.array([1.0, 0.6, 0.3]) / np.linalg.norm([1.0, 0.6, 0.3])
U_SHIFT = np.array([0.7, 0.45, 0.25]) / np.linalg.norm([0.7, 0.45, 0.25])

trapezoid = getattr(np, "trapezoid", None) or np.trapz


def n_in_for(h):
    """Grid-matched channel count, n_in(h) = 2 ceil(R_c / 4h).

    Reproduces the 1D round's pairing n_in = 16 at h = 0.2 bohr and stays at
    or above the kernel-resolution bound n_in ~ R_c / 2h."""
    return max(N_OUT, 2 * int(np.ceil(R_C / (4.0 * h))))


# =============================================================================
# Real Wigner-D matrices for any l (sampled-exact least squares)
# =============================================================================
def real_wigner_d(l, rot):
    """Real-spherical-harmonic rotation block D^(l) for arbitrary l.

    The relation Y_lm(R u) = sum_m' D_mm' Y_lm'(u) is linear and exact, so a
    least-squares solve on random sample directions recovers D to machine
    precision (verified against the closed-form l <= 2 blocks and the l = 3
    representation property in _self_check_wigner)."""
    if l == 0:
        return np.eye(1)
    rng = np.random.default_rng(987654321 + l)  # fixed: sampling is exact
    directions = rng.normal(size=(16 * (2 * l + 1), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    y_ref = np.column_stack(
        [real_sph_harm(l, m, directions) for m in range(-l, l + 1)]
    )
    y_rot = np.column_stack(
        [real_sph_harm(l, m, directions @ np.asarray(rot).T) for m in range(-l, l + 1)]
    )
    d_matrix, *_ = np.linalg.lstsq(y_ref, y_rot, rcond=None)
    return d_matrix.T


def rotation_z_to(direction):
    """A rotation matrix taking the z axis to ``direction``."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, u))
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    angle = np.arctan2(s, c)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) * np.cos(angle) + np.sin(angle) * cross + (
        1 - np.cos(angle)
    ) * np.outer(axis, axis)


def _self_check_wigner():
    rng = np.random.default_rng(SEED)
    r1, r2 = random_rotation_matrix(rng), random_rotation_matrix(rng)
    for l in (1, 2):
        err = np.abs(real_wigner_d(l, r1) - real_sh_rotation_matrix(l, r1)).max()
        assert err < 1e-12, f"Wigner-D l={l} vs closed form: {err:.1e}"
    d3a, d3b = real_wigner_d(3, r1), real_wigner_d(3, r2)
    err_orth = np.abs(d3a @ d3a.T - np.eye(7)).max()
    err_rep = np.abs(real_wigner_d(3, r1 @ r2) - d3a @ d3b).max()
    assert err_orth < 1e-12 and err_rep < 1e-12, (err_orth, err_rep)
    print(f"self-check Wigner-D: l<=2 closed form ok; l=3 orthogonality "
          f"{err_orth:.1e}, representation {err_rep:.1e}")


# =============================================================================
# Single-atom densities (per spin: total / 2)
# =============================================================================
def hydrogen_density(r):
    """Total H 1s density exp(-2r)/pi."""
    return np.exp(-2.0 * np.asarray(r, dtype=float)) / np.pi


_N_CACHE = DATA_DIR / "n_atom_Z7_pbe_psp8.npz"


def nitrogen_density_table():
    """(r, rho_total) for the N pseudopotential valence density (5 e)."""
    if not _N_CACHE.exists():
        print("generating N atom density (radial DFT solver, GGA_PBE, psp8)...")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from atom.solver import AtomicDFTSolver

            solver = AtomicDFTSolver(
                atomic_number=7, xc_functional="GGA_PBE",
                all_electron_flag=False, verbose=False,
            )
            res = solver.solve()
        assert res["converged"], "N atom SCF did not converge"
        order = np.argsort(res["quadrature_nodes"])
        r = res["quadrature_nodes"][order]
        rho = res["rho"][order]
        keep = np.concatenate([[True], np.diff(r) > 1e-12])
        r, rho = r[keep], rho[keep]
        n_elec = 4.0 * np.pi * trapezoid(rho * r**2, r)
        DATA_DIR.mkdir(exist_ok=True)
        np.savez(
            _N_CACHE, r=r, rho=rho,
            meta=json.dumps({
                "source": "atom-dft-mlxc AtomicDFTSolver",
                "atomic_number": 7, "xc_functional": "GGA_PBE",
                "pseudopotential": "psps/07.psp8 (valence: 5 electrons)",
                "n_electrons_check": n_elec,
            }),
        )
        print(f"  cached to {_N_CACHE.name} (integrates to {n_elec:.6f} e)")
    data = np.load(_N_CACHE)
    return data["r"], data["rho"]


def make_atom_density(kind):
    """Per-spin radial density rho_sigma(r) for one atom."""
    if kind == "H":
        return lambda r: 0.5 * hydrogen_density(r)
    r_tab, rho_tab = nitrogen_density_table()
    return lambda r: 0.5 * np.interp(
        np.asarray(r, dtype=float), r_tab, rho_tab, left=rho_tab[0], right=0.0
    )


class Diatomic:
    """Two identical atoms; positions are relative to the evaluation center.

    ``lam`` applies the uniform coordinate scaling about the evaluation
    center, rho -> lambda^3 rho(lambda r): atomic profiles dilate and the
    relative positions contract to p / lambda."""

    def __init__(self, kind, positions, lam=1.0):
        self.kind = kind
        self.atom = make_atom_density(kind)
        self.lam = float(lam)
        self.positions = [np.asarray(p, dtype=float) / self.lam for p in positions]

    def density(self, points):
        """Per-spin density at an (N, 3) array of relative coordinates."""
        rho = np.zeros(len(points))
        for p in self.positions:
            dist = np.linalg.norm(points - p[None, :], axis=1)
            rho += self.atom(self.lam * dist)
        return self.lam**3 * rho

    def rotated(self, rot):
        out = Diatomic.__new__(Diatomic)
        out.kind, out.atom, out.lam = self.kind, self.atom, self.lam
        out.positions = [np.asarray(rot) @ p for p in self.positions]
        return out


def molecule(kind, center, lam=1.0, bond_dir=U_BOND):
    """Diatomic with the evaluation center at one of three standard points."""
    bond = {"H": BOND_H2, "N": BOND_N2}[kind]
    half = 0.5 * bond * np.asarray(bond_dir)
    if center == "midpoint":
        positions = [half, -half]
    elif center == "atom":
        positions = [np.zeros(3), 2.0 * half]
    elif center == "offaxis":
        perp = np.cross(bond_dir, [0.0, 0.0, 1.0])
        perp /= np.linalg.norm(perp)
        positions = [half + OFFAXIS * perp, -half + OFFAXIS * perp]
    else:
        raise ValueError(center)
    return Diatomic(kind, positions, lam=lam)


# =============================================================================
# Shared SIMPLE pipeline (window coefficients + ball average -> descriptors)
# =============================================================================
def simple_from_window(c_window, rho_bar, n_in, l_max=L_MAX, n_out=N_OUT):
    """SIMPLE descriptors from window coefficients c_window[l] (n_in, 2l+1)
    and a ball-average callable rho_bar(R). Identical algebra for the
    continuum and grid paths; only the two inputs differ."""

    def g(radius):
        rb_safe = np.sqrt(max(rho_bar(radius), 0.0) ** 2 + RHO_MIN**2)
        return radius * (6.0 * np.pi**2 * rb_safe) ** (1.0 / 3.0)

    clamped = g(R_C) <= XI_TARGET
    r_ad = R_C if clamped else brentq(
        lambda radius: g(radius) - XI_TARGET, 1e-3, R_C, rtol=1e-13
    )
    rho_bar_window = rho_bar(R_C)
    rho_bar_safe = np.sqrt(rho_bar(r_ad) ** 2 + RHO_MIN**2)
    a_0 = a_n_closed_form(0, r_ad)[0]

    result = {"r_ad": r_ad, "clamped": clamped, "rho_bar_safe": rho_bar_safe}
    for l in range(l_max + 1):
        if l == 0:
            c_fluct = c_window[0].copy()
            c_fluct[:, 0] -= rho_bar_window * a_n_closed_form(n_in - 1, R_C)
            c_ad = np.zeros((n_out, 1))
            c_ad[:, 0] = rho_bar_window * a_n_closed_form(n_out - 1, r_ad)
        else:
            c_fluct = c_window[l]
            c_ad = np.zeros((n_out, 2 * l + 1))
        c_ad = c_ad + transfer_matrix(l, r_ad, n_out, n_in) @ c_fluct
        result[l] = c_ad / (a_0 * rho_bar_safe)
    return result


# =============================================================================
# Continuum reference (quasi-exact, arbitrary two-atom geometry)
# =============================================================================
U_NODES, U_WEIGHTS = leggauss(96)


def _axial_profile(mol, p, l, r):
    """Multipole profile rho_l(r) of one atom about the center, in the frame
    whose z axis points along the atom (m = 0 only by axial symmetry)."""
    b = float(np.linalg.norm(p)) * mol.lam
    r_scaled = mol.lam * np.atleast_1d(np.asarray(r, dtype=float))
    if b < 1e-12:
        if l > 0:
            return np.zeros(r_scaled.size)
        return mol.lam**3 * mol.atom(r_scaled)
    dist = np.sqrt(
        r_scaled[:, None] ** 2 + b**2 - 2.0 * r_scaled[:, None] * b * U_NODES[None, :]
    )
    return mol.lam**3 * (2 * l + 1) / 2.0 * np.sum(
        U_WEIGHTS * eval_legendre(l, U_NODES) * mol.atom(dist), axis=1
    )


def continuum_descriptors(mol, n_in, l_max=L_MAX, n_out=N_OUT):
    """Per-atom axial multipole profiles + Wigner-D rotation into the lab
    frame; then the shared SIMPLE pipeline with Gauss-Legendre quadratures."""
    quad = radial_gauss_grid(R_C, 512)
    r = quad.nodes

    c_window = {l: np.zeros((n_in, 2 * l + 1)) for l in range(l_max + 1)}
    for p in mol.positions:
        on_center = np.linalg.norm(p) < 1e-12
        d_blocks = None if on_center else {
            l: real_wigner_d(l, rotation_z_to(p)) for l in range(1, l_max + 1)
        }
        for l in range(l_max + 1):
            if on_center and l > 0:
                continue
            basis = window_basis(l, n_in)
            angular = np.sqrt(4.0 * np.pi / (2 * l + 1))
            c_axial = angular * np.einsum(
                "nj,j->n",
                basis.evaluate(l, r),
                quad.weights * r**2 * _axial_profile(mol, p, l, r),
            )
            if l == 0:
                c_window[0][:, 0] += c_axial
            else:
                c_window[l] += np.outer(c_axial, d_blocks[l][:, l])  # D[:, m=0]

    def rho_bar(radius):
        sub = radial_gauss_grid(radius, 192)
        rho_00 = sum(_axial_profile(mol, p, 0, sub.nodes) for p in mol.positions)
        return 3.0 / radius**3 * float(np.sum(sub.weights * sub.nodes**2 * rho_00))

    return simple_from_window(c_window, rho_bar, n_in, l_max, n_out)


# =============================================================================
# 3D Cartesian-grid pipeline
# =============================================================================
class LatticeStencil:
    """All lattice points within the window of an evaluation center.

    The lattice is x = (i + f) h with integer i and fixed fractional
    registration f in [0, 1)^3; the evaluation center is the origin of the
    relative coordinates. Kernel factors (radial basis values and real
    harmonics) are cached when the stencil is small enough and recomputed on
    the fly otherwise, so a stencil is reusable for every density,
    orientation, and scale at a given (h, n_in, registration)."""

    CACHE_LIMIT = 4 * 10**5  # points; ~60 MB of cached factors at n_in = 16

    def __init__(self, h, n_in, frac=(0.0, 0.0, 0.0), l_max=L_MAX):
        self.h, self.n_in, self.l_max = float(h), int(n_in), int(l_max)
        reach = int(np.ceil((R_C + h) / h)) + 1
        idx = np.arange(-reach, reach + 1)
        ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
        pts = (np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
               + np.asarray(frac, dtype=float)) * h
        dist = np.linalg.norm(pts, axis=1)
        keep = dist <= R_C + 0.5 * h  # ramp support of the shell sums
        self.points = pts[keep]
        self.dist = dist[keep]
        units = self.points.copy()
        safe = np.maximum(self.dist, 1e-300)
        units /= safe[:, None]
        units[self.dist < 1e-14] = (0.0, 0.0, 1.0)  # R_nl(0) = 0 for l > 0
        self.units = units
        self._inside = self.dist <= R_C
        self._cache = {} if len(self.points) <= self.CACHE_LIMIT else None

    def _factors(self, l):
        if self._cache is not None and l in self._cache:
            return self._cache[l]
        radial = window_basis(l, self.n_in).evaluate(l, self.dist)
        radial[:, ~self._inside] = 0.0
        harmonics = np.stack(
            [real_sph_harm(l, m, self.units) for m in range(-l, l + 1)]
        )
        if self._cache is not None:
            self._cache[l] = (radial, harmonics)
        return radial, harmonics

    def window_coefficients(self, rho):
        c_window = {}
        for l in range(self.l_max + 1):
            radial, harmonics = self._factors(l)
            c_window[l] = (radial * rho[None, :]) @ harmonics.T * self.h**3
        return c_window

    def shell_weights(self, radius):
        """Partial-cell ramp weights; exact for uniform densities after
        normalization by their own sum (preserves the exact HEG limit)."""
        return np.clip((radius - self.dist) / self.h + 0.5, 0.0, 1.0)


def grid_descriptors(mol, stencil, l_max=L_MAX, n_out=N_OUT):
    rho = mol.density(stencil.points)
    c_window = stencil.window_coefficients(rho)
    nearest = int(np.argmin(stencil.dist))

    def rho_bar(radius):
        w = stencil.shell_weights(radius)
        w_sum = np.sum(w)
        if w_sum == 0.0:  # radius below the nearest sample (off-lattice
            return float(rho[nearest])  # center): nearest-cell value
        return float(np.sum(w * rho) / w_sum)

    return simple_from_window(c_window, rho_bar, stencil.n_in, l_max, n_out)


# =============================================================================
# Metrics
# =============================================================================
def stack_d(result, l_max=L_MAX):
    return np.concatenate([result[l].ravel() for l in range(l_max + 1)])


def deviation_all(result, reference, l_max=L_MAX):
    delta = stack_d(result, l_max) - stack_d(reference, l_max)
    return np.linalg.norm(delta) / np.linalg.norm(stack_d(reference, l_max))


def per_l_deviation(result, reference, l_max=L_MAX):
    """Per-l deviation normalized by the all-channel reference norm (robust
    when a channel is symmetry-forbidden in the reference)."""
    norm = np.linalg.norm(stack_d(reference, l_max))
    return {
        l: np.linalg.norm(result[l] - reference[l]) / norm
        for l in range(l_max + 1)
    }


def rotate_reference(reference, rot, l_max=L_MAX):
    out = dict(reference)
    for l in range(1, l_max + 1):
        out[l] = reference[l] @ real_wigner_d(l, rot).T  # d'_m = sum_m' D_mm' d_m'
    return out


def power_spectrum(result, l_max=L_MAX):
    return np.concatenate(
        [np.sum(result[l] ** 2, axis=1) for l in range(l_max + 1)]
    )


# =============================================================================
# Self-checks
# =============================================================================
def _self_check_reference():
    """The general-geometry continuum reference must reproduce the 1D round's
    axial code path (central + off-center atom along z at b = 1.5 bohr), and
    be exactly covariant under rotation of the geometry."""
    from simple_validation_figures import environment_profiles, simple_descriptors

    ref_1d = simple_descriptors(environment_profiles(), 32)
    mol = Diatomic("H", [np.zeros(3), np.array([0.0, 0.0, 1.5])])
    ref_3d = continuum_descriptors(mol, 32)
    err = 0.0
    for l in range(L_MAX + 1):
        err = max(err, np.abs(ref_3d[l][:, l] - ref_1d[l]).max())
        if l > 0:  # m != 0 columns must vanish for the axial geometry
            err = max(err, np.abs(np.delete(ref_3d[l], l, axis=1)).max())
    assert err < 1e-9, f"continuum reference vs 1D axial path: {err:.1e}"
    print(f"self-check continuum reference vs 1D axial path: {err:.1e}")

    rng = np.random.default_rng(SEED + 1)
    rot = random_rotation_matrix(rng)
    direct = continuum_descriptors(
        Diatomic("H", [rot @ p for p in mol.positions]), 32
    )
    err_cov = deviation_all(direct, rotate_reference(ref_3d, rot))
    assert err_cov < 1e-9, f"continuum covariance: {err_cov:.1e}"
    print(f"self-check continuum covariance under rotation: {err_cov:.1e}")


def _self_check_grid(stencil):
    """Electron count and the HEG limit on the actual 3D lattice."""
    for kind, label in (("H", "H2"), ("N", "N2")):
        mol = molecule(kind, "midpoint")
        rho = mol.density(stencil.points)
        count = 2.0 * np.sum(stencil.shell_weights(R_C) * rho) * stencil.h**3
        print(f"self-check {label} electrons inside the R_c ball "
              f"(h={stencil.h}): {count:.3f}")

    rho0 = 0.01
    c_window = stencil.window_coefficients(np.full(len(stencil.points), rho0))
    res = simple_from_window(c_window, lambda radius: rho0, stencil.n_in)
    reference = (-1.0) ** np.arange(N_OUT) / (np.arange(N_OUT) + 1)
    err = np.abs(res[0][:, 0] - reference).max()
    err_aniso = max(np.abs(res[l]).max() for l in range(1, L_MAX + 1))
    print(f"self-check HEG on the 3D lattice (h={stencil.h}): monopole "
          f"{err:.1e}, l>0 leakage {err_aniso:.1e}")


# =============================================================================
# Tests
# =============================================================================
CENTERS = ("midpoint", "atom", "offaxis")
KINDS = ("H", "N")
KIND_LABEL = {"H": "pseudo-H$_2$", "N": "pseudo-N$_2$"}
KIND_COLOR = {"H": "tab:blue", "N": "tab:red"}
CENTER_MARKER = {"midpoint": "o", "atom": "s", "offaxis": "^"}


def test_convergence():
    """Deviation from the continuum reference (same n_in) vs h."""
    results, leakage = {}, {}
    for h in H_SWEEP:
        n_in = n_in_for(h)
        t0 = time.time()
        stencil = LatticeStencil(h, n_in)
        for kind in KINDS:
            for center in CENTERS:
                mol = molecule(kind, center)
                ref = continuum_descriptors(mol, n_in)
                res = grid_descriptors(mol, stencil)
                results[(kind, center, h)] = deviation_all(res, ref)
                if center == "midpoint":
                    leakage[(kind, h)] = max(
                        np.linalg.norm(res[l]) for l in (1, 3)
                    ) / np.linalg.norm(stack_d(res))
        del stencil
        print(f"convergence h={h} (n_in={n_in}, {time.time() - t0:.0f}s): "
              + ", ".join(f"{k}-{c} {results[(k, c, h)]:.2e}"
                          for k in KINDS for c in CENTERS))
    print("odd-l leakage at the midpoint (lattice inversion symmetry): max "
          f"{max(leakage.values()):.1e}")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    hs = np.array(H_SWEEP)
    for ax, kind in zip(axes, KINDS):
        for center in CENTERS:
            ax.loglog(hs, [results[(kind, center, h)] for h in H_SWEEP],
                      CENTER_MARKER[center] + "-", label=center)
        anchor = results[(kind, "midpoint", 0.2)]
        ax.loglog(hs, anchor * (hs / 0.2) ** 2, "k:", lw=1.0,
                  label=r"$\mathcal{O}(h^2)$ guide")
        ax.set(xlabel=r"$h$ (bohr)",
               ylabel=r"$\Vert d_{\rm grid} - d_{\rm exact}\Vert / \Vert d_{\rm exact}\Vert$",
               title=f"({'ab'[KINDS.index(kind)]}) {KIND_LABEL[kind]}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_3d_convergence.pdf")
    plt.close(fig)
    return results, leakage


def test_rotation(stencil):
    """Random rigid rotations about the evaluation center, fixed lattice."""
    rng = np.random.default_rng(SEED)
    rotations = [random_rotation_matrix(rng) for _ in range(N_ROTATIONS)]
    summary, samples = {}, {}
    for kind in KINDS:
        for center in ("midpoint", "offaxis"):
            mol = molecule(kind, center)
            ref_cont = continuum_descriptors(mol, stencil.n_in)
            per_l = {l: [] for l in range(L_MAX + 1)}
            spectra = []
            for rot in rotations:
                res = grid_descriptors(mol.rotated(rot), stencil)
                ref_rot = rotate_reference(ref_cont, rot)
                for l, v in per_l_deviation(res, ref_rot).items():
                    per_l[l].append(v)
                spectra.append(power_spectrum(res))
            spectra = np.array(spectra)
            spread = np.abs(spectra - spectra.mean(axis=0)).max() / np.abs(
                spectra.mean(axis=0)
            ).max()
            summary[(kind, center)] = ({l: max(v) for l, v in per_l.items()}, spread)
            samples[(kind, center)] = per_l
            print(f"rotation {kind}-{center}: covariant deviation "
                  + ", ".join(f"l={l} {max(v):.2e}" for l, v in per_l.items())
                  + f"; power-spectrum spread {spread:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ls = np.arange(L_MAX + 1)
    floor = 1e-17
    for kind in KINDS:
        for center in ("midpoint", "offaxis"):
            per_l = samples[(kind, center)]
            jitter = (0.10 if kind == "N" else -0.10) + (
                0.05 if center == "offaxis" else -0.05
            )
            for l in ls:
                vals = np.maximum(np.array(per_l[l]), floor)
                axes[0].semilogy(np.full(vals.size, l + jitter), vals, ".",
                                 ms=3, color=KIND_COLOR[kind], alpha=0.3)
            axes[0].semilogy(
                ls + jitter, np.maximum([max(per_l[l]) for l in ls], floor),
                CENTER_MARKER[center], ls="", color=KIND_COLOR[kind], ms=6,
                label=f"{KIND_LABEL[kind]}, {center}",
            )
    axes[0].set(xlabel=r"$\ell$", xticks=range(L_MAX + 1),
                ylabel=r"$\Vert d_\ell - [D^{(\ell)} d_{\rm exact}]_\ell\Vert / \Vert d_{\rm exact}\Vert$",
                title=f"(a) deviation from the covariant prediction, "
                      f"{N_ROTATIONS} rotations")
    axes[0].legend(fontsize=8, loc="center right")

    bars, labels, colors = [], [], []
    for kind in KINDS:
        for center in ("midpoint", "offaxis"):
            bars.append(summary[(kind, center)][1])
            labels.append(f"{KIND_LABEL[kind]}\n{center}")
            colors.append(KIND_COLOR[kind])
    axes[1].bar(range(len(bars)), bars, color=colors, alpha=0.75)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(bars)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set(ylabel=r"$\max_{n\ell} |\tilde P - \langle\tilde P\rangle| / \max\langle\tilde P\rangle$",
                title="(b) power-spectrum spread across orientations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_3d_rotation.pdf")
    plt.close(fig)
    return summary


def test_registration_and_scale(stencil):
    """(a) sub-grid translation of molecule + center; (b) scale invariance."""
    fractions = np.linspace(0.0, 1.0, 9)[1:]
    base = {kind: grid_descriptors(molecule(kind, "offaxis"), stencil)
            for kind in KINDS}
    reg = {kind: [] for kind in KINDS}
    for f in fractions:
        # molecule + center shifted together by delta = f h u: the relative
        # geometry is unchanged, the lattice registration becomes -delta/h.
        shifted = LatticeStencil(
            stencil.h, stencil.n_in, frac=tuple((-f * U_SHIFT) % 1.0)
        )
        for kind in KINDS:
            reg[kind].append(
                deviation_all(grid_descriptors(molecule(kind, "offaxis"), shifted),
                              base[kind])
            )
        del shifted
    reg = {kind: np.array(v) for kind, v in reg.items()}
    for kind in KINDS:
        print(f"registration {kind}: max deviation over sub-grid shifts "
              f"{reg[kind].max():.2e}")

    lambdas = np.exp(np.linspace(np.log(0.25), np.log(4.0), 17))
    mask = np.abs(lambdas - 1.0) > 1e-12
    scale, lam_clamp = {}, {}
    for kind in KINDS:
        ref_grid = grid_descriptors(molecule(kind, "midpoint"), stencil)
        ref_cont = continuum_descriptors(molecule(kind, "midpoint"), N_IN_CONT)
        lam_clamp[kind] = ref_cont["r_ad"] / R_C
        rows_grid, rows_cont = [], []
        for lam in lambdas:
            mol = molecule(kind, "midpoint", lam=lam)
            rows_grid.append(deviation_all(grid_descriptors(mol, stencil), ref_grid))
            rows_cont.append(
                deviation_all(continuum_descriptors(mol, N_IN_CONT), ref_cont)
            )
        scale[kind] = (np.array(rows_grid), np.array(rows_cont))
        i2 = int(np.argmin(np.abs(lambdas - 2.0)))
        print(f"scale {kind}: D(2) grid {scale[kind][0][i2]:.2e}, "
              f"continuum {scale[kind][1][i2]:.2e}, "
              f"clamp onset lambda {lam_clamp[kind]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for kind in KINDS:
        axes[0].semilogy(fractions, reg[kind], CENTER_MARKER["offaxis"] + "-",
                         color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    axes[0].set(xlabel=r"shift $|\delta| / h$ along $(0.7, 0.45, 0.25)$",
                ylabel=r"$\Vert d(\delta) - d(0)\Vert / \Vert d(0)\Vert$",
                title=f"(a) sub-grid registration, $h={stencil.h}$, "
                      f"off-axis center")
    axes[0].legend(fontsize=8)
    for kind in KINDS:
        rows_grid, rows_cont = scale[kind]
        axes[1].loglog(lambdas[mask], rows_cont[mask], "-", color=KIND_COLOR[kind],
                       label=f"{KIND_LABEL[kind]}, continuum $n_{{in}}={N_IN_CONT}$")
        axes[1].loglog(lambdas[mask], rows_grid[mask], "o", ms=4,
                       color=KIND_COLOR[kind],
                       label=f"{KIND_LABEL[kind]}, $h={stencil.h}$ grid")
    axes[1].axvspan(lambdas[0], max(lam_clamp.values()), color="0.92")
    axes[1].text(lambdas[0] * 1.1, axes[1].get_ylim()[0] * 3, "clamped",
                 fontsize=8)
    axes[1].set(xlabel=r"$\lambda$", ylabel=r"$D(\lambda)$",
                title="(b) scale invariance of the 3D pipeline (midpoint)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_3d_registration.pdf")
    plt.close(fig)
    return reg, scale


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    print(f"atom-dft-mlxc at {_PKG_ROOT}")
    nitrogen_density_table()  # build / load the cached N atom density
    _self_check_wigner()
    _self_check_reference()

    t0 = time.time()
    stencil_ref = LatticeStencil(H_REF, n_in_for(H_REF))
    print(f"reference lattice h={H_REF}: {len(stencil_ref.points)} stencil "
          f"points, n_in={stencil_ref.n_in} ({time.time() - t0:.1f}s)")
    _self_check_grid(stencil_ref)

    test_convergence()
    test_rotation(stencil_ref)
    test_registration_and_scale(stencil_ref)
    print(f"Wrote simple_3d_convergence.pdf, simple_3d_rotation.pdf, "
          f"simple_3d_registration.pdf to {FIG_DIR}/")
