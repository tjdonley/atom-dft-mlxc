#!/usr/bin/env python3
"""Three-dimensional Cartesian-grid validation of the SIMPLE descriptors.

Where the radial round (radial_validation.py) tested the SIMPLE pipeline
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
    simple_3d_magnitude.pdf    - per-channel resolution: rotation and
                                 registration deviations of each (n, l)
                                 channel against that channel's own
                                 magnitude, exposing the absolute noise
                                 floor and the noise-dominated channels.
    simple_3d_bispectrum.pdf   - bispectrum (third-order Clebsch-Gordan)
                                 invariants: rotational invariance and
                                 mirror parity on a chiral three-atom
                                 cluster, plus the vanishing of the
                                 pseudoscalar components for the (planar,
                                 achiral) pseudo-diatomics. Couplings from
                                 atom/descriptors/simple/invariants.py.

The SIMPLE pipeline, basis, rotations, and CG couplings live in
atom/descriptors/simple/; this script provides the test densities, the
continuum reference, the test harness, and the figures.

Run from the repository root (about 10 minutes):

    python3 tests/simple/cartesian_validation.py
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
from scipy.special import eval_legendre

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atom.descriptors.simple import (  # noqa: E402
    L_MAX,
    LatticeStencil,
    N_OUT,
    R_C,
    bispectrum_components,
    channel_magnitudes,
    flatten_bispectrum,
    grid_descriptors,
    mirror_matrix,
    n_in_for,
    power_spectrum,
    radial_gauss_grid,
    random_rotation_matrix,
    real_sh_rotation_matrix,
    real_sph_harm,
    real_wigner_d,
    rotation_z_to,
    self_check_couplings,
    simple_descriptors,
    simple_from_window,
    window_basis,
)

FIG_DIR = _REPO_ROOT / "docs" / "SIMPLE" / "figures"
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


# =============================================================================
# Self-check of the rotation machinery
# =============================================================================
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
    """Identical atoms at arbitrary positions relative to the evaluation
    center (two for the pseudo-diatomics; any number works and the chiral
    cluster uses three).

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

    __call__ = density  # usable directly as the density callable

    def rotated(self, rot):
        out = Diatomic.__new__(Diatomic)
        out.kind, out.atom, out.lam = self.kind, self.atom, self.lam
        out.positions = [np.asarray(rot) @ p for p in self.positions]
        return out


# Three H atoms, non-coplanar with the evaluation center (det of the
# position matrix is 2.51, far from zero): a CHIRAL environment, needed to
# exercise the pseudoscalar bispectrum components, which vanish identically
# for any planar arrangement (every two-atom environment is planar).
CHIRAL_POSITIONS = ((1.2, 0.4, -0.3), (-0.6, 1.1, 0.5), (0.2, -0.9, 1.3))


def chiral_cluster(lam=1.0):
    return Diatomic("H", [np.array(p) for p in CHIRAL_POSITIONS], lam=lam)


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


def channel_errors(result, reference, l_max=L_MAX):
    """Per-(n, l) deviation ||d_{nl.} - d^ref_{nl.}|| (norm over m)."""
    return np.concatenate(
        [np.linalg.norm(result[l] - reference[l], axis=1)
         for l in range(l_max + 1)]
    )


# =============================================================================
# Self-checks
# =============================================================================
def _self_check_reference():
    """The general-geometry continuum reference must reproduce the 1D round's
    axial code path (central + off-center atom along z at b = 1.5 bohr), and
    be exactly covariant under rotation of the geometry."""
    from radial_validation import environment_profiles

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
    summary, samples, channel_data = {}, {}, {}
    for kind in KINDS:
        for center in ("midpoint", "offaxis"):
            mol = molecule(kind, center)
            ref_cont = continuum_descriptors(mol, stencil.n_in)
            per_l = {l: [] for l in range(L_MAX + 1)}
            spectra, chan_errs, descriptors = [], [], []
            for rot in rotations:
                res = grid_descriptors(mol.rotated(rot), stencil)
                ref_rot = rotate_reference(ref_cont, rot)
                for l, v in per_l_deviation(res, ref_rot).items():
                    per_l[l].append(v)
                spectra.append(power_spectrum(res))
                chan_errs.append(channel_errors(res, ref_rot))
                descriptors.append({l: res[l] for l in range(L_MAX + 1)})
            spectra = np.array(spectra)
            spread = np.abs(spectra - spectra.mean(axis=0)).max() / np.abs(
                spectra.mean(axis=0)
            ).max()
            summary[(kind, center)] = ({l: max(v) for l, v in per_l.items()}, spread)
            samples[(kind, center)] = per_l
            channel_data[(kind, center)] = {
                "magnitude": channel_magnitudes(ref_cont),
                "error": np.array(chan_errs),  # (n_rot, n_channels)
                "spectra": spectra,
                "descriptors": descriptors,  # per-rotation grid results
                "reference": ref_cont,
            }
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
    return summary, channel_data


def test_registration_and_scale(stencil):
    """(a) sub-grid translation of molecule + center; (b) scale invariance."""
    fractions = np.linspace(0.0, 1.0, 9)[1:]
    base = {kind: grid_descriptors(molecule(kind, "offaxis"), stencil)
            for kind in KINDS}
    reg = {kind: [] for kind in KINDS}
    reg_channels = {kind: [] for kind in KINDS}
    for f in fractions:
        # molecule + center shifted together by delta = f h u: the relative
        # geometry is unchanged, the lattice registration becomes -delta/h.
        shifted = LatticeStencil(
            stencil.h, stencil.n_in, frac=tuple((-f * U_SHIFT) % 1.0)
        )
        for kind in KINDS:
            res = grid_descriptors(molecule(kind, "offaxis"), shifted)
            reg[kind].append(deviation_all(res, base[kind]))
            reg_channels[kind].append(channel_errors(res, base[kind]))
        del shifted
    reg = {kind: np.array(v) for kind, v in reg.items()}
    reg_data = {
        kind: {
            "magnitude": channel_magnitudes(base[kind]),
            "error": np.array(reg_channels[kind]),  # (n_shifts, n_channels)
        }
        for kind in KINDS
    }
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
    return reg, scale, reg_data


def test_channel_magnitudes(rot_data, reg_data):
    """Per-channel deviations against each channel's own magnitude.

    A deviation that is small relative to the all-channel norm can still be
    large relative to the channel it lives in. This test resolves the
    rotation and registration fluctuations channel by channel (one channel
    per (n, l), norm over m) and compares each against that channel's own
    magnitude, exposing the absolute noise floor and the magnitude below
    which a channel is noise-dominated."""
    forbidden = 1e-10  # symmetry-forbidden channels (exact zeros) excluded

    def summarize(label, mag, err_max):
        keep = mag > forbidden
        mag_k, err_k = mag[keep], err_max[keep]
        rel = err_k / mag_k
        order = np.argsort(mag_k)[::-1]
        cum = np.cumsum(mag_k[order] ** 2) / np.sum(mag_k**2)
        top = order[: int(np.searchsorted(cum, 0.99)) + 1]
        noise = rel > 1.0
        print(f"channels {label}: magnitudes {mag_k.max():.1e} to "
              f"{mag_k.min():.1e}; error floor (median) "
              f"{np.median(err_k):.1e}; 99%-power channels "
              f"{len(top)}/{len(mag_k)} with max relative error "
              f"{rel[top].max():.1%}; max relative error over all resolved "
              f"channels {rel.max():.1%}; noise-dominated (rel > 100%) "
              f"{int(noise.sum())} channels carrying "
              f"{np.sum(mag_k[noise] ** 2) / np.sum(mag_k**2):.1e} "
              f"of the total power")
        return rel[top].max()

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.2))
    guide_x = np.logspace(-3.6, 0.7, 10)
    for ax, exponent in zip(axes, (1, 1, 1)):
        for frac, label in ((1.0, "100%"), (0.1, "10%"), (0.01, "1%")):
            ax.loglog(guide_x, frac * guide_x**exponent, ls=":", color="0.6",
                      lw=0.9)
            ax.text(guide_x[-1], frac * guide_x[-1] ** exponent, label,
                    fontsize=7, color="0.45", va="bottom", ha="right")

    for kind in KINDS:
        for center in ("midpoint", "offaxis"):
            data = rot_data[(kind, center)]
            keep = data["magnitude"] > forbidden
            mag = data["magnitude"][keep]
            errs = data["error"][:, keep]
            axes[0].loglog(
                np.tile(mag, errs.shape[0]), errs.ravel(), ".", ms=2,
                color=KIND_COLOR[kind], alpha=0.15,
            )
            axes[0].loglog(
                mag, errs.max(axis=0), CENTER_MARKER[center], ls="", ms=4.5,
                color=KIND_COLOR[kind], mfc="none",
                label=f"{KIND_LABEL[kind]}, {center}",
            )
            spectra = data["spectra"]
            p_mean = spectra.mean(axis=0)
            p_keep = p_mean > forbidden**2
            axes[1].loglog(
                p_mean[p_keep],
                np.abs(spectra - p_mean).max(axis=0)[p_keep],
                CENTER_MARKER[center], ls="", ms=4.5,
                color=KIND_COLOR[kind], mfc="none",
            )
    for kind in KINDS:
        data = reg_data[kind]
        keep = data["magnitude"] > forbidden
        axes[2].loglog(
            data["magnitude"][keep], data["error"][:, keep].max(axis=0),
            CENTER_MARKER["offaxis"], ls="", ms=4.5,
            color=KIND_COLOR[kind], mfc="none", label=KIND_LABEL[kind],
        )

    axes[0].set(xlabel=r"channel magnitude $\Vert d_{n\ell\cdot}\Vert$",
                ylabel=r"$\Vert d_{n\ell\cdot} - [D^{(\ell)}d_{\rm exact}]_{n\ell\cdot}\Vert$",
                title="(a) per-channel covariant error, 24 rotations")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set(xlabel=r"$\langle\tilde P_{n\ell}\rangle$",
                ylabel=r"$\max\,|\tilde P_{n\ell} - \langle\tilde P_{n\ell}\rangle|$",
                title="(b) per-channel invariant spread across orientations")
    axes[2].set(xlabel=r"channel magnitude $\Vert d_{n\ell\cdot}(0)\Vert$",
                ylabel=r"$\max_\delta \Vert d_{n\ell\cdot}(\delta) - d_{n\ell\cdot}(0)\Vert$",
                title="(c) per-channel registration deviation")
    axes[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_3d_magnitude.pdf")
    plt.close(fig)

    summaries = {}
    for kind in KINDS:
        data = rot_data[(kind, "offaxis")]
        summaries[kind] = summarize(
            f"{kind}-offaxis rotation", data["magnitude"],
            data["error"].max(axis=0))
        # invariant spread, directly in P space (power fraction = P itself)
        spectra = data["spectra"]
        p_mean = spectra.mean(axis=0)
        keep = p_mean > forbidden**2
        p = p_mean[keep]
        spread = np.abs(spectra - p_mean).max(axis=0)[keep]
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order]) / p.sum()
        top = order[: int(np.searchsorted(cum, 0.99)) + 1]
        print(f"channels {kind}-offaxis P-spread: floor (median) "
              f"{np.median(spread):.1e}; 99%-power channels "
              f"{len(top)}/{len(p)} with max relative spread "
              f"{(spread / p)[top].max():.1%}; max relative spread over all "
              f"resolved channels {(spread / p).max():.1%}")
        summarize(f"{kind}-offaxis registration", reg_data[kind]["magnitude"],
                  reg_data[kind]["error"].max(axis=0))


def test_bispectrum(stencil, rot_data):
    """Bispectrum (third-order CG) invariants on the lattice.

    Three checks: (i) rotational invariance of every component across random
    orientations of a CHIRAL test density (three non-coplanar atoms), which
    populates the pseudoscalar couplings that planar environments cannot;
    (ii) parity -- under a mirror reflection the scalar components (even
    l1+l2+l3) must be unchanged and the pseudoscalar components (odd sum)
    must flip sign exactly; (iii) achirality of the pseudo-diatomics --
    their pseudoscalar components must vanish (any two-atom environment is
    planar), exactly in the continuum and to discretization accuracy on the
    grid."""
    floor_b = 1e-10
    rng = np.random.default_rng(SEED + 7)
    rotations = [random_rotation_matrix(rng) for _ in range(N_ROTATIONS)]
    mirror = mirror_matrix(U_SHIFT)

    # --- chiral cluster: rotation sweep + parity test on the grid
    mol = chiral_cluster()
    ref_cont = continuum_descriptors(mol, stencil.n_in)
    b_cont, parities = flatten_bispectrum(bispectrum_components(ref_cont))
    signs = np.where(parities == 0, 1.0, -1.0)
    b_base = flatten_bispectrum(
        bispectrum_components(grid_descriptors(mol, stencil)))[0]
    b_rot = np.array([
        flatten_bispectrum(bispectrum_components(
            grid_descriptors(mol.rotated(rot), stencil)))[0]
        for rot in rotations
    ])
    b_mirror = flatten_bispectrum(bispectrum_components(
        grid_descriptors(mol.rotated(mirror), stencil)))[0]
    b_cont_mirror = flatten_bispectrum(bispectrum_components(
        continuum_descriptors(mol.rotated(mirror), stencil.n_in)))[0]
    err_parity_cont = np.abs(b_cont_mirror - signs * b_cont).max()

    b_mean = b_rot.mean(axis=0)
    spread = np.abs(b_rot - b_mean).max(axis=0)
    parity_err = np.abs(b_mirror - signs * b_base)
    resolved = np.abs(b_mean) > floor_b
    rel_spread = spread[resolved] / np.abs(b_mean)[resolved]
    order = np.argsort(np.abs(b_mean)[resolved])[::-1]
    cum = np.cumsum(np.abs(b_mean)[resolved][order] ** 2)
    top = order[: int(np.searchsorted(cum / cum[-1], 0.99)) + 1]
    n_pseudo = int(np.sum(resolved & (parities == 1)))
    bad = rel_spread > 0.10
    threshold = np.abs(b_mean)[resolved][bad].max() if bad.any() else 0.0
    print(f"bispectrum chiral cluster: {resolved.sum()}/{b_mean.size} "
          f"resolved components ({n_pseudo} pseudoscalar); magnitudes "
          f"{np.abs(b_mean)[resolved].max():.1e} to "
          f"{np.abs(b_mean)[resolved].min():.1e}; 99%-power max relative "
          f"spread {rel_spread[top].max():.1%}; absolute spread floor "
          f"(median) {np.median(spread[resolved]):.1e}; every component "
          f"with |B| > {threshold:.1e} has relative spread < 10%; "
          f"accuracy vs continuum {np.abs(b_base - b_cont).max():.1e}")
    print(f"bispectrum parity (mirror): grid residual "
          f"{parity_err.max():.1e} (max component {np.abs(b_mean).max():.1e}); "
          f"continuum residual {err_parity_cont:.1e}")

    # --- pseudo-diatomics are planar, hence achiral: pseudoscalars vanish
    achiral = {}
    for kind in KINDS:
        data = rot_data[(kind, "offaxis")]
        b_dia_cont, par_dia = flatten_bispectrum(
            bispectrum_components(data["reference"]))
        b_dia_grid = np.array([
            flatten_bispectrum(bispectrum_components(res))[0]
            for res in data["descriptors"]
        ])
        pseudo = par_dia == 1
        achiral[kind] = (
            np.abs(b_dia_cont[pseudo]).max(),
            np.abs(b_dia_grid[:, pseudo]).max(),
            np.abs(b_dia_cont).max(),
        )
        print(f"bispectrum achirality {kind}-offaxis: max pseudoscalar "
              f"|B| continuum {achiral[kind][0]:.1e}, grid "
              f"{achiral[kind][1]:.1e} (max scalar |B| "
              f"{achiral[kind][2]:.1e})")

    # --- figure
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    guide_x = np.logspace(-7.2, 0.7, 10)
    for ax in axes:
        for frac, label in ((1.0, "100%"), (0.01, "1%")):
            ax.loglog(guide_x, frac * guide_x, ls=":", color="0.6", lw=0.9)
            ax.text(guide_x[-1], frac * guide_x[-1], label, fontsize=7,
                    color="0.45", va="bottom", ha="right")
    style = {0: ("tab:green", "scalar ($\\ell_1{+}\\ell_2{+}\\ell_3$ even)"),
             1: ("tab:purple", "pseudoscalar (odd)")}
    for par, (color, label) in style.items():
        mask = resolved & (parities == par)
        axes[0].loglog(np.abs(b_mean)[mask], spread[mask], ".", ms=3.5,
                       color=color, label=label)
        axes[1].loglog(np.abs(b_base)[mask], parity_err[mask], ".", ms=3.5,
                       color=color, label=label)
    axes[0].set(xlabel=r"$|\langle\tilde B\rangle|$",
                ylabel=r"$\max\,|\tilde B - \langle\tilde B\rangle|$",
                title=f"(a) invariance across {N_ROTATIONS} rotations "
                      f"(chiral cluster, $h={stencil.h}$)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set(xlabel=r"$|\tilde B|$",
                ylabel=r"$|\tilde B_{\rm mirror} - (-1)^{p}\,\tilde B|$",
                title="(b) parity under reflection (grid)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simple_3d_bispectrum.pdf")
    plt.close(fig)
    return rel_spread, parity_err, achiral


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"repository root {_REPO_ROOT}")
    nitrogen_density_table()  # build / load the cached N atom density
    _self_check_wigner()
    self_check_couplings()
    _self_check_reference()

    t0 = time.time()
    stencil_ref = LatticeStencil(H_REF, n_in_for(H_REF))
    print(f"reference lattice h={H_REF}: {len(stencil_ref.points)} stencil "
          f"points, n_in={stencil_ref.n_in} ({time.time() - t0:.1f}s)")
    _self_check_grid(stencil_ref)

    test_convergence()
    _, rot_channels = test_rotation(stencil_ref)
    _, _, reg_channels = test_registration_and_scale(stencil_ref)
    test_channel_magnitudes(rot_channels, reg_channels)
    test_bispectrum(stencil_ref, rot_channels)
    print(f"Wrote simple_3d_convergence.pdf, simple_3d_rotation.pdf, "
          f"simple_3d_registration.pdf, simple_3d_magnitude.pdf, "
          f"simple_3d_bispectrum.pdf to {FIG_DIR}/")
