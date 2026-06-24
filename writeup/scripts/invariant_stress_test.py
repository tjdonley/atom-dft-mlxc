#!/usr/bin/env python3
"""Stress-test of the SIMPLE invariants on realistic 3D Cartesian grids.

Goal: map the PRACTICAL LIMITS of the SIMPLE features -- not to show everything is
invariant, but to find where each feature breaks on real-space grids.

Features tested (through l=3):
  * s   -- dimensionless reduced gradient   |grad rho| / (2 k_F rho)        [l=1]
  * q   -- dimensionless reduced Laplacian  grad^2 rho / (4 k_F^2 rho)      [l=0, known unstable]
  * P   -- power spectrum   P_{nl} = sum_m rho_{nlm}^2                      (40 entries)
  * B   -- bispectrum (real-CG, all triangle triples l1<=l2<=l3)           (13000 entries)

Run at the fixed production settings R_c=6 bohr, n_out=10, n_in=20 (Lambda_max=2).

Systems (densities = linear combinations of single-atom radial densities; not SCF):
  pseudo-H2, pseudo-N2, pseudo-H2O, bulk Al (FCC), bulk Pt (FCC).

Cutoff R_c = 6 bohr (fixed, runtime override). Realistic grid spacings h = 0.1, 0.2, 0.3 bohr.
Ground truth = the analytic continuum-quadrature reference (P, B) and a finite-difference
reference on the analytic density (s, q) -- exact and cheap, so no fine grid is needed.

Tests: (1) translation (sub-grid registration), (2) rotation (random rigid rotations),
(3) scale (uniform rho->lambda^3 rho(lambda r)), (4) grid-spacing sensitivity (vs continuum).
Metric: relative error; entries with relative error > 10% are FLAGGED, unless the feature
magnitude is below a per-group floor (a large relative error on a ~zero feature is not
meaningful).

Writes data/invariant_stress_test.json and figures/invariance_*.pdf.
Run from the repository root:  python3 writeup/scripts/invariant_stress_test.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# R_c = 6 bohr override (runtime only; no source edits). R_C is imported by
# value into pipeline/derivatives, and _DEFAULT_RADIUS_SAMPLES + the basis cache
# are precomputed from it -- reset all three before any descriptor call.
# ---------------------------------------------------------------------------
R_CUT = 6.0
import atom.descriptors.simple.params as _params      # noqa: E402
import atom.descriptors.simple.pipeline as _pipe       # noqa: E402
import atom.descriptors.simple.derivatives as _deriv    # noqa: E402
for _m in (_params, _pipe, _deriv):
    _m.R_C = R_CUT
_pipe._DEFAULT_RADIUS_SAMPLES = np.linspace(R_CUT / 1024.0, R_CUT, 1024)
_pipe._BASIS_CACHE.clear()

from atom.descriptors.simple.pipeline import (  # noqa: E402
    LatticeStencil, simple_from_window, window_basis, n_in_for,
    invert_enclosed_moment,
)
from atom.descriptors.simple.bessel import (  # noqa: E402
    spherical_jn_zeros, a_n_closed_form, radial_gauss_grid,
)
from atom.descriptors.simple.invariants import (  # noqa: E402
    power_spectrum, bispectrum_components, flatten_bispectrum,
)
from atom.descriptors.simple.rotations import (  # noqa: E402
    real_sph_harm, real_wigner_d, random_rotation_matrix,
)
from atom.descriptors.simple.params import L_MAX  # noqa: E402
from atom.solver import AtomicDFTSolver  # noqa: E402

# Fixed production parameters (R_c=6 bohr set by the override above). n_out and n_in
# are decoupled from the grid: n_in is the design choice n_out*Lambda_max (Lambda_max=2),
# giving N_conv=n_in*(l_max+1)^2=320 at l_max=3 (App. parameter selection). The whole
# invariance suite is run at these fixed values for a practical assessment.
N_OUT = 10
NIN = 20

_DATA = Path(__file__).resolve().parent / "data"
_FIG = Path(__file__).resolve().parent.parent / "figures"
_OUT = _DATA / "invariant_stress_test.json"
BOHR = 1.8897259886                       # bohr per angstrom
trapz = getattr(np, "trapezoid", None) or np.trapz

_3PI2 = 3.0 * np.pi ** 2
_S_DEN = 2.0 * _3PI2 ** (1.0 / 3.0)        # s = |grad rho| / (_S_DEN rho^{4/3})
_Q_DEN = 4.0 * _3PI2 ** (2.0 / 3.0)        # q = lap rho / (_Q_DEN rho^{5/3})

# Per-feature-group magnitude floors (relative error below the floor is not flagged).
FLOOR = {"s": 1e-2, "q": 1e-2, "power_spectrum": 1e-8, "bispectrum": 1e-8}
FLAG = 0.10                                 # relative-error flag threshold (10%)


# =============================================================================
# Single-atom radial densities (total density; H analytic, others psp8 valence)
# =============================================================================
_Z = {"N": 7, "O": 8, "Al": 13, "Pt": 78}


class RadialDensity:
    """Total radial density rho(r) of one atom, with first/second derivatives."""

    def __init__(self, element):
        self.element = element
        if element == "H":
            self._mode = "H"
            self.rmax = 30.0
            return
        Z = _Z[element]
        cache = _DATA / f"{element}_Z{Z}_pbe_psp8.npz"
        if not cache.exists():
            print(f"  generating {element} (Z={Z}) valence density (GGA_PBE, psp8)...")
            solver = AtomicDFTSolver(atomic_number=Z, xc_functional="GGA_PBE",
                                     all_electron_flag=False, verbose=False)
            res = solver.solve()
            assert res["converged"], f"{element} SCF did not converge"
            r = np.asarray(res["quadrature_nodes"], float)
            rho = np.asarray(res["rho"], float)
            o = np.argsort(r)
            r, rho = r[o], rho[o]
            keep = np.concatenate([[True], np.diff(r) > 1e-12])
            r, rho = r[keep], rho[keep]
            _DATA.mkdir(exist_ok=True)
            np.savez(cache, r=r, rho=rho)
        d = np.load(cache)
        r, rho = d["r"], d["rho"]
        self._mode = "spline"
        self._cs = CubicSpline(r, rho, extrapolate=False)
        self._r0, self.rmax = float(r[0]), float(r[-1])

    def _eval(self, r, nu):
        r = np.asarray(r, float)
        if self._mode == "H":
            e = np.exp(-2.0 * r) / np.pi
            return {0: e, 1: -2.0 * e, 2: 4.0 * e}[nu]
        rc = np.clip(r, self._r0, self.rmax)
        val = self._cs(rc, nu)
        return np.where(r <= self.rmax, val, 0.0)

    def rho(self, r):
        return self._eval(r, 0)

    def d1(self, r):
        return self._eval(r, 1)

    def d2(self, r):
        return self._eval(r, 2)


_DENS_CACHE = {}


def radial_density(element):
    if element not in _DENS_CACHE:
        _DENS_CACHE[element] = RadialDensity(element)
    return _DENS_CACHE[element]


# =============================================================================
# Cluster: superposition of atomic densities at sites (lam = uniform scaling)
# =============================================================================
class Cluster:
    """Total density rho(x) = sum_a lam^3 rho_a(lam |x - p0_a/lam|), evaluation
    center at the origin. ``rotated`` rotates the (lam=1) site positions; ``scaled``
    sets the uniform-scaling factor lam (rho -> lam^3 rho(lam r))."""

    def __init__(self, sites, lam=1.0):
        self.sites = [(radial_density(el), np.asarray(p, float)) for el, p in sites]
        self.lam = float(lam)

    def density(self, points):
        points = np.atleast_2d(np.asarray(points, float))
        rho = np.zeros(len(points))
        for dens, p0 in self.sites:
            p = p0 / self.lam
            d = np.linalg.norm(points - p[None, :], axis=1)
            rho += dens.rho(self.lam * d)
        return self.lam ** 3 * rho

    def rotated(self, rot):
        out = Cluster.__new__(Cluster)
        out.lam = self.lam
        out.sites = [(dens, np.asarray(rot) @ p0) for dens, p0 in self.sites]
        return out

    def scaled(self, lam):
        out = Cluster.__new__(Cluster)
        out.lam = float(lam)
        out.sites = list(self.sites)
        return out


# =============================================================================
# System factories (positions relative to the evaluation center, in bohr)
# =============================================================================
def _fcc_sites(element, a, center, margin=3.0):
    """FCC lattice sites within R_c + margin of the evaluation center.

    center='atom': a lattice atom sits at the origin. center='octa': the
    octahedral interstitial (a/2,0,0) sits at the origin (a low-symmetry-for-the-
    grid but high-symmetry-for-the-crystal point)."""
    basis = a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    reach = int(np.ceil((R_CUT + margin) / a)) + 1
    pts = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                cell = a * np.array([i, j, k])
                for b in basis:
                    pts.append(cell + b)
    pts = np.array(pts)
    origin = {"atom": np.zeros(3), "octa": a * np.array([0.5, 0.0, 0.0])}[center]
    pts = pts - origin[None, :]
    keep = np.linalg.norm(pts, axis=1) <= R_CUT + margin
    return [(element, p) for p in pts[keep]]


# Standard geometries.
_B_H2 = 1.40                                   # bohr
_B_N2 = 2.074                                  # bohr
_OH = 0.96 * BOHR                              # 1.814 bohr
_HOH = np.deg2rad(104.5)
_A_AL = 4.05 * BOHR                            # 7.653 bohr
_A_PT = 3.92 * BOHR                            # 7.408 bohr
_UAX = np.array([1.0, 0.6, 0.3]); _UAX /= np.linalg.norm(_UAX)   # low-symmetry axis
_UPERP = np.cross(_UAX, [0.0, 0.0, 1.0]); _UPERP /= np.linalg.norm(_UPERP)


def _diatomic(el, b, center):
    half = 0.5 * b * _UAX
    if center == "atom":
        return [(el, np.zeros(3)), (el, b * _UAX)]
    if center == "bond":
        return [(el, half), (el, -half)]
    if center == "offaxis":          # 1.2 bohr off the bond axis
        o = 1.2 * _UPERP
        return [(el, half + o), (el, -half + o)]
    raise ValueError(center)


def _water(center):
    # O at origin; two H in a low-symmetry plane at the H-O-H angle.
    e1 = _UAX
    e2 = _UPERP
    h1 = _OH * (np.cos(_HOH / 2) * e1 + np.sin(_HOH / 2) * e2)
    h2 = _OH * (np.cos(_HOH / 2) * e1 - np.sin(_HOH / 2) * e2)
    sites = [("O", np.zeros(3)), ("H", h1), ("H", h2)]
    if center == "O":
        return sites
    if center == "H":                # recenter on one H
        return [(el, p - h1) for el, p in sites]
    if center == "offsite":          # a generic point off all atoms
        shift = np.array([0.8, -0.5, 0.6])
        return [(el, p - shift) for el, p in sites]
    raise ValueError(center)


# Each system: label -> {center_name: builder() -> list of sites}.
SYSTEMS = {
    "pseudo-H2": {"atom": lambda: _diatomic("H", _B_H2, "atom"),
                  "offaxis": lambda: _diatomic("H", _B_H2, "offaxis")},
    "pseudo-N2": {"atom": lambda: _diatomic("N", _B_N2, "atom"),
                  "offaxis": lambda: _diatomic("N", _B_N2, "offaxis")},
    "pseudo-H2O": {"O": lambda: _water("O"),
                   "offsite": lambda: _water("offsite")},
    "bulk-Al": {"atom": lambda: _fcc_sites("Al", _A_AL, "atom"),
                "octa": lambda: _fcc_sites("Al", _A_AL, "octa")},
    "bulk-Pt": {"atom": lambda: _fcc_sites("Pt", _A_PT, "atom"),
                "octa": lambda: _fcc_sites("Pt", _A_PT, "octa")},
}
SYS_COLOR = {"pseudo-H2": "tab:blue", "pseudo-N2": "tab:red",
             "pseudo-H2O": "tab:green", "bulk-Al": "tab:purple",
             "bulk-Pt": "tab:orange"}


# =============================================================================
# Continuum (quasi-exact) reference for a multi-type cluster
# =============================================================================
_UN, _UW = np.polynomial.legendre.leggauss(96)


def _rotation_z_to(direction):
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, u))
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    ang = np.arctan2(s, c)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * np.cos(ang) + np.sin(ang) * K + (1 - np.cos(ang)) * np.outer(axis, axis)


def _axial_profile(dens, b, lam, l, r):
    """l-th axial multipole of one scaled atom at distance b from the center."""
    r = np.atleast_1d(np.asarray(r, float))
    rs = lam * r
    bs = lam * b
    if bs < 1e-12:
        return (lam ** 3 * dens.rho(rs)) if l == 0 else np.zeros(rs.size)
    from scipy.special import eval_legendre
    dist = np.sqrt(np.maximum(rs[:, None] ** 2 + bs ** 2
                              - 2.0 * rs[:, None] * bs * _UN[None, :], 0.0))
    return lam ** 3 * (2 * l + 1) / 2.0 * np.sum(
        _UW * eval_legendre(l, _UN) * dens.rho(dist), axis=1)


def continuum_window(cluster, n_in, l_max=L_MAX):
    """Continuum window coefficients c_window[l] (n_in, 2l+1) for the cluster."""
    quad = radial_gauss_grid(R_CUT, 512)
    r = quad.nodes
    c_window = {l: np.zeros((n_in, 2 * l + 1)) for l in range(l_max + 1)}
    for dens, p0 in cluster.sites:
        p = p0 / cluster.lam
        b = float(np.linalg.norm(p))
        on_center = b < 1e-12
        dblocks = None if on_center else {
            l: real_wigner_d(l, _rotation_z_to(p)) for l in range(1, l_max + 1)}
        for l in range(l_max + 1):
            if on_center and l > 0:
                continue
            basis = window_basis(l, n_in)
            ang = np.sqrt(4.0 * np.pi / (2 * l + 1))
            c_axial = ang * np.einsum(
                "nj,j->n", basis.evaluate(l, r),
                quad.weights * r ** 2 * _axial_profile(dens, b, cluster.lam, l, r))
            if l == 0:
                c_window[0][:, 0] += c_axial
            else:
                c_window[l] += np.outer(c_axial, dblocks[l][:, l])   # D[:, m=0]
    return c_window


def continuum_descriptors(cluster, n_in, l_max=L_MAX):
    c_window = continuum_window(cluster, n_in, l_max)

    def rho_bar(radius):
        sub = radial_gauss_grid(radius, 192)
        rho0 = sum(_axial_profile(dens, float(np.linalg.norm(p0 / cluster.lam)),
                                  cluster.lam, 0, sub.nodes)
                   for dens, p0 in cluster.sites)
        return 3.0 / radius ** 3 * float(np.sum(sub.weights * sub.nodes ** 2 * rho0))

    d = simple_from_window(c_window, rho_bar, n_in, l_max, N_OUT)
    return d, c_window


# =============================================================================
# s and q from window coefficients (grid or continuum)
# =============================================================================
def _slope_l1(n_in):
    eps = 1.0e-6
    return window_basis(1, n_in).evaluate(1, np.array([eps]))[:, 0] / eps   # R'_{n1}(0)


def _lap_weights(n_in):
    k = (np.arange(n_in) + 1) * np.pi / R_CUT            # l=0 zeros / R_c
    R0 = window_basis(0, n_in).evaluate(0, np.array([1.0e-7]))[:, 0]
    return -k ** 2 * R0                                  # spectral Laplacian weights


def s_q_from_window(c_window, n_in, rho_center):
    """Reconstruct s and q at the center from the l=1 and l=0 window coefficients.
    grad rho = sqrt(3/4pi) sum_n c_{n1m} R'_{n1}(0);
    lap rho  = (1/sqrt(4pi)) sum_n c_{n00} (-k_n^2 R_{n0}(0))  [the unstable channel]."""
    grad = np.sqrt(3.0 / (4.0 * np.pi)) * np.linalg.norm(_slope_l1(n_in) @ c_window[1])
    lap = (1.0 / np.sqrt(4.0 * np.pi)) * float(_lap_weights(n_in) @ c_window[0][:, 0])
    rho = max(float(rho_center), 1e-300)
    s = grad / (_S_DEN * rho ** (4.0 / 3.0))
    q = lap / (_Q_DEN * rho ** (5.0 / 3.0))
    return s, q


def fd_s_q(cluster, h=2.0e-3):
    """Finite-difference s, q at the center on the analytic density (ground truth)."""
    e = np.eye(3) * h
    pts = np.vstack([np.zeros(3), e, -e])
    rho = cluster.density(pts)
    rho0 = rho[0]
    grad = (rho[1:4] - rho[4:7]) / (2.0 * h)
    lap = float(np.sum(rho[1:4] + rho[4:7] - 2.0 * rho0) / h ** 2)
    rho0 = max(rho0, 1e-300)
    s = np.linalg.norm(grad) / (_S_DEN * rho0 ** (4.0 / 3.0))
    q = lap / (_Q_DEN * rho0 ** (5.0 / 3.0))
    return s, q


# =============================================================================
# Feature extraction (grid and continuum) -> {group: vector}
# =============================================================================
def grid_features(cluster, stencil):
    rho = cluster.density(stencil.points)
    cw = stencil.window_coefficients(rho)
    radius = invert_enclosed_moment(*stencil.enclosed_moment_curve(rho))

    def rho_bar(R):
        w = stencil.shell_weights(R)
        ws = float(np.sum(w))
        return float(np.sum(w * rho) / ws) if ws > 0 else float(rho[np.argmin(stencil.dist)])

    d = simple_from_window(cw, rho_bar, stencil.n_in, L_MAX, N_OUT, radius=radius)
    rho_c = float(cluster.density(np.zeros((1, 3)))[0])
    s, q = s_q_from_window(cw, stencil.n_in, rho_c)
    return _assemble(d, s, q)


def continuum_features(cluster, n_in):
    d, cw = continuum_descriptors(cluster, n_in)
    rho_c = float(cluster.density(np.zeros((1, 3)))[0])
    s, q = s_q_from_window(cw, n_in, rho_c)
    return _assemble(d, s, q)


def _assemble(d, s, q):
    return {
        "s": np.array([s]),
        "q": np.array([q]),
        "power_spectrum": power_spectrum(d, L_MAX),
        "bispectrum": flatten_bispectrum(bispectrum_components(d, L_MAX))[0],
    }


# =============================================================================
# Relative-error / flagging
# =============================================================================
def rel_error(test, ref, group):
    test, ref = np.atleast_1d(test), np.atleast_1d(ref)
    mag = np.abs(ref)
    floor = max(FLOOR[group], 1e-3 * float(mag.max()) if mag.size else 0.0)
    keep = mag > floor
    re = np.abs(test - ref) / np.maximum(mag, 1e-300)
    return re, keep


def summarize(test, ref, group):
    """max/median relative error over above-floor entries, and the flag count."""
    re, keep = rel_error(test, ref, group)
    if not np.any(keep):
        return {"max": 0.0, "median": 0.0, "n_above_floor": 0, "n_flagged": 0}
    rek = re[keep]
    return {"max": float(rek.max()), "median": float(np.median(rek)),
            "n_above_floor": int(keep.sum()),
            "n_flagged": int(np.sum(rek > FLAG))}


GROUPS = ["s", "q", "power_spectrum", "bispectrum"]


# =============================================================================
# Tests
# =============================================================================
SPACINGS = (0.1, 0.2, 0.3)
H_INV = 0.2                                   # representative spacing for tests 1-3
N_ROT = 8
SCALES = np.array([0.6, 0.75, 0.9, 1.0, 1.1, 1.33, 1.67])
SHIFTS = np.linspace(0.0, 1.0, 6)[1:]          # sub-grid shift fractions of h
_SEED = 20260623
_SHIFT_DIR = np.array([0.7, 0.45, 0.25]); _SHIFT_DIR /= np.linalg.norm(_SHIFT_DIR)


def _iter_cases():
    for sysname, centers in SYSTEMS.items():
        for cname, builder in centers.items():
            yield sysname, cname, builder


def test_grid_sensitivity():
    """grid(h) vs continuum(same n_in): isolates grid-discretization error.
    Also reports the reconstruction ceiling continuum(s,q) vs FD-exact."""
    print("\n=== (4) grid-spacing sensitivity (grid vs continuum) ===")
    out = {}
    for sysname, cname, builder in _iter_cases():
        cl = Cluster(builder())
        fd_s, fd_q = fd_s_q(cl)
        key = f"{sysname}/{cname}"
        out[key] = {"per_h": {}, "reconstruction_vs_fd": {}}
        for h in SPACINGS:
            n_in = NIN                                # fixed production inner channels
            st = LatticeStencil(h, n_in)
            gf = grid_features(cl, st)
            cf = continuum_features(cl, n_in)
            out[key]["per_h"][h] = {g: summarize(gf[g], cf[g], g) for g in GROUPS}
            del st
        # reconstruction ceiling at the production channel count
        cf_fine = continuum_features(cl, NIN)
        out[key]["reconstruction_vs_fd"] = {
            "s": float(abs(cf_fine["s"][0] - fd_s) / max(abs(fd_s), 1e-30)),
            "q": float(abs(cf_fine["q"][0] - fd_q) / max(abs(fd_q), 1e-30)),
            "fd_s": float(fd_s), "fd_q": float(fd_q),
            "recon_s": float(cf_fine["s"][0]), "recon_q": float(cf_fine["q"][0])}
        rc = out[key]["reconstruction_vs_fd"]
        print(f"  {key:22s} h0.2 P/B max {out[key]['per_h'][0.2]['power_spectrum']['max']:.3f}"
              f"/{out[key]['per_h'][0.2]['bispectrum']['max']:.3f}; "
              f"recon s/q err {rc['s']:.2f}/{rc['q']:.2f}")
    return out


def test_rotation():
    """Spread of the (rotation-invariant) features across random rigid rotations."""
    print(f"\n=== (2) rotational invariance (spread over {N_ROT} rotations, h={H_INV}) ===")
    rng = np.random.default_rng(_SEED)
    rots = [random_rotation_matrix(rng) for _ in range(N_ROT)]
    n_in = NIN
    st = LatticeStencil(H_INV, n_in)
    out = {}
    for sysname, cname, builder in _iter_cases():
        cl = Cluster(builder())
        feats = [grid_features(cl.rotated(R), st) for R in rots]
        key = f"{sysname}/{cname}"
        out[key] = {}
        for g in GROUPS:
            stack = np.array([f[g] for f in feats])          # (N_ROT, dim)
            mean = stack.mean(axis=0)
            mag = np.abs(mean)
            floor = max(FLOOR[g], 1e-3 * float(mag.max()) if mag.size else 0.0)
            keep = mag > floor
            spread = np.abs(stack - mean[None, :]).max(axis=0) / np.maximum(mag, 1e-300)
            sk = spread[keep] if np.any(keep) else np.array([0.0])
            out[key][g] = {"max": float(sk.max()), "median": float(np.median(sk)),
                           "n_above_floor": int(keep.sum()),
                           "n_flagged": int(np.sum(sk > FLAG))}
        print(f"  {key:22s} spread s/q/P/B max "
              + "/".join(f"{out[key][g]['max']:.3f}" for g in GROUPS))
    del st
    return out


def test_translation():
    """Deviation under sub-grid registration shifts (system fixed, lattice shifted)."""
    print(f"\n=== (1) translation invariance (sub-grid shifts, h={H_INV}) ===")
    n_in = NIN
    base_st = LatticeStencil(H_INV, n_in, frac=(0.0, 0.0, 0.0))
    out = {}
    for sysname, cname, builder in _iter_cases():
        cl = Cluster(builder())
        ref = grid_features(cl, base_st)
        key = f"{sysname}/{cname}"
        per_g = {g: [] for g in GROUPS}
        for f in SHIFTS:
            frac = tuple((f * _SHIFT_DIR) % 1.0)
            st = LatticeStencil(H_INV, n_in, frac=frac)
            gf = grid_features(cl, st)
            for g in GROUPS:
                per_g[g].append(summarize(gf[g], ref[g], g))
            del st
        out[key] = {g: {"max": max(s["max"] for s in per_g[g]),
                        "n_flagged": max(s["n_flagged"] for s in per_g[g])}
                    for g in GROUPS}
        print(f"  {key:22s} max-over-shifts s/q/P/B "
              + "/".join(f"{out[key][g]['max']:.3f}" for g in GROUPS))
    del base_st
    return out


def test_scale():
    """Deviation of the dimensionless features under uniform scaling vs lambda=1."""
    print(f"\n=== (3) scale invariance (uniform lambda sweep, h={H_INV}) ===")
    n_in = NIN
    st = LatticeStencil(H_INV, n_in)
    out = {}
    for sysname, cname, builder in _iter_cases():
        cl = Cluster(builder())
        ref = grid_features(cl, st)
        key = f"{sysname}/{cname}"
        curves = {g: [] for g in GROUPS}
        for lam in SCALES:
            gf = grid_features(cl.scaled(lam), st)
            for g in GROUPS:
                curves[g].append(summarize(gf[g], ref[g], g)["max"])
        out[key] = {g: {"max": float(np.max(curves[g])),
                        "curve": [float(x) for x in curves[g]]} for g in GROUPS}
        print(f"  {key:22s} max-over-lambda s/q/P/B "
              + "/".join(f"{out[key][g]['max']:.3f}" for g in GROUPS))
    del st
    return out


def _stack_d(d, lmax=L_MAX):
    return np.concatenate([np.asarray(d[l]).ravel() for l in range(lmax + 1)])


RES_NINS = (16, 20, 32, 48)   # 20 = production n_in
RES_SYSTEMS = [("pseudo-N2", "atom"), ("pseudo-N2", "offaxis"), ("pseudo-H2", "atom")]


def test_scale_resolution():
    """Diagnose the scale-invariance deviations: they are the inner-channel (n_in)
    resolution bound of the scale transform (k* lambda <~ n_in pi/R_c, Sec. scale),
    NOT the adaptive-radius clamp. Computed in the CONTINUUM (no grid error); for a
    ladder of n_in we record the descriptor-vector L2 deviation D(lambda) (the Fig. 3
    metric) and the power-spectrum max-entry deviation, plus the adaptive radius (which
    stays well below R_c here -- unclamped). Deviations collapse as n_in grows."""
    print("\n=== (3b) scale invariance vs inner-channel count n_in (continuum) ===")
    out = {}
    for sysname, cname in RES_SYSTEMS:
        cl = Cluster(SYSTEMS[sysname][cname]())
        key = f"{sysname}/{cname}"
        out[key] = {"n_in": list(RES_NINS), "lambda": [float(x) for x in SCALES],
                    "D_vec": {}, "P_max": {}, "r_ad": {}, "any_clamped": False}
        for n_in in RES_NINS:
            d1, _ = continuum_descriptors(cl, n_in)
            v1, P1 = _stack_d(d1), power_spectrum(d1, L_MAX)
            Dv, Pm, rad = [], [], []
            for lam in SCALES:
                dL, _ = continuum_descriptors(cl.scaled(lam), n_in)
                Dv.append(float(np.linalg.norm(_stack_d(dL) - v1) / np.linalg.norm(v1)))
                Pm.append(float(summarize(power_spectrum(dL, L_MAX), P1, "power_spectrum")["max"]))
                rad.append(float(dL["r_ad"]))
                out[key]["any_clamped"] |= bool(dL["clamped"])
            out[key]["D_vec"][n_in] = Dv
            out[key]["P_max"][n_in] = Pm
            out[key]["r_ad"][n_in] = rad
        print(f"  {key:22s} D_vec(lambda=1.67): "
              + ", ".join(f"n{n}={out[key]['D_vec'][n][-1]:.3f}" for n in RES_NINS)
              + f"  | R_ad in [{min(out[key]['r_ad'][16]):.2f},"
              f"{max(out[key]['r_ad'][16]):.2f}] bohr, clamped={out[key]['any_clamped']}")
    return out


# =============================================================================
# Plots (paper style)
# =============================================================================
_GLABEL = {"s": r"reduced gradient $s$", "q": r"reduced Laplacian $q$",
           "power_spectrum": "power spectrum", "bispectrum": "bispectrum"}


def _perh(per_h, h):
    """Fetch a per-h entry whether the key is a float (live) or a string (from JSON)."""
    return per_h[h] if h in per_h else per_h[str(h)]


def _panel_grid(ax_list, results_grid):
    hs = np.array(SPACINGS)
    for ax, g in zip(ax_list, GROUPS):
        for sysname in SYSTEMS:
            ys = [max(_perh(results_grid[f"{sysname}/{c}"]["per_h"], h)[g]["max"]
                      for c in SYSTEMS[sysname]) for h in SPACINGS]
            ax.loglog(hs, ys, "o-", color=SYS_COLOR[sysname], label=sysname, ms=4)
        ax.axhline(FLAG, color="0.5", ls="--", lw=0.8)
        ax.loglog(hs, ys[-1] * (hs / hs[-1]) ** 2, "k:", lw=0.8)
        ax.set(xlabel=r"$h$ (bohr)", ylabel="max rel. error vs continuum",
               title=_GLABEL[g])


def plot_grid(results_grid):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    _panel_grid(axes.ravel(), results_grid)
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle("(4) Grid-spacing sensitivity: grid vs continuum (R_c=6 bohr); "
                 "dashed = 10% flag, dotted = O(h^2)")
    fig.tight_layout()
    fig.savefig(_FIG / "invariance_grid.pdf")
    plt.close(fig)


def _bar_panel(ax, results, g, title):
    keys = list(results.keys())
    vals = [results[k][g]["max"] for k in keys]
    colors = [SYS_COLOR[k.split("/")[0]] for k in keys]
    ax.bar(range(len(keys)), np.maximum(vals, 1e-6), color=colors)
    ax.axhline(FLAG, color="0.5", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("pseudo-", "").replace("bulk-", "") for k in keys],
                       rotation=60, ha="right", fontsize=6)
    ax.set(ylabel="max rel. dev.", title=title)


def plot_bars(results, fname, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5))
    for ax, g in zip(axes.ravel(), GROUPS):
        _bar_panel(ax, results, g, _GLABEL[g])
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(_FIG / fname)
    plt.close(fig)


def plot_scale_resolution(res):
    n_sys = len(RES_SYSTEMS)
    fig, axes = plt.subplots(1, n_sys, figsize=(3.7 * n_sys, 3.6), squeeze=False)
    for ax, (sysname, cname) in zip(axes[0], RES_SYSTEMS):
        key = f"{sysname}/{cname}"
        for n_in in RES_NINS:
            dv = res[key]["D_vec"]
            yv = dv[n_in] if n_in in dv else dv[str(n_in)]
            ax.semilogy(SCALES, np.maximum(yv, 1e-6),
                        "o-", ms=3, label=fr"$n_{{\rm in}}={n_in}$")
        ax.axhline(0.017, color="0.5", ls=":", lw=0.8)        # Fig. 3 continuum level
        ax.set(xlabel=r"scaling $\lambda$",
               ylabel=r"$D(\lambda)=\Vert \varrho(\lambda)-\varrho(1)\Vert/\Vert \varrho(1)\Vert$",
               title=key)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(r"Scale invariance is set by the inner-channel count $n_{\rm in}$ "
                 r"(continuum, $R_c=6$ bohr; $R_{\rm ad}$ unclamped); "
                 r"dotted $=$ Fig.~3 continuum level")
    fig.tight_layout()
    fig.savefig(_FIG / "invariance_scale_resolution.pdf")
    plt.close(fig)


def plot_scale(results_scale):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for ax, g in zip(axes.ravel(), GROUPS):
        for sysname in SYSTEMS:
            for c in SYSTEMS[sysname]:
                ax.semilogy(SCALES, np.maximum(results_scale[f"{sysname}/{c}"][g]["curve"], 1e-6),
                            "-", color=SYS_COLOR[sysname], lw=1.0, alpha=0.8)
        ax.axhline(FLAG, color="0.5", ls="--", lw=0.8)
        ax.set(xlabel=r"scaling $\lambda$", ylabel=r"rel. dev. vs $\lambda=1$",
               title=_GLABEL[g])
    handles = [plt.Line2D([], [], color=SYS_COLOR[s], label=s) for s in SYSTEMS]
    axes.ravel()[0].legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle(f"(3) Scale invariance, h={H_INV} bohr; dashed = 10% flag")
    fig.tight_layout()
    fig.savefig(_FIG / "invariance_scale.pdf")
    plt.close(fig)


_GLABEL_SUM = {"s": r"$s$", "q": r"$q$", "power_spectrum": r"$\tilde P$",
               "bispectrum": "bispec."}
_GCOLOR_SUM = {"s": "tab:blue", "q": "tab:red", "power_spectrum": "tab:green",
               "bispectrum": "tab:purple"}


def _entry_rel_devs(samples, ref, group):
    """Pooled per-entry relative deviations of each sample vs ref, keeping only
    above-floor reference entries (the floor makes a large relative error on a
    near-zero feature non-meaningful)."""
    ref = np.atleast_1d(np.asarray(ref, float))
    mag = np.abs(ref)
    floor = max(FLOOR[group], 1e-3 * float(mag.max()) if mag.size else 0.0)
    keep = mag > floor
    if not np.any(keep):
        return np.array([])
    out = [np.abs(np.atleast_1d(s) - ref)[keep] / mag[keep] for s in samples]
    return np.concatenate(out) if out else np.array([])


def summary_distributions(h=H_INV, n_rot=N_ROT):
    """Distributions behind the condensed main-text figure. Two aspects are kept
    separate because they answer different questions:

      * INVARIANCE -- how much a feature *changes* under a symmetry operation it should
        respect (rotation, sub-grid translation, uniform scaling within the theoretical
        range Lambda<=Lambda_max=n_in/n_out, grid refinement vs the continuum). q is
        invariant here (a rotation/translation/grid-consistent scalar), so it looks fine.

      * RECONSTRUCTION ACCURACY -- how close the *reconstructed value* is to the exact
        gradient/Laplacian (finite-difference ground truth). This is where the reduced
        Laplacian q fails catastrophically (tens-to-hundreds x in dense environments),
        while the reduced gradient s stays accurate.

    Returns (inv, recon, meta): inv[test][group] and recon[group] are pooled arrays of
    relative deviations across the test systems."""
    n_in = NIN
    lam_max = n_in / N_OUT
    lam_hi = 0.8 * lam_max                                   # stay safely within Lambda_max
    lams = [l for l in (0.65, 0.8, 0.9, 1.1, 1.25, 1.5) if (1.0 / lam_hi) <= l <= lam_hi]
    rng = np.random.default_rng(_SEED)
    rots = [random_rotation_matrix(rng) for _ in range(n_rot)]
    st = LatticeStencil(h, n_in)
    shift_st = [LatticeStencil(h, n_in, frac=tuple((f * _SHIFT_DIR) % 1.0)) for f in SHIFTS]

    inv = {t: {g: [] for g in GROUPS} for t in ("rotation", "translation", "scale", "grid")}
    recon = {"s": [], "q": []}
    print(f"  summary: h={h}, n_in={n_in}, Lambda_max={lam_max:.1f}, "
          f"scale lambda in [{min(lams):.2f},{max(lams):.2f}] (<= Lambda_max)")
    for sysname, cname, builder in _iter_cases():
        cl = Cluster(builder())
        rot_feats = [grid_features(cl.rotated(R), st) for R in rots]
        ref_grid = grid_features(cl, st)
        trans_feats = [grid_features(cl, sst) for sst in shift_st]
        scale_feats = [grid_features(cl.scaled(lam), st) for lam in lams]
        cont = continuum_features(cl, n_in)
        for g in GROUPS:
            mean = np.mean([f[g] for f in rot_feats], axis=0)
            inv["rotation"][g].append(_entry_rel_devs([f[g] for f in rot_feats], mean, g))
            inv["translation"][g].append(_entry_rel_devs([f[g] for f in trans_feats], ref_grid[g], g))
            inv["scale"][g].append(_entry_rel_devs([f[g] for f in scale_feats], ref_grid[g], g))
            inv["grid"][g].append(_entry_rel_devs([ref_grid[g]], cont[g], g))
        # reconstruction accuracy vs the exact gradient/Laplacian (finite differences)
        fd_s, fd_q = fd_s_q(cl)
        if abs(fd_s) > 1e-2:                                 # skip symmetry centers where grad rho=0
            recon["s"].append(abs(ref_grid["s"][0] - fd_s) / abs(fd_s))
        if abs(fd_q) > 1e-2:
            recon["q"].append(abs(ref_grid["q"][0] - fd_q) / abs(fd_q))
        print(f"    {sysname}/{cname} done")
    inv = {t: {g: (np.concatenate(inv[t][g]) if any(a.size for a in inv[t][g])
                   else np.array([1e-6])) for g in GROUPS} for t in inv}
    recon = {g: np.array(recon[g]) if recon[g] else np.array([1e-6]) for g in recon}
    return inv, recon, {"h": h, "n_in": n_in, "lambda_max": lam_max,
                        "lambda_range": [min(lams), max(lams)], "n_rot": n_rot}


def _box(ax, data_by_group, groups, label):
    """Grouped box-and-whisker (5-95 percentile whiskers) of relative deviations."""
    w = 0.8 / len(groups)
    x = np.arange(len(label))
    for i, g in enumerate(groups):
        data = [np.maximum(np.asarray(data_by_group[k][g], float), 1e-6) for k in range(len(label))]
        bp = ax.boxplot(data, positions=x + (i - (len(groups) - 1) / 2) * w, widths=w * 0.85,
                        patch_artist=True, whis=(5, 95), showfliers=True,
                        flierprops=dict(marker=".", ms=2, mfc=_GCOLOR_SUM[g],
                                        mec=_GCOLOR_SUM[g], alpha=0.4),
                        medianprops=dict(color="k", lw=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=_GCOLOR_SUM[g], alpha=0.65)
    ax.axhline(FLAG, color="0.4", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(label)


def plot_invariance_summary(inv, recon, meta):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   gridspec_kw={"width_ratios": [3, 1]})
    tests = ["rotation", "translation", "scale", "grid"]
    tlab = ["rotation", "translation",
            "scaling\n" + r"($\lambda\leq%.1f$)" % meta["lambda_range"][1],
            r"grid $h{=}0.2$"]
    _box(axa, [{g: inv[t][g] for g in GROUPS} for t in tests], GROUPS, tlab)
    axa.text(3.5, FLAG * 1.25, "10% flag", fontsize=7, color="0.4", ha="right")
    axa.set(ylabel="relative deviation (over systems)",
            title=r"(a) invariance: rotation / translation / scaling ($\leq\Lambda_{\max}$) / grid")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_GCOLOR_SUM[g], alpha=0.65) for g in GROUPS]
    axa.legend(handles, [_GLABEL_SUM[g] for g in GROUPS], fontsize=8, ncol=4, loc="upper left")

    # panel (b): reconstruction accuracy vs the exact gradient/Laplacian
    _box(axb, [{"s": recon["s"], "q": recon["q"]}], ["s", "q"], ["recon."])
    axb.set(title="(b) reconstruction\nvs exact",
            ylabel=r"$|f_{\rm SIMPLE}-f_{\rm exact}|/|f_{\rm exact}|$")
    axb.legend([plt.Rectangle((0, 0), 1, 1, fc=_GCOLOR_SUM[g], alpha=0.65) for g in ("s", "q")],
               [r"$s$ (gradient)", r"$q$ (Laplacian)"], fontsize=8, loc="upper left")
    fig.suptitle(r"SIMPLE invariants at production settings: "
                 r"$s$, $\tilde P$, bispectrum usable; $q$ not reconstructable",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_FIG / "invariance_summary.pdf")
    plt.close(fig)


def make_summary_figure():
    """Recompute the distributions (not stored in JSON) and draw the summary figure."""
    inv, recon, meta = summary_distributions()
    plot_invariance_summary(inv, recon, meta)
    print(f"wrote invariance_summary.pdf (Lambda_max={meta['lambda_max']:.1f})")


def regen_from_json():
    """Regenerate the stress-test figures (incl. the condensed summary) from cached JSON."""
    r = json.loads(_OUT.read_text())
    plot_grid(r["grid_sensitivity"])
    plot_bars(r["rotation"], "invariance_rotation.pdf",
              f"(2) Rotational invariance: spread over rotations, h={H_INV} bohr; dashed = 10% flag")
    plot_bars(r["translation"], "invariance_translation.pdf",
              f"(1) Translation: max dev. over sub-grid shifts, h={H_INV} bohr; dashed = 10% flag")
    plot_scale(r["scale"])
    plot_scale_resolution(r["scale_resolution"])
    make_summary_figure()   # recomputes the distributions (not stored in JSON)
    print(f"regenerated invariance_* (incl. invariance_summary) from {_OUT.name}")


# =============================================================================
# Main
# =============================================================================
def main():
    _FIG.mkdir(exist_ok=True)
    _DATA.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"Invariant stress-test  (R_c={R_CUT} bohr, l_max={L_MAX}, n_out={N_OUT})")
    grid = test_grid_sensitivity()
    rot = test_rotation()
    trans = test_translation()
    scale = test_scale()
    scale_res = test_scale_resolution()

    results = {"meta": {"R_c": R_CUT, "l_max": L_MAX, "n_out": N_OUT,
                        "spacings": list(SPACINGS), "h_invariance": H_INV,
                        "n_rotations": N_ROT, "flag_threshold": FLAG, "floors": FLOOR},
               "grid_sensitivity": grid, "rotation": rot,
               "translation": trans, "scale": scale,
               "scale_resolution": scale_res}
    _OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {_OUT}")

    plot_grid(grid)
    plot_bars(rot, "invariance_rotation.pdf",
              f"(2) Rotational invariance: spread over {N_ROT} rotations, h={H_INV} bohr; "
              "dashed = 10% flag")
    plot_bars(trans, "invariance_translation.pdf",
              f"(1) Translation: max dev. over sub-grid shifts, h={H_INV} bohr; dashed = 10% flag")
    plot_scale(scale)
    plot_scale_resolution(scale_res)
    make_summary_figure()
    print(f"wrote invariance_*.pdf to {_FIG}/   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
