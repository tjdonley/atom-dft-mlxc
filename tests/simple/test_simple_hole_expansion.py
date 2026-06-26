"""Gate tests for the direct-expansion SIMPLE exchange hole.

Phase A (this file, first block): the representation primitives and the HEG -> LDA limit.
Later phases append the parameter-free map, the orbital-hole reference, and the production
functional gates.

Provenance for every numeric gate: R_c, n_channels, xi*, nu (quadrature). Production basis
settings are R_c = 6 bohr, n_out = 10, l_max <= 3 (CODEMAP).
"""
import os

import numpy as np
import pytest

from atom.xc import simple_hole_expansion_explicit as ex
from atom.xc import orbital_hole as oh

_DATA = os.path.join(os.path.dirname(__file__), "data")

R_C = 6.0          # bohr (production)
N_CHAN = 16        # monopole channels for the explicit reference
NU = 512           # quadrature nodes


# --------------------------------------------------------------------------- #
# A3: closed-form vs quadrature for the per-basis-function moments
# --------------------------------------------------------------------------- #
def test_A3_charge_moments_closed_vs_quad():
    a_closed = ex.charge_moments(N_CHAN, R_C)
    a_quad = ex.charge_moments_quad(N_CHAN, R_C, nu=NU)
    assert np.allclose(a_closed, a_quad, atol=1e-10, rtol=0), \
        f"max |a_closed - a_quad| = {np.max(np.abs(a_closed - a_quad)):.2e}"


def test_A3_coulomb_moments_closed_vs_quad():
    b_closed = ex.coulomb_moments(N_CHAN, R_C)
    b_quad = ex.coulomb_moments_quad(N_CHAN, R_C, nu=NU)
    assert np.allclose(b_closed, b_quad, atol=1e-10, rtol=0), \
        f"max |b_closed - b_quad| = {np.max(np.abs(b_closed - b_quad)):.2e}"


def test_A3_odd_coulomb_moments_vanish():
    b = ex.coulomb_moments(N_CHAN, R_C)
    assert np.allclose(b[1::2], 0.0, atol=1e-14)
    assert np.all(np.abs(b[0::2]) > 1e-8)


# --------------------------------------------------------------------------- #
# A1: HEG hole projected onto the basis reproduces LDA exchange
#
# Two distinct truncation sources, disentangled here:
#   (resolution) finite N: need N >~ k_F R_c / pi to resolve the hole's oscillations.
#       Once satisfied, the HEG->LDA ratio SATURATES (adding channels does nothing).
#   (tail) finite R_c, i.e. finite dimensionless window x_c = k_F R_c: the hole tail
#       spills past R_c, so the saturated ratio sits slightly below 1 and the sum rule
#       below -1. Shrinks as x_c grows. (Phase B's constraint projection restores -1 exactly.)
# --------------------------------------------------------------------------- #
def test_A1_heg_ratio_saturates_in_N():
    """At a well-resolved density the ratio saturates by N=16 and is within ~0.2% of LDA."""
    rho = 2.0  # k_F R_c/pi ~ 7.4, so N=16 channels resolve the hole
    b = ex.coulomb_moments(N_CHAN, R_C)
    ratio16 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024), b) \
        / float(ex.lda_exchange_per_particle(rho))
    b32 = ex.coulomb_moments(32, R_C)
    ratio32 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 32, nu=2048), b32) \
        / float(ex.lda_exchange_per_particle(rho))
    assert ratio16 == pytest.approx(ratio32, abs=1e-4), \
        f"not saturated: ratio(16)={ratio16:.5f} ratio(32)={ratio32:.5f}"
    assert 0.995 < ratio16 <= 1.0 + 1e-6, f"ratio={ratio16:.5f}"


def test_A1_resolution_floor():
    """Under-resolved (N below k_F R_c/pi) the ratio is poor; resolving it fixes it."""
    rho = 2.0
    ratio8 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 8, nu=1024),
                                ex.coulomb_moments(8, R_C)) / float(ex.lda_exchange_per_particle(rho))
    ratio16 = ex.eps_from_coeffs(ex.project_hole(ex.heg_hole(rho), R_C, 16, nu=1024),
                                 ex.coulomb_moments(16, R_C)) / float(ex.lda_exchange_per_particle(rho))
    assert ratio8 < 0.95, f"expected poor resolution at N=8, got {ratio8:.5f}"
    assert ratio16 > 0.995 > ratio8


def test_A1_tail_capture_scale_free():
    """In the scale-free frame (fix x_c = k_F R_c, scale N with it), the ratio and sum rule
    approach 1 / -1 as the dimensionless window x_c grows."""
    rho = 2.0
    k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    last_ratio, last_q = None, None
    for xc in (20.0, 60.0):
        rc = xc / k_f
        n = int(2 * xc / np.pi) + 4
        c = ex.project_hole(ex.heg_hole(rho), rc, n, nu=2048)
        ratio = ex.eps_from_coeffs(c, ex.coulomb_moments(n, rc)) / float(ex.lda_exchange_per_particle(rho))
        q = ex.enclosed_charge(c, ex.charge_moments(n, rc))
        if last_ratio is not None:
            assert ratio > last_ratio and ratio > 0.999, f"x_c={xc}: ratio={ratio:.5f}"
            assert abs(q + 1.0) < abs(last_q + 1.0), f"x_c={xc}: sum rule {q:.4f} not closer to -1"
        last_ratio, last_q = ratio, q


# --------------------------------------------------------------------------- #
# A2: on-top is reconstructed exactly; sum rule carries the (documented) R_c tail deficit
# --------------------------------------------------------------------------- #
def test_A2_on_top_exact():
    rho = 2.0
    r0 = ex.radial_basis_at_origin(N_CHAN, R_C)
    coeffs = ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024)
    ot = ex.on_top(coeffs, r0)
    assert ot == pytest.approx(-0.5 * rho, rel=1e-3), f"on-top = {ot:.5f} vs -rho/2 = {-0.5*rho:.5f}"


def test_A2_sum_rule_tail_deficit():
    """At R_c=6 the projected HEG hole encloses ~-0.96 (4% tail spills past R_c). This is the
    modeling deficit the Phase-B constraint projection enforces back to exactly -1."""
    rho = 2.0
    a = ex.charge_moments(N_CHAN, R_C)
    coeffs = ex.project_hole(ex.heg_hole(rho), R_C, N_CHAN, nu=1024)
    q = ex.enclosed_charge(coeffs, a)
    assert -1.0 < q < -0.90, f"sum rule {q:.4f} outside expected tail band"


# ======================================================================= #
# PHASE B: parameter-free map (anchors + enclosed-charge switch + constraints)
# ======================================================================= #
def _uniform(rho):
    return lambda u: np.full_like(np.atleast_1d(u), rho, dtype=float)


def _hydrogenic_1s(Z):
    return lambda u: (Z ** 3 / np.pi) * np.exp(-2.0 * Z * np.atleast_1d(u))


# --- B1: HEG limit (lambda=0) -> LDA, with both constraints exact ------------ #
@pytest.mark.parametrize("rho", [0.5, 2.0, 5.0])
def test_B1_heg_limit_reproduces_lda(rho):
    coeffs, diag = ex.map_coeffs(_uniform(rho), R_C, N_CHAN, nu=1024, return_diagnostics=True)
    assert diag["lambda"] == pytest.approx(0.0), f"expected HEG (lambda=0), got {diag['lambda']}"
    eps = ex.eps_from_coeffs(coeffs, ex.coulomb_moments(N_CHAN, R_C))
    ratio = eps / float(ex.lda_exchange_per_particle(rho))
    # Within the finite-window band (constraints enforced exactly; deviation shrinks with R_c).
    assert ratio == pytest.approx(1.0, abs=0.025), f"rho={rho}: ratio={ratio:.5f}"


def test_B1_constraints_exact_in_heg():
    rho = 2.0
    coeffs, diag = ex.map_coeffs(_uniform(rho), R_C, N_CHAN, nu=1024, return_diagnostics=True)
    a = ex.charge_moments(N_CHAN, R_C)
    r0 = ex.radial_basis_at_origin(N_CHAN, R_C)
    assert ex.enclosed_charge(coeffs, a) == pytest.approx(-1.0, abs=1e-6)
    assert ex.on_top(coeffs, r0) == pytest.approx(-0.5 * rho, abs=1e-6)  # W=1/2 in bulk


def test_B1_heg_deviation_shrinks_with_window():
    """The (small) HEG-limit deviation from LDA is a finite-R_c tail artifact: it decreases
    monotonically as the window grows."""
    rho = 0.5
    devs = []
    for rc in (6.0, 10.0, 14.0):
        n = int(2 * (3 * np.pi ** 2 * rho) ** (1 / 3) * rc / np.pi) + 6
        eps = ex.eps_x_map(_uniform(rho), rc, n, nu=2048)
        devs.append(abs(eps / float(ex.lda_exchange_per_particle(rho)) - 1.0))
    assert devs[0] > devs[1] > devs[2], f"deviation not shrinking with R_c: {devs}"


# --- B2: one-electron limit (lambda=1) -> self-interaction-free --------------- #
def test_B2_one_electron_is_sic():
    """H-like 1s at the center: window holds ~1 electron -> lambda=1 -> hole = -density.
    eps_x(0) = -1/2 v_H(0) = -0.5 (Z=1), the exact self-interaction correction."""
    coeffs, diag = ex.map_coeffs(_hydrogenic_1s(1.0), R_C, N_CHAN, nu=1024, return_diagnostics=True)
    assert diag["lambda"] == pytest.approx(1.0, abs=1e-3), f"expected 1e (lambda=1), got {diag['lambda']}"
    eps = ex.eps_from_coeffs(coeffs, ex.coulomb_moments(N_CHAN, R_C))
    assert eps == pytest.approx(-0.5, abs=0.03), f"eps_x(0)={eps:.5f} (SIC -0.5)"


def test_B2_one_electron_constraints():
    coeffs, diag = ex.map_coeffs(_hydrogenic_1s(1.0), R_C, N_CHAN, nu=1024, return_diagnostics=True)
    a = ex.charge_moments(N_CHAN, R_C)
    r0 = ex.radial_basis_at_origin(N_CHAN, R_C)
    assert ex.enclosed_charge(coeffs, a) == pytest.approx(-1.0, abs=1e-6)
    # Fermi-Amaldi on-top -> -rho0/Q (= -rho0 for a fully-enclosed single electron, Q~1)
    assert ex.on_top(coeffs, r0) == pytest.approx(-diag["rho0"] / diag["Q_window"], abs=1e-6)


def test_B2_projection_barely_perturbs_fa_anchor():
    """The constraint projection leaves the Fermi-Amaldi anchor (-C/Q) nearly fixed: it already
    satisfies the sum rule (int = -1) and nearly the on-top, so the correction is tiny."""
    a = ex.charge_moments(N_CHAN, R_C)
    r0 = ex.radial_basis_at_origin(N_CHAN, R_C)
    C = ex.density_coeffs(_hydrogenic_1s(1.0), R_C, N_CHAN, nu=1024)
    Q = 4.0 * np.pi * float(np.dot(C, a))
    rho0 = 1.0 ** 3 / np.pi
    fa = -C / Q
    fixed = ex.constraint_project(fa, a, r0, sum_target=-1.0, ontop_target=-rho0 / Q)
    assert np.linalg.norm(fixed - fa) < 1e-2, f"shift {np.linalg.norm(fixed - fa):.2e}"


# --- B3: the enclosed-charge switch is smooth and monotone -------------------- #
def test_B3_switch_smooth_monotone():
    q = np.linspace(0.0, 3.0, 301)
    lam = ex.enclosed_charge_switch(q)
    assert lam[0] == pytest.approx(1.0) and lam[-1] == pytest.approx(0.0)
    assert np.all(np.diff(lam) <= 1e-12), "switch not monotone non-increasing"
    # C^2 (quintic smoothstep): first and second differences are continuous (no spikes)
    d2 = np.diff(lam, 2)
    assert np.max(np.abs(d2)) < 0.01, f"second difference spike {np.max(np.abs(d2)):.3f}"


# ======================================================================= #
# PHASE C: exact orbital-based exchange-hole reference (s-only: He, Be)
#
# Reference data generated by reports/hole_expansion/gen_orbital_hole_refs.py
# (all-electron EXX SCF + orbital-hole reconstruction). Tests load it; no SCF here.
# ======================================================================= #
def _load(name):
    path = os.path.join(_DATA, f"orbital_hole_{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}; run reports/hole_expansion/gen_orbital_hole_refs.py")
    return np.load(path)


# --- C1: the orbital-hole exchange energy matches the solver's exact exchange ---- #
@pytest.mark.parametrize("name", ["He", "Be"])
def test_C1_orbital_hole_reproduces_exact_Ex(name):
    d = _load(name)
    # headline: integrated orbital-hole E_x == solver oep_exchange to < 1 mHa
    assert abs(float(d["Ex_hole"]) - float(d["oep_exchange"])) < 1e-3, \
        f"{name}: E_x(hole)={float(d['Ex_hole']):.6f} vs oep={float(d['oep_exchange']):.6f}"


@pytest.mark.parametrize("name", ["He", "Be"])
def test_C1_eps_x_reproducible_from_saved_orbitals(name):
    """Recompute eps_x(r0) from the saved orbitals and confirm it matches the stored values
    (independent of the saved energy, guards the hole construction)."""
    d = _load(name)
    r_sorted, g_sorted, occ = d["r_sorted"], d["g_sorted"], d["occ"]
    r0_grid, eps_exact = d["r0_grid"], d["eps_exact"]
    idx = np.linspace(0, len(r0_grid) - 1, 6).astype(int)  # spot-check 6 points (speed)
    for i in idx:
        eps = oh.exact_eps_x(r0_grid[i], r_sorted, g_sorted, occ, n_u=160, n_mu=80)
        assert eps == pytest.approx(eps_exact[i], rel=1e-3), f"{name} r0={r0_grid[i]:.3f}"


# --- C1b: the SIMPLE projection of the exact hole reproduces eps_x (representation) - #
@pytest.mark.parametrize("name", ["He", "Be"])
def test_C1b_projection_reproduces_eps_x(name):
    """The windowed direct expansion reproduces eps_x where the exact hole is localized
    (core/valence, significant rho). It degrades in the diffuse density tail, where the exact
    exchange hole is intrinsically long-ranged (sits back in the bulk, far from r0) and a
    window around r0 cannot represent it. Since the tail carries negligible rho, the integrated
    E_x is unaffected (test_C1). Metric: the energy-weighted (rho r0^2) relative error in eps_x."""
    d = _load(name)
    b = ex.coulomb_moments(int(d["n_channels"]), float(d["r_c"]))
    eps_proj = 0.5 * 4.0 * np.pi * (d["rhotilde_exact"] @ b)
    r0, we = d["r0_grid"], d["rho_r0"] * 4.0 * np.pi * d["r0_grid"] ** 2
    # the physical metric: error in the exchange-energy contribution rho*eps_x (not per-point
    # eps_x, which is dominated by the diffuse tail where the exact hole is long-ranged but
    # rho is negligible). Reproduced to <0.5% (He) / ~2% (Be core, fixed-R_c resolution).
    e_proj = float(np.trapezoid(we * eps_proj, r0))
    e_exact = float(np.trapezoid(we * d["eps_exact"], r0))
    assert abs(1.0 - e_proj / e_exact) < 0.03, f"{name}: partial-E ratio {e_proj/e_exact:.4f}"


# --- C2: numerical angular average matches the closed-form hydrogenic 1s ---------- #
@pytest.mark.parametrize("Z", [1.0, 2.0])
def test_C2_angular_average_vs_analytic(Z):
    # build a dense radial grid and the hydrogenic density, compare <rho(r0+u)>_Omega
    r = np.linspace(1e-4, 20.0, 4000)
    rho = oh.hydrogenic_1s_density(Z)(r)
    for r0 in (0.2, 0.8, 2.0):
        u = np.linspace(0.05, 6.0, 30)
        num = oh.spherical_avg_radial(r, rho, r0, u)
        ana = oh.spherical_avg_hydrogenic_1s(Z, r0, u)
        assert np.allclose(num, ana, rtol=2e-3, atol=1e-6), \
            f"Z={Z} r0={r0}: max rel {np.max(np.abs(num/ana-1)):.4f}"


# ======================================================================= #
# PHASE D: production functional SIMPLE_HOLE_EXPANSION (adjoint, limits, SCF)
# ======================================================================= #
def _build_functional(gauge_fix=True, n=500):
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION, SIMPLEHOLEEXPParameters
    r = np.linspace(1e-3, 12.0, n)
    w = np.gradient(r)
    p = SIMPLEHOLEEXPParameters(r_c=6.0, n_channels=16, gauge_fix=gauge_fix)
    return SIMPLE_HOLE_EXPANSION(r_quad=r, quadrature_weights=w, params=p), r, w


# --- D1: discrete-adjoint potential == FD of the energy --------------------------- #
def test_D1_adjoint_matches_finite_difference():
    from atom.xc.evaluator import DensityData
    F, r, w = _build_functional(gauge_fix=False)
    rho = 0.5 * np.exp(-0.5 * r ** 2) + 0.02 * np.exp(-0.1 * r ** 2)
    ew = F.energy_weights

    def Ex(rh):
        C = np.array([op @ rh for op in F._ops])
        return float(np.sum(ew * rh * F._eps_from_coeffs(C, np.maximum(rh, 1e-12))))

    vx = F.compute_xc(DensityData(rho=rho)).v_x
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(50, len(r) - 50), 8, replace=False)
    for j in idx:
        h = 1e-6
        rp = rho.copy(); rp[j] += h
        rm = rho.copy(); rm[j] -= h
        fd = (Ex(rp) - Ex(rm)) / (2.0 * h) / ew[j]
        assert abs(vx[j] - fd) / (abs(fd) + 1e-12) < 5e-6, \
            f"r={r[j]:.2f}: v_x={vx[j]:.6f} fd={fd:.6f}"


# --- D2: HEG limit ~ LDA (scale invariance approximate at finite R_c) --------------- #
def test_D2_heg_limit_near_lda():
    """The scale-free (Q_S=2) hole recovers LDA in the HEG limit. Scale invariance is only
    APPROXIMATE at finite R_c (the n_in/n_out Bessel bases reconstruct the sub-window profile to
    finite resolution -- the SF-writeup 'breakdown'): the ratio drifts with density. At the new
    default R_C = 6 bohr: ~1.01 at rho=0.5, 0.986 at rho=1, 0.946 at rho=2, ~0.86 at rho=5 --
    within ~6% of LDA across the valence range and best near rho~1."""
    F, r, w = _build_functional()
    for rho_val, lo in ((1.0, 0.95), (2.0, 0.93)):         # valence-relevant; R_C=6 drift
        rho = np.full_like(r, rho_val)
        C = np.array([op @ rho for op in F._ops])
        ratio = F._eps_from_coeffs(C, rho)[len(r) // 2] / float(ex.lda_exchange_per_particle(rho_val))
        assert lo < ratio < 1.05, f"rho={rho_val}: HEG ratio {ratio:.4f} outside [{lo}, 1.05]"


# --- D3: scale-free SCF atoms (Q_S=2 normalization + contraction) -> near-exact ----- #
# With the prior-SF normalization (sum rule via the on-top scale Q_S=2, c_ad used only through
# the damped contractions g.c_ad/h.c_ad) and r_c = pipeline R_C, the closed-shell atoms are
# reproduced to ~mHa: He -1.027 (vs -1.026), Be -2.71 (1.8% over -2.666). H is near-SIC
# (test_D3_hydrogen...). The min-norm projection and raw-c_ad bugs are gone.
@pytest.mark.parametrize("Z,name,exact_ex,band", [
    (2, "He", -1.0258, (-1.045, -1.010)),   # near-exact (~1.6 mHa)
    (4, "Be", -2.6658, (-2.760, -2.640)),   # ~1.8% over (two electrons per spin)
])
def test_D3_scf_atoms(Z, name, exact_ex, band):
    from atom import AtomicDFTSolver
    s = AtomicDFTSolver(atomic_number=Z, xc_functional="SIMPLE_HOLE_EXPANSION",
                        all_electron_flag=True, domain_size=15.0, max_scf_iterations=250)
    res = s.solve()
    assert res["converged"], f"{name}: SCF did not converge"
    e_x = float(res["energy_components"].exchange)
    assert band[0] <= e_x <= band[1], \
        f"{name}: E_x={e_x:.4f} outside expected band {band} (exact {exact_ex})"


def test_D3_hydrogen_near_self_interaction_free():
    """H (1 electron) is the cleanest SIC test: exact exchange cancels the self-Hartree
    (E_x = -E_H). With the per-spin Fermi-Amaldi limit, the residual E_x + E_H is small."""
    from atom import AtomicDFTSolver
    res = AtomicDFTSolver(atomic_number=1, xc_functional="SIMPLE_HOLE_EXPANSION",
                          all_electron_flag=True, domain_size=15.0, max_scf_iterations=250).solve()
    assert res["converged"], "H: SCF did not converge"
    ec = res["energy_components"]
    # with the Q_S=2 normalization + contraction, H is self-interaction-free to ~2 mHa
    assert abs(ec.exchange + ec.hartree) < 0.01, \
        f"H not near-SIC: E_x+E_H = {ec.exchange + ec.hartree:.4f} (E_x={ec.exchange:.4f}, E_H={ec.hartree:.4f})"


# ======================================================================= #
# PHASE FP: kernel-mapped fixed-point hole SIMPLE_HOLE_KERNEL_FP
# The hole COEFFICIENTS are interpolated over fixed points by a kernel whose
# per-l SIMPLE distances are the kernel coordinates (l=1 == s^2); the LDA limit
# is enforced by anchoring the map at the HEG node, and the GEA2 slope by the
# amplitude of the single l=1 node -- no explicit gradient term, no enhancement
# factor. The energy is the direct hole integral eps_x = 2 pi R_ad^2 (rho~ . beta).
# ======================================================================= #
def _build_fp(n=600):
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP
    r = np.linspace(1e-3, 14.0, n)
    w = np.gradient(r)
    return SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w), r, w


def test_FP_uniform_is_lda():
    """Uniform density sits at the HEG node, where the kernel pins the shape to the
    moment-matched LDA hole: F_x = eps_x / eps_x^unif = 1 to machine precision."""
    from atom.xc.simple_hole_expansion import _C_LDA
    F, r, w = _build_fp()
    mid = len(r) // 2
    for rho0 in (0.25, 0.5, 1.0, 2.0, 5.0):
        rho = np.full_like(r, rho0)
        C = np.array([op @ rho for op in F._ops]); g = F._grad_op @ rho
        Fx = F._kernel_eps(C, rho, g)[mid] / (_C_LDA * rho0 ** (1.0 / 3.0))
        assert abs(Fx - 1.0) < 1e-5, f"rho={rho0}: F={Fx:.6f} != 1 (LDA limit broken)"


@pytest.mark.parametrize("rho0", [0.5, 1.0, 2.0])
def test_FP_gea_slope_from_l1_kernel(rho0):
    """The l=1 kernel node's amplitude is fixed so the small-gradient enhancement
    reproduces the exact second-order gradient expansion F_x -> 1 + (10/81) s^2 --
    purely from kernel scaling, with no explicit GEA term. The slope vs s^2 is 10/81
    and the linear-in-s coefficient is ~0 (exchange is even in grad rho)."""
    from atom.xc.simple_hole_expansion import _C_LDA
    F, r, w = _build_fp(n=1200)
    mid = len(r) // 2
    eps_unif = _C_LDA * rho0 ** (1.0 / 3.0)
    avals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    Fm1, svals = [], []
    for a in avals:
        rho = rho0 * np.exp(a * (r - r[mid]))               # exponential ramp about mid
        C = np.array([op @ rho for op in F._ops]); g = F._grad_op @ rho
        kF = (3.0 * np.pi ** 2 * rho[mid]) ** (1.0 / 3.0)
        svals.append(abs(g[mid]) / (2.0 * kF * rho[mid]))
        Fm1.append(F._kernel_eps(C, rho, g)[mid] / eps_unif - 1.0)
    s = np.array(svals); Fm1 = np.array(Fm1)
    slope = np.polyfit(s ** 2, Fm1, 1)[0]
    assert abs(slope - 10.0 / 81.0) < 0.05 * (10.0 / 81.0), \
        f"GEA slope {slope:.5f} vs {10/81:.5f}"
    c0, b1, b2 = np.polyfit(s, Fm1, 2)[::-1]
    assert abs(b1) < 0.1 * abs(b2), f"spurious linear-in-s term b1={b1:.2e} vs b2={b2:.2e}"


def test_FP_l1_tunable_no_lo_cap():
    """The l=1 RBF width is a TUNABLE parameter (Lieb-Oxford cap dropped): it is no longer
    pinned, defaults to params.fp_l1=0.5, and is settable. The 10/81 GEA slope is still enforced
    by c_G regardless of the width, so the gradient limit holds at any l1."""
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters, _GEA2
    F, r, w = _build_fp()
    assert F._fp_l1 == pytest.approx(0.5), f"default l1 {F._fp_l1} != 0.5 (still LO-calibrated?)"
    s = np.linspace(0.0, 30.0, 6000)
    assert F._fx_gea_axis(s).max() < 1.5, "peak should be a modest bump, not the LO-wide ramp"
    # width is settable via params, and the 10/81 slope holds at the new width
    F2 = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w,
                              params=SIMPLEHOLEKERNELFPParameters(fp_l1=1.5))
    assert F2._fp_l1 == pytest.approx(1.5)
    sm = (s > 1e-6) & (s < 0.2)
    slope = np.polyfit(s[sm] ** 2, F2._fx_gea_axis(s)[sm] - 1.0, 1)[0]
    assert slope == pytest.approx(_GEA2, rel=2e-2), f"slope {slope:.5f} != 10/81 at l1=1.5"


def test_FP_adjoint_matches_fd():
    """The discrete adjoint v_x (compute_xc) matches FD of the direct hole-integral
    energy through the C / rho / gradient channels (gauge_fix off to drop the
    constant gauge offset)."""
    from atom.xc.evaluator import DensityData
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters
    r = np.linspace(1e-3, 14.0, 500); w = np.gradient(r)
    F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w,
                              params=SIMPLEHOLEKERNELFPParameters(gauge_fix=False))
    ew = F.energy_weights
    rho = 0.5 * np.exp(-0.5 * r ** 2) + 0.05 * np.exp(-0.15 * (r - 2.0) ** 2) + 1e-3

    def Ex(rh):
        C = np.array([op @ rh for op in F._ops]); g = F._grad_op @ rh
        return float(np.sum(ew * rh * F._kernel_eps(C, np.maximum(rh, 1e-12), g)))

    vx = F.compute_xc(DensityData(rho=rho)).v_x
    rng = np.random.default_rng(1)
    for j in rng.choice(np.arange(60, len(r) - 60), 6, replace=False):
        h = 1e-6
        rp = rho.copy(); rp[j] += h; rm = rho.copy(); rm[j] -= h
        fd = (Ex(rp) - Ex(rm)) / (2.0 * h) / ew[j]
        assert abs(vx[j] - fd) / (abs(fd) + 1e-8) < 5e-4, f"r={r[j]:.2f}: {vx[j]:.6f} vs {fd:.6f}"


def test_FP_scf_converges_and_He_is_FA():
    """Reference-free, the functional reduces to LDA + Fermi-Amaldi: SCF converges,
    and spin-paired He (one electron per spin) is near-exact Fermi-Amaldi."""
    from atom import AtomicDFTSolver
    s = AtomicDFTSolver(atomic_number=2, xc_functional="SIMPLE_HOLE_KERNEL_FP",
                        all_electron_flag=False, max_scf_iterations=300)
    res = s.solve()
    assert res["converged"], "He: SCF did not converge"
    ec = res["energy_components"]
    hf = AtomicDFTSolver(atomic_number=2, xc_functional="HF", all_electron_flag=False,
                         max_scf_iterations=300).solve()
    Ehf = float(hf["energy_components"].hf_exchange)
    assert abs(float(ec.exchange) - Ehf) < 0.015, \
        f"He {ec.exchange:.4f} not near-FA HF {Ehf:.4f}"


def test_FP_closed_shell_functional_saved():
    """Pins the saved best closed-shell-only functional: kernel_fp_refs_closed_n512.npz at
    l0=0.7, l1=0.5, loaded cleanly via the refs_path param (no globals; baseline stays
    reference-free). It carries 512 reference nodes + 2 backbone, and cuts the non-SCF in-domain
    exchange error vs reference-free (Ne). NOTE: SCF convergence is atom-dependent for the
    referenced functional (He converges; Ne does not at 512 nodes) -- benchmarking is non-SCF."""
    import os, sys
    import atom.xc.simple_hole_expansion as She
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_KERNEL_FP, SIMPLEHOLEKERNELFPParameters
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from cache.refs.loader import load_hf
    closed = os.path.join(os.path.dirname(She.__file__), "data", "kernel_fp_refs_closed_n512.npz")
    assert os.path.exists(closed), "run cache/refs/regen/build_closed_shell_functional.py"
    hf = load_hf(10); o = np.argsort(np.asarray(hf["r"]))
    r = np.asarray(hf["r"])[o]; rho = np.maximum(np.asarray(hf["rho"])[o], 1e-12); w = np.asarray(hf["w"])[o]
    Ehf = float(hf["Ehf"])

    def ex(refs):
        p = SIMPLEHOLEKERNELFPParameters(fp_l0=0.7, fp_l1=0.5, refs_path=refs)
        F = SIMPLE_HOLE_KERNEL_FP(r_quad=r, quadrature_weights=w, params=p)
        cp = np.array([op @ rho for op in F._ops]); g = F._grad_op @ rho
        return F, float(np.sum(F.energy_weights * rho * F._kernel_eps(cp, rho, g)))

    _, e_free = ex(None)
    Fref, e_ref = ex(closed)
    assert len(Fref._fp_Xnodes) == 514, f"expected 512 refs + 2 backbone, got {len(Fref._fp_Xnodes)}"
    assert abs(e_ref - Ehf) < abs(e_free - Ehf), "closed-shell refs should reduce Ne non-SCF error"
    assert abs(e_ref - Ehf) < 0.15, f"Ne non-SCF err {1e3 * (e_ref - Ehf):.0f} mHa too large"


