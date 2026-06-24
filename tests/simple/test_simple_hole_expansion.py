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
# PHASE E: parameter-free second-order gradient correction (GEA2)
# ======================================================================= #
def _build_gga(gauge_fix=True, n=600):
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION_GGA, SIMPLEHOLEEXPGGAParameters
    r = np.linspace(1e-3, 12.0, n)
    w = np.gradient(r)
    p = SIMPLEHOLEEXPGGAParameters(r_c=6.0, n_channels=16, gauge_fix=gauge_fix)
    return SIMPLE_HOLE_EXPANSION_GGA(r_quad=r, quadrature_weights=w, params=p), r, w


def test_E1_enhancement_is_gated_s2():
    """The enhancement is the two-term gated s^2 form with the LB94-style soft floor:
    F = enh_floor + softplus_k(1 + s^2 (m_g g_HEG + m_h g_H1s) - enh_floor). Validates the
    wiring (both terms carry s^2; the floor keeps F > 0)."""
    F, r, w = _build_gga(n=800)
    rho = np.exp(0.05 * (r - 8.0)) + 0.01
    C = np.array([op @ rho for op in F._ops])
    R_ad, _ = F._R_ad(rho)
    g = F._grad_op @ rho
    Fx = F._eps_full(C, rho, g) / F._eps_sf(C, R_ad)
    d_heg, d_h1s = F._gates(C, R_ad)
    gheg = np.exp(-F.params.alpha_heg * d_heg)
    gh1s = 1.0 - np.exp(-F.params.alpha_h1s * d_h1s)
    e = F._s2_bounded(rho, g) * (F.params.m_g * gheg + F.params.m_h * gh1s)
    k, fl = F.params.enh_floor_k, F.params.enh_floor
    expect = fl + np.logaddexp(0.0, k * (1.0 + e - fl)) / k
    assert np.allclose(Fx, expect, rtol=1e-9)
    assert np.all(Fx > 0.0)                                          # floor keeps F positive


def test_E_gates_detect_their_limits():
    """The HEG gate distance D_HEG is ~0 for a uniform density (its signature), and the H1s gate
    distance D_H1s is ~0 for a hydrogenic 1s (it sits on the manifold) -- each gate detects its
    own limit in the scale-free c_ad feature space."""
    F, r, w = _build_gga()
    rho_u = np.full_like(r, 1.0)
    Cu = np.array([op @ rho_u for op in F._ops]); Radu, _ = F._R_ad(rho_u)
    d_heg_u, _ = F._gates(Cu, Radu)
    rho_1s = (1.5 ** 3 / np.pi) * np.exp(-3.0 * r)
    C1 = np.array([op @ rho_1s for op in F._ops])
    Rad1, _ = F._R_ad(np.maximum(rho_1s, 1e-12))
    _, d_h1s_1 = F._gates(C1, Rad1)
    mid = slice(len(r) // 5, 4 * len(r) // 5)
    assert np.median(d_heg_u[mid]) < 1e-2, "uniform density not at the HEG signature"
    sig = rho_1s > 1e-3 * rho_1s.max()
    assert np.min(d_h1s_1[sig]) < 1e-2, "1s density not on the H1s manifold"


def test_E1_gradient_adjoint_matches_fd():
    """The exact (variational) adjoint matches FD of the energy. Uses frozen_potential=False."""
    from atom.xc.evaluator import DensityData
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION_GGA, SIMPLEHOLEEXPGGAParameters
    r = np.linspace(1e-3, 12.0, 600); w = np.gradient(r)
    p = SIMPLEHOLEEXPGGAParameters(r_c=6.0, n_channels=16, gauge_fix=False, frozen_potential=False)
    F = SIMPLE_HOLE_EXPANSION_GGA(r_quad=r, quadrature_weights=w, params=p)
    rho = 0.5 * np.exp(-0.5 * r ** 2) + 0.05 * np.exp(-0.15 * (r - 2.0) ** 2) + 0.01
    ew = F.energy_weights

    def Ex(rh):
        C = np.array([op @ rh for op in F._ops])
        g = F._grad_op @ rh
        return float(np.sum(ew * rh * F._eps_full(C, np.maximum(rh, 1e-12), g)))

    vx = F.compute_xc(DensityData(rho=rho)).v_x
    rng = np.random.default_rng(1)
    for j in rng.choice(np.arange(60, len(r) - 60), 8, replace=False):
        h = 1e-6
        rp = rho.copy(); rp[j] += h
        rm = rho.copy(); rm[j] -= h
        fd = (Ex(rp) - Ex(rm)) / (2.0 * h) / ew[j]
        assert abs(vx[j] - fd) / (abs(fd) + 1e-12) < 5e-6, f"r={r[j]:.2f}: {vx[j]:.6f} vs {fd:.6f}"


def test_E1_reduces_to_expansion_without_gradient():
    """Uniform density -> s=0 -> enhancement=1 -> GGA energy density == the gradient-free
    expansion (and the gate is irrelevant since s^2=0)."""
    from atom.xc.simple_hole_expansion import SIMPLE_HOLE_EXPANSION, SIMPLEHOLEEXPParameters
    F, r, w = _build_gga()
    base = SIMPLE_HOLE_EXPANSION(r_quad=r, quadrature_weights=w,
                                params=SIMPLEHOLEEXPParameters(r_c=6.0, n_channels=16))
    rho = np.full_like(r, 1.0)
    C = np.array([op @ rho for op in F._ops])
    g = F._grad_op @ rho
    eps_gga = F._eps_full(C, rho, g)
    eps_base = base._eps_from_coeffs(np.array([op @ rho for op in base._ops]), rho)
    # s=0 -> e=0 -> F=1 to softplus precision (~1/exp(k(1-floor))); LDA limit preserved
    assert np.allclose(eps_gga, eps_base, rtol=1e-6)


def test_E1_gga_scf_converges():
    """The two-term GGA converges self-consistently (PSP, the target regime) and, by the He
    cancellation of the two opposite-sign gated terms, stays very close to the base for He
    (the FA limit is preserved)."""
    from atom import AtomicDFTSolver
    res = {}
    for func in ("SIMPLE_HOLE_EXPANSION", "SIMPLE_HOLE_EXPANSION_GGA"):
        s = AtomicDFTSolver(atomic_number=2, xc_functional=func, all_electron_flag=False,
                            max_scf_iterations=300)
        r = s.solve()
        assert r["converged"], f"{func}: SCF did not converge"
        res[func] = float(r["energy_components"].exchange)
    assert np.isfinite(res["SIMPLE_HOLE_EXPANSION_GGA"])
    # He: the two terms cancel -> GGA within a few mHa of base (both ~exact for He)
    assert abs(res["SIMPLE_HOLE_EXPANSION_GGA"] - res["SIMPLE_HOLE_EXPANSION"]) < 0.02, \
        f"GGA {res['SIMPLE_HOLE_EXPANSION_GGA']:.4f} far from base {res['SIMPLE_HOLE_EXPANSION']:.4f}"


# ======================================================================= #
# PHASE F: learnable residual layer with exact limits by construction (mechanism)
# ======================================================================= #
def test_F_residual_vanishes_at_both_anchors():
    """For ARBITRARY fitted weights, the gated learned residual is exactly zero at the HEG
    (lambda=0) and one-electron (lambda=1) anchors, so it cannot break either exact limit."""
    n_ch, n_feat = 24, 5
    a = ex.charge_moments(n_ch, R_C)
    r0 = ex.radial_basis_at_origin(n_ch, R_C)
    rng = np.random.default_rng(3)
    for _ in range(20):
        W = rng.standard_normal((n_ch, n_feat))
        feats = rng.standard_normal(n_feat)
        for lam in (0.0, 1.0):
            d = ex.learnable_residual(feats, W, lam, a, r0, n_ch)
            assert np.allclose(d, 0.0, atol=1e-12), f"residual nonzero at lambda={lam}"


def test_F_residual_is_charge_and_ontop_neutral():
    """At any intermediate lambda the residual carries zero charge and zero on-top change,
    so the sum rule (-1) and on-top constraints remain exact for any weights."""
    n_ch, n_feat = 24, 5
    a = ex.charge_moments(n_ch, R_C)
    r0 = ex.radial_basis_at_origin(n_ch, R_C)
    rng = np.random.default_rng(4)
    for _ in range(20):
        W = rng.standard_normal((n_ch, n_feat))
        feats = rng.standard_normal(n_feat)
        d = ex.learnable_residual(feats, W, 0.5, a, r0, n_ch)
        assert abs(4.0 * np.pi * np.dot(a, d)) < 1e-10, "residual changes enclosed charge"
        assert abs(np.dot(r0, d)) < 1e-10, "residual changes on-top value"
        # but it CAN change the energy channel (otherwise it would be useless)
        assert abs(np.dot(ex.coulomb_moments(n_ch, R_C), d)) > 0 or np.allclose(d, 0)


# ======================================================================= #
# PHASE KERNEL: fixed-point hole map (LDA-from-GEA + FA), operator-free
# ======================================================================= #
_KCH, _KNU = 16, 1024


def test_KERNEL_T1_uniform_is_lda():
    """Uniform density -> s=0, Q/2>>2 (W_FA=0) -> the kernel map returns the HEG anchor, so
    eps_x = LDA within the finite-R_c band, with the sum rule and on-top exact."""
    a = ex.charge_moments(_KCH, R_C); r0 = ex.radial_basis_at_origin(_KCH, R_C)
    b = ex.coulomb_moments(_KCH, R_C)
    for rho in (0.25, 0.5, 1.0, 2.0, 5.0):
        c, d = ex.kernel_map_coeffs(_uniform(rho), 0.0, R_C, _KCH, nu=_KNU, return_diagnostics=True)
        assert d["W_FA"] == 0.0
        F = ex.eps_from_coeffs(c, b) / float(ex.lda_exchange_per_particle(rho))
        assert 0.98 < F < 1.04, f"rho={rho}: F={F:.4f} outside finite-R_c LDA band"
        assert abs(ex.enclosed_charge(c, a) + 1.0) < 1e-6
        assert abs(ex.on_top(c, r0) - (-0.5 * rho)) < 1e-6


@pytest.mark.parametrize("rho0", [0.5, 1.0, 2.0])
def test_KERNEL_T2_gea_slope(rho0):
    """A slowly-varying density (uniform base rho0 with reduced gradient s) gives the exact GEA2
    enhancement F_x -> 1 + (10/81)s^2: the slope vs s^2 is 10/81 and the linear-in-s coefficient
    is ~0 (parity: exchange is even in grad rho; the gradient enters only as s^2)."""
    b = ex.coulomb_moments(_KCH, R_C)
    eps_unif = float(ex.lda_exchange_per_particle(rho0))
    s = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    Fm1 = np.array([ex.eps_from_coeffs(ex.kernel_map_coeffs(_uniform(rho0), si, R_C, _KCH, nu=_KNU), b)
                    / eps_unif - 1.0 for si in s])
    # slope vs s^2 (with intercept absorbing the finite-R_c F(s=0) offset)
    slope = np.polyfit(s ** 2, Fm1, 1)[0]
    assert abs(slope - 10.0 / 81.0) < 0.02 * (10.0 / 81.0), f"GEA slope {slope:.5f} vs {10/81:.5f}"
    # parity: fit c0 + b1 s + b2 s^2; the linear coefficient must be negligible vs the quadratic
    c0, b1, b2 = np.polyfit(s, Fm1, 2)[::-1]
    assert abs(b1) < 0.05 * abs(b2), f"spurious linear-in-s term b1={b1:.2e} vs b2={b2:.2e}"


@pytest.mark.parametrize("Z", [1, 2, 3])
def test_KERNEL_T3_one_electron_is_fa(Z):
    """A hydrogenic 1s holds <= one electron per spin (Q/2<=1) -> W_FA=1 -> the kernel map is the
    Fermi-Amaldi density-following hole -C/Q, matching the existing map_coeffs FA path, and is
    INDEPENDENT of the reduced gradient s (the GEA term is gated off): LDA/GEA and FA decouple."""
    b = ex.coulomb_moments(_KCH, R_C)
    c0, d = ex.kernel_map_coeffs(_hydrogenic_1s(Z), 0.0, R_C, _KCH, nu=_KNU, return_diagnostics=True)
    assert d["W_FA"] == 1.0 and d["Q_spin"] <= 1.0
    e0 = ex.eps_from_coeffs(c0, b)
    # s-independence (GEA gated off)
    es = ex.eps_from_coeffs(ex.kernel_map_coeffs(_hydrogenic_1s(Z), 0.8, R_C, _KCH, nu=_KNU), b)
    assert abs(es - e0) < 1e-9, f"FA limit depends on s: {1e3*(es-e0):.3f} mHa"
    # matches the existing Fermi-Amaldi path
    em = ex.eps_from_coeffs(ex.map_coeffs(_hydrogenic_1s(Z), R_C, _KCH, nu=_KNU), b)
    assert abs(e0 - em) < 1e-9, f"kernel FA {e0:.4f} != map_coeffs FA {em:.4f}"


def test_KERNEL_T4_rbf_reproduces_fixed_points():
    """The RBF interpolant reproduces every fixed point exactly (rhotilde(x_k)=rhotilde_k), and
    adding a node leaves the others unchanged -- the basis for adding more exact limits. With no
    fixed points it returns the supplied default (the HEG anchor) -> N=1 is LDA everywhere."""
    rng = np.random.default_rng(0)
    nodes = [(np.array([0.0, 0.0]), rng.standard_normal(_KCH)),
             (np.array([1.0, 0.5]), rng.standard_normal(_KCH)),
             (np.array([0.3, 1.2]), rng.standard_normal(_KCH))]
    for xk, yk in nodes:
        got = ex.rbf_interpolant(xk, nodes, default=np.zeros(_KCH), ell=0.7)
        assert np.allclose(got, yk, atol=1e-6), "RBF does not reproduce a fixed point"
    # empty fixed-point set -> the default (HEG anchor stand-in)
    dflt = rng.standard_normal(_KCH)
    assert np.allclose(ex.rbf_interpolant(np.zeros(2), [], default=dflt), dflt)


# ======================================================================= #
# PHASE KERNEL-SCF: production scale-free kernel functional
# ======================================================================= #
def test_KERNEL_SCF_adjoint_matches_fd():
    """The exact variational adjoint v_x of the kernel functional matches FD of E_x (<5e-6)."""
    from atom.xc.evaluator import DensityData
    from atom.xc.simple_hole_expansion import (SIMPLE_HOLE_EXPANSION_KERNEL,
                                               SIMPLEHOLEEXPKERNELParameters)
    r = np.linspace(1e-3, 12.0, 500); w = np.gradient(r)
    F = SIMPLE_HOLE_EXPANSION_KERNEL(r_quad=r, quadrature_weights=w,
                                     params=SIMPLEHOLEEXPKERNELParameters(gauge_fix=False))
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
        assert abs(vx[j] - fd) / (abs(fd) + 1e-12) < 5e-6, f"r={r[j]:.2f}: {vx[j]:.6f} vs {fd:.6f}"


def test_KERNEL_SCF_converges_and_He_is_FA():
    """The kernel functional converges self-consistently (PSP) with the exact adjoint (no
    floor/freeze), and spin-paired He (one electron per spin) is near-exact Fermi-Amaldi."""
    from atom import AtomicDFTSolver
    hf = AtomicDFTSolver(atomic_number=2, xc_functional="HF", all_electron_flag=False,
                         max_scf_iterations=300).solve()
    ker = AtomicDFTSolver(atomic_number=2, xc_functional="SIMPLE_HOLE_EXPANSION_KERNEL",
                          all_electron_flag=False, max_scf_iterations=300).solve()
    assert hf["converged"] and ker["converged"]
    Ehf = float(hf["energy_components"].hf_exchange)
    Ek = float(ker["energy_components"].exchange)
    assert abs(Ek - Ehf) < 0.015, f"He kernel {Ek:.4f} not near-FA HF {Ehf:.4f}"
