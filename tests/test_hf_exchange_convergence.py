"""Convergence must include the exchange operator used by HF/PBE0 orbitals."""

from dataclasses import replace

import numpy as np
import pytest

from atom import AtomicDFTSolver


BASE = dict(
    atomic_number=1,
    all_electron_flag=True,
    domain_size=4.0,
    finite_element_number=2,
    polynomial_order=2,
    quadrature_point_number=7,
    mesh_type="uniform",
    scf_tolerance=1e-10,
    max_scf_iterations=120,
    use_pulay_mixing=False,
    use_preconditioner=False,
    verbose=False,
)


def _solver(functional="HF", **overrides):
    parameters = dict(BASE, xc_functional=functional)
    if functional in ("HF", "PBE0"):
        parameters.update(max_scf_iterations_outer=50, use_oep=False)
    parameters.update(overrides)
    return AtomicDFTSolver(**parameters)


@pytest.fixture(scope="module")
def hartree():
    return _solver("None").solve()


def _eigenpair_residual(solver, result):
    """Independently rebuild the full Hamiltonian at the returned state."""
    driver = solver.scf_driver
    builder = solver.hamiltonian_builder
    ops = builder.ops_builder
    rebuilt = driver._compute_hf_exchange_matrices_dict(result["orbitals"])
    builder.set_hf_exchange_matrices(rebuilt)
    density = result["density_data"]
    zero = np.zeros_like(density.rho)
    v_x, v_c = zero, zero
    if driver.xc_calculator is not None:
        xc = driver.xc_calculator.compute_xc(density)
        v_x, v_c = xc.v_x, xc.v_c
    H = builder.build_for_l_channel(
        0, solver.poisson_solver.solve_hartree(density.rho), v_x, v_c,
        driver.switches, symmetrize=True, exclude_boundary=True,
    )
    basis = ops.get_global_interpolation_matrix()[:, 1:-1]
    coefficients = np.linalg.lstsq(basis, result["orbitals"], rcond=None)[0]
    y = np.linalg.solve(ops.get_S_inv_sqrt(exclude_boundary=True), coefficients)
    return np.linalg.norm(H @ y - y * result["eigen_energies"])


def test_hartree_density_does_not_converge_before_using_exchange(hartree):
    solver = _solver()
    result = solver.solve(rho_initial=hartree["rho"], save_intermediate=True)

    assert result["converged"]
    assert len(result["intermediate_info"].outer_iterations) > 1
    assert np.linalg.norm(result["rho"] - hartree["rho"]) > 1e-3
    assert np.linalg.norm(solver.hamiltonian_builder.H_hf_exchange_dict[0]) > 0.1
    assert _eigenpair_residual(solver, result) < 1e-8


@pytest.mark.parametrize("fraction", [0.25, 0.6])
def test_pbe0_returned_state_solves_rebuilt_exchange_hamiltonian(fraction):
    solver = _solver("PBE0", hybrid_mixing_parameter=fraction)
    result = solver.solve()

    assert result["converged"]
    assert _eigenpair_residual(solver, result) < 1e-8


@pytest.mark.parametrize("budget", [1, 2])
def test_exhausted_exchange_budget_is_unconverged(hartree, budget):
    solver = _solver(max_scf_iterations_outer=budget)
    result = solver.solve(rho_initial=hartree["rho"], save_intermediate=True)

    assert not result["converged"]
    assert len(result["intermediate_info"].outer_iterations) == budget
    # The returned orbitals retain their consumed, rather than newly built,
    # exchange operator; the missing next solve must not be concealed.
    consumed = solver.hamiltonian_builder.H_hf_exchange_dict[0]
    rebuilt = solver.scf_driver._compute_hf_exchange_matrices_dict(result["orbitals"])[0]
    assert np.linalg.norm(rebuilt - consumed) > 1e-4


def test_consistent_orbital_warm_start_can_converge_on_first_pass():
    solver = _solver()
    initial = solver.solve()
    assert initial["converged"]
    result = solver.scf_driver.run(
        initial["rho"], solver._get_scf_settings("HF"),
        orbitals_initial=initial["orbitals"], save_intermediate=True,
    )

    assert result.converged
    assert len(result.intermediate_info.outer_iterations) == 1


@pytest.mark.parametrize("failure", ["changing_exchange", "nan_exchange", "inner_failure"])
def test_density_fixed_point_alone_cannot_converge(monkeypatch, hartree, failure):
    solver = _solver(max_scf_iterations_outer=4)
    driver = solver.scf_driver
    # A controlled inner solver isolates exchange consistency from changes in
    # diagonalization or mixing: every pass returns exactly the same density.
    reference = _solver("None").scf_driver.run(
        hartree["rho"], dict(inner_max_iter=120, outer_max_iter=1,
                            rho_tol=1e-10, verbose=False),
    )
    assert reference.converged
    monkeypatch.setattr(driver, "_inner_loop", lambda **kwargs: replace(
        reference, converged=(failure != "inner_failure")
    ))
    calls = 0

    def exchange(orbitals, l_values=None):
        nonlocal calls
        calls += 1
        value = calls if failure == "changing_exchange" else 1.0
        if failure == "nan_exchange":
            value = np.nan
        return {0: value * solver.ops_builder_standard.get_S()}

    monkeypatch.setattr(driver, "_compute_hf_exchange_matrices_dict", exchange)
    result = driver.run(
        hartree["rho"], solver._get_scf_settings("HF"),
        orbitals_initial=hartree["orbitals"], save_intermediate=True,
    )

    assert not result.converged
    assert len(result.intermediate_info.outer_iterations) == 4


def test_zero_weight_exchange_does_not_block_local_pbe_fixed_point():
    solver = _solver("PBE0", hybrid_mixing_parameter=0.0)
    initial = solver.solve()
    assert initial["converged"]
    result = solver.scf_driver.run(
        initial["rho"], solver._get_scf_settings("PBE0"),
        save_intermediate=True,
    )

    assert result.converged
    assert len(result.intermediate_info.outer_iterations) == 1


def test_exchange_requires_consecutive_successes(monkeypatch, hartree):
    solver = _solver(max_scf_iterations_outer=5)
    driver = solver.scf_driver
    reference = _solver("None").scf_driver.run(
        hartree["rho"], dict(inner_max_iter=120, outer_max_iter=1,
                            rho_tol=1e-10, verbose=False),
    )
    monkeypatch.setattr(driver, "_inner_loop", lambda **kwargs: reference)
    values = iter([1.0, 1.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr(
        driver, "_compute_hf_exchange_matrices_dict",
        lambda orbitals, l_values=None: {
            0: next(values) * solver.ops_builder_standard.get_S()
        },
    )
    settings = dict(solver._get_scf_settings("HF"), n_consecutive=2)
    result = driver.run(
        hartree["rho"], settings, orbitals_initial=hartree["orbitals"],
        save_intermediate=True,
    )

    assert result.converged
    # First success is followed by an exchange change. Both succeeding passes
    # must use the same operator before the combined fixed point is accepted.
    assert len(result.intermediate_info.outer_iterations) == 4


def test_exchange_change_in_virtual_direction_does_not_block_stationarity(monkeypatch, hartree):
    solver = _solver()
    driver = solver.scf_driver
    reference = _solver("None").scf_driver.run(
        hartree["rho"], dict(inner_max_iter=120, outer_max_iter=1,
                            rho_tol=1e-10, verbose=False),
    )
    monkeypatch.setattr(driver, "_inner_loop", lambda **kwargs: reference)
    ops = solver.ops_builder_standard
    inv = ops.get_S_inv_sqrt(exclude_boundary=True)
    basis = ops.get_global_interpolation_matrix()[:, 1:-1]
    c = np.linalg.lstsq(basis, reference.orbitals, rcond=None)[0]
    y = np.linalg.solve(inv, c)[:, 0]
    virtual = np.arange(1.0, len(y) + 1)
    virtual -= y * (y @ virtual) / (y @ y)
    virtual /= np.linalg.norm(virtual)
    vector = np.linalg.solve(inv, virtual)
    exchange = np.zeros_like(ops.get_S())
    exchange[1:-1, 1:-1] = np.outer(vector, vector)
    count = 0

    def change_virtual_exchange(orbitals, l_values=None):
        nonlocal count
        count += 1
        return {0: count * exchange}

    monkeypatch.setattr(driver, "_compute_hf_exchange_matrices_dict", change_virtual_exchange)
    result = driver.run(
        reference.density_data.rho, solver._get_scf_settings("HF"),
        orbitals_initial=reference.orbitals, save_intermediate=True,
    )

    assert result.converged
    assert len(result.intermediate_info.outer_iterations) == 1
