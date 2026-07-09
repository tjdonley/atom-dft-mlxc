"""Focused regressions for repository-audit findings."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.data.data_processing import DataProcessor
from atom.descriptors.multipole import compute_descriptors_from_radial
from atom.scf.density import DensityData
from atom.scf.mixer import Mixer
from atom.scf.poisson import PoissonSolver
from atom.scf.response import ResponseCalculator
from atom.utils.occupation_states import (
    OccupationInfo,
    get_fraction_occupation_states,
)
from atom.xc.evaluator import (
    XCPotentialData,
    _xc_potential_data_flatten,
    _xc_potential_data_unflatten,
)
from atom.xc.hf import HartreeFockExchange
from atom.xc.lda import LDA_PZ
from atom.xc.meta_scan import _get_rho_tau_and_sigma
from atom.xc.oep import OEPCalculator


SMALL_GRID = {
    "domain_size": 4.0,
    "finite_element_number": 2,
    "polynomial_order": 2,
    "quadrature_point_number": 7,
    "verbose": False,
}


def _small_solver(**kwargs) -> AtomicDFTSolver:
    parameters = {
        "atomic_number": 1,
        "all_electron_flag": True,
        "xc_functional": "None",
        **SMALL_GRID,
    }
    parameters.update(kwargs)
    return AtomicDFTSolver(**parameters)


def _synthetic_full_spectrum(occupation_info, ops_builder):
    n_interior = len(ops_builder.physical_nodes) - 2
    l_terms = list(map(int, occupation_info.l_values))
    for l_value in occupation_info.unique_l_values:
        occupied_count = occupation_info.n_states_for_l(int(l_value))
        l_terms.extend([int(l_value)] * (n_interior - occupied_count))

    l_terms = np.asarray(l_terms, dtype=int)
    eigenvalues = np.linspace(-2.0, 3.0, len(l_terms))
    rng = np.random.default_rng(104729)
    orbitals = rng.normal(
        size=(len(ops_builder.quadrature_nodes), len(l_terms))
    )
    return eigenvalues, orbitals, l_terms


def test_nonlocal_pseudopotential_caches_virtual_only_channels():
    solver = _small_solver(
        atomic_number=8,
        all_electron_flag=False,
    )

    assert set(solver.hamiltonian_builder.H_nonlocal) == {0, 1, 2}
    assert 2 not in solver.occupation_info.unique_l_values


def test_valence_states_use_channel_ordinals_instead_of_principal_n():
    solver = _small_solver(
        atomic_number=26,
        all_electron_flag=False,
    )
    info = solver.occupation_info

    np.testing.assert_array_equal(info.l_values, np.array([0, 1, 2, 0]))
    np.testing.assert_array_equal(info.l_channel_ordinals, np.array([0, 0, 0, 1]))

    eigenvalues, orbitals, l_terms = _synthetic_full_spectrum(
        info, solver.ops_builder_standard
    )
    response = ResponseCalculator(info, solver.ops_builder_standard)
    chi_0 = response.compute_chi_0_kernel(eigenvalues, orbitals, l_terms)

    assert chi_0.shape == (len(solver.grid_data_standard.quadrature_nodes),) * 2
    assert np.all(np.isfinite(chi_0))


def test_sparse_l_channels_are_supported_by_response_and_oep():
    solver = _small_solver()
    info = OccupationInfo(
        z_nuclear=26,
        z_valence=8,
        all_electron_flag=False,
    )
    assert info.unique_l_values.tolist() == [0, 2]
    eigenvalues, orbitals, l_terms = _synthetic_full_spectrum(
        info, solver.ops_builder_standard
    )

    response = ResponseCalculator(info, solver.ops_builder_standard)
    chi_0 = response.compute_chi_0_kernel(eigenvalues, orbitals, l_terms)

    oep = OEPCalculator(
        ops_builder=solver.ops_builder_standard,
        ops_builder_dense=solver.ops_builder_dense,
        ops_builder_oep=solver.ops_builder_standard,
        occupation_info=info,
        use_rpa_correlation=False,
    )
    exchange_potentials = np.zeros((info.n_states, orbitals.shape[0]))
    oep_kernel, driving_term = (
        oep._compute_oep_kernel_and_exchange_driving_term(
            eigenvalues,
            orbitals,
            l_terms,
            exchange_potentials,
        )
    )

    assert np.all(np.isfinite(chi_0))
    assert np.all(np.isfinite(oep_kernel))
    assert np.all(np.isfinite(driving_term))


def test_scan_input_flooring_does_not_mutate_density_data():
    rho = np.array([0.0, 0.5])
    gradient = np.array([0.0, 0.25])
    tau = np.array([0.1, 0.2])
    density = DensityData(rho=rho.copy(), grad_rho=gradient.copy(), tau=tau)

    safe_rho, returned_tau, sigma = _get_rho_tau_and_sigma(density)

    np.testing.assert_array_equal(density.rho, rho)
    np.testing.assert_array_equal(density.grad_rho, gradient)
    assert safe_rho[0] > 0.0
    assert sigma[0] > 0.0
    assert returned_tau is tau

    with pytest.raises(ValueError, match="grad_rho.*tau"):
        _get_rho_tau_and_sigma(DensityData(rho=rho, tau=tau))


def test_rank_deficient_pulay_history_uses_least_squares():
    mixer = Mixer(
        use_pulay_mixing=True,
        use_preconditioner=False,
        pulay_mixing_parameter=0.5,
        pulay_mixing_history=3,
        pulay_mixing_frequency=1,
    )
    mixer.rho_in_store = np.zeros((4, 4))
    mixer.rho_out_store = np.ones((4, 4))

    mixed = mixer._pulay_mix_early(runs=2, preconditioner=None)

    assert np.all(np.isfinite(mixed))


def test_optimized_python_fails_closed_instead_of_disabling_assertions():
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    completed = subprocess.run(
        [sys.executable, "-O", "-c", "import atom"],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "cannot run with Python optimization" in completed.stderr


def test_rpa_energy_density_request_fails_before_scf_work():
    solver_stub = SimpleNamespace(xc_functional="RPA")

    with pytest.raises(NotImplementedError, match="not implemented"):
        AtomicDFTSolver.solve(solver_stub, save_energy_density=True)


def test_hf_differential_energy_density_maps_from_dense_grid():
    solver = _small_solver()
    exchange = HartreeFockExchange(
        ops_builder=solver.ops_builder_standard,
        ops_builder_dense=solver.ops_builder_dense,
        occupation_info=solver.occupation_info,
    )
    orbitals = np.ones(
        (
            len(solver.ops_builder_standard.quadrature_nodes),
            solver.occupation_info.n_states,
        )
    )

    energy_density = exchange.compute_exchange_energy_density(
        orbitals, method="differential_equation"
    )

    assert energy_density.shape == (len(solver.ops_builder_standard.quadrature_nodes),)
    assert np.all(np.isfinite(energy_density))


def test_xc_pytree_round_trip_preserves_meta_gga_derivatives():
    arrays = [np.full(3, float(index)) for index in range(1, 7)]
    original = XCPotentialData(*arrays)

    children, auxiliary_data = _xc_potential_data_flatten(original)
    restored = _xc_potential_data_unflatten(auxiliary_data, children)

    assert len(children) == 6
    np.testing.assert_array_equal(restored.de_x_dtau, arrays[4])
    np.testing.assert_array_equal(restored.de_c_dtau, arrays[5])


def test_lda_pz_zero_density_is_finite_and_negative_density_is_rejected():
    evaluator = LDA_PZ()
    density = DensityData(rho=np.array([0.0, 1e-30, 1.0]))

    exchange = evaluator.compute_exchange_generic(density)
    correlation = evaluator.compute_correlation_generic(density)

    assert np.all(np.isfinite(exchange.v_generic))
    assert np.all(np.isfinite(exchange.e_generic))
    assert np.all(np.isfinite(correlation.v_generic))
    assert np.all(np.isfinite(correlation.e_generic))
    assert correlation.v_generic[0] == 0.0
    assert correlation.e_generic[0] == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        evaluator.compute_exchange_generic(DensityData(rho=np.array([-1e-6])))


def test_derivative_matrix_cache_is_reused():
    solver = _small_solver()
    builder = solver.ops_builder_standard

    first = builder.get_derivative_matrix_with_quadrature_basis()
    second = builder.get_derivative_matrix_with_quadrature_basis()

    assert second is first


def test_poisson_eliminates_both_boundaries_and_computes_energy():
    solver = object.__new__(PoissonSolver)
    solver.laplacian = np.array(
        [
            [-2.0, 1.0, 0.0],
            [1.0, -2.0, 1.0],
            [0.0, 1.0, -2.0],
        ]
    )

    solution = solver.solve_1d(np.array([2.0, 0.0, 4.0]))
    np.testing.assert_allclose(solution, np.array([2.0, 3.0, 4.0]))

    solver.quadrature_nodes = np.array([1.0, 2.0, 3.0])
    solver.quadrature_weights = np.array([0.2, 0.3, 0.5])
    rho = np.array([0.1, 0.2, 0.3])
    potential = np.array([1.5, 1.0, 0.5])
    expected = 0.5 * np.sum(
        rho
        * potential
        * 4.0
        * np.pi
        * solver.quadrature_nodes**2
        * solver.quadrature_weights
    )
    assert solver.compute_hartree_energy(rho, potential) == pytest.approx(expected)


def test_radial_descriptor_projection_retains_anisotropic_spacing():
    result = compute_descriptors_from_radial(
        r_radial=np.linspace(0.0, 4.0, 41),
        rho_radial=np.exp(-np.linspace(0.0, 4.0, 41)),
        box_size=4.0,
        spacing=(1.0, 0.8, 0.5),
        atom_center=(2.0, 2.0, 2.0),
        rcuts=[0.5],
        l_max=0,
        eval_indices=np.array([[2, 2, 4]]),
        periodic=False,
    )

    np.testing.assert_allclose(result.spacing, (1.0, 0.8, 0.5))


@pytest.mark.parametrize("length", [3, 10, 20])
def test_lowpass_smoothing_handles_short_signals(length):
    radius = np.arange(1, length + 1, dtype=float)
    values = np.sin(radius)

    smoothed = DataProcessor.smooth_vxc_data(
        values,
        radius,
        r_threshold=0.5,
        method="lowpass",
    )

    assert smoothed.shape == values.shape
    assert np.all(np.isfinite(smoothed))


def test_cascade_smoothing_implements_every_documented_method():
    radius = np.arange(1, 13, dtype=float)
    values = np.sin(radius)

    smoothed = DataProcessor.smooth_vxc_data(
        values,
        radius,
        r_threshold=0.5,
        method="cascade",
        methods=["savgol", "spline", "exp_weighted"],
        kwargs_list=[
            {"window_length": 5, "polyorder": 2},
            {"s": 0.1},
            {"alpha": 0.2},
        ],
    )

    assert smoothed.shape == values.shape
    assert np.all(np.isfinite(smoothed))


def test_integer_fractional_occupation_api_input_is_supported():
    n, l, spin_up, spin_down = get_fraction_occupation_states(8)

    assert len(n) == len(l) == len(spin_up) == len(spin_down)
    assert np.sum(spin_up + spin_down) == pytest.approx(8.0)


def test_output_round_trip_preserves_iteration_limits_and_license(tmp_path):
    solver = _small_solver(
        xc_functional="PBE0",
        max_scf_iterations=17,
        max_scf_iterations_outer=9,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        solver.print_input_parameters()
    output_text = output.getvalue()
    output_path = tmp_path / "atom.out"
    output_path.write_text(output_text)

    restored = AtomicDFTSolver.from_output_file(output_path, verbose=False)

    assert "MIT License" in output_text
    assert restored.max_scf_iterations == 17
    assert restored.max_scf_iterations_outer == 9
