"""
Test case for AtomicDFTSolver.save_intermediate functionality.

This module tests the intermediate information saving feature in the SCF solver.
It verifies that:
1. When save_intermediate=False, intermediate_info is None
2. When save_intermediate=True, intermediate_info contains valid data
3. The intermediate_info structure matches the expected format
4. Inner and outer iteration data is correctly recorded
"""

import os
import sys
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atom.solver import AtomicDFTSolver


def print_test_passed(test_name: str):
    """Print test passed message."""
    print("\t {:<50} : test passed".format(test_name))


def print_test_failed(test_name: str, error_msg: str = ""):
    """Print test failed message."""
    print("\t {:<50} : test FAILED".format(test_name))
    if error_msg:
        print("\t\t Error: {}".format(error_msg))


def test_solve_without_intermediate():
    """
    Test that solve() with save_intermediate=False returns None for intermediate_info.
    """
    print("\n" + "=" * 60)
    print("Test: solve() without intermediate information")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose           = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=False)
        
        # Check that intermediate_info is None
        assert results['intermediate_info'] is None, \
            "intermediate_info should be None when save_intermediate=False"
        
        # Check that other results are present
        assert 'rho' in results, "Results should contain 'rho'"
        assert 'orbitals' in results, "Results should contain 'orbitals'"
        assert 'energy' in results, "Results should contain 'energy'"
        
        print_test_passed("solve() without intermediate information")
        return True
        
    except Exception as e:
        print_test_failed("solve() without intermediate information", str(e))
        return False


def test_solve_with_intermediate():
    """
    Test that solve() with save_intermediate=True returns valid intermediate_info.
    """
    print("\n" + "=" * 60)
    print("Test: solve() with intermediate information")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose           = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=True)
        
        # Check that intermediate_info is not None
        assert results['intermediate_info'] is not None, \
            "intermediate_info should not be None when save_intermediate=True"
        
        intermediate_info = results['intermediate_info']
        
        # Check that intermediate_info has the expected attributes
        assert hasattr(intermediate_info, 'inner_iterations'), \
            "intermediate_info should have 'inner_iterations' attribute"
        assert hasattr(intermediate_info, 'outer_iterations'), \
            "intermediate_info should have 'outer_iterations' attribute"
        assert hasattr(intermediate_info, 'current_outer_iteration'), \
            "intermediate_info should have 'current_outer_iteration' attribute"
        
        # Check types
        assert isinstance(intermediate_info.inner_iterations, list), \
            "inner_iterations should be a list"
        assert isinstance(intermediate_info.outer_iterations, list), \
            "outer_iterations should be a list"
        assert isinstance(intermediate_info.current_outer_iteration, (int, np.integer)), \
            "current_outer_iteration should be an integer"
        
        # Check that we have at least some inner iterations
        assert len(intermediate_info.inner_iterations) > 0, \
            "Should have at least one inner iteration"
        
        # Check structure of inner iteration info
        if len(intermediate_info.inner_iterations) > 0:
            inner_iter = intermediate_info.inner_iterations[0]
            assert hasattr(inner_iter, 'outer_iteration'), \
                "InnerIterationInfo should have 'outer_iteration' attribute"
            assert hasattr(inner_iter, 'inner_iteration'), \
                "InnerIterationInfo should have 'inner_iteration' attribute"
            assert hasattr(inner_iter, 'rho_residual'), \
                "InnerIterationInfo should have 'rho_residual' attribute"
            assert hasattr(inner_iter, 'rho'), \
                "InnerIterationInfo should have 'rho' attribute"
            assert hasattr(inner_iter, 'rho_norm'), \
                "InnerIterationInfo should have 'rho_norm' attribute"
            
            # Check types of inner iteration attributes
            assert isinstance(inner_iter.outer_iteration, (int, np.integer)), \
                "outer_iteration should be an integer"
            assert isinstance(inner_iter.inner_iteration, (int, np.integer)), \
                "inner_iteration should be an integer"
            assert isinstance(inner_iter.rho_residual, (float, np.floating)), \
                "rho_residual should be a float"
            assert isinstance(inner_iter.rho, np.ndarray), \
                "rho should be a numpy array"
            assert isinstance(inner_iter.rho_norm, (float, np.floating)), \
                "rho_norm should be a float"
        
        print_test_passed("solve() with intermediate information - structure check")
        print(f"\t  Number of inner iterations: {len(intermediate_info.inner_iterations)}")
        print(f"\t  Number of outer iterations: {len(intermediate_info.outer_iterations)}")
        print(f"\t  Current outer iteration: {intermediate_info.current_outer_iteration}")
        
        return True
        
    except Exception as e:
        print_test_failed("solve() with intermediate information", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_data_consistency():
    """
    Test that intermediate data is consistent and valid.
    """
    print("\n" + "=" * 60)
    print("Test: Intermediate data consistency")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=True)
        intermediate_info = results['intermediate_info']
        
        # Check that inner iterations have valid data
        for i, inner_iter in enumerate(intermediate_info.inner_iterations):
            assert inner_iter.outer_iteration >= 0, \
                f"Inner iteration {i}: outer_iteration should be >= 0"
            assert inner_iter.inner_iteration > 0, \
                f"Inner iteration {i}: inner_iteration should be > 0"
            assert inner_iter.rho_residual >= 0, \
                f"Inner iteration {i}: rho_residual should be >= 0"
            assert inner_iter.rho_norm > 0, \
                f"Inner iteration {i}: rho_norm should be > 0"
            assert len(inner_iter.rho) > 0, \
                f"Inner iteration {i}: rho should not be empty"
            assert np.all(np.isfinite(inner_iter.rho)), \
                f"Inner iteration {i}: rho should contain finite values"
        
        # Check that outer iterations have valid data (if any)
        for i, outer_iter in enumerate(intermediate_info.outer_iterations):
            assert outer_iter.outer_iteration > 0, \
                f"Outer iteration {i}: outer_iteration should be > 0"
            assert outer_iter.outer_rho_residual >= 0, \
                f"Outer iteration {i}: outer_rho_residual should be >= 0"
            assert isinstance(outer_iter.converged, bool), \
                f"Outer iteration {i}: converged should be a boolean"
            assert outer_iter.iterations > 0, \
                f"Outer iteration {i}: iterations should be > 0"
            assert isinstance(outer_iter.eigen_energies, np.ndarray), \
                f"Outer iteration {i}: eigen_energies should be a numpy array"
            assert isinstance(outer_iter.orbitals, np.ndarray), \
                f"Outer iteration {i}: orbitals should be a numpy array"
            assert isinstance(outer_iter.density_data, type(results['density_data'])), \
                f"Outer iteration {i}: density_data should be a DensityData instance"
            assert isinstance(outer_iter.inner_iterations, list), \
                f"Outer iteration {i}: inner_iterations should be a list"
        
        print_test_passed("Intermediate data consistency check")
        return True
        
    except Exception as e:
        print_test_failed("Intermediate data consistency", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_with_outer_loop():
    """
    Test intermediate information with a functional that uses outer loop (e.g., PBE0).
    """
    print("\n" + "=" * 60)
    print("Test: Intermediate information with outer loop (PBE0)")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "PBE0",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=True)
        intermediate_info = results['intermediate_info']
        
        # For PBE0, we should have outer iterations
        assert len(intermediate_info.outer_iterations) > 0, \
            "PBE0 should have at least one outer iteration"
        
        # Check outer iteration structure
        outer_iter = intermediate_info.outer_iterations[0]
        assert len(outer_iter.inner_iterations) > 0, \
            "Each outer iteration should have at least one inner iteration"
        
        # Check that inner iterations in outer_iter match the structure
        for inner_iter in outer_iter.inner_iterations:
            assert hasattr(inner_iter, 'outer_iteration'), \
                "Inner iteration in outer_iter should have 'outer_iteration' attribute"
            assert hasattr(inner_iter, 'inner_iteration'), \
                "Inner iteration in outer_iter should have 'inner_iteration' attribute"
        
        print_test_passed("Intermediate information with outer loop (PBE0)")
        print(f"\t  Number of outer iterations: {len(intermediate_info.outer_iterations)}")
        print(f"\t  First outer iteration has {len(outer_iter.inner_iterations)} inner iterations")
        
        return True
        
    except Exception as e:
        print_test_failed("Intermediate information with outer loop", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_save_intermediate_does_not_affect_results():
    """
    Test that save_intermediate parameter 'does' not affect the calculation results.
    
    This ensures that enabling intermediate information saving does not change
    the computed values (energy, density, orbitals, etc.).
    """
    print("\n" + "=" * 60)
    print("Test: save_intermediate does not affect results")
    print("=" * 60)
    
    try:
        # Create solver
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        
        # Run with save_intermediate=False
        results_without = atomic_dft_solver.solve(save_intermediate=False)
        
        # Run with save_intermediate=True
        results_with = atomic_dft_solver.solve(save_intermediate=True)
        
        # Check that intermediate_info is different (None vs not None)
        assert results_without['intermediate_info'] is None, \
            "intermediate_info should be None when save_intermediate=False"
        assert results_with['intermediate_info'] is not None, \
            "intermediate_info should not be None when save_intermediate=True"
        
        # Check that all other results are identical (within numerical precision)
        # Energy
        assert np.isclose(results_without['energy'], results_with['energy'], rtol=1e-10), \
            f"Energy mismatch: {results_without['energy']} vs {results_with['energy']}"
        
        # Density
        assert np.allclose(results_without['rho'], results_with['rho'], rtol=1e-10), \
            "Density mismatch between save_intermediate=False and True"
        
        # Orbitals
        assert np.allclose(results_without['orbitals'], results_with['orbitals'], rtol=1e-10), \
            "Orbitals mismatch between save_intermediate=False and True"
        
        # Eigen energies
        assert np.allclose(results_without['eigen_energies'], results_with['eigen_energies'], rtol=1e-10), \
            "Eigen energies mismatch between save_intermediate=False and True"
        
        # Convergence status
        assert results_without['converged'] == results_with['converged'], \
            "Convergence status mismatch"
        assert results_without['iterations'] == results_with['iterations'], \
            "Iterations count mismatch"
        
        # Residual (should be very close)
        assert np.isclose(results_without['rho_residual'], results_with['rho_residual'], rtol=1e-10), \
            "Residual mismatch"
        
        print_test_passed("save_intermediate does not affect results")
        print(f"\t  Energy: {results_with['energy']:.10f} Ha")
        print(f"\t  Converged: {results_with['converged']}")
        print(f"\t  Iterations: {results_with['iterations']}")
        print(f"\t  Residual: {results_with['rho_residual']:.6e}")
        
        return True
        
    except Exception as e:
        print_test_failed("save_intermediate does not affect results", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_info_completeness():
    """
    Test that intermediate_info contains complete and consistent data.
    """
    print("\n" + "=" * 60)
    print("Test: Intermediate information completeness")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=True)
        intermediate_info = results['intermediate_info']
        
        # Check that inner iterations match the final iteration count
        assert len(intermediate_info.inner_iterations) == results['iterations'], \
            f"Number of inner iterations ({len(intermediate_info.inner_iterations)}) should match final iterations ({results['iterations']})"
        
        # Check that the last inner iteration matches final results
        if len(intermediate_info.inner_iterations) > 0:
            last_inner = intermediate_info.inner_iterations[-1]
            assert np.isclose(last_inner.rho_residual, results['rho_residual'], rtol=1e-10), \
                "Last inner iteration residual should match final residual"
            assert np.allclose(last_inner.rho, results['rho'], rtol=1e-10), \
                "Last inner iteration density should match final density"
        
        # Check that inner iterations are in order
        for i in range(len(intermediate_info.inner_iterations) - 1):
            curr = intermediate_info.inner_iterations[i]
            next_iter = intermediate_info.inner_iterations[i + 1]
            assert curr.inner_iteration < next_iter.inner_iteration, \
                f"Inner iterations should be in order: {curr.inner_iteration} < {next_iter.inner_iteration}"
        
        # Check that outer_iterations is empty for non-outer-loop functionals
        assert len(intermediate_info.outer_iterations) == 0, \
            "GGA_PBE should not have outer iterations"
        
        print_test_passed("Intermediate information completeness")
        print(f"\t  Inner iterations recorded: {len(intermediate_info.inner_iterations)}")
        print(f"\t  Final iteration matches: {len(intermediate_info.inner_iterations) == results['iterations']}")
        
        return True
        
    except Exception as e:
        print_test_failed("Intermediate information completeness", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_info_with_outer_loop_completeness():
    """
    Test that intermediate_info is complete for outer loop calculations.
    """
    print("\n" + "=" * 60)
    print("Test: Intermediate information completeness (with outer loop)")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "PBE0",
            all_electron_flag = False,
        )
        
        results = atomic_dft_solver.solve(save_intermediate=True)
        intermediate_info = results['intermediate_info']
        
        # Should have outer iterations for PBE0
        assert len(intermediate_info.outer_iterations) > 0, \
            "PBE0 should have outer iterations"
        
        # Check that each outer iteration has inner iterations
        total_inner_in_outer = 0
        for outer_iter in intermediate_info.outer_iterations:
            assert len(outer_iter.inner_iterations) > 0, \
                f"Outer iteration {outer_iter.outer_iteration} should have inner iterations"
            total_inner_in_outer += len(outer_iter.inner_iterations)
            
            # Check that inner iterations in outer_iter match the iterations count
            assert len(outer_iter.inner_iterations) == outer_iter.iterations, \
                f"Inner iterations count ({len(outer_iter.inner_iterations)}) should match outer_iter.iterations ({outer_iter.iterations})"
        
        # Check that outer iterations are in order
        for i in range(len(intermediate_info.outer_iterations) - 1):
            curr = intermediate_info.outer_iterations[i]
            next_iter = intermediate_info.outer_iterations[i + 1]
            assert curr.outer_iteration < next_iter.outer_iteration, \
                f"Outer iterations should be in order: {curr.outer_iteration} < {next_iter.outer_iteration}"
        
        # Check that the last outer iteration's last inner iteration matches final results
        if len(intermediate_info.outer_iterations) > 0:
            last_outer = intermediate_info.outer_iterations[-1]
            if len(last_outer.inner_iterations) > 0:
                last_inner = last_outer.inner_iterations[-1]
                assert np.isclose(last_inner.rho_residual, results['rho_residual'], rtol=1e-10), \
                    "Last inner iteration residual should match final residual"
        
        print_test_passed("Intermediate information completeness (with outer loop)")
        print(f"\t  Outer iterations: {len(intermediate_info.outer_iterations)}")
        print(f"\t  Total inner iterations in outer loops: {total_inner_in_outer}")
        
        return True
        
    except Exception as e:
        print_test_failed("Intermediate information completeness (with outer loop)", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_info_consistency_across_runs():
    """
    Test that intermediate_info is consistent across multiple runs with same parameters.
    """
    print("\n" + "=" * 60)
    print("Test: Intermediate information consistency across runs")
    print("=" * 60)
    
    try:
        # Run 1
        solver1 = AtomicDFTSolver(
            atomic_number     = 13,
            verbose           = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        results1 = solver1.solve(save_intermediate=True)
        info1 = results1['intermediate_info']
        
        # Run 2 (same parameters)
        solver2 = AtomicDFTSolver(
            atomic_number     = 13,
            verbose           = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        results2 = solver2.solve(save_intermediate=True)
        info2 = results2['intermediate_info']
        
        # Check that iteration counts match
        assert len(info1.inner_iterations) == len(info2.inner_iterations), \
            "Number of inner iterations should be consistent across runs"
        
        # Check that residuals are similar (within numerical precision)
        for i, (iter1, iter2) in enumerate(zip(info1.inner_iterations, info2.inner_iterations)):
            assert np.isclose(iter1.rho_residual, iter2.rho_residual, rtol=1e-8), \
                f"Iteration {i+1}: residual mismatch between runs"
            assert np.allclose(iter1.rho, iter2.rho, rtol=1e-8), \
                f"Iteration {i+1}: density mismatch between runs"
        
        print_test_passed("Intermediate information consistency across runs")
        print(f"\t  Both runs had {len(info1.inner_iterations)} inner iterations")
        
        return True
        
    except Exception as e:
        print_test_failed("Intermediate information consistency across runs", str(e))
        import traceback
        traceback.print_exc()
        return False


def print_intermediate_summary(intermediate_info, max_inner=5):
    """
    Print a summary of intermediate information.
    
    Parameters
    ----------
    intermediate_info : IntermediateInfo
        The intermediate information object
    max_inner : int
        Maximum number of inner iterations to print
    """
    print("\n" + "-" * 60)
    print("Intermediate Information Summary")
    print("-" * 60)
    print(f"Total inner iterations: {len(intermediate_info.inner_iterations)}")
    print(f"Total outer iterations: {len(intermediate_info.outer_iterations)}")
    print(f"Current outer iteration: {intermediate_info.current_outer_iteration}")
    
    if len(intermediate_info.inner_iterations) > 0:
        print(f"\nFirst {min(max_inner, len(intermediate_info.inner_iterations))} inner iterations:")
        for i, inner_iter in enumerate(intermediate_info.inner_iterations[:max_inner]):
            print(f"  [{i+1}] Outer={inner_iter.outer_iteration}, "
                  f"Inner={inner_iter.inner_iteration}, "
                  f"Residual={inner_iter.rho_residual:.6e}, "
                  f"Rho_norm={inner_iter.rho_norm:.6e}")
    
    if len(intermediate_info.outer_iterations) > 0:
        print(f"\nOuter iterations ({len(intermediate_info.outer_iterations)}):")
        for outer_iter in intermediate_info.outer_iterations:
            print(f"  Outer {outer_iter.outer_iteration}: "
                  f"Converged={outer_iter.converged}, "
                  f"Iterations={outer_iter.iterations}, "
                  f"Residual={outer_iter.outer_rho_residual:.6e}, "
                  f"Inner iterations={len(outer_iter.inner_iterations)}")
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AtomicDFTSolver.save_intermediate functionality")
    print("=" * 60)
    
    # Run all tests
    test_results = []
    
    test_results.append(("solve() without intermediate", test_solve_without_intermediate()))
    test_results.append(("solve() with intermediate", test_solve_with_intermediate()))
    test_results.append(("intermediate data consistency", test_intermediate_data_consistency()))
    test_results.append(("save_intermediate does not affect results", test_save_intermediate_does_not_affect_results()))
    test_results.append(("intermediate info completeness", test_intermediate_info_completeness()))
    test_results.append(("intermediate info consistency across runs", test_intermediate_info_consistency_across_runs()))
    
    # Optional: test with outer loop (takes longer)
    # Uncomment to test with PBE0
    # test_results.append(("intermediate with outer loop", test_intermediate_with_outer_loop()))
    # test_results.append(("intermediate info completeness (outer loop)", test_intermediate_info_with_outer_loop_completeness()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:<50} : {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! ✓")
    else:
        print(f"\n{total - passed} test(s) failed. ✗")
        sys.exit(1)
    
    # Optional: Print detailed intermediate information from last test
    if test_results[-1][1]:  # If last test passed
        print("\n" + "=" * 60)
        print("Detailed Intermediate Information (from last test)")
        print("=" * 60)
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number     = 13,
            verbose       = False,
            xc_functional     = "GGA_PBE",
            all_electron_flag = False,
        )
        results = atomic_dft_solver.solve(save_intermediate=True)
        print_intermediate_summary(results['intermediate_info'])

