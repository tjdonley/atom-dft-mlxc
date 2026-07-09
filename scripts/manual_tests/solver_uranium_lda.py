"""
Test case for Uranium (Z=92) LDA_PZ calculation against reference values.

This module tests the accuracy of the AtomicDFTSolver for uranium atom
using LDA_PZ functional by comparing computed eigenvalues with reference
values from the featom paper.

Reference:
----------
Ondřej Čertík, John E. Pask, Isuru Fernando, Rohit Goswami, N. Sukumar, 
Lee. A. Collins, Gianmarco Manzini, Jiří Vackář,
High-order finite element method for atomic structure calculations,
Computer Physics Communications,
Volume 297,
2024,
109051,
ISSN 0010-4655,
https://doi.org/10.1016/j.cpc.2023.109051.
(https://www.sciencedirect.com/science/article/pii/S001046552300396X)

The reference implementation (featom) achieves a high level of accuracy 
(10^-8 Hartree) for total energies and eigenvalues of heavy atoms such as 
uranium in both Schrödinger and Dirac Kohn-Sham solutions.
"""




import os
import sys
import numpy as np
import time

# Add the repository root to the import path.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from atom.solver import AtomicDFTSolver


def print_test_passed(test_name: str):
    """Print test passed message."""
    print("\t {:<50} : test passed".format(test_name))


def print_test_failed(test_name: str, error_msg: str = ""):
    """Print test failed message."""
    print("\t {:<50} : test FAILED".format(test_name))
    if error_msg:
        print("\t\t Error: {}".format(error_msg))


def test_uranium_lda_pz_eigenvalues():
    """
    Test Uranium (Z=92) LDA_PZ calculation against reference eigenvalues.
    
    This test verifies that the computed eigenvalues match the reference
    values from the featom paper within acceptable tolerance.
    """
    print("\n" + "=" * 60)
    print("Test: Uranium (Z=92) LDA_PZ eigenvalues vs. featom reference")
    print("=" * 60)
    print("Reference: Čertík et al., Comput. Phys. Commun. 297, 109051 (2024)")
    print("=" * 60)
    
    # Reference eigenvalues from featom paper (in Hartree)
    ref_eigenvalues = np.array([
        -3689.35513984,
        -639.77872809,
        -619.10855018,
        -161.11807321,
        -150.97898016,
        -131.97735828,
        -40.52808425,
        -35.85332083,
        -27.12321230, 
        -15.02746007,
        -8.82408940,
        -7.01809220, 
        -3.86617513,
        -0.36654335,
        -1.32597632,
        -0.82253797,
        -0.14319018,
        -0.13094786
    ])
    
    try:
        start_time = time.time()
        
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 92,
            xc_functional             = 'LDA_PZ',
            domain_size               = 40.0,
            finite_element_number     = 8,
            polynomial_order          = 31,
            quadrature_point_number   = 70,
            mesh_type                 = "exponential",
            mesh_concentration        = 101.0,
            scf_tolerance             = 1e-11,
            verbose                   = True, 
            all_electron_flag         = True,
            use_oep                   = False,
            use_preconditioner        = True,
        )

        results = atomic_dft_solver.solve(save_energy_density=True)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nComputation time: {elapsed_time:.2f} seconds")
        
        # Extract results
        rho = results['rho']
        orbitals = results['orbitals']
        computed_eigenvalues = results['eigen_energies']
        
        print(f"\nrho.shape      = {rho.shape}")      # (n_grid_points,)
        print(f"orbitals.shape = {orbitals.shape}")   # (n_grid_points, n_orbitals)
        print(f"Number of eigenvalues computed: {len(computed_eigenvalues)}")
        print(f"Number of reference eigenvalues: {len(ref_eigenvalues)}")
        
        # Compare eigenvalues
        n_compare = min(len(computed_eigenvalues), len(ref_eigenvalues))
        computed_subset = computed_eigenvalues[:n_compare]
        ref_subset = ref_eigenvalues[:n_compare]
        
        differences = computed_subset - ref_subset
        max_diff = np.max(np.abs(differences))
        mean_diff = np.mean(np.abs(differences))
        rms_diff = np.sqrt(np.mean(differences**2))
        
        print(f"\nEigenvalue comparison (first {n_compare} eigenvalues):")
        print(f"  Max absolute difference:  {max_diff:.2e} Hartree")
        print(f"  Mean absolute difference: {mean_diff:.2e} Hartree")
        print(f"  RMS difference:           {rms_diff:.2e} Hartree")
        
        # Print detailed comparison
        print("\nDetailed comparison:")
        print(f"{'Index':<6} {'Computed':<15} {'Reference':<15} {'Difference':<15}")
        print("-" * 60)
        for i in range(n_compare):
            diff = computed_subset[i] - ref_subset[i]
            print(f"{i:<6} {computed_subset[i]:<15.8f} {ref_subset[i]:<15.8f} {diff:<15.8e}")
        
        # Check if energy density was saved
        if 'e_x_local' in results and 'e_c_local' in results:
            e_x_local = results['e_x_local']
            e_c_local = results['e_c_local']
            print(f"\ne_x_local.shape = {e_x_local.shape}")
            print(f"e_c_local.shape = {e_c_local.shape}")
        
        # Tolerance check: featom achieves 10^-8 Hartree accuracy
        # We use a slightly more relaxed tolerance for this test
        tolerance = 1e-5  # 0.01 mHartree
        
        if max_diff < tolerance:
            print_test_passed(f"Uranium LDA_PZ eigenvalues (max diff < {tolerance:.1e} Hartree)")
            return True
        else:
            error_msg = f"Max difference {max_diff:.2e} exceeds tolerance {tolerance:.1e}"
            print_test_failed("Uranium LDA_PZ eigenvalues", error_msg)
            print(f"\nWarning: Differences exceed tolerance but may still be acceptable")
            print(f"         depending on computational parameters and mesh settings.")
            return False
        
    except Exception as e:
        print_test_failed("Uranium LDA_PZ eigenvalues", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_uranium_lda_pz_basic():
    """
    Test basic Uranium (Z=92) LDA_PZ calculation without detailed comparison.
    
    This is a lighter test that just verifies the calculation completes
    successfully without comparing to reference values.
    """
    print("\n" + "=" * 60)
    print("Test: Basic Uranium (Z=92) LDA_PZ calculation")
    print("=" * 60)
    
    try:
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number             = 92,
            xc_functional             = 'LDA_PZ',
            domain_size               = 40.0,
            finite_element_number     = 8,
            polynomial_order          = 31,
            quadrature_point_number   = 70,
            mesh_type                 = "exponential",
            mesh_concentration        = 101.0,
            verbose                   = False,  # Less verbose for basic test
            all_electron_flag         = True,
            use_oep                   = False,
            use_preconditioner        = True,
        )

        results = atomic_dft_solver.solve(save_energy_density=True)
        
        # Basic assertions
        assert results is not None, "Results should not be None"
        assert 'rho' in results, "Results should contain 'rho'"
        assert 'orbitals' in results, "Results should contain 'orbitals'"
        assert 'eigen_energies' in results, "Results should contain 'eigen_energies'"
        
        rho = results['rho']
        orbitals = results['orbitals']
        eigen_energies = results['eigen_energies']
        
        assert len(rho) > 0, "Density should have non-zero length"
        assert orbitals.shape[0] > 0, "Orbitals should have non-zero shape"
        assert len(eigen_energies) > 0, "Eigenvalues should have non-zero length"
        
        # Check that eigenvalues are negative (bound states)
        assert np.all(eigen_energies < 0), "All eigenvalues should be negative (bound states)"
        
        print_test_passed("Basic Uranium LDA_PZ calculation")
        return True
        
    except Exception as e:
        print_test_failed("Basic Uranium LDA_PZ calculation", str(e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Uranium (Z=92) LDA_PZ Test Suite")
    print("=" * 60)
    print("Reference: Čertík et al., Comput. Phys. Commun. 297, 109051 (2024)")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Basic test (faster)
    test_results.append(("Basic calculation", test_uranium_lda_pz_basic()))
    
    # Detailed eigenvalue comparison test
    test_results.append(("Eigenvalue comparison", test_uranium_lda_pz_eigenvalues()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:<30} : {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)
