"""Tests for descriptor density utility helpers."""

import numpy as np
import pytest

from atom.descriptors.density import (
    centered_single_gaussian_density,
    normalize_density,
)


def test_normalize_density_uses_supported_trapezoid_api():
    r = np.linspace(0.0, 4.0, 41)
    rho = centered_single_gaussian_density(r, sigma=0.6, n_electrons=1.0)

    normalized = normalize_density(r, rho, n_electrons=2.0)

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    electron_count = 4.0 * np.pi * trapezoid(r * r * normalized, r)
    assert electron_count == pytest.approx(2.0)
