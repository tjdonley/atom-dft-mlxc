"""MCSH angular basis implementation for the multipole framework."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .basis import AngularBasis


class MCSHBasis(AngularBasis):
    """Maxwell-Cartesian spherical harmonic angular basis.

    This basis reproduces the current MCSH invariants used by the descriptor
    validation suite and the SPARC-aligned implementation in this repository.
    """

    name = "mcsh"

    _components = {
        0: [("000", 1.0)],
        1: [("100", 1.0), ("010", 1.0), ("001", 1.0)],
        2: [
            ("200", 1.0),
            ("020", 1.0),
            ("002", 1.0),
            ("110", 2.0),
            ("101", 2.0),
            ("011", 2.0),
        ],
    }

    def component_specs(self, l: int) -> Sequence[tuple[str, float]]:
        if l not in self._components:
            raise ValueError(f"MCSH basis does not support l={l}")
        return self._components[l]

    def evaluate_component(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        l: int,
        label: str,
    ) -> np.ndarray:
        if l == 0 and label == "000":
            return np.ones_like(dx)
        if l == 1:
            if label == "100":
                return dx
            if label == "010":
                return dy
            if label == "001":
                return dz
        if l == 2:
            r_sq = dx * dx + dy * dy + dz * dz
            if label == "200":
                return 3.0 * dx * dx - r_sq
            if label == "020":
                return 3.0 * dy * dy - r_sq
            if label == "002":
                return 3.0 * dz * dz - r_sq
            if label == "110":
                return 3.0 * dx * dy
            if label == "101":
                return 3.0 * dx * dz
            if label == "011":
                return 3.0 * dy * dz
        raise ValueError(f"Unknown MCSH component l={l}, label={label!r}")

    def combine_invariant(
        self,
        l: int,
        components: Sequence[tuple[float, float]],
    ) -> float:
        if l == 0:
            return float(components[0][1])
        power = 0.0
        for weight, zeta in components:
            power += weight * zeta * zeta
        return float(np.sqrt(power))
