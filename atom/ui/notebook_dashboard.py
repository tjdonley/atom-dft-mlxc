"""Interactive Jupyter dashboard for the atomic DFT solver.

The notebook imports this module and displays one widget tree; solver state,
queued jobs, plots, and exports all live here instead of in notebook cells.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display
except ImportError as exc:  # pragma: no cover - exercised only in missing optional envs
    raise ImportError(
        "The notebook dashboard needs ipywidgets, matplotlib, and IPython. "
        "Install them with `pip install ipywidgets matplotlib` or `pip install -e '.[dev]'`."
    ) from exc

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator
from atom.solver import VALID_XC_FUNCTIONAL_LIST


ATOMS: list[tuple[int, str, str]] = [
    (1, "H", "Hydrogen"), (2, "He", "Helium"), (3, "Li", "Lithium"),
    (4, "Be", "Beryllium"), (5, "B", "Boron"), (6, "C", "Carbon"),
    (7, "N", "Nitrogen"), (8, "O", "Oxygen"), (9, "F", "Fluorine"),
    (10, "Ne", "Neon"), (11, "Na", "Sodium"), (12, "Mg", "Magnesium"),
    (13, "Al", "Aluminum"), (14, "Si", "Silicon"), (15, "P", "Phosphorus"),
    (16, "S", "Sulfur"), (17, "Cl", "Chlorine"), (18, "Ar", "Argon"),
    (19, "K", "Potassium"), (20, "Ca", "Calcium"), (21, "Sc", "Scandium"),
    (22, "Ti", "Titanium"), (23, "V", "Vanadium"), (24, "Cr", "Chromium"),
    (25, "Mn", "Manganese"), (26, "Fe", "Iron"), (27, "Co", "Cobalt"),
    (28, "Ni", "Nickel"), (29, "Cu", "Copper"), (30, "Zn", "Zinc"),
    (31, "Ga", "Gallium"), (32, "Ge", "Germanium"), (33, "As", "Arsenic"),
    (34, "Se", "Selenium"), (35, "Br", "Bromine"), (36, "Kr", "Krypton"),
    (37, "Rb", "Rubidium"), (38, "Sr", "Strontium"), (39, "Y", "Yttrium"),
    (40, "Zr", "Zirconium"), (41, "Nb", "Niobium"), (42, "Mo", "Molybdenum"),
    (43, "Tc", "Technetium"), (44, "Ru", "Ruthenium"), (45, "Rh", "Rhodium"),
    (46, "Pd", "Palladium"), (47, "Ag", "Silver"), (48, "Cd", "Cadmium"),
    (49, "In", "Indium"), (50, "Sn", "Tin"), (51, "Sb", "Antimony"),
    (52, "Te", "Tellurium"), (53, "I", "Iodine"), (54, "Xe", "Xenon"),
    (55, "Cs", "Cesium"), (56, "Ba", "Barium"), (57, "La", "Lanthanum"),
    (58, "Ce", "Cerium"), (59, "Pr", "Praseodymium"), (60, "Nd", "Neodymium"),
    (61, "Pm", "Promethium"), (62, "Sm", "Samarium"), (63, "Eu", "Europium"),
    (64, "Gd", "Gadolinium"), (65, "Tb", "Terbium"), (66, "Dy", "Dysprosium"),
    (67, "Ho", "Holmium"), (68, "Er", "Erbium"), (69, "Tm", "Thulium"),
    (70, "Yb", "Ytterbium"), (71, "Lu", "Lutetium"), (72, "Hf", "Hafnium"),
    (73, "Ta", "Tantalum"), (74, "W", "Tungsten"), (75, "Re", "Rhenium"),
    (76, "Os", "Osmium"), (77, "Ir", "Iridium"), (78, "Pt", "Platinum"),
    (79, "Au", "Gold"), (80, "Hg", "Mercury"), (81, "Tl", "Thallium"),
    (82, "Pb", "Lead"), (83, "Bi", "Bismuth"), (84, "Po", "Polonium"),
    (85, "At", "Astatine"), (86, "Rn", "Radon"), (87, "Fr", "Francium"),
    (88, "Ra", "Radium"), (89, "Ac", "Actinium"), (90, "Th", "Thorium"),
    (91, "Pa", "Protactinium"), (92, "U", "Uranium"),
]

ATOM_BY_Z = {z: (symbol, name) for z, symbol, name in ATOMS}


def atom_label(z: int) -> str:
    """Return the display label used by atom dropdowns."""
    symbol, name = ATOM_BY_Z[int(z)]
    return f"{int(z):02d} {symbol} - {name}"


def html_table(headers: list[str], rows: list[list[Any]]) -> HTML:
    """Build a small escaped HTML table for widget outputs."""
    def esc(value: Any) -> str:
        text = str(value)
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    style = """
    <style>
    .atom-dft-table { border-collapse: collapse; font-size: 13px; width: 100%; }
    .atom-dft-table th, .atom-dft-table td { border: 1px solid #d8dee4; padding: 6px 8px; }
    .atom-dft-table th { background: #f6f8fa; text-align: left; }
    .atom-dft-note { color: #57606a; font-size: 13px; }
    </style>
    """
    return HTML(
        style
        + f"<table class='atom-dft-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def shell_counts(z: int) -> list[int]:
    """Approximate shell populations for the lightweight atom preview."""
    capacities = [2, 8, 18, 32, 32]
    remaining = int(z)
    counts: list[int] = []
    for capacity in capacities:
        if remaining <= 0:
            break
        take = min(capacity, remaining)
        counts.append(take)
        remaining -= take
    if remaining > 0:
        counts.append(remaining)
    return counts


class AtomDFTDashboard:
    """Selection-driven Jupyter UI for running ATOM calculations.

    Jobs are stored as snapshots of widget state so queued runs and exports
    are reproducible even after the user changes the visible controls.
    """

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = self._resolve_project_root(project_root)
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Runtime state intentionally stays outside ipywidgets. This keeps
        # result/export metadata tied to the completed job, not current inputs.
        self.state: dict[str, Any] = {
            "last_solver": None,
            "last_result": None,
            "last_descriptor": None,
            "last_descriptor_calculator": None,
            "last_descriptor_metadata": None,
            "queue": [],
            "batch_results": [],
        }

        self._build_widgets()
        self._wire_events()
        self.layout = self._build_layout()
        self._sync_electron_controls()
        self._update_descriptor_warning()
        self._redraw_atom_model()
        self._render_queue()

    @staticmethod
    def _resolve_project_root(project_root: str | Path | None) -> Path:
        """Find the repository root from a notebook or package context."""
        if project_root is not None:
            return Path(project_root).expanduser().resolve()

        cwd = Path.cwd().resolve()
        if (cwd / "atom").exists():
            return cwd
        for candidate in cwd.parents:
            if (candidate / "atom").exists():
                return candidate

        package_root = Path(__file__).resolve().parents[2]
        if (package_root / "atom").exists():
            return package_root
        return cwd

    def _read_available_psp_atomic_numbers(self) -> set[int]:
        """Return atomic numbers with a matching psps/XX.psp8 file."""
        psp_dir = self.project_root / "psps"
        available: set[int] = set()
        if not psp_dir.exists():
            return available
        for path in psp_dir.glob("*.psp8"):
            try:
                available.add(int(path.stem))
            except ValueError:
                continue
        return available

    def display(self) -> None:
        """Display the dashboard in a notebook cell."""
        display(self.layout)

    def _build_widgets(self) -> None:
        """Create widgets once; callbacks mutate their values and disabled state."""
        self.available_psp_atomic_numbers = self._read_available_psp_atomic_numbers()
        atom_options = [(atom_label(z), z) for z, _, _ in ATOMS]
        functional_options = list(VALID_XC_FUNCTIONAL_LIST)
        rcut_options = [
            ("0.5 bohr", 0.5), ("1.0 bohr", 1.0), ("1.5 bohr", 1.5),
            ("2.0 bohr", 2.0), ("3.0 bohr", 3.0), ("4.0 bohr", 4.0),
            ("5.0 bohr", 5.0), ("6.0 bohr", 6.0),
        ]

        self.atom_select = widgets.Dropdown(
            options=atom_options,
            value=1,
            description="Atom",
            layout=widgets.Layout(width="380px"),
        )
        self.n_electrons = widgets.BoundedFloatText(
            value=1.0,
            min=0.01,
            max=92.0,
            step=1.0,
            description="Electrons",
            layout=widgets.Layout(width="180px"),
            disabled=True,
        )
        self.all_electron = widgets.Checkbox(value=True, description="All-electron")
        self.custom_electrons = widgets.Checkbox(value=False, description="Custom electrons")
        self.charge_indicator = widgets.HTML()
        self.psp_indicator = widgets.HTML()
        self.xc_functional = widgets.Dropdown(
            options=functional_options,
            value="GGA_PBE",
            description="XC",
            layout=widgets.Layout(width="230px"),
        )
        self.use_oep = widgets.Checkbox(value=False, description="OEP for PBE0")

        self.preset = widgets.ToggleButtons(
            options=["Fast", "Balanced", "Accurate"],
            value="Fast",
            description="Preset",
            button_style="",
        )
        self.domain_size = widgets.BoundedFloatText(
            value=12.0, min=2.0, max=80.0, step=1.0, description="Domain",
            layout=widgets.Layout(width="170px"),
        )
        self.finite_elements = widgets.BoundedIntText(
            value=6, min=2, max=80, description="FE",
            layout=widgets.Layout(width="130px"),
        )
        self.polynomial_order = widgets.BoundedIntText(
            value=10, min=4, max=80, description="Order",
            layout=widgets.Layout(width="150px"),
        )
        self.quadrature_points = widgets.BoundedIntText(
            value=25, min=8, max=160, description="Quad",
            layout=widgets.Layout(width="150px"),
        )
        self.scf_tolerance = widgets.FloatLogSlider(
            value=1e-6, base=10, min=-10, max=-4, step=1, description="SCF tol",
            readout_format=".0e", layout=widgets.Layout(width="280px"),
        )
        self.max_scf_iterations = widgets.BoundedIntText(
            value=80, min=5, max=2000, description="Max iter",
            layout=widgets.Layout(width="170px"),
        )
        self.use_preconditioner = widgets.Checkbox(value=False, description="Preconditioner")
        self.verbose_solver = widgets.Checkbox(value=False, description="Verbose solver")
        self.save_intermediate = widgets.Checkbox(value=True, description="Convergence history")
        self.save_full_spectrum = widgets.Checkbox(value=False, description="Full spectrum")

        self.include_descriptors = widgets.Checkbox(value=False, description="Compute with SCF")
        self.descriptor_basis = widgets.Dropdown(
            options=["heaviside", "legendre"],
            value="heaviside",
            description="Radial basis",
            layout=widgets.Layout(width="220px"),
        )
        self.descriptor_order = widgets.IntSlider(
            value=0, min=0, max=4, step=1, description="Radial order",
            layout=widgets.Layout(width="260px"),
        )
        self.descriptor_rcuts = widgets.SelectMultiple(
            options=rcut_options,
            value=(0.5, 1.0, 1.5),
            description="Rcuts",
            rows=6,
            layout=widgets.Layout(width="220px"),
        )
        self.descriptor_lmax = widgets.ToggleButtons(
            options=[("l<=0", 0), ("l<=1", 1), ("l<=2", 2)],
            value=2,
            description="Angular l",
        )
        self.descriptor_warning = widgets.HTML()
        self.descriptor_box = widgets.BoundedFloatText(
            value=8.0, min=2.0, max=80.0, step=1.0, description="Box",
            layout=widgets.Layout(width="150px"),
        )
        self.descriptor_spacing = widgets.BoundedFloatText(
            value=0.8, min=0.1, max=3.0, step=0.1, description="Spacing",
            layout=widgets.Layout(width="170px"),
        )
        self.descriptor_periodic = widgets.Checkbox(value=True, description="Periodic grid")

        self.batch_atoms = widgets.SelectMultiple(
            options=atom_options,
            value=(1,),
            description="Atoms",
            rows=10,
            layout=widgets.Layout(width="380px"),
        )
        self.batch_functionals = widgets.SelectMultiple(
            options=functional_options,
            value=("GGA_PBE",),
            description="XC",
            rows=10,
            layout=widgets.Layout(width="220px"),
        )

        self.run_button = widgets.Button(
            description="Run selected atom",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="180px"),
        )
        self.posthoc_descriptor_button = widgets.Button(
            description="Compute descriptors",
            icon="cubes",
            layout=widgets.Layout(width="190px"),
        )
        self.add_job_button = widgets.Button(
            description="Add current job",
            icon="plus",
            layout=widgets.Layout(width="170px"),
        )
        self.add_batch_matrix_button = widgets.Button(
            description="Add selected batch",
            icon="th",
            layout=widgets.Layout(width="190px"),
        )
        self.run_queue_button = widgets.Button(
            description="Run queue",
            button_style="primary",
            icon="tasks",
            layout=widgets.Layout(width="150px"),
        )
        self.clear_queue_button = widgets.Button(
            description="Clear queue",
            icon="trash",
            layout=widgets.Layout(width="150px"),
        )
        self.export_descriptor_button = widgets.Button(
            description="Export descriptor NPZ",
            icon="download",
            layout=widgets.Layout(width="200px"),
        )

        self.progress = widgets.IntProgress(value=0, min=0, max=1, description="Progress")
        self.status = widgets.HTML("<b>Ready.</b>")
        self.angle = widgets.IntSlider(value=35, min=0, max=360, step=5, description="Angle")
        self.play = widgets.Play(value=35, min=0, max=360, step=5, interval=140, description="Rotate")
        widgets.jslink((self.play, "value"), (self.angle, "value"))

        self.log_output = widgets.Output()
        self.plot_output = widgets.Output()
        self.descriptor_output = widgets.Output()
        self.queue_output = widgets.Output()
        self.model_output = widgets.Output()

    def _wire_events(self) -> None:
        """Connect widget changes and buttons to dashboard actions."""
        self.preset.observe(self._apply_preset, names="value")
        self.atom_select.observe(self._on_atom_change, names="value")
        self.all_electron.observe(self._sync_electron_controls, names="value")
        self.custom_electrons.observe(self._sync_electron_controls, names="value")
        self.n_electrons.observe(self._update_charge_indicator, names="value")
        self.include_descriptors.observe(self._update_descriptor_warning, names="value")
        self.descriptor_rcuts.observe(self._update_descriptor_warning, names="value")
        self.descriptor_box.observe(self._update_descriptor_warning, names="value")
        self.descriptor_basis.observe(self._update_descriptor_warning, names="value")
        self.descriptor_order.observe(self._update_descriptor_warning, names="value")
        self.descriptor_lmax.observe(self._update_descriptor_warning, names="value")
        self.descriptor_spacing.observe(self._update_descriptor_warning, names="value")
        self.angle.observe(self._redraw_atom_model, names="value")
        self.run_button.on_click(self._run_single)
        self.posthoc_descriptor_button.on_click(self._compute_descriptors_from_last)
        self.add_job_button.on_click(self._add_current_job)
        self.add_batch_matrix_button.on_click(self._add_batch_matrix)
        self.run_queue_button.on_click(self._run_queue)
        self.clear_queue_button.on_click(self._clear_queue)
        self.export_descriptor_button.on_click(self._export_last_descriptor)

    def _build_layout(self) -> widgets.Widget:
        """Assemble the notebook tabs from the widgets created earlier."""
        header = widgets.HTML(
            """
            <h2 style="margin-bottom: 0.2rem">ATOM DFT Dashboard</h2>
            <div class="atom-dft-note">
            Select an atom and task, then use the buttons. No Python editing required.
            </div>
            """
        )

        advanced = widgets.Accordion(children=[
            widgets.VBox([
                widgets.HBox([
                    self.preset, self.domain_size, self.finite_elements,
                    self.polynomial_order, self.quadrature_points,
                ]),
                widgets.HBox([
                    self.scf_tolerance, self.max_scf_iterations, self.use_preconditioner,
                ]),
                widgets.HBox([
                    self.verbose_solver, self.save_intermediate, self.save_full_spectrum,
                ]),
            ])
        ])
        advanced.set_title(0, "Advanced SCF settings")

        setup = widgets.VBox([
            header,
            widgets.HBox([self.atom_select, self.all_electron, self.custom_electrons, self.n_electrons]),
            self.charge_indicator,
            self.psp_indicator,
            widgets.HBox([self.xc_functional, self.use_oep]),
            advanced,
            widgets.HBox([self.run_button, self.add_job_button, self.run_queue_button, self.clear_queue_button]),
            widgets.HBox([self.progress, self.status]),
            self.log_output,
        ])

        model = widgets.VBox([
            widgets.HTML("<b>Atom preview</b>"),
            widgets.HBox([self.play, self.angle]),
            self.model_output,
        ])

        descriptors = widgets.VBox([
            widgets.HTML("<b>MCSH / multipole descriptors</b>"),
            widgets.HBox([self.include_descriptors, self.descriptor_basis, self.descriptor_order]),
            widgets.HBox([self.descriptor_rcuts, widgets.VBox([
                self.descriptor_lmax,
                self.descriptor_box,
                self.descriptor_spacing,
                self.descriptor_periodic,
                self.descriptor_warning,
                widgets.HBox([self.posthoc_descriptor_button, self.export_descriptor_button]),
            ])]),
            self.descriptor_output,
        ])

        batch = widgets.VBox([
            widgets.HTML("<b>Batch builder</b>"),
            widgets.HBox([self.batch_atoms, self.batch_functionals]),
            widgets.HBox([self.add_batch_matrix_button, self.run_queue_button, self.clear_queue_button]),
            self.queue_output,
        ])

        tabs = widgets.Tab(children=[
            setup,
            model,
            widgets.VBox([self.plot_output]),
            descriptors,
            batch,
        ])
        for index, title in enumerate(["Setup", "Atom model", "SCF plots", "Descriptors", "Batch"]):
            tabs.set_title(index, title)
        return tabs

    def _apply_preset(self, change: dict[str, Any] | None = None) -> None:
        """Apply coarse mesh/descriptor presets to the visible controls."""
        presets = {
            "Fast": dict(domain=12.0, fe=6, order=10, quad=25, tol=1e-6, max_iter=80, spacing=0.8, box=8.0),
            "Balanced": dict(domain=16.0, fe=10, order=18, quad=43, tol=1e-7, max_iter=180, spacing=0.5, box=12.0),
            "Accurate": dict(domain=20.0, fe=17, order=31, quad=95, tol=1e-8, max_iter=500, spacing=0.3, box=20.0),
        }
        values = presets[self.preset.value]
        self.domain_size.value = values["domain"]
        self.finite_elements.value = values["fe"]
        self.polynomial_order.value = values["order"]
        self.quadrature_points.value = values["quad"]
        self.scf_tolerance.value = values["tol"]
        self.max_scf_iterations.value = values["max_iter"]
        self.descriptor_spacing.value = values["spacing"]
        self.descriptor_box.value = values["box"]

    def _on_atom_change(self, change: dict[str, Any] | None = None) -> None:
        self._sync_electron_controls()
        self._redraw_atom_model()

    def _sync_electron_controls(self, change: dict[str, Any] | None = None) -> None:
        """Keep electron count consistent with AE/PSP mode."""
        z = int(self.atom_select.value)
        if not self.all_electron.value:
            self.custom_electrons.value = False
            self.custom_electrons.disabled = True
            self.n_electrons.disabled = True
            self.n_electrons.value = float(z)
        else:
            self.custom_electrons.disabled = False
            self.n_electrons.disabled = not self.custom_electrons.value
            if not self.custom_electrons.value:
                self.n_electrons.value = float(z)
        self._update_charge_indicator()
        self._update_psp_indicator()
        self._refresh_batch_atom_options()
        self._refresh_action_availability()

    def _update_charge_indicator(self, change: dict[str, Any] | None = None) -> None:
        """Explain the current electron count without changing physics."""
        z = int(self.atom_select.value)
        electrons = float(self.n_electrons.value)
        charge = z - electrons
        if abs(charge) < 1e-12:
            charge_label = "neutral atom"
        elif charge > 0:
            charge_label = f"cation, charge +{charge:g}"
        else:
            charge_label = f"anion, charge {charge:g}"

        if not self.all_electron.value:
            detail = "PSP mode uses neutral valence setups, so electron count is locked."
        elif self.custom_electrons.value:
            detail = "Custom all-electron ion mode is enabled."
        else:
            detail = "Neutral all-electron atom; electron count follows the selected atom."

        self.charge_indicator.value = (
            f"<span class='atom-dft-note'><b>Electron setup:</b> "
            f"Z={z}, N={electrons:g} ({charge_label}). {detail}</span>"
        )

    def _update_psp_indicator(self) -> None:
        """Show whether the selected atom has a usable PSP file."""
        z = int(self.atom_select.value)
        if self.all_electron.value:
            self.psp_indicator.value = ""
            return

        if z in self.available_psp_atomic_numbers:
            self.psp_indicator.value = (
                "<span class='atom-dft-note'><b>PSP:</b> "
                f"psps/{z:02d}.psp8 is available.</span>"
            )
        else:
            self.psp_indicator.value = (
                "<span style='color:#b00020; font-size:13px'><b>PSP unavailable:</b> "
                f"No psps/{z:02d}.psp8 file exists. Switch to all-electron or select an atom with a PSP file.</span>"
            )

    def _refresh_batch_atom_options(self) -> None:
        """Restrict batch atom choices when PSP mode cannot run every element."""
        if self.all_electron.value:
            allowed = [z for z, _, _ in ATOMS]
        else:
            allowed = [z for z, _, _ in ATOMS if z in self.available_psp_atomic_numbers]
        options = [(atom_label(z), z) for z in allowed]
        current = tuple(z for z in self.batch_atoms.value if z in allowed)
        if not current and allowed:
            current = (allowed[0],)
        if any(z not in allowed for z in self.batch_atoms.value):
            self.batch_atoms.value = ()
        self.batch_atoms.options = options
        self.batch_atoms.value = current

    def _descriptor_config_from_widgets(self) -> dict[str, Any]:
        """Snapshot the current descriptor controls into a serializable config."""
        return {
            "angular_basis": "mcsh",
            "radial_basis": self.descriptor_basis.value,
            "radial_order": int(self.descriptor_order.value),
            "rcuts": tuple(float(value) for value in self.descriptor_rcuts.value),
            "l_max": int(self.descriptor_lmax.value),
            "box_size": float(self.descriptor_box.value),
            "spacing": float(self.descriptor_spacing.value),
            "periodic": bool(self.descriptor_periodic.value),
        }

    def _descriptor_config_error(self, config: dict[str, Any] | None = None) -> str | None:
        """Return a user-facing descriptor validation error, if any."""
        config = self._descriptor_config_from_widgets() if config is None else config
        rcuts = list(config["rcuts"])
        if not rcuts:
            return "Select at least one descriptor cutoff radius."
        max_rcut = max(rcuts)
        half_box = float(config["box_size"]) / 2.0
        if max_rcut > half_box:
            return (
                f"Largest rcut ({max_rcut:g}) exceeds half the descriptor box ({half_box:g}). "
                "Increase Box or select smaller Rcuts."
            )
        return None

    def _update_descriptor_warning(self, change: dict[str, Any] | None = None) -> None:
        """Refresh descriptor warnings and buttons after descriptor edits."""
        config = self._descriptor_config_from_widgets()
        error = self._descriptor_config_error(config)
        if error is not None:
            self.descriptor_warning.value = (
                f"<span style='color:#b00020; font-size:13px'><b>Descriptor warning:</b> {error}</span>"
            )
            self.posthoc_descriptor_button.disabled = True
        elif config["radial_basis"] == "legendre":
            self.descriptor_warning.value = (
                "<span class='atom-dft-note'><b>Note:</b> Legendre radial order selects one P_n kernel, "
                "not a cumulative P_0...P_n stack.</span>"
            )
            self.posthoc_descriptor_button.disabled = False
        else:
            self.descriptor_warning.value = ""
            self.posthoc_descriptor_button.disabled = False
        self._refresh_action_availability()

    def _refresh_action_availability(self) -> None:
        """Disable actions that would immediately fail validation."""
        psp_unavailable = (
            not self.all_electron.value
            and int(self.atom_select.value) not in self.available_psp_atomic_numbers
        )
        descriptor_unavailable = (
            bool(self.include_descriptors.value)
            and self._descriptor_config_error() is not None
        )
        disable_current_job = psp_unavailable or descriptor_unavailable
        self.run_button.disabled = disable_current_job
        self.add_job_button.disabled = disable_current_job
        self.add_batch_matrix_button.disabled = descriptor_unavailable

    def _make_descriptor_calculator(
        self,
        name: str = "multipole",
        config: dict[str, Any] | None = None,
    ) -> MultipoleCalculator:
        """Build a MultipoleCalculator from a saved config or live widgets."""
        config = self._descriptor_config_from_widgets() if config is None else config
        rcuts = list(config["rcuts"])
        if not rcuts:
            raise ValueError("Select at least one descriptor cutoff radius.")
        return MultipoleCalculator(
            angular_basis=config["angular_basis"],
            radial_basis=config["radial_basis"],
            radial_order=int(config["radial_order"]),
            rcuts=rcuts,
            l_max=int(config["l_max"]),
            box_size=float(config["box_size"]),
            spacing=float(config["spacing"]),
            periodic=bool(config["periodic"]),
            name=name,
        )

    def _collect_single_job(self, include_inline_descriptors: bool | None = None) -> dict[str, Any]:
        """Collect a job from the Setup tab controls."""
        electrons = (
            float(self.n_electrons.value)
            if self.all_electron.value and self.custom_electrons.value
            else float(self.atom_select.value)
        )
        return self._collect_job(
            atomic_number=int(self.atom_select.value),
            xc_functional=self.xc_functional.value,
            n_electrons=electrons,
            include_inline_descriptors=include_inline_descriptors,
        )

    def _collect_job(
        self,
        atomic_number: int,
        xc_functional: str,
        n_electrons: float | None = None,
        include_inline_descriptors: bool | None = None,
    ) -> dict[str, Any]:
        """Create the immutable job dictionary consumed by the solver."""
        z = int(atomic_number)
        electrons = float(z if n_electrons is None else n_electrons)
        if not self.all_electron.value:
            electrons = float(z)

        if xc_functional in ("EXX", "RPA"):
            oep_flag = True
        elif xc_functional == "PBE0":
            oep_flag = bool(self.use_oep.value)
        else:
            oep_flag = False

        descriptor_requested = (
            bool(self.include_descriptors.value)
            if include_inline_descriptors is None
            else bool(include_inline_descriptors)
        )
        if not self.all_electron.value and z not in self.available_psp_atomic_numbers:
            raise ValueError(
                f"No PSP file is available for Z={z}. Switch to all-electron mode "
                f"or choose an atom with a psps/{z:02d}.psp8 file."
            )
        descriptor_config = self._descriptor_config_from_widgets()
        if descriptor_requested:
            descriptor_error = self._descriptor_config_error(descriptor_config)
            if descriptor_error is not None:
                raise ValueError(descriptor_error)
        mode = "AE" if self.all_electron.value else "PSP"
        symbol = ATOM_BY_Z[z][0]
        atom_name = ATOM_BY_Z[z][1]
        return {
            "atomic_number": z,
            "symbol": symbol,
            "atom_name": atom_name,
            "n_electrons": electrons,
            "all_electron_flag": bool(self.all_electron.value),
            "mode": mode,
            "xc_functional": xc_functional,
            "use_oep": oep_flag,
            "domain_size": float(self.domain_size.value),
            "finite_element_number": int(self.finite_elements.value),
            "polynomial_order": int(self.polynomial_order.value),
            "quadrature_point_number": int(self.quadrature_points.value),
            "scf_tolerance": float(self.scf_tolerance.value),
            "max_scf_iterations": int(self.max_scf_iterations.value),
            "use_preconditioner": bool(self.use_preconditioner.value),
            "verbose": bool(self.verbose_solver.value),
            "save_intermediate": bool(self.save_intermediate.value),
            "save_full_spectrum": bool(self.save_full_spectrum.value),
            "descriptors": descriptor_requested,
            "descriptor_config": descriptor_config,
            "label": f"{symbol} {xc_functional} {mode}",
        }

    def _solver_kwargs_from_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Translate a queued job snapshot into AtomicDFTSolver kwargs."""
        keys = [
            "atomic_number", "n_electrons", "all_electron_flag", "xc_functional",
            "use_oep", "domain_size", "finite_element_number", "polynomial_order",
            "quadrature_point_number", "scf_tolerance", "max_scf_iterations",
            "use_preconditioner", "verbose",
        ]
        kwargs = {key: job[key] for key in keys}
        if job["descriptors"]:
            descriptor_config = job.get("descriptor_config")
            descriptor_error = self._descriptor_config_error(descriptor_config)
            if descriptor_error is not None:
                raise ValueError(descriptor_error)
            kwargs["descriptor_calculators"] = [
                self._make_descriptor_calculator(config=descriptor_config)
            ]
        return kwargs

    def _metadata_from_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Return export/result metadata that describes the completed job."""
        return {
            "atomic_number": int(job["atomic_number"]),
            "symbol": job.get("symbol") or ATOM_BY_Z[int(job["atomic_number"])][0],
            "atom_name": job.get("atom_name") or ATOM_BY_Z[int(job["atomic_number"])][1],
            "xc_functional": job["xc_functional"],
            "xc": job["xc_functional"],
            "mode": job.get("mode") or ("AE" if job["all_electron_flag"] else "PSP"),
            "all_electron_flag": bool(job["all_electron_flag"]),
            "n_electrons": float(job["n_electrons"]),
            "use_oep": bool(job["use_oep"]),
            "descriptors_requested": bool(job["descriptors"]),
            "descriptor_config": copy.deepcopy(job.get("descriptor_config")),
            "label": job.get("label"),
        }

    def _attach_result_metadata(self, result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        """Store dashboard metadata inside a solver result."""
        metadata = self._metadata_from_job(job)
        result["dashboard_metadata"] = metadata
        return metadata

    def _descriptor_name_from_result(self, result: dict[str, Any]) -> str | None:
        """Pick the descriptor result key to plot/export."""
        descriptor_results = result.get("descriptor_results") or {}
        if "multipole" in descriptor_results:
            return "multipole"
        if descriptor_results:
            return next(iter(descriptor_results))
        return None

    def _set_last_run_state(
        self,
        solver: AtomicDFTSolver,
        result: dict[str, Any],
        *,
        clear_descriptor_when_missing: bool = True,
    ) -> None:
        """Record the latest solver/result and any descriptor it produced."""
        self.state["last_solver"] = solver
        self.state["last_result"] = result

        descriptor_name = self._descriptor_name_from_result(result)
        if descriptor_name is None:
            if clear_descriptor_when_missing:
                self.state["last_descriptor"] = None
                self.state["last_descriptor_calculator"] = None
                self.state["last_descriptor_metadata"] = None
            return

        descriptor_result = result["descriptor_results"][descriptor_name]
        metadata = copy.deepcopy(result.get("dashboard_metadata") or {})
        metadata["descriptor_result_name"] = descriptor_name
        metadata["descriptor_source"] = "scf"

        self.state["last_descriptor"] = descriptor_result
        self.state["last_descriptor_metadata"] = metadata
        self.state["last_descriptor_calculator"] = self._make_descriptor_calculator(
            name=descriptor_name,
            config=metadata.get("descriptor_config"),
        )

    def _run_job(self, job: dict[str, Any]) -> tuple[AtomicDFTSolver, dict[str, Any], str]:
        """Run one job and capture solver stdout/stderr for the log panel."""
        captured = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            solver = AtomicDFTSolver(**self._solver_kwargs_from_job(job))
            result = solver.solve(
                save_intermediate=job["save_intermediate"],
                save_full_spectrum=job["save_full_spectrum"],
            )
        result["wall_time_seconds"] = time.perf_counter() - start
        self._attach_result_metadata(result, job)
        return solver, result, captured.getvalue()

    def _run_single(self, _: Any = None) -> None:
        """Button callback: run the currently selected atom."""
        job = self._collect_single_job()
        self.progress.max = 1
        self.progress.value = 0
        self.progress.bar_style = "info"
        self.status.value = f"<b>Running:</b> {job['label']}"
        with self.log_output:
            clear_output(wait=True)
            print(f"Starting {job['label']} at {time.strftime('%H:%M:%S')}")

        try:
            solver, result, captured = self._run_job(job)
            self._set_last_run_state(solver, result)

            self.progress.value = 1
            self.progress.bar_style = "success" if result.get("converged") else "warning"
            self.status.value = (
                f"<b>Done:</b> {job['label']} | converged={result.get('converged')} "
                f"| E={result.get('energy'):.8f} Ha"
            )
            with self.log_output:
                clear_output(wait=True)
                display(html_table(["Quantity", "Value"], self._result_summary_rows(result)))
                if captured.strip():
                    print("\nSolver output:\n")
                    print(captured.strip())
            self._plot_result(result)
            self._plot_descriptor(result)
        except Exception:
            self.progress.bar_style = "danger"
            self.status.value = "<b>Run failed.</b> See log output."
            with self.log_output:
                clear_output(wait=True)
                traceback.print_exc()

    def _result_summary_rows(self, result: dict[str, Any]) -> list[list[Any]]:
        """Format scalar result fields for the Setup tab summary table."""
        rows = [
            ["Converged", result.get("converged")],
            ["Iterations", result.get("iterations")],
            ["Final residual", f"{result.get('rho_residual', float('nan')):.3e}"],
            ["Energy (Ha)", f"{result.get('energy', float('nan')):.10f}"],
            ["Wall time (s)", f"{result.get('wall_time_seconds', float('nan')):.2f}"],
            ["Orbitals shape", result.get("orbitals").shape if result.get("orbitals") is not None else None],
            ["Density points", len(result.get("rho", []))],
        ]
        descriptor_results = result.get("descriptor_results") or {}
        if descriptor_results:
            rows.append(["Descriptors", ", ".join(descriptor_results.keys())])
        return rows

    def _residual_plot_data(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Extract residual history for inner-loop or outer-loop SCF runs."""
        info = result.get("intermediate_info")
        if info is None:
            return None

        inner_iterations = list(getattr(info, "inner_iterations", []) or [])
        if inner_iterations:
            return {
                "title": "SCF residual",
                "xlabel": "Iteration",
                "ylabel": "Density residual",
                "x": [it.inner_iteration for it in inner_iterations],
                "y": [it.rho_residual for it in inner_iterations],
            }

        outer_iterations = list(getattr(info, "outer_iterations", []) or [])
        if outer_iterations:
            outer_points = [
                it for it in outer_iterations
                if hasattr(it, "outer_iteration") and hasattr(it, "outer_rho_residual")
            ]
            if outer_points:
                return {
                    "title": "Outer SCF residual",
                    "xlabel": "Outer iteration",
                    "ylabel": "Density residual",
                    "x": [it.outer_iteration for it in outer_points],
                    "y": [it.outer_rho_residual for it in outer_points],
                }

            nested_inner = [
                inner
                for outer in outer_iterations
                for inner in (getattr(outer, "inner_iterations", []) or [])
            ]
            if nested_inner:
                return {
                    "title": "Nested SCF residual",
                    "xlabel": "Inner iteration (cumulative)",
                    "ylabel": "Density residual",
                    "x": list(range(1, len(nested_inner) + 1)),
                    "y": [it.rho_residual for it in nested_inner],
                }

        return None

    def _plot_result(self, result: dict[str, Any]) -> None:
        """Render density, radial distribution, and convergence plots."""
        with self.plot_output:
            clear_output(wait=True)
            r = result["quadrature_nodes"]
            rho = result["rho"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
            axes[0].plot(r, rho, color="#264653")
            axes[0].set_title("Electron density")
            axes[0].set_xlabel("r (Bohr)")
            axes[0].set_ylabel("rho(r)")

            axes[1].plot(r, 4.0 * np.pi * r**2 * rho, color="#2a9d8f")
            axes[1].set_title("Radial distribution")
            axes[1].set_xlabel("r (Bohr)")
            axes[1].set_ylabel("4*pi*r^2*rho")

            residual_data = self._residual_plot_data(result)
            if residual_data is not None:
                axes[2].semilogy(
                    residual_data["x"],
                    residual_data["y"],
                    marker="o",
                    color="#e76f51",
                )
                axes[2].set_title(residual_data["title"])
                axes[2].set_xlabel(residual_data["xlabel"])
                axes[2].set_ylabel(residual_data["ylabel"])
            else:
                axes[2].axis("off")
                message = (
                    "Enable convergence history\nto plot residuals."
                    if result.get("intermediate_info") is None
                    else "No residual samples found\nin convergence history."
                )
                axes[2].text(
                    0.5, 0.5, message,
                    ha="center", va="center",
                )
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def _plot_descriptor(
        self,
        result: dict[str, Any],
        descriptor_result: Any | None = None,
        calculator: MultipoleCalculator | None = None,
    ) -> None:
        """Render a compact view of the latest multipole descriptor."""
        if descriptor_result is None:
            descriptor_result = result.get("descriptor_results", {}).get("multipole")
        if descriptor_result is None:
            with self.descriptor_output:
                clear_output(wait=True)
                print("No descriptor result yet. Compute descriptors with SCF or post-process the last result.")
            return
        if calculator is None:
            calculator = self.state.get("last_descriptor_calculator") or self._make_descriptor_calculator()

        profile = calculator.extract_radial_profile(descriptor_result)
        center_index = int(np.argmin(profile["r"]))
        center_values = np.abs(profile["descriptors"][center_index])
        rcuts = np.array(profile["rcuts"], dtype=float)

        with self.descriptor_output:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for l_value in range(center_values.shape[1]):
                axes[0].plot(rcuts, center_values[:, l_value], marker="o", label=f"l={l_value}")
            axes[0].set_title("Center descriptor vs cutoff")
            axes[0].set_xlabel("Rcut (Bohr)")
            axes[0].set_ylabel("abs(descriptor)")
            axes[0].legend()

            axes[1].imshow(center_values.T, aspect="auto", origin="lower", cmap="viridis")
            axes[1].set_title("Center descriptor heatmap")
            axes[1].set_xlabel("Rcut index")
            axes[1].set_ylabel("l")
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            print(f"Descriptor tensor shape: {descriptor_result.descriptors.shape}")

    def _compute_descriptors_from_last(self, _: Any = None) -> None:
        """Button callback: compute descriptors from the latest SCF density."""
        result = self.state.get("last_result")
        if result is None:
            with self.descriptor_output:
                clear_output(wait=True)
                print("Run SCF first, then compute descriptors from the converged density.")
            return
        self.status.value = "<b>Computing descriptors from last SCF result...</b>"
        try:
            descriptor_config = self._descriptor_config_from_widgets()
            descriptor_error = self._descriptor_config_error(descriptor_config)
            if descriptor_error is not None:
                raise ValueError(descriptor_error)
            calculator = self._make_descriptor_calculator(
                name="multipole_posthoc",
                config=descriptor_config,
            )
            descriptor_result = calculator.compute_from_solver_result(result)
            metadata = copy.deepcopy(result.get("dashboard_metadata") or {})
            metadata.update({
                "descriptor_config": copy.deepcopy(descriptor_config),
                "descriptor_result_name": calculator.name,
                "descriptor_source": "posthoc",
            })
            self.state["last_descriptor"] = descriptor_result
            self.state["last_descriptor_calculator"] = calculator
            self.state["last_descriptor_metadata"] = metadata
            self.status.value = f"<b>Descriptors ready:</b> shape={descriptor_result.descriptors.shape}"
            self._plot_descriptor(result, descriptor_result, calculator)
        except Exception:
            self.status.value = "<b>Descriptor calculation failed.</b>"
            with self.descriptor_output:
                clear_output(wait=True)
                traceback.print_exc()

    def _add_current_job(self, _: Any = None) -> None:
        """Button callback: append the current Setup tab job to the queue."""
        try:
            self.state["queue"].append(self._collect_single_job())
            self.status.value = f"<b>Added job.</b> Queue length: {len(self.state['queue'])}"
            self._render_queue()
        except Exception:
            with self.queue_output:
                clear_output(wait=True)
                traceback.print_exc()

    def _add_batch_matrix(self, _: Any = None) -> None:
        """Button callback: add the selected atom/functionals matrix."""
        try:
            for z in self.batch_atoms.value:
                for functional in self.batch_functionals.value:
                    self.state["queue"].append(
                        self._collect_job(
                            atomic_number=int(z),
                            xc_functional=str(functional),
                            n_electrons=float(z),
                        )
                    )
            self.status.value = f"<b>Added selected batch.</b> Queue length: {len(self.state['queue'])}"
            self._render_queue()
        except Exception:
            with self.queue_output:
                clear_output(wait=True)
                traceback.print_exc()

    def _clear_queue(self, _: Any = None) -> None:
        """Button callback: remove all queued jobs."""
        self.state["queue"].clear()
        self.status.value = "<b>Queue cleared.</b>"
        self._render_queue()

    def _render_queue(self) -> None:
        """Refresh the queue table in the Batch tab."""
        with self.queue_output:
            clear_output(wait=True)
            if not self.state["queue"]:
                print("Queue is empty.")
                return
            rows = []
            for index, job in enumerate(self.state["queue"], start=1):
                descriptor_config = job.get("descriptor_config") or {}
                descriptor_label = "No"
                if job["descriptors"]:
                    rcuts = ", ".join(f"{rcut:g}" for rcut in descriptor_config.get("rcuts", ()))
                    box_size = descriptor_config.get("box_size")
                    box_label = f"{float(box_size):g}" if box_size is not None else "?"
                    descriptor_label = (
                        f"Yes ({descriptor_config.get('radial_basis')}, "
                        f"r=[{rcuts}], box={box_label})"
                    )
                rows.append([
                    index,
                    job["label"],
                    job["atomic_number"],
                    job["n_electrons"],
                    "AE" if job["all_electron_flag"] else "PSP",
                    descriptor_label,
                    f"{job['finite_element_number']}x{job['polynomial_order']} / q{job['quadrature_point_number']}",
                ])
            display(html_table(["#", "Job", "Z", "e-", "Mode", "Descriptors", "Mesh"], rows))

    def _run_queue(self, _: Any = None) -> None:
        """Button callback: run queued jobs in order."""
        jobs = list(self.state["queue"])
        if not jobs:
            self.status.value = "<b>Queue is empty.</b>"
            return

        self.progress.max = len(jobs)
        self.progress.value = 0
        self.progress.bar_style = "info"
        self.state["batch_results"].clear()
        self.state["last_descriptor"] = None
        self.state["last_descriptor_calculator"] = None
        self.state["last_descriptor_metadata"] = None
        failed_jobs = 0
        unconverged_jobs = 0
        rows = []
        for index, job in enumerate(jobs, start=1):
            self.status.value = f"<b>Running batch {index}/{len(jobs)}:</b> {job['label']}"
            try:
                solver, result, _captured = self._run_job(job)
                self._set_last_run_state(
                    solver,
                    result,
                    clear_descriptor_when_missing=False,
                )
                self.state["batch_results"].append((job, result))
                rows.append([
                    index,
                    job["label"],
                    result.get("converged"),
                    result.get("iterations"),
                    f"{result.get('rho_residual', float('nan')):.3e}",
                    f"{result.get('energy', float('nan')):.10f}",
                    f"{result.get('wall_time_seconds', float('nan')):.2f}",
                ])
                if not bool(result.get("converged", False)):
                    unconverged_jobs += 1
            except Exception as exc:
                failed_jobs += 1
                rows.append([index, job["label"], "FAILED", "", "", "", repr(exc)])
            self.progress.value = index
            with self.queue_output:
                clear_output(wait=True)
                display(html_table(
                    ["#", "Job", "Converged", "Iter", "Residual", "Energy (Ha)", "Seconds"],
                    rows,
                ))

        successful_jobs = len(jobs) - failed_jobs - unconverged_jobs
        if failed_jobs:
            self.progress.bar_style = "danger" if failed_jobs == len(jobs) else "warning"
        elif unconverged_jobs:
            self.progress.bar_style = "warning"
        else:
            self.progress.bar_style = "success"

        summary_parts = [f"{successful_jobs} succeeded"]
        if unconverged_jobs:
            summary_parts.append(f"{unconverged_jobs} unconverged")
        if failed_jobs:
            summary_parts.append(f"{failed_jobs} failed")
        self.status.value = f"<b>Batch finished:</b> {', '.join(summary_parts)}."
        if self.state.get("last_result") is not None:
            self._plot_result(self.state["last_result"])

    def _export_last_descriptor(self, _: Any = None) -> None:
        """Button callback: export the latest descriptor with saved metadata."""
        descriptor_result = self.state.get("last_descriptor")
        if descriptor_result is None:
            self.status.value = "<b>No descriptor to export.</b>"
            return
        output_dir = self.project_root / "outputs"
        output_dir.mkdir(exist_ok=True)
        metadata = copy.deepcopy(self.state.get("last_descriptor_metadata") or {})
        if not metadata and self.state.get("last_result") is not None:
            metadata = copy.deepcopy(
                self.state["last_result"].get("dashboard_metadata") or {}
            )

        z = metadata.get("atomic_number")
        symbol = metadata.get("symbol") or "atom"
        atom_part = f"{int(z):02d}_{symbol}" if z is not None else symbol
        filename_parts = [
            atom_part,
            metadata.get("xc_functional") or metadata.get("xc") or "xc",
            metadata.get("mode") or "mode",
            metadata.get("descriptor_result_name") or "multipole",
            "descriptor",
        ]
        filename = "_".join(self._safe_filename_part(part) for part in filename_parts) + ".npz"
        path = output_dir / filename
        self._write_descriptor_npz(path, descriptor_result, metadata)
        self.status.value = f"<b>Exported:</b> {path}"

    @staticmethod
    def _safe_filename_part(value: Any) -> str:
        """Make a metadata value safe for use in an output filename."""
        text = str(value).strip()
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "-", text).strip("-._")
        return safe or "unknown"

    @staticmethod
    def _json_ready(value: Any) -> Any:
        """Convert metadata values to JSON-compatible Python objects."""
        if isinstance(value, dict):
            return {str(key): AtomDFTDashboard._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AtomDFTDashboard._json_ready(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _write_descriptor_npz(self, path: Path, descriptor_result: Any, metadata: dict[str, Any]) -> None:
        """Write descriptor arrays plus dashboard metadata to an NPZ file."""
        np.savez_compressed(
            path,
            grid_indices=descriptor_result.grid_indices,
            grid_positions=descriptor_result.grid_positions,
            descriptors=descriptor_result.descriptors,
            rcuts=np.array(descriptor_result.rcuts),
            l_max=descriptor_result.l_max,
            spacing=np.array(descriptor_result.spacing),
            angular_basis=descriptor_result.angular_basis,
            radial_basis=descriptor_result.radial_basis,
            radial_order=descriptor_result.radial_order,
            center=np.array(descriptor_result.center),
            dashboard_metadata_json=np.array(
                json.dumps(self._json_ready(metadata), sort_keys=True)
            ),
        )

    def _redraw_atom_model(self, change: dict[str, Any] | None = None) -> None:
        """Refresh the simple rotating atom preview."""
        with self.model_output:
            clear_output(wait=True)
            fig = self._make_atom_model(int(self.atom_select.value), float(self.angle.value))
            display(fig)
            plt.close(fig)

    def _make_atom_model(self, z: int, angle: float = 35.0):
        """Create the Matplotlib 3D atom preview figure."""
        symbol, name = ATOM_BY_Z[int(z)]
        fig = plt.figure(figsize=(4.6, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter([0], [0], [0], s=420, color="#d1495b", edgecolor="#8b1e2d", linewidth=1.2)
        ax.text(0, 0, 0.08, symbol, color="white", ha="center", va="center", fontsize=13, weight="bold")

        theta = np.linspace(0, 2.0 * np.pi, 240)
        phase = np.deg2rad(angle)
        shell_color = "#2a9d8f"
        electron_color = "#264653"
        for shell_index, count in enumerate(shell_counts(z), start=1):
            radius = 0.55 + 0.32 * shell_index
            plane = shell_index % 3
            if plane == 0:
                x, y, zz = radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)
            elif plane == 1:
                x, y, zz = radius * np.cos(theta), np.zeros_like(theta), radius * np.sin(theta)
            else:
                x, y, zz = np.zeros_like(theta), radius * np.cos(theta), radius * np.sin(theta)
            ax.plot(x, y, zz, color=shell_color, alpha=0.32, linewidth=1.2)

            visible = min(count, 18)
            e_theta = np.linspace(0, 2.0 * np.pi, visible, endpoint=False) + phase / shell_index
            if plane == 0:
                ex, ey, ez = radius * np.cos(e_theta), radius * np.sin(e_theta), np.zeros_like(e_theta)
            elif plane == 1:
                ex, ey, ez = radius * np.cos(e_theta), np.zeros_like(e_theta), radius * np.sin(e_theta)
            else:
                ex, ey, ez = np.zeros_like(e_theta), radius * np.cos(e_theta), radius * np.sin(e_theta)
            ax.scatter(ex, ey, ez, s=24, color=electron_color, alpha=0.9)

        limit = 0.9 + 0.35 * max(1, len(shell_counts(z)))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.view_init(elev=22, azim=angle)
        ax.set_axis_off()
        ax.set_title(f"{name} ({symbol}), Z={z}", pad=4)
        fig.tight_layout()
        return fig
