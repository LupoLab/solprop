# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare stored spatial Luna fields with solprop, without executing Luna.

The entrance spectrum is shared; material samples come from
``luna_spatial_media.jl``. Cross-code evolution metrics are diagnostics, not global
error bounds inferred from a local controller tolerance.
Each invocation writes a new JSON record and compact observable arrays; existing
results are never overwritten. Run through the solprop environment.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from solprop import fourier
from solprop.backends.registry import select
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.sellmeier import PoleSumSusceptibility, ReferenceState
from solprop.spatial import SpatialMedium, on_axis_intensity
from solprop.stepper import PIController

# The test-only rate oracle lives beside the tests; a standalone run reaches it here.
sys.path.insert(0, str(Path(__file__).parents[1]))
from rate_oracles import adk_oracle

ARGON = PoleSumSusceptibility(
    gas=Gas.Ar,
    source="kingston",
    reference_state=ReferenceState(pressure_bar=1.0, temperature=273.15),
    wavelength_range=(110e-9, 1000e-9),
    citation=(
        "three-pole fit to Kingston (1964) and Börzsönyi (2008), Luna's :Ar "
        "(commit 56cc3ab); solprop's former registry default (dc4ca0d)"
    ),
    b=(0.00032323117217767093, 0.00011557814904827939, 0.00010909808164540697),
    c=(0.0045416501944977915, 0.011120847461156543, 0.0006827046691889898),
)
"""Luna's argon law, supplied as a caller's own data (solprop has none of its own)."""

EPS = np.finfo(float).eps
# Pin provenance when this module is loaded: another task may commit documentation
# while a long comparison queue is still executing this already-loaded code.
SOURCE_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def emit(message):
    """Write a durable progress line from this standalone command."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def gamma_n(count):
    """Return the standard positive-reduction floating-point bound."""
    return count * EPS / (1 - count * EPS)


def read_value(dataset) -> Any:
    """Decode a scalar or array in the Julia HDF5 provenance."""
    # HDF5's mixed scalar/array metadata is the dynamic boundary of this CLI.
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return [one.decode() if isinstance(one, bytes) else one for one in value]
    return value


def checked_gap(checks, name, actual, expected, bound):
    """Record and enforce an already-derived input-arithmetic bound."""
    if np.shape(actual) != np.shape(expected):
        raise ValueError(f"{name}: shapes {np.shape(actual)} != {np.shape(expected)}")
    scale = float(np.max(np.abs(expected)))
    value = float(np.max(np.abs(actual - expected))) / scale
    checks.append(
        {"name": name, "value": value, "bound": bound, "pass": value <= bound}
    )
    if value > bound:
        raise ValueError(f"{name}: {value:.17g} exceeds {bound:.17g}")


@dataclass(frozen=True, kw_only=True)
class WitnessCladding(GlassIndex):
    """Preserve Luna's signed complex samples, including interpolation overshoot."""

    samples: CubicSpline

    def _index_and_derivative(self, omega):
        axis = np.log(2e6 * np.pi * C / omega)
        return self.samples(axis), -self.samples(axis, 1) / omega


def cladding_from(group):
    """Interpolate Luna's own cladding on exactly the propagated frequencies."""
    omega = group["omega"][()][group["active"][()].astype(bool)]
    index = group["cladding"][()]
    wavelength_um = 2e6 * np.pi * C / omega
    order = np.argsort(wavelength_um)
    axis = np.log(wavelength_um[order])
    return WitnessCladding(
        name="SiO2",
        source="luna",
        citation="Luna material witness (Luna's default silica table)",
        wavelength_range=(
            float(wavelength_um.min()) * 1e-6,
            float(wavelength_um.max()) * 1e-6,
        ),
        samples=CubicSpline(axis, index[order]),
    )


def stored_plane(file, index, geometry):
    """Adopt Julia's reversed HDF5 axes in solprop's internal FFT convention."""
    value = file["Eω"][index]
    if geometry == "free":
        # HDF5: x,y,omega. Solprop: flattened y,x,omega, with x fastest.
        value = value.transpose(1, 0, 2).reshape(-1, value.shape[-1])
    value = np.ascontiguousarray(value[:, None, :])
    if geometry == "modal_product":
        return np.concatenate((np.zeros_like(value), value), axis=1)
    return value


def prepare(file, media, backend, rtol, checks, prefactor_index=None):
    """Build matching physics objects without changing the package defaults."""
    p = {key: read_value(value) for key, value in file["witness"].items()}
    name = Path(file.filename).stem
    geometry = (
        "radial"
        if name.startswith("radial_")
        else ("free" if name.startswith("free_") else "modal")
    )
    constructor = FieldGrid if p["family"] == "field" else EnvelopeGrid
    grid = constructor.from_limits(
        float(p["distance"]),
        float(p["wavelength"]),
        (float(p["lambda_min"]), float(p["lambda_max"])),
        float(p["time_window"]),
    )
    # Axes involve at most 64 elementary rounded operations; no interpolation or
    # propagation error enters this check. Declare the bound before reading gaps.
    axis_bound = gamma_n(64)
    for ours, theirs in (
        (grid.omega, "ω"),
        (grid.omega_fine, "ωo"),
        (grid.t, "t"),
        (grid.t_fine, "to"),
    ):
        checked_gap(
            checks, f"axis {theirs}", ours, file[f"grid/{theirs}"][()], axis_bound
        )
    group = media[name]
    # Both use the identical three-pole argon fit. Scale its coefficients to
    # cancel the independently computed EOS reference density.
    factor = ARGON.reference_density / float(media["argon_reference_density"][()])
    if geometry == "modal":
        # The modes keep their own EOS density, so the fit also cancels it against
        # the operating one, and the run's response density is the operating one.
        medium = SpatialMedium("Ar", float(p["pressure_bar"]), float(p["temperature"]))
        host_density = float(medium.density(np.array([0.0]))[0])
        factor = factor * float(p["density"]) / host_density
        response_density = float(p["density"])
    else:
        # A gas medium carries the one density its index and its responses take
        # (an explicit response density beside it is refused): hold it at the
        # operating density.
        response_density = None
    susceptibility = replace(ARGON, b=tuple(one * factor for one in ARGON.b))
    medium = SpatialMedium(
        "Ar",
        float(p["pressure_bar"]),
        float(p["temperature"]),
        susceptibility=susceptibility,
    )
    if geometry != "modal":
        medium = medium.at_density(float(p["density"]))
    active = group["active"][()].astype(bool)
    # Three pole evaluations, EOS cancellation and sqrt use <128 operations.
    # n~1 conditions this absolute/relative index comparison; it is not a phase bound.
    material_bound = gamma_n(128)
    checked_gap(
        checks,
        "bulk index",
        medium.index(grid.omega[active]),
        group["index"][()],
        material_bound,
    )
    responses: list[object] = [
        KerrResponse(float(p["chi3_per_particle"]), thg=grid.kind == "field")
    ]
    if bool(p.get("plasma", False)):
        # solprop's former ADK("Ar", hydrogenic=True), bit for bit (the test oracle).
        responses.append(PlasmaResponse("Ar", adk_oracle("Ar", hydrogenic=True)))
    settings = SimulationSettings(
        controller=PIController(rtol=rtol, atol=float(p["atol"])),
        save_planes=file["z"][()][1:],
        dz_max=float(p["distance"]) / 2,
        dz_init=float(p.get("initial_step", 1e-4)),
        density=response_density,
        backend=backend,
        prefactor_index=prefactor_index,
        polarisation_taper=True,
        ionisation_field_range=(1e3, 1e12) if bool(p.get("plasma", False)) else None,
    )
    space, modes = None, []
    if geometry == "modal":
        clad = cladding_from(group)
        checked_gap(
            checks,
            "cladding samples",
            clad.index(grid.omega[active]),
            group["cladding"][()],
            gamma_n(128),
        )
        modes = [
            MarcatiliMode(
                float(p["radius"]),
                "Ar",
                float(p["pressure_bar"]),
                temperature=float(p["temperature"]),
                kind=kind,
                n=int(n),
                m=int(m),
                phi=float(phi),
                loss=bool(p["loss"]),
                susceptibility=susceptibility,
                cladding=clad,
            )
            for kind, n, m, phi in zip(
                p["mode_kinds"], p["mode_n"], p["mode_m"], p["mode_phi"], strict=True
            )
        ]
        settings = replace(
            settings,
            components=tuple(p["components"]),
            quadrature=transverse_quadrature(
                modes,
                n_r=int(p["radius_nodes"]),
                n_theta=int(p["theta_nodes"]),
                full=bool(p["full_rule"]),
            ),
        )
        if p["components"] == "xy" and all(
            mode.principal_component() == "x" for mode in modes
        ):
            # Luna's explicit zero x component is invalid as a modal component
            # request. The product route retains x=0 as the independent state
            # polarisation and still exercises the two-component plasma arithmetic.
            # The resolved component is the profile's own direction, which is x
            # (ADR-031, docs/dev/decisions.md) where Luna's phi = 0 gives y; the scalar
            # profile itself is the same function of r.
            geometry = "modal_product"
            settings = replace(settings, components=("x",))
    elif geometry == "radial":
        space = RadialGrid(float(p["radius"]), int(p["n_r"]), window_factor=0)
        checked_gap(checks, "radial nodes", space.r, p["r"], gamma_n(128))
        checked_gap(checks, "radial frequencies", space.k_r, p["k_r"], gamma_n(128))
    else:
        # Luna expands the supplied radius by 10% but never applies xywin.
        # Use that actual extent and explicitly disable solprop's spatial absorber.
        space = FreeGrid(
            -float(p["x"][0]),
            int(p["n_x"]),
            -float(p["y"][0]),
            int(p["n_y"]),
            window_factor=0,
        )
        for axis in ("x", "y", "k_x", "k_y"):
            checked_gap(
                checks, f"Cartesian {axis}", getattr(space, axis), p[axis], axis_bound
            )
        settings = replace(settings, dz_max=float(p["distance"]))
    initial = stored_plane(file, 0, geometry)
    simulation = Simulation(
        grid,
        modes,
        responses,
        initial,
        settings=settings,
        spatial_grid=space,
        medium=medium if space is not None else None,
    )
    return simulation, geometry, p


def observables(simulation, field):
    """Reduce a spectrum to energy density and physical on-axis peak intensity."""
    grid, space = simulation.grid, simulation.spatial_grid
    if not np.all(np.isfinite(field)):
        raise ValueError("nonfinite saved field")
    density = fourier.spectral_density(field, grid.dt, kind=grid.kind)
    if space is None:
        per_mode = np.sum(density, axis=1)
        coefficients = np.stack(
            [
                mode.field(np.zeros((1, 2)))[0] / math.sqrt(mode.normalisation())
                for mode in simulation.modes
            ]
        )
        # The product route carries the two Cartesian directions on the state's
        # polarisation axis and one scalar profile, which is the mode's own principal
        # component: x for an unrotated capillary mode (ADR-031), y in Luna. A mode
        # with no principal component has no scalar profile to read, so it is named
        # rather than silently reduced to column 0.
        component = simulation.modes[0].principal_component()
        if component not in ("x", "y"):
            raise ValueError(
                "the modal-product route needs an axis-aligned mode; "
                f"{simulation.modes[0]!r} reports principal component {component!r}"
            )
        principal = 1 if component == "y" else 0
        axis = (
            np.einsum("m,mpw->pw", coefficients[:, principal], field)
            if field.shape[1] == 2
            else np.einsum("mp,miw->pw", coefficients, field)
        )
        temporal = fourier.inverse(axis, kind=grid.kind)
        if grid.kind == "field":
            temporal = fourier.analytic_signal(temporal)
        peak = float(np.max(0.5 * EPS0 * C * np.sum(np.abs(temporal) ** 2, axis=0)))
    else:
        per_mode = (
            0.5 * EPS0 * C * space.spectral_weights[:, None] * np.sum(density, axis=1)
        )
        peak = on_axis_intensity(grid, space, field)
        per_mode = np.sum(per_mode, axis=0, keepdims=True)
    spectrum = np.sum(per_mode, axis=0)
    if not np.all(np.isfinite(spectrum)) or not np.sum(spectrum) > 0:
        raise ValueError("spectral energy must be finite and positive")
    if not math.isfinite(peak) or peak < 0:
        raise ValueError("on-axis intensity must be finite and nonnegative")
    return spectrum, np.sum(per_mode, axis=1) * grid.d_omega, peak


def compare_observables(ours, theirs):
    """Report M2-style spectral shape/content and the along-z axis intensity."""
    a, b = ours["spectrum"], theirs["spectrum"]
    content = np.sum(a, axis=1) / np.sum(b, axis=1) - 1
    shape = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    axis_scale = float(np.max(theirs["axis_peak"]))
    axis_gap = (
        float(np.max(np.abs(ours["axis_peak"] - theirs["axis_peak"]))) / axis_scale
        if axis_scale > 0
        else None
    )
    return {
        "exit_content_relative": float(content[-1]),
        "max_content_relative": float(np.max(np.abs(content))),
        "exit_spectral_shape_l2": float(shape[-1]),
        "max_spectral_shape_l2": float(np.max(shape)),
        "axis_peak_along_z_relative_linf": axis_gap,
        "axis_peak_luna_max_w_m2": axis_scale,
        "axis_peak_along_z_absolute_linf_w_m2": float(
            np.max(np.abs(ours["axis_peak"] - theirs["axis_peak"]))
        ),
        "entrance_content_relative": float(content[0]),
        "entrance_spectral_shape_l2": float(shape[0]),
        "exit_mode_energy_ours_j": ours["mode_energy"][-1].tolist(),
        "exit_mode_energy_luna_j": theirs["mode_energy"][-1].tolist(),
    }


def advance(backend, simulation, file, geometry, artifact, result):
    """Stream saved planes into compact observable records."""
    ours, theirs, summaries = [], [], []
    propagation_seconds, reduction_seconds = 0.0, 0.0
    for index, z in enumerate(file["z"][()]):
        if index:
            tick = time.perf_counter()
            summary = backend.propagate_to(float(z))
            propagation_seconds += time.perf_counter() - tick
            summaries.append(asdict(summary))
        tick = time.perf_counter()
        state = backend.observe().require_field()[0]
        ours.append(observables(simulation, state))
        theirs.append(observables(simulation, stored_plane(file, index, geometry)))
        reduction_seconds += time.perf_counter() - tick
        if index == 0 or index % 25 == 0 or index == len(file["z"]) - 1:
            emit(
                f"{artifact.stem}: plane {index}/{len(file['z']) - 1}, z={z:g}, "
                f"propagation={propagation_seconds:.3f}s"
            )
    keys = ("spectrum", "mode_energy", "axis_peak")
    ours = {key: np.stack([one[i] for one in ours]) for i, key in enumerate(keys)}
    theirs = {key: np.stack([one[i] for one in theirs]) for i, key in enumerate(keys)}
    arrays = {
        f"{prefix}_{key}": value
        for prefix, data in (("solprop", ours), ("luna", theirs))
        for key, value in data.items()
    }
    np.savez_compressed(
        artifact,
        allow_pickle=False,
        z=file["z"][()],
        omega=simulation.grid.omega,
        **arrays,
    )
    result.update(
        metrics=compare_observables(ours, theirs),
        flat_band_metrics=compare_observables(
            ours | {"spectrum": ours["spectrum"][:, simulation.grid.inband_mask]},
            theirs | {"spectrum": theirs["spectrum"][:, simulation.grid.inband_mask]},
        ),
        entrance_energy_solprop_j=float(np.sum(ours["mode_energy"][0])),
        entrance_energy_luna_j=float(np.sum(theirs["mode_energy"][0])),
        summaries=summaries,
        propagation_seconds=propagation_seconds,
        reduction_seconds=reduction_seconds,
        status="complete",
    )


def run_one(path, media_path, backend_name, rtol, destination, prefactor_index=None):
    """Run one comparison, preserving failures and all emitted diagnostics."""
    label = f"{path.stem}-{backend_name}-{rtol:g}"
    target = destination / f"{label}.json"
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    result = {
        "case": path.stem,
        "backend": backend_name,
        "rtol": rtol,
        "prefactor_index": prefactor_index,
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": [],
        "source_commit": SOURCE_COMMIT,
        "runner_sha256": RUNNER_SHA256,
        "comparison_status": "diagnostic; no certified global cross-code bound",
    }
    backend = None
    with warnings.catch_warnings(record=True) as emitted:
        try:
            with h5py.File(path) as file, h5py.File(media_path) as media:
                simulation, geometry, p = prepare(
                    file, media, backend_name, rtol, result["checks"], prefactor_index
                )
                result["geometry"] = geometry
                result["luna_seconds"] = float(p["duration_seconds"])
                result["luna_commit"] = p["luna_commit"]
                result["requested_energy_j"] = float(p["energy"])
                result["grid"] = {
                    key: getattr(simulation.grid, key)
                    for key in ("n_t", "n_to", "n_omega")
                }
                problem = simulation.translate()
                backend = select(
                    backend_name, problem, grid=simulation.grid, modes=simulation.modes
                )
                result["memory_ledger"] = backend.memory_ledger()
                backend.set_initial(problem.initial.field0, None)
                result["setup_seconds"] = time.perf_counter() - started
                advance(
                    backend,
                    simulation,
                    file,
                    geometry,
                    destination / f"{label}.npz",
                    result,
                )
        except Exception as error:
            result.update(
                status="failed", error=str(error), traceback=traceback.format_exc()
            )
        finally:
            if backend is not None:
                backend.close()
    result["warnings"] = [
        {
            "category": one.category.__name__,
            "message": str(one.message),
            "filename": one.filename,
            "line": one.lineno,
        }
        for one in emitted
    ]
    result["wall_seconds"] = time.perf_counter() - started
    target.write_text(
        json.dumps(result, indent=2, default=lambda value: value.tolist()) + "\n"
    )
    for item in result["warnings"]:
        emit(f"{item['category']}: {item['message']}")
    emit(
        json.dumps(
            {key: result[key] for key in ("case", "backend", "status", "wall_seconds")}
            | {
                "metrics": {
                    key: value
                    for key, value in result.get("metrics", {}).items()
                    if not isinstance(value, list)
                },
                "error": result.get("error"),
            }
        )
    )
    return result["status"] == "complete"


def main():
    """Run selected cases; return failure if any case could not be compared."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument(
        "--backend", default="numba_cpu", choices=("numba_cpu", "numpy_ref", "native")
    )
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--prefactor-index",
        type=float,
        default=None,
        help="Explicit modal-prefactor diagnostic; 1 reproduces Luna",
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    outcomes = [
        run_one(
            args.directory / f"{case}.h5",
            args.directory / "media.h5",
            args.backend,
            args.rtol,
            args.destination,
            args.prefactor_index,
        )
        for case in args.cases
    ]
    raise SystemExit(0 if all(outcomes) else 1)


if __name__ == "__main__":
    main()
