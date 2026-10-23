# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input ports, bound launches and the same-basis model (SEP 31)."""

import dataclasses

import numpy as np
import pytest

from solprop.coupling import (
    CouplingLaunch,
    FieldPlane,
    InputPort,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneReference,
    basis_identity,
    same_basis,
)
from solprop.exceptions import CouplingError
from solprop.fourier import spectrum_to_internal, spectrum_to_optics
from solprop.grid import EnvelopeGrid
from solprop.pulses import GaussianPulse

LIMITS = (600e-9, 1.2e-6)


def grid_of(trange=1e-12, limits=LIMITS):
    return EnvelopeGrid.from_limits(1.0, 800e-9, limits, trange)


BASIS = PlaneBasis("modal", basis_identity("modal", {"modes": ["HE11"]}), ("HE11",))


def plane_of(grid=None, reference=None, metric=None):
    grid = grid_of() if grid is None else grid
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    samples = spectrum_to_optics(pulse.build(grid), kind=grid.kind)
    samples[..., ~grid.active_mask] = 0.0
    return FieldPlane.from_samples(
        samples,
        grid,
        PlaneGeometry(0.02),
        exterior="zero",
        basis=BASIS,
        reference=PlaneReference(0.4, 3e-11) if reference is None else reference,
        metric=metric,
    )


def port_of(grid=None, **changes):
    grid = grid_of() if grid is None else grid
    arguments = {
        "route": "modal",
        "grid": grid,
        "basis": BASIS,
        "shape": (1, 1, grid.n_omega),
        "geometry": PlaneGeometry(0.02),
        "frame": PlaneReference(0.1, 1e-11),
        "metric": PlaneMetric("identity"),
    }
    arguments.update(changes)
    return InputPort(**arguments)


def test_a_same_basis_launch_copies_and_inherits_the_reference():
    plane = plane_of()
    port = port_of()
    launch = same_basis(plane, port)
    # The identity model copies the coefficients: an exact structural contract.
    assert np.array_equal(launch.samples, plane.samples)
    assert not launch.samples.flags.writeable
    assert launch.origin == PlaneReference(0.4 - 0.1, 3e-11 - 1e-11)
    assert np.array_equal(
        launch.internal(), spectrum_to_internal(plane.samples, kind="envelope")
    )
    record = launch.record()
    assert record["model"] == "same_basis"
    assert record["port"] == port.identity
    assert record["origin"] == launch.origin.record()
    assert "array(" not in repr(launch)
    assert port.accepts(port_of()) is None


def test_a_port_names_every_disagreement_with_another():
    port = port_of()
    other_basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["HE12"]}))
    cases = {
        "basis": port_of(basis=other_basis),
        "lattice": port_of(grid_of(2e-12), shape=(1, 1, grid_of(2e-12).n_omega)),
        "side": port_of(geometry=PlaneGeometry(0.02, "after")),
        "enters at": port_of(geometry=PlaneGeometry(0.03)),
        "oriented": port_of(geometry=PlaneGeometry(0.02, x_axis=(0.0, 1.0, 0.0))),
        "metric": port_of(
            metric=PlaneMetric("dense", matrix=np.ones((1, 1, 512), complex))
        ),
    }
    for name, other in cases.items():
        refusal = port.accepts(other)
        assert refusal is not None and name in refusal, name


def test_same_basis_refuses_anything_but_the_identical_reference():
    plane = plane_of()
    other_basis = PlaneBasis("modal", basis_identity("modal", {"modes": ["HE12"]}))
    longer = grid_of(2e-12)
    cases = [
        (port_of(models=()), "not same_basis"),
        (port_of(basis=other_basis), "a coupling, not a relaunch"),
        (port_of(longer, shape=(1, 1, longer.n_omega)), "regrid it first"),
        (port_of(shape=(2, 1, 512)), "shape"),
        (port_of(geometry=PlaneGeometry(0.02, normal=(0.0, 1.0, 0.0))), "oriented"),
        (
            port_of(metric=PlaneMetric("dense", matrix=np.ones((1, 1, 512), complex))),
            "metric is identity",
        ),
    ]
    for port, match in cases:
        with pytest.raises(CouplingError, match=match):
            same_basis(plane, port)
    unknown = plane_of(metric=PlaneMetric("unknown", reason="no medium"))
    with pytest.raises(CouplingError, match="no medium"):
        same_basis(unknown, port_of())


def test_same_basis_refuses_light_the_receiving_band_would_remove():
    wide = grid_of(limits=(500e-9, 1.6e-6))
    plane = plane_of(wide)
    # The same lattice and a narrower requested band: the receiving run's hard mask
    # would remove light, and the relaunch refuses to do so silently.
    narrow = dataclasses.replace(
        wide, omega_collar=grid_of(limits=(780e-9, 820e-9)).omega_collar
    )
    with pytest.raises(CouplingError, match="outside the port's active band"):
        same_basis(plane, port_of(narrow))


def test_admittance_metrics_must_be_the_same_medium():
    grid = grid_of()
    weights = np.ones(1)
    gas = np.full((1, grid.n_omega), 1.0003 + 0j)
    port = port_of(
        metric=PlaneMetric("admittance", admittance=gas, weights=weights, scale=2.0)
    )
    same = PlaneMetric("admittance", admittance=gas.copy(), weights=weights, scale=2.0)
    assert port.accepts(dataclasses.replace(port, metric=same)) is None
    for metric in (
        PlaneMetric("admittance", admittance=gas * 1.0001, weights=weights, scale=2.0),
        PlaneMetric("admittance", admittance=gas, weights=weights, scale=1.0),
    ):
        assert port.accepts(dataclasses.replace(port, metric=metric)) is not None


def test_ports_and_launches_refuse_malformed_parts():
    grid = grid_of()
    with pytest.raises(CouplingError, match="needs a radial basis"):
        port_of(route="radial")
    with pytest.raises(CouplingError, match="shape"):
        port_of(shape=(1, 1, 3))
    with pytest.raises(CouplingError, match="reference index"):
        port_of(reference_index=np.ones(3))
    port = port_of(reference_index=np.ones(grid.n_omega), nodes=np.zeros((2, 2)))
    assert port.reference_index is not None and not port.reference_index.flags.writeable
    assert port.nodes is not None and not port.nodes.flags.writeable
    with pytest.raises(CouplingError, match="port's shape"):
        CouplingLaunch(
            port, np.zeros((2, 1, grid.n_omega), complex), PlaneReference(), "x", {}
        )
