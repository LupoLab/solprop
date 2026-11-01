# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route from its entry point: construction, launch, noise, output.

``Simulation.from_waveguide(..., provider=...)`` and
``prop_waveguide(..., provider=...)`` are the only ways into the route from a script;
what the translation layer builds from a guide is
``tests/translate/test_general_route.py`` and what the reference backend does with it is
``tests/backends/test_general_modal.py``.

The last tests are end-to-end systems, on a **weakly guiding**
step-index fibre over a millimetre: one HE11 channel against ``prop_gnlse`` at the
matched Poynting gamma, and the HE11 parity pair under a rotation and under a
circular launch. Every bound is assembled from quantities measured before the run.

**Both routes run with no absorber.** ``prop_gnlse`` builds none, and
``from_waveguide`` builds the grid's default collar like every other waveguide
entry point, so a comparison with the collar left on is not like for like: the
spectral collar filters the self-phase-modulation sidebands of the field-grid run
and the envelope run has nothing of the sort. The collar's own effect is measured
and recorded beside the comparison rather than folded into it.
"""

import dataclasses
import inspect
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bounds import departure
from legacy_geometry import LEGACY_GRID
from march_bounds import (
    FIXED_ORDER,
    STATISTIC_FACTOR,
    assert_matched,
    simulation_bound,
)
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    general_grid,
    general_guide,
)
from solprop import fourier
from solprop.backends import select
from solprop.backends.batch import LockstepRunner
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.constants import ROOM_TEMPERATURE, C
from solprop.coupling import CouplingModel, CouplingReport
from solprop.driver import Simulation, prop_gnlse, prop_waveguide
from solprop.driver import batch as batch_module
from solprop.driver.batch import (
    BatchMember,
    _modal_identity,
    compatible_groups,
    refuse_mixed_guides,
)
from solprop.driver.executors import Batch
from solprop.driver.run import run_problem
from solprop.driver.scan import Scan, run_scan
from solprop.driver.simulation import (
    _GAS_KEYWORD_DEFAULTS,
    _core_index,
    _in_channel,
    _refuse_gas_keywords,
    _shot_noise,
)
from solprop.exceptions import (
    BackendError,
    DriverError,
    GeneralModalBasisWarning,
    ModeError,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import IndexMaterial, MarcatiliMode, StepIndexMode
from solprop.modes.guide import ModalGuide, _profile_defect
from solprop.noise import photons_per_bin
from solprop.nonlinear.normalisation import modal_kerr_coefficient
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.output.hdf5 import resume
from solprop.output.memory import MemoryOutput
from solprop.physdata import n2_from_chi3
from solprop.physdata.sellmeier import gamma as gas_susceptibility
from solprop.processing import scans
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.stats import StatsSpec, analytic_field, energy, physical_energy
from solprop.stepper import FixedSchedule, Multistep
from solprop.translate.coupled import metric_at
from solprop.translate.general import (
    general_modal_route,
    interpolated_gram,
    metric_departure,
    modal_identity,
    occupancy_weight,
)
from solprop.translate.modal_tables import (
    DEFAULT_GRAM_ATOL,
    align_modal_sample,
    channel_area,
    project_transverse_field,
    sample_modal_table,
)

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

PULSE = {"wavelength": 835e-9, "fwhm": 20e-15, "energy": 1e-12, "shape": "gauss"}
RUN = {
    # A tenth of a millimetre in two steps: these tests are about construction,
    # the launch, the refusals and the identity of a repeated draw, and none of
    # them needs a long propagation. The two end-to-end systems below take the
    # whole millimetre.
    "save_planes": [GENERAL_Z_END / 10],
    "backend": "numpy_ref",
    "dz_max": GENERAL_Z_END / 20,
    "dz_init": GENERAL_Z_END / 20,
}


def kept(field: np.ndarray | None) -> np.ndarray:
    """Return a saved plane's field: every plane of these runs keeps its own."""
    assert field is not None
    return field


@pytest.fixture(scope="module")
def grid():
    """The field grid the construction tests use."""
    return general_grid()


@pytest.fixture(scope="module")
def guide(grid):
    """One HE11 channel of a 1.5 um strand, sampled over the run's active band."""
    return general_guide(grid=grid)


@pytest.fixture(scope="module")
def pair_guide(grid):
    """The two parities of that channel, declared degenerate."""
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        grid=grid,
    )


def build(guide, grid, **overrides):
    """Return a general modal simulation on this guide and grid."""
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        **({"integrator": Multistep(), **({"noise": None, **PULSE | RUN | overrides})}),
    )


# ------------------------------------------------------------------- construction


def test_a_provider_builds_a_simulation_that_translates_to_the_general_route(
    guide, grid
):
    """The entry point reaches the general route, with the guide's own channels."""
    simulation = build(guide, grid)
    assert simulation.general is not None
    assert simulation.modes == tuple(guide.mode_collection.modes)
    assert simulation.responses == ()
    problem = simulation.translate()
    assert problem.transform.modal_tables is not None
    assert problem.linear.modal_metric is not None
    assert problem.linear.modal_metric.kind == "identity"
    assert problem.responses.regional_responses is not None
    assert problem.linear.coupled_flow is None
    # The guide's own channels may be passed explicitly; nothing else may.
    assert (
        Simulation.from_waveguide(
            list(guide.mode_collection.modes),
            GENERAL_Z_END,
            provider=guide,
            grid=grid,
            **({"integrator": Multistep(), **({"noise": None, **PULSE | RUN})}),
        ).modes
        == simulation.modes
    )
    assert "guide channel" in str(simulation)


GAS_KEYWORDS = {
    "gas": "He",
    "pressure": 1.0,
    "pressure_floor": 0.1,
    "vacuum_end": "floor",
    "susceptibility": gas_susceptibility("He"),
    "temperature": ROOM_TEMPERATURE + 5.0,
    "kerr": False,
    "thg": True,
    "plasma": True,
    "preionisation": 0.1,
    "plasma_integration": "band",
    "raman": True,
    "mode_average": True,
    "n_r": 16,
    "n_theta": 9,
    "radial_rule": "gauss",
    "ionisation_field_range": (1e9, 1e11),
    "polarisation": "x",
    "envelope": True,
    "node_budget": 1024,
    "taper_coupling": False,
}
"""One value per refused keyword that is not the value meaning "not given".

Keyed by keyword, so that the parametrisation cannot list one twice or omit one:
:func:`test_from_waveguide_refuses_or_names_every_keyword_it_takes` asserts these
are exactly the refusal table's.
"""


@pytest.mark.parametrize("keyword", sorted(GAS_KEYWORDS))
def test_every_gas_keyword_is_refused_by_name_beside_a_provider(guide, grid, keyword):
    """A guide owns its materials, responses, rule and grid kind (defence in depth)."""
    with pytest.raises(DriverError, match=f"{keyword}="):
        build(guide, grid, **{keyword: GAS_KEYWORDS[keyword]})


def test_from_waveguide_refuses_or_names_every_keyword_it_takes():
    """Lock the signature to the refusal table, so a new keyword cannot be ignored.

    ``_from_general`` takes the keywords it carries by name and collects the rest
    in ``**refused``; every collected one must be in the table, or it would be
    swallowed silently -- accepted by the signature, never read, and never
    refused. The defaults must agree too: the table's entry is the value that
    means "not given", and a table entry that is not the signature's default would
    refuse a call that passed nothing.
    """
    taken = set(inspect.signature(Simulation.from_waveguide).parameters)
    named = set(inspect.signature(Simulation._from_general).parameters) - {"refused"}
    assert set(GAS_KEYWORDS) == set(_GAS_KEYWORD_DEFAULTS)
    assert taken == named | set(_GAS_KEYWORD_DEFAULTS)
    signature = inspect.signature(Simulation.from_waveguide).parameters
    for name, default in _GAS_KEYWORD_DEFAULTS.items():
        assert signature[name].default == default


def test_an_array_valued_gas_keyword_is_refused_by_name_and_not_by_a_crash():
    """`value == default` is elementwise for an array; its truth value raises.

    Driven directly, as the session's runtime hook refuses an array where the
    signature declares a pressure path; in production no hook runs and a caller
    reaching this table with an array must still be told which keyword it was.
    """
    with pytest.raises(DriverError, match="pressure="):
        _refuse_gas_keywords({"pressure": np.array([1.0, 2.0])})
    # And the "not given" values still pass, array-free.
    _refuse_gas_keywords(dict.fromkeys(GAS_KEYWORDS) | dict(_GAS_KEYWORD_DEFAULTS))


def test_the_gas_route_still_needs_a_gas_and_refuses_the_guide_keywords():
    """Without a provider: a gas is required and the guide's keywords are not."""
    with pytest.raises(DriverError, match="needs a gas fill"):
        Simulation.from_waveguide(
            (), GENERAL_Z_END, wavelength=835e-9, noise=None, integrator=Multistep()
        )
    with pytest.raises(DriverError, match="channel= belongs to the general"):
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            "He",
            wavelength=835e-9,
            channel=0,
            noise=None,
            integrator=Multistep(),
        )
    with pytest.raises(DriverError, match="observable_rtol= belongs to the general"):
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            "He",
            wavelength=835e-9,
            observable_rtol=1e-6,
            noise=None,
            integrator=Multistep(),
        )
    with pytest.raises(DriverError, match="observable_atol= belongs to the general"):
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            "He",
            wavelength=835e-9,
            observable_atol=1e-16,
            noise=None,
            integrator=Multistep(),
        )
    with pytest.raises(DriverError, match="kerr_form= belongs to the general"):
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            "He",
            wavelength=835e-9,
            kerr_form="analytic",
            noise=None,
            integrator=Multistep(),
        )
    # The default is not a value the gas route has to refuse: it names no model of
    # its own, and `thg=` is where a gas-filled field run says the same thing.
    with pytest.raises(DriverError, match="needs a gas fill"):
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            wavelength=835e-9,
            kerr_form="real",
            noise=None,
            integrator=Multistep(),
        )


def test_modes_of_ones_own_are_refused_beside_a_guide(guide, grid):
    other = general_guide(grid=grid, radius=1.6e-6)
    with pytest.raises(DriverError, match="supplies its own 1 ordered channel"):
        Simulation.from_waveguide(
            list(other.mode_collection.modes),
            GENERAL_Z_END,
            provider=guide,
            grid=grid,
            **({"integrator": Multistep(), **({"noise": None, **PULSE | RUN})}),
        )


def test_an_envelope_grid_is_refused_at_the_entry_point(guide):
    envelope = EnvelopeGrid.from_limits(GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15)
    with pytest.raises(DriverError, match="propagates on a field grid"):
        build(guide, envelope)


@pytest.mark.parametrize("keyword", ["observable_rtol", "observable_atol"])
def test_an_observable_tolerance_is_accepted_only_as_the_solvers_own(
    guide, grid, keyword
):
    """`None` means the solver's rtol and atol; a number is refused."""
    assert build(guide, grid, **{keyword: None}).general is not None
    with pytest.raises(DriverError, match="is not implemented"):
        build(guide, grid, **{keyword: 1e-7})


def test_a_guide_narrower_than_the_active_band_is_refused_at_construction(grid):
    narrow = general_guide(
        grid=FieldGrid.from_limits(GENERAL_Z_END, 835e-9, (830e-9, 840e-9), 60e-15)
    )
    with pytest.raises(DriverError, match="twice the band its run declares") as caught:
        build(narrow, grid)
    assert isinstance(caught.value.__cause__, TranslationError)


def test_the_route_constructors_refusals_reach_the_caller_as_driver_errors(grid):
    """`from_waveguide` promises `DriverError`; the bridge raises its own.

    The message is the translation layer's, unchanged and as specific, and the
    cause is kept, so a caller who wants the bridge's own type still has it.
    """
    unsampled = ModalGuide.from_step_index(
        [StepIndexMode(1.5e-6, core=1.45, cladding=1.0, profile_wavelength=835e-9)],
        core=IndexMaterial(1.45),
        cladding=IndexMaterial(1.0),
    )
    with pytest.raises(DriverError, match="declares no sampling") as caught:
        Simulation.from_waveguide(
            (),
            GENERAL_Z_END,
            provider=unsampled,
            grid=grid,
            **({"integrator": Multistep(), **({"noise": None, **PULSE | RUN})}),
        )
    assert isinstance(caught.value.__cause__, TranslationError)


# ------------------------------------------------------------------------- launch


def test_the_launch_is_the_pulses_own_amplitude_bit_for_bit(guide, grid):
    """No `sqrt(Re n_eff)` boundary: a Poynting amplitude is the true-power one.

    The provider normalises every channel to unit longitudinal Poynting flux, so
    the true-power interface amplitude **is** the state, the power weight is the
    constant one, and the launch that reaches the seam must be the pulse's own
    array unchanged -- not to a tolerance, bit for bit.
    """
    simulation = build(guide, grid)
    built = GaussianPulse(
        wavelength=PULSE["wavelength"], fwhm=PULSE["fwhm"], energy=PULSE["energy"]
    ).build(grid)
    problem = simulation.translate()
    assert np.array_equal(simulation._field0[0], built[0])
    assert np.array_equal(problem.initial.field0[0, 0], built[0])
    assert np.all(problem.linear.power_weight_constant == 1.0)
    # And the launched true-power energy is the pulse's.
    assert_allclose(
        float(energy(grid, problem.initial.field0[0])[0, 0]),
        PULSE["energy"],
        rtol=1e-12,
        atol=0.0,
    )


def test_a_pulse_enters_the_channel_it_is_told_to(pair_guide, grid):
    labels = [
        label.display_label for label in pair_guide.mode_collection.channel_labels
    ]
    assert labels == ["HE1,1:even", "HE1,1:odd"]
    for channel, row in ((None, 0), (0, 0), (1, 1), ("HE1,1:odd", 1)):
        state = build(pair_guide, grid, channel=channel)._field0
        assert np.any(state[row]) and not np.any(state[1 - row])
    for channel in ("HE9,9", 2, -1):
        with pytest.raises(DriverError, match="channel="):
            build(pair_guide, grid, channel=channel)


def projected_tables(pair_guide, grid):
    """Return the pair guide's reference tables on this grid."""
    return general_modal_route(
        pair_guide,
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=float(grid.omega0),
    ).tables


def test_a_declared_field_comes_back_as_the_amplitudes_it_was_built_from(
    pair_guide, grid
):
    """The round trip of the launch map, and the power it reports.

    A field synthesised from known amplitudes and projected back through the
    record's coordinate dual returns those amplitudes. The bound, before the
    numbers: the map is ``a = W^-1 B`` and the synthesis is its exact inverse, so
    the only error is the arithmetic -- a handful of epsilon times the node count,
    inside 1e-13 on this 288-node rule -- and **not** the basis's Gram defect,
    because the E and H halves of ``B`` are equal for a forward superposition
    whatever ``W`` is. The captured flux is ``a.H W a``, so it departs from
    ``sum|a|^2`` by the basis's own interpolated departure and by no more.
    """
    tables = projected_tables(pair_guide, grid)
    amplitudes = np.array([1.0, 0.3 - 0.7j])[:, None] * np.ones(grid.n_omega)
    field = np.einsum("qcmw,mw->qcw", tables.electric[0], amplitudes)
    launched = project_transverse_field(tables, field)
    # Inside the guide's support: outside it the stored reference is an explicit
    # zero, so the synthesised field is zero there and so is what comes back.
    support = np.any(np.abs(field) > 0.0, axis=(0, 1))
    assert support.sum() > 0
    assert_allclose(
        launched.amplitudes[:, support],
        amplitudes[:, support],
        rtol=1e-13,
        atol=0.0,
    )
    sampling = pair_guide.sampling
    assert sampling is not None
    departure = metric_departure(
        interpolated_gram(
            sample_modal_table(
                pair_guide.field_provider,
                sampling.omega,
                np.array([sampling.z]),
                sampling.nodes,
                degenerate_blocks=sampling.degenerate_blocks,
            ),
            tables,
        ),
        occupied=grid.inband_mask,
    )
    expected = float(np.sum(np.abs(amplitudes[:, 0]) ** 2))
    assert_allclose(
        launched.captured[grid.inband_mask],
        expected,
        rtol=2.0 * departure,
        atol=0.0,
    )
    # And nothing is missed: this field is exactly a superposition of the channels.
    assert launched.residual_share < 1e-25


def test_a_declared_field_the_channels_do_not_span_names_its_remainder(
    pair_guide, grid
):
    """What the bound channels miss is reported, not discarded.

    A field the guide's two channels cannot reproduce -- here one channel's own
    pattern with its sign flipped on every other node, which is neither of them
    and neither orthogonal to them -- leaves a residual, and the projection says
    how much of the declared field's own quadrature norm it is.
    """
    tables = projected_tables(pair_guide, grid)
    amplitudes = np.array([1.0, 0.0])[:, None] * np.ones(grid.n_omega)
    spoiled = np.einsum("qcmw,mw->qcw", tables.electric[0], amplitudes)
    spoiled[::2] *= -1.0
    launched = project_transverse_field(tables, spoiled)
    # Exactly one: the sign-flipped pattern is orthogonal to both channels, so
    # the projection is zero and the whole declared field is the residual. The
    # share is **not** clipped at one -- on a basis that is not orthonormal the
    # flux projection can overshoot -- so this is a measurement and not a cap.
    assert launched.residual_share == pytest.approx(1.0, abs=1e-4)
    # Nearly, and not exactly: the node rule does not resolve the sign flip
    # perfectly, so a little of it survives the projection. That little is what
    # keeps the share below one rather than a clip keeping it there.
    assert launched.residual_share < 1.0
    # The residual is a share of the declared field and is `nan` where nothing was
    # declared, rather than a zero that would read as "perfectly captured".
    empty = np.einsum("q,qcw->w", tables.weights[0], np.abs(spoiled) ** 2) <= 0.0
    assert np.all(np.isnan(launched.residual[empty]))
    assert np.all(launched.residual[~empty] >= 0.0)


INCIDENT_GUIDE = {"radius": 4e-6, "indices": (1.45, 1.44), "n_r": 24, "knots": 9}
"""A weakly guiding strand for the incident launch: a 4 um core at a 0.01 step.

The vector coupling model admits a solid guide only below its contrast screen,
``Delta = (1.45**2 - 1.44**2) / (2 * 1.45**2) = 6.9e-3`` against ``model_error / 8``
(SEP 31 section 4.3); the 1.45-in-air strand of the other tests is refused.
"""

INCIDENT_BUDGET = 0.2
INCIDENT_MODEL = CouplingModel("vector_paraxial", model_error=INCIDENT_BUDGET)
"""The budget the incident launches are coupled under: the 3 um beam's angular
content, the closure and the contrast together take about a quarter of it."""

INCIDENT_FILTER = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""Silence the active-band basis warning of the incident strand's nine knots.

The interpolated basis departs 1.8e-4 from orthonormal near the band's long edge,
just above ``gram_atol``, where the launched spectrum puts nothing; the gate, weighted
by the launch, passes. The warning itself is asserted elsewhere in this module.
"""


@pytest.fixture(scope="module")
def incident_guide(grid):
    """One HE11 channel of the weakly guiding strand."""
    return general_guide(grid=grid, **INCIDENT_GUIDE)


@pytest.fixture(scope="module")
def incident_pair(grid):
    """The two parities of that channel, declared degenerate."""
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        grid=grid,
        **INCIDENT_GUIDE,
    )


@pytest.fixture(scope="module")
def incident_circular(grid):
    """A declared circular basis of the same parity pair.

    The one fixture here whose `linear.modal_metric` is genuinely **dense**, so the
    launched energy is ``a.H W a`` and not the unweighted ``sum |a|^2``.
    """
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        circular=(0, 1),
        grid=grid,
        **INCIDENT_GUIDE,
    )


def incident_beam(waist, jones=(1.0,)):
    """Return an incident beam carrying the tests' pulse, launched at its waist."""
    return SpatialPulse(
        GaussianPulse(
            wavelength=PULSE["wavelength"], fwhm=PULSE["fwhm"], energy=PULSE["energy"]
        ),
        waist,
        jones=jones,
    )


def launched(guide, grid, incident, **overrides):
    """Return a general modal simulation coupling an incident beam."""
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        wavelength=PULSE["wavelength"],
        incident=incident,
        coupling=overrides.pop("coupling", INCIDENT_MODEL),
        **({"integrator": Multistep(), **({"noise": None, **RUN | overrides})}),
    )


def account(simulation) -> CouplingReport:
    """Return the coupling report a simulation recorded."""
    report = simulation.configuration["launch"]
    assert isinstance(report, CouplingReport)
    return report


@INCIDENT_FILTER
@pytest.mark.parametrize("kind", ["identity", "dense"])
def test_an_incident_launch_keeps_what_the_channels_capture_under_the_runs_metric(
    incident_pair, incident_circular, grid, kind
):
    """`incident=` launches the recovered amplitudes, and its report says so.

    The incident energy is the beam's over its whole plane in the paraxial measure;
    the launched energy is the state's physical energy under the run's own declared
    metric, which on the dense circular basis is not the unweighted sum. Nothing is
    rescaled, so the launched energy falls short of ``energy=`` by what the
    channels miss. The vector model names no uncaptured energy: F and the incident
    measure are not one positive metric. An x-polarised beam enters the even
    parity channel alone, and splits equally between the two circular ones.
    """
    guide = incident_pair if kind == "identity" else incident_circular
    simulation = launched(guide, grid, incident_beam(3e-6))
    report = account(simulation)
    # The beam's closed-form normalisation, then its radial plane's Parseval sum.
    assert report.incident_energy == pytest.approx(PULSE["energy"], rel=1e-10)
    assert 0.5 * report.incident_energy < report.launched_energy
    assert report.launched_energy < report.incident_energy
    assert report.captured is None and report.uncaptured is None
    assert report.budget <= INCIDENT_BUDGET
    coupling = cast("Mapping[str, object]", simulation.configuration["coupling"])
    assert coupling["model"] == "vector_paraxial"
    problem = simulation.translate()
    metric = metric_at(problem.linear.modal_metric, 0.0)
    assert (metric is None) == (kind == "identity")
    assert_allclose(
        float(physical_energy(grid, problem.initial.field0[0], metric=metric)),
        report.launched_energy,
        # The statistic whitens by an eigendecomposition per bin and the report
        # forms the quadratic form directly: rounding over the 65-bin reduction.
        rtol=1e-12,
        atol=0.0,
    )
    state = simulation._field0[:, 0]
    live = np.abs(state[0]) > 0.0
    if kind == "identity":
        # The even channel's E_y and the odd channel's E_x vary as sin(2 theta),
        # so their pairing with a uniformly polarised beam cancels on the
        # azimuthal rule to rounding of the node sum.
        assert np.max(np.abs(state[1])) < 1e-12 * np.max(np.abs(state[0]))
    else:
        assert_allclose(np.abs(state[1][live]), np.abs(state[0][live]), rtol=1e-6)


@INCIDENT_FILTER
def test_an_incident_launch_shares_a_coherent_beam_between_two_channels(
    incident_pair, grid
):
    """A beam polarised between the parity channels fills both, coherently.

    The Jones vector ``(1, 0.5j)`` puts half the even channel's amplitude in the
    odd one, a quarter period behind (conjugated into the internal convention).
    The two channels are a rotation of each other, so the total launched is the
    x-polarised beam's: what is recovered is the field, not a sum of separately
    projected intensities.
    """
    tilted = launched(incident_pair, grid, incident_beam(3e-6, (1.0, 0.5j)))
    straight = launched(incident_pair, grid, incident_beam(3e-6, (1.0, 0.0)))
    state = tilted._field0[:, 0]
    live = np.abs(state[0]) > 0.0
    assert_allclose(state[1][live] / state[0][live], -0.5j, rtol=1e-6)
    assert_allclose(
        account(tilted).launched_energy,
        account(straight).launched_energy,
        rtol=1e-6,
    )


@INCIDENT_FILTER
def test_a_beam_the_channels_cannot_see_is_a_valid_zero_launch(incident_guide, grid):
    """Zero capture by symmetry launches nothing and runs; no division by zero.

    A y-polarised beam against the x-polarised even channel: their pairing is the
    channel's E_y and H_x, which vary as sin(2 theta) and cancel on the azimuthal
    rule. The bound is the square of the node sum's rounding, ``(n_nodes eps)**2``
    of the incident energy, since an energy is quadratic in the amplitude.
    """
    simulation = launched(incident_guide, grid, incident_beam(3e-6, (0.0, 1.0)))
    # The subject is the zero launch; its rounding residue fills the whole window. The
    # check is read at run time, so it can be switched off after translation.
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    report = account(simulation)
    assert simulation.general is not None
    nodes = int(simulation.general.tables.electric.shape[1])
    bound = (nodes * np.finfo(float).eps) ** 2
    assert 0.0 <= report.launched_energy < bound * report.incident_energy
    output = cast("MemoryOutput", simulation.run(progress="none"))
    assert all(np.all(np.isfinite(kept(plane.field))) for plane in output.planes())
    assert "launched share" in str(report)


@INCIDENT_FILTER
def test_the_incident_energy_does_not_depend_on_the_guides_aperture(
    incident_guide, grid
):
    """The beam is normalised over its own plane, not over a guide's nodes.

    The same beam into two strands of different radius carries the same incident
    energy, bin for bin and to the bit, because no node of either guide enters it;
    what each launches differs.
    """
    wider = general_guide(grid=grid, radius=5e-6, indices=(1.45, 1.44), n_r=24, knots=9)
    beam = incident_beam(3e-6)
    narrow_account = account(launched(incident_guide, grid, beam))
    wide_account = account(launched(wider, grid, beam))
    np.testing.assert_array_equal(narrow_account.incident, wide_account.incident)
    np.testing.assert_array_equal(
        narrow_account.reference_index, wide_account.reference_index
    )
    assert narrow_account.launched_energy != wide_account.launched_energy


@INCIDENT_FILTER
def test_the_incident_launch_refuses_what_it_does_not_carry(
    incident_pair, pair_guide, grid
):
    """Two launches at once, pulse keywords beside a beam, beams it cannot place,
    a missing or stray model, and a strongly guiding core the model does not admit."""
    beam = incident_beam(3e-6)
    with pytest.raises(DriverError, match="two ways to say what is launched"):
        launched(incident_pair, grid, beam, channel=0)
    for keyword, value in (
        ("fwhm", 20e-15),
        ("width", 10e-15),
        ("energy", 1e-12),
        ("peak_power", 1.0),
        ("phases", (0.0, 0.0, 1e-30)),
        ("pulses", (GaussianPulse(wavelength=835e-9, fwhm=2e-14, energy=1e-12),)),
        ("shape", "sech"),
    ):
        with pytest.raises(DriverError, match=f"carries its own temporal.*{keyword}"):
            launched(incident_pair, grid, beam, **{keyword: value})
    temporal = beam.temporal
    with pytest.raises(DriverError, match="PlanarPulse"):
        launched(incident_pair, grid, PlanarPulse(temporal, 1e-6, waist_x=1e-6))
    with pytest.raises(DriverError, match="propz=0"):
        launched(incident_pair, grid, SpatialPulse(temporal, 3e-6, propz=-1e-4))
    with pytest.raises(DriverError, match="coupled under a declared model"):
        launched(incident_pair, grid, beam, coupling=None)
    with pytest.raises(DriverError, match="coupled under a declared model"):
        build(incident_pair, grid, coupling=INCIDENT_MODEL)
    # The 1.45-in-air strand fails the weak-guidance screen by name.
    with pytest.raises(DriverError, match="contrast"):
        launched(pair_guide, grid, beam)
    # And without a provider it is one of the general-guide keywords the
    # gas-filled route refuses by name, beside `channel=` and `kerr_form=`.
    with pytest.raises(DriverError, match="incident= belongs to the general modal"):
        Simulation.from_waveguide(
            (MarcatiliMode(125e-6, "He", 1.0),),
            GENERAL_Z_END,
            gas="He",
            pressure=1.0,
            grid=grid,
            incident=beam,
            **{"integrator": Multistep(), **({"noise": None, **PULSE})},
        )
    # A guide whose first region is not on the axis has no core to enter through.
    provider = incident_pair.field_provider
    core, cladding = provider.regions
    shifted = (dataclasses.replace(core, inner_radius=1e-7), cladding)
    with pytest.raises(DriverError, match="does not start there"):
        _core_index(shifted, grid)


class _LossyLaw:
    """A permittivity law with a small imaginary part, for the closure's refusal.

    The forward-field closure that lets a declared `E` alone fix the amplitudes
    rests on the Poynting cross matrix being Hermitian, which is a statement
    about a **lossless** basis. Nothing this package ships is lossy, so the
    refusal needs a law written here; it is put on the region and not on the
    modes, so the solve stays the real one and what is exercised is the check.
    """

    def permittivity(self, omega):
        """Return a complex relative permittivity over the given frequencies."""
        return np.full(np.shape(omega), (1.45 + 1e-9j) ** 2, dtype=complex)

    @property
    def omega_domain(self):
        """Return the validity interval in rad/s: everything."""
        return 0.0, float("inf")

    @property
    def identity(self):
        """Return the law's identity, as every material declares one."""
        return {"model": "test-lossy", "index": "1.45+1e-9j"}


def test_a_complex_permittivity_never_reaches_the_forward_field_closure(guide):
    """Where the launch map's lossless premise is actually enforced.

    Doubling the declared `E` half of the flux pairing is exact because the
    Poynting cross matrix of a **lossless** bound basis is Hermitian. A complex
    permittivity breaks that symmetry, the `H` half is then not the `E` half,
    and the projected amplitudes would be wrong by an amount only a growing
    residual share would show. No guard is needed at the launch, and this test
    says why: the providers refuse a complex material where they solve, so a
    lossy basis never becomes a guide, let alone a table to project into.
    """
    provider = guide.field_provider
    core, cladding = provider.regions
    lossy = dataclasses.replace(
        provider,
        regions=(dataclasses.replace(core, material=_LossyLaw()), cladding),
    )
    sampling = guide.sampling
    assert sampling is not None
    with pytest.raises(ModeError, match="lossless step-index law"):
        ModalGuide.from_provider(lossy, sampling.omega, nodes=sampling.nodes)


@INCIDENT_FILTER
@pytest.mark.medium
def test_a_poorly_captured_beam_launches_only_what_the_guide_takes(
    incident_guide, grid
):
    """A beam much wider than the core enters with the little the guide captures.

    A 40 um beam on the 4 um strand: the channel takes a few per cent of it. The
    run's own entrance plane, through `prop_waveguide`, carries the launched energy
    and not ``energy=``: nothing restores what missed. The report is in the file's
    configuration as well as the simulation's.
    """
    beam = incident_beam(40e-6)
    output = cast(
        "MemoryOutput",
        prop_waveguide(
            (),
            GENERAL_Z_END,
            provider=incident_guide,
            grid=grid,
            wavelength=PULSE["wavelength"],
            incident=beam,
            coupling=INCIDENT_MODEL,
            progress="none",
            **{"integrator": Multistep(), **({"noise": None, **RUN})},
        ),
    )
    report = account(launched(incident_guide, grid, beam))
    assert report.launched_energy < 0.05 * report.incident_energy
    assert report.incident_energy == pytest.approx(PULSE["energy"], rel=1e-10)
    entrance = kept(output.planes()[0].field)
    # Identity metric: the energy statistic is the sum of |a|^2, formed in the
    # same order as the report's up to the frequency-sum association.
    assert_allclose(
        float(physical_energy(grid, entrance)),
        report.launched_energy,
        rtol=1e-12,
        atol=0.0,
    )
    assert output.metadata is not None
    recorded = output.metadata.configuration["launch"]
    assert isinstance(recorded, CouplingReport)
    assert recorded.launched_energy == report.launched_energy


@INCIDENT_FILTER
@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_the_incident_launch_is_the_same_on_every_cpu_receiver(
    incident_guide, grid, backend
):
    """The launch is built before any backend: every CPU backend starts from it.

    Ten half-micrometre steps on each receiver: the stored entrance plane is the
    launched state, a lossless copy, so it agrees bit for bit with the one the
    simulation built, and its energy is the launched energy.
    """
    if backend == "native":
        pytest.importorskip("solprop_native")
    beam = incident_beam(3e-6)
    step = 5e-7
    simulation = launched(
        incident_guide,
        grid,
        beam,
        backend=backend,
        save_planes=[10 * step],
        dz_max=step,
        dz_init=step,
        schedule=FixedSchedule(np.full(10, step)),
    )
    output = cast("MemoryOutput", simulation.run(progress="none"))
    entrance = kept(output.planes()[0].field)
    reference = launched(incident_guide, grid, beam)._field0
    np.testing.assert_array_equal(entrance, reference)
    assert_allclose(
        float(physical_energy(grid, entrance)),
        account(simulation).launched_energy,
        rtol=1e-12,
        atol=0.0,
    )
    assert all(np.all(np.isfinite(kept(plane.field))) for plane in output.planes())


def test_a_multichannel_input_is_refused_by_the_launch(pair_guide, grid):
    """Two pulses at once go through `with_input`, not through the entry point."""
    simulation = build(pair_guide, grid)
    both = np.repeat(simulation._field0[:1], 2, axis=0)
    carried = Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=pair_guide,
        grid=grid,
        **({"integrator": Multistep(), **({"noise": None, **PULSE | RUN})}),
    ).with_input(both)
    # Both rows reach the seam, unchanged: the power weight is the constant one,
    # so `_power_initial` divides by exactly 1.0.
    problem = carried.translate()
    assert np.array_equal(problem.initial.field0[0], both)
    assert np.any(problem.initial.field0[0, 1])
    with pytest.raises(DriverError, match="places one pulse in one channel"):
        _in_channel(both, pair_guide, None)


# -------------------------------------------------------------------------- noise


@pytest.mark.medium
def test_the_same_seed_gives_the_same_realisation_and_the_same_run(guide, grid):
    """The modified shot-noise draw on the general route, and its reproducibility.

    In a Poynting-normalised basis one photon per channel per bin already is an
    energy-normalised true-power amplitude, so the existing draw is added to the
    state as it stands. Nothing multiplies it by an index.
    """
    # On the batch door's prescribed schedule at a fixed order, so the two runs of one
    # seed are one discrete march with the same stored realisation (SEP 25 4.5).
    first = batched_run(guide, grid, noise="input", seed=17)
    again = batched_run(guide, grid, noise="input", seed=17)
    other = batched_run(guide, grid, noise="input", seed=18)
    assert first.noise is not None and again.noise is not None
    assert other.noise is not None
    assert np.array_equal(first.noise.spectrum, again.noise.spectrum)
    assert not np.array_equal(first.noise.spectrum, other.noise.spectrum)
    photons = photons_per_bin(grid, first.noise.spectrum)[0, 0]
    assert_allclose(photons[grid.active_mask], 1.0, rtol=1e-12, atol=0.0)
    # The launch carries it: the seam's noise0 is the realisation unchanged.
    problem = first.translate()
    assert problem.initial.noise0 is not None
    assert np.array_equal(problem.initial.noise0[0], first.noise.spectrum)
    assert problem.initial.noise_model == "input"
    exits = [
        cast("MemoryOutput", run.run(progress="none")).planes()[-1].field
        for run in (first, again, other)
    ]
    bound = simulation_bound(first)
    assert_matched(exits[1], exits[0], bound, what="the same seed's run")
    # Another seed's realisation moves the exit far beyond the rounding budget.
    assert departure(exits[2], exits[0]) > 100 * bound


@pytest.mark.medium
def test_a_declared_basis_draws_once_in_the_reference_basis_and_carries_it(grid):
    """The draw is the reference basis's, pushed through `U.H`, with its own seed.

    A fixed-modulus random-phase draw is not invariant in distribution under
    unitary mixing, so a run in a declared basis must not draw again there: it
    draws once in the orthonormal true-power reference basis, with the same seed
    and the same stream, and carries that one realisation.
    """
    descriptors = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))
    plain = general_guide(descriptors=descriptors, blocks=((0, 1),), grid=grid)
    turned = general_guide(
        descriptors=descriptors, blocks=((0, 1),), circular=(0, 1), grid=grid
    )
    reference = build(plain, grid, noise="input", seed=17)
    declared = build(turned, grid, noise="input", seed=17)
    assert reference.noise is not None and declared.noise is not None
    assert declared.noise.seed == reference.noise.seed == 17
    sampling = turned.sampling
    assert sampling is not None and sampling.basis is not None
    # The same draw, carried: not a second one.
    assert_allclose(
        declared.noise.spectrum,
        np.einsum(
            "mn,npk->mpk",
            sampling.basis.transform.conj().T,
            reference.noise.spectrum,
        ),
        rtol=0.0,
        atol=0.0,
    )
    # The total photon number per bin is exactly what it was; per channel it is
    # redistributed, which is the whole point of carrying rather than redrawing.
    before = photons_per_bin(grid, reference.noise.spectrum).sum(axis=(0, 1))
    after = photons_per_bin(grid, declared.noise.spectrum).sum(axis=(0, 1))
    assert_allclose(after, before, rtol=1e-12, atol=0.0)
    per_channel = photons_per_bin(grid, declared.noise.spectrum)[:, 0]
    assert float(np.max(np.abs(per_channel[:, grid.active_mask] - 1.0))) > 0.1
    # And a run in the provider's own coordinates is bitwise what it was.
    assert np.array_equal(
        reference.noise.spectrum,
        _shot_noise("input", grid, 17, n_modes=2).spectrum,  # pyright: ignore[reportOptionalMemberAccess]
    )


# ------------------------------------------------------------- statistics, output


def test_the_route_records_its_own_names_and_refuses_the_legacy_ones(guide, grid):
    """The record's whitelist: `physical_energy` and `analytic_flux`, not `energy`.

    `energy` and `peak_power` are refused in favour of the route's own names,
    because they promise a sum of coordinate powers and a general basis does not
    make one. The refusal names the basis and the replacement, and never redefines
    the old name.
    """
    output = cast("MemoryOutput", build(guide, grid).run(progress="none"))
    recorded = output.planes()[-1].stats
    assert set(recorded) == {
        "omega0",
        "physical_energy",
        "analytic_flux",
        "fwhm_t",
        "fwhm_t_outer",
    }
    assert float(np.ravel(recorded["physical_energy"])[0]) > 0.0
    assert float(np.ravel(recorded["analytic_flux"])[0]) > 0.0


def test_the_spectral_flux_is_admitted_by_name_and_integrates_to_the_energy(
    guide, grid
):
    """The route's one array statistic: asked for, not defaulted.

    It is `n_omega` wide per plane where the other five are scalars, so it is in
    the whitelist and not in `_general_stats()`; the test above pins the default
    set. Its own claim is the closure: the grid's angular-frequency spacing times
    its sum is the `physical_energy` the same plane records, which is what makes
    it a density of that number and not a second quantity.
    """
    asked = build(
        guide,
        grid,
        stats=StatsSpec(names=("physical_energy", "spectral_flux")),
    )
    plane = cast("MemoryOutput", asked.run(progress="none")).planes()[-1]
    density = np.asarray(plane.stats["spectral_flux"])
    assert density.shape[-1] == grid.n_omega
    assert np.all(density >= 0.0)
    # Rounding alone: the two reductions run over the same bins in a different
    # order, and the end bins the analytic signal does not double carry nothing
    # on a collared grid.
    assert_allclose(
        float(np.sum(density) * grid.d_omega),
        float(np.ravel(plane.stats["physical_energy"])[0]),
        rtol=1e-13,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "name",
    ["energy", "energy_window", "peak_power", "peak_power_window", "peak_power_total"],
)
def test_every_legacy_power_name_is_refused_on_the_general_basis(guide, grid, name):
    """Each by name, with the replacement, at the run loop's admission."""
    asked = build(
        guide,
        grid,
        stats=StatsSpec(
            names=(name,),
            energy_windows=np.array(
                [[0.99 * float(grid.omega0), 1.01 * float(grid.omega0)]]
            ),
        ),
    )
    with pytest.raises(DriverError, match=f"{name} is unavailable for this"):
        asked.run(progress="none")


def test_peak_intensity_is_refused_with_the_companion_text(guide, grid):
    """The one name that needs E and H, refused with advice on what to use."""
    asked = build(guide, grid, stats=StatsSpec(names=("peak_intensity",)))
    with pytest.raises(DriverError, match="use analytic_flux or reconstructed E/H"):
        asked.run(progress="none")


def ordinary_simulation(names):
    """Return a plain GNLSE run: an orthogonal modal basis with no guide at all."""
    return Simulation.from_gnlse(
        0.11,
        1e-4,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=1.0,
        shape="sech",
        n_save=2,
        stats=StatsSpec(names=tuple(names)),
        backend="numpy_ref",
        noise=None,
        integrator=Multistep(),
    )


@pytest.mark.parametrize("name", ["physical_energy", "spectral_flux", "analytic_flux"])
def test_the_general_names_are_refused_on_an_ordinary_modal_run(name):
    """The other direction: an old file's basis carries no certified metric.

    A reviewer who asks a gas-filled capillary for `physical_energy` is told that
    the name needs what that run does not have, rather than given the arithmetic
    of `energy` under a name that promises more.
    """
    with pytest.raises(DriverError, match=f"{name} is unavailable for this modal run"):
        ordinary_simulation((name,)).run(progress="none")


def test_quadrature_error_is_refused_rather_than_recorded_as_nan(guide, grid):
    """No embedded coarse rule to difference, so no alarm exists.

    A `nan` at every plane would read as "no alarm raised" rather than "there is
    no alarm", which is a stronger claim than this route can make.
    """
    asked = build(guide, grid, stats=StatsSpec(names=("energy", "quadrature_error")))
    with pytest.raises(DriverError, match="quadrature_error is unavailable"):
        asked.run(progress="none")


def test_the_backend_refuses_an_inadmissible_name_on_its_own(guide, grid):
    """Asked directly, without the driver's loop (defence in depth)."""
    simulation = build(guide, grid)
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="peak_intensity is unavailable"):
            backend.stats(StatsSpec(names=("peak_intensity",)))
        with pytest.raises(BackendError, match="energy is unavailable"):
            backend.stats(StatsSpec(names=("energy",)))
        assert backend.stats(StatsSpec(names=("physical_energy", "analytic_flux")))
    finally:
        backend.close()


@pytest.mark.parametrize("direction", ["record_is_silent", "record_overclaims"])
def test_a_record_that_misnames_the_basis_is_refused_at_the_run_loop(
    guide, grid, direction
):
    """The lowest door: the record and the problem must agree, in both directions.

    `run_problem(general_problem, backend, handler, metadata=...)` with a record
    carrying no `modal` identity would write a file that `capability_of`
    reads as an **ordinary modal** run, and every legacy accessor would then be
    admitted on a three-component basis -- the one thing the whitelist exists to
    prevent. The other way round, a record claiming a guide for a problem that
    carries no tables would promise a reconstruction that cannot be rebuilt.
    """
    from driver.test_run import NullBackend  # noqa: PLC0415 - a test double

    simulation = build(guide, grid)
    general = simulation.metadata()
    if direction == "record_is_silent":
        problem = simulation.translate()
        metadata = dataclasses.replace(general, modal=None)
        expected = "carries transform.modal_tables and its run record carries"
    else:
        problem = ordinary_simulation(("energy",)).translate()
        metadata = general
        expected = "carries a general modal identity and the problem"
    with pytest.raises(DriverError, match=expected):
        run_problem(
            problem,
            NullBackend(shape=problem.shape),
            MemoryOutput(),
            metadata=metadata,
        )


def test_an_hdf5_file_carries_the_labels_and_the_identities(guide, grid, tmp_path):
    """The run record admits a general modal run at the run loop's own door.

    The other three doors -- `MemoryOutput.to_file`, a scan point's capture and
    its spool, and the batch member that stays refused -- belong to the file and
    are in `tests/output/test_modal_resume.py` and below.
    """
    path = tmp_path / "general.h5"
    simulation = build(guide, grid)
    simulation.run(path, progress="none")
    stored = resume(path).metadata
    assert stored.schema_version == 2
    assert stored.modal is not None
    assert dict(stored.modal)["basis"] == "general_modal"
    labels = [dict(one.label) for one in stored.modes if one.label is not None]
    assert [one["display_label"] for one in labels] == ["HE1,1:even"]
    assert stored.configuration_hash() == simulation.metadata().configuration_hash()


# --------------------------------------------------------------- the batch door

BATCH_Z = GENERAL_Z_END / 10
"""Length a batched general modal test propagates, m: a tenth of the cap."""


def batched_run(guide, grid, **overrides):
    """Return a general modal simulation on an explicit fixed schedule."""
    schedule = FixedSchedule.uniform(0.0, BATCH_Z, BATCH_Z / 4, save_planes=[BATCH_Z])
    settings = {
        "wavelength": PULSE["wavelength"],
        "fwhm": PULSE["fwhm"],
        "shape": PULSE["shape"],
        "energy": PULSE["energy"],
        "schedule": schedule,
        # No order choice on the prescribed schedule: a member and its solo run are
        # one discrete march (SEP 25 4.4).
        "integrator": FIXED_ORDER,
        "save_planes": [BATCH_Z],
        "backend": "numpy_ref",
    }
    simulation = Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        **({"integrator": Multistep(), **({"noise": None, **settings | overrides})}),
    )
    # A run this short has a collar of 2.5e5 1/m, which an explicit Adams history
    # replays in-stage only below 0.7 um steps; nothing reaches it here, so it is
    # split off and integrated exactly.
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step"
    )
    return simulation


@pytest.mark.medium
@pytest.mark.parametrize("runner_factory", [LockstepRunner, NumpyBatchBackend])
def test_a_general_guide_batch_is_what_its_members_run_alone(
    guide, grid, runner_factory
):
    """The batch door: two launches, one guide, each its own solo march.

    The narrow admission rule is that a general-guide batch shares the guide, the table
    and the gauge and its members differ in their launch and their noise seed alone.
    What makes that safe is that the route's source pass already sees one member's
    state: the lockstep runner hands each member its own singleton row, so nothing in
    the pass had to loop. Each member is its solo run's discrete march, so the two
    agree within that march's rounding budget (SEP 25: batch membership is not
    promised bitwise).
    """
    with ExitStack() as cleanup:
        simulations = [
            batched_run(guide, grid, energy=energy, noise="input", seed=13 + index)
            for index, energy in enumerate((1e-12, 2e-12))
        ]
        members, alone = [], []
        for simulation in simulations:
            problem = simulation.translate()
            for bucket in (members, alone):
                backend = select("numpy_ref", problem, grid=grid)
                cleanup.callback(backend.close)
                backend.set_initial(problem.initial.field0, simulation.noise)
                bucket.append(backend)
        runner = runner_factory(members)
        cleanup.callback(runner.close)
        runner.propagate_to(BATCH_Z)
        for owner in alone:
            owner.propagate_to(BATCH_Z)
        spec = simulations[0].translate().stats_spec
        batched_stats = runner.stats(spec)
        bounds = [simulation_bound(simulation) for simulation in simulations]
        for index, (observed, owner) in enumerate(
            zip(runner.observe_members(), alone, strict=True)
        ):
            solo = owner.observe()
            bound = bounds[index]
            assert_matched(observed.field, solo.field, bound, what="field")
            assert_matched(
                observed.noise_phase, solo.noise_phase, bound, what="noise phase"
            )
            for name, value in owner.stats(spec).items():
                assert_matched(
                    batched_stats[name][index],
                    value[0],
                    STATISTIC_FACTOR * bound,
                    what=name,
                )
        # The two launches really are different, far beyond the budget, so the
        # agreement above is a statement and not a tautology.
        first, second = (one.observe().require_field() for one in alone)
        assert departure(second, first) > 100 * max(bounds)


def test_two_guides_never_share_a_batch_group(guide, grid):
    """The descriptor separates them, and the runner refuses a mixed group anyway.

    Two strands of different radius share every entry a structural signature would hash
    -- shape, flags, table extents, response structure. It is `modal_tables.identity`
    that separates them, with the metric's gauge and the responses' model beside it; and
    because a descriptor is only as good as what it names, the group a caller assembles
    by hand is refused by name at the driver and again at the backend.
    """
    one = BatchMember(0, batched_run(guide, grid))
    other = BatchMember(1, batched_run(general_guide(grid=grid, radius=1.6e-6), grid))
    same = BatchMember(2, batched_run(guide, grid, energy=2e-12))
    # Different guides: two groups of one. The same guide with another launch:
    # one group of two, which is exactly the admitted difference.
    assert [len(group) for group in compatible_groups([one, other], 4)] == [1, 1]
    assert [len(group) for group in compatible_groups([one, same], 4)] == [2]
    problems = [member.simulation.translate() for member in (one, other)]
    with pytest.raises(DriverError, match="identical guide/table/gauge identity"):
        refuse_mixed_guides(problems)
    # And a general guide beside a run that is not one is named the same way.
    with pytest.raises(DriverError, match="identical guide/table/gauge identity"):
        refuse_mixed_guides([problems[0], ordinary_simulation(("energy",)).translate()])
    with ExitStack() as cleanup:
        loaded = []
        for problem in problems:
            backend = select("numpy_ref", problem, grid=grid)
            cleanup.callback(backend.close)
            backend.set_initial(problem.initial.field0, None)
            loaded.append(backend)
        with pytest.raises(BackendError, match="identical guide/table/gauge identity"):
            LockstepRunner(loaded)


def test_the_driver_refuses_a_mixed_group_before_a_backend_is_reached(
    guide, grid, monkeypatch
):
    """`refuse_mixed_guides` runs on the driver's own grouping path.

    Defence in depth, and what makes "refused at the driver and again at the
    backend" a true sentence rather than a description of a function nothing
    called. `compatible_groups` separates two guides by signature, so the
    refusal never fires on a group it built; a grouping rule that stopped naming
    the guide identity would silently put two guides in one group, and before
    this nothing between there and the backend would notice. The signature is
    blinded here to produce exactly that state, and the group is refused where
    it is formed -- no backend is constructed at all.
    """
    one = BatchMember(0, batched_run(guide, grid))
    other = BatchMember(1, batched_run(general_guide(grid=grid, radius=1.6e-6), grid))
    monkeypatch.setattr(batch_module, "_signature", lambda member: "one-layout")
    with pytest.raises(DriverError, match="identical guide/table/gauge identity"):
        list(compatible_groups([one, other], 4))
    # And the same blinded signature passes a group that really is one guide, so
    # the refusal above is the guide identity and not the blinding.
    same = BatchMember(2, batched_run(guide, grid, energy=2e-12))
    assert [len(group) for group in compatible_groups([one, same], 4)] == [2]


def test_a_linear_general_guide_groups_by_what_it_does_not_declare(guide, grid):
    """A guide whose regions are all linear has no response leaf to describe.

    The descriptor's `responses` entry is then `None`, and two linear members of
    the same guide still group; a linear guide and a responding one never do,
    which is the same physical distinction as two different radii.
    """
    linear = general_guide(grid=grid, chi3=None)
    members = [
        BatchMember(0, batched_run(linear, grid)),
        BatchMember(1, batched_run(linear, grid, energy=2e-12)),
    ]
    assert [len(group) for group in compatible_groups(members, 4)] == [2]
    identity = _modal_identity(members[0].simulation.translate())
    assert identity is not None
    assert identity["responses"] is None
    assert identity["metric"] is not None
    mixed = [members[0], BatchMember(2, batched_run(guide, grid))]
    assert [len(group) for group in compatible_groups(mixed, 4)] == [1, 1]
    with pytest.raises(DriverError, match="identical guide/table/gauge identity"):
        refuse_mixed_guides([one.simulation.translate() for one in mixed])


@pytest.mark.medium
def test_a_general_guide_batch_isolates_and_retries_a_failed_member(
    guide, grid, tmp_path
):
    """One member fails, the others complete, and a retry recovers it.

    The failure/recovery half of the batch admission rule: no averaging hides
    a member, the failed point keeps its own provenance -- its configuration hash
    and its seed -- and writes no arrays, and `retry_failed=True` finishes it once
    the cause is gone.
    """
    scan = Scan("batch", base_seed=772, energy=[1e-12, 2e-12, 3e-12])
    fail = [True]

    def reduce(output):
        assert output.metadata is not None and output.metadata.modal is not None
        noise = output.metadata.noise
        if fail[0] and noise is not None and noise.seed == scan.point(1).seed:
            raise RuntimeError("deliberate member reduction failure")
        return np.asarray(output["stats/physical_energy"][-1].sum())

    def builder(point):
        return batched_run(
            guide,
            grid,
            energy=float(point.values["energy"]),
            noise="input",
            seed=point.seed,
        )

    options = {
        "output": tmp_path / "batch.h5",
        "executor": Batch(3),
        "reductions": [scans.custom("captured", reduce, (), "J")],
        "progress": "none",
    }
    result = run_scan(scan, builder, **options)
    assert result.status.tolist() == ["done", "failed", "done"]
    assert "deliberate member reduction failure" in str(result.point(1)["error"])
    assert len(str(result.point(1)["configuration_hash"])) == 64
    assert not any(name.startswith("fields/") for name in result.point(1))
    # And it really was one batch of three, not three points that fell back to
    # running on their own: the failure test would pass either way otherwise.
    metadata = cast("Mapping", result.point(1)["metadata"])
    assert metadata["backend"]["settings"]["batch_size"] == 3
    assert metadata["modal"] is not None
    fail[0] = False
    assert (
        run_scan(scan, builder, retry_failed=True, **options).status.tolist()
        == ["done"] * 3
    )


# --------------------------------------------------- the two end-to-end systems

WEAK = {
    "radius": 4e-6,
    "indices": (1.45, 1.44),
    "n_r": 24,
    "knots": 13,
}
"""A weakly guiding step-index fibre: a 4 um core at a 0.01 index step.

Its vector coefficient is within a tenth of a per cent of the scalar
``n2 omega / (c A_eff)``, which is what "weakly guiding" means and what makes the
comparison with the calibrated GNLSE route a statement about the propagation
rather than about the two coefficients.
"""

WEAK_WINDOW = (790e-9, 890e-9)
WEAK_TRANGE = 200e-15
WEAK_FWHM = 40e-15
WEAK_ENERGY = 4e-11
WEAK_RTOL = 1e-6
"""Each run's allowance in the bounds below: its relative error in a phase."""
WEAK_RUN_RTOL = 1e-9
"""The runs' own tolerance. The phases compared are a few milliradians, so a field
error of ``rtol`` is a relative phase error of about ``rtol / 6e-3``: integrated at
1e-9 the circular ratio is within 2.2e-8 of its converged value (measured), far
inside :data:`WEAK_RTOL`."""
WEAK_GUIDANCE = 1e-3
"""Largest departure of the vector coefficient from the scalar one this fixture
may show and still be called weakly guiding. The comparison below rests on the
premise, so it is asserted and not only recorded."""

WEAK_STEPS = 200
"""Most accepted steps a collar-free millimetre of this fibre may take.

``dz_max = dz_init = L/10`` forces at least ten; the default Adams history climbs
from order one and took 166 at :data:`WEAK_RUN_RTOL` (measured). Asserted, so that a
change in what the march costs here is seen.
"""


def weak_energy_bound(steps):
    """Return the largest relative energy drift a collar-free run may show.

    With no absorber the guide is lossless, so the cumulative diagonal flow is a pure
    phase and the isotropic Kerr term conserves ``sum |a|^2`` exactly in the
    continuous limit; what moves it is the integration's error. Each accepted step's
    error is held to :data:`WEAK_RUN_RTOL` of the field, so twice that of the energy,
    over ``steps`` steps. Measured 9.5e-11 over 166 steps, against 3.3e-7.
    """
    return 2.0 * steps * WEAK_RUN_RTOL


WEAK_BASIS = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""Silence the active-band basis warning on the weakly guiding fixture.

Its thirteen knots span both grids' active bands, which reach 1.77 um, and the
interpolated basis is 1.1355e-4 from orthonormal at 1728.5 nm -- just over `gram_atol`,
so every route built on it warns. The **gate**, which is what these comparisons rest on,
is weighted by the 835 nm launch and measures 3.7012e-8. The warning itself is asserted
rather than silenced in
`test_the_interpolated_departure_is_measured_where_the_launch_is` and
`test_a_departure_the_active_band_reaches_is_warned_on_and_not_gated`, so nothing
silenced here goes unchecked.
"""

WEAK_COMMON = {
    "wavelength": 835e-9,
    "fwhm": WEAK_FWHM,
    "energy": WEAK_ENERGY,
    "shape": "gauss",
    "save_planes": [GENERAL_Z_END],
    "backend": "numpy_ref",
    "noise": None,
    "dz_max": GENERAL_Z_END / 10,
    "dz_init": GENERAL_Z_END / 10,
    "rtol": WEAK_RUN_RTOL,
}


@pytest.fixture(scope="module")
def weak():
    """The weakly guiding system: two grids, one guide band, and the coefficients.

    On the former grid geometry, on which the departures quoted here were measured.
    """
    field_grid = FieldGrid.from_limits(
        GENERAL_Z_END, 835e-9, WEAK_WINDOW, WEAK_TRANGE, **LEGACY_GRID
    )
    envelope_grid = EnvelopeGrid.from_limits(
        GENERAL_Z_END, 835e-9, WEAK_WINDOW, WEAK_TRANGE, **LEGACY_GRID
    )
    # One guide serves both routes, so its knots must cover the union of the two
    # grids' active bands: each starts at half its own lowest declared frequency.
    band = (
        min(
            float(field_grid.omega[field_grid.active_mask].min()),
            float(envelope_grid.omega[envelope_grid.active_mask].min()),
        ),
        max(
            float(field_grid.omega[field_grid.active_mask].max()),
            float(envelope_grid.omega[envelope_grid.active_mask].max()),
        ),
    )
    guide = general_guide(grid=field_grid, band=band, **WEAK)
    linear = general_guide(grid=field_grid, band=band, chi3=None, **WEAK)
    sampling = guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        guide.field_provider, sampling.omega, np.array([0.0]), sampling.nodes
    )
    return {
        "field_grid": field_grid,
        "envelope_grid": envelope_grid,
        "band": band,
        "guide": guide,
        "linear": linear,
        "table": table,
        "nodes": sampling.nodes,
        "knots": sampling.omega,
        "profile_error": sampling.profile_error,
    }


def weak_gamma(weak, omega):
    """Return the Poynting vector Kerr coefficient of the weak fibre at `omega`."""
    sample = weak["table"].evaluate(float(omega), 0.0)
    return modal_kerr_coefficient(
        sample.electric[:, :, 0],
        weak["nodes"].weights,
        np.where(weak["nodes"].region_index == 0, GENERAL_CHI3, 0.0),
        float(omega),
        metric=float(np.real(sample.flux_metric[0, 0])),
    )


def run_general(weak, provider, *, changes=None, state=None):
    """Return a completed general modal run, with **no absorber** by default.

    ``from_waveguide`` builds the grid's default collar, as every waveguide entry
    point does, and so does ``prop_gnlse`` (SEP 21), whose runs here take
    ``absorber=False``; a like-for-like comparison turns both off.
    The default-collar case needs no change, so it goes through the public
    function, which is where ``prop_waveguide(provider=...)`` is exercised.
    """
    if changes == {} and state is None:
        return cast(
            "MemoryOutput",
            prop_waveguide(
                (),
                GENERAL_Z_END,
                provider=provider,
                grid=weak["field_grid"],
                progress="none",
                **{"integrator": Multistep(), **({"noise": None, **WEAK_COMMON})},
            ),
        )
    simulation = Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=provider,
        grid=weak["field_grid"],
        **{"integrator": Multistep(), **({"noise": None, **WEAK_COMMON})},
    )
    simulation.settings = dataclasses.replace(
        simulation.settings, **({"absorber": None} if changes is None else changes)
    )
    if state is not None:
        simulation = simulation.with_input(state)
    return cast("MemoryOutput", simulation.run(progress="none"))


def total_energy_drift(run, grid):
    """Return the relative drift of a run's true energy, summed over its channels."""
    planes = run.planes()
    entrance = float(np.sum(energy(grid, planes[0].field)))
    exit_energy = float(np.sum(energy(grid, planes[-1].field)))
    return abs(exit_energy - entrance) / entrance


def straight_peak(run, grid, row=0):
    """Return a dispersion-only run's analytic field and the index of its peak."""
    straight = analytic_field(grid, run.planes()[-1].field[row, 0])
    return straight, int(np.argmax(np.abs(straight) ** 2))


def turned_by(run, straight, peak, grid):
    """Return the nonlinear phase a run accumulated at a linear run's peak."""
    turned = analytic_field(grid, run.planes()[-1].field[0, 0])
    return float(np.angle(turned[peak] / straight[peak]))


def gamma_departure(weak, gamma, omega0):
    """Return ``gamma(omega)/(gamma(omega_0) omega/omega_0) - 1`` over the active band.

    The measured departure of the guide's own frequency-dependent overlap from the
    one coefficient and one shock factor a GNLSE applies. Zero outside the active
    band, where the state is zero and no source reaches.
    """
    grid = weak["field_grid"]
    departure = np.zeros(grid.n_omega)
    for index in np.flatnonzero(grid.active_mask):
        one = float(grid.omega[index])
        departure[index] = weak_gamma(weak, one) / (gamma * one / omega0) - 1.0
    return departure


def surrogate_placements(weak, departure, gamma, omega0, launch):
    """Return what the measured `departure` does to the phase at the pulse's peak.

    The spectrum-weighted mean of `departure` is **not** that number. The phase at
    the peak is a first-order functional of the source, and a departure that is
    nearly antisymmetric in ``omega - omega_0`` -- this one is, to a measured
    4e-7 in its constant term -- cancels in the mean while its curvature does not.
    So the departure is carried through the source itself, on the launched
    spectrum, in the scalar surrogate of the four-wave overlap: the general route
    applies one conjugated profile at the **output** frequency and three at the
    **input** frequencies, and a single scalar whose fourth power reproduces the
    measured ``gamma(omega)`` splits the departure equally between the four.

    A vector overlap is not one scalar -- three Cartesian components and a flux
    metric move differently with frequency -- so which placement it resembles is
    not decided by this surrogate, and the largest of the three is the bound's
    leading term. The run's ``B = gamma P L`` is about 6e-3, so first order is
    the right order.

    Returns
    -------
    dict
        The relative departure of the effective coefficient at the peak, keyed by
        which slots the measured departure was placed in.
    """
    grid = weak["field_grid"]
    shock = np.where(grid.active_mask, grid.omega / omega0, 0.0)
    reference = analytic_field(grid, launch)
    peak = int(np.argmax(np.abs(reference) ** 2))

    def coefficient(inputs, output):
        """Return the effective coefficient at the peak, to first order."""
        filtered = analytic_field(grid, (1.0 + inputs) * launch)
        cubic = np.abs(filtered) ** 2 * filtered
        source = (
            -1j
            * gamma
            * shock
            * (1.0 + output)
            * fourier.forward(np.real(cubic), kind="field")
        )
        return -float(np.imag(analytic_field(grid, source)[peak] / reference[peak]))

    none = np.zeros_like(departure)
    # `gamma` is a quartic functional of the profile, so the scalar that
    # reproduces it enters each of the four slots as its fourth root.
    each = (1.0 + departure) ** 0.25 - 1.0
    plain = coefficient(none, none)
    return {
        name: coefficient(inputs, output) / plain - 1.0
        for name, (inputs, output) in {
            "four slots": (each, each),
            "the output slot": (none, departure),
            "the input slots": (departure, none),
        }.items()
    }


def profile_error_over(weak, band):
    """Return the largest measured profile defect on the knot gaps `band` overlaps.

    ``ModalSampling.profile_error`` is measured over the guide's **whole** band,
    at the entrance knot and at three held-out midpoints, and the guide is solved
    over more than twice the band a run declares. What the comparison below is
    sensitive to is the interpolation where the pulse actually has amplitude, so
    that is measured here, at the midpoint of every knot gap the pulse's band
    reaches into -- where a not-a-knot cubic is worst inside a gap.
    """
    knots, nodes, table = weak["knots"], weak["nodes"], weak["table"]
    low, high = band
    worst = 0.0
    for index in range(knots.size - 1):
        if knots[index + 1] < low or knots[index] > high:
            continue
        middle = float((knots[index] + knots[index + 1]) / 2)
        interpolated = table.evaluate(middle, 0.0)
        aligned = align_modal_sample(
            interpolated,
            weak["guide"].field_provider.solve(middle, 0.0, nodes=nodes),
            nodes,
            degenerate_blocks=table.degenerate_blocks,
        )
        worst = max(worst, _profile_defect(interpolated, aligned))
    return worst


PROFILE_MIDPOINT_MARGIN = 1.05
"""Inflation on a gap's midpoint defect, for the gap's own worst.

The midpoint is where a not-a-knot cubic is worst **inside** a gap, but it is one
sample of a continuum; the equivalent midpoint understatement of a P1 solution's own
error was measured at about 5 %, and the same allowance is made here.
"""


def bound_terms(weak, gamma, omega0, record_property):
    """Measure and record every term of the GNLSE comparison's bound but the runs'.

    Returns
    -------
    dict
        The leading surrogate term, the profile interpolation error over the
        pulse's band, the table's Gram defect and the channel's spline-against-
        generator departure, each dimensionless and non-negative.
    """
    field_grid = weak["field_grid"]
    launch = GaussianPulse(wavelength=835e-9, fwhm=WEAK_FWHM, energy=WEAK_ENERGY).build(
        field_grid
    )[0, 0]
    departure = gamma_departure(weak, gamma, omega0)
    inband = field_grid.omega[field_grid.inband_mask]
    weight = np.abs(launch[field_grid.inband_mask]) ** 2
    offsets = departure[field_grid.inband_mask]
    record_property("weak_gamma_window_spread", float(np.max(np.abs(offsets))))
    record_property(
        "weak_gamma_spectrum_offset", float(np.sum(weight * offsets) / np.sum(weight))
    )
    placements = surrogate_placements(weak, departure, gamma, omega0, launch)
    for name, value in placements.items():
        record_property(f"weak_gamma_surrogate[{name}]", value)
    generator = float(np.real(1j * weak["table"].evaluate(omega0, 0.0).generator[0, 0]))
    spline = float(
        weak["guide"].mode_collection.modes[0].table.beta(np.array([omega0]))[0]
    )
    terms = {
        "gamma(omega)": max(abs(value) for value in placements.values()),
        "profile error": PROFILE_MIDPOINT_MARGIN
        * profile_error_over(weak, (float(inband[0]), float(inband[-1]))),
        "gram defect": weak["table"].gram_defect,
        "beta": abs(spline - generator) / abs(generator),
    }
    for name, value in terms.items():
        record_property(f"weak_bound_term[{name}]", value)
    return terms


@pytest.mark.medium
def test_the_interpolated_departure_is_measured_where_the_launch_is(
    weak, record_property
):
    """The interpolated departure, on the fixture it was measured on, with the gate.

    The identity route's basis is not the identity between the table's knots, and
    the knot Gram defect alone does not measure that. The departure lives at the
    **long-wavelength end** of the sampled band -- 1.1e-4 near 1.73 um -- and a
    run's active mask reaches there while its light never does: inside the
    790--890 nm launch window it is 2.3e-7, and weighted by the pulse's own
    spectrum it is 3.7e-8, which is the relative error declaring ``W = I``
    actually costs this run's ``physical_energy``.

    The bounds, before the numbers: each of the three is positive; they are
    ordered whole-band > in-band > launch-weighted, because each looks at a
    narrower or more heavily discounted support than the last; and the one the
    route gates is inside ``DEFAULT_GRAM_ATOL``, which is what lets this fixture
    run at all. The whole-band figure is above it, which is the warning the next
    test is about.
    """
    field_grid = weak["field_grid"]
    launch = GaussianPulse(wavelength=835e-9, fwhm=WEAK_FWHM, energy=WEAK_ENERGY).build(
        field_grid
    )
    with pytest.warns(GeneralModalBasisWarning, match="1728.5 nm"):
        route = general_modal_route(
            weak["guide"],
            field_grid,
            z_max=GENERAL_Z_END,
            omega_reference=float(field_grid.omega0),
            launch=launch,
        )
    block = interpolated_gram(weak["table"], route.tables)
    whole = metric_departure(block)
    inband = metric_departure(block, occupied=field_grid.inband_mask)
    occupied, weight = occupancy_weight(field_grid, launch)
    weighted = metric_departure(block, occupied=occupied, weight=weight)
    for name, value in (
        ("knot", weak["table"].gram_defect),
        ("whole band", whole),
        ("in band", inband),
        ("launch weighted", weighted),
    ):
        record_property(f"weak_interpolated_departure[{name}]", value)
    assert 0.0 < weighted < inband < whole
    assert route.occupied_departure == pytest.approx(weighted, rel=1e-12)
    assert route.occupied_departure < DEFAULT_GRAM_ATOL
    # And the knot defect, which is all the route used to have, bounds none of
    # them: it is a maximum over the knots across the whole sampled band, so it and
    # the occupied departure are extrema over different supports and either may be
    # the larger. Here the knot defect is 6.97e-8, between the launch-weighted
    # 3.70e-8 and the in-band 2.32e-7, and three orders below the whole-band
    # 1.14e-4 it is routinely mistaken for.
    assert weighted < weak["table"].gram_defect < inband < whole


@pytest.mark.medium
def test_a_departure_the_active_band_reaches_is_warned_on_and_not_gated(
    weak, guide, grid, record_property
):
    """The gate certifies the launch; the warning certifies the run.

    ``occupied_departure`` is weighted by the **launched** spectrum, so a bin the
    launch does not occupy is discounted to nothing however far from orthonormal
    the interpolated basis is there. A broadening run occupies it in full. The
    route therefore also maximises the departure over every bin
    ``grid.active_mask`` admits, weighting nothing, and warns by name when that
    leaves ``gram_atol``.

    Two guides, both of which pass the gate. On the weakly guiding fixture the
    worst bin is 1.1355e-4 at 1728.5 nm and is **inside** the active band, four
    orders above the 3.70e-8 the gate saw, so it warns; on the 1.5 um strand at
    nine knots the whole active band is 2.457e-5, inside the budget, and it does
    not. Nothing is refused either way: a refusal would drop the weak fixture,
    and only the caller knows how far the run broadens.
    """
    field_grid = weak["field_grid"]
    with pytest.warns(GeneralModalBasisWarning, match="more knots"):
        loose = general_modal_route(
            weak["guide"],
            field_grid,
            z_max=GENERAL_Z_END,
            omega_reference=float(field_grid.omega0),
            launch=GaussianPulse(
                wavelength=835e-9, fwhm=WEAK_FWHM, energy=WEAK_ENERGY
            ).build(field_grid),
        )
    block = interpolated_gram(weak["table"], loose.tables)
    record_property("weak_active_departure", loose.active_departure)
    # The warned quantity is the unweighted maximum over the active mask, and on
    # this fixture the guide is sampled over exactly that band, so it is also the
    # whole-band maximum.
    assert loose.active_departure == pytest.approx(
        metric_departure(block, occupied=field_grid.active_mask), rel=1e-12
    )
    assert loose.occupied_departure < DEFAULT_GRAM_ATOL < loose.active_departure
    # The strand: the same measurement, inside the budget, and silent. No
    # `pytest.warns`, so the suite's warnings-are-errors policy is the assertion.
    tight = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )
    record_property("strand_active_departure", tight.active_departure)
    assert tight.active_departure < DEFAULT_GRAM_ATOL
    # The number is on the identity a file carries, quantised to MEASURED_DIGITS
    # as every other measured entry is, so that a re-translation on another BLAS
    # does not refuse its own resume.
    identity = modal_identity(
        tight,
        grid,
        channels=guide.mode_collection.channel_labels,
        has_noise=False,
    )
    metric = identity["metric"]
    assert isinstance(metric, dict)
    assert metric["active_departure"] == pytest.approx(tight.active_departure, rel=1e-5)


@WEAK_BASIS
@pytest.mark.slow
def test_a_weakly_guiding_channel_agrees_with_the_calibrated_gnlse(
    weak, record_property
):
    """One HE11 channel against `prop_gnlse` at the matched Poynting gamma.

    Both routes are given the **same** channel: the general one propagates the
    guide's three-component reference, the GNLSE one takes that channel's own
    index table as its fibre and the vector coefficient as its gamma, so the two
    linear operators are the same function of frequency and only the nonlinear
    model differs. Each run is divided by its own dispersion-only run -- a guide
    whose regions declare no response, and gamma = 0 -- which removes the frame,
    the grid and the dispersion and leaves the accumulated nonlinear phase.
    **Neither route carries an absorber**, which is what makes the comparison
    like for like; what the collar does is measured at the end and recorded.

    **The bound, assembled before the nonlinear runs**, is the sum of five
    measured terms and the controller's own:

    * the departure of the guide's frequency-dependent overlap from the GNLSE's
      one coefficient and one shock factor, carried through the source to the
      phase at the peak (:func:`surrogate_placements`), taken at the largest of
      the three placements a scalar surrogate cannot choose between. This is
      physics the scalar model does not carry, not an error in either route;
    * the guide's profile interpolation error **over the pulse's own band**,
      inflated for a gap's worst against its midpoint;
    * the table's measured sampled Gram defect, the identity metric's error row;
    * the departure of the channel's index-table spline from the table's own
      generator, which the two routes share and which therefore cancels in each
      one's ratio to its dispersion-only partner -- recorded and carried anyway,
      because it is the only other thing the two routes could disagree about;
    * the two grids' different sampling of the peak, measured on the
      dispersion-only runs before the nonlinear ones are started;
    * twice the controller's `rtol`, one for each route.
    """
    field_grid, envelope_grid = weak["field_grid"], weak["envelope_grid"]
    omega0 = float(field_grid.omega0)
    gamma = weak_gamma(weak, omega0)
    area = channel_area(weak["table"].evaluate(omega0, 0.0), weak["nodes"], 0)
    scalar = n2_from_chi3(GENERAL_CHI3, 1.45) * omega0 / (C * area)
    record_property("weak_gamma_vector", gamma)
    record_property("weak_gamma_scalar", scalar)
    difference = abs(gamma - scalar) / gamma
    record_property("weak_gamma_difference", difference)
    # The premise the comparison rests on, asserted and not only recorded.
    assert difference < WEAK_GUIDANCE

    terms = bound_terms(weak, gamma, omega0, record_property)
    channel = weak["guide"].mode_collection.modes[0]

    # The two dispersion-only runs first: they fix the peak each route's phase is
    # read at, and the difference between the two peaks is a term of the bound.
    common = {"progress": "none", **WEAK_COMMON}
    general_linear = run_general(weak, weak["linear"])
    gnlse_linear = cast(
        "MemoryOutput",
        prop_gnlse(
            0.0,
            GENERAL_Z_END,
            grid=envelope_grid,
            fibre=channel,
            shock=True,
            **{
                "integrator": Multistep(),
                "absorber": False,
                **({"noise": None, **common}),
            },
        ),
    )
    straight, peak = straight_peak(general_linear, field_grid)
    envelope_straight, envelope_peak = straight_peak(gnlse_linear, envelope_grid)
    power = float(np.abs(straight[peak]) ** 2)
    envelope_power = float(np.abs(envelope_straight[envelope_peak]) ** 2)
    sampling = abs(power - envelope_power) / power
    record_property("weak_peak_power_difference", sampling)

    bound = sum(terms.values()) + sampling + 2 * WEAK_RTOL
    record_property("weak_phase_bound", bound)
    assert 0.0 < bound < 1e-2

    general = run_general(weak, weak["guide"])
    gnlse = cast(
        "MemoryOutput",
        prop_gnlse(
            gamma,
            GENERAL_Z_END,
            grid=envelope_grid,
            fibre=channel,
            shock=True,
            **{
                "integrator": Multistep(),
                "absorber": False,
                **({"noise": None, **common}),
            },
        ),
    )
    turned = turned_by(general, straight, peak, field_grid)
    expected = turned_by(gnlse, envelope_straight, envelope_peak, envelope_grid)
    # Both routes have to be turning by about -gamma P L, or the comparison is of
    # two numbers that are both zero.
    assert abs(expected + gamma * envelope_power * GENERAL_Z_END) < 0.05 * abs(expected)
    record_property("weak_phase_general", turned)
    record_property("weak_phase_gnlse", expected)
    record_property("weak_phase_difference", abs(turned - expected) / abs(expected))
    assert_allclose(turned, expected, rtol=bound, atol=0.0)

    # True energy, conserved: with no absorber the guide is lossless, the linear
    # flow is a pure phase and the isotropic Kerr term conserves energy exactly,
    # so what is left is the integration's error (see weak_energy_bound).
    steps = general.planes()[-1].step_history.n_steps
    record_property("weak_general_steps", steps)
    assert steps <= WEAK_STEPS
    entrance = float(energy(field_grid, kept(general.planes()[0].field))[0, 0])
    exit_energy = float(energy(field_grid, kept(general.planes()[-1].field))[0, 0])
    record_property("weak_energy_drift", abs(exit_energy - entrance) / entrance)
    assert_allclose(exit_energy, entrance, rtol=weak_energy_bound(steps), atol=0.0)


@WEAK_BASIS
@pytest.mark.slow
def test_what_the_in_stage_collar_costs_the_same_comparison(weak, record_property):
    """The collar, not the route, sets the step and moves the answer.

    Recorded, not asserted against the GNLSE: with the default in-stage collar the
    same comparison reads 5.2e-3 instead of 1.0e-4, and the difference is the
    **spectral** collar filtering the self-phase-modulation sidebands of the
    field-grid run, which the envelope run has nothing equivalent to. It also
    takes 2.6 times as many steps (431 against 166), because the collar's rate --
    and so its stable step -- is set by the length the grid was built for and by
    the time window, and a general modal step is about a hundred times a fused
    one. Choosing ``absorber_scheme="split_step"`` reaches the same answer in the
    collar-free step count, which is the evidence that the step is the collar's
    doing and the answer the collar's arithmetic.
    """
    field_grid, envelope_grid = weak["field_grid"], weak["envelope_grid"]
    channel = weak["guide"].mode_collection.modes[0]
    common = {"progress": "none", **WEAK_COMMON}
    gnlse = cast(
        "MemoryOutput",
        prop_gnlse(
            weak_gamma(weak, float(field_grid.omega0)),
            GENERAL_Z_END,
            grid=envelope_grid,
            fibre=channel,
            shock=True,
            **{
                "integrator": Multistep(),
                "absorber": False,
                **({"noise": None, **common}),
            },
        ),
    )
    gnlse_linear = cast(
        "MemoryOutput",
        prop_gnlse(
            0.0,
            GENERAL_Z_END,
            grid=envelope_grid,
            fibre=channel,
            shock=True,
            **{
                "integrator": Multistep(),
                "absorber": False,
                **({"noise": None, **common}),
            },
        ),
    )
    envelope_straight, envelope_peak = straight_peak(gnlse_linear, envelope_grid)
    expected = turned_by(gnlse, envelope_straight, envelope_peak, envelope_grid)

    collar_free = run_general(weak, weak["guide"]).planes()[-1].step_history.n_steps
    for name, changes in (
        ("the default collar", {}),
        ("split_step", {"absorber_scheme": "split_step"}),
    ):
        run = run_general(weak, weak["guide"], changes=changes)
        reference = run_general(weak, weak["linear"], changes=changes)
        straight, peak = straight_peak(reference, field_grid)
        turned = turned_by(run, straight, peak, field_grid)
        steps = run.planes()[-1].step_history.n_steps
        record_property(f"weak_phase_with[{name}]", turned)
        record_property(f"weak_steps_with[{name}]", steps)
        # The collar absorbs, so this is not the collar-free run's arithmetic
        # drift: it is recorded beside it rather than bounded.
        record_property(
            f"weak_energy_drift_with[{name}]", total_energy_drift(run, field_grid)
        )
        record_property(
            f"weak_phase_difference_with[{name}]",
            abs(turned - expected) / abs(expected),
        )
        # The collar costs steps; the split-step scheme applies the same rates and
        # costs none of them.
        assert steps > 2 * collar_free if not changes else steps == collar_free


@pytest.mark.slow
def test_the_collar_and_not_the_route_sets_the_step_on_every_route():
    """The in-stage temporal collar does the same to `prop_capillary`.

    The same grid, the same length and the same step cap: the general route is not
    what makes a short run take four times its minimum number of steps.
    """
    grid = FieldGrid.from_limits(GENERAL_Z_END, 835e-9, WEAK_WINDOW, WEAK_TRANGE)
    counts = []
    # optmat's argon index covers this grid's active band, to 1774 nm (the former
    # registry's stopped at 1 um and warned; SEP 3), so the runs are quiet.
    for collar in (True, False):
        simulation = Simulation.from_capillary(
            50e-6,
            GENERAL_Z_END,
            "Ar",
            1.0,
            grid=grid,
            plasma=False,
            **{"noise": None, **WEAK_COMMON},
        )
        if not collar:
            simulation.settings = dataclasses.replace(
                simulation.settings, absorber=None
            )
        output = cast("MemoryOutput", simulation.run(progress="none"))
        counts.append(output.planes()[-1].step_history.n_steps)
    collared, free = counts
    assert free <= WEAK_STEPS
    assert collared > 3 * free


ROTATION_BOUND = 1e-12
"""Relative bound on rotating a degenerate pair's launch and rotating it back.

Predeclared, and the same number and the same reason as the bound on two
differently grouped sums over one quadrature: the two HE11 parities are exactly
degenerate and the Kerr tensor is isotropic, so the propagation commutes with the
rotation in exact arithmetic and what is left is rounding over the step loop. The
test checks the premises it rests on -- an identical propagation constant and a
cross flux inside the table's own Gram defect -- before it asserts the bound.
"""


@pytest.fixture(scope="module")
def weak_pair(weak):
    """The two parities of the weak fibre's HE11 channel, declared degenerate."""
    blocks = ((0, 1),)
    descriptors = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))
    field_grid, band = weak["field_grid"], weak["band"]
    pair = general_guide(
        descriptors=descriptors, blocks=blocks, grid=field_grid, band=band, **WEAK
    )
    linear = general_guide(
        descriptors=descriptors,
        blocks=blocks,
        grid=field_grid,
        band=band,
        chi3=None,
        **WEAK,
    )
    sampling = pair.sampling
    assert sampling is not None
    table = sample_modal_table(
        pair.field_provider,
        sampling.omega,
        np.array([0.0]),
        sampling.nodes,
        degenerate_blocks=blocks,
    )
    pulse = GaussianPulse(wavelength=835e-9, fwhm=WEAK_FWHM, energy=WEAK_ENERGY).build(
        field_grid
    )
    zero = np.zeros_like(pulse)
    theta = 0.37
    return {
        "guide": pair,
        "linear": linear,
        "table": table,
        "nodes": sampling.nodes,
        "knots": sampling.omega,
        "pulse": pulse,
        "theta": theta,
        "states": {
            # Linearly polarised, so `2|E|^2 E` and `(E.E) E*` are each rotation
            # invariant and only their sum is tested by the rotation.
            "even": np.concatenate([pulse, zero], axis=0),
            "rotated": np.concatenate(
                [np.cos(theta) * pulse, np.sin(theta) * pulse], axis=0
            ),
            # Circular: `E.E` vanishes at every node in the weak-guidance limit,
            # so `(E.E) E*` drops out and the coefficient falls to 2/3.
            "circular": np.concatenate([pulse, 1j * pulse], axis=0) / np.sqrt(2.0),
        },
    }


@pytest.fixture(scope="module")
def pair_runs(weak, weak_pair):
    """Propagate each launch of the parity pair, with and without its response."""
    return {
        name: {
            kind: run_general(weak, weak_pair[kind], state=state)
            for kind in ("guide", "linear")
        }
        for name, state in weak_pair["states"].items()
    }


@WEAK_BASIS
@pytest.mark.slow
def test_the_parity_pair_is_invariant_under_a_rotation_of_the_launch(
    weak, weak_pair, pair_runs, record_property
):
    """Rotate the launch by an angle, rotate the result back, compare to rounding.

    The two parities of one HE11 channel span a plane the guide is isotropic in:
    launching `cos(theta)` and `sin(theta)` into them and rotating the exit back
    by `-theta` has to reproduce the run launched into the even parity alone. A
    route that mixed the parities wrongly -- a cross term with the wrong sign, a
    projection that is not the reciprocal one -- fails this and passes a
    single-channel test.

    The launch is **linearly** polarised, so `2|E|^2 E` and `(E.E) E*` are each
    rotation-invariant and only their sum is tested here; the circular launch
    below is what separates them.

    True energy is conserved along both runs, to the derived collar-free bound.
    """
    field_grid = weak["field_grid"]
    sample = weak_pair["table"].evaluate(float(field_grid.omega0), 0.0)
    split = abs(sample.generator[0, 0] - sample.generator[1, 1]) / abs(
        sample.generator[0, 0]
    )
    cross = abs(sample.flux_metric[0, 1])
    record_property("pair_beta_split", float(split))
    record_property("pair_cross_flux", float(cross))
    assert split < ROTATION_BOUND
    assert cross < weak_pair["table"].gram_defect

    theta = weak_pair["theta"]
    rotated_out = pair_runs["rotated"]["guide"]
    straight = pair_runs["even"]["guide"].planes()[-1].field[:, 0]
    rotated = rotated_out.planes()[-1].field[:, 0]
    back = np.array(
        [
            np.cos(theta) * rotated[0] + np.sin(theta) * rotated[1],
            -np.sin(theta) * rotated[0] + np.cos(theta) * rotated[1],
        ]
    )
    scale = float(np.max(np.abs(straight)))
    assert scale > 0.0
    residual = float(np.max(np.abs(back - straight))) / scale
    record_property("pair_rotation_residual", residual)
    assert_allclose(back, straight, rtol=0.0, atol=ROTATION_BOUND * scale)
    # The rotation is not trivially satisfied: the rotated launch really does put
    # a comparable amplitude in the second channel.
    assert float(np.max(np.abs(rotated[1]))) > 0.3 * scale

    drift = total_energy_drift(rotated_out, field_grid)
    record_property("pair_energy_drift", drift)
    steps = rotated_out.planes()[-1].step_history.n_steps
    assert drift < weak_energy_bound(steps)


CIRCULAR_RATIO = 2.0 / 3.0
"""What `gamma_circ / gamma_lin` is for an isotropic Kerr tensor.

The analytic-signal polarisation is
``P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4``. For a linearly polarised field the
two terms are both ``|E|^2 E`` and the coefficient carries ``2 + 1 = 3`` quarters;
for a circular one ``E.E`` vanishes and only ``2`` quarters are left. The ratio is
therefore exactly 2/3 for a field that is circular **at every node**, which the
two HE11 parities of a weakly guiding fibre are to their weak-guidance error: the
even parity is x-polarised and the odd one y-polarised up to a longitudinal
component of order the index step. The test measures that departure on the
guide's own table before it propagates anything, and it is the bound's leading
term.
"""


@pytest.fixture(scope="module")
def circular_split(weak, weak_pair):
    """The 2:1 split's coefficients and its bound, measured before any propagation.

    One derivation, used by the superposition measurement and by the run in the
    declared circular basis: the two are the same physical statement, so a second
    copy of the bound could drift from the first.
    """
    field_grid = weak["field_grid"]
    omega0 = float(field_grid.omega0)
    sample = weak_pair["table"].evaluate(omega0, 0.0)
    chi3 = np.where(weak_pair["nodes"].region_index == 0, GENERAL_CHI3, 0.0)

    def coefficient(amplitudes):
        """Return the vector Kerr coefficient of a unit-power superposition."""
        weights = np.asarray(amplitudes, dtype=complex)
        flux = float(np.real(weights.conj() @ sample.flux_metric @ weights))
        return modal_kerr_coefficient(
            np.einsum("qcm,m->qc", sample.electric, weights),
            weak_pair["nodes"].weights,
            chi3,
            omega0,
            metric=flux,
        )

    linear_gamma = coefficient([1.0, 0.0])
    circular_gamma = coefficient(np.array([1.0, 1j]) / np.sqrt(2.0))
    declared = abs(circular_gamma / linear_gamma - CIRCULAR_RATIO) / CIRCULAR_RATIO
    profile = PROFILE_MIDPOINT_MARGIN * profile_error_over(
        weak_pair | {"guide": weak_pair["guide"]},
        (
            float(field_grid.omega[field_grid.inband_mask][0]),
            float(field_grid.omega[field_grid.inband_mask][-1]),
        ),
    )
    propagation = 2.0 * (profile + weak_pair["table"].gram_defect + 2 * WEAK_RTOL)
    return {
        "linear_gamma": linear_gamma,
        "circular_gamma": circular_gamma,
        "declared": declared,
        "propagation": propagation,
        "bound": declared + propagation,
    }


@WEAK_BASIS
@pytest.mark.slow
def test_a_circular_launch_pins_the_kerr_tensors_two_to_one_split(
    weak, weak_pair, pair_runs, circular_split, record_property
):
    """`(E.E) E*` is a third of the linear coefficient and none of the circular one.

    The rotation test above launches a common phase into the two parities, which
    is a linearly polarised field: ``2|E|^2 E`` and ``(E.E) E*`` are then each
    rotation-invariant and only ``alpha + beta`` is tested. A relative phase of
    ``i`` makes the field circular, ``E.E`` vanishes, and the accumulated
    nonlinear phase falls to **2/3** of the linear launch's -- which the two terms
    can only do if each carries its own share of the isotropic tensor.

    **The bound, assembled before the runs**, is the guide's own measured
    departure from a pointwise circular field -- the weak-guidance error, which is
    physics and not arithmetic -- plus twice the propagation terms of the GNLSE
    comparison: the table's Gram defect, its profile interpolation error over the
    pulse's band and the controller's `rtol` on each of the two runs. Twice, for
    the same reason the profile term is inflated there: each is a measurement at
    a few frequencies rather than a bound over the band.
    """
    field_grid = weak["field_grid"]
    linear_gamma = circular_split["linear_gamma"]
    circular_gamma = circular_split["circular_gamma"]
    bound = circular_split["bound"]
    propagation = circular_split["propagation"]
    record_property("pair_gamma_linear", linear_gamma)
    record_property("pair_gamma_circular", circular_gamma)
    record_property("pair_gamma_ratio_from_the_table", circular_gamma / linear_gamma)
    record_property("pair_weak_guidance_departure", circular_split["declared"])
    record_property("pair_circular_bound", bound)
    assert 0.0 < bound < 1e-2

    phases = {}
    for name in ("even", "circular"):
        straight, peak = straight_peak(pair_runs[name]["linear"], field_grid)
        phases[name] = turned_by(pair_runs[name]["guide"], straight, peak, field_grid)
    ratio = phases["circular"] / phases["even"]
    record_property("pair_phase_linear", phases["even"])
    record_property("pair_phase_circular", phases["circular"])
    record_property("pair_phase_ratio", ratio)
    record_property(
        "pair_phase_ratio_departure",
        abs(ratio - CIRCULAR_RATIO) / CIRCULAR_RATIO,
    )
    # Both launches carry the same total power, so the ratio of the phases is the
    # ratio of the coefficients and nothing else.
    assert phases["even"] < 0.0
    assert_allclose(ratio, CIRCULAR_RATIO, rtol=bound, atol=0.0)
    # And the propagated ratio is the table's own, to the propagation terms alone.
    assert_allclose(ratio, circular_gamma / linear_gamma, rtol=propagation, atol=0.0)

    drift = total_energy_drift(pair_runs["circular"]["guide"], field_grid)
    record_property("pair_circular_energy_drift", drift)
    steps = pair_runs["circular"]["guide"].planes()[-1].step_history.n_steps
    assert drift < weak_energy_bound(steps)


# --------------------------------------------- the same pair in the circular basis


BASIS_AGREEMENT_BOUND = 1e-10
"""Relative bound on the same launch propagated in two bases, declared first.

The general route is **covariant** under a declared unitary change of basis: the
reference transforms as ``e' = e U``, so the synthesis and the projection
transform as ``S' = S U`` and ``Q' = U.H Q``, the generator of an exactly
degenerate block is unchanged, and the isotropic tensor acts on the physical
three-component field, which no change of channel coordinates touches. The two
runs are therefore the same computation written in two coordinate systems, and in
exact arithmetic ``d(z) = U.H c(z)`` at every plane. What is left is rounding: the
source pass reduces over 288 nodes and 129 fine bins at each of at most
``WEAK_STEPS`` accepted steps, and the complex basis routes those same sums
through complex multiplies, which roughly doubles the operation count of the two
maps. The rotation test above measures 5.7e-16 for the real-orthogonal case; 1e-10
is four orders above any plausible growth of that, and it leaves room for the two
adaptive trajectories not being bit-identical.
"""


@pytest.fixture(scope="module")
def circular_basis(weak):
    """The same fibre and pair, declared in the circular basis, with its runs."""
    field_grid, band = weak["field_grid"], weak["band"]
    descriptors = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))
    shared = {
        "descriptors": descriptors,
        "blocks": ((0, 1),),
        "grid": field_grid,
        "band": band,
        "circular": (0, 1),
        **WEAK,
    }
    guide = general_guide(**shared)
    linear = general_guide(chi3=None, **shared)
    pulse = GaussianPulse(wavelength=835e-9, fwhm=WEAK_FWHM, energy=WEAK_ENERGY).build(
        field_grid
    )
    zero = np.zeros_like(pulse)
    # One channel, the right-handed partner, which is the `(e_even + i e_odd)`
    # superposition the linear-basis run launches as a two-channel state.
    state = np.concatenate([zero, pulse], axis=0)
    return {
        "guide": guide,
        "linear": linear,
        "state": state,
        "runs": {
            name: run_general(weak, member, state=state)
            for name, member in (("guide", guide), ("linear", linear))
        },
    }


@WEAK_BASIS
@pytest.mark.slow
def test_a_single_circular_channel_reproduces_the_two_to_one_split(
    weak, pair_runs, circular_basis, circular_split, record_property
):
    """The 2:1 split again, from a route run launching **one** circular channel.

    The measurement above launches the circular field as a superposition of the
    two parities and reads the ratio of two propagated phases. Here the guide
    itself declares the circular basis, so the same physical field is **one**
    channel of the run: the off-diagonal drive that the superposition carried in
    the state is now carried by the route's own transformed maps, and the split is
    reproduced only if the source pass really is covariant.

    The bound is the one declared for the superposition measurement -- the guide's
    departure from a pointwise circular field plus twice the propagation terms --
    because it is the same physical statement; what has changed is where the
    off-diagonal lives.
    """
    field_grid = weak["field_grid"]
    guide = circular_basis["guide"]
    labels = [label.display_label for label in guide.mode_collection.channel_labels]
    assert labels == ["HE1,1:left", "HE1,1:right"]
    record_property("circular_basis_labels", ",".join(labels))

    bound = circular_split["bound"]
    record_property("circular_basis_bound", bound)
    assert 0.0 < bound < 1e-2
    straight, peak = straight_peak(circular_basis["runs"]["linear"], field_grid, row=1)
    turned = analytic_field(
        field_grid, circular_basis["runs"]["guide"].planes()[-1].field[1, 0]
    )
    circular_phase = float(np.angle(turned[peak] / straight[peak]))

    reference, reference_peak = straight_peak(pair_runs["even"]["linear"], field_grid)
    linear_phase = turned_by(
        pair_runs["even"]["guide"], reference, reference_peak, field_grid
    )
    ratio = circular_phase / linear_phase
    record_property("circular_basis_phase", circular_phase)
    record_property("circular_basis_phase_ratio", ratio)
    record_property(
        "circular_basis_ratio_departure",
        abs(ratio - CIRCULAR_RATIO) / CIRCULAR_RATIO,
    )
    assert linear_phase < 0.0
    assert_allclose(ratio, CIRCULAR_RATIO, rtol=bound, atol=0.0)
    # And the propagated ratio is the table's own, to the propagation terms alone.
    assert_allclose(
        ratio,
        circular_split["circular_gamma"] / circular_split["linear_gamma"],
        rtol=circular_split["propagation"],
        atol=0.0,
    )

    drift = total_energy_drift(circular_basis["runs"]["guide"], field_grid)
    record_property("circular_basis_energy_drift", drift)
    steps = circular_basis["runs"]["guide"].planes()[-1].step_history.n_steps
    assert drift < weak_energy_bound(steps)


@WEAK_BASIS
@pytest.mark.slow
def test_the_two_bases_agree_on_the_same_launch(
    weak_pair, pair_runs, circular_basis, record_property
):
    """The linear-basis run, transformed, is the circular-basis run.

    ``d = U.H c``: the same physical launch, propagated in two declared bases, is
    one computation in two coordinate systems. Asserted at the exit plane on the
    field itself, not on a reduction of it, so a route that got the off-diagonal
    right only in the aggregate fails.
    """
    sampling = circular_basis["guide"].sampling
    assert sampling is not None and sampling.basis is not None
    transform = sampling.basis.transform
    # The circular-basis launch and the transformed linear-basis one are the same
    # state, which is the premise the comparison rests on.
    carried = np.einsum(
        "mn,npk->mpk", transform.conj().T, weak_pair["states"]["circular"]
    )
    entrance = float(np.max(np.abs(carried - circular_basis["state"])))
    record_property("circular_basis_launch_residual", entrance)
    assert entrance <= BASIS_AGREEMENT_BOUND * float(np.max(np.abs(carried)))

    linear = pair_runs["circular"]["guide"].planes()[-1].field
    declared = circular_basis["runs"]["guide"].planes()[-1].field
    expected = np.einsum("mn,npk->mpk", transform.conj().T, linear)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    residual = float(np.max(np.abs(declared - expected))) / scale
    record_property("circular_basis_exit_residual", residual)
    assert_allclose(declared, expected, rtol=0.0, atol=BASIS_AGREEMENT_BOUND * scale)
    # The launch really is one channel in this basis: the other row is empty.
    assert float(np.max(np.abs(circular_basis["state"][0]))) == 0.0
    assert float(np.max(np.abs(declared[0]))) > 0.0
    record_property(
        "circular_basis_partner_fraction",
        float(np.max(np.abs(declared[0]))) / scale,
    )


# ------------------------------------------------------- the delayed regional term


@pytest.mark.medium
def test_a_region_that_declares_the_tensor_raman_propagates_with_it(grid):
    """End to end, ten steps: the delayed term reaches the run and changes it.

    The region's one `chi3` is shared -- the instantaneous tensor keeps `1 - f_R` and
    the two delayed channels carry `f_R` -- so a run with the delayed response declared
    is **not** the same as one without, and the difference is above the stepper's own
    tolerance by orders of magnitude. The identity the run itself certifies is
    unchanged: the basis, the metric and the launched energy are what they were.
    """
    declared = SilicaTensorResponse()
    guide = general_guide(grid=grid, raman=declared)
    assert guide.regions[0].raman is declared
    route = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )
    assert route.responses is not None
    assert route.responses.n_terms == 2
    assert route.responses.model_id == "isotropic-kerr-real+silica-tensor-raman"
    # The instantaneous tensor is scaled by 1 - f_R: one susceptibility, shared.
    plain = general_modal_route(
        general_guide(grid=grid),
        grid,
        z_max=GENERAL_Z_END,
        omega_reference=float(grid.omega0),
    )
    assert plain.responses is not None
    assert_allclose(
        route.responses.instantaneous_tensor,
        (1.0 - declared.fraction) * plain.responses.instantaneous_tensor,
        rtol=1e-15,
        atol=0.0,
    )
    with_raman = cast("MemoryOutput", build(guide, grid).run(progress="none"))
    without = cast(
        "MemoryOutput", build(general_guide(grid=grid), grid).run(progress="none")
    )
    left = np.asarray(with_raman.planes()[-1].field)
    right = np.asarray(without.planes()[-1].field)
    scale = float(np.max(np.abs(right)))
    departure = float(np.max(np.abs(left - right))) / scale
    # The bound is the stepper's own rtol, 1e-8: anything at or below it would be
    # a step-control difference and not a physical one.
    assert departure > 1e-8
    # And the identity is unchanged: same basis, same metric, same launch.
    stored = with_raman.metadata
    assert stored is not None and stored.modal is not None
    modal = cast("dict[str, Any]", dict(stored.modal))
    assert modal["basis"] == "general_modal"
    assert modal["metric"]["kind"] == "identity"
    assert modal["responses"]["model_id"] == "isotropic-kerr-real+silica-tensor-raman"
    identity = modal["guide"]["regions"][0]["raman"]
    assert identity["fraction"] == pytest.approx(0.245)
    # The default response is optmat's tensor bundle since SEP 3's Phase 4a (Q8),
    # which the identity names by its bundle and digest rather than Lin and Agrawal's
    # declared parameters (tau1 12.2 fs); the kernels agree to 4e-16.
    assert identity["bundle"] == "silica_bundle/tensor"


@pytest.mark.medium
def test_fixed_steps_beyond_the_collar_are_refused_at_translation():
    """L/20 on the step-index strand breaks the collar's limit.

    Bounds, stated first. The in-stage temporal collar is an explicit term (see
    docs/dev/decisions.md): an explicit step is stable only while ``h alpha / 2``
    stays inside the method's real stability interval -- 0.0877 for the default
    Adams history, 7.0e-7 m on this grid's collar -- and twenty replayed steps of
    the millimetre are 5e-5 m, seventy times over. The translation refuses the
    schedule before the run, naming the limit and the step. The same schedule with
    the collar integrated exactly (``absorber_scheme="split_step"``) or with no
    collar runs and stays finite, so the limit that is broken is the collar's and
    not the route's.
    """
    length, steps = GENERAL_Z_END, 20
    guide = general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),)
    )

    def simulation():
        return Simulation.from_waveguide(
            (),
            length,
            provider=guide,
            grid=general_grid(),
            wavelength=835e-9,
            fwhm=30e-15,
            energy=1e-9,
            dz_max=length / steps,
            dz_init=length / steps,
            schedule=FixedSchedule(np.full(steps, length / steps)),
            n_save=2,
            backend="numpy_ref",
            noise=None,
            integrator=Multistep(),
        )

    with pytest.raises(TranslationError, match=r"stable .* only up to 6\.96\d*e-07 m"):
        simulation().translate()
    with pytest.raises(TranslationError, match="would diverge"):
        simulation().run(progress="none")
    for settings in ({"absorber_scheme": "split_step"}, {"absorber": None}):
        stable = simulation()
        # The subject is the collar's limit; the short window's light reaches its edge.
        stable.settings = dataclasses.replace(
            stable.settings, boundary_check=None, **settings
        )
        output = stable.run(progress="none")
        assert isinstance(output, MemoryOutput)
        assert all(np.all(np.isfinite(kept(plane.field))) for plane in output.planes())
