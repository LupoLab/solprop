# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output schema 7: what a general modal file records, and what it refuses to resume.

The file is the subject here: the labels and identities it carries, the round trip
through JSON and HDF5, the schema-5 and schema-6 fixtures that must keep reading, the
four doors that reach a writer, and the resume refusals -- a changed guide, a changed
table, a changed material, a reordered mode axis and an older file.

**Nothing here propagates far.** A resume is an identity statement, so the runs are a
tenth of a millimetre in two steps, and a replayed schedule at a fixed order, whose
resume continues the same march, is held to that march's rounding budget (SEP 25).
"""

import dataclasses
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from driver.test_resume import read_file, run_until
from march_bounds import FIXED_ORDER, assert_stored_matched, simulation_bound
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    general_grid,
    general_guide,
)
from solprop.driver import Simulation
from solprop.driver._scan_run import _Capture, _Options
from solprop.exceptions import OutputError, TranslationError
from solprop.modes.material import MaterialKerr, quantised_measurement
from solprop.modes.protocol import ChannelLabel
from solprop.output import HDF5Output
from solprop.output.hdf5 import check_cache, open_file, resume
from solprop.output.memory import MemoryOutput
from solprop.output.schema import (
    SCHEMA_VERSION,
    ModeRecord,
    RunMetadata,
    SavedPlane,
    _hashed_modes,
)
from solprop.processing import scans
from solprop.processing.modal import channel_labels, physical_energy
from solprop.stats import GENERAL_MODAL_ACCESSORS
from solprop.stepper import FixedSchedule
from solprop.translate.general import (
    general_modal_route,
    guide_identity,
    modal_identity,
)
from solprop.translate.modal_tables import DEFAULT_GRAM_ATOL

DATA = Path(__file__).resolve().parents[1] / "data"
SCHEMA_FIVE = DATA / "power-convention-schema5.h5"
SCHEMA_SIX = DATA / "general-modal-schema6.h5"
SCHEMA_SIX_DIGEST = "59fdd964fbab158c109abba1198908037c0142e744ece7b39d6066f11f740770"

MODAL_ROWS = {
    "basis",
    "channels",
    "channel count",
    "tables",
    "metric",
    "responses",
    "guide",
    "active band",
    "frame",
    "power weight",
    "noise basis",
    "accessors",
}
"""Every row `RunMetadata._describe` adds for a general modal identity."""

PULSE = {"lambda0": 835e-9, "fwhm": 20e-15, "energy": 1e-12, "pulseshape": "gauss"}
RUN = {
    # Two steps over a tenth of a millimetre. A resume is an identity statement
    # and needs no more; the route's arithmetic is qualified elsewhere.
    "save_planes": [GENERAL_Z_END / 20, GENERAL_Z_END / 10],
    "backend": "numpy_ref",
    "dz_max": GENERAL_Z_END / 40,
    "dz_init": GENERAL_Z_END / 40,
}


def _spooled_flux(output) -> np.ndarray:
    """Return the last plane's recorded peak flux, read back off the spool."""
    return np.asarray(output["stats/analytic_flux"][-1].sum())


def at(record, *keys) -> Any:
    """Return one nested entry of a serialised identity, loosely typed.

    The identity is JSON, so a static checker sees a union at every level; a test
    that reads one entry says which one it wants and gets it, exactly as
    ``tests/driver/test_resume.py``'s ``item`` does for an HDF5 file.
    """
    value = record
    for key in keys:
        value = value[key]
    return value


@pytest.fixture(scope="module")
def grid():
    """The field grid these runs are on."""
    return general_grid()


@pytest.fixture(scope="module")
def guide(grid):
    """One HE11 channel of a 1.5 um strand, sampled over the run's active band."""
    return general_guide(grid=grid)


def build(guide, grid, **overrides):
    """Return a general modal simulation on this guide and grid."""
    return Simulation.from_waveguide(
        (), GENERAL_Z_END, provider=guide, grid=grid, **(PULSE | RUN | overrides)
    )


@pytest.fixture(scope="module")
def written(guide, grid, tmp_path_factory):
    """One general modal run, written to an HDF5 file once."""
    path = tmp_path_factory.mktemp("modal") / "general.h5"
    build(guide, grid).run(path, progress="none")
    return path


# ------------------------------------------------------------- what is persisted


def test_the_record_carries_every_modal_identity(written, guide):
    """Labels, tables, metric, responses, guide, frame, band, noise and accessors."""
    stored = resume(written).metadata
    assert stored.schema_version == SCHEMA_VERSION == 25
    modal = stored.modal
    assert modal is not None
    assert set(modal) == {
        "basis",
        "n_components",
        "n_channels",
        "channels",
        "tables",
        "metric",
        "responses",
        "guide",
        "frame",
        "power_weight",
        "active_band",
        "noise_reference_basis",
        "accessors",
    }
    assert (at(modal, "basis"), at(modal, "n_components")) == ("general_modal", 3)
    assert at(modal, "metric", "kind") == "identity"
    assert at(modal, "metric", "defect") <= at(modal, "metric", "budget")
    assert at(modal, "responses", "model_id")
    assert at(modal, "guide", "provider", "model") == "bound-step-index"
    assert at(modal, "guide", "sampling", "n_knots") == len(guide.sampling.omega)
    assert [at(one, "name") for one in at(modal, "guide", "regions")] == [
        "core",
        "cladding",
    ]
    assert at(modal, "guide", "regions", 0, "kerr", "chi3") > 0.0
    assert at(modal, "guide", "regions", 1, "kerr") is None
    assert at(modal, "frame") == "shared co-moving"
    assert at(modal, "power_weight") == "constant"
    assert at(modal, "noise_reference_basis") is None
    assert "physical_energy" in at(modal, "accessors")


def test_the_measured_entries_of_the_key_are_quantised_and_the_declared_are_not(
    written, guide
):
    """Six significant digits on what was measured, and exactness on what was said.

    A Gram defect and a profile error come out of a dense solve and a reduction,
    so their last bits belong to the BLAS that did them; a re-translation of the
    same guide on another machine must not refuse a resume it should accept. A
    declared `chi3`, a radius and a knot enter exactly, because a caller who
    changes one means to.
    """
    stored = resume(written).metadata
    measured = {
        "modal.metric.defect": at(stored.modal, "metric", "defect"),
        "modal.guide.sampling.profile_error": at(
            stored.modal, "guide", "sampling", "profile_error"
        ),
        "modal.tables.profile_error_max": at(
            stored.modal, "tables", "profile_error_max"
        ),
    }
    for path, value in measured.items():
        assert value == quantised_measurement(value), path
        assert float(f"{value:e}") != 0.0, path
    sampling = guide.sampling
    assert sampling is not None
    assert measured["modal.guide.sampling.profile_error"] == quantised_measurement(
        sampling.profile_error
    )
    # Six digits is a real narrowing: the raw measurement has more of them.
    assert measured["modal.guide.sampling.profile_error"] != sampling.profile_error
    # And a perturbation below the last kept digit does not move the key at all.
    nudged = dataclasses.replace(
        sampling, profile_error=sampling.profile_error * (1.0 + 1e-9)
    )
    assert guide_identity(dataclasses.replace(guide, sampling=nudged))[
        "sampling"
    ] == at(stored.modal, "guide", "sampling")
    # The declared numbers are exact, to the bit: a `chi3` the caller wrote, and
    # the budget the defect was gated against.
    assert at(stored.modal, "guide", "regions", 0, "kerr", "chi3") == GENERAL_CHI3
    assert at(stored.modal, "metric", "budget") == DEFAULT_GRAM_ATOL


def test_the_labels_round_trip_through_the_file(written):
    """`ChannelLabel.as_mapping` out, `ChannelLabel.from_mapping` back."""
    stored = resume(written).metadata
    labels = [
        ChannelLabel.from_mapping(dict(one.label))
        for one in stored.modes
        if one.label is not None
    ]
    assert [one.display_label for one in labels] == ["HE1,1:even"]
    assert [dict(one.as_mapping()) for one in labels] == at(stored.modal, "channels")
    assert labels[0] == ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even")


def test_the_whole_record_survives_json_and_hashes_the_same(written):
    """A record read back from the file hashes to the digest the file was stamped."""
    stored = resume(written).metadata
    again = RunMetadata.from_json(stored.to_json())
    assert again.modal == stored.modal
    assert [one.label for one in again.modes] == [one.label for one in stored.modes]
    assert again.configuration_hash() == stored.configuration_hash()


def test_the_printed_block_prints_every_schema_seven_field_with_its_value(written):
    """The printing rule, checked by **value** and not by row heading.

    Every one of the thirteen entries of `RunMetadata.modal` reaches a row, and
    each row is asserted to carry a value read off the record rather than merely
    to exist: a heading with the wrong number under it would pass the weaker
    check. The `tables` row prints a 12-hex digest of `model_id`, which is a
    412-character JSON document on this one-channel fixture and filled the row
    with the opening of a serialised mapping before.
    """
    stored = resume(written).metadata
    block = str(stored)
    modal = stored.modal
    assert modal is not None
    digest = sha256(str(at(modal, "tables", "model_id")).encode()).hexdigest()[:12]
    for line, value in (
        ("basis:", "general_modal, 3 components"),
        ("channels:", "HE1,1:even"),
        ("channel count:", "1"),
        ("tables:", f"model {digest}"),
        ("metric:", "identity, defect"),
        ("responses:", "isotropic-kerr-real, dense-nodes, 1 region(s)"),
        ("guide:", "bound-step-index, 2 region(s), 9 knots on 288 nodes"),
        ("active band:", "Prad/s to"),
        ("frame:", "shared co-moving"),
        ("power weight:", "constant"),
        ("noise basis:", "none"),
        ("accessors:", f"{len(at(modal, 'accessors'))}: analytic_flux"),
    ):
        assert line in block, line
        assert value in block, (line, value)
    # The long `model_id` itself is on the record, and not in the printed row.
    assert str(at(modal, "tables", "model_id")) not in block
    assert "schema 25" in block
    # Twelve rows for thirteen fields: the basis row carries `n_components` too.
    printed = {key for key, _ in stored._describe()[1]}
    assert len(printed & MODAL_ROWS) == len(MODAL_ROWS)


@pytest.mark.parametrize(
    ("entry", "value", "expected"),
    [
        ("guide", None, "guide:         none"),
        (
            "guide",
            {"provider": {"model": "demo"}, "regions": [{}]},
            "demo, 1 region(s)",
        ),
        ("tables", None, "tables:        none"),
        ("tables", {"gauge_id": "g"}, "model unnamed, gauge g"),
        ("responses", None, "responses:     none"),
        (
            "responses",
            {"model_id": "m", "policy_id": "p", "n_responses": 0, "chi3": "later"},
            "m, p, 0 region(s), chi3 unrecorded",
        ),
    ],
)
def test_the_printed_block_survives_a_record_it_cannot_read(
    written, entry, value, expected
):
    """A record read from a file is data, and the block prints what it finds.

    A file written by another version, or edited by hand, may carry no guide
    entry, no tables, no responses, or ones with fields missing; the printed block
    says so rather than raising while a reader is trying to find out what the file
    is. Nothing here is a route solprop writes -- it is what a reader survives.
    """
    stored = resume(written).metadata
    modal = dict(cast("dict[str, object]", stored.modal))
    modal[entry] = value
    assert expected in str(dataclasses.replace(stored, modal=modal))


def test_a_record_carrying_a_modal_identity_below_schema_seven_is_refused(written):
    """The identity would sit outside the resume key, so two guides would share one."""
    stored = resume(written).metadata
    with pytest.raises(OutputError, match="requires output schema 7"):
        dataclasses.replace(stored, schema_version=6)


# --------------------------------------------- the accessor list is not in the key


def _written_with(path, metadata):
    """Write a one-plane file carrying exactly this record, and return the path."""
    handler = HDF5Output(path)
    handler.open(metadata)
    handler.save(
        SavedPlane(
            z=0.0,
            field=np.ones((len(metadata.modes), 1, metadata.grid.n_omega), complex),
        )
    )
    handler.close()
    return path


def _with_accessors(metadata, names):
    """Return this record with its published accessor list replaced."""
    assert metadata.modal is not None
    return dataclasses.replace(
        metadata, modal={**metadata.modal, "accessors": list(names)}
    )


def test_the_published_accessor_list_is_recorded_but_not_hashed(written):
    """What this build can compute is not what this run computed.

    `modal.accessors` is the one entry of the record that describes the
    **package version** rather than the run: every other entry answers "what was
    propagated". Leaving it in the resume key would mean that every accessor
    solprop ever gains invalidates every general modal file ever written -- for a
    capability those files gain, not one they lose. It is therefore kept in the
    record, where a reader wants it, and taken out of the key.
    """
    stored = resume(written).metadata
    assert stored.modal is not None
    published = at(stored.modal, "accessors")
    assert published == sorted(GENERAL_MODAL_ACCESSORS)
    assert len(published) == 14
    assert "spectral_flux" in published
    assert "plane_nodes" in published
    assert "reconstructed_flux" in published
    assert "accessors" not in at(stored.hashed(), "modal")
    assert set(at(stored.hashed(), "modal")) == set(stored.modal) - {"accessors"}
    # And a record carrying the eleven-name list of before `spectral_flux`,
    # `plane_nodes` and `reconstructed_flux` (the reconstructed trace's own name,
    # beside the `analytic_flux` statistic) existed hashes identically, so such a
    # file resumes into this build.
    before = _with_accessors(
        stored,
        [
            one
            for one in published
            if one not in {"spectral_flux", "plane_nodes", "reconstructed_flux"}
        ],
    )
    assert len(at(before.modal, "accessors")) == 11
    assert before.configuration_hash() == stored.configuration_hash()
    assert not before.differences(stored)


def test_an_older_accessor_list_resumes_and_a_lost_accessor_is_refused(
    written, tmp_path
):
    """The subset rule, both ways.

    A file whose list is **shorter** than this build's names only capabilities
    this build still has, so it resumes. A file naming one this build does
    **not** have is a genuine capability regression -- the run it describes could
    ask for a statistic nothing here can compute -- and is refused by the missing
    name, which the generic hash difference could never say, because the list is
    not in the key at all.
    """
    metadata = resume(written).metadata
    published = at(metadata.modal, "accessors")
    shorter = _written_with(
        tmp_path / "shorter.h5",
        _with_accessors(metadata, [one for one in published if one != "spectral_flux"]),
    )
    assert check_cache(shorter, metadata) is True
    longer = _written_with(
        tmp_path / "longer.h5",
        _with_accessors(metadata, [*published, "an_accessor_this_build_lost"]),
    )
    with pytest.raises(OutputError, match="an_accessor_this_build_lost"):
        check_cache(longer, metadata)
    # And a record with **no** accessor entry at all -- which a schema-7 file
    # written before the list existed would carry -- resumes: there is nothing to
    # compare, and the key never held it, so its digest is unchanged.
    assert metadata.modal is not None
    bare = dataclasses.replace(
        metadata,
        modal={
            key: value for key, value in metadata.modal.items() if key != "accessors"
        },
    )
    assert bare.configuration_hash() == metadata.configuration_hash()
    assert check_cache(_written_with(tmp_path / "none.h5", bare), metadata) is True


# ----------------------------------------------------- the older files still read


def test_the_schema_five_fixture_keeps_its_digest_and_its_labels():
    """A schema-5 fixture: an older file reads with its own convention labels."""
    stored = resume(SCHEMA_FIVE).metadata
    assert stored.schema_version == 5
    assert stored.power_convention == "vacuum"
    assert stored.modal is None
    assert "modal" not in stored.hashed()
    assert all(one.label is None for one in stored.modes)


def test_the_schema_six_fixture_reads_and_hashes_as_it_did():
    """The committed schema-6 file: two mode records, no labels, its own digest."""
    stored = resume(SCHEMA_SIX).metadata
    assert stored.schema_version == 6
    assert stored.power_convention == "poynting"
    assert stored.modal is None
    assert "modal" not in stored.hashed()
    assert [one.label for one in stored.modes] == [None, None]
    assert all("label" not in one for one in at(stored.hashed(), "modes"))
    assert stored.configuration_hash() == SCHEMA_SIX_DIGEST


@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        # A schema-5 file is refused one version earlier, by the schema-6 guard:
        # its amplitudes are vacuum-normalised and cannot be concatenated at all.
        (SCHEMA_FIVE, r"different run.*schema_version"),
        (SCHEMA_SIX, r"different run.*schema_version"),
    ],
)
def test_neither_older_fixture_resumes_a_general_run(
    fixture, reason, guide, grid, tmp_path
):
    """An older file carries no identity to check a guide against."""
    path = tmp_path / "older.h5"
    path.write_bytes(fixture.read_bytes())
    metadata = build(guide, grid).metadata()
    with pytest.raises(OutputError, match=reason):
        check_cache(path, metadata)


def test_a_schema_six_file_cannot_resume_any_schema_seven_run(tmp_path):
    """The 6 to 7 step is named, as the 5 to 6 one is.

    Not only a general modal run: the channel labels and the modal identity entered
    the resume key at schema 7, so an older file's digest and a new run's are not
    comparable whatever the basis, and the generic difference would say only that
    `schema_version` and `modal` moved. The file itself is unharmed, which the
    refusal says and this test asserts.
    """
    path = tmp_path / "older.h5"
    path.write_bytes(SCHEMA_SIX.read_bytes())
    ordinary = Simulation.from_gnlse(
        0.11,
        1e-4,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=1.0,
        pulseshape="sech",
        n_save=2,
        backend="numpy_ref",
    ).metadata()
    assert ordinary.modal is None
    assert ordinary.schema_version == SCHEMA_VERSION == 25
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, ordinary)
    assert resume(path).metadata.configuration_hash() == SCHEMA_SIX_DIGEST


def test_a_key_taken_below_schema_seven_drops_the_labels_a_mode_record_carries():
    """The guard that keeps every older file readable, exercised directly.

    A mode record built today carries a label, because every mode has one. Taken
    at schema 6 the key **drops** it, which is why the committed fixtures still
    hash to the digests they were stamped with; taken at 7 it is there.
    """
    from solprop.modes import MarcatiliMode  # noqa: PLC0415 - one construction

    record = ModeRecord.from_mode(MarcatiliMode(125e-6, "He", 1.0))
    assert record.label is not None
    assert record.label["family"] == "HE"
    assert "label" not in at(_hashed_modes([record], 6), 0)
    assert at(_hashed_modes([record], 7), 0)["label"] == record.label


# ----------------------------------------------------------------- resume refusals


def scheduled(guide, grid):
    """Return a general modal simulation that replays a fixed schedule.

    Ten steps over a tenth of a millimetre, at the Adams method's fixed top order. A
    replayed schedule takes the same steps whatever it is resumed from, and the kept
    history makes the resumed march the uninterrupted one's, so the two files agree
    within that march's rounding budget: the state's round trip through the file is a
    conjugation and a sign flip and the controller's own state does not enter a replay.
    """
    length = GENERAL_Z_END / 10
    planes = [length / 2, length]
    simulation = Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        **(
            PULSE
            | RUN
            | {
                "save_planes": planes,
                "schedule": FixedSchedule.uniform(
                    0.0, length, length / 10, save_planes=planes
                ),
                "integrator": FIXED_ORDER,
            }
        ),
    )
    # The collar of a run this short (2.5e5 1/m) is stepped in-stage by an explicit
    # Adams history only below 0.7 um; split off, it is integrated exactly.
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme="split_step"
    )
    return simulation


def test_the_same_schedule_resumes_onto_the_same_march(guide, grid, tmp_path):
    """A run stopped at its first plane and continued is the uninterrupted one.

    Exact: the positions and steps the schedule prescribes, and the step counts.
    Within the matched march's budget: the field and its statistics.
    """
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    scheduled(guide, grid).run(whole, progress="none")
    run_until(scheduled(guide, grid), part, after=1)
    assert read_file(part, "z").size == 2
    scheduled(guide, grid).run(part, resume=True, progress="none")
    keys = (
        "field",
        "z",
        "stats/physical_energy",
        "stats/analytic_flux",
        "stats/omega0",
        "stats/fwhm_t",
        "step_history/z",
        "step_history/h",
        "step_history/counts",
    )
    assert_stored_matched(
        {key: read_file(whole, key) for key in keys},
        {key: read_file(part, key) for key in keys},
        simulation_bound(scheduled(guide, grid)),
        exact=("z", "step_history/z", "step_history/h"),
    )
    # And the resumed file still says which guide produced it.
    assert resume(part).metadata.modal == resume(whole).metadata.modal


@pytest.mark.parametrize(
    ("what", "field"),
    [("radius", "guide"), ("chi3", "guide"), ("order", "channels")],
)
def test_a_changed_guide_table_or_material_refuses_by_the_field_that_moved(
    guide, grid, written, what, field
):
    """A resume names the mismatching field, not "the hash does not agree"."""
    stored = resume(written).metadata
    if what == "radius":
        other = general_guide(grid=grid, radius=1.6e-6)
    elif what == "chi3":
        other = general_guide(grid=grid, chi3=3.0e-22)
    else:
        other = general_guide(descriptors=(("HE", 1, 1, "odd"),), grid=grid)
    metadata = build(other, grid).metadata()
    changed = stored.differences(metadata)
    assert any(one.startswith(f"modal.{field}") for one in changed), changed
    with pytest.raises(OutputError, match="these fields changed"):
        check_cache(written, metadata)


def test_a_reordered_mode_axis_is_a_different_run(grid):
    """The reviewer's trap: the same two channels, declared the other way round."""
    one = general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        grid=grid,
    )
    other = general_guide(
        descriptors=(("HE", 1, 1, "odd"), ("HE", 1, 1, "even")),
        blocks=((0, 1),),
        grid=grid,
    )
    first = build(one, grid).metadata()
    second = build(other, grid).metadata()
    assert first.configuration_hash() != second.configuration_hash()
    changed = first.differences(second)
    assert any(one.startswith("modal.channels") for one in changed), changed
    assert any(one.startswith("modes") for one in changed), changed


def test_a_route_with_no_guide_identity_cannot_write_a_file(guide, grid):
    """Defence in depth: a hand-built route says nothing about its guide."""
    route = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )
    bare = dataclasses.replace(route, guide_identity=None)
    with pytest.raises(TranslationError, match="carries no guide identity"):
        modal_identity(bare, grid, channels=[], has_noise=False)


def test_a_label_count_that_disagrees_with_the_tables_is_refused(guide, grid):
    """One label per row of the mode axis, or the file would misname a channel."""
    route = general_modal_route(
        guide, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )
    with pytest.raises(TranslationError, match="one channel label per row"):
        modal_identity(route, grid, channels=[], has_noise=False)


def test_a_guide_without_sampling_has_no_identity_to_record(guide):
    """`from_step_index` declares no resolution, so nothing says what to rebuild."""
    bare = dataclasses.replace(guide, sampling=None)
    with pytest.raises(TranslationError, match="declares no sampling"):
        guide_identity(bare)


# ---------------------------------------------------------------- noise identity


def test_the_noise_realisation_is_recorded_and_reconstructed_on_resume(
    guide, grid, tmp_path
):
    """Seed, realisation, output and resume identity on the general route."""
    path = tmp_path / "noisy.h5"
    simulation = build(guide, grid, noise="input", seed=17)
    simulation.run(path, progress="none")
    stored = resume(path).metadata
    assert stored.noise is not None
    assert stored.noise.seed == 17
    assert at(stored.modal, "noise_reference_basis") == "orthonormal true power"
    assert simulation.noise is not None
    assert np.array_equal(stored.noise.spectrum, simulation.noise.spectrum)
    # The same seed rebuilds the same realisation, so the same file is the same run.
    again = build(guide, grid, noise="input", seed=17)
    assert check_cache(path, again.metadata()) is True
    # A different seed is a different run, and is refused by the noise field.
    other = build(guide, grid, noise="input", seed=18)
    changed = stored.differences(other.metadata())
    assert any(one.startswith("noise") for one in changed), changed
    with pytest.raises(OutputError, match="these fields changed"):
        check_cache(path, other.metadata())


# ------------------------------------------------- two of the four doors, in full


def test_the_memory_handler_writes_the_same_file_afterwards(guide, grid, tmp_path):
    """`MemoryOutput.to_file` knows nothing of the problem, and needs nothing."""
    output = cast("MemoryOutput", build(guide, grid).run(progress="none"))
    path = tmp_path / "from_memory.h5"
    output.to_file(path)
    stored = resume(path)
    assert stored.metadata.modal is not None
    last = output.planes()[-1].field
    assert stored.plane.field is not None and last is not None
    assert np.array_equal(stored.plane.field, last)


def test_a_scan_point_writes_its_spool_and_reads_the_identity_back(
    guide, grid, tmp_path
):
    """The scan's own door, through the file and not only through the constructor.

    A point with a **custom** reduction is the case that reaches HDF5: the capture
    opens an `HDF5Output` spool before the run loop is entered, writes every plane
    to it and reads it back to evaluate the reduction. At schema 6 that spool was
    refused where the point is built, so nothing downstream ever ran; here the run
    goes through it and the spooled file carries the schema-7 identity.
    """
    reduction = scans.custom("spooled_flux", _spooled_flux, (), "W")
    options = _Options(
        reductions=(reduction,), fields="all", field_planes=(), stats="all"
    )
    capture = _Capture(options, build(guide, grid), tmp_path)
    assert capture.field_shape == (1, 1, grid.n_omega)
    assert capture.count == len(RUN["save_planes"]) + 1
    build(guide, grid).run(capture, progress="none")
    results = capture.results()
    assert float(results["reduced/spooled_flux"]) > 0.0
    stored = resume(capture.custom_path).metadata
    assert stored.schema_version == SCHEMA_VERSION == 25
    assert stored.modal is not None
    assert at(stored.modal, "basis") == "general_modal"
    assert [one.label for one in stored.modes] != [None]


def test_a_material_region_that_declares_a_kerr_response_records_it(grid):
    """The region's declaration is in the identity, so a changed `n2` is a new run."""
    declared = general_guide(grid=grid, chi3=None)
    provider = declared.field_provider
    regions = list(provider.regions)
    assert regions[0].kerr is None
    written = guide_identity(declared)
    assert at(written, "regions", 0, "kerr") is None
    responding = general_guide(grid=grid, chi3=2.2e-22)
    assert at(guide_identity(responding), "regions", 0, "kerr") == {
        "chi3": 2.2e-22,
        "n2": None,
        "at": None,
    }
    assert MaterialKerr(chi3=2.2e-22).chi3 == 2.2e-22
    # The wavelength an n2 was measured at is in the key beside the coefficient it
    # converts: it multiplies the executed chi3 by an index ratio squared, and
    # nothing downstream of the leaf can see which frequency it came from.
    measured = general_guide(grid=grid, kerr=MaterialKerr(n2=2.6e-20, at=1550e-9))
    assert at(guide_identity(measured), "regions", 0, "kerr") == {
        "chi3": None,
        "n2": 2.6e-20,
        "at": 1550e-9,
    }


# -------------------------------------------- the declared change of basis in the key


PAIR = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))


@pytest.fixture(scope="module")
def basis_pair(grid):
    """The same parity pair in the provider's own basis and in the circular one."""
    return {
        "plain": general_guide(descriptors=PAIR, blocks=((0, 1),), grid=grid),
        "circular": general_guide(
            descriptors=PAIR, blocks=((0, 1),), circular=(0, 1), grid=grid
        ),
    }


def test_a_file_in_one_basis_refuses_to_resume_in_another(grid, basis_pair, tmp_path):
    """Four independent paths, each naming itself; the digest moves with them."""
    path = tmp_path / "circular.h5"
    build(basis_pair["circular"], grid).run(path, progress="none")
    stored = resume(path).metadata
    other = build(basis_pair["plain"], grid).metadata()
    changed = stored.differences(other)
    for field in (
        "modal.metric.kind",
        "modal.metric.gauge_id",
        "modal.tables.gauge_id",
        "modal.guide.sampling.basis",
        "modal.channels",
    ):
        assert any(one.startswith(field) for one in changed), changed
    assert stored.configuration_hash() != other.configuration_hash()
    with pytest.raises(OutputError, match="these fields changed"):
        check_cache(path, other)


def test_a_file_without_a_basis_refuses_to_resume_with_one(grid, basis_pair, tmp_path):
    """The other direction: the omitted key is not a hole.

    `guide.sampling.basis` is present only where a basis is declared, so a general modal
    file written before a basis could be declared keeps the key it was stamped with. The
    two gauges are always compared, and they are what refuses this direction.
    """
    path = tmp_path / "plain.h5"
    build(basis_pair["plain"], grid).run(path, progress="none")
    stored = resume(path).metadata
    other = build(basis_pair["circular"], grid).metadata()
    changed = stored.differences(other)
    assert any(one.startswith("modal.metric.gauge_id") for one in changed), changed
    assert any(one.startswith("modal.tables.gauge_id") for one in changed), changed
    with pytest.raises(OutputError, match="these fields changed"):
        check_cache(path, other)


def test_a_run_in_the_provider_s_own_basis_keeps_the_key_it_had(grid, basis_pair):
    """No `basis` entry appears where none is declared, so old files still resume."""
    plain = build(basis_pair["plain"], grid).metadata()
    sampling = at(plain.modal, "guide", "sampling")
    assert "basis" not in sampling
    declared = at(
        build(basis_pair["circular"], grid).metadata().modal, "guide", "sampling"
    )
    assert declared["basis"]["gauge_id"] == "circular(0,1)"
    assert declared["basis"]["blocks"] == [[0, 1]]


def test_a_file_in_a_declared_basis_refuses_the_metric_weighted_accessor(
    grid, basis_pair, tmp_path
):
    """Schema 7 stores the metric's identity, not its arrays, so the accessor stops.

    The run's own recorded `physical_energy` statistic carries the weighted value;
    the file-time accessor would return the unweighted one, which is the answer
    for a different basis.
    """
    path = tmp_path / "accessor.h5"
    build(basis_pair["circular"], grid).run(path, progress="none")
    with open_file(path) as handle:
        with pytest.raises(OutputError, match="basis metric is of kind 'dense'"):
            physical_energy(handle)
        # The label accessor is metric-free by construction and stays.
        assert channel_labels(handle)[0].handedness == "left"
