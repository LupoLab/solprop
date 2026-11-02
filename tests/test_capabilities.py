# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a results file can give, stated as its readers decide it (SEP 41, S16)."""

from dataclasses import replace

import pytest

from solprop import capabilities
from solprop.exceptions import OutputError, ProcessingError
from solprop.output import MemoryOutput
from solprop.processing import compression, modal, time
from solprop.processing._examples import example_general_output, example_output
from solprop.recipe import Recipe


def _spatial_output():
    """The example run relabelled as a radial one, as the compression tests do."""
    metadata = example_output().metadata
    assert metadata is not None
    spatial = MemoryOutput()
    spatial.open(
        replace(
            metadata,
            simulation_kind="radial",
            spatial={"geometry": "radial", "flux_model": "scalar-axial-admittance"},
        )
    )
    return spatial


# Each operation's own reader, raising the refusal the statement must repeat.
READERS = {
    "energy": time.energy,
    "peak_power": time.peak_power,
    "physical_energy": modal.physical_energy,
    "compression": compression.transform_limit,
}


@pytest.mark.parametrize(
    ("build", "basis"),
    [
        (example_output, "modal"),
        (example_general_output, "general_modal"),
        (_spatial_output, "spatial"),
    ],
)
def test_each_stated_refusal_is_the_reader_s_own(build, basis):
    statement = capabilities.output(build())
    assert statement.basis == basis
    for operation, reader in READERS.items():
        if operation in statement.refused:
            with pytest.raises((OutputError, ProcessingError)) as caught:
                reader(build())
            assert str(caught.value) == statement.refused[operation]


def test_an_orthogonal_modal_file_gives_physical_quantities_and_compression():
    statement = capabilities.output(example_output())
    assert statement.quantities["energy"].kind == "physical"
    assert statement.admits("compression") and statement.admits("energy")
    assert not statement.admits("physical_energy")
    assert statement.resumability is None


def test_a_general_modal_file_gives_coordinates_and_names_what_reconstruction_needs():
    statement = capabilities.output(example_general_output())
    assert statement.quantities["power"].kind == "coordinate"
    assert not statement.admits("energy") and not statement.admits("compression")
    assert statement.admits("physical_energy")
    assert "reference tables" in statement.needs["reconstructed_flux"]


def test_a_spatial_file_refuses_compression():
    statement = capabilities.output(_spatial_output())
    assert statement.quantities["field"].kind == "coordinate"
    assert not statement.admits("compression")


def test_a_file_says_its_planes_its_recipe_and_whether_it_continues(tmp_path):
    recipe = Recipe.example("soliton")
    document = recipe.to_dict()
    document["build"]["n_save"] = 5
    document["execution"] = {"fields": "planes", "field_planes": [0, -1]}
    path = tmp_path / "soliton.h5"
    Recipe.from_dict(document).run(path, progress="none")
    statement = capabilities.output(path)
    assert statement.recipe
    assert statement.field_planes == (0, 4)
    assert statement.statistics_planes == (1, 2, 3)
    assert statement.resumability is not None
    assert not statement.resumability.resumable  # complete


def test_a_handler_without_a_record_is_refused():
    with pytest.raises(OutputError, match="no run record"):
        capabilities.output(MemoryOutput())
