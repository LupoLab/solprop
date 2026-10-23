# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop.exceptions import ModeError
from solprop.modes.protocol import ChannelLabel, Mode, ParameterisedMode
from solprop.profiles import Constant


class _Uniform:
    """A trivial mode: index 1, unit field on a disc of radius 1 m."""

    z_constant = True
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("uniform", None, None, None, None, "uniform")

    def principal_component(self, z=0.0):
        return "x"

    def neff(self, omega, z=0.0):
        return np.ones_like(omega, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def dneff_domega(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        return np.pi

    def dimlimits(self, z=0.0):
        return "polar", (0.0, 0.0), (1.0, 2 * np.pi)

    def field(self, points, z=0.0):
        return np.stack([np.ones(len(points)), np.zeros(len(points))], axis=1)


class _NotAMode:
    def neff(self, omega, z=0.0):
        return omega


def test_a_complete_implementation_satisfies_the_protocol():
    mode = _Uniform()
    assert isinstance(mode, Mode)
    omega = np.array([1.0, 2.0])
    assert mode.neff(omega).dtype == complex
    assert mode.dimlimits()[0] == "polar"
    assert mode.field(np.zeros((3, 2))).shape == (3, 2)


def test_a_partial_implementation_does_not():
    assert not isinstance(_NotAMode(), Mode)


def test_a_plain_mode_is_not_a_parameterised_one():
    """The expansion is opt-in: a mode without the two members takes the z table."""
    assert not isinstance(_Uniform(), ParameterisedMode)


def test_declaring_the_two_members_makes_a_mode_parameterised():
    class _Parameterised(_Uniform):
        @property
        def parameter_paths(self):
            return {"density": Constant(1.0)}

        def at_parameters(self, values):
            return self

    assert isinstance(_Parameterised(), ParameterisedMode)


def test_a_mode_without_a_label_is_not_a_mode():
    members = {
        name: value
        for name, value in vars(_Uniform).items()
        if not name.startswith("__") and name != "channel_label"
    }
    assert not isinstance(type("_Unlabelled", (), members)(), Mode)


def test_a_channel_label_carries_its_own_fields_and_serialises_them():
    label = ChannelLabel("HE", 1, 2, "odd", "left", "HE1,2:odd")
    assert dict(label.as_mapping()) == {
        "family": "HE",
        "azimuthal_order": 1,
        "radial_order": 2,
        "parity": "odd",
        "handedness": "left",
        "display_label": "HE1,2:odd",
    }
    assert label == ChannelLabel("HE", 1, 2, "odd", "left", "HE1,2:odd")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (("", 1, 1, None, None, "x"), "needs a family"),
        (("HE", 1, 1, None, None, ""), "needs a family"),
        (("HE", -1, 1, None, None, "x"), "azimuthal_order must be"),
        (("HE", 1, 0, None, None, "x"), "radial_order must be"),
        # bool is an int in Python, so True would otherwise pass as an order here
        # while from_mapping refuses it. One rule at both doors.
        (("HE", True, 1, None, None, "x"), "azimuthal_order must be"),
        (("HE", 1, True, None, None, "x"), "radial_order must be"),
    ],
)
def test_a_channel_label_refuses_what_it_cannot_name(fields, message):
    with pytest.raises(ModeError, match=message):
        ChannelLabel(*fields)


def test_a_channel_label_comes_back_from_the_mapping_it_went_out_as():
    # The round trip a provider identity and output schema 7 need: the mapping has
    # been through JSON, so every value arrives as an ordinary Python object.
    for label in (
        ChannelLabel("HE", 1, 2, "odd", "left", "HE1,2:odd"),
        ChannelLabel("specified", None, None, None, None, "specified"),
    ):
        assert ChannelLabel.from_mapping(dict(label.as_mapping())) == label


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"family": None}, "family must be str"),
        ({"display_label": 3}, "display_label must be str"),
        ({"azimuthal_order": "one"}, "azimuthal_order must be int"),
        ({"radial_order": True}, "radial_order must be int"),
        ({"parity": 1}, "parity must be str"),
        ({"handedness": 1.5}, "handedness must be str"),
    ],
)
def test_a_channel_label_refuses_a_mapping_it_cannot_read(change, message):
    mapping = dict(ChannelLabel("HE", 1, 1, None, None, "HE1,1").as_mapping())
    with pytest.raises(ModeError, match=message):
        ChannelLabel.from_mapping({**mapping, **change})


@pytest.mark.parametrize("keys", [("family",), ("family", "azimuthal_order", "extra")])
def test_a_channel_label_refuses_a_mapping_of_the_wrong_shape(keys):
    mapping = dict(ChannelLabel("HE", 1, 1, None, None, "HE1,1").as_mapping())
    with pytest.raises(ModeError, match="needs exactly"):
        ChannelLabel.from_mapping({key: mapping.get(key) for key in keys})


@pytest.mark.parametrize(
    ("name", "message"), [("parity", "parity must be"), ("handedness", "handedness")]
)
def test_a_channel_label_refuses_a_partner_name_it_does_not_know(name, message):
    # Set past the constructor, as tests/modes/test_stepindex.py does for its own
    # Literal fields: the runtime hook of the test session refuses the wrong string
    # before __post_init__ sees it, and this check has to stay reachable in
    # production, where no hook runs.
    label = ChannelLabel("HE", 1, 1, None, None, "HE1,1")
    object.__setattr__(label, name, "sideways")
    with pytest.raises(ModeError, match=message):
        label.__post_init__()
