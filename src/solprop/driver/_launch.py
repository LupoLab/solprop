# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launch each input pulse into chosen modes of a modal run (SEP 41, S5).

The ``launch`` keyword of :meth:`~solprop.driver.simulation.Simulation.from_capillary`
and :meth:`~solprop.driver.simulation.Simulation.from_waveguide` maps mode labels to a
share and, optionally, a phase. This module resolves the labels, normalises the shares,
places each pulse in its modes and writes the record the run keeps as
``configuration["modal_launch"]``.

**Normalisation** (SEP 41 section 4). On a basis whose flux metric is the identity --
the Marcatili and supplied gas-filled sets, and a general guide in its provider's own
Poynting-normalised coordinates -- the energy of a launch is the sum of its modes'
energies, so the shares are **energy fractions**, normalised to one per pulse, and
each mode carries its share of the pulse's energy. On a general modal basis whose
declared metric ``W`` has cross terms, the energy ``a^H W a`` of the coefficients
depends on their relative phases: there the shares are relative amplitudes squared,
the coefficients are scaled so that their flux-metric energy is the pulse's, and the
split is a **coordinate** one, which the record says.

**The phase** is in the optics convention of ``docs/dev/conventions.md``: a mode's
launched spectrum is the pulse's times ``exp(i phase)`` in that convention, so a phase
adds to the pulse's carrier-envelope phase in that mode. The state is
internal-convention, so the factor crosses :func:`solprop.fourier.to_internal`.
"""

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex

from solprop import fourier
from solprop.exceptions import DriverError
from solprop.grid import TimeGrid
from solprop.modes.protocol import ChannelLabel
from solprop.pulses import Pulse, PulseSum
from solprop.stats import energy, physical_energy

__all__ = ["LaunchedMode", "launch_pulse", "refuse_beside"]

_ENTRY_KEYS = frozenset({"share", "phase"})
"""The keys a mode's entry may hold when it is a mapping rather than a share."""


@dataclass(frozen=True)
class LaunchedMode:
    """One mode a pulse is launched into.

    Attributes
    ----------
    index : int
        The mode's row in the state.
    label : str
        Its display label.
    share : float
        Its share, normalised so that the pulse's shares sum to one; dimensionless.
    phase : float
        Its phase, rad, in the optics convention.
    """

    index: int
    label: str
    share: float
    phase: float


def _number(value: object) -> float | None:
    """Return a real number as a float, or ``None`` for anything else.

    Parameters
    ----------
    value : object
        The value.

    Returns
    -------
    float or None
        The value, for an int or float that is not a bool; ``None`` otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def _entry(value: object, where: str) -> tuple[float, float]:
    """Return one mode's share and phase from its entry.

    Parameters
    ----------
    value : object
        A share, or a mapping with ``"share"`` and, optionally, ``"phase"``.
    where : str
        The entry's path, for a refusal.

    Returns
    -------
    tuple of float
        The share as given, and the phase, rad.

    Raises
    ------
    DriverError
        If the entry is neither, names another key, or holds a share that is
        negative or not finite, or a phase that is not finite.
    """
    if isinstance(value, Mapping):
        unknown = sorted(str(key) for key in value if key not in _ENTRY_KEYS)
        if unknown or "share" not in value:
            msg = (
                f"{where} is a mapping with keys {sorted(map(str, value))}: a mode's "
                "entry takes 'share' and, optionally, 'phase' in radians"
            )
            raise DriverError(msg, parameter=where)
        given, angle = value["share"], value.get("phase", 0.0)
    else:
        given, angle = value, 0.0
    share = _number(given)
    if share is None or not math.isfinite(share) or share < 0:
        msg = f"{where}: a share is a finite number of at least zero, got {given!r}"
        raise DriverError(msg, parameter=where)
    phase = _number(angle)
    if phase is None or not math.isfinite(phase):
        msg = f"{where}: a phase is a finite number of radians, got {angle!r}"
        raise DriverError(msg, parameter=where)
    return share, phase


def _resolve(key: object, labels: Sequence[str], where: str) -> int:
    """Return the row a launch key names, as ``channel=`` resolves it.

    A display label names its row; an index names that row. A label without its
    parity or handedness suffix (``"HE1,1"`` for ``"HE1,1:even"``) names the one row
    that completes it, and is refused where both partners are present.

    Parameters
    ----------
    key : object
        The key: a display label, a label without its suffix, or an index.
    labels : sequence of str
        The mode set's display labels, in row order.
    where : str
        The key's path, for a refusal.

    Returns
    -------
    int
        The row.

    Raises
    ------
    DriverError
        If the key is not a string or an integer, names no row, or leaves a parity
        open between two rows.
    """
    if isinstance(key, str):
        if key in labels:
            return labels.index(key)
        candidates = [label for label in labels if label.startswith(f"{key}:")]
        if len(candidates) == 1:
            return labels.index(candidates[0])
        if candidates:
            msg = (
                f"{where}: {key!r} leaves a parity open, and this mode set carries "
                f"both {candidates}; name the one to launch into"
            )
            raise DriverError(msg, parameter=where)
        msg = f"{where}: {key!r} is not a mode of this set, whose modes are {labels}"
        raise DriverError(msg, parameter=where)
    if isinstance(key, numbers.Integral) and not isinstance(key, bool):
        index = int(key)
        if 0 <= index < len(labels):
            return index
        msg = f"{where}: {index} is outside this set's {len(labels)} mode(s): {labels}"
        raise DriverError(msg, parameter=where)
    msg = f"{where}: a mode is named by its display label or its index, got {key!r}"
    raise DriverError(msg, parameter=where)


def _one_pulse(
    launch: object, labels: Sequence[str], where: str
) -> tuple[LaunchedMode, ...]:
    """Resolve and normalise one pulse's launch.

    Parameters
    ----------
    launch : object
        The mapping from mode to entry.
    labels : sequence of str
        The mode set's display labels, in row order.
    where : str
        The mapping's path, for a refusal.

    Returns
    -------
    tuple of LaunchedMode
        The modes in row order, their shares summing to one.

    Raises
    ------
    DriverError
        If the mapping is empty or not a mapping, names a mode twice, or has no
        positive share.
    """
    if not isinstance(launch, Mapping) or not launch:
        msg = (
            f"{where} maps mode labels to shares, as {{'HE1,1': 0.9, 'HE1,2': "
            f"{{'share': 0.1, 'phase': 1.57}}}}, got {launch!r}"
        )
        raise DriverError(msg, parameter=where)
    chosen: dict[int, tuple[float, float]] = {}
    for key, value in launch.items():
        at = f"{where}[{key!r}]".replace("'", '"')
        index = _resolve(key, labels, at)
        if index in chosen:
            msg = f"{where} names {labels[index]} twice"
            raise DriverError(msg, parameter=where)
        chosen[index] = _entry(value, at)
    total = math.fsum(share for share, _ in chosen.values())
    if not total > 0:
        msg = f"{where}: the shares sum to zero, so nothing would be launched"
        raise DriverError(msg, parameter=where)
    return tuple(
        LaunchedMode(index, labels[index], share / total, phase)
        for index, (share, phase) in sorted(chosen.items())
    )


def _resolve_launch(
    launch: object, labels: Sequence[str], n_pulses: int
) -> tuple[tuple[LaunchedMode, ...], ...]:
    """Return each pulse's launch: one mapping for every pulse, or one each.

    Parameters
    ----------
    launch : object
        The ``launch`` keyword.
    labels : sequence of str
        The mode set's display labels, in row order.
    n_pulses : int
        How many pulses the input holds.

    Returns
    -------
    tuple of tuple of LaunchedMode
        One launch per pulse.

    Raises
    ------
    DriverError
        If a list does not hold one mapping per pulse, or a mapping is refused.
    """
    if isinstance(launch, Mapping):
        one = _one_pulse(launch, labels, "launch")
        return (one,) * n_pulses
    if isinstance(launch, str | bytes) or not isinstance(launch, Sequence):
        msg = (
            "launch is a mapping from mode label to share, or a list of them, one "
            f"per pulse; got {launch!r}"
        )
        raise DriverError(msg, parameter="launch")
    if len(launch) != n_pulses:
        msg = (
            f"launch lists {len(launch)} mapping(s) and the input holds {n_pulses} "
            "pulse(s): give one mapping for every pulse, or one per pulse in pulses="
        )
        raise DriverError(msg, parameter=("launch", "pulses"), rule="launch-per-pulse")
    return tuple(
        _one_pulse(one, labels, f"launch[{index}]") for index, one in enumerate(launch)
    )


def refuse_beside(
    launch: object,
    *,
    channel: int | str | None = None,
    incident: object = None,
    coupling: object = None,
) -> None:
    """Refuse a launch beside another way of saying what is launched.

    Parameters
    ----------
    launch : object
        The ``launch`` keyword.
    channel : int or str or None, optional
        The named channel, which a launch generalises.
    incident, coupling : object, optional
        An incident beam and its coupling model, which compute a launch of their own.

    Raises
    ------
    DriverError
        If `launch` is given beside any of them.
    """
    if launch is None:
        return
    if channel is not None:
        msg = (
            "launch= generalises channel=, and this call gives both: put the pulse "
            "in one channel with launch={channel: 1.0}, or leave channel unset"
        )
        raise DriverError(
            msg, parameter=("launch", "channel"), rule="launch-or-channel"
        )
    for name, value in (("incident", incident), ("coupling", coupling)):
        if value is not None:
            msg = (
                f"launch= and {name}= are two ways to say what is launched: the "
                "shares name the modes a pulse enters, and an incident beam coupled "
                "under a model enters what the channels capture of it; give one"
            )
            raise DriverError(
                msg, parameter=("launch", name), rule="launch-or-incident"
            )


def _placed(
    built: Complex[np.ndarray, "n_modes_in n_pol n_omega"],
    modes: tuple[LaunchedMode, ...],
    n_modes: int,
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Place one pulse in its modes, each at its share and phase.

    Parameters
    ----------
    built : numpy.ndarray
        The pulse, shape ``(1, n_pol, n_omega)``, internal convention, in the square
        root of watts.
    modes : tuple of LaunchedMode
        Its modes.
    n_modes : int
        How many modes the set carries.

    Returns
    -------
    numpy.ndarray
        The pulse's part of the state, shape ``(n_modes, n_pol, n_omega)``.
    """
    part = np.zeros((n_modes, *built.shape[1:]), dtype=np.complex128)
    for mode in modes:
        # An optics-convention phase factor, carried into the internal state.
        rotation = complex(fourier.to_internal(np.array([np.exp(1j * mode.phase)]))[0])
        part[mode.index] = math.sqrt(mode.share) * rotation * built[0]
    return part


def launch_pulse(
    pulse: Pulse,
    launch: object,
    grid: TimeGrid,
    labels: Sequence[ChannelLabel],
    *,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], dict[str, object]]:
    """Launch an input's pulses into chosen modes, and return the state and its record.

    Parameters
    ----------
    pulse : solprop.pulses.Pulse
        The input: a pulse, or a
        :class:`~solprop.pulses.PulseSum` whose members are launched one by one.
    launch : object
        The ``launch`` keyword: a mapping from mode label or index to a share, or to
        ``{"share": ..., "phase": ...}`` with the phase in radians; or a list of such
        mappings, one per pulse.
    grid : solprop.grid.TimeGrid
        The run's grid.
    labels : sequence of solprop.modes.protocol.ChannelLabel
        The mode set's labels, in row order.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)`` at the launch plane, shape
        ``(n_modes, n_modes, n_omega)``; ``None``, the default, is the identity, on
        which the shares are energy fractions.

    Returns
    -------
    numpy.ndarray
        The state, shape ``(n_modes, n_pol, n_omega)``, internal convention, in the
        square root of watts.
    dict
        The record, ``configuration["modal_launch"]``: the ``split`` (``"energy"`` or
        ``"coordinate"``) and, per pulse, its modes' labels, normalised shares and
        phases in rad.

    Raises
    ------
    DriverError
        If a label, share or phase is refused, a list does not hold one mapping per
        pulse, or a pulse builds into more than one mode.
    """
    members: tuple[Pulse, ...] = (
        pulse.pulses if isinstance(pulse, PulseSum) else (pulse,)
    )
    names = [label.display_label for label in labels]
    launches = _resolve_launch(launch, names, len(members))
    state: Complex[np.ndarray, "n_modes n_pol n_omega"] | None = None
    for index, (member, modes) in enumerate(zip(members, launches, strict=True)):
        built = np.asarray(member.build(grid), dtype=np.complex128)
        if built.shape[0] != 1:
            msg = (
                f"pulse {index} builds into {built.shape[0]} modes, and a launch "
                "places a one-mode pulse in its modes; build the state yourself and "
                "pass it to Simulation.with_input"
            )
            raise DriverError(msg, parameter="launch")
        part = _placed(built, modes, len(names))
        if metric is not None:
            # A coordinate split: the shares fix the direction of the coefficients,
            # and the flux metric their length, so that the launched energy is the
            # pulse's whatever the cross terms make of the relative phases.
            target = float(np.sum(energy(grid, built)))
            launched = float(physical_energy(grid, part, metric=metric))
            part *= math.sqrt(target / launched)
        state = part if state is None else state + part
    assert state is not None
    record: dict[str, object] = {
        "split": "energy" if metric is None else "coordinate",
        "pulses": [
            {
                "labels": [mode.label for mode in modes],
                "shares": [mode.share for mode in modes],
                "phases": [mode.phase for mode in modes],
            }
            for modes in launches
        ],
    }
    return state, record
