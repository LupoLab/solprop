# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The GNLSE route: its grid, its fibre and its dispersion table."""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float

from solprop.constants import C
from solprop.driver.routes.keywords import _GRID_KEYWORDS, _about, _unset
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid, TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.tabulated import IndexTable
from solprop.nonlinear.gnlse import (
    GnlseProblem,
    SpecifiedMode,
    attenuation_from_db,
    problem_from,
)
from solprop.physdata.selection import MaterialSelection

__all__ = []


def _gnlse_grid(
    grid: TimeGrid | None,
    length: float,
    wavelength: float,
    wavelength_limits: tuple[float, float] | None,
    trange: float | None,
) -> TimeGrid:
    """Return the envelope grid a GNLSE run propagates on.

    - ``grid``: a grid given by the caller, or ``None`` to build one.
    - ``length``: fibre length, m.
    - ``wavelength``: reference wavelength, m.
    - ``wavelength_limits``: window limits, m.
    - ``trange``: time window, s.
    """
    if grid is not None:
        if grid.kind != "envelope":
            msg = (
                f"prop_gnlse propagates an envelope grid and this one is a "
                f"{grid.kind} grid; the GNLSE mapping is the envelope Kerr response "
                "(docs/guide/waveguides/gnlse.md), and a field-resolved run is "
                "prop_capillary's "
                "path"
            )
            raise DriverError(msg, parameter="grid", rule="gnlse-envelope-grid")
        return grid
    if wavelength_limits is None or trange is None:
        msg = (
            "give wavelength_limits and trange so a grid can be built, or a grid of "
            "your own"
        )
        raise DriverError(
            msg,
            parameter=_unset(wavelength_limits=wavelength_limits, trange=trange),
            rule="grid-or-limits",
        )
    with _about(renamed=_GRID_KEYWORDS):
        return EnvelopeGrid.from_limits(length, wavelength, wavelength_limits, trange)


def _gnlse_fibre(
    gamma: float | None,
    betas: Sequence[float] | np.ndarray | None,
    beta: object,
    grid: TimeGrid,
    *,
    wavelength: float,
    loss: float | None,
    shock: bool,
    fibre: Mode | IndexTable | None = None,
    n2: float | str | None = None,
    raman: bool = False,
    raman_fraction: float | None = None,
    selection: MaterialSelection | None = None,
) -> GnlseProblem:
    """Map one dispersion source and one nonlinear coefficient onto the GNLSE.

    - ``gamma``: nonlinear coefficient in 1/(W m), absent beside `n2`.
    - ``betas``: Taylor coefficients about the reference frequency, s^n/m.
    - ``beta``: a pair of frequency and beta arrays, a callable, or None.
    - ``wavelength``: positive reference wavelength, m.
    - ``loss``: power loss in dB/m; None means omitted. Refused beside `fibre`.
    - ``fibre``: whole dispersion and loss, including a positive propagation area for
      `n2`.
    - ``n2``: nonlinear index in m^2/W, or ``"recommended"`` under a `selection`;
      requiring `fibre`.
    - ``raman_fraction``: share of the nonlinearity held back from the Kerr term,
      between zero and one; None resolves as in
      :func:`solprop.nonlinear.gnlse.problem_from`.
    """
    if sum(value is not None for value in (betas, beta, fibre)) != 1:
        raise DriverError(
            "give exactly one of betas, beta and fibre",
            parameter=("betas", "beta", "fibre"),
            rule="betas-beta-or-fibre",
        )
    if (gamma is None) == (n2 is None):
        raise DriverError(
            "give exactly one of gamma and n2",
            parameter=("gamma", "n2"),
            rule="gamma-or-n2",
        )
    if n2 is not None and fibre is None:
        raise DriverError(
            "n2 requires fibre with an effective area",
            parameter=("n2", "fibre"),
            rule="n2-needs-fibre-area",
        )
    if fibre is not None and loss is not None:
        raise DriverError(
            "the fibre carries its loss; scale it on the mode",
            parameter=("fibre", "loss_db_per_m"),
            rule="fibre-carries-loss",
        )
    omega_ref = 2.0 * np.pi * C / wavelength
    mode = (
        None
        if fibre is None
        else _gnlse_mode(fibre, wavelength, needs_area=n2 is not None)
    )
    area = 1.0 if mode is None else mode.aeff()
    # The GNLSE mapping of ADR-026 (docs/dev/decisions.md) is unchanged. An externally
    # specified fibre replaces the placeholder mode, so its dispersion never enters the
    # unit prefactor. Its own names are the keywords' but for the area, the fibre's.
    with _about(renamed={"aeff": "fibre"}):
        mapping = problem_from(
            np.asarray([0.0] if betas is None else betas, dtype=np.float64),
            omega_ref=omega_ref,
            gamma=gamma,
            n2=n2,
            aeff=area,
            loss_db_per_m=0.0 if loss is None else loss,
            shock=shock,
            raman=raman,
            raman_fraction=raman_fraction,
            selection=selection,
        )
    if mode is not None:
        return dataclasses.replace(mapping, mode=mode)
    if betas is not None:
        return mapping
    nodes, values = _dispersion_table(beta, grid)
    with _about("beta"):
        table = SpecifiedMode.from_table(
            nodes,
            values,
            loss=attenuation_from_db(0.0 if loss is None else loss),
            aeff=mapping.aeff,
        )
    return dataclasses.replace(mapping, mode=table)


def _gnlse_mode(
    fibre: Mode | IndexTable, wavelength: float, *, needs_area: bool
) -> Mode:
    """Resolve a supplied table or mode with its propagation area at the pump.

    - ``wavelength``: positive reference wavelength, m: where a table's area column and
      a provider channel's surrogate area are evaluated.
    """
    if isinstance(fibre, IndexTable):
        if needs_area and fibre.area is None:
            raise DriverError(
                "n2 requires a fibre table with an effective-area column",
                parameter=("n2", "fibre"),
                rule="n2-needs-fibre-area",
            )
        area = (
            1.0
            if fibre.area is None
            else float(
                fibre.effective_area(np.array([2.0 * np.pi * C / wavelength]))[0]
            )
        )
        with _about("fibre"):
            return SpecifiedMode.from_index_table(fibre, aeff=area)
    if not math.isfinite(fibre.aeff(0.0)) or fibre.aeff(0.0) <= 0:
        raise DriverError(
            "fibre effective area must be positive and finite", parameter="fibre"
        )
    # aeff=None hands over the fibre's own declared area path when it has one and
    # samples the source's own area along z when its whole geometry moves; the
    # frozen callables forward the position, so the dispersion and the area move
    # together and a fibre moving along z needs no refusal (ADR-026,
    # docs/dev/decisions.md). The wavelength is where a provider channel's area is
    # solved, as a table's is read at it above; every other mode's area is not a
    # function of frequency and ignores it.
    with _about("fibre"):
        return SpecifiedMode.from_mode(fibre, wavelength=wavelength)


def _dispersion_table(
    beta: object, grid: TimeGrid
) -> tuple[Float[np.ndarray, " n_nodes"], Float[np.ndarray, " n_beta_nodes"]]:
    """Return the ``(omega, beta)`` table a caller's dispersion describes.

    A callable is sampled on the grid's **active band**, sorted and made unique, because
    that is exactly where the linear operator evaluates a mode
    (:func:`solprop.linear.dispersion_operator`) and a spline refuses to extrapolate
    beyond its data.

    - Returns: the frequencies in rad/s and the propagation constant in 1/m.
    """
    if callable(beta):
        nodes = np.unique(grid.omega[grid.active_mask])
        return nodes, np.asarray(beta(nodes), dtype=np.float64)
    if isinstance(beta, tuple) and len(beta) == 2:
        nodes, values = beta
        return (
            np.asarray(nodes, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
    msg = (
        f"beta must be a (omega, beta) pair of arrays or a callable of omega, got "
        f"{type(beta).__name__}"
    )
    raise DriverError(msg, parameter="beta")
