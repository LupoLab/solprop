# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from solprop.output import MemoryOutput
from solprop.output.schema import (
    GridRecord,
    RunMetadata,
    SavedPlane,
    StepperRecord,
    StepStats,
)
from solprop.plotting.statistics import _should_log, _trace, plot_stats
from solprop.processing._examples import example_grid, example_output, gvd_phase
from solprop.pulses import GaussianPulse
from solprop.stats import energy, peak_power


def test_stats_draws_one_panel_for_each_recorded_statistic(single):
    figures = plot_stats(single)
    assert len(figures) == 1
    figure = figures[0]
    assert figure.get_suptitle() == "Pulse statistics"
    # The default spec of a mode-averaged run records five statistics.
    assert len(figure.axes) == 5
    assert figure.axes[0].get_xlabel() == "Distance (cm)"


def test_stats_returns_nothing_for_a_run_with_no_statistics():
    assert plot_stats(example_output(with_stats=False)) == []


def test_stats_draws_both_groups_when_both_are_recorded():
    figures = plot_stats(_run_with_waveguide_statistics())
    assert [figure.get_suptitle() for figure in figures] == [
        "Pulse statistics",
        "Waveguide statistics",
    ]


def test_stats_converts_the_centroid_to_a_wavelength(single):
    figure = plot_stats(single)[0]
    (panel,) = [ax for ax in figure.axes if "wavelength" in ax.get_ylabel().lower()]
    # The example pulse is centred at 800 nm, and the centroid of its density is close
    # to that; the panel must show nanometres, not radians per second.
    centroid = np.asarray(panel.lines[0].get_ydata(), dtype=float)
    assert 700.0 < float(np.max(centroid)) < 900.0


def test_stats_scales_a_peak_power(single):
    figure = plot_stats(single)[0]
    (panel,) = [ax for ax in figure.axes if ax.get_ylabel().startswith("Peak power")]
    assert panel.get_ylabel() == "Peak power (MW)"


def test_stats_names_the_modes_of_a_multimode_run(multimode):
    figure = plot_stats(multimode)[0]
    (panel,) = [ax for ax in figure.axes if ax.get_ylabel() == "Energy (uJ)"]
    assert [str(line.get_label()) for line in panel.lines] == [
        "mode 1",
        "mode 2",
        "mode 3",
    ]
    assert panel.get_legend() is not None


def test_stats_numbers_the_columns_of_a_windowed_statistic():
    figure = plot_stats(_run_with_windowed_energy())[0]
    (panel,) = [ax for ax in figure.axes if ax.get_ylabel().startswith("Windowed")]
    assert [str(line.get_label()) for line in panel.lines] == ["1", "2"]


def test_stats_uses_a_logarithmic_ordinate_over_a_decade():
    figure = plot_stats(_run_with_waveguide_statistics())[1]
    scales = {ax.get_ylabel(): ax.get_yscale() for ax in figure.axes}
    assert scales["Number density (cm^-3)"] == "log"
    assert scales["Pressure (bar)"] == "linear"


def test_stats_reads_a_file_handler(on_file):
    assert len(plot_stats(on_file)) == 1


def test_stats_passes_keywords_to_the_plot(single):
    figure = plot_stats(single, linestyle="--")[0]
    assert figure.axes[0].lines[0].get_linestyle() == "--"
    plt.close(figure)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([[1.0], [100.0]]), True),
        (np.array([[1.0], [2.0]]), False),
        (np.array([[0.0], [100.0]]), False),
        (np.array([[-1.0], [-100.0]]), False),
    ],
)
def test_a_panel_is_logarithmic_only_over_a_decade_of_positive_values(values, expected):
    assert _should_log(values) is expected


def _synthetic(stats_of):
    """Return a run whose planes carry whatever ``stats_of(index)`` returns."""
    grid = example_grid()
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(),
        simulation_kind="mode_averaged",
        configuration={},
        stepper=StepperRecord(tableau="DOPRI5", controller={}),
    )
    seed = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    output = MemoryOutput()
    output.open(metadata)
    for index, z in enumerate(np.linspace(0.0, 0.1, 4)):
        output.save(
            SavedPlane(
                z=float(z),
                field=seed * gvd_phase(grid, 1e-27, float(z)),
                stats=stats_of(index),
            )
        )
    return output


def _run_with_waveguide_statistics():
    return _synthetic(
        lambda index: {
            "energy": np.asarray([[1e-6]]),
            "density": np.asarray(1e25 * 10.0**index),
            "pressure": np.asarray(1.0 + 0.1 * index),
        }
    )


def _run_with_windowed_energy():
    return _synthetic(
        lambda index: {
            "energy_window": np.asarray([[[1e-6]], [[0.5e-6 + 1e-8 * index]]])
        }
    )


def test_stats_prefers_the_between_planes_channel_when_there_is_one():
    """The denser trace, drawn from the same numbers, so the curve is the curve.

    A run with a periodic subset records the statistic at every period-th accepted
    step as well as at each plane; the along-z plot then has as many points as the
    run took steps rather than as many as it saved planes.
    """
    planes = example_output(with_step_stats=False)
    channel = example_output(with_step_stats=True)
    n_planes = planes["z"].size
    n_rows = channel["step_stats/z"].size
    assert n_rows > n_planes

    (from_planes,) = plot_stats(planes)
    (from_channel,) = plot_stats(channel)
    assert len(from_planes.axes) == len(from_channel.axes)
    plane_line = np.asarray(from_planes.axes[0].lines[0].get_xydata())
    channel_line = np.asarray(from_channel.axes[0].lines[0].get_xydata())
    assert plane_line.shape[0] == n_planes
    assert channel_line.shape[0] == n_rows
    plt.close("all")


def partial_channel(rows_per_plane=2):
    """A run that records two statistics at planes and only one between them."""
    grid = example_grid("envelope")
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    seed = pulse.build(grid)
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(),
        simulation_kind="mode_averaged",
        configuration={},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8}),
    )
    output = MemoryOutput()
    output.open(metadata)
    for index, z in enumerate(np.linspace(0.0, 0.1, 4)):
        field = seed * gvd_phase(grid, 1e-27, float(z))
        stats = {
            "energy": np.asarray(energy(grid, field)),
            "peak_power": np.asarray(peak_power(grid, field)),
        }
        positions = (
            np.empty(0)
            if index == 0
            else np.linspace(z - 0.01, float(z), rows_per_plane)
        )
        shape = np.shape(stats["energy"])
        rows = StepStats(
            z=positions,
            step=np.arange(positions.size),
            values={
                "energy": np.broadcast_to(
                    stats["energy"], (positions.size, *shape)
                ).copy()
            },
        )
        output.save(SavedPlane(z=float(z), field=field, stats=stats, step_stats=rows))
    return output


def test_stats_falls_back_to_the_planes_for_a_statistic_the_channel_omits():
    """The choice is per statistic, so a partial subset draws from both sources.

    A run may record a *subset* between planes -- that is what the subset is for -- and
    the panels for the rest must still be drawn, from the planes.
    """
    output = partial_channel()
    assert output.step_stat_names == ("energy",)
    n_planes = output["z"].size
    n_rows = output["step_stats/z"].size
    assert n_rows != n_planes

    z_channel, values_channel = _trace(output, "energy")
    assert z_channel.size == values_channel.shape[0] == n_rows
    z_planes, values_planes = _trace(output, "peak_power")
    assert z_planes.size == values_planes.shape[0] == n_planes

    figures = plot_stats(output)
    assert figures
    plt.close("all")


def test_stats_falls_back_when_the_channel_is_declared_but_empty():
    """A run whose subset recorded no rows still draws from its planes.

    Reachable: a period longer than the run, or a resumed run whose remaining planes
    are shorter than it. The layout names the channel and there is nothing in it.
    """
    output = partial_channel(rows_per_plane=0)
    assert output.step_stat_names == ("energy",)
    assert output["step_stats/z"].size == 0
    z, values = _trace(output, "energy")
    assert z.size == values.shape[0] == output["z"].size
    assert plot_stats(output)
    plt.close("all")
