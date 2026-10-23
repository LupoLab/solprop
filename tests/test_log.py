# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The one progress protocol, the period, the ETA and the duration helper.

``LogReporter`` is the only class in the package that takes a clock as an argument, and
this is why: the period and both ETA estimators are functions of wall time, and a test
that waited for them would be slow and flaky. :class:`Clock` below is moved by hand, so
every line in these tests is emitted at an instant the test chose.
"""

import builtins
import inspect
import io
import logging
import math
import sys
import types
import warnings
from inspect import unwrap

import pytest
import tqdm.std

import solprop.log
from solprop.backends.protocol import StepSummary
from solprop.log import (
    BarReporter,
    LogReporter,
    NullReporter,
    PlaneSummary,
    ProgressReporter,
    configure_logging,
    default_progress,
    default_reporter,
    format_duration,
    is_interactive,
    resolve_reporter,
    set_default_progress,
)


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now


class Lines(logging.Handler):
    """Collect formatted records instead of writing them anywhere."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def lines():
    """Return ``(**kwargs) -> (LogReporter, list[str], Clock)`` on a private logger."""
    made = []

    def build(**keywords):
        logger = logging.getLogger(f"solprop.test_reporter_{len(made)}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = Lines()
        logger.addHandler(handler)
        clock = Clock()
        reporter = LogReporter(logger=logger, clock=clock, **keywords)
        made.append(reporter)
        return reporter, handler.messages, clock

    return build


def periodic(messages):
    """Return only the periodic lines among ``messages``."""
    return [line for line in messages if line.startswith("progress ")]


def march(reporter, clock, *, n, seconds, z_end=1.0, z_of=None):
    """Take ``n`` accepted steps of equal length, spending ``seconds`` in total."""
    reporter.started(0.0, z_end, 2)
    for index in range(n):
        clock.now = (
            seconds[index]
            if hasattr(seconds, "__len__")
            else (seconds * (index + 1) / n)
        )
        position = z_of(index) if z_of is not None else z_end * (index + 1) / n
        reporter.step(position, z_end / n, accepted=True)
    return reporter


# ----------------------------------------------------------------- configure_logging


def test_configure_logging_installs_one_handler(tmp_path):
    logger = configure_logging("DEBUG")
    n_first = len(logger.handlers)
    logger = configure_logging("INFO", file=tmp_path / "run.log")
    # A second call replaces the handler it installed; the NullHandler stays.
    assert len(logger.handlers) == n_first
    assert logger.level == logging.INFO
    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert "hello" in (tmp_path / "run.log").read_text()
    configure_logging("WARNING")


def test_configure_logging_sets_the_progress_default(tmp_path):
    previous = default_progress()
    try:
        configure_logging("WARNING", file=tmp_path / "run.log", progress="none")
        assert isinstance(resolve_reporter(None), NullReporter)
    finally:
        set_default_progress(previous)
        configure_logging("WARNING")


@pytest.mark.parametrize("function", [set_default_progress, default_reporter])
def test_invalid_progress_has_a_runtime_diagnostic_without_the_test_type_hook(function):
    # Production has no beartype import hook; exercise its own input validation too.
    with pytest.raises(ValueError, match="loud"):
        if function is set_default_progress:
            inspect.unwrap(function)("loud")
        else:
            inspect.unwrap(function)(choice="loud")


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_detached_interactive_stream_falls_back_to_logging(error):
    class Detached(io.StringIO):
        def isatty(self):
            raise error("detached")

    assert not is_interactive(Detached())


def test_stream_without_terminal_query_is_not_interactive(monkeypatch):
    monkeypatch.setattr(sys, "stderr", object())
    assert not is_interactive()


# ------------------------------------------------------------------- format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0:00:00"),
        (0.4, "0:00:00"),
        (1.6, "0:00:02"),
        (751.4, "0:12:31"),
        (3600.0, "1:00:00"),
        (86400.0, "1 day, 0:00:00"),
        (90061.0, "1 day, 1:01:01"),
        (400000.0, "4 days, 15:06:40"),
        (-1.0, "unknown"),
        (math.inf, "unknown"),
        (math.nan, "unknown"),
    ],
)
def test_format_duration_reads_as_a_clock(seconds, expected):
    assert format_duration(seconds) == expected


# -------------------------------------------------------------------- the protocols


@pytest.mark.parametrize("reporter", [NullReporter(), LogReporter()])
def test_reporters_satisfy_the_protocol(reporter):
    assert isinstance(reporter, ProgressReporter)
    reporter.started(0.0, 1.0, 2)
    reporter.step(0.5, 0.5, accepted=True)
    reporter.plane(1.0, StepSummary(1.0, 1, 0, 0.5, 0.5, 0.01, True))
    reporter.finished(1.0)


def test_a_step_summary_is_a_plane_summary():
    """The structural type in `log` is what keeps that module clear of the seam."""
    assert isinstance(StepSummary(1.0, 8, 1, 0.4, 0.3, 0.02, True), PlaneSummary)


def test_a_reporter_may_override_one_call_only():
    seen = []

    class Counter(NullReporter):
        def plane(self, z, summary):
            seen.append(z)

    reporter = Counter()
    assert isinstance(reporter, ProgressReporter)
    reporter.started(0.0, 1.0, 1)
    reporter.step(1.0, 1.0, accepted=False)
    reporter.plane(1.0, StepSummary(1.0, 1, 1, 0.5, 0.5, 0.01, True))
    reporter.finished(1.0)
    assert seen == [1.0]


# ------------------------------------------------------------------------ the period


@pytest.mark.parametrize(("period", "expected"), [(10.0, 0), (0.1, 5)])
def test_the_period_holds_however_fast_the_steps_come(lines, period, expected):
    """Ten thousand steps in half a second: the count is the period's, not theirs."""
    reporter, messages, clock = lines(period=period)
    march(reporter, clock, n=10_000, seconds=0.5)
    assert len(periodic(messages)) == expected


def test_a_period_of_zero_leaves_the_plane_and_closing_lines(lines):
    reporter, messages, clock = lines(period=0.0)
    march(reporter, clock, n=1000, seconds=100.0)
    reporter.plane(1.0, StepSummary(1.0, 1000, 0, 0.5, 0.4, 100.0, True))
    reporter.finished(100.0)
    assert periodic(messages) == []
    assert len(messages) == 2
    assert messages[0].startswith("z = ")
    assert messages[1].startswith("finished at z = ")


def test_a_line_carries_the_position_the_counts_and_the_step(lines):
    reporter, messages, clock = lines(period=1.0)
    march(reporter, clock, n=100, seconds=10.0, z_end=0.2)
    line = periodic(messages)[0]
    assert line.startswith("progress 10.0 %: z = 0.0200 of 0.2000 m, ")
    assert "10 steps (0 rejected), dz 2.0e-03 m" in line
    assert "0:00:01 elapsed" in line


def test_a_rejected_step_is_counted_but_does_not_move_the_position(lines):
    reporter, messages, clock = lines(period=1.0)
    reporter.started(0.0, 1.0, 2)
    for index in range(100):
        clock.now = 0.1 * (index + 1)
        # Every tenth attempt is rejected, and `_march_to` reports the position it would
        # have reached, which is ahead of where the run actually is.
        accepted = index % 10 != 0
        reporter.step(0.01 * (index + 1), 0.01, accepted=accepted)
    line = periodic(messages)[0]
    assert "10 steps (1 rejected)" in line
    assert "z = 0.1000 of 1.0000 m" in line
    assert reporter.n_steps == 100
    assert reporter.n_rejected == 10


# --------------------------------------------------------------------------- the ETA


def test_the_eta_is_unknown_until_the_run_is_past_both_thresholds(lines):
    reporter, messages, clock = lines(period=1.0)
    march(reporter, clock, n=1000, seconds=10.0)
    published = periodic(messages)
    # Two periods is the later threshold here: by the first line the run is 10 % in.
    assert "ETA unknown" in published[0]
    assert all("ETA unknown" not in line for line in published[1:])


def test_a_run_that_has_covered_almost_nothing_says_nothing_about_its_eta(lines):
    reporter, messages, clock = lines(period=1.0)
    # Four periods in and one part in a thousand along: past two periods, short of 2 %.
    march(reporter, clock, n=1000, seconds=4.0, z_of=lambda i: 1e-6 * (i + 1))
    assert all("ETA unknown" in line for line in periodic(messages))


def test_the_eta_is_exact_on_a_constant_rate(lines):
    reporter, _, clock = lines(period=1.0)
    march(reporter, clock, n=1000, seconds=10.0)
    clock.now = 10.0
    reporter.finished(10.0)
    # A constant rate is the one case both estimators get right, so 1 % is a bound on
    # the arithmetic and the whole-second rounding, not on the estimator.
    assert reporter.eta_error < 0.01
    assert reporter.eta_error_whole_run < 0.01


def test_the_windowed_eta_recovers_from_a_change_of_rate_and_the_whole_run_one_does_not(
    lines,
):
    """The steps halve in speed halfway, which is the shape most real runs have."""
    reporter, _, clock = lines(period=1.0)
    reporter.started(0.0, 1.0, 2)
    elapsed = 0.0
    for index in range(1000):
        elapsed += 0.01 if index < 500 else 0.02
        clock.now = elapsed
        reporter.step((index + 1) / 1000, 1e-3, accepted=True)
    clock.now = 15.0
    reporter.finished(15.0)

    # The change is at t = 5 s and the window is five periods, so the first line whose
    # window lies wholly after the change is the one at t = 10 s.
    after = [
        (abs(windowed - true) / true, abs(whole - true) / true)
        for at, windowed, whole in reporter._predictions
        if (true := reporter._elapsed_final - at) > 0.0 and at >= 10.0
    ]
    assert after
    windowed_error, whole_error = after[0]
    assert windowed_error < 0.10
    assert whole_error > 0.30
    # And the windowed one stays right for the rest of the run, while the whole-run one
    # only creeps back as the remaining time it is wrong about runs out.
    assert all(error < 0.10 for error, _ in after)
    # The medians over the whole run are closer together than the lines after the
    # change are, because every line before it is common to both estimators.
    assert reporter.eta_error < reporter.eta_error_whole_run


def test_a_replayed_schedule_counts_steps_rather_than_metres(lines):
    """A fixed schedule's steps are not uniform in z, so its ETA is not either."""
    reporter, messages, clock = lines(period=1.0, n_steps=1000)
    # A quadratic z(i) at a constant cost per step: the rate of z says fifteen seconds
    # remain at the halfway point, the remaining steps say five, and five is right.
    march(reporter, clock, n=500, seconds=5.0, z_of=lambda i: ((i + 1) / 1000) ** 2)
    line = periodic(messages)[-1]
    assert "ETA 0:00:05 " in line


def test_finished_says_how_far_off_the_etas_were(lines):
    reporter, messages, clock = lines(period=1.0)
    march(reporter, clock, n=1000, seconds=10.0)
    clock.now = 10.0
    reporter.finished(10.0)
    closing = messages[-1]
    assert closing.startswith("finished at z = 1.000000e+00 m: 1000 steps (0 rejected)")
    assert "ETA error 0." in closing
    # Every periodic line but the first published an ETA, and every one is accounted
    # for: the closing line is a measurement, not a decoration.
    assert f"over {len(periodic(messages)) - 1} lines" in closing


def test_the_eta_error_is_nan_until_the_run_finishes(lines):
    reporter, _, clock = lines(period=1.0)
    march(reporter, clock, n=1000, seconds=10.0)
    assert math.isnan(reporter.eta_error)
    assert math.isnan(reporter.eta_error_whole_run)


def test_a_run_with_no_periodic_line_closes_without_an_eta_clause(lines):
    reporter, messages, clock = lines(period=10.0)
    march(reporter, clock, n=10, seconds=0.5)
    reporter.finished(0.5)
    assert "ETA error" not in messages[-1]
    assert math.isnan(reporter.eta_error)


# ------------------------------------------------------------- the plane and the end


def test_the_plane_line_and_the_closing_ratio(lines):
    reporter, messages, _ = lines(period=0.0)
    reporter.started(0.0, 1.0, 2)
    reporter.plane(0.5, StepSummary(0.5, 40, 1, 0.7, 0.4, 0.03, True))
    reporter.plane(1.0, StepSummary(1.0, 20, 0, 0.9, 0.5, 0.02, True))
    reporter.finished(0.06)
    assert messages[0] == (
        "z = 5.000000e-01 m: 40 steps (1 rejected), mean step 1.250e-02 m, 0.03 s"
    )
    assert messages[1] == (
        "z = 1.000000e+00 m: 20 steps (0 rejected), mean step 2.500e-02 m, 0.02 s"
    )
    assert reporter.n_steps == 60
    assert reporter.n_rejected == 1
    assert reporter.tolerance_ratio == 0.9
    assert "achieved/requested tolerance ratio 9.000e-01" in messages[2]


def test_a_plane_with_no_steps_does_not_divide_by_zero(lines):
    reporter, messages, _ = lines(period=0.0)
    reporter.started(0.0, 1.0, 1)
    reporter.plane(0.0, StepSummary(0.0, 0, 0, 0.0, 0.0, 0.0, True))
    assert "mean step 0.000e+00 m" in messages[0]


def test_nothing_is_logged_below_the_level(lines):
    reporter, messages, clock = lines(period=1.0, level=logging.DEBUG)
    march(reporter, clock, n=100, seconds=10.0)
    reporter.finished(10.0)
    assert messages == []
    # The estimates are still formed, so the run is still measured even when silent.
    assert not math.isnan(reporter.eta_error)


# ------------------------------------------------------------------------ the bar


def bar(*, level=logging.INFO, **keywords):
    """Return ``(BarReporter, stream, messages, Clock)`` drawing on a string."""
    stream = io.StringIO()
    logger = logging.getLogger(f"solprop.test_bar_{id(stream)}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    handler = Lines()
    logger.addHandler(handler)
    clock = Clock()
    reporter = BarReporter(
        file=stream, ncols=120, ascii=True, logger=logger, clock=clock, **keywords
    )
    return reporter, stream, handler.messages, clock


def drive(reporter, clock, *, steps=40, dt=0.25, dz=0.02, start=0):
    """Accept ``steps`` steps of ``dz`` at one every ``dt`` seconds."""
    for i in range(start + 1, start + steps + 1):
        clock.now = i * dt
        reporter.step(i * dz, dz, True)


def last_frame(stream):
    """Return what the terminal shows: the text after the last carriage return."""
    return stream.getvalue().split("\r")[-1]


def test_the_bar_shows_the_fraction_the_counts_and_this_package_s_eta():
    reporter, stream, messages, clock = bar(period=0.5)
    reporter.started(0.0, 1.0, 2)
    drive(reporter, clock, steps=40)
    frame = last_frame(stream)
    assert " 80.0%|" in frame
    assert "0.8/1 m" in frame
    assert "40 steps (0 rejected), dz 2.0e-02 m, ETA " in frame
    assert "(from the last " in frame
    # No periodic *line*: the bar is the periodic event's output.
    assert not any(message.startswith("progress ") for message in messages)
    reporter.finished(10.0)


def test_the_bar_says_eta_unknown_before_the_thresholds():
    reporter, stream, _, clock = bar(period=0.5)
    reporter.started(0.0, 1.0, 1)
    clock.now = 0.5
    reporter.step(0.005, 0.005, True)
    assert "ETA unknown" in last_frame(stream)
    reporter.finished(0.5)


def test_a_replayed_schedule_bar_counts_steps():
    reporter, stream, _, clock = bar(period=0.5, n_steps=100)
    reporter.started(0.0, 1.0, 1)
    # A redraw happens on each whole period: at one step per quarter second, on the
    # even steps, so the frame after 26 steps is the 26th.
    drive(reporter, clock, steps=26)
    assert "26/100 steps" in last_frame(stream)
    reporter.finished(6.5)


def test_planes_are_named_on_the_bar_and_logged_at_debug_only():
    loud, stream, messages, clock = bar(period=0.5, level=logging.DEBUG)
    loud.started(0.0, 1.0, 3)
    drive(loud, clock, steps=4)
    loud.plane(0.08, StepSummary(0.08, 4, 0, 0.5, 0.5, 0.01, True))
    drive(loud, clock, steps=2, start=4)
    assert "plane 1/3" in last_frame(stream)
    assert any(message.startswith("z = ") for message in messages)
    loud.finished(1.5)
    quiet, _, quiet_messages, clock = bar(period=0.5, level=logging.INFO)
    quiet.started(0.0, 1.0, 3)
    quiet.plane(0.08, StepSummary(0.08, 4, 0, 0.5, 0.5, 0.01, True))
    assert not any(message.startswith("z = ") for message in quiet_messages)
    quiet.finished(0.0)


def test_finished_fills_the_bar_closes_it_and_logs_the_closing_line_after():
    reporter, stream, messages, clock = bar(period=0.5)
    reporter.started(0.0, 1.0, 1)
    drive(reporter, clock, steps=40)
    clock.now = 11.0
    reporter.finished(11.0)
    assert reporter.bar is None
    assert "100.0%|" in last_frame(stream)
    assert "40 steps (0 rejected), done" in last_frame(stream)
    assert messages[-1].startswith("finished at z = ")


def test_interrupted_leaves_the_bar_restores_the_handlers_and_warns():
    stream = io.StringIO()
    package = logging.getLogger("solprop")
    console = logging.StreamHandler(stream)
    package.addHandler(console)
    try:
        before = list(package.handlers)
        reporter = BarReporter(file=stream, ncols=80, ascii=True, clock=Clock())
        reporter.started(0.0, 1.0, 1)
        assert console not in package.handlers
        reporter.interrupted(0.3, KeyboardInterrupt())
        assert reporter.bar is None
        assert package.handlers == before
        assert "stopped at z = 3.000000e-01 m by KeyboardInterrupt" in stream.getvalue()
    finally:
        package.removeHandler(console)


def test_a_record_logged_while_the_bar_is_up_goes_above_it():
    stream = io.StringIO()
    package = logging.getLogger("solprop")
    console = logging.StreamHandler(stream)
    package.addHandler(console)
    clock = Clock()
    try:
        reporter = BarReporter(
            file=stream, ncols=80, ascii=True, clock=clock, period=0.5
        )
        reporter.started(0.0, 1.0, 1)
        drive(reporter, clock, steps=4)
        package.warning("a warning mid-run")
        text = stream.getvalue()
        assert "a warning mid-run\n" in text
        # The bar is cleared before the line and redrawn after it.
        assert "%|" in text.split("a warning mid-run\n")[-1]
        reporter.finished(1.0)
        assert console in package.handlers
    finally:
        package.removeHandler(console)


def test_the_bar_and_the_line_share_one_estimator(lines):
    line, _, clock_line = lines(period=0.5)
    reporter, _, _, clock_bar = bar(period=0.5)
    for one, clock in ((line, clock_line), (reporter, clock_bar)):
        one.started(0.0, 1.0, 1)
        drive(one, clock, steps=20, dt=0.25)
        drive(one, clock, steps=20, dt=0.5, start=20)
    assert reporter._predictions == line._predictions
    clock_line.now = clock_bar.now = 40.0
    line.finished(40.0)
    reporter.finished(40.0)
    assert reporter.eta_error == line.eta_error


def test_bar_logging_preserves_filters_and_reports_write_errors(monkeypatch):
    stream = io.StringIO()
    original = logging.StreamHandler(stream)
    record_filter = logging.Filter("solprop")
    original.addFilter(record_filter)

    class BrokenWriter:
        @staticmethod
        def write(text, *, file):
            raise OSError("closed terminal")

    handler = solprop.log._BarHandler(BrokenWriter, original)
    errors = []
    monkeypatch.setattr(handler, "handleError", errors.append)
    record = logging.makeLogRecord({"name": "solprop", "msg": "diagnostic"})
    assert handler.filters == [record_filter]
    handler.emit(record)
    assert errors == [record]


@pytest.mark.parametrize("completion", ["finished", "interrupted"])
def test_bar_callbacks_after_closing_keep_working_without_a_terminal(completion):
    reporter, _, messages, clock = bar(period=0.5)
    reporter.started(0.0, 1.0, 1)
    reporter.finished(0.0)
    assert reporter.bar is None
    reporter.plane(0.1, StepSummary(0.1, 1, 0, 0.0, 0.0, 0.1, True))
    clock.now = 1.0
    reporter.step(0.2, 0.1, True)
    if completion == "finished":
        reporter.finished(1.0)
        assert messages[-1].startswith("finished")
    else:
        reporter.interrupted(0.2, KeyboardInterrupt())
        assert "KeyboardInterrupt" in messages[-1]


def test_default_reporter_follows_the_choice_and_the_stream():
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    quiet = default_reporter(choice="auto", stream=io.StringIO())
    assert isinstance(quiet, LogReporter)
    assert not isinstance(quiet, BarReporter)
    assert isinstance(default_reporter(choice="auto", stream=Terminal()), BarReporter)
    assert isinstance(default_reporter(choice="none"), NullReporter)
    assert isinstance(default_reporter(choice="bar", stream=io.StringIO()), BarReporter)
    with pytest.raises((ValueError, TypeError), match="loud"):
        default_reporter(choice="loud")  # type: ignore[arg-type] - the refusal under test


def test_a_jupyter_kernel_counts_as_interactive(monkeypatch):
    class ZMQInteractiveShell:
        pass

    class TerminalInteractiveShell:
        pass

    monkeypatch.setattr(builtins, "get_ipython", ZMQInteractiveShell, raising=False)
    assert is_interactive(io.StringIO())
    monkeypatch.setattr(
        builtins, "get_ipython", TerminalInteractiveShell, raising=False
    )
    assert not is_interactive(io.StringIO())


def test_the_configured_default_beats_auto_and_the_keyword_beats_the_default():
    assert default_progress() == "auto"
    try:
        set_default_progress("log")
        assert default_progress() == "log"
        assert isinstance(resolve_reporter(None), LogReporter)
        assert isinstance(resolve_reporter("none"), NullReporter)
        given = NullReporter()
        assert resolve_reporter(given) is given
        with pytest.raises((ValueError, TypeError), match="loud"):
            set_default_progress("loud")  # type: ignore[arg-type] - the refusal under test
    finally:
        set_default_progress("auto")


def test_a_bar_needs_a_positive_period():
    with pytest.raises(ValueError, match="positive and finite"):
        BarReporter(period=0.0, file=io.StringIO())
    with pytest.raises(ValueError, match="positive and finite"):
        BarReporter(period=math.inf, file=io.StringIO())


def test_the_missing_ipywidgets_warning_is_swallowed(monkeypatch):
    """A notebook without ipywidgets gets the text bar, and no error under -W error."""
    fake = types.ModuleType("tqdm.auto")

    def warn_then_fall_back(name):
        if name == "tqdm":
            warnings.warn("IProgress not found", UserWarning, stacklevel=2)
            return tqdm.std.tqdm
        raise AttributeError(name)

    fake.__getattr__ = warn_then_fall_back  # type: ignore[attr-defined] - PEP 562 on a stub
    monkeypatch.setitem(sys.modules, "tqdm.auto", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reporter = BarReporter(file=io.StringIO(), ncols=60, ascii=True)
        reporter.started(0.0, 1.0, 1)
        assert reporter.bar is not None
        reporter.finished(0.0)


def test_auto_falls_back_to_lines_with_a_warning_when_tqdm_is_missing(monkeypatch):
    """A kernel whose environment predates the dependency still runs, and is told."""

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(solprop.log, "_tqdm_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="needs tqdm.*uv sync"):
        reporter = default_reporter(choice="auto", stream=Terminal())
    assert isinstance(reporter, LogReporter)
    assert not isinstance(reporter, BarReporter)
    # Asked for by name, the bar is not silently replaced.
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    monkeypatch.setitem(sys.modules, "tqdm", None)
    with pytest.raises(ImportError, match="needs tqdm"):
        BarReporter(file=io.StringIO()).started(0.0, 1.0, 1)


def test_logging_configuration_can_choose_the_progress_default(tmp_path):
    package = logging.getLogger("solprop")
    previous_handlers, previous_level = package.handlers[:], package.level
    previous_progress = default_progress()
    # Protect any already-configured handler from configure_logging's replacement.
    package.handlers = []
    try:
        configured = configure_logging(file=tmp_path / "log.txt", progress="none")
        assert configured is package and default_progress() == "none"
    finally:
        for handler in package.handlers:
            handler.close()
        package.handlers = previous_handlers
        package.setLevel(previous_level)
        set_default_progress(previous_progress)


@pytest.mark.parametrize("function", [set_default_progress, default_reporter])
def test_progress_choice_validation_also_runs_without_the_test_type_hook(function):
    # The production value guard matters when beartype is not installed. The
    # session hook otherwise rejects the Literal before reaching this guard.
    with pytest.raises(ValueError, match="loud"):
        if function is set_default_progress:
            unwrap(function)("loud")
        else:
            unwrap(function)(choice="loud")


def test_a_closed_stream_is_not_an_interactive_terminal():
    stream = io.StringIO()
    stream.close()
    assert not is_interactive(stream)


def test_console_bridge_preserves_filters_and_routes_write_errors(monkeypatch):
    original = logging.StreamHandler(io.StringIO())
    selected = logging.Filter("accepted")
    original.addFilter(selected)

    def fail_write(*args, **kwargs):
        raise OSError("closed terminal")

    bridge = solprop.log._BarHandler(types.SimpleNamespace(write=fail_write), original)
    errors = []
    monkeypatch.setattr(
        bridge, "handleError", lambda record: errors.append((record, sys.exc_info()[1]))
    )
    rejected = logging.makeLogRecord({"name": "rejected", "msg": "hidden by filter"})
    accepted = logging.makeLogRecord({"name": "accepted", "msg": "visible"})
    bridge.handle(rejected)
    bridge.handle(accepted)
    assert bridge.filters == [selected]
    assert len(errors) == 1 and errors[0][0] is accepted
    assert isinstance(errors[0][1], OSError)


@pytest.mark.parametrize("ending", ["finished", "interrupted"])
def test_reporter_cleanup_survives_bar_construction_failure(monkeypatch, ending):
    reporter, _, messages, clock = bar(period=0.5)

    def fail_import():
        raise ImportError("terminal unavailable")

    monkeypatch.setattr(solprop.log, "_tqdm_class", fail_import)
    with pytest.raises(ImportError, match="terminal unavailable"):
        reporter.started(0.0, 1.0, 1)
    assert reporter.bar is None
    reporter._emit(clock.now)
    reporter.plane(0.0, StepSummary(0.0, 0, 0, 0.0, 0.0, 0.0, True))
    if ending == "finished":
        reporter.finished(0.0)
        assert messages[-1].startswith("finished")
    else:
        reporter.interrupted(0.0, RuntimeError("setup failed"))
        assert "setup failed" in messages[-1]
    assert reporter.bar is None
