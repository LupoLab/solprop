# Progress and logging

Progress reports help you judge whether a calculation is advancing, where it is spending
time and whether its local error control is working. A remaining-time estimate is
provisional: compression, ionisation or a changing guide can alter the cost per metre.

## Progress

Choose `progress="auto"` for an interactive bar or a log appropriate to the
execution environment. Use `"log"` for a batch job, `"none"` for silence, or
`"bar"` for an explicit bar. Configure logging when you want progress and warning
records retained in a file. The [execution resources](resources.md) chapter
explains backend and memory choices.

| the process | `progress="auto"` gives | what you see |
|---|---|---|
| a terminal (standard error is a TTY) | {py:class}`~solprop.log.BarReporter` | one line that redraws in place: a bar over z with the counts and the ETA |
| an IPython terminal | `BarReporter` | the same text bar |
| a Jupyter notebook with `ipywidgets` | `BarReporter` | an `ipywidgets` bar |
| a Jupyter notebook without it | `BarReporter` | the text bar in the cell's output |
| redirected (`2> run.log`), piped (`\| tee`), a batch scheduler, `pytest`'s capture | {py:class}`~solprop.log.LogReporter` | the three kinds of `INFO` line below, and never a carriage return |

So nothing is *printed* unless a bar is drawn, and a bar is drawn only where a person is
watching; a bar also prints the run's [summary](#what-a-run-says-before-it-starts) above
it and the closing line below it. The bar is the periodic line's content in another form: the fraction and the
position against the extent (or the steps against a replayed schedule's count), the
elapsed time, and in the postfix the attempted and rejected steps, the current step and
the ETA with the window it came from --

```text
plane 3/21  38.2%|#########              | 0.0573/0.15 m [00:41, 2211 steps (3 rejected), dz 6.1e-05 m, ETA 0:01:05 (from the last 0:00:08)]
```

-- and it is *not* tqdm's own remaining-time estimate, which is the whole-run one and
is measured below to be the worse of the two. Under a bar the per-plane line drops to
`DEBUG` and the bar's description names the plane instead; the closing line is logged
once the bar has closed, and a log record written
while the bar is up goes above it. `progress="bar"`, `"log"` and `"none"` force a
choice per call, `solprop.configure_logging(..., progress="log")` sets the process
default for a script, and a reporter object is used as given. A bar redraws every
0.5 s and ignores `progress_period`; the dependency is `tqdm`, imported when the first
bar is drawn; `ipywidgets` is optional. The gallery examples show no bar: sphinx-gallery
captures standard output and the bar draws on standard error.

The line reporter, {py:class}`~solprop.log.LogReporter`, turns the four events into three
kinds of `INFO` line; they reach the screen only once logging is configured:

```{code-block} python
import logging
from solprop import configure_logging

configure_logging(logging.INFO)
prop_capillary(..., progress="log", progress_period=1.0)
```

```text
INFO solprop.log z = 5.000000e-03 m: 49 steps (0 rejected), mean step 1.020e-04 m, 0.97 s
INFO solprop.log progress 5.4 %: z = 0.0054 of 0.1000 m, 51 steps (0 rejected),
     dz 2.0e-04 m, 0:00:01 elapsed, ETA unknown
INFO solprop.log z = 1.000000e-02 m: 26 steps (0 rejected), mean step 1.923e-04 m, 0.45 s
...
INFO solprop.log progress 49.9 %: z = 0.0499 of 0.1000 m, 282 steps (0 rejected),
     dz 2.0e-04 m, 0:00:05 elapsed, ETA 0:00:05 (from the last 0:00:04)
...
INFO solprop.log finished at z = 1.000000e-01 m: 543 steps (0 rejected) in 9.62 s,
     achieved/requested tolerance ratio 8.456e-02, ETA error 0.5 % over 8 lines
```

The **plane** line is one per saved plane. The **periodic** line exists because a plane
line is no use to a run that has few planes and takes an hour: without it such a run
would be silent from start to finish. It comes at most once per `progress_period`
seconds (10 s by default) however fast the steps are, so the volume of the log is a
property of the run's *duration*, not of its step rate. `progress_period=0` turns it off
and leaves the other two; `progress_period` is a keyword of
{py:meth}`~solprop.driver.simulation.Simulation.run` and of every `prop_*` function, and
there is no environment variable. The **closing** line summarises the run.

### What a run says before it starts

Every run -- `prop_gnlse`, `prop_capillary`, `prop_waveguide`, the free-space routes,
{py:meth}`~solprop.driver.simulation.Simulation.run` -- opens with one block saying what
it is about to compute, so that a default you did not choose is visible before the run
spends any time on it. A mode-averaged argon capillary with a gradient fill, shot noise
and plasma, and a GNLSE fibre, on a terminal:

```text
Starting propagation from z = 0 m to 10 mm
  solprop:        commit fbc01170e186, modified (docs/release-records, installed as 0...
  machine:        arm64, Apple M5 Max, 18 CPUs (18 usable), 128 GiB, Darwin 27.0.0
  backend:        numpy_ref
  threads:        1 (NumPy reference); budget 18 (affinity)
  precision:      double
  grid:           field, 4096 points, window 683.1 fs, band 150 nm to 4 um
  representation: mode-averaged, one mode, scalar
  medium:         Ar, 293.15 K, core radius 125 um
  pressure:       PressureGradient(1 bar to 0.7002 bar over 10 mm, 2 points)
  responses:      Kerr, third harmonic
                  plasma (Ar), rate optmat recommended
                  Raman off
  shot noise:     input model (default), seed 7
  integrator:     Multistep, variable order up to 5, exponential formulation
  propagator:     tabulate
  tolerances:     rtol 1e-06, atol 1e-10, dz_max 5 mm
  absorber:       temporal, nonlinear scheme; edges watched (guards 0.05 of the window, budget 1e-03)
  save planes:    2 after the entrance, 5 mm to 10 mm
  output:         HDF5Output run.h5
  start:          new run
finished at z = 1.000000e-02 m: 228 steps (0 rejected) in 0.64 s, achieved/requested tolerance ratio 8.862e-01
resources: 0.642 s wall, 0.592 s CPU, process peak memory 268 MiB

Starting propagation from z = 0 m to 50 mm
  ...
  representation: GNLSE, one mode, scalar
  medium:         fibre, gamma 0.11 /W/m
  responses:      Kerr
                  Raman off
  shot noise:     input model (default), seed 7
  integrator:     RungeKutta (DOPRI5)
  propagator:     tabulate
  tolerances:     rtol 1e-07, atol 1e-14, dz_max 25 mm
  ...
```

`solprop` names the code: the installed version for a release, and in a development
checkout the commit, whether the tree was modified, the branch and the version it was
installed as, which is the commit of the last install and not necessarily the one
running. `machine` names the hardware, without the host name. After the closing line,
`resources` gives the run's wall and CPU time (every thread's, so their ratio is the
parallel speed-up) and the process's peak resident memory, which in a session running
several runs includes the earlier ones and on a device does not count device memory.
The same values are in the file ([provenance](../results/provenance.md#the-machine-and-what-a-run-spent)).

Raman is always named, `off` when the run has none, with the fraction held back from the
Kerr term when it is on; shot noise is always named, with its model and seed or `off`,
and `(default)` when the model came from the `noise` keyword's default; a default that
resolved to no noise says why, as in `off (default: a Cartesian or line run carries no
noise by default: ...)`. A line longer than 88 characters is cut short with `...`, as the
`solprop` line is above (the `absorber` line is shown in full, as the output records it);
the output keeps every value in full, in its `run_summary` annotation
([provenance](../results/provenance.md)). The `absorber` line also says whether the run
watches its window's edges
([watching the window's edges](../numerics/boundaries.md#watching-the-windows-edges)).
`start` says `new run` or where a resume continues from. The block is built from the
translated problem and the run's metadata
({py:func}`~solprop.driver.summary.run_summary`), so it describes the run as it will
actually be stepped, and it follows `progress=` exactly as the progress lines do:

| `progress=` | the summary and the closing line |
|---|---|
| a bar (`"auto"` where a person is watching, or `"bar"`) | written to the bar's stream, standard error; if logging is configured with a console handler they are logged instead, so they appear once |
| `"log"` (`"auto"` everywhere else) | logged at `INFO` on `solprop.log`, seen once logging is configured |
| `"none"` | nothing |

A run into a plane store (memory or HDF5) also keeps it, as the annotation
{py:data}`~solprop.driver.summary.RUN_SUMMARY_ANNOTATION`
(`handler.annotations["run_summary"]`): a list of `{"title", "rows"}`, to which a resume
adds its own. It is a description for a person; a program should read the run's
[metadata](../results/provenance.md), which holds all of it. The points of a
[scan](../workflows/scans.md) run with `progress="none"` and show no summary.

### The ETA, and why it says where it came from

`ETA 0:00:05 (from the last 0:00:02)` is the estimate **and the window it was taken
over**. Two are formed at every line. The simple one is `remaining z / (z done / elapsed)`
— the whole run so far. The one that is printed is the same
arithmetic over a trailing window of `max(5 periods, 10 % of the elapsed time)`. The
difference matters because the seconds per metre are rarely constant: the steps are short
while a pulse compresses and long once it has, a gradient fill changes both the operator
and the plasma cost along z, and a plane-dense stretch pays for its planes. On three
measured cases the whole-run estimator's error was 1.6, 2.1 and 23 times larger. When
a fixed schedule is being replayed the total step count is known, and the estimate is
the remaining steps at the recent cost per step instead, which is right even where a
rate of metres would not be.

Until the run is past 2 % **and** past two periods the field reads `ETA unknown`, because
an extrapolation from one sample is not an estimate.

The closing line ends with `ETA error 38.9 % over 6 lines`: the median absolute relative
error of every ETA the run published, against what actually happened. It is reported on
every run for the same reason the tolerance ratio is — so that a number you are about to
rely on has been checked rather than assumed. A run too short to publish an ETA closes
without the clause.

`achieved/requested tolerance ratio` is what the run *achieved* against what it *asked
for*: the largest scaled error norm any accepted step scored. It is reported on every run
because the map from a local `rtol` to the global error at the output is problem
dependent and misses by one to three orders in either direction
([stepper.md §3](../../dev/stepper.md)) — calibrate, do not assume. A step that was too
coarse for the tolerance it is scaled by is logged as a warning naming the plane it
happened before. That is expected on a replayed schedule, which is not free to shorten
its steps; on an adaptive run it can only come from a controller that accepts over
tolerance (`reject_free`), because the history records the norm each step actually
scored, including a landing step clipped short onto a plane. The RDW case with a
true-vacuum end, whose last four steps collapse from half a millimetre to three microns
against a fill that reaches exactly zero, closes at a ratio of `1.6e-01`.

### Writing your own

Progress goes through one {py:class}`~solprop.log.ProgressReporter` protocol:
`started(z_start, z_end, n_planes)` once, `step(z, dz, accepted)` once per **attempted**
step from inside the backend's own loop, `plane(z, summary)` once per saved plane, and
`finished(total)` once. Subclass {py:class}`~solprop.log.NullReporter` and override the one call you want; the
rest stay no-ops. A fifth call, `interrupted(z, error)`, reaches a reporter that has it
({py:class}`~solprop.log.InterruptibleReporter`; `NullReporter` supplies a no-op) when
the loop raises, `KeyboardInterrupt` included, before the error propagates -- it is how
the bar gives the terminal back. A sixth, `summary(text)`
({py:class}`~solprop.log.SummarisingReporter`; `NullReporter` supplies a no-op), is
handed the start-of-run block before the first step, and the `resources` line after
`finished`. A reporter written to the four
calls alone is still complete and is simply not told.

```{code-block} python
from solprop.log import NullReporter

class Bar(NullReporter):
    def plane(self, z, summary):
        widget.update(z)

prop_capillary(..., progress=Bar())
```

Pass it as `progress=` and `progress_period` becomes your reporter's business. `step` is
called from inside the step loop, thousands of times a second on a small problem, so
anything it does must be cheap or be put on a clock of its own — the line reporter's
own `step` costs about 100 ns, five times the bare cost of the call.
Nothing a reporter does may change the run: the calls are observers made after each step
is decided, and a run with a reporter is the same computation as a run without one.
