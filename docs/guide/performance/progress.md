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
watching. The bar is the periodic line's content in another form: the fraction and the
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
the bar gives the terminal back. A reporter written to the four calls alone is still
complete and is simply not told.

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
