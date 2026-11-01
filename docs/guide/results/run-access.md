# Accessing and continuing a run

After propagation, inspect the saved field and its diagnostics together. This chapter
explains the output object, statistics and file continuation. For the first worked
analysis, follow the [RDW lesson](../../introduction/lessons/plot_01_rdw.rst); for
derived spectra and pulses, continue to [processing](processing.md).

## What comes back

An output handler holds the run: the positions, the stored spectra, the statistics and
the step history. Everything is indexed by name.

```{code-block} python
z = output["z"]                          # (n_z,) in m, beginning at the launch position
spectra = output["field"]                # (n_z, n_modes, n_pol, n_omega), optics convention
internal = output.fields("fft")          # the same arrays as a backend works in
background = output["noise_phase"]       # the shot-noise phase, with a noise model only
energy = output["stats/energy"]          # (n_z, n_modes, n_pol) in J
steps = output["step_history/counts"]    # attempts belonging to each plane
rows = output["step_stats/energy"]       # (n_rows, n_modes, n_pol), with a periodic spec
```

Stored spectra are in the **optics convention** and the file says so
(`spectral_convention = "optics"`): a spectral phase read out of one is referenced to
`t = 0` and has the sign a textbook gives it ([conventions.md](../../dev/conventions.md)
§2). The driver itself never conjugates anything; the output layer does it once, on the
way to storage. `fields("fft")` hands back the internal-convention arrays, which is what
an inverse transform expects. To turn the stored arrays into spectra, pulses and pulse
metrics, use [processing](processing.md) and [plotting](plotting.md); the file layout
is on [the output page](output.md).

The launch plane (normally `z = 0`) is saved before any step is taken, with an
empty step history, so
`output["field"][0]` is the input as the run actually started from it — masked to the
grid's active band, and with the input-model shot noise already added if there is any.
By default later planes are reached by landing on them. With
`interpolate_planes=True`, permitted save planes can instead be interpolated
inside accepted steps; this adds interpolation error and has documented restrictions ([the solver](../numerics/solver.md#caps-and-landing)).

### Amplitudes carry true power

A requested `energy=` or `peak_power=` is the energy or power the pulse carries as
Poynting flux in the guide, whatever the fill and its pressure, and so is every saved
field, energy, power and intensity. Internally the modal state keeps a
vacuum-normalised amplitude: the launch and the noise are divided by
$\sqrt{\operatorname{Re} n_\text{eff}}$ on the way in, and every saved plane and every
resume applies the inverse pair at its own position. The two differ by
$\operatorname{Re} n_\text{eff}$ in energy: 0.56 % at 20 bar of argon or 40 bar of
hydrogen, 0.03 % at 1 bar of argon. The GNLSE's unit prefactor index
(`prefactor_index=1.0`) makes the weight one, so that route is unaffected.
[Power and intensity](power-and-intensity.md) has the definitions.

## Statistics

Statistics are evaluated at every saved plane by the backend, in one pass over its own
buffers, and stored beside the field as `stats/<name>`. A run records
{py:func}`~solprop.stats.default_spec` unless a {py:class}`~solprop.stats.StatsSpec` is
passed as `stats=`; a spec with a `periodic` subset also records those statistics
between the planes, as `step_stats/<name>`. [Statistics](statistics.md) says how to ask
for them, what each costs and what may be recorded between planes, and
[the catalogue](statistics-catalogue.md) lists every name with its shape and unit.
`zdw` needs the modes and `density`, `pressure` and `core_radius` need the z-profiles,
which the driver passes to the backend, so they are available wherever the run has them.

## Writing to a file, and resuming it

Pass a path and the run streams to HDF5 instead of memory, one plane at a time:

```{code-block} python
output = prop_gnlse(..., output="run.h5")
```

The handler is closed when the run ends, so read the file back with
{py:func}`~solprop.output.hdf5.open_file` rather than through the returned handler; a
`MemoryOutput` stays readable.

If that run is interrupted — `Ctrl-C`, a crash, a queue limit, a laptop lid — the file
holds the entrance and every plane completed before the interruption, because the run
loop closes the handler in a `finally`; it can be opened and plotted as it is. Each plane
is also flushed to disk as it is saved, so a hard stop the loop cannot handle normally leaves the last flushed saves.
Filesystem or storage failure can still prevent recovery; a flush is not a
general durability guarantee against loss of the storage itself.
An exception the loop can handle also records a separate accepted terminal
checkpoint and an explicit failure or budget status, with the Adams march's state
beside it. It requests no extra material statistics while recovering. A hard kill
retains the last flushed regular save.
See [completion and checkpoints](resuming.md#completion-status-and-terminal-checkpoints).

**A scheduler's time limit.** Slurm, and most schedulers, send `SIGTERM` some seconds
before they kill a job. During a run it raises
{py:class}`~solprop.exceptions.RunTerminated` at once, abandoning a trial in flight
without its result, or, if a step's state, a plane or the march's snapshot is being
written, as soon as that commit is complete; `Ctrl-C` is held the same way. The run then
writes its terminal checkpoint and the march's state, as for any failure, and
`resume=True` continues it -- on `numpy_ref` and `numba_cpu` from the march's own
state -- so a job
killed at its limit loses only the step it was taking, not the steps since the last
plane. `RunTerminated` is a `SystemExit` with status 143, so an uncaught one ends the
script as a terminated process. The handlers are installed for the run alone, from the
main thread, and only where Python's defaults are in place: an application's own
`SIGTERM` handler is kept. Ask for a grace period longer than one step: Slurm's is
`KillWait`, or `--signal=TERM@<seconds>` sends the signal earlier.
An interrupted run with `output=None` is different: its planes sit in a `MemoryOutput`
that the raised exception never returns, so to keep them pass `output=MemoryOutput()`
yourself and hold the reference, or give a path. The same call with `resume=True`
continues a file:

```{code-block} python
output = prop_gnlse(..., output="run.h5", resume=True)
```

The file is checked before anything is written:
{py:func}`~solprop.output.hdf5.check_cache` compares the record of the run about to
start with the one the file carries, path by path. A file written with a different
layout, physics, guide or materials is refused by what changed; a changed tolerance or
integrator continues while there is work left, with a warning, and provenance such as
the optmat version is ignored; `resume="force"` continues across a changed physics in
the same file ([what a resume compares](resuming.md#what-a-resume-compares)). When it
continues, the
terminal checkpoint, or the last regular plane when no checkpoint exists, is read
back, converted to the internal convention and loaded at its actual absolute
position. The next scheduled plane is found from that position. Call ledgers are
restored before any response evaluation.

The key covers **what was computed and not how**. The backend's name is deliberately
not in it: a run started on one backend and continued on another agrees with an
uninterrupted run to about 2e-16, which is smaller than the difference between two
uninterrupted runs on the two backends, so it is the same run and refusing it would
protect nothing. The package versions and the git provenance are outside the digest for
the same reason, so that a resume survives an unrelated commit or a NumPy upgrade.

What *did* execute the run is recorded beside the digest rather than inside it, in the
`backend` group of {py:class}`~solprop.output.schema.RunMetadata`: the backend's name,
the propagator mode it resolved, and whatever else it reports — for `numba_cpu` the
threading regime, the four pool counts, the plan flag and the FFTW wisdom digest. The
thread configuration alone has been measured to change the time of one step by a factor
of nine, so a result that does not say how it was executed is not reproducible even when
it is correct. The backend's name is also a root attribute
of the HDF5 file, where a reader looks first. A resumed file keeps the record it was
opened with, so the backend it names is the one that *started* the run.

Two things follow from how the stepper is restarted. A run replaying a fixed schedule
([fixed schedules](../numerics/fixed-schedules.md#fixed-schedules-discover-calibrate-replay)) takes the same
steps whatever it is resumed from, and agrees with the uninterrupted run to round-off
(`numba_cpu` and `native` reuse a step's last stage as the next one's first and have none
to reuse after a resume). An adaptive Runge–Kutta run does
not: the controller starts again from `dz_init` rather than from the step it was on, so a
resumed run agrees with an uninterrupted one to the tolerance it was asked for, not to
the last bit. An adaptive Adams run -- the default -- continues from the march's state
the output keeps at each plane and checkpoint, and on `numpy_ref` and `numba_cpu`
continues the uninterrupted run's computation, agreeing with it to rounding
([the Adams integrator](../numerics/adams.md)). No resume promises identical bits
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).

For a resume to work in a *new process*, everything in the configuration has to hash the
same way twice. Objects that carry their data — the pulses, the profiles, the responses,
the dispersion-specified mode — do. A configuration holding a bare callable does not:
{py:func}`~solprop.output.schema.serialise` records one as its `repr`, which carries a
memory address. Prefer a specification over a closure wherever there is a choice.
