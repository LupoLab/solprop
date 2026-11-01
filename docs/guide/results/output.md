# Output: what a run leaves behind

Save the field, its diagnostics and the configuration together. Small runs can
stay in memory; longer runs can write planes to HDF5 as they are produced.
[Accessing a run](run-access.md) gives the common workflow. Use
[processing](processing.md) for physical spectra and pulse metrics, and
[plotting](plotting.md) for figures.

This chapter describes storage and field conventions. [Provenance](provenance.md)
explains the recorded model and materials; [resuming](resuming.md) explains
compatibility and checkpoints. Retain the file and its source script when a
result needs to be reproduced.

A modal file stores **true-power** amplitudes $b = \sqrt{\operatorname{Re} n_{\rm eff}}\,a$,
where $a$ is the backend's vacuum-normalised internal amplitude, and marks the `field`
dataset with `power_convention = "poynting"`. A resume divides the stored plane by the
same square root before loading it into the backend, and the stored noise realisation is
the true-power draw. See [power and intensity](power-and-intensity.md) for the
definitions and for how this amplitude's squared modulus differs from the flux of a
broadband pulse.

The run record of a capillary run carries that weight, so
{py:func}`solprop.processing.mixtures.get_power_weight` needs nothing but the file. It
is stored **factored** — a few frequency-resolved terms and one short basis row per
plane, whose product is bit for bit the translation's own evaluation — because the
record is serialised into the hashed JSON configuration, where one state-sized array
per plane would come to 26 MB for 201 planes of four modes on 2049 bins.


## Two handlers, one protocol

{py:class}`~solprop.output.memory.MemoryOutput` keeps everything in RAM. It is the
handler for fast, small problems and for the members of a scan, where a file per run
costs more than the run. Its plane-indexed arrays are allocated once, for the planes the
run will save, and grow by doubling only if it saves more, so a hundred planes cost one
allocation rather than a hundred copies. That store is part of the run's memory: the
[memory admission](../performance/resources.md#memory) charges it, at its planned size, as the group
`output`, and charges a file output nothing.

```{note}
For a run that records statistics between the save planes,
{py:meth}`~solprop.output.memory.MemoryOutput.planes` raises
{py:class}`~solprop.exceptions.OutputError` ("step statistics columns must have equal
row counts"): a column created when the first plane carrying rows arrives is short by
the planes saved before it. Read such a run through its accessors
(`output["field"]`, `output["step_stats/<name>"]`) instead.
```

{py:class}`~solprop.output.hdf5.HDF5Output` streams to one file, **opened once and kept
open** between snapshot transactions. Datasets are chunked one plane at a time and extendable along
`n_z`, so the writer never holds more than the current plane and a reader can pull one
plane out without touching the rest. They are created at the first save for every plane
the run will save, and the file's `n_saved` attribute counts the planes written (below).

The run loop tells either store how many planes to expect with
{py:meth}`~solprop.output.schema.PlaneStore.reserve`; a handler driven by hand works
without it, growing as it goes.

Both are {py:class}`~solprop.output.schema.PlaneStore` subclasses, and everything that
must not differ between them lives in that base class: the conversion applied on save,
the metadata overwrite rule, the accessor keys, the step-history layout, and the check
that every plane fits the layout the first one fixed (the field's shape, the statistic
names, whether a noise phase travels with it), refused by name in either handler. That is
what makes the two store a run *bitwise identically*, which is a test, and what makes
{py:meth}`~solprop.output.memory.MemoryOutput.to_file` able to write exactly the file the
other handler would have written.

A driver uses a handler like this (the entry points of [running](../setup/running.md) do it for
you when given `output=` a path or a handler):

```{code-block} python
from solprop.output import HDF5Output, MemoryOutput

handler = HDF5Output("run.h5", compression="lzf")
handler.open(metadata)          # once, at the start of the run
for plane in driver.run():      # a SavedPlane per save position
    handler.save(plane)
handler.close()                 # idempotent
```

## Statistics-only planes: which planes keep their field

A run that is wanted as spectra and diagnostics need not store a field at every plane.
`SimulationSettings(fields=..., field_planes=...)`
([SEP 22](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md)) chooses, with the
vocabulary {py:func}`~solprop.driver.scan.run_scan` already uses:

| `fields` | what each saved plane keeps |
|---|---|
| `"all"` (the default) | its field, its noise phase, its statistics and its step history |
| `"planes"` | the field on the saved planes `field_planes` names (the entrance is 0, negative indices count from the end), and the statistics and step history on every plane |
| `"none"` | the statistics and step history alone |

A run is **homogeneous**: it keeps its fields by one rule, declared to the output before
its first plane ({py:meth}`~solprop.output.schema.PlaneStore.retain`), and a plane that
breaks it is refused by name. A plane without its field is not copied, converted or
stored, and an in-memory output allocates no field block for it and is charged none of
it by the [memory admission](../performance/resources.md#memory): a statistics-only run
is charged its statistics. The march is unchanged — the statistics are computed from
the live state either way — so a statistics-only run's numbers are those of the same run
keeping every field. A plane that keeps no field is observed without one: nothing
crosses from the backend (on a device, no copy is made) and nothing scans the field for
finiteness, so a diverged run is caught by the step loop and its statistics.
`fields="planes"` is refused beside `step_planes`, whose plane count is not known in
advance. Beside `LowStorageRungeKutta()`, which accepts each step in place and so has no
state to checkpoint for a step abandoned in flight, the backend holds the state of the
last field-free plane, one more copy of the state, charged at
[admission](../performance/resources.md#memory) as `held_plane`: a run stopped in flight
after such a plane writes its terminal checkpoint from it and resumes there.
`fields` is outside the configuration hash, like
`resume_history`: it changes what is kept, not what is computed, and it may be changed
after the simulation is translated.

```{code-block} python
import dataclasses

simulation.settings = dataclasses.replace(simulation.settings, fields="none")
output = simulation.run()
output["stats/on_axis_spectrum"]        # every plane's statistics, as with every field kept
output["field_planes"]                  # the saved planes that kept a field: none
```

Reading a field that was not kept is refused, naming the run's `fields`:
`output["field"]`, `output["noise_phase"]` and
{py:meth}`~solprop.output.schema.PlaneStore.fields` need every plane's field and answer
only for an `"all"` run; {py:meth}`~solprop.output.schema.PlaneStore.plane_field` reads
one saved plane's field and works for any plane that kept one; `output["field_planes"]`
lists them. {py:mod}`solprop.processing` reads one plane through `plane_field`, so a
processing function asked for a kept plane works and one asked for another plane is
refused by name. {py:meth}`~solprop.output.memory.MemoryOutput.planes` returns
`field=None` where no field was kept. A run that stops on an error or a signal still
writes its terminal checkpoint **with** the field ([resuming](resuming.md)).

In this phase the backend still observes the field at every plane, and the driver drops
it: the storage, the preallocation, the record copies and the conversion go, the
device transfer and the collar diagnostic of a Cartesian run remain until the
observation itself can skip the field.

## What goes in, and in which convention

A {py:class}`~solprop.output.schema.SavedPlane` is what a driver hands over: the position
`z`, the field there as `(n_modes, n_pol, n_omega)` in the **internal** FFT convention
(or `None` on a plane that keeps no field), the statistics evaluated at that plane, the
step history since the previous plane, and the accumulated noise phase when a noise
model is running and the plane keeps its field.

The handler converts the field on the way out, because the file is where the internal
FFT convention meets the user's optics convention: stored spectra go through
{py:func}`~solprop.fourier.spectrum_to_optics`, which conjugates *and* moves the phase
origin to the centre of the time window, so that

$$ \tilde E_{\text{optics},k} = (-1)^k\,\overline{\tilde E_k} $$

and a stored spectral phase uses the declared optics convention and time origin.
An ordinary intensity spectrometer does not itself measure that phase. The `field` dataset
and the file itself carry the attribute `spectral_convention = "optics"` to say so.
{py:meth}`~solprop.output.schema.PlaneStore.fields` with `convention="fft"` converts
back, to the form a backend works in.

Two things stay internal, and are marked so rather than converted: the **noise
realisation** in {py:class}`~solprop.output.schema.NoiseRecord` and the per-plane
**noise phase**. Neither is a user-facing spectrum — they are the object the stepper adds
at every stage and the phase that advances it (see [noise](../media/noise.md)) — and
conjugating them would make them disagree with the state they have to be added to.

```{note}
Energies, spectral densities, masks and windows are convention-free: they depend on
$|\tilde E|^2$ only. Only an absolute spectral phase distinguishes the two conventions,
and that is exactly what the boundary exists to get right.
```

## The metadata overwrite rule

Metadata is written once.
{py:func}`~solprop.output.schema.resolve_metadata_write` is the single rule both handlers
obey: a second record describing the same run — the same configuration hash — is a no-op;
one describing a different run raises and names what changed; `force=True` replaces it
and logs a warning.

## What is on disk

For a modal run:

```
/                       attrs: schema_version, spectral_convention, field_layout,
                               configuration_hash, simulation_kind, grid_kind, created,
                               n_saved, fields, field_shape
/meta/json              the whole RunMetadata as one UTF-8 JSON string
/meta/outcome           status, call ledgers, execution provenance, machine and resources (JSON)
/meta/annotations/<name>  optional post-run annotations, one JSON string each
/z                      (n_z,)                              plane positions, m
/field                  (n_z, n_modes, n_pol, n_omega)      complex, optics convention
/noise_phase            (n_z, n_modes, n_pol, n_omega)      rad, FFT convention
/field_planes           (n_kept,)                           fields="planes" only
/stats/<name>           (n_z, ...)                          one dataset per statistic
/step_history/counts    (n_z,)                              attempts belonging to each plane
/step_history/z         (n_attempts,)                       every attempt of the run,
/step_history/h              "                              concatenated in order
/step_history/norm           "
/step_history/accepted       "
/step_history/wall_time      "
/step_stats/counts      (n_z,)                              rows belonging to each plane
/step_stats/z           (n_rows,)                           every row of the run,
/step_stats/step             "                              concatenated in order
/step_stats/<name>      (n_rows, ...)                       one dataset per statistic
/terminal_checkpoint/  optional; attrs: z, next_save_index, landed
  field                 (n_modes, n_pol, n_omega)           complex, optics convention
  stats/<name>          already available terminal statistics
  step_history/<column> attempts since the last regular save
  step_stats/<name>     already available periodic statistics
/march_snapshot/       optional; attrs: z (NaN while being written), method, order,
                       rejections, last_rejected, proposal, identities (JSON)
  state                 (n_batch, n_modes, n_pol, n_omega)  the accepted state, internal
  positions, slots      (n_entries,)                        the history's entries
  rows/<name>           (n_slots, ...)                      the engine's rows
```

**Field retention.** The root attribute `fields` is the run's rule and `field_shape`
the shape of one plane's field, stored or not. A `"planes"` run's `field` and
`noise_phase` hold exactly the kept planes, in order, one row each, and `field_planes`
lists their saved-plane indices; a `"none"` run has neither dataset.

**The saved-plane count.** The plane-indexed datasets (`z`, `field`, `noise_phase`,
`stats/*` and the two `counts`) are created for every plane the run will save, and the
step-history and between-planes columns grow by doubling; a run that saves more planes
than planned grows the plane-indexed ones by doubling too. The root attribute `n_saved`
is the number of planes actually written. Each plane's data is written and flushed
first and the count raised last, so a hard stop — a SIGKILL, an out-of-memory kill, a
lost node — leaves the count at the last complete plane, even part-way through the next.
The attempts and rows belonging to the counted planes are the sums of
`step_history/counts` and `step_stats/counts` over them, so the one count fixes every
column. Every reader, {py:func}`~solprop.output.hdf5.resume` and a handler continuing
the file use only what is counted, and a clean close trims every dataset to it, so a
finished file holds no unused rows and its datasets' extents are exactly the counts.
Read a file through {py:func}`~solprop.output.hdf5.open_file`, not
`h5py` directly, if it may be one a hard stop left behind.

The march snapshot is the Adams march's state at the last saved plane or terminal
checkpoint, rewritten in place at each one (`SimulationSettings.resume_history`). It is
outside the record and the configuration hash: it says where the march stood, not what
was computed, and a resume uses it only at its own position, with the same method and
on an engine that can load it, starting the history afresh otherwise.

A spatial run stores its field on its own transverse axes, described by the `spatial`
part of the record ([free-space propagation](../free-space/model.md)).

The identifying scalars are root attributes, where a reader looks for them first; the
whole record is a *dataset* rather than an attribute because HDF5 caps an attribute in
the object header at 64 kB, and a serialised configuration — a capillary with a
tabulated cladding glass, a stored noise realisation — goes past that.

Compression is `None`, `"gzip"` (with a level, `compression_opts`) or `"lzf"`: the
filters h5py ships, so a file opens with any h5py installation and needs no filter plugin
([dependencies.md](../../dev/dependencies.md)).

Both handlers answer the same accessor keys:

```{code-block} python
handler["field"]                  # (n_z, n_modes, n_pol, n_omega), optics convention;
                                  # refused unless the run kept every field
handler.plane_field(-1)           # one saved plane's field, if that plane kept one
handler["field_planes"]           # the saved planes that kept a field
handler["z"]                      # (n_z,)
handler.stat_names                # the statistics stored, sorted
handler["noise_phase"]            # (n_z, n_modes, n_pol, n_omega), rad, FFT convention,
                                  # only when the run carried a noise model
handler["stats/energy"]           # (n_z, n_modes, n_pol)
handler["step_history/norm"]      # every attempt, concatenated
handler.step_stat_names           # the statistics recorded between planes, sorted
handler["step_stats/energy"]      # (n_rows, n_modes, n_pol); empty without a channel
handler.fields(convention="fft")  # the internal arrays, exactly as they went in
```

## Step history

Every attempt the stepper made is recorded: its position, its length, its scaled error
norm, whether it was accepted, and how long it took. It is recorded whether the run was
adaptive or replayed a fixed schedule, because the embedded estimate costs nothing extra
and a replayed run still needs it: a largest scaled norm above one means a step was too
coarse for the tolerance the norm is scaled by ([stepper.md](../../dev/stepper.md) §8).
A rejected attempt that diverged, whose propagator or error overflowed (a long step
under a strong collar, more easily in single precision), records an infinite norm; the
stepper retried it at a much shorter step, so take the largest norm over the accepted
attempts. {py:func}`~solprop.output.hdf5.resume` returns the history recorded so far with the last
plane, and an appending handler numbers the next attempt where the file left off, so a
resumed run's file keeps one continuous record.

## Statistics between planes

A run whose {py:class}`~solprop.stats.StatsSpec` names a `periodic` subset also records
those statistics every `period` accepted steps, and they are stored in the same ragged
shape as the step history: flat columns for the whole run, with `step_stats/counts`
saying how many rows belong to each plane, so a resume appends and a scan needs no
padding. `step_stats/step` is the index of the accepted step a row was taken after, into
that plane's own step history — which counts rejected attempts, so it is not
`row * period`. The group is present, and empty, for a run that recorded no subset
([statistics](statistics.md#between-the-planes)).

## The TDSE response archive

A run with a TDSE response can write a second, append-only HDF5 file beside its output,
the raw response archive (`run(..., tdse_archive=TDSEArchive("run.tdse.h5"))`), holding
every provider evaluation with its exact input and raw record. The optical file stays
complete without it and records only its schema, run identifier and relative path; read
it with `open_tdse_archive` ([the TDSE response](../media/tdse.md#the-raw-response-archive)).

## The schema version

Every file records its `schema_version`; this release writes and reads output
schema **2** (`solprop.output.SCHEMA_VERSION`). Schema 1 was the first released layout,
that of 0.1.0; schema 2 adds `spectral_phase` to the record of a
{py:class}`~solprop.pulses.DataPulse`. A file of any other schema is refused by name
wherever it is opened, read or resumed: files written by an earlier release or by
development versions of solprop, which declared output schemas up to 28, are not read,
and nor is a file from a later release. Read such a file with the solprop version that
wrote it. A resume needs the file's schema to be this one, and checks that before it
compares the records ([what a resume compares](resuming.md#what-a-resume-compares)).

Every run records its `precision` (`"double"` or `"single"`,
[single precision](../performance/single-precision.md)) in its configuration, so a run
is never resumed in the other precision, and its plane-reference record
(`RunMetadata.reference`, [provenance](provenance.md#the-metadata-schema)), from which
a saved plane is read back as a coherent field plane
([coupling simulations](../workflows/coupling.md)).

How a run resumes from its last plane depends on what that plane is:

- **A run that saved its steps** (`step_planes`) resumes from its last plane,
  a step's end, like any landed plane; the requested planes still to come are those
  beyond it.
- **A dense run resumes from an interpolated plane only on request.** Its planes, but
  the end and the hard kinks, are interpolated inside steps, not states the march
  passed through. A plane is interpolated when no accepted step of the run ends on it.
  When the file's last plane is one, `resume=True` is refused, with either stepper and
  whatever `interpolate_planes` the resuming run asks for;
  `resume="interpolated"` (or `"force"`) continues afresh from it and
  appends its position to the annotation `interpolated_resumes`
  (`INTERPOLATED_RESUMES_ANNOTATION`), outside the configuration hash.

**Every field is recorded.** A record holds every field of every dataclass it
serialises, at its default too, so a change that adds a field to a recorded dataclass
moves the configuration hash. A resume compares the records path by path, so the new
field refuses a resume of an earlier file unless a release declares its neutral value,
the value that reproduces the code before it existed
([what a resume compares](resuming.md#what-a-resume-compares)). The test suite pins the
records of a modal, a radial and two Cartesian runs, and of a custom absorber at its
defaults, and keeps each release's whole records, so that such a change is deliberate.
An array input of more than 2²⁰ elements is recorded by the SHA-256 of its bytes, with
its dtype and shape (`ArrayDigest`), rather than inline.

**When the schema is bumped.** Only for a change an older reader would misread, or
could not append to: a dataset renamed, re-laid or given new units; an attribute that
changes meaning; anything added inside `stats`, `step_history` or `step_stats`; or
anything grown per plane. A new field of the JSON run record, a new root attribute or a
new top-level group written once does not bump it: readers already tolerate them, and a
resume decides by comparing the records (ADR-063,
[SEP 42](../../dev/seps/sep-0042-tolerant-resume.md)).

## A worked example

```{code-block} python
import numpy as np
from solprop.grid import EnvelopeGrid
from solprop.output import HDF5Output, MemoryOutput, resume
from solprop.output.schema import GridRecord, RunMetadata, SavedPlane, StepperRecord
from solprop.stepper import DOPRI5, PIController

grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
metadata = RunMetadata(
    grid=GridRecord.from_grid(grid),
    modes=(),
    simulation_kind="mode_averaged",
    configuration={"gamma": 0.11},
    stepper=StepperRecord.from_settings(DOPRI5, PIController(rtol=1e-8, atol=0.0)),
)

memory = MemoryOutput()
memory.open(metadata)
memory.save(SavedPlane(z=0.0, field=np.ones((1, 1, grid.n_omega), complex)))
memory.to_file("run.h5", compression="lzf")

state = resume("run.h5")
assert state.plane.z == 0.0
assert np.array_equal(state.plane.field, memory.planes()[0].field)
```

## Periodic snapshots while computing

{py:class}`~solprop.output.snapshots.SnapshotController` makes an immutable local
snapshot without rebuilding the backend, TDSE binding, native model or response
plans. Pass it as `snapshots=` to `Simulation.run` or a propagation convenience
function. The default, `None`, leaves the run unchanged. This is an execution policy,
outside the scientific configuration hash; `SimulationRunner`'s restriction on
reusing TDSE across inputs still applies.

A snapshot requires an unbatched HDF5 run with `step_planes=1`,
`fields="all"` and `interpolate_planes=False`. Adams runs must allow retained
history: explicit `resume_history=False` is refused, and a requested snapshot keeps
history even above the automatic retention-size cutoff. Every backend's Adams history
is supported, the native backend's resident one included: its engine copies the
history's rows to the host for the snapshot. Snapshots are qualified by fault
injection and fresh-process resumes on NumPy and Numba, on native CPU Adams and
Runge–Kutta runs and on short native TDSE windows; resuming a haCC snapshot in a fresh
process has not been qualified.

A request is serviced after an accepted endpoint, its archive trial decision,
optical observation, march state and cumulative execution record have committed.
It neither adds a save landing nor performs another observation or atomic evaluation,
and it does not change adaptive stepping. A request at the final endpoint still
records `running`; completion is recorded only when the live run finishes.

This short example requests one snapshot explicitly and restores it. In a fresh
process, reconstruct the same `Simulation` before the last call.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from solprop.driver import Simulation
from solprop.output import SnapshotController, read_snapshot, restore_snapshot

simulation = Simulation.from_gnlse(
    0.11,
    0.001,
    [0.0, 0.0, -1.183e-26],
    wavelength=835e-9,
    wavelength_limits=(500e-9, 2e-6),
    trange=1.4e-12,
    width=100e-15,
    peak_power=9.8,
    shape="sech",
    backend="numpy_ref",
    step_planes=1,
    dz_init=1e-4,
    dz_max=1e-4,
)
with TemporaryDirectory() as temporary:
    directory = Path(temporary)
    snapshots = SnapshotController(directory / "snapshots", interval_seconds=1800)
    snapshots.request()
    simulation.run(directory / "working.h5", snapshots=snapshots, progress="none")
    attempt = snapshots.last_result
    assert attempt is not None and attempt.disposition == "published"
    info = read_snapshot(attempt.generation)
    files = restore_snapshot(info.directory, directory / "restored")
    simulation.run(
        files.output, resume=True, tdse_archive=files.tdse_archive, progress="none"
    )
```

`interval_seconds` is positive and finite, measured with a monotonic clock from the
start of propagation, after preparation. It restarts after successful publication.
`request()` only sets a coalescing request flag. An application's signal handler may
call it; solprop installs no signal handler to request snapshots automatically:

```text
previous = signal.signal(signal.SIGUSR1, lambda number, frame: snapshots.request())
try:
    simulation.run("working.h5", snapshots=snapshots)
finally:
    signal.signal(signal.SIGUSR1, previous)
```

The frozen `last_result` describes the latest attempt: `published`, `deferred` or
`failed`, generation path if published, accepted position in metres, bytes copied,
copy/verification time and total pause time in seconds, and any failure. Failures and
capacity deferrals are logged, without Python warnings.

### Files, ownership and publication

At the boundary, solprop flushes and closes just the two HDF5 handles, retaining
both writer objects and their persistent advisory locks. It copies the closed
optical file and, when referenced, the TDSE archive with bounded buffers. Their
relative paths are preserved. Destination checksums and checkpoint identities are
verified before a versioned `manifest.json` is written last and the staging directory
is atomically renamed. Published files are read-only. The writers then reopen and
validate the working files in the same archive session before computation continues.
No native resource is closed during this pause.

The manifest records configuration/provider identities, native execution provenance,
archive UUID and session, position, next save index, saved-plane count, cumulative
logical/native accounting, relative filenames, sizes and SHA-256 checksums. The
manifest schema is independent of the existing optical and archive schemas. Its
`versions` and `git` fields describe the current execution, including after a resume;
the optical file keeps its original creation metadata. A directory
without a published manifest, including a hidden `.snapshot-*` directory, is
incomplete. Never resume in a published directory: use
{py:func}`~solprop.output.snapshots.restore_snapshot` to create writable working files.

Optical writes, closed-file annotations, archive writes and snapshot-store ownership
all use persistent `.writer.lock` sidecars. Competing writers fail immediately.
The locks remain held while HDF5 handles are closed; sidecar existence by itself does
not mean a writer is running. Do not remove these files to bypass ownership. Use a
filesystem that supports advisory locking and atomic directory rename.

### Bounded external transport

`max_snapshots=3` bounds all completed and incomplete generations, including staging;
the minimum is one. Only one copy runs at once. At capacity, computation continues
with one pending request. Solprop never deletes completed generations. Copy failures
and capacity deferrals retry at an eligible boundary after 60 seconds; a new explicit
request permits an earlier attempt. An unremovable failed staging directory still
counts against capacity.

Transport belongs in a separate campaign controller. For example, the following
pseudocode uses one upload at a time, verifies remote checksums before retiring local
copies, and retains the preceding verified generation. It must not upload the live
working files or change a generation that an upload is reading.

```text
for generation in published_manifests_ordered_by_checkpoint(store):
    info = read_snapshot(generation)       # ignore hidden staging directories
    upload_and_verify(info.directory)     # synchronous: at most one upload
    record_verified_remote_copy(info)
    retire_older_local_generations(keep_latest_verified=2)
```

This controller runs after publication while propagation continues. Retirement frees
capacity for the pending request; it should coordinate with its own upload workers
and keep a verified predecessor. No transport client, cloud credentials or AWS
library is part of solprop.

### Interruptions and failures

TERM and Ctrl-C wait through checkpoint persistence, suspension, copying, publication
and reopening. Pending termination is delivered after safe reopening, before another
step or atomic call. Application handlers are restored and their deferred calls
retain the application's behaviour. Signal latency includes both the current native
operation and the snapshot transaction; there is no hard shutdown deadline.
As with ordinary run signal handling, this deferral applies when propagation runs
in Python's main thread.

A copy, checksum or publication failure preserves prior generations. The run may
continue only after both working writers reopen and validate. For example, a full
snapshot disk produces a logged `failed` result and a delayed retry; a controller
can free capacity and call `request()` to retry sooner. Checkpoint writing, handle
closure or reopening failures stop the run clearly. The primary failure is retained
and cleanup failures are attached. A generation already published remains usable
even if the working files fail to reopen. A hard kill can leave a hidden incomplete
generation, but earlier complete generations remain independently verifiable.

A failed handle close retains writer ownership. Calling the writer's `close()`
again retries cleanup, including after a failed post-run annotation. If archive
creation fails before it can return a writer and its handle also cannot close,
ownership remains held until process exit; the exception reports both failures.
