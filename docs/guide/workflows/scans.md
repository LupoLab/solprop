# Parameter scans

A parameter scan runs the same simulation over a grid of inputs — pressures, energies,
durations, seeds — and collects what each point produced into one HDF5 file. This page
is for anyone who wants more than a handful of runs: it describes how a scan is declared,
how the points are built and executed (one after another, in parallel processes, in
lockstep batches, or spread over cluster jobs), what is kept of each point, and how the
collected file is read back.

The workflow has four parts:

1. declare the axes with a `Scan` (from {mod}`solprop.driver.scan`);
2. write a **builder**, a function that turns one `ScanPoint`, the values and seed of one
   point, into a {py:class}`~solprop.driver.simulation.Simulation`;
3. call {py:func}`~solprop.driver.scan.run_scan` with an executor, the reductions to
   compute in the worker, and what to retain;
4. read the file with {py:func}`solprop.processing.scans.open`.

## Declaring a scan

A scan evaluates a Cartesian product of inputs while keeping point identity separate
from execution order. The last declared variable varies fastest (NumPy C order).
Indices start at zero and remain global after `subset` or `chunk` selections, so a point
keeps its index, its seed and its place in the file however the scan is split.
`scan.add(name, values)` appends a variable after construction; `scan.subset(indices)`
keeps the named global indices and `scan.chunk(k, n)` the k-th of n round-robin subsets.

```{code-block} python
from solprop.driver.scan import Scan

scan = Scan("pressure_energy", pressure=[0.8, 1.0, 1.2], energy=[200e-6, 275e-6])
assert scan.shape == (3, 2)
assert scan.point(3).values == {"pressure": 1.0, "energy": 275e-6}
assert [point.index for point in scan.chunk(1, 2).points()] == [1, 3, 5]
```

## Builders and executors

The builder receives a `ScanPoint` and returns a `Simulation`. Variables use the
builder's units: pressure at the user interface is in bar, energy in J, lengths
in m and times in s. A builder can select any installed backend.

Five executors, all in `solprop.driver.executors`, run the points:

| executor | what it does |
|---|---|
| `Local()` | runs the points one after another in the current process; the default |
| `Pool(workers, threads=1, ...)` | runs independent points in parallel worker processes; the executor to use for independent CPU runs |
| `Batch(size)` | advances compatible fixed-schedule points in lockstep inside one backend |
| `Queue(path)` | a durable queue on a shared filesystem, drained by one or more controllers |
| `Slurm(script, ...)` | prepares a queue and submits a Slurm array job that drains it |

A builder's omitted `backend` resolves to the native backend when it is installed and otherwise,
with a warning, to the slower `numba_cpu`, as described in
[running a simulation](../performance/resources.md#backend-choice-and-scan-workers); every
executor keeps that concrete choice, so name the backend in the builder when the workers
may run on machines with different installations.

## Reuse a translated simulation

Translation — the cumulative linear operator, the modal transform, the ionisation rate
tables, the compiled responses — is the expensive part of building a run, and most scans
change only the fill or the input pulse. Build the base once, lazily, in each worker,
and derive every point from it. For example, this reduced pressure–energy scan follows
the helium RDW case's 125 µm capillary and 7.5 fs pulse
([running](../setup/running.md#a-capillary-end-to-end)). Its short length and coarse grid
demonstrate execution, not a converged RDW prediction.

```{code-block} python
from functools import cache
from solprop.driver import Simulation
from solprop.driver.scan import Scan, ScanPoint, run_scan
from solprop.driver.executors import Local, Pool
from solprop.physdata import Mixture
from solprop.processing.scans import energy_in_band, peak_power
from solprop.pulses import GaussianPulse


@cache
def base():
    mixture = Mixture.fill({"He": 1.0}, 1.0)
    simulation = Simulation.from_capillary(
        125e-6,
        0.002,
        mixture,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 1.5e-6),
        trange=200e-15,
        fwhm=7.5e-15,
        energy=200e-6,
        modes=1,
        raman=False,
        plasma=False,
        thg=False,
        n_save=3,
        backend="numba_cpu",
    )
    simulation.translate()
    return simulation


def build(point: ScanPoint):
    mixture = Mixture.fill({"He": 1.0}, float(point.values["pressure"]))
    pulse = GaussianPulse(
        wavelength=800e-9, fwhm=7.5e-15, energy=float(point.values["energy"])
    )
    return base().with_mixture(mixture).with_input(pulse)


if __name__ == "__main__":
    scan = Scan("pressure_energy", pressure=[0.8, 1.2], energy=[200e-6, 275e-6])
    result = run_scan(
        scan,
        build,
        executor=Pool(workers=2),
        output="pressure_energy.h5",
        reductions=[energy_in_band(250e-9, 400e-9), peak_power()],
        fields="planes",
        field_planes=[0, -1],
        stats="all",
    )
    ultraviolet_energy = result.as_grid("reduced/energy_in_band")
```

{py:meth}`~solprop.driver.simulation.Simulation.with_input` rebuilds the input record
and metadata while sharing propagation products, and validates only the replaced input;
its `configuration=` records keys of your own, such as a point's parameters, in the
derived run's record. Outside a scan, a
[runner](../setup/running.md#many-inputs-one-backend-the-runner) also keeps one backend for
many inputs; scan workers construct a backend per point. `noise="keep"` (the default) keeps the
base noise, `noise=None` disables it, and a `NoiseRealisation` replaces it.
{py:meth}`~solprop.driver.simulation.Simulation.with_mixture` changes the fill
([running](../setup/running.md#a-capillary-end-to-end)); combine the two for material and input
changes. Geometry, grid or mode-count changes need a new simulation and translation.
The ionisation rate tables keep the validity interval they were built with; an input
update does not enlarge that interval, so build the base at the highest energy the scan
reaches or pass `ionisation_field_range` explicitly. Raman and plasma choices in this
example are explicit. For molecular gases the ordinary Raman and ionisation defaults
apply when not specified; see [mixtures](../media/mixtures.md) and [ionisation](../media/ionisation.md).

**What reuse buys.** Reusing a translated base cuts the setup time of an eight-point
`Local` scan to between a half and a third of translating every point afresh; with a
two-worker `Pool`, where each worker translates its own base once, the total with startup
and metadata changes little. Keep the geometry, the grid and the ionisation tables'
validity range unchanged between points. Changing the input array removes the
constructor's pulse recipe from the metadata, because it no longer describes the field;
the exact initial spectrum is still recorded. Two points with equal physical fields
need not have equal configuration hashes when their construction provenance differs.

## Fixed-schedule batches

`Batch(size)` advances up to `size` points in lockstep, on one fixed step schedule, inside
one backend. It runs on `numpy_ref`, `numba_cpu` and `native` (CPU and GPU builds), and
each backend is tested to give a batch member the output the same point gives run alone,
to rounding: the same steps, with sums that may be ordered differently
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).

```{code-block} python
from solprop.driver.executors import Batch
from solprop.driver.scan import run_scan

# The builder must return fibre simulations with an explicit fixed schedule.
# It selects numpy_ref, numba_cpu or native as usual.
result = run_scan(
    scan,
    build_fixed_point,
    executor=Batch(4, memory_limit=8 * 1024**3),
    output="fixed_scan.h5",
    fields="planes",
    field_planes=[-1],
    stats="all",
)
```

Prepare the replay schedule with the `discover`, `calibrate` and `replay` workflow of
{mod}`solprop.stepper.schedule` ([fixed schedules](../numerics/fixed-schedules.md#fixed-schedules-discover-calibrate-replay))
and validate its transfer to the individual scan points before interpreting their
physics. A batch executor does not silently convert an adaptive simulation into a fixed
one, and it runs modal (fibre and capillary) propagation only, not radial, Cartesian or
planar grids. Adjacent points with matching grids, schedules, response layouts and
statistics can share a group; layout changes and the final partial batch make smaller
groups. Mixing backend names in a batch scan raises an error. Global point indices are
unchanged. General modal runs may be batched under a narrower rule
([running](../waveguides/supplied-guides.md#a-guide-instead-of-a-fill)).

Each member watches its own time window, as the point would run alone
([watching the window's edges](../numerics/boundaries.md#watching-the-windows-edges)):
a point whose light reaches an edge is stopped and recorded as failed with its
`BoundaryError`, and the other members of its batch go on.

Memory admission probes the selected backend and keeps 20 % host headroom. It
records baseline and probe RSS, workspace and output estimates, shared table
bytes already included in the baseline, and the admitted group size. An explicit
`memory_limit` (bytes) further restricts the physical or scheduler allocation. This is an
admission estimate, not an operating-system memory limit; the builder itself must
fit while constructing up to `size` translated points. A budget that cannot fit
one member is refused. Device backends report free memory through the same interface:
admission probes each member, uses its complete ledger as a conservative upper bound on
device storage, and restricts the group before constructing all members. Allocator
reserves are excluded from free space. The recorded device bound may include host
buffers, so it can admit fewer members than a device-only ledger would.

Each point records its actual batch size, execution strategy and resolved backend
policies. `numba_cpu` and `native` fuse the accepted update across members (`native`
combines the error partials of all members in the same native kernel), while each member
keeps its own workspace; the response kernels and the FFT and BLAS plans stay per member. This partial fusion does not amortise the whole GPU launch
sequence.

**What batching buys.** On the CPU, batching gives no speedup over running the points
one after another, so use `Pool` there. On a data-centre GPU, a native `Batch(8)` of
eight points of a shortened multimode resonant-dispersive-wave capillary on a fixed
515-step schedule gave about 1.1 times the throughput of `Local`, setup and output
included. That measures one shape on one schedule and does not predict others. For
comparison, on an 18-core x86-64 server a `Pool` of 8 to 18 one-thread workers gave
about four times the throughput of `Local` on a shortened six-mode hydrogen-filled
capillary, and one worker using all 36 hardware threads was slower than those pools.
Size workers within the actual CPU and memory allocation, and measure the workload that
matters to you; [performance tuning](../performance/tuning.md) explains the thread
settings of scan workers.

Fields keep the single-run saved-plane shape. Error histories, noise, species axes
and periodic statistics remain separate per member. Reductions use the same
processing definitions and temporary disk spool as `Local` and `Pool`. Completed
points are revalidated as on the other routes before being skipped (below), so changing
executor does not change what a point records.

## Processes and resource limits

`run_scan` runs on `Local()` when no executor is given and the command line has set none.
Use `Local()` for debugging or a small interactive scan, and `Pool()` for independent
CPU simulations. `Pool` uses spawned processes and one library thread per worker by
default. It sets thread environment variables before scientific libraries import,
respects Linux CPU affinity and Slurm/cgroup allocations, and keeps workers alive.
`fresh=True` creates a new worker for each point. `workers`, `threads`, `pin` and
`memory_limit` provide explicit resource limits; memory is specified in bytes.

A backend/RSS probe sizes workers. The estimate takes the larger of measured RSS
and baseline RSS plus the backend ledger, then reserves retained output and transfer
buffers. The controller's RSS and 20 % memory headroom are reserved separately. The
file records this calculation. It refuses an allocation that cannot fit one worker.
This is admission control, not an operating-system memory limit; adaptive periodic
statistics and custom reducer working memory can exceed the probe estimate.

Pool builders and custom reducers must be pickleable, importable module-level
callables. Keep entry-point execution under `if __name__ == "__main__"`. Nested
functions, lambdas and live backend handles cannot be dispatched. Validation happens
before the output file is created. A cached module-level base is local to each
worker, so no translated native objects cross process boundaries.

`python -m solprop.scan scan_script.py --local` sets the executor from the command line;
an executor supplied explicitly by the script takes precedence. One outer progress bar is
shown interactively (`progress=` of `run_scan`); otherwise progress is logged per point.
Individual runs never draw their own bars.

## Noise and repeatability

Use `Scan("name", base_seed=42, ...)` to derive a reproducible unsigned 64-bit seed
for each global index with
`SeedSequence([base_seed, global_index]).generate_state(1, dtype=uint64)[0]`.
An ordinary `seed=[...]` axis overrides automatic derivation. Subsetting or changing
execution order preserves the seed.

Every point's noise is reproducible, whether or not the builder seeds it. Noise the
builder leaves unseeded — a constructor's default noise, or an explicit model without
`seed=` — is drawn from `point.seed`. On a scan with neither a seed axis nor a base
seed, `point.seed` is `None` and the draw takes a seed derived from the scan's name
instead: the same derivation with the first eight bytes of the name's SHA-256 digest,
read little-endian, in place of `base_seed`. A builder may still seed the noise itself
with `seed=point.seed`. The file records the derivation, resolved point seed, actual
noise seed, single-run configuration hash and backend execution settings. A noise
draw without a recorded seed, or seeded with something other than `point.seed`, is
rejected. Rerunning a scan rebuilds each point with the same seed, so its completed
points agree with their records and are skipped.

## Retention, reductions and reading

By default scans retain all collected statistics, including periodic rows and species
axes, and no fields. `stats=["energy", "peak_power"]` filters statistics by name;
`stats="none"` removes them, which is discouraged: the statistics are the diagnostics
needed to assess a scan. `fields="planes"` requires explicit `field_planes`
saved-plane indices, including negative indices. `fields="all"` retains every saved
plane. A field reduction can retain one field independently of field-history storage.

The worker keeps only the fields the scan reads
([SEP 22](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md)): the planes `fields`
retains, the target plane of each built-in reduction that reads a field, and, while a
custom reduction reads fields, every plane of the run it is handed. The other planes are
run as statistics-only planes
([output](../results/output.md#statistics-only-planes-which-planes-keep-their-field)),
whose fields are never stored, so a scan of large spatial runs does not hold a field per
plane for nothing. `run_scan`'s arguments decide: a builder whose
`SimulationSettings` set `fields` or `field_planes` is refused by name.

**Reductions** are computed in the worker, from the point's complete output, so only
their results travel back and are stored. The predefined ones in
{py:mod}`solprop.processing.scans` call the ordinary [processing](../results/processing.md)
definitions: `spectrum_in_band` and `energy_in_band` (a wavelength band),
`energy_in_window` (a time window), `peak_power`, `mode_energies`, `field_at_plane`,
and, for spatial runs, `on_axis_intensity` and `beam_radius`. Spatial reductions require
the corresponding collected statistics. Band energy applies a closed rectangular
frequency-bin mask; it is not the tapered [`bandpass`](../results/processing.md#bandpass-windows)
window. Time-window energy sums the selected samples with the grid time step.
`spectrum_in_band(lambda_min, lambda_max)` keeps the spectral energy density, J/m and
summed over modes, on the run's active band, the bins that
{py:func}`~solprop.processing.fields.get_spectral_density` returns by default (on a
field grid this leaves out the zero and Nyquist bins, where no wavelength is defined),
at the wavelengths inside the closed band, ascending. The reader's
{py:meth}`~solprop.processing.scans.ScanOutput.spectrum_wavelength` returns that
abscissa, rebuilt from the recorded grid and band, so it matches the last axis of
`as_grid("reduced/spectrum_in_band")` sample for sample; `spectrum_wavelength(index)`
gives one point's abscissa when the points' grids differ.
The spectrum, band and window energies, peak power and mode energies read modal
amplitudes, so a radial, Cartesian or planar run refuses them by name; use its
spatial statistics or a custom reducer there. A saved field keeps the run's stored
units: V m radially, V/m on a Cartesian grid, √(W/m) per y frequency on a planar
one. Reducers carry units, dtype, parameters and source provenance.

`custom(name, fn, shape, units, dtype=...)` declares a reducer of your own: `fn` receives
the point's complete output handler and returns an array of the declared shape and dtype.
The worker spools that point to temporary local HDF5 storage and removes it afterwards.
By default the spooled run keeps every field. A reducer that reads only statistics,
positions and step histories declares `reads_fields=False`, and its spooled run then keeps
the fields the scan keeps anyway and no other; a field read inside it is refused by name.
The declaration is recorded in the reducer's provenance only when it is `False`.
A wrong shape, dtype or non-finite reduction marks the point failed and records the
traceback. When simulation metadata is available, failed points keep their configuration
hash, actual noise seed and backend policies. Readers expose that provenance while
keeping incomplete numerical arrays masked.

One controller owns the collected HDF5 file. Each successful payload is validated,
staged and flushed before the point is marked `done`. A rerun skips a completed point
whose rebuilt record is its stored one, or differs from it only in rounding, a declared
neutral value or provenance, as a resume compares them
([what a resume compares](../results/resuming.md#what-a-resume-compares)); a worker
whose digest differs hands its record back, and the controller compares the two.
Unlike a resume, a scan holds the numerics to agreement too: a finished point has
nothing left to continue, and tightening `rtol` is the commonest reason to rerun a
scan, so a rerun with changed numerics or physics is refused, naming what changed,
and is written to a new file. The record includes the point's material manifest: a
point resolves its materials, under its `materials=` selection or the default one, from
its own objects, so its optmat models and policies are frozen for that run and do not
depend on what another point resolved. A model whose numbers changed, or a changed
policy, is refused by name; another optmat version alone is provenance
([output](../results/provenance.md#the-material-manifest)). Failed points require
`retry_failed=True`. A stale `running` point becomes eligible after `stale_age` seconds
(600 by default). Queue ownership protection uses the live-owner and refreshed-lease rules
described below.

Uniform scans expose rectangular `stats/`, `reduced/` and optional `fields/` datasets.
Different shapes have separate `layouts/<digest>/` groups with global-point mappings.
Common metadata is content-addressed, with the constant and varying paths recorded.
Periodic statistics use concatenated values and offsets without padding. Statistics that
are legitimately undefined and stored as NaN (centroids, widths, beam radii,
zero-dispersion wavelengths and unavailable quadrature alarms) are preserved; non-finite
fields, energies and reductions are rejected. Step indices keep the single-run
convention, relative to each plane's interval, with per-plane counts.

```{code-block} python
from solprop.processing.scans import open

result = open("pressure_energy.h5")
indices = result.where(pressure=1.2)
point = result.point(int(indices[0]))
layouts = result.layouts
energy = result.as_grid("reduced/energy_in_band")
```

{py:class}`~solprop.processing.scans.ScanOutput` has `shape`, `status`, `where`, `point`,
`as_grid`, `layouts` and `spectrum_wavelength`. Readers mask unfinished points and recover committed staging
even if consolidation was interrupted. `as_grid` refuses arrays with incompatible shapes.
A failed point's `error` contains its traceback; inspect `status` before interpreting a
scan.

## Durable queues and Slurm

For several controllers sharing a filesystem, choose `Queue("pressure_energy.queue")`.
A queue keeps exact identities: its manifest hashes the builder's code and refuses a
change, and each point and collected artifact is checked by its configuration hash, so
the tolerance of a local rerun does not apply to it; `Slurm` drains a queue and
behaves alike.
Each controller runs a memory-sized spawned `Pool`. Point artifacts are fsynced before
publication; the collected file is written under one exclusive controller lock.
Readers take the corresponding shared lock. The filesystem must support coherent
advisory locks, atomic renames and directory fsync across the participating hosts.
The queue is tested on local disks (APFS and ext4); network and parallel filesystems
(NFS, Lustre, GPFS) differ in their locking guarantees, so check yours with a small
scan before relying on it.

```bash
uv run python -m solprop.scan pressure_energy.py --queue pressure_energy.queue --workers 4
uv run python -m solprop.scan pressure_energy.py --queue pressure_energy.queue --chunk 1/3
uv run python -m solprop.scan collect pressure_energy.queue --output pressure_energy.h5
```

Chunks are zero-based round-robin selections of global indices. Several invocations
can share the queue and output. The immutable manifest binds the axes, retention
settings, serialised builder and available builder source; changed scripts require a
new queue. Collection checks scan identity, point identity, configuration hashes and
artifact digests. Repeated collection of unchanged results leaves the file unchanged.
Artifacts are JSON and numerical HDF5 datasets, never executable pickle payloads.
Failed artifacts keep their error and available provenance; collection rejects
numerical arrays attached to a failed artifact.

A controller refreshes each active claim's lease. A live local PID with matching
creation identity remains protected even after `stale_age`; a remote owner remains
protected while its lease is refreshed. A dead local owner or expired remote lease
can be replaced. Unique claim tokens reject late results from superseded workers.
Failed points require `retry_failed=True` in the script. `.nowork` means the invocation
found no eligible work; inspect active owners and failures. `.done` is written and
fsynced before active queue state is removed. Keep the persistent lock files: deleting
a live lock file would create a second lock inode and defeat mutual exclusion.

```{note}
Recovery has known limits. A live but wedged owner on the same host holds its point
indefinitely, because the lease age is not consulted while its process is alive: stop
that process to release the point. An interrupted `Local` or `Pool` scan skips its
`running` points for `stale_age` seconds on restart. Container memory limits are read
from the root cgroup rather than the process's own cgroup, so on a shared node under a
cgroup manager without Slurm variables the workers can be given too much memory; a failing
platform probe raises outside the {py:class}`~solprop.exceptions.SolpropError`
hierarchy. The scan file grows on every resume, because HDF5 does not reclaim the
groups it recreates.
```

Inspect a generated Slurm script before submission:

```bash
uv run python -m solprop.scan pressure_energy.py --emit-slurm scan.slurm --queue pressure_energy.queue --ntasks 4 --cpus-per-task 8 --mem 16G --time 01:00:00
uv run python -m solprop.scan pressure_energy.py --submit --queue pressure_energy.queue --ntasks 4 --cpus-per-task 8 --mem 16G --time 01:00:00
```

`--emit-slurm` writes the script without running the scan builder or calling `sbatch`.
`--submit` prepares the durable manifest, submits the array and returns the currently
collected output, which may still contain unfinished points. Collect again after jobs
finish. The Python `Slurm` executor also exposes its returned `job_id`. Scripts and
queue/output paths must be accessible at the same paths on allocated nodes, with the
same environment available. `--mode chunk` emits disjoint round-robin task subsets;
queue mode (the default) lets tasks share claims dynamically.

Positive `--instances` divides each task's CPUs and memory among controllers. For GPU
resources such as `--gres gpu:2 --instances 2`, each instance receives one of
Slurm's assigned device identifiers and defaults to one worker. Missing assignments
fail before the scan starts. All worker thread settings are exported before Python
imports scientific libraries (see [performance tuning](../performance/tuning.md) for
what those settings are). `--fresh` creates a fresh process per point.

Submission records under `QUEUE/submissions/` keep the scheduler job ID and
generated script digest, and collection includes them in the execution audit.

An explicit executor in `run_scan` overrides a command-line default. To use these
command-line modes, omit the script's executor. An explicit `Slurm` executor recognises
its own emitted task environment and drains the prepared queue instead of submitting
recursively. Without a durable mode, `--workers`, `--threads`, `--memory-limit` or
`--fresh` select `Pool`; `--local` selects `Local` and rejects process resource controls.
Slurm memory uses `--mem`, while a direct `Pool` or queue accepts `--memory-limit` in
bytes.
