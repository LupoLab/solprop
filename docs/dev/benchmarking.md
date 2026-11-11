# Running the benchmarks

`benchmarks/` times **complete runs** of a small set of representative workloads. Each
release adds a few lines per case to a history file kept per hardware set, so that
regressions and improvements show from one release to the next. The design is
[SEP 38](seps/archive/sep-0038-performance-tracking.md), which replaced SEP 30's standard
suite. The rules for a performance claim and the indicative results are in
[Benchmarks](benchmarks.md); this page is the manual.

```sh
uv run poe bench                      # the plan and its estimated time
uv run python -m benchmarks track     # the tracking run, about an hour
uv run python -m benchmarks history   # what earlier runs recorded
uv run python -m benchmarks ab --case rdw --baseline-python OLD/bin/python
```

Every command uses the native CPU backend unless `--backend` says otherwise, and needs
it installed, as a plain `uv sync` does. Run on a quiet machine: a dedicated
workstation, or an exclusive allocation on a cluster node.

## The tracking set

`python -m benchmarks list` prints the plan: one line per case, stepper and width, and
the estimated time from the costs each case declares (complete runs on 16 cores of an
AMD EPYC 7B13, about 33 minutes for the tracking set; other machines differ). With no
selection it is the tracking set:

| Case | Interface | Problem | Widths |
|---|---|---|---|
| `gnlse_raman_shock` | `prop_gnlse` | Dudley–Genty–Coen supercontinuum, 0.15 m | 1, production |
| `gnlse_soliton` | `prop_gnlse` | N = 3 soliton, one soliton period | 1, production |
| `rdw` | `prop_capillary` | He resonant dispersive wave, four modes, 1.5 m | 1, production |
| `he_rdw_averaged` | `prop_capillary` | the same, mode-averaged | 1, production |
| `argon_blueshift` | `prop_capillary` | Ar ionisation blueshift, 0.2 m | 1, production |
| `argon_blueshift_averaged` | `prop_capillary` | the same, mode-averaged | 1, production |
| `argon_blueshift_envelope` | `prop_capillary` | the same on the envelope grid | 1, production |
| `h2_raman_averaged` | `prop_capillary` | H2 soliton self-frequency shift, 0.4 m | 1, production |
| `ctc` | `prop_capillary` | H2 comb-to-continuum, three modes, 0.13 m | production |
| `radial_filament_256` | `prop_radial` | Ar filament, 256 radial nodes, 3 cm | 1, production |
| `grin_nonlinear` | `prop_waveguide` | four-channel graded-index fibre, 2 cm | production |
| `kerr_beam_384` | `prop_free` | fused-silica Kerr beam, 384², four fixed RK4IP steps | production |
| `kerr_beam_192` | `prop_free` | the same at 192² | 1 |

The definitions are in `benchmarks/registry.py` and `benchmarks/builders.py`. Every
other registered case (the remaining modal, free-space and advanced-guide cases, and
the backend exercises, and the `sep47` set of short windows for
[SEP 47's matrix](#sep-47s-matrix-and-the-structural-counters)) runs when named with `--case`, `--set` or `--all`;
none is in the default selection. A case whose definition changes keeps its name and raises its
revision, and the history compares rows only within one revision.

## What is timed

Each configuration (case, stepper, width) runs in its own worker process, whose thread
pools are set in the environment before any scientific import. The worker:

1. builds the case, translates it and constructs the backend, timing these separately;
2. runs an untimed warm-up over the first 2 % of the guide, which builds FFT plans and
   fills caches;
3. times `--repeats` complete runs (default 3). Each restarts the backend from the
   translated launch and goes through `solprop.driver.run.run_problem`, the loop of
   `Simulation.run`, with a fresh boundary monitor as `Simulation.run` attaches one,
   so it includes the plane observations, the statistics and the boundary checks: what
   a user's run costs.

The result is the median of the repeats, with the minimum and maximum beside it, the
accepted and rejected steps, the source evaluations, the peak resident memory, and the
integrator and tolerances the run actually used. Repeats of one configuration take the
same steps; the summary says so when they do not.

**Library defaults.** A case is the interface call a user would write. It passes no
tolerance or integrator, so it runs with its interface's defaults
([SEP 37](seps/archive/sep-0037-noise-and-stepper-defaults.md)). A changed default therefore
shows in the history as a new integrator or tolerance on that row. Every noisy launch
fixes the same seed, so its repeats and its releases are comparable. The N = 3 soliton
is the one noiseless case: a soliton-recurrence study runs without noise, and SEP 37
measured a seed multiplying its steps by 65. The Kerr beams
run their declared fixed RK4IP schedule.

**Steppers.** `--stepper` takes `default`, `multistep`, `multistep-interaction` or `rk`,
or a comma-separated list to compare steppers in one run:

```sh
uv run python -m benchmarks track --case rdw --case ctc --stepper default,rk
```

An override replaces the integrator only; the tolerances stay the interface's. A
fixed-step case admits `default` only, and the plan says which it skipped.

**Widths.** Without `--threads`, each case runs at its declared widths: one thread and
the production width, or one of them for the most expensive cases. `--threads 1,8`
names the widths for every selected case. The production width is the number of
physical cores in the process's CPU allocation unless `--production` gives it; on a
cluster, give it explicitly.

**Threading.** `--threading` sets OpenMP's placement and waiting in every worker:
`backend` (the default since November 2026: unpinned, with the backend's own waiting,
none of the waiting variables set, so that a row measures what a user's default run gets),
`passive` (unpinned, idle threads sleep at once), `passive-bound` (the same, pinned
to neighbouring cores), `default` (the runtime's own short
spin before sleeping), `active` (idle threads spin), `bound` (threads pinned to
neighbouring cores) or `bound-active`. The policy is recorded with each row, and rows
compare only within one policy. Binding also confines SciPy's transforms on the host to the
main thread's core, so a free-space row's edge reads cost more under a bound policy than
under `passive` ([the edge against 0.1.0](benchmarks.md#the-transverse-edge-against-010)).

## On a GPU

With a native GPU build installed (CUDA or HIP; see the native backend's build guide),
`--device` runs every selected case once on the device:

```sh
uv run python -m benchmarks track --device
```

The host still runs the production width of threads, but each row's width is `device`.
The hardware name is the GPU model, for example `nvidia-a40` or `nvidia-h200`, plus the
instance family on a cloud machine, and the host CPU model is recorded beside it. An
AMD GPU whose `rocm-smi` series is the generic "AMD Radeon Graphics" is named from its
PCI device or subsystem ID instead. Each
row also records the device memory the run holds. The worker refuses a GPU build
without `--device` and a CPU build with it. The case costs are CPU costs, so a device
plan gives no time estimate.

## Recording a run

```sh
uv run python -m benchmarks track --append-history
```

A series measures a user's default run, so it runs calibrated
([SEP 46](seps/archive/sep-0046-machine-calibration.md) section 6.13): before the first
configuration, `track` runs `solprop calibrate --thorough` (`--calibrate` chooses another
level, or `none` to keep the profiles there are), and its workers apply the profile
(`SOLPROP_CALIBRATION=use`) without measuring inside a timed window. Each history row
records the profile its run applied in the `calibration` column (`built-in`, or the level
and generation, with `+refinement`), so a series that moves can be attributed to the
code or to a recalibration. `SOLPROP_CALIBRATION=off` holds the built-in table, for
experiments that must keep the settings fixed while the code changes; the first series
recorded after SEP 46 starts a new segment of the history.

The run's bundle goes to `.benchmarks/track/<date>-<hardware>/` (git ignores it): the
plan, each worker's request, environment and logs, and `results.json`, which is
rewritten after every configuration so that an interrupted run keeps what it
measured. `--archive DIR` copies the
bundle to a directory outside the repository.

`--append-history` adds one row per configuration to
`benchmarks/history/<hardware>.csv`, which is committed. Each row names its backend,
and rows compare only with rows of the same backend, so the `numba_cpu` rows recorded at
0.1.0, before [SEP 43](seps/archive/sep-0043-retire-numba-cpu.md) retired that backend, keep their own series beside the native one. It records only the whole default plan
with at least three repeats, and only when:

- every configuration finished, and its repeats took the same steps and reached the
  same final field to rounding, with the native backend running the threads it was
  given;
- the workers imported solprop from this checkout's `src/` (run from the checkout's
  own environment, `uv sync`, not an installed copy), with, for a
  native run, a native build from a commit after which nothing under `native/`
  changed but its documentation and tests;
- nothing under `src/`, `native/` or `benchmarks/` is uncommitted (the history files
  aside, so one checkout can record several runs), unless `--allow-dirty`, whose rows
  are marked dirty;
- `SOLPROP_FFTW_PLAN` is unset: a row does not record how it planned, and measured
  planning changes its construction time and its median.

The hardware name is the CPU model and the production width, plus the instance family
on a cloud machine, for example `apple-m5-max-18c`, `amd-epyc-7763-16c` or
`arm-neoverse-v2-16c-aws-m8g`; it never contains a host name. Give a cloud machine's
instance type with `--instance-type` where its DMI record does not name it (AWS's
does).

`python -m benchmarks history` prints, per hardware and width, one line per recorded
run: its date, solprop version, the configurations it ran and how many started on a
busy machine, and its change against the first and the previous recorded runs. A
change is the geometric mean of median-time ratios (below one is faster) over the
configurations both runs share, counted in brackets. Two rows are the same
configuration only with the same number of memory (NUMA) domains spanned, threading
policy, case revision, stepper, resolved integrator, tolerances and noise model. A
changed library default therefore starts a new series rather than reading as a
speed-up, and so does an allocation spread over more memory domains than usual: on a
cluster, keep the cores together (`--distribution=block:block` under Slurm).

## Recorded hardware

The history starts at 0.1.0. Each release reruns the same hardware sets the same way,
so that their rows stay one series:

| Hardware file | Machine | Width | `--instance-type` |
|---|---|---|---|
| `apple-m5-max-18c` | Apple M5 Max, macOS | 18 | — |
| `intel-xeon-gold-6240-36c` | 2× Intel Xeon Gold 6240, whole machine (exclusive Slurm job) | 36 | — |
| `amd-epyc-7763-16c` | 2× AMD EPYC 7763 cluster node, exclusive, 16 cores in one NUMA domain | 16 | — |
| `amd-epyc-9b45-16c-gcp-c4d` | Google Cloud `c4d-highcpu-32`, SMT off, Ubuntu 24.04 | 16 | `gcp c4d` |
| `amd-epyc-9b45-16c-gcp-c4d-windows` | the same shape, Windows Server | 16 | `gcp c4d windows` |
| `arm-neoverse-v2-16c-gcp-c4a` | Google Cloud `c4a-standard-16`, Ubuntu 24.04 | 16 | `gcp c4a` |
| `nvidia-a40` | NVIDIA A40 cluster node, CUDA build (`AMPERE86`) | device | — |
| `nvidia-h200-do` | cloud `gpu-h200x1-141gb` instance, CUDA build (`HOPPER90`) | device | `do` |
| `amd-mi325x-do` | cloud `gpu-mi325x1-256gb` instance, HIP build (`AMD_GFX942`) | device | `do` |

- **CPU rows time the release's wheel.** A clean checkout of the release tag, its
  environment made with `uv sync --python 3.14 --frozen --no-install-package
  solprop-native`, then `uv pip install solprop-native==<version>` from PyPI, and every
  command run as `.venv/bin/python -m benchmarks …`: a later `uv sync` or `uv run` builds
  native in the wheel's place. The wheel records the tag's commit, so the history accepts it.
- **numba rows** exist only at 0.1.0: a second run in the same checkout with
  `--backend numba_cpu`, on the Apple M5 Max, the Xeon Gold 6240 and the EPYC 7763. A later
  release cannot add them.
- **GPU rows build the tag's native backend** with the `cuda` or `hip` preset's
  options into a separate environment, out of tree (`uv build native --wheel
  --config-setting build-dir=…`), so that the checkout stays clean. ROCm 10 builds
  the HIP backend only with `-DSOLPROP_NATIVE_WARNINGS_AS_ERRORS=OFF` at 0.1.0 (fixed
  after it).
- **The A40's memory domains.** The 0.1.0 job held a whole node, so its host threads
  could run in all 8 NUMA domains and its rows record 8. At 0.2.0 every A40 node was
  partly busy: the job held 16 cores of a shared node, they spanned 3 domains, and
  those rows start a new series (their device times were 9 % faster than 0.1.0's in
  geometric mean). An exclusive job returns to the 0.1.0 series.
- **The instance type separates the operating systems.** The hardware name carries
  no operating system, so Linux and Windows on one cloud shape differ only by
  `--instance-type`. Google Cloud allowed a 32-vCPU C4D only in `us-central1` at
  0.1.0, so the two ran one after the other.
- **Windows** ends an SSH session's child processes with it: start the run through
  WMI (`Invoke-CimMethod Win32_Process Create`), and run one unrecorded case first,
  since the first worker after boot runs cold.
- **The Mac** runs with Spotlight indexing paused (`sudo mdutil -a -i off`) and the
  controlling session idle. At 0.1.0 one run of `gnlse_raman_shock` at 18 threads
  took 31 s against 21 s in the recorded run and in three repeats, with the same
  steps: an occasional slow mode to watch for.
- **The AMD GPU name.** ROCm 10 reports the cloud MI325X's Card Series as
  "AMD Radeon Graphics"; the harness then names the part from its PCI IDs. The 0.1.0
  rows, recorded before that fallback, had their `slug` and `gpu_model` set to it by
  hand; nothing else in them changed.

## Comparing two builds while iterating

`ab` times the same configurations with two prepared environments, for a change in
progress:

```sh
uv run python -m benchmarks ab --case rdw --case argon_blueshift --threads 1 \
    --length 0.2 --baseline-python /path/to/baseline/bin/python \
    --candidate-python /path/to/candidate/bin/python
```

Each build is calibrated before the rounds, once per interpreter and with its arm's
variables (`--calibrate`). A development A/B defaults to the `standard` level, which takes
a few seconds; `thorough`, `track`'s default, is for release benchmarks and for comparing
what a heavy user's calibration would give.
`--baseline-env` and `--candidate-env NAME=VALUE` set an arm's own variables over the
threading preset, and an empty value unsets one. A worker otherwise runs at the row's
width, which keeps a profile's team width out, and with it any setting fitted at
another width. To compare the built-in table with this machine's calibration as a
default run sees them, give both arms no thread count of their own:

```sh
uv run python -m benchmarks ab --set tracking --threads 16 --production 16 \
    --calibrate none --length 0.2 --rounds 3 --repeats 1 \
    --baseline-python "$(which python)" \
    --baseline-env SOLPROP_THREADS= --baseline-env SOLPROP_NATIVE_THREADS= \
    --baseline-env SOLPROP_CALIBRATION=off \
    --candidate-env SOLPROP_THREADS= --candidate-env SOLPROP_NATIVE_THREADS= \
    --candidate-env SOLPROP_CALIBRATION=use
```

The same command with `SOLPROP_CALIBRATION=off` in both arms is its A/A control.

Selection works as for `track`. `--length` runs that fraction of each guide, from the
entrance, for quick iteration. Each round runs every configuration once with each
build, one worker each, and alternates which build goes first; `--rounds` sets the
count (default 3) and `--repeats` the timed runs per worker (default 3). The harness
is copied into the output directory first, so both builds run the same benchmark code.

The summary gives, per configuration, each build's median time, the median over rounds
of the candidate/baseline ratio of the per-round medians (below one is faster), the
range of those ratios, whether both builds took the same steps, and the relative L2
difference of their final fields. Passing the same interpreter twice is an A/A
control: its ratio range shows the noise to expect on that machine. A one-second
reading before each worker labels a busy machine but never waits. `ab` gives no
adoption verdict; the [rules](benchmarks.md#rules) say how a claim is judged.

## SEP 47's matrix and the structural counters

[SEP 47](seps/sep-0047-run-ownership.md) section 7.2 compares builds on a predeclared
matrix of short windows, each holding the events it measures, with structural counters
beside the timings (section 7.3). `--set sep47` selects the windows added for it; each
finishes in seconds at its full declared length, so run it without `--length`:

```sh
uv run python -m benchmarks ab --set sep47 --threads 1 --counters \
    --baseline-python /path/to/baseline/bin/python
```

| Case | What one repetition is | Events it holds |
|---|---|---|
| `stats_period1_adams`, `stats_period1_rk` | 16 mm of `gnlse_soliton`'s fibre under `Multistep()` or `RungeKutta()` (no stepper override), energy and peak power as periodic statistics with period 1 | a row at every accepted step (about 190 and 40) |
| `snapshots_idle`, `snapshots_sparse`, `snapshots_every` | the same fibre to HDF5 with `step_planes=1` and a `SnapshotController` never asked, asked after every eighth accepted step, or asked at every accepted endpoint | a plane at every step; 0, 5 and 42 snapshots published |
| `with_input_repeat` | three runs of the fibre with other peak powers through `Simulation.with_input` | three preparations from one translation |
| `scan_local` | `run_scan` of four peak powers on `Local`, built with `with_input`, field-free | four points, no field copied to the host |
| `group_batch` | the same four points on `Batch(4)`, replaying ten 1.6 mm steps with the split-step collar | ten common steps of four members |
| `radial_filament_256_{saved,free}_{memory,hdf5}` | the radial filament's first 3 mm, four planes, every field kept or none, to memory or to HDF5 | four planes, transverse-edge reads and collar checks |
| `cartesian_filament_128_{saved,free}_{memory,hdf5}` | the Cartesian filament's first 1 mm, likewise | the same on a 128² grid |
| `taper_flow_4`, `taper_flow_17` | 40 µm of a He capillary tapering from 50 to 48 µm with taper coupling, 4 or 17 modes | a coupled flow on the native Adams march, and one wider than 16 channels on solprop's host march (`HostMarchWarning`) |

Today a snapshot controller needs `step_planes=1` and HDF5 output, so the snapshot cases
save a plane at every step; their variant without step planes becomes possible with SEP
47's independent snapshot scheduling, and the cases will be extended then. Every case uses
only public interfaces of the current code; one whose interface changes is updated with
it, and the harness copied into each A/B run keeps both arms on one definition.

The rest of the matrix is existing cases, run by name with a short `--length` window
where they are long: the cheap envelope and field steps (`gnlse_soliton`,
`he_rdw_averaged`), the temporal and transverse collar edges (`absorber_schemes`,
`spatial_absorber`), low-storage replay (`kerr_beam_192`), general z tables and varying
guides (`mixture_gradient`, `taper`, `taper_wall`, `moving_solid_guide`) and the large
fields at production sizes (`radial_filament_512`, `cartesian_filament_256`).

**Counters.** `--counters` (for `track` and `ab`) has each worker make one more
repetition after the timed ones with the native adapter's counters on, reported as the
receipt's `counted` and, in `ab`'s summary, per arm; counting therefore costs no timed
repetition anything. With counting off, the adapter's path is the one before the
counters except for a single attribute test per native call (a march call, a plane, a
statistics call, an edge read, a field crossing or load, a low-storage or group step, a
native session's `advance` or export): the per-attempt hooks are installed only while
counting, and nothing else is computed, so no ledger entry was taken for them.

A native run is served either by the native execution session
(`solprop_native.session.NativeSession`, whenever it admits the run) or by the
transitional bridge (`DriverSession` over the native backend's marches, for what the
session refuses). Both are counted into the same record, so that an arm on the bridge
and an arm on the session can be compared key for key; where the two differ in kind,
the session's key starts `session_`. The record (`solprop_native._counters`, private to
the harness) holds:

- `native_returns`: every return to Python, by cause. On the bridge, a native march
  call's reason (`reached`, `monitor` for a boundary-monitor check by distance, `bound`
  for its attempt or time limit, `watch`, `capacity` and the errors), with `host_step`
  split by the driver's work due on its last step: `statistics`, `step_planes`,
  `monitor` (the check by steps), `transverse` (a collar's per-step current) or
  `transverse_read`, joined with `+` when several fall due together; then
  `host_attempt` for each attempt solprop's own loop steps, `low_storage_step` for the
  low-storage march and `group_step` for a transitional native group's lockstep step.
  A native session returns once per `advance`, by its batch's event:
  `session_observation`, `session_snapshot` (a requested snapshot serviced),
  `session_warning`, `session_terminal`, or `session_progress` with its cause,
  `session_progress:buffer` (the attempt or row buffer filled),
  `session_progress:work_limit` (the caller's attempt bound) or
  `session_progress:deadline` (its time bound). Members of the native group session are
  native sessions and are counted so. A periodic row, a collar check, an edge read or a
  low-storage step is never a session return (SEP 47 section 7.3);
- `march_calls`: native march calls by march, `adams_march` and `rk_march` from Python
  on the bridge, and the native session's own calls inside its `advance`, read from its
  status, as `session_adams`, `session_runge_kutta` or `session_low_storage`;
- `attempts` and `accepted` (native march, native session and host attempts together;
  a session's from its delivered chunks), `march_attempts`, `host_attempts`,
  `low_storage_steps` and `group_steps` (a group's common steps);
- `planes` (calls of `propagate_to` on the bridge, observations after the launch in a
  session), `statistics_rows` (periodic rows, wherever taken) and
  `statistics_evaluations` (rows and plane statistics together), and
  `transverse_reads` (reads of the edge's faces or a collar's current made through
  Python: a native session reads its edge natively, so none);
- `field_crossings` with `field_bytes`: fields copied to the host, by where. On the
  bridge at a plane (`plane`), at every accepted step of a host Adams engine (`step`)
  or at its load (`load`); in a session, each observation's field exported into its
  destination in one native pass (`export`; a recipe asked again for the same batch is
  served from the first and not counted) and an Adams march's state and history copied
  for a snapshot (`snapshot`);
- `field_loads` with `load_bytes`: host fields loaded into the native state, by where:
  `start` (a run's start or resume, on either path) and `restore` (the Adams march's
  reload after a dense observation or at a method history's restore). solprop's host
  Adams engine reloads its masked state at every step, which is not counted; its `step`
  crossings mark that route.

The run's own record adds the planes saved and, for a snapshot case, the snapshots
published; the peak resident memory is the worker's, as for every row. A host attempt
makes several native calls, which are not counted one by one; a group run's counts
include each point's resource probe, a native session that returns at its launch; and a
device's launches,
transfers and allocations are not counted: they need a profiler. Counts explain a
timing and do not replace it; check that both arms hold the same events before
comparing their times (SEP 47 section 7.2).

A scan or group repetition builds its own backends, so its worker builds none beforehand
and warms up with one whole untimed repetition. `scan_local` watches each point's run
to count its steps; `Batch` watches none, so `group_batch`'s steps are counted only by
`--counters`, and `ab` reports its work as `unchecked` rather than `same`. Where a
repetition keeps no final field (field-free planes, a scan), finiteness is read from
its last plane's statistics and the repeat difference is not defined; `track` treats
either gap as unchecked, never as consistent.
