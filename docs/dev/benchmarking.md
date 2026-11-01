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
it installed (`uv sync --extra native`). Run on a quiet machine: a dedicated
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
the backend exercises) runs when named with `--case`, `--set` or `--all`; none is in
the default selection. A case whose definition changes keeps its name and raises its
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
`passive` (unpinned, idle threads sleep at once), `passive-bound` (the same, pinned
to neighbouring cores), `default` (the runtime's own short
spin before sleeping), `active` (idle threads spin), `bound` (threads pinned to
neighbouring cores) or `bound-active`. The policy is recorded with each row, and rows
compare only within one policy.

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

The run's bundle goes to `.benchmarks/track/<date>-<hardware>/` (git ignores it): the
plan, each worker's request, environment and logs, and `results.json`, which is
rewritten after every configuration so that an interrupted run keeps what it
measured. `--archive DIR` copies the
bundle to a directory outside the repository.

`--append-history` adds one row per configuration to
`benchmarks/history/<hardware>.csv`, which is committed. Each row names its backend,
and rows compare only with rows of the same backend, so a `--backend numba_cpu` run
keeps its own series beside the native one. It records only the whole default plan
with at least three repeats, and only when:

- every configuration finished, and its repeats took the same steps and reached the
  same final field to rounding, with the native backend running the threads it was
  given;
- the workers imported solprop from this checkout's `src/` (run from the checkout's
  own environment, `uv sync --extra native`, not an installed copy), with, for a
  native run, a native build from a commit after which nothing under `native/`
  changed but its documentation and tests;
- nothing under `src/`, `native/` or `benchmarks/` is uncommitted (the history files
  aside, so one checkout can record several runs), unless `--allow-dirty`, whose rows
  are marked dirty.

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
  environment made with `uv sync --python 3.14 --frozen` (no `native` extra), then
  `uv pip install solprop-native==<version>` from PyPI, and every command run as
  `.venv/bin/python -m benchmarks …`: a later `uv sync` or `uv run` removes the
  wheel. The wheel records the tag's commit, so the history accepts it.
- **GPU rows build the tag's native backend** with the `cuda` or `hip` preset's
  options into a separate environment, out of tree (`uv build native --wheel
  --config-setting build-dir=…`), so that the checkout stays clean. ROCm 10 builds
  the HIP backend only with `-DSOLPROP_NATIVE_WARNINGS_AS_ERRORS=OFF` at 0.1.0.
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
