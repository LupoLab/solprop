# Calibrating solprop to your machine

The native backend's CPU execution settings are properties of the machine: how each
operation threads (serially, on a few threads or on the whole team), how wide the team
is and which cores it takes, how idle threads wait between parallel regions, and how
much effort FFTW spends choosing its plans. A setting that runs one machine's cases 13 %
faster on average can run another's 1.4 to 1.7 times slower. **A calibration measures
them on your machine** and keeps the result, so that every later run uses it. It never
changes a result beyond rounding: it chooses among exact algorithms and thread counts,
as the built-in defaults do.

```sh
solprop calibrate                # standard: about a minute and a half
solprop calibrate --quick        # about 20 s
solprop calibrate --thorough     # about five minutes
solprop calibrate --show         # the profile this job would use, and its evidence
```

Run `solprop calibrate` once after installing solprop, on the machine (or the kind of
node) that runs your simulations. If you skip it, the first native run calibrates
quickly by itself, with a notice (below).

## What it measures

The calibration runs in child processes, each pinned to the CPUs of your job, and times
short windows of five problems that solprop ships: a GNLSE with Raman response, a
four-mode capillary with plasma, a mode-averaged capillary, a radial filament and a
one-dimensional free-space beam. The standard and thorough levels also time four of
them at production size (16384 samples, four modes on 5120, 384 radial nodes, 4096
columns), because a setting that speeds up small problems can slow large ones down. It
never runs your own simulation; for that, see
[refining one simulation](#refining-one-simulation).

1. **Waiting and the team.** Idle threads either sleep at once or spin briefly before
   the next region; spinning saves the 16 to 72 µs of a wake-up, sleeping frees the
   cores. The team may also be narrower than the job: the fastest kind of core on a
   chip with several, one socket of two (standard and thorough only, where the
   production-sized problems can show what a narrower team costs). Each candidate is a
   child process with those settings.
2. **Operation grain.** Each kind of operation (stage passes, responses, products,
   transforms, Raman histories, copies) threads above a size threshold. The
   calibration moves each threshold past the work sizes the problems actually run,
   and tries other medium widths and compact teams for one-column problems.
3. **FFT planning** (standard and thorough). Every transform the problems create is
   planned alone at FFTW's `measure` effort, and serial ones at `patient` too at the
   thorough level, within a time limit; each group of plans (serial rows, threaded rows,
   transverse planes) keeps the effort that runs fastest *inside the step*, against
   plans made as a run without a profile makes them. A more exhaustive plan is not
   always a faster one, and a plan chosen alone not always the best in the step. Only
   a winning effort's wisdom is kept, so later runs build those plans at once instead
   of measuring again; without one, runs plan as they would without a profile.

A candidate replaces the built-in value only if it **wins in every round** of
interleaved timings by more than the noise measured by timing the built-in setting
against itself, **and is slower on no problem** by more than that noise plus 5 %; the
winners are then validated together on fresh rounds, under the same bar. Ties and
anything uncertain keep the built-in value. On a busy machine (other work on the job's
cores or their hyperthreads, time stolen by a hypervisor, or timings noisier than 10 %)
the calibration says so and publishes nothing.

| level | time | waiting and team | grain | FFT planning | rounds |
|---|---|---|---|---|---|
| `--quick` | about 20 s | the wait only | medium width and compact team | plan policy | 3 |
| standard | about 90 s | yes, two narrower teams | thresholds of the main operations | `measure`, 0.25 s a plan | 5 |
| `--thorough` | about 5 min | yes | thresholds and full sizes of every operation | `measure` and `patient`, 20 s a plan | 7 |

The standard and thorough levels also calibrate the smaller allocations they can pin
children to inside the job, such as one socket, one NUMA node or one L3 cache group,
while their time allows, so that one calibration of a whole node covers most smaller
jobs on it.

## Where the results live

One directory per user, the platform's cache directory, which `SOLPROP_CACHE_DIR` moves:

| platform | directory |
|---|---|
| Linux | `$XDG_CACHE_HOME/solprop`, or `~/.cache/solprop` |
| macOS | `~/Library/Caches/solprop` |
| Windows | `%LOCALAPPDATA%\solprop\Cache` |

A profile is kept per machine and build: the processor and its cache and memory
layout, the operating system, the native build and the calibration's own version. A
new release, a rebuilt extension or another machine finds no profile and runs on the
built-in table, so nothing stale is ever applied. Within a profile there is one row per
**allocation**: the cores a job holds, of which kinds, and how many caches, NUMA nodes
and sockets they span. Eight cores sharing a cache behave differently from eight across
two sockets.

A job takes its own allocation's row. Without one, it takes the workspace settings of
the row with the same layout and the nearest smaller core count, and otherwise the
built-in table.

## The first run

With no profile for the machine, the build and the job's allocation, the first native
run prints a notice like

```text
CalibrationNotice: solprop is calibrating the native backend to this machine (quick,
about 20 s on 18 cores; once per machine, allocation and release). The profile is kept
in /home/me/.cache/solprop. Run `solprop calibrate` for a fuller calibration, or set
SOLPROP_CALIBRATION=use to never measure, or =off to ignore profiles.
```

and calibrates at the quick level before it starts. Importing solprop or the native
backend never calibrates. A script whose native backend has not loaded yet takes the
whole new profile; a notebook that has already run a native simulation applies the new
operation grain at once and the new waiting and team from its next session.

`SOLPROP_CALIBRATION` chooses:

| value | effect |
|---|---|
| `auto` (default) | use a profile; calibrate quickly on first use when there is none |
| `use` | use a profile; never measure implicitly |
| `off` | the built-in table and no stored FFT wisdom, whatever the store holds; for bisecting and for holding settings fixed while code changes |

The notice is a Python warning of its own category, shown once; silence it with the
standard warnings filters (`warnings.filterwarnings("ignore",
category=solprop.exceptions.CalibrationNotice)`) or with `SOLPROP_CALIBRATION=use`.

Concurrent first runs on one machine, such as the tasks of an array job, wait for the
one calibrating, then use its profile. A calibration that found the machine too busy
publishes nothing, and the next first run tries again after a day.

## Scans and clusters

A scan calibrates its workers' allocation at the standard level before it starts them,
in a child pinned like one worker, unless a standard or thorough row already matches;
a scan usually runs for a long time, often after waiting in a queue, so a minute up
front is cheap. Workers never calibrate. Remote scans calibrate on their own hosts.

On a cluster, calibrate once per node type and share the profile:

```sh
# on one node of each type, inside a job holding the whole node:
solprop calibrate --thorough
solprop calibrate --export /shared/solprop-profiles/node-type-a.json
# every user then points at the shared directory:
export SOLPROP_CALIBRATION_PATH=/shared/solprop-profiles
```

`SOLPROP_CALIBRATION_PATH` lists read-only directories searched after your own store,
separated like `PATH`; a file exported for one node type applies only on that type.
`solprop calibrate --import FILE` puts an exported profile into your own store.

## Reading `--show`

```text
allocation:   18 cores
profile:      standard, generation g20261108T152341Z-1272949-c0d036, 2026-11-08T15:23:41Z
              measured on this allocation
process:      threads 18, wait spin
workspace:    medium_threads 8 (built-in 4)
fitted under: threads 18, wait spin
decided:      the built-in rules kept: no candidate won every round and its validation;
              grain: admitted: medium 8; FFT planning: the plan policy's efforts
noise:        0.02 (A/A)
candidate:    passive/18 1.272 (lost or tied in a round: worst ratio 1.280 against 0.980)
```

`process` is what the job's native runtime starts with, `workspace` each operation
setting beside the built-in value it replaces, and `fitted under` the process settings
those were validated under: a job that sets its own wait or thread count keeps the
built-in table for them. Each candidate's score is its median time relative to the
built-in setting. Every run records the settings it actually applied in its metadata
(`calibration` in the backend's settings), so a result says what produced it.

`solprop calibrate --clear` forgets this machine and build's profiles.

## Refining one simulation

For a costly campaign, a second stage fits the settings to one simulation:

```text
report = simulation.calibrate(budget=60.0)
print(report.describe())
```

or `solprop calibrate --recipe run.toml`. It builds fresh native workspaces of the
simulation in the calling process and times trial steps of its own operators at its
start and a third and two thirds along it, under candidate operation settings; the
simulation itself is not changed. Speed alone never admits a candidate: its steps must
agree with the built-in's to within ten times the run's relative tolerance. The result
is kept under the simulation's *execution descriptor* (its grid, modes, nodes, chunk,
responses, integrator and the process's settings), and every later run or scan point
with the same descriptor applies it, for as long as the machine profile it refined
stays the same.

## Overrides

Explicit settings keep their precedence over any profile: `SOLPROP_THREADS`,
`SOLPROP_NATIVE_THREADS` and the pool variables, `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and
`KMP_BLOCKTIME`, `SOLPROP_FFTW_PLAN`, `SOLPROP_FFTW_WISDOM` and
`SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES` ([Performance tuning](tuning.md)). A setting
measured under conditions an override changes is not used with it: operation thresholds
fitted under spinning threads give way to the built-in table when you make the threads
sleep.

The design and its measurements are in
[SEP 46](../../dev/seps/sep-0046-machine-calibration.md).
