# Performance tuning

A solprop run spends its time in a handful of operations repeated at every stage of
every step: fast Fourier transforms between time and frequency, dense matrix products
between modal amplitudes and the transverse field (for multimode and general modal
guides), the pointwise nonlinear response (Kerr, plasma, Raman) at every quadrature node,
and the Raman convolution. Each of these runs on a **thread pool**, and the pools belong
to different libraries: the backend's own kernels, FFTW, and a BLAS. This page explains
which pools exist, how each is sized by default, which environment variables change
them, and how to choose settings for one large run, for a scan of many runs, and on a
GPU.

Two rules apply throughout:

- **Set threads before Python starts.** Every pool below is sized when its library is
  loaded or when the backend is built. Some libraries allow a smaller active
  thread count later, but this is not a portable way to reconfigure all pools. Put the variables on the command line or in the job script, not in
  `os.environ` halfway through a notebook.
- **Measure your own problem.** The defaults come from measurements on representative
  problems, but the best settings depend on the grid, the number of modes or nodes, the
  responses switched on and the machine. Time about ten steps of your real problem
  (a short fibre, `n_save=2`) at two or three settings before a long campaign.

None of the settings on this page changes the physics. Thread counts can change the
order of floating-point sums inside the transforms, so results from different thread
counts can differ by rounding (a few units in the last place); a run repeated with the
same settings on the same build is reproducible.

## Which backend

The backend decides which pools exist at all (see [Choosing a backend](backends.md)):

| backend | parallel pools | notes |
|---|---|---|
| `numpy_ref` | NumPy's BLAS only | the reference; single-threaded apart from matrix products, and roughly 6 to 40 times slower than threaded native |
| native CPU (`native`) | OpenMP kernels, FFTW (modal and spatial transforms), the Raman convolution transforms (Kokkos-FFT over FFTW, with selected macOS shapes on Accelerate's vDSP), a vendor BLAS | uses every CPU in the process's affinity mask unless told otherwise |
| native GPU (`native`, CUDA or HIP build) | the device | CPU pools are irrelevant; one device per process |

`backend="auto"` chooses the native backend, a required dependency, and has no fallback
([which backend a run uses](backends.md#which-backend-a-run-uses)). A **locally built** one
([The native backend](native.md)) goes a little further than the wheel, since it links a
vendor BLAS and compiles for the
machine's own instruction set; [which backend to use](backends.md#which-backend-to-use)
compares the backends by problem.

## Which stepper

Each interface has its own default integrator
([the defaults](../numerics/choosing-a-stepper.md#the-defaults)): `Multistep()` on
`prop_capillary`, `prop_radial` and `prop_planar`, `RungeKutta()` on `prop_gnlse`,
`prop_waveguide` and `prop_free`. `Multistep()` evaluates the nonlinear response once
per step; `RungeKutta()` usually six times, in fewer, longer steps. `Multistep()` tends
to gain where the response dominates a step's cost, `RungeKutta()` where it is cheap.
Compare the two on a short stretch of your own problem when it matters: the call counts
and the time per call are in the run's step history and log.
[Choosing a stepper](../numerics/choosing-a-stepper.md) sets out when each is faster,
with the measurements.

**Many save planes.** Landing on every save plane caps the step at the plane spacing.
With hundreds of planes, `interpolate_planes=True` steps across them and interpolates
each inside its step instead, and `step_planes=k` saves the march's own steps at no extra
cost ([the solver](../numerics/solver.md), "Choosing the saved planes").

## The thread budget

Every thread pool stays within the process's **thread budget**, the smallest of:

- its CPU affinity mask, the CPUs it may run on (narrowed by `taskset`, `numactl` or a
  batch scheduler's allocation);
- the physical cores that hold those CPUs, unless `SOLPROP_THREADS` is set: a core's
  second hardware thread shares its arithmetic units, and on 36 cores of two Intel Xeon
  Gold 6240 one thread per core ran every benchmark case 6 to 31 % faster than two
  (read on Linux and Windows; Apple silicon has one thread per core);
- the tightest Linux control-group CPU quota on its path (containers), rounded **down**:
  a quota limits CPU time, and a team of threads that exceeds it is throttled whole, so
  a quota of 1.5 CPUs runs one thread;
- `SOLPROP_THREADS`, if set.

An explicit count (`threads=` or a variable below) is clamped to the budget rather than
refused; the pools take turns, so
each may use all of it. Set `SOLPROP_THREADS` to each process's share when several
processes share CPUs without being pinned apart. That includes several solprop
processes in one CPU-limited container (a Kubernetes CPU limit, Docker `--cpus`): the
quota is the container's, and each process would otherwise take all of it. A cloud
virtual machine or a Slurm job has no quota by default (`cat /sys/fs/cgroup/cpu.max`
prints `max`); there the CPUs it may run on are the limit. `SOLPROP_OVERSUBSCRIBE=1` lets an
explicit count exceed the CPUs and the quota (never `SOLPROP_THREADS`), for a test that
compares thread counts on a small machine; a run has no reason to set it. The budget and
the limit that set it are recorded in the run's metadata (`thread_budget`,
`thread_budget_source`).

These settings apply whatever the pool:

| variable | what it sets | default |
|---|---|---|
| `SOLPROP_THREADS` | the process's thread budget, which the native backend's pools stay within; it may exceed the physical cores, up to the CPUs | the affinity mask, its physical cores and the quota |
| `SOLPROP_OVERSUBSCRIBE` | `1` lets an explicit count exceed the CPUs and the quota | `0` |
| `SOLPROP_BLAS_THREADS` | the width of every BLAS other than the native backend's own while a native backend lives ([below](#blas-and-vendor-libraries)) | 1 |
| `SOLPROP_FFTW_PLAN` | the native backend's FFTW planner effort: `estimate`, `measure` or `patient` | `measure`, except threaded row transforms, which stay at `estimate` |
| `SOLPROP_FFTW_WISDOM` | a file name, beside which the native backend keeps its FFTW wisdom ([below](#the-native-backends-pools)) | none |

**Free-space node groups.** A free-space, line or planar evaluation works through the
plane's columns in node chunks, each paying a fixed cost for its launches, its temporal
transforms and its gather and scatter. Their default budget is therefore 1 GiB, against
64 MiB on modal routes; `SimulationSettings.node_budget` sets it explicitly. On a laptop
short of memory a smaller budget saves memory at some speed; on a large device a larger
one can still help.

**Where a free-space step goes.** On the native backend the stages of a free-space
evaluation are profiling regions (`solprop_native.free.*`) that any Kokkos Tools timer
reports ([profiling](#profiling-the-native-backend)).

**FFTW planning.** `measure` times candidate algorithms once per transform shape and
keeps the fastest; that pays off on long runs, but the timing is only meaningful on an
otherwise idle machine, and planning a large multimode problem can take seconds. Use
`SOLPROP_FFTW_PLAN=estimate` for short runs, tests and busy shared machines, and
`SOLPROP_FFTW_WISDOM=~/.cache/solprop/wisdom` to plan once and reuse the plans across
runs of the same shapes (the wisdom's digest is recorded in each run, since plans chosen
from different wisdom are different executions). The native backend keeps those plans
in a file of its own (below).

## The native backend's pools

The native backend reads its thread settings once, when `solprop_native` is first
imported:

| variable | pool | default |
|---|---|---|
| `SOLPROP_NATIVE_THREADS` | the size of Kokkos' OpenMP pool, an explicit count within the thread budget (above the CPUs only with `SOLPROP_OVERSUBSCRIBE=1`, above `SOLPROP_THREADS` never) | the thread budget |
| `SOLPROP_NATIVE_KERNEL_THREADS` | response, stage and reduction kernels | sized per operation, up to the cap |
| `SOLPROP_NATIVE_FFT_THREADS` | FFTW threads for the modal and spatial transforms | sized per plan, up to the cap |
| `SOLPROP_NATIVE_RAMAN_FFT_THREADS` | threads for the Raman convolution transforms of complete node chunks | sized per plan, up to the cap |
| `SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS` | threads for the Raman transforms of the last, partial node chunk | sized from active rows, up to the cap |
| `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES` | the real-side samples per execution from which a modal or spatial FFT plan is threaded | 16,384 (32,768 on macOS) |
| `SOLPROP_NATIVE_BAND_TILES` | the field plasma column's band form: unset for the build's own (tiled on a GPU, a team per node on a CPU), `0` for the team column, a positive `N` for the tiled column with host teams of `N` threads | unset |

The four pool variables are optional positive integers, clamped to the cap; an invalid
value raises `ValueError` at import. An explicit pool setting forces its clamped width
and bypasses automatic sizing, including for short FFTs. Setting only the global cap
retains automatic sizing. GPU builds ignore these CPU pool settings.

On OpenMP CPU builds, operation dimensions select serial, intermediate or full-width
execution. Stage and reduction work, packing and temporal copies, nonlinear responses,
native dense transforms, coupled matrix functions and FFT plans have separate size rules. Response sizing accounts for both the total
work, active columns and fine samples per column. The choices and plans are cached when the
backend is constructed; no runtime tuning occurs during propagation. Partial chunks
and fused batch updates use their own work sizes. Vendor BLAS pools retain their
process-start settings. Non-OpenMP host builds preserve their execution-space policy;
`build_info().cpu_size_policy_enabled` reports whether operation widths are supported.

The thresholds below are shared across CPU hosts, except that macOS keeps higher stage
and FFT starts (8,192 and 32,768), where a short serial loop is still the faster one;
`C` is the clamped capacity of the
relevant pool, and sizes count work in the operation itself, not bytes. Between the two
thresholds an operation uses `min(4, C)` threads; equal thresholds give only serial or
full-width execution. A kernel operation (any row but the two FFT rows) of a backend that
already runs a full-width kernel takes the full pool instead of four threads: GNU OpenMP
ends the surplus threads of its team whenever a narrower parallel region follows a wider
one and starts them again for the next wide one, which cost a four-thread kernel between
full-width ones about half a millisecond each time. FFT plans keep their four threads,
since FFTW's threaded loops run on the runtime's whole team.

| operation | work used for sizing | threading starts | full pool starts |
|---|---|---:|---:|
| stage, update and reduction | elements in the field or fused batch | 4,096 (8,192 on macOS) | as threading |
| packing and temporal copies | entries actually copied | 65,536 | 65,536 |
| nonlinear response | active columns × components × fine samples | 32,768 | 196,608, or 65,536 fine samples per column, or 40 component columns |
| native/custom GEMM | output rows × inner dimension × samples | 1,048,576 | 1,048,576 |
| general modal synthesis/projection | active nodes × components × spectral samples × modes | 65,536 | 65,536 |
| coupled matrix functions | active spectral bins × modes³ | 32,768 | 32,768 |
| exponential Adams trial | elements in the field × history rows | 8,192 | 65,536 |
| modal/spatial FFT | real-side samples in the actual batch | 16,384, 32,768 on macOS (`SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`) | 131,072 |
| Raman FFT | real-side samples in the actual batch | 65,536 | 262,144 |

A response column is one active node and one component; its column-length and
column-count rules apply only past its threading start. General modal map updates use
the two maps' combined element count times the polynomial degree. Native GEMM sizing
never changes a vendor BLAS's width or selects a different provider.

The [CPU threading measurements](../../dev/benchmarks.md#kokkos-cpu-operation-size-threading)
explain the thresholds. For example, an 8,192-sample copy remains serial even when a
similarly sized arithmetic stage benefits from threading. Short response columns can
use four threads while long columns or wider batches use the full pool. The backend applies these
choices automatically; start by setting only the process cap when sharing a machine.

`python -m solprop_native` records the policy constants and explicit override mask.
Its existing thread counts remain process capacities. The backend's metadata records
`execution_sizing`, with the chosen `threads` and sizing `elements` for each operation;
`batch_execution_sizing` covers fused batch work. Response entries also report
`fine_samples` and `columns`, because equal total work can have different column
lengths. `build_info()` reports the resolved process values as `requested_threads`,
`runtime_threads`, `thread_count` (kernels), `fftw_threads`, `raman_fft_threads`,
`raman_tail_fft_threads`, `threaded_fft_min_samples`, `blas_threads` and
`blas_threading`; `cpu_size_policy_enabled` and `explicit_thread_pools` (a bit mask:
kernel 1, FFT 2, Raman 4, Raman tail 8) distinguish automatic sizing from explicit
requests.

Free-space transforms are planned at `SOLPROP_FFTW_PLAN`'s effort
(`measure` by default); `estimate` skips the measured planning for
short runs and tests. Answers at different thread counts may differ by rounding; solprop
holds them to its rounding bounds, not to identical bits
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).
Contiguous modal and Raman FFTW plans default to measured planning at one thread and
estimate planning at wider widths; the explicit effort overrides both, including
long Raman rows and partial chunks. Measured planning can add substantial construction
time even when the warmed propagation is faster. When construction dominates a short
run, keep the plans in a wisdom file (below) rather than give up measuring them;
`SOLPROP_FFTW_PLAN=estimate` skips planning altogether, but its plans ran one-thread
windows up to 1.6 times slower on an AMD EPYC 7763. On macOS with NEON FFTW, serial and long
Raman rows use Kokkos-FFT, while parallel power-of-two rows of at most 16,384 samples
retain vDSP. The [build guide](building-native.md) describes the coherent dependency
recipe and the fallback for a library without NEON codelets.

**Reusing the native backend's plans.** `SOLPROP_FFTW_WISDOM`
keeps the native backend's measured plans: a backend imports them before it
plans and adds the plans it measured after. Nothing is read or written unless the
variable names a file. The native backend links its own FFTW, so its plans go to a file
of their own beside the named one, not to the named file itself: `<name>.native-<key>`, whose key digests the FFTW
library's version and configuration and the processor model: machines of different
kinds that share a home directory keep separate plans, and FFTW refuses wisdom another
build of it wrote. The processes of a scan can share one file. Readers take no lock,
since the file is only ever replaced whole; a writer takes `<file>.lock`, merges what
others stored, and writes only when that adds a plan. A file that cannot be read or
written is reported in the log and costs only the measurement. Wisdom chooses among
FFTW's exact algorithms, so it changes no result beyond the rounding a fresh
measurement can change; the run's metadata records `wisdom_loaded` and the
`wisdom_digest` of the plans in force. For a scan of one-thread jobs it removes most of
each job's construction: on a Xeon Gold 6240, constructing the averaged helium
capillary took 1.40 s cold and 0.03 s with the wisdom in place, beside a 0.27 s window
(1.78 s and 0.05 s on an AMD EPYC 7763), and the comb-to-continuum capillary 5.0 s and
0.06 s ([measurements](../../dev/benchmarks.md#persistent-fftw-wisdom)). For example:

```sh
export SOLPROP_FFTW_WISDOM=~/.cache/solprop/wisdom
```

The first job of each shape measures and stores its plans; every later one, in any
process, finds them. Threaded row transforms are planned at estimate unless
`SOLPROP_FFTW_PLAN` asks otherwise, so they have no measured plans to keep; but plans
that a run under an explicit `measure` stored for a width are found by later runs at
that width, whatever their effort.

`OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS` do **not** size the Kokkos pool — the backend
initialises Kokkos with an explicit thread count, which takes precedence; set
`SOLPROP_NATIVE_THREADS` instead. (Kokkos prints a warning at start-up if
`OMP_NUM_THREADS` asks for fewer threads than it was given.)

**The default cap is the thread budget** (above): the CPUs in the process's affinity
mask, a container's quota and `SOLPROP_THREADS`. Automatic sizing avoids assigning all
of those threads to small operations. On a shared machine or when several solprop
processes run side by side, set `SOLPROP_THREADS` or
`SOLPROP_NATIVE_THREADS` to each process's CPU budget. The operation policy stays within
that budget.

### OpenMP placement and waiting

The native CPU backend runs on OpenMP, whose standard variables apply:

- `OMP_PROC_BIND` and `OMP_PLACES` pin threads to cores, which avoids threads migrating
  between cores and sockets: `OMP_PROC_BIND=spread OMP_PLACES=cores` is a good default
  for one run on a multi-socket machine. When none of `OMP_PROC_BIND`, `OMP_PLACES`,
  `GOMP_CPU_AFFINITY` and `KMP_AFFINITY` is set, the native backend's import sets
  `OMP_PROC_BIND=false`: the threads stay unpinned, as the runtimes leave them anyway,
  and Kokkos does not print its notice that the variable is unset, whose advice is for a
  run with the machine to itself. The run's metadata records what applied as
  `openmp_proc_bind`. With a build whose FFTW uses pthreads rather than OpenMP, prefer
  `OMP_PROC_BIND=false`.
- `OMP_WAIT_POLICY` decides what idle threads do between parallel regions: `active`
  keeps them spinning, `passive` lets them sleep at once. **The native backend chooses
  for you** when none of `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` and `KMP_BLOCKTIME` is set.
  It runs several thread teams in turn (Kokkos', the threaded FFTs', the BLAS's), and
  when its threads fill the CPUs one team's spinning threads take the cores the next
  needs: on 16 cores of an AMD EPYC 7763 the runtime's own waiting made the four-mode
  capillary 25 times slower (334 s against 13.5 s). So the import sets
  `OMP_WAIT_POLICY=passive`. The setting stays in the environment, so worker processes
  wait alike, and the run's metadata records what applied as `openmp_wait_policy`.
  Where hardware threads are spare and nothing else will use them, spinning is faster:
  on 36 cores of two Intel Xeon Gold 6240 (72 hardware threads), with the whole machine
  to itself, `OMP_WAIT_POLICY=active` ran the supercontinuum in 49 s against 94 s and
  the capillary in 9.7 s against 14.7 s. It is not the default because the runtime
  reads the policy once, at start-up, and work that arrives on the machine later takes
  those spare threads; set it for a run on a machine it has to itself. LLVM's OpenMP runtime (used on macOS) also
  honours `KMP_BLOCKTIME`, the time a thread spins before sleeping; do not set it for
  GNU OpenMP.
- On cores your run has to itself, such as a batch job's allocation, also pinning the
  threads helps: with idle threads sleeping, `OMP_PROC_BIND=close OMP_PLACES=cores` ran
  the tracking cases 11 % faster (geometric mean) on 16 cores of a shared AMD EPYC 7763
  node, the comb-to-continuum and graded-index cases 28 % and the radial filament 17 %,
  while on a dedicated socket of a Xeon Gold 6240 it changed little (1 % slower overall).
  Keep `close`: `spread` scatters a four-thread kernel over the EPYC's core complexes.
  Leave it unset where several processes share a machine, since each would pin its
  threads to the same first cores.

## BLAS and vendor libraries

The BLAS matters for the dense modal transforms of multimode capillaries, coherent walls
and the general modal route. A BLAS with its own thread pool beside the native backend's
oversubscribes the cores: with a pthreads OpenBLAS at the full core count, the modal and
wall cases ran 10 to 75 times slower than necessary. The Linux distribution build of the
native backend therefore carries an OpenBLAS built with OpenMP, which shares the
backend's runtime. At start-up the backend fits a host OpenBLAS to its own pool unless
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` or `OMP_NUM_THREADS` is set: a pthreads build
is held to one thread, an OpenMP build capped at the Kokkos thread count. NumPy's own BLAS, a separate library beside
the native backend's, serves only small host-side products during a native run, yet its
idle workers spin on the cores the native pool needs: while a native backend lives, every
BLAS threadpoolctl finds other than the backend's own is held to one thread
(`SOLPROP_BLAS_THREADS` sets another count), and the last backend to close gives the pools
back. The run's metadata records what applied as `host_blas`.

Guide construction and coupling happen before a backend exists. They also hold the
controllable host BLAS pools at one thread, or at `SOLPROP_BLAS_THREADS` within the
process budget. The hold covers `ModalGuide.from_provider`, a standalone graded-index
mode solve (including a node rule's tail certificate), `general_modal_route`, kernel
assembly, `FieldPlane.from_pulse` and `couple`. The general modal route builds its
large reference tables inside `Simulation.from_waveguide`, before kernel assembly;
an incident launch builds its plane's radial transform before coupling.
This overrides `OPENBLAS_NUM_THREADS` and the other vendor defaults for those calls,
then restores the previous limits even if construction fails. A native backend's own
BLAS keeps its fitted width. Nested or overlapping construction shares the first
scope's hold. Construction and live backend holds share each library's ownership:
the narrowest active request applies, and the last owner restores the original width,
even if the calls start and finish in different orders. Separate processes, including
scan workers, keep their own pools. Accelerate exposes no runtime
thread control to threadpoolctl, so these scopes leave it unchanged.

Outside a backend's lifetime and these construction scopes, a BLAS's own variables apply:

| library | thread variable | notes |
|---|---|---|
| OpenBLAS (pthreads build) | `OPENBLAS_NUM_THREADS` | the usual Linux default |
| OpenBLAS (OpenMP build) | `OMP_NUM_THREADS` | ignores `OPENBLAS_NUM_THREADS`; the best choice beside the native backend, which shares its runtime |
| Intel MKL | `MKL_NUM_THREADS`, or per domain `MKL_DOMAIN_NUM_THREADS` | e.g. `MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_BLAS=1,MKL_DOMAIN_FFT=8"` |
| Apple Accelerate | `VECLIB_MAXIMUM_THREADS` | threads through Grand Central Dispatch |
| AMD AOCL-BLIS | `BLIS_NUM_THREADS` | |

NumPy's own BLAS (used by `numpy_ref` and by solprop's translation layer) answers to the
same variables. A locally built native backend links a vendor BLAS (Accelerate on macOS,
MKL when oneAPI is available on Linux, otherwise OpenBLAS); its build record says which
(`python -m solprop_native`).

**One OpenMP runtime per process.** The native backend, FFTW and some BLAS builds each
use OpenMP, and two different OpenMP runtimes loaded into one process can crash or
oversubscribe the machine. solprop does not use pyFFTW, whose wheels bring a runtime of
their own, but a script or another package may. On macOS `solprop_native` must then be
imported first, and raises an error otherwise: put
`import solprop_native` at the top of such a script
([OpenMP runtimes](building-native.md#openmp-runtimes)).


:::{warning}
On Windows on Arm, setting `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or `GOTO_NUM_THREADS`
to a small number can crash or hang NumPy's own linear solves, with solprop not involved.
On NumPy 2.5.2, widths 1 and 2 failed at matrix sizes 23 to 31, and width 3 at size 31.
Leave them unset. solprop warns on import if it finds such a value. `threadpoolctl` limits
were safe at the same sizes, so `Pool` workers on this platform use those instead of the
variables.
:::

## Recipes

**One large run on a workstation.** Give the run every performance core, pin it, and let
idle threads sleep if you are using the machine for anything else:

```sh
SOLPROP_NATIVE_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=passive \
    python my_run.py
```

`numpy_ref` runs on one thread apart from its matrix products, and no setting changes that.

**How far a single run scales.** A single-mode fibre on a few thousand time samples
gains little beyond two to four threads: its transforms are short and each step is
mostly synchronisation. Multimode capillaries and the general modal route, whose
products and node loops are large, keep gaining to eight to eighteen threads. Free-space
runs with large transverse grids scale furthest.

**A scan of many runs.** For a scan over pressure, energy or noise seeds, many
single-threaded workers almost always beat one many-threaded run: each worker keeps its
own caches and nothing waits for synchronisation. solprop's
{class}`~solprop.driver.executors.pool.Pool` executor does this for you: every worker is a
newly spawned Python process started with **all** of
`SOLPROP_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` set to its `threads` (one by default) and
`OMP_WAIT_POLICY=passive`, so every pool is sized within that budget,
and on Linux each worker is pinned to its own CPUs. The number of workers is the number of CPUs available divided
by `threads`, reduced if a memory probe of the first point shows that fewer fit (four
fifths of the memory, less the controller's own, divided by one worker's footprint).
`run_scan` runs its points one after another in the calling process unless it is given
an executor, so ask for the pool, in the script or on the command line:

```sh
python my_scan.py                            # the script calls run_scan(..., executor=Pool())
python -m solprop.scan my_scan.py --workers 8    # or let the launcher choose Pool
```

A problem too large to fit many copies in memory can use fewer, wider workers,
`Pool(threads=4)` (or `--threads 4`). To run two scans side by side on one machine, give
each its own CPUs with `SOLPROP_SCAN_CPU_OFFSET` and `SOLPROP_SCAN_CPU_COUNT`. See
[scans and batches](../workflows/scans.md). Stay on the native backend at one thread per
worker too, and time one point of the scan with
your real number of workers before a large campaign.

If you manage the parallelism yourself — a job array, GNU `parallel`, several scripts
started by hand — each process needs every pool held to one thread explicitly, before Python starts:

```sh
SOLPROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 OMP_WAIT_POLICY=passive OMP_PROC_BIND=false \
    python point.py
```

On Windows on Arm, leave out `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`, for the reason
under [BLAS and vendor libraries](#blas-and-vendor-libraries).

On a multi-socket server, keep a multi-threaded run's threads within one socket
(`OMP_PLACES=cores` with `OMP_PROC_BIND=close`, or `taskset`/`numactl`): crossing
sockets usually costs more in memory traffic than the extra cores return for the
problem sizes solprop runs, and two runs, one per socket, are usually faster than one
run across both.

**A cluster.** {class}`~solprop.driver.executors.slurm.Slurm` writes an array job whose tasks
each run a `Pool` inside their allocation; it respects `SLURM_CPUS_PER_TASK`, and its
`threads` sets the same variables in every task.

**A GPU.** On a CUDA or HIP build of the native backend the CPU pools do not matter; the
backend uses one device per process, chosen with `CUDA_VISIBLE_DEVICES` or
`HIP_VISIBLE_DEVICES`. Small problems are limited by kernel-launch latency and often run
faster on the CPU: run them with the CPU wheel in a second environment. Run independent problems in
separate processes to use several devices. A job set up for a device names the requirement,
and stops before it starts if the environment has lost its GPU build
([a job that must run on a device](native.md#a-job-that-must-run-on-a-device)):

```sh
CUDA_VISIBLE_DEVICES=0 SOLPROP_REQUIRE_DEVICE=cuda python my_run.py
```

See [The native backend](native.md).

**A processor without AVX2.** An x86-64 wheel loads its x86-64-v2 build there, which takes
about 20 to 25 % longer by geometric mean; `solprop_native.build_info()` reports which build
is loaded, and `SOLPROP_NATIVE_CPU_VARIANT=x86-64-v2` asks for that build on a faster
processor, for comparisons ([The native backend](native.md#installing)).

**Memory.** Large multimode and general modal problems can be limited by memory before
threads; `SOLPROP_MEMORY_BUDGET` and the admission check are described in
[Execution resources](resources.md#memory).

## Checking what was used

The backend used is logged at INFO level, and the pools it chose are in the run's
backend record, which is saved with the output:

```python
import solprop
from solprop.driver import prop_capillary

solprop.configure_logging("INFO")  # prints "Using native backend", among others
run = prop_capillary(
    125e-6,
    1e-3,
    "He",
    1.0,
    wavelength=800e-9,
    wavelength_limits=(200e-9, 2e-6),
    trange=100e-15,
    fwhm=10e-15,
    energy=1e-9,
    n_save=2,
    progress="none",
    backend="native",
)
settings = run.metadata.backend.settings
print(
    {key: settings[key] for key in ("runtime_threads", "thread_count", "fftw_threads")}
)
```

The same record also carries `raman_fft_threads`, `blas_library` and the build profile;
`python -m
solprop_native` prints the whole build record without running anything.

## Profiling the native backend

Kokkos profiling hooks are compiled into every native build, and every kernel, region
and allocation carries a `solprop_native.*` label (for example
`solprop_native.transform.*`, `solprop_native.raman.*`, `solprop_native.plasma.*`,
`solprop_native.statistics.*`). Load any [Kokkos Tools](https://github.com/kokkos/kokkos-tools)
plugin at run time with `KOKKOS_TOOLS_LIBS`; nothing is linked into the package. For the
simple kernel timer:

```sh
git clone --depth 1 --branch 5.2.0 https://github.com/kokkos/kokkos-tools.git
cmake -S kokkos-tools -B kokkos-tools/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/kokkos-tools/install"
cmake --build kokkos-tools/build --target install --parallel

KOKKOS_TOOLS_LIBS="$PWD/kokkos-tools/install/lib/libkp_kernel_timer.so" \
  python my_simulation.py
```

On macOS the plugin is `libkp_kernel_timer.dylib`. At shutdown the timer reports call
counts and accumulated times for the `solprop_native.*` labels and for Kokkos-FFT's
`KokkosFFT::exec_plan` regions; the NVTX and ROCTX connectors pass the same labels to
Nsight Systems and the ROCm profilers. The free-space source evaluation's stages are
regions of their own (`solprop_native.free.core`, `.transverse_inverse`, `.pack`,
`.to_time`, `.clear_projected`, `.response`, `.to_frequency`, `.transverse_forward`);
with no tool loaded a region costs one flag test. For the Runge–Kutta march, the
backend's `metadata()["source_evaluations"]` counts the native source evaluations since
the initial state was set, rejected trials included.
