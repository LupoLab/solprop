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
| `numpy_ref` | NumPy's BLAS only | the reference; single-threaded apart from matrix products |
| `numba_cpu` | Numba kernels, FFTW, BLAS (and FFTW for the transverse transforms of free-space runs) | sizes each pool from the problem |
| native CPU (`native`) | OpenMP kernels, FFTW (modal and spatial transforms), the Raman convolution transforms (Kokkos-FFT over FFTW, with selected macOS shapes on Accelerate's vDSP), a vendor BLAS | uses every logical CPU unless told otherwise |
| native GPU (`native`, CUDA or HIP build) | the device | CPU pools are irrelevant; one device per process |

`backend="auto"` chooses the native backend when `solprop-native` is installed and
`numba_cpu` otherwise. The largest single gain is usually installing a **locally built**
native backend ([The native backend](native.md)), which links a vendor BLAS and compiles for the
machine's own instruction set; [which backend to use](backends.md#which-backend-to-use)
compares the backends by problem.

## Which stepper

The default integrator, `Multistep()`, evaluates the nonlinear response once per step;
`RungeKutta()` usually six times, in fewer, longer steps. Keep the default wherever the
response dominates a step, and compare the two on a short stretch of your own problem
when it matters: the call counts and the time per call are in the run's step history and
log. [Choosing a stepper](../numerics/choosing-a-stepper.md) sets out when each is faster,
with the measurements.

**Many save planes.** Landing on every save plane caps the step at the plane spacing.
With hundreds of planes, `interpolate_planes=True` steps across them and interpolates
each inside its step instead, and `step_planes=k` saves the march's own steps at no extra
cost ([the solver](../numerics/solver.md), "Choosing the saved planes").

## The `numba_cpu` pools

`numba_cpu` sizes three pools from the problem when the backend is built, and records
what it chose:

**Kernel threads** (the Numba parallel pool). The response pass over quadrature nodes
runs in parallel only for large problems: above $10^6$ elements
($n_{to} \times n_\text{modes} \times n_\text{pol} \times n_\text{nodes}$, with
$n_{to}$ the number of samples on the oversampled time grid), or above 262 144 for a
guided run whose response is evaluated at more than one transverse node. Below that
threshold the kernels run serially (the **small** regime), because waking a pool costs
more than the work saves. Above it (the **large** regime) the pool takes every CPU
available to the process.

**FFTW threads.** A batched transform of at least 65 536 elements
($n_\text{columns} \times n_{to}$) gets up to four threads; smaller transforms get one.
More than four threads rarely repays the synchronisation on the problem sizes solprop
runs; transforms well above half a million elements can gain from eight. The Raman
convolution transforms and, for free-space runs, the transverse transforms are sized by
the same rule.

**BLAS threads.** The modal synthesis and projection products get two BLAS threads at
or above 262 144 elements ($n_\text{rows} \times n_{to}$), one below. The BLAS pool is
set through [threadpoolctl](https://github.com/joblib/threadpoolctl); Apple's
Accelerate exposes no thread control at all, and is recorded as `uncontrolled`.

"Every CPU available" means the smallest of the machine's logical CPU count, the
process's CPU affinity (for example under `taskset` or a batch scheduler), a Linux
cgroup CPU quota (containers) and `SOLPROP_THREADS`. `SOLPROP_THREADS` is the process's
thread budget on every CPU backend: each pool is still sized by the rules above, but
never beyond it, and an explicit count is clamped to it too. Set it to each process's
share when several processes share CPUs without being pinned apart.

Each pool can be fixed from the environment:

| variable | pool | default |
|---|---|---|
| `SOLPROP_THREADS` | the cap on every pool below (and on the native backend's) | every available CPU |
| `SOLPROP_NUMBA_THREADS` | Numba kernel threads | all available CPUs above the size threshold, else 1 |
| `SOLPROP_NUMBA_REGIME` | `small` (serial node loop) or `large` (parallel node loop) | chosen from the size threshold |
| `SOLPROP_FFTW_THREADS` | FFTW threads for the temporal and Raman transforms | every available CPU from 1 048 576 elements, up to 4 from 65 536, else 1 |
| `SOLPROP_TRANSVERSE_FFTW_THREADS` | FFTW threads for the transverse transforms of free-space runs | the same rule |
| `SOLPROP_BLAS_THREADS` | BLAS threads | 2 above 262 144 elements, else 1 |
| `SOLPROP_FFTW_PLAN` | FFTW planner effort: `estimate`, `measure` or `patient` | `measure` |
| `SOLPROP_FFTW_WISDOM` | a file of FFTW wisdom to import before planning and export after | none |

**Free-space node groups.** A free-space, line or planar evaluation works through the
plane's columns in node chunks, each paying a fixed cost for its launches, its temporal
transforms and its gather and scatter. Their default budget is therefore 1 GiB, against
64 MiB on modal routes; `SimulationSettings.node_budget` sets it explicitly. On a laptop
short of memory a smaller budget saves memory at some speed; on a large device a larger
one can still help.

**Where a free-space step goes.** `SOLPROP_PASS_TIMERS=1`, read when a `numba_cpu`
backend is built, gives it per-pass wall-clock timers for the Cartesian, line and planar
evaluations: the transverse and temporal transforms, the packs, copies, clears, the
response and the crops of each node chunk, and the steppers' traversals around them
(`NumbaCpuBackend.pass_timers`, from `solprop.backends.numba_cpu.pass_timers`). It is a
measurement aid for [SEP 28](../../dev/seps/archive/sep-0028-free-space-evaluation.md) and its
benchmarks; unset, nothing is timed and nothing is added to a pass. On the native backend the same
stages are profiling regions (`solprop_native.free.*`) that any Kokkos Tools timer
reports.

Every explicit count is clamped to the CPUs available. `NUMBA_NUM_THREADS`, Numba's own
variable, sets the size of the pool Numba creates at import, and the kernel pool, chosen
or explicit, is clamped to it as well; the FFTW and BLAS pools are not, since they are
not Numba's. `NUMBA_THREADING_LAYER` chooses Numba's threading layer (`omp`, `tbb` or
`workqueue`) and is normally best left alone.

**FFTW's idle threads on Linux.** pyFFTW's Linux wheels thread FFTW through their own
copy of GNU OpenMP. If that runtime's idle threads spin, they take the cores the Numba
kernels need between transforms. On 8 cores of an x86-64 server that made a four-mode capillary step
five times slower whenever FFTW had more than one thread. `numba_cpu` therefore imports
pyFFTW with `OMP_WAIT_POLICY=passive` set for that runtime alone, and the environment is
restored straight away, so other OpenMP runtimes still read yours. This is skipped if
you have set `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT`, because then your choice applies,
and `active` brings the slowdown back. It is also skipped when your script imported
`pyfftw` before solprop. The run's metadata records what applied as `fftw_wait_policy`.

**Numba's own threads on Linux.** Numba's OpenMP threading layer loads a second GNU
OpenMP runtime when its threads first start, and with every core in the pool its idle
workers spin on the cores FFTW's threads need: on a 64-core AMD server the
comb-to-continuum benchmark window took 17 s, against 5.4 s with the workers waiting
passively. `numba_cpu` therefore starts Numba's threads with `OMP_WAIT_POLICY=passive`
for that runtime. When the runtime is already loaded under settings solprop cannot change
(the native backend's build may load the same library first), it chooses Numba's
`workqueue` layer instead, which uses no OpenMP and was as fast (5.5 s). Your own
`OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` or `NUMBA_THREADING_LAYER` is respected. The run's
metadata records what applied as `numba_openmp`.

**OpenMP binding and `numba_cpu`.** `OMP_PROC_BIND` (or `OMP_PLACES`, which binds under
GNU OpenMP even alone) made the same window take 45-47 s, whatever Numba's layer: the
bound runtimes place their threads on the same few cores. On Linux `numba_cpu` therefore
loads pyFFTW's runtime and Numba's with binding off, and your variables stay set for any
other runtime, such as the native backend's. When the native backend, imported first under
binding, has bound the main thread to one core, `numba_cpu` gives the main thread the
process's CPUs back before it plans its transforms, which recovers modal runs; free-space
runs in that combination still had FFTW transforms several times slower on Linux, so
`numba_cpu` warns. If `pyfftw` was imported before solprop under binding, solprop cannot
undo it and warns too. In short: do not set `OMP_PROC_BIND` or `OMP_PLACES` for
`numba_cpu` runs. Binding brought the native backend no
measurable gain on one socket either; on several sockets, place whole processes with
`numactl` or `taskset` instead.

**FFTW planning.** `measure` times candidate algorithms once per transform shape and
keeps the fastest; that pays off on long runs, but the timing is only meaningful on an
otherwise idle machine, and planning a large multimode problem can take seconds. Use
`SOLPROP_FFTW_PLAN=estimate` for short runs, tests and busy shared machines, and
`SOLPROP_FFTW_WISDOM=~/.cache/solprop/wisdom` to plan once and reuse the plans across
runs of the same shapes (the wisdom's digest is recorded in each run, since plans chosen
from different wisdom are different executions).

## The native backend's pools

The native backend reads its thread settings once, when `solprop_native` is first
imported:

| variable | pool | default |
|---|---|---|
| `SOLPROP_NATIVE_THREADS` | the process-wide cap; Kokkos' OpenMP pool is created with this many threads, or with `SOLPROP_THREADS` if that is smaller | `SOLPROP_THREADS`, else the number of logical CPUs |
| `SOLPROP_NATIVE_KERNEL_THREADS` | response, stage and reduction kernels | sized per operation, up to the cap |
| `SOLPROP_NATIVE_FFT_THREADS` | FFTW threads for the modal and spatial transforms | sized per plan, up to the cap |
| `SOLPROP_NATIVE_RAMAN_FFT_THREADS` | threads for the Raman convolution transforms of complete node chunks | sized per plan, up to the cap |
| `SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS` | threads for the Raman transforms of the last, partial node chunk | sized from active rows, up to the cap |
| `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES` | the real-side samples per execution from which a modal or spatial FFT plan is threaded | 32,768 |
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

The thresholds below are shared across CPU hosts; `C` is the clamped capacity of the
relevant pool, and sizes count work in the operation itself, not bytes. Between the two
thresholds an operation uses `min(4, C)` threads; equal thresholds give only serial or
full-width execution.

| operation | work used for sizing | threading starts | full pool starts |
|---|---|---:|---:|
| stage, update and reduction | elements in the field or fused batch | 8,192 | 8,192 |
| packing and temporal copies | entries actually copied | 65,536 | 65,536 |
| nonlinear response | active columns × components × fine samples | 32,768 | 196,608, or 65,536 fine samples per column, or 40 component columns |
| native/custom GEMM | output rows × inner dimension × samples | 1,048,576 | 1,048,576 |
| general modal synthesis/projection | active nodes × components × spectral samples × modes | 65,536 | 65,536 |
| coupled matrix functions | active spectral bins × modes³ | 32,768 | 32,768 |
| modal/spatial FFT | real-side samples in the actual batch | 32,768 (`SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`) | 131,072 |
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
(`measure` by default, as on `numba_cpu`); `estimate` skips the measured planning for
short runs and tests. Answers at different thread counts may differ by rounding; solprop
holds them to its rounding bounds, not to identical bits
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).
Contiguous modal and Raman FFTW plans default to measured planning at one thread and
estimate planning at wider widths; the explicit effort overrides both, including
long Raman rows and partial chunks. Measured planning can add substantial construction
time even when the warmed propagation is faster. Use `SOLPROP_FFTW_PLAN=estimate`
when construction dominates a short run. On macOS with NEON FFTW, serial and long
Raman rows use Kokkos-FFT, while parallel power-of-two rows of at most 16,384 samples
retain vDSP. The [build guide](building-native.md) describes the coherent dependency
recipe and the fallback for a library without NEON codelets.

`OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS` do **not** size the Kokkos pool — the backend
initialises Kokkos with an explicit thread count, which takes precedence; set
`SOLPROP_NATIVE_THREADS` instead. (Kokkos prints a warning at start-up if
`OMP_NUM_THREADS` asks for fewer threads than it was given.)

**The default cap is every logical CPU.** Automatic sizing avoids assigning all of
those threads to small operations. On a shared machine or when several solprop
processes run side by side, set `SOLPROP_THREADS` (which holds `numba_cpu` too) or
`SOLPROP_NATIVE_THREADS` to each process's CPU budget. The operation policy stays within
that budget.

### OpenMP placement and waiting

The native CPU backend runs on OpenMP, whose standard variables apply:

- `OMP_PROC_BIND` and `OMP_PLACES` pin threads to cores, which avoids threads migrating
  between cores and sockets: `OMP_PROC_BIND=spread OMP_PLACES=cores` is a good default
  for one run on a multi-socket machine. Kokkos prints a notice when `OMP_PROC_BIND` is
  unset; `OMP_PROC_BIND=false` silences it without pinning. With a build whose FFTW
  uses pthreads rather than OpenMP, prefer `OMP_PROC_BIND=false`.
- `OMP_WAIT_POLICY` decides what idle threads do between parallel regions. `active`
  keeps them spinning, which helps the many short kernels of small problems on a
  dedicated machine; `passive` lets them sleep, which is much better on a shared or busy
  machine, when a threaded BLAS needs the same cores, or with several processes per
  node — an idle pool that spins can cost a large factor on a machine that is also doing
  other work. LLVM's OpenMP runtime (used on macOS) also honours `KMP_BLOCKTIME`, the
  time a thread spins before sleeping; do not set it for GNU OpenMP.

## BLAS and vendor libraries

The BLAS matters for the dense modal transforms of multimode capillaries, coherent walls
and the general modal route. A BLAS with its own thread pool beside the native backend's
oversubscribes the cores: with a pthreads OpenBLAS at the full core count, the modal and
wall cases ran 10 to 75 times slower than necessary. The Linux distribution build of the
native backend therefore carries an OpenBLAS built with OpenMP, which shares the
backend's runtime. At start-up the backend fits a host OpenBLAS to its own pool unless
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` or `OMP_NUM_THREADS` is set: a pthreads build
is held to one thread, an OpenMP build capped at the Kokkos thread count. `numba_cpu`
sizes its BLAS pool through threadpoolctl. Otherwise a BLAS's own variables apply:

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
oversubscribe the machine. On macOS, `solprop_native` must be imported before anything
imports `pyfftw`, and raises an error otherwise. solprop imports `pyfftw` only when a
`numba_cpu` backend is built, so this matters only for a script that imports `pyfftw`
itself or runs `numba_cpu` before the native backend in the same process: put
`import solprop_native` at the top of such a script. [Building the native backend](building-native.md#openmp-runtimes-and-pyfftw) explains
how to build pyFFTW against the same FFTW when a binary wheel brings its own runtime.

## Recipes

**One large run on a workstation.** Give the run every performance core, pin it, and let
idle threads sleep if you are using the machine for anything else:

```sh
SOLPROP_NATIVE_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=passive \
    python my_run.py
```

Without the native backend, `numba_cpu` sizes itself; to force a larger problem onto more threads,
for example `SOLPROP_NUMBA_THREADS=8 SOLPROP_FFTW_THREADS=4 python my_run.py`.

**How far a single run scales.** A single-mode fibre on a few thousand time samples
gains little beyond two to four threads: its transforms are short and each step is
mostly synchronisation. Multimode capillaries and the general modal route, whose
products and node loops are large, keep gaining to eight to eighteen threads. Free-space
runs with large transverse grids scale furthest.

**A scan of many runs.** For a scan over pressure, energy or noise seeds, many
single-threaded workers almost always beat one many-threaded run: each worker keeps its
own caches and nothing waits for synchronisation. solprop's
{class}`~solprop.driver.executors.pool.Pool` executor does this for you: every worker is a
newly spawned Python process started with **all** of `NUMBA_NUM_THREADS`,
`SOLPROP_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` set to its `threads` (one by default) and
`OMP_WAIT_POLICY=passive`, so every pool of either backend is sized within that budget,
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
[scans and batches](../workflows/scans.md). At one thread per worker the choice of backend
is less clear-cut than for a single large run: single-threaded `numba_cpu` can match or
beat the single-threaded native backend on some problems and machines, so time one point of the scan
on both (`backend="numba_cpu"` and `backend="native"`) before a large campaign.

If you manage the parallelism yourself — a job array, GNU `parallel`, several scripts
started by hand — each process needs every pool held to one thread explicitly, and the
serial kernel regime selected, before Python starts:

```sh
NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue SOLPROP_NUMBA_REGIME=small \
SOLPROP_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 OMP_WAIT_POLICY=passive OMP_PROC_BIND=false \
    python point.py
```

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
faster on the CPU: pass `backend="numba_cpu"` for them. Run independent problems in
separate processes to use several devices. See [The native backend](native.md).

**Memory.** Large multimode and general modal problems can be limited by memory before
threads; `SOLPROP_MEMORY_BUDGET` and the admission check are described in
[Execution resources](resources.md#memory).

## Checking what was used

Every backend logs the pools it chose at INFO level, and records them in the run's
backend record, which is saved with the output:

```python
import solprop
from solprop.driver import prop_capillary

solprop.configure_logging(
    "INFO"
)  # prints "numba_cpu: small regime, 1 kernel threads, ..."
run = prop_capillary(
    125e-6,
    1e-3,
    "He",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(200e-9, 2e-6),
    trange=100e-15,
    fwhm=10e-15,
    energy=1e-9,
    n_save=2,
    progress="none",
    backend="numba_cpu",
)
settings = run.metadata.backend.settings
print({key: settings[key] for key in ("regime", "numba_threads", "fftw_threads")})
```

For the native backend the same record carries `runtime_threads`, `thread_count`,
`fftw_threads`, `raman_fft_threads`, `blas_library` and the build profile; `python -m
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
