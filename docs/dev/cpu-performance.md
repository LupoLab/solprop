# CPU performance: what was tried

This page records the CPU performance work on the compiled backend. It lists what was
adopted, what was measured and rejected, and what remains open. The aim is that nobody
repeats an experiment without knowing its result. Each entry gives the result in one or
two lines and, where they are summarised, links to its measurements in
[benchmarks](benchmarks.md#indicative-results) or to
[SEP 29](seps/sep-0029-compiled-backend-distribution.md). The open items are in the
[roadmap](roadmap.md), under "Performance and backends".

Timings follow the [benchmark rules](benchmarks.md#rules). They are short, bounded
windows: fresh processes, same machine, interleaved arms. Results are compared within
SEP 25's rounding bounds, never bitwise. A ratio "against `numba_cpu`" is that backend's
time divided by the native backend's: above 1, the native backend is faster.

## Build and libraries

**Adopted**
- **One OpenMP runtime for the BLAS** (Linux distribution build). OpenBLAS is built with
  OpenMP by the extension's own compiler, so it shares Kokkos's runtime. A bundled
  pthreads OpenBLAS kept a second pool beside Kokkos's, and oversubscription made the
  coupled-wall and modal capillary cases 10 to 75 times slower on a 32-core AVX-512
  x86-64 server, and similarly on Windows and aarch64. With one runtime the backend is 2
  to 7 times faster than `numba_cpu` on those cases at full width, and level with it at
  one thread ([SEP 29](seps/sep-0029-compiled-backend-distribution.md)).
  - At start-up the backend fits a host OpenBLAS to its own pool, unless the user sets
    `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` or `OMP_NUM_THREADS`.
    - A pthreads OpenBLAS is held to one thread. This covers a local build against a
      distribution's library, and a Windows build that falls back to
      `scipy-openblas32`.
    - An OpenMP OpenBLAS is capped at the Kokkos thread count. Left alone, it sizes itself
      to every logical CPU.
- **Accelerate on macOS.** It matched or beat both OpenBLAS variants and native GEMM,
  except at one thread on the TE/TM wall, where it is the fastest option.
- **x86-64-v3 distribution baseline.** An x86-64-v2 build's median cost against
  x86-64-v3 was within about 2 % on a 32-core AVX-512 server (six pairs, seven
  configurations) and 4 % on an 18-core-per-socket AVX-512 server, on the benchmark core.
  FFTW and OpenBLAS choose wider kernels at run time either way. Linux aarch64 uses
  Neoverse N1; Apple silicon uses M1.
  - The exception is the general modal route: on the 18-core-per-socket server, v2 is
    23–24 % slower than v3 at one thread and 7–12 % slower at full width.
  - The wheels were first built for v2 and accepted this cost; they now take v3
    (SEP 29, phase 3), and v2 is a source-build option. Below v3, `backend="auto"`
    falls back to `numba_cpu` with a warning.
- **Release `-O3`, checked IPO, Kokkos loop vectorisation, no fast-math**, with
  `-ffp-contract=fast` honouring the protected-operation pragmas
  ([conventions](conventions.md)).
- **Windows: Kokkos's team barrier yields instead of sleeping.** Its sub-microsecond
  backoff sleeps were rounded up to the 15.6 ms timer tick. That stalled the Raman cases
  at the default width: the reduced comb-to-continuum took 81 s against 2.0 s patched.
  macOS was measured unaffected; the patch applies to Windows builds only.

**Tried and not adopted**
- **MKL in the wheels.** It has no aarch64 or macOS build, it is 100–200 MB, and its AMD
  performance is uncertain. It remains available to local Linux builds.
- **Kokkos Kernels' native GEMM as the wheels' BLAS.** It is equal to OpenBLAS at full
  width, but 15–80 % slower at one thread on small modal shapes and 2.5–4 times slower on
  the TE/TM wall. It is kept as the fallback.
- **FFTW with AVX2 or AVX-512 codelets** (32-core AVX-512 server, `FFTW_ESTIMATE`). The
  comb-to-continuum window was 16 % slower with AVX2 and 7 % slower with AVX-512 at one
  thread; Kerr and radial were 2–3 % faster. Wider codelets change the planner's
  choices, and that can lose. The distribution FFTW keeps SSE2 and AVX with run-time dispatch.
- **`-mcpu=native` instead of the M1 baseline** on Apple silicon: 0–8 %.
- **GCC instead of Clang** on Apple silicon: 9 % faster on the multimode cases, 3–18 %
  slower on the simple ones.
- **A separate FFTW threads DLL on Windows.** It cannot be linked, so the threads are
  built into `fftw3.dll`.
- **Thread settings for the Windows Raman stall.** `KMP_BLOCKTIME=0`, passive waiting,
  hot teams, full-width pools and physical cores only all left it at 61–73 s. The cause
  was the barrier sleep (above).

## Threads

**Adopted**
- **Per-operation thread widths from the operation's dimensions**, chosen at construction
  ([measurements](benchmarks.md#kokkos-cpu-operation-size-threading)). An operation below its
  threshold runs serially; a medium one runs on four threads; a large one runs at full
  width. The thresholds and their reasons are in `include/solprop_native/core.hpp`. The
  global thread setting is a cap, and explicitly set pools force their (clamped) widths.
  The small cases became 2.0 times faster than one thread (geometric mean) on an
  18-core arm64 laptop and on a two-socket, 36-core AVX-512 server.
  - Two interactions were found and corrected during qualification:
    - a narrow response stage followed by a full-width FFT on the Cartesian filament;
    - a serialised coupled-flow matrix on the scalar wall.
- **Measured FFTW wisdom only for serial transforms by default**
  ([measurements](benchmarks.md#threaded-fft-plan-selection)). FFTW's construction-time
  trials sometimes chose a slower *threaded* plan: on a 32-core AVX-512 server at 32
  threads, the 204 × 2048 complex transform ran 64 × 32 instead of 32 × 64 and the
  window was 10 % slower. Serial measured planning keeps the serial gain on arm64.
  `SOLPROP_FFTW_PLAN` still selects `estimate`, `measure` or `patient` explicitly.

**Tried and not adopted**
- **`FFTW_PATIENT` planning.** It adds seconds of construction and gave no measured
  throughput gain.

## Kernels

**Adopted**
- **Radial route: partial node chunks reach the interleaved real GEMM**
  ([measurements](benchmarks.md#radial-analytic-chunk-dispatch)). Without it production
  filaments ran a scalar contraction, which took 94 % of the source time. The route went
  from 2.5–6 times slower than `numba_cpu` at one thread to 0.91–0.95 times its time.
- **General modal: shared cubic monomials across the three response outputs**, with each
  output's accumulation order unchanged. The tensor region fell from about 0.25 s to
  0.09 s per 40 source evaluations
  ([measurements](benchmarks.md#multimode-cpu-step-costs)).
- **General modal: plans for the active tail rows only.** 2–5 %.
- **Short serial Raman rows (≤ 16384 samples) on measured FFTW instead of Accelerate.**
  Faster in both directions at every size tested. Larger and parallel Raman transforms
  stay on Accelerate.
- **Resident Adams history: no dense-output volumes when planes are landed on**
  ([measurements](benchmarks.md#resident-adams-storage)). Storage
  went from 29 to 18 field volumes. The projected 384² Kerr-beam peak went from 62.7 to
  48.2 GiB, against `numba_cpu`'s 34.8 GiB.

**Tried and not adopted**
- **A pooled allocator.** Allocation service in the warmed propagation window is below
  0.25 % of the time, and the native backend makes fewer native requests than `numba_cpu`.
- **Vector transcendental functions (Accelerate vForce) in the propagator.** They save
  about 0.1 ms per window. The installed `numba_cpu` does not use SVML either, so this
  did not explain the gap.
- **A direct FFTW execution adapter instead of Kokkos-FFT's.** Matched probes showed no
  wrapper overhead.

## Open

These are in the [roadmap](roadmap.md):
- **Windows re-measurement** of all the above on the single-runtime wheel.
- **The TE/TM wall's complex GEMM** at one thread.
- **FFTW planning policy per transform shape**: measure against estimate, and wider codelets.
- **Compiler-target comparisons with identical libraries**: v3, v4 or a processor-specific
  target against v2. On an 18-core AVX-512 server, v4 gave 7–14 % over v3 on the general
  route only.
- **Explicit SIMD** in the general-modal response and in synthesis and projection.
  - An assessment found enough custom arithmetic outside FFTW and BLAS for this to be a
    credible experiment on AVX-512 x86-64 and arm64.
  - It found no evidence yet that it beats the current loops.
  - First compare compiler targets, then prototype at most one or two kernels.
  - A broad conversion to SIMD types is not justified.
- **Selective run-time kernel dispatch** (SEP 29 §6.2).
- **Batched vector transcendentals for the plasma response.** About 38 % faster on the
  exp/log work itself, but it needs the dependent recurrence reorganised.
