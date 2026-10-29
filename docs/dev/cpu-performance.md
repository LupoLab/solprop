# CPU performance: what was tried

This page records the CPU performance work on the compiled backend. It lists what was
adopted, what was measured and rejected, and what remains open. The aim is that nobody
repeats an experiment without knowing its result. Each entry gives the result in one or
two lines and, where they are summarised, links to its measurements in
[benchmarks](benchmarks.md#indicative-results) or to
[SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md). The open items are in the
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
  one thread ([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md)).
  - At start-up the backend fits a host OpenBLAS to its own pool, unless the user sets
    `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` or `OMP_NUM_THREADS`.
    - A pthreads OpenBLAS is held to one thread. This covers a local build against a
      distribution's library, and a Windows build that falls back to
      `scipy-openblas32`.
    - An OpenMP OpenBLAS is capped at the Kokkos thread count. Left alone, it sizes itself
      to every logical CPU.
- **Accelerate on macOS.** It matched or beat both OpenBLAS variants and native GEMM,
  except at one thread on the TE/TM wall, where it is the fastest option.
- **NEON FFTW on both Arm wheels**, in both precisions, with the Arm cycle counter.
  On Apple silicon it keeps the M1 baseline and deployment target, and FFTW and Kokkos
  share one OpenMP runtime. This enables FFTW's explicit codelets; compiler
  vectorisation alone does not select them
  ([build recipe](../guide/performance/building-native.md)).
  - Linux aarch64 (the manylinux wheel, Neoverse N1 baseline), measured on 16-core
    Google Axion (Neoverse V2) and Ampere Altra (Neoverse N1) machines. FFTW's own
    benchmark, 1 thread: 1.16–1.18× faster in double precision and 1.9× in single
    (geometric means over 22 one- and two-dimensional problems); at 16 threads
    1.07× and 1.2×.
  - solprop A/B, same wheels otherwise, five FFT-heavy cases with both steppers: no
    case slower beyond its round-to-round range, and a geometric-mean time of 0.95
    (V2) and 0.96 (N1) at 1 thread, 0.96 and 0.98 at 16 threads. The Raman and RK
    GNLSE cases gained most (8–14 %); radial filamentation is unchanged.
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
  - The thresholds switch on `__APPLE__` alone: macOS has its own stage and FFT starts,
    and every other build takes the x86-64 Linux values (Cascade Lake, Zen 3). Windows
    and linux-aarch64 therefore run unmeasured thresholds.
  - Two interactions were found and corrected during qualification:
    - a narrow response stage followed by a full-width FFT on the Cartesian filament;
    - a serialised coupled-flow matrix on the scalar wall.
- **A stable kernel team** ([measurements](benchmarks.md#stable-kernel-team)). GNU OpenMP
  ends surplus pool threads whenever a narrower region follows a wider one, so a
  four-thread kernel between full-width ones re-created twelve threads each time (about
  0.45 ms at 16 threads). A kernel pool with a full-width instance now gives it to its
  medium-width kernels; FFT plans keep four threads, as FFTW's loops keep the whole team.
  `h2_raman_averaged` became 1.9-2.1 times faster at 16 threads (faster than one thread
  again) and `argon_blueshift` 1.2-1.4 times, on a Xeon Gold 6240 and an AMD EPYC 7763.
- **Measured FFTW wisdom only for serial transforms by default**
  ([measurements](benchmarks.md#threaded-fft-plan-selection)). FFTW's construction-time
  trials sometimes chose a slower *threaded* plan: on a 32-core AVX-512 server at 32
  threads, the 204 × 2048 complex transform ran 64 × 32 instead of 32 × 64 and the
  window was 10 % slower. Serial measured planning keeps the serial gain on arm64.
  `SOLPROP_FFTW_PLAN` still selects `estimate`, `measure` or `patient` explicitly.
  - One policy function decides every seed, by shape and width: rows as above, free
    space's transverse planes measured at any width. Free space's temporal pair had the
    opposite rule (measured only when threaded) and now follows the rows'. All seeding
    happens before the first plan is made; the plasma streams' plans were made earlier
    and stayed at estimate.

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
- **Raman provider selection with NEON FFTW.** Serial and long rows use Kokkos-FFT;
  short parallel power-of-two rows (≤ 16384 samples) keep vDSP. A non-NEON local
  FFTW retains the previous short-serial FFTW rule. Every contiguous FFTW fallback,
  including full and tail batches, honours the requested planning effort. Serial
  measured planning improves warmed execution but can substantially increase
  construction time; `SOLPROP_FFTW_PLAN=estimate` remains available for short runs.
- **Deferred ordinary DOPRI5 modal diagnostics.** Only the last trial stage projects
  the embedded transverse rule, and accepted-step scoring transforms it. Fine-only
  stages avoid stacked coarse GEMM work. Held field-plasma streams follow the same
  schedule; other methods and routes retain their existing consumers. Accepted
  values and peaks survive rejected trials, and existing scratch is reused without
  a new field allocation. The [lightweight measurements](benchmarks.md#native-cpu-fft-and-diagnostic-scheduling)
  record the affected windows and their limits.
- **The exponential Adams passes as vector lanes on x86-64**
  ([measurements](benchmarks.md#the-adams-passes-on-x86-64-lanes)). A host block of 32
  elements is a structure of arrays and every step of the predict and correct passes a
  loop over its lanes; the strict parts keep each element's operations and rounding.
  The passes are 2–2.7 times faster on the AMD EPYC 7763 at one thread, single-mode
  capillary windows 30–37 % faster. What kept GCC's loops scalar, worth knowing for any
  lane kernel: a strict helper called from a contracting kernel is an out-of-line call
  (GCC will not inline across `optimize` attributes); a selection whose kept value
  depends on a lane flag becomes a conditional store, which GCC if-converts only with
  masked stores (AVX-512, not AVX2 under its Zen tuning), so select by bits; an ordered
  comparison taken on some lanes only "could trap" and blocks if-conversion, so use the
  quiet `std::isless`-family; AVX2 has no int64-to-double conversion; and under LTO or
  the `optimize` attribute `-fopt-info` reports nothing, so check a kernel with a copy
  compiled without them.
- **Resident Adams history: no dense-output volumes when planes are landed on**
  ([measurements](benchmarks.md#resident-adams-storage)). Storage
  went from 29 to 18 field volumes. The projected 384² Kerr-beam peak went from 62.7 to
  48.2 GiB, against `numba_cpu`'s 34.8 GiB.

- **Radial route: square analytic maps synthesise the retained spectrum**
  ([measurements](benchmarks.md#retained-radial-synthesis-and-raman-modal-products)).
  The product covers the coarse bins rather than the padded analytic signal; the
  radial filament took 0.93 of its time at one thread and 0.95 at 16 on an 18-core Xeon,
  and 0.87 at one thread on an AMD EPYC 7763.
- **Raman: convolve the modal products, not the node drives**, where they are fewer
  ([measurements](benchmarks.md#retained-radial-synthesis-and-raman-modal-products)).
  Three-mode CTC: 0.88 at 16 threads on the Xeon; 0.83 at one thread and 0.92 at 16 on
  the EPYC.

**Tried and not adopted**
- **Skinny products off the vendor BLAS** (the native-order kernel for small left
  matrices; plain loops for one mode). Both lost to the vendor library's threaded GEMM
  on argon and CTC ([measurements](benchmarks.md#retained-radial-synthesis-and-raman-modal-products)).
- **The CPU experiment's split Cartesian FFT schedule.** The 384² Kerr beam's transverse
  pair costs about 0.18 s an evaluation at 16 threads (roughly 50 GFLOP/s), a small
  share of its step; the experiment's slow direct plans do not occur here.
- **The CPU experiment's rate-interpolation blocks.** The band column already gathers
  each block's live samples into one vector-maths pass, which subsumes its zero-block
  exit and half-active scalar path.
- **A pooled allocator.** Allocation service in the warmed propagation window is below
  0.25 % of the time, and the native backend makes fewer native requests than `numba_cpu`.
- **Vector transcendental functions (Accelerate vForce) in the propagator.** They save
  about 0.1 ms per window. The installed `numba_cpu` does not use SVML either, so this
  did not explain the gap.
- **A direct FFTW execution adapter instead of Kokkos-FFT's.** Matched probes showed no
  wrapper overhead.
- **Complete vDSP removal.** Even with NEON codelets, short parallel FFTW plans
  regress former vDSP shapes. Keep the compact construction-time rule above.
- **Tap-first CPU band filtering.** It preserves tap order and passed the boundary,
  support-window and high-dynamic-range checks. It improved the x86 windows but
  repeatedly slowed the Apple-silicon argon window by 6–9%; an architecture-specific
  fork was not justified. The original traversal remains, with stronger tests.
- **An 8×32 Cartesian crossing tile and omitted unused register loads.** Three
  alternating rounds on the 192² Kerr window reduced the initial apparent gain to
  about 2%, below the complexity threshold. The 32×16 traversal remains.
- **A forced one-thread Accelerate BLAS limit.** An initial RDW gain disappeared
  over three alternating rounds on RDW, radial 256, two wall cases and general modal:
  the geometric mean ratio was 1.000. Existing process-start settings remain.

## Open

These are in the [roadmap](roadmap.md):
- **Windows re-measurement** of all the above on the single-runtime wheel.
- **Thread thresholds on Windows and linux-aarch64**, which take the x86-64 Linux values
  unmeasured.
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
- **Value-safe floating-point options** (`-fno-math-errno`, `-fno-trapping-math`), a
  subnormal count on long runs, and reduced precision: the roadmap's item on
  floating-point options.
- **Vector maths in the remaining per-element kernels** (ADR-059): the trapezoid and
  envelope plasma columns, whose dependent recurrences need a batched rate pass first,
  the Adams corrector and dense observation, and the separable stage propagators; and
  SLEEF on Linux aarch64 and AVX-512 against AVX2, unmeasured end to end. The roadmap's
  item on vector maths.
