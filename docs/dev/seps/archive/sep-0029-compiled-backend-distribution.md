# SEP 29 — The native backend in solprop, distributed as binaries

| | |
|---|---|
| **SEP** | 29 |
| **Title** | The native backend in solprop, distributed as binaries |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-20); Opus 5.5 (2026-10-22, revision) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-20 |
| **Resolution** | Accepted 2026-10-22 by John Travers, after phase 0, the revision of Part I, its AI re-review (R15–R18) and the responses in Part III. Phase 1 (one repository) is next; Q7 is settled in phase 3. Final 2026-10-24 by John Travers after phase 3, before the first release, with phase 4 (the release and its evidence) moved to the roadmap (Part II, "Closing"). |
| **Requires** | [SEP 28](sep-0028-free-space-evaluation.md) (the measurements that prompted this); [SEP 25](sep-0025-rounding-level-reproducibility.md) (rounding-level agreement between backends and builds) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | John Travers with Opus 5.5 (AI); phase 1 on `feature/sep-0029-one-repository`, phase 2 on `feature/sep-0029-parity`, phase 3 on `feature/sep-0029-wheels` and then on `main` of the public repository |
| **Manual pages** | [Installation](../../../introduction/installation.md); [The native backend](../../../guide/performance/native.md); [Building the native backend](../../../guide/performance/building-native.md); [Releasing and the native wheels](../../releasing.md) |

## Part I — Proposal

*Revised on 2026-10-22 after phase 0 and its consolidation (Part II), at the
maintainer's direction. The revision brings the native backend into solprop's repository
as a second distribution, publishes it as binaries, and moves the retirement of
`numba_cpu` out of this SEP into a later decision that the wheels' use will inform. The
response to the review (Part III) records what changed for each finding.*

### 1. Abstract

Bring the Kokkos backend into solprop's repository as `solprop_native`, released with
solprop as a second distribution (`solprop-native`), and publish it as prebuilt CPU
wheels for Linux (x86-64 and aarch64), macOS (arm64) and Windows (x64), so that
`pip install solprop[native]` gives a compiled backend with sound default performance
and no environment variables. The pure-Python `solprop` distribution stays installable
everywhere, with `numpy_ref` and `numba_cpu` unchanged; `backend="auto"` prefers the
native backend wherever it is installed. Device builds (CUDA, HIP) come from the same
source tree. Before the release, the native backend gains the two capabilities
only `numba_cpu` has, and its own versions of the Numba-compiled helpers every backend
uses. Retiring `numba_cpu` is not part of this SEP: it is a separate decision,
taken later on evidence from the wheels in use. Success is judged by platform coverage,
the default performance a new user meets, installations that work alongside the
scientific Python stack, and one repository instead of two.

### 2. Motivation and scope

**Why distribute the native backend.** On the machines measured, the native backend is
the faster compiled backend wherever threads pay, and, with its BLAS on the same OpenMP
runtime as Kokkos, it needs no settings to get there (Part II, "Phase 0 outcome and
consolidation"):

- **Speed.** At the default width the Linux distribution wheel is 1.3–2.7 times faster
  than `numba_cpu` on the modal cases and 5.4 times on the general modal route of a
  36-core Xeon; on Apple silicon it is level or ahead in every core case at the default
  width (1.95 times on the 384² Kerr beam, 1.7–3.0 times on the radial filament), and at
  one thread within 15 % of `numba_cpu` on every core case, ahead on some. At one thread
  on the Xeon, `numba_cpu` still leads on general modal (the native backend takes 1.27
  times its time, mostly the v2 floor) and on the reduced comb-to-continuum case (1.05–1.08
  times).
- **Fragility of `numba_cpu` on Linux.** A `numba_cpu` run holds several independent
  OpenMP runtimes in one process (Numba's threading layer, the copy pyFFTW's wheels
  bundle, and Kokkos's when present). Idle spinning, binding and import order each cost
  a factor of two to ten in SEP 28's measurements, curable only by settings a new user
  will not know to make. solprop now sets passive waiting for those runtimes, but the
  native backend avoids the problem by construction.
- **Every non-CPU target already needs it.** CUDA and HIP execution exist only through
  the native backend. With CPU wheels as well, the native code is the common route for
  every platform with a build, not an optional extra.

**Why one repository.** solprop and solprop-kokkos are two repositories coupled by a
kernel schema version. Every feature crosses both, with paired branches, paired merges,
sibling checkouts (`SOLPROP_KOKKOS_SOLPROP_SOURCE`) and two CI configurations; gate
failures often come from the two drifting apart. One repository gives atomic changes,
one version and one CI, while two distributions keep what the split is good for: a pure
package that installs anywhere.

**In scope:**
- moving solprop-kokkos into solprop's repository as the `solprop_native` package of a
  second distribution, `solprop-native`, released with solprop at the same version,
  before the repository is open-sourced;
- native parity: every capability `numba_cpu` has that the native backend lacks, and
  native versions of the Numba-compiled helpers shared by every backend (§6.6), so that
  the native backend can be the default everywhere it is installed without sending users
  back to `numba_cpu`;
- reproducible build recipes and CI for CPU wheels: Linux x86-64 (x86-64-v2) and
  aarch64 (Neoverse N1) manylinux wheels, macOS arm64 with a declared deployment target,
  Windows x64;
- safe loading on every shipped platform, and the `solprop[native]` extra;
- warnings for harmful settings that can be detected;
- documentation of source builds, including the device builds, from the same tree.

**Out of scope:**
- the freeze, deprecation and removal of `numba_cpu` (§6.3), now a later SEP;
- device (CUDA, HIP) wheels, which stay source builds (Q6);
- macOS x86-64 and Windows on Arm;
- changes to the numerics; `numpy_ref` stays the independent oracle.

### 3. Context

- The backend seam and registry ([backends](../../backends.md),
  [architecture](../../architecture.md) §3): `numpy_ref` (the oracle), `numba_cpu` and the
  native backend behind one kernel contract; `backend="auto"` picks the native backend
  when installed, else `numba_cpu` (ADR-029).
- The native build profiles LOCAL, DISTRIBUTION and REFERENCE
  ([ADR-034](../../decisions.md), amended 2026-10-21): DISTRIBUTION compiles for
  x86-64-v2, Neoverse N1 or Apple M1 and links a BLAS that shares the extension's
  threads (Accelerate on macOS; on Linux OpenBLAS built with OpenMP on the extension's
  own runtime). The package fits a host OpenBLAS's threads to the Kokkos pool at import.
  An import-time check refuses a processor below the baseline on Linux.
- Phase 0 (Part II): Windows x64 builds with Clang on the MSVC ABI and passes the suite;
  Linux aarch64 builds and passes; a single OpenMP runtime for Kokkos, FFTW and BLAS is
  the measured best design on Linux and Windows; the x86-64-v2 floor costs at most 2–4 %
  on the core cases and 23–24 % on the general modal route at one thread on a Xeon.
- The standard benchmark suite ([SEP 30](sep-0030-standard-benchmark-suite.md))
  measures the core cases on both backends with matched thread budgets.
- solprop and solprop-kokkos are two repositories and two distributions today, coupled
  by the kernel schema version, which each build checks at import.

### 4. Theory

No physics changes. Comparisons between backends, builds and thread counts follow SEP
25's distinction: matched discrete computations agree to rounding-level bounds;
independently adaptive runs agree within an accuracy envelope against a tighter or
stored converged reference, not step for step. Neither is bitwise.

### 5. Proposed interface

- **Install.** `pip install solprop` installs the pure-Python package, with `numpy_ref`
  and `numba_cpu`, everywhere. `pip install solprop[native]` adds `solprop-native`,
  pinned to the same version, wherever a wheel exists. Only wheels of `solprop-native`
  are published to the package index, with no source distribution, so where no wheel
  matches, the extra fails to resolve instead of building from source (Q7). Source
  builds, including CUDA and HIP, are made from the repository or a release's source
  archive, with documented build options.
- **Import name.** `solprop_native` replaces `solprop_kokkos`. Nothing has been published,
  so no alias is kept.
- **Selection.** `backend="auto"` picks the native backend when the `solprop_native`
  package is installed, and `numba_cpu` only when it is absent. Once the native package
  is present, any failure raises instead of selecting another backend: a processor below
  its baseline, a failed native load, a kernel-schema mismatch. A processor below the
  baseline raises a clear error naming the missing features, before any native code
  runs. An explicit `backend="native"` (the current `"kokkos"` name, renamed) never falls
  back. Run metadata records the selected backend and its build, not just the request.
- **Baselines.** x86-64 wheels use x86-64-v2; source builders can select x86-64-v3 or
  higher with `SOLPROP_NATIVE_CPU_BASELINE` (§6.2).
- **Threads.** The native backend sizes its pools from the process's CPU grant and the
  operation's size, and fits its BLAS to its pool at import; an explicit user setting
  always wins. Warnings, at backend construction, only for settings that are detectably
  harmful (R8): an explicit thread count above the CPUs the process may use, and
  `OMP_WAIT_POLICY=active` under a scan's workers.
- **`numba_cpu`.** Unchanged and fully supported throughout this SEP.

### 6. Implementation strategies

#### 6.1 Windows

Proved in phase 0 (Part II): Clang with the GNU driver on the MSVC ABI (MSVC STL and CRT,
release `/MD`), Kokkos OpenMP on upstream LLVM `libomp` (Microsoft's `libomp140` is not
redistributable), FFTW and OpenBLAS built static on the same `libomp`, repaired with
delvewheel. A Kokkos `Threads` build without OpenMP also works and is kept as the
fallback. Kokkos's team-barrier backoff is patched to yield on Windows, where
sub-microsecond sleeps last a 15.6 ms timer tick. Remaining for the release recipe:
the coexistence tests of R14 and a minimum desktop Windows version (R6).

#### 6.2 The conservative x86-64 build

Distribute one x86-64-v2 build (SSE4.2, no AVX2 requirement), with BLAS. Keep
`SOLPROP_NATIVE_CPU_BASELINE=x86-64-v3` as a source-build option. On the core cases the
v2 floor costs at most 2 % on Zen 5 and 4 % on a Xeon; on the general modal route it
costs 23–24 % at one thread and 7–12 % at full width on the Xeon (7 % on Zen 5), a cost
the maintainer accepts for the floor. Bundled BLAS and FFTW select CPU-specific kernels
at run time, provided their dispatch and fallback are safe on v2 (qualified under an
instruction emulator in phase 0, and again for the OpenMP OpenBLAS).

**Future option: selective kernel dispatch.** If profiling identifies a hot kernel
that benefits from explicit SIMD or another instruction-set specialization, one
wheel could contain a v2 implementation and separately compiled AVX2 or AVX-512
variants of that kernel. A v2-safe selector would check both CPU features and OS
register support before calling a variant. Import, initialization, common code
and the fallback would retain the v2 floor. This need not duplicate the whole
backend or require separate wheels. The choice would be justified by measured
workloads; wider instructions alone are not a performance guarantee.

Kokkos SIMD can express the vector operations portably, but does not supply this
runtime selector automatically. Its
[SIMD programming guide](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/SIMD.html)
describes selection for the compiled architecture; the
[Kokkos 5.2.1 implementation](https://github.com/kokkos/kokkos/blob/5.2.1/simd/src/Kokkos_SIMD.hpp)
chooses the native SIMD ABI with compile-time architecture macros. A future design
would therefore need explicit variant isolation and dispatch, an ordinary
pointer/scalar interface that does not expose ISA-dependent SIMD types, and
protection against inlining specialized instructions into baseline code. It would
record the selected variant, qualify every path against the numerical contract,
and test the installed wheel with higher instruction sets unavailable. This is a
recorded option for later optimization, not a requirement for the first v2 wheels.

#### 6.3 `numba_cpu` (moved out of this SEP)

This SEP keeps `numba_cpu` unchanged. Its freeze, deprecation and removal become a later
SEP, proposed once the wheels have been in use long enough to show their coverage and
any remaining gaps. With native parity (§6.6) done in this SEP, that later
decision concerns the removal itself:
- the freeze point, feedback period and rollback of R11;
- the restructuring of the test suite's cross-backend tests;
- the removal of `numba` and `pyfftw` as dependencies.

#### 6.4 One repository, two distributions

Resolved (Q3): option (b) of the original proposal, one repository with two
distributions.

- **`solprop`**, pure Python, built as today, with `numpy_ref` and `numba_cpu`.
- **`solprop-native`**, the compiled package `solprop_native`, built with
  scikit-build-core from a subdirectory of the same repository (for example
  `native/`), with its C++ sources, native tests and build presets.
- **Versions.** One version for both, released together; `solprop[native]` pins
  `solprop-native==` that version. The kernel schema check stays (R12): editable
  checkouts, local device builds and stale extensions can still combine mismatched
  halves, and the check catches them at import.
- **Device builds** are source builds of the same distribution with the device options;
  they install as `solprop-native` with a different build identity, so a CPU wheel and
  a device build are alternative installations of one package in an environment (R7).
  Requesting the CPU from a device build stays a documented option of that build.
- **History.** Both repositories are single-commit histories today, so the move is a
  directory import with the native repository's history preserved by merging it under
  `native/`.
- **The oracle stays independent** (R12): `numpy_ref` never calls into the native
  package.

#### 6.5 Release engineering

The phase 0 tests already found the main recipe gaps, recorded in Part II:

- **Linux** builds in a manylinux container (`manylinux_2_28`), so that the wheel's
  glibc and libstdc++ floor is the container's, not the build host's; OpenBLAS (OpenMP)
  and FFTW are built in the container and bundled by auditwheel.
- **macOS** sets an explicit deployment target and builds `libomp` and FFTW from source
  for it; Homebrew's copies require the build machine's macOS (27 for `libomp`, 26 for
  FFTW), so a wheel delocated from them would refuse older systems. The macOS wheel
  must also settle coexistence with pyFFTW's OpenMP runtime (§7, R15).
- **Windows** uses the phase 0 single-runtime recipe; delvewheel mangles the bundled
  DLL names so that they cannot collide with pyFFTW's, NumPy's or SciPy's (R14).
- **Free-threaded CPython** is not a target of the first release (R6).

Recipes run privately first, then in public CI once the combined repository is open source.

#### 6.6 Native parity

A capability audit (2026-10-22) compared the three backends' admission and refusal paths.
The native CPU backend refuses only two things `numba_cpu` runs. Everything else either
backend refuses, the other refuses too, through solprop's shared admission:
- more than two polarisation components;
- vector Raman with scalar-only terms;
- TDSE (reference-only by SEPs 12, 20 and 26);
- the `cf4` coupled-flow sections (reference-only until a use case needs them).

The work of this phase:

- **G1: the Chebyshev z table.** `solprop.translate.stage.require_device_ready` refuses a
  problem whose linear operator or response prefactor takes the general z table. That
  happens with:
  - a `SpecifiedMode` with moving dispersion;
  - three or more gas species on independent density paths, on the modal and radial
    routes;
  - a multimode taper with `taper_coupling=False`;
  - a kinked parameter, or modes that disagree about their parameters;
  - a forced table.

  Because `backend="auto"` chooses the native backend and never falls back, these runs fail
  unless the user names `numba_cpu`. Native CPU support evaluates the table on the host, as
  `numba_cpu` does, and uploads the per-stage arrays. That needs new native bindings but no
  kernel-schema change. Device builds may keep refusing it if the table's memory (the
  state's size times the degree) is prohibitive there. Estimated at three to five days.
- **G2: Adams history export.** The driver refuses snapshots with the default `Multistep()`
  on the native backend, because its resident Adams history cannot be exported. Resuming a
  native Adams run therefore restarts the multistep history. That is correct to the
  accuracy contract, since resume is not bitwise, but it happens without notice. Native
  support exports and imports the history rows in the snapshot format, and removes the
  refusal. Until then, resuming warns. Estimated at one to three days.
- **R2: the shared Numba helpers.** Three Numba-compiled kernels run on every backend's
  path:
  - the Chebyshev evaluations `_clenshaw` and `_clenshaw_batch` (`translate/chebyshev.py`),
    in translation and behind every stage's scalars;
  - the Adams basis rows `_basis` (`stepper/multistep.py`).

  **Who chooses the implementation (R16).** Never a process-wide switch, and never
  merely because the native package is installed: that would make `numpy_ref` run the
  native code it exists to check, and one simulation could change another's arithmetic.
  - **Translation is shared.** It runs before any backend is constructed and builds the
    problem every backend receives, so it keeps a single reference implementation: the
    plain-Python forms, accelerated by Numba where it is installed. The oracle's checks
    cover it.
  - **Per-stage and per-step evaluations belong to the backend that runs the stage.**
    These are the stage scalars behind `_clenshaw_batch` and the Adams basis behind
    `_basis`. The shared functions take the evaluator from their caller instead of
    choosing one:
    - `numpy_ref` passes the plain-Python forms;
    - `numba_cpu` keeps the Numba ones;
    - the native backend passes C++ versions from `solprop_native`.

  The C++ versions keep the strict floating-point protections SEP 25 relies on: no
  contraction or reassociation. A test asserts that with the native package installed,
  `numpy_ref` never reaches native code and native runs never reach the Numba
  dispatchers, including two backends in one process. If the change alters a protected
  driver hook or stage bundle, the kernel-schema rule applies as usual. This design is
  settled before G1 builds on it. Measure small problems, where the uncompiled forms
  cost six to thirty-six times more per call. Estimated at one to two days.
- **Documentation:** correct the user guide's statement of what the native backend runs,
  and the backend and driver texts that disagree with the code.

Afterwards the remaining differences between `numba_cpu` and the native backend are
tuning options only: `numba_cpu`'s FFTW planning effort and wisdom files, and the native
backend's process-wide thread settings. The later decision on `numba_cpu` (§6.3) then
concerns only its removal.

### 7. Verification and validation

- **Coverage.** Each wheel family passes the full suite, the slow lane and the
  native-selected lanes, installed from its repaired wheel on a clean machine without
  compilers, at the declared minimum OS (R6, R13). Unexpected native skips fail the lane.
- **Loading.** On each platform, a subprocess import on a processor below the baseline
  (an instruction emulator on x86-64) refuses cleanly before native code runs (R4).
- **Installation.** Plain `pip` and `uv` installs from a wheelhouse, with and without
  the extra, on supported platforms and on one without a wheel (R3).
- **Coexistence** on every platform family (R14, R15). The repaired wheel runs with the
  ordinary installed NumPy, SciPy, pyFFTW and Numba, in fresh processes, in both import
  orders, with interleaved short runs on the native backend and `numba_cpu`. Today's
  native loader refuses on macOS when pyFFTW was imported first, because pyFFTW's wheel
  carries its own OpenMP runtime. The release either shows that the wheel's own
  `libomp`, delocated under a distinct name, runs safely beside pyFFTW's, so that the
  guard can be removed, or documents the import-order rule as a macOS limitation. In the
  latter case, solprop's own import order and the loader's error message carry the rule.
  The guard is not removed without that evidence.
- **Performance.** The standard benchmark suite's core cases, adaptive Multistep and RK,
  at one thread and the production width, native against `numba_cpu` on the same
  machine with matched budgets, on each platform family (R9). The benchmark ledger
  records the baselines; a serious loss is reported per case, not averaged away.
- **Equivalence.** The existing cross-backend tests between `numpy_ref` and every build,
  under SEP 25's two kinds of bound (§4, R10).
- **Parity (§6.6).** The G1 triggers run on the native CPU backend against `numpy_ref`
  within those bounds, instead of being refused. A native Adams run snapshots and resumes
  with its history (R17):
  - once the history has formed, the native march snapshot is non-empty;
  - the restore does not take the restart path;
  - the restored rows, positions, slots, identities, order and controller state equal the
    stored ones exactly, since this is lossless state;
  - this holds for the compensated history, the periodic snapshot path and ordinary HDF5
    resume;
  - an incompatible history is refused explicitly.

  The propagation after the restore agrees with an uninterrupted run within the accuracy
  envelope; identical trajectories or work counts are not required. The
  native helpers of R2 agree with the plain-Python forms within the protected-arithmetic
  bounds, and the small-problem timings are recorded. A test asserts that no supported
  configuration directs a user to `numba_cpu`.

### 8. Documentation

Installation (the extra, platforms, baselines, source and device builds), the
performance guide (what a user need not set, the warnings), the backends page (the
native backend's name and selection), the architecture and structure pages (one
repository, two distributions), contributing (one checkout, one gate), the changelog.

### 9. Compatibility and impact

- `import solprop_kokkos` and `backend="kokkos"` are renamed (`solprop_native`,
  `"native"`); nothing is published, so no aliases are kept.
- `numba_cpu` is unchanged; feature SEPs keep their `numba_cpu` phase until a later SEP
  decides otherwise.
- Results agree with today's within SEP 25's bounds (§4).
- Contributors build one repository; C++ changes and their Python side land in one pull
  request.
- Users on platforms without a wheel keep `numba_cpu` and `numpy_ref`, and can build the
  native package from source.

### 10. Alternatives considered

- **One distribution carrying the extension** (option (a)): one wheel per platform for
  every Python-only change, an sdist that needs a C++ toolchain, and no install path on
  platforms without a wheel. Rejected for now.
- **Keep two repositories** (option (c)): keeps today's coordination cost.
- **Retire `numba_cpu` in this SEP**, as originally proposed: premature before the wheels
  have been used; the capability and dependency findings of R1 and R2 need their own
  decision.

### 11. Phases

0. **Private proving.** Done (Part II), with the consolidation of 2026-10-21/22.
   **Checkpoint**: the maintainer decides whether to proceed on this revision.
1. **One repository.**
   - Import the native repository under `native/`.
   - Rename to `solprop_native`, `solprop-native` and `"native"`.
   - One CI and one contributor workflow; the kernel schema check kept.
   - Done with no other code work in flight, because the renames touch every file of the
     native package.
2. **Native parity** (§6.6), in this order:
   - R2 and G2 in parallel: they touch different code;
   - then G1, which builds on R2's native Chebyshev evaluation;
   - with the documentation corrections.
3. **Open source and wheels.**
   - solprop is open-sourced as the one repository, so the native backend is public from
     the first release.
   - Reproducible recipes (§6.5), privately at first, then in public CI.
   - Safe loading on every platform, the `solprop[native]` extra, and the qualification of
     §7.
4. **Release and evidence.** Release both distributions, and collect evidence of coverage,
   default performance and coexistence from use.

A later SEP decides the future of `numba_cpu` on that evidence (§6.3).

### 12. Open questions

- **Q1.** Which Windows toolchain and libraries (§6.1)? *Resolved in phase 0: Clang on
  the MSVC ABI with one LLVM `libomp` for Kokkos, FFTW and OpenBLAS; Kokkos `Threads`
  as the fallback.*
- **Q2.** Fat wheel or two wheels for the x86-64 baselines (§6.2)? *Resolved on
  2026-10-21: one v2 wheel build, with v3 as a source-build option; see Part II.*
  *Revised in phase 3 (2026-10-23): one x86-64-v3 build, with v2 as a source-build
  option, and `backend="auto"` falling back to `numba_cpu` below it; see Part II,
  phase 3.*
- **Q3.** One package or two, and one repository (§6.4)? *Resolved on 2026-10-22: one
  repository, two distributions (`solprop`, `solprop-native`).*
- **Q4.** Is x86-64-v2 the right conservative baseline, or is v1 (no SSE4.2) needed for
  any population worth supporting? *Resolved on 2026-10-21: v2, with v3 available
  as a source-build option; see Part II.*
  *Revised in phase 3: x86-64-v3 (above).*
- **Q5.** Should the `numba_cpu` stopgap of §10 be done now, independently of this SEP?
  *Done: Linux `numba_cpu` sets a passive wait policy and keeps its CPU grant
  (2026-10-20).*
- **Q6.** Device wheels: a later SEP, or a phase here? *Later; device builds are source
  builds of `solprop-native` from the same repository (§6.4).*
- **Q7.** On a platform without a wheel, should `solprop[native]` fail to install or fall
  back to a source build? *Proposed, for phase 3 (R18): publish only `solprop-native`
  wheels, with no source distribution, to the index, so that the extra fails to resolve
  where no wheel matches. Document source builds from the repository or a release
  archive. Test the policy against the actual release artefacts. Confirmed when the
  release recipe is set up.* *Confirmed in phase 3 (2026-10-24): the release workflow
  publishes no `solprop-native` source distribution, and against the built wheelhouse
  `solprop[native]` fails to resolve on platforms without a wheel (Windows on Arm, musl
  Linux, Intel macOS, macOS before 14) while plain `solprop` installs; see Part II.*

### 13. References

1. ADR-034, "A distribution Kokkos build: a CPU baseline and the platform's BLAS",
   amended 2026-10-21 ([decisions](../../decisions.md)).
2. SEP 28 phase reports and the benchmark ledger entries "SEP 28 phase 0" and "SEP 28
   phases 1 and 2" ([benchmarks](../../benchmarks.md)).
3. [SEP 30](sep-0030-standard-benchmark-suite.md), the standard benchmark suite.
4. OpenMP Architecture Review Board, *OpenMP Application Programming Interface*,
   version 5.2 (2021), sections on `OMP_WAIT_POLICY`, `OMP_PROC_BIND` and `OMP_PLACES`.

## Part II — Implementation record

### Phase 0: conservative x86-64 floor — 2026-10-21

**Maintainer decision (Q2, Q4).** Use x86-64-v2 as the default for distributed
x86-64 wheels, with x86-64-v3 retained as a source-build option. The repeated
comparison below supports this choice. A v1 distribution build is not required.
The population evidence does not establish that v2-only machines are negligible: recent low-power clients
and retained research-cluster nodes remain relevant under the support scope in R5.

**Build and instruction qualification.** The existing
`SOLPROP_KOKKOS_CPU_BASELINE` option produced separate v2 and v3 distribution
wheels from solprop-kokkos `47c19b023134b6cd8f66094cc1271fb5c7126cad`, against
solprop `42e3d3a661892b6b56a7651c3da793fc1efef1c7`. Both are committed SEP 30
development revisions. GCC 13.3, Release, checked IPO and aggressive Kokkos loop
vectorisation were enabled, with no fast-math. The wheels contain byte-identical
FFTW, OpenMP and OpenBLAS libraries; the compiled backend extension differs with
the CPU baseline. They were repaired and installed as CPython 3.14
`manylinux_2_39_x86_64` wheels. This does not qualify the eventual older-glibc
release recipe.

Intel SDE 10.13.1, using `-nhm -chip_check_zcnt 1`, exposed SSE4.2 and POPCNT
while masking AVX and AVX2, including in subprocesses. A deliberately executed
AVX2 instruction was rejected. Under the same checks, the v2 wheel passed all
338 applicable Python backend tests; five tests skipped device dispatch or a
macOS-specific guard. The native suite passed 107 applicable tests across the
initial run and a targeted rerun, with one expected skip because v2 has no FMA
requirement. Both NumPy's and the backend's OpenBLAS selected Nehalem kernels.
Instruction checking covered child processes and loaded libraries, without
exclusions.

The first native run found a stale provider-name assertion: the scalar-envelope
path already uses interleaved real BLAS, but its test expected the older complex
BLAS label. The test-only correction in solprop-kokkos `1e37757` changes both
labels to `kokkosblas-tpl-interleaved-real`; its targeted SDE rerun passed. No
production code or numerical tolerance changed, and the installed wheel hashes
remained unchanged. The full repository gate, sanitizers and solprop-wide lanes
were not run for this qualification slice.

**Same-processor comparison protocol.** The maintainer requested a general
estimate of the baseline's cost within about twenty minutes. Four representative
SEP 30 quick-core workloads supply one fresh-process v2/v3 pair each: RDW, CtC
and radial filament with the default adaptive multistep method, plus the
fixed-RK4IP Kerr beam. Every measurement uses eighteen physical cores on an Intel
Xeon Gold 6240; Kerr is never run on one core. An exclusive allocation and
affinity to one socket isolate the measurements; the `blas1` policy holds BLAS
at one thread. The execution budget is twenty minutes, excluding scheduler wait.
The cases and their controllers stay unchanged, and the declared short windows
are never extrapolated to full runs.

The adaptive cases use the suite's existing reviewed envelopes, with tight RK
references at `rtol=1e-10`, refinement at `1e-9`, matched-method checks and reset
evidence. The radial methods share identical tight and coarse RK calculations:
the second qualification reuses the completed arrays after checking the numerical
configuration and artifact hashes, while its matched-method and candidate checks
run separately. At the maintainer's request, the Kerr NumPy reference is skipped;
Kerr remains in the timing selection with explicit `not-qualified` status. No
accuracy limit is invented or widened. Each case has only one measured pair:
there is no measured repeatability or statistical precision estimate, and these
spot checks do not establish the standard suite's adoption verdict or A/A band.

**Spot-check result.** All four pairs completed in 4 minutes 39 seconds. The
table gives warm window times, excluding process setup and the first window;
positive changes mean v2 took longer. RDW, CtC and radial filament use adaptive
multistep; Kerr uses fixed RK4IP. All eight samples passed the timing load guards,
imported the intended extension hashes and used eighteen Kokkos threads with
one BLAS thread. The two builds took identical accepted/rejected step counts
and source-evaluation counts in each pair.

| Case | Measured window | v2 / s | v3 / s | v2 time change | Accepted / rejected / source evaluations |
|---|---|---:|---:|---:|---:|
| RDW | 0.02–0.04 m | 0.1472 | 0.1415 | +4.0% | 21 / 0 / 21 |
| CtC | 0.00017–0.00034 m | 2.4413 | 2.4450 | −0.2% | 19 / 0 / 19 |
| Radial filament | 0–0.03 m, four planes | 7.9878 | 8.1511 | −2.0% | 227 / 2 / 230 |
| Kerr beam, 384 × 384 × 600 | Ten 0.5 µm steps after one untimed step | 30.0605 | 29.6636 | +1.3% | 10 / 0 / 40 |

Both builds passed the three adaptive accuracy checks against the same reference
receipts. Kerr retained its expected material-range warning and the requested
`not-qualified` reference status. No sample failed, exceeded its memory estimate,
or produced an unreviewed warning. The measured differences are all within about
four percent: this spot check finds no large v2 cost on these production-width
workloads, but a single pair cannot distinguish small ISA effects from timing
variation or establish a reliable ranking. The accepted v2 floor and optional
v3 build option remain the distribution direction.

These measurements do not substitute for performance on physical older hardware.
The comparisons with `numba_cpu` remain deferred until the ongoing Kokkos
performance work is ready. The default v2 release recipe, safe loading and
qualification of the final release wheels remain separate delivery work.
The emulator's controls and download were checked against
[Intel's SDE documentation](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
and its [official release](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html).

### Phase 0 follow-up: repeated Zen 5 comparison — 2026-10-21

**Scope and protocol.** The maintainer requested a broader v2/v3 comparison on
one on-demand AWS `c8a.8xlarge`: AMD EPYC 9R45 (Zen 5), 32 physical cores,
one hardware thread per core and 64 GiB RAM. Ubuntu 24.04 supplied glibc 2.39;
CPython 3.14.7 and the locked scientific package versions matched the earlier
campaign. The same frozen solprop and solprop-kokkos revisions and repaired
wheels described above were reused. The native baseline is the only intended
build difference. Both loaded OpenBLAS providers reported `SkylakeX` runtime
dispatch on this processor; neither wheel forces a Zen-specific BLAS kernel.

RDW, CtC and radial filament retain the adaptive-multistep windows above and run
at one and eighteen threads. Kerr retains its ten-step fixed-RK4IP window and
runs only at eighteen threads. Each of the seven configurations has six adjacent
fresh-process v2/v3 pairs. The first build alternates between rounds, giving
three pairs in each order; configuration order is shuffled after the first round.
Only one worker runs at a time. Affinity is one fixed physical core or the same
eighteen physical cores in one NUMA node, with BLAS held at one thread. Each
worker has its ordinary untimed preparation/first window and one timed window.

**Results.** All 42 pairs (84 measured windows) completed in 41 minutes
27 seconds of collection, within the fifty-minute work budget. All samples
passed the timing load guards. Imported extension identities, source revisions,
package versions, thread widths and paired resolved inputs were verified.
No failed sample, unreviewed warning or memory-estimate violation was recorded.

| Case | Threads | v2 median / s | v3 median / s | Median paired v2 change | Paired MAD / percentage points |
|---|---:|---:|---:|---:|---:|
| RDW | 1 | 0.1100 | 0.1098 | +0.63% | 1.02 |
| RDW | 18 | 0.0825 | 0.0815 | +1.37% | 2.45 |
| CtC | 1 | 2.1581 | 2.1460 | +2.09% | 2.05 |
| CtC | 18 | 1.3836 | 1.3740 | +0.67% | 0.18 |
| Radial filament | 1 | 58.0027 | 58.4410 | -0.96% | 0.79 |
| Radial filament | 18 | 4.7588 | 4.7985 | -0.90% | 1.82 |
| Kerr beam | 18 | 11.3665 | 11.3970 | +0.54% | 1.63 |

The change column is the median of the six paired values
`100 * (v2_time / v3_time - 1)`, so it need not equal the ratio of the two
separately listed time medians. Positive values mean v2 took longer. MAD is the
median absolute deviation of those paired changes, in percentage points; it is
dispersion, not a confidence interval. No sample was discarded. Individual paired
changes ranged from -7.47%
to +5.27% across the selection.

Actual work matched between builds and across all repeated windows: accepted /
rejected / source evaluations were 21 / 0 / 21 for RDW, 19 / 0 / 19 for CtC,
227 / 2 / 230 for radial filament and 10 / 0 / 40 for Kerr. Both thread widths
of every adaptive case passed the unchanged reference envelopes. Existing
reference arrays and receipts were hash-verified and reused; no new reference
propagation ran. Kerr remains `not-qualified`, with its reviewed material-range
warning and no NumPy reference, as requested.

**Interpretation and limits.** Every median paired difference is within
2.09% of zero. This sample finds no substantial v2 slowdown on the
selected Zen 5 workloads at either width. The small changes and observed spread
do not establish a consistent winner. These are repeated quick-mode windows on
one virtual machine, without an A/A control, calibration of short-window
repetitions, multiple independent instances or an adaptive-RK timing lane.
Warm times exclude process setup, the first window and separately recorded
observations. They neither predict full propagation time nor establish the
standard suite's adoption verdict. This follow-up adds no older-hardware
performance claim.

**Distribution decision and explanation.** Following these results, the maintainer
selected v2 as the default x86-64 wheel build and retained v3 as a source-build
option. Shipping both native variants is no longer required. This is a release
recipe decision for phase 1; this report does not change the existing build default.

The comparison changes the native extension's instruction baseline while retaining
byte-identical FFTW and OpenBLAS libraries, verified by their SHA-256 hashes.
The bundled OpenBLAS contains Haswell and SkylakeX kernels; disassembly confirms
256-bit and 512-bit vector instructions respectively. The same binary selected
`SkylakeX` in these Zen 5 samples and `Nehalem` in the earlier non-AVX emulation.
This is direct evidence of runtime selection independently of the extension's
baseline, consistent with the
[OpenBLAS distribution guidance](https://www.openmathlib.org/OpenBLAS/docs/distributing/).
The bundled FFTW identifies itself as `fftw-3.3.10-sse2-avx`: it includes SSE2
and AVX implementations, but not AVX2 or AVX-512 implementations. Its AVX selector
checks CPU features and OS register support with CPUID and XGETBV. FFTW documents
[runtime selection of enabled SIMD implementations](https://fftw.org/doc/Installation-on-Unix.html).
The propagation timings did not capture individual FFT plans or executed instruction
counts, so they do not establish which FFT codelets consumed the time.

One safely dispatched dependency library can therefore serve both extension builds;
a v2 compatibility floor does not restrict every executed instruction to v2.
The library's common code and fallback must meet that floor. Enabling additional
FFTW implementations is a separate optimization experiment, whose benefit is not
measured by this v2/v3 comparison. GCC's v2 and v3 levels both use generic tuning
unless another `-mtune` target is supplied; this was not a Zen 5-specific tuning comparison
([GCC x86 options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)).
These mechanisms explain why the baseline flag may have little overall effect.
They do not establish the runtime share of each kernel or prove that further
optimization is exhausted. Profiling would be needed to distinguish library work,
memory traffic, scalar arithmetic and threading costs in the remaining runtime.

**Verification and handoff.** The report change passed whitespace and spelling
checks, the SEP consistency tests and the Sphinx documentation build. No
production source or tolerance changed, and the successful SDE qualification
was not repeated. The full repository, slow and coverage gates remain outside
this documentation-only follow-up. The v2 release recipe, the v3 source-build
instructions and release-wheel qualification remain separate delivery work.

### Phase 0 follow-up: FFTW SIMD implementations on Zen 5 — 2026-10-21

**Question and controls.** A second on-demand AWS `c8a.8xlarge`, with the same
AMD EPYC 9R45, 32 physical cores, 64 GiB RAM and Ubuntu 24.04, tested whether
additional FFTW implementations help while the extension retains its v2 floor.
The frozen solprop and solprop-kokkos revisions from the preceding comparison,
CPython 3.14.7, package versions, v2 extension and OpenBLAS binary were retained.
Only the FFTW core library was replaced; its existing OpenMP wrapper and runtime
were retained. These were experimental installed environments, not newly repaired
release wheels.

Three FFTW 3.3.10 builds share GCC 13.3 and common flags
`-O3 -march=x86-64-v2 -mtune=generic -fPIC -ffp-contract=fast`, with no umbrella
fast-math. All enable SSE2 and AVX; the second additionally enables AVX2, and the
third additionally enables AVX2 and AVX-512. The freshly rebuilt AVX library is
the control. The original wheel's SSE2/AVX library was also sampled once per
configuration, but its original compiler options differ, so it is not the control
for attributing a change to the enabled SIMD implementations.

The option `--enable-avx-128-fma` was deliberately omitted: in
[FFTW 3.3.10's configuration](https://github.com/FFTW/fftw3/blob/fftw-3.3.10/configure.ac)
it enables legacy AMD FMA4, which this Zen 5 processor does not expose. It is
distinct from the `avx2_128` codelets included by `--enable-avx2`.

The same seven configurations and measured windows as the preceding comparison
were used: RDW, CtC and radial adaptive multistep at one and eighteen threads;
Kerr fixed RK4IP only at eighteen. Each configuration has three adjacent triples
of fresh processes, rotating AVX/AVX2/AVX-512, AVX2/AVX-512/AVX, then
AVX-512/AVX/AVX2. Configuration order is shuffled after round one. One worker
runs at a time on the same fixed core or eighteen cores, with BLAS at one thread.
Each worker has untimed preparation and a first window before its measured window.
Library hashes, runtime capabilities and planner wisdom were captured after timing.

**Results.** All 70 windows completed in 34 minutes 48 seconds of
collection: 63 repeated comparison windows and seven stock controls. All passed
the timing load guards, imported the intended extension and FFTW libraries, and
reported the requested Kokkos/FFTW widths with one BLAS thread. No sample was
discarded, no accuracy override was used, and no unreviewed warning was recorded.
An initial controller affinity check failed before the first sample; setting
profile affinity correctly allowed the prepared environments to run without
rebuilding or repeating a measured window.

| Case | Threads | AVX median / s | AVX2 median / s | AVX-512 median / s |
|---|---:|---:|---:|---:|
| RDW | 1 | 0.1109 | 0.1137 | 0.1129 |
| RDW | 18 | 0.0854 | 0.0847 | 0.0836 |
| CtC | 1 | 2.1792 | 2.5506 | 2.3562 |
| CtC | 18 | 1.3776 | 1.4874 | 1.4313 |
| Radial filament | 1 | 58.1195 | 58.4513 | 58.0969 |
| Radial filament | 18 | 4.8802 | 4.8451 | 4.7193 |
| Kerr beam | 18 | 11.5954 | 11.3391 | 11.3735 |

Both changes below are relative to the rebuilt AVX control. Each is the median
of three within-round values `100 * (candidate_time / avx_time - 1)`, not the
ratio of the separately listed time medians. Positive values mean slower.
MAD is the median absolute deviation in percentage points (pp); ranges show all
three paired values. Neither is a confidence interval.

| Case | Threads | AVX2 change / % | MAD / pp | Range / % | AVX-512 change / % | MAD / pp | Range / % |
|---|---:|---:|---:|---:|---:|---:|---:|
| RDW | 1 | +2.11 | 0.79 | +1.31 to +3.08 | +0.63 | 0.10 | +0.53 to +2.25 |
| RDW | 18 | -0.70 | 2.28 | -4.27 to +1.59 | -2.09 | 0.23 | -11.01 to -1.86 |
| CtC | 1 | +15.75 | 0.98 | +14.78 to +18.74 | +6.85 | 0.82 | +6.03 to +9.27 |
| CtC | 18 | +9.21 | 1.13 | +7.56 to +10.34 | +3.44 | 1.02 | +2.42 to +4.46 |
| Radial filament | 1 | +0.19 | 0.21 | -0.02 to +1.23 | -0.29 | 0.12 | -0.41 to -0.04 |
| Radial filament | 18 | -0.72 | 1.90 | -2.72 to +1.18 | -3.38 | 0.79 | -4.17 to -1.37 |
| Kerr beam | 18 | -2.03 | 1.44 | -3.75 to -0.59 | -3.45 | 0.32 | -3.77 to -0.55 |

Actual accepted / rejected / source-evaluation counts matched across all variants
and repetitions: RDW 21 / 0 / 21, CtC 19 / 0 / 19, radial filament 227 / 2 / 230,
and Kerr 10 / 0 / 40. All 60 adaptive windows passed the existing accuracy
envelopes against the retained references. The ten Kerr windows kept the
requested `not-qualified` status and expected material-range warning; their
NumPy reference was not restarted. No new reference or tolerance was introduced.

**What the libraries actually selected.** Native queries report
`DYNAMIC_ARCH` for OpenBLAS and select its `SkylakeX` kernel family. The unchanged
binary selects `Nehalem` under Intel SDE's non-AVX CPU model. The newly built
FFTW libraries report their enabled SSE2/AVX, AVX2 and AVX-512 implementations;
standalone plan/execution probes select the corresponding vector codelets.
Each new library also passed three small FFT probes under SDE Nehalem instruction
checking, selected SSE2 codelets and reported the higher ISAs unavailable. These
small fallback probes do not replace qualification of a final release wheel.

Wisdom exported from the propagation workers contains AVX2/AVX2_128 codelets for
the AVX2 build and AVX-512 codelets for the AVX-512 build, sometimes mixed with
narrower implementations. The available wider implementations are therefore
being selected. Wisdom records planned components; it does not provide dynamic
instruction counts, the time spent in each component, or a performance profile.
The common-code v2 floor and the availability of wider dispatched kernels are
compatible, as FFTW's
[installation documentation](https://fftw.org/doc/Installation-on-Unix.html)
and [OpenBLAS distribution guidance](https://www.openmathlib.org/OpenBLAS/docs/distributing/)
describe.

**Interpretation and limits.** This experiment does not show a uniform gain from
enabling wider FFTW implementations. CtC becomes slower in both new builds at
both widths. The other rows show smaller changes or gains, with the observed
spread above. In particular, instruction width is not a reliable proxy for the
speed of these complete propagation windows.

All workers use `FFTW_ESTIMATE`, which chooses plans using heuristics rather than
timing alternative algorithms ([FFTW planner flags](https://fftw.org/doc/Planner-Flags.html)).
Changing the enabled implementations also changes the available plan choices.
A different choice can lose despite using newer instructions; this is a plausible
explanation for the CtC result, not a measured attribution of its slowdown.
Transform shapes, memory traffic, non-FFT work and threading also matter.
This campaign did not measure their separate contributions, compare
`FFTW_MEASURE` or `FFTW_PATIENT`, or test another processor. Three pairs on one
VM, no A/A band, short windows and a single stock control per configuration
limit the precision and generality. No result is extrapolated to a full run.

**Decision and handoff.** Keep the maintainer's v2 wheel floor and optional v3
source build. A shared library with safe runtime dispatch can serve both.
These measurements do not justify enabling every wider FFTW implementation as
an unconditional performance improvement. Any later recipe change should compare
planning policy and representative transform shapes, preserve numerical checks,
and qualify the repaired release wheel on the minimum CPU model. Section 6.2
also records selective dispatch for future application kernels; Kokkos SIMD
provides compile-time vector types, not an automatic runtime CPU selector.
The production build recipe and defaults were not changed by this experiment.

**Verification.** The retained analysis checks all library identities, imported
builds, actual thread widths, paired resolved inputs, load guards, work counts
and numerical statuses. Raw records, build configurations, library binaries,
reference receipts and source inputs were archived. The documentation change
passed whitespace and spelling checks, SEP consistency tests and the Sphinx
build. Full repository, slow and coverage gates were not repeated for this
documentation-only report. The rental was terminated and its associated resources
removed after collection.

### Phase 0 outcome and consolidation — 2026-10-21

This subsection consolidates phase 0 for the checkpoint of §11: the Windows x64 and
Linux aarch64 builds, the same-machine timings against `numba_cpu`, and the
performance work they prompted. The subsections above record the x86-64 floor and the
Zen 5 follow-ups in detail. The other measurements are in the
[benchmark ledger](../../benchmarks.md) and the [CPU performance record](../../cpu-performance.md).

**Windows x64.** Two wheels were built and repaired with delvewheel. Both use Clang with
the GNU driver on the MSVC ABI, the MSVC STL and CRT, and release `/MD`. One uses the
Kokkos Threads host space, with no OpenMP DLL. The other uses Kokkos OpenMP on upstream
LLVM `libomp`; Microsoft's `libomp140` is not redistributable. FFTW 3.3.11 was built with
CMake, with its threads combined into `fftw3.dll`. The native code compiles without
warnings under `-Werror`.

The solprop default lane passed on Windows except for two `numba_cpu` cache tests,
whose file names exceed the 260-character path limit. The Windows defects it found were
fixed: the pool scans' process-handle truncation and pipe type, artifact fsync through a
read-only handle, and unguarded POSIX calls in the benchmark harnesses.

At the default thread count, Kokkos OpenMP stalled on the Raman cases: the reduced
comb-to-continuum took 81 s against 2.7 s. The cause was Kokkos's team-barrier backoff.
Its sub-microsecond sleeps are rounded up to Windows's 15.6 ms timer tick. solprop-kokkos
now patches the backoff to yield on Windows only. With the patch the reduced CtC takes
2.0 s, against 2.3 s on `numba_cpu`.

**Linux aarch64.** Built at the first attempt with a Neoverse N1 baseline. The default
lane passed except for five tests that assumed bitwise batched FFT rounding or immediate
inode reuse. Those tests now use bounds.

**One BLAS thread pool.** As built, the wheels bundled a pthreads OpenBLAS whose pool ran
beside Kokkos's own. On a 32-core Zen 5 this made the coupled-wall and modal capillary
cases 10 to 75 times slower than necessary. The same happened on Windows and aarch64.

An OpenMP OpenBLAS on the extension's own OpenMP runtime removed the loss with no
settings. It was 2 to 7 times faster than `numba_cpu` on the modal cases at full width,
and level with it at one thread. Kokkos Kernels' native GEMM matched it at full width,
but was 15 to 80 % slower at one thread on small modal shapes and 2.5 to 4 times slower
on the TE/TM wall. MKL is kept for local builds only: it has no aarch64 or macOS builds,
it is large, and its AMD performance is uncertain.

The Linux distribution build now builds OpenBLAS with OpenMP from pinned source, using
the extension's compiler. A pthreads OpenBLAS in any build (a local build against a
distribution's library, or the current Windows wheel) is held to one thread beside the
Kokkos pool, and an OpenMP OpenBLAS is capped at the Kokkos thread count, unless the user
sets the BLAS thread count. macOS keeps
Accelerate. Windows moves to the same single-runtime design in phase 1: Kokkos, FFTW and
OpenBLAS on LLVM `libomp`. That design was the fastest Windows variant on the modal and
wall cases.

**Per-core gaps.** The one-thread comparison against `numba_cpu` found four gaps. They
were addressed as engineering work, without a SEP:

| Gap at phase 0 | Change | Result |
|---|---|---|
| Radial free-space route 2.5–6 times slower at one thread on every platform | partial node chunks reach the interleaved real GEMM instead of a scalar contraction | 0.91–0.95 times `numba_cpu`'s time at one thread (Apple silicon, Xeon); 0.10–0.39 times at the default width |
| Multimode step 20–40 % slower at one thread on Apple silicon and Windows | shared cubic monomials; plans for active tail rows only; measured serial FFTW wisdom; FFTW for short serial Raman rows | 1.00–1.07 times `numba_cpu` at one thread on Apple silicon; 5–19 % faster at one thread on Zen 5. Measured *threaded* plans were sometimes slower and are no longer the default |
| Small problems slower at the default width than on one thread | per-operation thread widths chosen from the operation's dimensions at construction | small cases 2.0 times faster than one thread (geometric mean) on Apple silicon and the Xeon |
| Resident Adams history out of memory on the 384² Kerr beam at 64 GB | unused dense-output volumes omitted when planes are landed on | 29 → 18 field volumes; projected peak 62.7 → 48.2 GiB (`numba_cpu`: 34.8 GiB) |

The system allocator and vector transcendental functions were measured and ruled out as
causes of the multimode gap. The TE/TM wall's transforms remain a known one-thread cost.

**Instruction sets.** Phase 0 measured the two instruction-set baselines and the FFTW
codelet widths, recorded above. The x86-64-v2 floor costs at most 2 % (Zen 5) to 4 %
(Xeon) on the benchmark core. Wider FFTW codelets are not a uniform gain. The
distribution build now defaults to x86-64-v2.

The general modal route was not part of that comparison. With the same source and BLAS on
the Xeon, v2 is 23–24 % slower than v3 on general modal at one thread, and 7–12 % slower
at full width. At full width v2 is still 5.4 times faster than `numba_cpu` on general
modal; at one thread it runs at 0.79 times `numba_cpu`'s speed. The core cases stay
within 3 %. The maintainer accepted this cost for the conservative floor. The remedies are
the v3 source build, and later selective dispatch of the general-modal kernels (§6.2).

**The single-runtime BLAS on the Xeon.** The old distribution wheel (pthreads OpenBLAS)
was compared with the new one (OpenMP OpenBLAS, plus the BLAS fit) at the default width
(72 threads):

| Case | new wheel faster than old | new wheel faster than `numba_cpu` |
|---|---:|---:|
| Reduced comb-to-continuum, default stepper | 6.5× | 1.3× |
| Four-mode RDW, default stepper | 22× | 1.5× |
| Scalar wall | 10× | — |
| TE/TM wall | 26× | — |
| Ten-step full comb-to-continuum | 13× | 2.7× |

At one thread the two wheels are within 6 %, except for the v2 cost on general modal
described above.

**Checkpoint evidence.** For proceeding:
- every phase 0 platform builds and passes;
- the per-core gaps are closed or narrowed on the platforms re-measured;
- one conservative x86-64 wheel is viable;
- with one BLAS runtime, the compiled backend leads `numba_cpu` wherever threads pay, and
  is level with it at one thread.

Still open before phase 2:
- Windows re-measurement of the fixes on the single-runtime wheel;
- the remaining qualification rows of the thread policy and the 36-thread Xeon confirmation of the multimode change;
- the TE/TM wall at one thread on native GEMM;
- the comparison of the actual release wheel with `numba_cpu` (R9).

Standard-suite baselines of Kokkos against `numba_cpu` on the merged code are recorded in
the [benchmark ledger](../../benchmarks.md). The remaining performance items are in the
[roadmap](../../roadmap.md).

### Baselines and release-recipe tests — 2026-10-22

**Native against `numba_cpu` on the standard suite.** The standard benchmark suite's core
cases ([SEP 30](sep-0030-standard-benchmark-suite.md)), standard mode, matched
thread budgets, on five machines, after the consolidation above. Each entry is
`numba_cpu`'s median warm-window time divided by the native backend's, so above 1 the
native backend is faster. "Full" is the production width; MS and RK are the adaptive
Multistep and Runge–Kutta tracks. Both backends did identical work (accepted and rejected
steps, source evaluations) in every row, and every adaptive accuracy check passed.

| Case, track | Apple M5 Max (18) 1 thr | Apple M5 Max (18) full | Xeon Gold 6240 (36) 1 thr | Xeon Gold 6240 (36) full | Zen 5 Linux (32) 1 thr | Zen 5 Linux (32) full | Zen 5 Windows (16) 1 thr | Zen 5 Windows (16) full | Graviton4 (32) 1 thr | Graviton4 (32) full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RDW, MS | 1.03 | 1.11 | 1.06 | 1.35 | 1.08 | 1.03 | 1.03 | 1.53 | 1.21 | 1.17 |
| RDW, RK | 0.93 | 0.98 | 1.03 | 1.45 | 1.06 | 0.84 | 0.84 | 1.15 | 1.09 | 0.90 |
| CtC, MS | 0.92 | 1.27 | 1.14 | 2.51 | 1.03 | 1.41 | 0.97 | 1.68 | 1.08 | 1.31 |
| CtC, RK | 0.85 | 1.14 | 1.14 | 2.61 | 1.04 | 1.29 | 0.91 | 1.46 | 1.02 | 1.15 |
| Kerr 384², RK4IP | 1.88 | 1.95 | 1.53 | 1.96 | 1.29 | 1.31 | — | 1.21 | — | 1.19 |
| Radial, MS | 1.02 | 2.96 | 1.04 | 5.78 | 1.01 | 4.57 | 1.00 | 5.24 | 1.49 | 6.55 |
| Radial, RK | 0.97 | 1.70 | 1.12 | 3.92 | 1.09 | 3.13 | 0.90 | 3.16 | 1.62 | 5.15 |

- At the production width the native backend leads on every machine and case except RDW
  with adaptive RK on Zen 5 Linux (0.84), Graviton4 (0.90) and the Mac (0.98).
- At one thread it is level or ahead on the Xeon, Zen 5 Linux and Graviton4. `numba_cpu`
  keeps 3–16 % on CtC and the RK tracks on the Mac and Windows; the one-thread multimode
  step is the remaining per-core gap.
- Limits:
  - The Zen 5 Linux pair ran on two instances of the same type a day apart.
  - The Windows machine has 16 cores (a cloud quota), with an x86-64-v3 build.
  - The load guard flagged every Mac and Windows sample as contended, and those samples
    are kept as flagged (SEP 30 R4).
  - The one-thread Kerr beam was not run on Windows or Graviton4.

**Release-recipe tests.**
- **manylinux.** The Linux distribution wheel builds in a `manylinux_2_28` container once
  the heap ledger finds `mallinfo2` at run time; it is glibc 2.33 or later, and the
  container has 2.28. auditwheel tags it `manylinux_2_27_x86_64.manylinux_2_28_x86_64`.
  In a clean Python 3.14 container without a compiler, `check_build` passes and the
  native package's suite passes (360 passed, 6 skipped).
- **macOS deployment target.** The extension builds for macOS 14, but the bundled
  Homebrew libraries do not: `libomp` requires macOS 27 and FFTW macOS 26. The release
  recipe must build both for the deployment target.
- **Windows.** The single-runtime wheel at x86-64-v3 passes the solprop suite on the
  native backend: 16,690 passed. The two failures are the known `numba_cpu` cache tests
  whose file names exceed Windows' path limit.
- **Linux aarch64.** The first aarch64 build of the OpenMP OpenBLAS (`DYNAMIC_ARCH`)
  passes the solprop suite on the native backend on Graviton4 (16,891 passed). Against the
  previous pthreads-BLAS wheel it is 1.5–20 times faster at the default width on the
  BLAS-bound cases and level at one thread, as on the Xeon and Zen 5.

### Phase 1: one repository — 2026-10-22

**Branch** `feature/sep-0029-one-repository`.

**What was done.**
- **Import.** The native repository's `main` was merged under `native/` with
  `git subtree add --prefix=native <native repository> main`, keeping its history. That
  history was not the single commit §6.4 assumed: 290 commits had accumulated since its
  squash, and all of them are kept. The native tree's licence files and REUSE
  configuration gave way to the root's, except a copy of `LICENSE`, which the wheel's
  metadata needs because `license-files` cannot reach outside the project directory.
- **Renames, with no aliases.**
  - Package `solprop_native`, distribution `solprop-native`, backend name `"native"`
    (registry, `SOLPROP_TEST_BACKEND`, the benchmark suite's `--backend`, the memory
    model, the recorded test series), classes `NativeBackend` and `NativeBatchBackend`.
  - Environment variables, CMake options and macros `SOLPROP_NATIVE_*`; the option that
    compiles for the build machine is `SOLPROP_NATIVE_HOST_ARCH`, to avoid
    "native native".
  - **C++ namespace and files** (the decision §11 asked for): the namespace is
    `solprop_native`, with the former inner `native` namespace collapsed into it
    (`solprop_native::host`, `solprop_native::test`); the headers are
    `native/include/solprop_native/`; the allocation and kernel labels
    `solprop_native.*`. Names of the Kokkos libraries themselves are unchanged.
- **Packaging.**
  - The repository root is a uv workspace with `native/` as its member, and one lock.
    `uv sync --extra native` builds and installs the native package; a plain sync leaves
    it out.
  - One version: solprop takes it from the repository's tags with hatch-vcs, and
    solprop-native with scikit-build-core's setuptools-scm provider rooted at the
    repository; CMake takes it from scikit-build-core. The `native` extra pins
    `solprop-native==` that version through a hatch metadata hook (`hatch_build.py`).
  - solprop-native's `requires-python` lost its upper bound (`<3.15`): in one workspace
    it narrowed the whole lock. Wheels are tagged per interpreter in any case.
  - The sibling checkout is gone: the build records its own version as the solprop
    version it belongs to. The kernel-contract schema check at import is kept (R12).
- **One gate.** `poe check-native` runs the native gate (`native/scripts/check.sh`,
  now run from the workspace environment and asserting that the wheel's version is the
  repository's), and `poe check-all` runs solprop's gate and then the native one. The CI
  jobs `native-quality` and `native-suite` replace the cross-repository jobs; the CUDA
  workflow moved to the root.
- **Documentation.** The native package's build instructions moved into the manual
  ([Building the native backend](../../../guide/performance/building-native.md)), with the
  CUDA and HIP builds; the installation pages, the native and tuning pages, the
  contributor guides, the developer pages and the decisions describe one repository and
  two distributions. The native changelog was folded into the root changelog. SEPs,
  dated reports and the measurement ledger keep the names they were written with.

**Deviations.** The import kept 290 commits rather than one (above). The upper bound on
solprop-native's Python version was dropped.

**Verification.** The phase gate ran once, on macOS (Apple M5 Max) and on Linux (a
96-core Zen 5 instance, with GCC 16), with targeted reruns after each fix.
- **Linux.** `poe check` (16,187 passed, with the documentation build), `poe test-slow`
  (146 passed), the coverage lane, the default and slow lanes on the native backend
  (147 slow passed), and `poe check-native` (clang-tidy, the native tests, ASan and
  UBSan, 357 native pytest cases, the distribution wheel and its isolated smoke test).
- **macOS.** `poe check`, `poe test-slow` (146 passed), `poe cov` (100 % of statements
  and branches), the default lane on the native backend, and `poe check-native` (358
  native pytest cases, the wheel and its smoke test).
- **Found and fixed by the gate.**
  - The rename had skipped test identities with a number after the backend name, so the
    convergence records named writers that no longer existed.
  - The shorter namespace reflowed calls that clang-format lays out differently.
  - An editable install records the version of the commit it was installed at, so the
    native gate now reinstalls solprop before comparing the two halves' versions.
  - clang-tidy parsed a GCC 16 build with the system GCC's headers, and CMake passed it
    GCC's module-scanning flags; tidy now reads the compiling GCC's own installation, and
    module scanning is off.
- One timing-sensitive test of the benchmark runner (a crashed worker classified as
  timed out) failed once on the Mac under the load of two concurrent builds and passed
  when rerun alone.


### Phase 2: native parity — 2026-10-23

**Branches.** Three slices on their own branches (`feature/sep-0029-parity-r2`, `-g2`,
`-g1`), merged on `feature/sep-0029-parity` in the order R2 ∥ G2, then G1, as §11 asks.

**What was done.**
- **R16 and R2: who evaluates the shared helpers.** A frozen record of two operations,
  `solprop._helpers.Helpers` (`points`, the multi-position Clenshaw; `basis`, the Adams
  divided-difference rows and integrated weights), with three instances: `PLAIN` (the
  uncompiled Python forms), `NUMBA` (the cached dispatchers, created on first use) and
  `NATIVE` (C++ bindings in `solprop_native`). `StepDriver.helpers` names the backend's
  record: `numpy_ref` takes `PLAIN`, `numba_cpu` `NUMBA`, the native backend `NATIVE`,
  each batch backend its owner's. The driver passes it to every per-stage and per-step
  evaluation: the stage bundles and one-point stage reads, the Adams anchor, the
  compensation's rate, the Adams weights, and the reference's own z-table reads.
  Translation keeps the one shared evaluation, Numba where installed. Nothing switches
  process-wide. The C++ recursion is a header usable from host and device code, strict
  like the Python form; the native helpers write into caller-supplied buffers and
  allocate nothing. The new driver attribute changes the frozen surface, so the kernel
  schema is 34.
- **G2: Adams history export.** The native state copies its Adams rows (`store`, and,
  when compensated, `states`, `rates` and the frozen-residual flow's `applied`) to and
  from the host exactly; the resident engine exports and installs them in the host
  engines' format, and the snapshot refusal for native Adams is gone. A history whose
  rows an engine does not keep (the host engines' per-sample flow weights against native
  `applied`) is refused by name, and a resume that cannot install its history warns. A
  restore reloads the state together with its noise realisation.
- **G1: the z table on the native CPU backend.** The table's coefficients are uploaded
  once; before a trial, a native kernel evaluates the table at the trial's distinct
  positions into term slots (kept as a least-recently-used set keyed by position, so an
  accepted step's end is not evaluated again as the next anchor), and the stage rows weigh
  the slots with ±1, so every consumer of the terms (Runge–Kutta stages, low-storage
  flows, the Adams phase and compensation, dense output, noise, batches) runs unchanged.
  The compensated Adams rate reads the derivative table through one more slot. Device
  builds keep refusing the table, naming its size. The native `Batch` refusal is gone.
  The evaluation runs in blocks of 16 values with real and imaginary arrays, one strict
  function with no calls inside.

**Deviations from §6.6 and the capability audit.**
1. The audit's route (a), "one array with weight 1, rewritten per stage", cannot work: a
   native trial consumes every stage's increment in one call. The values at a trial's
   positions are the terms instead, weighed ±1.
2. The Adams march reads its stage scalars one position at a time, never through the
   stage bundles; those reads go through the backend's record too, or every backend's
   Adams runs would have used the shared compiled pass per stage.
3. The adapter owns the table evaluation; the helper record gained no third operation.
4. GCC does not inline a function marked with `optimize("fp-contract=off")` into a caller
   without it, so under GCC every strict helper is a call. The table kernel was written as
   one strict function with no inner calls; elsewhere this is unchanged (see Open
   questions).
5. A translation bug found by the radial trigger: the prefactor table for three or more
   species on independent density paths was left writeable, so that radial case could not
   be translated on any backend. Fixed.

**Verification.**
- New tests: native helpers against the plain forms within derived bounds (they agree to
  the last bit, recorded as information); the R16 isolation test in one process, for
  every march, the z table included (`numpy_ref` with native and Numba forbidden, native
  with Numba and the plain forms forbidden); the native Adams snapshot non-empty once the
  history forms, restored without the restart path, rows, positions, slots, identities,
  order and controller state equal exactly, for the plain, interaction, compensated and
  frozen-residual histories, the periodic path, HDF5 resume and a checkpoint between
  planes, incompatible histories refused, the propagation after the restore within the
  matched bound or the accuracy envelope; every z-table trigger against `numpy_ref` under
  each march (departures about 2e-15 against bounds about 3e-13), a z-table batch, dense
  output and noise; the blocked table kernel equal to the per-value recursion exactly.
- The gate, once with targeted reruns after each fix: on Linux (a 96-core Zen 5
  instance, GCC 16) `poe check`, `poe test-slow`, `poe cov` (100 %), the default and slow
  lanes on the native backend and `poe check-native`; on macOS the same lanes.
- Found by the gate and fixed: the native restore dropped the modified-noise realisation
  (two resume tests; it had been hidden by the restart path's looser bound); a native
  table kernel that met the speed target with Clang but not with GCC (deviation 4); six
  uncovered lines and branches; two type-narrowing gaps in new tests.

**Performance** (native time over `numba_cpu`'s for the same work; the full tables are in
the private ledger):
- Helpers, end to end per accepted step on small problems: 0.99–1.01 on Linux, 0.98–1.01
  on macOS.
- Adams runs with snapshots: total run time level with `numba_cpu` on Linux (1.398 s
  against 1.397 s, a kept save at each of about 110 steps) and on macOS.
- z-table runs, Linux, one thread and full width: moving dispersion 0.75–0.83, three
  gases 0.15–0.37, the multimode taper 0.34–0.56, radial with three gases 0.17–0.74; on
  macOS 0.19–1.01. The dry-air-to-vacuum case runs at 1.27–1.80 on Linux (0.83 on
  macOS), and so does its twin on a pressure gradient without the table (1.30–1.91): the
  excess is native's cost on the 126-row Raman composition under GCC, not the table.

**Open questions.**
- GCC builds and the strict helpers (deviation 4) may cost native performance on Linux
  beyond the table, the Raman composition case included; to be measured.
- Device builds could admit the z table when it fits in device memory; for a GPU check
  after the release.
- On Linux a kept native save costs about 4.5 ms against `numba_cpu`'s 1.6 ms on the
  benchmark case, unexplained; the run totals are level.

### Phase 3: open source and wheels — 2026-10-24

**Branches.** The wheel recipes on `feature/sep-0029-wheels`; after the history reset,
the fixes found by the public CI directly on `main`.

**What was done.**
- **Open source.** The repository's history was reset to one commit, and the repository
  was published as `LupoLab/solprop` under GPL-3.0-or-later (documentation
  CC-BY-4.0), with the native package inside it from the first public commit. The default
  branch is protected by a ruleset; Dependabot (pinned actions and the lock), secret
  scanning, private vulnerability reporting, Read the Docs and Codecov are set up.
- **Continuous integration.** One workflow of parallel jobs: lint and types, the
  documentation, the default lane on macOS, Linux and Windows, the slow lane, the
  lowest supported dependencies, coverage (100 % of statements and branches, with the
  native package built), the native gate (`native-quality`) and the default lane on the
  native backend (`native-suite`). GitHub's 16 GB runners needed swap and three test
  workers for the native and coverage jobs, and tests that had read the runner's free
  memory now fix it. Linux CPU builds of the native package now require GCC 14 or newer:
  GCC 13's code for the strict floating-point helpers cost native performance.
- **Wheels** (`.github/workflows/wheels.yml`; recipe and table in
  [Releasing and the native wheels](../../releasing.md)). cibuildwheel builds CPython 3.14
  wheels:
  - Linux x86-64 and aarch64 in `manylinux_2_28` with GCC 14, FFTW 3.3.11 and an OpenMP
    OpenBLAS with run-time kernel selection, repaired by `auditwheel`;
  - macOS arm64 with Homebrew's Clang, its own LLVM `libomp` and FFTW, Accelerate as
    the BLAS, deployment target 14.0, repaired by `delocate`;
  - Windows x64 with Clang on the MSVC ABI, FFTW and OpenBLAS built static on LLVM's
    `libomp`, repaired by `delvewheel`.

  Each wheel carries the licence texts of what it bundles. A test job per platform
  installs the repaired wheel beside the locked dependencies and runs the smoke test
  (versions, profile, bundled-library locations, licences), the build record, the
  coexistence script (both import orders in fresh processes, every package's thread pool
  started, interleaved native and `numba_cpu` runs) and the default lane on the native
  backend. The `cpu-floor` job runs the Linux x86-64 wheel under Intel's Software
  Development Emulator. The publishing jobs use trusted publishing, one GitHub
  environment per distribution (`pypi`, `pypi-native`), and refuse to publish a
  `solprop-native` source distribution (Q7).
- **The x86-64-v3 baseline, with a fallback** (revising Q2 and Q4). The x86-64 wheels are
  built for x86-64-v3; x86-64-v2 remains a source-build option. v3 removes v2's cost
  (2–4 % on the core cases, 7–24 % on the general modal route) and excludes processors
  without AVX2: Intel before Haswell (2013) and AMD before Excavator (2015). The package
  checks the processor on every wheel platform before any native code runs (Linux from
  `/proc/cpuinfo`, Windows through `IsProcessorFeaturePresent`), and raises
  `CpuBaselineError` naming the missing features. Under `backend="auto"`, and only for
  that refusal, solprop selects `numba_cpu` instead and warns once, naming the missing
  features and asking for an issue on the repository; `backend="native"` still raises.
  This amends §5, where any failure raised.
- **macOS and pyFFTW** (§7, coexistence). pyFFTW's macOS wheel carries its own `libomp`.
  With the guard bypassed, importing the native package after pyFFTW crashed in Kokkos's
  OpenMP initialisation, and initialising pyFFTW's runtime after the native package
  aborted on a duplicate runtime. The guard is kept, and the import-order rule is
  documented as a macOS limitation; solprop's own imports load the native package first.
- **Windows OpenBLAS.** Built by Clang for Windows, OpenBLAS's run-time kernel selection
  (`DYNAMIC_ARCH`) was not reliable at one thread: its AVX-512 kernels gave wrong complex
  results (the coupled TE/TM flow off by 5e-4 against rounding-level bounds), and its
  Haswell and Zen kernels, or any kernel family forced at run time, faulted in the radial
  route's real GEMM. The Windows wheel links OpenBLAS built for one fixed target, Haswell,
  the wheel's baseline, which phase 0 had qualified.

**Deviations.**
1. The x86-64 baseline is v3, not v2 (above), with the `auto` fallback.
2. The macOS deployment target is 14.0, that of NumPy's and SciPy's Accelerate wheels,
   rather than the 11.0 the recipe first reached.
3. The Windows OpenBLAS has no run-time kernel selection (above).
4. Phase 4 and the per-platform performance qualification were not done before this SEP
   was closed ("Closing", below).
5. §7's coverage clause is met on GitHub's runners, which carry compilers and run current
   operating systems, not on clean machines at the declared minimum systems; the wheels'
   test jobs run the default lane on the native backend, and the slow lane on the native
   backend runs in CI from a source build, not from the wheels; and no lane yet fails on
   unexpected native skips (R13). These go to the roadmap.

**Verification.**
- CI on `main` green in every job, after the fixes the first public runs found: the
  type check of native-only test imports, thread counts on small runners, tests that read
  the runner's memory, the oldest-dependency job's test tools, the native gate's
  compiler, and the Windows and macOS wheel recipes (PowerShell 7 for the Windows
  dependencies; Accelerate's BLAS naming declared, so the macOS configure needs no
  Fortran compiler).
- Wheels: the wheels workflow at the merge of SEP 34 (2026-10-24) built all four wheels
  and passed each platform's test job: Linux x86-64 and aarch64, macOS arm64 and Windows
  x86-64, the last with the fixed-Haswell OpenBLAS.
- CPU floor: under Intel SDE 10.13.1 the Linux x86-64 wheel ran its smoke test on an
  emulated Haswell; on an emulated Nehalem and Sandy Bridge its import was refused by
  the processor check, naming the missing features, before any unsupported
  instruction ran, and automatic selection fell back to `numba_cpu` with its warning.
  SDE emulates CPUID but not `/proc/cpuinfo`, which the check reads on Linux, so the
  job bind-mounts each older chip's flags over that file.
- Windows, outside CI: the fixed-Haswell wheel ran the default lane on the native backend
  on a 16-core Zen 5 Windows VM (16,337 passed); its two failures were two Numba cache
  tests exceeding the VM's path-length limit.
- Installation (§7, Q7), from the release workflow's wheelhouse (the solprop wheel and
  the four native wheels): `pip` and `uv` into fresh environments on macOS, with the
  extra (the native wheel installed, `auto` resolves to `native`) and without it (no
  native package, `auto` resolves to `numba_cpu`, no warning). Resolved for other
  platforms: the extra resolves on Linux x86-64 and aarch64 (`manylinux_2_28`), Windows
  x64 and macOS 14 on arm64, and fails to resolve, naming the missing wheel, on Windows
  on Arm, musl Linux, Intel macOS and macOS 13; plain `solprop` resolves on all of them.

### Closing — 2026-10-24

The maintainer set this SEP Final after phase 3 and before the first release. Phases 1–3
are merged and reported; the native backend runs everything `numba_cpu` runs (§6.6),
and `numba_cpu` is unchanged (§6.3). Moved to the [roadmap](../../roadmap.md):
- **The first release** of both distributions (phase 4), with the release checklist of
  [Releasing and the native wheels](../../releasing.md) and an archive DOI;
- **the per-platform performance qualification** of §7 (the standard suite's core
  cases, native against `numba_cpu`, at one thread and the production width, on each
  wheel family), after the release;
- **qualification on clean machines** at the declared minimum systems, the slow lane
  from the wheels, and failing on unexpected native skips;
- **evidence from use** (coverage, default performance, coexistence), for the later
  decision on `numba_cpu` (§6.3);
- of phase 2's open questions, the z table on GPU builds and the cost of a kept native
  save on Linux; the third, GCC and the strict helpers, is settled: the cost was GCC 13's
  code for the strict helpers, which GCC 14 and 16 do not share, and Linux CPU builds now
  require GCC 14;
- **macOS and pyFFTW**: removing the import-order rule needs a native macOS build
  without OpenMP, or a pyFFTW wheel that does not initialise its own `libomp`.

## Part III — History

- 2026-10-20: Drafted at the maintainer's request, to record the option while it is
  weighed; the maintainer is undecided.
- 2026-10-20: Reviewed by GPT-6 (AI), with John Travers's human review recorded at his
  request. Part I is unchanged. The findings, evidence and maintainer clarifications
  follow. Reviewed records completion of review, not acceptance; status remains Draft.
- 2026-10-20: Clarified R5 after the maintainer explicitly excluded Intel Macs,
  limited laptop/desktop compatibility to seven years, and identified older Linux
  clusters and modern Arm systems as the useful coverage targets. Part I remains unchanged.
- 2026-10-20: Added R14 on Windows toolchains and coexistence with existing Python
  numerical libraries, including inspection of the locked Windows wheels.
- 2026-10-20: Revised R14 after the maintainer raised MSVC with native threading:
  prioritise proving a Windows build without OpenMP, including its thread controls
  and idle-pool behaviour. This changes the review recommendation only.
- 2026-10-20: Extended R8 with the Linux/macOS trade-off: compare native threading
  deliberately, while retaining the measured OpenMP path as the initial default.
- 2026-10-21: Phase 0 on Windows x64 and Linux aarch64; same-machine timings against
  `numba_cpu`. The maintainer resolved Q2 and Q4: x86-64-v2 wheels, with v3 as a
  source-build option. The x86-64 floor was qualified, and repeated and FFTW-codelet
  comparisons were made on Zen 5.
- 2026-10-21: Phase 0 consolidated (Part II). The single-runtime Linux BLAS and the
  x86-64-v2 distribution default were implemented. The per-core gaps, the problem-size
  thread policy and the Adams memory were recorded. Status remains Draft pending the
  checkpoint decision.
- 2026-10-22: Revised at the maintainer's direction (Part I): the native backend moves
  into solprop's repository, before it is open-sourced, as a second distribution
  (`solprop-native`, module `solprop_native`) and is published as wheels; Q1, Q3, Q5 and Q6 resolved, Q7 added;
  the freeze and removal of `numba_cpu` move to a later SEP. The response to the AI review
  follows the review. Review status remains Reviewed; the revised Part I has not been
  re-reviewed.
- 2026-10-22: Native parity added as phase 2 at the maintainer's direction (§6.6), from a
  capability audit of the three backends. Integration is now phase 1, and open source and
  wheels phase 3. R1 and R2 are now taken in this SEP.
- 2026-10-22: AI re-review of the revised scope (R15–R18) recorded, and answered in the
  response to re-review with revisions to §5, §6.5, §6.6, §7 and Q7.
- 2026-10-22: Accepted by John Travers. Part I is frozen from here.

### AI review — 2026-10-20

**Assessment.** The direction is credible and worth proving. Keeping an independent
NumPy oracle and one compiled implementation shared by CPU and GPU is a substantial
maintenance improvement. The measured CPU advantages support building the wheels.
They do not yet establish that removing Numba preserves every supported workflow or
that the proposed installation works on the advertised platforms. Recommend proceeding
with the proving work, without committing to retirement until the conditions below
have evidence. The largest risks are functional coverage and distribution engineering;
very old CPU instruction sets are a smaller concern under the clarified support policy.

**Maintainer clarifications during review, including the follow-up.** John accepts a
measured, documented slowdown on rare hardware. Intel Macs and Raspberry Pi are
explicitly excluded. Laptop/desktop models released more than seven years ago need
not be supported; at this review date that means before approximately October 2019.
Older laptop/desktop coverage can be confined to x86-64. Older Linux clusters remain
a distinct compatibility interest; the seven-year limit was not applied to them.
Modern Arm machines on macOS, Linux and Windows are desirable. A broad x86 fallback
is useful if proportionate. His current use includes Apple silicon M5, Linux EPYC
Zen 3 and Zen 5, NVIDIA H200 and AMD MI325X; Windows has not been tested by him.
Platforms outside mainstream NumPy, SciPy or FFTW support may be excluded, as may
platforms unsupported by the latest stable Julia release: solprop need not support
more than Luna's language platform. These are support boundaries, not a requirement
to add every platform within their intersection. They do not approve any particular
performance margin or the final removal.

**Review basis.** Read all of Part I against the architecture, conventions, backend
contract, SEP 0, SEP 25, ADR-034, the SEP 28 benchmark ledger, the current Python
implementation, and solprop-kokkos's loader, backend and build configuration. Checked
upstream packaging and platform documentation, and the wheel filenames in `uv.lock`.
Ran a small import-blocking probe of the shared Numba helpers (R2). This is a design
review, not new Windows, legacy-CPU, GPU or wheel-performance qualification. Priorities:
**P1** needs resolution in the design before the dependent migration step; **P2**
needs a concrete decision or verification task before retirement. The drafting agent
should answer each numbered point in a separate response, following SEP 0.

#### R1 — P1: Retirement must cover workflows, not only machines

**Sections:** 1, 6.3, 7, 9, 11.

There are current Numba capabilities that a Kokkos wheel does not replace.
`solprop_kokkos.backend.KokkosBackend` calls
`solprop.translate.stage.require_device_ready` even for CPU execution. It refuses
general z tables, including independently varying mixture profiles and moving
dispersion that cannot use the parameter expansion. The refusal explicitly directs
users to Numba or NumPy. Separately, `solprop.driver.run._validate_snapshot_request`
refuses Kokkos Adams snapshots because its resident Adams history cannot be exported;
Numba supports them. These affect users on entirely modern, supported hardware.
Building Kokkos from source does not remove either limitation.

**Requested revision:** add a capability audit across routes, z representations,
steppers, statistics, batches, snapshots and resume. For each difference, require
either Kokkos CPU support before retirement or an explicitly accepted loss with a
measured, usable alternative. Add regression cases for the two concrete gaps above.
Distinguish the existing TDSE exception, which already requires NumPy on both compiled
backends, from new losses caused by this SEP. Do not treat successful benchmarks on
the common cases as proof of functional parity.

#### R2 — P1: Removing the backend does not remove the Numba dependency

**Sections:** 2, 6.3, 7, 11.

`solprop.translate.chebyshev._clenshaw` and `_clenshaw_batch`, and
`solprop.stepper.multistep._basis`, compile through `solprop._numba_cache` outside
`backends/numba_cpu/`. They serve shared translation and stepping. An isolated probe
that blocked imports of `numba` while leaving NumPy and solprop installed found that
both `clenshaw(0.0, np.array([1.0, 2.0]))` and
`integrated_weights([0.0, 1.0])` raise `ModuleNotFoundError`.
Making Numba an extra without addressing these callers breaks ordinary execution,
including the promised NumPy escape route.

**Requested revision:** inventory all runtime and test imports before phase 2, name
the replacement for these helpers, and test a genuinely Numba-free environment.
A native helper can accelerate the normal path, but the reference installation must
also work without the native package. Preserve the strict arithmetic protections of
the Clenshaw differences and integrated weights. Their comments record substantial
host-overhead savings, so measure small problems and translation as well as large
native steps after the change. This work belongs before optionalisation, not merely
in the final deletion.

#### R3 — P1: Resolve installation before promising an import-time fallback

**Sections:** 5, 6.2, 6.4, 7, 9, 11.

A required native dependency with no matching wheel is an installation failure or an
attempted source build. Python never reaches the warning and NumPy fallback in §5.
Platform markers can select OS and architecture, but they cannot express arbitrary
wheel availability, libc compatibility or x86 feature levels. Standard wheel tags
also do not select v2 versus v3 from CPUID. Two otherwise identical wheels with
different build numbers would not implement that choice. See the
[dependency-marker specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/),
[wheel-tag specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)
and [manylinux caveat](https://github.com/pypa/manylinux#manylinux_2_34-almalinux-9-based---alpha).

**Requested revision:** settle a workable distribution/dependency design in phase 0
or 1, before making the compiled package mandatory. Demonstrate ordinary `pip` and
`uv` installs from a wheelhouse with no compiler, including a platform without a native
wheel, an older supported OS, and a missing native release. Provide a documented
reference-only installation route. For separate baseline packages, specify how both
installation and selection work without install-time executable selectors or network
downloads on import. A fat wheel with one shared source and an early dispatcher is
the simpler candidate to prove.

Also define the fallback boundary: absent/unsupported native execution may trigger
the advertised warning, but invalid physics, schema mismatch and errors during
construction or propagation should remain errors. Explicit `backend="kokkos"` must
not silently change backend. Record the selected backend and build, not just the
requested `auto`, and never restart a partially executed run on NumPy.

#### R4 — P1: A conservative extension is insufficient unless the whole wheel is safe

**Sections:** 3, 5, 6.2, 7.

The existing `solprop_kokkos._cpu` guard reads the first flags row in Linux's
`/proc/cpuinfo` and returns without checking when that file is unavailable. It is
not a Windows feature detector. A v3 extension imported on an incompatible Windows
CPU could terminate the process rather than raise a catchable exception.
Dependencies and static initialisers can execute before a late selector.
The SEP 28 timing environment also used an FFTW built with `-march=znver3`; those
measurements do not qualify that library for a v2 wheel.

**Requested revision:** specify a baseline-safe loader on every shipped OS, including
OS support for AVX state as well as processor features, and safe behaviour when
detection is unavailable. Audit FFTW, BLAS, OpenMP, C++/Fortran runtimes and native
initialisation, not just the solprop extension's compiler flags. Shared FFT/BLAS
libraries may dispatch internally, provided their entry path meets the lowest
baseline. Test the installed, repaired wheel in subprocesses under a v2 CPU model,
on physical non-AVX2 hardware where available, and with masked VM features. Exercise
FFT, BLAS and a short propagation, not only import. Force each bundled variant in
qualification and record which actually ran. Emulation tests instruction safety,
not legacy-machine performance.

#### R5 — P2: Define the supported population from upstream evidence

**Sections:** 1, 2, 6.5, 7, 9, 11, Q4.

The maintainer's upstream boundary makes the coverage question tractable. The four
proposed CPU platform families match the intersection of the current official
Numba 0.67 and pyFFTW 0.15.1 CPython 3.14 wheel families recorded in `uv.lock`:
Linux x86-64, Linux aarch64, macOS arm64 and Windows x64. This is useful evidence
about the incremental change; it is not a count of people and does not by itself
verify all dependencies or OS versions.

| Population | Evidence and implication for this proposal |
|---|---|
| Linux/Windows x86-64 with v3 | Covered in design. Zen 3/5 are relevant qualification targets, but deployed OS/library compatibility still matters. |
| x86-64 with v2 but without v3 | The proposed fallback is valuable. This includes older mainstream CPUs and some newer low-power models; age alone is an unsafe selector. Intel lists the 2021 Pentium Silver N6000 with SSE4.2, illustrating why “modern” must not mean “AVX2”. Verify every required feature. |
| x86-64 below v2 | NumPy 2.5 documents `X86_V2` as its default x86 baseline. A custom v1 Kokkos build alone cannot make an incompatible NumPy wheel run. Under the maintainer's policy this need not block retirement; document custom-stack users separately. |
| Apple silicon | Covered in design. Choose and test the minimum macOS version; an M1 instruction baseline does not establish an OS deployment target. |
| Intel Macs | Explicitly excluded by the maintainer in the follow-up. No build-cost investigation or retirement blocker is needed for this population, regardless of upstream support or survey prevalence. |
| x86 Python under Rosetta on Apple silicon | A distinct interpreter configuration. Recommend documenting native arm64 Python as the supported route; Rosetta does not make an x86 Python process load an arm64 extension. |
| Linux aarch64 | Covered in design; choose an explicit generic Arm baseline rather than inheriting a build server's CPU. This does not create a Raspberry Pi support obligation. |
| Native Windows on Arm | Desirable in the maintainer's follow-up. Numba has a locked wheel, pyFFTW does not. Julia's current table lists x86 emulation on Arm, not a native Windows Arm tier. Evaluate native Kokkos/toolchain/library feasibility and give it an explicit milestone or deferral. Treat this as a desired target, not an additional removal blocker inferred from “would be useful”. An x64 Python under emulation is a separate combination, not native qualification. |
| musl Linux, FreeBSD, POWER, RISC-V and 32-bit systems | Do not call these newly abandoned Numba-wheel users as a group. Existing dependency wheels already differ or are absent. Apply the upstream boundary and document exclusions rather than promise that every source build is supported. |

Sources checked on 2026-10-20:
[NumPy 2.5 CPU build options](https://numpy.org/doc/stable/reference/simd/build-options.html),
[Numba's Intel-macOS withdrawal](https://numba.readthedocs.io/en/latest/release/0.63.0-notes.html),
[Intel N6000 specifications](https://www.intel.com/content/www/us/en/products/sku/212330/intel-pentium-silver-n6000-processor-4m-cache-up-to-3-30-ghz/specifications.html),
and [Julia's supported versions and platforms](https://julialang.org/downloads/support/).
The Julia page lists stable 1.13.2, Intel macOS and Windows x64 as Tier 1, Linux
aarch64 as Tier 2, and musl x86-64 as Tier 4. Do not equate Tier 2 with unsupported;
Tier 4 explicitly means currently not building. Pin the dated upstream matrix used
for each release decision. FFTW's portability is distinct from pyFFTW wheel coverage;
its [Windows installation page](https://www.fftw.org/install/windows.html) describes
native libraries callable from different compilers, without qualifying this Kokkos
integration.

**What v2/v3 mean for age and clusters.** These are cumulative instruction-set
requirements, not product dates. v2 uses SSE-era vector instructions without requiring
AVX; v3 adds AVX/AVX2, FMA and other features. As a rough chronology, v2 reaches back
to the Nehalem era around 2008–2009; v3 arrived in Intel Haswell in 2013 and AMD
Excavator in 2015. AMD Zen also meets v3. These are examples, not a rule that every
later processor qualifies; the loader still checks all required features and OS support.
See the NumPy feature table above,
[Red Hat's v3 compatibility analysis](https://developers.redhat.com/articles/2024/01/02/exploring-x86-64-v3-red-hat-enterprise-linux-10),
and [GCC's CPU feature descriptions](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

For a concrete cluster comparison, Intel's
[Xeon E5-2680 v2](https://www.intel.com/content/www/us/en/products/sku/75277/intel-xeon-processor-e52680-v2-25m-cache-2-80-ghz/specifications.html)
is Ivy Bridge, launched in Q3 2013, with AVX but without AVX2: it needs the v2 path.
The [E5-2680 v3](https://www.intel.com/content/www/us/en/products/sku/81908/intel-xeon-processor-e52680-v3-30m-cache-2-50-ghz/specifications.html)
is Haswell, launched in Q3 2014, and meets the v3 feature set. The model suffix and
instruction-set level are separate naming systems. Thus the fallback's useful cluster
population can be much older than the desktop support window.

The seven-year desktop window makes a v3 fast path a good fit for mainstream
Core/Ryzen workstations, but does not eliminate the v2 case: the 2021 N6000 example
above falls within the window. Recommend retaining the proposed v2 plus v3 design
for proving, with measured fallback slowdowns allowed. Treat model age as a support
and testing policy, not an artificial runtime rejection of an older compatible CPU.
Declare the cutoff per release rather than let it change underneath an installed
version. For older clusters, qualify the usable OS/library stack as well as the CPU:
the current dependency floor in R6 still applies even when the processor is covered.

**Available prevalence data.** The
[2025 Julia User & Developer Survey, pages 2 and 20](https://julialang.org/assets/2025-julia-user-developer-survey.pdf)
is a closer audience proxy than a general PC survey. Among 721 online respondents
recruited through the Julia community, 66% reported Linux/x86, 44% Windows, 30%
macOS/Arm, 12% Windows with WSL, 8% macOS/x86 and 6% Linux/Arm. These are overlapping
platform choices, not market shares or a Luna user census; they cannot be added to
calculate coverage. Intel Macs are now explicitly excluded by the maintainer; their
survey share is context, not an unresolved requirement. Eight percent reporting that
platform does not mean eight percent would lose all access: respondents may also use
supported machines or the NumPy backend.

The
[Steam survey for September 2026](https://store.steampowered.com/hwsurvey/?l=english),
read on 2026-10-20, reports AVX2 at 95.95%, SSE4.2 at 98.21%, FMA at 96.12% and
BMI2 at 95.91% in its combined-platform table. This is evidence of broad deployment
of those individual features among survey participants. It is a voluntary gaming
sample, not scientific users; the marginal percentages do not establish the joint
v2/v3 feature sets. In particular, “95.95% have AVX2” is not “95.95% meet v3”.
The [public PyPI dataset](https://docs.pypi.org/api/bigquery/) can support an
OS/architecture and wheel-download study, but download events are not distinct users;
caches, mirrors and automation distort counts, as
[PyPA explains](https://packaging.python.org/en/latest/guides/analyzing-pypi-package-downloads/).
No such query or solprop user census was performed in this review.

**Requested revision:** replace the unqualified population promise with a dated support
matrix and a modest evidence plan. A short survey of intended users plus the dependency
matrix is more relevant than a precise-looking global percentage. The evidence
supports expecting broad coverage, especially with v2 plus v3, but does not establish
a numeric fraction of solprop users. Exclusions allowed by the maintainer's policy
should be documented decisions, not perpetually “unresolved coverage gaps” preventing
phase 4.

#### R6 — P1: Specify deployment floors and a redistributable Windows toolchain

**Sections:** 5, 6.1, 6.5, 7, 11.

Architecture is only one compatibility axis. For example, the locked h5py wheels
already require glibc 2.28 on Linux. Building the new native library on a recent
cloud OS can unnecessarily raise the floor further through glibc or libstdc++
symbols. Likewise, testing on a recent Windows Server does not establish the oldest
supported desktop Windows version, and building against a recent macOS SDK does not
establish the intended deployment target.

The first Windows candidate needs particular care:
[Microsoft's `/openmp:llvm` documentation](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170)
still describes it as experimental and says the required Microsoft-provided libomp
DLLs are not redistributable. That does not prohibit using a suitable upstream LLVM
runtime; it means “MSVC with LLVM OpenMP” is not a sufficiently specified shipping
recipe. A successful private build is not evidence that its exact runtime can ship.

**Requested revision:** give each wheel family an OS/libc floor, Python ABI and
runtime-library recipe. State whether free-threaded CPython 3.14 is supported:
`cp314t` is a distinct qualification target. Test repaired wheels on clean machines
without the build tools or developer libraries, at the declared minimum versions.
Include the exact bundled-library versions, licence/source obligations and a
repeatable rebuild recipe. Scope Windows C/Fortran integer ABI, allocation ownership,
DLL search and OpenMP compatibility explicitly. Keep Windows Server as a reasonable
proving host, with a separate minimum-desktop-version check before claiming coverage.

#### R7 — P1: CPU distribution must preserve existing CUDA and HIP installations

**Sections:** 2, 5, 6.4, 9, 11, Q6.

Deferring GPU wheels is reasonable; deferring interaction with GPU *source builds*
is not. Today the CPU, CUDA and HIP builds use the same `solprop-kokkos`
distribution/import identity, and `auto` selects that installed package. A new required
CPU package, a new name, or a version constraint can overwrite a source-built GPU
backend, make it unsatisfiable, or install two distributions owning the same files.
Removing Numba also removes an independent compiled CPU choice in an environment
whose only Kokkos build targets a device.

**Requested revision:** define ownership and selection for CPU, CUDA and HIP before
phase 2. Decide whether variants are exclusive environments or coexist with distinct
module identities; state what `auto` means in each and how CPU execution is requested
in a GPU environment. Test install, upgrade, reinstall and uninstall around an existing
source-built device backend, including kernel-schema compatibility. Retain the
documented local CPU build for vendor BLAS and host-specific optimisation. H200 and
MI325X smoke/parity checks should protect the maintainer's existing workflows during
the packaging transition. This does not require publishing GPU wheels.

#### R8 — P1: Thread warnings cannot substitute for safe, early defaults

**Sections:** 1, 2, 5, 6.3, 7.

The current native package chooses its default thread request using `os.cpu_count()`
and initialises the runtime during import. Warnings at backend construction may be
too late to change its pool. Removing Numba and pyFFTW removes specific competing
runtimes, but not BLAS threads, OpenMP/FFTW interactions, scheduler allocations or
other extensions in the same process. The existing pool executor deliberately sets
thread limits before spawned workers import libraries; that infrastructure remains
necessary.

Also, `OMP_PROC_BIND` with multiple processes is not inherently harmful: processes
with disjoint CPU grants can benefit from binding. Merely discovering hybrid cores
does not prove the default is poor for a given workload. The ledger's measured costs
apply to its stated machines, cases and configurations, not to every warning recipient.

**Requested revision:** assign ownership of thread limits and runtime initialisation,
respect affinity/scheduler/cgroup grants before pools start, and qualify default
single-process and scan-worker configurations. Separate harmful configurations that
can actually be detected from informational advice. Test both import orders with
NumPy/SciPy, the transitional Numba extra, and representative native companions.
Retain process-allocation and BLAS control code until its remaining users are audited.
The acceptance result should show the no-settings run and its gap from a tuned run;
a warning alone does not meet the stated first-use performance goal.

**Follow-up — should native threading become the default on Linux and macOS too?**
It is a worthwhile comparison, but the Windows candidate in R14 does not establish
that conclusion. Native threading can remove solprop's OpenMP dependency and reduce
exposure to another extension's incompatible runtime or process-wide OpenMP settings.
It does not inherently produce one shared pool: Kokkos, FFTW and BLAS may still have
independent workers, waiting policies and placement.

OpenMP has practical advantages here beyond standardisation. With Kokkos and FFTW
linked to the same runtime, their parallel regions can use that runtime's worker
management and waiting controls. GNU documents
[passive waiting](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html)
and [core/socket/NUMA placement](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fPLACES.html);
[LLVM documents its runtime controls](https://openmp.llvm.org/design/Runtimes.html),
including the delay before workers sleep. Placement support is platform-dependent:
LLVM's documented `OMP_PROC_BIND` control is for Windows/Linux, so do not promise
Linux-style core pinning on Apple silicon. The existing solprop adapter also relies
on OpenMP execution instances for different operation thread counts (R14). These
are concrete capabilities to preserve or replace, not reasons to require OpenMP
indefinitely. Native pools can support such policies, but simply choosing them does
not supply solprop with equivalent controls.

Keep the measured OpenMP configurations as the initial Linux/macOS defaults and
compare (a) Kokkos and FFTW sharing one OpenMP runtime, (b) Kokkos OpenMP with native
FFTW threads, and (c) Kokkos Threads with native FFTW threads, using the same compiler
and BLAS where possible. The current native package already offers (b); its documented
affinity-inheritance caveat must be respected. Keep BLAS threading an independent
choice rather than force it onto OpenMP. Measure the same short workloads, small
transforms, large transforms, idle CPU use, interleaved Python work and parallel scan
workers on Apple silicon and Linux x86-64. The SEP 25 compiler comparison already
records severe costs from mismatched OpenMP runtimes, but supplies no Kokkos Threads
comparison. No universal speed or coexistence winner is established. Different
host execution-space builds can retain the same Kokkos kernel implementation.

#### R9 — P2: Judge the selected shipped build, with declared performance trade-offs

**Sections:** 2, 7, 9, 11.

Testing v2 on a modern machine is useful for diagnosis, but a fat wheel selects v3
there. Requiring forced-v2 performance to match host-JIT Numba on every modern case
would reject a successful delivery for a path users do not normally take. Conversely,
winning on a tuned many-core Zen 5 does not prove a good default on a laptop, a
four-core allocation, a different x86 vendor, or the new Windows toolchain.
The maintainer explicitly permits measured slowdowns on rare hardware.

**Requested revision:** predeclare which cases and machines gate mainstream performance,
the comparison window, repeats and noise criterion, and the accepted legacy exceptions.
Benchmark the actual selected wheel variant against Numba on each machine; keep
forced-v2 on a modern host as diagnostic evidence. Separate import/translation/FFT
planning/first-run latency from warm stepping and report peak memory. Include small
mode-averaged runs and concurrent scans alongside large spatial cases, and compare
adaptive cases at an independently verified accuracy. Do not hide a serious loss
behind an average speedup.

Clarify that “full comb-to-continuum propagation” in §7 means the full-resolution
case over about ten steps, not its full physical length; the opening short-window
rule and the repository's timing rule should agree. Keep the shared-machine Apple
measurements as supporting evidence, not the sole acceptance measurement.

#### R10 — P1: Correct the numerical-equivalence promise

**Sections:** 4, 7, 9.

“Results agree with today's to rounding” is too strong for independently adaptive
runs. SEP 25 distinguishes matched discrete computations/histories, which use
rounding budgets, from independent adaptive trajectories and ordinary replay, which
use an accuracy envelope against a tighter or stored converged reference.
Compiler, FFT and reduction changes can alter acceptance decisions without violating
that contract. A general rounding-level requirement would reject valid runs or
encourage inappropriate tolerance changes.

**Requested revision:** use SEP 25's distinction explicitly in theory, validation and
compatibility. Preserve exact storage/mask/identity contracts and strict protected
helpers, establish kernel and matched-step parity for each native variant, and verify
adaptive observables against the declared accuracy reference. Include resume across
supported builds as an accuracy/metadata contract, with a clear policy for files that
name the retired backend. Do not weaken existing bounds merely to pass a new compiler.

#### R11 — P2: Freeze after evidence, and make retirement a separate decision

**Sections:** 5, 6.3, 9, 11.

Section 5 freezes Numba at acceptance, whereas phase 2 is called “Default and freeze”.
Freezing at acceptance would remove feature coverage while Windows and the
distribution design remain unproved. Continuing seam changes would also make a
literally frozen optional backend unusable before its promised removal date.
“At least one release” supplies no minimum time for users to try new wheels, and
“no unresolved coverage gap” is ambiguous when some platforms are deliberately excluded.

**Requested revision:** choose one freeze point after the proving checkpoint and usable
wheels. During deprecation, retain bug fixes and compatibility with the changing
seam, while unsupported new features refuse by name. Define release milestones and
a minimum feedback period, explicit accepted exceptions, a maintained last release
with Numba or reproducible pin, and a rollback action if a wheel platform fails.
Require a recorded maintainer decision for defaulting, optionalisation and removal.
Update SEP 0, contributor instructions and feature-phase templates when that process
changes. The review's recommendation to prove the idea is not approval of removal.

#### R12 — P2: Keep package reorganisation independent and retain a compatibility check

**Sections:** 6.4, 7, 8, 11.

One repository with two distributions is a sensible provisional direction, but it is
not required to demonstrate the runtime benefit. “Released together” and “native
wheels released on their own cadence” need an explicit compatible-version rule.
Atomic source commits do not make installations atomic: users can combine old
native wheels, new Python code, editable checkouts and locally built GPU variants.
Even a single distribution can accidentally carry a stale extension during development.
The claim that one distribution needs no cross-package schema check should not be
read as permission to remove native/Python contract validation.

**Requested revision:** preserve the versioned seam and define package constraints,
build identity and import checks. Decide the repository migration separately after
the wheel proof, with source-build and contributor instructions, and avoid making it
a prerequisite for fixing default performance. Quantify maintenance savings against
the new wheel matrix, dependency rebuilds and C++ contributor setup. The Python
oracle must remain independently executable; reusing the compiled implementation
inside it would weaken the principal correctness check. Retain useful analytic,
cross-route and regression tests when deleting Numba-specific machinery.

#### R13 — P2: Qualify installed artefacts, including the absence of optional packages

**Sections:** 6.5, 7, 11.

“Each wheel passes the full suite” needs an executable matrix: which wheel hash,
baseline, OS floor, interpreter ABI and selected backend, and which expected platform
skips. A green suite with Kokkos absent or skipped would not qualify a release.
After deleting Numba, Python's 100% coverage also stops measuring the compiled path;
the native project's own tests and sanitizer checks still matter.

**Requested revision:** run the applicable existing gates against installed final
wheels, force Kokkos where it is being qualified, and fail on unexpected backend skips.
Include separate environments with Numba absent, native code absent, and the optional
Numba extra present during transition. Keep NumPy and native tests, protected
arithmetic checks, native sanitizer/static-analysis checks and scheduled performance
checks visible in the release plan. Record the allowed Windows POSIX-only skips
rather than claiming that those executors work there. Publish or retain the build
recipes, exact dependency versions, artefact hashes, supported matrix and test reports;
the private proving phase can precede this evidence without replacing it.

#### R14 — P1: Make Windows library coexistence a release requirement

**Sections:** 5, 6.1, 6.5, 7; extends R6 and R8.

**Maintainer clarification.** The Windows user should install a normal Python wheel
and work alongside their existing scientific packages without knowing how solprop
was built. No compiler, shell environment, DLL copying, special import order or
thread-runtime workaround should be part of ordinary use. This is achievable in
principle, but a successful isolated Kokkos build is insufficient evidence.

**Preferred first proving candidate, revised after the follow-up:** MSVC `cl`, the
Windows SDK and Microsoft C/C++ runtimes with release `/MD` linkage; Kokkos
`Threads`; FFTW's native Win32 threads; and a pinned private LP64 OpenBLAS build
without OpenMP. This directly addresses the maintainer's preference for ordinary
Windows Python compatibility without adding an OpenMP runtime. It is a recommendation
to test, not a qualified Windows recipe or a performance result.

This is a real library configuration: Kokkos documents
[`Threads` as a C++ threads execution space](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html#kokkos-threads)
and lists [MSVC among its compilers](https://kokkos.org/kokkos-core-wiki/get-started/requirements.html).
FFTW explicitly supports
[Win32 threads as well as OpenMP](https://www.fftw.org/fftw3_doc/Installation-and-Supported-Hardware_002fSoftware.html);
[Kokkos-FFT supports FFTW with the Threads backend](https://kokkos.org/blog/2026-kokkos-fft-v1-release/).
These component statements do not qualify their exact combined Windows build.
OpenBLAS also documents a
[native Windows ABI build](https://www.openmathlib.org/OpenBLAS/docs/install/).
The compiler used to build a private BLAS library need not be identical to the
extension's compiler when the actual C/Fortran ABI and runtime dependencies are
verified. Do not turn a preference for MSVC into a requirement to rewrite numerical
dependencies.

**Compare `clang-cl` with the same native thread libraries before adding OpenMP.**
This separates compiler/code-generation questions from thread-runtime questions.
Microsoft supports
[Clang in Visual Studio](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-170);
it uses the Microsoft standard library, and Clang targets
[MSVC ABI compatibility](https://clang.llvm.org/docs/MSVCCompatibility.html).
Compile the extension and its Kokkos C++ implementation coherently with the chosen
toolchain. A native-thread `clang-cl` wheel remains a conventional Windows binary;
the user need not install LLVM. The previous preference for `clang-cl` plus OpenMP
was too narrow: keep that combination as a measured comparison or fallback.
R6's caveat about Microsoft's experimental `/openmp:llvm` binaries still applies
to that alternative; an upstream LLVM runtime needs its own recorded build and
redistribution basis.

**Changes and evidence needed before calling this high performance:**

1. The current solprop-kokkos build rejects MSVC, forces `Kokkos_ENABLE_THREADS=OFF`,
   and selects Serial when its OpenMP option is disabled. Its FFTW configuration
   exposes OpenMP or POSIX threads, and its loader expects the CPU execution space
   to be OpenMP. Simply turning OpenMP off does not implement this candidate.
   Add an explicit host execution-space choice, the Win32 FFTW recipe and truthful
   build metadata, then prove the complete pinned Kokkos/Kokkos Kernels/Kokkos-FFT
   and nanobind extension on Windows.
2. Preserve effective thread controls. In solprop-kokkos `src/core.cpp`,
   `execution_instance(threads)` honours a reduced thread count only for the
   OpenMP CPU build; the other branch discards it. Thus a direct Threads port would
   lose the existing per-operation kernel/FFT thread selection even while reporting
   the requested values. Define how kernel, FFT and Raman limits are realised under
   Threads, verify actual execution and benchmark small FFTs as well as large ones.
3. Independent native pools still need coordination. Measure idle CPU use, pool
   handoff, oversubscription and repeated calls alternating between solprop and
   NumPy/SciPy/pyFFTW. Do not assume native workers sleep when unused, that a Kokkos
   fence suspends workers, or that OpenMP wait-policy settings control native pools.
   Establish safe concurrency and planner ownership, including calls from Python
   worker threads. Test limited CPU grants and representative hybrid-core desktops.
4. Threading and SIMD are separate choices. Avoiding OpenMP does not inherently
   remove vectorisation, but compiler/backend differences can affect the generated
   loops. Port the existing protected floating-point policy deliberately; no blanket
   `/fp:fast` to recover speed. Compare installed wheels with the same ISA, numerical
   policy, problem, planner effort and effective thread counts, using the short
   propagation windows of the benchmark policy. Report startup/planning, steady
   execution and idle/interleaved behaviour. No Windows timing was run in this review.
5. Keep numerical DLL isolation and the coexistence gates below. “Without OpenMP”
   means solprop's own dependency chain adds none; another Python package may still
   use OpenMP and must continue to work. x64 qualification does not establish native
   Windows Arm qualification.

**Alternatives to compare deliberately:**

| Candidate | Assessment |
|---|---|
| `clang-cl` with native Kokkos/FFTW threads | Closest alternative to the preferred MSVC build, retaining the Windows SDK and Microsoft runtime model. Compare the complete stack and generated-code performance without changing the threading design. |
| `clang-cl` with upstream LLVM OpenMP | Retain as a performance comparison or fallback if the native-thread candidates cannot meet the requirements. Adds runtime redistribution and coexistence obligations; its speed advantage must be measured. |
| MSVC `cl` with ordinary `/openmp` | Conventional Microsoft tooling, but its documented OpenMP 2.0 frontend must be checked against the actual Kokkos requirements. Changing the runtime alone does not add missing compiler support. Do not select it solely because CPython uses MSVC. |
| MinGW-w64 GCC targeting UCRT | Valid native Windows alternative if it wins qualification. The proposal's blanket “second C runtime” description is incomplete: modern UCRT builds use the same C-runtime family as MSVC. C++ ABI and support-library differences still matter. Confine mixed-toolchain boundaries to verified C/Fortran interfaces and preserve allocation ownership. |
| Intel LLVM with oneMKL FFT/BLAS | A credible alternative to prove, with its exact threading layer specified. Check the FFTW wrapper semantics, required CPU baseline and AMD performance; substituting oneMKL is not proof of FFTW equivalence. |
| Static private FFTW/BLAS | Can reduce numerical DLL lookup problems, at a cost in rebuilds and binary size. It does not justify statically duplicating OpenMP or the C runtime, and does not remove threading or allocator obligations. |

The [MSYS2 runtime matrix](https://www.msys2.org/docs/environments/) distinguishes
UCRT from legacy MSVCRT. Microsoft's
[DLL-boundary guidance](https://learn.microsoft.com/en-us/cpp/c-runtime-library/potential-errors-passing-crt-objects-across-dll-boundaries)
explains why memory must be released by its owning allocator and CRT objects must
not cross incompatible boundaries. NumPy buffers can be borrowed for computation
without transferring their ownership. Intel documents
[FFTW-wrapper limitations](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/using-fftw3-wrappers.html),
including no-op wisdom functions and unsupported negative strides; check the subset
solprop actually uses rather than reject or accept the provider wholesale.

**Evidence from the actual dependency wheels.** Downloaded the CPython 3.14 x64
Windows wheels named in `uv.lock`, verified their locked SHA-256 hashes, and inspected
the PE import and delay-import tables of all their `.pyd` and `.dll` files:

| Wheel | Bundled numerical libraries |
|---|---|
| NumPy 2.5.2 | `libscipy_openblas64_-327b2e0bcffce2882e0dc04cdeb4eaa6.dll` |
| SciPy 1.18.1 | `libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll` |
| pyFFTW 0.15.1 | `libfftw3-3.dll`, `libfftw3f-3.dll`, `libfftw3l-3.dll` |

The pyFFTW FFTW DLLs declare imports of `KERNEL32.dll` and legacy `msvcrt.dll`.
No separate OpenMP DLL appears in the inspected imports of these three wheels.
This is static evidence about those exact artefacts, not execution on Windows,
proof about dynamically loaded or statically incorporated code, or a promise for
other package versions and conda builds.

**Requested revision — private libraries and interoperable behaviour:**

1. Bundle solprop's numerical dependencies privately, with distinct DLL identities
   and rewritten dependent imports. Do not discover or borrow FFTW from pyFFTW or
   BLAS from NumPy/SciPy at runtime, or rely on global `PATH`. The
   [delvewheel repair mechanism](https://github.com/adang1345/delvewheel#name-mangling)
   provides the standard Windows-wheel approach to vendoring and name mangling.
   A private directory alone does not protect ordinary name-based dependency loads
   from an already loaded DLL with the same name. Account explicitly for libraries
   loaded by name at runtime, which static repair cannot necessarily discover.
2. Keep FFTW plans, buffers, planner state and cleanup within their owning library;
   never exchange plan pointers between solprop and pyFFTW. Private FFTW copies can
   coexist; using the generic names found in the inspected pyFFTW wheel risks
   accidental substitution. Do not “solve” this by requiring pyFFTW's removal.
3. Use a declared BLAS integer ABI: the current backend requires LP64 (32-bit
   BLAS indices even in 64-bit Python). NumPy's ILP64 and SciPy's LP64 libraries can
   coexist, and solprop can carry another compatible private copy. The
   [scipy-openblas maintainers](https://github.com/MacPython/openblas-libs)
   permit other projects to vendor their libraries but discourage runtime package
   dependencies and recommend exact build pins. Preserve prefixed symbols and DLL
   isolation. Prefer a build without an additional OpenMP runtime; verify the chosen
   artefact and qualify its thread limits.
4. Treat OpenMP separately from ordinary DLL isolation. First prove the native-thread
   candidate above, including BLAS without OpenMP. If an OpenMP alternative is
   selected, prefer one runtime across solprop's own Kokkos and FFTW components,
   or explicitly qualify the mixed pool design. Avoid adding OpenMP through BLAS
   unnecessarily. Unique filenames do not prove that multiple OpenMP instances cooperate.
   The [threadpoolctl discussion](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md)
   documents the general risk and specific Linux/macOS failures; it does not establish
   a Windows failure here. Test the actual Windows combinations. Static OpenMP linking,
   `KMP_DUPLICATE_LIB_OK`, or silently altering another package's runtime are not
   acceptable substitutes for that evidence.
5. Qualify installed repaired wheels on clean desktop Windows with python.org Python
   in a virtual environment, and in representative conda environments using OpenBLAS
   and MKL. Exercise both import orders and interleaved real FFT/BLAS/propagation
   calls with NumPy, SciPy and pyFFTW; include another OpenMP extension such as
   scikit-learn. Check numerical results, process stability, loaded DLL paths,
   concurrent/threaded calls and performance, not just imports. Verify that later
   NumPy/SciPy calls still work and their settings have not been permanently changed.
   Include clean machines without Visual Studio, LLVM, MSYS2 or oneAPI, and resolve
   any needed redistributable runtime through the documented wheel/deployment recipe.

**Recommended decision sequence.** Resolve the installation model and audit functional
gaps first; prove redistributable wheels on the intended CPU platforms and preserve
device-source installation; then assess the actual performance and operational cost.
A v2 fallback is a reasonable candidate, with measured legacy slowdowns accepted as
the maintainer requested. There is no evidence here that preserving a complete second
compiled backend indefinitely is necessary just for obscure hardware. There is equally
no basis yet to remove it before its useful capabilities and dependency role have
replacements.

### Response to review — 2026-10-22

The revision of 2026-10-22 changes the proposal's scope: the native backend moves into
solprop's repository as a second distribution and is published as wheels, while the
retirement of `numba_cpu` leaves this SEP for a later decision (§6.3). Several findings
were about that retirement; they are carried forward rather than resolved.

- **R1 (capabilities).** Taken in this SEP. The audit the review asked for was made on
  2026-10-22 (§6.6): the native CPU backend lacks only the general z table (G1) and Adams
  history export (G2), and phase 2 implements both. TDSE and CF4 are reference-only on
  both compiled backends, so they are not losses. The removal itself stays with the later
  SEP (§6.3).
- **R2 (shared Numba helpers).** Taken in this SEP (§6.6, phase 2). `numpy_ref` keeps the
  plain-Python forms, and native runs call C++ versions, so neither depends on Numba.
  `numba` stays a dependency of `solprop` while `numba_cpu` exists.
- **R3 (installation before fallback).** Taken. The native code is an extra, not a
  required dependency (§5), so a platform without a wheel installs `solprop` normally;
  the extra's behaviour without a wheel is Q7. Explicit `backend="native"` never falls
  back, a below-baseline processor raises a clear error, and run metadata records the
  selected backend and build (§5). Installation tests are in §7.
- **R4 (the whole wheel safe).** Taken. Phase 0 qualified the v2 extension and its
  bundled libraries under an instruction emulator, and the OpenMP OpenBLAS is requalified
  (Part II). A loader check on every platform, refusing before native code runs, is part
  of phase 2 (§5, §7).
- **R5 (population).** Answered by phase 0's population study (Part II): v2, with no v1.
- **R6 (deployment floors and the Windows toolchain).** Taken: manylinux_2_28 containers,
  an explicit macOS deployment target with `libomp` and FFTW built for it, the phase 0
  Windows recipe on upstream LLVM `libomp` rather than Microsoft's experimental
  `/openmp:llvm`, a minimum desktop Windows check, and no free-threaded CPython in the
  first release (§6.1, §6.5, §7).
- **R7 (device installations).** Taken: device builds are source builds of the same
  `solprop-native` distribution, alternative to the CPU wheel in one environment (§6.4).
  The transition is tested on H200 and MI325X source builds before the rename lands.
- **R8 (thread defaults).** Taken, and partly done: the native backend sizes its pools by
  operation and process grant, and fits its BLAS to its pool at import, before the
  backend is constructed (Part II). The warnings are cut to the detectably harmful
  settings (§5). The comparison of OpenMP against native threading on Linux and macOS is
  not pursued: one OpenMP runtime measured best wherever it was tried.
- **R9 (judge the shipped build).** Taken: the standard suite's core cases at one thread
  and the production width, native against `numba_cpu` on the same machine, per platform
  family, with losses reported per case (§7). Phase 0 timed the shipped v2 build, not a
  forced variant.
- **R10 (numerical equivalence).** Taken: §4, §7 and §9 now use SEP 25's two kinds of
  bound.
- **R11 (freeze after evidence).** Taken in full: no freeze in this SEP; the later SEP
  decides on evidence from the wheels in use (§6.3, §11).
- **R12 (package reorganisation).** Taken in part. The maintainer chose to make the
  reorganisation phase 1 rather than defer it, because the release is an opportunity to
  set the package names once; the kernel schema check and the independent oracle are
  kept, and the version rule is explicit (§6.4).
- **R13 (qualify installed artefacts).** Taken: §7 tests repaired wheels on clean machines,
  forces the native backend, and fails on unexpected native skips.
- **R14 (Windows coexistence).** Taken as a release requirement (§6.1, §6.5, §7). Phase 0
  measured the review's preferred native-thread candidate (Kokkos `Threads`, Win32 FFTW
  threads) and the single-OpenMP-runtime build; the single runtime was faster on the
  modal and wall cases, so it is the recipe, with `Threads` as the fallback. Bundled DLLs
  are name-mangled by delvewheel.

### AI re-review of the revised scope — 2026-10-22

**Reviewer:** GPT-6 (AI). Light re-review of Part I and the response above, checked
against the current registry, shared helpers, Adams persistence and native loader.
Parts I and II are unchanged by this review. The phase 0 measurements are treated as
reported evidence; no native builds, propagation runs or benchmark campaigns were repeated.

**Assessment.** Support the revised direction: one repository, two distributions,
native CPU parity, and a later decision on `numba_cpu`. Keeping Numba supported
removes the original retirement risk, and the measured single-runtime recipe and
accepted v2 cost do not need reopening. The remaining recommendations concern
implementation and release criteria. R15 is a release blocker until resolved or
explicitly accepted as a platform limitation; R16–R18 need concrete decisions or
tests in their owning phases, rather than a larger redesign.

#### R15 — P1: Include macOS in the coexistence release gate

**Sections:** 2, 5, 6.5, 7; follows up R8 and R14.

Section 7 names Windows alone for coexistence, but the current native loader in
`solprop-kokkos/src/solprop_kokkos/__init__.py` explicitly raises on macOS when
`pyfftw` is already imported. Its `test_pyfftw_first_is_refused_before_openmp_initialization`
asserts that refusal, and its test-session setup deliberately imports Kokkos first.
Thus a passing existing suite does not establish ordinary import-order independence.
With `numba_cpu` and pyFFTW retained, this remains a normal scientific environment,
not a configuration eliminated by this SEP. A single runtime within the native
wheel does not by itself resolve another wheel's bundled runtime.

**Recommendation:** qualify the repaired macOS wheel with the ordinary installed
NumPy, SciPy, pyFFTW and Numba packages, in fresh subprocesses for both import orders
and interleaved short backend runs. Include the same check on Linux. Resolve the
macOS restriction before claiming coexistence, or explicitly record its support
limitation and user-visible behaviour. Do not merely remove the guard: demonstrate
safe execution with the actual bundled libraries. This finding is based on code and
existing tests, not a newly reproduced runtime failure.

#### R16 — P2: Specify ownership of the shared-helper implementation

**Sections:** 6.4, 6.6 (R2), 7, 11 phase 2.

The independent Python oracle and native helper replacements are appropriate, but
the selection mechanism is missing. Today `clenshaw`, table evaluation and
`integrated_weights` choose their cached Numba helpers without a backend argument;
translation also calls them before constructing the backend. Choosing native helpers
merely because the package is installed would make `numpy_ref` use the code it is
meant to check. A process-wide replacement would also let one simulation change
another's implementation.

**Recommendation:** state where the helper provider is selected and passed, including
standalone translation, the NumPy oracle and mixed-backend use in one process. Keep
the existing Numba path for `numba_cpu`. Add a small test that NumPy remains independent
with the native package installed, and that native execution does not reach these
Numba dispatchers. If the chosen design changes a protected driver hook or stage
bundle, apply the normal kernel-schema rule; one repository does not remove that
contract. Settle this before G1 builds on the helpers.

#### R17 — P2: Verify that Adams history is actually restored

**Sections:** 6.6 (G2), 7.

The proposed resumed-versus-uninterrupted accuracy check can pass even when the
history is discarded: G2 correctly says the present restart is already valid under
that accuracy contract. It therefore cannot alone verify the new capability.
`AdamsMarch.snapshot` currently returns `None` for an engine without `snapshot_rows`,
and `Simulation._continue_march` returns immediately when the saved march is absent.

**Recommendation:** require a non-empty native march snapshot after history has
formed, successful restoration without the restart path, and preservation of its
stored rows, positions, slots, identities, order and controller state. Exercise the
compensated history and the periodic snapshot path as well as ordinary HDF5 resume;
test incompatible-history handling explicitly. Exact checks belong on the lossless
stored/restored state. Keep the proposed accuracy envelope for subsequent adaptive
propagation; this is not a request for identical trajectories or work counts.

#### R18 — P2: Make Q7's wheel-only installation promise executable

**Sections:** 5, 6.4, 6.5, 7, Q7; follows up R3.

An extra adds dependencies; it does not carry an installer policy forbidding source
builds. If a matching `solprop-native` sdist is published alongside the wheels, an
ordinary install can select it when no wheel matches. This follows from the
[dependency specification](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
and pip's separate
[`--only-binary` option](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-only-binary).
The choice is still open in Q7, although §5 reads as a settled guarantee.

**Recommendation:** resolve Q7 in the wheel-release phase: for example, publish only
native wheels to the index and document explicit source builds from the versioned
repository/source archive, or document a binary-only installer command if native
sdists are published. Test the chosen policy with all intended release artifacts
present, not only a wheelhouse from which sdists are absent. Also define the
`auto` boundary precisely: §5's “installed and loadable, else” is broader than the
current registry's absence-only fallback. A schema mismatch or failed native load
must not silently select another backend. Update Q7's phase reference from phase 2
to phase 3 when the author incorporates this clarification.

### Response to re-review — 2026-10-22

All four findings are taken; none changes the direction. Part I is revised as follows.

- **R15 (macOS coexistence).** Taken. §7's coexistence gate now covers every platform
  family with NumPy, SciPy, pyFFTW and Numba installed, in both import orders, with
  interleaved short runs. The macOS loader's pyFFTW-first refusal is to be resolved with
  evidence in phase 3, or documented as a macOS limitation; it is not removed without
  that evidence (§6.5, §7).
- **R16 (helper ownership).** Taken (§6.6). Translation keeps one shared reference
  implementation. The per-stage and per-step evaluations take their evaluator from the
  backend that runs the stage, and nothing switches process-wide. A test keeps
  `numpy_ref` independent of the native package, and native runs independent of the Numba
  dispatchers, in one process. The schema rule applies to any protected interface the
  design changes. The design is settled before G1.
- **R17 (restored history).** Taken (§7). G2's acceptance now checks the lossless state
  exactly: a non-empty snapshot, a restore without the restart path, and the stored rows,
  positions, slots, identities, order and controller state, on the compensated history,
  the periodic path and HDF5 resume, with incompatible histories refused. The propagation
  after the restore keeps the accuracy envelope.
- **R18 (wheel-only installation).** Taken (§5, Q7). The proposal is to publish only
  native wheels, with no source distribution, to the index, with source builds documented
  from the repository. It is confirmed in phase 3 and tested against the real release
  artefacts. §5 now states the `auto` boundary precisely: it falls back to `numba_cpu`
  only when the native package is absent, and a failed load or schema mismatch raises.
