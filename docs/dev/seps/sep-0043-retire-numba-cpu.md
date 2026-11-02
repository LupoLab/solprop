# SEP 43 — Retire `numba_cpu`

| | |
|---|---|
| **SEP** | 43 |
| **Title** | Retire `numba_cpu` |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-11-02) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-11-02 |
| **Resolution** | — |
| **Requires** | SEP 29 (compiled backend distribution; final) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

This SEP removes the `numba_cpu` backend and solprop's dependencies on `numba` and
`pyfftw`, and makes `native` the only compiled CPU backend. It first widens the native
wheels to the processors and platforms that only Numba reaches in practice: x86-64
processors below AVX2, on Linux and Windows, and Windows on Arm. The documentation build
and the continuous integration lanes that run on `numba_cpu` today move to the native
wheel. Success means: the native suite is green on all of those targets, a Numba-free
install imports and runs on `auto`, `native` and `numpy_ref`, the coverage gate holds
without Numba's paths, and no result changes beyond the accuracy bounds of §7.

### 2. Motivation and scope

Every test, route and kernel of `numba_cpu` is a second copy of what `native` does. The
0.1.0 benchmark record ([#68][68]) has it 1.5 to 2.8 times slower than native by geometric
mean, at the same step counts, with no configuration in which it wins (§3). Its remaining
value is reach, and that reach turns out to be narrow and cheap to replace.

Nobody depends on it yet. 0.1.0 was published on 2026-11-01 and has not been announced, so
the freeze, deprecation and feedback period that SEP 29 §6.3 anticipated would protect no
one, and this SEP does without them (§9).

**In scope:**

- a second, x86-64-v2 build of the extension in the Linux and Windows x86-64 wheels;
- a Windows on Arm wheel;
- moving the `docs`, `slow`, `oldest` and coverage lanes to the native wheel;
- replacing the Numba-compiled helpers that every backend shares (SEP 29 R2);
- removing the backend, `numba`, `pyfftw`, and the tests and golden files only they use;
- a single stable-ABI native wheel for Python 3.14 and 3.15, measured first.

**Out of scope:**

- any further performance work on `numba_cpu`: [#104][104], [#305][305], [#306][306] and
  [#308][308] close as not planned when this SEP is accepted;
- wheels for Intel macOS, ppc64le or musl, and processors below x86-64-v2 or Neoverse N1;
- GPU wheels ([#317][317]) and CUDA or HIP source builds, which are unchanged;
- changes to `numpy_ref` or to any numerics.

### 3. Context

**Backends.** `numpy_ref` is the reference and `native` the production backend (Tier 1,
[ADR-061](../decisions.md)). `numba_cpu` is Tier 2, and `backend="auto"` selects it when
the native package is absent or the processor is below the wheel's baseline
(`src/solprop/backends/registry.py`). [SEP 29](archive/sep-0029-compiled-backend-distribution.md)
§6.3 left its removal to a later SEP, which should settle the freeze, feedback and
rollback of its review item R11, the test suite's cross-backend tests and the two
dependencies. [#316][316] tracks that decision.

**Speed.** `numba_cpu` over `native` in the 0.1.0 history (`benchmarks/history/`), geometric
mean over the 22 configurations, all at the same step counts:

| Machine | One thread | Full width |
|---|---|---|
| 2× Intel Xeon Gold 6240 | 2.19 | 2.84 |
| AMD EPYC 7763 | 1.95 | 2.39 |
| Apple M5 Max | 1.53 | 2.11 |

**Reach.** What the Numba project's own distributions add to the native wheels' four
platforms, from PyPI and conda-forge on 2026-11-01:

| Target | Numba | solprop on it today | After this SEP |
|---|---|---|---|
| Windows on Arm | PyPI wheel | Blocked: pyFFTW has no wheel | Native wheel |
| Intel macOS | conda-forge only | solprop is not on conda-forge | `numpy_ref` where it installs |
| ppc64le | conda-forge only | solprop is not on conda-forge | `numpy_ref` where it installs |
| x86-64 below AVX2 | Compiles for the host | Runs on `numba_cpu` | Native x86-64-v2 build |
| Arm below Neoverse N1 | Compiles for the host | Runs on `numba_cpu` | `numpy_ref`, documented as dropped |

Numba's real advantage is that it has no fixed processor baseline. A second build at
x86-64-v2 removes most of that advantage at a measured cost (§6.1).

**Features.** The admission code and the SEP 29 capability audit (R1, closed by its native
parity phase) show no CPU problem kind that only `numba_cpu` runs. This is a reading of
those checks rather than a proof over every path; the test retargeting of phase 2 is
where a gap would appear, and any it finds goes into Part II.

**Issues.** It closes [#316][316], [#104][104], [#305][305], [#306][306] and [#308][308].
It answers the Windows on Arm half of [#317][317], and the pyFFTW half of [#319][319].

### 4. Theory

No physics changes. Comparisons follow SEP 25's distinction: matched discrete computations
agree to rounding-level bounds, and adaptive runs within an accuracy envelope.

### 5. Proposed interface

- `backend="numba_cpu"` raises a `BackendError` naming `native` and `numpy_ref`, and the
  release that removed it, in place of today's unknown-backend message. Old output and snapshots that record `numba_cpu` as their
  backend still load.
- `backend="auto"` picks `native` when it is installed and the processor is at or above
  the wheel's floor. Otherwise it picks `numpy_ref` and warns, stating the cost: 6 to 40
  times slower than threaded native (Q1).
- The x86-64 wheels hold both builds and choose at import. `SOLPROP_NATIVE_CPU_VARIANT=
  x86-64-v2` asks for the lower build, for tests and comparisons; a higher build than the
  processor runs is refused by name, as an unknown name is. `solprop_native.build_info()`
  and the benchmark build string report which build is loaded.
- `SOLPROP_NUMBA_*` and `SOLPROP_PASS_TIMERS` go; `SOLPROP_FFTW_PLAN` stays, since native reads it. Native has Kokkos
  regions for what the pass timers measured.

### 6. Implementation strategies

#### 6.1 x86-64-v2 beside x86-64-v3

The Linux and Windows x86-64 wheels carry `_core` at x86-64-v3 and `_core_v2` at x86-64-v2,
side by side. `solprop_native` chooses before it loads and registers the chosen one as
`solprop_native._core`, so nothing else knows which it is. CPython finds `PyInit__core`
from the module's name rather than the file's, so the second build needs no C++ change, and
`auditwheel` and `delvewheel` repair both alike. On Linux both link the one OpenBLAS with
run-time kernel selection. On Windows each links its own fixed target, `HASWELL` and
`NEHALEM`, because OpenBLAS's run-time selection, built by Clang for Windows, gave wrong
complex results and an access violation (`native/ci/build-deps-windows.ps1`).

The alternatives are in §10. Measured cost and correctness are in §7 and Part II.

#### 6.2 Windows on Arm

The wheel is built on `windows-11-arm` with Clang on the MSVC ABI and the runner's LLVM
libomp, as on x64. FFTW's CMake build has no NEON option, so the build recipe patches it in.
OpenBLAS is built for `NEOVERSEN1`. Kokkos and Kokkos Kernels each pass the x86-only
`-mcx16` to Clang on Windows, and are patched at fetch. The core keeps the compiler's
generic Armv8-A baseline, so no processor check is needed on this platform.

#### 6.3 The shared helpers

`translate/chebyshev.py` and `stepper/multistep.py` import Numba lazily, and `helpers=NUMBA`
is a default in `translate/stage.py`, `backends/exponential.py`, `backends/multistep.py`
and `backends/driver.py`. These become plain forms, keeping the protected arithmetic that
SEP 29 R2 asks for. Uncompiled forms cost 6 to 36 times per call, on a path that matters
mostly during setup and on `numpy_ref`. The history's `construction_s` is the gate, and a
vectorised NumPy form is written only where it fails. A test imports solprop with Numba
blocked.

#### 6.4 Tests

Native-against-Numba comparisons move to `numpy_ref`, which about 40 of the test modules
importing `numba_cpu` already use as well. The 4 modules that run only on Numba are ported.
The 28 `test_numba_*` modules and 7 Numba golden files go. Parametrizations drop the
`numba_cpu` value. Convergence records keep the rows Numba wrote, as evidence of what it
measured, through `RETIRED_BACKENDS` in `tests/test_convergence_records.py`, as they already
do for `numba_cuda`.

#### 6.5 Documentation and continuous integration on native

Moving these lanes was rehearsed on today's tree before anything was removed (Part II):

| Lane | Today | After | Rehearsal |
|---|---|---|---|
| `docs` (executes the lessons and tutorials) | `numba_cpu` | CI-built wheel | Passes under `-W`: 37 of 37 scripts, 5.3 min and 5.5 GB peak, against 8.4 min today |
| `slow` | `numba_cpu` | CI-built wheel, `SOLPROP_TEST_BACKEND=native` | 164 passed, 1 skipped, 22.5 min on the default runner, including the statistics cost budget |
| `oldest` | `numba_cpu` | Lowest dependencies, wheel installed `--no-deps` | All 3 shards green |
| `coverage` | Native wheel, driver tests on `numba_cpu` | Native wheel, driver tests on `native` at the removal, when the gate can pass | Report-only on the whole default lane: 99.87 % of what stays, with 44 lines and 32 branches missed (Part II) |
| New: `no-native` | — | No native package, a few default-lane modules on `numpy_ref`; checks `auto`'s warning and a real run | Measured locally: 193 passed in 23 s |

**The wheel on documentation-only changes.** `native-wheel` runs only when `lanes == full`
today, and `docs` runs on documentation-only changes as well, so `native-wheel` must run
for those too, on Linux only. It is cached by the hash of the native sources, so nearly
every run reuses a wheel. A cold Linux build takes 13 to 22 minutes. A documentation-only
change pays that only when the last change to `native/` merged without a push to `main`
building it: such a merge runs no lanes, so it leaves no cache for other pull requests.

**Read the Docs stays free of native.** It executes no gallery script, because all gallery
output is committed (`.readthedocs.yaml`, `docs/conf.py`). It needs only a clean build after
the removal.

**What ADR-061 said.** Its consequence "the documentation build installs no native backend:
a lesson or tutorial of a feature that `numba_cpu` does not run names `numpy_ref`" no
longer holds for the CI lane, and the new ADR supersedes it. A lesson may now use anything
`native` runs.

**Committed output.** `plot_11_dispersive_kerr` and `plot_05_moving_solid_guide` select
`numba_cpu` explicitly and move to `native`. Four more gallery scripts mention it in prose.
Regenerating their committed output is part of the removal, since the build refuses stale
output. `poe docs-examples` runs both on `native` on the 4-core runner: `plot_11` in 16 s
and `plot_05` in 884 s, which is a one-off cost when that script changes. The removal
compares their regenerated figures and printed values with the committed ones.

**Plain `uv sync`.** Without Numba, a compiler-free `uv sync` gets `numpy_ref`. A new
`poe fetch-wheel` installs the wheel CI built for the newest commit in the checkout's
history, using `gh`. That wheel's artifact retention grows from 3 to 14 days to match.

#### 6.6 Python 3.15

The native package moves to CPython's stable ABI: nanobind's `STABLE_ABI`, with
`wheel.py-api = "cp314"` in scikit-build-core. There is then one wheel per platform for 3.14,
3.15 and later, tagged `cp314-abi3`, and no build per interpreter, which is what makes a new
Python cheap. Nothing in the bindings goes beyond that ABI, so no binding code changes.
Part II has the measurements: `abi3audit --strict` is clean on all five wheels, CPython 3.15
loads each, and the wheel costs nothing against the version-specific one.

solprop's own support needs three more things:

- **h5py** is the one runtime dependency without a 3.15 wheel (h5py issue 2993). It builds
  from source against a system HDF5, as the Linux lane does, so the installation page says
  that until h5py publishes wheels, and a 3.15 lane proves solprop in the meantime.
- **The TDSE provider.** `driver/tdse.py` refuses Python other than 3.14, the version
  Pyroraptor is pinned to. That stays, and TDSE needs 3.14 until the provider moves. Four
  `test_tdse_binding` tests are gated on it.
- **Source builds** keep `find_package(Python 3.14 EXACT)` until a source build on 3.15 has
  been tried, which phase 4 does.

This does not block the removal, and the removal clears Numba and pyFFTW out of its way.

#### 6.7 Layer, contract and cost

This is a backend and packaging change. `KERNEL_SCHEMA_VERSION` does not change, and no
kernel changes. Wheels grow, Linux from 16.5 to 19.2 MB and Windows from 5.2 to 9.7 MB. The Windows on Arm
wheel is 4.7 MB. solprop drops two runtime dependencies. In the release wheels run, the
Windows on Arm build takes about 15 minutes and its default lane about 30. The x86-64
builds stay within the 19 to 30 minutes that they vary over on the hosted runners (22 on
each in a recent run without the second build). The forced-v2 lanes add about 50 minutes
each, and they run only in that workflow, not in `ci.yml`.

### 7. Verification and validation

| Claim | Check | Bound |
|---|---|---|
| Windows on Arm works | Default lane on `windows-11-arm`, native | Green |
| Selection is right | Intel SDE: Haswell gets v3; Nehalem and Sandy Bridge get v2 and `resolve_name() == "native"`; Penryn is refused with status 1 and no emulator error | Exact |
| v2 is correct | Default lane with v2 forced, Linux and Windows; v2 against v3 on the tracking set | Same step sequences; fields within about 1e-10, anything above it flagged |
| v2's cost is acceptable | Tracking set, Xeon Gold 6240 and EPYC 7763, interleaved fresh processes; native v2 against Numba compiled for Nehalem | v2 is not slower than Numba at v2 |
| Windows v2's BLAS | Coupled TE/TM at one thread and the radial real GEMM, which failed under OpenBLAS's run-time selection on Windows | Green with v2 forced |
| Numba-free install | Fresh environment with no `numba` or `pyfftw`: import, and a short run on `auto`, `native`, `numpy_ref` | Runs |
| Coverage | The 100 % gate on `solprop`, without Numba's paths | Holds, or the gaps are reported rather than the gate lowered |
| No regressions | `python -m benchmarks track` on the final tree against history; ledger entry | Same step counts |
| Removal is complete | A search finds Numba only in Final SEPs, convergence provenance and the tombstone | Exact |
| Python 3.15 | Native wheel on 3.15 on all five platforms; Linux default lane; 3.14 against 3.15 on the tracking set | Green; rounding-level |

### 8. Documentation

`introduction/installation.md` and `guide/performance/backends.md` state the targets
and the dropped ones; `guide/performance/native.md` the two builds and the setting;
`dev/backends.md`, `dev/tooling.md`, `AGENTS.md`, `native/AGENTS.md`, `dev/testing.md` and
the guide pages that name `numba_cpu` lose it; `CHANGELOG.md` records the removal. Final
SEPs stay as history.

### 9. Compatibility and impact

**R11's milestones, answered directly.** There is no freeze and no feedback period, because
there are no users to give feedback: 0.1.0 is a day old and unannounced. 0.2 is the last
release with `numba_cpu`. Rollback is pinning 0.2 or reverting the single removal pull
request. This overrides the advice in [#316][316] to wait two minor releases, and is
recorded as the maintainer's decision.

**Accepted losses:**

| Who | Before | After |
|---|---|---|
| Arm below Neoverse N1, Windows before version 2004, Intel macOS, ppc64le | `numba_cpu` where Numba installs | `numpy_ref` where it installs |
| x86-64 below x86-64-v2 (Intel before 2008, AMD before 2011) | `numba_cpu` | `numpy_ref`, unthreaded |
| Any CPU inside a GPU build | `numba_cpu` in process | A second environment with the CPU wheel, or `numpy_ref` in process |
| x86-64 below AVX2 | Numba compiled for the host | Native v2, in 0.44 to 0.51 of Numba's time |

**Other effects.**

- A script that names `numba_cpu` fails with a message that says what to use. Its output
  files still load.
- Validated results are unaffected: `numba_cpu` was never the reference.
- No difference from Luna.jl.
- `KERNEL_SCHEMA_VERSION` and the output schema do not change.

### 10. Alternatives considered

- **Keep `numba_cpu` in Tier 2 or move it to Tier 3.** It stays cheap only while nothing
  changes the seam; every kernel-contract change and every parity test still pays for it,
  and its reach is what §6 replaces.
- **Lower the single wheel to x86-64-v2.** Simple and one build, at 19 to 25 % by geometric
  mean, and up to about 50 % at one thread, for everyone with AVX2, to serve processors without
  it: mostly from before 2013, and later low-power Atom and Pentium parts. Rejected.
- **Runtime dispatch inside one build.** The pool of kernels spread through Kokkos, SLEEF
  and FFTW makes it far more invasive than a second build, for a smaller wheel only.
- **MKL on Windows x86-64.** MKL runs below AVX2 and would settle the BLAS question for
  Windows v2, as it was already kept for local builds in SEP 29 (it has no aarch64 or macOS
  builds). Bundling it means a large binary, Intel's licence, and a third BLAS in the
  distribution matrix while Windows on Arm still needs OpenBLAS. Rejected by the maintainer.
- **Leave Windows v2 out until someone asks.** Considered first. The fixed `NEHALEM`
  OpenBLAS costs one more dependency build and passes the suite, so the exception is not
  worth keeping.

### 11. Phases

Each is a separate pull request. Phases 1 and 2 are tooling and tests, which SEP 0 §2 lets
through without this SEP's single pull request, and they keep the last phase mechanical
and revertible.

1. **Wheels and lanes.** The spike's two builds and Windows on Arm, cleaned up, with
   `cpu-floor` extended. `docs`, `slow` and `oldest` on the wheel, `native-wheel` on
   documentation-only changes, the `no-native` lane.
2. **Helpers and tests.** `helpers=NUMBA` to plain, with the import-blocked test and the
   `construction_s` gate. Comparisons with Numba retargeted, with Numba still present.
3. **Removal**, `feature/sep-0043-retire-numba-cpu`, one revertible pull request. Delete the
   backend, `backends/bodies.py`, `_numba_cache.py`, the tests and golden files. Registry
   tombstone. Remove the dependencies and regenerate `uv.lock`. Documentation. New ADR
   superseding ADR-061's Tier 2 and its statement on the documentation build. `CHANGELOG.md`.
4. **Python 3.15.** The stable-ABI wheels and a Linux 3.15 lane. May land before phase 3.

Whether phase 3 merges before or after the 0.2 cut is the maintainer's call. 0.2 is meant
to be the last release with `numba_cpu`.

### 12. Open questions

- **Q1. What `auto` does without native.** Recommended: `numpy_ref` with a warning that
  states the cost. An error would be more honest about 6 to 40 times, and would break
  every first-run script on a platform with no wheel.
- **Q2. Whether native becomes a default dependency on wheel platforms.** Plain
  `pip install solprop` would otherwise land on `numpy_ref`, which is the wrong first
  impression. Recommended: yes, through environment markers on the wheel platforms. It
  conflicts with a compiler-free `uv sync` in the workspace, since a workspace member would
  then build from source, so it needs a uv source override to the published wheel or a
  dependency group. To settle before phase 3.

### 13. References

Issues and pull requests of the solprop repository:

[68]: https://github.com/LupoLab/solprop/pull/68
[92]: https://github.com/LupoLab/solprop/issues/92
[104]: https://github.com/LupoLab/solprop/issues/104
[305]: https://github.com/LupoLab/solprop/issues/305
[306]: https://github.com/LupoLab/solprop/issues/306
[308]: https://github.com/LupoLab/solprop/issues/308
[316]: https://github.com/LupoLab/solprop/issues/316
[317]: https://github.com/LupoLab/solprop/issues/317
[319]: https://github.com/LupoLab/solprop/issues/319

1. [#68][68], the 0.1.0 benchmark record.
2. [#316][316], the decision on `numba_cpu`.
3. [#317][317], Windows on Arm and GPU wheels.
4. [#319][319], Python 3.15; [#92][92], the question of capping `requires-python`.
5. x86-64 microarchitecture levels: *System V Application Binary Interface, AMD64
   Architecture Processor Supplement*, version 1.0.

## Part II — Implementation record

### Phase 0 report — the coverage spike

- **Scope:** 2026-11-01 to 2026-11-02, Claude Opus 5.5 for John C. Travers, on the
  unmerged branch `spike/native-coverage` from `05ee1f3`. Nothing here is merged; phase 1
  carries it over, cleaned up. It was run before the proposal was written, to find out
  whether the proposal was possible.
- **Delivered, as a working branch:** all five platforms build stable-ABI wheels through
  `wheels.yml`, with two extensions in the x86-64 wheels. New lanes force v2, rehearse `docs`,
  `slow` and `oldest` on native, and run 3.15.
- **Verification on hosted runners:**

  | Lane | Result |
  |---|---|
  | Windows on Arm wheel | Built and repaired, 4.7 MB; FFTW with NEON, OpenBLAS `NEOVERSEN1` with OpenMP |
  | Default lane on `windows-11-arm`, native (see below) | 15,478 passed; every one of 234 failures and 10 errors stops in `numba_cpu` at the empty `pyfftw` |
  | Default lane, v2 forced, Linux | 17,902 passed, 44 skipped |
  | Default lane, v2 forced, Windows | 17,824 passed, 122 skipped, which includes the two known cases of §7 |
  | Intel SDE | Haswell gets v3; Nehalem and Sandy Bridge get v2; Penryn is refused with status 1 |
  | `docs` on native | §6.5 |
  | `slow` on native | 164 passed, 1 skipped, 22.5 min |
  | `oldest` on native | All 3 shards green |
  | Stable ABI | `abi3audit --strict` clean on all five wheels; CPython 3.15.1 loads each |
  | Default lane on Linux, Python 3.15, native | 15,528 passed; failures as on Windows on Arm, plus 4 at the TDSE pin (§6.6) |

  pyFFTW has no Windows on Arm or 3.15 wheel, so these two lanes install an empty `pyfftw`
  and pass `-k "not numba"`, so that modules which import `numba_cpu` at the top still
  collect. What remains reaches Numba only through tests with no `numba` in their names.
  These lanes will be exact once phase 2 retargets those tests.

- **What Numba's retirement leaves untested** (report-only, EPYC 7763, the default lane on
  the wheel with `SOLPROP_TEST_BACKEND=native`, an empty `pyfftw` and `-k "not numba"`,
  coverage omitting `numba_cpu`, `bodies.py` and `_numba_cache.py`): 15,559 passed,
  234 failed and 10 errors, all stopping at the empty `pyfftw`, in 31 modules. The numbers
  match Windows on Arm's. 150 of the 244 are in three modules. Statement and branch
  coverage was 99.87 %, missing 44 of 45,046 lines and 32 of 13,374 branches in 18 files.
  The count is before phase 2 ports those tests.
- **The default lane with no native package** (report-only, EPYC 7763, the driver tests on
  `numpy_ref` and `-k "not numba"`): 14,355 passed in 4.1 min on 48 workers, about 85
  CPU minutes. Its 3 failures and 144 errors are from the copied tree having no git history
  and the node's allocation, and none comes from `numpy_ref`. That is too large for one more
  lane of every change, so `no-native` holds a few modules.
- **Cost of v2** (the tracking set, 22 configurations, length 1.0, 3 rounds, fresh processes
  interleaved, 16 threads pinned to one socket; every configuration took the same steps):

  | Comparison | Xeon Gold 6240 | EPYC 7763 |
  |---|---|---|
  | Native v2 over native v3, geometric mean | 1.246 | 1.193 |
  | The same, at one thread / at 16 | 1.34 / 1.17 | 1.30 / 1.11 |
  | Native v2 over Numba compiled for Nehalem | 0.444 | 0.511 |
  | The same, slowest configuration | 0.931 | 0.971 |
  | Numba for Nehalem over Numba for the host | 1.034 | 1.013 |

  Native at v2 beats Numba at v2 in all 22 configurations on both machines. Numba loses so
  little to the lower target because FFTW chooses its kernels at run time and Numba's own
  kernels vectorise little.

- **Accuracy of v2.** Final fields against v3 are at or below 9.2e-11 on every configuration,
  except `he_rdw_averaged` at one thread on the Xeon, at 3.1e-10. That case and `ctc` have a
  floor that high: two v3 builds differing only in their Python ABI differ by 3.8e-10 and
  2.0e-10 on them, so this is not a property of the v2 build.
- **Cost of the stable ABI.** `cp314-abi3` over `cp314-cp314`, both v3 on Python 3.14, on the
  EPYC: geometric mean 1.0017, same steps. The same wheel on CPython 3.15.2 over 3.14.7, with the
  same NumPy and SciPy, on the Xeon: geometric mean 1.0004, same steps, fields at or below
  1.3e-10.
- **Deviations from the first plan:** the builds sit side by side as `_core` and `_core_v2`
  rather than in subdirectories; the Windows on Arm core keeps the generic Armv8-A baseline,
  so it needs no `PF_ARM_*` check.
- **Sources checked:** Numba, pyFFTW and h5py availability against PyPI and conda-forge on
  2026-11-01 and 2026-11-02.
- **Limitations:** Windows and Windows on Arm timings are not recorded, since hosted runners
  are not fit for it. Below Windows 10 version 2004, the wheels are not tested.
- **Handoff:** phase 1 brings the wheels over and drops the spike's rehearsal jobs.

## Part III — History

- 2026-11-02: Drafted. Tracking issue [#392](https://github.com/LupoLab/solprop/issues/392).
