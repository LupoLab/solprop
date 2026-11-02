# SEP 43 — Retire `numba_cpu`

| | |
|---|---|
| **SEP** | 43 |
| **Title** | Retire `numba_cpu` |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-11-02) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
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
wheel. `solprop-native` becomes a required dependency: solprop is supported where native
installs, `numpy_ref` is a development tool and no longer a fallback, and any other
platform refuses to run, with a message asking the user to report it. Success means: the
native suite is green on all of those targets, a Numba-free install imports and runs on
`auto`, `native` and `numpy_ref`, a platform without native gets that message, the coverage
gate holds without Numba's paths, and no result changes beyond the accuracy bounds of §7.

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
- a single stable-ABI native wheel for Python 3.14 and 3.15, measured first;
- making `solprop-native` a required dependency, with a refusal on platforms it does not
  reach (§6.7);
- rewriting the README, installation page, guide and messages to say so (§8).

**Out of scope:**

- any further performance work on `numba_cpu`: [#104][104], [#305][305], [#306][306] and
  [#308][308] close as not planned when this SEP is accepted;
- wheels for Intel macOS, ppc64le or musl, and processors below x86-64-v2 or Neoverse N1,
  all of which become unsupported (§9);
- GPU wheels ([#317][317]) and CUDA or HIP source builds, which are unchanged;
- changes to `numpy_ref`'s arithmetic or to any numerics. Its documented role does change
  (§6.7).

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
| Intel macOS | conda-forge only | solprop is not on conda-forge | Unsupported |
| ppc64le | conda-forge only | solprop is not on conda-forge | Unsupported |
| x86-64 below AVX2 | Compiles for the host | Runs on `numba_cpu` | Native x86-64-v2 build |
| Arm below Neoverse N1 | Compiles for the host | Runs on `numba_cpu` | Unsupported |

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
- `pip install solprop` installs native, with no extra. There is no fallback backend:
  `backend="auto"` picks `native`, and where native can't run, `auto` and `native` raise a
  `BackendError` that gives the platform details and asks the user to report them (§6.7).
  This covers a platform with no wheel, a processor below the wheel's floor, and a native
  package that isn't installed.
- `import solprop` still works everywhere it installs, so results can be read on any
  machine.
- Every supported install has native. `numpy_ref` is not an alternative to it: it is the
  development and validation tool, and it also runs the few features native doesn't have yet
  (Q3), which behave as they do today. `backend="mlx"` stays an opt-in extra and a named choice
  on Apple silicon.
- The x86-64 wheels hold both builds and choose at import. `SOLPROP_NATIVE_CPU_VARIANT=
  x86-64-v2` asks for the lower build, for tests and comparisons; a higher build than the
  processor runs is refused by name, as an unknown name is. `solprop_native.build_info()`
  and the benchmark build string report which build is loaded.
- `SOLPROP_NUMBA_*` and `SOLPROP_PASS_TIMERS` go; `SOLPROP_FFTW_PLAN` stays, since native
  reads it. Native has Kokkos regions for what the pass timers measured.

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
| New: `no-native` | — | No native package, a few default-lane modules on `numpy_ref`; checks `auto`'s refusal and a real run on `numpy_ref` by name | Measured locally, before `auto` refused: 193 passed in 23 s |

**The wheel on documentation-only changes.** `native-wheel` runs only when `lanes == full`
today, and `docs` runs on documentation-only changes as well, so `native-wheel` must run
for those too, on Linux only. It is cached by the hash of the native sources, so nearly
every run reuses a wheel. A cold Linux build takes 13 to 22 minutes. A documentation-only
change pays that only when the last change to `native/` merged without a push to `main`
building it: such a merge runs no lanes, so it leaves no cache for other pull requests.

**Read the Docs stays free of native.** It executes no gallery script, because all gallery
output is committed (`.readthedocs.yaml`, `docs/conf.py`). It needs only a clean build after
the removal, and its install gains `--no-install-package solprop-native` (§6.7).

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

**Plain `uv sync`.** In a checkout it builds native (§6.7). A new `poe fetch-wheel` installs
the wheel CI built for the newest commit in the checkout's history, using `gh`, for anyone who
would rather not compile. That wheel's artifact retention grows from 3 to 14 days to match.

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

#### 6.7 Native as a required dependency

This is the maintainer's decision of 2026-11-02, and it replaces the fallback that earlier
drafts proposed. solprop's supported platforms are exactly the ones native reaches.

**The dependency.** `solprop-native==<version>` moves from the `native` extra into
`dependencies`, pinned by `hatch_build.py` as the extra is today, with no environment
markers. Markers could not express the limit in any case: they can't see the C library or
the macOS version.

**The fallback wheel.** A failure at install can't say what we want. `solprop` is a pure
Python wheel and native has no sdist, so no solprop code runs during `pip install`, and pip
reports only:

```text
ERROR: No matching distribution found for solprop-native==1.0
```

So each release also publishes a small `py3-none-any` wheel of `solprop-native` at the same
version. Installers rank it below every platform wheel, so they take it only when none
matches: musl, an older glibc or macOS, Intel macOS, ppc64le, and so on (Part II). It holds
only the message: importing it raises the refusal, and `solprop.backends.registry` recognises
it without importing it. A source build installed over it replaces it. Where another
dependency has no wheel for the platform, the installer fails on that one first and the
message is never reached.

**The refusal.** `auto` and `native` raise it on selection, and not at `import solprop`,
because the documentation build, `numpy_ref` by name and reading results on an unsupported
laptop all need the import. It states why (no matching wheel, the missing processor features,
or no native package) and prints a prefilled GitHub issue link:

```text
solprop does not run on this platform: no solprop-native wheel matches it.
  system: Darwin 24.1.0, machine: x86_64, Python: 3.14.7, solprop: 0.3.0
Please report this platform, so that we can consider supporting it:
  https://github.com/LupoLab/solprop/issues/new?title=...&body=...
```

The link carries the operating system, architecture, C library, Python and solprop versions,
and any missing processor features, and no host or user names. Nothing is sent: the user
reads the issue and submits it. For a processor below the floor or an unusual system, the
message also points to [building native from source](../../guide/performance/building-native.md).

**A release gate.** The fallback wheel would hide a missing platform wheel: if a release
lacks, say, the Windows on Arm wheel, those users silently get the refusal. `publish-native`
therefore checks that every platform wheel and the fallback are present before uploading
anything.

**The workspace.** Anyone working in a checkout is developing, so a plain `uv sync` compiles
native, which `native/` being a required workspace member gives without further configuration.
That gives the same packages as an install, with native built from source. `uv sync --extra
native` goes, with the extra. A first sync in a fresh worktree took 104 s at 12 jobs, with FFTW and
OpenBLAS already built on the machine (Part II), and a machine without the C++ toolchain stops
with CMake's message naming the compiler to install.

Automation that needs no local build opts out in two places, because `uv sync` ignores
`UV_NO_SYNC` and `uv run` has no flag to skip a package: `uv sync --no-install-package
solprop-native`, and `UV_NO_SYNC=1` so that a later `uv run` doesn't build it. That covers the
lanes in `ci.yml` and `wheels.yml`, which use CI's wheel or need no native, and `cuda.yml`, which
installs its own build. Read the Docs only needs the first. `poe fetch-wheel` follows the same
route on a machine without a compiler, such as Windows, for which the build guide has no
instructions.

**`numpy_ref` and `mlx`.** Every supported install has native, whichever backend a run uses.
`numpy_ref` remains the reference that every backend is tested against, and it stops being
described as an alternative way to run solprop. The features that run only on it stay there
(Q3). `mlx` is unchanged: its platforms are within native's, native is installed beside it, and
loads only if selected.

#### 6.8 Layer, contract and cost

This is a backend and packaging change. `KERNEL_SCHEMA_VERSION` does not change, and no
kernel changes. Wheels grow, Linux from 16.5 to 19.2 MB and Windows from 5.2 to 9.7 MB. The Windows on Arm
wheel is 4.7 MB. solprop drops two runtime dependencies, so a default Linux x86-64 install
downloads native's 19.2 MB in place of about 64 MB for Numba and llvmlite (PyPI, 2026-11-02),
and everyone gets the fast backend. In the release wheels run, the
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
| Required dependency | `pip install solprop` and `uv pip install solprop` from the release's files on each wheel platform, with no extra; `auto` is `native` | Exact |
| The fallback wheel | pip and uv choose it for tags no platform wheel matches, and never over one that matches; a source build replaces it | Exact |
| The refusal | No wheel, below the floor, and no native package each raise on `auto` and `native` with a working issue link; `import solprop` and `numpy_ref` by name still work | Exact |
| The release gate | `publish-native` stops when a platform wheel is missing, tested by removing one | Exact |
| The workspace | Plain `uv sync` in a fresh checkout builds native and `auto` is `native`; `--no-install-package` with `UV_NO_SYNC=1` builds nothing; the built wheel's metadata has the pinned requirement | Exact |
| The new policy in the documentation | A search of the README, `docs/introduction`, `docs/guide` and the messages finds no fallback, no `solprop[native]`, and `numpy_ref` only as §8 allows | Exact |
| Coverage | The 100 % gate on `solprop`, without Numba's paths | Holds, or the gaps are reported rather than the gate lowered |
| No regressions | `python -m benchmarks track` on the final tree against history; ledger entry | Same step counts |
| Removal is complete | A search finds Numba only in Final SEPs, convergence provenance and the tombstone | Exact |
| Python 3.15 | Native wheel on 3.15 on all five platforms; Linux default lane; 3.14 against 3.15 on the tracking set | Green; rounding-level |

### 8. Documentation

The documentation has to describe the new position throughout: **solprop runs where native
installs, and `numpy_ref` is for development.** That is more than removing `numba_cpu`.

**For users:**

- `README.md`: the install block becomes `pip install solprop`. "Fast CPUs and GPUs" loses
  the fallback paragraph, the `numba_cpu` tier and the line for a platform with no wheel.
- `introduction/installation.md`: the platform and processor tables become the definition of
  what is supported. It says what happens elsewhere and how to report it, and that no extra
  is needed.
- `guide/performance/backends.md`: `native` is the backend, `mlx` the special case on Apple
  silicon, `numpy_ref` the development reference. No tiers or fallback.
- `guide/performance/native.md`, `building-native.md` and `tuning.md`, and `native/README.md`:
  the two builds and the setting, a source build as the way past a wheel's limits, no extra
  and no fallback, and the warning on replacing a GPU build (Q5).
- The other guide pages: about 50 files under `docs/guide` name `numpy_ref`. Each is reworded
  so that it reads as a development tool, except where it is the only route (Q3), which says so
  as a limit of native.
- Tutorials: 11 of the 82 gallery scripts pass `backend="numpy_ref"`. Each moves to `native`
  where native runs it, and its committed output is regenerated and compared, as §6.5
  describes. The time this takes isn't measured yet.
- Messages: the strings in `solprop` and `solprop_native` that offer `numpy_ref` or
  `solprop[native]`, including the refusals in `driver/simulation.py`, `backends/registry.py`
  and `CpuBaselineError`, follow the same wording.

**For contributors:** `CONTRIBUTING.md`, `AGENTS.md`, `native/AGENTS.md`, `dev/contributing.md`,
`dev/testing.md`, `dev/tooling.md`, `dev/backends.md`, `dev/benchmarking.md`,
`dev/dependencies.md` and `dev/releasing.md` lose `numba_cpu` and `--extra native`, and state
that a plain `uv sync` compiles native, how `poe fetch-wheel` avoids that, and the fallback
wheel's place in a release. The comments in
`pyproject.toml` and `hatch_build.py` follow.

A fragment in `changelog.d/` records the removal and the required dependency as a breaking
change. Final SEPs stay as history.

### 9. Compatibility and impact

**R11's milestones, answered directly.** There is no freeze and no feedback period, because
there are no users to give feedback: 0.1.0 is a day old and unannounced. 0.2 is the last
release with `numba_cpu`. Rollback is pinning 0.2 or reverting the single removal pull
request. This overrides the advice in [#316][316] to wait two minor releases, and is
recorded as the maintainer's decision.

**Accepted losses:**

| Who | Before | After |
|---|---|---|
| Arm below Neoverse N1, Windows before version 2004, Intel macOS, ppc64le, musl, older glibc, macOS before 14 | `numba_cpu` where Numba installs | Unsupported: `auto` refuses and asks for a report. `numpy_ref` by name remains for development where it installs, and a source build of native where it compiles |
| x86-64 below x86-64-v2 (Intel before 2008, AMD before 2011) | Nothing from wheels: NumPy 2.5.2's own baseline is `X86_V2`, so `numba_cpu` can't import it either | Unchanged. Native's v2 build sits at NumPy's floor |
| Any CPU inside a GPU build | `numba_cpu` in process | A second environment with the CPU wheel; `numpy_ref` in process for development |
| A plain `pip install solprop` on a wheel platform | `numba_cpu`, with a warning | `native` |
| x86-64 below AVX2 | Numba compiled for the host | Native v2, in 0.44 to 0.51 of Numba's time |

**Other effects.**

- A script that names `numba_cpu` fails with a message that says what to use. Its output
  files still load.
- The `native` extra goes. Installers ignore an unknown extra with a warning, so
  `pip install "solprop[native]"` should still install solprop and native.
- Each solprop release now requires its native wheels, including for a change that touches
  only Python, because of the exact pin. They are already released together (SEP 29).
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
- **`auto` falls back to `numpy_ref` with a warning.** Earlier drafts proposed this, and it
  keeps first-run scripts working. It also leaves a supported-looking path that is 6 to 40
  times slower than threaded native, and it hides the platforms worth knowing about. Rejected
  by the maintainer.
- **Native as an extra, or a dependency with environment markers.** An extra leaves the default
  install without the backend solprop is built around. Markers can't see the C library or the
  macOS version, so those systems would still fail in the resolver with no message.
- **A required dependency and no fallback wheel.** The simplest packaging, but the user gets
  the installer's error and no way to report the platform. uv's does list the available
  platforms; pip's says only that no distribution was found.
- **Refusing at `import solprop`.** It would break the documentation build, `numpy_ref` by name
  and reading results on an unsupported machine.
- **Sending the platform report automatically.** Nothing leaves the user's machine unless they
  submit it.
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
   tombstone. Remove the dependencies and regenerate `uv.lock`. The required dependency, the
   fallback wheel, the refusal and the release gate of §6.7. Documentation, as §8 lists. New ADR
   superseding ADR-061's Tier 2 and its statement on the documentation build. A changelog
   fragment.
4. **Python 3.15.** The stable-ABI wheels and a Linux 3.15 lane. May land before phase 3.

Whether phase 3 merges before or after the 0.2 cut is the maintainer's call. 0.2 is meant
to be the last release with `numba_cpu`. The required dependency lands with the removal and not
before, since until then `numba_cpu` is still the fallback. A plain install on Windows on Arm
therefore has neither pyFFTW nor native until phase 3, and `auto` picks `numba_cpu` and fails
with an `ImportError`. Installing `solprop[native]` avoids that until the extra goes.

### 12. Open questions

- **Q1. What `auto` does without native. Settled 2026-11-02:** it refuses (§6.7).
- **Q2. Whether native becomes a default dependency. Settled 2026-11-02:** it is required on
  every platform, with no markers (§6.7).
- **Q3. Features that run only on `numpy_ref`. Settled 2026-11-02:** they stay there, unchanged.
  The rule is about installs, not runs: there is no supported solprop install without native,
  even where a feature uses only `numpy_ref`. Closing these gaps in native is separate work.

  | Feature | State today |
  |---|---|
  | The TDSE response | Requires `numpy_ref`; `auto` resolves to it for a TDSE response |
  | `CoherentWall(method="cf4")`, and a coupled flow with a CF4 section | `numpy_ref` only, with the Runge–Kutta march |
  | Single precision away from Apple silicon | `numpy_ref`; native refuses it and `mlx` needs Apple silicon |
  | The general route's delayed response beyond six terms on three components | `numpy_ref` only (Part II) |

- **Q4. What `uv sync` does in a checkout. Settled 2026-11-02:** it compiles native. Anyone in a
  checkout is developing, and users install wheels from PyPI (§6.7).
- **Q5. Source builds beside the exact pin. Settled 2026-11-02.** A GPU or local build is a source
  build at solprop's version. Reinstalling or upgrading at that version keeps it; a new release
  replaces it with that release's CPU wheel, with no message (Part II). The results stay correct,
  but the run moves to the CPU. Python has no finished standard for this: Wheel Variants (PEP 817)
  is still a draft. So source builds carry a PEP 440 local label (`0.3.0+cuda`), which satisfies
  the pin and shows in `pip list`; `building-native.md` says that upgrading a source build means
  rebuilding it from the new tag; and an optional setting makes a run refuse a CPU-only build, for
  GPU environments that want the guarantee. GPU wheels ([#317][317]) remain out of scope.

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
- **What native doesn't reach** (found while rewriting the documentation in phase 3, from
  reading the code, not by running each): the general route's delayed response is limited
  to six terms on three components, beyond which a problem runs only on `numpy_ref`; Adams
  beside a coupled flow falls back to the slower Python march above 16 channels or off the
  eigenbasis route, with a `HostMarchWarning`; and native has no separate override for its
  transverse FFT threads.
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
- **The required dependency, a design probe (2026-11-02):** fake wheels in a local index, with
  each platform simulated by its tags, on pip 26.2.2 and uv 0.12.7. Neither PyPI nor those
  machines were involved, so §7's checks remain to be done.

  | Platform | Wheel chosen, with a fallback wheel present |
  |---|---|
  | Linux x86-64 and aarch64, glibc 2.28 | Platform wheel |
  | Windows x64 and Arm | Platform wheel |
  | macOS 14 and 15 on arm64 | Platform wheel |
  | glibc 2.17, musl, Intel macOS, macOS 13 on arm64 | Fallback |
  | ppc64le (pip only) | Fallback |

  Without the fallback wheel, pip reports "No matching distribution found", and uv that the
  requirements are unsatisfiable, with a hint listing the platforms that have wheels.

  In a trial of the workspace from `main` with `solprop-native` added to `dependencies`, on the
  Xeon Gold 6240 with GCC 16.2: a first plain `uv sync` in a fresh worktree built native in 104 s at
  12 jobs, against FFTW and OpenBLAS already built on the machine; a second took under a second;
  and `uv run` then used it, with `auto` resolving to `native`. Without the toolchain in its
  environment, the sync stopped at CMake's message naming the compiler. After a sync with
  `--no-install-package solprop-native`, a plain `uv run` tried to build it, `UV_NO_SYNC=1` avoided
  that, and `uv sync` ignored `UV_NO_SYNC`. Pinning the required dependency in `hatch_build.py`
  wasn't tried: hatch doesn't allow a field to be both listed statically and marked dynamic, so the
  list may have to move.
- **Source builds beside the pin, a design probe (2026-11-02):** fake wheels again, pip 26.2.2 and
  uv 0.12.7, with the installed build standing for a GPU build:

  | Installed | Action | pip | uv |
  |---|---|---|---|
  | `1.0+cuda` | Install or `-U` solprop 1.0 (pin `==1.0`) | Kept, also with `--upgrade-strategy eager` | Kept |
  | `1.0`, no label | `-U` solprop 1.0 | Kept, also with eager | Kept |
  | `1.0+cuda` | `-U` when solprop 1.1 is released | Replaced by the 1.1 wheel, with no message | The same |

  Setting the label on a real build, which takes its version from `setuptools_scm`, wasn't tried.
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
- 2026-11-02: The maintainer settled Q1 and Q2: native is a required dependency, a platform
  without it is unsupported and refuses with a request to report it, `numpy_ref` is a
  development tool and `mlx` stays a special backend. §6.7, §8 and Q3 to Q5 added. Q3 then
  settled: features that run only on `numpy_ref` stay there, and every supported install has
  native. Q4 and Q5 then settled: a plain `uv sync` in a checkout compiles native, and source builds
  are labelled, rebuilt on upgrade and can opt into refusing a CPU-only build.

### AI review — 2026-11-02

**Reviewer:** GPT-6 (AI). Reviewed the whole proposal at
[`023a5351`](https://github.com/LupoLab/solprop/commit/023a53519fc4ee9033e334cb792ad8ea4e29ccca),
including Q1–Q5, against the architecture, conventions, SEP 0, SEP 25, SEP 29,
the registry and simulation entry points, shared helpers, packaging and workflows,
and representative tests. Related work read includes
[#395](https://github.com/LupoLab/solprop/pull/395),
[#398](https://github.com/LupoLab/solprop/pull/398),
[#399](https://github.com/LupoLab/solprop/pull/399) and the issues linked below.
This is a review, not acceptance or an implementation report. Part I and Part II
have not been edited; only the review metadata and this subsection were added.

**Assessment.** I support retiring `numba_cpu`. Removing a second compiled CPU
implementation reduces the cost of every future seam change, and the reference/native
pair is a sound long-term architecture. Two x86 builds preserve performance on newer
processors without making older ones pay for a source build. The lack of an announced
user base is a reasonable reason to shorten deprecation, provided the final release
and migration guarantees are explicit. I would not make retaining Numba the remedy
for the questions below.

I also support native in the ordinary installation and a refusal instead of an
automatic slow fallback. However, retiring Numba, limiting the support matrix,
publishing a refusal-only wheel, and compiling native for every contributor are
separate decisions. The last two introduce maintenance and usability costs that the
speed measurements do not answer. My principal recommendations are to preserve a
clearly documented role for reference-only science, reconsider the refusal-only
wheel, and make the packaging and qualification contracts precise before acceptance.

**Evidence and limits.** The code and workflow review is read-only. Small local
probes checked version matching, wheel replacement and the CPU selector; their
results are recorded with the relevant items. No native wheel, GPU build, long
propagation or timing series was rebuilt or rerun for this review. The spike's
numbers are evidence reported by its author, not independently reproduced here.
The external sources checked were the official Python, PyPA, pip, uv, nanobind and
OpenBLAS documentation. The review's probes used CPython 3.14.5; the wheel probe
used uv 0.13.6, not pip. All R items below remain open for the author/maintainer to
answer in a separate response, as SEP 0 requires. High priority means a contract or gate
to settle before acceptance; medium priority may be explicitly assigned to a phase.

1. **R1 — High: define support and automatic selection together (§5, §6.7, §8,
   Q1–Q3).** The unconditional statements that `auto` picks native and refuses
   without it conflict with retaining today's TDSE behaviour: `_tdse_entry` in
   `driver/simulation.py` changes `auto` to `numpy_ref` before the registry sees it.
   There is a second exception: `_single_backend` selects `mlx` for single precision
   when available. Neither case is an accidental fallback. Conversely, CF4 and the
   larger general delayed response need an explicit reference choice.

   **Recommendation/question:** write one selection table covering ordinary double
   precision, TDSE, single precision, explicit reference and explicit MLX, with a
   real native wheel, the stub, and no native package. Is native installation merely
   the supported packaging policy, or must every TDSE/MLX/reference run prove that
   native could execute on this machine? I recommend the former, with the two
   existing `auto` exceptions stated explicitly and the development escape hatch
   kept. Do not load a CPU or GPU runtime just to enforce an installation policy.
   Describe `numpy_ref` as the reference and the supported executor for the listed
   exceptions, rather than implying that TDSE science is now an unsupported
   development activity. Also preserve the rule that future features need native
   parity or an explicit accepted exception; Q3 should not become a general waiver.

   **Maintainer decision — 2026-11-02 (John C. Travers):** describe `numpy_ref` as
   the numerical reference backend and the executor for explicitly listed supported
   scientific features. Those features remain supported on the project's supported
   platforms. Explicit reference execution elsewhere may remain available without
   a support promise for those platforms. The drafting agent should reconcile this
   wording with the selection rules and the required native installation policy.

2. **R2 — High: the refusal-only wheel needs an installed-state contract, or should
   be deferred (§6.7, §7, Q2).** Wheel preference only answers a fresh resolution.
   In a disposable local-index probe, I installed a pure-Python stub at version 1.0,
   then added a compatible platform wheel at the same version. `uv pip install
   --upgrade` left the stub installed; `--reinstall` selected the platform wheel.
   This models a partial mirror, a newly added wheel, or a system whose compatibility
   changes. The package manager sees a satisfied version, not an unavailable backend.
   A local label helps identify a source build, but does not solve stub recovery.
   The [pip install reference](https://pip.pypa.io/en/stable/cli/pip_install/)
   likewise distinguishes upgrade from forced reinstallation; pip was not probed here.

   **Recommendation/question:** is a nicer first-run error worth a permanent second
   meaning of “solprop-native is installed”? My preference is to defer the stub,
   keep the required real dependency and document resolver failures, while providing
   a deliberate lightweight setup for development/result reading. If the stub stays,
   specify and test fresh install, same-version recovery, upgrade, source-build
   replacement, a partial wheel index, and normal project lock/sync with pip and uv.
   Give the stub and platform wheels consistent dependency metadata, a stable
   non-importing capability marker, and an exact recovery command. `available()`
   and diagnostic tools must not report the stub as an executable backend.

   **Maintainer decision — 2026-11-02 (John C. Travers):** omit the refusal-only
   wheel from this change. Normal installations require the real native package;
   accept the installer's resolution error on unsupported platforms and explain it
   on the installation page. Retain helpful runtime diagnostics for unsupported
   CPUs and broken installations where solprop can execute them. Remove the stub's
   design and its associated verification obligations when revising the proposal.

3. **R3 — High: cached ancestor wheels and the exact version pin do not yet form a
   coherent contributor workflow (§6.5, §6.7, Q4–Q5).** CI keys its wheel cache on
   native sources; `fetch-wheel` in #398 selects an ancestor artifact and warns if
   native changed. A Python-only commit still changes the VCS-derived package
   version, so that ancestor wheel does not satisfy the new required exact pin.
   A local version label in the requirement is also matched exactly: the probe
   accepted `0.3.0+cuda` for `==0.3.0`, but rejected a different development version
   and a `.cuda` suffix added to an already locally labelled development pin.
   This follows [PyPA's version-matching rules](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching).
   Skipping synchronization avoids a rebuild; it does not make mismatched metadata
   consistent. Furthermore, `native/` being unchanged alone does not certify the
   current Python adapter contract or the seam it consumes.

   **Recommendation/question:** define an explicit development exception, or require
   an artifact compatible with the current version and contract. Test a Python-only
   commit, a seam change, a native change, a dirty checkout, and an expired artifact.
   Specify what subsequent `uv run`, `uv sync` and a dependency check do. Reject an
   incompatible artifact rather than merely warning; retain its true build identity.
   Do not relabel an old binary to pretend it was built from the current commit.
   Exercise the real metadata hook, wheel and sdist, since Part II says the dynamic
   dependency metadata was not tried. Document how a released source build and a
   development build each receive their labels.

4. **R4 — High: turn Q5's GPU safeguard into a specified interface (§5, §7, Q5).**
   “An optional setting” does not yet tell an implementer what must be checked or a
   user how to protect a GPU job. An upgrade replacing a CUDA/HIP build with the CPU
   wheel can turn a short allocation into a much longer one; correct fields alone
   do not preserve the user's execution request. The CPU-only second environment
   in §9 is also a real workflow cost, since a GPU build and the CPU build do not
   have identical admission limits (for example
   [#363](https://github.com/LupoLab/solprop/issues/363)).

   **Recommendation/question:** name the setting, its accepted values and precedence,
   and require checking the actual build's execution space before allocating or
   propagating. A version suffix is provenance, not evidence of a usable GPU.
   State whether a missing device, the wrong device backend and an explicit
   `numpy_ref` run are refused. I favour an opt-in execution requirement, enabled
   in the documented GPU environment/job recipe, over changing every CPU user's
   default. Add real labelled-build install/upgrade tests to §7; the fake-wheel
   resolver probe does not test the SCM label or runtime guard.

5. **R5 — High: replace step-count equality with numerical acceptance criteria
   (§4, §7, Part II accuracy discussion).** “Same step counts” cannot establish no
   regression: equal counts can hide wrong fields, and harmless rounding can change
   adaptive decisions. Requiring the same step sequence across v2/v3 or interpreters
   conflicts with SEP 25. “About 1e-10” also leaves the norm, normalization, zero
   handling and whether it is a rounding or accuracy claim unspecified. A similar
   difference between two ABI builds demonstrates variability, not that either
   result is sufficiently accurate.

   **Recommendation/question:** use matched discrete computations for predeclared
   rounding budgets, and adaptive runs against a tighter/stored converged reference
   for per-case accuracy envelopes. Define the field and observable metrics and
   their units before comparison. Keep actual work counts as diagnostics and exact
   accounting, not an equality gate. Give performance its own measured criterion;
   the final tracking comparison needs times, spread and construction cost as well
   as step counts. Existing spike results can remain exploratory evidence, with
   any unexplained excess identified as an open qualification item.

6. **R6 — High: retire test implementations only after mapping their obligations
   (§6.4, §7, phase 2).** A `test_numba_*` filename does not establish that its
   numerical witness is disposable. For example, `test_numba_band_bounds.py`
   contains an operand-based absolute plasma error budget, including cancellation
   and pre-ionisation; several other modules cover hard-mask zeros, reductions and
   thread-sensitive invariants. Some may already have native equivalents, but the
   proposal gives no mapping. A high Python coverage percentage cannot identify a
   lost stress input or independent bound.

   **Recommendation/question:** make phase 2 deliver a compact inventory: each
   retiring test family/witness maps to a surviving reference/native test or has a
   reason that it exercises only a deleted implementation. Preserve scientific
   invariants, pathological inputs and independently derived bounds, while dropping
   Numba cache/dispatcher mechanics. Retargeting all comparisons mechanically to
   `numpy_ref` must not turn one into a comparison of an implementation with itself.
   Keep historical convergence and benchmark readers able to read retired names.

7. **R7 — High: qualify the wheel as the only production path, including absence
   and unexpected skips (§6.5, §7, §9).**
   [#313](https://github.com/LupoLab/solprop/issues/313) records unqualified minimum
   systems and native tests that may silently skip. The spike's failure attribution
   is useful, but an empty `pyfftw` and `-k "not numba"` are not the final supported
   install. Removing the alternative makes a packaging/runtime failure a complete
   production outage on that platform.

   **Recommendation/question:** make removal conditional on the final default suite
   on all five wheel platforms without the rehearsal filters, forced v2 where
   applicable, the planned slow lanes, no unexpected native
   skips, and fresh outside-checkout installs that actually propagate. Include a
   small compiler-free smoke run at each declared minimum OS, or explicitly narrow
   the support claim and assign the remaining qualification to #313. Treat a stub,
   an absent package, a below-floor CPU and a broken shared-library load as distinct
   cases; do not convert arbitrary import failures into “unsupported platform”.
   Importing, reading a saved result and running the reference by name must be
   exercised in the final no-native/stub environments, not just `import solprop`.

8. **R8 — High: certify the whole binary's CPU floor and fail safely when probing
   fails (§6.1–6.2, §7).** A direct probe of this revision's `_cpu.py`, with v3/v2
   records and an absent Linux cpuinfo file, returned no baseline shortfall and
   selected `_core` (v3). On a lower processor that can reach an illegal instruction
   instead of the promised refusal. This existing selector path is now tracked in
   [#443](https://github.com/LupoLab/solprop/issues/443); no native execution was
   performed in this review. Also, a generic Armv8-A core does not alone
   prove the Windows Arm wheel needs no check: it statically links OpenBLAS built
   with `TARGET=NEOVERSEN1`, `DYNAMIC_ARCH=OFF`. OpenBLAS's
   [build documentation](https://www.openmathlib.org/OpenBLAS/docs/build_system/)
   treats the target as a CPU architecture choice.

   **Recommendation/question:** specify what happens when CPU capabilities cannot
   be established; I recommend refusal before loading rather than assuming the
   highest variant is safe. Add missing/malformed probe cases. For Windows Arm,
   record the instruction requirements of the linked libraries and the minimum
   admitted Windows hardware, then justify the absent check or add one/lower the
   target. This review does not establish that the Arm wheel actually uses an
   unsupported instruction; it identifies a missing whole-wheel justification.

9. **R9 — Medium: distinguish Python coverage from assurance of the sole compiled
   implementation (§6.4–6.5, §7).** Removing `bodies.py` and its interpreted coverage
   pass shrinks the measured Python code while production execution moves further
   into C++. [#324](https://github.com/LupoLab/solprop/issues/324) records that native
   test-suite source coverage has not been measured. Keeping Python's 100 % is good,
   but it does not answer which C++ capabilities the surviving tests reach.

   **Recommendation/question:** retain the Python gate and native sanitizer/tests,
   and commission one informational native coverage run over the migrated suite,
   or explicitly defer that evidence to #324 with an owner/phase. No arbitrary C++
   percentage is requested. The useful deliverable is an account of unexercised
   capabilities and the surviving analytic/parity checks. Keep `numpy_ref` independent:
   block imports of native, Numba and pyFFTW when testing its helper and run paths.
   Native being a dependency must not make the oracle call the code it validates.

10. **R10 — Medium: define the helper cost gate and retain its arithmetic contract
    (§6.3, §6.8, phase 2).** `construction_s` is named as a gate without a case set,
    baseline, allowable increase or treatment of measurement noise. Shared helpers
    also serve stages, dense evaluation and the Python march. Native supplies its
    own `Helpers`, but a construction-only measurement cannot establish that every
    affected call is outside propagation. “No kernel changes” does not remove the
    need to qualify changes to protected coefficient evaluation.

    **Recommendation/question:** list which callers switch implementation and which
    retain native helpers; time short table-heavy construction, host-march and
    reference cases. Set an absolute/relative budget appropriate to each. Keep the
    strict operation order of compensated sums, small-argument formulae and basis
    construction. Any vectorization proposed after a failure needs its own
    conditioning/rounding argument, not only a timing win. The oracle must remain
    usable for validation even though its speed is not the production target.

11. **R11 — Medium: specify migration beyond opening an old file (§5, §9, §7).**
    “Old output and snapshots ... still load” does not settle continuing a run whose
    recipe/settings name `numba_cpu`, restoring Adams history, replaying a schedule,
    or reading a scan's historical execution metadata. These are distinct from
    loading a field, particularly after SEP 41 and SEP 42. The backend setting is
    mutable as an execution resource, so a deliberate override is plausible without
    changing scientific provenance, but it needs a documented path and tests.

    **Recommendation/question:** state whether old Numba runs are readable only or
    resumable on native/reference, and show the explicit backend override. Test a
    historical output, snapshot, recipe and short adaptive continuation, preserving
    the original execution record and appending the new one. Use SEP 42's normal
    compatibility checks and accuracy envelope; do not silently rewrite the backend
    name or bypass those checks. If any continuation is intentionally unsupported,
    document the refusal and the pinned-release recovery.

12. **R12 — Medium: Q4 should distinguish native development from Python/docs
    contribution (§6.5, §6.7, §8).** I agree a native developer should build the local
    tree. “Anyone in a checkout is developing” does not imply that a documentation
    author, a Python-only contributor or an offline researcher has a C++ toolchain,
    FFTW/OpenBLAS and GitHub artifact access. The 104 s measurement had the libraries
    prebuilt; it is not first-install cost. The checkout/network limitation is
    already tracked by [#312](https://github.com/LupoLab/solprop/issues/312).

    **Recommendation/question:** keep the default source build if desired, but make
    a reliable lightweight contributor setup a first-class documented command,
    rather than leaving it as two automation settings. Validate it with no compiler
    and no native import, and explain its limits. Fix R3 before recommending
    `fetch-wheel` as the alternative. State explicitly whether archive/offline
    builds remain deferred to #312; required native increases their importance for
    scientific reproducibility, although they need not block this retirement.

    **Maintainer decision — 2026-11-02 (John C. Travers):** keep plain `uv sync`
    building native by default, and provide one documented lightweight setup command
    for Python/documentation contributors without native or a compiler. It supports
    reference execution and documentation checks as a development workflow, not a
    production fallback. Native changes still require native qualification. The
    drafting agent should specify the command and resolve R3's compatibility details.

13. **R13 — Medium: make phase completion, evidence and rollback current and
    enforceable (§3, §7, §9, §11, Part II).** Part II says nothing has merged, but
    #395 is merged and its code is in the reviewed tree. #398 has a full CI result
    distinct from the spike, and #399 has a real 3.15 source-build result. The
    proposal still treats all of this as one unmerged rehearsal. “Coverage holds,
    or gaps are reported” also reads as permission to finish below the mandatory
    gate, and the multiple-PR plan needs an explicit SEP 0 interpretation rather
    than relying on a tooling exemption to cover all future phases.

    **Recommendation/question:** record what is already delivered with commit/CI
    links, which exact remaining PRs are prerequisites, and which measurements apply
    to which artifact. Require all mandatory gates to pass before removal/Final;
    reporting a failure is a phase report, not a substitute for passing. Keep
    remaining implementation phases on the agreed branch unless a specific
    standalone exception is recorded. Decide the last Numba release: “whether
    phase 3 merges before or after 0.2” is ambiguous beside “0.2 is the last release
    with Numba”. Pinning an existing release is an immediate rollback; reverting
    source after publication requires a new tested release. Before Numba's last
    release, retain enough qualification for it despite moving the CI lanes.

    **Maintainer decision — 2026-11-02 (John C. Travers):** remove `numba_cpu`
    before the 0.2 release. Version 0.2 therefore ships without Numba; it is not the
    last release retaining it. Update the phase ordering, compatibility text and
    migration/rollback guidance consistently. The required qualification gates
    still apply before removal.

14. **R14 — Medium: separate stable-ABI compatibility from Python support (§6.6,
    §7, phase 4).** The ABI strategy is sensible and nanobind
    [documents it](https://nanobind.readthedocs.io/en/latest/packaging.html).
    An ABI audit and extension import do not qualify the Python adapter, solprop,
    NumPy/SciPy/h5py and the TDSE provider together. Nor does `cp314-abi3` establish
    free-threaded support: Python
    [distinguishes the ABIs](https://docs.python.org/3/c-api/stable.html).

    **Recommendation/question:** say explicitly that these are ordinary CPython
    wheels, define the tested interpreter set and keep future-version compatibility
    separate from a support promise. Include a short actual run on 3.15 on each
    platform, plus the planned Linux suite without fake dependencies. Name whether
    h5py source installation makes 3.15 provisionally supported or remains a
    qualification lane. Keep this phase independent of removal if needed; do not
    delay retiring Numba just to advertise another Python version.

15. **R15 — Medium: retain external-library coexistence checks (§2, §6.4, §8).**
    Removing pyFFTW from solprop's dependencies removes one common collision, but
    another package in a notebook can still import it before native. The native
    initializer currently refuses that order on macOS, as
    [#315](https://github.com/LupoLab/solprop/issues/315) records. Removing every
    pyFFTW-related test or suggesting that this problem has disappeared would lose
    a still-relevant guard now that native is the standard path.

    **Recommendation/question:** keep a dedicated coexistence environment with the
    relevant external packages installed, while keeping production dependencies
    Numba/pyFFTW-free. Preserve the documented refusal until safety is demonstrated,
    update its retired-backend advice, and distinguish it from an unsupported CPU.
    Also retain shared host BLAS/resource management: removing Numba does not remove
    NumPy/SciPy construction or third-party thread pools.

16. **R16 — Medium: qualify the reach and speed argument, and reconcile related
    work (§2–3, §8–10).** The reported speed advantage is persuasive for the measured
    configurations. It is not evidence about every route/platform: #316 itself
    notes the busy Apple timings and points to
    [#314](https://github.com/LupoLab/solprop/issues/314) for missing Windows,
    aarch64 and scan comparisons. Likewise, solprop not being distributed on
    conda-forge does not itself prevent installing its Python package into an
    environment containing conda-forge Numba/pyFFTW. Dropping those systems may
    still be the right support decision; it should be described as a choice.

    **Recommendation/question:** retain the decision to retire, but bound the claims
    to the measurements and distinguish unavailable wheels from unsupported use.
    Say “no known users” if that is the evidence, rather than inferring zero users
    from no announcement. Audit open work that still names Numba: close only work
    specific to the retired implementation, retain shared physics/performance
    obligations, and link the existing reference-only gaps such as
    [#304](https://github.com/LupoLab/solprop/issues/304). I would keep established
    introductory examples using `auto` where they teach no backend-specific feature;
    reserve explicit native selection for examples that need its capabilities or
    performance. This reduces future documentation churn without reintroducing a
    fallback.

### Drafting handoff after the maintainer's decisions — 2026-11-02

The decisions recorded beside R1, R2, R12 and R13 settle those policy choices.
The drafting agent should now revise Part I and provide the separate response to
each review item required by SEP 0, distinguishing decisions implemented in the
text from qualification work assigned to a phase. The other technical questions,
including R3's wheel/version compatibility and R4's concrete GPU safeguard, still
need answers. These decisions do not mark the SEP accepted or the review complete.
