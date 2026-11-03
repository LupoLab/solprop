# SEP 43 — Retire `numba_cpu`

| | |
|---|---|
| **SEP** | 43 |
| **Title** | Retire `numba_cpu` |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-11-02) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | John C. Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-11-02 |
| **Resolution** | Accepted 2026-11-02 by John C. Travers, after the maintainer's human review, GPT-6's AI review and the response to it; [#393](https://github.com/LupoLab/solprop/pull/393) |
| **Requires** | SEP 29 (compiled backend distribution; final) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | In progress from 2026-11-02: John C. Travers, with Opus 5.5, on `feature/sep-0043-retire-numba-cpu`, [#415](https://github.com/LupoLab/solprop/pull/415) |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

This SEP removes the `numba_cpu` backend and solprop's dependencies on `numba` and
`pyfftw` before the 0.2 release, and makes `native` the only compiled CPU backend. It first
widens the native wheels to the processors and platforms that only Numba reaches in
practice: x86-64 processors below AVX2, on Linux and Windows, and Windows on Arm. The
documentation build and the continuous integration lanes that run on `numba_cpu` today move
to the native wheel.

`solprop-native` becomes a required dependency, so solprop's supported platforms are the
ones native installs on. On any other platform the installer's resolution error is the
answer, and the installation page explains it. `numpy_ref` remains the numerical reference
and the executor for a short list of supported features that native doesn't run yet.

Success means: the final suite is green on all five wheel platforms with no rehearsal
filters, a fresh install outside a checkout propagates on each, every gate of §7 passes, and
no result moves outside the rounding budgets and accuracy envelopes of §7.

### 2. Motivation and scope

Every test, route and kernel of `numba_cpu` is a second copy of what `native` does. In the
0.1.0 benchmark record ([#68][68]) it is 1.5 to 2.8 times slower than native by geometric
mean, with no winning configuration among the 22 measured on three machines (§3). That
evidence has limits: the Apple series ran on a busy machine, and there are no rows on Windows
or Linux aarch64, or for scans ([#314][314]). Its remaining value is reach, and that reach
turns out to be narrow and cheap to replace.

There are no known users. 0.1.0 was published on 2026-11-01 and hasn't been announced, so
the freeze, deprecation and feedback period that SEP 29 §6.3 anticipated would protect
nobody we know of, and this SEP does without them (§9).

**In scope:**

- a second, x86-64-v2 build of the extension in the Linux and Windows x86-64 wheels;
- a Windows on Arm wheel;
- moving the `docs`, `slow`, `oldest` and coverage lanes to the native wheel;
- replacing the Numba-compiled helpers that every backend shares (SEP 29 R2);
- removing the backend, `numba`, `pyfftw`, and the tests and golden files only they use;
- a single stable-ABI native wheel for Python 3.14 and 3.15, measured first;
- making `solprop-native` a required dependency (§6.7), with one lightweight setup for
  Python and documentation contributors;
- a named safeguard for GPU environments (§5, §6.7);
- a processor check that refuses when it can't establish the processor's features (§6.1, §6.2);
- rewriting the README, installation page, guide and messages to say so (§8).

**Out of scope:**

- any further performance work on `numba_cpu`. Issues specific to it close at the removal,
  and shared obligations stay open (§9);
- wheels for Intel macOS, ppc64le or musl, and processors below x86-64-v2 or Neoverse N1,
  all of which become unsupported, by choice (§9);
- GPU wheels ([#317][317]) and CUDA or HIP source builds, which are unchanged;
- changes to `numpy_ref`'s arithmetic or to any numerics. Its documented role does change
  (§5);
- source builds from an archive or offline, which stay with [#312][312], and the native gaps
  of §5, each of which stays with its own issue.

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

These are the configurations measured, not every route and platform ([#314][314]).

**Reach.** What the Numba project's own distributions add to the native wheels' four
platforms, from PyPI and conda-forge on 2026-11-01. A row with no wheel may still run in an
environment assembled by hand, for example solprop installed by pip beside conda-forge's Numba,
which wasn't tested. Dropping those systems is a support decision, not something forced on us:

| Target | Numba | solprop on it today | After this SEP |
|---|---|---|---|
| Windows on Arm | PyPI wheel | Blocked: pyFFTW has no wheel | Native wheel |
| Intel macOS | conda-forge only | No conda-forge package of solprop | Unsupported |
| ppc64le | conda-forge only | No conda-forge package of solprop | Unsupported |
| x86-64 below AVX2 | Compiles for the host | Runs on `numba_cpu` | Native x86-64-v2 build |
| Arm below Neoverse N1 | Compiles for the host | Runs on `numba_cpu` | Unsupported |

Numba's real advantage is that it has no fixed processor baseline. A second build at
x86-64-v2 removes most of that advantage at a measured cost (§6.1).

**Features.** The admission code and the SEP 29 capability audit (R1, closed by its native
parity phase) show no CPU problem kind that only `numba_cpu` runs. This is a reading of
those checks rather than a proof over every path; the test retargeting of phase 2 is
where a gap would appear, and any it finds goes into Part II.

**Issues.** It closes [#316][316]. It answers the Windows on Arm half of [#317][317], and the
pyFFTW half of [#319][319]. §9 lists what happens to the other open issues that name Numba.

### 4. Theory

No physics changes. Comparisons follow SEP 25's distinction: matched discrete computations
are held to rounding budgets fixed before the comparison, and adaptive runs to a per-case
accuracy envelope against a tighter or stored reference. Equal step counts establish neither.
§7 gives the metrics and bounds.

### 5. Proposed interface

**Installing.** `pip install solprop` installs native, with no extra. On a platform with no
matching wheel, the installer fails to resolve and solprop's code never runs. The installation
page explains that error and how to report the platform.

**Choosing a backend.** Native being installed is a packaging policy. A run doesn't have to
prove that native could execute on this machine, and nothing loads a CPU or GPU runtime just to
enforce the policy. `import solprop` doesn't load native, so reading results works wherever
solprop is present.

| Request | Native usable | Processor below the floor, or its features unreadable | Native fails to load | No native package |
|---|---|---|---|---|
| Double precision, `auto` | `native` | Refuses, naming the missing features or the failed check | The load's own error, unchanged | Refuses, saying how to install it |
| `native` by name | `native` | As above | As above | As above |
| TDSE response, `auto` | `numpy_ref`; native isn't consulted | The same | The same | The same |
| Single precision, `auto` | `mlx` with a Metal device, otherwise refuses; native isn't consulted | The same | The same | The same |
| `numpy_ref` by name | Runs; native isn't consulted | The same | The same | The same |
| `mlx` by name | Runs where MLX is installed; native isn't consulted | The same | The same | The same |

These cases stay distinct: an arbitrary import failure is never reported as an unsupported
platform. `auto` has no fallback for lack of native, and today's two `auto` exceptions, TDSE and
single precision, are unchanged.

**`numpy_ref`.** It is the numerical reference backend, and the executor for these supported
features, which remain supported on the supported platforms:

| Feature | Reached by | Tracked by |
|---|---|---|
| The TDSE response | `auto` or by name | |
| `CoherentWall(method="cf4")`, and a coupled flow with a CF4 section | By name | [#304][304] |
| Single precision away from Apple silicon | By name | [#342][342] |
| The general route's delayed response beyond six terms on three components | By name | |

Running it by name on an unsupported platform may keep working, without a support promise. This
list isn't a general waiver: a new feature still follows SEP 0's parity rule, and joins the list
only as an accepted exception.

**A safeguard for GPU environments.** `SOLPROP_REQUIRE_DEVICE` is an opt-in requirement on where
a run executes, set in the documented GPU job recipe:

| Value | Meaning |
|---|---|
| Unset or empty | No requirement, the default for every CPU user |
| `cuda` | The run must execute on a CUDA build that sees a device |
| `hip` | The run must execute on a HIP build that sees a device |
| `any` | The run must execute on a device build of either kind |
| Anything else | Raises a `BackendError` naming the accepted values |

It is an environment setting with no keyword, so there is no precedence to resolve, and scan
workers inherit it. It is checked at backend selection, before any state is allocated or a step
taken, by reading the actual build's `build_info().device_backend` and `device_count`, the facts
`check_build` already uses. A version label is never consulted. It refuses, with a `BackendError`,
a CPU build, the wrong device backend, and a device build that sees no device. It also refuses
`numpy_ref` and `mlx`, by name or through `auto` for a TDSE or single-precision run, because such
a run wouldn't meet the job's requirement. The message says to unset it for such a run.

**Other settings.**

- `backend="numba_cpu"` raises a `BackendError` naming `native` and `numpy_ref`, and the
  release that removed it, in place of today's unknown-backend message. §9 covers old files.
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

**When the processor can't be asked.** Today, a missing `/proc/cpuinfo` makes `select_variant`
choose the v3 build and `baseline_shortfall` return nothing ([#443][443]), so a lower processor
could reach an illegal instruction instead of the promised refusal. Windows x64 doesn't have this
gap: `IsProcessorFeaturePresent` is in `kernel32`, and it stays the check there, with its known
limits. On Linux, a wheel with a recorded baseline now asks in this order:

1. **`/proc/cpuinfo`,** as today.
2. **NumPy's `__cpu_features__`,** if that file is missing, unreadable or has no features line. It
   comes from NumPy's own `cpuid` and `xgetbv` checks, and has `X86_V2` and `X86_V3` directly. It is
   imported only at this point, because `_cpu.py` otherwise imports nothing.
3. **Refuse, before loading,** if neither answers. A wrong guess would be an illegal instruction
   with no Python error, and with two independent checks this should rarely be reached.

The name is private, which is acceptable for a fallback. A test fails loudly if NumPy moves it, and
at run time a missing name is simply no answer. `NPY_DISABLE_CPU_FEATURES` can only lower the build
chosen or lead to the refusal. On Linux aarch64 this helps less: NumPy's names cover only part of
Neoverse N1's list, with no atomics, CRC32 or RCpc, so there a missing `/proc/cpuinfo` refuses.

On x86-64, naming a build with `SOLPROP_NATIVE_CPU_VARIANT` is the deliberate way past that refusal,
on the user's word about their processor. A source build, which records no checked baseline, is
unaffected. This is a standalone fix, needed whatever happens to this SEP.

The alternatives are in §10. Measured cost and correctness are in §7 and Part II.

#### 6.2 Windows on Arm

The wheel is built on `windows-11-arm` with Clang on the MSVC ABI and the runner's LLVM
libomp, as on x64. FFTW's CMake build has no NEON option, so the build recipe patches it in.
OpenBLAS is built for `NEOVERSEN1`. Kokkos and Kokkos Kernels each pass the x86-only
`-mcx16` to Clang on Windows, and are patched at fetch.

The core keeps the compiler's generic Armv8-A baseline, but that doesn't cover the whole wheel:
it statically links OpenBLAS built with `TARGET=NEOVERSEN1` and `DYNAMIC_ARCH=OFF`, and the merged
wheel has no processor check on this platform. So `_cpu.py` gains one, using Windows's
`PF_ARM_*` feature queries, for what the linked libraries were built to use, and the installation
page's "any processor that runs Windows 11 on Arm" gets replaced by the requirement that check
enforces. The instruction requirements and the lowest Windows hardware admitted are recorded in
Part II. If a supported device would fall below it, the alternative is a lower OpenBLAS target,
with its cost measured.

#### 6.3 The shared helpers

`translate/chebyshev.py` and `stepper/multistep.py` import Numba lazily, and `helpers=NUMBA`
is a default in `translate/stage.py`, `backends/exponential.py`, `backends/multistep.py`
and `backends/driver.py`. These become plain forms. Every backend already names its own record:
`numpy_ref` takes `PLAIN` and native its C++ bindings, so their marches keep the helpers they
have. Only callers that take the default change, which are translation and direct calls.

The arithmetic contract stays: compensated sums, small-argument formulae and basis construction
keep their strict operation order, and `tests/test_helpers.py` holds the plain and native records
to its bound. Uncompiled forms cost 6 to 36 times per call, so the change has budgets fixed before
measuring, from alternating runs of the two trees with the A/A spread reported:

| Measured | Cases | Budget |
|---|---|---|
| `construction_s` | The tracking set, including its table-heavy cases | No case above the larger of 0.5 s and 20 % more |
| A short host march | Native's host Adams march, ten steps | Within the A/A spread |
| A short reference run | `numpy_ref`, ten steps | Within the A/A spread |

A vectorised NumPy form is written only where a budget fails, and it needs its own conditioning
and rounding argument, not only a faster time. A test imports solprop with `solprop_native`,
`numba` and `pyfftw` all blocked, and runs `numpy_ref`'s helpers and a short run, so that the
reference never calls the code it validates.

#### 6.4 Tests

A filename doesn't show that a test is disposable. `test_numba_band_bounds.py`, for example,
holds an operand-based absolute budget for the plasma column, including cancellation and
pre-ionisation. So phase 2 delivers an inventory, in Part II, of each retiring test family: the
surviving reference or native test that carries its obligation, or the reason it exercises only
the deleted implementation. Scientific invariants, pathological inputs and independently derived
bounds survive. Numba's cache and dispatcher mechanics don't.

Native-against-Numba comparisons move to `numpy_ref`, which about 40 of the test modules
importing `numba_cpu` already use as well. A Numba-against-reference comparison moves to native
against the reference, never to a comparison of an implementation with itself. The 4 modules that
run only on Numba are ported. The 28 `test_numba_*` modules and 7 Numba golden files go once the
inventory has accounted for them. Parametrizations drop the `numba_cpu` value. Convergence records
and benchmark readers keep the rows Numba wrote, as evidence of what it measured, through
`RETIRED_BACKENDS` in `tests/test_convergence_records.py`, as they already do for `numba_cuda`.

**Coexistence stays.** Removing pyFFTW from solprop's dependencies doesn't remove it from users'
environments, and a notebook can still import it before native, which macOS refuses
([#315][315]). `native/ci/coexistence.py` keeps running in a dedicated environment with NumPy,
SciPy, pyFFTW and Numba installed as outside packages, with `numpy_ref` replacing its `numba_cpu`
run. The refusal stays until the order is shown to be safe, its advice stops naming `numba_cpu`,
and it stays distinct from an unsupported processor. Shared host BLAS and resource management
stay too, with `threadpoolctl`, since NumPy, SciPy and third-party pools remain.

**Coverage.** Python's 100 % gate stays and applies throughout. It says nothing about which C++
capabilities the surviving tests reach, now that production runs only there, so phase 3 commissions
one informational native coverage run over the migrated suite, using the build option
[#324][324] describes. It has no percentage gate. Its product is an account in Part II of the
capabilities not exercised, and the analytic and parity checks that survive.

#### 6.5 Documentation and continuous integration on native

Moving these lanes was rehearsed before anything was removed (Part II). The rehearsal doesn't replace the final gates of §7:

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
the removal, and its install gains `--no-install-package solprop-native`, as the lightweight
setup does (§6.7).

**What ADR-061 said.** Its consequence "the documentation build installs no native backend:
a lesson or tutorial of a feature that `numba_cpu` does not run names `numpy_ref`" no
longer holds for the CI lane, and the new ADR supersedes it. A lesson may now use anything
`native` runs.

**Committed output.** `plot_11_dispersive_kerr` and `plot_05_moving_solid_guide` select
`numba_cpu` explicitly. They move to `auto` unless they teach something specific to a backend,
which reduces later documentation churn without bringing back a fallback. Four more gallery
scripts mention it in prose. Regenerating their committed output is part of the removal, since
the build refuses stale output. `poe docs-examples` runs both on `native` on the 4-core runner:
`plot_11` in 16 s and `plot_05` in 884 s, which is a one-off cost when that script changes. The
removal compares their regenerated figures and printed values with the committed ones.

**Plain `uv sync`.** In a checkout it builds native, and there is a lightweight alternative
(§6.7).

#### 6.6 Python 3.15

The native package moves to CPython's stable ABI: nanobind's `STABLE_ABI`, with
`wheel.py-api = "cp314"` in scikit-build-core. There is then one wheel per platform, tagged
`cp314-abi3`, and no build per interpreter, which is what makes a new Python cheap. Nothing in
the bindings goes beyond that ABI, so no binding code changes. Part II has the measurements:
`abi3audit --strict` is clean on all five wheels, CPython 3.15 loads each, and the wheel costs
nothing against the version-specific one.

ABI compatibility isn't Python support. These are ordinary CPython wheels: free-threaded CPython
has a different ABI and isn't a target. The tested interpreters are 3.14 and 3.15, and a later
version may load the wheel with no promise made. An audit and an import don't qualify the adapter,
solprop, NumPy, SciPy, h5py and the TDSE provider together, so 3.15 stays a **qualification lane,
not a supported version**, until the installation page can claim it, which needs h5py's wheels.
It needs:

- **A short run on 3.15 on each platform:** a real propagation, not only a load, in the
  `python-3-15` job. Where h5py has no wheel, that run uses the native package directly.

- **h5py** is the one runtime dependency without a 3.15 wheel (h5py issue 2993). It builds
  from source against a system HDF5, as the Linux lane does, so the installation page says
  that until h5py publishes wheels, and a 3.15 lane proves solprop in the meantime.
- **The TDSE provider.** `driver/tdse.py` refuses Python other than 3.14, the version
  Pyroraptor is pinned to. That stays, and TDSE needs 3.14 until the provider moves. Four
  `test_tdse_binding` tests are gated on it.
- **Source builds** on 3.15: #399 drops `EXACT` from `find_package(Python 3.14)`, and one source
  build on CPython 3.15.2 has built and loaded (Part II).

This doesn't block the removal, and the removal doesn't wait for it, though the removal clears
Numba and pyFFTW out of its way. The planned Linux suite on 3.15 runs without fake dependencies
once they have gone.

#### 6.7 Native as a required dependency

These are the maintainer's decisions of 2026-11-02. They replace the fallback that earlier drafts
proposed, and the refusal-only wheel that a later draft proposed.

**The dependency.** `solprop-native==<version>` moves from the `native` extra into `dependencies`,
with no environment markers, which can't see the C library or the macOS version in any case. In
`pyproject.toml` the entry is the bare name, so uv resolves it to the workspace member, and
`hatch_build.py` writes the exact version into the built metadata. A probe of that hook gave the
exact pin in both the wheel's and the sdist's metadata (Part II). It changes a statically listed
field, which hatch allows and the metadata standard doesn't intend, so if hatch stops allowing it
the list moves behind `dynamic`.

**Unsupported platforms.** `solprop` is a pure-Python wheel and native has no sdist, so no solprop
code runs during `pip install`, and the installer reports only that no matching distribution
exists. The installation page shows that error, says what it means, lists the supported platforms,
and gives a one-line command that prints the platform's details to include in an issue. Runtime
diagnostics stay wherever solprop can execute them: a processor below the floor and a broken
installation each get their own message, and the first also names where to report it.

**Exact pins and development.** The exact pin is a contract for released installs. A checkout has
three states, and only the first satisfies it. The other two are explicit development exceptions:

| Setup | Native | Metadata |
|---|---|---|
| `uv sync` | Built from this tree | Consistent: a workspace member, at the same version |
| The lightweight setup, below | None | Unmet, deliberately |
| `poe fetch-wheel` | CI's wheel from this commit or a compatible ancestor | Mismatched, deliberately: `uv pip check` reports it |

A Python-only commit changes the VCS-derived version, so an ancestor's wheel can't satisfy the pin,
and it isn't relabelled to pretend otherwise: its `build_info()` keeps the commit it was built from.
So `poe fetch-wheel` becomes strict. It takes an artifact only if the difference between that
artifact's commit and the working tree, including uncommitted changes, touches neither `native/`
nor the solprop modules that `solprop_native` imports. A test derives that module list from the
imports, so it can't go stale. Anything else is refused, not warned about, and so is an expired
artifact; each refusal names the two alternatives. At run time, `check_build`'s comparison of
`KERNEL_SCHEMA_VERSION` remains the seam's guard, since solprop's version is recorded and never
compared. A later plain `uv sync` or `uv run` builds native in its place.

**`uv sync` compiles.** A plain `uv sync` in a checkout builds native, since `native/` is a required
workspace member, and `uv sync --extra native` goes, with the extra. A first sync in a fresh worktree
took 104 s at 12 jobs on a machine where FFTW and OpenBLAS were already built, which isn't a
first-install cost (Part II). Without the C++ toolchain, the sync stops with CMake's message naming
the compiler, which also comes to name the lightweight setup.

**The lightweight setup.** One documented command serves Python and documentation contributors with
no native package and no compiler:

```sh
uv run --no-project tools/setup_light.py
```

It runs `uv sync --locked --no-install-package solprop-native` and says how to continue. This is a
development workflow, not a production fallback. It needs two facts explained, because uv can't be
configured otherwise: `uv run` has no flag to skip a package, and `uv sync` ignores `UV_NO_SYNC`. So
a later plain `uv run` would build native, and the contributor runs tools with the environment
activated, or with `UV_NO_SYNC=1`. There, `auto` refuses, tests take
`SOLPROP_TEST_BACKEND=numpy_ref`, and native-only tests skip. It supports the reference backend and
the lint, type, test and documentation checks, and a native change still needs a native build and
its qualification. Automation that needs no local build, such as Read the Docs and the lanes on CI's
wheel, uses the same two settings.

**Source and GPU builds.** Reinstalling or upgrading at the same version keeps a source build. A new
release replaces it with that release's CPU wheel, with no message, because the exact pin asks for
it (Part II). Python has no finished standard for this: Wheel Variants (PEP 817) is still a draft, so
it isn't relied on here.

- **A released source build** takes a PEP 440 local label, set by
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SOLPROP_NATIVE=0.3.0+cuda`. `==0.3.0` accepts it, and it shows
  in `pip list`. A real build produced that wheel and version.
- **A development build** is already labelled with its commit, and a pin that has a local label is
  matched exactly, so it takes no further label. `build_info()` says what it is.
- **The label is provenance, not evidence.** The same probe labelled a CPU build `+cuda`, and it
  reported `device_backend` `none`. That is why the safeguard of §5 reads the build itself.
- **Upgrading** means rebuilding from the new tag, as `building-native.md` says, and the GPU job
  recipe sets `SOLPROP_REQUIRE_DEVICE`, so a job that lost its GPU build stops before it starts.

**`mlx`** is unchanged: its platforms are within native's, native is installed beside it, and it
loads only if selected.

#### 6.8 Layer, contract and cost

This is a backend and packaging change. `KERNEL_SCHEMA_VERSION` doesn't change, and no native
kernel changes. The shared helpers' change is qualified under §6.3. Wheels grow, Linux from 16.5 to 19.2 MB and Windows from 5.2 to 9.7 MB. The Windows on Arm
wheel is 4.7 MB. solprop drops two runtime dependencies, so a default Linux x86-64 install
downloads native's 19.2 MB in place of about 64 MB for Numba and llvmlite (PyPI, 2026-11-02),
and everyone gets the fast backend. In the release wheels run, the
Windows on Arm build takes about 15 minutes and its default lane about 30. The x86-64
builds stay within the 19 to 30 minutes that they vary over on the hosted runners (22 on
each in a recent run without the second build). The forced-v2 lanes add about 50 minutes
each, and they run only in that workflow, not in `ci.yml`.

### 7. Verification and validation

Every row is a gate on the removal and on `Final`. Reporting a failure is a phase report, and
doesn't replace passing.

**How results are compared.** Equal step counts aren't a criterion: they can hide a wrong field,
and a harmless rounding difference can change an adaptive decision.

- **Field:** the relative L2 difference of the final complex field, $\|a-b\|_2/\|b\|_2$, as in
  `benchmarks/ab.py`; dimensionless, and absolute in the field's units if $\|b\|_2=0$.
- **Observables:** relative difference of the pulse energy, and the in-band relative L2 "shape"
  distance of the spectrum (`accuracy.md`).
- **Matched discrete computations (class R):** the `accuracy.md` §7 ladder and its budgets, fixed
  beforehand from operation counts, are run unchanged.
- **Adaptive runs (class T):** each candidate's error $E$ against a reference at `rtol`/10 on the
  v3 build at the same cases, within a safety factor of 2 on the v3 build's own error at the same
  `rtol`. The factor is fixed now, before any candidate is measured.
- **Work counts** are recorded and reported as a diagnostic, and aren't an equality gate. Where two
  runs happen to take the same step sequence, a field difference above 1e-10 is flagged for
  explanation.
- **Performance** is a separate criterion: median times with the A/A spread and `construction_s`,
  against the last history, with the verdict of `benchmarks.md`.

| Claim | Check | Bound |
|---|---|---|
| The final suite runs everywhere | The default lane on all five wheel platforms from the release wheels, with no empty `pyfftw` and no `-k "not numba"` | Green |
| Forced v2 and slow lanes | v2 forced on Linux and Windows x64; the slow lanes on the wheel ([#313][313]) | Green |
| No silent skips | A job fails on any native skip not on the list of expected ones ([#313][313]) | Exact |
| Fresh installs | `pip install solprop` and `uv pip install solprop` from the release's files, outside a checkout, with no extra, on each platform: `auto` is `native`, and a short run propagates and saves | Runs |
| Minimum systems | With no compiler, a smoke test and a short run on macOS 14, a `manylinux_2_28`-age Linux container, and Windows 10. Where no machine is available, the support claim narrows and [#313][313] keeps the rest | Runs, or the claim narrows |
| Windows on Arm's open items | The Part II list, including `plasma_band`, explained or fixed | No unexplained excess |
| Selection follows §5's table | Each row and column, with a real wheel and no native package, plus a below-floor processor and a forced load failure; messages stay distinct | Exact |
| No package in practice | With no native package: import, read a saved result, run `numpy_ref` by name | Runs |
| v2 selection | Intel SDE: Haswell gets v3; Nehalem and Sandy Bridge get v2 and `resolve_name() == "native"`; Penryn is refused with status 1 and no emulator error | Exact |
| Unreadable processors | With `/proc/cpuinfo` masked under Intel SDE, Haswell gets v3 and Nehalem v2 from NumPy's answer, and Penryn is refused. Missing and malformed files, with NumPy's name also unavailable, refuse before loading ([#443][443]), and a named x86-64 variant goes past that. A test fails if NumPy moves the name | Exact |
| Windows on Arm's floor | The new check's cases, and the recorded instruction requirements | Exact |
| v2 is correct | Default lane with v2 forced (class R); the tracking set against its reference (class T) | Class bounds |
| v2's cost is acceptable | Tracking set, Xeon Gold 6240 and EPYC 7763, interleaved fresh processes; native v2 against Numba compiled for Nehalem | v2 is not slower than Numba at v2 |
| Windows v2's BLAS | Coupled TE/TM at one thread and the radial real GEMM, which failed under OpenBLAS's run-time selection on Windows | Green with v2 forced |
| A numba-free tree | Fresh environment: `numba` and `pyfftw` absent, a short run on each backend. Separately, `numpy_ref` with `solprop_native`, `numba` and `pyfftw` blocked | Runs |
| Helpers | §6.3's budgets and `tests/test_helpers.py` | Budgets |
| No retired obligation lost | The inventory of §6.4 accounts for every retiring test family | Complete |
| Coverage | The 100 % gate on `solprop`, without Numba's paths, with the gaps in Part II closed; one informational native coverage run | Holds; run reported |
| No regressions | `python -m benchmarks track` on the final tree against history, with a ledger entry; matched and adaptive comparisons as above | Class bounds; performance verdict |
| The metadata | The built wheel and sdist each carry the exact pin, through the real hook; a checkout's install succeeds | Exact |
| The workspace | Plain `uv sync` in a fresh checkout builds native and `auto` is `native` | Exact |
| The lightweight setup | No compiler and no native import on the machine: the command, then lint, types, tests on `numpy_ref` and the documentation build | Runs |
| `poe fetch-wheel` | A Python-only commit outside the imported modules is accepted; a seam change, a native change, a dirty tree in those paths and an expired artifact are each refused | Exact |
| A real labelled build | `0.3.0+cuda` installed, reinstalled and upgraded with pip and uv, including the replacement at a new release | As §6.7 |
| The GPU safeguard | Each value against a CPU build, by name and through `auto`, refused before any state is allocated; positively on `cuda.yml`'s device | Exact |
| Old runs | §9's continuation cases, from files written by the published 0.1.0 | As §9 |
| Coexistence | `native/ci/coexistence.py` with pyFFTW and Numba installed as outside packages; the macOS refusal | Green |
| Removal is complete | A search finds Numba only in Final SEPs, convergence provenance, the coexistence environment and the tombstone | Exact |
| Python 3.15, as a qualification lane | The native wheel loads and runs a short propagation on all five platforms; Linux default lane with no fake dependencies; 3.14 against 3.15 by class R and class T | Green; class bounds |

### 8. Documentation

The documentation has to describe the new position throughout: **solprop runs where native
installs, and `numpy_ref` is the numerical reference, plus the executor of a short list of supported
features.** That is more than removing `numba_cpu`.

**For users:**

- `README.md`: the install block becomes `pip install solprop`. "Fast CPUs and GPUs" loses
  the fallback paragraph, the `numba_cpu` tier and the line for a platform with no wheel.
- `introduction/installation.md`: the platform and processor tables become the definition of
  what is supported. It shows the installer's error on any other platform, what it means, and how
  to report the platform, and it says that no extra is needed.
- `guide/performance/backends.md`: `native` is the backend, `mlx` the special case on Apple
  silicon, `numpy_ref` the reference and the executor of the list in §5, which it carries. It
  has §5's selection table, with no tiers or fallback.
- `guide/performance/native.md`, `building-native.md` and `tuning.md`, and `native/README.md`:
  the two builds and the setting, a source build as the way past a wheel's limits, no extra and no
  fallback, labels, rebuilding on upgrade, and the GPU job recipe with `SOLPROP_REQUIRE_DEVICE`.
- The other guide pages: about 50 files under `docs/guide` name `numpy_ref`. Each is reworded so
  that it reads as the reference, except where it is the supported executor, which says so as a
  limit of native. TDSE science in particular isn't described as a development activity.
- Tutorials: 11 of the 82 gallery scripts pass `backend="numpy_ref"`. Each moves to `auto` where
  native runs it and it teaches nothing specific to a backend, and keeps its explicit choice where
  it needs one. Moved output is regenerated and compared, as §6.5 describes. The time this takes
  isn't measured yet.
- Messages: the strings in `solprop` and `solprop_native` that offer `numba_cpu` or
  `solprop[native]`, including those in `driver/simulation.py`, `backends/registry.py`, the macOS
  refusal and `CpuBaselineError`, follow the same wording, and so does the single-precision message
  that calls `numpy_ref` "not a production backend".

**For contributors:** `CONTRIBUTING.md`, `AGENTS.md`, `native/AGENTS.md`, `dev/contributing.md`,
`dev/testing.md`, `dev/tooling.md`, `dev/backends.md`, `dev/benchmarking.md`,
`dev/dependencies.md` and `dev/releasing.md` lose `numba_cpu` and `--extra native`, and state that a
plain `uv sync` compiles native, what the lightweight setup supports and its limits, and when
`poe fetch-wheel` is refused. The comments in `pyproject.toml` and `hatch_build.py` follow.

A fragment in `changelog.d/` records the removal and the required dependency as a breaking
change. Final SEPs stay as history.

### 9. Compatibility and impact

**SEP 29's R11 milestones, answered directly.** There is no freeze and no feedback period, because
there are no known users to give feedback: 0.1.0 is unannounced. **`numba_cpu` is removed before
0.2.** 0.1.0 is therefore the last release with it, and 0.2 ships without Numba. If the gates of §7
aren't met, the removal doesn't merge and 0.2 waits, or the maintainers amend this SEP. Rollback is
pinning `solprop==0.1.0`, which works immediately. Reverting the removal once 0.2 is published
requires a new tested release. This overrides the advice in [#316][316] to wait two minor releases,
and is recorded as the maintainer's decision.

**Accepted losses:**

| Who | Before | After |
|---|---|---|
| Intel macOS, ppc64le, musl, older glibc, macOS before 14 | `numba_cpu` where Numba installs | Unsupported, by choice: the installer's error. A source build of native may still work, with no support promise |
| Arm below Neoverse N1, Windows before version 2004 | `numba_cpu` where Numba installs | Unsupported, by choice: `auto` and `native` refuse, naming the missing features |
| x86-64 below x86-64-v2 (Intel before 2008, AMD before 2011) | Nothing from wheels: NumPy 2.5.2's own baseline is `X86_V2`, so `numba_cpu` can't import it either | Unchanged. Native's v2 build sits at NumPy's floor |
| Any CPU inside a GPU build | `numba_cpu` in process | A second environment with the CPU wheel, a real workflow cost, since their admission limits differ (for example [#363][363]). `numpy_ref` in process remains |
| A plain `pip install solprop` on a wheel platform | `numba_cpu`, with a warning | `native` |
| x86-64 below AVX2 | Numba compiled for the host | Native v2, in 0.44 to 0.51 of Numba's time |

**Old runs.** Native and `numpy_ref` can continue a Numba run, with the script's or recipe's backend
changed deliberately:

| What | After the removal |
|---|---|
| Reading an output, a snapshot or a scan's execution metadata | Works, and keeps saying `numba_cpu`. The record isn't rewritten |
| A script or recipe that names `numba_cpu` | Refused by name, saying what to use. The backend is an execution resource, so changing it doesn't change the run's identity (SEP 42) |
| Continuing with the backend changed | Goes through SEP 42's normal compatibility checks. The original execution record stays, and the new one is added |
| Adams history in a snapshot | Crosses to the new backend, except a frozen-residual coupled flow's: a host engine's history continues on `numpy_ref`, and on native starts afresh and says why, as today |
| Anything refused | Recovered with `solprop==0.1.0` |

Phase 3 tests a real output, snapshot, recipe and short adaptive continuation, from files written by
the published 0.1.0, under class T of §7. It covers the Adams and frozen-residual cases, and that the
original record survives. Anything that turns out not to continue is documented as a refusal here.

**Open issues.** About 80 open issues mention Numba, and phase 3 audits them all. A first pass from
their titles, except where noted:

| Treatment | Issues |
|---|---|
| Closed at the removal, as specific to the deleted code | [#305][305], [#306][306] and [#308][308], which were read, and #327 and #348 |
| Numba's part removed, the rest kept | [#104][104], whose Python Adams march remains, which was read; #7, #281, #344 and [#314][314] |
| Kept | Shared physics and performance work; reference-only gaps such as [#304][304]; [#315][315] |

**Other effects.**

- Validated results are unaffected: `numba_cpu` was never the reference.
- No difference from Luna.jl.
- `KERNEL_SCHEMA_VERSION` and the output schema don't change.
- The `native` extra goes. Installers ignore an unknown extra with a warning, so
  `pip install "solprop[native]"` should still install solprop and native.
- Each solprop release now requires its native wheels, including for a change that touches
  only Python, because of the exact pin. They are already released together (SEP 29).
- A native packaging or load failure is now a complete outage on that platform, which is why §7
  qualifies the wheel as the only production path.

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
  macOS version, so those systems would still fail in the resolver.
- **A plain `uv sync` with no native, through a uv override.** It keeps every first sync
  compiler-free, and a probe showed it working, but a checkout then differs from an install by
  default. Rejected by the maintainer: anyone in a checkout is developing, and the lightweight setup
  covers those who aren't working on native.
- **A refusal-only `solprop-native` wheel for unsupported platforms.** A small pure-Python wheel
  at the same version, which installers take only when no platform wheel matches, could print a
  helpful message at run time. Rejected by the maintainer after review: wheel preference only
  answers a fresh resolution, so a stub already installed survives an upgrade once a real wheel
  appears, and "solprop-native is installed" would have gained a second meaning for a nicer error.
- **Refusing at `import solprop`.** It would break the documentation build, `numpy_ref` by name
  and reading results on an unsupported machine.
- **Sending the platform report automatically.** Nothing leaves the user's machine unless they
  submit it.
- **Other ways to read the processor when `/proc/cpuinfo` can't be read.** Defaulting to v3 risks an
  illegal instruction with no Python error. `numpy.show_runtime()` is the public view of NumPy's
  features, but it prints and returns nothing, so it would mean parsing text meant for reading, and
  it lists only dispatch targets. A small C probe, built for the lower baseline, avoids a private
  name at the cost of a compiled file in all five wheels for a rare case; `_core` can't check for
  itself, being built for v3. A third-party library such as pytorch/cpuinfo adds a dependency for
  one check.
- **Leave Windows v2 out until someone asks.** Considered first. The fixed `NEHALEM`
  OpenBLAS costs one more dependency build and passes the suite, so the exception is not
  worth keeping.

### 11. Phases

SEP 0 puts every phase of an accepted SEP on one branch and one pull request, and exempts tooling,
continuous integration, tests and bug fixes that stand on their own (§2). This SEP keeps to that:

**Already delivered, standing on their own.** These are useful whatever happens to this SEP (Part
II has their commits and runs):

- Two x86-64 builds and Windows on Arm wheels, #395, merged.
- The stable-ABI wheel, #399, on its own pull request, and the fixes that came from this work.

**Standalone, before the removal.** These are bug fixes:

- The processor check of §6.1 ([#443][443]) and §6.2.

**On `feature/sep-0043-retire-numba-cpu`, the SEP's one pull request (#415).** It already merges the
drafts #394, #398 and #401, which close unmerged once it carries their content. It takes each phase
in order:

1. **Lanes.** `docs`, `slow` and `oldest` on the wheel, `native-wheel` on documentation-only changes,
   the `no-native` lane, the strict `poe fetch-wheel`.
2. **Helpers and tests.** `helpers=NUMBA` to plain with §6.3's budgets and the import-blocked test.
   Comparisons with Numba retargeted, with the inventory of §6.4.
3. **Removal.** Delete the backend, `backends/bodies.py`, `_numba_cache.py`, and the tests and golden
   files the inventory has accounted for. Registry tombstone. Remove the dependencies and regenerate
   `uv.lock`. The required dependency, the lightweight setup, the selection of §5 and
   `SOLPROP_REQUIRE_DEVICE`. Old-run tests and the issue audit of §9. The informational native
   coverage run. Documentation, as §8 lists. New ADR superseding ADR-061's Tier 2 and its statement
   on the documentation build. A changelog fragment. Every gate of §7 passes before `Final`.

**After the removal, standing on its own.** The Python 3.15 lane is continuous integration, and
doesn't hold up the removal or 0.2.

**Order against 0.2.** The removal merges before 0.2, so 0.2 ships without Numba. The required
dependency lands with the removal and not before, since until then `numba_cpu` is still the
fallback. Until then, a plain install on Windows on Arm has neither pyFFTW nor native, and `auto`
picks `numba_cpu` and fails with an `ImportError`. Installing `solprop[native]` avoids that. No 0.x
release is expected to carry that state, because 0.2 follows the removal.

### 12. Open questions

None. All were settled by the maintainer on 2026-11-02, and these sections give the result:

- **Q1. What `auto` does without native.** It refuses, with the two existing exceptions of §5.
- **Q2. Whether native becomes a default dependency.** It is required, with no markers (§6.7).
- **Q3. Features that run only on `numpy_ref`.** They stay there and stay supported on the supported
  platforms (§5). There is no supported solprop install without native, even where a feature uses
  only `numpy_ref`.
- **Q4. What `uv sync` does in a checkout.** It compiles native, and one lightweight setup serves
  those who don't need it (§6.7).
- **Q5. Source builds beside the exact pin.** They are labelled when released, rebuilt on upgrade,
  and GPU jobs can require a device with `SOLPROP_REQUIRE_DEVICE` (§5, §6.7).

### 13. References

Issues and pull requests of the solprop repository:

[68]: https://github.com/LupoLab/solprop/pull/68
[92]: https://github.com/LupoLab/solprop/issues/92
[104]: https://github.com/LupoLab/solprop/issues/104
[304]: https://github.com/LupoLab/solprop/issues/304
[305]: https://github.com/LupoLab/solprop/issues/305
[306]: https://github.com/LupoLab/solprop/issues/306
[308]: https://github.com/LupoLab/solprop/issues/308
[312]: https://github.com/LupoLab/solprop/issues/312
[313]: https://github.com/LupoLab/solprop/issues/313
[314]: https://github.com/LupoLab/solprop/issues/314
[315]: https://github.com/LupoLab/solprop/issues/315
[316]: https://github.com/LupoLab/solprop/issues/316
[317]: https://github.com/LupoLab/solprop/issues/317
[319]: https://github.com/LupoLab/solprop/issues/319
[324]: https://github.com/LupoLab/solprop/issues/324
[342]: https://github.com/LupoLab/solprop/issues/342
[363]: https://github.com/LupoLab/solprop/issues/363
[443]: https://github.com/LupoLab/solprop/issues/443

1. [#68][68], the 0.1.0 benchmark record.
2. [#316][316], the decision on `numba_cpu`.
3. [#317][317], Windows on Arm and GPU wheels.
4. [#319][319], Python 3.15; [#92][92], the question of capping `requires-python`.
5. x86-64 microarchitecture levels: *System V Application Binary Interface, AMD64
   Architecture Processor Supplement*, version 1.0.
6. [PEP 440](https://peps.python.org/pep-0440/), version specifiers and local labels, and
   [PEP 817](https://peps.python.org/pep-0817/), Wheel Variants, a draft.
7. [#313][313], qualification on clean minimum systems; [#324][324], native coverage; [#312][312],
   source archives; [#443][443], the processor check; [#315][315], pyFFTW on macOS.

## Part II — Implementation record

### What has been delivered, and which evidence applies to what — 2026-11-02

The SEP isn't accepted, so nothing here is an implementation report under SEP 0 §8. It separates what
has landed from what is only rehearsed, since the phases below were written before any of it.

| Item | Pull request | State | Evidence |
|---|---|---|---|
| Two x86-64 builds, Windows on Arm wheels | #395 | Merged at `673f6ae` | Wheels run 41483897780 on `f8fd5f0`, success. Windows on Arm has its smoke test, not the suite |
| Windows on Arm scan workers | #430 | Merged at `6dcf4f5` | Scans on the Arm runner, in its description |
| Installation page's pointer to that warning | #441 | Merged | Documentation lanes |
| Native refusal fixes | #419 | Merged | Full CI |
| Stable-ABI wheel, 3.15 source builds | #399 | Open | A source build on CPython 3.15.2 on the Xeon. CI on `be1fcb8` still running when this was written |
| Lanes on the wheel, `poe fetch-wheel` | #398 | Draft, with conflicts | CI run 41401116888 on `f2ead18`, success. That is `ci.yml`'s wheel, not the spike's |
| Shared helpers | #394 | Draft | Local tests only. No benchmark |
| Tests retargeted | #401 | Draft | Local and CI |
| The removal | #415 | Draft | Windows on Arm and coverage failing, listed below |

**Which numbers belong to which build.** The v2 cost, accuracy and stable-ABI measurements in the
phase 0 report come from the spike branch, and they are exploratory. They aren't §7's gates, which
run on the final tree. The 3.1e-10 field difference on `he_rdw_averaged`, beside a 3.8e-10
difference between two v3 builds on that case, remains an open qualification item.

**Open items from #415, for §7's gates:**

- **`plasma_band` on Windows on Arm** deviates from its stored value by 3.75457e-06. It isn't
  explained, and no bound has been changed.
- **Windows on Arm's thread variables:** NumPy's own solves crash or hang with small values of them,
  with solprop not involved (#430). The Arm test lane must stop setting them, which also makes it the
  first full suite run on that platform.
- **Python coverage:** 10 lines and 8 branches are missing, in `backends/driver.py`,
  `numpy_ref/admission.py`, `numpy_ref/backend.py`, `numpy_ref/general_modal.py`, `backends/spatial.py`,
  `driver/executors/pool.py` and `translate/problem.py`. The gate isn't lowered: they get tests.

**Windows on Arm's floor** (§6.2) is recorded here when the check lands: what `TARGET=NEOVERSEN1`
lets the linked OpenBLAS use, and the lowest Windows hardware admitted.

### Phase 0 report — the coverage spike

- **Scope:** 2026-11-01 to 2026-11-02, Claude Opus 5.5 for John C. Travers, on the
  unmerged branch `spike/native-coverage` from `05ee1f3`. The branch itself is never merged;
  the table above says what has landed from it. It was run before the proposal was written, to
  find out whether the proposal was possible.
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
- **The required dependency, probes (2026-11-02).** Each is a trial on a disposable worktree from
  `main`, on the Xeon Gold 6240 with GCC 16.2, with pip 26.2.2 and uv 0.12.7 where an installer is
  named. None of it replaces §7's checks.

  | Probe | Result |
  |---|---|
  | The hook writes the pin over a bare name in `dependencies` | Accepted by hatch; `Requires-Dist: solprop-native==0.1.1.dev285+gff2937193.d20261102` in the wheel and in the sdist's `PKG-INFO` |
  | First plain `uv sync` in a fresh worktree, 12 jobs, with FFTW and OpenBLAS already built | 104 s. A second sync took under a second, and `uv run` then found `auto` is `native` |
  | The same with no C++ toolchain in the environment | Stops at CMake's message naming the compiler |
  | `uv sync --no-install-package solprop-native`, with no compiler or CMake on `PATH` | No native package; 493 tests passed, 6 skipped on a sample of modules; `poe types` clean, 270 s. Numba was still present, and the documentation build wasn't tried |
  | NumPy's `__cpu_features__` on 2.4.4, 2.5.0 and 2.5.2, on this Xeon | `X86_V2` and `X86_V3` present and true. With `NPY_DISABLE_CPU_FEATURES=X86_V3`, false. This machine supports both, so the masked SDE run of §7 is the real test, and Windows wasn't run |
  | A plain `uv run` after that sync | Tries to build native, and `UV_NO_SYNC=1` avoids it. `uv sync` ignores `UV_NO_SYNC`, and `uv run` has no flag to skip a package |
  | A real `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SOLPROP_NATIVE=0.3.0+cuda` build | `solprop_native-0.3.0+cuda-*.whl`, version `0.3.0+cuda`, accepted by `==0.3.0`, 105 s. It was a CPU build, and reported `device_backend` `none`, `device_count` 0 and its true commit |

  Resolver behaviour, with fake wheels and each platform simulated by its tags:

  | Installed | Action | pip | uv |
  |---|---|---|---|
  | `1.0+cuda` | Install or `-U` solprop 1.0 (pin `==1.0`) | Kept, also with `--upgrade-strategy eager` | Kept |
  | `1.0`, no label | `-U` solprop 1.0 | Kept, also with eager | Kept |
  | `1.0+cuda` | `-U` when solprop 1.1 is released | Replaced by the 1.1 wheel, with no message | The same |

  A refusal-only wheel was also probed: installers took it only where no platform wheel matched.
  The review's further probe found it survives an upgrade once a real wheel appears, and it was
  dropped (§10).
- **Deviations from the first plan:** the builds sit side by side as `_core` and `_core_v2`
  rather than in subdirectories; the Windows on Arm core keeps the generic Armv8-A baseline,
  so it needs no `PF_ARM_*` check.
- **Sources checked:** Numba, pyFFTW and h5py availability against PyPI and conda-forge on
  2026-11-01 and 2026-11-02.
- **Limitations:** Windows and Windows on Arm timings are not recorded, since hosted runners
  are not fit for it. Below Windows 10 version 2004, the wheels are not tested.
- **Handoff:** #395 brought the wheels over without the spike's rehearsal jobs, and phase 1 takes
  the lanes.

### Phases 1 to 3 report — all on one branch

The status of each gate is at the end of this report, and it says plainly which aren't met.

- **Scope:** 2026-11-02 and 2026-11-03, Opus 5.5 for John C. Travers, on
  `feature/sep-0043-retire-numba-cpu` ([#415](https://github.com/LupoLab/solprop/pull/415)),
  taking in #394, #398 and #401, which closed unmerged, as §11 says.
- **Delivered:**
  - **Selection and floor:** `src/solprop/backends/registry.py` (no fallback, the tombstone,
    `SOLPROP_REQUIRE_DEVICE`) and `native/src/solprop_native/_cpu.py` (`/proc/cpuinfo`, then
    NumPy, then `CpuFeaturesUnknownError`; `armv8.2-a` on Windows on Arm, recorded by
    `native/cmake/HostLibraries.cmake`).
  - **Packaging:** `pyproject.toml`, `hatch_build.py`, `uv.lock`, `tools/setup_light.py`,
    `native/scripts/fetch_wheel.py` (strict), `native/ci/install_solprop.py`,
    `native/scripts/check.sh`, `.readthedocs.yaml` and the three workflows.
  - **Removal:** `backends/numba_cpu/`, `backends/bodies.py`, `_numba_cache.py`, the 28
    `test_numba_*` modules and the 7 golden files, as accounted for in the inventory below.
  - **Tests:** `tests/driver/test_retired_backend_runs.py` with real files in
    `tests/data/retired_backends/`, and the ported obligations below. `native/ci/coexistence.py`
    keeps pyFFTW and Numba as outside packages.
  - **Documentation:** the pages §8 lists, ADR-064, the gallery on `auto` with regenerated
    output, and `changelog.d/sep-0043-*`.
- **Deviations from the proposal:**
  1. **Files written by 0.1.0 don't load (§9 says they do).** Published 0.1.0 wrote output
     schema 1, and this tree reads schema 2 only, by a policy that predates this SEP
     (`output/schema.py`, `_refuse_schema`). A real 0.1.0 file is refused on either backend,
     saying to read it with the version that wrote it. §9's table holds for a schema 2 file:
     a real stopped run from the last commit before the removal continues on `native` and
     `numpy_ref`. A resumed run's own record wasn't checked beyond the saved planes staying as
     written.
  2. **An unpinned install doesn't always fail (§5, §6.7 say the installer's error is the
     answer).** Resolvers backtrack: on a platform with no native wheel, `pip install solprop`
     and `uv pip install solprop` install 0.1.0, with Numba and pyFFTW, and no error. On a local
     index with fake wheels, yanking 0.1.0 restores the error, and `solprop==0.1.0` still
     installs. Yanking is the maintainer's decision. Until then the installation page says so
     and suggests `solprop>=0.2`.
  3. **§5's fourth `numpy_ref` feature isn't one.** "Beyond six terms on three components"
     misread `PAIR_HISTORIES = 6`. `numpy_ref` also requires `DELAYED_TERMS = 2`, and the only
     case native refuses and it accepts is two terms on other than three components, which the
     public route doesn't produce. The guide leaves the row out. The drafter's error, from
     Part II of the spike's report.
  4. **The processor check and Windows on Arm's floor went on the one branch, not standalone
     (§11).** The maintainer asked to consolidate.
  5. **With no visible device, the guard reports a load failure.** Kokkos stops as the package
     loads, before the guard can read the build, so `_check_required_device` turns that into the
     `BackendError` §5 promises, carrying Kokkos's message.
  6. **Class T in the old-run test uses `tests/march_bounds.py`'s reference at `rtol`/100 and
     factor 4,** the suite's standing rule, not §7's `rtol`/10 and 2, and it takes every plane:
     at the last plane an uninterrupted run is about 1e-11 from the reference by coincidence,
     against about 1e-8 at the third, on every backend.
  7. **`pip install "solprop[native]"` wasn't tried** after the extra went, so §9's "should still
     install" stays unverified.
- **Decisions:** the coexistence environment installs pyFFTW only where it has a wheel. A test
  environment with no native package takes `auto` as `SOLPROP_TEST_BACKEND` for the session, and
  `real_auto` keeps the refusal; a test whose child process runs `auto` skips there
  (`child_runs_auto`). Windows on Arm's lane is `poe test-windows-arm`, which sets no small BLAS
  thread variables and limits the pools after they load.
- **Verification:**

  | Check | Result | Where |
  |---|---|---|
  | Default lane on native, local | 17,530 passed, 46 skipped, 3 xfailed | Xeon Gold 6240, typecheck off |
  | Lightweight setup, no compiler on `PATH` for the sync | 15,656 passed, 769 skipped, 2 xfailed; documentation build clean under `-W` | Xeon, numpy_ref |
  | Gallery on native under `-W` | Built. Printed values match apart from wall times. Two figures moved visibly through the default shot-noise draw: with `noise=None`, native and `numpy_ref` agree to 1e-16 in the field and 7e-14 in the ionised fraction | Xeon |
  | Helper budgets, §6.3 | Tracking set geometric mean 1.0009, the same work on all 22. `construction_s`: at most +0.053 s and +0.9 %, against the larger of 0.5 s and 20 %. Host march 1.0001, 0.9999, 0.9935 against A/A 1.0020, 0.9957, 0.9946. `numpy_ref` 1.0016 against A/A 1.0038, fields bit-identical | EPYC 7763, whole node, fresh processes, against `main` at `ff29371` |
  | Field differences in that comparison | At most 1.0e-10 (`he_rdw_averaged` at 16 threads, 9.1e-11 on `ctc`), the cases with the spike's known floor; the rest at or below 1.1e-12 | EPYC 7763 |
  | `SOLPROP_REQUIRE_DEVICE`, real CUDA build labelled `0.3.0+cuda` | Every case as §5 says. Device suites 31 and 63 passed, 2 `mlx` skips | NVIDIA A40 |
  | Labelled builds, real wheels, pip 26.2.2 and uv 0.12.7 | `0.3.0+cuda` kept at 0.3.0 through install, `-U` and eager; replaced by the CPU wheel at 0.3.1 | Xeon, a CPU build |
  | Pin in built metadata, through the real hook | `Requires-Dist: solprop-native==<version>` in the wheel and the sdist's `PKG-INFO` | Xeon |
  | Coexistence | Passed locally with no outside packages. The run with pyFFTW and Numba installed is CI's | Xeon |
  | Informational native coverage, §6.4 | Migrated Python suite and `native/tests`: lines 87.48 %, functions 87.80 %, regions 87.99 %, branches 80.80 %. `core.cpp` at 15.7 % is the `Workspace` class, 101 of 120 functions, which the gtest binary tests and this run left out | Xeon, Clang 23, `results/native-coverage` |

- **The inventory (§6.4):** `results/sep43-test-inventory.md` in the campaign directory, not in
  git: 446 removed functions, of which 192 carried, 147 only the deleted implementation, 102 lost
  and 5 unconfirmed, under a strict rule that counts a looser bound as lost. Ports followed, with
  no bound loosened except as listed:
  - **P1, items 1 to 16:** all ported. Items 1, 2, 10 and 16 have weaker witnesses, because native
    exposes no time-domain current: the plasma column's per-sample budget is held at the stage, so
    an error confined to a few samples could pass. A test-only binding would restore it.
  - **P2, items 17 to 40:** ported or weaker, with item 30 not portable, since translation refuses
    a flow on the only non-identity crop.
  - **P3 and P4, items 41 to 83:** ported or weaker. Items 62, 65 and 82 aren't ported, and neither
    are the pad and the 1 ulp rate from the unconfirmed rows.
  - **Self-comparisons:** `test_the_two_propagator_modes_agree` and the coupled carry's "no route"
    are fixed. In a lane with no native package, parity tests still compare `numpy_ref` with
    itself, and the terminal says so.
  - **Changed after measuring, for review:** item 55's flow factors are exact on constant flows
    only. On tapers, the frozen block varies at about 1e-17 with BLAS width and Schur vectors aren't
    unique, so these took `2 gamma(n_deg)` and `10 n^2 u`. One accept residual went from per column
    to normwise, since transverse FFT rounding is normwise.
- **Sources checked:** NumPy's `npy_cpu_features.c`, FFTW's `x86-cpuid.h`, `avx2.c` and `neon.c`, and
  OpenBLAS's `dynamic.c`, `dynamic_arm64.c`, `cmake/cc.cmake` and `Makefile.arm64`, for how each
  reads a processor and what `TARGET=NEOVERSEN1` compiles for: `-march=armv8.2-a -mtune=neoverse-n1`.
  So Windows on Arm's floor is `armv8.2-a`, checked through CRC32 and the LSE atomics, the two parts
  Windows can report. The lowest Windows hardware admitted wasn't established.
- **Limitations:**
  - No hardware below the floor was run. Selection is by stand-ins and Intel SDE in CI.
  - `PF_ARM_*` has run only against a fake `kernel32`, until CI's Windows on Arm lane.
  - The Windows 10 minimum isn't qualified, so that claim narrows until #313 has a machine.
- **Open questions, each for the maintainer:**
  - **Q6.** Yank 0.1.0, so that deviation 2 becomes the error §5 describes?
  - **Q7.** `envelope_axes` and `envelope_moments` in `envelope_plasma.hpp` aren't protected from
    contraction, as Numba's were. Protect them, or drop them from the licence? Strict xfails in
    `test_native_floating_point_licence.py`; no numerical error seen.
  - **Q8.** A mistyped `SOLPROP_FFTW_PLAN` is silently MEASURE (`fft_plan.hpp`). Refuse it? Strict
    xfail in `native/tests/test_fftw_plan_refusal.py`.
  - **Q9.** Native's finish with content off the band differs from the reference by 1.09e-8 against
    3.2e-12, since the reference masks before and after the temporal absorber and native only after.
    No march reaches it, and on-band content measures 7.6e-16. Strict xfail.
  - **Q10.** Native ignores a truncated `response_order`, with no refusal; it exists only at contract
    level.
  - **Q11.** The multimode capillary's interaction march differs by 1.5e-10 across 1 and 4 threads and
    1.3e-10 between repeats, and by 0.0 under `SOLPROP_FFTW_PLAN=estimate`: steps 1.9e-12 apart,
    amplified about 65 times, unexplained. Above §7's flag, within the retired 3e-9.
  - **Q12.** Surviving thread tests may compare serial with serial, since native sizes each operation
    by its work (`test_the_flow_is_invariant_under_the_thread_count`). The new ones force the pools.
    The others weren't audited.
  - **Q13.** The issue audit (`results/sep43-issue-audit.md`, not in git) has closing and trimming
    text for 60 of 79 issues, and nothing has been posted. #300 needs a timing on an idle machine.
    #92's priority should rise, since a wrong refusal under `auto` now stops a run.

## Part III — History

- 2026-11-02: Drafted. Tracking issue [#392](https://github.com/LupoLab/solprop/issues/392).
- 2026-11-02: The maintainer settled Q1 and Q2: native is a required dependency, a platform
  without it is unsupported and refuses with a request to report it, `numpy_ref` is a
  development tool and `mlx` stays a special backend. §6.7, §8 and Q3 to Q5 added. Q3 then
  settled: features that run only on `numpy_ref` stay there, and every supported install has
  native. Q4 and Q5 then settled: a plain `uv sync` in a checkout compiles native, and source builds
  are labelled, rebuilt on upgrade and can opt into refusing a CPU-only build.
- 2026-11-02: AI review by GPT-6, with the maintainer's decisions on R1, R2, R12 and R13. Part I
  revised in response: no refusal-only wheel, removal before 0.2, a selection table, the
  lightweight setup, `SOLPROP_REQUIRE_DEVICE`, numerical acceptance criteria, and §11's phases.
- 2026-11-02: The maintainer settled how the processor is read on Linux: `/proc/cpuinfo`, then NumPy's
  features, then a refusal (§6.1, R8).
- 2026-11-02: Accepted by John C. Travers. Part I is frozen.
- 2026-11-02: Implementation started on `feature/sep-0043-retire-numba-cpu`, taking in #394, #398 and #401.

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

   **Maintainer decision — 2026-11-02 (John C. Travers):** on Linux, read `/proc/cpuinfo`, then
   NumPy's `__cpu_features__` if that fails, and refuse if neither answers. Windows x64 keeps its
   current query, and Windows on Arm gains a `PF_ARM_*` query.

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

### Response to review — 2026-11-02 (Opus 5.5, drafting agent)

Before this response, the review's claims about the code were re-read and held: `_tdse_entry` and
`_single_backend` (R1), `select_variant` and `baseline_shortfall` choosing v3 with no `/proc/cpuinfo`
(R8), `fetch_wheel.py` only warning (R3), the operand budget in `test_numba_band_bounds.py` (R6), and
the issues cited. New probes are in Part II. Review status stays `In review`: a human review is still
to come. All 16 points are taken, except as stated in R6, R9 and R14 on where the work happens. Items
marked **phase** are qualification assigned to a phase, and are not yet done.

1. **R1 taken, as the maintainer decided.** §5 has the selection table, by request and by native's
   state, with the TDSE and single-precision exceptions stated. Native's installation is a packaging
   policy, and nothing loads a runtime to enforce it. `numpy_ref` is the numerical reference and the
   executor of a listed set of supported features, with no support promise for running it elsewhere.
   The list isn't a waiver of SEP 0's parity rule. §8 carries the wording, including TDSE. I also kept
   single precision on the list, as in Q3. Its message today calls `numpy_ref` "not a production
   backend", which §8 changes, so the maintainer may want to check that listing.
2. **R2 taken, as the maintainer decided.** The refusal-only wheel, and its verification and release
   gate, are out of §5, §6.7 and §7. §10 records why, and §6.7 has the installation page explaining the
   installer's error. Runtime diagnostics stay for a low processor and a broken installation.
3. **R3 taken.** §6.7 defines three checkout states and makes the exception explicit, with metadata
   deliberately mismatched and reported only for a fetched wheel. `poe fetch-wheel` becomes strict: a
   difference from the artifact's commit in `native/` or the solprop modules native imports refuses,
   including uncommitted changes, and an expired artifact refuses. The wheel keeps its true identity.
   `KERNEL_SCHEMA_VERSION` stays the guard at run time. The real hook was exercised for wheel and sdist
   (Part II), and labels are defined for released and development builds. §7 has the five cases.
   **Phase 1** implements it.
4. **R4 taken.** §5 specifies `SOLPROP_REQUIRE_DEVICE`: values, no competing precedence, when it's
   checked, what it reads, and that it refuses a CPU build, a wrong backend, no visible device, and
   `numpy_ref` or `mlx`. It's opt-in, in the GPU job recipe. A probe supports reading the build, since a
   CPU build labelled `+cuda` reported no device. §9 records the second environment's cost, with #363.
   §7 adds real labelled-build tests. **Phase 3** implements it. I had no device for this, so the
   positive case and a device build's failure at Kokkos start-up are untested.
5. **R5 taken.** §4 and §7 replace step-count equality with named metrics, class R on the existing
   ladder, and class T with a reference at `rtol`/10 and a factor of 2 fixed now. Work counts are a
   diagnostic. Performance has its own criterion. Part II marks the spike's results as exploratory and
   keeps that excess open. **Phase 3** measures it.
6. **R6 taken, with the inventory assigned, not done here.** §6.4 makes it a phase 2 deliverable and a
   §7 gate, keeps scientific invariants and bounds, and forbids a comparison of an implementation with
   itself. Deleting those modules now waits for it. Historical readers stay. **Phase 2.**
7. **R7 taken.** §7's first rows are the final suite with no filters on all five platforms, forced v2,
   slow lanes, a failure on unexpected skips, fresh installs that propagate, and minimum systems, with
   the claim narrowing and #313 keeping the remainder where a machine isn't available. §5 keeps the
   failure cases distinct, and §7 exercises the no-package case. The refusal-only case fell away with R2.
   **Phase 3.**
8. **R8 taken, as the maintainer decided.** §6.1 asks `/proc/cpuinfo`, then NumPy's dict, and refuses
   before loading if neither answers. Windows x64's query can't realistically fail, so it isn't part of
   that order. §10 records the alternatives. I kept a named variant as the deliberate way past on x86-64,
   so a restricted environment has a route; Linux aarch64 has a single build, so it has none. §6.2 drops
   the statement that no check is needed, adds a `PF_ARM_*` query for Windows on Arm, and replaces the
   installation page's claim. As the review says, this doesn't establish that the wheel uses an
   unsupported instruction. Both are **standalone fixes** before the removal, under #443.
9. **R9 taken, choosing the run over deferral.** §6.4 keeps Python's gate and commissions one
   informational native coverage run with no percentage, in phase 3. #324 keeps the question of a
   standing collector. §6.3 and §7 add the test blocking native, Numba and pyFFTW from `numpy_ref`.
10. **R10 taken.** §6.3 says who changes and who keeps their helpers, sets budgets for construction, a
    host march and a reference run, keeps the strict arithmetic, and requires a rounding argument for
    any vectorised form. The budgets are my proposal, fixed before measuring. **Phase 2.**
11. **R11 taken.** §9's table says old runs are readable, and resumable on native or `numpy_ref` by
    changing the backend deliberately, under SEP 42's checks, keeping the original record. This rests on
    `resuming.md` and SEP 42, which keep the backend out of the run's identity, and on the existing
    cross-backend tests. Whether a new execution record is added on every route isn't confirmed. **Phase
    3** tests it from files written by 0.1.0, and documents any refusal.
12. **R12 taken, as the maintainer decided.** §6.7 keeps the build by default and specifies
    `uv run --no-project tools/setup_light.py`, its limits, and why uv needs the two settings. A probe
    with no compiler passed a sample of tests and the type check. The documentation build is left to §7.
    §2 states that archive and offline builds stay with #312. Its importance does grow with a required
    native, though it doesn't block this. **Phase 3**, with the required dependency.
13. **R13 taken, as the maintainer decided.** §9 and §11 remove `numba_cpu` before 0.2, with 0.1.0 as the
    last release that has it, and rollback corrected for what a revert needs. §7 now says that reporting
    a failure doesn't substitute for passing. Part II records what is delivered, with commits and runs,
    and which measurements apply to what. §11 interprets SEP 0: remaining phases go on the one branch, so
    #394, #398 and #401 close into #415, while #395, #399 and the bug fixes stand alone under §2.
14. **R14 taken, with a limit.** §6.6 says these are ordinary CPython wheels, separates compatibility
    from support, names the tested interpreters, and keeps 3.15 a qualification lane until h5py has wheels.
    It stays independent of the removal. The short run on each platform can't go through solprop where h5py
    has no wheel, so it uses native directly there. **After the removal.**
15. **R15 taken.** §6.4 keeps the coexistence check in a dedicated environment with outside packages,
    keeps the macOS refusal with new advice, and keeps host BLAS management. §7 has the gate. **Phase 3.**
16. **R16 taken.** §2 and §3 bound the speed claim to what was measured and say "no known users". §3 calls
    dropping those systems a choice. §9 replaces the blanket closure with a first-pass audit, read for four
    issues and by title for the rest, with the full audit in **phase 3**. §6.5 and §8 keep `auto` in examples
    that teach nothing specific to a backend.
