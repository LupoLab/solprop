# Releasing and the native wheels

solprop is released as two distributions from one repository, at one version taken from
the git tag ([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md)):

- **`solprop`**, pure Python: a source distribution and one universal wheel.
- **`solprop-native`**, the compiled backend built from `native/`: **wheels only**, one
  per supported platform. No source distribution is published, so that where no wheel
  matches, `pip install "solprop[native]"` fails to resolve instead of starting a C++
  build; source builds are made from the repository
  ([Building the native backend](../guide/performance/building-native.md)).

Versions are 0.x for now: a minor release may break the API, the output schema or the
resume key, and every break is listed under "Changed (breaking)" in the changelog
(ADR-058 in [decisions](decisions.md)). The first release is 0.1.0.

The wheels are built, repaired, qualified and published by
`.github/workflows/wheels.yml`. cibuildwheel's configuration is the
`[tool.cibuildwheel]` table of `native/pyproject.toml`; the libraries each wheel bundles
are built from pinned, checksummed sources by the scripts in `native/ci/`, each of which
also runs by hand.

## The wheels

| | Linux x86-64 | Linux aarch64 | macOS arm64 | Windows x64 |
|---|---|---|---|---|
| CI runner | `ubuntu-24.04` | `ubuntu-24.04-arm` | `macos-15` | `windows-2025` |
| build environment | `manylinux_2_28` container, GCC 14 | `manylinux_2_28` container, GCC 14 | Homebrew's Clang, deployment target 14.0 (macOS Sonoma) | Clang on the MSVC ABI (MSVC STL and CRT, `/MD`) |
| CPU baseline | x86-64-v3 | Neoverse N1 | Apple M1 | x86-64-v3 |
| host execution space | Kokkos OpenMP | Kokkos OpenMP | Kokkos OpenMP | Kokkos OpenMP |
| OpenMP runtime | GCC's `libgomp` | GCC's `libgomp` | LLVM `libomp`, built for the target | LLVM `libomp` from the LLVM installation |
| BLAS | OpenBLAS 0.3.34, OpenMP, `DYNAMIC_ARCH`, `solprop_` prefix (built by the configure step) | the same | Accelerate | OpenBLAS 0.3.34, OpenMP, one fixed Haswell target (no `DYNAMIC_ARCH`), static |
| FFTW 3.3.11 | shared, SSE2 and AVX codelets | shared, cycle counter, NEON codelets | shared, cycle counter, NEON codelets | static, SSE2 and AVX codelets |
| vector maths (ADR-059) | SLEEF 3.9.0, AVX2, static | the system libm | the system libm | SLEEF 3.9.0, AVX2, static |
| repair | `auditwheel` | `auditwheel` | `delocate` | `delvewheel` (mangled names) |

Every bundled library runs on the same OpenMP runtime as Kokkos, so a process holds one
thread pool for the backend. The wheel carries the licence text of every library compiled
into it or copied in by the repair (`LICENSE-*.txt` beside the extension). CPython 3.14
only; free-threaded CPython is not a target.

## Building a wheel locally

Every wheel is the distribution profile (`SOLPROP_NATIVE_PROFILE=DISTRIBUTION`), never
a build for the build machine's own architecture.

**macOS arm64.** cibuildwheel needs python.org's framework Python, which it installs only
on CI. `native/ci/build-wheel-macos.sh` runs the same steps with uv: it builds `libomp`
and FFTW for the deployment target (`native/ci/build-deps.sh`), builds the wheel and
repairs it with `delocate`.

```sh
brew install llvm cmake ninja
native/ci/build-wheel-macos.sh /tmp/wheels   # the repaired wheel lands in /tmp/wheels
```

Keep the script and `[tool.cibuildwheel.macos]` in step: they set the same deployment
target, CMake arguments and `delocate` check.

**Linux** (x86-64 or aarch64, on a machine of that architecture with Docker):

```sh
uvx cibuildwheel --platform linux --output-dir wheelhouse native
```

**Windows x64**, from a Visual Studio developer prompt (the x64 MSVC tools and SDK in
the environment), with LLVM, CMake and Ninja installed:

```sh
uvx cibuildwheel --platform windows --output-dir wheelhouse native
```

To build one platform's dependencies by hand: `native/ci/build-deps.sh PREFIX` (Linux,
macOS) or `native/ci/build-deps-windows.ps1 -Prefix PREFIX`.

## Qualifying a wheel

The workflow's test jobs install each repaired wheel beside solprop from the same
checkout and run, in order:

1. `python -I native/ci/smoke.py --version VERSION`: the package imports and passes
   `check_build()`, is the distribution build at the release's version, loads the FFT,
   BLAS and OpenMP libraries from inside the wheel, and carries their licences; on Arm,
   both precisions of its FFTW carry the NEON codelets;
2. `python -I -m solprop_native`: the build record;
3. `python -I native/ci/coexistence.py`: NumPy, SciPy, pyFFTW and Numba imported before
   and after `solprop_native`, each order in a fresh process, every package's thread
   pool started, and a short propagation run on the native backend, on `numba_cpu` and on
   the native backend again in the same process;
4. `uv run --no-sync poe test` with `SOLPROP_TEST_BACKEND=native`: solprop's default
   lane on the native backend.

On Linux x86-64, the `cpu-floor` job runs the wheel under Intel's Software Development
Emulator: the smoke test on a Haswell (x86-64-v3); a refusal before any native code runs
on a Nehalem (x86-64-v2, no AVX) and a Sandy Bridge (AVX without AVX2, FMA or BMI), naming
the missing features; and, on the Sandy Bridge, `backend="auto"` resolving to
`numba_cpu` with its one warning. It runs when the repository variables `SDE_URL` and
`SDE_SHA256` name an SDE release. The other platforms' checks (Windows through
`IsProcessorFeaturePresent`, Linux aarch64 through `/proc/cpuinfo`'s `Features`) are
unit-tested with stand-in processors in `native/tests/test_cpu_baseline.py`; no CI job
runs them on a processor below the baseline.

**macOS and pyFFTW.** pyFFTW's macOS wheel carries its own `libomp`, which it initialises
when it is imported. LLVM's OpenMP runtime refuses, or crashes on, a second initialised
copy, so the native package raises at import when pyFFTW is already loaded
([OpenMP runtimes and pyFFTW](../guide/performance/building-native.md#openmp-runtimes-and-pyfftw)).
solprop imports both backends lazily, and on macOS its `numba_cpu` backend imports the
native package, when installed, before pyFFTW, so a script that only calls solprop can
run both backends in either order; only a script that imports pyFFTW itself first meets
the refusal, whose message says what to do.
`coexistence.py` expects that refusal on macOS; `--bypass-guard` adds a report-only
scenario that hides pyFFTW from the guard, to test whether it is still needed.

**solprop's distributions.** The `solprop-dist` job builds solprop's sdist and wheel and
installs each from the file in a fresh environment, outside the checkout: the wheel
alone, then the wheel and the sdist each with the `native` extra beside this run's
Linux x86-64 wheel. In each it runs the check of the
[installation page](../introduction/installation.md#checking-the-installation): solprop
imports at the distributions' version, which on a release tag must be the tag's, and
`backend="auto"` resolves to `numba_cpu` without the extra and to `native` with it.
The extra pins solprop-native at solprop's own version (`hatch_build.py`), so two
distributions built at different versions fail here, before anything is published.

## Making a release

### Pre-flight checklist

A release tag builds and tests everything but publishes nothing until a maintainer with
admin rights on the repository and an account on PyPI has set up publishing. Once,
before the first tag:

1. **On PyPI**, under *Your account → Publishing*, add a pending trusted publisher for
   each project, both naming the owner `LupoLab`, the repository `solprop` and the
   workflow `wheels.yml`:
   - the project `solprop` with the environment `pypi`;
   - the project `solprop-native` with the environment `pypi-native`.

   PyPI ties a pending publisher to one project name per workflow and environment, hence
   the two environments. The first upload through a pending publisher creates its
   project.
2. **On GitHub**, under *Settings → Environments*, give both `pypi` and `pypi-native` a
   required reviewer, and restrict their deployments to *Selected branches and tags*
   with the single tag rule `v*`. Without the reviewer, anyone with write access could
   publish by pushing a tag.
3. Under *Settings → Secrets and variables → Actions*, add the repository variable
   `PUBLISH` with the value `true`. The publish jobs are skipped without it.
4. **On Zenodo**, signed in with GitHub, switch on `LupoLab/solprop` under *GitHub*
   (the organisation must grant Zenodo's application access). Zenodo archives each
   GitHub release, not a tag, with the metadata of `.zenodo.json`, and gives it a DOI,
   plus one concept DOI for every version.

Then, for every release:

5. On `main`, check that the changelog's `Unreleased` section describes the release,
   then move it under the new version, `## [X.Y.Z] - YYYY-MM-DD`, and give
   `CITATION.cff` the same `version` and `date-released`
   (`tests/test_release_metadata.py` checks that they agree). The commit that does so
   is the release commit.
6. Freeze the release's run records (ADR-063). For 0.2.0, refresh
   `tests/output/test_record_stability/frozen/0.2.0` with `write_frozen("0.2.0")` in
   `tests/output/test_record_stability.py`, on the tree being released, and include it in
   the release commit. For every later release, once it is tagged, check out the tag, run
   `write_frozen("X.Y.Z")` there and commit the new folder to `main`: the records come
   from the tree that was released, and are never regenerated. When a minor release is
   out, delete the folder of the release before it, and the entries of
   `solprop.output.compatibility.NEUTRAL_VALUES` that name that release.
7. Check that CI passed on the release commit. When it merged a pull request whose head
   has the same tree and passed CI, the push to `main` runs none of CI's lanes
   ([CI](tooling.md#ci)), and that pull request's run is the release commit's record.
8. With the release commit at the head of `main`, run the wheels workflow on it,
   `gh workflow run wheels.yml --ref main`, and wait for it to pass. It is the only
   workflow that builds and tests the Linux aarch64 wheel, and on `main` it otherwise
   runs only weekly. On a branch the publish jobs are skipped, so this is a dry run of
   the release.
9. Then tag that commit, as below.

### Tagging and publishing

1. Tag the release commit, `git tag -a vX.Y.Z COMMIT`, and push that tag on its own,
   `git push origin vX.Y.Z`: GitHub starts no workflow for a push of more than three
   tags at once. The tag starts the wheels workflow again, on the tagged commit (CI
   does not run on tags): it builds and tests every wheel, and builds solprop's
   distributions and checks an install of each. Both publish jobs wait for every
   wheel's `test` job and for `solprop-dist`, so the run that publishes has itself
   tested the release commit on every wheel platform, whichever lanes CI skipped on
   the push to `main`.
2. Publishing, through PyPI's trusted publishing, then waits for each environment's
   reviewer. `publish-native` uploads solprop-native's wheels from `pypi-native` first;
   `publish` uploads solprop's sdist and wheel from `pypi` only once every native wheel
   is up, since `solprop[native]` requires solprop-native at the same version. If an
   upload fails partway, re-run the failed jobs: they upload the same files, and PyPI
   accepts a file identical to one it holds. It refuses different files under a name it
   has seen, so a release that needs a change takes a new version.
3. Wait for `published-install`, which starts after both publish jobs succeed. On
   Linux x86-64, Linux aarch64, macOS arm64 and Windows x64 it creates a fresh CPython
   3.14 environment outside the checkout and uses pip to resolve and install
   `solprop[native]==X.Y.Z` from PyPI. It checks both distributions' versions,
   `pip check` and that `backend="auto"` resolves to `native`. It uses no local
   distribution, dependency lock or cached environment. Its per-platform artifacts
   retain the install report, logs, resolved versions and platform information,
   including on failure; each job also writes a summary. Check these before publishing
   the GitHub release. A failed check does not undo the PyPI upload: record the failure
   in an issue and repair it in a new version if the published files need changing.
   If PyPI has not yet made the new files available, rerun the failed install jobs once
   they appear; the check does not retry a failed resolution or hide its first result.
4. Publish a GitHub release from the tag, with the version's changelog section as its
   notes: `gh release create vX.Y.Z --verify-tag --title "solprop X.Y.Z" --notes-file
   NOTES.md`. Zenodo archives it and gives it a DOI; on the first release, add the
   concept DOI, which names every version, to `CITATION.cff` and the README.

### Rechecking a published release

The **Published install** workflow can also be dispatched by hand with a required
`version`, written as `X.Y.Z` or `vX.Y.Z`. It runs the same four-platform check without
building or publishing anything:

```sh
gh workflow run published-install.yml -f version=0.1.0
```

Its optional `readme_example=true` input additionally runs the first Python block of
that release's tagged README as a script, using Matplotlib's noninteractive `Agg`
backend. The source and output join the evidence artifact. This full-length example
is a manual release qualification, requested explicitly; the automatic post-publish
check only installs and checks the backend. It does not propagate a pulse. The manual
example is available to complete the first release's outstanding platform checks
([issue 74](https://github.com/LupoLab/solprop/issues/74)).

```sh
gh workflow run published-install.yml -f version=0.1.0 -f readme_example=true
```

Before the new workflow is on `main`, GitHub cannot dispatch it by name. The existing
**Wheels** workflow also accepts `published_version`: with that input it skips the
wheel builds and their dependent checks and runs only the published-install check.
The publishing jobs remain dependent on the skipped builds, so this mode cannot
publish. It has a separate concurrency group and does not cancel a wheel build on the
same branch. Select the branch whose workflow is being qualified:

```sh
gh workflow run wheels.yml --ref fix/published-install-checks \
  -f published_version=0.1.0 -f readme_example=true
```

Leaving `published_version` empty keeps the ordinary Wheels workflow unchanged.

The actions in the workflows are pinned by commit; when updating one, take the commit of
its release tag and keep the version in the trailing comment.

The release recipe on this page was set up and qualified by
[SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md).
