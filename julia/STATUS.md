# Julia experiment: status and handoff

**Draft experiment — do not merge. CPU only. No CI qualification.**

This branch demonstrates that Julia can own solprop's numerical backend while the
Python frontend, translation, shared controller and output system remain in use.
It is useful for trying cases, reproducing short timings and continuing development.
It is not ready for production integration or a claim of complete backend parity.
The experiment started from `1df3e4117d7f`, with kernel schema 34. Porting to the
separate SEP 34 medium-flux implementation is future work; do not assume a clean
merge would establish numerical compatibility.

Start with the [installation and runnable example](README.md#prepare-a-checkout),
then the [small Julia/native comparison](README.md#start-a-small-julianative-comparison).
The [manual](../docs/guide/performance/julia.md) describes the numerical and thread
contracts. The [benchmark ledger](../docs/dev/benchmarks.md#experimental-julia-cpu-prototype)
contains the measured windows, providers, spreads, correctness evidence and rejected
optimizations, including the final Mac and Xeon comparisons.

## What works

| Area | Implemented subset |
|---|---|
| Capillaries | Resolved and mode-averaged scalar field grids, constant radius, separable longitudinal material profiles |
| Mode-averaged envelopes | GNLSE Kerr, constant scalar Raman and translated self-steepening |
| Radial free space | Scalar fields in a constant medium, Kerr and one plasma channel |
| Cartesian and line free space | Scalar Kerr fields/envelopes and one field-plasma channel |
| Responses | Kerr with/without THG on fields; one scalar plasma channel; one constant scalar Raman component on admitted modal/radial routes |
| Integration | DOPRI5 with FSAL; resident exponential Adams on non-Cartesian routes; prescribed RK4IP on Cartesian/line routes |
| Output | Normal save planes, hard masks, both temporal-absorber schemes, default modal/spatial and supported plasma statistics |
| Platforms exercised | Apple Silicon macOS and Linux x86_64, private Julia 1.13.2, Python 3.14 |

The admission checks explicitly refuse unsupported combinations. In particular:
GPU execution, vector polarization, batching, noise, interpolated planes, tapered
or general modal guides, Cartesian Raman/Adams, resolved/radial envelopes,
envelope plasma/harmonics, multiple plasma/Raman components and advanced
dispersive/regional responses remain outside the experiment. See the manual and
`_refuse` in `src/solprop/backends/julia/_backend.py` for the detailed boundary.
Automatic backend selection is unchanged: request `backend="julia"` explicitly.

## Implementation size

Physical source lines, including comments and blank lines, counted on 2026-10-24:

| Implementation | Numerical code and bindings | Python adapter | Total |
|---|---:|---:|---:|
| Julia prototype | 1,395 | 1,320 | **2,715** |
| Native backend | 24,155 | 4,487 | **28,642** |

The prototype is approximately **10.5 times smaller by this measure** (90.5% fewer
lines). Julia's numerical column is the six `julia/src/*.jl` files. Native's is all
`.cpp`/`.hpp` under `native/src/` and `native/include/`, except `benchmark_gemm.cpp`
and `coverage_labels.cpp`, plus the 320-line `_core.pyi` binding declaration. Python
columns count `src/solprop/backends/julia/*.py` and
`native/src/solprop_native/*.py`, respectively. Tests, documentation, build files,
benchmarks and external dependencies are excluded from both totals.

This is an implementation-footprint comparison, **not equivalent feature coverage**.
Native has a much larger capability matrix and CPU/CUDA/HIP implementations. Julia
also uses compact formatting, including multiple statements on some lines. Both
reuse solprop's Python frontend and shared driver, and both rely on numerical
libraries. A future matched-capability comparison and readability review would be
more informative than treating this ratio as a language productivity benchmark.

## Performance so far

Julia is competitive on several realistic CPU cases, but it is not universally
faster than native. The final Mac round used an Apple M5 Max with 18 physical cores
and 128 GiB, **native's default Mac build settings**, and Julia with Accelerate
BLAS and FFTW. These are median seconds for ten prescribed steps, six warm samples
per cell from three alternating rounds. Setup, compilation and final observations
are outside these propagation timers; first-window/setup costs were recorded
separately. DOPRI cases use 61 source evaluations; Cartesian RK4IP uses 40.

| Case | Native 1 thread | Julia 1 thread | Native 18-thread budget | Julia 18-thread budget |
|---|---:|---:|---:|---:|
| RDW | 0.128269 | 0.157036 | 0.070999 | 0.068320 |
| CtC | 6.288408 | 5.325555 | 2.685244 | 2.107181 |
| Radial filament, 256 nodes | 0.649903 | 0.794537 | 0.288600 | 0.855188 |
| Averaged helium RDW | 0.010500 | 0.017543 | 0.010590 | 0.017662 |
| Resolved argon blueshift | 0.056584 | 0.050703 | 0.022913 | 0.019836 |
| Hydrogen Raman, 6 ps | 1.194044 | 1.051791 | 0.243912 | 0.437328 |
| GNLSE soliton | 0.005278 | 0.006534 | 0.005524 | 0.006570 |
| Averaged argon blueshift | 0.005019 | 0.005053 | 0.005156 | 0.005084 |
| Cartesian 192² × 300 | 9.334232 | 15.488092 | 1.217616 | 1.764022 |
| Cartesian 384² × 600 | 89.175180 | 77.226190 | 10.626641 | 11.544229 |

Julia is 1.27× faster for parallel CtC, 1.15× for parallel resolved argon, and
1.15× for serial large Cartesian propagation. The large Cartesian parallel result
is 8.6% slower than native. Native wins the radial and parallel long-pulse Raman
cases by substantial margins. Radial parallel timings were noisy, but their
ranges remain separated. The ledger records those ranges and the operation-level
FFT/BLAS policies; an 18-thread budget does not force every operation to use 18
threads. Large Cartesian FFT widths were explicitly set to 18/18.

The Xeon Gold 6240 results are less favorable: native wins every measured core
window, including the additional modal cases. The comparison explicitly included
GCC 16.2/OpenBLAS and Intel C++/MKL native builds. Neither native bundle wins
everything. On the large Cartesian cases, switching only Julia's FFT provider to
MKL reduced time by roughly 17–19%, with BLAS held fixed, but native remained ahead.
These are hardware/provider/configuration comparisons, not language-wide limits.

The final Mac field differences were at most 1.65e-15 relative L2 for modal/radial
cases and 8.69e-16 for the large Cartesian cases, with finite fields and matching
work counts. These matched-window comparisons supplement the NumPy-oracle checks;
they do not bound every adaptive propagation's global error. The 384² owned-buffer
ledger is 11.762 GB for Julia versus 7.567 GB for native, excluding runtime, opaque
FFT storage and export temporaries. Memory reduction remains worthwhile.

## Qualification and publication status

Development included focused NumPy-reference source/step/short-run comparisons,
rejection/reset and observation-ownership checks, tighter adaptive references,
interop checks and standalone Julia checks. The final combined plasma/absorber
slice passed 57 focused checks on Linux. The ledger records earlier stages and
failed or rejected experiments; these counts are not a full-suite certificate.

**The full Python/native gates, slow lanes and coverage qualification were not run
for this experiment. No tests or gates were run during draft-PR preparation.**
The new runnable example is the existing manual example factored into a script;
the final handoff commands have not been rerun in a fresh clone. Raw runtime caches,
large field witnesses and private campaign directories are not part of the PR.
Public evidence is in the ledger; the committed runners generate new receipts.

This draft intentionally has no CI qualification. Keep `[skip ci]` in every new
head commit while this policy applies: GitHub uses that marker, not draft status,
to [skip `push`/`pull_request` workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).
Required checks may remain pending. The
branch's `.readthedocs.yaml` also cancels at checkout with exit code 183, before
dependency installation or documentation generation, using
[Read the Docs' cancellation mechanism](https://docs.readthedocs.com/platform/stable/guides/build/skip-or-cancel-builds.html).
Remove that cancellation and
the skip policy when proposing a reviewed integration. Repository-wide CI and
other branches are unchanged.

## Next work

- [ ] **Independent review and schema port.** Agree an accepted SEP, review the
  protected arithmetic/normalization/ownership contracts, reconcile with current
  main and the SEP 34 medium-flux work, and simplify/format the compact Julia code
  for maintainability before any merge proposal.
- [ ] **Reproducible qualification.** Exercise the documented fresh-clone setup on
  macOS/Linux; expand NumPy-oracle, adaptive-convergence, state-lifetime and
  thread-count checks. Re-enable CI and run the full gates only when integration
  is proposed. Assess the shared benchmark-worker changes independently.
- [ ] **NVIDIA GPU implementation and tests.** Add CuArray allocation, device FFTs,
  contractions, reductions and synchronization; remove scalar host indexing and
  audit transfers across the Python boundary. Start with a small scalar Kerr
  capillary, then radial/Cartesian cases; check against NumPy and native before
  timing warm device windows, transfers and memory. KernelAbstractions and
  AcceleratedKernels dependencies alone do not make this a GPU backend.
- [ ] **AMD GPU implementation and tests.** Add ROCArray/AMDGPU FFT and contraction
  support, then repeat the same correctness and bounded performance qualification
  on actual hardware. Preserve the common scalar operations where practical.
- [ ] **Operation-sized CPU policies.** Profile radial high-thread regressions and
  parallel Raman, including the native Accelerate path. Measure FFTW planning,
  layout, task sizes and MKL/OpenBLAS/Accelerate separately. Retain baselines and
  stop tuning when repeatable gains disappear; do not enable umbrella fast-math.
- [ ] **Memory and startup.** Reduce large Cartesian resident volumes and copies;
  measure allocations, peak RSS and host-statistics/export costs as well as warm
  propagation. Investigate Julia precompilation/cache reuse and cold-start cost.
- [ ] **Capability extensions.** Prioritize Cartesian Raman/Adams, resolved/radial
  envelopes and envelope plasma, then mixtures/multiple responses, polarization,
  batching, noise, tapered/general geometries and interpolated output as needed.
  Add each capability with a small reference comparison and a ten-step A/B case.
- [ ] **Distribution.** Design runtime discovery/installation, platform support and
  packaging instead of relying on a source-tree-relative private Julia runtime.
  Keep explicit failure for unsupported features and preserve automatic selection.
- [ ] **Later precision work.** Establish Float32 error budgets before Metal or other
  single-precision GPU work. CPU Float64 measurements do not qualify that path.

## Code map

| Path | Role |
|---|---|
| `julia/src/SolpropJulia.jl` | Typed engine, modal/radial pipeline, transforms, Kerr, DOPRI and state lifecycle |
| `julia/src/responses.jl` | Translated plasma/Raman operations and population diagnostics |
| `julia/src/stages.jl`, `adams.jl` | Stage expansions and resident exponential Adams |
| `julia/src/free.jl`, `rk4.jl` | Cartesian/line transforms and prescribed RK4IP |
| `src/solprop/backends/julia/` | Private runtime, schema admission, shared-driver hooks, memory and observations |
| `julia/Project.toml`, `Manifest.toml`, `bootstrap.py` | Dependencies and checksummed private runtime setup |
| `julia/core_compare.py`, `core_profile.py` | Lightweight standard-case comparisons and profiling |
| `tests/backends/test_julia*.py`, `julia/test/` | Existing focused checks for future contributors to run |

No native numerical source was changed by the experiment. CPU tuning retained
measured improvements and rejected slower layout/fusion candidates rather than
shipping them as defaults.
