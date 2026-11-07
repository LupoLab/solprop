# Handoff: the Julia backend on a Mac (CPU comparison, then Metal)

For the agent picking this up on the maintainer's Mac (Apple M5 Max). The work is draft
PR #559, branch `feature/julia-backend-experiment` (do not merge; heads `[skip ci]`).
Read `AGENTS.md`, then [README.md](README.md) (setup, variables, files) and
[STATUS.md](STATUS.md) (what the engine is, what made it fast, the measurements, what is
left). The ledger rows are in `docs/dev/benchmarks.md`, "Julia backend experiment".
Native ideas found on the way are #560.

## Where it stands

`backend="julia"` steps solprop's translated problem behind the same `StepDriver` seam
as native. The engine is one Julia source (`julia/src`), run three ways:
- on a CPU, by Polyester tasks;
- on NVIDIA GPUs, through CUDA.jl (`ext/SolpropJuliaCUDAExt.jl`, environment `julia/cuda`);
- on AMD GPUs, through AMDGPU.jl (`ext/SolpropJuliaAMDGPUExt.jl`, environment `julia/rocm`).

It runs in binary64 or binary32 (SEP 33). The exponential Adams march itself runs in the
engine (`src/march.jl`). Its windows match native's steps, and match its fields to
rounding.

Julia's window time over native's, at commit `a6722029` (below 1, Julia is faster):

| Machine | Geometric mean | Range | Notes |
|---|---:|---|---|
| Intel Xeon Gold 6240, 16 threads | 1.13 | 0.78 to 1.26 | faster on `line`, `ctc_reduced`, `h2_raman_6ps` |
| Same, one thread | | 0.97 to 1.56 | |
| NVIDIA A40, binary64 | 1.97 | | |

The NVIDIA H200 (2.39) and the AMD Instinct MI325X (2.48) were measured before the fix
for compilation inside the first window, so their radial and free-space rows overstate
the gap.

## Step 1: a light CPU comparison with native on the M5 Max

PR #10's prototype was last measured on this Mac. The engine has since been rewritten,
so this is the first measurement of the current one there.

1. Set up, from a checkout of the branch:

   ```sh
   uv sync --extra julia                # native builds as AGENTS.md says
   juliaup add 1.13                     # or let the bootstrap download the release
   export SOLPROP_JULIA_EXE=$(julia +1.13 -e 'print(joinpath(Sys.BINDIR, "julia"))')
   uv run --no-sync python julia/bootstrap.py
   ```

   The first package build replays `src/precompile_trace.jl` (about a minute; see
   "Start latency" below).
2. Run the tests: `uv run --no-sync pytest tests/backends/test_julia.py -n 4`. They
   should all pass, CPU only; the GPU-marked ones are deselected.
3. Run the comparison. Use the performance cores only; macOS cannot pin threads, so
   find their count with `sysctl hw.perflevel0.physicalcpu`:

   ```sh
   P=$(sysctl -n hw.perflevel0.physicalcpu)
   uv run --no-sync python -m benchmarks ab --baseline-backend native --candidate-backend julia \
     --baseline-python .venv/bin/python --case rdw --case argon_blueshift --case h2_raman_6ps \
     --case ctc_reduced --case radial_filament_256 --case line --case kerr_beam_192 \
     --stepper default --noise none --threads 1,$P --length 0.02 --rounds 3 --repeats 1 \
     --out <a new directory>
   ```

   Record the commit, the machine and the rows in the ledger's Julia section.
4. Things to watch on Apple silicon:
   - **Efficiency cores.** Polyester's tasks are static: a task on an efficiency core
     holds back every pass, so more threads than performance cores may be slower.
   - **BLAS.** Julia uses OpenBLAS through libblastrampoline. Accelerate (AppleAccelerate.jl)
     may be faster for the GEMMs; the prototype had an Accelerate option.
   - **SIMD.** FFTW and `src/vmath.jl` (branch-free `exp`/`log`, plain Julia) vectorise
     with NEON's two binary64 lanes, not AVX-512's eight; the plasma's gains may be smaller.
   - **Compilation in the window.** `test_julia_first_window_compiles_nothing` holds the
     first timed window to under 50 ms of compilation; check it passes here.

## Step 2: a Metal extension

Metal.jl 1.10 has AbstractFFTs plans (on MPSGraph), `mul!` through MPS, and a
KernelAbstractions backend; AcceleratedKernels supports it. So the device layer, kernels,
reductions and scans should carry over, as they did for AMD tonight. That took about an
hour from writing the extension to all tests passing; see commit "an AMDGPU extension
and its environment" for the pattern:
- the extension file;
- the environment `julia/<gpu>/Project.toml`, resolved into its own `Manifest.toml`;
- the `[weakdeps]` and `[extensions]` entries in `julia/Project.toml`;
- `_GPU_PACKAGES` and the device check in `src/solprop/backends/julia/_runtime.py`;
- the source digest's file list in the same module;
- `bootstrap.py --<gpu>`;
- the benchmark worker keeping the named device under `--device`.

The one real problem is binary64. **Apple GPUs have none**, and the engine deliberately
keeps these in binary64 on the device, even in binary32 runs (SEP 33, 4.1 item 3):

| Where | What |
|---|---|
| `src/modal.jl` (`linear=upload(..., ComplexF64)`) | the linear operator |
| `src/dopri.jl` (`propagators!`) | the propagators' exponents, `exp(h L)` |
| `src/adams.jl` (`adams_trial_width!`: `x = ComplexF64(step * linear[f, m])`, `phi_values`) | the Adams phi functions |
| `src/stages.jl` (`delta_device`, `path.phi`) | a varying operator's stage increments |
| `src/free.jl` (the `op` tables, `free_element`, `free_propagators!`) | the free-space operator and its longitudinal root |

Two ways round it:

1. **Form those on the CPU in binary64 and hand the device binary32 results.**
   - Unified memory makes the handoff cheap, and Polyester threads on the host can form
     them.
   - The phi functions depend on `h L` per element and are formed every trial: about
     100 ns an element, so rdw's 10 k elements cost about 1 ms on one core.
   - DOPRI's propagators change only with `dz`.
   - It suits the capillary routes (rdw, argon, ctc, h2). Cartesian recomputes its root
     per element and stage, which is the costly case.
2. **Double-float arithmetic** (a binary64 value as a pair of binary32s) in those
   kernels, as the mlx backend's Adams flow does (#543). It is general but more work.

Suggested order:
1. Admit `precision="single"` only on Metal; refuse binary64 by name.
2. Get rdw (modal, Adams) running with option 1, then argon, then radial.
3. Hold off on free space.
4. Use `--fft-size pow2`: MLX's GPU FFTs fell back to Bluestein above 4096 points for
   other sizes (#543). Whether MPSGraph does the same is unchecked.

Then compare against the mlx backend on the same windows: Julia's single source against
hand-written MLX.

Things to check early on Metal:
- **In-place and batched plans.** Whether Metal.jl's FFT plans support in-place
  `plan_bfft!`/`plan_fft!` and transforms along dimension 1 of a matrix. The engine uses
  both; `src/observe.jl` and `src/responses.jl` have the in-place ones.
- **Real views.** `reinterpret` of an `MtlArray` (`real_rows` in `src/launch.jl`, for
  complex-by-real products).
- **Atomics.** Binary32 atomics in local memory (`bucket_kernel!` in `src/observe.jl`).
- **Scalar indexing.** `Metal.allowscalar(false)`, as the other extensions set.

## What this branch learned (applies on any device)

- **Kernel bodies capture values, not types.** A closure that captures `Type{T}` or a
  `Core.Box` is refused on a GPU. `SOLPROP_JULIA_CHECK=1`, which the tests set, refuses
  it on a CPU too. Capture `zero(T)` and similar as values.
- **Nothing may compile inside a timed window.** The benchmark worker's warm-up runs
  without a boundary monitor, so construction must warm every bridge entry with the
  argument types a run passes. A read-only NumPy array is a different bridge type from a
  writable one. Check with juliacall's `PYTHON_JULIACALL_TRACE_COMPILE=<file>` (with
  `PYTHON_JULIACALL_TRACE_COMPILE_TIMING=yes`) around a timed run.
- **Start latency.**
  - The first engine in a process compiles its configuration's specialisations: 34 to
    64 s, cut to 11 to 14 s by replaying `src/precompile_trace.jl` when the package
    builds.
  - Regenerate the trace after large changes: run each case with the trace variable set,
    collect the `precompile(` lines, `sort -u`.
  - On a new device, add its specialisations the same way.
  - A PackageCompiler system image would remove the rest; it has not been tried.
- **GPU windows are host-bound** on GPUs with full-rate arithmetic. Count launches and
  allocations per trial before tuning kernels; the scratch cache in `c.scratch`
  (`scratch` in `src/launch.jl`) exists for this.
- **Benchmark hygiene.**
  - Nothing else on the machine while an A/B runs: tonight's own test runs contaminated
    one.
  - Measured FFT plans differ between processes; for bit-for-bit comparisons set
    `SOLPROP_FFTW_PLAN=estimate`.
