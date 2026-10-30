# Native performance push: working log

This page is the working log of a performance campaign on the native backend. It covers
the NVIDIA A40 (CUDA, binary64), an AMD Instinct MI325X (HIP) and an NVIDIA H200 (CUDA),
and the CPU on a dual Intel Xeon Gold 6240 and a dual
AMD EPYC 7763 at one and sixteen threads. It records what was tried and what each
attempt measured. Adopted changes also have their entries in [benchmarks](benchmarks.md)
and [CPU performance](cpu-performance.md), which hold the full measurements. This page
will be folded into those pages, or removed, before the work is merged.

The baseline is `main` at `740b24d`. Timings follow the
[benchmark rules](benchmarks.md#rules): short `--length` windows, fresh processes,
interleaved arms on the same machine, and medians. Benchmark parameters and stepper
settings are not changed. A ratio is candidate time over baseline time, so below one is
faster. Each ratio is measured against the commit before the change, so the rows of a
case compound.

The accuracy rule has two levels:

- An execution change on the same step sequence keeps each step within about $10^{-10}$
  of the baseline. Any change that moves results clearly beyond rounding is listed under
  [flagged](#flagged-accuracy-items).
- Over an adaptive run, the step counts may shift slightly. The fields then agree within
  what the stepper's relative tolerance implies.

## Whole-branch confirmation

This is `main` against the branch at `25b8ebb` on the Intel Xeon Gold 6240. It covers the tracking set at one and sixteen threads, with `--length 0.1` and three interleaved rounds. Every case took the same steps, and the final fields agree with `main` to within $10^{-13}$.

| Case | One thread | Sixteen threads |
|---|---|---|
| `gnlse_raman_shock` | 0.849 | 0.755 |
| `gnlse_soliton` | 0.845 | 0.887 |
| `rdw` | 0.686 | 0.337 |
| `he_rdw_averaged` | 0.694 | 0.568 |
| `argon_blueshift` | 0.584 | 0.375 |
| `argon_blueshift_averaged` | 0.609 | 0.710 |
| `argon_blueshift_envelope` | 0.864 | 0.887 |
| `h2_raman_averaged` | 0.648 | 0.379 |
| `ctc` | 0.524 | 0.699 |
| `radial_filament_256` | 0.595 | 0.581 |
| `grin_nonlinear` | 0.669 | 0.797 |
| `kerr_beam_384` | 0.581 | 0.663 |
| `kerr_beam_192` | 0.585 | 0.702 |
| geometric mean | 0.663 | 0.613 |

On a shared AMD EPYC 7763 node (64 cores), `main` against `733e5df` gives a geometric mean of 0.706 at one thread and 0.743 at sixteen. Every case took the same steps, with fields within $2 \times 10^{-13}$. A run on an exclusive node is still to come.

On the NVIDIA A40, `main` against `2d9c77e`, which includes the device graph region and the
device transform work:

| Case | Ratio | Field difference |
|---|---|---|
| `he_rdw_averaged` | 0.617 | $1.0 \times 10^{-11}$ (flagged) |
| `argon_blueshift_averaged` | 0.588 | $3.6 \times 10^{-15}$ |
| `h2_raman_averaged` | 0.767 | $4.8 \times 10^{-14}$ |
| `rdw` | 0.723 | $2.3 \times 10^{-15}$ |
| `radial_filament_256` | 0.804 | $3.1 \times 10^{-15}$ |
| `kerr_beam_384` | 0.777 | $1.0 \times 10^{-15}$ |

Every case took the same steps. Within each job the rounds ranged within one per cent. In the
second job, `main` ran at two speeds on `he_rdw_averaged`, `rdw` and
`radial_filament_256`, so each ratio here is the product of two jobs: `main` against
`01d051c`, then `01d051c` against `2d9c77e`. Where `main` was steady in both jobs, the
direct ratio agrees to within 0.002. The device suites pass, except one free-space
thread-count test that `main` also fails.

## NVIDIA A40

| Change | Measured |
|---|---|
| one state copy at a load and an observation; the free-space absorber split on the stored columns; the split factor once a sample | `kerr_beam_384`: 0.812 |
| the Kerr-only free-space stage through one rank-3 transform | `kerr_beam_384`: 0.963 |
| the free-space prefactor with one reciprocal | `kerr_beam_384`: 0.981 |
| one pinned upload per trial (SEP 39 §6.7) | from 2.1 to 1.1 host-to-device copies a trial; time unchanged on this card |
| radial analytic synthesis through the real product, then on the retained spectrum | `radial_filament_256`: 0.902 |
| a stage's forward rows in one batched transform; a trial's norms from pinned memory | `he_rdw_averaged` 0.784, `argon_blueshift_averaged` 0.755, `h2_raman_averaged` 0.878 |
| a four-mode stage's coarse rows in the batched transform; custom kernels below eight modes; CUDA Adams reductions in blocks of 128; a device kernel for real GEMMs on compute capability 8.6 | `rdw` 0.799, `radial_filament_256` 0.930, `he_rdw_averaged` 0.934 (flagged), `argon_blueshift_averaged` 0.940, `h2_raman_averaged` 0.952 |

The four-mode `rdw` trial was bound by its transform shapes, not by its launches. Each of
its four real transforms took about 96 µs on 16 of the A40's 84 multiprocessors, and cuBLAS
took 71 µs for its eight-row projection. With all of its forward rows in one transform and
its few-row products on the custom kernels, its source has no vendor GEMM left and replays
as one graph. On the MI325X the same tree measured `rdw` 0.962 and the other cases within
one per cent; the reduction blocks and the GEMM kernel apply to CUDA only.

The A40's radial trial is bound by its binary64 rate throughout. The Kerr case's remaining
kernel time is its three-dimensional transforms and the separable fold, both bound by the
same rate.

## AMD Instinct MI325X and NVIDIA H200

HIP builds now step with the native Adams march. On the MI325X the SEP 39 targeted tests
and the device suites all pass. The earlier HIP default, the Python march, failed the
device lanes on its own warning. SEP 39's report for phase 5 records the qualification.

| Change | MI325X | H200 |
|---|---|---|
| the native march instead of the Python march, same build | wall −1 % to −3 %; host µs a trial 15–40 → 0.5–3 | wall −3 % to −23 %; host µs a trial 41–70 → 0.8–5.2 |
| a kernel's functor as its launch argument; 256-thread HIP range reductions; φ-functions and history rows in registers; no fence after an accepted step; the held currents in one plan | `he_rdw_averaged` 0.744, `argon_blueshift_averaged` 0.686, `h2_raman_averaged` 0.740, `rdw` 0.875, `radial_filament_256` 0.901 | 0.920, 0.902, 0.918, 0.969, 0.989 |
| a trial's norms read from pinned host memory; a stage's forward rows in one batched transform; the three radial held streams in one plan | `he_rdw_averaged` 0.903, `argon_blueshift_averaged` 0.858, `h2_raman_averaged` 0.888, `radial_filament_256` 0.969 | not measured |
| `main` against the branch (before the row above) | `he_rdw_averaged` 0.463, `h2_raman_averaged` 0.624, `rdw` 0.605, `radial_filament_256` 0.843, `kerr_beam_384` 0.650 | 0.790, 0.788, 0.841, 0.868, 0.722 |

Both devices spent most of a trial waiting for the host to launch kernels of 2–4 µs. A graph region now replays an Adams trial's source evaluation as one CUDA or HIP graph while its operations repeat. Against the tree before it, with the same steps and bitwise-equal fields:

| Case | MI325X | H200 | A40 |
|---|---|---|---|
| `he_rdw_averaged` | 0.915 | 0.991 | 0.972 |
| `argon_blueshift_averaged` | 0.893 | 0.918 | 0.962 |
| `h2_raman_averaged` | 0.885 | 0.964 | 0.978 |
| `rdw` | 1.029 | 0.987 | 0.999 |

The `rdw` and `radial_filament_256` sources hold a vendor GEMM or a reduction, which stops their region. On the MI325X's short `rdw` window, the one capture costs 3 %.

## CPU

| Change | Measured |
|---|---|
| the Adams predict and correct passes as vector lanes | one thread (Xeon): `he_rdw_averaged` 0.786, `h2_raman_averaged` 0.697, `rdw` 0.827, `radial_filament_256` 0.839 |
| Raman modal products instead of node drives | `ctc`: 0.833 at one thread (EPYC), 0.884 at sixteen (Xeon) |
| radial synthesis on the retained spectrum | `radial_filament_256`: 0.866 at one thread (EPYC) |
| a stable kernel team | sixteen threads: `h2_raman_averaged` 0.526 (Xeon), 0.483 (EPYC) |
| NumPy's BLAS pool held at one thread beside a native run | sixteen threads, geometric mean: 0.886 (Xeon), 0.855 (EPYC); `rdw` 0.462 (Xeon) |
| opt-in persistent FFTW wisdom (`SOLPROP_FFTW_WISDOM`) | one-thread construction, warm against cold: from 1.4–6.6 s to 0.03–0.1 s |
| the host stage entry as lanes; the separable fold, crossings and monitor without divisions; radial statistics through two products | one thread (Xeon): `kerr_beam_384` 0.689, `kerr_beam_192` 0.682, `radial_filament_256` 0.808, `gnlse_raman_shock` 0.842 |
| a compact team for one-column problems, with FFTW's threaded loops on the same team; one-mode node products on the response team | sixteen threads (Xeon): `gnlse_raman_shock` 0.774, `argon_blueshift` 0.680, `h2_raman_averaged` 0.895, `ctc` 0.947; `argon_blueshift` 0.601 (EPYC) |
| the band column's rates, integrands and filter as vector blocks; the general route's real cubic through a half-spectrum inverse; wider projection blocks; synthesis that zeroes only outside the crop | one thread (Xeon): `ctc` 0.703, `argon_blueshift` 0.641, `grin_nonlinear` 0.701, `rdw` 0.865; sixteen threads (Xeon): `ctc` 0.846, `grin_nonlinear` 0.802; one thread (EPYC): `ctc` 0.754, `argon_blueshift` 0.720, `grin_nonlinear` 0.749 |

Every A/B above took the same steps. Final fields agree with the baseline to between
$10^{-15}$ and $10^{-14}$.

## Flagged accuracy items

- **Adams lanes.**
  - The strict parts of the Adams passes are unchanged bit for bit. The other predictor and
    corrector sums contract in a different order, so the first differing step norm moves by
    $1.7\times10^{-16}$.
  - On `he_rdw_averaged` both builds take the same 411 steps. The step sizes agree to
    $10^{-11}$ until about step 220, then the pulse dynamics amplify the difference to
    $5\times10^{-5}$ by step 360.
  - The final field differs by $4.6\times10^{-12}$. This is the adaptive controller's
    sensitivity, not an arithmetic error.

- **Device Adams passes in registers.**
  - The per-element arithmetic is unchanged, but the device compiler contracts the
    unrolled predict and correct passes differently.
  - On `he_rdw_averaged` both builds take the same 411 steps. The final field moves by
    $1.3\times10^{-11}$ on the MI325X and $1.2\times10^{-11}$ on the H200.
  - Every other case is within $1.8\times10^{-13}$.

- **CUDA Adams reductions in blocks of 128.**
  - A trial's norms are summed in a different order on CUDA, so they move by rounding and
    the step sizes follow.
  - On `he_rdw_averaged` both builds take the same 410 steps, and the final field moves by
    $1.0\times10^{-11}$ on the A40. HIP's blocks of 1024 had the same effect, at
    $1.4\times10^{-11}$.
  - Every other case is within $6.2\times10^{-14}$. The H200 has not been measured.

## Measured and not adopted

- **Compiler and build options:**
  - `-fno-math-errno -fno-trapping-math -fcx-fortran-rules`: no net gain. Results were
    bitwise identical on most cases. The gain `-fno-math-errno` gives on Cartesian Kerr comes
    from instruction-cache layout, not from different code.
  - `-O2`, switching IPO off, a 512-bit preferred vector width, SLEEF's AVX-512 functions on
    the Xeon, and branch-boundary padding.
- **Threading:**
  - serial or full-width medium operations;
  - a BLAS width per product;
  - spinning OpenMP waits;
  - binding threads by default.
- **Sixteen-thread kernels:** Raman transforms from the modal transform's start on the
  compact team, and one-column products as team kernels. Each gained on the Xeon and lost
  on the EPYC.
- **FFT planning:** measured planning of threaded FFT rows speeds up a sixteen-thread
  window by up to 13 % but adds 1.8–9.5 s of planning.
- **Kernels:**
  - Hand-written skinny products and one-mode loops both lost to threaded OpenBLAS.
  - The `cpunative` Cartesian split FFT schedule was not ported: native's transverse pair
    shows no pathology.
  - A rank-3 FFTW transform for the host's Kerr-only stage.
  - Adams lane blocks of 16 and 64.
  - A device whole-transform node chunk.
  - Band statistics through vector maths.

## Findings outside the native backend

- On the A40, solprop's host code takes about half of a Cartesian Kerr window, and on the
  CPU about 5 s of a 9 s sixteen-thread window. That is hashing and serialising the
  configuration and converting the 1.77 GB field at each saved plane.
- solprop's power weights call `numpy.tensordot` at every saved plane, which woke NumPy's
  BLAS pool. The native backend now holds that pool, but the call remains.
- At sixteen threads `rdw` is limited by solprop's per-run Python inside the timed window,
  such as the simulation metadata and the configuration hash. The native part scales 4.3
  times.
- Building the averaged helium case's gas transport model takes about 2.25 s per process.
