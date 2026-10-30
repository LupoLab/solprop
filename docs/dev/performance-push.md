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

## NVIDIA A40

| Change | Measured |
|---|---|
| one state copy at a load and an observation; the free-space absorber split on the stored columns; the split factor once a sample | `kerr_beam_384`: 0.812 |
| the Kerr-only free-space stage through one rank-3 transform | `kerr_beam_384`: 0.963 |
| the free-space prefactor with one reciprocal | `kerr_beam_384`: 0.981 |
| one pinned upload per trial (SEP 39 §6.7) | from 2.1 to 1.1 host-to-device copies a trial; time unchanged on this card |
| radial analytic synthesis through the real product, then on the retained spectrum | `radial_filament_256`: 0.902 |

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
| `main` against the branch | `he_rdw_averaged` 0.463, `h2_raman_averaged` 0.624, `rdw` 0.605, `radial_filament_256` 0.843, `kerr_beam_384` 0.650 | 0.790, 0.788, 0.841, 0.868, 0.722 |

Both devices now spend most of a trial waiting for the host to launch kernels of 2–4 µs.
Their kernels keep the GPU busy only 35–50 % of the time. Capturing a trial's launches as
a graph is the largest remaining lever.

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
- Building the averaged helium case's gas transport model takes about 2.25 s per process.
