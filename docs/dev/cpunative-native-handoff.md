# CPU optimisation handoff for native

Use the [final comparison](cpunative-comparison.md) for wins, losses and scaling,
and the [experiment report](cpunative-experiment.md) for qualification and trial details.
The opportunities below come from measured cpunative changes;
they are hypotheses to test in native, not predicted native speedups. Their gains
overlap and must not be multiplied. Keep the standard recipes unchanged and include
developed fields: launch-only and short-window rankings can differ from full runs.
The frozen native source baseline is `cc6d8e2b0092757de0cf9357c40e475961da54f3`;
compare against that baseline before attributing gains to a later native change.

## Highest-value experiments

1. **Move radial contractions to retained spectra.** For square real static maps,
   synthesis/projection commute with temporal transforms. Synthesize before padding
   and project after cropping, so matrix products cover retained frequencies instead
   of the oversampled time grid. Keep the FFT count, analytic signal, normalisation,
   and separate polarisation/current streams unchanged. In cpunative, developed
   radial source time fell from 31.51 to 20.51 ms at one thread and 9.02 to 5.27 ms
   at 36 cores. Three repeated short-driver pairs reduced Adams/DOPRI5 time by
   14%/25% at one thread and 12%/20% at 36 cores.
   Start with `State::spectral_source` in `cpunative/src/core.cpp`; native's
   `GeneralWorkspace::synthesise` and `project` are in `native/src/general.cpp`.
   Do not move skinny capillary contractions this way: that would multiply FFT work.

2. **Convolve Raman modal products once, then reconstruct nodes.** For scalar real
   static maps, expand the squared node field into unordered modal products, using
   a factor of two for cross terms. The causal convolution is linear, so convolve
   each product once and reconstruct each node's result. With three modes this uses
   six products rather than 33 node drives. The implementation selects by
   `M*(M+1)/2`, node count, and scratch capacity; it is not a three-mode special case.
   Repeated CTC short runs improved by 12%/17% for Adams/DOPRI5 at one thread and
   7%/11% at eight cores; extra gain at 36 cores was small or negligible. Source
   checks cover all 1–8 modes and both paths; full CTC trajectories pass the unchanged
   independent envelopes. See `prepare_raman_pairs` and `response` in
   `cpunative/src/core.cpp`, and the Raman drive/convolution path in
   `native/src/general.cpp`. Preserve the full support, stage-dependent kernel,
   taper factors and node fallback. Vector/envelope extensions need their own algebra.

3. **Vectorise rate interpolation selectively.** Isolate the interpolation's
   `log`/`exp` calls and use high-accuracy vector math only on dense blocks. The
   prototype uses 256 samples, a zero-block exit, and scalar early exits below
   half-active occupancy. Unconditionally vectorising sparse tails nearly doubled
   RDW source time. SLEEF `u10` gave a portable improvement: a first matched-driver
   comparison reduced CTC time by 15%/19% at one thread and 5%/7% at 36 cores, with
   12%/13% for serial argon and little effect on RDW. See `cpunative/src/rates.cpp`
   and native's `ionisation_rate` in `native/src/general.cpp`. This changes the
   math-library accuracy contract: keep it an explicit, separately qualified variant.
   Preserve stable fractions, band integrals, defects, subnormals and finite checks;
   do not enable global fast-math. The experiment also tried SVML and MKL HA.

4. **Remove serial initialisation and copying from large observations.** Allocate
   owning, uninitialised trivially copyable storage, then copy in contiguous parallel
   partitions. Do not expose a live workspace view. A one-step standard-size Cartesian
   ablation at 36 cores fell from 8.45 to 6.78 s; serial time was unchanged. See
   `array(std::span<...>)` in `cpunative/src/bindings.cpp`. Native's
   `GeneralWorkspace::field_to_host` currently constructs a value-initialised vector
   and fills it with serial loops; also inspect the subsequent nanobind conversion
   for another copy. Keep GPU transfer/fence semantics separate and test retained
   observations after further propagation and closure.

5. **Control the large Cartesian FFT schedule.** Compare internally threaded volume
   plans with OpenMP-distributed serial `(x,t)` plane plans followed by batched
   transforms along `y`. The retained prototype uses blocks of 32 adjacent columns,
   full/tail plans, and reverses the order for the forward transform. At 32 EPYC
   cores, this reduced a complete cpunative run by 14%. At 96 cores it avoided a
   severe direct-FFTW planning regression: the isolated transform pair changed from
   about 27 s to 0.095 s. **Native did not exhibit that 27 s pathology**, so that
   ratio is not an expected native gain. See `cpunative/src/beam.cpp` and
   `cpunative/experiments/fft_volume.cpp`. Preserve new-array alignment, strides,
   disjoint ownership, planner locking and explicit normalisation. No extra volume
   buffer is needed. Use the direct path when too few planes can occupy the workers.

6. **Specialise only demonstrably skinny products.** A repeated hand-versus-OpenBLAS
   trial supports handwritten one-mode contractions at eight or more cores on the
   tested Linux x86-64 systems. Argon short runs improved by 4%/8% at eight cores and
   8%/19% at 36 for Adams/DOPRI5. Multimode results were mixed; radial strongly
   preferred BLAS. See `State::contract` and provider selection in
   `cpunative/src/core.cpp`. Accelerate won the tested Apple-silicon skinny products;
   Eigen did not justify an extension dependency. Retain dimension/architecture
   dispatch, not benchmark-name dispatch.

## Scheduling and measurement checks

- Audit OpenMP pool resizing between small row FFT batches and large flat loops.
  Keeping the cpunative team width stable reduced a developed RDW source call from
  roughly 6.2 to 2.1 ms at 36 Xeon cores. Native already has execution-pool policies;
  measure their transitions before changing them. Keep a genuinely serial path.
- Use one compatible OpenMP runtime across the extension, FFTW and OpenMP BLAS.
  Record effective widths and affinity before/after runtime initialisation. Lazy
  LLVM binding and pthread workers inheriting a bound main thread both invalidated
  early trials. Do not accept timings merely because `OMP_NUM_THREADS` is correct.
- Begin with `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, passive waiting and physical
  cores; compare alternatives on the same exclusive allocation. Existing native
  policy notes in `cpu-performance.md` still apply. More cores can make capillary
  runs slower even when they improve relative speedup against native.
- Audit cached temporal absorber factors and the real-dispersion prefactor branch
  in `cpunative/src/beam.cpp`. These avoid repeated exponentials and general complex
  square roots. Their individual full-run gains were not isolated, so profile first.

## Lower-priority directions

The controlled GCC 16/Clang 23/Intel comparison found no broad compiler winner.
The matched backend comparisons already give native the proposed GCC math flags.
OpenMP SIMD scans gave small mixed source changes; parallel scans inside individual
columns were slower. MKL FFT helped one Intel control but did not beat the selected
practical GCC/FFTW configuration. Direct FFTW wrapper removal alone had already
shown no measurable benefit in native's CPU record. A Cartesian statistics-fusion
prototype was discarded because the standard case requests no statistics and did
not exercise it. Prioritise measured work reduction and memory passes before these
directions.

## Reusable comparison tools

`cpunative.experiments.runtime` interleaves fresh baseline/candidate processes,
records effective thread widths, build provenance, work counts and quiet-launch
checks, and accepts separate Python environments and backend names. Use
`--fraction 0.0025 --rounds 3` for strategy trials; the approved experiment used
`--fraction 1` for final confirmation. `developed` and `observations` measure common
stored fields, while `accuracy` checks envelopes frozen from independent converged
references. The focused checks are in `cpunative/tests/`; they complement rather
than replace native's own tests. Keep each port as a separate measured ablation.
