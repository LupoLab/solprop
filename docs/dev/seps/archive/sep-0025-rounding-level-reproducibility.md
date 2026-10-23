# SEP 25 — Rounding-level reproducibility and the performance it frees

| | |
|---|---|
| **SEP** | 25 |
| **Title** | Rounding-level reproducibility and the performance it frees |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-17); GPT-6 (review revisions, 2026-10-17) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-17 |
| **Resolution** | Accepted 2026-10-17 by the maintainer, after the human and AI reviews and the second pass |
| **Requires** | [SEP 16](sep-0016-unified-multistep.md) (the default multistep stepper, its work–precision qualification and the host's work per trial); [SEP 18](sep-0018-band-exact-plasma-integrals.md) (the band rule whose traversal this SEP replaces) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers; Phase 0 on `feature/sep-0025-baseline` |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

solprop stops requiring bitwise agreement between independent floating-point
computations. That covers every backend against `numpy_ref`, and one backend run
twice, on one or many threads, as a batch member or alone, or as a scan point.
Exact structural contracts, including lossless storage, hard masks and integer
accounting, remain. Numerical comparisons use one of two criteria:

- **A rounding bound** for two computations with the same discrete method and history.
  It accounts for operation counts, reduction lengths, transform errors, conditioning
  and amplification along the tested propagation window.
- **An accuracy envelope** for two adaptive runs whose discrete trajectories may
  differ. It is set per case against a reference run at a tighter tolerance, or an
  existing stored converged reference. A local tolerance alone is not a bound on
  global error, so the envelope comes from that reference, not from `rtol`.

Numerical care stays mandatory: conditioning, cancellation, overflow and the IEEE
semantics that detect divergence. What goes is the rule that a kernel must perform
the same operations in the same order as another implementation.

The policy is recorded in both repositories' `AGENTS.md` and the design pages, and the
test suite is re-engineered to it. Then the optimisations the old rule blocked are
taken, together with the host-side overheads found in the same measurement:

- a vectorised band plasma rule, and parallel scans for the plasma column;
- fused multiply-add contraction and free reduction order;
- fused Kokkos stage kernels;
- GPU validation and optimisation on NVIDIA A40, followed by separate optimisation
  passes on NVIDIA H200 and AMD MI325X;
- a contraction-order-free general-modal projection;
- batched host evaluations at known stage positions;
- cached immutable execution facts, with the dynamic records kept current;
- cheaper output appends, and in-memory stores sized to the run's planes.

Success is judged in two ways. Every qualification result stays inside its declared
bound or accuracy envelope, and accuracy and work at matched error do not regress
under the predeclared margins of §7.3. This uses the work–precision methodology of
[SEP 16](sep-0016-unified-multistep.md).
A benchmark set on short windows also gets faster. That set includes the mode-averaged
helium RDW case motivated by the reported timing breakdown in §2.

### 2. Motivation and scope

**The policy and its reason.** solprop's answers are only as good as their discretisation
error. A capillary run is asked for a relative tolerance of 10⁻⁶ to 10⁻¹⁰ per step, and
its converged answer is compared against measurement at the per-cent level. The unit
roundoff of double precision is $u = 2^{-53} \approx 1.1\times10^{-16}$. The
difference between two summation orders is bounded in terms of $u$, the reduction
length and the sum of the magnitudes (§4.1). Its importance depends on cancellation
and subsequent amplification; it is not universally negligible. Requiring the last
bit to agree does not establish accuracy. The proposed replacement is a numerical
error budget that permits faster implementations while retaining accuracy tests.

The maintainer has never needed a bitwise-identical repeat, on one machine or across
machines, in twenty years of writing and using propagation codes. Debugging can use a
recorded step schedule (`stepper.md`; [ADR-013](../../decisions.md)). Replay is required
to meet the overall accuracy target; it need not reproduce either the bits or every
adaptive decision (§4.4).

The tests check both **implementation agreement within a rounding budget** and
**accuracy against independent references**. Agreement alone cannot establish
correctness, and the qualification envelopes are not universal global-error
guarantees for arbitrary inputs.

**What the rule costs today.** The bitwise rule is narrower than its reputation, because
backend agreement with `numpy_ref` has been a rounding bound for a long time
(`backends.md` §"Agreement is 1e-15, not bitwise"; `accuracy.md` §7). What remains has
three costs.

*It constrains several candidate kernel optimisations.*

- **The band plasma rule.**
  - [SEP 18](sep-0018-band-exact-plasma-integrals.md) specifies a lagged
    ring-buffer traversal so that the trapezoid's bitwise digests cannot move (its D7).
    Its §6 forbids fusing multiply-adds or reassociating the reference sum to save
    operations.
  - The Kokkos kernel writes every product and sum as its own statement for the same
    reason (`solprop-kokkos/src/general.cpp`, `band_window`). The vectorisable
    multipass column was never built.
  - The rule costs 1.5–2.5× the trapezoid's window on `numba_cpu`, by the
    [roadmap's](../../roadmap.md) record.
- **The single-node plasma column is serial.**
  - On a CPU build the response launches are parallel only over nodes. A mode-averaged
    run has one node, so its whole response column (every sample's rate, exponential,
    filters and three running sums) runs on one thread.
  - The device build already evaluates the same integrals as team parallel scans and
    accepts their different association. The host column keeps the serial order
    because that is the order that reproduces the reference's bits.
- **Order-fixed reductions.** These are the error norm (per-column partials summed
  serially), `accept_and_score` (a single work item on Kokkos), the general-modal
  projection (one accumulator per bin "in the reference's order"), and the native GEMM
  (held to `KokkosBlas::gemm` bit for bit).
- **Small FFTs on one thread.** Kokkos plans every transform below 8 192 real samples on
  one FFTW thread, partly because FFTW may choose the factorisation by the thread count
  (`backends.md`, "Small transforms on one FFTW thread"). The 8 192-sample measurement
  is the same either way; the invariance argument was half of the reason.
- **No contraction.** Neither backend may use fused multiply-add on purpose, and
  `numba_cpu` may use no `fastmath` flag (`AGENTS.md`; `backends.md` §3).

*It shapes the test suite and its infrastructure.*

- **Exact assertions.** The suite asserts exact equality on about 1 200 lines in 230
  files (`np.array_equal`, `assert_array_equal`) and refers to bitwise results on about
  170 lines in 96.
- **Pinned digests.** The digest records are pinned to one FFTW build. Seven failed on a
  wheel of FFTW 3.3.10 against the 3.3.11 they were made with, and some were replaced by
  recorded states at 10⁻¹⁴ of the peak once a fresh environment moved them by 10⁻²⁰ to
  10⁻¹⁷ (`testing.md` §5).
- **Pins hold on one platform only.** Every bitwise record holds only on the platform
  that made it (macOS on arm64, `tests/platform_pins.py`) and is **skipped by name
  elsewhere**, so a Linux run of the suite checks less than a Mac run.
- **Infrastructure kept for the rule:**
  - an FFTW wisdom-isolation fixture (`tests/fftw_witness.py`);
  - a Homebrew-built pyFFTW for the digests;
  - a `REFERENCE` Kokkos build profile that the Kokkos pins hold for alone (ADR-034).

*It distorts design documents.*

- Open SEPs set bit identity as their success criterion. For example,
  [SEP 22](sep-0022-memory-lean-spatial-execution.md) judges its runner and its
  statistics-only planes "by bitwise identities".
- [ADR-013](../../decisions.md) makes "a batch of N members is bit-identical to N sequential
  runs" a test.
- [ADR-022](../../decisions.md) requires compiled kernels to be "tested … for bitwise
  repeatability and for thread-count invariance".

**A measured motivating case.** The maintainer's timing study supplied this breakdown
for a mode-averaged helium RDW case. Its scripts and raw timings are not in this
repository, so the figures are motivation, not a baseline. Phase 0 re-measures on
short windows.

- **Input:** a 125 µm capillary 1.5 m long, graded 0.8 → 0 bar, driven by a 7.5 fs,
  275 µJ pulse at 800 nm.
- **Grid:** 8 192 fine real samples (`n_to`), 2 049 coarse frequencies
  (`n_omega`, hence `n_t = 4 096`); Phase 0 verifies the actual grid record.
- **Run:** 201 saved planes, `RungeKutta()`, measured on one pinned core of a Xeon Gold
  6240.
- **Total:** 4.05 s, best of three, with this split:

| part | time (s) | note |
|---|---|---|
| translation (dispersion tables along the gradient) | 0.46 | once per run |
| the C++ trial steps, band plasma rule | 2.15 | 606 steps; 1.38 s with the trapezoid, 1.05 s without plasma |
| host work in the step loop | 0.44 | stage scalars, controller, schedule, records |
| HDF5 output of 201 planes | 1.00 | about 5 ms a plane |

Four observations follow from the table:

- **The band rule is 0.77 s, 36 % of the kernel time,** over a trapezoid that costs
  0.33 s.
- **Eight threads make the run slower,** at 4.34 s. The FFTs are below the size that
  repays threads, and the only other heavy kernel is the serial column.
- **Host work: per trial, plus a per-plane record.** The 0.44 s in the table covers
  both. A profile of the plasma-free run puts the per-trial part at about 0.25 s over
  565 trials, 0.45 ms a trial. Most of that is the eight stage-scalar evaluations,
  each about six Chebyshev tables evaluated one at a time by scalar Clenshaw sums.
  The profile also shows about 2 ms a plane spent rebuilding and re-serialising the
  execution record. Profiled times overstate Python costs, so Phase 0 times these
  parts without a profiler.
- **The output writer resizes every dataset by one row at every plane,** about
  15 datasets with chunks of one plane each.

None of this matters for a run whose step costs 50 ms (the four-mode resolved case
does), but all of it matters for mode-averaged runs, scans of short runs and batches.

**In scope.**

1. The policy: definitions (§4), the wording in both `AGENTS.md` files, the design pages
   and a new ADR that supersedes the bitwise clauses of ADR-013, ADR-022, ADR-034 and
   ADR-039.
2. Classification and migration of arithmetic-equality tests, retaining exact
   structural contracts; removal of numerical digest/platform-pin infrastructure;
   and reusable error-budget helpers (§6.2).
3. Kernel changes on `numba_cpu` and Kokkos CPU that the old rule blocked (§6.6–§6.7):
   - the vectorised multipass band rule;
   - parallel scans for single-node plasma columns;
   - contraction;
   - free-order and parallel reductions;
   - the general-modal projection as a contraction of free order;
   - FFT threading set by speed alone.
4. Kokkos stage-kernel fusion (§6.7), subject to source-accumulation and stage
   dependency checks and a rounding budget for any changed evaluation order.
5. Host-side overheads (§6.3–§6.5):
   - batching known stage-scalar positions while retaining anchor reuse;
   - immutable execution facts cached per segment, dynamic metadata kept current;
   - HDF5 datasets preallocated with a saved-plane count, and planned-size in-memory
     stores;
   - batched dispersion sampling where the provider already supports it.
6. The `numpy_ref` simplifications that the old rule imposed (§6.8).
7. GPU performance work (§6.9): an initial CUDA validation and optimisation pass on
   NVIDIA A40 on a university cluster, followed by an NVIDIA H200 CUDA pass and an AMD MI325X HIP
   pass. Each includes profiling, measured optimisation experiments and numerical
   requalification; a successful build and correctness run alone do not complete it.

**Not in scope.**

- **Lower precision.** Everything stays in double precision; single or mixed precision
  would be its own SEP.
- **Any change to physics or physics defaults.** That includes the band rule's design, its taps
  and its certification: the taps stay, and only the traversal changes.
- **New steppers or error controllers.** The controller is unchanged; what changes is
  that a rounding difference may now flip one accept/reject decision.
- **Statistics-only planes and the construct-once runner.** Those stay with
  [SEP 22](sep-0022-memory-lean-spatial-execution.md). This SEP's output preallocation
  is the part independent of it.
- **Autotuning of thread counts.** That is [SEP 4](../sep-0004-execution-autotuning.md); this
  SEP only re-measures the fixed thresholds that the old rule shaped.
- **New GPU routes or an execution-autotuning interface.** The GPU passes optimise
  already supported computations and use measured static policies. New physics,
  runtime search and the device-memory admission redesign of SEP 22 remain separate.

### 3. Context

**Where the rule is written today.** The rules this SEP replaces, by kind:

| kind | where | today |
|---|---|---|
| backend against `numpy_ref` | `backends.md` ("Agreement is 1e-15, not bitwise"); `accuracy.md` §7 | already a rounding bound, except the named kernels below |
| kernels held to the reference's bits | `backends.md` (plasma and Raman `evaluate`, `raman_spectrum`, statistics on one state; the plasma running sums against `plasma_scan`/`plasma_current`); `kernel-contract.md` | bitwise |
| repeat runs, thread counts | `AGENTS.md` §backends ("Reductions are deterministic"); `backends.md` §3; ADR-022; `accuracy.md` §7 ("two runs, one thread against many: bitwise"); `solprop-kokkos/AGENTS.md` | bitwise, tested |
| batches and scans | ADR-013; `backends.md`; `accuracy.md` §7 ("both bitwise, on every backend") | bitwise, tested |
| fast path against general path | `testing.md` §3 | bitwise |
| fixed schedules | `stepper.md` ("bit-identical reruns") | bitwise |
| resume | ADR-039, ADR-046; [SEP 14](sep-0014-stepper-efficiency.md) | already rounding-level on the compiled backends; `numpy_ref` still bitwise |
| CPU against GPU | `backends.md`; `accuracy.md` §7 (`Bfft`, `Bscan`) | already rounding-level |
| build flags | `solprop-kokkos/CMakeLists.txt` (`-fno-fast-math`, `-fp-model=precise`); `numba_cpu/kernels.py` (no `fastmath`) | value-safe and order-preserving |
| digests and pins | `testing.md` §5; `tests/platform_pins.py`; `tests/kokkos_build.py`; `tests/fftw_witness.py` | macOS-arm64 only, one FFTW build |

**What already relaxed it, and why that worked.**

- [SEP 14](sep-0014-stepper-efficiency.md) dropped bitwise replay resume: keeping
  it cost about one evaluation in six (ADR-039).
- The device builds were qualified to rounding bounds fixed before any device result
  (`accuracy.md` §7).
- `testing.md` §5 replaced byte digests of the legacy entry points by recorded states at
  10⁻¹⁴ of the peak.

None of these moved a validated result. The rounding-bound ladder of `accuracy.md` §7
already carries the backend-equivalence evidence, and it is the template for the rest.

**The priorities.** [ADR-020](../../decisions.md) puts accuracy and reliability first,
then CPU performance. Its rule is that "no optimisation is accepted that moves a
validated result beyond its stated tolerance". This SEP keeps that rule word for word;
bitwise identity was never part of it.

**Prior work on the host and the band rule.**

- [SEP 16](sep-0016-unified-multistep.md) cut the host's work per trial from
  0.50 to 0.32 ms by caching table values. It left the remaining five distinct
  evaluations (0.12 ms) because moving them would "need numba in the translation layer".
- [SEP 18](sep-0018-band-exact-plasma-integrals.md)'s Q14 and Q15 left the
  multipass column to the roadmap ("The band rule's cost on the compiled backends"),
  with the condition that "either option must keep the reference's arithmetic". This
  SEP removes that condition and covers the item in full.

### 4. Theory


#### 4.1 What a reordering can change

For finite binary64 operands in round-to-nearest arithmetic, excluding overflow and
underflow, the standard model [1, 2, 9] is

$$\mathrm{fl}(a \mathbin{\mathrm{op}} b) = (a \mathbin{\mathrm{op}} b)(1+\delta),
\qquad |\delta| \le u = 2^{-53}.$$

Here $\mathrm{op}$ is a basic arithmetic operation with a defined finite result;
transcendental functions need their own error bounds. Gradual underflow requires an
additional absolute-error term. Flush-to-zero is not covered by this model and is
outside this SEP. Use $\epsilon = 2u = \mathtt{np.finfo(float).eps}$ when quoting
existing solprop bounds: $u$ and $\epsilon$ must not be interchanged.

For a sum of $n$ fixed real inputs, with each input added once, any addition tree
satisfies [1, 10]

$$\left|\widehat S-\sum_i x_i\right|
\le \gamma_{n-1}\sum_i|x_i|,\qquad
\gamma_k=\frac{ku}{1-ku},\quad ku<1.$$

A known maximum tree depth $d$ replaces $n-1$ by $d$. The difference of two such
computations is bounded by the sum of their two bounds. The relative condition
number is $\kappa=\sum_i|x_i|/|\sum_i x_i|$, undefined at a zero sum. Cancellation is
not evidence that the physical formulation is wrong; it is a reason to use an
absolute bound or a more accurate accumulation. An arbitrary reordering need not
improve the actual error even when it improves the worst-case bound.

Products introduce additional rounding: a real length-$n$ dot product has the
usual $\gamma_n\sum_i|a_i b_i|$ bound, and complex products require accounting for
their real operations. Fused multiply-add rounds $ab+c$ once, but permitting it does
not establish the accuracy of a whole kernel.

A stable FFT with sufficiently accurate twiddle factors has a normwise error of
order $u\log n$ [3]; the constant depends on the implementation and normalisation.
The existing device envelope in `accuracy.md` §7 is
$B_{\mathrm{fft}}(n)=512\epsilon\lceil\log_2 n\rceil$ for a round trip plus its
pointwise work. It was fixed before device qualification, not fitted to the new
implementation. It is not a universal theorem about every FFT plan. FFTW's plan
selection [4] is one reason to qualify multiple supported transform paths.

#### 4.2 What must still be protected

Permission to change evaluation order does not remove the numerical contract:

1. **Stable increments.** Form the represented phase increment from integrated
   expansion weights, not by subtracting two large accumulated phases. Preserve
   short-step and large-anchor tests, including the constant-operator path.
2. **Small responses.** Preserve `expm1` and stable small-argument φ-functions.
   The existing fraction calculation is $1-(1-p)e^{-X}$; an absolute error of order
   $u$ can dominate a small ionised fraction. Its acceptability must be assessed
   after density, current and propagation scaling, not asserted from the fraction
   alone. The equivalent form $p-(1-p)\operatorname{expm1}(-X)$ is a useful
   independent small-exposure witness. This SEP does not silently change the
   population model, exposure floor or loss-current prescription.
3. **Small defects and compensated sums.** Preserve embedded difference coefficients
   and the multistep defect construction. Reassociation must not erase a compensation
   term or reconstruct a small defect by subtracting large completed solutions.
4. **Finite-value detection and range.** No `nnan`, `ninf`, `nsz`, umbrella fast-math,
   flush-to-zero or approximate-function/reciprocal flags. Keep finite-value guards
   under strict arithmetic. Contraction and reassociation can still change intermediate
   overflow and underflow; their absence of `nnan`/`ninf` is not a proof of safety.
   Qualify the supported dynamic range and exercise non-finite failure paths.
5. **Signed accumulations.** Bound every prefix by the magnitudes entering it,
   including errors already present in its inputs. Only a sum of nonnegative inputs
   has condition number one. The plasma currents and the band-filtered exposure do
   not meet that assumption (§4.6). Use compensation or another justified
   accumulation if the absolute bound exceeds the response's error budget.

These checks apply to each enabled compiler flag and to hand-written reordering.
The reference remains independently readable; matching a changed shared body is
not sufficient evidence that its mathematics is still correct.

#### 4.3 Matched discrete computations: the rounding bound

The same step sizes alone do not define the same computation. A rounding comparison
also fixes the method, orders, startup and restart rules, history, boundary handling,
input/noise arrays and observation policy. For a multistep method the state in the
following derivation includes its history, not only the current field.

Let $F_j$ be that common exact-arithmetic discrete update, with local Lipschitz
factor $G_j$, and let $\eta_j$ bound the difference introduced by the two rounded
implementations on the same input. Then

$$\|e_j\|\le G_j\|e_{j-1}\|+\eta_j,$$

and hence

$$\|e_N\|\le\left(\prod_{j=1}^N G_j\right)\|e_0\|
+\sum_{k=1}^N\eta_k\prod_{j=k+1}^N G_j.$$

If $G_j\le 1+h_jL$ with $L\ge0$ in m⁻¹, and $\eta_j\le cuY$ on the tested state range,
this gives $e^{LZ}(\|e_0\|+NcuY)$, with $Z=\sum_j h_j$ [5]. This is a guide to the
bound's shape, not a proof for every case: nonlinear sensitivity and library errors
enter $\eta_j$, and $L$ is rarely known exactly.

In practice the constant is set before the comparison is run, as the `accuracy.md`
§7 ladder sets it ($16\epsilon$ for one step, $128\epsilon$ for eight): from the
operation and reduction counts with a stated safety margin (`testing.md` §3), not
fitted to the discrepancy observed. A case whose dynamics amplify strongly is
compared over a short window.

Instabilities can amplify numerical perturbations [6], with a gain that depends on
the perturbation's spectrum and direction. Rounding is not a substitute quantum
noise model. Use short windows, a justified gain budget, or separately validated
ensemble observables where a long-distance pointwise comparison is unsuitable.

#### 4.4 Adaptive runs and replay: the accuracy envelope

A rounding change can alter acceptance, proposed step size, order or restart
choices. Even without an accept/reject fork, a controller can propose slightly
different steps. These computations are compared by accuracy, not by the rounding
bound for a common discrete method.

The triangle inequality gives

$$\|y_a-y_b\|\le E_a+E_b$$

only if $E_a$ and $E_b$ bound the two errors against the same exact target, in the
same norm, at the same observation position. `rtol` and `atol` control a local
estimate; neither implies such a global bound. The target must also be explicit:
the exact solution of the spatial/time-discretised propagation problem is not the
continuum physical solution.

For a test, the envelope is set per case and comment-justified. It is the run's error
against a reference at a tighter tolerance, or against an existing stored converged
reference, times a safety factor fixed before the candidate is measured. Each run is
compared with that reference; a pairwise $2E$ check uses the same $E$. This is the
usual convergence practice. The formal three-level refinement of
[SEP 16](sep-0016-unified-multistep.md) is how a route's accuracy is
*established*; it is not repeated in every test.

**Replay needs to meet that overall accuracy requirement; exact reproduction is not
required.** The current `FixedSchedule` stores step lengths and a start position.
`AdamsMarch._replay_to` still chooses orders and does not replay rejected trials;
its docstring explicitly permits different orders after an adaptive rejection.
A resumed march can also rebuild its history. Thus replay of an adaptive run is
an accuracy comparison unless the complete discrete computation is matched.

For rounding-level kernel/parity tests, run **both** sides on the same prescribed
schedule with a fixed-order method, matched startup and restart history, and
matched save-plane handling. An adaptive run may discover that schedule, but its
result is not the rounding oracle for the two replays. Ordinary variable-order
replay and resumed-versus-uninterrupted comparisons use accuracy envelopes.
Recording and replaying every adaptive decision is outside this SEP; no new
replay API or deterministic switch is required.

Step and source-call counts across independent adaptive runs are performance
diagnostics, with any regression bands declared from baseline data. Exact
accounting identities remain exact: a ledger must count the calls actually made,
and a prescribed schedule must execute its specified steps. SEP 16's observed
order-choice sensitivity motivates measuring work at matched error, rather than
requiring identical adaptive counts.

#### 4.5 Noise-seeded and ensemble runs

A seeded field is a specified initial condition whose short-distance evolution can
still be tested. Use the same stored launch/noise arrays for arithmetic comparisons;
this SEP does not weaken the reproducibility of the random input generator.

Where the dynamics make a long-distance realisation unsuitable for a pointwise
comparison, the test compares over a short window, or compares ensemble observables
(mean spectrum, coherence) with a tolerance set from the ensemble size [6]. Small
deterministic short-window cases remain the primary tests; ensemble studies belong
in the slow lane.

#### 4.6 The plasma column as parallel scans

The field-grid column of SEP 18 has three causal running integrals at each node.
Writing the currents and polarisation per neutral-particle density, the stages are
exposure $X=\int w\,dt$, fraction $f=1-(1-f_0)e^{-\max(X,0)}$, free current
$j_{\rm free}=\int(q^2/m)fE\,dt$, and polarisation
$p_{\rm plasma}=\int(j_{\rm free}+j_{\rm loss})\,dt$, with
$j_{\rm loss}=I_p w(1-f)E/|E|^2$ under the existing zero-field convention. Here
$f_0$ is the preionised fraction, $X$ is dimensionless, $w$ is in s⁻¹, $E$ in V/m,
$I_p$ in J, $j$ in C m/s and $p_{\rm plasma}$ in C m. Multiplication by number density
recovers physical current density and polarisation. The band filter, exposure
floor, raw loss rate and boundary closures remain those of SEP 18.

Each scan consumes inputs known after the preceding scan and pointwise map.
A chunked prefix algorithm [7] forms local prefixes, scans chunk totals, then adds
offsets. For chunk length at most $m$ and an offset scan of depth $d$, a conservative
addition-depth bound is $D=m+d+1$; derive $d$ for the actual scan implementation,
including its up/down passes, rather than assuming $\log_2 P$ for every algorithm.
For already computed increments $x_i$,

$$|\widehat S_j-S_j|\le\gamma_D\sum_{i\le j}|x_i|.$$

Add the accumulated errors in forming the increments and propagate input errors
from earlier scans and maps. Compare two implementations with both error budgets.
A smaller depth improves this worst-case bound; it does not guarantee a smaller
observed error.

The **unfiltered** nonnegative rate has a well-conditioned exposure sum. The band
filter has negative taps and may produce negative increments; SEP 18 floors the
resulting exposure at zero. Both current integrands are signed and may cancel
strongly over carrier cycles. Their relative prefix errors need not be small.
Use absolute bounds based on prefix sums of magnitudes, then carry them through
population, loss current, density, projection and propagation scaling. Test a
preionised oscillatory field, zero crossings, tiny exposure, near-threshold rates,
long cancelling prefixes and pulses reaching either window edge.

For the interior stencil
$\tilde y_n=c_0y_n+\sum_{k=1}^Kc_k(y_{n-k}+y_{n+k})$, define
$A_n=|c_0y_n|+\sum_{k=1}^K|c_k|(|y_{n-k}|+|y_{n+k}|)$.
A conservative real-operation budget for the paired evaluation is
$\gamma_{2K+2}A_n$; add input and coefficient uncertainties where applicable.
Other contractions and the endpoint extrapolation get their own budgets.
Compute the tap norm from the actual full-precision taps. The illustrated default
taps in SEP 18 give $|c_0|+2\sum_{k>0}|c_k|\approx2.0913$, not 1.9.
The unchanged band certificate concerns the exact discrete multiplier, not the
rounding error of the complete nonlinear scan chain; qualify both separately.

The vectorised implementation must preserve the causal initial condition, endpoint
extrapolation, exposure floor and undershoot diagnostic, raw-rate loss work, and
all scalar/vector, preionised and multichannel cases. It may skip leading work only
when the entire required stencil is zero: a symmetric filter can respond before
the first nonzero raw sample, and preionisation can produce current even when the
rate is identically zero. Scratch storage and scan offsets enter the memory ledger.

### 5. Proposed interface

There is no new public Python interface. What changes is what solprop promises and how
the code and the tests keep the promise.

**The policy.** This text is to be adopted in `AGENTS.md` (solprop), `AGENTS.md`
(solprop-kokkos), `testing.md` and `backends.md`:

```text
Independent floating-point computations are compared by numerical error, not bits.
- No bitwise agreement is required across backends, builds, machines, thread counts,
  batch membership, scan execution or repeated propagation runs.
- Matched discrete methods and histories use rounding budgets that account for
  operations, magnitudes, conditioning, transforms and propagation sensitivity.
- Adaptive runs and ordinary replay meet a per-case accuracy envelope set against a
  tighter or stored converged reference; rtol/atol alone do not bound global error.
  Replay need not reproduce the original adaptive trajectory or agree with it at
  rounding level.
- Kernels may reorder, vectorise, thread and contract. Stable increments,
  small-argument functions, defect calculations and compensated sums keep their
  protections. Contraction is on by default for compiled kernels; the few protected
  helpers opt out and stay strict. Reassociation is enabled only on the loops that
  need it, per kernel. No umbrella fast-math,
  nnan/ninf, nsz, approximate functions/reciprocals, or flush-to-zero.
- Preserve exact structural contracts: lossless copies and storage, integer and
  identity metadata, hard-mask zeros and explicitly exact no-op/zero behavior.
  A mathematical zero reached through inexact arithmetic uses a justified absolute
  tolerance. Adaptive work counts may differ; accounting for actual work is exact.
```

**The user's promise**, on `guide/numerics/reproducibility.md`:

- **Numerical accuracy, not bitwise identity.** Qualified cases agree within their
  documented error budgets. Establish the accuracy of a new problem with a
  convergence study; local tolerances alone cannot guarantee its global error.
- **Rounding can change the steps and orders.** Adaptive trajectories and work counts
  may differ across machines, backends or thread layouts.
- **Replay retains the accuracy requirement.** A recorded step schedule is useful for
  repeat calculations and debugging. Calibrate it for the method and problem used;
  it need not reproduce the original adaptive run exactly or at rounding level.
- **Noise inputs remain controlled.** Compare matched realisations where well
  conditioned, and appropriate ensemble observables where sensitivity requires it.

**Build and compiler flags.**

The two licences carry different risk, so they get different defaults.

- **Contraction** (fused multiply-add) rounds once where the pair rounds twice. It is
  as accurate or more so almost everywhere. Its one hazard is exact cancellation:
  `x*y - y*x` need not be zero, and a product formed two ways may differ. It is
  therefore **on by default**, and a short list of protected helpers opts out. Those
  are the helpers that rely on an exact cancellation or form a small difference
  (§4.2).
- **Reassociation** lets the compiler treat addition as associative. That is what
  vectorises reductions, and it also licenses rewrites such as `(a + b) - a → b`,
  which silently remove a compensation term. It is therefore **opt-in per kernel**,
  on the specific loops that need it: reductions and the band stencil.

How each backend applies these:

- **Kokkos CPU:** value-safe compilation with `-ffp-contract=fast` on Clang/GCC.
  Protected blocks use `#pragma clang fp contract(off)` (or the standard
  `#pragma STDC FP_CONTRACT OFF`). Reassociation, where wanted, is scoped by
  `#pragma clang fp reassociate(on)` on the loop. Check once per compiler that the
  settings take effect: Clang resets contraction after `-fno-fast-math` if the order
  is wrong [12]. Settle the icpx equivalent in Phase 4. The `__FAST_MATH__` refusal
  stays.
- **Numba:** the `solprop._numba_cache.njit` wrapper applies `fastmath={"contract"}`
  by default, accepts `fastmath={"contract", "reassoc"}` from a kernel that asks for
  it, and accepts `strict=True` for a protected helper. It rejects any other flag and
  never `True` [11].
  - numba's flags apply to a whole compiled function, including the helpers inlined
    into it, so a strict helper is compiled separately and called, not inlined.
  - There are only a few such helpers, so the call cost is small.
- The evidence that a kernel is safe under these flags is its R comparison against
  `numpy_ref` on stressing inputs (§7.1), not an audit document. Check once that
  numba's cache key changes when a kernel's flags change.

Flush-to-zero, denormals-are-zero, approximate transcendental functions and reciprocal
approximations are deferred to separately justified work. There is no deterministic
mode in this SEP. These choices were confirmed in human review on 2026-10-17.

### 6. Implementation strategies

#### 6.1 Documents to change

| document | change |
|---|---|
| `AGENTS.md` (solprop) | the policy block of §5 under the backends rules; "Reductions are deterministic" and "no fastmath" replaced; the tolerance policy's bitwise sentence removed |
| `solprop-kokkos/AGENTS.md` | "repeat runs of one build are bitwise identical", the native GEMM held bitwise to `KokkosBlas::gemm`, and fast-math wording replaced by the same policy |
| `docs/dev/testing.md` | §3 and §4 rewritten (bounds, replays, bands); §5 digests and pins replaced by tolerance-scaled records |
| `docs/dev/accuracy.md` | §7 ladder separated into matched-discrete rounding budgets and adaptive accuracy envelopes; the "two runs, one thread against many: bitwise" row and the "scans and batches … bitwise" paragraph replaced |
| `docs/dev/backends.md` | §3 "No fastmath" and "Reductions … bitwise repeatable" replaced; the small-FFT rule re-stated as a speed threshold; the build-profile pins paragraph removed |
| `docs/dev/kernel-contract.md` | the bitwise exceptions (plasma, Raman `evaluate`, statistics on one state) become rounding bounds |
| `docs/dev/architecture.md` | `numba_cpu` description (contraction by default, strict opt-outs, reassociation per kernel); §8 fused passes cross-referenced for Kokkos |
| `docs/dev/stepper.md` | replay documented as meeting an overall accuracy target, with the Adams order/history limitation explicit |
| `docs/dev/decisions.md` | a new ADR, "Reproducibility to the error bound", superseding the bitwise clauses of ADR-013 (batches), ADR-022 (testing), ADR-034 (pins for the reference build) and ADR-039 (the `numpy_ref` resume clause) |
| `docs/dev/benchmarks.md` | the benchmark set of §7.4 and its first entries |
| `docs/dev/roadmap.md` | the band-rule cost item replaced by this SEP; this SEP in the draft list |
| open SEPs | success criteria stated as bit identities restated as bounds (§9) |

Archived SEPs are records and are not edited. Where a current page cites one of their
bitwise statements, the page changes, not the archive.

#### 6.2 The test suite

The re-engineering is the largest part of this SEP. It has five parts.

**An inventory first.** A script lists every exact-equality assertion and every
bitwise, digest or pin reference in both repositories' tests, and classifies it.

| class | meaning | becomes |
|---|---|---|
| P, structural contract | lossless storage/copies/views, integer accounting, identity metadata, hard masks and explicitly exact zero/no-op behavior; a plane row belongs here only if it reuses the same data rather than recomputing a statistic | stays exact; document why the contract is exact |
| R, matched discrete computation | backend, thread, batch, scan or execution-path comparisons with the same method, orders, history and schedule (§4.3) | a justified rounding budget; nontrivial witness plus separate zero and structural cases |
| T, accuracy | adaptive, variable-order replay or restart comparisons with differing discrete trajectories | each run against a tighter or stored converged reference at a per-case envelope fixed in advance; work counts as loose bands |
| D, digest or pinned record | byte digests, records valid on one platform or one FFTW build | recorded states compared at a bound of class R (if the discrete computation is matched) or T; the platform skips removed |
| N, noise | a seeded realisation compared at full length | a short-distance R comparison or an ensemble statistic (§4.5) |

The inventory is committed as a report beside the migration. Phase 1 is complete only
when every R, T, D and N entry has moved.

Hashes used for configuration identity, cache validation, artifact integrity or
provenance remain exact structural checks. Only digests used to require identical
numerical outputs are replaced; the inventory distinguishes these purposes.

**A bounds module.** `tests/bounds.py` gathers reusable derivations, keeping
case-specific reference data and sensitivity assumptions with the tests:

- $\gamma_k$;
- the FFT round-trip bound $B_{\mathrm{fft}}(n)$;
- the scan bound $B_{\mathrm{scan}}(n)$;
- a step budget including operation counts, input magnitudes and nonlinear sensitivity;
- the conditional propagation growth of §4.3;
- separately labelled accuracy envelopes and reference uncertainties from §4.4,
  tied to their problem, method, distance and observable.

Each bound is fixed before the candidate is measured, never fitted to the
discrepancy it then accepts. The comparison helper checks finiteness and rejects a
trivial (zero) reference field. Weak-bin and observable checks stay where the tests
already have them.

**Replay and independent accuracy checks.** For class R, compare two prescribed
runs with a fixed method/order and identical startup and restart history (§4.4).
For class T, test each adaptive or ordinary replay result against its independent
accuracy envelope. Retain tests of controller behavior, adaptive convergence and
source-call accounting; converting every adaptive test to fixed stepping would
remove coverage of the behavior most exposed to rounding changes.

**Infrastructure removed:**

- `tests/platform_pins.py` and the platform skips;
- the bitwise branch of `tests/kokkos_build.py` (the parity bounds remain);
- `tests/fftw_witness.py`'s plan isolation and its baseline, with `tests/test_fftw_witness.py`;
- the dependence on a Homebrew-built pyFFTW (the PyPI wheel becomes sufficient).

The `REFERENCE` Kokkos profile retains its name and tidy/sanitizer role. It no
longer defines privileged numerical pins; renaming a build interface is unnecessary.

**Recorded states.** These are the legacy entry points and the opt-in features left
off. They keep `ndarrays_regression`, at a bound derived as R for fixed-schedule cases,
or at a T envelope otherwise. A phase that changes their arithmetic recaptures them
after checking that old against new is within that bound, and says so in the commit
message.

**Determinism of the tests themselves.** A test whose result can depend on a fork is
written so that a fork cannot fail it: either it compares matched discrete
computations, or it uses an accuracy envelope. One deliberately near-threshold
controller case checks that a real fork passes.
The suite's own randomness rules (`testing.md` §4) are unchanged.

#### 6.3 Host work in the step loop

The shared `StepDriver` owns the host loop. Kokkos replaces a trial with a native
call, but still pays for host table evaluation, controller work and records.

- **Batch stage-scalar evaluation where positions are known.** DOPRI5 has seven
  stages, normally six new RHS evaluations with FSAL; the number of scalar-bundle
  evaluations is a separate measured quantity. `translate/stage.py` already supports
  a hoisted `StepAnchor`. Retain that reuse and evaluate distinct requested stage
  positions together, grouping by table interval and preserving kink/endpoint
  selection. Adams has different startup, endpoint and history needs; batch only
  positions actually required, without speculative response/provider calls.
  Preserve the stable integrated-weight increments, and compare the vectorised
  tables at boundaries, tiny steps and large anchors. Returning existing
  `StageScalars` objects through existing hooks can keep the seam unchanged; changing
  their fields or the protected driver surface requires a schema bump (§9).
- **Cache immutable execution facts.** `_record_execution` calls `backend.metadata()`
  at every plane, and that rebuilds facts that cannot change during a run: build
  information, pools and plans. The backend caches those once. Response ledgers,
  orders, attempts, TDSE facts and the outcome are still read fresh at every plane,
  so the persisted record is unchanged and needs no schema change. Terminal records
  are still written on every exit path ([ADR-047](../../decisions.md)).
- **Controller and schedule.** Leave their algorithm unchanged. Phase 0 measures their
  cost independently rather than relying on the unverified per-trial split in §2.

#### 6.4 Output stores

**HDF5: preallocation with a saved-plane count.** Today `_extend` grows each
per-plane dataset by one row at every plane, and `resume` and the readers take the
number of saved planes from the field dataset's extent (`output/hdf5.py`). So
preallocation needs one addition: a count of the planes actually written. This is
what Luna.jl's HDF5 output does: it creates the field dataset at the planned length
and records the number saved after each save.

- At the first save, create every per-plane dataset at the planned number of planes
  (`problem.stepper.save_planes`), with an unlimited leading dimension
  (`maxshape=(None, *trailing_shape)` [13]), so that a run saving more planes can
  still extend it.
- After each plane's data is written, update an `n_saved` attribute on the file.
  Write the data first and the count last, so an interruption leaves the count at
  the last complete plane.
- `resume` and every reader use `n_saved` when present, and the extent otherwise, as
  for files written before this change. A reader shows only the first `n_saved`
  planes.
- On a clean close, trim every dataset to `n_saved`, so that a finished file looks
  exactly as it does today.
- The step-history columns grow geometrically with their own count attribute, by
  the same rule.
- Cache the dataset handles instead of looking each one up by name at every plane.
  In the profile these lookups cost more than the resizes.

The only schema change is the optional `n_saved` attribute; files without it read as
before. Tests:
- an interrupted file, at the plane boundary and mid-plane, reads and resumes
  correctly;
- a run that saves more planes than planned extends its datasets;
- a pre-change file reads unchanged.

**Memory output.** `MemoryOutput` already keeps a count of valid planes separate from
its capacity. It starts with 16 planes and doubles by copying. Proposed: start with
the run's planned number of planes, and keep doubling for runs that save more. The
memory admission must charge that capacity; Phase 2 checks that it does.
Statistics-only planes stay with [SEP 22](sep-0022-memory-lean-spatial-execution.md).

These changes preserve stored numerical values exactly when fed the same `SavedPlane`
objects. End-to-end runs may still differ numerically and use R or T comparisons.

#### 6.5 Translation

On a z-varying guide the dispersion tables took 0.46 s on the measured case, mostly
in about 1 000 scalar `neff` calls. The public mode interface takes a scalar `z`, so
batching applies only to built-in modes whose internals already accept arrays; user
modes keep the scalar path, and no public signature changes. This is a lower
priority than §6.3–§6.4, and is done only if Phase 0 confirms the cost.

#### 6.6 `numba_cpu`

`numba_cpu` is **already fused**. Its stage kernels perform `architecture.md` §8's seven
passes in three traversals per stage: combination with lab frame, noise and pad; the
whole nonlinear column at one node; and crop, prefactor, windows, return and
accumulation (`numba_cpu/kernels.py`). Kernel fusion is therefore a Kokkos item
(§6.7). On `numba_cpu` the gains come from the arithmetic and the threading.

1. **Contraction and reassociation** on the kernels that opt in (§5), kept only where
   the benchmark shows a gain. Protected helpers and finite-value guards stay strict,
   including through inlining.
2. **The multipass band column.**
   - Replace the lagged ring-buffer pipeline (`make_band_plasma_column`) by whole-row
     passes: rate and its band filter, exposure scan and floor, fraction map, filtered
     free-current integrand and its scan, then the filtered total current and outer
     scan. Retain the raw rate separately for the loss current and work diagnostic.
   - Each pass is contiguous and is a candidate for vectorisation. Derive the scratch
     lifetime and size for scalar/vector and multichannel routes, including rate,
     fraction, filter, current, offsets and diagnostics; charge all live buffers.
     Three arrays per node is an estimate, not an allocation bound.
   - This is the roadmap's "vectorised multipass column".
3. **Threaded single-node columns.**
   - With the column in passes, every pointwise pass is a `prange` over chunks of $t$,
     and every scan follows the chunked dependency structure of §4.6. Avoid nested
     node/time parallel regions and oversubscription of BLAS/FFTW pools.
   - The regime threshold (`REGIME_ELEMENTS`, currently 10⁶) is re-measured for speed
     within the accuracy and memory requirements, and extended to choose chunked
     columns for single-node problems whose fine-grid size $n_{\mathrm{to}}$ repays it.
4. **Free-order reductions.** The error norm and the statistics partials keep their
   two-stage structure where it is fastest, but no longer need the serial index-order
   sum. A threaded tree is used where it measures faster.
5. **General-modal contractions** (`numba_cpu/general_modal.py`). These are synthesis,
   cubic, delayed and projection contractions "in the reference's order, nodes chunked
   as the reference chunks them". They become plain GEMMs through the existing BLAS
   pool, or numba loops ordered for locality, whichever measures faster.
6. **The coupled-flow channel sums and the dispersive mixture filter.** These were held
   to the reference's order "to the last bit", and are free to vectorise.
7. **FFTW planning.** `MEASURE` planning and the wisdom it stores are allowed without
   the isolation the digests needed. The plan flag stays a setting.

#### 6.7 Kokkos CPU

1. **Contraction.** Enable it as in §5, then remove the statement splits in
   `band_window`, `band_edge` and the complex products, which existed only for
   the bits.
2. **Stage-kernel fusion on the general engine.** One stage is about 8–10 launches and
   two FFTs today (`general.cpp`): `stage.phase`, `combine_pad`, the inverse FFT,
   `transform.scale`, one or two zero-fill `deep_copy`s, `synthesise`, the response
   column, `project`, the forward FFT, the absorber and `finish_accumulate`. Proposed
   fusions:
   - `stage.phase` into `combine_pad`;
   - the finish of stage $s$ with the combination of stage $s+1$ (both are pointwise
     over $\omega$, between the forward FFT of one stage and the inverse FFT of the
     next);
   - the zero-fill copies written by the kernels that fill the rest of the padded
     buffer;
   - the 1×1 `transform.scale`, `synthesise` and `project` of a single-mode or
     mode-averaged problem folded into the response column, as the Kerr-only engine
     (`core.cpp`, `kerr_1x1`) already does.

   These are candidates, not unconditional fusions. The general engine can project
   multiple node chunks and add Raman, plasma and absorber contributions. A fill
   must not erase partial accumulation, and a next-stage combination must wait for
   all source contributions and any coupled-flow action. Keep initial/last-stage,
   rejection/FSAL and reuse dependencies explicit. Fold 1×1 spatial maps only while
   preserving all normalisation, taper and projection factors; a single mode with
   resolved quadrature nodes is not a 1×1 map. Measure launch/pass counts after these conditions
   are met; changed summation or scaling order still needs an R budget.
3. **Reuse propagator work where valid.** Combination needs `exp(+phase)` and finish
   needs `exp(-phase)`. They are conjugates only for a purely imaginary phase; loss
   and normalisation terms have real parts. Assess shared sine/cosine work, separate
   magnitude factors or reciprocals with explicit range/error checks. Do not obtain
   an inverse from an underflowed forward value, and do not replace the protected
   phase increment. Charge any retained factors in the memory ledger.
4. **The band column** as passes, as on `numba_cpu` (§6.6.2).
5. **Parallel single-node columns.** The device decomposition already exists and is
   qualified to $B_{\mathrm{scan}}$ (`include/solprop_kokkos/plasma.hpp`, team
   `parallel_scan`). The host gets the same decomposition, with teams over chunks of
   $t$ instead of one-thread teams. `SOLPROP_KOKKOS_HOST_TEAM_BAND` becomes the
   production candidate rather than a qualification-only switch. Its existing
   device scan budget does not automatically cover a different CPU scan tree;
   qualify each host implementation under §4.6.
6. **Reductions.**
   - `fixed_order_reduce` (chunks of 1 024 combined serially), `accept_update`,
     `weighted_error` and `core.cpp`'s `accept_and_score` (a single work item) become
     `parallel_reduce` or a threaded two-stage reduction, whichever measures faster.
   - The host general-modal projection takes the device's relaxed form. The reported
     device bottleneck motivates a CPU experiment; it does not predict the host speedup.
   - The native GEMM is no longer held to `KokkosBlas::gemm` bit for bit; it is held to
     the GEMM rounding bound.
7. **FFT threads.** The 8 192-sample threshold for threaded plans is re-measured by
   speed alone, per build profile, and may move down where threads pay.

#### 6.8 `numpy_ref`

The reference stays the oracle: clearest form first, vectorised NumPy, no compiled
code. It gains from the policy too:

- **Write the plasma scans as cumulative sums.** `plasma_scan` and `plasma_current`
  become `numpy.cumsum`-based forms where they are not already, instead of loops kept
  for an order other backends had to match.
- **Retain stable formulations.** In particular, the constant-operator increment
  remains protected by §4.2; a bitwise test is not evidence that its implementation
  exists only for bit identity. Classify its exact structural and numerical checks
  separately before changing them.
- **Resume meets the accuracy envelope** of §4.4; only matched restart histories
  are R comparisons. Retire the ADR-039 bitwise promise without claiming that
  restart is always a rounding-only perturbation.

#### 6.9 GPU validation and optimisation

The GPU campaign is implementation work, with correctness and performance evidence
required on each target. It proceeds in this order:

1. **NVIDIA A40 on a university cluster:** establish the first CUDA validation baseline, profile the
   supported short-window cases, and implement and measure an initial optimisation
   pass. Run GPU work inside a Slurm allocation. This pass checks the shared device
   implementation before the later campaigns and delivers its own before/after record.
2. **NVIDIA H200:** validate the A40-qualified revision, establish an H200 baseline,
   then profile and optimise for this device. Remeasure the A40 choices rather than
   assuming that their relative costs or preferred launch geometry transfer.
3. **AMD MI325X:** validate the shared revision on HIP, establish an MI325X baseline,
   then profile and optimise its supported paths. Reassess CUDA-specific choices and
   retain HIP-specific policies only where the measurements justify them.

**Candidates.** Use the profile to prioritise stage fusion and launch overhead,
plasma filter/scan decomposition, team and vector sizes, reduction trees, direct
versus scratch-staged band stencils, general-modal contractions, and unnecessary
transfers or synchronisation. Include single-node and resolved-node workloads,
scalar/vector fields, Raman and single/multiple plasma channels where supported.
Compare existing FFT/GEMM paths and scratch/chunk choices within the current memory
contract; this does not add an autotuner or a new memory-admission policy. The
[SEP 18 benchmark record](../../benchmarks.md#band-rule-plasma-integration) already
shows different preferred band-staging policies on H200 and MI325X, so neither
vendor's setting is prescribed for the other.

**Numerical contract.** Existing device tests that require identical adaptive counts
become matched-discrete or accuracy comparisons under §6.2. Every optimisation
retains double precision, the compiler-flag rules of §5, protected formulations and
the R/T budgets. A changed scan/reduction geometry needs a budget for its own tree;
GPU hardware or a speedup is not a reason to loosen accuracy requirements. Keep
explicit refusals for unsupported routes. On each device run `uv run pytest -m gpu`,
the applicable solprop-kokkos native checks and the affected R/T and work–precision
cases. Record the device compiler settings with the results.

**Deliverable per device.** Record the baseline revision/build, a profile identifying
the dominant costs, the tested candidates, retained and rejected changes, numerical
errors and work–precision results, repeated before/after timings and peak memory.
Use device traces where available; missing privileged counters do not excuse
skipping the optimisation pass. Time the final result without profiler overhead.
Completion requires at least one measured optimisation experiment on a demonstrated
bottleneck, not a prespecified speedup: if no candidate wins, retain the baseline
and report the failed experiments and remaining bottleneck. Shared CUDA/HIP changes
made during a later pass must be checked again on affected earlier targets; rerun
their affected correctness and performance cases rather than the entire campaign.

#### 6.10 Expected gains

These are the drafting estimates, to be replaced by Phase 0's measurements; their
baseline figures inherit §2's evidence limitations. All rows are one thread unless stated.

| item | case | today | expected | basis |
|---|---|---|---|---|
| band column as passes, vectorised | He mode-averaged, Kokkos | +0.77 s over trapezoid | +0.1–0.3 s | stencil of 9 taps over contiguous rows, AVX2/AVX-512 |
| parallel single-node column | He mode-averaged, 8 threads | no gain (4.34 s) | the column 3–6× faster | device decomposition; scans at $n_{\mathrm{to}}$ = 8 192 |
| stage-kernel fusion | mode-averaged plasma, Kokkos | 8–10 launches per stage | about half | launch and memory-pass count |
| contraction | compute-bound columns (plasma, Raman, Kerr cubic) | — | 5–15 % on those kernels | FMA throughput; memory-bound passes gain little |
| general-modal GEMM | multimode resolved, CPU | serial ordered contraction | large on the projection | device precedent (86 % of device time) |
| stage scalars once per trial | any cheap step | 0.45 ms per trial | < 0.1 ms | profile |
| immutable execution facts cached | any run | ≈ 2 ms per plane reported | reduced serialization; dynamic records remain | profile hypothesis |
| HDF5 preallocation | any run writing HDF5 | ≈ 5 ms per plane | ≈ 1–2 ms | per-dataset resize and lookup |
| dispersion-table sampling | z-varying guides | 0.46 s per run | ≈ 0.1 s | ≈ 1 000 scalar `neff` calls |

The table contains hypotheses, not promised gains. Do not sum savings whose timing
regions overlap, infer a total speedup from kernel throughput alone, or extrapolate
short-window measurements to the full run. Phase 0 supplies reproducible baselines;
subsequent phases replace these estimates with measurements over the stated window.
The GPU passes have no assumed speedup. Their targets and gains come from the
separate A40, H200 and MI325X baselines and experiments of §6.9.

#### 6.11 Risks

- **Weakening tests while changing arithmetic.** Phase 1 changes tests before kernels,
  and keeps the analytic and convergence checks.
- **Adaptive sensitivity.** Steps, orders and work can change. Adaptive runs are
  checked against their accuracy envelopes, not only through fixed-step tests.
- **Cancellation and compiler scope.** The R tests of relaxed kernels use stressing
  inputs: weak plasma signals, oscillatory currents, tiny phase increments and
  non-finite guards.
- **Hidden state.** Race freedom, complete initialisation, cache validity, correct
  metadata and exact ledgers remain requirements. A rounding budget cannot excuse
  uninitialised data, races or stale configuration.
- **Interrupted output.** The `n_saved` count is written after each plane's data and
  read by every reader, so preallocated but unwritten planes are never shown as saved
  (§6.4).

### 7. Verification and validation

**7.1 Numerical agreement and independent accuracy.** Every changed kernel is tested
on one evaluation, one step and about eight to ten steps against `numpy_ref`, with
matched inputs/method/history for class R. Derive its budget before measuring it,
record the observed margin, and preserve positive/nontrivial witness checks and
weak-bin metrics. Also retain analytic, independently formulated and convergence
references, especially where both backends share a changed body. Plasma qualification
retains SEP 18's preionised-tone response, causal boundary tests, population/work
checks and independent time-domain reference. No full-length CtC or capillary run
is introduced.

**7.2 Threads, batches and scans.** Compare one and several threads, batch/solo and
scan/solo using matched discrete computations for R or independent accuracy
qualification for T. Cover chunk boundaries, uneven tails, multiple channels,
polarisation and zero/identity cases. Test exact accounting separately. Scope GPU
checks to routes each device backend supports; retain explicit named refusals.

**7.3 Work–precision.** Re-run [SEP 16](sep-0016-unified-multistep.md)'s
canonical work–precision cases, on short windows, with default Adams and with
`RungeKutta()`, before and after Phases 3 and 4. A change is accepted if, at each
tolerance, the error against the stored reference stays within a factor of 2 of the
baseline, and the calls at matched error do not grow by more than 10 %. Improvements
are allowed. The factors are fixed now, not after a failure.

**7.4 Benchmarks.** Phase 0 defines short windows (seconds each):
- mode-averaged He RDW, with the band rule and with the trapezoid;
- resolved one-mode Ar plasma blue shift;
- four-mode He;
- reduced CtC;
- a general-modal case.

Each is timed at one thread and at several physical cores, on Linux and on Apple
silicon, warm and unprofiled, best of several repeats. Translation, kernels, host
loop and output are timed as disjoint parts. The benchmark entry records the
hardware, revisions and thread settings. A change is kept if it is faster on the case
it targets and slower on none by more than run-to-run noise.

For the GPU passes, use the supported subset of these same cases plus short
single/multichannel and scalar/vector cases needed to exercise the chosen kernels.
Preserve realistic grid and batch sizes while limiting propagation to about ten
steps; do not reduce every workload until launch latency is its only cost. Record
each device's driver, CUDA/ROCm and Kokkos/FFT/BLAS versions, build flags, launch
geometry and memory use. Device-event timing and explicit completion at timing
boundaries must measure completed work, not just asynchronous enqueue time. Report
device kernels, transfers/synchronisation and end-to-end wall time separately,
including host and output work. Warm up compilation/plans, use repeated unprofiled
timings, and compare before/after on the same device. Cross-device throughput is
context, not a substitute for an optimisation comparison. Recheck adaptive
work–precision under §7.3 after each retained arithmetic or execution change.

**7.5 Failure paths.** The migrated tests still exercise non-finite inputs, output
interruption and resume. A test whose bound became looser is spot-checked once
with a deliberate perturbation (a flipped sign, a dropped term) to show it still
fails.

### 8. Documentation

- The pages of §6.1.
- A new user page, `guide/numerics/reproducibility.md`, with the promise of §5, an example
  of replaying a run's schedule, and how to compare noise-seeded runs.
- Docstrings that say "bitwise", "to the last bit" or "in the reference's order" where
  that was a requirement, rewritten to state the bound instead.
- The `CHANGELOG.md` entry at `Final` citing this SEP.
- The GPU campaign reports and benchmark ledger, with separate A40, H200 and MI325X
  measurements, qualified static policies and any remaining performance limitations.

### 9. Compatibility and impact

- **Results.** Reordered arithmetic is qualified to R budgets; adaptive/restarted
  runs satisfy their T envelopes. No blanket global-error
  guarantee follows from a local tolerance. ADR-020's validated-result requirement
  remains in force and is checked by §7.3.
- **Recorded states** are preserved through test migration. A later recapture needs
  a documented reason and old/new comparison against an independent reference;
  it is not an automatic consequence of arithmetic changes.
- **Schemas.** The rounding policy and band traversal need no kernel-schema change.
  A changed protected driver hook or stage bundle does require `SCHEMA_VERSION`,
  adapter and frozen-contract test updates in the same phase. The output schema is
  unchanged except for an optional `n_saved` attribute on HDF5 files (§6.4), which
  older files lack and still read correctly. The execution record keeps its content
  (§6.3).
- **Open SEPs.** Proposals not yet final whose success criteria or design rules are
  stated as bit identities are restated as bounds in Phase 1. These are SEPs 1, 2, 4, 9,
  10, 20, 22, 23 and 24, as far as a search finds; the inventory confirms the list.
  Accepted SEPs are amended by a dated note in their Part III, not by editing a frozen
  Part I (SEP 0).
- **Users.** No API change. A user who compared two runs bit for bit will see
  differences within the applicable numerical envelope, and is pointed to the
  reproducibility page.
- **Luna.jl.** Nothing to record on the differences page.

### 10. Alternatives considered

1. **Keep the rule and optimise within it.** This is the [roadmap's](../../roadmap.md) "a
   kernel specialised on $K$" for the band rule, worth an estimated 10–25 %. It leaves
   the serial single-node column, the order-fixed reductions, the digests and the
   platform pins, and every future kernel pays the same constraint. Rejected.
2. **Relax only the band rule.** This would take the largest single gain and keep
   everything else. It leaves two rules where one is enough, keeps the test
   infrastructure, and makes every later relaxation its own argument. Rejected.
3. **Keep bitwise repeat runs on one build and one thread layout,** today's Kokkos
   stance. A fixed tree can itself be repeatable, so this does not forbid all parallel
   reductions. It does constrain scheduling and implementation choices beyond the
   accuracy contract the maintainer wants. Rejected for that reason.
4. **An opt-in deterministic mode.** This would mean two code paths for every
   reduction and scan, each needing its own tests. Replay does not provide that
   identical guarantee, but the maintainer requires accuracy rather than identical
   arithmetic or decisions (§4.4). Rejected as unnecessary scope.
5. **Reproducible summation algorithms** [8], which give order-independent bitwise sums
   at a cost. They solve a problem this SEP shows solprop does not have, and they cost
   performance and complexity in every reduction. Rejected.
6. **Full fast-math.** It implies `nnan` and `ninf`, under which the divergence guards
   and non-finite norms cannot be trusted, and it licenses reformulations that
   reintroduce cancellation (§4.2). Rejected in favour of the explicit flag set.

### 11. Phases

Each phase is its own `feature/sep-0025-…` branch. Follow `AGENTS.md`'s phased-SEP
rule: fast lint/format/type checks and directly affected test modules per slice;
the full gate, slow lane, coverage lane and Kokkos-selected slow lane once at final
integration, repairing failures and rerunning their modules. Documentation-only
slices follow the documentation-only checks. The policy and test migration precede
arithmetic changes so later phases are judged against established budgets.

**Phase 0 — Baseline and inventory.** No production code changes.

- **Benchmark set:** the set of §7.4, with a per-part timer (translation, kernels, host
  loop, output) that needs no profiler, run on Linux and Apple silicon; the baseline
  recorded in `benchmarks.md`.
- **Inventory:** the test inventory of §6.2 for both repositories, committed as a
  report.
- **GPU preparation:** define the supported case matrix and profiling/timing harness,
  and preserve the pre-optimisation source revision. At each device pass, freeze and
  measure its incoming revision before tuning, retaining the original revision for
  comparison of the cumulative changes on that same device.

**Phase 1 — Policy, documents and tests.** No numerics change.

- **Documents:** the documents of §6.1, including the new ADR and the open SEPs' bounds.
- **Tests:** `tests/bounds.py`; every R, T, D and N test migrated; the pins, the FFTW
  witness and the bitwise Kokkos branch removed; the matching changes in the
  solprop-kokkos tests.
- **Exit:** migrated modules pass on the supported CPU platforms (including Windows
  for the in-tree backends), with no platform skips left for bitwise reasons.
  Legitimate unsupported-platform and feature skips stay.

**Phase 2 — Host and output.**

- **Changes:** batched stage scalars; cached immutable execution facts; HDF5
  preallocation with `n_saved` and planned-size memory stores; batched dispersion sampling for built-in
  modes.
- **Verification:** stored records round-trip as P; table values use R; the existing
  interruption and resume tests pass, plus the
  `n_saved` tests of §6.4; the benchmark set is re-run.

**Phase 3 — `numpy_ref` and `numba_cpu`.**

- **`numpy_ref`:** the simplifications of §6.8.
- **`numba_cpu`:** the flag set; the multipass band column; threaded single-node columns
  and chunked scans; free-order reductions; general-modal contractions; re-measured
  regime thresholds.
- **Verification:** §7.1–§7.3, benchmarks.

**Phase 4 — Kokkos CPU** (solprop-kokkos with its solprop adapter).

- **Changes:** contraction; stage-kernel fusion; propagator reuse; the band column as
  passes; host team scans for single-node columns; parallel reductions; the relaxed
  general-modal projection and GEMM bound; re-measured FFT thresholds.
- **Verification:** §7.1–§7.3, benchmarks.

**Phase 5 — GPU validation and optimisation.** Each subphase has its own branch and
Part II report. Prepare code and the measurement harness before device sessions;
agree any rental budget before the H200/MI325X campaigns.

- **5A — NVIDIA A40 on a university cluster, first validation and optimisation.** Build and qualify
  the current CUDA implementation against the CPU/reference and independent
  accuracy checks. Establish the A40 timing/profile baseline, carry out the first
  optimisation pass of §6.9, and requalify retained changes. Exit with a validated
  revision and the measured experiment record, including rejected candidates.
- **5B — NVIDIA H200 optimisation.** Start from the A40-qualified implementation,
  validate it on H200 and record its own baseline. Profile and tune the dominant
  kernels, launch geometry and data movement; compare candidate and baseline on H200
  at matched accuracy. Recheck affected A40 cases if shared CUDA code changes.
- **5C — AMD MI325X optimisation.** Validate the shared implementation on HIP, record
  its own baseline and perform the same profile–optimise–measure–requalify loop.
  Compare direct/staged filters, scan/reduction geometry and other profile-led
  candidates on MI325X. Recheck affected CUDA cases when shared code changes.
- **Exit:** all three device reports meet §6.9's optimisation evidence requirements
  and §7's numerical/performance checks. A build-and-pass result alone is incomplete.
  The final retained code is covered by the required checks on all affected devices,
  with the common revision and target-specific build/settings recorded.

**Phase 6 — Integration and closing.**

- Full integration checks above, including 100 % coverage and Kokkos CPU
  qualification, plus the completed GPU evidence from Phase 5; the benchmark record;
  the user page; changelog and schema notes; remaining deferred work reconciled with
  the roadmap; then `Final`.

### 12. Review decisions and implementation questions

The human review on 2026-10-17 confirmed Q1, the compiler scope in Q2–Q3, and the
accuracy requirement for replay. Remaining engineering details have named phases.

- **Q1 — Exact structural checks: resolved.** Preserve lossless storage/copies,
  hard-mask zeros, exact no-op contracts and integer accounting (§5). Independent
  numerical computations use the appropriate error budget.
- **Q2 — Compiler scope and denormals: resolved.** Contraction is on by default, with
  strict opt-outs for protected helpers; reassociation is opt-in per kernel (§5). Flush-to-zero and denormals-are-zero are
  outside this SEP; no global relaxation through the Numba wrapper.
- **Q3 — Approximate functions/reciprocals: resolved.** Keep `afn`, `arcp` and vendor
  approximate functions off. Any later proposal needs its own accuracy/range budget.
- **Q4 — Replay and a deterministic switch: resolved.** Replay must meet the overall
  accuracy target, not reproduce bits or adaptive decisions. Use matched fixed
  methods for R tests; ordinary replay uses T. Full decision replay and a
  deterministic mode are not required.
- **Q5 — Execution records and file capacity: resolved.** HDF5 datasets are
  preallocated, with a Luna-style `n_saved` count written after each plane and used
  by every reader (§6.4). The execution record caches only immutable facts in memory
  (§6.3).
- **Q6 — Memory admission: Phase 2.** Check that admission charges the planned-size
  in-memory store; statistics-only planes stay with SEP 22.
- **Q7 — Noise-seeded tests: Phase 1.** For each, choose a short window or an
  ensemble observable (§4.5).
- **Q8 — Build profile name: resolved in review.** Keep `REFERENCE` for its existing
  build/tooling role; remove its privileged numerical-pin role. A rename is not
  needed to change the reproducibility contract.
- **Q9 — Open SEP restatements: Phase 1.** Update drafts in place with history entries;
  use dated Part III amendments for frozen accepted/in-progress proposals. Preserve
  archived records and exact structural requirements. Classify each promise rather
  than replacing every occurrence of “bitwise” mechanically.
- **Q10 — Evidence and margins: Phase 0.** Replace the motivating timings with
  short-window measurements, and fix the margins of §7.3–§7.4 before any
  optimisation.
- **Q11 — GPU campaign scope: resolved by the maintainer.** Include validation and
  optimisation on NVIDIA A40 on a university cluster first, then separate NVIDIA H200 and AMD MI325X
  optimisation passes. Device-specific tuning and before/after evidence are required;
  the implementation details and any rental budget are settled before each session.

### 13. References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM,
   Philadelphia, 2002). [doi:10.1137/1.9780898718027](https://epubs.siam.org/doi/10.1137/1.9780898718027)
2. IEEE Computer Society, "IEEE Standard for Floating-Point Arithmetic," IEEE Std
   754-2019 (2019). doi:10.1109/IEEESTD.2019.8766229.
   [IEEE standard record](https://standards.ieee.org/ieee/754/6210/).
3. J. C. Schatzman, "Accuracy of the discrete Fourier transform and the fast Fourier
   transform," *SIAM J. Sci. Comput.* **17**, 1150–1166 (1996).
   [doi:10.1137/S1064827593247023](https://epubs.siam.org/doi/abs/10.1137/S1064827593247023)
4. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301.
   [Authors' paper](https://fftw.org/fftw-paper-ieee.pdf).
5. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd rev. ed. (Springer, Berlin, 1993).
   [doi:10.1007/978-3-540-78862-1](https://link.springer.com/book/10.1007/978-3-540-78862-1)
6. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135 (2006).
   [doi:10.1103/RevModPhys.78.1135](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.78.1135)
7. G. E. Blelloch, "Prefix sums and their applications," Technical Report
   CMU-CS-90-190, Carnegie Mellon University (1990).
   [Author's report record](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html).
8. J. Demmel and H. D. Nguyen, "Fast reproducible floating-point summation," in
   *Proc. 21st IEEE Symposium on Computer Arithmetic (ARITH)*, 163–172 (2013).
   doi:10.1109/ARITH.2013.9. [Authors' project and bibliography](https://bebop.cs.berkeley.edu/reproblas/index.php).
9. D. Goldberg, "What every computer scientist should know about floating-point
   arithmetic," *ACM Comput. Surv.* **23**, 5 (1991). doi:10.1145/103162.103163.
   [Publisher-authorised edited reprint](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).
10. N. J. Higham, "The accuracy of floating point summation," *SIAM J. Sci. Comput.*
    **14**, 783–799 (1993). [doi:10.1137/0914050](https://epubs.siam.org/doi/10.1137/0914050).
    [Author's full text](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf), §§2–3.
11. LLVM, [Language Reference: Fast-Math Flags](https://www.llvm.org/docs/LangRef.html#fast-math-flags),
    consulted 2026-10-17. Verify against the LLVM version used by each supported build.
12. Clang, [Users Manual: floating-point options](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior),
    consulted 2026-10-17. Effective option order and strict helper scope are qualified per compiler.
13. h5py, [Datasets: resizable datasets](https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets),
    consulted 2026-10-17.

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

### Phase 0 — Baseline and inventory (2026-10-17)

**Delivered:**
- **The benchmark set** of §7.4 in `benchmarks/sep25/`:
  - `cases.py`: six short adaptive windows of the full-size guides, end to end;
  - `parts.py`: the per-part timer, which wraps existing methods with wall clocks and
    needs no profiler;
  - `run.py`: a sweep in fresh processes per thread setting.
- **The baselines**, recorded in [benchmarks.md](../../benchmarks.md) ("SEP 25 phase 0"),
  with data under `benchmarks/sep25/data/`: Linux at 1 and 18 threads, Apple silicon at
  1 and 6.
- **The work–precision baseline** (§7.3) for the GNLSE and comb-to-continuum windows on
  numba and Kokkos, and `benchmarks/sep25/wp_check.py`, which applies the acceptance
  rule.
- **The test inventory** of §6.2: `benchmarks/sep25/test_inventory.py` and
  its report (recorded in the development archive). It lists 2 201 lines in 334
  files, each with a proposed class and the slice that migrates it.
- **GPU preparation:** the device baseline revision is tagged, and the first device,
  the A40 of 5A, now builds and passes its qualification lanes with a recorded profile.
  The session scripts are kept outside the public tree.

**Deviations:**
- **The general modal case** runs two channels (the HE11 parity pair). The four-channel
  set (with TE01 and TM01) did not meet the modal table's accuracy gate over the case's
  band at practical knot counts; tabulating it is not this SEP's subject.
- **The SEP 16 campaign harness** passed `RungeKutta()` into the kernel problem, where
  the Runge–Kutta march is `None`. That one line is fixed.

**Findings beyond the motivating split:**
- **A thread-pool defect, fixed on main as its own change.** Under `OMP_PROC_BIND`, an
  OpenMP runtime binds the main thread to one CPU.
  - The Numba pools and the scan executor sized themselves from that thread's
    affinity, so they ran on one thread.
  - A pool forced larger put every worker on that one CPU, because workers inherit the
    creating thread's affinity.
  - Now the process's CPUs are read at import, and the Numba pool starts with them. The
    reduced comb-to-continuum window runs 3.7 times faster on 18 cores than on one.
  - The baseline's numba rows at 18 threads measured the defect and are not recorded.
- **Fixed costs the motivating split did not separate:**
  - git provenance: four git processes, 65 ms a run on the Mac;
  - serialising the run record three times when the output opens;
  - a numba execution record that exported FFTW wisdom at every plane;
  - numba's plane statistics: 7–12 % of the resolved and modal runs.
  Phase 2 takes the first three; the last belongs to Phase 3.
- **Translation of the four-mode case** is dominated by the adaptive Chebyshev build of
  the modal power weight, not by scalar `neff` calls (§6.5).
- **The steppers:** the default multistep march is 2.3 times Runge–Kutta's time on the
  averaged window and 4 times on the general modal one. This is recorded for the
  maintainer; it is outside this SEP's scope.

**Margins fixed (Q10):**
- Work–precision: the factors of §7.3.
- Benchmarks: the run-to-run spread over three repeats had a median of 0.6 % on Linux
  (90th percentile 2.8 %) and 2.3 % on the Mac (5.8 %). Slowdowns beyond 3 % on Linux
  and 6 % on the Mac count as regressions.

**Verification:**
- Documentation checks.
- `tests/test_seps.py`.
- ruff and basedpyright on `benchmarks/sep25`.
- The checker accepts Kokkos's work–precision run against numba's.

**Limitations:**
- Mac timings were taken on a lightly loaded shared machine.
- The motivating 1.5 m run is not reproduced at full length. Only windows are timed,
  as the qualification rules require.

**Handoff:** Phase 1 migrates the tests against the inventory; Phase 2 starts from the
fixed costs above.

### Phase 1 — Policy, documents and tests (2026-10-17)

**Delivered:**
- **The policy** of §5 in both repositories' `AGENTS.md` and in
  [ADR-049](../../decisions.md). ADR-049 supersedes the bitwise clauses of ADR-013, 022,
  034 and 039, and restates those of ADR-046 and 047.
- **The design and guide pages:**
  - [testing.md](../../testing.md) §3–§5: the P/R/T/D/N classes, recorded states at a
    tolerance, and the pins removed;
  - [accuracy.md](../../accuracy.md) §7: matched-discrete budgets beside accuracy
    envelopes;
  - [backends.md](../../backends.md): the floating-point licence;
  - kernel-contract, architecture, stepper (what a replay reproduces) and roadmap;
  - the user guide's promises on steppers, replay, resume, batches, scans, threads and
    statistics.
- **The open proposals:** SEPs 1, 2, 4, 9, 10, 22, 23 and 24 restate bit identities as
  bounds, each with a history entry; SEP 20 has a Part III note.
- **`tests/bounds.py`** (renamed from the accelerator module) adds:
  - `higham_gamma`, `prefix_bound`, `stencil_bound`, `propagation_growth` and
    `step_budget`;
  - `AccuracyEnvelope`;
  - `assert_within`, which checks finiteness, rejects a trivial reference and returns
    the margin;
  - `assert_work_within`.
  Slice helpers live beside their tests: `tests/march_bounds.py` and
  `tests/backends/matched.py`. solprop-kokkos has its own `tests/bounds.py` and
  `tests/native/bounds.hpp`.
- **The migration.** Every R, T, D and N line of the inventory is migrated in both
  repositories, in five slices.
  - About 1 600 lines turned out to be structural (P) and stay exact, with a comment
    where the reason is not obvious.
  - About 450 assertions became rounding bounds (R), fixed from operation counts.
  - About 20 became accuracy envelopes (T), with adaptive work counts held to bands.
  - Every comparison of two runs on a fixed schedule pins a fixed-order method, since
    the order choice is the one decision rounding could fork.
- **Infrastructure removed:**
  - `tests/platform_pins.py` and every platform skip;
  - `tests/fftw_witness.py`, its test and its baseline;
  - the dead `tests/kokkos_build.py`;
  - the platform-witness capture and audit;
  - the opt-in plasma byte-witness plugin.
  The digest probes of the Kokkos thread tests write arrays compared at their bounds.

**Recorded states:**
- **Kept:** the four legacy entry-point records. Their fixed-order reruns are within
  4.7e-15 of them, inside bounds of 4e-13 to 6e-13.
- **Recaptured:** 12 records (the pinned varying-wall and vector-wall records and ten
  flow-collar records). They were re-recorded because they now pin a fixed-order
  method. Old and new differ by the change of method, not by rounding, as the commit
  states.
- **Re-recorded unchanged:** the run-record hash, the M2 mixture baseline and the
  quadrature-rule digest. Each became stored arrays, after confirming that today's
  values still reproduced the old digests.

**Perturbation checks (§7.5):** every family of loosened assertions was checked once
with a relative perturbation of 1e-9, a flipped sign or a dropped term. Every family
fails, with two exceptions:
- the planar repeat, whose bound is 5.7e-9 (the operator's square root), fails at 1e-7;
- the Kokkos phase-B statistics are held to 1e3 times their field budget and fail at
  1e-6. A tighter statistic bound is a Phase 4 follow-up.

**Verification:**
- The fast and medium lanes on the merged tree: 15 311 passed and 29 skipped on Apple
  silicon. On Linux (Xeon, GCC 13, OpenBLAS, FFTW 3.3.10), 15 305 passed and 6 failed.
  - Three of the six were bounds the migration set too tight for another platform:
    - an eigensolve bound sized by the wrong array length;
    - records of computed spectra, whose empty wings carry FFT rounding;
    - a spline's raw coefficients.
  - Two failed on Linux before this SEP: SEP 18's work-defect diagnostic was measured
    against its own size, not the work it is relative to.
  - One was a configuration digest the inventory missed, because it compares with
    `==`.
  All six are fixed, with each bound justified in the test. They pass on both
  platforms; the full Linux lane is not rerun.
- solprop-kokkos: Python tests (275 passed) and native ctest on the reference profile.
- ruff, basedpyright and the strict documentation build.

**Not run:**
- the GPU-marked tests (migrated, run in Phase 5);
- Windows, which only CI covers, after a push.

### Phase 2 — Host and output (2026-10-18)

**Delivered** (no schema change; every persisted record and table unchanged unless
stated):
- **Provenance and the run record.** `GitProvenance.detect` runs one
  `git status --porcelain=v2` per run, and asks once per process whether the checkout
  tracks the package; macOS reads its memory through `sysconf`. The record's hashed
  part is serialised once (`RunMetadata`), and the numba execution record takes the
  FFTW wisdom digest once, at its first record.
- **Stage scalars (2a).** `PiecewiseChebyshev.at_positions` evaluates a table at
  several positions in one compiled pass, operation for operation as `__call__`;
  `stage.scalars_at` builds a step's bundles from one read per table, and
  `StepDriver._stage_bundles` serves them from the hoisted anchor. `numba_cpu` and
  `numpy_ref` Runge--Kutta trials use it. The Kokkos adapter does not yet (Phase 4).
- **Output (2c, 2d; Q5, Q6).** `HDF5Output` creates its per-plane datasets for every
  reserved plane, caches the handles, grows by doubling and writes a root `n_saved`
  attribute after each plane's data; readers, `resume` and appending handlers read
  that count and fall back to extents; a clean close trims, so a finished file's
  layout is unchanged. `MemoryOutput` is allocated for its reserved planes, and memory
  admission now charges an in-memory store at its planned capacity (the `output`
  group); a file output is charged nothing.
- **Translation (§6.5, taken because Phase 0 found the four-mode case's translation
  at 23 % of its run).** The piecewise Chebyshev walk resolves the intervals it is
  about to visit together, one sampler call per round, and replays its own decisions,
  so every table is bitwise the interval-by-interval one. Frozen copies of a capillary
  whose susceptibility rows are unchanged share their origin's sampled terms, and the
  shared-sampling scope covers the whole assembly.
- **The statistics power weight** was tabulated at 1e-14, below the rounding noise of
  its own density mapping over a short gradient window (about 1e-13): it exhausted its
  512-interval budget and warned on every run, and was most of what translation still
  cost. With the maintainer's decision it now uses the default 1e-12, and a parameter
  expansion's acceptance is floored at its mapping's rounding noise. Statistics move
  at the 1e-13 level.

**Not done:** batched dispersion sampling through array-`z` mode paths (§6.5's second
half) was not needed once the walk was batched; it goes to the roadmap if a later
profile asks for it.

**Measured** (Xeon, exclusive, best of three; numba against main with the Phase 0
affinity fix, Kokkos against the Phase 0 baseline with the same Kokkos build; data in
`benchmarks/sep25/data/main-numba-xeon/` and `phase2-xeon/`):

| Case, Runge--Kutta | numba t1 | numba t18 | Kokkos t1 | Kokkos t18 |
|---|---|---|---|---|
| He averaged, band, memory / HDF5 | −15 / −27 % | −8 / −18 % | −7 / −22 % | −6 / −19 % |
| He averaged, trapezoid, memory / HDF5 | −20 / −32 % | −25 / −15 % | −9 / −25 % | −5 / −22 % |
| He four modes, memory / HDF5 | −19 / −20 % | −19 / −21 % | −21 / −24 % | −47 / −50 % |
| Ar blue shift, HDF5 | −2 % | −3 % | −2 % | −7 % |
| Reduced CtC, HDF5 | −2 % | −9 % | −2 % | −8 % |
| General modal, HDF5 | −6 % | −6 % | +1 % | −3 % |

- Per run: opening the output 100 → 47 ms, the per-plane execution record 50 → 15 ms,
  plane saves 90 → 35 ms, the stage scalars about a third of their time. The
  four-mode case translates in 0.17 s, against 1.06 s.
- No case is slower beyond the Phase 0 noise. The averaged cases' kernel time at 18
  threads varies by ±15 % between rounds of one build; Phase 2 does not touch kernels.
- Translation on the Mac (contended, warm): four modes 302 → 47 ms, averaged 45 →
  31 ms.

**Verification:**
- Targeted modules on the merged branch: translate, output, driver, modes, the driver
  and execution-record tests, `test_process` and `test_linear` (4 408 passed); the new
  tests in `tests/output/test_record_reuse.py`, `test_preallocation.py`,
  `tests/backends/test_execution_record_cost.py` and
  `tests/translate/test_batched_sampling.py`; a narrow-range expansion test that fails
  without the mapping floor.
- Tables: every array of the translated problems of the four capillary benchmark cases
  is bitwise unchanged by the batched walk (Apple silicon; the tests check it on each
  platform).
- ruff, basedpyright.

**Not run:** the full gate (Phase 6); Windows.

### Phase 3 — `numpy_ref` and `numba_cpu` (2026-10-18)

Two parallel slices: A (flags, the band column, threading) and B (reductions,
statistics, contractions, flow), merged and then rebased over Phase 2.

**Delivered:**
- **The licence (§5).** `solprop._numba_cache.njit` compiles with
  `fastmath={"contract"}` by default, accepts `{"contract", "reassoc"}` or `strict=True`,
  and refuses every other flag set, `fastmath=True`, and a strict helper that is
  inlined. The options of a kernel and of every dispatcher it calls are part of the
  disk-cache key; without that a caller kept serving contracted code after a callee
  became strict. Protected helpers compile strict and are called: the population maps,
  the elliptic decomposition and moments, the phi-functions, the multistep weights,
  correctors, carries and defects, the Adams basis rows and both Chebyshev evaluations
  (the one-point one and Phase 2's batched one, so a stage bundle equals its one-point
  read). Reassociation gained nothing on the band column and is used nowhere.
- **The band column as passes (§6.3).** Rate; filter, then the exposure scan fused
  with the fraction and weight; per component the integrand, its filter and its scan
  fused with the current; the current's filter and scan fused with the polarisation.
  Causal start, point reflection, exposure floor, the raw rate's loss current, several
  channels, scalar and vector are kept; stencils that read only zeros are skipped. Its
  scratch is six rows per node, the `plasma_work` ledger row, replacing the ring
  buffers. On its own it is 1.7 times faster.
- **Threading.** A time-parallel column (local prefixes, a scan of totals, offsets)
  was built and measured: numba's parallel regions ran each pass 1.8 to 6 times slower
  per thread, and six threads only matched the serial column at 131 072 samples, so it
  was dropped. Re-measured, `MODAL_REGIME_ELEMENTS` falls from 262 144 to 131 072, which
  threads the argon blue-shift case over its nodes.
- **Reductions and contractions (§6.3 items 4–6).** The accept pass and weighted
  residual thread their columns on states of at least 65 536 elements in 8 columns; the
  coupled flow's passes thread from 32 768 elements; a plane's plasma statistics share
  one scan per member (eight before, on a default capillary spec), values unchanged;
  the general-modal synthesis and projection index the leading bins directly when the
  crop is leading. They are matrix-vector products per bin, not GEMMs, so BLAS does
  not apply. `mixture_rows` and `numpy_ref`'s scans needed only their docstrings.
- **FFTW (item 7).** Re-measured on a loaded Mac: threads paid only from about 131 072
  elements; `FFTW_THREAD_ELEMENTS` is unchanged pending a Linux measurement.

**Bounds:** the band column is held to absolute per-sample budgets on the §4.6 stress
set, K = 4 and 0, scalar and vector (each filter by `stencil_bound` plus the
reflection's reach, each scan by `prefix_bound` at depth `n_to`); the worst case uses
0.26 % of its bound, and a 1e-12 density perturbation fails 24 of 28 one-channel cases.
The recycled-stage identity, the bodies' rate rows and the accept and flow passes have
re-derived bounds that admit contraction (margins 0.008, 0.15 and 1e-4).

**Work–precision (§7.3):** accepted by `wp_check` for GNLSE and CtC with the
Runge--Kutta, interaction and exponential methods (data `benchmarks/sep25/data/wp-phase3a/`).

**Measured** (Linux workstation, exclusive, best of three, against main before Phase 2,
so Phase 3 alone; data `benchmarks/sep25/data/phase3-xeon/` against `main-numba-xeon/`):

| Case, `numba_cpu` | RK t1 | RK t18 | default t1 | default t18 |
|---|---|---|---|---|
| Ar blue shift | −58 % | −85 % | −52 % | −75 % |
| Reduced CtC | −37 % | −13 % | −34 % | −12 % |
| He four modes | −33 % | −22 % | −25 % | −17 % |
| General modal | −20 % | −6 % | −12 % | −6 % |
| He averaged, band | −17 % | −18 % | −9 % | −1 % |
| He averaged, trapezoid | −2 % | 0 % | −2 % | +2 % |

Memory output shown; HDF5 output is alike. Plane statistics fell by a factor of 8 to 13
(argon 1.03 → 0.08 s). One row exceeds the Phase 0 noise threshold: the trapezoid case
with Runge--Kutta at 18 threads and HDF5 output, +12 %. Its rounds spread by ±15 % in
every build measured, the column it runs only gained contraction, and the case is
faster on one thread (0.40 s) than on 18; it is read as noise and left to the Phase 6
benchmark record.

**Verification:** every numba, band, envelope, memory, nonlinear, stepper, translate
and driver module on the merged branch (6 175 passed, randomised); the licence tests
(refusals, the cache key, strict helpers not inlined, `fmadd` absent from strict bodies
on arm64); ruff and basedpyright.

**Physics review:** no blockers. Its should-fixes were taken: the rate rows' bound now
carries the rounded interval offset through the cubic's slope, the reflection test
uses the reflected samples' magnitudes, and two comments were corrected. One is left
as a follow-up: the column's scratch is six fine-grid rows per node of a chunk, three
to six times the chunk's own rows where the rings were O(K); per-thread rows would be
far smaller on large free-space chunks. It is charged in the memory ledger.

**Not run:** the full gate (Phase 6); Windows; a thread-count R test for the band
column, since it has no time chunking.

### Phase 4 — Kokkos CPU (2026-10-18)

Two slices on solprop-kokkos (4A, 4B), a physics review of 4A, a kernel-schema bump
and a compiler comparison.

**4A delivered:**
- **The licence.** CPU builds contract across statements (`-ffp-contract=fast` for GCC,
  `fast-honor-pragmas` for Clang and icpx); `fp.hpp`'s strict blocks keep the protected
  arithmetic as written: the phase increments, the small-argument phi-functions, every
  ionised-fraction form, and the multistep defects, residual terms and compensation. On
  GCC a strict helper carries `optimize("fp-contract=off")` and is therefore not
  inlined into a contracting caller. Device builds keep their compilers' defaults: by the
  maintainer's decision no strict blocks are added for CUDA or HIP, whose protected
  formulas are held by the qualification bounds instead.
- **The band column as passes** (the numba design of Phase 3), host team scans as the
  production path, and the team-width contradiction removed.
- **Reductions** (`fixed_order_reduce`, `accept_update`, `weighted_error`,
  `accept_and_score`) as `parallel_reduce`; the native GEMM held to the GEMM bound.
- **The physics review** found no blockers. Its should-fixes were taken: every fraction
  and compensation site was moved into strict blocks (the claim had outrun the code), a
  NaN exposure now propagates as in the reference, and the column's field bound is the
  §4.6 chain (`tests/plasma_bounds.py`). That chain is rigorous but loose, up to 3e-3 on
  the stress set against measured departures of at most 6.6e-14, because it cannot
  assume the cancellation in the currents' scans. Window defects are caught instead by
  an exact one-thread test (windowed against whole-row filter and scan, bit for bit), and
  two deliberate window mutations fail it.
- **Deviation from §7.5, by the maintainer's decision:** the band comparisons also carry
  an empirical regression guard, 1e-11 relative (150 times the largest measured
  departure), beside the derived bound and never in place of it. It is re-measured, not
  widened, when the column's arithmetic changes deliberately.

**Schema 28.** `StepDriver._stage_bundles` joined the frozen driver surface, and the
Kokkos adapter takes a trial's stage scalars in one call (about 4 % of the averaged
case). solprop and solprop-kokkos moved together.

**4B delivered:**
- **Stage fusion.** One kernel, `stage.entry`, runs a stage's phase increment, its
  combination, the zero padding of the fine spectrum and the previous stage's finish;
  a stage that projects no longer zero-fills first; a one-node, one-mode, one-component
  problem folds its synthesis and projection into the response and band columns as a
  compile-time variant. Launches per trial: 71 to 30 on the averaged case, 83 to 54 on
  the four-mode and CtC cases. Every fused product is the same arithmetic as before;
  each stage of a trial is held to its evaluation bound and to the regression guard.
- **The inverse propagator** is formed from its own `exp(-a)`, never a reciprocal.
- **The relaxed host projection** was dropped: about 1 % of the general-modal case.
- **The threaded-FFT threshold** stays at 8 192 samples; it can be overridden with
  `SOLPROP_KOKKOS_THREADED_FFT_MIN_SAMPLES` and is reported by `build_info()`.
- **A GCC finding.** The Linux timing showed the default (Adams) stepper 30 to 39 %
  slower on the averaged cases. The cause was 4A's review fix: the Adams compensation
  had become a run of strict helpers, which GCC would not inline into the contracting
  kernel (`adams.exponential_correct` 4.6 times slower). Each bin's compensation is now
  one strict function, inside which GCC inlines the helpers; the arithmetic is
  unchanged, and the cases are faster than before 4A.

**Compiler comparison** (Linux workstation, solprop-kokkos after 4A, Runge--Kutta;
`benchmarks/sep25/data/compilers-xeon/`):
- GCC 15.3 is within ±4 % of GCC 13.3 at one thread.
- Clang 18 and icpx 2025.3 are 2 to 17 % slower than GCC on most cases at one thread,
  but faster on the general-modal case, by 36 % and 54 %. The cause was one loop, the
  three-component real-form contraction of `general.respond` (about 60 % of the run on
  GCC): GCC declined to vectorise the triangular nest and later unrolled it through the
  stack, stalling every sample on a failed store-to-load forward. A GCC-only
  `#pragma GCC unroll` (empty for Clang, icpx and device compilers, so their code is
  unchanged) takes the case from 29.3 s to 11.3 s at one thread and 11.5 s to 10.1 s at
  18, ahead of Clang (16.6 s) and icpx (12.9 s); unrolling changes no rounding. At 18
  threads about 80 % of the samples are OpenMP idle-spin, a serial section left as a
  lead. At 18 threads, with one OpenMP
  runtime per build, the three are within a few per cent. (Clang and icpx beside the
  system's GNU-OpenMP OpenBLAS ran several times slower at 18 threads: two OpenMP
  runtimes in one process, not the compiler.)
- Vector `exp` and `log` from libmvec (Clang `-fveclib=libmvec -fno-math-errno`; GCC
  takes them only under fast-math) gain about 3 % over plain Clang and do not beat GCC,
  so they are not adopted, by the maintainer's rule.
- icpx is a reference point only, not a dependency.
- The comparison found two build bugs, both fixed: icpx failed under `-Werror` on the
  deliberate override of `-fp-model=precise`'s contraction, and a build that found no
  BLAS broke its compile line on a semicolon in the recorded BLAS selection.

**Work--precision (§7.3):** accepted for GNLSE and CtC on Kokkos
(`benchmarks/sep25/data/wp-phase4/`).

**Measured** (Linux workstation, exclusive, best of three; final Kokkos main against the
Phase 0 baseline; `phase4-final-xeon/` against `baseline-xeon/`, memory output):

| Case, Kokkos | RK t1 | RK t18 | default t1 | default t18 |
|---|---|---|---|---|
| He four modes | −49 % | −63 % | −41 % | −49 % |
| He averaged, band | −24 % | −43 % | −7 % | −20 % |
| He averaged, trapezoid | −16 % | −29 % | −4 % | −11 % |
| Ar blue shift | −38 % | −32 % | −33 % | −15 % |
| Reduced CtC | −22 % | −19 % | −20 % | −10 % |
| General modal | 0 % | −2 % | 0 % | −1 % |

The general-modal row predates the GCC fix above (−60 % at one thread on its own
measurement). These include Phase 2's host and output gains, which share the same solprop. No case is
slower beyond the Phase 0 noise. The averaged case with the default stepper is still no
faster on 18 threads than on one.

**Verification:** solprop-kokkos `scripts/check.sh` (tidy, sanitizer, 307 tests, the
distribution wheel); solprop's Kokkos-selected tests against each slice's build (874
passed); ruff, basedpyright, clang-format. The first CUDA builds of the merged tree found
two native tests that CUDA refuses (a host lambda and a device lambda inside a test
body); both were fixed.

**Not run:** the full gate (Phase 6); Windows.

### Phase 5 — GPU validation and optimisation (2026-10-18)

Three devices: the A40 for free qualification, and two sessions each on a rented H200
and MI325X. Rentals came to US$9.49 of the H200's US$40 cap and US$5.44 of the
MI325X's, each destroyed as soon as its evidence was pulled.

**Qualification.**
- **Every device lane passes** on the H200 (CUDA 12.8, HOPPER90) and the MI325X
  (ROCm 7.14, gfx942): the native tests, the `gpu`-marked tests of both repositories,
  the band column's derived bounds and its regression guard, and all of solprop's
  Kokkos-selected tests (873) once the device records were written. The A40 (CUDA
  12.8, AMPERE86) passed its lanes at the Phase 4 revision; its other failures were
  the ones below, since fixed. A later A40 rerun was blocked by the shared cluster's
  CPU quota.
- **The full Kokkos-selected suite had never run on a device.** Doing so found:
  - the spatial-accuracy and backend-equivalence records had no `kokkos_cuda` or
    `kokkos_hip` rows. They were written on each device with `SOLPROP_WRITE_RECORDS`
    (295 and 44 rows per device; no existing row moved);
  - two thread-count tests assumed a host OpenMP space, which a device build does not
    have; they now skip there;
  - a compiled-Adams test still required two adaptive runs to make identical
    decisions. On the H200 a rounding-level change of the error norm forked one, as
    §4.4 allows; forked runs are now compared as accuracy (class T) against the
    reference method at a tighter tolerance;
  - CUDA refused two new native tests (a host lambda and a device lambda in a test
    body).
- **Device builds keep their compilers' default contraction** (maintainer's decision);
  the protected formulas are held by these bounds instead of strict blocks.

**Baselines** (`benchmarks/sep16/device_profile.py`, one host thread; march seconds,
session 1 at the Phase 4 revision):

| Case | H200 | MI325X | A40 (pre-SEP base) |
|---|---|---|---|
| gnlse exponential 1e-9 | 0.053 | 0.045 | 0.075 |
| ctc exponential 1e-6 | 0.231 | 0.347 | 0.860 |
| taper exponential 1e-9 | 2.185 | 1.605 | 4.913 |
| wall exponential 1e-9 | 1.581 | 1.179 | 3.430 |
| radial filament exponential 1e-7 | 0.451 | 0.323 | 3.195 |

**The bottleneck and its fix.** On every device the CtC case's band plasma column took
76 to 88 % of GPU time: it ran one team per node (49 nodes against 132 streaming
multiprocessors on an H200) walking 65 536 samples in turn. The device column is now
tiled: each running integral is two kernels with one team per (row, tile) of 1 024
samples, a local filter and scan in team scratch, then a reduction of the earlier tiles'
totals added as an offset. The scan tree's depth stays inside the derived `scan_depth`,
so no bound changed. CPU builds keep the team column, which is faster there. Measured as
an interleaved A/B on each device (`SOLPROP_KOKKOS_BAND_TILES=0` against the default):

| CtC exponential 1e-6 | team column | tiled | GPU kernel time |
|---|---|---|---|
| H200 | 0.23 s | 0.10–0.13 s | 0.346 → 0.082 s |
| MI325X | 0.344 s | 0.087 s | 0.614 → 0.094 s |

The other cases are unchanged within noise. CtC's march is now bound by its host time.

**Leads not taken in this phase:** the coupled taper and wall cases are launch- and
copy-bound (110 000 launches per two marches on the H200, 49 000 small buffer copies on
the MI325X, host time above march time); the general modal case at 18 CPU threads is
mostly OpenMP idle-spin, a serial section; and a host-side fusion of each tile pass's
finish into the next local kernel would save two launches per call.

**Not run:** an A40 rerun at the final revision (cluster quota); ncu counters (not
permitted on either NVIDIA host).

### Phase 6 — Integration and closing (2026-10-18)

- **The lanes, once, on the final tree** (solprop main with solprop-kokkos main):
  - the gate (`poe check`): lint, format, types, docstrings, spelling, licences, the
    default lane (15 500 passed on Apple silicon) and the strict documentation build;
  - the slow lane (143 passed) and the coverage lane (100 %, 15 500 passed);
  - the Kokkos-selected slow lane (144 passed);
  - on the Linux workstation, the default lane (15 500 passed) and solprop-kokkos's
    tests (312 passed); solprop-kokkos `scripts/check.sh` on Apple silicon.
  - A full gate run by another SEP's session before Phase 6 found schema-28
    leftovers, docstring sections, a misspelling, an unresolved Sphinx reference, two
    identity checks that could not hold with numba's compiler disabled, and coverage
    gaps; all were fixed before these lanes.
- **Documents:** the user guide's reproducibility page, the CHANGELOG (three entries that
  still promised bitwise results restated to rounding), the roadmap (the band rule's
  cost item retired), and this SEP to Final and the archive.
- **Not run:** Windows, which only CI covers, after a push.

## Part III — History

- 2026-10-17: Drafted, at the maintainer's request, after a timing breakdown of the
  mode-averaged helium RDW case. The references were transcribed from the drafting
  agent's knowledge, and their DOIs and page numbers are to be checked against the
  publishers at review.
- 2026-10-17: Reviewed by GPT-6 (AI) and John Travers (human); review status set to
  `Reviewed`, proposal lifecycle retained as `Draft`. Human review confirmed exact
  structural checks, per-kernel compiler relaxation, deferral of flush-to-zero and
  approximate functions, and replay judged by overall accuracy. The review corrected
  the local/global error distinction, Adams replay assumptions, unit-roundoff factor,
  signed plasma scans and filter boundaries, and inverse-propagator reuse. It added
  versioned committed output lengths, dynamic metadata and memory requirements,
  independent references, preservation of regression witnesses and measurable
  work–precision criteria. Remaining engineering questions have named phases (§12).
- 2026-10-17: Source review checked the repository's conventions, backend accuracy
  ladder, step schedule and Adams replay, scalar-stage seam, plasma fraction and
  SEP 18 rules, Kokkos stage finish, and HDF5 append/resume behavior. Publisher or
  author records were consulted for references 1–10; the accessible summation paper,
  prefix-scan chapter and Goldberg reprint support the numerical corrections, with
  compiler and h5py documentation supporting implementation constraints. Full text
  of the books, IEEE standard and every cited research paper was not reviewed;
  inaccessible publisher pages are supplemented by the linked author records.
  The unlinked motivating profile and predicted speedups were not reproduced and
  remain explicitly unverified pending Phase 0.
- 2026-10-17: At the maintainer's request, expanded device work from qualification
  alone to an A40 validation/optimisation pass followed by H200 and MI325X optimisation
  passes. Added per-device baselines, profiling, measured candidate experiments,
  numerical requalification and cross-device regression checks; moved integration
  and closing to Phase 6. Review status remains `Reviewed`.
- 2026-10-17: Second pass by Opus 5.5 at the maintainer's request, to keep the review's
  corrections while removing bookkeeping that the project's loose-tolerance
  convergence practice does not need. Accuracy envelopes are now per case, from a
  tighter or stored reference with a factor fixed in advance, rather than a
  three-level refinement in every test. Rounding bounds are fixed a priori from
  operation counts, as the ladder does, rather than proved. Compiler relaxation stays
  per kernel, but its evidence is the R test, not an audit record. HDF5 is not
  preallocated, so the committed-length schema change is withdrawn (Q5); only
  cheaper appends remain. The execution-record change is in-memory caching only. The
  noise-statistics, work–precision, benchmark and failure-sensitivity rules are
  simplified to fixed factors and spot checks. The GPU campaign, the review
  decisions and the numerical corrections (signed scans, tap norm, inverse
  propagator, Adams replay, $u$ against $\epsilon$) are unchanged.
- 2026-10-17: Maintainer decisions after the second pass. Contraction is on by
  default for compiled kernels, with strict opt-outs for protected helpers;
  reassociation stays opt-in per kernel, because it can remove compensation terms
  (§5, Q2). HDF5 preallocation is restored with a Luna-style saved-plane count
  written after each plane and read by `resume` and the readers, without the
  heavier commit protocol of the first review (§6.4, Q5).
- 2026-10-17: Accepted by the maintainer (John Travers). Part I is frozen; deviations
  found during implementation go in the phase reports.
- 2026-10-17: Picked up for implementation (status In progress): Phase 0 by Opus 5.5
  (AI) on `feature/sep-0025-baseline`.
- 2026-10-18: Phase 2 (host and output) delivered by Opus 5.5 (AI) on
  `feature/sep-0025-host`, with the maintainer's decision on the statistics weight's
  tolerance (1e-12, floored at the mapping's noise).
- 2026-10-18: Phase 3 (`numpy_ref` and `numba_cpu`) delivered by Opus 5.5 (AI) on
  `feature/sep-0025-numba-a` and `-numba-b`, merged as one branch.
- 2026-10-18: Phase 4 (Kokkos CPU) delivered by Opus 5.5 (AI) on solprop-kokkos
  `feature/sep-0025-kokkos-a` and `-kokkos-b`, with kernel schema 28; the maintainer
  added the band column's regression guard and kept device builds' contraction defaults.
- 2026-10-18: Phase 5 (GPU validation and optimisation) by Opus 5.5 (AI): A40, H200
  and MI325X qualification, device records, and the tiled device band column; GPU
  rentals US$14.93 in all.
- 2026-10-18: Final. Phase 6 closed the SEP: the user guide's reproducibility page, the
  CHANGELOG and roadmap, and the full gate, slow, coverage and Kokkos lanes on the final
  tree. Archived.
