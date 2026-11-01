# SEP 10 — Low-rank Raman kernels along density paths

| | |
|---|---|
| **SEP** | 10 |
| **Title** | Low-rank Raman kernels along density paths |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5; GPT-6 (revision, 2026-10-22) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-09-29 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md); [SEP 1](sep-0001-raman-response.md), the common execution contract and exact shifted-line reference from phase 1 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add an explicitly requested low-rank spectral representation of the sampled causal
Raman kernel along a run's density path. At a stage, reconstruct the kernel from
fixed complex rows and real coefficients, avoiding repeated line sampling, damping
and kernel transforms. Extend SEP 1's common execution contract without replacing
its exact sampled or analytic representations. Keep exact FFT as the default;
positive tolerance selects compression. Qualify kernel construction, coefficient
paths, source error and observable convergence separately, with bounded construction
cost and NumPy, Numba and Kokkos CPU parity. Constant kernels bypass SVD.

### 2. Motivation and scope

**Draft; implementation and production qualification have not started.** The current
review uses solprop `20a2bafa` and optmat 0.8.0. SEP 3 is Final; its material migration
is no longer prospective. The original prototype rank/timing observations below
describe optmat's sampled experiment, not the cost or accuracy of this implementation.

With per-line widths, rebuilding an effective kernel can dominate its convolution
on a density gradient. N2 has 106 lines / 43 damping groups, dry air 170 / 82 at
the documented reference state. H2 additionally shifts its Q lines with density.
SEP 1 supplies the exact stage-dependent sampled kernel; this SEP compresses that
family. Compression is a performance/accuracy trade-off, not a way to manufacture
missing material data or a prerequisite for representing a shifted line exactly.

Optmat's [response chapter][optmat-response] reports these exploratory ranks, using
64 densities and maximum spectral error relative to the sampled family's maximum:

| Gas | Density (amagat) | Window and spacing | Rank at $10^{-4}$ | Rank at $10^{-6}$ |
|---|---|---|---|---|
| N2 | 0.5–2 | 131 ps, 4 fs | 4 | 5 |
| N2 | 0.1–10 | 131 ps, 4 fs | 6 | 9 |
| O2, dry air | 0.1–10 | 131 ps, 4 fs | 6 | 8 |
| H2 | 1–30 | 66 ps, 2 fs | 5 | 6 |
| D2 | 1–30 | 66 ps, 2 fs | 4 | 5 |

These ranges are numerical compression experiments, not extensions of the independent
Lorentzian line model's physical validity. Optmat reports N2 Q-branch line overlap
at densities well below 10 amagat. A good low-rank reconstruction does not qualify
the independent-line physics there. Translation preserves all material warnings,
statuses and range records.

Included are fixed-temperature, fixed-composition density paths; ordinary scalar and
declared tensor responses; and constant regional silica kernels through their exact
static representation. Supplied models use their declared sampled/dephasing laws;
they need not expose analytic poles. Excluded are local thermal populations,
arbitrary varying composition, new collision laws, line mixing, pole fitting,
optical-frequency coupling, population dynamics and GPU qualification.

### 3. Context and compatibility

See the [Raman data](../../guide/media/raman-data.md),
[numerics](../../guide/media/raman-numerics.md), [kernel contract](../kernel-contract.md)
and SEP 1 §§3–6. Current `RamanCoefficients` carry undamped sampled rows, rate paths,
fraction, density, drive form and tensor weights. Compatible contributions are
combined before the drive transforms. Regional silica has a distinct table and
prefactor, not a gas-density leaf.

SEP 1 makes the model and execution representation explicit: `sampled`, `analytic`
and this SEP's `spectral` extension. Host analytic data remain available, and the
backend receives poles when executing an analytic payload. Low-rank rows cannot
be inverted to recover microscopic poles. Exact FFT and recurrence remain supported
representations of the selected response; this SEP does not delete them.

[SEP 11](sep-0011-finite-density-gas-index.md) changes how a consistent state is
obtained from pressure, temperature and composition. Compression consumes its
number-density path, without assuming an ideal or partial-pressure EOS. The same
path supplies linear optics, Raman amplitude, collision widths and line shifts.
Changing an EOS or material selection invalidates translation/cache identity.
For optmat mixtures, strengths already contain mole fractions and the response uses
the summed density of its labels. Do not multiply fractions twice. Supplied component
amplitudes and collision-density paths retain their separate meanings.

[SEP 6](archive/sep-0006-dispersive-nonlinearity.md) owns electronic dispersion.
Its filters never act on Raman's drive or emitted source. The new
[Maxwell–Bloch SEP](sep-0032-raman-maxwell-bloch.md) owns evolving molecular states;
its nonlinear material solver is not compressed by this linear kernel representation.

### 4. Theory and construction

#### One finite, sampled operator

Let $N=\mathtt{n\_to}$, $L=2N$ and $K=N+1$. At material state $n$ (number
density, m$^{-3}$), let $h_\kappa[j;n]$, $0\leq j<L$, be the exact sampled
kernel for tensor family $\kappa$. It includes the chosen endpoint weights, lag
taper where applicable, and zero padding for $j\geq N$. Its internal-convention
spectrum is

$$
H_\kappa[k;n]=\Delta\,\operatorname{rfft}_L(h_\kappa[\cdot;n])_k.
$$

For a gas, $h$ has units m$^5$ V$^{-2}$ s$^{-1}$ and $H$ has m$^5$ V$^{-2}$.
Density and the overall response fraction are applied once, outside this kernel,
as in the exact path. Regional silica retains its existing units and prefactor.
Do not replace this sampled transform by optmat's continuous infinite-time
`spectrum`: the latter uses the optics sign and a different window/quadrature.
It is a useful analytic limit only after accounting for those differences.

#### Density coverage, including vacuum

Construct over the resolved density range of the run's path, including endpoints
and zero when reached. Use piecewise Chebyshev sampling in density itself; logarithmic
density is not the required coordinate. Split at known linewidth-law transition
densities, in particular the capped Dicke switch, and at discontinuities of the
declared material/path law. Current optmat widths are finite at zero. Do not
reintroduce the old positive floor or extrapolate from the smallest positive sample.

Build a global basis for each nonzero medium/tensor family from samples over those
pieces. Enrich the density mesh deterministically at independent validation points
with the largest normalised residual, bisecting pieces as needed. Revisit validation
after enrichment; fitting a failing point is not itself an independent check.
The existing z-path interpolation separately respects its path breakpoints.
Negative or non-finite density is an input error. A stage outside the constructed
range is refused, never silently extrapolated.

#### Real projection and complex rows

For a half-spectrum $H$, define the real stacked vector

$$
q(H)=\begin{bmatrix}\operatorname{Re}H\\\operatorname{Im}H\end{bmatrix}
\in\mathbb R^{2K},\qquad
\mathsf A=[q(H(n_1)),\ldots,q(H(n_M))].
$$

Take a thin real SVD $\mathsf A=\mathsf U\Sigma\mathsf V^T$. For the first
$r$ orthonormal columns, define complex rows
$U_j[k]=\mathsf U_{k,j}+i\mathsf U_{K+k,j}$ and real coefficients

$$
c_j(n)=\mathsf U_{:,j}^{T}q(H(n)),\qquad
\widehat H(k;n)=\sum_{j=1}^r c_j(n)U_j[k].
$$

The projection is on the stacked real vector, not a transpose of a complex spectrum.
The SVD minimises the sampled real-matrix 2/Frobenius error [2]; it does not by itself
minimise the complex max norm used for acceptance. Real coefficient paths are fitted
along z by the existing `PiecewiseChebyshev` machinery, and acceptance checks the
reconstructed spectrum rather than each coefficient in isolation.

Process each independent tensor family separately, retaining its contraction map.
Do not combine unrelated isotropic and anisotropic responses merely to save rank.
Tensor row signs are algebraic; they are not population probabilities.

#### Causality, endpoints and zero families

In exact arithmetic, the SVD basis is a real linear combination of the causal input
histories, so it has the same support and homogeneous endpoint constraints. In
floating point, remove the forbidden padded tail in each inverse-transformed basis
history and preserve any model-mandated zero at lag zero; keep the nonzero silica
half-endpoint as part of its sampled shape. Transform those real histories back,
enforce real DC/Nyquist entries, and re-orthonormalise if necessary. Run all rank,
projection and validation checks on these final stored rows. Numerical inverse-FFT
tail residuals are bounded at rounding level, not claimed bitwise zero.

A family identically zero by its model is omitted (rank zero) without a division by
zero or an arbitrary normalisation floor. A fixed nonzero kernel is stored as its
exact sampled spectrum with coefficient one, no SVD and zero compression error.
This includes a constant density at fixed temperature/composition and the current
regional silica shapes. Constant pressure alone does not imply constant density
when temperature changes. An input identically at vacuum has zero macroscopic source.

#### Error allocation

For each nonzero family record a fixed scale $S_\kappa$, the largest
$\|H_\kappa(n)\|_\infty$ on the construction and independent validation states.
Report this scale and its units. It is a family-wide absolute error normalisation,
not a promise of relative accuracy near a spectral zero or at a much weaker state.
Update it and repeat acceptance when the density mesh grows.

For requested $0<\epsilon<1$, allocate half to basis truncation and half to
coefficient-path interpolation:

$$
e_{\rm basis}(n)=\|H(n)-\sum_jc_j(n)U_j\|_\infty
\leq\tfrac12\epsilon S_\kappa,
$$

$$
e_{\rm path}(z)=\|\sum_j[c_j(n(z))-\widehat c_j(z)]U_j\|_\infty
\leq\tfrac12\epsilon S_\kappa.
$$

Choose the smallest tested rank satisfying the basis criterion, then independently
check both errors and their combined residual against the exact sampled operator at
off-node z values, endpoints and the density-law transitions. Existing source-table
errors and transform/assembly roundoff are reported separately. If roundoff prevents
establishing the requested budget, fail with the achieved floor; never relabel it as
successful zero-tolerance compression.

The singular-value tail is a diagnostic, not the acceptance test. Finite off-node
testing supplies a measured validation envelope, not a proof of the maximum over
every real density. Say so in the manifest/manual. Adversarial tests and refinement
studies assess coverage; a model with an analytic interval error bound may additionally
report that bound without replacing the general validation requirements.

#### From kernel error to source error

Pad a drive $d$ with zeros to length $L$, and crop the first $N$ outputs after
convolution. For the raw forward DFT and its $1/L$ inverse,

$$
r=C\,F_L^{-1}\!\left[H\,F_Ld\right],\qquad
\|\delta r\|_2\leq\|\delta H\|_\infty\|d\|_2.
$$

Here $H$ is understood with its Hermitian completion; $C$ is a norm-nonincreasing
crop. The transform factors cancel because the DFT is unitary up to $\sqrt L$.
The temporal weight $\Delta$ is already in $H$. For one scalar gas contribution,

$$
\|\delta P_R\|_2\leq
\epsilon_0|\rho f|\,\|F\|_\infty
\|\delta H\|_\infty\|d\|_2.
$$

Sum the corresponding bounds with absolute tensor/contraction weights for multiple
families, and include projection and propagation prefactors when bounding the optical
RHS. This is a local absolute source bound. It neither bounds global propagation
error nor ensures relative accuracy for a weak selectively driven line. Compare
gain/phase in the physical bands of interest and tighten $\epsilon$ until short-run
observables meet their independently specified accuracy envelope. Optical z
tolerances alone cannot diagnose compression bias.

#### What full rank does not mean

Retaining every column of a finite training matrix reproduces its training columns
up to roundoff. It does not reproduce a continuous density family outside their
span, and coefficient interpolation adds another error. Therefore tolerance zero
selects SEP 1's exact sampled/analytic FFT evaluator, never a "full sampled family"
SVD. The small counterexample in Part III makes this distinction explicit.

### 5. Proposed interface

Use SEP 1's settings, forwarded by convenience constructors:

```text
# Default: no compression.
settings = SimulationSettings(..., raman_evaluator="fft",
                              raman_rank_tolerance=0.0)
# Explicit approximation request; this is not an accuracy recommendation.
settings = SimulationSettings(..., raman_evaluator="fft",
                              raman_rank_tolerance=1e-4)
```

Finite $0<\epsilon<1$ requests compression. Negative/non-finite values and values
at least one are refused. Positive tolerance with `"recursive"` is refused.
No `auto` policy or automatic $10^{-4}$ default is introduced. Exact constant and
zero cases satisfy a positive request without approximation and record that fact.

Record per-family rank, scale, construction/validation coverage, basis/path/combined
errors, interpolation error, limits, density interval, boundary policy and model
identity. Cache identity is built from inputs, grid, taper, algorithm version,
tolerance and construction policy, not floating-point singular vectors. The actual
manifest records the realised rank and diagnostics even when equivalent builds
choose different bases or a different rank close to a threshold.

Preserve the explicit `raman_shift="reference-density"` model when requested;
otherwise H2's exact shifted family is compressed. A supported approximate request
that cannot meet tolerance fails clearly, reporting achieved error and limiting
resource and suggesting exact FFT. It never silently relaxes tolerance, drops lines
or changes backend.

### 6. Implementation strategies

#### Payload and backend

Add the `spectral` variant of SEP 1's frozen leaf, with complex128 rows
`(n_rows, n_to_doubled // 2 + 1)`, matching real coefficient tables, tensor-family
mappings and an approximation record. Preserve the common leaf's fraction, label,
amplitude density, drive and tensor capability. Stage bundles evaluate coefficients;
the backend forms their real-times-complex sum and memoises it when those values
change. Preserve current aggregation and shared drive transforms across compatible
leaves. There is no kernel exponential or kernel FFT in a compressed rebuild.

The NumPy phase bumps the then-current schema, frozen leaf/stage literals, contract
and changelog, with native consumer compatibility and named refusals until support
lands. The consumer follows the repository layout then in force, including SEP 29.
Arrays cross the seam; host material objects and SVD work do not. Numba uses shared
no-allocation bodies, Kokkos CPU equivalent native assembly. GPU qualification is
separate. No dependency is added.

#### Bounded construction and memory

Build one family at a time and stream validation spectra. The initial implementation
uses finite translation limits: at most 257 retained training states per family,
rank 64, eight enrichment rounds and 512 MiB estimated peak construction workspace,
including the snapshot matrix, SVD copies/workspace, basis and validation scratch.
Check the estimate before allocation; the first reached limit stops construction
with a named error. These are engineering resource limits, not physical accuracy
claims. Phase 1 reports their adequacy and any reviewed changes; they and the
algorithm version are recorded. Coefficient paths retain the existing finite
`PiecewiseChebyshev` degree/interval limits and fail normally if unconverged.

For a doubled history of $2N$ samples, one real row costs $16N$ bytes and one
complex half-spectrum row $16(N+1)$ bytes. Their sizes are nearly equal, not a factor
of two apart. Storage savings come from fewer rows; coefficient tables, construction
peak, provider workspace and per-stage assembly buffers must be counted. A rank-one
static path skips all SVD workspace. Do not claim a memory or speed win before
measuring translation and complete stages.

#### Basis freedom and reproducibility

A largest-component-positive convention fixes the sign of an isolated singular
vector, not rotations within a repeated or nearly repeated singular subspace.
Do not compare arbitrary SVD columns across BLAS builds or use them as identity
metadata. Compare reconstructed kernels and source effects within their separate
approximation/rounding budgets; compare projectors only for equivalently selected,
spectrally separated subspaces. Near a rank threshold, equivalent validated runs
may select different ranks. All backends in one translated problem receive the
same rows and are compared using rounding bounds for that fixed representation.

### 7. Verification and validation

| Case | Reference and tolerance basis |
|---|---|
| Static/zero families | Exact sampled FFT; zero source/support metadata structurally exact, nonzero transforms bounded by roundoff; no SVD |
| Off-node reconstruction | Direct sampled optmat or supplied kernel, with the same sign, temporal weight, endpoint and taper; declared basis/path/combined budgets |
| Vacuum and linewidth switches | Direct finite optmat law at zero and on both sides of each switch; exact zero macroscopic source and no logarithmic floor |
| H2 frequency gradients | Independent exact line sampling and FFT, not frozen reference-density rows; include phase-sensitive delayed probes |
| Causality/endpoints | Impulses at the first/last sample, zero-padded tail, signed tensor drives and Lin–Agrawal half-endpoint; direct causal sums and rounding leakage bounds |
| Tensor and labels | Direct Hellwarth contractions, rotation covariance, split/reordered labels and one-time mixture fraction; derived summation bounds |
| Conditioning and weak lines | Cancellation, very small/nonzero families and selectively resonant beats; absolute source bound and observable-specific gain/phase error |
| SVD ambiguity | Repeated/clustered singular values and cutoffs; reconstruction, invariant subspace where defined, input-based cache identity |
| Failure | Snapshot/rank/round/workspace and path limits, too-small tolerances, out-of-range state and unsupported backend; clear refusal with no relaxed approximation |
| Backend | One source, one step, eight steps and short runs; fixed-representation operation/transform bounds, batch/tail/chunk/status checks |
| SEP 6 coexistence | Combined stages against separately assembled sources using the bare Raman field; normalisation, supported harmonic sectors and additive optical work as in SEP 1 §7 |

Use N2, O2, H2, D2, dry air, custom sampled responses and constant regional silica.
For propagated error, use exact FFT references for both unshifted N2 and shifted H2,
refine tolerance through at least three levels, and independently refine z and time
discretisation. Assess specified observables against a tighter reference; reducing
$\epsilon$ is evidence of convergence, not a theorem that error halves monotonically
through a rank change. Compression comparisons include approximation error;
backend comparisons for identical compressed arrays use rounding error alone.

Benchmark translation including SVD and validation, kernel rebuild, complete source
stage and about ten propagation steps at fixed accuracy and memory policy. Include
constant density, broad density range, a linewidth switch and long-lived shifted
coherence, and report grid/window, actual ranks, work and memory. The old 7 ms versus
0.2 ms optmat observation is motivation, not a target or an extrapolated speedup.

### 8. Documentation

Extend the Raman workflow/numerical/data pages with the opt-in setting, the exact
alternative, error/coverage semantics, resolved rank, failures and material validity.
Add a short density-gradient comparison example with tolerance convergence. Update
the common kernel and stage contract when implemented. Record complete-stage and
translation measurements in the benchmark ledger and notable changes in the
changelog. At Final, link manual pages to the archived SEP.

### 9. Compatibility and impact

Default exact FFT use remains unchanged apart from SEP 1's independently specified
shifted-line capability. Positive tolerance is a new recorded numerical approximation;
its tolerance, material/state/grid/boundary identities and algorithm policy enter
resume compatibility. Changing these cannot silently resume the same configured
source. Exact evaluator choice remains execution metadata. Old files remain readable;
additive provenance fields use the output schema policy if needed, without changing
saved field shapes or Fourier conventions.

No future EOS, Maxwell–Bloch or broadband Raman-coupling implementation is required
to finish this SEP. Their state/response owners remain separate. An exact response
with unavailable collision data is still unavailable after compression.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Exact sampled/analytic FFT | Retained default and oracle; potentially expensive changing-kernel preparation |
| SEP 1 recurrence | Exact sampled alternative when poles are available; cost scales with lines, independent of low-rank approximation |
| Direct spectral interpolation in density | Avoids SVD but may need more rows; a useful independent construction comparison |
| Full-rank finite training matrix as exact | Rejected: cannot span a general continuous density family |
| Automatic tolerance/default switch | Rejected for this scope: kernel error is not global propagation accuracy |
| Merge or prune physical lines | Outside scope; compression does not change the declared line list |

### 11. Phases

Implementation follows SEP 1's merged common-contract/NumPy foundation. Branches
use `feature/sep-0010-low-rank-raman-<phase>`.

- **Phase 1 — NumPy.** Spectral variant, schema/consumer checks, bounded construction,
  static/zero bypass, error records, explicit selectors, exact shifted-H2 comparisons,
  manual and independent local/source tests. Report resource-limit adequacy.
- **Phase 2 — Numba CPU.** Preallocated assembly and stage invalidation, parity,
  interpreted-body coverage and measured stage costs.
- **Phase 3 — Kokkos CPU and closure.** Native parity and ledger, final observable
  convergence, combined accuracy/cost/provenance evidence and documentation.

During development slices run the repository's fast checks and affected tests.
At the end of the SEP, run the full gate, slow lanes and coverage lane once on the
final tree, or take them from CI. Rerun failing modules after failures. Documentation-only
revisions use the documentation gate. Each phase reports actual results under Part II;
GPU qualification stays separate and unsupported device use is refused.

### 12. Decisions and remaining questions

The earlier open choices are resolved for this Draft: exact is the default, positive
tolerance is explicit, vacuum-aware piecewise density sampling is selected, and
constant supplied models bypass SVD. Maintainer scope directions are recorded below.

| Question | Disposition |
|---|---|
| Coverage and resource limits for difficult supplied models | Phase 1 measures the declared finite construction policy; failure is explicit. No universal interval certificate is promised. |
| Which ranks/tolerances give useful cost at achieved accuracy? | Phases 1–3 measure per case; no automatic policy or universal recommended tolerance is required. |
| GPU assembly/transfer costs | Separate device qualification after CPU closure. |

Human review and explicit acceptance remain outstanding. No pending physical-model
choice is required to implement the selected fixed-state sampled-kernel approximation.

### 13. References

1. optmat 0.8.0, [Raman response functions][optmat-response] and
   [line parameters](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman.md).
   The rank/window table is a sampled numerical experiment, not a physical-range
   qualification or a verified continuous error bound.
2. C. Eckart and G. Young, "The approximation of one matrix by another of lower
   rank," *Psychometrika* **1**, 211–218 (1936).
   [doi:10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

[optmat-response]: https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman_response.md

## Part II — Implementation record

No implementation phases have started.

## Part III — History

- 2026-09-29: Drafted from the low-rank design split out of SEP 3.
- 2026-10-17: Restated reproducibility for SEP 25, retaining exact input-based identities.
- 2026-10-22: Reviewed with SEP 1 and revised to use its execution contract, exact
  default, explicit approximation and current material/state semantics.
- 2026-11-01: Tracking issue [#120](https://github.com/LupoLab/solprop/issues/120).

### Maintainer directions — 2026-10-22

John Travers requested a joint acceptance-readiness review, retained the current lag
taper, chose explicit approximation tolerance with exact FFT default, adopted the
mixture-EOS direction for SEP 11, and kept broadband coupling separate. The completed
AI review leaves this Draft **In review**, pending his review and acceptance.

### AI review — 2026-10-22 — GPT-6

Review of the entire original Part I against SEP 1 and the current implementation.

1. **R1 (P1), §§3, 6 — Incompatible seam replacement.** Replacing all leaves with
   spectra while keeping poles only on the host prevents a backend recurrence.
   Extend a shared tagged contract and retain exact sampled/analytic evaluation.
2. **R2 (P1), §§4–5 — Full rank is not an exact continuous family.** A finite
   training span and interpolated coefficients need not reproduce unseen density.
   Zero tolerance must choose exact FFT. Part III's counterexample establishes this.
3. **R3 (P1), §4 — The proposed density floor is stale.** Current optmat widths
   are finite at vacuum and have capped-law transitions; logarithmic sampling and
   a fabricated floor omit the state the run may reach. Include zero and resolve
   the actual piecewise law.
4. **R4 (P1), §§4, 7 — Accuracy contracts are incomplete.** Clarify real-stacked
   projection, scale, combined compression/interpolation budget, support/endpoints
   and operator/source error. Sampled validation is not an interval proof and a
   kernel tolerance is not global propagation error or relative weak-line gain.
5. **R5 (P1), §§4–6 — SVD columns are not reproducible identities.** A sign rule
   cannot fix degenerate-subspace rotations. Compare kernels/projectors where
   defined, and bound construction resources with explicit failure.
6. **R6 (P2), §§4, 6 — Static/zero and memory claims need correction.** A zero
   family has rank zero, constant pressure need not imply constant density, and a
   complex half-spectrum row costs approximately the same as a real doubled row.
7. **R7 (P1), §§2–3, 7 — Material and composition boundaries are missing.** Exact
   shifted H2 needs a direct reference, mixture strengths contain fractions already,
   optmat's rank experiments cross some physical validity limits, and SEP 6 filters
   must not contaminate Raman. Record SEP 11 state compatibility and limits.
8. **R8 (P2), §§5, 11–12 — Defaults and completion policy conflict.** The proposed
   automatic $10^{-4}$ changes source accuracy without an observable budget; per-phase
   full-lane reruns conflict with current repository policy. Apply the agreed exact
   default and current testing policy.

### Response to review — 2026-10-22

- **R1:** §§3 and 6 extend SEP 1's contract; phase 1 is an explicit prerequisite.
- **R2:** §§4–5 specify exact FFT for zero tolerance; the counterexample is retained.
- **R3:** §4 samples density pieces with vacuum and linewidth switches, without a floor.
- **R4:** §4 defines the real projection, final-row validation, separate budgets and
  DFT operator/source bound. §7 adds exact shifted-H2 and observable refinement.
- **R5:** §§5–6 use input identities, reconstruction/subspace comparisons and bounded
  construction; rank threshold variations are recorded rather than hidden.
- **R6:** §§4 and 6 add static/zero paths and correct byte accounting.
- **R7:** §§2–3 and 7 distinguish material validity and mixture ownership from numerical
  compression, with SEP 6/11 coexistence and independent exact references.
- **R8:** §§5 and 11–12 implement the maintainer's explicit-tolerance direction and
  slice/final testing policy. No automatic/default policy remains to settle.

### Review probes and source audit — 2026-10-22

For $j=0,\ldots,255$, take the dimensionless causal family

$$
h_j(n)=e^{-0.01nj}\sin[(0.2+0.07n)j].
$$

A two-column time-domain matrix at $n=0.2,0.8$, retaining both thin-SVD columns, reconstructs
$h(0.5)$ with a maximum error **0.731160** relative to its own maximum. The two
training columns themselves are retained at roundoff. This is a counterexample
to the old exactness claim, not an estimate of physical gas compression error.
At $N=256$, float64 storage for $2N$ real samples is 4096 bytes; complex128 storage
for $N+1$ bins is 4112 bytes. This checks the corrected row memory accounting.

The current optmat response/width/shift APIs, material ranges and prototype table
were inspected; the old prototype timings were not rerun and no production SVD
or propagation benchmark is claimed. The SVD projection and DFT norm bound above
are derived explicitly. The bibliography of [2] is retained; its original text was
not newly inspected for coefficient/data transcription, of which there is none.
Documentation checks passed: `git diff --check`, codespell on all seven changed
documentation files, `uv run pytest tests/test_seps.py` (231 passed, two expected
process-SEP skips), and `uv run poe docs` with warnings treated as errors. A missing
roadmap subsection anchor from the first build was corrected before the successful
build. Full numerical gates are not required for this documentation-only revision.

All R1–R8 findings are addressed above. The joint review found no remaining design
contradiction with SEP 1; the finite construction limits and useful ranks remain
implementation measurements, not assumed successes. Human review and acceptance
remain outstanding. No production compression, backend parity or propagation-error
qualification is claimed by these review probes.
