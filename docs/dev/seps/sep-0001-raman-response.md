# SEP 1 — Efficient causal Raman evaluation

| | |
|---|---|
| **SEP** | 1 |
| **Title** | Efficient causal Raman evaluation |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; GPT-6 (revision, 2026-10-22) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-09-26 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Evaluate the existing causal sampled Raman operator by exact oscillator recurrences
where analytic structure is available, including its current lag taper and endpoint
weights. Keep exact padded FFT evaluation as the default and as the general alternative.
Define the common execution contract that [SEP 10](sep-0010-low-rank-raman.md)
extends with explicitly requested low-rank spectral approximation. Density-shifted
lines have an exact sampled FFT reference as well as a recurrence representation.
Success means independent causal and physical checks, bounded arithmetic error,
NumPy/Numba/Kokkos CPU parity and measured complete-stage costs and memory.
Population dynamics belong to [SEP 32](sep-0032-raman-maxwell-bloch.md).

### 2. Motivation and scope

**Draft; implementation and numerical qualification have not started.** The 2026-10-22
review checked the current optmat-backed code, not the pre-SEP-3 molecular factories
alone. The exploratory calculations recorded in Part III are algebra and design
checks, not qualification or evidence of a production speedup.

Long molecular histories, many transverse nodes and pressure gradients can make
Raman transforms or kernel rebuilding expensive. Recurrence costs grow with the
number of individual poles, whereas FFT application sees already combined kernels.
Neither method wins universally. Evaluation must preserve the physical response;
an optimisation must not replace a thermal ladder with one effective transition.

Included are all lines retained by the selected material model, supplied analytic
oscillators, ordinary scalar/tensor histories and the regional Lin–Agrawal shapes.
No further pruning is introduced. The shared representation handles density-varying
frequencies, including optmat's H2 Q-line shifts, and therefore lifts the exact-path
refusal once its backend is qualified. The explicitly selected reference-density
approximation remains a distinct recorded material choice.

Hollenbeck–Cantrell and arbitrary sampled kernels retain FFT evaluation. Continuous
drive interpolation, pole fitting/pruning, a coarser Raman grid, removal of the lag
taper and finite-support truncation remain separate work. Optical-frequency-dependent
coupling and Maxwell–Bloch dynamics are not prerequisites for this SEP or SEP 10.
The initial selector has no `auto` value; a later selection policy needs measured
crossover evidence and its own review.

### 3. Current implementation and related proposals

The baseline is solprop `20a2bafa`, with optmat 0.8.0. See the
[Raman manual](../../guide/media/raman.md), [data and validity](../../guide/media/raman-data.md),
[conventions](../conventions.md), [architecture](../architecture.md) and
[kernel contract](../kernel-contract.md).

{mod}`solprop.nonlinear.optmat_raman` retains optmat's source record and maps damping
groups and tensor families onto `RamanGroupLines` and `GroupDephasing`.
{mod}`solprop.translate.raman` samples tapered undamped rows;
{mod}`solprop.translate.responses` constructs `RamanCoefficients` and the separate
`RegionalResponseTables`. Their consumers are the NumPy and Numba response passes
and solprop-kokkos's ordinary and general-modal stages. Physics objects never enter
the backend.

At 293.15 K and 800 nm the current built-in counts are 20 H2 lines / 3 damping
groups / 4 tensor rows, 106 N2 lines / 43 groups / 72 rows, and 170 dry-air lines /
82 groups / 126 rows. Counts are material selections, not backend constants. A
group contains several frequencies; equal damping does not collapse it to one pole.
The older supplied `RotationalLines` factories are still supported but do not
describe the default built-in gas route.

Optmat's capped Dicke law is finite at vacuum. Its frequency and decay laws, rather
than a fabricated positive density floor, define the response at zero density;
the macroscopic source is exactly zero there. A supplied legacy singular dephasing
retains its existing finite translation convention and must be distinguished from
the built-in law.

For an optmat mixture, line strengths already contain mole fractions and one
composition response uses the summed density of its labels. Do not multiply those
fractions again. Supplied species responses can instead have a component amplitude
density distinct from the density used for collisions. The new records preserve
both meanings. Fixed reference temperature and composition remain the present
admission policy; neither SEP 1 nor SEP 10 silently adds local thermal populations
or arbitrary changing collision partners.

| Proposal | Relationship |
|---|---|
| SEP 3 | Implemented material ownership, data identities and Raman tensor mapping |
| [SEP 6](archive/sep-0006-dispersive-nonlinearity.md) | Electronic response composition and harmonic-sector checks; no Raman dispersion model |
| SEP 10 | Extends this SEP's execution contract with approximate spectral bases |
| [SEP 11](sep-0011-finite-density-gas-index.md) | Supplies consistent density paths from the selected EOS; is not a dependency of either evaluator |
| SEP 32 | Separate nonlinear molecular state model; its weak limit can use these exact kernel references |

#### Padding, drives and local histories

Let $N=\mathtt{n\_to}$ and $\Delta=\mathtt{dt\_fine}$. Optical oversampling
resolves nonlinear products; padding a length-$N$ drive and kernel to $2N$ makes
their FFT product a linear causal convolution. Removing padding does not justify
coarsening the optical grid. At one material state, the scalar operator is

$$
r_n=\Delta\sum_{m=0}^n h_{n-m}d_m,\qquad
P_{R,n}=\epsilon_0\rho F_n r_n.
$$

The real drive is $E^2$ on a field grid retaining harmonic beats, $|E_a|^2/2$
with those beats removed, and $|A|^2/2$ on an envelope grid. $F$ is the real
field or complex envelope respectively. The factor one half belongs to the drive.
The real Raman FFT pair is used on both grids. A length-$N$ circular convolution
adds $\Delta\sum_{m=n+1}^{N-1}h_{N+n-m}d_m$, an unphysical response to future samples.

Histories belong to synthesised physical fields at nodes, not to separate modal
amplitudes. Ordinary two-component tensors use trace/difference/off-diagonal drives;
regional three-component silica has six symmetric-pair histories and two delayed
channels. Modal interference, tensor weights and endpoint conventions are preserved.

Existing code already caches sampled rows and effective spectra, combines compatible
species/tensor contributions, uses planned transforms and bounded chunks, and handles
tail chunks at their actual width. Those are the baseline, not new savings.

### 4. Theory

All quantities use SI and the package's conventions. Gas $h$ and its line strengths
$K_\ell$ have units m$^5$ V$^{-2}$ s$^{-1}$, $d$ has V$^2$/m$^2$ and $r$ has m$^3$.
Silica keeps its own normalised shapes and prefactors. Frequencies are rad/s and decay
rates s$^{-1}$. Molecular variables are real float64 or complex128; local fields
are V/m after synthesis, never modal square-root-watt amplitudes treated as fields.
Every evaluation starts from zero temporal prehistory at its own spatial location.

#### Discrete recurrence: no ODE truncation error

For the untapered gas model write

$$
 h_j=\sum_{\ell=1}^{M}K_\ell
       e^{-\gamma_\ell j\Delta}\sin(\Omega_\ell j\Delta),
 \qquad j\geq0.
$$

$M$ counts individual lines. At a given propagation stage all line parameters
are fixed along the temporal sweep. Define

$$
 a_\ell=e^{(-\gamma_\ell+i\Omega_\ell)\Delta},\qquad
 s_{\ell,-1}=0,\qquad
 s_{\ell,n}=a_\ell s_{\ell,n-1}+d_n.
$$

Expanding the recurrence gives

$$
 s_{\ell,n}=\sum_{m=0}^{n}a_\ell^{n-m}d_m,
 \qquad
 \boxed{r_n=\Delta\sum_\ell K_\ell\operatorname{Im}s_{\ell,n}.}
$$

This is algebraically the **same sampled causal sum** as the padded FFT for the
untapered kernel. It is not forward Euler, not an approximate integration of
the molecular period, and not an infinite-history approximation. The current
sample enters the real part and makes no immediate imaginary contribution,
which correctly reproduces $h(0)=0$ for a sine kernel.

There is one complex multiply and one real injection per line per sample.
Sines, cosines and exponentials are calculated once per line and stage, rather
than per sample. The remaining line sum can use a fixed, documented order.
Overall application work is $O(MN)$ per drive, with $O(M)$ streaming state.

An equivalent second-order real filter for one line is

$$
 r_n=2b\cos\theta\,r_{n-1}-b^2r_{n-2}
      +\Delta K b\sin\theta\,d_{n-1},
 \quad b=e^{-\gamma\Delta},\quad\theta=\Omega\Delta,
$$

with zero prehistory. This gives a compact implementation, but the complex
state/real rotation form is the safer starting point when $\theta$ is small: it
avoids relying on cancellation between nearly equal second-order coefficients.
Both require error analysis for long, weakly damped histories. Repeated
multiplication introduces phase and magnitude drift, and roundoff in $a$ is
accumulated over time; stable poles alone do not certify sufficient accuracy.

The derivation is independent of an FFT convention because it is a causal
time-domain identity. When checking its spectrum against solprop, the internal
negative-exponent transform must still be used: a positive Raman line has
negative imaginary susceptibility near positive resonance and gives Stokes
gain. No extra optics-convention conjugation belongs inside this calculation.

#### The current Planck taper is a real obstacle, with a useful exact remedy

The molecular translation path actually samples $w_jh_j$, where $w$ is one
until the long-delay collar and then falls to zero. A Planck taper times an
exponential sine does not have the same small, constant-coefficient recurrence.
Replacing it with the bare recurrence changes the finite-window response.

There is an inexpensive structural way to retain it. Let $J_c$ be the first
sample at which $w_j\ne1$, let $L=N-J_c$, and define

$$
 g_k=(w_{J_c+k}-1)h_{J_c+k},\qquad 0\leq k<L.
$$

Then the tapered response is the untapered recurrence plus

$$
 \delta r_n=0\quad(n<J_c),
 \qquad
 \boxed{\delta r_{J_c+p}=\Delta\sum_{k=0}^{p}g_k d_{p-k},
       \quad0\leq p<L.}
$$

Only the **first $L$ drive samples** and a **length-$L$ correction kernel** are
needed. Only the last $L$ outputs change. Compute this small convolution by a
direct sum when tiny, or with a transform length at least $2L-1$ when larger.
For a narrow collar, the auxiliary transform is much smaller than $2N$.
The correction kernel is independent of the physical node and can be prepared
once for the stage and cached under the same material parameters.

This decomposition preserves the present sampled model in exact arithmetic,
including mixtures and tensor channel weights. The source of numerical
differences is then arithmetic order and recurrence roundoff, not a removed
boundary condition. For a gas gradient, forming its correction kernel also
costs work, potentially $O(ML)$ without reusable sampled data; include that
cost in the comparison.

The decomposition does not promise a win if the collar occupies most of the
window or the number of poles is large. Subtracting a long-delay contribution
can also expose cancellation near a zero output. Tests need absolute-summand
error bounds, not relative error alone.

Removing the taper altogether remains a plausible **separate** design change.
For the first $N$ outputs of an exactly causal convolution, lags beyond $N-1$
never enter. A hard cutoff beyond all usable lags causes no error in those
outputs; spectral ringing of the finite kernel is not by itself evidence of
incorrect finite-window convolution. Existing delayed-probe bias from the taper
would disappear. Nevertheless, this changes the documented operator and current
tests and therefore needs an explicit decision, shared by the FFT oracle and
the recursive route. A performance backend must not make that decision alone.

If one instead proposes neglecting the correction only for some inputs, the
relevant error is bounded by

$$
 |\delta r_n|\leq\Delta\sum_{j=J_c}^{n}
       |(1-w_j)h_j|\,|d_{n-j}|.
$$

For polarisation accuracy this must also be multiplied by $\epsilon_0\rho\lvert F_n\rvert$.
“The pulse is centred” or “the collar is small” is not a general error bound.

#### Molecular gases

Every retained `DampedOscillator` and every line in `RotationalLines` has the
required exponential-sine form. A recurrence can preserve frequencies,
Boltzmann/nuclear-spin weights, independent decay rates, mixture amplitude
densities and tensor weights exactly at the model level.

However, the FFT sees a combined response while recurrence sees all its poles.
Hydrogen may have only a few influential lines at a chosen temperature, but
that is not permission to replace its full ladder by one effective S(1)
transition. Nitrogen, oxygen, higher temperatures, mixtures and delayed revival
studies make this distinction particularly important.

If pruning is investigated later, use a bound on omitted response rather than
a population threshold alone. For non-negative $\gamma$ and a scalar drive,

$$
 \|\delta r\|_\infty\leq
 \|d\|_\infty\sum_{\ell\in\mathrm{omitted}}|K_\ell|
 \begin{cases}(1-e^{-\gamma_\ell T})/\gamma_\ell,&\gamma_\ell>0,\\
 T,&\gamma_\ell=0.
 \end{cases}
$$

This is conservative because it ignores cancellation and bounds $\lvert\sin\rvert$ by
one. Tensor weights, density and the final field must be included for a source
bound. A line small in a broadband impulse response can still matter for a
selectively resonant, long pump–Stokes beat. No universal pruning tolerance is
selected here.

#### The two silica models need different treatment

The default scalar Hollenbeck–Cantrell model [1] is a sum of 13 lines of the form

$$
 A_\ell e^{-\gamma_\ell t-\Gamma_\ell^2t^2/4}
        \sin(\omega_\ell t).
$$

The Gaussian factor prevents an exact small fixed-pole recurrence of the
simple form above, with state count independent of window length. Its sample-to-sample ratio varies with **lag**, so one cannot
make a correct convolution by using a single time-varying decay coefficient on
the accumulated state: contributions injected at different times have
different ages. Keep its FFT, use certified finite support, or approximate it
by a stable exponential sum with a separately declared fit error. Replacing
this model with a one-oscillator silica fit is a physics change.

The existing general-modal Lin–Agrawal tensor model [2, 3] is more promising. Its two
shapes are an exponential sine and

$$
 h_b(t)=(2/\tau_b-t/\tau_b^2)e^{-t/\tau_b}.
$$

The latter has an exact two-state repeated-pole realisation. With
$b=\exp(-\Delta/\tau_b)$, define zero-prehistory sums

$$
 u_n=b u_{n-1}+d_n,\qquad
 v_n=b(v_{n-1}+\Delta u_{n-1}).
$$

Then $v_n$ is the lag-weighted sum, and its sampled contribution is

$$
 r_{b,n}=\Delta\left(\frac{2u_n}{\tau_b}
                      -\frac{v_n}{\tau_b^2}
                      -\frac{d_n}{\tau_b}\right).
$$

The final subtraction changes the zero-delay coefficient from $2/\tau_b$ to
$1/\tau_b$, preserving the existing half-weight at the nonzero causal endpoint.
Omitting it introduces precisely the endpoint error the present model's tests
guard against. Preserve the negative late lobe as well. Its sign is part of
the model.

In the inspected general-modal translation route these silica kernels are
sampled and zero-padded **without** the gas path's Planck taper, so there is no
Planck correction to undo there. Sharing the shape states before forming the
two tensor channels may save work. This is a particularly attractive candidate
because the number of shape states is small and there are six histories per
node, but it lives on a separate regional-response contract and backend path.

#### Density-dependent poles

At a fixed z stage, use the translated $K_\ell(z)$, $\Omega_\ell(z)$ and
$\gamma_\ell(z)$ throughout that temporal sweep. For optmat's current fixed
temperature/composition, amplitudes are constant and its frequency shifts and widths
vary with density. The recurrence derivation is unchanged. Exact FFT evaluation
samples these same lines at the stage's frequencies and rates, applies the same
taper/endpoint rule, then transforms the sampled kernel. It must not reuse a row
sampled at another frequency. This exact route supplies SEP 10's shifted-H2 oracle.

"Exact" means no compression or temporal-quadrature approximation relative to this
sampled operator. It does not remove roundoff, the existing z-table interpolation
error, sampling error, finite-window error or material-data uncertainty. Frequency
path accuracy must account for phase accumulation $T\,|\delta\Omega|$, and rate
paths for $T\,|\delta\gamma|$ over the used history. Check the assembled kernel
against direct material evaluations at independent z points; small relative errors
in a line frequency alone do not establish an accurate long coherence.

### 5. Proposed interface

These settings are proposed, not available API. Add to `SimulationSettings` and
forward through the convenience constructors:

| Setting | Default and contract |
|---|---|
| `raman_evaluator` | `"fft"`; also `"recursive"`. Forced recurrence requires every selected delayed term and backend to support its analytic representation. |
| `raman_rank_tolerance` | `0.0`, exact evaluation. A finite value $0<\epsilon<1$ explicitly requests SEP 10's low-rank FFT representation. Negative, non-finite or $\epsilon\geq1$ values are refused. |

A positive tolerance with `"recursive"` is refused. Until SEP 10 is implemented,
positive tolerance is refused by name. A sampled custom kernel remains valid with
exact FFT. No implicit fitting, pruning, change of model or backend fallback occurs.
The selected backend refuses an unsupported evaluator before allocation.

```text
settings = SimulationSettings(..., raman_evaluator="recursive")
# Exact FFT remains the default. SEP 10 later adds:
settings = SimulationSettings(..., raman_evaluator="fft",
                              raman_rank_tolerance=1e-4)
```

Existing response factories, tensor declarations and `thg` semantics remain intact.
Record requested/resolved evaluation, representation, boundary/endpoint policy and
material identity. Pure exact-evaluator choice is execution metadata; a positive
compression tolerance is a source-approximation choice and enters configuration and
resume identity. Record representation algorithm versions without making floating-point
SVD vectors a physical identity. Old saved fields remain readable.

### 6. Common execution contract and implementation

#### Model ownership and representation

The host response owns the material laws and any analytic description. Translation
produces a frozen Raman leaf with common identity, label, amplitude-density table,
fraction, drive form, tensor capability and contraction weights. Its execution
payload has exactly one explicit tag:

| Tag | Backend data and evaluation |
|---|---|
| `sampled` | Existing undamped causal rows and decay tables, plus the explicit boundary/endpoint identity; exact FFT for supplied kernels without analytic metadata |
| `analytic` | Per-line SI amplitudes, angular-frequency and decay tables, line-to-tensor mappings, exponential-sine or repeated-pole shape tags, endpoint weights and taper/correction data; exact FFT or recurrence |
| `spectral` | Static complex spectral rows and real coefficient tables, contraction mappings and measured approximation record; FFT, introduced by SEP 10 |

The mathematical descriptions in §4 define the analytic shapes; a backend never
infers them from samples or imports optmat. Constant parameters use degree-zero
tables. Shape and tensor mappings may share identical poles without merging distinct
lines. Retain existing model/fraction conventions, including absolute optmat strengths.
Ordinary and regional leaves keep their existing prefactors and spatial ownership;
the same representation vocabulary is embedded in both, rather than reinterpreting
regional silica as a gas. An unsupported tag/shape/route fails by name.

Exact static spectra can use `spectral` with a unit coefficient and zero compression
error once SEP 10 adds that tag. No SVD is required for a fixed kernel. Until then,
the current cached exact FFT representation remains available.

#### Stages, memory and backends

Stage bundles must carry the evaluated analytic amplitudes, frequencies and rates,
or spectral coefficients, in a documented order. Changing any source-defining path
invalidates its cached kernel/pole/correction. Include tensor and boundary identity,
not density alone. Coefficients may be cached; temporal recurrence states are reset
for every new optical field, including rejected trials. Valid whole-source reuse
follows the existing stepper contract and never carries a molecular end state along z.

Keep the existing aggregation of compatible FFT kernels and shared tensor drives.
Recurrence can accumulate each line/shape contribution into the same source; its
taper corrections can likewise be combined when their drive and contraction match.
Preallocate all buffers, include them in chunk admission and `memory_ledger()`, and
count preparation, correction and tail-plan storage. The current ordinary Raman
work arrays cost $48N+16$ bytes per node/drive. Recurrence streams $O(M)$ state,
but a collar of length $L$ adds $O(L)$ scratch and preparation up to $O(ML)$ per
stage; retaining any length-$N$ history must appear in the ledger.

Use NumPy for the oracle and shared mathematical bodies for Numba, including
interpreted coverage. CPU nodes and batch members provide parallelism; any changed
reduction order is qualified to a rounding bound under SEP 25. GPU temporal scans
are later work, with no speed claim or unqualified capability advertised.

The common foundation and each later seam extension bump the then-current
`SCHEMA_VERSION`, including frozen stage-bundle literals and coordinated native
consumer checks. No schema number is reserved. The backend may live in the external
solprop-kokkos repository or, after SEP 29, the in-tree native package; qualification
and named refusals apply to the consumer actually built. No dependency is added.

#### Other responses and future material laws

With SEP 6, the electronic filters act only on the electronic field/products and
source. Raman uses the unfiltered physical field and its own emission; sum the
sources after their proper assembly. Preserve current `thg`/harmonic-sector
admission, and test both field and envelope forms in their supported limits.
An UPPE frequency prefactor or dispersive modal overlap is not Raman coupling
dispersion. Density from SEP 11 changes the translated state consistently but adds
no automatic Lorentz local-field factor to Raman or electronic nonlinearity.

### 7. Verification and validation

#### Recursive evaluator: distinguish algebra, quadrature and physics

1. **Discrete causal oracle.** On deliberately small arrays, compare against an
   explicit lower-triangular convolution sum, including an impulse, a late
   impulse, constant and signed drives, and arbitrary deterministic histories.
   Include signed off-diagonal tensor drives. An impulse near the right edge
   must not change any earlier response.
2. **Taper identity.** Compare recurrence plus correction with the existing
   tapered FFT over the complete window, including pump–probe lags inside the
   collar and a drive nonzero at the first sample. Exercise narrow/wide collars
   and boundary indices exactly at $J_c$.
3. **Conditioning.** Test nearly undamped lines, very small $\Omega\Delta$, many periods,
   cancellation between lines, large decay rates and the zero-density limits
   already handled by translation. Bound error with absolute convolution sums,
   coefficient errors and accumulated recurrence roundoff; do not require a
   relative error at an exact zero or promise bitwise FFT equality.
4. **Silica endpoint and shape.** For Lin–Agrawal, preserve $h_b(0^+)/2$, its
   negative late lobe, integrated strengths, both tensor channels and the
   scalar limit. Compare the shape-level and channel-level implementations.
5. **Continuous-time reference.** Independently verify impulse response,
   constant-drive response and harmonic susceptibility, separating quadrature from
   discrete evaluator parity. If the future interpolated-drive alternative is taken
   up, its separate proposal must require three temporal refinements and the expected
   order; that alternative is not a completion condition here.
6. **Existing physical checks.** Retain Stokes-gain sign and magnitude,
   field/envelope cycle-average parity, tensor rotation covariance, mixtures,
   gradients and their density/rate separation. Use the existing independent
   angular/quantum-response tests, not a fit to the FFT output.
7. **Backend and stage behaviour.** One source evaluation, one step and a short
   multi-step witness on NumPy, Numba and installed Kokkos CPU; GPU qualification
   separately. Include node chunk tails, batches and supported geometries,
   memory ledgers, thread-count invariance and rejected-step behaviour.

For reference, the continuous untapered single-line susceptibility under the
internal transform convention is

$$
 H(\omega)=\frac{K\Omega}{(\gamma+i\omega)^2+\Omega^2},
 \qquad H(0)=\frac{K\Omega}{\gamma^2+\Omega^2}.
$$

It supplies independent DC and spectral checks; a finite tapered kernel does
not equal it exactly. Finite window, taper and temporal sampling errors must
be separated in such comparisons.

#### A bounded performance comparison

Measure the same physical operator and input in the existing FFT route and
the recursive candidate. Representative axes are:

| Axis | Cases worth retaining |
|---|---|
| Fine-grid size | A small case, an ordinary working grid, a long history; include non-power-of-two sizes already supported |
| Pole content | Single vibration; full H2 ladder plus vibration; a denser ladder/mixture; Lin–Agrawal repeated-pole shapes |
| Drive count | Scalar; two-component isotropic; two-component trace/rank-two; general-modal six-history route |
| Material variation | Constant rates; varying rates; varying mixture amplitudes |
| Parallelism | One node and many nodes; one and several CPU threads; real GPU builds separately |
| Memory constraints | Fixed chunk size for arithmetic comparison, then fixed memory budget for end-to-end comparison |

Report separately the pole/correction preparation, drive construction,
convolution/recurrence, final source contraction, source-stage wall time,
planning/compilation time and peak/named workspace. Include actual retained
line counts. Hold FFT settings and thread counts fixed in controlled
comparisons. Measure full-ladder recurrence before an optional approximation
that prunes lines.

Whole-solver timings should use about ten accepted steps from an untimed pilot
or a sufficiently short fibre that it finishes in seconds. Use saved witnesses
for strongly broadened later states, alternating fresh-process A/B runs with
the A/A spread. Report the measured z interval and accepted/rejected steps;
do not extrapolate to a full capillary/fibre propagation. These are the
repository's existing benchmark rules.

#### Arithmetic error budget

The following is a plan, not a report of measured errors. Let $u$ be machine
unit roundoff and $\eta_k=ku/(1-ku)$ for $ku<1$. For a direct causal sum, an
absolute scale is

$$
 B_n=\Delta\sum_{m=0}^{n}|h_{n-m}d_m|.
$$

A floating-point summation bound is a justified multiple of $\eta_k B_n$,
where $k$ counts the actual products/additions and includes input/kernel
representation error. For line cancellation use the stronger line-resolved
scale $\Delta\sum_\ell |K_\ell|\sum_{m=0}^n|a_\ell|^{n-m}|d_m|$;
for a taper correction also retain the absolute scale of the two terms
before their cancellation. These bounds have the right dimensions and do
not vanish merely because the answer is zero.

For a computed recurrence $\widehat s_n=\widehat a\widehat s_{n-1}+d_n+\xi_n$,
the error obeys

$$
 |e_n|\leq |\widehat a|\,|e_{n-1}|
       +|\widehat a-a|\,|s_{n-1}|+|\xi_n|.
$$

Unroll this inequality over the actual history, then account for the final
line sum and source prefactor. This explicitly separates rounded pole
coefficients from arithmetic noise. FFT comparison needs the FFT oracle's
own transform/summation error as well; agreement is not expected bitwise.
A near-unit pole rounded slightly outside the unit disk needs a quantified
bound or a deliberate stable coefficient construction, not an assumption
that a mathematically damped line cannot drift. Tests must set strictly
positive tolerances from these scales and evaluated operation counts, with
an absolute rounding floor for zero signals. A single universal relative
threshold chosen after observing results is unacceptable.

Add density-shifted H2 against independently sampled optmat histories, vacuum,
linewidth-law transition points, changing rates/frequencies/amplitude density, the
same label split and reorder, and all ordinary/regional paths admitted today.
Check the exact frequency/rate table error separately from evaluator roundoff.
Run SEP 6 coexistence tests with a bare Raman source reference so accidental
electronic filtering cannot pass by comparing two identically modified paths.
For the same prescribed physical field, the Raman polarisation must be unchanged
when the electronic evaluator changes. Check modal/vacuum normalisation, drive
halves and the admitted harmonic sectors. Optical work is linear in the assembled
polarisation: compare $\int E\,\partial_t(P_e+P_R)\,dt$ with the separate contributions
under the same boundary convention (and the derived cycle average for envelopes).
Retain model-appropriate Raman photon/work checks; do not impose optical-energy
conservation on dissipative Raman transfer. These tests check source composition,
not agreement of propagated fields under different electronic models.

Matched discrete source/step/eight-step comparisons use precomputed arithmetic
bounds. Adaptive short runs use achieved-error envelopes against a tighter or
stored converged reference; matching work counts is not required. A compressed
source is never used as the exact oracle. Low-rank-specific checks belong to SEP 10.
No comparison runs a full-length capillary or fibre. Benchmarks include kernel
preparation, complete source stages and about ten steps, at matched physical inputs
and accuracy, with memory and actual line counts. A slower recurrence may still be
useful for memory; this SEP promises neither a speedup nor an automatic preference.

### 8. Documentation

Extend the existing Raman workflow, model, numerical and data pages with the selector,
discrete recurrence/taper derivations, supported analytic shapes, density-shifted
reference and error distinctions. Update the kernel contract, kernel-problem and
backend pages when their implementation changes, and add a short executable
evaluator example. Keep the current manual truthful until those phases land.
Implementation changes receive docstrings, changelog and benchmark entries. At Final,
link the manual to this SEP and archive it under SEP 0.

### 9. Compatibility and impact

Default FFT inputs keep their sampled operator and material selection. Shifted H2
on a gradient becomes available with its actual density law, while the recorded
reference-density approximation remains explicit. Recurrence changes arithmetic
order, not quadrature, populations, taper, endpoint weights or drive conventions.
It can alter adaptive decisions within the qualified accuracy envelope.

No saved-field axes or Fourier conventions change. Execution metadata records exact
evaluator changes; material/source approximation changes affect resume identity.
Unsupported recurrence is refused, never redirected to another backend. Existing
FFT use remains supported during rollout. No Maxwell–Bloch release is needed for
this SEP to become Final.

### 10. Alternatives and compatible future work

#### Continuous-time exponential updates are a second option

The same continuous convolution can be represented by

$$
 \dot x_\ell=\lambda_\ell x_\ell+d(t),\qquad
 \lambda_\ell=-\gamma_\ell+i\Omega_\ell,
 \qquad r(t)=\sum_\ell K_\ell\operatorname{Im}x_\ell(t).
$$

This form permits exact integration of the homogeneous oscillation and a
chosen interpolation of the drive. For a linear drive between $t_n$ and
$t_{n+1}$, with $z_\ell=\lambda_\ell\Delta$, direct integration gives

$$
 x_{\ell,n+1}=e^{z_\ell}x_{\ell,n}
 +\Delta\left[(\varphi_1(z_\ell)-\varphi_2(z_\ell))d_n
                 +\varphi_2(z_\ell)d_{n+1}\right],
$$

$$
 \varphi_1(z)=\frac{e^z-1}{z},\qquad
 \varphi_2(z)=\frac{e^z-1-z}{z^2}.
$$

Their small-argument limits are 1 and 1/2, evaluated with series/`expm1` rather
than subtracting nearly equal floating-point numbers. Higher-degree drive
interpolation gives further exponential moments. Recursive convolution of this
kind has established precedent in dispersive electromagnetics; Kelley and
Luebbers' PLRC work is a useful methodological reference, not a Raman
normalisation source. [Kelley's research and publication record](https://www.eg.bucknell.edu/~dkelley/research/research_FDTD.html)

This would change the temporal quadrature relative to today's sampled FFT
product. It needs independent continuous-time convergence tests, including the
initial boundary convention. The exact sampled recurrence derived above is a cleaner first performance
comparison because it isolates execution cost.

An exponential update removes a homogeneous-oscillator stability restriction;
it does **not** recover an unresolved drive or polarisation. Interpolating a
coarse drive still needs a demonstrated error budget. In particular, explicit
Euler is unsuitable for a nearly undamped oscillator: its amplification factor
has magnitude greater than one even when the true free response does not grow.

#### Certified finite support and overlap-save

If a kernel is negligible after $L_h$ samples, a full linear convolution needs
an FFT length at least $N+L_h-1$, rather than $2N$. Blocked overlap-save or
overlap-add uses small transforms with the required history overlap and retains
only uncontaminated samples. These methods change the workspace and scheduling,
not causality. A truncation must be certified, for example by

$$
 \|\delta r\|_\infty\leq
 \|d\|_\infty\int_{T_h}^{\infty}|h(\tau)|\,d\tau.
$$

For a short silica response on a very long optical window, this is worth
comparing with the full FFT. For a molecular coherence that hardly decays
inside the window, it offers little without exploiting the oscillator
structure. It also needs a cutoff valid over every density/linewidth reached
by a gradient. FFT-friendly sizes matter: shorter nominal length alone need
not mean a faster transform.

#### A coarser Raman-only grid

This can be worthwhile, but only as a controlled multirate approximation. The
candidate procedure is to filter the quadratic drive before decimating,
evaluate the molecular response on the smaller grid, then interpolate the
response back before multiplying the optical field. The retained bandwidth
must cover the response actually needed, including molecular resonances and
the relevant forcing bandwidth; it cannot be chosen from $T_2$ alone.

For perspective, the H2 vibration has a period of about 8 fs [5]. Its sinusoid
alone needs a spacing below about 4 fs for Nyquist sampling, and finer sampling
may be needed for the desired accuracy. A subcycle optical feature, broad pump–probe beats and the
instantaneous part of a shaped response impose additional constraints. These
numbers explain scale, not a recommended production grid.

Discarding the $2\omega$ part of $E^2$ is already an explicit approximation available
through solprop's drive form. It often works well far from the molecular
resonances, but neither Raman's low resonant frequency nor an exact recurrence
proves it valid for an arbitrarily broad real field. Downsampling without
filtering can alias optical beats directly into a Raman line. FFT filtering
itself also carries periodic-boundary assumptions that must be controlled.

The final product $rE$ still belongs on the optical fine grid. A dual-grid
method therefore saves only part of the Raman work and adds filtering,
interpolation and separate error controls. It is a second-stage investigation,
after establishing the same-grid recurrence crossover.

#### Short-response moment expansions

For a drive slowly varying across the entire response memory, expand

$$
 (h*d)(t)\simeq H_0d(t)-H_1\dot d(t)+\tfrac12H_2\ddot d(t)+\cdots,
 \qquad H_k=\int_0^\infty\tau^k h(\tau)\,d\tau.
$$

This explains familiar approximate Raman self-frequency-shift terms. It is
cheap in its validity regime, but omits the resolved coherent ringing,
resonant gain and delayed-probe effects motivating gas Raman here. It is not
the general performance solution for molecular gases.

#### An unpadded FFT with a causal boundary correction

For one exponential pole there is a mathematical route: solve the periodic
recurrence using the length-$N$ transfer function

$$
 H_k=(1-ae^{-2\pi i k/N})^{-1},
$$

then correct its periodic solution $c_n$ to zero prehistory by
$s_n=c_n-a^{n+1}c_{N-1}$. This is exact when that periodic inverse exists.
It is not simply the existing finite kernel transformed on a smaller grid.

For a sum of poles, boundary amplitudes are generally needed separately for
each pole; evaluating them or each periodic response can lose the intended
cost advantage. Weak damping near a periodic resonance makes the periodic
solution large and the subtraction ill-conditioned; an undamped resonant pole
can make it singular. A direct causal recurrence avoids this artificial
periodic problem. This option is mathematically interesting but is not the
recommended first implementation.

(sep1-broadband-raman)=
#### Optical-frequency-dependent Raman coupling

Three different dependences must be distinguished: the Raman-shift spectrum
$H(\Omega)$, strengths selected at a carrier wavelength, and a broadband transition
coupling depending on the participating optical frequencies. The first two already
exist. Optmat disperses its curated vibrational mean-polarisability elements to the
chosen carrier, while its anisotropies are not dispersed. Neither this SEP nor SEP 10
adds the third dependence.

The need can be material. At 295 K, optmat 0.8.0's summed H2 isotropic amplitudes,
relative to 800 nm, are 1.147877 at 400 nm and 1.482475 at 266 nm (Part III).
These are evaluations of a fitted material model, not gain measurements or
propagation errors. Bischel and Black measured UV enhancement beyond the Stokes
frequency's fourth-power cross-section scaling [6]. A fit at one pump does not
uniquely determine every nondegenerate coupling or its validity near resonance.

The named follow-up **Broadband Raman coupling** on the roadmap will derive the
drive and emitted polarisation together, preferably from a reciprocal transition
kernel. A separable off-resonant approximation could use filtered fields around
the same causal material response, with added FFTs per retained filter. Its rank
in optical-frequency pairs is distinct from SEP 10's rank along a density path.
Check reality, Hermiticity where a state model is used, frequency permutations,
causality/domain assumptions and field–matter work. This is feasible research,
not a qualified model, and does not require Maxwell–Bloch populations first.
Do not borrow SEP 6's electronic filters or multiply only the final Raman source
by a wavelength curve. SEP 32 retains the complex-coupling counterexample.

### 11. Phases

Branches use `feature/sep-0001-raman-response-<phase>`; implementation has not started.

| Phase | Deliverables and exit evidence |
|---|---|
| 1 — Common foundation and NumPy | Sampled/analytic payloads, stage mappings, exact shifted-line FFT and recurrence, ordinary and regional contracts, schema/native compatibility checks, selectors/refusals and manual; direct sums, conditioning, endpoints, gradients and memory |
| 2 — Numba CPU | Shared compiled bodies, preallocated correction workspaces and stage integration; oracle parity, batches/chunk tails/rejections, interpreted coverage and short benchmarks |
| 3 — Kokkos CPU | Equivalent native execution and ledger; parity, supported-range checks and threaded short-stage/propagation measurements |
| 4 — Qualification and closure | Consolidated supported-range/error and performance evidence, provenance/resume checks and documentation; retain FFT default and publish limitations |

SEP 10 can build on the merged phase-1 contract; its compressed path is not needed
to finish recurrence. Each phase is independently mergeable. Development slices
run the repository's fast checks and directly affected test modules. At the end
of the SEP, run the full gate, slow lanes and coverage lane once on the final tree,
or take them from CI. After failures, rerun failing modules rather than entire
successful lanes. Documentation-only revisions use the separate
documentation gate. GPU qualification is a later, explicitly named roadmap item.

Append actual phase reports under Part II: changes, decisions/deviations, sources,
checks, bounds, timings, limitations and handoff. No phase report is prospective.

### 12. Decisions and remaining questions

The scope/default decisions D1–D5 in Part III are the maintainer's instructions for
this revision, not acceptance of the whole SEP. No unresolved physics choice is
needed to implement the existing sampled operator.

| Question | Disposition |
|---|---|
| Maximum qualified pole/history range and coefficient construction | Phase 1 derives bounds and failure criteria; phases 2–3 validate them. Forced recurrence refuses an unqualified range. |
| Collar preparation caching and CPU crossover | Measure in phases 2–3, publish in phase 4; no default or automatic policy depends on a hoped-for speedup. |
| Device execution | Phase 4 records a GPU qualification destination; unsupported device recurrence is refused. |
| Other quadratures, taper removal, support truncation and pole fitting | Separate roadmap work, not phase completion requirements. |
| Broadband coupling and population dynamics | Named roadmap follow-up and SEP 32 respectively; neither blocks acceptance here. |

Human review and explicit acceptance remain outstanding.

### 13. References

1. D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for fiber-optic
   Raman gain spectrum and response function," *JOSA B* **19**, 2886–2892 (2002).
   [doi:10.1364/JOSAB.19.002886](https://doi.org/10.1364/JOSAB.19.002886).
2. Q. Lin and G. P. Agrawal, "Vector theory of stimulated Raman scattering and its
   application to fiber-based Raman amplifiers," *JOSA B* **20**, 1616–1625 (2003).
   [doi:10.1364/JOSAB.20.001616](https://doi.org/10.1364/JOSAB.20.001616).
3. Q. Lin and G. P. Agrawal, "Raman response function for silica fibers," *Optics
   Letters* **31**, 3086–3088 (2006).
   [doi:10.1364/OL.31.003086](https://doi.org/10.1364/OL.31.003086).
4. D. F. Kelley and R. J. Luebbers, "Piecewise linear recursive convolution for
   dispersive media using FDTD," *IEEE Trans. Antennas Propag.* **44**, 792–797 (1996).
   [doi:10.1109/8.509882](https://doi.org/10.1109/8.509882).
5. optmat 0.8.0, [Raman response](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman_response.md)
   and [line data](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman.md).
6. W. K. Bischel and G. Black, "Wavelength dependence of Raman scattering cross
   sections from 200–600 nm," *AIP Conference Proceedings* **100**, 181–187 (1983).
   [doi:10.1063/1.34046](https://doi.org/10.1063/1.34046);
   [authors' conference abstract](https://opg.optica.org/abstract.cfm?uri=EL-1983-TuB3).

The recurrence and taper identities are derived here; the silica endpoint is checked
against the existing cited model. No physical coefficient is changed. The current
review checked optmat's implementation/data records and the primary abstract for
[6], not every original coefficient table in [1–6]. Implementation must verify
transcribed analytic-shape data against the primary sources and report gaps.

## Part II — Implementation record

No implementation phases have started.

## Part III — History

- 2026-09-26: Drafted with both evaluator improvements and Maxwell–Bloch dynamics.
- 2026-10-17: Restated numerical reproducibility for SEP 25.
- 2026-10-22: Split the new material model into SEP 32; reviewed and revised the
  remaining proposal jointly with SEP 10, with compatibility changes to SEP 11.

### Maintainer decisions for this revision — 2026-10-22

John Travers requested the split, joint review and preparation for his subsequent
acceptance review, and selected these directions:

- **D1:** preserve the existing lag taper.
- **D2:** retain exact FFT as the default and require explicit low-rank tolerance.
- **D3:** keep optical-frequency-dependent Raman coupling a separate extension.
- **D4:** adopt a composition-preserving mixture-EOS direction in SEP 11.
- **D5:** keep SEPs 1 and 10 Draft / In review; acceptance follows human review.

These authorise the revision. They are not a completed human review of Part I.

### AI review — 2026-10-22 — GPT-6

Review of the complete pre-split Part I against the current code and SEP 0, jointly
with SEP 10; the findings are retained separately from the revised proposal.

1. **R1 (P1), §§2, 5, 11–12 — Independent deliverables are coupled.** Completion
   requires both an evaluator and a new material model with unresolved ensemble/data
   assumptions. Extract the latter, preserving the weak-limit derivation and limits.
   **Maintainer disposition:** split requested; D5 retains human review.
2. **R2 (P1), §§3, 6 — The baseline and seam predate optmat.** Default H2 has 20
   lines, not the supplied factory's old ladder count; current leaves do not expose
   density-varying poles. SEP 10's replacement would remove recurrence's inputs.
   Define one host/backend contract and an exact shifted-line FFT reference.
3. **R3 (P1), §§4, 7 — Numerical identity needs qualification.** Retain the taper,
   endpoint and drive factors, but separate pole roundoff and z-table phase error
   from sampling/quadrature. Include complete correction preparation in cost.
   **Maintainer disposition:** D1 preserves the operator.
4. **R4 (P1), §§5–6, 9 — Evaluator and approximation policies conflict across SEPs.**
   Keep exact default, explicit tolerance and named refusals; defer `auto`.
   **Maintainer disposition:** D2 chooses explicit approximation.
5. **R5 (P1), §§3, 6, 10 — Electronic dispersion is not Raman dispersion.** Preserve
   the bare Raman field and source beside SEP 6, and distinguish optical-pair
   dispersion from density compression. Bind all responses to SEP 11's state.
   **Maintainer disposition:** D3–D4 set the extension/EOS direction.
6. **R6 (P2), §§8, 11 — Documentation and test policy are stale.** Use the current
   manual hierarchy and slice/final gate policy; do not demand whole slow/coverage
   reruns at every slice or tie Final to the extracted Bloch work.

### Response to review — 2026-10-22

- **R1:** moved the Bloch theory, interfaces, numerical method, validation and open
  physics decisions into SEP 32. §§2 and 11–12 now close independently.
- **R2:** refreshed §3, added density-dependent poles in §4 and the tagged ordinary/
  regional seam in §6. SEP 10 explicitly extends this foundation.
- **R3:** retained the exact identities and expanded §§4 and 7's distinct error
  contracts. Cost includes preparation and collar work in §§6–7.
- **R4:** §5 defines the shared settings, invalid combinations and exact default.
  Automatic policy is excluded from these phases.
- **R5:** §§3, 6 and 10 state response ownership, EOS independence and the named
  broadband-coupling follow-up; SEP 11 is revised consistently.
- **R6:** §§8 and 11 use current manuals and testing policy. Human acceptance is
  outstanding; implementation tuning has named phase destinations.

### Review probes and source audit — 2026-10-22

- With NumPy, a 257-sample dimensionless damped sine, $\Delta=0.1$, decay 0.02,
  frequency 0.47, a collar starting at sample 219 with a squared linear fall, and
  `default_rng(510)` normal drive, recurrence plus delayed correction differed from
  direct causal convolution by $3.85\times10^{-15}$ at most. This checks the general
  correction identity with a synthetic taper; it does not qualify production Planck
  sampling, supported histories or a backend.
- With the same signed drive, $\tau_b=3$ and $\Delta=0.1$ in dimensionless units,
  the two-state repeated-pole update with the half-endpoint subtraction differed
  from direct sampled convolution by $2.30\times10^{-16}$ at most. This checks the
  endpoint identity, not the physical silica parameters or production integration.
- Repeating the optmat 0.8.0 line/group/nonzero-family count at 293.15 K and 800 nm
  confirmed §3's H2, N2 and dry-air counts and finite group decay rates at vacuum.
- Evaluating `raman_response("H2", 295.0, wavelength=...)` in optmat 0.8.0 and
  dividing the sum of `isotropic` amplitudes by its 800 nm value gives 1.147877
  at 400 nm and 1.482475 at 266 nm. The anisotropy is separate. This checks the
  implemented carrier dependence, not a broadband transition kernel or experiment.
- Current source/data ownership, tensor mapping, finite vacuum linewidths and SEP 6
  composition were inspected. The primary Bischel–Black abstract was read; its full
  coefficient table was not independently checked in this review. Existing source
  checks inherited by SEP 32 are attributed to its original drafting record.
- Documentation checks passed: `git diff --check`, codespell on all seven changed
  documentation files, `uv run pytest tests/test_seps.py` (231 passed, two expected
  process-SEP skips), and `uv run poe docs` with warnings treated as errors. The first
  docs build found one missing subsection anchor; an explicit label fixed it and
  the build passed. Introductory lessons and focused tutorials ran; worked examples
  were not rerun. No recurrence implementation, full propagation qualification or
  performance timing is claimed by this review.

All R1–R6 findings are addressed by revisions or explicit scope deferrals above.
The joint review found no remaining contradiction between SEPs 1 and 10's selected
contracts. Human review and acceptance are outstanding; implementation qualification
remains in the named phases. SEP 32's model/data questions and SEP 11's own admission/
default questions are not prerequisites for accepting this evaluator proposal.
