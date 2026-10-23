# SEP 26 — An edge-free gated TDSE source: hold the current instead of switching it off

| | |
|---|---|
| **SEP** | 26 |
| **Title** | An edge-free gated TDSE source: hold the current instead of switching it off |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-18) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-18 |
| **Resolution** | Accepted 2026-10-18 by the maintainer, after the human and AI reviews (R1–R13) and the drafting agent's response (branch `docs/sep-0026-accept`) |
| **Requires** | [SEP 12](sep-0012-tdse-nonlinear-response.md) (the `tdse_native` source and its `time_collar` gate); [SEP 20](sep-0020-hacc-helium-tdse.md) (the haCC response, the atomic window, L5, L6 and Q30) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | phase 1: Opus 5.5 (AI) for John Travers, branch `feature/sep-0026-phase-1` (solprop and solprop-kokkos); phase 2: Opus 5.5 (AI) for John Travers, branch `feature/sep-0026-validation`; phase 3 (promotion and the final gate): Opus 5.5 (AI) for John Travers, branch `feature/sep-0026-phase-3` |
| **Manual pages** | edge-free sections of the [haCC helium response](../../../guide/media/tdse.md), [TDSE in capillaries](../../../guide/media/tdse.md) and [TDSE configuration](../../../guide/media/tdse.md); SEP 26 paragraphs of [resuming](../../../guide/results/resuming.md) and [differences from Luna](../../../guide/workflows/luna.md) |

## Part I — Proposal

### 1. Abstract

A gated TDSE source today is the transform of $g(t)S(t)$. Here $S=J-j_1$ is the atom's
nonlinear remainder current, and $g$ is the Planck gate: the atomic window's, or the
grid's temporal collar. When $S$ is still nonzero in the trailing taper, the gate
switches it off over a few femtoseconds. That switch-off is not an electron's
response, but it puts near-infrared content into the source.

For haCC at the run_short peak-power plane, removing that switch-off from both gates
of the same 100 fs current reduces their 60-against-100 fs band difference by about
73% (§2). This SEP adds the source filter `"gated_derivative"`. On the active band it
replaces $\mathcal F[gS]$ by $\mathcal F[g\,\partial_tS]/(i\omega)$, evaluated as
$\mathcal F[gS]-\mathcal F[g'S]/(i\omega)$ with the analytic gate derivative $g'$. The
result is the transform of a current *held* at its taper-weighted value after the
window instead of being switched off. A steady current then contributes nothing at
$\omega\neq0$, all time variation inside the window is kept, and the form has no free
parameter.

The discrete form is admitted only on lattices that resolve the tapers, and each
configuration carries a computed steady-current certificate (§4.4). The filter is
opt-in until its validation passes and the maintainer has decided the remaining Q30
exception (§11).

Success means:

- the analytic and fake-provider tests close within their derived budgets;
- the archived L5 currents reproduce the supporting study's same-current split;
- the actual L5 comparison, an L6 segment, the SAE controls and the rate-plasma
  measurement are reported.

### 2. Motivation and scope

**The problem.** SEP 20 L5 failed its 1% band tolerance. On-axis haCC(1) rows on the
60 fs window and on a separately driven 100 fs window differ by 4.61% (peak-power
plane) and 1.43% (maximum-UV plane) over 90 nm–1.6 µm, almost all of it above 900 nm.
The maintainer kept the 60 fs default with a recorded exception (SEP 20 Q30).

The supporting provider study is summarised, with its identities, in
`benchmarks/sep26/evidence/0133-summary.md`
(reference 5). It found three things:

- **The late current is box-converged.** Changing the box from 80 to 240 bohr changes
  the 0.9–1.6 µm source by 0.13%. That is a box-to-box sensitivity: no box is an
  independently valid late-time reference after its own norm loss.
- **The escaped-electron term dominates what the gate removes.** It accounts for 99.6%
  of the nir norm that the 60 fs gate removes from the 100 fs current: a near-steady
  drift of about $3.3\times10^{-3}$ au at +50 fs.
- **The exact split** (§4.1), applied to that same 100 fs current under both gates:

$$
\mathcal F[gS]=\frac{\mathcal F[gS']}{i\omega}+\frac{\mathcal F[g'S]}{i\omega}.
$$

Same-current split (both gates applied to the 100 fs current; the 60 fs-driven current
is not used):

| case | 60/100 fs gate difference, band: full → edge-free | nir (0.9–1.6 µm) | uv (90–200 nm) | edge term over own nir source, 60 / 100 fs |
|---|---|---|---|---|
| haCC(1) peak power | 4.60% → 1.22% | 12.5% → 3.33% | 0.387% → 0.387% | 10.6% / 11.0% |
| haCC(5) peak power | 4.57% → 1.22% | 12.4% → 3.33% | 0.384% → 0.384% | 10.6% / 10.9% |
| haCC(1) maximum UV | 1.44% → 1.45% | 4.36% → 4.42% | 0.349% → 0.349% | 1.7% / 2.2% |

What this shows:

- **Peak power.** Removing the edge term reduces the band difference by about 73%.
  Both gates carry an edge term of about 11% of their own nir source, so a longer
  window alone does not remove it.
- **Maximum UV.** The drift is ten times smaller, and the difference does not shrink.
- **What remains** is the window sensitivity left after the edge is removed. It is not
  a measured physical error, because there is no converged late-time reference.
- **The 90–200 nm difference** does not change when the edge term is removed.

The edge term is a property of the gate, not of the model: any gated source whose
current is nonzero in the trailing taper carries it, SAE included.

**Who needs it.** Every near-infrared result of a windowed or collared TDSE run:
haCC's run_short campaign, and SAE runs with `source_filter="time_collar"`.

**In scope.**

- The new source form on `numpy_ref`, with its discrete definition, admission
  condition, certificate and rounding budget.
- The record, descriptor and schema changes.
- Last-evaluation diagnostics.
- Validation: the L5 replay, an L6 segment, the SAE controls and an offline
  rate-plasma measurement.
- A separate, later default promotion.

**Out of scope.**

- The window default (Q4).
- An explicit drift-current model, collisions and recombination (§10).
- The unfiltered `source_filter="none"` route (§4.5).
- Any change to the rate model's plasma route; only its offline measurement is in
  scope (§4.6, Q3).
- Compiled backends: TDSE stays NumPy-only (SEP 12's recorded exception).

### 3. Context

The source assembly is `solprop.backends.numpy_ref.tdse`.

- **`atomic_sources`** forms the unscaled remainder $S=J-j_1$ for `tdse_native`. For
  `source_filter="time_collar"` it multiplies $S$ by `_source_gate(problem)`: the
  atomic window's gate (`AtomicWindowRecord.gate_values`) when a window is set, else
  `grids.t_window_fine`. It then applies density and `remainder_scale`.
- **`project_atomic_sources`**:
  - projects the nodal current with the modal weights and taper scale;
  - transforms it on the fine lattice and crops it to the coarse bins
    (`passes.to_frequency`);
  - multiplies by `current_numerator` over $\operatorname{Re}n_{\rm eff}$. The
    numerator is $-\text{crop\_scale}/(2\varepsilon_0c)$ on `active_mask` bins
    ($\omega>\omega_{\min}/2$) and zero elsewhere;
  - forms the embedded same-row diagnostic with an absolute rounding floor
    (`_roundoff_floor`).

  The caller applies the spectral collar `grids.omega_window` once. The last
  evaluation's diagnostic is exposed as `metadata()["tdse_last_evaluation"]`.
- **Gates** are `solprop.grid.planck_taper` profiles: an atomic window
  $(t_-,t_-+w,t_+-w,t_+)$ with $w=5$ fs by default (SEP 20 §4.6), or the grid's
  temporal `Collar`. `TDSEAtomicWindow` checks only that $0<w<T_a/2$, and
  `resolve_atomic_window` that at least four samples are retained. Neither checks
  that the tapers are resolved.
- **`TDSEGasResponse.source_filter`** resolves to `"time_collar"` for haCC and for any
  windowed response, and to `"none"` for an unwindowed SAE response. Descriptors
  `solprop.tdse-descriptor.v1`–`v3` record it (kernel contract, versions 18 and 27).
  An unwindowed SAE `"time_collar"` response uses v2, and v3 requires haCC or a
  window.
- **`admit_tdse`** refuses a duplicate Kerr, plasma or ionisation response beside a
  TDSE response.

Manual pages: [TDSE capillary](../../../guide/media/tdse.md),
[TDSE configuration](../../../guide/media/tdse.md) and the
[haCC response](../../../guide/media/tdse.md). Roadmap: the haCC item after SEP 20,
and the SAE width-sensitivity limit of the
native qualification (recorded in the development archive).

### 4. Theory

The transforms use solprop's internal FFT convention,
$\tilde f(\omega)=\int f(t)e^{-i\omega t}\,dt$ ([conventions.md](../../conventions.md) §2),
in which $\mathcal F[f']=i\omega\mathcal F[f]$. In the optics convention, replace
$i\omega$ by $-i\omega$ throughout. Four things are kept apart below: the continuous
identity (§4.1), the discrete definition (§4.4), its alias error, which is qualified by
admission and a certificate (§4.4), and floating-point error, which is budgeted (§4.5).

#### 4.1 The continuous identity and the new source

Let $g$ be a Planck gate with breakpoints $z_-<o_-\le o_+<z_+$, so that $g=0$ outside
$[z_-,z_+]$, and let $S$ be the remainder current. Since $gS$ vanishes at both ends of
its support, the product rule gives the exact identity

$$
\mathcal F[g\,S'](\omega)=i\omega\,\mathcal F[gS](\omega)-\mathcal F[g'S](\omega).
$$

**Continuous definition.** For $\omega>0$, the edge-free source is

$$
\boxed{\;\hat S_{\rm ef}(\omega)=\frac{\mathcal F[g\,S'](\omega)}{i\omega}
=\mathcal F[gS](\omega)-\frac{\mathcal F[g'S](\omega)}{i\omega}\;}
$$

That is today's source minus an *edge term*. Here $g'$ is the analytic derivative of the
gate, nonzero only strictly inside the tapers; $S$ itself is never differentiated.

**Hold interpretation.** Let $S_{\rm h}(t)=\int_{-\infty}^{t}g\,S'\,dt'$ and
$H=\int gS'\,dt$. Then $\mathcal F[S_{\rm h}]=\mathcal F[gS']/(i\omega)+\pi H\delta(\omega)$,
so $\hat S_{\rm ef}=\mathcal F[S_{\rm h}]$ at every $\omega\neq0$. The held current is:

- zero before the window;
- $S(t)-c_-$ on the plateau, with $c_-=\int g'_-S\,dt$ the leading-taper term (§4.3);
- held at $H$ after $z_+$.

The form keeps every time variation of $S$ inside the window. That includes the
turn-on of the drift current during the pulse, its Brunel-type emission (reference 3).
Only the switch-off is replaced by a hold. Three cases show what this means:

- **A current constant over the whole window** has $S'=0$, so $\hat S_{\rm ef}=0$.
- **A current that rises inside the plateau and is then steady**, for example
  $S=H\Theta(t-t_c)$ with $g(t_c)=1$, keeps its rise: $\hat S_{\rm ef}=He^{-i\omega
  t_c}/(i\omega)$. Only the trailing switch-off is lost. For constant $S$ on the
  trailing taper, $\mathcal F[g'_+S]=S\,\mathcal F[g'_+]$ (with $g'_+\le0$), and that
  is exactly the term subtracted.
- **A current that varies inside the trailing taper** keeps $gS'$ there. Its
  variation after the window is dropped, which is the window's finite-memory error;
  this SEP does not remove it.

#### 4.2 The gate derivative and its evaluation

On an edge with outer breakpoint $a$ and inner $b$, let $\Delta=b-a$ (negative on a
falling edge) and $s=(x-a)/\Delta\in(0,1)$. The gate is
$g=1/(1+e^{\zeta})$ with $\zeta=1/s-1/(1-s)$ (the `_taper_edge` exponent), and

$$
g'(x)=\frac{1}{\Delta}\,h(s),\qquad
h(s)=\frac{1}{4\cosh^2(\zeta/2)}\left(\frac{1}{s^2}+\frac{1}{(1-s)^2}\right)>0 .
$$

As an analytic function, $g'$ is $C^\infty$ and integrates to $\pm1$ over each edge; it
is zero outside the tapers and tends to zero at the breakpoints.

**Evaluation.**

- $s$ is formed first. If $\min(s,1-s)\le1/740$, the value returned is the asymptotic
  zero, and no reciprocal is formed. In that region $|h|<10^{-300}$, since
  $\cosh^{-2}(\zeta/2)\le4e^{-738}$ and the rational factor is at most
  $2\cdot740^2$.
- Otherwise $h$ is evaluated as written. All terms are finite: $|\zeta|<740$ and
  $s^{-2}\le740^2$.
- Points with $s\le0$ or $s\ge1$, including the exact breakpoints, return zero.

The evaluated function therefore differs from the analytic $g'$ by at most
$10^{-300}/|\Delta|$ in the cut-off region, and by a few rounding errors elsewhere
(§4.5). It never forms an infinite intermediate. The existing gate's clipping is not
reused for the derivative. `planck_taper_derivative` sits beside `planck_taper` in
`solprop.grid`, and both `AtomicWindowRecord` and `Collar` expose it.

#### 4.3 The leading-taper term

Let $W_-=\Delta t\sum_n|g'_-(t_n)|$ be the sampled weight of the leading edge (1 up to
sampling; T1 checks it). In raw DFT units (§4.4), on each active bin,

$$
\left|\frac{E_{-,k}}{i\omega_k}\right|\le\frac{W_-}{\Delta t\,\omega_k}\max_{t_n\in(z_-,o_-)}|S_n|,
\qquad
\left\|\frac{E_-}{i\omega}\right\|_{B}\le\frac{W_-}{\Delta t}\max_{\rm lead}|S_n|\Big(\sum_{k\in B}\omega_k^{-2}\Big)^{1/2}
$$

over a band $B$.

**Why it is small.**

- haCC and windowed SAE start at $t_s\ge z_-$ in the field-free ground state with
  $A(t_s)=0$, so $S(t_s)=0$.
- The drive in the leading taper is below 1% of its peak at the run_short planes
  (SEP 20 §4.6), and $S$ is at least cubic in it.
- An unwindowed collared SAE run has its leading collar at the lattice start.

Subtracting both edges is the parameter-free form, and it holds the pre-window value,
which is zero. The last-evaluation diagnostic records the ratio $\rho_-$ and this bound
(§5).

#### 4.4 The discrete definition, admission and certificate

**Lattice and normalisation.** The fine response lattice has $N$ samples
$t_n=(n-N/2)\Delta t$, period $T=N\Delta t$, $\omega_k=2\pi k/T$,
$\omega_s=2\pi/\Delta t$ and $\omega_N=\omega_s/2$. The raw DFT is
$x_k=\sum_n x_ne^{-2\pi ikn/N}$, with no $\Delta t$ weight and its phase origin at
$n=0$. For a smooth $x$ supported in one period, Poisson summation gives

$$
\Delta t\,x_k=(-1)^k\sum_m\mathcal F[x](\omega_k+m\omega_s),
\qquad (-1)^k=e^{-i\omega_kT/2}.
$$

**Discrete definition.** Let $X_k=\mathrm{DFT}[g(t_n)S_n]_k$, today's source, and
$E_k=\mathrm{DFT}[g'(t_n)S_n]_k$. The source is

$$
\hat S_{{\rm ef},k}=X_k-\frac{E_k}{i\omega_k}\quad\text{on active bins},\qquad 0\ \text{elsewhere}.
$$

The common factor $(-1)^k$ cancels between the two terms. Bin $k=0$ and every bin below
$\omega_{\min}/2$ lie outside `active_mask`, where `current_numerator` is already zero,
so no further cutoff is needed. The largest weight is $1/\omega_{\rm lo}\le2/\omega_{\min}$
at the lowest active bin. For 90 nm–1.6 µm, $\omega_{\min}=1.18\times10^{15}$ rad/s
(187 THz) and the active edge is at 94 THz (3.2 µm).

**Alias error.** If the samples are those of a smooth $S$, the $m=0$ terms reproduce
the continuous identity. The $m\ne0$ terms of $X$ are today's alias, unchanged. Those of
$E/(i\omega)$ are new, and they depend on how well the sampled taper resolves $g'$. They
are not small on every admitted lattice. With a taper shorter than a sample, no sample
of $g'$ need be nonzero, and the steady-current cancellation fails completely; a review
probe gave ratio 1.0 on such a lattice.

The asymptotic decay of a Planck taper's spectrum (super-algebraic, from its essential
singularity) motivates resolving the taper well. It is not used as a bound. Instead:

- **Admission.** `gated_derivative` requires each taper to span at least 128 lattice
  samples ($w\ge128\,\Delta t$), for the atomic window's tapers or the collar's.
  Translation refuses a narrower taper and names the minimum.
- **Steady-current certificate.** At translation, for the actual lattice, gate and
  active band $A$, compute

$$
\kappa_{\rm ss}=\frac{\big\|\mathrm{DFT}[g]-\mathrm{DFT}[g']/(i\omega)\big\|_A}{\|\mathrm{DFT}[g]\|_A}.
$$

  This is exactly the discrete source of $S\equiv1$ relative to today's. Translation
  refuses with a clear error if $\kappa_{\rm ss}>10^{-10}$. The value is recorded in
  the run metadata and the last-evaluation diagnostic, not in the descriptor, so the
  identity never depends on a floating-point result. The certificate is one transform
  of a fixed gate per configuration.

**What the certificate shows.** A drafting-time probe of the window gate on the haCC
lattice ($N=16384$, $\Delta t=6.254\times10^{-18}$ s, active band up to 80 nm) gave:

| taper samples | 10 | 20 | 40 | 56 | 64 | 96 | 128 | 800 (5 fs, the default) |
|---|---|---|---|---|---|---|---|---|
| $\kappa_{\rm ss}$ | 7e-5 | 4e-6 | 6e-9 | 9e-11 | 2e-11 | 3e-14 | 7e-16 | 6e-16 |

The SAE parent's collar (about 2000 samples per taper) gives 2e-16. The 128-sample
admission leaves margin below the threshold, and the certificate, not this table, is the
qualification. It certifies the steady part exactly. For a general current, T3 checks
the identity on resolved lattices against an independent spectral-derivative oracle,
with refinement.

#### 4.5 Rounding budget

The budget is absolute and operand-based, in the raw FFT units of the existing
diagnostics. It is carried through multiplication, projection, transform, crop,
division and subtraction. Let $c_F\varepsilon\log_2N$ be the normwise relative error
bound of the forward transform (reference 2). It is assumed for the 2-, 3- and 5-smooth
sizes solprop's lattices use. T3 checks the constant at a power-of-two size and at a
$2^a3^b5^c$ size; any other size is flagged in the diagnostic as having an unqualified
constant. By Cauchy–Schwarz, a perturbation $\delta x$ moves any single bin by at most
$\sqrt N\|\delta x\|_2$.

The source error per active bin is bounded by the sum of:

- **Today's budget.** The existing `_roundoff_floor`, from the raw operands
  $|g|(|J|+|j_1|)$, unchanged, so that shadow cancellation is accounted for.
- **The edge transform.**
  $c_F\varepsilon\log_2N\,\sqrt N\,\|g'(|J|+|j_1|)\|_2/\omega_{\rm lo}$, using the raw
  operands for the same reason.
- **Gate-derivative evaluation.** A relative error of at most $(8+|\zeta_n|)\varepsilon$
  at sample $n$. The 8 is §4.2's operation count; $|\zeta|$ amplifies the exponent's
  own rounding inside $\cosh$. This gives
  $\varepsilon\sqrt N\,\|(8+|\zeta|)\,g'(|J|+|j_1|)\|_2/\omega_{\rm lo}$.
- **The division and subtraction.** $2\varepsilon(|X_k|+|E_k|/\omega_k)$.

Each term is multiplied by the absolute projection weights, the density, the scale and
$|\text{current\_numerator}|$, as `_roundoff_floor` does today. Aliasing is not part of
this budget; it is governed by §4.4.

**The spectral-derivative oracle** (T3) has its own budget. Its intermediate
$i\omega\,\mathrm{DFT}[S]$ reaches $\omega_N$, so its terms are
$c_F\varepsilon\log_2N\sqrt N(\omega_N\|S\|_2+\|g\,\mathrm{IDFT}(i\omega\,\mathrm{DFT}S)\|_2)/\omega_{\rm lo}$.
T3's tolerance is the sum of the two budgets.

**Numerical differentiation, for comparison.** The supporting study's centred
differences have relative error about $(\omega\Delta t)^2/6$, which is
$3\times10^{-3}$ at 90 nm on the haCC lattice. Its split closed to 3.3–5.3×10⁻⁴ of the band
norm. The identity form needs no derivative of $S$.

#### 4.6 Consistency with the existing pipeline

- **Remainder and shadows.** $S$ is the unscaled $J-j_1$, formed first as today, and the
  edge term uses the same $S$. Density, `remainder_scale`, projection, taper scale, crop,
  `current_numerator`, $1/\operatorname{Re}n_{\rm eff}$ and the spectral collar are
  linear, and act identically on both terms.
- **Window and padding.** Outside the atomic window, $S$, $g$ and $g'$ are zero on the
  lattice. The gate's breakpoints are the record's, never moved to samples, and they may
  be off-grid and asymmetric. The drive taper $gE$ is unchanged.
- **The grid collar.** For an unwindowed gated response, $g$ is the collar, whose zeros
  lie inside or on the lattice, so §4.1 holds. The same admission and certificate apply.
- **`source_filter="none"`.** There is no gate, so this SEP does not touch it.

#### 4.7 Every electron once, and the rate-plasma measurement

- **The atomic current.** $S$ contains the atom's whole nonlinear current: the bound
  part, the ionisation current and, for haCC, the escaped term. The form keeps all of
  its in-window variation. It removes only the gate's switch-off, which is not an
  electron's response, and it adds no current.
- **The linear current.** $j_1$ stays subtracted. It is not gated or held, and it is
  carried by the linear index (κ₁, SEP 20 §4.8).
- **Kerr and plasma.** `tdse_native` contains its cubic. `optmat_cubic` and
  `tdse_shadow` admit only `"none"`. `admit_tdse` still refuses a rate plasma beside a
  TDSE response.

**The rate model.** In continuous time its plasma current $J_{\rm pl}=J_{\rm free}+J_{\rm loss}$
(Drude current plus the ionisation-loss current of `passes.plasma_current`) is constant
after the pulse (reference 4). solprop's discrete rate route, however, gates the
*polarisation* $P=\rho\int J_{\rm pl}\,dt$ with `t_window_fine` before its transform.
Writing $J=P'$, its source contains

$$
i\omega\,\mathcal F[gP]=\mathcal F[gJ]+\mathcal F[g'P],
$$

whereas the held-current form of the same current is
$\mathcal F[gJ]-\mathcal F[g'J]/(i\omega)$. Holding $P$ instead would hold a different
quantity.

**The offline measurement (Phase 2, reported only; the production rate route is
unchanged).** Recompute the rate model's nodal $J_{\rm pl}$ and $P$ with
`passes.plasma_scan` and `passes.plasma_current` on fixed saved fields: the saved
planes of SEP 20's L6 rate run, projected to its nodes. Use the run's own running-integral
rule (`responses.plasma_integral`) and density. Then form three things on the fine
lattice, with the route's crop and polarisation prefactor:

- $Q_{\rm route}=\mathrm{DFT}[gP]$;
- $Q_{\rm held}=(\mathrm{DFT}[gJ]-\mathrm{DFT}[g'J]/(i\omega))/(i\omega)$;
- the closure residual $\|i\omega\,\mathrm{DFT}[gP]-\mathrm{DFT}[gJ]-\mathrm{DFT}[g'P]\|$.

The closure residual measures how far the discrete running integral departs from
$P'=J$. Report $\|Q_{\rm route}-Q_{\rm held}\|/\|Q_{\rm route}\|$ by band, the closure
residual and the certificate of $g$. This is a fixed-field source comparison, not a
propagated trajectory.

### 5. Proposed interface

- **`TDSEGasResponse.source_filter`** gains `"gated_derivative"`, the gate of
  `"time_collar"` with the edge term subtracted (§4.1).
  - It is opt-in through Phases 1 and 2. The omitted value resolves exactly as today:
    `"time_collar"` for haCC and for any windowed response, and `"none"` for an
    unwindowed SAE response.
  - Promotion to the haCC and windowed default is Phase 3, after the conditions of §11.
  - It is admitted for `tdse_native` with any gate: windowed SAE, windowed haCC, and
    unwindowed SAE or haCC on the grid collar.
  - It is refused for a taper below the §4.4 admission or a certificate above
    $10^{-10}$.
- **`"time_collar"` keeps its formula and code path.** Its descriptor bytes,
  identities and resume keys are unchanged exactly. Recomputing its sources is
  required to agree with recorded values within the §4.5 budget; a same-build bitwise
  repeat may be recorded as evidence, but is not a requirement (SEP 25).
- **Last-evaluation diagnostic.** `tdse_last_evaluation["projection"]` gains an `edge`
  entry for a gated response, with the same projection, spectral collar and raw FFT
  modal units as the existing entries. It covers two bands, the grid's `active_mask`
  and `inband_mask`, and records:
  - the norms of the truncating source $c\,X$, the edge term $c\,E/(i\omega)$ and the
    corrected source $c\,\hat S_{\rm ef}$, where $c$ is the current factor times the
    collar;
  - the ratios edge/truncating and edge/corrected;
  - the §4.5 floor of each norm;
  - $\rho_-$, which is $\max_{\text{rows, lead samples}}|S|$ over $\max_{\text{rows, window}}|S|$;
  - the band bound of §4.3 and $\kappa_{\rm ss}$ (also in the run metadata).

  A ratio whose denominator is at or below its floor is `None`, with a reason string
  ("denominator below rounding floor"); $\rho_-$ is `None` with "zero remainder" when
  $\max|S|=0$. The admission excludes an empty sampled taper. For `"time_collar"` the
  same entry is computed, so that the two forms can be compared on one evaluation.
  These are diagnostics, never gates. Per-call history is not stored: per-call edge
  analysis is an offline replay of raw archives (§7.3, §7.5).

```text
response = replace(hacc_preset(1), source_filter="gated_derivative")  # opt-in (Phases 1-2)
```

### 6. Implementation strategies

**Where.** In `solprop.grid`, `planck_taper_derivative` (§4.2). In `numpy_ref`,
`atomic_sources` also returns the gated edge array $g'S$, scaled the same way, and
`project_atomic_sources`:

- projects, transforms and crops the edge array as it does the current;
- divides it by $i\omega$ on `active_mask` bins;
- subtracts it before `current_numerator`;
- extends the embedded coarse projection, `_roundoff_floor` and the `edge` diagnostic.

The admission and certificate live in translation, beside `resolve_atomic_window` and
the collar resolution.

**Options.**

- **(a) Frequency-domain subtraction, as above (recommended).** It costs one extra
  fine FFT of one modal column per call (two with the embedded map), and $S$ is never
  differentiated.
- **(b) A time-domain held current.** This needs a cumulative quadrature and a periodic
  ramp. It is less accurate and not simpler.
- **(c) Differentiate $S$.** Centred differences have an $\mathcal O((\omega\Delta t)^2)$
  error, and a spectral derivative amplifies rounding by $\omega_N/\omega$. Both serve
  only as test oracles.

**Records and seam.**

- **Descriptor v4.** `"gated_derivative"` is recorded in a new
  `solprop.tdse-descriptor.v4`. It has v3's layout, with `model` and `atomic_window`
  (which may be `null`), plus the new value, so it covers unwindowed SAE on the collar
  as well as windowed SAE and haCC.
- **v1–v3** stay unchanged byte for byte, and each refuses `"gated_derivative"`.
  `_validate_filter_schema` admits v4 only with that value.
- **`SCHEMA_VERSION`** goes from 29 to 30 (kernel-contract row 30). No field moves, and
  a version-29 consumer refuses the problem. solprop-kokkos accepts version 30 and
  keeps refusing TDSE by name.
- **Output schema.** No bump: `source_filter` is already in the resume key whenever it
  is not `"none"`, and the last-evaluation metadata is not part of the archive schema.
  The κ₁ cache is keyed by basis identity and is unaffected.

**Backends.** NumPy only, as for every TDSE route; `numba_cpu` and Kokkos refuse TDSE
by name. There is no GPU work.

**Cost.** Each call adds one fine FFT, one nodal array ($n_{\rm nodes}\times N$) and
the extra diagnostic work. Phase 1 measures this with a bounded hot-path benchmark and
memory ledger (§7.2) before calling it negligible.

**Dependencies.** None.

### 7. Verification and validation

Equality is exact only for descriptor bytes, identities, resume keys, losslessly copied
arrays, masks and call accounting. Matched source recomputations use the §4.5 budgets.
Propagation and resume comparisons use accuracy envelopes (SEP 25). Each tolerance is
derived before evaluation and stated beside its assertion. "Raw units" means the
unweighted DFT of §4.4. Per-bin bounds hold on every active bin; band norms are 2-norms
over the stated bins.

#### 7.1 Analytic and unit tests (default suite, fast)

- **T1 The gate derivative.**
  - Against the independent form $-g(1-g)\,d\zeta/dx$, built from `planck_taper`'s own
    $g$ where $0.01<g<0.99$ (so $g(1-g)$ is not cancellation-limited), the tolerance
    per sample is the §4.5 evaluation budget plus the gate's own
    $(4+|\zeta|)\varepsilon$.
  - Against a fourth-order centred difference of `planck_taper` at two spacings, the
    error must fall by $16\pm2\times$ between them. That is a convergence check, not a
    bound from a sampled derivative.
  - The sampled weight $\Delta t\sum|g'|$ of each edge is reported against 1 at 64, 128
    and 800 samples per taper. The 128- and 800-sample cases agree within the
    certificate threshold. No exponential-convergence claim is made.
  - Values are finite and non-negative in $h$ on both sides of every breakpoint, at the
    exact breakpoints, at `nextafter` neighbours and at the §4.2 cut-off. Widths run from
    $10^{-30}$ s to 1 s, and include SI femtosecond scales. Warnings are errors.
- **T2 Steady and non-steady currents.** The tolerance is the §4.5 budget (absolute,
  raw units) plus $\kappa_{\rm ss}\|X\|_A$.
  - (a) $S\equiv H$ on the whole lattice: $\hat S_{\rm ef}=0$, while the truncating form
    is nonzero.
  - (b) $S=0$: both forms give exactly zero.
  - (c) An interior smooth rise followed by a steady tail keeps the analytic rise
    spectrum (T4's oracle) and differs from (a)'s truncating source.
  - (d) A current with a slope through the trailing taper keeps $gS'$ there: the result
    equals the spectral-derivative oracle (T3).
- **T3 Identity against a spectral derivative, with refinement.** Use a band-limited
  synthetic $S$: a smooth rise, a steady drift, a Gaussian-envelope oscillation and a
  slope through the trailing taper. Compare
  $\mathrm{DFT}[g\cdot\mathrm{IDFT}(i\omega\,\mathrm{DFT}S)]/(i\omega)$ with §4.4's form
  on active bins. The tolerance is the sum of the two §4.5 budgets. Cases:
  - asymmetric and off-grid breakpoints;
  - tapers at the admission limit and at 2×, 4× and 8× it, where agreement must not
    worsen as the taper resolution grows;
  - the highest retained frequency inside the band;
  - a power-of-two $N$ and a $2^a3^b5^c$ $N$.

  The same synthetic case also records the centred-difference split's
  $\mathcal O((\omega\Delta t)^2)$ miss.
- **T4 The analytic held step.** Take $S=\tfrac H2[1+\operatorname{erf}((t-t_c)/\sigma)]$,
  with the rise on the plateau at distance $d$ from both plateau edges. In raw units,
  the oracle on active bins is

$$
\frac{(-1)^k}{\Delta t}\,\frac{He^{-\omega_k^2\sigma^2/4}e^{-i\omega_kt_c}}{i\omega_k}.
$$

  The tolerance, per bin, is the sum of three terms:
  - the rise's mass outside the plateau, $H\operatorname{erfc}(d/\sigma)/(\Delta t\,\omega_k)$;
  - the alias $He^{-(\omega_N\sigma)^2/4}/(\Delta t\,\omega_k)$;
  - the §4.5 budget.
- **T5 A known oscillation is unchanged.** Take $S=A\cos(\omega_0t)e^{-(t/\tau)^2}$
  inside the plateau, with $t_p$ the distance from its centre to the nearest taper. Per
  bin in raw units, the two forms differ by at most
  $A e^{-(t_p/\tau)^2}W/(\Delta t\,\omega_k)$ plus the budget. Here $W=\Delta t\sum|g'|$.
- **T6 The leading taper.** An $S$ with known values in the leading taper gives the
  recorded $\rho_-$ and satisfies both §4.3 bounds.
- **T7 Admission and certificate.**
  - The review's counterexample is refused: $N=16384$, $\Delta t=6.254\times10^{-18}$ s,
    an $80\Delta t$ window and a $\Delta t/4$ taper.
  - So are a 127-sample taper and an under-resolved collar.
  - A 128-sample taper is admitted, with $\kappa_{\rm ss}\le10^{-10}$ recorded in the
    metadata.
  - The recorded certificate agrees with an independent recomputation in the test
    within $10\varepsilon\log_2N$ relative. It does not enter the descriptor or the
    identity.

#### 7.2 Fake-provider tests and the benchmark (default suite)

These use the existing SAE and haCC fake providers.

- **A fake post-pulse drift.** A fake current that rises inside the plateau and then
  stays steady through the trailing taper. Under `"gated_derivative"` the source keeps
  the rise spectrum and has no switch-off: it matches T2(c) within the budget, and the
  edge diagnostic's edge/truncating ratio equals T2(c)'s. Under `"time_collar"` it
  matches today's formula within the budget.
- **Unchanged routes.** For `"time_collar"` and `"none"`, descriptors, identities and
  resume keys are exactly unchanged, and their sources match stored witnesses (with
  provenance) within the §4.5 budget.
- **Resolution and refusals.**
  - The omitted filter resolves as today (Phases 1–2).
  - `"gated_derivative"` is refused with `optmat_cubic` or `tdse_shadow`, and with an
    under-resolved taper.
  - A window with `"none"` is refused.
  - v1–v3 refuse the new value, v4 refuses the old values, and a version-29 consumer
    refuses a version-30 problem.
- **Identity and resume.** Write a v4 descriptor and round-trip and resume it for each
  of: unwindowed SAE on the collar, windowed SAE, windowed haCC and unwindowed haCC. A
  `"time_collar"` run refuses to resume under the new filter, and the reverse. Every
  historical v1–v3 case still validates.
- **Every electron once.** The source equals $\mathrm{DFT}[gS]-\mathrm{DFT}[g'S]/(i\omega)$
  with the same $S$, density and scale, within the budget; no term enters twice, and
  $j_1$ never enters.
- **Diagnostics and calls.**
  - The `edge` entry holds the norms, floors and `None`-with-reason cases of §5.
  - Its values change with the last evaluation and keep no history.
  - The atomic ledger is exactly the same as under `"time_collar"`: no extra atomic
    calls.
- **Hot-path benchmark** (benchmark lane, bounded). Time `atomic_sources` plus
  `project_atomic_sources` under both filters on a fake-provider problem at the haCC
  lattice size, with warmup and interleaved repeats, reporting the distribution. Record
  the extra arrays in the memory ledger.

#### 7.3 The L5 replay (offline, no new atomic calls)

`benchmarks/sep20/window.py analyse` gains a filter option. It reads the archived
Phase 3 currents, hashed in `benchmarks/sep20/evidence/manifest.json`:

- the L4 candidate rows, which are the separately driven 60 fs current;
- the 100 fs reference.

It writes a derived analysis record. That record keeps each raw current's descriptor
and identity, and names the replay filter and analysis policy; the raw atomic
evaluation is not relabelled. Three comparisons are reported separately.

- **(i) Recorded figures.** `"time_collar"` reproduces the recorded L5 figures within
  the §4.5 budget, propagated through the ratio.
- **(ii) Same-current split, the anchor.** Both gates are applied to the 100 fs
  current. The metric is the dimensionless ratio
  $\delta=\|F_{100}-F_{60}\|_B/\|F_{100}\|_B$, with $F$ the edge-free source and $B$ the
  band.
  - The expected values are those of §2: 1.22% band and 3.33% nir at peak power, and
    1.45% and 4.42% at maximum UV.
  - The supporting figures used centred differences. Their split residual is
    $r\le5.4\times10^{-4}$ of the band norm (3.3–5.3×10⁻⁴ measured). Let $e_g$ be the replay's measured
    difference between the analytic and the centred-difference $g'$ edge terms,
    relative to the band norm.
  - The requirement is $|\delta-\delta_{\rm ref}|\le(2+\delta)(r+e_g)$ in the band. That
    is about $1.1\times10^{-3}$ (0.11 percentage points) for $e_g\le10^{-5}$. nir and uv
    are reported.
- **(iii) Actual L5 with the new filter.** The separately driven 60 fs current is
  compared with the 100 fs current, both edge-free. This is reported, with no tolerance
  borrowed from (ii).
  - Expectation: close to (ii), plus the startup and drive difference that L5's
    common-window comparison measured (0.036–0.046% band). The actual peak-power UV
    figure is about 0.405%, against 0.387% for the same-current split.
  - It remains a recorded failure of L5's 1% band tolerance if it exceeds 1%.

#### 7.4 L6 segment (CPU nodes; no GPU rental needed)

From SEP 20's L6 launch and pump, with haCC(1) fast at `rtol=1e-2`, 32-thread runs, the
provider pin of the existing CPU segment record and the full TDSE archive. Execution
details are in `benchmarks/sep26/README.md`.

**Runs.**

- (a) `"gated_derivative"` at 60 fs, about 1 h.
- (b) `"gated_derivative"` at 100 fs, about 1.7 h.
- (c) `"time_collar"` at 100 fs, about 1.7 h.
- The existing `"time_collar"` 60 fs CPU segment is the current form. If the pin has
  moved since it was recorded, rerun it.

These total 140–175 core-hours.

**Metric.** At the common endpoint, compare complex spectra by band (full, 90–200 nm,
0.9–1.2 µm and 1.2–1.6 µm) as $\|a-b\|_B/\|b\|_B$, with $b$ the current-form endpoint
(as in SEP 20 L6). The floor is the two runs' interior-restart agreement on the same
segment, the 0.073% full-band fresh-history figure of SEP 20 L6. A difference below that
floor is reported as unresolved.

**Gates.** L6's mechanical and sensible-optics gates.

**Investigation threshold (maintainer, R6).** A 90–200 nm difference between (a) and the
current form above 0.5% is investigated, not failed. The cause is named by three checks:

- matched-input sources at a few saved planes, which separate the direct source change
  from the propagated one;
- pinned scientific inputs, so that only the filter differs;
- a tighter-`rtol` control of the pair (`rtol=3e-3`), which separates integration error
  from propagated physical sensitivity.

`rtol=1e-2` is a local tolerance and does not certify a 0.5% global comparison.

**Reported.**

- Every pair's band differences: (a) against the current form, (a) against (b), and
  (c) against the current form.
- Energy, peak power, calls and wall time.

**Expectation, not a result.** The (a)–(b) difference above 900 nm should be well below
the current form's 60-against-100 fs difference.

#### 7.5 SAE controls

- **L6 per-call archives** (the `sae-full` and `sae-window` segment archives, hashed in
  the manifest).
  - Recompute each call's source under both forms offline, with the certificate and the
    T3 oracle on the real currents.
  - Report the edge fractions by band.
  - The first calls share the launch field, which gives a single-field
    collar-against-window comparison.
- **SEP 12's saved-current width study.** Replay
  `benchmarks/sep12/evidence/native-qualification/source-filter-study.py` with the new
  form, on its hashed external archive (retrieval in `benchmarks/sep26/README.md`).
  Whether the 0.75/1.25 width sensitivities fall is a hypothesis to test.
- **One SAE propagation.** Rerun the L6 `sae-window` segment (60 fs, 75–93 core-hours)
  with the new form, and report it as in §7.4.

**Pass for the SAE controls**, which the SAE research record needs before it moves:

- the replays reproduce their recorded `"time_collar"` figures within budget;
- every certificate is at most $10^{-10}$;
- the SAE rerun passes L6's mechanical and sensible-optics gates;
- the edge-free width sensitivity is no larger than the truncating one in any band.

If any of these fails, the SAE research record stays on `"time_collar"` and the finding
is reported. Recorded SAE results keep their identities and are not relabelled;
unwindowed SAE keeps `"none"` as its default.

### 8. Documentation

- **[haCC response](../../../guide/media/tdse.md).** The identity, the hold
  interpretation, admission and certificate, the opt-in, and, after Phase 2, Q30
  restated from §7's measured values.
- **[TDSE capillary](../../../guide/media/tdse.md).** The derivation and a worked
  comparison of the two forms on a synthetic drift, as a tested guide block.
- **[TDSE configuration](../../../guide/media/tdse.md),
  [resuming](../../../guide/results/resuming.md), [kernel contract](../../kernel-contract.md)**
  (row 30, descriptor v4) **and the [Luna differences](../../../guide/workflows/luna.md)**
  source-filter paragraph.
- Docstrings, the changelog citing SEP 26, the roadmap, and `benchmarks/sep26/`.

### 9. Compatibility and impact

- **Phases 1–2 change no default.** Nothing changes unless a response names
  `"gated_derivative"`. Opting in changes the near-infrared source by about 11% of the
  0.9–1.6 µm source at the peak-power plane and about 2% at maximum UV (same-current
  edge fractions, §2).
- **Phase 3** makes it the haCC and windowed default, if §11's conditions hold. New runs
  then get v4 descriptors; old files resume with their recorded filter.
- **Seam.** `SCHEMA_VERSION` 30 and descriptor v4. No output-schema bump (§6).
  `"time_collar"` keeps its formula, path, identities and resume keys exactly, and its
  numbers within the rounding budgets (SEP 25).
- **Q30 (expectation, not a result).** The exception should reduce to the window
  sensitivity that remains without the edge term: about 1.2% of the band and 3.3% of
  the nir at peak power, and 1.4–1.5% and 4.4% at maximum UV, in the same-current split.
  The actual L5 figures (§7.3 iii) decide it.
- **Luna** has no TDSE source; only the existing paragraph is updated.

### 10. Alternatives considered

- **Option 2, an explicit drift current.** The provider would deliver the escaped
  term's steady part, and solprop would propagate it ungated. It is rejected for now:
  - it needs work in both projects;
  - the steady part is an arbitrary choice with its own turn-on edge;
  - for a collisionless model it gives nothing that the hold does not already give at
    $\omega\neq0$.

  Reconsider it for collisions or recombination, where the drift decays and its decay
  radiates.
- **A longer default window.** About 1.67× the steps, and it still carries an edge term
  of the same size.
- **Fitting and subtracting a steady part.** Arbitrary, and it has its own edge.
- **Differentiating $S$ numerically.** Less accurate or rounding-amplified (§4.5); used
  only as oracles.
- **Band-first filtering.** A spectral mask cannot remove a step's in-band content.
- **Redefining `"time_collar"` in place.** That would invalidate recorded identities.

### 11. Phases and milestones

1. **Phase 1, implementation, opt-in** (`feature/sep-0026-phase-1`, plus a
   schema-acceptance commit in solprop-kokkos).
   - §4.2's derivative, the edge-free assembly, the admission and certificate, and the
     last-evaluation diagnostic.
   - Descriptor v4 and `SCHEMA_VERSION` 30.
   - T1–T7, §7.2's tests and the bounded hot-path benchmark with its memory ledger.
   - Coverage of the touched code, the manual pages and the changelog.
   - Gate: the per-phase checks plus the full gate, because the schema changes.
2. **Phase 2, validation** (`feature/sep-0026-validation`).
   - §7.3 offline (minutes) and §7.5's offline SAE controls.
   - §7.4's L6 segments and the SAE rerun on CPU nodes, with no GPU rental (about
     215–270 core-hours in all).
   - §4.7's offline rate-plasma measurement.
   - Each result is reported in Part II.
3. **Phase 3, promotion and final gate.**
   - If §11's promotion conditions hold, make `"gated_derivative"` the haCC and
     windowed default, and move the SAE record if §7.5 passed.
   - Restate Q30 in the manual and the roadmap (SEP 20 is Final and takes only an
     erratum note).
   - Run the end-of-SEP full gate, the slow lane, the coverage lane and the Kokkos CPU
     lane (schema acceptance and refusal).

**Blocking conditions.**

- **Acceptance:** this revision's response to R1–R13 is reviewed, and Q4 is recorded as
  deferred to Phase 3.
- **Default promotion:** Phase 2 has passed, meaning T1–T7, §7.3 (i)–(ii), §7.4's gates
  with every investigation closed, and a certificate of at most $10^{-10}$ on every
  replayed configuration. The maintainer has also decided Q4 and Q30's remaining
  exception from §7.3 (iii) and §7.4.
- **Final:** every phase is merged and reported, the end-of-SEP lanes pass, and the
  manual is updated.

### 12. Open questions

Resolved by the maintainer on 2026-10-18:

- **Q1 Naming and the old form.** The new value is `"gated_derivative"`.
  `"time_collar"` stays as the truncating comparison form, with its formula and code
  path. Its descriptor bytes, identities and resume keys are exactly unchanged, and its
  numerical recomputation agrees within rounding budgets (SEP 25).
- **Q2 SAE.** The SAE research record moves to the new form only if §7.5's pass
  conditions hold; otherwise it stays on `"time_collar"`. One SAE L6 rerun is funded.
  Unwindowed SAE keeps `"none"`.
- **Q3 The rate plasma.** This SEP does not change it. Phase 2 measures it offline
  (§4.7), and the measured number decides whether a follow-up SEP is needed.
- **Q5 L6 scope.** All three segments.
- **Q6 The SEP 12 archive.** Located. It is identified by hash, and its retrieval is
  described in `benchmarks/sep26/README.md`.

Deferred to Phase 3:

- **Q4 Q30's exception and the default.** Whether to promote `"gated_derivative"`
  (R11), whether the 60 fs default window stays, and how Q30's remaining exception is
  restated. The maintainer decides from §7.3 (iii) and §7.4.

### 13. References

1. D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash, "A tapering window for
   time-domain templates and simulated signals in the detection of gravitational waves
   from coalescing compact binaries," *Class. Quantum Grav.* **27**, 084020 (2010),
   eq. (7). doi:10.1088/0264-9381/27/8/084020
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM
   (2002), ch. 24 (the FFT's normwise rounding error). doi:10.1137/1.9780898718027
3. F. Brunel, "Harmonic generation due to plasma effects in a gas undergoing
   multiphoton ionization in the high-intensity limit," *J. Opt. Soc. Am. B* **7**,
   521–526 (1990). doi:10.1364/JOSAB.7.000521
4. M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec, "Light
   propagation in field-ionizing media: extreme nonlinear optics," *Phys. Rev. Lett.*
   **83**, 2930 (1999). doi:10.1103/PhysRevLett.83.2930
5. SEP 26 evidence summary of the provider's late-current study,
   `benchmarks/sep26/evidence/0133-summary.md` in the solprop repository:
   its equations, metric, tables, limitations and the SHA-256 identities of its
   records. For provenance, it is Pyroraptor report 0133 and its taper-edge addendum
   (2026-10-18).

## Part II — Implementation record

### Phase 1 report — implementation, opt-in (2026-10-18)

**Scope.** Phase 1 of §11: §4.2's derivative, the edge-free assembly, the admission and
certificate, the last-evaluation diagnostic, descriptor v4 and kernel schema 30, T1–T7,
§7.2's tests, the bounded hot-path benchmark and the manual. Implementer: Opus 5.5 (AI) for John Travers. Branches
`feature/sep-0026-phase-1` in solprop (from `e1a8274`; `main` at `b20697f`, SEP 22
phase 1, merged in at `d24ca70`) and in solprop-kokkos (from `2c8b9ca`: `3462b62`,
`de2a805`). Merge commits are the coordinating agent's to record on integration.

**Delivered.**

- `src/solprop/grid.py`: `planck_taper_derivative`, `planck_taper_derivative_error`,
  `Collar.window_derivative`.
- `src/solprop/nonlinear/tdse.py`: `source_filter="gated_derivative"`, opt-in; an
  omitted filter resolves as before; a window refuses only `"none"`.
- `src/solprop/translate/tdse.py`: descriptor v4 (version 3's layout, recording only
  `gated_derivative`; versions 1–3 refuse it and version 4 refuses every other filter);
  the 128-sample admission (`GATED_DERIVATIVE_TAPER_SAMPLES`), checked when a descriptor
  is formed and on every reconstructed record; `AtomicWindowRecord.gate_derivative`,
  `TDSEDescriptor.source_breakpoints`, `source_gate_derivative` and
  `source_gate_derivative_error`.
- `src/solprop/translate/tdse_source.py`: `steady_current_certificate`, cached per
  configuration and band, and `STEADY_CURRENT_CERTIFICATE_LIMIT`.
- `src/solprop/translate/problem.py`: `SCHEMA_VERSION` 30; `KernelProblem` validation
  refuses a certificate above 1e-10 and checks the recorded collar for both gated filters.
- `src/solprop/backends/numpy_ref/tdse.py`: `atomic_edge_source` (the scaled `g'S`), the
  edge-free projection `project_atomic_sources(..., edge=)`, the §4.5 budget
  (`FFT_ERROR_CONSTANT`) and the `edge` diagnostic. `backend.py` passes the edge array and
  records `metadata()["tdse_source"]`; `backends/memory.py` adds the `tdse_edge_sources`
  ledger row.
- Tests: `tests/test_planck_taper_derivative.py` (T1);
  `tests/backends/test_tdse_edge_free.py` (T2–T7, every electron once, the seam's
  refusals); `tests/backends/test_tdse_unchanged_routes.py` with the witness
  `tests/data/sep26_unchanged_sources.{npz,json}`, written from `e1a8274` by
  `tests/data/generate_sep26_unchanged_sources.py`;
  `tests/translate/test_tdse_descriptor_v4.py`;
  `tests/driver/test_tdse_gated_derivative_run.py` (the manufactured SAE binding and the
  fake haCC provider); `tests/edge_source_cases.py`; the schema literals of five existing
  modules.
- Benchmark: `benchmarks/sep26/hot_path.py`, its record
  `benchmarks/sep26/evidence/hot-path.json`, the ledger entry in `docs/dev/benchmarks.md`,
  and how to run it in `benchmarks/sep26/README.md`.
- Manual: `docs/guide/media/tdse-hacc.md` (identity, hold, admission, certificates,
  diagnostic, opt-in), `tdse-capillary.md` (the derivation and a tested worked comparison
  on a synthetic drift), `tdse-configuration.md` (the filter, its refusals, a tested
  example), `docs/guide/results/resuming.md`, `docs/guide/workflows/luna.md`,
  `docs/dev/kernel-contract.md` (row 30, descriptor v4; its stale "saved-output schema:
  19" header, left by SEP 22, now reads 20). Each page produced or extended ends with a
  link to this SEP. `CHANGELOG.md`.
- solprop-kokkos: `3462b62` accepts schema 30 and keeps refusing TDSE by name (its
  refusal test now includes `gated_derivative`); `de2a805` updates its last schema
  literal.

**Deviations from Part I, and why.**

1. *Evaluating $g'$ (§4.2).* $1/(4\cosh^2(\zeta/2))$ is evaluated as $e/(1+e)^2$ with
   $e=e^{-|\zeta|}$: written as $\cosh^2$ it overflows for $|\zeta|>709.8$, inside the
   admitted $|\zeta|<740$. $s$ and $r=1-s$ are both formed from $x$, as $(x-a)/\Delta$ and
   $(b-x)/\Delta$, so that neither comes from a cancellation. The analytic function, the
   cut-off at $\min(s,1-s)\le1/740$ and the zeros at and beyond the breakpoints are as
   specified.
2. *The derivative's rounding bound (§4.5).* A first-order count of that evaluation gives
   $(17+2.5|\zeta|)\varepsilon$, used as $(18+3|\zeta|)\varepsilon$; §4.5's
   $(8+|\zeta|)\varepsilon$ came from a shorter count. Near a breakpoint
   $\zeta\approx1/s$ carries $s$'s own relative rounding times $|\zeta|$, which no
   evaluation avoids.
3. *T1's oracle tolerance.* The closed form $-g(1-g)\,d\zeta/dx$ takes $1-g$ from a
   second gate evaluation at the mirrored point $a+b-x$ (exact on the test's dyadic
   lattice): forming $1-g$ from $g$ amplifies its rounding up to 100-fold where
   $0.01<g<0.99$. The oracle's budget is derived as $(20+4|\zeta|)\varepsilon$ (two gate
   evaluations at $(8+2|\zeta|)\varepsilon$ each by the same count, $d\zeta/dx$ and the
   products), in place of "the gate's own $(4+|\zeta|)\varepsilon$".
4. *The edge-transform budget term (§4.5)* also carries the rounding before the
   transform, $(n_{\rm nodes}+6)u$ of the projected raw operands (the subtraction
   $J-j_1$, the products by $g'$, density and scale, the taper division and the node
   reduction), which §4.5 does not list. $c_F=8$ (`FFT_ERROR_CONSTANT`) starts from
   Higham's radix-2 constant, about 3.4, with room for radix-3 and radix-5 levels; T3
   measured 0.092 and 0.106.
5. *T4's tolerance* adds a fourth term, the per-bin steady-current residual $H|r_k|$
   (the discrete edge-free source of $S\equiv H$), for the trailing taper's alias, which
   §4.4 governs and the three printed terms omit. It is about $10^{-16}$ of the source.
6. *T7's agreement.* At admitted resolutions $\kappa_{\rm ss}$ is rounding
   ($3.9\times10^{-16}$, against $5.2\times10^{-16}$ from an independent `numpy.fft`
   recomputation), so two transforms cannot agree relatively. The test reads "within
   $10\varepsilon\log_2N$ relative" as absolute agreement in $\kappa_{\rm ss}$, which is
   itself relative to $\lVert\mathrm{DFT}[g]\rVert_A$: $1.3\times10^{-16}$ against
   $2.7\times10^{-14}$. On an alias-dominated `time_collar` gate (20 samples,
   $\kappa_{\rm ss}=8.3\times10^{-6}$) the two agree to $4.8\times10^{-17}$ absolutely
   but only $5.8\times10^{-12}$ relatively: the numerator cancels terms the size of the
   denominator, so relative agreement is about $\varepsilon/\kappa_{\rm ss}$ at best.
7. *Interface (§6).* `atomic_sources` keeps its two-tuple; the edge array comes from the
   new `atomic_edge_source`, with the same $S$, density and scale, instead of a third
   return value. `project_atomic_sources` takes `edge=` keyword-only: required for
   `gated_derivative`, refused for `"none"` and optional for `"time_collar"`, which then
   has no `edge` entry. The backend always passes it.
8. *The certificate is also computed for `"time_collar"`* and recorded with
   `certificate_enforced: false`, so that the two forms compare on one evaluation, as §5
   asks of the diagnostic. It is never enforced there.

**Decisions left open by Part I.** The run-metadata record is
`metadata()["tdse_source"]`: the filter, the certificate (`None` if the band has no
truncating content), its limit, whether it is enforced and the gate's breakpoints. The
`edge` entry holds `source_filter`, `units`, `steady_current_certificate`,
`fft_constant_qualified` (a 2-, 3- and 5-smooth size); per band (`active`, `inband`) the
bin count, the three norms, their floors and both ratios with their reasons (the
corrected floor is the truncating floor plus the edge floor, which includes the division
and subtraction); and `leading_taper` (`ratio`, `reason`, `lead_samples`,
`sampled_weight` and the per-band bound). The band bound uses the absolute projection of
$|\rho sS|$, an upper bound on the modal current. The §4.5 terms use $1/\omega_{\rm lo}$
on every bin, as printed. No ADR: nothing here outlasts the SEP's own record.

**Verification.** On an 18-core Apple M5 Max (macOS 27, arm64), Python 3.14.5,
NumPy 2.5.2, SciPy 1.18.1; xdist bounded to 8 workers, one thread per worker
(`SOLPROP_KOKKOS_THREADS=1`, `OMP_PROC_BIND=false`); solprop-kokkos built from `de2a805`
against `d24ca70` (CPU, OpenMP, local profile, Release) and installed in the worktree
environment. The machine was shared (load average 5–8) throughout.

*Gate*, on `cd16e74` (the merge `d24ca70` and five test, figure and docstring
commits; the SEP text was uncommitted):

- `uv run poe check`: green (ruff, basedpyright, numpydoc, codespell, reuse; the
  default lane 15830 passed, 29 skipped; strict Sphinx). Its first run found the
  `one_run` structure figure stale at schema 29, which `poe figures` redrew.
- `uv run poe cov`: 100% of statements and branches (41097 and 12438). Its first run
  missed one branch, the empty leading taper, which a test now covers.
- `uv run poe test-slow`: the functional slow tests 147 passed, 1 skipped (on
  `d24ca70` plus the figure). The serial statistics-cost check then measured 27.2%
  against its 25% gate at a load average near 8; rerun alone at load 9–11 it passed.
  Nothing in this phase touches the statistics channel, so this was machine load.
- Kokkos-selected default lane (`SOLPROP_TEST_BACKEND=kokkos uv run poe test`): 15829
  passed, 29 skipped.
- solprop-kokkos: `tests/test_api_contract.py` and `tests/test_backend.py`, then its
  whole Python suite, 319 passed and 5 skipped, against the editable build.

Results, each tolerance derived before the comparison and stated beside its assertion:

| test | claim | measured |
|---|---|---|
| T1 closed form | within the derivative's and the oracle's bounds | at most $2.0\varepsilon$ relative, against at least $38\varepsilon$ |
| T1 fourth-order difference | the error falls $16\pm2$-fold | 15.96 ($2.46\times10^{-6}\to1.54\times10^{-7}$) |
| T1 edge weight | 128 and 800 samples agree within $10^{-10}$ | 64 samples $1+2.9\times10^{-11}$; 128 and 800 samples 1 to rounding |
| T1 finiteness | finite, $h\ge0$, exact zeros, no floating-point exception | widths $10^{-30}$ s to 1 s; every breakpoint, `nextafter` and cut-off neighbour |
| T2 (a) steady | $\lVert\hat S_{\rm ef}\rVert_A\le$ budget $+\kappa_{\rm ss}\lVert X\rVert_A$ | $6\times10^{-21}$ and $7\times10^{-21}$ against $2.6\times10^{-17}$ and $4.3\times10^{-17}$ (window, collar); truncating $1.4\times10^{-5}$ and $9.3\times10^{-7}$ |
| T2 (b) zero | exactly zero under both forms | exact; ratios `None` with their reasons |
| T2 (c), T4 held step | per bin within the four terms | at most $10^{-3}$ of the tolerance; $3.7\times10^{-16}$ relative |
| T3 identity | per bin within both budgets; no worse with resolution | at most $4.5\times10^{-6}$ of the tolerance; $3.5$–$8.3\times10^{-16}$ (4096) and $4.2$–$5.1\times10^{-16}$ (3840) relative, tapers of 128–1024 samples |
| T3 FFT constant | observed $\le c_F=8$ | 0.092 (4096) and 0.106 (3840 $=2^8\cdot3\cdot5$) |
| T3 centred difference | its miss is recorded | $1.7\times10^{-3}$ against $(\omega_{\max}\Delta t)^2/6=3.4\times10^{-3}$ |
| T5 oscillation | the forms differ by at most $Ae^{-(t_p/\tau)^2}W/(\Delta t\,\omega_k)$ plus the budget | at most $1.2\times10^{-3}$ of it |
| T6 leading taper | the recorded $\rho_-$; both §4.3 bounds | $\rho_-=0.0200$ to one rounding; 0.58 of the per-bin bound, 0.47 and 0.33 of the band bounds |
| T7 admission | the counterexample, 127 samples and a 121-sample collar refused; 128 admitted | as stated; $\kappa_{\rm ss}=3.9\times10^{-16}$ at 128 samples, in `tdse_source` |
| T7 certificate | an admitted taper with a failing certificate is refused | 20 fs (400-sample) taper, 90–200 nm, $q=1$: $1.8\times10^{-10}$, refused |
| §7.2 unchanged routes | identities exact; sources within twice the recorded floor | exact; within it, for `none`, the collar and the window from `e1a8274` |
| §7.2 runs | v4 round trip and resume for unwindowed and windowed SAE and haCC; cross-filter resume refused; equal ledgers | as stated; atomic and logical ledgers exactly equal |
| §7.2 fake drift | the run's edge ratio equals the direct T2 (c) one within its floors | under both forms, on each run's own problem |

The unchanged-route witness found a real regression during development: with the edge
array omitted, `time_collar`'s operand floor had lost its gate. It was fixed before the
first commit of the assembly.

*Certificates of the default configurations* (run_short: 60 fs over 90 nm–1.6 µm, the
six-node Gauss field-scale rule, 366 active bins from $6.1\times10^{14}$ to
$2.3\times10^{16}$ rad/s):

| configuration | lattice | samples per taper | $\kappa_{\rm ss}$ |
|---|---|---|---|
| haCC(1), haCC(5) and haCC(1) fast, 60 fs window with 5 fs tapers | 16384 × 6.25 as | 799 | $5.06\times10^{-16}$ |
| haCC(1), full lattice, grid collar | 16384 × 6.25 as | 3395 | $5.69\times10^{-15}$ |
| SAE research record (SEP 12, $q=32$), full lattice, grid collar | 65536 × 1.56 as | 13581 | $4.37\times10^{-15}$ |
| SAE research record, 60 fs window | 65536 × 1.56 as | 3198 | $4.84\times10^{-16}$ |

The haCC presets were represented by the contract copies of their records in
`tests/hacc_cases.py`, since the provider is not installed here; the certificate depends
only on the lattice, the gate and the band.

*Hot path* (`benchmarks/sep26/evidence/hot-path.json`, at `988add7`, load 6.4): one
source assembly and projection on the run_short lattice took 0.657 ms (MAD 0.013 ms)
on the pre-SEP-26 path, 1.131 ms (0.016 ms) for `time_collar` with its edge diagnostic
and 1.129 ms (0.018 ms) for `gated_derivative`: 1.72 times, with 3.1 MB more peak traced
allocation against a 4.0 MB `tdse_edge_sources` ledger row. Against minutes per atomic
call it is negligible; against the source assembly alone it is not.

*Not run.* None of Phase 2 (the L5 replay, the L6 segments, the SAE controls, the
rate-plasma measurement). No GPU, cluster or Linux run; CI's Linux, Windows and
lowest-version lanes run on the pull request. solprop-kokkos's `scripts/check.sh`
(native GoogleTest, sanitizers, clang-tidy, the wheel smoke test) was not run; its only
native change is the schema literal of the build record. Its Python suite passed (319
passed, 5 skipped) against the editable build.

**Sources checked.** The derivative of McKechan *et al.*'s eq. (7) (reference 1) was
re-derived and checked numerically against the closed form and a fourth-order
difference (T1). The FFT constant starts from Higham's Theorem 24.2 (reference 2) as the
implementer knows it; the chapter was not re-read for this phase, and the constant is
qualified by T3's measurement, not by the citation. No physical constant or coefficient
was transcribed.

**Limitations.**

- `numpy_ref` only; `numba_cpu` and solprop-kokkos refuse every TDSE problem by name
  (SEP 12's recorded exception).
- The certificate is relative to $\lVert\mathrm{DFT}[g]\rVert_A$. A long, well-resolved
  taper on a band far above its frequencies can be refused although its absolute edge
  error is at rounding (T7: $1.8\times10^{-10}$). No default or campaign configuration is
  near this.
- `time_collar` runs now pay the edge diagnostic: one more fine transform and about 3 MB
  per evaluation.
- The §4.5 budget is a conservative first-order indicator (T2 (a) sits four orders below
  it), not a certificate of the atomic integration.
- The leading-taper band bound is an absolute-projection upper bound, loose by 2–3 times
  on the T6 case.

**Open questions.**

- **Q7** (maintainer, Phase 3): should the certificate stay relative to the truncating
  source, or be compared with the band's own rounding budget, so that a resolved taper on
  a UV-only band is not refused? Phase 2 should record it on every replayed
  configuration either way.
- **Q8** (maintainer, Phase 3): keep computing the `edge` diagnostic under
  `time_collar`, or compute it only on request once Phase 2's comparisons are made?
- **Q9** (Phase 2): the SEP 20 scripts (`run_short_hacc.py`, `segment.py`, `window.py`)
  have no source-filter option yet.

**Handoff to Phase 2.**

- Opt in with `replace(response, source_filter="gated_derivative")`. The run then writes
  a v4 descriptor and records `execution["tdse_source"]`. A `time_collar` file refuses
  to resume under it and the reverse, so §7.4's runs (a)–(c) are fresh runs.
- For the offline replays (§7.3, §7.5), build a `TDSEResult` from the archived currents
  and call `atomic_sources`, `atomic_edge_source` and `project_atomic_sources`; the
  `edge` diagnostic gives the band norms and ratios with their floors directly.
  `steady_current_certificate(descriptor, omega, active)` gives any gated descriptor's
  certificate without an evaluation.
- Add the source-filter option to the SEP 20 scripts first (Q9); the derived analysis
  record must keep each raw current's descriptor and identity (§7.3).
- Rebuild solprop-kokkos from its `feature/sep-0026-phase-1` (schema 30) before any
  Kokkos lane on a Phase 2 tree.

### Phase 2 report — validation (2026-10-19)

**Scope.** Phase 2 of §11: §7.3's L5 replay, §7.5's offline SAE controls and SAE
rerun, §7.4's three L6 segments, §4.7's offline rate-plasma measurement and the
certificate of every replayed configuration. Implementer: Opus 5.5 (AI) for John Travers, on branch
`feature/sep-0026-validation` from `392c2305`. Part I is unchanged. Phase 1's
open questions were decided by the coordinating agent on 2026-10-18:

- Q7: the certificate stays relative to the truncating source. The refusal of a
  long taper on a UV-only band remains a known limitation (below).
- Q8: the edge diagnostic stays under `time_collar`.
- Q9: in scope here.

**Delivered.**

- *Q9.* `benchmarks/sep20/run_short_hacc.py` and `segment.py` take
  `--source-filter {default,time_collar,gated_derivative}` and
  `--window-duration-fs` (with `--atomic-window window`). Both are recorded in the
  provenance (requested and resolved filter) and in `run.json`. The run summary
  also keeps `tdse_source` and the last evaluation's `edge` entry.
  `benchmarks/hacc_perf/slurm/segment.sbatch` passes `SEGMENT_OPTIONS` through.
- *`benchmarks/sep20/window.py`.* `analyse --source-filter` replays the raw L5
  currents as a derived record, schema `solprop.sep20-window.v2`. The record keeps
  each raw descriptor identity, the replay filter, the replay descriptors and their
  certificates. The replay descriptor must differ from the raw one only in its
  filter and version. The analysis adds a same-current comparison, the 0.9–1.6 µm
  band and the edge fractions. `run` takes `--source-filter`.
  `benchmarks/sep20/sources.py` forms the edge-free source and its §4.5 floor by
  band.
- *Replays, `benchmarks/sep26/`.*
  - `l5.py` (§7.3 (i)–(iii));
  - `sae_calls.py` (the L6 per-call SAE archives);
  - `sae_width.py` (SEP 12's width study);
  - `rate_plasma.py` (§4.7);
  - `publish.py`, which copies a record into the tree without host names or local
    paths.
- *Tests.* `tests/test_sep20_window.py` (new), `tests/test_sep20_segment.py`
  (extended) and `tests/test_sep26_publish.py` (new).
- *Evidence, `benchmarks/sep26/evidence/`.*
  - `l5-*.json` and `l5-recorded-descriptors.json`;
  - `sae-calls.json`, `sae-width.json` and `rate-plasma.json`;
  - `l6-*.json`;
  - `manifest.json`, which holds the hashes of the raw segment outputs, the original
    records, the private pump, the jobs and the core-hours.
- *Documentation.* `benchmarks/sep26/README.md` and `benchmarks/sep20/README.md`.

**Inputs.** Every raw input was hashed against its manifest before use, and
everything matched:

- the L5 currents (`benchmarks/sep20/evidence/manifest.json`,
  `raw_files_sha256.session`);
- the L6 runs (`phase4.l6_runs`);
- the SEP 12 width study's eight cases (`source-filter-provenance.json`,
  `raw_case_hashes`, read only).

The private pump was used only where it was already staged, with mode 600. Only its
hash `d9a512d3…` is recorded. The private `run_short.py` was not used.

**Deviations from Part I, and why.**

1. *L5 descriptor identity.* The L5 currents were recorded at the provider pin
   `ea2bfcaa`. At the current pin `876b72d1`, the preset's provenance text differs:
   its `source` strings, one added `limitations` line and the calibration's
   `basis_identity`. That changes the descriptor identity, although no number and no
   other field changes. The recorded configurations were regenerated at the Phase 3
   solprop revision `db4ca407` with the provider at `ea2bfcaa`. Each hashes exactly to
   its recorded identity (`l5-recorded-descriptors.json`). `window.py` admits a
   harness only if its configuration differs from the recorded one in those four
   fields alone.
2. *(i)'s budget* propagates the combined floor of both sources into both the
   numerator and the denominator, $(2F+2\delta F)/(\lVert r\rVert-2F)$. This is
   conservative.
3. *(ii)'s $e_g$* is the sum, over both gates, of
   $\lVert c(\mathrm{DFT}[g'_{\rm cd}S]-\mathrm{DFT}[g'S])/(i\omega)\rVert_B$,
   relative to $\lVert F_{100}\rVert_B$. The record also carries half a unit in the
   last printed digit of the expected figures. It was not needed: every deviation is
   within the derived tolerance alone.
4. *SAE "recorded figures".* The L6 archives keep no per-call source figures. Their
   recorded figure is each run's `tdse_last_evaluation` complete-source norm,
   reproduced within twice its recorded floor.
5. *The current-form 60 fs segment was not rerun.* The provider pin has not moved
   since `benchmarks/hacc_perf/evidence/perf.json` (job 483804, solprop `d482cdf0`):
   `time_collar` keeps its formula and path. The SAE rerun is compared with SEP 20
   Phase 4's `sae-window` run at `ea2bfcaa`. No native file differs between
   `ea2bfcaa` and `876b72d1` (only `pyroraptor/hacc/*.py` changed), so the comparison
   differs only in the filter and the solprop revision.
6. *Rate plasma.* The rate problem is rebuilt from the case's settings with a
   placeholder pulse, because the pump is private and off this machine. The pulse
   enters no grid, map or coefficient. The coarse grid was checked equal to the run's
   saved grid. The L6 rate run saves two planes, 3.969 m and 4.05 m, and both were
   measured.

**Verification and results.**

*Machines.* The offline replays ran on an 18-core Apple M5 Max (macOS 27, arm64).
The software was Python 3.14.5 and NumPy 2.5.2, with Pyroraptor `876b72d1` (Clang 23,
OpenMP, Release) installed in the worktree environment. Each replay used 1–4 threads,
one job at a time. The load average was 5.8–6.0 before and after the final pass at
`7ad9f0dc`, and each replay took 3–6 s.

The L6 segments ran on cluster CPU nodes:

- 2 × EPYC 7763 nodes for haCC, and a 96-core node for SAE, as in Phase 4;
- GCC 16.2, Release, OpenMP, x86-64-v3, solprop `594b0c68`;
- each in a 32-CPU block with `--hint=nomultithread --distribution=block:block`.

Placement was verified with taskset and numactl:

| run | job | node | CPUs | NUMA domains | elapsed | core-hours |
|---|---|---|---|---|---|---|
| (a) | 483988 | node19 | 0–31 | 0–1 | 1 h 15 | 39.8 |
| (b) | 483989 | node19 | 32–63 | 2–3 | 1 h 46 | 56.4 |
| (c) | 483990 | node20 | 0–31 | 0–1 | 1 h 35 | 50.7 |
| SAE | 483991 | node14 | 0–31 | 0–2 | 1 h 21 | 43.2 |

With the build and the analysis, the total is 191.5 core-hours, against 215–270
estimated. Timing was not the aim; (a) and (b) shared a node.

*§7.3 L5* (`l5-replay.json`):

- **(i) Recorded figures.** All eight recorded `time_collar` ratios are reproduced:
  full lattice and common window, band and uv, both planes. The largest change is
  $3.2\times10^{-17}$, against budgets of $3.0\times10^{-12}$ to
  $1.1\times10^{-11}$. Passed.
- **(ii) Same-current anchor.** Edge-free, against 0133's figures; the tolerance is
  $(2+\delta)(r+e_g)$ with $r=5.4\times10^{-4}$. Passed in the band at both planes
  (the requirement), and nir and uv within the same tolerance:

  | plane | band δ (expected) | nir | uv | $e_g$ | tolerance |
  |---|---|---|---|---|---|
  | peak power | 1.2224% (1.22%) | 3.3260% (3.33%) | 0.3877% (0.387%) | $1.3\times10^{-6}$ | $1.09\times10^{-3}$ |
  | maximum UV | 1.4521% (1.45%) | 4.4238% (4.42%) | 0.3498% (0.349%) | $2.2\times10^{-7}$ | $1.09\times10^{-3}$ |

  The study's own centred-difference split, replayed here, gives 1.2226% and 1.4524%
  in the band.
- **(iii) Actual L5**, the separately driven 60 fs current against the 100 fs one.
  This is reported, and it is still a recorded failure of L5's 1% band tolerance at
  both planes:

  | plane | filter | band | nir | uv | common window (band) |
  |---|---|---|---|---|---|
  | peak power | `time_collar` | 4.607% | 12.49% | 0.405% | 0.036% |
  | peak power | `gated_derivative` | **1.230%** | **3.342%** | **0.405%** | 0.047% |
  | maximum UV | `time_collar` | 1.428% | 4.315% | 0.365% | 0.046% |
  | maximum UV | `gated_derivative` | **1.439%** | **4.379%** | **0.365%** | 0.052% |

  As §7.3 expected, (iii) is close to (ii) plus the startup and drive difference.
  The edge term is 10.7% (60 fs) and 11.0% (100 fs) of each source's own nir at peak
  power, and 1.7% and 2.2% at maximum UV.

*§7.5 SAE controls:*

- **L6 per-call archives** (`sae-calls.json`, 9 evaluations per run).
  - Both runs' recorded last-evaluation norms are reproduced:
    $1.1453056830\times10^{7}$ and $1.1495518654\times10^{7}$, each a change of
    $5.6\times10^{-7}$ against budgets of $8.7\times10^{-5}$ and
    $8.0\times10^{-5}$.
  - The T3 spectral-derivative oracle on the real currents agrees with the identity
    form to $2.7$–$5.8\times10^{-16}$ (band) and at most $3.3\times10^{-15}$
    (1.2–1.6 µm) relative, on every call of both runs.
  - Edge over truncating source, band / nir / 1.2–1.6 µm:
    - collar (`sae-full`): 0.07% / 0.26–0.32% / 0.65–0.82%;
    - 60 fs window: 2.5–2.9% / 9.5–10.4% / 21.5–23.3%.

    In 90–200 nm it is at most $2.6\times10^{-5}$ for both.
  - The shared launch field, window against collar, changes from 2.17% to 0.88%
    (band), from 8.14% to 3.25% (nir) and from 0.119% to 0.119% (uv), going from
    `time_collar` to `gated_derivative`.
- **SEP 12 width study** (`sae-width.json`).
  - All 66 recorded direct-ordering figures are reproduced (48 width sensitivities
    and 18 window comparisons). The largest change is $2.7\times10^{-7}$ of its
    budget.
  - The edge-free width sensitivity is no larger than the truncating one in every
    one of 48 case × gate × band comparisons; none is unresolved. It falls by
    3–1000 times. For example, the 60 fs launch window, narrow against matched,
    over 120–1600 nm goes from 3.48% to 0.82%, and the 120 fs and 240 fs windows fall
    to below $10^{-4}$.
  - The 60-against-120 and 240 fs window comparisons do not fall: 8.71% → 8.83%
    full band and 7.12% → 6.71% over 120–1600 nm (launch). That is the windows'
    history difference, as for haCC.
- **SAE rerun** (`l6-gates-sae.json`, `l6-sae-pairs.json`).
  - L6's mechanical and sensible-optics gates pass: energy over launch 0.9856, over
    rate 0.9949, and UV over rate 1.27.
  - Against Phase 4's `time_collar` `sae-window`: 0.318% full, 0.0087% in 90–200 nm
    (below the floor), 0.145% at 0.9–1.2 µm and 1.94% at 1.2–1.6 µm.
  - Against `sae-full` (collar), the window's endpoint sensitivity falls from 0.323%
    to 0.082% (full band) and from 1.35% to 0.72% (1.2–1.6 µm).
- **Pass for the SAE controls (§7.5): yes, all four conditions.** The replays
  reproduce their recorded figures, every certificate is at most $10^{-10}$, the
  rerun passes L6's gates, and the edge-free width sensitivity is no larger than the
  truncating one in any band. The SAE research record may therefore move in Phase 3.

*§7.4 L6 segments.* haCC(1) fast, `rtol=1e-2`, 32 threads and the full archive
(`l6-hacc-pairs.json`, `l6-gates-*.json`). Every run completed with 9 atomic calls
and 8 accepted steps. Every run passes L6's mechanical and sensible-optics gates:

- energy over launch 0.9852;
- energy over rate 0.9945;
- UV over rate 1.240;
- UV over `sae-full` and over `sae-window`, 0.975.

Endpoint spectra are compared as $\lVert a-b\rVert_B/\lVert b\rVert_B$. The floor is
0.073% (SEP 20 L6's restart sensitivity), and a figure below it is unresolved.

| pair (relative to) | full | 90–200 nm | 0.9–1.2 µm | 1.2–1.6 µm | above 0.9 µm |
|---|---|---|---|---|---|
| (a) gated 60 fs vs current form (`time_collar` 60 fs) | 0.334% | **0.0088%** (unresolved) | 0.152% | 2.03% | 0.345% |
| (a) vs (b) gated 100 fs | 0.091% | 0.087% | 0.058% | 0.77% | 0.131% |
| (c) `time_collar` 100 fs vs current form | 0.357% | 0.087% | 0.163% | 1.30% | 0.256% |
| (b) vs (c), both 100 fs | 0.091% | 0.0022% (unresolved) | 0.035% | 0.47% | 0.079% |

- Endpoint energy is $1.34001\times10^{-3}$ J for (a)–(c) against
  $1.34003\times10^{-3}$ J for the current form. Peak power is 590.64, 590.64,
  590.65 and 590.69 GW, and the 90–200 nm energy is 4.8800 µJ for all four.
- The certificates recorded in each run's metadata are $5.14\times10^{-16}$ (a),
  $5.06\times10^{-16}$ (b and c) and $4.46\times10^{-16}$ (SAE). The last
  evaluation's $\rho_-$ is at most $2.3\times10^{-11}$.
- **Investigation threshold.** The 90–200 nm difference between (a) and the
  current form is 0.0088%. That is 57 times below 0.5%, and below the floor. No
  investigation was triggered, so the matched-input and `rtol=3e-3` controls were not
  run.
- **Expectation.** The (a)–(b) window sensitivity above 900 nm is 0.131%, against
  the current form's 0.256% (c against the current form). It is about half, not
  "well below"; at 1.2–1.6 µm it is 0.77% against 1.30%. The 90–200 nm window
  sensitivity, 0.087%, is the same under both filters, as in L5.

*§4.7 rate plasma* (`rate-plasma.json`). This is a fixed-field source comparison on
the 33-node rule, with the band running-integral rule. The closure residual is
$3$–$10\times10^{-7}$ of $i\omega\,\mathrm{DFT}[gP]$ in every band, within that
rule's certified error of $4.9\times10^{-6}$:

| plane | $\lVert Q_{\rm route}-Q_{\rm held}\rVert/\lVert Q_{\rm route}\rVert$: band | uv | nir | 0.9–1.2 µm | 1.2–1.6 µm |
|---|---|---|---|---|---|
| 3.969 m | 3.37% | $1.8\times10^{-5}$ | 8.40% | 3.24% | 18.7% |
| 4.05 m | 7.36% | $1.4\times10^{-5}$ | 18.3% | 7.15% | 38.2% |

$P$ is held at its peak at the lattice end; its final-to-peak ratio is 1.0. The
collar's edge term $\mathrm{DFT}[g'P]$ is of the same size as this difference. The
collar's certificate is $1.8\times10^{-14}$. The production rate route is unchanged;
Q3 decides whether a follow-up SEP is needed.

*Certificates of every replayed configuration*, each at most $10^{-10}$:

| configuration | $\kappa_{\rm ss}$ |
|---|---|
| haCC lattice, 60 fs window | $5.06\times10^{-16}$ (offline); $5.14\times10^{-16}$ in run (a) |
| haCC lattice, 100 fs window | $5.53\times10^{-16}$ (offline); $5.06\times10^{-16}$ in runs (b) and (c) |
| SAE lattice, grid collar (`sae-full`) | $4.37\times10^{-15}$ |
| SAE lattice, 60 fs window | $4.84\times10^{-16}$ (offline); $4.46\times10^{-16}$ in the SAE rerun |
| SEP 12 cases, narrow, matched and wide gates (3000–5000 samples per taper) | $3.1$–$8.1\times10^{-16}$ |
| rate route, fine collar (§4.7) | $1.8\times10^{-14}$ |

The offline and in-run values differ at rounding level, because they were computed
by different FFT builds and machines.

*Checks.*

- ruff, ruff format and basedpyright on `benchmarks/sep20`, `benchmarks/sep26` and
  the touched tests: clean.
- `uv run pytest tests/test_sep20_window.py tests/test_sep20_segment.py
  tests/test_sep20_benchmarks.py tests/test_sep26_publish.py tests/test_seps.py`:
  245 passed, 1 skipped (the process-SEP skip).
- codespell, `git diff --check` and `uv run poe docs` for this text.

**§11 promotion conditions, item by item.**

- T1–T7: passed in Phase 1 and not rerun here.
- §7.3 (i) and (ii): passed.
- §7.4's gates: passed on all three runs and on the SAE rerun. No investigation was
  opened, so none is left open.
- Certificates: at most $10^{-10}$ on every replayed configuration; the largest is
  $1.8\times10^{-14}$.
- The maintainer's decision on Q4 and Q30: below.

**Maintainer's decision on Q4 and Q30** (2026-10-18, relayed by the coordinating agent; conditional).
The 60 fs default window stays. `gated_derivative` becomes the haCC and windowed
default in Phase 3. Q30's exception is restated at the measured actual-L5 size under
the new form: 1.23% band, 3.34% nir and 0.405% uv at peak power, and 1.44%, 4.38% and
0.365% at maximum UV. The condition was that the L6 segments show nothing unexpected:
the 90–200 nm change stays below the 0.5% investigation threshold, and every
investigation is closed. **The condition holds.** The 90–200 nm change is 0.0088%
((a) against the current form), and no investigation was opened.

**Sources checked.** No physical constant or coefficient was transcribed. Two
derivations were rechecked numerically, as described above:

- the held-current identity, against the spectral-derivative oracle on the real SAE
  currents;
- the discrete closure $P'=J$, against the running-integral rule's certified error.

**Limitations.**

- The certificate stays relative to the truncating source (Q7). A long,
  well-resolved taper on a band far above its frequencies can be refused although its
  absolute edge error is at rounding level. No configuration measured here comes
  near this: the largest certificate is $1.8\times10^{-14}$.
- The window sensitivity that remains under `gated_derivative` (L5: 1.23% and 1.44%
  band) is still a sensitivity, not an error against a converged late-time reference.
  L5 still fails its 1% band tolerance under either filter.
- The L6 comparisons are single trajectories at `rtol=1e-2`. Figures below the
  0.073% floor are reported as unresolved, and no tighter-tolerance control was run
  because none was triggered.
- The rate-plasma figure is a fixed-field source comparison on two saved planes, not
  a propagated trajectory.

**Open questions.**

- **Q10** (maintainer, with Q3): the rate route's gated polarisation differs from
  a held current by 3–7% of the band source and 8–18% in the near infrared at the L6
  planes, and by $10^{-5}$ in the UV. Does this justify a follow-up SEP?

**Not run.**

- §7.4's matched-input, pinned-input and `rtol=3e-3` investigation controls, because
  no investigation was triggered.
- Any GPU or Kokkos lane, the full gate, the slow lane and the coverage lane; these
  are Phase 3's.
- A rerun of the current-form 60 fs segment, because the provider pin has not moved.

**Handoff to Phase 3.**

- Promote `gated_derivative` to the haCC and windowed default. The maintainer's
  condition holds.
- Move the SAE research record, since §7.5 passed.
- Restate Q30 at the actual-L5 figures above.
- Run the end-of-SEP lanes.
- Rebuild solprop-kokkos for schema 30 before the Kokkos lane.

### Phase 3 report — promotion and the final gate (2026-10-19)

**Scope.** Phase 3 of §11: the default promotion, the move of the SAE research record,
Q30 restated, the Q3/Q10 rate-plasma record, and the end-of-SEP lanes. Implementer:
Opus 5.5 (AI) for John Travers. Branch
`feature/sep-0026-phase-3` from `4efb87bd`, the merge of Phase 2. Its commits are
`37853788` (the promotion, the record, the manual and the changelog) and `2e3fb485`
(the roadmap and SEP 20's erratum), followed by this report. Merge commits are the coordinating agent's to record. solprop-kokkos is unchanged: its `main` at `f40728d` already
accepts schema 30.

**Delivered.**

- *Promotion.* `src/solprop/nonlinear/tdse.py`: an omitted `source_filter` resolves to
  `"gated_derivative"` for haCC (windowed or on the grid collar) and for any windowed
  response. An unwindowed SAE response keeps `"none"`. `"time_collar"` stays
  selectable, and its formula, path, descriptor bytes, identities and resume keys are
  unchanged. The resolution happens in the response record before translation, so no
  descriptor version, kernel schema or output schema moves. Recorded files keep their
  explicit `"time_collar"` and resume under it only. The docstring states the
  promotion. `src/solprop/translate/tdse.py`: the admission refusal now also names
  `source_filter='time_collar'` as a remedy, because an omitted filter on an
  unresolving lattice is now refused instead of truncated.
- *Tests.*
  - `tests/hacc_cases.py`'s haCC contract fixture now names `"time_collar"`. Its lattices
    give 1–5 fs tapers of 4–62 samples, below the 128-sample admission. They test the
    binding, records and archive, not the source form; without the pin, 157 tests were
    refused at translation.
  - `tests/nonlinear/test_tdse_spec.py`, `tests/translate/test_tdse_descriptor_v4.py`
    and `tests/translate/test_tdse_contract.py` assert the new resolution (haCC
    windowed and unwindowed, windowed SAE, unwindowed SAE) and that the named
    truncating form is kept.
  - `tests/translate/test_tdse_descriptor_v4.py` gains a test that an omitted filter on
    a 127-sample taper is refused, not silently truncated, while the named
    `"time_collar"` builds a v3 descriptor.
  - `tests/test_sep20_window.py` follows the harness's new haCC default.
- *The SAE research record* (§7.5, Q2):
  `benchmarks/sep26/evidence/selected-gated-derivative.response.json`. It holds SEP 12's
  record with `source_filter="gated_derivative"` under the new calibration identity
  `sep26-native1537-cut400-q32-gated-derivative-research-v1`. The calibration's source
  text and its two filter-related limitations are rewritten; every other field is
  identical. SEP 12's `selected-filtered.response.json` is unchanged; its SHA-256
  `de50ad2a…` still matches SEP 20's manifest. `tests/test_sep26_research_record.py`
  checks the old hash and identity, that the two records differ only in the filter and
  the calibration's identity, source and limitations, and that the new record builds a
  v4 descriptor with a distinct identity on SEP 12's parent lattice at $q=32$, with a
  certificate within the limit.
- *Manual.*
  - `docs/guide/media/tdse-hacc.md`: the default; the restated window limitation (Q30);
    a separate SAE window limitation; the edge-free L6 table; the corrected L5 wording.
    Its example now asserts the default and builds the truncating form.
  - `tdse-configuration.md`: the resolution table; "The default, and what it changed";
    the tested example; the reconstruction section, which now downloads the moved
    record and keeps SEP 12's.
  - `tdse-capillary.md`: the source-filter row; the default note; "The research record
    after SEP 26".
  - `docs/guide/results/resuming.md`: resuming files written under the old default.
  - `docs/guide/workflows/luna.md`: the default.
  - `docs/dev/kernel-contract.md`: row 30 and the version-4 paragraph; no schema change.
- *Benchmarks.* The docstring of `benchmarks/sep20/run_short_hacc.py` and
  `benchmarks/sep20/README.md`: `--source-filter default` is now `gated_derivative`
  for haCC, and `--source-filter time_collar` repeats a SEP 20 haCC run.
- *Roadmap* (`docs/dev/roadmap.md`): the SEP 26 listing updated; Q30 restated in the
  haCC item; the SAE window sensitivity in the TDSE qualification item; a new item, *The
  field-grid rate plasma's collar truncation (SEP 26 Q3, Q10)*.
- *SEP 20.* One erratum note under Q30, pointing to this SEP.
- *Changelog.* A *Changed* entry marking the breaking default; the Phase 1 *Added*
  entry no longer says "no default changes".

**Deviations from Part I, and why.**

1. *The contract fixtures keep `"time_collar"`* rather than moving to the default. Their
   coarse lattices are not admitted under `"gated_derivative"`. Refining them would
   change the cost and meaning of 157 binding, record and archive tests, which are about
   neither source form. The new default's resolution, admission and runs are tested
   separately; `tests/driver/test_tdse_gated_derivative_run.py` covers the runs.
2. *The SEP 20 harness keeps SEP 12's `time_collar` record* as its default
   `--sae-record`, so that SEP 20's SAE runs and Phase 2's comparisons repeat
   unchanged. The moved record is selected with `--sae-record`. haCC's harness default
   follows the response's own default and so moves.

**Decisions.**

- An omitted filter on an unresolving lattice is refused at translation. It never falls
  back to `"time_collar"`: that would be a silent fallback, and the refusal names the
  remedies.
- The moved record's limitations state that the new form was propagated only on the
  60 fs window segment. On the full-lattice collar it was evaluated offline per call,
  with an edge term of 0.07% of the band.
- No ADR: the decision is recorded here and in the changelog.

**Q30 restated** (manual and roadmap), from Phase 2's figures under the new form:

- *Actual L5, against 100 fs.* At peak power, 1.23% of the band, 3.34% above 900 nm
  and 0.405% at 90–200 nm. At maximum UV, 1.44%, 4.38% and 0.365%. Both still fail
  L5's 1% band tolerance, and the 60 fs default is kept with this exception (the
  maintainer's decision of 2026-10-18).
- *L6 propagated window sensitivity, (a) against (b).* 0.091% full, 0.087% at
  90–200 nm and 0.131% above 0.9 µm.
- *Advice.* Use a 100 fs window where the near infrared matters.
- *The SAE limitation, stated separately.* The 60-against-120/240 fs window
  sensitivity is about 8.7% (`time_collar`) and 8.8% (edge-free) of the full band. The
  new form does not reduce it. It does reduce the filter-width sensitivity 3–1000 times
  and the window-against-collar differences (single field 2.17% → 0.88% band; L6
  endpoint 0.323% → 0.082% full).

**Q3/Q10, the rate plasma.** Reported only; the rate route is unchanged. The route's
gated $P$ differs from the held current by 3.4% of the band and 8.4% in the nir at
3.969 m, and by 7.4% and 18% at 4.05 m, against about $1.5\times10^{-5}$ in the UV. The
roadmap records this as an open item for the maintainer: a possible follow-up SEP for
the field-grid rate plasma's collar truncation.

**Verification.** The machine was an 18-core Apple M5 Max (macOS 27, arm64), with
Python 3.14.5, NumPy 2.5.2 and SciPy 1.18.1. xdist was bounded to 8 workers with one
thread per worker (`SOLPROP_KOKKOS_THREADS=1`, `OMP_PROC_BIND=false`). solprop-kokkos
`main` `f40728d` (schema 30) was built from a clean clone (CPU, OpenMP, local profile,
Homebrew Clang 23) and installed in the worktree's environment. The lanes ran
sequentially on `2e3fb485`, with this report not yet written. The machine was shared,
with other agents' work running, and the load average was 4.1–10.4 (1-minute 3.7–9.1):

| lane | result | wall |
|---|---|---|
| `uv run poe check` | green: ruff, format, basedpyright, numpydoc, codespell, reuse; the default lane 15920 passed, 29 skipped; strict Sphinx | 15.4 min |
| `uv run poe test-slow` | 147 passed, 1 skipped; the statistics-cost check passed | 6.8 min |
| `uv run poe cov` | **100%** of statements and branches (41542 and 12580). The lane exited 1 on one test (below) | 11.2 min |
| `SOLPROP_TEST_BACKEND=kokkos uv run poe test` | 15920 passed, 29 skipped | 10.3 min |
| `SOLPROP_TEST_BACKEND=kokkos uv run poe test-slow` | 148 passed; the statistics-cost check passed | 6.0 min |

The coverage lane's one failure was
`tests/backends/test_numpy_ref_backend.py::test_a_gradient_capillary_propagates_on_the_field_grid[numba_cpu]`.
The test asserts no log record, and caught an INFO record from
`solprop.translate.linop` (a raised tolerance floor). It concerns neither TDSE nor
anything this SEP changed, and it passed in the default and Kokkos lanes on the same
tree. Rerun alone, the module passed three times under the lane's coverage
configuration, once without coverage, and serially with the failing run's random seed
(130 passed, 1 skipped each time). It depends on which tests share a worker, so it is
recorded as an intermittent failure that predates this branch, not fixed here. The
coverage total was measured on the full run, which reached 100%.

Before the lanes, the TDSE-related modules (`tests/nonlinear`, `tests/translate`, the
TDSE backend, driver and output modules, the SEP 12, 20 and 26 scripts) passed. So did
the TDSE guide pages run with their slow snippets (8 passed). The lanes ran these
modules again.

*Not run.* No GPU, cluster or Linux run; CI's Linux, Windows and lowest-version lanes run
on the pull request. solprop-kokkos's own `scripts/check.sh` was not run: nothing
changed there.

**Backend parity.** TDSE runs on `numpy_ref` only. `numba_cpu` and solprop-kokkos refuse
every TDSE problem by name, `gated_derivative` included (tested in Phase 1 and in the
Kokkos lanes here). This is SEP 12's recorded exception, which SEP 26 §2 carries
("Compiled backends: TDSE stays NumPy-only"). It satisfies SEP 0's backend condition
for Final through that exception, whose roadmap item SEP 12 holds.

**Sources checked.** No constant, coefficient or citation was transcribed. Every figure
restated in the manual, the roadmap, the changelog and the moved record was checked
against Phase 2's report and evidence (`l5-replay.json`, `l6-hacc-pairs.json`,
`sae-width.json`, `sae-calls.json`, `l6-sae-pairs.json`, `rate-plasma.json`).

**Limitations.**

- The moved SAE record has not been propagated on its own full-lattice collar under the
  new form. Its propagated check is the 60 fs window segment; the collar was evaluated
  offline per call.
- The window sensitivities restated under Q30 are sensitivities, not errors against a
  converged late-time reference. L5 still fails its 1% band tolerance.
- The coverage lane carries the intermittent `numba_cpu` log-record failure above.
- Phase 1's report says that each page it produced or extended ends with a link to this
  SEP. That held for the three TDSE pages, not for `resuming.md` or `luna.md`; their
  SEP 26 paragraphs now link it inline.

**Open questions.**

- **Q10** (maintainer, with Q3; moved to the roadmap): does the rate plasma's collar
  truncation justify a follow-up SEP?
- Q1–Q9 are resolved. Q4 and Q30 were decided by the maintainer on 2026-10-18 (Phase 2
  report) and applied here.

**Handoff.** No phase follows. Final status, the archive move and the link fixes are in
a separate commit on this branch. The roadmap keeps two informal items: the rate-plasma
item (Q10), and the window and propagation questions in the haCC and TDSE qualification
items.


## Part III — History

- 2026-10-18: Drafted at the maintainer's choice of the held-current remedy (option 1)
  after Pyroraptor report 0133's taper-edge addendum.
- 2026-10-18: Q1–Q6 resolved by the maintainer (the coordinating agent's recommendations); §7.5 and
  §11 updated: one SAE L6 rerun funded, the SEP 12 archive located, and an offline
  rate-plasma measurement added to Phase 2. Merged into main as a Draft for review.

### AI review — 2026-10-18

**Reviewers and scope.** GPT-6 (AI) reviewed all of Part I against SEP 0,
[the conventions](../../conventions.md), the current TDSE source, descriptor, output
and diagnostic code, and the qualification records of SEPs 12, 19, 20 and 25.
John Travers requested that his human review also be recorded. The review status is
`Reviewed`; the proposal remains `Draft`. This records completion of review, not
acceptance or resolution of the findings below. Part I has deliberately not been
edited. The drafting agent should respond to R1–R13 individually, with the changed
sections or an explicit reason for leaving a recommendation open.

**Assessment.** The continuous product-rule identity and the held-current
interpretation are sound away from zero frequency. Keeping shadow subtraction first,
a distinct filter identity, the existing active mask and the separate field absorber
is a coherent design. The proposal is not yet ready for acceptance: the discrete
validity claim currently includes admitted configurations where it fails, some test
oracles are incorrect, and several validation claims exceed their evidence.

#### R1 — P1: qualify the discrete identity on the admitted lattices (§4.4, §6, T1–T3)

The continuous identity does not make the sampled formula exact to rounding for
arbitrary admitted gates. `TDSEAtomicWindow` only checks a positive taper shorter
than half the duration; `resolve_atomic_window` checks the window length and at least
four retained samples, without resolving either taper. An otherwise valid window
with a taper shorter than a sample can have no nonzero derivative samples. Then
`DFT[g'S]` is zero although `DFT[gS]` is not, and the proposed steady-current
cancellation fails completely.

A review probe used $N=16384$, $\Delta t=6.254\times10^{-18}$ s, an atomic duration
$80\Delta t$ and taper $\Delta t/4$. The existing resolver admitted it. For $S=1$,
the norm of the proposed source divided by that of the truncating source was 1.0
on $5.9\times10^{14}<\omega<2.1\times10^{16}$ rad/s. With the 60 fs window and 5 fs
tapers on the same lattice, that ratio was $5.9\times10^{-16}$ instead.

Please state a lattice/taper/retained-band validity condition and how translation
handles a configuration outside it. Either bound and test the alias contribution
for the admitted domain, or explicitly limit the rounding-level claim to qualified
configurations. A four-sample taper alone is not a rounding-level certificate.
Include asymmetric/off-grid breakpoints, widths near the admission limit and the
highest retained frequency, with refinement checks. The two campaign estimates and
one low-band synthetic signal do not cover the public interface.

Also separate the exact continuous identity, the discrete *definition*, aliasing
error and floating-point error in the prose. A saddle-point asymptotic with unspecified
algebraic factors is not a numerical upper bound. A trigonometric interpolant has a
line spectrum; formulate its Fourier bound using coefficient sums or the total
variation of its spectral measure, rather than an ordinary $L^1$ density. For T1,
$C^\infty$ regularity alone does not imply an exponential trapezoidal error bound,
and a sampled maximum of $g^{(5)}$ is not by itself an upper bound on that derivative.

#### R2 — P1: preserve the turn-on source in the steady-tail tests (T2, §7.2)

T2's last bullet incorrectly extends the zero-source result for a current constant
over the *whole* window to a current that turns on inside the plateau. For
$S=H\Theta(t-t_c)$ with $g(t_c)=1$, the retained source is
$H e^{-i\omega t_c}/(i\omega)$, not zero. Its smooth counterpart is precisely T4.
Deleting that term would delete the in-window emission that §4.1 intends to keep.
The fake-provider test's wording about a steady post-pulse tail has the same ambiguity.

Please split the assertions: a globally constant current cancels to the qualified
sampling/rounding floor; an interior rise followed by a steady tail retains the
analytic rise spectrum while losing only the artificial trailing switch-off.
Include a current varying through the taper to verify that $gS'$ is retained there,
and a zero-current case. In the review's smooth-rise probe, the source was nonzero
and agreed with the correctly normalised erf oracle to $4.3\times10^{-15}$ relative.

#### R3 — P1: use operand-based, conditioned rounding budgets (§4.4, T2–T3)

An FFT normwise bound does not directly give the same relative bound after masking
and division by frequency. The active bins can be arbitrarily small compared with
the full transform, and $1/\omega$ is not a uniform scaling. Nor can T3's error budget
be proportional only to $\|\hat S_{\rm ef}\|$: that is the result of a subtraction
and may vanish while its operands and their rounding errors do not. The measured
11% edge fraction in one near-infrared example is not a general conditioning bound.

Please derive an absolute budget through multiplication, projection, FFT,
mask/crop, division and final subtraction. For example, the FFT contribution from
$g'S$ can conservatively use a constant times
$\varepsilon\log N\,\sqrt{N}\|g'S\|_2/\omega_{\rm lo}$ for a raw, full DFT, before
including the actual projection/scaling bounds; a sharper bound needs its own
justification. Retain the raw $|J|+|j_1|$ operands for shadow cancellation, add gate
and derivative evaluation errors, and keep aliasing separate. Apply the same logic
to the spectral-derivative oracle, whose intermediate spectra can be much larger
than the final source. Specify the FFT assumptions behind the constant, including
any supported non-power-of-two sizes. [Higham's FFT chapter](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch24)
is a suitable starting point, but does not certify the stated weighted/cropped bound
without this additional conditioning analysis.

#### R4 — P2: correct the DFT normalisation, phase origin and signed-edge example (§4.1, §4.3–4.4, T4–T6)

The raw DFT has neither $\Delta t$ weighting nor its phase origin at zero
([conventions §2](../../conventions.md#2-fourier-transform-and-the-sign-of-propagation-adr-017)).
For the lattice in §4.4, Poisson summation needs the factor
$e^{-i\omega_kT/2}=(-1)^k$ when relating a raw coefficient to the centred continuous
transform. The factor cancels in a consistently written identity, but not when an
analytic spectrum is used as a complex-valued oracle. T4 currently labels a continuous
transform as being in DFT units. Its raw oracle should be

$$
\frac{(-1)^k}{\Delta t}\,
\frac{H e^{-\omega_k^2\sigma^2/4}e^{-i\omega_kt_c}}{i\omega_k},
$$

up to the stated tail and alias errors. With $t_c=2$ fs and $\sigma=2$ fs on R1's
resolved lattice, this matched at $4.3\times10^{-15}$ relative; the expression as
printed in T4 gave a relative discrepancy of 1.0.

Please give all T4–T6 bounds in the same units and specify whether they are per-bin
or band norms. T5's $A e^{-(t_p/\tau)^2}$ alone is a time-domain amplitude, not a
raw spectral-error bound. The leading-taper pointwise bound in §4.3 does not directly
bound a band 2-norm; summing it introduces
$\sqrt{\sum_{k\in B}\omega_k^{-2}}$, with the appropriate DFT scale and sampling error.
Finally, §4.1's steady-tail example has an extra minus sign:
$\mathcal F[g'_+S]=S\mathcal F[g'_+]$ for constant $S$; $g'_+$ is already negative.

#### R5 — P2: make derivative evaluation safe near the breakpoints (§4.2, T1)

Clipping $\zeta$ does not prevent overflow in $\Delta/u^2$ or $\Delta/v^2$.
The printed expression evaluated at `nextafter(0, 1)` for the valid rising edge
$(a,b)=(0,1)$ returned a nonfinite value, with overflow/division-by-zero warnings,
although the mathematical derivative tends to zero. Multiplying a tiny clipped
factor by infinity cannot repair that. The clipped numerical gate also is not
literally the analytic $C^\infty$ gate at every endpoint.

Please require a scaled or logarithmic evaluation, or a justified branch that
returns the asymptotic zero before forming dangerous reciprocals. Distinguish the
analytic function from its finite-precision evaluation, and test both sides of
all breakpoints, exact endpoints, very small valid widths and representative SI
scales with warnings treated as errors. The existing gate's clipping is not a
sufficient derivative algorithm.

#### R6 — P1: separate physical propagation changes from numerical failure (§7.4)

A negligible *direct* UV component of the removed source does not bound the UV
endpoint change in a nonlinear march. Changing the near-infrared field changes later
atomic driving and therefore later UV generation. The schematic perturbation equation
is $\delta y'=D\delta y+N'[y]\delta y+\delta N[y]$; the response term can transfer an
initially near-infrared perturbation into the UV. SEP 20's L4 discussion explicitly
calls SEP 19's transfer figures study targets, not transferable error bounds.

Please remove the inference that a change above 0.5% necessarily means an
implementation error. The reviewer recommends treating it as a threshold for
investigation: check matched-input sources, pin all scientific inputs, and use
controlled/tighter-step comparisons to distinguish numerical error from propagated
physical sensitivity. A hard accuracy gate instead requires an independently
justified envelope. Define the endpoint metric and denominator, its floor, and how
integration error is resolved at `rtol=1e-2`; that local tolerance alone does not
certify a 0.5% global comparison.

**Maintainer disposition (2026-10-18).** John Travers agreed: use the 0.5% value as
an investigation threshold, with the cause determined by convergence and controlled
comparisons. The drafting agent should incorporate this decision into §7.4.

#### R7 — P2: distinguish the same-current decomposition from actual L5 (§2, §7.3, §9)

The supporting taper addendum's 4.60% → 1.22% and 1.44% → 1.45% figures apply
both gates to the *100 fs current*. Actual L5 compares a separately driven 60 fs
current with the 100 fs current. The supporting report distinguishes these, and
`benchmarks/sep20/window.py` deliberately reports an additional common-window
comparison to expose the different startup and drive histories. They are not the
same experiment. For example, its actual peak-power UV comparison is about 0.405%,
whereas the same-current gate comparison is about 0.387%.

Please reproduce the same-current split as an anchor, then report the actual
new-filter L5 comparison separately, preserving each raw current's origin and
identity. Do not require the latter to match the former using a tolerance derived
only from the differentiation residual. That residual concerns closure of a sum;
it does not separately bound both differentiated terms' errors. A ratio metric also
needs the reference norm and its perturbation in the bound. State whether
$10^{-3}$ means a dimensionless ratio (0.1 percentage point) and pin the denominator.
Replaying raw currents under a new filter should create an explicit derived analysis
record, with the original descriptor and analysis policy retained, not relabel the
raw atomic evaluation.

The reported norm reduction can be described as a reduction by about 73%; it is not
an additive or orthogonal allocation of 73% of the error to one mechanism. Keep
box-to-box sensitivity, finite-window sensitivity and error against a converged
physical reference distinct. The supporting study itself states that no tested box
is an independently valid late-time reference after norm loss. Thus the remaining
difference should not automatically be called a measured physical error or evidence
that every remaining contribution is genuine emission.

#### R8 — P2: align compatibility promises with SEP 25 (§5, §7.2–7.3, §9, Q1)

The repeated promises of bitwise-identical sources, diagnostics and L5/L6 replay
conflict with [SEP 25](sep-0025-rounding-level-reproducibility.md) and the
current testing contract. Preserving the old filter's formula and execution path is
appropriate, but numerical recomputation can differ across builds, FFTs, thread
counts and adaptive histories without being a regression.

Please retain exact equality for descriptor bytes, identities, resume keys, stored
arrays copied losslessly, masks and accounting. Use justified rounding budgets for
matched source calculations and accuracy envelopes for independent propagation or
resume. Stored numerical witnesses need provenance and a comparison budget. A
same-build bitwise observation may be recorded as evidence, not made a portable
scientific requirement. Clarify Q1's intended guarantee in these terms rather than
weakening the numerical checks or changing the old filter.

#### R9 — P2: define diagnostics, their storage and the v4 cases (§5–6, §7.2)

"Relative to the source" needs an explicit denominator: the truncating source, the
corrected source, or both. These differ markedly for the steady-current control,
where the corrected source is zero. Define each band, projection/weighting, collar
placement, units and floor. Define $\rho_-$ for a zero remainder, an empty sampled
taper and multiple nodal rows. A ratio below its floor should be unavailable with an
explanation, not NaN, infinity or a misleading zero.

The existing backend keeps only `_tdse_last_projection` and exposes it through
`tdse_last_evaluation`; adding an `edge` dictionary there does not persist per-call
history. Please decide whether the promise is a last-evaluation diagnostic or an
actual per-evaluation record. For the latter, name the storage/callback, call identity
and predictor/rejected-step labelling, bounded retention policy, and any archive
schema implications. Extend the tests to distinguish these semantics and verify that
no extra atomic calls occur.

For descriptor v4, explicitly include unwindowed SAE with `gated_derivative` and the
grid collar: today that response follows the v2 layout, while v3's validator requires
haCC or an atomic window. Test round trips and resume for that case as well as windowed
SAE and haCC, all historical v1–v3 cases, and invalid schema/filter combinations.
An unchanged optical-output schema is plausible, but should follow this complete
record/storage analysis.

#### R10 — P2: define the deferred rate-plasma measurement in current variables (§4.6, §11, Q3)

The rate route gates a polarisation, whereas this proposal holds a current. If
$J=P'$, the existing continuous source contains
$i\omega\mathcal F[gP]=\mathcal F[gJ]+\mathcal F[g'P]$ before the common propagation
factor. Simply applying the proposed formula to $P$ would hold a different quantity;
it would not implement the same current comparison.

Please give the offline experiment's exact observable, derivative/integral ordering,
Drude and ionisation-loss treatment, and discrete factors. Identify which saved fields
or currents suffice to reconstruct it. Distinguish an offline comparison on fixed
saved fields from an alternative propagated trajectory. This can remain a reported
measurement and leave the production rate route unchanged, as Q3 intends.

#### R11 — P2: make default promotion and qualification milestones explicit (§7.5, §11–12)

Phase 1 changes an omitted filter to the new form before Phase 2 has established the
real-current comparisons, SAE controls and propagated consequences. Q2 defers
promotion of the SAE research record, but ordinary new windowed SAE runs would already
receive the changed default. Q4 also defers a maintainer decision until the new
measurements, despite the closing statement that no question remains.

The reviewer recommends implementing the option first and promoting the default only
after Phase 2 and the maintainer's Q30 disposition. Specify what happens if the controls
fail, what "pass" means for the SAE controls, and which decisions block acceptance,
default promotion and Final status.

**Maintainer disposition (2026-10-18).** John Travers agreed: keep
`gated_derivative` opt-in until Phase 2 has passed and he has decided the remaining
Q30 accuracy exception. The drafting agent should align §5, §9, §11 and the deferred
decisions in §12 with this ordering.

The final milestone should explicitly include the end-of-SEP full gate, slow,
coverage and applicable Kokkos CPU qualification required by the current phased-SEP
policy. A schema update in Phase 1 does not replace those final lanes. Add the required
bounded hot-path benchmark/memory ledger, including projection, the embedded map and
diagnostic work, before describing the overhead as measured negligible. No full-length
propagation is requested by this review.

#### R12 — P2: make the public proposal independently reproducible (§1–3, §7, §12–13)

The central numerical claims currently rely on an unpublished provider development
report and addendum without a public bibliographic record. The proposal also embeds
machine-specific execution plans and an ignored archive location in the maintainer's
checkout. These do not satisfy SEP 0's requirement that the proposal stand alone;
passing the current local-reference regular expression would not establish compliance.

Please publish a self-contained solprop evidence summary with the relevant equations,
metrics, tables, limitations and immutable input/analysis identities, or cite a stable
public report. Give artifact retrieval instructions that do not depend on a particular
checkout. Keep private scheduling and storage details in the execution handoff;
hardware belongs in measured benchmark evidence. No new physical claim should depend
on access to a private development register. The reviewer could inspect the supporting
report, but did not independently regenerate its atomic data or its addendum tables.

#### R13 — P3: correct and sharpen the primary-source citations (§13)

Reference 3's author is **F. Brunel**, not P. Brunel, and the cited 1987 paper concerns
absorption at an overdense plasma surface. For the gas ionisation-current mechanism
invoked here, add or use F. Brunel, "Harmonic generation due to plasma effects in a
gas undergoing multiphoton ionization in the high-intensity limit," *J. Opt. Soc.
Am. B* **7**, 521–526 (1990),
[doi:10.1364/JOSAB.7.000521](https://doi.org/10.1364/JOSAB.7.000521).
The publisher records verify the author and the distinction in scope
([1987 paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.52),
[1990 paper](https://opg.optica.org/josab/abstract.cfm?uri=josab-7-4-521)).

The Planck profile was checked against equation (7) of
[McKechan et al.](https://arxiv.org/abs/1003.2939), and its derivative was independently
re-derived; that paper does not establish the new discrete error bound. Higham's
chapter and FFT assumptions were checked as described in R3. The bibliographic record
of [Geissler et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.2930)
was checked; its full derivation and the Brunel full texts were not independently
re-audited. No new physical coefficient or constant was transcribed.

**Review evidence and limits.** The small numerical probes above used NumPy/SciPy
and the existing gate/window resolver, with no native provider or propagation calls.
They test specific claims and counterexamples, not the proposed implementation's
qualification. The production source, output and archive schemas are unchanged by
this review.

**Verification.** `git diff --check` and codespell on this SEP passed;
`uv run pytest tests/test_seps.py` passed (197 tests, one expected process-SEP skip);
`uv run poe docs` passed with `-W -n --keep-going`, including the required tutorial
executions. The checks used this review branch's own locked environment. A separate
comparison confirmed that Part I is unchanged. Full package, slow and coverage lanes
were not run because this change contains review documentation only.

- 2026-10-18: GPT-6 completed the AI review (R1–R13); John Travers's human review
  was recorded at his request. Review status set to `Reviewed`, lifecycle status kept
  at `Draft`. John confirmed the investigation threshold in R6 and validation before
  default promotion in R11. The drafting agent owns the remaining responses and
  proposal revisions.

### Response to review — 2026-10-18

The drafting agent (Opus 5.5) revised Part I. The coordinating agent decided each item under the
maintainer's delegation, and the maintainer decided R6 and R11. The review text above is
unchanged. Section numbers refer to the revised Part I: the former §4.5 and §4.6 are now
§4.6 and §4.7, and §4.5 is the new rounding budget.

- **R1, accepted, via a computed certificate.**
  - §4 now separates the continuous identity (§4.1), the discrete definition, its alias
    error (§4.4) and floating-point error (§4.5).
  - §4.4 adds an admission rule: each taper must span at least 128 samples.
  - §4.4 also adds the steady-current certificate
    $\kappa_{\rm ss}=\|\mathrm{DFT}[g]-\mathrm{DFT}[g']/(i\omega)\|_A/\|\mathrm{DFT}[g]\|_A$,
    computed at translation for the actual lattice, gate and active band. Translation
    refuses a value above $10^{-10}$, and the value is recorded in the run metadata. It
    is kept out of the descriptor so that identities never depend on a floating-point
    result.
  - A drafting-time table of $\kappa_{\rm ss}$ against taper resolution is given; the
    default 5 fs taper has 800 samples and $6\times10^{-16}$.
  - The saddle-point estimate is kept only as motivation, labelled as such, and the
    $L^1$ "bound" is removed.
  - The new T7 refuses the review's counterexample (ratio 1.0).
  - T3 covers asymmetric and off-grid breakpoints, the admission limit and 2–8× it with
    a refinement check, the highest retained frequency, and two FFT sizes.
  - T1 no longer claims an exponential trapezoid bound or treats a sampled $g^{(5)}$ as
    a bound. It uses an independent closed form, a convergence-rate check and a reported
    edge weight.
- **R2, accepted.** T2 is split into four cases: (a) a globally constant current gives
  zero to the budget plus the certificate floor; (b) zero current; (c) an interior rise
  plus a steady tail keeps the analytic rise; (d) a slope through the taper keeps $gS'$.
  §4.1 states the step case $He^{-i\omega t_c}/(i\omega)$. The §7.2 fake-provider
  drift now rises inside the plateau and is required to keep that rise.
- **R3, accepted.** The new §4.5 gives an absolute, operand-based budget in raw FFT
  units. It sums today's floor (raw $|J|+|j_1|$), the edge transform
  $c_F\varepsilon\log_2N\sqrt N\|g'(|J|+|j_1|)\|_2/\omega_{\rm lo}$, the
  derivative-evaluation error, and the division and subtraction. Each term is scaled by
  the absolute projection and the factors. Aliasing is kept separate (§4.4). The FFT
  assumption covers 2-, 3- and 5-smooth sizes; T3 checks two sizes, and other sizes
  are flagged. The spectral-derivative oracle has its own budget, and T3 uses the sum.
- **R4, accepted.**
  - §4.4 states the raw DFT normalisation and the $(-1)^k$ origin factor.
  - T4 uses the reviewer's corrected oracle.
  - T4–T6 are in raw units, per bin unless a band norm is stated. T5's bound is now
    spectral: $Ae^{-(t_p/\tau)^2}W/(\Delta t\,\omega_k)$.
  - §4.3 gives both the per-bin and the band-norm bound, the latter with
    $\sqrt{\sum\omega_k^{-2}}$ and the sampled edge weight.
  - The extra minus sign in §4.1 is removed.
- **R5, accepted.** §4.2 evaluates $g'$ through $s=(x-a)/\Delta$. It returns the
  asymptotic zero, with no reciprocal formed, when $\min(s,1-s)\le1/740$ (where
  $|h|<10^{-300}$), and it distinguishes the analytic function from its evaluation.
  T1 tests both sides of every breakpoint, the exact endpoints, `nextafter`
  neighbours, widths from $10^{-30}$ s to 1 s and SI scales, with warnings as errors.
- **R6, accepted (maintainer).** In §7.4, 0.5% in 90–200 nm is an investigation
  threshold, not an implementation-error inference. The cause is named by matched-input
  sources, pinned inputs and an `rtol=3e-3` control. §7.4 defines the endpoint metric,
  its denominator (the current-form endpoint) and its floor (SEP 20 L6's 0.073%
  restart sensitivity). It also states that `rtol=1e-2` certifies no 0.5% global
  comparison.
- **R7, accepted.**
  - §2 labels the table as the same-current split and says "reduces the norm
    difference by about 73%". It calls the remainder the window sensitivity that
    remains, with no converged late-time reference.
  - §7.3 reports (i) the recorded figures, (ii) the same-current anchor and (iii) actual
    L5, each with its own expectation. (iii) borrows no tolerance.
  - (ii)'s tolerance is the dimensionless ratio
    $\delta=\|F_{100}-F_{60}\|_B/\|F_{100}\|_B$, within $(2+\delta)(r+e_g)$, about
    0.11 percentage points.
  - The replay writes a derived analysis record that keeps the raw descriptor.
  - §9 states Q30's expectation in the same terms.
- **R8, accepted.** §5, §7 (preamble, §7.2, §7.3) and §9 restrict exact equality to
  descriptor bytes, identities, resume keys, lossless copies, masks and accounting.
  Matched source recomputations use the §4.5 budgets, and propagation uses envelopes.
  The old filter keeps its formula and code path; a same-build bitwise repeat is
  evidence, not a requirement. Q1 is restated in these terms.
- **R9, accepted, with the smaller scope.**
  - §5 makes the `edge` entry a last-evaluation diagnostic in `tdse_last_evaluation`,
    with no per-call store and no archive change. Per-call analysis is an offline replay
    (§7.3, §7.5).
  - The bands are `active_mask` and `inband_mask`, in the existing raw modal units with
    the collar. Both the truncating and the corrected norms are reported, with each
    ratio's denominator stated and a floor for each.
  - A ratio below its floor is `None` with a reason. $\rho_-$ is defined over rows,
    with the zero-remainder case covered; the admission excludes an empty taper.
  - §6 and §7.2 add descriptor v4 for unwindowed SAE on the collar, windowed SAE,
    windowed and unwindowed haCC, with round trips and resume for each. All v1–v3 cases
    and the invalid combinations are covered, and a test confirms no extra atomic calls.
- **R10, accepted.** §4.7 defines the offline rate-plasma measurement in current
  variables.
  - The route's source is $i\omega\mathcal F[gP]=\mathcal F[gJ]+\mathcal F[g'P]$, and the
    held form is $\mathcal F[gJ]-\mathcal F[g'J]/(i\omega)$.
  - $J=J_{\rm free}+J_{\rm loss}$ is recomputed on fixed saved L6 rate fields, with the
    run's own running-integral rule, crop and prefactor.
  - A closure residual measures the discrete $P'=J$.
  - It is reported only; the production rate route is unchanged.
- **R11, accepted (maintainer).** `gated_derivative` is opt-in through Phases 1 and 2
  (§5, §9). Phase 3 promotes it only after Phase 2 passes and the maintainer decides
  Q4 and Q30. §7.5 defines "pass" for the SAE controls, and says the record stays on
  `"time_collar"` if they fail. §11 lists what blocks acceptance, promotion and Final,
  and adds the end-of-SEP full, slow, coverage and Kokkos CPU lanes. §7.2 adds a
  bounded hot-path benchmark with a memory ledger before any "negligible" claim. §12
  reopens Q4 as deferred, and the "no question remains" line is removed.
- **R12, accepted in part.**
  - Part I no longer names machines, checkout paths or scheduling. Those are in
    `benchmarks/sep26/README.md`.
  - The new
    `benchmarks/sep26/evidence/0133-summary.md`
    holds the equations, metric, tables, limitations and the SHA-256 identities of the
    provider study's records, together with its revisions and retrieval by hash and
    archive layout. It is cited as reference 5, with the provider report as provenance.
  - Not done: the raw atomic arrays stay in the evidence archive and are not yet public,
    as for SEPs 12 and 20.
- **R13, accepted.** Reference 3 is now F. Brunel, *J. Opt. Soc. Am. B* **7**, 521
  (1990), for the gas ionisation current. The 1987 paper concerns absorption at an
  overdense plasma surface, so it is dropped. Reference 1 cites eq. (7) and reference 2
  cites the chapter.

**Checks.** `git diff --check`, codespell on the changed files,
`uv run pytest tests/test_seps.py` and `uv run poe docs` (strict).

- 2026-10-18: Part I revised in response to R1–R13 (above). Status stays `Draft`;
  `gated_derivative` is opt-in until Phase 3.
- 2026-10-18: Accepted by the maintainer after the review and response; Part I frozen.
- 2026-10-18: Status set to `In progress`; phase 1 started on
  `feature/sep-0026-phase-1` (solprop and solprop-kokkos).
- 2026-10-18: Phase 1 implemented on `feature/sep-0026-phase-1` (solprop and
  solprop-kokkos) and reported in Part II; awaiting the coordinating agent's review.
- 2026-10-19: Phase 2 (validation) run on `feature/sep-0026-validation` and reported in
  Part II; the maintainer's conditional Q4/Q30 decision is recorded there and its
  condition holds. Awaiting the coordinating agent's review.
- 2026-10-19: Phase 3 on `feature/sep-0026-phase-3`: `"gated_derivative"` became the
  haCC and windowed default, the SAE research record moved to it under a new identity,
  Q30 was restated and the rate-plasma question (Q10) moved to the roadmap. The
  end-of-SEP lanes passed, with one intermittent coverage-lane failure unrelated to
  this SEP. Reported in Part II.
- 2026-10-19: Final, prepared on `feature/sep-0026-phase-3` for the maintainer's
  approval at merge. Every phase is reported and the end-of-SEP lanes passed (Phase 3
  report). TDSE runs on `numpy_ref` only, under SEP 12's recorded exception;
  `numba_cpu` and solprop-kokkos refuse it by name. Moved to the roadmap: the rate
  plasma's collar truncation (Q3, Q10), the remaining window sensitivity (Q30, haCC
  and SAE) and the propagation of the moved SAE record on its own collar. Moved to
  `archive/`.
