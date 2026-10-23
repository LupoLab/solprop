# SEP 19 — Field-scale radial quadrature for single-mode capillaries

| | |
|---|---|
| **SEP** | 19 |
| **Title** | Field-scale radial quadrature for single-mode capillaries |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-12) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-12 |
| **Resolution** | Accepted 2026-10-12 by John Travers after human and AI review (draft merged for review in ffdeac8, review revisions in 24b9379); Q6–Q8 are Phase 1 deliverables. Final, approved by John Travers, 2026-10-13, after the end-of-SEP lanes. |
| **Requires** | [SEP 12](sep-0012-tdse-nonlinear-response.md) (the TDSE response whose cost motivates this) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 for John Travers. Phase 1 (translation and numpy_ref): merged 2026-10-12 (solprop f29d0a4, solprop-kokkos ca9827f). Phase 2 (numba_cpu and Kokkos CPU): merged 2026-10-12 (solprop e36e469, solprop-kokkos de600f6). |
| **Manual pages** | [Radial nodes for expensive responses](../../../guide/waveguides/radial-quadrature.md); [statistics](../../../guide/results/statistics-catalogue.md) |

## Part I — Proposal

### 1. Abstract

For one HE₁₁ mode in a gas that is uniform across the core, every quadrature node sees
the same waveform scaled by $\eta = J_0(u_{01} r/a)$. The radial projection is
therefore a one-dimensional integral over the field scale η, and each node is an
independent atom. This SEP proposes three changes.

1. **Radial rules that are Gaussian in η** for the HE₁₁ projection measure. They are
   meant for responses so expensive that the number of atoms per call matters: the
   TDSE of SEP 12 and its all-electron successors. There are two variants:
   - a Gauss rule;
   - a Gauss–Radau rule with one node on the axis, whose response can also supply
     supported on-axis statistics without another evaluation at the same state.
2. **Separate on-axis plasma statistics**, leaving existing node-based statistics
   unchanged. For a single HE₁₁ mode the default rule samples almost on axis; a
   few-node field-scale rule may miss a substantial part of the axial plasma peak.
   Multimode maxima need not lie on axis, so the two quantities stay distinct.
3. **The mode-averaged transform becomes an explicit kind** instead of being inferred
   from a node count of one. A one-node rule then means one node, and the one-node
   field-scale rule reproduces the effective-area model through the ordinary
   quadrature path.

**How success is judged:**

- The rules integrate the linear and Kerr terms exactly: from two nodes for the Gauss
  rule, three for Radau.
- On the rate model, it reproduces the benchmark errors of `benchmarks/radial/`.
- Its one-node special case reproduces the existing mode-averaged source within
  the independently bounded moment and floating-point error.
- The three rate-model backends agree on both the existing and the new statistics.
- No statistics request launches an additional TDSE atom.

### 2. Motivation and scope

An atomic response costs one atom per node per call. The default capillary rule has 33
nodes: `n_r=32` rounded up by the Kronrod extension. On a CPU, or for an all-electron
model whose single atom takes minutes, the cost is close to linear in the node count.
So the node count is a main cost lever that preserves the response model, while
changing the transverse discretisation error.

The usual cheap alternative is the mode-averaged (effective-area) model. It evaluates
one atom at the effective-area intensity. For ionisation-driven sources it is poor:
- on the SEP 12 rate-model parent, it gets the source at the launch plane 85% wrong;
- over a full run, it is 30% off in the final 120–200 nm energy and up to 16% off in
  peak power.

The analysis in `benchmarks/radial/README.md` measured three things:

- **How few nodes each rule needs.** This was measured on the rate model, and on a
  dense, 157-atom SAE scan of the SEP 12 launch waveform.
- **How a source error carries through the measured surrogate propagation.** In
  this case a 1–2% source error gives a comparable error in the generated ultraviolet,
  and about 10× less in the field and the peak power. This is not a stability bound
  for other pulses or for the TDSE evolution.
- **Where the rules stand against the effective-area model.** Gauss rules in η with 4
  or more nodes are more than ten times better than it in the measured surrogate.
  16 nodes give sub-percent source errors on the one scanned SAE waveform.
  Neither that error nor the provider's accuracy is a universal TDSE guarantee.

**In scope:**

- the two field-scale rules, their guards and their selection keyword;
- separately named on-axis rate-model statistics on supported modal routes,
  preserving all existing node statistics;
- the explicit mode-averaged kind;
- a manual page on choosing radial nodes, with the effective-area comparison.

**Out of scope:**

- changing any default node count or rule;
- multimode and HE₁ₘ rules, and transversely non-uniform media;
- adaptive panels, and interpolation or smoothing across nodes in η;
- a hybrid that integrates the perturbative part analytically;
- a TDSE ionisation statistic (none exists; see §5.3).

### 3. Context

**The radial rule today.**
- `solprop.modes.quadrature._radial_rule` builds a Gauss–Legendre rule on
  $\xi = r/a$. With `kronrod=True`, it extends it to the $2n+1$ Gauss–Kronrod rule
  with an embedded coarse rule.
- `transverse_quadrature(modes, n_r=…, kronrod=…)` builds the rule for a mode set.
  `quadrature_rule` refuses `n_r < 2`.
- `Simulation.from_capillary` exposes `n_r` (default `None`, selecting
  `default_radial_order`; 32 for one mode) and `kronrod` (default `True`). See [Waveguide modes](../../../guide/waveguides/modes.md) and
  [Multimode capillaries](../../../guide/waveguides/capillaries.md).

**From rule to matrices.**
- `solprop.translate.modal.multimode_transform` turns the rule into the `synthesis`,
  `projection`, `projection_coarse` and `node_weights` arrays of the kernel contract,
  plus a `taper_scale` for radius changes along $z$.
- Every backend (numpy_ref, numba_cpu, Kokkos) and the TDSE seam read only these
  arrays. No backend builds or inspects a rule.

**The TDSE route.**
- It evaluates one atom per node per call and never reads the quadrature alarm.
  `quadrature_error` and the plasma statistics are in `TDSE_UNAVAILABLE_STATISTICS`
  (`solprop.translate.tdse_source`).
- See [TDSE response in a helium capillary](../../../guide/media/tdse.md).

**The mode-averaged route.**
- `mode_averaged_transform` builds a one-node transform whose synthesis carries the
  effective-area conversion.
- `compile_responses` then folds the matching area factor into the prefactor.
- The flag `Transform.mode_averaged` is not stored. It is inferred as "no geometry, no
  modal tables, one node" (`solprop/translate/modal.py`). Its consumers include:
  - the prefactor's area factor (`solprop/translate/responses.py`);
  - the population weights (`solprop/backends/populations.py`);
  - the recorded simulation kind and the TDSE admission check
    (`solprop/driver/simulation.py`, `solprop/translate/problem.py`).
- No public route builds a one-node quadrature today, so the inference has never been
  wrong in use. But a one-node quadrature transform would be silently treated as
  mode-averaged, and its weights converted twice. Injecting a one-node rule in the
  benchmark did exactly that.

**The statistics.**
- `electron_density` selects the densest total-electron node;
  `species_electron_density` reports all species at that same node. Species peak rates
  take maxima over nodes, while the legacy single-population rate can use the
  densest node. These existing reductions are preserved, including their field and
  envelope details (`backends/populations.py`, `stats.species_statistics` and each
  backend's `_ionisation` path). They are not generally innermost-node values.
- `peak_intensity` is already evaluated on the axis, from the mode profiles
  (`StepDriver._on_axis_synthesis`, shared by the backends).
- For a single HE₁₁ profile the default rule has an innermost node at
  η = 1 − 10⁻⁶, so a monotone rate model gives nearly axial node maxima. A small
  field-scale rule can differ substantially (§4.4).

**The roadmap.** This answers in part the roadmap's later radial and modal
optimisation item (Q7), which kept 33 nodes for native TDSE work until a denser rule
was checked.

### 4. Theory

#### 4.1 The projection as an integral over the field scale

**The field at each node.** For the single mode
$F(r) = J_0(u_{01} r/a)$, with $u_{01}$ the first zero of $J_0$ (Marcatili and
Schmeltzer [1]), the field at radius $r$ is $\eta A(t)$, where
$\eta = J_0(u_{01}\xi)$ and $A$ is the on-axis field. In a medium that is uniform
across the core, the response at a node is a functional of its field alone:
$P(r,t) = \mathcal{F}[\eta A](t)$.

**The projected source.** It is

$$
S(t) = \int_0^1 \mathcal{F}[J_0(u_{01}\xi)A]\, J_0(u_{01}\xi)\,\xi\,d\xi
     = \int_0^1 \mathcal{F}[\eta A]\, d\mu(\eta),
\qquad
d\mu = \frac{\eta\,\xi(\eta)}{u_{01}\,J_1(u_{01}\xi(\eta))}\, d\eta ,
$$

up to the mode normalisation that solprop already carries. The same spatial
projection applies separately to an atomic current $J$; it does not change the
distinct polarisation/current UPPE prefactors or Fourier convention of SEP 12.

**The measure.** $\mu$ is positive and finite on $(0, 1]$. The derivative is
$d\eta/d\xi=-u_{01}J_1(u_{01}\xi)$; reversing the integration limits absorbs
its minus sign. Near the axis, $\xi|d\xi| \approx (2/u_{01}^2)d\eta$,
so $d\mu/d\eta \to 2/u_{01}^2$ as $\eta\to1$. At the wall
$d\mu/d\eta\sim\eta/[u_{01}J_1(u_{01})]$, so the measure vanishes smoothly.
Both $\eta$ and the reduced weights are dimensionless; physical area and modal
normalisation are restored below. A pressure gradient along $z$ changes only
the density, a scalar, so the same rule applies at every $z$. A radius change along
$z$ leaves $\xi$ and the rule unchanged; the existing `taper_scale` carries it.

**Validity.**
- One Marcatili HE₁₁ spatial profile, with the existing scalar or Cartesian
  two-component representation: each component is $\eta$ times its on-axis value.
  A mode merely labelled HE₁₁ with a different radial profile does not qualify.
  Two separately listed modes are outside this SEP even if they form a degenerate
  pair; the admitted representation has one spatial mode.
- A transversely uniform medium.
- No transverse coupling inside the response.

A general multimode field is not one waveform times a scale, so this rule is not
admitted for multimode sets.

#### 4.2 The rule

**Construction.** The $n$-node rule is the Gauss rule for $\mu$ on $\eta \in (0,1]$,
with nodes $\eta_k$ and weights $\lambda_k > 0$, for $n\ge2$.
The API's $n=1$ case is the separate moment-matched rule of §4.3. For $n\ge2$,
the Gaussian rule is computed from the recurrence coefficients of $\mu$,
discretised on a fine Gauss–Legendre rule in $\xi$, by the Stieltjes procedure (Gautschi [2]) and the Golub–Welsch eigenvalue method [3].

**On solprop's grid.** It is stored as nodes $\xi_k = \xi(\eta_k)$ with weights. In
that form it enters `multimode_transform` exactly where the Gauss–Legendre rule does
today (but see the weight convention below).

**Exactness.** The rule integrates polynomials in $\eta$ of degree $2n-1$ exactly.
For $n \ge 2$ this includes the linear and Kerr terms, $\eta$ and $\eta^3$.
This is spatial exactness for local linear and cubic functionals, including a
time convolution that remains cubic under a constant field rescaling; it does not
remove temporal discretisation or propagation error. Non-polynomial response terms
still require convergence checks. In floating point the statement is limited by the
accuracy of the discretised measure and the recurrence.

**The Radau variant.** The $n$-node Gauss–Radau rule for $\mu$ fixes one node at
$\eta = 1$, the axis. Its other $n-1$ nodes come from the same Jacobi matrix, with the
last diagonal entry modified so that 1 is an eigenvalue (Golub [5], §7).
If $J_{n-1}$ is the leading block, $b_{n-1}$ its coupling to the last row, and
$e$ the last unit vector, the replacement is
$\alpha_{n-1}^{R}=1-b_{n-1}^2 e^T(I-J_{n-1})^{-1}e$. This follows by the
Schur complement of $I-J_n$; solve the tridiagonal system, do not form an inverse.
Its weights are positive, and it is exact to degree $2n-2$, so the Kerr term needs $n \ge 3$.

**What the axis node buys.**
- The most intense atom, on the axis, is evaluated in every call. A response that
  switches on just below the peak intensity is therefore always sampled. With the Gauss
  rule at five nodes, the innermost atom sees only 92% of the peak intensity.
- A supported per-node observable is available on the axis without another atom
  when it belongs to the state being reported (§4.4, §5.3).

**What it costs.**
- On smooth responses it trails the Gauss rule at 4–5 nodes: on the rate model,
  the draft analysis reports 3×10⁻² against 2×10⁻³ at 4 nodes, and 10⁻³ against
  4×10⁻⁴ at 5 nodes. The rate-model Radau figures still need their reproducible
  evidence record in Phase 1; there is no general ordering of the errors.
- On the TDSE source it is in the same placement-dependent band as the Gauss rule from 4
  nodes (§4.5). At 3 nodes it is 23% off.
- Its one-node member would be the on-axis atom alone, which overweights ionisation.

**The axis node and the weight convention.** Today the polar rule stores weights
for $d\xi$ and `TransverseQuadrature.weights` supplies the radial Jacobian.
For the new rules define the dimensionless radial area weights
$q_k=\lambda_k/\eta_k$. Then

$$
2\pi a^2\sum_k q_k\eta_k\mathcal F[\eta_k A]
=2\pi a^2\sum_k\lambda_k\mathcal F[\eta_k A].
$$

At the Radau axis node $q_k=\lambda_k$ is finite. Trying to encode it as a
$d\xi$ weight $q_k/\xi_k$ would divide by zero. `TransverseQuadrature` therefore
carries explicit radial area weights and their coarse counterparts. Its `weights`
method uses them once, multiplying by $a^2$ and the azimuthal weights; callers still
receive physical weights in m². Existing polar rules retain their legacy $d\xi$
representation and multiplication order; Cartesian and slab rules are unchanged.
There must be exactly one active weight representation, validated on construction.

These are **source-adapted area weights**. They do not generally integrate a
constant area density exactly: that integral corresponds to $1/\eta$ against
$\mu$, outside polynomial exactness. For example, five Gauss field-scale nodes
have $2\sum q_k\simeq0.95944$, whereas the exact reduced area is 1. Do not
renormalise these weights to the geometrical area: that would destroy the linear
and cubic moments. Integrated populations and other non-source observables require
separate convergence checks; a source error bound does not certify them. In
particular, uniform preionisation is an explicit diagnostic test case.

**The coarse rule.** Neither variant has an embedded coarse rule. `projection_coarse` equals
`projection`, as with `kronrod=False` today, and the quadrature alarm is off.

#### 4.3 The effective-area special case at one node

**Derivation.** Write $M_k = \int \eta^k d\mu$. The effective-area model evaluates one
response at physical field $E_e=a_m/\sqrt{B A_\text{eff}}$, where
$B=\varepsilon_0c/2$ and $a_m$ is the internal vacuum-normalised modal amplitude.
This is the field convention of [conventions §4](../../conventions.md), not an
identification of broadband Poynting intensity with $B|E_e|^2$. With
$A_\text{eff} = (\int |F|^2 dA)^2 / \int |F|^4 dA$, its field scale is
$\eta_e^2 = \int J_0^4\,dA / \int J_0^2\,dA = M_3/M_1$. Its weight makes the Kerr term
exact: $\lambda_e \eta_e^3 = M_3$. Then $\lambda_e \eta_e = M_1$, so the linear term is
exact too. This is the unique one-node rule exact for $\eta$ and $\eta^3$. For HE₁₁,
$\eta_e = 0.752$, about 57% of the on-axis intensity.

**This is not the one-point Gauss rule for $\mu$.** That rule has
$\eta_G=M_1/M_0\simeq0.62423$ and $\lambda_G=M_0$, and is exact for 1 and
$\eta$, not $\eta^3$. `field_scale, n_r=1` deliberately selects
$(\eta_e,\lambda_e)$ instead. Polynomial exactness in §4.2 applies only to the
Gaussian family with $n\ge2$; the special member matches moments 1 and 3.

**Normalisation identity.** Put $I_2=2\pi a^2M_1$ and $I_4=2\pi a^2M_3$.
Then $N=B I_2$, $A_\text{eff}=I_2^2/I_4$ and
$\eta_e/\sqrt N=1/\sqrt{B A_\text{eff}}$. The one-node physical weight is
$2\pi a^2\lambda_e/\eta_e=A_\text{eff}$; its dual projection coefficient is
$B A_\text{eff}\eta_e/\sqrt N=\sqrt{B A_\text{eff}}$.
Thus both routes drive the same local response and give the same projected source,
including a non-polynomial response, provided their other response settings agree.
The identity also holds along a scale-invariant taper.

**Consequences.** Phase 1 checks this algebra and source equivalence against the
existing mode-averaged route. It does not promise identical diagnostic meanings or
relax response admission: TDSE remains a resolved-route feature, so its one-node
case is checked using a deterministic provider double, not an otherwise refused
mode-averaged TDSE run.

#### 4.4 The innermost node and the statistics

The largest node of the Gauss rule is below 1:

| Rule | Innermost node η | Intensity there (η²) |
|---|---|---|
| Gauss–Kronrod in ξ, 33 (default) | 0.999999 | 1.00 |
| Gauss in ξ, 20 | 0.99998 | 1.00 |
| Field scale, 16 | 0.995 | 0.99 |
| Field scale, 8 | 0.982 | 0.96 |
| Field scale, 5 | 0.958 | 0.92 |
| Field scale, Radau, any $n$ | 1 | 1 |

**Why this matters.** A rate scaling as $I^K$ with $K \approx 8$–10 at the innermost
5-node point is about half its on-axis value. In the low-depletion limit, a
separable power-law rate gives the same factor in the final electron density.
This bias in a sampled maximum is distinct from error in the weighted source
integral: accurate projection need not imply an accurately sampled axial peak.

**The fix.** Offer separate on-axis statistics beside the existing node-based
ones. The latter remain useful and can describe an off-axis maximum in a multimode
field. On a rate-model route an axial statistic costs at most one additional
node's rate integration per requested observation. A Radau axis row already
evaluated at that state can be reused. TDSE must never evaluate an extra atom for
statistics (§5.3).

#### 4.5 Evidence

All numbers are from `benchmarks/radial/` and its `evidence/` records, on the SEP 12
helium parent (800 nm, 1 mJ, 6 fs, peak ionised fraction about 1.6×10⁻³).

| Source of the error | 5 η | 8 η | 16 η | 20 ξ | 33 (default) | Mode-averaged |
|---|---|---|---|---|---|---|
| Rate model, total source at the launch | 4×10⁻⁴ | 4×10⁻⁷ | 10⁻¹⁰ | — | ≈10⁻¹⁰ | 85% |
| TDSE (SAE), dense-scan source at the launch | 1.5% | 1.4% | 0.3% | 0.12% | 0.14% | — |
| Propagated, TDSE-like structure, final 120–200 nm field | 0.4–2% | 0.1–1.1% | ≤0.02% | ≤0.02% | 3×10⁻⁸ | 150% |
| Propagated, peak power, worst plane | 0.1–0.5% | 0.06–0.3% | ≤0.013% | ≤0.01% | 2×10⁻⁸ | 16% |

**The Radau variant on the dense TDSE scan.** Errors over the full band:

| Nodes | 3 | 4 | 5 | 6 | 8 | 10 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| Gauss in η | 6.2% | 2.1% | 1.5% | 0.25% | 1.4% | 0.7% | 0.28% | 0.14% |
| Radau in η | 23% | 1.8% | 0.4% | 1.7% | 1.7% | 0.7% | 0.17% | 0.08% |

At 5 nodes Radau is the better of the two, and at 6 the worse. Neither is consistently
ahead below about 15 nodes, which is the placement dependence described next.

**The TDSE source is smooth but finely structured in η.** It oscillates along η²
with a period of about 0.15 and about 2% per atom. So below about 15 nodes the error depends on where the
nodes fall, and the 0.5–2% band in the table is a spread, not a trend. The propagated
rows use a calibrated stand-in for that structure on the rate model; they show how a
source error carries through this run, not what a TDSE run would give.

**Evidence limits.** The rule scores are evaluations of a spline through 157 saved
SAE rows, not fresh TDSE evaluations at each proposed node. The reference changes
by $7.3\times10^{-6}$ when using every other row and by $4.1\times10^{-4}$
with trapezoidal integration; these checks are not rigorous error bounds. The
reported errors use the relative spectral norm of `benchmarks.radial.dense_analysis`,
with full-band and 120–200 nm masks kept distinct. Only one waveform and one SAE
setup were scanned. Stronger fields, later compressed pulses and all-electron
responses need their own node-convergence studies. The Radau scores were recomputed
from those same saved rows during review; their reproducible evidence record is a
Phase 1 deliverable, without new atomic solves.

### 5. Proposed interface

#### 5.1 Selecting the rule

The radial rule becomes one keyword, `radial_rule`, on `Simulation.from_capillary`, the
other capillary constructors that take `n_r` today, and `transverse_quadrature`.
For a polar capillary the selection is:

| `radial_rule` | Nodes for `n_r = n` | Coarse rule | Valid for |
|---|---|---|---|
| `"gauss_kronrod"` (default) | smallest odd count ≥ $n$, at least 3 | embedded Gauss | existing supported geometries |
| `"gauss"` | $n$ | none | existing supported geometries |
| `"field_scale"` | $n$ | none | one HE₁₁ mode, uniform medium |
| `"field_scale_radau"` | $n$, one on the axis | none | one HE₁₁ mode, uniform medium |

`n_r` keeps its current meaning: `n_r=32` still produces 33 Kronrod nodes
with a 16-node embedded Gauss rule, not 65 nodes. For a transverse product rule
the table counts radial nodes; the total is multiplied by the angular node count.
Existing non-polar geometry behaviour is preserved; this table does not promise
a new embedded rule for Cartesian geometries. Field-scale rules are radial-only
(`n_theta=1`, no forced angular subdivision), so there is no angular coarse rule
that could accidentally reactivate the quadrature alarm.

Selecting either field-scale rule explicitly selects the resolved modal route,
even at `n_r=1`; it must not pass through an effective-area convenience shortcut.
An explicit contradictory route selection is refused.

`kronrod=` is removed from the affected public selection interfaces (Q1, resolved).
No compatibility alias is required by that decision. Phase 1 inventories all
constructors, helper factories, serialisation and documentation that expose it;
internal embedded-rule flags may remain. Field-scale selection through a low-level
factory must still validate the actual mode profile before translation.

The rate-model convenience constructors take the new keyword. TDSE continues to
use `Simulation`'s explicit response sequence; this SEP does not add a
`response=tdse` convenience argument. For an otherwise admitted SEP 12 setup,
with `replace` from `dataclasses`, the proposed selection is:

```text
rule = transverse_quadrature(
    [mode], radial_rule="field_scale", n_r=8,
)
sim = Simulation(
    grid, [mode], [tdse], initial,
    settings=replace(tdse_settings, quadrature=rule),
)
```

Here `mode`, `tdse_settings` and the other inputs are the existing qualified
explicit configuration; only its rule changes from 33 atoms to eight per call.

**What is refused:**

- **The wrong mode set.** Either field-scale rule with more than one mode, or with a
  mode that is not a supported Marcatili HE₁₁ profile, is refused with `ModeError`,
  which names the mode set.
- **The wrong node count.** `n_r < 1` is refused. `n_r = 1` is admitted for
  `"field_scale"` only, as the effective-area model on the quadrature path (Q2,
  resolved). `"field_scale_radau"` needs `n_r ≥ 4`: it is exact for Kerr from 3, but at
  3 it is 23% off on the TDSE. Every other rule keeps `n_r ≥ 2`.
- **An unsupported geometry or panel rule.** Field-scale rules require a polar
  core and the entire interval $0\le\xi\le1$; non-empty `breaks` and non-polar
  geometries are refused, not ignored.
- **A transversely non-uniform medium,** if one is ever admitted on the capillary
  route, is refused with either field-scale rule.
- **Unknown rule names or non-integer counts** are rejected before constructing
  arrays. There is no silent substitution of a conventional rule.

The run metadata records the rule name, requested and realised radial counts, total
node count, and the actual nodes and physical weights beside the existing
quadrature record. The one-node special case is identified explicitly. Resume
identity must distinguish the new rules and their realised weights.

#### 5.2 Separately named on-axis statistics

All existing statistics retain their present meanings, shapes and reductions.
In particular, `species_ionisation` remains an array of fractions with shape
`(n_species, n_nodes)`, not an integrated quantity. `species_ionisation_energy`
keeps its physical weighted sum, subject to the quadrature limitation in §4.2.

Add four names to `StatsSpec` and the catalogue:

| Name | Meaning | Shape | SI units |
|---|---|---|---|
| `on_axis_electron_density` | Total density left at the end of the time window on axis | `()` | m⁻³ |
| `on_axis_peak_ionisation_rate` | Largest temporal rate over the plasma species on axis | `()` | s⁻¹ |
| `on_axis_species_electron_density` | Each species' final electron density on axis | `(n_species,)` | m⁻³ |
| `on_axis_species_peak_ionisation_rate` | Each species' largest temporal rate on axis | `(n_species,)` | s⁻¹ |

Envelope rates use the existing cycle-averaged model. Non-ionising species retain
zero entries in the established species order; total density is the sum of the
species densities at the same axis point. These names are opt-in and are not
added to the default statistic set. There is no `axis=` selector and no approximate
quantity labelled as on-axis.

Initially admit these names for resolved modal rate-model routes whose physical
axis synthesis is already supported by the shared driver. This includes the
single-HE₁₁ capillary and existing static/scale-invariant multimode transforms;
in the latter the axis value need not be the transverse maximum. General modal
E/H tables, spatial geometries and mode-averaged runs are refused by name for the
new statistics until separately qualified. Existing statistics on those routes
remain available.

Synthesis uses the physical electric field in V/m on the fine time grid, with
exactly the existing pad/crop and taper factors; the Poynting intensity statistic
is not a substitute for that field. Rate tables, preionisation, mixture densities
and the selected running-integral method are identical to the existing node
calculation. Evaluate all requested axis quantities together once per observation.
When the rule has an axis node, reuse its row from that same observation; never
reuse a stale stage or rejected-step result. Do not append an extra row to node
inventories or include it in their weighted integrals.

The initial new names are saved-plane statistics (`between_planes=False`), so
periodic requests are refused consistently. Existing periodic statistics are
unchanged. This keeps the additional work out of the step loop.

For a small-node HE₁₁ rule, the guide explains that the existing maximum is a
sampled-node maximum and may miss the axis. The ratio $\eta^{2K}$ is only an
illustration for a rate proportional to $I^K$; no universal order $K$ or numerical
error bound is inferred for a user-supplied rate or a saturated ionised fraction.

#### 5.3 The TDSE route

**Unchanged response scope.** Existing plasma statistics and the quadrature alarm
stay refused on the TDSE route. The four new on-axis names are also added to its
explicit unavailable-statistic set. Neither field-scale rule changes the scalar
field-grid, Adams, NumPy optical-backend and other admission requirements of
SEP 12. Numba and Kokkos continue to refuse TDSE by name.

**No extra atom for statistics.** This is a requirement, including for a single
HE₁₁ capillary. There is no opt-in path that launches an axial atom solely to
supply a statistic.

**A future TDSE ionisation statistic** must come from an observable the atoms
already evaluated for the radial integral. Without an axis node it can report
only the maximum over those nodes, explicitly described as such. With a Radau
axis node, an additional on-axis statistic can read that row's own observable,
provided the result belongs to the state and time window being reported. A last
right-hand-side result is not automatically valid at a saved or interpolated
plane. If no matching result is retained, the statistic is unavailable; no extra
atomic evaluation is allowed to repair that absence.

**The observable and its retention are out of scope.** Norm lost to an absorber,
probability outside a radius and ground-state depletion are not interchangeable
ionisation measures. Choosing one and changing the atomic result/lifetime contract
belong to the provider and a separate proposal. This SEP only ensures that the
axis row, when present, has an unambiguous identity.

#### 5.4 The explicit mode-averaged kind

**The change.**
- `Transform` gains a stored field, `kind`, with the values `"mode_averaged"`,
  `"modal"`, `"radial"`, `"free"` and `"slab"`.
- `mode_averaged_transform` sets `"mode_averaged"`; resolved waveguide quadrature
  and general modal tables set `"modal"`, whatever their node count. Spatial
  constructors set `"radial"`, `"free"` or `"slab"` to match their geometry.
- Construction validates consistency with geometry, modal tables, weights and
  the mode-averaged constraints. There is no one-node inference as a default.
- `Transform.mode_averaged` becomes `kind == "mode_averaged"`. Its consumers (§3) retain their existing purpose and read this explicit value.

**The result.** A one-node quadrature transform is an ordinary modal transform. No
area factor is folded into its prefactor, its population weights are its node weights,
and its run records `simulation_kind="multimode"`.

**The Kokkos backend** treats `n_nodes == 1` as "no coarse rule" when it packs the
contract. That is correct for both kinds and stays.

### 6. Implementation strategies

**The layer.** Public mode/rule selection remains in `solprop.modes.quadrature`.
The numerical construction has two options:

- A: compute Stieltjes recurrence coefficients on a converged discrete measure,
  then solve the Jacobi eigenproblem with SciPy. Cache immutable rules by family
  and node count; a taper does not require a new rule.
- B: tabulate nodes and weights for a fixed range. This saves construction work
  but adds a data-generation and provenance obligation for every supported order.

A is recommended. The benchmark's 4000-point discretisation is a starting point,
not an accuracy certificate or a microsecond timing promise. Check independent
moments and refinement of the auxiliary rule; reject non-finite coefficients,
non-positive weights or a failed certificate. Phase 1 records the supported order
range, refinement limit and measured construction cost. Do not silently cap `n_r`.
Snap the certified Radau endpoint to exactly $\eta=1$, $\xi=0$ before inversion;
never divide by $\xi$, and check moments again after endpoint handling.

The existing recurrence in `modes.quadrature` is an exception to the general
architecture rule that hand-written numerics live in T/B. Phase 1 must reconcile
this explicitly in the architecture documentation, limiting the exception to
construction of transverse rules; no compiled or per-stage physics enters P.

**The area-weight convention.** `TransverseQuadrature.weights` owns the conversion
in §4.2. `mode_matrix` and `multimode_transform` continue to consume physical
weights. Preserve existing polar rules' arithmetic and bitwise digests; the new
representation must not silently reinterpret old `w_r` values. Audit all rule
constructors and consumers, including on-axis synthesis, polar panels, general
modal tables, fluence widths, populations, absorbers and serialisation. Reject
inconsistent fine/coarse representations rather than converting them twice.

**The kernel contract.** `Transform` is kernel-contract §4 and is a field of
`KernelProblem`; it is not outside the seam merely because T constructs it.
`ProblemFlags.mode_averaged` also carries its meaning to consumers. Adding the
stored kind requires a **schema bump in Phase 1**, updated contract documentation,
the frozen compatibility test and a changelog entry. Any new statistic admission
or axis-row data needed by a backend is designed in that same schema change.
Installed older backends must refuse the new schema; no silent fallback.

**Output and resume.** The new quadrature identity and statistic names must survive
MemoryOutput/HDF5 round trips and resume checks. Phase 1 updates the output schema
when changing its structured records, separately from the kernel schema, and
specifies legacy reads: missing new metadata must never mean a field-scale rule.
Existing statistic datasets retain their meanings.

**On-axis statistics in each backend:**

- **numpy_ref.** `_ionisation` and `_plasma_nodes` gain a separate axis evaluation
  using the shared `StepDriver._on_axis_synthesis`. It produces only the new names;
  legacy node reductions and node-array extents are unchanged.
- **numba_cpu.** The same, through its statistics path. It has an on-axis synthesis for
  `peak_intensity` to reuse.
- **Kokkos CPU.**
  - The statistics reductions run on the device over the node set.
  - The recommended approach is to synthesise the axis row with the existing
    `peak_intensity` machinery and run the plasma scan on that one row. This needs a
    small device path: one row, the same kernels.
  - The alternative is to append the axis row to the node set with zero projection
    weight. That costs one extra row in every right-hand side, not one per saved plane,
    so it is not recommended.
  - Either way the Kokkos wheel is rebuilt. GPU qualification follows the usual later
    campaign.

**With the Radau rule.** If the requested node statistics have already evaluated
the axis row at this observation, the new statistics reuse it. Otherwise evaluating
that row is still necessary, but no additional radial node is introduced. All
statistics buffers are preallocated, named in the memory ledger and bounded
independently of the number of saved planes.

**The explicit kind.** Update all `Transform` constructors and all consumers of
`mode_averaged`, including `ProblemFlags`, response normalisation, population
weights, TDSE admission and output provenance. Kokkos's native packing may continue
to omit a coarse projection for one node, but must honour the explicit kind wherever
effective-area behaviour is selected.

**Cost.**
- The rule changes nothing per step but the node count.
- The on-axis statistics add at most one rate integration on one row per saved
  plane, measured separately from propagation. It is incurred only when a new statistic is
  requested; TDSE incurs no new atomic work.

**Dependencies.** None new.

### 7. Verification and validation

**Rule construction and exactness.** For Gauss $2\le n\le32$, compare moments
$0\le k\le2n-1$ against independent adaptive integration of
$\xi J_0(u_{01}\xi)^{k+1}$, not the discrete measure used to construct the rule.
For Radau use $4\le n\le32$ and $0\le k\le2n-2$. The relative target is
$10^{-12}$, with reference integration error and accumulated floating-point error
bounded below that target and recorded. Also check the analytic identities
$M_0=J_1(u_{01})/u_{01}$ and $M_1=J_1(u_{01})^2/2$. Extend qualification to every
larger order the implementation admits. Verify ordered distinct nodes, positive
finite weights, refinement of the auxiliary rule and deterministic results.
Radau stores exactly one axis node; all other nodes are strictly interior.
The special $n=1$ rule is checked for moments 1 and 3, not the Gaussian moment set.

**Source normalisation.** On identical prescribed scalar and two-component
waveforms, linear and cubic sources agree with an independent high-order radial
integral for Gauss $n=2,4,8$ and Radau $n=4,8$. Include field/envelope grids and a
scale-invariant taper. Use tolerances derived from the measured moment error,
transform reduction length and FFT rounding. A short Kerr-only run then checks
accumulated error on approximately ten accepted steps with fixed shared settings;
a preliminary $10^{-10}$ relative field target must be justified by those bounds.
This is nonlinear-source quadrature exactness, not a quadrature approximation to
the already diagonal linear propagation operator.

**One-node identity.** Check the synthesis, physical weight $A_\text{eff}$ and
combined projection/prefactor algebra of §4.3, followed by Kerr and plasma source
comparisons with the existing mode-averaged route. Repeat on a taper and both time
grids for mutually admitted responses. Propagation uses a short window and a
rounding budget from the arithmetic, not a tolerance chosen after observing a
failure. New axis statistics are not part of the mode-averaged identity.

**Compatibility and integrated observables.** Existing rule matrices and
mode-averaged source digests stay unchanged. Check `kind` validation on every
constructor, including general-modal placeholders and all spatial geometries;
check the one-node resolved route reports `modal` and does not receive a second
area conversion. Compare integrated plasma inventory, including uniform
preionisation and a saturated fraction, with an independent area integral. Report
the separate quadrature error; do not assert area exactness for the new rules or
renormalise their source weights. Cover polar panels and non-polar legacy rules.

**Rate-model convergence.** At fixed saved waveforms, compare library-rule source
errors with `benchmarks/radial/evidence/surrogate.json`, preserving the recorded
response, time-integration setting, source component and spectral mask. Compare
candidate and benchmark node/weight arrays separately to isolate rule generation
from response changes. Use an independent dense reference and its refinement to
set the comparison tolerance, especially near the floating-point floor. The
propagation check uses about ten accepted steps against a denser radial rule.
The historical 4.2 m propagation records are contextual evidence only: do not
rerun a full-length parent in a test, lane or benchmark. A new full-length
scientific demonstration would belong in a manually run worked example.

**On-axis statistics.** Compare each new statistic against an independent
single-point rate integration at the same prescribed modal field. Test scalar and
vector fields, field and envelope grids, tapers, mixtures, preionisation and each
admitted plasma-integration method. Verify species sums and zero entries for
non-ionising species. Changing radial node count must not change the axis result
on that fixed field beyond arithmetic rounding. A five-node HE₁₁ example should
show why node maxima and axis values differ; an off-axis multimode maximum must
remain visible in the unchanged legacy statistics. A non-monotone custom rate
also guards against replacing a node maximum with the innermost value.

Instrument evaluation counts: when node data are already requested, a Radau
axis row is reused once; a non-axis rule needs at most one extra rate-model row.
Test axis-only requests, combined requests, consecutive planes and rejected-step
histories so that stale row reuse cannot pass. Periodic requests for the new names
are refused. Node inventories retain their original shapes and weights.

**Backend parity and lifecycle.** NumPy, Numba and Kokkos CPU agree on source,
one step, eight steps and requested statistics for five-node Gauss and Radau,
as well as the one-node special case where admitted. Apply the existing
summation-derived parity tolerances; verify bounded memory and thread-count
repeatability for new reductions. Each backend refuses a requested feature until
implemented. TDSE refusal on Numba/Kokkos remains an explicit test.

**Admission and provenance.** Cover wrong modes (including HE₁₂ and unsupported
HE₁₁-labelled providers), two-mode sets, panel boundaries, non-polar geometries,
unknown names, invalid counts, and incompatible `kind` fields. Test the unchanged
32→33 Kronrod rounding and defaults. Round-trip rule identity and all new statistic
names through outputs; reject mismatched resume quadratures and preserve the
interpretation of older output files.

**TDSE compatibility without scientific runs.** Q5 exempts a new TDSE solve or
propagation, not interface testing. Existing deterministic provider doubles check
one-node and multi-node synthesis, P/J projection, endpoint row identity, chunking
and metadata. All plasma statistic names remain refused before provider evaluation;
assert zero additional atomic calls. No new physical TDSE qualification is claimed.
Check that response admission uses the explicit kind and that `n_r=1` reaches the ordinary resolved P/J path.

**Evidence to retain.** Phase 1 records the independently measured moments and
construction timings and makes the saved-row Radau analysis reproducible. The
benchmark ledger records only the measured short window and the statistics cost;
no full-run timing is extrapolated.

### 8. Documentation

**A new guide page,** `docs/guide/waveguides/radial-quadrature.md` ("Radial nodes for expensive
responses"), linked from [Multimode capillaries](../../../guide/waveguides/capillaries.md),
[Waveguide modes](../../../guide/waveguides/modes.md) and
[TDSE response in a helium capillary](../../../guide/media/tdse.md). It covers:

- **Why the radial integral of a single-mode capillary is an integral over the field
  scale:** one atom per intensity.
- **The effective-area comparison:** the mode-averaged model as the one-node rule, and
  what it gets wrong for ionisation, including the special one-node admission.
- **The full rule comparison:**
  - the rate model;
  - the dense TDSE scan, and its structure in η²;
  - the measured surrogate propagation effect and the limits of that evidence.

  The figures are copied from `benchmarks/radial/`.
- **The expensive right-hand-side use case,** with starting choices for the scanned
  SAE case, not universal accuracy guarantees:

  | Goal | Setting |
  |---|---|
  | Default | 33 Gauss–Kronrod nodes |
  | Sub-percent, with no special rule | `"gauss"` with `n_r=20` |
  | Sub-percent | 16 field-scale nodes |
  | Percent level on the scanned SAE case, for exploration | 5–8 field-scale nodes |
  | As above, with the on-axis atom's own output | 5–8 Radau field-scale nodes |
  | Effective-area comparison only | `"field_scale"`, `n_r=1` |
  | Linear/Kerr exactness tests | `"field_scale"`, `n_r=2` or 3 |
  | Not recommended for quantitative ionisation | fewer than 4 nodes |

- **The cost on GPU and CPU:** distinguish the measured under-filled GPU batches
  and CPU cases; do not promise their scaling for another provider or workload.
- **All-electron responses:** the rule construction still applies, but the SAE
  node counts and accuracy claims do not qualify another atomic model.
- **When not to use it:** several modes, or when the quadrature alarm is wanted.

**Other documentation:**
- The statistics catalogue in [Statistics](../../../guide/results/statistics-catalogue.md) distinguishes
  the four opt-in on-axis names from the unchanged node reductions, states the
  initial route scope, and explains that TDSE never launches an atom for statistics.
- Docstrings are updated for `transverse_quadrature`, `from_capillary`, the statistic
  names and `Transform.kind`.
- Update kernel-contract, architecture, conventions and output/resume documentation,
  `README.md`, `docs/introduction/first-run.md` and `CHANGELOG.md` where the interface changes appear.
  Audit executable examples affected by the removed keyword and regenerate any
  affected gallery outputs under the established gallery rules.
- No gallery example is planned; the guide page carries the figures.

### 9. Compatibility and impact

- **Defaults:** the radial rule, node count, default statistic set and route remain
  unchanged. `n_r=32` still means 33 Kronrod nodes.
- **Breaking change:** the affected public `kronrod=` selection becomes
  `radial_rule=` (Q1). Existing recorded configurations must retain their legacy
  interpretation; this decision does not authorise rewriting old output files.
- **Statistics:** all existing definitions and reference values remain unchanged.
  The four separately named on-axis statistics are opt-in. No `axis=` selector is
  introduced. TDSE supports neither old nor new plasma statistics in this SEP.
- **Schemas:** adding `Transform.kind` requires a kernel schema bump in Phase 1.
  Output metadata/versioning is handled separately, including legacy reads and
  resume identity for rule family, realised nodes and weights.
- **Existing validated results:** propagation and population weights for every
  existing rule remain unchanged; mode-averaged runs keep their bitwise digests.
- **Luna.jl:** the implementation updates [the differences page](../../../guide/workflows/luna.md) for
  the new fixed field-scale rules and optional axial diagnostics.

### 10. Alternatives considered

- **Fewer Gauss nodes in ξ** (`kronrod=False`, small `n_r`). This is available today
  with no new code. On the dense TDSE scan, 20 nodes match the default. But below about
  15 nodes ξ-rules do no better than field-scale ones. On the rate model they converge
  more slowly at equal node count: 7×10⁻⁵ at 8 nodes, against 4×10⁻⁷ in η. It stays as
  guidance in the manual, not as the proposal.
- **An axis node with zero weight,** added to any rule for the statistics. It costs one
  extra atom in every call; on the TDSE route that is 20% more rows at five nodes.
  Rejected. Rate-model statistics may evaluate one extra row per saved plane;
  TDSE statistics use only already evaluated quadrature rows (§5.3).
- **Smoothing or interpolating across the nodes in η,** to beat the oscillation in the
  TDSE source. The oscillation is real, not noise, so smoothing would bias the source.
  Rejected.
- **Analytic integration of the perturbative part,** with nodes only for the remainder.
  This needs the response's own linear and cubic coefficients. The field-scale rule
  already integrates those terms exactly from two nodes, so it does not improve
  their linear/cubic exactness. It could still change the approximation of the remainder; that is a different rule-design problem,
  deferred rather than ruled out by polynomial exactness.

### 11. Phases

1. **Translation and numpy_ref.**
   - The explicit transform kind, with the one-node identity test.
   - The area-weight convention for polar rules.
   - `radial_rule`, with both field-scale rules, their guards and their metadata.
   - Four separately named on-axis statistics in numpy_ref; all old names unchanged.
   - Mandatory kernel schema bump, output/resume identity and refusal gates.
   - Fixed-waveform rate-model convergence and a short propagation comparison
     using the library rules.
   - The guide page.
   - **Verifies:** §7 on numpy_ref, including provider-double tests, except
     cross-backend parity. Other backends explicitly refuse unimplemented features.
2. **numba_cpu and Kokkos CPU.**
   - On-axis statistics in both backends.
   - Parity on the field-scale run.
   - The Kokkos wheel rebuilt against Phase 1's new kernel schema.
   - TDSE admission remains NumPy-only; this phase qualifies rate-model responses.
   - **Verifies:** backend parity, and the statistics tests on both backends.

Each phase follows the current phased-SEP checks in `AGENTS.md`. Before Final, run
the full gate, slow and coverage lanes and the Kokkos-selected CPU slow lane on
the integrated tree. GPU qualification remains the separate later campaign. Part II
records failures, sources, measured error/cost and any outstanding limitations.

### 12. Open questions

- **Q1 (resolved 2026-10-12).** `radial_rule=` replaces `kronrod=`.
- **Q2 (resolved 2026-10-12).** `n_r = 1` is admitted for `"field_scale"`: it is the
  effective-area model on the quadrature path, including on the TDSE route. The guide
  page says what it gets wrong.
- **Q3 (revised by John Travers during review, 2026-10-12).** Add separately
  named on-axis statistics and retain all existing node reductions. The earlier
  `axis="innermost"` selector is superseded. TDSE must never launch an extra atom
  for statistics; without a radial axis node, any future ionisation statistic
  reports only a node maximum (§5.3).
- **Q4 (resolved 2026-10-12).** The Gauss–Radau variant is offered as
  `"field_scale_radau"`, with `n_r ≥ 4`.
- **Q5 (resolved 2026-10-12).** No TDSE run is required before Final; it is left to
  later scientific use. Deterministic provider-double compatibility tests remain
  required; this exemption does not expand the existing TDSE backend/route scope.
- **Q6 (Phase 1).** Certify the numerical construction and admitted upper order;
  record auxiliary-grid refinement, moment residuals and cost. The implementer
  also supplies the reproducible saved-row Radau evidence and integrated-population
  convergence checks. These are qualification deliverables, not universal node-count
  guarantees.
- **Q7 (Phase 1).** Reconcile the existing quadrature module's numerical-construction
  exception with the architecture documentation, and inventory the public keyword
  migration and legacy output handling. This local question number is distinct from
  the TDSE roadmap's radial-optimisation Q7.
- **Q8 (Phase 1).** Whether to warn, once per run, when a node-maximum plasma
  statistic (`electron_density`, `peak_ionisation_rate` or a species form) is
  requested with a Gauss field-scale rule whose innermost node is well below the axis.
  The warning would name the node's η and point to the `on_axis_*` names. It would
  change no value and cost nothing. Its purpose is that a small-node run does not
  report an axial peak about 2× low with nothing but the guide to explain it. The
  implementer proposes the threshold, or reports why the guide alone suffices.

### 13. References

1. E. A. J. Marcatili and R. A. Schmeltzer, "Hollow metallic and dielectric waveguides
   for long distance optical transmission and lasers," *Bell Syst. Tech. J.* **43**,
   1783 (1964). doi:10.1002/j.1538-7305.1964.tb04108.x
2. W. Gautschi, *Orthogonal Polynomials: Computation and Approximation* (Oxford
   University Press, 2004). doi:10.1093/oso/9780198506720.001.0001
3. G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules," *Math. Comp.*
   **23**, 221 (1969). doi:10.1090/S0025-5718-69-99647-1
4. G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019), for the
   effective area.
5. G. H. Golub, "Some modified matrix eigenvalue problems," *SIAM Rev.* **15**, 318
   (1973). doi:10.1137/1015032

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

### Phase 1 report — translation and numpy_ref

- **Scope:**
  - Phase 1, 2026-10-12, implemented by Opus 5.5 for John Travers.
  - Branch `feature/sep-0019-field-scale-1` in solprop and in solprop-kokkos, developed
    in separate worktrees beside the concurrent SEP 17 work.
  - Merge commits are recorded at merge.
- **Delivered:**
  - **Rules** (`src/solprop/modes/quadrature.py`):
    - `field_scale_rule`, with Gauss and Gauss–Radau variants and the one-node
      effective-area special rule;
    - `check_field_scale_modes`;
    - `TransverseQuadrature` gains `radial_rule`, `area_r`/`area_r_coarse`,
      `requested_n_r` and `record()`, and checks that exactly one weight
      representation is present;
    - `quadrature_rule` and `transverse_quadrature` take `radial_rule=`.
  - **Transform kind and schemas:**
    - `Transform.kind` and the kernel schema 26 bump: `translate/modal.py`, `radial.py`,
      `planar.py`, `free.py`, `problem.py` and `docs/dev/kernel-contract.md`;
    - solprop-kokkos `bindings.cpp`, its tests and its changelog.
  - **Keyword migration:** `radial_rule=` on `Simulation.from_capillary`,
    `from_waveguide`, `prop_capillary` and `prop_waveguide`, and `PlanarGrid(x_rule=)`
    (`driver/simulation.py`, `driver/interface.py`, `modes/slab.py`).
  - **Metadata:** `RunMetadata.quadrature` at output schema 17
    (`output/schema.py`, `metadata_for`).
  - **On-axis statistics:**
    - the four `on_axis_*` statistics (`stats.py`, `backends/populations.py`, numpy_ref
      `_axis_statistics`, `_axis_row` and `_axis_node_index`);
    - refusals in `driver/run.py`, numba_cpu, Kokkos and
      `TDSE_UNAVAILABLE_STATISTICS`;
    - `SampledPeakWarning` and `_warn_sampled_peak`.
  - **Tests:** `tests/modes/test_field_scale.py` (51) and
    `tests/backends/test_on_axis_statistics.py` (17), plus the schema and keyword
    migrations across the suite.
  - **Documentation:**
    - the new guide page `docs/guide/waveguides/radial-quadrature.md`, with its two figures;
    - rows and a note in `docs/guide/results/statistics-catalogue.md`;
    - `docs/guide/waveguides/modes.md`, `docs/guide/results/resuming.md` and
      `docs/guide/media/tdse-capillary.md`;
    - `docs/guide/waveguides/capillaries.md` and its index;
    - `docs/dev/architecture.md`, `docs/guide/workflows/luna.md` and `CHANGELOG.md`.
  - **Evidence:** `benchmarks/radial/` scores the library rules on the saved dense
    scan and on the rate-model surrogate (`evidence/dense.json` and
    `evidence/surrogate.json`).
- **Deviations from the proposal:**
  - **The recurrence uses the Lanczos process, not Stieltjes.** The discretised Stieltjes
    procedure stopped converging from order 44, its coefficients losing accuracy at high
    degree. Lanczos with full reorthogonalisation on the same discrete measure is stable
    to order 128 at least.
  - **The slab keyword is `x_rule`.** The plan named `PlanarGrid`'s replacement for
    `kronrod=` `rule=`, but that name is already the grid's built rule.
- **Decisions:**
  - **Q6, construction:**
    - The discrete measure is a Gauss–Legendre rule in ξ, refined from 128 points by
      doubling up to 8192. The coefficients settle at 256 points for every order up to
      64, and at 512 up to 128. Successive refinements then differ at a rounding floor of
      1e-14 to 2e-14, so the recurrence tolerance is 1e-13.
    - Every rule is certified on construction against a measure refined once more, to
      1e-12 in each moment it must integrate. Measured against independent adaptive
      integrals, the residuals are about 3e-14, the largest 1.1e-13, over Gauss and
      Radau orders 2 to 64.
    - Orders 1 to 128 are certified. A larger order is built and certified the same way,
      or refused; it is never capped silently.
    - A rule takes about 6 ms to build and is cached by family and order.
  - **Q7, architecture:**
    - `docs/dev/architecture.md` records the numerical construction of transverse rules
      as the one numerical construction kept in the physics layer. It runs once per
      problem and enters no hot loop.
    - The keyword migration covers `quadrature_rule`, `transverse_quadrature`, the
      capillary and waveguide constructors and interfaces, the general-route keyword
      table and `PlanarGrid`.
    - `tests/data`'s Luna generators keep Luna's own `kronrod` name.
    - The internal `TransverseQuadrature.kronrod` still means "has an embedded rule".
  - **Q8, the warning:** implemented. `SampledPeakWarning`, once per run, when a
    node-maximum plasma statistic is requested with a Gauss field-scale rule whose
    innermost node is below η = 0.999. Five nodes, at η = 0.958, warn. The Radau rule and
    the default rule do not.
  - **Axis field:** the axis field uses both Cartesian components of the mode profiles,
    as `peak_intensity` does. When the state carries the components itself, the
    profile's magnitude is the scalar synthesis.
  - **Radau reuse:** a Radau axis node is found by matching its synthesis row to the
    on-axis synthesis. It supplies the axis row only from the same call's node scan.
    When on-axis names are requested, the species statistics share that scan; their
    values are unchanged, because they were already that scan.
- **Verification:**
  - **Rules:**
    - moments against independent adaptive integrals;
    - the closed forms $M_0$ and $M_1$;
    - the one-node rule's moments 1 and 3, with $\eta_e = 0.75202157$;
    - the Radau axis node exactly at η = 1, ξ = 0;
    - positivity, ordering, determinism and every guard.
  - **Source and runs:**
    - Linear and Kerr source exactness against a 64-node Gauss rule, scalar and rotated
      two-component, holds within the moment tolerance.
    - A Kerr run of ten fixed steps agrees with the default rule to 1.3e-16.
    - The one-node rule against the mode-averaged route, over ten fixed steps at
      100 µJ in argon, agrees to 1.7e-16 with Kerr only and 4.5e-16 with Kerr and
      plasma. There the plasma moves the field by 1.4e-4, and the mode-averaged answer
      differs from the resolved 33-node one by 7e-3.
  - **Metadata:** round-trips through JSON, a changed rule changes the resume key, and a
    record older than 17 carries no quadrature. The pinned hashes for output schemas 8
    to 11 still verify unchanged.
  - **On-axis statistics:**
    - the axis value is bitwise the same on 33 Gauss–Kronrod, 5 and 8 Gauss and
      5 Radau field-scale nodes;
    - the default rule's node maximum is 2.1e-5 from it, and five Gauss nodes' is
      0.44 of it;
    - a Radau rule reuses its axis row: zero extra scans when the node statistics ran,
      one otherwise;
    - consecutive planes scan their own fields;
    - an HE₁₁ − HE₁₂ field cancelled on the axis keeps its off-axis maximum in
      `electron_density`;
    - an envelope Ar/He fill in a tapered bore with 10% preionisation reads the same
      axis on every rule, with helium at zero;
    - every refusal is tested, including numba_cpu and Kokkos at construction.
  - **Evidence:**
    - On the saved dense SAE scan, the library rules reproduce the prototype's scores
      exactly.
    - On the rate-model surrogate, their source errors agree with the prototypes' to
      5e-4 relative on all four waveforms; §7 asks for 10%.
  - **Suites run, and what is still to run:**
    - The backend, driver, statistics and translation suites, 6602 tests, pass with
      the schema-26 Kokkos wheel installed.
    - The end-of-phase gate and coverage are recorded below when run.
    - The full slow, coverage and Kokkos slow lanes are left to the end of the SEP, per
      AGENTS.md.
    - No TDSE solve or propagation was run (Q5), and no full-length propagation in any
      test.
- **Sources checked:**
  - The Golub–Welsch construction and the Golub endpoint modification were checked at
    review against [3] and [5], §7.
  - The Lanczos route is Gautschi's [2], §2.2.3.2, which was not reread in full here.
- **Limitations:**
  - Phase 2 must implement the on-axis statistics in numba_cpu and Kokkos; both refuse
    them by name until then.
  - The field-scale node counts are qualified on one SAE waveform only (§4.5).
  - The area weights integrate the source, not the area, so integrated populations
    carry a quadrature error; this is documented, not certified.
  - No GPU qualification.
- **Open questions:** none new. Q6–Q8 are resolved above.
- **Handoff to phase 2:**
  - **numba_cpu:** replace the by-name refusal in `_refuse_unsupported` and `stats`
    with an axis row through its own `_on_axis_synthesis` and plasma scan, reusing
    `populations.axis_statistics`.
  - **Kokkos:** the same, through the existing `peak_intensity` axis machinery and one
    row of the plasma scan, replacing `_refuse_axis_statistics`.
  - **Parity:** numpy_ref is the oracle, and `tests/backends/test_on_axis_statistics.py`
    is the template.

### Phase 2 report — numba_cpu and Kokkos CPU

- **Scope:**
  - Phase 2, 2026-10-12, implemented by Opus 5.5 for John Travers.
  - Branch `feature/sep-0019-field-scale-2` in solprop and in solprop-kokkos.
  - Merge commits are recorded at merge.
- **Delivered:**
  - **Shared routines:**
    - `backends/populations.py` gains `axis_populations`, the one host scan of the
      axis row, and `axis_node`, the Radau axis-node detection.
    - `StepDriver._axis_statistics` (`backends/driver.py`) uses both for every backend,
      with the `axis_scans` counter.
    - numpy_ref's phase 1 code moved there unchanged in behaviour.
  - **numba_cpu:**
    - dispatches the four names;
    - records its own node scan (stash, compiled column or envelope populations) for
      the Radau axis node;
    - applies the route refusal, and drops its by-name refusal.
  - **Kokkos:**
    - reads the host state (`field_to_host`) and scans the axis row through the shared
      routine;
    - takes the device population scan of the same call for a Radau axis node;
    - drops its construction and `stats` refusals.
  - **Tests:** `tests/backends/test_on_axis_statistics.py` replaces the refusal tests
    with parity tests on both compiled backends, for a fresh axis scan and for Radau
    reuse.
  - **Documentation:** the statistics page, both changelogs and the roadmap.
- **Deviations from the proposal:**
  - **The axis row is scanned on the host on every backend,** rather than by a
    one-row device path in Kokkos as §6 recommended. The axis is one row per saved
    plane against every node of every stage. On a CPU the host scan and Kokkos's
    state copy cost nothing measurable beside a step, and one shared routine makes
    the backends agree by construction.
  - **Deferred:** a device path for the axis row, needed only to avoid the state copy
    on a GPU, is left to GPU qualification.
  - **Unchanged by this choice:** the Radau reuse still reads each backend's own
    compiled node scan.
- **Decisions:** none beyond the deviation above.
- **Verification:**
  - **Parity with numpy_ref:** numba_cpu and Kokkos CPU agree with numpy_ref on all
    four statistics to $10^{-10}$ relative, on five Gauss and five Radau
    field-scale nodes.
  - **Scan counts:** zero extra scans when a Radau rule's node statistics ran, one
    otherwise.
  - **Existing suites:** the statistics, numba and Kokkos backend modules pass,
    except a Kokkos test literal that still named schema 25 from phase 1, now
    corrected. The full test lane was not rerun for this phase.
  - **End-of-SEP lanes:** `poe check`, `poe test-slow`, `poe cov` and the Kokkos
    slow lane ran once on a Linux cluster node. No SEP 19 test failed. Twelve
    unrelated tests failed on that platform: FFTW wisdom and byte witnesses pinned to
    the macOS FFTW build, tolerances a few units in the last place too tight for Linux,
    and wall-clock budgets on a shared node. They were adjusted for Linux before Final.
- **Sources checked:** none new.
- **Limitations:**
  - No GPU qualification.
  - On a GPU the axis row costs one state copy per saved plane that requests it.
- **Open questions:** none.
- **Handoff:** none; the device path for the axis row waits for the GPU campaign.

## Part III — History

- 2026-10-12: Drafted, from the radial-node analysis in `benchmarks/radial/`.
- 2026-10-12: Q1, Q2, Q3 and Q5 resolved by John Travers; Phase 3 (TDSE check)
  removed.
- 2026-10-12: Q4 resolved by John Travers. The Gauss–Radau rule was added, with the
  area-weight convention it needs, free on-axis statistics, and the rule for a future
  TDSE ionisation statistic.
- 2026-10-12: AI review by GPT-6; John Travers confirmed his human review.
  Corrected Kronrod count semantics, the one-node Gaussian claim, the Jacobian
  orientation and source-versus-area exactness. Required the kernel schema bump,
  route guards, output/resume identity, short-window qualification and provider
  doubles. John chose separate on-axis statistics with unchanged legacy reductions
  and no additional TDSE atom under any statistics setting. Review status: Reviewed;
  lifecycle status remains Draft pending maintainer acceptance.
- 2026-10-12: Accepted by John Travers. The drafting agent agreed with the review
  revisions and added Q8, on a warning for node-maximum plasma statistics under small
  field-scale rules, as a Phase 1 question. Part I is frozen.
- 2026-10-12: Phase 1 implemented on `feature/sep-0019-field-scale-1`; report in Part II.
- 2026-10-12: Phase 1 merged; phase 2 implemented on `feature/sep-0019-field-scale-2`.
- 2026-10-12: Phase 2 merged; the end-of-SEP lanes ran.
- 2026-10-13: Final, approved by John Travers; archived.

**Review evidence (2026-10-12).** Read the full proposal against SEP 0, the
architecture/conventions and the quadrature, transform, population, statistics,
TDSE-admission and Kokkos-packing code. Re-derived the measure, endpoint limits
and effective-area identity. Independent adaptive Bessel integrals gave
$\eta_G=0.62422958$ and $\eta_e=0.75202157$; the five-node area ratio was
0.95943948 (Gauss) and 0.95173159 (Radau). Checked the draft's Radau table by
re-evaluating its rules on the saved 157-row spline; no atom or propagation was
run. Against independent adaptive integrals, the candidate Gauss and Radau
rules for orders 2–32 had a largest tested relative moment residual of
$5.55\times10^{-13}$ (including the mathematically defined, publicly refused
small Radau orders). These are checks of the benchmark prototypes, not a completed
library implementation. The records support this one waveform, not an all-electron
accuracy claim. Rate-model Radau source scores were not independently rerun.

Checked the Jacobi-eigenvector construction against Golub–Welsch [3] and the
endpoint modification against Golub [5], §7. Verified Gautschi's publication and
DOI with the publisher, but did not read the complete book. Marcatili's original
field derivation and Agrawal's effective-area discussion were not rechecked in
full text; the profile and normalisation here were checked against solprop's
implementation and conventions and the area identity was independently derived.
