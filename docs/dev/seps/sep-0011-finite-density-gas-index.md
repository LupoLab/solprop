# SEP 11 — Finite-density gas index and mixture optics

| | |
|---|---|
| **SEP** | 11 |
| **Title** | Finite-density gas index and mixture optics |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5; GPT-6 (compatibility revision, 2026-10-22) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-09-29 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md): optmat polarizabilities and material identities |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Replace solprop's linear gas susceptibility $\chi=\sum_sN_s\alpha_s/\varepsilon_0$
with optmat's constitutive law: Lorentz–Lorenz with refractivity virial terms,
applied once to the whole mixture. The selected upstream pair terms retain their
curated, estimated or unknown status. Against the finite-density model, the dilute-law
difference is about 0.02–0.05 % of $n-1$ at 1 bar, but 0.2–0.6 % at
10 bar and more above, comparable to the data improvements
[SEP 3](archive/sep-0003-optmat-materials.md) brings.

Because the new law is nonlinear in density, it changes solprop's frequency-only
per-particle caches, the fast paths that multiply them by density, mode freezing,
parameter expansions, frame derivatives and power weights. The same composition-preserving
mixture equation of state supplies the densities used by linear optics and nonlinear
responses. It does not itself change the
kernel contract, since linear operators and densities are already tabulated; the
separate Raman execution extensions belong to SEPs 1 and 10.

### 2. Motivation and scope

#### Status

**Draft; implementation not started.** SEP 3 is Final. The 2026-10-22 compatibility
revision checks solprop `20a2bafa` and optmat 0.8.0, especially the completed upstream
mixture optics/EOS and their current model selections. This is not a completed
standalone acceptance review; the questions in §12 remain visible.

#### Motivation

The relative difference of the dilute law in $n-1$ from optmat's `refractivity` at
800 nm and 294 K, was calculated with the then-audited optmat. SF6 condenses near 21 bar at
this temperature. The table is the original draft's model comparison, not a new
measurement or a determination of the total experimental error. It was not rerun
in the compatibility revision.

| Gas | 1 bar | 10 bar | 30 bar | 60 bar |
|---|---:|---:|---:|---:|
| He | −0.002 % | −0.016 % | −0.047 % | −0.093 % |
| H2 | −0.009 % | −0.086 % | −0.26 % | −0.50 % |
| Ar | −0.019 % | −0.19 % | −0.57 % | −1.16 % |
| N2 | −0.019 % | −0.19 % | −0.57 % | −1.13 % |
| Xe | −0.052 % | −0.55 % | −1.9 % | −6.5 % |
| SF6 | −0.055 % | −0.62 % | — | — |

Zero-dispersion and phase-matching pressures in gas-filled fibres are routinely set at
several to tens of bar. The finite-density differences motivate checking zero-dispersion and phase-matching
shifts within each material's data domain; the comparison does not alone establish
which approximation matches experiment more closely.

#### Included

- The finite-density law for pure gases and mixtures, with its derivatives.
- Unlike-pair terms and their uncertainty/status as selected by optmat, with a
  documented domain; estimates are not described as measurements.
- A composition-preserving mixture EOS for a shared pressure/temperature fill,
  with its method, domain and status recorded.
- Consistent labelled-population semantics.
- All consumers of the linear law, and their fast paths.
- Named air compositions under the new law.

#### Not included

- Density corrections to the nonlinear response: local-field factors and pair terms
  of $\chi^{(3)}$ are not in optmat's nonlinear model.
- New microscopic density corrections to Raman or ionisation. Existing density
  dependence of amplitudes, collision widths and line shifts consumes the same
  resolved state as linear optics.
- New mixture data or EOS fits: optmat owns those. Independently prescribed population
  densities remain inputs, rather than being reinterpreted as a common-pressure fill.

### 3. Context

SEP 3 §4.2 keeps $\chi=N\alpha/\varepsilon_0$ with a measured error and warning.
`GasFillMixin._sampled_susceptibility` caches frequency-only per-particle rows, and
`Mixture._optical_sum` multiplies them by density. Modes, the general modal
providers, the free-space routes and the parameter expansions assume that linearity.
The [mixtures](../../guide/media/mixtures.md) and [materials](../../guide/media/materials.md)
manuals describe the present law.

### 4. Theory

For a pure gas, optmat's law [1, 2] in terms of the SI per-particle refractive
polarizability $\alpha$ and the molar density $\rho=N/N_A$ is

$$
y=\frac{N\alpha(\omega)}{3\varepsilon_0}(1+b\rho),\qquad
b=\frac{B_R}{A_R},\qquad
\chi=\frac{3y}{1-y},\qquad
n=\sqrt{1+\chi}.
$$

Form $n-1=\chi/(\sqrt{1+\chi}+1)$ to keep dilute precision. Differentiating,

$$
\partial_\omega\chi=\frac{3\partial_\omega y}{(1-y)^2},\qquad
\partial_\omega n=\frac{\partial_\omega\chi}{2n}.
$$

For a frequency-independent ratio,
$\partial_\omega y=N(1+b\rho)\partial_\omega\alpha/(3\varepsilon_0)$; a
frequency-dependent virial needs its own derivative. The dilute limit is
$\chi=N\alpha/\varepsilon_0+O(N^2)$.

For a mixture, the general expansion is

$$
y=\sum_s A_s\rho_s+\sum_{s,t}B_{st}\rho_s\rho_t+\cdots,
\qquad A_s=\frac{N_A\alpha_s}{3\varepsilon_0}.
$$

Summing pure-species indices or susceptibilities after applying Lorentz–Lorenz to
each is incorrect. Sum the polarizability and virial contributions first and invert
once. The unlike-pair coefficients $B_{st}$ are independent data and cannot be
deduced from the pure-gas ratios [3].

**The proposed upstream policy:** retain optmat's self-pair records and the selected
unlike-pair model, then invert once. At optmat 0.8.0 no optical unlike-pair measurement
is curated for the admitted pairs. The estimate is
$B_{st}=(B_{ss}+B_{tt})/2$, with each like term's own dispersion. Its uncertainty
scale is $\max(|B_{ss}|,|B_{tt}|)$, with an explicit scale when a like term is unknown;
opposite signs must not cancel that uncertainty. This is a sensitivity assumption,
not a statistical uncertainty or a measured cross coefficient. Optmat's current
[mixture manual][mixture-manual] is the source of these policies.

Unknown self terms and optmat's provisional O2 term keep their status and range.
Use upstream uncertainty/domain evaluation, rather than duplicating constants or
silently replacing unavailable data by zero. A zero-pair sensitivity comparison
is a separately labelled model comparison. The pure-species limit and measured
uncertainty effects remain required checks.

**Labelled populations.** Repeated labelled populations of one chemical species are
combined for the mixing rule before the self-pair term is evaluated. Their labels stay
separate for plasma histories and output; otherwise splitting one population into two
would change its refractivity.

**Absorbing gases.** The law is complex-valued with $\operatorname{Im}\alpha\ge0$. The
physical branch of the square root has nonnegative absorption. Species-resolved
absorption statistics are then an approximation, since the index is not additive.

**Magnetic contribution.** optmat's refractive polarizability already includes it;
do not add it again.

**One thermodynamic state.** For declared mole fractions $x_s$, a shared fill uses
the mixture EOS for $N_{\rm total}(p,T,\mathbf x)$, then
$N_s=x_sN_{\rm total}$. Repeated labels of a chemical species combine for EOS,
optics and collision laws, and are distributed back by their declared shares for
labelled histories/output. Splitting or reordering labels cannot change the state.
An EOS inverse must use the same composition and method as its forward map.

By contrast, evaluating each pure EOS at $x_sp$ generally does not preserve $x_s$.
Optmat's `partial-density` method preserves composition and is not an alias for
that legacy partial-pressure rule. Explicitly supplied density paths remain
authoritative; no second EOS is applied to them.

### 5. Proposed interface

The law is a material selection:

```text
selection = MaterialSelection(density_law="lorentz-lorenz",
                              mixture_eos="recommended")
# Explicit legacy comparison: density_law="dilute",
# mixture_eos="legacy-partial-pressure".
```

After qualification `lorentz-lorenz` is the default for built-in gases. `dilute`
remains an explicit, recorded choice, and it is the only meaning of a supplied
per-particle susceptibility. These are proposed settings, not current API.

The mixture-EOS setting resolves through optmat's recommended selection or an
explicit supported `multi-fluid`, `virial` or `partial-density` method. Record the
resolved method and its status. `legacy-partial-pressure` preserves the old shared-fill
construction for explicitly requested comparisons; it is not claimed to preserve
the declared mole fractions. It is distinct from opting for dilute optical response
at a composition-preserving density. Independently supplied density/partial-pressure
populations retain their declared semantics and are not silently converted to a
shared fill. Supplied materials without the upstream finite-density/EOS law use
their explicit dilute/supplied path or refuse a requested unavailable law by name.

### 6. Implementation strategies

**Where density enters.** Every consumer that multiplies a frequency-only row by
density is revisited:

- mode tables and frozen modes;
- the general parameter expansion;
- frame derivatives and power weights.

A law nonlinear in density can still be tabulated at the required states, or
approximated by the general parameter expansion, with its error checked against
direct optmat evaluation. It cannot enter an algebraically linear fast path without
changing that path.

**Caches.** Per-particle frequency samples remain cacheable. The finite-density law
is evaluated at each state, not cached as one density-scaled row.

**Cost on a density gradient.** The nonlinear law can use the existing parameter
expansion; no general assertion of unchanged translation or stage cost is made. The
parameter expansion ([algorithms](../algorithms.md), *The parameter expansion*) samples
the linear operator at Chebyshev nodes in density and integrates scalar tables along
z; it needs the operator to be analytic in density, not linear. $y$ is a quadratic in
the densities, and $\chi = 3y/(1-y)$ is singular only at $y = 1$, twenty to two hundred
times beyond the densities of interest. The degree needed to represent
$\sqrt{1+\chi}-1$ to $10^{-13}$ from zero to the full density, at 800 nm and 294 K with
optmat at `ea0dcaf`:

| Gas and largest pressure | $y$ at that density | Dilute law | Lorentz–Lorenz with virial |
|---|---:|---:|---:|
| Ar, 30 bar | 0.005 | 5 | 5 |
| N2, 100 bar | 0.018 | 7 | 6 |
| Xe, 60 bar | 0.055 | 8 | 7 |

These old degree measurements motivate reuse of the expansion; they do not prove
equal cost for every wavelength, state path or mixture EOS. Fixed composition and
temperature with one total-density path can retain one scalar density parameter.
Varying temperature/composition or independently prescribed species paths use the
appropriate multi-parameter/host-table representation. Measure translation and
short-stage costs after updating all algebraically linear fast paths.

**The density of mixtures.** The legacy partial-pressure construction omits unlike
thermodynamic interactions and can alter the declared composition. Its difference
from a multi-fluid calculation can be comparable to the finite-density optical
correction, but one model's difference from another is not experimental error.
Upstream OEP 4 compared the alternatives against measured density and virial data;
its completed results and later selections replace this draft's earlier assumption
that multi-fluid was uniformly qualified.

The current optmat 0.8.0 selections, checked through `mixture_equation_of_state`, are:

| Composition | Selected method | Status |
|---|---|---|
| He/Ne, equal fractions | `virial` | validated |
| He/Ar, equal fractions | `virial` | provisional |
| Ne/Ar, equal fractions | `virial` | provisional |
| Named dry air (N2, O2, Ar) | `multi-fluid` | provisional |

These are evidence about the current upstream selection, not solprop constants or
a promise of validity over all pressures and temperatures. Optmat checks each
method's domain; the virial path may hold only where cross coefficients are supplied
in temperature, and a provisional model warns. Preserve that status/range policy and
refuse unavailable methods; do not switch EOS silently at a profile point. The
selected method is part of the run's identity. OEP 4's archived intermediate report
and today's air default differ; the versioned public API/manual, not an old report's
default table, determines the selection.

**Shared state and Raman.** Resolve the density path once in the physics/translation
layers and use it for the linear operator, power weights, electronic amplitudes,
Raman amplitudes, linewidths and line shifts, and neutral/plasma inputs as currently
defined. SEPs [1](sep-0001-raman-response.md) and
[10](sep-0010-low-rank-raman.md) consume that path; their execution representations
do not know which EOS produced it. Optmat mixture Raman strengths contain mole
fractions and use the total composition density, while supplied component amplitudes
can differ from their collision density. Preserve these ownership rules.

Changing EOS changes the physical density interval and therefore invalidates all
dependent response tables and caches, including compressed spectral bases and
shifted poles. Check range/admission at the resolved states, including vacuum where
the path reaches it. A composition-preserving EOS does not supply foreign-collider
linewidth data, local thermal Raman populations, line mixing or nonlinear local-field
factors. Keep those existing restrictions visible. For a varying temperature path,
the linear state uses that temperature while any declared reference-temperature
Raman approximation remains explicit and recorded.

Audit gas-flow construction, choking/regime checks, parameter scans and reconstructed
fills alongside the optical consumers. They must not compute a second inconsistent
density or infer fixed fractions from the legacy rule. The detailed flow admission
audit is Q5 before this SEP's acceptance; it does not block SEPs 1 and 10, which
accept the already resolved density paths.

**Kernel.** Linear operators and densities still arrive as tables. This proposal
does not require a new numerical backend model. Any discovered seam change follows
the normal schema/version policy and must be recorded rather than assumed absent.

**Upstream.** `optmat.gas.mixture.mixture_optics` and
`optmat.gas.mixture_eos.mixture_equation_of_state` already supply the laws, inverse,
identities, uncertainties and domains. Solprop adapts them and tests its consumers;
it does not reproduce virial data or rerun upstream fitting in this proposal.

### 7. Verification and validation

| Check | Reference | Tolerance basis |
|---|---|---|
| Pure gas | Direct optmat `refractivity` and its derivative at fixed states | Roundoff; derivative by multi-step finite differences |
| Mixture | One component equals the pure gas; splitting or reordering labels changes nothing; the dilute limit | Summation bounds |
| EOS | Selected upstream state and inverse, fixed composition, vacuum and method/domain refusals; explicit legacy comparison | Upstream numerical bounds plus interpolation error; model differences reported separately |
| Shared response state | The same total/partial densities reach optics, Kerr, Raman amplitude/width/shift and existing plasma inputs; constant and gradient fills | Exact identity/label contracts, bounded table reconstruction |
| Raman compatibility | Exact FFT/recurrence and low-rank tables rebuilt after an EOS change; foreign-collider and reference-temperature restrictions retained | SEPs 1/10 source and approximation budgets |
| Cross terms | Sensitivity of $n-1$ and GVD to a plausible $B_{st}$ range for air and He/Ar mixtures | Reported, not a tolerance |
| Fast paths | Tabulated or expanded operators against direct evaluation along density profiles | Measured interpolation or expansion error |
| Propagation | A short capillary run at 10–30 bar against the dilute law, as a declared physics change | Report, with step convergence |
| Backend parity | As SEP 3 | Existing bounds |

### 8. Documentation

Update the materials and mixtures manuals with the optical law, the selected mixture
EOS, cross-term estimates/domains, recorded statuses, and independent dilute/legacy
comparison choices. Explain the shared state consumed by Raman and the remaining
microscopic limitations. Update the design-tools manual, since
zero-dispersion and phase-matching pressures shift, and regenerate its tables.

### 9. Compatibility and impact

Results change at every pressure, by the amounts in §2 in $n-1$, with larger relative
changes in GVD near zero dispersion. The manifest records the optical density law,
EOS method/model/status, composition, temperature and pair policy. These enter identity and resume across changed source
states is refused. Existing files remain readable. A coupled default switch requires
the separate acceptance/qualification decision of Q3, not this compatibility edit.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Keep the dilute law | Simple, but errs by 0.2–6 % in $n-1$ at the pressures fibres use |
| Per-species Lorentz–Lorenz, then sum | Incorrect for mixtures |
| Correct only the self-pair term inside the linear fast path | Still nonlinear in density; only hides the change |

### 11. Phases

Branches: `feature/sep-0011-finite-density-gas-index-<phase>`.

- **Phase 0 — State/consumer audit.** Use delivered optmat APIs; resolve Q2–Q3
  and Q5, qualify the public selections and enumerate all fill/flow/cache consumers.
- **Phase 1 — NumPy.** Shared mixture state and finite-density law, all consumers,
  exact/dilute/legacy comparisons, metadata, manuals and regenerated design tables.
- **Phases 2 and 3 — Numba and Kokkos CPU.** State/operator parity and any identified
  fast-path changes; final short-run accuracy/cost and compatibility evidence.

Follow the current development-slice and final qualification policy. The archival
upstream work is delivered evidence, not an unstarted solprop phase. No full-length
propagation is a test or benchmark requirement.

### 12. Open questions and dispositions

- **Q1 — Pair policy: direction resolved in the compatibility revision.** Consume
  optmat's current pair records and sensitivity/domain policy, explicitly labelling
  estimates and unknown terms. Do not describe zero cross terms as the delivered
  upstream default. The standalone human review will assess the documented domain.
- **Q2 — Temperature/domain admission.** Before acceptance, complete the range matrix
  for the selected mixture EOS, pair terms and gas-flow temperature paths. Use
  upstream laws within their domains; no temperature extrapolation is assumed here.
- **Q3 — Default switch.** SEP 3 is Final. Qualify this as a separate change with
  explicit opt-in comparisons first; the eventual default switch and compatibility
  release are decisions for this SEP's human acceptance review, not tied to a
  completed SEP 3 phase.
- **Q4 — Mixture density: include it.** John Travers selected the composition-preserving
  mixture-EOS direction on 2026-10-22. Follow actual upstream choices/statuses,
  including provisional models, rather than assuming multi-fluid is uniformly better.
- **Q5 — Flow and public state admission.** Before acceptance, complete the consumer
  audit for common fills, independent population paths, supplied materials, varying
  temperature, flow/choking/transport and parameter scans. Fix the exact admitted
  combinations without inventing data or silently converting independently specified
  populations into a shared thermodynamic state. SEPs 1 and 10 remain independent
  of this decision because their input is the resolved state path.

### 13. References

1. optmat, [Conventions](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/conventions.md)
   and [Gas models](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/gas_models.md).
2. H. J. Achtermann, G. Magnus and T. K. Bose, "Refractivity virial coefficients of
   gaseous CH4, C2H4, C2H6, CO2, SF6, H2, N2, He, and Ar," *J. Chem. Phys.* **94**,
   5669 (1991), [doi:10.1063/1.460478](https://doi.org/10.1063/1.460478).
3. G. Garberoglio and A. H. Harvey, "Path-integral calculation of the second
   dielectric and refractivity virial coefficients of helium, neon, and argon,"
   *J. Res. NIST* **125**, 125022 (2020),
   [doi:10.6028/jres.125.022](https://doi.org/10.6028/jres.125.022).

4. optmat 0.8.0, [Gas mixtures at finite density][mixture-manual], and
   [OEP 4](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/archive/oep-0004-mixture-index.md).
   Use the current manual/API for final selected defaults; earlier phase reports
   record historical intermediate choices.

[mixture-manual]: https://github.com/LupoLab/optmat/blob/v0.8.0/docs/mixtures.md

## Part II — Implementation record

No implementation phases have started.

## Part III — History

- 2026-09-29: Drafted from the finite-density theory first proposed inside SEP 3,
  when SEP 3 was split into a data migration with the kernel unchanged, with the
  dilute-law error measured against the audited optmat.
- 2026-09-30: Added the cost of the law on a density gradient (none: the expansion's
  degree is unchanged) and the error of the partial-pressure mixture density, measured
  against CoolProp's multi-fluid models; linked optmat's OEP 4; added Q4.
- 2026-10-01: optmat's OEP 4 is Final (optmat 0.1.0); its link and the upstream note
  updated, and Q4 noted as SEP 3's D17 left it.

- 2026-10-22: Compatibility revision for SEPs 1 and 10: included a composition-preserving
  mixture EOS, corrected the upstream pair/default/status descriptions, specified
  shared response state and cache invalidation, and retained explicit questions for
  the standalone acceptance review. This is not a completed full review of SEP 11.

### Compatibility findings and response — 2026-10-22 — GPT-6

1. **C1 (P1), §§2, 6, Q4:** excluding EOS admits density/composition errors of the
   same scale as the proposed optical correction. **Decision:** John Travers selected
   inclusion. **Response:** §§2, 4–6 and Q4 bind the shared state to the mixture EOS.
2. **C2 (P1), §§4, 6, Q1:** zero unlike-pair terms and uniformly qualified multi-fluid
   models do not describe current optmat. **Response:** use the selected pair records,
   uncertainty assumptions, actual method/status table and range checks. This does
   not claim missing coefficients have become measurements.
3. **C3 (P1), §§3, 6–9:** Raman amplitudes, collision laws and linear optics must
   consume one consistent state, preserving labels and one-time mole fractions.
   **Response:** common-state and cache rules are explicit; local-field corrections,
   collision data and local thermal populations remain separate physical work.
4. **C4 (P2), §§6, 11–12:** a few old expansion degrees cannot prove unchanged cost,
   and completed upstream phases/default dates are stale. **Response:** retain the
   old measurements as motivation only, update phases and name the remaining flow/
   domain/default audit. SEP 11 stays Draft / Not reviewed pending its full review.

The installed optmat 0.8.0 public API was called for equal-fraction He/Ne, He/Ar,
Ne/Ar and named dry air. It returned respectively `virial/validated`,
`virial/provisional`, `virial/provisional` and `multi-fluid/provisional`, with
`ModelRangeWarning` on the provisional selections. Source/manual inspection checked
the pair policy and the distinction between partial-density and partial-pressure.
The old refractivity/degree tables and primary virial measurements were not remeasured
or independently re-transcribed in this compatibility review.
