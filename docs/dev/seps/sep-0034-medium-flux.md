# SEP 34 — Medium flux for free-space propagation

| | |
|---|---|
| **SEP** | 34 |
| **Title** | Medium flux for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 |
| **Status** | Draft |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-10-23 |
| **Resolution** | — |
| **Requires** | [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md): spatial execution and memory; [SEP 25](archive/sep-0025-rounding-level-reproducibility.md): qualification; [SEP 28](archive/sep-0028-free-space-evaluation.md): bounded spatial evaluation |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Give free-space launches, noise, energy, fluence and statistics one physical
power convention in the declared medium. Replace the vacuum weight with the
scalar forward equation's axial-admittance measure, using complex admittance
for local flux and its real part for total energy. Preserve the existing radial
gas-gradient capability by adding the adiabatic amplitude correction to its
integrated linear operator. Qualify the reference, Numba CPU and native CPU
implementations before activating the changed public semantics together.
This is a clean breaking change with no legacy normalisation or old-file
migration. It is the shared prerequisite for coupling in
[SEP 31](sep-0031-simulation-coupling.md) and solid carriers in
[SEP 5](sep-0005-solid-ionisation.md), not an implementation of either feature.

### 2. Motivation and scope

A user specifies the energy measured for a beam, not an equivalent electric
field norm in vacuum. At a plane normal to propagation the energy is the time
and area integral of axial Poynting flux. A launch in glass or gas must therefore
use the medium at that plane. An energy-meter reading in air before a window
becomes an in-medium energy through an explicitly specified Fresnel or measured
transmission; the launch builder must not invent that interface.

Today spatial launches and observations use the vacuum factor
$\varepsilon_0c/2$, while modal launches already use true power. Fixing only an
offline accessor would leave launch amplitudes, noise and in-run observations
inconsistent. Fixing only one backend would make the default backend disagree
with the reference. The correction belongs in a separate proposal from the
host-side field exchange and transport of SEP 31.

| Route or operation | Scope |
|---|---|
| Radial, Cartesian and uniform-in-y line free space | Shared scalar medium flux in uniform isotropic media; J and W for finite beams, J/m and W/m for line beams |
| Radial gas pressure/density profiles | Preserve smooth admitted profiles with the adiabatic correction, local source prefactors and local-plane measurement |
| Cartesian and line varying media | Remain unsupported; this proposal does not add a propagation route |
| Hybrid planar guide | Audit its modal/FFT power metric and all public spatial dispatches; do not multiply true-power slab coefficients by a bulk index |
| Ordinary and general modal guides | Retain their own derived metrics; provide consistent dispatch, not a new normalisation |
| Uniform absorbing scalar media | Integrated energy and attenuation use real admittance; local traces use the complex forward admittance |
| Longitudinally varying absorbing media | The explicitly qualified weak-loss adiabatic extension in §4.4, not arbitrary complex-index WKB |
| Interfaces, high-NA vector fields, crystals | Outside scope; SEP 31 handles declared planar interfaces, SEP 8 handles vector crystals |

No new material data, nonlinear response, spatial mode solver or dependency is
introduced. General reflected/evanescent scattering and nonadiabatic gradients
remain outside the propagation model. An abrupt step is not a gas-gradient
approximation merely because its index contrast is small.

### 3. Context

The [free-space model](../../guide/free-space/model.md) states the present vacuum
normalisation and documents the exact longitudinal root and radial pressure
profiles. The implementation paths to change together are:

- `solprop.spatial`: energy, fluence, intensity, on-axis measurements, spectral
  measures and the critical-power diagnostic;
- `solprop.pulses.SpatialPulse` and `solprop.noise.spatial_noise`: physical
  incident scaling and the one-photon-per-bin model;
- `solprop.translate.radial.radial_operators`: the integrated gradient operator,
  parameter expansions, frame tables and local nonlinear prefactor;
- `solprop.translate.spatial` and the spatial backend/statistics paths: bounded
  coefficient evaluation, reconstruction and in-run diagnostics;
- `solprop.processing`, output and input replacement: the medium/reference
  identity at the selected plane and the same physical measure on saved fields;
- `native/`: the native backend in this repository, distributed as
  `solprop-native`, under the repository's `native/AGENTS.md` instructions.

The source uses $-i\omega^2/(2\varepsilon_0c^2\operatorname{Re}k_z)$ for
polarisation. Its constants deliberately avoid an independently rounded
$\mu_0\varepsilon_0c^2$ product. The radial gradient already has a cumulative
diagonal operator, but omits the changing-admittance amplitude term.
This is a code/design audit, not a qualification result.

[SEP 5](sep-0005-solid-ionisation.md) originally sketched scalar flux.
[SEP 31](sep-0031-simulation-coupling.md)'s review separated this prerequisite
from coupling and explicitly chose physical launch energy and preservation of
radial gradients. [SEP 8](sep-0008-quadratic-crystal-optics.md)'s anisotropic
vector E/H metric remains distinct; [SEP 9](sep-0009-propagation-statistics.md)
consumes the shared spatial definition without moving its modal measurement
catalogue here. SEP 31 depends on this proposal, not conversely.

### 4. Theory

#### 4.1 What energy means

Use SI and the [optics convention](../conventions.md),
$\exp[i(kz-\omega t)]$. For analytic physical fields define

$$
 S_z(\mathbf r,t)=\frac12\operatorname{Re}
   [\mathbf E_a\times\mathbf H_a^*]\cdot\hat{\mathbf z},\qquad
 F(\mathbf r)=\int S_z\,dt,\qquad U=\int F\,dA.
$$

For a transparent nonmagnetic paraxial plane wave,
$\mathbf H_a=n_0\varepsilon_0c\,\hat{\mathbf z}\times\mathbf E_a$ and
$S_z=n_0\varepsilon_0c|E_a|^2/2$. The phase index appears here, not the group
index. In the narrowband transparent dispersive limit, the cycle-averaged
energy density is $u=\varepsilon_0n_0n_g|E_a|^2/2$; multiplying by
$v_g=c/n_g$ gives the same flux. This relation is not an energy-density model
for strongly absorbing or arbitrarily broadband matter.

At fixed requested energy, vacuum normalisation overstates the paraxial electric
field by $\sqrt{n_0}$ relative to medium normalisation. For a fixed supplied
physical field it underreports energy by $n_0$. Never rescale supplied V/m data
merely to keep an old reported energy unchanged. Energy crossing a plane is
conserved only when no energy is absorbed, reflected out of the forward model,
or removed by the finite numerical domain.

#### 4.2 Scalar axial admittance and local flux

For each transverse/frequency column in a homogeneous isotropic medium, set

$$
 k=n(\omega)\omega/c,\quad k_z=\sqrt{k^2-k_\perp^2},\quad
 \mathcal Y_c=\varepsilon_0c^2\,k_z/\omega,\quad
 Y_z=\operatorname{Re}\mathcal Y_c.
$$

Use the current forward branch and admitted propagating support. This is the
scalar equation's transverse electric/magnetic closure: for a TE plane wave it
follows from $\mathbf k\times\mathbf E=\omega\mu_0\mathbf H$.
Applying it independently to transverse components retains the current scalar
model; it is not the full TE/TM Maxwell metric at high NA.

Reconstruct $\mathbf G_a$ by applying **complex** $\mathcal Y_c$ before the
inverse spatial and temporal transforms, and use

$$
 S_z=\tfrac12\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*).
$$

Integration over the complete transverse plane and time diagonalises the
transforms, so total energy depends only on $Y_z$. Local quantities do not:
cross-frequency and cross-angle terms can contain
$\operatorname{Im}\mathcal Y_c$. Replacing $\mathcal Y_c$ by $Y_z$ before
reconstruction preserves total energy but can change local flux in an absorbing
medium. This extends the real-admittance sketch in the original SEP 31.
The two definitions coincide for transparent homogeneous propagating columns.

Local broadband flux and fluence can be signed. Do not clip them or silently
substitute an electric-field norm. A width/FWHM statistic requiring a positive
profile refuses a signed profile that its definition cannot handle. An explicitly
named field-intensity diagnostic may still measure $|E_a|^2$. Ionisation is driven
by physical E in V/m and its established rate convention, never by a signed
broadband flux. Local flux uses the declared forward linear closure; nonlinear
impedance corrections outside the scalar UPPE are not supplied by this SEP.

For uniform material loss a column evolves as $\exp(ik_zL)$ and its integrated
energy as $\exp[-2\operatorname{Im}k_zL]$. This attenuation test is distinct
from numerical absorber accounting. A spatial screen or collar need not commute
with the nonlocal admittance; measure its quadratic-form change rather than
calling a local field-norm decrement physical absorption.

#### 4.3 Discrete normalisation, endpoints and noise

Let $\hat E_{jpk}$ be the raw temporal DFT at transverse spectral column j,
polarisation p and frequency k, and $v_j$ its transverse Parseval weight.
For an envelope, $\gamma_k=1$. For a real-field grid, $\gamma_k=4$ on interior
positive frequencies and 1 at DC/Nyquist, matching the analytic-signal convention
of `fourier.energy_omega`. Then

$$
 U=\frac{\Delta t}{2N_t}\sum_{j,p,k}
       v_j\gamma_kY_{z,jk}|\hat E_{jpk}|^2.
$$

The reference time/space integration is
$\tfrac12\int\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*)$.
For a real constant admittance it reduces to
$Y_z\int|E_a|^2/2$, not an unqualified $\int E^2$: the analytic-signal
endpoints make those expressions differ if endpoints are occupied.
Use synthetic finite constant-admittance endpoint tests for the discrete
identity, without assigning the optical $1/\omega$ formula to DC.

Optical energy-normalised launches require unoccupied DC and unsupported
nonpropagating/cutoff columns. Use established hard-mask zeros. Reject division
by zero or ill-conditioned admittance. Identity storage retains structural
endpoint rules; representation changes must refuse a Nyquist component they
cannot represent. Numerical evanescent damping in the current route does not
turn an evanescent column into a physical forward power port.

For the declared one-photon-per-bin noise model, a realised coefficient in an
admitted transverse/frequency degree of freedom is scaled so its contribution
to this discrete energy is $\hbar\omega$. Use the local entrance metric and the
same transverse weights as the launch. This changes no noise ensemble or seed
semantics. Modified-noise bookkeeping retains its existing meaning; an energy
accessor or a relaunch never adds another draw automatically.

The planar hybrid route uses its derived modal/FFT quadratic form, including
cross terms if required. Its slab amplitudes are not raw bulk E samples. Audit
dispatch and current vacuum factors, and demonstrate agreement with physical
modal synthesis; never apply the bulk index twice.

#### 4.4 Radial adiabatic amplitude transport

The existing radial profile is uniform transversely and varies along z. In the
transparent case each spectral column satisfies the scalar linear equation

$$
 E''+q(z)^2E=0,\qquad q(z)=k_z(z)>0.
$$

Write $E=A(z)e^{i\int q\,dz}$. Substitution gives
$A''+2iqA'+iq'A=0$. Dropping $A''$ in the slow-variation expansion gives
$A'/A=-q'/(2q)=-\tfrac12\partial_z\ln Y_z$ and hence

$$
 \partial_z E=\left(ik_z-\tfrac12\partial_z\ln Y_z\right)E,\qquad
 E(z_1)=\sqrt{\frac{Y_z(z_0)}{Y_z(z_1)}}\,
        e^{i\int_{z_0}^{z_1}k_z\,dz}E(z_0).
$$

Thus $Y_z|E|^2$ is invariant within this forward model. Paraxially the
amplitude ratio becomes $\sqrt{n(z_0)/n(z_1)}$. Keep E as the propagated
variable; changing to a globally flux-normalised variable would also require
changing every nonlinear drive and source, with no benefit here.

In the internal convention and moving frame the accumulated linear operator is

$$
 \Phi(z_1)-\Phi(z_0)
 =-i\int_{z_0}^{z_1}[\operatorname{Re}k_z-R]\,dz
  -\int_{z_0}^{z_1}\operatorname{Im}k_z\,dz
  -\frac12\ln\frac{Y_z(z_1)}{Y_z(z_0)}
  -\frac12\int_{z_0}^{z_1}\alpha_{\rm num}\,dz .
$$

R is exactly the existing removed frame term. The real logarithmic term does
not change Fourier signs, carrier references or delay. Evaluate it from shared
endpoint admittance, using a stable log-ratio for near-equal values, rather than
integrating a numerically differentiated table. Compose endpoints consistently
across steps and restarts. The uniform-medium term is an exact zero.

Local magnetic reconstruction in the adiabatic linear model also has a gradient
term. Since $G=(-i/\mu_0\omega)\partial_z E$ in the optics convention,

$$
 \mathcal Y_{c,\rm ad}
 =\frac{\varepsilon_0c^2}{\omega}
   \left(k_z+\frac{i}{2}\partial_z\ln Y_z\right).
$$

Its real part remains $Y_z$, so total-energy and noise weights are unchanged;
use this closure for local traces to the same adiabatic order. The derivative
comes from the resolved smooth material path, not differencing noisy saved
fields. This is a reconstruction term, not a second amplitude correction.

For weak varying loss, keep the existing complex phase/attenuation and the real
admittance correction above. This is a declared separated-loss scalar model,
not the exact $k_z^{-1/2}$ WKB solution for an arbitrary complex q. Qualify its
error independently against the complex scalar wave equation over the admitted
loss/gradient range. Strongly absorbing varying profiles without that evidence
are refused. Uniform absorption remains supported by §4.2.

#### 4.5 Validity, reflection and source work

Local indicators $|q'|/q^2$ and $|q''|/q^3$, endpoint smoothness and a margin
from $q=0$ screen adiabaticity; none alone bounds accumulated reflection.
For the trial WKB field the equation residual divided by E is

$$
 r_{\rm WKB}=\frac34(q'/q)^2-\frac12q''/q.
$$

Use this residual and an independently converged one-dimensional scalar-wave
reference to qualify profile families and an explicit model-error budget.
Continuous piecewise profiles retain their declared derivative breakpoints;
derivative jumps need their own reflection/error assessment [2]. They are not
silently smoothed. A jump in the index itself requires an interface model.

A conservative transparent-profile reflection screen is available without a
wavelength heuristic. Define $\theta(z)=\int q\,dz$ and exact amplitudes by
$E=q^{-1/2}(a e^{i\theta}+b e^{-i\theta})$ and
$E'=i q^{1/2}(a e^{i\theta}-b e^{-i\theta})$. Their coupled equations are
$a'=(q'/2q)b e^{-2i\theta}$ and $b'=(q'/2q)a e^{2i\theta}$, preserving
$|a|^2-|b|^2$. For unit transmitted flux and no incoming wave at the far end,
write the magnitudes as $\cosh\rho$ and $\sinh\rho$. The equations bound
$|\rho'|\le|q'/(2q)|$. With
$\Lambda=\tfrac12\int|q'/q|dz$, this gives
$R\le\tanh^2\Lambda$ for reflected power relative to incident power.
This bound can be very conservative because it discards oscillatory cancellation.
Use it only as a sufficient reflection bound; field and phase errors still need
their separate reference bounds. When the screen is insufficient, qualify the
actual smooth profile with a refined independent reference or refuse the
requested accuracy. Do not describe a maximum local WKB indicator as a
certified global error.

**The abrupt-interface limit is not an acceptance target for this propagator.**
For a lossless step from n1 to n2, the amplitude ratio above always gives
$T_{\rm ad}=1$, whereas Fresnel continuity gives
$T_{\rm F}=4n_1n_2/(n_1+n_2)^2=1-R_{\rm F}$.
Sharpening a profile eventually leaves the adiabatic regime; the reference
scattering solution approaches the Fresnel limit, and the forward-only model
must signal that its error budget is exceeded. It cannot generate the missing
reflected wave. This corrects the R2 validation request in SEP 31's review.

The nonlinear source prefactors continue to use the **local** real $k_z$.
For an analytic current the field source is $-J/(2Y_z)$; combining it with
the energy weight gives the work term
$-\operatorname{Re}(E^*J)/2$. The derivative of the weight cancels the
adiabatic amplitude term in the energy derivative:

$$
 \partial_z\!\left(\tfrac12Y_z|E|^2\right)
 =-2\operatorname{Im}k_z\left(\tfrac12Y_z|E|^2\right)
   -\tfrac12\operatorname{Re}(E^*J)
$$

apart from polarisation work and declared numerical absorption.
No extra square root of the index belongs in J or in the physical ionisation
drive. Qualify polarisation/current work separately from gradient transport.

### 5. Proposed interface

Provide one resolved, immutable `SpatialFlux` specification in `solprop.spatial`,
binding the temporal/spatial representation, medium identity and reference plane.
Translation builds its numerical plan; physics does not import a backend.
The public spelling is proposed here:

| Entry point | Contract |
|---|---|
| `SpatialFlux.resolve(grid, space, medium, z=...)` | Resolve units, propagating support and local complex/real admittance or the hybrid modal metric |
| `spatial_energy(field, flux)` | Integrated physical energy on the declared plane; no vacuum default when the medium is omitted |
| `spatial_fluence(field, flux)` and local intensity/axis accessors | Reconstruct E and G with the same complex closure; signed quantities remain signed |
| `SpatialPulse.build(..., medium=..., reference=...)` | Normalise requested energy or peak axial power at the declared launch plane, after geometric preparation |
| `spatial_noise(..., flux=...)` | Normalise the existing draw in the same entrance metric |
| `AdiabaticAccuracy(field_error=..., reflected_energy=...)` | Explicit positive model-error budgets for a varying radial medium, separate from integration tolerance |
| `critical_power(..., index=...)` | Physical paraxial power at the specified real carrier index; not a broadband collapse prediction |

The exact array signatures follow the existing axis vocabulary and are fixed in
the first implementation phase. The required medium/reference is an intentional
breaking signature change. `SpatialPulse` remains the beam builder; SEP 31 adds
its field-plane adapter rather than duplicating its Gaussian.

For the present scalar monochromatic Kerr model, the physical critical-power
baseline is n0 times the current vacuum-measure value at fixed χ³. Derive this
from the same paraxial equation and reuse the existing cited Townes mass and
precision; do not add another empirical coefficient. Update the line-soliton
power scale and the planar diagnostic explanation consistently with their own
units/metric. A line beam still has no two-dimensional collapse threshold.

Record the flux-model identity, medium at the plane, adiabatic admission evidence,
units and source material identities. Accessors on a saved radial-gradient plane
resolve its actual z. Missing metadata or a metric/basis mismatch is an explicit
error. SEP 31 uses this same host plan; it need not ask a propagation kernel to
measure an offline field.

### 6. Implementation strategies

#### 6.1 Keep the physical state and bounded plans

Retain V/m in physical space and the existing Hankel/DFT state scaling. Physical
models and admission records belong in `solprop.spatial`, quadrature/transform
and admittance plans in translation, hot application in backends, and output
binding in processing. Share mathematical definitions and tests without making
backend code query a material model.

Use the existing separable material/root representation and bounded node/frequency
chunks. Local flux needs E and G synthesis, so memory and transform cost must be
accounted for explicitly. Avoid a permanent transverse-by-frequency admittance
array on the separable routes; radial tables retain their established bounds.
Update `memory_ledger()` and the bounds of SEPs 22 and 28.

For a gradient, the logarithmic amplitude contribution is an endpoint term in
the existing cumulative operator. Update both table and parameter-expansion
paths, current-source prefactors, inverse-stage growth bounds, interpolation,
batch reuse, dense output and resume. Reuse one resolved endpoint weight in
propagation and observations: otherwise two accurate interpolants can still
fail the discrete conservation identity. A conservation test of the represented
diagonal flow uses rounding bounds; comparison with the physical material path
also includes interpolation and WKB error.

There is no extra propagation FFT for the amplitude correction, but endpoint
evaluation, local G reconstruction and extra tables cost time and memory.
Measure them. The review's “costs nothing per step” is not a performance promise.

#### 6.2 Atomic public activation

Implement the reference first, then Numba CPU, then native CPU in `native/`
on the same feature branch. The public launch/noise/statistics switch and
gradient correction land **together** after parity, not in three independently
activated backend phases. A temporary development state may not reach main if
it makes the default compiled backend refuse a previously supported run or
assign a different energy to the same input.

Private preparation helpers/tests may merge before activation only if they
leave the existing public path unchanged. They are not a user-selectable legacy
normalisation. The activation removes replaced definitions and updates callers,
tests, manual and current-schema contracts in the same change.

Device kernels require the matching implementation and separate device evidence;
portable source alone does not establish GPU support. The activation report must
state the GPU capability matrix and any accepted, explicit temporary refusal.
There is no hidden host statistics fallback. Existing CPU support is preserved.

Any changed seam records, protected driver hooks or stage bundles bump
`SCHEMA_VERSION` and its literal contracts, with native consumers updated
together. Output metric and plane-reference semantics require an output-schema
bump. Version numbers are assigned in implementation, not preselected here.
No dependency is added.

### 7. Verification and validation

Use the repository's [accuracy](../accuracy.md) and [testing](../testing.md)
rules. Separate rounding error of a fixed discrete map from quadrature, material
interpolation, propagation and physical-model error. Qualify three refinement
levels when asserting convergence; physical energy conservation within WKB is
not proof that WKB approximates the full wave equation.

| Case | Required evidence |
|---|---|
| Uniform transparent medium | Analytic plane-wave index factor; spatial/temporal Parseval; radial/Cartesian symmetric-beam agreement; line units |
| Complex local admittance | Two-frequency and two-angle absorbing-medium references from the scalar E/H closure; detect the error from discarding imaginary admittance; total energy still equals real-admittance spectral integration |
| Field/envelope endpoints | Analytic-signal direct integration; separate synthetic endpoint identity from refused occupied optical DC/cutoff |
| Launch and supplied arrays | Correct physical requested energy at launch plane; fixed V/m data retain their amplitude; no implicit entrance-window transmission |
| Noise | Photon energy per admitted degree of freedom, fixed-seed consistency, no double draw and explicit invalid-weight errors |
| Gradient linear flow | Lossless represented endpoint-flow invariant within a rounding budget; exact zero correction in a uniform medium; forward/backward plane exchange and resume composition |
| Gradient physical model | Independently refined scalar Helmholtz/mode-matching references for smooth rising/falling gas paths, derivatives and cutoffs; field, phase and reflection errors against the declared budgets |
| Sharpening profile | Independent full-wave reference approaches Fresnel scattering; forward model exits admission rather than falsely producing the abrupt-interface transmission |
| Nonlinear work | Local polarisation/current prefactor consistency; matched uniform limit and short gradient Kerr/plasma cases; physical E remains the rate drive |
| Absorption | Uniform loss exponential, qualified weak-loss gradients, and separate physical versus numerical-absorber accounting |
| Signed/local statistics | No clipping; explicit refusal for undefined positive-profile measurements; common in-run/offline definitions and correct selected-plane medium |
| Other representations | Hybrid planar reconstruction/metric identity and modal dispatch; no accidental bulk-index multiplication of true-power coefficients |
| Backend and resources | NumPy/Numba/native CPU parity, source/target host exchange, current-schema round trips, memory ledger and short benchmarks |

Existing numerical absorbers and nonpropagating-support rules need regression
coverage. Show the effect of finite transverse/time windows independently.
For nonlinear runs, work balance and propagation accuracy have their own
envelopes; they are not expected to obey a pure-linear rounding invariant.
Benchmark construction, one statistic, one source evaluation and about ten
steps. Never extrapolate these timings to full experiments.

### 8. Documentation

Add `docs/guide/free-space/flux.md` with the meter-energy definition, phase versus
group index, complex local admittance, analytic-signal conventions, radial WKB
derivation, validity/refusal rules and a short rising/falling gas-profile example.
Link it from the free-space model, power/intensity, noise, input-pulse, radial
and coupling guides. Update docstrings and all affected callers together.

Recompute focused tutorial energy scales and critical-power ratios under the new
definition, preserving whether each example is meant to hold physical energy or
physical field fixed. Do not silently preserve both. SEP 31 owns its nonlinear
coupling tutorial; this SEP supplies its correct physical sizing.
Worked-example source changes require regeneration under the repository rules.

Update conventions, kernel/output contracts, backend capability pages,
architecture and the Luna differences page in implementation, along with the
changelog. A draft does not change current-design documentation to describe
unimplemented physics.

### 9. Compatibility and impact

This is an unreleased clean breaking change. There is no legacy vacuum option,
deprecation alias, old-file reader or migration obligation. Unsupported schemas
fail clearly. The scalar field state and Fourier convention remain unchanged,
but an energy-normalised pulse in a medium generally has a different physical
amplitude, and spatial observations report a different energy.

Smooth admitted radial pressure profiles continue to work with the corrected
amplitude transport. Nonadiabatic, discontinuous or unqualified strongly
absorbing gradients require another model. Cartesian/line gradient restrictions
remain as they are. Exact uniform propagation of a supplied array is unchanged;
its launch scaling or reported physical power can change.

The scalar measure and adiabatic closure do not replace SEP 8's vector crystal
fields. General junction scattering remains outside this SEP. Shared consumers
name SEP 34 as the flux dependency, while SEP 31 remains the coupling proposal.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Keep all flux work in SEP 31 | Couples backend-wide semantics to host field exchange; split the shared foundation |
| Correct only offline energy | Leaves launch, noise and in-run statistics inconsistent |
| Multiply every observable by one index | Only a narrowband paraxial limit; misses angular weighting and complex local flux |
| Use group index in the Poynting factor | Confuses transported energy density with energy crossing a plane |
| Use real admittance for every local trace | Correct total energy, incomplete local flux for absorbing broadband/angular fields |
| Refuse every gradient | Removes a working route; add the adiabatic amplitude term for admitted profiles |
| Demand the abrupt Fresnel limit from WKB | Mathematically impossible for a forward-flux-conserving reduction; use a scattering reference to delimit validity |
| Propagate a new flux-normalised field | Requires changing all nonlinear drives/sources; retain physical E and add the endpoint term |
| Activate one backend at a time | Creates inconsistent semantics or rejects default-backend runs; switch public paths atomically |
| Add a general reflected-wave solver now | Broader model and state; keep the independent reference for qualification, not production propagation |

### 11. Phases

No implementation starts before acceptance. Each mergeable phase appends a
report to Part II; the reference-to-native sequence inside Phase 2 consists of
development slices, not separate public activations.

| Phase | Deliverables | Exit evidence |
|---|---|---|
| **1 — Shared reference and admission** | Private physical/complex-flux reference, WKB endpoint and local-G derivations, independent scalar-wave witnesses, proposed signatures and provider/admission records | Discrete identities, reflected-wave counterexample, explicit error budgets and unchanged existing public behaviour |
| **2 — Atomic CPU implementation** | NumPy first, then Numba and native CPU; update launch/noise/statistics, gradient flow, critical-power references, host processing, schemas and manual together | All directly affected modules, three-CPU parity, native gate, memory/timing evidence and no default-backend support gap |
| **3 — Qualification and completion** | Integrated gradient/absorber/reference tests, updated galleries, final capability matrix and handoff to SEPs 5/31 | Full gate, slow/coverage lanes once under the standing policy, documentation and recorded source/limitation reports |

During development run fast checks and directly affected tests; after a failure
rerun failing modules, not whole lanes without cause. Changes in `native/` follow
its instructions and native gate. CPU and GPU evidence is recorded separately.
This draft requires only whitespace, changed-file spelling, SEP consistency
tests and Sphinx checks. No propagation implementation is claimed.

### 12. Open questions

| Question | Resolution required |
|---|---|
| **Q1 — Adiabatic admission data** | Phase 1 fixes evaluable derivative/profile bounds and independent reference coverage for gas mixtures, derivative breakpoints and weak varying loss. The forward model and refusal of abrupt steps are already proposed above. |
| **Q2 — Local trace closure** | Proposed resolution: complex homogeneous admittance and the stated adiabatic imaginary derivative term. Phase 1 independently verifies signs, cross-frequency terms and the approximation to the full scalar wave solution. |
| **Q3 — Hybrid planar audit** | Phase 1 establishes the actual modal/FFT metric and lists every affected public dispatch; no bulk formula is applied by analogy. |
| **Q4 — GPU activation** | Phase 2 records device evidence and any explicit accepted exception before public activation. No CPU exception is proposed. |
| **Q5 — Error and resource interfaces** | Phase 1 fixes exact signatures and model-error defaults, if any, from independent evidence; no unrecorded physics tolerance or memory allocation is permitted. |

The physical-energy meaning, no-legacy policy, split from SEP 31, preservation
of admitted radial gradients and three-CPU atomic activation are settled
requirements. The new proposal has not inherited SEP 31's reviews or acceptance.

### 13. References

1. M. Kolesik and J. V. Moloney, “Nonlinear optical pulse propagation simulation:
   From Maxwell's to unidirectional equations,” *Physical Review E* **70**,
   036604 (2004).
   [doi:10.1103/PhysRevE.70.036604](https://doi.org/10.1103/PhysRevE.70.036604).
2. R. Diamant and M. Fernández-Guasti, “Reflection coefficient due to a
   discontinuity in the nth order derivative of the refractive index,”
   *Journal of Optics* **16**, 015707 (2014).
   [doi:10.1088/2040-8978/16/1/015707](https://doi.org/10.1088/2040-8978/16/1/015707);
   [author manuscript](https://luz.izt.uam.mx/mfg/arti/11-15/refl-disc-jo2014.pdf).

**Source audit.** The current spatial model, launch/noise functions, radial
operator construction, conventions and critical-power documentation were
inspected. The publisher record and abstract of [1] were checked for its
unidirectional scope, not its complete derivation. The introduction and
derivative-discontinuity discussion of [2] were inspected for the distinction
between adiabatic and reflected-wave models; no coefficient from it is copied.
The flux, endpoint, WKB and work identities above are derived from the stated
scalar equations. A static Fresnel counterexample and a two-tone complex
admittance identity were evaluated. The latter confirms equal integrated energy
but different local traces when imaginary admittance is discarded, with a
rounding-sized residual against its analytic cross term. No new gradient
propagation or backend implementation was run or qualified.
Material constants remain supplied and referenced by optmat. The existing
critical-power coefficient is reused through its current cited derivation,
not revalidated to additional precision by this draft.

## Part II — Implementation record

## Part III — History

- 2026-10-23: Drafted from SEP 31 review R1, R2 and R5 and the maintainer's
  decisions to split the prerequisite, preserve admitted radial gradients and
  define launch energy physically. Included atomic CPU activation and a
  correction to the review's abrupt-interface validation target. The maintainer
  additionally chose full complex admittance for local flux traces; the real
  part remains the total-energy/noise measure. No implementation has started.
- 2026-10-23: Reviewed by Opus 5.5 (AI), A1–A7 below; Part I unchanged. Review
  status In review pending a human review; status remains Draft.
- 2026-10-23: John Travers's human review recorded at his request. Review status
  Reviewed records completion of review, not acceptance; status remains Draft.
- 2026-10-23: Maintainer decisions recorded beside A1 (default 10⁻⁴ adiabatic
  budget), A2 (short paid device qualification before activation) and A3 (exact
  in-run local flux). Part I unchanged.

### AI review — 2026-10-23

**Assessment.** This is a sound, well-scoped change, and it answers the question
that prompted it: `energy=` becomes the energy a meter would measure. The physics
is correct where I checked it.

- **Flux and index.** The phase-index flux and the energy-density cross-check
  (§4.1) are correct.
- **Admittance.** The complex/real admittance split (§4.2) is correct: full
  transverse and time integration leaves only diagonal $\operatorname{Re}\mathcal Y_c$.
- **WKB and closure.** The WKB amplitude term (§4.4) and the adiabatic magnetic
  closure $\mathcal Y_{c,\rm ad}$ follow from $\mathbf H=(-i/\mu_0\omega)\partial_z\mathbf E$.
- **Reflection bound.** The Bremmer equations and the $\tanh^2\Lambda$ bound (§4.5)
  are correct.
- **Work identity.** The work identity's cancellation of the gradient term is right.
- **Critical power.** Scaling the critical power by $n_0$ is right, because the
  Townes threshold fixes $\int|E|^2$ independently of $n_0$ at fixed χ³.

Atomic three-CPU activation addresses the merge hazard raised against SEP 31.

The main recommendation is a simplification. The draft's own reflection bound
extends to a closed-form forward-field certificate. That certificate makes
profile-family reference qualification, a required `AdiabaticAccuracy` argument
and the blanket refusal of index steps unnecessary for the gas profiles this SEP
exists to preserve (A1). The GPU consequence of a shared native source needs a
decision before acceptance (A2).

**Review basis.** Read all of Part I against SEP 0, SEP 31 (revised), the
free-space model page and the code: the source prefactor in
`spatial.SpatialMedium`'s radial operator, the vacuum factors in `solprop.spatial`,
`fourier.energy_omega`, `profiles.PiecewiseLinear` and the `ZProfile` protocol,
and the planar route's power weights. Each code claim in §3 matched. Both
references resolve via Crossref to the stated authors, titles, volumes and
article numbers. Ran an independent transfer-matrix solution of $E''+q^2E=0$
(200 000 layers) for the cases in A1. This is a design review; nothing was
implemented. Priorities as in SEP 31's review: **P1** before acceptance, **P2** a
revision or recorded decision, **P3** editorial.

#### A1 — P1: Replace profile qualification with the closed-form certificate

**Sections:** 4.5, 5 (`AdiabaticAccuracy`), 7 (gradient rows), 9, 11 (phase 1),
12 (Q1).
**Maintainer decision (2026-10-23):** `AdiabaticAccuracy` has a default budget
of 10⁻⁴ (forward-field error and reflected energy); callers may tighten it, and
a path exceeding it refuses.

§4.5 derives $|\rho'|\le|q'/2q|$ and hence $R\le\tanh^2\Lambda$, but then treats it as
a crude screen and requires independent scalar-wave references per profile family,
with index steps refused outright. The same coupled equations bound the error of
the forward wave the propagator actually keeps:

- From $a'=(q'/2q)be^{-2i\theta}$ and $|b|/|a|=\tanh\rho\le\tanh\Lambda$,
  both $|\partial_z\ln|a||$ and $|\partial_z\arg a|$ are at most
  $|q'/2q|\tanh\Lambda$. The forward wave's amplitude and phase therefore differ
  from the adiabatic model by at most $\Lambda\tanh\Lambda$ (relative), while
  reflected power is at most $\tanh^2\Lambda$.
- Here $\Lambda=\tfrac12\operatorname{TV}(\ln q)$ over the path. It is defined for
  kinks and for index jumps, where $q'$ is a delta. For a monotone path
  $\Lambda=\tfrac12|\ln(q_1/q_0)|$, so $\tanh^2\Lambda=((q_1-q_0)/(q_1+q_0))^2$ is
  **exactly the endpoint Fresnel reflectance**. The bound is attained by a step and
  is conservative for any smooth ramp.

Transfer-matrix check (forward error is $|t_{\rm exact}/t_{\rm WKB}-1|$ on the
transmitted field):

| Profile (800 nm) | R exact | tanh²Λ | forward error | Λ tanh Λ |
|---|---|---|---|---|
| Gas 0 → 3 bar (Δn = 8×10⁻⁴), tanh ramp, 1 mm | 2×10⁻²⁴ | 1.5×10⁻⁷ | 7×10⁻¹² | 1.5×10⁻⁷ |
| Same contrast, abrupt step | 1.60×10⁻⁷ | 1.60×10⁻⁷ | 8.0×10⁻⁸ | 1.6×10⁻⁷ |
| Non-monotone gas bump (Gaussian, 0.2 mm) | 6×10⁻²⁵ | 6.4×10⁻⁷ | 6×10⁻¹¹ | 6.4×10⁻⁷ |
| n = 1 → 1.5, tanh ramp, 0.5 μm | 7×10⁻¹⁰ | 0.040 | 1.4×10⁻³ | 0.040 |
| n = 1 → 1.5, abrupt step | 0.0400 | 0.0400 | 0.020 | 0.041 |

Every case sits inside both bounds. R + T = 1 to 10⁻¹¹ confirms the solver.

**Consequences.** For any gas path, Λ is about ½·Δn/n ≲ 10⁻³. The certified
forward error is ≲ 10⁻⁶ whatever the profile shape, including the kinks of
`PiecewiseLinear` and even pressure steps. Gas runs therefore need no reference
solve and no per-family qualification. The certificate also does not depend on
the abrupt-step limit failing: it states the forward model's error at a step
(R/2 in amplitude) rather than forbidding the step.

**Requested revision:**

- Admit a radial path per propagating column when its computed Λ (from the
  resolved material path, with $q=k_z(\kappa,\omega)$, so cutoff margins appear
  naturally) gives $\Lambda\tanh\Lambda$ and $\tanh^2\Lambda$ within budget.
- Give `AdiabaticAccuracy` a **default** budget, so ordinary gas runs keep their
  current signature; the maintainer prefers robust defaults to required tuning.
  Let callers tighten it.
- Admit index jumps under the same rule, evaluating the local-flux closure away
  from the jump plane, and refuse only on budget.
- Keep the independent scalar-wave solver as a **test witness** that verifies the
  certificate and the implementation, not as a per-family runtime qualification.
- The bound needs real q, so the weak varying-loss case keeps its separate
  qualification. That remains a genuine open item (Q1).
- Record the derivation and the table above in §4.5.

#### A2 — P2: Decide the native GPU state at activation now

**Sections:** 6.2, 12 (Q4).
**Maintainer decision (2026-10-23):** a short, bounded, paid device qualification
before activation (one NVIDIA and one AMD device, a few tens of calls, budget
stated in Part I); the spend itself still needs the maintainer's go. No GPU
refusal and no unqualified activation.

The native backend's CPU and GPU builds share source. An atomic CPU activation
therefore changes GPU spatial semantics in the same commit. Q4 admits an "accepted,
explicit temporary refusal". That would be a new regression for GPU spatial users,
created by a CPU-qualified change. GPU qualification also costs real money under
the standing device policy.

**Recommendation, for maintainer decision:** either include a short, bounded
device qualification in phase 2, before activation (a few tens of calls on one
NVIDIA and one AMD device, with a stated budget), or activate with the GPU build
enabled and listed as *unqualified* in the capability matrix until the next
device campaign. Prefer either to refusal, and settle the choice in Part I rather
than at activation.

#### A3 — P2: Enumerate which statistics need G

**Sections:** 4.2, 5, 6.1, 7.
**Maintainer decision (2026-10-23):** in-run local statistics use the exact
signed local flux, with G synthesised in-run, as the robust default; the extra
cost is measured and reported, not avoided by a cheaper carrier-admittance
substitute.

Energy and spectra need only the real weights. They cost one multiply per column.
Local fluence, peak intensity, on-axis traces and widths need G synthesis: extra
inverse transverse and temporal transforms per evaluation. That could roughly
double in-run statistics cost on every saved step. Peak-intensity statistics are
also what users read against ionisation thresholds.

**Requested revision:** list each existing spatial statistic and accessor as
weight-only or needs-G. For each needs-G statistic, decide whether it becomes
signed local flux (G synthesised in-run, cost measured), an offline/saved-plane
accessor only, or a named field-intensity diagnostic at a declared admittance.
For a transparent paraxial beam the last is $\tfrac12n_0\varepsilon_0c|E_a|^2$.
Without this list, §7's "common in-run/offline definitions" leaves the cost and
meaning of the most-used local statistic open.

#### A4 — P2: State which tutorials hold energy and which hold field

**Section:** 8.

§8 correctly says each example must choose whether physical energy or physical
field is held fixed. One consequence is worth recording. At fixed field, $P/P_{\rm cr}$,
the B-integral and all nonlinear dynamics are unchanged, because both powers scale
by $n_0$. At fixed energy, the field drops by $1/\sqrt{n_0}$: a negligible ≈ 4×10⁻⁴
in gas, about 18 % in glass. List the affected gallery items with their choice, so
the regeneration is reviewable and no gas example's figures move for a 10⁻⁴ reason.

#### A5 — P3: State the internal sign in Φ

**Section:** 4.4.

Φ is written with $-i\int\operatorname{Re}k_z$, which is the internal transform
sign rather than the optics convention used elsewhere in §4. Point to the
conventions section that defines the internal sign, so the logarithmic term's real
sign and the phase's imaginary sign can be checked against the code in one place.

#### A6 — P3: Interface spelling

**Section:** 5.

`spatial_energy(field, flux)` drops `grid` and `space`, which `SpatialFlux`
binds. Good. Use the same `(field, flux)` order for every accessor in the table,
and say whether `SpatialFlux.resolve(..., z=)` on a gradient path re-resolves per
call or is bound once per saved plane. The latter matters for SEP 31's
`FieldPlane.from_output`.

#### A7 — P3: Small points

- §4.3: the synthetic finite constant-admittance endpoint test is a good idea. Note
  that it tests only the discrete identity: no optical meaning is attached to DC.
- §4.5: the bound and its exactness at a step make "can be very conservative"
  accurate for smooth ramps but misleading in general. After A1, say instead
  "attained by an abrupt step; conservative for smooth profiles".
- §13: Diamant and Fernández-Guasti appeared online in December 2013 in the 2014
  volume; the citation as written is correct.
