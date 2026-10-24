# SEP 34 — Medium flux for free-space propagation

| | |
|---|---|
| **SEP** | 34 |
| **Title** | Medium flux for free-space propagation |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 (draft and 2026-10-23 review response) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-10-23 |
| **Resolution** | Accepted 2026-10-23 by John Travers after the A1–A7 responses and maintainer decisions recorded in Part III. Implement before SEP 31's public launch changes. Final 2026-10-24 by John Travers after phase 3, with the remaining extensions moved to the roadmap (Part II, "Closing"). |
| **Requires** | [SEP 22](sep-0022-memory-lean-spatial-execution.md): spatial execution and memory; [SEP 25](sep-0025-rounding-level-reproducibility.md): qualification; [SEP 28](sep-0028-free-space-evaluation.md): bounded spatial evaluation |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | John Travers with Opus 5.5 (AI); phases 1 and 2 on `feature/sep-0034-medium-flux`, phase 3 on `feature/sep-0034-phase-3` |
| **Manual pages** | [Energy and flux in a medium](../../../guide/free-space/flux.md) |

## Part I — Proposal

### 1. Abstract

Give free-space launches, noise, energy, fluence and statistics one physical
power convention in the declared medium. Replace the vacuum weight with the
scalar forward equation's axial-admittance measure, using complex admittance
for local flux and its real part for total energy. Preserve the existing radial
gas-gradient capability by adding the forward amplitude correction to its
integrated linear operator, with a total-variation certificate for real-index
paths. Qualify NumPy, Numba CPU, native CPU and bounded CUDA/HIP cases before
activating the changed public semantics together.
This is a clean breaking change with no legacy normalisation or old-file
migration. It is the shared prerequisite for coupling in
[SEP 31](sep-0031-simulation-coupling.md) and solid carriers in
[SEP 5](../sep-0005-solid-ionisation.md), not an implementation of either feature.

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
| Radial gas pressure/density profiles | Admit real-index paths, including declared weak steps, through the per-column certificate; use the forward amplitude correction, local source prefactors and local-plane measurement |
| Cartesian and line varying media | Remain unsupported; this proposal does not add a propagation route |
| Hybrid planar guide | The explicit modal/FFT metric in §4.3; retain its scalar slab synthesis and correct its currently vacuum-normalised coefficients |
| Ordinary and general modal guides | Retain their own derived metrics; provide consistent dispatch, not a new normalisation |
| Uniform absorbing scalar media | Integrated energy and attenuation use real admittance; local traces use the complex forward admittance |
| Longitudinally varying absorbing media | Deferred: uniform absorbing media remain supported, but a nonconstant absorbing radial path refuses pending a separate complex-index certificate |
| Interfaces, high-NA vector fields, crystals | Outside scope; SEP 31 handles declared planar interfaces, SEP 8 handles vector crystals |

No new material data, nonlinear response, spatial mode solver or dependency is
introduced. General reflected/evanescent scattering remains outside the model.
A real-index radial path, including declared weak steps, is admitted by §4.5's
per-column certificate; small gas index contrast alone is not an admission test.
Local observables describe the retained forward component, not an incident-plus-
reflected Maxwell field. No general scattering solver is introduced.

### 3. Context

The [free-space model](../../../guide/free-space/model.md) states the present vacuum
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

[SEP 5](../sep-0005-solid-ionisation.md) originally sketched scalar flux.
[SEP 31](sep-0031-simulation-coupling.md)'s review separated this prerequisite
from coupling and explicitly chose physical launch energy and preservation of
radial gradients. [SEP 8](../sep-0008-quadratic-crystal-optics.md)'s anisotropic
vector E/H metric remains distinct; [SEP 9](../sep-0009-propagation-statistics.md)
consumes the shared spatial definition without moving its modal measurement
catalogue here. SEP 31 depends on this proposal, not conversely.

### 4. Theory

#### 4.1 What energy means

Use SI and the [optics convention](../../conventions.md),
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
Use synthetic finite constant-admittance endpoint tests only for this discrete
identity. They attach no optical meaning to DC and do not evaluate the optical
$1/\omega$ formula there.

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

**Hybrid planar metric.** The current `PlanarGrid` synthesis uses
$E(x,y,\omega)=\sum_p V_p(x)A_p(y,\omega)$ with
$(\varepsilon_0c/2)\int V_p^*V_q\,dx=\delta_{pq}$.
Its stored coefficients are vacuum-normalised, despite carrying power-amplitude
units; `planar_mode_energy` currently states that explicitly. For the present
orthogonal scalar slab family, translation supplies

$$
 q_{p\ell k}=\sqrt{[(\omega_k/c)n_{{\rm eff},p}]^2-k_{y,\ell}^2},
 \quad w_{p\ell k}=c\,\operatorname{Re}q_{p\ell k}/\omega_k,
 \quad
 U_p=\frac{\Delta t}{N_t}\frac{\Delta y}{N_y}
       \sum_{\ell,k}\gamma_k w_{p\ell k}|\hat A_{p\ell k}|^2.
$$

The $\varepsilon_0c/2$ factor is already in V, so it is not applied again.
For local measurements synthesize G with the same V and the complex factors
$\varepsilon_0c^2q_{p\ell k}/\omega_k$ applied to A before synthesis.
Orthogonality diagonalises the integrated energy, while local interference
between modes remains. At $k_y=0$ the weight is $\operatorname{Re}n_{{\rm eff},p}$;
it is not generally the bulk index. This closes the metric choice before
implementation. Qualification checks the discrete Gram identity, including
quadrature error. A future nonorthogonal family requires its own full metric,
not this diagonal formula. Ordinary saved true-power modal amplitudes elsewhere
keep their existing metric and must not be weighted a second time.

#### 4.4 Radial forward amplitude transport

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

In the internal FFT convention of
[conventions §2](../../conventions.md), the plane wave has phase
$\exp[i(\omega t-\beta z)]$. In that convention and the existing moving frame,
the accumulated linear operator is

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

**Which field is retained.** Use the forward component in the exact
directional decomposition of the scalar wave equation, not the approximate
total field obtained by differentiating a truncated WKB ansatz. With
$E_\pm=q^{-1/2}a_\pm e^{\pm i\theta}$, $\theta' = q$, the full linear field
has $E=E_++E_-$ and $G=\varepsilon_0c^2q(E_+-E_-)/\omega$.
The retained component therefore has

$$
 G_+=\mathcal Y_c E_+ .
$$

Dropping backward coupling gives the amplitude equation above. Differentiating
this approximate E alone would instead add
$i\varepsilon_0c^2(\partial_z\ln Y_z)/(2\omega)$ to its local admittance.
That is a different approximate-total-field closure, not the directional
forward component controlled by the certificate below. It is removed from this
proposal by the maintainer's decision on 2026-10-23. Uniform absorbing media
still use the full complex $\mathcal Y_c$ of §4.2.

At a declared jump use the two one-sided values and the endpoint square-root
factor. Split steps, dense output and interpolation at that location; never
smooth a jump into a derivative spike. A saved plane at the jump records its
side explicitly (`before` or `after`); a request without a side is ambiguous and
refuses. Local traces use the corresponding side's forward closure and do not
pretend to describe the microscopic interface plane.

Nonconstant absorbing radial paths are deferred by maintainer decision.
Do not apply the real-q certificate to $\operatorname{Re}q$ while silently
ignoring its varying imaginary part. Uniform absorption remains supported;
varying loss needs a separately proposed complex-index bound and local model.

#### 4.5 Real-index path certificate and source work

For a propagating column let q be positive, real and of bounded variation,
constant outside the declared path. Its exact scalar linear scattering problem
has no incoming wave at the far end. Define

$$
 \Lambda=\tfrac12\operatorname{TV}_{[z_0,z_1]}(\ln q)
 =\tfrac12\int_{\rm smooth}|q'/q|\,dz
  +\tfrac12\sum_{\rm jumps}|\ln(q_+/q_-)| .
$$

Write $E=q^{-1/2}(a e^{i\theta}+b e^{-i\theta})$ and
$E'=iq^{1/2}(a e^{i\theta}-b e^{-i\theta})$. Substitution gives
$a'=(q'/2q)b e^{-2i\theta}$ and
$b'=(q'/2q)a e^{2i\theta}$, preserving $|a|^2-|b|^2$.
For unit transmitted flux and $b(z_1)=0$, the magnitudes are $\cosh\rho$
and $\sinh\rho$, with $|\rho'|\le|q'/(2q)|$.
Across a jump, continuity of E and E' gives a hyperbolic transfer with parameter
$\tfrac12\ln(q_+/q_-)$, so the same variation bound includes jumps.
Consequently [3],

$$
 R\le R_*=\tanh^2\Lambda,\qquad
 |\log(a(z)/a(z_0))|\le B_*=\Lambda\tanh\Lambda,\qquad
 \delta_*=\operatorname{expm1}(B_*).
$$

The continuous complex logarithm is used. The second inequality follows by
integrating $|a'/a|\le|q'/(2q)|\tanh\Lambda$; exponentiation bounds the
relative complex forward-field error by $\delta_*$. At a passive transmitted
endpoint, the logarithm of the transmission ratio has nonpositive real part,
so $|e^w-1|\le|w|$ and the review's tighter endpoint bound $B_*$ is also valid.
Use $\delta_*$ consistently for admission and interior forward coordinates,
avoiding two meanings of `field_error`. Evaluate small quantities stably.

For a monotone path $\Lambda=\tfrac12|\ln(q_1/q_0)|$ and
$R_*=[(q_1-q_0)/(q_1+q_0)]^2$, attained by the abrupt step and conservative
for smooth profiles with those endpoints. The forward approximation still
transmits unit flux; the certificate bounds that error instead of claiming
Fresnel transmission. Nonmonotone profiles use total variation, not endpoint
contrast: many small oscillations can fail. Near cutoff, q varies much more
than the bulk index, so even a gas path can fail.

**Admission.** `AdiabaticAccuracy(field_error=1e-4,
reflected_energy=1e-4)` is the default, chosen by the maintainer as a model
budget, not a physical constant or an integrator tolerance. Both values are
finite and positive; callers may tighten them. Every declared active
propagating column must satisfy $\delta_*\le\epsilon_{\rm field}$ and
$R_*\le\epsilon_{\rm reflection}$ over the entire run path. Do not reset the
budget per step or discard a failing column because the initial pulse is weak
there: nonlinear evolution can populate it. Existing nonpropagating support
rules remain explicit; crossing q = 0 refuses the certified path.

Construct a conservative upper enclosure of total variation for the resolved
material representation. Split at declared knots/jumps and isolate extrema of
the represented $\ln q$ (or bound its derivative on each interval); include
root-isolation and arithmetic error. For a certified monotone interval only
its endpoint log ratio is needed. The material interpolation error remains a
separate error envelope: a proof for an interpolant is not proof for an arbitrary
unresolved callable. A custom profile must supply bounded segments/variation
metadata or be resolved under an independently justified interpolation envelope;
sampling its derivative at finitely many points is not a certificate.
The admission record carries the represented path identity, variation enclosure,
interpolation envelope, per-column bounds, support and reference extent.
The exact metadata layout and polynomial enclosure algorithm are Phase 1
implementation choices under this fixed contract, not another physics decision.

The following values from review A1 are **review witnesses, not new qualification
results**. Its forward error means $|t_{\rm exact}/t_{\rm WKB}-1|$ at the exit.
The original review is preserved in Part III; its profile widths and rounded
values must become fully specified, refined tests before implementation claims
them as evidence.

| Review profile at 800 nm | Reported R | Reported R bound | Reported forward error | Reported B bound |
|---|---|---|---|---|
| Gas tanh ramp, Δn = 8 × 10⁻⁴, 1 mm | 2 × 10⁻²⁴ | 1.5 × 10⁻⁷ | 7 × 10⁻¹² | 1.5 × 10⁻⁷ |
| Same contrast, abrupt step | 1.60 × 10⁻⁷ | 1.60 × 10⁻⁷ | 8.0 × 10⁻⁸ | 1.6 × 10⁻⁷ |
| Nonmonotone Gaussian gas bump, 0.2 mm | 6 × 10⁻²⁵ | 6.4 × 10⁻⁷ | 6 × 10⁻¹¹ | 6.4 × 10⁻⁷ |
| n = 1 → 1.5 tanh ramp, 0.5 μm | 7 × 10⁻¹⁰ | 0.040 | 1.4 × 10⁻³ | 0.040 |
| n = 1 → 1.5 abrupt step | 0.0400 | 0.0400 | 0.020 | 0.041 |

Energy conservation to 10⁻¹¹ alone does not establish reflected powers of
10⁻²⁴; tiny reflected amplitudes need their own absolute error bounds.
Independent scalar-wave solves are test witnesses, never a runtime requirement
for each real-index profile family. No full-wave result is inferred from the
table's printed precision.

**Scope of the bound.** It certifies the retained forward part of the *linear
background scattering problem*. The omitted backward amplitude can be
$O(\sqrt{R_*})$, so neither total local E/H nor standing-wave maxima have
the forward-field error bound. It does not certify nonlinear feedback,
ionisation or arbitrary downstream sensitivity. Nonlinear runs retain the
repository's separate accuracy/refinement requirements. A plane records this
scope explicitly; a SEP 31 report must not relabel it a bound on the whole
nonlinear simulation.

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
| `SpatialFlux.resolve(grid, space, medium, z=..., side=...)` | Bind one physical plane, its support and complex/real admittance or the hybrid metric; side is required at a jump |
| `spatial_energy(field, flux)` | Integrated physical energy on the declared plane; no vacuum default when the medium is omitted |
| `spatial_fluence(field, flux)` and local intensity/axis accessors | Reconstruct E and G with the same complex closure; signed quantities remain signed |
| `SpatialPulse.build(..., medium=..., reference=...)` | Normalise requested energy or peak axial power at the declared launch plane, after geometric preparation |
| `spatial_noise(..., flux=...)` | Normalise the existing draw in the same entrance metric |
| `AdiabaticAccuracy(field_error=1e-4, reflected_energy=1e-4)` | Default forward-field and reflected-energy budgets; callers may tighten them; exceeding either refuses |
| `critical_power(..., index=...)` | Physical paraxial power at the specified real carrier index; not a broadband collapse prediction |

All field accessors use `(field, flux, ...)` in that order, including
`spatial_intensity`, `on_axis_intensity`, `on_axis_spectrum`,
`beam_radius`, `fwhm_r` and the three `planar_*` accessors.
The flux record binds grid, geometry, medium, z and jump side once; evaluating
it does not re-resolve the medium. A new plane needs a new record, with immutable
material/path data shared where possible. This is the record SEP 31 extraction
uses. Existing axis conventions fix the array contract; Phase 1 implements and
tests these signatures. The required medium/reference is an intentional
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

#### 5.1 Statistic and accessor decisions

The following inventory covers the existing spatial catalogue and the public
accessors in `solprop.spatial`. It is a contract for both in-run and saved-plane
evaluation. “G in frequency” needs coherent transverse synthesis but can use
temporal Parseval; it need not perform a temporal inverse transform.
“G in time” requires the same analytic temporal reconstruction as E.
“Exact” below means the signed flux of the declared scalar forward model.

| Existing name(s) | Evaluation and resulting meaning |
|---|---|
| `energy`, `spatial_energy`, `planar_mode_energy` | Real diagonal weights, or the derived modal quadratic form; no G synthesis |
| `beam_spectrum` | Real per-column axial-energy weights; its integral equals energy |
| `poynting_spectrum` | Remove this redundant spelling when `beam_spectrum` becomes axial flux; update callers and schemas, with no alias |
| `region_spectrum`, `aperture_spectrum` | Real axial weights with the existing angular membership/aperture quadrature; these remain declared angular-port measurements, not implicit facet transmission |
| `on_axis_spectrum`, `aperture_on_axis_spectrum` | Synthesize coherent E and G at the axis, applying the same aperture to both; form their same-frequency real pairing, not a squared coherent E sum |
| `spatial_fluence`, `planar_fluence` | G in frequency at physical nodes; signed time-integrated axial flux |
| `spatial_intensity`, `planar_intensity` | G in time; signed local axial-flux trace |
| `on_axis_intensity` | G in time at the axis; both accessor and saved statistic return its temporal maximum |
| `peak_intensity` | G in time at physical nodes; maximum signed axial flux over time and the transverse rule, including between-plane evaluations |
| `beam_radius`, `fwhm_r`; `radial_fluence_widths`, `cartesian_fluence_widths` | Widths of the same signed fluence; pure width helpers take an already computed profile, not a flux record |
| `absorbed_power` | Real metric weights for diagonal material, spectral, transverse-wavevector and evanescent damping channels, in the existing channel order; units remain J/m. Nonpropagating field-norm damping is not physical absorbed energy |
| Spatial collar diagnostic | Retain an explicitly named electric-field-norm boundary diagnostic; remove its claim to be a fraction of physical fluence. A physical masked-flux decrement requires its separate quadratic-form calculation |
| Density, pressure and other scalar profile statistics | Unchanged profile evaluation; neither flux weights nor G |
| Electron densities, species inventories, ionisation rates and plasma diagnostics | Retain physical-E rate/source definitions; flux is not substituted for E. Work diagnostics use the matching source-work identity |
| Raw fields/spectra and phase accessors | Preserve field values and units; no flux weighting of an amplitude merely because it is displayed |

Local-flux observables remain available in-run as requested, with no
carrier-admittance substitute and no automatic move to offline-only access.
Cache/reuse the E/G work shared by requested statistics and bound its chunks.
For a single axis, contract spectral columns directly instead of reconstructing
the full plane. Report added transforms, memory and timing by statistic group.

No negative flux is clipped. A positive-profile width refuses a negative
fluence profile outside its accounted arithmetic error, rather than changing
its measure. Existing undefined-shape cases (a hollow centre or no half-maximum
crossing) retain their documented undefined-width result; they are distinct
from a signed-profile refusal. The explicit field-norm collar diagnostic is
not a replacement intensity statistic and must be labelled as such.

The aperture coordinates retain their existing declared Fourier-plane mapping.
A corrected in-medium energy measure supplies no Fresnel transmission through
a real exit face; that requires SEP 31's explicit interface. The detector
measurement and its plane/medium are recorded so no energy change is invented.

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

The shared native source changes GPU semantics in the activation commit too.
The maintainer requires bounded qualification on one NVIDIA CUDA device and
one AMD HIP device **before activation**: no new GPU refusal and no activation
labelled unqualified. Prepare inputs, reference outputs and builds on CPU first.
On each device run at most 40 deterministic qualification invocations, each
with at most ten propagation steps; batch related assertions into those calls.
Cover every changed device path: real weights, complex local G, field/envelope
normalisation and noise, radial path endpoints/jumps, planar metrics, absorption,
saved/in-run statistics, and bounded memory. Record hardware/software versions,
error bounds and actual timings. This focused campaign supplements the native
gate; it does not rerun a full experimental gallery on rented hardware.

The total planning cap is **US$20**, including setup/build time, compute,
storage and teardown, selected by the maintainer on 2026-10-23. No resource is
provisioned by this draft, and the cap is not spending authorisation. At
implementation request the maintainer's explicit go with then-current rates
and a plan fitting the cap; stop before exceeding it. A blocked campaign
delays public activation rather than weakening its contract.
There is no hidden host statistics fallback. CPU and GPU evidence are reported
separately even though both are required for this activation.

Any changed seam records, protected driver hooks or stage bundles bump
`SCHEMA_VERSION` and its literal contracts, with native consumers updated
together. Output metric and plane-reference semantics require an output-schema
bump. Version numbers are assigned in implementation, not preselected here.
No dependency is added.

### 7. Verification and validation

Use the repository's [accuracy](../../accuracy.md) and [testing](../../testing.md)
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
| Gradient certificate | Independent scalar-wave test witnesses for monotone paths, bumps, repeated oscillations, kinks, declared steps and cutoff approaches; verify the TV enclosure and forward/reflection bounds separately from interpolation error, without runtime per-family solves |
| Sharpening profile | A weak monotone step can stay admitted with its bounded missing reflection; a large-TV path refuses. The exact reference approaches Fresnel scattering; the forward approximation never claims to do so |
| Nonlinear work | Local polarisation/current prefactor consistency; matched uniform limit and short gradient Kerr/plasma cases; physical E remains the rate drive |
| Absorption | Uniform loss exponential; explicit refusal of nonconstant absorbing paths; separate physical and numerical-absorber accounting |
| Signed/local statistics | No clipping; explicit refusal for undefined positive-profile measurements; common in-run/offline definitions and correct selected-plane medium |
| Other representations | Hybrid planar reconstruction/metric identity and modal dispatch; no accidental bulk-index multiplication of true-power coefficients |
| Backend and resources | NumPy/Numba/native CPU parity, the bounded CUDA/HIP campaign before activation, source/target host exchange, current-schema round trips, memory ledger and short benchmarks |

Existing numerical absorbers and nonpropagating-support rules need regression
coverage. Show the effect of finite transverse/time windows independently.
For nonlinear runs, work balance and propagation accuracy have their own
envelopes; they are not expected to obey a pure-linear rounding invariant.
Benchmark construction, one statistic, one source evaluation and about ten
steps. Never extrapolate these timings to full experiments.

### 8. Documentation

Add `docs/guide/free-space/flux.md` with the meter-energy definition, phase versus
group index, complex local admittance, analytic-signal conventions, radial WKB
derivation, the TV certificate and its scope, explicit jump sides,
validity/refusal rules and a short rising/falling gas-profile example.
Link it from the free-space model, power/intensity, noise, input-pulse, radial
and coupling guides. Update docstrings and all affected callers together.

The maintainer chose the following migration on 2026-10-23. Preserve the
physical electric field in lessons teaching nonlinear dynamics; retain physical
energy when that is the specified teaching input. These are choices of example
controls, not a legacy-normalisation switch.

| Source under `docs/` | Quantity held fixed and required update |
|---|---|
| `introduction/tutorials/plot_03_filament.py` | Physical incident E; retain the intended Kerr/plasma dynamics, report corrected energy and physical critical-power ratio |
| `introduction/tutorials/plot_04_crossing_beams.py` | Physical energy of each specified beam; recompute the corrected normalisation |
| `tutorials/free_space/plot_01_radial.py` | Weak diffraction case: physical incident energy. Nonlinear focusing case: physical E; update critical-power labels |
| `tutorials/free_space/plot_02_cartesian.py` | Specified physical incident energy; update the launch and fluence measurement |
| `tutorials/free_space/plot_03_line.py` | Physical E of the soliton and half-power comparison; derive the physical per-unit-width power from that field |
| `tutorials/free_space/plot_04_planar_hybrid.py` | Specified physical incident energy; use the explicit slab metric and preserve captured-share semantics |
| `tutorials/free_space/plot_05_low_storage.py` | Physical E for matched integrator comparisons; update the soliton-power label and spectral energy fraction |
| `tutorials/free_space/plot_06_four_wave_mixing_scan.py` | Physical E of each pump; report corrected energies and flux-based region/absorption measurements |
| `tutorials/free_space/plot_07_far_field_statistics.py` | Physical E of the nonlinear pumps; replace the duplicated beam/Poynting spectrum comparison and report corrected aperture and energy measurements |
| SEP 31's new `plot_08_radial_capillary_coupling.py` | New physical sizing at the agreed B = 0.3 rad target; no old field or energy to preserve |

Construct retained fields from their stated physical analytic amplitudes and
geometries, then measure them with the new metric; do not preserve an old
vacuum-normalisation helper in an example. At fixed E in a uniform medium the
propagation equations do not change. The narrowband paraxial power and critical
power both acquire n0; finite-angle/bandwidth ratios and all plotted flux traces
must nevertheless be recomputed, not assumed identical.

The current `docs/examples/` sources contain no free-space spatial launch in this
inventory; their ordinary modal powers are unchanged. Recheck the inventory at
implementation. Any worked-example source edit still requires regeneration.
Gallery inputs/output labels and ordinary guide snippets must be updated
together; run the normal galleries through Sphinx. SEP 31 owns its new coupling
tutorial and its propagated convergence evidence.

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

Real-index radial pressure profiles continue to work when their per-column TV
certificate meets the default or tightened budgets; declared weak jumps can
qualify too. Exceeded budgets, turning points, unresolved variation and
nonconstant absorbing paths refuse explicitly. Cartesian/line gradient restrictions
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
| Refuse every gradient or index jump | Removes useful gas paths; apply the per-column TV certificate, including declared jumps |
| Require a full-wave reference solve per profile family | Unnecessary for the real-q bound; retain independently refined solves as test witnesses |
| Use the differentiated WKB total-field closure locally | Inconsistent with interpreting the certificate as controlling the retained directional field; use G = 𝒴c E for that component |
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
| **1 — Shared reference and admission** | Private physical/complex-flux reference, directional endpoint closure, TV admission/enclosures, explicit slab metric, independent scalar-wave witnesses and the specified interfaces | Discrete identities, step and oscillatory-path witnesses, default/tightened budgets, complete accessor inventory and unchanged existing public behaviour |
| **2 — Atomic activation** | NumPy first, then Numba and native CPU; update launch/noise/statistics, gradient flow, physical power references, host processing, schemas and manual; qualify CUDA/HIP before the public switch | Directly affected modules, three-CPU parity, native gate, the approved bounded device campaign, memory/timing evidence and no backend support gap |
| **3 — Qualification and completion** | Integrated gradient/absorber/reference tests, updated galleries, final capability matrix and handoff to SEPs 5/31 | Full gate, slow/coverage lanes once under the standing policy, documentation and recorded source/limitation reports |

During development run fast checks and directly affected tests; after a failure
rerun failing modules, not whole lanes without cause. Changes in `native/` follow
its instructions and native gate. CPU and GPU evidence is recorded separately.
This draft requires only whitespace, changed-file spelling, SEP consistency
tests and Sphinx checks. No propagation implementation is claimed.

### 12. Open questions

No unresolved model choice is intentionally left to implementation. Acceptance
settles the following contracts; evidence that they are implemented correctly
belongs to the named phases.

| Former question | Decision and remaining evidence |
|---|---|
| **Q1 — Path admission** | Real positive q with the TV certificate, default 10⁻⁴ budgets and explicit jump sides. Phase 1 implements enclosures and independent witnesses. Nonconstant absorption is deferred to a separate proposal |
| **Q2 — Local trace closure** | G = 𝒴c E for the retained forward component; full complex admittance in uniform absorbing media. No gradient derivative term and no claim to recover omitted backward interference. Phase 1 tests signs and cross terms |
| **Q3 — Hybrid planar** | The diagonal metric and coherent local synthesis of §4.3. Phase 1 verifies the actual discrete Gram/Parseval identity |
| **Q4 — GPU activation** | Bounded CUDA and HIP qualification before activation, US$20 planning cap, actual spend requiring a separate go. Phase 2 records results and capability matrices |
| **Q5 — Interfaces and resources** | Field-first accessors bound to an immutable plane, shared in-run/offline semantics and the §5.1 inventory. Phase 1 fixes array record layouts and numerical enclosure details under those contracts; Phase 2 measures cost |

Physical incident energy, the clean breaking change, exact signed model flux
in-run, the explicit gallery control choices and SEP 34 activation before
SEP 31 implementation are settled. Acceptance does not claim completed tests,
device qualification or measured resource costs. If qualification requires a
different model or scope, return for a proposal revision; do not change the
accepted contract silently.

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

3. M. Visser, “Some general bounds for 1-D scattering,” *Physical Review A*
   **59**, 427–438 (1999).
   [doi:10.1103/PhysRevA.59.427](https://doi.org/10.1103/PhysRevA.59.427);
   [author manuscript](https://arxiv.org/abs/quant-ph/9901030).
   Equations (80)–(82) give the monotone step bound.

**Source audit.** The current spatial model, launch/noise functions, radial
operator construction, conventions and critical-power documentation were
inspected. The publisher record and abstract of [1] were checked for its
unidirectional scope, not its complete derivation. The introduction and
derivative-discontinuity discussion of [2] were inspected for the distinction
between adiabatic and reflected-wave models; no coefficient from it is copied.
The real-q monotone reflection bound was checked against [3], including its
step limit. The forward-coordinate and local-closure derivations above are
given explicitly rather than inferred from reflection alone.
The flux, endpoint, directional and work identities use the stated scalar
equations. A static Fresnel counterexample and a two-tone complex
admittance identity were evaluated. The latter confirms equal integrated energy
but different local traces when imaginary admittance is discarded, with a
rounding-sized residual against its analytic cross term. No new gradient
propagation or backend implementation was run or qualified.
Material constants remain supplied and referenced by optmat. The existing
critical-power coefficient is reused through its current cited derivation,
not revalidated to additional precision by this draft.

## Part II — Implementation record

### Phases 1 and 2 report — reference, admission and CPU activation

- **Scope:** phases 1 and 2 (§11) on CPU, 2026-10-23, Opus 5.5 (AI), one branch
  `feature/sep-0034-medium-flux`, developed in a worktree with parallel slices merged
  into it (the certificate and witness, the radial amplitude term, the compiled
  backends, the test migration and the documentation). The maintainer asked for
  targeted tests per slice, one full gate at the end and no backwards compatibility.
  The CUDA/HIP qualification of §6.2 followed with the maintainer's go (its own
  report below).
- **Delivered:**
  - `solprop.spatial.SpatialFlux` (`resolve(grid, space, medium, z=, side=)`), the
    field-first accessors of §5 and §5.1, `spatial_power` (power through the plane in
    time) and `peak_field` (the named physical-E diagnostic that binds ionisation);
    `critical_power(..., index=)`, physical (n0 × the former value).
  - Launches (`SpatialPulse`, `PlanarPulse`) normalise physical energy or peak axial
    power at the launch plane after pre-propagation, each `PulseSum` member separately;
    the planar incident beam's admittance is averaged over k_x by Gauss–Hermite.
    `spatial_noise(flux, ...)` draws one photon per admitted bin.
  - `solprop.adiabatic`: `AdiabaticAccuracy`, the per-column TV certificate
    (`certify_radial_path`, `certify_wave_path`), its admission record, and the
    profile variation and declared-jump metadata (`ZProfile.variation`, `jumps`);
    `SimulationSettings.adiabatic_accuracy`. The independent witness is
    `tests/oracles/scalar_wave.py`.
  - The radial endpoint term −½ ln(Y_z(z)/Y_z(0)) in Φ on the table, expansion and
    reuse paths (`translate/linop.py::endpoint_phi`, `translate/radial.py`).
  - Kernel schema 35: `KernelProblem.flux` (`translate.spatial.FluxPlan`), with the
    material z table of a varying radial medium and its admission. numpy_ref,
    numba_cpu and native CPU compute every §5.1 statistic from it;
    `poynting_spectrum` is removed. Output schema 23 records `flux_model` and
    `flux_plan`; processing rebuilds a saved plane's admittance at its position.
  - The collar diagnostic is named a field-norm fraction (`collar_field_fraction`).
  - Manual: `docs/guide/free-space/flux.md` and the pages and galleries of §8, the
    design pages, the changelog.
- **Deviations:**
  - `SpatialPulse.build` takes `medium=, z=, side=` rather than `reference=`: the
    flux record binds the same plane.
  - A save plane exactly on a declared jump is refused at translation instead of
    recording a side; a `SpatialFlux` there requires `side=`. In-run statistics at a
    jump evaluate the "after" side, as Φ does.
  - Species absorption on spatial runs keeps the vacuum weight: per column
    2 Im k_z · Y_z = ε0c(ω/c)ρ Im γ/2 exactly, so the admittance cancels.
  - A launch with more than rounding-level field norm in nonpropagating columns is
    refused rather than masked; the inactive band and DC remain hard masks.
- **Decisions:** none beyond Part I.
- **Verification so far** (targeted; the full gate, slow and coverage lanes follow
  once, below):
  - Discrete identities: energy = area integral of fluence = time integral of
    power to 1e-12; the complex cross term against an optics-convention E/H
    reference to 1e-12 (and an independent review's three-wave check to 4e-14).
  - Launch energy and peak power to rounding; one photon per admitted bin.
  - Radial argon ramp and a declared 1→3 bar step: energy conserved to 1e-11 at
    every plane on all three CPU backends; in-run equals the saved-plane measure to
    rounding; the step's R* equals the Fresnel reflectance.
  - Certificate against the witness (A1 table made explicit): the bounds hold, and
    the step attains R* = tanh²Λ. Every existing radial gradient case is admitted
    (Λ ≤ 4e-4).
  - Three-backend parity of every statistic on radial lossy, radial gradient,
    Cartesian, line and planar cases: worst 2.1e-15 against budgets of 6e-13 to
    6e-12. Native gate (`poe check-native`) passed on the compiled slice.
- **Cost:** a quick A/B on the Mac (contended; same-process medians, 64×64 Cartesian,
  n_t 256): energy and spectra on native 0.7 → 1.4 ms, on numba 2.4 → 1.5 ms; local
  statistics about 1.8× on native and 2× on numba (the closure synthesis).
- **Limitations:** the scalar forward model's scope (§2, §4.5); nonconstant absorbing
  paths refused; numba forms the local closure volume with NumPy FFTs on the host.
- **Open questions:** Q1 — closed by the device qualification report below. Q2 — a
  compiled closure synthesis for numba's local statistics, owner the implementer, not
  required for activation.
- **Handoff:** SEP 31's `FieldPlane` consumes `SpatialFlux`; SEP 5 uses the same
  measure.

### Device qualification report — CUDA and HIP

- **Scope:** §6.2's bounded device campaign, 2026-10-23, Opus 5.5 (AI), on
  `feature/sep-0034-medium-flux`, with the maintainer's go (an NVIDIA H100 or smaller at
  most US$7, an AMD device, SEP cap US$20), both rented. One AMD MI325X (gfx942,
  ROCm 7.14, AMD Clang 23, rocFFT and rocBLAS) and one NVIDIA H100 NVL (sm_90,
  CUDA 13.0.88, GCC 14.2, cuFFT and cuBLAS). Inputs and references were prepared on
  CPU first.
- **Cases** (20 invocations per device, none longer than 8 steps): five zero-step
  measurements of prepared states and seven fixed schedules, each compared with
  numpy_ref references computed on CPU. Together they cover real weights and the
  complex closure, the far-field pairings and absorbed power, widths, the radial
  material table, gradient endpoints, a declared jump, Kerr source prefactors, a
  field-grid launch with input noise, a Cartesian and a line far field with a
  transverse collar, and the planar metric with Kerr. The other invocations are the
  native package's own suite on the device build and the native cases of
  `tests/backends/test_flux_parity.py` and `test_flux_paths.py`.
- **Bound:** at most 1e-10 relative deviation per statistic, declared before the
  runs; native CPU meets it to 2.1e-15.
- **Results:**

  | device | 86 statistics, worst relative deviation | flux parity at the CPU rounding budgets | native suite | device memory high-water |
  |---|---|---|---|---|
  | MI325X | 9.8e-14 | 8 passed | 354 passed, 23 skipped, 1 failed* | 3.0 GB |
  | H100 NVL | 9.8e-14 | 8 passed | 355 passed, 23 skipped | 1.3 GB |

  \* The version-equality check failed on a tree patched in place; the clean CUDA
  build passes it.
- **Defects found and fixed** (all present on main before this SEP): nvcc rejected a
  typename-free dependent return type (`chebyshev.hpp`); HIP Clang's `-Werror`
  refused unused host-only helpers (`general.cpp`); GCC 13 as nvcc's host reported a
  false maybe-uninitialized (`z_table.hpp`). The CUDA build now requires GCC 14 as the
  host compiler and CUDA 12.8 or newer, the first toolkit admitting GCC 14; on the NVIDIA device,
  GCC 13 and a CUDA 12.6 `nvcc` are refused and GCC 14 with CUDA 13 configures.
- **Spend:** US$2.94 of the US$20 cap (NVIDIA US$2.26 of 7, including a first build on
  CUDA 12.8 with GCC 13 that the defects stopped; AMD US$0.68). Every rental
  was destroyed by its own id. The campaign records are in the private development
  archive.
- **Not covered:** timing (§6.2 asks for actual timings only of the qualification
  calls, recorded there), older CUDA architectures, and multi-device runs.

### Phase 3 report — qualification and completion

- **Scope:** phase 3 (§11), 2026-10-24, Opus 5.5 (AI), on `feature/sep-0034-phase-3`
  after phases 1 and 2 were merged. CPU only; no device spend.
- **Delivered:** `tests/backends/test_flux_gradients.py`, four integrated cases run on
  numpy_ref, numba_cpu and native CPU (§7, "Gradient linear flow", "Nonlinear work",
  "Absorption"):
  - **Plane exchange.** A beam carried up a 0.5→3 bar argon ramp and back down the
    reversed ramp returns each column's launched amplitude to within
    2(10⁻⁹ · max|ln(Y_z(L)/Y_z(0))| + γ₂₅₆), while one leg alone changes it by more than
    a hundred times that: the endpoint terms of the two legs cancel.
  - **Kerr through a gradient.** A 0.3 mJ beam with about 0.2 rad of on-axis nonlinear
    phase keeps its physical energy through the 0.5→3 bar ramp to 10⁻⁸, the
    controller's tolerance; a constant `Linear` path matches the uniform medium to γ₆₄.
  - **Plasma through a gradient.** Argon losses at 1, 2 and 3 bar are 1.9, 3.2 and
    4.3 %, concave in density. The 1→3 bar ramp loses 3.21 %, between the chord of the
    endpoint losses (3.10 %) and the loss at the mean pressure (3.23 %), as a loss
    driven by the physical E at the local density must.
  - **Uniform loss.** In a medium of index 1.5 + 2·10⁻⁵ i, the energy at each plane
    equals that of each column attenuated by exp(−Im k_z z), to 10⁻⁹, with about a
    quarter absorbed; the collar's field-norm fraction stays below 10⁻¹², so material
    loss and the numerical absorber are accounted for apart.
- **Already covered by phases 1 and 2:** a weak declared step admitted and a large-TV
  (glass) step refused (`tests/test_adiabatic.py`, `tests/backends/test_flux_paths.py`);
  resume through a gradient (`tests/backends/test_radial_media.py`). The galleries of §8
  are tutorials, which every documentation build runs; the phase 2 pull request's
  build ran them all. The capability pages (`docs/introduction/capabilities.md`, the
  backends page) were rechecked and need no change.
- **Deviations:** the plasma case checks the ordering that local-density driving
  implies rather than a work balance. The field's loss exceeds the ionisation energy
  (Ip times the electrons freed) by about 27 %. The excess is the kinetic energy the
  field gives the freed electrons, of the order of the ponderomotive energy (about
  6 eV at the peak, against argon's 15.8 eV), which no statistic reports.
- **Decisions:** none.
- **Verification:** the twelve cases above pass on the three CPU backends. No
  convergence is claimed. The plasma ordering holds at 16 fixed steps, and 32 steps
  move the ramp's loss by 6·10⁻⁴ of itself; at 8 steps it is 5 % off. The full gate,
  coverage, slow lanes and the wheels on four platforms ran once in CI on the phase 2
  pull request and passed; this pull request runs the test lanes again.
- **Open questions:** none new. Phase 2's Q2 moves to the roadmap (Closing).

### Closing — 2026-10-24

Phases 1–3 are merged and reported. Every free-space launch, noise draw and statistic
measures the medium's axial flux on numpy_ref, numba_cpu and native CPU, with parity
tests, and the CUDA and HIP builds were qualified before activation. The manual
documents it on [energy and flux in a medium](../../../guide/free-space/flux.md).
Moved to the [roadmap](../../roadmap.md):

- **A compiled closure synthesis for numba_cpu's local statistics** (phase 2, Q2):
  numba forms the closure volume G with NumPy FFTs on the host.
- **Free-space flux beyond the scalar forward model** (§2, §4.5): gradient scattering
  outside the certificate, a complex-index certificate for varying absorption, varying
  media on the Cartesian and line grids, and the vector E/H flux at high numerical
  aperture.
- **The hybrid planar route and the far-field pass on GPUs:** this SEP's device
  campaign ran one short planar Kerr case and Cartesian and line far fields on each
  device; the routes' full qualification against their CPU parity bounds remains.

Handoff: SEP 31's `FieldPlane` consumes `SpatialFlux`; SEPs 5, 8 and 9 use the same
measure.

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
- 2026-10-23: Revised Part I in response to A1–A7. Added the real-index path
  certificate, explicit statistic and gallery inventories, planar metric and
  activation budget. Recorded the maintainer's choices to defer varying
  absorption, use the forward-component local closure and cap device planning
  at US$20. These settle proposal contracts; no implementation or device
  qualification is claimed.
- 2026-10-23: Accepted by John Travers after the review responses, with an
  explicit request to commit and merge both SEPs 31 and 34. Part I is frozen.
  This proposal is implemented first. Qualification and the separate spending
  go remain phase requirements; the implementation record remains empty.
- 2026-10-24: Phases 1 and 2 merged with the device qualification; phase 3 report
  and Closing added. Final and archived.

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

### Response to review — 2026-10-23

The original review above is retained. Part I now records the following
resolutions, including the maintainer's decisions made during revision.
`Draft` and `Reviewed` remain distinct from formal acceptance.

- **A1 — Accepted with a precise scope and closure.** Section 4.5 derives the
  per-column total-variation certificate, including kinks and declared jumps,
  with default 10⁻⁴ forward-field and reflected-energy budgets. Independent
  wave solves remain test witnesses, not a runtime requirement for each family.
  The variation must be conservatively enclosed for the resolved path; sampling
  an arbitrary callable does not establish that bound. The review's
  $B_*=\Lambda\tanh\Lambda$ bounds the complex logarithmic error and, by
  passivity, the relative transmitted-field error at the exit. The proposal
  uses $\operatorname{expm1}(B_*)$ for a consistent relative error bound also
  covering interior forward coordinates. Three qualifications matter:
  - “Any gas path” is too broad: nonmonotone total variation and near-cutoff
    columns can exceed budget even at small bulk index contrast.
  - The certificate controls the forward component of the linear background
    problem. It does not bound an omitted backward field, its local interference,
    or nonlinear feedback. A small reflected-energy bound is not the same as
    a small total-local-field error.
  - The review's transfer-matrix numbers are preserved as review witnesses.
    Conservation to 10⁻¹¹ alone does not qualify reflected power near 10⁻²⁴;
    implementation must establish independent amplitude errors and refinement.
  The maintainer chose to defer nonconstant absorbing paths rather than leave
  the complex-index model open. Uniform absorbing media remain supported.
  The maintainer also chose `G = 𝒴c E` for the retained forward component
  (§4.4), removing the derivative term obtained by differentiating a truncated
  WKB total field. That earlier derivative is consistent with its different
  field interpretation; it is not the local closure controlled by this
  directional certificate. Declared jumps have explicit saved-plane sides.
- **A2 — Accepted; spending decision separated from design.** Section 6.2
  requires both CUDA and HIP qualification before the public switch, with at
  most 40 short deterministic invocations on each device. The maintainer chose
  a US$20 total planning cap including setup, compute, storage and teardown.
  Actual provisioning and spending still require an explicit go with current
  prices at implementation. There is no GPU refusal or unqualified activation.
- **A3 — Accepted; every existing measurement classified.** Section 5.1 lists
  real-weight, frequency-domain G and time-domain G evaluations, their signed
  meanings and the existing peak/width conventions. Local spectra are not
  generally diagonal-weight measurements; fluence can use temporal Parseval
  without an additional temporal inverse transform. Shared reconstruction and
  chunking keep costs explicit, to be measured in Phase 2. The redundant
  `poynting_spectrum` spelling is removed when `beam_spectrum` adopts axial
  flux. Ionisation still uses physical E, and the boundary-collar field-norm
  diagnostic is labelled separately. Section 4.3 also resolves the planar
  metric: the current slab coefficients are vacuum-normalised; the derived
  axial weights and coherent local synthesis replace an unspecified audit.
- **A4 — Accepted; maintainer control choices recorded.** Section 8 lists every
  affected current spatial gallery source. Nonlinear filament, soliton and
  four-wave-mixing lessons preserve their physical E; lessons prescribing
  incident energy preserve that physical energy. The mixed radial lesson
  specifies its two cases separately. At fixed E, uniform propagation is
  unchanged, but finite-angle/bandwidth flux ratios and plotted measurements
  must be recomputed rather than assuming a universal n0 rescaling.
- **A5 — Accepted.** Section 4.4 explicitly identifies the internal FFT sign and
  links the conventions that define it. The real endpoint amplitude term is
  unchanged by that conjugation.
- **A6 — Accepted.** Section 5 fixes field-first accessors and an immutable
  flux record resolved once per physical plane, including a side at a jump.
  A different plane binds a new record. SEP 31 extraction consumes that record.
- **A7 — Accepted.** Section 4.3 states that the synthetic endpoint case tests
  the discrete identity only and assigns no optical meaning to DC. Section 4.5
  distinguishes a monotone step attaining the reflectance bound from smooth
  profiles and nonmonotone total variation. The 2014 reference is retained.

The remaining work is implementation evidence: numerical enclosures and
independent references, backend parity, bounded device runs, resource costs and
gallery qualification. Part II remains empty; no propagation or device campaign
was run for this response.
