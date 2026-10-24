# SEP 31 — Coupling simulations and transporting ultrafast fields

| | |
|---|---|
| **SEP** | 31 |
| **Title** | Coupling simulations and transporting ultrafast fields |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 (draft and 2026-10-23 review response) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-22 |
| **Resolution** | Accepted 2026-10-23 by John Travers after the S1–S4 responses recorded in Part III. SEP 34's public activation precedes this proposal's incident-launch change. |
| **Requires** | [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md): plane storage and input replacement; [SEP 23](archive/sep-0023-far-field-aperture-spectra.md): aperture definitions; [SEP 25](archive/sep-0025-rounding-level-reproducibility.md): numerical qualification; [SEP 34](sep-0034-medium-flux.md): shared medium flux and radial adiabatic transport |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | John Travers with Opus 5.5 (AI); on `feature/sep-0031-simulation-coupling` |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Connect solprop simulations through a coherent field at a reference plane, spatial
synthesis and projection, linear optical transport, and a validated launch into
the next simulation. These four blocks cover coupling between hollow or solid
guides, incident Gaussian and arbitrary fields, nonlinear beam preparation before
a capillary, outcoupling to an observation plane, and compression between stages.
The incident field retains its absolute scale: light that is not captured is not
restored by normalisation. The separate [SEP 34](sep-0034-medium-flux.md) supplies
the shared medium-flux contract; coupling evaluates that measure on the host at
each reference plane. Qualification separates physical attenuation,
uncaptured forward power, representation error and model limitations, and compares
short complete workflows on all three CPU backends. A required focused tutorial
compares linear and mildly nonlinear radial focusing before coupling into a
capillary. This is a draft design, not an announcement of implemented interfaces.

### 2. Motivation and scope

A simulated pulse is often the input to another optical component. Carrying only
its spectrum or duration loses wavefront, modal coherence, polarisation and the
frequency dependence of its spatial profile. Carrying a bare array preserves more
information but leaves its basis, units, frame, grid and energy meaning implicit.
The intended users need to assemble experiments without rebuilding those
conversions in each script.

| Requested calculation | Composition proposed here |
|---|---|
| Capillary output into another capillary | Extract a modal plane; apply a direct overlap or outcouple, transport and recouple; launch the captured amplitudes |
| Nonlinear radial beam into a capillary | Run the gas gap with existing Kerr/plasma/THG responses; extract its entrance-plane field; project onto the capillary channels |
| Solid-core fibre into another fibre, or different hollow guides | Evaluate the source and target profiles at their actual end/entrance planes, with an explicit forward interface approximation and spectral overlap |
| Gaussian beam into a guide | Construct a normalised incident field, including curvature and polarisation, and use the same projection |
| Arbitrary analytic field into a guide | Sample the declared field and incident domain, establish its power measure, and project |
| Arbitrary Cartesian free-space beam into a guide | Reconstruct the complex field, remap to the projection rule and project without imposing radial symmetry |
| Capillary output at a distant free-space location | Synthesize the exit field, apply the declared outcoupling model and propagate to an observation plane |
| SPM, compression and another capillary stage | Compose ordinary simulations with spectral optics, focusing/apertures and coupling operations |

**Included:** existing transverse mode families with actual field profiles;
ordinary and general-modal outputs; scalar or transverse-polarisation radial and
Cartesian fields; Gaussian and nonseparable analytic/sampled inputs; alignment;
controlled temporal remapping; spectral compressors; thin focusing elements,
apertures and declared mirror/polarisation operations; homogeneous isotropic
linear gaps; physical incident-energy accounting; CPU qualification and provenance.
The uniform-in-y line route uses J/m and W/m throughout. It cannot be coupled to
a finite-energy two-dimensional beam without an explicitly supplied y profile.
The hybrid planar guide can participate through its actual modal/FFT synthesis;
it must not be mistaken for a grid of electric-field samples.

**Excluded:** a general electromagnetic junction solver with backward, radiation
and evanescent channels; feedback between stages or a cavity fixed-point solver;
new mode solvers; high-NA vector free-space propagation; anisotropic crystal
transport; resolved angular dispersion or spatial chirp inside prism/grating
compressors; nonlinear thin elements; automatic optical layout optimisation;
adaptive spatial meshes; and a new distributed workflow scheduler. The radial
adiabatic amplitude correction belongs to SEP 34, not to the host transport API.
Nonlinear gaps use the existing propagation routes. Field-grid Kerr/plasma/THG
is a qualification case; it is not the teaching tutorial's model.

The release has no backward-compatibility requirement for this change. Replace
inconsistent interfaces and defaults directly, update their callers, tests and
documentation, and add no legacy normalisation, deprecated aliases, compatibility
shims or old-file migration. A changed schema may refuse an older file clearly.

### 3. Context

The relevant implementation is already divided across the layers of
[architecture](../architecture.md):

- `solprop.driver.simulation.Simulation.with_input` accepts an array on an existing
  grid, in its existing basis, and shares translated tables. The
  [interior-launch workflow](../../guide/results/interior-launch.md) deliberately
  retains the parent's frame. It does not establish a transfer between unrelated
  guides or temporal grids.
- `solprop.pulses.SpatialPulse` builds Gaussian/super-Gaussian beams and can launch
  them before their waist; `PlanarPulse` and `modes.slab.gaussian_projection`
  provide a specialised planar projection. `DataPulse` interpolates a measured
  temporal spectrum and scales it to a requested energy; that scaling is unsuitable
  for transferring an already normalised simulation output.
- `solprop.translate.modal_tables.project_transverse_field` uses the general
  guide's coordinate dual. Its `ProjectedLaunch.residual` is an electric-field
  reconstruction residual in a quadrature norm. It is not a radiated-power
  calculation. `Simulation`'s `_projected_launch` subsequently rescales the
  projected shape to the pulse's full requested energy, even when capture is poor.
  The [general-modal manual](../../guide/waveguides/general-modal.md) documents this.
- `solprop.processing.modal` reconstructs E/H only against checked reference
  identities. Ordinary modal states instead use the true-power boundary
  $b=\sqrt{\operatorname{Re}n_{\rm eff}}a$. General-guide coordinates can carry
  a dense flux metric. A universal unweighted sum of squared amplitudes is wrong.
- `solprop.optics` already supplies `TransferFunction`, `TaylorPhase`, `Material`,
  `ChirpedMirror`, `GratingPair`, `PrismPair` and `Chain`.
  [Compression](../../guide/workflows/compression.md) and its tests cover their
  signs, loss conventions and parameter searches. This SEP reuses those models.
- `solprop.spatial` and the radial/Cartesian routes propagate physical electric
  fields, but their energy, intensity and fluence currently use vacuum admittance.
  [Free-space conventions](../../guide/free-space/model.md) explicitly identify
  that limitation. Existing far-field statistics are selected diagnostics, not a
  coherent output field at an arbitrary receiving plane.

Existing tests cover modal synthesis/recovery, the current projected-launch energy
semantics, optics/compressor models, spatial transforms and same-problem input
replacement. They do not qualify the cross-route compositions proposed here.
This assessment is a source/test audit, not a new test-suite result.

[SEP 34](sep-0034-medium-flux.md) owns spatial launch/noise/statistics consistency
and the adiabatic correction for the existing radial gas-gradient route. This SEP
and [SEP 5](sep-0005-solid-ionisation.md) consume its common measure; neither adds
another normalisation policy. [SEP 8](sep-0008-quadratic-crystal-optics.md) retains
its full-vector crystal E/H contract and interface solver.
[SEP 9](sep-0009-propagation-statistics.md) measures fields within a propagation
route; this SEP supplies transported fields without taking on its additional
measurement catalogue.
[SEP 21](sep-0021-absorbing-boundaries.md) owns wider production time windows and
escaped-field models; remapping a saved plane cannot recover already absorbed light.
[SEP 29](sep-0029-compiled-backend-distribution.md) has moved the compiled backend
into this repository as `native`, distributed as `solprop-native`. Qualification
uses its CPU build. This draft's historical review retains the names it reviewed.

### 4. Theory

#### 4.1 One physical plane, several representations

Use SI and the optics convention of [conventions](../conventions.md): positive
propagation phase $\exp(+i\beta L)$ and delay $\exp(+i\omega\tau)$. A plane has
an oriented local transverse basis, a forward normal and a physical origin. Its
temporal reference is independent of a simulation's local z origin. Reflection
changes local coordinates and polarisation consistently, not by an unexplained
conjugation of the field.

Let $\mathcal S_A(\omega)$ synthesize a source representation into a physical
field and $\mathcal P_B(\omega)$ recover the target coordinates. A declared
linear transport $\mathcal L$ gives

$$
 C(\omega)=\mathcal P_B(\omega)\mathcal L(\omega)\mathcal S_A(\omega),
 \qquad \mathbf b_B=C\mathbf b_A.
$$

This factorisation is a composition rule, not a claim that every combination of
ports and interfaces is physically valid. Each adapter declares its field model,
support, normalisation and approximation. Identity transfers keep their native
representation. A common quadrature can evaluate a direct overlap without storing
the intermediate field at every temporal frequency.

For an ordinary guide, synthesize the physical field with the existing
vacuum-normalised basis after dividing saved $b_m$ by
$\sqrt{\operatorname{Re}n_{{\rm eff},m}}$ at the selected plane. For a general
guide, use its E/H tables and metric at that plane. Use the source exit radius,
material and tracked gauge, not their entrance values. The target uses its own
entrance reference. A change of modal gauge changes the amplitudes and the maps
together and leaves the physical field unchanged.

#### 4.2 Shared plane measure

[SEP 34](sep-0034-medium-flux.md) derives and implements the scalar axial-admittance
measure, its analytic-signal endpoints, physical launch/noise scaling, statistics
and radial adiabatic transport. For a plane in an isotropic medium it supplies

$$
 Y_z(\boldsymbol\kappa,\omega;z)
 =\varepsilon_0c^2\frac{\operatorname{Re}k_z}{\omega},\qquad
 U=\frac12\int\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*)\,dA\,dt,
 \quad \widetilde{\mathbf G}_a
       =\mathcal Y_c\widetilde{\mathbf E}_a.
$$

Tildes here denote the transverse/frequency spectrum: apply admittance there,
then reconstruct G in space and time. Here $\operatorname{Re}\mathcal Y_c=Y_z$.
Local flux uses the full complex
admittance of the retained forward component, whereas total
energy uses its real part; replacing one by the other before reconstruction
would lose absorbing-medium interference terms.

The host plane evaluator and propagation statistics use the same definition and
resolved material at the selected z. Coupling does not require an extra kernel
statistic or a second flux implementation. Physical V/m remains the propagated
spatial state; ordinary and general modal ports retain their derived native
metrics. A planar hybrid port uses its modal/FFT metric, not a bulk index factor.

A saved plane from a radial path admitted by SEP 34 §4.5's per-column
total-variation certificate is a valid source. Its local admittance, jump side
where needed, path certificate and accumulated frame/amplitude references must
be available. Transport through that path remains an ordinary radial simulation;
the `FreeSpace` element below is homogeneous. Refuse a failed/missing certificate,
an ambiguous jump side or an unsupported absorbing path, not every gradient.
Record the certificate's scope: it bounds linear-background forward coordinates,
not the complete nonlinear simulation or an omitted reflected field. The scalar
metric is not a replacement for the full-vector Maxwell treatment of high-NA or
crystal fields.

On supported optical bands, `energy=` is the physical energy crossing the
declared plane, not a vacuum-equivalent field norm. For a lossless paraxial plane
wave its flux is $n_0\varepsilon_0c|E_a|^2/2$, using the phase index. SEP 34 explains
the distinction from energy density and the conversion of an external meter
reading through an explicit interface.

#### 4.3 Projection, power and an interface are different objects

For a lossless forward vector basis, define the Hermitian flux pairing

$$
 \langle\psi_m,\psi_n\rangle_F=
 \frac14\int(\mathbf e_n\times\mathbf h_m^*
             +\mathbf e_m^*\times\mathbf h_n)\cdot\hat{\mathbf z}\,dA,
 \qquad F_{mn}=\langle\psi_m,\psi_n\rangle_F.
$$

The coordinates of a represented field satisfy $\mathbf a=F^{-1}\mathbf B$,
where $B_m=\langle\psi_m,(\mathbf E,\mathbf H)\rangle_F$. The recovered
power is $\mathbf a^\dagger F\mathbf a$. The package's existing E-only shortcut
doubles one half of B for a forward superposition in its admitted lossless basis.
Its exact recovery on that subspace does **not** prove that an unrelated incident
beam has the target guide's magnetic field. Use incident H when available; an
E-only input must declare its forward closure and validity. Do not reuse a
nonlinear source-projection matrix to recover field coordinates.

For scalar fields in a common positive power space, let S contain target basis
columns and M the quadrature/power measure. Weighted least squares gives

$$
 G=S^\dagger M S,\qquad
 \mathbf a=G^{-1}S^\dagger M\mathbf f,\qquad
 \|\mathbf f\|_M^2=\mathbf a^\dagger G\mathbf a+
                 \|\mathbf f-S\mathbf a\|_M^2.
$$

The last equality follows because the residual is M-orthogonal to S. Only in
such a common positive space does it identify a nonnegative uncaptured share.
The existing electric-field residual and an arbitrary E/H flux projection do
not automatically satisfy this identity. A basis outside a common admitted power
space needs its declared interface approximation, not an assertion of orthogonal
energy loss.

Use F for the E/H flux matrix, G for the scalar least-squares Gram matrix,
and Q for a native representation's spectral power metric. They coincide only
when the synthesis and common measure establish that identity. In particular,
ordinary saved true-power coefficients have Q equal to the identity; a general
guide may have a dense Q. A profile overlap must not silently replace Q by G.

**Admitted models.** Make the following choices explicit in `CouplingModel`:

1. `same_basis`: exact coordinate recovery in a checked common modal reference,
   including a declared invertible basis/gauge change. Provider E/H and F recover
   coordinates; this is not an interface calculation.
2. `scalar_paraxial`: ordinary scalar capillary modes, incident
   Gaussian/analytic/sampled scalar or transverse-Jones beams, and scalar
   weak-guidance solid modes with physical profiles. Use the common local measure
   $M_0=Y_0(\omega)dA/2$, with $Y_0=n_{\rm ref}\varepsilon_0c$, on the whole
   incident plane; solve G and convert the resulting coordinates to the target's
   native Q. Different capillary radii and weakly guiding fibre cross-sections
   are admitted subject to the bounds below. The capillary core gas fixes
   $n_{\rm ref}$; its wall loss is propagated by the guide model, not inferred
   as facet transmission. Its scalar boundary approximation is reported.
3. `vector_paraxial`: provider E/H for lossless bound general modes, in the same
   weak-guidance/paraxial regime and a common transverse reference. Recover with
   F using both incident E and H. An E-only incident beam explicitly uses
   $\mathbf H_t=Y_0\hat{\mathbf z}\times\mathbf E_t$; this closure is qualified
   against plane-wave Maxwell fields and provider H. Longitudinal field
   reconstruction requires the provider or a separately declared closure;
   an $O(\theta^2)$ axial-power error is not an $O(\theta^2)$ bound on every
   omitted longitudinal field component.

All other physical junctions are refused: strong index contrast, unresolved facet
near fields, cutoff, unqualified lossy/nonorthogonal duals, arbitrary discontinuities
and a request for reflected or radiation fields. A bulk entrance/exit index
change uses a separate qualified planar interface before projection. Do not
substitute modal effective indices into its Fresnel coefficients.
Reference [1] explains why coordinate recovery alone is not boundary matching.

**Admission rule.** `CouplingModel` takes a required positive `model_error`
budget $\epsilon_m<1$; it is a dimensionless energy/axial-flux approximation
target, separate from numerical accuracy. No universal index contrast or angle
is called exact. For the two approximate models:

- choose $\theta_*$ by $1-\cos\theta_*=\epsilon_m/8$ and bound angular tails
  with §4.6's energy-weighted rule, for source and retained target subspace;
- require $\Delta=\max|n^2-n_{\rm ref}^2|/(2n_{\rm ref}^2)\le\epsilon_m/8$
  over the occupied band and dielectric regions of a weak-guidance solid
  provider; this criterion does not apply to a capillary wall as if it were a
  weakly guiding dielectric;
- bound the relative metric defect between G and target Q, and between the
  common source measure and its native Q, by $\epsilon_m/8$ each, using
  conditioned quadratic forms on the retained subspaces;
- for `vector_paraxial`, bound the transverse H-closure error in the same
  normalised operator norm by $\epsilon_m/8$; supplied incident H is retained.
  The bound concerns every admitted input in that port, not one lucky pulse.

The Δ screen is a conservative sufficient screen, not an immutable minimum
budget for every solid-fibre pair. A qualified provider-pair benchmark envelope
may replace **that screen** if it bounds the omitted vector/facet effects over
the declared wavelength, geometry, alignment and mode span to the requested
budget. Record its identity, range and independently bounded numerical error;
do not extrapolate it. Angular support, metric/closure checks, passivity and
the total budget still apply. Identical checked bases use `same_basis` directly
and need no weak-guidance screen for coordinate recovery.

These are conservative screening allocations, not a theorem that small contrast
alone bounds facet scattering. Each provider/model pair must additionally have
an independent plane-wave or mode-matching benchmark envelope for omitted
reflection and facet error. The sum of those errors, angular/metric/closure
defects and resolved tails must fit $\epsilon_m$. Unqualified combinations
remain refused. The independent benchmarks and budgets are deliverables, not
permission to choose a different model during implementation.

For a proposed passive finite map C between native metrics $Q_A,Q_B$, check

$$
 C^\dagger Q_BC\preceq Q_A
$$

within the separately recorded numerical and model bounds, using a conditioned
whitened-map singular-value or quadratic-form certificate. Small apparent gain
within a model bound is reported as a model defect, never as negative absorption;
outside the bound refine or refuse. Do not rescale C or clip efficiency to one.
This certificate is necessary, not sufficient, evidence of a physical interface.
Positive uncaptured power is claimed only in the common positive metric where
the orthogonal-residual identity holds.

Integrate incident power over its declared incident domain, including light
outside the target aperture. Integrating only over the target core would erase
geometrical clipping. For an analytic field, establish a tail bound or convergence
under domain enlargement. For measured/sampled data, record the finite-domain
assumption; unobserved tails are unknown, not measured losses. Separate errors in
the quadrature, the transverse window and the retained modal span.

#### 4.4 Analytic overlap references

For normalised scalar Gaussians at a common flat-wavefront plane and admittance,
$u_j=\sqrt{2/(\pi w_j^2)}\exp(-r^2/w_j^2)$, direct completion of the square
gives the captured power for relative displacement d,

$$
 \eta=\left(\frac{2w_1w_2}{w_1^2+w_2^2}\right)^2
       \exp\left[-\frac{2d^2}{w_1^2+w_2^2}\right].
$$

For centred beams with transverse phase difference
$\exp(i\Delta\boldsymbol\kappa\cdot\mathbf r)$, multiply the aligned result
by $\exp[-|\Delta\boldsymbol\kappa|^2w_1^2w_2^2/
(2(w_1^2+w_2^2))]$. A curvature difference replaces the real Gaussian inverse
width in the integral with its complex counterpart. These formulas are restricted
to the common paraxial scalar measure; they do not replace the medium/interface
calculation. Direct two-dimensional integration is an independent test of the
implemented alignment signs and amplitude.

For a scalar capillary fundamental, let u be the first positive zero of $J_0$.
Inside radius a, use the normalised profile
$v=J_0(ur/a)/(\sqrt{\pi}a|J_1(u)|)$ and zero outside. The Bessel integral
$\int_0^a rJ_0(ur/a)^2dr=a^2J_1(u)^2/2$ fixes that normalisation. Gaussian
capture is the square of $2\pi\int_0^a r v^*u_1dr$, with the incident Gaussian
normalised over the whole plane. Evaluate this integral independently and sweep
waist, curvature and mode count; do not paste an asserted universal optimum
coupling percentage into a test. Vector/profile corrections are assessed against
the actual `Mode` fields, not absorbed into this scalar reference.

#### 4.5 Reference phase, arrival time and temporal grids

A saved spectrum is in the run's moving frame. If its linear operator removed
$R(\omega,z)=\beta_0(z)+\beta_1(z)(\omega-\omega_{\rm ref})$, the optics
spectrum in the common reference acquires $\exp[+i\int R\,dz]$. Use exactly
the terms removed by that route: on a field grid R is $\beta_1\omega$; an
envelope with or without THG has the separately documented constant term. Track
the accumulated affine phase as a phase offset and time offset, so a long path
need not numerically translate a femtosecond pulse out of its window. A subsequent
stage subtracts its chosen reference once. No per-mode recentering is permitted.

Differentiate three conversions: the sign convention, the raw DFT's time origin,
and raw-DFT amplitude scaling. Let $s_k$ be an optics-convention raw spectrum
centred at t=0. Its normalised analytic spectral density amplitude is

$$
 f_k=\frac{\Delta t}{\sqrt{2\pi}}q_ks_k,
$$

where $q_k=1$ for an envelope and $q_k=2$ on interior positive field bins
(1 at endpoints). Then $\Delta\omega\sum f_k^\dagger Q_kf_k$ is the
energy on the modal representation; apply the spatial metric analogously.
Identical physical frequencies can be copied after this scaling and reference
conversion, then divided by the target scale. Copying raw DFT amplitudes between
different N and $\Delta t$ is not this operation.

For noncoincident lattices, use a complex linear Fourier remap of the source
finite record, with its actual time origin and envelope carrier. A direct
evaluation of its finite-time Fourier sum is the reference; zero padding,
Fourier resampling and chunked matrix evaluation may accelerate the same declared
map. Treat complex spectral zeros naturally, without interpolating an undefined
unwrapped phase. A larger time window extends a localised record only under an
explicitly checked edge/tail assumption. A shorter window or smaller band is an
explicit lossy operation with a report. No requested energy rescales the answer.

Conversion does not recover clipped, aliased, absorbed or never recorded content.
Refuse a target that cannot represent occupied positive frequencies, a materially
nonlocalised time record without a specified truncation, or a representation
outside the receiving route's physics. Field/envelope parity is checked only for
a common admitted physical model. In particular, envelope plasma with THG and
general-modal envelope propagation remain unsupported by this SEP.

#### 4.6 Transport between planes

At each absolute temporal frequency, homogeneous propagation is

$$
 \widetilde E(\boldsymbol\kappa,\omega;L)
 =\exp[i k_z(\boldsymbol\kappa,\omega)L]
  \widetilde E(\boldsymbol\kappa,\omega;0).
$$

Radial symmetry uses the order-zero Hankel representation; Cartesian propagation
uses the transverse Fourier representation with its own spatial origin and
scaling. Refuse significant unsupported evanescent launch content rather than
using a propagation kernel's numerical evanescent cap as an interface model.
Sampling an exact exponential poorly is still inaccurate. Bound the relevant
phase sampling and periodic wrap, qualify domain enlargement, and report any
intentional angular band removal. Reference [2] motivates these sampling checks;
its filtering is not permission to hide removed field energy.

Evaluate an independent output grid through the continuous/discrete reference
transform, with a fast uniform-grid path where valid. Chunked direct Fourier or
Bessel evaluation is the correctness reference for a scaled observation plane.
Do not allocate a transverse-by-transverse-by-frequency propagation tensor.

For a paraxial observation at L in a transparent uniform medium, the Fraunhofer
limit of the Fresnel integral is

$$
 E(\mathbf r,\omega;L)\simeq
 \frac{k e^{ikL}}{2\pi iL}e^{ik|\mathbf r|^2/(2L)}
 \int E(\mathbf r',\omega;0)
        e^{-i(k/L)\mathbf r\cdot\mathbf r'}dA'.
$$

Here the integral uses an explicitly negative-sign spatial transform, independently
of the positive-sign temporal optics convention. A sufficient far-field condition
over source radius R is $kR^2/(2L)\ll1$ for the occupied band; compare with the
finite-distance propagator and refine the output window and sampling. Preserve
the complex phase and the frequency-dependent angular-to-position map. A
far-field spectrum integrated through an aperture is not a substitute for this
field when another coherent element follows it.

A thin lens has paraxial factor
$\exp[-ik(\omega)r^2/(2f(\omega))]$ in the local plane. A focusing mirror
uses its declared focal power with its spectral reflection/Jones response.
An amplitude aperture multiplies by A, with intensity transmission $|A|^2$;
reuse `spatial.aperture_profile` with an explicit conversion from its intensity
weight. The operation's plane must say whether it is a near-field, image or
far-field plane. Rotations and reflections carry the coordinate handedness and
Jones axes consistently.

**Thin-screen model and bound.** Use the local paraxial electric-field screen
at each frequency; it is exact only in the local reference measure
$U_0(E)=\int Y_0|\widetilde E|^2\,d\boldsymbol\kappa\,d\omega/2$ with the
declared transform weights. For transparent media $Y_z/Y_0=\cos\theta$.
For a selected angular cone $\theta\le\theta_*$,

$$
 0\le U_0(E)-U_Y(E)
 \le (1-\cos\theta_*)U_{0,\rm cone}(E)+U_{0,\rm tail}(E)
 \equiv d(E).
$$

Evaluate both input and output, including the spectrum generated by a lens or
aperture. Given the required `model_error` $\epsilon_m$, choose
$1-\cos\theta_*=\epsilon_m/8$, and admit only if
$[d(E_{\rm in})+d(E_{\rm out})]/U_Y(E_{\rm in})$, plus numerical/domain error,
fits $\epsilon_m$. Zero input is an exact separate case. The tail is evaluated,
not silently discarded. An unresolved hard-edge tail or significant
nonpropagating content requires explicit removal with its own report or refusal.
For a reusable map, certify the whole declared source subspace; for a single
field, label the certificate as applying only to that field.

For a phase screen $U_0$ is invariant; the inequality bounds its axial-flux
defect. For a contractive amplitude/Jones screen, local loss
$L_0=U_0(E_{\rm in})-U_0(E_{\rm out})$ is nonnegative, and the difference
between L0 and the measured axial-energy decrement is bounded by the same sum
of d terms. Report that difference as approximation error, not absorption.
This gives a concrete admission rule; $|A|\le1$ alone is not enough in the
nonlocal Y metric.

For a mirror, specify incident/outgoing propagation normals and a shared
s direction perpendicular to their plane; at normal incidence the caller's
declared s axis resolves the degeneracy. Set
$\hat{\mathbf p}_{\rm in/out}=\hat{\mathbf s}\times\hat{\mathbf k}_{\rm in/out}$.
The right-handed local frames and the supplied complex p/s reflection
coefficients define the Jones map. Convert the field between these frames
explicitly, without conjugation; qualify normal-incidence and double-reflection
signs. Coefficients calibrated in another p/s convention must be converted.
The outgoing beam is forward relative to its own plane normal.

For a normal-incidence, homogeneous, nonmagnetic, lossless planar interface,
$t_E=2n_1/(n_1+n_2)$ and $r_E=(n_1-n_2)/(n_1+n_2)$ follow from tangential E/H
continuity. Power transmission is $T=(n_2/n_1)|t_E|^2$, giving $R+T=1$.
This is a useful declared interface model and analytic test. It is not obtained
by inserting two modal effective indices into a bulk Fresnel formula. More
general angular/polarisation coefficients require their own declared planar
geometry and flux test; arbitrary facets remain outside scope. Never apply a
Fresnel factor twice through both the element and the launch adapter.

Existing `Material`, mirrors and compressor pairs act through $H(\omega)$.
Their amplitude and phase affect every spatial sample coherently. `Material`
removes a reference group delay; mirror data can lack constant phase and delay;
grating/prism models remove affine phase and omit some efficiencies. Adapt those
facts explicitly into the plane's reference/calibration record. Require supplied
efficiency when a lossy prediction is requested from a phase-only model.
Negative material lengths or inverse mirror phases are reconstruction operations,
not passive physical elements. Keep their purpose explicit.

#### 4.7 Accounting across a chain

For each operation record input/output energy on the admitted support, its metric,
specified material/reflection loss, aperture/model interpretation, basis loss
where defined, intentional spectral/window removal, and unresolved numerical
or model error. Known losses add sequentially in joules; multiplying broadband
scalar efficiencies is generally wrong when an operation changes the spectrum.
Carry SEP 34's source certificate as a separate, scoped report item. Where
$\delta$ bounds the relative forward-field error in the same positive metric,
$2\delta+\delta^2$ bounds the corresponding relative energy discrepancy;
use that conversion and the admitted map norm before adding it to an
energy-valued `model_error` budget. Do not add a field error directly to an
energy-loss fraction. For successive field-error maps compose their bounds
conservatively, rather than assuming independent errors cancel. A nonlinear
source's background-path certificate alone supplies no end-to-end field-error
bound; include its separate propagation qualification and report what is unknown.
Retain spectral reports so wavelength-dependent coupling remains visible.

For multiple coherent input channels, even a diagonal target metric does not
permit projecting intensities: the interference is in C times the complex field.
For a dense target metric, individual diagonal channel contributions are not
independent physical powers. Report the total quadratic form and any meaningful
block shares; do not invent positive per-channel efficiencies by dropping cross
terms. A field entirely outside a retained subspace may give a valid zero launch;
it does not trigger a normalisation division by zero.

Carry the realised incoming field and its provenance once. The existing modified
noise scheme has auxiliary bookkeeping; reconstruct according to its documented
saved-field meaning, not by adding every stored noise array. A new stage does not
automatically draw another one-photon floor. Transporting a classical realisation
through loss does not model vacuum injected by unused quantum ports; quantum
loss-channel completion and ensemble claims are outside this SEP.

### 5. Proposed interface

#### 5.1 Plane, launch and report records

Use domain modules, with no generic utility or workflow framework. The proposed
public surface is:

| Name | Responsibility |
|---|---|
| `FieldPlane` | Immutable physical-plane record with native coefficients/samples, references, geometry, medium, metric and provenance |
| `FieldPlane.from_output(output, plane=..., source=...)` | Read one stored plane and validate/rebuild its spatial and frame reference |
| `FieldPlane.from_analytic(grid, geometry, field, medium=..., reference=...)` | Sample an optics-convention physical analytic field with declared support and units |
| `FieldPlane.from_samples(...)` | Bind existing physical arrays to explicit axes, units, domain, medium and reference |
| `pulses.SpatialPulse` | The one Gaussian/super-Gaussian beam builder, extended to bind an incident plane and its physical energy |
| `FieldPlane.from_pulse(pulse, grid, geometry, medium=..., reference=...)` | Build a plane from `SpatialPulse`; no second Gaussian implementation |
| `regrid(plane, grid, ...)` | Controlled temporal representation/window/band conversion with its report |
| `transport(plane, elements, ...)` | Apply an ordered sequence of spatial and spectral elements; return the resulting plane and report |
| `couple(plane, target, ...)` | Project into an explicit target port; return `CouplingResult` |
| `CouplingResult` | Target-bound `launch` and a structured `report`, including provenance and numerical/model certificates |
| `driver.Simulation.input_port()` | Return the receiving grid, basis, medium, entrance reference and admitted coupling models |
| `driver.Simulation.with_input(result.launch)` | Validate and install the already normalised target launch |
| `coupling.CouplingModel` | Select the §4.3 model and explicit approximation budget |
| `coupling.Outcouple` | Synthesize the guide exit under a declared forward interface model |
| `coupling.FreeSpace` | Homogeneous angular-spectrum/Hankel transport with an output-plane sampling specification |
| `coupling.ThinLens` | The qualified paraxial phase screen and its chromatic focal length |
| `coupling.Aperture` / `coupling.TransmissionMask` | Intensity-defined aperture converted to amplitude / directly supplied complex field transmission |
| `coupling.PlanarInterface` | A stated bulk Fresnel model in its admitted geometry |
| `coupling.PolarisationTransform` / `coupling.Mirror` | Qualified Jones map / spectral reflection and incoming/outgoing frame map |
| `coupling.SpectralTransfer` | Adapt a transfer function or compressor result, retaining affine-phase and efficiency metadata |

`FieldPlane`, `CouplingResult`, `regrid`, `transport` and `couple` are public
members of `solprop.coupling`; the table prefixes other modules explicitly.
`GaussianBeam` is not introduced. `SpatialPulse` remains the single beam builder,
and `FieldPlane.from_pulse` is its incident-field record boundary. The builder
and analytic/sampled constructors share one normalisation/projection path.
Replace ambiguous `propz` energy-at-waist semantics: requested energy is fixed
at the declared launch plane after geometric preparation. A waist reference
specifies shape/curvature; attenuation before that plane is a separate transport
operation. No deprecated alias or alternate restored-energy launch survives.

The plane records its native representation and units, temporal grid and absolute
frequency lattice, power metric, transverse coordinates or modal reference,
polarisation axes, oriented physical plane and resolved medium. Its temporal
record carries the time origin, carrier frequency/phase reference, accumulated
frame phase and delay, and which affine phase calibrations are known. Source
identity, saved-plane index and preparation history supply provenance. Owned
read-only arrays prevent a caller from mutating a validated plane through an
alias; frozen attributes alone are insufficient.

`source` supplies the original simulation or an equivalent verified reference
when an output does not embed its full modal tables. An extraction default may
select the final saved plane; an explicit z must resolve to a stored plane within
the declared coordinate tolerance, not silently choose a nearby physical plane.
Identity and integral frame metadata must be sufficient for later extraction.

An analytic callback accepts physical x/y coordinates and absolute angular
frequency, returning components in declared SI spectral units; provide a separate
time-sampled constructor for an analytic time-domain field. Fix their exact
array signatures against the standard axis vocabulary during Phase 2. Native
modal coefficients keep `(n_modes, n_pol, n_omega)`, sampled local fields use
`(n_nodes, n_components, n_omega)`, and general vector modes keep the existing
distinction between `n_pol=1` and three physical components. No scalar squeezing
occurs in these records.

`energy=` on an incident pulse means the physical energy crossing the declared
launch plane in its medium, before target coupling. A meter reading in air is
converted through an explicit window/interface and its transmission. Direct
modal amplitudes already describe light inside the guide.

The named physical inputs are SI lengths, seconds, rad/s and radians; gas pressure
uses the existing bar-valued user interface. Target geometry and medium are
explicit. A finite-domain input says whether it declares the field to be zero
outside its samples or leaves the exterior unknown. Report unknown tails rather
than normalising an arbitrary crop as a complete measured beam.

Alignment inputs specify lateral displacement in metres, paraxial tilt in radians
and polarisation rotation in the declared local axes. Their signs follow the
physical synthesis map. A radial operation requires cylindrical symmetry:
displacement, tilt, asymmetric masks and other symmetry-breaking operations
require an explicit Cartesian representation and its sampling qualification.
Vector profiles with angular structure are not silently reduced to order-zero
Hankel fields.

All spatial incident-field entry points converge on this implementation. Replace
the current `transverse=` shape-plus-restored-energy path; either accept the
incident-field record directly or route that spelling through its incident-energy
semantics. There is no retained alternative that silently restores missed energy.
Direct modal amplitudes remain an explicit internal launch, not an incident beam.
The phase-1 fix must supply the incident integration domain and power measure at
the existing profile boundary. Merely removing the final rescaling is insufficient
if the profile was already normalised over the receiving guide's aperture.

The target-bound launch includes a grid, basis, metric and entrance-position
identity. `with_input` refuses mismatch rather than interpreting coincident array
shapes as compatibility. It records the coupling provenance automatically and
adds no new noise unless the caller explicitly supplies it. A different initial
field may require a new ionisation table/rate reference: respect the existing
`with_input` shared-table limits and refuse an out-of-domain target rather than
silently rebinding physics.

#### 5.2 Composition sketches

The following is a proposed API sketch, not executable documentation:

```text
exit_plane = FieldPlane.from_output(first_run, plane=-1, source=first_simulation)
transported = transport(
    exit_plane,
    [Outcouple(external_medium), FreeSpace(distance),
     SpectralTransfer(compressor.transfer), ThinLens(focal_length),
     Aperture(radius), FreeSpace(distance_to_entrance)],
    geometry=receiving_plane_geometry,
)
coupled = couple(transported.plane, second_simulation.input_port(),
                 model=CouplingModel("scalar_paraxial", model_error=budget))
second_run = second_simulation.with_input(coupled.launch).run()
print(coupled.report)
```

`Outcouple` synthesises the guide's field and applies an explicitly selected
interface approximation; it is not an implicit unit-efficiency Fresnel solver.
For direct guide-to-guide transfer the same `couple` operation evaluates the
overlap without a transverse propagation grid. `SpectralTransfer` adapts existing
`optics` objects and compression results. Preserve `optics.Chain` as the
frequency-only multiplication. `transport` accepts a `Chain` directly as one
frequency-only block, and accepts a chain as an item inside an ordered spatial
sequence. The report records the block's factors and its position; only scalar
frequency multipliers within that block commute. It never reorders a block past
focusing, clipping, propagation, projection or a noncommuting Jones operation.

The radial-input workflow replaces `exit_plane` with the output of an ordinary
radial run and couples at the capillary entrance. Gaussian, analytic and sampled
fields replace only the source constructor. To obtain a far-field observation,
stop after transport to the specified output plane; do not project onto a guide.
A nonlinear gap is a spatial simulation between two such operations.

#### 5.3 Refusals and outputs

Refuse paths failing SEP 34's §4.5 certificate or outside its scope,
ambiguous jump sides, missing mode profiles or references,
inconsistent units/measures, ill-conditioned power metrics, occupied unsupported
bands/cutoff, unqualified lossy modal duals, unresolved windows, forbidden
field/envelope response combinations and invalid symmetry reductions. Identity
and zero transfers preserve exact structural contracts.

Return immutable structured diagnostics, with finite meaningful energies and
explicit unavailable quantities/reasons, rather than invented efficiencies.
When incident energy is zero, efficiency is undefined but the zero field remains
valid. A numerical tolerance is never used to hide a finite model defect.
Source and target hashes, selected planes, operation order, reference changes,
material identities, model choices and remapping/loss reports enter configuration
provenance. Arrays may be recorded by the existing digest mechanism. The copied
launch must be sufficient to run the next stage without reopening the source.

Scans compose these pure preparations in their existing builders. This SEP
does not add a scheduler, a multi-stage resume protocol or device arrays to the
public seam. Resume continues one simulation; coupling creates a new initial-value
problem with its own run identity.

### 6. Implementation strategies

#### 6.1 Layers and shared definitions

Recommend a physics-layer `solprop.coupling` domain for plane specifications,
reference records and optical models. Put quadrature, conditioned projection,
remapping plans and validated numerical maps under `solprop.translate`; saved-plane
extraction belongs in processing and target installation in the driver. Share the
existing general-modal reference/identity machinery without making physics import
processing or a backend. Update the architecture's package table and tested
structure figure when these modules are implemented.

The offline application is a host operation using NumPy/SciPy, independent of the
backend that created the source or runs the target. There is no reason to add a
new propagation backend for it. SEP 34 owns the hot spatial flux/statistics/noise
support that crosses the seam. There, T passes immutable arrays/factored
admittance data; B does not construct a material, mode or optical element.

Reuse aperture definitions, mode profiles, Hankel scalings, Fourier boundaries,
material selection and spectral optics. Replace the duplicated launch accounting
at its source. Do not build a second general-modal synthesis reference beside
the checked one. No dependency is added: NumPy/SciPy provide quadrature, linear
algebra, interpolation and transforms.

#### 6.2 Cost, memory and certification

An overlap plan costs approximately $O(N_\omega N_q M_A M_B)$ in a direct
evaluation and stores $O(N_\omega M_A M_B)$ coefficients if materialised. A
single-field projection can stream with $O(N_qM_B)$ work per frequency. Use node
and frequency chunks when maps or full fields would dominate memory. Record
the actual choice and bytes; do not conceal a large reconstruction behind a
small output record.

FFT/Hankel transport uses existing transforms on a common grid; independent-plane
direct evaluation costs $O(N_\omega N_{\rm in}N_{\rm out})$ and is chunked.
Use it as the reference and a supported small-problem implementation; qualify
accelerations against the same discrete map. Cache plans by geometry, grid,
material, basis and reference identity, not by a mutable object's address.
Factored $Y_z$ data follow SEP 34 and the spatial memory bounds of
[SEP 22](archive/sep-0022-memory-lean-spatial-execution.md) and
[SEP 28](archive/sep-0028-free-space-evaluation.md);
do not add a permanent transverse-by-frequency array when the root can be
evaluated from existing separable coefficients in a bounded workspace.

Certificates account separately for quadrature, basis conditioning, transverse
and temporal truncation, transform sampling and model approximation. Strict
admission must be expressed in physical/numerical error targets, not an arbitrary
universal percentage of missed light. Uncaptured physical power may be large and
perfectly valid; numerical uncertainty must be small enough for the requested
conclusion.

#### 6.3 Backends and schemas

The coupling/transport maps are host-side NumPy/SciPy operations. SEP 34 owns the
cross-backend flux changes, including an atomic public activation after
`numpy_ref`, `numba_cpu` and `native` CPU qualification; this SEP does not make a
working compiled spatial route temporarily refuse its existing inputs.

Plane reference, launch and driver changes bump the kernel schema only when they
alter its protected contract; update literal tests and native consumers in the
same change. Persisted plane reference and lineage changes bump the output schema.
Use current-schema read/write/resume tests. No old-file migration is required.

Test outputs and relaunches from all three CPU backends, including mixed
source/receiver pairs. The numerical host map is shared, not reimplemented three
times. GPU acceleration of offline maps is not required; native GPU propagation
and its flux kernels follow SEP 34's separate device qualification. No host
fallback is hidden inside a device statistic.

### 7. Verification and validation

Use [testing](../testing.md), [accuracy](../accuracy.md) and SEP 25. A matched
discrete operation uses a rounding budget based on reduction length, transform
error and conditioning. A changed grid, modal truncation or propagation uses
independent refinement and an accuracy envelope. Do not demand bitwise agreement
between independent backends. Copies, hashes, mode labels, explicit no-ops and
hard zeros retain exact structural tests.

| Case | Comparison and acceptance evidence |
|---|---|
| Shared plane flux | Consume SEP 34's host evaluator; agree with its in-run observations at uniform and admitted radial-gradient planes, using analytic-signal rather than real-field endpoint integration |
| Incident launch and noise | Requested physical energy before coupling and captured energy after coupling; carry SEP 34's already normalised noise without a second draw |
| Identity and gauge | Same-guide recovery, rotated degenerate bases and coherent two-mode interference; physical E/H and energy invariant under a simultaneous basis/amplitude transformation |
| Scalar overlap | Gaussian waist/displacement/tilt/curvature formulas and independent Gaussian–Bessel quadrature; deliberate poor and zero capture, with no restoration of energy |
| Cross-guide coupling | Different radii, mode sets and supported solid guides; direct overlap versus synthesis/project route; passive-map certificate and explicit failures outside the admitted interface model |
| Modal truncation | Retain successively larger nested mode sets; distinguish quadrature convergence from captured-subspace convergence and finite-domain effects |
| Spatial transfer | Radial versus Cartesian for a symmetric beam; displaced/tilted input on Cartesian; refusal of symmetry destruction on a radial grid |
| Linear transport | Gaussian diffraction and focus, independent Fresnel/Bessel evaluation, aperture diffraction, thin-element order, reflection/Jones signs, and finite-distance convergence to the Fraunhofer limit |
| Spectral optics | Reuse existing element references; explicit $\lvert H\rvert^2$ attenuation, phase-only invariance in its admitted metric, no double Fresnel factor and an efficiency report that changes with spectral reshaping |
| Temporal conversion | Equal-bin scaling, unequal windows/sample spacings, delays, CEP, carrier changes, spectral zeros and separated bands; field/envelope round trips only on common supported physics |
| Refusals | Missing saved fields/references or required path certificate, exceeded gradient budget, ambiguous jump side, nonconstant absorbing path, unsupported measure/endpoint/cutoff, invalid provider dual, unresolved time/space support and route/backend mismatch |
| Provenance | New-schema memory/HDF5 round trips, configuration identity changes, target-bound input mismatch, and no duplicate noise draw |

Use at least three refinement levels when asserting an order. Vary domain size,
node resolution, modal span and temporal sampling independently. A three-dimensional
array with more samples is not evidence of a larger physical domain. Compare
fields after applying the declared reference-frame conversion, not after fitting
away an unexplained phase or delay.

The short integrated qualification set must include all eight workflows of §2,
including a radial field-grid Kerr/plasma/THG entrance calculation followed by
capillary propagation, and capillary SPM followed by a specified compressor and
a second capillary. The plasma/THG case checks the transferred generated bands,
not only the fundamental's total energy. Tests use about ten propagation steps
or fibres short enough to finish in seconds. Refine the physical approximation
and propagation separately from the coupling calculation.

The tutorial below is an additional gallery deliverable. Its convergence evidence
must resolve the absolute change in incident-energy-normalised fundamental share
$s_{11}=U_{11}/U_{\rm incident}$ and first higher radial-mode share $s_{12}$.
For each claimed nonzero change, the combined independently estimated numerical
uncertainty of the two runs must be at most one tenth of its magnitude. This
factor is a reporting criterion, not a physical constant. Report total capture
separately; its change need not be the largest effect. Do not assume any sign.
Measure single/multimode overlaps, radial/Cartesian extraction, regridding and
scaled-plane transport with actual input/output sizes and memory. SEP 34 owns the
short-window benchmark ledger for hot flux changes. Never extrapolate a timing
into a full experimental run.

### 8. Documentation

Add a manual chapter `docs/guide/workflows/coupling.md` describing the four blocks,
their physical measures and approximations, source/target reference planes,
energy reports and all eight compositions. Include examples for a same-basis
handoff, a changed capillary radius, solid-fibre overlap, analytic and Cartesian
inputs, outcoupling and compression between stages. Cross-link input pulses,
general-modal launch, free-space setup, output access, polarisation and compression.
Update the API reference and public docstrings with units, shapes and refusals.

Update conventions, architecture/structure, kernel/output contracts, numerical
accuracy and backend capability pages with implementation. Record the changed
incident-energy and spatial-flux semantics in `CHANGELOG.md`, with a SEP link,
and the deliberate difference on the Luna comparison page. No changelog entry
claims functionality merely because this draft was added.

#### 8.1 Required focused tutorial

Add **Nonlinear beam preparation and coupling into a capillary** as
`docs/tutorials/free_space/plot_08_radial_capillary_coupling.py` (use the next free
gallery number if that name is taken at implementation). It is a focused tutorial,
executes on normal docs builds, and is linked from the new coupling chapter and
the capillary/free-space guides. The manual explains how to convert a meter
reading through an entrance interface. Generate its output through Sphinx; do not
hand-edit the rendered gallery.

Use a scalar linearly polarised Gaussian on an envelope grid in uniform argon.
The model contains Kerr self-focusing and self-phase modulation, with plasma,
THG and Raman omitted explicitly. The starting specification is:

| Quantity | Value |
|---|---|
| Vacuum wavelength | 800 nm |
| Intensity FWHM | 30 fs |
| Argon pressure | 3 bar, uniform in gap and capillary |
| Temperature | The declared material reference temperature, recorded in the example |
| Gas gap before entrance | 20 mm |
| Linear-focus 1/e² intensity radius | 37.5 μm, at the capillary entrance; w/a = 0.5 |
| Capillary radius | 75 μm |
| Capillary segment | 5 mm |
| Initial strength target | Linear-Gaussian on-axis nonlinear phase $B_{\rm lin}=0.3$ rad |

Determine energy from the physical material functions, not a copied nonlinear
coefficient. Let $n_0$ be the real gas index at the pump, $k_0=2\pi/\lambda_0$
the vacuum wavenumber and $P_0$ the peak axial power. In the narrowband paraxial
sizing model,

$$
 z_R=\frac{\pi n_0w_0^2}{\lambda_0},\quad
 I_0(z)=\frac{2P_0}{\pi w_0^2[1+(z/z_R)^2]},\quad
 n_2=\frac{3\chi^{(3)}}{4\varepsilon_0cn_0^2}.
$$

Integrating from $z=-L$ to the waist at zero gives

$$
 B_{\rm lin}=\frac{2k_0n_2P_0z_R}{\pi w_0^2}\arctan(L/z_R),\quad
 U=P_0\tau_{\rm FWHM}\frac{\sqrt\pi}{2\sqrt{\ln2}}.
$$

These equations follow directly from the Gaussian beam/pulse definitions and the
package's scalar Kerr normalisation. A sizing probe with the current material
functions at the revised waist gives about 12.1 μJ, 0.379 GW and 0.124 of the
physical critical-power estimate (the probe is recorded in the review response).
These are provisional analytic estimates, not a propagated or converged result.
Recompute with SEP 34's physical launch and critical-power conventions and
recorded optmat material identity. The earlier 50 μm, 14.1 μJ sizing is
superseded, not a regression target.

The detuning is deliberate: the review identified a near-stationary fundamental
overlap at the old waist. At w/a = 0.5, scalar Gaussian–Bessel quadrature puts
about 0.906 of the incident energy in the fundamental and 0.092 in the next
radial mode. These numbers size the lesson only; the propagated broadband
complex field and converged mode set determine its results. Mild Kerr evolution
must be demonstrated to change the named modal shares above their error bounds;
neither a lens estimate nor the static overlap probe proves that outcome.

Run two calculations from the identical physically focused incident field:
linear propagation through the gap, and propagation with Kerr enabled. At the
entrance, retain each full complex radial field and couple into the same capillary
mode set. Launch each captured result through the 5 mm capillary with identical
Kerr-only capillary physics, so the difference is the prepared incident field.
Do not replace either entrance field by a fitted Gaussian or its power spectrum.

Name the linear modal beating in the 5 mm segment. With
$\beta_m\simeq nk_0-u_m^2/(2nk_0a^2)$, the fundamental/next radial-mode
beat length is

$$
 L_{\rm beat}=\frac{2\pi}{|\beta_{12}-\beta_{11}|}
 =\frac{8\pi^2 n a^2}{(u_{12}^2-u_{11}^2)\lambda_0}
 \simeq22.5\ {\rm mm}.
$$

Here $u_{11},u_{12}$ are the first two J0 zeros, and the package labels the
family HE_1m. Thus 5 mm is about 0.22 beat periods: the output contains partial
linear mode beating, not merely modal filtering or a new nonlinear effect.
Evaluate the actual dispersive beta and differential wall loss across the
retained band when implementing the plots; the carrier estimate is explanatory.
The identical capillary physics in the two cases controls for this evolution.

Plot beam radius and on-axis intensity through the gap; entrance fluence and
wavefront change; entrance temporal and spectral changes; incident/captured energy
and modal shares; and the short capillary outputs. Specify the slice or spectral
component used for wavefront phase, masking phase where amplitude is unresolved.
Use physically meaningful modal/block shares under the target metric.

Before fixing the tutorial's grids and mode count, independently refine radial
domain, radial resolution, temporal window/band/sampling and retained modes.
Require visible but mild nonlinear changes and no unresolved collapse, collar
contamination or spatial aliasing. Check the magnitude of omitted ionisation
with the existing rate model as an offline diagnostic; if the Kerr-only
assumption is not defensible, reduce the strength target and document the reason
rather than silently adding plasma. The starting numbers are design coordinates,
not a promise that every omitted process is negligible.

Assess omitted ionisation at the revised, higher on-axis intensity; the review's
assessment of the old waist is not a qualification of the new one.

Keep the final tutorial deterministic and sized for a normal docs build. Full
nonlinear propagation is permitted for this focused gallery when needed to show
the effect, but its unit/parity regressions remain short. Report the runtime and
the numerical evidence with the implementation phase. This teaching calculation
is separate from the mandatory field-grid plasma/THG transfer test.

### 9. Compatibility and impact

The maintainer explicitly chooses a clean breaking change before release.
There are no legacy files to migrate for this work. Introduce no old-normalisation
switch, compatibility alias or deprecation period. Remove superseded launch
accounting, update callers and fixtures, and reject unsupported old schemas with
a clear version error rather than interpreting old data with new units.

The observable differences are intentional: a spatial launch's requested energy
uses axial flux; incident energy is reduced by actual coupling; modal capture is
frequency-dependent and coherent; and transformed fields carry explicit phase
and basis references. Already supplied physical V/m arrays are not arbitrarily
rescaled merely because their reported energy changes. Current-schema resume
retains strict identities. Frame/reference metadata and flux records require
versioned output/seam changes when implemented, not in this documentation patch.

SEP 34 owns corrected spatial energy, noise and observables and preserves admitted
radial gradient simulations through the adiabatic amplitude correction.
Coupling consumes its local plane measure; uniform-only restrictions apply to
the homogeneous `FreeSpace` element, not to all source simulations. The hybrid
planar and general-vector metrics keep their own derived meanings.

SEP 5 consumes SEP 34 for solid carriers; SEP 8 retains vector crystal
fields/interfaces; SEP 9 retains its in-guide measurements. Their dependencies
and the roadmap follow that division. Nonadiabatic scattering, wider detector
catalogues and multi-pass-cell assembly remain explicit future work. Spatially
resolved compressor layouts remain outside this SEP.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Bare output arrays between simulations | Convenient for an identical known basis; loses the reference, measure and remapping contract needed across routes |
| A separate helper for every source/target pair | Duplicates normalisation and Fourier boundaries; use synthesis, transport and projection adapters |
| Always materialise a Cartesian field | General but wasteful for direct modal overlaps and radial symmetry; retain native representations |
| Use `DataPulse` or restore launch energy after projection | Hides clipping and imperfect capture; preserve incident scale |
| Infer a Gaussian from radius/energy | Discards wavefront and space-frequency correlations produced by nonlinear propagation |
| Treat electric-field residual as lost power | Valid only for a derived common positive metric; separate the diagnostics |
| Multiply by one refractive index | Misses broadband and angular weighting and does not fix current work |
| Full junction scattering immediately | Needs reflected/radiation/evanescent bases and new solvers; outside the selected forward scope |
| Exact vector free-space optics immediately | Important at high NA, but a new propagation model; keep the current scalar/transverse-field route explicit |
| A fixed-window FFT for every distance | Can wrap or undersample the transfer; admit by sampling evidence and support independent output planes |
| Redesign the prism/grating models | Existing spectral models suffice for compensated ideal layouts; resolved spatial compressors are separate work |
| A multi-stage execution framework | Adds scheduling/resume complexity before the physical operations are qualified; compose ordinary simulations |
| Preserve vacuum-normalised APIs and old files | Explicitly rejected for this unreleased change; update the public interface cleanly |

### 11. Phases

No numerical implementation begins until acceptance. Implement SEP 34 first.
Its full public flux activation precedes SEP 31's incident-launch change,
as the maintainer chose on 2026-10-23, so there is no interval with corrected
modal launches but inconsistent spatial observations. SEP 34's private host
reference alone is not permission to merge SEP 31 phase 1 early. Each phase
records delivered paths, deviations, evidence, unresolved questions and handoff
in Part II.

| Phase | Deliverables | Required evidence |
|---|---|---|
| **1 — Incident-energy correction** | Fix the present profile-launch defect with the shared physical measure and existing profile inputs, before introducing `FieldPlane`; preserve captured amplitudes rather than restoring requested energy | Poor and zero capture, full incident-domain normalisation, coherent mode shares and unchanged direct inside-guide amplitudes; all CPU receivers |
| **2 — Plane exchange** | `FieldPlane`, source/target references, `input_port`, same-basis relaunch and temporal remapping | Identity, delay/CEP/carrier and equal/unequal-grid tests; current-schema persistence; all CPU source/receiver pairs |
| **3 — Coupling** | Unify the phase-1 implementation behind analytic/sampled/`SpatialPulse` inputs, the concrete §4.3 models, overlaps, target-bound launches and reports | Gaussian/Bessel references, independently qualified provider/interface envelopes, passive-map certificates, and separate domain/quadrature/modal refinement |
| **4 — Transport core** | Homogeneous propagation, outcoupling, finite and far observation planes, bounded thin screens, explicit reflection/Jones frames, and existing spectral optics | Independent diffraction and bulk-interface references, angular/flux-defect checks, ordering, explicit efficiencies and affine-phase provenance |
| **5 — Integrated qualification and tutorial** | Eight bounded compositions from §2, the detuned radial tutorial, manual and capability matrix | Short field-grid Kerr/plasma/THG transfer, SPM → compressor → capillary, named modal-share uncertainty, CPU parity and final gates |

**Final means this bounded core is complete.** Retain all eight compositions:
they exercise distinct seams (guide identity, spatial input, analytic/sampled
constructors, output-plane scaling and spectral phase). The required far field
is scalar and paraxial with its stated error bound; mirrors are local-frame
Jones/spectral maps, not coating or geometric-optics solvers; compressor blocks
reuse existing spectral transfer functions. No general junction scattering,
high-NA vector optics, resolved compressor layout, new in-run detector catalogue
or multi-pass-cell scheduler is required for Final. Each phase is useful before
Final; that status does not claim the wider optical-laboratory scope.

During each slice run fast checks and directly affected modules. At the end run
the full gate, slow lanes and coverage once, or use their CI evidence, as
`AGENTS.md` specifies. Run the native gate if protected native consumers change.
After a failure rerun the failing modules. Tests/timings use short propagations;
the focused gallery has the standing full-problem exception.

For this documentation revision run diff whitespace checks, codespell on changed
files, `tests/test_seps.py` and the warnings-as-errors Sphinx build. Phase 5, not
the draft, supplies the runnable tutorial.

### 12. Open questions

The physical model choices below are proposed explicitly for review, not delegated
to implementation. Qualification may narrow admission; a different model needs
a recorded proposal revision.

| Question | Disposition |
|---|---|
| **Q1 — Cross-model projection** | Proposed resolution: the three models and error-budget admission in §4.3. Phase 3 must qualify each provider pair; no unqualified pair is admitted merely because a projection can be computed. |
| **Q2 — Thin elements and reflection** | Proposed resolution: §4.6's local paraxial screen, input/output angular-flux defect bound and explicit p/s frames. Refinement cannot substitute for a missing model bound. |
| **Q3 — Numerical admission budgets** | Phase 2/3 implements the required model budget and justified numerical/conditioning/tail enclosures under §4.3. Qualified pair envelopes may replace the Δ screen, never the total error bound. No coupling-efficiency floor is introduced. |
| **Q4 — Plane persistence** | Phase 2 fixes and tests the minimal standalone reference record; missing information is refused. |
| **Q5 — Tutorial evidence** | The maintainer chose 37.5 μm on 2026-10-23. Phase 5 fixes grids/mode count, qualifies the Kerr-only regime and resolves changes in s11 and s12 to §7's criterion. |
| **Q6 — Deferred physics** | General junction scattering, high-NA vector optics, varying absorption and resolved compressors remain roadmap work. Certified real-index radial forward amplitude transport belongs to SEP 34. |
| **Q7 — Native backend location** | Closed: `native/` in this repository, distributed as `solprop-native`; changes use the same branch and native gate. |

Review completion recorded in the header describes the historical review below.
These revisions and SEP 34 still require acceptance; neither is marked Accepted.

### 13. References

1. G. K. Svendsen, M. W. Haakestad and J. Skaar, “Reciprocity and the scattering
   matrix of waveguide modes,” *Physical Review A* **87**, 013838 (2013).
   [doi:10.1103/PhysRevA.87.013838](https://doi.org/10.1103/PhysRevA.87.013838);
   [author manuscript](https://arxiv.org/abs/1301.2458).
2. K. Matsushima and T. Shimobaba, “Band-limited angular spectrum method for
   numerical simulation of free-space propagation in far and near fields,”
   *Optics Express* **17**, 19662–19673 (2009).
   [doi:10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662);
   [author repository](https://kansai-u.repo.nii.ac.jp/records/11127).
3. E. B. Treacy, “Optical pulse compression with diffraction gratings,”
   *IEEE Journal of Quantum Electronics* **5**, 454–458 (1969).
   [doi:10.1109/JQE.1969.1076303](https://doi.org/10.1109/JQE.1969.1076303).
4. L. Arissian and J.-C. Diels, “Carrier to envelope and dispersion control in a
   cavity with prism pairs,” *Physical Review A* **75**, 013814 (2007).
   [doi:10.1103/PhysRevA.75.013814](https://doi.org/10.1103/PhysRevA.75.013814).

**Source audit.** The package's conventions, spatial operators/functionals,
ordinary/general-modal synthesis and projection, input replacement and optics
implementations were inspected. The author manuscript of [1] was inspected for
its reciprocity and junction scope. Publisher/repository records establish [2]'s
sampling problem and bibliographic identity; its full algorithm is not transcribed
or independently qualified here. [3] and [4] are the existing compressor models'
primary references; [4]'s publisher abstract/identity was checked, but neither
compressor derivation was newly revalidated from its full primary text in this
draft. Reuse their existing implementation and tests; verify any new external
formula in its implementation phase. The Gaussian integrals, scalar flux/work
identity and tutorial sizing are derived above rather than copied coefficients.
Material values continue to come from optmat's resolved models and sources.
No new material constant or experimental validation is claimed.

## Part II — Implementation record

### Phase 1 report — incident-energy correction

**Delivered (2026-10-24, `feature/sep-0031-simulation-coupling`).**

- `pulses.SpatialPulse.incident(grid, xy, index=...)` returns an `IncidentField`:
  the beam given its requested energy or peak power over its **whole** transverse
  plane in the paraxial measure $M_0 = n_{\rm ref}\varepsilon_0c\,dA/2$, then
  sampled at the given points, with the energy each bin carries through the plane.
  It reuses `_Plane` and `_normalising_factor` (one "column" whose transverse
  integral is the profile's closed form
  $\int|u|^2dA=\pi w^2 2^{-1/N}\Gamma(1+1/N)$), as `PlanarPulse.build` does, and
  shares the profile and Jones normalisation with `SpatialPulse.build`. No second
  Gaussian is written.
- `Simulation.from_waveguide(provider=..., incident=SpatialPulse(...))` and
  `prop_waveguide(..., incident=...)` replace `transverse=`. The driver takes
  $n_{\rm ref}(\omega)=\operatorname{Re}\sqrt{\varepsilon}$ of the guide's core
  region (the region on the axis, whose law is z-independent, so it is the
  entrance's), samples the beam at the entrance nodes, projects it with
  `project_transverse_field` and launches the projected amplitudes **unscaled**.
  A zero capture is a zero launch.
- `driver.simulation.IncidentLaunch` records `reference_index`, `incident` and
  `captured` per frequency bin (J), with `uncaptured` (not clipped) and totals,
  as `configuration["launch"]`; it reaches the file's configuration and hash. The
  captured energy is $\mathbf a^\dagger W\mathbf a$ under the run's declared
  metric, the number `physical_energy` reads at the entrance. A derived run
  (`with_input`) drops the record with the constructor's pulse.
- Removed with no alias: `transverse=`, `_projected_launch`, the rescaling to
  `energy=`, the 3 % `GeneralModalBasisWarning` and `LAUNCH_UNCAPTURED_WARNING`.
  Channel launches and `with_input` arrays are unchanged.
- The advanced benchmark's graded-index case (`benchmarks/guides.py`) launches
  an incident 12 µm beam, so its launched energy is now the captured share.
- Manual: [general modal propagation](../../guide/waveguides/general-modal.md),
  [solid fibres](../../guide/waveguides/solid-fibres.md) (a runnable snippet),
  [supplied guides](../../guide/waveguides/supplied-guides.md), a comment in the
  graded-index tutorial, and the changelog.

**Deviations and why.**

1. *No analytic-callable input (item b).* A callable profile needs its own
   public record of support, tail bound and quadrature. Phase 3's
   `FieldPlane.from_analytic` is that record, and a phase-1 record would be a
   public interface made only to be replaced. It is deferred to phase 3.
2. *`propz` must be 0.* Carrying a beam from its waist to the entrance is
   transport, and §5.1 changes the `propz` meaning in phase 3. The driver and
   `SpatialPulse.incident` refuse a nonzero `propz` by name.
3. *Keyword `incident=`, not `transverse=`.* The beam carries its own temporal
   pulse, so the pulse keywords are refused beside it, as is `channel=`.
4. *No warning remains.* A poor capture is physics. No numerical defect is
   meaningful to warn about here: the projection's own failures are refusals.
5. *Scalar Jones on a vector guide is x-polarised*, the package's capillary
   default.

**Verification run.** Targeted modules only, `uv run --no-sync pytest -n auto`:
`tests/driver/test_general_modal.py`, `tests/translate/test_modal_transform_tables.py`,
`tests/test_pulses.py`, `tests/test_spatial_flux.py` and `tests/test_display.py`,
548 passed (medium tier included); `tests/test_benchmark_guides.py` 9 passed;
`tests/test_guide_snippets.py` (compilation) 18 passed; doctests of `pulses.py`
and `driver/simulation.py` pass. The solid-fibres page's snippets were executed
by hand (3.3 s; a 12 µm beam into the 25 µm graded-index guide captures 0.964).
ruff, ruff format, basedpyright (0 errors), numpydoc and codespell are clean on
the changed files, and `poe docs` builds clean (warnings as errors, 2 min 45 s). Evidence, on the 1.5 µm step-index strand (index 1.45 in air):

- `SpatialPulse.incident` agrees with `SpatialPulse.build` on a radial grid in
  index 1.45 to 1.5e-7 of the peak field, the beam's obliquity, for Gaussian and
  order-2 beams, energy and peak power, envelope and field grids.
- Poor capture: a 20 µm beam captures 0.9 % (bound asserted: < 2 %); the run's
  entrance-plane `physical_energy` equals the captured energy to 1e-12, not
  `energy=`. A 1.3 µm beam captures 92 %.
- Zero capture by symmetry: a y-polarised beam into the even HE11 channel captures
  8e-36 of the incident energy (bound $(n_{\rm nodes}\epsilon)^2$) and runs.
- Incident energy is bitwise the same for a 1.5 µm and a 2.5 µm strand; captured
  differs.
- Coherent shares: Jones $(1, 0.5i)$ gives $a_{\rm odd}/a_{\rm even}=-0.5i$
  (internal convention) to 1e-6, with the captured total of the x-polarised beam;
  on the dense circular basis an x-polarised beam splits equally.
- Receivers: ten 0.5 µm steps on `numpy_ref`, `numba_cpu` and native CPU each store
  an entrance plane bitwise equal to the launched state, whose energy is the
  captured energy.

Not run: the full gate, slow and coverage lanes (end of SEP), and GPU. The
graded-index worked example (`docs/examples/waveguides/plot_04_graded_index_multimode.py`)
still uses `transverse=`; it was **not** edited, because an edited worked example
must be re-run by hand at full length and the normal build refuses a source that
differs from its committed output.

**Limitations.** The measure uses the core's index over the whole plane, so on a
strongly guiding strand the uncaptured energy also carries the difference between
$M_0$ and the guide's flux metric (the 1.45-in-air strand's best capture is 98 %
for a 1 µm beam). The beam is centred, launched at its waist and transverse
(no $E_z$). §4.3's admission screens, model budget and passivity certificate are
not applied; they are phase 3's.

**Open questions.**

1. Should `with_input` accept a `SpatialPulse` on a general modal run, routed
   through the same incident launch, or wait for phase 2's target-bound launch?
2. Should `IncidentLaunch` carry the projection's electric-field residual as a
   labelled numerical diagnostic, or stay energy-only as now?
3. For a graded-index core, the on-axis law fixes $n_{\rm ref}$. Is that the
   intended reference, or should it be the index averaged over the beam?
4. The worked example `plot_04_graded_index_multimode` needs `incident=` and a
   re-run by hand. Its launched energy will fall to the captured share unless its
   requested energy is raised.

**Handoff to phases 2 and 3.** `SpatialPulse.incident` is the normalisation
`FieldPlane.from_pulse` should wrap, and `IncidentLaunch` is the precursor of
`CouplingResult.report`. Phase 3 replaces `_incident_launch` with `couple(...,
CouplingModel("scalar_paraxial", ...))`, adds the analytic and sampled inputs,
`propz` transport and the §4.3 admission, and should fold `IncidentLaunch` into
the report rather than keep both.

### Phase 2 report — plane exchange

**Delivered (2026-10-24, `feature/sep-0031-simulation-coupling`).**

- `src/solprop/coupling/` (P): `plane.py` (`FieldPlane` and its records), `port.py`
  (`InputPort`, `CouplingLaunch`, `same_basis`), `remap.py` (`regrid`,
  `RegridReport`) and `__init__.py`. NumPy and P modules only. The `__init__` holds a
  comment where the transport core's names are exported.
- `src/solprop/processing/planes.py` (X): `field_plane(output, plane=-1, *, z=None,
  source=None)`, re-exported from `solprop.processing`.
- `src/solprop/translate/coupling.py` (T): `frame_record` (the persisted frame integral),
  `port_metric` and `input_port` (fills an `InputPort` from a `KernelProblem`).
- `driver.Simulation.input_port(z=None)` and `with_input(launch)`; `metadata_for` writes
  the plane-reference record.
- Output schema 24: `RunMetadata.reference`. Kernel schema unchanged at 35: no protected
  contract field changed.
- `CouplingError` in `solprop.exceptions`; `coupling` in the package's submodules.
- Docs: architecture §3 (the X → T edges and how coupling crosses the layers) and §4,
  structure figure 1 (the new modules, with `coupling/transport` and `coupling/elements`
  dashed as planned), provenance and output pages, the API index, a draft
  `docs/guide/workflows/coupling.md` in the workflows index, a note in
  interior-launch, and the changelog.

**The `FieldPlane` contract** (frozen, `eq=False`; arrays owned, `writeable=False`):

| field | type and meaning |
|---|---|
| `kind` | `'modal' \| 'general_modal' \| 'radial' \| 'cartesian' \| 'planar'` (`planar` is defined but not yet produced) |
| `samples` | complex128 `(n_modes, n_pol, n_omega)`: the optics-convention raw DFT of each column referenced to the window centre, exactly as an output stores it |
| `grid` | the `TimeGrid` (layout, time samples, absolute ω lattice, carrier `grid.omega0`) |
| `units` | `'sqrt(W)'` (modal, general modal), `'V m'` (radial Hankel), `'V/m'` (Cartesian DFT) |
| `basis` | `PlaneBasis(kind, identity, labels, space)`: `identity` is a SHA-256 digest; `space` is the `RadialGrid`/`FreeGrid` of a spatial plane |
| `reference` | `PlaneReference(phase, delay, calibrated)`: absolute spectrum = samples × exp(i(φ + ωτ)), ω absolute; τ is the absolute time of the record's t = 0 |
| `geometry` | `PlaneGeometry(z, side, origin, normal, x_axis)`, right-handed (`y_axis = normal × x_axis`) |
| `metric` | `PlaneMetric(kind, matrix, admittance, weights, scale, flux, reason)`; kinds `identity`, `dense` (Q = conj(W), `(n, n, n_omega)`), `admittance` (`c k_z/ω` per column, Parseval weights, `eps0 c/2`, optional `SpatialFlux`), `unknown` (with a reason) |
| `exterior` | `'zero' \| 'unknown'` |
| `medium` | the `SpatialMedium` of a spatial plane when at hand, else None |
| `certificate` | read-only mapping: the raw `flux_plan["admission"]` dict with its `scope` |
| `nodal` | `NodalField(nodes, weights, electric, magnetic)`, `(n_nodes, 3, n_omega)` optics spectra, for a general modal plane read with `source=` |
| `provenance` | `PlaneProvenance(source, plane, z, lineage, history)` |

Methods: `omega`, `absolute()`, `spectral_energy()`, `energy()`, `node_spectra()` (a
spatial plane's E at its nodes), `from_pulse(pulse, grid, geometry, *, space, medium,
reference=None)` (reuses `SpatialPulse.build`) and `from_samples(samples, grid, geometry,
*, exterior, space=, medium= | basis=, metric=, reference=)`. Helpers:
`reference_at(record, z)`, `basis_identity(route, record)`, `space_record(space)`,
`lattice_identity(grid)` and `modal_basis_identity(modes, grid, z)`.

**The `InputPort` contract** (P value type, frozen): `route`, `grid`, `basis`
(identity, labels, space), `shape` `(n_modes, n_pol, n_omega)`, `geometry` (the
entrance z and side), `frame` (the receiving run's own frame accumulated from its path
origin to the entrance, without any inherited origin), `metric` (Q at the entrance:
identity for ordinary true-power coefficients, conj(W) for a declared general basis, the
axial admittance of the spatial plane measure), `reference_index` (real phase index
per bin, spatial media only so far), `nodes`, `node_weights`, `electric` and `magnetic`
(optics-convention `(n_nodes, 3, n_modes, n_omega)` per √W, filled for a general guide
that does not move), and `models` (`('same_basis',)`). `identity` digests the route,
lattice, basis, shape, z and side. `accepts(other)` names the first disagreement among
route, lattice, basis, shape, side, entrance (relative 1e-12), orientation and metric
(relative 1e-10). A `CouplingLaunch(port, samples, origin, model, provenance)` is bound
to its port; `internal()` gives the internal-convention array and `record()` the
`configuration["coupling"]` entry (`model`, `port`, `origin`, `source`).

**Plane-reference record (output schema 24).** `RunMetadata.reference` =
`{carrier, origin, parent, frame, basis}`. `frame` holds `frequency_origin` (the frame's
reference frequency on an envelope grid, zero on a field grid) and the exact Chebyshev
antiderivatives of the contract's frame tables `linear.beta0` and `linear.beta1`
(`phase`, `delay`, each `breakpoints` and `coefficients`). Then φ = φ₀ + Φ₀ − ω_o T and
τ = τ₀ + T. This uses the terms the route removes: `frame_quantities` and the spatial
`frame_quantities` return β₀ = 0 where the frame keeps it, so a field grid and an
envelope with THG carry only the linear term, measured from ω = 0 on a field grid and
from ω_ref on an envelope (`linear.py` `Frame`, `dispersion_operator`). `basis` is the
route and the basis identity at each planned plane (launch plane first).
`origin`/`parent` come from `configuration["coupling"]`.

**Deviations and why.**

1. *Extraction is `processing.field_plane`, not `FieldPlane.from_output`* (§5.1).
   P must not import X; extraction reads an `Output` and reuses
   `processing.modal`'s checked identities.
2. *The reference tables are outside the hash.* They are derived from the hashed problem.
   A resume on another build could evaluate them to a different last bit, and SEP 25
   forbids a bitwise resume key. The inherited origin, which carries information, is in
   `configuration["coupling"]`, inside the hash.
3. *An ordinary mode set's basis identity is its value at the plane*, recorded per planned
   plane: each mode's class, label, effective area and real n_eff on the active band at
   that z, quantised to 12 digits, plus the lattice. The full `ModeRecord` would make the
   exit of a 1 mm capillary a different basis from the entrance of a 2 mm one: a
   constant-pressure path records its span. A natural-step plane (not planned) has no
   recorded identity, so it gets an "unrecorded" digest that no port matches.
4. *`from_pulse` takes `space=`* beside the temporal `grid`, because the beam builder
   needs both.
5. *`same_basis` is a public function.* It is the minimal internal path the phase plan
   allows, needed by the guide page. Phase 3's `couple(..., CouplingModel("same_basis"))`
   should dispatch to it.
6. *Remap alias guard.* The direct sum is evaluated only on target bins within the source
   lattice's own span. Beyond it, the sum of samples spaced Δt repeats with period
   2π/Δt and returned aliased light. The first test of a cropped band found this:
   the energy rose from 1.0 to 1.48 μJ. Loss shares are of a **record norm**
   (column weights; vacuum-equivalent on a spatial plane, energy on an identity metric).
   The report's energies are physical.
7. *Planar planes are refused by name*: a `PlanarGrid` carries its slab modes and has no
   plain record to rebuild from.
8. *For a launch, `noise="keep"` means no noise*: the launch carries its source's
   realised field. Another floor is added only when a realisation is given.
9. *Runtime literal checks.* Membership checks on `Literal` fields were dropped. The test
   session's beartype rejects such values first, so the checks would be unreachable
   under coverage. Static typing covers them.
10. *Certificate.* On this branch, `flux_from_record` does not rebuild `AdiabaticAdmission`,
    so the plane carries the raw `flux_plan["admission"]` dict, read-only, with its scope.
11. *A general-modal plane reads E/H through the `electric_field` capability name*,
    since `field_plane` is not on the general-modal accessor whitelist.

**Verification run** (targeted, `uv run --no-sync pytest -n auto`; none fail):

- `tests/coupling/` (`test_plane.py`, `test_port.py`, `test_remap.py`): 48 passed.
- `tests/processing/test_planes.py`: 10 passed.
- `tests/driver/test_coupling_relaunch.py`: 13 passed, including the 8 medium-tier
  source/receiver pairs over `numpy_ref`, `numba_cpu` and native CPU.
- Schema-literal and record modules: 547 passed. Driver, output, processing and
  backend modules that read the schema or use `with_input`: 627 passed and 302 passed.
- `tests/test_structure_figures.py`: 278 passed; `tests/test_guide_snippets.py`
  (compiled): 19 passed. The coupling page's snippets run by hand in 3.7 s.
- Doctests of the changed modules: 62 passed.
- ruff, ruff format, basedpyright (0 errors), numpydoc and codespell are clean on the
  changed files.

Evidence:

- Remap against an independent double sum (no FFT) on equal, longer, finer, field and
  carrier-shifted targets, from envelope and field sources: within 64(N+N′)ε of the peak.
  Against the same pulse built on the target: 1e-12. Energy agrees to 1e-10. The copy
  path on an identical lattice is bitwise. A 40-sample delay moves the peak 40 samples
  later, and the absolute spectrum is invariant. A phase change is an exact phase. Two
  separated bands keep the gap at the tails' level (< 1e-12), and a sign-changing
  spectral zero follows the direct sum.
- Relaunch: 1 mm + 1 mm of an argon capillary (2 modes, field grid) against 2 mm. At
  rtol 1e-8 the uninterrupted run is 1.3e-10 from a 1e-11 reference and the relaunched
  one 1.4e-10. At rtol 1e-6 they are 1.3e-8 and 1.5e-8, so both follow the tolerance.
  The two references agree to 1e-12. Interior relaunch into the same run and a linear
  radial two-stage run (1e-10) also pass.
- A spatial plane's `energy()` equals the run's own `energy` statistic to 1e-12 (radial,
  Cartesian, uniform and gradient argon). A gradient plane carries its certificate.

`uv run poe docs` builds clean with warnings as errors.
Not run: the full gate, slow and coverage lanes (end of SEP, per AGENTS.md), and GPU.

**Limitations.** Only the identity model exists: a different basis, lattice or medium is
refused, not coupled. Ordinary-modal ports carry no profiles or reference index yet. The
remap's loss shares on a spatial plane are vacuum-weighted. Natural-step modal planes
cannot be relaunched by `same_basis`. Saved fields are taken as stored, and a
modified-noise run's auxiliary `noise_phase` is not reconstructed (§4.7). Planar planes
are refused.

**Open questions.**

1. Should the modal basis identity be a tolerance comparison of n_eff and A_eff, rather
   than a 12-digit digest? A value within a few ulp of a rounding boundary can split
   two identical guides. This is improbable but possible on gradient guides.
2. Should regrid's band loss be reported in physical energy, using the per-bin
   admittance, as well as in the record norm?
3. Should a modified-noise run's plane be refused, or reconstructed from its documented
   saved-field meaning (§4.7)? It is currently taken as stored.
4. Should `field_plane` join the general-modal accessor whitelist, so the capability gate
   names it?
5. `same_basis` refuses light outside the port's active band on an identical lattice.
   Should it crop with a report instead, as `regrid(crop_band=True)` does?
6. Should the architecture record `processing.mixtures`' existing import of
   `translate.flux` as a deliberate edge (now documented in §3), or move the plane-measure
   rebuild behind a D-layer accessor?

**Handoff to phases 3 and 4.**

- Phase 3 should add `couple(plane, port, model=CouplingModel(...))` with `same_basis`
  as its identity branch, and fold `IncidentLaunch` into `CouplingResult.report`. The
  ports need extending: ordinary capillary scalar profiles and the core-gas
  `reference_index`, weak-guidance step-index, and general modal E/H for moving guides
  (`translate.coupling.input_port` is the one place to fill them). `from_analytic` is
  still to be written. `CouplingLaunch` and `with_input`'s binding check are reusable as
  they stand.
- Phase 4 should keep `FieldPlane` immutable and return new planes. It should append
  to `provenance.history`, update `PlaneReference.calibrated` where an optical model
  drops an affine term, and re-resolve a spatial metric through `PlaneMetric.flux`.
  Mirrors and rotations should change `PlaneGeometry.normal`/`x_axis` together with the
  polarisation. The names belong at the marked comment in `coupling/__init__.py`, and
  the dashed boxes in `docs/dev/figures/build.py` turn solid when the files land.

### Phase 3 report — coupling

**Delivered (2026-10-24, `feature/sep-0031-simulation-coupling`).**

- `src/solprop/coupling/models.py` (P): `CouplingModel(kind, model_error=, basis_change=)`,
  `BasisChange`, `cone_angle`, `radial_rule` and `ScalarModeBasis` — the scalar profiles
  of an ordinary mode set: capillary HE1m as $J_0(u_mr/a)$ inside the core and zero
  outside, and a step-index fibre's HE1m as weak-guidance LP0m ($J_0(Ur/a)$,
  $J_0(U)K_0(Wr/a)/K_0(W)$, from the pole-free form of
  $UJ_1(U)/J_0(U) = WK_1(W)/K_0(W)$), frozen at the mode's declared profile wavelength as
  the run's profile is; closed-form normalisations and closed-form order-zero Hankel
  transforms (the Lommel integral inside, the $K_0J_0$ antiderivative outside, each
  checked against direct quadrature to 1e-15 relative), the exact whole-plane Gram and
  the Gram of the angular content inside a cone. A mode-averaged GNLSE fibre is
  unwrapped to the waveguide its dispersion forwards to.
- `src/solprop/coupling/coupler.py` (P): `couple(plane, port, *, model)` →
  `CouplingResult(launch, report)`; `CouplingReport` (Displayed) and `AdmissionItem`;
  `target_columns`; `PASSIVITY_ROUNDING`.
- `src/solprop/coupling/envelopes.py` (P): `PairEnvelope`, `CAPILLARY_ENVELOPE`,
  `STEP_INDEX_ENVELOPE`; their oracles are `tests/coupling/test_envelopes.py`.
- `src/solprop/coupling/outcouple.py` (P): `Outcouple(space, interface="matched",
  medium=|index=)` and `OutcoupleResult(plane, modal, synthesised, window)`.
- `coupling/plane.py`: `FieldPlane.from_analytic(grid, geometry, field, *, space,
  medium=|index=, support=, exterior=, reference=)` with `AnalyticRecord(support,
  domain_change, resolution_change)`; `FieldPlane.profiles` and `.analytic`;
  `from_pulse(..., index=)`; `from_samples(..., profiles=)`; `PlaneMetric.paraxial`.
- `coupling/port.py`: `InputPort.profiles` and `.contrast`; the public `same_basis` is
  removed (folded into `couple`); `check_same_medium` and `occupied_outside` are public
  helpers of the module.
- `translate/coupling.py`: `input_port(..., modes=, guide=)` fills the scalar profiles,
  the core's `reference_index` (the core gas of a capillary from
  `MarcatiliMode.core_index_minus_one`, a fibre's core), the step-index contrast, and for
  a general guide its contrast (`guide_contrast`) and admitted models.
- `processing/planes.py`: `field_plane(run, source=simulation)` attaches an ordinary
  modal plane's profiles, checked by configuration hash.
- Driver: `from_waveguide`/`prop_waveguide(..., incident=, coupling=)`;
  `_incident_launch` builds a radial plane of the beam in the core index and calls
  `couple`, installing the launch by `with_input` (so `configuration["coupling"]` is
  recorded) with the report as `configuration["launch"]`. `IncidentLaunch` is removed.
- Docs: the coupling chapter (models, budget and report, a Gaussian into a capillary, a
  radius change by direct overlap and through `Outcouple`, a solid-fibre overlap, the
  vector entry); general-modal, solid-fibres, supplied-guides and interior-launch pages;
  architecture §3 and §4 and the structure figure; the API index; the changelog.
  `benchmarks/guides.py` declares its model.

**Admission (§4.3) as implemented.** Each item is a relative energy discrepancy with a
scope (`field`, `map`, `model`); the counted items must sum to `model_error`, and the
items the SEP caps at $\epsilon_m/8$ are capped individually.

- *Angular.* $1-\cos\theta_* = \epsilon_m/8$, $\kappa_* = n_{\rm ref}(\omega/c)
  \sin\theta_*$. Source: a spatial plane's column energies (cone vs tail by $|\kappa|$),
  a modal plane's coefficients against its profiles' tail Gram; energy-weighted over
  bins. Target: the largest generalised eigenvalue of
  $(\epsilon_m/8)K + (1-\epsilon_m/8)T$ against the channel Gram $K$, maximised over
  occupied bins (scalar: closed-form transforms; general guide: node Fourier transform
  of the channels' $\mathbf e_t$ on a polar rule inside the cone, at up to five bins).
- *Contrast.* Scalar step-index pairs: $\Delta$ over both fibres' core and cladding
  against the target core; general guides: every region's law and a radial law sampled
  at 33 radii. Not applied to a capillary (item present, value `None`). A step-index pair
  whose $\Delta > \epsilon_m/8$ is admitted if `STEP_INDEX_ENVELOPE` covers it (the
  item notes the replacement); the envelope's bound is always counted.
- *Metric defects.* Target: $G = \operatorname{diag}(n_{\rm ref}/\operatorname{Re}n_{\rm
  eff})$ (times the Jones Gram) against $Q = I$; vector: the whitened extremes of $F$
  at the port's nodes against the run's declared $Q$. Source: $\sum|M_0 - Q_A|/\sum Q_A$
  over bins.
- *Closure.* $\delta^2 = \lambda_{\max}$ of the Gram of $\mathbf h_t - Y_0\hat z\times
  \mathbf e_t$ against that of $Y_0\hat z\times\mathbf e_t$, energy $2\delta+\delta^2$.
- *Passivity.* Largest whitened squared singular value of the map, $C Q_A^{-1}C^\dagger$
  against $Q_B$, over occupied bins and propagating columns; excess $\le 10^{-10}$ is
  rounding, up to the model's own metric, angular and closure items a reported model
  defect, beyond it a refusal. Never rescaled or clipped.
- *Numerical.* A spatial source's content in the outer fifth of its spectral window, an
  analytic plane's domain and resolution checks, the modal overlap quadrature (rule
  against its halving, plus each Gram against the exact one).
- *Report.* Incident energy in $M_0$ over the whole plane; launched energy in the native
  $Q$; captured $\mathbf a^\dagger G\mathbf a$ and uncaptured (unclipped) only for
  `scalar_paraxial`; the source's SEP 34 certificate carried separately; a zero field
  is a zero launch with `efficiency` `None`.

**Envelopes.**

- Capillary (`capillary-he1m-scalar-vs-marcatili-2026-10`): the scalar shares from
  sampled Cartesian planes against `MarcatiliMode.field` with its `normalisation` on an
  independent polar Gauss–Legendre rule; waists 0.3a–1.0a, displacement to 0.2a, tilt to
  1 mrad, 75 µm argon capillary at 800 nm with eight HE1m modes, and radius ratios
  0.5–2. Largest difference 1.8e-14 of the incident energy (the reference quadrature's
  rounding: the Marcatili HE1m field *is* the scalar profile); recorded bound 1e-12,
  numerical 1e-13; range HE1m count 1–8, radius ratio 0.5–2. *Corrected in phase 5:*
  because the reference is the scalar profile itself, this comparison bounds the
  quadrature and the implementation, not the omitted physics; it is now labelled a
  consistency check, and the envelope is re-measured against Marcatili's leaky core
  field (phase 5 report, item 1).
- Step index (`step-index-lp01-vs-vector-he11-2026-10`): LP01 overlap squared against
  $|F_{AB}|^2$ of the exact unit-power HE11 even fields
  (`StepIndexMode.electromagnetic_field`); core 1.45, contrast 0.001–0.01, V 1.5–2.4,
  radius ratio 0.7–1.43 at 1064 nm (constant indices: the overlaps depend on V alone).
  Largest difference 2.92e-4, at contrast 0.01, V 2.4 and ratio 0.7; the reference's
  refinement changes it by 7e-16. Recorded bound 3.0e-4, numerical 1e-12; LP01 only.

**Deviations and why.**

1. *The least-squares Gram is the exact one.* The fit uses the profiles' exact Gram
   (orthonormal, times the Jones Gram), not their transforms' Gram truncated to a plane's
   window: for a field the plane resolves the overlaps are exact by Parseval whatever of
   a profile lies beyond the window, so the coefficients are exact and the
   orthogonal-residual identity holds in the continuum. With the truncated Gram the
   capillary shares were biased by up to 7e-5 at 128² nodes.
2. *Weak-guidance step-index is coupled on the ordinary (GNLSE) route*, through the
   waveguide its mode-averaged fibre wraps, and only fibre → fibre: no envelope covers a
   beam into a fibre or a capillary ↔ fibre pair, so those are refused.
3. *The vector model has no recorded pair envelope.* The maintainer-agreed scope lists envelopes for the
   capillary and the solid scalar pair only; `vector_paraxial` reports an `envelope`
   item with value `None` and the reason, and relies on the contrast screen, the closure
   and the other items (open question 1).
4. *The vector passivity certificate covers the coordinates the field occupies*
   (scope `field`): a spatial plane's high-$\kappa$ columns are aliased on a guide's node
   rule and would show a spurious gain (0.38 on the test strand) through columns the
   field does not occupy.
5. *The driver's incident path builds the simulation twice.* It constructs the run with a
   zero input, takes its `input_port()`, couples, and installs the launch by
   `with_input` (sharing the translation); `configuration["pulse"]` and `["launch"]` are
   caller keys of that derived run, and `["coupling"]` is recorded.
6. *`from_pulse(index=)` keeps phase 1's measure.* With a reference index and no
   `SpatialMedium` (a guide's core), the beam is `SpatialPulse.incident` sampled on the
   grid, normalised in $M_0$; with a medium it is `build`, normalised in the axial flux.
   The difference is the beam's obliquity, reported as the source-metric item.
7. *Incident test guides changed.* The 1.45-in-air strand fails the contrast screen
   ($\Delta = 0.26$); the phase-1 incident tests now use a 4 µm core at a 0.01 step under
   `model_error=0.2`.
8. *Runtime `Literal` membership checks are not repeated* (`CouplingModel.kind`,
   `Outcouple.interface`), as in phase 2: the test session's runtime type checks reject
   them first, so the checks would be unreachable under coverage.
9. *`Outcouple` returns a `FieldPlane`, not a transport `Element`.* The transport core
   works on `SpatialState`, which the parallel transport branch is wiring to
   `FieldPlane`; `Outcouple` lives in its own module and is not exported from the
   transport section of `coupling/__init__.py`.
10. *The module is `coupler.py`, not `couple.py`.* The package exports the function
    `couple`, which would shadow a submodule of the same name: the API reference could
    not document both, and `solprop.coupling.couple` would mean two things.
11. *No schema change.* Neither the kernel schema (35) nor the output schema (24) moved;
    the report enters the configuration record as a dataclass.

**Verification run** (targeted, `uv run --no-sync pytest -n auto`; none fail except the
three pre-existing structure-figure tests named below):

- `tests/coupling/test_couple.py` (22), `test_envelopes.py` (10, medium included),
  `test_port.py`, `test_plane.py`, `test_remap.py`, `tests/processing/test_planes.py`,
  `tests/driver/test_coupling_relaunch.py` (medium included), `tests/driver/
  test_general_modal.py` (incident tests on all three CPU receivers),
  `tests/test_display.py`, `tests/test_pulses.py`, `tests/test_benchmark_guides.py`:
  all pass (869 + 9 in the last combined runs); doctests of the coupling modules pass.
- Evidence: §4.4 Gaussian formulas (waist, displacement, tilt, curvature) against a
  direct 2-D sum to 1e-10 and against two sampled planes' spectral overlap to 1e-9;
  Gaussian–Bessel shares against a 400-node Gauss–Legendre integral to 1e-11 for
  $w/a$ = 0.3, 0.5, 0.7, 1.0 (0.906 and 0.092 at 0.5); nested sets of 2, 4, 6 modes share
  their common shares to 1e-12, and doubling the window or the nodes moves them by
  1e-12; a 4a beam launches < 15 % with the uncaptured energy exactly incident minus
  captured; crossed polarisation and zero fields are exact zero launches; a rotated
  degenerate pair splits evenly with the same total to 1e-12; coherent two-mode sources
  add as fields and a gauge flip launches identically to 1e-13; a declared unitary basis
  change preserves energy to 1e-13 and a 1 % stretch is refused; a capillary's exit into
  a wider one by direct overlap and through `Outcouple` (radial 512 nodes, Cartesian
  128²) agree to 10× the synthesis window's representation error (measured 1.3e-8 and
  8e-7 relative); the synthesised plane carries the modal energy to 7e-8 (the matched
  exit's obliquity); a step-index pair's captured share is the LP01 overlap squared to
  1e-12.
- ruff, ruff format, basedpyright (0 errors), numpydoc and codespell are clean on the
  changed files. The coupling page's snippets run by hand in 5.5 s, the solid-fibres
  page's in 3.4 s (a 12 µm beam into the 25 µm graded-index guide launches 0.964 within
  a 0.032 budget).
- *Failing, pre-existing:* `tests/test_structure_figures.py` —
  `test_every_planned_module_is_still_unwritten[coupling/transport.py]` and
  `[coupling/elements.py]`, and `test_the_layers_figure_draws_the_whole_package`: the
  merged transport core's modules are still drawn as planned. This branch added its own
  four modules to the figure and redrew it; the transport branch owns turning its boxes
  solid.

Not run: the full gate, slow and coverage lanes (end of SEP), GPU. The guide snippets
were compiled only in the default lane and run by hand.

**Sources checked.** The LP0m eigenvalue equation and field (Gloge, *Appl. Opt.* **10**,
2252 (1971), eq. 15 with l = 0; Snyder and Love, *Optical Waveguide Theory*, §14-3) were
re-derived and checked numerically (U = 1.528 at V = 2); the Lommel and $K_0J_0$
integrals were derived and checked against quadrature; the capillary normalisation
against `MarcatiliMode.normalisation`. Not checked: Gloge's equation numbering against the
printed paper.

**Limitations.** The vector model's omitted facet reflection is bounded only through the
contrast screen and the closure. The general guide's angular item samples at most five
bins. The scalar profiles of a step-index fibre are its frozen-wavelength profile, as the
run's are; a fibre whose profile moves strongly across the band is represented as the run
represents it. `Outcouple` offers only the matched interface. Ports of general guides that
move (a sampled path) carry no E/H, so they admit only `same_basis`.

**Open questions.**

1. Should `vector_paraxial` require a recorded provider envelope (for example
   `StepIndexProvider` against an independent mode-matching calculation), as §4.3's
   "each provider/model pair" reads, or is the contrast screen with the closure enough?
2. Small-core weakly guiding fibres need budgets of about 0.1 because the LP01 field's
   angular spread is a tenth of a radian; is the §4.6 cone rule, designed for thin
   screens, the intended angular bound for a direct modal overlap?
3. Should a beam into a weakly guiding fibre (Gaussian → LP01) be admitted with its own
   envelope? The agreed scope admits one solid pair only.
4. Should the target-subspace angular item of a general guide be evaluated at every
   occupied bin rather than five?
5. Should `Outcouple` become a transport `Element` once the transport core takes
   `FieldPlane`, and should it offer a declared planar-interface model itself?

**Handoff to phases 4 and 5.** Merge with the transport branch: `coupling/__init__.py`
exports both sets; the structure figure's coupling box lists both sets of modules
(regenerate `layers.svg` after the merge); the coupling chapter's transport section and
this page's coupling sections are independent. Phase 5's tutorial can use
`field_plane` → `couple(..., scalar_paraxial)` for the
radial entrance field into the capillary; its $s_{11}$, $s_{12}$ come from the launch and
the report as the tests compute them.

### Phase 4 report — transport core

**Delivered (2026-10-24, `sep31/transport-integration`, on top of phase 2).** The
transport core of §4.6–§4.7 on `FieldPlane`: homogeneous propagation with finite and
far observation planes, bounded thin screens, explicit reflection and Jones frames, the
normal-incidence interface and the existing spectral optics. `Outcouple` (guide-exit
synthesis) belongs to phase 3 and is not part of this delivery.

- `src/solprop/coupling/elements.py` (P; NumPy and SciPy only):
  - `SpatialState`, the plain internal record every element acts on: a `TimeGrid`, a
    `RadialGrid` or `FreeGrid`, the internal-convention spectrum
    `(n_modes, n_pol ∈ {1, 2}, n_omega)` in the layout of `SpatialPulse.build`, a
    homogeneous `SpatialMedium` and a right-handed 3×3 local frame. It stays public in
    `solprop.coupling.elements` with the `Element` protocol, as the interface a new
    element is written against; it is not exported from `solprop.coupling`.
  - Elements (frozen; `apply(state) -> ElementResult(state, report)`): `FreeSpace`
    (`distance, medium, output, output_method, band_limit, frame_delay, frame_phase,
    evanescent_tolerance, sampling_tolerance, beyond_window`), `ThinLens`, `Aperture`
    (with `weighting`), `TransmissionMask`, `Alignment`, `PolarisationTransform`,
    `Mirror`, `PlanarInterface` and `SpectralTransfer`.
  - Records: `ElementReport`, `ScreenCertificate`, `SpectralCalibration`.
- `src/solprop/coupling/transport.py` (P): `transport(plane, elements, *,
  geometry=None, frame=None) -> (FieldPlane, TransportReport)` and `TransportReport`
  (the sequential ledger with `ledger_residual`, `reference_in`/`reference_out`,
  `frame` and `source_certificate`). The plain-state sequence is the private
  `_sequence`; the element-level tests use it through a conftest helper.
- `src/solprop/coupling/__init__.py`: exports `transport`, `TransportReport`, the nine
  elements and `SpectralCalibration`; nothing else changed there.
- `tests/coupling/`: `test_elements.py`, `test_transport.py` (element level, on
  `SpatialState`) and `test_transport_plane.py` (the `FieldPlane` path, including the
  run round trip); 46 tests.
- Docs: a self-contained "Transporting a plane between planes" section at the end of
  `docs/guide/workflows/coupling.md`; the API index row and an API page,
  `docs/api/transport.md`, for `TransportReport` (below); architecture §4; structure
  figure 1 with `coupling/transport` and `coupling/elements` solid; the changelog.

**The adapter.** No accessor was added to `plane.py`; the private helpers in
`transport.py` are `_state` (plane → state), `_reference`, `_geometry` and
`_frame_term`.

- *In.* Radial and Cartesian planes only. A modal or general modal plane is refused
  with a message naming `Outcouple`; a planar plane is refused by name. The plane must
  carry its medium (`field_plane(..., source=simulation)` or a constructor); the
  samples are converted with `fourier.spectrum_to_internal`, and the frame rows are
  `x_axis`, `y_axis`, `normal`. A plane whose exterior is `unknown` is refused when the
  sequence contains a `FreeSpace` (propagation depends on the whole field); local
  screens and spectral maps carry it.
- *Out.* `spectrum_to_optics` of the final state; `PlaneBasis.of_space` of the output
  grid; `PlaneMetric.from_flux(SpatialFlux.resolve(grid, space, medium, z, side))` in
  the output medium and grid; the exterior and the source certificate carried; one
  `provenance.history` entry per element (`transport[i]: <element>`, with
  `(reconstruction)` where flagged) and one for the frame source.
- *Reference.* Each gap removes `exp[-i(φ_f + ωτ_f)]` from the whole field and the
  reference gains `(φ_f, τ_f)`, so `absolute()` is unchanged by the removal. With
  `frame=` (a `solprop.linear.Frame`, normally `simulation.frame`) a gap without its
  own `frame_delay` removes `R L`, `R = β₀ + β₁(ω − ω_o)` of the gap's medium at the
  frame's reference frequency, with `β₀` only on an envelope grid that subtracts it
  and `ω_o` zero on a field grid: phase `(β₀ − ω_oβ₁)L`, delay `β₁L`, the law of
  `reference_at`. Without a frame only `β₁(ω₀)L` is removed and the report and history
  say so. Calibration flags: a `SpectralCalibration` with `removes_phase` (or
  `removes_delay`) True or None clears that term; False keeps it; a gap keeps the
  input's flags, since the removed term is known exactly.
- *Geometry.* Without `geometry=`, the plane moves by each gap's distance along its
  current normal (turned by a mirror's `outgoing`), `z` advances by the distances, a
  gap clears `side`, and the axes are the final frame's. A declared `geometry` is used
  as given after its normal and x axis are checked against the transported ones.

**Deviations and decisions, with reasons.**

1. **Errors.** Every refusal in `elements.py` and `transport.py` raises
   `CouplingError`, including grid-shaped ones (undersampled screens, a mirror image
   off the grid, wrong sample shapes): each is a refusal to admit a field or an element
   into a transport, not a defect of a grid object, so no borrowed class remained the
   right one.
2. **Frame source.** A `FieldPlane` records its reference, not the rate of its frame,
   so "the delay taken from the plane's reference frame" is the explicit `frame=`
   argument; its default reproduces the previous `β₁(ω₀)L`. The round trip below shows
   that `frame=simulation.frame` makes transport and continued propagation agree in
   stored samples and reference alike.
3. **Output methods** stay an explicit choice, `"angular"` or `"fresnel"`; nothing
   selects one. Both the out-of-window angular refusal and the wrap refusal of an
   angular output suggest `output_method="fresnel"`.
4. **Sampling and wrap now refuse.** A column at angle θ moves its rays by `L tan θ`;
   content whose rays pass the window half-width wraps (periodic) or reflects (radial),
   which is the same condition as sampling the propagator phase `k_zL` below its
   Nyquist rate in κ. Its reference-measure share above `sampling_tolerance`
   (default 1e-6) is refused on the same-grid and angular routes. `beyond_window=
   "remove"` declares the removal of those (column, bin) pairs as an angular band; it
   is booked as `removed` with `window_removed` in the diagnostics. The edge share stays
   a diagnostic on the same-grid route (field near a reflecting edge is not yet an
   error), and refuses on the angular-output route as before. `sampling_adequate` is
   gone.
5. **Aperture weighting — clarification.** `aperture_profile`'s `W` means what SEP 23's
   `ApertureSet.weighting` makes it mean (`translate/spatial.py`, `exponent = 1 if
   weighting == "quadrature" else 2`): `"amplitude"` (default) is a field transmission,
   power `W²`; `"quadrature"` is a power weight, field `√W`. `Aperture` takes
   `weighting` with that meaning, so a transported aperture and an in-run aperture
   statistic agree. §4.6's "intensity-defined aperture … explicit conversion from its
   intensity weight" is the `"quadrature"` case; the default follows the in-run
   statistics instead. The draft's unconditional `√W` is replaced.
6. **Doc drift, not fixed (`spatial.py` frozen).** `aperture_profile`'s docstring calls
   its result an "amplitude transmission" without saying that SEP 23's quadrature
   weighting uses it as a power weight.
7. **The function shadows its module.** §5.1 makes `transport` a member of
   `solprop.coupling`, so `solprop.coupling.transport` names the function and the
   generated reference documents the function on the module's page. As for
   `solprop.physdata.species`, the module's `TransportReport` is documented on a
   hand-written API page. The element module's two callable type aliases are private
   (`_Transmission`, `_SpectralFunction`), so the reference does not list
   undocumented targets.
8. The numerical-core decisions of the draft stand: per-bin energy repeats
   `spatial_energy`'s weighted sum (`_energy_spectrum`, checked against it to 1e-13);
   the reference measure `U₀` weighs by `Re n ε₀c`; the certificate's numerical term
   is `2δ + δ²` of the synthesis round trip plus the edge shares; tilt uses `k sin θ`;
   the mirror frame is `(p̂, ŝ, k̂)` with `p̂ = ŝ × k̂` (perfect conductor `r_p = +1,
   r_s = −1`); `PlanarInterface` is lossless at normal incidence with obliquity booked
   as approximation error; `SpectralTransfer` recognises phase-only models by type,
   refuses an efficiency on a lossy model, books zero bins as removed and flags
   negative lengths and inverse mirror phases as reconstructions.

**Verification actually run.** Targeted lanes only:
`uv run --no-sync pytest -n auto tests/coupling tests/test_structure_figures.py
tests/test_seps.py` (631 passed, 2 skipped), the doctests of the two modules, targeted
coverage of `coupling/elements.py` and `coupling/transport.py` (100 % of statements and
branches), `ruff check --fix`, `ruff format`, `ruff check`, `basedpyright` (0 errors),
`numpydoc lint`, `codespell`, `git diff --check` and `poe docs`.

| Check | Result (measured) |
|---|---|
| Round trip: `field_plane` at 1 mm of a linear radial argon run (25 µm waist, envelope grid, 64 radial nodes) → `transport([FreeSpace(1 mm)], frame=simulation.frame)` vs the run's 2 mm plane | stored samples agree to 5.6e-13 relative, reference phase and delay to 1e-12, energy to 1e-12; on-axis amplitude changes by 9 % over the gap; asserted at 1e-11 |
| The same without `frame=` | stored samples differ by the carrier phase (3.4e-2), absolute spectra agree to 1.0e-12 |
| The same on a field grid (by hand, not in the suite: 2.5 s) | 4.8e-13 with or without the frame (no `β₀` on a field grid) |
| Gaussian at L = 0.5, 1, 2 z_R: on-axis amplitude, Gouy phase, second-moment waist, curvature | within 3e-6 of the paraxial formulas (asserted 2e-5); wrap share ≤ 1e-6 |
| Fresnel vs angular output; vs `scipy.integrate.quad` of the Fresnel integral; Fresnel → Fraunhofer orders; Airy pattern; lens focus vs `λf/πw₀`; radial vs Cartesian | unchanged from the core (5.4e-5; rtol 2e-3; ratios 1.998 and 1.9995; 5.7e-3 of the peak; 1.3e-5; ≤ 1e-8) |
| Aperture `tanh` edge, both weightings | node samples are `W·E` or `√W·E` to 1e-8 of the field inside the radius; amplitude transmits less than quadrature by more than 1 % |
| Plane record | delays and phases accumulate (declared and default terms), a `TaylorPhase` keeps the reference calibrated, a `GratingPair` clears both terms and `absolute()` then refuses; a fixed `Frame` gives `(β₀ − ω₀β₁)L` and `β₁L`; a back-propagation restores the reference and is flagged in the history; a 90° fold turns the normal to +x and x to −z, moves the origin along the new normal and keeps the frame right-handed; a declared geometry is checked; an interface sets the medium and books 4 % reflection |
| Refusals | modal (naming Outcouple), planar, medium missing, unknown exterior before a gap, wrapped rays (with the `beyond_window="remove"` alternative reported), angular output at the source edge (suggesting `"fresnel"`), unknown weighting, every element refusal as `CouplingError` |

Not run: the full gate, the slow lanes, the whole-package coverage lane and backend
parity (nothing here crosses the backend seam). Sources: the Fresnel and Fraunhofer
integrals and the Gaussian-beam formulas are checked by the tests above; no new
constant or coefficient was transcribed.

**Limitations.**

- `fresnel_phase_bound` is conservative, often by orders of magnitude for a Gaussian.
- The wrap test is a ray criterion per column; it does not see a field that is
  already at the window edge (the edge share reports that on the same-grid route).
- The sampled phase-step check of a general mask cannot see a step beyond π; the lens
  and the tilt use their analytic gradients.
- The Fresnel route attenuates a lossy medium through `e^{ikL}` only.
- Cartesian Fresnel output is a dense per-bin separable sum, not benchmarked.
- A radial two-component state holds uniform polarisation only.
- The frame rate of a plane is not part of `FieldPlane`; without `frame=`, a plane
  written in a frame other than its medium's `β₁(ω₀)` gets a different (still exact)
  reference split.

**Open questions.**

1. Should `spatial.py` gain a public per-bin `spatial_energy_spectrum`, replacing
   `_energy_spectrum`?
2. Should a `FieldPlane` record its frame (or the run's frame record) so that
   `transport` needs no `frame=` argument?
3. Should `aperture_profile`'s docstring (item 6) be corrected when `spatial.py` is
   next open?
4. Should the same-grid edge share also refuse, with its own tolerance?

**Handoff.**

- Phase 3 places `Outcouple` in front of a sequence and adds it to the elements table
  in the manual's transport section and to `coupling/__init__.py`. The introduction of
  `docs/guide/workflows/coupling.md` still says transport follows later; it should be
  updated at merge with phase 3's own changes to that paragraph.
- `couple` takes the transported plane as it is: its metric is resolved, its reference
  and calibration flags are current, and `report.source_certificate` carries SEP 34's
  certificate separately.
- Phase 5 composes the eight §2 cases from these elements; a compressor result enters as
  `SpectralTransfer(callable, calibration=SpectralCalibration(...))`.

### Phase 5 report — integrated qualification

**Delivered (2026-10-24, `feature/sep-0031-simulation-coupling`).** The focused
tutorial and the graded-index worked example were delivered separately, on their own
sub-branch; this report covers the rest of phase 5.

1. **The capillary envelope, corrected** (an independent physics review found it).
   `CAPILLARY_ENVELOPE` compared the scalar J0 overlap with `MarcatiliMode.field`, which
   for HE1m is the same J0, so its 1e-12 bound measured the quadrature, not the omitted
   physics §4.3 asks for. Those two tests are now named as consistency checks of the
   closed-form transforms against quadrature. The envelope
   (`capillary-he1m-scalar-vs-leaky-marcatili-2026-10`) now compares the scalar profile
   with Marcatili's leaky core field $J_0(u_m(1 - i\delta)r/a)$,
   $\delta = |\nu_n|c/(\omega a)$, $\nu_n = (\varepsilon+1)/(2\sqrt{\varepsilon-1})$
   (Marcatili and Schmeltzer 1964, doi:10.1002/j.1538-7305.1964.tb04108.x), on an
   independent polar rule at the range's top $\delta = 10^{-2}$: Gaussian beams (waist
   0.3a–1.0a, displacement to 0.2a, tilt to $0.59/ka$, 1, 2, 4 and 8 modes) and
   coherent three-mode exits into eight modes at radius ratios 0.5–2. The discrepancy is
   the larger of the summed per-mode share changes and the least-squares captured share
   change: largest $2.9\times10^{-3}$ (equal radii, from the leaky modes'
   nonorthogonality), beams $1.8\times10^{-3}$; bound $6\times10^{-3}$ (safety factor
   two), numerical $10^{-10}$ (refining the rule changes it by $10^{-14}$). Halving
   $\delta$ halves the beam cases (first order, from tilt and displacement) and
   quarters the radius cases, so the allowance is the bound scaled linearly by the
   pair's energy-weighted $\delta$ (`PairEnvelope.scaled_by`, `allowance`): about
   $1.5\times10^{-3}$ for a 75 µm silica capillary at 800 nm (the reviewer's direct
   probe: $8.5\times10^{-5}$ at 75 µm, $3.4\times10^{-4}$ at 37.5 µm, both inside).
   `ScalarModeBasis.wall_term` carries $|\nu_n|$ from `MarcatiliMode.sampled_terms`;
   `ScalarModeBasis.leaky(omega)` gives $\delta$ per bin. A pair whose weighted
   $\delta$ exceeds $10^{-2}$ is refused (a 12 µm capillary at 800 nm is, at
   $1.57\times10^{-2}$); bins beyond the range in a covered pair add their incident
   share to the item in full.
2. **Other review items.** The modal overlap quadrature, an amplitude error, now
   enters the budget as $2\delta+\delta^2$ (§4.7). A step-index pair's V is checked at
   every occupied bin as well as at the profile wavelength; bins where either fibre's V
   leaves the envelope's range add their incident share to the envelope item. The
   angular items' text says that their cone term is the cone rule's allocation, not a
   measured defect. A new test pins the launch phase of a displaced, tilted beam against
   an independent quadrature of the complex overlaps (they agree to 2e-14 rad): no
   share can catch a sign error, and on every `FreeGrid` the origin phase is
   $e^{\mp i\pi m}$ for either sign, so `target_columns`' origin-phase sign is not
   observable on the grids the package builds.
3. **The certificate.** `processing.field_plane` takes the plane's path certificate
   from `flux_from_record(...).admission`, the rebuilt `AdiabaticAdmission` (path
   identity as `{"recorded": text}`), instead of the raw `flux_plan["admission"]` dict;
   `FieldPlane.certificate`, `CouplingReport.certificate` and
   `TransportReport.source_certificate` carry it, and the report prints its field error
   and reflection. Its scope (the linear background's forward field) is the record
   type's, stated in the docstrings. Phase 2 deviation 10 is resolved.
4. **The eight compositions** (`tests/coupling/test_workflows.py`, medium tier, 9 tests,
   10.6 s wall under four workers, the slowest 3.7 s). Each asserts incident = captured +
   uncaptured, uncaptured ≥ 0 per bin, a launch within the reported target-metric defect
   of the captured energy (never rescaled), the receiving run's entrance energy equal to
   the launch to rounding, the budget within `model_error`, and for transports a passive,
   closed ledger.
   - (a) A 75 µm, 3-mode capillary exit into a 100 µm, 6-mode one: direct overlap
     captures 0.99962, `Outcouple` → 2 mm argon → `couple` 0.99853; the launched run
     inherits the plane's reference. Source `numpy_ref`, receiver native CPU.
   - (b) 5 mm of 3 bar argon on a radial envelope grid (`numba_cpu`), 10 µJ at
     w/a = 0.5, into a capillary on native CPU: the fundamental share is 0.8715 linear
     and 0.8805 with Kerr.
   - Mandatory: 2 mm of 1 bar argon on a radial **field** grid with Kerr (THG on) and
     field plasma, 40 µJ, 30 µm: plasma absorbs 1.6e-3 (Kerr alone 1e-8). The harmonic
     band 230–300 nm carries 82 nJ; its incident energy matches the plane's band energy
     to 5e-6, the capillary captures 0.998 of it, the entrance stores the launch in that
     band to rounding, the entrance reference equals the source plane's, and the band's
     energy-weighted arrival time agrees to 1.1 as. Receiver native CPU.
   - (c) Step-index 2.18 µm into 2.57 µm (contrast 0.005, 1064 nm): captured share equal
     to an independent `quad` of the LP01 overlap squared to 1e-8 (0.99731); source
     `numba_cpu`, receiver native CPU.
   - (d) A Gaussian at w/a = 0.5: shares 0.906 and 0.092.
   - (e) An analytic field whose waist scales as 1/ω: per-bin captured share equal to a
     four-mode Gaussian–Bessel quadrature at that bin's waist to 1e-10.
   - (f) A Cartesian beam displaced 15 µm and tilted 2 mrad by `Alignment`: 0.792
     against 0.9994 centred; the same beam built directly agrees within 2e-3 (the tilt's
     index differs per bin); `Alignment` on a radial plane is refused.
   - (g) `Outcouple` then `FreeSpace(..., output_method="fresnel")` at 0.1, 0.2, 0.4 m on
     scaled grids: the scaled intensity patterns converge (the second mismatch below 0.6
     of the first); the Fresnel route books 2.1e-4 as approximation error.
   - (h) 20 mm of 3 bar argon at 100 µJ (bandwidth ×6.9) → `Outcouple` →
     `SpectralTransfer(Chain(TaylorPhase(-50 fs²)))` → a 4f relay (two 20 mm lenses, a
     300 µm tanh aperture in the Fourier plane, declared `beyond_window="remove"`) →
     an identical capillary on native CPU: efficiency 0.998, the compressor's ledger
     entry zero, its reference calibrated, the aperture's loss booked as specified
     loss, and the launched fundamental's peak power raised ×2.05.
5. **Documentation.** The coupling chapter: a "Composing simulations" section with all
   eight compositions, the chaining rules (lattices, `geometry=port.geometry`, separate
   budgets) and a runnable relay example; the corrected envelope text; the allocation
   and $2\delta+\delta^2$ notes; a stray duplicated paragraph removed. Conventions §4
   (plane reference, affine phase and delay, spectral-density scaling, metrics);
   accuracy §12 (how coupling uncertainties are kept apart); backends §9 (a coupling row
   and paragraph); the Luna page (Luna's single-overlap `GaussBeamPulse` against the
   per-bin coherent projection; nothing restored in either); cross-links from pulses,
   beams, reading output, polarisation and compression (general-modal already linked);
   the changelog. The kernel and output contract pages already record schemas 35 and 24.
   The package docstring's reference to `transport` now resolves.

**Deviations.** The tutorial is not part of this delivery (parallel sub-branch). The
capillary envelope's allowance scales with $\delta$, a new `PairEnvelope.scaled_by`
field, rather than holding one number over the range; the alternative, a constant
$6\times10^{-3}$, would overstate a 250 µm capillary's discrepancy tenfold. The
integrated compositions are medium tier by purpose (functional checks of short runs);
together they add about 18 s of CPU time to that tier.

**Verification run** (targeted, `uv run --no-sync pytest -n auto`; nothing fails):
`tests/coupling/` (all modules, including the new workflow, leaky-envelope, phase and
uncovered-band tests), `tests/processing/test_planes.py`,
`tests/driver/test_coupling_relaunch.py`, `tests/driver/test_general_modal.py`,
`tests/test_display.py`, `tests/test_benchmark_guides.py`,
`tests/test_structure_figures.py`, `tests/test_seps.py`, `tests/test_guide_snippets.py`
(compiled); the coupling page's snippets run by hand (4.4 s). ruff, ruff format,
basedpyright (0 errors), numpydoc and codespell on the changed files; `poe docs`. Not
run: the full gate, slow and coverage lanes, GPU.

**Sources checked.** Marcatili and Schmeltzer's leaky-mode core wavenumber
$k_ta = u_m(1 - i\nu_n/(ka))$ and $\nu_n$ for HE modes were taken as the package's own
`MarcatiliMode` uses them (`_cladding_term_and_derivative`) and as the reviewer stated them; the printed
paper's equation numbering was not checked. Luna's `GaussBeamPulse` launch
(`Interface.makeinputs`) was read to state the Luna difference accurately.

**Limitations.** The leaky-field envelope is a first measure of the omitted physics:
it does not include the field in the wall, the polarisation dependence of $\nu_n$
across EH/TE/TM families (not admitted to the scalar model anyway), or a lossy wall's
complex $\nu_n$ beyond its modulus. The step-index envelope still covers LP01 only.

**Open questions** (earlier phases' questions that remain, with recommendations; the
plain defects among them are fixed above or were superseded):

1. *(Phase 1 Q1, Q2)* Superseded: `couple` then `with_input` is the one path, and
   `CouplingReport` replaced `IncidentLaunch` with its items. Close.
2. *(Phase 1 Q3)* A graded core's on-axis index as $n_{\rm ref}$: keep it (it is the
   port's declared reference, and the vector model's metric item measures the rest).
3. *(Phase 1 Q4)* The graded-index worked example: handled on the tutorial sub-branch.
4. *(Phase 2 Q1)* Basis identity by digest: keep the digest; revisit only if a gradient
   guide splits on a rounding boundary in practice.
5. *(Phase 2 Q2)* Regrid band loss in physical energy: add it beside the record-norm
   share when the transport core's per-bin admittance is reused; not a defect.
6. *(Phase 2 Q3)* A modified-noise run's plane: refuse it by name until its auxiliary
   phase is reconstructed (recommended; currently taken as stored). **Resolved
   2026-10-24**: refused (A2 below).
7. *(Phase 2 Q4)* `field_plane` on the general-modal accessor whitelist: yes, but the
   whitelist feeds the schema-7 records, so it belongs with the next output-schema
   change.
8. *(Phase 2 Q5)* `same_basis` outside the band: keep refusing; `regrid(crop_band=True)`
   is the explicit crop.
9. *(Phase 2 Q6)* `processing` → `translate.flux`: record it as a deliberate edge (both
   `processing.mixtures` and `processing.planes` rebuild the plane measure there).
10. *(Phase 3 Q1)* A recorded provider envelope for `vector_paraxial`: recommended before
    Final, against an independent mode-matching calculation for `StepIndexProvider`.
11. *(Phase 3 Q2, Q3)* The cone rule for small-core direct overlaps and a
    Gaussian → LP01 envelope: maintainer decision; the cone rule is conservative there.
12. *(Phase 3 Q4)* The general guide's target angular item at every occupied bin: yes if
    its cost stays within a coupling's; measure first.
13. *(Phase 3 Q5)* `Outcouple` stays a separate synthesis step (it changes the
    representation), not an `Element`.
14. *(Phase 4 Q1, Q3)* `spatial_energy_spectrum` and the `aperture_profile` docstring:
    when `spatial.py` is next open.
15. *(Phase 4 Q2)* Record the frame in `FieldPlane`: yes, with the next output-schema
    change, so `transport` needs no `frame=`.
16. *(Phase 4 Q4)* Refuse the same-grid edge share with its own tolerance: recommended.
    **Resolved 2026-10-24**: refused beyond `edge_tolerance` (A3 below).
17. *(New)* Should a capillary's envelope range be extended past $\delta = 10^{-2}$
    (narrow capillaries in the infrared), with a re-measured bound?

**Handoff.** Final needs the tutorial's merge, the end-of-SEP full gate, slow and
coverage lanes, and the maintainer's decisions on questions 10, 11 and 17.

### Phase 5 tutorial report — nonlinear beam preparation and coupling into a capillary

**Delivered.** `docs/tutorials/free_space/plot_08_radial_capillary_coupling.py`
(focused tutorial, runs on every docs build), linked from
`docs/guide/workflows/coupling.md` (the Gaussian-into-capillary section),
`docs/guide/waveguides/capillaries.md` ("When to use it") and
`docs/guide/free-space/index.md`.

**Specification as run (§8.1).** Scalar, linearly polarised Gaussian on an envelope
grid; uniform argon at 3 bar and 293.15 K, the reference temperature of optmat's argon
polarisability (`species("Ar").susceptibility.reference_state`, recorded in the
output); 800 nm, 30 fs intensity FWHM; 20 mm gap; linear-focus 1/e² radius 37.5 µm at
the entrance; capillary radius 75 µm (w/a = 0.5), 5 mm long, eight HE1m modes. Kerr
only (`KerrResponse(gamma3("Ar"), thg=False)` in the gap; `kerr=True, thg=False,
plasma=False, raman=False` in the capillary). The beam is one `SpatialPulse` with
`propz=-20 mm`, built in the argon, for both runs.

**Strength from the material functions.** n0 − 1 = 7.7206e-4, n2 = 3χ3/(4ε0cn0²) =
3.0931e-23 m²/W, z_R = 5.527 mm; B_lin = 0.3 rad gives P0 = 0.3793 GW and
**U = 12.114 µJ**, P0/Pcr = 0.1238 (Pcr = 3.064 GW from `critical_power`).

**Results through `couple`** (`scalar_paraxial`, `model_error=0.05`; s_m = launched
energy of mode m in its native metric over `report.incident_energy`):

| | s11 | s12 | s13 | launched/incident | budget used |
|---|---|---|---|---|---|
| linear | 0.906180 | 0.092289 | 9.36e-4 | 0.99953 | 0.018 |
| Kerr | 0.882348 | 0.115623 | 1.68e-3 | 0.99971 | 0.020 |
| change | **−2.3832e-2** | **+2.3334e-2** | +7.4e-4 | +1.8e-4 | |

Peak on-axis intensity: linear 1.717e13, Kerr 1.956e13 W/cm². Wavefront change at the
800 nm bin (Kerr minus linear, masked below 1 % of the linear amplitude): +0.097 rad
on axis, +0.082 rad at r = w0, falling across the beam (the Kerr lens still
converging at the entrance). Capillary outputs (5 mm): linear HE11 11.060 µJ, HE12
0.987 µJ; Kerr-prepared HE11 10.790 µJ, HE12 1.252 µJ. In both cases HE12 hands energy
to HE11 along the capillary well beyond its wall loss: the capillary's own Kerr
intermodal coupling, identical physics in both cases.

**Agreement with an earlier scalar probe.** The probe's s11 = 0.90620 /
0.88237 and s12 = 0.09228 / 0.11562 differ from the `couple` shares by ≤ 2e-5. The
difference is the metric, not the field: the same planes give, for the linear run,
0.906180 (launched / M0 incident), 0.906188 (captured share in the common metric,
×n_ref/Re n_eff ≈ 1 + 8e-6) and 0.906191 (launched / the plane's axial-flux energy,
which differs from the M0 incident by the beam's obliquity, 1.2e-5). The changes
Δs11, Δs12 agree with the probe to 1e-5 of themselves.

**Independent confirmation.** A separate script evaluated each entrance field off the
nodes by the inverse Hankel series E(r) = Σ_j (w_j/2π) Ẽ(k_j) J0(k_j r) (it reproduces
`node_spectra()` to 1.6e-15), projected it on J0(u_m r/a)/(√π a|J1(u_m)|) with a
600-node Gauss–Legendre rule over the core, and normalised by the plane quadrature of
|E|² weighted by n_ref. Its common-metric shares agree with `couple`'s captured shares
to all eight printed digits (linear 0.90618793 / 0.09229346, Kerr 0.88235581 /
0.11562827).

**Refinement study** (scratch script, one variation at a time from the tutorial's
settings: radius 1 mm, n_r 256, window 300 fs, band 500 nm–1.4 µm (256 bins,
dt = 1.76 fs), rtol 1e-7, dz_max 1 mm in the study (0.25 mm in the gallery), eight
modes). δΔ is the change of the claimed difference relative to the baseline.

| variation | bins | lin s11 | lin s12 | Kerr s11 | Kerr s12 | δΔs11 | δΔs12 |
|---|---|---|---|---|---|---|---|
| baseline | 256 | 0.906180 | 0.092289 | 0.882348 | 0.115623 | 0 | 0 |
| radius 0.8 mm | 256 | = | = | = | = | +8.4e-12 | −5.9e-12 |
| radius 1.5 mm | 256 | = | = | = | = | −1.1e-11 | +8.1e-12 |
| n_r 192 | 256 | = | = | = | = | +6.7e-16 | −3.2e-16 |
| n_r 512 | 256 | = | = | = | = | 0 | 0 |
| window 200 fs | 128 | = | = | = | = | +2.2e-16 | +9.0e-16 |
| window 500 fs | 512 | = | = | = | = | −6.7e-16 | +2.8e-17 |
| band 400 nm–2 µm | 512 | = | = | = | = | +1.0e-15 | −2.1e-15 |
| dt halved | 256 | = | = | = | = | +1.1e-16 | −1.4e-17 |
| modes 4 | 256 | = | = | = | = | 0 | 0 |
| modes 6 | 256 | = | = | = | = | 0 | 0 |
| rtol 1e-6 | 256 | = | = | = | = | −2.7e-8 | +1.5e-8 |
| rtol 1e-9 | 256 | = | = | = | = | +4.6e-9 | −3.5e-9 |
| dz_max 0.25 mm | 256 | = | = | = | = | +1.7e-9 | −1.2e-9 |
| dz_max 0.1 mm | 256 | = | = | = | = | +4.6e-9 | −3.5e-9 |
| rtol 1e-9 + dz_max 0.1 mm | 256 | = | = | = | = | +4.6e-9 | −3.5e-9 |
| model_error 0.1 | 256 | = | = | = | = | 0 | 0 |

"=" is equal to the baseline at the six printed digits. Per-run uncertainty is the
root-sum-square over the five groups (domain, radial nodes, temporal, modes,
stepping) of each group's largest change; the linear run's is 1.6e-15, the Kerr run's
2.7e-8 (s11) and 1.5e-8 (s12). **Combined uncertainty: 2.7e-8 for Δs11 (1.1e-6 of
the change) and 1.5e-8 for Δs12 (6.2e-7 of it)**, each far below the §7 criterion of
one tenth; summing the two runs linearly instead changes nothing at that precision.
The step size, not rtol, is what limits the gap run here: with dz_max = 1 mm the
controller is capped and rtol alone does not refine it, so dz_max was varied too.
The modal span cannot move s11 or s12 (orthogonal profiles; the nested-set property
the phase 3 tests hold to 1e-12), and the recorded capillary envelope caps the span at
eight modes (`modes=16` is refused by name). Capture changes with the span (0.99945
with four modes against 0.99954 with eight, linear), as it should.

The 5 mm capillary outputs moved by ≤ 2e-6 µJ in HE11 and HE12 between eight and six
modes and between rtol at its default and 1e-8.

**Checks.** No collapse: P0/Pcr = 0.124 and the on-axis intensity rises smoothly by
14 %. Collar: the fluence in the outer tenth of the radial domain is 1.5e-30
(linear) and 3.5e-27 (Kerr) of the total. Aliasing: the coupling's numerical item
(content in the outer fifth of the transverse spectral window) is 7.9e-31 and
9.0e-28; the window and node refinements above confirm it.

**Omitted ionisation (offline).** optmat's recommended argon rate at 800 nm
(`OptmatRate("Ar", 800e-9)`, record `rate/default/Ar@800nm`, validated, sources
NIST ASD 2026, Tong 2002, Tong 2005) integrated over the real field of a 30 fs
Gaussian at the Kerr run's peak on-axis intensity, **1.96e13 W/cm², gives an ionised
fraction of 1.1e-7** (3.7e-8 at the linear 1.72e13 W/cm²). The electron density,
about 8e18 m⁻³, lowers the index by 2.2e-9 against a Kerr increase of 6.1e-6:
Kerr-only is defensible and the strength was not reduced.

**Beat length and the capillary.** Carrier formula 22.50 mm (0.222 periods in 5 mm).
From `MarcatiliMode` HE11/HE12 at 3 bar: L_beat 27.70, 22.50 and 18.95 mm at 650, 800
and 950 nm; 5 mm power loss HE11 2.2e-3 / 3.3e-3 / 4.6e-3 and HE12 1.1e-2 / 1.7e-2 /
2.4e-2.

**Meter reading.** The tutorial states that the energy is that through the launch
plane in the argon and points to `guide/free-space/flux.md` ("What energy means") and
the coupling chapter's `PlanarInterface` for converting a reading taken before a cell
window. No new manual text was needed: the flux page already says to multiply by the
window's transmission.

**Runtime.** 5.0 s in the Sphinx build (sphinx-gallery's own timing), 4.7–6.4 s
standalone, on the development Mac (gap runs about 0.3 s each, capillary runs about
0.1 s each). The whole `poe docs` build took 26 s with the other galleries cached.

**Docs build.** `uv run --no-sync poe docs` renders the tutorial; the build ends with
one warning, which is not from this work: `solprop/coupling/__init__.py`'s module
docstring references `solprop.coupling.transport.transport`, which Sphinx cannot
resolve (`py:func reference target not found`). It looks like the same
function-shadows-module problem phase 3 solved by naming its module `coupler.py`.

**Deviations and notes.**

1. The gallery's s_m are launched shares over the M0 incident energy, the
   definition the phase 3 tests use; the captured (common-metric) shares are larger
   by n_ref/Re n_eff (8e-6 for HE11) and are not printed.
2. The free-space gallery's `README.rst` still lists the older tutorials only; it
   was outside this task's file list.
3. The wavefront panel uses the 800 nm bin, masked below 1 % of the linear field's
   amplitude at that bin.

### Worked example `plot_04_graded_index_multimode.py` — re-run

The example now calls `prop_waveguide(..., incident=beam, coupling=CouplingModel(
"vector_paraxial", model_error=0.1))` instead of `transverse=`, reads its launch account
from `configuration["launch"]`, and says that the launched energy is the captured share
of the incident beam's (no longer renormalised to the declared energy). The
`project_transverse_field` preview and the `GeneralModalBasisWarning` filters are gone.
Its output was regenerated with `poe docs-examples plot_04_graded_index_multimode`
(10 min 23 s): incident 159.67 nJ, launched 0.99998 of it, budget 1.832e-2 of 0.1
(angular 1.25e-2, contrast 9.5e-3, closure 4.6e-3, passivity 1.1e-3); entrance shares
LP01 0.933, LP02 0.0625. The figures changed only in their last digits.

### Maintainer decisions — 2026-10-24

- **A2 — modified-noise planes are refused** (Phase 5 open question 6, from Phase 2
  question 3; resolved). `processing.field_plane` raises `ProcessingError` for a run whose
  `RunMetadata.noise.model` is `"modified"`, saying that the saved field leaves the
  one-photon background to the auxiliary `noise_phase`, which is not reconstructed into a
  plane, and suggesting `noise="input"` or `noise=None` for a source run. The check comes
  before the plane is selected, so every plane of such a run is refused; an input-noise
  run is read as stored, its background being in the saved field. Reconstruction
  according to the documented saved-field meaning (§4.7) stays possible later and would
  lift the refusal. Delivered: `src/solprop/processing/planes.py`,
  `tests/processing/test_planes.py`
  (`test_a_modified_noise_run_is_refused_by_name_and_an_input_noise_run_read`, a 0.5 mm
  capillary run), the coupling chapter and the changelog's phase 2 line.
- **A3 — the same-grid edge share refuses with its own tolerance** (Phase 5 open
  question 16, from Phase 4 question 4; resolved). `FreeSpace(edge_tolerance=1e-4)` is
  new; on the same-grid route (no `output`) a propagated field whose reference-measure
  share in the window's outer band (the declared collar, or the outer tenth of each
  half-width) exceeds it raises `CouplingError`, suggesting a larger window,
  `beyond_window="remove"` when the removal is not yet declared, or a Fresnel output
  plane. `edge_share` stays in the diagnostics; the angular-output route keeps its
  existing refusal at `sampling_tolerance`. The default was set from the shares the
  existing tests produce: the eight workflow compositions reach at most 6.3e-7 (the
  capillary exit into a wider capillary) and 3.6e-6 (the SPM, compressor and 4f relay
  composition, with its declared removal), and the well-resolved transport tests at most
  5.4e-6 (the sequential ledger); 1e-4 leaves a factor of about 20 above those and is met
  by a Gaussian beam on a collarless radial window of about 2.4 times its 1/e² radius.
  Three tests built deliberately near the edge now pass an explicit tolerance or assert
  the refusal: the sub-wavelength evanescent beam (1.4e-3 in the outer band before
  propagation), the aperture-then-gap order test (1.9e-3 after the declared removal), and
  the collarless-window test (1.6e-2 and 0.12 after removal, which now asserts the
  refusal and then admits it with `edge_tolerance=1.0`). Delivered:
  `src/solprop/coupling/elements.py`, `tests/coupling/test_transport.py`
  (`test_a_same_grid_field_in_the_outer_band_is_refused_by_its_own_tolerance`), the
  coupling chapter and the changelog's phase 4 line.

## Part III — History

- 2026-10-22: Drafted after auditing existing launch, projection, spatial flux,
  output and offline-optics interfaces. Selected forward coupling, spectral
  compressors, current free-space field models, controlled temporal remapping
  and uniform-medium flux. Required clean breaking changes with no legacy
  support and a mild-Kerr radial-input capillary tutorial. Identified cross-model
  projection and thin-screen flux as explicit pre-acceptance questions.
- 2026-10-23: Reviewed by Opus 5.5 (AI), with John Travers's human review recorded at
  his request. Part I is unchanged. The findings (R1–R10) and the maintainer's decisions
  on R1, R2 and R5 follow. Reviewed records completion of review, not acceptance;
  status remains Draft.
- 2026-10-23: Revised Part I in response to R1–R10, preserving the original review
  below. Split medium flux into SEP 34, proposed bounded interface models,
  moved the incident-energy defect to phase 1 and updated native-backend scope.
  John selected the 37.5 μm tutorial waist and full complex admittance for local
  flux traces. The response below records the resolutions and qualifications;
  status remains Draft.
- 2026-10-23: AI re-review of the revised Part I by Opus 5.5 (S1–S4), recorded
  after the response. No P1 finding remains; Part I unchanged by the reviewer.
- 2026-10-23: Maintainer decision on S2 recorded: phase 1 follows SEP 34's
  activation.
- 2026-10-23: Revised Part I for S1–S4 and the settled SEP 34 contracts. Added
  scoped path-certificate accounting, benchmark replacement of the weak-guidance
  screen and the corrected tutorial beat-length derivation. Retained SEP 34's
  full activation before the incident-energy fix. The response below records
  the rationale; no implementation or new propagation qualification is claimed.
- 2026-10-23: Accepted by John Travers after the review responses, with an
  explicit request to commit and merge both SEPs 31 and 34. Part I is frozen.
  SEP 34 is implemented first; this proposal's implementation record remains
  empty until its phases begin.
- 2026-10-23: In progress. Implementation started on a branch cut from SEP 34's
  implementation branch; it reaches main through its own pull request after
  SEP 34 has merged.
- 2026-10-24: Phase 2, plane exchange, implemented; report in Part II.

### AI review — 2026-10-23

**Assessment.** The direction is right and the physics is careful. Coupling through
a coherent field at a reference plane, with absolute incident scale and a refusal to
restore uncaptured light, fixes a real defect: `_projected_launch` today relaunches
the full declared energy however badly the channels capture the profile, as its own
docstring says. The derivations checked below are correct. The proposal is not yet
acceptable as written, for three reasons. It bundles a cross-backend change of
energy semantics with a host-only feature (R1). It removes a working capability,
radial gas-gradient runs, where a closed-form correction is available (R2). And its
two pre-acceptance gates, Q1 and Q2, ask for approval without proposing an answer
(R3). The tutorial's operating point may not show the effect it is meant to teach
(R4).

**Maintainer decisions during review.** John asked for the flux question to be
explained, then decided:

- **R1:** split the medium-flux prerequisite into its own SEP.
- **R2:** add the adiabatic amplitude term so that gradient runs keep working.
- **R5:** `energy=` means the physical energy (time-integrated axial Poynting
  flux) in the launch medium, because that is what an energy meter measures. He
  asked "What is the actual correct definition we should use for the flux";
  the answer is recorded under R5.

**Review basis.** Read all of Part I against SEP 0, the architecture, conventions,
the free-space model page and the current implementation. Checked:
`Simulation.with_input`, `_projected_launch`, `project_transverse_field`, the vacuum
½ε₀c factors throughout `solprop.spatial`, `fourier.energy_omega`'s endpoint
convention, `spatial.aperture_profile` and the `optics.Chain` docstring. Every
factual claim in §3 matched the code. Independently re-derived: the scalar flux and
its discrete Parseval form (§4.2), the least-squares capture identity (§4.3), the
Gaussian displacement and tilt overlaps, the J₀ normalisation (§4.4), the
spectral-density scaling (§4.5), the Fresnel/Fraunhofer kernel, thin-lens sign and
normal-incidence Fresnel coefficients (§4.6), and the tutorial's n₂, B-integral and
energy formulas (§8.1). The tutorial sizing reproduces: n₂ ≈ 3×10⁻²³ m²/W at 3 bar
gives P₀ ≈ 0.46 GW against the stated 0.443 GW. Peak intensity is about
1.1×10¹³ W/cm², where argon ionisation is negligible, so the Kerr-only model is
defensible. All four references resolve, via Crossref, to the stated authors,
titles, volumes and pages. Ran one numerical probe (R4). This is a design review;
no implementation was qualified.

Priorities: **P1** must be resolved in Part I before acceptance; **P2** needs a
concrete revision or recorded decision; **P3** is editorial. The drafting agent
should answer each point in a `Response to review` subsection, following SEP 0.

#### R1 — P1: Split the medium-flux prerequisite into its own SEP

**Sections:** 1, 3, 4.2, 6.3, 9, 11 (phases 1a–1c).
**Maintainer decision (2026-10-23):** split.

SEP 31 contains two different kinds of work. Phases 1a–1c change the energy
semantics of every free-space route. Today `spatial_energy`, the intensity and
fluence functionals, `SpatialPulse(energy=…)`, `spatial_noise` and the in-run
statistics all use the vacuum ½ε₀c factor. Phases 1a–1c replace it with the Y_z
weighting. The statistics live in backend kernels, so the change needs work on all
three backends, and it changes the physical results of solid-medium runs by about
the refractive index. SEP 5 also depends on it. Phases 2–5 are host-only
NumPy/SciPy operations between runs, and they need a consistent measure only at the
hand-over plane. That measure can be evaluated offline from a saved field without a
kernel change.

The bundle also creates a merge hazard. §11 calls the phases independently
mergeable, and §6.3 has incomplete backends "refuse the new contract by name" with
no vacuum fallback. Between phase 1a and phase 1c, main would refuse ordinary
spatial runs on the default compiled backend.

**Requested revision:** move §4.2 (including the R2 term), the launch, noise and
statistics consistency work, and phases 1a–1c into a separate SEP, "medium flux for
free space". Within that SEP, the host-side launch and noise normalisation and the
kernel statistics land together, or in an order that never leaves a backend
refusing a currently working run. SEP 31 then lists that SEP under **Requires**.
Its coupling and transport evaluate the plane measure on the host through the same
shared definition. Update SEP 5's dependency and the roadmap entries that currently
attribute the flux correction to SEP 31.

#### R2 — P1: Keep radial gas-gradient runs, with the adiabatic amplitude term

**Sections:** 4.2 (*Uniform media only*), 6.3, 9, 12 (Q6).
**Maintainer decision (2026-10-23):** add the term; it belongs in the new flux SEP (R1).

The radial route accepts a gas pressure `ZProfile`, for gas cells and jets (free-space
model, *Exact longitudinal operator*); Cartesian and line already refuse varying
media. As drafted, every operation needing the new flux contract refuses a
nonuniform medium, and no vacuum fallback remains. Energy-normalised launches and
energy statistics in those runs would therefore stop working. That is a regression
of a working capability, and it is avoidable.

For the scalar forward model in a medium varying slowly on the wavelength scale,
conservation of $Y_z|\hat E|^2$ per transverse/frequency column requires

$$
 \partial_z\hat E=\Bigl(ik_z-\tfrac12\partial_z\ln Y_z\Bigr)\hat E
 \;\Rightarrow\;
 \hat E(z_1)=\sqrt{\frac{Y_z(z_0)}{Y_z(z_1)}}\,
            e^{i\int_{z_0}^{z_1}k_z\,dz}\,\hat E(z_0).
$$

The amplitude factor is closed-form and diagonal. It joins the integrated linear
exponent that the gradient route already evaluates, so it costs nothing per step.
Paraxially it reduces to $\sqrt{n(z_0)/n(z_1)}$. It is the adiabatic (WKB) forward
approximation and neglects reflection from the gradient. For gases, Δn ≈ 10⁻³ over
millimetres, so that reflection is many orders below any tolerance. Check whether
the nonlinear source prefactor's local $k_z$ stays consistent with this choice of
propagated variable. As drafted, it uses the local value.

**Requested revision:** derive the term in the new SEP. Apply it on all three
backends. Qualify it as follows: a lossless gradient conserves the new energy to the
rounding budget, a uniform medium is unchanged, and a sharpening gradient tends
toward the transmitted share of an abrupt interface minus its reflection. Drop the
gradient refusal from §4.2, §9 and Q6. The remaining refusals cover only cases the
WKB model does not admit, such as abrupt steps that need an interface element.

#### R3 — P1: Q1 and Q2 need proposed answers, not open approval

**Sections:** 4.3, 4.6, 12 (Q1, Q2).

Both questions are marked "before acceptance", but Part I proposes no model for the
maintainers to approve. Accepting the SEP now would accept an open design, and
leaving the choice to implementation is exactly what §12's preamble forbids.

**Requested revision:** state the admitted cases concretely. For example, for Q1:

1. A scalar forward overlap in the common uniform-medium paraxial measure for:
   Gaussian or radial free-space fields into the ordinary capillary, capillary into
   capillary of a different radius, and weakly guiding solid fibres below a stated
   index contrast.
2. Provider E/H for general vector modes, with an E-only incident field closed by
   $\mathbf H=Y\hat z\times\mathbf E$ and a stated $O(\theta^2,\Delta n)$ error.
3. Everything else refused.

For Q2: the paraxial thin mask and lens in the local measure, with a defect bound
computed from the flux fraction beyond a stated angle where $Y_z$ departs from its
paraxial value by more than the tolerance, and refusal above it. Give the numbers or
the rule that fixes them. Q3 can then stay an implementation-phase budget.

#### R4 — P2: The tutorial sits at the stationary point of capture

**Sections:** 7, 8.1, 12 (Q5).

The linear-focus spot of 50 μm in a 75 μm capillary gives w/a = 0.667. Gaussian
capture into the EH₁₁ mode peaks at w/a ≈ 0.645 (η ≈ 0.981), so capture there is
nearly stationary in spot size. A direct quadrature of the §4.4 scalar overlap
gives:

| w/a | η₁₁ | η₁₂ |
|---|---|---|
| 0.645 | 0.9807 | 0.0052 |
| 0.647 (−3 %) | 0.9807 | 0.0048 |
| 0.667 (tutorial) | 0.9792 | 0.0020 |
| 0.687 (+3 %) | 0.9756 | 0.0004 |
| 0.500 | 0.9062 | 0.0923 |
| 0.515 (+3 %) | 0.9220 | 0.0764 |

At P/P_cr ≈ 0.15, the Kerr lens changes the entrance spot by a few per cent. Around
the tutorial's point that moves total EH₁₁ capture by only 10⁻³ to 10⁻⁴. That is
comparable with the numerical uncertainty that §7 requires the effect to exceed. The
EH₁₂ share, in contrast, changes by a factor of several, and at w/a = 0.5 the
fundamental's capture moves by about 1.6 % per 3 %.

**Requested revision:** choose deliberately. One option is to make the stationarity
the lesson: total capture is insensitive, while higher-mode content, wavefront and
spectrum change. The other is to detune the focus so that capture is first-order
sensitive. Name the quantity the convergence evidence must resolve in §7 and Q5,
rather than "the nonlinear change in modal capture".

#### R5 — P2: State what `energy=` physically means

**Sections:** 1, 4.2, 5.1, 9.
**Maintainer decision (2026-10-23):** physical energy in the launch medium.

John's question was the principled one. A user knows a beam's energy from a meter
and wants to model that energy. The answer should be in Part I, not left implicit in
the Y_z formula:

- Energy is the time-integrated axial Poynting flux through a plane. That is
  conserved, and it is what a meter measures.
- For a paraxial plane wave in a nonmagnetic medium, $S_z=\tfrac12 n\varepsilon_0c|E|^2$
  with the **phase** index (Re k/ω), not the group index. The energy density
  $\tfrac12\varepsilon_0nn_g|E|^2$ times $v_g=c/n_g$ gives the same result.
- The vacuum ½ε₀c|E|² equals the energy only for n = 1. It overstates the field by
  √n in glass, and by about 0.04 % (0.08 % in energy) in a few-bar gas. The modal routes already use
  true power, so the vacuum measure is the inconsistent one.
- A reading taken in air before an entrance window becomes the in-medium energy
  through a declared interface element (Fresnel or AR transmission), never
  implicitly.

**Requested revision:** add a short paragraph with these points to the new flux SEP
(R1), and a sentence to §5.1 here that `energy=` and a `GaussianBeam`'s energy mean
the physical energy at the declared launch plane in its medium. Add a note to §8 that
the manual explains the mapping from a meter reading.

#### R6 — P2: Fix the incident-energy defect early and on its own

**Sections:** 3, 5.1, 11.

The fix to `_projected_launch`, which stops it restoring missed energy, is a current
correctness defect. It does not depend on `FieldPlane`, regridding or transport. In
the phase table it sits inside phase 3, behind phase 2's plane-exchange machinery.

**Requested revision:** make the unified incident-energy launch (§5.1, *All spatial
incident-field entry points converge*) the first deliverable after the flux SEP, with
its own evidence: deliberate poor and zero capture, and no restored energy. Then
order the remaining phases.

#### R7 — P2: Interface gaps in §5

**Sections:** 5.1, 5.2.

1. `GaussianBeam` duplicates the Gaussian of `SpatialPulse`, while §5.1 says all
   spatial incident-field entry points converge. Say whether `GaussianBeam`
   replaces `SpatialPulse`'s Gaussian, wraps it, or is its incident-field record. A
   clean breaking change is allowed, so pick one.
2. The sketch in §5.2 uses `Simulation.input_port()`, `Outcouple`, `FreeSpace`,
   `SpectralTransfer`, `ThinLens` and `Aperture`. None of them is in the §5.1 table.
   List them with their modules and responsibilities, since the element vocabulary
   is the public surface users will write against.
3. Say whether `transport` accepts an existing `optics.Chain` directly (the frequency
   block) and how it reports that the chain's order is not significant while the
   spatial sequence's order is.

#### R8 — P2: The native-backend text is out of date

**Sections:** 3, 6.3, 11 (phase 1c), 12 (Q7).

SEP 29's first phase has already moved the compiled backend into this repository as
`native`; the free-space line-route text calls it that. The conditional "if SEP 29
has moved Kokkos…" is now settled, and the phase name "Kokkos CPU flux" uses the old
name.

**Requested revision:** use `native` (its CPU build), drop the conditional, and close
Q7. If R1 is taken, this moves with phases 1a–1c into the new SEP.

#### R9 — P2: The "complete" criterion is broad

**Sections:** 7, 11 (phase 5), 12 (Q6).

The integrated qualification requires all eight workflows of §2. Those include the
radial field-grid Kerr/plasma/THG transfer and an SPM → compressor → second-capillary
chain. They are good targets. But Final status then depends on the far field,
Jones/reflection frames and thin elements as well as the coupling core.

**Requested revision:** consider letting the SEP reach Final on the coupling and
transport core, with a clear list of what Final requires. Alternatively, state why
every workflow is needed for the first useful release. Either way, keep the field-grid
plasma/THG transfer, which is the case most likely to expose band and reference
errors.

#### R10 — P3: Editorial

- §4.2: the discrete rule uses the analytic-signal convention, which keeps the DC and
  Nyquist bins unscaled, as `fourier.energy_omega` does. Say that §7's "direct
  time/space integration" means $\tfrac12\int|E_a|^2$. A real-field $\int E^2$
  differs in those two bins.
- §4.3, §4.5: W denotes the flux pairing matrix, the least-squares Gram matrix and
  the per-frequency metric $W_k$. Distinguish them, or say that they coincide when
  M is the flux measure.
- §6.2 depends on SEP 28's memory bounds. Link SEP 28 there, or list it under
  **Requires**.
- §4.4: keep the sentence forbidding an asserted universal coupling optimum. The
  probe in R4 confirms that the value depends on the profile and the mode set.

### Response to review — 2026-10-23

The revised Part I addresses all ten findings. The original review above remains
the record of what was reviewed; its completion does not imply acceptance of this
revision. John selected the detuned tutorial and complex-admittance treatment
while the revision was being prepared.

- **R1 — Accepted.** [SEP 34](sep-0034-medium-flux.md) now owns spatial energy,
  launch, noise, statistics, adiabatic radial transport and their CPU-backend
  qualification. SEP 31 consumes that contract. SEP 34's implementation may be
  developed in private backend slices, but the public switch is atomic across
  NumPy, Numba CPU and native CPU: no intermediate merge changes the default
  convention while a supported CPU backend refuses existing runs. SEPs 5, 8
  and 9 and the roadmap now point to the appropriate owner.
- **R2 — Accepted with a correction to validation.** SEP 34 derives the
  `−½ ∂z ln Y` amplitude term, its endpoint form and its relationship to the
  existing radial linear operator. It preserves qualified smooth radial gas
  gradients. The proposed sharp-gradient Fresnel test cannot be a convergence
  target for this forward-only model: for n1 = 1 and n2 = 1.5 it predicts T = 1,
  while Fresnel gives T = 0.96 and R = 0.04. Even a gas-sized step to n2 = 1.0008
  has nonzero R ≈ 1.60 × 10⁻⁷. Instead, compare against an independently refined
  wave solution in the adiabatic range and refuse profiles outside the admitted
  error budget. Exact endpoint cancellation for represented weights is not a
  guarantee about interpolation or model error. No additional FFT is needed
  for the propagation correction, but derivatives, endpoint evaluation and
  local-flux reconstruction still have costs that must be measured.
- **R3 — Resolved by concrete proposals.** Section 4.3 names same-basis,
  scalar-paraxial and vector-paraxial models, their profile requirements and
  quantitative admission budgets. It requires provider/interface qualification
  beyond small-angle or weak-guidance screens; passivity alone cannot establish
  junction accuracy. Section 4.6 bounds the thin-screen discrepancy between
  local paraxial power and axial flux, including spectral tails, and specifies
  mirror coordinate frames. Unqualified pairs and exceeded budgets refuse
  explicitly. These are proposed acceptance contracts, not claims that the
  implementations have already met them.
- **R4 — Accepted; maintainer choice recorded.** The tutorial now uses 37.5 μm
  (w/a = 0.5) and B = 0.3 rad. It must resolve changes in incident-normalised
  fundamental and first higher radial-mode shares, each to a combined numerical
  uncertainty no larger than one tenth of the claimed change. Independent
  Gaussian/Bessel quadrature gives fundamental/second-mode shares about
  0.9062/0.0923 at w/a = 0.5 and 0.9220/0.0764 at 0.515. This supports detuning
  as a useful teaching choice, but does not predict the propagated nonlinear
  change. The review's comparison of a tiny change with numerical uncertainty
  remains a hypothesis until the propagation and refinement study is run.
- **R5 — Accepted and extended.** Incident energy is the physical energy crossing
  the specified plane in its medium. An external meter reading requires explicit
  interface transmission; phase index determines transparent plane-wave flux,
  while group index appears in stored energy density. A further issue arose
  during revision: in absorbing media the real admittance fixes integrated
  energy, but its imaginary part contributes to local broadband interference
  terms. John agreed to retain the full complex admittance for local traces and
  its real part for total energy and noise. SEP 34 derives both contracts.
- **R6 — Accepted.** Immediately after SEP 34, phase 1 fixes existing profile
  launches before introducing `FieldPlane`. The fix preserves the captured
  fraction using an explicit incident domain and measure; it does not merely
  delete the last rescaling after an aperture-normalised input. Phase 3 folds
  the same implementation into the unified coupling interface.
- **R7 — Accepted.** `SpatialPulse` remains the sole Gaussian/spatial pulse
  builder and `FieldPlane.from_pulse` consumes it. Section 5 lists modules,
  constructors, `Simulation.input_port` and the minimum transport elements.
  Existing frequency-only `Chain` values are accepted as one ordered transport
  block; their internal scalar multiplication does not reorder spatial optics.
- **R8 — Accepted.** The proposal uses the in-repository `native/` backend and
  its CPU build. Q7 is closed, and SEP 34 owns its flux implementation and parity.
- **R9 — Bounded scope retained and justified.** Section 11 makes Final depend on
  the eight named compositions within the restricted paraxial, forward and
  spectral-optics models. Each checks a distinct exchange or reference seam,
  so none is left implicit. General junction scattering, high-NA vector optics
  and spatially resolved compressors remain deferred. The short field-grid
  Kerr/plasma/THG transfer and capillary → compressor → capillary qualification
  remain mandatory.
- **R10 — Accepted.** SEP 34 states the analytic-signal DC/Nyquist convention and
  distinguishes its time-domain energy reference from a real-field endpoint
  identity. SEP 31 uses F for the E/H pairing matrix, G for the scalar
  least-squares Gram matrix and Q for the native power metric. Section 6.2 links
  SEP 28's memory contract. The warning against a universal coupling optimum
  remains in §4.4.

**Sizing evidence, not propagation qualification.** A fresh analytic calculation
with the current optmat 0.8.0 argon functions at 3 bar and 293.15 K gives
n0 ≈ 1.00077206, n2 ≈ 3.09311 × 10⁻²³ m²/W and zR ≈ 5.5266 mm for the
37.5 μm waist. The B = 0.3 estimate gives U ≈ 12.1136 μJ and P ≈ 0.379331 GW,
or approximately 0.1238 of the critical-power estimate converted to physical
power. The Gaussian sizing and independent scalar overlaps were recalculated;
no nonlinear propagation, omitted-response assessment, grid convergence or
backend qualification is claimed by this documentation revision. Those remain
required implementation evidence.

### AI re-review of the revision — 2026-10-23

**Assessment.** The revision resolves R1–R10. The split into
[SEP 34](sep-0034-medium-flux.md) is clean, and §4.3's three named coupling models
with an explicit `model_error` budget are concrete enough to approve. §4.6's
thin-screen bound is correct: $0\le U_0-U_Y\le d(E)$ holds because $Y_z/Y_0=\cos\theta$
on propagating columns, and the bound on the loss difference follows from the two
one-sided brackets. `SpatialPulse` remaining the only beam builder settles the
duplication. **No P1 finding remains.** SEP 31 can go forward for acceptance once
SEP 34, which it requires, is settled. The points below are refinements.

**Review basis.** Read the revised Part I in full, the response to review and the
diff against the reviewed text. Recomputed the revised tutorial sizing with the
stated optmat values: z_R = 5.527 mm, P₀ ≈ 0.380 GW and U ≈ 12.1 μJ, matching the
draft. Peak on-axis intensity rises to about 1.7×10¹³ W/cm², which is still far
below significant argon ionisation over 30 fs; §8.1 rightly requires it to be
reassessed. Checked the R2 correction with an independent transfer-matrix solution
of $E''+q^2E=0$ (recorded in the SEP 34 review).

**On the R2 correction.** The response is right, and the original request was
wrong. A forward-only adiabatic model transmits all the flux at a step. For
n = 1 → 1.5 the exact solution gives R = 0.0400, while the model gives T = 1. A
sharpening profile can only show the model leaving its admitted range; it cannot
converge to Fresnel transmission. The SEP 34 review (A1) adds a closed-form bound
that makes this precise.

#### S1 — P2: Use SEP 34's closed-form certificate for gradient sources

**Sections:** 4.2, 5.3, 7 (Refusals).

§4.2 and §5.3 refuse "a nonadiabatic gradient", which leaves the criterion to
SEP 34. The SEP 34 review (A1) recommends a closed-form, per-column certificate
in place of profile-family qualification. If it is adopted, refer to that rule
here rather than restating it. The coupling report then quotes the source plane's
certified forward-field error alongside the other model errors in the
`model_error` sum.

#### S2 — P2: Phase 1 may need only SEP 34's host reference

**Sections:** 11 (phase 1), 6.3.
**Maintainer decision (2026-10-23):** keep the present ordering. The
incident-energy fix lands after SEP 34's public activation, so there is no window
of mixed conventions. State that reason in §11.

As written, the incident-energy correction waits for SEP 34's full three-backend
activation. The defect it fixes is gross: a badly captured profile is relaunched at
the full requested energy, an error of order one. It also lives on the modal launch
path, not in a spatial kernel. The only physical input it needs is the incident
measure at the entrance plane, and that is SEP 34's host reference definition. In
a gas-filled capillary, the medium measure differs from the vacuum one by about
10⁻³.

**Recommendation, for maintainer decision:** either allow phase 1 to merge after
SEP 34's phase 1 (shared host reference), with that ≈10⁻³ inconsistency against
still-vacuum spatial observations documented until activation; or keep the
present ordering and state the reason, which is no window of mixed conventions.
Either is defensible. The SEP should say which was chosen.

#### S3 — P3: The weak-guidance screen is strict for ordinary fibres

**Section:** 4.3 (*Admission rule*).

With $\Delta\le\epsilon_m/8$, a standard step-index single-mode fibre (Δ ≈ 0.36 %)
needs $\epsilon_m\gtrsim2.9\,\%$. A 1 % model budget therefore refuses
fibre-to-fibre coupling between such fibres, even identical ones, where the scalar
overlap is exact by symmetry. The text already calls the allocation conservative
and requires independent benchmarks. Say whether those benchmark envelopes may
replace the Δ screen for a qualified provider pair. Otherwise the screen, not the
physics, sets the smallest usable budget.

#### S4 — P3: Name the mode beating in the tutorial capillary

**Section:** 8.1.

For a = 75 μm at 800 nm in near-unity-index gas, the EH₁₁–EH₁₂ beat length
$2\pi/\Delta\beta=8\pi a^2/[(u_{12}^2-u_{11}^2)\lambda]$ is about 22.5 mm. The
5 mm segment is roughly a quarter beat, and EH₁₂'s extra loss over 5 mm is
negligible. At w/a = 0.5, with about 9 % in EH₁₂, the output profiles therefore
show partial beating rather than modal filtering. That is fine, but the tutorial
text should say so, so that a reader does not mistake the beat phase for a
nonlinear effect. Comparing linear-prepared and Kerr-prepared launches over the
same segment already controls for it.

### Response to re-review — 2026-10-23

The original re-review above is retained. The following resolutions are now
in Part I; the proposal remains `Draft` pending formal maintainer acceptance.

- **S1 — Accepted with error units and scope made explicit.** Sections 4.2,
  5.3 and 7 refer to SEP 34's per-column real-index path certificate, including
  jump sides and refusal of nonconstant absorbing paths. Section 4.7 carries
  that certificate separately in the report. A relative field error δ implies
  an energy discrepancy bounded by 2δ + δ² in the same positive metric; this
  conversion and the map norm are required before adding it to an energy
  budget. The background-path certificate alone is not a bound on a nonlinear
  source simulation, which retains its separate qualification. Local flux uses
  SEP 34's retained forward-component closure.
- **S2 — Maintainer decision retained.** Section 11 explicitly requires SEP 34
  to be implemented and publicly activated first, including its required
  backend qualification. This avoids mixed launch and observation conventions.
  Its private host reference alone does not enable an earlier SEP 31 phase 1
  merge.
- **S3 — Accepted.** Section 4.3 allows an independently qualified provider-pair
  envelope to replace the conservative Δ screen over its declared range.
  It does not replace the angular/metric checks, passivity or total model-error
  budget. Identical checked bases already use `same_basis`, avoiding an
  artificial weak-guidance floor for coordinate recovery.
- **S4 — Accepted with a formula correction.** Section 8.1 derives
  $L_{\rm beat}=8\pi^2 n a^2/[(u_{12}^2-u_{11}^2)\lambda_0]$ from the scalar
  capillary beta expansion. The re-review's numerical 22.5 mm estimate is
  correct, but its printed formula is missing one factor of π. With the stated
  carrier index and J0 zeros, the static estimate is 22.5045 mm, so 5 mm is
  about 0.222 periods. The tutorial explains partial beating, uses the package's
  HE_1m labels and requires evaluation of actual dispersive beta and wall loss
  when the plots are implemented.

All rereview points have proposal-level answers. Grid/mode convergence,
interface qualification, the retained short field-grid Kerr/plasma/THG transfer,
the compressor workflow and the tutorial's nonlinear signal-to-error margin
remain mandatory implementation evidence, not unresolved design choices.
