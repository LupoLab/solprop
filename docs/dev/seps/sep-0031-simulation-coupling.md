# SEP 31 — Coupling simulations and transporting ultrafast fields

| | |
|---|---|
| **SEP** | 31 |
| **Title** | Coupling simulations and transporting ultrafast fields |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-10-22 |
| **Resolution** | — |
| **Requires** | [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md): plane storage and input replacement; [SEP 23](archive/sep-0023-far-field-aperture-spectra.md): aperture definitions; [SEP 25](archive/sep-0025-rounding-level-reproducibility.md): numerical qualification |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Connect solprop simulations through a coherent field at a reference plane, spatial
synthesis and projection, linear optical transport, and a validated launch into
the next simulation. These four blocks cover coupling between hollow or solid
guides, incident Gaussian and arbitrary fields, nonlinear beam preparation before
a capillary, outcoupling to an observation plane, and compression between stages.
The incident field retains its absolute scale: light that is not captured is not
restored by normalisation. A prerequisite replaces the vacuum spatial energy
functional with the uniform-medium scalar axial-flux measure, consistently in
launches, observations and noise. Qualification separates physical attenuation,
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
adaptive spatial meshes; a new distributed workflow scheduler; and smooth-gradient
amplitude transport. Nonlinear gaps use the existing propagation routes. Field-grid
Kerr/plasma/THG is a qualification case; it is not the teaching tutorial's model.

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

[SEP 5](sep-0005-solid-ionisation.md) derives a scalar axial-flux prerequisite for
solid ionisation. This SEP owns its uniform-medium implementation for the existing
spatial routes and the clean change of defaults; SEP 5 consumes it without a
second normalisation option. [SEP 8](sep-0008-quadratic-crystal-optics.md) retains
its different full-vector crystal E/H contract and interface solver.
[SEP 9](sep-0009-propagation-statistics.md) measures fields within a propagation
route; this SEP supplies transported fields and corrects existing spatial flux
semantics without taking on SEP 9's additional measurement catalogue.
[SEP 21](sep-0021-absorbing-boundaries.md) owns wider production time windows and
escaped-field models; remapping a saved plane cannot recover already absorbed light.
[SEP 29](sep-0029-compiled-backend-distribution.md) may relocate the Kokkos source;
qualification follows that backend's then-current home rather than duplicating it.

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

#### 4.2 Uniform-medium scalar axial flux

Retain physical E as the propagated state, including the radial transform's V m
units. For a homogeneous isotropic medium, let

$$
 k(\omega)=n(\omega)\omega/c,\qquad
 k_z=\sqrt{k^2-|\boldsymbol\kappa|^2},\qquad
 Y_z=\varepsilon_0c^2\frac{\operatorname{Re}k_z}{\omega}.
$$

Use the existing forward branch on admitted propagating support and the same
$\varepsilon_0,c$ constants as the source prefactor. This avoids a discrepancy
from independently rounded values of $\mu_0$ and $\varepsilon_0$. Multiplication
by $Y_z$ in transverse/frequency space defines a real spectral operator
$\mathcal Y$. With analytic fields $\mathbf G_a=\mathcal Y\mathbf E_a$,

$$
 S_z(\mathbf r,t)=\frac12\operatorname{Re}
    [\mathbf E_a(\mathbf r,t)\cdot\mathbf G_a^*(\mathbf r,t)],
 \quad F(\mathbf r)=\int S_z\,dt,\quad U=\int F\,dA.
$$

This follows the scalar equation's axial flux. A monochromatic paraxial wave
has $Y_z=n\varepsilon_0c$ and hence $S_z=n\varepsilon_0c|E_a|^2/2$.
Broadband or angularly broad fields require the spectral multiplier, not a
single index correction after squaring the time-domain field. This is not the
full TE/TM metric of a tightly focused vector Maxwell field.

To state the discrete rule, let $\hat E_{jpk}$ denote the stored raw temporal DFT
at transverse spectral column j, component p and frequency k, and let $v_j$ be
the existing transverse Parseval weights. Set $\gamma_k=1$ on an envelope grid;
on a real-field grid set $\gamma_k=4$ on interior positive bins and 1 at the two
real endpoints. Then

$$
 U=\frac{\Delta t}{2N_t}
   \sum_{j,p,k}v_j\gamma_kY_{z,jk}|\hat E_{jpk}|^2.
$$

The matching local fluence uses synthesis of E and $\mathcal Y E$ before their
product. It need not be positive at every point or instant. Width/FWHM operations
that require a positive profile must report an unsupported signed profile rather
than clipping it or silently measuring $|E|^2$ instead. Electric-field norms may
remain explicitly named field diagnostics, not physical energy.

Multiplying the existing scalar current prefactor
$-1/(2\varepsilon_0c^2\operatorname{Re}k_z/\omega)$ by the energy weight cancels
the admittance: spectral work is $-\operatorname{Re}(\hat E^*\hat J)/2$ in
analytic units. This independent work identity is a qualification target beside
Parseval; no change to ionisation's physical V/m argument follows from a new
measurement. For a uniform passive medium, propagation multiplies each column
by $\exp(i k_z L)$ and its energy by $\exp(-2\operatorname{Im}k_zL)$.

DC carries no optical power definition through this $1/\omega$ expression.
Require zero occupied DC, cutoff and nonpropagating support for energy-normalised
coupling launches, and a margin from cutoff established by conditioning. Exact
identity transfers on an already admitted grid retain its structural endpoint
rules. Refuse an occupied Nyquist endpoint when a representation change cannot
preserve its real-field meaning. Numerical evanescent attenuation in an existing
propagation is not a physical port or a substitute for an interface calculation.

Use the same measure for energy-normalised `SpatialPulse`, spatial noise, spectra,
fluence, on-axis flux and flux-derived statistics. A requested launch energy fixes
the integral once over the whole beam, never again after coupling. Normalise a
noise draw so that each admitted transverse/frequency degree of freedom carries
its declared photon energy in this metric; zero or ill-conditioned weights cannot
be repaired by division. Audit controller norms and name their measure explicitly.
The hybrid planar route needs its own modal/FFT metric, not an extra bulk index
factor multiplying already power-normalised slab amplitudes.

**Uniform media only.** A z-dependent $Y_z$ also changes amplitude transport in a
lossless forward model. Merely changing diagnostics on the present varying-gas
radial operator would not establish energy conservation. This SEP does not add
that propagation term. An operation requiring this qualified spatial contract
refuses a nonuniform medium; no legacy normalisation or hidden constant-medium
substitution bypasses the refusal. Smooth gradients remain a separate proposal.

#### 4.3 Projection, power and an interface are different objects

For a lossless forward vector basis, define the Hermitian flux pairing

$$
 \langle\psi_m,\psi_n\rangle_F=
 \frac14\int(\mathbf e_n\times\mathbf h_m^*
             +\mathbf e_m^*\times\mathbf h_n)\cdot\hat{\mathbf z}\,dA,
 \qquad W_{mn}=\langle\psi_m,\psi_n\rangle_F.
$$

The coordinates of a represented field satisfy $\mathbf a=W^{-1}\mathbf B$,
where $B_m=\langle\psi_m,(\mathbf E,\mathbf H)\rangle_F$. The recovered
power is $\mathbf a^\dagger W\mathbf a$. The package's existing E-only shortcut
doubles one half of B for a forward superposition in its admitted lossless basis.
Its exact recovery on that subspace does **not** prove that an unrelated incident
beam has the target guide's magnetic field. Use incident H when available; an
E-only input must declare its forward closure and validity. Do not reuse a
nonlinear source-projection matrix to recover field coordinates.

For scalar fields in a common positive power space, let S contain target basis
columns and M the quadrature/power measure. Weighted least squares gives

$$
 W=S^\dagger M S,\qquad
 \mathbf a=W^{-1}S^\dagger M\mathbf f,\qquad
 \|\mathbf f\|_M^2=\mathbf a^\dagger W\mathbf a+
                 \|\mathbf f-S\mathbf a\|_M^2.
$$

The last equality follows because the residual is M-orthogonal to S. Only in
such a common positive space does it identify a nonnegative uncaptured share.
The existing electric-field residual and an arbitrary E/H flux projection do
not automatically satisfy this identity. A basis outside a common admitted power
space needs its declared interface approximation, not an assertion of orthogonal
energy loss.

Use forward overlap as an explicitly named approximation for weakly reflecting,
matched-admittance guide junctions and aperture launches. Preserve the ordinary
capillary model's approximations; its leaky wall attenuation does not supply a
complete exterior E/H solution. Full-vector bound-guide coordinate recovery is
admitted where the provider and closure are qualified. Strong index contrast,
unresolved facet near fields, cutoff and a lossy/nonorthogonal provider without a
derived dual are not silently treated as exact butt splices. Reference [1]
distinguishes modal reciprocity from the full boundary-matching problem.

For a proposed passive finite map C between native metrics $W_A,W_B$, require

$$
 C^\dagger W_BC\preceq W_A
$$

within a stated numerical and approximation budget. Compute a conditioned
whitened-map singular-value or equivalent quadratic-form certificate. This is
necessary, not sufficient, evidence of a correct interface. It does not license
scaling a failed map down until it passes. Refine quadrature or reject the model;
do not cap efficiency at one. Approval of the scalar/vector bridge and its
admissible junction cases is the pre-acceptance question Q1.

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
(1 at endpoints). Then $\Delta\omega\sum f_k^\dagger W_kf_k$ is the
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

**Thin-screen qualification matters.** A pointwise mask with $|A|\le1$ is
contractive in a local paraxial power measure, but not automatically in the
nonlocal $\mathcal Y$ measure, since mask multiplication and $\mathcal Y$ do not
commute. Sharp edges also generate high angles. Likewise, a thin phase factor
does not model the entire E/H boundary condition. State the paraxial approximation
and independently bound its angular/flux defect, or refuse the case; do not label
every difference of two full-metric energies as absorbed screen energy. Q2 gates
this model before acceptance. The optical-ordering and loss reports retain this
distinction.

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
| `GaussianBeam` | Incident Gaussian temporal/spatial specification, curvature, displacement and Jones vector, normalised once before coupling |
| `regrid(plane, grid, ...)` | Controlled temporal representation/window/band conversion with its report |
| `transport(plane, elements, ...)` | Apply an ordered sequence of spatial and spectral elements; return the resulting plane and report |
| `couple(plane, target, ...)` | Project into an explicit target port; return `CouplingResult` |
| `CouplingResult` | Target-bound `launch` and a structured `report`, including provenance and numerical/model certificates |
| `Simulation.with_input(result.launch)` | Validate and install the already normalised target launch |

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
                 model="forward_overlap")
second_run = second_simulation.with_input(coupled.launch).run()
print(coupled.report)
```

`Outcouple` synthesises the guide's field and applies an explicitly selected
interface approximation; it is not an implicit unit-efficiency Fresnel solver.
For direct guide-to-guide transfer the same `couple` operation evaluates the
overlap without a transverse propagation grid. `SpectralTransfer` adapts existing
`optics` objects and compression results. Preserve `optics.Chain` as the
frequency-only multiplication; an ordered spatial sequence must not inherit its
statement that element order is immaterial.

The radial-input workflow replaces `exit_plane` with the output of an ordinary
radial run and couples at the capillary entrance. Gaussian, analytic and sampled
fields replace only the source constructor. To obtain a far-field observation,
stop after transport to the specified output plane; do not project onto a guide.
A nonlinear gap is a spatial simulation between two such operations.

#### 5.3 Refusals and outputs

Refuse unsupported medium gradients, missing mode profiles or references,
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
new propagation backend for it. Hot spatial flux/statistics/noise support does
cross the seam and requires backend work. T passes immutable arrays/factored
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
Factored $Y_z$ data should preserve the spatial memory bounds of SEPs 22 and 28;
do not add a permanent transverse-by-frequency array when the root can be
evaluated from existing separable coefficients in a bounded workspace.

Certificates account separately for quadrature, basis conditioning, transverse
and temporal truncation, transform sampling and model approximation. Strict
admission must be expressed in physical/numerical error targets, not an arbitrary
universal percentage of missed light. Uncaptured physical power may be large and
perfectly valid; numerical uncertainty must be small enough for the requested
conclusion.

#### 6.3 Backends and schemas

Implement uniform spatial flux on `numpy_ref`, then `numba_cpu`, then Kokkos CPU.
Each incomplete backend refuses the new contract by name. Update in-run spatial
statistics and offline equivalents together, including far-field energy weights
within their existing measurement geometry. Keep a beam-coordinate diagnostic
distinct from axial flux where that distinction is already physically meaningful;
do not silently redefine all six SEP 23 outputs as the same observable.

New flux/reference records or protected driver changes bump `SCHEMA_VERSION` and
its literal tests in the same phase. Saved plane references, metric identity and
meaningful lineage require an output-schema bump. Current-schema read/write and
resume tests are required, but old-file readers/migrations are not. If SEP 29 has
moved Kokkos into `solprop-native`, make the same changes there instead of starting
parallel changes in the old distribution.

Offline host operations support outputs from any qualified CPU backend. Test
NumPy-created inputs launched on Numba/Kokkos and the reverse. GPU acceleration
of the offline maps is not required for this proposal; separately qualify GPU
propagation and new native flux kernels on a device under the standing policy.
No host fallback is hidden inside a device statistic.

### 7. Verification and validation

Use [testing](../testing.md), [accuracy](../accuracy.md) and SEP 25. A matched
discrete operation uses a rounding budget based on reduction length, transform
error and conditioning. A changed grid, modal truncation or propagation uses
independent refinement and an accuracy envelope. Do not demand bitwise agreement
between independent backends. Copies, hashes, mode labels, explicit no-ops and
hard zeros retain exact structural tests.

| Case | Comparison and acceptance evidence |
|---|---|
| Uniform spatial flux | Direct time/space integration versus spectral Parseval on radial, Cartesian and line grids; monochromatic paraxial index factor; signed local-flux cases; current work and passive material attenuation |
| Launch and noise | Requested incident energy before coupling, captured energy after coupling, and photon energy per admitted noise degree of freedom; reject zero/cutoff metric divisions |
| Identity and gauge | Same-guide recovery, rotated degenerate bases and coherent two-mode interference; physical E/H and energy invariant under a simultaneous basis/amplitude transformation |
| Scalar overlap | Gaussian waist/displacement/tilt/curvature formulas and independent Gaussian–Bessel quadrature; deliberate poor and zero capture, with no restoration of energy |
| Cross-guide coupling | Different radii, mode sets and supported solid guides; direct overlap versus synthesis/project route; passive-map certificate and explicit failures outside the admitted interface model |
| Modal truncation | Retain successively larger nested mode sets; distinguish quadrature convergence from captured-subspace convergence and finite-domain effects |
| Spatial transfer | Radial versus Cartesian for a symmetric beam; displaced/tilted input on Cartesian; refusal of symmetry destruction on a radial grid |
| Linear transport | Gaussian diffraction and focus, independent Fresnel/Bessel evaluation, aperture diffraction, thin-element order, reflection/Jones signs, and finite-distance convergence to the Fraunhofer limit |
| Spectral optics | Reuse existing element references; explicit $\lvert H\rvert^2$ attenuation, phase-only invariance in its admitted metric, no double Fresnel factor and an efficiency report that changes with spectral reshaping |
| Temporal conversion | Equal-bin scaling, unequal windows/sample spacings, delays, CEP, carrier changes, spectral zeros and separated bands; field/envelope round trips only on common supported physics |
| Refusals | Missing saved fields/references, gradient medium, unsupported measure/endpoint/cutoff, invalid provider dual, unresolved time/space support and route/backend mismatch |
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
must show that the nonlinear change in modal capture is larger than numerical
uncertainty; the sign of that change is not assumed. Measure single/multimode
overlaps, radial/Cartesian extraction, regridding and scaled-plane transport with
actual input/output sizes and memory. Hot flux changes carry a short-window
benchmark ledger entry. Never extrapolate a timing into a full experimental run.

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
the capillary/free-space guides. Generate its output through Sphinx; do not
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
| Linear-focus 1/e² intensity radius | 50 μm, at the capillary entrance |
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
functions gives about 14.1 μJ, 0.443 GW and 0.145 times the current critical-power
estimate. These are provisional analytic estimates, not a propagated or
converged result. Recompute with the corrected launch convention and recorded
optmat material identity. Do not transplant the old vacuum energy scale.

Run two calculations from the identical physically focused incident field:
linear propagation through the gap, and propagation with Kerr enabled. At the
entrance, retain each full complex radial field and couple into the same capillary
mode set. Launch each captured result through the 5 mm capillary with identical
Kerr-only capillary physics, so the difference is the prepared incident field.
Do not replace either entrance field by a fitted Gaussian or its power spectrum.

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

Corrected physical-energy semantics are qualified for uniform spatial media.
The existing gradient route does not acquire conservation by relabelling it:
operations requiring that contract refuse it pending a separate forward-amplitude
derivation. No fallback retains the vacuum convention. The hybrid planar metric
and general-vector metric keep their own derived meanings; neither is assigned
the scalar bulk formula blindly.

SEP 5 consumes the shared scalar flux prerequisite and retains its solid-carrier
work. SEP 8 retains vector crystal fields/interfaces. SEP 9 retains its in-guide
measurement scope. Update those drafts and the roadmap so only one proposal owns
the common normalisation change. Preserve uncovered gradient, detector and
multi-pass-cell work explicitly. Spatially resolved compressor layouts remain
future work even though their ideal spectral transfer is usable between stages.

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

No numerical implementation begins until acceptance. Each phase records delivered
paths, deviations, evidence, unresolved questions and handoff in Part II. The
following phases are independently mergeable after their applicable checks;
backend refusals remain explicit until parity is delivered.

| Phase | Deliverables | Required evidence |
|---|---|---|
| **1a — NumPy flux foundation** | Uniform scalar spatial flux, launch/noise/statistics consistency, current-schema metadata, admission rules and manual derivation | Parseval, monochromatic/angular limits, source work, material loss, signed local flux, refusal cases and short benchmark |
| **1b — Numba flux** | Matching kernel/statistics implementation and memory accounting | Rounding-budget parity, interpreted-kernel coverage and short timing |
| **1c — Kokkos CPU flux** | Native flux/launch/statistics support in the backend's current repository | CPU parity, schema refusal and memory/timing evidence; no hidden host statistics fallback |
| **2 — Plane exchange** | `FieldPlane`, extraction, reference/frame records, same-basis relaunch and temporal remapping | Identity, scaling, delay/CEP and carrier tests; current-schema storage and target validation; all CPU source/target combinations |
| **3 — Coupling** | Analytic/sampled/Gaussian inputs, scalar and admitted vector projections, direct guide overlaps, reports and unified incident-energy launches | Independent Gaussian/Bessel/general-mode references, passive certificates, domain/quadrature/modal refinement and deliberate loss cases |
| **4 — Transport** | Outcoupling, homogeneous gaps, scaled observation planes, far field, thin focusing/apertures/Jones operations and spectral-optics composition | Independent diffraction and interface tests, thin-screen model bounds, operation order, bandwidth/efficiency/reflection signs and resource ledger |
| **5 — Workflows and tutorial** | All eight compositions, nonlinear-input tutorial, manual examples, final capability matrix and cross-SEP/roadmap reconciliation | Short field-grid plasma/THG transfer, compressor chain, tutorial convergence and clean docs, complete CPU qualification |

During each slice run fast checks and directly affected modules. At the end run
the full gate, slow lanes and coverage once, or use their CI evidence, as
`AGENTS.md` specifies. After a failure rerun the failing modules. Do not perform
full-length propagation as a test or timing. Device qualification is separate
and recorded under the standing backend policy.

The draft itself is documentation-only: run diff whitespace checks, codespell on
changed files, `tests/test_seps.py`, and the warnings-as-errors Sphinx build.
Do not implement a fake tutorial against nonexistent APIs to make a draft look
delivered. Phase 5 supplies its runnable source and generated gallery.

### 12. Open questions

These are explicit acceptance or qualification gates, not permission to choose a
different physics model silently during implementation.

| Question | Resolution required |
|---|---|
| **Q1 — Cross-model projection and interface admission** | Before acceptance, approve the concrete scalar-to-vector and cross-guide forward overlap models, including their common power spaces, E-only closures, passivity/conditioning bounds and refused index-contrast/facet cases. Same-basis recovery alone is insufficient evidence. |
| **Q2 — Thin elements with angular flux** | Before acceptance, approve the paraxial thin-mask/lens and reflection-frame model, its quantified angular/flux defect and refusal conditions. A local amplitude mask does not automatically conserve the nonlocal axial-flux measure. |
| **Q3 — Numerical admission budgets** | Phase 2/3 fixes error-target interfaces and justified conditioning/tail thresholds using independent refinement. No universal coupling-efficiency threshold is introduced. |
| **Q4 — Source references and current-schema persistence** | Phase 2 determines the minimal accumulated frame/reference record needed for standalone extraction, and tests reconstruction against source simulations. Missing physical reference information is refused, never guessed. |
| **Q5 — Tutorial parameters and convergence** | Phase 5 recomputes the proposed Gaussian sizing, verifies the Kerr-only regime, fixes mode count/grids and records whether a smaller strength was needed. The 14.1 μJ estimate is not a golden result. |
| **Q6 — Deferred physics** | Smooth-gradient amplitude transport, full junction scattering, high-NA vector optics and resolved spatial compressors remain roadmap work. None is required to claim the narrower qualified scope as Final. |
| **Q7 — Native backend location** | At Phase 1c use the backend home established by SEP 29 and record its commit; this is source coordination, not a scientific dependency on the distribution design. |

The selected scope, clean breaking-change policy, uniform-medium restriction and
Kerr-only tutorial model are settled. They are not reopened by these questions.

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

## Part III — History

- 2026-10-22: Drafted after auditing existing launch, projection, spatial flux,
  output and offline-optics interfaces. Selected forward coupling, spectral
  compressors, current free-space field models, controlled temporal remapping
  and uniform-medium flux. Required clean breaking changes with no legacy
  support and a mild-Kerr radial-input capillary tutorial. Identified cross-model
  projection and thin-screen flux as explicit pre-acceptance questions.
