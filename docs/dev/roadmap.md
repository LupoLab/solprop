# Roadmap

This page lists functionality solprop does not have yet, and known issues in what it has.
It is not a schedule: the items are grouped by area, and none is promised.
Some qualification sequences state the order in which their checks are needed.
What exists is described in the user manual; how it is built, in the rest of these
developer pages.

Every item is weighed against the project's priorities (ADR-020 in
[decisions.md](decisions.md)): accuracy and reliability first, then CPU performance, then
GPU. Taking one up follows ADR-033: an issue or pull request, an analytic or independent
reference with a stated tolerance, and the design pages updated in the same change. A
major item — a new model, route or numerical scheme, a changed default or convention —
starts as an [enhancement proposal](seps/index.md) (SEP, ADR-035).
Functionality that Luna.jl has and solprop lacks is added when a use case needs it; the
deliberate differences are listed on [Differences from Luna](../guide/workflows/luna.md).

## Accepted enhancement proposals

SEPs that have been accepted and are not yet final, each waiting to be picked up or
being implemented.

- [SEP 17 — Broadband two-photon continuum response beside strong-field ionisation](seps/sep-0017-gas-two-photon-ionisation.md):
  prescribed-field diagnostics first, then an Ar/He continuum driven by the squared VUV
  field, with polarisation and optical work from one interaction. The first model gives
  final yield and optical feedback; additional plasma needs a separate electron-timing
  qualification. Existing optmat OEP 6 quantities suffice for this approximation.
  Before production implementation, its qualification covers threshold data and
  completion, reactive matching, atomic validity and the complex convolution's work,
  passivity and window convergence. A coupled optmat OEP is needed only if qualification
  identifies a new material contract. Ne/Xe, mixtures, mode-averaged and multimode
  routes, bound response and other geometries remain follow-up work; the first route is
  a single-mode capillary with node quadrature.
- [SEP 21 — Temporal boundaries: wider windows, absorption and the escaped field](seps/sep-0021-absorbing-boundaries.md):
  in progress. A campaign against wide-window references compares windows without
  temporal absorption, monitored by guard regions, with collar strategies at matched
  physical accuracy, including fixed asymmetric placement and prescribed frames
  (phase 0a), and qualifies an escaped-field record separately (phase 0b). The
  qualified strategies, guard monitor, absorbed-energy and occupancy diagnostics and
  the keyword switches follow; an implicit–explicit collar only where it improves on
  every admissible alternative. Spatial collar changes, adaptive recentering/expansion,
  a general offset-grid API, independently asymmetric absorber ramps and implicit
  Runge–Kutta remain open.

## Draft enhancement proposals

SEPs under discussion; their scope may change and implementation has not been accepted.
The [SEP index](seps/index.md) lists every proposal and its current status.

- [SEP 1 — Efficient causal Raman evaluation](seps/sep-0001-raman-response.md):
  exact sampled recurrences, including the current lag taper and silica endpoints,
  beside default exact FFT evaluation. A common execution contract retains analytic
  poles, permits exact FFT evaluation of density-shifted lines and supports SEP 10's
  explicit approximation. Population dynamics are separated into SEP 32.
- [SEP 2 — Two-photon absorption](seps/sep-0002-two-photon-absorption.md):
  constant-coefficient, narrowband scalar absorption on field and envelope grids.
  Broadband and tensor absorption, and solid carrier dynamics, remain outside its scope.
- [SEP 4 — Bounded execution autotuning](seps/sep-0004-execution-autotuning.md):
  explicit CPU policy search on short windows, with isolated processes, numerical
  validation, bounded total cost and reproducible execution plans. Native GPU
  controls, chunking, propagator selection and implicit tuning remain open below.
- [SEP 5 — Solid ionisation and bulk filamentation](seps/sep-0005-solid-ionisation.md):
  scalar bulk silica/YAG photoionisation, explicit simple avalanche and carrier
  removal, collisional currents, optmat solid rate models and parameter bundles
  (which need upstream optmat work first), using SEP 34's shared medium flux and
  radial adiabatic transport.
  Delayed-heating kinetics, subcycle rates and solid-guide ionisation remain
  follow-up work below; source-normalisation questions precede acceptance.
- [SEP 7 — Laser gain dynamics in solid-core fibres](seps/sep-0007-laser-gain-dynamics.md):
  Yb/silica spectroscopy through optmat, conventional steady and time-dependent
  populations, pump and ASE transport, and scalar spatial gain dynamics for nonlinear
  amplifiers. The broadband filtered-field closure has a separate qualification gate;
  general vector gain, other ions, stochastic field ASE and oscillators remain open.
- [SEP 8 — Second-order nonlinearity and birefringent crystal optics](seps/sep-0008-quadratic-crystal-optics.md):
  vector forward propagation in homogeneous crystals, quadratic frequency mixing,
  BBO/LBO material curation, field/envelope grids, interfaces, physical flux and
  a separately qualified statistical OPG model. Guided overlaps, varying crystal
  profiles, resonant THz response and bidirectional propagation remain open.
- [SEP 9 — Broadband beam measurements and propagation statistics](seps/sep-0009-propagation-statistics.md):
  frequency-dependent modal fluence and time-resolved beam widths, spectral filters,
  per-mode durations, transverse peak intensity and distinct quadrature/modal-span
  diagnostics. General-guide embedded quadrature and current-source span diagnostics
  remain open; SEP 34 covers the shared medium-flux correction.
- [SEP 10 — Low-rank Raman kernels along density paths](seps/sep-0010-low-rank-raman.md):
  explicitly requested spectral compression of density-dependent Raman kernels,
  extending SEP 1's contract and exact shifted-line FFT reference. Separate basis and
  coefficient-path budgets, independent-state validation and observable convergence
  bound the approximation; zero tolerance selects exact evaluation.
- [SEP 11 — Finite-density gas index and mixture optics](seps/sep-0011-finite-density-gas-index.md):
  optmat's Lorentz–Lorenz law with refractivity virials, applied once to a mixture,
  and a composition-preserving mixture EOS. One resolved state feeds linear optics,
  Raman amplitudes, widths and shifts. Current upstream model selections, estimates
  and validity records govern admission; nonlinear local-field corrections remain open.
- [SEP 32 — Raman Maxwell–Bloch dynamics](seps/sep-0032-raman-maxwell-bloch.md):
  a separate scalar two-level population/coherence response, with equilibrium
  subtraction, a weak-response limit and local temporal integration. Its ensemble,
  data and integration questions must be settled before acceptance; shared-level
  ladders, vector saturation and repeated-window histories remain future work.

The informal items below record work not fully covered by these proposals. When adding
or revising a SEP, remove an informal item only if the proposal covers it in full;
otherwise link the SEP, state what it covers and keep the remaining work explicit.
Moving work into a proposal does not mean it is implemented.

## Physics and models

**Spectral-domain responses.** A response declares its domain, and `spectral` is one of
the declared values (ADR-011). [SEP 2](seps/sep-0002-two-photon-absorption.md) partly
covers this item with a scalar, constant-coefficient, narrowband two-photon loss current.
[SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md) covers dispersive electronic
third-order response in qualified transparent gases, with two FFT-based evaluators.
[SEP 8](seps/sep-0008-quadratic-crystal-optics.md) covers second-order
($\chi^{(2)}$) mixing in homogeneous birefringent crystals, with vector linear/source
operators, orientation, walk-off and explicit nonlinear dispersion. Guided quadratic
overlaps, resonant THz/phonon response and general anisotropic third-order response
remain open, as do broadband mixed-frequency and tensor absorption.
[SEP 17](seps/sep-0017-gas-two-photon-ionisation.md) proposes an effective continuum
optical response for Ar/He with memory and a common polarisation/work derivation. Its
pair-sum approximation does not determine general nondegenerate or channel-coherent
amplitudes. Those, other species, solid responses and additional electron timing
remain open. Ne/Xe, mixtures and mode-averaged or multimode gas configurations require
later qualification beyond the first single-mode capillary implementation.

**Beyond the scalar dispersive electronic response.**
[SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)'s evaluators are scalar, off-resonant
and for gases. Open beyond it:

- tensor dispersion: the vector response holds the isotropic Kleinman ratio
  $\chi_{xxyy}=\chi_{xxxx}/3$ at every frequency, where the tensor components of a gas
  disperse differently; independently supported components are needed first;
- fused silica: its degenerate $n_2$ curve does not determine its nondegenerate
  response, and no curated nondegenerate data exist to factorise, so silica keeps the
  carrier-evaluated cubic;
- the GNLSE route, whose fitted $\gamma$ is one constant and which refuses a dispersive
  policy by name;
- the bounded region's few xenon and CO quartets that mix legs on either side of a
  400 nm carrier and are not shown better than it (SEP 6 §4.10): whether they need a
  narrower bound.

**The constant cubic's number-conserving rule beyond an octave.** The field-grid
`thg=False` rule $3|E_a|^2E/4$ and the general modal route's `kerr_form="analytic"` lack
the projection onto positive frequencies, $\Pi_+$, that the dispersive responses'
number-conserving sector applies. Below an octave the two agree exactly; beyond it the
constant rule does a little net work on the field
([SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md), §4.9). Adding $\Pi_+$
to it changes results only for fields wider than an octave.

**The strong-field approximation for harmonic generation.** A single-atom dipole response
computed in the strong-field approximation (the Lewenstein model), driven by the
propagated field, would model high-harmonic generation and its propagation in gas-filled
guides and in free space. It is a memory response of the fused time-domain column;
its cost is set by the recollision integral, and a validation needs an independent
single-atom reference such as a time-dependent Schrödinger solution.
[SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md) delivers a bounded TDSE integration for
helium capillaries, with illustrative 120–200 nm reporting and explicitly
unqualified physical-window and calibration errors; it does not implement
the SFA response or complete high-harmonic propagation described here.

**Gas charge-state ladders and avalanche ionisation.** Each ionising species is one channel
from the neutral ([mixtures](../guide/media/mixtures.md), ADR-027). Successive ionisation
through coupled charge-state populations, and collisional (avalanche) ionisation, are not
modelled; both matter at high intensity and pressure. Multiple ionisation and
dissociation fit the kernel contract as further channels and further values of
`electrons_per_event`, without a new seam.
The solid avalanche closure proposed in [SEP 5](seps/sep-0005-solid-ionisation.md)
does not cover gas avalanche or coupled gas charge states.

**Ionisation in solids.** [SEP 5](seps/sep-0005-solid-ionisation.md) covers scalar
bulk free-space propagation with solid photoionisation, simple avalanche, carrier
removal and collisional optical currents. Solid-core fibres and regional general
modal guides remain open, as do delayed-heating avalanche (multiple/dynamical rate
equations), genuinely subcycle solid rates, tensor/anisotropic carrier response,
inter-pulse memory and GPU qualification. YAG Raman requires its own qualified data
when needed. The proposed bulk model does not supply those extensions or material
damage dynamics.

**Phase-resolved ionisation rates.** The Yudin–Ivanov rate is available for research
comparisons of the envelope plasma model, not in production propagation. Adopting it
needs a separate loss-current closure and a resolution of the rate/current questions set
out on [envelope plasma](../guide/media/envelope-plasma.md).

**Raman population dynamics and data.** The Raman responses are fixed kernels
([Raman](../guide/media/raman.md)). This item is partly covered by
[SEP 32](seps/sep-0032-raman-maxwell-bloch.md), which proposes scalar two-level population
dynamics within one temporal window. [SEP 3](seps/archive/sep-0003-optmat-materials.md)
delivered optmat's per-line responses with density-dependent widths;
[SEP 1](seps/sep-0001-raman-response.md) extends exact evaluation to shifted lines
along density gradients and [SEP 10](seps/sep-0010-low-rank-raman.md) adds optional
compression. Shared-level
rotational and vibrational ladders and accumulation between separate pulse windows
remain open. SEP 3 retains the missing nitrous-oxide rotations and provisional
vibrational data as limitations. It also refuses Raman in mixtures whose collision
partners optmat has not curated, such as H2, D2, CH4, N2 or O2 in a noble-gas buffer
other than argon. Lifting that refusal needs foreign-gas broadening coefficients curated
upstream in optmat, which is an optmat roadmap item. Q-branch line mixing at high
density, modern H2/D2 level energies and the H2/D2 gain excess are optmat's open items
too. No proposal supplies the glass overlap of a hollow-core
fibre, so wall Raman remains open on the gas-capillary route; the general modal route's
material regions are the building blocks.

**Broadband Raman coupling.** Raman-shift dependence and carrier-selected strengths
already exist; dependence on the participating optical frequencies across a broadband
field does not. UV generation motivates a separate material/response proposal.
An off-resonant separable coupling could retain FFT or recurrent oscillator histories,
but its drive and emitted polarisation must be derived together, with reality,
reciprocity and energy transfer checked. Curated frequency coverage and weak-line
accuracy are open. This work is independent of Maxwell–Bloch populations and does
not block SEPs 1 or 10. SEP 6's electronic filters do not supply Raman dispersion;
see {ref}`SEP 1 §10 <sep1-broadband-raman>`.

**Gain with saturation.** A linear operator may carry prescribed gain; gain saturation,
population inversion and an amplifier or laser population model do not yet exist.
[SEP 7](seps/sep-0007-laser-gain-dynamics.md) proposes Yb/silica amplifier dynamics with
steady and transient rate equations, pump/ASE transport and scalar population resolution.
Qualification of its broadband filtered-field closure is an explicit decision, not an
assumed capability. General vector and moving-guide gain, additional dopants and their
host-specific kinetics, stochastic field ASE, oscillator dynamics, thermal feedback and
photodarkening remain outside that proposal's implementation scope.

**Resonant gas susceptibilities.** The gas models are off-resonance fits. Near
ultraviolet and vacuum-ultraviolet resonances a user-supplied dispersion does not make the
nonlinear or ionisation response valid there; a resonant susceptibility model would.
[SEP 3](seps/archive/sep-0003-optmat-materials.md) partly addresses the material data through
curated dispersion and available absorption, but does not supply a complete resonant
nonlinear or ionisation model.
[SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md) adds off-resonant electronic
dispersion and identifies UV/VUV validity gaps; it does not supply the missing
resonant response or its nonlinear absorption. Beyond a gas's band edge its
evaluators hold a constant labelled best effort, and the helium dispersive wave's
energy moves by 30 to 55 % between that and the carrier value, depending on the mode
set ([the worked example](../auto_examples/processes/plot_07_rdw_dispersive_kerr)): a
resonant electronic response of helium in the vacuum ultraviolet is the missing
reference.
[SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md) implements a helium TDSE
response ([TDSE response](../guide/media/tdse.md)); its ultraviolet validity and its
behaviour near resonances remain open, and it does not supply a general resonant
material model.
[SEP 17](seps/sep-0017-gas-two-photon-ionisation.md) proposes an Ar/He continuum
optical response with its dispersive partner, subject to completion and background
matching. It does not supply a validated general resonant susceptibility, bound-state
dynamics or (2+1) ionisation; those material inputs and propagation models remain open.
optmat OEP 8's bound-state material work is separate from the continuum approximation.

**Noise transport along a gradient.** The modified noise is advanced by the unitary part
of the linear flow ([noise](../guide/media/noise.md), ADR-008). With true-power amplitudes
(ADR-030) that leaves the noise floor drifting along a pressure gradient, by at most 0.3 %
of the noise amplitude over a 20 bar gradient. Transporting the draw with the power weight
would remove the approximation.
[SEP 16](seps/archive/sep-0016-unified-multistep.md) qualified modified-noise stepping while
preserving the existing model; it did not remove this physical normalisation limitation.

## Waveguides and geometries

**Material absorption in step-index fibres.** The step-index eigenproblem takes the core
and cladding indices as real, so absorption in the glass does not enter the modes' loss.
A confinement-factor correction, each region's absorption weighted by the share of the
mode's power it carries, would add it without a complex eigensolve.

**Integrated waveguides.** Support chip-scale nonlinear optics in silicon nitride
(SiN), silicon (Si) and lithium niobate waveguides, with geometry-dependent dispersion,
loss and vector mode profiles. The general modal route provides a starting point;
cross-section providers or imported mode-solver data, material-specific nonlinear
responses and validated examples remain to be developed. The scope should account for
second-order nonlinearity and crystal orientation, nonlinear absorption and carrier
dynamics where relevant, alongside Kerr and Raman responses.
[SEP 8](seps/sep-0008-quadratic-crystal-optics.md) proposes bulk crystal tensors
and orientation handling; guided quadratic overlaps and integrated-waveguide
providers remain outside its implementation scope.

**Multi-pass cells (MPCs).** Model pulse compression in gas-filled and solid-state
multi-pass cells by composing repeated bulk propagation with focusing mirrors,
dispersive optical elements and the chosen cell geometry. A cell-level workflow would
need to carry the spatial and temporal field from pass to pass, account for mirror
dispersion and loss, and expose the evolution of the beam, spectrum and compressible
pulse duration. [SEP 31](seps/archive/sep-0031-simulation-coupling.md) added field
handoffs, focusing, reflection and spectral optics on top of the existing free-space
routes ([coupling simulations](../guide/workflows/coupling.md)). Cell geometry, assembly
and validation against independent propagation or experimental data remain open.

**Coupling beyond the forward model.**
[SEP 31](seps/archive/sep-0031-simulation-coupling.md) delivered forward overlaps,
scalar homogeneous gaps and spectral compressors. A general junction calculation with
reflected, radiated and evanescent fields, high-NA vector free-space transport,
gradient transport outside SEP 34's certificate, varying absorption, resolved
prism/grating angular dispersion and spatial chirp, and feedback between stages
remain separate extensions.
[SEP 34](seps/archive/sep-0034-medium-flux.md) covers the existing radial route's
certified forward amplitude correction, including admitted weak steps; it does
not add general gradient scattering or a complex-index path certificate.

**A graded-index facet envelope for the vector coupling model.** `vector_paraxial` into a
step-index guide counts a recorded envelope for the facet reflection and backward fields
it omits (`VECTOR_STEP_INDEX_ENVELOPE`, against an independent full-vector mode-matching
solution of the facet). A graded-index guide has no such envelope: it is admitted with an
uncounted envelope item, its facet omission bounded only by the weak-guidance contrast
screen and the H closure. A graded-index reference — the same mode-matching solver with
a layered core — would give it a recorded bound
([SEP 31](seps/archive/sep-0031-simulation-coupling.md), decision A1; [coupling simulations](../guide/workflows/coupling.md)).

**The coupling envelopes beyond their measured range.** Each recorded pair envelope holds
only where it was measured ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), Part II, "Closing"):
- the capillary envelope stops at the leaky parameter $\delta = 10^{-2}$, so a narrow
  capillary in the infrared is refused; extending it needs a re-measured bound, and is
  worth doing only when such a pair is needed;
- the beam-into-fibre envelope covers LP01 alone (V 1.5–2.4, contrast up to 0.01); near
  V 2.4 its in-cone figure is dominated by the cone's truncation of the widening HE11
  spectrum rather than by the vector correction, and could be sharpened;
- the vector facet envelope rests on the analytic Fresnel ceiling in the narrow bands
  just above the HE12 and HE13 cutoffs, which no practical reference cylinder resolves;
- the fibre-pair `STEP_INDEX_ENVELOPE` is a constant bound whose order in the index
  contrast was not measured; re-measuring it would let it scale as the others do;
- every envelope is measured at the profile wavelength against the run's frozen
  profile, so a fibre whose profile moves strongly across the band is represented only
  as the run represents it.

**The general guide's angular item at every occupied bin.** `couple` into a general
modal guide evaluates the target's angular admission at no more than five bins of the
band. Evaluating it at every occupied bin closes the gap, if its cost stays within a
coupling's; the cost is to be measured first ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phase 3).

**Other exit interfaces for `Outcouple`.** `Outcouple` synthesises a guide's exit field
under the matched interface only. A declared planar-interface model at the exit (the
end face of a fibre into air, for example) would make the facet's reflection and
refraction explicit, as `PlanarInterface` does between two media in transport
([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phase 3).

**Vector coupling into a moving general guide.** The input port of a general guide whose
geometry moves along z (a sampled path) carries no E/H reference at the entrance, so it
admits only `same_basis`; `vector_paraxial` needs the entrance section's fields from the
path ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phase 3).

**Elliptical hollow capillary modes.** Extend hollow-capillary modelling to elliptical
bores specified by their two semi-axes, with mode profiles, dispersion, wall loss and
nonlinear overlaps that resolve the geometry and polarisation. This needs a mode model
and transverse quadrature validated against an independent reference, including
recovery of the circular-capillary limit as the semi-axes become equal. The choice
between an analytic approximation and numerically supplied modes remains open.

**Zeisberger antiresonant modes.** The analytic antiresonant model of Zeisberger and
Schmidt is not implemented; the antiresonant family is Vincetti's tube-lattice model
([modes](../guide/waveguides/modes.md)). A second model gives an independent check on the first and
covers geometries described by a single effective wall.

**A coherent-wall (Robin) slab.** The planar guides truncate the field at the plates, with
one wall-loss pair per channel ([spatial decompositions](../guide/setup/spatial-decompositions.md)).
The coherent wall that couples capillary modes through the condensed outgoing Robin
condition has no planar counterpart; with one, slab modes would exchange power through
the wall as `HE_1m` modes do.

**Planar guides resolved in y, beyond one family.** The hybrid planar route (slab modes in
x, an FFT in y) carries one polarisation family with a constant gap and constant media. Both
families together, a gap or medium varying along z, and a settled normalisation for a noise
draw on the planar grid (refused at translation today) are open.

**Single-azimuthal-order radial free space.** A full radial and azimuthal expansion is not
built: it needs one Hankel transform per order, on radial nodes the orders do not share,
and beams that break cylindrical symmetry belong on the Cartesian grid. A single order-$m$
transform is a small change to the order-zero quasi-discrete Hankel transform. It would be
exact for cylindrical-vector (radially or azimuthally polarised) beams under isotropic
responses; for a scalar vortex, exact only on the envelope grid without third-harmonic
generation, and blind to azimuthal break-up.

**Cartesian media varying along z.** Cartesian free space requires a medium constant along
z: its seam stores frequency-only material data and evaluates $\Phi$ as $D\,\Delta z$.
Varying profiles and tabulation are refused by name ([Cartesian](../guide/free-space/cartesian.md)).
The radial grid already supports varying media through cumulative expansions.
[SEP 8](seps/sep-0008-quadratic-crystal-optics.md) proposes homogeneous crystal
slabs and resolved longitudinal nonlinear domain reversal; it does not cover
continuously varying dielectric tensors, crystal orientation or temperature.

**Taper coupling for tube-lattice fibres.** A multimode capillary whose radius varies
along z carries the taper coupling between its `HE_1m` modes by default. A Vincetti
tube-lattice taper keeps the adiabatic route, although its fields are Marcatili's and the
connection would be the same; admitting it is a small change with its tests.

**The envelope grid on the general modal route and the coupled flows.** The general modal
route, the coherent wall and the taper coupling run on the field grid; an envelope grid
is refused by name ([general modal](../guide/waveguides/general-modal.md)). The coupled flows need an
increasing frequency axis, and the envelope grid is in FFT order. `prop_gnlse` covers the
scalar envelope case.

**Hybrid graded-index sectors.** Graded-index guides provide the `LP` sectors under weak
guidance (index contrast up to 0.02) and the radial `TE_0m`/`TM_0m` sectors. Hybrid
`HE`/`EH` sectors of azimuthal order one or more are refused; they are needed for strongly
guiding profiles and for polarisation effects in graded-index fibres.

## Numerics

**Multistep integration with general varying noncommuting coupled flows.**
[SEP 16](seps/archive/sep-0016-unified-multistep.md) covered diagonal and separable operators,
adiabatic tapers and the constant-Schur and frozen-residual coupled sections, which
already carry a constant block with an explicit residual, for its exponential
formulation. CF4 sections and other general varying coupled generators without an explicit
residual need a separate propagator/residual derivation, action-error control,
history/dense-output construction and stability analysis. Exponentiating the integrated
matrix is not generally its exact flow. The Runge–Kutta march keeps them; the Adams
methods refuse such a problem by name rather than omit its coupling.

**A coupled flow with no source under the Adams methods.** Beside a frozen-residual
coupled flow the explicit residual joins the compensation, which is sampled with the
source (ADR-042, ADR-044). A run with no nonlinear source has no sample, so its steps
are the frozen block's alone and their error, second order in the step, is not seen
by the controller: a linear graded-index taper at one bin is 6.4e-4 from the converged
answer at ten steps whatever `rtol` ([the Adams methods](../guide/numerics/adams.md#limitations)).
Sampling the residual without a source, as a cheap evaluation of the linear part
alone, would put it under control.

**The carrier frame's weak-band control and step floor.**
[SEP 35](seps/archive/sep-0035-carrier-frame-exponential-adams.md)'s
`carrier_frame="scalar"` gives a weakly populated, driven bin a residual rotation that the
step-size norm, which the populated bins dominate, does not see: near the launch such a
band came out a few hundred times less accurate at matched total-field error. Scoring a
forming weak band in the norm, or a per-bin shift chosen by the source's own rotation
rather than a frequency boundary, might keep the policy's calls and recover the bands.
Two measured but unexplained costs belong with it: more calls where the in-stage collar
binds the step (842 against the default's 754 on a short dense-gas window), and the cap's
floor of about $|\kappa| L$ steps, not yet measured on a long, weakly nonlinear
dense-gas propagation.

**The exponential formulation beside the `modified` noise.** The noise's source is
partly linear in each weak bin's own field and rotates with its free phase, which the
exponential formulation interpolates in the laboratory frame. Until a
developed noisy window, such as the CtC comb, measures it, the manual prefers
`Multistep(formulation="interaction")` beside that noise.

**Qualifying the TDSE response.** The helium TDSE response
([SEP 12](seps/archive/sep-0012-tdse-nonlinear-response.md),
[SEP 20](seps/archive/sep-0020-hacc-helium-tdse.md),
[SEP 26](seps/archive/sep-0026-edge-free-tdse-source.md)) is a research option, demonstrated
over short capillary segments ([accuracy and limitations](../guide/media/tdse.md#accuracy-and-limitations)).
Open: a converged physical time window and atomic preparation, against which the default
60 fs window's history sensitivity becomes a stated error rather than a recorded
difference; a calibration of the single-active-electron current scale and of the
ionisation yields that is independent of the linear, cubic and rate matching it rests
on; numerical qualification over longer segments and other waveforms, then a
full-length capillary demonstration with a worked example; and publication of the
atomic provider, Pyroraptor ([dependencies](dependencies.md)), so that the response can
be installed from a released package. Other gases, double ionisation and solver
comparisons on the actual atomic source lie beyond these.

**A quadrature-error estimate on Cartesian waveguide runs.** A radial run scores its
transverse quadrature against an embedded coarse rule and reports the difference as the
`quadrature_error` statistic; a Cartesian waveguide run has no embedded rule, so the
statistic is unavailable there.

**Light at a waveguide's exact cutoff in free space.** Grazing incidence on a spatial
grid is handled arithmetically (the residuals are exact and a warning names significant
content near cutoff), but there is no physical treatment of content sitting at the exact
cutoff, where the longitudinal wavenumber vanishes; a run with significant power there
is outside the model.

**Open numerical questions in the stepper.** Error norms for observables far below the
pump, reject-free stepping at matched global error, schedule transfer beyond the
measured ranges (including from a coarser transverse grid to the production one),
round-off growth with varying operators and FSAL through a coupled flow are measurable
questions rather than planned work; they are listed in [open questions](open-questions.md).

**An embedded low-storage pair for adaptive runs.**
[SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md)'s march is
replay-only; an optional extension would add an embedded pair with adaptive control
on a fifth or sixth volume, qualified against the adaptive Runge–Kutta march as the Adams
methods were. The SEP recommends starting from Balac and Mahé's ERK4(3)-IP
(doi:10.1016/j.cpc.2012.12.020), which keeps the march's stencil, kernels and checks,
once its coefficients, stages and FSAL structure are confirmed from the paper; a 2R or
3S* scheme only if one register decides, whose inner nodes need inverse flows and
fresh qualification. No memory-bound adaptive run needs it yet, and the
march with an external calibration (`calibrate`, `replay`) covers the fixed-step use.

**An absolute tolerance on density paths.** The adaptive Chebyshev builder accepts an
interval by a tolerance relative to each element, with ADR-040's sampling-noise floor
where floating-point z is too coarse. Near a choked gas end the density, and with it
the nonlinear response, is a fraction of a percent of its peak, so an absolute tolerance
against the path's peak would stop refinement there earlier. It would complement the
floor, not replace it; worth adding if translation near near-vacuum ends shows up in a
profile ([SEP 14](seps/archive/sep-0014-stepper-efficiency.md)).

**A narrow-support fallback for general guides.** A general modal guide must be sampled
over the run's whole active band, which reaches to half the smallest declared frequency
and is wider than the declared band. A guide that does not span it is refused, and the
user narrows the grid. A fallback would propagate only on the guide's support and warn
when energy reaches a guard band inside its edge; it needs an optional band threaded
through the shared linear-operator construction that every route runs, and a definition
of the warning threshold.

**Observable tolerances on the general modal route.** `observable_rtol` and
`observable_atol` accept only `None`, meaning the controller's own tolerances. A number
would need an error allocation over the table's frequency interpolation, the transverse
rule and the step, and named physical observables to allocate against; until then a
number is refused rather than recorded and ignored.

**Noise and batches beside a coupled flow.** A noise model beside a coupled flow (a wall
or the taper coupling) is refused, because a draw's transport through the flow is not
measured; so is a coupled flow in a lockstep batch. Both need a derivation and a test
before they can be admitted.

**Independent spatial references for the vector and graded-index routes.** The scalar bore
and taper are checked against an independent finite-element solve. The vector wall is
qualified by the exact single-family Maxwell roots and the mixed-coupling eigenvalue
evidence; a resolved-cladding vector solve in a mixed state would be the stronger check.
Graded-index and vector spatial references, and a demonstration of a mode crossing its
cutoff along a taper, are also open.

## Performance and backends

The [CPU performance record](cpu-performance.md) lists what has been tried, adopted and
rejected on the compiled CPU backend, with its measurements; read it before taking up
any of the first ten items.

- **The Adams carrier frame on devices and on cheap sources**: native's
  `carrier_frame="scalar"` kernels are written for host and device but refused on device
  builds until a GPU segment runs their parity tests and ten-step timing; on CPU each step
  costs 4–7 % more than the default's on a cheap source, which the per-history-row
  rotation of the defect reducers could reduce
  ([SEP 35](seps/archive/sep-0035-carrier-frame-exponential-adams.md)).
- **BLAS threads in the Windows and manylinux recipes**: one OpenMP runtime for Kokkos,
  FFTW and OpenBLAS in the Windows wheel (set up in the wheels workflow, to be proved
  there and timed), the same check on the manylinux release recipe, and the Kokkos
  team-barrier backoff (sleeps that Windows rounds up to its scheduler tick) reported
  upstream.
- **The problem-size thread policy**: the rows that measurement noise left unresolved are
  within about 1 % of the pre-policy build on the 36-core Xeon (2026-10-22). Open items:
  - the scalar wall, 10 % slower there at the default width, against an A/A spread under
    1 % (on Apple silicon it is faster); find the cause;
  - the taper, short-line and planar controls;
  - the stage and FFT starts on macOS, kept at 8,192 and 32,768 while Linux uses 4,096
    and 16,384 ([smooth grid sizes](benchmarks.md#smooth-grid-sizes)): on Apple
    silicon the serial stage is faster even at 8,196 elements, so sweep the crossover
    there on a quiet machine (`native/benchmarks/size_cliffs.py` and the SSFS and
    reduced comb-to-continuum cases) and set its own values.
- **The multimode step at full width on Linux**: at 36 threads on the Xeon, general-modal
  RK is about 6 % slower than before the per-core changes (reduced comb-to-continuum is
  4 % faster). Estimate planning is as slow, so FFT plan selection is ruled out; find the
  cause.
- **The TE/TM wall's complex GEMM**, slower in Kokkos Kernels and OpenBLAS than in a
  vendor BLAS for its shapes: improve the kernel without letting this rare
  configuration decide the wheels' BLAS.
- **Windows re-measurement** of the CPU fixes against `numba_cpu` on the single-runtime
  wheel, and a Windows CI lane that fits its memory (fewer xdist workers or a larger
  page file). Two `numba_cpu` cache tests exceed Windows's 260-character path limit.
- **FFTW planning policy per transform shape**: `estimate`, `measure` and saved wisdom,
  and wider codelets, compared per representative shape and processor and qualified on
  the minimum CPU model.
- **Compiler targets with identical libraries** (x86-64-v2, v3, v4 and processor-specific
  targets), which matter most on the general modal route.
- **Explicit SIMD** in at most one or two kernels (the general-modal response, synthesis
  and projection), after the compiler-target comparison.
- **Selective run-time kernel dispatch**: AVX-512 variants of a hot kernel in the
  x86-64-v3 wheel behind a CPU-feature selector
  ([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md), §6.2).
- **Batched vector transcendentals for the plasma response**, which need the dependent
  plasma recurrence split into a batched rate pass.

**Further fusion in the stage passes.** Candidates, each subject to the source-accumulation
and stage-dependency checks of [SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)
§6.7: `stage.phase` into the combination; the finish of one stage with the combination of
the next; the zero-fill copies into the kernels that fill the rest of the padded buffer;
and a 1×1 map's transform scale, synthesis and projection into the response column
([architecture.md §8](architecture.md)).

**Floating-point options and reduced precision.** What the floating-point policy
([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md) §5) leaves untried,
with the expected gains. Build options promise little: the transforms and the BLAS are in
libraries these flags do not reach, and the policy already takes contraction and
per-kernel reassociation. Only reduced precision promises a large factor, and mainly on
GPUs with slow double precision.

- Decided: contraction on, with the protected blocks strict; reassociation per kernel; no
  umbrella fast-math, `nnan`/`ninf`, `nsz`, or approximate functions and reciprocals.
  Measured and not adopted: libmvec `exp` and `log` under Clang (about 3 %, still behind
  GCC) and Accelerate's vForce in the propagator (about 0.1 ms per window).
- Value-safe options, not yet measured: `-fno-math-errno` and `-fno-trapping-math` change
  no rounded result. Under GCC (the Linux builds) the first lets `sqrt` compile to one
  instruction and vectorise; the second lets loops with conditional floating-point work
  be if-converted. A few per cent at most, in the custom kernels only.
- Subnormals: no build sets flush-to-zero. On x86-64 an operation on a subnormal can cost
  of the order of a hundred cycles (NVIDIA GPUs and Apple silicon take them at full
  speed), and spectral wings held down by an absorber or a per-step `exp(-rate dz)` could
  reach that range on long runs. Count the subnormals in a long production run's field
  first; flush-to-zero needs its own justification (SEP 25) only if they appear.
- Double-precision transcendentals on GPUs with a reduced FP64 rate (NVIDIA's GA102 and
  AD102 parts, such as the A40, run FP64 at 1/64 of FP32): the stage combination
  evaluates `exp`, `cos` and `sin` per bin and stage, which there may cost as much as the
  transforms. Profile first. Remedies that stay in double include deriving one stage's
  propagator from another's (squaring a half-step propagator) and reusing propagators
  while `dz` is unchanged.
- A single-precision source evaluation inside a double-precision solver: the transforms,
  the responses and the modal GEMMs in FP32; the state, the linear phase increments, the
  Adams history and the running sums in FP64. Its rounding enters each step scaled by
  `dz`, about FP32's epsilon times the step's nonlinear phase, so the error estimate's
  floor stays near $10^{-8}$, against about $10^{-5}$ for a solver entirely in FP32.
  Expected gain, unmeasured: up to the bandwidth factor of two on CPUs and data-centre
  GPUs, possibly several times on reduced-FP64 GPUs. A first test needs no new data
  types: round only the source evaluation to `complex64` in `numpy_ref`, on coherent
  cases and one noise-seeded case. Noise-seeded runs may fail it, since FP32 rounding can
  sit near a one-photon-per-mode seed for pulses of a microjoule and above. Adoption
  would be a SEP.
- Entirely single precision, and Metal: [SEP 33](seps/archive/sep-0033-single-precision-mlx.md)
  (final) built the precision setting, a `numpy_ref` oracle and an MLX backend for
  Apple GPUs on the fibre, radial and Cartesian routes with every response but the
  refused ones. Its open questions: whether a rule comparing binary32 Adams with
  binary64 Adams should admit `Multistep()` under single precision (binary64 Adams has
  the same errors at the tolerances the study used); `mlx` cases in the standard
  benchmark suite (SEP 30); and a fused stage kernel for the launch-bound small plasma
  capillaries. Open beyond it: single precision on
  `native` and `numba_cpu` (Kokkos, Kokkos Kernels and Kokkos-FFT support it on every
  backend in use), on the routes SEP 33 refuses, and noise models under single
  precision, which SEP 33 refuses as a known limitation: binary32 rounding can reach a
  one-photon-per-mode seed at high pulse energy. Kokkos has no Metal backend, and one
  would need a single-source C++ compiler for Metal (the runtime part is modest: 5 to
  10 thousand lines for each existing GPU backend).

**Benchmark platform qualification beyond the initial CPU suite.**
[SEP 30](seps/archive/sep-0030-standard-benchmark-suite.md) covers the standard suite, portable
records and durable incremental exports, with CPU timing collections on macOS and
Linux and an initial native coverage collection on macOS. Linux native coverage,
Windows execution, cloud launch/collection recipes and GPU timing and coverage
need separate qualification. Keep future platform results in distinct machine/build
records; successful CPU coverage does not establish coverage of device code.

**Remaining benchmark references and estimates.** Complete independent reference
checks for the 512-node radial filament, 128/256-square Cartesian filaments and
768-square Kerr beam when those workloads are needed for validated comparisons.
Keep their incomplete qualification visible while permitting exploratory timing.
Add runtime forecasts from compatible pilot receipts and automatic consumption of
compatible measured memory receipts. Until then, planning reports unknown costs
and admission uses conservative provisional budgets.

**The low-storage march's remaining volumes and names.** Follow-ups of
[SEP 24](seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md):
- *The planar route's stage register.* Its finish could write the next pass's
  input into the column volume, as the Cartesian finish writes it into the transverse
  scratch, saving one volume when no in-stage collar is present.
- *A varying modal prefactor on `numba_cpu`.* On a modal route whose response
  prefactor varies along z, each pass combines the prefactor's terms into a fresh
  state-sized array (both marches, `NumbaCpuBackend._prefactor_for`); a held buffer
  would take it into the ledger, as the march's recomputed flows are.
- *Hook names.* Rename `_adams_lab_source` and `_adams_propagator`, which the
  low-storage engines also call, with the next change to the kernel contract.
- *Device qualification.* The march's passes pass their device tests on a HIP build;
  the CUDA build is to be qualified on an NVIDIA GPU, and device timings against the
  IP-DOPRI5 replay are to be measured.

**An opt-out for the held field-free plane.** Under the low-storage march a field-free
plane holds one extra field volume so that a run stopped in flight can resume from it
([SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)). That volume is the
difference between the march's five volumes and a four-volume budget. Offer a setting
that drops it for runs that never resume, such as many short runs in a scan, with the
abandoned step then recorded without a checkpoint. The default stays as it is.

**The dispersive polynomial on mixtures of distinct leg bounds.**
[SEP 6](seps/archive/sep-0006-dispersive-nonlinearity.md)'s polynomial evaluator shares its
derivative fields only between species with equal leg bounds, so air (N2, O2 and Ar,
three bounds) costs 4.3 to 7.6 times the constant cubic's stage on `numba_cpu`, above
the 4 times of its target. Reduce the per-species transforms
without approximating the operator, or measure a bounded-support grouping first.

**The on-axis statistics on device builds.** The native backend computes
[SEP 19](seps/archive/sep-0019-field-scale-quadrature.md)'s `on_axis_*` statistics from a
host copy of the state, one per saved plane that requests them. A device scan of the
axis row, as the SEP's §6 describes, is not built.

**The device gallery under the active-band norm.** The stepper gallery's AMD CDNA 3 and
NVIDIA Hopper rows ([choosing a stepper](../guide/numerics/choosing-a-stepper.md)) were not
measured under the error norm that scores the whole active band, which the CPU rows use.
Regenerate the device rows under it and redraw the gallery.

**A device-sized node chunk.** The source's node chunk is sized by
`SimulationSettings.node_budget` (1 GiB by default on spatial grids, 64 MiB on modal
routes) at translation, whatever the backend. On a device a chunk is a kernel launch: the
128 × 128 Cartesian filament evaluates its plasma in 161 chunks a call, and with every
column in one chunk IP-DOPRI5 ran 4 times faster and the exponential Adams method 2.4
times. A chunk of any size gives the same fields to rounding, so the device backend can
choose its own from the device's memory when it is built, without translation importing
the device runtime, which a scan must not initialise before it forks its workers.

**Plasma statistics between planes from the peak-field node.** A per-step statistics
row's plasma statistics scan every quadrature node's time column. The Runge–Kutta march
reads them off its last stage, which is the accepted state; an Adams step's one
evaluation is at its predicted state, so under the default a row scans afresh and, at a
period of one, adds 80 % to the reduced M2 run ([statistics](../guide/results/statistics.md)).
`electron_density` and `peak_ionisation_rate` describe one node, the one of largest
ionised fraction: synthesising every node (a cheap matrix product) and scanning only the
node of largest peak field, or the two or three largest, would cut a row's scan by the
node count (8 to 24 on a capillary, about 256 on a radial free-space grid). The peak
need not be on the axis (a TE₀₁ or ring-shaped field), and the node of largest field is
not always the node of largest fraction where pulse shapes differ across the beam, so
the rule is to be checked against the full scan on the qualification cases and applied
to plane statistics too, keeping a row the plane's own to rounding. Statistics resolved
over every node (`species_ionisation`) keep the full scan.

**The split-step collar on the general modal route, native against the reference.**
With `absorber_scheme="split_step"` on a general modal guide, `numpy_ref` carries a
launch's content outside the active band (it decays over the steps) and `native` sets it
to zero at once; inside the active band the two agree, and with the in-stage collar or
none they agree everywhere. Not yet traced: one of the two
applies the hard band limit where the other does not. [SEP 21](seps/sep-0021-absorbing-boundaries.md)'s campaign
needs it resolved before it compares the split scheme on a general modal guide.

**Fixed-order Adams norms on devices.** On a native host build the Adams engine's
magnitude and defect norms sum fixed chunks in index order and combine the partials
serially, so an adaptive run is the same at every thread count, as the backends owe
([backends](backends.md)). A device build still reduces them with Kokkos's own
`parallel_reduce`, whose association is the launch's; the fixed-order device form (a
fixed block count, each block reducing its own contiguous range) is to be built and
measured.

**Spatial convergence records on CUDA.** The spatial routes' convergence records
(`spatial_accuracy`, and `backend_equivalence` for the polarised and free-space routes)
have no `native_cuda` rows, so the native-parametrised tests that read them cannot pass on
a CUDA build until they are recorded there.

**Susceptibility sampling on tabulated and rectangular modes.** Along a pressure gradient,
these modes evaluate the gas susceptibility, and its dilute-scaling check, at every z
sample instead of once per frequency array as the capillary modes do. It costs time, not
accuracy; worth sharing the capillary's sampling if a gradient run on these modes shows
the cost.

**Execution autotuning beyond existing CPU controls.**
[SEP 4](seps/sep-0004-execution-autotuning.md) covers explicit search over existing
CPU pool, FFT-planning and OpenMP policies, including plan application and validation.
Runtime-selectable Kokkos GEMM providers and GPU team/reduction geometries remain open:
they need an enumerated, qualified native capability set before becoming candidates.
[SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md) profiled data-centre NVIDIA and
AMD GPUs and tiled the device band column; runtime enumeration and automatic policy
selection remain with this informal item, as do the coupled cases' launch and copy
counts that SEP 25 records.
Chunk-size tuning needs route-specific memory admission; the general modal route does
not use the ordinary transform's `node_chunk`. Propagator-policy tuning must account
for the requested mode in today's resume hash and route-specific refusals. Implicit
tuning on the convenience interfaces, broad reuse across scan points and native batch
tuning also remain outside SEP 4; none may infer full-run savings from a short prefix.

**CF4 on the compiled backends.** The commutator-free fourth-order flow, the certified
alternative to the frozen split for a varying coupled flow, runs on `numpy_ref` alone.
The frozen split is the default and the more accurate on every case measured, so CF4 on
`numba_cpu` and `native` is needed only if a use case favours it; its Gauss-node weights
are evaluated inside the backend, not carried in the stage bundle.

**The z table on GPU builds.** Every CPU backend runs the general z table, the native
CPU build in its own kernels (SEP 29 section 6.6), but the native GPU builds refuse it,
naming its size: the table is the state's size times the degree times the intervals,
held resident. Two routes would admit it there. A memory check against
`available_device_memory` before allocating would admit the tables that fit, at the cost
of device qualification of the table kernel. For a `SpecifiedMode` whose dispersion is
declared at several positions along the guide, a compiled representation would declare
the interpolation coordinate a parameter and expand the dispersion in it, as the operator
expansion already does for a fill's density, so the GPU builds could run it without the
table.

**A compiled closure synthesis for `numba_cpu`.** The local spatial statistics (fluence,
intensities, the on-axis spectrum) need the closure `G` formed with the complex
admittance; `numba_cpu` forms that volume with NumPy FFTs on the host, where native forms
it in its own kernels ([SEP 34](seps/archive/sep-0034-medium-flux.md), phase 2, Q2).
A compiled pass would remove the host round trip.

**The cost of a kept native save on Linux.** On the snapshot benchmark case a kept
native Adams save costs about 4.5 ms against `numba_cpu`'s 1.6 ms on Linux, unexplained;
the run totals are level
([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md), phase 2).

**The hybrid planar route on GPUs.** native CPU executes the planar kind resolved in y; the
CUDA and HIP builds have run it only in one short Kerr case of
[SEP 34](seps/archive/sep-0034-medium-flux.md)'s device qualification. Its full device
qualification follows the other routes' parity bounds ([backends](backends.md)).

**The far-field pass on GPUs.** The CUDA and HIP builds compile the native backend's
far-field pass ([SEP 23](seps/archive/sep-0023-far-field-aperture-spectra.md)) but have
run it only on the short Cartesian and line cases of
[SEP 34](seps/archive/sep-0034-medium-flux.md)'s device qualification: its band partials live in global memory, with a team barrier per column. Device
qualification measures it against the CPU parity bounds of
`tests/backends/test_native_far_field.py`, and its optimisation holds a thread's slot
accumulators in registers and runs a large aperture set in groups (SEP 23 §6).

**Multi-channel columns on GPUs.** On the CUDA and HIP builds a column with several
ionisation or population channels (a gas mixture, for example) takes a serial loop over
its nodes; only the single-channel column uses the team-parallel path. Device
qualification covers the multi-channel results, not their speed;
[SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)'s device optimisation
included these columns, and a channel-specific redesign remains open.

**Device qualification of optmat material runs.** No run with optmat materials, Raman
responses or ionisation rates has been qualified on a CUDA or HIP build. The kernels see
only material-agnostic arrays, so no change is expected; the next device qualification
is to include one.

**GPU continuous integration.** The gate runs on CPUs. The native CUDA and HIP builds are
qualified by hand on data-centre NVIDIA and AMD Instinct GPUs; a scheduled device lane
would catch regressions in the `gpu`-marked tests between qualifications.

**Native source builds without a checkout.** The published wheels are built from the
repository, but `solprop-native` cannot be built from a source archive or offline: its
CMake configure records its own commit through `git rev-parse` and fails without one, and
by default it fetches Kokkos, Kokkos Kernels and (for the Linux distribution build)
OpenBLAS from GitHub ([dependencies](dependencies.md)). A release archive that builds
needs the commit to come from the package version and the sources to be vendored or
fetchable by checksum.

**Wheel qualification beyond CI**
([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md), §7). The wheels' test
jobs run on GitHub's runners, which carry compilers and current operating systems. Still
to do: installs on clean machines at the declared minimum systems (macOS 14, the
`manylinux_2_28` generation, Windows 10); the slow lane on the native backend from the
wheels, not only from a source build; and a lane that fails on unexpected native skips.

**Per-platform performance of the wheels**
([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md), §7). The standard
suite's core cases, adaptive Multistep and Runge–Kutta, native against `numba_cpu` with
matched budgets, at one thread and the production width, on each wheel family, recorded
in the benchmark ledger with any serious loss per case. To run once the first release is
out.

**macOS: pyFFTW imported before the native package.** pyFFTW's macOS wheel carries and
initialises its own `libomp`, and the native package refuses to import after it (the
rule is documented in [Building the native backend](../guide/performance/building-native.md)).
Lifting the rule needs a macOS native build without OpenMP (Kokkos Threads and threaded
FFTW, measured first) or a pyFFTW wheel that does not initialise its own runtime.

**The future of `numba_cpu`.** With the native wheels published, a later SEP decides
whether `numba_cpu` stays, on evidence from use: how often `auto` falls back, coverage
across the installed platforms, default performance and coexistence reports
([SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md), §6.3).

**Raman convolution length.** The Raman convolution is always built on a doubled grid of
`2 n_to` samples, including for short-lived responses such as silica's.
[SEP 1](seps/sep-0001-raman-response.md) partly addresses this cost with exact sampled
recurrences for supported kernels and the existing lag-taper correction. A convolution
sized to the response's useful support remains outside its implementation phases; it must stay causal, bound its truncation
error and be benchmarked on short runs.

**Batch fault isolation.** In a lockstep scan batch, a fault at group level (a
non-finite error norm, for example) fails every member of the group, so one diverging
point can cost up to the group size less one points that `Local` or `Pool` would
complete. Re-dispatching the survivors through the per-point path is the preferred fix;
either it or a per-member abort changes the documented failure semantics of `Batch`
([scans](scans.md)).

**Runner reuse and field-free members in scans.** A scan worker constructs one backend
per point rather than keeping one `SimulationRunner` for its points, so a large spatial
scan pays a backend's allocation and transform planning per point; and a lockstep batch
(`observe_members`) observes every member's field at every plane, since `Batch`
requires fibre geometry. Both were left out of
[SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md): the first needs the
worker to check each point's problem against its runner's, the second a field-free
member observation.

**The compiled backends' extra volumes in the memory estimate.** `estimate_memory`
counts the reference's base buffers on every backend: on the Cartesian kind the
Runge–Kutta march holds 13 state volumes on `numpy_ref`, 16 on `numba_cpu` and 15 on
native CPU, and the difference, gigabytes on a large grid, is declared not counted
([SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md) §6.1). Count them per
backend so that admission is exact there too.

**Cold-start compilation.** `numba_cpu`'s parallel kernels are compiled in every new
process rather than read from numba's disk cache, because parallel closures compiled in
different processes can collide in the cache. A collision-free cache would remove the
first-run compile cost from every worker of a scan.

**OpenMP core places and the Numba allocation cap.** With `OMP_PLACES=cores` on a
machine with two hardware threads per core, OpenMP can bind the main thread to
two logical CPUs. `solprop._process.allowed_cpus` recognises a one-CPU runtime
binding but treats this two-CPU mask as a deliberate allocation restriction,
capping Numba, FFTW and BLAS at two despite a larger request. The
[SEP 27 campaign](seps/archive/sep-0027-held-plasma-current.md) exposed this in execution
metadata. Its corrected measurements use matching explicit places containing
one logical CPU per physical core and assert actual pool sizes. A general repair
must distinguish an OpenMP place from a deliberately narrowed scheduler grant
and retain the existing affinity tests.

**Held-current source cost and compiled attribution.** The
[SEP 27 campaign](seps/archive/sep-0027-held-plasma-current.md) measures the cost of replacing
the gated equation on existing routes. On its 18-thread radial case, Numba's
source costs 1.67–1.72 times the gated source and its ten-step window
1.53–1.67 times the gated window, while the native backend's corresponding ratios are
0.58–0.59 and 0.46–0.47. Other modal and spatial cases also have resolved increases;
the per-case measurements and noise ranges belong to the
[benchmark ledger](benchmarks.md). The NumPy profiles identify extra transforms and
chunk passes; compiled explanations still need direct profiling. Investigate the
current products, projections, transforms, copies and chunk counts without changing
the production equation or dropping accounted storage. The radial and modal routes
are outside [SEP 28](seps/archive/sep-0028-free-space-evaluation.md)'s free-space
evaluation and need their own work.

**The general route's memory estimate.** A general modal guide is sized before its route is
built and admitted against a declared budget. The estimate does not count the transient
knot table the reference is interpolated from, nor the base, delayed and flow groups, and
a large guide is warned on twice (at construction and at `run`).

**Large arrays in the run record.** A spatial run's metadata (`metadata_for` in
`solprop/driver/simulation.py`) carries the geometry's `spatial_weights`,
`spectral_weights`, `window` and `absorber_rate`, four numbers per transverse node, and a
mixture run's processing record (`processing_record` in
`solprop/driver/material_output.py`) carries the dense node synthesis matrix ($n_r^2$
numbers on a radial grid) and the species absorption cross sections. All of them are JSON
text inside the configuration that the resume hash covers: at 512 × 512 the four geometry
vectors alone are about 20 MB, serialised and hashed on every resume check. The geometry
vectors follow from the grid's parameters, and the matrices could be stored as datasets
outside the hashed text; either keeps the record small.

**The radial statistics synthesis on the native backend.** The radial spatial statistics
(`peak_intensity`, the fluence widths) synthesise the intensity at every node and time
sample by a direct $O(n_r^2 n_t)$ sum rather than a matrix product. These are per-stage
and per-plane times, not allocations.

**A copy-free Adams sample.** The Adams sample on `numba_cpu` makes 16
passes of which 4 are needed (identity fills, a stage clear, an unused accumulation, a
returned copy). A `numba_cpu` Adams engine through the existing hook, or a destination
argument for `_adams_lab_source` (a kernel-contract change), would remove them; either needs
lifetime and rejection tests across startup, restarts, corrected-field evaluations and
dense observation, for both formulations.

**The march's register traffic on devices.** On a data-centre NVIDIA GPU the low-storage
march takes 51.6 and 227 ms a step at $384^2$ and $768^2$ by 600 against 30.8 and 168.7 ms
for a dedicated single-transform engine. A transform probe showed the transforms near
their best (one rank-3 transform would save 4-12 %); the rest is the fold, already near
the card's bandwidth, and the volumes each pass reads and writes. A march that touches
fewer volumes per pass is the remaining lever.

**Population statistics without a current stash.** On a field-grid run with plasma
channels, each requested population statistic (the `species_*` statistics,
`electron_density`, `peak_ionisation_rate`) rescans the populations at every node: on
`numpy_ref` at every plane, and on `numba_cpu` when the accepted step's stash is not
current, as at the entrance. Seven requested names cost seven scans where one would do;
the envelope-plasma path already shares one scan between them.

## Tools and analysis

**Full dispersive-wave phase matching.** The design tools phase-match resonant
dispersive waves with the linear mismatch and an optional Kerr term
([design tools](../guide/workflows/design-tools.md)). A plasma term, a Raman term and the full
soliton wavenumber would predict emission where ionisation or the Raman shift moves the
soliton.

**Nonlinear diagnostics beyond the proposed statistics.**
[SEP 9](seps/sep-0009-propagation-statistics.md) covers the transverse-maximum modal
intensity, modal beam widths, filtered quadrature norms where an embedded rule exists,
and a polarisation-span residual. A general modal guide still needs an independently
qualified embedded coarse rule before it can report quadrature error. Extending the
span residual to plasma/current sources needs an explicit source-space convention;
neither extension is included in SEP 9's implementation scope.

**A custom spectral rate apart from the collar's.** `custom_absorber` sums the user's
power rate into the collar's `spectral_rate`, which is what the seam carries, so the
spectral channel of the far-field statistic `absorbed_power` reports the collar and the
user's rate together. An omit-default `user_rate` on `Absorber`, summed at translation,
would let the two be attributed without moving any record
([SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)).

**An absorber switch on the keyword interfaces.** `prop_capillary`, `prop_waveguide` and
their `Simulation.from_*` constructors build the default spectral and temporal collar
when `absorber=None`, and there is no keyword to run without an absorber or with the
split-step scheme: today that takes replacing the frozen `SimulationSettings` before the
first run ([boundaries](../guide/numerics/boundaries.md)). An explicit `absorber=False` and an
`absorber_scheme=` keyword would make both one argument. Related: spatial runs carry no
absorber unless one is given, although every spatial grid has a collar; whether they
should default to one deserves a design check. [SEP 21](seps/sep-0021-absorbing-boundaries.md) proposes the two
keywords and a qualified convenience path for monitored propagation without temporal
absorption while retaining the spectral collar; the spatial default is not in its scope.

**An offset or walk-off-centred time window.** The temporal grid is symmetric about the
frame, but walk-off is usually one-sided: the Raman self-frequency-shift example plots
−200 fs to +1.7 ps inside a ±1.75 ps window, so most of its leading half carries
nothing. An offset time axis, or a frame chosen to centre the expected walk-off, would
roughly halve such grids and keep content away from the temporal collar.
[SEP 21](seps/sep-0021-absorbing-boundaries.md) measures fixed asymmetric launch
placement and prescribed frames, including fastest-light and midpoint frames, and
delivers qualified setting support and guidance. The remaining work is
a general offset time-axis API, independently asymmetric absorber ramps, and adaptive
recentering or grid expansion, with consistent history, response memory, noise and
output-coordinate transport. Those require a separate design and remain outside SEP 21.

**A phase-space filter for the temporal boundary.** Soffer and Stucchio's
time-dependent filter (J. Comput. Phys. 225, 1218, 2007) deletes only the windowed
Fourier content a free flow would carry out of the window, so it adds no stiff term
to the equation. Its free-buffer assumption, its interval cap for broadband fields and
its restart of multistep history are untested here.
[SEP 21](seps/sep-0021-absorbing-boundaries.md) §4.9 analyses it; a prototype on the
Runge–Kutta path, against the wide-window reference, was removed from that SEP's
campaign and would be measured first.

**A directional collar occupancy.** Classifying the content of each temporal collar as
outgoing, incoming or unresolved from its windowed spectrum and group delay could help
separate wrap-around and reflection from ordinary outgoing radiation, though incoming
content can also be genuine returning physics. Removed from
[SEP 21](seps/sep-0021-absorbing-boundaries.md); worth evaluating offline on saved
planes against its wide-window cases before adding a diagnostic.

**A self-focusing collapse estimate.** The critical power is provided
({func}`~solprop.spatial.critical_power`), but not the Marburger estimate of the collapse
distance, which the radial tutorial computes by hand.

**A polarisation launch across mode rows.** The keyword interfaces fill only the first
row of a mode set, so a Jones vector spread over the two members of a degenerate pair
needs a hand-built input; a vector capillary run through the keyword interfaces is also
not yet validated end to end ([polarisation](../guide/setup/polarisation.md)).
[SEP 31](seps/archive/sep-0031-simulation-coupling.md) added explicit incident-field
polarisation and coherent projection across target channels. The keyword-interface
convenience and its complete vector-capillary qualification remain open.

**Structured scan values.** A scan axis takes one-dimensional scalar values, so a list of
paired gradient pressures is expressed as an integer index with the profile built in the
builder ([scans](../guide/workflows/scans.md)). Structured axis values would need explicit
serialisation and provenance in the scan file.

**Further offline elements.** [SEP 31](seps/archive/sep-0031-simulation-coupling.md)
added focusing, apertures and ordered field transport (`coupling.transport`) beside the
existing material, mirror, grating and prism transfer functions
([compression](../guide/workflows/compression.md)). Its aperture shares
`spatial.aperture_profile` and its weighting with the in-run aperture statistics, so the
two agree where their plane, flux and paraxial assumptions do
([SEP 23](seps/archive/sep-0023-far-field-aperture-spectra.md)).
Offline propagation through a z-varying waveguide mode as an element remains open.

**Field planes in the output schema.** Two changes wait for the next output-schema
change: `field_plane` on the general modal route's accessor whitelist (the whitelist
feeds the recorded schema), and the plane's frame recorded in `FieldPlane`, so that
`transport` no longer needs a `frame=` argument for a plane written in a frame other
than its medium's ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phases 2 and 4).

**Regrid's band loss in physical energy.** `regrid` reports the share of a plane lost
outside the new temporal band in the record's norm. Reporting it in physical energy as
well, through the per-bin admittance the transport core already evaluates, would give the
loss in joules beside the share ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phase 2).

**A public per-bin spatial energy spectrum.** The transport elements use a private
`_energy_spectrum` (`coupling/elements.py`); a public `spatial.spatial_energy_spectrum` per frequency bin would
replace it. In the same change, `spatial.aperture_profile`'s docstring should say that its
result is a field transmission under the default `"amplitude"` weighting and a power
weight under `"quadrature"` ([SEP 31](seps/archive/sep-0031-simulation-coupling.md), phase 4).

**Field planes from modified-noise runs.** `processing.field_plane` refuses every plane of
a run with `noise="modified"`, whose saved field leaves the one-photon background to the
auxiliary `noise_phase`. Reconstructing that phase into the plane, following the
documented saved-field meaning, would lift the refusal
([SEP 31](seps/archive/sep-0031-simulation-coupling.md), decision A2; [coupling simulations](../guide/workflows/coupling.md)).

**Far-field detection beyond SEP 23.** The far-field statistics
([statistics](../guide/results/statistics.md), *Far-field apertures and angular regions*)
leave the following in-run extensions for later, each as new names or fields that
leave the six statistics' shapes unchanged
([SEP 23](seps/archive/sep-0023-far-field-aperture-spectra.md)).
[SEP 31](seps/archive/sep-0031-simulation-coupling.md) added coherent offline observation
planes, apertures and polarisation optics; its partial coverage is noted below.

- *Polarisation-resolved detection*: an analyser Jones vector on `ApertureSet`, or
  Stokes-resolved aperture spectra. Per-component spectra answer only an analyser
  aligned with the grid's axes; any other angle needs $\operatorname{Re}A_xA_y^*$ too.
- *Tabulated radial profiles*: a user-supplied $W(\rho)$ for a measured
  apodisation, with its own interpolation and parity rules.
- *The exact mapping*: an option placing a component at
  $z_m\boldsymbol\kappa/k_z^{\rm vac}$ with the vacuum obliquity, for wide-angle
  emission, where the paraxial map's $O(\theta^2)$ error matters.
- *Other shapes*: rectangular and elliptical apertures, among them a
  two-dimensional slit (a line grid's aperture already is one).
- *Radial and planar-hybrid geometry*: refused today, having no Cartesian transverse
  wavevector; a centred aperture on a radial grid would weight the Hankel
  spectrum's rings. SEP 31 added radial offline transport and apertures;
  extensions to these in-run statistics remain open.

## Release preparation

**The first release.** Publish `solprop` and `solprop-native` at one version from a tag,
following the checklist in [Releasing and the native wheels](releasing.md): the wheels
workflow with publishing enabled, the archive DOI, and a fresh install of
`solprop[native]` on each wheel platform. The native wheels and their recipe were
qualified by [SEP 29](seps/archive/sep-0029-compiled-backend-distribution.md).

**A resume-compatibility policy before the first public release.** The output schema
records every field of a recorded dataclass, at its default too
(ADR-051 in [decisions](decisions.md)),
so a release that adds a field moves every configuration hash, and a run written by the
previous release can no longer be resumed. Before 1.0, decide between computing a
resume hash with the stored file's own schema rules and reintroducing omit-at-default
as a documented rule from 1.0 on.

## Known issues

**Large-V step-index modes.** `StepIndexMode`'s characteristic equation and its derivatives
use unscaled modified Bessel functions `kv`/`kvp`. At large cladding arguments both
underflow to zero while their ratio is finite (at argument 1600, $K_0$ and $K_1$ are both
zero; the scaled ratio is about 6.2e-4), and the division fails. A 300 µm, NA 0.22 fibre
at 250 nm fails at construction with an invalid division and no roots assigned. The fix is
exponentially scaled ratios throughout the characteristic equation, derivatives,
normalisation and exterior fields, qualified at large V and against independent
references at ordinary V; suppressing the warning would hide the failure.

**The zero-dispersion search misses paired zeros.** `modes.base.zdw`, and the `zdw`
statistic that uses it, test only the signs of $\beta_2$ at the ends of the window before a
bracketed root search. An even number of interior crossings — two zero-dispersion
wavelengths, or the sign changes of a gas resonance inside the window, as with krypton or
xenon — reads as no zero, and the search raises. The design tools' sampled search with its
pole filter does not have the defect. Reconciling the two needs a policy for several roots
(all roots, a seed, or the root nearest the pump) and regression tests for zero, one and
several roots and for poles.

**`MemoryOutput.planes()` with between-plane statistics.** For a run that records
statistics between save planes, `planes()` raises `OutputError` ("step statistics columns
must have equal row counts"). A step-statistics column created when the first plane
carrying rows arrives is short by the planes saved before it.
[SEP 9](seps/sep-0009-propagation-statistics.md) requires this existing defect to be
fixed before qualifying its additional periodic scalar statistics.

**Modal accessors on spatial files.** `processing.energy`, `get_power` and
`get_spectral_density` return a modal reading of a radial, Cartesian or planar file
instead of refusing it; the numbers they give there are not energies. The scan reductions
built on them already refuse spatial runs by name.

**Free-space flux beyond the scalar forward model.** Spatial energy, intensity and
fluence are the scalar axial flux of the medium at the plane
([SEP 34](seps/archive/sep-0034-medium-flux.md)), and a radial gas gradient is admitted only
within its real-index adiabatic certificate. Gradient scattering outside that
certificate, a radial path with varying (or any) loss, which needs a complex-index
bound, varying media on the Cartesian and line grids, and the full vector TE/TM flux at
high numerical aperture remain open. [SEP 8](seps/sep-0008-quadratic-crystal-optics.md)
requires E/H flux for its crystal route.

**A lossy multimode taper at the default step bound.** The taper coupling carries each
channel's loss inside its block, and over the default `dz_max` of half the fibre the
flow's conditioning guard refuses a strongly lossy taper. The refusal names `dz_max` and
the `taper_coupling=False` opt-out; a smaller default `dz_max` for multimode tapers would
avoid it.

**Scan queues and platform probes.** A live but wedged queue owner on the same host holds
its point indefinitely, because the lease age is not consulted while its process is alive.
An interrupted `Local` or `Pool` scan skips its `running` points for `stale_age` seconds on
restart. Container memory limits are read from the root cgroup rather than the process's
own cgroup path (CPU quotas follow the path, ADR-052), so a shared node under a cgroup
manager without Slurm variables is mis-sized for memory, and a failing platform probe raises outside the `SolpropError` hierarchy. The
scan file grows on every resume, because HDF5 does not reclaim the groups it recreates.
[SEP 4](seps/sep-0004-execution-autotuning.md) partly covers the platform probes:
it requires effective allocation detection shared with scans, including nested
cgroups or an explicit conservative allocation when detection is unavailable.
Queue leases, interrupted-point recovery and HDF5 growth remain separate work.

**The dispersion-only `Air` entry in a hand-built mixture.** `gas="air"` on the keyword
interfaces becomes a nitrogen/oxygen mixture with every channel. The built-in entry
`Air` itself carries only a refractive index: put into `Mixture.fill` by hand it has no
Kerr, plasma or Raman data: with `kerr=True` it is refused (`Air` has no Kerr data),
but with `kerr=False` a hand-built `Mixture.fill({"Air": 1.0}, ...)` run translates with
no nonlinear response and no warning. It should refuse by name or steer to `Mixture.air`
([mixtures](../guide/media/mixtures.md)).
[SEP 3](seps/archive/sep-0003-optmat-materials.md) partly covers this area by migrating air
data and named compositions; the handling of this hand-built mixture still needs an
explicit refusal or constituent expansion and a regression test.

**Provenance of the chirped-mirror curves.** The measured mirror GDD and reflectance
curves read by `ChirpedMirror.lookup` ([compression](../guide/workflows/compression.md)) carry
no vendor citation. They are device data, outside
[SEP 3](seps/archive/sep-0003-optmat-materials.md)'s material migration, and need their sources
and measurement conditions recorded, or replacing with cited curves.

**Further built-in materials.** [SEP 3](seps/archive/sep-0003-optmat-materials.md) migrates the
gases and solids solprop already ships, and adds none. optmat also recommends
candidates:
- CO2, with dispersion, Kerr, a Raman response with provisional Q-branch widths and a
  rate model; the first candidate;
- CO, with a rate but no Raman data;
- gold and copper, as tables;
- further crystals.

Each needs separate admission and qualification as a built-in material. O3 waits for
absorbing gas indices and non-ideal density to be handled properly.

**Memory keywords on the keyword interfaces.** `memory_budget=` is a keyword of
`Simulation.run` and `Simulation.from_waveguide`; the `prop_*` functions read only
`SOLPROP_MEMORY_BUDGET` ([choosing a backend](../guide/performance/resources.md#memory)).
Likewise the field retention of [SEP 22](seps/archive/sep-0022-memory-lean-spatial-execution.md)
(`fields=`, `field_planes=`) is set on `SimulationSettings` or through
`run_scan`, not as a keyword of the `prop_*` functions. Offer both there together.

**Two slow coupling-envelope tests in the medium tier.** Two of the recorded-envelope
oracles in `tests/coupling/test_envelopes.py` take about 30–50 s each in the medium tier.
If CI time becomes a constraint, they can move to the slow tier, with a cheaper check of
the same envelope left in the medium tier ([SEP 31](seps/archive/sep-0031-simulation-coupling.md)).
