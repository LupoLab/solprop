# Vector fields, nonlinear projection and power

A general modal calculation reconstructs the electric field across each material region,
evaluates its nonlinear polarisation and projects that source back onto the guided
channels. The electric and magnetic mode fields define the power metric used in that
projection. The [practical workflow](solid-fibres.md) introduces the objects; this
chapter gives their physical meaning and equations.

## The three objects

A **provider** calculates the modes at a frequency and position. A **table**
samples those solutions across the required band and follows each physical
channel between samples. The **synthesis and projection maps** then reconstruct
the field and its nonlinear source on the simulation grid. These are distinct
sources of numerical error: mode solving, interpolation and transverse integration.
The record layout used to execute them is documented in the
[development notes](../../dev/architecture.md#5-data-flow-of-one-run).

A **provider** ({mod}`solprop.modes.vector`) solves one frequency and position on a
declared rule of Cartesian nodes and returns the three complex Cartesian components of
$\mathbf E$ and $\mathbf H$ per √W, the generator $-i\beta$, the Hermitian flux metric
and its reciprocal partners, all in the internal Fourier convention. A
{class}`~solprop.translate.modal_tables.DenseModalTable` samples a provider over a band,
tracks each channel through it and **refuses a basis whose sampled Gram is further from
the identity than its declared budget** — a rule that truncates or under-resolves the
fields returns a basis that is not orthonormal where it is sampled, and no provider
renormalises that away. {func}`~solprop.translate.modal_tables.modal_transform_tables`
evaluates the table on the grids a run will use and stores the execution maps.

{meth}`~solprop.modes.guide.ModalGuide.from_provider` records what it solved on — the
knots, the position, the node rule, the declared degenerate blocks and the **measured**
profile interpolation error — as a {class}`~solprop.modes.guide.ModalSampling`, and that
is what a run rebuilds its reference from. A guide built by
{meth}`~solprop.modes.guide.ModalGuide.from_step_index` carries no sampling, because
nothing there declares a transverse resolution, and is refused by name when a run asks
for it.

## The synthesis and projection maps

The state is a set of channel amplitudes $a_m(\omega)$ in √W on the run's coarse grid.
Two maps carry it to the nodes and back.

**Synthesis** $S$ is the channel's own electric profile, tabulated on the coarse
spectral grid — the one the state lives on — in V/(m √W):

$$\tilde E_c(q,\omega) = \sum_m S(q,c,m,\omega)\, a_m(\omega).$$

Because $S$ is complex and depends on frequency, the synthesis happens in the frequency
domain — a complex profile can only act on an analytic signal, not on a real time trace.
The synthesised spectrum is placed in the run's oversampled fine spectrum, where the
material response is evaluated; the oversampled grid is where the *field* lives, not
where the map does, and a map tabulated there would carry columns no state can reach.
The one-sided spectrum is doubled away from its zero and Nyquist bins, which is exactly
{func}`solprop.fourier.analytic_signal`'s convention, so one inverse transform per node
and component gives the analytic field $\mathbf E_a(t)$ whose real part is the physical
field. This reconstructed field is in V/m; its squared modulus is not a power
in watts. Physical power uses the electric and magnetic fields through the
Poynting pairing described in [power and intensity](../results/power-and-intensity.md).

**Projection** $Q$ is the source functional of a bound reciprocal basis, on the coarse
grid, in V m/√W:

$$\frac{\mathrm d a_m}{\mathrm d z}
  = -i\omega\,[W^{-1}]_{mn}\int \mathbf e_n^{*}\!\cdot\mathbf P\,\mathrm dA / 4,
\qquad
Q(m,q,c) = [W^{-1}]_{mn}\,\overline{e_n(q,c)}\,w_q / 4 ,$$

with $W$ the Hermitian flux metric and $w_q$ the area weights. The $-i\omega$ is kept
separately, as `source_frequency_factor`, because it is a property of the output
frequency and not of the profiles: **every spectral source uses the output frequency's
projection and prefactor**, while each participating input uses its own. Replacing the
whole four-frequency interaction by a single $\gamma(\omega)$ would not be the same
model.

Three frequency dependences meet in this source, and none stands in for another. The
**modal** one is in $S$ and $Q$: each input is synthesised with its own frequency's
profile and the output projected with its own. The **UPPE frequency factor** is the
$-i\omega$ of the output, which in the envelope limit is self-steepening. The
**material** one is the susceptibility's dependence on all four frequencies of each
quartet; a region's constant $\chi^{(3)}$ has none, while a region declaring its gas
fill takes the run's dispersive electronic policy
([a frequency-dependent electronic response](../media/dispersive-kerr.md); see
[the response at a node](modal-projection.md#the-response-at-a-node)). A frequency-dependent area or
overlap is geometry; it composes with material
dispersion and does not replace it.

Source projection and amplitude recovery are different operations. The record also
carries a **coordinate dual**, $a = W^{-1}B$ with
$B_m = \int(\mathbf E\times\mathbf h_m^{*} + \mathbf e_m^{*}\times\mathbf H)_z\,\mathrm dA/4$,
which recovers amplitudes from a physical field and is *not* the nonlinear projection.

**The nonlinear prefactor is not the UPPE factor on this route.** On the scalar routes
the prefactor row `responses.prefactor` carries $-i\omega/(2\varepsilon_0c\,\mathrm{Re}\,
n_{\rm eff})$ ([the kernel problem](../../dev/kernel-problem.md#where-the-normalisations-live)).
Here the $-i\omega$ is the record's own `source_frequency_factor` and the reciprocal
normalisation is inside $Q$, so what is left for `responses.prefactor` is the spectral
crop scale $n_t/n_{to}$, which is where the crop scale is on every route.

## The response at a node

Each node belongs to exactly one material region, and each region declares its own
responses through {class}`~solprop.modes.material.MaterialKerr`, by $\chi^{(3)}$ or by
$n_2$. A declared $n_2$ is converted with
$\chi^{(3)} = 4\varepsilon_0 c\,n_0^2 n_2/3$ against the region's **own** linear index,
and is then constant across the band.

**The route's cubic is the real one.** `kerr_form="real"`, the default, evaluates the
real-field cubic the field-grid routes elsewhere in this manual evaluate,

$$\mathbf P = \varepsilon_0\chi^{(3)}(\mathbf E\cdot\mathbf E)\,\mathbf E ,
  \qquad \mathbf E = \operatorname{Re}\mathbf E_a ,$$

which radiates the **third harmonic** along with the fundamental.
`kerr_form="analytic"` evaluates instead the analytic-signal form

$$\mathbf P = \frac{\varepsilon_0\chi^{(3)}}{4}
  \left(2|\mathbf E|^2\mathbf E + (\mathbf E\cdot\mathbf E)\mathbf E^{*}\right),$$

which carries the fundamental band alone — what an envelope route models, and therefore
what a like-for-like comparison with {func}`~solprop.driver.interface.prop_gnlse` wants.
The terms with two unconjugated factors and one conjugated factor in the real
cubic sum to $\operatorname{Re}$ of the analytic-signal polarisation above.
This identifies the fundamental contribution when it is spectrally separated
from the third-harmonic contribution. It does not imply identical propagated
fields for arbitrary broadband pulses, whose generated bands can overlap. The
algebraic comparison on the reference strand was measured at
1.59 × 10⁻¹⁷ of the source's own scale on the reference strand, which is rounding.

{func}`~solprop.translate.responses.isotropic_kerr_tensor` writes both as the
dimensionless rank-four tensor
$T_{ijkl} = (\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})/s$
of the general contraction
$P_i = \varepsilon_0\chi^{(3)}T_{ijkl}E_jE_k\overline{E_l}$ — **the conjugated field is
the last index** — with $s=4$ for the analytic signal and $s=3$ for the real field,
which is what makes the scalar limit $\varepsilon_0\chi^{(3)}E^3$ come out right. Which
one a run used is in its record, as the `model_id` `isotropic-kerr-real` or
`isotropic-kerr`.

The real cubic needs room for its harmonic: the source is evaluated on the fine grid and
cropped back, so the folded harmonic must land outside the coarse band, which it does
exactly when $n_{to} \ge 2 n_t$ — `FieldGrid`'s default `ffac=6`. A narrower grid is
**refused by name** rather than aliased into.

**The delayed response: the silica tensor Raman model.** A region may declare
{class}`~solprop.nonlinear.raman.SilicaTensorResponse` beside its `MaterialKerr`, and
the two then share the region's one $\chi^{(3)}$: the instantaneous tensor keeps
$1 - f_R$ and the delayed channels carry $\tfrac32 f_R$, which is the field-domain
amplitude that makes the two add to **one** $\chi^{(3)}$ on the fundamental
($\tfrac23\cdot\tfrac32 f_R + (1-f_R) = 1$; see
[the Raman page](../media/raman-models.md#fused-silica-as-a-vector-response) for why a field-domain
kernel of integral $X$ acts there like a Kerr term of $\tfrac23X$).
It is the Lin–Agrawal vector response
of fused silica — Opt. Lett. **31**, 3086 (2006), Eq. (1), with an isotropic delayed term in $a(\tau)$ and a
dyadic one in $b(\tau)$,

$$P_i^{\rm R} = \varepsilon_0\chi^{(3)}\Big[E_i\!\int_0^\infty\! a\,
   (\mathbf E\cdot\mathbf E)(t-\tau)\,\mathrm d\tau
   + \sum_j E_j\!\int_0^\infty\! b\,E_i(t-\tau)E_j(t-\tau)\,\mathrm d\tau\Big],$$

written on the two channels the seam carries: the **trace** channel and the
three-dimensional traceless **rank-two** channel $T(Q) = (3Q - \operatorname{tr}Q\,I)/2$.
Matching the two forms gives $r_{\rm trace} = a + b/3$ and $r_{\rm rank} = 2b/3$, whose
**shape** integrals are $w_a + (w_b + w_{ab})/3$ and $2(w_b + w_{ab})/3$ — **5/6** and
**1/6** for the declared weights, because $w_a = 0.75$ — and not the paper's own 3/4 and
1/4. The drive is the
**six** independent entries of the real symmetric pair history $Q_{kl} = E_kE_l$; the
convolution runs on a doubled, zero-padded axis so that a hundred-femtosecond tail
cannot wrap onto the head of the pulse, and its endpoint rule is the trapezoid, because
$h_b(0^+)$ is not zero.

By default the shapes and $f_R = 0.245$ are optmat's tensor silica bundle
(`SilicaBundle.resolve(tensor=True)`), Lin and Agrawal's: $\tau_1 = 12.2$ fs,
$\tau_2 = 32$ fs, $\tau_b = 96$ fs and weights 0.75/0.21/0.04, the published values of
Opt. Lett. **31**, 3086 (2006), which a caller may also give explicitly as a
{class}`~solprop.physdata.raman_types.LinAgrawalRaman`; the tensor structure is that of Lin and Agrawal,
J. Opt. Soc. Am. B **20**, 1616 (2003). For a
fixed linear polarisation the model reduces exactly to the standard fibre one,
$E(t)\int \tfrac32 f_R h_R(\tau)E^2(t-\tau)\mathrm d\tau$ with $h_R = a + b$
integrating to one;
a **cross-polarised** probe arriving after the pump instead sees $a(\tau)$ alone, three
quarters of the response and with the boson peak cancelled out of it, which is the
delay a scalar model cannot produce. The scalar silica response of the other routes
({func}`~solprop.nonlinear.raman.response_for`, the Hollenbeck–Cantrell shape of
optmat's default silica bundle) is a different model, reached by a different route
([the Raman response](../media/raman.md)).

A region that declares no response has no
polarisation at its nodes, which is how a linear cladding is said to be linear, and a
guide whose regions are *all* linear skips the source pass entirely.

**A region filled with gas.** In place of a `MaterialKerr`, a region may declare its
gas fill, {class}`~solprop.modes.material.GasKerr`, holding a
{class}`~solprop.physdata.mixtures.Mixture` of constant densities: a silica strand in
argon, say, whose evanescent field responds to the gas. Its electronic response is then
the gas's own, under the run's `materials=` selection
([materials](../media/materials.md)). Under a carrier policy (the default) the region is the
constant cubic above with $\chi^{(3)} = \sum_s N_s\gamma_{3,s}(\lambda_0)$, the Kerr
coefficient at the run's carrier. Under one of the dispersive policies of
[SEP 6](../../dev/seps/archive/sep-0006-dispersive-nonlinearity.md), `"symmetric-kerr"` or
`"gas-polynomial"`, each species of the fill is a channel of the frequency-dependent
cubic ({class}`~solprop.nonlinear.responses.DispersiveKerrResponse`) acting at the
region's nodes on the real three-component field, the scalar kernel times the
isotropic Kleinman tensor: `kerr_form="real"` takes its full cubic and `"analytic"` its
number-conserving sector, and the harmonic room above is required alike. Under
`"symmetric-kerr"` a fill of several species takes one filter from its own Kerr
diagonal, as any symmetric mixture does
([mixtures](../media/dispersive-kerr.md#mixtures-and-vectors)). A
`MaterialKerr` region stays the constant cubic under every policy, so a guide may carry
both laws, one region each. The fill is the region's nonlinear medium only; its linear
law is still the region's `material`. The dispersive channels run on `numpy_ref`,
`numba_cpu` and `native`; the compiled backends apply their filters in the route's
synthesis. The record names the fill in the guide's
identity, the channels under `modal["dispersive"]` and each channel's model, support
and spectral regions, and a mixture's filter with the composition it is built from, in
the material manifest.

**An `n2` declares the wavelength it was measured at.** A region's compiled leaf carries
one susceptibility, so a declared $n_2$ becomes one $\chi^{(3)}$ through
$\chi^{(3)} = 4\varepsilon_0 c n_0^2 n_2/3$ at one frequency, and the choice of
frequency multiplies the result by the square of an index ratio — 2.13 % for fused
silica between 800 nm and 2 µm. `MaterialKerr(n2=..., at=<wavelength in m>)` is that
declaration; with no `at`, the conversion is made at the run's own reference frequency.
Because nothing downstream of the leaf can say which frequency the number came from,
the route **refuses** an undeclared `n2` whenever the region's own index moves the
converted $\chi^{(3)}$ by more than
{data}`~solprop.translate.general.N2_CONVERSION_RTOL` (one per cent) across the run's
active band: declare `at=`, or declare `chi3=` outright. A constant index, or a gas
below a few bar, never reaches the budget and needs no declaration.

Putting that polarisation into the source functional at unit amplitude leaves the
**vector** self-phase-modulation coefficient

$$\gamma = \frac{\omega\varepsilon_0}{16\,W_{mm}}\int\chi^{(3)}
  \left(2|\mathbf E|^4 + |\mathbf E\cdot\mathbf E|^2\right)\mathrm dA ,$$

which is {func}`solprop.nonlinear.normalisation.modal_kerr_coefficient`. Its `metric=`
is the metric the route's own contraction shares: a positive float read as the channel's
own sampled $W_{mm}$, or the whole Hermitian metric of a degenerate block with
`channel=` saying which row this field is, in which case the coefficient carries
$[W^{-1}]_{mm}$ — which equals $1/W_{mm}$ only while $W$ is diagonal. Omitted, the
identity is assumed, which is what a provider's normalisation declares. Both
the longitudinal field and the $|\mathbf E\cdot\mathbf E|^2$ term of a vector channel are
in it and neither is in the scalar $n_2\omega/(cA_{\rm eff})$: on a 250 nm silica strand
the vector coefficient is 25 % larger (0.541771892 against 0.432424108 W⁻¹m⁻¹),
while on a weakly guiding 4 µm core at a 0.01 index step the two agree to
**5.1 × 10⁻⁴** — which is what weak guidance means, and which is the limit in which the
scalar routes of the rest of this manual are the same physics.

## Normalisation, and why the power weight is one

The step-index provider solves each mode once per frequency and axial position
within a normalisation integral, then evaluates that solved mode at the radial
quadrature points. Full-domain and exterior-tail integrals retain the same
adaptive tolerances. Reusing the solution avoids repeated eigenvalue searches
without changing the normalisation or adding an approximate mode cache.

A provider normalises each channel to unit longitudinal Poynting flux over its whole
physical domain, so a channel amplitude is a **true-power** amplitude from construction:
$|a|^2$ is watts, with no $\sqrt{\operatorname{Re}n_{\rm eff}}$ conversion anywhere. That
is already the true-power amplitude every route presents at its interface, so the run's
statistics power weight is the constant one and the general route never needs the
conversion the scalar mode-set routes apply between their internal amplitude and that
interface. The
basis metric is a separate statement: `linear.modal_metric` records *which* Gram the
basis has, and for a guide in the provider's own coordinates it is the identity,
certified by the table's construction gate and carrying the measured defect in its error
row. That defect is **gated, not merely recorded**: the route refuses a basis further
from orthonormal than its `gram_atol` (`DEFAULT_GRAM_ATOL`, 1e-4, by default), because
executing an identity metric that is not the identity makes a continuous wave's
$\mathrm da/\mathrm dz$ wrong by $|1/W_{00}-1|$ and nothing downstream would say so.

**And it is gated between the knots as well as at them.** A run does not propagate at the
table's knots; it propagates at its own bins, where the *interpolated* basis is further
from orthonormal than the sampled one — three orders further on a sparsely sampled guide.
The route therefore measures the interpolated Gram at those bins too, and gates it against
the same `gram_atol`, because to first order that departure is what declaring $W=I$ costs
the run's `physical_energy`. It is measured **where the launched light is**: the
interpolation error lives at the long-wavelength end of the sampled band, where a pulse
declared in the near infrared puts none of its own spectrum. On the weakly guiding test
fibre the whole-band figure is $1.1\times10^{-4}$ near 1.73 µm, inside the 790–890 nm
launch window it is $2.3\times10^{-7}$, and weighted by the pulse's own spectrum — which
is what `from_waveguide` passes, since it builds the launch first — it is
$3.7\times10^{-8}$. The number is on the record as `occupied_departure`, in the modal
identity beside the knot defect, and in the printed block. It is an interpolation error,
so a guide refused here is fixed by **adding knots**, not by widening the rule.

**The weighting is the launched spectrum, so the gate certifies the launch and not the
run.** The active band is where light *can* go, and a broadening run occupies bins the
launched spectrum discounted to nothing — on that same fibre the worst bin, 1.73 µm, is
inside `grid.active_mask`, so a run that broadened an octave would pay
$1.1\times10^{-4}$ of its `physical_energy` there rather than the $3.7\times10^{-8}$ the
gate saw. The route therefore also measures the **unweighted** maximum over the whole
active band, records it as `active_departure`, and emits a
{py:class}`~solprop.exceptions.GeneralModalBasisWarning` naming the worst bin's
wavelength when it leaves `gram_atol`. It warns rather than refusing because only you
know how far the run will broaden; the fix, again, is more knots.
