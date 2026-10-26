# Spatial decompositions

A spatial description determines which changes of beam shape a calculation can
represent. One effective area follows a prescribed mode; several guided modes
allow their amplitudes to exchange; a free-space grid resolves the field over a
transverse domain. Every description has boundaries and symmetry assumptions.
Choose one that includes the spatial processes you intend to study, then check
its resolution independently of the propagation step size.

The [illustrated introduction](../../introduction/spatial-description.md) gives
the first comparison. This chapter provides the detailed expansions, units,
boundaries and limitations. A saved field's transverse axis may label guided
modes or spatial frequencies: its meaning is set by this choice, not by the
array's shape alone. [Choosing a decomposition](#choosing-a-decomposition)
collects the decisions at the end.

The cross-section figures use the package's mode fields, quadrature nodes and
grid boundaries. They follow your colour-scheme preference and are drawn to
scale unless stated otherwise. They show a representation of the field, not a
propagated result.

## The map

What each decomposition represents, and what it assumes:

| decomposition | `n_modes` is | transform | transverse boundary | symmetry assumed |
|---|---|---|---|---|
| [mode-averaged](#mode-averaged-one-field-and-one-area) | one mode | none: one node, the mode's effective area | none resolved; the guide is in $n_{\rm eff}(\omega)$ | one fixed profile |
| [capillary, radial](#capillary-modes-on-a-radial-rule) | `HE_1m` channels | real matrix projection on a radial Gauss(–Kronrod) rule, one azimuthal node | field truncated at $r=a$; wall in the complex $n_{\rm eff}$ | azimuthal, one linear polarisation |
| [capillary, full 2-D](#capillary-modes-on-the-full-cross-section) | Marcatili modes of any order, or polarisation rows | real matrix projection on a polar product rule, $n_\theta \ge 4h+1$ | as above | none |
| [coherent wall](#the-capillary-wall-two-ways) | `HE_11`…`HE_1N`, or one azimuthal order's two families, coupled | as the radial or full rule, plus coupled linear propagation | condensed outgoing Robin condition at $r=a$ | as the radial rule (scalar), or one azimuthal order (`te_tm`) |
| [antiresonant](#antiresonant-tube-lattice) | Marcatili fields with Vincetti's index | as the capillary | field truncated at $r=a$; tubes in the fitted $n_{\rm eff}$ and loss | as the capillary |
| [step-index, exact](#step-index-fibres-exact) | one mode, or a provider's channels | mode-averaged, or the general route | $\mathbf E$, $\mathbf H$ matched at $r=a$; exact $K_n$ exterior | one bound, lossless mode at a declared wavelength |
| [graded-index](#graded-index-providers) | a provider's channels | the general route | exact decaying Bessel exterior beyond the matching radius | one radial sector (TE₀, TM₀ or LP) |
| [general modal, vector](#the-general-modal-route-vector-fields) | a provider's channels, with $\mathbf E$ and $\mathbf H$ | complex, frequency-dependent synthesis $S$ and projection $Q$ over three components | the provider's own | the provider's own |
| [a guide that moves along z](#guides-that-move-along-z) | as its route | rescaled profiles; the capillary's connection; moving $S(z)$, $Q(z)$ | as its route | adiabatic unless the connection is carried |
| [rectangular hollow](#rectangular-hollow-guides) | `RectMode` rows, x and y partners separate | real matrix projection on a Cartesian Gauss rule, $6h+6$ nodes per axis | field truncated at the walls; both wall pairs in $n_{\rm eff}$ | none |
| [radial free space](#radial-free-space) | transverse Hankel frequencies $k_r$ | order-zero QDHT, a dense matrix transform | $E = 0$ at $R$; absorbing collar when an absorber is declared | cylindrical symmetry of each Cartesian component |
| [Cartesian free space](#cartesian-free-space) | $(k_x, k_y)$ columns, x fastest | 2-D FFT | periodic; absorbing collar when an absorber is declared | none |
| [planar, uniform in y](#planar-hollow-guides-uniform-in-y) | `SlabMode` rows, TE and TM separate | real matrix projection on a slab Gauss(–Kronrod) rule in x, one y node of unit width | field truncated at the plates; one wall pair in $n_{\rm eff}$ | uniform in y |
| [line free space](#line-free-space-x-only) | $k_x$ columns, one $k_y = 0$ | 1-D FFT: the Cartesian FFT with $n_y = 1$ | periodic in x; absorbing collar in x when an absorber is declared | uniform in y |
| [planar, resolved in y](#planar-guides-resolved-in-y) | `(p, k_y)` columns, the slab channel fastest | slab rule's real matrix projection in x times an FFT in y | field truncated at the plates, one wall pair per channel; periodic in y with an absorbing collar | one polarisation family, constant gap and media |
| [radial and azimuthal](#radial-and-azimuthal-free-space-considered-not-built) | considered, not built | | | |

How each is reached, and where it runs:

| decomposition | amplitude, energy | entry point | backends |
|---|---|---|---|
| [mode-averaged](#mode-averaged-one-field-and-one-area) | √W, J | `prop_gnlse`, `Simulation.from_gnlse`; `prop_capillary(mode_average=True)` | all three |
| [capillary, radial](#capillary-modes-on-a-radial-rule) | √W, J | `prop_capillary(modes=N)`, `Simulation.from_capillary` | all three |
| [capillary, full 2-D](#capillary-modes-on-the-full-cross-section) | √W, J | `prop_waveguide(modes, ..., gas=...)`, `Simulation.from_waveguide` | all three |
| [coherent wall](#the-capillary-wall-two-ways) | √W, J | `prop_capillary(..., wall=CoherentWall())` | all three (CF4 on `numpy_ref` only) |
| [antiresonant](#antiresonant-tube-lattice) | √W, J | `prop_capillary(..., lattice=TubeLattice(...))` | all three |
| [step-index, exact](#step-index-fibres-exact) | √W, J | `prop_gnlse(fibre=StepIndexMode(...))`; the general route | all three |
| [graded-index](#graded-index-providers) | √W true power, J | `prop_waveguide((), ..., provider=guide)` | all three |
| [general modal, vector](#the-general-modal-route-vector-fields) | √W true power, J | `prop_waveguide((), ..., provider=guide)`, `Simulation.from_waveguide` | all three |
| [a guide that moves along z](#guides-that-move-along-z) | √W, J | the route's own, with a `ZProfile` | all three, including a moving solid guide |
| [rectangular hollow](#rectangular-hollow-guides) | √W, J | `prop_waveguide(modes, ..., gas=...)` | all three |
| [planar, uniform in y](#planar-hollow-guides-uniform-in-y) | √(W/m), J/m: per metre of y | `prop_waveguide(modes, ..., gas=...)`, `ModeCollection.slab` | all three |
| [radial free space](#radial-free-space) | V m (spectrum), J | `prop_radial`, `Simulation.from_radial` | all three |
| [Cartesian free space](#cartesian-free-space) | V/m, J | `prop_free`, `Simulation.from_free` | all three |
| [line free space](#line-free-space-x-only) | V/m, J/m: per metre of y | `prop_free(grid, LineGrid(...), ...)` | all three |
| [planar, resolved in y](#planar-guides-resolved-in-y) | √(W/m) per y frequency, J | `prop_planar`, `Simulation.from_planar` with a `PlanarGrid` | all three |

"All three" backends means `numpy_ref`, `numba_cpu` and the native backend's CPU build.
The native backend's GPU builds (CUDA and HIP) also run the mode-averaged, multimode, general modal,
coherent-wall and moving-guide routes, and the CUDA build runs radial, Cartesian and line
free space; the planar guide resolved in y is not qualified on a GPU, and a CF4
section runs on `numpy_ref` alone ([choosing a backend](../performance/backends.md),
[the native backend](../performance/native.md)).
Every modal route writes energy in joules and amplitudes in the true-power √W of
[power and intensity](../results/power-and-intensity.md): $b = \sqrt{\operatorname{Re}n_{\rm eff}}\,a$
on the scalar routes, and the provider's own normalisation on the general route, where
the power weight is one. The planar guide uniform in y is the exception: its integrals
are over x per metre of y, so its energies are J/m and its amplitudes √(W/m), and the run
records that it is (`RunMetadata.transverse_measure`); so is the line free space.

## Mode-averaged: one field and one area

```{figure} ../figures/mode_averaged.svg
:width: 100%
:alt: A capillary cross-section with its HE11 profile and the uniform disc of equal effective area; beneath, the two intensity profiles along x.

The mode-averaged route replaces the mode by a uniform disc of area $A_{\rm eff}$. The
Kerr term is the same either way; the on-axis intensity is not.
```

**Expansion.** One scalar amplitude, $E(x, y, t) = A(t)\,e(x,y)$, of which only two
numbers of the profile survive: the effective index $n_{\rm eff}(\omega)$ and the area
$A_{\rm eff} = (\int|e|^2\,dA)^2/\int|e|^4\,dA$. The response sees the field
$E = A/\sqrt{\varepsilon_0 c A_{\rm eff}/2}$ at one "node", which is the mode itself
({func}`~solprop.translate.modal.mode_averaged_transform`).

**Boundary.** None is resolved: whatever confines the mode is in $n_{\rm eff}$ and its
loss.

**Units.** $A$ is in √W and the energy in J. The GNLSE's mapping keeps the unit index
weight, $w = 1$ ([the GNLSE](../waveguides/gnlse.md)).

**Propagation.** The one-node transform, with the synthesis
$1/\sqrt{\varepsilon_0 c A_{\rm eff}/2}$ and the projection one, and the prefactor
{func}`~solprop.nonlinear.normalisation.mode_average_prefactor`. It is the one-node case
of the multimode pipeline, not a second code path ([the solver
design](../../dev/stepper.md#one-step-pass-by-pass)). A varying
area is one scalar along z, `transform.taper_scale` ([the kernel problem](../../dev/kernel-problem.md)).

**Valid when** one guided mode carries the pulse and its profile does not depend on the
intensity. The self-phase modulation is then exact, because $A_{\rm eff}$ is defined so
that the quartic overlap is $\int\hat e^4\,dA = 4/[(\varepsilon_0c)^2A_{\rm eff}]$
([the transverse quadrature](../waveguides/mode-sets.md#what-the-tensor-is-worth-checking-against)). A
one-mode run on a nine-node radial rule reproduces a mode-averaged run to
$1.1\times10^{-11}$, the rule's own error ([the solver
design](../../dev/stepper.md#one-step-pass-by-pass)).

**Not valid when** energy moves between modes, or when the beam's shape changes with
power: self-focusing, or a guide whose modes the pulse deforms. Nor is it the same
physics for a response that is not cubic. An ionisation rate reads the field amplitude,
and the mode-averaged field is the *mean* over the area. For a capillary's
$\mathrm{HE}_{11}$ the on-axis intensity is 1.768 times $P/A_{\rm eff}$, so a
mode-averaged plasma is the plasma of a lower intensity. The `peak_intensity` statistic
is always on the axis, from the mode profile, so it does not report the intensity the
rate saw.

**Entry points.** `prop_gnlse` and `Simulation.from_gnlse`, with $\beta(\omega)$ and
$\gamma$ or $n_2$ declared, or `fibre=` a mode or a table ([the GNLSE](../waveguides/gnlse.md),
[running a simulation](running.md)). A `Simulation` whose settings carry no quadrature
builds the one-node transform for any single mode, and `mode_average=True` selects it on
the gas-filled routes (`prop_capillary`, `prop_waveguide` and their `Simulation`
constructors). **`prop_capillary(modes=1)` is not mode-averaged** in solprop unless
asked: it resolves its one `HE_11` channel on the radial rule below, so its plasma sees
the field on each ring.

## Capillary modes on a radial rule

```{figure} ../figures/capillary_radial.svg
:width: 100%
:alt: A capillary cross-section with twelve radial Gauss nodes on one radius and the rings they stand for; beneath, HE11, HE12 and HE13 along x.

`HE_1m` modes have an azimuthally symmetric magnitude, so the rule is radial: each node
is a ring, and one azimuthal node carries the whole $2\pi$.
```

**Expansion.** $\mathbf E(r,\theta,t) = \hat x\sum_m a_m(t)\,J_0(u_{0m}r/a)/\sqrt{N_m}$,
the Marcatili $\mathrm{HE}_{1m}$ set at $\phi = 0$, which is x-polarised (the default
capillary polarisation; see [polarisation](polarisation.md)).
`prop_capillary(modes=N)` builds the contiguous set $\mathrm{HE}_{11}$ to
$\mathrm{HE}_{1N}$ ({class}`~solprop.modes.capillary.MarcatiliMode`).

**Boundary.** The field is truncated at the bore, $r = a$, where every $J_0(u_{0m}r/a)$
vanishes. The wall enters only through the complex $n_{\rm eff}$ of Marcatili and
Schmeltzer: its real part gives the waveguide dispersion and its imaginary part each
channel's own loss ([the Marcatili model](../waveguides/capillary-model.md#the-marcatili-model)). Numerically,
the Gauss nodes are strictly interior, so the wall is never sampled.

**Normalisation.** $N = \tfrac{\pi}{2}a^2J_1(u)^2\sqrt{\varepsilon_0/\mu_0}$ in closed
form, and $A_{\rm eff}$ likewise ([normalisation and effective
area](../waveguides/capillary-model.md#normalisation-and-effective-area)).

**Propagation.** A real synthesis matrix of the modes at the nodes and its weighted
transpose ({func}`~solprop.modes.quadrature.mode_matrix`), on a rule of $n_r$
Gauss–Legendre nodes in $r/a$ ({func}`~solprop.modes.quadrature.transverse_quadrature`).
`prop_capillary` asks for the Kronrod extension by default, whose embedded Gauss rule
gives the `quadrature_error` alarm at no extra cost. The quartic overlap tensor of four
`HE_1m` modes is converged to $2\times10^{-14}$ at $n_r = 32$
([the transverse quadrature](../waveguides/mode-sets.md#the-transverse-quadrature)). Each channel has its
own diagonal linear operator.

**Valid when** the launch and the medium keep the field in the `HE_1m` family: one
linear polarisation, a cylindrically symmetric launch, and an isotropic response.
$\mathrm{HE}_{11}$ can populate $\mathrm{HE}_{12}$ but not $\mathrm{HE}_{21}$
([polarisation](polarisation.md#angular-integration-and-plasma)). The Marcatili index
also needs a bore large compared with the wavelength: the ratio $u/(ka)$ is small and the
reflection at the wall is grazing.

**Not valid when** the launch is tilted or offset, or the polarisation is not linear and
uniform. Both need other azimuthal orders or two components: use the full rule below.
It is also wrong where the wall moves power between channels, which the diagonal loss
cannot do: use [the coherent wall](#the-capillary-wall-two-ways).

**Entry points.** `prop_capillary(radius, length, gas, pressure, modes=N)` and
`Simulation.from_capillary` ([running a simulation](running.md), [the capillary
chapter](../waveguides/capillaries.md)). A supplied set of `HE_1m` or tabulated modes on the
same bore goes through the same route.

## Capillary modes on the full cross-section

```{figure} ../figures/capillary_full.svg
:width: 100%
:alt: A capillary cross-section with a six by nine polar product rule and the TE01 transverse field drawn as arrows at every node; beneath, HE11 and TE01 along x.

A set with azimuthal structure needs the angle resolved. The rule is a product of
radial nodes and offset equispaced angles, and both Cartesian components are
synthesised at every node.
```

**Expansion.** $\mathbf E = \sum_m a_m\,\mathbf e_m(r,\theta)/\sqrt{N_m}$ over any
Marcatili modes — $\mathrm{HE}_{nm}$, $\mathrm{EH}_{nm}$ (a negative $n$), $\mathrm{TE}_{0m}$,
$\mathrm{TM}_{0m}$ — with both Cartesian components of each. A polarisation-resolved set
has two representations: separate mode rows (each hybrid mode followed by its partner,
`ModeCollection.with_polarisations`), or one scalar profile with a two-component state,
`n_pol = 2`. The product of the transform's and the state's polarisation counts is at
most `MAX_N_POL = 2` ([polarisation](polarisation.md#two-equivalent-representations),
[the kernel contract](../../dev/kernel-contract.md) §4).

**Boundary.** As on the radial rule: truncated at $r = a$, the wall in $n_{\rm eff}$.

**Propagation.** The same real matrix projection on a polar product rule of $n_r \times n_\theta$
nodes. The angles are the periodic trapezoid rule, offset by half a step, which is exact
for trigonometric polynomials of degree $n_\theta - 1$. A Kerr projection of modes of
azimuthal order up to $h$ is a quartic of degree $4h$, so the rule needs
$n_\theta \ge 4h + 1$. `transverse_quadrature` warns below it, and the default 16 covers
$h \le 3$. The offset keeps a product like $E_xE_y$ from vanishing at every node
([the transverse quadrature](../waveguides/mode-sets.md#the-transverse-quadrature)).

**Valid when** the Marcatili model is: a large bore and grazing incidence, each mode
attenuated on its own. The launch and polarisation can be arbitrary, provided the set
contains the modes they couple to.

**Not valid when** the set is truncated below the modes the physics populates: a
response or a launch that couples outside the set loses that energy silently, and only
the quadrature alarm and a larger set reveal it. A rule below $4h+1$ aliases the Kerr and
Raman terms.

**Entry points.** `prop_waveguide(modes, length, gas, pressure, ...)`,
`Simulation.from_waveguide` and `prop_capillary(modes=[...])`, which infer the
components the set needs: $E_x$ and $E_y$ for a $\mathrm{TE}_{0m}$, $\mathrm{TM}_{0m}$,
EH, higher-order HE or rotated set, one state component throughout (`polarisation=` is
refused by name). `prop_capillary`'s own $\mathrm{HE}_{1m}$ set resolves
$E_x$. A supplied set resolves what it needs beside a coherent wall too: the wall's
flow acts on the channels of the one-component state and never reads the synthesis.
A direct `Simulation` takes `components=` and a
two-component state.

## The capillary wall, two ways

```{figure} ../figures/capillary_wall.svg
:width: 100%
:alt: Two capillary cross-sections with the same radial rule; left labelled E = 0 at r = a with per-channel loss, right labelled with the condensed Robin condition; beneath each, the HE1m profiles, and on the right their tangents at the wall.

The same basis and rule, two wall models. Left, the Marcatili wall: each channel has its
own loss and nothing moves between them. Right, the coherent wall: the outgoing boundary
condition, condensed onto the basis through each mode's slope at the wall, couples every
pair.
```

**Truncated.** The field of each channel vanishes at $r = a$ and the wall is its
$\operatorname{Im}n_{\rm eff}$. The loss is **incoherent**: channels attenuate
independently, and power never moves between them through the wall.

**Coherent.** `CoherentWall()` condenses the outgoing Robin condition
$\partial_rE = ik_{\rm eff}E$ of the exact radial problem onto the retained basis
$\mathrm{HE}_{11}\ldots\mathrm{HE}_{1N}$. The result is the rank-one generator
$G = D + g(a)\,hh^{\rm T}$, where $h_m$ is the $m$-th mode's slope at the wall. It is
carried as the `linear.coupled_flow` leaf ({class}`~solprop.modes.capillary.CoherentWall`,
[the coherent wall](../waveguides/coherent-wall.md#the-coherent-wall)). `impedance="te_tm"` condenses the two
conditions the glass really imposes, one for the tangential and one for the normal field.
It couples the two radial families of one azimuthal order
([the vector boundary response](../waveguides/vector-wall.md#the-vector-boundary-response)).

**When each is needed.** One channel, or channels far apart in propagation constant
compared with the wall's coupling: the truncated wall. The coherent wall reproduces its
single-channel loss to $-(4N^2t^2 + u_m^2/2(ka)^2)$ relative, $1.2\times10^{-5}$ for a
125 µm bore at 800 nm. Several `HE_1m` channels that carry energy, an aperture that fills
them together, and above all a **taper**: the coherent wall. Only it carries the
taper's connection between the channels ([below](#guides-that-move-along-z)).

**Valid when** the condensation is: it is quasi-static and paraxial. Three ratios —
$u_N/(ka)$, $u_N/(k_0a)$ and $2N/|ak_{\rm eff}|$ — are computed over the active band and
recorded. The run warns within a factor ten of
{attr}`~solprop.modes.capillary.CoherentWall.validity_limit` and refuses above it.

**Not valid when** the set is not contiguous from `HE_11`, spans two azimuthal orders or
two polarisations, sits beside a tube lattice or a noise model, or declares a loss other
than the model's own. Each is refused by name. The `te_tm` wall is checked against the
exact vector Maxwell roots family by family, but it has not been checked against a
resolved-cladding solve of a mixed state.

**Entry point.** `prop_capillary(..., wall=CoherentWall())`. The scalar and TE/TM blocks
run on all three backends, straight, tapered or in a gradient; `method="cf4"` runs on
`numpy_ref` alone ([the capillary chapter](../waveguides/capillaries.md)). The block is
scalar in the **state**, one component per channel, but not in the synthesis: a
$\mathrm{TE}_{0m}$, $\mathrm{TM}_{0m}$ or EH set beside the `te_tm` wall resolves
$E_x$ and $E_y$, so its Kerr and plasma terms see the whole field.

## Antiresonant tube lattice

```{figure} ../figures/antiresonant.svg
:width: 100%
:alt: A single-ring antiresonant fibre to scale, six tubes around a 15 micrometre core, with HE11 contours in the core; beneath, the HE11 profile, zero outside the core.

The tubes shape the index and the loss. The field the run propagates is Marcatili's,
truncated at the core radius.
```

**Expansion.** The capillary's: Marcatili fields of the core, on the capillary's rule,
with {class}`~solprop.modes.antiresonant.VincettiMode` supplying the index.

**Boundary.** The field is truncated at the core radius $a$. The tubes — outer radius
$r$, wall $t$, $N$ of them, clear gap $\delta = 2[(a+r)\sin(\pi/N) - r]$ — enter only
through the fitted $n_{\rm eff}(\omega)$ and loss of Vincetti (2016), Vincetti and Rosa
(2019) and Rosa, Melli and Vincetti (2021), in the form set out in
[the five conventions](../waveguides/antiresonant.md#the-five-conventions).

**Propagation.** Exactly the capillary's. The field normalisation and the protocol's
`aeff` are Marcatili's. The fitted area `effective_area(omega)` is a
diagnostic, not the propagation area.

**Valid when** the fits are: they are empirical, calibrated over $0.05 \le t/r \le 0.2$,
$6 \le N \le 10$ and the first transmission bands. Construction logs once outside those
ranges, and once for a mode other than $\mathrm{HE}_{11}$, whose corrections are
fundamental-mode fits.

**Not valid when** the dispersion near a resonance matters: against a converged
finite-element solution the group-velocity dispersion is where the fit is weakest, and at
a band edge the fitted loss can be 150 times too low
([accuracy against a finite-element solution](../waveguides/antiresonant.md#accuracy-against-a-finite-element-solution)).
Nor does it model coupling to cladding modes, or light in the tubes.

**Entry point.** `prop_capillary(..., lattice=TubeLattice(...))`, or a supplied
`VincettiMode` set ([the Vincetti model](../waveguides/antiresonant.md#antiresonant-tube-lattice-fibres-the-vincetti-model)).

## Step-index fibres, exact

```{figure} ../figures/step_index.svg
:width: 100%
:alt: A silica strand in air with its core and exterior; beneath, E_x along x, which jumps at the core boundary, and along y, which is continuous.

The exact mode is not truncated. Along x the field is radial and jumps by the
permittivity ratio at the boundary; along y it is tangential and continuous.
```

**Expansion.** One exact vector mode of a circular step-index core,
{class}`~solprop.modes.stepindex.StepIndexMode`: $J_n(ur/a)$ inside, $K_n(wr/a)$
outside, with the Snyder–Love hybrid, TE and TM fields in Cartesian components. The
profile is frozen at a declared `profile_wavelength`.

**Boundary.** $\mathbf E_t$ and $\mathbf H$ are continuous at $r = a$, and so is the radial
displacement, so the radial $E$ jumps. The exterior is the exact decaying $K_n$, carried
to a declared outer radius $R_{\rm out} = \max(2a,\ a[1 + \ln(10^8)/w])$, by which the
exterior field has fallen by a factor $10^8$. Normalisation and area integrate adaptively, split at the core
([step-index fibres](../waveguides/step-index.md#step-index-fibres)).

**Propagation.** Two ways. Mode-averaged, through `prop_gnlse(None, length, fibre=mode,
n2=...)` or `Simulation.from_gnlse`: the fibre supplies $\beta(\omega)$, its loss and
`aeff()`. Vector, through the general route below:
`ModalGuide.from_step_index(...).field_provider` is a
{class}`~solprop.modes.guide.StepIndexProvider`, and
`ModalGuide.from_provider(provider, knots, nodes=...)` solves it over the band.
`prop_waveguide` refuses a bare `StepIndexMode` and names the entry to use.

**Valid when** one bound, lossless mode carries the pulse and its profile is the one at
the profile wavelength. Weak guidance is not assumed: the eigenproblem is the full
vector one, and a material loss is a declared attenuation, not a complex index.
Weak guidance is the limit in which the exact mode becomes the scalar one: on a 4 µm core
at a 0.01 index step the vector and scalar nonlinear coefficients agree to
$5.1\times10^{-4}$, and on the 0.25 µm strand of the next figure they differ by 25 %.
The mode-averaged route uses the scalar area either way.

**Not valid when** several modes are wanted on the scalar routes: a set of exact vector
modes is not orthogonal under the electric-field inner product those routes use.
Use the general route, which pairs them through the flux metric. A taper is
not scale invariant: the eigenpair is re-solved along z.

## Graded-index providers

```{figure} ../figures/graded_index.svg
:width: 100%
:alt: A graded-index core and its exterior with the provider's node rule, split at every finite element; beneath, the TE01 and TE02 fields along x.

The provider's rule splits at every finite element and at the matching radius. Beyond
it the field is an exact Bessel continuation, not nodes.
```

**Expansion.** The channels of one radial sector of
$\varepsilon(r) = \varepsilon_{\rm axis}[1 - 2\Delta(r/a)^p]$, solved by
{class}`~solprop.modes.graded_index.GradedIndexProvider`: continuous linear finite
elements over `elements` core elements and the declared `extent`, matched to a decaying
Bessel exterior. `family="TE"` or `"TM"` is the vector TE₀ or TM₀ sector; `family="LP"`
is labelled a weak-guidance scalar approximation.

**Boundary.** The frequency-dependent boundary eigenproblem at the matching radius
`extent × a`, with the exact decaying exterior beyond it. The power carried beyond is
integrated analytically and certified; a channel with more than
`MAX_EXTERIOR_TAIL_FRACTION` = 1e-4 of its power there is refused by name.

**Propagation.** The general route, below: `ModalGuide.from_provider(provider, knots)`
and `prop_waveguide((), ..., provider=guide)`
([radial graded-index providers](../waveguides/graded-index.md#radial-graded-index-providers)).

**Valid when** the sector is the physics: the TE₀ and TM₀ sectors are vector-exact within
the finite elements, second order in the element count for the eigenvalues and for
integrals such as the Kerr coefficient. LP needs a permittivity contrast of at most 0.02.

**Not valid when** pointwise $E_z$ near the interface matters: a P1 field's derivative
converges at first order, about 1 % at 96 elements. Hybrid HE and EH sectors of a graded
core are refused by name. The LP sector carries no longitudinal field.

## The general modal route: vector fields

```{figure} ../figures/general_modal.svg
:width: 100%
:alt: A 250 nanometre silica strand with its provider's node rule and the transverse field of HE11 as arrows; beneath, the magnitudes of E_x and E_z along x.

On a strongly guiding strand the longitudinal field is comparable with the transverse
one. The general route carries all three components at every node.
```

**Expansion.** $\mathbf E(\mathbf r,t) = \sum_m a_m(t)\,\mathbf e_m(\mathbf r,\omega)$, with
the provider's three-component $\mathbf E$ and $\mathbf H$ per √W at every frequency, on
its own node rule ({mod}`solprop.modes.vector`, {class}`~solprop.modes.guide.ModalGuide`).

**Boundary.** Whatever the provider solves: for a step-index provider, $\mathbf E$ and
$\mathbf H$ matched at the core with an exact $K_n(\gamma r)$ exterior. The node rule
reaches a declared outer radius, and the power it discards there is certified, at the
lowest frequency of the band.

**Normalisation.** Each channel carries unit Poynting flux over the whole domain, so
$|a|^2$ is watts and the power weight is one. The basis is gated: its sampled and
interpolated Gram must be within `gram_atol` (1e-4) of the identity where the launched
light is, and it warns where light could go
([normalisation](../waveguides/modal-projection.md#normalisation-and-why-the-power-weight-is-one)).

**Propagation.** Synthesis $S$ is complex and frequency dependent, so it acts in the
frequency domain. The response is evaluated per node and region with the full
three-component contraction. Projection $Q = W^{-1}\bar e\,w/4$ uses the Hermitian flux
metric $W$ ([the synthesis and projection maps](../waveguides/modal-projection.md#the-synthesis-and-projection-maps)).
A declared unitary change of basis — a circular pair, say — is carried with its metric.
A split block becomes a coupled flow.

**Valid when** the route is the scalar one's superset: the vector $\gamma$ of a 250 nm
silica strand is 25 % above the scalar $n_2\omega/(cA_{\rm eff})$, and on a weakly
guiding 4 µm core at a 0.01 index step the two agree to $5.1\times10^{-4}$. That
agreement is where the scalar routes on this page are the same physics.

**Not valid when** the route refuses it; the refusals are not approximations:
`peak_intensity` and the scalar-route power names (use `analytic_flux`), `quadrature_error`
(no embedded rule), a nonorthogonal basis, a guide that does not span the active band,
`n_pol` > 1, and numeric `observable_rtol` ([declared, refused,
recorded](../waveguides/general-modal.md#declared-refused-recorded)).

**Entry point.** `prop_waveguide((), length, provider=guide, ...)` and
`Simulation.from_waveguide(..., provider=guide)`, on a field grid
([the solid-fibre chapter](../waveguides/solid-fibres.md)).

## Guides that move along z

```{figure} ../figures/moving_guide.svg
:width: 100%
:alt: A capillary tapering from 50 to 40 micrometres in the z-x plane, with the radial node lines following the bore, the local HE11 profile at each end, and the kinks marked.

In a scale-invariant taper the nodes ride the bore and the basis moves with it. What
moves between the channels is the connection, and only some routes carry it.
```

A taper or a gradient is declared by giving a radius, a pressure or a provider's region
radii as a {class}`~solprop.profiles.ZProfile`; the call is otherwise the same as for a
straight guide ([profiles along the fibre](../waveguides/profiles.md)). Three different things can move.

**Scale-invariant modes** — capillary, Vincetti, and a rectangle whose two widths follow
one profile — depend on $r$ only through $r/a(z)$. The matrices rescale exactly,
$S(z) = S(0)\,a(0)/a(z)$ and $W(z) = W(0)\,a(z)/a(0)$, by one scalar
({func}`~solprop.modes.quadrature.taper_scale`); their product does not move. A pressure
gradient in a fixed bore moves only the index.

**The connection.** A moving basis costs a term,
$\langle\psi_m|\partial_z\psi_n\rangle$. For the capillary's $\mathrm{HE}_{1m}$ it is
$C_{mn} = (a'/a)\,2s_ms_nu_mu_n/(u_m^2 - u_n^2)$, zero where the bore is straight
([a guide that varies along z](../waveguides/coherent-wall.md#a-guide-that-varies-along-z)). **A multimode
capillary taper carries it by default**, with or without the coherent wall (the *taper
coupling*). On a 50 to 40 µm transition over a fifth of a millimetre it
converts 7 % of the energy out of $\mathrm{HE}_{11}$. `taper_coupling=False` is the
**adiabatic** opt-out, on which each channel follows its local mode; a moving set the
connection is not derived for is adiabatic too and warns once,
`AdiabaticTaperWarning`.

**A moving solid guide.** A solid guide's modes change *shape* along a taper, because
its normalised frequency moves. `ModalGuide.from_provider` samples it along its path, in
Chebyshev sections cut at every kink. The route carries the moving maps $S(z)$, $Q(z)$
and the solid connection $\Gamma = W^{-1}S$ in a frozen-residual coupled flow
([a guide whose geometry moves along z](../waveguides/moving-modal.md#a-guide-whose-geometry-moves-along-z)).
Every backend executes it, the native backend on a GPU included.

**Valid when** the step resolves the connection: on the coherent wall a declared step
above the advisory limit is warned about by name, and a fixed schedule must land on
every kink. A retained set describes a taper to second order in its total contraction.

**Not valid when** the conversion matters and the route omits it — a multimode taper run
adiabatically, by `taper_coupling=False` or on a set the connection is not derived for —
or when light leaves the retained set, which no basis carries.

## Rectangular hollow guides

```{figure} ../figures/rectangular.svg
:width: 100%
:alt: A rectangular hollow guide to scale, 100 by 50 micrometres, with a twelve by twelve Cartesian Gauss rule; the x-polarised fundamental along x beneath and along y beside it.

Two wall pairs, and the polarisation decides which pair sees the TE-like and which the
TM-like reflection.
```

**Expansion.** {class}`~solprop.modes.rectangular.RectMode` rows,
$\cos$ or $\sin(m\pi x/2a)\,\cos$ or $\sin(n\pi y/2b)$ inside the rectangle, each with one
Cartesian component. The x- and y-polarised partners are *not* degenerate, so they are
separate rows with their own operators
([rectangular polarisation partners](polarisation.md#rectangular-polarisation-partners)).

**Boundary.** Truncated at the walls. Each wall pair adds its complex transverse
wavenumber to $\beta^2 = k^2\varepsilon_{\rm co} - \kappa_x^2 - \kappa_y^2$ (Laakmann and
Steier). For x polarisation the $x = \pm a$ walls see the TM-like coefficient
$\varepsilon_{\rm cl}/(\varepsilon_{\rm co}s)$ and the $y = \pm b$ walls the TE-like $1/s$,
$s = \sqrt{\varepsilon_{\rm cl} - \varepsilon_{\rm co}}$, with the filled-core contrast.
Complex wall corrections are kept in the phase as well as the loss.

**Normalisation.** $N = \varepsilon_0cab/2$ and $A_{\rm eff} = 16ab/9$.

**Propagation.** A Cartesian Gauss–Legendre product rule, `n_r` and `n_theta` being the
x and y counts. Trigonometric products have no finite exactness degree, so the bound is
measured: {func}`~solprop.modes.quadrature.cartesian_nodes`, $6h + 6$ per axis (the
measured counts for a relative 1e-12 are 12, 18, 22, 26 and 35 nodes per axis at highest
orders 1, 2, 3, 4 and 6), which both defaults of `prop_waveguide` take. The rule has no Kronrod
extension, so no `quadrature_error` is recorded and the alarm is off.

**Valid when** the wall-reflection model is: a guide wide compared with the wavelength, at
grazing incidence, in both directions.

**Not valid when** either half-width approaches the wavelength, or the two partners are
meant to exchange power. The shared-profile layout is refused for `RectMode`, because it
would give both partners one index. A modal transform refuses a taper whose two widths
move independently.

**Entry point.** `prop_waveguide([modes], length, gas, pressure, ...)` and
`Simulation.from_waveguide` ([hollow rectangular waveguides](../waveguides/rectangular.md#hollow-rectangular-waveguides)).

## Radial free space

```{figure} ../figures/radial_free.svg
:width: 100%
:alt: A radial grid of 24 nodes at the zeros of J0 with its absorbing collar; beneath, a Gaussian beam at the nodes and the collar window.

The order-zero Hankel transform samples at the zeros of $J_0$, so the field vanishes at
$R$. With an absorber declared, the collar absorbs before the beam reaches it.
```

**Expansion.** $E_p(r,t) = \int_0^\infty k\,H_p(k,t)\,J_0(kr)\,dk$, one order-zero
transform per Cartesian component $p$. The state's columns are the transverse
frequencies $k_i = \alpha_i/R$ ({class}`~solprop.grid.RadialGrid`).

**Boundary.** $E = 0$ at $R$: the nodes $r_i = R\alpha_i/\alpha_{N+1}$ sit at the zeros of
$J_0$. When the run declares an absorber (free-space entry points carry none by default, and
without one the field reflects at $R$), a Planck collar over the outer `window_factor` (a
tenth by default) absorbs as a rate inside every stage, not as a mask ([spatial collar](../free-space/model.md#spatial-collar-and-statistics)).

**Units.** The stored spectrum is in V m and the field in V/m. The weights carry the whole
$2\pi$; the energy is in J, the axial Poynting flux of the medium at the plane
([energy and flux in a medium](../free-space/flux.md)).

**Propagation.** The raw quasi-discrete Hankel transform, a dense matrix transform, with the exact
$k_z = \sqrt{k^2 - k_r^2}$ ([the Hankel transform](../free-space/hankel.md), [exact longitudinal
operator](../free-space/model.md#exact-longitudinal-operator)). The medium may vary along z: a gas
pressure profile is allowed, with the forward amplitude term that conserves each column's
energy, when it meets the
[adiabatic certificate](../free-space/flux.md#the-adiabatic-certificate).

**Valid when** each Cartesian component is cylindrically symmetric about the axis: a
round beam on axis, with a uniform polarisation.

**Not valid when** anything breaks that symmetry: an astigmatic or tilted beam, a vortex,
multiple filaments, azimuthal modulational instability. The finite transform is not
exactly orthogonal; its round-trip error is bounded, not zero — $3.06\times10^{-11}$ at
$N = 256$ ([finite-transform error](../free-space/hankel.md#finite-transform-error)). The critical
power ({func}`~solprop.spatial.critical_power`) is the scalar paraxial Townes value,
and does not predict where an exact-UPPE pulse collapses.

**Entry point.** `prop_radial` and `Simulation.from_radial`
([construct and run a beam](../free-space/model.md#construct-and-run-a-beam)).

## Cartesian free space

```{figure} ../figures/cartesian_free.svg
:width: 100%
:alt: A periodic Cartesian box of 16 by 12 samples with its absorbing collar, a Gaussian beam in the middle and arrows showing that the box is periodic; beneath, the beam at the nodes and the collar window along the middle row.

The box is periodic; its extra tenth on each side is the absorbing collar, active when the run declares an absorber. Nothing is
assumed about symmetry.
```

**Expansion.** $E_p(x,y,t)$ on a periodic $n_x \times n_y$ grid, stored as its raw
forward DFT; column `ix + n_x*iy`, x fastest ({class}`~solprop.grid.FreeGrid`).

**Boundary.** Periodic, with an absorbing collar when the run declares an absorber (none by
default, so light leaving one side re-enters on the other): the periodic half-width is
`radius * (1 + window_factor)` on each axis, and the collar is the product of two Planck
windows over the added width ([Cartesian free space](../free-space/cartesian.md)).

**Units.** Stored and physical fields in V/m; spatial weights $\Delta x\Delta y$ and
spectral weights $\Delta x\Delta y/N$; energy in J, the axial flux of the medium.

**Propagation.** The 2-D FFT with the separable exact operator: frequency-dependent $k^2$
beside one $k_\perp^2$ vector, with the grazing-incidence arithmetic retained in two
parts ([independent numerical budgets](../free-space/cartesian.md#independent-numerical-budgets)).

**Valid when** the medium is constant: no symmetry is assumed, so any azimuthal
structure, any polarisation pattern and any number of filaments are represented.

**Not valid when** the medium varies along z: varying profiles and the `tabulate`
propagator are refused by name. The cost is the driver: $n_xn_y$ columns per time
sample, and $n_xn_y\log(n_xn_y)$ per transform
([Cartesian example and memory](../free-space/model.md#cartesian-example-and-memory)).

**Entry point.** `prop_free` and `Simulation.from_free`.

## Planar hollow guides, uniform in y

```{figure} ../figures/slab.svg
:width: 100%
:alt: Two silica plates 127 micrometres apart, to scale in x, with the gas-filled gap between them extending unbounded in y, the slab rule's 24 Gauss nodes along the mid-line, and the TE and TM polarisation arrows; beneath, the TE1 and TE3 profiles across the gap.

One wall pair and a free direction: the guide is uniform in y, so everything it integrates
is per metre of y.
```

**Expansion.** {class}`~solprop.modes.slab.SlabMode` rows, $\cos$ or $\sin(p\pi x/2a)$
inside the gap $|x| \le a$, uniform in y, each with one Cartesian component: TE along y
(parallel to the plates, the low-loss family every compressor uses) and TM along x. The
two families are *not* degenerate, so TE and TM of one order are separate rows.

**Boundary.** Truncated at the plates. The one wall pair adds its complex transverse
wavenumber $\kappa_p = (p\pi/2a)(1 - iq/ka)$ to $\beta^2 = k^2\varepsilon_{\rm co} -
\kappa_p^2$: the rectangular guide's model with $b \to \infty$ (Laakmann and Steier,
Miyagi et al., Arnold et al.), TE seeing the TE-like $q = 1/s$ and TM the TM-like
$\varepsilon_{\rm cl}/(\varepsilon_{\rm co}s)$, with the filled-core contrast and the
same code as `RectMode`. The loss is
$(1/a)(p\lambda/4a)^2\operatorname{Re}q$: $p^2$ and $a^{-3}$, and TE1 loses about a
quarter of what a capillary of radius $a$ does.

**Normalisation.** $N = \varepsilon_0ca/2$ and $A_{\rm eff} = 4a/3$, both **per metre of
y**: the run's powers are W/m and energies J/m, intensities still W/m². The mode declares
`transverse_measure = "per_unit_width"`, the run records it, and processing and the plots
label from it; a slab row cannot share a set with an area mode.

**Propagation.** A Gauss rule in x on $[-a, a]$ emitting $(x, 0)$ points with a unit y
weight, so the pipeline downstream is the Cartesian one with one y node. The node bound is
the measured $6h + 6$ ({func}`~solprop.modes.quadrature.cartesian_nodes`, re-measured on
the one-dimensional products, which reach 1e-12 at 12, 17, 22 and 26 nodes for
$h = 1\ldots4$); unlike the Cartesian product rule the
slab rule takes the Kronrod extension, so `quadrature_error` is recorded and the alarm
runs. A varying gap is scale invariant: the matrices rescale by $\sqrt{a(0)/a(z)}$, and
each channel follows its local mode adiabatically (the connection is omitted, and says
so). A Gaussian launch projects onto the modes through
{func}`~solprop.modes.slab.gaussian_coupling`: 98.8 % into TE1 at $w_x = 0.735a$.

**Valid when** the wall-reflection model is, $p\lambda/4a \ll 1$ past Brewster, and the
beam is wide enough in y that its structure there evolves slowly: the centre of a planar
compressor's line focus. {func}`~solprop.modes.slab.gaussian_to_per_unit_width` converts a
Gaussian beam's energy in y to the J/m at its centre.

**Not valid when** anything happens in y. Whole-beam self-focusing, filamentation and the
modulational instability of a wide beam in y set a real planar compressor's energy limit
(Arnold et al. 2009, $B \lesssim 15.5$), and a field uniform in y cannot show any of them,
nor has a critical power. Resolving y is the [hybrid decomposition](#planar-guides-resolved-in-y).

**Entry point.** `prop_waveguide([modes], length, gas, pressure, ...)`,
`Simulation.from_waveguide` and `ModeCollection.slab`, mode-averaged or multimode; a single
slab through `prop_gnlse(fibre=...)` stays per unit width
([hollow planar waveguides](../waveguides/planar.md#hollow-planar-waveguides)).

## Line free space: x only

```{figure} ../figures/line_free.svg
:width: 100%
:alt: A periodic line of 16 samples in x with its absorbing collar at both ends, the beam region extending unbounded in y, and the E_x polarisation arrow; beneath, a Gaussian beam at the samples and the collar window.

The Cartesian grid's x axis with one y sample of unit width: the beam is uniform in y, and
everything integrated over the cross-section is per metre of y.
```

**Expansion.** $E_p(x,t)$ on a periodic grid of $n_x$ samples, stored as its raw forward
DFT, uniform in y ({class}`~solprop.grid.LineGrid`). It is
the [Cartesian grid](#cartesian-free-space) with $n_y = 1$: one y column carrying
$k_y = 0$, no y collar and a unit y measure.

**Boundary.** Periodic in x with an absorbing collar over the added `window_factor` of
the half-width when the run declares an absorber, exactly the Cartesian x axis; nothing bounds or varies in y.

**Units.** Fields in V/m; spatial weights $\Delta x$ and spectral weights $\Delta x/n_x$,
in m, so the energy is in **J/m** and the power in **W/m**, per metre of y; intensity
and fluence are local and stay W/m² and J/m². A {class}`~solprop.pulses.SpatialPulse`
on a line takes its energy in J/m, the run records `transverse_measure =
"per_unit_width"`, and processing and the plots label from it, as for the slab.

**Propagation.** The Cartesian contract unchanged — the same separable exact operator
with $k_\perp^2 = k_x^2$ and the same two-axis FFT with a unit y extent — on all three
backends ([the line](../free-space/model.md#the-line-uniform-in-y)).

**Valid when** the medium is constant and the beam is uniform in y over the region of
interest: a line focus or a wide sheet beam, the free-space counterpart of the slab.

**Not valid when** anything happens in y, and in particular not as a model of a round
beam: the (1+1)-D physics differs. A Gaussian's axial amplitude falls as
$(1+(z/z_R)^2)^{-1/4}$ and its Gouy phase is $\tfrac12\arctan(z/z_R)$, and Kerr
self-focusing does not collapse: its stationary beam is the stable soliton
$\operatorname{sech}(x/x_0)$ at a power per metre of y that depends on $x_0$ (that power
is derived in [the line](../free-space/model.md#the-line-uniform-in-y)). There is no critical
power, and
{func}`~solprop.spatial.critical_power` refuses a line by name. A sheet beam's
modulational instability in y, which breaks up a real one, is outside the model. Varying
media are refused as on the plane. The cost is $n_x$ columns per time sample.

**Entry point.** `prop_free` and `Simulation.from_free` with a `LineGrid`.

## Planar guides resolved in y

```{figure} ../figures/planar_hybrid.svg
:width: 100%
:alt: The two silica plates of the uniform slab, to scale in x, with the slab rule's nodes repeated on twelve periodic y rows, the outer rows in an absorbing collar, and the TE polarisation arrow; beneath, the TE1 and TE3 profiles across the gap.

Slab modes in x on every y sample: the plates bound x, and y is periodic with an absorbing
collar, as the Cartesian grid's axes are.
```

**Expansion.** $E(x, y, t) = \sum_p V_p(x)\,A_p(y, t)$: the
{class}`~solprop.modes.slab.SlabMode` channels of one gap in x, and in y a periodic grid of
$n_y$ samples, each column $(p, k_y)$ the raw y-DFT of a slab amplitude $A_p(y)$
({class}`~solprop.modes.slab.PlanarGrid`). This is the expansion of every planar
compression model in the literature (Arnold et al. 2010, eq. (9)) and the physically complete planar guide: what the uniform slab above leaves out,
it resolves.

**Boundary.** In x, the uniform slab's: the field truncated at the plates and each
channel's wall pair in its complex $\kappa_p(\omega)^2$. In y, the Cartesian grid's: periodic
with the Planck collar over the added `window_factor` of the half-width, applied as an
in-stage rate when the run declares an absorber, so that a beam that diffracts or breaks
up in y does not wrap around the periodic grid.

**Normalisation and units.** Modal amplitudes per y sample in √(W/m), exactly the uniform
slab's; the geometry integrates over y, so energy is in **J** and power in **W**, and
the run's measure is `"area"`. A {class}`~solprop.pulses.PlanarPulse` is the astigmatic
launch: $w_x$ projected onto the modes, with signs, and $w_y$ sampled on the y axis; its
energy is the incident beam's physical energy in the fill, so the modes carry the
coupled share (98.8 % into TE1 at $w_x = 0.735a$). The energy of each mode uses the
explicit slab metric, whose weight at $k_y = 0$ is the mode's real effective index, not the
bulk index ([the hybrid planar guide](../free-space/flux.md#the-hybrid-planar-guide)).
{func}`~solprop.spatial.planar_mode_energy` and
{func}`~solprop.spatial.planar_fluence` split a saved plane by mode and map it on the
$(x, y)$ nodes, and {func}`~solprop.spatial.planar_intensity` gives the signed axial
flux there in time. The peak of the physical field, {func}`~solprop.spatial.peak_field`,
binds an unbound optmat rate's energy per event, as on the other spatial
routes ([ionisation](../media/ionisation.md)); the run's `beam_radius` and `fwhm_r` are the **widths in y** of the fluence
integrated across the gap.

**Propagation.** Each column propagates with
$k_z = \sqrt{k^2\varepsilon_{\rm co} - \kappa_p(\omega)^2 - k_y^2}$, exact (non-paraxial)
in y, with the free-space evanescent model and prefactor $-i\omega^2/(2\varepsilon_0c^2
\operatorname{Re}k_z)$ per column. The nonlinear pass synthesises the field at the slab
rule's nodes on every y sample, evaluates the responses node by node and projects back.
A field uniform in y reproduces the uniform slab to the operator's rounding (2e-10 over ten
steps); a linear Gaussian beam diffracts as the (1+1)-D Gaussian and decays as each mode
([resolving y](../waveguides/planar.md#resolving-y)).

**Valid when** the uniform slab's wall model is, and the medium and the gap are constant
along z. It shows what happens in y: whole-beam self-focusing and the modulational
instability of a wide beam, which set a planar compressor's energy limit (Arnold et al.
2009, $B \lesssim 15.5$).

**Not valid when** the gap varies (refused by name: the frozen operator cannot carry a
moving slab mode, and a moving one is not implemented), when the fill varies along z or
is a mixture (refused, whatever the material selection; a single gas named on the slab
is optmat's, its law and equation of state,
[materials](../media/material-selection.md#selecting-a-runs-materials)),
or when both TE and TM are wanted: the state is scalar, one polarisation family. The cost
is $n_{\rm modes}\,n_y$ columns and $n_x n_y$ nodes per time sample.

**Entry point.** `prop_planar(grid, PlanarGrid(modes, radius_y, n_y), responses, launch,
settings=...)` and `Simulation.from_planar`; `numpy_ref`, `numba_cpu` and native CPU
execute it, the compiled two within derived bounds of `numpy_ref`.

## Radial and azimuthal free space: considered, not built

```{figure} ../figures/radial_azimuthal.svg
:width: 100%
:alt: The radial nodes of the QDHT of orders zero, one and two on three lines; no two orders share a node.

The QDHT of order $m$ samples at the zeros of $J_m$. No two orders share radial nodes,
and the nonlinear response needs one common grid.
```

An expansion $E = \sum_m E_m(r)\,e^{im\theta}$ in azimuthal orders looks like the natural
step between the radial and the Cartesian grids. It was analysed and **not built**; this
is why.

**The full expansion** needs one Hankel transform per order. The QDHT of order $m$
samples at the zeros of $J_m$, so no two orders share radial nodes. The nonlinear
response is evaluated pointwise, on one common $(r, \theta)$ grid, so every stage would
need an interpolation between the orders' grids or a non-orthogonal transform. Anything
that breaks cylindrical symmetry enough to need many orders — filamentation,
multiple-filament break-up, the azimuthal modulational instability of a vortex ring — is
done better and more cheaply on the Cartesian grid.

**A single order $m$** would be cheap: the order-zero matrix transform with $J_m$ in place of $J_0$.
It is exact for **cylindrical-vector beams**, radially or azimuthally polarised
($E_r$ or $E_\phi$ at order one). These stay closed under any isotropic local response,
third harmonic and plasma included, because the polarisation is parallel to the field. For a
**scalar vortex** it is exact only as an envelope without third-harmonic generation:
a real field's third harmonic carries order $3m$. And it cannot show a vortex's
azimuthal instability, which dominates above the critical power.

**Use instead:** the radial grid for a round beam of uniform polarisation; the Cartesian
grid for anything with azimuthal structure. A capillary's azimuthal *mode sets*
([above](#capillary-modes-on-the-full-cross-section)) are a different thing: modes of a
guide, not transforms of free space. The single-order transform is recorded as a
candidate on the [roadmap](../../dev/roadmap.md).

## Choosing a decomposition

Start from the symmetry you can defend, then take the cheapest decomposition that
assumes no more.

1. **One guided mode whose shape the pulse does not change**, and only Kerr and Raman
   responses: [mode-averaged](#mode-averaged-one-field-and-one-area). Cost: one node,
   the $n_\omega$ transforms.
2. **A gas-filled capillary, linear polarisation, round launch**: the
   [radial rule](#capillary-modes-on-a-radial-rule) with `HE_1m` channels, also for one
   mode when a plasma is on. Cost: $n_{\rm modes}\times n_r$ per transform.
3. **The same, with channels the wall couples, or a taper**: add the
   [coherent wall](#the-capillary-wall-two-ways). Cost: a coupled flow of
   $n_{\rm modes}^2$ per step.
4. **Azimuthal structure or two polarisations in a capillary**: the
   [full rule](#capillary-modes-on-the-full-cross-section). Cost:
   $n_{\rm modes}\times n_r n_\theta$, with $n_\theta \ge 4h + 1$.
5. **A single-ring antiresonant fibre**: [Vincetti](#antiresonant-tube-lattice), at the
   capillary's cost.
6. **A rectangular hollow guide**: [Cartesian rule](#rectangular-hollow-guides), cost
   $n_{\rm modes}\times(6h+6)^2$. **A planar guide whose beam is wide in y**, per metre
   of y: the [slab rule](#planar-hollow-guides-uniform-in-y), cost
   $n_{\rm modes}\times(6h+6)$; nothing that happens in y is represented.
7. **A solid fibre, single mode, $\gamma$ and $\beta(\omega)$ enough**:
   [step-index](#step-index-fibres-exact) through `prop_gnlse(fibre=)`.
8. **A solid fibre whose vector fields matter** — a small core, several channels, a
   graded index, a declared circular basis, a taper: the
   [general route](#the-general-modal-route-vector-fields). Cost:
   $n_{\rm modes}\times 3\,n_{\rm nodes}$, complex and frequency dependent.
9. **A free beam, round, uniform polarisation**, possibly in a pressure gradient:
   [radial free space](#radial-free-space). Cost: $n_r^2$ per transform.
10. **A free beam with any other structure**, in a constant medium:
    [Cartesian free space](#cartesian-free-space). Cost: $n_xn_y\log(n_xn_y)$ per
    transform.
11. **A free beam uniform in y** — a line focus, a sheet beam — in a constant medium:
    [line free space](#line-free-space-x-only), per metre of y. Cost: $n_x\log n_x$
    per transform; nothing that happens in y is represented.
12. **A planar guide whose beam does something in y** — self-focusing, break-up, a
    finite height: the [hybrid planar grid](#planar-guides-resolved-in-y), in J. Cost:
    $n_{\rm modes}\times(6h+6)$ per y sample plus $n_y\log n_y$ per mode.
