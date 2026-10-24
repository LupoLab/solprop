# General modal propagation

This page is for anyone propagating in a solid or graded-index guide whose modes cannot be
treated as scalar: a small or high-contrast core, several vector channels, a declared
circular basis, or a guide whose geometry moves along z. It sets out the model the
**general modal route** solves, how to run it, what it measures about its own accuracy,
and what it refuses. How the route relates to every other transverse decomposition solprop
has is set out, with a figure each, in [spatial decompositions](../setup/spatial-decompositions.md);
the step-by-step workflow, with snippets that run as written, is
[solid-core fibres](solid-fibres.md).

A waveguide mode is not a scalar. A bound hybrid mode of a solid or a hollow fibre has
$E_x$, $E_y$ **and** $E_z$, and its nonlinear polarisation mixes all three; the
transverse scalar profile the rest of solprop propagates against is the weakly guiding
limit of that, which is excellent for a large hollow core and progressively worse as the
index contrast rises and the core shrinks. The **general modal route** propagates the
full three-component field instead: a provider solves $\mathbf E$ and $\mathbf H$ at
quadrature nodes across the cross-section, the run synthesises the physical vector field
there, evaluates the material's own response at each node, and projects the result back
onto the channels with the reciprocity relation that defines them.

It is reached through the waveguide entry points, with one extra argument:
{meth}`~solprop.driver.simulation.Simulation.from_waveguide` and
{func}`~solprop.driver.interface.prop_waveguide` take a `provider=` in place of a gas
fill. What the route supports, and what it refuses rather than approximates, is set out
in the last two sections of this page; read them before planning a run. In short: the route
runs on **a field grid, for a guide of constant geometry or a solid guide whose geometry
moves along z**, on `numpy_ref`, `numba_cpu` and the native backend (its CPU build and
its CUDA and HIP device builds alike); a coupled flow runs on all of them too, except
that a flow with a CF4 section runs on `numpy_ref` alone. A general modal run writes,
reads back and resumes an HDF5 file (output schema 7), a scan point writes one, and a
batch of launches on one guide runs in lockstep. A declared basis over a block that is
only *nearly* degenerate propagates as a coupled matrix flow, and a guide whose geometry
moves carries the connection of its moving basis, which couples its channels.
[Declared, refused, recorded](#declared-refused-recorded) is the whole route on one
screen.

## Running one

Three objects, in order: the material regions, the guide, the run. An executed version
with four vector channels, and the rule for the band the guide must span, is
[a step-index fibre, with its vector modes](solid-fibres.md#a-step-index-fibre-with-its-vector-modes).

```{code-block} python
import numpy as np

from solprop.driver import prop_waveguide
from solprop.grid import FieldGrid
from solprop.modes import (
    IndexMaterial,
    MaterialKerr,
    MaterialRegion,
    ModalGuide,
    StepIndexMode,
)

length = 1e-3
grid = FieldGrid.from_limits(length, 835e-9, (790e-9, 890e-9), 200e-15)

# 1. The geometry, and the materials that own the responses. A region with no
#    MaterialKerr is linear, which is how a cladding is said to be linear.
core_law, cladding_law = IndexMaterial(1.45), IndexMaterial(1.44)
mode = StepIndexMode(4e-6, core=1.45, cladding=1.44, profile_wavelength=835e-9)
regions = (
    MaterialRegion(
        "core", 0.0, mode.core_radius, core_law, (MaterialKerr(n2=2.6e-20),)
    ),
    MaterialRegion("cladding", mode.core_radius, None, cladding_law),
)
strand = ModalGuide.from_step_index(
    [mode], core=core_law, cladding=cladding_law, regions=regions
).field_provider

# 2. The guide: the provider, solved on its own node rule at its own knots, over
#    the run's whole ACTIVE band -- twice the declared one, see below.
active = grid.omega[grid.active_mask]
guide = ModalGuide.from_provider(
    strand,
    np.linspace(0.99 * active[0], 1.01 * active[-1], 13),
    nodes=strand.node_rule(0.99 * float(active[0]), n_r=24),
)

# 3. The run. The guide supplies its own channels, so `modes` is empty.
output = prop_waveguide(
    (),
    length,
    provider=guide,
    grid=grid,
    lambda0=835e-9,
    fwhm=40e-15,
    energy=4e-11,
    n_save=2,
    backend="numpy_ref",
)
```

The run warns once, with a {py:class}`~solprop.exceptions.GeneralModalBasisWarning`:
thirteen knots leave the interpolated basis $1.1\times10^{-4}$ from orthonormal near
1.73 µm, at the long-wavelength edge of the active band where the launched pulse puts no
light. The warning, and why it is a warning rather than a refusal, are explained under
[normalisation](modal-projection.md#normalisation-and-why-the-power-weight-is-one); more knots remove it.

**`provider=` takes a guide, not a provider.** Despite the keyword's name, what it wants
is a {class}`~solprop.modes.guide.ModalGuide` — a field provider *together with* its
material regions, its ordered channels and the
{class}`~solprop.modes.guide.ModalSampling` that records what they were solved on — and
a bare {class}`~solprop.modes.vector.ModalProvider` carries none of those.

`channel=` chooses which channel the pulse enters, by index or by display label
(`"HE1,1:even"`); the default is the first. The launch is the pulse's amplitude placed
in that channel **unchanged**: a provider normalises every channel to unit longitudinal
Poynting flux, so a channel amplitude already is a true-power amplitude in $\sqrt{\rm W}$
and no $\sqrt{\operatorname{Re}n_{\rm eff}}$ conversion happens anywhere on this route.
The launched energy is the pulse's, to the last bit.

**`incident=` launches an incident beam instead.** Where `channel=` names one channel,
`incident=` takes a {class}`~solprop.pulses.SpatialPulse` — a Gaussian or super-Gaussian
beam carrying its own temporal pulse — and launches what the guide's channels capture of
it, coupled under a declared model
([SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md);
[coupling simulations](../workflows/coupling.md)):

```text
beam = SpatialPulse(GaussianPulse(wavelength=835e-9, fwhm=20e-15, energy=1e-9), 3e-6)
simulation = Simulation.from_waveguide(
    (), length, provider=guide, lambda0=835e-9, grid=grid, incident=beam,
    coupling=CouplingModel("vector_paraxial", model_error=0.2),
)
simulation.configuration["launch"]   # the CouplingReport: incident, launched, budget
```

The beam's `energy=` (or `peak_power=`) is met **over the whole beam** in the paraxial
measure $M_0 = Y_0\,\mathrm dA/2$ of the guide's core medium, $Y_0 = n_{\rm ref}(\omega)
\varepsilon_0 c$ with $n_{\rm ref}$ the real index of the core at each frequency,
integrated over the beam's own plane in closed form
({meth}`~solprop.pulses.SpatialPulse.incident`) — never over the guide's quadrature
nodes, which cover only the guide. The beam becomes a radial
{py:class}`~solprop.coupling.plane.FieldPlane` that resolves it
({py:meth}`~solprop.coupling.plane.FieldPlane.from_pulse` with the core's index), and
{py:func}`~solprop.coupling.coupler.couple` recovers its coordinates in the channels.

The map is the flux pairing, $\mathbf a = F^{-1}\mathbf B$ with
$B_m = \int(\mathbf E\times\mathbf h_m^* + \mathbf e_m^*\times\mathbf H)_z\,\mathrm dA/4$
and $F$ the same pairing of the channels with each other, on the guide's nodes. The
beam's $\mathbf E$ is its field at the nodes; its $\mathbf H$ is the explicit forward
closure $\mathbf H_t = Y_0\hat{\mathbf z}\times\mathbf E_t$ of a paraxial beam in the
core. That closure is not exact for a guided field -- a channel's own $\mathbf h_t$ is
$n_{\rm eff}\varepsilon_0c\,\hat{\mathbf z}\times\mathbf e_t$ to leading order, with
corrections where the index changes -- so the model admits a guide only where it is
small: the coupling's admission budget (`model_error`) bounds the closure's error
against the provider's own $\mathbf H$, the weak-guidance contrast of the guide's
regions, the angular content of the beam and of the channels, and the metric defect of
$F$ against the run's declared metric. A strongly guiding core, such as a silica strand
in air, fails the contrast screen and is refused by name. The pairing also omits the
light the entrance facet reflects; for a step-index guide that omission is bounded by a
recorded envelope against an independent mode-matching solution of the facet, counted
in the budget and refused outside its range (contrast to 0.05, V from 1.5 to 8; see
[the coupling chapter](../workflows/coupling.md#the-admission-budget-and-the-report)).
A graded-index guide has no recorded envelope; its report says so and leaves the item
uncounted. It is *not* $Q$, the record's
other map ([modal projection](modal-projection.md#the-synthesis-and-projection-maps)),
which contracts a *polarisation* for a **source** term. For a field you sample yourself,
{py:func}`~solprop.translate.modal_tables.project_transverse_field` applies the
coordinate dual of the channels' own span, whose E-only closure is exact for a forward
superposition of the channels.

**The launch is the recovered amplitudes, unscaled.** Light the channels do not capture
is lost at the entrance, not restored: a beam much wider than the core launches a small
share of its energy, and a beam the channels cannot see at all — a y-polarised beam
against a single x-polarised channel — launches nothing, which is a valid zero launch.
The coupling's {class}`~solprop.coupling.coupler.CouplingReport` is recorded as
`configuration["launch"]` and reaches the file's configuration: the **incident** energy
per frequency bin, the **launched** energy (the state's physical energy under the run's
own declared metric, which is what the run's `physical_energy` statistic reads at the
entrance), the admission items and the budget; the launch's provenance is
`configuration["coupling"]`. The vector model names no uncaptured energy: $F$ and the
incident measure are not one positive metric, so their difference is not a physical
loss, and the report says so rather than inventing one. No warning is raised for a poor
capture: it is physics, not a defect.

The beam is launched at its waist (`propz=0`; otherwise it is refused, because carrying a
beam to the entrance is transport, which this launch does not model) and centred on the
guide's axis; a one-component Jones vector is polarised along x. The beam carries its own
pulse, so `fwhm`, `width`, `energy`, `peak_power`, `phases`, `pulses` and `pulseshape` are
refused beside it, as is `channel=`. Direct amplitudes — a named channel, or a state
passed to {meth}`~solprop.driver.simulation.Simulation.with_input` — already describe
light inside the guide and are launched as they are.

Every gas keyword is **refused by name** beside a provider — `gas`, `pressure`,
`temperature`, `kerr`, `thg`, `plasma`, `raman`, `n_r`, `n_theta` and the rest — because
a guide owns its materials, its responses and its transverse rule, and a keyword
silently ignored would let you believe a fill or a resolution had been applied.
`observable_rtol` and `observable_atol` are accepted as `None`, which means the
solver's own `rtol` and `atol`: the route's basis is orthonormal in true power, so the
controller's weighted norm on the amplitudes already is a norm on the physical channel
powers.

For a worked example that runs and plots, see
{doc}`../../auto_tutorials/waveguides/plot_05_general_modal`; for a **graded-index** guide, its
self-imaging beat and a launch declared as a field, see
{doc}`../../auto_tutorials/waveguides/plot_06_graded_index`; for a declared basis of a **split** pair,
propagated as a coupled flow and checked against the provider's own basis, see
{doc}`../../auto_tutorials/waveguides/plot_07_split_block`. The [guide to guided
propagation](solid-fibres.md) sets out the workflow step by step, and the
[graded-index worked example](../../auto_examples/waveguides/plot_04_graded_index_multimode.rst) runs a
multimode fibre at full size.

**A guide must be solved over at least twice the band a run declares.**
`Grid.active_mask` runs from half the lowest declared frequency to the window's highest,
so the `FieldGrid` above, declared over 790–890 nm, has an active band of
724.9–1774.0 nm — a factor 2.45 in frequency — and the guide's table must cover all of
it, at the low end of which a weakly bound mode reaches furthest into the cladding and a
truncating node rule discards most of it. Outside the table's own knots the stored
reference is an explicit zero, and a run that reaches there is refused with both bands
in the message rather than propagated against zeros. An `EnvelopeGrid` of the same
window has a slightly wider active band, so a guide meant to serve a comparison against
{func}`~solprop.driver.interface.prop_gnlse` has to cover the union of the two.

**The collar, and not the route, sets the step on a short grid.** `from_waveguide`
builds the grid's default temporal collar, applied as a rate inside every stage
([absorbing boundaries](../numerics/boundaries.md#the-two-temporal-schemes)). Its rate is set by the
length the *grid* was declared for and by the time window, and on a short run that rate,
not the nonlinearity, is what the step controller reacts to. On the millimetre above the
general route takes 42 accepted steps with the collar and exactly 10 without, and
`prop_capillary` on the same grid does the same thing: this is not the general route's
doing, it only matters more here because a general modal step costs about a hundred
fused ones. Two settings change it: `absorber_scheme="split_step"` applies the same rates
once per step, outside the stage, and reaches the same answer in the collar-free step
count; `absorber=None` in the settings removes the absorber, for a lossless check.
`absorber=None` passed to `from_waveguide` itself means "build the default collar", so
replace the settings before the first `translate()` or `run()` instead:

```{code-block} python
import dataclasses

from solprop.driver import Simulation

simulation = Simulation.from_waveguide(
    (), length, provider=guide, grid=grid, lambda0=835e-9, fwhm=40e-15, energy=4e-11
)
simulation.settings = dataclasses.replace(
    simulation.settings, absorber_scheme="split_step"
)
```

`prop_gnlse` builds no absorber at all, so a like-for-like comparison with it needs one
of the two.

## Noise, statistics and what comes back

Because the basis is orthonormal in true power, **one photon per channel per bin already
is an energy-normalised amplitude**: the shot-noise draw of every other route
({doc}`../media/noise`) is carried as it stands, with nothing multiplying it by an index, and a `seed=` reproduces the draw
exactly. In a **declared** basis the draw is made once, in the orthonormal
true-power reference basis, with the same seed and the same stream, and pushed through
$U^{\mathsf H}$ ({py:func}`~solprop.noise.transformed_noise`): the total photon number
in each bin is exactly what it was and the per-channel numbers are redistributed, which
is the point — a fixed-modulus random-phase draw is *not* invariant in distribution
under unitary mixing, so drawing again in the declared basis would be a different
stochastic model. A basis with unequal channel norms is neither of those and is refused
by name.

The route's statistics are **`physical_energy`** and **`analytic_flux`**, and they are
its defaults. Both are scalars: a coordinate energy and a coordinate power are not
additive on a basis that is not orthonormal, so the names that promise one are refused
here rather than redefined. `physical_energy` is the same-frequency product of the state
with the basis's flux metric, integrated over frequency; `analytic_flux` is the peak over
time of that same **same-frequency flux**, $\max_t \mathbf a^{\mathsf H}W\mathbf a$. On
an identity metric — every guide in the provider's own coordinates — that is
$\max_t\sum_m|a_m(t)|^2$ and their arithmetic is exactly `sum(energy(...))` and
`peak_power_total`, and their docstrings say so.

A third name, **`spectral_flux`**, is the same quantity resolved in frequency: the
spectral density $\mathrm dU/\mathrm d\omega$ of `physical_energy`, in J/(rad/s), summed
over the channels after the whitening exactly as `physical_energy` is. It is **admitted
but not a default** — it is `n_omega` wide per plane where the other five statistics are
scalars — so ask for it in `statistics=`; `d_omega` times its sum is the
`physical_energy` the same plane records.

`analytic_flux` is **not** the cross-section flux of the reconstructed fields. That one
carries the guide's cross-frequency flux kernel as well, and the two peaks differ by at
most $\sum_{ij}|B_i||B_j||K_{ij}-1|$ — measured at 1.3e-5 of the peak inside a bound of
3.4e-5 for one $\mathrm{HE}_{11}$ channel of a 1.5 µm strand and a 20 fs pulse, while the time integrals agree exactly. For
the flux itself ask {py:func}`solprop.processing.modal.reconstructed_flux`, which
rebuilds E and H and returns the flux as a trace in time; see
[statistics](../results/statistics.md).

`energy`, `peak_power`, `peak_power_total`, `energy_window`, `peak_power_window` and
`peak_intensity` are therefore **refused by name**, each with the replacement to use, at
all four places a statistic can be asked for: the run loop's statistics admission, the
backends' `stats`, the accessors of {py:mod}`solprop.processing.modal` and the
scalar-route {py:func}`~solprop.processing.time.energy` and
{py:func}`~solprop.processing.time.peak_power`, which read only the stored record.
`omega0`, `fwhm_t` and `fwhm_t_outer` are admitted, and they **take the metric** where
the identity no longer holds: they are functionals of the same physical same-frequency flux
$\mathbf a^{\mathsf H} W\mathbf a$ that `analytic_flux` peaks, so a run does not report
the width of a quantity whose peak it does not report. `zdw` takes no metric: it is a
property of the channels' propagation constants and of no basis at all, and a unitary
transform of an exactly degenerate block leaves every one of them where it was; beside a
**split** block, which has no per-channel propagation constant, it is withheld (above).
The backend evaluates $W$ at each plane and hands it over; see [statistics](../results/statistics.md). `quadrature_error` is **refused by name** too, rather than recorded as
the `nan` it would be: a provider's node rule has no embedded coarse rule to difference,
and a `nan` at every plane would read as "no alarm was raised" when what is true is that
**a general modal run has no automatic transverse-resolution alarm** — the rule's tail
bound and its sampled Gram are checked once, when the guide is built, and nothing watches
the resolution during the run.

**Output schema 7 carries this basis.** A general modal run writes an HDF5 file, reads it
back and resumes it: the record persists the channel labels row by row, the provider, the
sampling the reference is rebuilt from, the materials and their declared Kerr responses,
the table and metric identities with the measured Gram defect, the shared co-moving
frame, the run's active band, the basis the noise was drawn in, and the list of accessors
the basis admits. All of that is inside the resume key, so a run continued against a
changed guide, table, material or channel order is refused by the field that moved, and a
file written before schema 7 is never resumed into one. A **declared change of basis** is
in the key four times over — the metric's kind moves to `dense`, both gauges gain
`+circular(i,j)`, the channels carry handedness instead of parity, and the sampling
carries the declaration with its matrix — so a file written in one basis cannot resume in
another. The sampling's entry appears only where a basis is declared, so a run in the
provider's own coordinates carries no such entry; the two gauges are always compared, and
they refuse a resume in the other direction.
{meth}`~solprop.output.memory.MemoryOutput.to_file` and a scan point write the same file.

**A batch member may be a general modal run**, under a narrower admission rule than the
one for fibre runs ({doc}`../workflows/scans`) and alongside it: the members of a general-guide group
share the physical guide, the table, the gauge, the materials and one explicit fixed
schedule, and differ in their **launch and their noise seed** and in nothing else. What
separates two guides is `modal_tables.identity` — its three declared identities and its
extents — in the grouping signature, beside the metric's kind and gauge and the compiled
responses' model and policy; two strands of different radius share every *layout* entry
of the signature and are told apart by what they physically are. Because a descriptor is only
as good as what it names, a group assembled by hand is refused by name at the driver and
again at the backend: *general-guide batch requires identical guide/table/gauge
identity*. The source pass is unchanged by batching: the lockstep runner hands each
member its own singleton row, and a batched run is what its members are alone, to
rounding.
A **fused** batch axis — one problem of shape `(n_batch, ...)` carrying modal tables — is
refused by the reference backend, because the stage evaluates one member's
polarisation.

The file stores channel amplitudes, not fields. {py:mod}`solprop.processing.modal` is the
way back — given the run's own tables, rebuilt from the guide the record names, it
synthesises $\mathbf E$ in V/m and $\mathbf H$ in A/m at the guide's nodes and forms
`reconstructed_flux` in time, `instantaneous_flux` of the real fields with its
sum-frequency term, and `coordinate_intensity`, the flux density each channel would
carry alone, which is **not additive** and is never clipped; `plane_nodes` returns the nodes and area
weights those values live on, which on a moving guide are each plane's own. **A
reconstruction needs the guide beside the file**: the arrays are not in it, and `guide=`
is a **required** keyword of all six reconstruction accessors, so the sampling and the
regions are always checked too. See [output](../results/output.md).

## Declared, refused, recorded

This is the whole route in one place, each line pointing at the section that explains
it.

**What you declare.**

| declaration | where | explained in |
|---|---|---|
| the guide: a provider, its material regions and their responses, solved on its own node rule at knots spanning the run's **active** band | `ModalGuide.from_provider(provider, omega, nodes=...)` | [running one](#running-one), [the three objects](modal-projection.md#the-three-objects) |
| each region's response: the instantaneous cubic by `chi3=`, or by `n2=` with the wavelength `at=` it was measured at, and the silica tensor Raman model beside it; or a gas fill, whose law the run's electronic policy sets | `MaterialRegion(..., (MaterialKerr(...), SilicaTensorResponse(...)))`, `MaterialRegion(..., (GasKerr(mixture),))` | [the response at a node](modal-projection.md#the-response-at-a-node) |
| the cubic's form | `kerr_form="real"` (the default, third harmonic included) or `"analytic"` | [the response at a node](modal-projection.md#the-response-at-a-node) |
| the launch | `channel=` (an index or a label) or `incident=` (a `SpatialPulse` coupled under `coupling=`, of which the captured share enters) | [running one](#running-one) |
| a unitary change of basis — of an exactly degenerate block, which stays diagonal, or of a split one, which becomes a coupled flow | `basis=` and `degenerate_blocks=` on `from_provider` | [declaring a circular basis](modal-bases.md#declaring-a-circular-basis), [a split block](modal-bases.md#a-declared-basis-of-a-block-that-is-only-nearly-degenerate) |
| the statistics | `physical_energy` and `analytic_flux` (the defaults), `spectral_flux` by name | [noise, statistics and what comes back](#noise-statistics-and-what-comes-back) |

**What is refused, and why.** Every refusal names itself, at the entry point and again
at the translation layer and in the backend.

* Every gas keyword beside a provider: the guide owns its materials, its responses and
  its transverse rule, and a keyword silently ignored would read as applied.
* A CF4 section on any backend but `numpy_ref`: it has no execution path there (its
  Gauss-node weights are evaluated inside the reference backend), and nothing falls
  back.
* A guide that does not span the active band, or whose sampled or interpolated Gram is
  further from orthonormal than `gram_atol`: the run would propagate against zeros, or
  against a basis it declares orthonormal and is not.
* An `n2` with no `at=` whose conversion to $\chi^{(3)}$ moves by more than one per cent
  across the band, and the real cubic on a grid with no room for its harmonic.
* A nonunitary basis, or one acting outside a declared block; and beside a split block,
  a noise model (its transport through a coupled flow is not measured), a lockstep batch
  member (it applies neither half of the flow) and a block whose exponent exceeds
  `MAX_GENERAL_FLOW_EXPONENT`.
* Beside a guide whose geometry moves: a snapshot position other than the entrance, a
  run longer than the declared path, a noise model, a lockstep batch, a fixed schedule
  that straddles a kink, and a CF4 flow or no flow at all beside the moving maps
  ([a guide whose geometry moves along z](moving-modal.md#a-guide-whose-geometry-moves-along-z)).
* The scalar-route power names, `quadrature_error` and — beside a split block — `zdw`: a
  statistic the basis does not define is refused with its replacement where it has one,
  rather than redefined.
* `observable_rtol` and `observable_atol` as numbers, which nothing would enforce.

**How to read the record.** `output.metadata.modal` is the run's schema-7 modal
identity, and every entry of it is in the key a resume is checked against. `basis` says
which kind of channels the state carries; `guide` names the provider, the sampling it
was solved on and — only where one was declared — the change of basis with its matrix;
`tables` and `metric` carry the table and metric identities with the measured Gram
departures; `channels` lists the labels row by row; `accessors` lists the statistics and
reconstructions the basis admits, which is where a withheld `zdw` is visibly absent; and
`flow` appears **only** where the run carries a coupled flow, with the entries tabulated
under {ref}`reading the record <coherent-wall-record>`.

**The same leaf elsewhere.** The coupled flow a split block becomes is the
`linear.coupled_flow` leaf the capillary's **coherent wall** also becomes, and the
execution is shared: one Schur matrix-function action per stage, with the certificate
checked against the measured action at every step. The wall is documented in
three parts across two chapters: [the coherent wall](coherent-wall.md#the-coherent-wall) at constant
geometry; [a guide that varies along z](coherent-wall.md#a-guide-that-varies-along-z), where the
capillary's own connection enters as an explicit residual beside frozen sections; and
[the vector boundary response](vector-wall.md#the-vector-boundary-response), which couples the
two radial families of one azimuthal order. Each has its gallery example:
{doc}`../../auto_tutorials/capillaries/plot_07_coherent_wall`,
{doc}`../../auto_tutorials/capillaries/plot_08_coherent_wall_taper` and
{doc}`../../auto_tutorials/capillaries/plot_09_vector_wall`.

## What is implemented, and what is not

Implemented, and executed by every backend — `numpy_ref`, `numba_cpu`, and `native` in its
CPU build and its CUDA and HIP device builds — with the one exception of a CF4 section,
which runs on `numpy_ref` alone:

* `Simulation.from_waveguide(..., provider=guide)` and
  `prop_waveguide(..., provider=guide)`, with the launch, the noise draw, the statistics
  above, in-memory results and an HDF5 file that reads back and resumes (schema 7);
* the three-component nonlinear source, chunked over nodes with its own memory ledger;
* the instantaneous cubic in two declared forms, `kerr_form="real"` (the default, third
  harmonic included) and `"analytic"`, and the silica tensor Raman model as a region's
  declared delayed response;
* a region's gas fill under the run's electronic policy, the dispersive cubic of SEP 6
  on every backend;
* a declared **unitary change of basis**, with its circular-pair constructor, its dense
  metric, its relabelled channels, its carried noise draw and its place in the resume
  key — of an **exactly degenerate** block, which stays diagonal, and of a **split**
  one, which becomes the `linear.coupled_flow` leaf described above;
* the **connection** of a basis that moves along z, measured from a provider with the
  flux self-terms kept ({py:func}`~solprop.modes.connection.modal_connection`), and the
  connection a declared moving change of coordinates costs
  ({py:func}`~solprop.modes.connection.basis_connection`);
* a **moving solid guide**: the provider re-solved along its declared radius path, its
  sectioned synthesis and projection maps, and the frozen-residual coupled flow that
  carries its connection
  ([a guide whose geometry moves along z](moving-modal.md#a-guide-whose-geometry-moves-along-z));
* a **graded-index** guide as readily as a step-index one: any provider that solves its
  own E and H, with `GradedIndexProvider(core_responses=…)` declaring what its core
  responds with;
* a launch into a **named channel** or a declared **transverse field**, and a **batch**
  of launches on one guide.

The route admits one combination and no other: a **field** grid; an identity metric
whose certified defect is inside the route's budget, or the dense metric of a declared
unitary change of basis of one; a constant geometry (one Chebyshev interval of degree
zero along z) or a moving solid guide's sectioned maps beside the frozen-residual flow
that carries their connection; a state with `n_pol = 1`; the constant unit power weight;
and each region's declared responses — the instantaneous cubic in either form, and the
delayed silica tensor Raman model beside it — over the reference's own three components.
Everything else is refused by name, at the entry point, again at the translation layer
and again in the backend. Every backend admits through one list, the reference's, run
under its own name, and nothing falls back: a leaf without a kernel is refused by its
dotted name, and `backend="auto"`, which selects `native` where it is installed, runs a
general guide and a coupled flow there. `numba_cpu` executes the route through an
execution path of its own — its fused pipeline is written for one real transverse matrix
in the time domain, and this route's synthesis is a complex frequency-dependent map of
three components — in exactly the combinations `numpy_ref` admits, and agrees with it to
rounding (a few 1e-16 after ten steps); `native` agrees with `numpy_ref` to a derived
rounding bound. See [backends](../performance/backends.md) and [the native backend](../performance/native.md).

Some refusals are **permanent** and not a gap to be filled. `peak_intensity`, like
`energy` and `peak_power`, is refused because a general basis does not define it, and
solprop never redefines an established name on a basis it does not describe: the
replacement is `analytic_flux`, or reconstructed E and H with an explicit detector
definition. The same holds for every scalar-route power name above.

Not implemented, and refused rather than approximated:

* **`quadrature_error`**, refused by name rather than recorded as a `nan`, because a
  provider's node rule carries no embedded coarse rule to difference. Unlike the power
  names this one is a gap: a second rule, or a tail estimate, would give the alarm
  something to report;
* the **reference arrays in the file**. Output schema 7 records the guide's *identity*
  and not its tens of megabytes of stored E and H, so a reconstruction is made by
  rebuilding the tables from the guide — tens of milliseconds for the route, about 0.7 s
  if the guide itself must still be solved — and every accessor checks them against the
  identity the file carries, path by path, with the guide, which they require;
* **`observable_rtol` and `observable_atol` as numbers.** `None` is accepted and means
  the solver's own `rtol` and `atol`, which on this route is a true statement; a
  different number is refused by name, because the separate error allocation it asks
  for — over the table's frequency interpolation, the transverse rule and the step — is
  not implemented, and accepting a number nothing enforces would be a silent claim about
  accuracy;
* a **backward or evanescent** declared launch. `incident=` (above) recovers the
  incident field through the flux pairing with the forward-field closure, and nothing in the
  arithmetic can see that a declared field was not forward-propagating; such a launch is
  built by hand and passed to
  {meth}`~solprop.driver.simulation.Simulation.with_input`;
* a **nonorthogonal** basis: `modal_metric` of kind `"diagonal"`, which is a
  power-orthogonal basis with unequal channel norms and which nothing here builds, and
  any `"dense"` metric that is not the unitary change of basis of a certified
  orthonormal reference described above. A *nonunitary* transform leaves a skewed or
  leaky basis whose flow is a matrix exponential and whose noise transport nothing has
  measured, and it is refused at three layers by name;
* a guide sampled over a support **narrower than the run's active band**. An opt-in
  fallback — the field zero-filled outside the support, the run warning by name when
  energy reaches its edge — would suit guides whose every solve is expensive, such as
  finite-element tables, but it is not built: the linear operator asks every channel for
  a propagation constant across the whole active band, through the one site every route
  in the package shares. A guide that does not span the active band is therefore
  refused, with both bands in the message;
* a **CF4 section** on the compiled backends. `numpy_ref` evaluates a CF4 step's
  Gauss-node weights itself, and `numba_cpu` and `native` have no path for them.

Two approximations are worth naming because a reader would otherwise take them as exact.
Outside the guide's sampled band the stored reference is an **explicit zero** rather than
an extrapolated profile — which is why the table must span the whole active band, twice
the declared one or more, as above. And on a constant guide the node rule is frozen at
the entrance geometry (a moving solid guide's rule moves with its interfaces, section by
section): the rule's certified tail bound and its sampled Gram belong to that geometry
and to no other.

The incident-beam launch (`incident=` under `coupling=`) and its coupling budget were
proposed and carried out by [SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md).
