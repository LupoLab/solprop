# Coupling simulations

A simulated pulse is often the input to another optical component: a second capillary,
a fibre, a gas cell. Carrying only its spectrum or duration loses its wavefront, its
modal coherence and the frequency dependence of its transverse profile; carrying a bare
array leaves its basis, units, frame and energy meaning implicit.
{py:mod}`solprop.coupling` connects simulations through a **coherent field at a
reference plane** instead ([SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md)).
This page covers reading a saved plane as a field plane, moving it between temporal
grids, and coupling it into another simulation: a relaunch into the same basis, a
beam or a guide's exit into a capillary or another fibre, a guide's exit field
synthesised on a spatial grid, and transport of a spatial plane through free space,
lenses, apertures, mirrors, interfaces and spectral optics.

## A plane, and what it means

A {py:class}`~solprop.coupling.plane.FieldPlane` holds a field's native coefficients
exactly as an output stores them -- the optics-convention raw spectrum of each
column, referenced to the centre of the time window -- together with everything
needed to use them:

| part | what it says |
|---|---|
| `kind`, `units`, `basis` | the representation: ordinary true-power mode coefficients (`modal`, √W), general-modal coordinates (`general_modal`, √W), the Hankel spectrum of a radial field (`radial`, V m) or the transverse DFT of a Cartesian one (`cartesian`, V/m); the basis carries an identity digest |
| `grid` | the temporal grid: field or envelope layout, time samples, the absolute frequency of every bin, the carrier |
| `reference` | the affine reference of the stored spectrum (below) |
| `geometry` | the plane's position, the side of a declared index jump, and its oriented right-handed transverse frame |
| `metric` | how energy is measured there: the identity for true-power coefficients, a general basis's Gram, or the medium's axial admittance for a spatial plane ([medium flux](../free-space/flux.md)) |
| `exterior` | whether the record is zero outside its samples or unknown there |
| `certificate` | the scoped path certificate of a run through a gas gradient: it bounds the linear background path, not the whole nonlinear run |
| `provenance` | the source run's configuration hash, the plane, its lineage and what was done to it |

The plane owns read-only copies of its arrays, so it cannot be changed through an
alias. `plane.energy()` is the physical energy crossing the plane in joules,
`plane.spectral_energy()` the energy per bin.

**The reference.** A run's spectrum is stored in its moving frame: the linear operator
removed $R(\omega,z) = \beta_0(z) + \beta_1(z)(\omega - \omega_o)$ (the
[frame](../../dev/conventions.md), with $\omega_o$ the reference frequency on an envelope
grid and zero on a field grid). The absolute optics-convention spectrum is

$$
\tilde E_{\rm abs}(\omega) = \tilde E_{\rm stored}(\omega)\,
e^{i(\phi + \omega\tau)},\qquad \phi + \omega\tau = \int_0^z R\,dz ,
$$

with ω the absolute frequency. τ is the absolute time of the record's $t = 0$ and φ a
carrier phase; `plane.absolute()` applies them. Keeping the pair as numbers, rather
than shifting the samples, means that a long path never translates a femtosecond pulse
out of its window. Each run records its frame integral, so any saved plane's reference
is rebuilt from the file alone (output schema 24,
[provenance](../results/provenance.md#the-metadata-schema)).

## Reading a plane back

{py:func}`~solprop.processing.planes.field_plane` reads one saved plane; by default the last.
A position must name a stored plane: a nearby plane that was not saved is refused,
never substituted.

```python
from solprop.driver import Simulation
from solprop.processing import field_plane

simulation = Simulation.from_capillary(
    100e-6,
    1e-3,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15,
    fwhm=20e-15,
    energy=20e-6,
    modes=2,
    n_save=3,
    noise=None,
)
run = simulation.run(progress="none")
plane = field_plane(run)  # the exit plane
middle = field_plane(run, z=0.5e-3)  # a stored plane by position
print(plane, plane.reference.delay, plane.energy())
```

A general modal file records the identities of its E/H reference and not the
reference's arrays, so its physical fields (`plane.nodal`) and a dense basis Gram are
read with the simulation that ran it, `field_plane(run, source=simulation)`; the
identities are checked first, as {py:mod}`solprop.processing.modal` checks them. A
spatial plane's measure is rebuilt from the file's plane measure; with `source=` it
also carries the medium. A hybrid planar run's planes are refused by name, and so are
the planes of a run with the modified noise model (`noise="modified"`): its saved field
leaves the one-photon background to an auxiliary noise phase, which is not reconstructed
into a plane, so the stored samples alone would drop it in the next stage. A source run
meant for coupling uses the default `noise="input"`, whose background travels in the
saved field, or `noise=None`.

## Moving a plane between grids

{py:func}`~solprop.coupling.remap.regrid` puts a plane on another temporal grid -- a longer
or finer window, a field grid instead of an envelope one, another carrier -- or into
another reference. Copying raw DFT amplitudes between grids of different size or
spacing is not this operation. The plane's analytic signal is its finite record, zero
outside its window, and each target bin receives its finite-time Fourier sum

$$
s'_m = \frac{\Delta t}{\Delta t'\,q'_m}\,e^{i\Delta\phi}
       \sum_j a_j\,e^{i\omega'_m (t_j + \Delta)},
$$

with $q = 2$ on interior bins of a field grid and 1 otherwise, and $\Delta$,
$\Delta\phi$ the change of reference, so that the absolute spectrum is unchanged. The
direct sum is the implementation; complex spectral zeros and separated bands need no
special case, since no phase is unwrapped and nothing is interpolated.

```python
from solprop.coupling import regrid
from solprop.grid import FieldGrid

wider = FieldGrid.from_limits(1e-3, 800e-9, (300e-9, 2e-6), 400e-15)
moved, report = regrid(plane, wider)
print(report.method, report.band_loss, report.window_loss)
print(moved.energy() / plane.energy())  # no renormalisation, and none needed here
```

What cannot be represented is refused, and what is removed is reported. Light outside
the target's active band is removed only with `crop_band=True`, light outside its time
window only with `truncate=True`, each with its share in the
{py:class}`~solprop.coupling.remap.RegridReport`. A record whose edges carry light, or whose
exterior is unknown, is not extended into a larger window unless `truncate=True`
declares it zero outside. Nothing is ever renormalised: a conversion does not recover
clipped, aliased or absorbed light. `delay=` and `phase=` change the reference: a later
delay moves the record earlier in its window, with the absolute field unchanged.

## Relaunching into the same basis

A receiving simulation describes its entrance with
{py:meth}`~solprop.driver.simulation.Simulation.input_port`: the route, the temporal lattice and
basis with their identities, the entrance plane and the frame accumulated there, the
native metric at the entrance, the reference index of its entrance medium and the
coupling models it admits. {py:func}`~solprop.coupling.coupler.couple` computes a launch
bound to that port under a declared {py:class}`~solprop.coupling.models.CouplingModel`.
The identity model, `same_basis`, launches a plane into a port of the **same** basis on
the same lattice, at an identically oriented plane in the same medium: the coefficients
are copied as they stand, and the plane's reference becomes the launched run's origin.
The launched run therefore continues the source's absolute time and carrier phase
without moving the pulse in its window.

```python
from solprop.coupling import CouplingModel, couple

second = Simulation.from_capillary(
    100e-6,
    1e-3,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15,
    fwhm=20e-15,
    energy=20e-6,
    modes=2,
    n_save=3,
    noise=None,
)
relaunch = couple(
    field_plane(run), second.input_port(), model=CouplingModel("same_basis")
)
continued = second.with_input(relaunch.launch).run(progress="none")
print(field_plane(continued).reference.delay)  # the whole 2 mm path's
```

Two 1 mm capillaries relaunched this way reproduce one 2 mm capillary: the comparison
is an accuracy envelope, since the second run restarts its step controller, and the
tests hold it to the tolerance's level on every pair of the three CPU backends. The
basis of an ordinary mode set is its value at the plane -- each mode's class, label,
effective area and effective index on the active band -- so the exit of one capillary
and the entrance of an identical one of another length are the same basis, and a
different radius is not. A simulation can also continue from inside its own path:
`simulation.input_port(z)` is an interior entrance of a modal run.

`same_basis` also takes a declared invertible change of basis or gauge,
`CouplingModel("same_basis", basis_change=BasisChange(matrix, source, target))`, which
must carry the plane's metric onto the port's -- it relabels the same field -- and, for
a general modal plane read with its simulation, recovers the coordinates in a port of
the same guide with the flux pairing, $\mathbf a = F^{-1}\mathbf B$, refusing a field
that is not in the port's span.

{py:meth}`~solprop.driver.simulation.Simulation.with_input` refuses a launch computed for any other
port -- another basis, lattice, entrance, orientation or medium -- whatever the shapes of
the arrays. It records the coupling in `configuration["coupling"]` (the model, the port,
the inherited origin and the source plane's provenance, so a run launched from a plane
names its parent), and it draws no new noise: the launch carries the realised field of
its source, and a realisation is added only when one is given explicitly. The receiving
run keeps its ionisation tables and their ranges, as every derived run does.

## Coupling between different representations

A different guide, radius or medium is not a relaunch. Three quantities must then be
kept apart ([SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md), section 4.3): the
Hermitian E/H flux pairing $F$ of a modal basis, the Gram $G = S^\dagger MS$ of a scalar
least-squares fit in a common measure $M$, and a representation's native power metric
$Q$ -- the identity for ordinary true-power coefficients, a dense Gram for a declared
general basis, the axial admittance for a spatial plane. They coincide only where a
model establishes it, and the uncaptured energy is a physical, nonnegative loss only
in a common positive metric. The models are:

| model | what it admits | what it computes |
|---|---|---|
| `same_basis` | an identical basis, a declared basis change, a general modal plane into its own guide | the coordinates exactly; no budget |
| `scalar_paraxial` | a radial or Cartesian beam (Gaussian, analytic or sampled) into a capillary or a weakly guiding step-index fibre; a capillary's exit into another capillary of any radius and mode set; one weakly guiding step-index fibre into another | the fit $\mathbf a = G^{-1}S^\dagger M\mathbf f$ in $M_0 = Y_0\,dA/2$, $Y_0 = n_{\rm ref}\varepsilon_0c$, then $b = \sqrt{\operatorname{Re}n_{\rm eff}}\,a$ |
| `vector_paraxial` | a beam, or an ordinary guide's exit, into a general modal guide that carries its E/H at fixed nodes | $\mathbf a = F^{-1}\mathbf B$ with the incident E and the forward closure $\mathbf H_t = Y_0\hat z\times\mathbf E_t$ |

Every other pair is refused by name. The reference index $n_{\rm ref}$ is the port's:
the core gas of a capillary, the core of a fibre. The scalar profiles are the run's own:
a capillary's $HE_{1m}$ modes are $J_0(u_mr/a)$ inside the core and zero outside, and a
step-index fibre's $HE_{1m}$ modes are taken in the weak-guidance limit as $LP_{0m}$,
$J_0(Ur/a)$ inside and $J_0(U)K_0(Wr/a)/K_0(W)$ outside, frozen at the mode's declared
profile wavelength as the run's profile is
({py:class}`~solprop.coupling.models.ScalarModeBasis`). A family with angular structure
(TE, TM, EH, $HE_{nm}$ with $n \ne 1$) has no scalar profile, and its port admits
`same_basis` alone.

**Overlaps.** A spatial plane's columns are the Hankel or Fourier spectrum of its field,
so a profile's overlap is taken by Parseval in that same spectral quadrature, with the
profile's closed-form transform: the incident norm, the overlaps and the Gram all use one
measure, the orthogonal-residual identity
$\|f\|_M^2 = \mathbf a^\dagger G\mathbf a + \|f - S\mathbf a\|_M^2$ holds, and a
profile's kink at the core radius is never sampled. Two modal planes are overlapped on a
Gauss--Legendre rule broken at both core radii.

### The admission budget and the report

`model_error` is a required energy budget $\epsilon_m$ in (0, 1) for the two approximate
models: the relative energy discrepancy the approximation, its tails and its numerical
errors together may cause. It is not a coupling-efficiency floor -- a poorly captured
beam is physics -- and it is separate from numerical accuracy. A cone half-angle
follows from it, $1 - \cos\theta_* = \epsilon_m/8$, and each coupling is admitted against
these items, every one a relative energy discrepancy:

| item | what it bounds | limit |
|---|---|---|
| `angular` | $[(1 - \cos\theta_*)U_{\rm cone} + U_{\rm tail}]/U$ of the incident field, and of the worst unit field in the retained target subspace; the cone term is the cone rule's allocation, about $\epsilon_m/8$ of the cone share, not a measured defect | the total |
| `contrast` | the weak-guidance screen $\Delta = \max|n^2 - n_{\rm ref}^2|/(2n_{\rm ref}^2)$ of a solid guide; not applied to a capillary wall; a recorded pair envelope may replace it | $\epsilon_m/8$ |
| `target metric` | $G$ (or $F$) against the target's native $Q$, whitened | $\epsilon_m/8$ |
| `source metric` | the common measure $M_0$ against the source's native metric | $\epsilon_m/8$ |
| `closure` | $2\delta + \delta^2$ of the H-closure's field error $\delta$ (vector model) | $\epsilon_m/8$ |
| `passivity` | the whitened certificate $C^\dagger Q_BC \preceq Q_A$: an apparent gain within the model's own defects is reported here, never rescaled or clipped; a larger one refuses | the total |
| `envelope` | the recorded provider-pair envelope's allowance, plus the incident share of any bins outside its range | the total |
| `numerical` | the source's content at its spectral edge, an analytic field's domain and resolution checks, the overlap quadrature (an amplitude error $\delta$ counted as $2\delta + \delta^2$) | the total |

The counted items must sum to at most $\epsilon_m$. The
{py:class}`~solprop.coupling.coupler.CouplingReport` carries the incident energy over the
whole plane in $M_0$, the launched energy in the target's native metric, the captured
and uncaptured energy where a common positive metric exists (`None`, with the reason,
for the vector model), every item with its scope -- this field, the whole map, or the
model -- the source plane's exterior, and the source's SEP 34 path certificate,
separately and with its scope. A zero field is a valid zero launch whose `efficiency` is
`None`. The recorded envelopes ({py:mod}`solprop.coupling.envelopes`) bound what the
approximate models omit, measured by the test suite against independent physics:

- *Capillary.* The scalar profile $J_0(u_mr/a)$ against Marcatili's leaky core field
  $J_0(u_m(1 - i\delta)r/a)$, $\delta = |\nu_n|c/(\omega a)$ with $\nu_n$ the wall
  term (Marcatili and Schmeltzer 1964). Over waists 0.3a--1.0a, displacements to 0.2a,
  tilts to $0.59/ka$, 1--8 modes and radius ratios 0.5--2 the captured shares differ by
  at most $2.9\times10^{-3}$ at $\delta = 10^{-2}$, the range's top (an 18.8 µm silica
  capillary at 800 nm); the bound is twice that, $6\times10^{-3}$. The difference
  falls at least linearly with $\delta$, so the reported allowance is the bound scaled
  by the pair's energy-weighted $\delta$: about $1.5\times10^{-3}$ for a 75 µm
  capillary at 800 nm.
- *Step index.* The LP01 overlap against the full-vector flux overlap of the exact
  $HE_{11}$ fields: $3.0\times10^{-4}$ over contrasts to 0.01, V from 1.5 to 2.4 and
  radius ratios 0.7--1.43. V is checked at the profile wavelength and at every occupied
  bin, since it grows with frequency.
- *A beam into a step-index fibre.* The scalar LP01 overlap against the beam's
  full-vector coupling into the exact $HE_{11}$ pair: the flux pairing
  $\frac14\int(\mathbf E\times\mathbf h^* + \mathbf e^*\times\mathbf H)\cdot\hat z\,dA$
  with the beam's $\mathbf H$ and $E_z$ from its own plane-wave spectrum in the core
  index. The bound is not sampled over beams but maximised over every incident field
  inside the widest admitted cone (both captures are rank-two quadratic forms, so the
  largest eigenvalue of their difference bounds them all): $0.467\Delta$ at V 1.5,
  falling to $0.284\Delta$ at V 2.4 (conservative there, where the cone's edge truncates
  the widening $HE_{11}$ spectrum); at V 1.5 it is linear in $\Delta$ and independent of
  the core index and the wavelength.
  Gaussian beams of waist 0.5a--2.6a, displaced to 0.5a and tilted to half the
  acceptance angle reach $3.3\times10^{-3}$ at $\Delta = 0.01$ and V 1.5, a centred
  beam of waist 0.8a. The bound is $10^{-2}$ at contrast 0.01, scaled by the fibre's
  contrast (it is first order in $\Delta$): about $5\times10^{-3}$ for a 0.005 step.
  It holds for any beam shape, alignment and polarisation, for V from 1.5 to 2.4 (LP01
  alone); the angular item counts what lies outside the cone.

- *Vector model into a step-index guide.* The flux pairing with the incident field
  against an independent full-vector mode-matching solution of the planar entrance
  facet: the incident and reflected fields in the input medium, the fibre's forward
  guided and radiation fields, tangential $\mathbf E$ and $\mathbf H$ continuous,
  solved per azimuthal order in a large conducting cylinder on its own TE/TM basis.
  The exact amplitude differs from the model's by
  $-\tfrac14\langle\mathbf u_i, D(Y_0 + Y_f)^{-1}D\,\mathbf e_m\rangle$ with
  $D = Y_f - Y_0$ the difference of the fibre's and the medium's admittance operators,
  so the omission is second order in the index step; at weak guidance it is the local
  Fresnel estimate $-(n_{\rm cl} - n_{\rm co})^2/(8n_{\rm co})$ times the beam's
  cladding overlap with the mode, which the solution matches to a few per cent. Summed
  over an azimuthal order's guided modes the omission is a quadratic form of the
  incident field, so, as for the beam into a fibre, the bound is maximised over every
  field inside the widest admitted cone, of any shape and polarisation. At contrast
  0.05 it rises towards every cutoff, where a newborn mode's power moves into the
  cladding: the largest resolved value is $1.37\times10^{-3}$ (V = 3.905, just above
  the $HE_{12}$ cutoff), against $7.4\times10^{-4}$ at V = 1.5. A mode wholly in the
  cladding approaches the normal-incidence Fresnel ceiling $R(2-R)/(1-R) \approx
  \Delta^2/2 = 1.39\times10^{-3}$, R the core--cladding reflectance; the bound is
  twice that, $3.0\times10^{-3}$, which also covers the narrow bands just above the
  $HE_{1m}$ cutoffs that the reference cannot resolve. It bounds the energy launched
  into all guided modes together and is scaled by the squared weighted contrast over
  $0.05^2$: about $1.2\times10^{-4}$ at contrast 0.01. The beam is in the core's
  index: a beam in air meets a bulk step first (a separate planar interface), and the
  model refuses it through its source metric. The incident beam's own $\mathbf H$ closure is the
  angular item's, not the envelope's. A graded-index guide has no recorded envelope;
  its item has no value and is not counted.

A pair outside its envelope's range is refused, not extrapolated; bins of a covered pair
whose $\delta$, V or contrast leave the range carry an unbounded discrepancy, so their share of
the incident energy is added to the item in full.

### A Gaussian beam into a capillary

A beam is a field plane: {py:meth}`~solprop.coupling.plane.FieldPlane.from_pulse` builds
one from the package's {py:class}`~solprop.pulses.SpatialPulse`, and
{py:meth}`~solprop.coupling.plane.FieldPlane.from_analytic` samples any analytic field,
a callable of the node positions and the absolute angular frequency returning the
optics-convention spectrum $\tilde E(\mathbf r,\omega)$ in V s/m, with a declared
support and exterior; it records how the plane's energy changes when the window is
doubled and when the spacing is halved, and a coupling counts both as numerical error.

```python
import numpy as np

from solprop.grid import RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.coupling import FieldPlane, PlaneGeometry

capillary = Simulation.from_capillary(
    75e-6,
    1e-3,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15,
    fwhm=20e-15,
    energy=20e-6,
    modes=4,
    n_save=3,
    noise=None,
)
port = capillary.input_port()
beam = SpatialPulse(
    GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=10e-6), 37.5e-6
)
incident = FieldPlane.from_pulse(
    beam,
    port.grid,
    PlaneGeometry(0.0),
    space=RadialGrid(400e-6, 256),
    medium=SpatialMedium("Ar", 1.0),
)
coupled = couple(
    incident, port, model=CouplingModel("scalar_paraxial", model_error=0.05)
)
print(coupled.report)  # incident, launched, uncaptured, the budget
entered = capillary.with_input(coupled.launch)
```

At $w/a = 0.5$ the fundamental takes about 0.906 of the incident energy and the next
radial mode 0.092. The launch carries what the modes capture: nothing is rescaled to
the requested energy. A displaced, tilted or curved beam is a Cartesian plane
(`space=FreeGrid(...)`); a radial plane cannot carry them. The
[capillary coupling tutorial](../../auto_tutorials/free_space/plot_08_radial_capillary_coupling.rst)
couples the entrance field of a beam focused through 20 mm of argon, with and without
the Kerr effect, and follows both launches through a capillary.

### A change of capillary radius

A modal plane read with its simulation, `field_plane(run, source=simulation)`, carries
the run's scalar profiles; its coefficients are coupled into another capillary's port
by direct overlap, without a transverse grid:

```python
exit_plane = field_plane(run, source=simulation)  # 100 um, two modes
wider = Simulation.from_capillary(
    120e-6,
    1e-3,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15,
    fwhm=20e-15,
    energy=20e-6,
    modes=4,
    n_save=3,
    noise=None,
)
into_wider = couple(
    exit_plane,
    wider.input_port(),
    model=CouplingModel("scalar_paraxial", model_error=0.05),
)
print(into_wider.report.efficiency)
```

The same result comes from synthesising the exit field and projecting it:
{py:class}`~solprop.coupling.outcouple.Outcouple` writes the modal field onto a radial or
Cartesian grid under an explicitly declared forward interface model. The one model is
`"matched"`, an exit into a medium of the core's own index where the transverse field is
continuous; a change of index is a separate planar interface, applied once, never an
implied Fresnel factor.

```python
from solprop.coupling import Outcouple

synthesised = Outcouple(RadialGrid(600e-6, 512)).apply(exit_plane)
through = couple(
    synthesised.plane,
    wider.input_port(),
    model=CouplingModel("scalar_paraxial", model_error=0.05),
)
difference = np.abs(through.launch.samples - into_wider.launch.samples)
print(np.max(difference) / np.max(np.abs(into_wider.launch.samples)))
```

The tests hold the two routes together to the synthesis window's numerical error.

### A solid fibre overlap

A step-index fibre's mode-averaged run carries its waveguide's profile, taken in the
weak-guidance limit; two fibres of one material pair and different radii are coupled by
the same direct overlap. The pair is admitted within the recorded envelope's range;
the contrast screen applies too, and the envelope stands in for it at a budget whose
eighth is below the contrast. Small cores are not very paraxial: the LP01 field of a
2 µm core spreads over a tenth of a radian, so such a pair needs a budget of about 0.1.
A beam enters a fibre the same way, as a radial or Cartesian plane sampled in the
core index (a plane in another medium is a change of medium, which the source-metric
item refuses: a bulk entrance face is a separate planar interface), within its own
envelope above.

```python
from solprop.modes import StepIndexMode


def fibre(radius):
    mode = StepIndexMode(
        radius, core=1.45, cladding=1.45 * np.sqrt(0.99), profile_wavelength=1064e-9
    )
    return Simulation.from_gnlse(
        None,
        1e-3,
        fibre=mode,
        n2=2.6e-20,
        lambda0=1064e-9,
        wavelength_limits=(900e-9, 1300e-9),
        trange=400e-15,
        fwhm=100e-15,
        energy=1e-12,
        n_save=2,
    )


source_fibre, target_fibre = fibre(2.18e-6), fibre(2.57e-6)
fibre_run = source_fibre.run(progress="none")
spliced = couple(
    field_plane(fibre_run, source=source_fibre),
    target_fibre.input_port(),
    model=CouplingModel("scalar_paraxial", model_error=0.1),
)
print(spliced.report.efficiency)

fibre_port = target_fibre.input_port()
fibre_beam = FieldPlane.from_pulse(
    SpatialPulse(GaussianPulse(wavelength=1064e-9, fwhm=100e-15, energy=1e-12), 3e-6),
    fibre_port.grid,
    PlaneGeometry(0.0),
    space=RadialGrid(100e-6, 256),
    index=np.asarray(fibre_port.reference_index),
)
launched = couple(
    fibre_beam, fibre_port, model=CouplingModel("scalar_paraxial", model_error=0.1)
)
print(launched.report.efficiency)  # about 0.992: a 3 um waist nearly matches LP01
```

### A beam into a general modal guide

The vector model couples into a guide that solves its own E/H. Its entry point is
`Simulation.from_waveguide(provider=..., incident=SpatialPulse(...),
coupling=CouplingModel("vector_paraxial", model_error=...))`, which couples the beam
through `couple` and records the report as `configuration["launch"]`
([general modal propagation](../waveguides/general-modal.md)). A strongly guiding core
fails the contrast screen and is refused; the closure is qualified against the
provider's own H, and for a step-index guide the facet reflection the model omits is
counted through its recorded envelope (above).

## Transporting a plane between planes

{py:func}`~solprop.coupling.transport` carries a radial or Cartesian field
plane through an ordered sequence of linear optical elements and returns the plane at
the last element's output together with a
{py:class}`~solprop.coupling.transport.TransportReport`. A modal plane holds guide
coefficients, not a field on a transverse grid, and is refused: its exit field is
synthesised first by `Outcouple`. The plane needs its medium, so a saved plane is read
with `field_plane(run, source=simulation)`.

| element | what it does |
|---|---|
| {py:class}`~solprop.coupling.elements.FreeSpace` | homogeneous propagation $e^{ik_zL}$ with the exact $k_z=\sqrt{(\omega n/c)^2-\kappa^2}$, on the plane's own grid or onto another output plane |
| {py:class}`~solprop.coupling.elements.ThinLens` | the paraxial phase screen $e^{-ik(\omega)r^2/2f(\omega)}$, with a constant or chromatic focal length |
| {py:class}`~solprop.coupling.elements.Aperture` | the edge profile $W$ of {py:func}`~solprop.spatial.aperture_profile`, with the weighting of the in-run aperture statistics |
| {py:class}`~solprop.coupling.elements.TransmissionMask` | a complex field transmission $T(x, y, \omega)$, as a function or as samples |
| {py:class}`~solprop.coupling.elements.Alignment` | a lateral displacement and a tilt of a Cartesian field |
| {py:class}`~solprop.coupling.elements.PolarisationTransform` | a uniform Jones matrix in the plane's local axes |
| {py:class}`~solprop.coupling.elements.Mirror` | spectral p/s reflection and the change of frame |
| {py:class}`~solprop.coupling.elements.PlanarInterface` | the normal-incidence Fresnel interface between two lossless media |
| {py:class}`~solprop.coupling.elements.SpectralTransfer` | any {py:mod}`solprop.optics` transfer function, or a compressor's response, acting on every sample |

**Order.** The elements are applied exactly in the order given. An
{py:class}`~solprop.optics.Chain` or any other known transfer function may stand in the
sequence directly: it is one frequency-only block, its factors commute among
themselves, and it is never moved past focusing, clipping, propagation or a Jones map.
Every element reports its input and output energies in joules (the axial Poynting flux
of the medium, [medium flux](../free-space/flux.md)), the per-frequency spectra of both,
the declared loss, any intentional removal and the signed approximation error that
closes its ledger. The report of the sequence chains them: each element's input energy
is the previous one's output, losses add in joules, and broadband efficiencies are never
multiplied. A source plane's path certificate is carried as its own item,
`report.source_certificate`, because it bounds the producing run, not the transport.

**The plane's record follows the field.** Each `FreeSpace` removes a frame term from the
whole field so that the pulse stays in its window, and adds it to the plane's
reference, so `plane.absolute()` stays exact over any distance. Pass the source
simulation's frame, `transport(plane, elements, frame=simulation.frame)`, and a gap
removes the same affine term $R L$ the run would have removed: the transported plane
then has the same stored spectrum and reference as continuing the run. Without a frame a
gap removes only the group delay $\beta_1(\omega_0)L$ of its medium, and the report says
so. A spectral element that drops its own constant phase or group delay (a grating or
prism pair, a chirped mirror's data) leaves that term of the reference uncalibrated. An
interface changes the plane's medium, a mirror turns its axes, and the power metric is
resolved again in the output medium and grid. Every operation is appended to
`plane.provenance.history`. `geometry=` declares the output plane in the receiving
system's coordinates; its axes must be the ones the field arrives with.

**Free space and its output plane.** Without `output=`, the field stays on its grid,
propagated exactly in its periodic (Cartesian) or reflecting (radial) representation.
A column at angle θ moves its rays by $L\tan\theta$; content whose rays would leave the
window wraps back, which is the same as sampling the propagator's phase below its
Nyquist rate, and is refused beyond `sampling_tolerance`. `beyond_window="remove"`
declares the removal of that angular band instead, and the removed energy is reported.
Significant nonpropagating content is refused too, or removed explicitly with
`band_limit=`. The ray test does not see a field that has already spread into the
window's outer band (the declared collar, or the outer tenth of each half-width), where
the representation wraps or reflects it; a propagated field whose share there exceeds
`edge_tolerance` (default $10^{-4}$) is refused too, and the refusal suggests a larger
window, `beyond_window="remove"` when the removal is not yet declared, or a Fresnel
output plane. The share is reported as `edge_share` either way. An output plane of
another radius or sampling is the caller's explicit choice of method; nothing switches
between them automatically:

- `output_method="angular"` sums the propagated spectrum at the output nodes. It is
  exact and nonparaxial but lives inside the source window: a field that reaches the
  source window's edge is refused, and the refusal suggests the Fresnel route.
- `output_method="fresnel"` evaluates the paraxial Fresnel integral of the source
  samples at the output nodes. It reaches any observation window, including the far
  field; it reports its paraxial phase bound $k(\rho_{\max}+r_{\rm occ})^4/8L^3$ and
  refuses a kernel that would alias on the source nodes.

Both build their kernels in blocks; no transverse-by-transverse-by-frequency array is
formed.

**The thin-screen certificate.** A lens, an aperture, a mask and a tilt are local
electric-field screens at each frequency. A screen is exact in the local reference
measure $U_0$, which weighs every plane wave by $\operatorname{Re} n\,\varepsilon_0 c$;
the physical axial energy $U_Y$ weighs it by a further $\cos\theta$. With the required
`model_error` $\epsilon_m$ and the cone $1-\cos\theta_*=\epsilon_m/8$, the defect
$d(E)=(1-\cos\theta_*)U_{0,\rm cone}+U_{0,\rm tail}$ bounds $U_0-U_Y$ for the input
and for the output field, and the screen is admitted only when
$[d(E_{\rm in})+d(E_{\rm out})]/U_Y(E_{\rm in})$, plus the transform's round-trip and
edge error, is at most $\epsilon_m$ (SEP 31 §4.6). The certificate,
`report.elements[i].certificate`, applies to that one field. A screen whose phase changes
by π or more between neighbouring nodes where the field is, or one with gain, is
refused. An aperture's `weighting` has the meaning it has for the in-run aperture
statistics: `"amplitude"` (the default) applies $W$ to the field, so the power goes as
$W^2$; `"quadrature"` takes $W$ as a power weight and applies $\sqrt W$. The
intensity-defined aperture of SEP 31 is the `"quadrature"` case; a hard aperture is the
same either way.

**Mirrors and Jones frames.** A mirror takes its incoming and outgoing propagation
directions and an s axis perpendicular to both (at normal incidence the s axis
resolves the degeneracy). Each beam has the right-handed frame
$(\hat p, \hat s, \hat k)$ with $\hat p = \hat s\times\hat k$; the field leaves as
$(r_pE_p, r_sE_s)$ with no conjugation, and the outgoing plane's axes keep the p/s
components of the incoming ones, so the reflection appears as a mirror image of the
transverse pattern along p. In this convention a perfect conductor has $r_p=+1$,
$r_s=-1$ at any incidence; coefficients calibrated in another convention are converted
first. Two reflections restore the field and its frame.

**The Fresnel interface.** {py:class}`~solprop.coupling.elements.PlanarInterface`
applies $t_E=2n_1/(n_1+n_2)$ per frequency, with power transmission
$T=(n_2/n_1)|t_E|^2$ and the reflected $R=|r_E|^2$ reported as loss; $R+T=1$. It is
the normal-incidence model: oblique columns' departure from $T$ is reported as
approximation error. The output plane is in the second medium, so applying the same
factor twice -- in an element and again in a launch -- is refused.

**Spectral optics.** {py:class}`~solprop.coupling.elements.SpectralTransfer` adapts
`Material`, `ChirpedMirror`, `GratingPair`, `PrismPair`, `TaylorPhase` and a `Chain` of
them, keeping each factor's affine-phase record (`report.calibration`). Losses are kept
per frequency; bins a transfer function sets exactly to zero are reported as removed. A
phase-only model asked for a lossy prediction needs an explicit `efficiency=`, and an
efficiency on a lossy model is refused because it would count the loss twice. A
negative material length or an inverse mirror phase is flagged as a reconstruction,
not a passive element. A callable that is not a known optics element states its
calibration with
{py:class}`~solprop.coupling.elements.SpectralCalibration`.

**Example: a Gaussian focused by a lens.**

```python
from solprop.coupling import FieldPlane, FreeSpace, PlaneGeometry, ThinLens, transport
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium

grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
air = SpatialMedium()
beam = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-3)
plane = FieldPlane.from_pulse(
    beam, grid, PlaneGeometry(0.0), space=RadialGrid(4e-3, 256), medium=air
)
small = RadialGrid(0.5e-3, 128)  # the focal plane, sampled finely
focus, report = transport(
    plane,
    [
        ThinLens(0.5, model_error=1e-3),
        FreeSpace(0.5, air, output=small, output_method="fresnel"),
    ],
)
print(focus.geometry.z, focus.energy(), report.elements[0].certificate.ratio)
```

The second-moment radius at the focus is the Fourier-plane waist
$\lambda f/\pi w_0 \approx 127$ µm to about $10^{-6}$ here (the tests hold the lens
and the free-space output to their analytic references); the focal plane carries the
launched microjoule less $10^{-6}$ of it. That is the lens's own axial-flux defect: the
converging plane waves cross the plane obliquely, carrying $\cos\theta$ of their local
measure, and the report books it as the lens's approximation error, inside its
certificate (ratio $2.5\times10^{-4}$ against the budget of $10^{-3}$).

## Composing simulations

The blocks above compose into the calculations
[SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md) set out to support. Each is an
ordinary chain of runs and planes; nothing schedules them, and each run keeps its own
step control. The test suite runs every one of them as a short integrated case
(`tests/coupling/test_workflows.py`), asserting its energy ledger.

| calculation | composition |
|---|---|
| a capillary's output into another capillary | `field_plane(run, source=simulation)`, then `couple` into the second port by direct overlap, or `Outcouple` → `transport([FreeSpace(...)])` → `couple` across a gap |
| a nonlinear beam into a capillary | a radial (or Cartesian) gas gap with the ordinary Kerr, plasma and harmonic responses; its last plane read with `field_plane`, coupled with `scalar_paraxial` into a capillary on the same lattice |
| a solid-core fibre into another fibre | the exit plane of a step-index run coupled into another step-index run, within the recorded envelope ([a solid fibre overlap](#a-solid-fibre-overlap)) |
| a Gaussian beam into a guide | `FieldPlane.from_pulse` of a `SpatialPulse`, or `from_waveguide(incident=...)` for a general modal guide |
| an arbitrary analytic field into a guide | `FieldPlane.from_analytic`, a callable of the nodes and the absolute frequency; nonseparable fields (a waist that changes with frequency) are fine |
| an arbitrary Cartesian beam into a guide | a Cartesian plane, aligned with `Alignment` (displacement and tilt) or sampled directly; a radial plane refuses anything that would break its symmetry |
| a capillary's output at a distant plane | `Outcouple` onto a spatial grid in the exit medium, then `FreeSpace(L, medium, output=..., output_method="fresnel")` to any observation plane, the far field included |
| SPM, a compressor and another capillary | the first capillary's exit, `Outcouple`, `SpectralTransfer` of an optics element or `Chain`, lenses, apertures and free space, then `couple` into the second capillary |

Three things to keep in mind when chaining:

- **Lattices.** `couple` needs the plane on the port's temporal lattice. Build the
  receiving run on the source's grid (`grid=` of the constructors), or move the plane
  with `regrid` first. A gap whose output carries a harmonic needs a receiving band that
  holds it: the active band is checked, and light outside it is refused unless cropped
  explicitly.
- **Planes in the receiver's coordinates.** A transported plane moves along its normal,
  so its origin is where the optics put it. Pass `geometry=port.geometry` to `transport`
  to declare the last plane in the receiving run's own coordinates; its axes must be the
  ones the field arrives with, or the coupling is refused as oriented differently.
- **Budgets.** Every approximate step has its own budget: a coupling's `model_error`, a
  thin screen's `model_error`, a free-space gap's `sampling_tolerance` (or
  `beyond_window="remove"`, which books the removed angular band). They are reported
  separately and never multiplied into one efficiency.

For example, a capillary's output relayed 1:1 by two lenses, compressed on the way, into
an identical capillary:

```python
from solprop.coupling import Aperture, SpectralTransfer
from solprop.optics import TaylorPhase

argon = SpatialMedium("Ar", 1.0)
receiving = Simulation.from_capillary(
    100e-6,
    1e-3,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15,
    fwhm=20e-15,
    energy=20e-6,
    modes=2,
    n_save=3,
    noise=None,
)
port = receiving.input_port()
exit_field = Outcouple(RadialGrid(1.2e-3, 512), medium=argon).apply(
    field_plane(run, source=simulation)
)
gap = FreeSpace(0.05, argon, beyond_window="remove")
relayed, ledger = transport(
    exit_field.plane,
    [
        SpectralTransfer(TaylorPhase((0.0, 0.0, -20e-30), 800e-9)),
        gap,
        ThinLens(0.05, model_error=0.05),
        gap,
        Aperture(600e-6, model_error=0.05, edge=20e-6, kind="tanh"),
        gap,
        ThinLens(0.05, model_error=0.05),
        gap,
    ],
    frame=simulation.frame,
    geometry=port.geometry,
)
second = couple(relayed, port, model=CouplingModel("scalar_paraxial", model_error=0.05))
print(ledger.specified_loss, ledger.removed, second.report.efficiency)
```

Field planes, coupling models and their budgets, and transport between planes were
proposed and carried out by [SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md).
