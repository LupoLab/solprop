# Coupling simulations

A simulated pulse is often the input to another optical component: a second capillary,
a fibre, a gas cell. Carrying only its spectrum or duration loses its wavefront, its
modal coherence and the frequency dependence of its transverse profile; carrying a bare
array leaves its basis, units, frame and energy meaning implicit.
{py:mod}`solprop.coupling` connects simulations through a **coherent field at a
reference plane** instead ([SEP 31](../../dev/seps/sep-0031-simulation-coupling.md)).
This page covers reading a saved plane as a field plane, moving it between temporal
grids, and coupling it into another simulation: a relaunch into the same basis, a
beam or a guide's exit into a capillary or another fibre, and a guide's exit field
synthesised on a spatial grid.

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
    100e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15, fwhm=20e-15, energy=20e-6, modes=2, n_save=3, noise=None,
)
run = simulation.run(progress="none")
plane = field_plane(run)              # the exit plane
middle = field_plane(run, z=0.5e-3)   # a stored plane by position
print(plane, plane.reference.delay, plane.energy())
```

A general modal file records the identities of its E/H reference and not the
reference's arrays, so its physical fields (`plane.nodal`) and a dense basis Gram are
read with the simulation that ran it, `field_plane(run, source=simulation)`; the
identities are checked first, as {py:mod}`solprop.processing.modal` checks them. A
spatial plane's measure is rebuilt from the file's plane measure; with `source=` it
also carries the medium. A hybrid planar run's planes are refused by name.

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
print(moved.energy() / plane.energy())   # no renormalisation, and none needed here
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
    100e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15, fwhm=20e-15, energy=20e-6, modes=2, n_save=3, noise=None,
)
relaunch = couple(field_plane(run), second.input_port(), model=CouplingModel("same_basis"))
continued = second.with_input(relaunch.launch).run(progress="none")
print(field_plane(continued).reference.delay)   # the whole 2 mm path's
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
kept apart ([SEP 31](../../dev/seps/sep-0031-simulation-coupling.md), section 4.3): the
Hermitian E/H flux pairing $F$ of a modal basis, the Gram $G = S^\dagger MS$ of a scalar
least-squares fit in a common measure $M$, and a representation's native power metric
$Q$ -- the identity for ordinary true-power coefficients, a dense Gram for a declared
general basis, the axial admittance for a spatial plane. They coincide only where a
model establishes it, and the uncaptured energy is a physical, nonnegative loss only
in a common positive metric. The models are:

| model | what it admits | what it computes |
|---|---|---|
| `same_basis` | an identical basis, a declared basis change, a general modal plane into its own guide | the coordinates exactly; no budget |
| `scalar_paraxial` | a radial or Cartesian beam (Gaussian, analytic or sampled) into a capillary; a capillary's exit into another capillary of any radius and mode set; one weakly guiding step-index fibre into another | the fit $\mathbf a = G^{-1}S^\dagger M\mathbf f$ in $M_0 = Y_0\,dA/2$, $Y_0 = n_{\rm ref}\varepsilon_0c$, then $b = \sqrt{\operatorname{Re}n_{\rm eff}}\,a$ |
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
| `angular` | $[(1 - \cos\theta_*)U_{\rm cone} + U_{\rm tail}]/U$ of the incident field, and of the worst unit field in the retained target subspace | the total |
| `contrast` | the weak-guidance screen $\Delta = \max|n^2 - n_{\rm ref}^2|/(2n_{\rm ref}^2)$ of a solid guide; not applied to a capillary wall; a recorded pair envelope may replace it | $\epsilon_m/8$ |
| `target metric` | $G$ (or $F$) against the target's native $Q$, whitened | $\epsilon_m/8$ |
| `source metric` | the common measure $M_0$ against the source's native metric | $\epsilon_m/8$ |
| `closure` | $2\delta + \delta^2$ of the H-closure's field error $\delta$ (vector model) | $\epsilon_m/8$ |
| `passivity` | the whitened certificate $C^\dagger Q_BC \preceq Q_A$: an apparent gain within the model's own defects is reported here, never rescaled or clipped; a larger one refuses | the total |
| `envelope` | the recorded provider-pair envelope's bound | the total |
| `numerical` | the source's content at its spectral edge, an analytic field's domain and resolution checks, the overlap quadrature | the total |

The counted items must sum to at most $\epsilon_m$. The
{py:class}`~solprop.coupling.coupler.CouplingReport` carries the incident energy over the
whole plane in $M_0$, the launched energy in the target's native metric, the captured
and uncaptured energy where a common positive metric exists (`None`, with the reason,
for the vector model), every item with its scope -- this field, the whole map, or the
model -- the source plane's exterior, and the source's SEP 34 path certificate,
separately and with its scope. A zero field is a valid zero launch whose `efficiency` is
`None`. The recorded envelopes ({py:mod}`solprop.coupling.envelopes`) are measured by the
test suite against independent references: the capillary's against the overlap of its
own vector Marcatili fields (a difference of 1.8e-14 of the incident energy over waists
0.3a--1.0a, displacements to 0.2a, tilts to 1 mrad and radius ratios 0.5--2), the
step-index pair's against the full-vector flux overlap of the exact $HE_{11}$ fields
(3.0e-4 over contrasts to 0.01, V from 1.5 to 2.4 and radius ratios 0.7--1.43). A pair
outside its envelope's range is refused, not extrapolated.

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
    75e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15, fwhm=20e-15, energy=20e-6, modes=4, n_save=3, noise=None,
)
port = capillary.input_port()
beam = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=10e-6), 37.5e-6)
incident = FieldPlane.from_pulse(
    beam, port.grid, PlaneGeometry(0.0), space=RadialGrid(400e-6, 256),
    medium=SpatialMedium("Ar", 1.0),
)
coupled = couple(incident, port, model=CouplingModel("scalar_paraxial", model_error=0.05))
print(coupled.report)                 # incident, launched, uncaptured, the budget
entered = capillary.with_input(coupled.launch)
```

At $w/a = 0.5$ the fundamental takes about 0.906 of the incident energy and the next
radial mode 0.092. The launch carries what the modes capture: nothing is rescaled to
the requested energy. A displaced, tilted or curved beam is a Cartesian plane
(`space=FreeGrid(...)`); a radial plane cannot carry them.

### A change of capillary radius

A modal plane read with its simulation, `field_plane(run, source=simulation)`, carries
the run's scalar profiles; its coefficients are coupled into another capillary's port
by direct overlap, without a transverse grid:

```python
exit_plane = field_plane(run, source=simulation)        # 100 um, two modes
wider = Simulation.from_capillary(
    120e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15, fwhm=20e-15, energy=20e-6, modes=4, n_save=3, noise=None,
)
into_wider = couple(exit_plane, wider.input_port(),
                    model=CouplingModel("scalar_paraxial", model_error=0.05))
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
through = couple(synthesised.plane, wider.input_port(),
                 model=CouplingModel("scalar_paraxial", model_error=0.05))
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

```python
from solprop.modes import StepIndexMode

def fibre(radius):
    mode = StepIndexMode(radius, core=1.45, cladding=1.45 * np.sqrt(0.99),
                         profile_wavelength=1064e-9)
    return Simulation.from_gnlse(
        None, 1e-3, fibre=mode, n2=2.6e-20, lambda0=1064e-9,
        wavelength_limits=(900e-9, 1300e-9), trange=400e-15, fwhm=100e-15,
        energy=1e-12, n_save=2,
    )

source_fibre, target_fibre = fibre(2.18e-6), fibre(2.57e-6)
fibre_run = source_fibre.run(progress="none")
spliced = couple(field_plane(fibre_run, source=source_fibre), target_fibre.input_port(),
                 model=CouplingModel("scalar_paraxial", model_error=0.1))
print(spliced.report.efficiency)
```

### A beam into a general modal guide

The vector model couples into a guide that solves its own E/H. Its entry point is
`Simulation.from_waveguide(provider=..., incident=SpatialPulse(...),
coupling=CouplingModel("vector_paraxial", model_error=...))`, which couples the beam
through `couple` and records the report as `configuration["launch"]`
([general modal propagation](../waveguides/general-modal.md)). A strongly guiding core
fails the contrast screen and is refused; the closure is qualified against the
provider's own H.
