# Coupling simulations

A simulated pulse is often the input to another optical component: a second capillary,
a fibre, a gas cell. Carrying only its spectrum or duration loses its wavefront, its
modal coherence and the frequency dependence of its transverse profile; carrying a bare
array leaves its basis, units, frame and energy meaning implicit.
{py:mod}`solprop.coupling` connects simulations through a **coherent field at a
reference plane** instead ([SEP 31](../../dev/seps/sep-0031-simulation-coupling.md)).
This page covers the first part of that design: reading a saved plane as a field
plane, moving it between temporal grids, and relaunching it into a simulation of the
same basis. Coupling between different guides, incident beams and the optical
transport between planes follow in later releases of this chapter.

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
native metric at the entrance and the models it admits. The identity model,
{py:func}`~solprop.coupling.port.same_basis`, launches a plane into a port of the **same**
basis on the same lattice, at an identically oriented plane in the same medium: the
coefficients are copied as they stand, and the plane's reference becomes the launched
run's origin. The launched run therefore continues the source's absolute time and
carrier phase without moving the pulse in its window.

```python
from solprop.coupling import same_basis

second = Simulation.from_capillary(
    100e-6, 1e-3, "Ar", 1.0, lambda0=800e-9, wavelength_limits=(400e-9, 1.6e-6),
    trange=200e-15, fwhm=20e-15, energy=20e-6, modes=2, n_save=3, noise=None,
)
launch = same_basis(field_plane(run), second.input_port())
continued = second.with_input(launch).run(progress="none")
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

{py:meth}`~solprop.driver.simulation.Simulation.with_input` refuses a launch computed for any other
port -- another basis, lattice, entrance, orientation or medium -- whatever the shapes of
the arrays. It records the coupling in `configuration["coupling"]` (the model, the port,
the inherited origin and the source plane's provenance, so a run launched from a plane
names its parent), and it draws no new noise: the launch carries the realised field of
its source, and a realisation is added only when one is given explicitly. The receiving
run keeps its ionisation tables and their ranges, as every derived run does. A different
basis, lattice or medium is a coupling or an interface, not a relaunch, and is refused by
name.
