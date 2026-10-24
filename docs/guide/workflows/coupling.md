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

## Transporting a plane between planes

{py:func}`~solprop.coupling.transport.transport` carries a radial or Cartesian field
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
`band_limit=`. An output plane of another radius or sampling is the caller's explicit
choice of method; nothing switches between them automatically:

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
small = RadialGrid(0.5e-3, 128)   # the focal plane, sampled finely
focus, report = transport(
    plane,
    [ThinLens(0.5, model_error=1e-3),
     FreeSpace(0.5, air, output=small, output_method="fresnel")],
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
