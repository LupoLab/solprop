# Energy and flux in a medium

This page defines what a free-space run's energy, power, fluence and intensity mean.
Each is the axial Poynting flux of the scalar forward field in the medium at the
measured plane, so a launch energy is what an energy meter in that medium would read.
The page also derives how a radial run carries the field through a gas whose pressure
varies along z, and gives the certificate that decides whether such a path is admitted.
The design is [SEP 34](../../dev/seps/sep-0034-medium-flux.md). The operator, the units
and the transverse grids are described on [the free-space model](model.md) page.

## What energy means

Use SI units and the optics convention $\exp[i(kz-\omega t)]$
([conventions](../../dev/conventions.md)). For the analytic fields $\mathbf E_a$ and
$\mathbf H_a$, the axial flux, the fluence and the energy crossing a plane normal to z are

$$
S_z=\tfrac12\operatorname{Re}[\mathbf E_a\times\mathbf H_a^*]\cdot\hat{\mathbf z},\qquad
F=\int S_z\,dt,\qquad U=\int F\,dA .
$$

For a transparent, nonmagnetic paraxial plane wave,
$\mathbf H_a=n_0\varepsilon_0c\,\hat{\mathbf z}\times\mathbf E_a$, so
$S_z=n_0\varepsilon_0c|E_a|^2/2$. The **phase** index $n_0$ appears here, not the group
index. In the narrowband transparent limit the cycle-averaged energy density is
$u=\varepsilon_0n_0n_g|E_a|^2/2$, and multiplying it by the group velocity $c/n_g$ gives
the same flux. The group index governs how fast the energy moves, not how much crosses
the plane. (This density relation does not hold for strongly absorbing or arbitrarily
broadband matter, and nothing here uses it.)

Two consequences follow. At a fixed requested energy, a medium of index $n_0$ carries a
field $\sqrt{n_0}$ smaller than vacuum would. A fixed field carries $n_0$ times the energy
it would carry in vacuum. A supplied field array in V/m is never rescaled to keep an
energy unchanged: it is the physical field, and its energy is measured in the medium.
An energy measured in air before a window is not the energy inside the window. The
launch does not model the window's Fresnel transmission. Multiply by the transmission
yourself and give the in-medium energy.

## Axial admittance and local flux

For each transverse column of wave number $k_\perp$ and each frequency, in a homogeneous
isotropic medium,

$$
k=\frac{n(\omega)\omega}{c},\qquad k_z=\sqrt{k^2-k_\perp^2},\qquad
\mathcal Y_c=\frac{\varepsilon_0c^2k_z}{\omega},\qquad Y_z=\operatorname{Re}\mathcal Y_c .
$$

$\mathcal Y_c$ is the scalar equation's axial admittance, on the forward branch the run
propagates. It follows from $\mathbf k\times\mathbf E=\omega\mu_0\mathbf H$ for a
transverse-electric plane wave. The model applies it to each Cartesian component
independently, as the scalar UPPE does, so it is not the full TE/TM Maxwell metric at
high numerical aperture. Paraxially $\mathcal Y_c\to n\varepsilon_0c$, and in vacuum
$Y_z/(\varepsilon_0c)$ is the obliquity $\cos\theta$.

**Total energy needs only the real part.** Integrated over the whole plane and over
time, the transforms diagonalise and

$$
U=\tfrac12\sum_{\text{columns},\,\omega}Y_z\,|E|^2
$$

with the grid's Parseval weights. Energy, `beam_spectrum`, the angular-port spectra and
the noise draw use these real weights.

**Local flux needs the complex admittance.** A local measurement reconstructs the
magnetic closure $\mathbf G_a$ by applying the **complex** $\mathcal Y_c$ to each column
before the inverse spatial and temporal transforms, and forms

$$
S_z=\tfrac12\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*).
$$

In an absorbing medium the cross terms between two frequencies or two angles contain
$\operatorname{Im}\mathcal Y_c$. Replacing $\mathcal Y_c$ with $Y_z$ before the
reconstruction keeps the total energy but changes the local flux. For transparent
homogeneous propagating columns the two agree. `spatial_fluence`, `spatial_intensity`,
`on_axis_intensity`, `on_axis_spectrum`, `peak_intensity` and the planar fluence and
intensity are local and use $\mathcal Y_c$.

**The local flux is signed.** Two oblique waves in an absorbing medium, or a broadband
field, can give a negative $S_z$ at some point and time. The value is not clipped and is
not replaced by $|E_a|^2$. A width needs a nonnegative profile, so `beam_radius`,
`fwhm_r` and their statistics refuse a fluence that is negative beyond its rounding
budget, with a {py:class}`~solprop.exceptions.GridError` naming the node. This refusal is
distinct from the existing undefined widths of a hollow or uncrossed profile, which are
NaN.

**The electric field is a separate, named diagnostic.**
{py:func}`~solprop.spatial.peak_field` returns $\max\sqrt{\sum_p|E_{a,p}|^2}$ in V/m. This
is the quantity that drives ionisation: rate models take the physical field in V/m,
never an intensity derived from a signed flux, and a run checks its ionisation range
against the input's peak field. Local flux uses the declared forward linear closure; no
nonlinear correction to the impedance is included.

For a uniformly absorbing medium a column evolves as $e^{ik_zL}$ and its energy as
$e^{-2\operatorname{Im}k_zL}$. This physical attenuation is accounted separately from the
numerical absorbers: the spatial collar's diagnostic remains an electric-field-norm
measure of the field near the edge, labelled as such, not a fraction of the physical
fluence ([spatial collar and statistics](model.md#spatial-collar-and-statistics)).

## The discrete measure, endpoints and noise

Let $\hat E_{jpk}$ be the raw temporal DFT at transverse spectral column $j$,
polarisation $p$ and frequency bin $k$, and $v_j$ the column's transverse Parseval
weight. Then

$$
U=\frac{\Delta t}{2N_t}\sum_{j,p,k}v_j\gamma_kY_{z,jk}|\hat E_{jpk}|^2,
$$

with $\gamma_k=1$ on an envelope grid. On a real-field grid $\gamma_k=4$ on the interior
positive frequencies and 1 at DC and Nyquist, the analytic-signal convention of
{py:func}`~solprop.fourier.energy_omega`. The time and space reference is
$\tfrac12\int\operatorname{Re}(\mathbf E_a\cdot\mathbf G_a^*)$. For a constant real
admittance it is $Y_z\int|E_a|^2/2$, which is not the same as an unqualified $\int E^2$
when the endpoint bins are occupied.

Only active, propagating columns at positive frequency are forward power ports. Every
other bin has zero admittance: DC, inactive frequencies, and nonpropagating (cutoff or
evanescent) columns. A launch is hard-masked to zero in the inactive band and at DC, as
everywhere. A beam that puts more than rounding-level content into nonpropagating
columns is refused, since no energy can be assigned to that content: widen the waist or
refine the transverse grid. The numerical evanescent damping of the run does not make an
evanescent column a physical power port. A medium with no propagating column, such as a
metal-like index, has no forward flux to normalise and is refused.

The one-photon-per-bin noise model ([shot noise](../media/noise.md)) is normalised in
the same measure. Each admitted transverse column, polarisation and frequency bin carries
$\hbar\omega$ at the entrance plane, in the entrance medium, with the launch's transverse
weights. A bin with no admittance carries no photon. The random stream is drawn as
before, so seeds keep their meaning.

## Using the measure

{py:class}`~solprop.spatial.SpatialFlux` binds the temporal grid, the transverse grid,
the medium and one plane, $z$ and (at a declared jump) its side, and resolves the plane's
admittance once. Every spatial accessor takes the field first and the plane's measure
second:

| entry point | returns |
|---|---|
| `SpatialFlux.resolve(grid, space, medium, z=0.0, side=None)` | the measure of the plane at `z` in `medium` |
| `spatial_energy(field, flux)`, `planar_mode_energy(field, flux)` | energy in J (J/m on a line or a planar guide), from $Y_z$ |
| `spatial_power(field, flux)` | axial power in W in time; its time integral is the energy |
| `spatial_fluence(field, flux)`, `planar_fluence(field, flux)` | signed fluence in J/m² at the nodes |
| `spatial_intensity(field, flux)`, `planar_intensity(field, flux)` | signed local flux in W/m² at the nodes and times |
| `on_axis_intensity(field, flux)`, `on_axis_spectrum(field, flux)` | the axial flux and its spectral density on the axis |
| `beam_radius(field, flux)`, `fwhm_r(field, flux)` | widths of the signed fluence; refused if it is negative |
| `peak_field(grid, space, field)` | the largest analytic field in V/m, a named field diagnostic |

The same record serves the launch and the noise:
`SpatialPulse.build(grid, space, medium=..., z=..., side=...)` normalises the requested
energy or peak power at that plane, and
{py:func}`~solprop.noise.spatial_noise` takes a resolved `SpatialFlux`.
{py:func}`solprop.stats.collect` takes `flux=` for a saved plane, and in-run statistics use the
measure of the plane at which they are recorded. A saved run records the measure
(`flux_model` and `flux_plan` in its spatial metadata), and processing resolves each
saved plane at its own position. A plane measured at a different position, or in a
different medium, needs a new record.

The paraxial critical power, {py:func}`~solprop.spatial.critical_power`, takes the
carrier index as `index=`: it measures the same physical power, $n_0$ times the
vacuum-measure value at fixed $\chi^{(3)}$ ([critical power](model.md#critical-power-and-self-focusing)).

## Radial gas paths: forward amplitude transport

On the radial grid a gas pressure may vary along z. The medium is uniform across the
beam, so each spectral column obeys, in the transparent case, the scalar equation

$$
E''+q(z)^2E=0,\qquad q(z)=k_z(z)>0 .
$$

Write $E=A(z)e^{i\int q\,dz}$. Substitution gives $A''+2iqA'+iq'A=0$, and dropping $A''$
in the slow-variation expansion gives $A'/A=-q'/(2q)=-\tfrac12\partial_z\ln Y_z$. Hence

$$
\partial_zE=\left(ik_z-\tfrac12\partial_z\ln Y_z\right)E,\qquad
E(z_1)=\sqrt{\frac{Y_z(z_0)}{Y_z(z_1)}}\;e^{i\int_{z_0}^{z_1}k_z\,dz}\,E(z_0).
$$

So $Y_z|E|^2$, each column's contribution to the energy, does not change along the path.
Paraxially, the amplitude ratio is $\sqrt{n(z_0)/n(z_1)}$. A field entering denser gas
loses amplitude, and regains it when the gas thins again. Without this term, a fixed
field amplitude would carry $n(z)$ times its entrance energy.

The run keeps the physical field $E$ as its state, because every nonlinear drive and
source is written in terms of $E$. The amplitude term is added to the cumulative linear
operator as a real endpoint term, $-\tfrac12\ln[Y_z(z)/Y_z(z_0)]$, evaluated from the
endpoint admittances rather than by differentiating a table. It changes no phase, carrier
reference or delay, and it is exactly zero in a uniform medium. The nonlinear source
prefactors keep the local $\operatorname{Re}k_z$. The derivative of the energy weight
then cancels the amplitude term in the energy balance, which leaves only material loss,
polarisation work and declared numerical absorption.

The retained field is the forward component of the exact directional decomposition, with
$\mathbf G_+=\mathcal Y_c\mathbf E_+$ locally. It is not a total field from a
differentiated WKB ansatz: the backward wave and its standing-wave interference are
dropped, and local traces describe the forward component only. The certificate below
bounds what is dropped. The same transport was already used for modal amplitudes along a
guide ([power and intensity](../results/power-and-intensity.md#true-power-amplitudes)).

## The adiabatic certificate

Dropping the backward wave is an approximation, and its error is bounded. For a column
whose $q$ is positive, real and of bounded variation on the run path $[z_0,z_1]$, and
constant outside it, let

$$
\Lambda=\tfrac12\operatorname{TV}_{[z_0,z_1]}(\ln q)
=\tfrac12\int_{\text{smooth}}\left|\frac{q'}{q}\right|dz
+\tfrac12\sum_{\text{jumps}}\left|\ln\frac{q_+}{q_-}\right| .
$$

The exact scalar scattering problem, with no wave incident from beyond $z_1$, then
satisfies ([Visser 1999](https://doi.org/10.1103/PhysRevA.59.427), equations 80–82)

$$
R\le R_*=\tanh^2\Lambda,\qquad
\left|\frac{a_{\text{exact}}}{a_{\text{forward}}}-1\right|\le
\delta_*=\operatorname{expm1}(\Lambda\tanh\Lambda),
$$

for the reflected power fraction $R$ and the relative complex error of the retained
forward amplitude. The derivation writes
$E=q^{-1/2}(ae^{i\theta}+be^{-i\theta})$, with $\theta'=q$. The equations
$a'=(q'/2q)\,b\,e^{-2i\theta}$ and $b'=(q'/2q)\,a\,e^{2i\theta}$ conserve
$|a|^2-|b|^2$. A jump enters through the same hyperbolic transfer, with parameter
$\tfrac12\ln(q_+/q_-)$. The full derivation is SEP 34 §4.5.

For a monotone path, $\Lambda=\tfrac12|\ln(q_1/q_0)|$ and
$R_*=[(q_1-q_0)/(q_1+q_0)]^2$. An abrupt step **attains** this bound: it is the Fresnel
reflectance of the step. A smooth profile with the same endpoints reflects far less, and
the bound is conservative for it. A path that rises and falls counts both halves. Many
small oscillations add up, so a nonmonotone path is bounded by its total variation, not
by its endpoint contrast. Near cutoff, $q$ varies much more than the bulk index, and even
a gas path can fail there.

**Admission.** {py:class}`~solprop.adiabatic.AdiabaticAccuracy` holds two budgets:
`field_error` bounds $\delta_*$ and `reflected_energy` bounds $R_*$, both $10^{-4}$ by
default. They are model budgets, not physical constants or integrator tolerances; tighten
them with `SimulationSettings(adiabatic_accuracy=AdiabaticAccuracy(...))`. At translation
every propagating column of the active band must meet both budgets over the whole run
path. The budget is not reset per step, and a column is not excused because the input is
weak there, since nonlinear evolution can fill it. Otherwise the run is refused with a
{py:class}`~solprop.exceptions.MaterialError` naming the worst column, its frequency and
its $\Lambda$. The refusals are:

- a column whose $\Lambda$ exceeds either budget;
- a column that is propagating at one point of the path and not at another (it crosses
  $q=0$);
- a varying path in a medium with any loss: $\operatorname{Im}k_z=\operatorname{Im}k^2/(2\operatorname{Re}k_z)$
  then varies with $\operatorname{Re}k_z$ even where the absorbing component is uniform.
  A uniformly absorbing medium is supported, but a varying loss needs a complex-index
  bound that is not yet provided;
- a custom profile ({py:class}`~solprop.profiles.FromCallable`) that does not declare
  where it may turn (`turning=`), its kinks and its jumps, because sampling a callable
  bounds nothing between the samples.

{py:func}`~solprop.adiabatic.certify_radial_path` evaluates the certificate directly and
returns an {py:class}`~solprop.adiabatic.AdiabaticAdmission`. The admission records the
largest $\Lambda$, both bounds, the worst column, the support and whether every piece was
certified monotone, in which case $\Lambda$ is attained rather than bounded. A run
stores it in its plane measure, with the tolerance to which its operator tables resolve
the represented path as a separate interpolation envelope. The
enclosure does not sample. Every material solprop reads has a wave-number square affine
in its number densities. On pieces where it is monotone, the variation is the endpoint
log ratio. Elsewhere it is bounded from the densities' variation and the smallest $q^2$
on the piece, with the arithmetic error included. The certificate is of the *represented*
path, the function the profile evaluates. How well that represents a physical gas cell is
a separate question.

**Scope.** The bound certifies the forward part of the *linear background* scattering
problem. The omitted backward amplitude can be as large as $\sqrt{R_*}$, so neither the
total local field nor a standing-wave maximum has the $\delta_*$ bound. It says nothing
about nonlinear feedback, ionisation or how sensitive a result is to the path. A
nonlinear run's accuracy is set by its own tolerances and refinement. Cartesian and line
runs still take only a medium that is constant along z.

## Index jumps and the side of a plane

A profile can declare jumps, for example a pressure step from one cell to the next.
The certificate counts each jump as $\tfrac12|\ln(q_+/q_-)|$ of $\Lambda$, and the
amplitude term applies the endpoint square-root factor across it without smoothing it
into a derivative spike. At a jump the medium has two one-sided values, so a plane there
is ambiguous. `SpatialFlux.resolve` and `SpatialPulse.build` refuse a position at a
declared jump unless `side="before"` or `side="after"` says which one is meant. They also
refuse a side at a position where there is no jump. A run refuses a save plane placed
exactly on a declared jump; save just before or after it. The two sides give different
measures of the same field:

```python
import numpy as np

from solprop.exceptions import MaterialError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.profiles import FromCallable
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy

grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
space = RadialGrid(radius=1.5e-3, n_r=32)
step = FromCallable(
    lambda z: np.where(z < 0.5e-3, 1.0, 2.0),  # bar
    (0.0, 1e-3),
    jumps=np.array([0.5e-3]),
    turning=np.array([]),  # monotone between the declared jump and the ends
)
cells = SpatialMedium("Ar", pressure=step)
try:
    SpatialFlux.resolve(grid, space, cells, z=0.5e-3)
except MaterialError as error:
    print("refused:", str(error).split(":")[0])

before = SpatialFlux.resolve(grid, space, cells, z=0.5e-3, side="before")
after = SpatialFlux.resolve(grid, space, cells, z=0.5e-3, side="after")
beam = SpatialPulse(
    GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), waist=300e-6
)
field = beam.build(grid, space, medium=cells, z=0.5e-3, side="before")
print(f"before: {spatial_energy(field, before) * 1e6:.6f} uJ")
print(f"after:  {spatial_energy(field, after) * 1e6:.6f} uJ")
```

The field built to carry 1 µJ in 1 bar of argon would carry 1.000257 µJ if the same
field stood in 2 bar. The difference is the index ratio. During a run, the forward
transport across the jump instead reduces the field by $\sqrt{Y_z^-/Y_z^+}$, so the
energy stays 1 µJ.

## Example: a gas cell that fills and empties

A 1 mm argon cell whose pressure rises linearly from 0.2 bar to 2 bar at its centre and
falls back to 0.2 bar at its exit. The beam is wide, with a 300 µm waist and a 0.35 m
Rayleigh length, so diffraction over the cell is negligible and the amplitude change
comes from the gas alone:

```python
import math
from dataclasses import replace

from solprop.adiabatic import AdiabaticAccuracy, certify_radial_path
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.linear import Frame
from solprop.profiles import PiecewiseLinear
from solprop.spatial import SpatialDispersion, peak_field
from solprop.stats import StatsSpec
from solprop.stepper import PIController

fill = PiecewiseLinear.from_stages([0.0, 0.5e-3, 1e-3], [0.2, 2.0, 0.2])  # bar
cell = SpatialMedium("Ar", pressure=fill)

omega0 = np.array([2 * math.pi * C / 800e-9])
n_entry, n_centre = (float(cell.index(omega0, z)[0].real) for z in (0.0, 0.5e-3))
print(f"n0 = {n_entry:.7f} at the entrance, {n_centre:.7f} at the centre")

admission = certify_radial_path(
    SpatialDispersion(cell, grid, Frame.comoving(grid)), space.k_r**2, (0.0, 1e-3)
)
print(
    f"Lambda = {admission.variation:.3g}, R* = {admission.reflection:.3g}, "
    f"delta* = {admission.field_error:.3g}, attained: {admission.exact}"
)

settings = SimulationSettings(
    controller=PIController(rtol=1e-8, atol=1e-14),
    save_planes=[0.5e-3, 1e-3],
    dz_init=1e-4,
    dz_max=2.5e-4,
    stats=StatsSpec(names=("energy",)),
)
run = Simulation.from_radial(grid, space, cell, [], beam, settings=settings).run()
for plane in run.planes():
    flux = SpatialFlux.resolve(grid, space, cell, z=plane.z)
    print(
        f"z = {plane.z * 1e3:.1f} mm: {spatial_energy(plane.field, flux) * 1e6:.9f} uJ, "
        f"peak field {peak_field(grid, space, plane.field) * 1e-9:.6f} GV/m"
    )
print(f"forward amplitude ratio at the centre: {math.sqrt(n_entry / n_centre):.6f}")
```

The index rises from 1.0000514 to 1.0005144. The path is two monotone pieces, so
$\Lambda$ is attained: $4.67\times10^{-4}$ at the worst column, against
$\ln(n_\text{centre}/n_\text{entry})=4.63\times10^{-4}$ at the carrier, with
$R_*=\delta_*=2.2\times10^{-7}$, far inside the $10^{-4}$ budgets. The energy stays
1 µJ at every plane. The peak field falls from 0.402977 to 0.402883 GV/m at the centre,
a ratio of 0.999768 against the predicted $\sqrt{n_\text{entry}/n_\text{centre}}=0.999769$.
It returns to 0.402975 GV/m at the exit, where the residual is the beam's diffraction.

A tighter budget than the path can meet refuses the run at translation:

```python
strict = replace(settings, adiabatic_accuracy=AdiabaticAccuracy(reflected_energy=1e-7))
try:
    Simulation.from_radial(grid, space, cell, [], beam, settings=strict).translate()
except MaterialError as error:
    print("refused:", str(error).split(":")[0])
```

## The hybrid planar guide

The [planar guide](../waveguides/planar.md) synthesises its field from slab modes $V_p(x)$
and amplitudes $A_p(y,\omega)$, with $(\varepsilon_0c/2)\int V_p^*V_q\,dx=\delta_{pq}$. Its
energy weight per mode, y frequency and temporal bin is $w=c\operatorname{Re}q/\omega$, with
$q=\sqrt{(\omega n_{\mathrm{eff},p}/c)^2-k_y^2}$. At $k_y=0$ this is the mode's real
effective index, not the bulk index. The $\varepsilon_0c/2$ is already in $V_p$ and is
not applied again. Local planar fluence and intensity synthesise $G$ from the same modes,
with the complex factors $\varepsilon_0c^2q/\omega$, so the interference between modes
remains. {py:class}`~solprop.pulses.PlanarPulse` normalises the incident Gaussian's
physical energy in the fill medium, averaging the bulk admittance over the Gaussian's x
wave numbers.

## References

1. M. Kolesik and J. V. Moloney, "Nonlinear optical pulse propagation simulation: From
   Maxwell's to unidirectional equations," *Phys. Rev. E* **70**, 036604 (2004),
   [doi:10.1103/PhysRevE.70.036604](https://doi.org/10.1103/PhysRevE.70.036604).
2. M. Visser, "Some general bounds for 1-D scattering," *Phys. Rev. A* **59**, 427–438
   (1999), [doi:10.1103/PhysRevA.59.427](https://doi.org/10.1103/PhysRevA.59.427);
   equations 80–82 give the monotone step bound.
3. R. Diamant and M. Fernández-Guasti, "Reflection coefficient due to a discontinuity in
   the nth order derivative of the refractive index," *J. Opt.* **16**, 015707 (2014),
   [doi:10.1088/2040-8978/16/1/015707](https://doi.org/10.1088/2040-8978/16/1/015707).
