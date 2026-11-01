# The free-space model

This is the reference for free-space propagation, a beam in a bulk medium rather than in a
waveguide. It defines what a free-space run evolves and in which units, derives the exact
longitudinal operator, the absorbing spatial collar and the critical power, and gives the
definitions behind the line grid, the run's memory and its CPU execution. Setting up,
running and reading a beam, with snippets that run as written, is the how-to,
[beams in bulk media](beams.md). Where free space sits among solprop's other transverse
decompositions, with a figure for each and the assumptions each makes, is in
[spatial decompositions](../setup/spatial-decompositions.md).

A free-space run evolves the physical electric field itself, not the amplitudes of
guided modes. Its state columns are transverse spatial frequencies, from one of three
grids:

| grid | columns | assumes | medium | entry point |
|---|---|---|---|---|
| {py:class}`~solprop.grid.RadialGrid` | Hankel frequencies $k_r$ of the order-zero [Hankel transform](hankel.md) | each Cartesian polarisation component cylindrically symmetric | constant, or a gas whose pressure varies along z within the [adiabatic certificate](flux.md#the-adiabatic-certificate) | `prop_radial`, `Simulation.from_radial` |
| {py:class}`~solprop.grid.FreeGrid` | 2-D Fourier frequencies $(k_x, k_y)$, x fastest ([Cartesian free space](cartesian.md)) | nothing: periodic box | constant | `prop_free`, `Simulation.from_free` |
| {py:class}`~solprop.grid.LineGrid` | $k_x$ alone | uniform in y ([the line](#the-line-uniform-in-y)) | constant | `prop_free`, `Simulation.from_free` |

The radial transform does not represent azimuthal structure: a tilted, astigmatic or
multi-filament beam needs the Cartesian grid. Everything else is shared with fibre
propagation: the temporal grid ([grids](../setup/grids.md)), the Kerr and plasma responses, the
interaction-picture stepper ([the solver](../numerics/solver.md)) and the output handlers
([output](../results/output.md)).

## Field units and transverse measure

The intensity is the axial Poynting flux of the medium at the measured plane,
$S_z=\tfrac12\sum_p\operatorname{Re}(E_{a,p}G_{a,p}^*)$ in W/m², summed over the Cartesian
polarisation components $p$, where $G=\mathcal Y_cE$ applies the axial admittance
$\mathcal Y_c=\varepsilon_0c^2k_z/\omega$ to each spectral column. For a paraxial beam in a
transparent medium it is $n_0\varepsilon_0c|E_a|^2/2$. Integrating over time gives the
fluence in J/m², and integrating that over the cross-section gives the energy in J.
[Energy and flux in a medium](flux.md) defines the measure, shows why local values are
signed, and names the separate field diagnostic. A
{py:class}`~solprop.grid.RadialGrid` carries the physical area weights for $2\pi r\,dr$
and the spectral weights for $2\pi k_r\,dk_r$. Its stored Hankel spectrum is in V m; the
physical-space field is in V/m. The [Hankel transform page](hankel.md) derives both
scalings and the raw transform's truncation and floating-point error: it is not exactly
orthogonal at finite size. The Cartesian grid's stored spectrum is a raw DFT, also in
V/m ([Cartesian free space](cartesian.md#transform-and-energy)).

{py:class}`~solprop.pulses.SpatialPulse` wraps any scalar temporal pulse in a transverse
beam. Its transverse field is `exp(-(r/waist)**(2*order))`, so `waist` is the 1/e²
intensity radius and `order=1` is a Gaussian. The wrapper normalises the beam in the
medium's flux measure at the launch plane, so the temporal pulse's `energy` or
`peak_power` is the whole beam's physical energy or peak axial power there: what an energy
meter in that medium reads. The inactive band and zero frequency are set to zero first,
and a beam with more than rounding-level content in nonpropagating columns, which carry
no energy, is refused. The optional `jones=` vector is in the optics convention and is normalised before
it is applied; each component then has its own scalar transform.

{py:func}`~solprop.noise.spatial_noise` draws the same one-photon-per-bin model as the
fibre routes ([shot noise](../media/noise.md)) and divides each transverse column and bin
by the square root of its energy weight in the entrance medium,
`epsilon_0*c*spectral_weight*Re(c*k_z/omega)/2`, so each admitted bin carries
`hbar*omega` ([the discrete measure](flux.md#the-discrete-measure-endpoints-and-noise)).
The `input` and `modified` noise models use the same draw, with
the same propagation semantics and recorded seed as on a fibre.

## Exact longitudinal operator

{py:class}`~solprop.spatial.SpatialMedium` names the medium: a gas, a glass, a gas mixture
or a constant complex refractive index; `SpatialMedium()` is vacuum. A gas can carry a
pressure {py:class}`~solprop.profiles.ZProfile` along z on the radial grid, admitted
when it meets the [adiabatic certificate](flux.md#the-adiabatic-certificate); its density
and susceptibility use the same material functions as a capillary
([materials](../media/materials.md)): optmat's built-in gas, or a `susceptibility=` of the
caller's own beside optmat's equation of state. A run resolves a named gas into its
one-population mixture and binds an optmat rate's energy per event to the input's peak
field; its declared responses are the caller's, so a Kerr response at optmat's
coefficient is `KerrResponse(gamma3(gas, wavelength=wavelength))`
([materials](../media/material-selection.md#selecting-a-runs-materials)). A mixture carries its own
temperature, so `temperature=` beside one is either the mixture's or refused. The
medium's density is also the one its nonlinear responses take: an explicit response
density (`SimulationSettings.density`) beside a gas or a mixture is refused, since the
dispersion and the nonlinearity would then describe different fills. To hold a gas at a
given number density, give the medium that density, `SpatialMedium("Ar").at_density(rho)`.

Let $k=n(\omega,z)\omega/c$ and $q=k^2-k_r^2$. For $\Re q>0$, the principal square
root defines $k_z=\sqrt q=\kappa+i\eta$. The internal linear operator is
$D=-\eta-i(\kappa-\text{frame})-\alpha_\omega/2$, with $\alpha_\omega$ the spectral
absorber rate. In a pressure gradient the accumulated operator also carries the real
forward amplitude term $-\tfrac12\ln[Y_z(z)/Y_z(z_0)]$, with $Y_z=\varepsilon_0c^2\kappa/\omega$,
which keeps each column's energy constant ([radial gas paths](flux.md#radial-gas-paths-forward-amplitude-transport)); it is exactly zero in a uniform medium ([absorbing boundaries](../numerics/boundaries.md)). No paraxial approximation is made:
$k_z$ is exact for every transverse frequency. A positive imaginary material index is
loss and a negative one is gain. The frame is evaluated at transverse frequency zero and
follows the field and envelope conventions of [the linear operator](../numerics/linear-operator.md).

The nonlinear polarisation prefactor, before temporal crop scaling, is
$-i\omega^2/(2\varepsilon_0c^2\kappa)$, with the local $\kappa$. It uses the same `epsilon_0`
as the flux measure: independently rounded `mu_0` and `epsilon_0` table entries need not satisfy
their defining relation to machine precision, so the coefficient is derived from
`epsilon_0` and the exact `c`.

For $\Re q\le0$ the column is evanescent, and the declared numerical model uses the
attenuation `min(abs(Im(sqrt(q))), 200)` in m⁻¹ and the clamped prefactor `-i*omega/2`.
The value 200 m⁻¹ is a numerical choice that keeps evanescent columns bounded; it is not
a material absorption coefficient, and it does not cap propagating material loss or gain.
Inactive temporal frequencies have zero operator and prefactor.

**Setting the evanescent rate.** Physically a non-propagating column decays as
$e^{-|\Im k_z|z}$, with $|\Im k_z|\approx(k_\perp^2-\Re k^2)^{1/2}$ of order
$10^6$–$10^7$ m⁻¹ on a sub-wavelength grid. Over a 40 µm slab the default cap attenuates
such a column by less than one per cent, so a thin sample whose evanescent content must
decay as it physically does sets the cap itself,
`SimulationSettings(evanescent_rate=...)`, in m⁻¹. `None`, the default, keeps
{py:data}`~solprop.spatial.EVANESCENT_RATE`. The largest physical rate the grid holds is
{py:func}`~solprop.spatial.physical_evanescent_rate`: at each frequency
$\Im\sqrt{k^2-k_\perp^2}$ is monotone in $k_\perp^2$, so the search sorts the columns
once and evaluates the largest transverse square at every frequency, in
$O(n_\omega\log n_{\mathrm{modes}})$. For glass of index 1.45 on a 0.4 µm grid it is
about $7\times10^6$ m⁻¹ at 1.1 µm:

```{code-block} python
from solprop.spatial import physical_evanescent_rate

rate = physical_evanescent_rate(space, medium, grid)
settings = dataclasses.replace(settings, evanescent_rate=rate)
```

The setting is refused on a run without a spatial grid and must be positive and finite;
a `SpatialPulse` launched with `propz=` keeps the default cap for its pre-propagation,
since a physical rate would overflow a backward launch. The setting is recorded in the
run's configuration only when given.

**The exponent budget.** Every backend forms the inverse propagator
$e^{-D\Delta z}$, whose decaying columns grow as $e^{\lambda_-\Delta z}$ with
$\lambda_-=\max(-\Re D)$; past $\ln(\mathrm{DBL\_MAX})\approx709.78$ the factor is
infinite and the next transform spreads it to every column. A gain medium grows the
forward factor as $e^{\lambda_+\Delta z}$, over one step for the Runge-Kutta marches and
over up to $q+1$ steps for an Adams method of largest order $q$. An adaptive run finds
its own limit, since its controller rejects a non-finite trial and shrinks; a step the
run cannot shrink -- a replayed schedule's longest step, or the floor `dz_min` -- is
checked at translation against sufficient bounds on $\lambda_\pm$ from the same
sorted search. Past the limit translation refuses the run, naming the binding term,
the factor and the admissible step and rate: for example, a replayed step of 0.1 mm
with `evanescent_rate=7.4e6` forms $e^{740}$ and is refused, naming steps of at most
96 µm. Within a tenth of the limit it warns
({py:class}`~solprop.exceptions.ExponentBudgetWarning`): such a run completes only while
its field stays below $e$ to the margin. A run is refused only when it could not
complete, and the default cap keeps every default schedule far inside the budget. The check covers the
Cartesian, line and planar kinds from the separable operator's search, and the radial
kind from its operator table: the exact extremes of $\Re D$ in a constant medium, and a
sufficient bound from the rate's Chebyshev series in a varying one. A radial refusal
names the radial operator's attenuation as a whole, since the evanescent cap, the
spectral rate and the medium's loss are not separable there, and the admissible step.

**Launching away from the waist.** A `SpatialPulse` with `propz=` set is carried from its
waist by this exact operator before the run starts, with the entrance medium held fixed
during the launch; a negative distance launches before the waist, so the beam focuses in
the run. Exponential overflow raises. By default the launch is made in the run's own
entrance medium; `SpatialPulse.build(grid, space, medium=SpatialMedium())` launches through
vacuum instead, and the array it returns is the run's input. Its requested energy is then
the energy in vacuum: the same field carries $n_0$ times as much in the run's medium,
which the run's `energy` statistic reports
([focusing, and launching away from the waist](beams.md#focusing-and-launching-away-from-the-waist)).
The [radial tutorial](../../auto_tutorials/free_space/plot_01_radial.rst) launches its beam
this way.

## Transverse boundaries

A free-space grid has a hard edge: the radial transform holds the field at zero at its
outer radius, and the Cartesian, line and planar grids are periodic, so light that
reaches the edge is reflected or wraps round into the aperture, and the answer is wrong.
solprop's answer, from [SEP 40](../../dev/seps/sep-0040-transverse-absorbing-boundaries.md),
is **room first, detection always, and absorption only for declared fast light**.

**Room first.** The grids take `collar_fraction`, the room outside the requested size,
as a fraction of it: `RadialGrid(R, n_r, collar_fraction=f)` puts its wall at
`outer_radius = R*(1+f)`, and `FreeGrid`, `LineGrid` and `PlanarGrid` make their period
`radius*(1+f)` on each axis; `n_r`, `n_x`, `n_y` count every sample. The default is 0: the
requested size is the grid's. For a slow beam a bare wall further out beats any designed
collar at the same outer radius -- on the suite's self-focusing case a wall at twice the
radius is wrong by `4e-5` in field where the best collar there gives `1e-3` -- so the first
remedy for light at the edge is more room at the same spacing.

**Detection always.** The [boundary monitor](../numerics/boundaries.md#watching-the-windows-edges)
watches the transverse edge of every free-space run. With no real-space collar, whatever
reaches the edge comes back, so it judges, per face, the energy in the outer 5 % of the
*outer* size above its launch value. Energy shares above `1e-6` of the launch warn and
above `1e-4` stop the run (`BoundaryCheck`'s `budget` and `stop_factor`, or its own
`transverse_budget`), and the stop names a size and a sample count to try. That size is a
lower bound: light generated late, by a filament, may need more. Reads are spaced so that
light cannot cross the edge unseen between two of them, at most `min(L/16, 4g/v)` apart,
with `g` the edge's width and `v` the spread of the current spectrum, and the spacing is a
ceiling on the step: a low-source run, which would otherwise take long steps, takes a few
more. A fixed schedule cannot be held to it, and reports itself as `coarse`.

Every run reports a `transverse_status` in its monitor record and says at the start what
it will be able to report: `clear` (read at the spacing, below the warning), `warned`,
`stopped`, `insensitive` (the input noise's allowance exceeds the warning), `coarse`,
`unwatched` (this backend does not read the edge yet; `numpy_ref` does) or `off`
(`boundary_check=None`). Only `clear` is a clean verdict.

**The safety ramp.** Every free-space run also carries a transverse-wavenumber ramp on the
outer 10 % of each axis's wavenumbers, radial and planar included, folded exactly into the
linear operator at no cost per step. It removes what reaches the grid's resolution limit,
where nothing physical should be, with a strength set by the grid, not the run length. A
launch with physical light in that band is judged before the first step: above `1e-6` of
its energy it warns, and above `1e-4` it is refused, because the ramp would erase it.
Refine the grid, launch with less tilt, declare the content with
`SpatialAbsorber(discard_above=...)`, or remove the ramp with `spatial_absorber=False`.
What the ramp removes later is reported as `ramp_removed` and warns above the budget.

### Spatial collar and statistics

**A collar on request.** A run whose fast light cannot be given room -- four-wave-mixing
products at an angle, conical emission -- asks for a real-space collar in the grid's room:

```python
from solprop.boundaries import SpatialAbsorber

space = RadialGrid(300e-6, 230, collar_fraction=2.6)
prop_radial(
    grid,
    space,
    medium,
    responses,
    pulse,
    settings=settings,
    spatial_absorber=SpatialAbsorber(),
)  # band from the launch
prop_free(
    ..., spatial_absorber=SpatialAbsorber(angle=25e-3)
)  # protect sin(theta) <= 25e-3
```

The collar is designed at translation from a *protected* band of transverse wavenumbers
(an explicit `kappa`, an `angle`, or three times the launch's rms spread) and its width,
as a complex absorbing potential whose strength attenuates the band's edge by
`log(5e3)` nepers on the way out and back. Translation then propagates the launch through
it, linearly, and compares with a bare grid large enough not to return it: a design that
returns more than `4e-8` of the launch's energy is refused, naming a fraction that passes,
and saying when a bare wall at the same size returns less. A design is also refused below
the profile's floor (`kappa_p * w >= 22.3`), near cutoff, or with no room. A weak generated
or launched signal is protected only if its angle is declared: the launch-derived band
covers the launch, not what it will generate. The design check is rerun for a new launch
given by `with_input`.

The collar is an in-stage rate, `F diag(-alpha/2) B` in every stage, so the error
controller sees it, under a step limit that stops a protected packet crossing it unseen
within one step; a fixed schedule beyond that limit, or beyond the damping bound
`alpha0 h / 2 = 1.3`, is refused. With a collar, reaching the edge is intended, and the
monitor judges what *returns*: the inward energy in a strip over the collar, resolved by a
windowed transform, and the inward current across its onset. A slow reflected spread under
a collar warns rather than stops; the design check is what keeps that case out. The
collar's removal is recorded as `spatial_removed`, a share of the launch's field norm.

**The record.** The specification is stored under `configuration["spatial_absorber"]`
when one is given, and the resolved design's derived values -- the ramp's start and
strength, a collar's strength, band, threshold and step limit, and its check -- under the
run metadata's `transverse` record, outside the resume key.

**Statistics.** The default set for a spatial run (`default_spec("radial")` or
`default_spec("free")`) is `energy`, `on_axis_intensity`, `peak_intensity`,
`beam_radius`, `fwhm_r` and `on_axis_spectrum`. The [statistics catalogue](../results/statistics.md)
distinguishes `peak_intensity`, the largest axial flux over the physical samples, from
`on_axis_intensity`, reconstructed separately on the axis. Both are signed local flux in
the medium at the plane ([local flux](flux.md#axial-admittance-and-local-flux)). The widths
use the time-integrated fluence, and refuse one that is negative beyond its rounding
budget: `beam_radius` records the second-moment radius and the
interpolated central 1/e² radius, and `fwhm_r` is a full width. An empty beam has no
width, and a profile whose axis fluence lies below a threshold (hollow at that level) has
no central width at it; these entries are NaN. The thresholds are fractions of the peak
fluence, so a flat top whose first samples ripple above the reconstructed axis value keeps
its widths. The saved on-axis spectrum is the real spectral fluence density, in J s/m²,
summed over polarisation: the same-frequency pairing of the coherent on-axis $E$ and $G$.

{py:func}`~solprop.spatial.radial_fluence_widths` is the public host reduction for an
already integrated radial fluence. It validates the supplied samples and weights; the
native backend calls it after transferring its small fluence vector.

### Far-field and Fourier-plane statistics

A Cartesian or line run can also record the spectrum a detector behind the sample would
see through apertures in a far-field or Fourier plane, in boxes of transverse
wavevector and of the whole beam, and the power the absorbers remove
([far-field apertures](../results/statistics.md#far-field-apertures-and-angular-regions)).
They rest on three statements about where the light goes after the run's last plane.

**The far zone, or a Fourier plane.** The run ends at a planar exit face into vacuum
(air, to the precision of the mapping); phase matching there conserves the transverse
wavevector $\boldsymbol\kappa$, so a component leaves at $\sin\theta=c|\boldsymbol\kappa|/\omega$
whatever the medium of the run, and only components with $|\boldsymbol\kappa|<\omega/c$
leave at all. Fresnel transmission at the face, which depends on angle and polarisation,
is not modelled, as at every end of a simulation. In the far zone of the exit plane --
Fresnel number $a^2/(\lambda z_m)\ll1$ for a source of transverse size $a$ at distance
$z_m$ -- the field in a direction is the angular spectrum at that direction's
$\boldsymbol\kappa$, by stationary phase (L. Mandel and E. Wolf, *Optical Coherence and
Quantum Optics*, Cambridge University Press, 1995,
[doi:10.1017/CBO9781139644105](https://doi.org/10.1017/CBO9781139644105)), and it meets
the plane $z_m$ at $z_m\tan\theta$. In the back focal plane of a lens of focal length $f$
the paraxial relation $\mathbf x=f c\boldsymbol\kappa/\omega$ holds within thin-lens
optics whatever the distance from the sample to the lens (J. W. Goodman, *Introduction
to Fourier Optics*, 4th ed., W. H. Freeman, 2017), so an aperture's `distances` entry is
either $z_m$ or $f$.

**The paraxial error.** The statistics use the paraxial position $z_m\sin\theta$,
$\mathbf x\simeq(z_mc/\omega)\boldsymbol\kappa$, short of the far-zone one by the factor
$\cos\theta$: a relative error $1-\cos\theta\approx\theta^2/2$, $1.25\times10^{-3}$ at
50 mrad. It is the statistic's definition, exact for a Fourier plane, not an error the
tests bound; an exact $\tan\theta$ mapping for wide-angle emission is a roadmap item.

**The axial flux in the medium.** Every spectrum weights each transverse column by its
real axial admittance $Y_z=\varepsilon_0c^2\operatorname{Re}k_z/\omega$ in the run's
medium, as `energy` does ([energy and flux in a medium](flux.md)), so `beam_spectrum`
integrates to the energy. It is the power crossing the run's last plane inside the
medium. The exit face's Fresnel transmission is not modelled, so the corrected measure
supplies no transmission into vacuum; that needs an explicit interface. After the face,
the flux of each direction in vacuum differs from the in-medium value by that
transmission and by the change of obliquity.

## Construct and run a beam

The how-to builds and runs beams with `prop_radial` and `prop_free`
([beams in bulk media](beams.md#a-beam-in-argon)), which combine construction and
execution. `Simulation.from_radial` and `Simulation.from_free` take the same arguments and
separate the two:

```{code-block} python
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import PIController

grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
space = RadialGrid(radius=600e-6, n_r=64)
pulse = SpatialPulse(
    GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9),
    waist=60e-6,
)
settings = SimulationSettings(
    controller=PIController(rtol=1e-8, atol=1e-14),
    save_planes=[0.5e-3, 1e-3],
    dz_init=1e-4,
    dz_max=5e-4,
    stats=default_spec("radial"),
    backend="numpy_ref",
)
simulation = Simulation.from_radial(
    grid,
    space,
    SpatialMedium(),
    [],
    pulse,
    settings=settings,
)
output = simulation.run()
```

The empty response sequence gives exact linear diffraction in the selected temporal
frame; add response objects — a `KerrResponse`, a plasma response — to the sequence for
nonlinear propagation ([nonlinear responses](../media/nonlinear.md)). The absolute tolerance is
in the stored Hankel spectrum's units, V m. The settings carry no absorber, so the collar
here absorbs nothing ([above](#spatial-collar-and-statistics)). Translation is cached, so
a second `run()` reuses it.

A spatial run's output metadata stores the spatial axes, frequency ordering,
integration weights, collar, units and transform scalings. The state-layout names are
those of every other run, with `n_modes` interpreted as transverse spectral columns.
The geometry is part of the resume key.

## Critical power and self-focusing

For a scalar monochromatic paraxial field in an isotropic Kerr medium,
$i\partial_z A+(2k)^{-1}\nabla_\perp^2 A+g|A|^2A=0$, with
$g=3k\chi^{(3)}/(8n^2)$ and $k=2\pi n/\lambda_\mathrm{vac}$.
The positive radial stationary solution satisfies
$Q''+Q'/r-Q+Q^3=0$. If $\alpha=\int_0^\infty Q^2r\,dr$, rescaling that
equation gives a threshold on $\int|A|^2dS$ in which the index cancels. The paraxial
axial power in the medium is $n_0\varepsilon_0c/2$ times that integral
([energy and flux in a medium](flux.md)), so

$$P_\mathrm{crit}=\frac{\alpha\,n_0\varepsilon_0c\lambda_\mathrm{vac}^2}{3\pi\chi^{(3)}}.$$

At fixed $\chi^{(3)}$ this is $n_0$ times the value of a vacuum field measure.
{py:func}`~solprop.spatial.critical_power` uses `alpha=1.86226`, the Townes value; call it
with the wavelength, the medium's $\chi^{(3)}$ and its real carrier index, for example
`critical_power(800e-9, chi3("Ar", pressure=1.2, wavelength=800e-9), index=n0)` with
`chi3` and `refractive_index` from {py:mod}`solprop.physdata`. The independent
stationary boundary-value calculation gives `1.86225552049`; the paper's approximate
`1.86225` misses a half-last-decimal rounding allowance of `5e-6` by giving an error
of `5.52049e-6`. This numerical check is not an interval certificate.
[Fibich and Gaeta (2000), equations 6–7](https://www.math.tau.ac.il/~fibich/Manuscripts/critical00.pdf)
derive the Townes baseline and the Gaussian negative-Hamiltonian sufficient value
`alpha=2`; their computed threshold for a collimated Gaussian is `1.8962` (their Table 1),
less than 2 % above the Townes value. Their power is the physical axial Poynting power
in the medium, as every spatial energy in solprop is.

The [radial tutorial](../../auto_tutorials/free_space/plot_01_radial.rst) runs a beam focused as
in vacuum and carrying 2 nJ in argon, far below this baseline, and separately a
collimated diagnostic whose field carries twice it. Twice `1.86226` exceeds the Gaussian sufficient value `2`; this does not predict
where a temporally varying, exact-UPPE pulse will collapse. Plasma, higher-order
responses, discretisation and departure from the paraxial approximation further
limit such a comparison.

For comparison the tutorial prints the Marburger estimate of the collapse distance,
$z_\mathrm{M}=0.367z_R/\sqrt{(\sqrt{P/P_\mathrm{crit}}-0.852)^2-0.0219}$,
the collimated Gaussian fit of [Marburger (1975)](https://doi.org/10.1016/0079-6727(75)90003-8),
in the form given by [Couairon and Mysyrowicz (2007), equation 5](https://doi.org/10.1016/j.physrep.2006.12.005),
with $z_R=kw_0^2/2$.
solprop does not compute it and nothing depends on it: no uniform accuracy bound for that
fit is claimed here. The tutorial's diagnostic run stops at `0.3 z_R` to illustrate the
initial contraction, not a resolved collapse.

## Cartesian example and memory

`FreeGrid(radius_x, n_x, radius_y, n_y)` names x before y, and wraps at the requested
half-widths unless `collar_fraction` gives it room ([transverse boundaries](#transverse-boundaries)); a Cartesian run is shown in
[Cartesian and line beams](beams.md#cartesian-and-line-beams). The [Cartesian tutorial](../../auto_tutorials/free_space/plot_02_cartesian.rst)
adds argon Kerr propagation. The scalar second-moment width uses the fluence centroid, and
the interpolated widths use circular averages of the bilinear fluence interpolant; both
definitions and their sampling errors are derived under
[fluence widths](cartesian.md#fluence-widths).

**Memory.** A Cartesian run costs $n_xn_y$ columns per time sample, so its memory is
what limits it. Compiled stages retain one full transverse scratch and bounded temporal columns.
The node budget, 1 GiB by default on free-space, line and planar grids (64 MiB on
modal routes), allows twenty complex-equivalent buffers per physical
component per fine time sample. Eight cover the response and its FFT pairs, three
cover doubled Raman scratch, 3.5 cover device plasma scans, two cover split-step
transforms and 2.5 cover statistics. One more covers real-FFT endpoint bins. This
inventory is conservative even though some response families cannot coexist. Persistent
frequency-only material vectors and one `kperp_sq` vector replace coefficient volumes.
The native backend stores its FFT scratch with transverse axes last, so Kokkos-FFT does not allocate
full-volume transposition temporaries on execution. Canonical state order is unchanged.

The buffer ledger counts retained arrays and separately records process allocator
net growth while constructing opaque FFTW plans. The native backend additionally measures device
free-memory decrease across plan construction. These plan measurements include vendor
workspace but are not upper bounds or process high-water marks: allocator reuse and
shared caches affect them. Whether a full-size run fits a GPU is therefore settled by
measuring its plan usage on that device, not by the ledger alone.

## The line, uniform in y

{py:class}`~solprop.grid.LineGrid` is the Cartesian grid's x axis alone: the same centred
samples, FFT order and Planck collar, with one y sample at $y = 0$ carrying $k_y = 0$,
no y collar and a unit y measure. The field is uniform
in y, so every transverse integral is over x **per metre of y**: the spatial and
spectral weights are $\Delta x$ and $\Delta x/n_x$ in m, the energy is in J/m and the
power in W/m, while intensity and fluence stay local, in W/m² and J/m². The run records
`transverse_measure = "per_unit_width"`, and processing and the plots label from it, as
for the [slab](../waveguides/planar.md#hollow-planar-waveguides).
{py:class}`~solprop.pulses.SpatialPulse` takes its energy or peak power in J/m or W/m on a
line; its waist is the 1/e² intensity half-width in x.
{py:func}`~solprop.modes.slab.gaussian_to_per_unit_width` converts the energy of a beam
of 1/e field radius $w_y$ in y to the J/m at its centre. The noise draw is one photon per
column and bin per metre of y, a declared convention.

The run is the Cartesian one with $n_y = 1$: the same separable exact operator with
$k_\perp^2 = k_x^2$, the same two-axis FFT with a unit y extent, and the same collar,
statistics and memory layout, on `numpy_ref`, `numba_cpu` and `native` (its CPU build and
its CUDA build). The widths are one-dimensional: the
second moment $2\sqrt{\langle(x-\bar x)^2\rangle}$ is the 1/e² half-width of a Gaussian
fluence, as the plane's $\sqrt{2\langle r^2\rangle}$ is its radius, and the threshold
widths average the two sides of the centroid. Varying media are refused, as on the plane.

**The physics of one free dimension.** A Gaussian of waist $w_0$ still widens as
$w_0\sqrt{1+(z/z_R)^2}$ with $z_R = kw_0^2/2$, but its axial amplitude falls as
$(1+(z/z_R)^2)^{-1/4}$ and its Gouy phase is $\tfrac12\arctan(z/z_R)$, half the round
beam's. The tests measure both, and the width, against these closed forms to 2e-6, the
size of the exact operator's first non-paraxial term $(kw_0)^{-2}$.

Kerr self-focusing in one transverse dimension does not collapse. The equation above,
$i\partial_zA + (2k)^{-1}\partial_x^2A + g|A|^2A = 0$ with $g = 3k\chi^{(3)}/(8n^2)$, has
the stable stationary solution $A_0\operatorname{sech}(x/x_0)$ when
$gkA_0^2 = 1/x_0^2$, so $A_0^2 = 8/(3k_0^2\chi^{(3)}x_0^2)$ and the index cancels from the
field, as it does in the Townes threshold. Its physical power per metre of y, the axial
flux $n_0\varepsilon_0c|A|^2/2$ integrated over x, is

$$P'_\mathrm{sol} = \frac{n_0\varepsilon_0 c}{2}\int|A|^2dx = \frac{8n_0\varepsilon_0c}{3k_0^2\chi^{(3)}x_0}
= \frac{2n_0\varepsilon_0c\lambda_\mathrm{vac}^2}{3\pi^2\chi^{(3)}x_0}
\approx 0.3419\,\frac{P_\mathrm{crit}}{x_0},$$

a family of solitons, one for every width, rather than a threshold. This power is derived
here and held by the tests described next; it has not been checked against an independent
(1+1)-D soliton reference. In
argon at 1 bar and 800 nm a 100 µm soliton carries 3.1×10¹³ W/m. The tests hold it over
three diffraction lengths $k x_0^2$ at the temporal peak of a 1 ps pulse to 3×10⁻⁴ of
its peak, and see half its power diffract. There is no critical power:
{py:func}`~solprop.spatial.critical_power` refuses a `LineGrid` by name rather than
return a number in W that a power in W/m cannot be compared with. A sheet beam's
modulational instability in y is outside the model.

A line run is shown in [Cartesian and line beams](beams.md#cartesian-and-line-beams), and
the [line tutorial](../../auto_tutorials/free_space/plot_03_line.rst) runs a Kerr line beam
at the soliton power and at half of it.

## CPU execution settings

The default backend is the installed native package, and otherwise `numba_cpu`; see
[backend selection](../performance/resources.md#backend-choice-and-scan-workers). Thread counts
and the other performance settings shared by every route are on the
[performance page](../performance/tuning.md); what follows is specific to free space.

On `numba_cpu`, radial envelopes use real BLAS products for both parts of the complex
field, including vector and chunked transforms. Cartesian envelopes arrange their shared
transverse scratch for contiguous x/y FFTs; field runs keep contiguous frequency columns.
Large Cartesian problems parallelise independent stage columns, and envelopes also
parallelise the strided copies to and from the bounded temporal buffers. These choices
preserve the transform scaling and each component's arithmetic.

A free-space run has two FFT pools, one for the temporal transforms and one for the
transverse ones. `SOLPROP_FFTW_THREADS` sets the temporal pool and
`SOLPROP_TRANSVERSE_FFTW_THREADS` the transverse pool; unset, each uses a size-based rule
of one or four workers, capped by the available CPUs. Direct `NumbaCpuBackend`
construction also accepts `fftw_threads` and `transverse_fftw_threads`, and
`SOLPROP_NUMBA_THREADS` sets the kernel pool. Set them before starting Python, for
example

```console
SOLPROP_NUMBA_THREADS=12 SOLPROP_FFTW_THREADS=4 \
SOLPROP_TRANSVERSE_FFTW_THREADS=12 python beam.py
```

These counts were measured as a starting point for a large Cartesian beam on one
twelve-core machine; they are not defaults, and the best counts depend on the machine and
the grid, so time a few steps before a long run. The run's metadata records both FFT pools, the
scratch layout, the parallel passes and the radial GEMM selection.

The native backend uses column loops for Cartesian stage operations. Its builds that
link a vendor BLAS can pack whole scalar radial envelopes into real BLAS buffers when the
shape and the node budget permit it, and the memory ledger counts those buffers
([the native backend](../performance/native.md)).

## Related calculations

For a first calculation, follow [beams in bulk media](beams.md), the [radial filament](../../introduction/lessons/plot_03_filament.rst) or the [crossing-beam lesson](../../introduction/lessons/plot_04_crossing_beams.rst). Use [Hankel](hankel.md) and [Cartesian](cartesian.md) transforms for the detailed spatial sampling conventions, and [boundaries](../numerics/boundaries.md) when checking energy lost at a collar.
