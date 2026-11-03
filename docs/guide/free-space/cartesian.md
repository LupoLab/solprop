# Cartesian free-space propagation

Choose a Cartesian grid for tilted, astigmatic or crossing beams, and for
structure that can develop in either transverse direction. The grid represents
a periodic box. Its width must contain the beam throughout propagation, while
its spacing must resolve both its envelope and any interference fringes.
An absorber can remove light near the boundary, but cannot compensate for an
under-resolved beam.

The [crossing-beam lesson](../../introduction/lessons/plot_04_crossing_beams.rst)
builds a coherent array launch and checks motion, fringes, energy and spatial
refinement. The [beam workflow](beams.md) covers `SpatialPulse`. This chapter
explains the transform, energy measure, grazing-incidence limits and beam-width
statistics needed to interpret those calculations.

## The grid

{py:class}`~solprop.grid.FreeGrid` is built as
`FreeGrid(radius_x, n_x, radius_y, n_y, collar_fraction=0)`, x before y. The periodic
half-width is `radius*(1+collar_fraction)` on each axis, and `n_x`, `n_y` count every
sample; the default wraps at the requested half-widths, and a fraction gives the room a
requested collar is designed in ([transverse boundaries](model.md#transverse-boundaries)).
{py:class}`~solprop.grid.RadialGrid` places its room the same way, outside its `radius`. A flattened
column is `ix + n_x*iy`: x varies fastest. The frequencies are in FFT order, and odd
sizes put the origin between samples. {py:class}`~solprop.grid.LineGrid` is the x axis
alone, with one y sample of unit width
([the line](model.md#the-line-uniform-in-y)).

## Grid sizing and the transverse safety ramp

On a periodic grid of spacing $\Delta x$ the sampled wavenumbers satisfy
$|k_x|\le k_{x,\max}\approx\pi/\Delta x$. A cubic source evaluated pointwise in space --
Kerr, and Raman, which is delayed in time but local in $(x,y)$ -- multiplies three field
factors, so content within $|k_x|\le K$ produces products within $3K$. The discrete
product is generated **already folded** at $k-2k_{x,\max}$: envelope Kerr acting on a
beam at $0.95\,k_{x,\max}$ and one at zero lands at $1.9\,k_{x,\max}$, which appears at
$-0.1\,k_{x,\max}$. Folded products miss the energetic band only while $K<k_{x,\max}/2$,
the **one-half rule**, the cubic analogue of Orszag's two-thirds rule for quadratic
products (doi:10.1175/1520-0469(1971)028<1074:OTEOAI>2.0.CO;2). Grid sizing is the
anti-aliasing: the grid must be fine enough that the content carrying energy satisfies
the rule, and refining it checks that. Plasma and ionisation sources are not polynomial
in the field, so no multiple of $K$ bounds their products, and only refinement judges
the grid.

Every free-space run carries a **safety ramp** in $k_\perp$
([SEP 40](../../dev/seps/archive/sep-0040-transverse-absorbing-boundaries.md)). With
$\rho=\max(|k_x|/k_{x,\max},|k_y|/k_{y,\max})$ its power rate is $\alpha_s P(\rho)$, zero
for $\rho\le1-f_s$ and rising as a raised cosine to $\alpha_s$ at the grid's largest
wavenumber, with $f_s = 0.1$ by default; on a {py:class}`~solprop.grid.LineGrid` it acts on
$k_x$ alone. The max-norm protects the rectangle the grid defines, since aliasing in $x$ is
decided by $k_x$ alone, and each axis's strength is designed from its own size: content in
the ramp's upper half is attenuated by $\log(5\times10^3)+2$ nepers over the distance in
which it would cross the aperture and come back, with the larger of the two axes'
strengths on the profile. The rate is diagonal in the transverse wavenumber, so it enters
the separable operator as an exact rate on the active band, $D\to D-\alpha_s P/2$, at no
cost per step and with no effect on the noise phase. It damps content that reaches the
outer band, from cascades or drifting towards the edge, and keeps rounding from
accumulating at the Nyquist edge. It **cannot** remove an alias that in-band content
generates, because the fold happens inside the product's evaluation and lands where
$P=0$; that is the grid's job. The ten per cent touches only content that should not be
there on a grid sized by the rule: products between $K$ and $k_{x,\max}$, a four-wave-mixing
signal leaving at a larger angle say, are physical and are not damped.

```{code-block} python
from solprop.boundaries import SpatialAbsorber

# a wider ramp, no collar; spatial_absorber=False removes the ramp altogether
settings = dataclasses.replace(
    settings, spatial_absorber=SpatialAbsorber(real_space=False, safety_fraction=0.2)
)
```

A launch with physical light in the ramp's band is judged before the first step and
refused above `1e-4` of its energy, unless declared with `discard_above`, which also starts
the ramp there. The ramp and a requested real-space collar
([transverse boundaries](model.md#transverse-boundaries)) are complementary: the collar
removes light that has walked to the edge of the box and protects a declared band of
angles, the ramp acts on the grid's highest wavenumbers wherever they are. On an
anisotropic box each face of a collar is designed on its own axis. The specification is
recorded in the run's configuration when one is given. The far-field statistic
`absorbed_power` reports the power the ramp removes as its transverse channel
([far-field statistics](model.md#far-field-and-fourier-plane-statistics)).

### Reading the risk of aliasing

A run does not report that its products fold: an adaptive step can converge to the aliased
answer, and a folded product is ordinary light once it lands. What a run can report is whether
the field still satisfies the one-half rule. A product of order $p$ of a band $|k|\le K$
reaches $pK$, and the grid stores it at $pK-2k_{\max}$. That lands inside the band when
$pK-2k_{\max}\ge-K$, that is when $K\ge 2k_{\max}/(p+1)$: $k_{\max}/2$ for a cubic product
and $2k_{\max}/3$ for a quadratic one, Orszag's two-thirds. So the share of the beam's energy
outside the inner box $|k_x|\le k_{x,\max}/2$, $|k_y|\le k_{y,\max}/2$,

$$ R(\omega) = 1-\frac{S^{R}(\omega)}{S^{\rm beam}(\omega)}, $$

reads whether a cubic fold can land in band. Both spectra are
[far-field statistics](model.md#far-field-and-fourier-plane-statistics), on
`numpy_ref` and on the native backend, on Cartesian grids and lines, at saved planes; radial
grids refuse them, having no Cartesian wavevector. A region is half-open, so the box is widened
by half a sample to take in the column at $+K$. Summing the two spectra over a band gives one
number for that band.

```python
import numpy as np

from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController, RungeKutta


def inner_box(space, fraction=0.5):
    """The box of a fraction of each axis's largest wavenumber, one row of a region table."""
    half_x = fraction * np.abs(space.k_x).max() + 0.5 * abs(space.k_x[1] - space.k_x[0])
    half_y = fraction * np.abs(space.k_y).max() + 0.5 * abs(space.k_y[1] - space.k_y[0])
    return np.array([[-half_x, half_x, -half_y, half_y]])


def aliasing_reading(output):
    """The energy's share outside the box, at each plane."""
    beam = np.asarray(output["stats/beam_spectrum"]).sum(axis=-1)
    inside = np.asarray(output["stats/region_spectrum"])[:, 0].sum(axis=-1)
    return 1.0 - np.divide(inside, beam, out=np.ones_like(beam), where=beam > 0)


length, half_width = 1e-3, 1e-3
grid = EnvelopeGrid.from_limits(length, 800e-9, (600e-9, 1200e-9), 300e-15, thg=False)
temporal = GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-9).build(grid)[0, 0]
temporal = 2e8 * temporal / np.abs(np.fft.ifft(temporal)).max()  # V/m at the peak

for n_x in (16, 32):
    space = FreeGrid(half_width, n_x, half_width, 4)
    x, _ = np.meshgrid(space.x, space.y)
    fringes = 2 * np.cos(5 * np.pi * x / half_width)  # k = +-5 of the fundamental
    launch = np.fft.fft2(fringes[..., None], axes=(0, 1)) * temporal
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-16),
        save_planes=[length],
        dz_max=length / 8,
        dz_init=length / 16,
        integrator=RungeKutta(),
        backend="numpy_ref",
        spatial_absorber=False,
        boundary_check=None,
        stats=StatsSpec(
            names=("beam_spectrum", "region_spectrum"),
            spectral_regions=inner_box(space),
        ),
    )
    output = Simulation.from_free(
        grid,
        space,
        SpatialMedium(1.45),
        [KerrResponse(2.2e-22, thg=False)],
        launch.reshape(-1, 1, grid.n_omega),
        settings=settings,
        noise=None,
    ).run(progress="none")
    print(n_x, np.round(aliasing_reading(output), 3))
```

On 16 samples the fringes at $\pm5$ lie past $k_{x,\max}/2=4$ and the reading is 1, and the
product at $\pm15$ folds to $\mp1$: false low-angle light. On 32 samples they lie inside
$k_{x,\max}/2=8$, the reading is 0, and the product at $\pm15$ is real. Refining the grid until
the reading falls is the check. The share of the *source* near the cutoff would not see this,
since the folded product sits far from the cutoff, at $\mp1$.

It is a proxy, and it says less than a measurement would:

- It says that a fold *can* land in band, not how large it is. A reading of 1 with a nonlinear
  phase of a hundredth is a small error, and the size has to come from refining the grid.
- It weights by the beam's energy, as every spectrum there does. A weak field past the box reads
  as a small share, and a fold of it can still matter against something weaker. The admittance in
  that weight is zero on an evanescent bin, so light there is not counted at all.
- Light that has already folded in band lowers the reading, and light folded past the box raises
  it, so it is read along $z$ and against a refined run, not at one plane.
- A box of fixed $\kappa$ is not a fixed angle, since the angle goes as $c\kappa/\omega$. So it says
  nothing about the populated angle against the margin of the scalar forward model.
- It is the cubic limit. A quadratic product has the limit above, with `fraction=2/3`, and
  [SEP 8](../../dev/seps/sep-0008-quadratic-crystal-optics.md) owns padding and projection for such
  products, which this chapter does not provide.
- Plasma and ionisation sources are not polynomial in the field, so no box bounds them, and only
  refinement judges their grid, as above. A reading near 0 does not clear them.

## Transform and energy

For $N=N_xN_y$, the stored transverse spectrum is the raw forward DFT,
$F_{pq}=\sum_{mn}f_{mn}\exp[-2\pi i(pm/N_x+qn/N_y)]$ over the sample indices
$m$ and $n$. Its inverse divides
by $N$. Physical and stored Cartesian fields both have units V/m. Radial
Hankel spectra instead have units V m. Applying the two geometric sums to
$\sum|F|^2$ gives the discrete Parseval identity

$$\Delta x\Delta y\sum_{ij}|f_{ij}|^2
 =\frac{\Delta x\Delta y}{N}\sum_{pq}|F_{pq}|^2.$$

Thus physical area weights are $\Delta x\Delta y$ and spectral weights
are $\Delta x\Delta y/N$. Weighting each column and bin by its real axial admittance
$Y_z=\varepsilon_0c^2\operatorname{Re}k_z/\omega$ in the medium, with the temporal energy
functional, gives joules ([energy and flux in a medium](flux.md)). Dividing the
one-photon noise amplitude by the square root of the same weight,
$(\Delta x\Delta y/N)\,Y_z/2$, leaves one photon in every admitted column,
polarisation and bin for both noise models.

Reconstruction at the geometric origin uses
$N^{-1}\exp[-i(k_xx_0+k_yy_0)]$ per stored column. This phase is necessary
because the physical axes begin at negative half-widths, and also works
when neither axis samples zero. Each polarisation is transformed separately.
For a real-field run, inverse transverse transforms precede temporal inverse
transforms: conjugate symmetry relates $(k_x,k_y,\omega)$ to
$(-k_x,-k_y,-\omega)$, not to $(k_x,k_y,-\omega)$.

## Independent numerical budgets

Each bound here is derived before the quantity it bounds is measured. Write
$\gamma_m=m\epsilon/(1-m\epsilon)$. The small-grid transform tests budget
$\gamma_{64N}$ times the input 1-norm for round trips and on-axis sums, and
$\gamma_{128N}$ times energy for Parseval. These conservatively cover direct
DFT sums, twiddle evaluation and both FFTs without assuming radix-two sizes.
They certify the discrete transform, independently of continuum sampling.

For $f(x,y)=\exp[-(x^2+y^2)/w^2]$, the infinite Fourier transform is
$\pi w^2\exp[-w^2(k_x^2+k_y^2)/4]$. Poisson summation bounds discretisation
by the sum of its shifted spectral Gaussians at reciprocal lattice points;
finite-window truncation is bounded separately by the omitted spatial
Gaussian lattice sum. Both sums have decreasing positive tails, bounded
by their first term plus the complementary Gaussian integral. No fitted
error from a measured FFT enters this budget. Exact diffraction comparisons
add the nonparaxial phase remainder derived on the Hankel page; they do not
compare exact $k_z$ with a paraxial Gaussian at roundoff tolerance.

Only constant media are supported; varying profiles and the `tabulate`
[propagator mode](../performance/resources.md#memory) are refused. Near grazing incidence, $k_z^2=k^2-k_\perp^2$ is a difference of
nearly equal numbers. Compensated arithmetic reduces cancellation error, but
cannot remove the physical singularity at $k_z=0$.

The [arithmetic derivation](../../dev/accuracy.md#10-free-space-tabulated-step-index-and-rectangular-modes)
gives the independently derived uncertainty $B_q$ in the real part of this
difference $q$,

$$ B_q=8\epsilon |\operatorname{Re}q|+64\epsilon^2 S+32\eta, $$

with $\epsilon=2^{-52}$, $S$ the sum of the magnitudes of the high parts of $k^2$ and
$k_\perp^2$, and $\eta$ the smallest binary64 subnormal. Its sign is certified when
$|\operatorname{Re}q|>B_q$. This is an
arithmetic bound for the supplied coefficients, not an uncertainty in material
data or a certificate of spatial convergence.

Translation emits {py:class}`~solprop.exceptions.GrazingIncidenceWarning` once if
an active frequency/transverse pair remains inside this uncertainty. The search
includes all tied high parts, requires no spatial-frequency table, and identifies
one affected bin and its signed difference and uncertainty in m^-2. It evaluates
grid coefficients, including unoccupied bins; it does not measure field strength.
Propagation continues using the computed sign and the nonpositive clamp.
Python warnings-as-errors settings can promote this diagnostic to an exception.

Accurate arithmetic does not remove the lossless $1/k_z$ singularity at exact
cutoff. An exact zero remains in the documented evanescent clamp and is warned;
no additional angular filter or physical regularisation is introduced. A field with
significant power at exact grazing incidence is outside this model, and treating it
would need a separate derivation. Cartesian runs record the arithmetic in their
configuration (`spatial_arithmetic`), so a checkpoint written with a different
arithmetic — a single rounded square, say — cannot silently resume with different
coefficients.

## Fluence widths

Widths integrate the signed axial flux over time first, and refuse a fluence that is
negative beyond its rounding budget ([local flux](flux.md#axial-admittance-and-local-flux)).
The scalar second-moment radius
is $\sqrt{2\sum F[(x-\bar x)^2+(y-\bar y)^2]/\sum F}$, where the centroid
uses the same fluence. For uniform rectangular pixels the area cancels.
For interpolated widths, bilinear interpolation supplies the fluence on
concentric rings about that centroid. Radial spacing is at most half the
smaller pixel spacing. The common angular rule has arc spacing at most
that radial spacing at the largest circle that can fit inside the sampled rectangle.
Radial nodes are fixed multiples of the half-pixel spacing; only their outer cutoff
depends on the centroid. Thus a rounded centroid cannot change the whole quadrature
rule and turn an arithmetic perturbation into a sampling change. Only circles inside the rectangle
of sampled points are used; there is no extrapolation or periodic wrapping
of widths. The first crossing of the central averaged fluence divided by
$e^2$ gives the radius; twice the half-maximum radius gives FWHM. Empty,
hollow (axis below the threshold) and uncrossed profiles follow the catalogue's NaN
convention; the thresholds are fractions of the peak of the circular averages.

For a smooth continuum fluence, bilinear interpolation contributes at most
$(\Delta x^2\sup|F_{xx}|+\Delta y^2\sup|F_{yy}|)/8$. The periodic trapezoid
contributes at most $\Delta\theta^2\sup|\partial_\theta^2F|/12$ to the angular
average, and radial linear interpolation at most $h_r^2\sup|F_{rr}|/8$.
A threshold-position budget divides the sum of profile and central-level
errors by a lower bound on the magnitude of the slope near that crossing.
These are continuum sampling errors in addition to arithmetic roundoff;
refining the Cartesian grid must reduce them. The exact discrete moment
has only its reduction-roundoff budget. Comparisons with a continuum moment
also include finite-window and pixel-sampling tails.

On a [line](model.md#the-line-uniform-in-y) (one y sample) the widths are
one-dimensional: the second moment is $2\sqrt{\sum F(x-\bar x)^2/\sum F}$, the
1/e² half-width of a Gaussian fluence, and the threshold widths average the linear
interpolant on the two sides of the centroid at the same half-pixel radii, so the
angular term above is absent.
