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
`FreeGrid(radius_x, n_x, radius_y, n_y, window_factor=0.1)`, x before y. The periodic
half-width is `radius*(1+window_factor)` on each axis; the extra width is the absorbing
collar, so the default adds ten per cent on each side. This differs from
{py:class}`~solprop.grid.RadialGrid`, whose `radius` includes its collar. A flattened
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

{py:class}`~solprop.boundaries.TransverseCollar` is a **safety ramp** in $k_\perp$, set
as `SimulationSettings(transverse_absorber=TransverseCollar(strength, fraction=0.1))`.
With $\rho=\max(|k_x|/k_{x,\max},|k_y|/k_{y,\max})$ its power rate is
$\alpha_\perp=S\,P(\rho)$, zero for $\rho\le1-f$ and rising as a raised cosine to $S$ at
the grid's largest wavenumber; on a {py:class}`~solprop.grid.LineGrid` it acts on $k_x$
alone. The max-norm protects the rectangle the grid defines, since aliasing in $x$ is
decided by $k_x$ alone. The rate is diagonal in the transverse wavenumber, so it enters
the separable operator as an exact rate on the active band, $D\to D-\alpha_\perp/2$, at
the cost of one load and one add per element and with no effect on the noise phase:
whatever steps the run takes, each column falls by $e^{-\alpha_\perp L/2}$ over a
distance $L$. It damps content that reaches the outer band, from cascades or drifting
towards the edge, weakening the products that would fold back with it, and it keeps
rounding from accumulating at the Nyquist edge. It **cannot** remove an alias that
in-band content generates, because the fold happens inside the product's evaluation and
lands where $P=0$; that is the grid's job. The default ten per cent touches only content
that should not be there on a grid sized by the rule: products between $K$ and
$k_{x,\max}$, a four-wave-mixing signal leaving at a larger angle say, are physical and
must not be damped. A strength with $SL_{\mathrm{int}}/2\ge\ln10^3$ removes 60 dB of field
at the edge over an interaction length $L_{\mathrm{int}}$; its upper bound is the
[exponent budget](model.md#exact-longitudinal-operator).

```{code-block} python
from solprop.boundaries import TransverseCollar

settings = dataclasses.replace(
    settings, transverse_absorber=TransverseCollar(strength=2e4, fraction=0.1)
)
```

The real-space [spatial collar](model.md#spatial-collar-and-statistics) and this ramp are
complementary: the first removes light that has walked to the edge of the box and
selects nothing by angle, the second acts on high-angle components wherever they are and
does nothing for a beam leaving the box. The ramp is refused on the radial kind, which
is not periodic, on the hybrid planar kind and on modal runs. It is recorded in the
run's configuration as its two numbers, and only when set. The far-field statistic
`absorbed_power` reports the power it removes as its transverse channel
([far-field statistics](model.md#far-field-and-fourier-plane-statistics)).

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
are $\Delta x\Delta y/N$. Multiplication by $\epsilon_0c/2$ and the temporal energy
functional gives joules. Dividing the one-photon noise amplitude by $\sqrt{\epsilon_0c\Delta x\Delta y/(2N)}$ preserves its
per-column, per-polarisation, per-bin energy for both noise models.

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

Widths integrate intensity over time first. The scalar second-moment radius
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
