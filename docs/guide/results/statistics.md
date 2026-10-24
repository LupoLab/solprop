# Plane statistics

Statistics describe how selected quantities change along propagation without
requiring you to reconstruct every field. Start with energy, peak power and pulse
duration; add band energies, modal or spatial quantities, and plasma diagnostics
when they answer a specific physical question.

The [RDW lesson](../../introduction/lessons/plot_01_rdw.rst) introduces the plots.
This chapter explains selection and interpretation, including basis restrictions
and sampling between planes. The [catalogue](statistics-catalogue.md) lists names,
shapes and units; the [definitions](statistics-definitions.md) give the equations.
A statistic is a measurement of the represented field, not an independent
certificate of its accuracy.

**Which amplitude they are computed from.** The array functions of
{py:mod}`solprop.stats` take FFT-convention spectra of shape `(n_modes, n_pol, n_omega)`.
A backend holds the vacuum-normalised internal amplitude $a$ and, when it collects a
run's statistics, applies the power weight $w = \operatorname{Re} n_\text{eff}$, so that
the numbers are those of the [analytic Poynting flux](power-and-intensity.md):

- `energy`, the windowed energies and the spectral centroid `omega0` multiply the
  internal spectral squared modulus by $w$;
- the temporal peaks and widths use the Hermitian product of the analytic fields formed
  from $a$ and $w a$, and the on-axis statistic the corresponding physical fields;
- a route that overrides the prefactor's index with a constant (the GNLSE, whose
  constant is one) uses that constant as the weight;
- a free-space run weights each transverse column by the axial admittance of the medium
  at the plane, real for energies and complex for local flux, so its spatial statistics
  are the physical axial Poynting flux there ([energy and flux in a medium](../free-space/flux.md)).

A saved modal plane holds the **true-power** amplitude $b = \sqrt{w}\,a$ instead
(`output.fields("fft")` returns it in the Fourier convention). The array functions
applied to it measure $|b|^2$, whose integral over frequency agrees with the flux but
whose broadband cross terms in time differ slightly; [processing](processing.md#the-metrics)
quantifies the difference. The sign convention has no effect on any of these quantities.
The normative statements are [conventions.md](../../dev/conventions.md) §4 and
[kernel-contract.md](../../dev/kernel-contract.md) §8.


## Where they are evaluated, and why not everywhere

Statistics belong to saved planes (see the [design decisions](../../dev/decisions.md)).
Evaluating every statistic on every accepted step would be expensive: the temporal ones
need a full inverse transform for the analytic field, a check of the transverse
reconstruction needs a transform evaluation of its own, and the zero-dispersion
wavelength needs a root search. Together those cost about a fifth of a step, and on a
GPU they can dominate it.

So the field is saved at planes and the statistics are taken there. A run that
wants step-level diagnostics names a subset in
{py:attr}`~solprop.stats.StatsSpec.periodic` with a period, and a backend evaluates that
subset every $N$ accepted steps, inside its own step loop, from the state that step has
just committed. See [between the planes](#between-the-planes) below.

Two things keep that affordable. The *subset* is named, so nothing pays for a statistic
it does not want -- and the catalogue says which statistics may be in it at all, so the
one that costs a root search per call is refused by name rather than quietly made the
run's per-step cost. And the *period* throttles what is left. Measured at $N = 1$ (a row
after every accepted step) with energy, peak power, the pulse duration and the electron
density, the channel costs about **2.5 %** of a step under the Runge–Kutta march on a
shortened four-mode resonant-dispersive-wave capillary and **2.4 %** on a shortened
six-mode hydrogen-filled capillary: the plasma statistics are read off what the step's
last stage left behind, which is the state the step accepted. Under the default
[Adams method](../numerics/adams.md) a step's one evaluation is at its predicted state, not the
accepted one, so the plasma statistics scan the columns afresh, about the cost of a
source evaluation: the channel adds **80 %** to the first case's run at $N = 1$. A
period of ten brings that to about 8 %; the statistics at the save planes are unaffected. Evaluating the power weight of each row (the
$\operatorname{Re} n_{\rm eff}$ factor of the Poynting flux, see
[power and intensity](power-and-intensity.md)) adds about **0.06 ms** per row, under one
percentage point of the first case's step.

## The spec

{py:class}`~solprop.stats.StatsSpec` is the object of the kernel contract's §8: the names
to evaluate, the spectral windows the windowed statistics need, the periodic subset and
its period. It validates on construction — an unknown name, a repeated name, a window
that is not ordered, a periodic statistic the run does not record, all raise
{py:class}`~solprop.exceptions.OutputError` — so a mistake surfaces where the run is set
up rather than at the first save plane.

{py:func}`~solprop.stats.default_spec` returns what a run of each kind records unless
told otherwise; it names no periodic subset, so the channel is off unless asked for.

```{code-block} python
import numpy as np
from solprop.stats import StatsSpec, collect, default_spec

spec = default_spec("multimode", with_profiles=True)
values = collect(spec, grid, plane, modes=modes, profiles=profiles)
values["energy"].shape        # (n_modes, n_pol)
values["peak_power_total"]    # a zero-dimensional array

bands = StatsSpec(
    names=("energy", "energy_window"),
    energy_windows=np.array([[1.2e15, 2.4e15], [2.4e15, 4.8e15]]),   # rad/s
)
```

## Between the planes

Naming a subset in {py:attr}`~solprop.stats.StatsSpec.periodic` turns the channel on:

```{code-block} python
from solprop.stats import StatsSpec

names = ("energy", "peak_power", "fwhm_t")
spec = StatsSpec(names=names, periodic=names, period=10)   # every tenth accepted step
output = prop_capillary(..., stats=spec)

output["step_stats/z"]         # (n_rows,) in m, where each row was taken
output["step_stats/energy"]    # (n_rows, n_modes, n_pol) in J
output["step_stats/counts"]    # rows belonging to each saved plane
```

The rows are stored the way the step history is -- flat columns concatenated over the
whole run, with `step_stats/counts` saying how many belong to each plane -- so a resume
appends and a scan needs no padding. `step_stats/step` indexes the plane's own step
history, which counts rejected attempts too, so it is not `row * period`. A run that
recorded no subset reads back as an **empty channel** rather than an error, so a reader can ask unconditionally; the along-z plots of
{py:mod}`solprop.plotting.statistics` do exactly that, and draw the denser trace when
there is one.

**A row is the plane's own number.** The channel evaluates through the same
{py:meth}`~solprop.backends.protocol.Backend.stats` a save plane uses, on the state the
accepted step committed, so a row that lands on a plane is what that plane records, to
the statistic's rounding. There is one definition of each statistic and no second implementation to drift
from the first.

**The period counts accepted steps, from the load.** A rejected attempt is not one. A
resume restarts the count, so a resumed run's rows are not the uninterrupted run's rows
unless each plane's accepted-step count is a multiple of the period.

**Which statistics may be in the subset** is the `between_planes` column of the catalogue
below. What disqualifies one is cost per step, not difficulty: `zdw` is a root search per
mode, which would dominate the step if paid on every accepted one; `peak_intensity` is
admitted because its on-axis matrix is built once and rescaled with the taper rather than
rebuilt on every call. A spec naming `zdw` in the periodic subset raises
{py:class}`~solprop.exceptions.OutputError` when it is built.

## Which names a basis admits

The mode axis of an ordinary modal run is an orthogonal one- or two-component basis
whose coordinate powers add up to the physical power. A
[general modal guide](../waveguides/general-modal.md) resolves three Cartesian components at its own
quadrature nodes, and its coordinate quantities are not additive in that way. Rather than
redefine the familiar names on a basis they do not describe, the output record says which **capability** produced a run, and every accessor that
capability does not admit is refused by name
({py:mod}`solprop.output.capability`). The gate is applied at **four** doors: at
{py:class}`~solprop.stats.StatsSpec` admission in the run loop, in each backend's
`stats`, in the accessors of {py:mod}`solprop.processing.modal`, and in the
modal-amplitude processing functions {py:func}`~solprop.processing.time.energy` and
{py:func}`~solprop.processing.time.peak_power`. The
last two see only the stored record, and `peak_power(modes="sum")` is
`peak_power_total`, so without that door the number the other three refuse would come
back through a different import.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - capability
     - what it admits
   * - ``modal``
     - An orthogonal one- or two-component basis under its recorded power convention,
       which is every ordinary modal run (GNLSE, capillary, supplied gas-filled modes):
       ``energy``, ``peak_power``, ``peak_intensity`` and every other accessor
       on this page. The general names are refused, because such a run carries
       neither a certified flux metric nor a stored E/H reference.
   * - ``general_modal``
     - Thirteen names: ``physical_energy``, ``spectral_flux``, ``analytic_flux``,
       ``omega0``, ``fwhm_t``,
       ``fwhm_t_outer`` and ``zdw``, and from :py:mod:`solprop.processing.modal` the
       reconstruction accessors ``electric_field``, ``magnetic_field``,
       ``instantaneous_flux`` and ``coordinate_intensity`` together with
       ``channel_labels`` and ``plane_nodes``. ``energy``, ``peak_power``, ``peak_power_total``, their
       windowed forms and ``peak_intensity`` are refused by name, with the
       replacement. The same thirteen are the ``accessors`` list a general modal record
       publishes, so a file says what it admits.
   * - ``spatial``
     - A radial or Cartesian beam, unchanged, and the general names refused.
```

The refusal names the basis and what to use instead, and never redefines the old name:

```{code-block} text
peak_power is unavailable for this nonorthogonal or three-component modal basis; use
analytic_flux or reconstructed E/H with an explicit detector definition. A general modal
guide resolves E_x, E_y and E_z at its own quadrature nodes, and its coordinate
quantities are not additive, so the old name is refused rather than redefined
```

**The two scalar general names are not indexed by channel, and that is the point.** `physical_energy`
is the
same-frequency product of the state with the basis's flux metric, integrated over
frequency; `analytic_flux` is the peak over time of that same **same-frequency
flux**, $\max_t \mathbf a^{\mathsf H}W\mathbf a$. Neither is indexed by channel, because
a coordinate energy and a coordinate power are not additive on a basis that is not
orthonormal. On an identity metric their arithmetic is exactly `sum(energy(...))` and
`peak_power_total`, and their docstrings say so.

**`spectral_flux` is the same quantity as a density in frequency**, in J/(rad/s): the
spectrum analogue of `physical_energy`, $\mathrm dU/\mathrm d\omega$ formed from the
same $\mathbf a^{\mathsf H}W\mathbf a$ rather than integrated over it. It is indexed by
frequency and by nothing else — the channel axis is summed after the whitening, for
the same reason `physical_energy` is a scalar — and `d_omega` times its sum is
`physical_energy`, exactly on every interior bin (see [grids](../setup/grids.md) for the two end
bins the analytic signal does not double, which any grid built here apodises to zero).
It is **admitted but not a default** of the general route: it is `n_omega` wide per
plane where the route's other statistics are a few numbers each, so a run asks for it by
name in its `stats=` spec. For the same reason it is not available *between* save
planes.

**`analytic_flux` is not the cross-section flux**, and the difference is physics rather
than rounding. The physical
$\int \mathrm{Re}(\mathbf E_a \times \mathbf H_a^*)/2 \cdot \hat{\mathbf z}\,\mathrm dA$
of the reconstructed fields carries the guide's **cross-frequency** flux kernel
$K_{ij} = \sum_q w_q [e_x(\omega_i) h_y^*(\omega_j) - e_y(\omega_i) h_x^*(\omega_j)]/2$,
whose diagonal a provider normalises to one and whose off-diagonal it says nothing
about. The two peaks therefore differ, by at most $\sum_{ij}|B_i||B_j||K_{ij}-1|$. For
one HE11 channel of a 1.5 µm strand on 288 nodes carrying a 20 fs pulse they differ by
1.3e-5 of the peak, inside that bound of 3.4e-5. The time integrals agree much more
closely, to 5.0e-7 in the same case, because the cross-frequency terms are an integer
number of cycles in the window and integrate away. Even a positive spectral metric
does not make an instantaneous cross-frequency flux a sum of coordinate powers. For the
flux itself, ask {py:func}`solprop.processing.modal.reconstructed_flux`, which
reconstructs E and H from the guide's own reference and returns the flux as a trace in
time; it has a name of its own because it is not this statistic's quantity.

`omega0`, `fwhm_t` and `fwhm_t_outer` stay admitted, and they **take the metric**, as
the section below describes.

`quadrature_error` is refused on a general guide for a different reason — a provider's
node rule carries no embedded coarse rule to difference, so the transverse-resolution
alarm has no error estimate at all and would be `nan` at every plane. Because
that reason is not the basis, the whitelist lets the name through as the
basis-independent step diagnostic it is and the run loop refuses it; for the same
reason it is not in the accessor list a record publishes. The gas and spatial names —
`electron_density`, the `species_*` family, `pressure`, `density`, `core_radius`,
`beam_radius` and the on-axis ones — carry their own reason too: a guide has no gas
fill and no transverse grid, which is not a statement about its basis.

The time-resolved companions live in {py:mod}`solprop.processing.modal`: the physical
flux, the instantaneous flux of the real fields with its sum-frequency term, and the
per-channel `coordinate_intensity`, which is **not additive** and is never clipped. See
[output](output.md).

## Metric-weighted statistics

Six statistics are functionals of the physical same-frequency flux
$\sum_{mn}\bar a_m W_{mn} a_n$ — its integral (`physical_energy`), its density in
frequency (`spectral_flux`), its peak
(`analytic_flux`), its two widths (`fwhm_t`, `fwhm_t_outer`) and the spectral centroid
of its per-channel share (`omega0`) — and {py:data}`~solprop.stats.METRIC_WEIGHTED_STATISTICS`
names them. Each takes `metric=`, the basis's flux metric $W(\omega)$ of shape
`(n_modes, n_modes, n_omega)`. **`metric=None` is the identity** and is the path an
ordinary modal or spatial run takes, bit for bit the unweighted statistic.

There is **one** weighting, shared by all six. The Hermitian positive-definite square root
$W^{1/2}$ turns the quadratic form into a sum of squares — $\mathbf c = W^{1/2}\mathbf a$
gives $\mathbf c^{\mathsf H}\mathbf c = \mathbf a^{\mathsf H} W\mathbf a$ because
$W^{1/2}$ is Hermitian — so each statistic is the ordinary, unweighted statistic of the
whitened amplitudes and cannot drift from the others. $W^{1/2}$ is the unique positive-definite
square root, so the orthonormal basis it picks depends on $W$ alone, and it reduces to
the identity exactly where $W$ does. On a [general modal guide](../waveguides/general-modal.md) that
declares a unitary change of basis, $W$ is the identity plus the sampled Gram defect, so
these numbers move by that defect and by nothing else.

`omega0` on a declared basis is the **whitened** column's centroid rather than the
coordinate column's, and since $W^{1/2} = I + O(\delta)$ it is the declared channel's to
within the basis's Gram defect and moves by no more. Its empty-column rule is
relative for the same reason: whitening mixes, so a channel nothing was launched into
carries $W^{1/2}_{01}$ times its launched partner rather than an exact zero, and a
column whose density sums to at most one machine epsilon of the largest column's is
`nan` instead of a finite centroid of rounding noise.

`zdw` takes no metric: it is a property of the channels' propagation constants and of no
basis at all. It is **withheld** beside a coupled flow — a declared basis of a split
block, whose rotated channels carry only the block's mean propagation constant — and
asked for by name there it is refused at the driver, the backend and the file
({py:data}`~solprop.stats.WITHHELD_WITH_A_FLOW`,
[general modal propagation](../waveguides/modal-bases.md#a-declared-basis-of-a-block-that-is-only-nearly-degenerate)).

The run's own statistics get $W$ from the backend, which evaluates the
`linear.modal_metric` leaf at each plane through
{py:func}`solprop.translate.coupled.metric_at`. A **file-time** accessor cannot: the output
record holds the metric's identity — its kind, its gauge and its measured defect —
and not its arrays, so {py:func}`solprop.processing.modal.physical_energy` refuses a
record whose metric is not the identity by name rather than returning the unweighted
number, which would be the answer for a different basis. Read the run's own recorded
statistic, or {py:func}`solprop.processing.modal.reconstructed_flux`, which reconstructs
the physical flux from the guide's stored reference and needs no metric at all.

## Far-field apertures and angular regions

A spectrally resolved measurement after a thin sample rarely sees the whole beam: the
light travels a distance $z_m$, or passes a lens of focal length $f$, and part of it
enters a spectrometer through an aperture. Five statistics of a Cartesian or line
free-space run ({py:class}`~solprop.grid.FreeGrid`, {py:class}`~solprop.grid.LineGrid`)
record what such a detector measures, at every saved plane, without storing the field:

| name | what it is |
|---|---|
| `aperture_spectrum` | the spectrum collected through each aperture of an {py:class}`~solprop.stats.ApertureSet` |
| `aperture_on_axis_spectrum` | the on-axis spectral fluence behind each aperture placed in a Fourier plane |
| `region_spectrum` | the spectrum in each box of transverse wavevector of `spectral_regions` |
| `beam_spectrum` | the whole-beam spectral energy density, the axial flux of the medium at the plane |
| `absorbed_power` | the power the spectral, transverse, evanescent and material absorbers remove, per unit length |

Their shapes and units are in the [catalogue](statistics-catalogue.md) and their
equations in the [definitions](statistics-definitions.md#far-field-spectra).
TG-FROG is the canonical use: three beams pass holes at three corners of a square, the
signal emerges towards the fourth, and the trace is the spectrum through that corner,
delay by delay. Concentric centred apertures give an angularly resolved spectrum
(differences are annuli); a hole that no beam should reach is a dark-corner diagnostic of
crosstalk, aliasing or collar scattering.

### The mapping

A stored column $c$ carries the axis values $\mathbf k^{(s)}_c=(k_x,k_y)$ of
`FreeGrid.k_x` and `k_y` in FFT order. In the internal convention the inverse
transverse transform synthesises it with $e^{+i\mathbf k^{(s)}\cdot\mathbf x}$, so at
$\omega>0$ it travels along $(-\mathbf k^{(s)},k_z)$: its **physical** transverse
wavevector, positive for a beam travelling towards $+x$, is
$\boldsymbol\kappa_c=-\mathbf k^{(s)}_c$. The
[crossing-beam lesson](../../introduction/lessons/plot_04_crossing_beams.rst) builds its
$+x$ beam with the internal phase $e^{-i\kappa x}$ for this reason. Every position and
wavevector you give (aperture centres, region edges) is physical.

Phase matching at a planar exit face conserves $\boldsymbol\kappa$, so a component leaves
into vacuum at $\sin\theta=c|\boldsymbol\kappa|/\omega$ whatever the run's medium. In the
far zone of the exit plane it reaches the plane $z_m$ at $z_m\tan\theta$; the
statistics use the paraxial position

$$
\mathbf x\simeq s(\omega)\,\boldsymbol\kappa,\qquad s(\omega)=\frac{z_m c}{\omega},
$$

which is exact in the back focal plane of a lens of focal length $f=z_m$ within paraxial
thin-lens optics. The conditions and the paraxial error are on the
[free-space model page](../free-space/model.md#far-field-and-fourier-plane-statistics).
An aperture of centre $\mathbf X_a$ therefore collects column $c$ at bin $k$ through its
profile evaluated at $\rho_a=|s_a(\omega_k)\boldsymbol\kappa_c-\mathbf X_a|$: a
**chromatic** weight on the stored spectrum, which needs no inverse transform. At fixed
aperture the collected disc in $\boldsymbol\kappa$ grows with frequency, so an aperture
collects one cone of directions; an achromatically tilted beam lands at the same place at
every frequency, while its spot shrinks as $1/\omega$. The weight is exactly zero at and
below zero frequency, where the mapping does not exist, and wherever
$s(\omega)|\boldsymbol\kappa_c|\ge z_m$: those components cannot leave the face, and an
aperture is admitted only inside the paraxial disc $|\mathbf X_a|+D_a/2<z_a$, so it
collects propagating light only.

The lattice spacing in the mask plane is $s(\omega)\Delta\kappa$, largest at the lowest
collected frequency. A hard edge sampled on it has a staircase error of first order in
that spacing over $D$; a smooth edge of width $w$ is resolved once $w\gtrsim2s\Delta\kappa$,
and its error then falls exponentially. One exception: a `tanh` profile is not flat at its
own centre (its slope there is $-\operatorname{sech}^2(r/w)/2w$), so where the beam is
bright at the centre of a `tanh` aperture with $r/w\lesssim5$ the weight has a cone there
and the lattice error falls only as the cube of the spacing (about $10^{-5}$ of the
collected energy at $r/w=2$ and a spacing of a twelfth of the beam's far-field radius).
`rcos` and `supergauss` are flat at the centre.

**Which profile.** `hard` with `amplitude` weighting stays the default: it needs no edge
parameter and is exact for a physical hard stop, at the staircase error above. For a
smooth quadrature weight, prefer `rcos`: it is exactly one inside $r-w$ and zero outside
$r+w$, flat at its centre and has no tail bias. Quadrature `tanh` is admitted only at
$w\le D/8$ ($r/w\ge4$), where the centre slope is about $0.02$ of the case measured
above and its error of order $2	imes10^{-7}$. Amplitude `tanh` with $2\le r/w<4$ is the
one choice that reaches the $10^{-5}$ class: an {py:class}`~solprop.stats.ApertureSet`
built so warns once ({py:class}`~solprop.exceptions.ApertureEdgeWarning`, naming `rcos`)
and runs what was asked.

### Edge profiles and the two weightings

{py:func}`~solprop.spatial.aperture_profile` defines the four profiles once, with
$r=D/2$ and $t=(\rho-r)/w$:

| `profile` | $W(\rho)$ | `edges` |
|---|---|---|
| `hard` | $1$ for $\rho\le r$, else $0$ | `None` |
| `tanh` | $\tfrac12[1-\tanh t]$ | width $w$ in m |
| `rcos` | $\tfrac12[1-\sin(\pi t/2)]$, $t$ clipped to $[-1,1]$: exactly one inside $r-w$ and zero outside $r+w$ | width $w$ in m |
| `supergauss` | $\exp[-(\rho/r)^n]$ | order $n\ge2$ |

`tanh` and `rcos` share the edge symmetry $W(r+u)+W(r-u)=1$. The two **weightings**
answer different questions:

- **`amplitude`** (the default) models a physical mask of amplitude transmission $W$:
  the collected energy carries $W^2$. Standing in for a hard hole it under-counts by
  $\int W(1-W)\,du$ per unit length of edge, $w/2$ for `tanh` and $w/4$ for `rcos`, at
  first order in $w$.
- **`quadrature`** uses a smooth profile once, as a quadrature weight for
  $\int_\text{hole}|E|^2$ over a *hard* hole of diameter $D$, replacing the staircase of
  a sampled indicator by a smooth sum. A symmetric edge's defect $g=W-H$ is odd, so the
  weight is unbiased at zeroth order in one dimension. In two dimensions the measure
  $\rho\,d\rho$ adds an area $\pi\sigma^2$, with $\sigma^2=4m_2w^2$ the variance of the
  edge kernel $-W'$ and $m_2=\int_0^\infty t\,W_1(t)\,dt$ the first moment of the
  unit-width outer tail; evaluating the profile at

  $$
  D_q=2\sqrt{(D/2)^2-4m_2w^2},\qquad
  m_2=\frac{\pi^2}{48}\ (\texttt{tanh}),\quad
  m_2=\frac14-\frac{2}{\pi^2}\ (\texttt{rcos}),
  $$

  restores the hard hole's area exactly for an intensity uniform across the edge (for
  `tanh` up to its exponential tail, a relative area error of $-1.3\times10^{-5}$ at
  $w=D/8$, which is why quadrature weighting admits `tanh` only up to $w=D/8$). The
  residual for a non-uniform intensity is $\pi\sigma^2r\,\partial_\rho\bar I$, of the
  same order as the one-dimensional slope term. {py:func}`~solprop.spatial.quadrature_half_width`
  forms $D_q/2$; {py:data}`~solprop.spatial.EDGE_MOMENTS` holds $m_2$. Quadrature
  weighting is refused with `supergauss`, which has no symmetric edge, and coincides with
  amplitude weighting for `hard`.

The on-axis spectrum behind an aperture is the spectral fluence at the image of the
transverse origin in a unit-magnification $4f$ relay whose Fourier plane, at $f=z_a$,
holds the mask. The amplitude is linear in the field, so $W$ enters it once under both
weightings (at $D_q$ under quadrature weighting); with $W\equiv1$ it is
`on_axis_spectrum`.

### Regions and the beam

A region is a half-open box
$[\kappa^{\rm lo}_x,\kappa^{\rm hi}_x)\times[\kappa^{\rm lo}_y,\kappa^{\rm hi}_y)$ of
physical transverse wavevector, in rad/m, with infinite edges allowed. Membership is
exact, and half-open intervals that partition the line partition the columns: regions
that tile the plane tile the columns, $\kappa=0$ belongs to $[0,\infty)$, and on an even
axis the Nyquist column (stored $-\pi/\Delta x$, physical $+\pi/\Delta x$) belongs to the
non-negative side. A quadrant is one row, `(-inf, 0, -inf, 0)`. A region is not
chromatic: it selects the same $\boldsymbol\kappa$ at every frequency, hence angles that
shrink as $1/\omega$, except for half-planes and quadrants, which select the sign of the
angle at every frequency. A chromatic cone is a centred aperture.

Every spectrum here weights each column by its real axial admittance in the run's
medium, $Y_z=\varepsilon_0c\,\eta$ with $\eta=c\operatorname{Re}k_z/\omega$ and $k_z$ the
root the stages use; $\eta=0$ on inactive and evanescent bins and at $\omega\le0$
([energy and flux in a medium](../free-space/flux.md)). In vacuum $\eta=\cos\theta$, the
obliquity; in a medium $\eta\approx\operatorname{Re}n$ for a paraxial beam.
`beam_spectrum` is the collected spectrum with $W\equiv1$: by the transverse Parseval
identity its sum times `d_omega` is `energy` (with the field grid's documented end-bin
rule, [grids](../setup/grids.md)). The on-axis spectra pair the coherent on-axis field
with its magnetic closure at the same frequency, so each column carries its own
admittance. These are the powers crossing the run's last plane inside the medium; the
transmission of an exit face is not modelled.

`absorbed_power` is four powers in J/m, in the order `spectral`, `transverse`,
`evanescent`, `material` ({py:data}`~solprop.spatial.ABSORBED_CHANNELS`), each a power
rate weighted by the same real admittance as `energy`: the spectral
absorber's rate on every bin, as the stages apply it; the transverse-wavevector rate,
zero until the operator carries one; the evanescent cap
$2\min(|\operatorname{Im}k_z|,\alpha_{\rm ev})$ on active evanescent bins; and
$2\operatorname{Im}k_z$ on active propagating bins, the loss of a complex index (negative
for gain), exactly zero for a real one. Together they are the whole real part of the
separable operator, so their sum is $-dU/dz$ of `energy` under the linear flow. The
physical-space spatial collar and the temporal collar are diagonal in other
representations and are not included (the collar diagnostic watches the first), nor are
nonlinear losses. It is the one far-field statistic admitted
[between the planes](#between-the-planes), where an absorber budget is watched.

### The line, end bins and what is refused

On a line ($n_y=1$) the spectra and powers are per metre of $y$ and the on-axis spectra
stay local. An aperture is a slit of full width $D$ along $x$, its $y$ centre must be
zero, its edges are straight and $D_q=D$; a region selects nothing unless its
$\kappa_y$ interval contains zero, and is refused otherwise. At a field grid's
zero-frequency and Nyquist bins every spectrum follows `spectral_density`'s documented
rule; both bins are zero on every grid solprop builds.

The first four statistics are determined by a saved spectrum and the measure of its
plane, so {py:func}`~solprop.stats.collect` forms them from a plane with
`flux=SpatialFlux.resolve(grid, space, medium, z=plane.z)`; `absorbed_power` needs the
run's linear operator, which a plane does not carry, and `collect` refuses it by name. A spec naming a statistic whose
table it does not carry is refused when it is built, and so are radial and planar-hybrid
geometry, which have no Cartesian transverse wavevector. All five run on `numpy_ref`,
through the reference pass {py:func}`~solprop.spatial.aperture_statistics`, and on
`numba_cpu`, which computes them natively in one compiled pass over the state, threaded
over bands of columns whose sums do not depend on the thread count; it agrees with the
reference within a rounding bound, and decides every hard edge as the reference does.
On `numba_cpu` the same pass also forms `on_axis_spectrum` (an aperture with
$W\equiv1$), and a plane's five statistics cost about as much as `energy`. The native
backend runs the same pass natively, in the same order of sums, and agrees with both
within that bound; a solprop-native build from before this pass is refused them by
name, before it is built.
Lockstep batch members must carry equal tables.

```{code-block} python
import numpy as np
from solprop.stats import ApertureSet, StatsSpec

holes = ApertureSet(
    centres=[[4e-3, 4e-3], [-4e-3, 4e-3], [-4e-3, -4e-3], [4e-3, -4e-3]],  # m
    diameters=[1.5e-3] * 4,
    distances=[0.25] * 4,  # m, exit face to mask plane (or a lens's focal length)
    edges=[40e-6] * 4,
    profile="tanh",
    weighting="quadrature",
    labels=("pump 1", "pump 2", "probe", "signal"),
)
quadrants = np.array([[-np.inf, 0.0, -np.inf, 0.0], [0.0, np.inf, -np.inf, 0.0],
                      [-np.inf, 0.0, 0.0, np.inf], [0.0, np.inf, 0.0, np.inf]])
spec = StatsSpec(
    names=("energy", "beam_spectrum", "aperture_spectrum", "region_spectrum",
           "absorbed_power"),
    apertures=holes, spectral_regions=quadrants,
    periodic=("absorbed_power",), period=10,
)
settings = SimulationSettings(save_planes=planes, stats=spec, ...)
output = prop_free(grid, space, medium, responses, launch, settings=settings)
output["stats/aperture_spectrum"]   # (n_z, 4, n_omega) in J/(rad/s)
```

The [far-field statistics tutorial](../../auto_tutorials/free_space/plot_07_far_field_statistics.rst)
runs all five on a free-space run. The design is
[SEP 23](../../dev/seps/archive/sep-0023-far-field-aperture-spectra.md), and the
medium measure is [SEP 34](../../dev/seps/sep-0034-medium-flux.md).

## What `collect` guarantees

Each statistic is evaluated by the public function of the same name, so there is exactly
one definition of each and no path computes anything twice. Reductions run over the contiguous frequency and time axes, and two evaluations of the
same input agree to rounding ([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)). Every value is a NumPy array — zero-dimensional for a scalar — and its shape
is the `shape` field of its {py:class}`~solprop.stats.StatDefinition`.

A dependency that is missing raises {py:class}`~solprop.exceptions.OutputError` naming
what is wanted: a windowed statistic with no windows, `zdw` with no modes, a profile
statistic with no profile, and the message lists the profiles that were given.

## A worked example

```{code-block} python
import numpy as np
from solprop.grid import EnvelopeGrid
from solprop.output.schema import SavedPlane
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec, collect

grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
field = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)

spec = StatsSpec(
    names=("energy", "energy_window", "peak_power", "fwhm_t"),
    energy_windows=np.array([[-np.inf, 2.4e15], [2.4e15, np.inf]]),
)
values = collect(spec, grid, SavedPlane(z=0.0, field=field))

assert np.isclose(values["energy"].sum(), 1e-6, rtol=1e-12)
assert np.isclose(values["energy_window"].sum(), values["energy"].sum(), rtol=1e-12)
assert np.isclose(values["fwhm_t"], 30e-15, rtol=(grid.dt / 30e-15) ** 2)
```

## Species populations

The [species-statistics catalogue](statistics-catalogue.md#species-populations)
lists the mixture populations, component densities, ionisation rates and energy
inventory a mixture run records, with their definitions.
