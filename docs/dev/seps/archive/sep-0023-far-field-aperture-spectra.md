# SEP 23 — Far-field aperture and angular-region spectra for free-space runs

| | |
|---|---|
| **SEP** | 23 |
| **Title** | Far-field aperture and angular-region spectra for free-space runs |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-16) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-16 |
| **Resolution** | Accepted 2026-10-18 by John Travers after his final review, on the revisions of 2026-10-18 recorded in Part III; Part I is frozen from here |
| **Requires** | [SEP 22](sep-0022-memory-lean-spatial-execution.md) phase 1, for the `omit_default` record rule; if this SEP lands first, its phase 1 introduces that rule |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | phase 1: Opus 5.5 (AI) for John Travers, branch `feature/sep-0023-phase-1`; phase 2: Opus 5.5 (AI) for John Travers, branch `feature/sep-0023-phase-2`; phase 3: Opus 5.5 (AI) for John Travers, branches `feature/sep-0023-phase-3` in solprop and solprop-kokkos; phase 4: Opus 5.5 (AI) for John Travers, branches `feature/sep-0023-phase-4` in solprop and solprop-kokkos |
| **Manual pages** | [Far-field apertures and angular regions](../../../guide/results/statistics.md#far-field-apertures-and-angular-regions); [far-field and Fourier-plane statistics](../../../guide/free-space/model.md#far-field-and-fourier-plane-statistics); [catalogue](../../../guide/results/statistics-catalogue.md); [definitions](../../../guide/results/statistics-definitions.md#far-field-spectra); [tutorial](../../../auto_tutorials/free_space/plot_07_far_field_statistics.rst) |

## Part I — Proposal

### 1. Abstract

Add six statistics to Cartesian and line free-space runs that record what a spectrally
resolved detector behind the sample would measure: the spectrum collected through each of
a set of circular apertures in a plane at distance $z_m$ (a spectrometer entrance, a
pinhole, the holes of a mask), the on-axis spectrum after the same aperture acts in a
Fourier plane, the spectrum in boxes of transverse wavevector, the whole-beam spectrum,
its axial-flux counterpart, and the power the spectral, transverse and evanescent
absorbers remove per unit length. A plane-wave component of transverse wavevector
$\boldsymbol\kappa$ reaches the mask plane at $\mathbf x\simeq z_mc\,\boldsymbol\kappa/\omega$,
so an aperture is a *chromatic* weight on the stored transverse spectrum and needs no
inverse transform. The tables travel on `StatsSpec` as leaf fields, like
`energy_windows`, so the kernel schema does not change; one pass reads the state once per
plane. Success is judged by closed-form profile values and Gaussian far-field fractions,
identities with `energy` and `on_axis_spectrum`, Parseval and tiling to derived rounding
bounds, the absorbed power against the energy decrement, three-backend parity within
summation bounds, exact hard-edge membership, and thread-count agreement on `numba_cpu`
within a rounding bound fixed before measurement.

### 2. Motivation and scope

A spectrally resolved measurement after a thin nonlinear sample rarely sees the whole
beam. The light leaves the sample, travels a distance $z_m$ (or passes a lens of focal
length $f$), and part of it passes an aperture into a spectrometer. In that plane each
plane-wave component has settled at a position fixed by its direction, so the aperture
selects *directions*, and because a fixed direction corresponds to a transverse
wavevector proportional to frequency, it selects a different part of the transverse
spectrum at every frequency.

Transient-grating frequency-resolved optical gating (TG-FROG) is the canonical case
[1, 2]. Three beams pass through holes at three corners of a square and are focused into
the sample, the folded BOXCARS geometry [4]; two of them write a transient grating that
diffracts the third into the direction $\mathbf k_1-\mathbf k_2+\mathbf k_3$, which
emerges towards the fourth corner. A mask passes that corner to a spectrometer, and the
trace is the collected spectrum, delay by delay. The same measured quantity, a spectrum
of a parametrised nonlinear process, is the input of the general retrieval algorithms of
the FROG and d-scan family [3]. Simulating such a trace means propagating a state of
order $768\times768$ transverse samples by 600 frequencies (5.3 GiB) through the sample
for each of hundreds of delays, and keeping, at each delay and save plane, one spectrum
per hole: a few kilobytes. Other uses are the angularly resolved spectrum of a filament's
conical emission (concentric centred apertures, whose differences are annuli), a
dark-corner diagnostic (light reaching a hole that no beam should reach: crosstalk,
aliasing or scattering from the collar), and the spectrum through a slit on a
[line grid](../../../guide/free-space/model.md). Today a user must save every plane's field
and reduce it afterwards, which is prohibitive for gigabyte states, and the statistics
catalogue has no per-frequency transverse reduction: `on_axis_spectrum` reconstructs one
point, and `energy` integrates over frequency.

**In scope.** Cartesian ({py:class}`~solprop.grid.FreeGrid`) and line
({py:class}`~solprop.grid.LineGrid`) free space on field and envelope grids, one or two
polarisation components; circular apertures with four edge profiles in the paraxial
far-field (or Fourier-plane) mapping; boxes of transverse wavevector; the beam and
axial-flux spectra; the absorbed power of the channels diagonal in the stored
$(\boldsymbol\kappa,\omega)$ representation; `numpy_ref`, `collect` on a saved plane,
`numba_cpu` and Kokkos CPU, with GPU qualification later under the standing policy.

**Out of scope.** Radial runs (the mask-plane radius is Cartesian) and the hybrid planar
kind (its columns are slab modes), both refused by name; Fresnel transmission, detector
response and polarisation analysers (Q1); tabulated or non-circular apertures (Q2, Q6);
the exact $\tan\theta$ mapping (Q5); near-field apertures, which need an inverse
transverse transform; a medium-aware normalisation, which belongs to
[SEP 5](../sep-0005-solid-ionisation.md) (Q3). No new dependency, output-schema change or
changed default.

### 3. Context

The [statistics page](../../../guide/results/statistics.md) and the
[free-space model](../../../guide/free-space/model.md) describe today's spatial statistics:
`energy`, `on_axis_intensity`, `peak_intensity`, `beam_radius`, `fwhm_r` and
`on_axis_spectrum`, the default set of `stats.default_spec("free")`. Their host
definitions are in `solprop.spatial` (`spatial_energy`, `on_axis_spectrum`, …), using the
geometry's `spectral_weights` ($\Delta x\Delta y/N$ for the raw transverse DFT,
[Cartesian free space](../../../guide/free-space/cartesian.md)) and
`fourier.spectral_density`. In a run, `backends.spatial.collect_spatial` evaluates them
on `numpy_ref` and also on `numba_cpu`, whose spatial statistics are this same host code
over a borrowed physical-space scratch; `_axis_statistics` forms the on-axis spectrum as
`np.einsum("r,rpw->pw", geometry.on_axis, field)`. `stats.collect(..., space=...)`
evaluates the saved-field ones on a plane after the run.

In solprop-kokkos, `backend.py` maps the spatial names to the flag values 1 to 16 of the
native `spatial_statistics` (`_SPATIAL_STAT_FLAGS`), whose check in `src/general.cpp`
refuses any other bit (`flags & ~31`); its `axis_spectrum` pass gives one work item to
each frequency and loops serially over every column. The Python layer refuses an unknown
statistic by name, and a fresh run collects the entrance plane's statistics before its
first step, so the refusal precedes any propagation.

A `StatDefinition` records `backend` — whether only a backend can form the value because
the saved field does not carry what it needs — and `between_planes`, whether the periodic
channel may evaluate it ([kernel contract](../../kernel-contract.md) §8, ADR-012). The
contract carries the whole request as one leaf, `stats_spec`, a frozen `StatsSpec`
validated against the catalogue; `energy_windows` is a field of that leaf, not a row of
the contract.

[SEP 9](../sep-0009-propagation-statistics.md) proposes beam measurements on modal routes
and leaves the spatial routes' statistics as they are. [SEP 5](../sep-0005-solid-ionisation.md)
proposes a medium-aware axial flux for spatial routes, with admittance
$Y_z=\operatorname{Re}k_z/(\mu_0\omega)$; this SEP reports that quantity's spectrum as a
statistic without changing any normalisation.
[SEP 22](sep-0022-memory-lean-spatial-execution.md), drafted alongside, provides
statistics-only save planes, which make these spectra the only record a long scan keeps,
and a transverse-wavevector absorption rate, `SeparableOperator.kperp_rate`, whose power
the second channel of `absorbed_power` reports.
[SEP 21](sep-0021-absorbing-boundaries.md) proposes cumulative `absorbed_energy`
statistics for the temporal and spectral collars (in J); `absorbed_power`'s spectral channel
is the rate (in J/m) whose integral along $z$ is the spectral one of those. SEP 21
accumulates its energies per accepted step by trapezoidal quadrature, so the two agree
within SEP 21's quadrature bound, not to rounding, when both exist (a verification item
SEP 21 §7 carries once its statistics land). SEP 21
leaves the spectral collar and the transverse collars out of its own scope, so there is no
overlap in what is changed. The roadmap's known issue *The vacuum flux
factor in spatial functionals* applies to every spectrum here, and its *Further offline
elements* item names an aperture filter that could share the profile definition (Q4).
ADR-007 (absorbing boundaries as rates) and ADR-017 (the sign convention) are the
decisions this builds on.

### 4. Theory

SI units and the conventions of [conventions.md](../../conventions.md) throughout. $E_{c,q,k}$
is the stored internal spectrum of column $c$, polarisation $q$ and frequency bin $k$;
$\omega_k$ is the bin's physical angular frequency on either grid kind.

#### 4.1 Which way a stored column travels

In the internal FFT convention a plane wave is $e^{i(\omega t-\mathbf k\cdot\mathbf r)}$.
Column $c=i_x+n_xi_y$ carries the axis values
$\mathbf k^{(s)}_c=(k_x[i_x],k_y[i_y])$ of `FreeGrid.k_x` and `k_y`, in FFT order, and the
inverse transverse DFT synthesises it with $e^{+i\mathbf k^{(s)}\cdot(\mathbf x-\mathbf x_0)}$.
At $\omega>0$ it is therefore the component
$e^{i(\omega t+\mathbf k^{(s)}\cdot\mathbf x-k_zz)}$, which travels along
$(-\mathbf k^{(s)},k_z)$. The physical transverse wavevector, in the optics sense that a
beam travelling towards $+x$ has $\kappa_x>0$, is

$$
\boldsymbol\kappa_c=-\mathbf k^{(s)}_c .
\tag{1}
$$

The [crossing-beam lesson](../../../introduction/lessons/plot_04_crossing_beams.rst) builds its
$+x$ beam with the internal phase $e^{-i\kappa x}$, that is at $k_x^{(s)}=-\kappa$. Every
existing statistic depends on $|\mathbf k^{(s)}|^2$ alone and never met this sign. Every
position and wavevector at the interface below (aperture centres, region edges) is
physical, and the sign lives in one place, the mask-plane radius (4).

#### 4.2 From the transverse spectrum to the mask plane

The light leaves the sample through a planar exit face normal to $z$ into vacuum (air, to
the precision of the mapping). Phase matching at the face conserves $\boldsymbol\kappa$,
so a component's internal transverse wavevector is also its wavevector in vacuum, where
it travels at the angle

$$
\sin\theta=\frac{c\,|\boldsymbol\kappa|}{\omega},
\tag{2}
$$

whatever the medium of the run; only components with $|\boldsymbol\kappa|<\omega/c$
leave the face. Fresnel transmission, which depends on angle and polarisation, is not
modelled, as at every end of a simulation. In the far zone of the exit plane — Fresnel
number $a^2/(\lambda z_m)\ll1$ for a source of transverse size $a$ — the field in the
direction $\hat{\mathbf s}$ is proportional to the angular spectrum at
$\boldsymbol\kappa=(\omega/c)\hat{\mathbf s}_\perp$, by stationary phase [5]. In the plane
$z=z_m$ that direction meets

$$
\mathbf x=z_m\tan\theta\;\hat{\boldsymbol\kappa}
 =\frac{z_m\,\boldsymbol\kappa}{\sqrt{\omega^2/c^2-|\boldsymbol\kappa|^2}} .
$$

The **paraxial step** replaces the vacuum $k_z$ by $\omega/c$:

$$
\mathbf x\simeq s(\omega)\,\boldsymbol\kappa,\qquad s(\omega)=\frac{z_m c}{\omega}
\quad[\mathrm{m^2}] .
\tag{3}
$$

The paraxial position is $z_m\sin\theta$, short of the exact one by the factor
$\cos\theta$: a relative error $1-\cos\theta\approx\theta^2/2$, $1.25\times10^{-3}$ at
50 mrad. Within paraxial thin-lens optics the same relation is exact in the back focal
plane of a lens of focal length $f$, with $z_m\to f$, whatever the distance from the
sample to the lens [6]; the mask may be a far-field plane or a Fourier plane. Under (3)
the components that cannot leave the face, $|\boldsymbol\kappa|\ge\omega/c$, land at
$|\mathbf x|\ge z_m$. Every weight is therefore defined as zero wherever
$s(\omega)|\mathbf k^{(s)}_c|\ge z_m$, whatever its profile — the `tanh` and `supergauss`
tails are unbounded and would otherwise reach those columns, which carry energy inside a
dense medium — so an aperture confined to $|\mathbf x|<z_m$ (section 5) collects
propagating light only. The paraxial map's Jacobian $d^2\kappa=(\omega/cz_m)^2d^2x$ cancels the
far-zone amplitude factor, so the energy behind an amplitude mask $W(\mathbf x)$ is the
transverse Parseval integral of the spectrum weighted by $W(s\boldsymbol\kappa)^2$.

#### 4.3 An aperture is a chromatic weight

For aperture $a$ with centre $\mathbf X_a=(X_a,Y_a)$, diameter $D_a$ and distance $z_a$,
(1) and (3) give the distance of column $c$'s image from the centre at bin $k$:

$$
\rho_a(c,k)=\bigl|s_a(\omega_k)\boldsymbol\kappa_c-\mathbf X_a\bigr|
 =\sqrt{\bigl(s_ak^{(s)}_{x,c}+X_a\bigr)^2+\bigl(s_ak^{(s)}_{y,c}+Y_a\bigr)^2},
 \qquad s_a=\frac{z_ac}{\omega_k} .
\tag{4}
$$

With an edge profile $W_a(\rho)$ (section 4.4), the collected spectral energy density is

$$
S_a(\omega_k)=\frac{\varepsilon_0c}{2}\sum_c w_c\,W_a\!\bigl(\rho_a(c,k)\bigr)^p
 \sum_q\operatorname{sd}(E_{c,q,k}),
\tag{5}
$$

in J/(rad/s), with $w_c$ the geometry's `spectral_weights`, $\operatorname{sd}$ the
`fourier.spectral_density` prefactor ($2\delta t^2/\pi$ on a field grid and
$\delta t^2/2\pi$ on an envelope grid, times $|E|^2$), polarisations summed after squaring,
and $p=2$ (amplitude weighting) or $p=1$ (quadrature weighting, section 4.5). The
discrete Parseval identity of the transverse DFT makes (5) exact on the grid: with
$W\equiv1$ it is the beam spectrum (9), and $\delta\omega\sum_kS(\omega_k)$ is the
`energy` statistic. The normalisation is solprop's vacuum flux functional, with the
recorded caveat for dense media (section 4.8, Q3).

At fixed aperture the collected disc in $\boldsymbol\kappa$ has radius
$D_a\omega/(2z_ac)$ and centre $\mathbf X_a\omega/(z_ac)$: it grows with frequency, so the
aperture collects one cone of directions. An achromatically tilted beam,
$\boldsymbol\kappa=(\omega/c)\sin\theta\,\hat{\mathbf x}$ as in the lesson, lands at
$z_a\sin\theta$ at every frequency. The lattice spacing in the mask plane is
$\Delta x_m(\omega)=s(\omega)\Delta\kappa$, largest at the lowest collected frequency; a
hard edge sampled on that lattice has a staircase error of first order in
$\Delta x_m/D$, and a smooth edge of width $w$ is resolved when $w\gtrsim2\Delta x_m$.

#### 4.4 Edge profiles

With $r=D/2$ and $\rho\ge0$, the profiles are defined once, in
`spatial.aperture_profile(radius, half_width, edge, kind)`:

| `kind` | $W(\rho)$ | `edge` | $W(0)$, $W(r)$, $W(\rho\to\infty)$ |
|---|---|---|---|
| `hard` | $1$ for $\rho\le r$, else $0$ | not used | $1$, $1$, $0$ |
| `tanh` | $\tfrac12[1-\tanh((\rho-r)/w)]$ | width $w$ in m | $\tfrac12[1+\tanh(r/w)]$, $\tfrac12$, $\sim e^{-2(\rho-r)/w}$ |
| `rcos` | $\tfrac12[1-\sin(\pi t/2)]$, $t=\operatorname{clip}((\rho-r)/w,-1,1)$ | width $w$ in m | $1$, $\tfrac12$, $0$ beyond $r+w$ |
| `supergauss` | $\exp[-(\rho/r)^n]$ | order $n\ge2$ | $1$, $e^{-1}$, $0$ |

The raised cosine has compact support: it is exactly one inside $r-w$, exactly zero
outside $r+w$, monotone and continuously differentiable, so a pass may skip the sine
outside the edge. `tanh` and `rcos` share the **edge symmetry**
$W(r+u)+W(r-u)=1$ for $|u|\le r$, on which section 4.5 rests; `supergauss` has
none, and $n=2$ is a Gaussian amplitude mask.

#### 4.5 Amplitude and quadrature weighting

The two weightings answer different questions. **Amplitude** weighting models a physical
mask of amplitude transmission $W$: the field is multiplied by $W$ and the collected energy
carries $W^2$, $p=2$ in (5). **Quadrature** weighting uses a smooth profile once, $p=1$, as
a quadrature weight for $\int_{\text{hole}}|E|^2$ over a *hard* hole of diameter $D$,
replacing the staircase of a sampled indicator by a smooth sum.

Let $u$ run across one edge and $H$ be the hard indicator. For a symmetric edge the
defect $g=W-H$ is odd, so $\int g\,du=0$ and $\int u\,g\,du=\sigma^2/2$, where
$\sigma^2=\int u^2(-W')\,du$ is the variance of the edge kernel $-W'$. For a smooth
intensity $f$,

$$
\int W f\,du-\int Hf\,du=\tfrac12\sigma^2f'(0)+O(w^4) .
$$

A weight used once is unbiased at zeroth order in $w$. Its square is not symmetric about
the edge:
$\int(W^2-H)\,du=-\int W(1-W)\,du$, which is $-w/2$ for `tanh` and $-w/4$ for `rcos`, so
$W^2$ under-counts a hard hole by $f(0)\,w/2$ (or $w/4$) per unit length of edge, at first
order. For a physical soft mask that is the right answer; standing in for a hard hole it
is a bias.

In two dimensions the measure $\rho\,d\rho$ adds one order: for an intensity uniform across
the edge, $\int W(\rho-r)\,2\pi\rho\,d\rho=\pi(r^2+\sigma^2)$ exactly (for `rcos` when
$r>w$; for `tanh` up to its exponential tail, which leaves a relative area error of
$-3.5\times10^{-3}$ at $w=D/4$, $-1.3\times10^{-5}$ at $w=D/8$ and $10^{-11}$ at $w=D/20$
by numerical quadrature, so quadrature weighting admits `tanh` only up to $w=D/8$,
section 5). Evaluating the profile at the corrected diameter

$$
D_q=2\sqrt{(D/2)^2-4m_2w^2},\qquad m_2=\int_0^\infty t\,W_1(t)\,dt=\frac{\sigma^2}{4w^2},
\tag{6}
$$

with $W_1$ the unit-width outer tail, restores the hard hole's area exactly. The values are
$m_2=\pi^2/48\approx0.2056$ for `tanh` and $m_2=1/4-2/\pi^2\approx0.04736$ for `rcos`;
`hard` needs no correction. The residual for a non-uniform intensity is
$\pi\sigma^2r\,\partial_\rho\bar I$, of the same order as the one-dimensional slope term,
with $\bar I$ the azimuthal mean. On a line the edges are straight and $D_q=D$. These
values and the exactness of (6) for a uniform intensity were checked by numerical
quadrature while drafting (residuals below $2\times10^{-10}$ at $w=r/10$). Quadrature
weighting is refused with `supergauss`, which has no symmetric edge, and coincides with
amplitude weighting for `hard`, where $W^2=W$.

#### 4.6 The on-axis spectrum through an aperture

With $\alpha_c$ the geometry's existing `on_axis` weights,
$N^{-1}e^{-i\mathbf k^{(s)}_c\cdot\mathbf x_0}$,

$$
A_{a,q}(\omega_k)=\sum_c\alpha_c\,W_a\!\bigl(\rho_a(c,k)\bigr)E_{c,q,k},\qquad
S^{\mathrm{ax}}_a(\omega_k)=\frac{\varepsilon_0c}{2}\sum_q\operatorname{sd}\bigl(A_{a,q}\bigr),
\tag{7}
$$

in J s/m². This is the spectral fluence at the image of the transverse origin in a
unit-magnification $4f$ relay whose Fourier plane, at focal length $f=z_a$, holds the
mask. $W$ enters once under both weightings, because the amplitude is linear in the field;
under quadrature weighting the profile is evaluated at $D_q$, by the same edge argument.
With $W\equiv1$, (7) is `on_axis_spectrum`. Mathematically $\alpha_c=\pm1/N$ on even and
odd grids alike, but the stored vector carries the rounding of the exponential, so it is
multiplied as stored, as a complex number.

#### 4.7 Regions of transverse wavevector

A region is a half-open box
$R_j=[\kappa^{\rm lo}_x,\kappa^{\rm hi}_x)\times[\kappa^{\rm lo}_y,\kappa^{\rm hi}_y)$
in rad/m, with infinite edges allowed, and

$$
S^R_j(\omega_k)=\frac{\varepsilon_0c}{2}\sum_{\boldsymbol\kappa_c\in R_j}w_c
 \sum_q\operatorname{sd}(E_{c,q,k}) .
\tag{8}
$$

Membership is exact: it compares $-\mathbf k^{(s)}_c$, an exact negation, with the edges.
Half-open intervals that partition the real line partition the columns, so regions that
tile the plane tile the columns: $\kappa=0$ belongs to $[0,\infty)$, and on an even axis
the Nyquist column (stored $-\pi/\Delta x$, physical $+\pi/\Delta x$) to the non-negative
side. A quadrant is one row, $(-\infty,0,-\infty,0)$. A region is not chromatic: it
selects the same $\boldsymbol\kappa$ at every frequency, hence angles that shrink as
$1/\omega$. Regions bounded only by zero and infinite edges (half-planes, quadrants) are
the exception, being defined by the sign of $\boldsymbol\kappa$, which is the sign of the
angle at every frequency. A chromatic cone is a centred aperture.

#### 4.8 The beam and axial-flux spectra

The beam spectrum is (5) with $W\equiv1$,

$$
S^{\rm beam}(\omega_k)=\frac{\varepsilon_0c}{2}\sum_cw_c\sum_q\operatorname{sd}(E_{c,q,k}).
\tag{9}
$$

The axial-flux spectrum weights it by the dimensionless obliquity factor

$$
O(c,k)=\frac{\operatorname{Re}k_z(c,k)}{\mu_0\omega_k\,\varepsilon_0c}
 =\frac{c\operatorname{Re}k_z(c,k)}{\omega_k},\qquad
S^{P}(\omega_k)=\frac{\varepsilon_0c}{2}\sum_cw_c\,O(c,k)\sum_q\operatorname{sd}(E_{c,q,k}),
\tag{10}
$$

with $k_z$ the principal root of the compensated square
$q_z=(k^2-|\mathbf k^{(s)}|^2)+(k^2_{\rm low}-|\mathbf k^{(s)}|^2_{\rm low})$ of the run's
`translate.spatial.SeparableOperator`, the root the stages use, and $O=0$ on inactive and
non-propagating bins ($\operatorname{Re}q_z\le0$, the stages' test) and at
$\omega_k\le0$. Because $\mu_0\varepsilon_0c^2=1$, $\varepsilon_0cO$ is SEP 5's admittance
$Y_z$, and $S^P$ is the spectral density of the scalar-equation axial-flux energy that
SEP 5 proposes. It is written as the vacuum-normalised beam spectrum times $O$ so that it
shares (9)'s arithmetic and normalisation, the ratio $S^P/S^{\rm beam}$ is directly the
beam-averaged obliquity, and the paraxial vacuum limit $O\to1$ is visible; multiplying
$O$ back by $\varepsilon_0c$ recovers the raw surrogate
$\sum\operatorname{Re}(k_z)|E|^2/(2\mu_0\omega)$. In vacuum $O=\cos\theta\le1$, equal to
one only at $\boldsymbol\kappa=0$; for a Gaussian beam of waist $w_0$,
$S^P/S^{\rm beam}=1-(kw_0)^{-2}-(kw_0)^{-4}-3(kw_0)^{-6}-\dots$ with $k=\omega/c$, an
asymptotic series of one-signed terms (the exact form is in section 7, item 8). In a
medium, $O\approx\operatorname{Re}n(\omega)$ for a paraxial beam: this is the factor by
which the vacuum functionals of (5), (8) and (9) are low (Q3).

#### 4.9 Power removed by the absorbers

The real part of the separable linear operator damps each stored element at a power rate
that is a sum of channels. With the exact energy weights of `fourier.energy_omega`,
$\nu_k=(\delta t/n_t)\times\{1,4,\dots,4,1\}$ on a field grid and $\delta t/n_t$ on an
envelope grid,

$$
P_j=\frac{\varepsilon_0c}{2}\sum_cw_c\sum_q\sum_k\nu_k\,r_j(c,k)\,|E_{c,q,k}|^2,
\qquad j\in(\text{spectral},\ \text{transverse},\ \text{evanescent},\ \text{material}),
\tag{11}
$$

in J/m, where $r_{\rm spectral}=\alpha_{\rm abs}(\omega_k)$ is the operator's
`spectral_rate`, applied on every bin as the stages apply it; $r_{\rm transverse}$ is
SEP 22's `kperp_rate`, zero without it; and
$r_{\rm evanescent}=2\min(|\operatorname{Im}k_z|,\alpha_{\rm ev})$ on active
non-propagating bins, with $\alpha_{\rm ev}$ the evanescent field cap
(`spatial.EVANESCENT_RATE`, or SEP 22's `evanescent_rate` setting); and
$r_{\rm material}=2\operatorname{Im}k_z$ on active propagating bins, the loss (or, with
the opposite sign, the gain) of the medium's complex index, exactly zero for a real index.
Then $-\sum_jP_j$ is exactly the derivative of the `energy` statistic along $z$ under the
linear flow restricted to these channels, which are the whole of the operator's real part.
Excluded are the physical-space spatial collar and the temporal collar, which are diagonal
in other representations (the collar's share is monitored by the existing collar
diagnostic, `backends.spatial.spatial_collar_fraction`), and nonlinear losses. The four
channels are fixed from the first release, so the statistic's shape never changes.

#### 4.10 Zero frequency, end bins and the line

The factor $s(\omega)$ exists only for $\omega>0$: $W=0$ and $O=0$ at and below zero
frequency, so a field grid's zero-frequency bin adds nothing to (5), (7) and (10), active
or not. At a field grid's zero-frequency and Nyquist bins every spectrum follows
`spectral_density`'s documented rule (an overcount, as the analytic signal does not double
them); both bins are zero on every grid solprop builds. On a line ($n_y=1$, $k_y=0$) the
spectra (5), (8)–(10) are per metre of $y$ and the absorbed powers in J/m², while (7)
stays local; an aperture's $y$ centre must be zero, and (4) reduces to
$|s_ak^{(s)}_x+X_a|$, a slit of full width $D$ along $x$ with straight edges and $D_q=D$.
A region selects nothing on a line unless its $\kappa_y$ interval contains zero.

### 5. Proposed interface

Names are a proposal. A frozen, identity-compared dataclass in `solprop.stats`:

```text
ApertureSet(
    centres,           # (n_apertures, 2) m, mask-plane (x, y), physical axes
    diameters,         # (n_apertures,) m, finite and positive
    distances,         # (n_apertures,) m, exit face to mask plane, or a lens's f
    edges=None,        # (n_apertures,): m for tanh and rcos, order >= 2 for supergauss;
                       # None for hard
    profile="hard",    # "hard" | "tanh" | "rcos" | "supergauss", shared by the set
    weighting="amplitude",  # "amplitude" (W^2) | "quadrature" (W once), shared
    labels=None,       # n_apertures unique non-empty strings; default "0", "1", ...
)
```

Validation, raising `OutputError` by name: one to 16 apertures; every array finite and of
the stated shape; positive diameters and distances; for `rcos` $0<w\le D/4$, and for
`tanh` $0<w\le D/4$ under amplitude weighting and $0<w\le D/8$ under quadrature weighting,
which keeps the edge clear of the centre, the radicand of (6) positive and the `tanh`
tail's area bias below $2\times10^{-5}$ (section 4.5);
for `supergauss` an order of at least 2; `edges=None` with `hard` (a value would be
silently unused); quadrature weighting refused with `supergauss`; and every aperture inside
the paraxial disc, $|\mathbf X_a|+D_a/2\,(+w_a$ for `rcos`$)<z_a$ (section 4.2). Arrays
are owned, read-only copies, as `energy_windows` is; labels are recorded with the spec in
the run's configuration and name the rows in the manual's plots.

`StatsSpec` gains two fields: `apertures: ApertureSet | None = None` and
`spectral_regions`, shape `(n_regions, 4)` in rad/m, rows
$(\kappa_x^{\rm lo},\kappa_x^{\rm hi},\kappa_y^{\rm lo},\kappa_y^{\rm hi})$, empty by
default, at most 16 rows, no NaN, `lo < hi` on both axes, infinite edges allowed. A spec
naming a statistic that needs a table it does not carry is refused at construction. The
catalogue gains six entries (new axis names `n_apertures` and `n_regions` join
conventions.md §6):

| name | shape | units | requires | `backend` | `between_planes` |
|---|---|---|---|---|---|
| `aperture_spectrum` | `(n_apertures, n_omega)` | J/(rad/s) | apertures, spatial geometry | no | no |
| `aperture_on_axis_spectrum` | `(n_apertures, n_omega)` | J s/m² | apertures, spatial geometry | no | no |
| `region_spectrum` | `(n_regions, n_omega)` | J/(rad/s) | regions, spatial geometry | no | no |
| `beam_spectrum` | `(n_omega,)` | J/(rad/s) | spatial geometry | no | no |
| `poynting_spectrum` | `(n_omega,)` | J/(rad/s) | spatial geometry, separable operator | yes | no |
| `absorbed_power` | `(4,)` | J/m | spatial geometry, separable operator | yes | yes |

`backend` keeps its catalogue meaning: the first four are determined by a saved spectrum
and the geometry, exactly like `on_axis_spectrum`, so `collect` forms them from a plane;
the last two need the linear operator, which a saved plane does not carry, so `collect`
refuses them by name. `between_planes` is false for the frequency-resolved arrays, as for
the existing `on_axis_spectrum` and `spectral_flux`: a row per accepted step would multiply
the step-statistics storage by `n_omega`. The four absorbed powers are scalars whose cost
is one read of the state, and the periodic channel is where an absorber budget is watched.

```text
holes = ApertureSet(
    centres=[[4e-3, 4e-3], [-4e-3, 4e-3], [-4e-3, -4e-3], [4e-3, -4e-3]],
    diameters=[1.5e-3] * 4, distances=[0.25] * 4, edges=[40e-6] * 4,
    profile="tanh", weighting="quadrature",
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
output["stats/aperture_spectrum"]      # (n_z, 4, n_omega) J/(rad/s)
```

The statistics compose with `prop_free`, `Simulation.from_free`, scans and SEP 22's
statistics-only planes like any other; `collect(spec, grid, plane, space=FreeGrid(...))`
re-analyses a saved Cartesian plane with the first four. Lockstep batch members must carry
equal tables. Refused by name: radial and planar-hybrid geometry; a line with a non-zero
$y$ centre or a region excluding $\kappa_y=0$; `poynting_spectrum` and `absorbed_power`
without a separable operator; a general modal guide (the existing gate, since the names
require spatial geometry); a backend that does not implement a name.

### 6. Implementation strategies

**Layers and the seam.** The profiles, `ApertureSet`, the `StatsSpec` fields and the
catalogue belong to the physics layer (`spatial.py`, `stats.py`). One translation helper,
`translate.spatial.aperture_tables(spec, grids, geometry)`, forms the derived constants
once — each aperture's half-width ($D/2$ or $D_q/2$), reciprocal edge width, order,
profile code and exponent $p$, and the scale table $s_a(\omega_k)$, shape
`(n_apertures, n_omega)`, zero at $\omega_k\le0$ — and every backend calls it at
construction, so all of them use the same $s_a$ to the bit. The kernel contract is
unchanged: the tables are fields of the `stats_spec` leaf, as `energy_windows` is, read
only when a name that uses them is requested, so a backend that does not know a name
refuses it rather than misreading a table. `SCHEMA_VERSION` stays; the §8 row's text gains
the two tables.

**One radius, one root.** The radius (4) is evaluated as written — `sqrt` of a sum of
squares, not `hypot`, which is not correctly rounded and differs between libraries — and
without floating-point contraction (one of the protected helpers that opt out of
contraction under [SEP 25](sep-0025-rounding-level-reproducibility.md)). Every operation
is then a correctly rounded IEEE operation on identical inputs, and a hard edge decides
identically on every backend, an exact hard-mask contract;
otherwise a column within an ulp of a hard edge could fall on different sides on two
backends, a difference no summation bound covers. $\operatorname{Re}k_z$, the propagating
test and the evanescent clamp are factored out of `separable_values` (a helper returning
the root and the test) and out of the bodies `spatial_element` and `longitudinal_rate` (a
body the stage path then calls), so the statistics read the root the stages use; the
stages compute the same expressions, so their results agree with today's within a
rounding bound fixed before measurement.

**numpy_ref and `collect`.** A reference `aperture_statistics` in the physics layer,
`solprop.spatial`, beside `on_axis_spectrum` — the profile, the $s_a$ table and the
reduction live there, because `backends/spatial.py` imports `solprop.stats` (a reduction
placed there would close an import cycle) and the driver reaches a backend only through
the seam ([architecture](../../architecture.md) §3) — called from `collect_spatial`, walks
the columns in chunks of `transform.node_chunk`,
forms $\sum_q\operatorname{sd}(E)$ once per chunk, accumulates each requested slot, and
adds chunk partials in ascending order. The on-axis amplitude keeps the existing
contraction, `np.einsum("r,rpw->pw", on_axis[rows], W[:, None, :] * field[rows])`: with
$W\equiv1$ the product is the field exactly (a multiplication by one), so the result
agrees with `on_axis_spectrum` within the summation bound, with one chunk or across chunks
(a drafting check found NumPy's `einsum` changing at rounding level under re-chunking,
so no exact identity is claimed). `stats.collect` calls the
same function for the four saved-field names with `space=FreeGrid`, and refuses
`poynting_spectrum` and `absorbed_power` inside `_spatial_statistic`, which every name
reaches before the `backend` check when `space` is given. Because `numba_cpu`
evaluates spatial statistics through `collect_spatial`, phase 1 delivers it too.

**numba_cpu.** Three bodies join `define_bodies`: `mask_plane_radius` (4),
`aperture_weight` (an integer profile code, with the raised cosine's early-out) and
`region_member` (the half-open test on $-\mathbf k^{(s)}$). A two-stage kernel in
`numba_cpu/spatial.py` replaces the host path. Stage one runs `prange` over contiguous
bands of columns — whole rows on a plane, blocks of $x$ samples on a line — whose count,
$\min(n_{\rm units},\lfloor B/(8n_{\rm slots}n_\omega)\rfloor)$ with a module constant
$B=128$ MiB, is fixed by the shapes and never by the thread count. Each band writes its
row of partials, shape `(n_bands, n_slots, n_omega)`, in the slot order
`[beam, poynting, absorbed × 4, regions…, per aperture (collected, Re and Im of the
on-axis amplitude per polarisation)]`, present slots only: at most
$n_{\rm slots}=6+n_{\rm regions}+n_{\rm apertures}(1+2n_{\rm pol})$, plus $2n_{\rm pol}$
for `on_axis_spectrum`. Columns run in ascending order with frequency innermost; stage two
sums the bands serially in ascending order (a fixed, shape-determined order), and a small finishing step applies $\varepsilon_0c/2$, the
spectral prefactor and $|A|^2$. The complex `on_axis` is multiplied as complex, and
$\operatorname{Re}k_z$ is recomputed per element by the shared body, never stored. A
`shared_mask` path, taken when every aperture shares centre and distance, computes the
radius once per element: the same strict expression on the same inputs, so its hard-edge
decisions are the general path's exactly and its sums agree within a rounding bound.
`on_axis_spectrum` becomes the $W\equiv1$ slot of the pass, which makes the identity of
section 7 hold to rounding by construction. The partials and the scale table are
allocated at construction from the spec, named in `memory_ledger()`, and reused by the
periodic channel.

**Kokkos CPU.** A native `aperture_statistics(flags, apertures, profile_code, exponent,
regions, intensity_factor)` beside `spatial_statistics`, with its own flag mask (values
from 32; 1 to 16 are `spatial_statistics`') and an argument check refusing other bits.
`Problem` gains `spatial_k_x` and `spatial_k_y`, packed from `geometry.spectral_axes`;
`apertures` carries the host-computed constants and scale table; the bindings expose the
method as they expose `spatial_statistics`; Python packs and unpacks in a fixed order
(beam, Poynting, absorbed, regions, collected, on-axis) with the existing consumed-count
check, and $|A|^2$ is formed natively so only `n_omega` values per aperture cross. A team
policy over the same bands, each team's threads over frequency and each thread looping its
band's columns in ascending order, associates every sum as `numba_cpu` does, so agreement
rests on summation order; the radius is compiled without contraction (a flag for that
unit, or round-to-nearest intrinsics on a device). Folding the existing `axis_spectrum`
into the pass as the $W\equiv1$ slot is recommended: one work item per frequency, each
looping over all $n_xn_y$ columns, is 600 work items at the size above and does not scale
on a device. The fold also adopts the host's rule at the zero-frequency and Nyquist bins,
which the native pass weights by one rather than four today; the bins are zero on
solprop's grids, so no value changes. On a device the slot accumulators live in
registers, and a large set may run in groups that re-read the state (measured in
phase 3). The partials join the native ledger beside `spatial_statistics_partials`.

**Cost.** A pass reads the state once, $16n_xn_yn_{\rm pol}n_\omega$ bytes (5.66 GB at
$768\times768\times600$). Each aperture adds about 40 flops per element and, for `tanh` or
`supergauss`, one transcendental (none for `hard`; a sine only inside the edge for
`rcos`): six smooth apertures cost a few hundred flops and $2\times10^9$ transcendentals at
that size, transcendental-bound on a CPU and between memory- and compute-bound on a
device. A Runge–Kutta step makes several state-sized passes and transforms per stage, so a
pass is estimated at a few per cent of a step, paid at save planes and periodic rows only;
the estimate is to be measured. The partials at that size, six apertures, one polarisation
and no regions (24 slots, one band per row), are $768\times24\times600\times8$ B
$=88.5$ MB (95.8 MB with `on_axis_spectrum`), and never exceed $B$. Benchmarks use
`bench_stats.py` on a ten-step Cartesian window ([benchmarks.md](../../benchmarks.md)).

| Option | For | Against | Choice |
|---|---|---|---|
| Save every field, reduce afterwards | No new code | Storage and transfer prohibitive for gigabyte states | Status quo |
| Host reduction on every backend | One implementation | A device-to-host copy of the state per plane on Kokkos | Phase 1 only |
| Fused banded pass per backend | One state read, deterministic, device-resident | Three implementations | **Recommended** |
| Physical-space masks | Near-field apertures | An inverse transform per plane; a different measurement | Out of scope |

No new dependency is needed.

### 7. Verification and validation

Each tolerance below is set by a stated bound and asserted positive.

1. **Profiles.** Closed-form values at $\rho=0$, $D/2$ and large $\rho$; the raised
   cosine's compact support and monotonicity (`hypothesis`); the edge symmetry to rounding;
   $m_2$ against SciPy quadrature to $10^{-12}$.
2. **Direction.** A beam launched towards $+x$ as in the crossing-beam lesson is collected
   by an aperture at $+X$ and not by its mirror image (bounded by the Gaussian tail); an
   achromatic tilt's image stays at $z_m\sin\theta$ at every frequency while its spot
   shrinks as $1/\omega$, so a hole wider than the spot at the lowest frequency collects
   it fully at every frequency; a fixed-$\kappa$ launch moves as $1/\omega$.
3. **$W\equiv1$.** On a grid whose largest $|\boldsymbol\kappa|$ is below $\omega_{\min}/c$,
   a centred hard aperture enclosing every image gives `aperture_spectrum` equal to
   `beam_spectrum` and `aperture_on_axis_spectrum` equal to `on_axis_spectrum` on each
   backend within a rounding bound fixed before measurement (the summation bound); every
   weight is exactly one.
4. **Gaussian far field.** A waist-plane Gaussian beam gives the collected fraction
   $1-\exp[-D^2/(2w_{ff}^2)]$, $w_{ff}=2z_m/(kw_0)$, through a centred hard aperture. The
   closed form is exact for the statistic's own (paraxial) mapping, so the tolerance is the
   edge-sampling error, bounded by the energy of the columns within one lattice diagonal of
   the edge, plus the Poisson and window bounds of the
   [Cartesian page](../../../guide/free-space/cartesian.md). Smooth profiles are compared with
   the one-dimensional integral of the same profile (SciPy quadrature, independent of the
   lattice); under refinement the hard error falls at first order and erratically, the
   smooth error quickly once $w\ge2\Delta x_m$. The paraxial map's own physical error,
   $O(\theta^2)$, is documented, not tested.
5. **Quadrature unbiasedness.** A spectrum uniform across the edge region collects the hard
   area to lattice error, plus the `tanh` tail's residual of section 4.5, with $D_q$,
   misses it by $\pi\sigma^2$ without the correction, and by the first-order term
   (section 4.5) under $W^2$.
6. **Parseval.** $\delta\omega\sum_kS^{\rm beam}_k$ equals `energy` within $2\gamma_m$ of the
   total, $m$ the number of summed terms plus their roundings, with
   $\gamma_m=m\epsilon/(1-m\epsilon)$; a fixture with non-zero end bins checks the documented
   overcount instead.
7. **Regions.** Four quadrants sum to the beam spectrum within the same bound; membership of
   columns exactly on an edge, at $\kappa=0$ and at Nyquist is as section 4.7 states.
8. **Axial flux.** On a vacuum fixture $S^P/S^{\rm beam}$ matches the exact paraxial
   closed form, the mean of $\sqrt{1-u}$ over $u$ exponentially distributed with mean
   $2/(kw_0)^2$ (by quadrature, or its Dawson-function form), to the lattice and rounding
   bound; the asymptotic series $1-(kw_0)^{-2}-(kw_0)^{-4}-3(kw_0)^{-6}-\dots$ has
   one-signed terms, so no single term bounds its remainder (1.3 times the third term at
   $kw_0=5$) and it is not the test's reference. The ratio lies strictly below one for any
   transverse content in vacuum; in a gas at $kw_0\gg1$ it matches
   $\operatorname{Re}n(\omega)$ from `SpatialMedium` within the obliquity term.
9. **Absorbed power.** On linear runs with a spectral absorber, evanescent content and,
   after SEP 22, a transverse rate, each channel equals $-dU/dz$ of the closed-form linear
   decay to rounding, and on a complex-index fixture the material channel equals the
   propagating columns' decay (loss) and growth (gain, the sign reversed); over one fixed
   step $h$ at three levels, $U(z+h)-U(z)+hP(z)$ converges
   at first order and the trapezoid form at second, within the order tolerance of
   [testing.md](../../testing.md).
10. **Cross-backend.** `numba_cpu` and Kokkos CPU against `numpy_ref` within
    $2\gamma_m\sum|\text{terms}|$ per element, $m=4n_{\rm modes}$ for the real sums and
    $8n_{\rm modes}$ for the complex amplitude, plus the documented ulp error of the
    transcendental routines; hard-edge membership exact (a hard-mask contract) on a
    fixture that places columns within a few ulps of an edge.
11. **Thread counts and paths.** `numba_cpu` at 1, 2 and 4 threads within a rounding
    bound fixed before measurement; the `shared_mask` path's membership exactly the
    general one's and its sums within that bound; a native GoogleTest (`tests/native/test_aperture.cpp`) against
    a straightforward serial host reduction.
12. **Errors and records.** Missing tables, radial and planar geometry, a bad edge for each
    profile, quadrature with `supergauss`, more than 16 rows, an aperture outside $|x|<z_m$,
    a line with a $y$ centre, a periodic subset naming an array statistic, `collect` asked for
    a backend name, lockstep members with different tables; the configuration digest of a
    run without the new fields unchanged (a stored-digest test).

Ledger entries record the pass on `numba_cpu` and Kokkos CPU on a ten-step window.

### 8. Documentation

A section *Far-field apertures and angular regions* on the
[statistics page](../../../guide/results/statistics.md) (the mapping, the two weightings, the
direction convention, the line), and on the [free-space model](../../../guide/free-space/model.md)
page (the far-zone and Fourier-plane conditions, the paraxial error, the vacuum-flux
caveat). The six rows in `statistics-catalogue.md` and their equations in
`statistics-definitions.md`; docstrings with units and shapes on every new name; the §8 row
of the kernel contract; `n_apertures` and `n_regions` in conventions.md §6; the pass in
`backends.md`; ledger entries in `benchmarks.md`. A focused tutorial,
`docs/tutorials/free_space/plot_05_far_field_apertures.py`, collects a Gaussian beam's
spectrum through two apertures in a mask plane (one centred, one off-axis catching a
tilted second beam), checks the collected fraction against the closed form and shows the
quadrature correction. A `CHANGELOG.md` line cites this SEP.

### 9. Compatibility and impact

Additive and opt-in: no default changes, `default_spec("free")` is unchanged, and no
existing statistic changes beyond rounding. The kernel schema and the output schema are
unchanged; the new statistics are float64 arrays stored like the others. The two
`StatsSpec` fields follow SEP 22's hash-stability rule: each declares
`metadata={"omit_default": True}`, so `output.schema._serialise_dataclass` leaves it out
of the configuration record while it holds its default (`None`, and an empty table
compared as empty), as the function already does for `PlasmaResponse.integration`; every
existing resume key and scan digest, and SEP 22's stored digests, stand; the marker and
`_serialise_dataclass`'s support for it are SEP 22's phase 1, and this SEP's phase 1 adds
them if it lands first. A backend without the pass is refused on the solprop side with a
message naming `numpy_ref` and `numba_cpu`, before any step; an older solprop-kokkos
build's own generic refusal is reached only if that check is bypassed.
Moving `on_axis_spectrum` into the fused pass changes it by rounding on `numba_cpu` and
Kokkos, within a rounding bound fixed before measurement, not its definition or its
agreement with `numpy_ref`; a recorded state that includes it is checked old against new
within that bound and recaptured in that phase, with the reason recorded. Luna.jl has no
counterpart, so nothing goes on the differences page.

### 10. Alternatives considered

**Executable callbacks per plane** would cover any measurement, but break the rule of one
definition per statistic, cannot run on a device and cannot be recorded or hashed; SEP 9's
capability audit likewise counts arbitrary callbacks as no new observable.
**User-supplied weight arrays** $W(c,\omega)$ cost as much memory per aperture as the
state (2.8 GB in float64 at the size above), where an analytic profile needs five numbers.
**Masks in transverse wavevector only** miss the physics: a fixed hole selects directions,
and its $\boldsymbol\kappa$ grows with $\omega$; regions are offered beside apertures for
what they are exact at. **Kernel-contract fields** for the tables would bump the schema for
data that only the statistics read.
**Extending SEP 9's beam measurement** does not fit: it is modal, and it explicitly leaves
the spatial routes alone. **The exact $\tan\theta$ mapping** with its obliquity factor is
more accurate at wide angles and costs one more root per element; it is deferred (Q5)
because the cases above are at small angles and the paraxial definition is the one the
closed-form tests and the Fourier-lens reading share.

### 11. Phases

Each phase merges with the fast checks of `AGENTS.md`; the full gate, slow and coverage
lanes run once at the end of the SEP.

- **Phase 1 — Reference, `collect` and the manual.** `spatial.aperture_profile`,
  `stats.ApertureSet`, the `StatsSpec` fields, the six catalogue entries, the translation
  helper, the reference pass in `backends/spatial.py`, the root helper factored out of
  `separable_values`, `collect` for the four saved-field names, the fields' hash metadata,
  the manual sections and catalogue rows. `numba_cpu` gains the statistics through the
  shared host path; a solprop-side admission refuses the names on a backend without the
  pass (Kokkos until phase 3) naming `numpy_ref` and `numba_cpu`. Verifies items 1–9 and
  12 of section 7 on `numpy_ref`.
- **Phase 2 — `numba_cpu` pass.** The three bodies, the root body factored out of
  `longitudinal_rate`, the banded kernel with the `shared_mask` path, `on_axis_spectrum`
  folded in, the ledger entries, equivalence and determinism tests (items 3, 10, 11) and a
  benchmark ledger entry.
- **Phase 3 — Kokkos CPU.** The native pass, `Problem`'s axes, the bindings, packing and
  unpacking, the fold of `axis_spectrum`, the native test, the memory ledger and
  `scripts/check.sh`; parity with `numpy_ref` (item 10) and a ledger entry. GPU
  qualification follows separately.
- **Phase 4 — Tutorial and completion.** The tutorial, the remaining ledger entry, the
  changelog, the end-of-SEP lanes and the open questions moved to the roadmap.

### 12. Open questions

- **Q1 — Per-polarisation outputs (phase 1 decision).** Should
  `aperture_on_axis_spectrum` (and perhaps `aperture_spectrum`) also be offered per
  polarisation component, for polarisation-resolved detection behind an analyser?
- **Q2 — Tabulated radial profiles (roadmap).** Is a user-supplied tabulated $W(\rho)$
  wanted later, for a measured apodisation, with its own interpolation and parity rules?
- **Q3 — The vacuum flux factor in a dense medium (with SEP 5).** Inside a medium the
  spectra (5), (8) and (9) are low by about $\operatorname{Re}n(\omega)$. Should they carry
  SEP 5's admittance once its normalisation policy exists? Until then the manual gives the
  paraxial factor and the ratio $S^P/S^{\rm beam}$.
- **Q4 — The offline aperture filter (roadmap).** Should the *Further offline elements*
  item's aperture filter share `aperture_profile`, so that a saved field filtered offline
  and the in-run statistic agree?
- **Q5 — The exact mapping (decided while drafting).** The paraxial map (3) is the
  statistic's definition, with its documented $O(\theta^2)$ error; an option evaluating
  $\mathbf x=z_m\boldsymbol\kappa/k_z^{\rm vac}$ with the vacuum obliquity, for wide-angle
  emission, is a roadmap item, not part of this proposal.
- **Q6 — Other shapes (roadmap).** Rectangular and elliptical apertures, among them a
  two-dimensional slit; a line grid's aperture is already a slit.
- **Q7 — A material channel (decided while drafting).** `absorbed_power` reports the
  material loss or gain on propagating support as its fourth channel (section 4.9), so
  the four channels together are the operator's whole real part and the shape `(4,)` is
  fixed from the first release.

### 13. References

1. J. N. Sweetser, D. N. Fittinghoff and R. Trebino, "Transient-grating frequency-resolved
   optical gating," *Opt. Lett.* **22**, 519 (1997).
   [doi:10.1364/OL.22.000519](https://doi.org/10.1364/OL.22.000519)
2. R. Trebino, K. W. DeLong, D. N. Fittinghoff, J. N. Sweetser, M. A. Krumbügel,
   B. A. Richman and D. J. Kane, "Measuring ultrashort laser pulses in the time-frequency
   domain using frequency-resolved optical gating," *Rev. Sci. Instrum.* **68**, 3277
   (1997). [doi:10.1063/1.1148286](https://doi.org/10.1063/1.1148286)
3. N. C. Geib, M. Zilk, T. Pertsch and F. Eilenberger, "Common pulse retrieval algorithm:
   a fast and universal method to retrieve ultrashort pulses," *Optica* **6**, 495 (2019).
   [doi:10.1364/OPTICA.6.000495](https://doi.org/10.1364/OPTICA.6.000495)
4. J. A. Shirley, R. J. Hall and A. C. Eckbreth, "Folded BOXCARS for rotational Raman
   studies," *Opt. Lett.* **5**, 380 (1980).
   [doi:10.1364/OL.5.000380](https://doi.org/10.1364/OL.5.000380)
5. L. Mandel and E. Wolf, *Optical Coherence and Quantum Optics* (Cambridge University
   Press, 1995), the angular-spectrum representation and its asymptotic far-zone field.
   [doi:10.1017/CBO9781139644105](https://doi.org/10.1017/CBO9781139644105)
6. J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (W. H. Freeman, 2017), Fourier
   transforming properties of lenses. No DOI.

**Source audit.** The DOIs of [1–5] were resolved on 2026-10-16 and their titles, authors,
volumes, pages and years checked against the publishers' and Crossref's records; their
texts were not re-read for this draft. [6] has no DOI and its edition was not checked.
Every equation of section 4 is derived here; the moments $m_2$, the $W^2$ biases, the
exactness of (6) and the Gaussian obliquity series were checked by numerical quadrature
while drafting. No constant is transcribed from a source.

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8),
following the skeleton in [the template](../template.md).*

### Phase 1 report — reference pass, `collect` and the manual

- **Scope:** phase 1 of §11 (the reference pass, `collect` and the manual; no kernel or
  output schema change), 2026-10-18, Opus 5.5 (AI) for John Travers, branch
  `feature/sep-0023-phase-1` from `main` b20697f (kernel schema 29, output schema 20,
  after the phase 1 merges of SEP 22 and SEP 24), merged with `main` at 392c230 (SEP 26
  phase 1, kernel schema 30; a changelog-only conflict) and merged into `main` on
  2026-10-18 by the merge commit that follows the commit carrying this report.
  solprop-kokkos is unchanged.
- **Delivered:**
  - *Profiles and the reference pass* (§4, §6), `src/solprop/spatial.py`:
    `aperture_profile` (the four kinds of §4.4, with the argument formed as
    `(rho - r) * (1/w)` and the raised cosine exactly one and zero outside its edge),
    `check_aperture_profile`, `quadrature_half_width` (equation (6)) with
    `EDGE_MOMENTS` ($m_2$), `aperture_scale` (equation (3), $(z_ac)/\omega_k$, exactly
    zero at $\omega_k\le0$), `mask_plane_radius` (equation (4), evaluated as written),
    the record `ApertureTables`, and `aperture_statistics`, the chunked pass of
    equations (5) and (7) to (11), with `FAR_FIELD_STATISTICS`, `OPERATOR_STATISTICS`,
    `ABSORBED_CHANNELS` and the operator hook `AxialFactors`. A weight is exactly zero
    at $\omega_k\le0$ and where the column's image lies at or beyond $z_a$ from the
    origin (§4.2, §4.10).
  - *The request* (§5), `src/solprop/stats.py`: `ApertureSet` (frozen,
    identity-compared, owned read-only tables, every refusal of §5 by name through
    `OutputError`, `same_as`), `check_aperture_choices`, `MAX_APERTURES` and
    `MAX_REGIONS`; `StatsSpec.apertures` and `StatsSpec.spectral_regions` with the
    `omit_default` marker, `n_apertures`, `n_regions`, `same_tables`, and the refusal at
    construction of a name whose table the spec does not carry; the six catalogue entries
    of §5's table, `backend` and `between_planes` as given; `collect` forms the four
    saved-field names in one pass and refuses `poynting_spectrum` and `absorbed_power` by
    name inside `_spatial_statistic`.
  - *Translation* (§6), `src/solprop/translate/spatial.py`: `longitudinal_root` (the
    root, the propagating test and the evanescent clamp, factored out of
    `separable_values`, which now calls it), `axial_factors` (a chunk's obliquity and
    four rates, from that root) and `aperture_tables(spec, grids, geometry)` (which
    refuses radial, planar-hybrid and missing geometry, and on a line an aperture off
    $y=0$ and a region excluding $\kappa_y=0$). `src/solprop/translate/problem.py`: the
    leaf `stats_spec.spectral_regions` may hold infinite entries (`_UNBOUNDED_LEAVES`).
  - *Backends and doors:* `src/solprop/backends/spatial.py` (`FarFieldTables`, formed at
    construction, and `collect_spatial(..., tables=, operator=)` calling the pass in
    chunks of `transform.node_chunk`); `src/solprop/backends/numpy_ref/backend.py` and
    `src/solprop/backends/numba_cpu/backend.py` (the tables at construction and the host
    pass at every plane and periodic row; `numba_cpu` now reconstructs the physical node
    volume only for a call that names a physical-space statistic);
    `src/solprop/backends/registry.py` (`FAR_FIELD_BACKENDS`, `statistics_refusal` and
    the refusal in `select`); `src/solprop/driver/run.py` (the same refusal for a spec
    handed to the run loop); `src/solprop/backends/batch.py` (lockstep members carry
    equal tables); `src/solprop/driver/memory.py` (`n_apertures` and `n_regions` in the
    plane budget).
  - *Tests:* `tests/test_far_field_statistics.py` (46) and
    `tests/backends/test_far_field_statistics.py` (14), a record test in
    `tests/output/test_record_stability.py`, and the two new axes in
    `tests/output/test_preallocation.py`.
  - *Pages:* the section *Far-field apertures and angular regions* of the
    [statistics page](../../../guide/results/statistics.md), *Far-field and Fourier-plane
    statistics* on the [free-space model](../../../guide/free-space/model.md), the six rows
    of the [catalogue](../../../guide/results/statistics-catalogue.md), *Far-field spectra*
    in the [definitions](../../../guide/results/statistics-definitions.md),
    [conventions.md](../../conventions.md) §6, the §8 row of the
    [kernel contract](../../kernel-contract.md), and `CHANGELOG.md`. The tutorial is
    phase 4's.
- **Deviations from the proposal:**
  - *What the tables hold.* §6 lists the per-aperture constants and the scale table.
    `ApertureTables` also carries the geometry's column axes, weights and on-axis vector
    and the spec's regions, so the pass reads one record on both paths (`collect` with a
    `FreeGrid`, a backend with a `SpatialGeometry`); `aperture_tables` returns `None`
    for a spec that names no far-field statistic. The reciprocal edge width and the
    profile code are properties of the record (`inverse_widths`, `profile_code`).
  - *The operator reaches the physics layer as a callable.* §6 places the reduction in
    `solprop.spatial` and the root helper in the translation layer, which the physics
    layer may not import; the pass therefore takes `axial`, a per-chunk function, and
    `collect_spatial` binds `translate.spatial.axial_factors` to the run's
    `SeparableOperator`. The statistics still read the stages' root.
  - *Tables at construction, and per call for a different spec.* Both CPU backends form
    the tables when they are built (which also refuses a geometry before any step) and
    reuse them for every spec that carries the problem's own apertures (by identity) and
    regions (by value): the periodic channel's and the reference backend's filtered
    specs do. A spec handed to `run_problem` with other tables gets tables formed for
    that call, by the same helper and so with the same bits.
  - *Infinite region edges and the seam's finiteness rule.* `KernelProblem.validate`
    refuses any non-finite array in the problem tree, which would refuse the quadrants of
    §5; the region leaf is exempted by name, and NaN stays refused. The infinite energy
    windows `StatsSpec` documents were refused at translation for the same reason; at
    the maintainer's decision that leaf is exempted too (Q8).
  - *The smooth-profile reference of §7 item 4.* §7 expects the smooth error to fall
    "quickly once $w\ge2\Delta x_m$". That holds where the profile is flat at the
    aperture's centre. `tanh` is not: $W'(0)=-\operatorname{sech}^2(r/w)/2w$, so with a
    bright beam at the centre and $r/w=2$ the weight has a cone there and the lattice
    error falls as $\Delta x_m^3$ (measured $9.8\times10^{-6}$ of the collected energy at
    $\Delta x_m=w_{ff}/12.7$ and $1.2\times10^{-6}$ at half that spacing, against
    $2\times10^{-14}$ off axis). The test therefore compares an off-axis `tanh` hole
    with the profile's integral against the Gaussian's closed angular average (a scaled
    $I_0$), and a centred `supergauss` mask, flat at its centre, with
    $r^2/(r^2+w_{ff}^2)$. The definition is unchanged; the manual states the caveat
    (Q10).
  - *The transverse channel of item 9* is §4.9's exact zero until SEP 22's phase 2 adds
    `kperp_rate`, and is tested as that zero.
  - *The on-axis identity of item 3* is held to an elementwise bound formed from the
    magnitudes of the summed terms: a chunked complex sum and one contraction may cancel
    differently.
- **Decisions:**
  - *Q1 — per-polarisation outputs: not offered.* The six shapes are fixed from the first
    release, and per-component $|A_x|^2$ and $|A_y|^2$ answer only an analyser aligned with
    the grid's axes: detection behind an analyser at any other angle needs
    $\operatorname{Re}A_xA_y^*$ as well. Polarisation-resolved detection is better
    designed as an analyser Jones vector on `ApertureSet`, or as Stokes-resolved spectra,
    added as new names that leave these unchanged; it goes to the roadmap at phase 4
    (Q9).
  - *The admission* is `backends.registry.statistics_refusal`, applied by `select` to the
    problem's spec before a backend is imported, so it holds whether or not
    solprop-kokkos is installed, and by `run_problem` to a spec handed to the loop.
    `backend="auto"` resolving to Kokkos is refused, not moved.
  - *Errors.* A request or geometry refusal is an `OutputError`, as for every other
    statistics request; the registry's is a `BackendError` and the run loop's a
    `DriverError`. The two choice validators take any object, because the test session's
    runtime type checker refuses a misspelt literal before the dataclass is built.
  - *`numba_cpu`'s node volume.* The host path reconstructed the physical node volume (an
    inverse transverse transform of the state) at every spatial statistics call; it now
    does so only when `peak_intensity`, `beam_radius` or `fwhm_r` is asked for, so a
    periodic `absorbed_power` row reads the state once, as §5 says. No value changes.
  - *`collect`'s chunk*, without a transform, bounds the pass's temporaries at 64 MiB.
  - No ADR: nothing here outlasts the SEP beyond what Part I records.
- **Verification** (the branch tree, on a busy Mac, every command through
  `uv run --no-sync`):
  - The 61 new tests pass. With them, `tests/test_stats.py`, `tests/output`,
    `tests/translate`, `tests/driver/test_memory.py`, `tests/driver/test_run.py`,
    `tests/test_free_statistics.py`, `tests/test_spatial_statistics.py`,
    `tests/test_seps.py`, `tests/test_structure_figures.py` and
    `tests/test_guide_snippets.py` (the default lane's compile) pass, 2455 passed and 4
    skipped, and `tests/backends` passes, 3969 passed and 9 skipped. The doctests of the
    nine changed modules pass (58).
  - Item 1: closed-form values (`hard` and `rcos` exact, `tanh` and `supergauss` within
    four eps), the raised cosine's compact support and monotonicity under `hypothesis`,
    the edge symmetry within $4\epsilon(1+(r+2|u|)/w)$, $m_2$ against SciPy quadrature to
    $10^{-12}$, and equation (6) restoring the hard area to $2\times10^{-10}$ by
    quadrature (and the uncorrected area $\pi(r^2+\sigma^2)$).
  - Item 2: a beam tilted 20 mrad towards $+x$, built as the crossing-beam lesson builds
    its beam, is collected by the aperture at $+z\sin\theta$ to $1-10^{-5}$ at every
    in-band frequency and not by its mirror (within the Gaussian tail beyond the mirror's
    edge less a lattice diagonal); one fixed $\kappa$ moves as $1/\omega$ between two
    apertures, while a half-plane region holds it at every frequency to $10^{-12}$.
  - Item 3: every weight exactly one; `aperture_spectrum` equals `beam_spectrum` within
    $\gamma(2n+16)$ and `aperture_on_axis_spectrum` equals `on_axis_spectrum` within its
    elementwise summation bound.
  - Item 4: the centred hard hole within the energy lying within one lattice diagonal of
    its edge, bin by bin; smooth `tanh` holes under both weightings within
    $2\times10^{-10}$ (worst $2.0\times10^{-14}$ amplitude, $1.2\times10^{-15}$
    quadrature, over 55 bins); a Gaussian amplitude mask within $\gamma(8n)$ of
    $r^2/(r^2+w_{ff}^2)$.
  - Item 5: with $D_q$ a uniform spectrum collects the hard area to
    $-1.16\times10^{-11}$, the `tanh` tail's residual at $w=D/20$ (bound $10^{-9}$);
    without the correction it collects $\pi(r^2+\sigma^2)$, $\sigma^2/r^2=8.2\times10^{-3}$;
    under $W^2$ its deficit is 0.918 of the first-order $\pi rw$ at $w/r=0.1$.
  - Item 6: $\delta\omega\sum_kS^{\rm beam}_k$ equals `energy` within $2\gamma_m$; with
    non-zero end bins on a field grid the excess is the documented
    $3\delta t(|E_0|^2+|E_{\rm Nyq}|^2)/n_t$, within $2\gamma_m$.
  - Item 7: four quadrants tile the beam within $2\gamma(n+16)$; membership at
    $\kappa=0$, on an interior edge, at Nyquist and at $\pm\Delta\kappa$ is exact (every
    other region exactly zero).
  - Item 8: in vacuum at $kw_0=20$, $S^P/S^{\rm beam}$ against the exact mean of
    $\sqrt{1-u}$: worst $6.3\times10^{-15}$ against the bound $10^{-10}$, with
    $1-O\ge1.9\times10^{-3}$; in 5 bar argon at $kw_0\approx800$ the ratio lies below
    $\operatorname{Re}n$ and within $2/(kw_0)^2$ of it.
  - Item 9: each channel against its share of $-2\operatorname{Re}D$ recomputed from the
    operator's coefficients in the test, within $2\gamma_n$ of the summed magnitudes, for
    loss and for gain ($\pm10^{-5}$ in the index, a spectral collar, evanescent columns);
    the four against the stages' own `separable_values`; the material channel reversed
    with gain; over one step $h=(0.02/r_{\max})\{1,\tfrac12,\tfrac14\}$ the explicit form
    converges at observed orders 0.996 and 0.998 and the trapezoid at 1.994 and 1.997
    (gate $\pm0.15$); a periodic row at a plane is the plane's value within its rounding
    bound.
  - Item 12: every refusal of §5 and of the item by name; the pinned records of output
    schema 20 unchanged, a spec's configuration hash unchanged by an empty region table
    and changed by apertures.
  - `numba_cpu` against `numpy_ref` for all six names within $\gamma(4n+64)$, and a
    two-plane `numba_cpu` run with periodic `absorbed_power` rows.
  - ruff, ruff format, basedpyright (the changed modules and tests: no errors), the
    numpydoc lint of the changed modules and codespell pass. Coverage measured with
    `coverage-cpu.ini` over the new and related test modules leaves no added line or
    branch of a changed module uncovered (a report restricted to the lines this phase
    adds). `poe docs` (`-W -n --keep-going`) builds clean.
  - **At review, on Linux before the merge with SEP 26:** `poe check` (15 816 passed, 29
    skipped), `poe cov` at 100 %, `poe test-slow` (145 passed, 1 skipped) and the
    Kokkos-selected default and slow lanes (146 slow passed) pass. The lanes' one failure,
    a historical problem record that predates `spectral_regions`, was fixed at review by
    excluding that field as the record excludes every later field. After the merge with
    `main`, the related modules (2407 tests) pass with a kernel-schema-30 Kokkos build.
    The review also admitted infinite energy-window edges (Q8) and added an end-to-end
    check that a launched tilt drifts towards the aperture that collects it.
  - **Not run:** any benchmark (the ledger entries are phases 2 to 4), GPU.
- **Sources checked:** every relation of §4 used here was re-derived: $m_2$ (and checked
  against quadrature), equation (6) and the $W^2$ deficit (against quadrature), the
  Gaussian far-field fraction, a Gaussian mask's $r^2/(r^2+w^2)$, the exact obliquity
  average, and $\int_0^{2\pi}e^{a\cos\phi}\,d\phi=2\pi I_0(a)$ for the off-axis
  reference. References [5] and [6], cited on the free-space page, were not re-read; no
  constant is transcribed from a source.
- **Limitations:** radial and planar-hybrid geometry refused; the Kokkos backend refused
  until phase 3; `numba_cpu` forms the statistics through the host pass (its banded
  kernel is phase 2); the transverse channel is zero until SEP 22's phase 2; `collect`
  cannot form the two operator statistics; the paraxial map only (Q5); a `tanh` aperture
  with a bright beam at its centre and a small $r/w$ converges only at third order in
  the mask-plane spacing.
- **Open questions:**
  - Q8 — *Infinite energy windows in a run* (an issue outside this SEP). `StatsSpec`
    documents infinite window edges, but `KernelProblem.validate` refused them at
    translation. **Resolved at review (the maintainer):** the leaf joins
    `_UNBOUNDED_LEAVES`. Every backend reads the edges only in comparisons, the Kokkos
    pass included, so no native change is needed; NaN stays refused.
  - Q9 — *Polarisation-resolved detection* (the roadmap, at phase 4): an analyser Jones
    vector on `ApertureSet`, or Stokes-resolved spectra, as new names (Q1's decision).
  - Q10 — *The `tanh` centre* (phase 4, with the tutorial). **Decided at review (the
    maintainer accepted this recommendation; built in phase 4):** keep
    `hard` with amplitude weighting as the default, since it needs no edge parameter.
    Name `rcos` as the preferred smooth profile for quadrature weighting: it is exactly
    one inside $r-w$ and zero outside $r+w$, flat at its centre, and has no tail bias.
    Quadrature `tanh` already needs $w\le D/8$, i.e. $r/w\ge4$, where the centre slope
    is $\operatorname{sech}^2(4)/\operatorname{sech}^2(2)\approx0.02$ of the measured
    case, so the error is of order $2\times10^{-7}$. Only amplitude `tanh` with
    $2\le r/w<4$ reaches the measured $10^{-5}$ class; there `ApertureSet` should warn
    once, naming `rcos`, and still run what was asked. The manual keeps its caveat and
    adds the recommendation.
- **Handoff:** phase 2 replaces `numba_cpu`'s host path by the banded kernel and must
  match `aperture_statistics`: the weight is `aperture_profile` at the strict radius of
  `mask_plane_radius`, the argument multiplied by `inverse_widths`, and exactly zero
  where the scale is zero or where the column's own strict radius about the origin
  reaches the aperture's distance; the scale is `ApertureTables.scale`, so the
  reference's to the bit; the on-axis amplitude is complex, with $W$ once under both
  weightings; the absorbed power uses `fourier.energy_weights` and `axial_factors`'
  four rates. `FarFieldTables` already forms the tables at construction, and the
  backend's statistics call no longer reconstructs the node volume for these names.

### Phase 2 report — the `numba_cpu` pass

- **Scope:** phase 2 of §11 (the `numba_cpu` pass; no kernel or output schema change),
  2026-10-19, Opus 5.5 (AI) for John Travers, branch `feature/sep-0023-phase-2` from
  `main` 8674f98 (kernel schema 32, output schema 20). solprop-kokkos is unchanged; it
  still refuses the six names (phase 3). Merged into `main` on 2026-10-19 by the merge
  commit that follows the commit carrying this report.
- **Delivered:**
  - *Bodies* (§6), `src/solprop/backends/bodies.py`: `longitudinal_root` (the root, the
    propagating test and the evanescent clamp, factored out of `longitudinal_rate`,
    which now calls it), `mask_plane_radius` (equation (4), a **protected** body:
    compiled strict and called, never contracted), `aperture_weight` (the four profile
    codes, the raised cosine's early-out, and the exact zeros at zero scale and where
    the column's own strict radius about the origin reaches the distance, as
    `aperture_statistics` defines them) and `region_member` (the half-open test on the
    negated stored axes).
  - *The pass* (§6), `src/solprop/backends/numba_cpu/spatial.py`: `make_far_field`
    (stage one, a `prange` over contiguous bands of columns, columns ascending and
    frequency innermost, the shared-mask path), `far_field_sum` (stage two, the bands
    summed serially in ascending order) and `FarFieldPass` (partials held from
    construction, the host finishing step: `epsilon_0 c/2`, the spectral-density
    factor, the energy weights, `|A|^2`). `src/solprop/backends/spatial.py`:
    `FarFieldLayout`, `far_field_layout` (the slot order of §6, present slots only),
    `far_field_bands` (units are rows on a plane and `LINE_BLOCK = 64` columns on a
    line; `FAR_FIELD_BAND_BUDGET = 128 MiB`), `far_field_partials`, `AXIS_FOLDED`, and
    `collect_spatial(..., axis=)`.
  - *The backend*, `src/solprop/backends/numba_cpu/backend.py`: the pass replaces the
    host path for the six names and `on_axis_spectrum` on the Cartesian and line kinds
    (`_spatial_member`), its bands in a `prange` under the spatial stages' own rule
    (`_parallel_spatial`); `statistics_partials` in `memory_ledger()`.
  - *The estimate*, `src/solprop/backends/memory.py`: `far_field_memory`, the same row
    to the byte in the `base` group on `numba_cpu`.
  - *Tests:* `tests/backends/test_numba_far_field_kernels.py` (24, added to
    `poe cov-kernels`), and `HOST.mask_plane_radius` among the protected helpers of
    `tests/backends/test_numba_licence.py`.
  - *Benchmark:* `benchmarks/sep23/phase2_costs.py` and the ledger entry
    [SEP 23 phase 2](../../benchmarks.md#far-field-statistics-pass).
  - *Pages:* [backends.md](../../backends.md) §3.13 and §8, the
    [statistics page](../../../guide/results/statistics.md) (*Far-field apertures and
    angular regions*: `numba_cpu` computes them natively), `CHANGELOG.md`.
- **Deviations from the proposal:**
  - *The axis statistics share the `W = 1` slot.* §6 folds `on_axis_spectrum` into the
    pass. The slot's amplitude is also handed to `collect_spatial`, so
    `on_axis_intensity` reads it rather than forming the contraction again; a request
    naming either axis statistic runs the pass. Both change by rounding only, within the
    bound of `tests/backends/test_numba_far_field_kernels.py`; no recorded state of
    `numba_cpu` held them, so nothing was recaptured (§9).
  - *The line's units.* §6 says "blocks of x samples" on a line without a size; a block
    is `LINE_BLOCK = 64` columns, a module constant.
  - *Two buffers in the row.* `statistics_partials` holds the band partials and their
    band sum, `8 (n_bands + 1) n_slots n_omega` bytes. The scale table stays in the
    tables `FarFieldTables` already held in phase 1 (the translated problem's own
    tables, which the estimate does not count).
  - *The transcendental term of item 10's bound.* An aperture weight's `tanh`, `sin`
    or `exp` carries an absolute error of a few ulps of one, which the tail's
    cancellation in `1 - tanh` does not shrink, so a relative bound on the terms is not
    enough in a `tanh` tail (one element exceeded it by a factor 4.6 before the term
    was added). The bound adds 8 eps of the beam's magnitudes to `aperture_spectrum`
    and 12 eps of the axis's to `aperture_on_axis_spectrum`.
- **Decisions:**
  - *When the pass is parallel:* under the spatial stages' rule (`numba_cpu`'s large
    regime with more than one thread and a state above the regime threshold), so a
    small problem runs it serially; the result does not depend on it.
  - *When it is compiled:* stage one is a parallel closure, compiled at the first
    statistics call (the entrance plane, before any step) and never disk-cached; its
    serial form and stage two are cached. No eager compilation at construction.
  - *The shared-mask path* is taken exactly when every centre and distance are equal;
    `FarFieldPass(..., shared=)` forces or forbids it for the tests and refuses forcing
    it on distinct apertures.
  - No ADR: nothing here outlasts the SEP beyond what Part I records.
- **Verification** (the branch tree, on a busy Mac, every command through
  `uv run --no-sync`):
  - `tests/backends/test_numba_far_field_kernels.py`: 24 passed, compiled and with
    `NUMBA_DISABLE_JIT=1`. Item 10: the pass against `aperture_statistics` (and the
    reference contraction for `on_axis_spectrum`) on field and envelope grids, one and
    two polarisations, all four profiles, both weightings, a transverse collar and
    evanescent columns (every absorbed channel live), a gain medium with content at
    zero frequency, a partial request and a line of three blocks, elementwise within
    `3 gamma(8 n + 64)` of the sums of the magnitudes (the reference formed from `|E|`
    and `|alpha|`) plus the transcendental term; worst margins 1.4e-3 of the bound
    (`beam_spectrum`), 8.3e-4 (`aperture_spectrum`), 2.1e-4
    (`aperture_on_axis_spectrum`). Hard-edge membership: the radius body equals
    `spatial.mask_plane_radius` to the bit over a random set, and a hole whose edge
    passes through one column's image (at, a hair inside, a hair outside) is decided
    as the reference decides. Item 3: every weight exactly one; `aperture_spectrum`
    against `beam_spectrum` and `aperture_on_axis_spectrum` against
    `on_axis_spectrum` within the same bound. Item 11: 1, 2 and 4 threads, and the
    shared-mask path forced, taken and forbidden, within the same bound. Bands: the
    edges, a budget forcing two uneven bands (same values within the bound) and a
    budget below one band. The ledger row equals `far_field_memory` and the estimate's
    base exceeds the reference's by it, to the byte. Refusals: tables missing, a forced
    shared path, a radial problem.
  - The phase 1 modules `tests/backends/test_far_field_statistics.py` (17) and
    `tests/test_far_field_statistics.py`, `tests/backends/test_memory.py`,
    `tests/driver/test_memory.py` (191 together), the bodies, licence and cache-policy
    tests and the spatial backend modules (`test_free_*`, `test_line_free.py`,
    `test_on_axis_statistics.py`, `test_collar_diagnostic.py`,
    `test_spatial_execution.py`, `test_kperp_collar.py` and others: 540) pass.
  - Coverage (`coverage-cpu.ini`, the kernel test module interpreted, then the related
    modules compiled): every added line and branch of the changed modules covered.
  - ruff, ruff format, basedpyright (the changed modules and tests: no errors), the
    numpydoc lint and doctests of the changed modules, codespell, `poe docs`.
  - Benchmarks: the ledger entry; on ten-step windows the pass costs 1.07 and 1.28
    times `energy` alone, 37 and 30 times less than the host pass, about 2.5 % of a
    step.
  - **At review, on Linux with the schema-32 wheel:** `poe check` (16 082 passed, 29
    skipped), `poe cov` at 100 %, the Kokkos-selected default (16 082 passed) and slow
    (148 passed) lanes, and `poe test-slow` (146 passed, 1 skipped) but for the air
    tutorial's 60 s gallery budget, which took 80 s on a loaded node (the known
    contention failure; its physics assertions passed).
  - **Not run:** Kokkos's own pass (phase 3), GPU.
- **Sources checked:** no new relation; the pass re-implements the phase 1 definitions,
  against which it is tested.
- **Limitations:** the Kokkos backend still refuses the names (phase 3); the pass's
  stage one compiles at the first statistics call of a parallel run (a few seconds,
  not cached).
- **Open questions:** none new. Q9 and Q10 stay with phase 4.
- **Handoff:** phase 3 mirrors the pass natively: the slot layout of
  `backends.spatial.far_field_layout` and the band edges of `far_field_bands` (the
  same units and budget), the radius compiled without contraction, the zeros of
  `aperture_weight`, the root of `longitudinal_root`; the Python side packs the tables
  as `FarFieldPass._apertures` does and unpacks the band sums as `_finish` does. The
  tests of `tests/backends/test_numba_far_field_kernels.py` take a Kokkos variant with
  the same bounds.

### Phase 3 report — the Kokkos CPU pass

- **Scope:** phase 3 of §11 (the native Kokkos pass; no kernel or output schema
  change), 2026-10-19, Opus 5.5 (AI) for John Travers, branches
  `feature/sep-0023-phase-3` of solprop (from `main` 9f0eb0e, kernel schema 32, output
  schema 20) and of solprop-kokkos (from `main` 391ca7d, schema 32). Merged with `main` at
  7affe12 (SEP 22 phase 3, no conflict) and into `main` on 2026-10-19 by the merge commits
  that follow the commit carrying this report, solprop-kokkos alongside.
- **Delivered:**
  - *The native pass* (§6), solprop-kokkos `include/solprop_kokkos/far_field.hpp` and
    `src/far_field.cpp`: `FarFieldPass`, `far_field_layout` and
    `far_field_band_count`/`far_field_band_edge` (solprop's slot layout and band edges,
    the same units, `LINE_BLOCK` and budget), the element functions `far_field_radius`
    (protected: strict on the host, round-to-nearest intrinsics in a device build),
    `far_field_weight` and `far_field_member`. Stage one is a team per band, each
    (band, bin) owned by one thread walking the band's columns in ascending order (one
    thread per team on the host, so frequency innermost as on `numba_cpu`); stage two
    adds the bands in ascending order and applies `intensity_factor`, the
    spectral-density prefactor, the energy weights and `|A|^2` natively.
    `GeneralState::aperture_statistics(flags, apertures, profile_code, exponent,
    regions, intensity_factor)` in `src/general.cpp`, its flags from 32 (the six names
    and `on_axis_spectrum`) refusing any other bit; the binding and stub
    (`src/bindings.cpp`, `src/solprop_kokkos/_core.pyi`).
  - *`Problem`*: `spatial_k_x`, `spatial_k_y` (validated on free space, refused
    elsewhere) and `spatial_far_field_slots`, which sizes the partials at construction;
    ledger rows `far_field_partials` and `far_field_tables`.
  - *The adapter*, `src/solprop_kokkos/backend.py`: `FarFieldTables` formed at
    construction, the payload (`_aperture_payload`, `(n_apertures, 6 + n_omega)` rows
    and `(n_regions, 4)` boxes), the flags, the unpacking in the fixed order with the
    consumed-count check, `on_axis_spectrum` on free space folded into the pass, and
    `KokkosBackend.far_field_statistics`.
  - *solprop*: `backends/registry.py` (`FAR_FIELD_BACKENDS` gains `kokkos`;
    `statistics_refusal` reads a build's declared names through `_plugin_far_field`
    and refuses the rest by name before the backend is built), `backends/memory.py`
    (`far_field_memory` for Kokkos).
  - *Tests*: solprop-kokkos `tests/native/test_aperture.cpp` (10, against a serial
    host reduction) and two cases in `tests/test_free.py`; solprop
    `tests/backends/test_kokkos_far_field.py` (14) with
    `kokkos_far_field_thread_probe.py`, and the refusal and estimate cases of
    `tests/backends/test_far_field_statistics.py` and
    `tests/backends/test_numba_far_field_kernels.py` updated.
  - *Benchmark*: `benchmarks/sep23/phase3_costs.py` and the ledger entry
    [SEP 23 phase 3](../../benchmarks.md#far-field-statistics-pass).
  - *Pages*: [backends.md](../../backends.md) §3.13 and §8, the
    [statistics page](../../../guide/results/statistics.md), `CHANGELOG.md`; the
    solprop-kokkos README and CHANGELOG.
- **Deviations from the proposal:**
  - *The admission reads the build.* §11 removes the Kokkos refusal of the six names;
    an older wheel must still refuse them by name. With no schema bump, the registry
    asks the installed build which far-field statistics it declares
    (`KokkosBackend.far_field_statistics`) and refuses the others, so a build from
    before this phase is refused before it is built, as phase 1's admission was.
  - *The device path keeps its partials in global memory.* §6 expects the slot
    accumulators in registers on a device, the set run in groups if large. The pass
    runs as written on a device build (compiled, not run here): each thread owns its
    bins and adds into the band's partials, with a team barrier per column. Moving the
    accumulators into registers is a device optimisation for the GPU qualification.
  - *Two exact shortcuts.* A `tanh` weight beyond `|t| = 22` is exactly 0 or 1 when
    `tanh` is correctly rounded there (it lies within 1.6e-19 of `±1`), so the
    transcendental is skipped (the raised cosine's early-out of §4.4, for `tanh`); and
    apertures sharing a distance share the image's distance from the origin, the
    radius too when they share a centre (§6's shared-mask path, widened). The first
    halved the pass's cost on the ledger's case.
  - *`on_axis_intensity` is not folded.* Phase 2 had `numba_cpu`'s `on_axis_intensity`
    read the `W = 1` slot. On Kokkos it stays `spatial_statistics`' own: its axis
    amplitude is formed and inverse-transformed natively, and only `axis_spectrum` was
    named for the fold. The radial and planar kinds keep flag 8's loop for
    `on_axis_spectrum`.
- **Decisions:**
  - *No kernel-schema bump* (33 stays free): the new `Problem` fields are the
    native adapter's, packed from the existing `geometry.spectral_axes` and the spec;
    the contract tuples and `KernelProblem` are unchanged.
  - *Band edges natively*: the native pass computes solprop's band formula itself
    (`far_field_band_count`), checked against solprop's shapes in the native test and
    through the ledger's partials, which equal `far_field_partials` to the byte.
  - *Ledger*: `far_field_partials` holds the partials alone, `8 n_bands n_slots
    n_omega` bytes (no band-sum buffer: stage two finishes as it adds), which the
    estimate counts; `far_field_tables` (axes, tables, packed output; the tables and
    output grow at the first call) is not estimated, as `numba_cpu`'s tables are not.
  - No ADR: nothing here outlasts the SEP beyond what Part I records.
- **Verification** (the branch trees, on a busy Mac; every solprop command through
  `uv run --no-sync` with the branch's wheel installed):
  - Native: `solprop_kokkos_native_tests --gtest_filter=*FarField*`, 10 passed (all
    flags on field and envelope grids, one and two polarisations, all four profiles,
    both exponents, a line of three blocks, the three radius paths), each value within
    `3 gamma(8 m + 64) + 8 eps` of the host reduction's magnitudes, exact zeros exact.
  - `tests/backends/test_kokkos_far_field.py`: 14 passed. Item 10: Kokkos against
    `numpy_ref` within phase 2's bound (`3 gamma(8 n + 64)` of the magnitudes plus the
    transcendental term) and against `numba_cpu` within twice it, on the four
    profile/weighting cases with a transverse collar and evanescent columns, a line of
    three blocks, a gain medium with zero-frequency content and a partial request;
    item 3's identities; a column at a hard edge decided as the reference decides;
    item 11: 1, 2 and 4 threads (one process each) within the bound, on the shared
    path; the ledger's `far_field_partials` equal to the estimate's row and the base
    estimate equal to `numpy_ref`'s plus it, to the byte; a request beyond the held
    partials; a run on Kokkos and `numba_cpu` (entrance plane within twice the bound,
    later planes and the periodic `absorbed_power` rows within 1e-9 of their scale);
    the native refusals.
  - `tests/backends/test_far_field_statistics.py`, `tests/test_far_field_statistics.py`,
    `tests/backends/test_numba_far_field_kernels.py`, `tests/backends/test_memory.py`,
    `tests/driver/test_memory.py`: pass. solprop-kokkos `tests/test_free.py`: 56
    passed. Doctests of `registry` and `memory`; ruff, ruff format, basedpyright, the
    numpydoc lint and codespell of the changed modules; coverage of `registry` and
    `memory` leaves no added line or branch uncovered.
  - The Kokkos-backed spatial modules `tests/backends/test_free_absorber.py`,
    `test_free_noise.py`, `test_spatial_execution.py`, `test_on_axis_statistics.py`,
    `test_line_free.py`, `test_planar_kokkos.py` and `test_kokkos_low_storage.py`: 293
    passed (`on_axis_spectrum`'s fold on free space among them).
  - `scripts/check.sh` in solprop-kokkos against this branch of solprop: green (format,
    lint, strict typing, reuse, clang-format, clang-tidy, native tests, ASan+UBSan
    native tests, pytest 332 passed and 5 skipped, the distribution wheel and its
    smoke test), Homebrew Clang 23.1.3, debug presets and a Release wheel, OpenMP, four
    threads for pytest. The branch's local-profile wheel (`6e7f09b`) is installed in
    the solprop worktree. `poe docs` builds clean.
  - Benchmarks: the ledger entry; the pass costs 14.9 and 16.3 times `energy` (which
    on Kokkos is one parallel read), about 2.3 % of a step.
  - **At review, on Linux with a wheel built from the solprop-kokkos branch:** `poe
    check` (16 097 passed, 29 skipped), `poe cov` at 100 %, the Kokkos-selected default
    (16 097 passed) and slow (148 passed) lanes, and `poe test-slow` but for the air
    tutorial's 60 s budget (62 s on a loaded node). The first Linux build failed on two
    sign conversions GCC treats as errors (`-Wsign-conversion`), fixed at review. After
    the merge with `main` the related modules (570 tests) pass.
  - **Not run:** GPU (device builds compile the pass; not run).
- **Sources checked:** no new relation; the pass re-implements the phase 1 definitions,
  against which it is tested.
- **Limitations:** the device path is unqualified and keeps its accumulators in global
  memory; `on_axis_intensity` on Kokkos does not read the `W = 1` slot. Found at review,
  for phase 4 (waste, not error): the native pass evaluates the aperture weights even
  when no aperture statistic is requested (a periodic `beam_spectrum` sample with
  apertures declared), where `numba_cpu` sets the aperture count to zero; gate it on the
  layout's aperture slot. And Kokkos's partials are sized for `on_axis_intensity`, which
  it does not fold; leave it out of Kokkos's slot count in the payload and the estimate.
- **Open questions:** none new. Q9 and Q10 stay with phase 4.
- **Handoff:** phase 4 has all three backends. The GPU qualification should measure
  the device pass (registers for the slot accumulators, groups of apertures) against
  `tests/backends/test_kokkos_far_field.py`'s bounds.

### Phase 4 report — tutorial and completion

- **Scope:** phase 4 of §11 (the tutorial, the decided `tanh` warning, the two Kokkos
  follow-ups of the phase 3 review, the remaining ledger entry and the open questions;
  no kernel or output schema change), 2026-10-20, Opus 5.5 (AI) for John Travers,
  branches `feature/sep-0023-phase-4` of solprop (from `main` cf97217, kernel schema 32,
  output schema 20) and of solprop-kokkos (from `main` 95b7502, schema 32). The
  end-of-SEP lanes, `Final` and the move to the archive follow at the merge.
- **Delivered:**
  - *The tutorial* (§8),
    `docs/tutorials/free_space/plot_07_far_field_statistics.py`: two pulses crossing in a
    thin Kerr sample on a 224 × 64 × 128 Cartesian grid, eight replayed steps on the
    default backend, statistics-only planes at every step with the exit field kept; an
    `ApertureSet` of four raised-cosine holes under quadrature weighting (the two beams,
    the signal $2\mathbf k_1-\mathbf k_2$, a dark hole), the four quadrants,
    `beam_spectrum`, `poynting_spectrum` and `absorbed_power` sampled every step. It
    shows the paraxial map at three frequencies, the signal and dark hole along $z$
    (the signal peaks near the coherence length $\pi/|\Delta k|$), hard holes and
    raised cosines under both weightings (through `collect` on the launch) against the
    exact collected fraction of a Gaussian beam, the run's statistic against `collect`
    and against the definition written out on the kept exit plane, the quadrants'
    tiling, the beam-averaged obliquity and the absorbed energy against the drop of
    `energy`. It runs in about 2 s on the Kokkos CPU build. Gallery
    `docs/tutorials/free_space/README.rst`.
  - *Q10* (§4.4, §5): `exceptions.ApertureEdgeWarning`; `ApertureSet` warns once for
    `tanh` under amplitude weighting with $r/w<4$, naming the apertures and `rcos`, and
    builds the set as asked (`src/solprop/stats.py`). The statistics page gains *Which
    profile*: `hard` and amplitude stay the default, `rcos` is recommended for
    quadrature weighting, and the warning. The phase 1 smooth-hole test now expects the
    warning at $r/w=2$.
  - *Kokkos follow-ups*: solprop-kokkos `src/far_field.cpp` gates the aperture loop and
    the shared-radius block on `layout.apertures >= 0`; `src/solprop_kokkos/backend.py`
    sizes `spatial_far_field_slots` with
    `far_field_partials(problem, axis_folded=AXIS_SPECTRUM_ONLY)`. solprop
    `backends/spatial.py` (`AXIS_SPECTRUM_ONLY`, the `axis_folded` keyword of
    `far_field_layout` and `far_field_partials`) and `backends/memory.py`
    (`far_field_memory` counts Kokkos's slots without `on_axis_intensity`).
  - *The four-wave-mixing tutorial* of SEP 22 drops `backend="numba_cpu"`: on the
    default (Kokkos) backend it ran in 7.0 s against 11.9 s on `numba_cpu`, with every
    printed figure equal to the digits shown.
  - *Tests*: solprop `tests/test_far_field_statistics.py` (the warning, once, naming
    both wide apertures; five sets that build without it),
    `tests/backends/test_kokkos_far_field.py` (`on_axis_intensity` takes no native
    slot, ledger equal to the estimate to the byte; a beam, region and absorbed request
    with apertures declared against the reference); solprop-kokkos
    `tests/native/test_aperture.cpp` (`DeclaredAperturesCostNothingUnlessAsked`).
  - *Benchmark*: `benchmarks/sep23/phase4_costs.py` and the ledger entry
    [SEP 23 phase 4](../../benchmarks.md).
  - *Pages*: [statistics](../../../guide/results/statistics.md),
    [backends.md](../../backends.md) §3.13, [roadmap](../../roadmap.md), `CHANGELOG.md`; the
    solprop-kokkos CHANGELOG.
- **Deviations from the proposal:**
  - *The tutorial's file and scope.* §8 names
    `docs/tutorials/free_space/plot_05_far_field_apertures.py` with two apertures (one
    centred, one off-axis catching a tilted beam), the closed form and the quadrature
    correction. `plot_05` was taken by the low-storage tutorial, so it is `plot_07`, and
    it covers the wider scope set for this phase: a crossing-beam run with four holes
    under both weightings and two profiles, the quadrants and the operator statistics. The closed form is checked on the beam 1 hole at the
    entrance, against the exact fraction of each profile (the hard closed form, and the
    profile's own integral for the raised cosines).
  - *What the closed form shows.* §8 expects the tutorial to "show the quadrature
    correction". On a lattice of six points per far-field radius the raised cosines
    match their own integrals to $3.5\times10^{-4}$ where the hard hole's staircase
    reaches $4\times10^{-2}$; quadrature weighting then departs from the hard hole by
    the slope term of §4.5, $2.8\times10^{-2}$ here (where the edge sits on the beam's
    steepest part, with $w=2\Delta x_m$), and amplitude weighting by its first-order
    deficit, $9\times10^{-2}$. The tutorial says so rather than claiming a smaller
    error for quadrature than for the hard hole on this lattice.
- **Decisions:**
  - *Q10, as decided by the maintainer:* the warning class is a `SolpropWarning`
    subclass, raised once per set (one warning naming every wide aperture), at
    `2 <= r/w < 4` (the lower end is the existing refusal); `r/w = 4` exactly does not
    warn.
  - *The Kokkos slot count* is a keyword of solprop's layout functions rather than a
    Kokkos-side formula, so the payload and the estimate share one definition.
  - *The remaining ledger entry* of §11 is phase 4's measurement of the gate and the
    slot count (the passes' own entries are phases 2 and 3).
  - No ADR: nothing here outlasts the SEP beyond what Part I records.
- **Verification** (the branch trees, on a busy Mac; every solprop command through
  `uv run --no-sync` with this branch's wheel installed):
  - `tests/test_far_field_statistics.py` (52), `tests/backends/test_far_field_statistics.py`,
    `tests/backends/test_numba_far_field_kernels.py`, `tests/backends/test_kokkos_far_field.py`
    (16), `tests/test_free_statistics.py`, `tests/backends/test_on_axis_statistics.py`
    together under xdist, `tests/backends/test_memory.py` (98) and
    `tests/driver/test_memory.py` (47) separately (the two same-named modules collected
    in one xdist session cannot find a conftest fixture, a collection quirk unrelated
    to this phase), and `tests/test_seps.py`: pass. Doctests of the four changed
    modules (32), basedpyright (no errors), the numpydoc lint, ruff, ruff format and
    codespell of the changed files. Coverage (`coverage-cpu.ini`) over the far-field
    modules leaves no added line or branch of `stats`, `exceptions`,
    `backends.spatial` or `backends.memory` uncovered.
  - The tutorial runs clean under `-W error`, figures inspected; its checks:
    `collect` against the run $4.5\times10^{-16}$ (relative), the definition against
    the run $2.8\times10^{-16}$ in collected fraction, the quadrants' tiling
    $7\times10^{-16}$, the obliquity 1.44992 against the index 1.45, the absorbed
    energy $1.0832\times10^{-8}$ J against the energy drop $1.0829\times10^{-8}$ J.
    The dark hole's reading at the exit ($6.4\times10^{-6}$ of beam 1's hole) moved
    by 3 % when the y spacing was halved and by $10^{-4}$ when the x spacing was
    refined, so it is the beams' nonlinear halo, not the grid.
  - solprop-kokkos `scripts/check.sh` against this solprop branch: green (format,
    lint, strict typing, reuse, clang-format, clang-tidy, the native tests with the
    eleven far-field cases among them, ASan+UBSan native tests, pytest 332 passed and 5
    skipped, the distribution wheel and its smoke test), Homebrew Clang 23.1.3. Its
    first run failed on clang-format: two lines of `src/far_field.cpp` from `main` and
    the new native test, formatted with the pinned clang-format. The branch's
    local-profile wheel (built before that format-only commit) is installed in the
    solprop worktree.
  - Benchmarks: the ledger entry; a `beam_spectrum` sample with apertures declared
    cost 4.43 times a bare one before the gate and 0.87 and 1.02 times after; the band
    partials for `beam_spectrum` with `on_axis_intensity` fall from 1.125 MiB to
    0.375 MiB.
  - `poe docs` (`-W -n --keep-going`): clean; the new tutorial executed in 1.3 s.
  - **End-of-SEP lanes**, on the branches merged with `main` after SEP 27 (solprop
    61b70a8, solprop-kokkos 713abe3), on Linux (AWS, a wheel built from the branch with
    GCC 16): `poe check` (16 600 passed), `poe test-slow` (147 passed) and the
    Kokkos-selected default (16 600 passed) and slow lanes pass but for two cases of
    `tests/backends/test_band_plasma_parity.py` (the trapezoid rule on the mixture, on
    `numba_cpu` and Kokkos: the electron density 4e-8 from the reference against the
    test's 2e-8). The same two fail on `main` alone on Linux and pass on macOS; they
    arrived with SEP 27's plasma source and are not this SEP's (recorded for that
    work). `poe cov`, which ran out of memory beside another lane on a small instance,
    passed on macOS at 100 % with all 16 602 tests.
  - **Not run:** GPU (the device pass compiles; it is not run).
- **Sources checked:** no new relation. The tutorial's closed form (the Gaussian's
  collected fraction through a centred hole), the coherence length of the
  self-diffraction signal, $\Delta k\approx-4K_0^2/k$ for $2\mathbf k_1-\mathbf k_2$ in
  a dispersionless medium, and the obliquity's paraxial value were re-derived; the
  slope residual of §4.5 accounts for the quadrature figure ($-2.7\times10^{-2}$
  estimated, $-2.8\times10^{-2}$ measured).
- **Limitations:** as at phase 3 (the device pass unqualified; radial and planar-hybrid
  geometry refused; the paraxial map only). The warning concerns amplitude `tanh`
  only; quadrature `tanh` is admitted at $w\le D/8$ where the centre's error is of
  order $2\times10^{-7}$.
- **Open questions and their destinations** (every question of the SEP):
  - Q1 — per-polarisation outputs: decided in phase 1 (not offered); the alternative is
    Q9.
  - Q2 — tabulated radial profiles: the [roadmap](../../roadmap.md), *Far-field detection
    beyond SEP 23*.
  - Q3 — the vacuum flux factor in a dense medium: the roadmap, same item, with SEP 5.
  - Q4 — the offline aperture filter: the roadmap, *Further offline elements*.
  - Q5 — the exact mapping: decided while drafting; the option is on the roadmap,
    *Far-field detection beyond SEP 23*.
  - Q6 — other shapes: the roadmap, same item.
  - Q7 — a material channel: decided while drafting, built in phase 1.
  - Q8 — infinite energy windows: resolved at the phase 1 review.
  - Q9 — polarisation-resolved detection: the roadmap, *Far-field detection beyond
    SEP 23*.
  - Q10 — the `tanh` centre: decided by the maintainer, built here.
  - The device pass (phase 3's handoff): the roadmap, *The far-field pass on GPUs*.
  - No new question.
- **Handoff:** the end-of-SEP lanes (full gate, `poe cov`, `poe test-slow`, the
  Kokkos-selected lanes, a Linux GCC build of solprop-kokkos with the strict warnings),
  then `Final`, the move to `docs/dev/seps/archive/` (the **Manual pages** field and the
  relative links follow the move) and the roadmap's accepted-list entry removed. The
  solprop and solprop-kokkos branches merge together: the Kokkos adapter imports
  `AXIS_SPECTRUM_ONLY`.

## Part III — History

- 2026-10-16: Drafted.
- 2026-10-16: Reviewed by Opus 5.5 (AI); seven should-fix items applied: the fourth
  `absorbed_power` channel carried into §6–§7, the axial-flux test against the exact
  paraxial form, the weight set to zero beyond the paraxial disc, the `tanh` width limit
  under quadrature weighting, the reference reduction placed in the physics layer, the
  dependence on SEP 22's record rule declared, and a refusal that names a supporting
  backend.
- 2026-10-17: Restated for [SEP 25](sep-0025-rounding-level-reproducibility.md).
  Bounds replace bit identities between separate computations: `numba_cpu`'s thread-count
  invariance (§1, §7 item 11), the `shared_mask` path's sums against the general path's
  (§6, §7 item 11), the $W\equiv1$ identities with `beam_spectrum` and `on_axis_spectrum`
  (§6, §7 item 3), the stages after the root is factored out (§6), and `on_axis_spectrum`
  in the fused pass (§9), all within a rounding bound fixed before measurement; a
  recorded state is recaptured only after an old-against-new check at that bound. Kept
  exact as structural: hard-edge membership from the strict, uncontracted radius (§6,
  §7 items 10 and 11), the weight's exact one, and the configuration digest of a run
  without the new fields (§7 item 12, §9).
- 2026-10-18: Revised against SEP 25 Final and SEP 21 as reviewed: the `einsum` remark
  restated as a rounding-level fact (§6); the agreement between `absorbed_power`'s
  spectral channel and SEP 21's cumulative `absorbed_energy` is to SEP 21's quadrature
  bound, not to rounding (§3). The Kokkos facts of §3 and §6 (flag values 1 to 16 and the
  serial `axis_spectrum` loop) and the host path of §3 were re-checked against the
  current tree and stand.
- 2026-10-18: Accepted (John Travers) after his final review; Part I frozen.
- 2026-10-18: In progress. Phase 1 (the reference pass, `collect` and the manual)
  implemented on `feature/sep-0023-phase-1`, with no kernel or output schema change; its
  report is in Part II.
- 2026-10-19: Phase 2 (the `numba_cpu` pass) implemented on `feature/sep-0023-phase-2`,
  with no kernel or output schema change; its report is in Part II.
- 2026-10-19: Phase 3 (the Kokkos CPU pass) implemented on `feature/sep-0023-phase-3` in
  solprop and solprop-kokkos, with no kernel or output schema change; its report is in
  Part II.
- 2026-10-20: Phase 4 (the tutorial, the `tanh` warning, the Kokkos follow-ups and the
  open questions) implemented on `feature/sep-0023-phase-4` in solprop and
  solprop-kokkos, with no kernel or output schema change; its report is in Part II.
- 2026-10-20: Final (John Travers, on the agreed plan). Phases 1–4 merged with green closing lanes; open questions moved to the roadmap in the phase 4 report; moved to the archive.
