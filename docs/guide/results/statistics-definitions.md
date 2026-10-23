# Definitions of pulse statistics

Energy, peak power and width are different measurements of a propagated field. This chapter gives their conventions and explains why broadband Poynting flux, pulse shape and spatial representation matter. The [RDW lesson](../../introduction/lessons/plot_01_rdw.rst) introduces these measurements on an actual run.

## The definitions

## The analytic signal, once

Modal run power is the cycle-averaged Hermitian flux
$P_m(t)=\sum_p\operatorname{Re}[a_{mp}(t)a_{w,mp}(t)^*]$, with $a_w$ the
analytic signal formed from $w a$. The real instantaneous sum-frequency term is
excluded. Standalone array functions use $|b(t)|^2$; see the
[power definitions](power-and-intensity.md) for their broadband difference.
On an envelope grid the carrier is removed before the inverse transform;
on a field grid {py:func}`~solprop.fourier.analytic_signal` constructs the
positive-frequency signal. Both members of the flux product use this construction.

That is one construction, from one place, and it follows SciPy's convention: the
zero-frequency bin is not doubled and the Nyquist bin is kept. Only that convention makes
$\operatorname{Re} E_a = E$ an identity and the two energy functionals an exact Parseval
pair. A construction that doubles the zero-frequency bin and discards the Nyquist bin
agrees with it on any apodised grid, where the spectral collar has driven both end bins
to zero, and differs at the $1/N$ level otherwise.

## Energy, and windows that tile

{py:func}`~solprop.stats.energy` is {py:func}`~solprop.fourier.energy_omega`, the exact
discrete form of the energy functional of [conventions.md](../../dev/conventions.md) §4,
which agrees with the time-domain functional to round-off rather
than to a quadrature error.

{py:func}`~solprop.stats.energy_window` applies a hard gate, $\omega_{lo} \le \omega <
\omega_{hi}$. The half-open edge is what makes windows laid end to end assign every bin
exactly once, so their energies sum to the total exactly — which is a test. There is
no tapered edge, because a windowed *energy* never returns to the time domain and has
nothing to gain from a smooth one.

## Peak power

For modal run statistics the reductions are:

$$ P^{\text{peak}}_m = \max_t P_m(t) , \qquad
   P^{\text{peak}}_{\text{tot}} = \max_t \sum_m P_m(t) . $$

The polarisation components of one mode are summed before the maximum because they are
two components of one field and their powers add at each instant. The modes are summed
before the maximum too, so `peak_power_total` is the peak of the total power and **not**
the largest of the per-mode peaks, nor their sum: modes that peak at different times
never carry their combined power at any single instant.

One reduction serves all three statistics, the windowed one included, and a
regression test checks it on a two-mode counter-example where the largest row of
per-mode powers, `[5.0, 0.5]`, is not the peak of the sum, `5.5`.

## Two temporal widths

The total power is formed, the half-maximum level taken from its global peak with no
baseline subtraction, and the crossings found by linear interpolation between the two
samples that bracket them. Which two samples is the whole question when a trace has
structure:

- {py:func}`~solprop.stats.fwhm_t` walks **outwards from the peak** and stops at the first
  sample below half maximum on either side. A dispersive wave, a soliton that has walked
  off, a noise pedestal — none of them widens the answer, which is why this is the number
  to read as a pulse duration.
- {py:func}`~solprop.stats.fwhm_t_outer` takes the **first and last** sample anywhere in
  the window above half maximum, so it spans every satellite.

On a clean single pulse the two agree; the gap between them says how much sits outside
the central feature.

The interpolation is written as $x_1 + (L - y_1)(x_2 - x_1)/(y_2 - y_1)$. Forming the
slope $m$ and intercept $c$ of the line and evaluating $(L - c)/m$ instead would, on a
time axis of order $10^{-12}$ with an intercept of order $10^{12}$, throw away most of
the significand.

```{note}
A trace that never falls to half its maximum — a continuous wave, an empty mode, a pulse
that fills the window — has no width, and the answer is `nan`. Likewise a mode carrying
no light has no spectral centroid. These are the two places in the catalogue where a
value is genuinely undefined, and each docstring says so; a *missing dependency*, by
contrast, raises.
```

## Profiles and dispersion along z

`density`, `pressure` and `core_radius` are one implementation,
{py:func}`~solprop.stats.profile_value`, reading three different
{py:class}`~solprop.profiles.ZProfile` paths at the plane's position; the catalogue gives
each its own units.

{py:func}`~solprop.stats.zdw` root-finds $\beta_2 = 0$ for each mode, which is why it is
off by default. {py:func}`~solprop.stats.collect` narrows the search window to the grid's
own band with {py:func}`~solprop.stats.inband_wavelengths`, because a zero-dispersion
point outside the simulated band says nothing about the run and hunting for one there
drives every material model into extrapolation. Each call stands alone -- no search
starts from the previous plane's answer, so one failure cannot poison the later planes --
and a mode with no zero in the window raises rather than returning a missing value that
propagates.

## Far-field spectra

The six far-field statistics of a Cartesian or line run
([far-field apertures](statistics.md#far-field-apertures-and-angular-regions)) are one
pass over the stored spectrum $E_{c,q,k}$ of column $c$, polarisation $q$ and frequency
bin $k$, with the geometry's Parseval weights $w_c$ (`spectral_weights`), its on-axis
weights $\alpha_c$ (`on_axis`) and the spectral density
$\operatorname{sd}(E)$ of {py:func}`~solprop.fourier.spectral_density`
($2\delta t^2|E|^2/\pi$ on a field grid, $\delta t^2|E|^2/2\pi$ on an envelope grid).
Column $c$ carries the stored axes $\mathbf k^{(s)}_c$ and the physical transverse
wavevector $\boldsymbol\kappa_c=-\mathbf k^{(s)}_c$. Aperture $a$, of centre
$\mathbf X_a$, half-width $r_a$ ($D/2$, or $D_q/2$ under quadrature weighting) and
distance $z_a$, sees the column's paraxial image at the distance

$$
\rho_a(c,k)=\sqrt{\bigl(s_ak^{(s)}_{x,c}+X_a\bigr)^2+\bigl(s_ak^{(s)}_{y,c}+Y_a\bigr)^2},
\qquad s_a=\frac{z_ac}{\omega_k},
$$

evaluated as written, without `hypot` or a fused multiply-add, so a hard edge decides
identically on every backend. The weight $W_a=W(\rho_a)$ of
{py:func}`~solprop.spatial.aperture_profile` is exactly zero at $\omega_k\le0$ and where
$s_a|\mathbf k^{(s)}_c|\ge z_a$. Then

$$
\begin{aligned}
S_a(\omega_k)&=\frac{\varepsilon_0c}{2}\sum_cw_cW_a^p\sum_q\operatorname{sd}(E_{c,q,k}),
&p&=2\ \text{(amplitude)},\ 1\ \text{(quadrature)},\\
S^{\rm ax}_a(\omega_k)&=\frac{\varepsilon_0c}{2}\sum_q\operatorname{sd}\Bigl(\sum_c\alpha_cW_aE_{c,q,k}\Bigr),\\
S^R_j(\omega_k)&=\frac{\varepsilon_0c}{2}\sum_{\boldsymbol\kappa_c\in R_j}w_c\sum_q\operatorname{sd}(E_{c,q,k}),\\
S^{\rm beam}(\omega_k)&=\frac{\varepsilon_0c}{2}\sum_cw_c\sum_q\operatorname{sd}(E_{c,q,k}),\\
S^P(\omega_k)&=\frac{\varepsilon_0c}{2}\sum_cw_c\,\frac{c\operatorname{Re}k_z(c,k)}{\omega_k}\sum_q\operatorname{sd}(E_{c,q,k}),\\
P_j&=\frac{\varepsilon_0c}{2}\sum_cw_c\sum_q\sum_k\nu_k\,r_j(c,k)\,|E_{c,q,k}|^2,
\end{aligned}
$$

for `aperture_spectrum`, `aperture_on_axis_spectrum`, `region_spectrum` (a region
$R_j$ is a half-open box of $\boldsymbol\kappa$), `beam_spectrum`, `poynting_spectrum`
and `absorbed_power`. In $S^P$, $k_z$ is the principal root of the run's compensated
square, the root the stages use, and the obliquity is zero on inactive and evanescent
bins. In $P_j$, $\nu_k$ are the exact energy weights of
{py:func}`~solprop.fourier.energy_weights` and the rates are the spectral absorber's
$\alpha_{\rm abs}(\omega_k)$ on every bin, the transverse-wavevector rate (zero until the
operator carries one), $2\min(|\operatorname{Im}k_z|,\alpha_{\rm ev})$ on active
evanescent bins and $2\operatorname{Im}k_z$ on active propagating bins. With $W\equiv1$,
$S_a$ is $S^{\rm beam}$ and $S^{\rm ax}_a$ is `on_axis_spectrum`; $\delta\omega
\sum_kS^{\rm beam}_k$ is `energy` on every interior bin, and $-\sum_jP_j$ is the
derivative of `energy` along $z$ under the linear flow.

