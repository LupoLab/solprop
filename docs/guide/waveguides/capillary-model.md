# Capillary modes and dispersion

A hollow capillary combines material dispersion in its gas fill with waveguide
dispersion and leakage at the wall. This chapter gives the Marcatili model, its
normalisation and its limits. Start with [running a capillary](capillaries.md) or the
[multimode RDW lesson](../../introduction/lessons/plot_01_rdw.rst) for a complete
calculation.

## The Marcatili model

Marcatili and Schmeltzer solved a hollow dielectric tube asymptotically in the limit where
the core radius $a$ is large compared with the wavelength. With $k = \omega/c$,
$\varepsilon_{\rm co}$ the core permittivity and $u_{nm}$ a Bessel zero,

$$n_{\rm eff} = \sqrt{\varepsilon_{\rm co}
  - \left(\frac{u_{nm}}{ka}\right)^2\left(1 - \frac{i v_n}{ka}\right)^2}$$

is the **full** model, and expanding to leading order in $1/(ka)$ gives the **reduced**
one,

$$n_{\rm eff} = 1 + \frac{\varepsilon_{\rm co} - 1}{2}
  - \frac{c^2u_{nm}^2}{2\omega^2a^2}
  + i\,\frac{c^3u_{nm}^2v_n}{a^3\omega^3}.$$

Three terms, and each does one thing. The first is the gas. The second is the waveguide,
pushing the index *below* one and giving a hollow capillary its anomalous dispersion. The
third is the loss, and it goes as $a^{-3}$ — which is why a capillary is made as wide as
the beam allows.

The mode indices pick the Bessel zero and the cladding term:

| family | $u_{nm}$ | $v_n$ | azimuthal order |
|---|---|---|---|
| $\mathrm{HE}_{nm}$, $n \ne 0$ | $j_{n-1,m}$ | $(\varepsilon_{\rm cl}+1)/(2\sqrt{\varepsilon_{\rm cl}-1})$ | $\lvert n-1\rvert$ |
| $\mathrm{TE}_{0m}$ | $j_{1,m}$ | $1/\sqrt{\varepsilon_{\rm cl}-1}$ | 1 |
| $\mathrm{TM}_{0m}$ | $j_{1,m}$ | $\varepsilon_{\rm cl}/\sqrt{\varepsilon_{\rm cl}-1}$ | 1 |

Here $j_{p,m}$ is the $m$-th zero of $J_p$ and $\varepsilon_{\rm cl}$ the cladding
permittivity ([the cladding](#the-cladding)). A mode is declared by its radius in metres,
its fill and its pressure in bar, with the family and indices as keywords:

```{code-block} python
from solprop.modes import MarcatiliMode

he11 = MarcatiliMode(125e-6, "He", 1.0)  # HE11, x-polarised, full model
he12 = MarcatiliMode(125e-6, "He", 1.0, m=2)  # the next radial order
te01 = MarcatiliMode(125e-6, "He", 1.0, kind="TE", n=0)
reduced = MarcatiliMode(125e-6, "He", 1.0, model="reduced")
empty = MarcatiliMode(125e-6)  # evacuated: gas=None, pressure 0
```

The defaults are `kind="HE"`, `n=1`, `m=1`, `model="full"`, `loss=True`, a fused-silica
cladding and a fill at 293.15 K. A gas named here is the built-in gas, optmat's
([SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md)): its polarisability and its
equation of state (one species per gas, shared by every mode that names it), as in a
run's fill. `phi` rotates the mode about the axis
([normalisation and effective area](#normalisation-and-effective-area)),
`susceptibility=` gives another law (`gamma(gas, model=...)` names another of optmat's,
or a caller's own) beside optmat's equation of state, and `mixture=` replaces the gas
and pressure
([tapers and pressure gradients](#tapers-and-pressure-gradients)).

**What it approximates.** Both forms are asymptotic in $1/(ka)$: they describe a guide
whose bore is many wavelengths across, a low-order mode ($u_{nm} \ll ka$) and a wall met
at grazing incidence. The loss is the leakage of that grazing reflection, attenuated
independently for each mode; the coupling the wall causes *between* modes is left out,
and is what [the coherent wall](coherent-wall.md#the-coherent-wall) adds. The cladding term $v_n$ assumes
$\operatorname{Re}\varepsilon_{\rm cl} > 1$: a metal wall, or silica in the extreme
ultraviolet, lies outside the model ([loss](#loss) says what happens there).

A positive imaginary part is loss ([conventions](../../dev/conventions.md) §5), so the
reduced model's loss term enters with a **plus** sign. Getting that sign backwards turns
loss into gain with no other symptom, so solprop evaluates the index through one
arithmetic kernel, whichever path asks for it, and a test asserts that loss is loss.

A hollow capillary has no mode at zero frequency — the waveguide term diverges — so the
index refuses $\omega \le 0$ rather than returning a clamped value. A run never asks:
the linear operator evaluates modes on the active band only.

## Forming $n_{\rm eff} - 1$

A gas fill has $\chi \sim 10^{-4}$ at one bar and far less at low pressure, and the
waveguide term of a wide bore is smaller still, so $n_{\rm eff}$ is $1$ plus a number
many orders of magnitude smaller. Building $n_{\rm eff}$ and subtracting one throws away
as many digits as that number is small. Both models therefore form the refractivity
**directly**:

$$n_{\rm eff} - 1 = \frac{z}{\sqrt{1+z}+1}, \qquad z = \chi - w, \qquad
w = \left(\frac{u_{nm}}{ka}\right)^2\left(1 - \frac{i v_n}{ka}\right)^2$$

for the full model, and term by term for the reduced one. The difference is measurable:
an implementation that forms $n_{\rm eff}$ and then subtracts one can agree with this
one on $n_{\rm eff}$ to $4\times10^{-16}$ and still differ on $n_{\rm eff}-1$ by
$10^{-11}$, five orders worse than the quantity it is derived from. The same
cancellation, left in, caps the Chebyshev tail of the linear operator's expansion along
z at about $10^{-12}$ instead of $2\times10^{-16}$.

The rationalised form keeps the small $z$ in the numerator over a denominator near two,
so it is accurate to rounding for a single gas and a mixture alike, and
{class}`~solprop.modes.rectangular.RectMode` and the slab guide share it. The
apparently equivalent $\mathrm{expm1}\!\left(\tfrac12\,\mathrm{log1p}(z)\right)$ is not
exact for a complex $z$: NumPy's complex `log1p` forms $1 + z$ and rounds there, which
costs $1.3\times10^{-12}$ relative at $z = 10^{-4} + 10^{-7}i$ and eleven digits in the
large-guide limit, where $z \sim -10^{-11}$.

## The frequency derivative

{meth}`~solprop.modes.capillary.MarcatiliMode.dneff_domega` is analytic, in closed form,
for both models. That is a requirement rather than an optimisation: the co-moving frame
rides on $\beta_1$, and a finite-difference $\beta_1$ has a roughness floor near
$5\times10^{-8}$ that caps the accuracy of the whole cumulative operator at $10^{-8}$.
The analytic form is also cheaper than a difference stencil, which costs several full
index evaluations per frequency.

Writing $s = c/(\omega a)$, so that $s' = -s/\omega$, the reduced model differentiates to

$$\frac{\mathrm{d}n_{\rm eff}}{\mathrm{d}\omega}
= \frac{1}{2}\rho\frac{\mathrm{d}\gamma}{\mathrm{d}\omega}
+ \frac{u_{nm}^2s^2}{\omega}
+ i\,u_{nm}^2 s^3\left(v_n' - \frac{3v_n}{\omega}\right),$$

and the full model to $(\chi' - w')/(2n_{\rm eff})$ with $w'$ from the product rule. The
chain runs through the analytic derivative of the gas susceptibility
({func}`~solprop.physdata.sellmeier.gamma`) and of the cladding index, and $v_n'$ follows
from $\mathrm{d}v_n/\mathrm{d}\varepsilon_{\rm cl}$ in closed form per family.

$\beta_2$ is the one quantity solprop differences: the second derivative of a gas
susceptibility is not part of the material interface, so
{func}`~solprop.modes.base.beta2` is a Richardson difference of the *analytic* $\beta_1$.
It reaches about $5\times10^{-9}$ relative on an evacuated capillary and $1.3\times10^{-8}$
at worst against the symbolic result for a filled one, where a second difference of
$\beta$ itself reaches only about $10^{-6}$.

The frequency derivative has independent contour-integral and symbolic checks.
Their configurations and measured residuals are in the
[qualification record](../../dev/accuracy.md#11-capillary-modes-raman-and-the-moving-basis).
These checks assess the derivative of the stated model; they do not establish
that the Marcatili approximation describes a particular guide.

### The derivative along z

{meth}`~solprop.modes.capillary.MarcatiliMode.dneff_dz` is analytic too. The position
reaches the index only through the core radius and the fill's number density, so
$\mathrm{d}n/\mathrm{d}z = (\partial n/\partial a)\,a' + \sum_s(\partial n/\partial\rho_s)\,
\rho_s'$ with the declared profiles' own analytic slopes; for the full model, with
$n_{\rm eff}^2 = 1 + \sum_s\gamma_s\rho_s - W$ and $W = (u_{nm}s)^2(1 - iv_ns)^2$,

$$\frac{\mathrm{d}n_{\rm eff}}{\mathrm{d}z}
= \frac{1}{2n_{\rm eff}}\left(\sum_s\gamma_s\rho_s'
+ \frac{2u_{nm}^2s^2}{a}(1 - iv_ns)(1 - 2iv_ns)\,a'\right),$$

and the reduced model differentiates its polynomial. The loss setting acts on the slope's
imaginary part as it does on the index's, and a clamped or floored value has a zero slope.
The coherent wall on a varying guide reads the real part, the slope of the power weight,
for each channel's own normalisation ([a guide that varies along
z](coherent-wall.md#a-guide-that-varies-along-z)). It is checked
against a contour integral in each of the two scalars ($1.7\times10^{-10}$) and against
a fourth-order difference along z of the implemented index: $3\times10^{-10}$ on a taper,
and $4\times10^{-6}$ on a pressure gradient, whose gas slope is small enough that the
index's own $10^{-16}$ rounding sets the floor. A structured or tabulated index is not
Marcatili's, and `dneff_dz` refuses it by name.

## Loss

`loss=True` uses the model's own loss and `loss=False` removes it. A **float** scales the
model loss, a fitting knob; a **callable** is an absolute $\alpha(\omega)$ in m⁻¹ that
*replaces* it, for measured fibre loss. The two therefore mean different things, which is
deliberate and worth remembering. With a callable the real part of the index — and so
$\beta$ and $\beta_1$ — stays fully analytic; only the imaginary part is differenced, and
only if the callable carries no `derivative` attribute, in which case the constructor logs
that it will.

**The model's loss is never negative.** Where the complex square root lands on the branch
that reads as gain — a cladding index below unity: a metal, or silica in the extreme
ultraviolet, outside Marcatili's regime — the imaginary part is clamped at zero, its
derivative with it, and the mode logs the fact once: a passive waveguide cannot amplify.
A callable `loss` is the caller's and is not clamped: an explicitly negative $\alpha$ is
gain the caller asked for, which the linear operator's loss policy allows.

```{note}
The full model floors $\mathrm{Re}\,n_{\rm eff}$ at $10^{-3}$, so that the
mode-averaged prefactor $1/n_{\rm eff}$ cannot blow up far below the band. It is
{data}`~solprop.modes.capillary.EVANESCENT_INDEX_FLOOR`, documented rather than buried in
a kernel, and where it binds the derivative is reported as zero because the floored value
is constant. The reduced model has no floor.
```

## The cladding

The cladding enters the Marcatili index only through $v_n$, which depends on the
cladding permittivity alone, and $v_n$ carries the whole of the loss. The loss of a
capillary simulation is therefore only as good as the cladding data, which must be cited
and auditable. The default, `fused_silica()`, is optmat's recommended silica,
`fit/SiO2`: Franta's universal dispersion model refitted to absolute refractometry,
causal and absorbing from 24.8 nm to 125 µm, with its exact derivative
([materials](../media/material-selection.md#solids)). A lossless-only route (the step-index modes) takes its transparent
approximation. Evaluating a model outside its qualified range emits
{class}`~solprop.exceptions.MaterialRangeWarning`, as the gas models do. A cladding of
the caller's own is any {class}`~solprop.physdata.glasses.GlassIndex`: a
{class}`~solprop.physdata.glasses.SellmeierGlass`, a
{class}`~solprop.physdata.glasses.TabulatedGlass`, or a
{class}`~solprop.physdata.glasses.CompositeGlass` combining one's real part with
another's absorption, such as Malitson's real index with Franta's absorption.

## Normalisation and effective area

The transverse profile of a hybrid mode is $J_{n-1}(u_{nm}r/a)$ times a unit vector that
makes the angle $n(\phi - \theta)$ with the local radial direction, so its **magnitude**
does not depend on $\theta$ — which is why a set of $\mathrm{HE}_{1m}$ modes needs no
azimuthal quadrature at all. In
Cartesian components that unit vector is
$(\cos((n-1)\theta - n\phi),\,-\sin((n-1)\theta - n\phi))$: $\phi$ rotates the whole
pattern about the axis, from $x$ towards $y$, and $\phi = 0$ is the `even` member of the
degenerate pair for every $n$ — the $\cos((n-1)\theta)$ pattern about $x$, uniformly
x-polarised for $n = 1$, so that the default capillary mode is polarised along $x$
(see the [design decisions](../../dev/decisions.md)). Both the normalisation and the
effective area are closed forms,

$$N = \frac{\pi}{2}a^2 J_{n'}(u_{nm})^2\sqrt{\varepsilon_0/\mu_0},
\qquad
A_{\rm eff} = a^2\,\frac{2\pi\,J_{n'}(u_{nm})^4/4}
  {\int_0^1 \xi\,J_{n-1}(u_{nm}\xi)^4\,\mathrm{d}\xi},$$

with $n' = n$ for a hybrid mode and 2 for a transverse one. Dividing the profile by
$\sqrt{N}$ gives the vacuum-normalised basis used with the internal amplitude $a$.
The monochromatic modal power is $\operatorname{Re}n_{\rm eff}|a|^2$; input and
output use $b=\sqrt{\operatorname{Re}n_{\rm eff}}a$. See
[power and intensity](../results/power-and-intensity.md) for the broadband flux definition.
The effective area is purely geometric: it does not depend on the frequency or the
fill.

A mode's `azimuthal_order` is the highest azimuthal harmonic of its Cartesian field
components, as the family table of [the Marcatili model](#the-marcatili-model) gives it:
$\lvert n-1\rvert$ for a hybrid mode and 1 for a transverse one. It is never negative —
$\mathrm{HE}_{-1,m}$ has order 2 — because it
sizes the azimuthal rule below.

## Tapers and pressure gradients

Both the core radius and the pressure accept a number or a
{class}`~solprop.profiles.ZProfile` ([profiles along z](profiles.md)), so a taper and a
pressure gradient are ordinary cases rather than special ones:

```{code-block} python
from solprop.modes import MarcatiliMode
from solprop.profiles import Linear, PressureGradient

taper = Linear(125e-6, 75e-6, 1.5)  # 125 to 75 µm over 1.5 m
fill = PressureGradient.two_point(1.5, 0.8, 0.0, radius=taper)  # 0.8 bar to vacuum
mode = MarcatiliMode(taper, "He", fill)
```

The effective index follows the local radius and gas density. A pressure gradient
changes material dispersion and nonlinear strength, while a taper also changes
the transverse profiles. Position enters through two scalars, $a(z)$ and $\rho(z)$, so
everything frequency-dependent can be sampled once as
{class}`~solprop.modes.capillary.MarcatiliTerms`:

```text
terms = mode.sampled_terms(omega)  # the expensive part, once
index = mode.neff_from_terms(terms, radius, density)  # per position, cheap
```

The two paths give the same answers, to rounding. The Chebyshev profile builder evaluates
the operator this way at hundreds of positions; the agreement of the two paths is checked
in the [qualification record](../../dev/accuracy.md#11-capillary-modes-raman-and-the-moving-basis).

A mixture takes the place of the gas and the pressure: `MarcatiliMode(15e-6,
mixture=Mixture.air(6.0))` sums the species' per-particle susceptibilities, weighted by
their number densities, before the complex root ([mixtures](../media/mixtures.md)). Its scalar
parameters follow the composition's shape: a fill at one total pressure declares
`core_radius` and `pressure`, so the expansion in two parameters still applies and a
one-component fill is bitwise the single-gas mode above; independent per-species
profiles declare one `density:<label>` path each, which the linear operator then
resolves on the z table.

**A multimode taper carries the taper coupling by default.** As the bore changes, the
modes themselves change shape, and writing the field in the local modes at every $z$
couples them: this **connection** converts a launch between the $\mathrm{HE}_{1m}$
orders. It is carried on a capillary's contiguous $\mathrm{HE}_{11}$ to
$\mathrm{HE}_{1N}$ set whose radius varies, lossless or not, with or without
[a coherent wall](coherent-wall.md#a-guide-that-varies-along-z); without one it is the only coupling
([the taper coupling without a wall](taper-coupling.md#the-taper-coupling-without-a-wall)). On a 50 to
40 µm transition over a fifth of a millimetre it moves 7 % of the energy out of
$\mathrm{HE}_{11}$ (with the wall), against $10^{-18}$ on the **adiabatic** route.
That route, the explicit opt-out `taper_coupling=False`, lets each channel follow its own
local index and loss and moves no power between channels. A moving set the connection is
not derived for (a tube lattice, a sparse or mixed set, tabulated modes, a slab) stays
adiabatic and warns once, by name
({py:class}`~solprop.exceptions.AdiabaticTaperWarning`).
