# Hollow rectangular waveguides

A hollow rectangular guide confines the field between two pairs of walls. Its aspect
ratio and polarisation affect both dispersion and loss. The equations below use the
large-core approximation; the field is truncated at the walls and the wall response
enters the complex effective index.


{class}`~solprop.modes.rectangular.RectMode` describes a gas-filled hollow guide of
rectangular cross-section, half-widths $a$ along x and $b$ along y, with dielectric or
metal walls. Like the capillary it is a grazing-incidence model of a guide many
wavelengths wide, and it shares the capillary's gas fill and loss policy. The mode index
`m` counts along x and `n` along y. Each positive integer pair has two distinct
polarisation partners, `polarisation="x"` (the default) and `"y"`: their fields occupy
different electric-field components and their propagation constants usually differ, so
they are separate mode rows.

```{code-block} python
from solprop.modes import RectMode, ModeCollection
from solprop.physdata import silver

mode = RectMode(50e-6, 12.5e-6, "Ar", 5.0, cladding=silver())  # x-polarised (1, 1)
pair = ModeCollection.with_polarisations([mode])  # the x and y partners
```

The Laakmann–Steier wall-reflection model gives

$$
n_{\mathrm{eff}}=\sqrt{1+\chi-X-Y},\qquad
X=\left(\frac{m\lambda}{4a}\right)^2
  \left(1-i\frac{\lambda a_c}{2\pi a}\right)^2,
\quad
Y=\left(\frac{n\lambda}{4b}\right)^2
  \left(1-i\frac{\lambda b_c}{2\pi b}\right)^2.
$$

Write $s=\sqrt{\epsilon_{\mathrm{cl}}-\epsilon_{\mathrm{co}}}$ with
$\epsilon_{\mathrm{co}}=1+\chi$. For x polarisation,
$(a_c,b_c)=(\epsilon_{\mathrm{cl}}/(\epsilon_{\mathrm{co}}s),1/s)$; exchange these for
y polarisation. Both the contrast and the TM-like coefficient's permittivity
*ratio* use the **filled-core** permittivity. The reason is the wall's reflection at
grazing incidence. With the wall normal along x, the cladding's normal wavenumber is
$\sqrt{k^2(\epsilon_{\rm cl}-\epsilon_{\rm co})+\kappa_x^2}\simeq ks$, and expanding
the two Fresnel coefficients about grazing incidence gives
$r_s\simeq-\exp(-2\kappa_x/(ks))$ and
$r_p\simeq-\exp(-2(\epsilon_{\rm cl}/\epsilon_{\rm co})\kappa_x/(ks))$; the paired-wall
round trip $\exp(4ia\kappa_x)r^2=1$ then gives $\kappa_x$ as in $X$ above with
$q_{\rm TE}=1/s$ and $q_{\rm TM}=\epsilon_{\rm cl}/(\epsilon_{\rm co}s)$. The printed
paper, which treats an evacuated core, writes $\epsilon_{\rm cl}/s$ with
$s=\sqrt{\epsilon_{\rm cl}-1}$; the two agree only when the core is evacuated. At 800
nm, argon and the dimensions above, the loss
changes from that evacuated-core form are

| wall | x, 1 bar | x, 10 bar | y, 1 bar | y, 10 bar |
|---|---|---|---|---|
| constant 1.45 | +0.0216% | +0.219% | −0.0276% | −0.276% |
| silver | −0.0193% | −0.194% | −0.0533% | −0.534% |
| aluminium | −0.0267% | −0.268% | −0.0545% | −0.546% |

and the real refractivity moves by up to 1.1e-4 relative (aluminium, y, 1 bar).

Complex wall corrections affect phase as well as attenuation. Each paired wall's
TE- or TM-like boundary condition gives a complex transverse wavenumber: for a complex
$q=A+iB$ the correction factor is $1+B/(ka)-iA/(ka)$, so keeping $A$ for the
attenuation while dropping $B$ from the phase would be inconsistent with the same
expansion. solprop retains the whole complex $\kappa_x^2+\kappa_y^2$ in
$\beta^2=k^2\epsilon_{\mathrm{co}}-\kappa_x^2-\kappa_y^2$, so the wall's contribution to
$\operatorname{Re}n_{\rm eff}$ is kept, as a principled continuation of the first-order
boundary approximation. The squared factors retain some higher orders without being a
complete higher-order solution, and the model is not exact at arbitrary wavelength or
aspect ratio. The paper's equations were checked through a text extraction, not the
typeset page, and the content of its published erratum
([doi:10.1364/AO.15.002029](https://doi.org/10.1364/AO.15.002029)) has not been checked;
the coefficients above rest on the derivation given here.

**Validity, and the warning.** The model is first order in each wall pair's grazing
angle, $m\lambda/4a$ (and $n\lambda/4b$), and in its wall correction,
$\lambda|q|/2\pi a$ (and $\lambda|q|/2\pi b$), with $\lambda$ the vacuum wavelength
(Laakmann and Steier's conditions 1–2: small grazing angles, well past Brewster's).
Wherever either ratio exceeds **0.1** over the evaluated band, the mode raises a
{class}`~solprop.exceptions.GrazingIncidenceWarning` naming the ratio, its value and the
wavelength — once per mode, its frozen copies along z included. The neglected terms are
second order in these ratios, about 1 % of the mode's decrement $X$ at 0.1 and growing
quadratically beyond it, and a grazing angle of 0.1 rad is still well short of glass's
Brewster grazing angle $\arctan(1/n)$ (0.6 rad for silica). A metal's TM-like
coefficient is large ($|q|\approx15$ for silver at 2 µm), so on a metal-walled guide the
wall correction usually reaches the threshold first. The warning does not refuse:
narrow the band towards shorter wavelengths, widen the guide or keep lower orders.

The analytic derivative chains through both material slopes and the wall corrections.
Refractivity is formed directly, by the exactly rationalised
$z/(\sqrt{1+z}+1)$ with $z=\chi-X-Y$, which is accurate to rounding in the large-guide
limit where the spelling matters. The {class}`~solprop.modes.fill.GasFillMixin` shares
gas resolution, mixtures, species
absorption, parameter freezing, the four-way loss policy, gain clamp and evanescent
floor with Marcatili and Vincetti modes.

The field is a product of cosines for odd indices and sines for even indices, supported
inside the rectangle. Its closed forms are

$$N=\frac{\epsilon_0 c}{2}ab,\qquad A_{\mathrm{eff}}=\frac{16ab}{9}.$$

The basis normalisation uses the **vacuum** admittance
$\tfrac12\sqrt{\epsilon_0/\mu_0}$, as for every mode: assembly and output convert
between this internal basis and the true-power amplitude with
$b=\sqrt{\operatorname{Re}n_{\rm eff}}a$, and run statistics evaluate the Poynting
flux ([power and intensity](../results/power-and-intensity.md)).

Cartesian Gauss quadrature integrates these trigonometric products, which have no
finite polynomial exactness degree, so the bound is measured:
{func}`~solprop.modes.quadrature.cartesian_nodes` is $6h+6$ nodes per axis for
highest order $h$, covering the measured 12, 18, 22, 26 and 35 nodes that reach a
relative $10^{-12}$ at orders 1, 2, 3, 4 and 6. For `(n, m) = (3, 2)` 22 nodes per
axis are enough and 20 leave $6.8\times10^{-11}$. A rule below the bound warns, as a
coarse azimuthal rule does, and both Cartesian defaults of
`prop_waveguide` take it.

Gas-filled rectangular guides run through
{func}`~solprop.driver.interface.prop_waveguide` and
{meth}`~solprop.driver.simulation.Simulation.from_waveguide`, which take a supplied
mode set rather than a radius and a mode count; `n_r` and `n_theta` are then the x
and y node counts. A Cartesian product rule has no Kronrod extension, so those runs
record no `quadrature_error` and the quadrature alarm is off; the node bound above
is what stands in its place.

Both widths may be profiles. Uniform scaling is available when both half-widths are
constant or both axes share the same profile object; independent tapers support the
linear operator and mode-averaged area, while a modal transform refuses a
non-scale-invariant taper. Two width paths and a density path — three parameters —
are carried on the linear operator's z table rather than by its
[expansion in the parameters](../numerics/linear-tables.md#the-expansion-in-the-parameters),
which every CPU backend runs (the native backend's GPU builds do not). Constant widths and a constant fill
report `z_constant=True`, so the fixed-quadrature transform evaluates the
fields once rather than at every step.

Reference: K. D. Laakmann and W. H. Steier, *Waveguides: characteristic modes of hollow
rectangular dielectric waveguides*, Applied Optics **15**, 1334–1340 (1976),
[doi:10.1364/AO.15.001334](https://doi.org/10.1364/AO.15.001334).
