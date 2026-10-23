# Raman response models

A response function describes how a molecular vibration, rotation or silica network continues to affect the field after excitation. This chapter gives the oscillator strengths, dephasing laws and normalisations used by the [Raman workflow](raman.md). Keep the response model, its thermodynamic state and the temporal resolution consistent.

## The models

Each model specifies an undamped response and a dephasing law. The latter
sets how quickly molecular coherence decays and can depend on the density
along a pressure gradient. {class}`solprop.nonlinear.raman.Dephasing` represents
that dependence. A built-in
gas's response is optmat's, mapped onto the same terms
([Raman data](raman-data.md#the-same-convention-mapped-onto-terms)); the vibrational mode and the
rotational ladder below are what a caller's own molecular constants, a
{class}`~solprop.physdata.raman_types.MolecularRaman` on a custom species, are built
into.

## The damped oscillator

Every gas-phase line is the impulse response of a driven oscillator in the weak-damping
limit, {class}`solprop.nonlinear.raman.DampedOscillator`:

$$
h(t) = K \sin(\Omega t)\, \mathrm{e}^{-t/\tau_2}, \qquad t > 0 .
$$

Its integral is $K\Omega/(\Omega^2 + \tau_2^{-2})$, so
{func}`solprop.nonlinear.raman.normalised_damped_oscillator` scales $K$ by the
reciprocal of that factor and gives a line of a stated total strength — the form a
fibre GNLSE states its Raman contribution in.

## The vibrational mode

{func}`solprop.nonlinear.raman.vibrational` builds one oscillator at the mode frequency
with

$$
K = \frac{\left(4\pi\varepsilon_0\, \partial\alpha/\partial Q\right)^2}
         {2\mu\Omega_v\,\varepsilon_0} ,
$$

the $4\pi\varepsilon_0$ converting the tabulated *volume* polarisability derivative to
an SI one, and the trailing $\varepsilon_0$ the [response convention](raman.md#the-response-function-and-its-units). The
constants are the species' own, a
{class}`~solprop.physdata.raman_types.VibrationalRaman`.

Here $Q$ is a displacement coordinate (the bond length for N2), $\mu$ its effective
mass and $\Omega_v$ the angular frequency of the mode. The induced-dipole energy is
$-\alpha_{\rm SI}(Q) E_{\rm real}^2/2$, so the force is
$(d\alpha_{\rm SI}/dQ) E_{\rm real}^2/2$; solving the driven oscillator and
multiplying its displacement by the same derivative gives the denominator two. The
cycle average belongs to the drive, not to $K$: the envelope drive is already
$|A|^2/2$ ([Raman numerics](raman-numerics.md#the-drive-and-why-it-differs-between-the-grids)), and a further
half in $K$ — a denominator of four — would count it twice and halve every vibrational
amplitude. An independent quantum-oscillator impulse test verifies the strength. It
checks the common formula, not the calibration of a supplied derivative and effective
mass.

For a mass-weighted normal coordinate the derivative and kinetic energy use a
different convention: convert both consistently before constructing the model.

## The rotational ladder

A linear molecule's levels are $E_J = 2\pi\hbar c\,[B J(J+1) - D (J(J+1))^2]$ and the
Raman-active transitions are the $S$ branch, $J \to J+2$.
{func}`solprop.nonlinear.raman.rotational_nonrigid` returns a
{class}`solprop.nonlinear.raman.RotationalLines` holding the levels, their Boltzmann
populations $p_J$ with the nuclear-spin statistical weights, the transition frequencies
and the line weights

$$
K\,\frac{(J+1)(J+2)}{2J+3}
\left(\frac{p_{J+2}}{2J+5} - \frac{p_J}{2J+1}\right),
\qquad
K = -\frac{(4\pi\varepsilon_0)^2}{\varepsilon_0}\,\frac{2}{15}\,
    \frac{\Delta\alpha^2}{\hbar} .
$$

$K$ is negative and so is the bracket, because the population per magnetic sublevel
falls with $J$; the lines therefore come out positive, with the same sign as a
vibration. The ladder is truncated wherever $E_J$ stops increasing with $J$: beyond
that the two-term expansion has left its range of validity and the levels it predicts
are an artefact. For hydrogen's constants ($B = 58.9$, $D = 0.05$ cm⁻¹), whose
centrifugal distortion is large, that is $J(J+1) > B/2D = 589$, so $J \le 24$. The ladder otherwise runs from
`MolecularOptions.min_j` to `max_j` (default 0 and 50), and the populations are
normalised over the levels retained.

One linewidth is used for every $J$. The $J$ dependence is real and not modelled, and
this is the model's weakest point.

**A temperature along z.** The populations are a Boltzmann ladder fixed for the run:
the undamped rows the kernel samples are built once per grid, so populations that
followed a temperature varying along the fibre would need a new kernel at every
position, which solprop does not do. A run whose gas temperature is a path therefore
computes them at **one reference temperature**: `MolecularOptions(temperature=...)`
if given, otherwise the mean over the run $[0, L]$ weighted by **that species' own**
number density,

$$\bar T_s = \frac{\int_0^L \rho_s\,T\,\mathrm{d}z}{\int_0^L \rho_s\,\mathrm{d}z}$$

({func}`~solprop.nonlinear.mixtures.raman_temperature`, by adaptive quadrature across the
breakpoints of both). The mean warns with a
{class}`~solprop.exceptions.RamanTemperatureWarning` naming the species, the reference
and the spread of $T(z)$ over the run, since the populations do not follow $T(z)$; an
explicit `MolecularOptions(temperature=...)` is the caller's own choice and logs the
same message at INFO instead. The populations are the species' own, so the mean
weights the positions where that species is: in a gradient it leans to the dense,
high-pressure end, and in independent populations each species takes its own — nitrogen
fed from one end (1 to 0.1 bar) and helium from the other, the gas warming from 294 K
to 400 K over a centimetre, holds nitrogen at 330.4 K, where the total density would
give 352.9 K. The average never reaches past the fibre, even for a constant pressure
(whose density has no end) or a temperature path longer than the run. A molecule without
a rotational ladder, or a run with
`rotation=False`, is not warned. The linewidths are unchanged: they follow the density
only, as at a constant temperature, and their own temperature dependence is not
modelled either way. A constant temperature is used as it is.

## Fused silica

A glass has no resolved rotational ladder. Its scalar Raman response is the shape $h_R$
of optmat's silica bundle ({class}`~solprop.physdata.glasses.SilicaBundle`): Hollenbeck
and Cantrell's fit, of unit integral, with the bundle's own Raman fraction
([materials](material-selection.md#fused-silicas-nonlinear-response)), sampled on the run's
delays ({class}`~solprop.nonlinear.raman.SampledShape`). The shape has no density
dependence: a glass is a fixed medium.

A caller's own line set of damped, Gaussian-broadened lines
({class}`~solprop.physdata.raman_types.IntermediateBroadeningRaman`) is evaluated by
{class}`solprop.nonlinear.raman.IntermediateBroadening`,

$$
h(t) \propto \sum_i A_i\, \mathrm{e}^{-\gamma_i t}\,
             \mathrm{e}^{-\Gamma_i^2 t^2/4}\, \sin(\omega_i t) .
$$

The widths *are* the dephasing, so this model has no density dependence either. The
amplitudes are relative, so the model is normalised to a stated integral, and **the
normalisation is analytic**:

$$
\int_0^\infty \mathrm{e}^{-\gamma t}\mathrm{e}^{-\Gamma^2t^2/4}\sin(\omega t)\,dt
= \mathrm{Im}\left[\frac{\sqrt{\pi}}{\Gamma}\,
  w\!\left(\frac{\omega + \mathrm{i}\gamma}{\Gamma}\right)\right],
$$

with $w$ the Faddeeva function {data}`scipy.special.wofz`, the numerically stable
spelling of $\mathrm{e}^{z^2}\mathrm{erfc}(z)$ in the half plane these arguments occupy.

## Fused silica, as a vector response

The orientation average, tensor channels and envelope conventions shared by every vector
Raman response are derived in [polarisation-dependent Raman](raman-vector.md); this
section gives silica's own tensor model. A second silica model lives beside the scalar
one and does not replace it:
{class}`solprop.nonlinear.raman.SilicaTensorResponse`, the delayed **tensor** response
of Lin and Agrawal, which a material region of a general modal guide declares beside its
{class}`~solprop.modes.material.MaterialKerr` (see
[general modal propagation](../waveguides/general-modal.md)). By default it is optmat's tensor
bundle, `SilicaBundle.resolve(tensor=True)`; Lin and Agrawal's own parameters are a
{class}`~solprop.physdata.raman_types.LinAgrawalRaman`, given explicitly. It is a
different fit of the same physics — two normalised shapes rather than one,

$$ h_a(t) = \frac{\tau_1^2+\tau_2^2}{\tau_1\tau_2^2}\,
            \mathrm{e}^{-t/\tau_2}\sin(t/\tau_1), \qquad
   h_b(t) = \frac{2\tau_b - t}{\tau_b^2}\,\mathrm{e}^{-t/\tau_b} , $$

with $\tau_1 = 12.2$ fs, $\tau_2 = 32$ fs and $\tau_b = 96$ fs — and it carries what a
scalar fit cannot: the **isotropic** response $a = w_a h_a$ and the **anisotropic** one
$b = w_b h_b + w_{ab} h_a$ are separate functions of delay, so a cross-polarised probe
sees $a$ alone where a co-polarised one sees $a + b$. The seam's two channels are the
trace and the traceless rank-two one, whose **shapes** integrate to
$w_a + (w_b + w_{ab})/3$ and $2(w_b + w_{ab})/3$ — **5/6** and **1/6** for Lin and
Agrawal's weights, because $w_a = 0.75$, and something else for any others — and sum to
one. The
scalar limit is the ordinary fibre model with $h_R = a+b$ integrating to one. The Raman
fraction is $f_R = 0.245$ of the region's own $\chi^{(3)}$, and the instantaneous Kerr
term keeps $1-f_R$ of it.

**The kernels carry $\tfrac32 f_R$, not $f_R$.** $f_R$ and $n_2$ are *envelope*
quantities: $f_R$ is the share of the effective nonlinearity **on the fundamental** the
delayed term carries. In the field domain the delayed polarisation's drive is the cycle
average $|A|^2/2$, so a kernel of integral $X$ contributes $\tfrac12 X|A|^2A$ against
the instantaneous cubic's $\tfrac34\chi^{(3)}|A|^2A$ — it acts like a Kerr term of
$\tfrac23X$ — and carrying the share $f_R$ of one $\chi^{(3)}$ needs
$X = \tfrac32 f_R\chi^{(3)}$
({data}`~solprop.nonlinear.raman.SILICA_TENSOR_AMPLITUDE`, applied in `kernels` and
nowhere else). It is the same $\tfrac32$, for the same reason, as the scalar mapping
under [Raman in a fibre GNLSE](raman.md#raman-in-a-fibre-gnlse), and with it the two shares
add to exactly one $\chi^{(3)}$ on the fundamental: $\tfrac23\cdot\tfrac32 f_R + (1-f_R) = 1$.

**The endpoint quadrature is the trapezoid.** $h_b(0^+) = 2/\tau_b$ is not zero — this
is the one kernel in the package that does not vanish at zero delay — so a kernel
sampled on $k\,\delta t$ and integrated by the rectangle-rule transform product misses
$\delta t\,h(0^+)/2$ and converges only to first order. The sample **at** the jump
therefore carries the mean of its two one-sided limits, which is half $h(0^+)$; that
turns the sum into the trapezoid rule, restores second order, and costs no causality,
because the half-weighted sample multiplies $Q(t)$ itself and no earlier one. The late
lobe of $h_b$, where $t > 2\tau_b$ and it goes negative, is left unclipped: it is part
of the response and not a gain. Every other kernel in this package vanishes at zero
delay, where the two rules coincide, so nothing else is affected.

This model is reached only through a material region of a general modal guide.
{func}`~solprop.nonlinear.raman.response_for` does not return it: `"SiO2"` there is the
scalar Hollenbeck–Cantrell shape of optmat's default bundle, which is what a fibre GNLSE
uses.

## Dephasing, and what happens at zero density

{class}`solprop.nonlinear.raman.Dephasing` turns a
{class}`solprop.physdata.raman_types.Linewidth` — full width at half maximum
$C + A/n + Bn$ in hertz, with $n$ the density in amagat — into the factor
$\mathrm{e}^{-t/\tau_2}$, $\tau_2 = 1/(\pi\,\mathrm{FWHM})$. The middle term is Dicke
narrowing, which makes lines *narrower* as the density rises until collisional
broadening takes over.

A coherence decaying as $\exp(-t/\tau_2)$ has a Lorentzian spectral line with
FWHM $1/(\pi\tau_2)$ in Hz, or $1/(\pi c\tau_2)$ in inverse metres.
Its HWHM is half that number. The intensity of a CARS signal proportional to
that coherence squared decays twice as fast, so an intensity lifetime is
$\tau_2/2$. These molecular linewidth conventions are separate from the
angular-frequency Lorentzian/Gaussian parameters of the silica model.

At zero density there are three cases, and which applies is fixed by the coefficients:

| linewidth | $\tau_2(0)$ | meaning |
|---|---|---|
| $A > 0$ | $0$ | the Dicke term diverges: no coherence at all |
| $A = C = 0$ | $\infty$ | the linewidth vanishes: an undamped coherence |
| $A = 0,\ C > 0$ | $1/(\pi C)$ | the density-independent term survives |

All three are handled without special cases in a run (a collapsed coherence is given a
finite decay rate that kills it within one sample; see [Raman numerics](raman-numerics.md#sampling-the-kernel-and-the-density-it-depends-on)).
The density multiplies the polarisation anyway, so at zero density the choice is
invisible in the result and matters only for whether the result is a finite number.
