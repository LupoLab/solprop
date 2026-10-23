(primer-maxwell)=
# 1. Light in a medium: from Maxwell's equations to a propagation problem

You know the physics this chapter covers. Most of it is in the first chapters of any
optics textbook. It is worth going through again carefully, though, because every
convention in solprop is fixed here: the units, the sign of the Fourier transform, what
counts as "linear" and what as "nonlinear", and the relation between a real electric
field, its analytic signal and an envelope. Factors of 2 and signs of $i$ that go wrong
here propagate into everything else.

By the end of the chapter you should be able to write Maxwell's equations in the form the
propagation equation starts from, and say why the linear response is kept apart from the
nonlinear one. You should also be able to state what causality and the reality of the
field require of a susceptibility, and convert between the Fourier convention of this
primer and the one solprop uses internally. Finally, you should be able to explain how
the field, the analytic signal and the envelope are related, and where the factors of 2
between them go.

## 1.1 What a propagation calculation asks

The problem solprop solves can be stated in one sentence. You know the electric field of
a pulse on a plane $z=0$, for all times and all transverse positions, and you want to
know it on every later plane $z>0$ after it has travelled through a medium. That medium
might be a gas-filled capillary, a silica fibre, a slab of glass or a cell of argon.

Two features of this problem shape the whole primer. First, $z$ plays the role that time
plays in most evolution problems. The quantity marched forward is a whole time history
$E(x,y,t)$ on a plane, and the marching variable is a position. Reformulating Maxwell's
equations so that this works, with $z$ as the evolution variable and only the forward
direction kept, is what Chapter 2 is about.

Second, the field is a broadband object. A 10 fs pulse at 800 nm has a bandwidth of tens
of terahertz. After soliton self-compression or supercontinuum generation it may span
from the vacuum ultraviolet to the mid infrared. Every frequency sees its own refractive
index, its own loss, its own mode shape and its own share of the nonlinear response. The
natural representation is therefore spectral: one complex amplitude per frequency (and
per transverse mode or transverse wavenumber), coupled together only through the
nonlinearity.

Everything else is a choice about how much of this to keep. You can keep the full
transverse structure (Chapter 6) or a few guided modes (Chapters 4 and 5). You can keep
just one mode, replaced by an effective area (Chapter 3). In time, you can carry the real
optical field with its carrier or only an envelope around a carrier frequency (§1.6 and
Chapter 11). The medium's response can be a simple Kerr term or a sum of Kerr, Raman,
photoionisation and plasma contributions (Chapters 7–10). The [capability
overview](../introduction/capabilities.md) lists the combinations solprop supports; this
primer explains what each of them assumes.

## 1.2 Maxwell's equations in a medium

solprop works in SI units throughout ([conventions](../dev/conventions.md#1-units)). In SI,
Maxwell's equations for a non-magnetic medium are

$$
\nabla\cdot\mathbf D=\rho_f,\qquad \nabla\cdot\mathbf B=0,\qquad
\nabla\times\mathbf E=-\partial_t\mathbf B,\qquad
\nabla\times\mathbf H=\mathbf J+\partial_t\mathbf D,
$$

with $\mathbf B=\mu_0\mathbf H$. The medium enters through the displacement
$\mathbf D=\varepsilon_0\mathbf E+\mathbf P$, where $\mathbf P$ is the polarisation
(dipole moment per unit volume, in C/m²), and through the free current density
$\mathbf J$ (A/m²) of charges that are not bound. In this primer these are the electrons
freed by photoionisation.

The first important decision is how to split $\mathbf P$. Write

$$
\mathbf D=\varepsilon_0\,\epsilon*\mathbf E+\mathbf P_{\rm NL},
$$

where $\epsilon*\mathbf E$ is the linear, causal response of the medium, a time
convolution with a memory function, and $\mathbf P_{\rm NL}$ is everything else. This is
the form Kolesik and Moloney (2004, Eq. 2) start from, and Couairon et al. (2011,
Eqs. 3–4) write the same thing in the frequency domain as
$\hat{\mathbf D}=\varepsilon_0\epsilon(\omega)\hat{\mathbf E}+\hat{\mathbf P}$.

The split is not arbitrary. All of the linear chromatic dispersion and absorption should
go into $\epsilon$, so that only the truly nonlinear interactions are left in
$\mathbf P_{\rm NL}$ and $\mathbf J$ (Kolesik, Townsend and Moloney 2012, Sec. II B).
Kolesik and Moloney (2014, §2.1) put it as advice: include as much as you can in the
linear operator, so that the response term is small. You will see in Chapter 2 that this
is also what keeps the backward-travelling field small. It is the reason the
unidirectional equation works at all.

From here on, $\mathbf P$ without a subscript means the nonlinear polarisation, and
$\mathbf J$ the free-electron current. Chapter 10 shows that a current can always be
rewritten as an equivalent polarisation, with $\partial_t\mathbf P_J=\mathbf J$, so in
practice solprop handles both through the same machinery.

## 1.3 The linear response: susceptibility, causality and loss

For a linear, isotropic, homogeneous medium the convolution becomes a product in the
frequency domain:

$$
\tilde{\mathbf P}_{\rm lin}(\omega)=\varepsilon_0\chi(\omega)\tilde{\mathbf E}(\omega),
\qquad \epsilon(\omega)=1+\chi(\omega)=n^2(\omega).
$$

Here $n(\omega)$ is complex in general, and its imaginary part describes absorption. Two
physical requirements constrain $\chi(\omega)$, and both matter in practice.

**Reality.** The field and the polarisation are real functions of time. A real response
function has a spectrum that satisfies

$$
\chi(-\omega)=\chi^*(\omega)
$$

for real $\omega$. The same holds for $\epsilon$ and for $n$ with the branch chosen
continuously (Brée, Amiranashvili and Bandelow 2012, §2). This is why a code that
propagates a real field needs to store only the non-negative frequencies. It is also why
it must take care at zero frequency and at the Nyquist frequency (Chapter 11). Some papers
write $n^2(-\omega)=n^2(\omega)$, which holds only for a lossless medium. The conjugate
form is the general one.

**Causality.** The polarisation at time $t$ can depend only on the field at earlier
times, so the memory function $\chi(\tau)$ vanishes for $\tau<0$. In the Fourier
convention of this primer (§1.4), $\chi(\tau)e^{i\omega\tau}$ decays for $\tau>0$ when
$\operatorname{Im}\omega>0$. So $\chi(\omega)$ is analytic in the upper half of the complex
frequency plane. Combined with $\chi\to0$ as $|\omega|\to\infty$, that analyticity gives
the Kramers–Kronig relations,

$$
\operatorname{Re}\chi(\omega)=\frac1\pi\,\mathcal P\!\!\int_{-\infty}^{\infty}
\frac{\operatorname{Im}\chi(\omega')}{\omega'-\omega}\,\mathrm d\omega',\qquad
\operatorname{Im}\chi(\omega)=-\frac1\pi\,\mathcal P\!\!\int_{-\infty}^{\infty}
\frac{\operatorname{Re}\chi(\omega')}{\omega'-\omega}\,\mathrm d\omega'
$$

(Jackson 1999, §7.10). Dispersion and absorption are therefore not independent. A
refractive index that varies with frequency must come with absorption somewhere in the
spectrum.

Two consequences of causality turn up in practice. First, a model of $n(\omega)$ fitted
over a finite range, such as a Sellmeier formula, is not automatically causal. It has
poles where its resonances are placed, and evaluating it outside its fitted range can
give nonsense. Chapter 7 explains how solprop's material library,
[optmat](https://github.com/LupoLab/optmat), states the range over which each model is
qualified, and what solprop does at the edges. Second, the singularities of
$\epsilon(\omega)$ limit any Taylor expansion of the propagation constant about a carrier.
A 20th-order expansion of $k(\omega)$ about 800 nm in a model of water with a 500 nm resonance fails beyond a radius set by
the nearest resonance, and adding more orders does not help (Kolesik, Townsend and Moloney
2012, Figs. 1–2; Amiranashvili, Vladimirov and Bandelow 2010, Fig. 1). This is one reason
solprop never builds the linear operator from a Taylor series unless you ask it to
(Chapter 3).

**The sign of loss.** In this primer's convention, a plane wave is
$e^{i(kz-\omega t)}$ with $k=n\omega/c$. If $\operatorname{Im}n>0$ the wave decays as it
propagates. solprop always quotes indices this way: $\operatorname{Im}n_{\rm eff}>0$ is
loss, and the field amplitude decays as $e^{-\alpha z/2}$ with
$\alpha=2(\omega/c)\operatorname{Im}n_{\rm eff}$, the power attenuation coefficient
([the linear operator](../guide/numerics/linear-operator.md#from-the-effective-index)).
Waveguide papers do not all agree on whether their quoted loss is per unit field or per
unit power. That factor of 2 is worth checking whenever a loss is taken from the
literature (Chapter 4).

## 1.4 Two Fourier conventions, and how solprop uses them

There are two common sign conventions for the Fourier transform, and solprop uses both:
one at its interface and one inside. This primer, like the rest of the documentation,
writes every equation in the **optics convention**,

$$
\tilde E(\omega)=\int_{-\infty}^{\infty}E(t)\,e^{+i\omega t}\,\mathrm dt,
\qquad
E(t)=\frac1{2\pi}\int_{-\infty}^{\infty}\tilde E(\omega)\,e^{-i\omega t}\,\mathrm d\omega .
$$

In this convention $\partial_t\to-i\omega$, a plane wave is $e^{i(\beta z-\omega t)}$, and
propagation over a distance $z$ multiplies the spectrum by $e^{+i\beta z}$. A delay by
$\tau$ multiplies the spectrum by $e^{+i\omega\tau}$. Positive group-delay dispersion is
a positive parabola $\tfrac12\beta_2(\omega-\omega_0)^2$ in the spectral phase, and a
positive Taylor phase coefficient gives a positively chirped pulse. These are the
conventions of most textbooks and of the propagation literature this primer cites
(Kolesik and Moloney 2004; Couairon et al. 2011; Kinsler 2010). They are also what
solprop's pulse builders accept, what its processing functions return and what its HDF5
files store ([grids and Fourier conventions](../guide/setup/grids.md#fourier-signs-and-the-phase-origin)).

Inside, solprop uses the sign that NumPy, pocketfft and FFTW implement directly,
$\tilde E(\omega)=\int E\,e^{-i\omega t}\mathrm dt$. Its kernels, its linear operator
and its stored state all live in that convention, which is also Luna's. For a real field
the two spectra are complex conjugates, bin by bin. The conversion is a conjugation,
applied in exactly three named places: the pulse builders, the output handlers and the
processing accessors ([conventions](../dev/conventions.md#2-fourier-transform-and-the-sign-of-propagation-adr-017)).
You only meet the internal convention if you build a custom spectral array or read the
guide's numerical chapters, which state their formulas in it. The box below is the
dictionary.

:::{admonition} The two conventions side by side
:class: note

| | This primer and solprop's interface (optics) | Inside solprop (FFT) |
|---|---|---|
| Transform | $\tilde E=\int E\,e^{+i\omega t}\mathrm dt$ | $\tilde E=\int E\,e^{-i\omega t}\mathrm dt$ |
| Time derivative | $\partial_t\to-i\omega$ | $\partial_t\to+i\omega$ |
| Plane wave | $e^{i(\beta z-\omega t)}$ | $e^{i(\omega t-\beta z)}$ |
| Linear propagation | $\partial_z\tilde E=\big[+i\beta-\tfrac\alpha2\big]\tilde E$ | $\partial_z\tilde E=\big[-i\beta-\tfrac\alpha2\big]\tilde E$ |
| Nonlinear prefactor (Chapter 2) | $+\dfrac{i\omega}{2\varepsilon_0c\operatorname{Re}n_{\rm eff}}$ | $-\dfrac{i\omega}{2\varepsilon_0c\operatorname{Re}n_{\rm eff}}$ |
| Envelope | $E=\operatorname{Re}\{A(t)e^{-i\omega_0t}\}$ | $E=\operatorname{Re}\{A_{\rm int}(t)e^{+i\omega_0t}\}$, with $A_{\rm int}=A^*$ |
| Loss | $\operatorname{Im}n_{\rm eff}>0$ | the same (indices are always quoted in the optics sense) |

Two further details belong to the internal side only. The stored spectrum is the **raw
forward DFT** of the sample array, with no $1/N$ and no $\delta t$. Its phase origin is
the first sample, not the window centre. Converting a stored spectrum to an optics
spectrum referenced to $t=0$ therefore takes a conjugation and a factor $(-1)^k$ on the
storage index $k$: $\tilde E_{{\rm optics},k}=(-1)^k\operatorname{conj}(\tilde E_{{\rm
internal},k})$. The functions `fourier.spectrum_to_optics` and `spectrum_to_internal` do
both steps; `to_optics` and `to_internal` only conjugate.
:::

Energies, spectral densities and masks involve only $|\tilde E|^2$, so they are the same
in both conventions. Signs differ only for phases and for any equation with an explicit
$i$. In particular, two guide pages that both write an envelope can look as if they
disagree. One of them (the [envelope plasma](../guide/media/envelope-plasma.md) chapter)
uses the optics envelope $\mathbf a(t)$ with $\mathbf E=\operatorname{Re}\{\mathbf a
e^{-i\omega_0t}\}$, and the other (the [Kerr page](../guide/media/nonlinear.md#one-cubic-four-forms))
the internal one. They are complex conjugates of each other and describe the same field.

## 1.5 The wave equation in the frequency domain

Take the curl of Faraday's law, substitute Ampère's law and $\mathbf
D=\varepsilon_0\epsilon*\mathbf E+\mathbf P$, and transform to the frequency domain with
$\partial_t\to-i\omega$. The result is

$$
\nabla^2\tilde{\mathbf E}-\nabla(\nabla\cdot\tilde{\mathbf E})
+\frac{\omega^2n^2(\omega)}{c^2}\tilde{\mathbf E}
=-\mu_0\omega^2\tilde{\mathbf P}-i\mu_0\omega\tilde{\mathbf J}
$$

(Couairon et al. 2011, Eq. 6; the same equation is Kinsler 2010, Eq. 11, Andreasen and
Kolesik 2012, Eq. 11, and Ferrando et al. 2005, Eq. 1). Everything linear is on the left,
with $n(\omega)$ complete and exact. Everything nonlinear is on the right.

The term $\nabla(\nabla\cdot\tilde{\mathbf E})$ couples the field components. From
$\nabla\cdot\mathbf D=\rho_f$ it equals
$\nabla[(\rho_f-\nabla\cdot\mathbf P)/\varepsilon_0\epsilon]$, so it is of the order of
the nonlinearity times a transverse gradient (Couairon et al. 2011, Eq. 31; Kinsler 2010,
after Eq. 10). It matters for beams focused to a waist of about a wavelength, where it
helps arrest collapse (Kolesik, Moloney and Mlejnek 2002, Fig. 2). It is negligible for
the beams and guided modes of the problems solprop is built for. Dropping it and keeping
one transverse field component (or two, for two polarisations) gives the scalar equation

$$
\big(\partial_z^2+\nabla_\perp^2\big)\tilde E+k^2(\omega)\tilde E
=-\mu_0\omega^2\tilde P-i\mu_0\omega\tilde J,
\qquad k(\omega)=\frac{\omega}{c}n(\omega)
$$

(Couairon et al. 2011, Eq. 9). In a homogeneous medium the transverse dependence is
removed by a transverse Fourier transform, $\nabla_\perp^2\to-k_\perp^2$:

$$
\partial_z^2\tilde E+k_z^2\tilde E=-\mu_0\omega^2\tilde P-i\mu_0\omega\tilde J,
\qquad k_z(\omega,k_\perp)=\sqrt{\frac{\omega^2\epsilon(\omega)}{c^2}-k_x^2-k_y^2}
$$

(Kolesik and Moloney 2004, Eq. 22). In a waveguide the same job is done by expanding
the field in the guide's modes, and $k_z$ is replaced by each mode's propagation
constant $\beta_m(\omega)$ (Chapter 4). Either way, the wave equation is now a set of
second-order ordinary differential equations in $z$, one per frequency and transverse
label, coupled only through $\tilde P$ and $\tilde J$. Chapter 2 turns them into
first-order equations.

Two remarks about what has been kept. The equation is still exact apart from the scalar
reduction. Nothing has been assumed about slowly varying envelopes, paraxial beams or
narrow bandwidths. In free space solprop keeps this exact $k_z$, not its paraxial
expansion (Chapter 6). The vector structure of guided modes, including their longitudinal
field components, is restored on solprop's general modal route, which projects the full
vector polarisation onto vector modes (Chapter 5).

## 1.6 The real field, the analytic signal and the envelope

The electric field is real. Its spectrum therefore carries redundant information:
$\tilde E(-\omega)=\tilde E^*(\omega)$. The positive frequencies alone determine the
field, and it is convenient to build a complex signal from them. The **analytic signal**
is

$$
E_a(t)=\frac1\pi\int_0^\infty\tilde E(\omega)\,e^{-i\omega t}\,\mathrm d\omega,
\qquad E(t)=\operatorname{Re}E_a(t),
$$

which doubles the positive frequencies and discards the negative ones. Its modulus is
the instantaneous amplitude of the oscillation, with the carrier removed. Averaged over
an optical cycle, the intensity of a plane wave in a medium of index $n$ is
$I=\tfrac12\varepsilon_0cn|E_a|^2$. This is why the factor $\tfrac12\varepsilon_0c$ appears so often in solprop: half the vacuum
admittance $\varepsilon_0c=1/Z_0$, it turns $|E_a|^2$ into an intensity. The analytic signal is not just a theoretical device. A spectral code
constructs it whenever it pads the positive frequencies with zeros before an inverse
transform (Couairon et al. 2011, §3.2.2). On a discrete grid, solprop doubles the interior
positive-frequency bins and keeps the zero-frequency and Nyquist bins once each, as
`scipy.signal.hilbert` does, so that $\operatorname{Re}E_a$ reproduces $E$ exactly
([zero frequency and Nyquist](../guide/setup/grids.md#zero-frequency-and-nyquist)).

An **envelope** removes a carrier frequency $\omega_0$ chosen by you:

$$
E_a(t)=A(t)\,e^{-i\omega_0t},\qquad E(t)=\operatorname{Re}\{A(t)e^{-i\omega_0t}\}.
$$

$A(t)$ is complex. Its spectrum is the positive-frequency part of $2\tilde E$, shifted
down by $\omega_0$. If the pulse is narrowband, $A$ varies slowly and needs far fewer
samples than $E$. That saving is the whole reason to use an envelope. It costs nothing in
generality as long as nothing is thrown away. Chapter 2 shows that the envelope form of the
propagation equation is the field form written in shifted frequency variables (Couairon et
al. 2011, Eqs. 38–40). An envelope *model* becomes an approximation only when its
nonlinear polarisation drops frequency components, such as the third harmonic, or when its
dispersion or coupling is truncated.

```{figure} figures/representations.svg
:width: 100%
:alt: Left, a 6 fs pulse at 800 nm as an oscillating real field inside the modulus of its analytic signal. Right, the real field's two-sided spectrum at plus and minus the carrier, the analytic signal's doubled positive-frequency spectrum, and the envelope's spectrum centred at zero detuning.

One pulse in the three descriptions. The real field's spectrum has two halves, related by
$\tilde E(-\omega)=\tilde E^*(\omega)$. The analytic signal keeps the positive half, doubled.
The envelope is the same spectrum shifted down by the carrier. Computed with NumPy in the
optics convention by `docs/primer/figures/primer_figures.py`.
```

The factor of 2 in the analytic signal is where most normalisation errors in pulse
propagation come from, so it is worth seeing once where it goes. Because the field and the
polarisation are both real and are both represented by their analytic signals or
envelopes in the same way, the factor cancels from the linear propagation equation. It
reappears in three places:

1. **Energy.** On a field grid, the energy sums the interior positive-frequency bins with
   weight 4 and the end bins with weight 1. On an envelope grid every bin has weight 1
   ([energy and spectral density](../guide/setup/grids.md#energy-and-spectral-density)).
2. **Intensity.** The cycle average introduces the $\tfrac12$ in
   $\tfrac12\varepsilon_0cn|E_a|^2$.
3. **Nonlinear coefficients.** Cubing $E=\tfrac12(E_a+E_a^*)$ gives
   $\tfrac18(E_a^3+3|E_a|^2E_a+{\rm c.c.})$. The analytic signal of the cubic polarisation
   at the fundamental is therefore $\tfrac34|E_a|^2E_a$, and that of the third harmonic is
   $\tfrac14E_a^3$. Chapter 8 makes this precise; the same $\tfrac34$ appears in Couairon
   et al. (2011, Eq. 100) and, as $\tfrac38$ of the positive-frequency part, in Brée et
   al. (2012).

:::{admonition} Field grid or envelope grid?
:class: note

solprop offers both: a **field grid** that carries the real $E(t)$ and stores its
one-sided spectrum, and an **envelope grid** that carries $A(t)$ and stores its full
spectrum ([choose a field or an envelope](../guide/setup/grids.md#choose-a-field-or-an-envelope)).
Use a field grid when the carrier matters: when the pulse becomes very broadband, when the
third harmonic or other harmonics are generated, or when the response depends on the
instantaneous field rather than its cycle average, as tunnelling ionisation does. An
envelope grid is cheaper for narrowband problems, and solprop's
[GNLSE route](../guide/waveguides/gnlse.md) uses one. Chapter 11 explains how each grid
is sized.
:::

## 1.7 Where the transverse structure goes

The equation of §1.5 holds for each transverse Fourier component in a homogeneous medium.
In a waveguide the role of the transverse wavenumber is taken by the mode index. The
mathematics is the same: expand the field in eigenfunctions of the transverse linear
problem and propagate the expansion coefficients. Only the eigenfunctions change, from
plane waves $e^{i\mathbf k_\perp\cdot\mathbf r_\perp}$ to guided-mode profiles
$\mathbf e_m(\mathbf r_\perp,\omega)$ with propagation constants $\beta_m(\omega)$.
Kolesik and Moloney (2004, Eq. 18) write a single coupled-mode equation that covers bulk
media, fibres and capillaries. Chapter 2 derives the propagation equation from it, and
Chapters 4–6 return to it for each geometry.

In practice this choice is the first decision of any solprop calculation: one mode with
an effective area, a set of guided modes, or a transverse grid in free space. The
[spatial description](../introduction/spatial-description.md) page shows these choices
side by side as pictures. [Spatial decompositions](../guide/setup/spatial-decompositions.md)
gives the boundary conditions, normalisations and limits of each.

:::{admonition} What to remember
:class: tip

- solprop is SI throughout. The nonlinear polarisation $\mathbf P$ and the free current
  $\mathbf J$ are the only nonlinear quantities. All linear dispersion and loss goes into
  $\epsilon(\omega)=n^2(\omega)$.
- Reality requires $\chi(-\omega)=\chi^*(\omega)$. Causality makes $\chi$ analytic in the
  upper half plane (optics convention), hence Kramers–Kronig. Fitted index laws should be relied on only
  within their qualified range; a finite-band fit does not by itself establish a causal extension.
- $\operatorname{Im}n>0$ is loss. $\alpha=2(\omega/c)\operatorname{Im}n$ is the power
  attenuation coefficient, and the field decays as $e^{-\alpha z/2}$.
- The primer and solprop's interface use $\tilde E=\int E\,e^{+i\omega t}\mathrm dt$.
  Inside, solprop uses the FFT sign, and the two are related by conjugation (with a
  $(-1)^k$ phase-origin factor for stored spectra).
- The frequency-domain wave equation is exact apart from the scalar reduction: its
  left-hand side carries the full $n(\omega)$, its right-hand side
  $-\mu_0\omega^2\tilde P-i\mu_0\omega\tilde J$.
- $E=\operatorname{Re}E_a=\operatorname{Re}\{Ae^{-i\omega_0t}\}$. The factor 2 of the
  analytic signal cancels from the propagation equation and reappears in energies,
  intensities and nonlinear coefficients ($\tfrac34$ and $\tfrac14$).
:::

The precise statements are in the [conventions](../dev/conventions.md), in [grids and
Fourier conventions](../guide/setup/grids.md) and in [the linear
operator](../guide/numerics/linear-operator.md). [Chapter 2](uppe.md) turns the wave
equation into the first-order, forward-only equation that solprop actually solves.
