(primer-kerr)=
# 8. The Kerr effect

The instantaneous electronic third-order response, the Kerr effect, is the nonlinearity
behind almost everything in ultrafast nonlinear optics: self-phase modulation, solitons,
self-focusing, four-wave mixing and third-harmonic generation. Its physics is simple to
state. The bound electrons of an atom respond to the field anharmonically, and the leading
correction to the induced dipole is cubic in the field. What needs care is the
bookkeeping. You have to relate the cube of a real field to its envelope, $\chi^{(3)}$ to
$n_2$ and $\gamma$, a scalar to a vector, and a single coefficient frozen at the carrier to
a susceptibility that depends on all four frequencies of each mixing process.

This chapter does that bookkeeping once, carefully. By the end you should know the factors
$\tfrac34$ and $\tfrac14$, how $n_2$ follows from $\chi^{(3)}$, which form of the cubic solprop
evaluates on each grid, and what happens to a polarisation vector. You should also know when
the carrier-frequency coefficient is good enough and what solprop's dispersive alternatives
compute.

## 8.1 The cubic polarisation

For an isotropic medium and a linearly polarised field, the instantaneous third-order polarisation
is

$$
P(t)=\varepsilon_0\chi^{(3)}E(t)^3 .
$$

For a gas, $\chi^{(3)}=N\gamma^{(3)}$ with a per-particle coefficient $\gamma^{(3)}$ in m⁵/V² (Chapter
7). solprop's Kerr response takes that per-particle coefficient together with the number density
profile, so that a gradient changes the nonlinearity in step with the index
([the Kerr response](../guide/media/nonlinear.md#the-kerr-response)).

Write the real field as $E=\tfrac12(E_a+E_a^*)$ with $E_a=Ae^{-i\omega_0t}$ (§1.6). Expanding the
cube gives

$$
E^3=\tfrac34\operatorname{Re}\big(|A|^2A\,e^{-i\omega_0t}\big)+\tfrac14\operatorname{Re}\big(A^3e^{-3i\omega_0t}\big).
$$

The first term oscillates at the carrier and drives self-phase modulation. The second oscillates at
three times the carrier and drives third-harmonic generation. For a monochromatic field the harmonic term has a quarter of the cubic's amplitude (a third of the fundamental term's), so it carries a tenth of the polarisation's spectral power. Whether that matters depends on phase matching. In a gas-filled
fibre the third harmonic is usually far from phase-matched and stays weak. The same factors appear
throughout the literature: Couairon et al. (2011, Eqs. 98–100) give the expansion and
$\mathcal P=\tfrac34\varepsilon_0\chi^{(3)}|\mathcal E|^2\mathcal E$ for the fundamental.

solprop evaluates one of four forms of this cubic, depending on the grid and on whether you ask for
the harmonic ([one cubic, four forms](../guide/media/nonlinear.md#one-cubic-four-forms)). In the
optics sign of this primer:

| grid | harmonic | polarisation |
|---|---|---|
| field | kept | $\varepsilon_0\chi^{(3)}E^3$, the cube of the real field |
| field | dropped | $\tfrac34\varepsilon_0\chi^{(3)}\lvert E_a\rvert^2E$ |
| envelope | dropped | $\tfrac34\varepsilon_0\chi^{(3)}\lvert A\rvert^2A$ |
| envelope | kept | $\tfrac14\varepsilon_0\chi^{(3)}\big(3\lvert A\rvert^2+e^{-2i\omega_0t}A^2\big)A$ |

These are the forms solprop's constant Kerr response implements. The last is the envelope
equation of Genty et al. (2007, Eq. 4). Its exponent has the opposite sign in the internal
convention, which is why the guide writes $e^{+2i\omega_0t}$. It is not the exact analytic signal of a broadband
cubic: that also contains the positive-frequency part of $3|A|^2A^*e^{+2i\omega_0t}$ (Bache
2016, Eq. 18), which drops out only when the spectrum is narrow enough that this term falls at
negative absolute frequency. For a spectrum wider than an octave it does not, and no amount of
oversampling restores it. The dispersive responses keep the full harmonic sector, with the
projection onto positive frequency
([the harmonic sector](../guide/media/dispersive-kerr.md#the-harmonic-sector-thg)). On a field grid
the first form needs no special treatment: the cube of the real field contains the harmonic
automatically, provided that the grid samples three times the highest frequency (Chapter 11). On an
envelope grid the harmonic term needs a grid wide enough to hold a spectrum at $3\omega_0$, so it needs spectral room on the grid. If the fundamental and harmonic spectra overlap, two separate envelopes become
inconsistent (Couairon et al. 2011, §2.4.2). A single field grid is then the right representation.

## 8.2 From $\chi^{(3)}$ to $n_2$ and $\gamma$

The nonlinear index $n_2$ is defined through $n=n_0+n_2I$, with $I=\tfrac12\varepsilon_0cn_0|E_a|^2$.
For the fundamental term, the change of susceptibility is $\tfrac34\chi^{(3)}|E_a|^2$, and
$\Delta(n^2)=2n_0\Delta n$, so

$$
n_2=\frac{3\chi^{(3)}}{4\varepsilon_0cn_0^2}
$$

(Couairon and Mysyrowicz 2007, Eq. 29; Couairon et al. 2011, after Eq. 100). solprop reads $n_0$ at the
same carrier. On the GNLSE route it takes $n_0=1$ in this relation, deliberately and consistently
with the prefactor (Chapter 3). On the general modal route a declared $n_2$ is converted with the
region's own index. For silica that conversion changes by 2% between 800 nm and 2 µm, and solprop
refuses an undeclared conversion that would vary by more than 1% across the band.

optmat stores a gas's second hyperpolarisability in the Taylor convention used by Shelton and Rice,
$p=\alpha E+\tfrac12\beta E^2+\tfrac16\gamma_TE^3$, so that

$$
\gamma^{(3)}=\frac{\gamma_T}{6\varepsilon_0},\qquad \chi^{(3)}=\frac{N\gamma_T}{6\varepsilon_0}
$$

([the electronic Kerr coefficient at the
carrier](../guide/media/material-selection.md#the-electronic-kerr-coefficient-at-the-carrier)). For
argon at 800 nm, $\gamma_T\approx1258$ atomic units gives $\gamma^{(3)}=1.48\times10^{-51}$ m⁵/V².
At 1 bar and 20 °C that gives $n_2\approx1.0\times10^{-19}$ cm²/W, to be compared with roughly
$2.5\times10^{-16}$ cm²/W for fused silica. The gas is three orders of magnitude weaker per unit
intensity. A gas-filled fibre compensates by using intensities near $10^{14}$ W/cm², which a solid
cannot survive.

## 8.3 Vectors

With two transverse field components, the isotropic cubic depends on how they are combined. For
the real field the polarisation is $\varepsilon_0\chi^{(3)}(\mathbf E\cdot\mathbf E)\mathbf E$. In
terms of analytic signals or envelopes $a_q$,

$$
P_p=\tfrac12\Big(\sum_q|a_q|^2\Big)a_p+\tfrac14\Big(\sum_qa_q^2\Big)a_p^*
$$

(in units of $\varepsilon_0\chi^{(3)}$), which for $p=x$ is
$\tfrac34\big[(|a_x|^2+\tfrac23|a_y|^2)a_x+\tfrac13a_y^2a_x^*\big]$
([the vector form](../guide/media/nonlinear.md#the-vector-form-falls-out-of-the-same-algebra)).
Two consequences are worth remembering. Cross-phase modulation between orthogonal linear
polarisations is $\tfrac23$ of self-phase modulation. And the last term, which couples $a_x^*$ to $a_y^2$,
makes the two linear polarisations exchange energy coherently. In a circular basis that term vanishes,
and self-phase modulation of each circular component is $\tfrac23$ of the linear-polarisation value
(Poletti and Horak 2008, Eq. 14). On the general modal route the same algebra is written as the
isotropic tensor $T_{ijkl}=(\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})/s$,
with $s=3$ for the real field and $s=4$ for the analytic signal (Chapter 5).

## 8.4 A coefficient frozen at the carrier

A gas's hyperpolarisability is not a constant. It rises towards the ultraviolet as the frequencies
approach the first electronic resonance. Argon's Kerr coefficient at 267 nm is twice its value at
800 nm. Xenon's at 400 nm is nearly twice its value at 2 µm. For a mixing process with four different
frequencies the coefficient depends on all four. The default policy, `carrier-kerr`, ignores this: it
evaluates the coefficient once at the run's carrier wavelength and uses that one number for every
process, including third-harmonic generation. The manifest records the ratio of the harmonic's own
coefficient to the one used, 1.18 for argon at 800 nm, so the size of the approximation is visible.

For self-phase modulation near the carrier this is exact. It becomes an approximation when the spectrum
spans an octave or more, or when widely separated colours mix: a pump and a deep-ultraviolet dispersive
wave, or a pump, signal and idler of four-wave mixing.

## 8.5 A frequency-dependent electronic response

The general third-order polarisation of a real scalar field is a double integral over the input
frequencies,

$$
\tilde P^{(3)}(\omega_o)=\frac{\varepsilon_0}{(2\pi)^2}\iint\chi^{(3)}(-\omega_o;\omega_1,\omega_2,\omega_3)\,
\tilde E(\omega_1)\tilde E(\omega_2)\tilde E(\omega_3)\,\mathrm d\omega_1\,\mathrm d\omega_2,\qquad
\omega_3=\omega_o-\omega_1-\omega_2
$$

([one susceptibility, all mixing processes](../guide/media/dispersive-kerr.md#one-susceptibility-all-mixing-processes)).
Self- and cross-phase modulation, four-wave mixing and third-harmonic generation are not separate terms.
They are slices of this one integral. With a constant $\chi^{(3)}$ it collapses to
$\varepsilon_0\chi^{(3)}E(t)^3$, which is why the constant cubic needs no bookkeeping of processes. A
dispersive model must keep that property: one polarisation, with every quartet weighted by its own four
frequencies.

optmat fits each gas's hyperpolarisability over a quartet of signed wavenumbers $\nu_j$, with the output
counted negative so that the four sum to zero, using the dispersion expansion of Bishop and De Kee (1996)
written relative to the static value in Shelton's form:

$$
\gamma_T=\gamma_0\big[1+AS_2+BS_2^2+B'S_4\big],\qquad S_2=\sum_j\nu_j^2,\quad S_4=\sum_j\nu_j^4 .
$$

Its Kerr diagonal, $(\nu,\nu,-\nu)\to\nu$, is $\gamma_K(\nu)/\gamma_0=1+4A\nu^2+(16B+4B')\nu^4$.

The tempting shortcut is to multiply the constant cubic's spectrum by the Kerr coefficient of each
*output* frequency. That is right for monochromatic self-phase modulation and wrong for everything
else. In argon, the quartet $400+400-800\to266.7$ nm would receive the 266.7 nm Kerr value, 50% above
the polynomial's value for that quartet. Worse, the coefficient would change when a different member of
a lossless quartet is called the output. The response would then do net work on the field, which a
transparent medium cannot ([why a coefficient of the output frequency is
wrong](../guide/media/dispersive-kerr.md#why-a-coefficient-of-the-output-frequency-is-wrong)).
solprop does not offer it.

solprop instead offers two dispersive policies, both opt-in
([the two evaluators](../guide/media/dispersive-kerr.md#the-two-evaluators)):

- **`gas-polynomial`** evaluates optmat's fit exactly at every quartet. The polynomial is expanded into
  products of spectrally filtered fields, $E$, $DE$ and $D^2E$ with $D$ multiplying by
  $(\omega/2\pi c)^2$. That costs three inverse and three forward transforms where the constant cubic
  needs one of each.
- **`symmetric-kerr`** applies one filter $g(\nu)=[\gamma_K(\nu)/\gamma_0]^{1/4}$ to every leg: filter the
  field, cube it, filter the polarisation. It is exact on the Kerr diagonal and agrees with the
  polynomial to second order for every quartet. Its median error over random quartets between 400 nm
  and 2 µm is 0.05% for argon, and its largest about 1%. For a scalar field it costs almost nothing
  beyond the constant cubic.

Both evaluators are lossless: with no absorber active, their kernels are real and symmetric in the four
legs, and the continuum response does no net work on the field. Where a leg of a quartet goes beyond the
gas's data, the response is bounded, never refused. Each leg is clamped at a frequency derived from the
fit's own validity, and the run records how much of the field lies in each region
([beyond the gas's data](../guide/media/dispersive-kerr.md#beyond-the-gass-data-bounded-never-refused)).

Does it matter? In solprop's two demonstration cases, the carrier policy underestimates the 100–150 nm
energy of a helium dispersive wave by 23%, and shifts the best pressure for deep-ultraviolet four-wave
mixing in argon by 5% ([what the demonstration cases
measured](../guide/media/dispersive-kerr.md#what-the-demonstration-cases-measured)). In the argon case
every leg is inside the data, so the difference is the carrier approximation's error. In the helium case
the quartets reach past the data, so the difference measures the sensitivity of the result to
coefficients that no off-resonant model can fix. That is a useful thing to know about an experiment's
prediction.

## 8.6 Beyond the cubic

Two extensions of the bound-electron response come up in the filamentation literature. solprop models
neither. The first is a higher-order Kerr effect, in which $\chi^{(5)}$ and higher terms would saturate
and invert the nonlinear index at high intensity. Loriot et al. (2009) reported such a negative
contribution in nitrogen, oxygen and argon above a few tens of TW/cm², and Béjot et al. (2010) argued
that it could arrest filamentation without ionisation. Absolute measurements by Wahlstrand, Cheng and
Milchberg (2012) then found the response of all five noble gases quadratic in the field "up to the
ionization threshold of each gas", consistent with a Kramers–Kronig analysis of the ionisation and with
time-dependent Schrödinger calculations. Kolesik and Moloney (2014, §3.2) review the models of the bound
and freed electrons that this debate produced. The second is a model of the
whole bound-electron response, including its dispersion and its non-instantaneous part, fitted to
time-dependent Schrödinger calculations (Schuh et al. 2014). Within solprop, the place where a full
atomic response replaces the Kerr and plasma terms together is the optional helium TDSE response
(Chapter 10).

:::{admonition} What to remember
:class: tip

- $E^3=\tfrac34\operatorname{Re}(|A|^2Ae^{-i\omega_0t})+\tfrac14\operatorname{Re}(A^3e^{-3i\omega_0t})$. The harmonic
  term has a quarter of the cubic's amplitude, a tenth of its spectral power.
- solprop evaluates the real cube on a field grid, or one of three envelope forms. The harmonic needs
  sampling to $3\omega_{\max}$.
- $n_2=3\chi^{(3)}/(4\varepsilon_0cn_0^2)$ and $\chi^{(3)}=N\gamma_T/(6\varepsilon_0)$ in optmat's Taylor
  convention. Argon at 1 bar: $n_2\approx10^{-19}$ cm²/W.
- Orthogonal linear polarisations cross-phase-modulate at $\tfrac23$. A circular component self-phase-modulates
  at $\tfrac23$ of the linear value.
- The default freezes the coefficient at the carrier. A coefficient of the output frequency is wrong, and is
  not offered. The dispersive policies evaluate the fit at every quartet, exactly or symmetrically, and
  bound the legs beyond the data.
:::

The precise statements are in [nonlinear responses](../guide/media/nonlinear.md), [a frequency-dependent
electronic response](../guide/media/dispersive-kerr.md) and [selecting and inspecting material
data](../guide/media/material-selection.md). [Chapter 9](raman.md) adds the response that is not
instantaneous.
