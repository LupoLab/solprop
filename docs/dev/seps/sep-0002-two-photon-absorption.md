# SEP 2 — Two-photon absorption

| | |
|---|---|
| **SEP** | 2 |
| **Title** | Two-photon absorption |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-26 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add explicitly declared, scalar two-photon absorption (TPA) to waveguide and spatial
propagation, first as a constant-coefficient, narrowband loss law. Derive its
normalisation from absorbed power and implement it as a dissipative current, using
the existing transformations and UPPE projection. Qualify envelope and field grids,
then NumPy, Numba CPU and Kokkos CPU, against analytic attenuation, physical energy
balance and one another. Disabled problems retain their existing execution path and
workspaces. A wavelength-dependent *degenerate* absorption curve does not by itself
specify broadband TPA: this proposal distinguishes the first model from the
additional material response needed for broadband DUV pulses, mixed-frequency
absorption, crystal tensors and semiconductor carrier feedback. Those limitations
are admission rules and documented modelling choices, not silent approximations.

### 2. Motivation and scope

Two immediate uses are nonlinear attenuation in silicon-core fibres and propagation
of DUV pulses through fused silica. In both, linear loss and real Kerr response are
insufficient. However, three different tasks must be distinguished:

1. **Direct TPA loss:** two photons are removed, with a rate quadratic in intensity.
   Its monochromatic coefficient is measurable without choosing a carrier model.
2. **Broadband TPA:** photons at different frequencies interact. A steep absorption
   edge makes a constant-coefficient approximation inadequate, even when a field
   grid resolves the carrier perfectly.
3. **The material after absorption:** electron–hole populations, their absorption
   and refraction, trapping, recombination, heating and cumulative defects.

The recommended first scope is task 1 on scalar envelope and field grids:

| Route | Initial supported interpretation |
|---|---|
| GNLSE | Measured bulk coefficient and explicitly supplied effective area; true-power amplitude |
| Mode-averaged UPPE | Homogeneous responding medium and a scalar mode with the existing normalisation |
| Ordinary multimode UPPE | Scalar local fields, coherent synthesis before evaluating the loss, a common responding medium |
| Radial, Cartesian, line and planar spatial routes | Scalar local fields in the homogeneous media already admitted by each route |

The geometry admission rules that already apply still apply. In particular, a line
or planar calculation retains its existing transverse measure and output units.
The first scope excludes general full-vector modal guides with distinct material
regions; treating their three local components as one scalar would be incorrect.
It also excludes vector/isotropic and crystal-tensor TPA, arbitrary wavelength
arrays or callbacks as a broadband response, saturation, higher-order absorption,
solid ionisation, generated-carrier feedback, pulse-train memory, damage and
thermal evolution. TPA may coexist with independently justified Kerr and Raman
responses; it consumes no Raman fraction.

This is a useful but limited first feature. It is **not** a complete silicon
propagation model or a quantitative model of a DUV continuum crossing an absorption
edge. If either is the first required application, Q1 must enlarge the accepted
scope before implementation begins. Section 4 states what additional physics that
would require. No stock silicon or silica coefficient is selected implicitly.

### 3. Context

The [nonlinear model](../../guide/media/nonlinear.md),
[power and intensity](../../guide/results/power-and-intensity.md),
[general modal model](../../guide/waveguides/general-modal.md),
[architecture](../architecture.md) and [kernel contract](../kernel-contract.md)
provide the starting point. The proposal addresses the TPA item under
*Spectral-domain responses* in the [roadmap](../roadmap.md), and is related to, but
does not implement, its separate *Ionisation in solids* item.

The relevant current code is:

- `nonlinear/responses.py`: real Kerr coefficients and the response traits;
  `nonlinear/normalisation.py`: physical-field synthesis and the UPPE prefactor;
  `nonlinear/gnlse.py`: the GNLSE calibration with `prefactor_index=1.0`.
- `translate/responses.py`: compiled response leaves and their validation;
  `translate/envelope_plasma.py`: an existing typed current/source assembly;
  `translate/stage.py`: stage scalars. `SCHEMA_VERSION` is 15 at drafting.
- `backends/numpy_ref/passes.py` and `envelope_plasma.py`: response evaluation,
  projection and explicit current transforms. `backends/bodies.py` and
  `numba_cpu/kernels.py`: the scalar bodies and specialised response traversal.
- `modes/material.py`, `translate/general.py` and each backend's
  `general_modal.py`: a separate regional response path. `MaterialResponse` currently
  admits `MaterialKerr` and `SilicaTensorResponse`; those types do not already
  provide arbitrary complex susceptibility or TPA.
- In `solprop-kokkos`, `src/solprop_kokkos/backend.py`,
  `include/solprop_kokkos/core.hpp`, `src/core.cpp`, `src/general.cpp` and
  `include/solprop_kokkos/envelope_plasma.hpp`: admission, native data and resident
  execution. A source cannot be added solely to the Python oracle.

The seam's `spectral_responses` is currently a tuple specifically of
`EnvelopePlasmaAssembly`, validated to agree with `envelope_plasma`. It is not an
open plugin list. NumPy and Numba admit the existing envelope assembly; Kokkos
explicitly rejects spectral entries other than that assembly. Adding a second
source requires typed contract and admission changes, including a check of *every*
entry when plasma and TPA coexist. Merely checking that plasma is present would
not detect an unimplemented extra term.

Some older prose still says that spectral responses are unused or universally
refused. The executable envelope-plasma path supersedes that description; phase 1
must reconcile the affected documentation. ADR-017 fixes the Fourier signs,
ADR-026 the GNLSE calibration and ADR-030 the modal power normalisation. None of
those conventions changes here.

### 4. Theory

#### 4.1 Definition and units

Use $\beta_{\mathrm{TPA}}$, never an unqualified $\beta_2$, which would collide with
group-velocity dispersion. Define the coefficient by the monochromatic,
homogeneous, plane-wave intensity equation

$$
\frac{dI}{dz}=-\alpha I-\beta_{\mathrm{TPA}}I^2,
\qquad I\ [\mathrm{W\,m^{-2}}],\quad
\beta_{\mathrm{TPA}}\ [\mathrm{m\,W^{-1}}].
$$

This is the measured convention of [2–5]. It is an **intensity** loss coefficient;
the corresponding field attenuation contains a factor of one half. Unit
conversions must be explicit:

$$
1\ \mathrm{cm/GW}=10^{-11}\ \mathrm{m/W},\qquad
1\ \mathrm{cm/W}=10^{-2}\ \mathrm{m/W}.
$$

Neither a molecular cross section in GM nor a modal coefficient in
$\mathrm{W^{-1}m^{-1}}$ is this quantity. A molecular coefficient would additionally
need number density, photon energy and a stated event-counting convention. No such
conversion is inferred by this SEP.

#### 4.2 Polarisation, sign and index factors

Write the real field as
$E_{\mathrm r}=\operatorname{Re}[\mathcal E(t)e^{-i\omega_r t}]$, using the optics
convention of [conventions](../conventions.md). For a transparent, nonmagnetic
medium at the reference frequency,

$$
I=\frac{n_r\epsilon_0c}{2}|\mathcal E|^2,
\qquad n_r=\operatorname{Re}n_{\mathrm{material}}(\omega_r)>0.
$$

The resonant scalar cubic polarisation envelope is
$\mathcal P^{(3)}=(3/4)\epsilon_0(\chi_R+i\chi_I)|\mathcal E|^2\mathcal E$.
Insertion into the optics UPPE factor $+i\omega_r/(2\epsilon_0cn_r)$ gives

$$
\partial_z\mathcal E\big|_{\mathrm{TPA}}
=-\frac{3\omega_r\chi_I}{8cn_r}|\mathcal E|^2\mathcal E,
\qquad
\boxed{\beta_{\mathrm{TPA}}
=\frac{3\omega_r\chi_I}{2\epsilon_0c^2n_r^2}}.
$$

This agrees with [1, Eq. (8)]. In SI the denominator contains $\epsilon_0$.
Inside solprop, conjugation changes $+i\chi_I$ to $-i\chi_I$, and the propagation
factor is $-i\omega/(2\epsilon_0c\operatorname{Re}n_{\mathrm{eff}})$.
Their product must give **negative real** attenuation. At the reference frequency
the internal polarisation coefficient is therefore

$$
\mathcal P_{\mathrm{TPA,int}}
=-i\frac{\epsilon_0^2c^2n_r^2\beta_{\mathrm{TPA}}}{2\omega_r}
  |\mathcal E_{\mathrm{int}}|^2\mathcal E_{\mathrm{int}}.
$$

The material index in this conversion and the modal index in the propagation
prefactor serve different purposes. The former must not be replaced by a fitted
dispersion's $c\beta/\omega$, and neither factor is explained away by the FFT
normalisation. Highly absorbing media, for which this transparent-medium
perturbation is inadequate, are outside the proposed model.

#### 4.3 Recommended finite-band model: a dissipative current

The measured scalar loss law can be represented by a current envelope

$$
\mathcal J_{\mathrm{TPA}}=C_J|\mathcal E|^2\mathcal E,
\qquad
C_J=\frac{\beta_{\mathrm{TPA}}n_r^2\epsilon_0^2c^2}{2}.
$$

Here $\mathcal J$ has units $\mathrm{A\,m^{-2}}$ and $C_J$ has units
$\mathrm{A\,m\,V^{-3}}$. Its cycle-averaged work density is

$$
q=\frac12\operatorname{Re}(\mathcal E^*\mathcal J)
=\frac{C_J}{2}|\mathcal E|^4
=\beta_{\mathrm{TPA}}I^2\ge0.
$$

This derives the coefficient from energy balance without modal conventions. The
finite-band extension **holds $C_J$ fixed**. Its effective imaginary susceptibility
is $4C_J/(3\epsilon_0\omega)$; it does not simultaneously hold $\chi_I$ fixed.
The two models agree at $\omega_r$, but are different approximations away from it.
Calling both simply “constant TPA” would hide that distinction.

In the internal FFT convention $J=\partial_tP$ gives

$$
\widetilde P_{\mathrm{TPA}}(\omega)
=-\frac{i}{\omega}\widetilde J_{\mathrm{TPA}}(\omega),
\qquad
\partial_z\widetilde E\big|_{\mathrm{TPA}}
=-\frac{\widetilde J_{\mathrm{TPA}}}
        {2\epsilon_0c\operatorname{Re}n_{\mathrm{eff}}(\omega)}
$$

for the ordinary collinear prefactor. For the spatial UPPE, the corresponding
factor is $-\mu_0\omega/(2k_z)$; retain the existing transverse dependence of
$k_z$. The $-i/\omega$ multiplier is a representation of current as polarisation,
not a degenerate material absorption curve. It is applied only on the positive
active band and is zero elsewhere, including DC. A shock-disabled calculation
must freeze the matching current denominator and UPPE numerator together, as the
existing envelope loss source does, so that their cancellation remains exact.
It must not accidentally acquire a factor $\omega_r/\omega$.

On a field grid, form the analytic field with the established endpoint and
positive-bin rules, then the real current
$J_{\mathrm r}=C_J|E_a|^2 E_{\mathrm r}$, and transform it. Its instantaneous work
$E_{\mathrm r}J_{\mathrm r}=C_J|E_a|^2 E_{\mathrm r}^2$ is nonnegative.
The envelope form is its rotating-wave limit. There is no new $E_{\mathrm r}^3$
loss term: that would add carrier-harmonic response not specified by the measured
degenerate coefficient. Positive-frequency projection, multiplication, padding
and cropping need explicit tests; a Hilbert transform alone does not make an
arbitrary broadband product a material model. Field/envelope equivalence is
claimed in the resolved narrowband limit, not for pulses of arbitrary bandwidth.

This is a phenomenological, finite-band, cycle-averaged electronic-loss model.
The analytic-field construction is not a microscopic causal response valid at
all frequencies. It does not predict the accompanying Kerr dispersion. The
reference coefficient and $n_r$ must vary negligibly over the model's declared
band for its intensity interpretation to hold. No universal bandwidth tolerance
is invented: a material-specific allowance is part of Q2 and the model's record.

#### 4.4 Modal power and overlap

For the ordinary scalar modal route, let $w=\operatorname{Re}n_{\mathrm{eff}}$,
$b=\sqrt w\,a$ and $P=|b|^2$, as in ADR-030. In the monochromatic mode-averaged
limit, $E=a/\sqrt{\epsilon_0cA_{\mathrm{eff}}/2}$. Combining this synthesis with
the current and its projection gives

$$
\partial_z b\big|_{\mathrm{TPA}}
=-\frac{\eta}{2}|b|^2b,
\qquad
\eta=\frac{\beta_{\mathrm{TPA}}}{A_{\mathrm{eff}}}
       \frac{n_r^2}{w^2}.
$$

Thus $dP/dz=-\eta P^2$. The familiar $\eta=\beta_{\mathrm{TPA}}/A_{\mathrm{eff}}$
is recovered when the scalar mode's index equals the material index, or for an
explicitly calibrated effective scalar GNLSE. It is not a universal formula for
strongly confined vector modes.

The GNLSE deliberately uses index one, and its internal amplitude is already
true $\sqrt{\mathrm W}$. Translate a bulk coefficient with an explicit area to
the calibrated synthetic current coefficient
$C_{J,\mathrm{GNLSE}}=\beta_{\mathrm{TPA}}\epsilon_0^2c^2/2$, rather than carrying
the physical material index into that synthetic normalisation. A directly
specified Kerr $\gamma$ does not determine a physical TPA area: $\gamma$ with an
explicit area is sufficient, but $\gamma$ alone is not. A future measured *modal*
TPA coefficient must have a separately named interface and units.

In a spatially resolved multimode calculation, synthesise the coherent local
field first and project the current afterwards. Applying independent loss to
each mode would omit cross absorption and interference. The total power lost is
the quadrature of $q$; an individual modal power may increase through redistribution.
For a future general modal extension, use the E/H flux metric and material-region
assignment. In a mode normalised to one watt, a scalar local law would give
$\eta=\int_{\mathrm{material}}\beta_{\mathrm{TPA}}
(n_r\epsilon_0c/2)^2|e_b|^4\,dA$ where that local scalar intensity approximation
is justified. A crystal tensor instead needs its full component contraction.
Neither a vacuum effective area nor the same response in core and cladding
substitutes for this calculation.

For homogeneous spatial propagation, a local field balance is the primary check.
The current spatial outputs use vacuum-admittance energy, not a general dispersive
medium flux; multiplying that statistic by one constant is valid only in the
nondispersive collinear limit. This SEP does not silently redefine spatial energy.

#### 4.5 Analytic attenuation

With constant nonnegative $\alpha$ and $\beta_{\mathrm{TPA}}$, no diffraction or
dispersion and entrance intensity $I_0(t)$,

$$
I(z,t)=\frac{I_0(t)e^{-\alpha z}}
 {1+\beta_{\mathrm{TPA}}I_0(t)L_{\mathrm{eff}}},\qquad
L_{\mathrm{eff}}=\frac{1-e^{-\alpha z}}{\alpha}.
$$

At $\alpha=0$, $L_{\mathrm{eff}}=z$ and
$I=I_0/(1+\beta_{\mathrm{TPA}}I_0z)$. Evaluate the finite-$\alpha$ expression
with `expm1` in an independent reference. Replace $I,\beta$ by $P,\eta$ for a
single fixed mode. With zero linear loss and a Kerr coefficient $\gamma$ in
the optics convention,

$$
b(z,t)=\frac{b_0(t)}{\sqrt{1+\eta|b_0(t)|^2z}}
\exp\!\left[\frac{i\gamma}{\eta}
\ln(1+\eta|b_0(t)|^2z)\right].
$$

The limit $\eta\to0$ recovers ordinary SPM. This verifies both the absolute
attenuation and its interaction with nonlinear phase; “energy decreased” alone
cannot detect a wrong coefficient by orders of magnitude.

Experiments usually measure transmitted **pulse energy**, not the peak intensity
ratio. For a collimated input intensity
$I_{\mathrm{ext}}(r,t)=I_{\mathrm{pk,ext}}
\exp(-2r^2/w_0^2-4\ln2\,t^2/t_{\mathrm{FWHM}}^2)$, let
$T_{\mathrm{in}},T_{\mathrm{out}}$ be the single-pass surface power transmittances.
Integrating the local solution over radius and time, with
$s=2r^2/w_0^2$ and $u=2\sqrt{\ln2}\,t/t_{\mathrm{FWHM}}$, gives

$$
\frac{U_{\mathrm{out}}}{U_{\mathrm{in}}}
=T_{\mathrm{in}}T_{\mathrm{out}}e^{-\alpha L}A(q),\qquad
q=\beta_{\mathrm{TPA}}T_{\mathrm{in}}I_{\mathrm{pk,ext}}L_{\mathrm{eff}},
$$

$$
A(q)=\frac{1}{\sqrt\pi}\int_{-\infty}^{\infty}\!du
\int_0^\infty\!ds\,
\frac{e^{-s-u^2}}{1+qe^{-s-u^2}}
=\frac{1}{q\sqrt\pi}\int_{-\infty}^{\infty}
\ln(1+qe^{-u^2})\,du.
$$

Here $A(0)=1$ and $A(q)=1-q/(2\sqrt2)+O(q^2)$, consistent with the
weak-absorption open-aperture expression in [7, Eq. (1)]. This derivation assumes
negligible diffraction and self-focusing in the sample and neglects multiple
internal reflections. For equal uncoated surfaces $T_{\mathrm{in}}=
T_{\mathrm{out}}=1-R$. A simulation launched inside the material must not apply
the entrance transmission twice. Evaluate `log1p` and the small-$q$ limit safely.

Reference [10] uses Gaussian averaging, but its printed energy-ratio definition
is inverted relative to its transmission plots, and Eq. (2) shows only one
surface factor while defining $R$ for one surface. Its printed temporal variable
also needs reconciling with a Gaussian FWHM. These statements were checked in the
original pages. Preserve the published Table 2 coefficients as reported; do not
silently “correct” them without the underlying data and fitting conventions.
Use the explicitly defined observable above as the independent numerical oracle.

#### 4.6 Why a degenerate spectrum is insufficient for broadband TPA

The resonant third-order polarisation in the optics convention has the form
([1, Eq. (1)])

$$
\widetilde P_i^{(3)}(\omega_s)=
\frac{3\epsilon_0}{4(2\pi)^2}\sum_{jkl}\iint
\chi^{(3)}_{ijkl}(-\omega_s;\omega_1,-\omega_2,\omega_3)
\widetilde E_j(\omega_1)\widetilde E_k^*(\omega_2)
\widetilde E_l(\omega_3)\,d\omega_1d\omega_2,
\quad \omega_3=\omega_s-\omega_1+\omega_2.
$$

Degenerate measurements specify only a slice,
$\chi^{(3)}(-\omega;\omega,-\omega,\omega)$, of this function. Neither an
output-frequency multiplier on $\mathcal F[|E|^2E]$ nor a filter on each input
can be deduced uniquely from that slice. In particular, photon pairs can be
absorbed when their **sum** energy reaches an allowed transition even though one
frequency lies below the degenerate half-gap threshold.

Positivity of a multiplier is also insufficient for passivity. As a mathematical
counterexample, use Fourier-series amplitudes $(1,1,-0.1)$ at three consecutive
positive bins, all others zero. The cubic coefficient at the last bin is
$1+4(-0.1)+(-0.1)^3=0.599$. A nonnegative loss multiplier equal to one only
there gives
$\partial_z\sum_k|a_k|^2=-2(-0.1)(0.599)=+0.1198$.
All mixing frequencies can be retained, so this is not an aliasing artefact.
The example rejects a general passivity claim; it does not say that every
restricted spectral approximation fails in its fitted application.

Published silicon work [8] compares coupled-wave and single-GNLSE descriptions
and finds that an output-frequency description cannot independently prescribe
both self and cross nonlinear coefficients. It supports an average-frequency
approximation for its measured silicon configuration, not a universal rule for
silica near its DUV edge. A future broadband phase must name its approximation,
validate nondegenerate absorption, preserve the appropriate photon/energy balance
and state where a fit is allowed. It cannot be accepted as “use the measured
$\beta(\omega)$” without this information.

The first useful broadband validation target is therefore a **restricted
two-band silicon model**, before an arbitrary-continuum response. Reference [8]
uses disjoint pump and signal spectra near 1850 and 1300 nm, separately computed
mode overlaps and nonlinear coefficients evaluated at the mean frequency. Its
successful comparison supports that approximation in its experimental regime;
it does not establish it for silica or for overlapping broad bands. Its §III B
also shows that a first-order dispersion of the Kerr coefficient can worsen the
comparison, while ignoring TPA dispersion underestimates cross absorption.
Do not tie dispersive Kerr and TPA corrections to one shared shock coefficient.

Any such continuation must derive pair-event balance. For true-power amplitudes
and pure cross absorption, write
$\partial_zP_p=-c_pP_pP_s$ and $\partial_zP_s=-c_sP_pP_s$, with
$c_p,c_s\ge0$ in $\mathrm{W^{-1}m^{-1}}$. One photon from each band is lost per
event, so

$$
\frac{-\partial_zP_p}{\hbar\omega_p}
=\frac{-\partial_zP_s}{\hbar\omega_s},\qquad
\frac{c_p}{\omega_p}=\frac{c_s}{\omega_s}.
$$

The total lost power per unit length is the common event rate times
$\hbar(\omega_p+\omega_s)$. In the convention of [8, Eqs. (1–2)],
$c_p=4\operatorname{Im}\gamma_{ps}$ and
$c_s=4\operatorname{Im}\gamma_{sp}$. Thus the cross coefficients in power
coordinates need not be equal. Re-derive the projection and carrier-generation
factors together; rounded published modal coefficients are comparison data,
not exact identities. This remains a subsequent constitutive proposal, not an
expansion of phase 1. The supplement referenced by [8] has not yet been checked.

A useful research option is a positive pair-absorption kernel in photon-number
coordinates. If amplitudes $a_k$ are normalised so $|a_k|^2$ is photon number,
form pair channels
$B_{\nu,\Omega}=\sum_{\omega_i+\omega_j=\Omega}g_{\nu,ij}a_i a_j$
with symmetric coupling and weights folded into $g$. The dissipative functional
$R=\tfrac12\sum_{\nu,\Omega}|B_{\nu,\Omega}|^2$ is nonnegative. Choosing
$\dot a=-\partial R/\partial a^*$ makes its photon-number derivative
$-4R\le0$; within each fixed-$\Omega$ channel the removed energy is the pair
energy $\hbar\Omega$ times the event rate. The coefficient and event convention
must be calibrated in the degenerate limit. This is a possible structure, not a
material model derived here: the measured diagonal does not determine the
off-diagonal $g$, phase coherence, causal dispersive partner or tensor response.
It is therefore not part of the production implementation proposed below.

#### 4.7 Material evidence and validity

The following independently measured values are reference points, not defaults
or an interpolation across all wavelengths:

| Material and measurement | Reported coefficient | SI coefficient | Qualification |
|---|---|---|---|
| Fused silica, 248 nm, [2, Table 1] | $4.5\times10^{-11}\ \mathrm{cm/W}$ | $4.5\times10^{-13}\ \mathrm{m/W}$ | Paper states ±50% absolute and ±15% relative uncertainty separately |
| Suprasil, 264 nm, [3, Table 3] | $(1.7\pm0.2)\times10^{-11}\ \mathrm{cm/W}$ | $(1.7\pm0.2)\times10^{-13}\ \mathrm{m/W}$ | Femtosecond measurement; retain sample identity |
| Fused silica, 211 nm, [4, Table 2] | $(50\pm8)\times10^{-11}\ \mathrm{cm/W}$ | $(5.0\pm0.8)\times10^{-12}\ \mathrm{m/W}$ | 250 fs measurement |
| Fused silica 1AK, Heraeus, 213 nm, [10, Table 2] | $(0.82\pm0.08)\ \mathrm{cm/GW}$ | $(8.2\pm0.8)\times10^{-12}\ \mathrm{m/W}$ | 24 ps; absolute-intensity uncertainty |
| Fused silica 7980 (ArF), Corning, 213 nm, [10, Table 2] | $(0.76\pm0.08)\ \mathrm{cm/GW}$ | $(7.6\pm0.8)\times10^{-12}\ \mathrm{m/W}$ | Same experiment; retain grade |
| Fused silica 8655, Corning, 213 nm, [10, Table 2] | $(0.79\pm0.08)\ \mathrm{cm/GW}$ | $(7.9\pm0.8)\times10^{-12}\ \mathrm{m/W}$ | Same experiment; retain grade |
| Fused silica 312, Heraeus, 213 nm, [10, Table 2] | $(0.80\pm0.08)\ \mathrm{cm/GW}$ | $(8.0\pm0.8)\times10^{-12}\ \mathrm{m/W}$ | Same experiment; retain grade |
| MgF₂, Fairfield crystal, 213 nm, [10, Table 2] | $(0.004\pm0.0004)\ \mathrm{cm/GW}$ | $(4.0\pm0.4)\times10^{-14}\ \mathrm{m/W}$ | 24 ps; measured sample/configuration |
| LiF, manufacturer unknown, 213 nm, [10, Table 2] | $<0.001\ \mathrm{cm/GW}$ | $<10^{-14}\ \mathrm{m/W}$ | Upper limit, not zero or a measured central value |
| Intrinsic crystalline silicon, 1220 nm, [7, Fig. 1 and accompanying text] | $(2.1\pm0.4)\ \mathrm{cm/GW}$ | $(2.1\pm0.4)\times10^{-11}\ \mathrm{m/W}$ | 300 K, nominally 200 fs, 125 μm wafer with [001] normal |

The table cells and their units were checked; the 264 nm and 211 nm table images
were also checked. Text extraction of [4] contains unmapped glyphs on several
pages, including the table page; the image confirms the quoted row. The 264 nm
paper's extraction has an unmatched figure caption on another page, which is not
evidence about the table's accuracy. No figure digitisation has been performed.
References [3,5] show why pulse duration, sample and irradiation history belong
with the data: transient and cumulative absorption cannot automatically be fitted
as an instantaneous TPA coefficient. Fresnel surface losses must also be separated
from absorption within the material.

The 213 nm table cells and the 1220 nm silicon point were checked against the
original PDF pages. Reference [10] uses 24 ± 2 ps Gaussian pulses at 1 Hz and
reports coefficients inferred from the intensity dependence of transmitted
energy. The fused-silica values agree across its tested grades within the quoted
errors, while the linear losses differ. They are about 1.5–1.6 times the 211 nm
femtosecond value in [4]. Keep these as separate measurement records; neither
the small wavelength difference nor the different pulse duration alone has been
shown to explain that discrepancy. Do not fit a universal silica curve through
both experiments or reduce their shared intensity-calibration uncertainty by
treating the four grades as independent repeated measurements.

Reference [7] supplies a silicon spectrum over 850–2200 nm and an explicit
1220 nm reference point. Its fitted indirect-transition model is a sum of
allowed/forbidden phonon-assisted contributions, not the direct-gap expression
below. It also reports a substantial mismatch between measured Kerr dispersion
and a reconstruction from its approximate nondegenerate TPA model. Use the
measured Kerr and TPA data independently within their qualification, without
enforcing that approximate reconstruction as an identity. A digitised spectrum
or analytic fit requires a reproducible source-data record and a review of the
underlying indirect-transition theory; no fitted spectrum is introduced here.

For a future coefficient catalogue, retain material grade, temperature, crystal
and polarisation information when reported, pulse conditions, uncertainty type,
and whether a value is a measurement or an upper limit. Unknown conditions stay
unknown. A single measurement still does not determine a valid wavelength band.

For comparison, the two-parabolic-band model in [6, Eqs. (1–2)] uses

$$
x=\frac{\hbar\omega}{E_g},\qquad
F_2(x)=\frac{(2x-1)^{3/2}}{(2x)^5}\quad(x>1/2),
\qquad F_2=0\quad(x\le1/2).
$$

In SI it can be written
$\beta=K_{\mathrm{SI}}\sqrt{E_p}F_2(x)/(n^2E_g^3)$, with energies in joules and
$K_{\mathrm{SI}}$ in $\mathrm{m\,J^{5/2}/W}$. A quoted numerical $K$ from a
cm/GW and eV convention must be converted with both unit factors:
$K_{\mathrm{SI}}=10^{-11}K_{\mathrm{lit}}(1\ \mathrm{eV\ in\ J})^{5/2}$.
If the alternative variable $y=2\hbar\omega/E_g$ is used, the shape is
$(y-1)^{3/2}/y^5$ with threshold $y>1$, not the previous expression with $y$
substituted for $x$.

Reference [6] reports poorer agreement for wide-gap materials near the absorption
edge than for the semiconductors on which the model was established. Silicon's
indirect-gap, phonon-assisted process is a further reason not to reuse this
formula as a universal stock model; [7] provides a dedicated measured spectrum.
No fitted gap, power-law exponent or universal scaling constant is adopted here.
Constants $c,\epsilon_0,\hbar$ and the eV conversion come from `solprop.constants`.

Nonlinear Kramers–Kronig relations involve nondegenerate response functions
([6, Eq. (4)] and [9,11]). A Hilbert transform of the degenerate TPA curve alone
does not supply a physically complete broadband Kerr response. Independently
measured local Kerr and TPA coefficients can be used over a stated narrow band;
they must not be advertised as a causal model over an unrestricted spectrum.

More specifically, [9, §II] holds the pump frequency fixed while integrating the
probe-frequency response, and takes the degenerate limit afterwards. Its
absorptive counterparts include electronic Raman and optical Stark terms as well
as TPA. This electronic Raman contribution is not the delayed vibrational Raman
response already provided by solprop. Reference [11, §II] corrects a
degenerate/nondegenerate factor of two and restores quadratic-Stark terms omitted
in [9]; both papers discuss removing unphysical infrared divergences of the
two-band calculation. A broadband implementation must resolve these conventions,
units and limits by derivation and independent quadrature, rather than copying
either paper's table into a temporal kernel. The papers inform a future model;
they do not turn the constant-current approximation into a complete causal one.

#### 4.8 Generated carriers are a separate constitutive model

For unit quantum yield in a narrowband semiconductor, the local pair-generation
law would be

$$
\partial_t N=\frac{\beta_{\mathrm{TPA}}I^2}{2\hbar\omega_r}
-\frac{N}{\tau},\qquad N\ [\mathrm{m^{-3}}],\quad\tau\ [\mathrm{s}].
$$

The two photons create one electron–hole pair. Carrier refraction and absorption
would then contribute $[ik_0\Delta n(N)-\alpha_{\mathrm{FC}}(N)/2]b$ in optics
convention. This is a possible subsequent model, motivated by [1]; none of its
lifetimes, effective masses, mobilities, quantum yields or empirical carrier
coefficients is fixed by this SEP.

The existing atomic plasma model is not this model: it has neutral depletion,
gas rate tables and its own ionisation-energy loss. Adding both losses for the
same excitation counts absorbed energy twice. A future solid model must derive
generation and optical TPA loss from the same events, then add only the
subsequent carrier response. Carrier histories are recomputed from the local
field at each RHS evaluation with a declared temporal initial condition; they
are not evolved along z as if they travelled with the pulse. Pulse trains need
additional temporal boundary conditions. For multimode carriers, a local
generation proportional to $|E|^4$ and an optical carrier overlap proportional
to $|E|^2$ generally involve a sixth-order overlap, not a universal
$A_{\mathrm{eff}}^{-2}$ rule. Silica trapping/defect dynamics require different
evidence again. These extensions need their own accepted constitutive proposal.

### 5. Proposed interface

Use a small immutable coefficient record, tentatively `TPACoefficient`, containing
`beta` (finite, nonnegative, m/W), `at` (positive vacuum wavelength, m),
`valid_wavelengths` (positive ordered bounds, m, containing `at`),
`outside_band_fraction` (a declared maximum fraction of spectral weight outside
that band, dimensionless), and `source` (measurement/model identity and stated
conditions). These declare the constant-current approximation's validity, not a
sampled spectral curve.
The record retains the original coefficient and source even if zero is compiled
to an absent runtime term. Its provenance preserves the measurement conditions,
reported uncertainty and bound-versus-measurement distinction of §4.7; an upper
limit must not silently become a nominal coefficient.

`TPAResponse(coefficient, index=...)` binds that record to the positive real
material index at `at` for the ordinary UPPE/spatial routes. Physics converts it
to $C_J$; a backend never looks up a material. Follow the existing material-law
pattern if the index is resolved from a law rather than supplied as a number.
Use the normal `responses` sequence and simulation constructors. The first
version accepts a constant bulk response with density one: a gas density path
must not accidentally multiply a bulk coefficient. Varying materials and regional
TPA are explicitly refused.

The GNLSE convenience path takes the same coefficient record as `tpa=...` and
performs its index-one calibration. `tpa=None` is the sole off value. Bulk TPA
requires an explicitly supplied physical `aeff`, even with a user-specified Kerr
`gamma`; the existing numerical default of one square metre is not evidence of
a physical area. Moving areas already supported by a route use that route's
existing taper scaling, once only. Exact public spelling and placement are Q3.

Illustrative calls, not executable API:

```text
coefficient = TPACoefficient(
    beta=beta_measured, at=wavelength_measured,
    valid_wavelengths=qualified_interval,
    outside_band_fraction=declared_allowance, source=measurement_identity,
)
responses = (KerrResponse(...), TPAResponse(coefficient, index=n_material))

prop_gnlse(..., gamma=gamma_measured, aeff=area_measured, tpa=coefficient)
```

Validate the reference wavelength and the occupied launch band against the
declared model band, and monitor departure from it as propagation broadens the
spectrum. A single measurement does not establish a broad validity interval: a
custom record must explicitly declare that approximation. Do not require every
numerically active bin to lie inside the material band: field-grid padding and
collars deliberately extend beyond the occupied spectrum. Conversely, the
coefficient must not be set to zero outside its measured band to manufacture an
absorption edge.

The proposed guard uses a declared maximum fraction of spectral weight outside
the model band, checked initially and after accepted steps. The allowance is
part of the model-validity record, distinct from the RK tolerance, and has no
invented universal default; Q2 must settle it before acceptance. Use the modal
power measure, or the spatial route's positive quadrature-weighted spectral
measure as a *bandwidth diagnostic*, without calling the latter a general
in-medium energy. Existing window-statistic kernels can supply the reductions.
The model stops with a named validity error when the allowance is exceeded; it
never clips the spectrum to satisfy its material data. An accepted-step guard
does not certify unobserved intermediate-stage bandwidth, which must be checked
in the short convergence studies. Preserve the current masking conventions.

Reject nonfinite/negative values, missing area, incompatible density, arbitrary
TPA spectral arrays/callbacks, unsupported component counts or regional laws,
and every unimplemented backend combination before allocating workspaces. A
`FieldGrid` TPA response must work without a Kerr response beside it. Its need for
an analytic field must be represented independently of the current Kerr flag.
TPA with a separately requested THG response is initially refused until the
material law is qualified across both bands; the ordinary non-TPA THG route is
unchanged.

Scans vary the immutable physical declaration between translated problems. Resume
and batching compare its identity and compiled factors, including reference
wavelength, model form and area calibration. Do not reuse tables merely because
two runs have the same numerical `beta`. Existing launch, output and processing
amplitudes keep their conventions. Record `model="constant-current-scalar-tpa"`
and the coefficient provenance in run metadata; no carrier-density statistic is
fabricated by a loss-only run.

### 6. Implementation strategies

#### 6.1 Recommended placement and contract

Keep coefficient validation, SI conversion and the narrowband derivation in P,
sampling/compilation in T and current evaluation in B. A typed optional TPA leaf
contains the compiled current coefficient, model identity and supported grid
form; the source assembly records the active mask and the matched $-i/\omega$
factor. These records carry bulk rather than per-particle units. Use a typed
union for post-transform assemblies if extending `spectral_responses`; do not
turn it into a tuple of arbitrary callables or object dictionaries.

Phase 1 bumps `SCHEMA_VERSION` to the next available number and updates the
contract, literal field tests, capabilities and refusal paths together. Do not
reserve 16 in advance of other work. Changes to stage bundles or the protected
driver surface would also be versioned; none is necessary merely to carry a
constant TPA coefficient. Keep the ordinary polarisation and TPA current streams
distinct until the latter has its current-to-polarisation multiplier. Folding
the multiplier into the ordinary Kerr/Raman sum would change those responses.

The execution sequence is physical-field synthesis, local current, the existing
nonlinear temporal window, spatial projection, temporal transform/crop, current
assembly, then the existing propagation prefactor. Follow each geometry's actual
order: Cartesian sources must be transformed at physical nodes before the
transverse transform; a general rule about modal projection cannot replace that
path. A spatially uniform scalar multiplier commutes with a linear projection,
but a future node-dependent material coefficient must be applied before
projection. Embedded spatial-error estimates must see the TPA source too.

No new integrator or operator splitting is needed. Let the existing embedded RK
error estimate see the loss in every stage. A strong TPA term shortens the
nonlinear length $1/(\beta I)$ and can increase the required number of accepted
steps even when the source evaluation itself is cheap.

#### 6.2 Backend work and enabled cost

| Work | Expected difficulty and cost |
|---|---|
| Scalar envelope current in NumPy | Small physics calculation; moderate integration work across transforms, provenance, admission and tests |
| Field-grid support | Moderate: analytic-field planning, correct half-spectrum factors, independent TPA-only admission and parity |
| Numba CPU | Moderate: add the body and specialised enabled source path, respect closure-cache rules and ledger ownership |
| Kokkos CPU | Moderate: extend adapter and native records, resident views/plans, bindings/stubs, native and seam tests |
| Full-vector regional/crystal TPA | Substantial: tensor/passivity data, region compilation and all general-modal executors; excluded from first scope |
| Broadband DUV or solid carrier feedback | Substantial modelling and validation work, not a change of coefficient dtype |

These are engineering estimates, not measured schedules. The small algebraic
body is not the dominant development cost; the independent normalization checks
and geometry/backend coverage are.

For $M$ processed physical nodes, $C$ components and $T$ fine time samples, the
local current is $O(MCT)$. A straightforward implementation uses one extra forward
temporal transform per current stream, $O(MCT\log T)$, plus projection and, where
needed, spatial transforms. Envelope TPA needs no population scan and no new
inverse transform to recover the field. Field TPA may need an additional analytic
transform and complex scratch if the existing family only synthesises a real
field. Where an analytic field is already present it should be reused.

Workspaces are chunk bounded: one current buffer, its spectral/projected scratch
and, where necessary, analytic-field scratch and FFT plans. List actual shapes
and bytes in `memory_ledger()`; do not allocate a full nodes-by-time-by-stage
array. Every buffer and plan is allocated at backend construction, including
partial-chunk handling. On Kokkos the new source remains resident; no extra
host/device transfer or unplanned fence is permitted per stage.

The enabled validity guard adds a reduction over the state after accepted steps,
and a planned scalar status transfer on devices. Account for it separately from
the source transforms. It is absent on disabled and compiled-zero problems.

Some envelope Kerr and TPA evaluations share the same cubic product. A uniform
coefficient and identical projection/windowing may allow sharing its transform,
or an enabled-only fused path. That is an optimisation to qualify after a clear
reference implementation. Raman, different material regions, harmonic terms
and different coefficient profiles invalidate casual reuse. A complex Kerr
coefficient is not the default implementation because it conflates these cases
and the current and susceptibility approximations of §4.3.

#### 6.3 Cost when disabled

The required design is construction-time selection of the existing non-TPA path.
An absent or exactly zero compiled TPA leaf creates no field-sized buffers, plans,
source passes, transforms, reductions, population histories or device transfers.
It must not force all field-grid runs into the analytic-field family. Preserve
the arithmetic and reduction order of the old path; do not add a zero complex
term to every sample. Numba uses a separate specialisation or an unchanged old
factory; Kokkos selects the old launch path outside the element loop.

This makes zero additional algorithmic work achievable. It does **not** establish
zero wall-clock overhead by inspection: additional branches, code size, register
pressure, plan selection and altered compiler decisions can affect even disabled
cases. Section 7 requires exact output checks where the disabled case runs the
unchanged old path (a structural no-op contract), outputs within a rounding bound
fixed before measurement where it runs a separate specialisation, an unchanged
memory/transform ledger and interleaved short timings.
There will be a one-time schema compatibility cost for external backends. No
performance claim is made before those measurements.

No new runtime dependency is proposed. NumPy/SciPy and the current backend FFT
libraries supply the necessary operations.

### 7. Verification and validation

The first oracle is the derived physical law, then `numpy_ref`, not another
implementation. Each numerical tolerance must be written before comparing
backends and explained by conditioning, summation/FFT bounds or the discretisation
under test. All asserted tolerances are positive.

| Check | Quantity and reference | What sets the tolerance |
|---|---|---|
| Unit conversion and threshold | SI factors; $2\hbar\omega=E_g$ for any subsequently admitted two-band model | Floating-point evaluation of exact unit factors; near-threshold one-sided cases |
| Absolute coefficient | A monochromatic local field and mode, at synthetic indices 1, 1.5 and 3.5 | Operation count times machine epsilon; detects missing index and factor-two errors |
| Pure attenuation | $I_0/(1+\beta I_0z)$ and its finite-linear-loss form | Independent analytic reference, RK error and finite temporal/spatial resolution |
| Kerr plus TPA | Complex amplitude and logarithmic nonlinear phase in §4.5 | Same, with a separate phase-conditioned absolute bound |
| Grid convergence | Three step sizes and three suitable temporal resolutions | Observed order against the method's order; separate time sampling from z integration |
| Loss balance | Total modal flux loss or local physical work $\int q\,dt\,dA$ | Quadrature error, source projection error and RK error; no unweighted radial spectral sum |
| Geometry parity | One mode versus mode average; wide-beam collinear limit; equivalent radial/Cartesian input | Analytic normalisation plus transverse quadrature/diffraction truncation |
| Field/envelope | Same narrowband physical pulse, coefficient and intensity | Carrier-resolution and rotating-wave error quantified by successive bandwidth reduction |
| Coherent multimode response | Cross absorption and interference from directly synthesised fields | Independent physical-space quadrature and deterministic projection bounds |
| Backend parity | One RHS, one step, eight steps, statistics and a complete short run | FFT and summation bounds plus step accumulation; NumPy is the backend oracle |
| Disabled and zero term | Fields, histories, counters, workspace and plans against the old path | Exact for counters, workspace, plans and for outputs of the unchanged old path (structural no-op); a rounding bound fixed before measurement for outputs of a separate specialisation; timing distribution rather than a fabricated universal percentage |

Add input/admission tests for every refusal in §5, especially a TPA-only field
problem, plasma plus an unimplemented TPA entry, negative/NaN coefficients,
nonphysical index, bulk density misuse and spectral validity violations.
Verify the current's positive/negative-frequency symmetry and its real DC/Nyquist
handling without dividing by zero. Verify padding independently with a direct
DFT/convolution on a small grid, and demonstrate that the active mask removes no
in-band product in each analytic test. Different masks are not evidence of a
material absorption edge.

Loss is monotone for a pure absorptive scalar reference with the correct complete
energy metric, not necessarily in each spectral bin or individual mode. Keep the
three-tone counterexample in §4.6 as a design check for any future spectral
approximation; do not assert that an absorbing nonlinear medium can never fill a
previously empty frequency bin. Do not use the current spatial vacuum-energy
statistic as a general in-medium flux oracle.

For experimental validation, independently integrate the analytic local
transmission over the paper's spatial and temporal pulse shapes, including the
stated Fresnel and linear losses. Use the quoted measurement uncertainty as a
separate physical-validation allowance; do not loosen numerical parity tolerances
to match it. The table points in §4.7 check units and provenance, but a model with
a supplied measured coefficient reproducing itself is not an independent
validation of an invented spectral fit.

In particular, compare direct radial/time quadrature with the Gaussian
transmission formula in §4.5, its weak-loss limit and the correct linear/surface
limit. Keep 211 nm femtosecond and 213 nm picosecond silica comparisons separate,
and use the explicit silicon point in §4.7 for a traceable single-frequency check.
Before claiming agreement with a measured transmission curve, resolve the paper's
observable, pulse-width and surface conventions and obtain or reproducibly
digitise its data; matching a supplied coefficient is only a calibration check.
For the subsequent two-band model, add pair-event reciprocity and separate
self/cross tests before reproducing [8]. Do not reproduce its full 2 cm experiment
in a test or timing: qualify on short windows, and reserve a full comparison for
a manually run worked example.

Benchmark off, explicit-zero, TPA-only, Kerr-only and Kerr-plus-TPA on a small
mode-averaged case, a chunked multimode case and a representative spatial case,
on each CPU backend. Include field and envelope grids, one and several threads,
and partial node chunks. Warm up and interleave baselines and candidates. Record
RHS time, transform counts, peak/owned memory, compilation/setup separately,
accepted/rejected steps and time for about ten accepted steps in
[the benchmark ledger](../benchmarks.md). Also replay a fixed short schedule to
separate kernel cost from changed adaptive work. Do not time full-length
propagation or extrapolate a short timing to it.

Each implementation phase runs `uv run poe check`, and before commit/merge
`uv run poe test-slow` and `uv run poe cov`. The coverage lane exercises the new
bodies with JIT disabled. Kokkos CPU additionally runs its native/distribution
gate and solprop parity suite with the freshly installed build. GPU qualification
is separate on an actual device; a successful CPU build is not a GPU result.

### 8. Documentation

Add `docs/model/two-photon-absorption.md` with the coefficient definition,
derivation, current model, material evidence, normalization of each route,
validity restrictions and short worked examples. Link it from the nonlinear and
running pages and the manual toctree. Update the GNLSE guide, public docstrings,
response/admission tables, kernel contract, architecture and `CHANGELOG.md`.
Keep `README.md` and `docs/introduction/first-run.md` consistent with the qualified scope.

A quick scalar tutorial should compare short propagation with the exact nonlinear
attenuation law and show the effect of using a measured coefficient. A DUV
worked example is appropriate only after its bandwidth is justified; a silicon
example must say whether carrier effects are negligible or omitted. Do not
present a loss-only simulation as a validated general silicon model. Regenerate
any edited gallery outputs through the required gallery command.

At Final, the manual pages link to this SEP, and the changelog cites it. Record
deliberate differences in normalization, coefficient validation and broadband
scope on [the differences page](../../guide/workflows/luna.md) when they are implemented.

### 9. Compatibility and impact

No physics default changes. Existing problems do not acquire TPA by naming a
solid material, nor do they change their Kerr/Raman allocation. There is a seam
version bump in phase 1; external backends must update their named member checks
or refuse the new schema. During staged delivery, an unsupported backend refuses
TPA by name and identifies a supporting backend; no silent fallback is allowed.

The field dataset, optical Fourier convention and power normalisation stay as
they are. Add model/provenance metadata to the existing response identity rather
than adding a new evolving state dataset. Resume must reject a changed TPA
declaration. No population is implied by the presence of a TPA coefficient.
General modal, crystal and broadband capabilities remain explicit refusals until
their separate model/admission work is accepted and qualified.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Add a complex value to every Kerr coefficient | Small for a fixed-prefactor envelope equation; conflates real-field support, THG, Raman partition and TPA provenance, and may change disabled arithmetic |
| Output-frequency $\beta(\omega)\mathcal F[|E|^2E]$ | Cheap and sometimes useful as a fitted approximation; degenerate data do not fix cross absorption, and nonnegative coefficients do not guarantee passivity |
| Frequency-filter the field before the cubic | Can be made dissipative with an adjoint filter, but then predicts a separable mixed-frequency response that needs independent justification |
| Exact nonlinear attenuation in a split step | Useful independent reference in the local nondispersive limit; introduces splitting errors and awkward multimode projection into the production stepper |
| Generic complex tensor/spectral response engine first | Could eventually serve many models, but substantially widens the seam and testing before the scalar use case is delivered |
| Reuse atomic plasma loss/rate tables | Wrong excitation and carrier constitutive model for solids, with a risk of counting optical energy loss twice |
| Microscopic causal broadband response first | The principled route when broadband accuracy is essential; needs nondegenerate/tensor data, a dispersive partner and a larger research and validation effort |

The recommendation is the explicit scalar current first, with restricted
admission. Acceptance of that scope is a physics decision, not permission to
approximate arbitrary DUV or silicon cases silently.

### 11. Phases

Each phase uses its own `feature/sep-0002-two-photon-absorption-...` branch and
appends its report to Part II in its merging pull request.

1. **NumPy reference and contract.** Resolve Q1–Q4, deliver the physical declaration,
   compilation, schema bump, source assembly and scalar grid/geometry scope of
   §2, with the analytic tests, manual page, provenance and a short example.
   Update the Numba and Kokkos schema/admission adapters sufficiently to preserve
   old supported problems and refuse TPA explicitly; this compatibility work
   does not implement their numerical TPA paths. Verify the disabled oracle path
   and record its baseline. No unverified stock coefficients are required.
2. **Numba CPU.** Deliver the shared body, specialised current/analytic paths,
   preallocated workspaces, memory ledger, interpreted coverage and parity on
   every phase-1 admitted geometry. Measure enabled and disabled cost, including
   thread-count agreement within a rounding bound fixed before measurement and the
   existing cache policy. NumPy remains the oracle.
3. **Kokkos CPU.** Deliver native data/bindings, resident current transforms and
   assembly, exact admission, ledger and short-run parity. Run both repositories'
   required gates with the CPU build installed. Keep all old launches available
   for the off case. Record GPU qualification as not run unless a device lane is
   actually completed.
4. **Validation and completion.** Resolve the chosen material record(s) in Q2,
   distinguish custom coefficients from shipped data, complete analytic and
   experimental checks and compare all three backends. Use the newly checked
   213 nm records and explicit silicon point as candidate calibration cases;
   first reconcile transmission conventions with the Gaussian oracle and retain
   the discrepancy between the two silica experiments. Acquire or reproducibly
   digitise curve data before claiming an independent experimental reproduction.
   This phase can add named narrowband data records only after their source
   conditions and validity are established; it does not turn isolated measurements
   into an unrestricted interpolation. Complete the tutorial/manual, settle
   remaining questions or move excluded work to the roadmap, and archive the SEP
   as Final only for the accepted scalar scope. If wider physics is required,
   revise the Draft before acceptance or write a subsequent SEP; do not quietly
   broaden this phase.

### 12. Open questions

- **Q1 — First physical scope, maintainer, before acceptance.** Is scalar loss with
  negligible/omitted carrier feedback and a justified narrow band the intended
  first deliverable? If broadband DUV or silicon carrier dynamics must work first,
  extend the theory and phases before freezing Part I.
- **Q2 — Material and validity, maintainer/material reviewer, before acceptance.**
  Which material grade/crystal orientation, wavelengths, pulse durations and
  intensity regime are the target? Define how a user declares constant-current
  validity. The measurements in §4.7 are useful anchors but do not fix a
  permissible broadband interpolation. Resolve which silica experiment and
  sample conditions apply; do not average away their discrepancy. Phase 4 owns
  any shipped records.
- **Q3 — Public interface, maintainer, before phase 1.** Confirm the coefficient
  record/response split, explicit-area requirement for GNLSE and model identity.
  Decide how an existing material law supplies the reference index without
  inventing a second material catalogue.
- **Q4 — Initial refusal policy, maintainer, before acceptance.** Confirm exclusion
  of regional full-vector TPA, THG coexistence and atomic-plasma overlap in the
  same solid. Independent responses in physically separate media need a regional
  model, not an unchecked sum. Any narrower accepted backend/geometry exception
  must be written into this proposal.
- **Q5 — Broadband continuation, subsequent physics SEP.** Obtain a justified
  nondegenerate response for the intended DUV/semiconductor band; compare
  pair-kernel, causal response and restricted coupled-band approaches. Start the
  comparison from the disjoint-band silicon case in [8] with independent
  self/cross coefficients and pair-event reciprocity. Prove energy/event balance
  before discussing an FFT optimisation. This is not a silica constitutive law.
- **Q6 — Source and data completion, material reviewer.** The five requested main
  articles [7–11] have now been obtained and their relevant content checked.
  The supplementary derivation referenced by [8] is still needed before relying
  on its detailed coupled/single-equation correspondence. Published tables are
  available; underlying transmission/spectral data or reproducible digitisation
  remain necessary for fitting or reproducing experimental curves. These are
  specific validation tasks, not a reason to guess missing coefficients.

### 13. References

1. Q. Lin, O. J. Painter and G. P. Agrawal, “Nonlinear optical phenomena in silicon
   waveguides: modeling and applications,” *Optics Express* **15**, 16604 (2007).
   [doi:10.1364/OE.15.016604](https://doi.org/10.1364/OE.15.016604).
   Checked: frequency mixing, susceptibility conversion, modal overlap, tensor
   dependence and carrier equations, especially Eqs. (1), (8), (28–36), (47–48).
2. A. J. Taylor, R. B. Gibson and J. P. Roberts, “Two-photon absorption at 248 nm
   in ultraviolet window materials,” *Optics Letters* **13**, 814 (1988).
   [doi:10.1364/OL.13.000814](https://doi.org/10.1364/OL.13.000814).
   Checked: coefficient convention, Table 1 and uncertainty statement.
3. A. Dragomir, J. G. McInerney and D. N. Nikogosyan, “Femtosecond measurements
   of two-photon absorption coefficients at λ = 264 nm in glasses, crystals, and
   liquids,” *Applied Optics* **41**, 4365 (2002).
   [doi:10.1364/AO.41.004365](https://doi.org/10.1364/AO.41.004365).
   Checked: transmission law, sample dependence, pulse conditions and Table 3.
4. S. A. Slattery and D. N. Nikogosyan, “Two-photon absorption at 211 nm in fused
   silica, crystalline quartz and some alkali halides,” *Optics Communications*
   **228**, 127–131 (2003).
   [doi:10.1016/j.optcom.2003.09.086](https://doi.org/10.1016/j.optcom.2003.09.086).
   Checked: Table 2 against its image, measurement duration and uncertainty.
5. E. Eva and K. Mann, “Calorimetric measurement of two-photon absorption and
   color-center formation in ultraviolet-window materials,” *Applied Physics A*
   **62**, 143–149 (1996).
   [doi:10.1007/BF01575711](https://doi.org/10.1007/BF01575711).
   Checked: instantaneous absorption versus transient/cumulative induced loss.
6. R. DeSalvo, A. A. Said, D. J. Hagan, E. W. Van Stryland and M. Sheik-Bahae,
   “Infrared to ultraviolet measurements of two-photon absorption and n₂ in wide
   bandgap solids,” *IEEE Journal of Quantum Electronics* **32**, 1324–1333 (1996).
   [doi:10.1109/3.511545](https://doi.org/10.1109/3.511545).
   Checked: Eqs. (1–4), native units, threshold and limitations near the edge.
7. A. D. Bristow, N. Rotenberg and H. M. van Driel, “Two-photon absorption and
   Kerr coefficients of silicon for 850–2200 nm,” *Applied Physics Letters* **90**,
   191104 (2007).
   [doi:10.1063/1.2737359](https://doi.org/10.1063/1.2737359).
   Full article checked for spectral range, experimental conditions, the explicit
   1220 nm point, indirect-gap interpretation and Kerr-reconstruction limitation.
   Original pages checked because extraction has unmapped glyphs; no curve
   digitised or fit coefficients adopted.
8. N. Poulvellarie, C. Ciret, B. Kuyken, F. Leo and S.-P. Gorza, “Highly
   Nondegenerate Two-Photon Absorption in Silicon Wire Waveguides,” *Physical
   Review Applied* **10**, 024033 (2018).
   [doi:10.1103/PhysRevApplied.10.024033](https://doi.org/10.1103/PhysRevApplied.10.024033).
   Published article checked, especially Eqs. (1–6), its restricted mean-frequency
   approximation and self/cross coefficient mismatch. The cited supplement is
   absent; its experiment has not been reproduced here.
9. M. Sheik-Bahae, D. C. Hutchings, D. J. Hagan and E. W. Van Stryland,
   “Dispersion of bound electron nonlinear refraction in solids,” *IEEE Journal
   of Quantum Electronics* **27**, 1296–1309 (1991).
   [doi:10.1109/3.89946](https://doi.org/10.1109/3.89946).
   Full text checked for fixed-pump nonlinear Kramers–Kronig reasoning, the
   two-band threshold/scaling and infrared-divergence treatment. The fixed-pump
   discussion was also checked on the original page. Read with the corrections
   in [11]; the microscopic derivation has not been independently reproduced.
10. S. Patankar et al., “Two-photon absorption measurements of deep UV
    transmissible materials at 213 nm,” *Applied Optics* **56**, 8309–8312 (2017).
    [doi:10.1364/AO.56.008309](https://doi.org/10.1364/AO.56.008309).
    Published article, exact Table 2 cells and original pages checked: coefficients,
    sample identities, uncertainties, pulse conditions and transmission notation.
    Extraction has unmapped glyphs on pages 1–2. Underlying fit data have not been
    obtained, and the experiment has not been independently reproduced.
11. M. Sheik-Bahae, J. Wang and E. W. Van Stryland, “Nondegenerate optical Kerr
    effect in semiconductors,” *IEEE Journal of Quantum Electronics* **30**,
    249–255 (1994).
    [doi:10.1109/3.283767](https://doi.org/10.1109/3.283767).
    Full text and original correction pages checked for the nondegenerate
    dispersion relation, pair threshold, factor-of-two correction and additional
    Stark terms in §II. Its tabulated functions have not been transcribed or
    independently re-derived for an implementation.

## Part II — Implementation record

## Part III — History

- 2026-09-26: Drafted after analysis of the current response, normalisation and
  backend contracts and the primary literature. Proposed scalar current scope;
  broadband, tensor and carrier decisions remain explicit open questions.
- 2026-09-27: Reviewed the five newly supplied main articles. Added 213 nm material
  records, a silicon reference point and a derived Gaussian transmission oracle;
  strengthened measurement provenance, corrected-theory and subsequent two-band
  validation requirements. Retained the proposed scalar first scope and recorded
  the remaining supplement/data gaps.
- 2026-10-17: Restated for [SEP 25](archive/sep-0025-rounding-level-reproducibility.md).
  The disabled-term checks (§6.3 and the §7 table's last row) are exact only where the
  disabled case runs the unchanged old path, a structural no-op, and for counters,
  workspace, plans and the memory/transform ledger; a disabled case compiled as a
  separate specialisation is held to a rounding bound fixed before measurement.
  Phase 2's thread invariance (§11) becomes thread-count agreement within such a
  bound.
