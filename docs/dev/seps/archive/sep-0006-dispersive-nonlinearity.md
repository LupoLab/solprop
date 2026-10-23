# SEP 6 — Frequency-dependent electronic third-order response

| | |
|---|---|
| **SEP** | 6 |
| **Title** | Frequency-dependent electronic third-order response |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | Fable 5.1 (AI); Opus 5.5 (AI); John Travers (human) |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | Accepted by John Travers, 2026-10-07, after the human and AI reviews, in the SEP 6 final-review discussion; Final, approved by John Travers, 2026-10-08 |
| **Requires** | [SEP 3](sep-0003-optmat-materials.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers; Phases 0–4 merged from `feature/sep-0006-dispersive-nonlinearity-0` to `-4` (and `-1c`, `-2c`) in solprop, Phases 1 and 3 in solprop-kokkos |
| **Manual pages** | [A frequency-dependent electronic response](../../../guide/media/dispersive-kerr.md); [materials: the dispersive electronic policies](../../../guide/media/material-selection.md#the-dispersive-electronic-policies) |

## Part I — Proposal

### 1. Abstract

Add two opt-in, frequency-dependent electronic third-order responses: a symmetric
factorization built from the degenerate Kerr dispersion curve, and an evaluator
of the full curated gas dispersion polynomial. Both construct one nonlinear
polarization containing all retained mixing processes; neither identifies SPM,
XPM, FWM or THG events during propagation. Retain the carrier-evaluated model as
the initial default. The factorized model offers low cost with a measurable
approximation to nondegenerate mixing; the polynomial offers exact evaluation
of the stated material approximation at a larger but still FFT-based cost.
Qualify numerical correctness, material validity and propagation sensitivity
separately. The first target is transparent gases, especially broadband SPM and
widely separated phase-matched FWM, for every gas optmat publishes a
hyperpolarizability model of. Two demonstration cases anchor the comparison:
deep-ultraviolet FWM in argon and vacuum-ultraviolet resonant dispersive-wave
emission in helium. Silica and VUV operation expose additional data and model
gaps. Success requires documented errors and costs, appropriate invariants, and
parity across NumPy, Numba and Kokkos within the admitted scope; the Kokkos
implementation is required, not optional.

### 2. Motivation and scope

**Accepted; implementation not started.** This proposal builds on SEP 3's migration
to curated material data, now final, and deliberately goes beyond its
carrier-evaluated nonlinearity. The proposal was first audited against solprop
`27f2296`, optmat `b7ee5a70214aaf95c095b4e336449c1d08a61307` and solprop-kokkos
`87ac816190ffb0c3978de7e236b6e30d7c068ab8`, and re-audited for its final review
against solprop `b70f2fb`, optmat `v0.3.0`
(`3f1eaa4c2e7c18f9eda31e2a6f40677763d9890e`, the tag solprop pins) and
solprop-kokkos `fa5d57c1597e716cb8428bace7e57f53d1e14b29`. The fifteen stored
third-order fits are byte-identical between the two optmat revisions; v0.3.0
adds model identities and statuses to them. The calculations below are proposal
experiments, not a qualified propagator or an implementation-phase report.

The motivating simulations use pumps from 343 to 2000 nm and can generate
radiation from the VUV to the IR. Broadband SPM and phase-matched FWM matter more
often than explicit THG. A model should represent the coupling of those bands
without dividing a continuous spectrum into named processes or assigning
unrelated coefficients to forward and reverse conversion.

A full-field $E^3$ term is general in the **frequency combinations it includes**.
It still assumes the same susceptibility for every combination. Its broad
spectral support does not make its material response dispersive. Conversely,
introducing material dispersion need not abandon a single broadband field:
the susceptibility's mathematical structure can make its convolution efficient.

Included here are the electronic response, its data interface, frequency-domain
evaluation, field/envelope conventions, mixtures, spatial/modal integration,
backend cost, provenance and qualification.

**Gas scope.** The dispersive evaluators are supported for every gas for which
optmat publishes a hyperpolarizability model, with no gas list of solprop's own:
the scope follows optmat. At v0.3.0 that is fifteen gases — He, Ne, Ar, Kr, Xe,
H2, D2, N2, O2, CO2, CH4, SF6, CO, N2O and O3 — plus mixtures of them. solprop
builds twelve of them as species today; the existing air interfaces are
compositions of those, not an independent nonlinear fit. SEP 3 (D13) left
CO2, CO and O3 as built-in gases to separately qualified follow-up work; when
solprop admits them their electronic models resolve through the same path with no change to
this proposal, and §4.7–§4.8 already tabulate them. A gas whose optmat record
lacks the four dispersion parameters is refused by name, never given a
borrowed curve. Where each gas's fit is supported, and where its resonances
lie, are material facts that optmat publishes; how a run uses the fit beyond
that support is solprop's modelling rule (§4.10), which never refuses a run
for the frequencies its field occupies: beyond a gas's data the response is
bounded and labelled. The labels differ by gas: CO2, CH4, SF6
and N2O have no curated two-photon limit, so everything beyond their data is
best effort; O3 absorbs throughout its range; and air reaches O2's two-photon
resonance at wavelengths shorter than about 305 nm.

**Demonstration cases.** Two cases, chosen because they sit on either side of
the material domain, run through qualification, the tutorial and the manual
(§7, §8):

- **DUV FWM in argon**: a 400 nm pump and an 800 nm seed in an argon-filled
  capillary generate 266.7 nm, $\omega_s=2\omega_{400}-\omega_{800}$, the
  phase-matched deep-ultraviolet source of Durfee et al. [9]. The generating
  quartet lies inside argon's fitted frequency sums; the signal's own Kerr and
  cross-phase quartets exceed them but stay within argon's leg bound and far
  from its resonances, so the rule of §4.10 admits the whole case.
- **VUV RDW in helium**: the M2 case of the [testing plan](../../testing.md) — a
  7.5 fs, 275 µJ, 800 nm pulse in a 125 µm-radius capillary with helium from
  0.8 bar to vacuum, emitting a resonant dispersive wave at 100–150 nm [10].
  The field extends below helium's two-photon resonance at a 120.3 nm
  fundamental, and such runs routinely reach below 80 nm. No off-resonant
  model is valid there. The dispersive evaluators still run it, with the
  legs bounded and that band labelled best effort (§4.10), so the comparison
  is a sensitivity study, not a qualified prediction (§4.7).

Raman and ionisation remain distinct physical responses, with their data
migration in SEP 3 and Raman algorithms in [SEP 1](../sep-0001-raman-response.md).
This proposal does not supply a new Raman dispersion model, nonperturbative
bound-electron model, ionisation law, or generic resonant complex susceptibility.
It must compose correctly with them. The helium TDSE response of
[SEP 12](sep-0012-tdse-nonlinear-response.md), now final, owns the
complete electronic response of its gas, including its cubic policy, and refuses
an accompanying `KerrResponse`; a dispersive evaluator is likewise refused beside
it for the same gas. Offering a dispersive cubic as one of the TDSE coupling's
own cubic policies is later work, not part of this proposal.
[SEP 2](../sep-0002-two-photon-absorption.md)
is relevant to nonlinear absorption; independently adding its loss term is not
automatically a causal completion of an arbitrary real dispersive Kerr law.

The recommendation is **two supported propagation options**, not a cheap model
with a polynomial used only in tests. Both remain explicit choices while their
useful domains and observable-specific accuracy are established.

### 3. Context

The current [nonlinear response chapter](../../../guide/media/nonlinear.md) derives the
four constant-coefficient cubics: real field and envelope, each with and without
the harmonic term. Relevant implementation boundaries are:

| Component | Present behavior and implication |
|---|---|
| `physdata/selection.py`, `MaterialSelection.electronic` | The electronic policy of a run; `ELECTRONIC_POLICIES` admits `"recommended"` and `"carrier-kerr"`, both the Kerr coefficient at the carrier |
| `physdata/nonlinear_data.py`, `OptmatKerr` and `CarrierKerr` | Adapts optmat's dispersive hyperpolarizability model and evaluates its Kerr process at the carrier: one number per species and run |
| `physdata/selection.py`, `MaterialManifest` | A run's material record: model identities, the `electronic` entries and the policies in force, which enter the resume key |
| `nonlinear/responses.py`, `KerrResponse` | A scalar `chi3` per unit density, with `thg` and a species label; `evaluate` is the reference cubic |
| `translate/responses.py`, `KerrCoefficients` and `_compile_kerr_components` | Scalar coefficients and density paths; compatible species can be summed before the cubic |
| `backends/numpy_ref/passes.py` | Oversampled time transforms, field synthesis, responses, projection and spectral prefactors |
| `backends/numba_cpu/backend.py`, `_respond_chunk` and `_project` | Fused time-domain response work and chunked projections; extra transforms can disturb this fusion |
| `backends/numpy_ref/backend.py`, `_nonlinear_general` | Frequency-dependent general modal maps and regional response assembly |
| solprop-kokkos, `src/solprop_kokkos/backend.py` and its compiled stage implementation | Host translation followed by resident backend work; new spectral response arrays and workspaces must cross the versioned seam explicitly |

These are paths in solprop's `src/solprop/` unless the package is named explicitly.
The [kernel contract](../../kernel-contract.md), [architecture](../../architecture.md)
and [backend policy](../../backends.md) govern the implementation. At the final
review the kernel schema is 23 and the output schema 15; an implementation
chooses the next available versions rather than reserving numbers in this Draft.

Two later changes bear on cost and qualification. `Multistep()` is now the
default integrator ([SEP 16](sep-0016-unified-multistep.md)), and its error norm
scores every bin of the run's
[active band](../../../guide/setup/grids.md#the-active-band-and-absorbing-collars).
A dispersive response changes the
adaptive trajectory as well as the cost of each nonlinear call, so whole-run
comparisons are made at equal achieved error with SEP 16's work–precision
measurements, not per step. The same active band — the grid's wavelength
limits with their collars — is the natural carrier of the band $B$ of §4.5.

The existing UPPE output factor contains optical frequency. In the appropriate
envelope limit this gives self-steepening. It does **not** supply the omitted
input-frequency dependence of the material susceptibility. Similarly, a
frequency-dependent effective area or modal overlap is a geometric effect:
it must compose with material dispersion, not be counted as its replacement.
See [GNLSE](../../../guide/waveguides/gnlse.md) and
[general modal propagation](../../../guide/waveguides/general-modal.md).

### 4. Theory

#### 4.1 One susceptibility, all mixing processes

Use the optics Fourier convention of [conventions](../../conventions.md),
$\widetilde E(\omega)=\int E(t)e^{+i\omega t}\,dt$. For a real, scalar field,
$\widetilde E(-\omega)=\widetilde E(\omega)^*$. Write
$\omega_3=\omega_o-\omega_1-\omega_2$. The third-order polarization is

$$
\widetilde P^{(3)}(\omega_o)=\frac{\epsilon_0}{(2\pi)^2}
\iint\chi^{(3)}(-\omega_o;\omega_1,\omega_2,\omega_3)
\widetilde E(\omega_1)\widetilde E(\omega_2)
\widetilde E(\omega_3)\,d\omega_1\,d\omega_2.
$$

For vector fields, replace the scalar susceptibility by $\chi_{ijkl}^{(3)}$
and contract the three field components. This scalar derivation does not
determine those independent tensor functions.

The names describe slices of the same integral:

| Contribution | Example inputs and output |
|---|---|
| SPM | $(\omega,\omega,-\omega)\to\omega$ |
| XPM / pump-probe | $(\omega_r,\omega_p,-\omega_p)\to\omega_r$ |
| Degenerate-pump FWM | $(\omega_p,\omega_p,-\omega_i)\to\omega_s=2\omega_p-\omega_i$ |
| THG | $(\omega,\omega,\omega)\to3\omega$ |

All permutations and inverse exchanges belong to the same response. An FFT
product automatically sums them with their multiplicities. Different mixing
triplets can have the same output frequency: an output-only coefficient cannot
distinguish them.

A constant $\chi_c$ leaves the convolution as
$P^{(3)}(t)=\epsilon_0\chi_c E(t)^3$. A generic unrestricted discretization
requires two input sums for each output, approximately $O(N_\omega^3)$ work
for the whole polarization and potentially $O(N_\omega^3)$ stored kernel
values. This is not the proposed implementation. A finite separable expansion
of rank $R$ needs filtered fields, products and transforms, with work scaling
as $O(RN_t\log N_t)$ and workspaces that can be reused between terms.

#### 4.2 Material normalization and the carrier approximation

For a gas, let $N$ be number density in m$^{-3}$ and let $\gamma_T$ be the
Taylor-convention second hyperpolarizability in C m$^4$ V$^{-3}$. Then

$$
P^{(3)}=\frac{N\gamma_T}{6}E^3,
\qquad \chi^{(3)}=\frac{N\gamma_T}{6\epsilon_0},
\qquad \gamma_3=\frac{\gamma_T}{6\epsilon_0}.
$$

The last quantity is solprop's per-particle coefficient in m$^5$/V$^2$;
$\chi^{(3)}$ is in m$^2$/V$^2$. The factor of six follows from the molecular
dipole Taylor expansion [1, 2]. For a transparent bulk scalar material with
$E=\mathrm{Re}(A e^{-i\omega t})$ and intensity $I=n_0\epsilon_0 c|A|^2/2$
([conventions](../../conventions.md)),
$n_2=3\chi_K^{(3)}/(4n_0^2\epsilon_0 c)$; for a dilute gas ($n_0=1$) this is
optmat's $n_2=N\gamma_K/(8\epsilon_0^2c)$ [1]. Normalize a factorization using
the electronic susceptibility curve, not the fourth root of $n_2$ alone when
$n_0$ varies. SEP 3 governs the dilute-gas and local-field approximations.

Define $\chi_K(\omega)=\chi^{(3)}(-\omega;\omega,\omega,-\omega)$.
The carrier approximation uses $\chi_c=\chi_K(\omega_0)$ everywhere. It is
the zeroth-order approximation over the participating frequency region, even
when the field itself is represented without a narrowband expansion.

Two separate errors matter: freezing the diagonal Kerr curve, and replacing
off-diagonal mixing by that diagonal value. A narrow pump can have a small
off-diagonal error for one distant FWM quartet while the Kerr coefficient of
the generated band differs much more. No single bandwidth percentage is a
universal error estimate.

#### 4.3 Why an output-only curve is inadequate

The inexpensive expression

$$
\widetilde P_{\rm out}(\omega)
=\epsilon_0\chi_K(\omega)\mathcal F[E^3](../\omega)
$$

matches monochromatic SPM, but assigns the signal's *degenerate* Kerr
coefficient to every FWM pathway producing that signal. It need not improve
upon the pump's constant coefficient; §4.7 gives a concrete counterexample.

It also generally violates the exchange symmetry of the lossless scalar
quartet: the coefficient changes when a different member is treated as the
output. An apparent invariant obtained by reweighting energy with $\chi_K$
would not be solprop's physical energy. This is distinct from deriving a
different field normalization and consistently transforming every operator.
Do not implement an output-only material multiplier as the dispersive option.

#### 4.4 Symmetric factorization from nonlinear data

In a transparent domain with a real, positive electronic Kerr curve, choose
a positive reference $\chi_*$ and a real even multiplier

$$
g(\omega)=\left[\frac{\chi_K(|\omega|)}{\chi_*}\right]^{1/4}.
$$

Use the same function on all four legs:

$$
\chi_{\rm sym}^{(3)}(-\omega_o;\omega_1,\omega_2,\omega_3)
=\chi_*g(\omega_o)g(\omega_1)g(\omega_2)g(\omega_3),
\qquad
P_{\rm sym}^{(3)}=\epsilon_0\chi_*G[(GE)^3].
$$

Here $G=\mathcal F^{-1}g\mathcal F$. The input field is filtered, cubed,
and the polarization is filtered once more. The result is still one broadband
polarization. SPM, FWM and THG acquire their coefficients through the same
four-frequency rule. The positive reference cancels out of the final kernel;
using a static or carrier reference changes conditioning, not the model.

For a gas use the per-particle curve and apply $N$ once outside the operator.
Do not hide changing density in four field multipliers. A constant positive
curve gives the original cubic exactly. A zero, negative or complex curve
requires another factorization and is outside this first model's domain;
taking an arbitrary complex fourth root is not a proposed extension.

Matching every degenerate Kerr value does **not** determine the general
susceptibility. This is a separability assumption, with the full permutation
symmetry appropriate to the specified lossless scalar approximation. It is
not a claim of frequency-independent tensor ratios or of universal Kleinman
symmetry near resonances.

Generalized Miller formulas produce a related product of *linear*
susceptibilities under anharmonic-oscillator assumptions [3]. Multiple
oscillators and their nonlinear couplings need their own treatment; a measured
linear index does not uniquely fix nonlinear dispersion. Gas measurements
show that simple linear-dispersion estimates can substantially underestimate
nonlinear dispersion [4, 5]. The proposed $g$ is therefore fitted to or derived
from curated **nonlinear** data. Miller scaling is an alternative model to
qualify, not a substitute for that curation.

#### 4.5 Full gas polynomial as a filtered cubic sum

Use signed vacuum wavenumber $\nu_j=\omega_j/(2\pi c)$ in m$^{-1}$, with
$j=o,1,2,3$, and define

$$
S_2=\sum_j\nu_j^2,\qquad S_4=\sum_j\nu_j^4,
\qquad
\gamma_T=\gamma_0[1+A S_2+B S_2^2+B' S_4].
$$

$A$ has units m$^2$ and $B,B'$ have units m$^4$. The curated tabulations
record multipliers of $10^{-10}$ cm$^2$ and $10^{-20}$ cm$^4$, which become
$10^{-14}$ m$^2$ and $10^{-28}$ m$^4$, respectively. $\gamma_0$ must also
be converted from its recorded units to SI before translation. optmat names
the same two sums $W_2$ and $W_4$ and evaluates them in cm$^{-1}$ [1];
$S_2$ and $S_4$ here are those sums in SI, $S_2=10^4\,W_2$ and
$S_4=10^8\,W_4$ numerically.

The expansion follows the low-frequency susceptibility analysis of Bishop
and De Kee [6]; their additive coefficients must not be confused with the
coefficients normalized by $\gamma_0$ here. optmat supplies the fit and its
uncertainties [1]. The scalar gas fits are not a complete tensor reconstruction:
the primary derivation treats diagonal components, and the molecular mean
requires the curation's additional assumptions. Calculations of tensor
components show why their distinct dispersion cannot be inferred from one
scalar function [7].

Define the spectral derivative operator

$$
D=-\frac{1}{(2\pi c)^2}\partial_t^2,
\qquad \mathcal F[Df](../\omega)=d(\omega)\widetilde f(\omega),
\qquad d(\omega)=\left(\frac{\omega}{2\pi c}\right)^2.
$$

The output-frequency factors differentiate a product; input-frequency
factors differentiate its individual fields. Expanding the symmetric sums
and counting equal input permutations gives

$$
\begin{aligned}
\frac{6P_{\rm poly}^{(3)}}{N\gamma_0}
={}&E^3+A\{D(E^3)+3E^2DE\}\\
&+B\{D^2(E^3)+3E^2D^2E+6D(E^2DE)+6E(DE)^2\}\\
&+B'\{D^2(E^3)+3E^2D^2E\}.
\end{aligned}
$$

For example, the $6E(DE)^2$ comes from twice the three cross-products
between squared input frequencies in $S_2^2$. Grouping output multipliers
gives an efficient equivalent:

$$
\begin{aligned}
\frac{6\widetilde P_{\rm poly}^{(3)}}{N\gamma_0}
={}&[1+Ad+(B+B')d^2]\mathcal F[E^3]\\
&+[3A+6Bd]\mathcal F[E^2DE]\\
&+\mathcal F[3(B+B')E^2D^2E+6BE(DE)^2].
\end{aligned}
$$

This uses three inverse transforms to obtain $E,DE,D^2E$, and three forward
transforms of product combinations. There is no three-dimensional frequency
table. An $A$-only truncation uses two of each. A general low-rank representation
could later extend the same architecture, but is not required for this polynomial.

This is exact **evaluation of the polynomial on the resolved, dealiased grid**.
The polynomial is itself a truncated off-resonant material model. Its growth
at large frequency is not evidence of a physical divergence or a valid VUV
response. Do not use finite differences for $D$ or extrapolate the polynomial
onto arbitrary padded FFT frequencies.

The cut-off must be symmetric. The oversampled fine grid reaches three times
the coarse band and the cubic's products land there, so the weights outside
the admitted domain need a stated value. Zeroing the *output* multipliers
alone destroys the four-leg symmetry of the kernel, and with it the lossless
identity of §4.9. The consistent rule is one band projection $\Pi_B$ applied
identically to the field entering every product and to the polarization
leaving it: the restricted kernel
$\Pi_B(\omega_o)\Pi_B(\omega_1)\Pi_B(\omega_2)\Pi_B(\omega_3)\chi^{(3)}$ is
still fully symmetric. The same rule applies to the symmetric factorization,
whose $g$ is likewise evaluated only inside $B$. Which band $B$ is admitted is
the domain question of §5 and Q2; the projection is the numerical
consequence, and §7 tests it.

Two bands must not be confused. $B$ is a numerical band: the run's active band,
outside which the field is negligible and the projection costs nothing
physical. The **material support** is where the fit is qualified. Inside $B$
but outside the support, §4.10's rule bounds the legs or refuses. Shrinking
$B$ to the support instead would silently remove the nonlinear coupling of
occupied bins — the generated signal in both demonstration cases of §2 — and is
not an admissible substitute for that rule.

#### 4.6 Why the symmetric approximation can work well

Normalize by $\gamma_0$. On the Kerr diagonal,

$$
\frac{\gamma_K(\nu)}{\gamma_0}
=1+4A\nu^2+(16B+4B')\nu^4.
$$

Expanding its fourth root,
$g(\nu)=1+A\nu^2+(4B+B'-3A^2/2)\nu^4+O(\nu^6)$.
Multiplication of four such factors gives

$$
\frac{\gamma_{\rm sym}-\gamma_{\rm poly}}{\gamma_0}
=\left(B-\frac{A^2}{2}\right)(4S_4-S_2^2)+O(\nu^6).
$$

Thus the factorization agrees through second order for all quartets and
exactly on the Kerr diagonal. The leading error measures the spread of the
four squared frequencies, since
$4S_4-S_2^2=\sum_{j<k}(\nu_j^2-\nu_k^2)^2$. The exact fourth-root construction
also contains higher-order terms; this expansion does not bound its UV error.

Another diagnostic is the quartet's effective frequency
$\nu_{\rm eff}=\sqrt{S_2/4}$. Evaluating $\gamma_K(\nu_{\rm eff})$ differs
from the polynomial only by $\gamma_0 B'(S_2^2/4-S_4)$. This can be very
accurate, but $\nu_{\rm eff}$ belongs to the **whole quartet**. There is no
unique effective frequency to assign to an output bin receiving many triplets.
It is a useful interpretation of the polynomial, not a justification for an
output-only multiplication.

#### 4.7 Numerical scale of the approximations

The following calculations use the optmat fits [1], at the audited revision,
of all fifteen gases in scope (§2). Ratios cancel density and the
Taylor normalization. They
measure disagreement between models, not experimental accuracy. No literature
coefficient table is copied into solprop by this proposal.

First, the variation of the degenerate curve itself is substantial. The
third column onward compares nondegenerate quartets as described below.

| Gas | $\gamma_K(400\,\mathrm{nm})/\gamma_K(2000\,\mathrm{nm})$ | Symmetric median (%) | Symmetric 95th pct. (%) | Symmetric max. (%) | Output-only median (%) | Output-only 95th pct. (%) | Output-only max. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| He | 1.1195 | 0.006 | 0.063 | 0.124 | 0.914 | 4.622 | 6.907 |
| Ne | 1.1498 | 0.007 | 0.079 | 0.153 | 1.128 | 5.745 | 8.621 |
| Ar | 1.3708 | 0.053 | 0.494 | 0.913 | 2.432 | 13.700 | 21.206 |
| Kr | 1.5414 | 0.115 | 0.966 | 1.714 | 3.197 | 19.573 | 30.897 |
| Xe | 1.8673 | 0.260 | 1.921 | 3.212 | 4.467 | 30.224 | 48.560 |
| H2 | 1.4198 | 0.047 | 0.401 | 0.718 | 2.765 | 15.230 | 23.493 |
| D2 | 1.4114 | 0.046 | 0.399 | 0.717 | 2.718 | 14.957 | 23.058 |
| N2 | 1.3414 | 0.047 | 0.450 | 0.837 | 2.275 | 12.689 | 19.584 |
| O2 | 1.5476 | 0.183 | 1.611 | 2.908 | 2.942 | 20.357 | 32.781 |
| CH4 | 1.6282 | 0.133 | 1.057 | 1.831 | 3.622 | 22.320 | 35.321 |
| SF6 | 1.2516 | 0.105 | 1.112 | 2.141 | 1.371 | 10.120 | 16.394 |
| N2O | 1.4441 | 0.048 | 0.378 | 0.656 | 2.175 | 10.798 | 18.890 |
| CO2 | 1.4428 | 0.082 | 0.728 | 1.325 | 2.758 | 16.243 | 25.408 |
| CO | 1.4233 | 0.019 | 0.133 | 0.208 | 2.895 | 15.120 | 23.054 |
| O3 | 1.0749 | 0.036 | 0.193 | 0.360 | 0.829 | 3.172 | 4.493 |

**Reproduction and interpretation.** NumPy `default_rng(4)` draws a
`(300000, 3)` array of independent wavelengths uniformly on 400–2000 nm.
Convert to SI vacuum wavenumbers and form
$\nu_o=\nu_1+\nu_2-\nu_3$. Keep positive outputs in the same wavelength
interval and $S_2$ no larger than that gas's largest fitted value, with no
10% warning margin. This leaves 208109 quartets for each gas except N2O
(175846) and O3. O3's fit is declared only from 700 nm, so its row draws and
keeps wavelengths on 700–2000 nm instead (204331 quartets), and its diagonal
ratio is $\gamma_K(700\,\mathrm{nm})/\gamma_K(2000\,\mathrm{nm})$. For
each retained quartet compute the polynomial, the geometric mean of the four
degenerate Kerr values, and the Kerr value at the output. Errors are
$100|\gamma_{\rm candidate}/\gamma_{\rm poly}-1|$; percentiles use NumPy's
default linear interpolation. Python 3.14.5 and NumPy 2.5.2 were used; the
final review reproduced every earlier row exactly from the stored v0.3.0 fits
and added the CO2, CO and O3 rows by the same procedure. The diagonal ratio
column is a separate curve evaluation: in particular, N2O's 400 nm value exceeds
its fitted-$S_2$ support and rests on borrowed
dispersion assumptions [1]. The quartet columns enforce the stated cutoff.

This is neither a uniform-frequency sample nor a pulse-weighted distribution,
and the sampled maximum is not a worst-case bound. Fitted-$S_2$ admission alone
does not certify a previously unmeasured process. A map of fit uncertainty and
primary-data support is needed alongside these comparisons.

For selected degenerate-pump FWM quartets, define
$\omega_s=2\omega_p-\omega_i$. The error columns below are signed relative
to the polynomial. The ratio column shows how much the actual quartet
coefficient differs from the pump's degenerate Kerr value.

| Pump / idler / signal (nm) | Gas | $\gamma_{\rm FWM}/\gamma_K(\omega_p)$ | Output-only error (%) | Symmetric error (%) |
|---|---|---:|---:|---:|
| 800 / 2000 / 500 | He | 1.0052 | +4.130 | +0.028 |
| 800 / 2000 / 500 | Ar | 1.0139 | +12.005 | +0.237 |
| 800 / 2000 / 500 | Xe | 1.0266 | +25.777 | +1.023 |
| 800 / 2000 / 500 | N2 | 1.0130 | +11.134 | +0.213 |
| 1030 / 2000 / 693.603 | Ar | 1.0053 | +5.200 | +0.066 |
| 1030 / 2000 / 693.603 | Xe | 1.0094 | +10.045 | +0.325 |

All frequencies in these rows lie in the declared fundamental-wavelength
window, and their $S_2$ values are below fitted maxima. For the argon 800 nm
case, the frozen pump coefficient is only about 1.37% low for this quartet,
whereas substituting the signal's Kerr coefficient makes it 12% high.
This directly answers why a frequency-dependent output coefficient can be
worse than the constant approximation.

Two tempting extensions illustrate the validity trap:

| Pump / idler / signal (nm) | Formal calculation | Why it is not qualified |
|---|---|---|
| 515 / 2000 / 295.552 | Ar: output-only +46.595%, symmetric +0.951% | The signal's degenerate curve, needed by both constructions, is below its declared 400 nm range; a small symmetric error against the polynomial is not a physical validation |
| 343 / 1030 / 205.760 | Ar: $S_2/S_{2,\max}=1.453$; Xe: 1.650 | The pump itself is below the declared range and these quartets also exceed fitted frequency sums |

The first row does not prove the polynomial is invalid solely because its
output is below 400 nm: the published process range is a **fundamental** range,
not a universal cutoff on every output. It proves that a scalar wavelength
window and one $S_2$ threshold are insufficient descriptions of arbitrary
quartet validity. A factorization sampling a degenerate curve at each leg has
different data needs from a process fit evaluated at its fundamental.

**The demonstration cases.** The same comparison for the two cases of §2, with
the carrier coefficient $\gamma_K(\omega_c)$ added. Errors are signed relative
to the polynomial. $S_2/S_{2,\max}$ measures the quartet against the fitted
sums; a ratio above one is beyond the data. The last two columns anticipate
§4.10: the polynomial under its leg bound, as a ratio to the unbounded
polynomial, and whether the quartet's frequencies lie inside the gas's band
edge.

| Quartet (nm) | Gas | $\gamma_{\rm poly}/\gamma_K(800\,\mathrm{nm})$ | Carrier error (%) | Symmetric error (%) | Output-only error (%) | $S_2/S_{2,\max}$ | Leg-bounded / polynomial | Inside band edge |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 400 + 400 − 800 → 266.7 | Ar | 1.345 | −4.25 at 400; −25.65 at 800 | +0.556 | +50.00 | 0.98 | 1 | yes (189.4 nm) |
| SPM of 266.7 | Ar | 2.018 | −36.17 at 400; −50.44 at 800 | 0 | 0 | 1.97 | 1 | yes |
| XPM of 266.7 by 400 | Ar | 1.612 | −20.13 at 400; −37.98 at 800 | −0.025 | +25.13 | 1.42 | 1 | yes |
| THG of 800 → 266.7 | He | 1.060 | −5.66 at 800 | +0.311 | +20.60 | 0.32 | 1 | yes (120.3 nm) |
| 250 + 250 − 1500 → 136.4 | He | 1.485 | −32.68 at 800 | −0.012 | +88.95 | 1.49 | 0.949 | yes |
| 200 + 200 − 600 → 120.0 | He | 1.785 | −43.96 at 800 | −1.229 | +106.15 | 2.11 | 0.878 | no, 120.0 < 120.3 nm |
| XPM of 120 by 800 | He | 1.963 | −49.06 at 800 | −2.284 | +87.43 | 2.46 | 0.771 | no |
| SPM of 120 | He | 3.679 | −72.82 at 800 | 0 | 0 | 4.81 | 0.628 | no; its pair reaches 1.002 $E_2$ |
| SPM of 80 | He | 11.27 | −91.13 at 800 | 0 | 0 | 10.81 | 0.205 | no; past $E_2$ |

In **argon** the generating quartet is inside the fitted frequency sums. The
pump's own Kerr coefficient is only 4% low for it, whereas a carrier at the
800 nm seed is 26% low; the output-only construction is 50% high, and the
symmetric model is within 0.6% of the polynomial. The signal's own SPM and
cross-phase quartets exceed the fitted sums, but every leg stays within
argon's leg bound (215.9 nm), so the bounded polynomial is the polynomial
itself, a lower bound on $\gamma$ there (§4.10). Every frequency is far inside
argon's band edge: two 266.7 nm photons carry 9.3 eV, below its first excited
states, whose resonance lines are at 104.8 and 106.7 nm [11]. The whole case
is admitted by default.

In **helium** the M2 case's harmonic sector is inside the fitted sums, and
the fit is exact for legs down to the leg bound (151.9 nm, the third
harmonic of Bishop and Pipin's 455.6 nm calculation [12]). But the RDW band,
and a run's field more generally, reaches past helium's band edge: the degenerate two-photon
$1s^2\to1s2s\,{}^1S$ resonance at a 120.3 nm fundamental. Helium's other
low-order resonances follow from NIST's levels [11, 12]: three-photon
$1s2p\,{}^1P$ and $1s3p\,{}^1P$ resonances at 175.3 and 161.1 nm, and
three-photon ionisation below 151.3 nm, which agree with
[SEP 12 §4.4](sep-0012-tdse-nonlinear-response.md). A field with
frequencies beyond the edge has an SPM quartet past the resonance, which no
resonance-free expansion represents. The dispersive evaluators run the case
anyway, with each leg held at the leg bound (§4.10) and the band beyond the
edge recorded as best effort. The comparison measures how sensitive the RDW's
wavelength and energy are to the treatment of its VUV legs: held at the 800 nm
value by the carrier model, or raised along the fit to the leg bound and held
there by the dispersive ones. It is reported as that sensitivity, never as a
qualified prediction. A resonant helium response, such as SEP 12's TDSE
coupling, is the physical reference there.

**Pump dynamics and VUV emission in one response.** An RDW is emitted through
four-wave mixing: three components of the broadened soliton produce a VUV
photon. Neither dispersive evaluator gives a band one coefficient, so modelling
the emission does not cost accuracy at the pump. The soliton's own quartets —
SPM at 800 nm, and its broadened wings down to about 250 nm at maximum
compression — lie inside helium's fitted data and are evaluated exactly. The
SPM quartet at 800 nm keeps $\gamma_K(800\,\mathrm{nm})$, exactly the carrier
value. A generating quartet such as 250 + 250 − 1500 → 136.4 nm takes a
coefficient set by all four of its legs: 1.41 times the 800 nm value under the
leg bound (1.49 unbounded), where the carrier model is about 30% low. In the
symmetric model each leg contributes $\gamma_K^{1/4}$, so the single VUV leg
enters only as a fourth root. The largest coefficients belong to quartets with
two or more VUV legs — cross-phase modulation of the RDW by the pump (1.51),
its own SPM (2.31) — and they scale with the weak RDW field, so they hardly
act back on the pump. The RDW's wavelength is set by phase matching to the
soliton's nonlinear wavenumber, which comes from the exactly evaluated SPM, so
it is expected to move little between the models. The conversion efficiency,
and the RDW's evolution after emission, are what the comparison tests. This
is the output-only construction's failure (§4.3) avoided: that construction
would give every quartet producing a VUV photon the VUV leg's degenerate value.

These rows reuse the §4.7 procedure on the stated quartets. The quartets with
a VUV leg are illustrative members of a broad soliton spectrum, not a
weighting of it.

#### 4.8 Material validity, causality and missing data

The fits of fourteen of the fifteen gases in scope declare fundamental
wavelengths from 400 nm to 20 µm [1]; O3's declares 700 nm. Their fitted
frequency sums and independent evidence differ
by gas; optmat's 10% warning margin is a software warning policy, not an
accuracy bound. 343 nm pumping is already outside that declared interval.
“VUV–IR” is the desired application range, not this Draft's qualified material
range.

**The harmonic sector has a different domain in each evaluator.** When the
third harmonic is retained, the polynomial evaluates the THG quartet
$(\nu,\nu,\nu)\to3\nu$ at $S_2=12\nu^2$, which stays within the fitted
frequency sums down to the pump wavelengths below [1]. The symmetric
factorization instead samples the degenerate curve on the output leg at
$3\nu$, so it needs $\gamma_K$ at one third of the pump wavelength, which is
inside the declared 400 nm range only for pumps of 1200 nm or longer, and
inside the fitted Kerr sums, $4(3\nu)^2\le S_{2,\max}$, only for the pumps of
the third column.

| Gas | Shortest pump whose THG quartet is within fitted $S_2$ (nm) | Shortest pump whose harmonic leg is within fitted Kerr $S_2$ (nm) | Shortest pump whose harmonic leg is within the declared Kerr range (nm) |
|---|---:|---:|---:|
| He | 456 | 789 | 1200 |
| Ne, Ar, H2, N2, SF6 | 648 | 1122 | 1200 |
| D2, O2, CH4 | 674 | 1167 | 1200 |
| Kr, Xe, CO2 | 690 | 1196 | 1200 |
| CO | 695 | 1203 | 1200 |
| N2O | 1150 | 1992 | 1200 |
| O3 | 1203 (resonant; optmat gives no THG) | 2084 | 2100 |

The second and third columns are Kerr-diagonal criteria, which §4.10 shows
are too strict: the data reach harmonic legs well beyond the Kerr diagonal of
the fitted sums. Under §4.10's leg bound both evaluators share the first
column, since $3\nu\le\nu_c$ exactly when $12\nu^2\le S_{2,\max}$. For
pumps at or above it the harmonic sector is evaluated without bounding; the
symmetric model's $g$ then samples the Kerr curve up to $3S_{2,\max}$, a
polynomial extrapolation of the diagonal whose departure from the polynomial
is measured in Phase 0. For shorter pumps the harmonic legs are bounded. O3's
third harmonic falls in its Hartley and Huggins absorption bands for
fundamentals up to about 1050 nm [1], so its harmonic sector is resonant
whatever the evaluator. The harmonic-sector decision of §4.9 and Q4 still
precedes acceptance.

The relevant gaps are:

| Gap | Needed before the corresponding claim |
|---|---|
| Arbitrary-quartet validity | Largely closed by optmat's leg support and resonance limits (§4.10); its public record (OEP 5) must be released, and polyatomic two-photon limits curated |
| UV/VUV electronic response | Curated measurements or calculations of nondegenerate response, truncation error and, where needed, its imaginary part |
| Nonlinear absorption and causality | Compatible dissipative/resonant terms; a real polynomial and an independently chosen loss fit are not automatically one causal response |
| Tensor dispersion | Independently supported isotropic tensor components or a named scalar-times-tensor approximation |
| Molecular electronic separation | Preserve the residual vibrational contamination and borrowed-prior qualifications of the curated fits; do not double count them as new exact electronic data |
| Silica | A coherent electronic/Raman split and evidence for off-diagonal susceptibility, beyond a degenerate $n_2(\omega)$ curve |

A real even $g$ gives a symmetric, self-adjoint band operator. It is not a
globally causal absorbing material law. The polynomial is likewise a
low-frequency expansion, not a resonant susceptibility. Near resonances, a
causal pole/auxiliary-oscillator model or a more general symmetry-constrained
low-rank response may be appropriate. Those require additional data and a
separate derivation; neither scalar extension here promises them.

Optmat's gas fits carry different absolute-scale, statistical and systematic
uncertainties [1]. For example, several molecular fits retain inseparable
vibrational contributions, and N2O's dispersion is constrained partly by
priors. Being exact relative to those fits does not remove those uncertainties.
The silica Kerr curve is useful input for a candidate symmetric approximation,
but does not uniquely supply its nondegenerate electronic response. Do not
manufacture silica polynomial coefficients from that curve alone.

#### 4.9 Lossless consistency and the meaning of THG selection

For a periodic or decaying real field and a real, even, self-adjoint $G$,
which commutes with $\partial_t$, the proposed symmetric response obeys

$$
\int (\partial_t E)P_{\rm sym}\,dt
=\epsilon_0\chi_*\int \partial_t(GE)(GE)^3\,dt=0.
$$

By integration by parts this also gives zero net $\int E\partial_tP\,dt$.
It is a necessary lossless-work identity. The polynomial obeys the same
identity because all four frequency legs enter symmetrically and their signed
sum is zero. A test must use dealiased products and compatible projection.
Temporal windows and absorbers can introduce physical or numerical losses;
disable them for this test and account for them separately in a run.

An independent discrete convolution check used 128 samples, Fourier bins
$\pm3,\pm7,\pm11$, Hermitian random amplitudes and dimensionless
$d=(k/10)^2$, $A=0.05$, $B=0.02$, $B'=-0.003$. The polynomial FFT expression
agreed with the explicit 216 input-triplet sum at relative $L^2$ error
$4.3\times10^{-15}$. The normalized work residual was below
$7\times10^{-17}$ for the constant, symmetric and polynomial forms, versus
about $1.5\times10^{-3}$ for the output-only form. These synthetic coefficients
check algebra; they are not material recommendations or test tolerances.

**Broadband process selection requires another careful distinction.** Let
$u=E_a$ be the optical analytic signal, so $E=(u+u^*)/2$, and let $\Pi_+$
retain strictly positive optical frequencies. Ignoring a DC component, the
analytic signal of the full cubic is

$$
2\Pi_+[E^3]=\frac14\Pi_+[u^3+3u^2u^*+3u(u^*)^2].
$$

The last term can have positive-frequency support for a sufficiently broad
field. It must not be discarded merely because a conventional narrowband
envelope derivation would put it outside the band. Analytic-signal formulations
make this issue explicit [8]. The same projection must be derived after the
electronic input filters, or term by term for the polynomial.

The existing field-grid `thg=False` rule is
$3|E_a|^2E/4$, whereas the envelope rule keeps $3|A|^2A/4$.
Their documented narrowband parity does not establish arbitrary-band parity
or lossless exchange. A direct diagnostic with dimensionless
$E(t)=\cos t+0.2\sin3t$ gives
$\langle E' E^3\rangle=0$ but
$\langle E'(3|E_a|^2E/4)\rangle=-0.075$.
Evaluating the current `KerrResponse` on 128 equally spaced samples reproduces
those values to roundoff after removing its constant multiplier. The partial
cubic retains part of inverse harmonic exchange without its forward partner.
This is a qualification gap in the existing switch when those bands are
occupied, not a reason to change unrelated operators in this documentation PR.

For the new dispersive response, recommend defining harmonic exclusion by
retaining the two-positive/two-negative **quartets**, consistently for every
output. In analytic variables the scalar number-conserving cubic is then
$\Pi_+[3|u|^2u/4]$; apply the corresponding symmetric kernel to each quartet.
The complete response retains all allowed quartets, including harmonic and
inverse harmonic terms. There is still no runtime SPM/FWM classification.
This choice and its relationship to the existing `thg` interface must be
settled before acceptance (Q4). Until derived and qualified, a backend refuses
the unsupported new response/grid/THG combination.

For full lossless electronic propagation test physical energy, but do not
assert conservation of total photon number across THG. For an explicitly
number-conserving four-wave approximation test its photon invariant as well.
Raman and plasma require their own energy-transfer accounting. The discrete
UPPE normalization and modal projections must preserve the relevant continuum
identity; the local work identity alone does not prove an entire solver correct.

#### 4.10 Beyond the fitted data: a leg bound, never a refusal

A run's field routinely extends far beyond any gas's data: helium RDW grids
reach below 80 nm. The carrier model runs there today with no domain at all, a
constant value extrapolated from the carrier to every quartet. The rule for the
dispersive evaluators is therefore: **never be less physically sound than the
carrier, and never refuse a run for the frequencies its field occupies.**
optmat's analysis of where its fits are supported [12] gives three facts that
make such a rule possible.

- **The legs of the data.** The four signed legs of a quartet sum to zero, so
  by the Cauchy–Schwarz inequality $\nu_j^2\le\tfrac34 S_2$, with equality for
  the third-harmonic output. No quartet within the fitted sums has a leg
  beyond $\nu_c=\sqrt{3S_{2,\max}/4}$: 151.9 nm for He; 215.9 nm for Ne, Ar,
  H2, N2 and SF6; 224.7 nm for D2, O2 and CH4; 230.1 nm for Kr, Xe and CO2;
  231.5 nm for CO; 383.4 nm for N2O; 401.0 nm for O3. The largest fitted
  single legs are harmonic legs at 229–244 nm (151.9 nm for He), 1.6–1.7
  times further out in wavenumber than the Kerr diagonal of the fitted sums.
- **The resonances.** In the sum over states a single leg reaches the lowest
  dipole-coupled level $E_1$, and a pair sum $|\nu_a+\nu_b|$ the lowest level
  two dipole steps away, $E_2$. Every frequency $\nu$ of a field has an SPM
  quartet whose pair sum is $2\nu$, so a field stays clear of both only while
  it occupies wavelengths longer than the **band edge**, $\min(E_1, E_2/2)$:
  He 120.3 nm; Ne 133.5 nm; Ar 189.4 nm; Kr 216.7 nm; Xe 256.0 nm; H2
  201.7 nm; D2 201.1 nm; N2 290.1 nm; O2 304.6 nm; CO 308.9 nm. These are for a
  field of one linear polarisation. A vector field also reaches the even
  $J=1$ levels of the atoms and CO's $\Sigma^-$ state, which move the edges of
  Ne, Ar, Kr, Xe and CO by 1–3 nm. No two-photon state of CO2, CH4, SF6 or N2O
  is curated, and O3 absorbs throughout its range, its Chappuis and Wulf bands
  reaching 1047 nm.
- **Monotonicity.** Every fit has $A>0$, $B\ge0$ and $B+B'>0$, so it rises
  with each $\nu_j^2$ at least as fast as $\gamma_0A$, inside its data and
  beyond. No theorem makes physical $\gamma$ do the same [2], but every
  curated calculation rises toward the first resonance, and faster than the
  polynomial: beyond the data the polynomial is low by 0.7–3.6% at about twice
  $S_{2,\max}$ for Ne, Ar and N2, about 5% for H2 and D2, and 28% for helium's
  THG at four times it. The rise fails above a two-photon resonance: xenon's
  $n_2$ changes sign at 248.4 nm.

**The rule.** The rule is solprop's modelling decision, not a material
property: optmat supplies the facts it rests on. Bound every leg of both
evaluators at the gas's $\nu_c$:
$\nu_j\to\min(|\nu_j|,\nu_c)$. For the polynomial this replaces the multiplier
$d(\omega)$ of §4.5 by $\min(d(\omega),\nu_c^2)$; for the symmetric model it
evaluates $g$ at $\min(|\nu|,\nu_c)$. Both remain real, even and identical on
all four legs, so the lossless identity of §4.9 and the FFT cost are
unchanged. The bound divides every run's band into three regions, which are
recorded with the run and can be marked on its outputs:

| Region | Where | What the evaluators give | Standing |
|---|---|---|---|
| Supported | Every quartet within $S_{2,\max}$, and every quartet whose legs are all within $\nu_c$ | The exact model: the bound never touches a quartet within the fitted sums | Exact within the fit; beyond $S_{2,\max}$ a polynomial extrapolation that is a lower bound below resonance |
| Bounded | Legs beyond $\nu_c$, field inside the band edge | Each such leg held at $\nu_c$ | A lower bound on $\gamma$. Against carriers at 400, 800 and 1030 nm it is never worse within the data, and is shown no worse beyond them for every quartet except a few dozen Xe and CO quartets that mix legs either side of a 400 nm carrier, and N2O against a 400 nm carrier, itself beyond N2O's single fitted value |
| Best effort | Field beyond the band edge, or beyond the data of a gas with no curated two-photon limit | The same bounded evaluation | No off-resonant model is valid here. The response is held at the gas's value at the edge of its data: the constant extrapolation the carrier model makes everywhere, anchored at the last supported leg instead of at the carrier, and never the polynomial's $\nu^4$ growth. Recorded and labelled, never refused |

The helium SPM quartet at 80 nm shows the difference: the raw polynomial gives
11.3 times the carrier's coefficient, the bounded evaluation 2.31 times
(§4.7). Refusing the best-effort region would push such runs back onto the
carrier model. That model is worse wherever the bound is shown better, and no
better founded beyond the edge. A resonant response — SEP 12's TDSE coupling
for helium, or a future resonant model — is the physical route there.

Three other bounds were evaluated and rejected [12]:

- **The Kerr diagonal of the fitted sums.** It changes up to 27 fitted values
  of a gas, by up to −26%, and is worse than a 400 nm carrier for 11–16% of the
  quartets within the fitted sums: −9.3% for the argon DUV quartet of §4.7,
  against the carrier's −4.3%.
- **The largest fitted leg.** It is exact on the data but moves quartets
  within the fitted sums by up to −5.4% (Xe).
- **The band edge as a leg bound.** It cuts the data of Xe, N2, O2 and CO. A
  pair resonance is reached by two legs together, so it is not a per-leg limit
  at all.

The raw polynomial beyond the data is not offered: its $\nu^4$ growth through
the resonances is not a physical picture.

**Ownership.** optmat states the material facts as well as it can, in the
support record of each hyperpolarizability model proposed by its OEP 5 [13]:
the largest fitted $S_2$ and the legs of the fitted data, the levels $E_1$ and
$E_2$ for one and for any polarisation, the ionisation energy, and their
sources and uncertainties. solprop never hard-codes them. From them solprop
derives $\nu_c$, the band edges and the regions, and it alone sets the rule:
bounding beyond the data, labelling, and never refusing are modelling
choices. The figures above are optmat's at the cited revision, quoted for this
proposal's argument. The record carries its own identity, which enters the
run's manifest and resume key. Phase 1 needs an optmat release that provides
it.

### 5. Proposed interface

Extend SEP 3's material-backed electronic selection. `MaterialSelection.electronic`
exists and `ELECTRONIC_POLICIES` holds `"recommended"` and `"carrier-kerr"`;
add `"symmetric-kerr"` and `"gas-polynomial"`. `"recommended"` keeps meaning
`"carrier-kerr"` until Q6 is decided. The two new values are proposed names, not
APIs present in this Draft:

```text
selection = MaterialSelection(electronic="gas-polynomial")  # or "symmetric-kerr"
argon = species("Ar", selection=selection)
fill = Mixture.fill({argon: 1.0}, 3.0, temperature=294.0)
run = prop_capillary(radius, length, "Ar", 3.0, lambda0=..., materials=selection)
```

Both dispersive options resolve optmat's hyperpolarizability model of the gas,
with its identity, units, density convention and domain. `gas-polynomial` is
refused for a material without that model; it does not silently turn into a
diagonal fit. A positive
real supplied electronic susceptibility curve can support the symmetric model
under its declared assumptions. The existing explicit scalar `KerrResponse`
remains a constant-coefficient response; it does not acquire dispersion merely
because a built-in gas shares its label.

Reference wavelength selection continues to matter for the carrier model:
`MaterialSelection.carrier_wavelength` remains the wavelength of the carrier
Kerr coefficient, and of the Raman pump dispersion and the ionisation rate
under every electronic policy. For the dispersive choices, changing the
envelope carrier while holding the same physical spectral samples must not
change the material response. A
reference used to normalize $g$ is numerical metadata, not a new physical
choice. Polynomial coefficients and filters are evaluated at absolute optical
frequencies, including the sign treatment required by the representation.

The resolved response should carry enough information to identify the data
model, evaluation strategy, tensor approximation, material state convention,
support record and the regions of §4.10. The draft does not
require a general public susceptibility-tensor array API. A structured
material response is preferable to adding several unrelated coefficient
arguments to `KerrResponse`.

Validate the requested active spectral support including collars, not only the
pump or the full padded transform array. Padded zeros do not justify querying
the material model outside its domain. Static/DC limits, the Nyquist bin,
Hermitian completion and any permitted interpolation are explicit. Validation
must consider the interactions retained by the selected operator: a valid
pump alone cannot certify every output. The leg bound of §4.10 is the only
modification of the model, and it is recorded. No silent clipping, blanket
warning suppression, or automatic model switching is proposed.

Record the extent of each region with the output, and distinguish permission
to evaluate a formula from physical qualification. The region check is made
once, on the active band at translation, never per quartet in the hot loop.

§4.10 replaces a selectable extrapolation policy with one rule. Beyond a
gas's data the legs are bounded, and the run is never refused for the
frequencies its field occupies. What a run records is the region each part of
its active band falls in — supported, bounded or best effort — with the
support record's identity; these enter the manifest and the resume key, and
outputs and plots can mark the regions. A run whose field enters the bounded
or best-effort region logs that once, naming the gas and the band, as SEP 3's
range notices do; it does not warn per step. A vector run takes the
any-polarisation band edge.

### 6. Implementation strategies

#### Ownership and numerical representation

Optmat owns material coefficients, source selection, covariances, validity and
identity, including where each model is supported and where the material
resonates. solprop owns the derived polarization operator, its numerical
execution and the modelling rule for using a model beyond its support
(§4.10). SEP 3's adapters resolve the public records; no database or fit
implementation is copied into solprop. Optmat callbacks and Python objects
never enter a backend stage.

The physics layer defines reference operators. Translation constructs spectral
weights on the actual fine grid, response descriptors and density paths. The
seam carries only supported, validated numerical representations. A new
spectral response bundle, rather than an overloaded scalar `KerrCoefficients`,
makes the changed scheduling explicit. The first implementation phase that
changes these fields bumps `SCHEMA_VERSION`, its literal contract tests and
the changelog. An out-of-tree backend checks capability and schema by name.

The compiled implementation uses preallocated buffers, planned transforms and
deterministic sums. It keeps optmat identities on the host and response arrays
resident where the backend executes. Every additional field/product workspace
appears in the memory ledger. No new third-party numerical dependency is
proposed: the existing FFT and array facilities suffice.

#### Response composition and mixtures

The evaluators reach a run through the existing route: the translation layer
asks each species' electronic model for weights on the fine grid, exactly as it
now asks `OptmatKerr` for one carrier value. A run whose electronic response is
SEP 12's TDSE coupling refuses a dispersive policy for the same gas (§2).

For a pure gas with only electronic Kerr, the symmetric multiplier can be
folded into the existing input spectrum and output polarization. It need not
add any FFTs. If Raman or plasma also needs the unfiltered physical field,
both fields must exist or be recomputed, and the electronic polarization may
need a separate output transform/projection before summation. Applying $g$
to the total polarization would spuriously alter Raman and plasma.

For a mixture, in general

$$
P_{\rm sym}=\sum_s\frac{N_s\gamma_{*,s}}{6}
G_s[(G_sE)^3].
$$

Different $G_s$ prevent the current reduction to one scalar
$\sum_sN_s\gamma_{3,s}$, and independent filters cost approximately one
response path per species. The recommended alternative is one filter built
from the mixture's own diagonal curve,
$\gamma_{K,\rm mix}(\nu)=\sum_sN_s\gamma_{K,s}(\nu)$, factorized exactly as
in §4.4. Its separability assumption is the pure-gas one, it keeps a single
response path, and its error against the per-species polynomial sum is
measured by the procedure of §4.7. The filter depends on the composition, not
only on the total density: where composition varies along $z$ it must be
rebuilt per region or per stage, and the recorded model identity names the
composition it was built for. Any other approximate common filter needs its
own demonstrated error and recorded identity.

The polynomial has a useful mixture advantage. When species share the same
spatial field and scalar operator, form the four stage coefficients
$\sum_sN_s\gamma_{0,s}(1,A_s,B_s,B'_s)/6$ and use the same $E,DE,D^2E$.
Thus its FFT count need not grow with species count. Keep regional/spatially
varying coefficients at their proper nodes; the ability to sum four
coefficients does not make different material regions uniform.

#### Geometry and transform counts

| Setting | Symmetric strategy | Polynomial strategy |
|---|---|---|
| Scalar uniform material, electronic response only | One inverse and one forward transform, as for the constant cubic; two spectral multiplications | Three inverse and three forward transforms with grouped products |
| Uniform legacy modal material | A common scalar filter commutes with frequency-independent synthesis/projection; transform modal columns and reuse chunking | Derivative fields can likewise be transformed on modal columns; extra synthesized fields and product projections remain |
| General modal maps / regional materials | Respect frequency-dependent synthesis and region-specific filters; a common scalar multiplier can commute only where genuinely common | Share derivative work where possible, but retain regional tensors/coefficient paths and the existing frequency-dependent maps |
| Radial, Cartesian, planar or line spatial routes | Filter temporal spectra at each material field point; existing spatial work remains | Additional temporal transforms and derivative/product buffers at each field point |
| Raman/plasma and multiple distinct gas filters | Preserve bare fields and separate source assembly; cost can exceed the isolated two-transform estimate | Reuse the bare field, but retain all electronic product channels and unfiltered other sources |

Complex envelope transforms, positive-frequency projections and harmonic
selection change the constants in this table. Six real transforms is the
simple scalar full-field polynomial count, not a promise for every route.
Likewise, a common filter does not imply that all regional maps commute.

With $C$ transformed columns and $N_t$ fine-grid samples, both methods remain
$O(CN_t\log N_t)$ for fixed response rank. The polynomial needs at least the
additional derivative fields or an equivalent recomputation schedule;
temporary products can be streamed. Chunk sizes should follow the actual
memory ledger, not a fixed multiplier on current allocations. For many modes,
extra dense synthesis/projection work can matter more than the FFT count.
On GPUs, memory bandwidth, batch size and transform plans must be measured;
CPU ratios cannot be transferred to device runs.

#### Exploratory timings and what they establish

An initial isolated CPU experiment measured the following medians per cubic
polarization evaluation. These are **not solprop propagation benchmarks**.

| Real field shape | Constant (µs) | Symmetric (µs; relative) | Polynomial (µs; relative) |
|---|---:|---:|---:|
| `(1, 4096)` | 20.0 | 25.0; 1.26× | 78.6; 3.94× |
| `(8, 16384)` | 1008.3 | 1078.9; 1.07× | 3311.1; 3.28× |

The experiment used SciPy real FFTs with one worker, NumPy products and
allocated temporaries, single-threaded external math libraries, warm-up calls,
and the median of seven groups of twenty evaluations. It excludes material
loading, propagation, modal projection, Raman, plasma and adaptive stepping.
Those figures establish plausible kernel-scale cost, not production speed.

A repeat during drafting with all fixed spectral weights precomputed gave
23.1/24.0/79.8 µs for the first shape and 1023.2/1109.1/3440.6 µs for the
second (constant/symmetric/polynomial). The corresponding relative costs are
1.04/3.45 and 1.08/3.36. The variation cautions against promising an exact
percentage from a small microbenchmark.

For reproducibility, the repeat used Python 3.14.5, NumPy 2.5.2 and SciPy
1.18.1; `default_rng(42)` generated standard-normal real arrays in the shape
order above, transformed once before timing. It used dimensionless
$d_k=(k/(N_t/2))^2$, $A=0.05$, $B=0.02$, $B'=-0.003$,
$g_k=[1+4Ad_k+(16B+4B')d_k^2]^{1/4}$, and the grouped formula in §4.5.
Each function was warmed five times. Spectral weights were constructed outside
the timed call; field/product arrays were allocated inside it. Each of seven
samples used twenty calls timed with `perf_counter_ns`. The synthetic weights
measure the arithmetic paths and make no material claim. No propagation was
timed, and these results should not enter the production benchmark ledger as
though an implemented backend had been measured.

Whole-stage cost depends on the fraction spent in these operations, extra
projections and response coexistence. Whole-run cost also depends on accepted
and rejected steps, since changing the physics can change the adaptive
trajectory. Measure those separately on short windows. Do not extrapolate a
microbenchmark or ten-step timing to a full fibre run.

### 7. Verification and validation

Separate three layers of evidence: numerical implementation against the
specified kernel; approximation error against an independently evaluated
material model; and material-model agreement with physical measurements or
calculations. An extremely accurate FFT implementation can still use an
unsupported susceptibility.

| Test family | Comparison and acceptance basis |
|---|---|
| Units and normalization | Taylor factor of six, SI coefficient conversion ($10^{-10}$ cm$^2$, $10^{-20}$ cm$^4$ and the $W_2/S_2$ scale), electronic $n_2/\chi^{(3)}$ conversion against both forms in §4.2, and density linearity; algebraic tolerances from floating-point reductions |
| Constant and monochromatic limits | Recover existing constant cubics in their shared scope; recover the Kerr diagonal and the distinct polynomial THG quartet, including degeneracy factors |
| Explicit convolution | Small signed-frequency direct sums for arbitrary multitone fields, not just one pump; include FWM, THG and inverse exchange; reduction-length error bounds |
| Symmetry and reality | Input permutation, lossless four-leg exchange within the model assumptions, Hermitian symmetry, negative-frequency completion and DC/Nyquist handling |
| Work and propagation invariants | The local identity in §4.9 and physical energy for lossless full-field propagation, with the band projection of §4.5 in place, and a negative check that an output-only cut-off breaks it; photon number only for the corresponding number-conserving sector |
| Mixture filter | The mixture-curve symmetric filter of §6 against the per-species polynomial sum, by the §4.7 procedure, for at least one binary mixture |
| Grid and carrier | Dealiasing convergence, larger time windows, absolute-frequency derivative weights, carrier-reference independence, and field/envelope parity beyond a narrowband pulse |
| Mixtures and regions | Explicit per-species sum versus aggregated polynomial coefficients; identical versus distinct filters; regional material response and density-profile variation |
| Response coexistence | Electronic changes do not filter or renormalize Raman/plasma sources; verify combined energy-transfer accounting |
| Backend parity | One nonlinear evaluation, one step and about eight to ten steps across NumPy, Numba and Kokkos CPU, on reduced forms of both demonstration cases among others; the Kokkos device build under `pytest -m gpu`; explicit capability refusal only while a backend's phase is still open |
| Gas coverage | Every gas optmat fits resolves both evaluators, or is refused by name when its record or domain does not admit the request; the CO2, CO and O3 records are exercised at the model level until solprop builds them as species |
| Leg bound and regions | The bound leaves every fitted value and every quartet within $S_{2,\max}$ exact, and holds legs beyond $\nu_c$; the lossless identity holds with it; the region of each part of the active band is recorded, a field reaching the best-effort region runs and logs once, and a vector run takes the any-polarisation edge |
| Identity and errors | Changed material fit, support record, evaluator, band or tensor approximation invalidates resume/cache compatibility; malformed inputs fail before kernels |

Use independent convolution code for the oracle: reproducing the same grouped
FFT expression in a second function does not independently check the algebra.
Include the §4.9 fundamental-plus-third-harmonic example and quartets that
cross the analytic-signal projection boundary. A narrowband test alone cannot
qualify the proposed THG semantics.

For model comparisons, publish frequency/quartet maps with 0.1%, 1% and 5%
contours as descriptive levels, not predetermined acceptance thresholds.
Show supported and extrapolated regions separately. Include maximum error,
representative spectral weighting and the fit's uncertainty; preserve
correlations when comparing values from the same fitted curve. A covariance
alone is not a bound on a truncated model or its borrowed priors.

Propagation qualification should include a short broadband SPM case, seeded
widely separated phase-matched FWM, and a weaker THG/reverse-exchange check.
The two demonstration cases of §2 serve: the argon DUV FWM case as the seeded,
phase-matched FWM witness, and a reduced helium RDW case — the reduced M2 size
of the [testing plan](../../testing.md), about ten steps — as the broadband,
harmonic-retaining field-grid witness, whose band reaches the bounded and
best-effort regions of §4.10. Use phase matching obtained from the admitted
linear model and geometry.
Compare complex spectral phase, nonlinear phase/B-integral, signal/idler
conversion, pump depletion and energy in specified weak UV bands. Also vary
nonlinear phase matching: an error in coupling or nonlinear phase can strongly
alter gain and conversion near a sensitive operating point. A coefficient
error of 1% does not imply a spectral-energy error of 1%.

Keep material parameters, input pulse and linear operator fixed when comparing
the electronic evaluators. First establish step/grid convergence within each
model; then attribute the remaining difference to the model. Do not retune
pressure or pulse energy to restore an old spectrum. SEP 3's data changes
need their own comparison axis. Short-window tests establish local numerical
and physical behavior; scientifically representative longer runs belong in
the worked-example gallery under the repository's existing policy.

Benchmark isolated response, full nonlinear stage and a short propagation
window separately. Cover one mode, enough modes to expose projection cost,
one spatial route, a mixture, and Raman/plasma coexistence. Report grid,
oversampling, modes/nodes, threads, buffers/peak memory, accepted/rejected
steps and timings; compare evaluators at equal achieved error with SEP 16's
work–precision measurements under the default `Multistep()`. Record implemented
hot-path results in
[the benchmark ledger](../../benchmarks.md), with no full-run extrapolation.

### 8. Documentation

Implementation extends the nonlinear response chapter with the unified
susceptibility, both operators, source conventions and the exact scope of the
THG choice. Add a material-validity section and examples showing how to choose
an evaluator and inspect the model identity. Update the GNLSE and general
modal pages to distinguish material dispersion, modal dispersion and the
UPPE frequency factor. Update the architecture and kernel contract when the
spectral response bundle actually exists.

A quick tutorial compares carrier, symmetric and polynomial responses on the
two demonstration cases of §2, exposing both coefficient and propagation
differences, with the extrapolated region marked:

- **DUV FWM in argon**: the 400 nm + 800 nm → 266.7 nm capillary, single mode,
  at the phase-matching pressure of the admitted linear model; it compares
  the signal energy and spectrum and shows the coefficient table of §4.7.
- **VUV RDW in helium**: the case of the existing RDW tutorial
  (`docs/tutorials/capillaries/plot_02_rdw_emission.py`), which is M2's in a
  shorter, two-mode form; it compares the RDW band's energy and centre
  wavelength as a sensitivity to the extrapolated VUV legs (§4.7).

Each run stays as short as its point allows, as tutorials run on every docs
build. The full-size M2 comparison belongs with the existing worked example
(`docs/examples/processes/plot_00_vuv_dispersive_wave.py`), re-run by hand
with the three evaluators; an argon DUV worked example beside the existing
helium FWM one is optional. The nonlinear manual page carries both cases'
coefficient tables and the domain maps of §7. Physical lengths follow the
gallery rules; tests and benchmark lanes remain short. Public docstrings state units,
domain, approximations and error behavior. Record changed operator semantics
on the Luna differences page if an accepted implementation introduces them.

### 9. Compatibility and impact

The carrier-evaluated electronic model remains the initial default inherited
from SEP 3. Both dispersive options are opt-in and are never selected by backend
availability, species count or the generated bandwidth. No material data return
to solprop. An unsupported backend names the missing response and refuses it.

New propagation results are expected to differ: nonlinear phases, spectral
transfer, phase matching, compression and adaptive step histories can all
change. A model change is not a numerical regression to conceal by relaxing
tolerances. Existing validated carrier cases keep their model selection;
new dispersive cases receive their own qualified witnesses. Reproduce older
published results with their recorded environments and model identities.

Store the electronic evaluator and version, material fit identity/digest,
normalization, tensor and harmonic-sector assumptions, domain/extrapolation
policy, and any user-supplied model identity. Include these in scan and resume
keys. A file whose existing metadata cannot establish those identities cannot
silently resume with a different response. SEP 3 delivered the carrier for
these: a run's `MaterialManifest` records each model's optmat identity and the
`electronic` policy, and they enter the resume key. The new policies add the
evaluator, the band and the extrapolation record to the manifest's electronic
entry, with the support record's identity and the regions of §4.10; that
changes the persisted layout, so an output-schema bump (from 15
at the final review) is expected, and the kernel seam change requires its own
kernel-schema bump (from 23). Keep older files readable under the established
output policy.

The `thg=False` issue in §4.9 is explicit compatibility work. Acceptance must
choose how a consistently restricted new response is distinguished from the
existing constant partial cubic. This Draft does not silently redefine the
old response or claim broadband parity that its current tests do not establish.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Keep one carrier coefficient | Useful baseline and initial default; cannot represent arbitrary broadband material dispersion |
| Multiply the final cubic spectrum by $\chi_K(\omega)$ | Cheap but generally incorrect for nondegenerate quartets and lossless exchange; not recommended |
| Separate coefficients for SPM, XPM, FWM and THG | Artificial for a continuous spectrum; easy to break inverse-process consistency; the joint kernel already handles the processes |
| Quartet effective-frequency Kerr curve | Often a good interpretation of the gas polynomial; cannot be implemented by assigning one effective frequency to each output |
| Miller scaling from linear refractive index alone | Useful under additional oscillator assumptions; not reliable enough to replace curated nonlinear dispersion |
| Symmetric factorization only | Attractive cost, but a sampled few-percent error may matter; the agreed recommendation also supports polynomial propagation |
| Polynomial only | Removes the factorization error within the model, but adds transforms/projections where the cheaper option may be sufficient |
| General dense frequency kernel | Too expensive for the intended grids; retain only as a small-grid independent oracle |
| General low-rank or causal oscillator model | Plausible future treatment of richer tensor/resonant response; requires data, symmetry and causality constraints beyond this proposal |

### 11. Phases

#### Phase 0 — Mathematical and material qualification

Confirm SI adapters and public material records with optmat. Reproduce the
quartet comparisons, direct-convolution reference and invariant diagnostics.
Derive broadband field/envelope and harmonic-sector forms for both evaluators.
Set a supported transparent-gas domain and document per-species evidence,
including molecular/tensor approximations. Agree observable-specific accuracy
and runtime targets without confusing fit uncertainty with numerical tolerance.
Resolve Q1–Q4 sufficiently for a first supported implementation, and confirm
the leg bound and region labels of §4.10 on the two demonstration cases. Name remaining
UV/VUV and silica work explicitly; do not defer a fundamental model decision
to a backend engineer.

#### Phase 1 — NumPy reference and public response

Implement both electronic choices, their translation and the versioned seam,
starting with scalar gases on field and envelope grids, for every gas optmat
fits (§2). Preserve
the carrier response and explicit user data. Add metadata/resume handling,
manual pages and the short example together with direct-sum, invariant,
convergence and short propagation tests. Qualify mixtures and route support
explicitly; refuse any combination not yet supported. Implement a tensor
extension only under the accepted Phase 0 model, not by inferring it from a
scalar fit. NumPy is the oracle.

#### Phase 2 — Numba CPU

Add the same operators with preallocated workspaces, transform planning,
species aggregation and deterministic kernels. Qualify all Phase 1 routes,
mixed response cases and memory accounting against the reference. Record
isolated, stage and short-window performance, including cases where projections
dominate. No `fastmath` or silent fallback. Update the benchmark ledger.

#### Phase 3 — Kokkos CPU, then device qualification

Extend solprop-kokkos to the new seam and operators, keeping arrays resident
through each stage. **This phase is required, not optional**: no exception for
solprop-kokkos is written into this SEP, and it does not go `Final` while
Kokkos refuses any combination that `numpy_ref` admits. Match NumPy and Numba
across admitted geometries, mixtures, response combinations and metadata on
the CPU build. The device build compiles the same operators from the one
source and is qualified with `pytest -m gpu` under the existing separate GPU
policy; CPU results do not establish device performance or parity. Record
memory and short-window cost for both.

#### Phase 4 — Scientific qualification and closing scope

Publish the agreed accuracy/cost comparisons and application guidance for both
options, including the full-size demonstration cases of §8. Decide whether
either becomes a default in a separately explicit
compatibility decision. Silica and resonant UV/VUV claims require their own
curated inputs and evidence; unresolved work becomes named roadmap items or
issues. All required backends must support the accepted scope before Final,
with no exception for solprop-kokkos (Phase 3). Append each phase's
verification, source audit, limitations and handoff to Part II when delivered.

### 12. Open questions

- **Q1 — Acceptance by observable.** Which coefficient, nonlinear phase and
  conversion errors are acceptable for the priority SPM and widely separated
  FWM cases? Maintainers decide during Phase 0. No fixed 1% promise follows
  from the sample table; weak UV output and sensitive phase matching need
  separate criteria.
- **Q2 — Material domains beyond the diagonal.** Answered in substance:
  optmat's support analysis and its draft OEP 5 [12, 13] supply the material
  facts, and §4.10 sets solprop's rule on them. Remaining before Phase 1:
  - optmat's release of the support record, carrying material facts only;
  - for optmat: whether O2's spin-forbidden $a$ and $b$ states, which pair
    differences reach inside the range, count as resonances, and the record's
    units;
  - for solprop: whether the bounded region is reported as a lower bound or
    as an estimate, and whether the few Xe and CO quartets not shown better
    than a 400 nm carrier need a narrower bound. The region check is
    solprop's, made on its active band from optmat's levels.

  Resonant response in the best-effort region, including helium's VUV, is a
  named curation/model workstream.
- **Q3 — Tensor and silica scope.** Which scalar-times-isotropic-tensor
  approximation, if any, is acceptable for the first vector release? What
  independent nondegenerate silica data justify its factorization? Resolve
  admitted scope before implementation; leave unsupported claims explicit.
- **Q4 — Harmonic-sector compatibility.** Adopt the complete quartet response
  and the consistently number-conserving restriction of §4.9 for the new
  choices? How should these map to `thg` and general-modal `kerr_form` without
  silently changing legacy partial cubics? This is a pre-acceptance physics
  and interface decision, supported by the diagnostic already identified and
  by the harmonic-leg domain table of §4.8: under the leg bound, both
  evaluators keep the full cubic exact only for pumps at or above its first
  column (648–695 nm for most gases, 456 nm for helium).
- **Q5 — Performance targets.** What extra time and memory are acceptable
  for the priority single-mode, multimode and mixed-response workloads?
  Measure complete stages before choosing buffer/recomputation trade-offs.
- **Q6 — Eventual default.** Should a qualified dispersive model replace
  carrier evaluation, and for which materials/domains? Keep carrier as the
  initial default; decide only after scientific and backend qualification.

### 13. References

1. optmat, [Third-order nonlinearity](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear.md),
   [Third-order accuracy](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear_accuracy.md)
   and [Fused silica](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/silica.md).
   Public material-model and curation reports; numerical comparisons use the
   model revision identified in §2, not an independent experimental dataset.
2. D. P. Shelton and J. E. Rice, “Measurements and calculations of the
   hyperpolarizabilities of atoms and small molecules in the gas phase,”
   *Chemical Reviews* **94**, 3–29 (1994),
   [doi:10.1021/cr00025a001](https://doi.org/10.1021/cr00025a001).
3. W. Ettoumi, Y. Petit, J. Kasparian and J.-P. Wolf, “Generalized Miller
   formulae,” *Optics Express* **18**, 6613 (2010),
   [doi:10.1364/OE.18.006613](https://doi.org/10.1364/OE.18.006613).
4. V. Mizrahi and D. P. Shelton, “Dispersion of nonlinear susceptibilities
   of Ar, N2, and O2 measured and compared,” *Physical Review Letters*
   **55**, 696–699 (1985),
   [doi:10.1103/PhysRevLett.55.696](https://doi.org/10.1103/PhysRevLett.55.696).
5. J. J. Pigeon, S. Ya. Tochitsky, E. C. Welch and C. Joshi, “Experimental
   study of the third-order nonlinearity of atomic and molecular gases using
   10-µm laser pulses,” *Physical Review A* **97**, 043829 (2018),
   [doi:10.1103/PhysRevA.97.043829](https://doi.org/10.1103/PhysRevA.97.043829).
6. D. M. Bishop and D. W. De Kee, “The frequency dependence of nonlinear
   optical processes,” *Journal of Chemical Physics* **104**, 9876–9887 (1996),
   [doi:10.1063/1.471752](https://doi.org/10.1063/1.471752).
7. C. Hättig and P. Jørgensen, “Coupled cluster calculations of the
   frequency-dependent second hyperpolarizabilities of Ne, Ar, N2, and CH4,”
   *Journal of Chemical Physics* **109**, 2762–2778 (1998),
   [doi:10.1063/1.476833](https://doi.org/10.1063/1.476833).
8. M. Conforti, A. Marini, T. X. Tran, D. Faccio and F. Biancalana,
   “Interaction between optical fields and their conjugates in nonlinear
   media,” *Optics Express* **21**, 31239–31252 (2013),
   [doi:10.1364/OE.21.031239](https://doi.org/10.1364/OE.21.031239).
9. C. G. Durfee III, S. Backus, H. C. Kapteyn and M. M. Murnane, “Intense 8-fs
   pulse generation in the deep ultraviolet,” *Optics Letters* **24**, 697–699
   (1999), [doi:10.1364/OL.24.000697](https://doi.org/10.1364/OL.24.000697).
10. J. C. Travers, T. F. Grigorova, C. Brahms and F. Belli, “High-energy pulse
    self-compression and ultraviolet generation through soliton dynamics in
    hollow capillary fibres,” *Nature Photonics* **13**, 547–554 (2019),
    [doi:10.1038/s41566-019-0416-4](https://doi.org/10.1038/s41566-019-0416-4).
11. A. Kramida, Yu. Ralchenko, J. Reader and NIST ASD Team, *NIST Atomic
    Spectra Database*, National Institute of Standards and Technology,
    [doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F).

12. optmat, [Where the fits are supported](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear.md#where-the-fits-are-supported)
    and [Beyond the fitted data: bounds on the legs](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear_accuracy.md#beyond-the-fitted-data-bounds-on-the-legs),
    at optmat `a9e5dbf`: the fitted legs, resonance limits, monotonicity
    evidence and per-leg bound comparisons that §4.10 adopts.
13. optmat, [OEP 5 — a public record of where the hyperpolarizability fits are supported](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/archive/oep-0005-hyperpolarizability-support.md),
    Draft, at optmat `0d5a0ab`.

**Source audit.** The nonlinear dispersion expansion and its diagonal-component
scope were checked in [6], the distinct tensor dispersions in [7], and the
oscillator assumptions and experimental limitations of linear-index scaling
in [3–5]. The analytic-signal issue was checked against [8] and independently
against the current solprop cubic. The Taylor convention and fit/domain
interpretation use [1, 2] and SEP 3's source audit. The model ratios, quartet
survey and algebraic diagnostics were calculated independently for this Draft.
The demonstration cases rest on [9, 10] for their configurations, on solprop's
own M2 definition and SEP 12's helium resonance positions, and on [11] for
argon's resonance lines and helium's levels; their bibliographic details were
checked, their measured results are not used. The §4.10 support figures are
optmat's [12]: the leg bound and the per-row bounded values were re-derived
independently from the stored fits and agree, while the resonance energies and
the carrier comparison counts were taken from [12] and not recomputed. No new
primary coefficient table was transcribed. This work does not re-fit the
underlying measurements or validate VUV extrapolation, nonlinear absorption, a
complete tensor model, or an implemented dispersive propagator.

## Part II — Implementation record

### Phase 0 — mathematical and material qualification

- **Scope:** Phase 0 of §11, 2026-10-07, implemented by Opus 5.5 (AI) for John
  Travers, on `feature/sep-0006-dispersive-nonlinearity-0`. No package code changes:
  Phase 0 fixes the algebra and the material evidence that Phase 1 implements.
- **Delivered:**
  - `tests/dispersive_kerr_oracle.py`: the independent oracle, an explicit sum over
    signed input triplets of a band-limited periodic field with the kernel evaluated
    leg by leg, scalar and isotropic vector; and the FFT forms derived below, for both
    evaluators, with the leg bound, in both sectors, on field and envelope frames.
  - `tests/nonlinear/test_dispersive_kerr_algebra.py`: 37 tests holding the FFT forms
    to the oracle and to the invariants of §4.9, the band rule of §4.5 and the bound of
    §4.10, with negative checks of the output-only weighting and of the existing
    field-grid `thg=False` cubic.
  - `tests/sep6_quartets.py`, which writes `tests/data/sep6_quartets.json`, and
    `tests/test_sep6_quartets.py`, which recomputes it. The record holds §4.7's survey,
    FWM and demonstration tables, §4.8's harmonic table, §4.10's leg bounds and the SI
    conversion check. Each is evaluated through optmat's public
    `HyperpolarizabilityModel` at the pinned v0.3.0, with the identity digest of every
    model.

#### The operators Phase 1 implements

Write the bounded leg weight $\tilde d(\omega)=\min\{(\omega/2\pi c)^2,\nu_c^2\}$, a real
even spectral multiplier, with $\nu_c=\sqrt{3S_{2,\max}/4}$ per gas (§4.10). For a
product of three real fields $X,Y,Z$ with analytic signals $x,y,z$ ($X=\mathrm{Re}\,x$),
each factor is $(x+x^*)/2$. The analytic polarisation $u_P=2\Pi_+P$, with $\Pi_+$ the
projection onto absolute frequencies above zero, is then

$$
u_P=\tfrac14\Pi_+\big[\,xyz+(xyz^*+xy^*z+x^*yz)+(xy^*z^*+x^*yz^*+x^*y^*z)\,\big].
$$

The three groups are the quartets whose signed legs, the output counted negative, have
three, two and one positive members. A zero-frequency leg belongs to none; the active
band never contains $\omega=0$.

- **The full cubic** keeps all three groups. On a field grid it is the real product
  $XYZ$ itself.
- **The number-conserving sector** keeps the middle group:
  $P_{\rm NC}=\mathrm{Re}\{\tfrac14\Pi_+[xyz^*+xy^*z+x^*yz]\}$. For $X=Y=Z=E$ this
  is $\mathrm{Re}\{\tfrac34\Pi_+[|u|^2u]\}$, §4.9's form.

Each sector restricts the kernel to a set of quartets closed under every permutation of
the four signed legs, so a symmetric kernel stays symmetric and §4.9's work identity
holds in both sectors. In the conserving sector the sum
$\sum\chi\,a_o^*a_3^*a_1a_2$ over quartets $\omega_o+\omega_3=\omega_1+\omega_2$ is real,
being its own conjugate under $(o,3)\leftrightarrow(1,2)$, so the photon-number rate
vanishes too. In the full sector it does not, because of the harmonic quartets.

Both evaluators are sums of such products:

- **The polynomial** is §4.5's grouped form with $D$ replaced by the multiplier
  $\tilde d$: four products, $E^3$, $E^2\tilde DE$, $E^2\tilde D^2E$ and
  $E(\tilde DE)^2$, each with its output multiplier.
- **The symmetric model** is one product of $GE$, with
  $g(\omega)=[\gamma_K(\tilde\nu)/\gamma_0]^{1/4}$. The reference is the static
  $\gamma_0$, so $g(0)=1$ and $g\ge1$ for every fit (§4.10).

A sector is applied to each product, and the output multipliers commute with $\Pi_+$
and with taking the real part.

**Envelope grids.** With $A=u\,e^{-i\omega_0t}$, the convention of
`nonlinear.responses.thg_carrier`, the same expression holds with every multiplier
evaluated at the absolute frequency $\omega_0+\Delta\omega$. The first group carries
$e^{+2i\omega_0t}$, as the existing harmonic does, and the third carries
$e^{-2i\omega_0t}$. The third group is new. It is the one-into-three exchange, and
reaches positive absolute frequencies only for fields wider than an octave, so an
envelope fine grid must hold it or remove it by the band projection, which acts on
inputs and output alike.

**Vector fields.** The scalar kernel multiplies the isotropic tensor
$T_{ijkl}=(\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})/3$.
Three distinct filtered fields contract as
$\tfrac13[(X\!\cdot\!Y)Z+(X\!\cdot\!Z)Y+(Y\!\cdot\!Z)X]$, which is $(E\cdot E)E$ when
they coincide. This is a named approximation: Kleinman's ratio
$\chi_{xxyy}=\chi_{xxxx}/3$ is held at every frequency. optmat's $\gamma$ is the mean
of parallel linear polarisations, and the components' distinct dispersion [7] is not
represented.

#### Decisions

Proposed in this phase and accepted by John Travers on 2026-10-07; they bind Phase 1
onward.

- **Q4, the harmonic sector.** For the two new policies, `thg=True` and
  `kerr_form="real"` select the full cubic, and `thg=False` and
  `kerr_form="analytic"` select the number-conserving sector, with $\Pi_+$. The
  existing constant cubic is left as it is.
  - The existing rules equal the new sector whenever $|u|^2u$ has no
    negative-frequency content, that is for any band narrower than an octave
    ($\omega_{\max}<2\omega_{\min}$). This is tested.
  - Beyond an octave the field-grid `thg=False` cubic lacks $\Pi_+$ and does net
    work, also tested.
  - Adding $\Pi_+$ to it is a small correction to an existing operator, outside this
    SEP's scope. It is proposed as a separate change (open question 2).
- **Q3, vector scope.** The scalar kernel times the isotropic Kleinman tensor, recorded
  as such in the manifest. Silica keeps its carrier-evaluated cubic: no curated
  nondegenerate silica data exist to factorise (§4.8).
- **Q1, accuracy by observable.** Numerical acceptance is separated from model
  comparison.
  - Implementations meet the oracle to `1e-12` relative on the Phase 0 grids.
  - Backends meet the reference under the existing parity rules.
  - Propagations converge on step and grid ladders.
  - Differences between the evaluators and the carrier model are reported, not gated.
    For the demonstration cases they cover: argon's 266.7 nm energy, conversion and
    spectral phase; helium's RDW band energy and centre wavelength; the B-integral.
    Phase 4 turns them into application guidance.
- **Q5, performance.** Targets are proposed from §6's kernel measurements, to be
  measured on whole stages in Phases 1–2:
  - **Time:** the symmetric nonlinear stage within 1.3 times the constant cubic's on
    the same route, the polynomial within 4 times.
  - **Memory:** the symmetric model adds at most one field-sized buffer, needed only
    when Raman or plasma also needs the bare field; the polynomial adds two derivative
    fields and one product buffer.
  - A missed target opens an optimisation slice. It never refuses a run.
- **Q2, what remains.** The bounded region's standing becomes a *bounded estimate*,
  read as a lower bound below resonance by solprop and not certified by optmat (its
  OEP 5 review). The region check runs on the whole active band, which contains every
  generated output the grid can carry.

#### Deviations from Part I

Part I is frozen; these qualify it, and Phase 1 follows them.

- **"A lower bound on γ" (§4.7, §4.10)** reads *bounded estimate*: solprop's
  interpretation of the evidence, which optmat does not certify. Just beyond the data
  the polynomial lies 0.1–0.2% above two calculations (argon's and N2's THG); farther
  out it lies below every calculation. "Shown no worse than the carrier" beyond the
  data rests on the same interpretation.
- **The band edge (§4.10)** is a sufficient bound: below it no leg reaches $E_1$ and no
  pair reaches $E_2$. For one linear polarisation the SPM pair reaches $E_2$ exactly at
  it. For a vector field the any-polarisation edge is conservative: two photons of
  equal frequency do not reach an even $J=1$ level. A field beyond the edge is
  labelled best effort, which stays correct.
- **The polyatomic limits (§4.10)** are evidence, not levels, so the edge is undefined
  for CO2, CH4, SF6, N2O and O3:
  - no CH4 or SF6 absorption is detected at the sampled wavelengths down to 197.7 nm;
  - N2O's 217.5 nm is where an integration over its B band begins;
  - O3's 1047 nm is the end of the curated table, not an onset; O3 absorbs to at
    least there.

  The policy is unchanged: these gases are best effort beyond their data.

#### Verification

- **The algebra:** 37 tests.
  - The FFT forms of both evaluators, bounded and unbounded, in both sectors, scalar
    and vector, agree with the triplet sum to $1.3\times10^{-14}$.
  - The work identity holds to $4\times10^{-17}$ in every case. The photon-number
    identity holds in the conserving sector to $6\times10^{-17}$, and fails in the full
    sector, as it must.
  - The envelope frame equals the field frame to $1.5\times10^{-15}$ at two carriers,
    which is §5's carrier independence.
  - Monochromatic fields give $\tfrac34\gamma_K$ and $\tfrac14\gamma_{\rm THG}$ at the
    polynomial's own sums, and $\tfrac14 g(3\omega)g(\omega)^3$ for the factorisation's
    harmonic.
  - The constant limit is `KerrResponse(thg=True)` exactly, and `thg=False` below an
    octave.
  - Projecting the band on every leg keeps the work identity; projecting the output
    alone breaks it.
  - The leg bound changes no quartet within the fitted sums.
- **The material record.**
  - solprop's SI evaluation — the Taylor factor of six, $10^{-10}$ cm² → $10^{-14}$ m²,
    $10^{-20}$ cm⁴ → $10^{-28}$ m⁴ — equals optmat's `gamma3` for the Kerr and
    harmonic processes, and its `gamma_au` for the pump-probe process, to
    $3.3\times10^{-16}$ for all fifteen gases.
  - Every row of §4.7's survey, FWM and demonstration tables, §4.8's harmonic table
    and §4.10's leg bounds reproduces through optmat's public API.
- **Not run:** no propagation, since none exists yet; no timing; no Kokkos or Numba
  work.

#### Per-species evidence for the first implementation

The leg bound, the largest fitted leg and the band edge for one linear polarisation,
in nm, from optmat at `18ad60a` [12, 13]; "—" is undefined.

| Gas | Leg bound | Largest fitted leg | Band edge | Notes |
|---|---:|---:|---:|---|
| He | 151.9 | 151.9 | 120.3 | Fitted to Bishop and Pipin's calculation alone; measurements validate |
| Ne | 215.9 | 229.0 | 133.5 | |
| Ar | 215.9 | 229.0 | 189.4 | |
| Kr | 230.1 | 244.1 | 216.7 | $B'$ held at its prior |
| Xe | 230.1 | 244.1 | 256.0 | $n_2$ changes sign at 248.4 nm, past the edge |
| H2, D2 | 215.9, 224.7 | 229.0, 231.5 | 201.7, 201.1 | Vibrational part subtracted |
| N2 | 215.9 | 229.0 | 290.1 | Vibrational part subtracted; edge set by the two-photon $a\,{}^1\Pi_g$ state |
| O2 | 224.7 | 238.3 | 304.6 | Vibrational part subtracted; $B'$ at its prior; limits air |
| CO2, CH4, SF6 | 230.1, 224.7, 215.9 | 231.5, 238.3, 229.0 | — | Fits include a small vibrational part |
| CO | 231.5 | 231.5 | 308.9 | |
| N2O | 383.4 | 580.0 | — | One measured value; dispersion borrowed from CO2 |
| O3 | 401.0 | 694.6 | — | Theory only; absorbs throughout its range |

#### The rest of this report

- **Sources checked:** optmat's manual at `18ad60a`, "Where the fits are supported"
  and "Quantities derived from the facts", and OEP 5, read for the facts quoted above.
  No new literature.
- **Limitations:**
  - The FFT forms are references on periodic dimensionless grids, not the package's
    grids, windows or modal projections; Phase 1 qualifies those.
  - The proposed targets are unmeasured on whole stages.
  - The band edges are quoted from optmat's manual until its support record is
    released and pinned.
- **Open questions:**
  1. *Resolved:* the proposals for Q1, Q3, Q4 and Q5 were accepted on 2026-10-07.
  2. Add $\Pi_+$ to the existing field-grid `thg=False` cubic and to
     `kerr_form="analytic"`, as a separate change; it alters results only for fields
     wider than an octave.
  3. The optmat release that carries OEP 5's support record, and the pin move to it,
     before Phase 1 reads $\nu_c$ and the edges.
  4. Whether an envelope fine grid built for harmonics also holds the one-into-three
     group, or projects it away, measured in Phase 1 against the field grid on a
     super-octave case.
- **Handoff:** Phase 1 starts from `main` with this branch merged.
  - It implements the operators above in `numpy_ref` behind the two new policies,
    tested against `tests/dispersive_kerr_oracle.py`.
  - It reads $\nu_c$ and the edges from optmat's support record, never from this
    report.
  - It records the regions and the record's identity in the manifest.
  - `tests/sep6_quartets.py` is rerun whenever the optmat pin moves; its test reports
    a changed model.

### Phase 1a — the dispersive response on numpy_ref

- **Scope:** slice 1a of Phase 1 (§11), 2026-10-07, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-1` in solprop and in
  solprop-kokkos. Both dispersive policies through the material layer, the response,
  the capillary driver routes, translation and kernel schema 24, executed by `numpy_ref`
  on its fused node path; every other route and backend refuses the response by name.
  Slice 1b adds the general modal, slab and Cartesian free-space routes; slice 1c the
  manual chapter, the tutorial and the demonstration tests.
- **Delivered:**
  - `physdata/selection.py`: `ELECTRONIC_POLICIES` gains `"symmetric-kerr"` and
    `"gas-polynomial"`; `DISPERSIVE_POLICIES` maps them to their evaluators;
    `"recommended"` is still the carrier policy.
  - `physdata/nonlinear_data.py`: `DispersiveKerrModel`, optmat's fit in SI —
    `gamma3_static = gamma0 au / (6 eps0)` in m⁵/V², `a` in m² (×1e-14), `b`, `b'` in
    m⁴ (×1e-28) — with `leg_bound = support.leg_max_in_w2` and both band edges from
    optmat 0.4.0's support record, the model and support records, and `regions()`;
    `ElectronicRegions`; `OptmatKerr.dispersive_model()`.
  - `nonlinear/responses.py`: `DispersiveKerrResponse` (evaluator, the SI model, `thg`,
    `species`; traits: convolution, field and envelope grids, vector-isotropic),
    `DispersiveWeights`, and `dispersive_cubic`, the Phase 0 forms on a fine grid's
    spectral layout at absolute frequency, the arithmetic both the reference
    `evaluate` and the `numpy_ref` pass run.
  - `nonlinear/mixtures.py`: `electronic_response`, the response a selection's policy
    names; `responses_for` uses it, so the mixture route and scans
    (`ResponseSelection`) follow the policy.
  - `translate/responses.py`: the leaf `DispersiveKerrTerms` on
    `CompiledResponses.dispersive_kerr`, `dispersive_terms`, the `dispersive_kerr`
    stage, `dispersive_regions`; `translate/stage.py`: the bundle row
    `dispersive_kerr_densities`; `translate/problem.py`: schema 24 and the leaf's
    validation.
  - `backends/numpy_ref/passes.py`: `dispersive_kerr_polarisation`;
    `backends/driver.py`: `refuse_dispersive_kerr`, used by `numpy_ref` (route
    refusals) and `numba_cpu` (the leaf); solprop-kokkos: a refusal by name in
    `_refuse_unsupported` and the native build record at schema 24.
  - `driver/simulation.py`, `driver/material_output.py`: the capillary routes build the
    response; the GNLSE and general modal routes refuse a dispersive policy; the
    manifest records the law.
  - Tests: `tests/physdata/test_dispersive_kerr_model.py`,
    `tests/nonlinear/test_dispersive_kerr_response.py`,
    `tests/translate/test_dispersive_kerr_terms.py`,
    `tests/backends/test_dispersive_kerr.py`, `tests/driver/test_dispersive_policy.py`,
    and the Kokkos refusal in solprop-kokkos `tests/test_backend.py`.
  - `docs/dev/kernel-contract.md` (schema 24, the leaf, the bundle row), the
    `CHANGELOG.md` entries of both repositories, and one-line updates of the solver and
    materials pages.

#### Decisions

- **The leaf carries sampled multipliers, not coefficients.** `DispersiveKerrTerms`
  holds each channel's multipliers on the problem's fine spectral layout — `g`, or the
  polynomial's bounded leg weight, its two output multipliers and its two constant
  product weights — with `Π₊` and the static coefficient, so no backend evaluates
  material physics. The pass and the reference share `dispersive_cubic`; the
  independent check of that arithmetic is Phase 0's triplet sum.
- **The band projection of §4.5 is the existing one.** The state is masked to the active
  band and the prefactor is zero outside it, so inputs and output are projected alike;
  the multipliers are evaluated on the whole fine grid, bounded there by the leg bound.
- **One electronic response per population.** A constant `KerrResponse` and a dispersive
  one on the same label are refused; on different labels both stages run. All
  electronic responses of a problem agree on `thg` (one harmonic sector) and all
  dispersive ones on their evaluator (one law). Raman's field-grid drive form reads the
  dispersive sector as it reads the constant `thg`.
- **The policy is the run's, not the caller's response alone.** A constant
  `KerrResponse` declared beside a dispersive selection is refused, so a record never
  names one law and propagates another; `from_gnlse` (a fitted `gamma`) and the general
  modal route (regional Kerr tensors) refuse a dispersive policy by name.
- **Regions and the record.** At translation each dispersive population's active band,
  `ω_min/2` to `1.1 ω_max`, is classified per frequency: supported (within `ν_c` and
  inside the band edge), bounded (beyond `ν_c`, inside the edge) and best effort
  (beyond the edge, or beyond `ν_c` where the edge is undefined), with the
  any-polarisation edge when the node field has two components. A band beyond the
  support is logged once at INFO, naming the gas and the band; nothing is refused. The
  manifest's `electronic` entries record evaluator, sector, tensor approximation, model
  and support digests, leg bound, both band edges and the regions;
  `policies["electronic_response"]` names the law with each population's support digest
  and regions, and the support record joins the manifest's records — all in the resume
  key.
- **No output-schema bump.** The persisted layout is unchanged: the new content lives in
  the manifest's existing `policies`, `electronic` and `records` entries, a carrier run's
  record and digest are bit for bit what they were, and older files keep resuming. §9
  expected a bump; John Travers may still prefer one (open question 3).

#### Deviations from Part I

- **Mixtures are the exact per-species sum.** Each population is its own channel with
  its own multipliers, leg bound and density: `P = Σ_s ε₀γ₀,s ρ_s cubic_s(E)`. §6's
  mixture-curve symmetric filter and aggregated polynomial coefficients are not built;
  they become Phase 2 optimisations to be measured against this sum, which is the
  reference they approximate.
- **Kokkos refuses the leaf in this phase**, by name, as §11 allows while a backend's
  phase is open; Phase 3 implements it.

#### Verification

All on macOS (Apple silicon), Python 3.14.5, optmat 0.4.0; tolerances are Phase 0's `1e-12`
(decision Q1) unless stated.

- **The response against the triplet sum** (`direct_sum`), on dimensioned grids with
  argon's model and the bound active (bin 12 at 200 nm, `ν_c` at 215.9 nm): both
  evaluators, both sectors, scalar and isotropic vector, field frame and envelope frame
  at two carriers — worst relative error 2.4e-15 over 24 cases.
- **Zero dispersion:** the full cubic equals `KerrResponse` to two FFT round trips; the
  conserving sector equals `thg=False` below an octave (field and envelope) and departs
  from it beyond. The whole `numpy_ref` stage derivative equals the constant cubic's to
  4.3e-16 (full) and 9.1e-16 (conserving) on the mode-averaged capillary, and to
  4.2e-16 on the radial route.
- **The pass against the reference:** the compiled pass equals
  `DispersiveKerrResponse.evaluate` bit for bit on the capillary's node field, field and
  envelope grids, both sectors, both evaluators; a He/Ar mixture's pass equals the sum
  of its species' evaluations to 4e-16 of the largest sample.
- **Work and energy:** the response does no net work (both evaluators, both sectors,
  below 1e-12). Ten fixed DOPRI5 steps over 2 mm of argon at 2 bar, lossless, no
  absorber, 10 fs and 30 µJ at 800 nm: the conserving sector conserves energy to
  1.4e-14; the full cubic to 1.5e-9 at ten steps and 1.0e-12 at twenty, the
  integrator's error falling with the step.
- **Open question 4 of Phase 0:** on `EnvelopeGrid.from_limits(..., thg=True)` (150 nm
  to 4 µm) a super-octave field (250 nm, 800 nm and 2.5 µm) evaluated in the full sector
  agrees with a four times finer evaluation across the active band to 1.2e-14 (carrier
  800 nm) and 7.5e-14 (carrier 400 nm): the harmonic grid holds the one-into-three
  group, as it holds the constant cubic's harmonic.
- **Material:** the SI model equals optmat's `gamma3` for the Kerr and harmonic processes
  to 1e-14 (measured 3.3e-16) for all fifteen gases; the leg bound changes no quartet
  within the fitted sums; regions: helium to 90 nm reaches all three (bounded 120.3–
  151.9 nm), argon's DUV band to 1.1 × 210 nm only supported and bounded, argon's
  any-polarisation edge 192.1 nm for a vector run, N2 and SF6 best effort as stated.
- **Refusals by name:** `numba_cpu` and Kokkos refuse `responses.dispersive_kerr`;
  `numpy_ref` refuses it on the general modal, slab and free routes; a supplied species
  and air under a dispersive policy, a constant Kerr beside it, the GNLSE and the general
  modal driver routes are refused; changing the policy, the evaluator or the sector
  changes the resume key, and `check_cache` refuses naming the changed
  `materials.policies` entries.
- **Runs** (`pytest -n auto`, Kokkos CPU installed from this branch): the five new
  modules, 103 tests (with Phase 0's algebra and the SEP checks, 267) in 15 s; `tests/backends` 2660 passed (after clearing a stale
  numba disk cache, below), `tests/translate`, `tests/nonlinear` and
  `tests/physdata` 3320 passed, `tests/driver`, `tests/output`, `tests/physdata` and
  `tests/test_seps.py` 2924 passed; solprop-kokkos's whole suite against this branch,
  234 passed and 4 skipped. Two `numba_cpu` vector-Raman cases
  (`test_vector_complete_run_matches_reference_and_modal_routes`, SF6, analytic field)
  failed once against a numba disk cache the session had filled under xdist; they pass
  on the base commit and here after the worktree's `.nbi`/`.nbc` files were removed,
  and no code they run changed. `ruff`, `basedpyright` (no errors in the changed files), `numpydoc lint` and
  `codespell` on the changed files.
- **Cost (indicative only):** the `numpy_ref` stage on the 2 mm argon capillary, one
  thread, same-process medians on a busy macOS workstation: symmetric 2.5–5.8 times and
  polynomial 6.8–8.6 times the constant cubic's stage. The reference is written for
  clarity — full complex transforms, allocations in every product — and decision Q5's
  targets (1.3 and 4 times) are for the optimised stages of Phase 2, where they are
  measured.
- **Not run:** no full gate (`poe check`, slow lane, coverage, docs build) — they run at
  the end of the SEP; no Numba or Kokkos execution of the response; no demonstration
  case.

#### The rest of this report

- **Sources checked:** optmat 0.4.0's `HyperpolarizabilityModel.support`
  (`HyperpolarizabilitySupport`: `leg_max_in_w2`, `band_edge`, `model_digest`,
  `identity`) and `AU_HYPERPOLARIZABILITY`, read from the installed package; Part I
  and the Phase 0 report. No new literature.
- **Limitations:**
  - The reference evaluates every product with full complex transforms; its cost is
    well above the targets of Q5 (above).
  - The region classification is per frequency of the active band; it does not
    identify which quartets a field actually drives.
  - On an envelope grid the harmonic groups can still wrap when the field itself
    reaches the edges of the active band, as the constant cubic's harmonic does; the
    measurement above is for fields inside the band.
  - The manual chapter, the tutorial and the demonstration cases are slice 1c's.
- **Open questions:**
  1. Phase 0's open question 4 is answered above for fields inside the active band; the
     lead may want the edge case (content in the collars) written into the manual.
  2. Q5's targets are missed by the reference, as expected; Phase 2 measures the
     optimised stage. Confirm that the reference need not meet them.
  3. Whether to bump the output schema although the persisted layout did not change
     (§9 expected one); a bump would make every schema-15 file unresumable.
  4. The policy check refuses a constant `KerrResponse` beside a dispersive selection on
     every route, including the spatial ones whose responses are the caller's; confirm
     that this strictness is wanted.
- **Handoff:**
  - **Slice 1b** adds the general modal route (its regional responses and
    `kerr_form`), the slab and the Cartesian free-space pipelines: each evaluates
    `responses.dispersive_kerr` per node through `dispersive_cubic`, and its refusal in
    `refuse_dispersive_kerr` and in the driver's `_refuse_dispersive_policy` for the
    general route is removed as it lands.
  - **Slice 1c** writes the nonlinear manual chapter (both operators, the sectors, the
    regions, the record), the tutorial with the argon DUV and helium RDW cases, and
    their demonstration tests.
  - **Phase 2** (Numba) implements the leaf against `numpy_ref`, with the mixture-curve
    filter and aggregated polynomial coefficients measured against the per-species sum.

### Phase 1b — the remaining numpy_ref routes

- **Scope:** slice 1b of Phase 1 (§11), 2026-10-07, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-1` (solprop only;
  solprop-kokkos unchanged). `numpy_ref` evaluates `responses.dispersive_kerr` on every
  route it runs: the Cartesian free-space and line pipelines, the hybrid planar
  pipeline and the general modal route, where a region's gas fill takes the run's
  electronic policy. `numba_cpu` and Kokkos still refuse the leaf by name. Slice 1c
  (the manual chapter, the tutorial, the demonstration tests) runs in parallel.
- **Delivered:**
  - `backends/driver.py`: `refuse_dispersive_kerr(problem, backend)` refuses the leaf on
    a backend without it; `numpy_ref` no longer calls it and refuses no route.
  - Free space, line and planar: no code change beyond the refusal. `_nonlinear_free`
    and `_nonlinear_slab` already run the fused response pass, with the stage bundle
    and the harmonic carrier, at every node of a chunk, so the dispersive stage of
    slice 1a is evaluated per field point, between the unchanged spatial transforms.
  - `modes/material.py`: `GasKerr(fill)`, a region's gas fill as its electronic
    response (a `Mixture` of constant densities); `MaterialRegion.gas`; a region declares
    one electronic response, `MaterialKerr` or `GasKerr`, and a `SilicaTensorResponse`
    still needs a `MaterialKerr`. Exported from `solprop.modes`.
  - `translate/general.py`: `general_modal_route(..., selection=)`. `_regional_responses`
    resolves each gas region through `electronic_response` at the carrier
    `2 pi c / omega_reference`: a carrier policy adds it to the constant regional cubic
    with `chi3 = sum_s N_s gamma3_s(lambda_0)`; a dispersive policy makes each species a
    channel of the dispersive leaf, labelled `"<region>:<population>"`, with the region
    as its node region, `kerr_form="real"` its full cubic and `"analytic"` its
    number-conserving sector, under the same harmonic-room rule. The route carries the
    leaf (`GeneralModalRoute.dispersive`) and the responses it was sampled from
    (`electronic`); the guide's identity names the fill (key `gas`, written only for a
    gas region) and the modal identity the channels (key `dispersive`, written only
    when present). Each channel's spectral regions are logged once at INFO with the
    any-polarisation band edge.
  - `translate/responses.py`: `DispersiveKerrTerms.regions` (int32, one node region per
    channel, present exactly on the general modal route), `dispersive_terms(...,
    regions=)`, and `compile_responses(..., regional_dispersive=)`;
    `translate/problem.py`: the leaf's regions validated against the node rule, int32
    admitted for them.
  - `backends/numpy_ref/general_modal.py`: `dispersive_polarisation`, called from
    `general_source` with the stage bundle's densities, adds each channel's
    `coefficient rho cubic(Re E_a)` at its region's nodes, by `dispersive_cubic`, to the
    polarisation the regional tensors give; the projection is unchanged.
    `backends/numpy_ref/admission.py` admits the leaf beside the general route's tables
    when it names regions.
  - `driver/simulation.py`: the general modal entry point passes its selection to the
    route and no longer refuses a dispersive policy; the manifest takes the
    any-polarisation edge for a general route's channels.
  - Tests: `tests/backends/test_dispersive_kerr_routes.py` (39); the three route
    refusals of `tests/backends/test_dispersive_kerr.py` are replaced by it.
  - `docs/guide/waveguides/general-modal.md` (the gas region), `docs/dev/kernel-contract.md` (the
    leaf's `regions`, every route on `numpy_ref`), one changelog line.

#### Decisions

- **A gas region is declared by its fill, `GasKerr`.** The general modal route had no
  gas region: its regions' laws are linear indices and their responses one absolute
  `chi3`. The slice's scope, "a region whose material is a gas optmat fits", is read as
  a region declaring its gas fill as its electronic response, beside its own linear
  law, rather than a new permittivity law: the providers' material-agreement checks are untouched,
  and a carrier-policy run of the same guide is the constant cubic every other route
  evaluates for a gas. Proposed, not yet agreed (open question 1).
- **The leaf is amended at schema 24.** `regions` joins `DispersiveKerrTerms` without a
  new schema version: schema 24 has not left this branch, and Kokkos refuses the whole
  leaf by name whatever its fields. The general route reuses the population leaf and
  its stage-bundle densities rather than adding a regional leaf, so every backend that
  refuses the leaf refuses it on this route too.
- **The general route filters in time, as the reference does.** The route's
  frequency-domain synthesis would let the multipliers act on the synthesised spectrum
  before the inverse transform. The
  reference instead takes `Re E_a` and calls `dispersive_cubic`, so its arithmetic is
  the response's own; applying the filter during synthesis saves one transform per
  node and component and is Phase 2's to measure.
- **The vector sector is the isotropic Kleinman cubic on three components (Q3).** On
  the real field it is the real form's tensor and in the conserving sector the analytic
  form's, which is what makes the zero-dispersion limit exact on this route.
- **A mixture region is one channel per species**, as on the population routes; two
  regions of one gas are two channels.

#### Deviations from the slice's scope

- **"A gas core and a constant solid region."** The step-index provider guides only
  with its core above its cladding, so the tested guide is a 1.5 µm strand of a
  constant solid (`MaterialKerr`) in an argon exterior (`GasKerr`); no shipped provider
  has a gas core.
- **The general route's reference at a node is `evaluate`'s arithmetic, not
  `evaluate`.** `DispersiveKerrResponse.evaluate` admits the state's one or two
  components and a general node has three; the test applies its expression,
  `(rho coefficient) dispersive_cubic(E, weights)`, with the response's own weights.
- **Energy.** None of the three routes' energy measures is conserved exactly by the
  constant cubic itself (below), so a lossless run is held to the constant cubic's
  change on the same route rather than to zero.

#### Verification

All on macOS (Apple silicon), Python 3.14.5, optmat 0.4.0, `numpy_ref`, ten fixed
DOPRI5 steps where a run is stepped.

- **One evaluation at a node, bit for bit:** the pass the free pipeline calls equals
  `evaluate` at the brightest node for the field grid's full and conserving sectors
  and the envelope grid's conserving sector, both evaluators; the planar pipeline, both
  sectors and evaluators; the general route, both forms and evaluators, at the gas
  region's brightest node, with the solid core's nodes untouched.
- **Zero dispersion against the constant cubic, whole stage derivative**, as a fraction
  of the largest bin: free 2.9e-16 (full) and 4.9e-16 (conserving), line 2.9e-16 and
  2.5e-16, planar 1.8e-16 and 3.1e-16, general modal 1.2e-16 (real) and 2.3e-16
  (analytic, against a region declaring `chi3 = N gamma3_0`). The conserving sector
  is compared on a launch confined to 300–450 THz: a Gaussian's tails reach beyond an
  octave at the 1e-6 level, and there the sector departs from the `thg=False` rule by
  3.7e-6 of the largest bin, as decision Q4 says it must.
- **Two laws in one guide:** a solid core (`chi3 = 1e-29` m²/V², weak so that the gas
  is a fifth of the source) and a gas exterior give the sum of each alone to 1.4e-15.
  Under the carrier policy the gas region is the constant cubic with `N gamma3(835 nm)`
  to 1e-14; a supplied species without optmat's model is refused under a dispersive
  policy naming the region and the population, and runs under the carrier one.
- **Energy:** each route's own measure moves for the constant cubic by an amount
  independent of the step and proportional to the pulse energy — the free route's
  paraxial `eps0 c / 2 int |E|^2` against the `k_z`-weighted flux (-1.40066e-11 for a
  15 fs, 3 µJ beam; a beam three times wider moves it a hundredth as much), the planar
  route's truncated slab basis (6.484e-9), the general route's identity metric
  (-6.246e-12 for 20 µJ in the strand). The dispersive runs move it by -1.40060e-11,
  6.493e-9 and -6.253e-12 (both sectors): within 1% of the constant cubic's, which is
  the test's bound.
- **Refusals by name:** `numba_cpu` and Kokkos refuse `responses.dispersive_kerr` on the
  free, planar and general routes (six cases).
- **Runs** (`pytest -n auto`, Kokkos CPU installed from the branch): the new module, 39
  tests in 17 s (19 s serially, the slowest 4.0 s; the four energy tests are
  `medium`); `tests/backends` route and general modules with slice 1a's, 410 passed;
  `tests/translate`, 1168 passed; the general modal, policy, manifest, moving, GRIN,
  split-block, solid multimode, waveguide and resume driver modules, 182 passed;
  `tests/modes` with the general modal and manifest driver modules, 1039 passed. Two
  `numba_cpu` cases of `test_free_nonlinear.py` failed once in a combined `-n auto` run
  and passed on rerun, alone and combined; no code they run changed, and slice 1a saw
  the same with the numba disk cache under xdist. `ruff`, `basedpyright` (no errors),
  `numpydoc lint` and `codespell` on the changed files.
- **Cost (indicative only):** the `numpy_ref` stage, one thread, same-process medians on
  a busy workstation, against the constant cubic on the same route: free 1.5 times
  (symmetric) and 2.3 times (polynomial), planar 3.0 and 5.9, the general strand 5.7
  and 6.5. The reference allocates in every product; Phase 2 measures the targets.
- **Not run:** no full gate, slow lane or coverage lane (the SEP's end); no
  solprop-kokkos suite (unchanged); no moving general guide with a gas region (its
  route takes the same regional compile).

#### The rest of this report

- **Sources checked:** Part I, the Phase 0 and Phase 1a reports, and the code of
  slice 1a. No new literature or material data.
- **Limitations:**
  - The memory ledger counts none of the dispersive response's buffers, on any route.
  - A gas region's density is constant along z, as a general guide's geometry is; a
    moving guide's gas region is compiled but not tested.
  - The general route's energy measure, like the free and planar ones, moves with the
    nonlinearity by itself; the tests bound the dispersive response's share of it, not
    the measure.
- **Open questions:**
  1. *Resolved:* `GasKerr(fill)` in `solprop.modes` is the way a general modal region
     declares a gas (accepted by John Travers, 2026-10-08).
  2. *Resolved:* the unreleased schema-24 leaf keeps `regions` without a new schema
     version (accepted, 2026-10-08).
  3. *Resolved:* `DispersiveKerrResponse.evaluate` admits three components (accepted,
     2026-10-08; done at the Phase 1 merge, with the triplet-sum test extended to three
     components).
- **Handoff:**
  - **Slice 1c** may now run its demonstrations on any `numpy_ref` route; a general
    modal example needs a `GasKerr` region.
  - **Phase 2** (Numba) implements the leaf on every route, the general route's
    `regions` included, with the in-synthesis filter of the general route and the
    memory-ledger rows measured there.

### Phase 1c — manual, tutorial and demonstration tests

- **Scope:** slice 1c of Phase 1 (§11), 2026-10-07, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-1c`, from slice 1a's head.
  The manual chapter, the quick tutorial on both demonstration cases of §2 and their
  short demonstration tests (§7, §8). No package code changed.
- **Delivered:**
  - `docs/guide/media/dispersive-kerr.md`, the manual chapter: one susceptibility for every
    mixing process; why an output-frequency coefficient is wrong; the two evaluators,
    their forms, error and cost; the harmonic sectors and how `thg` and `kerr_form`
    select them, with the envelope edge case of Phase 1a's open question 1; the leg
    bound, the band edge and the three regions, with every gas's figures read from
    optmat 0.4.0's support records; how pump dynamics and VUV emission share one
    response (§4.7); the demonstration-case coefficient table and a map of the
    symmetric model's error over degenerate-pump FWM in argon and helium with the
    fitted-sum, leg-bound and band-edge boundaries; mixtures (the exact per-species
    sum); vectors (the isotropic Kleinman tensor); choosing an evaluator and reading
    the manifest; limitations.
  - `docs/guide/media/nonlinear.md` (a section, the traits table and the mixture rule),
    `materials.md` (the electronic policies and a section on the dispersive ones),
    `gnlse.md` and `general-modal.md` (material dispersion, modal dispersion and the
    UPPE frequency factor kept apart), `docs/guide/workflows/luna.md` (a capability beyond Luna, whose
    $\gamma_3$ is one constant per gas).
  - `docs/tutorials/capillaries/plot_11_dispersive_kerr.py`: the argon DUV FWM case and
    the helium RDW case under the three policies.
  - `tests/driver/test_dispersive_demonstrations.py`: 15 tests, `medium`.
  - The `CHANGELOG.md` entry of slice 1a extended; this report.

#### Decisions

- **The argon case.** A 50 µm-radius capillary, 0.5 m, a 400 nm pump and an 800 nm
  seed of 10 µJ and 30 fs each, the grid's reference at the pump, wavelengths from
  210 nm so that the active band (to 190.9 nm) stays inside argon's 189.4 nm band edge.
  The radius is a choice of this tutorial, not Durfee et al.'s geometry. The pressure
  is the linear phase matching of the HE₁₁ mode, the zero of
  $2\beta_p-\beta_i-\beta_s$ formed from `MarcatiliMode.neff_minus_one` with `brentq`:
  209.4 mbar (93.1 mbar at 75 µm, 52.3 at 100 µm). The coefficient table of §4.7 is
  computed in the tutorial from `DispersiveKerrModel` and reproduces §4.7's argon and
  helium rows to the printed digits.
- **The helium case is not reduced.** `plot_02_rdw_emission.py`'s case — 1 m, 1.2 bar
  to vacuum, two modes, 100 nm to 1.5 µm, 200 fs — runs in 8–11 s per policy on
  `numpy_ref`, so it is used whole, changed only in backend and policy.
- **Every run selects `backend="numpy_ref"`,** the carrier runs too, so the
  comparison is on one backend; the default would choose Kokkos, which refuses the
  leaf by name.
- **The demonstration tests run ten fixed steps over 2 mm** with the split-step
  absorber scheme: the default collar of a guide that short is too stiff for an
  explicit step of a tenth of it (the run refuses with a stability warning otherwise).
  The helium test takes M2's geometry, grid, pulse and two modes at its constant
  entrance pressure, not the gradient.
- **Zero dispersion through the whole driver.** optmat's `with_parameters` model has
  no support record of its own, which `DispersiveKerrModel.from_optmat` refuses, so the
  test substitutes an equivalent constructed model at the material layer
  (`OptmatKerr.dispersive_model`, monkeypatched): `a = b = b' = 0`, the static value
  the carrier's Kerr coefficient, the real model's leg bound and edges.
- **The near-infrared bound is computed, not quoted.** §4.7's survey procedure
  (`default_rng(4)`, uniform wavelengths, output kept in band) on the run's own active
  band, 545 nm to 2.4 µm, gives the symmetric model's largest sampled error, 0.34 %,
  below §4.7's 400–2000 nm argon maximum of 0.913 %; the test compares the evaluators'
  nonlinear change of the field (the run minus a linear one) against it.

#### Deviations from Part I

- §8's "domain maps of §7" are one figure, the symmetric model's error over
  degenerate-pump FWM in two gases with the region boundaries. Maps of the fit's
  uncertainty and of other quartet families are not drawn; they belong with Phase 4's
  application guidance.
- The full-size M2 comparison in the worked example and the optional argon worked
  example (§8) are not done: they are Phase 4's, run by hand.

#### Verification

All on macOS (Apple silicon), Python 3.14.5, optmat 0.4.0, Kokkos CPU installed from
the slice 1a branch (not used by these runs).

- **Demonstration tests** (`pytest -n 4`, 13 s; the module's fixture runs the six
  demonstration runs once per worker, 6.3 s): 15 passed.
  - Both cases under all three policies complete, finite, on `numpy_ref`.
  - Regions: argon supported and bounded only (bounded 190.9–215.9 nm); helium all
    three, supported to its 151.9 nm leg bound and best effort below its 120.3 nm edge.
  - With the dispersion zeroed, both evaluators reproduce the carrier run's exit field
    to 4e-16–8e-16 of its largest bin (bound 1e-13); the real models move it by 4e-4 to
    1e-3, asserted above 1e-5.
  - Near-infrared window (2 cm of 1 bar argon, 20 µJ, 30 fs; the nonlinear change is
    0.20 of the linear field): symmetric and polynomial differ by 6.3e-6 of the
    nonlinear change, against the bound of 0.34 %.
- **Tutorial**, 39.6 s in the docs build and 41 s run as a script, on `numpy_ref`,
  deterministic (no noise):
  - Argon: the 250–285 nm signal peaks at 3.17 µJ at 0.44 m under the carrier policy
    and at 2.62 µJ at 0.32 m under both dispersive ones; at the exit 3.08 µJ against
    2.15 µJ (symmetric) and 2.17 µJ (polynomial). The dispersive runs lead by 7–8 % in
    the first centimetres, as the square of the generating quartet's 4 % predicts, and
    then saturate earlier: the signal's cross-phase modulation by the pump is 20 %
    stronger than the carrier's.
  - Helium: the 100–140 nm band at the exit carries 3.63 µJ (carrier), 3.98 µJ
    (symmetric) and 3.97 µJ (polynomial), −8.5 % for the carrier; its centre moves from
    133.2 to 134.2 nm.
  - Per-run cost: argon 7.1 / 8.5 / 10.4 s, helium about 7.5 / 9 / 11 s (carrier /
    symmetric / polynomial), on a shared workstation.
- **Step tolerance:** the tutorial's argon and helium runs under the carrier and
  polynomial policies at the default `rtol=1e-6`, `atol=1e-10` and at `1e-8`, `1e-12`:
  the exit signal and RDW-band energies agree to the fourth or fifth digit (argon
  3.0779 and 2.1694 µJ at both; helium 3.6348 against 3.6347, and 3.9738 at both), so
  the policy differences are the models', not the stepper's.
- **Docs build** (`poe docs`, every tutorial executed): the first build failed on two
  warnings from slice 1a's `dispersive_cubic` docstring, which named the private
  `_isotropic` and `_product_spectrum` with `:func:` roles that cannot resolve; they
  are now literals (`src/solprop/nonlinear/responses.py`, docstring only), and the
  rebuild succeeds with no warning. The tutorial took 39.6 s in the build, the longest
  of the gallery (the next is 6.6 s): every other tutorial runs on Kokkos, this one on
  `numpy_ref`.
- `ruff`, `basedpyright` (the test module), `codespell` and the public-tree scan on the
  changed files; `tests/test_seps.py`, 127 passed.
- **Not run:** grid convergence of the tutorial's runs; no Numba or Kokkos execution;
  no full gate.

#### The rest of this report

- **Sources checked:** optmat 0.4.0's support records, through
  `DispersiveKerrModel.from_optmat`, for every leg bound and band edge the manual
  quotes (all fifteen gases, both polarisations); Part I's §4.7 tables, recomputed.
  Durfee et al. [9] is cited for the process only; its geometry was not used. No new
  literature.
- **Limitations:**
  - The tutorial's runs are converged in step tolerance (above) but not checked on a
    finer grid or wider window.
  - The demonstration tests are ten-step functional checks; they do not test
    conversion, phase matching or the RDW.
  - The manual's statements about the general modal route (its `kerr_form` mapping, its
    refusal) describe this branch; slice 1b changes them.
- **Open questions:**
  1. In a one-mode exploration of the helium case (not in the tutorial) the 100–140 nm
     band at the exit differed by about 60 % between the carrier and the dispersive
     policies (2.2 against 3.5 µJ), against 9 % with two modes: the sensitivity
     depends strongly on the mode set. Phase 4's full-size comparison should report it
     per mode count.
  2. *Resolved:* the 40 s tutorial, the gallery's slowest by a factor of six, was
     accepted by John Travers on 2026-10-08 until a compiled backend runs the
     dispersive response; Phase 2 moves it off `numpy_ref`.
- **Handoff:**
  - **Slice 1b**'s routes were reconciled with these pages at the merge of the two
    slices: `dispersive-kerr.md`'s harmonic-sector paragraph and limitations,
    `materials.md`'s route sentence, the `general-modal.md` paragraph and the
    CHANGELOG entry now state the general modal, free-space and planar routes.
  - **Phase 4** reruns `docs/examples/processes/plot_00_vuv_dispersive_wave.py` with
    the three evaluators at full size, and may add the argon worked example.

### Phase 2a — numba_cpu's fused path

- **Scope:** slice 2a of Phase 2 (§11), 2026-10-08, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-2` (solprop only;
  solprop-kokkos unchanged at kernel schema 24, which it still refuses by name).
  `numba_cpu` executes `responses.dispersive_kerr` on its fused node path: the
  single-mode, mode-averaged multimode and radial routes, field and envelope grids, both
  evaluators, both sectors, one or two components, mixtures as Phase 1's exact
  per-species sum, beside Raman, field plasma and envelope plasma. Slice 2b adds the
  free-space, planar and general modal routes, §6's mixture optimisations and the
  tutorial's move off `numpy_ref`.
- **Delivered:**
  - `backends/numba_cpu/dispersive.py`: `DispersiveWorkspace` (buffers, plans and the
    per-stage `prepare`, `scale`, `respond_chunk`, `finish`, `finish_coarse`,
    `memory_ledger`), `DispersiveLayout`, `DispersiveMatrices`, the product-kernel
    factories `make_dispersive_self` and `make_dispersive_pairs` (closure-specialised,
    never disk-cached), the spectral kernels `filter_spectra`, `weigh_terms`,
    `combine_spectra`, `add_windowed`, `apply_window`, and `DISPERSIVE_LEDGER_ROWS`.
  - `backends/bodies.py`: twelve bodies of the sector products — the isotropic
    contraction, its number-conserving and harmonic groups for two components, and the
    closed forms of `(u, u, u)`, `(u, u, f)` and `(u, f, f)` for one, in each form.
  - `backends/numba_cpu/backend.py`: the workspace on the fused path
    (`_allocate_dispersive`, `_modal_to_time`, `_over_nodes`, `_on_accepted_step`), its
    ledger rows and metadata (`dispersive_alone`, `dispersive_planning_seconds`);
    `kernels.column_spec` takes the analytic family for the conserving sector on a field
    grid; `backends/driver.py`: `refuse_dispersive_kerr` now refuses only the routes off
    the fused path.
  - Tests: `tests/backends/test_numba_dispersive_kerr.py` (82) and
    `tests/backends/test_numba_dispersive_kernels.py` (36, in the coverage lane's
    interpreted pass, `pyproject.toml`); the product factories in
    `test_numba_cache_policy.py`; Phase 1's refusal tests narrowed to Kokkos and to
    `numba_cpu`'s remaining routes.
  - `benchmarks/sep6/dispersive_cost.py`, the Q5 measurement; the ledger entry in
    `docs/dev/benchmarks.md`; `docs/dev/backends.md` §3.11, its route table and limits;
    `docs/dev/kernel-contract.md` and `docs/guide/media/dispersive-kerr.md` on what runs where;
    a changelog line.

#### Decisions

- **The multipliers act on the modal columns, not at the nodes.** The legacy transform's
  synthesis and projection are real matrices independent of frequency, and the temporal
  collar is the same at every node, so `P G[(G E)^3] = G[P (G E)^3]`. The input
  multipliers act on the padded modal spectrum the stage combination already formed —
  the field spectrum before the stage's inverse transform, so the response forms no
  forward transform of the node field — and one batched inverse transform gives every
  filtered field; the products are formed at the nodes and projected by the same GEMMs
  as the polarisation; one batched forward transform, the output multipliers and the
  channels' densities summed in the spectrum, one inverse transform and the collar
  finish them on the columns. On the 33-node capillary rule that is a few transforms
  of a few columns where the reference takes them per node.
- **The collar stays after the output multiplier**, as the reference applies it to the
  summed polarisation (`w G[(G E)^3]`). It costs the output's forward and inverse
  transforms of the columns, which `G[w (G E)^3]` would fold into the stage's own; that
  fold changes the operator at the window's edges and is not this backend's to make
  (open question 1).
- **A lone response owns the node and projected buffers.** With no constant Kerr term,
  Raman, plasma or envelope plasma beside it, the response column and the
  polarisation's projection do not run and the response writes the projected buffer. A
  symmetric response of one species also takes the stage's forward plan and buffers for
  its product and, with no in-stage absorber reading the bare modal field, filters the
  stage's spectrum in place: no field-sized buffer of its own. Beside other responses
  the bare field stays theirs, so Raman and plasma are never filtered.
- **The embedded rule is finished lazily**: its products are projected at every stage
  and transformed when a step is accepted, from the last stage's buffers, once per
  evaluation, before the alarm transforms the embedded projection — the envelope
  currents' scheme.
- **The polynomial subtracts the bare analytic signal's zero-frequency constant.** It
  reads the bare field the analytic family synthesised, which keeps the zero-frequency
  bin once; `Π₊` excludes it. The state carries some: the split-step absorber's product
  in time puts content at every bin after the band mask. Without the correction one
  evaluation on a state with such content disagreed at 1e-6 (a ten-step plasma run at
  3.5e-7); with it, at a few ε. The symmetric evaluator's filtered fields carry `Π₊` in
  their multipliers.
- **The node budget is unchanged.** The response adds one node buffer (symmetric beside
  other responses) or two (polynomial) of the chunk's size, which `node_chunk`, sized at
  translation, does not count (open question 3).

#### Deviations from Part I

- **Q5's targets are missed in places** (below): the symmetric stage on one node and on
  a mixture, the polynomial on a mixture, and the memory targets beside Raman or plasma
  and for the polynomial's products. Q5 makes a miss an optimisation task, never a
  refusal; the causes and the measured remedies are in the open questions.
- The tutorial still runs on `numpy_ref`; moving it is slice 2b's, with the routes it
  needs.

#### Verification

macOS (Apple silicon, 18 cores), Python 3.14.5, optmat 0.4.0, pyFFTW 0.15.1 on FFTW
3.3.11, Kokkos CPU installed from the kokkos main checkout (schema 24, not exercised but
for its refusal).

- **Parity with `numpy_ref`** (`test_numba_dispersive_kerr.py`, relative L2; bounds from
  the transforms and node reductions both backends perform, asserted positive):
  - one evaluation's projected polarisation on the 33-node, three-mode argon capillary,
    both evaluators × field and envelope × both sectors: 2.3–4.0 ε on the fine rule,
    2.1–3.9 ε on the embedded one, against about 1600 ε (symmetric) and 2400 ε
    (polynomial); the whole stage source agrees as the constant cubic's does on the same
    problem (150–1400 ε from the prefactor's band edges, identical for the three
    policies);
  - one Runge–Kutta step (seven evaluations) on the same capillary: 0.9–1.7 ε against
    7400–13000 ε; ten replayed steps of the default integrator: 3.0–8.2 ε;
  - the quadrature alarm after ten steps, which reads the lazily finished embedded rule:
    alarms of 5e-15 and 7e-13 agreeing to 3.4e-17;
  - the mode-averaged route (one node), a two-component elliptical launch, the radial
    route on field and envelope grids (the latter on the real radial GEMM), a He/Ar
    mixture, nitrogen's Raman beside it (both sectors), field and envelope plasma beside
    it, five node chunks of eight, a state with zero-frequency content, and the
    Runge–Kutta march: one evaluation and ten steps each within their bounds; beside the
    plasma column the bound adds the carrier policy's own disagreement on the same
    problem (916 ε on the fine rule);
  - bitwise across two runs and across one and four threads with the parallel regime
    forced, with and without Raman.
- **Kernels** (`test_numba_dispersive_kernels.py`, compiled and interpreted): the closed
  forms against the general contraction and the reference's `_isotropic` groups to
  8 ε, the real products bit for bit; the product kernels against the reference's
  sector groups, every form, one and two components, with and without the offset, to
  16 ε of the largest sample; the spectral kernels against NumPy; one small evaluation
  end to end against `numpy_ref`.
- **Memory ledger** (tested): every `dispersive_*` row named at zero on every problem
  and released on close; a lone symmetric response holds no field-sized buffer beside
  the split-step absorber and one (its filtered modal field) beside the in-stage one;
  the polynomial two derivative fields, two node buffers, three projected products per
  rule and their spectra; beside Raman, one node buffer, one projected product per rule,
  their spectrum and its output.
- **Q5, cost** (`benchmarks/sep6/dispersive_cost.py` at `fec4e09`, the ledger entry of
  `docs/dev/benchmarks.md`): the RDW grid (8192 fine samples), a 125 µm helium
  capillary at 0.8 bar; same-process medians of seven interleaved rounds on a
  workstation at a load average of 16 to 30, so ratios only. Stage ratios to the
  constant cubic, one thread; default pools:

  | case | symmetric, thg / no thg | polynomial, thg / no thg |
  |---|---|---|
  | one mode-averaged node | 1.51 / 1.64; 1.58 / 1.55 | 2.94 / 3.55; 2.96 / 3.64 |
  | four modes, 33 nodes | 1.17 / 1.11; 1.28 / 1.20 | 2.56 / 3.45; 2.62 / 2.69 |
  | He/Ar mixture, four modes | 2.17 / 2.08; 2.34 / 2.05 | 4.04 / 5.67; 3.72 / 3.92 |

  Met: the symmetric evaluator on the multimode capillary, the polynomial on one
  species and on the mixture in the default pools. Missed: the symmetric evaluator on
  one node and on the mixture, the polynomial on the mixture at one thread. The ten-step
  windows, which carry the integrator's own per-step work, move by 1.04–1.62
  (symmetric) and 1.26–3.34 (polynomial). The isolated response — the constant cubic's
  column against the dispersive response's filter, transforms, syntheses, products,
  projections and output — is 4–10 µs against 40–71 µs (symmetric) and 109–240 µs
  (polynomial) on one node, 207–315 µs against 423–675 µs and 1010–2450 µs on four
  modes. An earlier measurement at a load average of 38, before the stage's views were
  made once, gave the same ratios within 0.1 but for the mixture's polynomial (4.25 and
  5.81; 3.72 and 4.11).
- **Runs** (`pytest -n auto`): the new modules, 118 tests in about a minute; with the
  `numba_cpu` modules, the cache policy, Phase 1's dispersive modules, `test_driver.py`,
  `test_bodies.py`, `test_memory.py`, `test_vector_raman.py`, `test_envelope_plasma.py`
  and `test_radial_nonlinear.py`, 1071 passed at the branch's head. An earlier run of
  the set saw one case of `test_numba_cpu_plasma_scan.py` (the blocked plasma column,
  which this slice does not touch) fail once and pass on rerun, alone and under xdist,
  as Phase 1 saw against a numba disk cache filled under xdist. The kernel module with
  `test_bodies.py` under `NUMBA_DISABLE_JIT=1`, 80 passed; `dispersive.py` 100 % covered
  by the new modules (interpreted kernels, compiled rest). The docs build with `-W -n`
  passes with the galleries unexecuted; executed, the planar-hybrid tutorial
  (`docs/tutorials/free_space/plot_04_planar_hybrid.py`) fails translation on the
  spatial collar's stability limit (a fixed 1 cm step against 6.96 mm), which nothing
  in this slice touches (open question 5). `ruff`, `basedpyright` (no
  errors) and the public-tree scan on the changed files.
- **Not run:** the full gate, the slow lane and the coverage lane (the SEP's end); no
  Linux or quiet-machine timing; no Kokkos change.

#### The rest of this report

- **Sources checked:** Part I §6 and §11, the Phase 0 and Phase 1 reports, the Phase 1
  code (`dispersive_cubic`, the `numpy_ref` pass, the leaf). No new literature or
  material data.
- **Limitations:**
  - The symmetric stage on one node costs about 1.5 to 1.6 times the constant cubic's,
    most of it the output's transform pair (open question 1).
  - A mixture costs a response path per species (Phase 1's exact sum).
  - Beside Raman or plasma the response holds its own node field, projected products,
    their spectrum and its output: four real modal fields on the full cubic and ten on
    the analytic family, against Q5's one. The polynomial's three projected products
    per rule and their spectra exceed its one product buffer (eleven and twenty-two
    units beside its two derivative fields).
  - The node chunk is sized without the response's node buffers.
  - Timings are same-process ratios on a heavily loaded workstation.
- **Open questions:**
  1. *Resolved (John Travers, 2026-10-08):* the collar is applied before the output
     multipliers, `G[w (G E)^3]` for `w G[(G E)^3]` and likewise for the polynomial's
     output weights, in the reference and the backends; slice 2b makes the change.
  2. *Resolved:* whether an optimisation slice is needed is decided on slice 2b's
     measurements, after the mixture optimisations land.
  3. *Resolved:* `node_chunk` counts the response's node buffers; slice 2b makes the
     change.
  4. *Resolved:* the split-step absorber's out-of-band content is raised as a separate
     issue, outside SEP 6.
  5. The planar-hybrid tutorial no longer translates (above); it needs shorter steps,
     `RungeKutta()` or a longer boundary, outside this SEP (raised separately).
- **Handoff:**
  - **Slice 2b** adds the free-space and line routes (their nodes are the columns, so
    the input multipliers can act on each column's spectrum before its temporal inverse
    transform, the per-node scheme of §6), the planar route (rows to time a block at a
    time, likewise), the general modal route (frequency-dependent synthesis: the
    in-synthesis filter of Phase 1b's decision, measured there), §6's mixture
    optimisations measured against the exact sum, and the tutorial off `numpy_ref`. It
    removes `refuse_dispersive_kerr` as the routes land.
  - **Phase 3** (Kokkos) can take this slice's structure: the multipliers on the modal
    columns, the lazy embedded rule and the zero-frequency offset.

### Phase 2b — the remaining routes, the decided changes and the mixtures

- **Scope:** slice 2b of Phase 2 (§11), 2026-10-08, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-2` (solprop only;
  solprop-kokkos unchanged at kernel schema 24, which it still refuses by name). The two
  changes decided on 2026-10-08 — the collar before the output multipliers, and the
  node budget counting the response — in the reference and `numba_cpu`; `numba_cpu` on
  the Cartesian free-space, line, planar and general modal routes, so that it runs the
  response wherever `numpy_ref` does; §6's mixture optimisations, the exact polynomial
  grouping adopted and the symmetric mixture-curve filter measured; the tutorial off
  `numpy_ref`; Q5 measured again. Phase 2 is complete with this slice.
- **Delivered:**
  - `nonlinear/responses.py`: `dispersive_cubic(..., window=)` and
    `DispersiveKerrResponse.evaluate(..., window=)` apply the temporal collar to each
    product in time, before `Π₊` and the output multipliers (`G[w (G E)^3]`); `None`
    is a unit collar, bit for bit the operator without one.
    `backends/numpy_ref/passes.py`: `dispersive_kerr_polarisation(..., window=)`, and
    `nonlinear_polarisation` adds the dispersive term after windowing every other
    response's, never windowing it twice. The general modal route has no collar.
  - `backends/numba_cpu/dispersive.py`: the response's windowed products are
    transformed and their weighted spectra land in the stage's own polarisation
    spectrum (`add_spectra`, `weigh_spectrum`), so it has no inverse transform; a lone
    symmetric response of one species projects into the stage's projected buffer and
    takes the stage's forward transform; `DispersiveTerms` and
    `dispersive_terms_layout` group the polynomial's species of equal leg bound and
    give each region (on the general route) its own cube; `StageWeights` weighs the
    channels and groups once per stage. The `dispersive_total_spectrum` and
    `dispersive_total_time` ledger rows are gone with the buffers.
  - `backends/numba_cpu/dispersive_spatial.py`: `SpatialDispersive` for the Cartesian,
    line and planar routes, per chunk of columns or block of rows.
    `backends/numba_cpu/dispersive_general.py`: `GeneralDispersive` and its
    three-component kernels (`make_general_dispersive`), the multipliers in the
    route's synthesis. `backends/bodies.py`: `dispersive_isotropic3` and
    `dispersive_conserving3`. `backends/numba_cpu/general_modal.py`: the general
    workspace runs the response and skips an identically zero regional contraction.
    `backends/driver.py`: `refuse_dispersive_kerr` removed.
  - `translate/problem.py`: `_dispersive_chunk`, the node chunk's admission of the
    response's buffers.
  - Tests: `tests/backends/test_numba_dispersive_routes.py` (new, 103);
    `test_numba_dispersive_kerr.py` (the spectral comparison, the krypton–xenon group,
    85), `test_numba_dispersive_kernels.py` (the new kernels and bodies, the layout,
    43), `tests/nonlinear/test_dispersive_kerr_response.py` (the collar's two orders),
    `tests/translate/test_dispersive_kerr_terms.py` (the node admission),
    `test_dispersive_kerr_routes.py` (the window in the node checks, the unit collar,
    Kokkos's refusal alone), `test_numba_cache_policy.py`.
  - `benchmarks/sep6/dispersive_cost.py` (the new routes and mixtures),
    `benchmarks/sep6/mixture_curve.py` (new); the ledger entry in
    `docs/dev/benchmarks.md`; `docs/dev/backends.md` §3.11, the route table and the
    limits; `docs/dev/kernel-contract.md`; `docs/guide/media/dispersive-kerr.md` (the collar,
    the cost, mixtures, backends), `materials.md`, `general-modal.md`, `solver.md`;
    `docs/tutorials/capillaries/plot_11_dispersive_kerr.py` on `numba_cpu`; the
    changelog.

#### Decisions

- **Where the collar sits in the conserving sector.** The decision places the collar on
  the cubic, before the output multipliers. In the conserving sector the cubic in time
  is the complex inner product of analytic signals, `Re{¼ Π₊[S]}`, and `Π₊` is applied
  in the spectrum with the output multipliers, as the Phase 0 forms apply it; so the
  collar multiplies `S`, before `Π₊`: `Re{¼ Π₊[w S]}` under `G`. This keeps the output
  strictly on positive frequencies and lets a backend finish the response in the
  stage's own transforms. With the dispersion removed, it is the constant `thg=False`
  rule exactly under a unit collar; under the free route's real collar it departs from
  it by 1.2e-9 of the largest bin, the collar's leakage below zero frequency that `Π₊`
  removes and the constant rule keeps (open question 1).
- **The general route keeps its absent collar**: it has none on either backend.
- **The polynomial's exact grouping is `numba_cpu`'s default**, as the slice's scope
  decided: a group is the channels whose leg-weight rows are equal bit for bit (and, on
  the general route, whose region is the same). `numpy_ref` keeps the per-species sum.
- **The node budget** counts, on the fused path, no node buffer for a lone symmetric
  response of one species, one beside other responses and two for the polynomial; a
  Cartesian or planar column adds two complex-equivalent buffers per product (in time
  and in frequency) to the free route's twenty. The general route sizes its chunk from
  its maps and is not changed (open question 4).
- **The mixture-curve filter is measured, not offered**, as the slice's scope requires: no
  policy, option or leaf selects it; the benchmark builds its one-channel leaf by hand.

#### Deviations from Part I

- §6 recommends the mixture-curve filter for symmetric mixtures; it is measured and
  left for a decision, not adopted (the slice's scope).
- §6's polynomial aggregation is adopted only for species of equal leg bound, the exact
  case. Air's three species have three leg bounds and gain nothing (open question 2).

#### Verification

macOS (Apple silicon, 18 cores), Python 3.14.5, optmat 0.4.0, pyFFTW 0.15.1 on FFTW
3.3.11, Kokkos CPU installed from the kokkos main checkout (schema 24, exercised only for
its refusal); a load average between 6 and 43.

- **The collar** (`test_dispersive_kerr_response.py`): a unit collar gives the operator
  without one bit for bit, both evaluators, sectors and grids. An 8 fs pulse in the
  window's flat middle gives the two orders to 5e-15–3.3e-12 of the largest sample; one
  at nine tenths of the half-window separates them by 3.2e-3–5.9e-3, of which the part
  inside the flat middle is 0.4–1.1 % of the part where the collar varies in the full
  sector and on envelope grids, and a third on a field grid's conserving sector, whose
  `Π₊` is a Hilbert transform. The free and planar routes' node checks equal
  `evaluate(..., window=)` bit for bit; the free route's zero-dispersion check runs at
  a unit collar (above).
- **Parity with `numpy_ref`** (relative L2; bounds from the transforms and reductions
  both backends perform, asserted positive):
  - the fused path (`test_numba_dispersive_kerr.py`, now comparing the stage's
    polarisation spectra, where the response joins): unchanged in kind from Phase 2a,
    every case within its bound, and the krypton–xenon polynomial, one group, one
    evaluation and ten steps in three sectors;
  - free space and the line, both evaluators, field full and conserving, envelope
    conserving and full: one evaluation 2.2–4.9 ε against bounds of 640–3100 ε, ten
    steps 0.65–1.5 ε; beside Raman 3.1–5.0 ε and 0.8–1.3 ε; a helium–argon medium; a
    partial chunk; one Runge–Kutta step;
  - the planar route, both sectors: 3.7–5.2 ε against 1800–3500 ε, ten steps
    0.78–1.3 ε; beside Raman 3.3–5.4 ε; beside the in-stage collar, which reads the
    bare rows; a partial block; one Runge–Kutta step;
  - the general route, gas alone, beside a solid core and a helium–argon region, real
    and analytic forms: 0.47–1.8 ε against 560–1100 ε, ten steps 0.53–1.0 ε; one
    Runge–Kutta step;
  - bitwise across runs and across one and four threads with the parallel regime forced
    on the fused, free-space, planar and general routes.
- **Kernels** (`test_numba_dispersive_kernels.py`, compiled and interpreted): the
  three-component bodies against the reference's `_isotropic` and conserving group to
  8 ε; the general product kernels against the reference at their region's nodes and
  exactly zero at another's; the layout's groups, cubes and terms; the spectral
  kernels against NumPy.
- **Node budget**: a chunk admits two nodes at twice the per-node bytes and refuses one
  byte short, for the four fused-path cases and a free-space column, as Raman's
  admission is tested.
- **Mixtures**: the krypton–xenon polynomial is one group (two filtered fields, three
  products) and agrees with the reference's per-species sum as the one-species cases do.
- **Runs** (`pytest -n auto`): the dispersive, route, general, free, planar, radial,
  cache-policy, driver, memory and bodies modules, 676 passed and 8 skipped; every
  `tests/backends/test_numba_*.py`, 1007 passed; `tests/nonlinear`, 1012 passed;
  `tests/translate`, 1174 passed and 3 skipped; the driver's dispersive modules, 25
  passed. The kernel modules under `NUMBA_DISABLE_JIT=1`, 87 passed. `ruff`,
  `basedpyright` (no errors) and the public-tree scan on the changed files. The test
  counts above preceded the in-place filter of the last commit, after which the route
  module (103) was run again.
- **The tutorial** on `numba_cpu` prints exactly Phase 1c's figures: argon's signal
  3.17 µJ at 0.44 m (carrier) and 2.62 µJ at 0.32 m (both dispersive), 3.08, 2.15 and
  2.17 µJ at the exit; helium's 100–140 nm band 3.63, 3.98 and 3.97 µJ, centres 133.19
  and 134.18 nm. Its time in the docs build: 15.9 s, against 39.6 s on `numpy_ref` (Phase 1c), at a load average
  near 10; run as a script under a load average of 43 it took 116 s of wall time for
  23 s of CPU time.

#### Q5 measured again

Same-process stage ratios to the constant cubic (`benchmarks/sep6/dispersive_cost.py`,
seven interleaved rounds, medians; the full table, the windows and the response's own
times are the ledger entry of `docs/dev/benchmarks.md`). One thread; default pools in
brackets where they differ by more than 0.1.

| case | symmetric, thg / no thg | polynomial, thg / no thg |
|---|---|---|
| one mode-averaged node | 1.12 / 1.31 (1.35) | 2.49 / 3.51 |
| four modes, 33 nodes | 0.97 / 0.92 | 2.58 / 3.52 (2.09 / 2.38) |
| He/Ar, four modes | 2.02 / 1.93 (1.78 / 1.68) | 4.16 / 5.98 (3.26 / 3.71) |
| Kr/Xe, one leg bound | 2.07 / 1.89 | 2.42 / 3.21 (2.27 / 2.68) |
| air, three leg bounds | 2.98 / 2.67 | 5.38 / 7.55 (4.32 / 4.77) |
| Cartesian, 16 × 16 | 0.96 / 1.14 | 2.19 / 3.17 (1.95 / 2.46) |
| line, 64 | 1.01 / 1.19 | 1.88 / 2.65 (1.69 / 2.13) |
| planar | 1.15 / 1.04 (1.21 / 1.23) | 2.56 / 3.72 (2.57 / 3.08) |
| general modal, real / analytic | 1.15 / 0.51 | 2.32 / 1.44 |

- **Met:** the symmetric evaluator for one species on every route (one node's analytic
  sector at 1.31–1.35, at the target, where Phase 2a measured 1.51–1.64); the
  polynomial for one species everywhere, and for krypton and xenon, now one group.
- **Missed:** the symmetric evaluator on every mixture, 1.68–3.00: the exact response is
  one product per species, which no backend can merge; the polynomial on mixtures of
  distinct leg bounds, helium and argon at one thread and air in every pool.
- The ten-step windows move by 0.98–1.90 (symmetric) and 1.17–3.87 (polynomial).
- Where the fold of the collar into the stage's own transforms was estimated (Phase 2a:
  1.15 times on one node, full cubic), it measured 1.09–1.12.

#### The mixture-curve filter, measured

`benchmarks/sep6/mixture_curve.py`: helium and argon, 50/50, and air as nitrogen, oxygen
and argon (78.08/20.95/0.97). The filter is `g_mix = [γ_K,mix/γ_K,mix(0)]^(1/4)` with
`γ_K,mix(ν) = Σ_s N_s γ_K,s(ν̃_s)`, each species' leg bounded inside its own term; for a
pure gas it equals that gas's own filter to 2.2e-16.

- **The coefficient** (§4.7's procedure: `default_rng(4)`, 300 000 draws on
  400–2000 nm, positive outputs in band, `S₂` within the smallest fitted maximum of the
  species; 208 109 quartets for each mixture), in per cent, median / 95th percentile /
  maximum:

  | mixture | curve vs polynomial sum | exact symmetric sum vs polynomial sum | curve vs exact symmetric sum |
  |---|---|---|---|
  | He/Ar | 0.052 / 0.488 / 0.903 | 0.052 / 0.480 / 0.887 | 0.0005 / 0.0077 / 0.0157 |
  | air | 0.078 / 0.728 / 1.350 | 0.077 / 0.708 / 1.305 | 0.0005 / 0.0206 / 0.0446 |

  The mixing error alone is fifty times below the factorisation's own: the curve is as
  good an approximation of the polynomial as the exact per-species symmetric sum is.
- **A short propagation** (`numba_cpu`, a 125 µm capillary at 1 bar, 3 cm, 30 fs and
  150 µJ at 800 nm, one mode, the default integrator), the departure of each exit
  field from the polynomial's as a fraction of the polynomial's nonlinear change
  (the field minus a linear run's), and the 250–290 nm band's share of the exit energy:

  | mixture | carrier | exact symmetric | curve | band: carrier / symmetric / curve / polynomial |
  |---|---|---|---|---|
  | He/Ar | 5.80 % | 0.27 % | 0.29 % | 1.58e-3 / 1.780e-3 / 1.780e-3 / 1.755e-3 |
  | air | 5.01 % | 0.29 % | 0.36 % | 1.67e-4 / 1.475e-4 / 1.502e-4 / 1.367e-4 |

- **The cost** (one thread, stage ratio to the constant cubic on that capillary): the
  curve 1.26 (He/Ar) and 1.24 (air), the exact symmetric sum 2.24 and 3.29, the
  polynomial 3.82 and 5.23.

So the curve costs what one gas costs and departs from the exact symmetric sum by about
a tenth of the symmetric factorisation's own error in these cases; John Travers may
decide whether to offer it (open question 3).

#### The rest of this report

- **Sources checked:** Part I §4.4, §4.7, §6 and §11; the Phase 0, 1 and 2a reports; the
  Phase 1 and 2a code. No new literature or material data; every leg bound and
  coefficient is read from optmat 0.4.0 through `DispersiveKerrModel`.
- **Limitations:**
  - A symmetric mixture costs one product per species, and a polynomial mixture one
    group per distinct leg bound; air's species have three.
  - The general route's node chunk is sized by its maps without the response's buffers:
    per chunk it holds the filtered fields, the products and their spectra, two
    complex-equivalent buffers per field and product of three components per node.
  - The general modal guide tested has a narrow 815–860 nm band, over which the
    multipliers vary little; the kernels are held to the reference on random fields,
    and the fused and spatial routes on broad bands.
  - The tutorial's wall time depends on the machine's load; it was measured on a busy
    workstation.
  - Timings are same-process ratios on a loaded workstation; no Linux or quiet-machine
    timing.
- **Open questions:**
  1. *Resolved (John Travers, 2026-10-08):* the conserving sector's collar acts on the
     analytic product before `Π₊`, as implemented.
  2. *Resolved:* no grouping by approximate equality; species group only when their leg
     bounds are equal, so the grouping stays an exact rewriting.
  3. *Resolved:* the mixture-curve filter becomes the symmetric evaluator's operator for
     mixtures, on every backend, recorded in the manifest; the exact per-species sum
     stays the test reference that measures it. Slice 2c implements it.
  4. *Resolved:* the general route's chunk counts the response's buffers too (slice 2c).
  5. *Resolved:* the polynomial's Q5 miss on mixtures of distinct leg bounds is a
     roadmap item, not a slice of this phase.
- **After the merge of main** (the split-step absorber now masks the band after its
  product, so an accepted state carries no out-of-band content): the zero-frequency
  offset the stage subtracts is kept as a guard. It costs one reduction per row, is
  zero on a clean state, and keeps the conserving sector's exclusion of the
  zero-frequency bin independent of how a state was produced.
- **Handoff:**
  - **Phase 3** (Kokkos CPU, then device) implements the leaf against `numpy_ref` and
    `numba_cpu`, taking this phase's structure: the multipliers off the nodes (on the
    modal columns, on each chunk's columns or rows, in the general route's synthesis),
    the collar on the products and the response's spectrum added to the stage's, the
    lone symmetric response on the stage's own transforms, the polynomial's grouping,
    the lazy embedded rule and the zero-frequency offset. The kernel schema is
    unchanged by this slice.
  - Phase 3 copies the structure after slice 2c (the mixture-curve filter).

### Phase 2c — the symmetric mixture filter and the general route's budget

- **Scope:** slice 2c of Phase 2 (§11), 2026-10-08, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-2c` (solprop only;
  solprop-kokkos unchanged at kernel schema 24, which it still refuses by name). The two
  decisions of Phase 2b's open questions 3 and 4: §6's mixture-curve filter as the
  symmetric evaluator's operator for a mixture, on `numpy_ref` and `numba_cpu` and every
  route, recorded in the manifest and the resume key, with the exact per-species sum
  kept as the test reference; and the general modal route's node chunk counting the
  response's buffers.
- **Delivered:**
  - `nonlinear/responses.py`: `mixture_filter(filters, factors)`, the operator's
    definition: `F = Σ_s f_s` and `g_mix = [Σ_s f_s g_s⁴ / F]^(1/4)` with
    `f_s = ε₀ γ₃,ₛ ρ_s`; one species returned unchanged; a unit filter at `F = 0`.
  - `translate/responses.py`: `DispersiveKerrTerms.groups()` (the symmetric evaluator's
    channels acting at the same nodes; the polynomial's of equal leg weight and region,
    the grouping `numba_cpu` already took) and `evaluations(densities)`, the cubics the
    reference evaluates at a stage with their factors. The leaf's fields are unchanged.
  - `backends/numpy_ref/passes.py` (`dispersive_kerr_polarisation`) and
    `general_modal.py` (`dispersive_polarisation`) evaluate `evaluations`: one cubic per
    symmetric group, one per polynomial channel.
  - `backends/numba_cpu/dispersive.py`: `DispersiveTerms` lays out one field and one
    product per symmetric group (`mixtures`, `quartics`, `input_scale`,
    `output_scale`); `StageWeights` holds the stage's own input and output rows, rebuilds
    a mixture's with the kernel `mixture_rows` and skips an update whose densities equal
    the last; the workspace weighs the channels in `prepare`, before it filters (the
    backend's `_modal_to_time(bundle)`), and a symmetric mixture takes the lone path.
    `dispersive_spatial.py` and `dispersive_general.py` read the stage's rows.
  - `translate/problem.py`: `_dispersive_node_buffers` counts groups.
    `backends/numpy_ref/general_modal.py`: `general_dispersive_buffers(leaf)` and
    `general_node_chunk(..., dispersive=)`, used by `_general_maps`, the chunk both
    backends take; `backends/memory.py` estimates the same chunk and, on `numba_cpu`,
    its dispersive bytes.
  - `driver/material_output.py`: `MIXTURE_FILTER_RULE`; `material_manifest(..., leaf=)`
    writes `policies["electronic_response"]["mixture_filters"]` for each symmetric
    group of several species -- the rule, the general modal region (`None` elsewhere),
    the composition as each population's label and the SHA-256 of its density path's
    record -- and each such population's electronic entry names its filter (`filter`);
    `driver/simulation.py` passes the translated leaf.
  - Tests: `tests/nonlinear/test_dispersive_kerr_mixture.py` (new, 11, with
    `exact_symmetric_sum`, the test reference); `test_numba_dispersive_kernels.py` (the
    layout, the stage's rebuild and the kernel, bit for bit against `mixture_filter`, 4);
    `test_numba_dispersive_kerr.py` (a mixture of varying composition, ten steps; a lone
    mixture's ledger, 3); `test_dispersive_kerr.py` (the `numpy_ref` pass, 1);
    `test_dispersive_kerr_routes.py` (a general mixture region bit for bit, the general
    chunk, the region's manifest, 7); `test_dispersive_kerr_terms.py` (the fused
    admission of a mixture, 2); `tests/driver/test_dispersive_policy.py` (the record and
    its change with the composition, 2).
  - `benchmarks/sep6/dispersive_cost.py` (the filter's rebuild timed);
    `benchmarks/sep6/mixture_curve.py` (the policy against Phase 2b's hand-built curve);
    the ledger entry of `docs/dev/benchmarks.md`; `docs/guide/media/dispersive-kerr.md` (the
    mixtures section, the cost, the record), `nonlinear.md`, `general-modal.md`;
    `docs/dev/backends.md` §3.11 and §11; `docs/dev/kernel-contract.md`; the changelog.

#### The operator, and its normalisation

For channels `s` acting at the same nodes, with `f_s = ε₀γ₃,ₛρ_s` at the stage and
`K̂_s = g_s⁴ = γ_K,s(ν̃_s)/γ₃,ₛ` each species' Kerr diagonal under its own leg bound,

`P = F G_mix[w (G_mix E)³]`, `F = Σ_s f_s`, `g_mix = [Σ_s f_s K̂_s / F]^(1/4)`.

This is §6's filter factorised as §4.4 factorises a pure gas: on the Kerr diagonal
`F g_mix⁴ = Σ_s f_s K̂_s = ε₀ γ_K,mix`, the mixture's degenerate coefficient
`ε₀ Σ_s N_s γ_K,s(ν̃_s)`. The reference is the mixture's static coefficient `F`, so
`g_mix(0) = 1`; any other positive `γ_*` moves `(F/γ_*)^(1/4)` into the filter and `γ_*`
out of the coefficient, and the cubic, of fourth degree in the filter, is unchanged
(§4.4: the reference cancels). Both are tested: the diagonal identity to 8 ε and the
reference's cancellation to 64 ε of the largest sample.

#### Decisions

- **The leaf is unchanged; the schema stays 24.** Each channel keeps its `g_s`, and
  `K̂_s` is recovered as `(g_s²)²`, a few roundings from the curve the translation
  sampled. The leaf's meaning for a symmetric group of several channels changed --
  one cubic under the mixture filter, no longer the per-species sum -- and
  `docs/dev/kernel-contract.md` states it; no out-of-tree backend runs the leaf (Kokkos
  refuses it whole by name), as when Phase 2b changed the collar's order at the same
  version. John Travers may still want a version for it (open question 1).
- **Groups by node region.** Every channel of a fused, free-space or planar problem is
  one mixture; on the general modal route each region's gas fill is its own. The
  polynomial is unchanged: the exact sum on `numpy_ref`, the exact grouping on
  `numba_cpu`.
- **Rebuilt per stage, when the densities change.** The filter depends on the
  composition. `StageWeights.update` returns at once when the stage's densities equal the
  last update's, so a fill of constant composition builds it once (every capillary fill
  at one pressure, every general modal gas region), and a composition varying along z
  rebuilds it at every stage: `O(n_omega × n_species)` and two square roots per bin,
  4–8 µs on the benchmark grids (below). A shared fill under a pressure gradient has
  one composition in exact arithmetic, but its species' density tables are fitted
  separately and their ratio moves in the last bits, so a filter precomputed once would
  not be identical to the rebuilt one; it is not precomputed. No cheaper exact scheme
  was found: the fourth root of a convex combination does not factor, and the rebuild
  is about 2 % of the symmetric stage.
- **The arithmetic is the reference's, in its order.** Species summed in channel order
  from a zero, the quotient by `F`, the fourth root as two correctly rounded square
  roots: `mixture_rows` equals `mixture_filter` bit for bit (tested), so the backends
  build the same filter.
- **A vacuum gives a unit filter.** Densities are non-negative (the problem's domain
  check), so `g_mix⁴` is a convex combination of the `K̂_s`; where every density is zero
  the response is zero whatever the filter, which is taken as one rather than `0/0`.
- **The fused path weighs the channels before it filters.** The input multipliers of a
  mixture are the stage's, so `prepare` (before the stage's inverse transform) updates
  the weights; `begin` still calls the update, now a no-op.
- **A mixture takes the lone path.** One product, so alone it is the lone symmetric
  response: no field-sized buffer of its own beside the split-step absorber (tested).
- **The record.** `mixture_filters` is written only when a symmetric group has several
  species, so a pure gas's and a polynomial run's records and resume keys are what they
  were. The composition's identity is each population's label and the SHA-256 of its
  density path's record as the run's configuration records a mixture
  (`Mixture.serialise`): sampled values, breakpoints and declarations, independent of
  the translation's Chebyshev fit, which could differ in the last bits between machines.
  The policy enters the resume key; the configuration already carried the densities.
  The §4.10 regions stay per species.
- **The general chunk counts two buffers per node and component for each filtered field
  and each product** of the compiled general workspace: the symmetric evaluator one of
  each per gas region, the polynomial two fields and two products per group and one cube
  per region (`general_dispersive_buffers`), over the longer of the fine time and
  spectral axes, as `general_node_chunk` counts its own. `numpy_ref`, whose reference
  pass allocates per call, takes the same chunk.

#### Deviations from Part I and the slice's scope

- **The exact per-species symmetric sum is no longer a backend's operator.** It is the
  test reference (`exact_symmetric_sum`) and the survey's column; Phase 2b's propagation
  of it cannot be repeated through a backend, so `mixture_curve.py` now compares the
  policy with Phase 2b's hand-built curve instead.
- **The memory estimator** (`backends/memory.py`) takes the general chunk with the
  response's buffers and counts its bytes on `numba_cpu`, so that its chunk is the
  backend's; on the fused and spatial routes it still counts none of the response's
  buffers (open question 3).

#### Verification

macOS (Apple silicon, 18 cores), Python 3.14.5, NumPy 2.5.2, optmat 0.4.0, pyFFTW 0.15.1
on FFTW 3.3.11, Kokkos CPU installed from the kokkos main checkout (schema 24, exercised
only for its refusal).

- **The form against Phase 0's triplet sum** of the kernel `F Π_j g_mix(ν_j)`, `g_mix`
  evaluated leg by leg from helium's and argon's models (argon's leg bound active):
  both sectors, one and two components, below `1e-12` (decision Q1).
- **One species is its own operator, bit for bit:** `mixture_filter` returns the row
  itself; a one-channel leaf's `evaluations` is its `weights` and `coefficient ρ`; every
  pre-existing single-species parity and reference test passes unchanged. Helium split
  into two populations keeps its own filter to 4 ε.
- **Against the exact per-species sum** (`test_dispersive_kerr_mixture.py`): a
  three-colour field (500, 800 and 1600 nm) confined
  to 400–2000 nm, the conserving sector, on the capillary's own grid. At an output bin
  on 400–2000 nm every contributing quartet has its four legs in §4.7's survey domain;
  each contribution is a positive coefficient times field amplitudes, so the departure
  at that bin is at most the largest relative departure of a coefficient times the
  exact sum of the spectral magnitudes, which for this zero-phase field is the exact sum
  itself. The bound is Phase 2b's measured mixing error, the curve against the exact
  symmetric sum over 208 109 quartets: 1.57e-4 (He/Ar), 4.46e-4 (air). Measured: 9.2e-5
  and 2.8e-4 of the bin's value at worst, margins 1.7 and 1.6 (a random-phase field of
  the same spectrum: 3.2e-5 and 8.9e-5 against the magnitudes' sum); relative L2 over
  the whole spectrum 4.2e-5 and 1.6e-4.
- **Backends.** `numba_cpu` against `numpy_ref`: the He/Ar mixture's one evaluation and
  ten steps (both evaluators, field full and envelope conserving) within Phase 2a's
  bounds; a mixture of varying composition, 2 to 0.5 bar of helium against 0.5 to 2 bar
  of argon over the 2 mm guide, ten steps under both evaluators within the same bounds;
  the free-space and general routes' helium–argon cases (both evaluators, one evaluation and ten steps) within theirs; the
  general route's mixture region bit for bit against `dispersive_cubic` under
  `mixture_filter`. Bitwise across runs and threads as before.
- **Budget:** the fused admission of a symmetric He/Ar mixture is that of one species
  (no node buffer alone), the polynomial's two; the general chunk admits two nodes at
  twice a node's bytes and one a byte short, counting 4 (symmetric) and 10 (polynomial)
  buffers on the strand, and `_general_maps` takes it.
- **The record:** a He/Ar capillary names its filter, the composition and two distinct
  density identities, and its entries the filter; a pure gas and a polynomial mixture
  record none; changing the composition changes the policy; a general region's filter
  names region 1.
- **Runs** (`pytest -n auto`, per directory): `tests/backends` 3519 passed and 9 skipped
  (8 min at a load average near 30); `tests/nonlinear` 1023 passed; `tests/translate`
  1176 passed and 3 skipped; `tests/driver` 1452 passed. After the last code commit
  (the general layout's keys, the manifest's membership) `tests/driver` with
  `test_numba_dispersive_routes.py` again, 1555 passed, and the changed modules'
  doctests, 42 passed; `tests/test_sol.py` and
  `tests/test_structure_figures.py` 273 passed; `tests/test_seps.py` 127 passed and 1 skipped. The
  kernel module with `test_bodies.py` under `NUMBA_DISABLE_JIT=1`, 91 passed, and with
  the compiled dispersive modules (417 passed) the changed lines are covered:
  `dispersive_spatial.py` 100 %, `dispersive.py` and `dispersive_general.py` 99 % with
  only pre-existing branches missed. `ruff`, `basedpyright` (no errors) and the
  public-tree scan (no finding in a changed line) on the changed files. `poe docs` (every tutorial executed) succeeds with no warning.
- **Not run:** the full gate, the slow lane and the coverage lane (the SEP's end); no
  Linux or quiet-machine timing; no Kokkos change.

#### Q5 measured again

Same-process stage ratios to the constant cubic on the four-mode capillaries
(`benchmarks/sep6/dispersive_cost.py` at `8a6583d`, seven interleaved rounds, medians,
a load average of 16 to 20; the full tables are the ledger entry of
`docs/dev/benchmarks.md`). One thread; default pools in brackets.

| case | symmetric, thg / no thg | rebuild of the filter | polynomial, thg / no thg |
|---|---|---|---|
| helium alone (reference) | 1.00 / 1.03 (1.06 / 1.09) | – | 2.55 / 3.42 (2.34 / 2.54) |
| He/Ar | 1.00 / 0.93 (1.04 / 1.09) | 7.7 µs | 4.10 / 5.91 (3.42 / 3.84) |
| Kr/Xe | 1.03 / 0.92 (1.11 / 1.00) | 4.3–5.0 µs | 2.36 / 3.18 (2.36 / 2.49) |
| air | 1.02 / 0.90 (1.06 / 1.03) | 5.2 µs | 5.20 / 7.27 (4.35 / 5.04) |

- **Met:** the symmetric evaluator on every mixture, 0.90–1.11, where Phase 2b measured
  1.68–3.00 for the exact per-species sum; a mixture costs what helium alone costs on
  the same capillary. Its rebuild, paid only where the composition changes, is about
  2 % of the stage. Ten-step windows: 0.94–1.07.
- **Unchanged:** the polynomial, which this slice does not touch, within the machine's
  noise of Phase 2b; its miss on mixtures of distinct leg bounds is the roadmap's.
- `mixture_curve.py`: the policy's filter equals Phase 2b's hand-built curve to one ulp
  (2.2e-16); 3 cm adaptive runs under the two give exit fields 4.9e-9 (He/Ar) and
  1.2e-15 (air) apart, the first the step controller's response to that ulp, and the
  same departures from the polynomial as Phase 2b's curve (0.287 % and 0.361 %).

#### The rest of this report

- **Sources checked:** Part I §4.4, §4.7, §6 and §11; the Phase 0 to 2b reports and
  their decisions; the Phase 2b code and benchmarks. No new literature or material data:
  every coefficient and leg bound is read from optmat 0.4.0 through
  `DispersiveKerrModel`.
- **Limitations:**
  - The bound on the operator's departure from the exact sum is Phase 2b's sampled
    maximum over 400–2000 nm quartets, not a supremum, and holds only for quartets in
    that domain; quartets with an ultraviolet leg (the RDW band) depart more, and are
    not bounded by a test here.
  - The rebuild runs at every stage of a composition varying along z, including a shared
    fill under a pressure gradient whose composition is constant in exact arithmetic.
  - The memory estimator counts the response's buffers on the general route only.
  - Timings are same-process ratios on a loaded workstation; no Linux or quiet-machine
    timing.
- **Open questions:**
  1. *Resolved (John Travers, 2026-10-08):* schema 24 stays; no version for the
     changed meaning of a symmetric group, as for the collar's order.
  2. *Resolved:* the exact per-species symmetric sum stays a test reference only; no
     leaf or option reaches it.
  3. *Resolved:* the memory estimator counts the response's buffers on the fused and
     spatial routes too; done at the start of Phase 3 (solprop side).
  4. *Resolved:* slice 2c merged, completing Phase 2.
- **Handoff:**
  - **Phase 3** (Kokkos CPU, then device) implements the leaf against `numpy_ref` and
    `numba_cpu` with this phase's structure, now including the symmetric mixture
    filter: one product per group of channels at the same nodes, its filter rebuilt by
    `mixture_filter`'s arithmetic (channel order, two square roots, a unit filter at a
    vacuum) whenever the stage's densities change, the general chunk counting the
    response's buffers. The kernel schema is unchanged by this slice.

### Phase 3a — Kokkos CPU's fused path

- **Scope:** slice 3a of Phase 3 (§11), 2026-10-08, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-3` in solprop and in
  solprop-kokkos (from solprop `fabc01b` and solprop-kokkos `895af49`). Kokkos CPU
  executes `responses.dispersive_kerr` on its fused node path: the mode-averaged,
  multimode and radial routes, field and envelope grids, both evaluators, both sectors,
  one or two components, mixtures (the polynomial's equal-bound groups; the symmetric
  mixture filter), beside Kerr, Raman, plasma and envelope plasma, against `numpy_ref`
  and `numba_cpu`. The Cartesian free-space, line, planar and general modal routes stay
  refused by name for slice 3b; the device build is slice 3c's. With it, Phase 2c's
  decision 3: the memory estimate counts the response's buffers on the fused and
  spatial routes. The kernel schema stays 24.
- **Delivered, solprop-kokkos:**
  - `include/solprop_kokkos/dispersive.hpp`, `src/dispersive.cpp`:
    `DispersiveResponse` (the layout, the per-stage weights and the mixture filter's
    rebuild, the filter, the synthesis and projection GEMMs, the product functors, the
    forward transform, the landing and weighing kernels, the lazy embedded finish, the
    ledger), the sample bodies of `bodies.py`'s dispersive forms, and
    `dispersive_mixture_rows`, `mixture_filter`'s arithmetic.
  - `include/solprop_kokkos/core.hpp`: the layout's fields on `Problem`,
    `TrialScalars::dispersive_densities`, `GeneralState::projected_to_host` (a
    diagnostic: the last evaluation's projected spectra on both rules) and
    `dispersive_path`. `src/general.cpp`: the response inside the stage
    (`evaluate_stage_source`), its validation, plans, densities, the embedded finish at
    the score, the ledger and close. `src/bindings.cpp`, `_core.pyi`: the layout, the
    trial's densities, the two accessors.
  - `src/solprop_kokkos/backend.py`: `_dispersive_payload` (the layout of `numba_cpu`'s
    `dispersive_terms_layout`, from the leaf's `groups()`), the analytic family for the
    conserving sector on a field grid, each stage's densities in the trial's material
    rows, `_refuse_dispersive_route` (the routes off the fused path, by name), the
    `dispersive_path` metadata. `__init__.py`: `DISPERSIVE_KERR_FIELDS` (new),
    `dispersive_kerr` in `RESPONSE_FIELDS`, `dispersive_kerr_densities` in
    `STAGE_SCALARS_FIELDS`, all checked by `check_build`.
  - Tests: `tests/native/test_dispersive.cpp` (8), `tests/test_dispersive.py` (13), the
    contract and refusal tests of `test_backend.py` and `test_api_contract.py`, the
    native tests' trial scalars; `CMakeLists.txt`, `scripts/check.sh`; `CHANGELOG.md`,
    `README.md`, `AGENTS.md` (the public tuple).
- **Delivered, solprop:**
  - `tests/backends/test_kokkos_dispersive_kerr.py` (87) and
    `kokkos_dispersive_thread_probe.py`; `test_dispersive_kerr.py`'s Kokkos test now
    admits the leaf on the fused path.
  - `backends/memory.py`: `dispersive_memory`, in the `base` group of
    `estimate_memory`; `translate/problem.py`: `dispersive_node_buffers`, now public;
    `tests/backends/test_memory.py` (15 new).
  - `benchmarks/sep6/dispersive_cost.py --backend kokkos`; the ledger entry of
    `docs/dev/benchmarks.md`; `docs/dev/backends.md` (§4, §9, §11),
    `docs/dev/kernel-contract.md`, `docs/guide/media/dispersive-kerr.md`, the tutorial's
    note on backends; the changelog.

#### Decisions

- **`numba_cpu`'s structure, natively** (the Phase 2 handoff). Before the stage's inverse
  transform the response weighs its channels at the stage and filters the padded modal
  spectrum, and one batched inverse transform gives every filtered field; per node
  chunk the real synthesis GEMM puts them at the nodes, a functor forms the products in
  the leaf's sector, and the projection GEMMs take them to both rules' columns; one
  batched forward transform and a landing kernel add their weighted spectra to the
  stage's own polarisation spectrum. Alone, the response replaces the response column,
  the projection and the stage's forward transforms; a lone symmetric response reads the
  stage's node buffer, filters the stage's spectrum in place when no in-stage absorber
  reads the bare field, and outside a field grid's analytic family is the stage's own
  polarisation, weighed by `g` in the stage's spectra on both rules. The polynomial
  subtracts the bare analytic signal's zero-frequency constant (the guard Phase 2b
  kept).
- **The collar acts at the nodes, in the product functors.** `numba_cpu` multiplies the
  projected products; the collar is the same at every node and the projection is
  linear, so `P (w X) = w (P X)` and the operator is the decided `G[w (G E)^3]` either
  way. At the nodes it costs no pass of its own, and the borrowed product, the stage's
  polarisation, carries it already; the two orders differ by rounding.
- **The embedded rule's products are finished when the step is scored.** The engine
  transforms its own embedded projection at every stage, so a borrowed product's
  embedded spectrum is weighed there; the response's own embedded products are
  projected every stage and transformed and landed once, when the accepted step's
  quadrature is scored, from the last stage's products and weights.
- **The layout is the adapter's, the arithmetic native.** The adapter packs the leaf as
  `dispersive_terms_layout` does (every multiplier carries the sector's `Π₊` and
  factor; a mixture's term names its own factor), and the engine executes the tables.
  The stage's weights -- `ε₀ γ₀ ρ` per channel, each symmetric mixture's coefficient and
  filter rows, the polynomial group's pair constants, the terms summed per product in
  term order from zero -- are rebuilt only when a stage's densities differ from the last
  rebuild's, compared on the host; a mixture's rows by `mixture_filter`'s arithmetic
  (channel order, two square roots, a unit filter at a vacuum), held bit for bit to its
  closed form by a native test. The factors cross through a ring of four host staging
  buffers, as the stage scalars do, so a rebuild allocates nothing and needs no fence.
- **The dense transforms take the build's GEMM**: the real matrices times the samples'
  interleaved real views through `selected_gemm` (the vendor BLAS, the native-order
  kernel or the custom one, as the build's provider says); with more than one node
  chunk each chunk's columns of the scaled projections are packed once a stage.
- **The contract tuples.** `DISPERSIVE_KERR_FIELDS` is a new public tuple, as
  `MODAL_TABLES_FIELDS` is for its leaf: the leaf's members the adapter reads
  (`carrier`, `coefficients`, `evaluator`, `filters`, `n_channels`, `outputs`,
  `positive`, `products`, `regions`, `sector`); its grouping is solprop's own
  `groups()`, which the adapter calls. The schema stays 24: no solprop field changed.
- **Phase 2c's decision 3, the memory estimate.** `dispersive_memory(problem, backend)`
  returns rows of the `base` group. On the fused path both compiled backends hold the
  response's filtered fields, node slots, products, embedded products and spectra by
  one rule, and the rows are those ledgers' to the byte. On the Cartesian, line and
  planar routes (`numba_cpu`) the row is the complex-equivalent buffers per node and
  component the node budget admits (`dispersive_node_buffers`) times the chunk, as the
  general route's chunk counts its own; Kokkos refuses the response there. `numpy_ref`,
  whose pass allocates per call, and the general route, whose chunk already counts the
  response, add nothing. The multipliers, scratch and packed projections are named
  among what is not counted.

#### Deviations from Part I and the slice's scope

- The collar's place in the arithmetic (above): the same operator, a different
  rounding from `numba_cpu`'s.
- **Envelope plasma is compared on the whole stage source.** The engine assembles the
  envelope currents into the projected spectrum, which the reference keeps beside it,
  so an envelope-plasma problem's projected spectra differ by the currents themselves
  (a relative 1.8); the stage source holds them on both.
- **Thread invariance is bitwise where every transform is short.** The response's
  batched transforms follow the engine's rule: a transform of fewer than 8192
  real-side samples is planned on one FFTW thread. On the mode-averaged route all are,
  and ten steps are bitwise at one and four threads; the multimode capillary's batched
  products exceed it and may move by rounding with the thread count, as the engine's
  own long transforms may.

#### Verification

macOS (Apple silicon, 18 cores), Python 3.14.5, NumPy 2.5.2, optmat 0.4.0, pyFFTW 0.15.1
on FFTW 3.3.11; solprop-kokkos at `41ca10e` (the branch's native code), the local
profile (Homebrew Clang 23.1.2, Accelerate, native architecture), Kokkos 5.2.1,
Kokkos-FFT 2.0.0, OpenMP, `OMP_PROC_BIND=false`; a load average of 16 to 44.

- **Parity** (`test_kokkos_dispersive_kerr.py`; relative L2; `numba_cpu`'s bounds, from
  the transforms and node reductions both backends perform, asserted positive):
  - one evaluation's projected polarisation on the 33-node three-mode argon capillary,
    both evaluators × field and envelope × both sectors: 2.3–4.3 ε on both rules
    against `numpy_ref` and 0.5–2.3 ε against `numba_cpu`, under bounds of 1064–1870 ε;
  - one Runge–Kutta step (seven evaluations): 1.2–1.5 ε against 7400–13100 ε; ten steps
    of the default integrator (the resident Adams engine): 5.8–6.7 ε against
    `numpy_ref` and 5.6–6.7 ε against `numba_cpu`, under 155 000–273 000 ε; ten
    Runge–Kutta steps 5.7–5.9 ε;
  - the quadrature alarm after ten steps, which reads the embedded products finished at
    the score: 7.2e-13–7.5e-13 (full cubic) and 4.9e-15 (conserving), within 4.7e-17
    of the reference's;
  - the mode-averaged route 2.4–4.2 ε (ten steps below 0.1 ε), a two-component launch
    2.8–3.8 ε, the radial route 2.7–5.0 ε; a He/Ar mixture 2.9–4.1 ε (ten steps
    5.9–6.8 ε), krypton and xenon as one polynomial group 2.9–4.1 ε (6.6–7.3 ε), a
    varying composition (the filter rebuilt per stage) over ten steps, nitrogen's Raman
    beside it 2.5–3.4 ε (6.4–7.2 ε), five node chunks of eight, a state with
    zero-frequency content;
  - beside plasma, the stage source: the carrier policy's own disagreement with the
    reference is 268 ε (the field's plasma column) and 178 ε (envelope plasma), and
    the dispersive runs sit within 0.2 ε of it; ten steps 6.8–19.5 ε, the carrier's
    6.6–6.7 ε, under 155 000–273 000 ε;
  - bitwise: two runs of one build (Raman beside the polynomial), and ten steps of both
    evaluators at one thread, again, and at four (the mode-averaged route).
- **Native** (`tests/native/test_dispersive.cpp`): the contraction against the tensor
  sum and the closed forms against the general groups to 16–32 ε; the mixture rows
  against `mixture_filter`'s closed form bit for bit, and a vacuum's unit filter; a lone
  unit filter on one node is the constant cubic to 64 ε of a direct sum and holds no
  field-sized buffer; malformed layouts and a trial without densities are refused before
  allocation; a problem without the response names its rows at zero. The native suite,
  80 tests, passes.
- **Adapter** (`tests/test_dispersive.py`): a symmetric mixture's one product and its
  factor; krypton and xenon's one polynomial group and its terms; the family by sector;
  the general route's refusal; one evaluation of each evaluator and sector against
  `numpy_ref` within its bound.
- **Memory.** The ledger, tested: a lone symmetric response with the split-step absorber
  holds no field-sized buffer, and a mixture likewise; beside the in-stage collar one
  filtered field; the polynomial two filtered fields, two node slots, three products
  per rule and their spectra; beside Raman a node slot, its products and spectra;
  chunked nodes their packed projections. `estimate_memory` equals both compiled
  backends' rows to the byte on six fused cases and a chunked one, and the Cartesian
  route's admitted buffers bound `numba_cpu`'s holdings.
- **Runs** (`pytest -n auto`, solprop-kokkos `41ca10e` installed as a wheel): the new
  module, 87 passed in 2 min; `tests/backends`, 3621 passed and 9 skipped (7 min 23 s);
  every other directory with `src`'s doctests, 10761 passed and 19 skipped, with one
  failure of `test_convergence_records.py`'s writer check, which needs `tests/backends`
  collected beside it and passes when it is (the per-directory split, not this slice).
  solprop-kokkos's gate, `scripts/check.sh`, green once at the slice's end in 14 min
  48 s: Ruff, strict Pyright, the pinned clang-format, clang-tidy with strict warnings
  and the native tests (reference profile, Debug), the native tests under ASan and
  UBSan, the editable local build with the package's suite (247 passed, 5 skipped, four
  threads), the distribution wheel and its clean import. `ruff`, `basedpyright` (no
  errors in the changed modules) and the public-tree scan on solprop's changed files.
- **Not run:** no Linux or quiet-machine timing; no device build (slice 3c); solprop's
  slow and coverage lanes and the docs build (the SEP's end).

#### Q5 measured on Kokkos CPU

`benchmarks/sep6/dispersive_cost.py --backend kokkos` (the ledger entry of
`docs/dev/benchmarks.md`): stage ratios to the constant cubic, seven interleaved rounds,
medians, a load average of 17 to 34. One thread; the default pools in brackets.

| case | symmetric, thg / no thg | polynomial, thg / no thg |
|---|---|---|
| one mode-averaged node | 1.03 / 1.21 (1.21 / 1.35) | 1.95 / 2.87 (1.43 / 1.86) |
| four modes, 33 nodes | 1.06 / 1.02 (1.20 / 1.08) | 2.06 / 3.65 (1.34 / 1.73) |
| He/Ar | 1.02 / 1.00 (1.17 / 1.11) | 3.18 / 6.19 (1.49 / 2.19) |
| Kr/Xe | 1.03 / 1.00 (1.26 / 1.09) | 1.83 / 3.46 (1.28 / 1.67) |
| air | 1.03 / 0.97 (1.21 / 1.15) | 3.87 / 8.32 (1.58 / 2.13) |

- **Met:** the symmetric evaluator on every case at one thread, 0.97–1.21; the
  polynomial for one species and for krypton and xenon, 1.83–3.65, and on every mixture
  in the default pools.
- **Missed:** the symmetric one-node analytic sector in the default pools, 1.35, at a
  load average above 30, where every default-pool stage ran slower than at one thread;
  the polynomial on the mixtures of distinct leg bounds in the analytic sector at one
  thread, 6.19 (He/Ar) and 8.32 (air), as on `numba_cpu` (5.91 and 7.27): the roadmap's
  item.
- Ten-step windows: 0.99–1.14 (symmetric), 1.15–5.11 (polynomial).

#### The rest of this report

- **Sources checked:** Part I §6 and §11; the Phase 0 to 2c reports; `numba_cpu`'s
  `dispersive.py` and `bodies.py`, `numpy_ref`'s pass, `dispersive_cubic` and
  `mixture_filter`; solprop-kokkos's `AGENTS.md`, `README.md` and its stage pipeline. No
  new literature or material data.
- **Limitations:**
  - Kokkos runs the response on the fused node path only; the other routes refuse it.
  - The device build compiles the same source (functors over a device range) but has
    been neither built nor run here.
  - The analytic sector's lone products are complex beside a real polarisation, so they
    hold their own products, embedded products and spectra (three complex fields),
    above Q5's memory target, as on `numba_cpu`; so does the response beside Raman or
    plasma.
  - The memory estimate's spatial count is the node budget's, an upper bound of
    `numba_cpu`'s holdings rather than its rows.
  - Timings are same-process ratios on a loaded workstation; the default-pool ones are
    dominated by contention.
- **Open questions:**
  1. *Resolved (John Travers, 2026-10-08):* `DISPERSIVE_KERR_FIELDS` is accepted as a
     public name of solprop-kokkos.
  2. *Resolved (John Travers, 2026-10-08):* the collar at the nodes, before the
     projection, rather than on the projected products is accepted: the same operator,
     a different rounding.
  3. *Resolved (John Travers, 2026-10-08):* the symmetric one-node analytic sector's
     1.35 in the default pools is re-measured on a quiet machine before any
     optimisation is considered; no action now.
  4. *Resolved (John Travers, 2026-10-08):* the response's memory beside Raman or
     plasma and in the analytic lone path, above Q5's memory target, is accepted and
     recorded; no optimisation slice.
  5. *Resolved (John Travers, 2026-10-08):* the stale kernel schema of
     solprop-kokkos's `README.md` was corrected on the branch (schema 24). The memory
     estimator's design -- the ledgers' rows to the byte on the fused path, the node
     budget's upper bound on the spatial routes until Kokkos's own rows exist there --
     is accepted; slice 3b gives the spatial routes Kokkos's rows.
- **Handoff:**
  - **Slice 3b** (the Cartesian free-space, line, planar and general modal routes on
    Kokkos CPU) takes `numba_cpu`'s per-chunk schemes (`SpatialDispersive`,
    `GeneralDispersive`): the response per chunk of columns or block of rows on the
    separable engine, and in the general route's synthesis with each region's cube and
    mixture. It removes those routes from `_refuse_dispersive_route`, extends the
    adapter's layout to region keys, and gives `dispersive_memory` Kokkos's spatial
    rows.
  - **Slice 3c** (the device build) compiles `src/dispersive.cpp` for CUDA and HIP (its
    product functors run over a device range, its other kernels are device lambdas,
    the factors cross through the staging ring) and qualifies the fused path with
    `pytest -m gpu` and device bounds; `test_kokkos_dispersive_kerr.py` skips on a
    device build until then.

### Phase 3b — the remaining Kokkos CPU routes

- **Scope:** slice 3b of Phase 3 (§11), 2026-10-08, implemented by Opus 5.5 (AI) for
  John Travers on `feature/sep-0006-dispersive-nonlinearity-3` in solprop and in
  solprop-kokkos (from solprop `5e11eeb` and solprop-kokkos `ceeb5ee`, slice 3a's
  heads). Kokkos CPU executes `responses.dispersive_kerr` on the Cartesian free-space
  and line routes, the hybrid planar route and the general modal route, with
  `numba_cpu`'s per-chunk structure, against `numpy_ref` and `numba_cpu`; with the fused
  path of slice 3a it runs the response on every route `numpy_ref` admits, and its
  adapter refuses none of them. A coupled wall flow beside the response, which slice
  3a's native code still refused, runs too. The memory estimate gives Kokkos's own rows
  on these routes. The kernel schema stays 24; the device build is slice 3c's.
- **Delivered, solprop-kokkos:**
  - `include/solprop_kokkos/dispersive.hpp`, `src/dispersive.cpp`: `DispersiveWeights`
    (the layout and the stage's weights, every route's; one cube per distinct group key
    for the polynomial), `SeparableDispersive` (the Cartesian, line and planar routes,
    chunk by chunk), `GeneralDispersive` (the general modal route, in its synthesis),
    the three-component bodies `dispersive_isotropic_n` and `dispersive_conserving_n`,
    and the shared filter, landing and weighing kernels; `DispersiveResponse` (the fused
    path) now holds a `DispersiveWeights`, its arithmetic unchanged.
    `include/solprop_kokkos/slab.hpp`: the planar kind's slab rule in x
    (`slab_contract`), moved out of `src/general.cpp` for the response to share.
  - `include/solprop_kokkos/core.hpp`: `Problem::dispersive_group_keys` and
    `dispersive_node_region` (the general route's). `src/general.cpp`: the response in
    the separable stage (`evaluate_separable_stage`), its densities, plans, ledger, path
    and close; `src/general_modal.cpp`, `general_modal.hpp`: `GeneralModalSource` owns
    the general route's response, skips the regional contraction and its transform when
    no tensor or delayed response makes one, and writes or adds the response's spectra
    ahead of the projection; `src/bindings.cpp`: the two new arrays.
  - `src/solprop_kokkos/backend.py`: `_refuse_dispersive_route` removed; the layout
    carries each group's region and every node's region on the general route, one cube
    per region, and the route's real field family; the general node chunk counts the
    response's buffers (`general_dispersive_buffers`), as `numpy_ref` and `numba_cpu`
    take it.
  - Tests: `tests/native/test_dispersive.cpp` (three new: the three-component
    contraction against the tensor sum and, for two components, the two-component body
    bit for bit; the cube layout; the regions' refusals), `tests/test_dispersive.py`
    (the general route's layout, both evaluators), `tests/test_backend.py` (the
    free-space route runs, its path and ledger, in place of its refusal);
    `scripts/check.sh`; `CHANGELOG.md`, `README.md`, the contract tuples' docstrings.
- **Delivered, solprop:**
  - `tests/backends/test_kokkos_dispersive_routes.py` (133) and
    `kokkos_dispersive_thread_probe.py`'s route cases; `test_dispersive_kerr_routes.py`'s
    refusal test removed.
  - `backends/memory.py`: `dispersive_memory` gives Kokkos's rows on the Cartesian, line
    and planar routes and on the general modal route (`_separable_dispersive_memory`,
    `_general_dispersive_memory`); `tests/backends/test_memory.py`.
  - `benchmarks/sep6/dispersive_cost.py --backend kokkos` on every case; the ledger entry
    of `docs/dev/benchmarks.md`; `docs/dev/backends.md` (§4, the route table, §11),
    `docs/dev/kernel-contract.md`, `docs/guide/media/dispersive-kerr.md`, the tutorial's note
    on backends; the changelog.

#### Decisions

- **`numba_cpu`'s per-chunk structure, natively** (Phase 3a's handoff). On the
  separable routes the response filters each chunk's packed spectrum before the
  stage's inverse transform, forms its products under the collar at the chunk's nodes
  once the stage has put the bare field there -- a free-space chunk's temporal rows are
  its nodes; a planar block's `(p, y)` rows are synthesised at its `(x, y)` nodes and
  the products projected back by the stage's own slab rule -- transforms them and lands
  their weighted spectra in the chunk's polarisation spectrum after the stage's forward
  transform, or in its place when alone. Free space's inverse transform writes the
  filtered fields straight into the product blocks they become. On the general route
  the multipliers act on each chunk's synthesised spectrum, the products are formed at
  each group's region's nodes and zero elsewhere, and their spectra land ahead of the
  projection, written rather than added when no regional tensor or delayed response
  forms the chunk's polarisation (whose contraction and transform are then skipped).
- **A lone symmetric response filters in place on every separable route.** Kokkos's
  in-stage absorber on these routes is a pass of its own that packs each chunk afresh,
  so nothing reads the bare field after the material pass; `numba_cpu`'s planar stage
  reads the block's rows once for both, and there filters into a buffer of its own
  beside the in-stage collar. The operator is the same; Kokkos holds less.
- **The general route keeps its synthesis's `1 / n_to`.** Kokkos's general source folds
  the inverse transform's normalisation into the cubic rather than into the synthesis,
  so the response applies it to what it reads -- the filtered spectrum and the bare
  signal -- a power of two on every grid solprop builds, which moves no bit.
- **One to three Cartesian components on the general route.** The contraction sums the
  components the route resolves in order, which is `numba_cpu`'s three-component sum
  with the absent components zero.
- **The coupled wall flow beside the response is admitted.** Slice 3a's native
  validation refused `coupled_flow` with the response, though `numpy_ref` and
  `numba_cpu` run it and Kokkos's flow only carries each stage's input around the fused
  stage source; with the refusal gone it agrees with the reference within the flow's
  own ten-step bound.
- **The estimate is exact on Kokkos.** `dispersive_memory(problem, "kokkos")` returns the
  rows Kokkos's ledger names on every route, to the byte, in the `base` group, the
  general route's included (Kokkos's general chunk group no longer adds them).
  `numba_cpu`'s spatial rows stay the node budget's upper bound, and its general rows
  the chunk group's, as accepted for slice 3a.
- The contract tuples are unchanged: the adapter reads `regions` and `region_index`,
  both listed already.

#### Deviations from Part I and the slice's scope

- The planar route's lone symmetric response filters in place beside the in-stage
  collar, where `numba_cpu` does not (above).
- The coupled wall flow's admission is beyond the slice's list of routes; it closes
  the one refusal of a combination `numpy_ref` admits that slice 3a left.
- Q5's polynomial target is missed on the general modal route at one thread (below).

#### Verification

macOS (Apple silicon, 18 cores), Python 3.14.5, NumPy 2.5.2, optmat 0.4.0, pyFFTW 0.15.1
on FFTW 3.3.11; solprop-kokkos at `925df76` (the branch's native code at `c9bc4e7`), the
local profile (Homebrew Clang 23.1.2, Accelerate, native architecture), Kokkos 5.2.1,
Kokkos-FFT 2.0.0, OpenMP, `OMP_PROC_BIND=false`; a load average of 11 to 16.

- **Parity** (`test_kokkos_dispersive_routes.py`; relative L2 of the stage source, or
  of the field after the steps; `numba_cpu`'s bounds, derived from the transforms and
  node reductions both backends perform, asserted positive):
  - free space and the line, both evaluators, field full and conserving, envelope
    conserving and full: one evaluation 2.1–4.6 ε against `numpy_ref` and 0.002–2.2 ε
    against `numba_cpu`, under 640–4500 ε; ten steps of the default integrator
    below 0.005 ε against `numpy_ref` and 0.65–1.5 ε against `numba_cpu`, under
    93 000–663 000 ε; beside Raman 2.9–4.4 ε and 1.4–1.8 ε (ten steps below 0.001 ε
    and 0.8–1.3 ε); a helium–argon medium (the symmetric mixture filter rebuilt
    natively; the polynomial's two groups), one evaluation and ten steps; a partial
    chunk; one Runge–Kutta step;
  - the planar route, both sectors: 3.4–5.8 ε and 2.4–2.9 ε against 1800–3500 ε, ten
    steps 0.16–0.28 ε and 0.75–1.2 ε; beside Raman 3.1–5.9 ε; beside the in-stage
    collar, in place; a partial block; one Runge–Kutta step;
  - the general route, a gas strand alone, beside a constant solid core and a
    helium–argon region, real and analytic forms, both evaluators: 0.54–1.6 ε and
    0.3–0.95 ε against 560–1090 ε, ten steps below 0.6 ε and 0.53–0.96 ε; krypton
    and xenon one group and one cube; one Runge–Kutta step;
  - a coherent wall flow (a three-mode helium capillary, ten steps) beside each
    evaluator: 1.0e-15–1.1e-15 against the flow's ten-step bound of 2.2e-13, where the
    response moves the field from the carrier policy's by 2.2e-7;
  - the ten-step departures from `numpy_ref` sit far below one ε because these short
    windows are dominated by the linear propagation, which Kokkos's separable and
    general engines form as the reference does; the one-evaluation comparisons are the
    sharp ones;
  - bitwise: two runs of one build on each route and evaluator, and ten steps at one
    thread and at four on the free-space, planar and general routes (the response's
    transforms below the engine's 8192-sample threshold run on one FFTW thread, as the
    routes' own do).
- **Memory** (tested): `dispersive_memory(problem, "kokkos")` equals the native ledger
  to the byte on 30 cases -- free space (both sectors, the envelope grid, beside
  Raman), the planar route (likewise) and the general route (both forms, the three
  strands), both evaluators -- and every other response row is zero. A lone symmetric
  free-space response holds no field-sized buffer; the free-space polynomial writes its
  filtered fields into its product blocks (three products and their spectra, no filtered
  field); the planar polynomial two node slots.
- **Native** (`tests/native/test_dispersive.cpp`, 11 dispersive tests; the native suite
  83): the three-component contraction against the tensor sum to 32 ε; the cube
  layout; the regions' refusals before allocation.
- **Runs**: see "The gates" below.

#### Q5 measured on Kokkos CPU's other routes

`benchmarks/sep6/dispersive_cost.py --backend kokkos --cases free line planar general`
(the ledger entry of `docs/dev/benchmarks.md`): stage ratios to the constant cubic, seven
interleaved rounds, medians, a load average of 12 to 15. One thread; the default pools
in brackets.

| case | symmetric, thg / no thg | polynomial, thg / no thg |
|---|---|---|
| Cartesian, 16 × 16 | 0.99 / 1.08 (1.10 / 1.07) | 1.59 / 2.27 (1.36 / 1.59) |
| line, 64 | 1.02 / 1.14 (1.00 / 1.08) | 1.52 / 2.08 (1.16 / 1.35) |
| planar | 0.96 / 1.02 (1.39 / 1.61) | 2.12 / 3.56 (2.15 / 2.86) |
| general modal, real / analytic | 1.04 / 0.81 (1.15 / 1.15) | 4.96 / 4.15 (2.64 / 2.86) |

- **Met:** the symmetric evaluator on every route at one thread, 0.81–1.14, and in the
  default pools but on the planar route; the polynomial on the Cartesian, line and
  planar routes in every pool (1.16–3.56), and on the general route in the default
  pools.
- **Missed:** the symmetric planar route in the default pools, 1.39 and 1.61, where
  every planar stage ran two and a half times slower than at one thread (contention);
  the polynomial on the general route at one thread, 4.96 and 4.15, where `numba_cpu`
  measured 2.32 and 1.44: alone, the response replaces the regional contraction but adds
  five batched transforms of the chunk's rows (two filtered fields, three products) to
  the route's own two, on a 256-sample grid where the transforms are most of the stage
  (open question 1).
- Ten-step windows: 0.80–1.13 (symmetric), 1.16–4.75 (polynomial).

#### The gates

Run at the end of the slice, after its report was interrupted, on the
heads solprop-kokkos `c5c000b` (the slice's code, with a lint fix: an over-long
docstring line and an unused import the refusal's removal left) and solprop `460e0c9`:

- `scripts/check.sh` in solprop-kokkos: green in 8 min 0 s — Python format, lint and
  strict typing, C++ format, clang-tidy and the native tests, the native tests under
  ASan and UBSan, the editable build's package suite (249 passed, 5 skipped), the
  distribution wheel and its clean import. Its first two runs failed on the lint above
  and on a dependency fetch two concurrent builds in one build directory had collided
  on; neither touched the code.
- solprop, against a wheel of `c5c000b` (local profile): `tests/backends` 3751 passed,
  9 skipped; `tests/nonlinear` and `tests/translate` 2199 passed, 3 skipped;
  `tests/driver` 1452 passed; `test_seps`, `test_sol`, `test_structure_figures` 400
  passed, 1 skipped.
- Not run: solprop's full gate, slow and coverage lanes (once, at the end of the SEP);
  the device build (slice 3c).

#### The rest of this report

- **Sources checked:** Part I §6 and §11; the Phase 2b, 2c and 3a reports;
  `numba_cpu`'s `dispersive_spatial.py`, `dispersive_general.py`, `dispersive.py` and
  `general_modal.py`; `numpy_ref`'s free, planar and general passes; solprop-kokkos's
  `AGENTS.md`, `README.md`, its separable stage and general modal source. No new
  literature or material data.
- **Limitations:**
  - The device build compiles the same source (the new functors run over a device
    range, the slab rule is the stage's team kernel) but has been neither built nor run
    here.
  - The general route's polynomial misses Q5 at one thread (above).
  - The parity cases are `numba_cpu`'s: the planar route on a field grid only, one
    polarisation component on the separable routes; plasma and envelope plasma beside the
    response on these routes are executed (the stage pipeline is the fused path's) but
    not compared.
  - A process that has run both Kokkos and `numba_cpu` ends in a segmentation fault at
    interpreter exit, with the carrier policy as with the response and on slice 3a's
    fused path alike; the runs themselves complete and agree. It predates this slice.
  - Timings are same-process ratios on a loaded workstation; no Linux or quiet-machine
    timing.
- **Open questions:**
  1. *Resolved (John Travers, 2026-10-08):* the general route's polynomial at one
     thread (4.96 and 4.15) is accepted as the cost of the route's transforms; it meets
     Q5 in the default pools. Folding the cube's transform into the route's own stays
     an optimisation idea.
  2. *Resolved:* the buffer layout is each backend's to choose where the operator is
     the same.
  3. *Resolved:* the coupled wall flow beside the response is admitted.
  4. *Resolved:* the crash at interpreter exit is an issue of its own, outside SEP 6.
- **Handoff to slice 3c (device qualification).** The run, on a rented GPU after
  its plan is reviewed, must cover:
  - **Builds:** the `cuda` preset (CUDA, `HYBRID` GEMM over cuBLAS, `HOPPER90`,
    OpenMP off) and the `hip` preset (`AMD_GFX942`, `HYBRID` over rocBLAS), each with
    the native tests; on CUDA also `cuda-custom` (`CUSTOM` GEMM, no TPL BLAS), which
    exercises the fused path's synthesis and projection GEMMs without a vendor BLAS. The
    wheel of each build installed into solprop's environment and `check_build()` passed.
  - **Code paths:** every native kernel of `src/dispersive.cpp` on the device range
    (`over_nodes`'s two-dimensional policy): `SelfProduct`, `PairProducts`,
    `GeneralCube`, `GeneralPairs`; the filter, landing, weighing, offset and mixture
    kernels; the staging ring's host-to-device copies of the factors; the slab rule's
    team kernel on the response's subviews; and the transforms on cuFFT and rocFFT
    (the response's c2r plans write over their input, which the filter rewrites every
    stage).
  - **Tests:** `test_kokkos_dispersive_kerr.py` and `test_kokkos_dispersive_routes.py`,
    which skip on a device build today, run under device bounds: one evaluation, one
    Runge–Kutta step and ten steps of the fused path's multimode capillary (both
    evaluators, field and envelope, both sectors, the embedded rule's lazy finish), a
    He/Ar mixture of varying composition (the rebuild), nitrogen's Raman beside it; the
    Cartesian free-space and line routes (both evaluators, the four sector cases, beside
    Raman, a partial chunk); the planar route (both sectors, a partial block); the
    general route (the gas strand alone and beside the solid core, both forms); the
    coherent wall flow. The device bound adds the device projection's summation order
    (`docs/dev/backends.md` §11) to the CPU bound; repeatability is bitwise run to run,
    thread invariance does not apply. The ledger tests and
    `dispersive_memory`'s equality hold unchanged on a device.
  - **Measurements:** Q5's stage and ten-step ratios on the device for the fused-path
    cases and the four routes of this slice, and the response's device memory
    (`memory_ledger()`), labelled per device.

### Phase 3c — device qualification

- **Scope:** slice 3c of Phase 3 (§11), 2026-10-08, run by Opus 5.5 (AI) for John
  Travers on the heads of `feature/sep-0006-dispersive-nonlinearity-3` (solprop
  `1bfbd5c`, solprop-kokkos `c5c000b`), on one AMD Instinct device (CDNA 3, gfx942) in a
  12-minute session, and on one NVIDIA Hopper device (`HOPPER90`) in a 32-minute
  session at solprop `3f7cd57` (the same code with this report). The device builds
  compile the same source as the CPU build; no code changed on a device.
- **Delivered:**
  - `tests/accelerator_bounds.py`: `dispersive_device_bound`, the dispersive comparisons'
    device envelope — the predeclared `fft_bound` per transform the stage makes, with
    the CPU bound's count of transforms and node reductions and its factor of two, the
    larger of the CPU and device bounds holding.
  - `tests/backends/test_kokkos_dispersive_kerr.py`, `test_kokkos_dispersive_routes.py`:
    run on a device build under that envelope (they skipped there); the two thread-count
    tests stay host-only (a device build is held run to run).
  - `benchmarks/sep6/data/device/hip/`: the build record, the test counts and every
    cost row; the ledger entry of `docs/dev/benchmarks.md`; `backends.md` §11 and the
    manual's backend notes.
- **Verification (on the device):**
  - The build: `hip` preset, HYBRID GEMM over rocBLAS, rocFFT, Kokkos 5.2.1, Kokkos-FFT
    2.0.0, Kokkos Kernels 5.2.0, ROCm 7.14 prerelease, Release, AMD Clang 23; native
    tests passed (`ctest`).
  - solprop-kokkos's suite: 254 tests, 0 failures, 13 skipped.
  - solprop's dispersive device tests (`test_kokkos_dispersive_kerr.py`,
    `test_kokkos_dispersive_routes.py`, `test_kokkos_gpu.py`, `test_memory.py`): 316
    passed, 2 skipped (the host-only thread tests) — one evaluation, one step and ten
    steps on every route against `numpy_ref` and `numba_cpu` under the device envelope,
    both evaluators and sectors, field and envelope grids, the mixtures (the symmetric
    filter rebuilt on the device; the polynomial's groups), Raman, plasma and the
    coupled wall flow beside the response, bitwise repeatability run to run, and the
    memory estimate equal to the device ledger.
  - `pytest -m "gpu and not slow"`: 2 passed.
  - **CUDA** (`cuda` preset, HYBRID GEMM over cuBLAS, cuFFT, GCC 13): the native tests
    passed; solprop-kokkos's suite 254 tests, 0 failures, 13 skipped; solprop's
    dispersive device tests 316 passed, 2 skipped (the host-only thread tests);
    `pytest -m "gpu and not slow"` 2 passed; and a CUSTOM-GEMM build (the fused path's
    synthesis and projection without a vendor GEMM) passed the two dispersive modules,
    218 passed, 2 skipped.
- **Q5 on the devices:** every case meets it on both. AMD: the symmetric evaluator at
  0.86–1.30 of the constant cubic's stage, the polynomial at 1.02–3.53, including air
  (2.06–2.10), which missed on the host; the planar route's symmetric response sits at
  the target (1.29–1.30). Hopper: 0.95–1.28 and 1.01–3.28 (air 1.50–1.55). The tables
  are the ledger entries.
- **Deviations:** none from the plan's correctness scope. The CUDA session's first build
  stopped on a missing timing utility in the device image, installed before the
  build was rerun; no code changed.
- **Limitations:** one AMD generation on a prerelease ROCm stack and one Hopper device;
  the device departures are held to the envelope, not recorded per case; no device
  sanitizer was run (ROCm ships none; CUDA's was not used).
- **Open questions:** none.
- **Handoff:** Phase 3 is complete on CPU and on one device generation; it merges in
  solprop and solprop-kokkos together. Phase 4 (§11) follows: the full-size
  comparisons of the demonstration cases (§8; the helium RDW per mode count, Phase 1c),
  application guidance, and the end-of-SEP gate and lanes before Final.

### Phase 4 — scientific qualification and closing scope

- **Scope:** Phase 4 of §11, 2026-10-08, implemented by Opus 5.5 (AI) for John Travers
  on `feature/sep-0006-dispersive-nonlinearity-4` (solprop only; solprop-kokkos
  unchanged at main `78add4f`, installed from it). The full-size demonstration
  comparisons of §8 as worked examples, the helium RDW's sensitivity by mode count
  (Phase 1c's open question 1), the application guidance, a recommendation on Q6, the
  closing scope of SEP 0 §6 short of `Final`, and the end-of-SEP gate and lanes. No
  package behaviour changed; the lanes' fixes are listed below.
- **Delivered:**
  - `docs/examples/processes/plot_07_rdw_dispersive_kerr.py` and its generated page:
    the M2 case (1.5 m, He 0.8 bar to vacuum, 7.5 fs, 275 µJ, 800 nm, plasma on) under
    the carrier, symmetric and polynomial policies with four modes — the 100–150 nm
    band along the fibre, the pump's peak power, the exit spectrum with the bounded and
    best-effort bands shaded from the manifest — then with two and one modes, and
    without plasma; its docstring states that the VUV legs make it a sensitivity
    study, not a qualified prediction.
  - `docs/examples/processes/plot_08_duv_fwm_dispersive_kerr.py` and its page, §8's
    optional argon example: 1 m of a 50 µm capillary, 10 µJ pump at 400 nm and seed at
    800 nm, a 19-point pressure scan around the linear phase matching under the three
    policies, and the pump's depletion in photons at each one's best pressure.
  - `benchmarks/sep6/rdw_mode_count.py` and `benchmarks/sep6/data/rdw_mode_count.json`:
    the nine M2 runs (three policies × one, two, four modes) at the default and a
    hundred times tighter step tolerance, on M2's grid and a wider one (70 nm to 6 µm
    over 600 fs), 36 runs.
  - `benchmarks/sep6/fit_uncertainty.py`: §4.7's symmetric errors beside the fit's own
    one-sigma dispersion uncertainty (below).
  - `docs/guide/media/dispersive-kerr.md`: the stage cost on every backend (`numba_cpu`,
    Kokkos CPU, the two device builds) in one table; the full-size RDW figures; a new
    section *When the carrier is enough, and which evaluator to choose* (the measured
    demonstration results, guidance by observable, which evaluator, what the regions
    mean for a result, the backends); *Reading what a run recorded* as its own
    section; a closing link to this SEP.
  - Current pages: `materials.md`, `general-modal.md` (two places), `luna.md` and
    `solver.md` no longer say that Kokkos refuses the response or that the reference
    evaluates per species (it evaluates per symmetric group since Phase 2c);
    `materials.md` ends with a link to this SEP beside SEP 3's; the guide's backend
    table (`docs/guide/index.md`) gains the response's row; the processes gallery's
    `README.rst` names the two new examples.
  - `docs/dev/roadmap.md`: the SEP's entry restated; new informal items *Beyond the
    scalar dispersive electronic response* (tensor dispersion, silica, the GNLSE route,
    the xenon and CO bounded quartets) and *The constant cubic's number-conserving rule
    beyond an octave* (Phase 0's open question 2); the RDW sensitivity added to
    *Resonant gas susceptibilities*.
  - `CHANGELOG.md`: a Phase 4 entry citing SEP 6; the stale "Kokkos refuses it"
    sentences of the Phase 1 and 2 entries removed; the device qualification noted.

#### The demonstration results

All on the default backend, Kokkos CPU (solprop-kokkos `78add4f`, the local profile,
default thread pools), macOS on Apple silicon, Python 3.14, optmat 0.4.0, at a load
average of 15 to 20.

**Helium RDW (M2), the 100–150 nm band at the exit** (µJ; the change is the carrier's
against the polynomial's; centres in nm, carrier / symmetric / polynomial):

| Modes | Carrier | Symmetric | Polynomial | Carrier vs polynomial | Symmetric vs polynomial | Centres |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 6.44 | 9.95 | 9.90 | −35.0 % | +0.43 % | 135.99 / 132.80 / 132.85 |
| 2 | 10.83 | 14.12 | 14.10 | −23.2 % | +0.14 % | 135.22 / 132.75 / 132.80 |
| 4 | 10.35 | 13.54 | 13.52 | −23.5 % | +0.13 % | 134.55 / 132.18 / 132.23 |

- The active band is supported from 151.9 nm, bounded from 120.3 to 151.9 nm and best
  effort from 81.8 to 120.3 nm: the RDW band straddles the last two.
- With four modes the pump's peak power, which oscillates along the second half of the
  fibre with a period of about 7 cm in step under every policy, is 4 % higher under
  the carrier policy (106 against 102 GW at its largest), and 8 % less energy leaves the
  fibre (142.1 against 155.0 µJ). Without plasma (carrier and polynomial only): 128
  against 118 GW, the same 220.3 µJ at the exit, and 12.29 against 14.75 µJ in the band
  (−17 %). The plasma therefore carries part of the electronic response's difference:
  the lower peak ionises less.
- Accepted steps rise by a fifth under the dispersive policies (1870 against 2254 and
  2236 with four modes; a third with two, 7 % with one).
- **Convergence** (`rdw_mode_count.py`): at `rtol=1e-8`, `atol=1e-12` every band energy
  moves by 1–2 parts in 10⁶ and every centre by less than 0.01 nm; on the wider grid
  the band energies move by 0.4–0.9 % and the centres by 0.03–0.06 nm, and the
  polynomial's excess over the carrier, 53.8 / 30.2 / 30.7 % for one, two and four
  modes on M2's grid, is 53.7 / 29.8 / 30.4 %. The differences are the models'.
- **Run times:** the worked example (nine runs with plasma, two without) 1 min 29 s in
  `poe docs-examples`; the 36 benchmark runs 7 min 11 s, 5.5–26 s each.

**The mode-count sensitivity (Phase 1c's open question 1).** At full size the
dispersive policies give the band 54 % more energy than the carrier policy with one
mode and 30–31 % more with two or four; the two- and four-mode figures agree. Phase 1c's
reduced case (1 m, 1.2 bar, 100 nm to 1.5 µm) gave about 60 % with one mode and 9 % with
two: the one-mode sensitivity is the same, the two-mode one is three times larger at
full size. A mode count is part of the observable's definition: the sensitivity is to be
reported with it, and one mode overstates it for this case. The physical reason the
mode set matters this much was not isolated.

**Argon DUV FWM**, 50 µm, 1 m, the linear phase matching of HE₁₁ at 209.4 mbar:

| | Carrier | Symmetric | Polynomial |
|---|---:|---:|---:|
| best pressure (scan step 5.2 mbar) | 204.1 mbar | 193.7 mbar | 193.7 mbar |
| largest signal along the fibre there | 3.33 µJ | 3.29 µJ | 3.27 µJ |
| the same at 204.1 mbar; at 209.4 mbar | 3.33; 3.22 µJ | 3.01; 2.68 µJ | 3.01; 2.68 µJ |
| signal at the 1 m exit, at the own best pressure | 2.37 µJ | 2.69 µJ | 2.69 µJ |

- The carrier policy places the best pressure 10 mbar (5 %) high, overstates the
  signal by 11 % at its own best pressure and by 20 % at the linear phase matching, and
  agrees on the best yield to 2.5 %. The symmetric evaluator is within 0.4 % of the
  polynomial at the best pressure and 1.1 % at worst across the scan.
- At each best pressure the signal peaks near 0.5 m with 0.22 signal photons per
  launched pump photon (44 % of the pump's photons used) and about a quarter of the
  pump band's energy left; beyond that it falls.
- **Why it adds to the tutorial:** a fibre twice as long (the signal's turn-over and
  decline), the pressure scan (the operating point moves; the best yield does not) and
  the pump depletion in photons. The tutorial's half metre at the linear pressure shows
  only the 20 % overestimate.
- **Run time:** 5 min 36 s in `poe docs-examples` (60 runs).

**The fit's own uncertainty** (`fit_uncertainty.py`; Phase 1c deferred this map to
Phase 4). On §4.7's quartets, the one-sigma uncertainty of each quartet's dispersion
$\gamma(q)/\gamma_0$ from the marginal covariance of optmat's $(A, B, B')$ (Birge-scaled),
against the symmetric error, in per cent, median / maximum, and the fraction of
quartets whose symmetric error exceeds it:

| Gas | Symmetric error | One sigma | Error above sigma |
|---|---|---|---:|
| He | 0.006 / 0.124 | 0.015 / 0.109 | 0.26 |
| Ne | 0.007 / 0.153 | 0.141 / 0.274 | 0.00 |
| Ar | 0.053 / 0.913 | 0.162 / 0.366 | 0.21 |
| Kr | 0.115 / 1.714 | 1.066 / 3.197 | 0.00 |
| Xe | 0.260 / 3.212 | 0.481 / 2.682 | 0.32 |
| H2, D2 | 0.047 / 0.718 | 0.063 / 0.499 | 0.42 |
| N2 | 0.047 / 0.837 | 0.209 / 0.437 | 0.10 |
| O2 | 0.183 / 2.908 | 0.500 / 1.415 | 0.23 |
| CH4 | 0.133 / 1.831 | 0.404 / 1.353 | 0.17 |
| SF6 | 0.105 / 2.141 | 0.491 / 1.019 | 0.11 |
| N2O | 0.048 / 0.656 | 2.265 / 5.109 | 0.00 |

The symmetric factorisation's typical error is below the fit's statistical uncertainty
of the dispersion in every gas; its largest errors, at the widest-spread quartets, exceed
it in a tenth to two fifths of the quartets for most gases. The figures depend on the
anchor: against the pump's own Kerr value rather than $\gamma_0$ the correlated part
cancels and the uncertainty is several times smaller. A covariance is not a bound on
the truncation or on borrowed priors (§4.8). Published in the manual at John Travers's
decision (open question 4), relative to $\gamma_0$ with the anchor stated.

#### The guidance written

`docs/guide/media/dispersive-kerr.md`, *When the carrier is enough, and which evaluator to
choose*:

- **By observable:** self-phase modulation near the carrier is exact under every
  policy, and the carrier policy is adequate where the Kerr diagonal varies little over
  the occupied band (12 % from 2 µm to 400 nm in helium, 37 % in argon, 87 % in xenon,
  §4.7); a band generated far from the carrier (a dispersive wave, an FWM signal, UV
  from a near-infrared pump) needs a dispersive policy; a phase-matched wavelength moves
  by a few nanometres; an operating point moves by several per cent while the best yield
  hardly changes.
- **Which evaluator:** `"symmetric-kerr"` for production and scans (the cost of the
  carrier policy, within half a per cent of the polynomial on the RDW and about one on
  the argon scan); `"gas-polynomial"` as the reference, for checks in the most
  dispersive gases and for mixtures sharing a leg bound.
- **The regions for a result:** supported is exact relative to the fit; bounded is an
  estimate read as a lower bound; best effort is a sensitivity, reported with the
  carrier result beside it, never as a correction.
- **Backends:** every backend and every route but the GNLSE; device builds qualified.

#### Q6 — a recommendation for John Travers (no default changed)

**Recommendation: keep `"carrier-kerr"` as the default (`"recommended"` unchanged), and
recommend `"symmetric-kerr"` in the manual for broadband, multi-colour and
UV-generating gas runs, as now written. Revisit when a resonant VUV response of helium
exists or when per-gas nondegenerate measurements qualify the bounded region.**

- *For a switch:* inside the data the dispersive policies are exact relative to the
  fit and the carrier approximation's errors are real (argon: the operating point 5 %
  off, the signal 11–20 % too high at a given pressure); the symmetric evaluator costs
  0.8–1.35 times the constant cubic's stage on every backend, and a whole four-mode RDW
  run 1.35 times on Kokkos CPU here (one timing on a loaded machine), most of it a fifth
  more accepted steps; §4.10 makes it never less
  sound than the carrier.
- *Against a switch now:*
  1. The dispersive policies refuse what the carrier policy admits — a supplied species
     without optmat's model, a constant `KerrResponse` declared beside the selection,
     the GNLSE route. As the default they would turn those runs into refusals, or make
     `"recommended"` mean different laws by route and species, which §5 (no automatic
     model switching) and §9 rule out.
  2. Every gas run's results change — the RDW energy by 30 %, every adaptive step
     history — including the worked examples, tutorials and the comparisons with stored
     witnesses; reproducing an earlier result would need `electronic="carrier-kerr"`.
  3. The largest effects found are in best-effort regions (helium's VUV), where the
     dispersive result is a sensitivity, not a better prediction; as the default it
     would present a held constant as the expected answer.
- *Options:* (A) keep the carrier default (recommended); (B) `"recommended"` becomes
  `"symmetric-kerr"` for every gas with optmat's model, with an explicit rule for
  supplied species, a declared `KerrResponse` and the GNLSE route; (C) select by the
  run's band — excluded by §9.

#### The gates

Run once, sequentially, at solprop `d441363` (this branch's examples, guidance and
benchmarks committed), solprop-kokkos `78add4f` installed, `SOLPROP_KOKKOS_THREADS=1`,
`OMP_PROC_BIND=false`, on a workstation at a load average of 15 to 21:

- **`poe check`** (13 min 38 s): ruff, the formatter, basedpyright, numpydoc and
  codespell clean; the default lane 14 527 passed, 28 skipped and **1 failed**:
  `tests/test_distribution.py::test_loadgroup_reuses_module_fixtures_and_spreads_independent_cases`,
  whose subprocess starts two xdist workers under a 45 s timeout and timed out under
  the load. It passes alone and in the targeted run below; it was not changed. The
  sequence stopped there, so the gate's docs step did not run in it; the docs builds
  of `poe docs-examples` had found one nitpick warning, a `:func:` role on the private
  `_route_bytes` in `dispersive_memory`'s docstring (Phase 3a), fixed as a literal
  (`d441363`, before the lanes).
- **`poe test-slow`** (9 min 29 s): 137 passed, 1 skipped, **1 failed**:
  `tests/driver/test_stepindex_gnlse.py::test_stepindex_strand_propagates_on_both_cpu_backends`,
  one sample of 2048 at 2.1e-7 of the largest against its bound of 2e-7 (`numpy_ref`
  against `numba_cpu` over fifteen centimetres of a step-index GNLSE strand, a route
  SEP 6 does not touch). Rerun alone through `poe test-slow` on that module it passes
  (2 passed, 36 s). The likely cause is `numba_cpu`'s FFTW planning under
  `FFTW_MEASURE`, which picks its algorithms by timing and so its rounding by the
  machine's load; not changed here (open question 5).
- **`poe cov`** (12 min 53 s): **99.83 %, 46 lines and branches uncovered**, and the
  same `test_distribution` timeout. All but three lines were SEP 6's: error paths and
  rare branches its phases added and never ran through the lane. Fixed in `96ebb05`
  with tests: `DispersiveKerrModel`'s malformed coefficients, a model without a
  support record and a record of another model; `DispersiveKerrTerms`'s field
  validation, a problem's leaf on the wrong route, an unknown population, an
  undeclared general region, a carrier flag against its sector, a dispersive response
  beside a TDSE gas or beside a general guide's own channels, the node buffers without
  the leaf; `numpy_ref`'s pass and general source refusing mismatched densities and
  carriers; a general node chunk holding no gas node, on both CPU backends; the
  general route's region notice; `numba_cpu`'s embedded rule finished once, and
  absent on the radial route; `GasKerr`'s display. One line was dead: the constant
  Kerr components' own thg check, which `compile_responses` has preceded with the
  electronic responses' since Phase 1a; it is removed. The other two lines were
  SEP 12's, the TDSE build record of an installed `solprop-tdse` distribution (not
  installed here), now tested with a stand-in distribution.
- **The Kokkos-selected slow lane** (`SOLPROP_TEST_BACKEND=kokkos poe test-slow`,
  8 min 34 s): 139 passed.
- **The fixes were verified by targeted runs, not a second lane run** (on John
  Travers's instruction a rerun of the lanes was stopped): ruff and basedpyright on the
  changed files; `pytest -n auto` on the eleven affected modules with the failing
  `test_distribution` case, 549 passed; a coverage run of the eight dispersive and TDSE
  modules showing every listed gap covered but the general route's region notice and
  undeclared region, whose two tests were added after it and pass.
- **Documentation:** `poe docs-examples` built both worked examples (its build reports
  two unreadable tutorial images, an artefact of that mode, which executes no
  tutorial); with this report, `tests/test_seps.py` (127 passed, 1 skipped) and
  `poe docs` (every tutorial executed, no warning) pass.

#### Deviations from Part I and the phase's scope

- **Branched from `main` at `03d046e`**, Phase 3's merge `73eb312` plus a numba-cache
  fix merged since, so that the lanes run on the tree that will be merged.
- **A companion worked example, not an extension of `plot_00`:** nine runs, a
  mode-count study and a plasma check would bury the canonical example and the README
  figure it draws.
- **The argon example's geometry is the tutorial's** (50 µm, 10 µJ each), not Durfee
  et al.'s; one mode only.
- **The symmetric evaluator was not run without plasma** (only the carrier and the
  polynomial).
- **Fit-uncertainty maps** (§7, deferred by Phase 1c) are published as a table in the
  manual, not drawn as maps.

#### The rest of this report

- **Sources checked:** Part I §2, §4.7, §7, §8, §11, §12; every Part II report;
  optmat 0.4.0's `HyperpolarizabilityModel` (parameters, covariance, reduced
  chi-squared, `gamma_sigma_au`'s Birge scaling) read from the installed package.
  Durfee et al. [9] for the process only. No new literature.
- **Limitations:**
  - Timings are from one loaded workstation; the run times above are wall times of the
    examples, not benchmarks.
  - The RDW comparison is a sensitivity of an observable to unqualified VUV legs; the
    argon comparison is exact relative to optmat's fit but carries the fit's own
    uncertainty and one mode.
  - The pressure scan resolves the best pressure to one step, 5.2 mbar.
  - Why the RDW's sensitivity depends on the mode set was not analysed.
  - `plot_00_vuv_dispersive_wave` was not re-run (its source is unchanged): its page
    still shows its output from before SEP 16 (9.86 µJ in 100–150 nm), where the
    companion's carrier run with the same physics gives 10.35 µJ.
- **Open questions** (every earlier one is now answered, on the roadmap or below):
  1. *Resolved (John Travers, 2026-10-08):* **Q6** — the carrier policy stays the
     default; the manual recommends `"symmetric-kerr"` for broadband or UV-generating
     gas runs.
  2. *Resolved:* no output-schema bump (Phase 1a's open question 3).
  3. *Resolved:* a constant `KerrResponse` beside a dispersive selection is refused on
     every route (Phase 1a's open question 4).
  4. *Resolved:* the fit-uncertainty comparison is published in the manual, relative to
     $\gamma_0$, with the anchor's effect stated.
  5. *Resolved:* the two load-sensitive tests were made load-independent rather than
     widened: the step-index strand test plans `numba_cpu`'s transforms with
     `FFTW_ESTIMATE` (6.9e-9 against its unchanged 2e-7 bound), and
     `test_distribution`'s subprocess and rendezvous limits, which are hang guards,
     are long enough that a loaded machine's slow worker start cannot trip them.
  - *Answered:* Phase 1a's open question 2 (Q5's targets are for the compiled stages;
    `numpy_ref` is the oracle, measured in Phases 2 and 3); Phase 1c's open question 1
    (above); Phase 2a's open question 5 (the planar-hybrid tutorial runs on `main`
    under the step controller). *On the roadmap:* Phase 0's open question 2 (the
    constant cubic's $\Pi_+$); §12 Q2's xenon and CO quartets; the tensor, silica and
    GNLSE scope; the resonant VUV response; the polynomial's mixture cost. *Outside
    solprop:* §12 Q2's O2 spin-forbidden states (optmat's). *Outside the SEP:* Phase 3b's
    open question 4, the crash at interpreter exit after both compiled backends ran; a
    minimal probe here (a Kokkos and a `numba_cpu` run in one process) exited cleanly.
- **What remains to set `Final`:** John Travers's decisions on the questions above; then
  the `Final` pull request of SEP 0 §6 — status, the **Manual pages** field,
  `git mv` to `archive/`, the index row and toctree entry moved, the relative links
  fixed (the manual pages link `../dev/seps/sep-0006-…`), the roadmap's SEP entry
  removed (its informal items stay), and the CHANGELOG line (it cites SEP 6).
- **Handoff:** Phase 4 merges into solprop alone; solprop-kokkos is unchanged.

## Part III — History

- 2026-09-27: Drafted with two proposed dispersive evaluators, gas quartet
  comparisons, exploratory cost measurements and an explicit broadband
  harmonic-selection qualification question.
  Renumbered from 4 to 6 before initial merge to avoid concurrently assigned SEPs.
- 2026-09-28: Review revision. Corrected the gas-fit count (twelve of optmat's
  fifteen, solprop's inventory), stated the intensity convention and optmat's
  $n_2$ form, mapped $S_2/S_4$ to optmat's $W_2/W_4$, added the symmetric band
  cut-off rule, the harmonic-leg domain table, the mixture-curve filter for
  the symmetric model, the output-only median and 95th-percentile columns,
  and the matching test rows. All §4.7 and §4.9 numbers were reproduced
  independently from the stored fits.
- 2026-09-29: Recorded the 2026-09-28 review by Fable 5.1 (AI) in the new review
  fields. Review status is `In review`: a human review is still needed before
  acceptance.
- 2026-10-07: Final review by Opus 5.5 (AI) and review by John Travers (human);
  review status `Reviewed`. Re-audited against solprop `b70f2fb`, optmat
  `v0.3.0` and solprop-kokkos `fa5d57c` (SEP 3 final, kernel schema 23, output
  schema 15, `Multistep()` the default integrator, SEP 12's TDSE response
  final). At the reviewers' request: the gas scope follows optmat's fifteen
  hyperpolarizability fits rather than a solprop list, with CO2, CO and O3
  added to §4.7–§4.8; the argon DUV FWM and helium VUV RDW (M2) demonstration
  cases were added with their coefficient table and run through qualification,
  the tutorial and the manual; and the solprop-kokkos implementation is
  required, with no exception. Also: §5 written against the delivered
  `MaterialSelection.electronic`, the numerical band separated from the
  material support (§4.5), composition with the TDSE response stated, and
  §9's persisted record made concrete. Every §4.7 survey row, FWM row and the
  §4.9 field-grid diagnostic were reproduced exactly on the current tree.
  Same day, after optmat's analysis of where its fits are supported (optmat
  `a9e5dbf`, OEP 5 draft `0d5a0ab`): added §4.10. Every leg is bounded at
  $\sqrt{3S_{2,\max}/4}$, which is exact within the fitted sums. Runs are never
  refused for the frequencies their field occupies, as helium runs reaching
  below 80 nm require. The band is labelled supported, bounded or best effort
  against each gas's resonance band edge. Q2 is answered in substance, with
  its remaining items listed. The demonstration-case table gained the
  bounded values and band-edge column, with a paragraph on how pump dynamics
  and VUV emission share one response. John Travers confirmed the policy
  "label, never refuse" for every gas, O3 and the polyatomics included.
- 2026-10-07: Accepted by John Travers, whose review was the human review
  recorded above. Part I is frozen. Phase 1 needs an optmat release of the
  support record proposed by optmat's OEP 5.
- 2026-10-07: Phase 0 delivered: the operators' field, envelope and vector forms in
  both harmonic sectors, an independent triplet-sum oracle with the algebra tests, and
  the material record reproduced through optmat's public models. John Travers
  accepted its decisions on Q1, Q3, Q4 and Q5; Part I's "lower bound", band-edge and polyatomic
  statements are qualified in the report after optmat's OEP 5 review. Status
  `In progress`.
- 2026-10-07: Phase 1, slice 1a delivered: both dispersive policies on `numpy_ref` for
  the single-mode, multimode and radial routes, kernel schema 24, the manifest's
  dispersive law and regions, per-species mixture channels, and refusals by name on the
  other routes, on `numba_cpu` and on Kokkos.
- 2026-10-07: Phase 1, slice 1b delivered: `numpy_ref` evaluates the dispersive response
  on the free-space, line, planar and general modal routes, a general modal region
  declaring its gas fill (`GasKerr`) under the run's electronic policy; `numba_cpu` and
  Kokkos refuse it by name.
- 2026-10-07: Phase 1, slice 1c delivered: the manual chapter on the frequency-dependent
  electronic response, the tutorial comparing the three policies on the argon DUV FWM
  and helium RDW cases, and their ten-step demonstration tests on `numpy_ref`.
- 2026-10-08: Phase 1 merged in solprop and solprop-kokkos (kernel schema 24). John
  Travers accepted `GasKerr`, the unversioned `regions` field, three-component
  `evaluate` and the 40 s tutorial.
- 2026-10-08: Phase 2, slice 2a delivered: `numba_cpu` executes the dispersive response
  on its fused node path (single-mode, multimode, radial; field and envelope; both
  evaluators and sectors; mixtures; beside Raman and plasma), the multipliers acting on
  the modal columns, with the Q5 measurements; the free-space, planar and general modal
  routes stay refused by name for slice 2b.
- 2026-10-08: Phase 2, slice 2b delivered: the collar applied to the dispersive products
  before the output multipliers (`G[w (G E)^3]`) in the reference and `numba_cpu`, and
  the node chunk counting the response's buffers, both decided by John Travers that day;
  `numba_cpu` on the Cartesian free-space, line, planar and general modal routes, its
  refusal removed; the polynomial's species of equal leg bound grouped, an exact
  rewriting; the symmetric mixture-curve filter measured for a decision; the tutorial on
  `numba_cpu`; Q5 measured again on every route.
- 2026-10-08: Phase 2 merged. Phase 2, slice 2c delivered: the symmetric evaluator of a
  mixture takes one filter from the mixture's own Kerr diagonal (§6), on `numpy_ref`
  and `numba_cpu` and every route, rebuilt as the composition changes along z and
  recorded with its composition in the manifest and the resume key, with the exact
  per-species sum its test reference; the general modal route's node chunk counts the
  response's buffers; both decided by John Travers that day. Q5 met for every
  symmetric mixture.
- 2026-10-08: Phase 2 complete: slice 2c merged (the symmetric evaluator's mixture
  filter, the general route's node budget); its open questions decided by John
  Travers. Phase 3 (Kokkos) begins.
- 2026-10-08: Phase 3, slice 3a delivered: Kokkos CPU executes the dispersive response
  on its fused node path (single-mode, multimode, radial; field and envelope; both
  evaluators and sectors; mixtures under the symmetric mixture filter; beside Kerr,
  Raman and plasma) with `numba_cpu`'s structure, against `numpy_ref` and `numba_cpu`;
  the memory estimate counts the response's buffers on the fused and spatial routes
  (Phase 2c's decision 3). The free-space, line, planar and general modal routes stay
  refused by name on Kokkos for slice 3b; the device build is slice 3c's.
- 2026-10-08: Phase 3, slice 3a's open questions decided by John Travers (all five
  accepted). Slice 3b delivered: Kokkos CPU executes the dispersive response on the
  Cartesian free-space and line routes, the planar route and the general modal route
  with `numba_cpu`'s per-chunk structure, and beside a coupled wall flow, against
  `numpy_ref` and `numba_cpu`; Kokkos refuses none of the response's routes, and the
  memory estimate gives its rows on each to the byte. The device build is slice 3c's.
- 2026-10-08: Phase 3, slice 3b's open questions decided by John Travers; slice 3c
  qualified the device builds on one AMD CDNA 3 and one NVIDIA Hopper device: every
  dispersive test passes under the predeclared device envelope, and Q5 is met in every
  case on both.
- 2026-10-08: Phase 4 delivered: the helium RDW (M2) and the argon DUV FWM run at full
  size under the three policies as worked examples; the RDW's sensitivity by mode
  count (54 % with one mode, 30–31 % with two or four, converged in step and grid);
  the application guidance in the manual; a recommendation on Q6 (keep the carrier
  default) for John Travers's decision; the closing scope short of `Final`; the
  end-of-SEP lanes, run once, their coverage gaps fixed and verified by targeted runs.
- 2026-10-08: Phase 4 merged with John Travers's decisions (Q6: the carrier policy stays
  the default; the fit-uncertainty comparison published). **Final**, approved by John
  Travers; moved to the archive. Its remaining work is on the roadmap.
