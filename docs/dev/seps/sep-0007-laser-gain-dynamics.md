# SEP 7 — Laser gain dynamics in solid-core fibres

| | |
|---|---|
| **SEP** | 7 |
| **Title** | Laser gain dynamics in solid-core fibres |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; Opus 5.5 (2026-10-26, a note in §4.6) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | — |
| **Requires** | [SEP 3 — Materials, Raman responses and ionisation rates from optmat](archive/sep-0003-optmat-materials.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add coupled laser population, pump and signal dynamics to solid-core fibre propagation,
with Yb-doped silica and gain-managed nonlinear amplification (GMNA) as the first
application. Retain the complex signal field, dispersion, Kerr and Raman physics while
solving stimulated absorption and emission, pump depletion, reabsorption, spontaneous
decay and amplified spontaneous emission (ASE). Provide a conventional, spectrally
resolved steady population model and deterministic time-dependent rate equations,
including recovery between pulses. Separately qualify the broadband filtered-field
closure of Chen and Wise [1, 2]; its use is a physical-model choice, not a backend
optimization. Begin with a mode-averaged signal and a declared pump profile, then resolve
transverse populations and coherent scalar modes. Spectroscopy belongs in optmat; gain
equations and propagation belong in solprop. Success requires conservation and
convergence tests, independent rate-equation references, useful GMNA and partial
parabolic amplification examples, and parity on NumPy, Numba and Kokkos CPU. This is a
substantial extension of the solver and its state contract, not an additional prescribed
linear gain curve.

### 2. Motivation and scope

#### Status and evidence

**Draft; no gain solver has been implemented or timed.** The architectural assessment
uses solprop at `fe404ac4df1b3344907ed1ac63625aab8cb54426` and solprop-kokkos at
`87ac816190ffb0c3978de7e236b6e30d7c068ab8`. The material assessment uses optmat at
`2dee887f0b8104ad152b5b59a17b553013ff40c5`. Equations below are re-derived in solprop's
conventions. The source register in §13 distinguishes papers read in full from resources
still needed for data curation. Published simulations are validation targets, not the
definition of a correct implementation.

GMNA uses the evolution of a spectrally structured gain and absorption profile together
with nonlinear spectral broadening [3]. It cannot generally be described by multiplying
a fixed gain spectrum by a pulse-energy saturation factor. Wang et al. [4] distinguish
GMNA from partial parabolic amplification and its gain-managed form (GMPPA). Their
flat-gain study is explanatory; their realistic Yb study needs spectrally resolved gain.
The realistic model cites the steady rate-equation treatments [5, 6]. Consequently,
reproducing that study does **not** independently validate the transient closure in
Chen and Wise [1]. Both are valuable targets, with different purposes.

The initial scientific deliverable is a nonlinear Yb amplifier with physically specified
pumping and doping. A user must be able to ask whether a steady calculation suffices,
how a burst depletes the inversion, how it recovers, and how the resulting spectrum and
phase affect compression. Output energy alone is inadequate evidence.

#### Required scope and staged extensions

| Capability | Proposed scope |
|---|---|
| Material | Yb in a specifically identified silica host, with two thermally equilibrated manifolds; a typed transition structure extensible to more levels |
| Signal | One coherent forward signal initially; mode-averaged GNLSE and scalar modal propagation, retaining existing Kerr, Raman and dispersion |
| Geometry | Fixed step-index fibre first; explicit doped radius/profile and core or cladding pumping; radial population resolution and several coherent scalar modes subsequently |
| Population timing | Steady average-power operation; specified initial inversion; isolated pulses, finite bursts and periodic pulse trains, with population memory across gaps |
| Pump | One or more incoherent narrowband channels, co-, counter- or bidirectional; explicit spatial overlap, depletion and boundary conditions |
| ASE | Deterministic spectral power transport in both directions, including depletion and an explicit modal/polarization count |
| Broadband transient signal | A separate filtered-field model, conditional on the qualification decision in §§4.5 and 11; no automatic substitution for the conventional model |
| Execution | NumPy reference, Numba CPU, Kokkos CPU; GPU qualification separately |

Here **full rate equations** means time-dependent populations for every level declared
by the material model and every included transition. It does not mean resolving all
Stark sublevels, optical coherences or an arbitrary ion's unknown kinetics. For Yb,
thermal equilibration within each manifold is an assumption, and the absorption and
emission cross-sections need not be equal. Models with excited-state absorption,
cross-relaxation, upconversion or concentration quenching require additional measured
data and their own qualification. A generic transition table does not make Er, Tm, Ho
or Nd automatically supported.

The mandatory transient baseline is the conventional narrowband/flat-cross-section
limit; the broadband steady model already addresses many GMNA experiments. Broadband
intrapulse saturation is the extra scientific deliverable under investigation, not
something the steady model is claimed to solve. Before acceptance, decide whether the
filtered-field model's qualification is satisfactory or retain that phase as an explicit
research gate. Failure of that gate requires a revised proposal; it must not be hidden
by calling a steady solver “full transient”.

Full vector gain on arbitrary general modal guides, moving guides, coherent
counterpropagating signals, oscillator round trips, spatially resolved pump ray tracing,
thermal lensing, photodarkening, damage, dopant-induced mode reshaping, and stochastic
ASE interacting with Kerr are follow-up work. A rod or photonic crystal fibre may be
represented by supplied, qualified effective modal data; this SEP does not add a mode
solver for those structures. Neglecting ASE is an explicit approximation requiring a
small-ASE comparison, not a default justified by the presence of a seed.

### 3. Context

The current [solid-fibre guide](../../guide/waveguides/solid-fibres.md),
[GNLSE](../../guide/waveguides/gnlse.md) and [general modal model](../../guide/waveguides/general-modal.md)
already provide much of the passive propagation physics. The missing functionality is
the “Gain with saturation” item in the [roadmap](../roadmap.md). Prescribed gain in a
linear operator supplies neither populations nor a pump energy budget.

The first delivery should extend the existing mode-averaged GNLSE interface, supplying
gain overlap independently of its Kerr area. The current scalar `from_waveguide` path
expects gas-filled modes; solid step-index fibres instead use the GNLSE reduction or
the general vector provider. Consequently the later scalar multimode gain milestone
needs an explicit solid-material projection/admission extension. It cannot be obtained
by attaching a gain flag to the gas-filled route. A step-index-derived dispersion and
Kerr coefficient do not by themselves specify the dopant/pump overlap.

The relevant boundaries are:

- [Architecture](../architecture.md): physical models, translation and numerical
  backends remain separate. A pump/population boundary-value problem needs an amplifier
  orchestration layer, not material knowledge inside FFT kernels.
- [Stepper](../stepper.md): the commuting linear phase is state independent. A gain
  spectrum determined by the pulse or by time-dependent populations is not generally
  such an operator. A frozen-gain calculation is permissible only within a converged
  outer solve, or as a separately qualified splitting.
- [Kernel contract](../kernel-contract.md): `KernelProblem`, stage bundles and the
  protected `StepDriver` surface are versioned together; the audited schema is 15.
  `src/solprop/backends/driver.py` currently accepts or rejects a field trial. Gain adds
  coupled state and rollback obligations, so implementation requires a schema bump.
- [Conventions](../conventions.md): ordinary modal internal amplitudes are
  vacuum-normalised, whereas interface amplitudes are in true square-root watts.
  Population rates must use physical optical flux, not the unconverted internal array.
- [General modal propagation](../../guide/waveguides/general-modal.md): the present provider route
  uses physical vector fields, a flux metric, and a field grid. It cannot be advertised
  as a scalar envelope gain route without a new derivation and implementation.

[SEP 3](archive/sep-0003-optmat-materials.md) establishes the dependency and provenance pattern
for optmat, but its scope is gases and passive silica properties. It does not supply
rare-earth spectroscopy. This proposal needs an upstream gain-spectroscopy extension
in addition to SEP 3. [SEP 32](sep-0032-raman-maxwell-bloch.md) proposes a Raman
Maxwell–Bloch response; its molecular parameters and population normalization are not
rare-earth gain data. Its causal-response reasoning may help validation, but it is not
a prerequisite to the conventional amplifier solver.

### 4. Theory

#### 4.1. Variables, normalization and physical assumptions

Use SI throughout. Let $N_d(\mathbf r,z)$ be the active-ion density in
$\mathrm{m^{-3}}$, $N_i$ a manifold population in the same units, and
$\sum_i N_i=N_d$. Cross-sections $\sigma_j(\omega,T)$ are in $\mathrm{m^2}$,
lifetimes in seconds, angular frequency $\omega>0$ in $\mathrm{rad\,s^{-1}}$,
and pump and ASE channel powers in watts. $t$ is retarded pulse time;
$t_{\mathrm{lab}}=t+\beta_{1,\mathrm{ref}}z$ is laboratory time. There is no
second use of the FFT window length as a repetition period.

For the scalar derivation let $b_m(t)$ be a power-normalised modal amplitude and
$F_m(\mathbf r,\omega)$ a profile with
$\int |F_m|^2\,dA=1$. Thus $F_m$ has units $\mathrm{m^{-1}}$, and the local
scalar amplitude $A=\sum_m F_m b_m$ has units $\sqrt{\mathrm W}/\mathrm m$.
When profiles depend on frequency, synthesize this field spectrally before the inverse
transform. A fixed profile is an explicit approximation, especially for a spectrum
broadened far from the seed. The interface Fourier convention is

$$
\widetilde A(\omega)=\int A(t)e^{+i\omega t}\,dt,\qquad
A(t)=\frac{1}{2\pi}\int\widetilde A(\omega)e^{-i\omega t}\,d\omega.
$$

An envelope uses its **absolute** optical frequency $\omega_0+\Omega$ in
cross-sections and photon energies. The backend uses the existing conjugation boundary
and raw-DFT normalization; it introduces no additional sign convention. Rates on a
real-field grid use the analytic positive-frequency field, not the square of a real
carrier oscillation. DC and nonphysical frequencies carry no stimulated transition.

The squared scalar amplitude used below defines the chosen gain closure's irradiance.
It must not silently replace solprop's temporal Poynting statistic: for dispersive modal
normalization, that statistic is
$\operatorname{Re}[a(t)a_w^*(t)]$, with $a_w$ synthesized from $Wa$, rather
than $|b(t)|^2$, where $W=\operatorname{Re}n_{\mathrm{eff}}$ for ordinary
modal normalization. Their time integrals agree but their unequal-frequency interference
weights differ. The initial GNLSE route has the existing $W=1$ calibration; later
adapters must quantify this distinction and preserve the documented output observables.

Assume fixed temperature, weak perturbation of the passive guide, independent active
ions for the initial Yb model, and rapid relaxation within each manifold compared with
the population dynamics being modelled. Optical coherence is eliminated. Extremely
short pulses, coherent Rabi dynamics, spectral hole burning within a manifold and
nonthermal sublevel populations are outside this rate model. The material record must
state where those approximations have evidence; “two manifolds” does not establish a
universal minimum pulse duration.

#### 4.2. Conventional rates and the steady limit

For mutually incoherent spectral channels with nonnegative irradiance density
$I_\omega$ satisfying $I=\int I_\omega d\omega$, the absorption or emission
rate per ion is

$$
R_j(\mathbf r,t)=\int_0^\infty
 \frac{\sigma_j(\omega)}{\hbar\omega}I_\omega(\mathbf r,t)\,d\omega.
$$

The dimensions are $\mathrm{s^{-1}}$. Different incoherent pump channels and ASE add
as rates. A transition from level $u$ to level $v$ contributes $-R_jN_u$ to
$\dot N_u$ and $+R_jN_u$ to $\dot N_v$. Decay contributes the corresponding
branching rates; pair-transfer terms have coefficients in $\mathrm{m^3\,s^{-1}}$
and population products. A stoichiometric representation makes ion conservation
explicit. Only transitions backed by data are enabled.

For Yb define $f=N_2/N_d$, with $N_1=N_d(1-f)$. Summing pump, signal and ASE
rates gives

$$
\partial_t f=R_a-(R_a+R_e+\tau^{-1})f.
$$

Here $\tau$ is the **total** upper-manifold lifetime under the specified conditions;
radiative branching is tracked separately where needed. With constant rates,

$$
q=R_a+R_e+\tau^{-1},\qquad f_{\mathrm{ss}}=\frac{R_a}{q},\qquad
f(t+\Delta t)=f_{\mathrm{ss}}+
 [f(t)-f_{\mathrm{ss}}]e^{-q\Delta t}.
$$

This supplies an exact reference and a positivity-preserving update for piecewise
constant rates, **not** a claim of fourth-order accuracy for time-varying rates. Use
stable exponential differences near $q\Delta t=0$ and treat the $q=0$ limit
explicitly. For a prescribed sequence of rates, compose affine maps
$f_{\mathrm{out}}=a f_{\mathrm{in}}+c$. The periodic solution is
$f_{\mathrm{in}}=c/(1-a)$ when the map contracts. Use a stable evaluation of
$1-a$; a map with no relaxation has no unique periodic inversion. This avoids
optimizing a scalar initial population with a general minimizer. Coupling rates back to
the propagating signal and pump still requires an outer solution.

For a stationary pulse train with period $T_{\mathrm{rep}}$, the conventional
average signal rate at a spatial point is

$$
\overline R_j=
\frac{1}{T_{\mathrm{rep}}}\int_0^\infty
 \frac{\sigma_j(\omega)}{\hbar\omega}
 |\widetilde A(\omega)|^2\frac{d\omega}{2\pi}.
$$

This retains same-frequency interference between coherent modes through
$|\sum_m F_m\widetilde b_m|^2$; it does not discard the phase of the propagated
signal. It is compatible with both a GNLSE and a unidirectional field equation [5, 6].
On an envelope grid with $N$ samples and raw FFT coefficients $\widehat b_k$, a
modal energy bin is $\Delta t|\widehat b_k|^2/N$. Multiply by
$1/T_{\mathrm{rep}}$, not by $1/(N\Delta t)$, for its average power.
The field-grid analytic factors, including the factor four for interior positive
frequency bins, follow [conventions §4](../conventions.md).

The steady approximation neglects population variation within a period. High repetition
rate alone does not justify it: pulse extraction must also leave the gain sufficiently
unchanged. For uniform doping and one mode, a useful sensitivity measure is

$$
\delta\ln G(\omega)=\int dz\,
 \Gamma(\omega)N_d[\sigma_a(\omega)+\sigma_e(\omega)]\,\delta f(z).
$$

Compare this and output observables with a periodic transient calculation. A small
fractional change in $f$ can matter greatly near transparency. The steady solver
should report effective recovery rates and extraction diagnostics, without inventing a
universal repetition-frequency threshold.

#### 4.3. Spatial averaging, depletion and gain

Let $u_m=|F_m|^2$ and $\Gamma_m(\omega)=\int_{A_d}u_m\,dA$, where $A_d$
is the doped area. Uniform population averaging over that area gives the per-ion rate
from a narrowband mode of power $P_m$:

$$
R_{j,m}=\frac{\Gamma_m\sigma_j P_m}{A_d\hbar\omega_m},\qquad
g_m(\omega)=\Gamma_m N_d[
 \sigma_e(\omega)f-\sigma_a(\omega)(1-f)].
$$

$g$ is a **power** gain coefficient in $\mathrm{m^{-1}}$, so the field coefficient
is $g/2$. The saturation area is governed by $A_d/\Gamma$, not by the Kerr area
$A_{\mathrm{eff}}=(\int|F|^2dA)^2/\int|F|^4dA$. These quantities agree only
under particular approximations. Uniform cladding pumping gives
$\Gamma_p=A_d/A_{\mathrm{clad}}$; a nonuniform or core pump requires its own
normalized profile. Neither pump launch efficiency nor background absorption is an ion
cross-section.

Mode averaging assumes a common inversion throughout the doped area. A radial shell
model retains local populations and resolves spatial hole burning even for one signal
mode. The same quadrature and profiles must enter depletion and signal gain, otherwise
their photon budgets disagree. For several scalar coherent modes,

$$
K_{mn}(\omega)=\int dA\,F_m^*F_n
 [\sigma_e(\omega)N_2-\sigma_a(\omega)N_1],\qquad
\partial_z b_m\big|_{\mathrm{gain}}=\tfrac12\sum_nK_{mn}b_n
$$

in the stationary-population limit. Off-diagonal terms can survive a nonuniform
inversion. Independent scalar gains per mode are not a general multimode model.
Population saturation uses the local coherent sum before taking its modulus; unrelated
pumps and incoherent ASE instead sum as intensities. Degenerate-basis rotations must
leave physical predictions invariant.

For the fully vector general route, neither this scalar $F_m^*F_n$ formula nor local
axial Poynting flux is a sufficient derivation. The latter is not necessarily a positive
instantaneous absorption rate. A later extension must derive a material polarization or
current from local electric fields and project it with the provider's reciprocal flux
metric, consistently with dipole orientation and frequency normalization. Existing
vector step-index modes can inform a weak-guidance reduction, but that reduction must
be named and checked; the exact vector provider is not silently relabelled scalar.

#### 4.4. Time-dependent conventional signal rates

For a coherent pulse whose occupied bandwidth sees nearly constant
$\sigma_{a,e}$ and photon energy, the established local rate is

$$
R_{a,e}(t)=\frac{\sigma_{a,e}(\omega_0)}{\hbar\omega_0}|A(t)|^2,
\qquad
\partial_z A\big|_{\mathrm{gain}}=
 \tfrac12[\sigma_eN_2(t)-\sigma_aN_1(t)]A.
$$

This couples the complete population history to the complex field and includes
intrapulse depletion. It is exact within the narrowband rate assumptions, not a model
for arbitrary broadband GMNA. It supplies the transient baseline, including pump-only
recovery, pulse-pair depletion and the appropriate Frantz–Nodvik limit [7]. With
reabsorption or a populated lower manifold, use the corresponding two-manifold equations
rather than applying a four-level saturation formula unchanged.

For a broadband coherent field there is no unique, positive, infinitely resolved
instantaneous spectral irradiance $I_\omega(t)$. A spectrogram defines one after
choosing a window, and changes temporal and spectral resolution together. A
frequency-to-time mapping from instantaneous chirp is another approximation, which
fails for general multivalued chirp or interfering pulses. Neither follows merely from
writing the population ODE. Conventional broadband transient approaches [8, 9] should
be examined as independent comparators; their optical and temporal closures must be
specified along with their rates.

#### 4.5. The filtered-field transient closure: useful, but separately qualified

Chen and Wise [1, 2] propose replacing a time-frequency power representation by
filtered complex fields. For one transition define the spectral operators

$$
S_j=\mathcal F^{-1}\sigma_j(\omega)\mathcal F,\qquad
Q_j=\mathcal F^{-1}
 \sqrt{\frac{\sigma_j(\omega)}{\hbar\omega}}\mathcal F.
$$

In local scalar normalization their deterministic stimulated terms are

$$
R_j(t)=|Q_j A(t)|^2,\qquad
\partial_z A\big|_j=\frac{s_j}{2}N_{u(j)}(t)S_j A(t),
$$

where $s_j=+1$ for emission and $-1$ for absorption, and $N_{u(j)}$ is the
population from which that transition starts. Apply the cross-section filter **before**
multiplication by the time-dependent population. Those operations do not commute.
This is a concise broadband closure with positive rates, natural interference terms,
and no explicit two-dimensional time-frequency array. It has the correct constant
population and narrowband limits. The paper and its extensive supplement give useful
comparisons, including transient and burst operation. Peer review and these examples
support investigating it; they do not establish universal accuracy or community
agreement. No claim of consensus is needed for this proposal.

There are two specific questions to resolve before making it a production default.

**Stimulated photon balance.** Put
$H=\mathcal F^{-1}(\hbar\omega)\mathcal F$ on the positive active band,
and use $\langle x,y\rangle=\int x^*(t)y(t)dt$. The photon fluence is
$\mathcal N=\langle A,H^{-1}A\rangle$. For a single transition the proposed
field equation implies

$$
\frac{1}{s_j}\partial_z\mathcal N
=\operatorname{Re}\langle H^{-1}A,N_{u(j)}S_jA\rangle,
$$

whereas the population equation counts stimulated events as

$$
\int N_{u(j)}(t)|Q_jA(t)|^2\,dt.
$$

For a constant population these are equal by Parseval. For a general varying
population they are **not an operator identity**. Parseval for the unweighted rates
does not establish equality after multiplication by $N(t)$. This matters because each
stimulated transition should exchange one photon with one ion, under the assumptions
of the reduced model.

A small dimensionless algebraic diagnostic makes the distinction reproducible. Take
two equal-amplitude positive-frequency components, photon energies $(1,1.25)$,
cross-sections $(1,4)$, and a population proportional to
$1+\epsilon\cos[(\omega_2-\omega_1)t]$. Averages over a beat period give
$4.2+2.4\epsilon$ for the field expression and
$4.2+\sqrt{3.2}\epsilon$ for the counted events. They coincide for
$\epsilon=0$, but differ for nonzero $\epsilon$. These are arbitrary scaled
test values, not Yb measurements or a prediction of error in a published experiment.
A physical, self-consistent population trajectory may suppress the difference because
it varies slowly. Qualification must quantify that suppression and the resulting error
in the intended pulse/burst regime, rather than assert exact conservation.

One possible alternative uses photon amplitudes $c=H^{-1/2}A$ and
$T_j=\mathcal F^{-1}\sqrt{\sigma_j}\mathcal F$, with
$\partial_z c=s_jT_jN_{u(j)}T_jc/2$. It balances the stated rate exactly and
has the same stationary limit. It is a **different constitutive closure**, not an
algebraic repair that can be substituted without physical validation; in particular it
does not settle causality.

**Causality and eliminated polarization.** A real nonconstant cross-section filter
usually has a two-sided temporal impulse response. Absorption and emission spectra
alone do not specify the phase of a causal susceptibility, its dephasing time, or an
inhomogeneous distribution of transitions. Placing a changing population outside a
filter is a further adiabatic approximation to a polarization with memory. Test a
finite-line causal polarization/Maxwell–Bloch reference and examine the ratio of the
population variation time to that memory. Any neglect of dopant dispersion needs a
measured or bounded impact; a finite-band Hilbert transform with unknown tails is not
automatically a reliable susceptibility. Do not count passive host dispersion twice.

The recommended decision is to retain both **steady spectral power rates** and the
**candidate filtered-field transient closure** under distinct names. The narrowband
transient model is their common controlled limit. Compare closures at fixed material,
grid and pump conditions; expose the chosen closure and its qualified regime in output.
If the field closure fails the budget or causal tests materially for GMNA, retain it as
a research implementation until a physically justified alternative is accepted. Do not
hide disagreement with clipping or a normalization adjustment.

#### 4.6. Pump transport, ASE and laboratory time

For either propagation direction, define $P^\pm\geq0$ as power flowing in that
direction, never as a signed power. A narrowband pump or ASE bin obeys

$$
\left(\pm\partial_z+v_g^{-1}\partial_{t_{\mathrm{lab}}}\right)P^\pm
=(g-\alpha)P^\pm+S^\pm.
$$

The pump has $S=0$; $\alpha\geq0$ is independent background power loss. In
the signal's retarded coordinates this becomes

$$
\left[\pm\partial_z+
 (v_g^{-1}\mp\beta_{1,\mathrm{ref}})\partial_t\right]P^\pm
=(g-\alpha)P^\pm+S^\pm.
$$

Specify $P^+(0,t_{\mathrm{lab}})$ and $P^-(L,t_{\mathrm{lab}})$ at opposite
ends. Counter-pumping is a boundary-value problem; it cannot be implemented by reversing
a forward nonlinear signal run. Steady operation removes the time derivative. A
quasi-static pump approximation in a transient calculation is permitted only when its
neglected transit-time effect is tested; arbitrary time-dependent boundary pumping needs
characteristic transport and a common laboratory timeline.

For one directional guided spatial/polarization mode, the spontaneous source density
per unit angular frequency is, in the conventional rate model [10],

$$
S_\omega^\pm=
 \frac{\hbar\omega}{2\pi}
 \int dA\,u(\mathbf r,\omega)\sigma_e(\omega)N_2(\mathbf r,t).
$$

Integrate over each ASE bin; multiply by a mode count only when the channel explicitly
aggregates those modes. A factor two for two polarizations is not applied again if both
polarizations are already separate channels. Guided spontaneous emission seeds the ASE
transport; stimulated ASE absorption/emission enters the population rates. The total
lifetime already includes spontaneous depopulation, so adding the guided source does
not justify subtracting it a second time from $N_2$. Partition radiative escape,
guided emission and nonradiative heat in the energy account.

Deterministic ASE powers model mean depletion, including light outside the coherent
pulse window. They do not model shot-to-shot noise or ASE–Kerr interactions in that
window. Stochastic field ASE would require a separately derived normalization, circular
complex statistics, distributed increments proportional to $\sqrt{\Delta z}$,
and consistent increments when a step is split or rejected. Adding input shot noise
alone is not distributed spontaneous emission. Whichever form it takes, stochastic ASE
needs the noise to see only the emission gain, not the net gain. Either the noise is
injected separately and acted on by the emission part of the gain operator alone, as in
the split of Chen and Wise (arXiv:2410.20567), which solprop's modified-noise injection
path (ADR-008) could carry, or a gain Langevin term supplies the distributed
spontaneous emission. [SEP 37](archive/sep-0037-noise-and-stepper-defaults.md) keeps that
injection path until this choice is made. Such stochastic work is deferred here; there
is no claim of reproducing all stochastic figures in [1, 2].

#### 4.7. Pulse windows and the population timeline

Resolve coherent pulses on their existing fine grids; propagate the slow populations,
pump and incoherent ASE through the gaps on a separate time description. Gap boundaries
come from the user's pulse arrival times or period. Use exact affine recovery where
rates are constant, and an error-controlled ODE/transport solve otherwise. Populations
before and after a pulse belong to a physical position $z$ and a laboratory time;
they are not recycled as the initial population at a different spatial position.

An isolated pulse needs a specified initial inversion profile, possibly prepared by a
preceding pump-only calculation. A burst updates that profile after each event. A
periodic problem closes the entire pump/population history over its declared period;
its fixed-point residual is distinct from the field's propagation error. Independent
scan members have independent reservoirs; successive pulses of a burst do not.

The scaled Fourier transform in [1, 2] is not required for this design. A change of
variables by itself saves no degrees of freedom; savings require justified coarsening
or removal of a rapidly varying phase. The supplement discusses restrictions involving
rapid gain modulation and frequency mixing. Coarsening such a representation in the
presence of Kerr, Raman and four-wave mixing needs its own error analysis. Short pulse
windows plus slow gaps provide a simpler first strategy without altering solprop's
Fourier convention.

#### 4.8. Material data and the optmat boundary

Optmat should supply an immutable **doped-host spectroscopy record**, not a universal
“Yb cross-section” array. At minimum it identifies the host composition, dopant and
concentration convention, temperature, manifold labels, absorption and emission
transitions, wavelength/frequency range, lifetime and its definition, source and dataset
version, uncertainties, and every processing step. Additional transitions require their
own radiative branching, nonradiative and transfer data. Pump geometry, active-ion
density inferred for a particular fibre, launch efficiency and propagation length remain
device parameters in solprop, with provenance.

The present passive refractive-index interface is not a container for an inverted gain
medium. Its passive extinction constraints and lack of population state are intentional.
Keep host dispersion and nonlinear response separate from active spectroscopy, and
evaluate the population-dependent gain in solprop. No optmat object or source lookup
crosses the numerical seam; translation samples and validates the selected immutable
data before execution.

Follow optmat's [curated-data contract](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/data.md):
register the doped host identity, define the new transition/emission/lifetime quantities
and required metadata in its vocabulary, then add source records, reproducible
extraction scripts, raw values, evidence-backed errata and review-register entries.
Its current generic photoabsorption quantity is not already a complete gain-transition
schema; nor does a gas-phase ion identity identify an ion embedded in silica. Preserve
printed values and mark reproduced data rather than fitting both a reproduction and
its original. Validate extraction against the library and apply optmat's own full gate
when those changes are made. Fitted or digitized spectra need an uncertainty and
processing record distinct from a manufacturer's original table.

The acquisition/curation dossier is:

| Needed information | Primary source or resource | Work still required before use as built-in data |
|---|---|---|
| Yb absorption/emission spectra, host and manifold context | Pask et al. [11]; Paschotta et al. [12]; Coherent/Nufern manufacturer curve [13] | Obtain original numeric curves or document digitization; establish which host/fibre each represents, temperature, absolute scale and uncertainty |
| The spectra underlying a particular GMNA comparison | Methods and supplements of [1–6], plus the originating manufacturer | Obtain the exact dated manufacturer dataset; an unnamed “typical Yb” curve is insufficient for a quantitative reproduction |
| Upper-state lifetime and quenched fractions | Paschotta et al. [14], plus measurements for the chosen doped host | Distinguish total and radiative lifetime, concentration dependence and any unbleachable component; do not assume one commonly used lifetime is universal |
| Absorption/emission consistency | McCumber [15] | Thermal-equilibrium applicability, temperature and the partition/zero-line information needed for the McCumber energy; use as a check, not an invented replacement spectrum |
| Doped/core/cladding dimensions, NA, pump absorption and passive loss | Original fibre datasheet, e.g. the Thorlabs Yb family [16], and the exact experiment's methods | Distinguish core and cladding absorption, test wavelength, concentration, bend state and launch assumptions; archive the actual product/revision |
| Host index, Kerr and Raman response | SEP 3's silica sources and a composition-specific source where doping matters | Quantify whether passive fused silica is adequate for the selected fibre rather than silently treating every doped host as pure silica |
| Causal transient reference | A measured line/inhomogeneous model with dephasing information | Not supplied by cross-sections alone; remains a data/research task for the field-closure gate |
| Later Er/Tm/Ho/Nd kinetics | Host-specific primary spectroscopy and transfer measurements | Separate upstream records and qualification; coefficients measured in fluoride glass do not become silica coefficients |

In particular, the provenance of the dated Liekki Yb table used in some published-model
workflows still needs its originating manufacturer resource. No primary public numeric
record for that table was established in this review. The public Coherent curve is a
useful lead, not evidence that the two datasets are interchangeable. Acquire these
resources before promising a numerical reproduction; the present SEP does not install
data copied from a propagation code.

Cross-sections are functions evaluated at corresponding wavelengths and frequencies:
$\sigma(\omega)=\sigma(2\pi c/\omega)$. There is **no spectral-density
Jacobian** for a cross-section value; converting a power density does require one.
Archive raw data, then record unit conversion, smoothing, interpolation and any
extrapolation separately. Shape-preserving interpolation may be appropriate, but
arbitrary exponential wings or suppression beyond a table are model choices with
particular impact on GMNA. Refuse an unsupported active band or require an explicit,
qualified extension. Do not silently clip a negative fitted cross-section into validity.

For ground-state, weak-probe, unsaturated pump absorption,
$\alpha_{\mathrm{ion}}=\Gamma_pN_d\sigma_a(\omega_p)$ and a measured
power attenuation $a_{\mathrm{dB}}$ converts to
$\alpha=(\ln10/10)a_{\mathrm{dB}}$. Inferring $N_d$ this way requires
subtracting background loss and declaring the pump overlap. Uncertain overlap and
cross-section scale make the inferred density uncertain; it is not an independent
spectroscopic measurement. Preserve that covariance when comparing models.

Under thermal equilibrium, the McCumber relation has the form
$\sigma_e/\sigma_a=\exp[(E_{\mathrm{MC}}-\hbar\omega)/(k_BT)]$ [15].
Neither $E_{\mathrm{MC}}$ nor a temperature is supplied by a generic ion name.
Constants come from the project's established SI constant source, not rounded values
transcribed from a separate implementation.

### 5. Proposed interface

Introduce an amplifier specification around an ordinary signal simulation, with gain
physics distinct from temporal boundary conditions and numerical execution. Proposed
names below are design sketches; they are not existing APIs or a mandate to add a
constructor for every line.

```text
material = optmat gain-spectroscopy record, selected by immutable identity
signal = existing GNLSE or supported scalar modal Simulation
doping = DopedRegion(material, density=..., radius=...)
pumps = (PumpChannel(wavelength=..., power=..., direction="forward",
                     profile=UniformCladding(radius=...)),)

amplifier = Amplifier(signal, doping=doping, pumps=pumps,
                      ase=SpectralASE(edges=..., modes=...),
                      population_space="averaged",
                      signal_rates="spectral_average")
steady = amplifier.solve(PeriodicTrain(period=...), population="steady")

transient = amplifier.with_signal_rates("narrowband").solve(
    PulseSequence(pulses=..., arrival_times=...),
    population=InitialInversion(profile=...),
)

candidate = amplifier.with_signal_rates("filtered_field").solve(
    PeriodicTrain(period=...), population="periodic",
)
```

All lengths are metres, wavelength metres, density $\mathrm{m^{-3}}$, power watts,
arrival times/period seconds, and inversion a fraction in $[0,1]$ for Yb. Doped and
pump profiles have explicit normalization; ASE bins declare whether edges use angular
frequency or vacuum wavelength and how mode counts are represented. A zero launched
power is permitted; lengths, period and bin widths are positive. Coincident coherent
pulses must be combined on one field window rather than processed as independent
events. Separate pulse events require nonoverlapping resolved windows and a defined gap.

The user chooses the population boundary condition: steady, periodic, or an initial
state with a laboratory time. No pump-only steady state is silently assumed for an
arbitrary burst. A pump-preparation operation may construct that initial state. Pump
transit is explicit (`steady`, qualified `quasistatic`, or time-dependent transport),
and temporal/spectral tolerances are distinct from the longitudinal controller.

An amplifier result contains the ordinary saved signal plus pump/ASE spectra and
direction, inversion versus position and requested time, population boundary states,
absorbed pump, extracted signal, photon/energy-budget residuals, population and outer
iteration residuals, step history, and all material/model identities. Full population
trajectories are opt-in because they can dominate output size. Checkpoints contain the
reservoir and boundary-solver state necessary to resume, not only a signal spectrum.

Refuse unsupported combinations before allocating a run: general vector/moving guides,
coherent backward signals, unsupported ions or bands, a steady closure for a request
that explicitly requires intrapulse dynamics, and unqualified backend/model pairs.
Transient scan/batch support must keep state isolated and is enabled only after its
parity tests. `backend="auto"` does not change the selected physical closure. No
hidden fallback from filtered-field to steady gain, or from a missing backend feature
to a different solver, is allowed.

### 6. Implementation strategies

#### 6.1. Recommended separation

Recommend `solprop/gain.py` for level/transition definitions and validated amplifier
specifications, `driver/amplifier.py` for pump/population boundary conditions and pulse
sequences, and `translate/gain.py` for typed, sampled coefficient tables and profiles.
These additions to the package tree are agreed at acceptance and documented in
[architecture §4](../architecture.md). Avoid putting an amplifier controller inside
`nonlinear.responses`, whose existing responses do not own a long-lived pump reservoir.

The ordinary signal propagation remains the numerical core. The amplifier driver
coordinates it with pump and ASE solves, and with the timeline. For a steady problem,
iterate signal spectra, populations and both directions of incoherent transport until
their joint residual and relevant observables converge. A prescribed gain table can be
used inside a pass, but a single pass is not the coupled solution. Use continuation or
damping where necessary, reporting failure if the residual does not contract; a fixed
number of passes is not a convergence criterion.

For transient gain, evaluate the causal population history using the **trial-stage
field and pump** at each longitudinal stage, or use a coupled method with independently
demonstrated order. The population ODE is in time, not an extra derivative in $z$.
Recomputing its history at a stage does not advance the accepted physical reservoir
multiple times. Store accepted state separately from candidate state, with explicit
ownership of initial population profiles at each physical position.

For nonperiodic pumping or counterpropagating time-dependent ASE, use a space-time
transport/boundary iteration on characteristics, coupled to the pulse events. This is
more work than a unidirectional field march and must be budgeted as such. Admit memory
for required longitudinal checkpoints, or deliberately recompute passes; do not pretend
that an arbitrary transient boundary-value problem is local to one FFT window.

#### 6.2. Numerical choices and the contract

Prefer an exact affine two-manifold update over an unconstrained nonlinear optimizer.
For genuinely multilevel stiff systems, an independent SciPy implicit ODE solver is a
reference; a production solver would need an explicit acceptance decision and backend
implementation. Do not make a general stiff solver a hidden requirement of the initial
Yb milestone. Positivity and ion conservation are mathematical properties to preserve
or test. Large negative populations trigger refinement/failure; clipping and
renormalization must not conceal a failed solve.

Freezing populations at the start of an RK4 signal step, updating the pump with Euler,
or using a first-order gain increment under a square root can reduce the order of the
**coupled** solution regardless of the field stepper's name. Choose a demonstrated
coupled method, or a declared second-order symmetric split with its own controller;
do not inherit a fourth-order claim from the passive solver. The default recommendation
is stage-consistent deterministic coupling, using step doubling as an initial independent
error estimate until an embedded estimate has been qualified.

The error budget covers the field, pump/ASE powers, population fractions, temporal
population integration, transverse/frequency quadrature and the outer boundary solve.
Use absolute tolerances near zero populations and dark channels, and track gain
sensitivity near transparency. Refining $\Delta z$ cannot repair a fixed temporal
population error. Adaptive rejection restores **all** candidate state and cached stage
data. Replay carries sufficient information to reproduce the coupled accepted path.

Extend `KernelProblem` with immutable gain tables and flags and extend the stage/driver
contract with typed coupled state, observations and transaction boundaries. Preserve
the existing field array layout; do not disguise pump power or population fractions as
extra coherent modes. Bump `SCHEMA_VERSION` at the implementing commit, update the
literal seam tests and the changelog, and coordinate the Kokkos admission check. Do not
reserve a schema number in this draft.

For ordinary modal amplitudes $b=W^{1/2}a$, where
$W=\operatorname{Re}n_{\mathrm{eff}}$ on the relevant route, the gain increment
at fixed $z$ transforms as

$$
\partial_z a\big|_{\mathrm{gain}}
=W^{-1/2}\,\mathcal G_b[W^{1/2}a].
$$

The existing normalization transport handles any $z$ dependence; it is not added
again. GNLSE and general-modal conventions differ and need their own adapters rather
than a blanket refractive-index factor. Projection, rates and output budgets are tested
in physical normalization before backend optimization.

#### 6.3. Backends and dependencies

Implement the NumPy reference first, with small independent ODE/transport references.
Numba uses shared kernel bodies, preallocated buffers and deterministic reductions.
For Yb, a serial temporal recurrence per spatial column is simple and predictable;
parallelism is across population nodes, independent samples and channels, not assumed
across causally dependent time points. A parallel affine scan is a later measured
optimization, with reproducibility tests.

Kokkos CPU must carry the same state and stage semantics natively, keep arrays resident
between stages, and expose only host observations through the seam. Its existing
[backend contract](https://github.com/LupoLab/solprop-kokkos/blob/main/README.md)
does not already provide this feature. Porting only the cross-section multiplication
does not complete parity. Single-mode workloads may not supply enough independent
columns for a GPU to offset launch and transfer costs; the CPU path remains a primary
target. GPU testing follows on a device and does not gate initial CPU qualification.

No new numerical dependency is proposed beyond the approved NumPy/SciPy stack and
SEP 3's optmat dependency. Do not introduce a MATLAB runtime, a new FFT convention, or
a second material database. Any new stiff-solver or accelerator dependency needs its
own agreement.

#### 6.4. Expected cost and memory

These are structural estimates, not measurements or forecasts of complete-run time.
Let $N$ be the fine temporal/spectral grid length, $M$ the signal-mode count, $Q$
the population-node count, $L_v$ the number of levels, $C$ the number of stimulated
transitions, $B$ the number of incoherent frequency channels, $K_z$ the longitudinal
boundary-grid size, and $J$ the number of outer passes.

| Work | Leading cost or storage | Practical consequence |
|---|---|---|
| Steady rates and gain | Spectral contractions plus $O(Q L_v)$ populations; local-field synthesis about $O(QMN)$ | The simplest single-mode case should add modest work per pass; coupled pump/ASE passes can dominate total cost |
| Filtered-field rates and gain | About $O(CQN\log N+QL_vN)$ with local fields | For two Yb transitions, a direct implementation needs roughly two inverse filters for rates, two for gain, and one forward transform of the summed gain term per local column/evaluation, if the input spectrum is reused |
| Stage-consistent transient coupling | Those evaluations at every required stage, plus temporal refinement | More expensive than once-per-step frozen populations; benchmarking must compare matched accuracy |
| Full population history | $8QL_vN$ bytes for float64, or fewer with a dependent ground level | At $N=32768$, $Q=32$, $L_v=2$, two population arrays use 16 MiB; storing just $f$ uses 8 MiB |
| One complex scalar field | $16N$ bytes | At the same $N$, 0.5 MiB; stage and FFT buffers multiply this |
| Steady pump/ASE boundary arrays | About $O(K_z(B+QL_v))$ values | Add directions, modes and saved histories explicitly in the memory ledger |
| Arbitrary transient boundary transport | Additional slow-time/channel storage and/or repeated passes | Longitudinal coherent-field checkpoints may dominate; choose storage versus recomputation explicitly |

The transform count is an explanatory upper-level decomposition, not a required kernel
layout. Shared profiles and transforms can reduce it; frequency-dependent multimode
synthesis can increase it. Avoid storing a full $Q M^2 N$ coherence tensor when
synthesizing the local field gives the needed contraction directly. Do not promise
linear scaling in $M$ for every general modal projection.

A fine grid spanning a $1\,\mathrm{\mu s}$ period at $10\,\mathrm{fs}$
spacing contains $10^8$ samples: one complex128 array is 1.6 GB before stages or
populations. The separate timeline removes the need for that allocation. A dense
$32768\times32768$ float64 time-frequency array alone is 8 GiB; avoiding it is a
clear memory advantage of a filtered-field closure. It does not establish an advertised
speedup over a well-designed conventional solver. In particular, neither a scaled
transform nor an optimizer iteration count can be translated into a universal speedup.

Measure translation, warmed rate kernels, a coupled trial, about ten accepted steps,
and a short complete boundary solve separately. Record $N,M,Q,B$, stages, rejected
steps, inner/outer tolerances, $J$, resident memory and backend/thread settings in
[benchmarks.md](../benchmarks.md). Compare at matched population, pump and signal error.
Never extrapolate a ten-step timing to a metre-scale amplifier.

### 7. Verification and validation

#### Analytic and independent tests

1. **Rates and units:** constant-rate Yb recovery, a pump-only steady inversion, ground
   and excited limits, zero density, zero cross-section, zero pump and transparent
   populations. Compare with the exact affine solution; tolerances follow machine
   precision and composition length, not arbitrary decimal agreement.
2. **Linear transport:** prescribed constant inversion gives field amplitude
   $e^{(g-\alpha)z/2}$ and power $e^{(g-\alpha)z}$; check both pump directions,
   Beer attenuation, a distributed constant ASE source, polarization multiplicities,
   and laboratory-to-retarded-time characteristics.
3. **Saturation:** the narrowband, no-pump/no-decay four-level limit reproduces
   Frantz–Nodvik [7]. Independently integrate the two-manifold system with
   reabsorption; compare pulse pairs and recovery. Do not use the same recurrence as
   both solver and purported independent reference.
4. **Steady/periodic consistency:** compare direct steady equations with time-resolved
   periodic solutions as extraction per pulse tends to zero at fixed average power.
   Verify the affine periodic map and its near-identity limit. Compare a finite burst
   with sequential single-pulse solves sharing the same reservoir and pump history.
5. **Photon and energy accounts:** isolate every transition and compare ion events with
   pump, signal and ASE photon exchange, using frequency-resolved photon energy. The
   full optical energy balance also includes stored excitation, radiative escape,
   background loss and quantum-defect/nonradiative heat; it is not optical energy
   conservation in a pumped amplifier. Budget tolerances include quadrature and
   integration errors. Material uncertainty is reported separately.
6. **Field-closure decision:** reproduce the §4.5 algebraic diagnostic, then use
   self-consistent narrowband, two-colour and strongly chirped/unchirped pulses with a
   finite-line causal reference. Sweep population variation versus response memory,
   bandwidth, inversion and filtering/padding. Measure stimulated-budget residual,
   spectral phase and compressed-pulse errors. A residual that survives numerical
   refinement is model error, not a reason to loosen a numerical tolerance.
7. **Spatial consistency:** averaged and resolved populations agree for uniform
   intensity/inversion; converge radial shells and mode-profile interpolation; compare
   one-mode/multimode and coherent/incoherent limits. Check degenerate-basis invariance,
   shared-reservoir competition and gain matrices with off-diagonal terms.
8. **Coupled convergence and rollback:** three-level refinement in $\Delta z$ with
   temporal/spatial errors made negligible, then separate temporal, ASE-bin,
   transverse and boundary-grid refinements. Assert the declared order. Force rejected
   steps and check pump, population, ASE and signal against a direct accepted path.
   Tighten outer residuals independently of local error estimates.
9. **Representation/backend parity:** field and envelope grids in their common regime,
   physical amplitude conversions, one gain evaluation, one trial and eight accepted
   steps, statistics, a short complete amplifier and checkpoint/resume. NumPy is the
   numerical oracle; it is not the independent physics reference. Numba and Kokkos CPU
   also pass determinism, thread-count, memory-ledger and state-isolation tests.

#### Published physics targets

Use the conventional studies [5, 6], the original GMNA experiment [3], Chen and Wise's
transient comparisons [1, 2], and Wang et al.'s GMNA/GMPPA comparison [4]. Reconstruct
each target from a complete parameter sheet: exact cross-sections and host, doping or
measured pump absorption with overlap assumptions, pump direction and coupled power,
passive dispersion and loss, input field including spectral phase, repetition rate,
geometry, ASE treatment and compressor. Missing inputs are unresolved uncertainty,
not fit parameters adjusted until a plot looks right.

For GMNA, compare inversion and residual pump versus $z$, evolution of gain and
reabsorption across the spectrum, signal energy, red/blue spectral extent and the
compressed temporal profile. For Wang's PPA study, include parabolic misfit,
dechirped duration, peak power, Strehl ratio and sensitivity to seed duration/chirp.
Its experimental result and larger-area scaling prediction are separate targets; a
predicted peak power is not an observed datum. Use the paper's actual compressor model
when comparing compressed pulses. Do not treat transform-limited duration as measured
compressibility. Define spectral-edge thresholds and fitting windows explicitly.

For power $P=|b|^2$ and its fitted parabolic profile $P_{\mathrm{fit}}$, the
misfit used in [4] is

$$
M=\left[\frac{\int(P-P_{\mathrm{fit}})^2dt}{\int P^2dt}\right]^{1/2}.
$$

Strehl is the compressed peak power divided by the transform-limited peak power for
the same spectrum, with compressor losses and energy normalization stated. Compare
curves and phases, not just a single best-fit number. Experimental tolerance includes
reported uncertainty, digitization and incomplete material/device knowledge. A
parameter sensitivity study separates those limits from discretization error.

Routine tests and timings use short fibres or approximately ten accepted steps. Full
GMNA evolution and publication-size optimizations belong only in manually executed
[worked examples](../testing.md), with committed outputs and measured run time. No
full-length propagation is added to a test, coverage lane or benchmark. The target
examples are not prerequisites for every documentation build.

### 8. Documentation

Add a gain model page deriving the rate, spatial, timing and ASE conventions; an
amplifier user-guide chapter covering data selection, pump preparation, steady versus
transient operation, diagnostics and refusal conditions; and API docstrings for every
new public object. Extend the solid-fibre guide and GNLSE page with the supported
composition and its scalar/vector limitations. Update the architecture, stepper,
kernel-contract, output and benchmark documents with their implementation changes.

A quick tutorial should show weak-probe gain and population recovery on a short case.
Worked examples should cover steady GMNA, GMNA versus GMPPA with compression, and
deterministic burst depletion/recovery. Add a filtered-field comparison only once the
closure is qualified, stating the applicable regime and the omitted stochastic physics.
The optmat documentation owns data provenance and uncertainty; solprop links it rather
than maintaining duplicate tables. README, usage and changelog entries explain the
new capability when it actually lands, not when this draft is written.

### 9. Compatibility and impact

Existing passive simulations retain their physical defaults. Gain is explicit and
unsupported combinations fail by name. The new state and stage fields require a
coordinated seam revision in solprop and solprop-kokkos. The result/checkpoint format
needs a corresponding versioned extension for material identities, population and
boundary-solver state, ASE representation, chosen closure and approximation flags.
Resume must reject mismatched materials, temporal boundary conditions or closures;
fields-only passive checkpoints do not initialize a gain reservoir.

This is a **large feature across three repositories**. Optmat needs a new class of
curated data, solprop needs an amplifier orchestration layer and extended transactional
state, and Kokkos needs native coupled kernels and admission logic. Adding the scalar
steady Yb formula is small; providing a reliable transient amplifier is not. The main
scientific risk is the broadband transient closure, while the main engineering risks
are coupled adaptivity, counterpropagating boundary conditions and memory management.

As a planning estimate, not a measured schedule, allow roughly **four to eight engineer
months** for curated Yb data, the reference solver, conventional transient operation,
documentation, qualification and both optimized CPU backends. This assumes familiarity
with the three projects, access to the necessary material data, and a fixed scope; it
includes substantial testing work. Scalar spatial/multimode generalization adds work
within that range only if existing projection machinery is reusable. General vector
gain, new ions and stochastic ASE are additional projects. Resolving a failed
field-closure qualification is open-ended research and is not included in that estimate.
Use the independently mergeable milestones below to obtain real cost evidence early.

No claim of performance parity with external implementations is made without running
matched-accuracy short benchmarks. No new universal material coefficients are adopted
by this document. Deliberate behavioural differences from Luna.jl are recorded on
[the differences page](../../guide/workflows/luna.md) when implemented, based on verified comparisons.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Prescribed Gaussian gain or $g_0/(1+E/E_{\mathrm{sat}})$ | Useful controlled baseline; cannot predict pump depletion, shifting reabsorption, burst recovery or a material-dependent GMNA gain spectrum |
| Conventional steady spectral rate equations only | Recommended first production milestone and often enough for high-repetition-rate GMNA; explicitly insufficient for arbitrary pulse/burst extraction |
| One nominal saturation area for all physics | Rejected: pump overlap, doped-area averaging and Kerr effective area are distinct |
| Time-frequency powers for all broadband transients | Useful independent comparator; window dependence and potentially large storage/cost must be quantified; not selected as the universal production representation |
| Instantaneous-frequency gain from pulse chirp | Restricted approximation, unsuitable for arbitrary phase, interference or nonmonotonic chirp |
| Filtered-field closure as the only model | Premature: retain its practical advantages but resolve the photon-balance and causal-response questions and preserve the conventional steady reference |
| Full optical Maxwell–Bloch model everywhere | Valuable independent limit/reference; generally requires spectroscopy absent from power cross-sections and much greater state/parameter complexity |
| All ions, vector modes and stochastic noise at once | Makes the first useful amplifier depend on unresolved spectroscopy and several independent numerical schemes; stage these capabilities |
| Put active gain into the passive index provider | Conflates a population-dependent reservoir with a passive material property and does not solve pump/population coupling |
| Treat populations as extra coherent modes | Wrong state semantics, units, error norm and normalization; use a typed coupled state |
| Fine FFT grid across an entire repetition period | Unnecessary memory and runtime cost; use pulse windows and an explicit population timeline |

### 11. Phases

Each implementation phase uses its own `feature/sep-0007-…` branch, appends a report
to Part II, and passes the applicable full gate, slow lane and coverage lane. A backend
without a delivered capability refuses it by name. The phases are ordered to obtain a
useful conventional amplifier before committing to the less established closure.
Before acceptance, settle Q1–Q4: in particular, agree the field-closure qualification
criteria and which outcomes fit the accepted scope. Phase 1 then applies those criteria;
an outcome requiring a different physical model returns the proposal for revision.

1. **Data and closure qualification.** Curate one identified Yb/silica spectroscopy
   bundle in optmat, including uncertainty and processing provenance. Complete the
   missing source acquisitions in §4.8. Derive normalization adapters and an independent
   two-manifold reference. Run the algebraic and causal-model comparisons for the
   filtered-field candidate and record the decision against the agreed criteria. Do
   not label synthetic test coefficients as measured Yb data.
2. **NumPy steady amplifier.** Deliver mode-averaged Yb gain, spectral signal rates,
   co-/counter-pump and deterministic bidirectional ASE boundary solves, spectra and
   budget output, and the manual page. Verify exact limits and independent steady
   references; add a full GMNA worked example only after its parameter sheet is
   complete. This is a named steady capability, not completion of the SEP.
3. **NumPy deterministic transient amplifier.** Deliver initial/periodic population
   boundary conditions, pulse windows and dark gaps, stage-consistent narrowband
   coupling, pump transit or an explicitly bounded approximation, transactional
   rejection and resume. Verify recovery, pulse pairs, finite bursts and the steady
   limit. Introduce the qualified broadband closure from phase 1 with its distinct
   model name, limits and comparison example. If it was not qualified, revise the
   accepted scope explicitly; do not mark broadband transient work delivered.
4. **Numba parity.** Implement the delivered steady and transient scope with shared
   bodies, deterministic reductions, memory accounting and CPU benchmarks. Verify
   every stage, short-run parity, error control, rollback, resume and the coverage lane.
5. **Kokkos CPU parity.** Deliver the same scope natively under the new contract with
   host observations, state isolation and matched-accuracy benchmarks. No Python
   per-stage fallback or host/device transfer loop stands in for native support.
6. **Resolved scalar populations and modes.** Add radial shells and supported fixed
   scalar modal sets, with coherent mode competition and frequency-dependent overlap
   where qualified. Supply the solid-material projection/admission missing from the
   present scalar gas route, reusing general-route quadrature where its normalization
   permits the agreed weak-guidance reduction. Implement NumPy, then Numba, then Kokkos
   CPU on separate phase branches/reports. Verify spatial convergence, one-mode reduction and basis
   invariance; document the remaining general-vector restriction.

`Final` requires the accepted scope on all three CPU backends, manual links to this
SEP, a changelog citation and explicit destinations for outstanding questions. General
vector gain, other ions and stochastic field ASE remain roadmap items, not exceptions
silently counted as completed phases. GPU qualification is separate as in the repository
qualification policy.

### 12. Open questions

1. **Before acceptance:** what error and regime bounds would qualify the filtered-field
   closure after the stimulated-budget and causal-reference comparisons? If they fail,
   should broadband transient gain remain research scope or use another explicitly
   derived closure? This is a physics decision for the maintainers, not an
   implementation detail.
2. **Before acceptance:** which host-specific Yb dataset and fibre establish the first
   reproducible GMNA benchmark, and can the underlying numeric cross-sections and
   lifetime information be obtained with suitable redistribution terms? Quantify what
   remains uncertain if only manufacturer plots are available.
3. **Before acceptance:** agree the supported scalar route and weak-guidance reduction
   for step-index fibres, and the error evidence required before admitting a broad
   spectrum with a frozen profile. General vector projection is intentionally deferred.
4. **Before acceptance:** agree the public amplifier/timeline interface and the initial
   pump-transit scope. Arbitrary transient counter-pumping needs the space-time solver;
   accepting only a qualified quasi-static approximation must be reflected in scope.
5. **Phase 2:** choose ASE frequency bins and directional/polarization aggregation using
   convergence and budget tests, and establish an observable bound for omitting ASE.
6. **Phase 3:** choose the coupled integration order and temporal error allocation after
   short prototypes. Establish complete rollback/replay and checkpoint semantics before
   optimizing the kernels.
7. **Phases 3–5:** determine the storage/recomputation trade-off for population and
   longitudinal boundary histories under memory admission; measure how many coupled
   passes typical short benchmark cases need.
8. **Phase 6 / later proposal:** quantify when resolved scalar populations cease to be
   adequate and a vector dipole/current model or a changed passive mode basis is needed.
9. **Later proposals:** curate additional ions, temperature dependence and quenching,
   and derive stochastic ASE with adaptive-step-consistent noise before using it for
   shot-to-shot predictions, noise seeding or oscillators.

### 13. References

#### Primary literature and data resources

1. Y.-H. Chen and F. Wise, “Field-based treatment of transient gain in short-pulse
   optical amplifiers,” *Optica* **12**, 879 (2025).
   [doi:10.1364/OPTICA.557373](https://doi.org/10.1364/OPTICA.557373).
2. Y.-H. Chen and F. Wise, supplementary document for [1], including rate-equation
   derivations, scaled transforms, numerical methods and comparisons (2025).
   [doi:10.6084/m9.figshare.28912205](https://doi.org/10.6084/m9.figshare.28912205).
3. P. Sidorenko, W. Fu and F. Wise, “Nonlinear ultrafast fiber amplifiers beyond the
   gain-narrowing limit,” *Optica* **6**, 1328–1333 (2019).
   [doi:10.1364/OPTICA.6.001328](https://doi.org/10.1364/OPTICA.6.001328).
4. W. Wang, Y.-H. Chen and F. Wise, “Partial parabolic amplification in rare-earth-doped
   optical fiber,” *Photonics Research* **14**, 3415 (2026).
   [doi:10.1364/PRJ.596759](https://doi.org/10.1364/PRJ.596759).
   The [author preprint, including supplemental material](https://arxiv.org/html/2603.27554v2)
   supplies additional optimization and comparison details;
   [doi:10.48550/arXiv.2603.27554](https://doi.org/10.48550/arXiv.2603.27554).
5. R. Lindberg, P. Zeil, M. Malmström, F. Laurell and V. Pasiskevicius, “Accurate
   modeling of high-repetition rate ultrashort pulse amplification in optical fibers,”
   *Scientific Reports* **6**, 34742 (2016).
   [doi:10.1038/srep34742](https://doi.org/10.1038/srep34742).
6. Y.-H. Chen, H. Haig, Y. Wu, Z. Ziegler and F. Wise, “Accurate modeling of ultrafast
   nonlinear pulse propagation in multimode gain fiber,” *JOSA B* **40**, 2633–2642
   (2023). [doi:10.1364/JOSAB.500586](https://doi.org/10.1364/JOSAB.500586).
   [Author preprint record](https://preprints.opticaopen.org/articles/preprint/Accurate_modeling_of_ultrafast_nonlinear_pulse_propagation_in_multimode_gain_fiber/23674605).
7. L. M. Frantz and J. S. Nodvik, “Theory of pulse propagation in a laser amplifier,”
   *Journal of Applied Physics* **34**, 2346–2349 (1963).
   [doi:10.1063/1.1702744](https://doi.org/10.1063/1.1702744).
8. R. Paschotta, “Modeling of ultrashort pulse amplification with gain saturation,”
   *Optics Express* **25**, 19112–19116 (2017).
   [doi:10.1364/OE.25.019112](https://doi.org/10.1364/OE.25.019112).
9. O. Vanvincq and G. Bouwmans, “Numerical simulation of the non-linear propagation of
   ultrashort pulses in fiber amplifiers using time-frequency representations,”
   *Journal of Lightwave Technology* **41**, 314–319 (2023).
   [doi:10.1109/JLT.2022.3212094](https://doi.org/10.1109/JLT.2022.3212094).
10. C. R. Giles and E. Desurvire, “Modeling erbium-doped fiber amplifiers,”
    *Journal of Lightwave Technology* **9**, 271–283 (1991).
    [doi:10.1109/50.65886](https://doi.org/10.1109/50.65886).
11. H. M. Pask et al., “Ytterbium-doped silica fiber lasers: versatile sources for the
    1–1.2 μm region,” *IEEE Journal of Selected Topics in Quantum Electronics*
    **1**, 2–13 (1995). [doi:10.1109/2944.468377](https://doi.org/10.1109/2944.468377).
12. R. Paschotta, J. Nilsson, A. C. Tropper and D. C. Hanna, “Ytterbium-doped fibre
    amplifiers,” *IEEE Journal of Quantum Electronics* **33**, 1049–1056 (1997).
    [doi:10.1109/3.594865](https://doi.org/10.1109/3.594865).
13. Coherent/Nufern, manufacturer
    [Yb absorption/emission curve](https://www.coherent.com/resources/application-note/components-and-accessories/specialty-optical-fibers/yb-absorption-emission.pdf).
    Public graphical resource; numeric-table provenance, host and measurement
    conditions require clarification before curation.
14. R. Paschotta, J. Nilsson, P. R. Barber, J. E. Caplen, A. C. Tropper and D. C. Hanna,
    “Lifetime quenching in Yb doped fibres,” *Optics Communications* **136**, 375–378
    (1997). [doi:10.1016/S0030-4018(96)00720-1](https://doi.org/10.1016/S0030-4018(96)00720-1).
15. D. E. McCumber, “Einstein Relations Connecting Broadband Emission and Absorption
    Spectra,” *Physical Review* **136**, A954–A957 (1964).
    [doi:10.1103/PhysRev.136.A954](https://doi.org/10.1103/PhysRev.136.A954).
16. Thorlabs, [archived Yb-fibre catalogue data](https://www.thorlabs.com/catalogpages/1068.pdf)
    and [example dated product datasheet](https://www.thorlabs.com/catalogpages/obsolete/2018/YB1200-6_125DC-PM.pdf).
    These are device-data leads, not a selection of that product for the first benchmark.

#### Source verification and remaining acquisitions

The full main texts of [1] and [4] and the relevant derivation, applicability, numerical
and validation sections of the substantial supplement [2] were examined. The original
PDF equations for [1]'s field/rate filters and [4]'s parabolic misfit were checked
visually against the extracted text. The public preprint of [4] confirms the existence
and content of additional supplementary sections; its typography and numerical
parameters should be checked against the journal supplement before reproduction.
The primary record/abstract of [6] confirms its steady population scope; its complete
journal text was not independently retrieved in this review. Primary records or
accessible text were located for [5], [8], and [10–16]; no cross-section ordinates,
lifetime or transfer coefficient was adopted from them here. Full acquisition and
equation/data checking of [3], [5–12], [14] and [15] remains part of phase 1 where
needed for its selected benchmarks and material bundle. Reference [9] is an acquisition
lead identified through the literature, not a solver independently reproduced here.

The manufacturer curve [13] is available, but the numeric datasets underlying the
chosen experiments, a host-matched lifetime and the causal transient spectroscopy are
not yet a curated bundle. These are explicit deliverables for optmat, not work that
SEP 3 already completed. No physical propagation benchmark or independent experimental
reproduction has been run in preparing this proposal. The two-tone diagnostic in §4.5
is an algebraic consistency check, not experimental evidence.

## Part II — Implementation record

## Part III — History

- 2026-09-27: Drafted with separate conventional and filtered-field gain models,
  a Yb/silica data dossier, coupled-state and timing requirements, and phased CPU
  qualification. Broadband transient closure and exact material selection remain
  explicit acceptance questions.
- 2026-10-26: Added a note to §4.6 that stochastic ASE needs the noise to see only the
  emission gain, through a separate injection path or a gain Langevin term (SEP 37).
