# SEP 5 — Solid ionisation and bulk filamentation

| | |
|---|---|
| **SEP** | 5 |
| **Title** | Solid ionisation and bulk filamentation |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; Opus 5.5 (2026-09-29 revision for SEP 3's D15); GPT-6 (2026-10-22 coordination with SEP 31; 2026-10-23 split to SEP 34) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md): material adapters and resolved material identities; [SEP 34](archive/sep-0034-medium-flux.md): shared medium flux and radial adiabatic transport |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add solid photoionisation, mobile-carrier dynamics and collisional optical currents
to solprop's existing radial and Cartesian free-space routes, for single-pulse
filamentation and white-light generation inside homogeneous transparent solids.
Fused silica and undoped YAG are the first material targets. Recommend a
cycle-averaged solid Keldysh rate spanning multiphoton and tunnelling regimes, an
explicitly selected multiphoton alternative, and a bounded population model with
optional avalanche and carrier removal. Following
[SEP 3](archive/sep-0003-optmat-materials.md)'s decision D15, optmat owns the solid
photoionisation rate models, with their normalisation, averaging declaration and
validation, together with the curated material parameters and their provenance.
solprop tabulates those rates and owns the population/current equations, the
avalanche closure, translation and execution. This needs upstream optmat work,
specified in a written optmat work package that is still to be drafted (§3.2).
Carrier-resolved optical propagation is supported, but does not turn a
cycle-averaged rate into subcycle solid-state physics. Success requires numerical
and energy-accounting tests, three-backend CPU parity, medium-aware
launch/diagnostics, and documented comparisons with silica and YAG experiments. This
is a substantial response and material-contract extension, not a new spatial
propagator.

### 2. Motivation and scope

The immediate experiment is a focused femtosecond pulse entering a YAG plate:
diffraction, dispersion and Kerr self-focusing reshape it, photoionisation
creates carriers, the carriers change the index and absorb light, and the
coupled evolution generates a continuum. Silica supplies an important second
case, with delayed Raman response and rapid carrier trapping. The measurements
of Silva et al. [3] and Couairon et al. [2] are starting points, not exact
numerical truth. Cheng et al. [4] provide a longer-pulse YAG comparison that
makes the role and limitations of avalanche especially clear.

**Included in the implementation scope:**

- Homogeneous, isotropic linear dielectric response, constant along z within
  one material segment; radial symmetry for single filaments, Cartesian
  transverse planes for asymmetry and multiple filaments, and the existing
  line geometry with quantities per unit invariant width.
- Scalar, linearly polarised field and envelope grids, with the same declared
  cycle-averaged solid photoionisation model. Ordinary field-grid Kerr may
  generate harmonics; envelope plasma with explicit THG remains refused.
- Solid Keldysh photoionisation and a calibrated integer-order multiphoton
  alternative, both evaluated by optmat rate models, a selectable single-rate
  avalanche approximation, Drude absorption/refraction, and explicit recombination
  versus irreversible trapping within one pulse window.
- Source-specific silica and YAG material bundles and solid rate models in
  optmat; a validated explicit-parameter route for other transparent solids,
  through optmat's constructor from explicit parameters (SEP 3, D7).
- Absolute physical V/m fields, a medium-aware energy/flux contract for this
  route, carrier and work diagnostics, reproducible output and resume.
- NumPy reference, Numba CPU and Kokkos CPU implementations in that order.

**Deferred:** arbitrary elliptic polarisation, crystal tensor carrier transport
and orientation-dependent ionisation, anisotropic linear propagation,
semiconductor Bloch equations, coherent interband polarisation, solid
high-harmonic generation, dynamic band structure, dense-plasma reflection,
material damage, ablation and hydrodynamics. Heat accumulation, defects and
carrier memory between separate pulses, detrapping/reionisation, carrier
diffusion and transport between spatial nodes are also outside this proposal.

Delayed-heating avalanche models are compared in §4.5 and §10; they are a
follow-up, not an unreported part of the baseline. Gas avalanche, gas charge-state
ladders, solid-core fibres and region-dependent ionisation in general modal
guides remain separate work. Adding YAG here does not migrate every remaining
solid/metals dataset under SEP 3. Its Raman spectrum is not silently replaced
by silica Raman.

A simulation begins at the **inside entrance plane** of a crystal and ends
at its inside exit plane. The prescribed launch includes the actual focal
geometry and transmitted pulse energy. Air propagation and normal-incidence
Fresnel transmission can be explicit offline operations in an example; no
interface, refraction, reflected wave or etalon is simulated by pretending that
a discontinuous index is a smooth longitudinal profile.

### 3. Context

#### 3.1 Audited implementation and reuse

This Draft audits solprop at
`92112210357be14e0d1dfacd75c9eb3152839425`, optmat at
`c671daea59f73304169e6b77bd463bb133c7c4fd`, and solprop-kokkos at
`87ac816190ffb0c3978de7e236b6e30d7c068ab8`. Re-audit these interfaces before
implementation; the hashes document the evidence, not package-version pins.

| Existing component | Reusable part | Required difference |
|---|---|---|
| `solprop.spatial.SpatialMedium` and `SpatialDispersion`; [free space](../../guide/free-space/model.md), [Cartesian propagation](../../guide/free-space/cartesian.md) | Bulk dispersion, longitudinal square root, diffraction, physical spatial fields and UPPE prefactors | A nongas medium currently has dimensionless `density=1`. That cannot become an available carrier density |
| `driver.interface.prop_radial`, `prop_free` and `Simulation.from_radial/from_free` | Launch, spatial geometry, adaptive/fixed stepping, save planes | Admit an explicit solid response and normalise its launch with its medium |
| `nonlinear.responses.PlasmaResponse` | Memory-response organisation, zero-field handling, physical-current accounting | Gas species, single-event survival and collisionless free-electron current do not describe a solid |
| `nonlinear.ionisation` and `translate.ionisation` | Rate protocol, table/cache machinery and accuracy checks; after SEP 3 these tabulate optmat rate models | The `keldysh` helper is a gas adiabaticity parameter, not a solid production rate; gas rates cannot be parameterised by a bandgap. A solid rate is a distinct optmat model with its own protocol |
| [Envelope plasma](../../guide/media/envelope-plasma.md), `translate.envelope_plasma` | Separate current-source streams and explicit spectral assembly | Gas cycle quadrature starts with an instantaneous rate. Solid Keldysh is already cycle averaged and must not be averaged again |
| `backends.numpy_ref.envelope_plasma` and `backends.bodies` | Per-node temporal traversal and separate drive/loss organisation | Closed-form gas exposure no longer suffices with avalanche and removal |
| solprop-kokkos `backend.py`, `envelope_plasma.hpp` and `plasma.hpp` | Validated host records, residency, CPU column loops and deterministic scans | Coupled nonlinear populations cannot simply use the additive gas prefix scan |
| `spatial_energy`, `spatial_intensity`, `spatial_fluence` and spatial statistics | Transform weights and Parseval accounting | Their vacuum-admittance functional is not the solid's dispersive axial flux |
| Population statistics and output/resume | Saved-plane re-evaluation, deterministic reductions, provenance | Peak mobile, final mobile and trapped densities differ once carriers disappear |

The [architecture](../architecture.md), [conventions](../conventions.md) and
[kernel contract](../kernel-contract.md) remain the layering rules. Reuse the
longitudinal interaction-picture stepper. Material populations are recomputed
along time for **each trial RHS at each z**, including rejected steps; they are
not quantities advected from one z plane to the next.

#### 3.2 Optmat readiness, rate ownership and the SEP 3 boundary

SEP 3 migrates gas and solid data, gas ionisation rates and Raman responses,
including immutable material identities.

**Rate ownership (SEP 3, D15).** SEP 3 removes solprop's own gas-rate formulas:
rate models live in optmat, and solprop tabulates them. D15 extends that principle
to solids:
- **optmat owns** the Keldysh-family and multiphoton rate models, including the
  normalisation question of §4.2 and its reference values, the declaration that the
  rate is cycle-averaged, and the rates' validation. It also owns the carrier,
  collision, removal and avalanche coefficients as curated data.
- **solprop owns** carrier kinetics, the avalanche closure, collisional currents,
  tabulation, energy bookkeeping and propagation.

The first version of this Draft had solprop evaluate the rates from optmat
parameters. That plan is withdrawn, so that each rate law has one implementation
with one normalisation.
It does not provide solid carrier kinetics or a complete YAG model. The
upstream contract must distinguish an equilibrium optical model from an
excitation gap, photoionisation reduced mass from transport mass, available
transition density from atom/formula-unit density, and measured quantities
from fitted effective parameters and historical assumed inputs.

The audit found curated records for [2–5], including silica collision/trapping
data and YAG model parameters, but no public solid-ionisation parameter-bundle
API. The solid model registry recommends a silica model; a YAG catalogue entry
and individual nonlinear measurements are not a recommended broadband YAG law.
The upstream ionisation API currently concerns gases, and optmat has no solid rate
model yet; its roadmap lists solid multiphoton and field rates as next. This is
useful existing curation, not a completed propagation dependency.

**Upstream work required before acceptance.** This SEP needs optmat work of the same
kind SEP 3's Phase 0 needs:
- solid rate models: the Keldysh family with a resolved normalisation (Q1), and the
  calibrated multiphoton form;
- their public contract and identity;
- explicit-parameter construction and overrides;
- the carrier-parameter bundles of §5.1;
- validation against the sources in §4.6.

That work is to be specified in a written optmat work package, reviewed and
implemented upstream before this SEP is accepted. The work package has **not yet been
drafted**. Its absence is an acceptance blocker, not an implementation detail.

This proposal specifies the required upstream contract in §5.1 without relying
on another repository's private notes or source files. There is no new material
database in solprop. Until an installable optmat release supplies a complete
resolved bundle, a built-in solid selection refuses the missing capability;
it does not fill it with gas data or undocumented constants.

#### 3.3 Lessons from other implementations

Karpowicz's published simulation [8] is a useful comparison for a computationally
compact nonlinear-absorption plus Drude-current approach. The general lesson
is to separate carrier production, energy removal and current propagation.
An empirical power of the real field alone does not supply the solid Keldysh
crossover, carrier depletion, trapping or the energy delay before impact
ionisation. Section 4 derives this proposal independently; no external source
implementation is part of its specification or validation oracle.

### 4. Theory

#### 4.1 Optical representation, intensity and validity

Use the optics convention of [conventions](../conventions.md):

$$
 E(t)=\operatorname{Re}\{a(t)e^{-i\omega_0t}\},\qquad
 \widehat g(\Omega)=\int g(t)e^{+i\Omega t}\,dt .
$$

Here $a$ is a physical electric-field envelope in V/m. For the first, linearly
polarised model define the **rate-driving intensity**

$$
 I_{\rm rate}=\frac12 n_0\varepsilon_0c\,|a|^2,\qquad
 n_0=\operatorname{Re}n(\omega_0)>0 .
$$

It has units W/m². The photoionisation law takes peak field $F=|a|$, with
declared reference angular frequency $\omega_0$ in rad/s. On a field grid,
obtain $F$ from the analytic signal of the real field. This is a local
monochromatic approximation even when the optical field is broadband.
Neither substituting $|E(t)|$ into an averaged Keldysh rate nor inventing
an instantaneous frequency from a noisy phase derivative is authorised.

Distinguish $I_{\rm rate}$ from broadband axial Poynting flux (§4.7), and from
the current vacuum-admittance statistic. Harmonics and separated colours
undermine the single-frequency rate closure even when optically resolved.
Record the reference frequency, report broadening and temporal compression,
and qualify the rate over its declared domain. A field-grid result is not
automatically reliable for few-cycle ionisation; subcycle theories such as
[7] need a separate response and energy derivation.

The baseline assumes dilute excited carriers, an essentially unchanged bound
dielectric function and a forward, predominantly transverse field. It is not
a model of metallisation. A supplied material cannot override those assumptions
just by declaring a larger numerical range.

#### 4.2 Photoionisation: multiphoton, crossover and tunnelling

**Multiphoton alternative.** For a source whose $K$-photon coefficient
$\sigma_K$ is a probability rate per available transition,

$$
 w_{\rm MPI}(I)=\sigma_K I^K,\qquad
 R_{\rm MPI}=N_v w_{\rm MPI},\qquad
 [\sigma_K]={\rm s}^{-1}({\rm m}^2/{\rm W})^K .
$$

$N_v$ is the remaining available transition density in m⁻³; $R$ is a
volumetric production rate in m⁻³ s⁻¹. A coefficient reported per volume
instead multiplies $N_v/N_0$, not $N_v$. The low-field photon order is
usually the smallest energetically allowed integer near
$\lceil E_g/(\hbar\omega_0)\rceil$, but selection rules and the source's
threshold definition must be retained. A wavelength-specific fitted power
law must not be extrapolated by merely changing this integer.

For an intensity-loss coefficient $\partial_z I=-\beta_K I^K$,

$$
 R_0=\frac{\beta_K I^K}{K\hbar\omega_0},\qquad
 [\beta_K]={\rm m}^{2K-3}{\rm W}^{1-K}.
$$

Thus $\sigma_K=\beta_K/(N_0K\hbar\omega_0)$ only when both coefficients describe
the same event, wavelength and intensity convention. This prevents unit
mistakes and double counting with [SEP 2](sep-0002-two-photon-absorption.md).

**Solid Keldysh family.** The effective two-band model uses a gap $E_g>0$
in J and a reduced electron–hole mass $m_r>0$ in kg:

$$
 \gamma_s=\frac{\omega_0\sqrt{m_rE_g}}{eF},\qquad
 m_r^{-1}=\frac{1}{m_e^*}+\frac{1}{m_h^*}.
$$

This is the solid parameter; the atomic expression has a different factor
under its square root. Large $\gamma_s$ approaches multiphoton excitation,
small $\gamma_s$ tunnelling, and neither limiting power law covers the
crossover. The complete rate is a volume rate $R_0(F,\omega_0)$ before
depletion; use $R_{\rm PI}=R_0N_v/N_0$.

Define elliptic integrals by their **parameter**, not modulus, and Dawson's
function explicitly:

$$
 {\cal K}(m)=\int_0^{\pi/2}\frac{d\theta}{\sqrt{1-m\sin^2\theta}},
 \quad {\cal E}(m)=\int_0^{\pi/2}\sqrt{1-m\sin^2\theta}\,d\theta,
 \quad D(x)=e^{-x^2}\int_0^x e^{u^2}\,du .
$$

Write

$$
 m_1=\frac{\gamma_s^2}{1+\gamma_s^2},\quad m_2=1-m_1,\quad
 x=\frac{2E_g}{\pi\hbar\omega_0}\frac{{\cal E}(m_2)}{\sqrt{m_1}},
 \quad k=\lfloor x+1\rfloor,\quad v=k-x,
$$

$$
 A=\pi\frac{{\cal K}(m_1)-{\cal E}(m_1)}{{\cal E}(m_2)},\quad
 B=\frac{\pi^2}{d\,{\cal K}(m_2){\cal E}(m_2)} ,
$$

$$
 R_0=\frac{p\,\omega_0}{9\pi}
       \left(\frac{m_r\omega_0}{\hbar\sqrt{m_1}}\right)^{3/2}
       \sqrt{\frac{\pi}{2{\cal K}(m_2)}}
       \sum_{j=0}^{\infty}e^{-(k+j)A}
                D\!\left(\sqrt{B(j+2v)}\right).
$$

The prefactor has units m⁻³ s⁻¹. Converting the elliptic **moduli** in
original Keldysh [1], Eqs. (37–39), to the parameters above gives
$(p,d)=(2,2)$. Couairon et al. [2], Eqs. (7–8), print $(2,4)$ and explicitly
discuss their changed Dawson argument; Déziel et al. [6], Appendix A,
print $(4,2)$. A modulus-to-parameter substitution alone does not change
either numerical factor.
**These are not interchangeable spellings of one implementation.** The
Draft recommends the Keldysh family, with the Couairon form as the initial
filamentation comparator. Production normalisation is gated on Q1, which is
resolved upstream in optmat's rate model (§3.2).

A further source inconsistency needs resolution. Evaluating the printed [2]
formula with $E_g=9e$ J, $m_r=0.64m_e$, $n_0=1.45$, $\lambda_0=800$ nm and
$I=3.5\times10^{17}$ W/m² gives $\gamma_s\simeq0.99915$ and
$R_0\simeq6.54\times10^{38}$ m⁻³ s⁻¹; its text quotes
$1.6\times10^{38}$ m⁻³ s⁻¹. The printed [1] and [6] expressions give
approximately $4.37\times10^{38}$ and $8.74\times10^{38}$ m⁻³ s⁻¹
respectively at those inputs. These exploratory direct
evaluations are not qualified reference values or evidence that one
source's physics is preferable. Increasing the positive channel sum from
32 to 2000 terms leaves these rounded values unchanged; direct quadrature
of the elliptic and Dawson integrals reproduces them independently of
the special-function calls. Q1 must resolve the derivation, state counting
and reference numbers before acceptance; fitting a multiplier
to one printed point is not a resolution.

The upstream production evaluator must handle zero field exactly, weak-field
underflow, elliptic limits, cancellation in $\mathcal K(m_1)-\mathcal E(m_1)$, and a
converged channel sum with a tail bound. A bound on Dawson's function and
the geometric factor $e^{-A}$ can bound the positive omitted tail.
Locate channel closings $x(F)\in\mathbb Z$ explicitly and preserve the
chosen model's one-sided behaviour. Do not smooth them to make a spline
or the z stepper happy. The model exposes its channel-closing fields so that
solprop can segment its table there. Capped trial evaluation is only an overflow
defence; accepted states outside the physical/table domain must be reported and
refused. The derivation above states what solprop requires of the upstream model
and what its tables are checked against. It is not a second implementation in
solprop.

#### 4.3 Available states, avalanche and removal

Let $N_0$ be the available transition density, $n$ the mobile electron–hole
pair density, $n_t$ the trapped-excitation density, and $N_v=N_0-n-n_t$.
These are model populations, not numbers of chemical ions. Recommend

$$
 R_{\rm PI}=R_0(F)\frac{N_v}{N_0},\qquad
 R_{\rm av}=\alpha(I_{\rm rate})n\frac{N_v}{N_0},
$$

$$
 \dot n=R_{\rm PI}+R_{\rm av}-(\nu_r+\nu_t)n,\qquad
 \dot n_t=\nu_t n,\qquad
 \dot N_v=-R_{\rm PI}-R_{\rm av}+\nu_r n .
$$

Recombination $\nu_r$ returns a transition to the available pool; trapping
$\nu_t$ does not. Both rates are nonnegative in s⁻¹. The populations sum to
$N_0$ and stay nonnegative. Initial $n,n_t$ are explicit with $n+n_t\le N_0$;
initial drift current is zero. No removal means both rates are zero. An
effective lifetime must name its destination reservoir: silica's short
mobile-carrier lifetime is not automatically ground-state recovery.

The simple avalanche option uses a cycle-averaged Drude heating estimate,

$$
 \alpha(I)=\eta\frac{\sigma_D(\omega_0)I}{E_{\rm imp}},\qquad
 \sigma_D(\omega)=\frac{e^2\nu_c}
 {\varepsilon_0cn_0m_D(\nu_c^2+\omega^2)} .
$$

$\sigma_D$ is in m²; $E_{\rm imp}>0$ is effective energy per impact event
in J; $0\le\eta\le1$ is a declared efficiency. The historical closure
[2,4] uses $E_{\rm imp}=E_g$, $\eta=1$. These are assumptions, not universal
measurements. The survival factor bounds production and is approximately
one in the dilute limit of much of the literature.

**Avalanche is explicitly selected**, not automatically enabled with
photoionisation. No avalanche still allows Drude absorption when $\nu_c>0$.
The simple model has no heating delay and must be labelled accordingly.
MPI and Keldysh are alternative rates for the same transitions, not additive
channels.

#### 4.4 Current, optical work and Maxwell coupling

For the first scalar transport model, define $m_D$ independently of $m_r$:

$$
 \dot J_D+\nu_JJ_D=\frac{e^2}{m_D}nE,\qquad
 \nu_J=\nu_c+\nu_r+\nu_t .
$$

Carriers are born with zero mean drift; removal takes away the mean current
carried by the removed particles. There is no extra $(\dot n/n)J_D$ birth
term. If fitted optical damping already includes removal, the bundle must
declare its decomposition to avoid counting it twice. Separate electron
and hole currents would give
$e^2n[(m_e^*(\nu_e-i\omega))^{-1}+
(m_h^*(\nu_h-i\omega))^{-1}]E$ in the stationary limit. Only equal damping
permits collapse to a reduced transport mass; a photoionisation mass alone
does not establish that condition.

Let $\epsilon_{\rm PI}(F)$ be declared optical energy per photo-event.
Use $K\hbar\omega_0$ for the calibrated MPI absorption model. The common
Keldysh filamentation closure uses $E_g$ [2]; label this `gap-work`.
It omits excess excitation energy and is not an exact multiphoton-energy
model. A channel-energy closure needs an additional derivation. Store the
selected closure in provenance.

With $Q_{\rm PI}=\epsilon_{\rm PI}R_{\rm PI}$ in W/m³, a scalar envelope
loss current has the required cycle-averaged work:

$$
 j_{\rm PI}=\frac{2Q_{\rm PI}}{|a|^2}a,\qquad
 \frac12\operatorname{Re}(a^*j_{\rm PI})=Q_{\rm PI}.
$$

Use its vanishing-rate limit at zero field. On a field grid reconstruct
$J_{\rm PI}=\operatorname{Re}(j_{\rm PI}e^{-i\omega_0t})$ from the analytic
field. Its real instantaneous work oscillates; the identity is
**cycle averaged** under §4.1, not $EJ_{\rm PI}=Q_{\rm PI}$ at every sample.
Using $Q_{\rm PI}/E$ at carrier zero crossings would be singular.

Substitution of the carrier representation derives the envelope equation,

$$
 (\partial_t+\nu_J-i\omega_0)j_D=\frac{e^2}{m_D}na .
$$

For constant density/damping the optics conductivity is
$\sigma_{\rm opt}(\omega)=ne^2/[m_D(\nu_J-i\omega)]$, with positive real
part. The corresponding susceptibility is

$$
 \Delta\varepsilon_r(\omega)=
 -\frac{ne^2}{\varepsilon_0m_D(\omega^2+i\nu_J\omega)} .
$$

Its real part lowers the index; its imaginary part gives loss. Conjugating
to the internal FFT convention gives $\partial_t+\nu_J+i\omega_0$ in the
envelope equation. Do not copy an optics sign into an internal denominator.

Record $\int EJ_D\,dt$ (or its envelope counterpart) and
$\int Q_{\rm PI}\,dt$ separately, plus photo/impact/removal event counts.
Avalanche spends energy already absorbed by free carriers: adding
$E_gR_{\rm av}$ as another optical loss double counts it. The photo-loss
current uses $R_{\rm PI}$, never $\dot n$, which can be negative during
trapping.

The simple avalanche closure approximates available collisional heating
by $\sigma_D I n$, allocating
$E_{\rm imp}R_{\rm av}=\eta\sigma_D I n\,N_v/N_0$ to impacts.
This is a stationary heating approximation, not a transient carrier-energy
equation. Compare cycle-averaged Drude work to that budget and record its
error. Fast removal or modulation can invalidate it: the current damping
is $\nu_J$, whereas the simple heating coefficient uses $\nu_c$.
A large discrepancy disqualifies the simple heating closure even if
Maxwell/current integration converges. Recombination/trapping energies
leave the tracked mobile system; they do not generate optical gain.

For positive propagating $k_z$, the scalar forward Maxwell equation gives

$$
 \partial_z\widehat E\big|_{\rm NL}
 =\frac{i\mu_0\omega^2}{2k_z}\widehat P_{\rm bound}
  -\frac{\mu_0\omega}{2k_z}
        (\widehat J_D+\widehat J_{\rm PI}) .
$$

This follows from the $\partial_tJ$ and $\partial_t^2P$ sources in the
wave equation and forward factorisation. Internal assembly changes the
polarisation factor to $-i\mu_0\omega^2/(2\operatorname{Re}k_z)$ and
keeps the current factor real and negative, following the existing spatial
prefactor convention in weakly absorbing media. Carry a combined current
stream beside bound polarisation; no extra integral to manufacture plasma
polarisation is needed. DC and nonpropagating support need an explicit
admission/mask policy, not division by zero.

#### 4.5 Why a more elaborate avalanche model may be needed

In multiple rate equations [5,6], populations $n_j$ climb a conduction
energy ladder before its top population can impact-ionise. A simple
undepleted, constant-rate example is

$$
 \dot n_0=R_{\rm PI}-hn_0+2a_{\rm imp}n_L,\quad
 \dot n_j=h(n_{j-1}-n_j)\ (0<j<L),\quad
 \dot n_L=hn_{L-1}-a_{\rm imp}n_L .
$$

Here $h$ is the heating rate and $a_{\rm imp}$ the impact rate in s⁻¹.
Summing gives $\dot n=R_{\rm PI}+a_{\rm imp}n_L$: replacing $n_L$ by all
of $n$ misses the heating delay. Ladder length depends on photon energy
and impact threshold, not merely the gap. The momentum-conserving threshold
in [6] also involves effective masses and ponderomotive energy.

Déziel et al. [6] instead evolve mean carrier energy and obtain impact
production from the hot fraction of a distribution. This is a promising
small-state alternative for three-dimensional propagation, but assumes
a distribution and requires additional energy/impact parameters. It is
not an arbitrary delay attached to a simple avalanche coefficient.

Recommend the explicitly qualified single-rate model first, keeping a
separate population/current seam for a future energy model. When avalanche
determines the result, compare no avalanche, the simple model and an
independent delayed-heating calculation; report sensitivity. Agreement
with a damage threshold alone does not validate the carrier history or
emitted continuum. The pulse-duration dependence studied in [9] is useful
background for separating heating regimes, not a universal single-shot
filamentation calibration.

#### 4.6 Material parameters and source evidence

Each shipped value needs its source, equation/table/page, conditions, units,
uncertainty when given, model role and status. Convert to SI once in
optmat's resolved bundle. Fundamental constants come from the existing
CODATA-backed constants, not decimal copies.

| Quantity | Silica evidence | YAG evidence and unresolved work |
|---|---|---|
| $n(\omega)$, absorption and derivative | SEP 3's complex silica law, with grade/range policy | A qualified YAG law over the simulated band is a prerequisite; a catalogue entry is insufficient |
| Kerr and delayed response | SEP 3's consistent electronic/total $n_2$ and Raman bundle | [3] assumes $n_2=7\times10^{-20}$ m²/W, not a new broadband measurement; curate wavelength, polarisation and delayed fraction |
| Excitation gap $E_g$ | [2] uses $9e$ J and discusses competing gap choices | [3,4] use $6.5e$ J; [4] cites Xu–Ching [10]. Check the original gap interpretation |
| Photoionisation mass $m_r$ | [2] uses $0.64m_e$, citing Fischetti et al.; the original assignment needs checking | The reviewed YAG main texts do not establish a recommended reduced mass; a silica/free-electron substitute is not a YAG measurement |
| Transport mass and momentum time | [2] fits $10^{-14}$ s to transmission; it is model dependent | [3,4] estimate $3\times10^{-15}$ s; keep transport and photo masses separate |
| Available density $N_0$ | [2] uses $2.1\times10^{28}$ m⁻³ as background density | [4] uses $7\times10^{28}$ m⁻³ without establishing its counting basis; treat it as effective until resolved |
| Removal | [2] uses $150\times10^{-15}$ s and discusses trapping-dominated loss | [3,4] establish no general YAG lifetime; explicit neglect over a qualified window differs from invented data |
| MPI coefficients | [2]'s sixth-order weak-field coefficient converts to $9.6\times10^{-94}$ s⁻¹(m²/W)⁶; some reproduced experimental coefficients have disputed normalisations | [4]'s eleventh-order coefficient at 2.05 µm converts to $1.3\times10^{-170}$ s⁻¹(m²/W)¹¹; [3]'s seventeenth-order value is a comparison bound, not a recommendation |
| Avalanche | Select a heating/impact closure and calibration conditions | Same requirement; absorption cross-section (m²) and impact coefficient (m²/J) are distinct |

The conversions are $I_{\rm SI}=10^4I_{\rm W/cm^2}$,
$N_{\rm SI}=10^6N_{\rm cm^{-3}}$, and
$\sigma_{K,\rm SI}=10^{-4K}\sigma_{K,\rm W/cm^2}$ for a per-site MPI
coefficient. A volume rate additionally needs the density conversion.

A concrete audit check shows why compatible bundles matter. With
$m_D=m_e$, $n_0=1.80$ and $\tau_c=3$ fs, the Drude formula gives
$\sigma_D=2.058\times10^{-21}$ m² at 2.05 µm, agreeing with [4]'s rounded
$2.06\times10^{-21}$ m². At 3.1 µm it gives
$4.093\times10^{-21}$ m², whereas [3] prints $2.6\times10^{-24}$ m².
The latter was checked on the printed page; its avalanche equation also
contains an unexplained field rather than intensity factor. Do not import
that cross-section as a default or silently correct the source. Derive the
working coefficient from declared parameters, record the inconsistency,
and retain [3] as a qualitative experimental comparison.

The vacuum critical-density scale
$n_{\rm cr,0}=\varepsilon_0m_D\omega_0^2/e^2$ is a diagnostic, not a solid
damage threshold. For a real background $\varepsilon_b=n_0^2$, the
constant-density Drude condition $\operatorname{Re}\varepsilon=0$ occurs
at $\varepsilon_b\varepsilon_0m_D(\omega_0^2+\nu_J^2)/e^2$. Forward,
weak-excitation propagation can fail before that. Output
$|\Delta\varepsilon|/|\varepsilon_b|$, depletion and domain excursions.
A model validity limit can stop a run; density clipping must not make an
invalid run appear valid.

#### 4.7 Medium-aware spatial energy and launch

Keep the existing physical state: V/m in Cartesian geometry and V m in
radial spectral geometry. Change its **measurement and launch contract**.

For the admitted scalar forward-wave approximation, sample axial admittance
on propagating spectral support:

$$
 Y_z(k_\perp,\omega)
 =\varepsilon_0c^2\frac{\operatorname{Re}k_z(k_\perp,\omega)}{\omega}.
$$

For transparent uniform media, if $E_a$ is the analytic physical field and
$G_a$ is synthesised after multiplying its spectrum by $Y_z$, define

$$
 S_z(\mathbf r,t)=\tfrac12\operatorname{Re}(E_aG_a^*),\qquad
 U=\int S_z\,dA\,dt .
$$

For uniform absorbing media, local G uses SEP 34's complex admittance; its real
part $Y_z$ weights total energy. Certified real-index radial paths use its
retained forward-component closure; omitted backward interference is outside
that local measurement.
Equivalently use temporal Parseval and Hankel/Cartesian weights with $Y_z/2$.
This is the metric of the **scalar equation**: metric-weighting its current
source recovers $-\operatorname{Re}(\widehat E^*\widehat J)/2$.
It is not the full TE/TM Maxwell metric of tightly focused vector beams.
A paraxial monochromatic wave gives $n_0\varepsilon_0c|a|^2/2$, recovering
the old functional in the vacuum paraxial limit. Dispersive/angular weighting
cannot be replaced by a constant $n_0$.

Use the same metric in energy-normalised launch, fluence, energy spectra,
error control where an energy norm is declared, and work tests.
Pointwise broadband flux need not be nonnegative; do not clip it into a
rate drive. Rates use §4.1's positive proxy. Specify zero-frequency/endpoints
and refuse launch energy on nonpropagating support.

[SEP 34](archive/sep-0034-medium-flux.md) owns the shared implementation for isotropic
media, including launches, noise, observations, endpoint rules, certified radial
forward transport and backend qualification. Use its constants consistently with
the scalar source prefactor.
This response consumes that convention without a separate normalisation switch.
The change replaces the vacuum-weighted definitions directly; no legacy policy
or old-file migration is required. Admitted radial gradients and declared weak
steps use SEP 34's forward amplitude correction. Paths outside its real-index
certificate, varying absorption and general gradient scattering require a
separate model.

### 5. Proposed interface

#### 5.1 Upstream material contract

Optmat supplies an immutable `SolidCarrierParameters`-style record; spelling
is provisional until agreed upstream. It contains:

- Material identity, doping/composition, grade, temperature and orientation
  assumptions; source IDs, content digests and validity ranges.
- `band_gap` (J), `photo_reduced_mass` (kg), `available_density` (m⁻³) and
  precise counting/transition convention.
- `transport_mass` (kg), `collision_rate`, `recombination_rate` and
  `trapping_rate` (s⁻¹), distinguishing measured, fitted, assumed and
  deliberately neglected processes.
- Rate-specific MPI order/coefficient with rate units and reference
  wavelength, or the bundle compatible with the selected Keldysh definition.
  Formula identity and material identity are separate.
- Impact-energy/efficiency assumptions when avalanche is requested, and
  resolved optical/nonlinear laws required by the run.
- The **solid rate model** itself. It returns the volume rate $R_0(F;\omega_0)$ in
  m⁻³ s⁻¹ before depletion, vectorised, and declares:
  - that it is cycle-averaged, for linear polarisation, with its reference
    frequency;
  - its field range and channel-closing fields;
  - its formula identity and status flag, following SEP 3's identity contract;
  - a constructor from explicit parameters, and overrides that mark the identity
    modified (SEP 3, D7).

A historical study bundle can be useful without being `recommended`.
Missing fields remain missing and incompatible selections refuse by name.
Transport mass is never auto-filled from the photoionisation mass.
Optmat owns the rate models, the data and compatible bundle selection. solprop
tabulates the rate and evaluates the kinetic and current equations (§3.2, SEP 3
D15). Dependency and public release/API details follow SEP 3; the solid additions
are specified in the upstream work package of §3.2.

#### 5.2 Solprop objects and example

Propose `OptmatSolidRate`, a thin adapter that tabulates an optmat solid rate
model, and `SolidPlasmaResponse` and a small `SolidCarrierModel` configuration under
`solprop.nonlinear`. There is no solprop implementation of the Keldysh or multiphoton
formula. Add
the agreed modules to the architecture/structure inventory when implemented.
Use a dedicated solid leaf, not `PlasmaResponse(gas=...)` or a gas `Species`.

A rate declares `cycle_averaged`, scalar linear polarisation, reference
frequency, dimensional output and supported field range.
`SolidCarrierModel` explicitly chooses `avalanche="none"` or
`"single_rate"`, removal reservoirs and photo-work closure. A response
carries initial mobile/trapped densities and a resolved bundle. Initial
densities default to zero, not an arbitrary avalanche seed.

Illustrative, deliberately non-executable sketch:

```text
parameters = resolve_solid_carriers(
    material="YAG", model_id=<qualified optmat bundle>,
    temperature=294.0,
)
rate = OptmatSolidRate(parameters, model=<optmat Keldysh model, Q1 resolved>,
                       wavelength=3.1e-6)
carriers = SolidCarrierModel(
    avalanche="single_rate", photo_work="gap-work",
    removal=<explicit, source-qualified choice>,
)
plasma = SolidPlasmaResponse(parameters, rate, carriers)
medium = SpatialMedium.from_optmat(<resolved YAG optical model>)
responses = [<matched bound Kerr response>, plasma]
# Use existing grid, spatial pulse and settings constructors.
# Pulse energy and focus refer to just inside the entrance face.
result = prop_radial(grid, space, medium, responses, pulse, settings=settings)
```

No convenience `plasma=True` chooses a solid model implicitly. Scans vary
declared overrides/model selections, each entering run identity. Supplied
bundles obey the same units, ranges and provenance checks as built-ins.

Refuse mixed gas/solid populations at one node in this first version,
arbitrary vector/elliptic states, general modal/slab guides, z-varying solid
parameters, duplicate MPI/TPA loss for one transition, mismatched material/
rate identities and unqualified backends. Absorption that does not generate
carriers remains a separate response.

#### 5.3 Output

Distinguish peak mobile density over the pulse, mobile density at its end,
trapped density, photo-event count, impact-event count, photoionisation
work and Drude work. Densities are m⁻³ and local integrated work is J/m³.
Transverse integrals of that work are J/m, becoming pulse-energy losses
in J only after integration along z. The line geometry remains per unit
invariant width.

Do not reuse a gas statistic that assumes final density is the maximum.
Propagation and saved/periodic statistics use the same population evaluator.
Full histories are optional with an explicit storage cost, not mandatory
output at every stage.

Record resolved optmat identities, overrides and omitted processes,
rate/formula identity, carrier frequency, photo-work and transport/removal
closures, integrator/table tolerances and spatial normalisation. Include
these in translation caches, scans and HDF5/in-memory resume. Read older
files as recorded; refuse resume without enough information to reconstruct
the physical model.

### 6. Implementation strategies

#### 6.1 Separate solid response, shared execution concepts

| Strategy | Assessment |
|---|---|
| Replace gas ionisation potential with a bandgap | Small diff, wrong rate/density semantics and missing transport/removal; reject |
| Arbitrary callbacks inside kernels | Python flexibility breaks the native seam; reject |
| Reuse gas tables unchanged | Share interpolation primitives, but units, cycle convention, thresholds and validity policy need a distinct contract |
| Compile a dedicated solid population/current record | Recommended: preserve gas physics and expose solid assumptions |
| Full semiconductor kinetic solver now | Far larger state and data requirements than a first filamentation model |

optmat owns the rate definitions; solprop's physics layer owns a readable population
reference. Translation resolves optmat once, builds tables, samples
admittance/current factors and validates combinations. Backends receive arrays,
scalar constants and small enumerated choices; no optmat or special-function
evaluation occurs inside a stage.

Add a `SolidCarrierAssembly` leaf to `ResponseTables`: rate table with its
units/convention, population parameters, current coefficients, initial
conditions, source factors and domain controls. It is distinct from
`plasma_channels` and gas cycle moments. Initial scope needs no z-dependent
carrier stage tuple; any later stage-bundle extension is versioned separately.

The response leaf and flux data change the seam: **bump `SCHEMA_VERSION`
in Phase 1**, update literal contract tests, native admission/binding
requirements, cache identities and changelog. Extend the output schema then
too. Ship the corresponding Kokkos adapter compatibility update in that
phase: existing responses still qualify on the revised schema, while the
new solid leaf is refused before allocation until Phase 3. Older Kokkos
releases correctly reject the unknown schema. The installed-backend
preference must not silently switch the requested backend.

#### 6.2 Rate tables and temporal integration

Tabulate $R_0(F)$ in log field/log rate, with explicit zeros and a mixed
absolute/relative budget. Segment at channel closings and data boundaries.
Compare off-knot points and integrated production to direct evaluation of the
upstream rate model, and take the segment boundaries from its declared
channel-closing fields. A global cubic needing millions of knots to hide a kink is
unacceptable. Table, temporal and z-step accuracies are independent controls.

Recommend a positivity-preserving symmetric composition of exact population
transfer subproblems:

- Photoionisation transfers $N_v$ to $n$ at frozen $w=R_0/N_0$.
- Avalanche alone conserves $M=N_v+n$ and has the logistic solution of
  $\dot n=(\alpha/N_0)n(M-n)$.
- Removal decays $n$ at $\nu_r+\nu_t$, dividing the removed amount between
  $N_v$ and $n_t$ in the declared rate ratio.

For example: photo half-step, avalanche half-step, removal full-step,
avalanche half-step, photo half-step, with field coefficients sampled
consistently at the temporal midpoint. This is second-order for smooth
coefficients and preserves the population simplex. Record event increments
from each transfer, not by subtracting almost equal total populations.
Use stable `expm1`/logistic limits; never clip negative populations to hide
an error.

This numerical recommendation requires independent adaptive-ODE verification
under Q4. Refine temporal substeps until populations **and currents/work**
converge. Refining z does not repair temporal error. A hard substep budget
returns failure status, not silently truncated integration. Subdivision and
operation-order rules are shared across CPU backends.

Integrate current causally from the left window edge. Exponential recurrences
for piecewise constant/linear drive avoid an explicit stability restriction
$\Delta t\,\nu_J\ll1$; accuracy still needs changes in $nE$ or $na$ resolved.
Evaluate exponentials of interval lengths, not $e^{+\nu_Jt}$ and
$e^{-\nu_Jt}$ across the whole window. Internally the envelope uses complex
damping $\nu_J+i\omega_0$.

A spectral $1/(\nu_J+i\omega)$ multiplier is a constant-coefficient alternative,
but imposes periodic finite-window semantics without padding/boundary
correction. Prefer causal recurrence and verify its stationary limit.
Refuse a truncated input with appreciable left-edge field rather than inventing
its previous carrier history.

#### 6.3 Backends

**NumPy reference first.** Clear multipass rates, populations, currents and
source assembly; independent SciPy ODE comparisons in tests. Support both
spatial transforms with their existing scaling. Carry the scalar analytic
field explicitly on field grids when needed by the rate.

**Numba CPU next.** Compile local arithmetic in the body factory, fuse
histories where possible and parallelise independent spatial columns.
Preallocate everything. Keep strict arithmetic, deterministic reductions,
no `fastmath` and the existing cache policy.

**Kokkos CPU next.** Add compiled records, native views and bindings, keep
currents/history buffers resident and list all plans/allocations in the
ledger. Start with one column traversal per host work item. Qualify against
NumPy and across CPU thread counts.

**GPU later.** An additive plasma prefix scan is not a solver for coupled
nonlinear populations. First assess node-parallel sequential time scans;
measure occupancy and serial cost before choosing another algorithm.
Affine current recurrences have associative compositions, but the complete
population map generally does not. Existing gas-GPU speedups are not evidence
for this response. CPU completion does not establish CUDA/HIP support.

#### 6.4 Size of change and expected performance

This is a **large cross-layer feature**: scientific curation upstream,
reference physics and energy semantics, then two compiled implementations.
A planning estimate is 15–25 solprop source modules touched or added and
5–10 native/adapter files in solprop-kokkos, plus tests, documentation and
optmat rate-model, data and API work. These are scope estimates, not measured future diffs
or elapsed-time commitments. Rate evaluation is the smaller part; resolving
material ambiguities and validating energy/launch conventions may take
longer than writing the population loop.

Let $S$ count nonlinear RHS calls, $N$ spatial columns, $T=n_{\rm to}$ fine
time samples and $q$ average temporal subdivision. Excluding transforms,

$$
 C_{\rm carrier}=O(SNTq),\qquad
 C_{\rm table}=O(N_{\rm knots}C_{\rm direct\ rate}) .
$$

Table construction is paid per distinct resolved rate, not per stage.
No elliptic integrals or channel sums run in the hot loop. An $L$-level
energy ladder adds approximately $O(L)$ population work and state per
column; a small energy model has fewer variables but potentially stiff
nonlinear dynamics.

Transform count matters. The current stream adds a forward temporal
transform and its spatial projection/transform beside bound polarisation.
A field-grid averaged rate can require an extra complex inverse transform
for the analytic field unless already available. On an envelope grid
$|a|$ is available. Reusing gas elliptic-polarisation cycle quadrature would
add work for a physical operation the solid model does not require.

| Regime | Expected cost driver |
|---|---|
| Small radial envelope problem | Lookup, temporal scan, dispatch and substeps can be a significant fraction |
| Large Cartesian field problem | Transforms, extra current buffers and memory bandwidth, with possible temporal serial cost |
| Rapid avalanche/removal | More temporal subdivisions; strong optical effects may also cause more z-step rejections |
| Prescribed carriers without photoionisation | Drude recurrence/current transform remain; no avalanche does not mean no cost |
| Later delayed-heating model | More state and temporal work; no reliable fixed slowdown without measurement |

A measured cost model is $C_1/C_0=1+f(r-1)+g$: $f$ is the old response
fraction, $r$ its new cost ratio and $g$ extra transform cost relative to
the old total. Illustratively, $f=0.2$, $r=3$, $g=0.2$ gives 1.6; it is
**not a forecast for YAG**. Adaptive propagation can change $S$ too.
No full-crystal runtime or universal CPU/GPU speedup is claimed.

A float64 history costs $8NT$ bytes, a complex128 stream $16NT$.
For 256 radial nodes and 8192 fine samples these are 16 MiB and 32 MiB;
for $256^2$ Cartesian points and 4096 fine samples, 2 GiB and 4 GiB.
These are **per array**, before solver stages, FFT workspaces and output.
Fusing populations into per-column state avoids full histories for every
population, but not all globally transformed current buffers. Include
reference intermediates, analytic fields, current spectra, diagnostics and
substep scratch in `memory_ledger()`.

Record construction, rate-table size, one RHS, about ten accepted steps
with rejection counts, memory and diagnostic overhead in the
[benchmark ledger](../benchmarks.md). Compare matched Kerr-only, photo-only,
photo-plus-Drude and avalanche/removal cases. Do not extrapolate windows
to complete crystal lengths. Full experiments belong in manually run
worked examples.

No new numerical dependency is proposed beyond NumPy/SciPy and the existing
native stack. Optmat follows SEP 3; any other dependency requires agreement.

### 7. Verification and validation

Separate **formula correctness**, **numerical convergence** and **physical
model adequacy**. Matching another code or a spectral picture establishes
none of all three together.

1. **Rates and sources.** Q1 is resolved upstream before production admission.
   optmat's own qualification checks the elliptic parameter/modulus conversion,
   channel-tail and roundoff bounds, threshold orders and asymptotic limits.
   solprop checks event/volume units, SI conversion, the declared averaging,
   off-knot interpolation and integrated production against direct upstream
   evaluation, and independently re-checks at least the limits of §4.2 and one
   reference value per model. Figure comparisons are limited by graphical precision.
   Underflow uses an absolute production budget, not relative error at zero.
2. **Population invariants and analytic limits.** Zero field/zero seed stays
   unexcited; carriers decay into the specified reservoir; populations remain
   nonnegative and sum to $N_0$. Photo-only, initially untrapped evolution is
   $n=N_0-(N_0-n_{\rm init})e^{-\int w\,dt}$. Avalanche-only is logistic;
   dilute constant-drive $\dot n=R+an$ gives
   $n=n_{\rm init}e^{at}+R\,\mathrm{expm1}(at)/a$ with its $a=0$ limit.
   Compare event counts too. Tolerances follow roundoff and analytic accuracy.
3. **Temporal convergence.** Compare transfer/current recurrences to a
   separately implemented, tightly converged adaptive ODE solver. Halve time
   spacing/substeps at three levels and require second-order asymptotics for
   smooth coefficients. Study threshold crossings separately. Exercise
   stiff cases and the subdivision-budget failure path.
4. **Drude and optical work.** Recover complex conductivity, negative index
   correction, positive absorption, current transients, removal damping and
   the collisionless limit. Monochromatic MPI obeys
   $I(z)=I(0)[1+(K-1)\beta_K I(0)^{K-1}z]^{-1/(K-1)}$
   for $K>1$ and negligible depletion; $K=1$ gives exponential absorption.
   Check the envelope work factor two and its averaged real-field limit.
   Detect double counting of impact loss. Separate exact Maxwell work tests
   from the simple avalanche's approximate heating budget.
5. **Spatial flux and launch.** Verify Parseval with $Y_z$, monochromatic
   index scaling, resolved radial/Cartesian agreement and lossless linear
   energy conservation. With current alone compare the z change of flux to
   integrated work, resolving quadrature and boundary loss independently.
   Never substitute the rate intensity proxy for physical flux.
6. **Short propagation and representations.** Test prescribed plasma,
   photoionisation, trapping and avalanche separately and with Kerr/Raman.
   Field/envelope parity applies only to a common slowly varying model,
   with tolerance derived from time/bandwidth error. Broadband/THG cases
   have separate resolution and model-validity studies.
7. **Backends and state.** Compare one RHS, one step, eight steps, each
   statistic and a short saved run. Check thread invariance, repeatability,
   rejection/retry independence, resume and scan identity. No population
   persists from a rejected trial. Derive positive tolerances from reduction
   lengths, conditioning and table/temporal budgets. Test every refusal and
   missing-data path.
8. **Experiments.** A YAG worked example based on [3] compares angular and
   integrated spectra and focusing evolution, including sensitivity to
   $n_2$, gap, masses and collision time. Retain its source discrepancies.
   A silica example based on [2] compares transmission and carrier/fluence
   evolution, distinguishing trapping from recombination; tight-focus data
   do not certify a scalar model at high numerical aperture. Use [4] for
   a 2.05 µm simple-avalanche comparison, not damage certification. Duration/
   intensity sweeps against [5,6] expose the missing heating delay.

Set numerical tolerances before comparisons and justify their error sources.
Material uncertainty does not weaken backend parity requirements. Report
experimental geometry, uncertainty, data/model gaps and sensitivity; do not
promise every fine feature of a continuum.

Automated propagation tests and benchmarks obey the approximately ten-step/
seconds-scale rule. Full-size literature runs are manual worked examples.
Code phases run `poe check` and, before commit/merge, `poe test-slow` and
`poe cov`. Kokkos also runs its native/sanitizer/wheel/Python gate and
solprop's gate with the freshly installed CPU backend. GPUs are qualified
separately on actual devices.

### 8. Documentation

Add `docs/model/solid-ionisation.md` with derivation, model matrix,
material selection, flux/intensity distinction, energy bookkeeping, source
discrepancies and worked examples. Extend [free space](../../guide/free-space/model.md),
[Cartesian propagation](../../guide/free-space/cartesian.md),
[ionisation](../../guide/media/ionisation.md),
[nonlinear responses](../../guide/media/nonlinear.md) and
[envelope plasma](../../guide/media/envelope-plasma.md) to distinguish solid and
gas models. Update launch, statistics, output/resume and processing docs.

Add a short deterministic bulk tutorial, plus manually run YAG white-light
and silica worked examples. State incident/transmitted energy, focus,
grade, omitted processes, backend and runtime. A bulk example is not a
certified damage prediction. Commit regenerated worked-example outputs.

Update architecture/structure, conventions, kernel contract, backend
capabilities, accuracy and benchmarks with implementation. Docstrings state
units and meanings of each mass, density, rate and work quantity. Record
the deliberately distinct solid closure on
[Differences from Luna](../../guide/workflows/luna.md), update usage/README capability
lists and add a changelog entry citing the SEP. At Final the manual links
to the archived proposal.

### 9. Compatibility and impact

The feature is explicit and additive to gas physics: no ADK/PPT defaults,
gas survival or gas-current convention changes. Spatial launch and observation
semantics use SEP 34's shared axial-flux convention, which replaces the old
vacuum-weighted default without a legacy option. Physical V/m state layout
stays unchanged.

Response/flux records require a seam bump with literal tests and Kokkos
check-by-name requirements updated together. New metadata/statistic
semantics require an output-schema update. Unsupported files fail clearly;
no old-file reader or migration is required for this change. Unknown material
or metric identities prohibit automatic resume.

Changing later from simple avalanche to delayed heating, or from gap-work
to another energy closure, changes physics and run identity. It is not a
backend optimisation. Tensor carrier dynamics and solid guides are refused
even when their geometrical ingredients exist.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| MPI only | Useful calibrated limit; cannot follow tunnelling/crossover during focusing |
| Add MPI and Keldysh | Double counts the same transitions; choose one compatible rate |
| Atomic ADK/PPT with $I_p=E_g$ | Incorrect solid band/state-density model |
| Real-field power loss plus Drude | Compact phenomenology needing explicit cycle calibration; not a complete solid rate/population model |
| Always enable simple avalanche | Immediate heating can dominate short-pulse predictions incorrectly; explicit selection/sensitivity required |
| MRE immediately | Resolves heating delay but adds state, parameters and qualification scope; separate extension |
| DRE immediately | Fewer energy variables, but still needs impact/distribution assumptions and energy bookkeeping |
| Subcycle theory or semiconductor Bloch equations | Appropriate for coherent/few-cycle physics; much broader first implementation |
| Instantaneous Drude index from carrier density | Misses finite-window current transients and obscures population/current coupling |
| Copy constants into solprop | Conflicts with SEP 3's single data authority and loses provenance |
| Evaluate the solid rates in solprop from optmat parameters | The first version of this Draft. It gives two homes to the normalisation question and diverges from SEP 3's D1/D15; withdrawn |
| Retain vacuum and axial-flux normalisation policies | Duplicates semantics; use SEP 34's common uniform-medium convention with a clean breaking change |
| Declare damage at critical density | A termination criterion does not supply lattice/defect/damage dynamics |

### 11. Phases

Each phase is independently mergeable and appends its SEP 0 report to Part II:
paths, deviations, decisions, actual checks, sources verified/unverified,
short-window benchmarks, limitations, numbered questions and handoff.
Use `feature/sep-0005-solid-ionisation-<phase>` branches and matching Kokkos
branches. A Draft does not authorise implementation.

Before acceptance, the optmat work package of §3.2 is written, reviewed and
implemented; Q1 is resolved within it. The material/energy/flux choices in Q2–Q4 are
settled, or the remaining data requirements become binding phase-entry criteria.

**Phase 0 — Upstream rate models, source resolution and data.** Deliver optmat's
solid rate models (§3.2, §5.1), vocabulary and compatible immutable bundles, a YAG
optical/Kerr selection, source/range status and public source report. Verify units,
mass/density semantics and discrepancies. Agree an installable dependency contract.
Historical assumed bundles do not become recommendations without evidence.

**Phase 1 — NumPy physics, flux and integration.** Consume SEP 34's qualified
medium-flux foundation. Deliver references, translation,
launch/diagnostics, schema changes, output/resume, refusals, numerical tests and
manual. Completion covers field/envelope, silica/YAG, radial/Cartesian/line and all
selected baseline processes. Numba/Kokkos refuse the feature until qualified; the
matching Kokkos schema compatibility update preserves its existing-route
qualification. Compare temporal integration with independent ODE solutions and
report averaged-rate limitations.

**Phase 2 — Numba CPU.** Deliver compiled bodies/column traversal, source
assembly, memory ledgers, statistics and the complete CPU parity matrix.
Benchmark construction, RHS, about ten steps and diagnostics. Run required
solprop gates/lanes; remove refusals only for supported combinations.

**Phase 3 — Kokkos CPU.** Extend bindings, tables/views, admission, transforms,
population/current execution and statistics. Qualify parity, thread
invariance, resource lifetime, native sanitizers and packaging. Run both
repositories' gates. GPU admission stays refused until separately
qualified; portable source is not evidence of device support.

**Phase 4 — Physical comparisons and completion.** Run full worked examples
manually and publish material sensitivity, convergence, source-check and
limitation reports. Move remaining questions to named roadmap/issues,
update changelog/capabilities and archive at Final. NumPy, Numba and
Kokkos CPU parity are required, with no exception proposed.
Delayed-heating/subcycle models, solid guides and GPU remain follow-ups.

### 12. Open questions

- **Q1 — Production Keldysh definition.** Maintainers and optmat, which owns the
  rate model (§3.2), in the upstream work package before acceptance: reconcile [1],
  [2] and [6], including state counting, prefactor, Dawson argument, elliptic
  conventions and the quoted silica rate. Approve a derived law and independent
  reference values. Named comparators may coexist; an unqualified default or fitted
  multiplier is not acceptable.
- **Q2 — Material bundles.** Optmat and maintainers, Phase 0: resolve YAG
  masses, density counting, removal and broadband optical/nonlinear laws;
  trace silica mass, lifetime and gap to primary sources. Separate historical
  bundles from recommendations and state when neglecting an unmeasured
  process is acceptable.
- **Q3 — Energy/population closure.** Maintainers, before acceptance:
  approve gap-work for Keldysh, photon-work for MPI, explicit simple
  avalanche, and trapping versus recombination. Set validity tests for
  transient heating and few-cycle/broadband failure. Requiring accurate
  heating delay expands scope and requires revising this Draft.
- **Q4 — Flux and numerics.** Maintainers before acceptance, with numerical
  qualification in Phase 1: approve the scalar admittance metric,
  launch/output policy, nonpropagating support and symmetric transfer/current scheme. Set
  independent temporal/table budgets and failure handling. No clipping
  may hide negative populations or an energy mismatch.
- **Q5 — Experimental evidence.** Phase 0/4 owners: obtain/check [3]'s
  supplementary rates and input geometry; resolve its focal-length and
  absorption discrepancies where possible; identify quantitative reference
  data. Unavailable evidence stays a limitation, not an invented oracle.
- **Q6 — Later work.** Phase 4 owner: create follow-ups for delayed-heating
  avalanche, phase-resolved rates, vector/anisotropic currents, regional
  solid-guide responses, YAG Raman when needed, inter-pulse memory and
  device qualification. Discussion here is not implementation coverage.

These are scientific review questions, not invitations to make undocumented
choices while coding. The proposal remains Draft until its acceptance-critical
questions are settled.

### 13. References

1. L. V. Keldysh, “Ionization in the field of a strong electromagnetic wave,”
   *Sov. Phys. JETP* **20**, 1307–1314 (1965), translated from *Zh. Eksp.
   Teor. Fiz.* **47**, 1945 (1964).
   [Journal record](https://www.jetp.ras.ru/cgi-bin/index/e/20/5/p1307?a=list).
   No DOI established in this audit.
2. A. Couairon, L. Sudrie, M. Franco, B. Prade and A. Mysyrowicz,
   “Filamentation and damage in fused silica induced by tightly focused
   femtosecond laser pulses,” *Phys. Rev. B* **71**, 125435 (2005),
   [doi:10.1103/PhysRevB.71.125435](https://doi.org/10.1103/PhysRevB.71.125435).
3. F. Silva, D. R. Austin, A. Thai, M. Baudisch, M. Hemmer, D. Faccio,
   A. Couairon and J. Biegert, “Multi-octave supercontinuum generation from
   mid-infrared filamentation in a bulk crystal,” *Nat. Commun.* **3**, 807
   (2012), [doi:10.1038/ncomms1816](https://doi.org/10.1038/ncomms1816).
4. S. Cheng, G. Chatterjee, F. Tellkamp, A. Ruehl and R. J. D. Miller,
   “Multi-octave supercontinuum generation in YAG pumped by mid-infrared,
   multi-picosecond pulses,” *Opt. Lett.* **43**, 4329–4332 (2018),
   [doi:10.1364/OL.43.004329](https://doi.org/10.1364/OL.43.004329).
5. B. Rethfeld, “Unified model for the free-electron avalanche in
   laser-irradiated dielectrics,” *Phys. Rev. Lett.* **92**, 187401 (2004),
   [doi:10.1103/PhysRevLett.92.187401](https://doi.org/10.1103/PhysRevLett.92.187401).
6. J.-L. Déziel, L. J. Dubé and C. Varin, “Dynamical rate equation model for
   femtosecond laser-induced breakdown in dielectrics,” *Phys. Rev. B*
   **104**, 045201 (2021),
   [doi:10.1103/PhysRevB.104.045201](https://doi.org/10.1103/PhysRevB.104.045201).
7. P. A. Zhokhov and A. M. Zheltikov, “Field-cycle-resolved photoionization
   in solids,” *Phys. Rev. Lett.* **113**, 133903 (2014),
   [doi:10.1103/PhysRevLett.113.133903](https://doi.org/10.1103/PhysRevLett.113.133903);
   [author preprint](https://arxiv.org/abs/1402.5530).
8. N. Karpowicz, “Open source, heterogeneous, nonlinear optics simulation,”
   *Opt. Continuum* **2**, 2244–2254 (2023),
   [doi:10.1364/OPTCON.502567](https://doi.org/10.1364/OPTCON.502567).
9. B. C. Stuart, M. D. Feit, S. Herman, A. M. Rubenchik, B. W. Shore and
   M. D. Perry, “Nanosecond-to-femtosecond laser-induced breakdown in
   dielectrics,” *Phys. Rev. B* **53**, 1749 (1996),
   [doi:10.1103/PhysRevB.53.1749](https://doi.org/10.1103/PhysRevB.53.1749).
10. Y.-N. Xu and W. Y. Ching, “Electronic structure of yttrium aluminum
    garnet,” *Phys. Rev. B* **59**, 10530 (1999),
    [doi:10.1103/PhysRevB.59.10530](https://doi.org/10.1103/PhysRevB.59.10530).

**Source-check boundary.** The primary articles [2–6] were read for the
models discussed. Printed pages [2] 125435-3, [3] 4, [4] 4330 and
[6] 045201-9 were inspected for equations/units, and [5]'s ladder derivation
was read. Existing optmat source classifications were compared with the
articles. SI MPI conversions and Drude spot checks were recomputed.
The direct Keldysh calculations expose an unresolved discrepancy; they
are not a qualified rate implementation.

The available extraction of [3] has glyph/identity warnings and no
supplement; its printed Methods confirms the disputed cross-section.
The extraction of [4] has damaged mathematical glyphs, checked against
its printed rate equation/coefficient. The extraction of [6] has an author
matching warning; the printed author line and appendix identify Déziel,
Dubé and Varin and the equations. No machine-generated figure description
was used as scientific evidence.

The original [1] printed Eqs. (37–39) were checked, including the modulus
conversion, but their derivation and state counting have not been fully
reconciled with later formulas. The [7] preprint supplies the scope of a
subcycle alternative, not an implementation specified here. The identity
and relevant approach of [8] were examined, but no timing/accuracy claim is
imported from it.
Reference [9] is pulse-duration background; [10] is the band-structure
lead cited by [4]. Neither supplies a newly verified coefficient here.
Original silica mass/lifetime sources, [10]'s gap interpretation, YAG
lifetimes and [3]'s supplement remain Phase 0 work. Missing parameters
must not be supplied from memory.

## Part II — Implementation record

No implementation phases have started.

## Part III — History

- 2026-09-27: Drafted from spatial/gas-plasma/backend and upstream-material
  audits and literature on silica/YAG filamentation and carrier kinetics.
  Identified the spatial-flux prerequisite, separate photo/transport masses,
  source-normalisation discrepancies and limits of immediate-heating
  avalanche. CPU scope is explicit; solid-guide, subcycle and delayed-heating
  extensions remain on the roadmap.
- 2026-09-29: Revised for SEP 3's decision D15. The solid photoionisation rate models
  move to optmat, which owns their normalisation, averaging declaration and
  validation. solprop tabulates them and keeps carrier kinetics, avalanche, currents
  and propagation. The adapters `SolidKeldyshRate` and `SolidMultiphotonRate` became
  `OptmatSolidRate`. Q1 became upstream work, and an optmat work package specifying
  the solid rate models, bundles and contract was made an acceptance requirement.
  That work package has not yet been drafted.
- 2026-10-22: Coordinated the shared scalar flux foundation with SEP 31. It owns
  uniform-medium launch, noise and observation changes; this proposal consumes
  them without a second normalisation policy or old-file compatibility layer.
  Solid rates, kinetics and their qualification remain this proposal's scope.

- 2026-10-23: Reassigned the shared medium-flux prerequisite to
  [SEP 34](archive/sep-0034-medium-flux.md) following SEP 31's review and split.
  SEP 31 retains field exchange and coupling; the shared flux proposal
  includes admitted radial adiabatic transport and complex local flux.
- 2026-10-23: Aligned the shared-flux description with SEP 34's review response:
  the real-index path certificate, forward-component local closure and deferral
  of varying absorption. Solid-ionisation scope is unchanged.
- 2026-11-01: Tracking issue [#116](https://github.com/LupoLab/solprop/issues/116).
