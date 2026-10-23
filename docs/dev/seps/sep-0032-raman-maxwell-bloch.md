# SEP 32 — Raman Maxwell–Bloch dynamics

| | |
|---|---|
| **SEP** | 32 |
| **Title** | Raman Maxwell–Bloch dynamics |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; GPT-6 (revision, 2026-10-22) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-10-22 |
| **Resolution** | — |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add a separately declared scalar Raman Maxwell–Bloch material response for stimulated
population transfer, saturation, differential AC Stark shifts and prepared molecular
coherence. Start with explicitly specified constant-polarisability two-level subsystems
on the ordinary scalar modal field route with resolved transverse quadrature. Recover
a matched causal convolution in the weak-excitation limit, qualify temporal material
error independently of z integration, and implement NumPy, Numba and Kokkos CPU in
that order. This proposal is extracted from SEP 1; its unresolved model questions do
not block [SEP 1](sep-0001-raman-response.md) or [SEP 10](sep-0010-low-rank-raman.md).

### 2. Motivation and scope

**Draft; not reviewed as a standalone proposal; implementation not started.**
Strong molecular excitation, Stark phase and explicitly prepared states are reasons
to use this model even if it is slower than fixed-population convolution. Optical
pump depletion and a Raman comb alone are not evidence that molecular populations
must evolve. The weak thermal ladder already describes rich coherent spectroscopy.

The initial target is a user-specified model and a separately labelled paper-based
H2 example after data reconciliation. It is not an automatic saturating replacement
for optmat's full thermal ladder. Keep scalar real-field drive, fixed microscopic
couplings and relaxation parameters, and a prescribed initial state per subsystem
at the first sample. Initially refuse envelope/vector fields, unsupported geometries,
shared-level ladders, non-resolved saturation and an ionisation/depletion combination
without a derived closure. The precise first ensemble choice remains Q1–Q2.

Shared populations, angular sublevels, general vector response, optical-frequency
coupling, Langevin noise, time-varying baths and accumulation outside one temporal
window are later physical models, not hidden implementation requirements.

### 3. Context and conventions

The [Raman manual](../../guide/media/raman.md) describes fixed-population kernels.
SEP 1 supplies exact evaluators and SEP 10 adds density-path compression; neither
supplies saturable microscopic parameters. The kernel's weak strength cannot fix
coupling, ensemble fraction and equilibrium inversion independently. The equations
below retain the derivations from the original SEP 1 draft and cite their sources;
their extraction is not a new scientific qualification.

SI units and [conventions](../conventions.md) apply: V/m local fields, rad/s transition
frequencies, s$^{-1}$ decay, m$^{-3}$ molecular density and F m$^2$ polarisabilities.
The density matrix is dimensionless. Physical fields come from the established
synthesis, not from interpreting modal power amplitudes as electric fields.

SEP 6's electronic response remains separate and must not filter this material's
drive or emitted source. SEP 11 supplies the common physical density and equilibrium
index. Its finite-density linear law does not, by itself, derive a local-field or
pair correction to the Bloch response. Equilibrium subtraction must be revisited
explicitly if such corrections are later added. Neither SEP is a licence to count
the equilibrium linear response twice.

### 4. Theory

#### Maxwell–Bloch equations and the equilibrium subtraction

Use $R=\rho_{12}=x+iy$ for the molecular coherence, reserving $\rho$ without
indices for molecular number density. Let $w=\rho_{22}-\rho_{11}$, with unit trace, and
let $\alpha=\alpha_{12}=\alpha_{21}$ be real. Belli et al. [1, p. 297, Eqs. (2)–(4)] give the following real-field equations

$$
 \dot R=(-\gamma_2+i\Omega)R
     +\frac{i}{2\hbar}\big[(\alpha_{11}-\alpha_{22})R+\alpha w\big]E^2,
$$

$$
 \dot w=-\gamma_1(w+1)-\frac{2\alpha}{\hbar}E^2\operatorname{Im}R,
 \qquad \gamma_1=T_1^{-1},\quad\gamma_2=T_2^{-1}.
$$

The relaxation target $w=-1$ assumes the lower state initially occupied.
An independently justified thermal or prepared-state model would replace it
by $w_{\rm eq}$ and supply the corresponding relaxation law.

The paper's polarisation expression includes the equilibrium contribution:

$$
 P=\rho\left[\alpha_{11}
       +(\alpha_{22}-\alpha_{11})\frac{1+w}{2}
       +2\alpha\operatorname{Re}R\right]E.
$$

Solprop already includes equilibrium linear susceptibility in the refractive
index and propagation operator. Its **nonlinear increment**, for equilibrium
coherence zero, should therefore be derived as

$$
 \boxed{\Delta P_R=\rho\left[
       \frac{\alpha_{22}-\alpha_{11}}2(w-w_{\rm eq})
       +2\alpha\operatorname{Re}R\right]E.}
$$

For independent effective transitions their contributions are summed with
explicit population/ensemble weights. Do not count the same equilibrium
polarisability once for vibration and again for rotation. The paper's $\alpha$
values are SI polarisabilities in F m²; this formula has **no extra $\epsilon_0$**.
Conversion to solprop's per-density kernel convention introduces the $\epsilon_0$
division below. The electronic Kerr contribution remains separately declared;
its source and frequency range must be consistent with the Raman model.

The envelope counterpart requires a stated carrier-averaged derivation. Under
the usual separation of optical and molecular timescales, the drive becomes
$\lvert A\rvert^2/2$ and the resulting slowly varying susceptibility multiplies $A$.
This is not an exact replacement for the real-field equations in the
subcycle regime explored by the paper. Start qualification on a scalar field
grid, then establish envelope parity in a controlled narrowband limit.

#### Maxwell–Bloch weak-excitation limit

Freeze $w=w_0$, neglect the differential Stark term, and impose $R(t_{\min})=0$.
For any real drive $d(t)$, the solution is

$$
 R(t)=\frac{i\alpha w_0}{2\hbar}
       \int_{t_{\min}}^t e^{(-\gamma_2+i\Omega)(t-s)}d(s)\,ds.
$$

Its coherent polarisation is consequently

$$
 2\rho\alpha E\operatorname{Re}R
 =\rho E\frac{-\alpha^2 w_0}{\hbar}
     \int_{t_{\min}}^t e^{-\gamma_2(t-s)}
                       \sin[\Omega(t-s)]d(s)\,ds.
$$

Matching $P_R=\epsilon_0\rho E(h*d)$ gives

$$
 \boxed{K=-\frac{\alpha^2w_0}{\epsilon_0\hbar}.}
$$

For a lower-state molecule, $w_0=-1$, the kernel amplitude is positive.
If only fraction $f$ of the molecules belongs to this effective transition,
the amplitude per total density becomes $-f\alpha^2w_0/(\epsilon_0\hbar)$. Degeneracies,
orientation averages and transition sharing must be handled consistently;
they cannot all be replaced by an unexplained fitted fraction.

This result has several practical consequences:

- The existing convolution already represents deterministic impulsive
  coherence, delayed pump–probe interaction, and transient or steady driven
  response within its material assumptions.
- It can also deplete the **optical pump**: the propagated optical field evolves
  self-consistently. Frozen **molecular populations** do not imply an
  undepleted optical pump.
- Stimulated Stokes/anti-Stokes generation and a Raman comb do not by themselves
  prove that population dynamics are required.
- Adding an ODE with $w$ fixed and the Stark term off adds a new numerical
  representation of the same line. That is the performance opportunity
  in SEP 1.
- Matching a weak line strength does not uniquely identify its strong-field
  model: $f$, $w_0$ and $\alpha$ enter as a product in the weak limit, but produce
  different saturation behaviour. Existing kernel amplitudes alone cannot
  manufacture a physically calibrated saturable model.

For an initially unexcited equilibrium medium, scaling the optical field by
a small factor $\varepsilon$ gives $R=O(\varepsilon^2)$ and stimulated
population changes $w-w_{\rm eq}=O(\varepsilon^4)$. The leading coherent
polarisation is therefore $O(\varepsilon^3)$. With zero diagonal difference,
population feedback into that coherence first changes the polarisation at
$O(\varepsilon^7)$; differential Stark shifts and the population-dependent
diagonal polarisation can introduce $O(\varepsilon^5)$ corrections. These
orders follow by substitution into the equations, at a fixed finite time
interval and with other data fixed. They provide more discriminating weak
limit tests than agreement at one low input amplitude. They do not apply
unchanged to an initially prepared nonzero coherence.

#### Additional Bloch physics, and its limits

| Effect | Present convolution | Full scalar two-level Bloch model |
|---|---|---|
| Causal coherence and ringing | Included | Included |
| Optical pump depletion in propagation | Possible | Possible |
| Stimulated molecular population transfer | Populations fixed | Included for the declared two states |
| Saturation / coherent population oscillations | Absent | Included within the two-level approximation |
| Differential AC Stark shift | Absent from the fixed kernel | Included through $\alpha_{11}-\alpha_{22}$ |
| Population-dependent refractive response | Absent | Included through $w-w_{\rm eq}$ |
| Arbitrary initial molecular coherence | Not represented by the standard zero-history kernel alone | Natural initial condition |
| Thermal rotational ladder and revivals | Existing gas ladder contains many lines | Lost if replaced by one effective transition |
| Strong rotational alignment and vector coupling | Weak tensor response only | Needs a resolved angular-state model, not independent x/y Bloch copies |
| Spontaneous Raman noise | Not supplied by the deterministic response | Still requires a noise model |
| Electronic resonance or ionisation physics | Separate models | Still separate; off-resonant elimination has a validity limit |

The paper makes a deliberate effective-model choice: decoupled rotational
S(1) and vibrational Q(1) contributions of ortho-hydrogen. Its success for that
experiment is not a general validation of this reduction for warm mixtures,
prepared spin populations, picosecond revivals or arbitrary polarisation.
Solprop's present rotational ladder is richer in weak-excitation spectroscopy
than a two-transition replacement, even though it lacks saturation.

Independent two-level equations also cease to be a consistent ladder model
when transitions share states. For example $J\to J+2$ and $J+2\to J+4$ must share
the population of $J+2$. Evolving two unrelated inversions can create mutually
inconsistent populations. A strong-excitation rotor needs shared populations,
the appropriate coherences and magnetic sublevels, with a sourced relaxation
operator. Vector excitation adds the required angular coupling. Summing
effective independent transitions is admissible only with a clearly stated
scope.

Even a single rotational branch contains magnetic sublevels with different
couplings. A scalar coupling chosen to reproduce an averaged weak strength
can reproduce a second moment of those couplings without reproducing their
individual strong-field rotations. Averaging saturating responses and
saturating an averaged response are different operations. The effective
orientation average in the paper should not be promoted to a general
strong-alignment model.

A vibrational two-level saturation model also needs care: a nearly harmonic
vibration supports further excited levels and is not generally a saturable
two-level system at high excitation. Adding Bloch dynamics does not remove
the need to justify the level truncation.

Likewise, propagating an optical noise seed through deterministic Bloch
equations does not by itself establish the correct spontaneous or thermal
Raman noise. A claim of that capability would require a consistent material
noise model and its relation to damping and equilibrium populations.

#### Determine need from molecular excitation, not pulse energy alone

A useful diagnostic is the fixed-population coherence $R_{\rm lin}(t)$ reconstructed
using matched microscopic couplings. If $\lvert R_{\rm lin}\rvert$ approaches the two-level
bound of 1/2, the linear response is certainly leaving its valid regime. For
an initially pure lower state without relaxation, conservation gives

$$
 4|R|^2+w^2=1,\qquad
 \rho_{22}\simeq|R|^2\quad\text{for small excitation}.
$$

Small populations do not automatically make Stark shifts irrelevant. Also
assess the differential shift

$$
 \delta\Omega(t)=\frac{\alpha_{11}-\alpha_{22}}{2\hbar}d(t)
$$

against the relevant detuning/linewidth and its accumulated phase over the
driven interval. These diagnostics depend on the **local synthesised field**,
its beating and duration, and the transition data, not just the launched
energy or the number of optical octaves generated. No numerical threshold is
adopted as a new solprop default in this SEP.

For the paper's pulses, $T_1$ is much longer than the driven temporal interval;
neglecting relaxation over that interval may be reasonable after checking it.
That does **not** justify freezing populations: stimulated transfer acts on
the coupling timescale, not on $T_1$.

#### Published effective hydrogen parameters and data reconciliation

The following are values printed by Belli et al. [1, p. 297], **not proposed
new registry defaults**:

| Effective H2 transition | $\alpha_{11}$ | $\alpha_{22}$ | $\alpha_{12}=\alpha_{21}$ |
|---|---:|---:|---:|
| Rotation S(1) | 8.9 | 9.4 | 0.85 |
| Vibration Q(1) | 8.9 | 9.73 | 1.55 |

Each entry is in units of $10^{-41}$ F m$^2$. The paper assumes
$T_1\simeq20$ ns and uses density-dependent coherence lifetimes, with density
ratio $\eta$ measured relative to a specified standard number density [1].
Its frequencies in Hz must be multiplied by $2\pi$ to obtain angular frequencies.
The original drafting record reports that its printed data were checked against the
paper's modelling page; the underlying calibration and a complete
experimental reconstruction have not been independently verified.

A sensitivity calculation can use the illustrative rotational input tuple
$(9.95,10.44,0.80)$ in the same units. **These numbers are arithmetic inputs
to this comparison, not cited material data, a calibrated alternative, or
another default.** Compared with
the paper's tuple, its diagonal difference is 0.49 instead of 0.50; its weak
coherent strength scales by $(0.80/0.85)^2\simeq0.886$ at fixed ensemble
weight. The common diagonal offset mostly concerns the equilibrium index,
whereas the diagonal difference changes Stark shifts and the
population-dependent increment. Thus similar-looking parameter tables cannot
be interchanged while claiming the same physical model. No solver tolerance
can correct an 11% coupling-strength change.

A linewidth formula in pressure must not be substituted for one in $\eta$:
their equivalence requires a reference temperature, pressure and equation of
state. Likewise distinguish coherence decay rate, linewidth FWHM in Hz, and
linewidth in rad/s. Solprop's current `Linewidth` convention relates a
Lorentzian FWHM in Hz to $\gamma_2=\pi\,\mathrm{FWHM}$, as documented in the
[Raman manual](../../guide/media/raman.md). Phase 0 must document conversions for
any chosen data set rather than inferring them from a symbol called pressure.

Current scalar strengths, rotational populations and linewidths do not supply
all diagonal polarisabilities, ensemble fractions or a population relaxation
model. Reproducing the paper's effective transitions and extending solprop's
thermal molecular registry are different objectives. The recommended initial
objective is a transparent user-specified two-level model with a separately
labelled paper-based example after data reconciliation, not a new saturating
replacement for the existing hydrogen factory. No new gas-data validity range
is asserted here.

### 5. Proposed interface

These are proposed research interfaces, not available API.

#### New Bloch response and state

Propose `RamanBlochState`, `RamanBlochTransition` and `RamanBlochResponse` in
the nonlinear domain, with immutable physical data. Proposed transition
fields are:

| Field | Meaning, units and permitted range |
|---|---|
| `label`, `subsystem` | Nonempty identifiers for provenance and an explicitly declared two-level subsystem |
| `omega` | Positive finite transition angular frequency, rad/s |
| `alpha12` | Real finite off-diagonal polarisability, F m$^2$; a sign convention is recorded and used consistently |
| `delta_alpha` | Real finite $\alpha_{22}-\alpha_{11}$, F m$^2$; zero disables the differential shift and population-dependent index together |
| `fraction` | Participating fraction per total molecular number density, dimensionless, in $[0,1]$ |
| `gamma1`, `gamma2` | Non-negative finite rates in s$^{-1}$, with $\gamma_2\geq\gamma_1/2$; initially constant in time and z |
| `equilibrium_inversion` | $w_{\rm eq}\in[-1,1]$, explicitly stated |
| `initial_state` | `RamanBlochState(coherence=R0, inversion=w0)`, finite and satisfying $4\lvert R_0\rvert^2+w_0^2\leq1$ |
| `citation` | Physical data source, reference state and validity statement; an uncalibrated research input must be labelled as such |

An omitted initial state means zero coherence and the declared equilibrium
inversion, not automatically $w=-1$ at every temperature. Only the diagonal
difference is needed in the nonlinear response after subtracting equilibrium;
an absolute diagonal polarisability must not be added again to the source.
Molecular density follows the existing response density convention; a
transition fraction is not applied a second time through the mode fill.

`RamanBlochResponse(transitions=..., ensemble_model=...)` requires a nonempty
transition sequence and an explicit ensemble interpretation. The proposed
choices are disjoint subensembles, whose participating fractions sum to at
most one, or independent internal factors (for example a deliberately
factorised rotational/vibrational effective model of the same molecules).
The latter does not require fractions to sum to one across factors, but its
factorisation and absence of shared-level dynamics must be explicit in
provenance and documentation. Repeated transitions sharing a populated level
within one subsystem are refused. Identifiers cannot prove a physical
factorisation; the supplied data's scientific interpretation remains part of
the model declaration. Q1 must settle the exact first-release ensemble
contract before acceptance.

The first response uses the resolved real scalar field drive $E^2$ and the
first-order equations of §4. Fixed populations and disabled Stark terms are
separate limits in validation, not coupled switches that silently redefine
the model. A public diagnostic switch for frozen populations is unnecessary
for the initial release: weak-drive convergence and a distinct fixed-kernel
response supply that comparison. If added later, its state and relaxation
semantics must be stated explicitly.

#### Temporal numerical controls and outputs

Propose `SimulationSettings.raman_bloch_substeps`, a positive integer required
when a Bloch response is used, and otherwise `None`. It specifies equal
material subintervals within each optical fine-grid interval; the second-order
symmetric rotation/relaxation candidate in §6 is the initial algorithm. This
explicit research control avoids inventing a universal accuracy default.
A deterministic drive interpolant is required; phase 1 must compare linear
interpolation of the quadratic drive against a sufficiently resolved local
field reference. Substepping converges the interpolated drive problem but
cannot remove an optical-grid interpolation error. The initial state is at
the first fine-grid sample, and source values are emitted at those samples.
There is no invented interval before the first sample.

```text
state = RamanBlochState(coherence=0, inversion=-1)
line = RamanBlochTransition(label="declared line", subsystem="two levels",
                           omega=omega_SI, alpha12=alpha_SI,
                           delta_alpha=diagonal_difference_SI,
                           fraction=participating_fraction,
                           gamma1=rate1_SI, gamma2=rate2_SI,
                           equilibrium_inversion=-1, initial_state=state,
                           citation=source_and_validity)
raman = RamanBlochResponse(transitions=[line], ensemble_model="disjoint")
settings = SimulationSettings(..., backend="numpy_ref",
                              quadrature=resolved_transverse_rule,
                              raman_bloch_substeps=chosen_positive_integer)
simulation = Simulation(real_field_grid, scalar_modes,
                        responses=[electronic_kerr, raman], initial=pulse,
                        settings=settings)
```

The sketch does not supply invented calibrated constants or numerical
accuracy claims. A manual example must replace every placeholder and verify
its time-grid and substep convergence before publication.

Responses remain usable in ordinary simulations and parameter scans. Batch
admission keeps the existing common-structure/fixed-schedule rules; initial
states and coefficients are immutable per problem/member and temporal scratch
is member-owned. Record model identity, transition data and sources, ensemble
interpretation, initial state, equilibrium state, evaluator, taper/quadrature
policy, numerical controls and resolved backend in provenance. Material state
is recomputed from each local field history, not checkpointed as a propagated
z state. Resume compatibility must include physical data, initial-state and
material discretisation choices; pure evaluator metadata follows the current
execution-metadata convention once parity is established.

Initially, bounded local-material diagnostics and tests should expose maximum
excitation, maximum Bloch-ball residual and the nonlinear polarisation needed
for qualification. A permanent saved per-node/per-time density-matrix output
is not required and would reverse much of the memory benefit. Q6 assigns
compact optional diagnostics and their output layout to phase 1. If they
enter `StatsSpec` or saved arrays, they require normal units, shapes, schemas
and processing support. They must not be hidden debug-only physics outputs.

Refuse envelope/vector excitation, shared-level ladders, unsupported optical
routes, a non-resolved saturating mode average, unsupported bath profiles,
invalid states/rates, and an ionisation/depletion combination lacking a
derived closure. Explain the supported route and backend in the error.
Prepared initial coherence that varies with z or across transverse nodes is
future work; the first record supplies one explicit state per subsystem.

### 6. Implementation strategies

Physics owns levels, SI couplings, ensemble/initial-state semantics and the bath.
Translation compiles frozen arrays and numerical controls. Backend bodies own the
preallocated temporal sweep, status checks and source contraction. Driver/output
own provenance and any explicitly specified compact diagnostics. No new dependency
is needed; SciPy provides independent matrix-exponential references.

This is a distinct response leaf, not a `raman_evaluator` value. SEP 1's recurrence
setting and SEP 10's compression tolerance do not approximate this state evolution.
Reject duplicate declared subsystems represented by both Bloch and fixed kernels;
scientific overlap that cannot be inferred from identifiers remains a documented
model responsibility. The ordinary mode-average cubic calibration is insufficient
for saturation: use resolved local fields and qualify transverse quadrature.

#### The temporal state belongs inside each nonlinear evaluation

At a given propagation stage, solprop knows the entire local optical field
history. Integrate the molecular equations forward in physical/retarded time
at each physical node to obtain that stage's polarisation, then project it
back onto the modes. The existing UPPE interaction-picture z stepper can stay.

The initial molecular state is prescribed at $t_{\min}$ for that **location**.
It must be reinstated for every evaluation of a different optical stage field,
including rejected z trials. Carrying the final coherence from the previous
z stage into the next would confuse spatial propagation with temporal memory.
Coefficients may be cached; a whole valid stage evaluation can participate in
the existing FSAL reuse; molecular end states are not general stage caches.

Two pulses in one window naturally share a molecular history. Pulses outside
the simulated window, a pre-excited medium, repetition-rate accumulation, or
residual coherence varying along z need explicit initial-state data or a
separate multi-pulse model. A nanosecond $T_2$ does not require simulating a
nanosecond window if no observable samples those late delays and the chosen
initial state is valid. This is equally true of the present causal convolution.

#### Given the field, the full Bloch update is a small linear system

Although the resulting optical response is nonlinear, the density-matrix
equations are linear and affine **in the material state for a prescribed
stage field**. Put $X=2\operatorname{Re}R$, $Y=2\operatorname{Im}R$,

$$
 \omega_e(t)=\Omega+\frac{\alpha_{11}-\alpha_{22}}{2\hbar}d(t),
 \qquad g(t)=\frac{\alpha}{\hbar}d(t).
$$

Then

$$
 \frac{d}{dt}\begin{bmatrix}X\\Y\\w\end{bmatrix}
 =\begin{bmatrix}
 -\gamma_2&-\omega_e&0\\
 \omega_e&-\gamma_2&g\\
 0&-g&-\gamma_1
 \end{bmatrix}
 \begin{bmatrix}X\\Y\\w\end{bmatrix}
 +\begin{bmatrix}0\\0\\\gamma_1w_{\rm eq}\end{bmatrix}.
$$

Without relaxation this is a rotation of the Bloch vector, with angular vector
$(-g,0,\omega_e)$. For a constant drive over one interval the rotation can be
evaluated analytically. It preserves $X^2+Y^2+w^2$ and avoids resolving the free
oscillation solely for the stability of a generic explicit method.

A useful first candidate is a midpoint-driven rotation with exact half
relaxations on either side:

- free relaxation maps $X,Y$ by $\exp(-\gamma_2 h)$ and maps
  $w$ to $w_{\rm eq}+(w-w_{\rm eq})\exp(-\gamma_1 h)$;
- the coherent part is an exact constant-Hamiltonian rotation;
- a symmetric composition is second order for sufficiently smooth input.

For a physical two-level relaxation model require $\gamma_2\geq\gamma_1/2$, with the excess
interpreted as pure dephasing, and a valid equilibrium population. Forward
relaxation maps and unitary rotations then preserve admissibility. A higher
order method involving negative dissipative substeps cannot automatically
claim the same property. Alternatively a small affine exponential or a
commutator-free/Magnus method can be investigated, with its own error and
positivity analysis. Exponential integrators for time-dependent Hamiltonians
provide methodological background; this does not select a specific scheme.
[Hochbruck and Lubich](https://doi.org/10.1137/S0036142902403875),
[Blanes et al.](https://arxiv.org/abs/0810.5488)

Matrices at different drive values generally do not commute. One exponential
of the pulse-integrated intensity is not the general solution. Exact rotation
on a subinterval still leaves error from the time dependence/interpolation of
the drive and from relaxation splitting. Very strong fields or subcycle
structure may require smaller substeps.

The same affine-map composition idea as the scalar recurrence could parallelise material time
updates, now with small matrices instead of scalars. It is more expensive and
requires a rounding bound for the arithmetic tree. CPU node-parallel sweeps are the sensible
first target; GPU competitiveness is a separate measurement.

An independent Hamiltonian reference follows directly from the state
definitions. Write the molecular density matrix as $\varrho$, to distinguish
it from number density $\rho$:

$$
 \varrho=\begin{bmatrix}(1-w)/2&R\\R^*&(1+w)/2\end{bmatrix},
 \qquad
 H=\frac{\hbar}{2}
     \begin{bmatrix}-\omega_e&-g\\-g&\omega_e\end{bmatrix}.
$$

The commutator equation $\dot\varrho=-i[H,\varrho]/\hbar$ reproduces the
coherent terms above. For constant drive, computing
$U=\exp(-iHh/\hbar)$ and $U\varrho U^\dagger$ provides an oracle independent
of a three-component rotation implementation. An identity term in $H$ has
been removed because it does not change the density matrix.

The relaxation rates also have a concrete physical interpretation: upward
and downward population rates are respectively
$\gamma_\uparrow=\gamma_1(1+w_{\rm eq})/2$ and
$\gamma_\downarrow=\gamma_1(1-w_{\rm eq})/2$, with pure dephasing
$\gamma_\phi=\gamma_2-\gamma_1/2$. Non-negativity of all three rates explains
the stated admissibility conditions. This is a selected Markovian two-level
bath model; it does not establish a measured molecular collision law.

#### Internal error must not be hidden from qualification

The z stepper's embedded estimate measures its z integration error for the
source it receives. A biased or under-resolved temporal material solver can
produce a smooth wrong source that the embedded pair never diagnoses.
Temporal material error must therefore be bounded/tested separately, and any
internal adaptive error settings recorded in provenance. “The z tolerance
was tightened” is not adequate qualification of a Bloch time solver.

Do not silently clip $w$ to $[-1,1]$ or rescale coherence onto the Bloch ball
after an inaccurate step. Such repairs can hide instability and change the
response. Use a physical update, check finite state and physical bounds with
derived rounding allowances, and report failure through the backend's normal
status path.

#### Cost and backend contract

For $Q$ physical histories, $M$ two-level subsystems and $N$ samples, with $s$
material substeps per interval, the sweep costs $O(QMNs)$ and streams $O(QM)$
state beside optical fields and declared diagnostics. Measure interpolation,
rotation/relaxation, projection and complete optical stages. A strongly excited
source may also require more optical z steps. Comparison with a fixed kernel is
an evaluator comparison only in the matched weak limit at matched source accuracy.

CPU node/batch parallelism comes first; matrix affine-map scans on GPUs are a later
measurement with a derived rounding allowance. No performance claim is made here.
Every buffer belongs in memory admission and the ledger. No allocation, logging,
Python objects or exceptions enter compiled loops; use normal backend status codes.
Shared Numba bodies run interpreted in the coverage lane.

The response arrays, initial states, substep controls, evaluated density and optional
diagnostics require an explicit seam extension and a bump of the then-current
schema with frozen literals and coordinated native consumer checks. Unsupported
backends/routes refuse by name before allocation. Native implementation follows
the repository layout then in force, including any SEP 29 integration.

### 7. Verification and validation

#### Bloch model: stronger requirements than matching one spectrum

| Test | What it establishes |
|---|---|
| Zero field with arbitrary valid initial state | Analytic free precession, $T_1$ recovery and $T_2$ decay |
| Lossless constant drive | Agreement with an independent two-level Hamiltonian exponential; Bloch-length conservation |
| Valid dissipative evolution | Unit trace, Hermiticity, non-negative eigenvalues; $4\lvert R\rvert^2+w^2\leq1$ within rounding error |
| Matched weak-field limit | The kernel coefficient $-f\alpha^2w_0/(\epsilon_0\hbar)$, Stokes sign and field/envelope factors |
| Stark shift enabled/disabled | Phase shift with the correct sign and magnitude, separately from population transfer |
| Nonlinear polarisation at equilibrium | Exactly zero increment: no equilibrium index counted twice |
| Two delayed pulses | Correct surviving coherence and population, with no circular or inter-stage history leakage |
| Time-grid/substep refinement | Material-solver error independent of z-step error |
| Transverse quadrature refinement | Correct averaging of a non-cubic, saturating response |
| Short coupled propagation | Agreement among backends without demanding identical adaptive trajectories beyond the established numerical contract |

Use a separately constructed Hamiltonian/density-matrix evolution as an oracle,
not the same three component equations in another loop. Include thermal or
mixed initial states if supported, and explicitly test rejection of invalid
rates and nonphysical states. The no-relaxation invariant above is a direct
test of the frequency-dependent variant if it claims the same state meaning.

Conservation claims need a specified model. Optical energy can transfer into
molecular excitation even without phenomenological damping. With damping it
also leaves to the bath. A lossless restricted model can test a derived
field-plus-matter balance; the existing restricted Raman photon-number checks
remain useful where their assumptions hold. Do not demand optical energy
conservation for arbitrary Raman propagation or silently ignore the molecular
reservoir when testing a new model.

An independently obtained propagation result can be a regression witness when
its complete inputs, units, strengths and switches are known. It is not an
independent truth oracle merely because it came from another solver. Agreement with the paper's final supercontinuum alone cannot isolate
whether population evolution, the effective lines, modal propagation or a
different intensity calibration caused a difference. Start with local material
tests and short propagation windows, not a full reproduction run.

#### Separate accuracy budgets

Use at least three substep refinements and three optical-grid refinements; for the
chosen midpoint rotation/relaxation method expect second order for smooth resolved
forcing before roundoff dominates. Compare constant-drive evolution with an
independently constructed density-matrix Hamiltonian exponential. Use weak-field
amplitude scaling to distinguish cubic response, fifth-order Stark/diagonal terms
and seventh-order population feedback in the stated limits of §4.

Do not compare different quadratures on one grid as though they should agree at
roundoff. The weak convolution reference is the matched continuous model converged
in time; SEP 1's exact sampled implementation provides a distinct discrete reference.
The lag taper of the existing kernel is a finite-window policy, not a term in these
Bloch equations. For weak-limit comparisons retain only delays before that taper
or quantify its contribution separately. Do not introduce a lag taper into the
material state silently to match one finite-window FFT.

Test valid mixed/prepared states, tensor/route refusals, duplicate subsystems,
stage resets, rejected trials and valid whole-source reuse. Require a source-error
budget independently of the optical stepper; Q4 assigns its quantitative allocation.
Matched fixed histories use derived rounding/integration bounds, adaptive runs a
case-specific accuracy envelope against a tighter reference. Never clip states onto
the Bloch ball to pass a test.

Performance and coupled qualification use about ten optical steps or a fibre short
enough to finish in seconds, with the measured z interval and actual work reported.
A full experimental reproduction is optional worked-example research, not a gate.

### 8. Documentation

Add `docs/guide/media/raman-maxwell-bloch.md`, linked from the Raman workflow and
media navigation. Cover equations/units, equilibrium subtraction, ensemble and data
identities, initial conditions, bath validity, time/substep/z/transverse convergence,
source diagnostics and unsupported combinations. Add full public docstrings and a
short focused tutorial after the model/data gate. A calibrated paper preset is not
advertised until its microscopic inputs and interpretation are reconciled.
Update kernel/backend documentation and the changelog with actual implementation.

### 9. Compatibility and impact

The response is additive and opt-in. Existing fixed-population factories and FFT
defaults are unchanged. Prepared initial states are re-established for every local
temporal history; they are not carried along z or checkpointed as an extra optical
state. Model data, ensemble interpretation, initial/equilibrium states, interpolation
and substeps enter provenance/resume identity. Old optical files remain readable;
new diagnostics need documented units/shapes and any required output schema bump.
Changed data or material discretisation cannot silently resume the same source.

### 10. Alternatives and future extensions

#### First-order state with generic nested ODE integration

The state $(\operatorname{Re}R,\operatorname{Im}R,w)$ is the recommended
physical representation, but it does not imply a general adaptive ODE solve
with spline queries and heap storage for every transition. Such an approach
is useful as an independent reference, and may cope conveniently with very
irregular forcing. In a production source stage it adds interpolation,
controller and allocation overhead and a second tolerance hierarchy inside
each z trial. It can store three length-$N$ histories unnecessarily when the
polarisation can be emitted during a sweep.

Use seconds throughout. If a derivation uses femtoseconds as its independent
variable, $dy/dt_{\rm fs}=10^{-15}dy/dt_{\rm s}$; copying its derivative scale
into a seconds-based solver would be wrong by $10^{15}$. Tolerances on $R$,
$w$ and a derivative state also need their own units/scales. Freezing $w$ and
turning off $\alpha_{11}-\alpha_{22}$ are independent choices: one does not
imply the other. Finally, reconstruct only the nonlinear increment in §4;
adding $\alpha_{11}E$ again duplicates the equilibrium refractive index.

#### Second-order polarisation form

Set the Stark term to zero and let $S=2\alpha\operatorname{Re}R$, the coherent
change in polarisability per molecule. Eliminating $\operatorname{Im}R$ from
the first-order equations, for constant $\alpha$, $\Omega$ and $\gamma_2$, gives

$$
 \ddot S+2\gamma_2\dot S+(\Omega^2+\gamma_2^2)S
       =-\frac{\alpha^2\Omega}{\hbar}d\,w,
 \qquad
 \dot w=-\gamma_1(w-w_{\rm eq})
       +\frac{d}{\hbar\Omega}(\dot S+\gamma_2S).
$$

This follows from $\dot{\operatorname{Re}R}=-\gamma_2\operatorname{Re}R
-\Omega\operatorname{Im}R$. The state is $(S,\dot S,w)$, not $(R,w)$, and the
coherent polarisation is $\rho E S$ with any ensemble weight. Treating $S$ as
$\operatorname{Re}R$ again would multiply the coupling twice.

Dropping $\gamma_2^2$ is a weak-damping approximation, not an exact rewrite.
It shifts the free oscillation frequency from $\Omega$ to
$\sqrt{\Omega^2-\gamma_2^2}$ in the underdamped case. Even a small relative
shift can accumulate phase over a long coherence. A time-dependent
Stark-shifted frequency introduces further terms from its derivative; merely
substituting an instantaneous $\Omega$ does not recover the full first-order
model. There is no automatic speed gain: the evolving-population problem
still has three real variables, and the derivative variable can be less well
scaled. Retain this as an analytical cross-check, not the selected production
state.

#### Frequency-dependent Raman coupling

Consider a candidate filtered optical field
$E_s=\mathcal F^{-1}[\alpha(\omega)\mathcal F(E)]$, complex forcing $F=EE_s$,
and emitted source $2\rho\operatorname{Re}(E_sR)$ with explicit ensemble
weight. This requires an optical-field FFT and transition-specific inverse
transforms/interpolants. It does not eliminate all FFT cost. If
$\alpha(\omega)=\alpha$ is a real constant, $E_s=\alpha E$ and the candidate
must recover the constant-coupling model with zero diagonal difference.

For complex $F$, the apparently natural pair

$$
 \dot R=i\Omega R+\frac{iwF}{2\hbar},\qquad
 \dot w=-\frac{2}{\hbar}\operatorname{Im}(RF)
$$

fails a necessary lossless density-matrix invariant:

$$
 \frac{d}{dt}(4|R|^2+w^2)
 =-\frac{8w}{\hbar}\operatorname{Re}R\operatorname{Im}F.
$$

It agrees with ordinary Bloch length only when the right-hand side vanishes,
including real constant coupling. A real-valued but asymmetric
positive/negative-frequency coupling array can produce a complex $E_s$;
using a Raman-shifted optical argument on one frequency side is one way this
can arise. Thus the issue is not avoided merely by tabulating real numbers.

For a Hermitian Hamiltonian with the displayed coherence equation, the
population coupling involves the corresponding conjugate. However, changing
one conjugation is not a full derivation of a broadband material model:
the driving kernel and emitted polarisation must be derived together, with
causality, Hermiticity, reality and the energy balance checked. A response
may depend on both participating optical frequencies
$\alpha(\omega_{\rm in},\omega_{\rm out})$, and a single frequency filter is
an additional factorisation. Filtering before and after multiplication by
coherence are not equivalent. Its validity away from electronic resonances
must be stated. Reference [1] establishes a constant-polarisability model,
not this frequency-dependent candidate. This is a separate research proposal,
with no accepted implementation or qualification claim here.

The **Broadband Raman coupling** roadmap item is independent of population
dynamics. A separable transition kernel might share filtered-field machinery with
a weak response and a state model, but the drive and emission must be derived
together. SEP 6's electronic fit is not transition-polarisability data. Retain this
candidate and invariant counterexample as constraints, not an endorsed model.

### 11. Phases

All phases are proposed. Use `feature/sep-0032-raman-maxwell-bloch-<phase>` branches.

| Phase | Deliverables and required evidence |
|---|---|
| 0 — Model and data gate | Resolve Q1–Q3; reconcile the declared example's ensemble, spin/orientation weights, density reference, microscopic constants and bath validity; no inferred saturation data from kernel strength alone |
| 1 — NumPy | Response/state records, resolved scalar modal field route, structure-preserving sweep and explicit controls, schema/native refusals, manual/tutorial and agreed diagnostics; local Hamiltonian/weak/Stark/positivity and separate convergence checks |
| 2 — Numba CPU | Shared no-allocation bodies, stage integration, batching/status/ledger; source and short-run parity, state-reset/rejected-step checks and interpreted coverage |
| 3 — Kokkos CPU | Matching native response and diagnostic reductions; local/coupled parity, memory, errors and measured short costs |
| 4 — Qualification and closure | Consolidate accuracy, provenance/resume and source audit; finish CPU admission, publish limits and name later device work |

Each phase appends its actual report under Part II. During slices run the repository's
fast checks and directly affected tests. At the end, run the full gate, slow lanes
and coverage lane once on the final tree or take them from CI; rerun failed modules
rather than every successful lane. No full-length propagation
is a gate or timing requirement. Final requires all three CPU backends or an
exception explicitly accepted into this SEP, not an assumed omission.

### 12. Open questions

These are physics/numerical design questions for this Draft, not prerequisites for
accepting SEPs 1 and 10. Human and AI review of this standalone Part I remain needed.

| ID | Question and recommended direction | Destination |
|---|---|---|
| Q1 | Disjoint subensembles only, or also explicitly factorised independent internal subsystems? Never independent populations for shared levels. | Before acceptance; phase 0 records the resolved data interpretation |
| Q2 | Confirm user-specified two-level physics and a reconciled paper H2 example as the first target, with resolved scalar modal field admission. A saturated thermal ladder is a different model. | Before acceptance |
| Q3 | Confirm explicit positive substeps and the symmetric second-order method, or design an adaptive material controller first. | Before acceptance |
| Q4 | Select and qualify the drive interpolant and source-error allocation independently of substep, optical-grid and z error. | Phase 1, rechecked in phases 2–3 |
| Q5 | Reconcile example polarisabilities, ensemble/spin weights, relaxation and reference states with primary data; distinguish calibrated data from user research inputs. | Phase 0; no preset until resolved |
| Q6 | Specify compact population/coherence diagnostics, units/reductions and output/provenance semantics without storing every temporal state. | Phase 1 |
| Q7 | Qualify device execution and decide priorities among shared ladders, vector coupling, noise, depletion and repeated-pulse histories. | Phase 4 assigns named follow-ups; no implicit delivery here |

### 13. References

1. F. Belli, A. Abdolvand, W. Chang, J. C. Travers and P. St. J. Russell,
   "Vacuum-ultraviolet to infrared supercontinuum in hydrogen-filled photonic
   crystal fiber," *Optica* **2**, 292–300 (2015).
   [doi:10.1364/OPTICA.2.000292](https://doi.org/10.1364/OPTICA.2.000292).
2. M. Hochbruck and C. Lubich, "On Magnus integrators for time-dependent
   Schrödinger equations," *SIAM Journal on Numerical Analysis* **41**, 945–963 (2003).
   [doi:10.1137/S0036142902403875](https://doi.org/10.1137/S0036142902403875).
3. S. Blanes, F. Casas, J. A. Oteo and J. Ros, "The Magnus expansion and some
   of its applications," *Physics Reports* **470**, 151–238 (2009).
   [doi:10.1016/j.physrep.2008.11.001](https://doi.org/10.1016/j.physrep.2008.11.001);
   [author preprint](https://arxiv.org/abs/0810.5488).

The original SEP 1 drafting record reports inspection of [1], p. 297, Eqs. (2)–(4)
and its printed effective parameters. That historical check is retained, not claimed
as a new inspection during extraction. The underlying calibrations and complete
experiment have not been independently re-qualified. References [2–3] provide
methodological background, not an order/positivity certificate for this dissipative
update. The weak coefficient, state Hamiltonian, second-order elimination and
complex-drive invariant are re-derived in this document. No new parameter default
is introduced.

## Part II — Implementation record

No implementation phases have started.

## Part III — History

- 2026-10-22: Extracted from the 2026-09-26 SEP 1 draft at John Travers's request.
  Retained the microscopic theory, proposed interfaces, numerical method, validation,
  data caveats and open model decisions. Renumbered phases/questions and corrected
  the planned manual location. Fixed-response evaluator work remains SEP 1; density
  compression remains SEP 10. Extraction does not constitute standalone acceptance
  or a completed standalone review.
