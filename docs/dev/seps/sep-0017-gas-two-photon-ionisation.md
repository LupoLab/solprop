# SEP 17 — Broadband two-photon continuum response beside strong-field ionisation

| | |
|---|---|
| **SEP** | 17 |
| **Title** | Broadband two-photon continuum response beside strong-field ionisation |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-08); GPT-6 (review revisions, 2026-10-08; substantial revision, 2026-10-12) |
| **Status** | Accepted |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI); GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-08 |
| **Resolution** | Re-accepted 2026-10-12 by John Travers after the independent review and the three follow-up corrections recorded in Part III. Phase 0 qualification remains required before production implementation in Phase 1. The earlier acceptance and return to Draft remain in the history. |
| **Requires** | [SEP 3](archive/sep-0003-optmat-materials.md) (material data); [SEP 6](archive/sep-0006-dispersive-nonlinearity.md) (electronic response composition); optmat OEP 6's implemented continuum data |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

**Revision and review scope.** This is the revised design of 2026-10-12. The header's
review status and reviewers refer to this Part I, reviewed on 2026-10-12 (Part III).
The earlier human and AI reviews, acceptance and numerical checks remain in Part III as
history; they do not qualify the new continuum response. Implementation has not begun.

### 1. Abstract

Propose an opt-in, scalar, field-grid response for direct two-VUV-photon ionisation in
Ar and He. A positive distribution of effective continuum states is driven by the
squared analytic VUV field. Its weights come from optmat's existing cross sections.
The same interaction determines a retarded atomic response, its polarisation and its
energy exchange, rather than assigning a window-calibrated local rate and a separate
loss current. The first deliverable is prescribed-field diagnostics and propagation
with this continuum optical response, including its dispersive part, with final
ionisation-yield diagnostics. It does not identify transient continuum occupation with
a free-electron density. Additional plasma feedback is a separate qualification gate,
not silently omitted from a supposedly complete two-photon model. Success means
reproducing OEP 6's integrated weak-field limits, conserving the declared energy,
converging the continuum and source projection, and establishing an atomic and material
validity domain relevant to the RDW test. No new optmat quantity is required for this
first approximation; section 4.9 records the material gaps and the triggers for a
coupled OEP. The continuum approximation and its dispersive matching require review
before production implementation.

### 2. Motivation and scope

The maintainer's question is whether two-photon absorption of a VUV resonant dispersive
wave (RDW) can explain its suppression near the two-photon ionisation onset in a gas-filled
capillary driven by a self-compressing 800 nm pump. Pump-driven ionisation can blue-shift
the pump and detune RDW generation. That mechanism and attenuation after generation can
coexist. Distinguish departure from a phase-matching curve from the shortest observable
RDW across optimised conditions; report the centroid, blue edge and detection limit
separately. The input duration is a dataset parameter, not fixed by this proposal; use
the generated local VUV field, not the input pump duration, for atomic tests.

The sequence is: replay existing fields to estimate loss and final yield; add the
continuum optical response if those estimates could matter; add extra plasma feedback
only if its effect cannot be bounded as negligible and its timing has been qualified.
A small excitation probability per atom can still produce appreciable propagation loss
through many atoms. Conversely, small direct loss does not prove negligible plasma
feedback. A substantial prescribed-field correction requires a coupled propagation
calculation; it is not itself a prediction of the attenuated output.

The existing strong-field rate is evaluated at a declared pump carrier. Applied to the
total field, VUV oscillations can change its nonlinear average without representing
absorption of VUV photon pairs. Removing that content from the rate and adding a
continuum response are separate changes: both need controlled comparisons. Real
pump-assisted channels and dressing are also separate physics, not recovered by the
split. The immediate objective is the Ar/He RDW experiment; four-wave-mixing sources
and broader gas propagation remain possible later applications.

**First scope:** pure Ar or He, fixed linear polarisation, field grids and the
single-mode capillary route with a resolved transverse field quadrature. Apply the
atom at physical field nodes and project back to the mode; a peak-power or mean-field
substitution does not reproduce a fourth-power yield. The first-scope response is,
however, homogeneous of degree three in the local field with linear selectors and a
fixed neutral profile. For a frequency-independent mode shape $F(r)$ it therefore
factorises exactly: the node response is $F^3$ times one modal response, and its
projection is the Kerr overlap $\int N(r)F^4\,dA$ (section 4.4). The radial rule is
still needed for the pump plasma, and for any later saturating closure. The electronic
excitation is perturbative at each atom, although the propagated VUV may change
substantially. A pump-ionisation baseline is allowed only when its neutral depletion is
negligible over the relevant VUV interaction or is a fixed initial neutral fraction for
that interaction. A varying depletion cannot be inserted into an integrated diagnostic
or this stationary-coefficient response without a new derivation.

**Deferred:** additional two-photon plasma feedback until section 4.7's gate;
strong depletion or saturation; mixed pump–VUV amplitudes and pump dressing; bound
resonances and their (2+1) ionisation; mixtures, other gases, multiple optical modes,
envelope grids, vector fields and other geometries. OEP 8's material coefficients can
support a separate bound-response SEP, particularly the He 1s2s feature. They are not a
dependency of the direct continuum model. Atomic TDSE calculations are validators, not
required inside every optical right-hand-side evaluation.

### 3. Context

The earlier version of this proposal was accepted with Q3, Q4, Q13 and Q18 as
qualification blockers. Its window-wide rate calibration reproduced integrated yield
and energy but did not establish ionisation timing or the spectral work of its current.
This revision replaces that candidate with an explicit weak-excitation continuum and
narrows the first delivery to Ar/He optical response and final-yield diagnostics.
Shared depletion and additional VUV plasma are conditional follow-up work, not promised
by the initial optical model. The old question numbers are retained in section 12,
with their changed resolutions and scope made explicit. There is one implementation
track and no separate rate-based two-photon response to enable beside it.

The existing [plasma response](../../guide/media/field-plasma.md#the-plasma-response) supplies
pump-ionisation and free-electron currents. [SEP 18](archive/sep-0018-band-exact-plasma-integrals.md)
provides band-exact running integrals, but does not supply a physical ionisation-time
history. The proposed optical continuum supplies a polarisation directly; it does not
need to pass through the plasma current's running integral. Q18 still requires the
correct source factor and discrete work. References in the archived SEP 18 to this
proposal's former section 4.9 describe the earlier design; the current treatment is
in sections 4.3 and 4.6 and Q18.

[SEP 12](archive/sep-0012-tdse-nonlinear-response.md) provides a TDSE comparison
route; its native response must be used without replacing its absorptive cubic by a
real Kerr term. Its calibrated material and numerical limitations remain separate from
this SEP's model error. For structural atomic comparisons use native scale one and
`source_filter="none"`: rescaling optical current does not rescale microscopic
ionisation yield. Use an atomic model's own cross section to test the response's
structure, and compare that cross section with optmat separately. The in-tree helium
TDSE route is not an argon validator.

optmat [OEP 6][oep6] supplies `two_photon_ionisation_model`, `cross_section`,
`pair_cross_section`, thresholds, region masks, uncertainties and identities.
[OEP 8][oep8] and its review explicitly leave propagation polarisation, electron
accounting and energy balance in solprop. The optmat adapter and material identities
follow [SEP 3](archive/sep-0003-optmat-materials.md); no cross-section table is copied
into solprop. These APIs must be present in the optmat version pinned for implementation;
the current optmat floor is not evidence that all of them are installed.

### 4. Theory

#### 4.1 Fields, units and the known integrated constraint

Use the [optics convention](../conventions.md),
$\widetilde A(\omega)=\int A(t)e^{i\omega t}dt$. At one physical node let
$A=E_{V,a}$ be the analytic VUV field in V/m, with real field $E_V=\operatorname{Re}A$;
optmat's positive-frequency field is $E_V^+=A/2$. Let $N_g$ be the initially available
neutral density in m⁻³, fixed during the atomic calculation. Local-field corrections are
not introduced. The cross-section convention uses vacuum-equivalent local-field
intensity $I_0=\epsilon_0c|A|^2/2$, not modal power or the dispersive Poynting flux.
The latter is still used for the propagated field's physical energy.

In the pair-sum approximation of OEP 6, define

$$
S(\Omega)=\int A(t)^2e^{i\Omega t}dt,\qquad
k(\Omega)=\frac{\epsilon_0^2c^2\sigma^{(2)}(\Omega/2)}{\hbar^2\Omega^2}.
$$

The two-photon matrix element is assumed independent of frequency split at fixed
$\Omega$. Its validity needs tests of the actual single-photon support and splits;
a supported mean photon energy alone is insufficient. Reference [5] supplies an
amplitude-calculation methodology, but its two-photon calculation uses equal photon
energies, with intermediate and final energies $E_0+\hbar\omega$ and
$E_0+2\hbar\omega$. It does not validate unequal-photon-energy pairs at fixed sum;
that requires separate nondegenerate calculations or independent pulse comparisons.
The total yield and energy per initially neutral atom are

$$
P_\infty=\frac{1}{2\pi}\int k(\Omega)|S(\Omega)|^2d\Omega,\qquad
U_\infty=\frac{1}{2\pi}\int\hbar\Omega k(\Omega)|S(\Omega)|^2d\Omega.
\tag{1}
$$

All continuum integrals are over positive angular frequencies above the appropriate
threshold. $\sigma^{(2)}$ has units m⁴ s; $k$ has units m⁴ V⁻⁴ s⁻¹, and
$S$ has units V² m⁻² s. Thus $P_\infty$ is dimensionless and $U_\infty$ is in J.
For a long monochromatic pulse, (1) gives $P_\infty/T=\sigma^{(2)}(I_0/\hbar\omega)^2$.
The pulse energy removed per length is the transverse integral of $N_gU_\infty$,
in J/m. Equation (1) is not a saturated probability law.

#### 4.2 The minimal effective continuum

Choose a declared finite continuum domain $\mathcal D$ on which material data or
explicitly labelled completion models are available. With positive quadrature weights
$w_j$ in s⁻¹ and nodes $\Omega_j$, set

$$
g_j=\sqrt{\frac{w_j k(\Omega_j)}{2\pi}},\qquad
\dot b_j=-i\Omega_j b_j+i g_j A^2,\qquad b_j(t_0)=0.
\tag{2}
$$

Here $b_j$ is dimensionless and $g_j$ has units m² V⁻² s⁻¹. The ground amplitude is
fixed at one: these are weak-excitation auxiliary equations, not a saturation model.
One effective channel per energy suffices for the *declared separable approximation*.
Its scalar coupling phase can be absorbed into $b_j$; no measured scattering phase is
being inferred. This simplification does not recover channel interference between
different frequency splits, angular distributions or electron trajectories.

Solving (2) gives

$$
b_j(t)=i g_j\int_{t_0}^t e^{-i\Omega_j(t-s)}A(s)^2ds.
\tag{3}
$$

After the field vanishes, $|b_j|^2=g_j^2|S(\Omega_j)|^2$. Therefore
$\sum_j|b_j|^2$ and $\sum_j\hbar\Omega_j|b_j|^2$ converge to (1) on $\mathcal D$.
This proves the integrated normalisation, not physical transient accuracy.

Equation (2) is retarded in the specified analytic drive. A Fourier band selector and
analytic-signal construction are themselves nonlocal in time; do not call the whole
filtered real-field model microscopically causal. Test sensitivity to the selector and
exclude apparent advance caused by its tails from an atomic-memory claim. There is no
window-wide yield coefficient that changes an earlier atomic response when only a
later part of the *prescribed drive* is changed.

#### 4.3 Polarisation and work from the same interaction

A classical complex-coordinate Hamiltonian, per initially neutral atom, for (2) is

$$
\mathcal H=\hbar\sum_j\Omega_j|b_j|^2
 -\hbar\left(Q^* A^2+Q A^{*2}\right),\qquad Q=\sum_j g_j b_j.
\tag{4}
$$

Indeed $i\hbar\dot b_j=\partial\mathcal H/\partial b_j^*$.
With $E_V=\operatorname{Re}A$, the analytic dipole consistent with analytic work
normalisation is

$$
d_a=-4\frac{\partial\mathcal H}{\partial A^*}
       =8\hbar A^*Q,\qquad
\frac{d\mathcal H}{dt}=-\tfrac12\operatorname{Re}(d_a\dot A^*).
\tag{5}
$$

The partial derivative holds the atomic coordinates fixed. Integrating by parts,
with field and interaction vanishing at both endpoints, gives

$$
\tfrac12\operatorname{Re}\int A^*\dot d_a\,dt
 =\mathcal H(t_1)-\mathcal H(t_0)
 =\sum_j\hbar\Omega_j|b_j(t_1)|^2\ge0.
\tag{6}
$$

These identities are exact for the auxiliary model. Instantaneous work can be negative
through reversible exchange; do not clip it. Positive completed-pulse work is the
passivity claim, not pointwise dissipation. Endpoint interaction terms must be included
for an unfinished pulse. The proposed physical polarisation is
$P_V=N_g\operatorname{Re}\Pi_+d_a$, where $\Pi_+$ retains positive frequencies without
analytic-signal doubling. This projection preserves completed-window work against the
positive-frequency $A$; derive the finite-window endpoint and discrete residuals before
implementation. No extra two-photon ionisation-loss current is added: (5) already pays
the full absorbed pair energy, including ionisation and photoelectron kinetic energy.

Equations (2)–(6) define an effective completion of the measured on-shell response.
They are not asserted to be the unique off-shell atomic Hamiltonian. The continuum
coordinates have no spatial electron information. Their completed-pulse energy identity
alone cannot qualify a plasma model. Coupling through $A^2$ keeps only the
sum-frequency pathway that is resonant with the continuum. The counter-rotating
($\Omega+2\omega$) and difference-frequency ($|A|^2$) terms of a real-field quartic
response are omitted. They are nonresonant parts of the existing electronic response,
so their omission is part of the matching problem of section 4.5, not an extra loss.

#### 4.4 Finite bandwidth, memory and numerical representation

Eliminating $b_j$ gives $Q=iR*A^2$ with the retarded kernel
$R(t)=\Theta(t)\sum_j g_j^2e^{-i\Omega_jt}$. In the continuum limit this is an integral
over $k/(2\pi)$. Its reactive and absorptive parts are linked by the same retarded
construction. Near-threshold memory is retained; replacing it by a carrier coefficient
is only a separately tested limit.

A finite undamped quadrature can recur. Resolve its energy spacing over the entire
atomic time window and demonstrate that recurrence occurs outside the window; for a
uniform spacing the recurrence scale is $2\pi/\Delta\Omega$. Converge both quadrature
spacing and window length independently. Do not use an arbitrary damping width to hide
recurrence: Lorentzian broadening can create artificial subthreshold absorption. A
rotating frame and exact homogeneous evolution with converged integration of the drive
can avoid resolving the high continuum frequency with a general-purpose ODE solver.
First implement the explicit auxiliary equations as an oracle. A zero-padded, linear
retarded convolution may replace them only after time-origin, padding, endpoint and
work tests; a circular FFT convolution with its initial state wrapped around is wrong.

The model is cheap relative to an atomic TDSE, not necessarily to an instantaneous
cubic. With $N_c$ continuum nodes, $N_t$ fine time samples and a chunk of $N_x$ physical
field nodes, the direct cost is $O(N_xN_cN_t)$ and a streamed implementation needs
$O(N_x(N_c+N_t))$ storage. Recurrence-free resolution needs
$N_c\gtrsim(\Omega_{\max}-\Omega_{\min})T/2\pi$ for a window $T$: a 15 eV domain and a
1 ps window give $N_c\approx3.6\times10^3$. With $N_t\sim3\times10^4$ and sixteen radial
nodes that is about $2\times10^9$ complex updates per right-hand-side evaluation,
before convergence margins. The direct form is therefore an oracle, not a plausible
production kernel for propagation.

Two analytical reductions are expected to carry production. First, $Q=iR*A^2$ is
exactly a linear retarded convolution before sampling and truncation. Adapt the
doubled-grid padding and window-convergence approach of the Raman translation
(`solprop.translate.raman`); its existing arrays and real-transform path are for real
kernels. Here both $R$ and the drive $A^2$ are complex and require corresponding
complex arrays and transforms, not direct use of the real Raman kernel interface.
The kernel can be sampled from the continuum integral directly, without discrete
continuum nodes or their recurrence. Its long tail still needs convergence checks.
A collar-shaped tail taper changes that kernel and can alter its absorptive spectrum;
it is not an exact reduction or a guarantee of passivity. Qualify its optical work,
passivity and window dependence against the untapered auxiliary oracle, and enlarge
the window until the relevant delays are unaffected by the taper. Second, the
transverse factorisation of section 2 makes the single-mode response one modal
evaluation scaled by $F^3$ instead of $N_x$ node evaluations. The per-node path stays as a test of that equality. Benchmark measured
$N_c$, accuracy and transform costs before the choice is fixed in Phase 1. No pole fit
or new dependency is required for the first oracle.

If the selected material model or declared completion has a finite, nonzero limit of
$k$ immediately above threshold and zero below, it has a step. That step gives $R(t)$
a $1/t$ tail and a logarithmic singularity in the stationary monochromatic reactive
coefficient. A nonzero step is a conditional model assumption here: the stored Ar/He
curves do not determine the response throughout their threshold gaps. A finite-duration
pulse drives the time-domain convolution, which remains finite; it does not acquire an
infinite polarisation at threshold. Define the memory time operationally as the kernel
truncation beyond which the admitted finite-pulse results converge. Converge quadrature,
window and kernel sampling against the chosen threshold structure explicitly.

#### 4.5 Material support and the dispersive matching problem

Physical zero below threshold is distinct from `unavailable` and `unsupported_detail`.
Report all three, using optmat's region masks. The cross section controls integrated
loss at occupied pair energies, but unoccupied continuum energies can still influence
the reactive response. Consequently a small missing *pair-power* fraction does not
bound the dispersive error or the missing absorption without additional information.

For the prescribed-field diagnostic, report the retained integral and missing support
separately. For propagation require an explicit continuum completion record naming the
energy domain, outer taper, gap assumptions, material identity and uncertainty variants.
A zero outside a finite domain is a declared truncated-model assumption, never an
optmat physical value. Do not silently fill a threshold gap, continue a constant cross
section to infinity, or label a truncated response fully qualified. Vary the outer
domain and taper independently of quadrature and test their effect on the pulse.

The continuum polarisation includes a real third-order contribution. Adding it to the
existing electronic response risks counting some dispersion twice. The proposed first
matching rule is a real, conservative subtraction of the continuum model's
monochromatic reactive coefficient at a declared VUV reference frequency, with the
existing background kept at that reference. Its coefficient is computed from the
chosen model, not fitted to the observed cut-off. Derive the subtraction from a local
quartic Hamiltonian and apply it through the same field and source selectors, so its
completed-pulse work is zero. Phase 0 must verify its factor, reference sensitivity and
comparison with the existing electronic model. The latter is already best effort in
this VUV region; matching to it does not turn it into validated dispersion. If predictions
depend materially on this choice, the first model cannot settle the experimental
mechanism. Do not remove the entire reactive kernel by hand and retain the absorption
while calling the result the causal model of (2).

The threshold is where the matching is least secure. To fix the sign and limit,
consider $k(\Omega)=k_0>0$ on the finite pair-frequency interval $[a,b]$ and zero
elsewhere. Before the conservative matching subtraction, the real coefficient of
$|A|^2A$ in the stationary monochromatic dipole $d_a$ is

$$
C_{\mathrm{re}}(\omega)
 =\frac{4\hbar}{\pi}\operatorname{PV}\int_a^b
      \frac{k_0}{\Omega-2\omega}\,d\Omega
 =\frac{4\hbar k_0}{\pi}\ln\left|\frac{b-2\omega}{a-2\omega}\right|,
 \qquad 2\omega\ne a,b.
$$

It grows positively as $2\omega$ approaches the lower edge $a$. This is the undamped,
stationary limit of the declared step model, not a divergence of the finite-pulse
response. In the atom, bound Rydberg structure below threshold and, in Ar, autoionising
structure between the ionic fine-structure thresholds are outside this model (Q11)
and can affect matching in the RDW-relevant band. Absorption is constrained by optmat's
selected calculated cross sections only on their qualified support: missing and
unresolved threshold regions remain uncertainties in absorption as well as dispersion.

Phase 0 also runs a labelled absorptive-only sensitivity variant, replacing the retarded
kernel spectrum by $\widetilde R_{\mathrm{abs}}(\Omega)=k(\Omega)/2$, with
$Q_{\mathrm{abs}}=iR_{\mathrm{abs}}*A^2$. It is a comparison, not the causal model and
not an admitted production optical mode. If the experimental conclusion differs
between the two, the reactive threshold response is an open model dependence and is
reported as such. Agreement does not validate a shared missing-data completion.

#### 4.6 Band split, source projection and pump baseline

Choose a real, even spectral partition $H(\omega)$ with $0\le H\le1$: zero for
$|\omega|\le\omega_c-\Delta$, a Planck transition to one at $|\omega|=\omega_c$,
and one above it. Define

$$
E_V=\mathcal F^{-1}[H\widetilde E],\qquad E_p=E-E_V,\qquad
A=\operatorname{analytic}(E_V).
$$

Use the fine-grid endpoint and Nyquist rules of `solprop.fourier.analytic_signal`.
The bands are real and sum to the original field. The proposed cut is
$\hbar\omega_c=\min(E_1,E_2/2)$ from optmat's one-polarisation electronic support record,
where $E_1$ and $E_2$ are its first one- and two-photon excitation energies. This is
SEP 6's boundary of qualified electronic dispersion, not the ionisation threshold.
Read the levels, thresholds and support-record identity from optmat; do not hard-code
a wavelength. The transition lies below the cut; Q2 fixes and qualifies its width.

A pair with one photon below the cut can exceed the ionisation threshold only if its
partner exceeds $I_p/\hbar-\omega_c$. Report the actual single-photon and pair support
against this limit. The split cannot justify ignoring all mixed pairs, pump dressing,
or higher-order pump-assisted channels. Content remaining in the pump band is not
automatically within the strong-field rate's validity, either. The earlier mixture
rule is outside the revised first scope and needs a later qualification.

If $A$ is synthesised through a spectral selector $B$, return the
continuum source through $B^\dagger$. In the real-field work inner product,
$\langle E,B^\dagger J_V\rangle=\langle BE,J_V\rangle$; filtering only the drive loses
this identity. The same requirement applies to the pump loss-current branch [6] when its
rate sees a selected pump field. Source projection onto the single optical mode must
be the adjoint of physical-field synthesis in the declared transverse quadrature.
Density is applied once. Measure any source outside the retained spectral and modal
space, and separate its work from collars and linear attenuation.

Derive the internal-sign implementation from the conventions' named boundary helpers;
(5) is in the optics convention. Reuse the polarisation source factor for $P_V$.
A direct-current implementation must use the existing current source factor, not
integrate a VUV current by a trapezoidal prefix and apply an unrelated prefactor.
Qualify the actual discrete work, spectral crop and time endpoints, not only (6).

Keep three comparisons distinct: the legacy total-field pump rate; the declared split
pump baseline; and that same split baseline plus continuum optical response. The
change caused by removing VUV content from a pump-carrier rate must not be attributed
to two-photon absorption. A split-only baseline is a labelled diagnostic, not a complete
recommended model. Use exactly the same split in the pump-only and combined tests.
The existing pump electrons still drive the plasma current through the total local
field. No VUV two-photon energy is charged to the pump loss current. Retain the
strong-field rate's own energy per event and pre-ionisation semantics, checking its
table range and peak-weighted energy against the *filtered pump field*: its peak need
not be bounded by the original field's peak. Record pump event energy, continuum
optical work and free-electron work separately. A future shared-depletion model must
count each electron once and use the same discrete event counts for population and
energy; the initial weak-excitation model does not claim to implement that closure.

#### 4.7 Additional electrons: final yield is not a birth-time law

$P_c(t)=\sum_j|b_j(t)|^2$ is nonnegative, but can decrease and includes off-resonant,
virtual excitation while the field is present. Even a subthreshold drive can have
$P_c(t)>0$ and $P_c(\infty)=0$. Therefore neither $N_gP_c(t)$ nor a clipped positive
$\dot P_c$ is an admissible default free-electron density/rate. A spectral factor
$|H|^2=k$ with a guessed phase and $|h*A^2|^2$ also does not establish the electron
birth history or its compatibility with (5).

**The first implementation supplies final added-electron density
$N_gP_c(\infty)$ as a diagnostic only.** Its optical feedback includes the continuum
polarisation; the VUV-produced electrons do not yet enter the Drude current. Label every
such propagation result `continuum optical response; additional plasma omitted`.
This omission is acceptable for the experimental inference only after its possible
effect has been bounded. Vary diagnostic electron histories over physically plausible
birth intervals as a sensitivity experiment, including temporal overlap with the pump;
such scans do not constitute a derived or generally rigorous bound on all histories.
If sensitivity is material, stop short of selecting an experimental mechanism.

**Gate for a later plasma-enabled phase:** identify an outgoing-electron or other
independently justified timing model; show no free electrons for reversible
subthreshold excitation; compare the predicted current/work as well as final yield
with converged atomic calculations on VUV-only and pump-plus-VUV fields. A delayed
absorber flux is not automatically the birth rate at the atom. Any chosen time
coarse-graining or escape-time prescription and its sensitivity must be declared.
A new atomic calculation may be needed; this is not information in $\sigma^{(2)}$.
Only then specify shared neutral depletion and the free-electron current and re-review
the added scope. Do not exponentiate (1), invent a positive rate, or count the ionisation
energy again in the plasma loss. Failure of this gate leaves a useful optical-response
model, not a completed test of all two-photon feedback.

#### 4.8 Weak-excitation and omitted-physics qualification

Track both final yield and the maximum auxiliary occupation; neither alone establishes
weak atomic dressing. Test intensity scaling, spectral split sensitivity, chirp and
pump overlap against independent atoms. Start with the existing, fixed initial neutral
fraction; require pump depletion over the atomic memory time to be negligible for the
stationary response. A slowly varying multiplier on (5) can change its energy identity
and is not an authorised shortcut. Strong depletion, extra charge states and stimulated
recombination are outside the first qualification.

Bound resonances, higher-order VUV processes and pump-assisted channels can remain
important even when direct one-pump-plus-one-VUV ionisation is energetically closed.
The continuum-only result does not bound their contribution. Use optmat's existing
material records, and the Ar calculations they curate [3, 4], with atomic comparisons
to assess their relevance. The Ar calculations
in optmat's OEP 8 review do not validate all near-threshold wavelengths, and helium validation
does not qualify argon. The current He and Ar support intervals are read from optmat at
translation and recorded, not copied as permanent numerical limits in solprop.

#### 4.9 Does this need a coupled optmat OEP?

**Decision proposed: no new optmat OEP is required for the initial model in this SEP.**
The auxiliary weights, spectral completion choices, dipole and energy identities are
solprop's propagation approximation. They do not create a new measured material
quantity. optmat continues to supply its cross sections and other material data.

| Input or missing information | Present source | Consequence |
|---|---|---|
| Degenerate $\sigma^{(2)}$, thresholds, masks, uncertainty and identity for Ar/He | Implemented OEP 6 | Enough to construct and normalise (2) on supported domains; update the dependency pin before implementation |
| Pump rates and energy per event; density and electronic response | Existing optmat interfaces used by SEP 3/6/17 | Reuse; do not transcribe into solprop |
| Bound levels and their coefficients | OEP 6 records and proposed OEP 8 | Separate bound-response work; not needed to implement the direct continuum |
| Cross sections in the gaps immediately above threshold, especially Ar's ionic fine-structure region | Incomplete OEP 6 support | Additional curation/calculation may be essential to a threshold claim; do not replace by zero |
| Cross sections outside the chosen domain and real-response matching information | Partial data; no uniquely fixed completion | Recorded solprop model sensitivity; refuse an accuracy claim if it dominates |
| Nondegenerate amplitudes or channel-coherence information | Not supplied by the degenerate curve | Validate the separable approximation first; a new material representation would require a coupled OEP if needed |
| Electron birth/current during overlapping pulses | Not encoded by a total cross section | solprop response derivation and atomic validation, not a new optmat plasma-polarisation API |

New sources, errata and added points for the existing cross-section quantity follow
optmat's ordinary curation and register rules, not a new OEP merely because the data
are missing. A new OEP becomes necessary if qualification demands a new quantity
family/schema, a nondegenerate material amplitude or overlap model, a new fitted
material model/default, or another major optmat interface change. Its scope must name
that additional material contract; the causal field response remains in a solprop SEP.
Do not draft a placeholder OEP promising a plasma polarisation or an unspecified
complex response. Reassess this decision at the Phase 0 gate, recording the evidence.
This distinguishes sufficiency for building an explicit approximation from sufficiency
for a physically qualified prediction at every wavelength of the experiment.

### 5. Proposed interface

Names are a proposal, not implemented API. Use one declaration with explicit modes:

```text
continuum = TwoPhotonContinuum(
    species="Ar",                       # or "He"
    material="optmat",
    split=BandSplit(...),
    completion=ContinuumDomain(...),     # explicit support/completion and matching record
    coupling="diagnostic",              # or "optical" after Phase 0 qualification
)
```

`coupling` is not called `mode`, which in solprop names a waveguide mode.
`diagnostic` consumes physical fields or saved fields with their documented
power-to-field conversion and transverse mode normalisation. It reports (1), support
fractions and whether the pulse completed. It changes neither the field nor the plasma.
`optical` adds (5) and its declared conservative matching subtraction to the existing
response combination, with pump ionisation optional. It reports final continuum yield
and work, but no additional Drude density. There is no `plasma=True` option until the
later gate has supplied and reviewed a concrete model.

`ContinuumDomain` specifies angular-frequency endpoints and quadrature resolution,
outer taper, unsupported-region treatment and matching reference, all with SI units.
Numerical tolerances are separate from material/model uncertainty. User completion
variants are recorded as assumptions with identities, never silently promoted to optmat
recommended values. Default physical mode is off; no default arbitrary gap filling.
Refuse incompatible routes, polarisation, species, nonfinite/negative material values,
nonpositive quadrature weights, invalid domains, a backend without the response and
simultaneous native TDSE or another two-photon optical response for the same atom.

Missing spectral coverage must not be hidden by a successful run. Diagnostic mode can
report a retained integral and `incomplete`; optical mode needs an explicit completion
policy or refuses the material construction. Dynamically occupied unsupported physics
is reported and prevents a qualified interpretation; it is not converted to a silently
clipped source. Define empty-denominator and empty-band diagnostics without division
by zero. Scope refusals do not turn small support fractions into error guarantees.

Record material/source digests, optmat version, split, completion and matching policy,
continuum quadrature, model version, qualification label, omitted plasma flag and
initial neutral fraction in the run manifest and resume identity. Statistics separate
computed atomic work, projected source work, final continuum yield, endpoint residuals,
missing-data exposure and numerical boundary loss. Atomic auxiliaries reset at the
start of the time window on every optical RHS call, including rejected optical steps;
there is no hidden history across propagation distance or across RHS retries.

### 6. Implementation strategies

| Option | Advantage | Limitation | Recommendation |
|---|---|---|---|
| Carrier-evaluated local rate/current | Smallest cost | Cannot resolve threshold-straddling pair spectra | Scale check only |
| Earlier window-calibrated rate | Correct integrated weak-field totals | Unqualified timing and source-work distribution | Comparison only; replaced by the continuum candidate |
| Explicit effective continuum, (2)–(6) | Retarded auxiliary dynamics and one work identity; finite model is inspectable | Continuum resolution, reactive completion and electron-timing limits; $O(N_xN_cN_t)$ per evaluation | Reference oracle; production only if the convolution fails its qualification |
| Retarded convolution of the same model | Raman-like doubled-grid kernel; no continuum-node recurrence; cost independent of $N_c$ | Kernel-tail, padding and source-work qualification; same physical limits | Expected production route, after oracle parity (section 4.4) |
| Full atomic TDSE | Independent dynamics, within its atomic approximation | Much greater cost and atom-dependent errors | Selected validation runs |

Physics-layer declarations and the optmat adapter are separate from translation's
quadrature, coupling and selector arrays. Backends contain numerical loops and no
material selection. Add a typed immutable continuum leaf to the kernel contract and
bump `SCHEMA_VERSION` on implementation; do not choose a future version number now.
The polarisation/current source factors already exist. Reuse the disabled path exactly,
with no continuum buffers or transforms when off. The driver's step/retry semantics
are unchanged. Diagnostics and source projections must share the physical quadrature.

Implement `numpy_ref` and its material-independent oracle first, then `numba_cpu`, then
Kokkos CPU. Unsupported backends refuse by name. Qualify GPU separately under the
standing backend policy. Avoid materialising the continuum-by-time trajectory in
production; chunk physical nodes and stream the auxiliary state. No core dependency
addition is proposed. A future reduced kernel cannot discard threshold memory merely
to meet a speed target.

### 7. Verification and validation

The distinction is between exact identities of the chosen approximation and independent
validation against atoms. Tolerances are set by convergence and source uncertainty,
not by accepting the first residual obtained.

- **Normalisation:** long, weak monochromatic pulses reproduce optmat's rate and
  absorbed pair energy. Compare direct quadrature of (1) with independent integration
  of (2) for Gaussian, chirped, two-colour and delayed-pulse drives, including pairs
  on either side of threshold. Keep coherent cross terms.
- **Energy:** compare (6), atomic final energy and the actual projected optical source
  work; refine time, continuum energy and transverse quadrature separately. Test
  incomplete endpoints, empty bands, real-field zeros, broad spectra, selector
  transitions, DC/Nyquist rules, positive-frequency projection and spectral cropping.
  Verify the matching subtraction does zero completed-pulse work.
- **Retardation:** hold an earlier analytic drive fixed while changing a later pulse;
  earlier auxiliary states must agree. Separately quantify band-selector tails.
- **Continuum convergence:** vary the outer domain/taper as a model study, spacing as
  numerical quadrature, and the window as an endpoint/recurrence study. Never conflate
  convergence of a truncated continuum with completeness of the material response.
  Resolve a declared threshold step separately: check the stationary reactive
  coefficient against its signed principal-value logarithm away from the step edges,
  and verify a finite, converged response for pulses centred across the threshold.
- **Convolution parity:** the complex doubled-grid convolution matches the auxiliary
  oracle on the same drive. Delayed pulses whose separation enters the kernel taper
  expose its error; enlarge the window and converge that error separately from sampling.
  Check optical work and passivity as well as waveform error; padding alone does not
  qualify a tapered kernel.
- **Transverse factorisation:** on a single mode, per-node evaluation and the scaled
  modal evaluation agree to rounding, with and without a fixed neutral profile.
- **Threshold sensitivity:** compare the causal response with the labelled
  absorptive-only variant on the replayed and propagated RDW cases (section 4.5).
- **Weak response:** yield scales as field amplitude to the fourth power, dipole to the
  third; compare narrowing bandwidth with the local attenuation law. Field depletion
  along distance may be finite while single-atom excitation remains perturbative.
- **Independent atom:** compare VUV-only pulses first, then add delayed and overlapping
  pump fields. Compare total absorbed work, final yield and complex nonlinear current
  or dipole where available. Use the atomic model's own cross section for a structural
  comparison; separately compare it with optmat's material values. Do not calibrate the
  absorptive response with an unrelated Kerr scale. Helium correlated calculations
  and argon calculations require separate evidence; an SAE discrepancy is not a
  numerical tolerance to absorb. The existing optmat checks at sampled energies do
  not settle the near-threshold case.
- **Electron gate:** demonstrate the virtual-population counterexample before trying
  any plasma closure. Final yield agreement alone does not pass section 4.7.
- **Propagation:** use about ten steps or seconds-long short segments for qualification,
  including no-pump, split-pump baseline and combined optical response. Compare energy,
  spectrum and per-channel work, not only peak wavelength. Full experimental scans
  belong in a worked example or research run, never the test suite.
- **Backend parity:** match NumPy, Numba and Kokkos CPU on identical descriptors and
  short segments, including disabled-path equality and repeated/rejected RHS calls.

**Drafting check, not atomic validation:** independent DOP853 integration of a
three-node synthetic continuum in dimensionless units verified the factor in (5).
With $\hbar=1$, $\Omega=(3.2,4.0,5.1)$, $g=(0.006,0.008,0.005)$ and
$A=\exp(-t^2/2-2it+0.13it^2)$ on $[-8,8]$, relative/absolute solver tolerances
$10^{-11}/10^{-13}$, the final occupation is $0.000318831632448$.
Independent Simpson integration of the Fourier amplitudes agrees to $6.8\times10^{-12}$
relative, and $-\tfrac12\operatorname{Re}\int d_a\dot A^*dt$ agrees with final
continuum energy to $8.6\times10^{-12}$, at both 6001 and 12001 time samples.
These synthetic numbers are reproducible formula checks, not material data or production
test tolerances. No atomic validation or RDW propagation was run for this draft.

### 8. Documentation

Add `docs/model/two-photon-continuum.md` with the derivation, material/solver boundary,
field normalisation, selectors, support/completion rules, optical-work identity,
convergence instructions and explicit limits on electron timing. Link it from the
nonlinear, ionisation, material and running pages. Document each public declaration and
manifest field. A short tutorial compares the diagnostic with the optical response in
a supported band; it must not claim a threshold prediction from incomplete data.
A worked example replays a declared RDW dataset and then performs the controlled
pump/continuum comparisons, with the actual input pulse and geometry. Publish all
omitted-physics labels alongside its spectra, and link the final SEP from its manual.

### 9. Compatibility and impact

Additive and opt-in. No existing material choice, default response, convention or
simulation result changes when disabled. The new leaf changes the kernel schema and
requires backend admission and resume compatibility checks. Record the response and
model assumptions in output metadata. Update the manual and changelog on implementation,
not merely because this draft exists. Deliberate changes to the treatment of VUV content
in the pump rate are documented on the Luna differences page with the baseline tests.
No code or data in optmat changes as part of this draft; release/pin work precedes the
adapter implementation. The first scope now excludes Ne/Xe, mixtures, mode-averaged
and multimode routes that the earlier design intended to support. Those remain on
the roadmap. SEP 2 remains independent: its narrowband attenuation law provides a
limiting check, not a second optical response to combine with this one.

### 10. Alternatives considered

The former window-calibrated rate and separate real loss current are not the selected
candidate in this revision. Matching integrated yield does not establish loss timing
or spectral work. The previous review's two-tone current residual and pump-source
counterexample remain regression cases for Q4/Q18; they are preserved in Part III.
Likewise, a quotient using the total field to force instantaneous ionisation-loss work
can be singular at a total-field zero while the pump rate is nonzero. Adjoint source
coupling is the proposed route, with its discrete qualification still required.

A positive rate obtained by taking a square root of a cross section is not enough:
its phase, timing and source work are unspecified. A carrier coefficient cannot provide
the threshold response of a broadband pulse. A full nondegenerate, channel-resolved
amplitude model may eventually be necessary but is not a prerequisite for testing the
explicitly separable approximation. A separate optmat OEP solely to reproduce its
existing cross sections would add process without material content. Conversely, the
absence of a new OEP must not be used to hide missing material data or transfer the
propagation closure into optmat. A fitted cut-off or arbitrary damping chosen to match
the experiment would not discriminate the hypotheses and is excluded.

### 11. Phases

Retain the qualification-first numbering: Phase 0 precedes production implementation
in Phase 1. Each implementation phase uses its own `feature/sep-0017-…` branch and
appends a report to Part II with decisions, checks, source evidence, limitations and
handoff. The current Part I must be reviewed and re-accepted before implementation.

- **Phase 0 — Qualification and prescribed-field diagnostic.** Derive and review the
  continuum and dipole normalisation, selectors, discrete work and conservative
  matching term. Build the synthetic oracle and research replay diagnostic; audit the
  actual optmat pin, material coverage and experimental pulse definitions. Compare
  selected atoms, assess omitted channels and document the first admitted domain.
  Resolve the production gates in Q1–Q4, Q6, Q9, Q13 and Q16–Q18; reassess Q20 and
  benchmark the execution routes of Q21.
  A useful diagnostic can be delivered even if the optical-response gate fails, but
  that does not complete the SEP or authorise production optical feedback.
- **Phase 1 — NumPy optical response.** Implement the immutable declarations, adapter
  and seam, physical-node response, manifest and diagnostic outputs; tests, manual and
  changelog. Include the conservative matching term and expressly omit additional VUV
  plasma. Qualify short propagation segments against the oracle and split baseline.
- **Phase 2 — Numba CPU.** Compiled loops, shared descriptor admission, parity,
  coverage and short performance ledger. Unsupported backends refuse by name.
- **Phase 3 — Kokkos CPU.** Native implementation and parity, descriptor admission
  and the checks required by both repositories. GPU qualification remains separate.
- **Phase 4 — Scientific example and completion.** Representative RDW replay and a
  worked example with the declared pump/continuum comparisons. Record failures of
  material or atomic qualification plainly. Complete the repositories' required final
  gates and CPU parity, document the admitted domain and what the optical-only model
  can establish, and move unresolved plasma and omitted physics to the roadmap.

**Conditional plasma extension:** if added-electron feedback could alter the scientific
conclusion, derive and review section 4.7's timing/current model in a revision before
acceptance or a later SEP after Part I is frozen. Identify any new optmat material
contract at that point and draft its coupled OEP. Full VUV plasma feedback and strong
shared depletion are not claimed delivered by the optical-only phases. Ne/Xe,
mixtures, mode-averaged and multimode configurations are also follow-up work, each with
its own material, source-projection and atomic qualification.

### 12. Open questions

Q1–Q18 retain their subjects and identities from the earlier review. This revision
changes the proposed response and some earlier scope decisions; old review references
in Part III refer to the text reviewed at that time. Q19–Q21 make the new gates explicit.

- **Q1 — optmat release and references (Phase 0; production gate).** Confirm the pinned
  release supplies OEP 6's required APIs, the selected Ar/He source identities and
  uncertainty/region semantics. Update the pin before adapter implementation.
- **Q2 — Cut, width and mixtures (Phase 0).** Qualify section 4.6's material-derived
  cut and transition width against actual fields. Mixtures are now deferred; the
  earlier common-cut rule is not an admitted configuration in the first model.
- **Q3 — Broadband response (Phase 0; production gate).** Replace the local-rate
  candidate with (2)–(6). Verify its normalisation and weak-excitation domain; validate
  the frequency-split approximation separately for Ar and He. An exact integrated
  identity is not evidence for physical transient accuracy or a saturation law.
- **Q4 — Source coupling (Phase 0; production gate).** Derive and qualify the adjoint
  selectors for both continuum polarisation and pump loss, including physical-field
  synthesis and modal projection. Retain the earlier counterexamples. Positive work
  against a selected band alone is insufficient; coordinate with Q18.
- **Q5 — Default (unchanged).** Opt-in. No new default is proposed by this revision.
- **Q6 — Missing material coverage (Phase 0; production gate).** Read actual pair and
  single-photon support, especially near Ar's threshold. Diagnostic mode may report
  an incomplete retained integral; optical mode requires a declared completion or
  refuses construction. Dynamically occupied gaps are reported and block a qualified
  inference, not silently clipped. A small missing share is no accuracy guarantee.
  This replaces the earlier warning-only construction policy.
- **Q7 — Continuum without a pump rate (revised scope).** Permit VUV-only diagnostics
  and optical response for qualification and supported applications. A pump rate is
  optional. A split-only baseline is a labelled comparison, not a complete model.
- **Q8 — Envelope grids (deferred; roadmap).** Refused in the first implementation.
- **Q9 — SEP 6 and reactive matching (reopened; Phase 0 production gate).** The
  continuum now includes a real response. Derive and review the conservative reference
  subtraction, test reference/domain sensitivity and avoid double counting with the
  electronic background. Its best-effort VUV label is not upgraded by matching.
  Report the conditional monochromatic logarithm, finite-pulse convergence and the
  absorptive-only comparison; if the conclusion depends on them, the reactive response
  near threshold stays open. Shared threshold data gaps also limit both variants.
- **Q10 — SEP 2 (independent).** Use its narrowband attenuation limit as a check. Its
  solid coefficient and proposed local current are not this gas continuum model.
- **Q11 — Bound resonances and (2+1) ionisation (deferred; roadmap).** optmat OEP 8
  supplies proposed material work; a separate solprop response is still needed. Bound
  coefficients do not complete this continuum model or justify omitting bound physics.
- **Q12 — Polarisation (deferred; roadmap).** Fixed linear polarisation only; vector
  fields and other polarisation states need material and response qualification.
- **Q13 — Timing and coherence (Phase 0; production gate).** The window-wide rate is
  replaced, but the new model must pass delayed-pulse, coherent cross-term, selector-tail
  and memory tests. Hold the earlier prescribed analytic drive fixed when testing
  retardation. Do not infer electron birth times from the auxiliary occupation (Q19).
- **Q14 — Other species and routes (deferred; roadmap).** Ne/Xe, mixtures,
  mode-averaged and multimode cases, free-space, planar and general modal routes need
  separate admission and qualification. The first route is a single-mode capillary
  with resolved physical-node quadrature. For the continuum response alone a
  mode-averaged single mode is exact (section 2); its exclusion is a scope choice
  because its pump plasma is not.
- **Q15 — Interface, statistics and grid (Phase 1).** Finalise section 5's declarations
  (the `coupling` keyword avoids the waveguide sense of `mode`),
  manifest identities, numerical admission, zero-denominator semantics and separate
  atomic/source work, final yield and support diagnostics. Rejected optical steps
  must not leave hidden atomic history.
- **Q16 — Omitted VUV and pump-assisted channels (Phase 0; then roadmap).** Assess
  three-photon ionisation, bound excitation and pump dressing/assistance in the actual
  fields, using available optmat quantities and atomic comparisons. A significant
  unmodelled channel prevents attribution; its implementation is outside this scope.
- **Q17 — Atomic validation (Phase 0; production gate).** Establish the frequency
  splits, intensities, durations and pump overlaps for each species. Compare complex
  current/dipole, work and final yield, using the atom's own cross section for structural
  tests. Helium is not an argon validator. No fixed 6 or 10 fs duration substitutes for
  the generated local VUV pulse; a missing suitable reference remains an evidence gap.
- **Q18 — VUV work and discrete source (Phase 0; production gate).** Qualify (6) under
  positive-frequency projection, adjoint selectors, finite endpoints, spectral crop,
  transverse quadrature and the correct source factor, with density applied once.
  The direct polarisation avoids an unnecessary current integral, but not these work
  tests. The pump branch uses the existing band-rule plasma path. Set tolerances by
  convergence before production implementation.
- **Q19 — Added plasma (conditional extension).** Is its effect negligible for the
  selected experiment? If not, derive and validate an electron-timing/current model
  before feeding extra electrons into the plasma or adding shared depletion. Neither
  transient occupation nor a clipped derivative is a valid default. Optical-only
  results cannot close a materially plasma-dependent mechanism question.
- **Q20 — Coupled optmat proposal (reassess at Phase 0).** None for the initial
  cross-section-based approximation. Existing-quantity gaps are curation work. New
  nondegenerate/channel-coherence representations, fitted material models or major
  material interfaces require a named coupled OEP before their implementation.
- **Q21 — Cost and continuum compression (Phase 0 benchmark; Phase 1 choice).** Select
  recurrence-free quadrature for the oracle and benchmark it against the doubled-grid
  kernel convolution and the transverse factorisation of section 4.4. The direct
  auxiliary form is expected to be too costly for propagation. Do not broaden or
  discard spectral structure for speed.

### 13. References

1. optmat, [OEP 6 — Two-photon ionisation and absorption of the noble gases][oep6],
   Final; its integrated pair formula and [current data/model manual][optmat-data].
2. optmat, [OEP 8 — The bound two-photon resonances of the noble gases for propagation][oep8],
   Draft, Reviewed; see its entries in optmat's [review register][optmat-register].
3. C. McKenna and H. W. van der Hart, "Multiphoton ionization cross sections of neon and
   argon," *J. Phys. B* **37**, 457–470 (2004).
   [doi:10.1088/0953-4075/37/2/013](https://doi.org/10.1088/0953-4075/37/2/013).
4. I. D. Petrov, B. M. Lagutin, V. L. Sukhorukov, A. Knie and A. Ehresmann, "Correlation
   and polarization effects in two-photon photoionization of Ar," *Phys. Rev. A* **93**,
   033408 (2016). [doi:10.1103/PhysRevA.93.033408](https://doi.org/10.1103/PhysRevA.93.033408).
5. A. Mihelič and M. Horvat, "Calculation of multiphoton ionization amplitudes and cross
   sections of few-electron atoms," *Phys. Rev. A* **103**, 043108 (2021).
   [doi:10.1103/PhysRevA.103.043108](https://doi.org/10.1103/PhysRevA.103.043108).
6. M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec, "Light
   propagation in field-ionizing media: extreme nonlinear optics," *Phys. Rev. Lett.*
   **83**, 2930 (1999). [doi:10.1103/PhysRevLett.83.2930](https://doi.org/10.1103/PhysRevLett.83.2930).

**Source audit.** [1–2] and optmat's curated source records for [3–5] were read for this
draft. The authors' accessible preprint of [5], arXiv:2103.16422, was checked for its
amplitude/cross-section method and, on follow-up review, section II.2's equal-photon-energy
formulation. It does not supply the required nondegenerate validation or validate the
reduced transient model proposed here. Primary papers [3–4] and [6] were not independently rechecked in
full for this draft; their bibliographic details and roles follow the curated records
and the earlier version of this proposal. No numerical material value was transcribed. Equations (2)–(6) are the
explicit candidate derived here from [1]'s integrated constraint, with the synthetic
check in section 7; they are not attributed to a source as a published atomic model.

[oep6]: https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/archive/oep-0006-two-photon-ionisation.md
[oep8]: https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/oep-0008-bound-two-photon-response.md
[optmat-data]: https://github.com/LupoLab/optmat/blob/v0.8.0/docs/two_photon.md
[optmat-register]: https://github.com/LupoLab/optmat/blob/v0.8.0/REGISTER.md

## Part II — Implementation record

*Empty until Phase 0. No implementation phase has been completed.*

## Part III — History

**Historical scope.** The reviews and acceptance below concern the earlier Part I.
Their section numbers, question resolutions and validation results describe that
version. They are preserved as evidence, not applied to the substantially revised
Part I above. The revised Part I was reviewed on 2026-10-12 (the last section).

- 2026-10-08: Drafted, paired with optmat's OEP 6 (Accepted the same day), with its
  relation to SEP 2 (Draft) and SEP 6 (Final) worked out.
- 2026-10-08: GPT-6 completed the AI review and revised Part I. John Travers stated
  that his human review is not yet complete; only the AI reviewer is recorded, so
  review status was In review under SEP 0 and lifecycle status remained Draft.
- 2026-10-08: John Travers confirmed completion of his human review and accepted
  the SEP in the review discussion. Review status is Reviewed and lifecycle status
  is Accepted. Part I is frozen on acceptance; Q3, Q4, Q13 and Q18 remain explicit
  Phase 0 prerequisites requiring resolution and review before Phase 1.
- 2026-10-09: Opus 5.5 checked the review revisions against the code and numerically:
  the two-tone residual (8/33), the pump-current counterexample (−1/8), the adjoint-filter
  work identity and the cumulative trapezoid's multiplier,
  $(\theta/2)\cot(\theta/2)$ of the exact one at $\theta=\omega\Delta t$, which at the
  default `ffac = 6` is 0.907 at $\omega_{\max}$ for one integral and 0.82 for the two
  of the Drude current. That factor also applies to the existing field-grid plasma
  path, independently of this SEP. The drafting-agent field now names the review
  revisions' agent. Part I is unchanged.

### AI review, 2026-10-08

The review covers all of Part I against accepted optmat OEP 6, SEP 0 and the current
plasma, Fourier, TDSE and kernel contracts. Findings are corrected in the proposal or
assigned to explicit Phase 0 decisions; no replacement physical closure is adopted
by this review.

- **Energy and passivity:** the proposed real current does not generally pay the
  pair-energy target, and positive work against a filtered band does not prove
  positive work against the propagated field. Added analytic counterexamples and an
  adjoint-filter candidate; reopened Q4 and added Q18 as prerequisites for Phase 1.
- **Timing and coherence:** the window calibration is nonlocal in time; separated
  coherent pulses retain weighted spectral interference. Gating changes the physical
  functional and cannot automatically repair timing. Q3 and Q13 require a reviewed
  closure and qualified regime before implementation.
- **Discrete propagation:** cumulative trapezoidal integration differs from the exact
  current multiplier inside the active band. Discrete channel attribution, source
  work, cropping, collars and density factors need an explicit consistent accounting.
- **OEP 6 coverage:** zero in a missing interval is a downstream omission, not an
  optmat value. Incomplete pair power supplies no relative loss bound; a propagated
  solution is not a lower bound. Pair-mean support also cannot qualify individual
  photon energies or nondegenerate amplitudes.
- **TDSE and admission:** macroscopic current scaling does not scale microscopic
  ionisation yield. Comparisons use unfiltered native scale one. Added empty-band and
  zero-yield semantics, actual-bin grid admission, mixture-split consistency and
  separate event-energy and work diagnostics.

**Evidence and limits.** Independently checked the analytic-signal normalisation
against OEP 6's $E^+$ formula and re-derived the work and filtering identities. A
65,536-point periodic numerical check reproduces the two-tone residual $8/33$ and
the pump-current counterexample's work $-1/8$ against the total field, to floating-point
precision. Independent Bessel quadrature and SI constants reproduce §4.11's transverse
factor 0.565536 and initial energy-loss coefficient 0.635747 m⁻¹; the ripple factors are
1.863965 and 29.788855. Checked optmat's proposal and data-page coverage, and inspected
solprop's actual plasma integrals and TDSE source assembly. No cross-section dataset
or source coefficient was changed. The primary atomic-data papers were not independently
re-audited here; their evidence remains optmat's curated records and OEP 6's source
audit. No TDSE, backend implementation or propagation qualification was performed.

**Verification of this revision.** `git diff --check`, codespell on this SEP and its
roadmap entry, `pytest tests/test_seps.py` (134 passed, one expected process-SEP skip),
and `poe docs` with Sphinx warnings treated as errors. Commands ran through `uv` in
the existing project environment, with synchronisation disabled and a writable
temporary cache. The analytic counterexamples, the monochromatic rate normalisation,
the adjoint-filter work identity and the trapezoidal Fourier multiplier were checked
numerically. The full package, slow and coverage lanes were not run: this revision
changes proposal documentation only.

### Substantial revision, 2026-10-12

- John Travers authorised consolidation into a revised SEP 17 and its return to Draft
  under SEP 0. The earlier acceptance and reviews above remain part of the record.
- Replaced the proposed window-calibrated local rate and separate VUV loss current
  with the explicit weak-excitation continuum candidate in Part I. The first delivery
  is prescribed-field diagnostics and continuum optical feedback with final yield;
  additional plasma needs a separately derived and validated timing/current model.
- Narrowed initial admission to pure Ar/He, fixed linear polarisation and a single-mode
  capillary with physical-node quadrature. Preserved the broader species, mixtures,
  modal and omitted-physics work on the roadmap.
- Retained Q1–Q18 identities, revised their recommendations where the response changes
  them, and added Q19–Q21 for additional plasma, material-contract reassessment and
  continuum cost. Kept the Phase 0 qualification gate before Phase 1 implementation.
- Assessed optmat's existing material inputs: no new OEP for the initial approximation;
  data gaps and a future new material contract remain explicit. No optmat data changed.
- Status Draft; current Part I Not reviewed. The previous reviewers remain recorded
  in the dated history; neither their prior approval nor this drafting work counts as
  review of the revised design.

### AI review of the revision, 2026-10-12

Opus 5.5 read the whole of the revised Part I against SEP 0's acceptance criteria,
optmat OEP 6 and solprop's modal, Raman and Fourier code. Findings were edited into
Part I; none needed the drafting agent's clarification.

- **Checked and correct.** Equation (1) agrees with OEP 6's $K|S_2|^2$ under
  $A=2E^+$, and its monochromatic limit gives $\sigma^{(2)}(I_0/\hbar\omega)^2$. The
  solution (3), the Hamiltonian derivative of (4), the factor four in (5), the work
  identity (6) and the positive-frequency projection of $P_V$ were re-derived. An
  independent DOP853 run reproduced section 7's drafting check: final occupation
  $3.18831632448\times10^{-4}$, Fourier quadrature to $10^{-11}$ and the work
  identity to $10^{-11}$ relative.
- **Transverse factorisation (section 2, Q14).** The claim that an effective area
  cannot reproduce the fourth-power response was too strong for this model. With
  linear selectors, a fixed neutral profile and the frequency-independent Marcatili
  $\mathrm{HE}_{11}$ shape, the response factorises exactly and projects through the
  Kerr overlap. Recorded as an exact reduction and a test; the radial rule stays for
  the pump plasma.
- **Cost and execution route (sections 4.4, 6, Q21).** Added a cost estimate showing
  the direct auxiliary form is about $10^9$ complex updates per right-hand side for a
  representative capillary run. Recommended the existing Raman doubled-grid
  convolution machinery as the expected production route, with the auxiliary
  equations as its oracle.
- **Threshold structure (sections 4.4, 4.5, Q9).** A finite cross section at threshold
  makes the kernel decay as $1/t$ and the reactive coefficient diverge
  logarithmically at the two-photon onset, the RDW-relevant band, where the omitted
  Rydberg and autoionising structure matters most. Added an operational memory time,
  threshold convergence tests and a labelled absorptive-only sensitivity comparison.
- **Omitted pathways (section 4.3).** Stated that $A^2$ coupling keeps only the
  resonant sum-frequency pathway; the nonresonant terms belong to the matching problem.
- **Interface and references.** Renamed the proposed `mode=` keyword to `coupling=`,
  avoiding the waveguide sense of mode. References [3–6] were listed but uncited; each
  is now cited where it is used.

**Evidence and limits.** The derivations and the three-node check were redone
independently; no atomic calculation, kernel benchmark or propagation was run. The
cost figure is an estimate from the stated sizes, to be replaced by Phase 0's
benchmark. The primary atomic-data papers were not re-audited.

- 2026-10-12: Opus 5.5 completed the AI review of the revised Part I (above). The human
  review is recorded for John Travers, who asked on 2026-10-12 for the review status
  to be set to Reviewed. Review status Reviewed; lifecycle status remains Draft
  pending re-acceptance.

### Follow-up AI review and re-acceptance, 2026-10-12

GPT-6 reviewed the revised Part I and the independent review changes, checking the
material support records, Raman and modal implementation, and the cited primary
amplitude paper. The three corrections requested by John Travers were applied:

- **Threshold structure and evidence.** Made the nonzero threshold step conditional,
  derived the signed principal-value logarithm for a finite constant-$k$ interval,
  and distinguished the stationary monochromatic singularity from finite-pulse
  response. Corrected the description of calculated cross sections and retained
  missing/unresolved threshold data as limits on absorption and dispersion alike.
- **Amplitude citation.** Checked [5], section II.2, against its equal-photon-energy
  equations. It supports an amplitude method, not the nondegenerate validation that
  this proposal still requires.
- **Convolution implementation.** Distinguished the exact convolution identity from
  sampled and tapered execution. Raman supplies padding/window methodology; the
  continuum requires complex kernel/drive arrays and transforms. Made taper-induced
  changes to optical work, passivity and convergence explicit in the qualification.

**Evidence and limits.** Independent principal-value quadrature matched the signed
logarithm at six frequencies on either side of the lower edge, with maximum absolute
error $1.6\times10^{-14}$ in a dimensionless constant-k example. A separate synthetic
Fourier-grid calculation checked the absorptive-only work identity (relative error
$2.2\times10^{-16}$) and transverse factorisation with a fixed nonuniform neutral
profile ($1.4\times10^{-16}$). These are algebra checks, not atomic validation or
production tolerances. No new atomic, kernel-performance or propagation calculation
was run. No material number was transcribed; other primary atomic-data papers were
not re-audited by this follow-up review.

- 2026-10-12: GPT-6 completed the follow-up AI review after the corrections above;
  John Travers instructed that SEP 17 be Accepted. The earlier human and Opus 5.5
  reviews remain recorded. Status Accepted; review status Reviewed. Phase 0's
  scientific and numerical qualification gates remain prerequisites for production
  implementation; acceptance does not claim they have passed. Part I is frozen.
