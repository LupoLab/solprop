# SEP 12 — TDSE-driven nonlinear response for gas-filled capillaries

| | |
|---|---|
| **SEP** | 12 |
| **Title** | TDSE-driven nonlinear response for gas-filled capillaries |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; GPT-6 (2026-10-04 revision) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI); Opus 5.5 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-03 |
| **Resolution** | Accepted by John Travers, 2026-10-04, in the implementation-plan discussion |
| **Requires** | [SEP 3](sep-0003-optmat-materials.md); [SEP 13](sep-0013-expensive-response-steppers.md) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | GPT-6 Astra (coordinating agent), with bounded implementation and independent review agents; `feature/sep-0012-calibration`, `feature/sep-0012-integration`, `feature/sep-0012-gpu` |
| **Manual pages** | [TDSE response](../../../guide/media/tdse.md); [configuration](../../../guide/media/tdse.md); [CPU example](../../../guide/media/tdse.md); GPU report (recorded in the development archive); [installation](../../../guide/media/tdse.md) |

## Part I — Proposal

### 1. Abstract

Add solprop-tdse as an optional, batched atomic-response engine for scalar,
carrier-resolved propagation in helium-filled capillaries. Keep optmat's linear
dispersion and initially its electronic cubic polarisation, and replace the
rate-based plasma closure with a TDSE current from which the same discrete
calculation's first and third orders have been removed. Compare the optmat cubic
with a calibrated TDSE third-order dipole. Retain an optional comparison with
unscaled native TDSE nonlinearity using the implemented first-order-only shadow;
this is a different source model, not an equivalent calibrated Kerr shortcut.
Use the fundamental mode on a resolved radial quadrature, with about 16 points
as a cost-reduction candidate. NumPy propagation uses SEP 13's delivered
variable-order Adams PEC method, `Multistep()`, at the maintainer-approved
`rtol=1e-2`; the atoms can independently run on a CPU or GPU. Stepper selection
is settled: this SEP contains no RK propagation comparison, new stepper or
multirate investigation. Select compact current output when no dipole is needed.

Prioritise calibration of the linear and cubic responses and the effective
ionisation electron count, TDSE integration, and a short GPU test propagation.
The whole SEP has a fixed rented-GPU budget of US$100, about 26 device-hours
at the quoted $3.80/h for one MI325X (§6.7). Calibration and integration run
on CPU only. The GPU work verifies the device path and then propagates a
short segment of the canonical case from a saved field. The full-length TDSE
run is not part of this SEP. Start helium ionisation calibration from the hypothesis $N_{\mathrm{eff}}=2$,
keeping it distinct from the linear/cubic calibration and checking its transfer
to the current remainder. Reduced atomic accuracy is permitted for an explicitly
labelled demonstration; retain the calibrated source accounting and report the
remaining error. Spend expensive calls on selected saved waveforms and targeted
checks, not broad method or parameter sweeps. Scientific claims require evidence
for the particular observables, including weak UV emission. Numba and Kokkos
propagation and broad multimode/optimisation studies are not completion requirements.

### 2. Motivation and scope

An extremely self-compressed pulse can contain substantial energy across an
octave or more, have a strongly asymmetric subcycle waveform and drive ionisation
with spectral components far from the input carrier. A rate parameterised by a
carrier frequency and instantaneous field cannot in general retain the coherent
history, excitation and interference of that waveform. This is a reason to test
a TDSE closure, not a claim that every broad pulse invalidates every PPT-like
rate. The comparisons must identify the conditions and observables for which
the additional calculation matters.

The intended application is soliton self-compression and deep-UV generation in
a **constant-radius helium capillary with a longitudinal pressure gradient**.
The first target band extends from the IR to approximately 200 nm. The motivating
capillary physics is described by Travers et al. [3]. The initial model is
scalar, linearly polarised, a single pulse in initially neutral gas, with the
ordinary diagonal capillary linear operator and its existing wall loss.

**Canonical use case supplied by the maintainer.** Use this existing rate-based
propagation configuration as the comparison input, with the subsequently
agreed 120 nm grid edge and Adams selection:

```python
import solprop
from solprop.driver import prop_capillary
from solprop.gasflow import steady_fill
from solprop.stepper.multistep import Multistep

fill = steady_fill("He", [0.0, 4.2 / 2, 4.2], [0.0, 0.2, 0.0], radius=256e-6)
run = prop_capillary(
    256e-6,  # Core radius, m.
    4.2,  # Length, m.
    "He",
    fill,
    lambda0=800e-9,
    fwhm=6e-15,
    energy=1e-3,
    modes=1,
    trange=60e-15,
    wavelength_limits=(120e-9, 1.6e-6),
    rtol=1e-2,
    noise=None,
    n_save=50,
    backend="numpy_ref",
    integrator=Multistep(),
)
```

Pressure arguments are in bar at this existing user interface; the fill peaks
at $2\times10^4$ Pa at its central port. Preserve `steady_fill`'s choked-end
treatment and port breakpoint, rather than substitute a smooth density curve
or literal zero-density endpoints. The initial TDSE grid spans 120 nm–1.6 µm;
the approximately 200 nm scientific target does not qualify the entire
numerical band. The maintainer expects little generated field at the short
edge. Check that truncation's effect on the retained source and observables in
bounded windows; it is an expectation to verify, not an error bound. Physical
extension below 120 nm is deferred, with no mandatory 90 nm production run.
Phase 1 states the admitted material/spectral domain, including any limits
between 120 and 200 nm. Use the supplied `rtol=1e-2`: the maintainer accepts
this accuracy/cost choice with Adams. A tighter Adams run is only a targeted
short-window diagnostic when a specific error question warrants its cost;
there is no mandatory tolerance sweep or RK comparator. This tolerance is
not a guarantee of 1% global or weak-band accuracy.
The snippet uses the existing rate-based response with explicit NumPy/Adams,
not a TDSE adapter. Freeze the remaining resolved defaults when recording
the reference. The
maintainer explicitly selected no noise and 50 save planes, also matching
SEP 13's comparison settings. The rate-based run is cheap on CPU and supplies
the saved fields for every TDSE check. A full-length TDSE version is outside
this SEP (§7.3) and is listed on the [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics); tests,
timings and the GPU test propagation use short segments of this physical case,
started from those saved fields.

The fundamental is `HE_11` in solprop's naming. Its source is evaluated over the
actual radial profile and projected back onto the mode, even when it is the
only retained mode. This is the existing capillary quadrature route with one
mode, not the separate mode-averaged effective-area route. The test propagation is
explicitly a one-mode model. Broader `HE_1m` convergence is later work: a
converged transverse integral alone does not establish negligible modal feedback.

Included are the hybrid response, current-source assembly, atomic execution
lifecycle, pressure gradients, targeted atomic and radial checks, provenance,
diagnostics and qualification of the CPU/GPU atomic path actually used.
SEP 13 is `Final`: this SEP consumes its implementation and tests the coupling's
history, budgets and response identities. Its deferred multirate study moves to
the [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics); it does not delay integration
or the test propagation.

Not included are envelope grids, vector polarisation, tapers, coherent-wall
coupled linear flows, general modal guides, free-space propagation, gas mixtures,
excited or preionised initial states, inter-pulse memory, collisions, multiple
ionisation, or production argon. Propagation into resonant VUV/XUV bands and
complete high-harmonic propagation need additional material and approximation
checks. The TDSE will still generate microscopic high-frequency content, which
must be handled correctly when sampling the response (§4.8).

**Execution is deliberately split.** NumPy propagates the optical
field; solprop-tdse performs the expensive atomic solves, including on a GPU.
There is no need to send the optical propagation through solprop-kokkos to use
a GPU for the atoms. Native integration with solprop-kokkos is a possible later
optimisation with a separate runtime and qualification cost.

### 3. Context

#### 3.1 Current solprop

The inspected solprop revision is `7de4049`. The governing documents are
[architecture](../../architecture.md), [conventions](../../conventions.md),
[the kernel contract](../../kernel-contract.md), [the stepper](../../stepper.md),
[nonlinear responses](../../../guide/media/nonlinear.md) and
[power and intensity](../../../guide/results/power-and-intensity.md).

| Existing component | Consequence for this proposal |
|---|---|
| `modes/quadrature.py` and `translate/modal.py` | Real modal synthesis and weighted projection already support one or several capillary modes and embedded radial rules |
| `nonlinear/responses.py` and `translate/responses.py` | Responses are currently compiled into named coefficient groups; an external atomic response is not an existing arbitrary callback facility |
| `backends/numpy_ref/passes.py` and `backend.py` | The field-grid plasma path integrates current to polarisation before the common spectral polarisation prefactor; this proposal adds a direct current stream |
| `backends/numba_cpu/backend.py` | Existing FSAL execution supplies a cost comparator; a compiled TDSE coupling is deferred |
| `translate/stage.py` | Density and frequency-dependent prefactor weights are evaluated at each stage position; an atomic response must use the same stage density |
| `driver/simulation.py` and `backends/registry.py` | `auto` selects NumPy when `integrator=Multistep(...)` is given; otherwise the existing preference is installed Kokkos, then Numba |
| `stepper/multistep.py` and `backends/numpy_ref/multistep.py` | Delivered variable-order PEC, source-call ledger, budget stop, dense observation and fresh-start resume; no atomic binding yet |
| `output/schema.py` and `translate/problem.py` | Saved-output schema is 9, including `StepperRecord.method`; the kernel schema is independently 16 |

Paths in this table are relative to solprop's `src/solprop/`. The kernel-contract
page's introductory versions still say kernel 15 / output 7, while its version
history and executable schemas say 16 / 9; executable versions are authoritative.
An implementation must use the then-current versions and
reconcile the affected design documentation rather than reserve version numbers
in this Draft.

The default Runge–Kutta integrators remain IP-DOPRI5 and IP-Tsit5, both
fifth-order embedded pairs. Numba normally uses six fresh nonlinear evaluations per accepted step
after startup through first-same-as-last (FSAL) reuse. The NumPy reference
evaluates all seven stages. Save-plane resets, rejection and state-changing
events affect the actual count; six times the accepted-step count is not a
complete cost ledger.

[SEP 13](sep-0013-expensive-response-steppers.md), ADR-036 and ADR-037
deliver the NumPy-only `Multistep()` method: a one-call PEC with a predictor of
order one to five and a corrector one order higher. PECE, plain AB and matched
pairs remain benchmark comparators, not public production choices. Its manual,
[low-call stepping](../../../guide/numerics/adams.md), defines the current
contract; §6.4 below records what this coupling inherits and still has to add.
The six-call Numba figure remains a comparator, not a reason to require a
compiled TDSE coupling phase.

[SEP 6](sep-0006-dispersive-nonlinearity.md) proposes a frequency-dependent
electronic cubic response and is relevant to comparison with TDSE, but is not
a prerequisite: the baseline here is the existing optmat carrier-evaluated
cubic. [SEP 9](../sep-0009-propagation-statistics.md) distinguishes quadrature
resolution and modal-span diagnostics; it does not yet define a general current
source diagnostic. Neither draft is treated as implemented functionality.

#### 3.2 Upstream snapshot and evidence status

The inspected solprop-tdse documentation revision is `3e99e319`; the original
optmat snapshot is `b2f418d`, with the ionisation calibration references
rechecked at `717b0bc` during the human-review revision. The review reconciles the completed stepper with the atomic execution,
strong-field, first-order-shadow and compact-response studies. References
[9–12] identify companion **public documentation**, which carries the API,
measured results and their limits. The proposal does not depend on private
work orders, development paths or another implementation's source.

| Upstream work | Delivered evidence and its scope | Consequence for SEP 12 |
|---|---|---|
| Response execution and projector optimisation | Fused full/shadow RK stages, persistent plans, explicit deterministic or vendor-BLAS projection, CPU and H200/MI325X complete-call measurements at 1/8/30 rows; engineering integration `0603607`, qualified source `de43a8b` | Reuse the public host API and actual row-batch execution; older production-basis timings are not strong-field timings |
| Strong-field convergence | `strong_field` preset and corrected, observable-specific angular/cut/step studies; original runs used `d070208`, analysis `83243a0`, integrated with the execution work | Useful starting configurations; physical-box, broadband and compressed-waveform qualification remain open (§3.3) |
| First-order-only shadow | `shadow_order=1`, with `orders == (1,)`, is implemented; CPU integration/gate `323b7f0`; measured order-3-to-1 CPU savings depend on basis, rows and projection mode | Native retention no longer needs a provider extension; either cubic replacement still needs order 3 |
| Compact response and storage | `ResponsePlan(outputs="total_current")`, typed `TotalCurrentResponse`, shared owned field tables and a fixed device shadow-observation tree; CPU implementation `4b634fe`, merge `36f17c9`, device alignment correction and full CPU gate `beb163e` | Select output by required observables (§5.2); this removes native work and storage, not merely Python columns |
| Latest device follow-up | Corrected H200 and MI325X builds each pass 187 focused Python and 20 GPU cases; native results 141/141 and 124/141 respectively [12] | Order-one/compact numerical paths have actual device evidence; 17 known HIP runtime host-allocation assertions still fail, so no strict allocation-free HIP claim |
| Helium linear and cubic calibration | Public dipole-form calibration and spectral-cut artifact study on the higher-cut basis [9]: $\kappa_1=1.764$ and $\kappa_3=1.537\pm0.002$, each flat to 0.1% or better over 400 nm–2 µm | Keep optmat cubic as the initial baseline and calibrated dipole as the required comparison; native current is a different physical closure. The linear mismatch of §4.4 is already measured in-range; phase 1 reproduces it on the selected basis and extends it below 400 nm |
| Helium static ionisation ratio | Static Stark widths of the SAE ground state against Parker's two-electron rates [14]: $\Gamma_{\mathrm{ref}}/\Gamma_{\mathrm{SAE}}=1.758$ and $1.650$ at 0.10 and 0.15 au ($3.5\times10^{18}$ and $7.9\times10^{18}$ W/m²); Scrinzi's reference gives 1.722 and 1.627 [9] | Existing like-for-like rate evidence for Q1, below the starting value 2 and falling with field (§4.5); not a laser-field, few-cycle or current comparison |
| Same-potential one-electron yields | Curated tRecX yields of one electron in the same Tong–Lin helium potential, velocity gauge, 800 nm, 6 fs, $2.4\times10^{18}$–$2.1\times10^{19}$ W/m² [10] | An independent same-model check of the SAE engine's ionisation, separating code error from SAE model error before any electron-count comparison; yields there subtract bound populations up to $n=10$, which `outside` does not (§4.2) |

The latest device chapter [12] supersedes the older shadow/backend-guide
statements that order one or the fixed tree lack any device execution. It
does **not** qualify a one-femtosecond compressed pulse, establish warmed
full-window timing distributions or provide portable device wheels. The GPU
scratch-alignment fix must be present in the admitted build. Historical timings
retain their actual source identities, including the MI325X timing before
that fix and its separate post-fix numerical checks.

CPU and GPU full/compact common outputs agree under the provider's documented
arithmetic contract. The fixed device tree may change shadow observations in
their last bits relative to the old ordered sum; vendor projection has its
own derived rounding allowance. Neither is a cross-device bitwise guarantee.
Qualify the **subtracted** source in absolute units as well as its operands.
Refresh the capability/build/evidence matrix in each implementation report.

#### 3.3 What the strong-field study does and does not qualify

The reference pulses were 6 fs at 800 nm and 3 fs at 200 nm, not the intended
broadband, approximately 1 fs compressed field. The following differences are
reported at a common physical box, relative to the study's finer angular/cut/
step reference [9]. Intensities here are converted to SI W/m².

| Reference case | Finding | Starting point for phase 1, not an accuracy guarantee |
|---|---|---|
| 800 nm, $5\times10^{18}$ W/m² | `production` has about 6–7% angular error in total current and 22% in total acceleration; increasing only the cut leaves the angular defect | At least $l_{\max}=49$ with the higher cut; the recorded worst fixed-box difference there is $1.6\times10^{-3}$ |
| 800 nm, $10^{19}$ W/m² | `strong_field`, $l_{\max}=79$, has a worst fixed-box difference $8.4\times10^{-3}$; completed-current difference is $2.8\times10^{-4}$ | Start from `strong_field`; $l_{\max}=99$ is only a starting point for a $10^{-3}$ target, with further angular refinement required |
| 200 nm, up to $2\times10^{18}$ W/m² | Angular extent is already converged at $l_{\max}=19$ in the measured cases; cut/step effects dominate | Do not infer the IR-driven broadband requirement from this isolated UV pulse |

The preset's resolved identity is 80 angular channels and 266 radial nodes,
with $R_0\simeq4.233\times10^{-9}$ m, spectral cut
$E_{\rm cut}\simeq1.744\times10^{-15}$ J and maximum step
$h_t\simeq1.260\times10^{-19}$ s. These rounded SI equivalents correspond
to the provider's 80-bohr, 400-Hartree, 1/192-atomic-time-unit settings, converted
with its public unit scales; implementation must retain the exact resolved
configuration, not reconstruct it from these rounded values.

The preset's $8.4\times10^{-3}$ figure **excludes physical-box error**. Changing
$R_0$ from approximately 4.23 to 7.41 nm changes the strong 800 nm total current
by about 0.9–1%, and total acceleration by 2.4–6.7%. At the lower strong-field
case, successive acceleration differences do not decrease. The neglected-
outer-force current bound does not bound all such discretisation differences,
acceleration or harmonic spectra. A larger box must be studied together with
angular, radial and absorber resolution before assigning a source error.

The first/third shadows are unchanged under the reported angular refinement
but still depend on the cut. Their total-current step studies do not establish
fourth order for every observable: the third-order dipole gives an order near
1.46 on one triple, then a nonmonotone difference. Lost norm can instead show
the fifth-order accumulated RK4 damping floor. Qualification must address
each **consumed** current/dipole and the cancellation-sensitive remainder;
neither a norm monitor nor a preset's single error number suffices. Phase 1
uses this evidence to choose targeted screens and state the reduced-accuracy
test propagation's limits; it need not repeat every upstream ladder. The
published ladders have incomplete radius coverage and are not an independently
converged all-electron reference.

### 4. Theory

#### 4.1 Coordinates, units and approximation

Use $r$ for the macroscopic capillary radius coordinate, $z$ for propagation,
$t$ for retarded time, and $\boldsymbol{x}$ for the electron coordinate inside
one atom. The atom sees the local real field in V/m, with its polarisation axis
identified with the atomic solver's single driving axis. Atomic position and
capillary radius are independent discretisations.

For an electron of charge $-e$, a local SAE potential and a spatially uniform
atomic driving field give the velocity-gauge Hamiltonian

$$
i\hbar\partial_t\psi=
\left[\frac{(\boldsymbol{p}+e\boldsymbol{A}(t))^2}{2m_e}
      +V_{\mathrm{SAE}}(|\boldsymbol{x}|)\right]\psi,
\qquad E(t)=-\partial_t A(t).
$$

This is the electric-dipole, nonrelativistic, fixed-nucleus model. A solver may
remove the spatially uniform $A^2$ term by a time-dependent global phase; the
observable and gauge conventions must still match [9]. Helium uses the named
Tong–Lin SAE model [4], obtained from the public provider with its complete
identity. Solprop does not duplicate the potential coefficients or atomic
unit constants.

Each spatial point begins in the same prepared stationary ground state before
the pulse. There is no wavefunction transported along $z$: each plane contains
different atoms. Resetting between calls is therefore physical for this model.
Within a call, the full waveform drives one coherent temporal evolution. An
Adams source evaluation uses its trial optical field for a complete atomic solve.

Pressure and temperature determine the initial number density $n_0(z)$ in
m$^{-3}$ through the existing equation of state. The same fill determines the
linear gas index. At constant bore radius, the scalar capillary profiles can
remain fixed while dispersion, loss, amplitude power weights and response
density vary along $z$. The existing cumulative linear operator, including its
normalisation term, remains authoritative.

#### 4.2 Which microscopic current is usable

The in-box dipole derivative is not the current of all electrons after the
absorber removes outgoing probability. Let $N_0$ be the measured initial norm,
$N_{\mathrm{in}}(t)$ the norm inside atomic radius $R_0$, and
$F_{\mathrm{in}}=\langle-\hat{\boldsymbol{\epsilon}}\cdot\nabla V\rangle_{|\boldsymbol{x}|<R_0}$
in newtons, with $\hat{\boldsymbol{\epsilon}}$ the driving axis; $j$ and $E$ below
are components along that axis.
The completed charge-dipole acceleration used by the provider is

$$
\dot j_{\mathrm{tot}}(t)
=-\frac{e}{m_e}F_{\mathrm{in}}(t)
 +\frac{e^2}{m_e}E(t)N_0,
\qquad j_{\mathrm{tot}}(t_0)=0.
$$

It restores the field acceleration of escaped probability once, while neglecting
the binding force outside $R_0$. Its corresponding neglected-current bound is

$$
|\delta j(t)|\le\frac{e}{m_e}
\sup_{\rho\ge R_0}|V'(\rho)|\int_{t_0}^t
\left[N_0-N_{\mathrm{in}}(\tau)\right]d\tau.
$$

Here $\rho=|\boldsymbol{x}|$ and $V'$ is the radial derivative of the
spherical SAE potential. This bounds one omitted force, not every
discretisation or absorption error.
The numerical basis and $R_0$ must also resolve the important returning
trajectories. The provider reports the bound and the underlying diagnostics [9].

Use `GridResponse.plasma.total_current`, or `TotalCurrentResponse.total_current`
on the compact path, in C m/s per represented atom.
`response.current` is an in-box velocity-form current. The diagnostic
`inbox_current` and `free_current` are split according to where acceleration
occurred, not according to a partition into permanently bound and permanently
free electrons. Neither part alone supplies the required macroscopic current.

Likewise, `outside` is missing probability at a specified atomic boundary. It
is not automatically a converged ionisation probability, neutral depletion, or
physical electron density. Record its meaning and radius. A physical density
statistic needs independent calibration; it must not inherit the old plasma
statistic's name merely because the array has a similar shape.

#### 4.3 Discrete perturbative subtraction

For a real amplitude parameter $\lambda$, inversion symmetry gives the
weak-field expansion of the **discrete** response map

$$
j[\lambda E]=\lambda j_1[E]+\lambda^3j_3[E]
             +\lambda^5j_5[E]+\cdots.
$$

The shadow propagates those coefficients beside the full solution, using the
same initial state, finite basis, spectral projector, absorber, RK steps,
sample grid and current integration. The quantities returned as orders 1 and
3 already include their expansion coefficients; no extra factorial is applied.
For the two cubic-replacement models, set

$$
j_{\mathrm{rem}}[E]=j_{\mathrm{tot}}[E]-j_1[E]-j_3[E],
\qquad J_{\mathrm{rem}}=n_0(z)s_{\mathrm{rem}}j_{\mathrm{rem}}.
$$

The subtraction removes the first and third Taylor orders to rounding; it
does not remove higher-order basis errors or make the finite basis exact.
For sufficiently weak fields the residual scales as $\lambda^5$ until
subtraction noise dominates. Its rounding scale depends on the magnitudes of
the three operands, not on the much smaller remainder. Tests need an absolute
floor derived from those operands rather than an impossible relative bound.

Subtract from the uncorrected total current. Its first order includes the
spectral-cut and discrete sum-rule defects. Correcting the total first, then
subtracting the uncorrected shadow, introduces a mismatch. Subtracting an
idealised linear susceptibility in place of the discrete shadow similarly
leaves discretisation defects which can compete with the desired source.

No separate rate-based plasma, Brunel current, ionisation-loss current or
additional electron-force term is added for these electrons. Their response is
already in the completed total. Density is the initial atom density, not an
extra neutral fraction multiplying a response which already includes depletion.

#### 4.4 Replacing linear dispersion and comparing Kerr models

Linear dispersion belongs in $D_m(\omega,z)$, using optmat. The nonlinear
callback supplies **no restored linear current**. Two cubic replacements are
compared on top of exactly the same TDSE remainder.

**Calibration is a primary deliverable.** Before interpreting a propagation,
measure the weak-field dipole response $p_1$ of the selected one-electron model
against optmat's per-atom linear polarisability over the admitted band. Record
its amplitude, phase and frequency dependence; a diagnostic ratio
$\kappa_1(\omega)=\alpha_{\mathrm{ref}}(\omega)/\alpha_{\mathrm{SAE}}(\omega)$
is meaningful only where the denominator is resolved. It diagnoses the SAE
linear mismatch; the actual coupling still subtracts the exact discrete $j_1$
and uses optmat in the linear operator. Do not multiply that shadow by
$\kappa_1$ before subtraction. Calibrate the cubic independently as below,
then calibrate ionisation (§4.5). Store the three results and their ranges in
one calibration record, with the atomic potential, basis, temporal discretisation,
reference density and units. One electron-count factor does not calibrate all
three responses. Reuse upstream calibration evidence where the configuration
matches, and spend new calls on the missing comparisons. The linear comparison
already exists in range: on the higher-cut basis $\kappa_1(\mathrm{He})=1.764$,
within 0.1% from 400 nm to 2 µm [9]. Phase 1 therefore reproduces it on the
selected basis, records the cut dependence of $j_1$ there, and extends the
comparison below 400 nm, where the band's short edge lies.

**Baseline, `optmat_cubic`.** Use the existing full real-field cubic, with
third-harmonic generation retained:

$$
P_{3,\mathrm{ref}}(t,z)=\epsilon_0\chi^{(3)}(z)E(t)^3
=\frac{n_0(z)\gamma_{\mathrm{ref}}}{6}E(t)^3.
$$

The final equality uses optmat's Taylor-series hyperpolarizability convention
$p=\alpha E+\gamma E^3/6$ [10]; $\gamma$ here is in SI units. Obtain the
coefficient through solprop's existing adapter, not another unit conversion.
The existing carrier-evaluated coefficient is an approximation for broadband
mixing. A real $E^3$ includes all cubic frequency combinations but gives them
the same coefficient. The UPPE frequency factor supplies the propagation shock
factor, not the omitted material dispersion.

**Comparison, `tdse_shadow`.** Use the third-order charge dipole,

$$
P_{3,\mathrm{ref}}(t,z)=n_0(z)\kappa_3 p_3[E](t),
\qquad p_3=\texttt{shadow.component("charge\_dipole", 3)}.
$$

It enters as a polarisation directly. Its associated current is
$\partial_tP_{3,\mathrm{ref}}$ in the source accounting, but numerical
differentiation is unnecessary. Do not replace it by $\kappa_3j_3$.
The discrete current and the derivative of the dipole can differ through
spectral-cut artifacts even at third order.

The upstream calibration [9] reports $\kappa_3=1.537\pm0.002$ for helium from
a higher-cut dipole calculation against optmat, with near-constant Kerr ratios
over 400 nm–2 µm. This is a reported candidate, not a new measurement in this
SEP. The same documentation reports approximately 0.7% and 4% dipole-form
cubic artifacts on the production basis at 800 nm and 2 µm respectively, and
a much larger current-form defect, including a wrong sign at long wavelength.
Increasing the basis cut is therefore part of qualifying this option.

$\kappa_3$ already matches a per-atom reference hyperpolarizability. It is
independent of $s_{\mathrm{rem}}$ and must not be multiplied by that factor.
First reproduce the carrier Kerr calibration and selected THG/short-pulse
checks on a small waveform set, reusing compatible upstream results. Broader
frequency and waveform comparisons are required only for the wider scientific
claims they support: one degenerate Kerr coefficient cannot certify the full
third-order response functional.

The optmat nonlinear fit documents a fundamental range beginning at 400 nm
and ending at 20 µm [10]. Neither an extrapolated comparison at 253 nm nor
linear-index validity to shorter wavelengths validates arbitrary mixing down
to 200 nm. Phase 1 must state the admitted input-frequency combinations and
the support of the replacement, then qualify or explicitly limit the deep-UV
claim. No frequency-by-frequency switch to a different Kerr model is inserted
without a causal, non-overlapping derivation. Nonlinear absorption at lower
order near thresholds is a further reason the present substitution cannot be
extended unchanged into the VUV.

Upstream evidence narrows, but does not close, this question for the
degenerate Kerr coefficient: below 400 nm the helium SAE Kerr ratio stays within
0.6% of optmat's extrapolation to 253 nm, and within 0.4% of Bishop and Pipin's
calculated data [9]. Third-harmonic generation is different: the SAE's
third-order dipole approaches its own $3\omega$ resonances there, and no THG
ratio is reported at 253 nm. Real helium's low-order resonances fall inside or at
the edge of the admitted band. The degenerate two-photon
$1s^2\to1s2s\,{}^1S$ resonance (20.62 eV) corresponds to a 120.3 nm fundamental,
at the grid edge. The degenerate three-photon $1s2p\,{}^1P$ (21.22 eV) and
$1s3p\,{}^1P$ (23.09 eV) resonances correspond to 175 nm and 161 nm, and
three-photon ionisation (24.59 eV) opens below 151 nm. Non-degenerate
combinations of in-band frequencies reach the same states. The SAE places its resonances at its own energies, which phase 1 takes
from the prepared spectrum rather than assuming helium's. These positions
bound where `tdse_shadow`'s dipole and the remainder's fifth-order terms can
be compared with a resonance-free reference cubic. Phase 1 lists them in the
admitted domain.

##### Optional native cubic retention with only the first-order shadow

The full TDSE response already contains its native cubic term. If that term is
retained without a separate calibration or replacement, linear subtraction is
sufficient for the accounting:

$$
J_{\mathrm{native}}=n_0(z)(j_{\mathrm{tot}}-j_1),
\qquad P_{3,\mathrm{ref}}=0.
$$

Call this research option `tdse_native`. Use the same uncorrected completed
current and its exact discrete first order; optmat still supplies linear
dispersion. Do not add optmat Kerr, a third-order dipole, or another plasma
term. In §4.6 the current stream is $J_{\mathrm{native}}$ instead of
$J_{\mathrm{rem}}$, with zero replacement polarisation. The retained nonlinear
response starts at cubic order in a weak-field amplitude ladder.

Use the delivered provider `shadow_order=1` capability. The
triangular shadow recursion closes on $c_0,c_1$; the helium s-state packed
representation then needs two shadow channels instead of the two-plus-four
channels for orders through three. The full TDSE state is unchanged. This
removes higher-order shadow work and observations, not a fixed fraction of
the complete call. The completed CPU comparison found about 11–14% savings
on its production-basis cells and about 5% at 30 strong-field rows, not a
two-thirds saving; §6.7 records the measurement scope. Device qualification
of order one now exists [12], but its order-3-to-1 cost difference at qualified
compressed-waveform settings still needs a matched comparison.

The distinction from `tdse_shadow` is essential. In current accounting notation,
the calibrated hybrid can be written as

$$
J_{\mathrm{hybrid,NL}}=
n_0s_{\mathrm{rem}}(j_{\mathrm{tot}}-j_1)
+n_0\left(\kappa_3\partial_t p_3-s_{\mathrm{rem}}j_3\right).
$$

The second bracket still needs third-order information. Equal multipliers do
not make it vanish on a finite basis: $j_3$ and $\partial_t p_3$ can differ by
the documented spectral-cut artifact. Even setting $\kappa_3=1$ in the dipole
replacement is therefore not generally the native-current model. The
derivative above explains the accounting; the implementation retains separate
polarisation/current streams and never differentiates the sampled dipole here.

For the initial native comparison require `remainder_scale=1.0` and no
`cubic_scale`. Applying a multiplier to $j_{\mathrm{tot}}-j_1$ would scale
cubic, higher-order and plasma response together, which is a new model. The
helium ionisation starting factor 2, optmat rate factor 1.47 and dipole factor
near 1.537 do not establish that model. Leaving the SAE cubic unscaled also
leaves its mismatch to the reference
helium nonlinearity; numerical convergence cannot remove that model uncertainty.

Native retention also retains the third-order current artifact which the two
replacement models subtract. Gate its use on current-level cut/basis and
temporal convergence over the actual broadband waveform, retained-band work,
and comparison with both replacement models. A higher cut can force smaller
atomic steps and erase the shadow saving. Compare costs at each model's
qualified settings, with model uncertainty reported separately. Native
retention is optional and does not delay the calibrated baseline or the test propagation.
The calibrated-dipole comparison can be made on saved atomic waveforms; it
does not require a second full propagation.

#### 4.5 Remainder scale and strong-field consistency

For helium, start from the maintainer's $N_{\mathrm{eff}}=2$ hypothesis for
converting a one-electron ionisation response to the atom's first-ionisation
response. This is the first candidate to test, not a reason to repeat an
unfocused search over arbitrary scales. Parker et al. provide two-electron
laser-field rates at 390 and 780 nm [13] and static-field rates [14]. Their
factor-of-two statement in [13] concerns the cycle-averaged ADK electron count;
it does not by itself establish a universal factor for this SAE potential or
its broadband current. Optmat's separate factor 1.47 fits its chosen analytic
rate formula to the static data [10]. Its denominator is not this TDSE solver,
so 1.47 is contextual evidence, not the starting TDSE scale. Do not transfer
helium's candidate to argon or other gases.

There is already direct static evidence for this SAE potential. The ratio of
Parker's two-electron static rate [14] to the SAE ground state's Stark width
is 1.758 at 0.10 au and 1.650 at 0.15 au ($3.5\times10^{18}$ and
$7.9\times10^{18}$ W/m²), where a few-cycle pulse starts to ionise helium
appreciably. Scrinzi's static helium rates give 1.722 and 1.627 [9].
The bound responses give similar ratios, $\kappa_1=1.764$ and
$\kappa_3=1.537$ (§4.4). The factor two is the weak-field leading-order electron
count of a closed $s$ shell [10]; at these fields the atom's rate is below twice
this SAE rate, and the ratio falls between the two fields. The starting value 2 is therefore
expected to overstate this ionisation by roughly 14–21%, before any laser-field,
few-cycle or coherent-current effect. Phase 1 compares the declared 2 with this
evidence and with Parker's 780 nm laser-field rates [13] before spending atomic
calls on a separate scale search. A field-dependent ratio is a further reason
why a single constant $s_{\mathrm{rem}}$ is a compromise, whose error over the
reached intensities must be stated.

Distinguish the measured ionisation ratio

$$
N_{\mathrm{eff}}^{(\Gamma)}=
\frac{\Gamma_{\mathrm{He,reference}}}{\Gamma_{\mathrm{SAE}}}
$$

from $s_{\mathrm{rem}}$, the multiplier of a coherent current remainder.
Rates are in s$^{-1}$; the ratio is dimensionless. Compare like definitions
and driving fields: Parker's steady laser-field rates are not few-cycle
yields, and 780 nm is not 800 nm. Use optmat's curated fits/tables rather than
redigitising them. For pulse benchmarks compare depletion or first-ionisation
probability on the same waveform and time interval. In a rate description,
$Y=1-\exp[-\int\Gamma(t)dt]$ shows why multiplying a yield by two is only a
low-depletion approximation. This relation is a diagnostic comparison, not
an extra rate closure in the TDSE source. $N_{\mathrm{eff}}=2$ is not two
electrons released per first-ionisation event or permission to exceed the
single-ionisation model's depletion range.

The initial cubic-replacement research configuration therefore declares
`remainder_scale=2.0` explicitly, labelled as the helium starting hypothesis.
Phase 1 checks its ionisation evidence and the amplitude, phase and work of
the resulting current; use the unscaled result as a diagnostic control from
the same atomic solve. Changing $s_{\mathrm{rem}}$ in source assembly does not
require rerunning an identical atomic waveform. A constant scale is retained
only within the supported range, with remaining model uncertainty recorded.
Do not infer a sample-wise current multiplier from a time-dependent rate ratio.
If the comparison rejects 2 or any constant, record the evidence and revise
the calibration explicitly. Keep $\kappa_1$, $\kappa_3$ and
$s_{\mathrm{rem}}$ independent and apply each physical normalisation once.

There is a separate saturation issue even if all numerical errors vanish.
If the full TDSE current approaches a predominantly free-electron response,
the hybrid current formally contains

$$
J_{\mathrm{hybrid}}\simeq n_0s_{\mathrm{rem}}j_{\mathrm{free}}
+\left(J_{1,\mathrm{ref}}-n_0s_{\mathrm{rem}}j_1\right)
+\left(J_{3,\mathrm{ref}}-n_0s_{\mathrm{rem}}j_3\right).
$$

The bracketed terms need not vanish when the reference linear and cubic
responses were calibrated independently. Before saturation, $j_{\mathrm{rem}}$
also contains the SAE's bound fifth- and higher-order response, which
$s_{\mathrm{rem}}$ scales as if it were ionisation. With $\kappa_1$ and
$\kappa_3$ near 1.5–1.8, there is no reason for the atom's bound fifth order to
be exactly $s_{\mathrm{rem}}$ times the SAE's. The weak-field ladder required by
§4.3 gives the perturbative fifth-order coefficient $j_5$ of the saved
waveforms. Report its estimate at the actual amplitude beside the remainder,
so that the scale's effect on the bound and the ionising parts is visible. Thus correct weak-field replacement
does not prove correct depletion of the bound response at high field. The
hybrid is a proposed calibrated model, not an algebraic route to an exact
all-electron response. Qualification must test the bound-response cancellation,
source work and saturation over the admitted range. If this fails, restrict
that range or return the closure to scientific review; do not conceal the
failure with an empirical neutral mask.

#### 4.6 Direct current source and signs

For the internal convention, a temporal derivative transforms as $+i\omega$,
and a forward plane wave has factor $\exp[i(\omega t-\beta z)]$. The transverse
wave equation has source $\mu_0\partial_t^2P+\mu_0\partial_tJ$.
Factoring out the forward linear solution and neglecting its second derivative
in $z$ gives [1]

$$
-2i\beta\partial_z E=-\mu_0\omega^2P+i\mu_0\omega J,
\qquad
\partial_zE\big|_{\mathrm{source}}=
-\frac{i\omega}{2\epsilon_0cn}P-\frac{1}{2\epsilon_0cn}J.
$$

Use the real effective index in the source denominator, as solprop already
does; wall loss belongs in $D$. The optics-convention polarisation coefficient
has the opposite imaginary sign, while the real negative current coefficient
is unchanged. For $J=\sigma E$ with real positive conductivity, the source
attenuates the field, a useful independent sign witness.

For the modal state, let $\mathcal Q_m$ be the existing normalised projection:

$$
\partial_z\widehat a_m=D_m\widehat a_m+N_m,\qquad
N_m=-\frac{i\omega}{2\epsilon_0c\,\mathrm{Re}\,n_{\mathrm{eff},m}}
       \mathcal Q_m[\widehat P_{3,\mathrm{ref}}]
    -\frac{1}{2\epsilon_0c\,\mathrm{Re}\,n_{\mathrm{eff},m}}
       \mathcal Q_m[\widehat J_{\mathrm{rem}}].
$$

Crop/pad scales and the declared spectral windows remain separate numerical
factors. Build the current prefactor directly; do not divide the polarisation
prefactor by $i\omega$, especially at zero frequency. The existing optical
active band excludes DC. Recording a microscopic residual DC current does not
mean this optical model propagates an electrostatic field.

For a smooth compact polarisation, $\widehat{\partial_tP}=i\omega\widehat P$
makes the two source paths identical. With a finite window the boundary term
must be kept. If a taper $W(t)$ is applied,
$\partial_t(WP)=W\partial_tP+W'P$; independently tapering $P$ and $J$ does not
preserve their equivalence. The taper belongs in the numerical contract and
its effect must converge away in the physical observation window.

#### 4.7 Modal normalisation and radial integration

For real scalar modal profiles $e_m(r)$ define

$$
\mathcal N_m=\frac{\epsilon_0c}{2}\int |e_m|^2dA,\qquad
S_m(r)=\frac{e_m(r)}{\sqrt{\mathcal N_m}},\qquad
E(r,t)=\sum_m S_m(r)a_m(t).
$$

Here $a_m(t)$ is the real time-domain internal amplitude on the field grid.
Its analytic-signal power convention, and the true-power interface amplitude
$b_m(\omega)=\sqrt{\mathrm{Re}\,n_{\mathrm{eff},m}}\,a_m(\omega)$, are those of
[conventions](../../conventions.md). Do not apply that power conversion again
before the atomic calculation.

For either physical source $X=P$ or $J$ the quadrature projection is

$$
\mathcal Q_m[X]=\frac{\epsilon_0c}{2}
\sum_q w_q S_m(r_q)X(r_q),\qquad
dA=2\pi r\,dr.
$$

The weights already contain the area Jacobian; the matrices already carry
their normalisation. Using the existing synthesis/projection matrices and
testing their units avoids a second $2\pi r$, density or effective-area factor.
The mode normalisation remains the independent analytic one, not a value
renormalised on the candidate quadrature to force a perfect Gram matrix.

At one mode, all nodes have the same temporal waveform scaled by $S_1(r_q)$.
The projected TDSE response still is not the response at one representative
intensity. At several modes, dispersion creates different local temporal
shapes and phases, so even a waveform-specific amplitude interpolation from
the one-mode case cannot be reused indiscriminately.

#### 4.8 Three time/frequency resolutions

Distinguish the optical propagation band, the response output spacing
$\Delta t_{\mathrm{resp}}$, and the atomic RK step $h_t$. On the current public
API [9], an integer subdivision makes every parent sample an RK endpoint:

$$
h_t=\Delta t_{\mathrm{resp}}/k\le h_{\max},\qquad
N_{\mathrm{atomic\ steps}}=(M-1)k.
$$

The public provider currently expresses the maximum step in atomic units.
Solprop's proposed interface accepts seconds and performs that conversion once
using the provider's unit scale. Tightening $h_t$ alone does not improve the
fourth-order interpolation of the parent driving field or the output sampling
and current integration on that parent lattice.

TDSE response can contain harmonics and coherences above the propagated
optical band. The existing cubic oversampling rule is not a general aliasing
guarantee for that response. A fine atomic march which returns coarse samples
can alias those components into the retained band **before** solprop applies
its spectral mask. Masking afterward cannot repair that aliasing.

The baseline uses a common, explicitly resolved fine grid for the drive and
returned response, with refinement until the retained source converges.
The atomic cost is set mainly by $h_t$, which the spectral cut already bounds
(about $1.26\times10^{-19}$ s for `strong_field`), and not by the parent
spacing. The recorded strong-field workload used a parent spacing of about
$3.0\times10^{-18}$ s (§6.7), whose Nyquist frequency is about 680 eV in
photon energy. The helium cutoff $I_p+3.17U_p$ at $10^{19}$ W/m² and 800 nm
is about 210 eV. A parent spacing whose Nyquist frequency clears the cutoff of
the saved waveforms, with margin, is therefore the preferred baseline. Estimate
that cutoff from their peak field and longest significant wavelength. Aliasing
is then expected to be negligible, which the refinement above confirms, and
the solprop fine grid it implies
(about $2\times10^4$ samples over the canonical window) is cheap compared with
one atomic call. A
future cheaper output path must low-pass filter at the resolved lattice before
decimation; that capability must not be assumed to exist in the current
`ResponsePlan`. The comparison must also widen the propagated band where
feasible to test feedback from generated light, independently of output
sampling. Discarding that light is a model truncation, not just an FFT detail.

Supply chronological samples, even though the optical grid labels time about
the pulse centre. A translation of the time origin does not change the samples
or CEP. The electric-field route sets $A(t_0)=0$ exactly; it does not force
$A(T)=0$. Require a sufficiently quiet leading interval for a ground-state
start. Do not remove pulse area, shift the carrier or subtract mean current
to make a test pass.

A quiet electric-field tail need not be a quiet response tail: freed electrons
retain drift and bound coherences persist. Converge window length and any
source taper, with an interior observation region. A shortened atomic window
or analytic field-free continuation is a later, separately bounded
optimisation, not a default shortcut.

#### 4.9 Source work and error budgets

The physical source work per unit propagation length is

$$
W_{\mathrm{NL}}(z)=\int dt\int dA\,
E(r,t,z)\left[\partial_tP_{3,\mathrm{ref}}+J_{\mathrm{rem}}\right],
$$

in J/m. Compare it with the change of pulse energy after separately accounting
for linear wall loss, absorbing collars, retained-band flux and numerical
error. Use solprop's analytic Poynting energy functional, not a new broadband
intensity convention. Reactive bound response can return energy temporarily;
neither an individual source component nor its instantaneous work must be
positive. The hybrid is not a closed Hamiltonian model, so an exact atomic
energy identity cannot simply be asserted after replacing its response orders.

Keep a numerical budget with distinct contributions from the atomic basis,
atomic step, response lattice/window, radial quadrature, modal span and
longitudinal integration. SAE error, reference-material uncertainty and
calibration uncertainty are separate model errors. The z-step error estimator
does not measure the first five numerical contributions. Tightening its
tolerance below a noisy source floor can multiply atomic calls without
improving the answer.

### 5. Proposed interface

#### 5.1 Scientific specification and execution choice

Introduce a frozen `TDSEGasResponse` in the nonlinear response layer. The names
below are proposed interfaces, not existing functionality. Initially compose it
through `Simulation`'s explicit response sequence, which makes ownership of the
gas response unambiguous. A capillary convenience keyword can follow once the
validation and exclusivity rules are shared with that constructor.

| Specification | Meaning and validation |
|---|---|
| `target` | `"helium"` initially; a named SAE potential and ground-state identity, not just a chemical label |
| `basis` | Frozen, serialisable atomic configuration, including a named preset if used and all its resolved numerical values; a preset name alone is insufficient |
| `max_atomic_step` | Positive finite time in seconds; translated to the provider's unit convention and recorded with the actual subdivision count |
| `kerr_model` | `"optmat_cubic"` initially recommended; `"tdse_shadow"` is the required comparison; `"tdse_native"` is the gated unscaled native-current comparison |
| `remainder_scale` | Required positive finite dimensionless $s_{\mathrm{rem}}$ for cubic replacement; start helium calibration explicitly at 2.0 (§4.5); must be 1.0 for `"tdse_native"`; no hidden occupation multiplier |
| `cubic_scale` | Required positive finite $\kappa_3$ for `"tdse_shadow"`; absent for `"optmat_cubic"` and `"tdse_native"` |
| `calibration` | Identity, basis, spectrum, uncertainty and applicable range of the selected calibration; an explicit unqualified research configuration is distinguishable from a qualified one |
| `sampling` | Frozen response-grid and window policy, resolved to physical spacing, number of samples, interpolation and tail treatment; the baseline shares the declared fine field grid |

The initial state is always the prepared ground state. Translation requests
shadow order three for either cubic replacement, or order one for
`tdse_native`. Admission checks that the installed build supplies the required
orders and selected result contract. Do not silently
downgrade a replacement model to first-order subtraction or change physics to
save time. Qualification may explicitly run native retention with order three
as an oracle, ignoring its third-order output, and compare it with order one;
this does not make order-three execution the cheap path. The spec owns the
complete electronic nonlinear response, including its selected cubic policy.
It refuses an accompanying legacy `KerrResponse` or `PlasmaResponse` for the same
gas. The real-field cubic includes third-harmonic generation; an envelope or a
fundamental-only cubic approximation is not an equivalent option.

Execution controls belong to a separate frozen `TDSEExecution` configuration:
CPU or a named available device/build, device identity, memory limit, maximum
batch rows, an optional positive `max_atomic_calls` limit (§6.4), and
`complement="deterministic"` or explicit `"vendor_blas"`.
Deterministic projection is the initial choice; preserve the selected provider
label and its arithmetic contract. A vendor mode must be available, qualified
and measured for the actual batch/basis; no automatic switch follows from an
older device ranking. Controls cannot change the scientific model silently.
The proposed sketch uses
already constructed field grid, capillary modes, pulse and density profile:

```text
response = TDSEGasResponse(
    target="helium",
    basis=qualified_helium_basis,
    max_atomic_step=qualified_step_seconds,
    kerr_model="optmat_cubic",
    remainder_scale=2.0,
    calibration=helium_starting_calibration,
    sampling=qualified_sampling,
)
settings = SimulationSettings(
    ...,
    backend="numpy_ref",
    integrator=Multistep(max_calls=response_budget),
    rtol=1e-2,
    tdse_execution=TDSEExecution(
        device="cuda",
        memory_limit=budget_bytes,
        max_atomic_calls=atomic_budget,
    ),
)
simulation = Simulation(
    grid=field_grid,
    modes=[fundamental_capillary_mode],
    responses=[response],
    initial=pulse,
    settings=settings,
    profiles=density_profiles,
)
```

Here `2.0` is the explicit helium starting hypothesis from §4.5, with its
calibration status recorded; it is not an implicit multiplier on the cubic.
The actual CPU/device vocabulary must match the qualified provider API; the
adapter must not invent an unsupported device-selection mechanism. Translation
checks material identity, constant bore, scalar field grid, modal family and
sampling consistency before assembling expensive atomic operators.

Use explicit `backend="numpy_ref"` and `integrator=Multistep(...)` for TDSE.
Preserve SEP 13's existing `auto`-to-NumPy resolution with that integrator.
The initial TDSE route requires Adams and refuses an incompatible integrator
before atomic setup. RK propagation support and comparisons are outside this
SEP; existing non-TDSE RK behavior stays unchanged. Explicit Numba and Kokkos
requests fail by name. Atomic CPU/GPU choice remains independent of this flag.

Refuse envelope grids, vector fields, tapers, mixtures, excited/preionised
states, inter-pulse history, argon and unsupported geometries by name. Retain
`Multistep`'s refusals of coupled flows, modified noise, fixed schedules and a
non-DOPRI5 tableau; use `noise=None` for the initial qualification. Refuse
missing provider capabilities, incompatible schema versions and unavailable
devices at preparation. No fallback to PPT or a different atomic basis is
permitted after admission.

#### 5.2 Translation, runtime binding and results

Translation resolves the specification into an immutable atomic descriptor
and the numerical source factors. `KernelProblem` carries this descriptor and
its identity, not a live `ResponsePlan`, Python physics object or native pointer.
The versioned execution contract gains a narrow host response binding, passed
explicitly when constructing a driver. Its minimal responsibilities are to
prepare a configuration, evaluate real host waveforms, report memory/capabilities
and close resources. It is a declared atomic service, not an unversioned
arbitrary nonlinear callback. The adapter normalises these public provider
result types into that one binding:

| Source model | Provider request | Quantities consumed |
|---|---|---|
| `optmat_cubic` | `shadow_order=3, outputs="total_current"` | Completed total current and orders 1 and 3; solprop constructs the optmat cubic |
| `tdse_shadow` | `shadow_order=3, outputs="full"` | The same currents plus order-3 charge dipole; compact output cannot provide this dipole |
| `tdse_native` | `shadow_order=1, outputs="total_current"` | Completed total current and order 1; no added cubic |

The full result remains an explicit oracle/diagnostic option for either compact
route. Output selection follows the model and requested diagnostics at plan
construction; missing quantities are unavailable, never zeros. Order zero is
a timing control only. The adapter applies subtraction and density/calibration
once; the provider performs none of them.

Use solprop axis names at the binding: fields have shape `(n_nodes, n_to)` in
V/m with time contiguous. The adapter maps them to the provider's `(rows,
samples)` vocabulary. A result identifies its input/configuration and contains
the total current, the first-order shadow current, third-order current/dipole
only as required by the selected source, explicit available order labels,
per-row validity/status and the available numerical monitors. Select by order
identity: `shadow.component(...)` on `GridResponse`, or an index resolved from
`TotalCurrentResponse.shadow_orders` into `shadow_total_current`. The compact
record has no `shadow` object or `component` method. Never assume `[:, 1]`
exists in a single-order result. Units are C m/s or C m per
represented atom. All density and replacement calibration
belong to solprop's declared source assembly. A provider field lacking a
required validity indication must be checked by the adapter; successful return
alone is not evidence that every row is usable.

The driver constructs and owns the runtime after admission, and passes it to
the NumPy backend through that binding. Host orchestration invokes it between
field reconstruction and source assembly. A later compiled port would use the
same host boundary, not a TDSE Python call inside `njit` or a hidden global
handle. CPU host arrays cross this boundary; native/device objects remain
owned by solprop-tdse.
For a simultaneous propagation batch, flatten `(n_batch, n_nodes)` only when
the atomic and sampling configurations match, and retain the inverse row map
for errors and projection. Otherwise schedule separate plans. Batched atomic
rows do not require independent propagation trajectories to share adaptive
z-step decisions.

Proposed output records include the physical specification, resolved basis and
prepared-state identities, potential/model revision, provider version and build,
precision, device, sampling and actual atomic step, requested/returned shadow
orders, native-retention or replacement policy, material selections,
calibration identity, radial rule, modes and z method. Store execution facts
separately from scientific configuration so a device change can be audited.
Never serialise native handles as provenance.

Diagnostics distinguish `outside_probability` from ionised-electron density;
the former is a solver diagnostic with the limitations in §4.2. It is available
as `outside` on the full result, not as a compact result field: request a full
plan/evaluation when that diagnostic is needed, with its cost recorded.
Both result types retain validity, outcomes, monitors and neglected-force
bounds; check validity at every consumed row/sample, including truncated failed
rows. Record retained
source work, response validity, relevant basis/absorber monitors and call counts.
Large nodal histories are opt-in at selected planes. Reuse an available
accepted-state evaluation when it matches the requested field exactly. Otherwise
a requested atomic diagnostic costs a new solve and is counted explicitly.

### 6. Implementation strategies

#### 6.1 Recommended stage pipeline and lifecycle

The pipeline is

```text
modal spectrum
  -> existing fine-grid transform and real radial field reconstruction
  -> persistent batched TDSE evaluation from the ground state
  -> same-map shadow subtraction + selected reference cubic
  -> separate P and J transverse projections and transforms
  -> separate spectral source factors, summed in the internal convention
  -> z integrator
```

Retain the current transform normalisations, mode norms and stage density.
Add a current accumulation buffer and current spectral factor beside the
polarisation equivalents. Both feed the same modal derivative. Do not implement
this as an undocumented nonzero-frequency division by $\omega$ to reuse the
polarisation buffer. The NumPy path is the readable implementation of these
operations and feeds the delivered Adams method.

Preparation constructs operators, spectral setup, the prepared ground state and
response plans once. Admit the combined host and device memory requirement,
including shadow states, observable buffers and transient native workspace;
report it with the normal memory ledger. If all radial nodes do not fit, prepare
stable chunks with plans for their row counts, including the final partial
chunk. Reuse those plans throughout propagation. Chunking must reproduce the
unchunked row results within justified arithmetic tolerances. It is not a
license to change the atomic basis or sample grid under memory pressure.

Each evaluation restarts every row from the ground state. The plan retains
allocated workspace, not quantum history from the previous z stage. Respect
the public result ownership and lifetime: both result types return owning,
read-only SI arrays which survive subsequent calls and plan closure. Keep
assembly buffers separate; do not mutate those arrays. Close plans before spectral setups and
operators, and all before native runtime teardown, including on exceptions.
Parallel parameter scans must budget their aggregate GPU memory and use an
explicit scheduling policy; they must not accidentally create an unbounded
number of persistent plans on one device.

Inspect row failures, validity masks and finiteness before using a source.
An invalid atomic evaluation aborts with z, attempted step/stage, radial row,
input identity and atomic configuration. Reducing the z step is not a general
repair for an unconverged basis or invalid time grid. Valid sources may still
cause ordinary z-step rejection through the integrator's error estimate.

Exact reuse is allowed only for identical field, atomic configuration and
sampling. A microscopic per-atom response can in principle be reused at another
density, but its density-weighted modal derivative cannot; retain this
distinction in cache keys. Reuse only as permitted by SEP 13's response and
history contract. Its labelled PEC approximation is not an exact source cache.
If any identity changes, invalidate cached derivatives and multistep history
as applicable.

#### 6.2 NumPy propagation with independent atomic acceleration

Use the public host API [9] and NumPy Adams from
[SEP 13](sep-0013-expensive-response-steppers.md), with one expensive
source call per mature trial. The maintainer has selected this method at
`rtol=1e-2`; invest in calibration, atomic execution and source assembly.
There is no TDSE/RK fallback or repeated method selection. Native
solprop-kokkos integration, shared device memory and a Numba port are outside
this SEP's completion requirements. Measure the host fraction before proposing
a compiled port; atomic CPU/GPU parity remains a separate contract.

Propose an optional `tdse` extra, with a phase-1 dependency review before it is
added. The reviewed compatibility baseline is the public API at `3e99e319`,
including the `beb163e` device correction; phase 1 records an installable
version/build or exact development revision rather than inventing a release
number. Capability admission covers orders, output selection, current units,
fresh reset, ownership, monitoring, projection mode and the actual execution
space. `ResponsePlan` does not itself provide a `device=` keyword: the adapter
must validate the installed build/runtime and refuse a mismatched request.
This Draft adds no dependency.
CPU integration tests need a small qualified atomic fixture; expensive helium
and GPU qualification live in explicitly selected lanes. Ordinary solprop
installation and non-TDSE runs must not import or initialise the atomic engine.

#### 6.3 Focused radial check

Retain the actual radial field and modal projection. Useful batching reduces
its cost, so radial optimisation is secondary to calibration and integration.
Start from the existing fixed Gauss–Kronrod rule; screen one smaller fixed
rule with about 16 points, using the existing quadrature machinery where
possible. With the default `kronrod=True`, the one-mode default rule has 33
nodes: `n_r=32` is rounded up to the Kronrod extension of the 16-point Gauss
rule. `n_r=16` gives 17 nodes with an embedded 8-point Gauss rule. The
measured 16- and 32-row device timings (§6.7) are each one row short of
these rules and are not measurements of them. On a small set of saved entrance and compressed waveforms, compare
its projected **current remainder and complete spectral derivative** with an
independent denser rule, around 32 points or the existing rule. Use absolute
errors as well as relative errors near cancellation. If the smaller rule
misses a sharp ionisation region, keep the denser rule and proceed; do not
start an open-ended search for the smallest rule.

The embedded coarse projection can reuse fine-rule atomic rows with zero
coarse weights where appropriate. It is an error indicator, not an independent
reference: both projections can miss the same narrow radial feature. Save raw
responses so source-scale and cubic comparisons reuse the same solves. Compare
whole-batch latency before concluding that halving the row count halves cost.
Keep the selected rule fixed through a run and record its nodes and weights.

Adaptive panels, area-coordinate rules, amplitude interpolation and broad
multimode convergence are later work on the [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics).
The one-mode test propagation claims accuracy within its stated modal model. If a later
claim needs negligible higher-mode feedback, establish that separately;
small higher-mode energy alone is not a bound on fundamental-mode error.
Reconstructing on-axis optical intensity costs no new atomic solve; an on-axis
atomic diagnostic does if that point was not in the evaluated rule.

#### 6.4 Consume the independently qualified stepper

[SEP 13](sep-0013-expensive-response-steppers.md) is complete. Use the
public `Multistep()` and its NumPy march; do not implement a private stepper
inside the atomic adapter. The delivered defaults are predictor order up to
five, `variable_order=True`, `dense="off"`, `interval_cap=False`,
`equal_approach=True` and `max_growth=2.0`. Fixed PEC3/4 remains available as
`Multistep(order=3, variable_order=False)`. AB, PECE and matched pairs are
`ComparisonMultistep` benchmark controls only. Use the delivered variable-order
PEC at `rtol=1e-2`, as selected by the maintainer. Do not add a method comparison,
new weights/controller, multirate prototype or mandatory `1e-3` run. This
choice applies to the TDSE configuration, not the global solprop default.

SEP 13 measured the canonical 4.2 m case with existing Kerr/plasma sources,
50 common planes and no noise, against IP-DOPRI5 at `rtol=1e-8`. These are
historical proxy results, not TDSE measurements or permission to run a
full-length TDSE benchmark:

| Run | Source calls | Worst window field error | Peak-power error | UV-band field error |
|---|---:|---:|---:|---:|
| IP-DOPRI5, `rtol=1e-3` | 1344 | $1.4\times10^{-3}$ | $8.7\times10^{-4}$ | 0.25 |
| `Multistep()`, split collar, `rtol=1e-3` | 292 | $1.3\times10^{-3}$ | $1.7\times10^{-3}$ | 0.054 |
| `Multistep()`, split collar, `rtol=1e-2` | 140 | 0.013 | 0.019 | 0.32 |
| IP-DOPRI5, `rtol=1e-2` | 1127 | 0.015 | 0.013 | 0.75 |

These inherited numbers use SEP 13's 90 nm grid edge, not the agreed 120 nm
TDSE grid. They motivate the maintainer's Adams choice; do not repeat the
RK comparison or represent these errors as TDSE measurements. The accepted
`rtol=1e-2` choice does not certify weak UV accuracy. A specific source,
tail or integration discrepancy may justify one tighter Adams short-window
check after inspecting the atomic response; there is no general stepper
qualification campaign. Actual implementation defects still require fixes
and focused regressions.

The lifecycle follows the delivered contract:

| Concern | Inherited behavior and TDSE obligation |
|---|---|
| History | Retain the position and density of each evaluated derivative and transport it with the existing exact diagonal flow; a save does not discard history |
| PEC source identity | The stored source belongs to the predictor; its atomic diagnostics do not describe the corrected accepted field |
| Restart | A hard fill-port kink, a fresh-start resume, or three consecutive rejections restarts history from the accepted state; charge each new startup evaluation |
| Output | Land on save planes by default; `dense="adams"` observes without new source calls and never inserts interpolated fields into history; `dense="hermite"` remains a reference, not a production option |
| Resume | No multistep history is persisted; resume from the saved field at order one, charge startup, verify atomic/model identity and record whether that field was landed or interpolated |
| Budget | `Multistep(max_calls=...)` reserves before a trial and raises `ResponseBudgetError` with last accepted position/field and calls spent; do not discard or partially commit an unpaid trial |
| Norm and admission | Retain the accepted-start-state norm of ADR-037 and the existing route/noise/schedule restrictions |

Keep the **in-stage temporal collar** as the default. `absorber_scheme="split_step"`
can remove PEC's collar stability limit, but its splitting error is first order
where field reaches the collar. SEP 13 found a $2\times10^{-4}$–$7\times10^{-4}$
field-error floor on other cases when this happened. Its canonical split-collar
results apply only while the field stays clear. With TDSE, check the atomic
current/coherence tail and source taper as well as the optical field; a quiet
entrance pulse is insufficient evidence. Use one targeted tail/window check if choosing split-step for the test propagation;
otherwise retain the default and record its cost. Dense observations are not
needed for 50 landed saves; investigate their weak-band interpolation only if
opting into them. The strong pump dominates the step norm, and the per-step
collar is not applied inside an interpolated step.

**Accounting still to add in SEP 12.** The delivered budget covers Adams
startup/trials/restarts; it is not yet an atomic-service budget shared with
requested diagnostics or audits. Define one ledger at the response binding:
count every attempted `ResponsePlan.evaluate`, including each radial chunk,
diagnostic, corrected-state audit and failed call, and retain the parent logical
source-evaluation count separately. Reserve all chunks needed for a logical
evaluation before starting it, and all required evaluations before a trial;
charge actual attempts and release unused reservations after rejection/failure.
Keep `Multistep.max_calls`' existing logical-call meaning, and add
`TDSEExecution.max_atomic_calls`, shared by Adams and diagnostics.
Both limits must pass admission before work begins. The output records both
units, elapsed atomic time, rows and substeps, so chunking cannot silently
multiply a nominal call budget. Exact accepted-field diagnostics, including
those at dense observations, need a charged solve unless an identical response
is already available. No automatic PECE-like extra call is hidden in `stats()`.

#### 6.5 Deferred stepper research

SEP 13 deferred its multirate Q5 to this proposal. The maintainer's review
now defers it beyond SEP 12: the delivered Adams method is sufficient for
integration and the test propagation. The [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics)
retains the actual-TDSE residual question and any future method comparison.
No residual survey, correction prototype, RK run or additional stepper
optimisation is a phase gate here. SEP 13's Q6 is addressed only by correct
coupling and targeted checks on the source/observables used, not by reopening
the method selection.

#### 6.6 What Richter et al. establish, and what needs re-derivation

Richter et al. [2] couple microscopic TDSE calculations to a cylindrical,
paraxial free-space propagation model. They consider soft-core 1D hydrogen and
3D hydrogen, extract the linear response separately, and use a linear
propagator for resonant media. Their transverse calculation uses 128 radial
points at 2 µm spacing. This is evidence that a coherent microscopic response
can change macroscopic propagation; it is not a helium capillary validation or
a basis recommendation for the present solver. The related resonant propagator
[7] and non-instantaneous cubic study [8] motivate further comparisons, but
their algorithms and numerical claims are not adopted here without review.

Their Eq. (4) advances an accepted field $y_n$ and a predicted field
$\widetilde y_n$, schematically

$$
\widetilde y_{n+1}=y_n+h f(\widetilde y_n),\qquad
y_{n+1}=y_n+\frac h2\{f(\widetilde y_n)+f(\widetilde y_{n+1})\}.
$$

Thus the new evaluation is at the predictor, and the following step reuses it.
It is a second-order PEC scheme with stored predicted-state information, not
ordinary Heun with a fresh accepted-state derivative. The displayed equations
have constant $h$; their intensity-dependent step selection does not supply a
general variable-step error estimator for solprop. Startup, changing steps and
interaction-picture transport are already derived and implemented in SEP 13.
The paper is context, not another stepper implementation task for SEP 12.

The authors observe rapid growth near strong bound transitions and compare
two responses to it. Filtering above angular frequency $9\times10^{15}$ rad/s
suppresses the strong transition of their 1D model near
$10.9\times10^{15}$ rad/s. It changes the calculated ionisation and linear
absorption. Their alternative extracts a forward source from present and
previous nonlinear polarisations. The useful lesson is to diagnose resonance
and directionality; neither filter nor correction can be imported as a
physically neutral stabiliser.

There are two reproducible consistency problems in the printed construction:

1. Equation (5) contains the same $P_{\mathrm{NL}}(z)$ in both interpolation
   terms. Literally simplifying it gives $-P_{\mathrm{NL}}(z)$, independent
   of the interpolating position. It cannot be the stated interpolation of
   current and previous source values, and fails a constant-source check.
2. For $x=k(\omega)\,dz$, Eq. (7) prints

   $$
   C=\frac{e^{2ix}-1}{2ix},\qquad
   K=\frac{iC[ixC+1-C]}{x(1-|C|^2)}.
   $$

   Substitution of $x=1$ gives approximately
   $K=0.789371+0.361262i$, although the accompanying text calls $K$ real.
   This discrepancy is not explained by changing the Fourier sign: conjugation
   does not make a complex number real.

These observations identify inconsistencies in the published expressions, not
proof that the authors' implementation or plotted results are wrong. There is
no uniquely justified typographical repair. A future candidate correction would have to
start with a defined projection/inner product onto forward and backward waves,
derive the coefficients in solprop's convention, and pass constant-source,
linear-source, pure-forward/pure-backward and small-step limits. The two
directional basis functions become nearly dependent over a short interval;
the conditioning and cancellation of such a projection must be analysed.
Where relevant, compare with an independently integrated bidirectional
manufactured problem. Do not add the printed correction to the baseline.

For their 8 fs, 830 nm, $10^{18}$ W/m² hydrogen example, the reported losses
were close for 0.25 and 0.5 µm z steps and changed substantially at 1 µm.
These are case-specific results, not a longitudinal mesh requirement for
helium or capillary modes. In another comparison, applying four times a TDSE
correction every fourth step gave an almost unchanged integrated transverse
profile while some excited populations differed by about a factor of three.
That experiment rules out using that profile alone as validation of sparse
updates. It does not prove that every controlled multirate method fails.

The following are conditional diagnostics for an observed instability, not
a required suite of new atomic studies:

| Possible cause | Discriminating test | Consequence |
|---|---|---|
| z-integrator stability or a noisy error estimate | Inspect source noise and existing SEP 13 tests, then a targeted tighter Adams short window if needed | Diagnose the observed defect; broader method research stays outside this SEP |
| Unresolved drive interpolation or aliased response | Refine returned-response spacing separately from internal atomic step; inject known out-of-band tones | Oversample before filtering; smaller atomic steps alone may not help |
| Periodic-window wraparound | Lengthen the field-free tail and inspect endpoint current/coherence and interior field | Qualify the window/tail treatment; do not remove a physical residual current |
| Shadow cancellation or spectral-cut artifacts | Weak-field scaling, independent cut/basis refinement and absolute cancellation errors | Repair/qualify the response; z refinement cannot remove a systematic source artifact |
| Invalid reference material model | Compare in-range and extrapolated spectral contributions and alternative calibrated responses | Restrict the validity claim or obtain a valid model |
| Forward approximation breakdown | Directional manufactured tests and an independent bidirectional reference where needed | Reconsider the physical approximation; neither arbitrary filtering nor a smaller z step proves it valid |

These tests are complementary. A resonance may expose both a source defect
and an integrator stability limit. A stable run is not necessarily accurate.

#### 6.7 Cost, memory and evidence required

For an actual run the accounting identity is

$$
T=T_{\mathrm{prepare}}+
\sum_{\text{all calls}}T_{\mathrm{response}}
 (n_{\mathrm{rows}},n_{\mathrm{to}},h_{\mathrm{atom}},\mathrm{basis},
 \mathrm{shadow},\mathrm{device})
+T_{\mathrm{host}}+T_{\mathrm{output}}.
$$

Measure the complete public call: field staging, reset, march, shadow,
observables, copies and synchronisation. Kernel-only rates or throughput at
thousands of rows cannot predict latency at 8–30 radial nodes. Record both
wall time and call counts, total rows, atomic substeps, chunk counts and peak
host/device memory. Synchronisation must be included when timing a GPU call.
Report preparation separately and use warmed repeated short evaluations for
steady-state latency, with variability and the build/device identity.

**Earlier production-basis execution study [9].** The fused order-three,
full-output configuration used 201 parent samples and 1,000 atomic steps,
including plasma observables and monitors. Its H200 whole-batch call divided
by that step count was 0.515 / 0.548 / 0.810 ms at 1 / 8 / 30 rows on the
`5024293` default-kernel lineage. This demonstrates useful radial batching,
not strong-field or full-window latency. Corrected vendor-mode measurements
have separate source identities; hardware counters were incomplete and the
small-batch mode ranking differed between cards. Neither vendor BLAS nor more
CPU threads is universally faster.

**Completed first-order comparison [9].** The twelve CPU cells measured
orders 0/1/3/5 at 1/8/30 rows, 17 parent samples, one runtime/provider thread
and eight interleaved warmed rounds. Production used 80 atomic steps,
`strong_field` 384; measured native source was `11d97c4`, driver `acf9fa9`.
Order 3 to 1 saved paired medians of about 11–14% on production. At 30
strong-field rows it saved 5.50% deterministic and 5.01% vendor, with order-one
complete-call medians of 64.487 and 59.698 s respectively. The one-row
strong-field vendor result was variable, without a robust gain. These are
short CPU windows, not GPU savings or full-pulse predictions.

**Latest strong-field device measurements [12].** The engineering workload
uses the resolved preset in §3.3, first-order shadow, unchanged monitoring and
16 or 32 distinct fields. The full grid has 19,845 parent samples, 24 substeps
per interval and 476,256 RK4 steps over 60.0004257 fs. Its spacing is about
$3.024\times10^{-18}$ s. This is a recorded workload, not a required TDSE
sampling rate or a scientifically qualified compressed waveform. The public
screen measured one deterministic first evaluation for each entrance-pulse
configuration, including ingestion, reset, march, outputs, copies and
synchronisation, but excluding setup/construction:

| Device | Rows | Earlier full output, s | Shared compact output, s |
|---|---:|---:|---:|
| H200 | 16 | 300.290 | 298.116 |
| H200 | 32 | 483.172 | 479.561 |
| MI325X | 16 | 404.991 | 394.337 |
| MI325X | 32 | 476.319 | 461.157 |

H200 used `4b634fe` plus the `beb163e` alignment correction; MI325X timings
used `4b634fe` and were followed by separate corrected-build checks. These
single first-call pairs have asymmetric idle-process residency and do not
establish a warmed speedup distribution. Full-window compressed-pulse timing,
the intermediate shared-full arm and matched order-three replacement-model
calls remain unmeasured. **The 298–480 s compact range is evidence for native
order-one execution only, not the cost of either cubic-replacement model.**

Three warmed alternating rounds of the five-sample/96-step cropped fixtures
showed modest combined savings: about 1.5–1.8% on H200 deterministic, 2.1–2.8%
on H200 vendor, 2.5–3.8% on MI325X deterministic and 3.0–4.8% on MI325X vendor.
These screens start fresh at the cropped input and do not evolve a complete
pulse. Separate profiled first-party kernel time at 32 deterministic rows was
dominated by derivative/projector work (about 80% H200, 72% MI325X); the old
shadow observations were only 0.77% and 2.30%. Thus older one-row order-three
shadow percentages cannot predict a large gain here. Profiles add fences and
omit direct vendor-BLAS intervals; they are not a full-call decomposition.

The storage improvement is more substantial. A consistent 32-row full-window
CPU inventory, including shared operators, falls from 1,185,978,072 bytes to
931,974,104 with field sharing and 510,317,376 with shared compact output:
56.97% combined. This is counted persistent storage, not peak host RSS or GPU
memory. Device construction readings instead fall from 1.430 to 1.074 GB on
H200 and 2.271 to 1.915 GB on MI325X; the latter includes about 0.300 GB already
occupied before setup. Admission must include runtime/allocator overhead,
returned arrays and transient preparation. Older ledgers omit some host/input
staging and cannot be compared by summing unlike categories.

Measure complete calls for the selected configuration's actual basis, lattice,
window and output mode on a representative compressed waveform before relying
on entrance-pulse cost. Do not infer stronger-basis timing by scaling a
production-basis measurement. No 1/8/30-row matrix, native-model campaign or
repeated timing study is required. Any later native comparison needs its own
qualified cut/step, which can erase its shadow saving.

**The GPU budget.** The SEP has a fixed rented-GPU budget of **US$100**: about
26 device-hours on one MI325X at the quoted $3.80/h. A rented instance is
billed for as long as it exists, including provisioning, building and idle
time, and a stopped instance is usually still billed. The budget therefore ends
only when the instance is destroyed. Stop new work at $90, about 23.7 h, and
leave the rest for billing granularity and teardown. Phases 1 and 2 use no
rented GPU time: calibration, integration and every coupling test run on CPU,
on weak-field ladders, cropped windows and small fixtures. A full-window
`strong_field` call on CPU would take hours, so CPU work never uses one.

The measured MI325X first calls above are 394–405 s at 16 rows and 461–476 s
at 32 rows, for the strong-field workload of a 60 fs window. The CPU comparison
above used about 4.8 times fewer atomic steps on the production basis than on
`strong_field` for the same samples. With a similar cost per step, that gives
an **estimate**, not a measurement, of about 100 s per full-window call on the
production basis. The plan below uses those figures and is revised from the
first measured call.

| Item (phase 3) | Planned device-hours | Planned cost |
|---|---:|---:|
| Provision, build the device extension, admission and focused GPU tests | 2.0 | $7.60 |
| Representative calls on one compressed waveform: `strong_field` and the production basis at 17 rows, `strong_field` at 33 rows | 0.5 | $1.90 |
| Atomic screens: two further saved waveforms, both bases, 17 rows | 0.5 | $1.90 |
| Short test propagation: at most 30 atomic calls at 17 rows | 3.5 | $13.30 |
| Debugging and one rerun | 3.0 | $11.40 |
| A second session, if one is needed (provision and build only) | 2.0 | $7.60 |
| **Planned total** | **11.5** | **$43.70** |

The difference between the planned total and the $90 stop is contingency, not
an allowance for more work. Each session records its start, destruction time
and the calls it spent. If the work ends early, destroy the instance at once.
Spending past the plan needs a recorded reason; spending past $90 needs the
maintainers' agreement.

Choose the basis for the test propagation from the measured representative
call. Use `strong_field` if 30 calls of it fit the 3.5 h item (about 420 s per
call at 17 rows). Otherwise use the production basis, labelled as reduced
accuracy, and compare the two on the saved waveforms. Keep Adams at
`rtol=1e-2`. A mature PEC trial uses one logical source evaluation; startup,
rejection, chunks, audits and requested diagnostics also count against
`max_atomic_calls`.

For scale only, a full-length canonical run would need about SEP 13's 140
logical calls at `rtol=1e-2` (§6.4): 11.7–18.7 h at 300–480 s per call, or
$44–71. That count includes that run's own startup and rejections. It excludes
extra radial rows, order-three shadows, chunks and diagnostics, and assumes that
the TDSE source leaves the step sequence unchanged. It does not fit this budget
beside the development work, and is a separate later decision (§7.3).

### 7. Verification and validation

#### 7.1 Metrics and acceptance

Use a reproducible input/configuration bundle and freeze the comparison metrics
before choosing an efficient discretisation. For the fundamental, compare the
complex output amplitude on a common physical grid, with the interface's true
power normalisation, for example

$$
\epsilon_1=
\frac{\|b_1-b_{1,\mathrm{ref}}\|_{L^2(d\omega)}}
     {\|b_{1,\mathrm{ref}}\|_{L^2(d\omega)}}.
$$

Adams at `rtol=1e-2` is the maintainer-approved longitudinal accuracy/cost
choice. Phase 1 fixes the physical observables and calibration comparisons;
phase 3 reports the test propagation's source accuracy and model limits. Where short-window
reference data exist, report the measured field, duration and peak-power errors
and show 1%, 3% and 5% bands as context. Do not commission full reference
propagations or a tolerance sweep just to fill that table. The 50 save planes
are optical observations, not requests for 50 extra atomic diagnostics.
`rtol=1e-2` does not mean 1% global accuracy. Do not optimise over a time shift
or global phase to hide physical delay or CEP differences. Use the same
propagation frame, energy convention, time origin and sampling. For pulses
with satellites, record energy concentration and a second duration measure
alongside FWHM.

An independently refined run is a reference with uncertainty, not exact truth.
Allocate the selected total target across atomic, sampling, transverse and z
errors; do not allow each axis the full allowance and call their combination
converged to that same allowance.
Refine axes separately and then jointly, documenting the observed combined
error and the remaining reference uncertainty. Calibration/SAE uncertainty is
reported separately even when it exceeds the numerical target.

For a predeclared UV band, compare energy, complex spectrum, phase above a
fixed amplitude floor, filtered temporal shape and delay. Use a pump-energy
normalised absolute complex-field or energy error alongside relative UV error
when emission approaches zero. The weak-band tolerance must be chosen for the
scientific question in phase 1; a pump-dominated 1% norm alone cannot qualify
weak UV emission. Apply the same principle to a small current remainder after
large shadow terms cancel.

#### 7.2 Focused validation matrix

The matrix specifies what evidence a claim needs, not a Cartesian product of
all fields, bases, outputs, devices and methods. Reuse the upstream evidence
in §3 and saved response bundles. Run cheap manufactured tests for every
coupling contract; reserve real atomic calls for the missing source evidence.

| Layer | Required cases and observables | Acceptance basis |
|---|---|---|
| Atomic/source limits and lifecycle | Zero field/density, reversal, independent rows, repeat/reset/chunk behavior; invalid/nonfinite output, closed runtime and incompatible resume | Analytic/mock coverage plus one small actual CPU fixture; no history leakage, partial source or hidden fallback |
| Linear calibration | Weak-field dipole versus optmat, exact discrete first-order subtraction, units and density | Recorded linear amplitude/phase mismatch; no restored linear current, fitted-shadow substitution or double counting |
| Cubic calibration | Same-map amplitude ladder, carrier Kerr and selected THG/short-pulse checks; optmat cubic versus calibrated dipole on saved fields | Correct coefficient, phase and Taylor convention; fifth-order remainder above the derived cancellation floor; third-order current artifact excluded |
| Ionisation calibration and hybrid work | Parker/optmat references versus the actual SAE model; start $N_{\mathrm{eff}}=2$, compare unscaled control from the same outputs; check current/work/depletion | Like rates or like yields, normalization once, documented relation and uncertainty between ionisation ratio and current scale; no implicit double-ionisation model |
| Optional native source | First-order native retention on existing compatible evidence or selected saved fields | Explicit uncalibrated cubic and current-artifact limits; no mandatory native propagation or performance campaign |
| Output and arithmetic | Required full/compact outputs and the CPU/device mode actually used | Owning arrays, bounds and required fields preserved; cancellation errors bounded in absolute SI units; no obligation to requalify unused vendor modes or devices |
| Source assembly | Manufactured P/J, conductor attenuation, density scaling, independent modal integrals, derivative/boundary-term equivalence | Analytic signs, dimensions, transform normalisation and work accounting |
| Atomic numerics and sampling | Reuse upstream ladders; screen the selected basis, atomic step, box, response lattice/window and 120 nm edge on selected saved fields | Measured sensitivity and explicit unresolved errors for the test propagation; stronger convergence evidence before claiming quantitative source/UV accuracy |
| Radial projection | About 16 points versus one independent denser rule on a few representative fields | Report projected remainder and complete-source discrepancy; retain the denser rule if needed without a broader optimisation study |
| Adams coupling | Existing manufactured tests plus a bounded actual-TDSE run covering pressure/port, budget and restart | Correct predicted/accepted source identity and charged work; no RK comparison, new method or repeated general order study |
| Collar and observations | Inspect actual optical/source tails; refine the window only where needed; default landed saves | No unexplained wraparound or mismatched diagnostics; split/dense alternatives checked only if used |
| Cost and test propagation | Startup, trials, failures, chunks, saves and explicitly requested diagnostics on the selected execution path; device-hours per session | Complete call ledger, the $100 GPU budget respected, and a reproducible short segment with honest numerical/model limitations |

SEP 13 already supplies polynomial, transport, manufactured stability and
order tests. Reuse them. Corrected-state PEC audits cost a full atomic solve:
perform one only to answer a stated discrepancy, not after every accepted step.
A failure in a targeted check still needs a repair or an explicit restriction
on the demonstrated domain; cheaper settings do not excuse incorrect coupling.

#### 7.3 Resource policy and the short test propagation

Work in this order. Steps 1–3 use CPU only; step 4 is the only rented-GPU
work, within the budget of §6.7.

1. **Offline and cheap tests.** Derive the SI/sign/normalisation checks, use
   manufactured responses for the adapter and Adams lifecycle, and reuse
   compatible upstream calibration, convergence and device evidence. Run the
   rate-based canonical case on CPU and save a small waveform set covering
   the entrance, compression and a useful UV-generating state, with parent
   provenance. Do not rerun a full parent propagation inside a test.
2. **Calibration on CPU.** Weak-field ladders, the linear and cubic
   comparisons and the ionisation comparisons of §4.4–4.5 use short or cropped
   windows and small row counts. Before a longer CPU batch, record the question,
   waveforms, resolved basis/output, maximum attempted atomic calls and
   wall-time budget. Increase one axis only when its uncertainty matters to the
   next decision; use three levels when claiming a convergence order.
3. **Integrate and verify the segment launch on CPU.** Exercise the actual
   CPU interface on a small fixture, with propagation tests of about ten
   accepted steps. The test propagation starts at an interior plane $z_0$ of the
   canonical case, from its saved one-mode field, with the fill evaluated on
   $[z_0, z_0+L]$. Verify that launch with the rate-based response first: a
   segment started from the saved field at $z_0$ must reproduce the parent
   run's saved field at $z_0+L$ to within the Adams tolerance. This check is
   cheap and separates launch errors from TDSE effects.
4. **Verify the device path, then propagate a short segment (GPU).** Provision
   one instance and build the device extension. Run the admission and focused
   GPU tests and the representative calls and screens of §6.7. Then run the
   test propagation with the chosen basis at 17 rows. Start just before the
   rate-based run's self-compression point, where the field is most
   broadband and ionises most, and stop at `max_atomic_calls=30`. Compare it
   with the rate-based segment from the same field, which costs nothing on
   the GPU. Save the raw atomic responses of the propagation so that later
   source comparisons can reuse them. Pull results off the instance as they are
   produced, and destroy it as soon as the work ends.

The test propagation may use a smaller basis, lower cut or larger stable atomic
step than a scientific calculation would, selected from upstream studies and
these screens. Record every resolved value and the observed sensitivity on the
saved waveforms; label unbounded errors as unqualified. Do not loosen physics
calibration, omit required shadow orders, change source definitions or
silently shorten the response tail to buy speed. Distance covered, calls spent
and completion status are part of the result. A budget stop is a valid,
recorded outcome for the segment; it does not trigger more spending.

The full-length 4.2 m TDSE run is outside this SEP. Whether to run it is a
later decision, informed by the measured per-call cost and the segment's
call count. It is listed on the [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics) with the
conditional cost of §6.7.

The test propagation is not held behind exhaustive strongest-field
convergence, a full high-accuracy reference run, native-model propagation or a multimode
campaign. Broader quantitative claims still require their own evidence.
Prioritise low-order matching and ionisation calibration, then refine only
source errors that limit the chosen main-pulse/UV observables. If hybrid
work/depletion is inconsistent, restrict the scientific claim and record the
model question; do not add a fitted neutral mask or hidden loss/filter term.
Closed/large-box work checks and independent references such as tRecX [5]
are useful where existing evidence is available or a specific discrepancy
justifies the cost. Agreement of implementations alone does not establish
basis convergence or all-electron accuracy.

### 8. Documentation

Add `docs/guide/media/tdse-capillary.md`, a manual chapter covering the hybrid
equations, direct-current sign, actual radial projection, validity limits,
sampling, installation/execution choice, lifecycle, diagnostics and the error
budget. Explain the selected Adams `rtol=1e-2`, helium calibration record,
small radial-rule screen and the CPU-first, fixed-GPU-budget resource policy.
List the test propagation's settings and their limits rather than implying a
universally qualified preset.

Update the nonlinear-response and plasma chapters, backend guide, kernel
contract, stepper design, conventions where a new source quantity is named,
output/resume guide and benchmark ledger. Public dataclasses and runtime
protocols document units, shapes, ownership, failure and supported ranges.
Record the intentional response-model differences on the Luna differences page
without implying that TDSE makes the approximation exact.

Provide a small runnable CPU example in that chapter once the dependency is
qualified, bounded to a short propagation. The GPU test propagation of §7.3 is
reported in that chapter from its saved output, with its basis, row count,
distance covered, calls, device-hours, cost, completion status and provenance.
Its comparison with the rate-based segment from the same field is shown beside
the calibration and the known source errors. It is not executed by the docs
build or by `poe docs-examples`, and it is not a regression test. User-facing
documentation calls it a short test propagation at reduced or stated atomic
accuracy, not a qualified scientific demonstration. A worked example of the
full 4.2 m case waits for the later decision of §7.3.

### 9. Compatibility and impact

Existing non-TDSE results, response selection and automatic backend policy stay
unchanged. This is a new opt-in physical closure, not a replacement for PPT in
existing simulations. Initial restrictions are validated before expensive
preparation. NumPy is the supported propagation backend for this SEP;
atomic CPU/GPU parity is a separate qualification axis.

The atomic descriptor, runtime binding, separate current source factors,
stage bundles and any new protected driver surface require a **kernel schema
version increment**, corresponding contract tests and a changelog entry.
Choose the next version at implementation time. The output schema must also
be incremented for persisted TDSE metadata/history as required by its own
compatibility rules; it is independently versioned. Refuse incompatible
checkpoint identities rather than reconstructing a different atomic model
under an old record.

**Requested backend exception:** acceptance explicitly permits this SEP to
reach `Final` with NumPy propagation only. Numba and solprop-kokkos must refuse
the feature by name; they continue to support existing propagation. A compiled
host port, device-resident optical pipeline, shared native Kokkos runtime and
zero-copy exchange are later work if measured costs justify them. This
exception does not waive analytic coupling checks or the solprop-tdse
CPU/device qualification for the execution path actually used. RK is not a
TDSE requirement.

This design selects Adams at `rtol=1e-2` and an explicit helium calibration
starting value of 2. It does not establish a universal current multiplier or
validate extrapolated 200 nm material response. Record the retained physical
calibration and the test propagation's admitted scope in Part II.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Keep PPT/ADK alone | Essential cost/physics comparator and existing supported route; cannot answer the proposed coherent broadband-response question by itself |
| Propagate the full raw TDSE current with no replacements | Useful self-consistent SAE control, with the corresponding SAE linear operator; gives up the desired empirical linear/Kerr material model and does not remove cut artifacts automatically |
| Scale the third-order total-current shadow to restore Kerr | Rejected for the initial coupling because its documented cut artifact can dominate; use the charge dipole or optmat cubic |
| First-order-only shadow with native cubic retention | Provider capability and compact output are delivered; optional unscaled research comparison still needs current-level scientific qualification and is not equivalent to calibrated dipole replacement |
| Compact current output for every model | Suitable for optmat replacement and native retention; rejected for calibrated TDSE dipole because the compact record contains no dipole |
| Add TDSE on top of the existing plasma and Kerr responses | Rejected: double counts the same response orders/electrons |
| One effective-area atomic solve | Rejected as the first milestone: a highly nonlinear atomic response must be integrated over the actual radial field |
| One-mode radial projection for every application | Selected test-propagation model only; negligible higher-mode feedback is a later claim requiring its own evidence |
| Shared native solprop-kokkos/TDSE path | Additional runtime/build/device-memory contract with little established benefit when atomic work dominates; defer until measured host costs justify it |
| NumPy only | Selected for propagation: SEP 13 provides low-call methods and atomic execution independently uses CPU/GPU; compiled host ports are deferred |
| RK, implicit or new multirate z integration | Outside this SEP after maintainer review; delivered Adams at `rtol=1e-2` is the selected method |
| Surrogate/intensity lookup or sparse correction kicks | Potential research optimisation with strong waveform/history dependence; requires held-out source and propagation validation, not only beam-profile agreement |
| Automatic spectral filtering to remove growth | Can remove physical radiation and alter excitation/ionisation; admissible only as a declared, converged band/window model |

### 11. Phases

Prioritise calibration, integration and a short GPU test propagation. Phases
1 and 2 use CPU only; phase 3 holds the whole rented-GPU budget of §6.7.
These three phases replace
the earlier five-phase programme: there is no standalone stepper or transverse
optimisation phase. Each is independently mergeable and reports evidence,
resource use and limitations in Part II. Cheap coupling work can proceed
alongside calibration; an expensive scientific result cannot bypass it.

| Phase | Deliverables | Gate before claiming completion |
|---|---|---|
| **1. Calibration and atomic setup** | Review the optional dependency/build and capability matrix; reusable saved-waveform harness and budgets; linear/cubic matching to optmat and helium ionisation starting at $N_{\mathrm{eff}}=2$ against Parker/optmat; select economical initial atomic settings | Explicit per-atom normalisation, same-map subtraction, separate linear/cubic/current scales and uncertainty; reproduce compatible calibration evidence or record discrepancies; identify the test propagation's supported and unqualified regimes |
| **2. Adams/TDSE integration** | Frozen specification and binding; current/polarisation assembly; resolved one-mode radial projection, fill/port, output/provenance, failure/resume handling, shared atomic budget and manual; a segment launch from a saved one-mode field; manufactured tests and a small actual CPU demonstration | Correct signs/units/work accounting, result ownership, predicted/accepted identities and all attempted calls charged; a rate-based segment launched at $z_0$ reproduces the parent run at $z_0+L$ within the Adams tolerance; delivered Adams at `rtol=1e-2`, no RK comparison or new stepper |
| **3. GPU verification and short test propagation** | Device build and admission on one rented GPU; representative calls and screens on saved waveforms (two bases, 17- versus 33-row rule); a short segment of the canonical case of at most 30 atomic calls, compared with the rate-based segment from the same field; saved responses, measured costs and limitations | Device path verified; segment run within the $100 GPU budget (stop at $90) with its ledger and completion status; calibration/source accounting retained; sensitivities and unresolved errors reported; no full-length run and no quantitative main-pulse/UV claim beyond the evidence obtained |

A verified device path and a short test propagation are the deliverable;
a full-length or fully converged strong-field run is not. `Final` requires these scoped phase
gates and the repository's normal implementation checks, not proof of every
broader physical claim. Unresolved scientific extensions move to the roadmap
with their evidence; do not label an unqualified UV spectrum as validated.
If upstream changes only execution, repeat affected parity/performance checks;
if it changes response meaning or discretisation, revisit the affected
calibration. Do not repeat unaffected campaigns.

The repository's phased-SEP rules apply during implementation and its full
final gate/slow/coverage policy before `Final`. Existing backends retain their
normal gates; the NumPy-only exception in §9 applies to this coupling. Run
expensive atomic qualification separately on the path used, not through every
ordinary test worker. This proposal-writing change needs documentation checks only.

**Acceptance decision.** This design selects Adams at `rtol=1e-2`, the
NumPy-only propagation exception, the result-selection/binding strategy,
separate calibration of linear/cubic/ionisation responses, helium's starting
factor 2, the three phases above and a fixed US$100 rented-GPU budget. It
permits a labelled short test propagation with reduced atomic accuracy, and
defers the full-length run, stepper research and broad radial/modal
optimisation.
The source policies remain physically distinct; optional native retention
is not a requirement to run another demonstration. John Travers's human review
and this AI review are recorded. The proposal is `Draft` / `Reviewed`; formal
acceptance and its resolution are still a separate maintainer decision under
[SEP 0](../sep-0000-process.md#acceptance). A second AI review (Opus 5.5) added
the existing helium calibration evidence of §3.2 and §4.4–4.5, which bears on
the starting factor 2 without changing it. At the maintainer's request it
then replaced the full-length demonstration with CPU-only calibration and
integration, and a short GPU test propagation within US$100.

### 12. Open questions

1. **Q1 — Helium ionisation/current calibration (phase 1).** How well does
   $N_{\mathrm{eff}}=2$ match the actual SAE solver against Parker/optmat,
   and over what range can $s_{\mathrm{rem}}=2$ reproduce current/work/depletion?
   Keep rate, yield and coherent-current evidence separate. The optmat
   analytic-rate factor 1.47 is not the TDSE default. *Partly answered:* the
   matched static ratio for this SAE potential is 1.758 at 0.10 au and 1.650
   at 0.15 au (§4.5), so 2 is expected to overstate this ionisation by
   14–21% at those fields. Optmat's same-potential tRecX yields at 800 nm, 6 fs
   check the SAE engine first. *Decided by the maintainer:* the starting value
   stays 2, the electron count derived with Parker's rates, which the
   maintainer judges the most reliable reference. Still open: laser-field and
   few-cycle ratios, and the transfer to the coherent current.
2. **Q2 — Linear/cubic matching and saturation (phase 1).** Record the linear
   amplitude/phase mismatch and exact subtraction, independently qualify the
   reference cubic/calibrated dipole on selected fields, and state where bound
   response replacement remains consistent as ionisation grows. *Partly
   answered:* in range, $\kappa_1=1.764$ and $\kappa_3=1.537\pm0.002$ are
   measured on the higher-cut basis [9]; the selected basis and the
   band below 400 nm remain.
3. **Q3 — Material and band validity (phases 1 and 3).** What evidence supports
   approximately 200 nm observables and the initial 120 nm edge? Separate
   illustrative spectra from quantified UV accuracy. Extension below 120 nm
   and a general resonant model remain later work. The degenerate Kerr ratio
   is known to 253 nm, but THG is not; helium's two- and three-photon
   resonances lie at 120.3, 161 and 175 nm, with three-photon ionisation below
   151 nm (§4.4).
4. **Q4 — Economical atomic settings (phases 1 and 3).** Which basis, cut,
   step, box and sample/window settings make the test propagation useful
   within its budget, and which errors remain unqualified? Phase 3 chooses
   between `strong_field` and the production basis from the first measured
   call (§6.7).
5. **Q5 — Installable dependency/build (phase 1, before adding the extra).**
   Record the supported build implementing the reviewed `3e99e319` public
   contract and `beb163e` fix, capability checks, runtime selection and optional
   installation. Portable GPU packaging is not established by device runs.
6. **Q6 — UV metric (phases 1 and 3).** Choose the UV band, amplitude floor
   and absolute/relative errors for the intended claim. A pump-dominated norm
   does not qualify weak emission, even with the accepted z tolerance.
7. **Q7 — Small radial rule (phase 3).** Is the 17-node rule adequate against
   the 33-node default on representative fields? One call each on one
   compressed waveform answers it within the budget. If not, retain the denser
   rule and revise the segment's call budget.
   Panels, interpolation and broad modal convergence move to the roadmap.
8. **Q8 — Adams integration (phases 2 and 3; narrowed SEP 13 Q6).** Are
   source identity, port/restart, budgets and tails correct on the selected
   TDSE path? Diagnose specific discrepancies with cheap tests or a targeted
   Adams refinement; method selection is settled and RK is outside scope.
9. **Q9 — Cost of a later full run (phase 3 evidence; the run is outside
   this SEP).** What do the measured per-call cost and the segment's call
   count imply for a full-length run? Report the estimate with its
   assumptions; do not claim a full-run cost from a short timing, and do not
   start one within this SEP.
10. **Q10 — Argon and other gases (outside this SEP).** Establish their own
    occupation, magnetic-sublevel and correlation treatment and response
    calibration; do not transfer helium's factor 2. Sum each physical
    contribution with its occupation exactly once.
11. **Q11 — Native-response economy (optional, after this SEP).**
    Can unscaled native cubic retention meet a useful scientific aim at lower
    cost? Existing first-order/compact execution evidence supports a later
    comparison; it does not remove the baseline calibration or require a
    native-model propagation in these phases.
12. **Q12 — Multirate residual (outside this SEP; SEP 13 Q5).** Deferred by
    the maintainer's review to the [roadmap](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md#numerics),
    along with new stepper optimisation. No investigation or prototype is
    required to complete SEP 12.

### 13. References

1. M. Kolesik and J. V. Moloney, “Nonlinear optical pulse propagation simulation:
   From Maxwell's to unidirectional equations,” *Physical Review E* **70**,
   036604 (2004). [doi:10.1103/PhysRevE.70.036604](https://doi.org/10.1103/PhysRevE.70.036604).
   Maxwell/UPPE context; the source signs in §4.6 are derived here in solprop's
   convention rather than transcribed.
2. M. Richter, F. Morales, S. Patchkovskii and A. Husakou, “Simulation of
   spatiotemporal light dynamics based on the time-dependent Schrödinger
   equation,” *Optics Express* **31**, 39941–39952 (2023).
   [doi:10.1364/OE.499406](https://doi.org/10.1364/OE.499406).
   The original drafting audit inspected the full article, including
   Eqs. (4)–(7), stability comparisons and sparse TDSE corrections. The
   printed-expression checks retained in §6.6 are this proposal's analysis;
   the present revision updates the coupling evidence, not that paper audit.
3. J. C. Travers, T. F. Grigorova, C. Brahms and F. Belli, “High-energy pulse
   self-compression and ultraviolet generation through soliton dynamics in
   hollow capillary fibres,” *Nature Photonics* **13**, 547–554 (2019).
   [doi:10.1038/s41566-019-0416-4](https://doi.org/10.1038/s41566-019-0416-4).
   Application context; no performance or TDSE-validation claim is inferred.
4. X. M. Tong and C. D. Lin, “Empirical formula for static field ionization
   rates of atoms and molecules by lasers in the barrier-suppression regime,”
   *Journal of Physics B* **38**, 2593–2600 (2005).
   [doi:10.1088/0953-4075/38/15/001](https://doi.org/10.1088/0953-4075/38/15/001).
   Source of the named model-potential family in the companion atomic-model
   documentation. No potential coefficients are copied into this SEP.
5. A. Scrinzi, “tRecX — an environment for solving time-dependent Schrödinger-like
   problems,” *Computer Physics Communications* **270**, 108146 (2022).
   [doi:10.1016/j.cpc.2021.108146](https://doi.org/10.1016/j.cpc.2021.108146).
   Independent-reference method context, not evidence of a completed comparison
   for this proposed coupling.
6. J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*,
   third edition (Wiley, 2016).
   [doi:10.1002/9781119121534](https://doi.org/10.1002/9781119121534).
   General multistep/error/stability framework. The interpolation weights and
   interaction-picture transport and error estimates now belong to the
   prerequisite SEP 13, derived independently from interpolation moments.
7. F. Morales, M. Richter, V. Olvo and A. Husakou, “Propagator operator for
   pulse propagation in resonant media,” *Optics Express* **29**, 29128 (2021).
   [doi:10.1364/OE.435012](https://doi.org/10.1364/OE.435012).
   Bibliographic context identified through [2]; its full algorithm was not
   independently audited for this proposal.
8. F. Morales, M. Richter, M. Ivanov and A. Husakou, “Non-instantaneous
   third-order optical response of gases in low-frequency fields,”
   *Optics Express* **30**, 23579 (2022).
   [doi:10.1364/OE.458765](https://doi.org/10.1364/OE.458765).
   Bibliographic/abstract context; its numerical results are not used to
   calibrate this response.
9. **solprop-tdse public documentation**, revision `3e99e319`:
   *The solprop response backend*
   (`docs/guide/solprop-backend.md`); *Device-resident response interface*
   (`docs/model/device-response.md`, especially the public host path);
   `docs/model/plasma.md`; `docs/model/perturbative-shadow.md` (including the
   linear and cubic calibration, its below-400 nm Kerr checks against
   Bishop and Pipin's data, and limitations); `docs/model/static-stark.md`
   (the helium static-rate ratios against Parker's and Scrinzi's references);
   `docs/model/atomic-models.md`;
   `docs/model/time-stepping.md`; `docs/model/accuracy.md`;
   `docs/guide/standalone-solver.md`; the complete-call tables in
   `docs/dev/kernel-mappings.md` and `docs/dev/cpu-performance.md`; and
   `docs/dev/roadmap.md` / `docs/dev/device-builds.md` for reconciled capability
   and qualification scope. The shadow cost table carries the completed
   twelve-cell first-order CPU comparison; the accuracy chapter carries the
   corrected strong-field limits. These public contracts/manual pages were
   inspected alongside the execution, convergence, first-order and
   response-optimisation reports. The numerical
   findings used here are restated in the proposal with their public manual
   sources, so no private report is needed to implement it. Recorded results
   are upstream evidence, not new measurements by this SEP. Package-relative
   document identities are given because no stable published documentation URL
   was established; there is no dependency on a private note or local checkout.
10. **optmat documentation**, revision `b2f418d`: `docs/nonlinear.md` and
    `docs/ionisation_rates.md`, including material-fit validity and the helium
    rate-factor provenance. These were inspected alongside solprop's current
    [material adapter design](sep-0003-optmat-materials.md). The development
    dependency is pinned separately; implementation must record the resolved
    installed material revision, not assume it equals this inspected snapshot.
    The human-review revision rechecked `docs/ionisation_rates.md`,
    `src/optmat/data/sources/parker_2007.toml` and `parker_2009.toml`, and
    `src/optmat/gas/ionisation_validation.py` at `717b0bc`: 1.47 calibrates the
    specified analytic rate, not the present SAE current. The second AI review
    read, at the same revision, the *effective number of electrons* section
    (the weak-field factor two of a closed $s$ shell, after Tolstikhin, Madsen
    and Morishita) and `src/optmat/data/sources/travers_2026_trecx.toml`, the
    curated one-electron tRecX yields for helium's Tong–Lin potential at
    800 nm, 6 fs.
11. **solprop-kokkos public README**, revision `b201d28`, installation,
    backend-selection and runtime/build contract; and solprop's
    [backend design](../../backends.md) and [kernel contract](../../kernel-contract.md).
    Used to define the host integration boundary and compiled-backend exception,
    not as a source of atomic-performance claims.
12. **solprop-tdse public GPU response record**, revision `3e99e319`,
    `docs/dev/gpu-response-performance.md`, *GPU response cost and memory*.
    Primary source for the latest H200/MI325X first-call and cropped timing
    tables, storage denominators, exact tested/measured source distinction,
    alignment correction and remaining HIP failures. It supersedes older
    device-unqualified statements in [9] only within its tested scope. Neither
    compressed full-window timing nor warmed full-window distributions were
    measured. The owning public arrays, bounds and output-selection contracts
    are specified in [9]; source/model qualification remains this SEP's work.

13. J. S. Parker, K. J. Meharg, G. A. McKenna and K. T. Taylor,
    “Single-ionization of helium at Ti:Sapphire wavelengths: rates and scaling
    laws,” *Journal of Physics B* **40**, 1729–1743 (2007).
    [doi:10.1088/0953-4075/40/10/008](https://doi.org/10.1088/0953-4075/40/10/008).
    Two-electron steady laser-field rates and fitted trends at 390/780 nm;
    the ADK discussion on p. 1739 explicitly distinguishes the per-electron
    formula and its factor two for the atom. The article text and optmat's
    curated source record were checked for that scope. This supports an
    electron-count starting hypothesis, not a measured factor for the present
    SAE solver's broadband current; no new atomic calibration was performed.
14. J. S. Parker, G. S. J. Armstrong, M. Boca and K. T. Taylor,
    “From the UV to the static-field limit: rates and scaling laws of
    intense-field ionization of helium,” *Journal of Physics B* **42**,
    134011 (2009).
    [doi:10.1088/0953-4075/42/13/134011](https://doi.org/10.1088/0953-4075/42/13/134011).
    Two-electron static-field reference, Table 1, curated in optmat. The
    article's scope and curated data provenance were checked; a new fit of
    the present TDSE solver to these data remains phase 1 work.

## Part II — Implementation record

### Implementation agreement — 2026-10-04

John Travers accepted Part I and the three-phase implementation plan. The
NumPy-only propagation exception, independent atomic execution, helium starting
scale 2, and US$100 rented-GPU ceiling are accepted. Implementation starts from
solprop `444b920`, including the optmat 0.3.0 pin, in an isolated worktree.
Part I remains frozen from this decision; findings and deviations belong here.

The agreed report band is 120–200 nm. Phase is reported above $10^{-3}$ of
the reference band's peak spectral amplitude; band energies below $10^{-6}$
of the parent pulse energy are flagged as weak. The 17-node radial screen
requires both the projected remainder and complete source to agree with the
33-node rule within 1%, with an absolute cancellation floor. New atomic
calibration on an Apple M5 Max is limited to two hours; longer targeted CPU work may
run on a workstation or a cluster with explicit allocations and limits. These report and
screening thresholds do not establish scientific UV accuracy.

The optional `tdse` extra and exact reviewed source pin are approved, with
Python 3.14 admission and separately verified CPU/device builds. The coordinating agent
coordinates resources and integrates bounded worker contributions. Phase 3
starts only after the CPU gates pass; no new GPU work starts past US$90.
Acceptance does not mean that any implementation or new qualification has
already been delivered.

### Phase 1 foundation — 2026-10-04

The optional extra pins the approved provider commit
`3e99e319ff48024b35083f262dfdab7c36871388`; optmat remains at 0.3.0. A fresh
CPU extension reports that source revision and passes the provider's build
admission. Ordinary solprop import does not import the provider. This is build
admission, not atomic accuracy qualification.

The manual harness under `benchmarks/sep12/` saves physical fields, numerical
inputs, hashes, raw responses and numerical/resource records. The canonical
rate-based parent completed once with all 50 planes, 142 logical source calls
and no atomic calls. Its largest saved peak power occurs at the final plane,
4.2 m: the launch is therefore at 4.114285714 m, which is also the plane of
largest 120–200 nm energy, 3.90368 µJ. An endpoint maximum is not evidence of
post-compression propagation. Preserve this limitation in the segment report.

Independent review corrected the calibration envelope to the upstream smooth
ramp, required fit-window stability diagnostics, and made the UV temporal and
spectral metrics use one recorded rectangular mask. The harness enforces a
process deadline and preserves invalid outcomes instead of treating them as
successful responses. Static-reference reuse and new atomic calibration are
still separate work; no calibration gate is claimed here.

Before the maintainer's revised verification instruction, the SEP consistency
check passed (106 tests, one expected skip), as did eight parent-harness and
eight calibration-algebra tests. On 2026-10-04 the maintainer explicitly deferred
**all further test execution, the full gate and coverage until development is
complete**. This overrides the routine per-slice test schedule for this
implementation. The new tests continue to be written beside the code. Remote
scientific calibration jobs remain individually budgeted evidence runs.

### Phase 1 calibration and resource decisions — 2026-10-04

The maintainer extended the US$100 ceiling to cover CPU and GPU cloud rentals
together, retaining the stop-new-work threshold of US$90. Targeted long CPU
qualifications may continue detached from the VPN session; each has an explicit
question, CPU/memory allocation and process deadline. The Apple M5 Max remains limited
to two hours of new atomic calibration. No cloud resource has yet been rented.

The new three-row weak-field witness reproduces the upstream higher-cut
linear/cubic calibration with the pinned engine and optmat 0.3.0. Each row uses
a smooth ramp, separate full/half-window fits, order-three full outputs and a
perturbative angular basis through l=5. This basis is not a strong-field setting.

| Wavelength | Linear factor, high cut | Carrier cubic factor, production | Carrier cubic factor, high cut | THG factor, high cut |
|---|---:|---:|---:|---:|
| 800 nm | 1.7641924 | 1.5263249 | 1.5371273 | 1.5370821 |
| 400 nm | 1.7629839 | 1.5341746 | 1.5365689 | 1.5265338 |
| 253 nm | 1.7604918 | 1.5272966 | 1.5279945 | 1.2864149 |

The production cut changes the 800 nm carrier calibration by about 0.7%.
The high-cut 800 nm carrier and THG factors reproduce the approximately 1.537
reference, while the short-wavelength THG discrepancy remains substantial.
Optmat flags the 253 nm cubic and the 400/253 nm THG references as extrapolations.
These numbers do not establish broadband response accuracy and do not calibrate
ionisation. Unscaled first/third current shadows are still subtracted exactly,
and `remainder_scale=2.0` remains the explicit starting hypothesis.

The production batch used one atomic call for three rows and 139.38 s; the
higher-cut batch used one call and 945.90 s on a different CPU. These are
resource records, not comparable speed benchmarks. Compact records under
`benchmarks/sep12/evidence/` identify builds, actual steps, source hashes,
fit windows and reference warnings. Raw outputs are preserved independently of
the implementation checkout. The amplitude ladders and pulse checks remain
in progress at this report boundary.

The recovered reference pulse has nonzero vector-potential endpoints. Its
original samples and zero-outside spline convention are retained. The
same-potential comparison uses explicit unchanged H0 ground coefficients for
that nonstationary reference launch, separately from propagation's prepared
ground-state policy. Bound populations are selected with the recovered n≤10
labelling rule; `outside` is never substituted for ionisation. Original label
spectra and radial completeness remain unresolved and prevent an unqualified
matched-yield claim. A first reference job failed before an atomic call because
the stage-table initial potential was implicit; its corrected runner supplies
the exact first vector-potential sample explicitly and retains the failed record.

### Phase 2 implementation and review — 2026-10-04

The implementation introduces frozen physics, sampling, calibration and execution
records, an immutable provider-free descriptor and an explicit driver-owned host
binding. Stable radial chunks own persistent plans, with setup/host/device memory
admitted together and every native attempt charged before invocation. NumPy
projects polarisation and current separately; the current coefficient is formed
directly. The existing Adams formulas and controller are unchanged.

Kernel schema 17 adds the descriptor, current numerator, absolute initial position
and cumulative-ledger restoration. Output schema 10 adds scientific TDSE identity,
completion/failure state, separate unscheduled terminal checkpoints and execution
history. An interior launch is a new run with parent provenance. Same-model resume
checks identity, restores cumulative budgets and the actual accepted position, and
starts Adams with fresh order-one history. Older non-TDSE files remain readable.

Independent code review checked sign, density, modal/crop normalisation,
subtraction order, reservations, failure charging and predictor labels. It found
and corrected a separate-density override, direct-seam sampling/cubic-ownership
gaps, combined memory admission, inherited fine-collar refinement and recovery of
a scheduled observation whose statistics failed. Tests for these cases are
written before verification began, under the maintainer's end-only testing rule.
The verification results below supersede that development handoff.

One window policy is explicit: optmat's cubic keeps its existing temporal source
window, while atomic dipole/current histories receive no extra taper or mean
subtraction. The in-stage field collar remains. A source-model comparison near
the tails therefore contains a window-policy contribution as well as a physical
calibration difference. Same-row quadrature diagnostics use the actual evaluated
field label and absolute cancellation floors; no predictor result is relabelled
as an exact accepted-field diagnostic.

The sibling Kokkos adapter implements the schema increment and explicitly refuses
TDSE before native allocation. Its supported native propagation algorithms are
unchanged. Full verification and GPU qualification remain in progress; this
report does not claim a completed phase.

### Deferred verification and bounded CPU witnesses (2026-10-05)

After all implementation and reporting modules were handed back, the deferred
verification stage began. Lint, formatting, typing, docstrings and spelling
checks found and corrected declaration imports, annotations and documentation.
The focused tests also found a helium string/enum identity comparison and the
zero-dimensional form of scalar statistics at the recovered-save boundary.
These were corrected without changing the response model. Manufactured
projection, budget, admission, output, resume and harness checks now pass their
focused runs. The complete core, slow, coverage and Kokkos-selected slow lanes
are running in a detached, resource-limited allocation. The sibling compatibility
gate is separate; no full-gate success is claimed yet.

The real CPU provider advanced a manufactured weak pulse through two 1 micrometre
Adams steps: three attempted atomic calls, no failures or rejections, 54.364 s
worker wall time with two library threads. All inputs, microscopic currents,
monitors and outcomes were preserved. This uses the production radial basis
with `lmax=5`, three spatial nodes and an 8 fs requested window. Its explicit
unity absorber avoids an unresolved collar in that tiny window; the witness
qualifies integration only. It makes no physical, broadband, boundary or
strong-field accuracy claim. The canonical run retains its in-stage collar.

A new rate-based segment from the saved field at 4.114285714 m reached 4.2 m
in eight accepted steps and nine logical source calls, with no atomic calls or
rejections. The complex full-spectrum difference from the parent's endpoint was
0.1718%, below the declared 1% reproduction screen. Full energy differed by
0.00831%. UV complex spectrum and UV energy differed by 1.1047% and 1.1029%;
their phase RMS above the fixed amplitude floor was 0.006549 rad. The UV values
are separate observations, not a claim of 1% UV agreement. Worker wall time was
4.060 s on the shared CPU, which is not a qualified performance timing.

The retained reference-pulse population diagnostic at
`5.68981e18 W/m²` gave a production-cut, per-electron bound-population deficit
of 0.0015655476 against 0.0015607667 in the reference's shell-selection rule,
a 0.3063% difference. The nonzero initial vector potential is preserved in the
stage table and the reference's unchanged field-free initial coefficients are
explicit. The missing independent state correspondence and radial convergence
still prevent a qualified yield-agreement claim. This does not calibrate the
coherent current or change the declared remainder scale of two.

Evidence schema 2 factors out the common atom-number density from both Kerr
and current sources at a fixed optical stage for numerical comparisons and
cancellation floors, retaining actual-density sources and microscopic tails
separately. The earlier implementation note incorrectly described the canonical
entrance and maximum-compression planes as zero-density endpoints. Both are
finite choked outlets: the default `steady_fill` resolves the nominal zero
chamber pressures to 39.4840966923 Pa at z = 0 and 4.2 m, corresponding to
9.7554867210e21 atoms m⁻³ at 293.15 K. The saved HDF5 endpoint statistics and
resolved pressure metadata confirm these values. The parent rate response and
TDSE qualifier use identical compiled density values at the entrance, central
port, segment start and exit; their common profile-table evaluation differs
from the analytic path by at most 2.4e-14 relatively at these positions.
The completed GPU record agrees with the exit density. No fill, density or
propagation setting changed in this correction.

The per-density screen remains valid and also has a well-defined limit for
other profiles with a true zero-density endpoint. Its regression detects a
12.5% source-shape difference despite zero physical sources; that manufactured
case guards against a trivial zero-versus-zero radial pass and does not describe
the canonical parent. Neither density normalisation nor the finite outlet
density resolves the separate temporal-tail and atomic-calibration limits.

### Completed phase 1 CPU evidence and limits — 2026-10-05

The weak subset, both amplitude ladders, the weak 6 fs pulse and the
higher-accuracy population retry have completed. This closes those scheduled
experiments; it does not establish a fully converged atomic response or complete
the implementation gates. The saved-only commands
`benchmarks/sep12/calibration_report.py` and
`benchmarks/sep12/calibration_yield_report.py` verify archive identities and
produce the compact `calibration-current-report` and `calibration-yield-report`
JSON/Markdown evidence under `benchmarks/sep12/evidence/`. The current report
also includes a figure. Original JSON, raw amplitudes/currents, pulse samples,
outcomes, monitors and resource records are preserved. Reanalysis makes no new
atomic calls and retains the original scientific script hashes separately
from the analysis-code hash.

**Linear and cubic response.** The higher-cut weak subset confirms the earlier
linear factors 1.764192 at 800 nm and 1.762984 at 400 nm, and carrier cubic
factors 1.537127 and 1.536569. The 800 nm THG factor is 1.537082. These support
a provisional narrowband dipole calibration at the sampled frequencies.
The 400 nm THG reference and both 253 nm cubic comparisons carry optmat
extrapolation warnings. Neither those rows nor the numerical 120 nm grid limit
admit an ultraviolet material model or arbitrary broadband frequency mixing.
The weak production and higher-cut records used different CPU builds; their
small fitting-window spreads are not a total uncertainty estimate.

**Amplitude ladder and numerical sensitivity.** The 800 nm ladder has peak
fields 2.05688, 4.11377 and 8.22753 GV/m, corresponding to 0.004, 0.008 and
0.016 atomic units. Both presets use the same input arrays, CPU build and
perturbative angular basis through l=5. The carrier extraction spans 12 cycles,
with saved samples from 15.181523 to 47.201504 fs; the complete 62.890992 fs
window includes the smooth ramps. Ratios below use unweighted discrete L2
norms, with no phase or time alignment. The numerator is the complete
$j_{\rm tot}-j_1-j_3$ remainder, containing fifth and higher orders.

| Carrier-window quantity | 2.05688 GV/m | 4.11377 GV/m | 8.22753 GV/m |
|---|---:|---:|---:|
| Production remainder / third-order current | 0.036729% | 0.147185% | 0.601748% |
| Higher-accuracy remainder / third-order current | 0.035240% | 0.141215% | 0.573162% |
| Production–higher remainder difference / higher remainder | 6.7623% | 6.7865% | 6.7961% |

The adjacent amplitude orders are 5.00262 and 5.03152 for production, and
5.00261 and 5.02105 for higher accuracy. Full-window ratios and orders are
also retained. The remainder exceeds the declared eight-epsilon operand-norm
subtraction floor by at least $1.06\times10^7$ in the carrier window. This
resolves arithmetic cancellation only; it excludes accumulated atomic
evolution and spectral-cut errors. The provider's neglected-current bound
concerns a contribution to total current, not a bound on the remainder error.

The linear and cubic current waveforms differ between presets by 10.3053%
and 14.3387%, respectively, in the same carrier window. This is materially
larger than the dipole-fit variation. The spectral cut changes from 60 to
400 hartree while the internal step changes from 0.025 to 1/192 atomic time
units. The pair therefore measures combined preset sensitivity, not an isolated
cut error or a converged uncertainty. The current-shadow subtraction remains
the exact unscaled same-map subtraction; these results do not justify replacing
the dipole calibration with a fitted third-order current.

At the declared dipole scale 1.537, the production-preset 6 fs, 800 nm weak
pulse differs from the instantaneous optmat cubic dipole by 1.58081% in the
full-window L2 norm at all three amplitudes. The complex-spectrum comparison
agrees by Parseval. Its remainder orders are 5.00297 and 5.02198. No matching
higher-preset pulse was run, so this is a model-and-preset comparison, not a
broadband accuracy bound. Signed per-atom work is retained as a sample-grid
diagnostic; it is not an all-electron energy-closure calibration.

**Reference-pulse bound populations.** At the exact retained intensity
$5.68981\times10^{18}$ W/m², recomputing the recovered n≤10 selection from
the saved complex state amplitudes gives:

| Pulse-end candidate | Yield per SAE electron | Difference from retained tRecX reference |
|---|---:|---:|
| Reference | 0.00156076673156 | — |
| Production | 0.00156554757859 | +0.306314% |
| Higher accuracy | 0.00155911997292 | −0.105510% |

The production–higher difference is 0.412259% of the higher yield. Both
calculations retain the same archived 6 fs pulse, finite endpoints,
zero-outside interpolation rule and unchanged H0 initial vector at nonzero
initial vector potential. Their saved input, state-energy and state-normalisation
arrays are identical. Comparison uses interpolation to the reference pulse
end at 36 fs, not the end of the additional field-free tail. Tail-end candidates
are 0.00158250038 and 0.00157602246; n≤8 and n≤9 selections are also saved.

The same 30-channel, finite-radius basis is used in both presets. Both miss
the six n=10 states with l=0 through 5, and the original reference label spectra
are absent. The recovered ordering rule therefore does not independently
establish eigenstate correspondence or complete the reference's observable.
Radial convergence and bitwise parity with the reference interpolation are
also unqualified. The numerical proximity is a useful same-potential witness,
not a fully matched, converged yield qualification. Missing probability is
never substituted for this bound-population observable.

**Resource and qualification handoff.** The four-case CPU calibration job
used four atomic calls: 945.905 s for higher-cut weak fits, 1106.919 s for the
higher-accuracy ladder, 328.709 s for the production ladder and 246.157 s for
the weak pulse. The completed higher-accuracy population retry used one call
and 738.327 s; the earlier 900 s timeout remains charged. These are observed
resource costs on shared CPUs, not comparable performance benchmarks. New Mac
atomic work remains 54.364 s for the separate manufactured integration witness.

Eleven saved-report unit tests passed without native calls, as did their focused
format, lint, type and spelling checks. The sibling Kokkos compatibility gate
passed on the source represented by commit `477328f`, including native and
sanitizer checks, 216 Python tests and the isolated wheel smoke check. This
does not complete the core's gate. Its first remote check failed, including
176 failures caused by an incorrect Numba thread ceiling, and remaining lanes
were cancelled pending correction; other reported regressions also require
their recorded repairs and rerun. No full core-gate, sampling-screen,
strong-field or short TDSE-segment success is claimed at this boundary.

Q1 remains open for the transfer from Parker/ionisation evidence to coherent
current: same-potential one-electron yields cannot determine an effective
electron count or calibrate `remainder_scale=2`. Q2 has narrowband dipole
evidence but unresolved current-preset and broadband uncertainty; Q3 has no
new ultraviolet material admission. The physical configuration remains an
explicit research configuration. The next phase must retain these limitations
while screening source sampling, atomic settings and radial projection on the
saved canonical fields.

### Phase 3 device verification and bounded propagation — 2026-10-05

**Delivered scope.** The NumPy/Adams optical driver now consumes the pinned
native CPU or HIP response through explicit host bindings. Frozen scientific
descriptors, persistent plans, exact discrete-shadow subtraction, separate P/J
assembly, absolute-position launches, attempted-call reservations, owning results,
terminal checkpoints and scientific-identity resume are implemented. Numba and
Kokkos propagation refuse TDSE explicitly under the accepted backend exception.
The optional native-response and calibrated-dipole models have contract tests; the
actual propagation uses optmat cubic plus the scale-two current remainder.

**Device evidence.** One MI325X/gfx942 device used the exact provider revision
`3e99e319ff48024b35083f262dfdab7c36871388`, with an independently checked loaded
native build. The focused provider tests passed (137), CPU/GPU physics checks
passed (20), and optical adapter checks passed (93). The native allocation lane
retained 17 known failures among 141 cases, with no additional failures. Those
are failed checks, not an allocation-free qualification. Three saved tiny CPU
inputs were replayed on HIP. Raw total and first currents agree at about
1.5e-10 relative and cubic currents at about 2.4e-10; cancellation prevents a
relative-parity claim for the weak remainder: its 91.3–244.2% relative
difference is 9.58e-31–2.42e-30 in the unweighted sampled microscopic-current L2 norm (A m per represented
atom, summed over rows and samples without time or density weights), above the approximately
2.85e-34 A m host subtraction indicator. That indicator
does not bound native evolution error.

**Measured selection.** Twelve completed order-three saved-waveform calls used
2918.014 s of atomic-call wall time. Complete compressed-field calls took
93.380 s (production, 17 nodes), 103.905 s (production, 33 nodes), 506.095 s
(strong-field, 17 nodes), and 615.107 s (strong-field, 33 nodes). The entrance
17-versus-33-node remainder difference was 2.1491%, so the required combined
screen selects 33 despite passing compressed and launch-field screens. The
production q32-versus-q64 complete-source and remainder differences at 33 nodes
were 0.41746% and 0.42022%; q64 is retained. The entrance production/strong-field remainder differs by 3.51726%, so
there is no general subpercent agreement between presets. Independent cut,
step, box and 33-versus-denser convergence are not established by these comparisons.

The retained response lattice has 65,536 samples across the unchanged
68.3139 fs optical period. The production atomic step is 5.21194e-19 s (two
substeps per response interval). Thirty strong-field calls alone exceed the
3.5-hour allocation; the explicitly reduced-accuracy production configuration
is used under the accepted cost rule. Its call cap remains 30, including
startup, rejected trials and any diagnostic or radial-chunk solves. No RK
comparison, optical stepper change or full-length TDSE propagation was made.

**Actual propagation.** The run stopped on its atomic budget at
z = 4.119922806819549 m, after advancing 5.6370925 mm from the saved launch.
Its 30 valid atomic calls comprise one startup and 29 predictor evaluations;
22 trials were accepted and seven rejected. There were no failed native calls,
no diagnostic calls and no reservations left over. The sole regular save is
the launch; the unscheduled terminal checkpoint retains the accepted field and
next-save index. A CPU rate-based segment from the identical launch ends at
that exact stopping position. This is a valid computational budget stop,
not completion of the requested interval through z = 4.2 m.

The TDSE worker took 3118.782 s, or 3121.230 s including the 1.193 s rate
comparison and report writing. The enclosing process took 3123 s. Maximum
host RSS was 2043539456 bytes. The 10-second whole-device VRAM samples peaked
at 2015842304 bytes; this is a sampled resident-device counter, not proof of
allocation freedom or a captured instantaneous peak. The first actual 33-node
electric-field input and its total, first and third currents are byte-identical
to the separately saved launch-field screen. This independently exercises the
interior launch and provider invocation without another atomic solve.

**Negative physical result.** This particular original-window, scale-two
optmat-cubic hybrid configuration is rejected for quantitative propagation.
Its final total energy is 4.162496 mJ, versus 0.847298 mJ in the rate model;
120–200 nm energy is 1.073117 mJ versus 3.871753 microjoules. Full and UV
complex differences are 2.04929 and 16.61833 relative to the corresponding
rate spectra, with pump-normalised differences 1.88635 and 1.03405. The coarse
reporting-grid peak power is 5.86860 TW versus 0.417742 TW. A finite in-window FWHM is unavailable
for the TDSE field, whose peak reaches the boundary; its RMS duration is
27.9923 fs versus 3.61801 fs. The UV windowed-centroid difference is
−4.67556 fs and is not a meaningful qualified pulse delay.

Independent saved-data work analysis identifies the gain-producing source.
With the correct negative optical current factor, the retained
j_total−j1−j3 term produces net projected optical gain from the third saved
call onwards. The third-order shadow's positive field-to-atom work eventually
exceeds that of the full native current; its exact subtraction then leaves
negative work. At call 30 the beyond-cubic source's per-density energy
derivative is +1.180e-23 J m², compared with +2.421e-30 J m² from optmat
cubic. Both scale-one and scale-two hybrid controls produce gain. The full
native current and j_total−j1 controls remain absorptive on all 30 stored
inputs. Those controls are neither a new propagation nor a qualification or
recommendation of the native-response model.

The first and last 10% of the time record contain 65.86% of total energy and
98.10% of UV energy, versus 0.904% and 0.117% in the rate reference. The final
predictor's peak intensity also substantially exceeds the original saved-field
screens. The work analysis retains predictor identities and the exact retained
FFT/projection/collar definition; it is not an accepted-state continuum energy
balance. Endpoint-trapezoid microscopic work changes the last central remainder
by about 2.35% without changing its sign. Independent review found no current
sign, density or crop-normalisation defect in the inspected coupling.

**Scientific claims and limits.** The quiet-start and physical-window
requirements of §4.8 were not achieved. The compressed field has non-negligible endpoints and its projected
remainder has its largest non-DC spectral bin near 21.0676 eV, outside the
propagated band. This bin alone does not identify a resonance. A saved 17-node compressed q64 smooth-partition diagnostic attributes 99.6403% of the
optical-band remainder norm and 99.7070% of its UV norm to the trailing part;
these are interfering complex contributions, not fractions of erroneous or
emitted energy. No area removal, input taper or residual-current subtraction
was introduced. The q32/q64 comparison holds the optical period and nearly
identical atomic step while changing the final native sample by one q64
spacing, so it cannot settle this finite-record error.

The saved scale-one control changes the compressed local UV source by 49.5590%
while holding optmat cubic fixed. It demonstrates sensitivity to the declared
closure, not a propagated energy change or a calibration of effective electron
count. Static Parker evidence, narrowband linear/cubic fits and incomplete
same-potential bound-population comparisons remain distinct. The 120–200 nm
plots are illustrative model comparisons, not quantitatively validated helium
predictions. Physical preparation history, tail convergence, saturation and
resonant material validity remain open.

**Verification and corrections.** The full `poe check` passed at `bd925b7`
(12650 tests passed, 28 skipped; 604.08 s for tests and 836.76 s for the
complete gate). `poe test-slow` passed at `aa7f9bb` (138 passed, one skipped,
984.68 s). The final Kokkos CPU-selected slow lane passed all 139 tests in
966.71 s at `129912d`, with unchanged source hashes across the run.
The numerical implementation has not changed since `cdedb43`; `66e48cc`
corrects only the position printed in an atomic-budget exception. Its affected
paths were tested separately, and the complete coverage suite ran on that
correction. The source identities, narrow intervening changes and preserved
failed attempts are recorded in the committed verification manifest. The first complete CPU coverage attempt passed
397 interpreted-kernel tests and 12652 remaining tests, but failed the required
100% coverage threshold at 99.914%. Eighteen statements and the remaining
branches in admission, ledger restoration and backend refusal paths received
provider-free regression cases; the threshold was not waived. The unchanged
original coverage database was copied and 125 focused tests passed while
appending coverage on identical library sources and configuration. The strict
combined result covers all 33104 statements and 9994 branches (100%).
The original `poe cov` exit status remains one; the corrected combined
coverage result is a separately recorded successful check, not a relabelled run. The first remote core run used an invalid Numba
thread ceiling and was cancelled after its failed check. Further local runs
exposed seven machine-dependent FFTW byte witnesses and an inherited xdist
identity in the benchmark collection subprocess. Both failures are preserved
in the evidence. The witness repair uses a fixed test-only FFTW plan with
independently reproduced baseline arrays; it does not alter the production
planner or relax numerical tolerances. The collector repair removes only
three worker-identity variables from its child environment. Independent review
approved both repairs. The first slow run also exposed three resume tests that treated the new
execution-history JSON as a numerical array. The repair preserves every
bitwise numerical comparison and checks the history separately. Finally, the
GPU budget exception incorrectly printed z=0 although its saved outcome and
checkpoint held the correct accepted position. A message-only repair now
reports that position; the original GPU evidence remains unchanged.
Independent review approved both corrections. Native packages in the isolated
macOS environment were rebuilt against one OpenMP runtime; the main checkout environment was untouched.

The sibling schema-17 compatibility gate passed native, sanitizer, Python and
isolated-wheel checks, including 216 passed Python tests and six skips. New
TDSE modules reached complete focused statement/branch coverage, with further
coordinator, output and resume cases exercised. An independent reviewer checked the coupling signs, units, Fourier origin,
source reconstruction, predictor labels, numerical reporting and negative
physical conclusion. The offline GPU/work-report tests passed all 29 cases;
their lint, direct type and spelling checks passed. Regenerating from the
formatted scripts preserved every numerical result. The updated manual built
with warnings as errors in 12.57 s without running atomic work.
After the archive transition, the SEP/index/roadmap structure checks passed
(106 tests, one process-SEP skip), as did spelling and `git diff --check`.
The final documentation build passed with `-W -n --keep-going` in 10.31 s.
The separate reviewed worktree remains committed for inspection; nothing was
pushed.

**Resources and preservation.** The one GPU existed for 8673 s through the
successful deletion request, costing an estimated US$9.1548 at the admitted
US$3.80/hour quote. Conservatively counting through the first confirmed API
absence gives 8698.097 s (2.41614 device-hours) and US$9.1813. This is the
whole cumulative cloud programme; no CPU droplet was rented. The ledger uses
the conservative estimate, including setup, builds, tests, screens, propagation,
idle time, collection and teardown. Deletion returned HTTP 204 and a subsequent
lookup returned HTTP 404; no tagged droplet remained. All 280 scientific
files, totalling 2932615078 bytes, matched their remote SHA-256 manifest before
destruction. Mac
atomic work remained 54.364 s, below its two-hour limit. Completed CPU jobs,
failed launches and the timed-out higher-cut population attempt remain charged
in the ledger. Raw inputs, responses, checkpoints, build identities and resource
logs are preserved independently of managed worktrees. Committed scripts,
compact records and figures reproduce the reports without launching atomic
work during documentation builds.

### Scoped completion and handoff — 2026-10-05

All three scoped implementation phases
are complete, including a negative canonical physical experiment. This status
follows §7.3's explicit instruction to restrict the scientific claim and record
the model question when hybrid work/depletion is inconsistent. It does not
admit the failed original-window hybrid configuration for physical prediction,
or imply a successful physical demonstration.
Q5 (the tested optional build), Q6 (frozen UV metrics), Q7 (retain 33 nodes for
this screen), and Q8 (the implemented Adams integration contract) are resolved
within their recorded bounds. Q4 has a measured economical configuration,
not converged strong-field accuracy. Q1–Q3 retain the specific calibration and
window limitations above. Q9 remains a later full-run decision: no full-run
cost is inferred from this short segment. Q10–Q12, broader gas/mode support,
native-response campaigns and multirate/stepper research remain outside scope.
These scientific and optimisation questions move to the roadmap with links
to the saved evidence. Part I remains the frozen accepted design; this record
and the manual state the delivered behavior and its limits.

## Part III — History

- **2026-10-05:** Completed all three scoped phases and the required engineering
  qualification. The bounded canonical hybrid experiment is physically rejected;
  its source-work evidence and unresolved scientific questions are recorded in
  Part II and the roadmap. Status changed to Final and the SEP was archived.
  Part I remains unchanged apart from relative links required by this move.

- **2026-10-04:** John Travers accepted the revised proposal and implementation
  plan, including the backend exception, optional dependency and compute limits.
  Implementation began on `feature/sep-0012-calibration` from `444b920`, under
  GPT-6 Astra's coordination. Status changed from Draft to Accepted and then
  In progress; Part I is frozen.

- **2026-10-03:** Draft started from the agreed scope; no implementation begun.
- **2026-10-04:** Detailed physical derivation, interfaces, paper analysis,
  optimisation candidates and qualification gates recorded. Draft / Not reviewed.
- **2026-10-04:** After updated complete-call evidence, made SEP 13 the
  numerical prerequisite, moved stepper implementation/selection there, changed
  this coupling to NumPy-only propagation and revised performance/qualification
  status. Physical response accounting remains subject to its scientific gates.
- **2026-10-04:** Added the gated native-cubic comparison and requested
  first-order-only shadow capability. Distinguished its cost saving from the
  existing calibrated-dipole model, which still requires third-order data.
- **2026-10-04:** Reviewed and revised by GPT-6 (AI) against completed SEP 13
  and the execution, strong-field, first-order-shadow and response-optimisation
  evidence, including the latest GPU follow-up. Recorded
  delivered variable-order PEC, current schema versions, output-selection
  matrix, arithmetic/device limits, actual timing/storage evidence and
  explicit remaining scientific gates. Inherited the bounded multirate
  question; specified logical versus atomic-call accounting and fresh-start
  resume. Updated the canonical case with the maintainer's 120 nm grid edge,
  `rtol=1e-2`, `noise=None` and 50 saves. Review status is `In review` with
  this AI review recorded; human review and acceptance resolution remain with
  the maintainer. No propagation or new atomic measurement was performed.

- **2026-10-04:** John Travers completed the human review. Revised by GPT-6
  (AI) to select Adams at `rtol=1e-2`, remove TDSE/RK comparisons and stepper
  research, and focus on calibration, integration and an affordable demo.
  Reduced the programme to three phases; retained a focused approximately
  16-point radial check and deferred broader radial/modal and multirate work
  to the roadmap. Added separate linear/cubic/ionisation calibration, starting
  helium at `N_eff=2` while distinguishing it from optmat's analytic-rate
  factor 1.47 and from coherent-current scaling. Added an explicit resource
  policy and permitted labelled reduced atomic accuracy for the demo.
  Review status is `Reviewed`; status remains `Draft` pending acceptance.
  No atomic solve, new propagation or performance measurement was run.
- **2026-10-04:** Reviewed and revised by Opus 5.5 (AI). Checked the
  canonical snippet, defaults, schema versions and Adams settings against
  solprop at `96be071`. Re-derived the velocity-gauge Hamiltonian, the completed
  current and its bound, and the P/J source coefficients in the internal
  convention. Checked the modal normalisation, the strong-field preset's SI
  conversions, the workload sizes and the printed $K$ of §6.6. Checked the
  upstream calibration, timing and storage figures against solprop-tdse
  `3e99e319` and optmat `717b0bc`. Added existing evidence: the in-range
  $\kappa_1=1.764$, the static SAE-to-atom ionisation ratios 1.758 and 1.650
  against the starting value 2, optmat's same-potential one-electron yields,
  helium's two- and three-photon resonances in the 120–200 nm band, and the
  fifth-order bound content that $s_{\mathrm{rem}}$ also scales. Recommended a
  parent spacing whose Nyquist frequency clears the harmonic cutoff. Corrected the
  radial rows to the 17- and 33-node Kronrod rules, and added a conditional
  canonical-run time from SEP 13's call count. Replaced upstream work-order
  numbers with names, and clarified the force notation of §4.2. The starting
  factor 2, the scope and the phases are unchanged. No atomic solve,
  propagation or measurement was run.
- **2026-10-04:** At the maintainer's request, Opus 5.5 (AI) replaced the
  full-length canonical demonstration with a fixed US$100 rented-GPU budget,
  about 26 device-hours on one MI325X at $3.80/h, stopping new work at $90.
  Calibration and integration now run on CPU only. Phase 3 verifies the device
  path, chooses the atomic basis from one measured call, and runs a short
  segment of the canonical case of at most 30 atomic calls from a saved
  rate-based field. A cheap CPU check that a rate-based segment reproduces its
  parent run comes first. The planned GPU use is 11.5 h ($43.70). The production-basis
  call cost of about 100 s is an estimate from the upstream step counts. The
  full-length run, conditionally $44–71 of device time, moved to the roadmap.
  No GPU time was used.
- **2026-10-04:** The maintainer kept the helium starting factor 2, derived
  with Parker's rates, which the maintainer judges the most reliable
  reference; the static ratios of §4.5 remain recorded evidence for phase 1.
