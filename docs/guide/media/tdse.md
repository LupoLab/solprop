# TDSE response in a helium capillary

For a helium-filled capillary, solprop can replace the electronic Kerr response and the
rate-based plasma together by the microscopic current of a time-dependent Schrödinger
equation (TDSE). At every radial node and every source evaluation, fresh helium atoms
are driven by the local real field, and the part of their current beyond linear response
becomes the nonlinear source of the propagation. In one calculation the response holds
the bound nonlinearity beyond the cubic, ionisation, the coherent current of partly freed
electrons and the atomic resonances that a rate model omits. The physics in outline is in
§10.6 of the primer's [ionisation chapter](../../primer/ionisation.md).

```{admonition} Availability
:class: note

The atomic engine, Pyroraptor, is a separate, optional package. It is not yet publicly
available; it is planned to be. solprop declares no dependency on it and imports it only
when a TDSE response is prepared; without it, preparing a TDSE response raises a
`BackendError` naming the missing provider. The records, their validation, identities
and descriptors work without it. The TDSE response is a research feature, not a
qualified replacement for the rate models.
```

## Two atomic models

| `model` | Atom | Current scale |
|---|---|---|
| `"sae"` | a single active electron in a model helium potential | empirical: 1.537 in the selected research configuration |
| `"hacc"`, haCC(1) | a correlated two-electron neutral coupled to channels of a frozen He⁺ 1s ion and one active electron | exactly one |
| `"hacc"`, haCC(5) | as haCC(1), with the four $n=2$ ionic channels added | exactly one |

In the single-active-electron (SAE) model one electron stands in for two, so its current
is multiplied by an empirical scale chosen from weak-field calibration
([below](#how-the-research-settings-were-chosen)). haCC (hybrid anti-symmetrised coupled
channels; Majety, Zielinski and Scrinzi, *New J. Phys.* **17**, 063002 (2015),
doi:10.1088/1367-2630/17/6/063002) computes the whole atom's current, summed over both
electrons, so a multiplier would double count it: a haCC response requires a scale of
exactly one, and its residual model error (ionic truncation, correlation, basis) is
measured, not absorbed into a scale. It uses a mixed gauge, length inside a gauge radius
and velocity outside (Majety, Zielinski and Scrinzi, *J. Phys. B* **48**, 025601 (2015),
doi:10.1088/0953-4075/48/2/025601), and infinite-range exterior complex scaling as its
absorber (Scrinzi, *Phys. Rev. A* **81**, 053845 (2010), doi:10.1103/PhysRevA.81.053845).
Its total current is an escaped-electron closure, which counts each electron once,
including the momentum of electrons that have left the box. The difference between
haCC(1) and haCC(5) measures the sensitivity to the four excited-ion channels, not the
full ionic-truncation error: both omit higher channels and the ionic continuum. haCC(1)
is about eight times cheaper per atom.

## The physics of the response

### The source

With $j_{\rm tot}$ the provider's total current of one atom (C m/s), $j_1$ its first-order
shadow from the same discrete calculation, $n_0(z)$ the atom density, $s$ the current
scale and $g(t)$ the temporal gate (below), the native source
(`kerr_model="tdse_native"`) is

$$
J_{\rm NL}(r_q,t)=n_0(z)\,s\,g(t)\bigl[j_{\rm tot}(t)-j_1(t)\bigr],\qquad P_{\rm NL}=0,
$$

with the edge correction of the next section. Only the first-order current is
subtracted, since linear dispersion is already in optmat's propagation constant; the cubic
and every higher order stay inside the current, and no optmat Kerr or plasma term is
added for these atoms. The shadow is subtracted unscaled, then density and scale enter
once. In the internal Fourier convention the projected current enters mode $m$ as
$-\mathcal Q_m[\widehat J_{\rm NL}]/(2\epsilon_0c\operatorname{Re}n_{{\rm eff},m})$ through
the existing modal projection, which supplies the area weights; a one-mode run still
integrates the actual radial field. No mean, pulse area or residual current is removed.

Two research policies for SAE, `"optmat_cubic"` and `"tdse_shadow"`, instead subtract
the first and third orders and supply a separate cubic polarisation (optmat's, or a
calibrated third-order charge dipole). They are never gated, and haCC refuses them: its
provider has no third order of the closed current.

### The atomic window

A source evaluation drives the atom over a finite interval. On the full optical lattice
the gate $g$ is the grid's temporal Planck collar. `TDSEAtomicWindow(duration,
taper=5e-15)` instead drives the atom only over the requested duration (60 fs in the
motivating case), with Planck tapers inside it, which saves atomic steps: the atom starts
in its ground state at the window's first sample, the returned currents are padded with
zeros, and $g$ is the window's gate. haCC responses use a window by default, whose
duration the capillary entry point takes from `trange`; SAE responses default to the full
lattice. The gate is fixed in the co-moving frame. It is an approximation: what the atom
would do outside it is dropped.

### The edge-free gated source

A current still flowing in the trailing taper, such as the drift of electrons freed by a
strong pulse, would be switched off by the gate over a few femtoseconds, and the
switch-off radiates in the near infrared although no electron does. The source removes
that edge. With $S=j_{\rm tot}-j_1$ and $gS$ vanishing at both ends of its support,
$\mathcal F[(gS)']=i\omega\mathcal F[gS]$ in the internal FFT convention, and the product
rule gives, for $\omega>0$,

$$
\hat S_{\rm ef}(\omega)=\frac{\mathcal F[g\,S'](\omega)}{i\omega}
=\mathcal F[gS](\omega)-\frac{\mathcal F[g'S](\omega)}{i\omega}.
$$

The left side is the transform of $S_{\rm h}(t)=\int_{-\infty}^t gS'\,dt'$, the current
*held* at its taper-weighted value after the window instead of switched off. The right
side needs only the gate's analytic derivative $g'$
({func}`~solprop.grid.planck_taper_derivative`), never a derivative of $S$. A current
constant over the window contributes nothing at $\omega\neq0$, a rise of height $H$ at
$t_c$ inside the window keeps its spectrum $He^{-i\omega t_c}/(i\omega)$, and every
variation inside the window is kept; only the current after the window, its finite
memory, is lost. This is the only form of the native source; there is no option to choose
another.

The sampled $g'$ must resolve the tapers. Translation computes the steady-current
certificate $\kappa_{\rm ss}$, the active-band norm of the discrete edge-free source of
$S\equiv1$ relative to the full transform of the gate, and refuses a run whose value
exceeds $10^{-10}$ with a `TranslationError`: widen the taper or raise `TDSESampling`.
Resolved gates certify at rounding level, about $10^{-17}$. The certificate is recorded in
the run's execution metadata, and the last-evaluation diagnostic
(`metadata()["tdse_last_evaluation"]`) reports the norms of the truncating source, the
edge term and the corrected source, with their rounding floors.

### The weak-field consistency check

Because nothing rescales the haCC current, every haCC configuration is checked against
optmat's helium before its first propagation call. A fixed weak pulse (4 fs,
$10^{11}$ W/cm², at the run's reference wavelength) drives one atom. From the first-order
dipole $\hat p_1=\hat j_1/(i\omega)$ the check forms the complex ratio
$\kappa_1(\omega)=\alpha_{\rm optmat}(\omega)/\alpha_{\rm haCC}(\omega)$ of the per-atom
polarisabilities over the pulse's band and reduces it to the largest $|\kappa_1-1|$. Up
to `warn` (0.1) the run proceeds; up to `refuse` (0.3) it warns once with a
`TDSEConsistencyWarning`; beyond that, or for an invalid measurement, it raises a
`TDSEConsistencyError` before any propagation call. `TDSEConsistency(warn, refuse,
enforce)` sets this policy, which is never part of an identity. The result is cached per
configuration and provider revision under `$SOLPROP_CACHE_DIR/tdse-consistency/` (by
default under `$XDG_CACHE_HOME/solprop/` or `~/.cache/solprop/`); deleting the cache is
always safe. {func}`~solprop.driver.tdse_consistency.check_consistency` runs the check on
its own. For the presets $|\kappa_1-1|$ is 0.035 (haCC(1)) and 0.019 (haCC(5)), worst at
500 nm. The cubic coefficient is measured as evidence, not gated: its ratio to optmat's
is 1.052 and 1.004 over 600–1,200 nm.

## Declaring a response

The records live in {mod}`solprop.nonlinear.tdse`, which loads no native code. Each is
frozen and carries a stable identity digest of its resolved fields.

| Record | What it declares |
|---|---|
| `TDSEGasResponse` | the target (helium), basis, maximum atomic step in s, `kerr_model`, the explicit scales, the calibration, the sampling, `model`, `atomic_window` and, for haCC, `consistency` |
| `TDSEBasis` | the SAE finite-element radial lattice, angular cutoff, potential identity, spectral cut, complex scaling and observation cuts, in SI units |
| `HaccBasis` | the haCC ion count, neutral square, radial axis, angular limit, exterior scaling, gauge radius, spectral cut and RK4 step ceiling, with an optional preset identity |
| `TDSECalibration` | the evidence: identity, matching basis identity, wavelength domain, source, and `status="research"` with its limitations or `"qualified"` with an uncertainty |
| `TDSESampling` | a power-of-two refinement of the fine time lattice for the atomic drive |
| `TDSEAtomicWindow` | the driven interval and its taper |
| `TDSEConsistency` | the haCC check's bounds and its `enforce` switch |
| `TDSEExecution` | the atomic resources: `device` (`"cpu"`, `"cuda"` or `"hip"`, matching the installed build, with no fallback), `memory_limit`, `max_batch_rows`, `max_atomic_calls` and `complement` |

Execution resources are not part of the scientific identity. With the provider installed,
{func}`~solprop.driver.tdse.basis_from_preset` resolves an SAE preset and
{func}`~solprop.driver.tdse.hacc_preset` one of the provider's qualified haCC presets:
`hacc_preset(1)` for haCC(1) and `hacc_preset(2)` for haCC(5), each also as a `"fast"`
variant with a lower spectral cut and a larger step (2.3 times fewer steps, changing the
current by at most $1.5\times10^{-4}$ of its peak). A preset supplies its basis, its atomic
step and a research calibration record. It is qualified for an 800 nm drive below the
helium threshold, up to $8\times10^{14}$ W/cm², over its window; a run beyond that is
flagged with a `TDSEDomainWarning`, not refused. A custom basis is for research and is
admitted with a `TDSEUnqualifiedModelWarning`. Constructing the records needs no
provider:

```python
from solprop.constants import AU_ENERGY, AU_TIME, BOHR_RADIUS as a0
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEAtomicWindow,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)

edges = tuple(x * a0 for x in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0))
basis = HaccBasis(
    1,
    1,
    edges,
    (6,) * 6,  # haCC(1), neutral square, radial axis, element orders
    "particle",
    2,  # angular limit and order
    12 * a0,
    0.15,
    6,
    1.0 / a0,  # irECS: R0, angle, infinite order, decay
    4 * a0,  # gauge radius
    30 * AU_ENERGY,  # spectral cut
    0.05,  # RK4 step ceiling, atomic units
    source="a tiny unconverged basis for a quick look",
)
calibration = TDSECalibration(
    f"custom-{basis.identity[:12]}",
    "research",
    basis.identity,
    (62e-9, 1.6e-6),
    source="a custom research basis",
    limitations=("custom haCC basis: unqualified, no preset accuracy",),
)
response = TDSEGasResponse(
    "helium",
    basis,
    basis.max_step_au * AU_TIME,
    "tdse_native",
    1.0,
    calibration,
    TDSESampling(8),
    model="hacc",
)
assert basis.label == "haCC(1) (custom)" and not basis.qualified
assert response.atomic_window == TDSEAtomicWindow()  # duration from trange
```

A haCC response is refused with a `MaterialError` unless `kerr_model` is
`"tdse_native"`, `remainder_scale` is exactly 1.0 and the atomic step is the basis's
`max_step_au` in seconds; the basis type must match `model`.

## The capillary entry point

`prop_capillary` and `Simulation.from_capillary` take the response as `tdse=` and its
resources as `tdse_execution=` (`TDSEExecution()` when omitted):

```text
run = prop_capillary(
    265e-6, 4.05, "He", fill, wavelength=800e-9, pulses=[pulse], modes=1,
    trange=60e-15, wavelength_limits=(9e-8, 1.6e-6),
    tdse=response, tdse_execution=TDSEExecution(max_atomic_calls=400),
    radial_rule="field_scale", n_r=6, rtol=1e-2, output="run.h5",
)
```

The response is the complete electronic response of the fill, so `plasma`, `raman`,
`thg`, `preionisation` and the other ionisation settings must be left unset, or the call
is refused naming them. `backend="auto"` resolves to `"numpy_ref"`, the only backend that
runs a TDSE response; the run needs the adaptive `Multistep()` integrator and a single
scalar HE₁₁ mode. A TDSE run carries no shot noise by default: it is expensive, rarely
repeated as a noise ensemble, and its checks are coherent. Input shot noise
(`noise="input"`) can be asked for, with a warning; its draw is part of the
launch, so the atom is driven by the pulse plus the draw, inside the same gated window
and on the same active band, and the realisation is stored with the run like any other
([SEP 37](../../dev/seps/archive/sep-0037-noise-and-stepper-defaults.md), section 6). One photon
per bin drives the atom far below the weak-field consistency check's intensity, where
its response is the certified linear one. `noise="modified"` is refused: its draw is
kept apart from the field the atom is driven by. Each radial node is one atom in every call, so the node
count is the main cost: a rule other than a field-scale rule, the 33-node default
included, raises one `TDSECostWarning` (see [radial nodes for expensive
responses](../waveguides/radial-quadrature.md)). With `tdse=`, every accepted step is
saved and only the final plane is requested by default, so saving never forces an extra
atomic call; an explicit `n_save`, `save_planes` or `step_planes` is honoured. Every
attempted provider call is charged against `max_atomic_calls` before it is made, failures
included. A capped or interrupted run resumes; the gate and window are part of the resume
identity, so a run never resumes under another one.

## The raw response archive

`tdse_archive=TDSEArchive("run.tdse.h5")` on `prop_capillary`, `Simulation.run` or
`check_consistency` writes every provider evaluation of the run to a separate,
append-only HDF5 file, one session per run or resume: the exact drive passed to the
provider and every array of its raw record, with dtype, shape and SI unit, before
solprop subtracts, scales or reduces anything. `outputs="full"` (the default) asks the
same calls for the provider's full observable record, `"required"` for only what the
source needs; neither adds a call or changes the march. Each evaluation is labelled with
its trial, position and role (`startup`, `predictor`, `corrector`, `diagnostic` or
`rejected`); a predictor's record is the response to the predicted field, not to the
corrected one the optical file saves. The optical file records the archive's path and
run identifier but never needs it, and no run reads it back.
{func}`~solprop.output.tdse_archive.open_tdse_archive` reads it one array at a time.

## How the research settings were chosen

The selected SAE research configuration keeps the provider's radial lattice, exterior
scaling, potential and ground state. Its other settings were chosen by four controlled
checks on a short segment near the compression point of a helium capillary compressor,
with a 120 nm–1.6 µm optical band and 33 radial nodes. Each check applied a 1% numerical
screen to full-band and 120–200 nm complex sources, fields and energies; passing it is
numerical agreement, not 1% physical accuracy.

| Quantity | Selected | Why |
|---|---|---|
| Current scale | 1.537 | direct weak-current carrier fits at 400, 800 and 1600 nm, with residuals against optmat of 0.027%, 0.045% and 0.339%; applying it to every higher order is an empirical extension |
| Spectral cut | 400 hartree | a 60-hartree cut fails the full-source screen, and 200 hartree the independent 1600 nm weak-current check |
| Angular cutoff | `lmax=29` | agrees with `lmax=79` on the compressed, entrance and launch fields |
| Sampling | `TDSESampling(32)`, atomic step $1.158\times10^{-19}$ s | agrees with a finer reference at half the step; the actual subdivision is read back from the provider |
| Physical window | 60 fs requested (68.3 fs realised) | a practical choice: 60, 120 and 240 fs histories disagree beyond the screen, so the window is a recorded limitation |
| Optical tolerance | `rtol=1e-2` | agrees with `3e-3` over the segment to 0.07% in the full band and 0.7% in 120–200 nm |

The edge-free form reduces the sensitivity of these sources to the gate's width 3 to
1000 times (3.48% to 0.82% over 120 nm–1.6 µm), but not to the history: the 60 fs window
still differs from 120 and 240 fs windows by about 8.7% over the full band.

The haCC presets are the provider's, with stated sensitivities in the closed current of
about 1.2% (accuracy), 0.95% (box) and 0.11% (gauge) on a strongly driven 800 nm
waveform. For haCC(1), a 6-node field-scale radial rule agrees with 16 nodes to 0.65%
over 90 nm–1.6 µm and 1.24% over 90–200 nm. Over an 81 mm segment ending at a capillary's
exit, haCC(1) and haCC(5) differ by 0.17% over the full band and 1.8% over 90–200 nm,
SAE and haCC agree to 0.3–0.4% and 1–3%, and the rate model gives 24–27% less 90–200 nm
energy than the atomic models. These are model differences over one segment, not errors
against an exact solution.

## Accuracy and limitations

- **Research status.** Every calibration record is `status="research"`, with no assigned
  uncertainty. Propagation has been demonstrated over short segments, about 8 cm each,
  near the end of helium capillary compressors, not over a full capillary, and an
  optical-tolerance study exists only for one of them.
- **The atomic window.** The current after the window is lost. For haCC at the default
  60 fs, the separately driven source differs from a 100 fs one by about 1.2% over the
  band, 3–4.5% above 900 nm and 0.4% over 90–200 nm; use
  `TDSEAtomicWindow(duration=100e-15)` where the near infrared matters. The SAE
  record's history sensitivity is larger (above).
- **The SAE scale** is empirical and sampled at three wavelengths; the ultraviolet
  references and the higher-order scaling are extrapolated, and the radial lattice and
  exterior scaling are unqualified.
- **The optical band** reaches 120 nm (SAE) or 90 nm (haCC) numerically; that is not a
  claim of validated ultraviolet accuracy.
- **Scope.** Helium only, one scalar HE₁₁ mode on the uncoupled modal capillary route,
  `numpy_ref` only (`numba_cpu` and the native backend refuse it by name). Double ionisation, ionic
  channels beyond $n=2$, ion motion, collisions, recombination and space charge are
  omitted.
- **Cost.** Each source evaluation is a full atomic calculation at every node: minutes
  per call on a many-core CPU or a data-centre GPU (fast haCC(1), six nodes, 60 fs:
  about five minutes on 32 cores), so a short segment of nine calls takes about an hour,
  and haCC(5) several times more. Size `memory_limit` and `max_atomic_calls` before a
  campaign.

See also §10.6 of the primer's [ionisation chapter](../../primer/ionisation.md),
[radial nodes for expensive responses](../waveguides/radial-quadrature.md), and the API
reference of {mod}`solprop.nonlinear.tdse`, {mod}`solprop.driver.tdse` and
{mod}`solprop.driver.tdse_consistency`.
