# SEP 37 — Shot noise on by default, and an integrator and tolerances chosen for each interface

| | |
|---|---|
| **SEP** | 37 |
| **Title** | Shot noise on by default, and an integrator and tolerances chosen for each interface |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-26) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-10-26 |
| **Resolution** | — |
| **Requires** | [SEP 16](archive/sep-0016-unified-multistep.md) (the multistep family and its work–precision qualification); [SEP 30](archive/sep-0030-standard-benchmark-suite.md) (the benchmark suite); [SEP 31](archive/sep-0031-simulation-coupling.md) (field planes and coupled stages) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Every solprop run is noiseless unless it asks for noise, every interface steps with
`Multistep()` by default, and the default tolerances differ between interfaces for
reasons that are only partly written down. This SEP proposes:

- `noise="modified"` as the default wherever the model can be carried;
- a default integrator chosen for each `prop_*` interface on measured work–precision,
  with no dependence on the physics a run switches on;
- default tolerances for each interface, set by a stated accuracy target;
- removal of every noise refusal that exists only because the transport was not built
  or not measured.

Tests and worked examples fix their seeds, so they stay reproducible. The SEP opens with
an exploration phase (phase 0) that reads the design record behind each choice and
measures what the new defaults cost before anything changes. Success is judged in four
ways:

- the modified model agrees with an exact reference where one exists;
- the chosen integrators win their work–precision comparisons, with noise on and off;
- the tolerances meet their targets;
- each lifted refusal is qualified like any other route.

### 2. Motivation and scope

**A noiseless default hides itself.** A user who does not pass `noise=` gets no shot
noise. Nothing is printed, and the run record says so only on inspection. Spontaneous
processes then do not start: modulational instability, Raman from noise and the
shot-to-shot incoherence of a supercontinuum. Coherent dynamics seeded by the pulse
itself look cleaner than they would in an experiment. The noise guide calls the modified
model the one to use "when the seeding is distributed", which covers most
supercontinuum and gas-filled hollow-core work. Luna.jl, whose interface solprop's keyword
surface follows, has shot noise on by default with the modified model.

**One default integrator does not suit every interface.** `Multistep()` became the
default everywhere with SEP 16 (ADR-045). The evidence was strongest where the source is
expensive: 5.5 to 5.8 times faster than IP-DOPRI5 on the full resonant-dispersive-wave
capillary and 2 to 3 times on a soliton-to-continuum launch. The same record has
IP-DOPRI5 ahead on the single-mode GNLSE window on `numba_cpu`, by 1.2 to 1.4 times.
With shot noise in the field the gap on the GNLSE window widens to 3.6–4 times
(preliminary, §3.4). A default that changes with the physics switched on would be
fragile, and the author rejects it. A default for each interface is not, because an
interface is chosen by the user and fixes the equation solved.

**The default tolerances are only partly explained.** `prop_gnlse` asks for
`rtol = 1e-8` and `atol = 1e-14` √W. Those were set with the first driver, and no
reason was recorded; they suit high-fidelity soliton evolution, and the soliton
tutorial asks for `1e-9` to show a recurrence to about $10^{-9}$. `prop_capillary` and
`prop_waveguide` ask for `1e-6` and `1e-10` √W, the values Luna.jl documents, and the
docstring justifies the looser value by the cost of a modal step. `prop_radial` and
`prop_free` have no default tolerance at all: their `SimulationSettings` take a
controller from the caller.

**In scope:**
- the defaults of `prop_gnlse`, `prop_capillary`, `prop_waveguide`, `prop_radial`,
  `prop_free` and `prop_planar`, and of the `Simulation.from_*` constructors behind them:
  noise model, integrator and formulation, `rtol` and `atol`;
- every place where a noise model is refused today (§3.3): a classification, and the
  implementation or measurement that lifts each refusal not grounded in physics;
- seeds for the test suite and the worked examples;
- the manual's account of noise, cost and stepper choice.

**Out of scope:**
- **New noise physics.** No truncated-Wigner vacuum, no Raman Langevin noise and no
  technical (pump) noise. §10 records why, and §12 asks whether a later SEP should take
  them up.
- **Noise under single precision.** It is refused for an accuracy reason, not a missing
  implementation (§3.3), and belongs to the precision work.
- **Start-of-run diagnostics.** These print the noise model, integrator and tolerances
  when a run begins. That is separate work and needs no SEP, though this SEP relies on
  it (§5.4).
- **Default changes elsewhere**, such as grids, save planes, absorbers or Raman.

### 3. Context

#### 3.1 The noise models

[Shot noise](../../guide/media/noise.md) describes the draw:
- one photon per active bin, mode and polarisation, with a deterministic modulus and a
  uniform random phase;
- masked to the active band on both grids;
- one generator stream per seed.

The same draw serves both models:
- **input** adds it to the field at $z = 0$;
- **modified** (ADR-008) injects it into the nonlinear operator at every stage, advanced
  by the unitary part of the cumulative linear operator, $N(z) = N_0\,e^{i\,\mathrm{Im}\,\Phi}$.

The realisation and its seed are stored with the output. A `seed` of `None` draws from
the operating system and records what it drew.

#### 3.2 The defaults today

| Interface | Noise | Integrator | `rtol` | `atol` (√W) |
|---|---|---|---|---|
| `prop_gnlse` | `None` | `Multistep()` | 1e-8 | 1e-14 |
| `prop_capillary` | `None` | `Multistep()` | 1e-6 | 1e-10 |
| `prop_waveguide` | `None` | `Multistep()` | 1e-6 | 1e-10 |
| `prop_radial`, `prop_free` | `None` | `Multistep()` | caller's controller | caller's controller |
| `prop_planar` | not carried | `Multistep()` | caller's controller | caller's controller |

#### 3.3 Where a noise model is refused today

| Where | The reason given today | Class |
|---|---|---|
| `precision="single"` | binary32 rounding can reach a one-photon seed at high pulse energy (SEP 33) | accuracy limit |
| the TDSE response | noise left out of the initial qualification (SEP 12) | not built |
| an interior launch, `initial_z > 0` | none recorded | not built |
| a coherent wall, the taper coupling, a split block, a moving solid guide | the draw's transport through a coupled, nonunitary flow is not measured ([roadmap](../roadmap.md)) | not built; needs a derivation |
| the hybrid planar grid | no photon measure is declared for slab amplitudes ([roadmap](../roadmap.md)) | not built |
| a general modal guide with a diagonal metric | no transport derived; nothing builds such a metric | not built, and moot |
| interpolated save planes (`interpolate_planes=True`) with `modified` | the draw is added outside the interpolant (SEP 14 §5) | not built |
| `processing.field_plane` of a `modified` run | the noise phase is not reconstructed into a plane (SEP 31, decision A2) | not built |

Only the single-precision refusal is grounded in physics or numerics. A binary32 state
cannot hold a seed below its rounding level. Even that is conditional: it binds only
when the seed is small against the pulse, at high energy. Every other refusal records
work that was never done.

#### 3.4 Preliminary measurements

These were taken while drafting, as same-process A/B runs on an Apple M5 Max laptop,
counting accepted steps. Wall times are indicative only.

**Capillary worked examples.** These cover SPM, soliton self-compression, four-wave
mixing, XPM, ionisation blue shift and two supercontinuum ensembles (16–256 µm bores,
µJ–mJ pulses). Each took the same number of steps, within 2 %, as written and with
`noise="modified"` forced: for example 2476 against 2476, and 3761 against 3792.

**The N = 3 GNLSE soliton tutorial** (20 pJ, 100 fs, 500–2000 nm band) behaves very
differently:

| Integrator, `rtol` | No noise | `input` | `modified` |
|---|---|---|---|
| `Multistep()`, 1e-9 | 1126 | 73199 | 73199 |
| `Multistep(formulation="interaction")`, 1e-9 | 5189 | 108759 | 108760 |
| `RungeKutta()`, 1e-9 | 1800 | 21845 | 21845 |
| `Multistep()`, 1e-6 | 467 | — | 18863 |
| `RungeKutta()`, 1e-6 | 507 | — | 4193 |

- **The noise model makes no difference to the cost**, contrary to the manual's "about
  eight times" for the modified model alone. That figure was quoted from Luna.jl and not
  reproduced in solprop.
- **`atol` makes no difference.** Running from `1e-14` to `1e-6` √W left the step
  counts unchanged.
- **The noise's size relative to the field is what matters.** Scaling the draw by $s$
  brings the RK count from 21845 at $s = 1$ through 8153 and 2219 to 1816 at
  $s = 10^{-6}$, against 1800 without noise. The draw is $7\times10^{-3}$ of the field's
  norm here.

**Accuracy** was measured against an RK reference at `rtol = 1e-11`, with the same draw.
The noise part is the run's departure from the noiseless run.

| Run | Steps | Field error | Error in the noise part |
|---|---|---|---|
| RK, adaptive, `rtol` 1e-9 | 21845 | 6e-10 | 1e-7 |
| RK, adaptive, `rtol` 1e-7 | 8152 | 4e-6 | 6e-4 |
| RK, adaptive, `rtol` 1e-6 | 4193 | 7e-5 | 1.1e-2 |
| RK, the noiseless run's schedule replayed | 1800 | 3e-4 | 5.7e-2 |
| `Multistep()`, adaptive, 1e-9 | 73199 | 5e-8 | 8e-6 |
| `Multistep()`, adaptive, 1e-6 | 18863 | 2e-4 | 3.2e-2 |
| `Multistep()`, the noiseless run's schedule replayed | 1126 | 5e-3 | 0.78 |

So the extra steps are needed. A schedule discovered without noise gets the noise part
wrong, by 6 % on RK and 78 % on `Multistep()`. ADR-008's advice that ensembles replay
fixed schedules therefore needs the schedule to be calibrated with noise. These numbers
come from one case on one machine. Phase 0 repeats them on the standard suite.

### 4. Theory

#### 4.1 What the two models represent

Both models are the semiclassical one-photon-per-mode seeding of the supercontinuum
literature [1, 2]: a classical field whose ensemble over random phases stands in for
vacuum fluctuations. Neither is a quantum model. The truncated-Wigner vacuum [3]
assigns half a photon per mode with a Gaussian amplitude, and adds Langevin terms for
loss and the Raman response.

**Input noise** propagates the seed with the field. It is dispersed, amplified and
attenuated. Under loss the seed decays below the vacuum level it stands for. In a
quantum treatment, loss couples the field to a reservoir whose vacuum re-enters the
mode (fluctuation–dissipation), so the noise floor does not decay.

**Modified noise** keeps the seed at one photon per mode at every $z$ and lets only the
unitary phase act on it. It is the classical analogue of that re-entering vacuum. Two
consequences follow:
- the seeding stays correct over a lossy guide;
- a background propagated from $z = 0$ cannot drive four-wave mixing that the physics
  does not have.

That is the case for making it the default. The case has two weak points, which phase 0
must address:
- the model's origin is credited to Chen and Wise, for a gain medium without
  dispersion, but solprop has not traced it to a publication;
- the size of the dispersive correction that ADR-008 adopted was reported elsewhere and
  has not been reproduced in solprop.

#### 4.2 An exact reference

In a lossless medium with anomalous dispersion, spontaneous modulational instability
from vacuum has a closed form in the linearised (Bogoliubov) approximation. A
continuous wave of power $P_0$ seeds sidebands at $\pm\Omega$ with gain
$g(\Omega) = |\beta_2\Omega|\sqrt{\Omega_c^2 - \Omega^2}$, where
$\Omega_c^2 = 4\gamma P_0/|\beta_2|$ [4]. The quantum treatment of the same linearised
equations fixes the sideband photon number grown from vacuum, as a function of
$\Omega$ and $z$. Phase 0 derives that expression, or cites it from a primary source,
before it is used as a tolerance. The comparison asks
how close each classical prescription comes to that expectation, per frequency and
along $z$, with and without dispersion and with and without loss. It answers whether
`modified` is the more principled default by measurement, not by argument.

#### 4.3 Why noise costs steps

A step controller holds the embedded difference below $\mathrm{rtol}\,\lVert y\rVert + \mathrm{atol}$
on the whole state. A seed of one photon per bin over $n$ active bins carries an energy
of about $n\hbar\bar\omega$ against the pulse's $U$, so its share of the norm is

$$ r \approx \sqrt{n\hbar\bar\omega / U} . $$

For the 20 pJ soliton, $n \approx 10^4$ and $U/\hbar\bar\omega \approx 8\times10^7$,
giving $r \approx 10^{-2}$, close to the measured $7\times10^{-3}$. For a µJ capillary
pulse $r \approx 10^{-4}$, and for a mJ pulse $10^{-5}$.

When $r \gtrsim \mathrm{rtol}$, the controller has to integrate the seed's own dynamics:
its beat against the pulse at large detuning, and its parametric growth. That costs
steps, whichever model carries the seed. When $r \ll \mathrm{rtol}$, the seed is below
what the controller resolves and costs nothing. This explains both halves of §3.4. It
also makes the cost of a noise default a matter of pulse energy against tolerance,
which phase 0 measures.

### 5. Proposed interface

#### 5.1 Noise

- **The default is `noise="modified"`** on every interface and constructor that carries
  noise. `seed=None` keeps its meaning: draw from the operating system and record it.
  `noise=None` stays available and switches noise off explicitly. `noise="input"` is
  unchanged.
- **A route that still refuses noise.** Until its refusal is lifted (§11), a run on such
  a route that did not ask for noise falls back to none. It records that fallback in its
  metadata and says so in the start-of-run summary. A run that asked for noise
  explicitly is still refused, as today. The fallback needs a default value that a
  caller cannot pass by mistake, a sentinel resolved per route. §12 asks whether a
  refusal would be better than a fallback.
- **Single precision** keeps refusing an explicit noise model. Under the default, it
  falls back to none, with the note.

```text
run = prop_capillary(150e-6, 1.0, "He", 0.5, lambda0=800e-9, energy=200e-6, fwhm=10e-15)
run.metadata.noise          # NoiseRecord(model='modified', seed=<recorded entropy>)
run = prop_capillary(..., noise=None)          # noiseless, as today's default
run = prop_capillary(..., seed=20261026)       # reproducible
```

#### 5.2 Integrators

Each interface gets its own default, written in its docstring and in one table in
[choosing a stepper](../../guide/numerics/choosing-a-stepper.md), and recorded in the run
configuration as today. The table is phase 1's result. The expectation from §3.4 and
ADR-045 is `RungeKutta()` for `prop_gnlse` and `Multistep()` for the modal and spatial
interfaces, but each entry is measured, not assumed. Nothing depends on the responses
switched on. A user who wants the other method passes `integrator=` as today.

#### 5.3 Tolerances

Each interface's default `rtol` and `atol` is set by a stated target: a global error on a
named observable of a canonical case, for example the output spectral energy density on
the in-band mask. The target is stated in the docstring, so the default can be checked.
`prop_radial` and `prop_free` gain defaults, so that their settings no longer need a
controller. §12 asks whether one target can serve every interface. It also asks whether
a run with noise should have its own default, which would make the tolerance depend on
a setting. The author's rule excludes dependence on the physics, and noise is a setting
of the run rather than physics, so the question stays open.

#### 5.4 What the user sees

The start-of-run summary, which is separate work, prints the resolved noise model and
seed, any fallback, the integrator and the tolerances. For a run with noise it also
prints the seed's share of the field norm, $r$ in §4.3, so that a slow low-energy run
explains itself.

### 6. Implementation strategies

**Layer.** The defaults are driver settings: `solprop.driver.interface` and
`solprop.driver.simulation`. The kernel contract and `SCHEMA_VERSION` are unchanged,
because the noise record and the integrator's representation are already in the
output. The fallback note may need one field in the noise record. If it does, the output
schema version changes, and phase 2 says so.

**Lifting the refusals** happens in the layers that refuse today. Each lift is its own
phase with its own qualification (§11):

| Refusal | Work | Verification |
|---|---|---|
| interpolated planes | add the draw at the interpolated position, as each stage does | landed and interpolated planes agree to the interpolant's error |
| field planes | reconstruct the noise phase into the exchanged plane (SEP 31 A2) | a coupled pair with modified noise matches the uncoupled run |
| interior launch | define what a launch at $z > 0$ carries: input noise is already in a coupled spectrum, and the modified draw starts at $z$ | an interior launch matches the same run from $z = 0$ to its tolerance |
| planar grid | declare the photon measure of a slab amplitude per unit width | photon number per bin is one, as on the other grids |
| coupled flows | derive the draw's transport through the nonunitary flow, probably its unitary polar factor, and measure it | the noise floor along a coupled guide stays at one photon per mode |
| TDSE | admit noise in the atomic source and measure its cost | the TDSE response with noise matches the PPT run's noise statistics within the response difference |

**Backends.** The noise injection exists on `numpy_ref`, `numba_cpu` and the native
backend. The lifts begin on `numpy_ref` and follow SEP 0's order. On `numba_cpu`, the
plasma statistics' stash is disabled under `modified` (it reads the committed field),
which becomes the default path. Phase 2 measures that cost and decides whether the stash
should learn the noise.

**Cost.** None on capillary-scale energies (§3.4). On low-energy GNLSE runs, 8 to 12
times the noiseless step count with RK, more with `Multistep()`. The defaults are chosen
with that cost known and printed. It is not hidden.

### 7. Verification and validation

- **The models against the exact reference** (§4.2). Use a lossless anomalous-dispersion
  continuous wave, then the same with loss. Compare the sideband photon number per
  frequency against the Bogoliubov expectation along $z$, for `input` and `modified`. The
  tolerance is set by the ensemble size: the standard error of the mean photon number
  over the seeds.
- **The noise floor under loss.** On a lossy guide with no pulse, the modified floor stays
  at one photon per mode and the input floor decays at the loss rate. The check is
  exact, to round-off.
- **Work–precision for each interface.** Use the standard suite (SEP 30) on short
  windows, with the core cases of each interface, with noise on and off. Compare
  `Multistep()` in both formulations against `RungeKutta()`, at matched error on the
  suite's observables and on the error in the noise part (§3.4). Run on one CPU thread
  and at the production width.
- **Tolerances.** For each interface's canonical case, map `rtol` to the global error on
  the target observable, and record the chosen default and its measured error.
- **Each lifted refusal**, as in the table of §6.
- **Ensemble statistics.** On a modulational-instability ensemble, check that the
  coherence $|g_{12}|$ and the mean spectrum converge in `rtol`. Then state the loosest
  `rtol` that keeps them within a stated fraction.

### 8. Documentation

- [Shot noise](../../guide/media/noise.md): the default, the cost measured in solprop
  in place of the quoted factor of eight, the share-of-norm rule of §4.3, and the result
  of §4.2.
- [Choosing a stepper](../../guide/numerics/choosing-a-stepper.md): the table of
  defaults for each interface and the evidence behind it.
- [Fixed schedules](../../guide/numerics/fixed-schedules.md) and ADR-008: calibrate an
  ensemble's schedule with noise on.
- The docstrings of every `prop_*` interface and `Simulation.from_*` constructor: noise,
  integrator and tolerances, each with its reason.
- The worked examples and tutorials: a fixed `seed` where noise is now on, and
  `noise=None` only where the lesson is noiseless.

### 9. Compatibility and impact

- **Every default run becomes stochastic.** Two runs of the same script differ
  realisation by realisation unless they fix `seed`. Users who compared outputs bitwise
  must pass a seed, or `noise=None`.
- **Existing results** change at the level of the noise for capillary-scale runs, and
  more visibly where noise seeds a spontaneous process. Low-energy GNLSE runs become
  slower (§3.4).
- **Tests.** About 65 test modules build runs without naming `noise`. Those comparing
  against a deterministic reference pass `noise=None`. A few tests cover the default
  itself.
- **The output schema** changes only if the fallback note needs a field (§6).
- **Luna.jl.** Shot noise on by default with the modified model is Luna.jl's own default,
  so the differences page loses an entry. A different default integrator for each
  interface has no Luna.jl counterpart and is recorded there.
- **Nothing is public yet**, so no deprecation path is needed. The changelog records the
  change.

### 10. Alternatives considered

- **Keep `None` and document it.** This is the status quo, and it is how the gap went
  unnoticed. It is rejected because noise is part of the physics being modelled.
- **`input` as the default.** It costs the same (§3.4), and its seed decays under loss.
  Phase 0 may still prefer it if §4.2 shows the modified model to be worse in lossless
  propagation.
- **A truncated-Wigner vacuum** (half a photon, Gaussian amplitude, Langevin terms). This
  is the principled quantum-statistical model [3], but it is new physics and its Raman
  Langevin term is a project of its own. It is left for a later SEP (§12).
- **A default integrator that depends on the physics switched on.** It was rejected by
  the author as fragile: a keyword elsewhere in a script would change the method.
- **An error norm that ignores the noise.** It cannot be built under the input model,
  where the seed is part of the state, and §3.4 shows the noise part needs the steps.

### 11. Phases

0. **Exploration and the record.** No code change. Read the archived SEPs, ADRs, the
   noise and stepper guides and the history behind each default and each refusal of
   §3.3, and record what each was based on. Trace the modified model's origin. Derive or
   cite §4.2's reference. Repeat §3.4 on the standard suite for every interface, with
   noise on and off, and with both integrators and formulations. Report, and revise
   Part I before review.
1. **Integrators and tolerances for each interface.** Set the defaults and their
   docstrings, and update the stepper table and the examples' re-runs. Verify with the
   work–precision and tolerance checks of §7.
2. **The noise default.** Add the sentinel, the per-route fallback with its note, the
   docstrings and the noise guide. Seed the tests and examples, and pin noiseless
   references. Verify with §4.2's comparison and the loss check, then re-run the
   examples.
3. **Interpolated planes, field planes and interior launches with noise.** Each piece is
   independently mergeable.
4. **The planar draw.**
5. **Noise through coupled flows:** the derivation, then the implementation and its
   qualification.
6. **Noise beside the TDSE response.**

Phases 3 to 6 retire fallbacks one by one. Each removes its route from the fallback list.

### 12. Open questions

- **Q1.** Is `modified` the right default, or should phase 0's comparison (§4.2) prefer
  `input` in lossless propagation? Decided by phase 0's measurement.
- **Q2.** On a route that still refuses noise, should the default fall back to none with
  a note (§5.1), or refuse and ask for `noise=None`? A refusal is honest, but it breaks
  every default run on those routes until phases 3 to 6 land.
- **Q3.** The integrator table for each interface. Phase 1.
- **Q4.** One accuracy target for every interface's tolerance, or one for each? And
  should a noisy run get its own default `rtol`, given §3.4's 1 % noise-part error at
  `1e-6`? Phases 0 and 1.
- **Q5.** Should a run whose seed share $r$ (§4.3) exceeds its `rtol` warn, or only
  print it? Phase 2.
- **Q6.** Is a coupled stage's source run a special case, defaulting to `input` until
  phase 3 lifts the field-plane refusal? Phase 2.
- **Q7.** Should a later SEP take up a truncated-Wigner vacuum and Raman Langevin noise?
  Deferred past this SEP.

### 13. References

1. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135 (2006). doi:10.1103/RevModPhys.78.1135
2. C. Adamu *et al.*, "Noise and spectral stability of deep-UV gas-filled fiber-based
   supercontinuum sources driven by ultrafast mid-IR pulses," *Sci. Rep.* **10**, 4912
   (2020). doi:10.1038/s41598-020-61847-w
3. P. D. Drummond and J. F. Corney, "Quantum noise in optical fibers. I. Stochastic
   equations," *J. Opt. Soc. Am. B* **18**, 139 (2001). doi:10.1364/JOSAB.18.000139
4. G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019), chapter 5.

## Part II — Implementation record

## Part III — History

- 2026-10-26: Drafted, with the preliminary measurements of §3.4.
