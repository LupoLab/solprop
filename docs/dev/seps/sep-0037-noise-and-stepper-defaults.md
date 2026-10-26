# SEP 37 — Shot noise on by default, and an integrator and tolerances chosen for each interface

| | |
|---|---|
| **SEP** | 37 |
| **Title** | Shot noise on by default, and an integrator and tolerances chosen for each interface |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-26) |
| **Status** | Draft |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
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

- the modified model passes an exact identity with the input model where no reservoir
  acts, and an exact quantum reference for spontaneous modulational instability;
- the chosen integrators win their work–precision comparisons, with noise on and off;
- the tolerances meet their targets;
- each lifted refusal is qualified like any other route.

### 2. Motivation and scope

**A noiseless default hides itself.** A user who does not pass `noise=` gets no shot
noise. Nothing is printed, and the run record says so only on inspection. Spontaneous
processes then do not start: modulational instability, Raman from noise and the
shot-to-shot incoherence of a supercontinuum. Coherent dynamics seeded by the pulse
itself look cleaner than they would in an experiment. Luna.jl, whose interface solprop's
keyword surface follows, has shot noise on by default with the modified model.

**The modified model is the principled default** (§4.1). Without a reservoir, that is
with a lossless, gainless linear operator, it is the input model written in shifted
coordinates. The two are one equation and cannot make different predictions; the
modified field simply leaves out the seed's floor. Where a reservoir acts, they differ,
and the modified model is the right one: linear loss, Raman scattering, gain and plasma
absorption all couple the field to a reservoir. A seed placed at the input is
attenuated by those losses below the vacuum level it stands for, and it puts a
one-photon floor under the whole spectrum. The modified seed is held at the vacuum level
and seeds afresh wherever the reservoir acts.

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
reason was recorded. They suit high-fidelity soliton evolution: the soliton tutorial
asks for `1e-9` to show a recurrence to about $10^{-9}$. But noiseless soliton
recurrence is an occasional study, and a user who wants it can ask for a tight
tolerance. The GNLSE is used mostly for supercontinuum, where a looser tolerance is
enough, and that tolerance should be judged by a convergence test on the canonical
supercontinuum parameters of Dudley, Genty and Coen [1].

`prop_capillary` and `prop_waveguide` ask for `1e-6` and `1e-10` √W, the values Luna.jl
documents; the docstring justifies the looser value by the cost of a modal step.
Production practice is looser still: `rtol = 1e-5` is common for general use, and runs
with the TDSE response use `1e-3` to `3e-3`. `1e-6` is kept only if a stated
convergence test supports it.

`prop_radial` and `prop_free` have no default tolerance at all: their
`SimulationSettings` take a controller from the caller.

**In scope:**
- the defaults of `prop_gnlse`, `prop_capillary`, `prop_waveguide`, `prop_radial`,
  `prop_free` and `prop_planar`, and of the `Simulation.from_*` constructors behind them:
  noise model, integrator and formulation, `rtol` and `atol`;
- every place where a noise model is refused today (§3.3): a classification, and the
  implementation or measurement that lifts each refusal not grounded in physics;
- seeds for the test suite and the worked examples;
- the manual's account of noise, cost and stepper choice.

**Out of scope:**
- **New noise physics.** No half-photon (truncated-Wigner) vacuum, no Langevin terms
  for Raman or loss, no thermal phonon occupation and no technical (pump) noise. §4.1
  says what each would change, and §12 asks whether a later SEP should take them up.
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
  reproduced in solprop. This case has no reservoir, so the two models are the same
  equation (§4.1); equal step counts are what the theory predicts.
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

**Where one photon per mode comes from.** The seed is not the vacuum itself. In the
symmetrically ordered (Wigner) picture that a classical field simulation implicitly
uses, the vacuum carries half a photon per mode [3]. One photon per mode is the
equivalent input noise of a phase-insensitive amplifier. Smith showed that the
spontaneous Raman or Brillouin emission along a fibre, summed and amplified by the gain
remaining after each point, is the same, at high gain, as injecting one photon per mode
at the input and propagating only the stimulated equations [4]. The supercontinuum
literature adopted this seed with a random phase on every spectral bin [1, 2].

**Two kinds of process.** A phase-sensitive (parametric) process such as modulational
instability or four-wave mixing is unitary. It has no reservoir, and the vacuum enters
once, at the input. A phase-insensitive process (Raman gain, amplified spontaneous
emission, linear loss, plasma absorption) couples the field to a reservoir. By the
fluctuation–dissipation theorem, that reservoir feeds fluctuations in continuously.

**The modified model** moves the seed $N_0$ out of the initial condition and into the
nonlinear and gain operators [5]. As first published it held the seed fixed in $z$. In
a dispersive medium that is wrong, because the seed and the field it seeds drift out
of phase. solprop's form advances the seed by the unitary part of the linear operator,
$N(z) = N_0\,e^{i\,\mathrm{Im}\,\Phi(\omega, z)}$ (ADR-008), and passes the full field
$y + N$ to the nonlinear response.

**Without a reservoir, the two models are one equation.** Let $\chi$ be the input
model's field, with
$\partial_z \chi = \hat D\chi + \hat N(\chi)$ and $\chi(0) = y_0 + N_0$, where $\hat D$ is
linear and unitary and $\hat N$ is any nonlinear operator: Kerr, shock, Raman's real
part. Define $\psi = \chi - N(z)$, with $\partial_z N = \hat D N$. Then, by linearity of
$\hat D$,

$$ \partial_z \psi = \hat D \psi + \hat N(\psi + N), \qquad \psi(0) = y_0 , $$

which is the modified model. The identity holds to all orders in the seed, not only to
linear order. The modified field contains only light that has interacted. Its spectrum
has no floor, and the background is restored on reading, from the stored noise phase.

**With a reservoir, the two differ, and the modified model is the physical one.** If
$\hat D$ includes loss, the modified seed keeps its modulus, by the
fluctuation–dissipation argument above, and the shift acquires a term that re-injects
vacuum at the loss rate. A seed propagated from the input is instead attenuated below
the vacuum. Under gain, the seed's growth belongs to the source term, not to the seed.
A Raman response is lossy on the anti-Stokes side and supplies its own spontaneous
Stokes scattering, and so is a reservoir in the same sense. So is plasma absorption.
This is why the modified model is not just a presentation of the input model: it adds
physics wherever a reservoir acts, and it changes nothing where none does.

**What neither model gets right.** One photon per mode is the high-gain shortcut for
phase-insensitive processes. For a parametric process the exact seed is half a photon,
so both models overestimate spontaneous parametric power by a factor of two at high
gain [3, 6]. Because modulational instability enters a supercontinuum logarithmically,
this shifts its development length by about $\ln 2/(2\gamma P_0)$, a few per cent of
$L_\mathrm{MI}$. The scheme that is exact for both kinds of process is half a photon at
the input, plus a symmetrically ordered Langevin term for every reservoir: Raman
(with its thermal occupation $n_\mathrm{th}$, which is large for the rotational shifts
of nitrogen) and loss [3]. That is new physics, outside this SEP. Separately, solprop's
draw has a fixed modulus and a random phase. A circular Gaussian amplitude has the same
second moments, but it is the correct choice where single-mode intensity statistics
matter, as in transient Raman initiation, whose seed occupies about one temporal mode
(§12, Q8).

#### 4.2 Exact references

**The identity.** In a run with no reservoir (Kerr, no loss, no Raman), an `input` run
must equal the `modified` run with the transported seed added back, to rounding, for
every seed. That is an exact test of solprop's implementation of ADR-008, including how
the full field reaches the nonlinear response inside each stage.

**Spontaneous modulational instability from vacuum.** Linearise the normalised NLSE
$i\partial_z\psi + \tfrac12\partial_\tau^2\psi + |\psi|^2\psi = 0$ about a continuous wave of
power $P$. The sideband pair at $\pm\Omega$ then undergoes a Bogoliubov transformation
with $D = \Omega^2/2 - P$ and $K = \sqrt{P^2 - D^2}$, and the spontaneous sideband photon
number after a length $L$ is exactly

$$ \langle n \rangle = |\nu|^2 = \frac{P^2}{K^2}\sinh^2 KL $$

[6, 7]. A classical seed of $n_0$ photons per mode, propagated by the same linearised
equation, gives $(|\mu|^2 + |\nu|^2)\,n_0 = (1 + 2|\nu|^2)\,n_0$. With $n_0 = 1$ that is the
target for both solprop models, per frequency across the gain band and along $z$. The
same comparison with a lossy linear operator then separates the two models.
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

Each interface's default `rtol` and `atol` is set by a stated convergence test on a
canonical case of that interface, and the test is named in the docstring, so the
default can be checked.

- **`prop_gnlse`**: the supercontinuum of Dudley, Genty and Coen [1]. The default is
  the loosest `rtol` at which the mean output spectrum and the coherence $|g_{12}|$ of a
  seeded ensemble have converged to a stated fraction. It is expected to be looser than
  today's `1e-8`. A noiseless soliton-recurrence study asks for its own, tighter value,
  and the manual says so.
- **`prop_capillary` and `prop_waveguide`**: the same test on the standard suite's
  capillary cases. `1e-6` stays if the test supports it. Common practice, `1e-5`, is
  measured beside it.
- **`prop_radial` and `prop_free`** gain defaults by the same test on their canonical
  cases, so that their settings no longer need a controller.
- **A response does not set a default.** The TDSE response is usually run at `1e-3` to
  `3e-3`, but a default that changed with the response switched on would break the
  rule of §5.2. The manual records the practice instead.

`atol` is set with `rtol`. §3.4 shows it does not affect noisy runs; it matters for weak
fields.
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

- **The identity** (§4.2). On Kerr-only, lossless cases of `prop_gnlse`,
  `prop_capillary` and `prop_free`, and on every backend, an `input` run equals the
  `modified` run plus the transported seed. The bound is rounding, scaled by the step
  count.
- **The models against the exact reference** (§4.2). Use a lossless anomalous-dispersion
  continuous wave. Compare the sideband photon number per frequency against
  $1 + 2|\nu|^2$ along $z$, for `input` and `modified`. The tolerance is set by the
  ensemble size: the standard error of the mean photon number over the seeds. Then add
  loss and check that the models separate as §4.1 predicts.
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
- **Ensemble statistics.** On the canonical supercontinuum [1] and a capillary
  ensemble, check that the coherence $|g_{12}|$ and the mean spectrum converge in `rtol`.
  Then state the loosest `rtol` that keeps them within a stated fraction (§5.3).
- **Conservation.** In lossless runs, the drift of $\int |y|^2\,d\tau$ with noise on is a
  cheap discriminator between a correct and an inconsistent seed treatment. Record it
  beside each comparison.

### 8. Documentation

- [Shot noise](../../guide/media/noise.md): the default; the model's origin [4, 5] in
  place of "not traced"; the equivalence without a reservoir and the difference with
  one (§4.1); the cost measured in solprop in place of the quoted factor of eight; the
  share-of-norm rule of §4.3; the results of §4.2; and what neither model does (the
  factor of two in parametric seeding, thermal Raman occupation).
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
- **`input` as the default.** Without a reservoir it is the same equation as the
  modified model and costs the same (§3.4, §4.1). With a reservoir it is the less
  physical of the two: its seed decays under loss, and it leaves a floor under the
  whole spectrum. Rejected.
- **A truncated-Wigner vacuum** (half a photon, Gaussian amplitude, Langevin terms for
  each reservoir). This is the scheme that is exact for parametric and
  phase-insensitive seeding alike [3], but it is new physics, and its Raman Langevin
  term is a project of its own. It is left for a later SEP (§12, Q7).
- **A default integrator that depends on the physics switched on.** It was rejected by
  the author as fragile: a keyword elsewhere in a script would change the method.
- **An error norm that ignores the noise.** It cannot be built under the input model,
  where the seed is part of the state, and §3.4 shows the noise part needs the steps.

### 11. Phases

0. **Exploration and the record.** No code change. Read the archived SEPs, ADRs, the
   noise and stepper guides and the history behind each default and each refusal of
   §3.3, and record what each was based on. Run the identity of §4.2 on the existing
   implementation (Q1). Repeat §3.4 on the standard suite for every interface, with
   noise on and off, and with both integrators and formulations, and run the
   supercontinuum convergence test of §5.3 at today's tolerances. Report, and revise
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

- **Q1.** Does solprop's modified model pass the identity of §4.2 on every route and
  backend, including how the seed reaches each stage of `Multistep()`, whose source
  history is sampled at the stages? Phase 0, then phase 2.
- **Q2.** On a route that still refuses noise, should the default fall back to none with
  a note (§5.1), or refuse and ask for `noise=None`? A refusal is honest, but it breaks
  every default run on those routes until phases 3 to 6 land.
- **Q3.** The integrator table for each interface. Phase 1.
- **Q4.** The convergence target behind each interface's tolerance (§5.3): one
  fraction for every interface, or one for each? Phases 0 and 1.
- **Q5.** Should a run whose seed share $r$ (§4.3) exceeds its `rtol` warn, or only
  print it? Phase 2.
- **Q6.** Is a coupled stage's source run a special case, defaulting to `input` until
  phase 3 lifts the field-plane refusal? Phase 2.
- **Q7.** Should a later SEP take up the half-photon vacuum with Langevin terms for
  Raman (including its thermal occupation) and loss, which fixes the factor of two in
  parametric seeding (§4.1)? Deferred past this SEP.
- **Q8.** Should the draw's amplitude become a circular Gaussian, with the same second
  moments, so that single-mode statistics such as transient Raman initiation come out
  right (§4.1)? It changes every seeded realisation, so if it is wanted, it belongs in
  phase 2, with the default.
### 13. References

1. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135 (2006). doi:10.1103/RevModPhys.78.1135
2. C. Adamu *et al.*, "Noise and spectral stability of deep-UV gas-filled fiber-based
   supercontinuum sources driven by ultrafast mid-IR pulses," *Sci. Rep.* **10**, 4912
   (2020). doi:10.1038/s41598-020-61847-w
3. P. D. Drummond and J. F. Corney, "Quantum noise in optical fibers. I. Stochastic
   equations," *J. Opt. Soc. Am. B* **18**, 139 (2001). doi:10.1364/JOSAB.18.000139
4. R. G. Smith, "Optical power handling capacity of low loss optical fibers as
   determined by stimulated Raman and Brillouin scattering," *Appl. Opt.* **11**, 2489
   (1972), appendix A. doi:10.1364/AO.11.002489
5. Y.-H. Chen and F. Wise, "A simple accurate way to model noise-seeded ultrafast
   nonlinear processes," arXiv:2410.20567 (2024). doi:10.48550/arXiv.2410.20567
6. M. J. Potasek and B. Yurke, "Squeezed-light generation in a medium governed by the
   nonlinear Schrödinger equation," *Phys. Rev. A* **35**, 3974 (1987).
   doi:10.1103/PhysRevA.35.3974
7. E. Brainis, D. Amans and S. Massar, "Scalar and vector modulation instabilities
   induced by vacuum fluctuations in fibers: numerical study," *Phys. Rev. A* **71**,
   023808 (2005). doi:10.1103/PhysRevA.71.023808

## Part II — Implementation record

## Part III — History

- 2026-10-26: Drafted, with the preliminary measurements of §3.4.
- 2026-10-26: Revised after the author's comments. §4 now rests on the modified model's
  origin and its exact equivalence to the input model without a reservoir, with a
  quantum reference for spontaneous modulational instability. The tolerance targets
  follow production practice. Q1 and Q8 are new, and the factor of two in parametric
  seeding is recorded as deferred physics.

### Human review — 2026-10-26

John Travers confirmed that he has read the proposal and that his human review is
complete. Recorded at his request; no acceptance decision or resolution of the AI
findings below is inferred from that confirmation.

### AI review — 2026-10-26

Reviewer: GPT-6 (AI). Reviewed all of Part I at PR #30, revision `653ac95`, against
the current implementation, conventions, SEP 0 and the sources identified below.
**Recommendation: revise before acceptance.** The motivation for measured defaults is
sound, but the reservoir argument, observable definitions and qualification contract
need correction or decisions. `Reviewed` records completed reviews, not acceptance.
Part I and Part II are unchanged. All R1–R14 below remain open for the drafting
agent's response and the maintainer's decision.

#### Findings and questions

1. **R1 — P1: The claimed reservoir distinction does not follow from the equations
   (§§2, 4.1, 10).** The change of variables works for *any* deterministic nonlinear
   functional when the draw solves the full homogeneous linear equation. In particular,
   a delayed Raman response or nonlinear plasma absorption inside $\hat N$ does not
   invalidate it. With unitary $\hat D$, the two reconstructed fields remain identical
   even when that nonlinear response is dissipative. The Raman probe below confirms
   this in the existing implementation. Conversely, write $\hat D=\hat D_u+\hat D_r$
   and $N'=\hat D_u N$. The reconstructed modified field $\chi=\psi+N$ obeys
   $\chi'=\hat D\chi+\hat N(\chi)-\hat D_rN$. The last term is a forcing correlated
   with the *same initial draw*, not newly sampled reservoir fluctuations. For pure
   loss with power coefficient $\alpha$, its forcing covariance is proportional to
   $\alpha^2\langle N(z)N(z')^*\rangle/4$, not to a delta function in $z-z'$.
   Drummond–Corney [3], section VI.C (arXiv equations 75–77), specifies the latter for
   gain/loss and Raman noise. This does not preclude matching particular observables:
   the corrected readout in R3 gives the exact mean $G-1$ for constant scalar gain
   from vacuum. It is not a general reservoir replacement. For example, follow that
   gain by a passive attenuator of power transmission $\eta$, attenuating $\psi$
   while retaining the same unattenuated $N$, with $\langle|N|^2\rangle=1$.
   The corrected readout becomes
   $[1+\sqrt{\eta}(\sqrt G-1)]^2-1$, whereas the amplified spontaneous photons must
   become $\eta(G-1)$. With $G=4$ and $\eta=1/4$, these are 1.25 and 0.75 photons.
   This is an analytic counterexample to the general claim, not a solprop amplifier
   simulation. **Recommendation:** state the actual algebraic condition
   for equivalence, distinguish physical reservoirs from their deterministic response,
   and describe the current modified model as an approximation with qualified scope.
   A constant restored floor alone cannot establish fluctuation–dissipation accuracy.
   What reservoir observables, beyond that floor, justify choosing this default?

2. **R2 — P1: Separate the cited models from solprop's variant (§§4.1, 8, 10).**
   Chen–Wise [5], equations 2a–2c, keeps only terms first order in the noise and
   separates emission gain from net gain. Solprop evaluates the full nonlinear
   response of $\psi+N$. Keeping the full response is mathematically justified for
   the exact change of variables in R1; this finding does not recommend replacing it
   with the paper's truncated expansion. The differences include the nonlinear
   expansion and gain treatment, as well as phase transport and the draw distribution;
   the paper's results cannot qualify this variant by attribution alone. Also, a
   truncated-Wigner
   treatment is not generally exact for nonlinear quantum propagation: [3], section VI,
   explicitly drops higher derivative terms at large photon number. **Recommendation:**
   enumerate these differences, identify which reference results actually apply, and
   limit the exact half-photon statement to the linearised Bogoliubov problem with the
   correct observable ordering. Keep a full reservoir model as separate physics work.

3. **R3 — P1: Define the measured field before specifying spectra, coherence and
   conservation (§§4.1–4.2, 5.3, 7).** The saved modified field is $\psi$, whereas the
   identity concerns $\chi=\psi+N$. The current `processing.fields.get_spectrum` reads
   the saved field through `processing.axes._plane_field`; it does not automatically
   add back the noise. Subtracting a noise *amplitude* also does not subtract a photon
   background from an intensity: the interference term remains. For a linearised
   sideband $\chi_s=\mu N_s+\nu N_i^*$ and transported background $uN_s$, with
   independent input phases and $|u|=1$, the two predictions are

   $$\langle|\chi_s|^2\rangle=(|\mu|^2+|\nu|^2)n_0,\qquad
   \langle|\psi_s|^2\rangle=(|\mu-u|^2+|\nu|^2)n_0.$$

   Thus $1+2|\nu|^2$ is the target for the reconstructed field, not the saved modified
   spectrum, when $n_0=1$. At the entrance these are already $n_0$ and zero.
   [Chen–Wise v5, Appendix 2, equation A8](https://arxiv.org/html/2410.20567v5)
   explicitly prescribes $|\psi+N|^2-|N|^2$ for the constant-gain example. With
   $\psi=(\sqrt G-1)N$, its ensemble mean is $n_0(G-1)$, giving the correct
   spontaneous mean for $n_0=1$ even at low gain. That estimator must be distinguished
   from both the raw saved spectrum and the classical MI target containing the seed;
   for vacuum MI, the symmetrically ordered half-photon calculation instead subtracts
   one half from the mean reconstructed intensity. Likewise,
   $\int|\psi|^2dt$ need not be conserved in a lossless Kerr run; the probe below has
   a relative change of $1.75\times10^{-7}$ while the full-field energy changes only
   at rounding level. **Recommendation:** name the field and its power/flux metric for
   every observable, specify any reconstruction API and its Fourier convention, and
   derive the appropriate conservation test. The quantum result $|\nu|^2$, the
   classical one-photon result and any normally ordered estimator are distinct targets.
   For the MI reference, also supply its SI scaling, pump frame, undepleted-pump range
   and the finite $K\to0$ limit; a lossy pump needs a separately derived reference.

4. **R4 — P1: An exact continuous identity is not an adaptive rounding contract
   (§§3.4, 4.2, 7; Q1).** `numpy_ref.passes.error_norm` scales by the propagated
   state, which is $\chi$ for input noise and $\psi$ for modified noise. Its denominators
   can differ, so acceptance, step sizes and Adams order/history can differ. Equal step
   counts in one example are evidence about that example, not a consequence of the
   identity. Further, a physically lossless fibre still has numerical absorption by
   default; the spectral collar is part of the linear exponent and is omitted from
   the draw's transport. **Recommendation:** establish a rounding comparison with
   matched methods, prescribed steps and Adams histories/orders, constant metrics and
   disabled absorbers. Derive its budget from operations, transforms and conditioning,
   not step count alone. Test independent adaptive runs against a converged reference
   with an accuracy envelope, as SEP 25 requires. Add separate boundary tests rather
   than calling a default-collar run lossless. The small RK probe is not qualification
   of the other routes, Adams formulations or compiled backends.

5. **R5 — P1: Phase-only transport does not preserve photons in a changing metric
   (§§3.3, 4.1, 6–7).** Conventions §4 divides a modal draw by $\sqrt{w(0)}$, with
   $w=\operatorname{Re}n_\mathrm{eff}$, while observations multiply by $\sqrt{w(z)}$.
   Transporting that internal draw only by phase therefore gives a restored energy
   per bin of $\hbar\omega\,w(z)/w(0)$, not $\hbar\omega$, even in a lossless guide.
   The real normalisation term $-\tfrac12\log[w(z)/w(0)]$ is not physical absorption.
   A varying radial admittance has the corresponding issue. This case is not confined
   to the coupled-flow refusal. **Recommendation:** explicitly distinguish metric
   transport from dissipation, add varying-index/admittance cases to phase 0, and
   decide whether this SEP changes that transport or restricts its default/claims.
   A physical photon-floor assertion needs the local flux metric, not just an unchanged
   internal modulus.

6. **R6 — P1: Specify a composable stochastic transport before lifting coupled-flow
   refusals (§6, phase 5).** The unitary polar factor of a nonunitary propagator
   generally does not compose: $U(F_2F_1)\ne U(F_2)U(F_1)$. A rule that takes a polar
   factor per numerical step can therefore change the noise when the step schedule,
   a save plane or a restart changes. For example, the passive contractions
   $F_1=\operatorname{diag}(1,1/2)$ and
   $F_2=\left(\begin{smallmatrix}3/4&1/4\\1/4&3/4\end{smallmatrix}\right)$
   each have unitary polar factor $I$, but the factor of $F_2F_1$ differs from $I$
   by 0.1564 in Frobenius norm (a direct two-by-two numerical check).
   Preserving a total noise norm does not establish
   the correct correlations or the covariance in a nonorthogonal power metric.
   **Recommendation:** derive the transport in a declared physical basis, specify
   its dependence on the continuous path and any reservoir approximation, and test
   composition, subdivision, basis changes and restart covariance. Retain the refusal
   until that derivation exists; if reservoir increments are needed, acknowledge the
   new physics rather than treating this as merely missing plumbing.

7. **R7 — P1: A coupled or interior launch must preserve the noise decomposition
   (§6, phase 3; Q6).** Restoring $N$ into a source plane and then drawing a new
   modified seed at the receiving launch adds a second background to the next
   nonlinear evaluation. Restarting the shifted field with a new independent seed
   instead loses the existing correlations. Merely saying that the draw starts at
   $z>0$ does not define a continuation. `Simulation.with_input(Launch)` currently
   clears implicit noise specifically to avoid a second floor. **Recommendation:**
   define whether a plane carries the full field, the shifted field plus its draw, or
   an explicitly converted representation; preserve the same draw and its phase origin
   through identity coupling and continuation. Test a split run at multiple positions
   against the unsplit run, plus lossy coupling and regridding under the chosen model.
   Audit the saved-output and seam changes this requires: §6 cannot yet promise an
   unchanged `SCHEMA_VERSION`, or an output change limited to a fallback note.

8. **R8 — P2: The fallback contract is incomplete and can silently change the
   experiment (§§5.1, 5.4, 6; Q2).** Changing only `precision`, interpolation or the
   response can turn default noise off; changing a backend must not do so. A metadata
   note is not visible in a run with progress disabled, and the start summary this
   depends on is expressly outside the SEP. With an explicitly supplied `seed` but
   omitted `noise`, a fallback must not silently discard the user's seeding intent.
   Field-plane extraction happens after propagation, so the source cannot resolve a
   different default based on an unknown future use. **Recommendation:** prefer an
   actionable refusal requiring explicit `noise=None` on unsupported configurations;
   no published compatibility constraint is cited here. If fallback is chosen,
   specify requested and resolved model, reason, seed handling and durable metadata,
   and make its user notification a delivery prerequisite. Test omission, explicit
   `None`, explicit models and supplied seeds at each entry point; never catch a
   backend construction error and reinterpret it as unsupported noise.

9. **R9 — P1: Fix a convergence protocol before selecting defaults (§§5.3, 7,
   phases 0–1; Q3–Q4).** Naming the Dudley–Genty–Coen review [1] does not select a
   unique simulation: the exact parameters, figure/case, grid, length, responses,
   spectral region and observable are needed. A relative tolerance on coherence is
   ill-defined near zero, and relative spectral error is ill-defined in empty bins.
   Comparing independent seed ensembles can hide integration bias inside sampling
   error; a one-standard-error pass threshold would also fail frequently by chance.
   **Recommendation:** declare paired seed sets, a refined deterministic reference for
   each seed, three refinement levels, held-out ensemble members, signal floors,
   absolute coherence and suitable spectral error measures, and an explicit confidence
   rule. Choose observable-specific accuracy budgets before the measurement, with
   the same protocol across interfaces, and separate integration, grid, model and
   Monte Carlo errors. Validate reused schedules across seeds, not only the seed that
   discovered them. Use bounded windows/stored witnesses for tests and timings; put
   any necessary full supercontinuum study in a permitted gallery workflow.

10. **R10 — P2: The norm ratio is a heuristic, not a step-cost criterion (§§3.4,
    4.3, 5.4, 6; Q5).** Pure linear propagation can be integrated exponentially without
    additional source error at any $r/\mathrm{rtol}$. Conversely, an initially tiny
    perturbation can grow into a resolved instability. The controller uses its chosen
    band and the internal state norm, not generally a whole-band physical-energy norm;
    spatial measures, varying indices, mode and polarisation counts matter. The
    quoted capillary estimate $10^{-4}$ also exceeds its quoted $10^{-6}$ tolerance
    despite the reported small step-count change. **Recommendation:** label $r$ as an
    initial size diagnostic with stated assumptions, and measure the actual controller
    norm if it is printed. Do not trigger a warning or infer zero cost from this
    ratio alone. Supply the revision, backend, seeds, grids, measured windows, norm
    definitions, wall-time samples and uncertainty behind §3.4 before using it to
    choose a default. Accepted steps alone do not establish runtime, especially once
    the modified-noise statistics path disables the Numba stash.

11. **R11 — P1: Planar noise requires a transverse normalisation decision (§§3.3,
    6, phase 4).** An energy per unit width is in J/m, while a photon's energy is in
    J. A slab metric alone cannot turn one into the other. The existing `LineGrid`
    manual explicitly chooses one photon per column and bin *per metre of y*; that is
    a declared convention, not a width-independent quantum vacuum prescription.
    **Recommendation:** decide whether planar noise adopts that same convention or
    introduces a finite quantisation/effective width, derive its amplitude and units,
    and document what changing that width does to fluctuations. Test a matching line
    or finite-width reference with the same measure. Keep the refusal until that
    choice is made; do not classify the missing physical measure as purely mechanical.

12. **R12 — P2: TDSE/PPT agreement is not an independent noise qualification
    (§6, phase 6).** The two responses can have different spectra, memory and
    ionisation, so "within the response difference" supplies no defined bound. Noise
    also broadens the atomic drive and may probe frequencies beyond the response's
    qualified window. **Recommendation:** name the TDSE limits and observables that
    are checked independently (zero drive, weak-field response and a deterministic
    perturbed-drive reference), specify bandwidth and convergence budgets, and retain
    the common supplied realisation through the atomic source. Treat PPT as a
    comparison in a stated shared regime, with a quantified model discrepancy, rather
    than the acceptance oracle. If that cannot be specified now, defer this lift.

13. **R13 — P2: Resolve the API and acceptance boundary before implementation
    (§§5.2–5.3, 6, 11).** `prop_radial`, `prop_free` and `prop_planar` take a caller's
    `SimulationSettings`, whose controller is currently required; defaulting the
    tolerances needs a public signature/precedence design. The scope includes planar
    and general modal constructors, but the proposed tolerance list does not cover
    them explicitly. Phase 0 says to revise before review, while the concrete
    integrator/tolerance table is postponed to phase 1 after acceptance would freeze
    Part I. Phase 3's "independently mergeable" also needs reconciliation with SEP 0's
    single implementation PR. **Recommendation:** complete phase 0, record the default
    matrix and the rules for omitted settings versus explicit controllers, and request
    a response review before acceptance. Alternatively freeze a fully specified
    selection rule with approval gates. Keep all implementation phases on the one
    branch/PR, or explicitly split the later physics proposals and retain their
    roadmap work. Reassess schema impacts after R5–R7 and R11 have decisions.

14. **R14 — P2: A Gaussian draw is a separate model choice, not just a seed
    migration (§§4.1, 9; Q8).** Fixed modulus and circular Gaussian draws share second
    moments but have different fourth moments: for mean occupation $n_0$,
    $\langle|N|^4\rangle=n_0^2$ and $2n_0^2$, respectively. That changes nonlinear
    statistics as well as individual realisations. The relevant basis matters:
    for $c=\sum_k u_kN_k$, independent equal-occupation phase draws and
    $\sum_k|u_k|^2=1$ give
    $\langle|c|^4\rangle=n_0^2(2-\sum_k|u_k|^4)$.
    A single amplified temporal mode can combine many spectral bins and approach
    Gaussian statistics; it is not necessarily a single fixed-modulus draw.
    Determine the projections onto the actual gain modes before inferring an error
    in pulse-energy tails from the spectral-bin fourth moment. Changing the draw
    also changes the existing exact-per-bin photon test and the declared-basis
    sampling contract.
    **Recommendation:** retain the current distribution for a defaults-only change,
    with its limitations stated, or explicitly extend the scope and qualify both the
    distribution and the observables. In the latter case record the distribution and
    its version, preserve old-file/restart interpretation, and test moments, tails,
    transformed covariance and sampling uncertainty. A seed alone must not be claimed
    to guarantee bitwise reproducibility across backends or adaptive histories (SEP 25).

#### Recommendations on the existing open questions

These are reviewer recommendations, not maintainer resolutions.

| Question | Context and recommendation |
|---|---|
| Q1 — identity | Make the corrected identity a phase-0 prerequisite. The short RK probe passes, including Raman; all routes, grids, Adams formulations and backends still need the R4 qualification. |
| Q2 — refusal or fallback | Prefer refusal requiring explicit `noise=None`; otherwise require the complete, visible resolution contract in R8. |
| Q3 — integrators | Treat RK for GNLSE and Multistep for modal/spatial routes as candidates only. Choose the table from matched-accuracy runtime evidence and define settings precedence (R9–R10, R13). |
| Q4 — accuracy targets | Use observable-specific budgets with one common convergence/confidence protocol. Fix the budgets, floors and case definitions before selecting tolerances (R9). |
| Q5 — norm warning | Print an optional size diagnostic; do not warn solely because $r>\mathrm{rtol}$ (R10). |
| Q6 — coupled source | Do not change a source's default based on future extraction. Preserve the refusal until the decomposition is transported; an interim input-noise source must be explicit (R7–R8). |
| Q7 — reservoir physics | Yes: a later proposal should derive the half-photon/Gaussian and reservoir terms, their ordering corrections and qualified regime. Do not call the current default a substitute for it (R1–R2). |
| Q8 — Gaussian amplitude | Defer for this defaults change unless the maintainer explicitly wants the broader stochastic-model change and its qualification in scope (R14). |

Further decisions raised by this review are the observable/reconstruction contract
(R3), the treatment of changing power metrics and coupled transport (R5–R7), the
physical transverse width of planar noise (R11), and the scope of TDSE qualification
(R12). Their recommendations are recorded beside each finding.

#### Follow-up checks — 2026-10-26

Further derivation and source checks refine R1–R3 and R14 without changing the
recommendation to revise before acceptance. The full nonlinear response preserves
the stated change of variables; the published corrected estimator recovers the
constant-gain mean but fails the gain-then-loss counterexample under persistent-seed
transport; and Gaussianity must be assessed in the measured mode. These are review
clarifications, not resolutions of the proposal's open questions.

#### Evidence and limits of this review

- Inspected the driver defaults, noise drawing and phase transport, nonlinear stage
  injection, error-norm scaling, output/processing accessors, launch handling,
  refusal checks, the conventions and SEP process. No implementation was changed.
- Ran four short NumPy-reference GNLSE simulations: input and modified noise, each
  with Raman off and on, using RK, ten uniform steps over $10^{-3}$ m, 128 temporal
  samples, seed 37, $\gamma=0.1$ W$^{-1}$m$^{-1}$, $\beta_2=-2\times10^{-26}$ s²/m,
  800 nm carrier, 600–1200 nm limits, a 200 fs requested window, 30 fs Gaussian pulse,
  $10^{-14}$ J energy, shock off and both absorber rates zero. The relative field
  defect $\|\chi-(\psi+N)\|/\|\chi\|$ over saved planes was
  $1.19\times10^{-16}$ without Raman and $1.35\times10^{-16}$ with it. Full-field
  energy drift was at most $6.7\times10^{-16}$; shifted-field drift was
  $1.75\times10^{-7}$ and $1.54\times10^{-7}$ respectively. These are counterexample
  probes, not a tolerance calibration, benchmark or cross-backend qualification.
- Checked the relevant equations in
  [Chen–Wise, arXiv:2410.20567v2](https://arxiv.org/html/2410.20567v2),
  [v5, including Appendix 2's readout](https://arxiv.org/html/2410.20567v5), and
  [Drummond–Corney, arXiv:quant-ph/9912095v2](https://arxiv.org/html/quant-ph/9912095v2),
  and the linearised MI discussion in
  [Brainis–Amans–Massar, arXiv:quant-ph/0412096](https://arxiv.org/html/quant-ph/0412096).
  The remaining cited papers were not independently checked in full. The draft's
  performance tables, production-practice tolerances, full supercontinuum ensembles,
  all-route identity and proposed refusal lifts were not reproduced or qualified.
- Independently derived the scalar gain/attenuation example in R1 and checked its
  arithmetic. Derived R14's projected fourth moment by pairing the independent
  phases; an exact finite phase quadrature reproduces $1$, $3/2$ and $7/4$ for
  one, two and four equally weighted bins with $n_0=1$. These are analytic checks,
  not qualification of a particular transient Raman mode or its output statistics.
