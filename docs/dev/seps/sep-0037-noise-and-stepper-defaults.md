# SEP 37 — Shot noise on by default, and an integrator and tolerances chosen for each interface

| | |
|---|---|
| **SEP** | 37 |
| **Title** | Shot noise on by default, and an integrator and tolerances chosen for each interface |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-26); Opus 5.5 (2026-10-26, revision after review) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-26 |
| **Resolution** | Accepted 2026-10-26 by John Travers, with the 10 % tie margin of §5.2, after his human review, the GPT-6 AI review and the response to it, with decisions D1–D7 recorded in Part III ([LupoLab/solprop#30](https://github.com/LupoLab/solprop/pull/30)). |
| **Requires** | [SEP 16](archive/sep-0016-unified-multistep.md) (the multistep family and its work–precision qualification); [SEP 30](archive/sep-0030-standard-benchmark-suite.md) (the benchmark suite); [SEP 31](archive/sep-0031-simulation-coupling.md) (field planes and coupled stages) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0037-noise-and-stepper-defaults` ([LupoLab/solprop#30](https://github.com/LupoLab/solprop/pull/30)): phases 0–3 implemented 2026-10-27; the phase 0 tables await the maintainer's approval |
| **Manual pages** | — |

## Part I — Proposal

*Revised on 2026-10-26 after the GPT-6 AI review. The response to the review (Part
III) says what changed for each finding; the maintainer's decisions are recorded
there as D1–D5.*

### 1. Abstract

Every solprop run is noiseless unless it asks for noise, every interface steps with
`Multistep()` by default, and the default tolerances differ between interfaces for
reasons that are only partly written down. This SEP changes those defaults, using models
and methods solprop already has. It is not a noise-physics project. It proposes:

- **`noise="input"` as the default**: one photon per mode with a random phase, added to
  the input field. This is the seed of the supercontinuum literature, and the one the
  author's group has used for years. The modified model stays available, with its
  limitations documented.
- **A default integrator for each `prop_*` interface**, chosen by a rule fixed in this
  SEP and never dependent on the physics a run switches on.
- **Default tolerances for each interface**, set by a stated convergence protocol.
- **Lifting the input-noise refusals that exist only because nobody built them**: the
  TDSE response, coupled flows and explicit interior launches.

Where noise cannot be carried, a run that did not ask for noise falls back to none. The
fallback is recorded and printed. Tests and worked examples fix their seeds.

Success is judged in four ways:
- the default resolves as specified at every entry point;
- each lifted refusal passes its stated checks;
- the chosen integrators win their comparison under the rule;
- the tolerances meet the protocol's budgets.

### 2. Motivation and scope

**A noiseless default hides itself.** A user who does not pass `noise=` gets no shot
noise. Nothing is printed, and the run record says so only on inspection. Spontaneous
processes then do not start: modulational instability, Raman scattering from noise and
the shot-to-shot incoherence of a supercontinuum. Coherent dynamics seeded by the pulse
itself look cleaner than they would in an experiment.

**The least bad model solprop knows how to implement is the input model.** It is what
the literature uses [1, 2] and what the author's group has run for years. Its known
limits are stated in §4.1. The modified model is not provably better (§4.2):
- **With a unitary linear operator, the two models are the same equation.** That holds
  whatever the nonlinear response is, Raman and plasma absorption included.
- **They differ only where the linear operator is not unitary.** That means propagation
  loss, which every default capillary run has, plus the absorbing collars and the
  normalisation along a varying guide. There the modified model's extra term is not
  reservoir noise, and it can be worse than the input model.
- **Its saved field leaves out the seed.** That changes what a spectrum or a coherence
  computed from the output means.

So the input model is the default, and the modified model stays an option.

**One default integrator does not suit every interface.** `Multistep()` became the
default everywhere with SEP 16 (ADR-045). The evidence was strongest where the source is
expensive: 5.5 to 5.8 times faster than IP-DOPRI5 on the full resonant-dispersive-wave
capillary, and 2 to 3 times on a soliton-to-continuum launch. The same record has
IP-DOPRI5 ahead on the single-mode GNLSE window on `numba_cpu`, by 1.2 to 1.4 times, and
the preliminary measurements of §3.4 find RK further ahead once noise is on. A default
that changed with the physics switched on would be fragile, and the author rejects it. A
default for each interface is not fragile in that way, because the user chooses the
interface and the interface fixes the equation solved.

**The default tolerances are only partly explained.**
- **`prop_gnlse`** asks for `rtol = 1e-8` and `atol = 1e-14` √W. These were set with the
  first driver, and no reason was recorded. They suit noiseless soliton recurrence,
  which is an occasional study; a user who wants it can ask for a tight tolerance. The
  GNLSE is used mostly for supercontinuum, where a looser tolerance is enough.
- **`prop_capillary` and `prop_waveguide`** ask for `1e-6` and `1e-10` √W, the values
  Luna.jl documents. The docstring justifies them by the cost of a modal step.
  Production practice is looser still: `rtol = 1e-5` is common, and runs with the TDSE
  response use `1e-3` to `3e-3`.
- **`prop_radial`, `prop_free` and `prop_planar`** have no default tolerance at all.
  Their `SimulationSettings` require a controller from the caller.

**In scope:**
- the defaults of every `prop_*` interface and of the `Simulation.from_*` constructors
  behind them: noise model, integrator and formulation, `rtol` and `atol`;
- the input-noise refusals that are not grounded in physics or numerics (§3.3);
- a floor-subtracted spectrum in processing, which gives an input run the clean
  spectrum of the modified model's readout (§5.5);
- seeds for the test suite and the worked examples;
- the manual's account of noise, cost and stepper choice.

**Out of scope:**
- **Changes to the modified model.** It is kept as it is, with no new work. Its own
  refusals (interpolated planes, field planes, coupled flows) stay, and so do its
  readout and its transport along a varying guide; they are recorded on the roadmap.
  Whether to keep or remove it is decided when [SEP 7](sep-0007-laser-gain-dynamics.md)
  settles whether gain media need its separate noise-injection path (D7).
- **New noise physics.** No half-photon vacuum, no Langevin terms for loss or Raman,
  no thermal phonon occupation, no Gaussian amplitude for the draw and no technical
  (pump) noise.
- **Noise on the hybrid planar grid**, which needs a transverse normalisation decision
  first, and **noise under single precision**, which binary32 cannot hold (§3.3).
- **Default changes elsewhere**, such as grids, save planes, absorbers or Raman.

### 3. Context

#### 3.1 The noise models

[Shot noise](../../guide/media/noise.md) describes the draw:
- one photon per active bin, mode and polarisation, with a deterministic modulus and a
  uniform random phase;
- masked to the active band on both grids;
- one generator stream per seed.

The same draw serves both models:
- **input** adds it to the field at the launch. It then propagates with the field under
  the full linear and nonlinear operators.
- **modified** (ADR-008) keeps the draw apart from the propagated field. It advances the
  draw by the unitary part of the cumulative linear operator,
  $N(z) = N_0\,e^{i\,\mathrm{Im}\,\Phi}$, and passes the full field $\psi + N$ to the
  nonlinear response at every stage. The saved field is $\psi$, without the draw.

The realisation and its seed are stored with the output. A `seed` of `None` draws from
the operating system and records what it drew. A seed fixes the realisation. It does not
make adaptive runs on different backends agree bit for bit (SEP 25).

#### 3.2 The defaults today

| Interface | Noise | Integrator | `rtol` | `atol` (√W) |
|---|---|---|---|---|
| `prop_gnlse` | `None` | `Multistep()` | 1e-8 | 1e-14 |
| `prop_capillary` | `None` | `Multistep()` | 1e-6 | 1e-10 |
| `prop_waveguide` | `None` | `Multistep()` | 1e-6 | 1e-10 |
| `prop_radial`, `prop_free` | `None` | `Multistep()` | caller's controller | caller's controller |
| `prop_planar` | not carried | `Multistep()` | caller's controller | caller's controller |

#### 3.3 Where noise is refused today, and what this SEP does

| Where | Refuses | The reason given today | This SEP, for the input default |
|---|---|---|---|
| `precision="single"` | both models | binary32 rounding swamps a one-photon seed (SEP 33) | an accuracy limit: kept; the default falls back to none |
| the TDSE response | both models | noise left out of the initial qualification (SEP 12) | a scope choice: lifted for `input` (§6) |
| an interior launch, `initial_z > 0` | both models | none recorded | lifted for an explicit `input`; the default takes no new draw (§5.1) |
| a coherent wall, the taper coupling, a split block, a moving solid guide | both models | the draw's transport through a nonunitary coupled flow is not measured | that question is the modified model's alone; lifted for `input` (§6) |
| the hybrid planar grid | both models | no photon measure is declared for slab amplitudes | needs a normalisation decision: kept; the default falls back |
| a general modal guide with a diagonal metric | both models | a one-photon draw is not defined in that basis; nothing builds one | kept; the default falls back |
| interpolated save planes | `modified` | the draw is added outside the interpolant (SEP 14) | not affected by the input default |
| `processing.field_plane` | `modified` | the noise phase is not reconstructed into a plane (SEP 31) | not affected by the input default |

Only the single-precision refusal rests on numerics. The planar and diagonal-metric
refusals rest on a missing definition. The rest are scope choices or concern the
modified model only.

#### 3.4 Preliminary measurements

These were taken while drafting, at commit `1d4f099`, on the native backend (what
`backend="auto"` resolved to), on an Apple M5 Max laptop, with seed 1. They count
accepted steps, and each wall time is a single sample. They are evidence about these
cases, not a qualification, and phase 0 repeats them under §5.2's rule.

**Capillary worked examples** cover SPM, soliton self-compression, four-wave mixing,
XPM, ionisation blue shift and two supercontinuum ensembles (16–256 µm bores, µJ–mJ
pulses). Each took the same number of steps, within 2 %, with and without noise.

**The N = 3 GNLSE soliton tutorial** (20 pJ, 100 fs, 500–2000 nm band, the tutorial's
grid) behaves very differently:

| Integrator, `rtol` | No noise | `input` | `modified` |
|---|---|---|---|
| `Multistep()`, 1e-9 | 1126 | 73199 | 73199 |
| `Multistep(formulation="interaction")`, 1e-9 | 5189 | 108759 | 108760 |
| `RungeKutta()`, 1e-9 | 1800 | 21845 | 21845 |
| `Multistep()`, 1e-6 | 467 | — | 18863 |
| `RungeKutta()`, 1e-6 | 507 | — | 4193 |

- **The two models cost the same here.** The manual's "about eight times" for the
  modified model alone was quoted from Luna.jl and is not reproduced in this case.
- **`atol` made no difference.** Values from `1e-14` to `1e-6` √W left the counts
  unchanged.
- **Scaling the draw changes the cost.** Scaling it by $s$ took the RK count from 21845
  at $s = 1$, through 8153 and 2219, to 1816 at $s = 10^{-6}$, against 1800 without
  noise.

**Accuracy** was measured against an RK reference at `rtol = 1e-11` with the same draw.
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

In this case the extra steps are needed: a schedule discovered without noise gets the
noise part wrong. A replayed schedule for a noisy ensemble should therefore be
calibrated with noise on, across more than the seed that discovered it.

### 4. Theory

#### 4.1 The input model and its known limits

**Where one photon per mode comes from.** In the symmetrically ordered (Wigner) picture,
which a classical field simulation implicitly uses, the vacuum carries half a photon
per mode [3]. One photon per mode is instead the equivalent input noise of a
phase-insensitive amplifier. Smith showed that the spontaneous Raman or Brillouin
emission along a fibre, amplified by the gain remaining after each point, is the same at
high gain as one photon per mode injected at the input and propagated by the stimulated
equations alone [4]. The supercontinuum literature adopted this seed with a random phase
on every spectral bin [1, 2].

**What it gets wrong, and by how much:**
- **Parametric processes.** For modulational instability and four-wave mixing, which
  have no reservoir, the exact seed is half a photon. One photon overestimates
  spontaneous parametric power by a factor of two at high gain [3, 6]. Modulational
  instability enters a supercontinuum logarithmically, so this shifts its development
  length by about $\ln 2/(2\gamma P_0)$, a few per cent of $L_\mathrm{MI}$.
- **Linear loss.** Under linear loss, the seed decays with the field instead of staying
  at the vacuum level. Spontaneous processes far into a lossy guide are then under-seeded
  by about the guide's power transmission up to that point.
- **The floor.** The seed sits as a one-photon floor under the whole spectrum.
- **Thermal seeding.** Spontaneous Raman scattering at a shift $\Omega$ is seeded at
  $n_\mathrm{th}(\Omega) + 1$ photons on the Stokes side and $n_\mathrm{th}$ on the
  anti-Stokes side, with $n_\mathrm{th} = 1/(e^{\hbar\Omega/k_BT} - 1)$ [3]. For vibrational
  shifts (tens of THz) $n_\mathrm{th}$ is negligible, but for the rotational lines of
  nitrogen at room temperature (0.36–5 THz) it is about 1 to 17. One photon per mode
  then under-seeds rotational Raman by up to an order of magnitude, and seeds its
  anti-Stokes side where thermal phonons would. Neither noise model in solprop includes
  this; a Raman Langevin term with the thermal factor would.
- **Statistics.** The fixed-modulus draw has the right second moments but not the
  fourth moments of a vacuum mode. A gain mode that spans many spectral bins sums many
  independent phases and approaches Gaussian statistics anyway.

These limits are known and bounded. Each is either small where the model is used or
visible to the user. Fixing them needs the half-photon vacuum with a Langevin term for
each reservoir [3], which is new physics outside this SEP.

#### 4.2 Why the modified model is not the default

Write the linear operator as $\hat D = \hat D_u + \hat D_r$, with $\hat D_u$ its unitary
part. Let the modified draw obey $\partial_z N = \hat D_u N$, and let the modified field
obey $\partial_z\psi = \hat D\psi + \hat N(\psi + N)$, for any deterministic nonlinear
operator $\hat N$. Then the reconstructed field $\chi = \psi + N$ obeys

$$ \partial_z \chi = \hat D\chi + \hat N(\chi) - \hat D_r N . $$

- **When $\hat D_r = 0$, $\chi$ is exactly the input model's field.** This holds to all
  orders in the seed, whatever $\hat N$ contains, including a delayed Raman response
  and plasma absorption. A short probe on `numpy_ref` reproduced this identity to
  $10^{-16}$, with and without Raman (Part III).
- **When $\hat D_r \neq 0$, the two differ by the forcing $-\hat D_r N$.** That covers
  propagation loss, the absorbing collars and the real normalisation term along a
  varying guide. The forcing is correlated with the original draw. It is not newly
  sampled reservoir noise, and it does not follow the fluctuation–dissipation form of a
  reservoir [3].
- **It can be the worse of the two.** Take a gain $G$ followed by a passive transmission
  $\eta$. The modified model's corrected readout gives $[1 + \sqrt\eta(\sqrt G - 1)]^2 - 1$
  spontaneous photons, against the true $\eta(G - 1)$. With $G = 4$ and $\eta = 1/4$,
  that is 1.25 photons against 0.75.

So the modified model equals the input model where the linear operator is unitary.
Elsewhere it is an approximation of unqualified accuracy. That is not "provably better
and not worse elsewhere", and it is not adopted as the default.

Its practical differences add to this. The saved field excludes the seed, and no
processing function restores it, so spectra and coherence computed from the output
describe $\psi$, not $\chi$. Several routes refuse it (§3.3). A draw held at constant
internal modulus also gives a restored energy per bin of $\hbar\omega\,w(z)/w(0)$ along
a varying guide, where $w$ is the power weight of the mode. All of these are documented
and left on the roadmap.

solprop's modified model is also not the published one [5]. It evaluates the full
nonlinear response of $\psi + N$ rather than the first-order expansion, it has no
separate emission gain (solprop has no gain media), and it transports the draw by the
dispersive phase rather than holding it fixed. The paper's results therefore qualify
the published model, not solprop's variant.

#### 4.3 What noise costs

The controller holds the embedded difference below $\mathrm{rtol}\,\lVert y\rVert + \mathrm{atol}$
on its internal state norm. A seed of one photon per bin over $n$ active bins carries an
energy of about $n\hbar\bar\omega$, against the pulse's $U$. Its initial share of the
norm is then about

$$ r \approx \sqrt{n\hbar\bar\omega / U} , $$

about $10^{-2}$ for the 20 pJ soliton and $10^{-4}$ to $10^{-5}$ for µJ–mJ capillary
pulses. This is a size diagnostic, not a step-cost criterion. Linear propagation of the
seed costs nothing under an exponential integrator at any $r$. A small seed can grow
into a resolved instability. And the capillary runs above showed no cost even with
$r$ above `rtol`. The cost of noise is measured, not predicted.

### 5. Proposed interface

#### 5.1 Noise

- **The default is `noise="input"`** on every interface and constructor that carries
  noise. `seed=None` keeps its meaning. `noise=None` switches noise off explicitly.
  `noise="modified"` is unchanged, with its limits documented (§4.2).
- **A route that cannot carry noise falls back.** These routes are single precision,
  the hybrid planar grid and a general guide with a diagonal metric. A run on one of
  them that omits `noise` resolves to no noise. The noise record stores the requested
  model (`default`), the resolved model (`None`) and the reason. The start-of-run
  summary prints the resolution, and `Output.metadata` carries it.
- **Fallback limits.** The list of fallback routes is closed. A backend choice never
  changes the resolution: a backend that cannot run a resolved model fails as today,
  and that failure is never reinterpreted as unsupported noise. A run that passes
  `seed=` without `noise=` on a fallback route is refused, because the seed states an
  intent the fallback would discard.
- **An interior launch** (`initial_z > 0`, an explicit spectrum) takes the spectrum as
  given under the default, with no new draw. This is recorded as a resolution, because
  the spectrum may already carry a realisation from an earlier run. An explicit
  `noise="input"` adds one draw at `initial_z`.
- **A coupled launch** from a field plane (`Simulation.with_input` with a `Launch`) keeps
  today's behaviour: the plane carries whatever the source run carried, and no second
  draw is added.

```text
run = prop_capillary(150e-6, 1.0, "He", 0.5, lambda0=800e-9, energy=200e-6, fwhm=10e-15)
run.metadata.noise          # model='input', seed=<recorded entropy>, requested='default'
run = prop_capillary(..., noise=None)          # noiseless
run = prop_capillary(..., seed=20261026)       # reproducible realisation
```

#### 5.2 Integrators

Each interface gets its own default, chosen in phase 0 by this rule:

- **Cases:** the standard suite's core cases for that interface (SEP 30), on short
  windows, each with noise on and off.
- **Candidates:** `Multistep()` in both formulations and `RungeKutta()`.
- **Selection:** take the candidate with the lowest median wall time at matched
  accuracy under §5.3's protocol. Measure on one CPU thread and at the production
  width, on the suite's declared machines.
- **Ties:** a candidate within 10 % of the fastest counts as tied, and a tie keeps
  today's default.

The resulting table, with its evidence, goes into the phase 0 report. It takes effect
only once the maintainer approves it there, which is this SEP's approval gate. The
expectation is `RungeKutta()` for `prop_gnlse` and `Multistep()` for the modal and
spatial interfaces. The default never depends on the responses switched on, and a user
who wants another method passes `integrator=` as today.

#### 5.3 Tolerances

Each interface's default `rtol` is the loosest of three levels, a factor of ten apart,
that meets the budgets below on that interface's canonical case:

- **Cases:**
  - `prop_gnlse`: the standard suite's `gnlse_raman_shock`, which is the
    supercontinuum of Dudley, Genty and Coen [1]: 15 cm, 10 kW, sech, $\beta_2$ to
    $\beta_{10}$, Raman and shock.
  - `prop_capillary` and `prop_waveguide`: one capillary case and one waveguide case of
    the suite.
  - `prop_radial`, `prop_free` and `prop_planar`: one case each from the suite.
- **Protocol:**
  - one set of seeds shared across every tolerance level (paired seeds);
  - a reference for each seed at a tolerance a hundred times tighter than the
    tightest level;
  - comparisons on stored witnesses or short windows, so that no full-length run is
    needed in tests.
- **Budgets:**
  - the ensemble-mean spectral energy density, in decibels, within 0.5 dB of the
    reference wherever the reference is above −40 dB of its peak;
  - the coherence $|g_{12}|$ within 0.02, absolute, over the same region;
  - for a noiseless case, the same spectral budget on the single run.

A noiseless soliton-recurrence study asks for its own tighter value, and the manual says
so. A response does not set a default: the TDSE practice of `1e-3` to `3e-3` is
recorded in the manual as a recommendation. `atol` is set with `rtol` on the same cases.

For `prop_radial`, `prop_free` and `prop_planar`, `SimulationSettings` gains a default
for its controller. An explicit controller always wins, and an omitted one takes the
interface's default. The default records which applied.

#### 5.4 What the user sees

The start-of-run summary, delivered separately, prints the resolved noise model and
seed, any fallback and its reason, the integrator and the tolerances. It is a
prerequisite of phase 2: the noise default does not land before the summary does. The
seed share $r$ of §4.3 may be printed as a size diagnostic, but nothing warns on it.

#### 5.5 A floor-subtracted spectrum

`processing.get_spectral_density` and `processing.get_spectrum` gain a keyword
`noise_floor`, either `"keep"` (the default) or `"subtract"`. `"subtract"` removes the
injected seed's energy, exactly one photon per active bin and channel, from the spectral
energy density, using the route's own photon measure, the one `noise.photons_per_bin`
inverts.

- **The equivalence.** The draw has a fixed modulus, so this is exact for one
  realisation. Where the linear operator only changes the phase, the subtracted spectrum
  of an input run equals the modified model's corrected readout $|\psi + N|^2 - |N|^2$
  for the same draw.
- **What it is not.** It is not a photon-number estimator under loss. There the input
  seed has decayed, and the subtraction removes more than remains.
- **Signs and refusals.** Values near the floor can be negative, by the interference
  term and the seed's decay. The keyword returns them signed, and the plotting helpers
  mask non-positive values on a logarithmic axis. It refuses a run without noise, and a
  modified run, whose saved field holds no seed.

### 6. Implementation strategies

**The default and its resolution** are driver settings, in `solprop.driver.interface`
and `solprop.driver.simulation`. A sentinel default resolves per route, before
translation, by the rules of §5.1. The noise record gains the requested model and the
reason for any fallback, so the output schema version changes. Input noise is part of
the initial state, so the kernel contract is unchanged, and so is every backend: no
backend has an input-noise path of its own. The `numba_cpu` stash of the plasma
statistics is disabled only under the modified model, so the default does not touch it.

**Lifting the input-noise refusals:**

| Refusal | Work | Checks |
|---|---|---|
| the TDSE response | admit `input`, which only changes the drive field the atom sees | the drive at the entrance equals the pulse plus the draw, and the realisation is stored; the draw alone, with no pulse, gives the TDSE's weak-field (linear-susceptibility) response within its documented weak-field accuracy; the active band lies inside the response's declared window, as it must for the pulse alone |
| coupled flows | admit `input`: the draw enters as part of the entrance state, in the channel basis, and the flow propagates it like any field | the entrance basis is orthonormal in true power, else the run is refused, as `check_noise_basis` does for general guides; one photon per channel and bin at the entrance; exit energy equals the noiseless run's plus the transmitted seed, within the flow's certificate |
| explicit interior launch | add one draw at `initial_z` | photon number per bin at the launch is one; the default case takes no draw (§5.1) |

The modified model keeps all its refusals.

### 7. Verification and validation

- **Resolution.** At every entry point, test four cases: `noise` omitted, `noise=None`,
  each explicit model, and `seed=` without `noise=`. Cover the routes that carry noise,
  the fallback routes and the interior launch, and check the stored record and the
  printed summary.
- **Lifted refusals**, as in the table of §6.
- **The floor-subtracted spectrum.** It equals the modified readout of the same draw to
  rounding, in a lossless Kerr and Raman case. It removes exactly one photon per bin from
  an unpropagated noisy input, leaving the pulse's spectrum. It refuses as §5.5 states.
- **Integrators and tolerances**, by the rules of §5.2 and §5.3.
- **Examples.** Re-run every worked example whose result changes under the new
  defaults, with a fixed seed.
- **The input model itself** is unchanged. Its existing tests (one photon per bin, phase
  statistics) stand.

### 8. Documentation

- [Shot noise](../../guide/media/noise.md):
  - the default and the fallback rules;
  - the input model's origin [4] and known limits (§4.1);
  - an honest account of the modified model: its origin [5], solprop's variant, where
    it equals the input model and where it differs, its readout and its refusals
    (§4.2);
  - the cost as measured in solprop, in place of the quoted factor of eight;
  - the floor-subtracted spectrum and what it is not (§5.5);
  - what neither model does.
- [Choosing a stepper](../../guide/numerics/choosing-a-stepper.md): the default for
  each interface and the evidence behind it.
- [Fixed schedules](../../guide/numerics/fixed-schedules.md): calibrate an ensemble's
  schedule with noise on, across seeds.
- The docstrings of every `prop_*` interface and `Simulation.from_*` constructor: noise,
  integrator and tolerances, each with its reason, and the recommendations of §5.3.
- The worked examples and tutorials: a fixed `seed` where noise is now on, and
  `noise=None` only where the lesson is noiseless.

### 9. Compatibility and impact

- **Every default run becomes stochastic.** Two runs of the same script differ by
  realisation unless they fix `seed`. Users who compared outputs must pass a seed, or
  `noise=None`.
- **Existing results** change at the level of the seed, and more visibly where noise
  seeds a spontaneous process. Low-energy GNLSE runs become slower (§3.4).
- **Tests.** About 65 test modules build runs without naming `noise`. Those comparing
  against a deterministic reference pass `noise=None`, and a few tests cover the
  default itself.
- **The output schema version changes** for the extended noise record. Older files read
  as before, with the requested model taken as the resolved one.
- **Luna.jl** defaults to its modified model. solprop's default is the input model, and
  the differences page records why (§4.2).
- **Nothing is public yet**, so no deprecation path is needed. The changelog records the
  change.

### 10. Alternatives considered

- **Keep `None` and document it.** This is the status quo, and it is how the gap went
  unnoticed. Rejected, because noise is part of the physics being modelled.
- **`modified` as the default.** It equals the input model where the linear operator is
  unitary, and it is of unqualified accuracy elsewhere (§4.2). It also changes what the
  saved field means, and it refuses more routes. Rejected for the default; it stays an
  option.
- **Refuse, rather than fall back, where noise cannot be carried.** This is the
  reviewer's preference, and it is honest, but every default single-precision run would
  then need `noise=None`. Rejected in favour of a closed, recorded and printed fallback
  (D2).
- **Removing the modified model now.** It adds little physics (§4.2), and it carries
  refusals and a different saved field. But its separate injection path is what
  amplified spontaneous emission in a gain medium would need, with the noise seeing only
  the emission gain. The decision waits for SEP 7 (D7).
- **A half-photon vacuum with reservoir Langevin terms**, or a Gaussian draw. This is
  the scheme that is exact for both kinds of seeding [3], but it is new physics, with a
  qualification of its own. Left for a later proposal.
- **An integrator that depends on the physics switched on.** Rejected by the author as
  fragile.

### 11. Phases

The phases are made on one branch, `feature/sep-0037-noise-and-stepper-defaults`, and
land in one implementation pull request, as SEP 0 requires. Each phase is qualified and
reported on its own.

0. **Exploration and measurement.** Record what each default and refusal was based on.
   Run §5.2's rule and §5.3's protocol on the stated cases, at today's defaults and at
   the candidates. Report the integrator and tolerance tables for the maintainer's
   approval.
1. **Integrators and tolerances.** Set the approved defaults and their docstrings, add
   the settings default and its precedence, and re-run the affected examples.
2. **The noise default.** Add the sentinel, the resolution rules of §5.1, the extended
   noise record, the docstrings and the noise guide. Seed the tests and examples, and
   pin the noiseless references. This phase follows the start-of-run summary.
3. **Input noise on the TDSE, coupled-flow and interior-launch routes**, with the checks
   of §6, and the floor-subtracted spectrum of §5.5.

### 12. Open questions

The review's questions are decided (D1–D5, Part III). Two remain, each with its
destination:

- **Q1.** Is the channel basis at the entrance of each coupled-flow route orthonormal in
  true power, so that the input draw is one photon per channel there? Phase 3; a route
  where it is not stays refused.
- **Q2.** Does any standard-suite case for `prop_planar` exist on which to set its
  tolerance? If not, phase 0 adds one or leaves `prop_planar` without a default.
  Phase 0.

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

## Part II — Implementation record

### Phase 0 report — measurement

- **Scope:** phase 0, 2026-10-26/27, Opus 5.5 (AI) for John Travers. The source was
  commit 71a671d, measured on an AMD EPYC 7763 (Zen 3) cluster node with the native
  backend, LOCAL profile, GCC 16.2, FFTW 3.3.11 and OpenBLAS 0.3.34. Each arm took three
  timing samples at 1 and at 16 threads, with node-local environments. It cost about 3.7 h
  of wall time and 410 CPU-hours. The harness, the raw records and the full report are
  kept with the project's development records.
- **Cases:** the maintainer-approved tracking set:
  - `gnlse_raman_shock`, `gnlse_soliton`;
  - `rdw`, `he_rdw_averaged`, `argon_blueshift` (with its averaged and envelope
    variants), `h2_raman_6ps_averaged`, `ctc`;
  - `radial_filament_256`, `grin_nonlinear` (a 0.02 m window), `kerr_beam_192`,
    `planar`.
- **Integrator rule (§5.2):** the geometric mean over cases and noise arms of each
  candidate's time over the fastest, with a 10 % tie margin.

  | Interface | ms (exponential) | msi (interaction) | rk | Rule's choice |
  |---|---|---|---|---|
  | `prop_gnlse` (1 / 16 threads) | 2.48 / 4.84 | 2.01 / 3.33 | 1.00 / 1.00 | `RungeKutta()` |
  | `prop_capillary` | 2.15 / 2.26 | 1.45 / 1.47 | 1.15 / 1.09 | split (see phase 1) |
  | `prop_waveguide` | 4.7–7.4 | 1.00 | 1.26–1.29 | msi, inadmissible (see phase 1) |
  | `prop_radial` | 1.00 | tie / 1.3 | — / 2.4 | `Multistep()` |
  | `prop_planar` | tie | 1.00 / tie | — | `Multistep()` |

  On the noisy `gnlse_raman_shock` case, RK at 1e-5 took 32 s against Multistep's 141 s
  at 1e-6, at 1 thread. On the capillary, RK wins 8 of 14 rows at 1 thread and 11 of 14
  at 16, but Multistep wins `rdw` (RK is 1.6–2.0× slower) and the noiseless `ctc`.
- **Tolerance protocol (§5.3):**

  | Interface | Case | Passing levels | Proposed |
  |---|---|---|---|
  | `prop_gnlse` | `gnlse_raman_shock`, RK, 16 seeds | 1e-7, 1e-8, 1e-9 (and 1e-6) | 1e-7 |
  | `prop_capillary` | `rdw`, 16 seeds; noisy `ctc` check | RK 1e-5; ms needs 1e-6 (Δ\|g₁₂\| 0.043 at 1e-5) | 1e-6 with ms |
  | `prop_waveguide` | `grin_nonlinear`, 4 seeds | 1e-5 | 1e-5 |
  | `prop_radial` | `radial_filament_256`, 16 seeds | 1e-5 | 1e-5, atol 0 |
  | `prop_planar` | `planar`, noiseless | 1e-5 | 1e-5, atol 0 |

  `atol` was never binding: atol = 0 gave identical steps and errors on the GNLSE, `rdw`,
  `ctc` and `grin_nonlinear`. Q2 is answered: `planar` serves.
- **Not measured:** `kerr_beam_384` and noisy free space (`kerr_beam_192`'s steps do not
  depend on `rtol` over its window, so free space gets no default); `grin_nonlinear`
  beyond its window; the modified model; any machine but this one.
- **Anomalies:** on most cases every candidate passes at the loosest level, 1e-4, so the
  comparison there is not accuracy-limited. Noise changes the capillary step count only
  on `ctc` (1.4–1.6×).

### Phase 1 report — integrators and tolerances

- **Scope:** phase 1, 2026-10-27, Opus 5.5 (AI) for John Travers. The defaults below
  are **provisional, pending the maintainer's approval** of the phase 0 tables (§5.2's
  approval gate).
- **Delivered:** `src/solprop/driver/simulation.py`:
  - `_interface_integrator` gives `RungeKutta()` to `from_gnlse` and `from_waveguide`;
  - `DEFAULT_RTOL` is 1e-7, the new `CAPILLARY_RTOL` 1e-6, `WAVEGUIDE_RTOL` 1e-5,
    and `SPATIAL_RTOL` 1e-5 with `SPATIAL_ATOL` 0;
  - `SimulationSettings.controller` is optional, its other fields keyword-only, and
    `_spatial_settings` fills the radial and planar default. Free space refuses a missing
    controller, and a direct `Simulation` needs one.

  The docstrings, the stepper guide's table, ADR-055 and the changelog follow.
- **Deviations:**
  - **Admissibility.** The rule now chooses among the methods every route of the
    interface admits. The interaction-formulation Adams method won the waveguide case but
    refuses the coupled flows of the split block and the moving guide, so the waveguide
    default is RK, which is 4× faster than today's default there.
  - **The capillary is held at `Multistep()` and 1e-6.** The rule's aggregate prefers RK
    with 1e-5, but the evidence is split: Multistep wins `rdw` and the noiseless `ctc`,
    the production workloads SEP 16 qualified it on. RK would also refuse default TDSE
    runs and move most of the capillary test references. The maintainer decides this.
- **Verification:** the default lane and the slow lane on the cluster node (below, at
  Final), and the defaults checked by construction.

### Phase 2 report — the input-noise default

- **Scope:** phase 2, 2026-10-27, Opus 5.5 (AI) for John Travers, branch
  `feature/sep-0037-noise-and-stepper-defaults`.
- **Delivered:**
  - `src/solprop/driver/simulation.py`: `NoiseSetting` and the `"default"` keyword
    default on every `Simulation.from_*` constructor; `_resolve_noise`, called where
    each route draws; `Simulation.noise_resolution`, carried through `with_input` and
    derived runs; the planar route's resolution; and `SINGLE_PRECISION_NOISE`,
    `PLANAR_NOISE` and `INTERIOR_LAUNCH_NOISE`.
  - `src/solprop/driver/interface.py`: the same default on every `prop_*` interface.
  - `src/solprop/output/schema.py`: `NoiseResolution`, and
    `RunMetadata.noise_resolution` at output schema 26, stored beside the hash.
  - `src/solprop/driver/summary.py`: the noise row marks a default model and gives a
    fallback's reason.
  - `src/solprop/backends/driver.py`: the input draw is added at the run's launch
    position rather than at z = 0.
  - Tests: `tests/driver/test_noise_default.py`.
  - Manual: `docs/guide/media/noise.md`, `interior-launch.md`, `coupling.md`, `luna.md`.
  - Tutorials and worked examples now pass a fixed seed or `noise=None`.
- **Deviations from the proposal:**
  - The general guide's diagonal metric is not a fallback route. Nothing builds such a
    metric, and its translation-time refusal stands.
  - A seed beside `noise=None` is now refused on every route, as the spatial routes
    already did.
- **Decisions:**
  - The resolution is a dataclass, not a free mapping.
  - It stays out of the configuration hash, because the resolved model is already in
    the hashed noise record.
  - The direct `Simulation(...)` constructor, which takes a realisation rather than a
    model, keeps `None` as its default.
- **Verification:**
  - The full default lane on an AMD EPYC 7763 cluster node (64 workers): 14851 passed.
    The one failure is a numba licence cache-key test that passes on an Apple M5 Max
    laptop and fails only in
    that environment.
  - The source doctests pass.
  - The test suite keeps its noiseless references explicit: about 380 driver calls in
    `tests/` now say `noise=None`.
- **Limitations:** the native backend is unchanged here (input noise enters through the
  shared `set_initial`), and it was not built for these runs.

### Phase 3 report — input noise on more routes, and the floor

- **Scope:** phase 3, 2026-10-27, Opus 5.5 (AI) for John Travers; merged into the
  implementation branch.
- **Delivered:**
  - Input noise beside the coherent wall, the taper coupling, the split block and the
    moving solid guide. The refusals in `translate/problem.py` and the shared backend
    admission list now name only the modified model.
  - Input noise beside the TDSE response (`translate/tdse_source.py`, the seam check
    and the driver).
  - An explicit input draw at an interior launch (phase 2's `set_initial` rule).
  - `noise_floor` on `processing.get_spectral_density` and the plotting helpers.
  - Tests: `tests/driver/test_coupled_input_noise.py`,
    `tests/driver/test_tdse_input_noise.py` and `tests/processing/test_noise_floor.py`.
- **Q1, answered:** every coupled route's entrance basis is orthonormal in true power:
  - the capillary HE₁ₘ (and TE/TM) sets, by Bessel orthogonality and per-mode power
    normalisation;
  - the split block's dense metric, a declared unitary rotation of a
    Poynting-normalised reference, which keeps the total per bin;
  - the moving guide's identity metric, whose Gram departure is gated along the path.

  No route needed a named refusal.
- **Deviations:**
  - The unpropagated-floor identity keeps its interference term $2\,\mathrm{Re}(\psi^*N)$,
    which for one draw is as large as the seed.
  - `get_spectrum` refuses `"subtract"`, since no amplitude has the signed density as its
    modulus squared.
- **Verification:**
  - At the entrance, one photon per channel and bin to 3e-13 on all four coupled routes.
  - The exit-energy identity to between 1.1e-6 and 5.8e-5 of the seed's exit energy (the
    Kerr cross-phase on the seed).
  - `numba_cpu` matches `numpy_ref` to 3e-15.
  - TDSE: the drive equals the pulse plus the draw to 2e-16, and the draw alone is 3.5e-11
    of the intensity at which the weak-field response is certified.
  - The floor-subtracted density equals the modified readout to 1.2e-15.
- **Limitations:**
  - A full TDSE weak-field run through the real atomic engine was not made: it needs the
    pinned Pyroraptor.
  - The native backend admits input noise beside a coupled flow by the shared list but
    was not run.
  - The coupled routes' modal identity records still write no noise reference basis.


## Part III — History

- 2026-10-26: Drafted, with the preliminary measurements of §3.4.
- 2026-10-26: Revised after the author's comments. §4 now rests on the modified model's
  origin and its exact equivalence to the input model without a reservoir, with a
  quantum reference for spontaneous modulational instability. The tolerance targets
  follow production practice. Q1 and Q8 are new, and the factor of two in parametric
  seeding is recorded as deferred physics.

- 2026-10-26: Human review by John Travers and AI review by GPT-6, recorded below.
- 2026-10-26: Revised after the review and the maintainer's decisions D1–D5. The default
  is now `input`, with the modified model kept as an option. The refusal lifts cover
  input noise only, the fallback contract is complete, and the integrator rule and
  tolerance protocol are fixed. The response to every finding is at the end of Part III.
- 2026-10-26: Added the floor-subtracted spectrum (§5.5, D6).
- 2026-10-26: Recorded that the modified model is kept, with no new work, until SEP 7
  settles whether gain media need its injection path (D7).
- 2026-10-26: Accepted by John Travers, with the 10 % tie margin of §5.2. Part I is
  frozen from this revision.
- 2026-10-27: Phases 0 to 3 implemented on one branch (Part II). The integrator and
  tolerance tables are applied provisionally, pending the maintainer's approval; the
  capillary's integrator is held at `Multistep()` for his decision.

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

### Maintainer decisions — 2026-10-26

John Travers, on the review. His guidance: no research project; the least bad noise model
solprop knows how to implement; `input` has been used for years, so it, or anything
provably better and not worse elsewhere, is an improvement.

- **D1 (R1–R3, R14; old Q1, Q7, Q8).** The default is `noise="input"`. The modified model
  is not provably better, so it stays an option, with its limits documented. Its
  readout, its transport along a varying guide, its coupled-flow transport, a
  half-photon vacuum and a Gaussian draw are all outside this SEP.
- **D2 (R8; old Q2, Q6).** Where noise cannot be carried, a run that omits `noise` falls
  back to none, recorded and printed. The maintainer also asked why the TDSE should
  refuse input noise at all, since it is only part of the input field. It need not:
  that refusal is a scope choice of SEP 12 (D5).
- **D3 (R9; old Q4).** The lean convergence protocol of §5.3: paired seeds, a tight
  reference for each seed, three levels, 0.5 dB on the mean spectrum above −40 dB, 0.02
  absolute on $|g_{12}|$, and the canonical cases named there. The maintainer approves
  the resulting table in the phase 0 report.
- **D4 (R11).** Planar noise stays refused. The default falls back on the planar grid,
  and the roadmap item stays open.
- **D5 (R12).** Input noise is admitted beside the TDSE response. The refusal was a
  qualification-scope choice: the atom is driven only inside its gated, edge-free
  window (SEP 26), and the seed's band is the grid's, which the propagated pulse fills
  anyway. The checks are those of §6.
- **D6.** The maintainer asked for a floor-subtracted spectrum, which keeps the one
  benefit of the modified model's readout that costs nothing (§5.5).
- **D7.** The modified model is kept as it is, with no new work. Whether to keep or
  remove it is decided when SEP 7 settles whether gain media need its separate
  noise-injection path: amplified spontaneous emission needs the noise to see only the
  emission gain, as in Chen and Wise's split, or a gain Langevin term (§2, §10).

### Response to review — 2026-10-26

Drafting agent: Opus 5.5. Part I is revised throughout.

- **R1 — accepted.** The reservoir claim was wrong. §4.2 now states the condition
  exactly: the two models agree when the draw is transported by the full homogeneous
  linear operator, whatever the nonlinear response is. They differ by $-\hat D_r N$
  where the linear operator is non-unitary. That forcing is correlated with the
  original draw, not reservoir noise, and the review's gain-then-loss counterexample is
  quoted. This finding, with D1, is why the default is now `input`.
- **R2 — accepted.** §4.2 lists how solprop's variant differs from the published model.
  §4.1 limits the half-photon statement to the parametric (linearised) case, and the
  reservoir model is left as separate work.
- **R3 — accepted, and largely dissolved by D1.** The input model saves the full field,
  so every existing observable keeps its meaning. §4.2 records that the modified
  model's saved field excludes the seed and that processing does not restore it.
  Reconstruction and the Bogoliubov comparison are no longer part of this SEP.
- **R4 — accepted.** The exact identity is no longer a qualification contract. §3.4's
  step counts are labelled as evidence about their cases, and the modified model is
  unchanged by this SEP.
- **R5 — accepted.** The restored energy $\hbar\omega\,w(z)/w(0)$ along a varying guide
  is documented in §4.2 as a limit of the modified model. The input draw is part of the
  state, so the full operator, including the normalisation term, carries it.
- **R6 — accepted.** The modified model keeps its coupled-flow refusal. Under the input
  model there is no transport question: the draw is part of the entrance state, and the
  flow propagates it like any field. §6 lifts the refusal for `input` only, behind an
  entrance-basis check (Q1).
- **R7 — accepted.** §5.1 defines the launches. A coupled launch keeps today's
  behaviour, with no second draw. An interior launch under the default takes its
  spectrum as given, and an explicit `input` adds one draw. The schema change is now
  stated (§6, §9).
- **R8 — fallback kept (D2), with the full contract.** The list of fallback routes is
  closed; backends never change the resolution; requested and resolved models and the
  reason are stored; the summary prints them and is a prerequisite of phase 2; and
  `seed=` without `noise=` is refused on fallback routes. The four-case test matrix is
  in §7.
- **R9 — accepted, in the lean form of D3.** §5.3 names the cases, the paired seeds,
  the reference, the levels, the signal floor and absolute coherence. Witnesses and
  short windows are used for tests. Monte Carlo error is controlled by the pairing
  rather than a separate confidence rule, a simplification taken on purpose.
- **R10 — accepted.** $r$ is labelled a size diagnostic (§4.3), with no warning. §3.4
  now gives its provenance: the commit, backend, seed, grids and single wall samples.
- **R11 — accepted; planar stays refused (D4).**
- **R12 — accepted in part (D5).** PPT is no longer an oracle. The TDSE check is now the
  weak-field response to the draw alone, plus drive equality and the band condition
  (§6).
- **R13 — accepted.** §5.2 freezes a selection rule, with a tie margin of 10 % (a
  drafting choice for the maintainer's approval) and an approval gate in the phase 0
  report. §5.3 adds the settings default and its precedence. §11 puts every phase on one
  branch and one implementation pull request.
- **R14 — accepted.** The draw's distribution is unchanged (D1), and its limit is stated
  in §4.1.
- **Reviewer's recommendations on Q1–Q8.** Followed for Q1 (no qualification contract),
  Q3 (candidates only, chosen by rule), Q4 (one protocol), Q5 (diagnostic only), Q6 (no
  default changed by future extraction), Q7 (a later proposal) and Q8 (deferred). Not
  followed for Q2, by D2.
