# Choosing a stepper

solprop has three integrators, chosen with `integrator=`; this page is their overview.
`Multistep()` is an exponential Adams method: one source evaluation per
accepted step, a history of the last few, and an order it chooses between one and five
([the Adams integrator](adams.md)). `RungeKutta()` is the interaction-picture
Dormand–Prince 5(4) pair: six evaluations with FSAL reuse, seven otherwise,
and no multistep history
([the Runge–Kutta integrator](solver.md#the-rungekutta-integrator)). Both integrate the
represented diagonal linear flow exponentially and share the error norm and
controller. Coupled flows have additional restrictions and residual errors;
check [the limitations](adams.md#limitations) for the chosen formulation.
`LowStorageRungeKutta()` is a third, for fixed-step runs whose state is too large for
either: classical RK4 in the interaction picture, replaying a prescribed schedule in
three copies of the state, with no error estimate ([below](#the-low-storage-march)).
This page sets out what decides between them, what was measured, and how to recognise
a problem that wants another one.

## The defaults

Each interface has its own default integrator and tolerances, chosen by
[SEP 37](../../dev/seps/archive/sep-0037-noise-and-stepper-defaults.md)'s rule: the fastest
method at matched accuracy on the benchmark suite's cases for that interface, with noise
on and off, among the methods every route of the interface admits. A default never
depends on the physics a run switches on; pass `integrator=`, `rtol=` or `atol=` to
choose another.

| Interface | Integrator | `rtol` | `atol` | Shot noise |
|---|---|---|---|---|
| `prop_gnlse` | `RungeKutta()` | 1e-7 | 1e-14 √W | `"input"` |
| `prop_capillary` | `Multistep()` | 1e-6 | 1e-10 √W | `"input"` |
| `prop_capillary` with the TDSE response | `Multistep()` | 1e-6 | 1e-10 √W | none |
| `prop_waveguide` | `RungeKutta()` | 1e-5 | 1e-10 √W | `"input"`; none for slab modes |
| `prop_radial` | `Multistep()` | 1e-5 | 0 | `"input"` |
| `prop_free` (Cartesian, line) | `RungeKutta()` | 1e-4 | 0 | none |
| `prop_planar` | `Multistep()` | 1e-5 | 0 | none (not carried) |

Under `precision="single"` every interface takes `RungeKutta()` and no noise, and its
`rtol` is the larger of the table's and `1e-6`, at most `1e-5`: `1e-6` on `prop_gnlse`
and `prop_capillary`, `1e-5` on the others, free space included
([single precision](../performance/single-precision.md)). The
spatial interfaces' tolerances apply where their `SimulationSettings` give no
controller; a free-space run that names no integrator takes `RungeKutta()`. A default
noise that resolves to none says why in the run record and the start-of-run summary,
and asking for noise beside the TDSE response is warned on: TDSE and Cartesian runs
are slow and rarely about noise sensitivity.

On the GNLSE, the Runge–Kutta march is 2.5 to 4.8 times faster than the exponential
Adams method at matched accuracy, more with noise. On the waveguide route it is 4 times
faster; the interaction-formulation Adams method is faster still there but refuses the
coupled flows some waveguide routes carry. On the capillary the Adams method is the
default for every representation, by the maintainer's decision against the rule
([ADR-055](../../dev/decisions.md)): it wins the runs the capillary is mostly used for,
and loses the single-mode runs with plasma or Raman. With several resolved modes the
source beats at the modes' phase mismatches: the interaction-picture march has to
resolve that beat within its steps (going from one resolved mode to four multiplies its
steps by 3.6 on the RDW capillary), while the exponential Adams method carries it in its
propagator and is 2.3 to 2.9 times faster at equal `rtol`. With one mode, or a
mode-averaged one, the Adams method at 1e-6 was 1.3 to 2.1 times faster than the
Runge–Kutta march at 1e-5 on the Kerr-only worked examples, and more accurate (on
self-phase modulation in argon, a field error of 4e-5 against 3e-4). It spends one
response evaluation per step where the Runge–Kutta march spends six, so an expensive
response widens the gap. The dispersive Kerr policies
(`MaterialSelection(electronic="symmetric-kerr")` or `"gas-polynomial"`) cost more per
evaluation than the carrier Kerr default: on a single-mode argon four-wave-mixing scan
with them, the Adams method was 3.6 times faster. On one mode or a mode-averaged one
with Raman or plasma, `RungeKutta()` is the faster choice: 2 to 2.9 times at 1e-5 on the
Raman self-frequency-shift and ionisation blueshift examples, and up to six times on
the benchmark suite's mode-averaged argon and hydrogen cases, at a field error still
far inside the spectral budget below. There the Adams method needs about as many source
calls as the Runge–Kutta march, and on a cheap source each of its steps costs up to four
times a Runge–Kutta stage. The TDSE response runs only on `Multistep()`. The radial and
planar defaults hold their methods, which are faster there than the Runge–Kutta march
(by 1.4 to 2.4 times on the radial filament), and set a tolerance where the caller gave
no controller. A free-space run takes the Runge–Kutta march: the Adams history's
memory rules the multistep method out for a large transverse state. Its tolerance is
looser because the spectrum never set it: on a Cartesian Kerr beam near self-focusing,
every level from 1e-3 to 1e-6 kept the spectrum within 0.002 dB of a tight reference,
and 1e-4 was taken for a field error 2.5 times smaller than at 1e-3. The adaptive march
needs no step calibration, and on that beam it was 26 to 53 times faster than a
0.5 µm fixed RK4IP schedule. A fixed `LowStorageRungeKutta()` step calibrated to the
same error is cheaper still, by about 1.6 times, and uses about half the memory.

The tolerances were set by a convergence test: the loosest of three levels a decade
apart at which a seeded ensemble's mean spectrum stays within 0.5 dB, above −40 dB of
its peak, and its coherence within 0.02 of a reference a hundred times tighter. A
noiseless soliton recurrence wants a tighter `rtol`; a TDSE run is usually run at
`1e-3` to `3e-3`.

## The short answer

Beyond the defaults above, prefer `Multistep()` wherever a source evaluation is the
expensive part of a step: several modes, ionisation and plasma,
Raman, a spatial state (radial, Cartesian, line or planar), a gas gradient or a taper,
a time-dependent Schrödinger equation. There it takes two to fourteen times fewer calls
than the Runge–Kutta march and, measured, up to six times less time.

Choose `integrator=RungeKutta()` when

- **the source is cheap and the state small** — a single-mode envelope (GNLSE)
  supercontinuum, where it is `prop_gnlse`'s default: the Adams method still takes
  fewer calls, but a call costs less than a Runge–Kutta step's own linear work, and
  with shot noise the Adams history resolves the seed's dynamics in many more steps;
- **the state is too large for the Adams history** ([large problems](#large-problems-memory));
- **an explicitly integrated damping term binds the step**: a strong in-stage collar the
  field reaches, or a replayed schedule whose steps are long against it — the Adams
  method's negative-real stability interval is 0.09 to 1 by order, DOPRI5's 3.3
  ([stability](adams.md#stability-and-the-temporal-collar));
- **a handful of fixed steps is all a run takes**: the Adams history starts at order
  one at the start of a schedule (a resume continues it on the CPU backends, from the
  snapshot the output keeps);
- **the backend is not `numpy_ref`, `numba_cpu` or `native`**.

Choose `integrator=LowStorageRungeKutta()` with a `schedule=` when the state is too
large for `RungeKutta()`'s stage rows and the run is a fixed-step one whose accuracy you
calibrate on the observable that matters ([the low-storage march](#the-low-storage-march)).

Within `Multistep()`, the interaction formulation (`formulation="interaction"`) is the
alternative where the source is linear in each weak bin's own field, as beside the
`modified` noise ([which formulation suits which source](adams.md#the-exponential-formulation)).

**Dense gases on field grids.** On a field grid (or an envelope grid with the third
harmonic) in a dense gas the carrier phase turns thousands of radians per metre, and
the default `Multistep()` spends steps following it; at a loose tolerance it can also
accumulate a large error over many honestly estimated steps (0.97 on an argon capillary
at `rtol = 1e-3`, against 0.07 for `RungeKutta()`; [limitations](adams.md#limitations)).
There:

- for a dense, multimode, Raman-active run such as a comb-to-continuum capillary, use
  `Multistep(carrier_frame="scalar")`: the fastest method measured, about half the
  default's and IP-DOPRI5's time;
- for a single-mode dense-gas run (an argon blue shift, a soliton self-frequency
  shift), use `Multistep(formulation="interaction")`;
- at a loose tolerance, `RungeKutta()` or `rtol <= 1e-4` with the default.

[When to use the carrier frame](adams.md#the-carrier-frame) gives the measurements and
its caveats: a weak band that is still forming, and a floor of about $|\kappa| L$
steps.

## What decides

**What a call costs against what a step costs.** Every step of either integrator also
pays fixed work that does not depend on the source: propagating the state, forming the
error norm, the controller, landing. The Runge–Kutta march spreads that over six calls;
the Adams method pays it for each call, and adds its history's arithmetic. When a call
dominates — a multimode transform pair, an ionisation rate, a Raman convolution, a
transverse transform — fewer calls is less time, nearly in proportion. When a call is
a few fast Fourier transforms of a single mode, the step's own work weighs as much, and
three times fewer calls can be slower. The compiled backends make calls cheap, which is
why the ranking is measured per backend rather than read off the call count.

**How complicated the nonlinearity is.** A richer response costs more per call, which
favours the Adams method, and its dynamics decide how far each step can go. The
exponential formulation integrates a *driven* source — the four-wave mixing, Raman and
plasma terms that build a dispersive wave or a continuum — at any linear phase per
step, where the interaction picture's history limits a step to a linear phase of about
one in the populated bins. That is why it leads most on the gas-filled capillaries with
ionisation (5.5 to 5.8 times faster than the Runge–Kutta march on the full RDW case,
with twelve to fourteen times fewer calls) and on the Raman-active multimode launch. A
source linear in a weak bin's own field rotates with that bin's free phase and favours
the interaction picture; on real problems it lives in the dark wings
([weak wings](adams.md#limitations)).

**How smooth the dynamics are, and how tight the tolerance.** The Adams method takes
more, shorter steps than the Runge–Kutta march at the same tolerance, each one call.
Smooth propagation at tight tolerance raises its order and lengthens its steps; abrupt
events — a fission, a kink in a profile — lower the order for a while. `rtol` means
something different to each integrator, so compare their errors, not their
tolerances. At `rtol` of 5e-2 or more the Adams method is unreliable and warned on.

**How long the run is, and how many.** A run's first few steps rebuild the history from
order one, a fixed cost that a short run feels most. A single short run hardly matters;
in a scan of thousands, measure one point with each integrator before choosing, and
spread the points over a `Pool` of one-thread workers, which scales with the cores:
batching in lockstep buys nothing on the CPU and about 1.1 times on a GPU
([scans](../workflows/scans.md#fixed-schedule-batches)).

**Planes.** Both integrators interpolate save planes inside their steps
(`interpolate_planes=True`) at the step's own order and no call; landing on them stays
the default, and with many planes the Adams method's saving grows, since landing caps
its steps.

## Measured

Indicative ratios at matched error against a converged reference, one thread on a shared
Darwin/arm64 workstation (the [gallery](#the-gallery) below has every point, and the
devices; [the benchmark notes](../../dev/benchmarks.md#indicative-results) say how to read
such ratios). Ratios are the Runge–Kutta march's time over the default's: above one, the
default is faster.

| case | the source | calls, `RungeKutta()` : `Multistep()` | time ratio, `numba_cpu` | time ratio, native CPU |
|---|---|---|---|---|
| full RDW capillary (four modes, ionisation, gas gradient) | expensive, varying | 12 to 14 : 1 | 5.5–5.8 | 6.1–6.3 |
| CtC launch (six modes, Raman) | expensive | 2.3 to 3.7 : 1 | 2.0–3.2 | 2.0–3.3 |
| radial focusing beam, filament with ionisation | spatial | 2.5 to 6 : 1 | 1.2–2.4 | 1.3–2.2 |
| Cartesian focusing beam (48 × 48) | spatial | 3 to 6 : 1 | 0.9–2.0 | 1.2–2.4 |
| full canonical capillary (one mode, ionisation, gas gradient) | moderate, varying | 1.8 to 2.4 : 1 | 1.0–1.3 | 1.1–1.45 |
| coupled flows: a coherent wall, a taper | cheap, coupled | 1.8 to 4 : 1 | 0.6–1.4 | 0.7–1.7 |
| line and planar self-focusing | cheap, spatial | 2.4 to 6 : 1 | 0.8–1.8 | 0.9–2.3 |
| GNLSE supercontinuum window | cheap, one mode | 3.6 to 4 : 1 | 0.7–0.8 | 1.05–1.1 |

Below one, the Runge–Kutta march was faster: on the GNLSE window on `numba_cpu`, at
the tightest targets on the line, planar and radial-aperture cases, and at one or two
targets of the coupled taper and the Cartesian focus.

On data-centre AMD and NVIDIA GPUs the ranking is broadly the same: the exponential
formulation was 4.4 to 4.8 times faster on the RDW capillary, 1.5 to 2.5 times on the
CtC launch and the Cartesian focus and 1.1 to 1.3 times on the 512-node filament, while
the Runge–Kutta march led on the canonical capillary and the coupled wall, by 1.25 to
1.9 times, because a device's fixed cost per step is larger. These device ratios were
measured with the error norm over the requested band rather than the whole active band,
under which the Runge–Kutta march pays up to a fifth more calls on the capillaries.

## Large problems: memory

Most problems fit many times over, and memory need not enter the choice. It does when
the state is large: a fine Cartesian grid, many modes over a wide window, a large batch
on a device. The Adams history keeps further copies of the state beside what the march
already holds — its source samples and, with a varying operator, the states and rates
of its compensation:

| backend | history, copies of the state |
|---|---|
| `numpy_ref` | 7; 21 with a varying operator |
| `numba_cpu` | 7 in the interaction picture; 16, or 30 with a varying operator, in the exponential one |
| `native` (CPU and device) | 29; about 60 with a varying operator |

What that adds depends on the rest. On a capillary the operator tables dominate, a
hundred to several hundred copies of the state, and the history adds 8 to 55 %. On a
Cartesian free-space grid the Runge–Kutta march holds 15 (`native`) or 16 (`numba_cpu`)
copies, by the backends' ledgers, and the default 32 (`numba_cpu`) to 44 (`native`): two
to three times the memory. The low-storage march holds four or five:

| integrator, on a Cartesian free-space grid | copies of the state |
|---|---|
| `Multistep()` | 20 (`numpy_ref`), 32 (`numba_cpu`), 44 (`native`) |
| `RungeKutta()` | 13 (`numpy_ref`, counted), 16 (`numba_cpu`), 15 (`native`) |
| `LowStorageRungeKutta()` | 5 with the half-step flow tabulated, 4 recomputed; one more with the `modified` noise; on `numba_cpu` and `native` one fewer without an in-stage collar |

At 768 × 768 columns and 600 bins one copy is 5.3 GiB: 26 GiB under the low-storage
march, against 79 or 84 GiB for the Runge–Kutta march and 232 GiB for Adams on the native backend. The estimate is made
before anything is allocated ({py:func}`solprop.backends.memory.estimate_memory`,
group `adams_history`). A run that would fit with `RungeKutta()` but not with its
history is refused before it starts, naming `RungeKutta()`: against the machine's
memory less a fifth when no budget is declared, and on a device against four fifths of
its free memory. A declared `memory_budget` or `SOLPROP_MEMORY_BUDGET` is admitted as
declared ([execution resources](../performance/resources.md#memory)).

## The low-storage march

`LowStorageRungeKutta()` is the classical fourth-order Runge–Kutta method in the
interaction picture anchored at each step's midpoint, Hult's RK4IP
([SEP 24](../../dev/seps/archive/sep-0024-low-storage-interaction-picture-runge-kutta.md)). Only
forward half-step flows are applied, and on a constant operator one factor
$e^{hD/2}$ serves every one of them. Each slope is folded into a running sum as soon as
it is produced, so beside the source pipeline's scratch a step needs three copies of
the state: the state itself, overwritten in place, an accumulator and the next stage's
input. Each step makes four source evaluations.

**What it costs and what it gives up.** It replays a prescribed schedule and nothing
else, so it needs `schedule=` (`FixedSchedule.uniform(...)` for plain fixed steps). It
has no embedded error estimate: every step's recorded norm is 0.0, meaning "not
estimated", and the run's closing line says so. Its accuracy is the schedule's,
measured on the observable with three-level Richardson calibration
([fixed schedules](fixed-schedules.md#the-low-storage-march)), which is also what makes
it the right tool for a weak band under a strong pump: the whole-field norm an adaptive
march controls is blind to a band holding a small fraction of the energy. Fourth order
needs more steps than fifth asymptotically; on stiff problems DOPRI5 measures about
fourth order at fixed step too, so compare the two at equal error, not at equal step.

**What it refuses**, by name and before anything is allocated: no schedule,
`interpolate_planes`, a coupled linear flow, the TDSE response, and a step past RK4's
negative-real stability interval beside an in-stage collar, $h\alpha_\max/2 \le 2.785$
against DOPRI5's 3.3 (about 45 steps over a run with the default temporal collar). A
Every backend runs it, on every route it admits it on, and `backend="auto"` resolves
as for any integrator: the native backend when it is installed, otherwise `numba_cpu`. On
`numba_cpu` the four passes are compiled register kernels with none of the Runge–Kutta
march's buffers; on the native backend a step is one native call making the same four passes in
the backend's own memory, and a plane that keeps no field holds its state there too.
Without an in-stage collar each pass writes the next one's input straight into the
transverse scratch, so the stage register is not held either. On a Cartesian Kerr beam
the march
[took 1.7 to 1.8 times less time per step](../../dev/benchmarks.md#low-storage-interaction-picture-rk)
than the IP-DOPRI5 replay on `numba_cpu`, and
[1.8 to 2.0 times less](../../dev/benchmarks.md#low-storage-interaction-picture-rk)
on native CPU, in a quarter of the memory; what both marches spend is mostly the
source evaluation itself. The compiled backends do not form the stiffness indicator
below, so the warning is `numpy_ref`'s alone.

**What a failure costs.** Because each step is accepted in place, an exception or a
signal between a step's first source evaluation and its commit leaves a state that is
neither end of the step. The backend then refuses to be observed or stepped until a
state is loaded again, and the run writes no terminal checkpoint, saying why: the file
ends at its last saved plane, from which `resume=True` continues, and `step_planes=k`
bounds the loss to `k` steps. A last plane that kept no field is held by the backend,
one more copy of the state, and checkpointed instead
([field retention](../results/output.md)). A field that diverges is caught on the accepted state, in
flight, with the same consequence. The march also watches its own stiffness: when the
indicator $h\rho$ of SEP 24 §6.3 passes 2.785 on a step -- strong plasma loss near
the stability limit -- it logs a warning.

**The flow.** `propagator_mode="tabulate"`, the default (`"auto"`) for a constant
operator, holds one copy of $e^{hD/2}$, reused while the step stays within eight ulps of
the step that built it; `"recompute"` forms it in each pass instead, the default for a
varying operator. The two agree to rounding. The free-space and planar kinds admit the
table under this march only.

## Recognising the regime

- **Stability-limited history.** Many rejected trials, or the order falling to one or
  two while the error estimate stays small: a stiff term the explicit history
  integrates (an in-stage absorber, a strong plasma term) limits the step, not
  accuracy. Use the split-step collar where the pulse never reaches the collar
  ([stability](adams.md#stability-and-the-temporal-collar)), or `RungeKutta()`.
- **Reference-limited error.** Errors that stop falling as `rtol` tightens: below the
  reference's own refinement (the dotted floor in each figure), nothing is resolved.
- **Source-limited accuracy.** The error is set by the response model's own
  discretisation (a transverse quadrature, a TDSE grid); a tighter `rtol` buys calls,
  not accuracy.
- **Dark wings.** A total-field error that stays well above `rtol` at tight tolerances
  while the populated bins follow it: free radiation far below the peak, which the
  exponential formulation integrates loosely ([weak wings](adams.md#limitations)).
  Score the populated band, or take the interaction formulation if those bins matter.

## The gallery

Each figure pairs, per backend, the case's worst-plane error against (left) source
calls and (right) the median of interleaved timed rounds; every ladder point is drawn
against calls and only the timed points against time. The dotted line is the
reference's refinement. Errors are scored per case: the
complex field and the weak band for the fibres and capillaries, the field in the scored
band for free space. IP-DOPRI5 is `RungeKutta()`; "Multistep, exponential" is the
default and "Multistep, interaction" its other formulation. "IP-DOPRI5, growth limit 2"
is IP-DOPRI5 allowed to grow a step by up to twice the last, as the Adams march is,
beside the default controller's 1.1.

CPU timings are one thread on a shared Darwin/arm64 workstation, compared within one
process; the device timings are the native HIP build on one data-centre AMD GPU and, for six
cases, the native CUDA build on one data-centre NVIDIA GPU, with the error norm over the requested
band; the CPU points use the default norm over the whole active band. Each figure's
points are in the CSV beside it.

```{figure} ../figures/stepper-gallery/gnlse_longer.svg
The GNLSE window: soliton fission in a fibre (envelope, one mode).
{download}`CSV <../figures/stepper-gallery/gnlse_longer.csv>`
```

```{figure} ../figures/stepper-gallery/ctc.svg
The CtC launch window: a multimode capillary, the expensive-source regime.
{download}`CSV <../figures/stepper-gallery/ctc.csv>`
```

```{figure} ../figures/stepper-gallery/canonical_full.svg
The full canonical helium capillary (4.2 m, steady fill), scored on the field and the
weak band. {download}`CSV <../figures/stepper-gallery/canonical_full.csv>`
```

```{figure} ../figures/stepper-gallery/rdw_full.svg
The full RDW capillary, scored on the field, the weak band and the UV.
{download}`CSV <../figures/stepper-gallery/rdw_full.csv>`
```

```{figure} ../figures/stepper-gallery/coupled_wall.svg
A coupled flow: a coherent capillary wall's mode coupling.
{download}`CSV <../figures/stepper-gallery/coupled_wall.csv>`
```

```{figure} ../figures/stepper-gallery/coupled_taper.svg
A coupled flow crossing sections: a taper.
{download}`CSV <../figures/stepper-gallery/coupled_taper.csv>`
```

```{figure} ../figures/stepper-gallery/radial_focus.svg
Free space, radial: a beam self-focusing at 0.9 of the critical power over 8 cm.
{download}`CSV <../figures/stepper-gallery/radial_focus.csv>`
```

```{figure} ../figures/stepper-gallery/cartesian_focus.svg
Free space, Cartesian (48 × 48): the same focusing beam.
{download}`CSV <../figures/stepper-gallery/cartesian_focus.csv>`
```

```{figure} ../figures/stepper-gallery/radial_filament.svg
Free space, radial: a filament at three times the critical power with ionisation (256
nodes on the CPU; 512 on the device). {download}`CSV <../figures/stepper-gallery/radial_filament.csv>`
```

```{figure} ../figures/stepper-gallery/radial_boundary.svg
Free space, radial: the focusing beam on a 0.3 mm aperture.
{download}`CSV <../figures/stepper-gallery/radial_boundary.csv>`
```

```{figure} ../figures/stepper-gallery/line.svg
Free space, line: a self-focusing strip. {download}`CSV <../figures/stepper-gallery/line.csv>`
```

```{figure} ../figures/stepper-gallery/planar.svg
A planar guide: two slab modes. {download}`CSV <../figures/stepper-gallery/planar.csv>`
```

The qualification on this page was proposed and carried out by
[SEP 16](../../dev/seps/archive/sep-0016-unified-multistep.md); the dense-gas advice
comes from [SEP 35](../../dev/seps/archive/sep-0035-carrier-frame-exponential-adams.md).
