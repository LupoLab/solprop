# Fixed schedules and repeatable propagation

A fixed schedule prescribes where the solver steps. Use it to replay positions or run compatible problems in a batch, after checking its accuracy for the parameter range. Accepting every prescribed step does not make its local error acceptable. Adaptive stepping remains the starting point for discovering a suitable schedule.

## Fixed schedules: discover, calibrate, replay

Adaptive stepping does not remove the convergence study, and fixed stepping is not less
accurate. It wins outright in three places: for ensembles, where identical steps make
the numerical error a common systematic that cancels between members instead of a
per-member variance that adds; for repeat calculations and debugging, since a replayed
run takes the recorded steps and meets the original run's accuracy target (it need not
reproduce its bits, [SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md));
and on a device, where there is nothing to synchronise. Both modes are therefore first
class, and batch lockstep execution uses fixed schedules only.

1. **Discover.** One adaptive run records $h(z)$;
   {py:obj}`~solprop.stepper.schedule.discover` turns the accepted step lengths into a
   {py:obj}`~solprop.stepper.schedule.FixedSchedule`.
2. **Calibrate.** {py:obj}`~solprop.stepper.schedule.calibrate` replays it at $h$, $h/2$ and
   $h/4$ and reads the observed order from the ratio of successive differences, then
   estimates the error the base schedule actually carried by Richardson extrapolation.
3. **Replay.** {py:obj}`~solprop.stepper.schedule.replay` rescales the schedule along the
   measured exponent to reach a target error. Rescaling follows the discovered profile,
   so wherever the adaptive run chose to go slowly, the replayed one does too.

```{eval-rst}
.. code-block:: pycon

    >>> import numpy as np
    >>> from solprop.stepper import FixedSchedule, calibrate, replay
    >>> base = FixedSchedule(np.full(8, 0.125))
    >>> # Stand-in for a run: an observable whose error falls as h**5.
    >>> quantity = lambda schedule: 1.0 + schedule.max_step**5
    >>> calibration = calibrate(base, quantity, order=5, requested=1e-8)
    >>> round(calibration.observed_order, 6)
    5.0
    >>> replay(base, calibration, calibration.achieved_error / 32).n_steps
    16
```

Two refusals are built into this. The **order gate** rejects an observed order more than
three above the scheme's: refining removed something that is not truncation error, so
the coarse schedule is off its convergence curve rather than merely large. Three is the
margin because a legitimate strongly nonlinear stretch reads about half an order high
while this failure mode reads nine orders high, and a two-level extrapolant cannot tell
them apart at all. Separately, when the coarsest difference sits within a factor of 1000
of the round-off floor, `calibrate` marks the result **inconclusive** and declines to
pronounce an order — below the floor the differences measure accumulated rounding, and
the same ratio has come back as $-0.83$ there and been read as healthy. An inconclusive
calibration cannot be replayed from; that is the point of measuring rather than assuming.

### Plain fixed steps

A fixed-step run with no adaptivity at all needs no discovery.
{py:obj}`~solprop.stepper.schedule.FixedSchedule.uniform` cuts the run at every save plane
and hard kink and covers each segment with the fewest equal steps of at most `dz`, so no
step exceeds `dz` and every target is an edge of the schedule. There is no dense output,
so a fixed schedule *must* contain the planes; this constructor guarantees it.

```{eval-rst}
.. code-block:: pycon

    >>> from solprop.stepper import FixedSchedule
    >>> FixedSchedule.uniform(0.0, 1.0, 0.3, save_planes=[0.5]).steps
    array([0.25, 0.25, 0.25, 0.25])
```

Replaying a schedule is not blind. The embedded error estimate costs nothing extra, so a
fixed run still computes the scaled norm every step, records it in the step history and
summarises it in {py:obj}`~solprop.stepper.controller.ControlRecord`; when
`met_tolerance` is false a step was too coarse for the tolerance the norm is scaled by,
and the driver warns. That is the fixed-mode counterpart of the `dz_min` error: neither
mode fails silently.

**Ensembles with noise.** Calibrate a schedule for a noisy ensemble with noise on, and
check it on more than the seed that discovered it. A schedule discovered on a noiseless
run integrates the seed's own dynamics poorly: on a 20 pJ fibre soliton it got the
noise-driven part of the field wrong by 6 % with `RungeKutta()` and 78 % with
`Multistep()` ([SEP 37](../../dev/seps/archive/sep-0037-noise-and-stepper-defaults.md), §3.4).

## The low-storage march

`integrator=LowStorageRungeKutta()` ([choosing a stepper](choosing-a-stepper.md#the-low-storage-march))
replays a schedule and does nothing else: it has no embedded estimate, so there is no
norm to report and none to warn on. Every step's recorded norm is 0.0, meaning "not
estimated", and `met_tolerance` is then vacuously true. Its accuracy is calibrated from
outside, on the quantity you will report, with `calibrate(..., order=4)`:

- **Calibrate the observable, not `rtol`.** The march's `rtol` sets nothing but the
  recorded tolerance and the spatial collar's diagnostic threshold. The observable can be
  anything the run produces: an energy in a band orders below the pump, a peak, a
  difference of two strong fields. A whole-field norm sees a band holding a fraction
  $\delta$ of the energy at about $\sqrt\delta$ of its weight, which is why calibrating
  on the band itself is the safer check.
- **`requested` is a label.** A uniform schedule was not discovered by an adaptive run,
  so there is no tolerance it was asked for; pass the controller's `rtol` or the error
  you are aiming at, which `calibrate` only quotes in its message.
- **The order gate reads four.** With `order=4` a ladder observing more than seven is
  refused: refining removed something that is not truncation error. In the asymptotic
  range ($|\operatorname{Im} D|\,h/2 \lesssim 1$ in the populated bins) the march reads
  four; a coarser base reads lower and is not yet a calibration.
- **A weak observable needs its floor.** The strong field's rounding leaks into a weak
  band and sets the floor `calibrate(floor=...)` must be told. The difference between two
  backends on one schedule is pure rounding and measures it: `numba_cpu` or `native` against
  `numpy_ref`, or one backend's two flow treatments (`propagator_mode="tabulate"` and
  `"recompute"`).
- **The flow table follows the schedule.** Tabulating a constant operator holds one
  copy of the state, $e^{\bar h D/2}$, built for the mean step $\bar h$ of each run of
  equal steps and reused along it: a uniform segment builds it once, and the phase it
  applies over the segment is the segment's own. A schedule whose steps all differ (a
  discovered one) rebuilds it every step, where `propagator_mode="recompute"` is the
  cheaper choice.

A third-order soliton's first compression, scored on the light below 780 nm, which holds
less than a millionth of the energy (a few seconds):

```python
import numpy as np

from solprop.driver import Simulation
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, calibrate, replay

length = 0.2  # m


def weak_band(schedule):
    """Replay `schedule` and return the energy fraction below 780 nm."""
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=2e-12,
        width=100e-15,
        peak_power=9 * 11.83e-27 / (0.11 * 1e-26),
        pulseshape="sech",
        n_save=2,
        noise=None,  # a noiseless soliton: the calibration is about the stepper
        schedule=schedule,
        integrator=LowStorageRungeKutta(),
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
    )
    output = simulation.run(progress="none")
    spectrum = np.abs(output["field"][-1, 0, 0]) ** 2
    wavelength = 2 * np.pi * 299792458.0 / simulation.grid.omega
    return float(spectrum[wavelength < 780e-9].sum() / spectrum.sum())


base = FixedSchedule.uniform(0.0, length, length / 32)
calibration = calibrate(base, weak_band, order=4, requested=1e-6)
print(f"observed order {calibration.observed_order:.2f}")  # 3.88
production = replay(base, calibration, calibration.achieved_error / 10)
print(production.n_steps)  # 58
```

A replay is accepted in place, so an exception or a signal inside a step leaves no
state to checkpoint: the run says so and its file ends at the last saved plane, from
which `resume=True` continues; when that plane kept no field, the state held there is
checkpointed instead ([field retention](../results/output.md)). Save planes often enough (`step_planes=k` saves the end
of every `k`-th step) to bound what a stop loses.
