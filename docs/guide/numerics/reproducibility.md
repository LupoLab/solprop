# Reproducibility

solprop promises numerical accuracy, not identical bits. Two runs of the same problem
on different machines, backends, builds or thread counts agree within the accuracy the
problem was qualified to; they need not agree in every bit, and the faster code paths
depend on that freedom. The policy and its derivations are in
[SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md).

## What is promised

- **Accuracy, not bitwise identity.** Qualified cases agree within their documented
  error budgets. A new problem's accuracy comes from a convergence study (see
  [accuracy](accuracy.md)); the local tolerances `rtol` and `atol` alone do not bound
  its global error.
- **Rounding can change the steps.** An adaptive run accepts or rejects a step, and an
  Adams method picks its order, by comparing an error estimate with the tolerance.
  Near a threshold a rounding-level difference can tip one decision, after which the
  two runs take different steps. Their step counts, call counts and trajectories may
  then differ across machines, backends or thread layouts, while both meet the
  tolerance. Exact accounting of the work a run actually did is kept.
- **Replay keeps the accuracy requirement.** A recorded step schedule is useful for
  repeat calculations and debugging; calibrate it for the method and problem. A replay
  meets the original run's accuracy target; it need not reproduce the original run at
  rounding level.
- **Noise inputs stay controlled.** A seeded noise realisation is the same draw every
  time. Compare matched realisations where the problem is well conditioned, and
  ensemble observables where it is sensitive to the noise.
- **Some things stay exact.** Lossless storage and copies, round trips through output
  files, integer and identity metadata, hard-mask zeros and explicit zero or no-op
  results are exact, and the run record's configuration hash identifies a run.

## Comparing two runs

Compare the observable you care about, relative to its size, against the accuracy you
qualified the problem to. For two runs of one problem, a difference far below that
accuracy is the expected outcome; a bitwise comparison is not a test of correctness. If
two adaptive runs differ by more than expected, run both at a tighter tolerance: if
their difference falls with the tolerance, it was the truncation error both runs share,
not a defect.

## Replaying a schedule

When the steps themselves must be the same, for a batch run in lockstep, for debugging,
or to make a numerical error common to every member of an ensemble, record the schedule
once and replay it ([fixed schedules](fixed-schedules.md)):

```{code-block} python
from solprop.stepper import calibrate, discover, replay

# accepted: the accepted step lengths (m) of one adaptive run, in order.
# quantity(schedule): runs the problem on a schedule and returns the observable.
schedule = discover(accepted)
calibration = calibrate(schedule, quantity, order=5, requested=1e-8)
fixed = replay(schedule, calibration, calibration.achieved_error / 4)
```

Runs on the same fixed schedule take the same steps, so their differences are rounding
alone, and they agree far more closely than two adaptive runs do. That is still a
numerical agreement, not a bitwise one.

## Seeded noise

A {py:class}`~solprop.noise.NoiseRealisation` carries the seed that drew it, and the
same seed gives the same draw on any machine. Two runs with the same realisation differ
only by the propagation's own rounding, which a sensitive process (modulational
instability, supercontinuum from noise) can amplify. Compare such runs through ensemble
observables over several seeds — the mean spectrum, the coherence — rather than one
realisation's field at full length.
