# Scans and executors

A scan runs one simulation per point of a grid of parameters, and an executor decides
where and how those points run. The [scan manual](../guide/workflows/scans.md) works through a
pressure–energy example and states the retention, seed and serialisation rules; the
modules involved are:

- {mod}`solprop.driver.scan`: `Scan` (ordered variables and their global points),
  `ScanPoint` and `run_scan`;
- {mod}`solprop.driver.executors`: the execution policies — `Local`, a spawned process
  `Pool`, a lockstep `Batch`, a durable `Queue` and `Slurm`;
- {mod}`solprop.processing.scans`: the reductions a worker evaluates for each point
  (energy in a band, a spectrum, mode energies, a custom reduction) and the reader of a
  collected scan (`open`, `ScanOutput`);
- {mod}`solprop.output.scan`: the scan file, into which the controller commits each
  completed point;

Backend and storage interfaces below are mainly for contributors extending an executor:

- {mod}`solprop.output.scan_artifacts`: the validated per-point results a durable queue
  stores;
- {mod}`solprop.driver.executors.queue`: collecting a queue's results, safely repeatable;
- {mod}`solprop.driver.executors.slurm`: writing and submitting a Slurm array script;
- {mod}`solprop.driver.batch`: grouping compatible points into a batch and splitting its
  output back into one run per point;
- {mod}`solprop.driver.executors.batch`: the controls of a fixed-schedule batch;
- {mod}`solprop.backends.batch`: the backend side of a batch, in which every member
  takes the same fixed steps and is observed on its own.
