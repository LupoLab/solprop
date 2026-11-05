# Execution resources

Budget memory before a large propagation or scan. The state size grows with time
samples, spatial nodes, modes and polarisations; saved planes can add a substantial
output cost. Choose an adequate model and resolution first, then select a
[backend](backends.md) and [thread settings](tuning.md) suitable for the machine.

## Backend choice and scan workers

`backend="auto"` resolves to the native backend, and refuses if it is missing or the
processor is below its minimum ([choosing a backend](backends.md#which-backend-a-run-uses)).
It does not silently retry another backend if the chosen one refuses a representation.
The resolved backend and runtime settings are recorded with the output.

Thread and runtime settings must be in place before the relevant libraries load.
Avoid giving every process all CPU threads: budget workers and threads together.
See the [scan executor guidance](../workflows/scans.md#processes-and-resource-limits)
and [tuning recipes](tuning.md#recipes).

## Memory

The general modal route holds dense frequency-dependent maps and the guide's stored E/H
reference for the whole run: $16\,Q\,C\,M\,n_\omega$ bytes per map, for $Q$ nodes, $C$
components, $M$ channels and $n_\omega$ frequency bins, up to fourteen such maps on a
constant guide. That grows quickly: the graded-index worked example (412 nodes, four
channels, 2049 bins) holds 2.3 GB. A coherent wall adds the flow's per-section matrices.

solprop doesn't check a run's memory before it starts. A backend allocates and first
touches its buffers when it's constructed, so a run that doesn't fit normally fails
near the start, in one of two ways: a failed allocation raises `MemoryError`, and a run
over a scheduler's or the kernel's limit is killed, with exit status 137 and no Python
error. Under Slurm, `sacct` reports such a job as `OUT_OF_MEMORY`.

To see the size first, {py:meth}`~solprop.driver.simulation.Simulation.estimate` builds
the backend and reports what it holds; a backend's
{py:meth}`~solprop.backends.protocol.Backend.memory_ledger` names every buffer, so you
can see which one to shrink: fewer channels or nodes for the maps and the reference, a
narrower frequency window for all of them. Nothing is ever approximated to fit. If a
large run with the default integrator doesn't fit, try `RungeKutta()` or
`LowStorageRungeKutta()`, which keep no Adams history
([choosing a stepper](../numerics/choosing-a-stepper.md#large-problems-memory)).

**The initial field.** The translated problem keeps the run's initial state for as long
as it lives, on the host. A
`Simulation` copies an array you pass when it is built, so that changing your array
afterwards changes neither the run nor its record. On a free-space or spatial route the
translated problem holds that same copy, one state's bytes in all. On a modal route it
holds the launch converted to its internal amplitude instead, beside the simulation's
copy, so an array input costs two states there. An array of another dtype than complex128 is kept once more as given,
for the record. A broadcast view such as `numpy.broadcast_to(column, shape)` copies only
the block it repeats, and a solprop pulse's build is a new array that is not copied at
all. A caller who loads its own state into one backend for
run after run can pass a broadcast placeholder and pay for almost no held copy at all, and
a [runner](../setup/running.md#many-inputs-one-backend-the-runner) holds no copy of an input
recorded by its digest or a stated identity, and no reference to it once it is loaded.

```python
import numpy as np

from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium

grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (700e-9, 950e-9), 150e-15, thg=False)
space = FreeGrid(4e-4, 4, 3e-4, 3, collar_fraction=0)
column = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9).build(grid)
placeholder = np.broadcast_to(column, (space.n_columns, 1, grid.n_omega))
settings = SimulationSettings(
    save_planes=[1e-4], dz_max=5e-5, dz_init=5e-5, backend="numpy_ref"
)
problem = Simulation.from_free(
    grid, space, SpatialMedium(), [], placeholder, settings=settings, noise=None
).translate()
print(problem.initial.field0.strides[1])  # 0: one column held, not a state
```

**The output.** A run kept in memory (`output=None` or a
{py:class}`~solprop.output.memory.MemoryOutput`) holds every plane it saves: the field,
the noise phase and the statistics of each. 201 saved planes of a large grid is not
small, and this is the part of a run's memory that can run out partway through. Fewer
planes (`n_save`) or a file (`output="run.h5"`), which writes each plane out, shrink it.
The step history grows with the steps taken.

**The propagator table.** `propagator_mode` on
{py:class}`~solprop.driver.simulation.SimulationSettings` decides whether the stage
propagators are kept for the whole step (`"tabulate"`) or recomputed in each stage
(`"recompute"`), which changes the memory held, not the method. Its
default, `"auto"`, leaves the choice to the backend and records what was resolved in the
backend record: `numpy_ref` tabulates, `native` recomputes, and a
free-space or planar run always recomputes and refuses `"tabulate"`.

**Operators tabulated along z.** A linear operator or nonlinear prefactor that varies
along z but cannot be expanded in the mode set's scalar parameter paths is tabulated
along z instead ([varying operators](../numerics/linear-tables.md#the-table)). On a
large multimode set that table can run to gigabytes. `numpy_ref` and a native CPU build run it; a native GPU build refuses it
([backends](backends.md)).

**A general modal guide.** Its route — the dense reference and the two execution maps,
nearly all of a large guide's memory — is built when the simulation is *constructed*,
so a guide that doesn't fit fails in
{py:meth}`~solprop.driver.simulation.Simulation.from_waveguide`, before `run`.

The backends' memory ledgers are in the [backend design](../../dev/backends.md#8-memory-ledger).
