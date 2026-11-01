# Execution resources

Budget memory before a large propagation or scan. The state size grows with time
samples, spatial nodes, modes and polarisations; saved planes can add a substantial
output cost. Choose an adequate model and resolution first, then select a
[backend](backends.md) and [thread settings](tuning.md) suitable for the machine.

## Backend choice and scan workers

`backend="auto"` resolves to the native backend when installed and to Numba otherwise. It does
not silently retry another backend if the chosen one refuses a representation.
Use an explicit name for a scan spanning machines with different installations.
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

Before a backend allocates anything, the driver **estimates** the run's persistent bytes
from the same formulas the backends' ledgers use
({func}`~solprop.backends.memory.estimate_memory`, equal to each ledger to the byte) and
admits the run against a budget
({func}`~solprop.driver.memory.admit_memory`). Undeclared, the budget is four fifths of
the memory available — the smallest of the physical memory, a Linux cgroup limit and a
Slurm allocation — and a run over it is **warned** about; a declared budget refuses a
run over it, unless it is prefixed `warn:`. One case is refused even undeclared: a run
that would fit but for the Adams method's history is refused before it starts, naming
`RungeKutta()`, which keeps none
([choosing a stepper](../numerics/choosing-a-stepper.md#large-problems-memory)); `warn:` or `"none"`
lets it try anyway. A budget is a byte count or a number with a
unit (`"16 GiB"`, `"12e9"`, `"64 GB"`), or `"none"` for no check. Declare it for one run
with `Simulation.run(memory_budget=...)`, or for every run in a process, including the
`prop_*` functions and scan workers, through the environment:

```python
from solprop.driver import Simulation

simulation = Simulation.from_capillary(
    125e-6,
    1e-3,
    "He",
    1.0,
    wavelength=800e-9,
    wavelength_limits=(200e-9, 2e-6),
    trange=100e-15,
    fwhm=10e-15,
    energy=1e-9,
    n_save=2,
    backend="numba_cpu",
)
budgeted = simulation.run(memory_budget="warn:8 GiB", progress="none")
print(budgeted.metadata.backend.settings.get("memory"))
```

```sh
SOLPROP_MEMORY_BUDGET="64 GiB" uv run python my_run.py   # or "warn:64 GiB", or "none"
```

The `prop_*` functions have no `memory_budget=` keyword, so the environment variable is
their only way to declare a budget; likewise field retention (`fields=`,
`field_planes=`) is set on {py:class}`~solprop.driver.simulation.SimulationSettings` or
through {py:func}`~solprop.driver.scan.run_scan`, not as a `prop_*` keyword.

The estimate is recorded in the file's backend record under `memory`, with its groups
(`base`, `general_maps`, `general_reference`, `general_chunk`, `coupled_flow`,
`adams_history`, ...), so
you can see which one to shrink: fewer channels or nodes for the maps and the reference,
a narrower frequency window for all of them. Nothing is ever approximated to fit. Not
counted, and named on every estimate: FFTW's planning heap and per-stage transients.

**The output.** A run kept in memory (`output=None` or a
{py:class}`~solprop.output.memory.MemoryOutput`) holds every plane it saves, allocated at
the start for the entrance and every save plane, so the admission adds it as the group
`output`: the field, the noise phase and the statistics of each plane
({py:func}`~solprop.driver.memory.plane_bytes`). 201 saved planes of a large grid is not
small; fewer planes (`n_save`) or a file (`output="run.h5"`), which writes each plane out
and is charged nothing, shrink it. The step history, which grows with the steps taken,
is not counted.

**The propagator table.** `propagator_mode` on
{py:class}`~solprop.driver.simulation.SimulationSettings` decides whether the stage
propagators are kept for the whole step (`"tabulate"`) or recomputed in each stage
(`"recompute"`), which changes the memory held in the `base` group, not the method. Its
default, `"auto"`, leaves the choice to the backend and records what was resolved in the
backend record: `numpy_ref` tabulates, `numba_cpu` and `native` recompute, and a
free-space or planar run always recomputes and refuses `"tabulate"`. When `base` is what
does not fit, `propagator_mode="recompute"` is one of the remedies the refusal names.

**Operators tabulated along z.** A linear operator or nonlinear prefactor that varies
along z but cannot be expanded in the mode set's scalar parameter paths is tabulated
along z instead ([varying operators](../numerics/linear-tables.md#the-table)). On a
large multimode set that table can run to gigabytes, and, like the problem's other
tables, it is not in the estimate. `numpy_ref` and `numba_cpu` run it; `native` refuses it
([backends](backends.md)).

**A general modal guide is admitted twice.** Its route — the dense reference and the two
execution maps, nearly all of a large guide's memory — is built when the simulation is
*constructed*, before `run` can see a translated problem. So
{py:meth}`~solprop.driver.simulation.Simulation.from_waveguide` admits the guide first
({py:func}`~solprop.driver.memory.admit_guide_memory`), under the same budget and the same
refuse-or-warn rule: `general_reference`, `general_maps` and `general_chunk` are estimated
from the guide's declared shape alone — its channels, its node rule's nodes, the grid's
bins and time samples, a moving guide's sections and degree
({py:func}`~solprop.backends.memory.estimate_guide_memory`) — and equal to the byte what
the translated estimate later finds for those groups. A guide over a budget declared to
refuse is refused before any of it is allocated; `from_waveguide(memory_budget=...)`
declares that budget, and otherwise `SOLPROP_MEMORY_BUDGET` or the machine's default
applies. The whole run is admitted again by `run`.

The estimate's groups, the backends' memory ledgers and the formulas behind them are in
the [backend design](../../dev/backends.md#8-memory-ledger).
