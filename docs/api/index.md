# API reference

One page per public module, class and function, generated from the docstrings. The
[user guide](../guide/index.md) explains how the objects fit together; this reference
describes their arguments, defaults, units, return values and errors. Quantities are in SI units unless a docstring says otherwise.

For interactive work, {mod}`solprop.sol` gathers the names a session needs behind one
import (`import solprop.sol as sol`; see [the interactive namespace](../guide/workflows/interactive.md)).
In scripts and library code, import from the home modules listed below.

## Find an interface by task

**Setting up and running a simulation**

| module | what it holds |
|---|---|
| {mod}`solprop.driver` | the entry points `prop_gnlse`, `prop_capillary`, `prop_waveguide`, `prop_radial`, `prop_free` and `prop_planar`, the `Simulation` object behind them, and `run_problem`, the loop over save planes that takes a translated problem, a backend already holding its initial state and an output handler, for a batch runner of your own |
| {mod}`solprop.grid` | time and frequency grids (`FieldGrid`, `EnvelopeGrid`) and the free-space transverse grids (`RadialGrid`, `FreeGrid`, `LineGrid`) |
| {mod}`solprop.pulses` | initial pulses: `GaussianPulse`, `SechPulse`, `CustomPulse`, `DataPulse`, and spatial and planar pulses |
| {mod}`solprop.profiles` | quantities that vary along z: radius, pressure and temperature profiles such as `PiecewiseLinear` and `PressureGradient` |
| {mod}`solprop.gasflow` | the steady gas flow of a pressure gradient: `steady_fill`, choked ends, `flow_report` |
| {mod}`solprop.boundaries` | absorbing boundaries in time and frequency |
| {mod}`solprop.stepper` | the Runge–Kutta tableaux, the low-call Adams methods (`multistep`), the step-size controller and fixed step schedules (`FixedSchedule`) |
| {mod}`solprop.noise` | shot-noise realisations |
| {mod}`solprop.driver.scan`, {mod}`solprop.driver.executors` | parameter scans and where their points run ([scans and executors](scans.md)) |

**The physics**

| module | what it holds |
|---|---|
| {mod}`solprop.modes` | waveguide modes: Marcatili capillary modes and the `CoherentWall`, antiresonant, rectangular, slab and [tabulated](tabulated.md) modes, the step-index and graded-index providers, and `ModalGuide` for the general modal route |
| {mod}`solprop.nonlinear` | the nonlinear responses — Kerr, Raman, plasma and ionisation — and the mode-averaged GNLSE |
| {mod}`solprop.nonlinear.tdse` | frozen helium TDSE physics, calibration, sampling and execution declarations, the haCC basis, the atomic window and the consistency policy ([the TDSE response](../guide/media/tdse.md#declaring-a-response)) |
| {mod}`solprop.driver.tdse` | atomic preset resolution (`basis_from_preset`, the haCC `hacc_preset` and `hacc_basis`); its binding interfaces are for developers |
| {mod}`solprop.driver.tdse_consistency` | the haCC weak-field consistency check (`check_consistency`, `ConsistencyResult`), its cache and the cubic evidence extraction ([the consistency check](../guide/media/tdse.md#the-weak-field-consistency-check)) |
| {mod}`solprop.spatial` | bulk media and free-space (radial, Cartesian, line) propagation, with the spatial measures of a beam |
| {mod}`solprop.adiabatic` | the real-index path certificate for a medium that varies along z: `AdiabaticAccuracy` and its admission record |
| {mod}`solprop.linear` | the linear operator: dispersion, loss and the moving frame |
| {mod}`solprop.physdata` | material data: gases and mixtures ([species data](species.md)), glasses and metals, refractive indices, nonlinear, Raman, ionisation and transport data |
| {mod}`solprop.polarisation` | Jones optics and the description of transverse polarisation |
| {mod}`solprop.fourier`, {mod}`solprop.hankel` | the Fourier conventions and transforms, and the radial (Hankel) transform |
| {mod}`solprop.constants` | physical constants in SI units |

**Reading and using the results**

| module | what it holds |
|---|---|
| {mod}`solprop.output` | the in-memory and HDF5 output handlers, the run's metadata record, `open_file` and resuming a run, live snapshots (`SnapshotController`, `read_snapshot`, `restore_snapshot`; [the output page](../guide/results/output.md)), and the raw TDSE response archive (`TDSEArchive`, `open_tdse_archive`; [the TDSE response](../guide/media/tdse.md#the-raw-response-archive)) |
| {mod}`solprop.processing` | post-processing of a saved run: fields, spectra, pulse metrics, coherence, compression, the physical fields of a general modal run, and `field_plane`, a saved plane read back for coupling |
| {mod}`solprop.coupling` | coupling simulations through a field plane: `FieldPlane`, `regrid`, `InputPort` and the `same_basis` relaunch; `transport` through the linear elements `FreeSpace`, `ThinLens`, `Aperture`, `TransmissionMask`, `Alignment`, `PolarisationTransform`, `Mirror`, `PlanarInterface` and `SpectralTransfer` ([coupling simulations](../guide/workflows/coupling.md); [the transport report](transport.md)) |
| {mod}`solprop.stats` | the statistics recorded at the saved planes, and `StatsSpec` to choose them |
| {mod}`solprop.plotting` | the standard figures: propagation maps, slices, spectrograms, statistics |
| {mod}`solprop.optics` | linear optics applied to a pulse outside a propagation: materials, chirped mirrors, grating and prism pairs |
| {mod}`solprop.tools` | design calculators: soliton numbers, dispersive-wave phase matching, nonlinear lengths |

**Session, logging and errors**

| module | what it holds |
|---|---|
| {mod}`solprop.sol` | the interactive namespace |
| {mod}`solprop.log` | logging and progress reporting (`configure_logging`) |
| {mod}`solprop.exceptions` | every error and warning solprop raises, under `SolpropError` and `SolpropWarning` |
| {mod}`solprop.scan` | the command-line launcher of a scan script |

**Internals.** {mod}`solprop.translate` turns the physics objects into the kernel problem
a backend executes, and {mod}`solprop.backends` holds the in-tree backends (`numpy_ref`,
`numba_cpu`), the registry that selects a backend by name and the memory estimate. They
are documented for developers and backend authors rather than for everyday use; the
design behind both is in the [developer documentation](../dev/index.md).

```{toctree}
:hidden:

species
tabulated
scans
transport
```

## Complete generated reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   solprop
```

## The loss specification and the evanescent floor

Every gas-filled hollow mode — the capillary's `MarcatiliMode`, the rectangular
`RectMode`, the planar `SlabMode` and the antiresonant modes — takes its `loss=` in one
vocabulary, `LossSpec`, and its full model floors the real part of the effective index
at `EVANESCENT_INDEX_FLOOR` far below the band, where the square root would otherwise
turn evanescent. Both are defined in {mod}`solprop.modes.fill` and are also importable
from {mod}`solprop.modes.capillary` and {mod}`solprop.modes`; they are documented here
under both module paths.

```{eval-rst}
.. autodata:: solprop.modes.fill.LossSpec

.. autodata:: solprop.modes.fill.EVANESCENT_INDEX_FLOOR

.. autodata:: solprop.modes.capillary.LossSpec

.. autodata:: solprop.modes.capillary.EVANESCENT_INDEX_FLOOR
```
