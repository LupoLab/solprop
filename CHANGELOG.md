# Changelog

All notable changes to solprop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from git tags.

<!-- start changelog -->

## Unreleased

### Added

- Performance-history rows for 0.2.0 (`benchmarks/history/`) on the same nine hardware sets
  as 0.1.0.

### Changed

- The README, the package metadata, the desktop front end's Help and the native backend's
  error messages now link the manual of the newest release (`/en/stable/`) rather than the
  development version (`/en/latest/`), and the installation page's advice for a platform
  without a native wheel now allows for the yanked 0.1.0.
- Building a graded-index guide and run, and a large free-space launch, take less time: a mode solve evaluates its material once instead of once per finite element, a free-space launch forms each distinct column's admittance once, and neither repeats a table evaluation or a pass over the beam it already made. On the `grin_nonlinear` benchmark the build and translation go from 24 s to 16 s on one core, and on `kerr_beam_384` the launch's build from 12.6 s to 7.1 s. Every number of the translated problem is unchanged.

### Fixed

- The boundary monitor's record counts each committed step once: a step left unchecked and then checked at its saved plane is no longer counted among both the checked (`steps`) and the skipped steps, so the two add up to the committed steps. Under interpolated planes, a plane that no check read since the previous one, as a later plane inside one step, records its collar and guard occupancies as unmeasured (`None`) rather than the previous collar reading and empty guards (#133).

## [0.2.0] - 2026-11-07

solprop 0.2.0 adds a desktop front end, describes every run as a recipe that can be
checked, saved and rerun, and retires the numba backend in favour of the native one,
which is now required. Every break is listed first, under "Changed (breaking)", with how
to migrate (ADR-058).

Highlights:

- **solprop-gui**, a desktop front end, **experimental** in this release and published
  as a third distribution at solprop's own version (`pip install "solprop[gui]"`): it
  sets up, runs, scans and reads simulations without a script.
- **Recipes, the command line and design solvers**
  ([SEP 41](docs/dev/seps/archive/sep-0041-recipes-and-front-ends.md)): a TOML recipe
  for every route, the `solprop` command, pins that set a radius or a pressure from a
  zero-dispersion wavelength, a dispersive-wave wavelength or a soliton order, chosen
  launch modes, displaced and tilted beams, and parameter metadata for front ends.
- **The native backend is required**
  ([SEP 43](docs/dev/seps/archive/sep-0043-retire-numba-cpu.md)): `numba_cpu` and the
  numba and pyFFTW dependencies are gone, `pip install solprop` installs
  `solprop-native`, and new wheels cover Windows on Arm and x86-64 processors without
  AVX2.
- **Exact resume**: a file continues only under an identical configuration hash, as in
  0.1.0, a refusal names the fields that differ (ADR-066), and `solprop resume`
  continues a run from its file alone.
- **Output schema 2**: every field of a recorded dataclass is recorded, a `DataPulse`
  keeps its phase, large record arrays are stored beside the record's JSON, and a file
  no longer grows with the square of its planes. Read a 0.1.0 file with 0.1.0.
- **Transverse absorbing boundaries**
  ([SEP 40](docs/dev/seps/archive/sep-0040-transverse-absorbing-boundaries.md)): every
  free-space run carries a safety ramp by default, a designed real-space collar on
  request, and a boundary monitor that warns or stops when light reaches the edge; that
  edge is now read where the state lives
  ([SEP 44](docs/dev/seps/archive/sep-0044-native-edge-read.md)), so a device run
  copies no state to check it.
- **MLX performance**: compiled stages run a trial 1.5 to 2.6 times faster on an Apple
  GPU, single precision admits the Adams integrator, and single-precision grids are
  sized for MLX's transforms.

### Changed (breaking)

#### The native backend replaces numba

- The `numba_cpu` backend, and the `numba` and `pyfftw` dependencies, are removed
  ([SEP 43](docs/dev/seps/archive/sep-0043-retire-numba-cpu.md), ADR-064).
  `backend="numba_cpu"` raises `BackendError`, naming `native` and `numpy_ref`. A file
  written by 0.1.0 is refused by its output schema, as every file of another schema is:
  read it with `solprop==0.1.0`, which is also the rollback. To migrate, pass
  `backend="native"` (or leave `backend="auto"`), or `numpy_ref` for the reference.
- The settings that only `numba_cpu` read are gone: `SOLPROP_NUMBA_THREADS`,
  `SOLPROP_NUMBA_REGIME`, `SOLPROP_FFTW_THREADS`, `SOLPROP_TRANSVERSE_FFTW_THREADS`,
  `SOLPROP_PASS_TIMERS`, `NUMBA_DISABLE_JIT` and `NUMBA_NUM_THREADS`. Drop them from job
  scripts; they now do nothing. `SOLPROP_THREADS`, `SOLPROP_NATIVE_THREADS`,
  `SOLPROP_FFTW_PLAN`, `SOLPROP_FFTW_WISDOM` and `SOLPROP_BLAS_THREADS` stay.
- `solprop-native` is a required dependency, with no extra: `pip install solprop`
  installs it, and solprop is supported where it installs
  ([SEP 43](docs/dev/seps/archive/sep-0043-retire-numba-cpu.md)). On any other platform,
  including Intel macOS, ppc64le, musl and macOS before 14, the installer reports that no
  matching distribution exists; the installation page explains that and how to report a
  platform. The `native` extra is gone: install `solprop`, not `solprop[native]`.
- `backend="auto"` has no fallback: it is `native`, and refuses with no native package
  or on a processor below the floor, such as Arm below Neoverse N1. A TDSE response still
  resolves to `numpy_ref` and single precision to `mlx`. `numpy_ref` stays selectable by
  name, as the numerical reference and for the features only it runs. Where `auto`
  refuses, pass `backend="numpy_ref"` explicitly.
- `KERNEL_SCHEMA_VERSION` and `KernelProblem.schema_version` are gone, and with them the
  kernel schema number, and `solprop_native.BuildInfo.schema_version` is now
  `source_digest`, the SHA-256 of the C++ sources and `CMakeLists.txt` the extension was
  compiled from: a native run's record names it in `backend.settings` in place of the
  number. Neither enters a configuration hash or a resume check.
  `solprop_native.check_build()` takes no problem, and in an editable checkout refuses an
  extension built from other sources than `native/`, naming `uv sync --reinstall-package
  solprop-native` (ADR-068). To migrate, read `build_info().source_digest` where code
  read `schema_version`, call `check_build()` with no argument, and drop
  `solprop.translate.KERNEL_SCHEMA_VERSION` from imports and checks: solprop-native is
  pinned at solprop's own version, which is now the whole contract.
- solprop no longer checks a run's memory against a budget before allocating (ADR-065).
  `memory_budget=`, `SOLPROP_MEMORY_BUDGET`, `estimate_memory`, `estimate_guide_memory`,
  `solprop.driver.memory`, `MemoryBudgetError` and `MemoryBudgetWarning` are gone. A run
  that doesn't fit now fails with `MemoryError`, or is killed by the kernel or a
  scheduler. Scan workers and TDSE's `memory_limit` are still bounded; going over that
  limit raises `DriverError`, and it now covers atomic storage alone. Drop
  `memory_budget=` from calls and `SOLPROP_MEMORY_BUDGET` from job scripts; bound memory
  with the scheduler instead, and read `Simulation.estimate()` for an estimate.

#### Output files and resume

- The output schema is 2: a `DataPulse` records its `spectral_phase` (below), and every
  field of a recorded dataclass is recorded, at its default too. This release refuses a
  file written by 0.1.0, and every configuration hash moves, so a 0.1.0 run cannot be
  resumed; read such a file with 0.1.0. Code that reads a file with h5py meets these
  changes of layout:
  - `/meta/arrays/<name>`: a run record's numeric arrays of more than 4096 elements,
    which `/meta/json` now references (below);
  - `/record_arrays/`: the same arrays in a scan file, shared between points;
  - `/meta/outcome`: a Multistep call ledger's predictor orders as `[order, count]`
    pairs (below);
  - `absorbed_power`: 5 columns, where it had 4 (below);
  - a top-level `recipe` group in a run built from a recipe.

  Read files through `solprop.output` and `solprop.processing` where possible; they
  handle each of these.
- `resume` takes `False`, `True` or `"interpolated"`, and any other value is a
  `ValueError`: a mistyped word used to read as `True`. A resume continues only on an
  exact configuration hash, as in 0.1.0, and a refusal now names the fields that changed
  (ADR-066). A changed tolerance or integrator, an optmat release, a commit that adds or
  moves a recorded number, and another kind of machine are all refused; a long job has to
  stay on one CPU family to resume, since the record holds numbers computed on the
  machine that built it. An appending `HDF5Output` holds the file's own record as its
  `metadata` after a resume, not the one offered. To migrate, pass one of the three
  values; to continue under changed settings, launch a new segment from a saved plane or
  write a new file.
- An output file no longer grows with the square of its planes. `HDF5Output` rewrites
  `/meta/outcome` in place instead of deleting and recreating it at every plane, a saved
  plane builds and validates its run record once instead of nine times, and the step
  driver keeps only the error norms its controller reads, not one per accepted step: a
  Multistep run saved step by step (`step_planes=1`) costs 19.5 kB a plane at 707 planes
  and 19.3 kB at 3479, where it cost 23.8 kB and 31.5 kB. To get there a Multistep run's
  call ledger (`logical_calls` and `execution.response_calls` in the run's outcome,
  `/meta/outcome` in an HDF5 file) records its predictor orders run-length encoded, as
  `[order, count]` pairs, where 0.1.0 wrote one integer per accepted step. Code that read
  that list should expand the pairs, or read the record through `CallLedger.from_record`,
  which gives the orders step by step
  ([#80](https://github.com/LupoLab/solprop/issues/80)).
- `absorbed_power` has a fifth channel, `custom`, the power a `custom_absorber` rate
  removes, and its `spectral` channel is now the collar alone, so a frequency-selective
  loss reads apart from the numerical collar. The statistic's shape goes from `(4,)` to
  `(5,)`; the first four channels keep their positions and their sum stays `-dE/dz`.
  `Absorber.user_rate` holds the user's share of `spectral_rate`, which stays the total,
  so the operator does not move (#231). Code that read `absorbed_power[..., spectral]`
  as all the spectral loss under a `custom_absorber` should add the new `custom` column
  (index 4).

#### Free-space transverse boundaries

- A free-space run's transverse boundary is a `SpatialAbsorber`
  ([SEP 40](docs/dev/seps/archive/sep-0040-transverse-absorbing-boundaries.md)), given as
  `SimulationSettings.spatial_absorber` or the `spatial_absorber` keyword of
  `prop_radial`, `prop_free`, `prop_planar` and their `Simulation.from_*` builders. By
  default every radial, Cartesian, line and planar run carries a designed safety ramp on
  its outer 10 % of transverse wavenumbers, folded exactly into the operator, radial and
  planar included; `spatial_absorber=False` removes it. A real-space collar is designed
  on request, in the grid's room outside its requested size, from a protected band
  (`kappa`, `angle`, or the launch) and its width, not from the run length, and is
  refused by name where it cannot meet its budget. `TransverseCollar`,
  `transverse_collar_rate`, `SimulationSettings.transverse_absorber` and
  `Absorber.spatial` are removed, and `Absorber.collar_length` no longer sets a
  real-space rate. To migrate, drop `transverse_absorber=` for the default ramp, or pass
  `spatial_absorber=SpatialAbsorber(kappa=...)` (or `angle=...`) with a grid
  `collar_fraction` that leaves it room.
- Every spatial grid takes `collar_fraction` in place of `window_factor`, default 0, and
  puts the room it asks for **outside** the requested size on every geometry
  ([SEP 40](docs/dev/seps/archive/sep-0040-transverse-absorbing-boundaries.md)).
  `RadialGrid`'s `radius` is now the requested aperture, the Dirichlet boundary is at
  `outer_radius = radius * (1 + collar_fraction)`, and `n_r` counts every sample; the
  transform, the nodes and the effective area use the outer radius. Fractions of 1 and
  more are allowed. `FreeGrid(rx, nx, ry, ny)` and `LineGrid` lose their inert 10 %
  margin, so their spacing at the same sample count changes; give
  `collar_fraction=0.1` for the old grid. `RadialGrid(R, n_r)` keeps its wall and
  spacing. To keep a radial wall at `R` with a collar, give `radius=R / (1 + c)` and
  `collar_fraction=c`.
- A free-space run whose light reaches its transverse boundary now warns or stops, where
  it used to log and carry on
  ([SEP 40](docs/dev/seps/archive/sep-0040-transverse-absorbing-boundaries.md)). The
  boundary monitor reads the outer 5 % of the grid on every backend. It warns above 1e-6
  of the launch energy and stops at 1e-4, naming a size and a sample count to try, and
  `BoundaryCheck.transverse_budget` sets those levels apart from the time window's. The
  safety ramp's removal is judged at the same levels. Its monitor record reports a
  `transverse_status`, with `ramp_removed` and `spatial_removed`. Reads are at most
  `min(L/16, 4g/v)` apart, a ceiling on the step, so a low-source run takes a few more
  steps. A launch with more than 1e-4 of its energy in the ramp's band is refused. A run
  that now stops was losing light at its edge: enlarge the grid as the message says,
  request a real-space collar, or set `transverse_budget` deliberately.

#### Names, keywords and models

- Names and keywords with two meanings or two spellings are settled, with no aliases;
  `docs/dev/conventions.md` section 7 gives the rule for each. Rename the calls as
  listed:
  - `prop_2d`, `time_1d` and `spectrogram` take `time_range`, not `trange`, which stays
    the scalar window length on `prop_*`. `DEFAULT_TRANGE` is `DEFAULT_TIME_RANGE`.
  - `stats.zdw` and `modes.base.zdw` take `wavelength_limits`, not `wavelength_range`,
    as does `wavelength_to_axis`. `wavelength_range` is now only the attribute that says
    where a data model is valid.
  - `bandpass` and `Passband` take `spec_range`, not `limits`, as the plots and
    accessors do; `Passband.limits` is `Passband.spec_range`. A call passing the pair
    by position is unaffected.
  - `band_coherence` takes `db_min`, not `floor_db`.
  - `plotting.energy` and `plotting.stats` are `plot_energy` and `plot_stats`, the names
    they already had in `solprop.sol`.
  - `backends.protocol.StepHistory` is `BackendHistory`. Its `z` is each attempt's end
    and `output.schema.StepHistory`'s is its start, so the two no longer share a name.
  - `RUN_SUMMARY_ANNOTATION` moves from `driver.summary` to `output.schema`, beside the
    other annotation names. The stored name is still `"run_summary"`.
- `DataPulse.from_time` is replaced by the class `TimeDataPulse`: write
  `TimeDataPulse(t=t, field=field, energy=energy)`. It holds the time record and has no
  `omega`, `spectrum` or `spectral_phase`. The method kept the spectrum at the record's
  own bin spacing and splined its magnitude and phase, which cannot follow the fringes
  of pulses far apart: pulses at 100 and 600 fs in a record of 800 fs, of relative
  amplitude −0.9, came out 85 % wrong with no warning. The class evaluates the record's
  transform exactly at the grid's frequencies, 3e-12 from the direct sum on that case,
  less its mean: a baseline is an offset of the instrument, not part of the field.
- `get_spectrum` has no `noise_floor` keyword. Its annotation admitted `"subtract"`,
  which it always refused: an amplitude has no part that is the floor's power, so only
  `get_spectral_density` subtracts it. Drop `noise_floor="keep"`, the default, from a
  call that passes it; the result is unchanged. A call that still passes it raises
  `TypeError` (#96).
- `solprop.optics.Material` refuses a `temperature` for a glass or a mixture with
  `MaterialError`, where 0.1.0 ignored it: drop the argument there. The `temperature` of
  `Material`, `ModeCollection.slab` and `ModeCollection.vincetti` now defaults to
  `None`, which a built-in gas reads as room temperature
  (`solprop.constants.ROOM_TEMPERATURE`, 293.15 K) as before; given a `mixture=`, `slab`
  and `vincetti` now take the mixture's own temperature, where 0.1.0 refused the mixture
  beside its room-temperature default
  ([#158](https://github.com/LupoLab/solprop/issues/158)).
- `Mixture.fill`, `Mixture.air` and `Mixture.with_fill` refuse `floor=` beside a
  constant pressure or a supplied `ZProfile`, and `with_fill` refuses it without a new
  `pressure=`, with a `MaterialError`; the simulation constructors and
  `prop_capillary`/`prop_waveguide` refuse `pressure_floor=` in the same cases. 0.1.0
  accepted the floor there and ignored it. Drop the keyword where the pressure is
  constant or a profile; it applies only to end pressures or a stage table (#165).
- An envelope run's frame beside a third harmonic now follows one rule,
  `solprop.translate.responses.declares_harmonic_carrier`, which translation checks
  against the compiled harmonic carrier (#77). Two things change for users. First, an
  explicit `Frame(omega0, subtract_beta0=True)` beside a `KerrResponse` or
  `DispersiveKerrResponse` with `thg=True` on an envelope grid is now refused with a
  `TranslationError`, where it was accepted and drove the harmonic with a spurious phase
  mismatch: drop the explicit `frame=` to take the co-moving frame, which keeps `beta_0`
  there, or pass `Frame(omega0, subtract_beta0=False)`. Second, a Raman response's `thg`
  (the `RamanResponse` default is `True`), which an envelope grid ignores, no longer
  keeps `beta_0` beside `KerrResponse(thg=False)` when no frame is given: such a run,
  built from explicit responses or on a spatial route, now removes `beta_0`, so its
  stored envelope's carrier phase differs from before while intensities and spectral
  densities agree to the stepper's accuracy, and its run record now states the frame its
  operator used. The keyword interfaces, which set the Raman flag from `thg`, are
  unaffected.
- `mode_average=True` beside `tdse=` is refused by name when the simulation is built. It
  used to build, then fail at translation with a message that did not name it; such a run
  never ran, so drop one of the two.

### Added

#### The desktop front end

- The desktop front end, `solprop-gui`, **experimental** in this release: its window,
  its workflow and its files may change without notice before 1.0. It is a third
  distribution, built from `gui/` in this repository at solprop's own version; install
  it with `pip install "solprop[gui]"` and start it with `solprop-gui`. It sets up, runs
  and reads simulations without a script, and has a chapter of its own in the user
  guide:
  - Home offers a problem class for each common geometry (capillary, antiresonant fibre,
    radial filament, Cartesian and line beams, GNLSE, step-index and graded-index
    fibres), each starting from a few chosen examples, and a waveguide designer that
    needs no pulse; a recipe or results file of another route opens on a form built from
    solprop's parameter metadata;
  - each class has one setup sheet, checked as it is edited, with the numbers solprop
    computes for it and a preview plot; a number such as the soliton order can be typed
    as a target, which writes the recipe's `[pins]`;
  - runs go in separate processes with live views, stop and resume, or in a service
    process that keeps them going when the window closes;
  - results are read on pages of the class, all at one plane, with plot tools (zoom,
    readouts, widths at half maximum, peaks) and publication figures drawn by
    `solprop.plotting`;
  - scans run a setup over a grid of its parameters, record bands, spectra and pulses,
    and open any point as a setup;
  - every setup saves as a recipe or shows as a Python script.

#### Recipes and the command line

- Recipes ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)):
  `solprop.recipe.Recipe` reads a TOML description of a run, with quantities in their
  units (`"125 um"`, `"1.2 bar"`), checks it (`check()`, each issue at its parameter
  path), builds and runs it, saves it and exports it as the Python script it stands for
  (`to_python()`). Example recipes ship with the package (`Recipe.examples()`). Errors
  and warnings that concern a value carry its `parameter`; writing TOML takes the
  `tomli-w` dependency. A run built from a recipe records it in a top-level `recipe`
  group of its output, as written and as resolved (the recipe that rebuilds the run).
  See [recipes and the command line](https://solprop.readthedocs.io/en/latest/guide/setup/recipes.html).
- Recipes on every route. The `gnlse` and `capillary` recipes came first; `radial`,
  `free` and `planar` recipes write the grids, medium, responses, beam and settings as
  tables of registered types (`RadialGrid`, `FreeGrid`, `LineGrid`, `PlanarGrid`,
  `SpatialMedium`, `KerrResponse` with `gamma3`, `EnvelopePlasmaResponse`,
  `PlasmaResponse`, `OptmatRate`, `SpatialPulse`, `PlanarPulse`, `SimulationSettings`,
  `PIController`, `default_spec`, `SpatialAbsorber`), the absorber under `settings`;
  `Recipe.preview_input()` gives a free-space beam's power, energy and fluence at the
  launch plane, and `Recipe.modes()` the planar route's slab channels. A GNLSE recipe can
  hold a step-index fibre: `StepIndexMode`, with `glass` or an index for its core and
  cladding, is its `fibre`, and `gamma = "none"` spells the `None` that derives the
  coefficient from `n2` and the fibre's area (#472). A waveguide recipe can hold a
  graded-index fibre: `ModalGuide.on_grid` builds a guide from a provider with its knots
  and node rule taken from the time grid, and `GradedIndexProvider`,
  `PowerLawPermittivity`, `IndexMaterial` and `MaterialKerr` are registered (#473).
- The `solprop` command (also `python -m solprop`) runs, checks and exports recipes, and
  forwards `solprop scan`. `Simulation.run(stop=...)` takes a flag that stops the run,
  writes its state and returns, and `solprop run` stops cleanly on Ctrl-C.
  `solprop resume` and `solprop.recipe.resume_file` continue a run from its file alone,
  and `solprop.output.resumability` says whether a file can continue, from where and
  with which `resume=`, whether it can rebuild itself, and, on request, whether the
  rebuilt run's hash matches.
- Scans from recipes: a recipe's `[scan]` table gives `Recipe.scan()`, ready for
  `run_scan`. `run_scan(stop=)` takes the same flag as a run: Local and Pool stop the
  point in flight, Pool through a message on the worker pipe, Batch between groups; a
  stopped point is not committed and a rerun computes it; Queue and Slurm refuse it.
  `run_scan(progress=)` accepts a `solprop.log.ScanReporter` (and a
  `WatchingScanReporter` for each point's run, relayed from Pool workers).
- Design solvers for any guide ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)):
  `solprop.tools.solve_parameter` finds every value of a guide or fill parameter
  (a radius, a pressure, a core size, a wall thickness) that meets a target, `ZDW`,
  `RDW` or `SolitonOrder`, discarding sign changes through a wall resonance, and
  returns them as `DesignRoots` with the attainable range when there is none.
  `tools.radius_zdw` inverts `zdw` in the radius; `tools.soliton_peak_power` and
  `tools.pulse_energy` give the pulse for a soliton order. A recipe's `[pins]` table
  sets a value from such a target (`Recipe.resolve()`, `Recipe.pin_roots()`), solved at
  each build and each scan point and recorded with the run. `solprop.grid.suggest`
  proposes `wavelength_limits` and `trange` from the input pulses, as a starting point.
- Previews before a run: `Recipe.modes()` and `Recipe.preview_input()` show the mode set
  and the launched pulse. `Recipe.modes()` gives each mode's `beta`, `beta2`, effective
  index, group delay (`beta1`), third-order dispersion (`beta3`), nonlinear coefficient
  (`gamma`), loss, effective area and zero-dispersion wavelength over the band; the new
  `solprop.modes.base.beta3` differences the analytic `beta1` twice. It also gives each
  mode's phase matching to a soliton at the pump in the launch mode: the
  `phase_mismatch` of a dispersive wave in that mode over the band and its phase-matched
  wavelengths `rdw` (intermodal for every mode but the first), with the soliton's
  nonlinear wavenumber where the recipe's pulse and fill give one. The mode set says
  which (`pump`, `nonlinear`). The GNLSE route has a mode preview too: its one specified
  mode, phase-matched with the recipe's `gamma`.
- Parameter metadata for front ends: `solprop.recipe.parameters` gives each route
  keyword's forms, units, display unit, default rule and manual section, and
  `solprop.recipe.metadata.parameters` takes a registered type's name
  (`"GaussianPulse"`, `"PressureGradient.two_point"`, `"RadialGrid"` and the rest) and
  gives its keywords the same metadata: forms with units and ranges, a level, a one-line
  summary, a display unit, a manual anchor and the signature's default.
  `solprop.recipe.metadata.per_unit_width(route, values)` says when a recipe's pulse
  energy and peak power are per metre of y (a waveguide recipe of slab modes), and the
  waveguide route's energy and peak-power summaries say so, so a front end shows J/m and
  W/m there.
- Capabilities: `solprop.capabilities.route` lists a route's keywords, their values and
  the rules that couple them, each refusal naming its rule (`SolpropError.rule`);
  `solprop.capabilities.species` what a gas admits: its ionisation models with their
  status, its Raman response, and its data ranges; and `solprop.capabilities.output`
  what a results file can give: its basis, which quantities are physical fluxes, which
  planes keep fields, which operations its readers refuse and why.
- `Simulation.estimate()` estimates a run's memory, its output size and the cost of one
  right-hand-side evaluation; it is an estimate only, and nothing is admitted or refused
  on it. `Simulation.output_bytes()` is its part that times nothing: the size of a run's
  output without constructing a backend, for a front end to show as the setup changes.

#### Example recipes and tutorials

Every route now has a recipe a front end can start from, each mirroring a tutorial built
with the plain `prop_*` interface:

- `Recipe.example("hollow_rectangular")`: an argon-filled hollow silica channel in its
  lowest x-polarised mode, on the waveguide route.
- `Recipe.example("radial_kerr")`: a collimated beam in argon under the Kerr effect on a
  Hankel grid.
- `Recipe.example("filament")`: a filament in argon with plasma, with a tutorial that
  reads one.
- `Recipe.example("line_beam")`: a line beam on a `LineGrid` under the Kerr effect
  (#476).
- `Recipe.example("crossing_beams")`: two beams crossing in argon whose interference
  fringes the Kerr effect focuses, from which the desktop front end's Cartesian class
  starts; a one-beam Cartesian recipe still opens, on a one-beam sheet.
- `Recipe.example("step_index_fibre")`, the strand of the step-index tutorial (#472), and
  `Recipe.example("graded_index_fibre")`, with a new tutorial (#473).
- `Recipe.example("fibre_supercontinuum")`: the canonical photonic-crystal-fibre
  supercontinuum of Dudley, Genty and Coen (Rev. Mod. Phys. 78, 1135 (2006), Fig. 3),
  also in the supercontinuum gallery, which the desktop front end lists for its GNLSE
  class.
- The Cartesian and planar tutorials and the planar compressor ship as examples.

#### Pulses and launches

- `solprop.pulses.load_spectrum` reads a measured spectrum from a text file (frequency
  or wavelength, power density, and an optional phase) into a `DataPulse`. A wrapped
  phase column is unwrapped only where the power is above a noise floor, a phase in
  the FFT sign convention of a Luna.jl file is read with `phase_convention="fft"`, and
  a wrapped column sampled too coarsely to unwrap is warned on
  ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)).
- `prop_capillary`, `prop_waveguide` and their `Simulation` constructors take `launch=`,
  which puts each pulse into chosen modes with a share and an optional phase:
  `launch={"HE1,1": 0.9, "HE1,2": {"share": 0.1, "phase": 1.57}}`, or a list with one
  mapping per pulse of `pulses=`. Modes are named as `channel=` names them; the shares
  are energy fractions, and on a guide that declares a change of basis a coordinate
  split scaled to the pulse's energy under the flux metric. The run records the launch
  as `configuration["modal_launch"]`, which a resume compares; a run without one
  records nothing
  ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)).
- `SpatialPulse` takes `displacement=(d_x, d_y)` and `tilt=(theta_x, theta_y)`, as
  `coupling.Alignment` defines them, applied at the launch plane on a Cartesian grid
  with the energy set after them; a tilt the grid's band, window or time window cannot
  carry is refused, and so is either on a radial grid, a y component on a line grid, a
  `PlanarPulse` and a guide's incident beam. A record without them reads as an unmoved
  beam
  ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)).
- `pulses.SpatialPulseSum` launches several beams together on the free-space routes, the
  sum of its `SpatialPulse` members' fields, each built and scaled as it would be alone:
  two beams crossing at an angle, a pump beside a probe. Recipes hold it
  (`type = "SpatialPulseSum"`, `beams = [...]`), so a crossing-beam run is a recipe.
- `Simulation.with_input` and `SimulationRunner.run` take `input_identity=`, a string the
  run record holds with the array's dtype and shape (`solprop.output.StatedInput`) in
  place of the SHA-256 of an array input's bytes. Without it, every hash is unchanged.
  The identity is not checked: two different arrays stated alike have one configuration
  hash, so a wrong identity passes a resume and a scan's cache. A derived run no longer
  digests its parent's input to check its `configuration=` keys either (#449).

#### Routes and interfaces

- `prop_waveguide` and `Simulation.from_waveguide` take `interpolate_planes` and
  `step_planes`, and `prop_capillary`, `prop_waveguide` and their constructors take
  `polarisation_taper`, as `prop_gnlse` does; `step_planes` defaults to `None` on
  every interface that takes it, which on `prop_gnlse` behaves as its `0` did. The
  guide lists the deliberate differences between the routes' keywords
  ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)).
- `Simulation.from_radial`, `from_free` and `from_planar` take `absorber` and
  `absorber_scheme`, as `prop_radial`, `prop_free` and `prop_planar` and the modal
  `from_*` do.
- One table of the route defaults, `solprop.driver.defaults.ROUTE_DEFAULTS`, with the
  functions that resolve each default (`default_rtol`, `default_integrator`,
  `default_controller`, `default_scheme`, `resolve_noise`); the run record's new
  `defaults` entry (`DefaultsResolution`), the start-of-run summary and the settings'
  display say which of the integrator, `rtol` and the collar scheme took the route's
  default, and the summary names a Multistep carrier frame
  ([#87](https://github.com/LupoLab/solprop/issues/87)).
- `precision="single"` now admits the Adams integrator, `Multistep()` in its exponential
  formulation, on `numpy_ref` and on `mlx`, where its history stays on the GPU and each
  trial's flow, moments and compensation are formed there in double-float arithmetic.
  Binary32 Adams agrees with binary64 Adams to the method's own truncation on eight
  survival cases (the guide's single-precision page has the record), and on the
  capillaries it is closer to the binary64 result than binary32 Runge–Kutta is. A
  single-precision Adams history now resumes in binary32. A single-precision run now
  takes each interface's own default integrator, as a binary64 one does: `Multistep()`
  on `prop_capillary`, `prop_radial` and `prop_planar`, where it was `RungeKutta()`.
  (#342)
- `SOLPROP_REQUIRE_DEVICE` (`cuda`, `hip` or `any`) makes a run refuse before allocating
  anything unless it would execute on a native device build that sees a device, so a job
  doesn't fall onto a CPU wheel that a new release installed over its source build.
- `solprop.backends` exports `thread_budget` and `ThreadBudget`: how many threads a
  process may run at once, and the limit that sets it, for a front end to show.

#### Output, observers and handlers

- A reporter that implements `observe_plane` (`solprop.log.PlaneObserver`) is shown each
  saved plane as a lazy view, whose spectrum, field, power and spectral density come
  from the processing functions, fetched from a device only on request; a failing
  observer is detached and the run completes. On a spatial run, `PlaneView.stored()`
  returns the plane's stored spectrum (what the `solprop.spatial` functionals take with
  the run's `SpatialFlux`) and `PlaneView.field_plane()` the plane as
  `processing.field_plane` reads it, whose `spectral_energy` is the beam's energy per
  bin.
- `solprop.output.RetainingOutput`, the documented protocol for a handler of your own
  that takes field-free planes: its one method, `retain(retention, field_shape)`, is
  called once per run, after `open` and before any `save`, with the rule resolved against
  the run's planes. A run that keeps fewer than every field used to look the method up
  without declaring it anywhere (#450).
- `solprop.output.CallbackOutput`, an output handler built from a callable: it gives each
  saved plane to the callable and keeps nothing, so a reducing scan on
  `SimulationRunner` is a few lines and costs no volume beyond the plane's own.
  `check_cache`, `resume` and `resumability` now refuse any output handler by name
  (#499).
- `processing.save_field_plane` and `processing.load_field_plane` keep a `FieldPlane` in
  an HDF5 file with its own schema version, so a saved plane can launch another run
  later. A recipe can't yet name one as its launch (#456).

#### Free-space readers and apertures

- Readers of a radial beam from its file, in `solprop.processing`: `on_axis_spectra`,
  `radial_spectrum`, `far_field`, `on_axis_power` and `on_axis_spectrogram`.
  `solprop.tools.beam_params` gives a Gaussian beam's Rayleigh range, critical power,
  power ratio and Marburger collapse distance, and `beam_through_lens` the waist a lens
  makes.
- `processing.spatial_grid_of` returns a saved spatial run's transverse grid from its
  record, and `processing.fluence_map` lays one saved plane's fluence out on its axes, in
  any bulk medium and for a file written by a script. `node_fluence` now also reads
  radial and Cartesian runs without a gas mixture. A hybrid planar file is refused by
  name (#455).
- `solprop.spatial.spatial_energy_spectrum(spectrum, flux)` splits the energy that
  `spatial_energy` measures into frequency bins, in J (J/m on a line), on radial,
  Cartesian, line and planar planes, and the bins sum to the total. It is not the density
  `beam_spectrum`. The coupling elements used it privately, and their reports do not
  change. On a spatial file, the spectral readers' refusal now names it on every route,
  planar included ([#184](https://github.com/LupoLab/solprop/issues/184)).
- `solprop.spatial.spatial_power`, `spatial_intensity` and `on_axis_trace`, and
  `solprop.processing.on_axis_power`, take `oversampling=`, resampling the time axis by
  zero-padding the spectrum as `get_power` does, so a beam's pulse is drawn smoothly and
  its peak found between samples. `solprop.processing.time_axis(grid, oversampling)`
  returns the times they lie on.
- An `ApertureSet` row may be an ellipse, a rectangle or a half-plane (`shapes`,
  `extents`), with its own edge profile (`profiles`), so a far-field statistic can
  express an instrument's mask: a knife edge at a fixed position in a far-field plane,
  whose wavevector moves with frequency, no longer needs the field on the host. A
  half-plane is half-open and tiles the columns with its complement. Line grids take
  half-planes bounded in x and rectangles, and the offline `Aperture` takes the same
  shapes. Unset, the new fields leave every existing result as it was (#446, part of
  #186).
- `absorber(grid, walkoff=...)` sizes the temporal collar from a group-delay drift over
  the grid's own length, counted over both edges, as `warn_walkoff` checks, and leaves
  the spectral rate alone (#226). The [boundaries guide](docs/guide/numerics/boundaries.md)
  now also says that `boundary_N` gives a slice its own strength and that
  `boundary_length=parent.reference_length` keeps the parent's, and gives a recipe for a
  [spectral walk-off absorber](docs/guide/numerics/boundaries.md#a-spectral-walk-off-absorber)
  through `custom_absorber`, for runs whose walk-off can be predicted, with its geometry,
  its strength, its limits and a per-band loss check (#226).
- The [Cartesian chapter](docs/guide/free-space/cartesian.md) gives a reading of the risk
  of transverse aliasing, the beam's energy outside `|k| <= k_max / 2` from
  `region_spectrum` and `beam_spectrum`, tested through `FreeGrid` runs, and states what
  the proxy leaves out (#216).

#### Plotting

- Cross-section drawings in the library
  ([SEP 41](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0041-recipes-and-front-ends.html)):
  `solprop.plotting.geometry.outline` gives the outline of a capillary, antiresonant,
  step-index, graded-index, rectangular or planar guide, or of a radial, Cartesian, line
  or planar grid, as shapes in metres; `draw(ax, guide, palette=...)` paints it in
  Matplotlib, and `thumbnail(route, representation, size)` makes a small figure of a
  route for a front end's picker. Colours come from an explicit `Palette` (`LIGHT`,
  `DARK`). The user guide's cross-section figures draw their outlines from it.
- `solprop.processing.get_spectrogram` returns a plane's spectrogram as data (gate
  positions, spectral axis and map, per unit of the axis asked for, with a `bandpass=`
  window), the map `plotting.spectrogram` draws, so a front end can draw it
  interactively.
- `solprop.sol` exports `wavelength_to_axis`, which the plots' documentation sends the
  reader to.

#### Distribution and benchmarks

- `solprop-native` wheels for Windows on Arm (`win-arm64`), and an x86-64-v2 build
  beside the x86-64-v3 one in the Linux and Windows x86-64 wheels, chosen at import, so
  processors without AVX2 now run the native backend. The v2 build runs about 20 to 25 %
  longer by geometric mean. `SOLPROP_NATIVE_CPU_VARIANT=x86-64-v2` asks for it.
- `solprop-native` wheels use CPython's stable ABI, so one wheel serves Python 3.14 and
  later. solprop itself is not yet supported on 3.15
  ([#319](https://github.com/LupoLab/solprop/issues/319)).
- A published-install check after each PyPI release, on every wheel platform: fresh
  CPython 3.14 environments resolve `solprop` with pip and verify its version,
  dependencies and default backend. The same check can be run by hand for a named
  release, with an optional run of its tagged README example and retained logs.
- The installation page now explains how to install a development version from CI's
  wheels with no compiler, and `poe fetch-wheel` now finds CI's Windows on Arm wheel.
- The first performance-history rows (`benchmarks/history/`), recorded at 0.1.0 on six
  CPU and three GPU hardware sets, with numba runs beside the native ones on three of
  them; `docs/dev/benchmarking.md` lists how each is run. The history takes any backend:
  each row names it, and rows compare only within one backend; a run the history
  recorded no longer makes the next run in that checkout dirty. The benchmark harness
  names an AMD GPU from its PCI IDs where ROCm reports only "AMD Radeon Graphics".

### Changed

#### MLX and single precision

- The `mlx` backend compiles each stage on the GPU with `mx.compile`, so a stage's
  element-wise passes run as a few fused kernels and its graph is not rebuilt in Python,
  and forms a stage's propagator and its inverse in one kernel. A trial takes 1.5 to 2.6
  times less time on the capillary, radial and Cartesian routes, measured on an Apple M5
  Max, and the results are bitwise unchanged. `SOLPROP_MLX_FUSED=0` evaluates operation
  by operation. (#343)
- The `mlx` backend's Runge–Kutta march now reuses the last stage of an accepted DOPRI5
  step as the first stage of the next, as `native` does, so a step takes six evaluations
  of the nonlinear source instead of seven where the temporal collar is applied in the
  stages. The results are bitwise the same. (#343)
- On `mlx`, the transverse edge is read on the GPU where no collar is requested, rather
  than by copying the whole state to the host at every read, and a Cartesian run's
  operator is built in one pass and no longer forms an unused prefactor. On an Apple M5
  Max the 384 x 384 x 750 Kerr beam takes 0.82 of `native`'s time, the 192 x 192 one
  0.40. (#343)
- On `mlx`, a saved plane's field-grid plasma statistics (the ionised fractions and peak
  rates at every node) are now evaluated on the GPU with the propagation's own rate, scan
  and population kernels, as the envelope grid's already were, rather than on the host
  in binary64 from every node's field. On a 128 x 128 Cartesian argon filament the
  statistics of a plane took about 3 s and now take under 0.1 s; the values agree with
  the reference to about 1e-6 of their largest. (#343)
- A single-precision `prop_capillary`, `prop_gnlse` or `prop_waveguide` run on `mlx` (or
  `backend="auto"`) now builds its grid for MLX's transforms: past 4096 samples a power
  of two, which MLX's GPU transforms run six to twelve times faster than the 2-3-5-smooth
  sizes FFTW prefers. The helium RDW capillary's grid is 16384 fine samples instead of
  10240, and a trial took a quarter of the time. `FieldGrid.from_limits` and
  `EnvelopeGrid.from_limits` take `transforms="mlx"` for a grid of one's own, and
  `fft_cost` takes the library. Binary64 runs and single-precision runs on `numpy_ref`
  are unchanged. (#85)
- A single-precision free-space run (`prop_free`, Cartesian and line) now takes free
  space's binary64 default tolerance, `rtol=1e-4`, instead of `1e-5`: binary32 at `1e-4`
  was measured as close to a tight binary64 reference as binary64 at `1e-4` on a radial
  filament, a radial beam with its transverse collar and a Cartesian filament. The
  Cartesian filament takes 48 steps instead of 61. (#85)

#### GPU and native performance

- A `Local` scan on a CUDA or HIP build keeps its backend between points that differ
  only in their input, as `Simulation.with_input` builds them, so such a scan no longer
  allocates and plans a device's backend for each point. A point's stored results and
  record are unchanged to rounding.
- A lockstep `Batch` scan no longer moves a member's field to the host at a plane the
  scan doesn't read: `BatchBackend.observe_members` takes `field=`, as `observe` does,
  and a scan asks for what a lone point of the same scan keeps. On an NVIDIA H200
  `Batch(8)` spent 22 % of a plane interval observing at 8 steps a plane, and on an AMD
  Instinct MI325X 13 to 16 %.
- A device run's `on_axis_*` plasma statistics no longer copy the state to the host at
  each save plane: only the axis row's spectrum crosses, and the host scans it as before
  (#359). `StepDriver` gains the `_axis_row_populations` hook, and `_axis_statistics`
  takes `None` for a backend that reads the row itself.
- A device's low-storage step costs less beyond its transforms. On a CUDA or HIP build, a
  free-space envelope's split-step temporal collar transforms every row of the state at
  once, in place, rather than a node chunk at a time, whatever `node_budget` says, and
  holds no coarse buffers; a flow's lossless bin takes no exponential; and a device's
  first input crosses to the scratch through tiles. At 768 x 768 x 600 a step's collar
  takes 14 kernels and 10 transforms where it took 119 and 220. On an NVIDIA A40 the step
  fell from 2.66 s to about 2.56 s, and on an AMD Instinct MI300X a point takes 1.6 %
  less time. Results are unchanged to rounding
  ([#361](https://github.com/LupoLab/solprop/issues/361)).
- On a CUDA or HIP build, a Kerr-only free-space envelope's stage holds its transverse
  scratch in the state's order, so its rank-3 transform keeps frequency contiguous and
  its input and fold no longer cross layouts. A step on an NVIDIA H200 takes 204 ms at
  768 x 768 x 600 where it took 225 ms, and 36 ms at 384 x 384 x 600 where it took 51 ms,
  with 1.3 GiB less memory there. On an AMD Instinct MI325X it takes 199 ms where it took
  211 at 768 x 768 x 600, and 52 ms where it took 57 at 384 x 384 x 600. Results are
  unchanged to rounding.
- On a CUDA or HIP build, a free-space run's transforms of the transverse scratch share
  one work area instead of holding one each. At 768 x 768 x 600 a run holds about 16 GiB
  less, 22.6 GiB against 38.5 GiB on an NVIDIA A40, and 23.4 GiB on an AMD Instinct
  MI300X. Results are unchanged.
- On a GPU build, a large state now crosses to and from the host through a ring of
  pinned chunks into huge-page memory, rather than taking its page faults one thread at a
  time inside the driver's copy. In a bare copy of a 5.27 GiB state that scheme reached
  33 GiB/s against 1.8 GiB/s on an NVIDIA H200, and 50 GiB/s against 7.9 GiB/s on an AMD
  Instinct MI325X. A state under 256 MiB takes no ring, and the ring is listed in
  `memory_ledger` as `host_crossing_ring`. A saved plane's finiteness check and its
  conversion to the stored convention now run on the process's threads, and
  `MemoryOutput` converts straight into its own block rather than copying. Stored fields
  are unchanged to the bit.
- On the native backend, `beam_radius` and `fwhm_r` no longer take a time transform on a
  Cartesian, line or planar grid, and on a Cartesian plane they reduce where the field
  lives: only a few moments and one value per ring reach the host, not a copy of the
  fluence at every node. `on_axis_intensity` on a Cartesian or line grid now reads the
  far-field pass's on-axis sums instead of walking every column per bin. They agree with
  `numpy_ref` to rounding. A fluence negative by a little more than its rounding budget
  may now be read as rounding, because the budget's scale is a bound on the reference's,
  which it reaches when the closure is proportional to the field. `solprop.spatial`
  gains `FluenceRings`, `fluence_rings` and `ring_fluence_widths`, the parts of
  `cartesian_fluence_widths` that a backend shares.
- A low-storage run no longer holds a state volume at its last field-free plane, so
  `memory_ledger()` and `estimate()` are one volume smaller, 5.27 GiB at 768² × 600. A
  step abandoned in flight after such a plane records `failed` with no terminal
  checkpoint and says plainly when the file cannot be resumed; a stop between steps
  checkpoints as ever (ADR-067, #345, #500).
- Building a large free-space launch is faster, with every bit of the launch as it was:
  its passes over the beam's elements run in threads, its admittance is formed once, and
  the transverse residual no longer takes a Python `Fraction` per node. Digesting an
  array not held in its own order no longer builds a whole volume, so a broadcast input
  costs a few megabytes. (#493, #347)

#### Records, defaults and interfaces

- A run record's numeric arrays of more than 4096 elements are now hashed by their bytes
  and stored beside its JSON: `/meta/json` holds a reference, and `/meta/arrays/<name>`
  the array, or `/record_arrays/<name>` in a scan, named by its digest, dtype and shape,
  shared between points. The text of a 512² Cartesian run's record falls from about
  23 MB to under 1 MB, and the first configuration hash from seconds to milliseconds,
  with every array still readable. `RunMetadata.from_json` takes the arrays as
  `arrays=`, and `RunMetadata.record_arrays` returns them. A noise realisation moves too
  ([#309](https://github.com/LupoLab/solprop/issues/309)).
- The output schema is bumped at most once between releases, not for each change that
  moves it (ADR-063).
- A run asking for the `zdw` statistic is no longer stopped when its band holds no
  zero-dispersion point: the plane records `nan` with a warning. A direct `stats.zdw`
  call still raises, now `NoZeroDispersionError`, a `ModeError`.
- `capillary_params`, `soliton_params` and the other design tools that report a
  zero-dispersion wavelength find the zero nearest the pump by refining the sign changes
  outwards from it with array samples, not all of them, and no longer raise
  `ValueError` when the search meets a gas resonance (xenon in a 14 um capillary at
  15 bar). The array refinement makes them about 2.5 times faster, and on an
  antiresonant fibre, whose resonances give dozens of changes, the search from the pump
  takes about 0.17 s where it took about 3 s; the answer is the same.
  Zeros clear of the gas resonances are unchanged within 1e-8; a sign change reported
  beside a resonance, which only a pump in the vacuum ultraviolet selects, can move
  within the search's 3.6e13 rad/s sample interval.
- `solprop.recipe` loads its names on first use (PEP 562), as `solprop` itself does, so
  `solprop.recipe.quantity`, `paths`, `document` and `errors` import without the driver,
  the backends, h5py or SciPy: about 40 ms rather than half a second, for a front end
  that parses quantities in a process that never runs a simulation.
- `solprop.backends.protocol.Backend` now declares what the run layer asks of a backend,
  which it used to look up by name: `set_stop`, `attach_monitor`, `in_flight`,
  `march_snapshot`, `march_snapshot_nbytes`, `march_export_refusal`, `restore_march`,
  `edge_memory`, `timed_step`, `closed` and `reads_transverse_edge`. `StepDriver` defines
  all of them, so a backend built on it changes nothing.
- `solprop.plotting` draws its maps (`prop_2d` and `spectrogram`) rasterized, so a figure
  saved as PDF or SVG keeps its axes and text as vectors and stays small.
- The `antiresonant_fibre` example and the antiresonant tutorial's propagation show
  resonant dispersive-wave emission: 1 µJ, 30 fs at 800 nm through 20 cm of fibre in
  5 bar of argon, field resolved over 150–1600 nm with photoionisation and 201 saved
  planes, in place of 2 µJ over 5 cm at 1 bar on an envelope grid. They now record the
  default statistics, not energy alone.

#### Project, documentation and tooling

- The package classifier is now Development Status 4 - Beta. The interfaces may still
  change between minor releases.
- The README, `CITATION.cff` and the citing page name solprop's Zenodo DOI,
  [10.5281/zenodo.23854248](https://doi.org/10.5281/zenodo.23854248), which covers
  every release.
- Planned work has moved to [GitHub issues](https://github.com/LupoLab/solprop/issues),
  labelled by type, priority and area: the roadmap now lists only SEPs (ADR-062). Issue
  templates and `CONTRIBUTING.md` say what an issue or a pull request should hold and
  what to check before opening one.
- Notable changes are recorded in separate release-note fragments and collected into one
  rolling draft release PR after merges. Implementation PRs no longer edit the shared
  changelog, avoiding conflicts between unrelated changes. Release tags and publication
  remain manual.
- A plain `uv sync` in a checkout now compiles the native backend, and
  `uv sync --extra native` is gone. Python and documentation contributors with no
  compiler can use `uv run --no-project tools/setup_light.py`, and `poe fetch-wheel`
  installs the wheel CI built, refusing one that doesn't fit the tree.
- Continuous integration now compiles the CUDA and HIP targets on a change to `native/`,
  with no device, so a change that breaks a GPU build is found before it merges. A build
  from source no longer stops on a compiler's warning:
  `SOLPROP_NATIVE_WARNINGS_AS_ERRORS` now follows `BUILD_TESTING`, so it stays on for the
  presets and is off for `pip` and `uv` builds, while continuous integration keeps it on
  for its wheels. (#151)
- Corrected the mixtures guide on spatial fluence and on who refuses a z table, three
  worked examples' prose, how `ReferenceState.temperature` reads where a source declares
  none, and what the antiresonant guide claims for the dispersion width's fit to the
  finite-element solutions.

### Fixed

#### Noise, scans and resume

- A scan derived from one cached simulation, the scans guide's own recipe, now gives
  every point its own noise seed. `with_input(noise="keep")` and `with_mixture` keep the
  ability to draw unseeded noise again, and the draw replaces only the initial noise of
  the shared translation. Before, such points shared the base's draw, depended on the
  worker under `Pool`, could not be rerun in a new process and failed under `base_seed`.
  Resuming a derived, unseeded run now takes the file's seed, as a constructor's does,
  instead of being refused.
- One object returned by a scan builder for every point no longer collapses a noise
  ensemble under `Batch`, where every member ran on the last point's noise while the
  file recorded each point's own seed. The draw now goes on a copy, and the builder's
  object is left as it is. A file written by such a scan holds the wrong noise for its
  recorded seeds and a rerun does not detect it: run those points again.
- Resuming a file that holds only its launch plane, from an `input`-noise run stopped
  before its first saved plane, no longer adds the noise draw to the stored field a
  second time, which had doubled the noise amplitude for the rest of the run on every
  backend. `set_initial` takes `resume=True` for a stored field.
- A scan file whose points' run records declare another output schema, a 0.1.0 scan
  among them, is refused by name, in the words a run file is refused with, when it is
  read (`ScanOutput`, `read_point`) or resumed or collected (`run_scan`, a queue's
  collector). Before, only the scan container's own version was checked, and that
  stayed 1, so such a scan was read as if its layout were this release's.
- A recipe with a `[scan]` no longer drops the `[execution]` keys `run_scan` does not
  take: `resume_history` now reaches each point's settings, as the scan file's recorded
  recipe says it does.
- `stop=` now stops a native `Multistep()` run during its march instead of at its next
  save plane: within a native call's bound, 256 attempts or 0.1 s, and the attempt in
  flight. With one plane the flag was ignored and the run finished as `complete`.
  `solprop run` and `solprop resume`, whose first SIGTERM or Ctrl-C reaches a run
  through that flag, stop the same way; they used to report a stop that the file showed
  had not happened (#519). A signal that comes too late to stop the run now exits with
  status zero and says the file is complete (#521).
- On Windows on Arm, `Pool` scans no longer fail or hang in NumPy's linear solves.
  Workers now limit BLAS through `threadpoolctl` instead of thread variables, which crash
  or hang NumPy's OpenBLAS on that platform at small widths, and `import solprop` warns
  if such a value is set.

#### Boundaries and the monitor

- Under `absorber_scheme="split_step"`, the default on the spatial routes, the boundary
  monitor judges the time window's guards on the power each step's split finds, halfway
  through its factor, instead of the state it leaves, whose guards it had damped by about
  `e^-10`: a pulse walking out of the window is now stopped where the in-stage scheme
  stops it (#127). Each reading reports that power's energy and its guards' and
  collar's energies (`solprop.monitor.BoundaryEnergies.found`) and the state's energy
  per walk-off bucket (`BoundaryEnergies.occupancy`). Steps are also held to the split
  collar's crossing limit, half the collar's width over the walk-off of the fastest light
  holding more than `1e-6` of the energy, so that no step carries light through the
  collar and round the window undamped (issues #149 and #440). It is read from the launch
  at translation, which holds `dz_max` and refuses a longer fixed schedule, and again from
  the state at every check, which caps the steps after it and stops a fixed schedule it
  falls below. See [absorbing
  boundaries](docs/guide/numerics/boundaries.md#the-two-temporal-schemes).
- Under input noise, the boundary monitor subtracts at most one floor from the collar's
  removal: the floor is put in once at the launch and cannot lose more. It subtracted
  the floor's rate over the whole run, about 37 floors under the default collar, so a
  removal of a few 1e-3 read as zero, with no warning, and the split scheme's
  subtraction grew with the number of steps
  ([#126](https://github.com/LupoLab/solprop/issues/126)).
- A resumed run's monitor record continues from the latest execution that kept one: a
  second hard kill before the monitor started, or an execution with
  `boundary_check=None`, no longer restarts the totals, and the stretch run unwatched is
  listed under `unwatched`. The signed removal is kept and restored, so a stop and
  resume reads as the whole run; the quiet collar's state is restored; and a resume on a
  held row no longer counts the resumed state in the next row's guard maxima
  ([#128](https://github.com/LupoLab/solprop/issues/128)).
- `BoundaryCheck(quiet_collar=True)` no longer records the collar as left out on the
  Cartesian, line and planar routes, whose stages keep it, or caps their steps for
  nothing ([#94](https://github.com/LupoLab/solprop/issues/94)).
- A scan point keeps its closing boundary-monitor record, without the per-plane rows,
  under every executor (`read_point(...)["boundary_monitor"]`); a batched point no
  longer stores the launch's zeros; and a `BoundaryWarning` raised as an error fails its
  own batched point alone, as under `Local`
  ([#132](https://github.com/LupoLab/solprop/issues/132)).
- The default boundary check no longer brings a native run's state to the host. The
  transverse edge, a requested collar included, is read where the state lives and only
  numbers cross, so a field-free run on a device copies no state at all, and on the CPU
  a free-space Kerr beam's check costs less. A resumed run with a requested collar now
  restores its current, baselines and peaks, so it no longer judges against zero.
  ([SEP 44](docs/dev/seps/archive/sep-0044-native-edge-read.md), #385, #512)
- The default boundary check no longer does work that nothing uses. A saved plane the
  step that reached it already read takes no second read, which on a device was a copy
  of the whole state. A fixed schedule's reads of the transverse edge now follow its next
  step rather than `dz_init`, which read one step late with a short `dz_init`, past the
  spacing the reads promise, and at every step with a long one. The split collar's
  occupancy reduces per column, without an atomic per element, which took most of a
  monitor reading on a GPU. `energy` and `peak_intensity` no longer sum on one GPU
  thread, an Adams run over the snapshot limit takes no snapshot to learn its size, and a
  runner's points share the walk-off table rather than upload it each.
- A boundary monitor built for another grid is now refused by name on every backend,
  instead of a bare `ValueError` on `native` and an accident in `matmul` on `numpy_ref`.
  A direct call to `native`'s `stats` now checks every route for names the route doesn't
  define, and refuses a general flux statistic under a weight other than one rather than
  ignoring the weight.
- `warn_walkoff` suggested a `boundary_N` counted from the default strength when it was
  given a `boundary_length`; it now counts from the length in force. The guide and the
  docstring of `no_absorber` no longer say that an explicit absorber switches on a
  free-space run's real-space collar or that the collar reads the reference length
  (#226).
- Documented the limits of `FixedSchedule.from_history`. Every collar refusal a replayed
  history can reach (the temporal collar's stable step, and a requested spatial collar's
  stable step and its fixed-schedule crossing and damping bounds) now prints its limit
  rounded down and names the remedy: `dz_max` at or under the printed limit in the
  discovery run, run with the replay's integrator where the limit is a stability limit,
  or, for the temporal collar alone, `absorber_scheme="split_step"` in both runs. The
  docstring and the [fixed schedules guide](docs/guide/numerics/fixed-schedules.md) state
  these limits and the interpolated-plane and non-abutting-history ones, relate
  `from_history` to `discover`, and replace the claim that a `Multistep` replay makes its
  adaptive run's truncation errors: compare two replays of one schedule instead, and for
  a comparison of the arithmetic alone replay with fixed orders (`RungeKutta()` or
  `Multistep(variable_order=False)`), since a variable-order replay's order choice can
  turn on rounding (#82).

#### Copies and memory

- A `Simulation` built from an array input now copies the array at construction, so
  changing it before the run no longer changes the run, its construction-time checks or
  its recorded configuration (#434). `Simulation.initial` is a read-only view of that
  copy as the caller gave it, so an input of another dtype is recorded unchanged and
  older files resume. On free-space and spatial routes the translated problem holds the
  same copy; on a modal route it holds the converted launch beside it, two states. A
  `numpy.broadcast_to` input copies only the block it repeats.
- A translated problem no longer holds a full copy of a broadcast initial field (#5): a
  `numpy.broadcast_to` placeholder, of any numeric dtype, costs only the block it
  repeats. Every other initial field, noise realisation and batch parameter is now
  copied, so changing the caller's array after translation can no longer change the
  problem, and the caller's array is no longer made read-only; before, a contiguous
  complex128 field and a noise realisation were shared with the caller.
- The native backend no longer keeps the last host state it loaded alive after
  `set_initial`: a temporary, such as an input noise sum, a resumed plane or a delay
  scan's input, was one host field volume held until the next load, outside
  `memory_ledger()`. A resident Adams march no longer keeps a host copy of the state for
  the backend's life either; solprop's host engine, which reads one, lists it as
  `adams_host_state` (#435).
- A `SimulationRunner` no longer holds an input through its march. An array recorded by
  its digest is read in place rather than copied, and neither it, the launch converted
  from it nor the derived simulation outlives the load, so a delay scan that forms each
  input in the call holds it only until the backend has it (#447).
- A run hands its output each plane without copying the field. `SavedPlane` copied an
  observation's field, and again when the driver added the plane's statistics, so a
  handler that only reduces a plane paid two field-sized copies and a peak of two volumes
  per plane. `plane.field` is read-only and nothing changes it afterwards, so a handler
  can still keep it; a `SavedPlane` constructed by hand copies, as before (#448).
- A run of a guide that is the same along its length (a fixed fibre, a uniform fill)
  records its mode basis once instead of once for each saved plane, so a step-index fibre
  with 201 saved planes now starts at once rather than after about 95 s.
- A run into `MemoryOutput`, the default handler, no longer computes the configuration
  hash for a debug message that is switched off, which took seconds on a large grid.
- Guide and incident-plane construction, kernel translation and coupling now limit
  NumPy's and SciPy's controllable BLAS pools to one thread, or `SOLPROP_BLAS_THREADS`
  within the process budget, and restore their previous limits afterwards. This
  prevents the thousands of small graded-index eigensolves from occupying every CPU
  before a backend starts. Vendor thread defaults are overridden only during
  construction; the native backend's own BLAS keeps its fitted width.

#### Pulses, modes and design tools

- A tilted `SpatialPulse` travels towards +x (+y) for a positive `tilt`, as documented
  and as `coupling.Alignment` tilts a beam; it went the other way. Scripts that
  compensated by negating the tilt must remove the minus sign.
- `DataPulse.from_spectral_density` interpolates the phase it is given. It used to wrap
  the phase into the complex spectrum and unwrap it again, which cannot follow a change
  of more than pi between samples, so a strongly chirped pulse was built wrongly with no
  warning: 20 000 fs² of group-delay dispersion on a 0.4 nm axis at 800 nm came out
  13.5 % away from the converged spectrum, on a 1.6 nm axis 100 %. A `DataPulse` given a
  complex `spectrum` still unwraps it, and its docstring states that limit.
- A measured field in time puts a pulse where it sat in the record (`TimeDataPulse`,
  above). `DataPulse.from_time` unwrapped the phase, and a pulse more than half the
  record's length from `t = 0` changed its phase by more than pi per bin, so it was moved
  by one record length with no warning: at 500 fs in a record from 0 to 800 fs it came
  out at −300 fs.
- `StepIndexMode` builds fibres up to V = 20000 and refuses larger ones with a message
  that says so; it refused most fibres above V of about 26, a 100 µm core of numerical
  aperture 0.22 at 1030 nm among them, because its root scan lost close pairs of HE and
  EH roots, and from a cladding argument of about 700 the unscaled modified Bessel
  functions underflowed. It also refused every EH family from an azimuthal order of about
  40, where J_n underflowed at small u. The lost roots are now bracketed by each family's
  own factor of the characteristic product; the scan runs from the family's first cutoff
  to the one above the requested radial order, so a low-order mode costs the same at any
  V; and the modified Bessel ratios and exterior fields use the exponentially scaled
  functions, with a recurrence where those overflow at large order. Effective indices
  below V = 26 move by at most one rounding (#148).
- `tools.lambda_rdw_all` skips a sign change of the phase mismatch through a resonance of
  the guide, where the mismatch has a pole or is not a number, instead of raising a bare
  `ValueError` from SciPy on an antiresonant fibre's default search window (#538).
- A `soliton_order` pin on a GNLSE recipe reads the recipe's `gamma` or `n2`, where it
  failed with a bare `AttributeError` looking for the mode's gas fill.
- A recipe without a required keyword now raises `RecipeError`, naming the keyword in its
  `parameter`, from `build`, `resolve`, `modes`, `preview_input`, `pin_roots` and `run`,
  where the route's constructor raised a bare `TypeError` (#454).
- The `step_index_fibre` example and its tutorial run 5 cm on a 2 ps window with the
  boundary monitor on, saving 201 planes; over 15 cm on 1 ps its Raman-shifted solitons
  left the window unnoticed and the run lost 71 % of its energy (#544). The `rdw`,
  `spm_compressor` and `graded_index_fibre` examples and their gallery scripts save 201
  planes, for smoother plots along z.

#### Readers and plots

- On a radial, Cartesian, line or planar output, the processing readers that read each
  transverse column as a modal amplitude in sqrt(W) now refuse the file by name with an
  `OutputError` that names the spatial functional to use, instead of returning wrong
  numbers silently (a 3 nJ launch read back as 4.8e-15 J). They are `energy`,
  `peak_power`, `get_power`, `get_spectral_density`, `fwhm_t`, `fwhm_f`,
  `time_bandwidth`, `arrival_time`, `field_autocorrelation`,
  `intensity_autocorrelation` and `coherence_time`, the plots `prop_2d`, `time_1d`,
  `spec_1d` and `plot_energy`, and a plane observer's `power` and `spectral_density`.
  Use `solprop.spatial.spatial_energy`, `spatial_power` and their neighbours on a saved
  plane, or the run's `stats/energy`, which is unchanged; `get_spectrum`, `get_field` and
  `get_phase` still return the stored columns. `solprop.capabilities.output` gains
  `refused_readers`, which states these refusals with the readers' own text, while it
  still admits a spatial file's `energy` statistic. Modal files read as before (#147,
  #406).
- `get_spectrogram`, and so `plotting.spectrogram`, no longer alias on a narrow envelope
  grid. The real field is now oversampled until it resolves the carrier: an 800 nm pulse
  on a 600 nm to 1100 nm band used to appear near 2.9 µm.
- Bad input to the processing and plotting helpers is refused by name, with its units,
  instead of failing late or silently. `Passband` refuses other than two limits, or a
  width that is not positive, when built; it raised `IndexError` on use, or kept two of
  three limits. `wavelength_to_axis` refuses a wavelength limit that is not positive on a
  frequency axis, where zero raised `ZeroDivisionError`. `spectral_bands` and
  `band_coherence` refuse `n_bands` below one, and a band with nothing to average stays
  `nan` without a `RuntimeWarning`. `spectrogram` refuses a `spec_range` that misses the
  data, as `get_spectral_density` does, instead of drawing an empty view (#96).
- A gallery page re-run on its own keeps its `MaterialRangeWarning`: the docs build's
  import of `solprop.sol` silenced it for every script after.

#### Backends, single precision and platforms

- An adaptive `RungeKutta()` run on `mlx` is no longer held to short steps by binary32's
  exponent range: a trial whose collar or loss would overflow the interaction picture's
  inverse propagator (`|Re dPhi|` past 88.7) is combined in the lab frame by forward
  flows instead, where it was rejected and shrunk before. A radial Kerr beam whose first
  trial reaches 459 now takes its millimetre in one step, as binary64 does, against 14
  steps and 2 rejections. Runs within the range are unchanged. (#343)
- A single-precision run with the low-storage march or the exponential Adams method is
  no longer refused when a step's damping exceeds binary32's exponent range: those
  methods apply forward flows only, which underflow harmlessly in a damped bin, so only
  growth is limited. The Cartesian Kerr beams' replayed 0.5 µm steps, refused before on
  their spectral collar, now run in binary32 and agree with binary64 to 2e-7. The check
  uses the operator's rate bounds on the free-space and radial routes instead of forming
  the operator over the grid (seconds at 384 x 384 x 750). (#343)
- Fixed the reference's single-precision path leaving binary32 on two paths that it then
  narrowed without a check: the split-step collar now keeps a complex64 state in
  binary32, as the correction `a + F(F^-1(a)(f - 1))` that `mlx` uses (binary64 runs are
  unchanged), and a fixed-schedule batch of single-precision `numpy_ref` members now
  steps each member through its own binary32 trial instead of the binary64 batch
  arithmetic, and a lockstep batch refuses members of mixed precisions. The finished
  state is now refused by name if it is not complex64 (#79).
- `Simulation.estimate()` no longer crashes the process on the native backend under
  `LowStorageRungeKutta()`, and on every backend it now times that march's own step, so
  its time per step is the low-storage march's (#492). A trial step on a native
  low-storage state now raises a named error instead of faulting.
- The native backend now refuses an unknown `SOLPROP_FFTW_PLAN` value, which it had read
  as `measure`, and a problem whose `response_order` omits a response it holds, which it
  had run anyway. An empty `SOLPROP_FFTW_PLAN` now counts as unset (#460).
- A native run under Runge-Kutta or Adams now records the `propagator_mode` it used,
  `recompute`, when `tabulate` was asked for; it had recorded `tabulate`, which that
  march can't do. The run's configuration still keeps what was asked (#464).
- On an AMD device, the far-field pass decided about 3 % of the columns exactly at a hard
  aperture's edge differently from `numpy_ref`, because the compiler fused products and
  sums the pass meant to round separately. The radius is now rounded as on every other
  build.
- The native backend's HIP build compiles again under its default warnings-as-errors:
  ROCm's Clang refused a thread-local variable that only an OpenMP build uses (0.1.0
  needed `-DSOLPROP_NATIVE_WARNINGS_AS_ERRORS=OFF`).
- With `/proc/cpuinfo` unreadable, native chose its x86-64-v3 build, so a lower processor
  could reach an illegal instruction
  ([#443](https://github.com/LupoLab/solprop/issues/443)). It now asks NumPy, and refuses
  before loading if neither answers. The Windows on Arm wheel gains a processor check
  too, for the Armv8.2-A OpenBLAS it links.
- The versions of solprop, solprop-native and solprop-gui now come from `v*` tags only,
  so another tag on a commit can no longer set them. Building any of them now needs
  setuptools-scm 9, and solprop-native requires NumPy 2.5, as solprop does (#92).
- `poe` tasks now run in the environment `poe` was started in, so
  `uv run --no-sync poe fetch-wheel` no longer tries to build the native backend first,
  and `uv run --no-sync poe test` no longer replaces a fetched wheel with a local build.
  The contributor guide now gives the command that works with no compiler.
- [Building the native backend](docs/guide/performance/building-native.md) records why
  the macOS refusal after pyFFTW stays and what would lift it, and no longer says that
  importing solprop first avoids it: importing solprop does not load the backend (#315).
- Correct the 0.1.0 limitation statement about `mlx`: its custom Metal kernels already
  run in continuous integration's macOS default lane. That lane now requires Metal, so
  losing the device fails the job rather than silently skipping the kernel tests. MLX
  coverage, runtime type checks and an explicit CPU-device pass remain absent.

## [0.1.0] - 2026-10-31

The first public release of `solprop` and of its compiled backend, `solprop-native`,
which is released at the same version. Releases are 0.x for now: a minor release may
break the API, the output schema or the resume key, and lists every such break under
"Changed (breaking)" (ADR-058).

### Added

- **Propagation equations.** The unidirectional pulse propagation equation (UPPE) and
  the generalised nonlinear Schrödinger equation (GNLSE), for the real electric field or
  a complex envelope, in waveguides and in free space.
- **Waveguides.**
  - Gas-filled hollow capillaries, with resolved modes (the default) or mode-averaged
    (`mode_average=True` on `prop_capillary` and `prop_waveguide`). Tapers,
    pressure gradients and gas flow are supported, as is a coherent wall that couples
    the modes.
  - Antiresonant hollow-core fibres (the Vincetti model).
  - Step-index, graded-index, rectangular and planar guides.
  - Modes tabulated from arrays or from external mode solvers.
  - A general modal route for any guide that supplies its own electric and magnetic
    mode fields, including guides that change along their length.
- **Free space.** Radial, Cartesian and line grids with the exact (non-paraxial)
  diffraction operator; planar guides can be resolved across their width. Launches,
  shot noise, energies, fluences and intensities measure the medium's axial Poynting
  flux at the plane, so a requested energy is the physical energy there
  ([SEP 34](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0034-medium-flux.html)).
  Far-field and aperture statistics.
- **Media and responses.**
  - Gases, gas mixtures, glasses and metals from the [optmat](https://github.com/LupoLab/optmat)
    material library.
  - Kerr and third-harmonic nonlinearity, including a dispersive electronic response.
  - Photoionisation with atomic and molecular rate models, and plasma on field and
    envelope grids.
  - Rotational and vibrational Raman responses, scalar and vector.
  - Quantum shot noise, on by default
    ([SEP 37](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0037-noise-and-stepper-defaults.html)):
    one photon per mode with a random phase, added to the input field (`noise="input"`)
    and recorded with its seed. A resumed run draws the same realisation again, and a
    scan draws each point's from a seed of its own.
    `noise=None` switches it off and `noise="modified"` keeps the realisation apart
    from the field. The Cartesian, line and hybrid planar grids, slab mode sets, the
    TDSE response and single precision carry none by default and record why. Spectral
    densities can subtract the one-photon floor (`noise_floor="subtract"`).
  - An optional helium TDSE atomic response for research. Its native backend is not
    yet publicly available.
- **Integrators and their defaults.**
  - Two adaptive integrators: a low-call exponential Adams method (`Multistep()`) and
    an embedded interaction-picture Runge–Kutta pair (`RungeKutta()`), with a
    low-storage Runge–Kutta variant and fixed schedules. The linear propagation is
    integrated exactly.
  - Each interface has its own default integrator and tolerances, chosen by
    measurement: `RungeKutta()` for `prop_gnlse` (`rtol` 1e-7), `prop_waveguide` (1e-5)
    and `prop_free` (1e-4); `Multistep()` for `prop_capillary` (1e-6), `prop_radial`
    and `prop_planar` (1e-5). The
    [choosing-a-stepper page](https://solprop.readthedocs.io/en/latest/guide/numerics/choosing-a-stepper.html)
    tabulates them and says when to choose another.
  - `Multistep(carrier_frame="scalar")`, an opt-in carrier frame for the Adams history
    that is the fastest method measured on dense, multimode, Raman-active runs such as
    a comb-to-continuum capillary
    ([SEP 35](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0035-carrier-frame-exponential-adams.html)).
  - Backends agree with each other to bounded, rounding-level differences.
- **Time-window boundaries**
  ([SEP 21](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0021-absorbing-boundaries.html)).
  Every interface absorbs at the edges of the time window and of the spectrum. The
  temporal collar lies outside the requested window: the grid widens to
  `trange / (1 - 2 collar_fraction)` (`collar_fraction` 0.1), so the whole request is
  flat, and transform sizes default to `fft_size="auto"`, the cheapest 2-3-5-smooth
  size up to the next power of two. `absorber=False` switches the soft collars off and
  keeps the same grid. A boundary monitor, on by default (`boundary_check=` on every
  `prop_*` function and `Simulation.from_*` constructor), warns (`BoundaryWarning`) and
  then stops the run (`BoundaryError`) when light above the shot-noise floor reaches
  the window's edge, and records the energy the collar removed.
- **Coupling simulations**
  ([SEP 31](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0031-simulation-coupling.html)).
  `solprop.coupling` hands a field from one simulation to the next. A `FieldPlane`
  holds a coherent field at one plane with its reference and provenance; `couple` maps
  it into another simulation's `InputPort` under a declared `CouplingModel` and returns
  an energy report; `transport` carries a free-space plane through lenses, apertures,
  masks, mirrors, interfaces and spectral filters; `Outcouple` synthesises a guide's
  exit field. An incident beam launches into a general modal guide with the share of
  its energy that the guide captures. Each step is admitted against an error budget and
  refuses what it cannot certify.
- **Backends.**
  - `numpy_ref`, the NumPy reference, against which every other backend is tested
    within derived error bounds, and `numba_cpu`, compiled with Numba. The backends come
    in tiers: new features land on `numpy_ref` and `native`; `numba_cpu` is mature and
    takes a new feature only where its design says so; `mlx` is experimental.
  - The optional [native backend](https://solprop.readthedocs.io/en/latest/guide/performance/native.html),
    the solprop-native distribution: C++20 kernels on Kokkos for multi-core CPUs
    (OpenMP) and for NVIDIA and AMD GPUs (CUDA, HIP). `pip install "solprop[native]"`
    installs a precompiled CPU wheel on macOS on Apple silicon, Linux on x86-64 and
    aarch64, and Windows on x86-64. `backend="auto"` selects it when installed, except
    on a processor below the wheel's CPU baseline, where it uses `numba_cpu` and warns
    once ([SEP 29](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0029-compiled-backend-distribution.html)).
    On the native backend the exponential Adams march runs in compiled code, many steps
    per call
    ([SEP 39](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0039-native-adams-march.html)).
  - One thread budget per process, which every pool of both CPU backends stays within:
    the smallest of the CPU affinity mask, the physical cores that hold it, the
    control group's CPU quota and `SOLPROP_THREADS` (ADR-052).
  - Single precision
    ([SEP 33](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0033-single-precision-mlx.html)):
    `precision="single"` holds the state, the transforms and the responses in binary32,
    with the controller in binary64. `numpy_ref` runs it as the reference, and the `mlx`
    backend (`pip install "solprop[mlx]"`) on the Apple GPU. A
    [survival record](https://solprop.readthedocs.io/en/latest/guide/performance/single-precision.html)
    states which problems keep their accuracy.
- **Running and recording.**
  - HDF5 output with periodic snapshots and provenance, and resuming an interrupted
    run, which continues the Adams history.
  - A summary at the start of every run of what it is about to compute (the code, the
    machine, the backend, the grid, the medium, the responses, the noise, the
    integrator, the absorber and the output), and at its end the wall time, CPU time
    and peak memory. Each execution records its machine and what it spent in the file.
  - Statistics along the propagation.
  - Parameter scans on process pools and Slurm clusters.
- **Analysis and tools.**
  - Processing and plotting. Spectral ranges and band-pass windows are given in the
    units of any spectral axis (`spec_range`, `Passband`).
  - Ensemble coherence by spectral band, and replay of a run's accepted steps
    (`FixedSchedule.from_history`), to compare two precisions or backends on one
    discretisation.
  - Soliton and dispersive-wave design tools.
  - Pulse compressors: Taylor phase, chirped mirrors, materials, gratings and prisms.
  - An interactive session through `import solprop.sol as sol`.
- **Performance tracking** for developers
  ([SEP 38](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0038-performance-tracking.html)):
  `python -m benchmarks track` times about a dozen representative workloads, with each
  interface's defaults, at one thread and at the production width, and records them by
  hardware; `python -m benchmarks ab` compares two builds.
- **Documentation.** An introduction with executable lessons, a user guide, a
  long-form primer on the physics and numerics, focused tutorials and worked examples,
  the API reference, and a developer's guide with the design records (SEPs and ADRs).

### Changed (breaking)

Nothing was released before 0.1.0. For scripts and files from a development checkout:

- **Output files of development versions are not read.** The output schema restarts at
  1 (`solprop.output.SCHEMA_VERSION`); keep the development install that wrote such a
  file to read it. The kernel contract's version is `KERNEL_SCHEMA_VERSION`, separate
  from it.
- **Defaults changed late in development**, so a development script's results change:
  shot noise is on; each interface takes its own integrator and tolerances; the
  temporal collar lies outside the requested window, which widens the grid
  (`collar_placement="inside"` rebuilds the former geometry); transform sizes are
  `fft_size="auto"`; and the boundary monitor is on.
- **Renamed or removed.** `prop_gnlse`, `prop_capillary`, `prop_waveguide` and their
  `Simulation.from_*` constructors take `wavelength` and `shape`, not `lambda0` and
  `pulseshape`, and `responses_for` takes `wavelength`, as the pulses and the design
  tools already did. The plots' `wavelength_range` is `spec_range`, in the units of
  their axis, and `solprop.plotting.style.spectral_limits` is gone
  (`wavelength_to_axis` converts a wavelength range); `bandpass` takes `limits`, not
  `wavelength_limits`; `poynting_spectrum` is gone (`beam_spectrum` is the axial-flux
  spectrum); the `same_basis` function is gone (use `couple`); and `solprop.driver` no
  longer exports `propagate_fixed_rk4`, the tests' fixed-step oracle. `Multistep()`
  refuses the `PIController` settings it does not read.

### Known limitations

- **The `mlx` backend is experimental.** Its Metal kernels are tested by hand on a
  Metal device, not in continuous integration, and its coverage is not measured there.
  The single-precision survival record is the reference implementation's (on `mlx` for
  two cases). On the default `fft_size="auto"` grids MLX's GPU transform runs four to
  eight times slower than on powers of two, so a run can be slower than binary64
  `native`: build single-precision grids with `fft_size="pow2"`. The guide's speed table
  predates the current grids and is marked so.
- **Free space has no accurate transverse absorber.** By default a radial grid's edge is
  a hard wall and a Cartesian or line grid wraps around; the optional real-space collar
  (an absorber given to a spatial run, `Absorber.spatial`) removes only part of the light that reaches it, at a large cost
  in steps, and the boundary monitor watches the time window only. Size the transverse
  window so that the light stays clear of its edge.
  [SEP 40](https://solprop.readthedocs.io/en/latest/dev/seps/archive/sep-0040-transverse-absorbing-boundaries.html)
  (a draft) rebuilds these boundaries.
- **A run is resumed by the release that wrote it.** A run's configuration hash covers
  every recorded field, so a release that adds one changes every hash, and within 0.x
  the output schema may change too: finish a run with the release that started it.
- **GPU builds are made from source.** The wheels are for CPUs; CUDA and HIP builds of
  solprop-native are built from the repository
  ([building the native backend](https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html)).
- **CPython 3.14 only.** solprop-native, pyFFTW and h5py have no wheels for 3.15 yet.
- **macOS: import solprop before pyFFTW.** pyFFTW's macOS wheel starts its own OpenMP
  runtime, so the native backend refuses to load in a process that imported pyFFTW
  first; solprop itself imports them in the working order.
