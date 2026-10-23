# Interactive use: the session namespace

Use {py:mod}`solprop.sol` to explore a calculation at a prompt or in a notebook.
One import exposes common simulation, plotting and analysis functions and
configures progress reporting. It also silences material-range warnings: turn
these back on when assessing whether the chosen model covers your spectrum.
For a reproducible script, import the required objects from their home modules
and make logging and warning settings explicit.

```{code-block} python
import solprop.sol as sol  # or: from solprop.sol import *

p = sol.capillary_params(
    125e-6, "He", 1.0, energy=275e-6, fwhm=7.5e-15, wavelength=800e-9
)
print(p)
run = sol.prop_capillary(
    125e-6,
    1.0,
    "He",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(100e-9, 1.5e-6),
    trange=200e-15,
    fwhm=7.5e-15,
    energy=275e-6,
)
sol.prop_2d(run)
sol.compress_taylor(run)
```

It is for interactive use only. Library code, and any script you mean to keep, should
import from the home modules, as the rest of this manual does.

## What importing it does

{py:mod}`solprop.sol` is the one module of the package whose import has side effects;
that is its purpose. Importing it calls {py:func}`~solprop.sol.start_session`, which

1. **configures logging at `INFO`** through {py:func}`solprop.log.configure_logging`, so a
   run reports its set-up and, on a terminal or in a notebook, draws its progress bar
   ([progress](../performance/progress.md)); and
2. **silences {py:class}`~solprop.exceptions.MaterialRangeWarning`**. This makes exploration quieter, but can hide material extrapolation.
   Re-enable these warnings with `sol.material_warnings(True)` when assessing
   the validity of a selected wavelength band. The library's default is unchanged: a
   script that imports from the home modules still sees it.

Both are idempotent — import it twice, reload it, call `start_session()` again: one
handler, one filter. Nothing is printed, and matplotlib's backend and interactive mode
are left alone.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - to
     - call
   * - see the range warnings again
     - ``sol.material_warnings(True)``
   * - change the log level
     - ``sol.set_log_level("DEBUG")`` (or ``"WARNING"`` for a quiet session)
   * - log to a file, or choose the progress style
     - ``sol.configure_logging("INFO", file=..., progress="log")``
   * - undo everything
     - ``sol.end_session()``
```

Silencing installs one `"ignore"` filter for that one category, and
{py:func}`~solprop.sol.material_warnings` with `True`, like `end_session()`, removes
exactly that entry: whatever was in force before the import is in force again, a
stricter filter of your own included, which is never touched or shadowed. The notices about a
pressure floor or a vacuum end are log records, not warnings, and are not touched.

`import solprop` alone does none of this and stays cheap: the namespace is loaded on
first use, like every other submodule. Importing `solprop.sol` takes noticeably longer,
because it loads matplotlib, h5py and numba.

## What is in it

The names are flat, explicit and sorted — `sol.__all__` is the list — and every one is
the same object as in its home module:

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - group
     - names
   * - runs
     - ``prop_capillary``, ``prop_gnlse``, ``prop_waveguide``, ``prop_radial``, ``prop_free``, ``prop_planar``, ``Simulation``, ``SimulationSettings``, ``Multistep``, ``RungeKutta``, ``Scan``, ``ScanPoint``, ``run_scan``, ``Local``, ``Pool``, ``Queue``, ``Slurm``, ``Batch``
   * - outputs
     - ``open_file``, ``resume``, ``MemoryOutput``, ``HDF5Output``, ``open_scan``, ``ScanOutput``
   * - building blocks
     - the grids, the pulses, ``MarcatiliMode``, ``VincettiMode``, ``TubeLattice``, ``StepIndexMode``, ``RectMode``, ``SlabMode`` (with ``gaussian_coupling``, ``gaussian_projection``, ``per_unit_width_to_gaussian`` and ``gaussian_to_per_unit_width``), ``PlanarGrid``, ``IndexTable``, the tabulated modes, ``ModeCollection``, ``Gas``, ``Mixture``, ``species``, ``MaterialSelection``, the glasses and ``glass``, ``Constant``, ``Linear``, ``PiecewiseLinear``, ``PressureGradient``, ``DensityPath``, ``OptmatRate``, ``KerrResponse``, ``PlasmaResponse``, ``EnvelopePlasmaResponse``, ``RamanResponse``, ``DampedOscillator``, ``RotationalLines``, ``SpecifiedMode``
   * - mode and material numbers
     - ``beta``, ``beta1``, ``beta2``, ``alpha``, ``zdw``, ``aeff``, ``loss_length``, ``db_per_m``, ``transmission``, ``n2``, ``refractive_index``, ``density``
   * - design tools
     - everything in :py:mod:`solprop.tools` (:doc:`design-tools`)
   * - plots
     - ``prop_2d``, ``time_1d``, ``spec_1d``, ``spectrogram``, ``plot_energy``, ``plot_stats``
   * - processing
     - ``grid_of``, ``get_field``, ``get_power``, ``get_power_weight``, ``get_spectrum``, ``get_spectral_density``, ``get_phase``, ``bandpass``, ``energy``, ``peak_power``, ``fwhm``, ``fwhm_t``, ``fwhm_f``, ``arrival_time``, ``time_bandwidth``, ``specres``, ``g12``, the autocorrelations and ``coherence_time``, and the general modal accessors ``electric_field``, ``magnetic_field``, ``reconstructed_flux``, ``physical_energy`` and their companions (:doc:`../results/processing`)
   * - optics and compression
     - ``TaylorPhase``, ``Material``, ``ChirpedMirror``, ``Chain``, ``chirped_mirrors``, ``GratingPair``, ``PrismPair``, ``PrismGeometry``, ``littrow_angle``, ``brewster_angle``, ``brewster_apex``, ``compress_taylor``, ``compress_material``, ``compress_mirrors``, ``compress_gratings``, ``compress_prisms``, ``optimise_cep``, ``transform_limit``, ``apply_transfer``, ``Compression``, ``CepOptimum`` (:doc:`compression`)
   * - the session
     - ``start_session``, ``end_session``, ``set_log_level``, ``material_warnings``, ``configure_logging``
```

**One rule for a name two modules share:** the processing function keeps it and the plot
takes a `plot_` prefix. {py:func}`solprop.processing.time.energy` is `sol.energy`, the number;
{py:func}`solprop.plotting.propagation.energy` is `sol.plot_energy`, the figure; likewise
`sol.plot_stats`. `numpy` and `matplotlib.pyplot` are deliberately **not** re-exported, so
`from solprop.sol import *` never shadows your own `np` and `plt`. A test compares
`sol.__all__` with the public names of the home modules, so the namespace cannot fall
behind them. `SpecifiedMode` is here, but the dispersion data it is built from is not:
`TaylorDispersion`, `TabulatedDispersion`, `DispersionPath`, `ConstantDispersion`,
`InterpolatedDispersion` and `as_dispersion_path` are imported from
{py:mod}`solprop.nonlinear.gnlse`, where they live beside the mapping that uses them.

The names are this package's own snake case, with no capitalised aliases; a table
mapping Luna.jl's names to these is in [Noteworthy differences from Luna](luna.md).

## What things look like when you print them

Typing an object at a prompt should tell you what it is. Three rules, kept by every
class you are likely to meet:

- `repr(x)` is **one line and never holds an array**, so it is safe in a log record or an
  error message: `GaussianPulse(wavelength=800 nm, fwhm=30 fs, energy=1 uJ)`.
- `print(x)` is a short aligned block in the units people quote — nm, fs, µJ (written
  `uJ`: the text is ASCII), µm, bar.
- IPython and Jupyter show the block when the object is the last expression of a cell
  (through `_repr_pretty_`); the plain Python prompt shows the one line.

Printing never runs a calculation that can warn or that costs more than a fraction of a
millisecond (a capillary's area is a constant times its radius squared; a mirror's
dispersion at its band centre is one spline evaluation), and it opens no file except an
HDF5 handler's own, read-only. A scan file is never opened to be printed, because
reading one waits for its writer.

In IPython or a notebook (at the plain prompt, use `print`):

```text
In [1]: grid
FieldGrid: a real field grid
  time samples:      2048 (4096 oversampled)
  time window:       455.4 fs, step 222.4 as
  undamped window:   -150 fs to 150 fs
  frequency samples: 1025, step 2.196 THz
  spectral band:     200-2000 nm
  reference:         800 nm
  length:            100 mm

In [2]: simulation
Simulation: 2 modes
  grid:      FieldGrid(n_t=2048, window=455.4 fs, band=200-2000 nm, reference=800 nm)
  mode 0:    MarcatiliMode(HE11, a=125 um, He, model=full, loss=True)
  mode 1:    MarcatiliMode(HE12, a=125 um, He, model=full, loss=True)
  responses: KerrResponse, PlasmaResponse
  input:     <2x1x1025 complex>
  noise:     none
  settings:  SimulationSettings
               save planes: <200 planes, 500 um to 100 mm>
               tolerances:  rtol 1e-06, atol 1e-10
               (and the rest of the settings)

In [3]: run
MemoryOutput
  state:           open
  planes:          201, 0 m to 1 m
  field per plane: 2 x 1 x 4097 (modes x components x omega)
  statistics:      energy, fwhm_t, omega0, peak_intensity, peak_power, zdw, ...
  between planes:  none
  run:             RunMetadata(multimode, field grid of 8192, 2 modes, numba_cpu, ...

In [4]: pulse
GaussianPulse
  wavelength:     800 nm
  fwhm:           30 fs
  energy:         1 uJ
  spectral phase: CEP 0.3 rad, GDD 250 fs^2
  delay:          0 s

In [5]: modes
ModeCollection: 2 modes
  0: MarcatiliMode(HE11, a=125 um, He, model=full, loss=True)
  1: MarcatiliMode(HE12, a=125 um, He, model=full, loss=True)
```

An {py:class}`~solprop.output.hdf5.HDF5Output` says whether its file is open, closed,
read-only or gone. A {py:class}`~solprop.tools.SolitonParams` prints the block shown in
[design tools](design-tools.md), and a
{py:class}`~solprop.processing.compression.Compression` says `NOT CLEAN` when its search did not
converge inside its bounds.

A capillary mode built by the run interfaces carries its fill as a mixture, and prints
the names of its species.
