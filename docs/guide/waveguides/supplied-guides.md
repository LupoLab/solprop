# Supplying a waveguide

Use an explicit mode set when its geometry or dispersion is supplied by a rectangular,
planar, tabulated or other supported guide. A general modal guide instead supplies its
own vector fields and material regions. The two forms use the same entry point, but
their ownership of the gas fill and nonlinear responses differs.

## Supplied gas-filled waveguides

{func}`~solprop.driver.interface.prop_waveguide` takes an explicit mode sequence in
place of a capillary radius and mode count. Its construction counterpart is
{meth}`~solprop.driver.simulation.Simulation.from_waveguide`. Both use the same
construction body as the capillary entry, keeping its pulse, fill, response and
numerical defaults. Supplied modes keep their geometry, wall material and loss; the
requested gas or `Mixture` replaces their own fill (through `with_mixture`).

```{code-block} python
from solprop.driver import prop_waveguide
from solprop.modes import RectMode
from solprop.physdata import aluminium

output = prop_waveguide(
    [RectMode(50e-6, 10e-6, cladding=aluminium())],
    0.001,
    "Ar",
    5.0,
    lambda0=800e-9,
    wavelength_limits=(450e-9, 1200e-9),
    trange=100e-15,
    fwhm=30e-15,
    energy=1e-8,
    envelope=True,
    plasma=False,
    raman=False,
    n_r=12,
    n_theta=12,
    n_save=2,
    backend="numpy_ref",
)
```

This short example emits material-range warnings: the padded FFT spectrum
extends beyond the fitted argon band. For a Cartesian (rectangular) set, `n_r` and
`n_theta` are the x- and y-axis node counts, and both defaults take
{func}`~solprop.modes.quadrature.cartesian_nodes` of the set's highest transverse
order $h$, $6h+6$. A coarser rule warns. The
[mode guide](rectangular.md#hollow-rectangular-waveguides) explains the measured
requirement; 22 per axis resolve the `(3,2)` quartic example to $10^{-12}$ and the
bound rounds that to 24.

A **hollow planar** set ({class}`~solprop.modes.slab.SlabMode`,
[the mode guide](planar.md#hollow-planar-waveguides)) takes the same entry. It is
uniform in y, so its rule is Gauss(-Kronrod) in x alone: `n_r` is the x node count,
defaulting to $6h+6$ where that exceeds the radial default, and `n_theta` is refused.
The run is **per unit width**: `energy=` is J/m and `peak_power=` W/m, the recorded
powers and energies are too, and the metadata says so
(`RunMetadata.transverse_measure`), which is what the processing labels and plots
read. The pulse enters the first mode; a Gaussian beam's projection onto the set is
{func}`~solprop.modes.slab.gaussian_coupling`.

The component selection follows the supplied fields. An x/y rectangular partner pair
has **two mode rows**, each with its own index, and one state component per row; the
transform resolves both physical electric-field components. A two-component state
sharing a single rectangular index is refused. The set `prop_capillary` builds itself
(`modes=N`) resolves the x component alone, the direction its unrotated `HE_1m` modes
point. A set supplied to `prop_capillary` resolves its components as here, from its own
fields, and so it does beside a coherent wall, whose coupled flow never reads the
synthesis. That matters for a mode whose field is not along x: a `TE_01` mode resolved in
x alone would keep only 3/8 of its Kerr overlap and drive its plasma with $|E_x|$ alone.
Every supplied set is checked for orthonormality to $10^{-6}$.

The output includes `configuration["waveguide"]` with ordered mode records, length
and temperature. Geometry paths accompany the fill profiles. The default statistics
omit `core_radius` when no mode supplies one, and also omit `quadrature_error`:
a Cartesian product rule has no embedded coarse rule, so the quadrature alarm has
no error estimate and the statistic would be NaN at every plane. On-axis intensity
samples the transverse origin for either polar or Cartesian guides.
`StepIndexMode` has no gas-fill interface and is refused here with guidance to
`prop_gnlse(fibre=...)`, which is how the gallery propagates it.

### A guide instead of a fill

`prop_waveguide` and `Simulation.from_waveguide` take a **`provider=`** in place of
`gas`: a {class}`~solprop.modes.guide.ModalGuide` that solves its own
three-component E and H at quadrature nodes, whose material regions own their
responses and whose channels are normalised to unit Poynting flux. That is the
[general modal route](general-modal.md), and it is a different model, not a
different fill:

```{code-block} python
output = prop_waveguide(
    (),
    length,
    provider=guide,
    grid=grid,
    lambda0=835e-9,
    fwhm=40e-15,
    energy=4e-11,
    n_save=2,
    backend="numpy_ref",
)
```

The guide supplies the ordered channels, so `modes` is empty; `channel=` names the
one the pulse enters, by index or by display label. Every gas keyword above --
`gas`, `pressure`, `temperature`, `kerr`, `thg`, `plasma`, `raman`, `n_r`,
`n_theta`, `kronrod`, `envelope`, `node_budget` -- is refused **by name** beside a
provider, because the guide owns what each of them would set.
`observable_rtol` and `observable_atol` are accepted as `None`, meaning the
solver's own `rtol` and `atol`; a number is refused by name. **`kerr_form`**
chooses the instantaneous cubic the regions respond with: `"real"`, the default,
is the real-field cubic the field-grid routes above evaluate, third harmonic
included, and `"analytic"` is the analytic-signal form, which carries the
fundamental band alone and is what a like-for-like comparison with `prop_gnlse`
wants. The real form needs `n_to >= 2 n_t`, which is `FieldGrid`'s default
`ffac=6`, and is refused on a narrower grid
([general modal propagation](general-modal.md)).

It takes a **graded-index** guide as readily as a step-index one, and launches
either into a named channel with `channel=` or as an incident beam with
`incident=`, of which the channels launch the share they capture, the rest recorded as
uncaptured rather than restored. It runs on a **field** grid on `numpy_ref`, `numba_cpu` and the native
backend, CPU and GPU builds alike, for a constant guide and for a moving one or a
declared coupled flow (a split block or a coherent wall); a CF4 wall runs on `numpy_ref`
alone. `backend="auto"` selects `native` where it is installed.

It **writes and resumes an HDF5 file**: the file carries the channel labels, the
provider, sampling, table and metric identities and the accessors the basis admits, so
a resume against a changed guide, table, material or channel order is refused by the
field that moved, and a file written at an output schema before 7 is never resumed into
one ([output](../results/resuming.md#what-a-general-modal-runs-key-holds)). A scan point writes one
too. A **batch** member may be a general modal run, under a narrower admission rule than
an ordinary fibre batch's ([scans and batches](../workflows/scans.md#fixed-schedule-batches)): the
members of a group share the guide, the table, the gauge, the materials and one explicit
fixed schedule, and differ in their launch and their noise seed alone. What separates two
guides is `modal_tables.identity` in the grouping signature, because every other entry
there describes an execution layout two guides share entirely; a group assembled outside
that grouping is refused by name, at the driver and again at the backend. A batched run
is what its members are alone, to rounding.

A reconstruction of E and H needs the **guide beside the file**: the reference
arrays are not stored in it, so rebuild the tables from the guide and pass the
guide as well -- `guide=` is a **required** keyword of all six reconstruction
accessors, with no default, so the sampling and the regions are always checked too
([reading E and H back](../results/resuming.md#reading-e-and-h-back)).

Its statistics are `physical_energy` and `analytic_flux`, the route's own scalar
names, with `spectral_flux` -- the same physical energy as a density in frequency,
in J/(rad/s) -- admitted by name where it is asked for; `energy`, `peak_power`,
`peak_power_total`, their windowed forms and `peak_intensity` are refused by name
**with the replacement**, wherever they are asked for: when the run's statistics are
admitted, in each backend's `stats`, in `solprop.processing.modal`, and in
`solprop.processing.time.energy` and `peak_power`, which see only the stored record.
`quadrature_error` is refused as well and has **no** replacement: a provider's
node rule carries no embedded coarse rule to difference, so there is no
transverse-resolution alarm to offer instead
([statistics](../results/statistics.md#which-names-a-basis-admits)).

Like every field-grid route it applies the grid's default temporal absorber inside every
stage (`SimulationSettings.absorber_scheme="nonlinear"`, see [boundaries](../numerics/boundaries.md)),
which can set the step. `prop_gnlse` builds no absorber at all, so a comparison with it
needs either `absorber_scheme="split_step"` or `absorber=None` here. Read
[the general modal page](general-modal.md) before planning a run.
