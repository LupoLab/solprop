# Reading the output of a modal run

Every run returns an output handler — in memory, or an HDF5 file with `output=` — and
everything is read from it through the accessors of {mod}`solprop.processing` and the
figures of {mod}`solprop.plotting`, exactly as for any other run ([analysing a
run](../../auto_tutorials/basics/plot_03_analysing_a_run.rst), [output](output.md)). This
chapter covers what is different about a capillary with a coherent wall and about the
general modal route: which quantities a basis defines, what the record holds, and how the
physical fields are rebuilt.

## A capillary with a coherent wall

Nothing changes. The coupled flow couples the modes; it does not change the basis the
state is written in, so each saved channel is still a Marcatili mode's true-power
amplitude, and every ordinary accessor — `energy` per mode, `peak_power`, `fwhm_t`,
`get_spectral_density`, the propagation maps — reads it as on any modal run. On a taper each
mode's amplitude carries that mode's own normalisation at every position, so a mode's
`energy` is that mode's energy there. What the wall adds is the record's `flow` entry
([the capillary chapter](../waveguides/capillaries.md#reading-what-the-wall-was)).

## The general modal route

A general modal run's state is a set of **channel amplitudes** in √W, one per channel of
the guide, normalised to unit Poynting flux. On the provider's own basis they are
orthonormal in true power; on a declared basis (a circular pair, a rotated split block)
they are not orthogonal in the same sense, and the basis carries a dense metric $W$. Two
consequences decide what a run reports.

**The statistics are the physical ones, and they are scalars.** `physical_energy` is the
energy in the physical field, $\int \mathbf a^{\mathsf H}W\mathbf a\,d\omega$ summed
over the channels; `analytic_flux` is the peak over time of the flux built from the
same pairing of each frequency with itself; `spectral_flux`, requested by name, is the
spectral density of `physical_energy`, in J/(rad/s). On the
provider's own basis these are exactly `sum(energy)` and `peak_power_total`; on a
declared one they are what those would have to become. The names that promise a
per-channel energy or power (`energy`, `peak_power`, `peak_power_total`,
`energy_window`, `peak_power_window`, `peak_intensity`) are **refused by name** on this
route, each with its replacement, rather than redefined; `quadrature_error` is refused
too, because a provider's node rule has no embedded coarse rule to compare with — there
is no automatic transverse-resolution alarm during a general modal run. `fwhm_t`,
`fwhm_t_outer` and `omega0` stay, taking the metric. `zdw` stays, except beside a split
block, which has no per-channel propagation constant.

```python
import numpy as np

from solprop.driver import prop_waveguide
from solprop.exceptions import OutputError
from solprop.grid import FieldGrid
from solprop.modes import (
    IndexMaterial,
    MaterialKerr,
    MaterialRegion,
    ModalGuide,
    StepIndexMode,
)
from solprop.processing import get_spectral_density
from solprop.processing import energy as energy_per_mode
from solprop.processing.modal import electric_field, physical_energy, reconstructed_flux
from solprop.stats import StatsSpec
from solprop.translate.general import general_modal_route

core, cladding = IndexMaterial(1.45), IndexMaterial(1.44)
pair = [
    StepIndexMode(6e-6, core=1.45, cladding=1.44, profile_wavelength=835e-9, parity=p)
    for p in ("even", "odd")
]
regions = (
    MaterialRegion("core", 0.0, 6e-6, core, (MaterialKerr(n2=2.6e-20, at=835e-9),)),
    MaterialRegion("cladding", 6e-6, None, cladding),
)
provider = ModalGuide.from_step_index(
    pair, core=core, cladding=cladding, regions=regions
).field_provider
grid = FieldGrid.from_limits(1e-3, 835e-9, (790e-9, 890e-9), 200e-15)
active = grid.omega[grid.active_mask]
low = 0.99 * float(active[0])
guide = ModalGuide.from_provider(
    provider,
    np.linspace(low, 1.01 * float(active[-1]), 25),
    nodes=provider.node_rule(low, n_r=24),
    degenerate_blocks=((0, 1),),
)
run = prop_waveguide(
    (),
    1e-3,
    provider=guide,
    grid=grid,
    lambda0=835e-9,
    fwhm=40e-15,
    energy=1e-9,
    n_save=3,
    dz_max=1e-4,
    dz_init=1e-4,
    progress="none",
    stats=StatsSpec(names=("physical_energy", "analytic_flux", "spectral_flux")),
)
print(run.metadata.modal["accessors"])
print("physical energy (J):", physical_energy(run))

try:
    energy_per_mode(run)
except OutputError as refusal:
    print("refused:", str(refusal)[:100], "...")
```

The spectra are read as for any run — `get_spectral_density` sums the channels' true-power
spectral densities:

```python
wavelength, density = get_spectral_density(run, "wavelength", plane=-1, modes="sum")
print(f"spectral peak at {wavelength[np.argmax(density)] * 1e9:.1f} nm")
```

**The file holds amplitudes, not fields.** The run's record names the guide — the
provider, the sampling, the materials and their responses, the channel labels row by row,
the table and metric identities with their measured Gram departures, and the basis — but
not its tens of megabytes of stored E and H. The physical fields are rebuilt from the
guide: {func}`~solprop.translate.general.general_modal_route` gives the run's own tables,
and every reconstruction accessor takes them **and** the guide, and checks both against
the identity the file carries, path by path, before it reads anything:

```python
route = general_modal_route(guide, grid, z_max=1e-3, omega_reference=float(grid.omega0))
field = electric_field(run, route.tables, plane=-1, guide=guide)  # V/m, (nodes, 3, t)
flux = reconstructed_flux(run, route.tables, plane=-1, guide=guide)  # W, (t,)
density = reconstructed_flux(run, route.tables, plane=-1, density=True, guide=guide)
print(field.shape, f"peak flux {flux.max():.3e} W")
print(f"peak flux density {density.max() * 1e-4:.3e} W/cm^2")
```

`reconstructed_flux` is the flux of the reconstructed fields, a trace in time, which
carries the guide's cross-frequency flux kernel as well; the run's own `analytic_flux`
statistic is the peak of the same-frequency one, and the two peaks differ by at most
$\sum_{ij}|B_i||B_j||K_{ij}-1|$, where $K_{ij}$ is the guide's flux kernel between
frequencies $\omega_i$ and $\omega_j$, one on its diagonal (measured: 1.3e-5 of the peak
on a strongly guiding strand). Their time integrals agree to far better than that,
because the cross-frequency terms integrate away. `instantaneous_flux` (the real fields'
flux, with its sum-frequency term), `magnetic_field` and `coordinate_intensity` (the
flux density each channel would carry alone, which is **not** additive) are the others,
and `plane_nodes` returns the node positions and area weights they are all given on;
[output](resuming.md#reading-e-and-h-back) and
[statistics](statistics.md#which-names-a-basis-admits) have the definitions.

**What the file's key protects.** A general modal file resumes only against the same
guide, table, materials, channel order and basis — every one of them is in the resume
key — and a reconstruction is refused against a guide that is not the one the file names.
Keep the script that builds the guide beside the file:

```python
from solprop.output import open_file

run.to_file("fibre.h5")
with open_file("fibre.h5") as saved:
    again = reconstructed_flux(saved, route.tables, plane=-1, guide=guide)
    print("the same flux from the file:", bool(np.allclose(again, flux)))
```
