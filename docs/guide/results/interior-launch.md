# Starting from a saved interior field

A saved field can be the input to a new calculation beginning at its physical position
inside a guide. Use this to compare response models over the same subsequent interval. A
new segment records a new launch; continuing an interrupted calculation is instead
described under [file continuation](run-access.md#writing-to-a-file-and-resuming-it).

## Starting from an interior saved plane

An explicit modal spectrum can be launched at `initial_z`, in metres, while retaining
the original guide length, absolute density and pressure profiles, propagation frame
and coarse grid. The default is zero. A positive launch requires `noise=None`, no
fixed schedule, and an array shaped `(n_modes, n_pol, n_omega)` in the internal FFT
convention with true-power amplitudes. Every requested save must lie strictly beyond
the launch, and the launch must precede the guide end.

To retain an existing simulation's translated tables, use `with_input`:

```{code-block} python
from solprop.output import open_file

with open_file("parent.h5") as parent_output:
    plane_index = 10
    z_start = float(parent_output["z"][plane_index])
    spectrum = parent_output.fields(convention="fft")[plane_index]
    parent_hash = parent_output.metadata.configuration_hash()

segment = simulation.with_input(spectrum, initial_z=z_start, noise=None)
segment.run("segment.h5")
```

Here `simulation` is the matching parent configuration. An explicit `initial_z`
filters its existing save planes to those strictly beyond the launch; omitting the
keyword keeps the existing launch and saves. The new output includes the launch
plane, and its metadata hashes the new input and launch coordinate. Normalisation
uses the effective index at `z_start`, so a pressure gradient is sampled at the same
physical position as the saved parent field. To change the response model, construct
a new `Simulation` with the parent grid, modes, profiles and frame, the replacement
responses, this `spectrum` and `initial_z=z_start`; include the parent hash and plane
index in its `configuration` provenance and write a new output file.

The same continuation can be made through plane exchange, which checks the basis and
records the provenance for you: `couple(field_plane(parent_output, plane_index),
simulation.input_port(z_start), model=CouplingModel("same_basis")).launch` is a launch
that `simulation.with_input` accepts, at `z_start`, with the parent named in
`configuration["coupling"]`
([coupling simulations](../workflows/coupling.md#relaunching-into-the-same-basis)).

For a TDSE response, `SimulationSettings.tdse_execution` explicitly selects the
atomic runtime resources beside `backend="numpy_ref"` and a `Multistep` integrator.
The declared response sampling refines the existing fine lattice once, retaining
the coarse grid, period and collars. An inherited standard temporal collar is
evaluated on the finer lattice with the same physical reference length; a custom
absorber must already carry matching fine samples. Changing execution device,
batch width or memory limits does not alter the scientific resume identity.
