The reconstruction accessors of `solprop.processing.modal` (`electric_field`,
`magnetic_field`, `reconstructed_flux`, `instantaneous_flux`, `coordinate_intensity`,
`plane_nodes`) rebuild the run's reference tables from the required `guide=` and no
longer take `tables` positionally: `electric_field(run, guide=guide)`. Pass
`tables=` (from the new `solprop.processing.modal.reference_tables(run, guide=guide)`) to
reuse one rebuild across many calls; a script no longer needs
`solprop.translate.general.general_modal_route`. They read the last plane by default
(`plane=-1`) instead of requiring `plane=` or `z=`. The tables alone are rebuilt by the
new `solprop.translate.general.reference_tables`.
