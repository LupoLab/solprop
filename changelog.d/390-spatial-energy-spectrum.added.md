`solprop.spatial.spatial_energy_spectrum(spectrum, flux)` splits the energy that
`spatial_energy` measures into frequency bins, in J (J/m on a line), and the bins sum to
the total. It is not the density `beam_spectrum`. The coupling elements used it
privately, and their reports do not change
([#184](https://github.com/LupoLab/solprop/issues/184)).
