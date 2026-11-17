`is_air`, which says whether a gas name means air (`Gas.Air`, or `"air"` in any case),
moved from `solprop.driver.mixtures` to `solprop.physdata.gases` beside `as_gas` and
`is_molecule`, and is exported from `solprop.physdata`. Import it as
`from solprop.physdata import is_air`.
