# Materials and nonlinear responses

A material model supplies dispersion and response parameters; the response model
defines how the local optical field acts on the medium. Check both the material
validity range and the assumptions of the response. The [radial filament
lesson](../../introduction/lessons/plot_03_filament.rst) combines Kerr and plasma;
the [comb-to-continuum example](../../auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum.rst)
shows a molecular Raman response in a noisy pulse.

Start with material selection and the nonlinear overview, then follow the
processes relevant to your calculation. Detailed equations follow their physical
meaning and configuration. A numerically converged run still depends on the
material data and approximations described here.

```{toctree}
:maxdepth: 1
:caption: Materials and electronic response

materials
material-selection
mixtures
nonlinear
dispersive-kerr
```

```{toctree}
:maxdepth: 1
:caption: Molecular and glass Raman response

raman
raman-models
raman-vector
raman-numerics
raman-data
```

```{toctree}
:maxdepth: 1
:caption: Ionisation and noise

ionisation
field-plasma
envelope-plasma
noise
```

```{toctree}
:maxdepth: 1
:caption: Atomic response for research

tdse
```
