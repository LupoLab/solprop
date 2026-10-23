# Where to go next

After a first run, use the [user guide](../guide/index.md) to develop your model and
check the result. It is arranged as a reading sequence, but each section also
provides an entry point for a particular task.

| Your next task | Read |
|---|---|
| Choose a time window and wavelength band | [Grids and Fourier conventions](../guide/setup/grids.md) |
| Use another waveguide or a taper | [Waveguide propagation](../guide/waveguides/index.md) |
| Model focusing or filamentation | [Free-space propagation](../guide/free-space/index.md) |
| Choose a material, Raman response, or ionisation model | [Materials and nonlinear responses](../guide/media/index.md) |
| Check convergence and numerical accuracy | [Numerical methods and accuracy](../guide/numerics/index.md) |
| Save a run, plot it, or interpret its statistics | [Results and interpretation](../guide/results/index.md) |
| Explore parameters or compress the output pulse | [Design and workflows](../guide/workflows/index.md) |
| Run on another backend or tune performance | [Backends and performance](../guide/performance/index.md) |

Browse the [examples](../examples/index.md) for calculations that combine these
steps. The [API reference](../api/index.md) documents function arguments, units,
return values, and exceptions.

## Project information and help

The [solprop repository](https://github.com/LupoLab/solprop) hosts the source code
and issue tracker. When reporting a problem, include a small reproducible script,
the package and backend versions, and the behaviour you expected. For a question
about a physical result, state the model, parameters, numerical resolution, and
reference used for the comparison.

[About solprop](about.md) describes the project. The [changelog](../changelog.md)
records changes between versions, and
[differences from Luna](../guide/workflows/luna.md) explain deliberate modelling
and convention differences. To contribute code or documentation, use the
[developer’s guide](../dev/index.md).

```{toctree}
:maxdepth: 1

about
../changelog
```
