# User guide

This guide explains how to build a propagation model, run it, and assess the result.
It covers waveguides and free space, the material responses that act on a pulse,
and the numerical methods used to evolve it.

If this is your first simulation, begin with the
[introduction](../introduction/index.md). Then read the guide in the order below,
or start with the chapter for your geometry. Detailed equations, conventions,
and approximation limits are part of these chapters; the
[API reference](../api/index.md) gives the precise arguments of each function.

| To… | Start with… |
|---|---|
| Choose a model, input pulse, and computational window | [Setting up a simulation](setup/index.md) |
| Propagate in a capillary, optical fibre, or another guide | [Waveguide propagation](waveguides/index.md) |
| Model a beam, focus, or filament in a bulk medium | [Free-space propagation](free-space/index.md) |
| Choose the material and nonlinear processes | [Materials and nonlinear responses](media/index.md) |
| Check whether a calculation is resolved | [Numerical methods and accuracy](numerics/index.md) |
| Read, plot, and interpret a saved run | [Results and interpretation](results/index.md) |
| Design a configuration, compress a pulse, or scan parameters | [Design and workflows](workflows/index.md) |
| Select a backend or tune a large calculation | [Backends and performance](performance/index.md) |

Each section links to [examples](../examples/index.md) that put the models to use.
When comparing a simulation with an experiment, read both the physical assumptions
of its model and the numerical checks needed to resolve it. A small integration
tolerance does not establish that the time window, bandwidth, or spatial basis is
adequate.

```{toctree}
:maxdepth: 2

setup/index
waveguides/index
free-space/index
media/index
numerics/index
results/index
workflows/index
performance/index
```
