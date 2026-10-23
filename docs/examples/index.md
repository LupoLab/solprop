# Examples

Use these calculations to explore a physical process, learn a particular interface,
or adapt a complete simulation to your experiment. Each gallery page provides its
Python source and a notebook download.

## Start with a complete workflow

The [introduction](../introduction/index.md) leads from installation to propagation
and analysis. Its learning sequence covers a multimode dispersive wave, an N = 3
soliton, a radial filament, and Cartesian free space. Use that
sequence if you are new to solprop.

| Introductory lesson | Main question |
|---|---|
| [Dispersive-wave propagation](../introduction/lessons/plot_01_rdw.rst) | How do I run and analyse a multimode capillary calculation? |
| [N = 3 soliton](../introduction/lessons/plot_02_soliton.rst) | How do soliton order, compression and recurrence relate? |
| [Radial filament](../introduction/lessons/plot_03_filament.rst) | How do self-focusing and plasma appear in the output? |
| [Crossing beams](../introduction/lessons/plot_04_crossing_beams.rst) | How do I launch custom arrays and interpret coherent interference? |

## Find an example by subject

| Subject | Focused demonstrations | Worked simulations |
|---|---|---|
| Pulse propagation and nonlinear processes | [Capillary tutorials](../auto_tutorials/capillaries/index.rst) | [Nonlinear processes](../auto_examples/processes/index.rst): compression, dispersive waves, mixing, Raman, and ionisation |
| Supercontinuum and noise | [Input pulses and their shot-noise floor](../auto_tutorials/basics/plot_02_pulses.rst) | [Supercontinuum generation](../auto_examples/supercontinuum/index.rst): mechanisms, ensembles, noise, and coherence |
| Waveguide geometry | [Waveguide tutorials](../auto_tutorials/waveguides/index.rst) | [Waveguide physics](../auto_examples/waveguides/index.rst): gradients, tapers, vector modes, and planar compressors |
| Beams and free space | [Free-space tutorials](../auto_tutorials/free_space/index.rst): radial, Cartesian, line, and planar calculations | Follow their links to the [free-space guide](../guide/free-space/index.md) for model and resolution choices |
| Inputs, analysis, and workflows | [Basics](../auto_tutorials/basics/index.rst) and [optical tools](../auto_tutorials/tools/index.rst) | Use the analysis in a worked simulation as a starting point for your own output |

## Two kinds of calculation

**Focused demonstrations** isolate a feature or a modelling choice. Some use
reduced resolution to make the calculation easy to inspect; read their stated
limitations before using the parameters for quantitative work.

**Worked simulations** follow a problem from setup to interpretation at its full
propagation length. They can take substantially longer to run. Their pages include
stored figures and the runtime and backend of the recorded calculation.

Use the [user guide](../guide/index.md) for the assumptions, detailed equations,
and convergence checks behind an example, and the [API reference](../api/index.md)
for individual arguments.

```{toctree}
:maxdepth: 2

Focused demonstrations <../auto_tutorials/index>
Worked simulations <../auto_examples/index>
```
