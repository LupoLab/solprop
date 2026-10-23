# Introduction

solprop simulates how an ultrashort light pulse changes as it propagates through a
waveguide or a bulk medium. Dispersion, diffraction, nonlinear polarisation and
ionisation can reshape its spectrum, duration and transverse field. These
chapters take you from an installation to interpreting those changes in four
complete calculations.

Begin with [installation](installation.md), the [capability overview](capabilities.md)
and [choosing a spatial description](spatial-description.md). The first propagation
lesson introduces the workflow used throughout the manual. Later lessons explain
the additional choices needed for a different equation and for spatially
resolved beams.

| Lesson | What you will learn |
|---|---|
| [Resonant dispersive-wave propagation](lessons/plot_01_rdw.rst) | Set up a multimode capillary, read spectral and temporal maps, select a band, and interpret energies and statistics. |
| [An N = 3 soliton](lessons/plot_02_soliton.rst) | Relate pulse parameters to soliton order, follow compression and recurrence, and distinguish energy conservation from field accuracy. |
| [A radial filament](lessons/plot_03_filament.rst) | Connect self-focusing and plasma to fluence, intensity and electron density; recognise the limits of radial symmetry. |
| [Two crossing beams](lessons/plot_04_crossing_beams.rst) | Launch custom Cartesian arrays, interpret interference, and check beam motion, superposition and spatial resolution. |

Each lesson executes as a complete script and includes downloadable Python and
notebook versions. Read the text between cells: it explains why the parameters
were chosen and which conclusions need further convergence checks. Runtime and
backend are recorded with the output. Full propagation is intentional; the
figures should show the physical development the lesson describes.

The [user guide](../guide/index.md) supplies the practical details, equations and
numerical methods behind the lessons. To understand them from the beginning, read the
[primer](../primer/index.md): one long account of the physics and numerics, from
Maxwell's equations to a converged result. The [examples](../examples/index.md) extend
them to other processes and fuller comparisons. Use the previous/next links to
read in sequence, or the chapter links to return to a particular task.

```{toctree}
:maxdepth: 1

installation
capabilities
spatial-description
lessons/plot_01_rdw
lessons/plot_02_soliton
lessons/plot_03_filament
lessons/plot_04_crossing_beams
next-steps
```
