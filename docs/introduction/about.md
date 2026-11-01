# About solprop

**Everything, fast, with rigour.**

solprop aims to be one unified code for nonlinear pulse propagation: highly accurate,
fast, and scalable from a laptop to a GPU cluster, and able to solve the propagation
problems of ultrafast nonlinear optics wherever they arise.

- **Everything.** Guided and free-space propagation, the full field and the envelope,
  single-mode and multimode, gases and solids, in one code with one set of conventions,
  so that a new model extends every route at once rather than one script at a time.
- **Fast.** The compiled native backend runs on CPUs and GPUs, the same problem also runs
  on a NumPy reference and a Numba CPU fallback, and parameter scans spread over a process
  pool or a cluster.
- **With rigour.** Every model is derived and cited, tested against analytic solutions
  and independent references, checked for convergence, and held to the NumPy reference
  on every backend within bounds derived in advance.

AI-aided development ([below](#use-of-ai)) is what makes this ambition realistic. It
lets us bring in the best available models, implement them to the highest standards and
verify them rigorously, at a pace that a small team combining software with
experimental research could not otherwise sustain.

The foundation is already broad: the UPPE and the GNLSE; hollow capillaries with
coherent walls, tapers, pressure gradients and gas flow; antiresonant, step-index,
rectangular and planar guides, tabulated modes and a general modal route for any guide
with its own vector fields; radial, Cartesian, planar and line free space; Kerr,
ionisation and plasma, and rotational and vibrational Raman responses in gases and
their mixtures. To our knowledge, no earlier pulse propagation code has covered as much
in one package. The [capability overview](capabilities.md) describes the current
models and their limits.

It is a foundation to build on. We intend to keep adding the physics that pulse
propagation needs — the [roadmap](../dev/roadmap.md) lists what is missing, from
second-order nonlinearity and harmonic generation to ionisation in solids — until
solprop covers every physical propagation scenario, each addition specified, verified
and documented through an [enhancement proposal](../dev/seps/index.md).

**Join us.** If you have a problem solprop cannot yet solve, a model it should include,
a measurement or independent result to compare against, or code to contribute, open an
issue or propose a SEP; the
[contributing guide](../dev/contributing.md)
explains how to start.

## A history of the code

solprop draws on more than two decades of work on numerical models of nonlinear optical propagation. Its history begins in 2004, when John Travers wrote his first nonlinear Schrödinger equation solver at Imperial College London, with help from fellow PhD student Andrey Rulkov. Originally written in MATLAB, the code quickly grew into a comprehensive generalised nonlinear Schrödinger equation (GNLSE) solver, supporting the rapid development of supercontinuum research in solid-core photonic crystal fibres. Long discussions with John Dudley during his visits to Imperial helped shape this early work.

Several implementations followed as the research developed. A Fortran code supported simulations of supercontinuum generation in tapered optical fibres. Around 2010 the Fortran code was rewritten in C and used extensively for continuous-wave supercontinuum simulations. The code was subsequently revised and rewritten in Python, with contributions from Edmund Kelleher extending it to model mode-locked fibre lasers.

After John moved to the Max Planck Institute for the Science of Light in Erlangen, the focus broadened to ultrafast pulse propagation in gas-filled hollow-core fibres, including kagome fibres. Modelling ionisation and plasma dynamics led to a new, field-resolved Python implementation based on the unidirectional pulse propagation equation (UPPE), drawing on techniques developed by Kolesik and Kinsler. Wonkeun Chang’s contemporaneous work on a similar MATLAB code made an important contribution to this development.

The need for larger simulations and a wider range of physical models subsequently led to a C++ implementation, accelerated on GPUs using the Thrust library. Francesco Tani helped extend the code to describe multimode dynamics and interactions between fibre modes. This generation also incorporated Maxwell–Bloch modelling (thanks to significant input from Amir Abdolvand and Federico Belli) of Raman responses and resonant third-order nonlinearities in xenon, as well as the strong-field approximation for high-harmonic generation.

The next major development took place at Heriot-Watt University in Edinburgh. John had been using Julia since 2013, and he saw it as a way to make the propagation code easier to develop and maintain. Christian Brahms joined the group from Imperial, where he had developed his own Python code for hollow-core fibre propagation. Together, Christian and John undertook a new Julia implementation that became ([Luna.jl](https://github.com/LupoLab/Luna.jl)).

Luna brought together the capabilities of the earlier codes and extended them to spatially resolved propagation in radial and Cartesian geometries, including free-space filamentation, alongside a growing collection of nonlinear response models. Released as open-source software ([Luna.jl](https://github.com/LupoLab/Luna.jl)), it has been used by research groups around the world to study nonlinear propagation in hollow-core fibres, particularly soliton dynamics and resonant dispersive-wave emission.

In 2026, solprop began a new stage in this history. AI-assisted development made it possible to pursue a broader range of waveguide models, geometries and numerical methods at a pace that would previously have been difficult for a small team combining software development with experimental research. solprop builds on the experience gained through the earlier codes, with support for vector modes including axial field components, a wider range of nonlinear response models, and detailed treatment of hollow-capillary waveguides.

Numerical validation is central to this effort, through extensive testing of convergence and precision and comparison with Luna. The aim is to give the community a foundation for increasingly ambitious simulations and a deeper understanding of nonlinear dynamics across a broad range of optical waveguide structures.

## The name

The name is a play on words, twice over. It follows Luna, its predecessor: where Luna
is the moon, *sol* is the sun, bigger and brighter, though perhaps, it must be admitted,
less elegant. It also reads as *soliton propagation*, the subject at the heart of John
Travers's research, and one he remains obsessed with and passionate about.

## Use of AI

solprop is written almost entirely by AI coding agents, working under the direction of
its lead developer. This was a deliberate choice, and it is worth being clear about what
it means for anyone relying on the code.

- **Almost all of the code is AI-written.** The physics modules, numerics, backends,
  tests and documentation were produced by AI agents. The lead developer chooses the
  models, sets the standards, reviews the results and decides what is accepted.
- **Correctness is controlled by design.** Every change passes through a carefully
  designed development harness. Each model is derived and documented before it is
  implemented, tested against analytic results and independent references, and checked
  for convergence. Every compiled backend is tested against the NumPy reference within
  rounding bounds derived before any measurement. The lead developer's two decades of
  experience with propagation codes decide what is checked, and how.
- **Fewer bugs, better documentation, less line-by-line reading.** Compared with
  hand-written code of the same scope, we expect solprop to have fewer bugs and much more
  complete documentation. We acknowledge, though, that its individual lines of code have
  had less direct human scrutiny than those of a traditionally written code: the tests,
  more than a human reader, are what vouch for them.
- **More rigorous numerics.** AI makes it practical to derive error bounds, run
  convergence studies and add accuracy checks that a small team would usually have to
  leave out.
- **Complete test coverage.** Every line and branch of the package is exercised by a
  broad test suite of more than 11,500 tests: unit and invariant tests, comparisons with
  analytic solutions and independent references, convergence records, and parity
  between the backends.
- **Faster development.** solprop has far more functionality than its hand-written
  predecessors, developed in a fraction of the time.

The [accuracy records](../dev/accuracy.md) and [testing guide](../dev/testing.md)
describe that evidence and its scope.

If a result looks wrong, report it with a small reproducible script, the package
and backend versions, and the comparison that raised the question. The
[issue tracker](https://github.com/LupoLab/solprop/issues) is the place to do so.

## How solprop relates to Luna

solprop draws on experience with Luna while using its own implementation and
model choices. [Noteworthy differences from Luna](../guide/workflows/luna.md)
records differences in conventions, defaults, formulae, material data and
capabilities. Read it when moving a simulation between the two codes or comparing
their results, and check that the physical and numerical inputs match.

## Citing solprop

If you use solprop in published work, please cite its Zenodo record,
[doi:10.5281/zenodo.23854248](https://doi.org/10.5281/zenodo.23854248), with the
version you used (`python -c "import solprop; print(solprop.__version__)"`). That DOI
names every release, and the record lists each release's own DOI. `CITATION.cff` in
the repository holds the same reference, which GitHub offers as *Cite this repository*.

## Licence

solprop is free software: you can redistribute it and modify it under the terms of the
GNU General Public License, version 3 or (at your option) any later version
(`GPL-3.0-or-later`); see the `LICENSE` file in the repository. This documentation and
its figures are licensed under the Creative Commons Attribution 4.0 International
licence (`CC-BY-4.0`). The chirped-mirror data taken from Luna.jl keep Luna's MIT licence.
Every file in the repository states its copyright and licence, in an SPDX header or in
`REUSE.toml`, following the [REUSE specification](https://reuse.software/).

The optional compiled backend, solprop-native, is built from the repository's `native/`
directory and distributed as a separate package under the same licence.
