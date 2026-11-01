# What solprop can do

solprop solves forward pulse-propagation problems in waveguides and free space.
The main model is the unidirectional pulse propagation equation (UPPE). For a
single guided envelope with specified dispersion and nonlinearity, the
generalised nonlinear Schrödinger equation (GNLSE) provides a convenient
description. An envelope calculation is not automatically a GNLSE calculation:
the spatial representation and the response model matter too.

## What question does a propagation calculation answer?

You specify a field at an entrance plane and ask how it evolves forward through a
specified medium. Outputs can include the spectrum after broadening, pulse power
and duration, energy in a chosen ultraviolet band, modal content, a beam's
transverse fluence, and plasma density. Parameter scans repeat that question for
different inputs or structures. The calculation predicts the consequences of a
chosen physical model; it does not infer an experimental geometry or material
response from the desired result.

A useful workflow has four parts:

1. Choose the propagation equation, spatial description and material responses.
2. Set the input pulse or beams, geometry, grids and accuracy controls.
3. Run, saving enough fields and statistics for the observables of interest.
4. Interpret those observables and vary the numerical resolution and model
   assumptions that could affect the conclusion.

The [RDW lesson](lessons/plot_01_rdw.rst) follows all four parts. The detailed
[simulation setup](../guide/setup/index.md) chapters explain each choice.

## Pulses and propagation

A pulse can be described by its real electric field or by a complex envelope
around a reference frequency. The field description resolves the optical carrier;
the envelope description can reduce the sampling needed for a narrowband pulse.
The choice also determines how carrier-sensitive responses are represented.
The [grid chapter](../guide/setup/grids.md) explains the choices, sampling, and
Fourier conventions.

Inputs include Gaussian and sech pulses, measured or custom pulses, and coherent
sums of pulses with different wavelengths, delays, and phases. Spatial runs add a
transverse beam profile. See [input pulses](../guide/setup/pulses.md) and
[polarisation](../guide/setup/polarisation.md).

The UPPE combines forward linear propagation with a nonlinear source. It can
carry a real field or a complex envelope, with a guided-mode expansion or a
spatial grid. The GNLSE route instead describes a single envelope with prescribed
dispersion coefficients and nonlinear strength. The
[N = 3 soliton lesson](lessons/plot_02_soliton.rst) uses its cubic limit to explain
compression and recurrence. In either route, an option being available does not
mean that every response is enabled or valid for every geometry.

The input may have spectral phase, delay, chirp, multiple coherent components,
or one or two transverse polarisation components where supported. General vector
modal guides use their electric and magnetic mode fields to reconstruct physical
observables. Conventions matter when importing a spectrum: solprop's user-facing
spectra use the optics Fourier sign, while custom array launches state the
internal transform they require. See [grids and Fourier conventions](../guide/setup/grids.md)
and [pulses](../guide/setup/pulses.md) before combining data from another code.

## Waveguides and free space

For guided propagation, solprop offers effective-area models, resolved capillary
mode sets, and general modal guides with electric and magnetic mode fields.
Guide models include gas-filled capillaries, antiresonant fibres, step-index and
graded-index fibres, rectangular and planar guides, and supplied modal data.
Tapers and pressure gradients describe changes along the propagation direction.
The [waveguide guide](../guide/waveguides/index.md) explains the assumptions and
supported combinations.

For free-space propagation, radial grids describe cylindrically symmetric beams,
while Cartesian grids resolve both transverse coordinates. A line grid represents
a field uniform in the other transverse direction. These descriptions have
different computational costs, symmetries, and transverse measures; use
[choosing a spatial description](spatial-description.md) to decide between them.
Free-space energy, fluence and intensity are the physical axial Poynting flux in the
medium, so a launch energy is what an energy meter there would read, and a radial gas
pressure gradient is admitted under a certificate that bounds the reflection the forward
model omits ([energy and flux in a medium](../guide/free-space/flux.md)).

| Description | Typical calculation | Main modelling choice |
|---|---|---|
| Effective-area single mode | A fibre envelope or mode-averaged hollow guide | Assume a fixed profile; no modal transfer or local transverse response |
| Resolved capillary modes | Few-cycle compression, RDW emission, Raman or plasma in gas | Retain enough modes and enough transverse quadrature to resolve their combined field |
| General modal guide | Solid-fibre or supplied vector modes | Use the guide's mode fields and physical Poynting observables |
| Radial spatial grid | Symmetric focusing and an axial filament | Assume cylindrical symmetry; test radial resolution and domain radius |
| Cartesian spatial grid | Crossing beams or asymmetric propagation | Resolve both transverse axes and any phase gradients |
| Line or planar description | Propagation with one invariant or guided transverse direction | Track the correct transverse measure, including quantities per unit width |

The [radial filament](lessons/plot_03_filament.rst) and
[crossing beams](lessons/plot_04_crossing_beams.rst) show two different reasons to
resolve space directly. The [examples gallery](../examples/index.md) adds
antiresonant, solid, rectangular and planar guide demonstrations.

## Material responses

Material dispersion, nonlinear coefficients, Raman responses, and ionisation-rate
models come from the optmat material library. The available responses include
instantaneous and frequency-dependent electronic nonlinearity, third-harmonic
generation, rotational and vibrational Raman, photoionisation and plasma, and
shot noise. Gas mixtures can carry several species with their own responses.

Availability depends on the grid, geometry, and response combination. In
particular, cycle-averaged envelope plasma has different assumptions from
carrier-resolved plasma, and not every Raman model supports vector fields.
The [materials and response chapters](../guide/media/index.md) state these limits
and explain how to select a model and inspect its provenance.

The material selection is part of the scientific input. A run records the models
it uses so that results can be traced to their dispersion laws, nonlinear
coefficients, rate models and validity ranges. Gas pressure and temperature affect
density and therefore both linear and nonlinear response. A warning about a
material fit's range should lead to an interpretation check, not merely to a
smaller integration step. See [materials](../guide/media/materials.md)
and [gas mixtures](../guide/media/mixtures.md).

The instantaneous Kerr response describes intensity-dependent refraction;
delayed Raman response describes molecular or material memory; photoionisation
and plasma add a changing electron population and its optical response. These
mechanisms can coexist and influence each other. The [MI ensemble](../auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum.rst)
and [comb-to-continuum calculation](../auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum.rst)
illustrate how different responses lead to different routes to broadband light.

An optional microscopic helium TDSE response is intended for research
configurations with explicit calibration and convergence checks. Its
[manual](../guide/media/tdse.md) explains its current scope and
limitations; it should not be treated as a generally qualified replacement for
the rate-based plasma models.

## Results, workflows, and execution

Runs can save fields and statistics in memory or to HDF5. Processing functions
recover spectra, temporal traces, energies, widths, and other observables; plotting
functions produce propagation maps and slices. Design tools estimate useful
length scales and phase matching, while offline optics and compressors apply
spectral transfer functions. Parameter scans organise repeated calculations.

Start with [results and interpretation](../guide/results/index.md), then
[design and workflows](../guide/workflows/index.md). The
[examples](../examples/index.md) show self-compression, dispersive-wave emission,
supercontinuum generation, Raman dynamics, mode conversion, and beam propagation.

The separate native backend provides fast CPU and GPU execution and is the one to use by
default. The NumPy reference and the slower Numba CPU fallback are included with solprop.
Choose a backend using
the [supported combinations](../guide/setup/model-selection.md#which-backend-runs-what)
and the [backend guide](../guide/performance/backends.md);
the installed native build determines whether it executes on a CPU or GPU.


## Scope and limits

The propagation direction is privileged. These are forward models, so reflected
waves and a general bidirectional Maxwell problem are outside their scope.
A chosen spatial symmetry excludes the corresponding symmetry-breaking dynamics:
a radial calculation cannot predict azimuthal filament breakup, and a one-mode
calculation cannot predict transfer into unrepresented modes. Some guide models
are asymptotic or idealised and require dimensions large compared with wavelength.
Material data and response approximations have their own independent limits.

Numerical accuracy also has several independent controls. A small local step
error does not establish adequate temporal bandwidth, transverse sampling,
quadrature, modal basis size or boundary distance. Conserved energy alone does
not prove a correct pulse shape. Begin with the observable you intend to trust,
then use the [accuracy chapters](../guide/numerics/index.md) to identify the
refinements that can change it.

Backend support is [documented by combination](../guide/setup/model-selection.md#which-backend-runs-what)
of geometry, grid and response.
A faster backend does not introduce a different physical model, and an
unsupported combination is refused rather than silently substituted. Research
features such as the optional TDSE response require the additional calibration
and convergence work stated in their own chapters. Qualification
reports are evidence for specified cases, not universal accuracy guarantees.

Continue with [choosing a spatial description](spatial-description.md), then
use the [first lesson](lessons/plot_01_rdw.rst) to put these choices together.
