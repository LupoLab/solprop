# Mode averaging

Mode averaging follows one temporal amplitude while replacing the transverse
nonlinear response by an effective-area model. Use it when one guided mode
carries the pulse and a resolved response across the core is unnecessary. It is
the spatial description used by the [GNLSE](gnlse.md) and can also be selected
for a single capillary mode.

A **resolved single-mode** calculation is different: it retains one propagated
amplitude but evaluates the response at multiple points across that mode.
`prop_capillary(modes=1)` selects this resolved calculation. The distinction
matters for ionisation and other strongly intensity-dependent responses.
The [spatial introduction](../../introduction/spatial-description.md) compares
both descriptions with multimode and free-space propagation.

## What the effective area preserves

For a transverse profile $e(x,y)$, the effective area is

$$A_{\mathrm{eff}}=
\frac{\left(\int |e|^2\,dA\right)^2}{\int |e|^4\,dA}.$$

This definition preserves the single-mode cubic Kerr overlap. It does not
preserve every local intensity or the spatial integral of an arbitrary function
of intensity. In particular, replacing a peaked profile by an equivalent uniform
area changes the field sampled by an ionisation rate.

```{figure} ../figures/mode_averaged.svg
:width: 100%
:alt: A capillary HE11 profile beside a uniform disc with the same effective area, and their intensity profiles along a diameter.

The effective-area model preserves the cubic self-coupling integral. The field
near the axis and the spatially integrated ionisation response need not agree.
This is a comparison of model profiles, not a propagated result.
```

A prescribed single-mode profile cannot deform through self-focusing or populate
another guided mode. Use a [multimode capillary](capillaries.md) or
[free-space calculation](../free-space/beams.md) when that evolution matters.

## Select it explicitly

The GNLSE entry point selects mode averaging automatically. For a capillary,
construct a `Simulation` and remove the transverse quadrature. The statistics
must also be selected for the mode-averaged representation:

```python
from dataclasses import replace

from solprop.driver import Simulation
from solprop.stats import default_spec

simulation = Simulation.from_capillary(
    125e-6,
    1e-5,  # a short configuration check, in metres
    "Ar",
    1.0,  # bar
    lambda0=800e-9,
    wavelength_limits=(500e-9, 1200e-9),
    trange=200e-15,
    fwhm=30e-15,
    energy=1e-9,
    modes=1,
    plasma=False,
    raman=False,
    backend="numpy_ref",
)
simulation.settings = replace(
    simulation.settings,
    quadrature=None,
    components=None,
    stats=default_spec("mode_averaged"),
)
assert simulation.translate().flags.mode_averaged
```

The [MI ensemble](../../auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum.rst)
and [comb-to-continuum example](../../auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum.rst)
use this configuration at full length, retaining plasma and, for hydrogen, Raman.
Their saved metadata identifies `simulation_kind="mode_averaged"` and records no
transverse quadrature. Those examples illustrate the chosen approximation; they
do not establish agreement with a spatially resolved plasma calculation.

## Interpret the amplitude and the intensity separately

The input and saved modal amplitudes are normalised to true power. In a
monochromatic scalar mode, $b=\sqrt{\operatorname{Re}n_{\mathrm{eff}}}\,a$,
where $a$ multiplies the vacuum-normalised transverse profile internally.
For a pulse, broadband power and energy use the definitions in
[power and intensity](../results/power-and-intensity.md).

An effective-area intensity and the mode's on-axis intensity answer different
questions. The nonlinear response in a mode-averaged calculation samples the
former. A statistic reconstructed from the transverse mode may report the
latter. Do not interpret a reported peak intensity as proof that the response
resolved that peak across the core.

To assess this approximation, compare with a resolved single-mode calculation
and converge its radial rule. To assess transfer into other modes, increase the
retained mode set as a separate check. Decreasing the propagation tolerance
cannot correct a missing transverse degree of freedom. See
[mode sets and quadrature](mode-sets.md) and
[radial nodes for expensive responses](radial-quadrature.md).
