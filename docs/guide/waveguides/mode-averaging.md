# Mode averaging

Mode averaging follows one temporal amplitude while replacing the transverse
nonlinear response by an effective-area model. Use it when one guided mode
carries the pulse and a resolved response across the core is unnecessary. It is
the spatial description used by the [GNLSE](gnlse.md), and
`prop_capillary(..., mode_average=True)` selects it for a single capillary mode.

A **resolved single-mode** calculation is different: it retains one propagated
amplitude but evaluates the response at multiple points across that mode.
`prop_capillary(modes=1)`, the default, selects this resolved calculation. The
distinction matters for ionisation and other strongly intensity-dependent responses.
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

The GNLSE entry point selects mode averaging automatically. For a gas-filled
capillary or a supplied gas-filled mode, pass `mode_average=True` to
`prop_capillary` or `prop_waveguide`, or to `Simulation.from_capillary` or
`Simulation.from_waveguide`:

```python
from solprop.driver import prop_capillary

output = prop_capillary(
    125e-6,
    1e-5,  # a short configuration check, in metres
    "Ar",
    1.0,  # bar
    wavelength=800e-9,
    wavelength_limits=(500e-9, 1200e-9),
    trange=200e-15,
    fwhm=30e-15,
    energy=1e-9,
    plasma=False,
    raman=False,
    backend="numpy_ref",
    mode_average=True,
)
assert output.metadata.simulation_kind == "mode_averaged"
```

The run carries one mode, so `modes` greater than one is refused, as are `n_r`,
`n_theta` and `radial_rule`, which shape a transverse rule the run does not have.
The default statistics are the capillary's own, plasma and species included, less
the quadrature estimate. Everything else (a tapered bore, a pressure gradient, a
tube lattice, an envelope grid) runs as it does resolved. Luna.jl makes the
opposite default choice: its `prop_capillary` is mode-averaged unless it is given
a mode count ([coming from Luna.jl](../workflows/luna.md)).

The [MI ensemble](../../auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum.rst)
and [comb-to-continuum example](../../auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum.rst)
use this switch at full length, retaining plasma and, for hydrogen, Raman.
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
