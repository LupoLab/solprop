# Assessing numerical accuracy

A solver tolerance controls part of a calculation's numerical error. It does not
choose an adequate time window, resolve a narrow spatial feature, validate a
material model or compensate for omitted modes. Assess accuracy against the
quantity you intend to use: a band energy, pulse duration, conversion efficiency,
complex field or spatial intensity can converge at different rates.

The [soliton lesson](../../introduction/lessons/plot_02_soliton.rst) uses recurrence
and energy conservation as checks. The [crossing-beam
lesson](../../introduction/lessons/plot_04_crossing_beams.rst) compares superposition,
beam motion, fringe spacing and spatial refinement. These checks are useful
because each has a reference independent of the plotted appearance.

## Change one resolution at a time

Keep the physical problem and the measurement fixed while changing its numerical
representation. Compare results at the same propagation positions and with the
same spectral windows and field conventions.

| Control | What it resolves | Useful comparison |
|---|---|---|
| `rtol`, `atol`, and a step cap | Evolution along the propagation direction | Tighten tolerances or reduce `dz_max`; compare the chosen observable |
| Temporal sample spacing and response oversampling | Carrier, pulse structure and nonlinear products | Refine the sampled response while retaining the same physical window |
| Time-window duration | Walk-off, delayed response and pulse tails | Enlarge the window, preserving adequate sample spacing and an unaffected measurement region |
| Frequency band and spectral collar | Generated frequencies and their absorption | Extend the band; inspect energy and signal near the collar |
| Number of modes | Accessible guided fields and intermodal transfer | Add physically relevant modes while resolving their overlap integrals |
| Transverse quadrature | Local nonlinear response within the retained modes | Increase nodes independently of mode count |
| Spatial spacing and extent | Diffraction, fringes, focusing and boundaries | Refine spacing and enlarge the domain as separate checks |
| Operator tables and longitudinal sections | Varying dispersion, geometry and coupling | Tighten table tolerance or refine sections, alongside propagation-step refinement |

An embedded radial indicator compares two related quadrature rules. Both can miss
the same feature, so a small indicator is useful evidence but does not replace an
independent refinement. See [transverse quadrature](../waveguides/mode-sets.md)
and [expensive-response radial rules](../waveguides/radial-quadrature.md).

```{eval-rst}
.. plot::
   :include-source: False
   :filename-prefix: numerical-error-floor
   :alt: Illustrative error curves fall as the propagation step is reduced, then level off at different temporal or spatial resolution floors.
   :caption: Illustrative error budget, not a measured convergence order. Each curve is the sum of a fifth-power step contribution and a fixed grid contribution. Reducing the propagation step cannot remove a temporal or spatial discretisation error. The actual order and floor must be measured for the problem and observable.

   import matplotlib.pyplot as plt
   import numpy as np

   step = np.geomspace(0.01, 1.0, 200)
   fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
   for floor, label, colour in zip(
       (1e-2, 1e-4, 1e-6),
       ("coarse time / spatial grid", "intermediate grid", "finer grid"),
       ("#d55e00", "#0072b2", "#009e73"), strict=True,
   ):
       ax.loglog(step, step**5 + floor, color=colour, lw=2, label=label)
   ax.loglog(step, step**5, color="0.45", ls="--", label="step contribution alone")
   ax.set_xlabel("propagation step / reference length (dimensionless)")
   ax.set_ylabel("illustrative relative error")
   ax.set_ylim(3e-7, 2)
   ax.invert_xaxis()
   ax.grid(which="both", alpha=0.18)
   ax.legend(frameon=False, fontsize=9)
```

## Interpret differences and conservation checks

Use at least three resolution levels when estimating a convergence order. Two
levels show a difference, but cannot by themselves establish the asymptotic rate
or exclude a shared bias. An observed plateau can come from another unresolved
axis, material interpolation, floating-point error or the observable's definition.

Energy conservation is valuable for a lossless model, but a conserving method can
still have phase or pulse-shape errors. With ionisation, Raman energy exchange,
waveguide loss or absorbing boundaries, optical energy need not be conserved.
Check the expected energy accounting and the loss channels instead of treating
every decrease as a solver defect. The [power and intensity
definitions](../results/power-and-intensity.md) specify the measured energy.

For a weak generated band, report both its absolute energy and a relative
difference. A large percentage difference in an almost empty band and a small
percentage error in the total field can describe the same result. For complex
fields, state whether comparisons permit a global phase or time shift; aligning
curves removes those errors from the measurement.

## Tolerances, boundaries and coupled propagation

The [solver error norm](solver.md#the-error-norm) scales a local estimate. Its
tolerance is not a guaranteed global error bound, and equal tolerances in two
methods do not promise equal accuracy. Compare measured errors when choosing
between [Adams and Runge–Kutta](choosing-a-stepper.md).

Keep significant light away from an [absorbing collar](boundaries.md) unless the
observable deliberately measures the remaining field after that absorption. A
smooth absorber reduces wraparound; it also changes any light reaching it.
Increasing absorber strength cannot recover discarded physics.

For a coupled linear problem with no nonlinear source, the current exponential
Adams path has a limitation: its frozen-block residual is not controlled by the
ordinary source error estimate. Refine `dz_max` or use the Runge–Kutta route as
described under [Adams limitations](adams.md#limitations). Tapers also require
convergence of the local modes and their longitudinal interpolation.

Finally, distinguish numerical agreement from physical validity. Two backends
can agree on the same approximation; a refined calculation can still use
extrapolated material data or omit a relevant degree of freedom. The
[material validity chapters](../media/index.md) and the selected geometry's
limitations complete the assessment.
