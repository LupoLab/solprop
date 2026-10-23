(primer-accuracy)=
# 13. Knowing when to believe a result

A simulation always produces a number. The question this chapter addresses is when that
number deserves to be believed. The answer has two independent halves. The **numerical**
question is whether the equation has been solved accurately with the chosen
representation. The **physical** question is whether the chosen equation, representation
and material data describe the experiment. A converged answer to the wrong model is still
wrong, and an excellent model solved too coarsely is no better.

Dudley, Genty and Coen (2006, §IV.B) put the numerical half in one sentence: "it is essential
always to test the simulation fidelity through a convergence study. This involves
methodically changing the computational resolution". The chapter covers how to do that in
solprop, what a tolerance does and does not control, what conservation proves, and how
solprop's backends and its comparison with Luna are qualified.

## 13.1 What a tolerance controls

The tolerance of an adaptive integrator controls the local error of each step in $z$, scaled as
described in §12.5. It controls nothing else. In the guide's words, it "does not choose an
adequate time window, resolve a narrow spatial feature, validate a material model or compensate
for omitted modes" ([assessing numerical accuracy](../guide/numerics/accuracy.md)). Even in $z$, the
relation between the local tolerance and the global error is loose. The global error a tolerance
buys can miss the request by one to three orders of magnitude in either direction, and equal
tolerances in two methods do not promise equal accuracy. solprop records each run's measured control,
including the largest and mean scaled norms and the rejection rate, so that runs can be compared by
their errors, not their tolerances.

The guide's illustrative error budget makes the point. The total error of a run is the sum of a step
contribution, which falls as a power of the step, and a fixed contribution from every other resolution.
Refining the step drives the first towards zero and then reaches a floor set by the second. Below that
floor, a tighter tolerance only costs time.

## 13.2 One resolution at a time

Each axis of the representation has its own control, and each must be refined separately while the
physical problem and the observable are held fixed
([change one resolution at a time](../guide/numerics/accuracy.md#change-one-resolution-at-a-time)):

| control | what it resolves |
|---|---|
| tolerance or a step cap | evolution along $z$ |
| sample spacing and `ffac` | the carrier, pulse structure and nonlinear products (Chapter 11) |
| window duration | walk-off, delayed responses, pulse tails |
| frequency band and collar | generated frequencies and their absorption |
| number of modes | intermodal transfer (Chapter 5) |
| transverse quadrature | the local response within the retained modes (Chapter 5) |
| spatial spacing and extent | diffraction, focusing, boundaries (Chapter 6) |
| operator tables and sections | a varying guide and its coupling (Chapter 12) |

Use at least three levels of each when estimating a convergence order. Two levels show a difference but
cannot establish the asymptotic rate, or exclude a bias shared by both. An observed order that is
implausibly high is a warning, not a success. It usually means that the levels are not yet asymptotic,
or that another axis is limiting. One more test deserves mention because it is cheap and probes the
envelope representation directly. Kolesik and Moloney (2004, Sec. VI G) suggest repeating a run with a
shifted reference frequency and not trusting the parts of the spectrum that change (§2.7).

## 13.3 What conservation proves

Energy or photon-number conservation is the most popular check of a propagation code, and the weakest. A
conserving method can still have phase errors, timing errors and pulse-shape errors. Heidt (2009) noted
that the per-bin errors behind a photon-number check "can assume positive and negative values ... which can
cancel each other when integrated". With ionisation, Raman scattering, waveguide loss or absorbing collars,
optical energy is not conserved at all, and the useful check is an energy budget. solprop's statistics
support one, with energies by mode and band, species and plasma diagnostics and material absorption, but
they are not a complete ledger of every loss channel: attributing the loss to the wall, the collars, Raman
and ionisation separately needs additional accounting or controlled comparisons, such as runs with a
response switched off.

A sharper check compares the full complex field with a known answer. A higher-order soliton recurs after one
soliton period, and its field at the recurrence carries a definite global phase, $\pi/4$ at $z_0=\tfrac\pi2L_D$.
solprop's tests assert that phase rather than fitting it. Metrics that compare only intensities, such as
Hult's (2007, Eq. 13), understate the error by ten to a hundred and forty times
([traps](../dev/stepper.md#10-traps-each-silent-each-measured); the [N = 3 soliton
lesson](../introduction/lessons/plot_02_soliton.rst) shows the recurrence). Kolesik and Moloney's frequency-shift
test, the soliton recurrence and an independent refinement of each resolution are all evidence of the same
kind: comparisons against something the plotted result does not already contain.

For a weak observable, such as a dispersive wave far below the pump, report both its absolute and its relative
change. A large percentage difference in an almost empty band and a small percentage error in the whole field
can describe the same result. The global error norm of §12.5 gives such a band no control of its own, so it is
always the observable to converge explicitly.

## 13.4 Numerical agreement is not physical validity

Two calculations that agree can share the same approximation. A converged run can still use material data
outside their qualified range (Chapter 7), a rate model with a stated uncertainty factor of 1.3 to 3 (Chapter 10), an
antiresonant fit outside its validated geometry (Chapter 4), a radial symmetry the experiment breaks (Chapter 6),
or a mode set that omits the mode the energy goes into (Chapter 5). The checks for these are physical sensitivity
studies: vary the uncertain input by its plausible range and see whether the conclusion survives.

## 13.5 Backends

solprop runs the same problem on a NumPy reference, a Numba CPU backend and a compiled native backend for CPUs and
GPUs. A faster backend introduces no different physics, and an unsupported combination is refused, not quietly
substituted. Agreement between backends, and between repeats, is not bitwise, and is not required to be: NumPy contracts complex multiplies into fused
multiply-adds and Numba does not, and a GPU orders its arithmetic differently. So every comparison is held to a bound
derived in advance from the operations involved, as a multiple of the machine precision. For example, the bound
for a full adaptive soliton is 1600 times the precision, against a measured deviation of $6\times10^{-16}$
([backend equivalence](../dev/accuracy.md#7-backend-equivalence)). The comparisons are against nontrivial
reference fields, so a backend returning zeros fails by disagreeing with them; exact agreement is
an acceptable outcome. (A different order of arithmetic is allowed; the backend contract requires
the reductions themselves to be deterministic.) Backend parity qualifies the implementation of the
equations. It cannot qualify the equations.

## 13.6 Comparison with Luna

solprop draws on Luna's experience, and it is compared with Luna wherever the two model the same physics
([validation against Luna](../guide/workflows/luna.md#validation-against-luna)). At the level of kernels the
agreement is close: plasma currents to $5\times10^{-14}$, capillary indices to $3\times10^{-10}$, linear operators
to better than $10^{-10}$. At the level of whole runs, one case study is instructive.

On a four-mode, 1.5 m resonant-dispersive-wave capillary with a pressure gradient to vacuum, the energy in the
second mode at the exit differed from Luna's production run by a factor of 0.56. That looks like a disagreement
about physics. It was not. Luna's production tolerance is fixed at $10^{-6}$, and refining it to $10^{-9}$ moved
Luna's own answer by a factor of about 1.76 in that mode. Against the converged Luna, the two codes agree to half a per
cent, while solprop's answer moves by $2.6\times10^{-6}$ over three decades of tolerance
([milestone M2](../dev/accuracy.md#5-milestone-m2-the-rdw-example-four-modes-15-m-gradient)). An unconverged
reference can masquerade as a disagreement. The same study found that this case converges only at order 1.4–1.5,
not five, because the pressure gradient's derivative diverges at a true vacuum end.

The deliberate differences between the codes are listed in the guide
([more accurate numerics, the same physics](../guide/workflows/luna.md#more-accurate-numerics-the-same-physics)).
Most of them have come up in this primer:

- an exact cumulative operator instead of a right-endpoint rule (§12.3);
- an analytic group delay (§2.6);
- absorbers as rates (§12.7), with a minimum collar width (§11.5);
- a zero prefactor outside the band (§2.9);
- band-exact plasma integrals (§10.4);
- exact Parseval energies (§11.6).

## 13.7 How solprop verifies itself

solprop's tests follow a ladder, from the simplest problem with an exact answer to the full equation
([verification ladder](../dev/stepper.md#11-verification-ladder-for-solprops-solver-tests)):

1. the linear propagation alone, against direct quadrature of $\Phi$, to $10^{-14}$;
2. the soliton recurrence, with its phase, at the integrator's order;
3. analytic $z$-varying dispersion and loss;
4. the full UPPE against a three-level Richardson extrapolation, with the order gate of §12.9.

Each ladder writes a convergence record that is compared on every run, so that a change to the code that degrades
convergence is caught. These records are evidence for specified cases, not universal accuracy guarantees. The
[accuracy records](../dev/accuracy.md) and the [testing guide](../dev/testing.md) describe their scope.

:::{admonition} What to remember
:class: tip

- A tolerance controls only the local error in $z$. Its global error can miss by orders of magnitude, so compare
  measured errors, not tolerances.
- Refine each resolution separately, at three levels, with the observable fixed. Treat an implausibly high observed
  order as a warning.
- Conservation is necessary, not sufficient. Compare complex fields, with phase, against independent references:
  recurrence, frequency-shift tests, refinement.
- Converge weak observables explicitly. The global norm gives them no control of their own.
- Numerical agreement is not physical validity. Test the sensitivity to material data, rates, geometry and symmetry.
- Backends are held to bounds derived in advance. Parity qualifies the implementation, not the physics.
- An unconverged reference can look like a disagreement: converge both codes before comparing them.
:::

The precise statements are in [assessing numerical accuracy](../guide/numerics/accuracy.md), [noteworthy
differences from Luna](../guide/workflows/luna.md), [the accuracy records](../dev/accuracy.md) and [the testing
guide](../dev/testing.md). [Chapter 14](synthesis.md) puts the whole primer together.
