# Mode sets and transverse quadrature

A multimode calculation has two spatial choices: which modes to retain, and where to
evaluate the nonlinear response across their shared cross-section. Increasing the number
of modes and increasing the quadrature resolution address different errors. The [RDW
lesson](../../introduction/lessons/plot_01_rdw.rst) shows how to interpret the resulting
modal energies.

## The transverse quadrature

The multimode transform integrates over the cross-section with one **fixed** rule, chosen
once per problem, never an adaptive one. The argument is not economy. The Kerr
polarisation is cubic in the field, so its projection is a *quartic* in the transverse
profiles; those are Bessel functions of $r/a$, analytic, and a Gauss rule of modest order
integrates a quartic of them to rounding. Measured against a 2048-node reference, the
quartic overlap tensor of four $\mathrm{HE}_{1m}$ modes is converged to $2\times10^{-14}$
at $n_r = 32$ — the reference's own rounding, not the rule's.

**Radially**, $n_r$ Gauss–Legendre nodes on $\xi = r/a \in (0,1)$, with the Jacobian and
the azimuthal $2\pi$ folded into the weights. With `radial_rule="gauss_kronrod"` (the
capillary default) the count is rounded up to $2n+1$ and the rule becomes the Kronrod extension of the $n$-point Gauss rule, whose
Gauss subset is an **embedded coarse rule**: the difference of the two projections is an
error estimate that costs nothing, because the two rules share every node the coarse one
has. solprop computes that extension with Laurie's algorithm rather than shipping a table,
so any order is available, and checks it node by node against the Julia package
QuadGK.jl's implementation of the same algorithm.

**In the field scale**, for one Marcatili HE₁₁ mode in a uniform gas, the radial
projection is an integral over $\eta = J_0(u_{01}r/a)$, and `radial_rule="field_scale"`
and `"field_scale_radau"` are the Gauss and Gauss–Radau rules for it: exact for the
linear and Kerr source from two (Radau: three) nodes, the Radau rule with one node on the
axis, and the one-node member the effective-area model. They carry radial area weights,
have no embedded coarse rule, and refuse any other mode set. When to use them, with the
evidence, is in [radial nodes for expensive responses](radial-quadrature.md).

**Azimuthally**, $n_\theta$ equispaced nodes offset by half a step — the periodic
trapezoid rule, which is a Gauss rule on the circle and is exact for trigonometric
polynomials of degree at most $n_\theta - 1$. A quartic product of modes with azimuthal
order up to $h$ reaches degree $4h$, so
{func}`~solprop.modes.quadrature.transverse_quadrature` warns below
$n_\theta = 4h + 1$; the default 16 covers $h \le 3$.

The offset is not cosmetic. On-grid nodes at $0, \pi/2, \pi, 3\pi/2$ annihilate products
such as $E_xE_y$ for a superposition of hybrid modes, so two successive levels of an
adaptive rule both return zero, the level-to-level error estimate is zero, and the
integrator declares convergence on a wrong value. The fixed offset rule cannot be fooled
that way, and has no convergence decision in it to fool.

### What the tensor is worth checking against

The rule's job is one number per pair of modes, and for a set of $\mathrm{HE}_{1m}$
modes that number has a closed form worth stating, because it is the whole of the
intermodal Kerr coupling at leading order. A field that is the fundamental alone drives a
polarisation $\propto \hat e_1(r)^3$, which *factorises*: whatever the pulse, the
response, the shock term or the third harmonic, the drive it projects into mode $p$ is
that of mode 1 times

$$\frac{T_p}{T_1} = \frac{\int \hat e_1^3\,\hat e_p\,\mathrm{d}A}
                          {\int \hat e_1^4\,\mathrm{d}A},
\qquad \hat e_m = \frac{J_0(u_m r/a)}{\sqrt{N_m}} .$$

For the first four modes that is $1$, $0.4785423$, $0.0221346$ and $-0.0023723$ —
geometry, with no gas, no wavelength and no normalisation constant in it. The fourth is
negative because $J_0(u_4r/a)$ changes sign three times inside the bore. The
fundamental's own entry has a closed form too,

$$T_1 = \int \hat e_1^4\,\mathrm{d}A = \frac{4}{(\varepsilon_0c)^2 A_\text{eff}},$$

which is the identity that makes a one-mode multimode run and a mode-averaged one agree.
The transform's overlap tensor reproduces these ratios to better than $10^{-11}$, so
when the higher modes of a multimode $\mathrm{HE}_{1m}$ run look suspicious, the
intermodal Kerr coupling is not the place to look.

### Tapers cost two scalars

For a scale-invariant mode the profile depends on $r$ only through $r/a$ and $N \propto
a^2$, so at nodes placed at $r = a\xi$ the normalised field is exactly proportional to
$1/a$ and the weights to $a^2$. Moving along a taper therefore rescales the matrices
exactly:

$$S(z) = S(0)\,\frac{a(0)}{a(z)}, \qquad W(z) = W(0)\,\frac{a(z)}{a(0)},$$

which is {meth}`~solprop.modes.quadrature.ModeMatrices.rescaled`, and their product — the
thing the transform actually applies — does not move at all.

## Mode sets

{class}`~solprop.modes.collection.ModeCollection` is an ordered set of modes sharing one
cross-section, validated once, because the mode axis of the kernel state is positional and
one transform matrix serves the whole set. It reports the highest azimuthal order and
whether the azimuthal coordinate has to be resolved, and it is a `Sequence[Mode]`, so the
quadrature and the mode matrices take it directly.

For a vector run, {meth}`~solprop.modes.collection.ModeCollection.with_polarisations`
expands a spatial set into the polarisation-resolved one: each hybrid mode is followed by
its partner, the same mode turned by $\pi/(2n)$, which turns the pattern a further
quarter turn from $x$ towards $y$ and so exchanges the two Cartesian components: the
pair runs $x$ then $y$ for $\mathrm{HE}_{1m}$. A transverse mode is carried once —
$\mathrm{TE}_{0m}$ and $\mathrm{TM}_{0m}$ are different modes, not two states of one.

{func}`~solprop.modes.collection.overlap` integrates adaptively with
{func}`scipy.integrate.dblquad`, deliberately *not* on the fixed rule, so that the rule's
orthonormality can be checked against something independent rather than against itself.

## A capillary worked example

The four lowest $\mathrm{HE}_{1m}$ modes of a helium-filled capillary, evaluated
directly from their mode objects: the transverse profiles a multimode run projects onto,
and the loss each carries.

```{eval-rst}
.. plot::
   :filename-prefix: modes-capillary-he1m
   :include-source: False
   :caption: The four lowest HE_1m modes of a 125 um capillary filled with 1 bar of
             helium. Left, the transverse profiles: each is a Bessel function of r/a
             vanishing on the wall, and they are mutually orthogonal, which is what the
             quadrature rule has to reproduce. Right, the loss, which grows as the
             square of the Bessel zero u_1m and as the inverse cube of the radius --
             HE_14 leaks 24 times faster than HE_11.

   import matplotlib.pyplot as plt
   import numpy as np

   from solprop.constants import C
   from solprop.modes import MarcatiliMode
   from solprop.modes.base import db_per_m

   radius = 125e-6
   modes = [MarcatiliMode(radius, "He", 1.0, m=m) for m in range(1, 5)]

   fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.4))

   r = np.linspace(0.0, radius, 400)
   points = np.stack([r, np.zeros_like(r)], axis=-1)
   for index, mode in enumerate(modes):
       profile = mode.field(points)[:, 0] / np.sqrt(mode.normalisation())
       left.plot(r * 1e6, profile / np.max(np.abs(profile)),
                 color=f"C{index}", label=f"HE$_{{1{index + 1}}}$")
   left.axhline(0.0, color="0.7", linewidth=0.8)
   left.set_xlabel(r"$r$ ($\mu$m)")
   left.set_ylabel("normalised field")
   left.legend(fontsize="small", frameon=False)
   left.grid(alpha=0.3)

   wavelength = np.linspace(400e-9, 1000e-9, 300)
   omega = 2 * np.pi * C / wavelength
   for index, mode in enumerate(modes):
       right.semilogy(wavelength * 1e9, db_per_m(mode, omega), color=f"C{index}")
   right.set_xlabel(r"wavelength (nm)")
   right.set_ylabel("loss (dB/m)")
   right.grid(alpha=0.3)

   fig.tight_layout()
```

The zero-dispersion wavelength follows from the balance between the gas and the waveguide,
and moves with pressure:

```{code-block} python
from solprop.modes import MarcatiliMode
from solprop.modes.base import zdw

mode = MarcatiliMode(125e-6, "He", 0.4, model="reduced")
zdw(mode)  # 3.767e-07 m
```

{func}`~solprop.modes.base.zdw` brackets the root of $\beta_2$ and **raises** when there is
none in the window, rather than returning a missing value that would propagate silently
into later arithmetic: a mode with no zero-dispersion point is a fact the caller has to
see.
