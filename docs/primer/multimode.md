(primer-multimode)=
# 5. Many modes, and guides that change

A single mode with an effective area is a good model only as long as the light stays in
that mode. Often it does not. A capillary pumped near its damage threshold self-focuses
and fills its higher-order modes. A plasma defocuses the beam and pushes energy outwards.
A taper converts the fundamental mode into its neighbours. In a graded-index fibre, tens
of modes with nearly equal propagation constants exchange energy constantly. Modelling
any of these needs a set of modes that interact through the nonlinearity and, where the
guide changes or the wall couples them, linearly as well.

This chapter covers how that is done. It starts with the overlap-tensor formulation of
the multimode GNLSE and then sets beside it the approach solprop takes, which evaluates
the response on a transverse quadrature. You will see why the two are identical for
Kerr, and why solprop's version also handles ionisation and plasma. You will see how
many quadrature nodes an ionisation calculation needs, and how the coherent wall and the
taper coupling make the modes exchange energy linearly. The chapter ends with the
general modal route for solid-core fibres, with its full vector modes and flux metric.

## 5.1 The overlap-tensor formulation

Poletti and Horak (2008) derived the multimode generalised nonlinear Schrödinger equation
from the same coupled-mode projection as Kolesik and Moloney (2004). For mode $p$ with
transverse field $\mathbf F_p$ and normalisation $N_p$, the nonlinear source is the
projection

$$
\frac{\partial A_p}{\partial z}=\frac{i\omega}{4N_p(\omega)}\int\mathbf F_p^*\cdot\tilde{\mathbf
P}_{\rm NL}\,e^{-i\beta_pz}\,\mathrm dx\,\mathrm dy
$$

(their Eq. 5). The derivation separates the forward and backward fields and neglects the
interactions between them, "but without any restriction on the bandwidth". Inserting a
Kerr and Raman polarisation built from the modal expansion, and evaluating the mode
profiles at the carrier, gives sums over three further mode indices weighted by overlap
integrals,

$$
Q_{plmn}=\frac{\varepsilon_0^2n_0^2c^2}{12}\,
\frac{\int[\mathbf F_p^*\cdot\mathbf F_l][\mathbf F_m\cdot\mathbf F_n^*]\,\mathrm dx\,\mathrm dy}{N_pN_lN_mN_n}
$$

(their Eqs. 6–7). Wright et al. (2018, Eqs. 1–2) write the same equation for real scalar
modes with $S_{plmn}=\int F_pF_lF_mF_n/(\prod\int F^2)^{1/2}$, the multimode generalisation
of $1/A_{\rm eff}$.

The formulation is elegant, but its cost grows quickly. Each step evaluates three nested
sums over all modes for every mode, which is $O(M^4N)$ for $M$ modes on $N$ time samples.
Symmetry removes many coefficients: for 10 modes of a small step-index fibre, 17,872 of the
20,000 vanish identically (Poletti and Horak 2008, §4A). Even so, the measured scaling
remained about $M^{3.4}$. Wright et al. (2018, §III D) put the break-even at "about 10–30"
modes, beyond which full-field methods "become increasingly advantageous". The formulation
also has two structural limits. Only cubic responses fit the tensor form, so ionisation
cannot be written this way. And the frequency dependence of the mode shapes is kept only to
first order, through per-coefficient shock times, which Poletti and Horak themselves say
"cannot be justified a priori".

## 5.2 solprop's approach: synthesise, evaluate, project

solprop does not build an overlap tensor. At each evaluation of the nonlinear term, it:

1. **synthesises** the physical field at a set of quadrature nodes across the cross-section
   from the modal amplitudes, $E(\mathbf r_q,t)=\sum_ma_m(t)e_m(\mathbf r_q)$;
2. **evaluates** the material response pointwise in time at every node: Kerr, Raman,
   ionisation and plasma, whatever the run declares;
3. **projects** the polarisation back onto each mode with the quadrature weights and the
   prefactor of §2.9.

The synthesis and the projection are matrix products, two matrix multiplications per
evaluation, skinny in the number of modes and wide in time
([mode sets and transverse quadrature](../guide/waveguides/mode-sets.md)). For the Kerr term
on a field grid, the result is exactly the tensor contraction:

$$
\frac{\partial a_p}{\partial z}\bigg|_{\rm NL}=\frac{i\omega}{4\operatorname{Re}n_{{\rm eff},p}}\,
\varepsilon_0\chi^{(3)}\sum_{lmn}\Gamma_{plmn}\,\mathcal F[a_la_ma_n],
\qquad\Gamma_{plmn}=\int\hat e_p\hat e_l\hat e_m\hat e_n\,\mathrm dA
$$

([several modes at once](../guide/media/nonlinear.md#several-modes-at-once), written here in
the optics sign, with $\chi^{(3)}=\rho\gamma^{(3)}$ and $\hat e$ the vacuum-normalised profiles of §4.2). The same algebra, done in a different order, gives the same answer: the
cube of a sum is the sum of the cubes and cross terms. The difference lies in cost and
generality. The work grows as the number of modes times the number of nodes, not as $M^4$,
and the response at a node can be any function of the local field.

Three properties make this accurate for the cubic responses.

- **The algebra is the tensor contraction.** Synthesising, cubing and projecting is algebraically
  the contraction with the overlap tensor defined by the same quadrature.
- **The radial rule converges rapidly.** The projection of a cubic response is a quartic in the
  mode profiles. For a capillary those are Bessel functions of $r/a$, which are analytic, so a
  Gauss rule converges quickly, though not exactly at any finite order. For four low
  $\mathrm{HE}_{1m}$ modes the overlap tensor is converged to $2\times10^{-14}$ with 32 radial
  nodes; more oscillatory, higher-order modes need more nodes, so refine the rule with the mode set. The default rule is Gauss–Kronrod. Its embedded
  Gauss subset gives an error estimate at no extra cost, because the two rules share nodes.
- **The azimuthal rule is chosen to be exact.** For modes with azimuthal order up to $h$, a
  quartic product has angular degree $4h$. An offset periodic trapezoid rule with
  $n_\theta\ge4h+1$ nodes integrates it exactly. The offset matters: nodes placed at
  $0,\pi/2,\pi,3\pi/2$ would annihilate products such as $E_xE_y$ and fool any adaptive
  scheme into accepting a wrong value.

Because the modes are multiplied at the same instant at every node, they must share one
co-moving frame (§2.6). The coupling between them can be checked in closed form. A weak
seed in a second mode, driven by a pump in the first, grows in a ratio to the pump's own
nonlinear change that depends only on the overlap ratio, the index ratio and the phase mismatch.
Against the first-order solution that keeps the pump's own dispersion, solprop's test suite
agrees to better than $10^{-5}$.

## 5.3 How many nodes does a response need?

For Kerr and Raman the answer is "enough to integrate a quartic", which is few. For
ionisation it is different, because the rate is a steep, non-polynomial function of the
local field. For one $\mathrm{HE}_{11}$ mode in a uniform gas, the field at radius $r$ is the
on-axis waveform scaled by $\eta=J_0(u_{01}r/a)$. The projected source is therefore an
integral over this **field scale**,

$$
S(t)=\int_0^1F[\eta A](t)\,\mathrm d\mu(\eta),\qquad\mathrm d\mu=J_0(u_{01}\xi)\,\xi\,\mathrm d\xi ,
$$

where $F$ is the response of one atom to a given field
([radial nodes for expensive responses](../guide/waveguides/radial-quadrature.md)). The
mode-averaged model of Chapter 3 turns out to be the one-node rule of this integral. It
evaluates a single response at $\eta_e=0.752$, where the intensity is 57% of the on-axis
value, weighted to make the Kerr term exact. For a Kerr response that is all that is needed.
For an ionisation-driven source it is a poor quadrature. On a test case of a 1 mJ, 6 fs pulse
in helium, it gets the source at the entrance 85% wrong and the generated 120–200 nm energy
30% wrong.

Rules built on the field scale converge exponentially. Five or six nodes give 0.1% for a
rate model. A time-dependent Schrödinger response has fine structure along $\eta^2$ and needs
more, around 15–20 nodes. The default capillary rule, Gauss–Kronrod with 33 nodes, is
generous for rate models and adequate for that one case. These numbers come from one
waveform, and the guide page says so. The lesson is general, though: the transverse
resolution needed is set by the most nonlinear response, and no tolerance on the
$z$-integration can make up for too few nodes.

## 5.4 The coherent wall

The Marcatili model of Chapter 4 gives each mode its own loss, independent of the others.
That is right only on average. The wall is where the loss happens, and two modes that graze
it at the same time exchange amplitude there with a definite phase. Andreasen and Kolesik
(2013, PRE) made the physical point by comparison. In mid-infrared filamentation in a capillary,
the standard leaky-mode expansion overestimates the loss of nonlinearly reshaped beams. A
beam that has pulled away from the wall loses less than the sum of its modal losses says it
should. Their remedy was a simulation that resolves only the bore and represents the glass by
a semitransparent boundary. The field leaving the bore obeys an outgoing Robin condition,
$\partial_rE=ik_oE$ with $k_o=k\sqrt{n^2-1}$ for the field tangential to the glass and $k_o/n^2$
for the field normal to it (their Eqs. 10–15; Andreasen and Kolesik 2013, JLT, Eqs. 6, 9 and
18–19).

solprop's **coherent wall** takes this boundary condition and condenses it onto the modal basis
([the coherent wall](../guide/waveguides/coherent-wall.md#the-coherent-wall)). For the
$\mathrm{HE}_{1m}$ channels of a capillary, the linear generator becomes

$$
G=D+H,\qquad D=-\frac{i}{2ka^2}\operatorname{diag}(u_m^2),\qquad H=g(a)\,hh^T,
$$

written in the optics convention, where $h_m$ is the radial derivative of each orthonormal
basis function at the wall and $g(a)$ carries the wall's impedance. Instead of a separate
imaginary part for each mode, the wall becomes a rank-one complex-symmetric matrix that couples
every pair of modes with a definite phase. The condensation, a truncated Mittag-Leffler
expansion of the exact root equation, is solprop's own; Andreasen and Kolesik solved the
boundary condition on a grid. With a single channel it reproduces Marcatili's loss. With many,
it describes the coherent exchange. solprop records three validity ratios for each run, for
the paraxial, grazing-incidence and quasi-static assumptions, and refuses a mode set that
violates them.

The **vector wall** goes one step further. Marcatili's hybrid-mode loss factor is the mean of
the TE and TM factors (Chapter 4). That mean is exact when the wall's two *impedances* are
averaged, and it hides a coupling between the hybrid families when they are kept apart. The
vector wall keeps the TE and TM impedances separate, and its generator is rank two in each
azimuthal order ([vector boundary response](../guide/waveguides/vector-wall.md)). solprop's
guide also notes that, in the continuum limit, the field average of Andreasen and Kolesik's
Eq. 15 amounts to averaging admittances instead, which would make the hybrid loss too low by
a factor $4n^2/(1+n^2)^2\approx0.87$. That is solprop's reading of the paper's equation; the
paper itself, with its finite grid, reports excellent agreement with a resolved cladding.

## 5.5 Tapers and the connection between modes

When the radius of a capillary changes along $z$, the modes change shape with it, and a field
that was purely $\mathrm{HE}_{11}$ is no longer purely the local $\mathrm{HE}_{11}$ a short distance
later. In coupled local-mode theory this appears as a **connection**,
$S_{mn}=\langle\psi_m|\partial_z\psi_n\rangle$, which transfers amplitude between the local modes.
For the $\mathrm{HE}_{1m}$ modes of a capillary it has a closed form,

$$
C_{mn}=\frac{a'}{a}\,\frac{2s_ms_nu_mu_n}{u_m^2-u_n^2}\quad(m\ne n),\qquad C_{nn}=0,
$$

with $s_m=\operatorname{sgn}J_1(u_m)$
([a guide that varies along z](../guide/waveguides/coherent-wall.md#a-guide-that-varies-along-z)).
In the true-power amplitude it is real and antisymmetric, so it moves power without creating or
destroying it. solprop applies this **taper coupling** to every multimode capillary taper by
default, with or without the coherent wall
([mode conversion in a capillary taper](../guide/waveguides/taper-coupling.md)). The effect is
not subtle. A taper from 50 to 40 µm radius over 0.2 mm leaves 3.8% of the energy in
$\mathrm{HE}_{12}$ and 3.3% in $\mathrm{HE}_{13}$. Against a scalar finite-element calculation, a 125
to 62.5 µm taper over 1 cm moves 36% out of $\mathrm{HE}_{11}$, and the taper coupling reproduces
that to a fraction of a per cent. The cost of a non-adiabatic taper is this conversion into
lossier higher modes, more than the extra loss of the smaller radius.

A pressure gradient needs no connection, because the gas changes the index but not the shape
of the modes. Chapter 12 explains how the linear coupling is propagated. Because the coupled
generator is not diagonal, its exponential is not a simple product of phases. solprop applies
it as a matrix function, section by section along the guide.

## 5.6 The general modal route: vector modes

The scalar routes above treat each mode as a single transverse field component with a vacuum
normalisation. That is excellent for large hollow cores, where the longitudinal field is
negligible. It fails for small solid cores, nanowires and high-index-contrast guides. Afshar and
Monro (2009) and Turner, Monro and Afshar (2009) showed what changes in a full-vectorial model.
The modes have significant longitudinal components $e_z$, so the transverse electric fields of
different modes are no longer orthogonal (their Eq. 36). The correct normalisation is the
Poynting flux, which involves the magnetic field. The Kerr overlap acquires a $|\mathbf e\cdot\mathbf
e|^2$ term. For silicon nanowires, the nonlinear coefficient can be twice the standard value.
Lægsgaard (2012) gives the corresponding equations for tapers, with careful attention to how the
normalisation enters when it varies along $z$.

solprop's **general modal route** implements this
([general modal propagation](../guide/waveguides/general-modal.md)). A provider supplies, for
each channel and at every frequency of the grid, the full three-component $\mathbf E$ and $\mathbf
H$ at its own quadrature nodes. These are normalised to unit Poynting flux, through the Hermitian
flux metric

$$
W_{mn}=\tfrac14\int(\mathbf e_n\times\mathbf h_m^*+\mathbf e_m^*\times\mathbf h_n)_z\,\mathrm dA .
$$

The field is synthesised from all channels at every node, the material's own vector response is
evaluated there, and the result is projected back with

$$
\frac{\mathrm da_m}{\mathrm dz}\bigg|_{\rm NL}=\frac{i\omega}{4}[W^{-1}]_{mn}\int\mathbf
e_n^*\cdot\tilde{\mathbf P}\,\mathrm dA
$$

([vector fields, nonlinear projection and power](../guide/waveguides/modal-projection.md)),
in the optics sign. This is Poletti and Horak's Eq. 5 with unit flux and a non-diagonal metric.
Each input frequency uses its own mode shape, so the four-frequency structure of the interaction
that Lægsgaard (2007) identified is carried exactly, not linearised. The vector Kerr polarisation
has a real-field form, $\varepsilon_0\chi^{(3)}(\mathbf E\cdot\mathbf E)\mathbf E$, and an
analytic-signal form,
$\tfrac14\varepsilon_0\chi^{(3)}(2|\mathbf E|^2\mathbf E+(\mathbf E\cdot\mathbf E)\mathbf E^*)$,
which equals Afshar and Monro's Eq. 28. On a 250 nm silica strand, the vector nonlinear coefficient
comes out 25% above $n_2\omega/(cA_{\rm eff})$. For a weakly guiding 4 µm core the two agree to
$5\times10^{-4}$.

The route has its own limits, which solprop states. It needs a field grid, a lossless bound basis
and one polarisation state per channel. It does not take plasma. A general modal step costs
about a hundred times as much as a fused scalar step. Declared bases are supported: a circular
basis diagonalises the polarisation dependence of the isotropic Kerr tensor, so that
self-phase modulation falls to $\tfrac23$ of its linear-polarisation value, as in Poletti and
Horak's Eq. 14 ([circular and coupled modal
bases](../guide/waveguides/modal-bases.md)). A provider whose guide changes along $z$ supplies
the connection $\Gamma=W^{-1}S$ with $S_{mn}=\langle\psi_m|\partial_z\psi_n\rangle$, which moves no
power by itself ([general modal guides that vary along the
propagation](../guide/waveguides/moving-modal.md)). It plays the same role as the
$-\tfrac12\partial_z\ln w$ term of the scalar routes, and as the $-\tfrac12\partial_zN_m$ term of
Lægsgaard (2012, Eq. 26).

:::{admonition} What to remember
:class: tip

- The overlap-tensor multimode GNLSE costs $O(M^4N)$, handles only cubic responses, and keeps the
  frequency dependence of the modes to first order.
- solprop synthesises the field at quadrature nodes, evaluates the response pointwise and projects
  back: two matrix products per evaluation. For Kerr this is identical to the tensor contraction;
  for ionisation it is the only option.
- Offset trapezoid azimuthal rules with $n_\theta\ge4h+1$ integrate the Kerr projection exactly in angle;
  fixed Gauss radial rules converge rapidly and must be refined with the mode set.
- The effective-area model is the one-node rule in the field scale, at 57% of the axial intensity.
  Ionisation needs several nodes or more; tighter $z$-tolerances cannot compensate.
- The coherent wall condenses Andreasen and Kolesik's outgoing wall onto the modes, so the wall
  loss couples them with a definite phase. The vector wall keeps TE and TM apart.
- Capillary tapers convert between modes through an antisymmetric connection. It is on by default.
- The general modal route carries full vector $\mathbf E$ and $\mathbf H$, normalises by the flux
  metric $W$, and projects with $W^{-1}$, at a cost of about a hundred scalar steps per step.
:::

The precise statements are in [mode sets and transverse
quadrature](../guide/waveguides/mode-sets.md), [radial nodes for expensive
responses](../guide/waveguides/radial-quadrature.md), [coherent capillary
walls](../guide/waveguides/coherent-wall.md), [multimode
capillaries](../guide/waveguides/capillaries.md) and [solid-core
fibres](../guide/waveguides/solid-fibres.md). [Chapter 6](free-space.md) removes the waveguide
altogether.
