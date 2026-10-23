# SEP 8 — Second-order nonlinearity and birefringent crystal optics

| | |
|---|---|
| **SEP** | 8 |
| **Title** | Second-order nonlinearity and birefringent crystal optics |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; GPT-6 (2026-10-22 coordination with SEP 31; 2026-10-23 split to SEP 34) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | — |
| **Requires** | SEP 3 for the optmat dependency and material identity contract; crystal data extensions additionally required |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add a tensor second-order polarization and a vector, forward-propagating bulk
crystal route for second-harmonic generation (SHG), sum- and difference-frequency
generation (SFG/DFG), optical parametric amplification (OPA), and a separately
qualified statistical model of optical parametric generation (OPG). Begin with
beta-barium borate (BBO) and lithium triborate (LBO), representing respectively
uniaxial and biaxial optics, without encoding either material in backend kernels.
Derive the linear modes, nonlinear source projection, longitudinal field, interfaces,
and energy flux consistently from Maxwell's equations. Support real-field and
broadband analytic-envelope grids, plane waves, one transverse coordinate, and
two transverse coordinates; retain diffraction, spatial walk-off, group-velocity
mismatch, depletion, back-conversion, and noncollinear mixing. optmat owns material
data and their provenance; solprop owns geometry and propagation. Qualification
requires analytic mixing limits, Maxwell eigenmode and interface checks,
conservation laws, material-specific measurements, and parity on NumPy, Numba CPU
and Kokkos CPU. This is a substantial new route, not just an additional pointwise
response.

### 2. Motivation and scope

#### 2.1 Concrete targets

The initial problems are femtosecond type-I and type-II BBO frequency conversion,
BBO seeded OPA with noncollinear beams, and LBO SHG/SFG including temperature-tuned
phase matching. They require more than two independently supplied refractive
indices. A finite beam samples different directions of a wave-normal surface;
the eigenpolarizations depend on frequency and direction, and the electric field
need not be perpendicular to its wavevector. LBO has three principal indices,
not an ordinary/extraordinary pair valid for arbitrary directions.

An implementation should also predict when reduced models cease to apply:
insufficient angular or temporal resolution, a material fit used outside its
support, neglect of UV absorption, or a stochastic calculation dominated by its
chosen vacuum-mode cutoff. A successful SHG spectrum alone does not demonstrate
correct depletion, flux, or parametric gain.

#### 2.2 Included in the proposed implementation

- Homogeneous, nonmagnetic reciprocal dielectric slabs at a uniform prescribed
  temperature; an arbitrary fixed proper rotation from material to laboratory
  coordinates. BBO and LBO are the required material qualifications.
- Two forward electromagnetic branches with three-component physical electric
  and magnetic fields. Plane-wave, line and Cartesian free-space geometries,
  including beams with different transverse wavevectors.
- Constant instantaneous conservative tensors and an explicit, separable
  dispersive model of the Miller type, qualified against available nonlinear
  data. Neither is a universal constitutive law.
- SHG, SFG, DFG, seeded OPA, cascaded quadratic effects and pump depletion on
  both temporal grids. Conversion orders emerge from the one polarization;
  they are not separately summed models.
- A restricted, lossless input-vacuum model for OPG, with an exact undepleted-pump
  benchmark and stated semiclassical limitations for depletion (§4.10).
- Linear entrance and exit scattering at parallel plane faces, with reflected
  energy reported and a single transmitted pass propagated through the slab.
  An alternative input already inside the crystal is explicit.
- Resolved, prescribed longitudinal domain reversal for materials that support
  it. This is a generic capability, not a claim that BBO or LBO can be
  periodically poled by the methods used for ferroelectrics.
- True axial energy/fluence and polarization diagnostics for the new route,
  material identities, restart checks, scans, memory admission and backend parity.

The mathematical material input may be a symmetric dielectric tensor in a fixed
material frame, so the linear design is not limited to diagonal uniaxial data.
Initial *material qualification* is narrower: transparent uniaxial and orthorhombic
biaxial crystals, plus weak passive loss where its perturbative treatment is
validated. Availability of a mathematical input is not qualification of every
crystal class or nonlinear response.

#### 2.3 Explicit limits and follow-up work

This SEP does not implement a resonator/OPO, a bidirectional nonlinear solver,
multiple internal reflections, surface SHG, a general multilayer/coating solver,
nonlinear evanescent waves, or near-grazing propagation. It does not solve heat
flow, photorefraction, damage, solid ionisation, stress or electro-optic bias.
Prescribing a uniform oven temperature is included; calculating that temperature
from absorbed average power is not.

DC electrostatics and resonant THz/phonon generation are excluded. An optical DFG
calculation must contain its idler within a qualified positive-frequency material
band. Computing a near-zero-frequency polarization does not qualify its radiation.
Strongly absorbing, gyrotropic, magnetic and indefinite media require further
constitutive and forward-mode analysis.

No general anisotropic Kerr/Raman/absorption model is invented here. Existing
responses compose only when their physical-component and material contracts have
been independently qualified. Scalar measured Kerr or TPA values are not silently
promoted to complete tensors. Guided-wave quadratic overlaps, arbitrary transverse
domain patterns, continuously varying orientation/temperature, and radial crystal
propagation remain follow-up work. A radial pump does not make a birefringent
nonlinear propagation cylindrically symmetric.

This scope is a recommendation for acceptance, not an already agreed physics
default. The questions in §12 identify decisions requiring maintainer agreement.

### 3. Context

The audit of solprop is at commit
fe404ac4df1b3344907ed1ac63625aab8cb54426; the Kokkos backend audit is at
87ac816190ffb0c3978de7e236b6e30d7c068ab8. The relevant contracts are
[architecture](../architecture.md), [conventions](../conventions.md),
[the kernel contract](../kernel-contract.md), [Cartesian propagation](../../guide/free-space/cartesian.md),
[general modal propagation](../../guide/waveguides/general-modal.md),
[nonlinear polarization](../../guide/media/nonlinear.md), and [noise](../../guide/media/noise.md).
The current kernel schema is 15 and the saved-output schema is 7.

| Existing facility | Useful foundation | Required extension |
|---|---|---|
| *spatial.SpatialMedium*, *SpatialDispersion* | Scalar bulk index, spatial launch and frame construction | A tensor constitutive law, orientation and mode-dependent physical normalization |
| *grid.LineGrid*, *FreeGrid* | One/two transverse FFT coordinates, x-fast column order | Crystal admission and angular support; a plane-wave route without fictitious beam area |
| *translate.spatial.SeparableOperator* | Compact frequency rows and transverse squared wavevectors | General crystals cannot be represented by the existing scalar square root |
| *nonlinear.responses.ResponseTraits* | Pointwise/spectral response traits, tensor vocabulary | Quadratic frequency mixing and three physical components, distinct from two stored polarizations |
| *translate.responses.CompiledResponses* | Typed response data and fused execution | Quadratic coefficients, dispersive filters, domain signs and explicit support |
| General-modal E/H tables and metric | Experience with three components and flux-aware reconstruction | Local crystal maps at each transverse-frequency sample, without a dense spatial modal basis |
| *backends/numpy_ref/backend.py*, *backends/numba_cpu/spatial.py* | Existing stepper, FFTs and spatial passes | Small polarization blocks, three-component synthesis/projection and crystal statistics |
| Kokkos *core.hpp*, *core.cpp* and Python adapter | Persistent device state and native spatial FFTs | Matching plain-array leaves and native quadratic/vector passes |
| *backends/spatial.py*, spatial processing/output | Spatial measures and diagnostics | An E/H axial-flux functional; today's vacuum flux factor is insufficient |

The standard state remains
$(n_{\rm batch},n_{\rm modes},n_{\rm pol},n_\omega)$ with frequency contiguous.
For the proposed crystal route, $n_{\rm pol}=2$ continues to mean laboratory
$(x,y)$ electric-field components; $n_{\rm modes}$ enumerates transverse Fourier
columns. Three local physical components do not mean changing that state axis
to length three. The two wave branches are an internal factorization of each
local linear block.

[SEP 3](archive/sep-0003-optmat-materials.md) proposes material ownership and identity
through optmat, but its implementation scope is the gases and solids solprop already
ships, with the principal indices of MgF2 and sapphire as fixed-axis choices. It does
**not** supply BBO/LBO crystal records or second-order tensors. Those are additional
upstream deliverables here. At the audited optmat revision
a5c21ea376074028a945ba3923437e4c02523924, its solid model provides complex
permittivity and frequency derivatives for an individual axis. Its registry and
observations include BBO, including refractive, absorption and third-order data.
They do not yet constitute the coherent linear/thermal/quadratic bundle required
below; no LBO material entry or public second-order tensor model was found.

[SEP 6](archive/sep-0006-dispersive-nonlinearity.md) shares the idea of filtered nonlinear
products and the need for susceptibility symmetries. It is not a prerequisite:
this SEP must work with today's electronic response capabilities. Shared filtered
execution machinery may be consolidated if both proposals are accepted; neither
may silently expand the other's physics.

### 4. Theory

#### 4.1 Fields, frames, tensors and units

All equations in this section use the optics convention
$\widetilde E(\omega)=\int E(t)e^{+i\omega t}\,dt$ and a plane wave
$e^{i(\mathbf k\cdot\mathbf r-\omega t)}$. The internal FFT convention is its
named boundary conversion, as in [conventions §2](../conventions.md).
The material law is

$$
\mathbf D=\epsilon_0\boldsymbol\epsilon_r(\omega,T)\mathbf E
            +\mathbf P^{(2)}+\mathbf P^{(3)}+\cdots,\qquad
\mathbf B=\mu_0\mathbf H .
$$

$\boldsymbol\epsilon_r$ is dimensionless, $E$ is in V/m, $H$ in A/m,
$P$ in C/m², and $\chi^{(2)}$ in m/V. Let the columns of $R\in SO(3)$ be the
material basis vectors expressed in the laboratory basis. Then

$$
\mathbf E_c=R^T\mathbf E_l,\qquad
\boldsymbol\epsilon_l=R\boldsymbol\epsilon_cR^T,\qquad
\chi^{(2),l}_{ijk}=R_{ia}R_{jb}R_{kc}\chi^{(2),c}_{abc}.
$$

Rotation must apply to all three indices. Crystal cut, incident wavevector,
electric polarization, Poynting direction and nonlinear-tensor axes are separate
quantities. A pair of cut angles alone leaves rotation about the cut normal
unspecified; an explicit rotation matrix or a documented three-angle convention
is required. Optical principal axes must be related to crystallographic axes by
a signed, right-handed transformation. Reordering indices by their magnitudes
does not determine the signs or the nonlinear tensor.

For an instantaneous response,

$$
P_i^{(2)}(t)=\epsilon_0\sum_{jk}\chi^{(2)}_{ijk}E_j(t)E_k(t)
           =2\epsilon_0\sum_{jk}d_{ijk}E_j(t)E_k(t).
$$

The canonical contracted order is $(xx,yy,zz,yz,xz,xy)$:

$$
\mathbf P^{(2)}=2\epsilon_0 d
\begin{pmatrix}E_x^2&E_y^2&E_z^2&2E_yE_z&2E_xE_z&2E_xE_y\end{pmatrix}^{T}.
$$

Thus a published $d$ in pm/V is multiplied by $10^{-12}$ to obtain SI $d$,
then by two to obtain $\chi^{(2)}$. The two mixed-input permutations are a
**separate** factor of two. The public record identifies which convention it
contains; the canonical backend contraction never guesses.

Intrinsic permutation symmetry exchanges the two input index/frequency pairs.
It does not justify exchanging an output index with an input while leaving
frequencies unchanged. For a real, frequency-independent *conservative*
instantaneous model, however, a cubic potential
$V_3=\epsilon_0\chi_{ijk}^{(2)}E_iE_jE_k/3$ with
$P_i=\partial V_3/\partial E_i$ requires full index symmetry. Then
$\mathbf E\cdot\partial_t\mathbf P^{(2)}
 =2\partial_t V_3$, whose pulse-integrated contribution vanishes.
An arbitrary real 3×6 table with only input symmetry need not conserve energy.
The constant model must explicitly identify its dispersionless/Kleinman
approximation; a measured SHG table must not be made into a conservative
broadband model merely by calling it real [1,6,7].

#### 4.2 A Maxwell construction of the two forward branches

For a fixed positive $\omega$ and transverse wavevector
$\mathbf q=(k_x,k_y)$, a homogeneous linear crystal satisfies

$$
\left[\mathbf k\mathbf k^T-|\mathbf k|^2I+
       \frac{\omega^2}{c^2}\boldsymbol\epsilon_l\right]\mathbf e=0,\qquad
\mathbf h=\frac{\mathbf k\times\mathbf e}{\mu_0\omega}.
$$

The unknown is $k_z$; $k_x,k_y$ are the actual conserved tangential
wavevectors. This equation gives four longitudinal roots and two forward
branches. Selecting roots solely by the sign of $k_z$ is inadequate in general:
use axial flux, continuity from the transparent forward solution and, for passive
complex roots, decay away from the interface. The initial scope excludes
ill-conditioned forward/backward separation.

A useful independent formulation uses
$\psi=(E_x,E_y,H_x,H_y)^T$. Maxwell's equations eliminate the longitudinal
components as

$$
H_z=\frac{k_xE_y-k_yE_x}{\omega\mu_0},\qquad
E_z=-\frac{\epsilon_{zx}E_x+\epsilon_{zy}E_y}{\epsilon_{zz}}
    -\frac{k_xH_y-k_yH_x}{\omega\epsilon_0\epsilon_{zz}}
    -\frac{P_z}{\epsilon_0\epsilon_{zz}}.
$$

The remaining equations are

$$
\begin{aligned}
\partial_zE_x&=ik_xE_z+i\omega\mu_0H_y,&
\partial_zE_y&=ik_yE_z-i\omega\mu_0H_x,\\
\partial_zH_x&=ik_xH_z-i\omega
 [\epsilon_0(\epsilon_{yx}E_x+\epsilon_{yy}E_y+\epsilon_{yz}E_z)+P_y],\\
\partial_zH_y&=ik_yH_z+i\omega
 [\epsilon_0(\epsilon_{xx}E_x+\epsilon_{xy}E_y+\epsilon_{xz}E_z)+P_x].
\end{aligned}
$$

This defines $\partial_z\psi=iB\psi+iC\mathbf P$ without a memorized
angle formula [2]. In this particular ordering,

$$
C=
\begin{pmatrix}
0&0&-k_x/(\epsilon_0\epsilon_{zz})\\
0&0&-k_y/(\epsilon_0\epsilon_{zz})\\
0&-\omega&\omega\epsilon_{yz}/\epsilon_{zz}\\
\omega&0&-\omega\epsilon_{xz}/\epsilon_{zz}
\end{pmatrix}.
$$

This source includes $P_z$. Simply discarding that component before projection
loses an anisotropic contribution even for some collinear problems.

For numerical conditioning, rescale magnetic components by the vacuum impedance
$Z_0$ before solving the 4×4 eigenproblem; transform $C$ consistently.
Translation verifies the characteristic residual, branch flux, inverse-map
residual and conditioning. In the transparent case, the eigenvectors may be
chosen real, but algorithms must not rely on a fixed sign from a library eigensolver.
Track degenerate subspaces and their projectors; at an optic axis, the physical
two-dimensional propagator remains meaningful when individual branch labels do
not. Sorting roots independently by index at every sample is not tracking.

For a direction specified rather than $\mathbf q$, a complementary check uses
the transverse displacement eigenproblem. With
$Q=I-\hat{\mathbf k}\hat{\mathbf k}^{T}$,

$$
Q\boldsymbol\epsilon_l^{-1}Q\,\mathbf d_j=n_j^{-2}\mathbf d_j,\qquad
\mathbf d_j\perp\hat{\mathbf k},\quad
\mathbf e_j\ \parallel\ \boldsymbol\epsilon_l^{-1}\mathbf d_j.
$$

These are the wave-normal indices. $\mathbf D$ is transverse; $\mathbf E$
usually is not. They must not be substituted for each other in the quadratic
tensor.

#### 4.3 Forward nonlinear propagation and the stored state

Let $V_+$ contain the two forward right eigenvectors of $B$, and $L_+$
the corresponding rows of the inverse of the *full* four-mode eigenvector
matrix. Then $L_+V_+=I$. Projecting the first-order Maxwell system gives

$$
\partial_z a_j=ik_{z,j}a_j+i(L_+C\widetilde{\mathbf P})_j.
$$

This is the forward-mode approximation: nonlinear driving of the backward
subspace is neglected [3]. It does not require a paraxial expansion of $k_z$.
In the scalar isotropic collinear limit it must give

$$
\partial_z\widetilde E=i\frac{n\omega}{c}\widetilde E
       +i\frac{\omega}{2\epsilon_0cn}\widetilde P .
$$

This limit fixes the sign, factor of two and physical units independently.
The internal source has the corresponding negative imaginary prefactor.
Loss is represented separately from the real-index nonlinear normalization,
consistent with existing solprop conventions.

Let $U$ be the transverse electric rows of $V_+$. Preserve the stored state
$u=(E_x,E_y)^T=Ua$ and define

$$
K=U\,{\rm diag}(k_{z,1},k_{z,2})U^{-1},\qquad
S=UL_+C,\qquad
\mathbf E=F_Eu,\quad \mathbf H=F_Hu .
$$

The maps $F_E,F_H$ contain all three physical components obtained from the
linear eigenvectors. They and $S$ depend on $(\omega,\mathbf q)$, not just
on the central propagation angle. Reject nearly singular $U$ rather than
switching state definitions silently. In a constant slab the exact linear
propagator is $U\,{\rm diag}(e^{ik_{z,j}\Delta z})U^{-1}$, multiplied by
the common frame phase. It remains local in frequency and transverse
wavevector. There is no need for a dense coupling between all spatial columns.

Evaluation of $\mathbf P$ uses the *linear* reconstruction $F_Eu$ to first
order in material nonlinearity. The algebraic term
$-P_z/(\epsilon_0\epsilon_{zz})$ already enters the source through $C$;
feeding it back into the constitutive evaluation would add higher-order
nonlinear longitudinal corrections. The proposed UPPE omits that feedback and
backward-wave feedback. Its small parameter is of order
$\|\chi^{(2)}E\|/\|\epsilon_r\|\ll1$ per optical cycle, together with weak
coupling to excluded branches, not necessarily small accumulated conversion.
Phase 0 must bound these omissions against an independent Maxwell calculation.
Near a degeneracy with the backward subspace, a longitudinal resonance, a
strong interface reflection or wavelength-scale domain structure, small $d$
alone does not establish validity.

#### 4.4 Birefringence, walk-off and phase matching

For a uniaxial crystal with optic axis $c$, let $\theta$ be the angle of the
wavevector to $c$. The displacement eigenproblem gives

$$
\frac{1}{n_{\rm ex}(\theta)^2}
 =\frac{\cos^2\theta}{n_o^2}+\frac{\sin^2\theta}{n_e^2}.
$$

Consequently $n_{\rm ex}(0)=n_o$ and $n_{\rm ex}(\pi/2)=n_e$.
These two limits are mandatory tests. For a biaxial crystal the two eigenvalues
must be obtained using all three principal dielectric components; replacing
LBO by a fitted uniaxial effective index is not the proposed model.

The axial wavevector surface supplies both spatial and temporal transport:

$$
\frac{d\mathbf r_\perp}{dz}
 =-\nabla_{\mathbf q}k_z,\qquad
\frac{dt}{dz}=\left.\frac{\partial k_z}{\partial\omega}\right|_{\mathbf q}.
$$

The minus sign in spatial walk-off follows by stationary phase of
$e^{i(\mathbf q\cdot\mathbf r_\perp+k_zz)}$. In a transparent medium this
equals $\mathbf S_\perp/S_z$. The Hessian of $k_z$ supplies anisotropic
diffraction; derivatives with respect to $\omega$ give group delay and
dispersion. Thus exact angular-spectrum propagation already contains walk-off.
Adding a separate walk-off translation would count it twice.

For $\omega_3=\omega_1+\omega_2$, translation invariance requires
$\mathbf q_3=\mathbf q_1+\mathbf q_2$, and the longitudinal mismatch is

$$
\Delta k=k_{z,3}-k_{z,1}-k_{z,2}.
$$

SHG has $\omega_1=\omega_2$; DFG and OPA use the negative-frequency
conjugate of one input. Type I/II labels describe selected eigenpolarizations,
not laboratory x/y labels for every angle. A phase-matching tool must return
all bracketed solutions within a requested angular/temperature interval,
their branch identities and local sensitivities. It must not equate the
lowest mismatch on a coarse scan with a root.

For narrow angular and temporal spectra, useful estimates are
$L_{\rm walk}\sim w/|\rho_1-\rho_2|$ with
$\rho_j=-\partial_{\mathbf q}k_{z,j}$, and
$L_{\rm GVM}\sim\tau/|\partial_\omega k_{z,1}-\partial_\omega k_{z,2}|$.
Here $\rho$ is a slope, not an angle. Temperature and angle sensitivities follow
from derivatives of $\Delta k$ at fixed geometry. At noncritical phase matching
the first angular derivative can vanish; acceptance then depends on the
quadratic term rather than an infinite first-order bandwidth.

#### 4.5 Mixing factors and independent analytic checks

Write a real scalar projection as
$E(t)=\frac12\sum_r[A_r e^{i(k_rz-\omega_rt)}+{\rm c.c.}]$.
For distinct input frequencies, the two permutations in $E^2$ give the
polarization phasor $P_3=2\epsilon_0d_{\rm eff}A_1A_2$.
For degenerate SHG, $P_2=\epsilon_0d_{\rm eff}A_1^2$.
These are phasors of real fields, not positive-frequency Fourier coefficients.
They must not be assigned the same degeneracy multiplier.

In the collinear scalar, transparent, lossless limit, with real reciprocal
$d_{\rm eff}$ and $\Delta k=k_3-k_1-k_2$, the slowly varying phasors obey

$$
\begin{aligned}
A_3'&=i\frac{\omega_3d_{\rm eff}}{n_3c}A_1A_2e^{-i\Delta kz},\\
A_1'&=i\frac{\omega_1d_{\rm eff}}{n_1c}A_3A_2^*e^{+i\Delta kz},\\
A_2'&=i\frac{\omega_2d_{\rm eff}}{n_2c}A_3A_1^*e^{+i\Delta kz}.
\end{aligned}
$$

With $I_j=n_j\epsilon_0c|A_j|^2/2$ these imply

$$
\frac{I_1'}{\omega_1}=\frac{I_2'}{\omega_2}
 =-\frac{I_3'}{\omega_3},\qquad
(I_1+I_2+I_3)'=0 .
$$

For SHG the harmonic equation instead has
$A_2'=i\omega_2d_{\rm eff}A_1^2e^{-i\Delta kz}/(2n_2c)$
and the fundamental equation remains
$A_1'=i\omega_1d_{\rm eff}A_2A_1^*e^{+i\Delta kz}/(n_1c)$.
Then $I_1+I_2$ and the fundamental-equivalent photon flux
$N_1+2N_2$ are conserved; $N_1+N_2$ is not.
For undepleted fundamental and zero injected harmonic,

$$
I_2(L)=\frac{2\omega_1^2d_{\rm eff}^2}
 {n_1^2n_2\epsilon_0c^3}I_1^2L^2
 {\rm sinc}^2(\Delta kL/2),\qquad {\rm sinc}(x)=\frac{\sin x}{x}.
$$

For nondegenerate OPA with constant pump $A_p$, the exponential gain parameter
is

$$
g^2=\frac{\omega_s\omega_i d_{\rm eff}^2|A_p|^2}{n_sn_ic^2}
       -\frac{\Delta k^2}{4}.
$$

Matched seeded signal gain is $\cosh^2(gL)$ when the idler is initially zero.
These formulas qualify mixing constants and signs; the general anisotropic
coupling is the Maxwell projection in §4.3, not this scalar coefficient with
an effective index substituted.

Large phase-mismatched SHG produces a Kerr-like phase. Eliminating its rapidly
oscillating harmonic gives a correction proportional to
$-d_{\rm eff}^2/\Delta k$ in this convention. Do not add a fitted cascaded
$n_2$ when the same harmonic is propagated explicitly. Intrinsic Kerr data
must be separated from cascading contributions [7].

#### 4.6 Broadband analytic envelopes and the propagation frame

Define the physical analytic field
$\mathcal E=2\Pi_+E$ for strictly positive frequencies and write
$\mathcal E=Ae^{-i\omega_0t}$ in the optics convention.
$\Pi_+$ keeps strictly positive absolute frequencies without doubling them.
For a symmetric-input instantaneous tensor,

$$
\mathcal P_i^{(2)}
 =\frac{\epsilon_0}{2}\Pi_+\sum_{jk}\chi^{(2)}_{ijk}
 \left(\mathcal E_j\mathcal E_k+
       \mathcal E_j\mathcal E_k^*+\mathcal E_j^*\mathcal E_k\right).
$$

This follows by expanding $E=(\mathcal E+\mathcal E^*)/2$ and taking twice
the positive-frequency part. The double-conjugate sum has negative support.
In envelope variables, the candidate before positive-absolute-frequency
projection is

$$
P_{A,i}^{(2)}=\frac{\epsilon_0}{2}\sum_{jk}\chi^{(2)}_{ijk}
\left[A_jA_ke^{-i\omega_0t}
 +(A_jA_k^*+A_j^*A_k)e^{+i\omega_0t}\right].
$$

Projection means remodulate, select $\omega>0$, and demodulate, or the exactly
equivalent mask $\omega_0+\Omega>0$ on the envelope spectrum.
The internal carrier signs are reversed by the Fourier boundary convention.
The SFG and DFG terms, the factor $1/2$, and the projector are all essential
[4]. A nonlinear envelope is not obtained by applying the real-field square
to a complex envelope, nor by keeping the SFG term alone.

A pure moving frame subtracts $\beta_{\rm frame}\omega$ from every branch,
where $\beta_{\rm frame}$ has units s/m. It leaves every resonant frequency
sum unchanged. A common additional constant phase subtraction $\kappa_0$
does not: the transformed quadratic source acquires explicit factors
$e^{\pm i\kappa_0z}$. The recommended route subtracts **only** the term
linear in absolute frequency, on both grids, and retains carrier phase.
There must be a declared carrier-mixing requirement rather than an accidental
dependency on a flag named for third-harmonic generation.

Carrier factors must also be periodic on the implemented transform lattice.
Phase 0 must specify the frequency origin and require a bin-aligned carrier,
or supply an exact remapping with a demonstrated error bound; periodic FFTs
must not silently treat an arbitrary nonperiodic carrier factor as exact.
Field/envelope parity compares complex spectra, conversion and depletion,
not just harmonic magnitudes. The exact projected envelope formulation can
cost as much as the real-field representation when the same octave-spanning
band is retained. A speedup is not promised.

#### 4.7 Dispersive quadratic response

The general response is a two-frequency convolution,

$$
\widetilde P_i^{(2)}(\omega)=\frac{\epsilon_0}{2\pi}
\sum_{jk}\int_{-\infty}^{\infty}
\chi^{(2)}_{ijk}(-\omega;\omega',\omega-\omega')
\widetilde E_j(\omega')\widetilde E_k(\omega-\omega')\,d\omega'.
$$

The signed-frequency notation includes both SFG and DFG. Reality requires
simultaneous sign reversal to conjugate the susceptibility. Lossless energy
exchange imposes further index/frequency permutation relations; input
permutation alone is insufficient [1]. Degenerate SHG measurements sample
only a slice of this function.

A useful *declared approximation* in a transparent region is

$$
\chi^{(2)}_{ijk}(-\omega_3;\omega_1,\omega_2)
 =\Delta_{ijk}\,f_i(\omega_3)f_j(\omega_1)f_k(\omega_2),
\qquad f_i(\omega)=\epsilon_i(\omega)-1 ,
$$

in a fixed principal material frame, with
$f_i(-\omega)=f_i(\omega)^*$. Each measured component is calibrated using
its own output/input frequencies and axes. In a lossless conservative version,
the underlying real coupling $\Delta_{ijk}$ is symmetric and the real even
filters preserve the required exchange symmetry. If measurements cannot be
reconciled with that model within their uncertainty, record the discrepancy
and restrict or replace the model. Never silently average incompatible tensor
entries, or fit unrelated frequency legs independently.

Execution is three operations: filter each material-frame input field, form the
quadratic tensor products in time, and filter each output component. A more
general low-rank sum uses several such terms. Multiplying only the final
polarization spectrum by a scalar curve omits input-frequency dispersion.
Applying a scalar index at the propagation angle to every tensor leg also
confuses material axes with propagation eigenvectors.

Miller scaling has experimental support as an approximation across several
frequency-conversion crystals [6]; it is not a causal resonant model and must
not be extrapolated through an infrared phonon band or absorption edge.
No universal frequency-independent $d$ is inferred from a single reference
wavelength. Constant and dispersive models remain distinct optmat identities.
General off-diagonal dispersive material filters are a future extension unless
their fixed-frame tensor construction is explicitly qualified in Phase 0.

#### 4.8 Flux, pulse energy, interfaces and absorption

The correct cycle-averaged analytic axial flux is

$$
S_z(t,\mathbf r_\perp)
 =\frac12{\rm Re}\left[
 \mathcal E_x\mathcal H_y^*-\mathcal E_y\mathcal H_x^*
 \right],\qquad
U=\int dt\,dA\,S_z .
$$

Synthesize E and H with their frequency-dependent maps before forming a time
trace. For a monochromatic mode with unit polarization shape, define
$w_j=\frac12{\rm Re}(\mathbf e_j\times\mathbf h_j^*)_z$;
its flux is $w_j|a_j|^2$. In transverse laboratory amplitudes the corresponding
spectral Hermitian metric follows from $F_E,F_H$ and includes any cross terms.
It must be positive on the admitted transparent forward subspace.
For full Cartesian fields use the existing spatial Parseval weights and raw-DFT
temporal factors; for a line the result is J/m, and for a plane wave it is
fluence J/m². An arbitrary simulation box must not turn plane-wave intensity
into an advertised pulse energy.

Multiplying the existing vacuum spatial energy by one carrier index does not
give the broadband anisotropic flux. A frequency-dependent square-root weight
can normalize integrated spectral energy but generally does not reproduce the
temporal Poynting trace. Local flux need not be positive at every instant;
clipping would break its integral. Report the chosen flux convention in output.

At a plane interface, $\omega$ and $\mathbf q$ are continuous and the four
tangential E/H components match. For incidence from the left,

$$
V_{\rm left,+}a_{\rm in}+V_{\rm left,-}a_{\rm refl}
 =V_{\rm right,+}a_{\rm trans}.
$$

Solve this 4×4 system with the same scaled fields and branch normalization used
inside the slab. Check flux balance including reflection, not
$|t|^2+|r|^2=1$ without admittances. The exit solves the analogous problem.
Initial interface qualification is between isotropic external media and the
crystal. Multiple reflected passes and coating interference are excluded; an
externally supplied coating transfer must not masquerade as a bare interface.
Strong Fresnel reflection requires assessing the single-pass approximation
against the use case even when the linear interface calculation is exact.

For weak passive absorption let
$\epsilon_r=\epsilon'_r+i\epsilon''_r$ in the optics convention. The linear
modes and nonlinear projection use the transparent reference, and a
flux-normalized loss matrix is derived from dissipated power

$$
Q_{\rm abs}=\frac{\omega\epsilon_0}{2}
             \mathbf E^\dagger\epsilon''_r\mathbf E .
$$

For an isolated mode $\alpha_j=Q_{{\rm abs},j}/S_{z,j}$ is the intensity
attenuation coefficient in m⁻¹; its amplitude decays as
$e^{-\alpha_jz/2}$. Off-diagonal dissipation cannot be discarded merely
because the real dielectric tensor has been diagonalized. Keep the resulting
2×2 dissipative block or refuse the independent-attenuation approximation when
its error is not bounded. Negative dissipated power is an invalid passive record.
Near-resonant complex eigenmodes are outside this weak-loss model.

#### 4.9 Resolution, domain reversal and retained bandwidth

A real quadratic product reaches twice the largest participating frequency.
The fine temporal Nyquist frequency must exceed that support, including guard
bands, before cropping. A cubic response present in the same evaluation still
sets the larger support. The check uses the retained spectral support and actual
fine grid, not only the pulse's central wavelength or the name of a grid flag.
The same issue exists in transverse Fourier space: two populated wavevectors
sum. Temporal padding cannot prevent angular aliasing. Pad the transverse
product or admit only a band with a demonstrated product-support margin; test
both axes and the negative-frequency conjugate wavevector.

For a real field,
$\widetilde E(-\omega,-\mathbf q)=\widetilde E(\omega,\mathbf q)^*$.
At fixed nonzero $\mathbf q$ the temporal half-spectrum is not independently
Hermitian. Reconstruct transverse physical coordinates before a real temporal
inverse transform, as the existing Cartesian route does.

The active band must contain pump, seed, idler, harmonic and any retained
cascaded bands, and all populated angles must remain forward and away from
cutoff. Excluding a generated band is a Galerkin truncation, not a prediction
that the material cannot generate it. Monitor nonlinear source in guard bands,
field power near boundaries, and energy removed by absorbers. Resolve conclusions
against broader bands. Zero frequency is explicitly excluded; never divide by
it or replace an invalid dispersion value by vacuum.

For a prescribed sign function $s(z)=\pm1$, use
$\chi^{(2)}(z)=s(z)\chi^{(2)}$. Each domain wall is a hard step boundary with
the same one-sided stage semantics as other solprop kinks. For a 50% duty-cycle
square wave of period $\Lambda$, its first Fourier coefficient has magnitude
$2/\pi$, so the long-scale first-order coupling is
$d_{\rm eff,QPM}=2d_{\rm eff}/\pi$ and
$\Delta k_{\rm QPM}=\Delta k-2\pi/\Lambda$ for the matching Fourier order [5].
Resolving the domains and using this averaged coefficient simultaneously would
double count quasi-phase matching. The initial implementation resolves domains;
the averaged form is a validation limit, not a second default solver.

#### 4.10 OPG, classical seeds and the meaning of vacuum noise

The deterministic three-wave equations have an exact zero-signal/zero-idler
solution. Roundoff is not a physical seed. Seeded OPA, technical-noise-driven
conversion and vacuum-started OPG must have distinct input metadata.
The existing fixed-modulus one-photon draw must not be silently reused as a
quantum vacuum in crystal simulations. Quantum amplifier noise and phase-space
evolution provide the relevant checks [23,24].

For orthogonal discrete modes normalized so that one excitation has energy
$\hbar\omega_\mu$, propose a Wigner vacuum draw

$$
\alpha_\mu=(X_\mu+iY_\mu)/2,\qquad
X_\mu,Y_\mu\sim{\cal N}(0,1),\qquad
\langle|\alpha_\mu|^2\rangle=\frac12 .
$$

The spatial/temporal quadrature and crystal flux metric map these amplitudes
into stored field coefficients. For coupled polarizations, whiten the positive
flux metric before drawing independent modes, then transform back. The
periodic aperture, time window, admitted frequency/angle band and seed are part
of the model. A single plane wave or line has no specified three-dimensional
vacuum mode volume: require an explicit quantization area or missing transverse
width before enabling this option.

For an undepleted phase-matched pump, the exact two-mode transformation is

$$
\alpha_s(L)=\cosh r\,\alpha_s(0)
       +e^{i\varphi}\sinh r\,\alpha_i(0)^*,\qquad r=gL,
$$

with the analogous idler relation. It predicts
$\langle|\alpha_s(L)|^2\rangle-\frac12=\sinh^2r$ and the associated
signal–idler correlation. This is an analytic qualification, including the
small-gain limit, of normalization and normally ordered mean output.
The phase follows from the chosen pump and source convention. This linear
Bogoliubov transformation preserves canonical commutators; the vacuum term is
necessary for that property [23].

For a depleted pump, classical propagation of Wigner initial conditions is a
truncated phase-space approximation, not an exact quantum solver. A cubic
interaction Hamiltonian also generates higher derivatives in the exact Wigner
evolution; dropping them must be stated [24]. Initially qualify only observables
and occupancies for which comparison with a small independent quantum model and
vacuum-band refinement is satisfactory. Entanglement, arbitrary low-photon
statistics and absolute spontaneous spectra outside that qualification are not
promised.

Subtract half a photon per detected admitted output mode for normally ordered
mean energy; do not subtract from the propagated field or clip each realization.
Report statistical confidence intervals separately from discretization errors.
Use deterministic independent random streams by realization, unaffected by batch
partition or backend. Freeze the physical noise cutoff while refining numerical
padding. Absorptive loss would require reservoir noise; the initial OPG mode
therefore refuses material loss and loss through occupied numerical collars.
That restriction does not prohibit deterministic lossy frequency conversion.

#### 4.11 Material properties and the optmat handoff

A crystal name and a Sellmeier equation are insufficient. The following is the
minimum data contract, with optional capabilities kept distinct from missing
information. SI conversion happens once at the material boundary; source units
and conventions remain in provenance.

| Property | Required representation | Use and admission consequence |
|---|---|---|
| Material identity | Chemical composition, polymorph, doping where relevant, point group, reference state, record/version digest | Distinguish beta-BBO from alpha-BBO; distinguish a material from one fitted model |
| Coordinate frame | Named signed crystallographic and dielectric axes, handedness, conversion matrices and angle conventions | Rotate linear and nonlinear data together; never combine tables with unknown relative orientation |
| Linear response | $\epsilon_{ab}(\omega,T)$ or compatible principal-axis models, derivatives and joint validity domain | Modes, phase matching, walk-off, group delay and diffraction; all participating bands need coverage |
| Linear absorption | Axis-resolved $\epsilon''$ or intensity absorption, sample conditions and uncertainty | Weak-loss model, UV/IR efficiency; zero absorption is an explicit idealization, not missing data interpreted as zero |
| Quadratic response | Signed tensor, contraction convention, reference frequency tuple for each coefficient, temperature, measurement geometry, uncertainty | SHG, SFG, OPA and reciprocal back-conversion; a scalar effective coefficient is specific to one geometry |
| Nonlinear dispersion | Frequency-dependent measurements or a named fit with its residuals, tensor/frequency symmetry and validity | Choose constant or Miller-type approximation deliberately; data at degenerate frequencies do not determine every nondegenerate interaction |
| Temperature response | Joint $\epsilon(\omega,T)$ or $dn_i/dT$ with integration reference and supported temperature interval | LBO temperature tuning and BBO thermal sensitivity; include mixed wavelength/temperature derivatives |
| Expansion and geometry | Thermal expansion tensor when physical dimensions are specified at a reference temperature | Crystal length, cut normal and domain period at operating temperature; do not expand a length already specified at that temperature |
| Intrinsic electronic $\chi^{(3)}$ | Tensor or explicitly limited projection, wavelength/pulse-duration conditions and separation from cascading | Competition with self-focusing and nonlinear mismatch; absent in a pure-quadratic run, never manufactured from an unrelated scalar $n_2$ |
| Raman/phonon response | Tensor selection rules, resonance strengths and damping with state dependence | Needed before resonant IR/THz or delayed crystal response is admitted; outside the initial qualified scope |
| Nonlinear absorption/carriers | Frequency and polarization dependence, pulse conditions, carrier model where used | Needed for high-fluence UV predictions; SEP 2 alone does not supply a general crystal tensor |
| Thermal/damage/device data | Conductivity, heat capacity, density, stress-optic constants, measured damage protocol, coatings and surface loss as applicable | Required by future thermal/device models; a quoted damage threshold is not an intensity-clipping constitutive law |

Group index, GVD, walk-off, phase-matching angles and effective nonlinearities
should normally be **derived** from the selected records. Measured values of those
quantities are independent validation data, not additional terms to apply.
For a diagonal dielectric record,
$n_{g,i}=n_i+\omega\,\partial_\omega n_i$; for an oblique wave the relevant
axial derivative is instead $\partial_\omega k_z|_{\mathbf q}$.
An empirical $dn/dT$ fit must be integrated from its declared temperature; it
must not be treated as a temperature-independent offset when it is not.

**BBO.** The target is beta-BaB₂O₄, point group $3m$, with two principal
indices. For a material frame with its threefold axis along $c$ and a mirror
plane $bc$, one conventional $3m$ pattern is

$$
d=\begin{pmatrix}
0&0&0&0&d_{15}&-d_{22}\\
-d_{22}&d_{22}&0&d_{15}&0&0\\
d_{31}&d_{31}&d_{33}&0&0&0
\end{pmatrix}.
$$

Here the sign of $d_{22}$ is defined by this chosen frame; a different signed
crystallographic frame must transform the full tensor. The instantaneous
conservative approximation additionally requires $d_{15}=d_{31}$. The
pattern alone assigns neither measured signs nor numerical values [7,11].
Need ordinary and extraordinary dispersion, their thermal dependence,
the orientation/sign convention for the dominant transverse quadratic coefficient
and the smaller longitudinal coefficients, and their wavelength dependence.
The relative signs can affect azimuthal tuning and interference, even when an
isolated SHG measurement determines only a magnitude. Preserve measured departures
from a dispersionless symmetry in the source record; the propagator's
approximation is a separate model decision. The Kerr analysis in [7] illustrates
both the significance of cascading corrections and uncertainty of weaker tensor
components. BBO UV absorption data [18,19] also show why “inside the transparency
window” is not synonymous with negligible loss.

**LBO.** The target is LiB₃O₅, orthorhombic point group $mm2$, with three
principal indices and temperature-dependent birefringence. A frequently used
nonlinear tensor convention puts the polar axis in its third coordinate; optical
axis labels sorted by index need not use that same coordinate. Acquire and
verify the signed transformation in the original sources before writing a
default tensor. In a material frame with polar axis $c$ and mirror planes
$ac,bc$, $mm2$ permits the contracted pattern

$$
d=\begin{pmatrix}
0&0&0&0&d_{15}&0\\
0&0&0&d_{24}&0&0\\
d_{31}&d_{32}&d_{33}&0&0&0
\end{pmatrix}.
$$

This is a symmetry pattern in the stated frame, **not** an assertion that a
published LBO optical $(x,y,z)$ table uses this ordering. Under the additional
instantaneous conservative approximation $d_{15}=d_{31}$ and
$d_{24}=d_{32}$; otherwise different frequency legs must be specified.
Temperature-tuned noncritical matching is an essential LBO validation, not
an optional extrapolation from room-temperature indices [13–16].
The Maker-fringe coefficients reported in [25] are **relative to a KDP
reference**, not numbers that can be assigned pm/V directly. Curation must
recover the reference coefficient, wavelength, convention and uncertainty
before converting to SI. Retain the separately measured input/output
permutations before fitting any conservative approximation. The LBO nonlinear
refraction measurement in [26] is a useful further input, but its measured
projection and pulse conditions do not supply a complete Kerr tensor.

**General applicability.** Material identities, tensor records and rotations
must support later crystals without changing kernels. KTP and lithium niobate,
for example, motivate domain reversal and temperature-dependent records, but are
not automatically qualified by BBO/LBO tests. Monoclinic materials can have a
frequency-dependent rotation of dielectric principal axes; the material frame
must stay fixed and the full dielectric tensor must carry that dispersion.
A single static rotation of three independently fitted indices is insufficient.

#### 4.12 Prioritized source acquisition and verification

References below are concrete candidate inputs to curation, not a license to
copy coefficients without their definitions. “Metadata checked” means title,
authors/year and DOI were resolved against the publisher or registry. It does
not mean a PDF's equations, uncertainties, axis conventions or validity range
were checked. “Accessible table” likewise does not certify omitted footnotes.
The drafting audit transcribes **no numerical BBO or LBO material coefficients**.
The unit conversions and mathematical factors above are derived, not fitted data.

| Priority | Source | Required evidence to extract | Drafting evidence and remaining work |
|---|---|---|---|
| P0 | Eimerl et al. 1987 [8] | BBO linear, mechanical and thermal baseline; axes and experimental conditions | Metadata checked; obtain complete article and verify equations/tables |
| P0 | Kato, Umemura and Mikami 2010 [9] and Kato et al. 2021 [10] | BBO Sellmeier/thermo-optic models and the later revision; phase-matching residuals | Metadata checked; acquire both and compare domains and changed predictions before selecting a default |
| P0 | Shoji et al. 1999 [11] | Absolute BBO quadratic calibration, weak components, reference wavelengths, uncertainty and signs actually measured | Publisher abstract and Tables 1–2 inspected; full methods, equations and footnotes still required |
| P0 | Alford and Smith 2001 [6] | BBO/LBO nonlinear wavelength dependence, nondegenerate OPA evidence and limits of Miller scaling | Publisher abstract and visible table inspected; full measurement conditions and tensor mapping required |
| P0 | Chen et al. 1989 [12] | LBO structure, axis conventions, original optical and nonlinear characterization | Publisher abstract inspected; acquire full paper and check coordinate definitions |
| P0 | Lin et al. 1990 [25] | LBO Maker-fringe tensor measurements, relative signs, KDP reference calibration and uncertainties | Publisher-deposited abstract and registry metadata checked; obtain full article to convert its relative coefficients and verify its axes |
| P0 | Velsko et al. 1991 [13] | LBO index, thermal response, quadratic coefficients, absorption, angular/thermal acceptance | Metadata checked; full tables and uncertainties required |
| P0 | Kato 1994 [14] and Kato, Grechin and Umemura 2018 [15] | LBO Sellmeier reference and revised thermo-optic relation as a matched pair | Metadata checked; verify their joint prediction of measured noncritical matching |
| P1 | Tang, Cui and Dunn 1995 [16] | Independent LBO thermal-index measurements and phase-matching checks | Publisher abstract inspected; measurements cover a stated temperature interval, which must be converted to K with its reference preserved |
| P1 | Zhang, Kong and Zhang 2000 [17] | BBO near-IR-edge dispersion and OPG/absorption context | Metadata checked; do not extend transparent Sellmeier fitting through absorption without the measurements |
| P1 | Bache et al. 2013 [7] | BBO intrinsic Kerr, tensor conventions, cascading corrections and source trail for relative signs | Author full-text version inspected; follow its cited primary sign measurements before curating those signs |
| P1 | DeSalvo et al. 1996 [18]; Dragomir et al. 2002 and erratum [19] | BBO measured nonlinear refraction/absorption, polarization and cut dependence; distinguish linear loss from TPA | Extracted original full texts and validation reports inspected; erratum metadata checked but correction text still needed; no coefficients transcribed here, no promotion to a complete nonlinear tensor |
| P1 | Hanson and Dick 1991 [20] | LBO temperature-tuned noncritical OPG as a validation target | Publisher abstract inspected; experimental bandwidth, pump, aperture and collection conditions needed before absolute comparison |
| P1 | Maingot et al. 2023 [26] | LBO nonlinear refraction, polarization/cut and cascading separation | Publisher abstract inspected; acquire full measurements and analysis before using as intrinsic Kerr input |
| P2 | Additional primary measurements to identify during curation | Complete weak tensor components, signed LBO coefficients, $d(T)$, axis-resolved weak absorption over chosen bands, intrinsic LBO Kerr and sample uncertainty | Open data gaps; a review, supplier table or another implementation is only a lead |

P0 is the acquisition packet for initial BBO/LBO defaults. P1 supplies independent
checks and realistic validity limits; it can become P0 for a UV, long-wavelength
or high-intensity target. A universal catalog transparency range must never
replace the measured domain of a particular fit.

The two inspected absorption papers carry bibliographic-validation warnings
because identity was matched from bibliographic evidence rather than a usable
printed DOI. Their extracted table grids report no findings; the 2002 extraction
has an unmatched figure-caption warning. Neither warning is evidence against
the measurements, but any numerical import must check the relevant original
page and table. This SEP relies on their qualitative material limitations,
not on unverified extracted numeric cells.

The optmat handoff must deliver: source PDFs/records with stable citations;
checked source units and tensor convention; raw observations separated from fits;
named model selections and comparison residuals; uncertainty and validity domains;
signed coordinate transformations; and immutable identities for the resolved
linear, thermal and nonlinear bundle. solprop must consume those public records
without a second private coefficient database. An unavailable coefficient,
a symmetry-forbidden zero and a deliberately neglected small component are three
different states.

### 5. Proposed interface

Names here are provisional. Follow the existing frozen dataclass and Simulation
patterns; implementation acceptance settles exact public signatures. The
conceptual objects are:

| Proposed object | Contract |
|---|---|
| *CrystalMaterial* | A resolved optmat record bundle or explicitly supplied, provenance-bearing dielectric/quadratic laws; no backend-specific objects |
| *CrystalOrientation* | Proper 3×3 rotation, or three documented angles in rad; a cut convenience constructor also requires the in-plane reference |
| *CrystalSlab* | Material, orientation, uniform positive temperature in K, positive length in m, optional resolved domain edges/signs; dimensions refer to operating temperature unless a reference state is explicitly supplied |
| *QuadraticResponse* | One constant or dispersive tensor model in material coordinates, with its validity and symmetry declaration; no gas-density multiplication |
| *CrystalInput* | Optics-convention incident or internal fields, physical units, external medium if incident, and explicit normalization by J, J/m or J/m² according to geometry |
| *CrystalVacuum* | Explicit Wigner input ensemble, seed, physical mode band and quantization measure; only the qualified OPG scope |
| *Simulation.from_crystal* | Assembles a slab, temporal/spatial grid, input, response set, frame and backend using the existing run/output interface |
| *crystal_phase_matching* | Finds phase-matching roots and returns mismatch, eigenpolarizations, walk-off, GVM and sensitivities with material identities |

Illustrative setup, not executable documentation:

~~~text
material = CrystalMaterial.from_optmat(
    "BBO", linear_model=chosen_linear_record,
    quadratic_model=chosen_quadratic_record,
)
slab = CrystalSlab(
    material=material,
    orientation=CrystalOrientation.from_matrix(crystal_to_lab),
    temperature=294.0, length=1e-3,
)
response = QuadraticResponse(material.quadratic)
simulation = Simulation.from_crystal(
    slab=slab, grid=field_grid, spatial_grid=line_grid,
    input=CrystalInput.internal(field, energy_per_width=energy_per_width),
    responses=(response,), frame=group_delay_frame,
    backend="numpy_ref",
)
result = simulation.run()
~~~

An input in laboratory x/y is projected through the crystal maps; “ordinary”
and “extraordinary” are optional branch-aware launch helpers for uniaxial
materials, not alternate meanings of x/y. Frequency-dependent eigenpolarizations
are retained in those helpers. Noncollinear input specifies transverse phase
or transverse wavevector; a vacuum incidence angle must first pass through the
interface construction, rather than be added to the internal cut angle.

Users explicitly select the nonlinear model while defaults are being qualified.
Requesting a seed-free deterministic OPG prediction must produce an explanatory
error or an explicitly acknowledged zero-seed deterministic experiment, not a
run seeded from floating-point artifacts. OPG ensembles report both raw
symmetrically ordered and corrected mean observables.

Reuse Local/Pool scans with material/angle/temperature parameters captured by
the builder. Native lockstep Batch needs separate admission of identical layouts,
support, field maps and schedules; it is not enabled simply because single runs
work. Saved outputs identify the crystal route, physical measures, internal versus
external planes, lab/material axes, interfaces, material records, noise convention
and approximation choices. Resume reconstructs and verifies maps and their
identities before accepting a field.

Refuse unsupported grids/backends by name, invalid rotations, incompatible
tensor conventions, missing material bands, unqualified loss/noise combinations,
unsupported longitudinal profiles, a scalar launch that suppresses a required
branch, singular forward maps and appreciable evanescent/grazing content.
These are physical or numerical admission decisions, not fallback opportunities.

### 6. Implementation strategies

#### 6.1 Recommended decomposition

Keep physical definitions in a proposed *solprop.crystal* domain module and
*solprop.nonlinear.quadratic*, with optmat adapters under *physdata*.
The architecture diagram and package layout must be updated when these modules
are implemented. Crystal geometry and interfaces remain solprop responsibilities.
No new numerical dependency is needed: NumPy/SciPy provide matrix decompositions,
root solving and the independent small-system references.

Translation constructs and validates the 4×4 Maxwell system, forward maps,
2×2 linear blocks, nonlinear contraction/filter records and flux functional.
Backends consume arrays and scalar tags only. They never evaluate a Sellmeier
equation, infer a point group, run a material registry, or solve an angular
root problem in a Runge–Kutta stage.

The new route should have a small optional crystal record rather than overload
the scalar *SeparableOperator* with inconsistent meanings. Candidate leaves:

- Linear eigenvalues/factorization or equivalent local 2×2 generator; damping
  and angular/support admission data.
- Three-by-two E/H synthesis maps and two-by-three polarization source maps,
  or a qualified compressed representation sufficient to reproduce them.
- Constant tensor or low-rank material-frame filter/contraction records,
  coarse/fine support maps, carrier-mixing metadata and domain-sign schedule.
- Flux reconstruction/metric, coordinate identities and output reference data.

Shapes and ownership are frozen in the kernel contract before implementation.
They use the existing standard axes, adding a documented two-branch axis only
where the factorization actually needs it. No field-sized matrices are attached
to Python physics objects in the stage path.

At least one **SCHEMA_VERSION bump is required** when these leaves first cross
the seam. The protected linear-flow/driver hooks and any new stage fields are
part of the same versioned surface. Subsequent seam-changing phases bump again
rather than promising one version number in advance. Schema literals, the
contract manual, native adapter tests and changelog change together.
An installed backend compiled against the current version still needs explicit
crystal capability checks; reading the imported version constant is not feature
detection.

#### 6.2 Linear evolution and nonlinear passes

The constant 2×2 linear block can be applied exactly through its qualified
factorization. Retain the interaction-picture stepper and its embedded error
estimate; add a local polarization-flow implementation instead of replacing the
stepper. At degenerate eigenvalues use the continuous block/projector limit.
An absorber not commuting with this block remains in the proper split/nonlinear
part, as required by its existing contract.

One nonlinear evaluation consists of:

1. Apply spectral E synthesis (and material-frame filters when present).
2. Inverse transverse transforms, then inverse temporal transforms, into three
   real physical components on a field grid.
3. Contract the quadratic tensor, accumulating compatible responses in a
   specified deterministic order.
4. Transform the three polarization components back; apply output filters,
   source maps, cropping and the interaction-picture factors.

On the envelope grid, apply the exact carrier products and positive-frequency
projection of §4.6. Ordinary Kerr must still see the *unfiltered* physical field
when quadratic filters are active. It must not accidentally see a linear
polarization surrogate. Shared transforms are fused only when their operands
are mathematically identical.

NumPy establishes the oracle. Numba reuses scalar body factories and typed
preallocated kernels, with no fastmath or cached parallel closures. Kokkos CPU
implements the same array contracts and operation/reduction order, and retains
the maps/device arrays between steps. CPU qualification is required; CUDA/HIP
qualification follows on devices. Deterministic field parity uses the same
input noise realization, not backend-specific random generators.

#### 6.3 Options for angular/frequency tables

| Representation | Benefit | Cost/risk | Recommendation |
|---|---|---|---|
| Two scalar indices at the central direction and transverse E only | Small change and cheap | Misses general biaxial eigenvectors, longitudinal E and angular walk-off | Restricted comparison model only; insufficient for this SEP |
| Dense local Maxwell maps over $(\omega,k_x,k_y)$ | Clear oracle; supports general admitted orientations | Tables scale with full spatiotemporal spectral volume | First reference representation, with hard memory admission |
| Compute analytic crystal maps in backend stages | Small stored tables | Duplicates physics, costly roots and branch tracking, backend drift | Reject |
| Interpolate/compress translated maps or generator/projector data | Lower memory for smooth angular regions | Phase errors, gauge discontinuities and optic-axis degeneracy need budgets | Qualify after the dense oracle; use matrix/subspace data near crossings |
| Separate narrowband coupled-wave fields | Excellent cost for isolated bands | Separate carriers, crossings, missing cascades and different model | Independent validation and possible later reduced solver |

Do not interpolate a gauge-dependent eigenvector through a sign flip. Compression
must bound $L\,\delta k_z$, nonlinear-source error and flux error on independent
check nodes, across the admitted band and angular aperture. Relative index error
alone is inadequate: phase matching subtracts large nearly equal wavevectors.

#### 6.4 Expected runtime and memory

There is no implemented crystal route to time. The following are operation and
storage estimates, not benchmark results or promised speedups. Let
$M=N_\perp N_\omega$ and $M_f=N_{\perp,f}N_{t,f}$; multiply evolving buffers
by batch size $B$. A two-component complex128 state requires $32BM$ bytes.
One three-component real fine-grid array requires $24BM_f$ bytes, or twice
that for complex envelope components. The number of state-sized Runge–Kutta
buffers is inherited from the selected stepper and must be included.

A deliberately redundant dense reference can retain six complex E-map
coefficients, six H-map coefficients, six source-map coefficients, eight
coefficients for $U,U^{-1}$ and two eigenvalues per sample: **448M bytes**
before flags, workspaces, absorbers or flux tables. Some are redundant, H maps
may be reconstructed for statistics, and transparent maps can be real;
these savings must be measured rather than assumed in admission.

| Illustrative coarse spectral grid | One two-component state | Redundant map estimate |
|---|---|---|
| 256 transverse columns × 2049 frequencies | 16.0 MiB | 224 MiB |
| 64 × 64 columns × 1025 frequencies | 128 MiB | 1.75 GiB |
| 256 × 256 columns × 2049 frequencies | 4.00 GiB | 56.0 GiB |

These examples show why line calculations and table compression matter.
They are allocation arithmetic, not recommended physical grids. Full transverse
FFT stages also require whole-plane working storage; ordinary node chunking
does not make an arbitrary three-dimensional transform fit into a small budget.
Translation peak memory, temporary eigensolver arrays, FFT plans and host/device
copies all count. Stream table construction in bounded tiles without running
physics in stages.

At fixed grids, a constant tensor contraction is $O(M_f)$ with six input
products and three accumulations. It is likely a small addition to an existing
two-component Kerr pass. Reconstructing three physical components requires
three inverse and three forward component transforms instead of two and two:
the **transform-count ratio is 3/2** when all components are necessary, before
padding and extra diagnostics. Linear and source maps cost $O(M)$ small
matrix-vector products. None of this implies an overall 1.5× runtime; memory
traffic, FFT shape and backend overhead set the actual ratio.

For rank-one Miller scaling with a quadratic response alone, input/output
filters can share those transforms. Coexisting unfiltered Kerr, distinct
low-rank filters, or a literal analytic projector can require additional
transform groups. If $r$ independent filtered quadratic terms cannot share
operands, nonlinear FFT work grows approximately linearly with $r$.
By contrast, a direct general two-frequency convolution scales as
$O(N_\perp N_\omega^2)$ and is suitable only for small oracle tests.

The largest practical cost changes may come from physics, not the local tensor:

- Keeping both pump and harmonic/idler can enlarge the retained bandwidth and
  fine grid. Keeping long group-delay separation enlarges the temporal window.
- Walk-off and noncollinearity enlarge transverse domains; the shortest retained
  wavelength and angular support determine transverse sampling.
- Phase mismatch, rapid exchange and weak-field accuracy can reduce accepted
  step size. Exact linear evolution removes a linear stability restriction,
  not the need to resolve the oscillatory nonlinear interaction.
- Resolved domain walls force step landings. Very fine poling can dominate
  stage count even when the averaged conversion is slow.
- OPG uncertainty requires multiple realizations. Independent runs scale nearly
  linearly in total work, with ensemble mean error decreasing as
  $N_{\rm realizations}^{-1/2}$ for finite-variance observables.

Field and broadband envelope storage/FFT costs must be compared at the same
physical support and observable error. Do not promise that an envelope is
cheaper merely because it is called an envelope. GPU benefit is most plausible
for large spatial transforms or ensembles, but large tables can exhaust device
memory before compute throughput is useful.

Benchmark ten accepted steps, or a shorter crystal that completes in seconds,
for plane-wave, line and small Cartesian problems, both grids, with and without
dispersion filters. Report translation separately, cold and warm execution,
all grid sizes, support, threads, memory, accepted/rejected steps and errors
in weak generated bands. Record the measured window in
[the benchmark ledger](../benchmarks.md); do not extrapolate to full-size runs.

#### 6.5 Size of the change

The pointwise tensor body is a small component. This is a **large feature across
physics, translation, every backend, statistics, output and material curation**.
Expect changes in several dozen existing/new source, test and manual files
across solprop and solprop-kokkos, plus independent optmat work. The precise
count depends on the accepted representation and cannot be inferred from the
number of tensor coefficients.

For planning by a contributor already familiar with the seam, a rough effort
range is 2–4 person-weeks for mathematical/material-contract qualification,
4–8 for a tested reference route and physical outputs, 3–5 each for Numba and
Kokkos CPU parity, and 3–6 for envelope/domain/noise qualification and complete
documentation. These overlap imperfectly and suggest roughly **15–28
person-weeks**, excluding waiting for papers, upstream curation and separate
device qualification. They are engineering estimates, not a delivery schedule.
Noise validity, full spatial flux, interfaces and table-memory design carry
more uncertainty than the quadratic contraction. A field-only collinear prototype
would be much smaller but would not complete the proposed scope.

### 7. Verification and validation

#### 7.1 Independent references and error budgets

Keep three questions separate: whether the discrete implementation solves its
equations, whether the forward/model approximations are adequate, and whether
the selected material data describe the experiment. Backend parity answers only
the first of these. Agreement with another forward code sharing the same
approximations is useful evidence, not the oracle.

For short sums use a roundoff budget based on
$\gamma_m=m\epsilon/(1-m\epsilon)$, multiplied by the relevant operand norms
and map condition numbers. FFT budgets follow the existing Cartesian transform
analysis. No fixed relative tolerance near a tensor zero, a degenerate
eigenvalue, zero generated energy or exact phase matching is meaningful:
use absolute dimensional error bounds there.

| Test family | Independent comparison or invariant | Quantity and tolerance source |
|---|---|---|
| Tensor contraction | Explicit 3×3×3 sum against contracted form; single-tone and two-tone expansions | Full vector P, mixed factors and SI dimensions; operation-count roundoff |
| Rotations | Rotate all fields/tensors/geometry together and compare laboratory observables | Covariance, signs, 180-degree rotations and forbidden components; norm/conditioning bounds |
| Crystal symmetry | Point-group transformations leave the declared tensor invariant | Exact forbidden entries and transformed allowed components; source uncertainty separate from roundoff |
| Linear modes | Maxwell determinant/nullspace and transverse-displacement eigenproblem against the 4×4 construction | Root and E/H residuals, flux, isotropic and uniaxial limits; conditioned backward error |
| Degeneracies | Rotate an arbitrary basis inside an exactly degenerate subspace | Propagator, source and flux independent of eigenvector gauge; projector error |
| Walk-off/diffraction | Narrow Gaussian centroid against $-\nabla_{\mathbf q}k_zL$; paraxial Hessian solution against exact surface | Centroid and widths; angular Taylor remainder plus finite-window/sampling error |
| Delay/dispersion | Narrowband pulse delay against $\partial_\omega k_zL$ at fixed transverse wavevector | Arrival, GVM and phase; Taylor truncation and time-window error |
| Interfaces | Independent tangential Maxwell solve and isotropic Fresnel limits | E/H continuity, R+T and phase; conditioned linear-solve error |
| Undepleted conversion | SHG sinc-squared curve and distinct-frequency SFG solution | Complex generated amplitude and efficiency versus length/mismatch; pump-depletion and envelope approximation bounded separately |
| Depleted conversion | Independent high-accuracy coupled-wave ODE solve [1] | Pump/seed/idler amplitudes, phases and Manley–Rowe invariants; ODE reference tolerance below tested spatial error |
| OPA | Analytic gain including mismatch, plus phase-sensitive seeded degenerate cases | Signal gain, idler phase and deamplification; roundoff and approximation budget |
| Cascading | Large-mismatch asymptotic phase, both mismatch signs | Effective phase and correction scaling; inverse-mismatch expansion, not a fitted n2 |
| Dispersive response | Small-grid direct signed-frequency convolution | Complex P for several triads, including negative inputs; quadrature/roundoff with exactly the same filters |
| Energy and loss | Poynting flux, polarization work and prescribed passive dissipation | Total axial energy, boundary losses and no double-counted Fresnel factors; deterministic reduction plus discretization budget |
| Domains | Resolved periodic domains against $2/\pi$ long-scale limit | Complex conversion and convergence with domains per interaction length; averaging remainder and stage landing |
| Noise | Vacuum ensemble and exact Bogoliubov gain/correlation | Half-photon baseline, corrected mean and variance; independently computed confidence intervals |
| Material models | Measured indices, phase-matching, temperature tuning and nonlinear calibration | Source uncertainty, model residual and angular/thermal metrology, never backend roundoff |

An independent Maxwell check should use a small linear boundary-value calculation
and a weakly nonlinear resolved-wave reference, not the same projected source
routine invoked twice. Compare forward/backward energy, longitudinal fields and
quadratic corrections as intensity and angular aperture vary. This is a bounded
qualification experiment, not an FDTD backend deliverable. Published anisotropic
frequency-domain theory [21] provides another consistency reference, with its
approximations read explicitly.

#### 7.2 Discretization and backend matrix

For deterministic cases refine at least three levels separately in step size,
time bandwidth/window, transverse spacing/domain and any material-table
interpolation. Confirm the order appropriate to the selected stepper on smooth
problems; domain edges must land exactly. For spectral convergence, measure
errors against the band/window tails and aliasing support instead of claiming a
universal algebraic order.

Qualification must include:

- One linear application, one nonlinear evaluation, one step, eight steps,
  each statistic and one complete *short* run, across NumPy, Numba and Kokkos CPU.
- Field/envelope parity in complex amplitudes and physical flux, with identical
  support, phase origin, frame and launch; sum, difference, back-conversion and
  cascaded harmonics, not only one isolated harmonic.
- BBO type I and II, an out-of-principal-plane LBO orientation, LBO noncritical
  temperature tuning, and a generic supplied tensor/dielectric model to detect
  material-name special cases.
- Both transverse directions in a Cartesian problem; a line test agreeing with
  a y-invariant Cartesian problem in the correct per-width units.
- A noncollinear three-wave case with the expected transverse momentum sum,
  phase mismatch and spatial separation.
- Serial/parallel repeatability, scan reproducibility, saved/restarted versus
  uninterrupted propagation, and batch/member parity wherever admitted.
- Failure tests for unsupported data bands, invalid signed axis maps, singular
  U, nonpassive loss, insufficient padding, unresolved domain walls, missing
  noise measure and unsupported backend capabilities.

The pump can hide an error in a weak generated channel in the existing global
norm. Phase 0 must assess that controller against weak-band *absolute* amplitude
and conversion goals. If necessary, introduce an explicit band/branch error
weight with a physical floor, tested at vanishing seed and at depletion.
Do not claim an OPG or tiny-conversion tolerance from a pump-dominated norm.
If the controller change is broader than this route, agree its separate design
before altering existing default runs.

#### 7.3 Experimental validation and benchmarks

Use sources [6,8–20] to build small validation records with the actual
temperature, reference wavelength, axes, cut, input polarization and uncertainty.
For BBO, test angular phase matching and nonlinear calibration independently:
a wrong d cannot repair a wrong phase-matching angle. For LBO, qualify angular
and noncritical temperature tuning before interpreting conversion efficiency.
Historical room-temperature models and newer thermal models are comparisons,
not automatic replacements for one another.

An OPG experimental comparison also needs the pump's temporal/spatial profile,
collection angle, spectral resolution and ensemble statistics. Without those,
phase-matching wavelengths can be checked, but absolute measured conversion
cannot be certified. The acquisition register explicitly retains this gap.

Performance entries follow §6.4 and the ten-step rule. Test and lane runs must
not execute a full experimental propagation. A worked gallery example can run
a real crystal problem by the separate gallery procedure, with its outputs and
actual runtime committed.

### 8. Documentation

Implement a manual page *docs/model/crystal-optics.md* carrying the current
Maxwell derivation, coordinates, flux, tensor convention, frame, material-model
limits, phase matching, interfaces, domain reversal and OPG interpretation.
Add a short user-guide page for setting up BBO/LBO, choosing field versus
envelope grids, and diagnosing missing idler bands or insufficient angular
resolution. Its executable snippets remain short.

The first reference phase should include a small deterministic BBO SHG tutorial
and a line example showing spatial walk-off. Later add LBO temperature tuning,
noncollinear OPA and a small OPG ensemble comparison. A full-size focused
conversion example belongs in the manually executed worked gallery; it must
not become a test or normal timing.

Update the free-space, Cartesian, nonlinear, materials, noise, statistics,
running, backend-selection and output/processing pages where behavior changes.
Update architecture, conventions, kernel contract and benchmark ledger in the
same implementation branches. Record deliberate differences on
[Differences from Luna](../../guide/workflows/luna.md), including any difference in transverse
field approximation, envelope carrier handling, source normalization and noise.
The current documentation-only proposal does not claim those behaviors exist.

Every new public class/function needs units, coordinate conventions, array
shapes, validity, named refusals and runnable examples in its NumPy-style
docstring. Material references belong to the data records as well as adapter
documentation. At Final, manual pages link to this SEP and the changelog cites it.

### 9. Compatibility and impact

The crystal feature does not change existing scalar spatial, gas, capillary,
modal and GNLSE defaults. SEP 34 separately changes the shared uniform-medium
scalar spatial flux convention. The crystal leaf is absent on
those problems and allocates no crystal maps or physical-component buffers.
No scalar gas-density normalization is applied to a bulk crystal susceptibility.
The existing one-photon noise models retain their names and meaning.

The new route changes the linear operator from scalar to local 2×2, its
response projection from a scalar prefactor to a vector map, and its spatial
observables from a vacuum weight to E/H flux. These are substantive seam and
output changes, even though the outer state dimensions remain familiar.
Output needs a new schema version when the first crystal files are written:
record units, coordinate/reference identities and the observable model
explicitly. Older processing must refuse an unknown crystal schema rather
than calculate modal energy from its arrays. The kernel version and output
version remain independent.

Storing tangential laboratory E preserves the existing x/y interpretation at
the public boundary; reconstruction supplies Ez/H and branch diagnostics.
HDF5 spectra remain in the optics convention and internal restart conversion
uses the named boundary helpers. Geometry-dependent complex maps must obey
the transverse negative-wavevector/Fourier-origin convention as well as temporal
conjugation; tests must cover both, not insert ad hoc sign changes in kernels.

The general spatial vacuum-flux issue on the roadmap is solved here **only
for the new crystal route**. [SEP 34](sep-0034-medium-flux.md) owns the
uniform isotropic scalar flux correction and its clean breaking change, with
no legacy normalisation or old-file migration. Its scalar admittance must not
replace this proposal's anisotropic vector E/H metric or interface solver.
Likewise, a piecewise crystal *domain sign* does not implement a continuously
varying dielectric medium or an orientation path.

The optmat requirement and any minimum public API version are agreed through
SEP 3 and Phase 0. No dependency is added by this draft. A bundled coefficient
fallback would defeat the proposed provenance contract and is not recommended.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Add a scalar d_eff response to today's free-space medium | Useful for a fixed collinear teaching model, but cannot supply general crystal modes, rotations, flux or noncollinear mixing |
| Treat laboratory x/y as independent ordinary/extraordinary branches | Valid only in restricted planes and frames; fails arbitrary biaxial orientations and frequency-dependent eigenvectors |
| Store eigenbranch amplitudes as the public polarization axis | Simplifies linear stepping but changes x/y semantics, exposes gauge choices and complicates output; retain lab transverse amplitudes and factorize internally |
| Carry three independent E components as propagation state | Introduces a constrained longitudinal degree of freedom and redundant integration; reconstruct it from forward Maxwell modes instead |
| Extend the existing dense general-modal route directly | Reuses E/H ideas but can accidentally create dense transverse coupling and waste storage; share validated small records/helpers where appropriate, keep FFT geometry |
| Propagate separate pump/signal/idler envelopes | Often fastest for narrow isolated bands; excludes unanticipated cascades and overlapping broadband bands unless repeatedly extended; keep as an independent reduced reference |
| Use a literal full signed-frequency convolution in production | General and clear, but quadratic spectral cost; use for small oracle checks and assess low-rank physical models |
| Use one frequency-independent measured tensor for every process | Avoids filters but can violate reciprocity or misrepresent dispersion; allow only a named, qualified conservative approximation |
| Promote Miller scaling to an exact law | Unsupported near resonances and not fully determined by sparse data; retain fit residuals and qualified domains |
| Reuse a random-phase one-photon seed as vacuum OPG | Changes noise moments and spontaneous gain baseline; use an explicitly derived separate model |
| Add full FDTD or bidirectional Maxwell propagation | Broader interface/near-field validity but a different state, time integration and cost model; outside this forward-route SEP |

The publication on heterogeneous nonlinear simulation [22] demonstrates the
practical relevance of broadband crystal calculations, but does not establish
solprop's performance or replace its derivation and independent qualification.

### 11. Phases

Each phase is independently mergeable, carries a Part II report and uses a
topic branch of the form *feature/sep-0008-quadratic-crystal-optics-PN*.
All implementation phases run the full gate, slow lane and coverage lane
required by [testing](../testing.md), with Kokkos CPU included when installed.
Documentation-only curation/planning updates follow the documentation checks.
Backends without a newly introduced capability refuse it explicitly.

#### Phase 0 — Mathematical and material contract qualification

Resolve Q1–Q6 below: material frame and source choices, conservative response,
forward projection, support, flux and noise scope. Obtain the P0 source packet
and agree the optmat public bundle contract. Derive/check the first-order
Maxwell source and weak-loss model against independent small problems.
Measure conditioning and anticipated table memory, and evaluate the existing
controller on weak generated bands. Write decision records for lasting choices.
This phase may merge derivations and bounded qualification tools, but must
not enable unqualified production crystal defaults.

#### Phase 1 — NumPy field route, deterministic conversion and physical outputs

Deliver the constant conservative quadratic response, translated homogeneous
crystal maps, exact local polarization flow, three physical components, plane/
line/Cartesian fields, uniform temperature, bare linear interfaces and true
flux. Include BBO/LBO material selections only after the upstream records pass
curation. Provide deterministic SHG/SFG/DFG/OPA, depletion and phase-matching
tools, support checks, memory admission, output/resume metadata and the first
manual/tutorial. Bump kernel/output schemas and add explicit compiled-backend
refusals. Qualify the tensor, Maxwell, propagation and flux rows of §7.

#### Phase 2 — NumPy envelope, nonlinear dispersion and resolved domains

Add the exact analytic-envelope products/projector, carrier/frame admission,
qualified material-frame filters, and resolved domain-sign kinks. Test complex
field/envelope parity, direct-convolution agreement, mixed response operands,
lossless exchange and the averaged-domain limit. Qualify weak passive loss
as a deterministic feature if Phase 0 admitted it. Any seam extension gets its
own version bump; NumPy remains the oracle for every new leaf.

#### Phase 3 — Numba CPU deterministic parity

Implement Phases 1–2 with the existing body-factory and cache policies.
Verify the complete short-run/observable/backend matrix, deterministic reductions,
memory ledger, restart, thread counts and refusal cases. Record bounded warm/cold
performance. Optimize map storage only with errors measured against the dense
oracle and an updated memory admission bound.

#### Phase 4 — Kokkos CPU deterministic parity

Implement the identical leaves and arithmetic in solprop-kokkos, preserving
resident arrays and native transform plans. Qualify Phases 1–2 against NumPy
and Numba, including three-component physics, interfaces/observations,
filters/domains and low generated signal. Record its actual peak memory and
short-window cost. CUDA/HIP runs are a separate device qualification, not inferred
from a successful compilation.

#### Phase 5 — Explicit OPG ensemble qualification

First implement the flux-normalized noise, observables and oracle on NumPy;
then qualify the same scope on Numba and Kokkos CPU before closing the phase.
Verify exact undepleted-pump gain and correlations, seed invariance across
execution layouts, physical-band versus padding refinement, and independent
small-system evidence for the admitted depleted/high-occupation regime.
Require an explicit ensemble accuracy report and limitations in the manual.
If no defensible depleted-pump regime is established, the supported OPG scope
remains undepleted/qualified only; acceptance must specify whether that
restricted result satisfies this phase, rather than calling arbitrary classical
noise propagation a completed quantum model.

#### Phase 6 — Integrated qualification and closure

Qualify composition, Local/Pool scans and any admitted Batch execution,
bounded-memory large-grid admission and curated-material provenance. Complete
the manuals and worked examples, source-check register and benchmark ledger.
Move every deferred capability to the roadmap or an issue. Final requires
NumPy/Numba/Kokkos CPU parity for **all accepted features**, including the
accepted OPG scope, or a documented exception approved in the accepted SEP.
Move the SEP to the archive only after those criteria are met.

### 12. Open questions

These are decisions for acceptance or the named phase, not silent assumptions.
The draft recommends a direction so they can be reviewed concretely.

1. **Q1 — Material frames and source bundles; before Phase 1.** Which exact
   BBO and LBO linear/thermal/quadratic records become recommended, and what
   signed axes and reference frequency tuples do they use? Recommend comparing
   the P0 sources and measured phase-matching residuals before selection;
   neither the newest fit nor a familiar coefficient table wins by default.
2. **Q2 — Conservative nonlinear model; before acceptance.** Are the constant
   dispersionless and symmetric Miller-type models sufficient for the first
   qualified wavelength ranges? Recommend both as explicit choices, with no
   forced symmetrization of measurements and no resonant extrapolation.
   Assign unresolved weak/sign/temperature data to optmat curation.
3. **Q3 — Forward/vector accuracy; Phase 0.** What bounds on angle, nonlinear
   perturbation and root conditioning make the neglected backward and nonlinear
   longitudinal feedback acceptable? Recommend a derived source projection with
   linear three-component synthesis, certified against a small Maxwell reference;
   do not promise arbitrary numerical aperture.
4. **Q4 — Envelope lattice and bandwidth; Phase 0.** Require bin-aligned
   carriers initially, or qualify a more general exact remapping? Recommend
   the simpler explicit requirement unless existing grid semantics require a
   different solution. Transverse product aliasing needs the same early decision.
5. **Q5 — Flux and weak-band controller; Phase 0.** Agree the crystal output
   schema and whether a route-specific band error allocation is needed.
   Recommend true E/H flux from the start and explicit absolute weak-band goals.
6. **Q6 — OPG scope; before acceptance.** Is exact undepleted-pump statistical
   qualification plus bounded semiclassical depleted operation an adequate first
   capability? Recommend the separate opt-in model in §4.10, initially lossless;
   do not conflate it with exact spontaneous/entangled-state simulation.
7. **Q7 — Material loss and interfaces; Phase 0.** Agree the perturbative-loss
   admission test and external media for initial interface qualification.
   Recommend bare plane faces with reflected energy accounted for and a
   separately explicit already-internal launch.
8. **Q8 — Table representation; Phases 1–4.** What compact generator/map
   representation meets memory and interpolation-error budgets, including optic
   axes? Start with the dense oracle on admitted sizes; record the chosen
   compression and its error independently of backend arithmetic.
9. **Q9 — Missing crystal responses; curation and roadmap.** Which intrinsic
   Kerr/absorption data are adequate for the chosen high-intensity examples?
   Keep incomplete tensor, Raman, thermal/damage and THz extensions separate;
   pure-quadratic examples must state what they omit.
10. **Q10 — Guided and varying-media extensions; roadmap.** Reuse the material
    and quadratic contracts later for general modal overlaps and profiles, but
    derive the overlap normalization, moving basis and interface behavior in
    their own scope. Discussion here does not count as delivered coverage.

### 13. References

The mathematical construction above is derived for solprop's conventions.
References supply independent theory, experiments and material-data candidates.
The source-check distinctions in §4.12 remain part of the proposal: bibliography
verification is not verification of all coefficients in an article.

1. J. A. Armstrong, N. Bloembergen, J. Ducuing and P. S. Pershan,
   “Interactions between Light Waves in a Nonlinear Dielectric,”
   *Physical Review* **127**, 1918–1939 (1962).
   [doi:10.1103/PhysRev.127.1918](https://doi.org/10.1103/PhysRev.127.1918).
2. D. W. Berreman, “Optics in Stratified and Anisotropic Media: 4×4-Matrix
   Formulation,” *JOSA* **62**, 502–510 (1972).
   [doi:10.1364/JOSA.62.000502](https://doi.org/10.1364/JOSA.62.000502).
3. M. Kolesik and J. V. Moloney, “Nonlinear optical pulse propagation simulation:
   From Maxwell's to unidirectional equations,” *Physical Review E* **70**,
   036604 (2004).
   [doi:10.1103/PhysRevE.70.036604](https://doi.org/10.1103/PhysRevE.70.036604).
4. M. Conforti, F. Baronio and C. De Angelis, “Nonlinear envelope equation for
   broadband optical pulses in quadratic media,” *Physical Review A* **81**,
   053841 (2010).
   [doi:10.1103/PhysRevA.81.053841](https://doi.org/10.1103/PhysRevA.81.053841);
   [author manuscript](https://arxiv.org/abs/1001.1874).
5. M. M. Fejer, G. A. Magel, D. H. Jundt and R. L. Byer, “Quasi-phase-matched
   second harmonic generation: tuning and tolerances,” *IEEE Journal of Quantum
   Electronics* **28**, 2631–2654 (1992).
   [doi:10.1109/3.161322](https://doi.org/10.1109/3.161322).
6. W. J. Alford and A. V. Smith, “Wavelength variation of the second-order
   nonlinear coefficients of KNbO₃, KTiOPO₄, KTiOAsO₄, LiNbO₃, LiIO₃,
   β-BaB₂O₄, KH₂PO₄, and LiB₃O₅ crystals: a test of Miller wavelength scaling,”
   *JOSA B* **18**, 524–533 (2001).
   [doi:10.1364/JOSAB.18.000524](https://doi.org/10.1364/JOSAB.18.000524).
7. M. Bache, H. Guo, B. Zhou and X. Zeng, “The anisotropic Kerr nonlinear
   refractive index of the beta-barium borate (β-BaB₂O₄) nonlinear crystal,”
   *Optical Materials Express* **3**, 357–382 (2013).
   [doi:10.1364/OME.3.000357](https://doi.org/10.1364/OME.3.000357);
   [author version, revision 3](https://arxiv.org/abs/1209.3158v3).
8. D. Eimerl, L. Davis, S. Velsko, E. K. Graham and A. Zalkin, “Optical,
   mechanical, and thermal properties of barium borate,” *Journal of Applied
   Physics* **62**, 1968–1983 (1987).
   [doi:10.1063/1.339536](https://doi.org/10.1063/1.339536).
9. K. Kato, N. Umemura and T. Mikami, “Sellmeier and thermo-optic dispersion
   formulas for β-BaB₂O₄ (revisited),” *Proceedings of SPIE* **7582**,
   75821L (2010).
   [doi:10.1117/12.841913](https://doi.org/10.1117/12.841913).
10. K. Kato, N. Umemura, T. Okamoto and K. Miyata, “Updated Sellmeier equations
    of β-BaB₂O₄,” *Nonlinear Frequency Generation and Conversion: Materials and
    Devices XX*, SPIE (2021).
    [doi:10.1117/12.2576038](https://doi.org/10.1117/12.2576038).
11. I. Shoji et al., “Absolute measurement of second-order nonlinear-optical
    coefficients of β-BaB₂O₄ for visible to ultraviolet second-harmonic
    wavelengths,” *JOSA B* **16**, 620–624 (1999).
    [doi:10.1364/JOSAB.16.000620](https://doi.org/10.1364/JOSAB.16.000620).
12. C. Chen, Y. Wu, A. Jiang, B. Wu, G. You, R. Li and S. Lin, “New
    nonlinear-optical crystal: LiB₃O₅,” *JOSA B* **6**, 616–621 (1989).
    [doi:10.1364/JOSAB.6.000616](https://doi.org/10.1364/JOSAB.6.000616).
13. S. P. Velsko, M. Webb, L. Davis and C. Huang, “Phase-matched harmonic
    generation in lithium triborate (LBO),” *IEEE Journal of Quantum Electronics*
    **27**, 2182–2192 (1991).
    [doi:10.1109/3.135177](https://doi.org/10.1109/3.135177).
14. K. Kato, “Temperature-tuned 90° phase-matching properties of LiB₃O₅,”
    *IEEE Journal of Quantum Electronics* **30**, 2950–2952 (1994).
    [doi:10.1109/3.362711](https://doi.org/10.1109/3.362711).
15. K. Kato, S. G. Grechin and N. Umemura, “New thermo-optic dispersion formula
    for LiB₃O₅,” *Laser Physics* **28**, 095403 (2018).
    [doi:10.1088/1555-6611/aac9df](https://doi.org/10.1088/1555-6611/aac9df).
16. Y. Tang, Y. Cui and M. H. Dunn, “Thermal dependence of the principal
    refractive indices of lithium triborate,” *JOSA B* **12**, 638 (1995).
    [doi:10.1364/JOSAB.12.000638](https://doi.org/10.1364/JOSAB.12.000638).
17. D. Zhang, Y. Kong and J.-Y. Zhang, “Optical parametric properties of
    532-nm-pumped beta-barium-borate near the infrared absorption edge,”
    *Optics Communications* **184**, 485–491 (2000).
    [doi:10.1016/S0030-4018(00)00968-8](https://doi.org/10.1016/S0030-4018(00)00968-8).
18. R. DeSalvo, A. A. Said, D. J. Hagan, E. W. Van Stryland and M. Sheik-Bahae,
    “Infrared to ultraviolet measurements of two-photon absorption and n₂ in
    wide bandgap solids,” *IEEE Journal of Quantum Electronics* **32**,
    1324–1333 (1996).
    [doi:10.1109/3.511545](https://doi.org/10.1109/3.511545).
19. A. Dragomir, J. G. McInerney and D. N. Nikogosyan, “Femtosecond measurements
    of two-photon absorption coefficients at λ = 264 nm in glasses, crystals,
    and liquids,” *Applied Optics* **41**, 4365–4376 (2002).
    [doi:10.1364/AO.41.004365](https://doi.org/10.1364/AO.41.004365).
    Erratum: **41**, 5655 (2002),
    [doi:10.1364/AO.41.005655](https://doi.org/10.1364/AO.41.005655).
    Some records of the original use “Dragonmir”; the publisher's current
    author/citation entries use “Dragomir.” Match the work by DOI and check the
    correction before numerical curation.
20. F. Hanson and D. Dick, “Blue parametric generation from temperature-tuned
    LiB₃O₅,” *Optics Letters* **16**, 205–207 (1991).
    [doi:10.1364/OL.16.000205](https://doi.org/10.1364/OL.16.000205).
21. H. Guo, X. Zeng, B. Zhou and M. Bache, “Nonlinear wave equation in frequency
    domain: accurate modeling of ultrafast interaction in anisotropic nonlinear
    media,” *JOSA B* **30**, 494–504 (2013).
    [doi:10.1364/JOSAB.30.000494](https://doi.org/10.1364/JOSAB.30.000494).
22. N. Karpowicz, “Open source, heterogeneous, nonlinear optics simulation,”
    *Optics Continuum* **2**, 2244–2254 (2023).
    [doi:10.1364/OPTCON.502567](https://doi.org/10.1364/OPTCON.502567).
23. C. M. Caves, “Quantum limits on noise in linear amplifiers,”
    *Physical Review D* **26**, 1817 (1982).
    [doi:10.1103/PhysRevD.26.1817](https://doi.org/10.1103/PhysRevD.26.1817).
24. A. Polkovnikov, “Phase space representation of quantum dynamics,”
    *Annals of Physics* **325**, 1790–1852 (2010).
    [doi:10.1016/j.aop.2010.02.006](https://doi.org/10.1016/j.aop.2010.02.006);
    [author manuscript](https://arxiv.org/abs/0905.3384).
25. S. Lin, Z. Sun, B. Wu and C. Chen, “The nonlinear optical characteristics
    of a LiB₃O₅ crystal,” *Journal of Applied Physics* **67**, 634–638 (1990).
    [doi:10.1063/1.345765](https://doi.org/10.1063/1.345765).
26. B. Maingot, E. Neradovskaia, C. Claudet, N. Forget and A. Jullien,
    “Measurement of nonlinear refractive indices of bulk and liquid crystals by
    nonlinear chirped interferometry,” *Optics Letters* **48**, 3243–3246 (2023).
    [doi:10.1364/OL.487261](https://doi.org/10.1364/OL.487261).

## Part II — Implementation record

## Part III — History

- 2026-09-27: Drafted. Proposed a Maxwell-based vector crystal route, constant
  and dispersive quadratic responses, BBO/LBO material curation, explicit
  statistical OPG qualification, and staged qualification on all CPU backends.
- 2026-10-22: Linked SEP 31's shared uniform isotropic scalar flux work and clean
  breaking changes. The crystal E/H metric, interfaces and quadratic physics
  remain separate; no scalar-flux substitution is implied.

- 2026-10-23: Reassigned the shared medium-flux prerequisite to
  [SEP 34](sep-0034-medium-flux.md) following SEP 31's review and split.
  SEP 31 retains field exchange and coupling; the shared flux proposal
  includes admitted radial adiabatic transport and complex local flux.
