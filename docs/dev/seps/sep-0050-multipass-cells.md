# SEP 50 — Multi-pass cells assembled from free-space passes

| | |
|---|---|
| **SEP** | 50 |
| **Title** | Multi-pass cells assembled from free-space passes |
| **Author** | Matthieu Guer |
| **Drafting agent** | Claude Opus 5.5 (2026-11-12); Claude Fable 5.2 (revision after the AI review, 2026-11-13) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-11-12 |
| **Resolution** | — |
| **Requires** | SEP 22, SEP 31, SEP 34, SEP 40, SEP 47 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

A multi-pass cell (MPC) folds a long nonlinear path between two concave mirrors: a pulse
crosses a gas fill, or one or more bulk plates, tens of times, and is refocused at every
bounce. This SEP proposes to simulate gas-filled and solid-state MPCs in solprop without
a new propagation route and without a change to the seam. A cell is described once, in
the physics layer, together with its linear and Kerr-corrected design: the eigenmode of
the cell, the beam sizes at the mirrors and at the centre, the critical-power ratio and
the B-integral per pass, after Hanna et al. [4]. A driver-level executor then runs the
cell as an ordered sequence of ordinary free-space runs, one per nonlinear segment and
pass, joined at each mirror by the linear elements of SEP 31, in an unfolded
optical-path coordinate. Each nonlinear segment keeps one translated problem and one
backend for every pass (`SimulationRunner`, SEP 22). The cell owns the chain's energy
ledger, its noise identity and its resume, and a scan accepts a builder of cells as it
accepts a builder of simulations. With plates the Kerr correction of [4], derived for a
fill, no longer holds as published; the design keeps its moment form segment by segment
and states the closure it then rests on (section 4.4). Success is judged by the
short-segment checks listed in [#191](https://github.com/LupoLab/solprop/issues/191) as
tests, by the design reproducing the published design numbers of [2], [4], [10], [12] and
[13], each in its publication's conventions, and by worked examples that reproduce the
beam-size evolution of [4], the spectra and compressed durations of the four argon cells
of [10], and the spectra of a solid-state cell [12, 13, 14] within stated tolerances.

### 2. Motivation and scope

Gas-filled MPCs are the standard post-compression stage for millijoule ytterbium lasers:
pulses of 300–700 fs are broadened by self-phase modulation over 15–30 passes in argon
and compressed to below 20 fs with transmissions above 90 % [9, 10]. Bulk MPCs do the same
with fused silica inside the cell, first the mirror substrates themselves [2], then plates
in air or in vacuum [12, 13, 15], up to 30 mJ [14]. Their design is a compromise
between the B-integral per pass, the peak power against the critical power, the fluence
on the mirrors, ionisation at the focus and the spatio-spectral homogeneity of the output,
and the beam size from pass to pass depends on the Kerr lens: launched on the linear
eigenmode, a cell near the critical power drifts by tens of per cent in mirror fluence
[4, 10]. Analytic and one-dimensional models size a cell; only a spatio-temporal
propagation over all passes shows the drift, the aberrations of the Kerr lens, the
homogeneity of the broadened spectrum and the compressibility of the output.

solprop already propagates the field of a beam in a bulk medium with the exact
longitudinal operator, Kerr, plasma and Raman responses, on radial and Cartesian grids,
and SEP 31 supplies field planes, thin lenses, mirrors, interfaces, spectral optics and
the handoff of a plane into another run. Nothing assembles them into a cell
([#191](https://github.com/LupoLab/solprop/issues/191)).

**In scope.**

- A two-mirror cell with spherical mirrors of the same radius of curvature $R$ and
  separation $L$, $0 < L < 2R$, simulated in its unfolded, on-axis equivalent on a
  `RadialGrid`; the Cartesian grid admitted for the same symmetric cell, as a check. The
  two mirrors may carry different coatings, as a complementary pair of chirped mirrors
  does [15].
- A gas fill at constant density, or an ambient medium (vacuum, a gas, air) with one or
  more plane-parallel plates of bulk media at normal incidence. A plate is declared once,
  by the interval it occupies from the first mirror, is crossed in the reverse order on
  the return pass (section 4.1) and carries its own nonlinear responses. A mirror crossed
  in its own substrate, as in [2], is a plate touching the mirror, crossed before and
  after the reflection, with the net power of the meniscus, $2/R$ in the thin limit,
  given to the mirror.
- Mirror coatings as frequency-only responses: reflectance, group-delay dispersion or a
  tabulated phase, through the existing optics models.
- The linear and Kerr-corrected design of the cell (sections 4.3 and 4.4) and a launch
  matched to its eigenmode at the first mirror.
- The executor, its ledger, its resume at pass granularity, and per-pass observables:
  energy, on-axis intensity at the centre, second-moment beam radius at the mirrors and
  at the centre, spectral width, the transform limit of the beam-integrated spectrum, the
  duration after one spectral transfer applied to the whole beam, and an on-axis
  intensity integral as a diagnostic of the B-integral (section 4.6).
- Scans of cells: `run_scan` accepts a builder that returns a cell, so the pressure,
  the energy, the length, the chirp of the launch, the plates or the coating can be
  scanned with the `Local`, `Pool`, `Queue` and `Slurm` executors, the existing resume
  and the collected file (section 5.5).

**Out of scope, explicitly.**

- The input and output coupling optics (entrance window, matching lenses, injection and
  extraction mirrors) and their nonlinearity. The first version starts at the first cell
  mirror; propagating the design back to the matching optics, as [4] section 4 describes,
  is follow-up work.
- The Herriott spot pattern on the mirrors, re-entrance conditions and injection angles.
  In the unfolded model they do not change the propagated field; the pattern only adds a
  fluence per spot, clipping at a mirror edge and, on a Cartesian grid, the astigmatism of
  oblique incidence. Follow-up work.
- Interaction between passes: overlap of the beams at the mirrors or the centre, residual
  plasma, heating and gas dynamics between pulses. Counter-propagating waves, as in
  [#191](https://github.com/LupoLab/solprop/issues/191).
- Brewster-angle plates, mirrors of different radii, astigmatic or off-axis mirrors,
  dispersion-engineered coating design, damage models of the mirrors.
- Mirror events inside a native march (section 10), resampling between grids of
  different size from pass to pass, and recipes and front-end hooks for cells (SEP 41).
- A launch of beam quality other than one. The design takes a beam quality as a
  parameter of its second-moment law (section 4.3); the launch it generates is an ideal
  Gaussian, and a measured or constructed field is the caller's own plane.
- Scans of cells on the `Batch` executor, which advances single kernel problems in
  lockstep (section 5.5), and a search for the focus along a pass: the fixed observation
  plane is the centre (section 4.6).

### 3. Context

**Free space.** A free-space run evolves the physical field on the order-zero Hankel
columns of a `RadialGrid` or the Fourier columns of a `FreeGrid`, with the exact
$k_z=\sqrt{n^2\omega^2/c^2-k_\perp^2}$, the axial flux of the medium as its energy measure
([the free-space model](../../guide/free-space/model.md), SEP 34) and transverse boundaries
watched by the monitor of SEP 40. A run's medium is constant along z, except a gas whose
pressure varies on the radial grid inside the adiabatic certificate
([#193](https://github.com/LupoLab/solprop/issues/193),
[#206](https://github.com/LupoLab/solprop/issues/206)): a gap followed by a plate is two
runs, not one.

**Coupling.** SEP 31 ([coupling simulations](../../guide/workflows/coupling.md)) gives the
`FieldPlane` with its affine reference, `transport` through `ThinLens`, `FreeSpace`,
`PlanarInterface`, `PolarisationTransform`, `Mirror` and `SpectralTransfer`, each with a
report and, for a screen, a thin-screen certificate; `Simulation.input_port`, `couple`
with the `same_basis` model and `Simulation.with_input` relaunch a plane into a run on
the same lattice and basis. A spatial port admits only `same_basis`, in the same medium,
at the same position and with the same orientation, so a transported plane is declared at
the receiving entrance (`transport(..., geometry=port.geometry)`). `Mirror` turns the
plane's forward axis, which `couple` then refuses as oriented differently: a folded cell
cannot be relaunched as it stands, and section 4.1 unfolds it. Two capillaries relaunched
this way reproduce one to the tolerance's level; SEP 31 section 9 names multi-pass-cell
assembly as future work. A `SpectralTransfer` records which affine spectral phase its
model drops: a `TaylorPhase` applies its coefficients as given, while `ChirpedMirror`
data lack the constant phase and the group delay, which leaves the plane's reference
uncalibrated in those two terms, and `FieldPlane.absolute()` then refuses.

**Reuse and run ownership.** `SimulationRunner` (SEP 22 section 5.2) keeps one translated
problem and one backend for many inputs; its `run` takes a pulse or an array, not a
`CouplingLaunch` (an array drops the coupling record and the inherited reference), and it
refuses an input whose noise model differs from its own. Its public `run` takes no stop
flag: a scan stops a running point through the runner's private path. SEP 47 keeps
orchestration between simulations in Python and forbids a change of physics inside a run
or a Python hook inside a native march (SEP 47 section 5.1): a mirror can only act
between runs, which is what this SEP does.

**Scans.** `run_scan` ([scans](../../guide/workflows/scans.md)) takes a `Scan` and a
builder that turns each `ScanPoint` into a `Simulation`, which it checks by type. `Local`
and `Pool` run whole points, `Queue` and `Slurm` drain them through pools, and `Batch`
advances compatible points as one lockstep group on one backend (SEP 47 section 6.7),
each a single kernel problem. It draws unseeded noise from the point's seed, or from the
scan's name and the point's index when the scan declares none, keeps one backend for
consecutive points that differ only in their input, stops a running point within a step,
skips completed points on a rerun and collects statistics, declared reductions and, when
asked (`fields`, `"none"` by default), fields into one file read by `ScanOutput`. A
point is one run.

**Design helpers.** `solprop.spatial.critical_power` gives the Townes baseline, its mass
1.86226 fixed in the code; `solprop.tools.beam_params`, `pulse_peak_power` and
`self_focusing_power` give the numbers of one Gaussian beam. Nothing computes a cavity
eigenmode, an ABCD propagation of the complex beam parameter or a Kerr-corrected beam.
`solprop.processing.compression.transform_limit` accepts a grid and a spectrum;
`apply_transfer` and the compressors read an output, and the compressors refuse a
spatial one, whose summed $|a|^2$ is not a power, asking for a physical-flux objective
that does not exist yet. `solprop.spatial.spatial_power` gives the axial power through a
plane in time. Among the spatial statistics, `peak_intensity` is the largest axial flux
over the transverse nodes and time, `on_axis_intensity` the largest in time on the axis,
and `beam_radius` carries two radii of the fluence, the second-moment radius
$\sqrt{2\langle r^2\rangle}$ and the $1/e^2$ radius.

**Issues.** This SEP answers [#191](https://github.com/LupoLab/solprop/issues/191) and
follows its list of qualification checks. It bears on:

- [#175](https://github.com/LupoLab/solprop/issues/175): a `FieldPlane` does not carry the
  frame it was written in; the executor passes the frame explicitly until it does.
- [#238](https://github.com/LupoLab/solprop/issues/238): a relaunched run's boundary
  monitor starts at zero and no object owns a chain's ledger; the cell owns its own
  (section 5.3).
- [#705](https://github.com/LupoLab/solprop/issues/705): a reused runner does not compare
  execution options; the executor builds each runner from one settings object, and the
  fix should precede phase 2.
- [#218](https://github.com/LupoLab/solprop/issues/218): `SpatialPulse` has a scalar waist;
  the matched launch, whose waist varies with frequency, is built with
  `FieldPlane.from_analytic` instead.
- [#618](https://github.com/LupoLab/solprop/issues/618) and
  [#78](https://github.com/LupoLab/solprop/issues/78): the transverse judgement of a
  relaunched run and the collar strength of a sliced run, which every pass after the first
  meets; `with_input` also re-checks a spatial run's transverse design for a launch that is
  not a multiple of the original, and warns when its band exceeds the designed one.
- [#206](https://github.com/LupoLab/solprop/issues/206) and
  [#193](https://github.com/LupoLab/solprop/issues/193): why a solid-state pass is a
  sequence of runs.

Two changes merged while this SEP was drafted: the split of `simulation.py`, which moved
`SimulationRunner` into `driver/runner.py` and replaced `from_radial` by `from_free` with a
`RadialGrid`, and SEP 49, which brought the TDSE response to native and left `numpy_ref` a
complete backend, with a new feature landing on both (ADR-077, ADR-068). The proposal adds
no backend code and is written against the concepts, not the names; the implementation
follows whichever names hold when it starts.

**Literature.** The cell is Herriott's [1]. MPC compression was first demonstrated in [2]
with the mirror substrates as the nonlinear medium: at each bounce the beam crosses 13 mm
of fused silica behind a concave anti-reflection face and is reflected by the convex back
face, and a transit of 18 bounces per mirror reaches an estimated B-integral of about
$5\pi$. [3] analysed the scheme; [4] derived the stationary beam in the presence of the
Kerr lens and the effective power of a pulse, for gas-filled cells only, noting that the
formalism "could be used" for cells with plates. [9] reviews the field; its Table 1 lists
three fused-silica and three argon cells with the mirror fluence, the peak intensity and
the B-integral computed for linear mode matching. [10] reports four argon cells designed
with the corrected ABCD model and compares a spatio-temporal simulation with experiment.
Four solid-state works serve here as references: two 2 mm plates in air with pre-chirped
200 µJ pulses and the spectrum measured pass by pass [12]; a hybrid cell of two 0.5 mm
plates in air whose nonlinear mode matching is published beside the linear one [13];
24 plates of 0.8 mm in vacuum at 30 mJ, near the edge of stability [14]; and two cells of
1 mm plates in air in series [15], in which nonlinear matching reduced the mirror radius
by about 30 % with five plates near the centre and raised it by about 15 % with two plates
near the mirrors.

### 4. Theory

SI units throughout; the optics sign convention at the interfaces of
[conventions.md](../conventions.md). $\lambda_0$ is the vacuum wavelength,
$k_0=2\pi/\lambda_0$, $n_0$ and $n_2$ the linear and nonlinear indices of the fill at the
carrier, $w$ the second-moment radius of the fluence, $w^2=2\langle r^2\rangle$, which is
the $1/e^2$ intensity radius of a Gaussian, and $M^2$ the beam quality, one for a
Gaussian.

#### 4.1 The unfolded cell

Unfold the cell about each mirror: pass $j=0,1,\dots,N-1$ propagates over
$z\in[0,L]$ in its own coordinates, and the unfolded optical-path position is
$z_{\rm abs}=jL+z$. A spherical mirror of radius $R$ adds the path $2s(r)$ of its sag
$s(r)=R-\sqrt{R^2-r^2}=r^2/2R+O(r^4/R^3)$ across the beam. In the paraxial limit the
reflected field, written in the next pass's coordinates, is

$$
\tilde E_{j+1}(r,\omega,0)=r_c(\omega)\,e^{-i k(\omega) r^2/R}\,\tilde E_j(r,\omega,L),
\qquad k=\operatorname{Re}n\,\omega/c ,
$$

a thin lens of focal length $R/2$ in the ambient medium followed by the coating's
amplitude reflection coefficient $r_c(\omega)$. Pass $j$ ends on the second mirror when $j$
is even and on the first when it is odd, and takes that mirror's $r_c$; the lens is the
same for both, so a pair of coatings leaves the design unchanged. The quartic term of the
sag is $k r^4/(4R^3)$; it is below $10^{-2}$ rad for a 2.5 mm beam on a 1 m mirror at
1 µm and is dropped, with the paraxial error of the screen bounded by SEP 31's thin-screen
certificate for each field (section 4.6 of SEP 31). The real cell reflects the beam: with
`Mirror`'s convention the outgoing local axes keep the p/s components of the incoming
ones and the pattern is imaged along p. A rotation by $\pi$ about the s axis carries the
outgoing frame $(\hat p_{\rm out},\hat s,\hat k_{\rm out})$, with
$\hat k_{\rm out}=-\hat k_{\rm in}$ at normal incidence, back onto the incoming one; in the
unfolded coordinates the reflection is therefore the Jones map
$\operatorname{diag}(r_p,r_s)$ in the local axes, the image $x\to-x$ along p (an identity
for each component on a radial grid) and the screen above. At normal incidence a
dielectric coating has $r_p=-r_s$ in this convention, so a circular polarisation changes
handedness at each bounce, as it physically does; an isotropic Kerr response is
unaffected.

**Planes and reflections.** Pass $j$ starts at its entrance plane, $z=0$, just after the
reflection that precedes it, and ends at its exit plane, $z=L$, where it arrives on the
mirror, before that reflection. The launch is the entrance plane of pass 0: the field
leaving the first mirror, whose reflection is not applied to it. A cell of $N$ passes
therefore applies $N-1$ reflections, the one after pass $j$ on the second mirror when $j$
is even and on the first when it is odd, and its result is the exit plane of pass $N-1$,
before any further reflection: the plane where the beam is picked off in a Herriott cell.
Injection and extraction mirrors, which in [10] carry the coating of the cell mirrors,
belong to the coupling optics (section 2), and the caller applies them to the launch and
to the result. A scalar cell takes the coating's $r_c$ as $r_s$; a field polarised along
p differs from it by the sign $(-1)^{N-1}$, which has no effect at normal incidence.

**Plates.** A plate is declared once, by the interval $[p,p+d]$ it occupies measured from
the first mirror. An even pass meets the plates in that order. An odd pass starts on the
second mirror and meets them in the reverse order, a plate entering at $L-p-d$, with the
gaps reversed and each interface crossed from the other side. The design (section 4.4)
and the executor use this one sequence. A cell whose plates are symmetric about the
centre has identical passes; any other has two kinds of pass, a period of two passes, and
a different beam on each mirror. A plate may touch a mirror ($p=0$ or $p+d=L$): the gap
between them has zero length and is not an element, and the beam crosses that plate at
the end of one pass and again at the start of the next, on either side of the
reflection, which is the substrate geometry of [2].

An off-axis Herriott path is, in the same paraxial and small-angle limit, the same
sequence of passes along the ray: the mirrors act on the beam about its local axis, with
tangential and sagittal focal lengths $R\cos\theta/2$ and $R/(2\cos\theta)$ at incidence
$\theta$. The difference is $O(\theta^2)$, a few $10^{-4}$ for the injection angles of
[10]; it is neglected here and belongs to the Cartesian follow-up. The passes are
independent: they are separated in time by $L/c$, nanoseconds, and nothing carries from
one to the next except the field.

#### 4.2 Linear design

A pass of the empty cell followed by its reflection, from just after one mirror to just
after the next, is
$M=\begin{pmatrix}1&L\\-2/R&1-2L/R\end{pmatrix}$ [11]. Its trace gives the Gouy phase per
pass, $\cos\psi=1-L/R$, real for $0<L<2R$. The eigenmode has its waist at the centre and
the Rayleigh range fixed by the geometry, $z_R^2=L(2R-L)/4$, whatever the frequency, so

$$
w_0^2(\omega)=M^2\frac{\lambda_0 z_R}{\pi n_0},\qquad
w_m^2(\omega)=w_0^2\left[1+\left(\frac{L}{2z_R}\right)^2\right],
$$

for a beam of quality $M^2$ (section 4.3), and the launch matched at every frequency is
the isodiffracting Gaussian, $M^2=1$, whose waist grows as $\sqrt{\lambda_0}$. Near the
concentric limit $w_0\propto(2R-L)^{1/4}$: a 0.5 mm change of $L$ on a 1 m cell moves
$w_0$ by 2.4 %, which is why a reproduction must use the exact geometry.

With plates, a pass is the product of the free-space, interface and mirror matrices of
its segments, with reduced lengths $d/n$ [11], and the period of the cell is the round
trip of two passes of section 4.1, which reduces to one pass when the plates are
symmetric about the centre. $0<L<2R$ certifies the empty cell only. The design requires
$|A+D|/2<1$ of the round-trip matrix with the plates in place, and refuses a cell that
fails it. The eigenmode is then the fixed point of the round-trip map of the complex beam
parameter that has a real beam size, found numerically and checked against the closed
form above for the gas cell. An asymmetric layout has one beam size on each mirror.

#### 4.3 Kerr-corrected design

**The aberrationless beam.** Bélanger and Paré [5] showed, and [4] uses, that a continuous
Gaussian beam of power $P$ in a homogeneous Kerr medium follows
$w^2(z)=w_0^2[1+\sigma(z/z_R)^2]$ and $R(z)=z[1+(z_R/z)^2/\sigma]$, with
$z_R=\pi n_0w_0^2/\lambda_0$ and $\sigma=1-P/P_c$. It propagates through any ABCD system
with the generalised reduced parameter [4, eq. (8)]

$$
\frac{1}{\hat q}=\frac{n_0}{R}+i\sqrt\sigma\,\frac{\lambda_0}{\pi w^2},
$$

that is, as a linear beam of wavelength $\lambda_0\sqrt\sigma$.

*Re-derivation.* For the paraxial equation
$2ik\partial_zA+\nabla_\perp^2A+2k^2(n_2/n_0)|A|^2A=0$, $k=n_0k_0$, the second moment
$V=\int r^2|A|^2\,dS$ obeys $V''=(2/k^2)H$ with the conserved
$H=\int|\nabla_\perp A|^2\,dS-k^2(n_2/n_0)\int|A|^4\,dS$ [8], so $V(z)$ is exactly parabolic
in a homogeneous segment, however the profile deforms. For a Gaussian at its waist,
$\int|\nabla A|^2=2P/w_0^2$ and $\int|A|^4=P^2/(\pi w_0^2)$, so
$H=(2P/w_0^2)(1-P/P_G)$ with $P_G=\lambda_0^2/(2\pi n_0n_2)$: the law above, exact for the
second moment of a Gaussian start, with $P_c=P_G$. A wavefront curvature adds its own term
to $H$ and leaves the nonlinear one unchanged, which is the split of $1/\hat q$ into a real
geometric part and an imaginary part scaled by $\sqrt\sigma$. After a lens the profile is
no longer Gaussian and the law becomes the aberrationless approximation.

**Beam quality.** A beam of quality $M^2$ has, by definition, $M^4$ times the diffraction
term of the Gaussian of the same second-moment radius:
$\int|\nabla_\perp A|^2\,dS=2M^4P/w_0^2$ at its waist. If its intensity profile stays
close to a Gaussian, so that $\int|A|^4\,dS$ keeps the Gaussian value, then
$H=(2P/w_0^2)(M^4-P/P_G)$ and the laws above hold with

$$
\sigma=1-\frac{P}{M^4P_c},\qquad z_R=\frac{\pi n_0w_0^2}{M^2\lambda_0},\qquad
\frac{1}{\hat q}=\frac{n_0}{R}+i\sqrt\sigma\,\frac{M^2\lambda_0}{\pi w^2},
$$

the beam propagating as a linear beam of wavelength $\lambda_0M^2\sqrt\sigma$. This is the
form the design uses: the beam quality multiplies the diffraction and leaves the Kerr
lens of the profile unchanged. It is a parameter of the design only. A value of $M^2$
does not determine a field, and the launch the design generates is the Gaussian, $M^2=1$
(section 5.1). [10] applies its beam quality differently: it scales the beam size by
$M^2$ and computes $\sigma$ without it, which multiplies the Kerr term by $M^4$ as well,
10 % at $M^2=1.05$ and 59 % at 1.26. The two forms differ by 6 % in the mirror fluence of
its Fig. 2 (section 7), and the design reproduces [10] with an effective power multiplied
by $M^4$.

**The critical power.** In the form $P=\alpha\lambda_0^2/(4\pi n_0n_2)$ of
`critical_power`, the derivation gives $\alpha=2$; the Townes baseline is
$\alpha=1.86226$; the computed collapse threshold of a collimated Gaussian is
$\alpha\approx1.89$, written $3.77\lambda_0^2/(8\pi n_0n_2)$ [6, 7], and is the value [4]
and [10] use. They place $\sigma=0$ at the power where a Gaussian actually collapses; the
derivation places it where $H$ vanishes, slightly above. The difference is 6 % in $P/P_c$:
for the cell of [4] section 3, $\sigma$ moves from 0.595 to 0.618, $w_0$ by 0.9 % and the
B-integral by 1.9 %, against the 30 % by which aberrations raise the simulated B-integral
above the aberrationless estimate in the same paper. The design's default is the
derivation's $\alpha=2$ (Q1, resolved): it is the constant for which the moment law is
the exact second moment of a Gaussian, in a fill and in a plate alike (section 4.4), and
[9] takes the same $\lambda_0^2/(2\pi n_0n_2)$. The published $3.77/8\pi$ is selectable,
to reproduce [4] and [10]; it rescales that exact law by 6 % and is then a calibration,
not the variance identity. The cell record names three powers apart: the Townes baseline,
the collapse threshold of a Gaussian and the variance scale. None of them is a guarantee
against collapse (section 4.4). `critical_power` fixes the Townes mass today; it gains
the mass as a keyword, defaulting to today's value, rather than the design keeping a
second formula. It refuses a zero susceptibility, so the design treats a segment without
a Kerr index, vacuum or a linear medium, as $x=0$ without calling it.

**The nonlinear index.** The design takes $n_2$ where the run takes its coefficients,
from optmat: `solprop.physdata.n2` for a gas, at the carrier and the medium's temperature
(293.15 K by default), and the glass's Kerr model for a plate. For argon at 1030 nm optmat
0.9.1 gives $n_2=1.000\times10^{-23}$ m²/W per bar (model `hyperpolarizability/Ar`, fitted
to Hättig et al.'s coupled-cluster values and to Shelton's measurements and compilation;
fit uncertainty 0.2 %), linear in pressure to 0.06 % up to 3.5 bar. The literature used here
takes $9.7\times10^{-24}$ m²/W per bar ([4], [10]) and $7.0\times10^{-24}$ ([9], Table 1):
optmat is 3.1 % and 43 % higher. [10] quotes its value as $(9.7\pm1.2)\times10^{-24}$, a
measurement uncertainty of 12 %, which the 0.2 % of optmat's fit does not replace:
optmat's value lies well inside it. The temperature alone moves $n_2$ by 1 % per 3 K
through the density, and none of [4], [9] and [10] states it.

*Glass.* For fused silica at 1030 nm optmat's long-pulse $n_2$ is $2.53\times10^{-20}$
m²/W, against $2.2\times10^{-20}$ in [14], $2.74\times10^{-20}$ in [13] and
$2.8\times10^{-20}$ in [9]; [12] and [15] state none. It is a total. optmat's silica bundle
([SEP 3](archive/sep-0003-optmat-materials.md) section 4.5) splits it into an electronic
share $(1-f_R)n_2$, which a `KerrResponse` carries, and a delayed share $f_Rn_2$, which a
`RamanResponse` carries, with $f_R=0.219$ for its copolarised shape and 0.245 for its
tensor one. The design's index of a segment is the long-pulse total for the polarisation
of the launch,

$$
n_2^{\rm design}=c_e(1-f_R)\,n_2+c_Rf_R\,n_2 ,
$$

with $c_e=c_R=1$ for a linear polarisation and the factors of section 4.4 for a circular
one. It is the index a pulse meets when it is long against the delayed response, some
tens of femtoseconds in silica, as the pulses of the cells here are (170 fs and more).
For a shorter pulse the delayed share acts only in part, and the design overestimates
the lens by at most $f_R$. An atomic gas has no delayed share.

*Overrides.* `n2` on the cell or on a plate replaces the design's total index of that
segment, for a linear polarisation, to reproduce a publication. It changes the design
only. The run's responses are the caller's, as on every free-space route; to hold the
same index the caller scales them to it, the Kerr coefficient and the delayed strength by
the same factor, which keeps $f_R$. Setting the Kerr coefficient alone from a total index
and keeping the Raman response counts the delayed share twice. The cell record lists,
for each segment, the design's index with its two shares and their polarisation factors
beside the run's response models and coefficients, so that a mismatch is visible.

*Sensitivity.* With $x=P_{\rm eff}/(M^4P_c)$, $d\ln B_{\rm axis}/d\ln n_2=1+x/(2\sigma)$,
1.2 to 1.3 for the cells here. For the cell of [4] and the four cells of [10] at the
pressures of its Table 2, in the design's own conventions, optmat's argon changes
$\sigma$ by −1.0 to −1.9 %, the matched mirror radius by −0.2 to −0.5 %, the mirror
fluence by +0.5 to +1.0 % and the B-integral by +3.6 to +4.1 % against the index of [4]
and [10]; the index of [9] would lower the B-integral by a third, and the 12 % that [10]
quotes is ±15 % on it. The B-integral sets the broadening: with the rule of [9] that the
compression factor is about $B/\pi$, 4 % more B-integral is about 4 % off the
Fourier-limited duration, the size of the agreement between simulation and measurement in
[10] (18.5 against 19.2 fs for MPC2). The effect on the propagated results is measured,
not assumed (section 7).

**Pulses.** Each time slice of a pulse whose shape is fixed over a pass (dispersion
negligible there) is its own continuous beam with power $P(t)$. The energy-weighted mean
square radius is
$\langle r^2\rangle=\int P(t)\langle r^2\rangle_t\,dt/E=(w_0^2/2)[1+\bar\sigma z^2/z_R^2]$
with $\bar\sigma=1-P_{\rm eff}/P_c$ and

$$
P_{\rm eff}=\frac{\int P^2\,dt}{\int P\,dt},
$$

which is eq. (6) of [4], obtained there from the energy-averaged nonlinear phase. A
Gaussian gives $P_{\rm eff}=P_{\rm peak}/\sqrt2$, a sech² $2P_{\rm peak}/3$. The design
evaluates the integrals on the launched pulse rather than assuming its shape. The mean
hides the slices: $P_{\rm eff}<P_c$ does not put the peak below $P_c$, a Gaussian at
$P_{\rm eff}=0.8P_c$ peaking at $1.13P_c$, and a slice above its critical power has no
stationary beam (section 4.4).

**Stationary beam.** Setting the wavefront radius at the mirror, $z=L/2$, equal to $R$ gives
$z_R^2/\sigma=L(2R-L)/4$, hence the form of [4, eq. (3)], here with the beam quality,

$$
w_0^2=M^2\frac{\lambda_0L}{2\pi n_0}\sqrt{\sigma\left(\frac{2R}{L}-1\right)}
=\sqrt\sigma\,w_{0,\rm lin}^2 ,
$$

and the same factor $\sigma^{1/4}$ on the mirror radius; $\psi$ is unchanged, and
$w_{0,\rm lin}$ is the linear waist of section 4.2 with its $M^2$.

**From pass to pass.** The cell keeps a stationary beam only for one $\sigma$, and the
energy falls at each reflection. The design follows it with a fixed-pulse estimate: the
temporal shape of the launch is kept, and the effective power entering pass $j$ is
$P_{{\rm eff},j}=T_jP_{\rm eff}$, with $T_j$ the product of the energy transmissions of
the elements crossed before it, each coating's reflectance weighted by the spectrum of
the launch, so that two different coatings, or one that varies across the band, enter as
they are. Two results are reported under two names. The *matched radius* of pass $j$ is
the stationary radius for $\sigma_j$: the beam a launch matched at that pass would need.
The *drift* is the radius of the one launch that was matched for pass 0, carried through
the map of each pass with that pass's own $\sigma_j$ (the moment map of section 4.4), as
[10] follows its fluence over the round trips. The drift is what is compared with the
propagation; substituting $\sigma_j$ into the stationary formula does not give it. The
estimate states what it leaves out: the pulse changes in a cell. A chirped coating or a
plate reshapes it at constant energy, and a phase alone changes $P_{\rm eff}$ ($10^5$
fs² on a 300 fs Gaussian lowers it to 0.31 of its value); self-phase modulation with
dispersion reshapes it over tens of passes, as in the pre-chirped cell of [12]. The
design propagates no temporal surrogate. Where the shape changes over the cell, its
per-pass numbers are those of the launch's shape, and the propagated run is the result.

**B-integral.** With $I_0(z)=2P_{\rm peak}/(\pi w^2(z))$,

$$
B_{\rm axis}=k_0n_2\int_{-L/2}^{L/2}I_0\,dz
=\frac{8\pi n_0n_2}{\lambda_0^2}\frac{P_{\rm peak}}{M^2\sqrt\sigma}
\arctan\sqrt{\frac{L}{2R-L}}
$$

per pass on the axis, with $n_2$ the design's index of the fill. For $M^2=1$ this is the
expression [4, eq. (4)] calls the beam-averaged B-integral per round trip, the beam
average of a Gaussian being half the on-axis value. $\sigma$ uses $P_{\rm eff}$ and the
prefactor $P_{\rm peak}$, as [4] specifies. The design reports both quantities under
these two names.

**The conventions of [10].** What [10] publishes, what its co-author adds and what this
SEP infers by recomputation are kept apart.

| | Published in [10] | Stated by the author of this SEP, a co-author of [10] (2026-11-13) | Inferred here |
|---|---|---|---|
| Power in $\sigma$ | "$\sigma=1-P_0/P_{\rm crit}$ where $P_0$ is the beam's peak power", with $P_{\rm crit}=3.77\lambda^2/(8\pi n_0n_2p)$ | The numbers were computed with $P=E/(\sqrt2\,\tau_{\rm FWHM})$ | That power gives the 170 mJ/cm² of Fig. 2; the peak power would give 191 mJ/cm² for MPC3 against the 160 of Table 2 |
| Beam quality | $M^2=1.05$ in Fig. 2, $M^2_{x,y}$ of each laser in Table 1, a B-integral that "includes the beam quality factor" | — | The beam size carries $M^2$ and $\sigma$ does not: the 111 and 170 mJ/cm² of Fig. 2 follow, where $M^2$ in $\sigma$ would give 160 |
| Fig. 2 | $R=500$ mm, $L=978$ mm, 2 bar, 2 mJ, 300 fs | — | — |
| MPC3 in Table 2 | 1200 mbar, 160 mJ/cm² and $2.0\times10^{13}$ W/cm²; the text calls the column the optimal pressure, and its section 2.4 reports MPC3 compressed to 20 fs at 1.2 bar | 1.2 bar must be the pressure of the 20 fs measurement; the fluence was in all likelihood computed at another pressure | 2 bar, the pressure of Fig. 2, reproduces both columns: 156 mJ/cm² and $1.96\times10^{13}$ W/cm², against 131 and $1.65\times10^{13}$ at 1.2 bar |
| Pressures of the other cells | MPC1: 3500 mbar in Table 2, 3.0 bar in section 2.4. MPC2: 1300 mbar in Table 2, 19.2 fs at 1.1 bar in section 2.4, 19.4 fs at 1.3 bar in section 3.2. MPC4: 1100 mbar in Table 2, 1.15 bar in section 2.4 | — | — |
| Argon | $n_2=(9.7\pm1.2)\times10^{-24}$ m²/W per bar | — | — |

$E/(\sqrt2\,\tau_{\rm FWHM})$ is the effective power of a Gaussian whose peak is taken as
$E/\tau_{\rm FWHM}$; it exceeds the exact $P_{\rm eff}$ by 6.4 % for a Gaussian and by
20 % for a sech². The design uses the exact $P_{\rm eff}$ on the launched pulse, the
variance constant and the beam quality in $\sigma$. It reproduces [10] with that
publication's conventions declared: `critical="published"`, the index of [10], and
`power` $=M^4E/(\sqrt2\,\tau)$, where the factor $M^4$ cancels the one the design puts in
$\sigma$ (section 5.1).

#### 4.4 Cells with plates: what survives of the Kerr correction

[4] derives its correction for a Kerr medium that fills the cell and leaves plates to
future work. With plates, one ingredient survives and four fail.

**What survives: the moment law, segment by segment.** The variance identity holds in
every homogeneous segment, gas or glass, thin or thick, and every linear element (free
space, thin lens, plane interface) maps the second moments exactly, whatever the profile
[8]. With $W=w^2$ the square of the second-moment radius and $U=ww'$, a segment of index
$n_s$ and critical-power ratio $x_s=P_{\rm eff}/P_{c,s}$ gives

$$
W''=2\left[(M^4-x_s)\frac{\lambda_0^2}{\pi^2n_s^2W}+\frac{U^2}{W}\right],
$$

constant along the segment, so $W$ is a parabola in $z$ for either sign of $M^4-x_s$;
equivalently $w''=c_s/w^3$ with $c_s=(M^4-x_s)\lambda_0^2/(\pi^2n_s^2)$. A thin lens of
focal length $f$ maps $U\to U-W/f$; a plane interface from $n_a$ to $n_b$ keeps $W$ and
scales $U$ by $n_a/n_b$. For a fill this is the $\hat q$ form of section 4.3. The closure
is that the beam has, at the entrance of each segment, a Gaussian intensity profile and
the beam quality of the design: the first fixes the Kerr term of the bracket from $W$,
the second its diffraction term. A segment without a Kerr index has $x_s=0$.

**What fails.**

- *The closed form.* The stationary beam of section 4.3 assumes one $\sigma$ over the whole
  pass. With plates the eigenmode is a fixed point of the round-trip map of $(W,U)$, the
  two passes of section 4.1, checked against the closed form when the plates are
  removed. The map is nonlinear, so a root returned by a solver is not yet the beam of
  the cell, and three things are specified.
  - *The branch.* The fixed point is the one reached from the linear eigenmode of section
    4.2 by raising the power continuously from zero to $P_{\rm eff}$.
  - *Its stability.* Each segment is the flow of $w''=c_s/w^3$, each lens a shear of $w'$
    and each interface a rescaling of $w'$ that the return interface undoes, so the
    round-trip map preserves area in $(w,n\,w')$. Its Jacobian $J$ at the fixed point has
    unit determinant, and the stationary beam is stable when $|\operatorname{tr}J|/2\le1$.
    This is the counterpart of $|A+D|/2<1$: at zero power
    $\operatorname{tr}J/2=\cos2\Psi$, with $\cos\Psi=(A+D)/2$ of the round trip, which is
    at most one in magnitude exactly when the linear cell is stable.
  - *The refusals.* A cell whose continuation ends before $P_{\rm eff}$, by a fold, by a
    failure to converge or by $|\operatorname{tr}J|/2>1$, has no stable stationary beam
    in this model. The design says so, with the power at which the branch was lost,
    rather than return another root.
- *The complex parameter $\hat q$.* A plate runs far above its own critical power. Fused
  silica has $P_c\approx4$–5 MW at 1030 nm with the $n_2$ of [13] and [14] and the constant
  of [6], and the cells of [12, 13, 14] run at 0.3 to 60 GW, $x_p$ from about $10^2$ to
  $10^4$. Then $\sigma_p\ll0$ and $\sqrt{\sigma_p}$ has no real meaning. The plate does not
  collapse because it is much thinner than its self-focusing distance, of order
  $z_R/\sqrt{x_p}$. The moment law, in real arithmetic, holds for either sign while $W$
  stays positive. The design requires $W>0$ throughout every segment, at its ends and at
  the interior minimum of its parabola, and refuses a cell in which $W$ reaches zero.
  $W>0$ is positivity of the model, not a proof that the field does not collapse: the
  centre of a beam collapses before its variance vanishes, which is why the computed
  threshold of a Gaussian lies below the power where $H$ vanishes (section 4.3), and a
  slice can exceed its critical power while $P_{\rm eff}$ does not. The design is a
  sizing estimate. Whether a field survives is the result of the propagated run, and no
  run is refused admission on the design's criteria.
- *The strength of the Kerr lens of a thin plate.* For a plate of thickness
  $d\ll z_R/\sqrt{x_p}$ in an ambient medium of index $n_a$, the moment law with the exit
  rule $U_a=(n_p/n_a)U_p$ reduces to a thin lens in the ambient medium of

  $$
  \frac{1}{f_K}=x_p\frac{\lambda_0^2d}{\pi^2n_an_pw^4}
  =\frac{4}{\alpha}\,\frac{n_2P_{\rm eff}d}{\pi n_aw^4},
  \qquad P_{c,p}=\frac{\alpha\lambda_0^2}{4\pi n_pn_2},
  $$

  with $P_{\rm eff}$ the power inside the plate, whatever the beam quality, which enters
  the diffraction term only. For $\alpha=2$ this is exactly the change of the second
  moment that the phase $k_0n_2I_0d\,e^{-2r^2/w^2}$ imprints on a Gaussian intensity
  profile; a direct radial integration of that phase kick gives the same
  lens and is the independent check kept for it (section 7). With the published constant
  the lens is 6 % stronger than that exact kick. The parabolic expansion of the same
  phase about the axis gives $1/f=8n_2Pd/(\pi n_aw^4)$, $2\alpha$ times stronger, four
  times for the default: the centre of the beam is focused more than its wings, and the
  second moment follows the average. For the admitted ambient media $n_a-1$ is below
  $10^{-3}$; the factor is kept, not approximated. The two lenses give different
  eigenmodes. [13] and [15] take their matching from the same earlier work, not read
  here, and [14] from an earlier paper of its authors; none prints its lens. The numbers
  [13] and [15] publish are those of the moment lens, not of the on-axis one (below, and
  Q7).
- *One lens for the whole pulse.* Each time slice sees its own lens, from zero in the
  wings to its full value at the peak, and $P_{\rm eff}$ weights the slices by energy. In
  the gas cells of [4] and [10] $\sigma$ goes from 1 in the wings to 0.4–0.6 at the peak;
  in a plate cell the wings see no lens and the peak slice the full lens of every plate,
  and near the edge of stability, where the mirror radius varies as
  $(2R-L)^{-1/4}$, the spread of the eigenmode across the pulse grows, which is the regime
  of [14]: with its geometry (mirrors of 1 m, a round-trip $|A+D|/2$ of 0.992, 0.8 mm
  plates before the mirrors, 30 mJ in 460 fs, $M^2=1.17$) the moment map gives a beam
  diameter on the mirrors of 6.6 mm for $P\to0$, 9.5 mm for $P_{\rm eff}$ and 10.2 mm for
  the peak slice, where [14] reports about 7.2 mm. The design reports the matched mirror
  radius for $P\to0$ and for the slice at the peak. Where the peak slice has no stable
  stationary beam, in a fill when $P_{\rm peak}\ge M^4P_c$ and in a cell with plates when
  its branch is lost, the design reports that, with the power at which it happens, in
  place of a radius.

**Where the plates sit.** A fill scales the radii at the mirrors and at the centre by the
same factor $\sigma^{1/4}$; the lens of a plate moves them in directions set by its
position, and no single $\sigma$ for the cell can follow that. At the centre of a
symmetric cell the eigenmode enters the plate with $U=W/(2f_K)$, so the beam still widens
into the plate and the centre is wider than the linear waist, with a waist on either side;
beside a mirror the lens adds to the mirror's power and brings the cell toward the edge of
stability, so the mirror radius grows. [15] reports both signs (section 3), and its
magnitudes follow from the moment lens. For its geometry ($R=0.2$ m, $L=0.38$ m, 1 mm
plates, $M^2=1$, optmat's index) the moment map gives −32 % on the mirror radius for its
first cell (128.5 µJ in 1 ps, $x_p=19$, five plates taken within 1 to 3 cm of the
centre, their positions not being published), where [15] reports about −30 %; and +17 %
for its second (two plates 11 cm from the centre, with the 33.4 µJ in 46 fs that [15]
itself simulates to account for its beam quality and its Brewster plates, $x_p=105$),
where [15] reports about +15 %. The on-axis lens gives −51 % and +47 %.

Whether the stationary beam is stable depends on the layout. In these symmetric cells,
to $x_p=10^4$, and in the geometry of [14], to $x_p=4.6\times10^4$, it persists and stays
stable as the power rises, because the lens weakens as $w^{-4}$ when the beam grows. An
asymmetric layout can lose it. With two unequal plates ($R=0.3$ m, $L=0.55$ m, 1 mm at
50 mm and 2 mm at 400 mm from the first mirror) a stationary beam
exists at every power tried, to $x_p=250$, but $\operatorname{tr}J/2$ falls from −0.71
at zero power through −1 at $x_p\approx102$; beyond, a perturbation changes sign and
grows at every round trip, threefold at $x_p=200$. The stability test of the design is
therefore not a formality, and these few cells are observations, not properties the
design assumes.

**Air and polarisation.** In a hybrid cell the ambient gas is its own segment, with
$\sigma_a=1-P_{\rm eff}/(M^4P_{c,a})$; [13] leaves the air out of its matching and bounds
its B-integral per pass instead. For a circular polarisation the index of an
isotropic medium is lower than for a linear one, by a factor that depends on the
response. The instantaneous electronic response gives $c_e=2/3$ [13]. A delayed response
with Hellwarth's isotropic part $a$ and anisotropic part $b$
($R_{ijkl}=a\,\delta_{ij}\delta_{kl}+\tfrac12b(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$,
[SEP 3](archive/sep-0003-optmat-materials.md) sections 4.4 and 4.5), of integrals
$\bar a+\bar b=1$, gives in the long-pulse limit

$$
c_R=\bar a+\tfrac12\bar b :
$$

the isotropic part responds to the total intensity whatever the polarisation, and the
anisotropic part to the slow part of $E_iE_j$, of which a circular field keeps half on
each axis and none between them. For the tensor shape of silica, $\bar a=0.75$,
$\bar b=0.25$ and $f_R=0.245$, so the total factor is
$0.755\times\tfrac23+0.245\times0.875=0.718$, 8 % above two thirds. The design applies
$c_e$ and $c_R$ to the two shares of section 4.3 when the launch declares a circular
polarisation, and refuses there a delayed response that has no tensor decomposition, as
a vector run refuses it. The run's responses need no factor: the vector Kerr and Raman
responses act on the two components of the field.

**Validity.** The design is aberrationless in the sense above, monochromatic in its beam
parameters, and neglects the change of the pulse over a pass and over the cell (section
4.3). In a fill the Kerr phase is spread over the pass, and [4] finds the simulated
B-integral 30 % above the aberrationless estimate. In a plate cell the Kerr phase of a
pass is concentrated in millimetres, about 0.3 rad per plate in [13] and [14], and the
beam leaves each plate with an aberration of that order, which the closure ignores until
the cavity reshapes the beam.

Two published designs can be compared with it. [13] (0.75 mJ, 598 fs sech², circular
polarisation, two 0.5 mm plates 280 mm apart in air, $R=300$ mm, $L\approx550$ mm,
$M^2=1.24\times1.28$) computes, with the Kerr lenses of its plates, a waist at the centre
of its cell of 344 µm in diameter, against 372 µm without them, a ratio of 0.925, and a
focus of the beam to inject that is slightly larger and about 4 mm from the centre. Its
conventions can be read from it: the air is left out of the matching; the beam quality
enters the Rayleigh range of a beam of the measured size, as in section 4.3; the index
of silica is $2.74\times10^{-20}$ m²/W, times two thirds; and its B-integral in air,
0.5 rad per pass, is close to that of a peak power of $E/\tau$. In those conventions the
moment lens gives a centre waist of 335 to 339 µm, a ratio of 0.90 to 0.91, for a peak
power of $E/\tau$ or of the sech², and an injected focus of 352 to 354 µm, 4.4 to 5.0 mm
from the centre. The on-axis lens gives 273 to 280 µm and a focus 13 to 14 mm away.
[15] agrees, with both of its cells (above).

These are comparisons with two design calculations of one lineage, not with a
measurement or a propagation. They show which lens those designs amount to; whether the
closure holds is still judged on propagated cells (Q7). The first draft quoted 0.926 for
[13]: that was the injected waist, with the air in the matching, the effective power and
$M^2$ out of $\sigma$, and its agreement was fortuitous.

The design sizes the grid and the launch; the propagated run is the result. The
comparison of the two, pass by pass, is itself an output (section 7) and, for plates, a
deliverable of phase 3.

#### 4.5 References, frames, noise and energy

**References.** Each pass runs in its own moving frame; the `FieldPlane` reference
carries the delay and carrier phase from pass to pass, so the stored spectrum stays in
its window. Whether the absolute field is known depends on the coatings. A coating whose
model keeps its constant phase and its group delay, a `TaylorPhase` or a callable whose
calibration says so, leaves the reference calibrated, and `plane.absolute()` is then
exact over the whole cell. `ChirpedMirror` data lack both terms: the reference of every
plane after the first such reflection is uncalibrated in phase and delay,
`absolute()` refuses it as it does today, and nothing in the cell needs it, since a
relaunch takes the stored samples and the reference as they are. The cell record keeps
the calibration status of each pass and a resume restores it. A caller who knows the
missing terms gives them as the coating's calibration (section 5.1).

**Noise.** The cell has one noise identity. `MultipassSimulation` takes the `noise` and
`seed` of a free-space constructor. The input model, the default wherever the route
carries noise, is drawn once, by the executor, into the launch of pass 0. `None` adds
nothing, which is also how a caller passes a plane that already carries a realisation: a
plane holds samples and cannot say whether a noise floor is in them, so the executor
never guesses. The modified model is refused (section 5.4). A draw without a seed takes
system entropy, and its seed is written to the cell record before pass 0 starts, so that
a resume which finds no committed pass redraws the same realisation. A relaunch carries
the realised field and draws none, and every runner is built without a noise model,
since a runner refuses an input whose noise model differs.

**Energy.** Energies are the axial flux of the medium (SEP 34), so a plate's Fresnel
loss and a coating's reflectance appear in joules in the ledger.

#### 4.6 Diagnostics

**Radii.** The beam radius at the mirrors and at the centre is the second-moment radius
of the fluence, $w=\sqrt{2\langle r^2\rangle}$, the first of the two radii of the
`beam_radius` statistic and the quantity of the moment law. It equals the $1/e^2$ radius
for a Gaussian and exceeds it for an aberrated beam with wings; the $1/e^2$ radius is
given beside it, never under the same name. The centre is the fixed plane $z=L/2$ of each
pass. It is not the focus: a cell with a central plate has a waist on either side of it
(section 4.4), and an asymmetric layout need not focus there. No focus observable is
offered; the largest on-axis intensity along a nonlinear segment is in its statistics.

**The intensity integral.** The diagnostic of the B-integral of pass $j$ is

$$
\mathcal B_j=k_0\sum_sn_{2,s}\int_s\max_tI(0,t,z)\,dz ,
$$

the integral along the pass of the largest on-axis intensity in time. It is read from
the `on_axis_intensity` statistic, not from `peak_intensity`, which is the maximum over
the transverse nodes as well and leaves the axis for a ring or an aberrated beam. It is
the nonlinear phase of one temporal slice only while the same slice stays the most
intense, and it is a diagnostic, not a quantity the run integrates. $n_{2,s}$ is the
design's index of segment $s$ (section 4.3), so the diagnostic is defined for a scalar
field, or a declared circular one, in the long-pulse limit, and is not offered
otherwise. A gap propagated as a linear element has no response and contributes nothing.
The integral is taken segment by segment: the samples of a medium are joined within it,
and no interval crosses an interface, where the index jumps and the intensity with it.
Its sampling is set, not inherited from the step history. The statistic is recorded at
every accepted step of a nonlinear segment, through the periodic statistics; the
trapezoid sum $T_h$ over all the samples is compared with $T_{2h}$ over every second
one, and $|T_h-T_{2h}|/3$ is reported as the quadrature error. A pass whose error exceeds
a stated fraction of $\mathcal B_j$ is marked unresolved rather than reported as a
number, and is resolved by a smaller `dz_max`. The rule is verified against the analytic
integral of a Gaussian beam (section 4.3) at three samplings before it is used to compare
matching models.

**Durations.** Three quantities are kept apart.

- The *spectral transform limit* is that of the beam-integrated spectral energy
  $S(\omega)$, computed from $\sqrt S$ with `transform_limit`. It is a property of the
  spectrum and a lower bound, not a pulse that a compressor produces when the spectrum
  or its phase varies across the beam.
- The *compressed pulse* is what a common compressor gives: one spectral transfer
  $H(\omega)$ applied to the field at every transverse node, then the axial power
  through the plane in time, $P(t)=\int S_z(r,t)\,dS$, with the frequency-dependent
  admittance of the medium (`spatial_power`, SEP 34). Its duration is the full width at
  half maximum of $P(t)$, and its quality the peak of $P(t)$ against that of the
  transform limit at the same energy.
- The *on-axis pulse* is the same transfer applied to the on-axis field alone, the
  nearest counterpart of a measurement that samples the centre of the beam.

For a beam separable in space and frequency the three durations agree. They do not in
general: two transverse channels of equal energy carrying disjoint spectra have a
spectral transform limit whose peak is twice the peak of their summed compressed power.
Each comparison with a publication names the quantity it uses (section 7). A
spatio-spectral homogeneity measure is chosen in phase 4 (Q4).

### 5. Proposed interface

#### 5.1 Description and design

```text
from solprop.multipass import CellMirror, MultipassCell, Plate
from solprop.optics import TaylorPhase
from solprop.spatial import SpatialMedium

mirror = CellMirror(radius=0.3, reflectance=0.9995,
                    coating=TaylorPhase((0.0, 0.0, 25e-30), 1030e-9), model_error=1e-3)
cell = MultipassCell(mirror, length=0.595, passes=27, medium=SpatialMedium("Ar", 1.0))
design = cell.design(pulse)        # CellDesign: psi, rayleigh_range, waist(omega),
                                   # radius on each mirror (matched, and drift per pass),
                                   # sigma per pass and segment, power_ratio, stability,
                                   # plate Kerr focal lengths, b_axis and b_beam per
                                   # pass, mirror fluence
launch = design.matched_plane(grid, space)   # FieldPlane: entrance plane of pass 0
```

`MultipassCell(mirror, *, length, passes, medium, plates=(), n2=None)`: `mirror` one
`CellMirror`, or a pair `(first, second)` of equal radius whose coatings differ; `length` in
m with $0<L<2R$; `passes` a positive integer, the passes simulated, not round trips, $N$
passes applying $N-1$ reflections (section 4.1); `medium` the fill or the ambient medium;
`plates` a sequence of `Plate(medium, *, thickness, position, responses=(),
interface="fresnel" | "matched", n2=None)`. `position` is the distance $p$ of the plate's
first face from the first mirror: the plate occupies $[p,p+d]$, is crossed from that face
on even passes and from the other on odd ones (section 4.1), and may touch a mirror.
`responses` are the plate's own nonlinear responses, so that two glasses in one cell each
keep their Kerr and Raman models; those of the fill or of the ambient medium are given to
the executor (section 5.2). `n2` (m²/W) overrides the design's total index of the fill or
ambient medium, and a plate's its own, for a linear polarisation; None, the default,
takes it from optmat with its electronic and delayed shares (section 4.3). An override
reproduces a publication and changes the design only: the run's responses stay the
caller's, scaled by the caller as section 4.3 says, and the cell record keeps the two
side by side for each segment.

`CellMirror(radius, *, reflectance, coating=None, calibration=None, model_error)` expands
to the elements of section 4.1; `calibration` states the constant phase and the delay of
a coating whose data lack them (section 4.5).

`cell.design(pulse, *, power=None, beam_quality=1.0, critical="variance" | "published",
plate_lens="moment" | "on_axis")` returns a frozen `CellDesign`, with the two lenses of
section 4.4 for plates. `beam_quality` is the $M^2$ of section 4.3, a parameter of the
design and of nothing else. `power` (W), None by default, replaces the $P_{\rm eff}$
computed on the pulse by a publication's effective power; [10] is
`power` $=M^4E/(\sqrt2\,\tau)$ with `critical="published"`. The design reports, for each
pass and each mirror, the matched radius and the drift (section 4.3), the stability of
the cell and of its stationary beam, and for the peak slice a radius or the statement
that it has none (section 4.4).

`design.matched_plane(grid, space, *, medium=None)` returns the launch: an isodiffracting
Gaussian at the entrance plane of pass 0, sampled and normalised in the cell's medium, or
in `medium` when the beam of this design is to be launched into another medium, as a
mismatch study does (section 5.5). It refuses a design whose `beam_quality` is not one:
no Gaussian has that quality, and enlarging a Gaussian's waist changes its Rayleigh
range, not its quality.

The design refuses, each by name, $L\ge2R$, a pair of mirrors of different radii, a cell
whose round trip with its plates in place is unstable, a non-positive $\sigma$ in a fill
or an ambient gap, a segment in which the moment law reaches $W=0$, a cell whose
stationary branch is lost before $P_{\rm eff}$, a circular launch on a delayed response
without a tensor decomposition, and a pulse with no finite energy. A segment without a
Kerr index is linear and is not an error.

#### 5.2 Running a cell

```text
from solprop.driver import MultipassSimulation

simulation = MultipassSimulation(cell, grid=grid, space=RadialGrid(4e-3, 256),
                                 responses=[KerrResponse(...)], settings=settings,
                                 noise="input", seed=7)
result = simulation.run(launch, output="cell-run/")
result.ledger                     # per pass and per element, in joules
result.output_plane               # FieldPlane: exit of the last pass, before a reflection
result.plane(26, "exit")          # any pass: "entrance", "centre" or "exit"
result.pass_outputs(26)           # the ordinary run outputs of the pass's segments
result.series("mirror_radius")    # per-pass observables against z_abs
```

`MultipassSimulation(cell, *, grid, space, responses=(), settings, noise="default",
seed=None)` is imported from the driver, where it lives (section 6.2); the physics module
`solprop.multipass` holds the description and the design only. `responses` are those of
the cell's medium, the fill or the ambient gas; a plate brings its own (section 5.1).
`noise` and `seed` are those of section 4.5. The executor builds one `Simulation` and one
`SimulationRunner` per distinct nonlinear segment: the whole pass in a gas cell, with or
without responses; each plate; and the ambient gaps when `responses` is not empty, equal
segments sharing a runner. Ambient gaps without responses are `FreeSpace` elements. Every
segment uses the same temporal grid and the same transverse grid.

A pass is the ordered sequence of its segments and elements, reversed on odd passes
(section 4.1). A handoff between two segments goes through the interface element and the
receiving port, as the mirror does between passes. Three planes of every pass are defined
whatever the layout: its entrance, its exit (section 4.1) and its centre, $z=L/2$.

- The *centre* is a save plane of the run that contains it, at that run's local
  position. When it falls in a linear gap, that `FreeSpace` is split there and the
  intermediate plane is measured. When it falls exactly on a face of a plate, it is taken
  on the ambient side.
- The *exit* is the last plane of the pass's last run, carried through the linear
  elements that follow it in the pass, if any. The last run of every pass always retains
  the field of its last plane, whatever field retention the caller asks for: it is the
  checkpoint of the pass (section 5.3).

`result.plane(j, where)` returns these planes as `FieldPlane`s with their medium,
geometry, reference and calibration status: the exit always, recomputed from the
checkpoint when a linear element follows it, and the others when their fields were
retained. `result.output_plane` is the exit plane of the last pass, before any further
reflection, with $N-1$ coating factors; it is not, in general, the last plane of the last
run. `result.pass_outputs(j)` gives the ordinary outputs of the runs of pass $j$, in the
order crossed. `with_launch(plane)` returns a copy bound to a launch, which `run` uses
when it is given none; scans need it (section 5.5).

`run(launch=None, *, output, stop=None, progress=...)` polls `stop` between segments and
hands it to the segment in progress, which stops within a step (section 6.2); the cell
then ends at its last committed pass. Every runner is closed when `run` returns, raises
or is stopped, and when the construction of the executor fails part-way.

#### 5.3 What a run produces

- **A directory.** One ordinary output per segment run, readable by every existing
  reader, and a versioned cell record, the manifest, beside them. The record holds the
  cell, the design, the identity of the run and, for each committed pass, the
  configuration hash of each segment run, its $z_{\rm abs}$ offset, the transport reports
  of the elements crossed, the measurements of its centre plane, the runs'
  boundary-monitor totals, the thin-screen certificate of the mirror that preceded it and
  the calibration status of its exit plane.
- **The identity** of a cell run is a hash over the digest of the launch, the noise model
  and seed, the ordered elements with their coatings and calibrations, the media and the
  responses of every segment, the grids and the settings. A rerun into a directory whose
  identity differs is a conflict and is refused; nothing is overwritten.
- **The ledger** closes for each pass and over the cell. Losses in joules (coatings,
  interfaces, absorption, the boundary monitor's removal, removed content) are kept apart
  from bounds (the defects of the screens, the error terms of the transports, the
  accuracy of each run). The bounds are summed in absolute value from the launch, so a
  bound grows along the chain and no relaunch resets it, and the removal of the boundary
  monitor is accumulated over the cell. This is the ledger of a chain that #238 found
  missing.
- **Resume is a transaction for each pass.** A pass is committed when its segment
  outputs are closed, the last of them holding the checkpoint (section 5.2), and its
  entry, with the reports of every element of the pass, is then published in the record
  by an atomic replacement of the record file. The reflection that follows a pass
  belongs to the preparation of the next one: it is recomputed from the checkpoint, and
  its report is stored with that next pass, so the result of no element is persisted
  without its pass. A rerun with the same identity takes the advisory lock of the
  directory, keeps the committed passes, removes the files of an uncommitted pass and
  those only, and restarts that pass from its entrance. Every element is thus applied
  exactly once in the result, wherever the interruption fell. A pass whose run ends
  before its last plane, by a boundary stop, a failure or a stop request, is not
  committed. The checkpoints live in the cell's own directory, so the `fields="none"` of
  a scan, which governs the collected file, does not remove them.

#### 5.4 Refused

The modified noise model (`field_plane` refuses it), TDSE responses (a runner refuses
them), planar and line grids, a medium varying along a pass, different grids between
segments, a plate outside $[0,L]$ or overlapping another, oblique plates, mirrors of
different radii and astigmatic mirrors, a launch asked of a design whose beam quality is
not one, a circular launch on a delayed response without a tensor decomposition, a scan
of cells on the `Batch` executor, and a rerun whose identity differs from the
directory's, each by name. The design's own refusals (an unstable cell, a lost
stationary branch, section 5.1) refuse the design, not a run: a launch the caller
supplies runs in any cell the executor can build.

#### 5.5 Scans of cells

```text
from solprop.driver import MultipassSimulation
from solprop.driver.scan import Scan, run_scan

def build(point):
    cell = MultipassCell(mirror, length=0.495, passes=27,
                         medium=SpatialMedium("Ar", float(point.values["pressure"])))
    launch = cell.design(pulse).matched_plane(grid, space)   # matched at every point
    return MultipassSimulation(cell, grid=grid, space=space, responses=responses,
                               settings=settings).with_launch(launch)

scan = Scan("mpc1-pressure", pressure=np.linspace(0.5, 3.5, 7))
result = run_scan(scan, build, output="mpc1-pressure.h5",
                  reductions=[cell_spectral_width(), cell_transform_limit()])
```

`run_scan` accepts a builder that returns a `MultipassSimulation` bound to its launch
(`with_launch`, as `Simulation.with_input`), all points of a scan being of one kind.

**The launch belongs to the point.** A plane carries the metric of its medium, and the
port refuses a relaunch into a medium of another index (`check_same_medium`, to
$10^{-10}$, where argon moves the index by about $3\times10^{-4}$ per bar). The builder
therefore makes the launch in the medium of each point. A matched study recomputes the
design at the point, as above. A mismatch study keeps the beam of a nominal design,
`nominal.matched_plane(grid, space, medium=cell.medium)`: the radius and the wavefront
curvature at the entrance plane and the energy are held, and the plane is sampled and
normalised in the point's medium. The field amplitude and the peak axial power then
follow from the medium and are not held; a study that holds another quantity says which.
An old plane is never relabelled, and no interface is inserted to pass the guard.

**Executors.** `Local` runs the points in turn, `Pool` in worker processes, and `Queue`
and `Slurm` drain them through pools; a point's passes run in one worker. `Batch` is
refused for cells, by name. It advances compatible points in lockstep, each a single
kernel problem under one group session, and a cell is a chain of runs with host elements
between them: admitting it needs a backend change, which this SEP excludes (section 10).
On every admitted executor the stop flag reaches the segment in progress through the
runner (section 6.2), and a stopped point keeps its committed passes; the point's
reporter receives the calls of each segment run with its pass index; a failed segment
fails the point and leaves it resumable; a `Queue` lease covers the whole point and is
renewed between segments; a completed point is skipped on a rerun, and an interrupted one
resumes from its last committed pass (section 5.3).

**Memory.** A point holds all its runners at once, so its admission is the sum of their
backends' `memory_ledger()` totals, with the translated tables they do not share, the
planes retained in memory, one transport workspace and the host copy of the field in a
handoff. It is not that of its largest segment. Building and closing one backend at a
time would bound it by the largest, but would pay allocation and planning at every
segment of every pass, which reuse exists to avoid; the lifecycle of section 5.2 is the
one adopted.

**What is collected.** Each point runs its cell into its own directory beside the
collected file, so every pass output stays readable. The collected file holds, for each
point, the identity of the cell record, the statistics of the runs of the last pass, the
cell's exit plane when `fields` asks for fields, and the declared reductions, which for a
cell receive the cell's result rather than one run and declare the shape of what they
return. A per-pass reduction has the shape of the passes, so it refuses a scan of
`passes`, which is not needed: one cell of $N$ passes gives every shorter one through its
pass outputs, the extraction optics being out of scope.

**Seeds and reuse.** The point's seed is the cell's single noise draw (section 4.5):
noise the builder leaves unseeded is drawn from `point.seed`, or from the scan's name and
the point's index when the scan declares no seed, and noise the builder seeds must use
`point.seed`, as for simulations. Consecutive points that change only the launch (energy,
chirp, duration) reuse the cell's runners, as consecutive simulation points reuse their
backend; each relaunch reruns the transverse checks of `with_input` and keeps the
material and rate admission of the runner's problem. A point that changes the medium,
the geometry or the plates rebuilds the runners. Whether the design is recomputed at each
point (a matched cell) or kept from a nominal point (a mismatch study) is the builder's
choice.

### 6. Implementation strategies

#### 6.1 Architecture

**A — composition on the host (recommended).** Each pass is one or a few ordinary runs; the
mirror, the coating, the interfaces and the linear gaps are applied with `transport`
between them; the next pass is launched with `couple(..., same_basis)` into the same
runner's port, after declaring the transported plane at the next entrance
(`geometry=port.geometry`). This uses only qualified elements with their certificates and
reports, needs no seam or native change, and runs wherever the free-space route runs. A
mirror is a discontinuity of the field's representation, so a march must stop there and
restart its history in any architecture: the composition loses no accuracy. Its cost is a
host copy of the field and the transform pair of the element at every handoff (section
6.4).

**B — mirror events inside the march.** A new leaf of `PropagationProblem` would declare
screens at fixed positions, the march would land on them, apply them on the device and
restart its history; one run would cover the cell. It would keep a cell on the device and
make it an ordinary `Simulation` for scans and the front end, but it is a change of
physics inside a run, which SEP 47 section 5.1 excludes; it needs kernels on both
backends, a rewrite of the frozen tables and an in-run version of the thin-screen
certificate, and a solid-state cell would also need media varying abruptly along z
(#193, #206). Rejected for now; A would be its reference if measurements ever justify it.

**Executor or recipe.** #191 allows a documented loop in user code if that is enough. It is
not: the loop must hold the ledger across runs (#238), the noise identity, the resume, the
frame and reference bookkeeping, the matched launch and the per-pass observables, and
every user would rewrite them. The executor stays thin: a loop over passes and segments,
with no scheduling of its own.

#### 6.2 Layers and modules

- **P** — `src/solprop/multipass.py` (new): `MultipassCell`, `CellMirror`, `Plate` with its
  responses, `CellDesign` and the design of section 4, with NumPy and SciPy only: the
  round-trip moment map over the pass sequence of section 4.1, its continuation in power,
  its stability and its refusals. It expands a mirror into `ThinLens` and a
  `SpectralTransfer` of the coating, and builds the matched launch with
  `FieldPlane.from_analytic`. It imports nothing from the driver. That suffices for a
  scalar radial cell. For two
  components, and on the Cartesian grid where the image along p matters, `Mirror` gains an
  unfolded form that applies its Jones map and image and keeps the forward axis (phase 3);
  no new element class is added. `PlanarInterface` gains a matched, coated case
  (`interface="matched"`): the medium changes with the field scaled to conserve the axial
  flux, for an antireflection-coated plate. `critical_power` gains its mass as a keyword
  (section 4.3). Registering the module touches `architecture.md` section 4, the figure
  builder behind `structure.md` figure 1 and the package's list of submodules, which the
  structure and package tests check.
- **D** — `src/solprop/driver/multipass.py` (new): `MultipassSimulation`, exported from
  `solprop.driver` beside `Simulation`; the pass loop, the ledger, the cell record with
  its transaction and its lock, and the noise identity. `SimulationRunner.run`, in
  `driver/runner.py`, gains two things its private paths already do: it accepts a
  `CouplingLaunch` bound to its port, as `Simulation.with_input` does, and a `stop` flag
  (`solprop.log.StopFlag`), as the scan's `_run_point` does. The scan driver accepts a
  cell point (section 5.5). That is more than its type check: the worker of a point
  reads a `Simulation` for its reseeding, its statistics, its memory probe, its record
  and its capture, and each of these gets its counterpart for a cell, in the scan
  modules, which hold the logic of a point today; the `Batch` path gets the refusal.
  `simulation.py` is not touched for it.
- **X** — `src/solprop/processing/multipass.py` (new): the per-pass series of section 5.2
  from the pass outputs and the cell record, the intensity integral with its quadrature
  error (section 4.6), and one plotting function. In `processing/compression.py`, the
  compressors gain the physical-flux objective that their refusal of a spatial output
  names: one transfer applied to every column of a spatial plane and the axial power in
  time (section 4.6). Their refusal of a spatial output without that objective stays.
- **T**, **B** — none. `PropagationProblem`, `RunRequest`, the stage bundles and the frozen
  tables of `tests/backends/test_driver.py` are unchanged. The new modules enter
  `architecture.md` section 4 and `structure.md` figure 1.

#### 6.3 Backends

No backend code. Every run is an ordinary free-space run on the backend it selects; the
qualification of the cell is that of its runs, with parity between `numpy_ref` and the
native CPU build (ADR-068, ADR-077). A reused runner on a
GPU build compiles but is not qualified today ([backends](../backends.md)); a cell on a GPU
inherits that status, plus one host round trip of the field at every handoff.

#### 6.4 Cost in time and memory

At every handoff the executor adds: the host copy of one field of
$n_r\times n_{\rm pol}\times n_\omega$ complex values (8 MB for 128 × 1 × 4096); the element's
transform pair on the host, $O(n_r^2n_\omega)$ for the radial transform; and a cold start of
the step controller. A gas cell has one handoff per pass, at the mirror. A cell with
plates has one at every segment boundary: $2k+1$ per pass for $k$ plates in a nonlinear
ambient gas, the last of them the mirror, and $k$ for plates in linear gaps, where the
gaps, the interfaces and the mirror are applied in one transport. A gas cell holds one
backend. A cell with plates holds one per distinct nonlinear segment, three for two equal
plates placed symmetrically in a nonlinear ambient gas (the plates, the outer gaps, the
inner gap), and their memory adds (section 5.5). The last run of every pass retains the
field of its last plane, the checkpoint; other fields are retained only when asked. The
overhead against the propagation is recorded in a benchmark ledger entry for two short
cases, a gas cell and a cell with two plates in a nonlinear gas, on about ten steps per
segment over three passes, with the allocations and the transfers counted, not estimated.

#### 6.5 Cost in code

Expected, from `poe size`, over the five phases: about 2100 lines of code in `src/solprop`
(design and description 620, with the round-trip moment map of section 4.4, its
continuation and its stability; executor, record and transaction 600; processing, the
flux objective of the compressors and plotting 400; the runner and interface changes 110;
scans of cells 300; documentation helpers 70), about 650 lines of public docstrings, and
about 1900 lines of tests. The first draft expected 1550, 520 and 1350. The difference is
what the AI review added to the contracts (sections 4.4, 4.6, 5.2, 5.3 and 5.5); it is
estimated by scaling the first figures, not measured.

The implementation is one branch and one pull request, its phases milestones on it
(section 11). The size workflow measures the net growth of every line of the Python
files of `src/solprop` against the merge base, code, docstrings, comments and blank lines
together: about 3000 lines here, twice its budget of 1500. No removal offsets it. The
reuse is in what is not written: no element class, no reader, no executor and no backend
code. Whether that pull request carries the `size-ok` label, or the SEP is granted the
delivery by phases that SEP 47 was, is the maintainers' decision (Q8).

#### 6.6 Dependencies

None: the fixed point of section 4.2 uses `scipy.optimize`.

### 7. Verification and validation

Every comparison belongs to a class of [testing.md](../testing.md) section 3, and its
tolerance is constructed before the reference result is known: a rounding bound from
operation counts for a matched computation (class R), an accuracy envelope against a
tighter or converged reference for runs whose trajectories may differ (class T). The
design's fast tests are closed forms and matched algebra, class R. A published number is
a witness: the test asserts the design's own value for the stated inputs, and the
distance to the publication is recorded with those inputs, not turned into a tolerance.

**Design (fast tests).**

- $\psi$, $z_R$, $w_0$ and $w_m$ against the closed forms of section 4.2, including the
  near-concentric sensitivity and a beam quality other than one; the numerical fixed
  point against the closed form for the gas cell, to a rounding bound.
- $P_{\rm eff}$ for a Gaussian ($P_{\rm peak}/\sqrt2$) and a sech² ($2P_{\rm peak}/3$) pulse on a
  grid, to the quadrature's error.
- The generated launch: its radius and curvature against the design, and its beam
  quality, measured from its second moments in space and in angle, equal to one within
  the grid's error; `matched_plane` refused for a design of another beam quality.
- The cell of [4] section 3 (1030 nm, 700 fs, 7 mJ, argon at 1 bar with
  $n_2=0.97\times10^{-23}$ m²/W, $R=1$ m, $L=1.995$ m), in the conventions of [4]
  (`critical="published"`): $\sigma=0.595$, the matched waist $\sigma^{1/4}w_{0,\rm lin}$,
  $B=4.26$ rad (4.3 in [4]). The design's default constant gives $\sigma=0.618$ and
  $B=4.17$ rad. [4] quotes a linear waist of 131 µm, which corresponds to $L=1.9945$ m,
  against 128 µm at 1.995 m.
- MPC3 of [10] in its Fig. 2 (2 mJ, 300 fs, $M^2=1.05$, argon at 2 bar, $R=0.5$ m,
  $L=0.978$ m), in the conventions of [10] (section 4.3): the linear mirror fluence, 111
  mJ/cm² in [10] and 110.9 here, and the matched fluence, 170 and 170.1. The design's own
  conventions give 151 mJ/cm² with the index of [10] and 153 with optmat's, recorded
  beside them. The 164 mJ/cm² of the first draft was the exact $P_{\rm eff}$ with the
  published constant and $M^2$ out of $\sigma$.
- The four cells of [10], its Table 2, with the lasers of its Table 1 and the conventions
  of [10]: the matched mirror fluence, 70, 133, 156 and 145 mJ/cm² against 75, 130, 160
  and 140 for MPC1 to MPC4, and the focus intensity with a Gaussian peak power, 1.9, 4.7,
  2.0 and 2.3 × 10¹³ W/cm² against 2.0, 5.3, 2.0 and 2.3. MPC3 is computed at 2 bar
  (section 4.3); the 1.2 bar of the table gives 131 mJ/cm² and $1.65\times10^{13}$. The
  differences, −7 to +3 % on the fluence and −11 to −2 % on the intensity, are recorded.
  They are not a tolerance: the table prints two figures, which is 0.7 to 3.8 % for these
  values, and the inputs behind it are published only in part (the pressure and the
  duration of each calculation, the peak power taken for the intensity).
- The moment map of section 4.4 against the $\hat q$ form for a fill, to rounding, with
  and without a beam quality, and against the closed form with the plates removed.
- Stability: the round-trip trace with the plates in place; $\operatorname{tr}J/2$
  against $\cos2\Psi$ at zero power and the unit determinant of $J$; a cell with a fixed
  lens of 0.05 m at the centre of $R=0.3$ m, $L=0.55$ m, whose pass has a half-trace of
  −1.29, refused although $L<2R$; a continuation that loses its branch, reported with its
  power; the asymmetric pair of the next item, stable at $x_p=50$
  ($\operatorname{tr}J/2=-0.77$) and refused at $x_p=200$ ($-1.57$), its branch lost at
  $x_p\approx102$; the peak slice of a fill above $M^4P_c$ reported as having no
  stationary beam; a cell in vacuum designed without a call to `critical_power`.
- The order of the plates: two unequal plates at asymmetric positions ($R=0.3$ m,
  $L=0.55$ m, 1 mm at 50 mm and 2 mm at 400 mm from the first mirror, $x_p=50$). Crossed
  in reverse on the return pass, the stationary radii are 645 µm on the first mirror and
  576 µm on the second; repeating the forward order on every pass gives 605 µm on both,
  6 % and 5 % off. A symmetric pair gives the same radius either way and cannot test the
  order. A plate touching a mirror, with its gap of zero length.
- From pass to pass: the drift of a launch through passes of falling $\sigma_j$ against
  the matched radii of the same passes, which differ; a cell with two different
  reflectances; the fixed-pulse limit stated in the design's output.
- The index of a glass: the design's total for silica is the sum of its electronic and
  delayed shares; for a circular launch its factor is 0.718 with the tensor bundle; the
  copolarised bundle is refused there; an override changes the design and is listed
  beside the run's coefficients.
- The linear eigenmode of [12] ($R=300$ mm, $L\approx550$ mm, 1037 nm): radii of 0.57 mm on
  the mirrors, 0.17 mm at the centre and 0.34 mm in the plates, which the closed form gives
  to the precision of the quoted values.
- The matched waist of [13]: 372 µm in diameter for the linear match, which the closed form
  gives (370 µm) with the published $M^2=1.24\times1.28$. Its nonlinear match in its own
  conventions (section 4.4): a centre waist of 344 µm and a focus to inject about 4 mm
  away, where the moment lens gives 335 to 339 µm and 4.4 to 5.0 mm, and the on-axis lens
  273 to 280 µm and 13 to 14 mm.
- The linear mirror radius of [2], 460 µm, which the closed form gives (459 µm) for
  $R=0.35$ m and $L=0.54$ m with the meniscus given to the mirror (section 2).
- The two cells of [15] (section 4.4): the mirror radius falls with plates at the centre
  and rises with plates beside the mirrors, the centre radius moving the other way; −32 %
  and +17 % with the moment lens against the −30 % and +15 % of [15], and −51 % and +47 %
  with the on-axis lens. The positions of the five plates of its first cell are an
  assumption, stated with the test.
- The thin-lens limit of a plate against the moment law in an ambient medium of index
  other than one and, for $\alpha=2$, against an independent radial integration of the
  Gaussian phase kick: for $w=0.7$ mm, $P=10^8$ W, $n_2=2.53\times10^{-20}$ m²/W, $d=1$
  mm and $n_a=1.5$, both give $1/f=4.472\times10^{-3}$ m⁻¹, where the formula without
  $n_a$ gives $6.708\times10^{-3}$. The index 1.5 checks the algebra; it is not a use
  case. The interface rule of section 4.4, and the refusals.

**Elements (fast tests).** The unfolded mirror against `Mirror` followed by the rotation of
section 4.1 on a Cartesian plane, with a field that has no symmetry in $x$ or $y$; two
flat, lossless reflections restore the field, its handedness and its image, where one
alone does not; a scalar cell against a two-component one launched along s, and along p
up to the sign of section 4.1, with the constant phase of the coating in both; the
ledger of a coating closes bin by bin; a `ChirpedMirror` coating leaves the reference
uncalibrated, `absolute()` refused and the status kept through a resume; the matched
interface conserves the axial flux to the transform's rounding.

**Composition (medium tier, a few passes of about ten steps).** The checks of #191:

- **Identity handoff:** a flat, lossless mirror between two passes of $L$ against one run of
  $2L$, for a scalar field on the radial grid, where that reflection is the identity (on
  a Cartesian grid or with two components the known image and Jones map are undone
  first), as an accuracy envelope (class T), on both CPU backends.
- **Paraxial Gaussian through an ABCD system:** a linear cell launched on its eigenmode keeps
  its second-moment radius at the mirrors and the centre; launched off it, the radius
  oscillates with the period the Gouy phase predicts; an aberrated launch, a Gaussian
  with a quartic phase, follows the linear moment law in its second-moment radius while
  its $1/e^2$ radius does not; tolerances from the thin-screen certificate and a grid
  convergence study.
- **Coherent phase and delay:** with a calibrated coating, the plane's reference after
  several passes equals the sum of the frame integrals, and the absolute fields on the
  two sides of each mirror satisfy the stated map, the screen and the Jones matrix of
  section 4.1, with consistent references. With `ChirpedMirror` data the same map holds
  on the stored samples and `absolute()` is refused.
- **Energy:** in a linear cell whose field stays inside the grid, the exit energy of each
  pass equals its entrance energy bin by bin within the run's bound, and the next
  entrance is that times the coating's reflectance to the element's rounding bound. With
  a Kerr response the bins exchange energy and only the totals are compared, within the
  run's accuracy envelope. Absorbers and interfaces have their own ledger terms, and the
  ledger of the cell closes within the bounds accumulated from the launch (section 5.3).
- **Dispersion:** the spectral phase of a linear cell of $N$ passes is that of its $N-1$
  reflections, $\lceil(N-1)/2\rceil$ times the second mirror's coating and
  $\lfloor(N-1)/2\rfloor$ times the first's, plus the fill's dispersion over $NL$.
- **Noise:** noiseless, seeded, unseeded (the seed read back from the record) and
  supplied-realisation launches; no draw after the first launch; an interruption before
  the first committed pass redraws the same realisation.
- **Resume:** interruptions after a segment completes, after the mirror is applied and
  before the pass record is published each resume to a cell that applies every element
  once and closes the same ledger; a resumed cell repeats an uninterrupted one within an
  accuracy envelope (class T); a boundary stop is not a committed pass; a changed
  identity is a conflict; every runner is closed after a failure in construction, a
  failure in a run and a stop.
- **Weak Kerr:** for $P\ll P_c$ the intensity integral of section 4.6 approaches
  $B_{\rm axis}$ with $\sigma\to1$, and its quadrature error falls as it should over
  three samplings of the analytic Gaussian integral.
- **Field against envelope:** a narrowband pulse through a mirror and through a plate on
  a field grid and on an envelope grid, within an accuracy envelope.
- **Radial against Cartesian** on the symmetric on-axis cell, as #191 asks (phase 3).

**Solid state (phase 3).** A glass medium with a Kerr response on the radial route, which
nothing refuses today but no test covers; Fresnel ledger $R+T=1$ through `PlanarInterface`;
flux conservation through the matched interface; the design's fixed point with a plate
against the linear run. One silica plate with the Kerr and Raman responses of the bundle,
a long pulse and a linear polarisation: the on-axis phase against $k_0n_2I_0d$ with the
total index, the record listing the design's two shares beside the run's coefficients,
and both scaled to a publication's index changing the two alike. Two different glasses
in one cell, each with its own responses. The change of the second moment across one
thin plate, for a long pulse at low B-integral, against the moment lens of section 4.4,
which is exact there for $\alpha=2$: this checks the run, not the closure. Then the
closure itself: a plate cell launched on the eigenmode of each lens, whose mirror radius
is followed over the passes; the one that keeps the beam more stationary, and by how
much, is the evidence for Q7. A cell of two unequal plates at asymmetric positions,
propagated at a power where its stationary beam is stable, against the design's two
mirror radii; and at a power where the design finds it unstable, to see whether the
propagated radii depart as the map predicts.

**Integrated references (galleries, run by hand).** Published simulations and measurements
are witnesses, not truth. Each comparison fixes its metric and its tolerance in the phase
report before its inputs are adjusted, and keeps three things apart: the numerical error
of the run, from its convergence; the uncertainty of the inputs carried to the metric
(the 12 % of [10] on $n_2$ is ±15 % on the B-integral); and the uncertainty of the
measurement. Its inputs are tabulated as in section 4.3: published, corrected by a named
person, selected. A reproduction whose unstated inputs were tuned is labelled as tuned,
not as a validation. A Fourier-limited duration quoted from a publication is compared
with the spectral transform limit of section 4.6, and a compressed duration with the
compressed pulse and the on-axis pulse, both reported, unless the publication says which
part of the beam it measured.

- [4] section 3, with its mirrors of 99.5 %: the beam radius at the mirrors and the centre
  over 16 passes for the linear and the nonlinear launch (2.5 → 1.9 mm against
  2.18 ± 0.02 mm at the mirrors, 103 to 133 µm against 117 ± 1 µm at the centre), the peak
  intensity at the centre (36 to 81 against 43 to 64 TW/cm²) and the on-axis B-integral of
  the first round trip (11.2 rad in [4]). The radius [4] plots is a time-averaged RMS
  radius, the second-moment radius of section 4.6.
- [10] MPC2 (2 mJ, 320 fs, $R=300$ mm, $L=595$ mm, argon about 1 bar, 27 passes as in the
  simulation of [10], whose Table 2 extracts at pass 25): the spectral width at $1/e^2$, the
  Fourier-limited duration against pressure and its $1/p$ trend, and the compressed
  duration with $-1200$ fs² and $+2000$ fs³ (18.5 fs in [10]); the measured spectra and
  durations of [10] as experimental witnesses. The measured durations of [10] are
  retrievals of a single-shot SHG-FROG on the collimated beam, and its Fourier-limited
  durations those of measured spectra; [10] does not say whether its simulated profile is
  taken on the axis or over the beam. The spectral transform limit, the compressed pulse
  and the on-axis pulse of section 4.6 are therefore all three reported against them.
  The coating of the injection and extraction mirrors, the same as the cell's in [10], is
  applied by the gallery to the launch and to the result (section 4.1).
- [10] MPC1, MPC3 and MPC4: the spectral width at $1/e^2$ (140–160 nm for all four
  cells), the Fourier-limited durations (18.3, 19.5 and 20.5 fs measured, 17.9 fs for
  MPC2) and the compressed ones; and the spectral width and Fourier-limited duration of
  MPC1 against pressure at 800 µJ (its Fig. 3), as a scan (phase 5). MPC3 runs at the
  1.2 bar of its measurement; where [10] gives two pressures for a cell (section 4.3)
  both are run. [10] shows that the peaks near 1030 nm in the short cells depend on the
  temporal contrast of the input (its Fig. 6), so spectral shapes carry the uncertainty of
  the launched pulse, and the comparison is on widths and durations.
- [12]: two 2 mm fused-silica plates 12.5 cm from the mirrors, in air, 41 passes, 200 µJ
  at 1037 nm stretched from 170 fs to 600 fs. With a negative pre-chirp the spectrum
  narrows to 3.8 nm at pass 6 and broadens from pass 7 to 49.6 nm (46.7 fs
  Fourier-limited) at pass 41; with a positive one it reaches 63.1 nm (43.3 fs). The
  narrowing pass by pass tests the per-pass observables directly. [12] states a
  transmission of 81 % with the plates and 95 % without them, which puts the loss of one
  plate crossing near 0.2 % if both figures hold for the same 41 passes, and a beam
  quality of 1.12 before the cell and 1.15 after it. Its $n_2$ and its mirror dispersion
  are not stated; they enter as declared assumptions with their ranges. The dataset its
  authors deposited (doi:10.57760/sciencedb.09731) is inspected in phase 3 for what else
  it fixes, before any loss or pulse is fitted.
- [13]: 11 round trips, mirrors with a group-delay dispersion below 30 fs² in magnitude, a
  compressor of −5500 fs²; the cell in air alone (about 100 fs Fourier-limited) and with
  its two plates (42 fs Fourier-limited, 45 fs compressed by SHG-FROG, 0.5 rad per pass
  in the air and about 0.3 rad in each plate), in circular polarisation, which exercises two
  field components, the change of handedness at each bounce (section 4.1), the tensor
  Raman response of the plates and the polarisation factors of the design (section 4.4).
- [14]: 24 passes near the edge of stability, mirrors of 1 m about 2 m apart, a 0.8 mm
  plate before each mirror in vacuum, 30 mJ, about 460 fs, $M^2=1.16\times1.18$. Its
  stability parameter, $|A+D|/2\approx0.992$, is that of the round trip, which puts $L$
  at $1.998R$ and agrees with the beam diameter of about 7.2 mm it reports on the
  mirrors; read as that of one pass it would give 4.9 mm. The comparison is on the width
  at −10 dB (32.5 nm measured, 32.2 nm simulated in [14]), the Fourier-limited duration
  (91.8 fs measured, 93 fs compressed), the transmission (90 %) and the beam diameter on
  the mirrors (section 4.4). It stresses the design's sensitivity near the edge and the
  radial grid, whose mirror spot is almost 30 times the focus.

**Sensitivity to the nonlinear index (phases 2 to 4).** The comparisons with measurements
use optmat's $n_2$, which is what a user of solprop gets. Each of them, for [4], [10] and
the solid-state reference chosen in phase 3, is run a second time with the publication's
index ($9.7\times10^{-24}$ m²/W per bar of argon for [4] and [10], $2.2\times10^{-20}$ m²/W
for the silica of [14]; for [12] the declared range), through the design's `n2` and the
run's responses scaled to it as section 4.3 says, each launch matched by its own design.
The phase report gives, for each reference, the change of the spectral width, the
Fourier-limited and compressed durations, the mirror radius over the passes and the
B-integral, against the design's estimate of section 4.3 (+3.6 to +4.1 % on the
B-integral for argon). The index of [10] carries its own 12 %, so the two runs bracket
less than that uncertainty. Where the difference between the two runs exceeds the
tolerance against the measurement, the comparison states both.

**Scans (phase 5).** A scan of three cells reproduces the same cells run alone: within a
rounding bound on a prescribed schedule (class R, a scan point and its run), within an
accuracy envelope for adaptive runs (class T), on each admitted executor, `Local` and
`Pool`, and `Queue` and `Slurm` where the platform has them. `Batch` refuses a cell by
name. A matched and a mismatched pressure scan run through the public builder, and a
launch kept from another pressure is refused by the port. A rerun skips completed points
and resumes an interrupted one from its last committed pass; a stop reaches the segment
in progress; consecutive points that differ only in energy reuse the runners and rerun
the transverse checks; the admission of a point with plates is the sum of its runners;
a per-pass reduction refuses a scan of `passes`; the seeds of cell points are those of
simulation points, explicit and derived.

**Convergence.** Radial samples and outer radius, temporal samples, tolerance and the
screen budget, at three levels each.

### 8. Documentation

- A manual page, `docs/guide/workflows/multipass-cells.md`: the unfolded model and its
  limits, the design and its conventions, including the closure for plates and the source
  of the nonlinear index and how to reproduce a publication's, running a gas
  and a solid-state cell, scanning cells, the ledger, resuming, and what is refused; a
  section of [scans](../../guide/workflows/scans.md) on builders of cells; a section of
  [compression](../../guide/workflows/compression.md) on spatial planes, with the three
  durations of section 4.6.
- Docstrings for every public name; a row in the composition table of
  [coupling simulations](../../guide/workflows/coupling.md).
- A focused tutorial in `docs/tutorials/free_space/`: a short linear-then-Kerr cell of a few
  passes, running in seconds.
- Three worked examples in `docs/examples/`: the matching study of [4]; the four cells of
  [10], with the pressure scan of MPC1; and a solid-state cell, [13] or [14], chosen in
  phase 3 by the completeness of its data (Q5).
- `architecture.md` section 4, `structure.md` figure 1, the capability matrix and a
  `changelog.d/` fragment per phase.

### 9. Compatibility and impact

Additive. No default changes, no change to the run record or the output schema of a single
run, and none to the seam. The cell record is a new file with its own version.
`SimulationRunner.run` gains an accepted input type and a stop flag. `PlanarInterface`
gains an option whose default keeps today's behaviour. The compressors gain an objective
for spatial planes, and still refuse a spatial output without it. `critical_power` gains
a keyword whose default is today's mass. `run_scan` refuses a cell on `Batch`. Nothing is
expected on the differences-from-Luna page.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Mirror events inside the native march (B) | One run per cell and no host round trip, but a change of physics inside a run (SEP 47 section 5.1), kernels on both backends, a seam change and media varying abruptly along z for plates; no accuracy gain, since a march restarts at a mirror anyway |
| A reduced propagation model (ABCD and a one-dimensional pulse equation, or a hybrid model) | Fast and useful for sizing, which the design keeps; it cannot show aberrations, spatio-spectral coupling or homogeneity, which is why a full propagation is wanted |
| Propagation in the Laguerre–Gauss basis of the cavity | Few modes in the linear regime and mirrors diagonal, but the basis depends on frequency and the mode count grows with nonlinearity; a new route |
| A rescaled transverse coordinate following the beam size | Would shrink the grid of a near-concentric cell; an optimisation for later, measured against the fixed grid |
| One HDF5 file with a group per pass | One file, but a change to the output layer; a directory of ordinary run files and a cell record reuses every reader (Q2) |
| A fresh `with_input` run per pass | Correct but pays the backend's allocation and planning at every pass; the runner pays it once |
| A loop in user code only | Leaves the ledger, the noise identity, the resume and the observables to every user |
| A scan loop over cells in user code | Loses the executors, the resume, the seeds and the collected file that `run_scan` already provides |
| The on-axis parabolic Kerr lens as the only plate lens | Simple, but $2\alpha$ times the second-moment response of the beam and inconsistent with the $\sigma$ of a fill in the same design; kept as an option for comparison (Q7) |
| The $\hat q$ parameter of [4] extended to plates as published | Undefined for $\sigma<0$, which every plate of [12, 13, 14] reaches; the moment law is the same for $\sigma>0$ and holds for both signs |
| The published constant of $P_c$ as the design's default | Would not reproduce [4] and [10] by default in any case, since the index and the effective power already differ; makes the plate lens a calibration of the exact Gaussian kick; kept as an option (Q1) |
| The beam quality as [10] applies it, in the beam size and not in $\sigma$ | Multiplies the Kerr term by $M^4$, 10 % at 1.05 and 59 % at 1.26; reproduced through `power` (section 4.3) |
| A generated launch that realises $M^2>1$ | $M^2$ does not determine a field; a family of fields would be a model of its own; the caller's plane is admitted as it is |
| One backend alive at a time in a cell with plates | Bounds a point's memory by its largest segment, but pays allocation and planning at every segment of every pass |
| Cells on the `Batch` executor | Needs a group session over chains of runs with host elements between them: a backend change |
| One file for the exit plane of each pass | Makes the exit plane a stored object, but doubles the number of files; the checkpoint in the last run's output and a recomputed exit plane need none (Q2) |

### 11. Phases

| Phase | Deliverables | Required evidence |
|---|---|---|
| **1 — Cell design** | `multipass.py`: the description with the plate layout and its reverse traversal, the linear and Kerr-corrected design with the beam quality, the round-trip moment map for plates with its continuation, stability and refusals, both plate lenses, the matched radius and the drift, the matched launch plane; the design section of the manual page | Closed forms, the numbers of [2], [4], [10] (Fig. 2 and Table 2, in its conventions), [12] and [13], the signs of [15], the stability and plate-order checks, $P_{\rm eff}$ for two shapes, refusals |
| **2 — Gas-filled cells** | `driver/multipass.py`, the runner accepting a launch and a stop flag, the ledger, the cell record with its transaction, the planes of a pass, the noise identity, minimal per-pass series; the tutorial; the worked example of [4] | The composition checks of section 7 but the radial–Cartesian one; parity on the CPU backends; the benchmark ledger entry of the gas case; the comparison with [4] and its sensitivity to $n_2$ |
| **3 — Solid-state cells** | Plates with their responses, the matched interface, linear and nonlinear ambient gaps, the centre plane in a gap, the unfolded `Mirror` for two components, the Cartesian check, the solid-state worked example | The solid-state checks of section 7, including the closure comparison for Q7; radial against Cartesian; the benchmark ledger entry of the two-plate case; the comparison with the chosen reference among [12, 13, 14] and its sensitivity to $n_2$ |
| **4 — Observables and compression** | `processing/multipass.py`, the flux objective of the compressors, plotting, the homogeneity measure, the worked example of the four cells of [10], the complete manual page | The reference comparisons of section 7 with their metrics, tolerances and input tables, and their sensitivity to $n_2$; the compression of a separable beam and of a beam with a spatially varying chirp; convergence studies; final gates |
| **5 — Scans of cells** | `run_scan` accepting cells on `Local`, `Pool`, `Queue` and `Slurm`, reductions on a cell result, the pressure scan of MPC1 in the worked example of [10], the section of the scans page | The scan checks of section 7; the spectral width and Fourier-limited duration of MPC1 against pressure |

The implementation is one branch, `feature/sep-0050-multipass-cells`, and one pull
request, a draft from the start, as SEP 0 asks. The phases are milestones on that branch,
each qualified and reported in Part II when it is reached; they are not separate pull
requests unless the maintainers grant that (Q8).

The follow-ups go in issues: the spot pattern and re-entrance, the coupling optics and
their nonlinearity, recipes and front-end hooks for cells (SEP 41), astigmatic mirrors on
the Cartesian grid, cells on the `Batch` executor, generated launches of beam quality
above one, a tool that packs a finished cell into one file, and mirror events inside the
march if a measurement ever calls for them.

### 12. Open questions

1. **Q1 — The convention of $\sigma$:** resolved by the author on 2026-11-13. The design
   uses the exact $P_{\rm eff}$ of eq. (6) of [4] on the launched pulse and $P_{\rm peak}$
   in the B-integral. The default constant of $P_c$ is the derivation's
   $\lambda_0^2/(2\pi n_0n_2)$, as the AI review recommended (R1, R2) and the author
   decided; the published $3.77\lambda_0^2/(8\pi n_0n_2)$ stays selectable (section 4.3).
   The beam quality enters $\sigma$ as $M^4$ (the author, 2026-11-13, on R4). [10]
   computed with $E/(\sqrt2\,\tau)$ and without $M^2$ in $\sigma$, which the design
   reproduces through `power`; the fluence of MPC3 in its Table 2 was in all likelihood
   not computed at the 1.2 bar the table prints, and 2 bar reproduces it (section 4.3).
   The nonlinear index comes from optmat, and its effect against the literature's values
   is measured (sections 4.3 and 7).
2. **Q2 — The form of the output:** a directory of ordinary run files and a cell record
   (proposed), or one file. Left to the review (the author, 2026-11-13). The author
   prefers the directory, which every existing reader and the resume use as they are, and
   asks the reviewers to weigh the number of files it makes: one per nonlinear segment and
   pass, so 27 for the MPC2 of [10], 48 for [14], 82 for [12] with its air linear and 205
   with its air nonlinear, and 189 for the pressure scan of MPC1 at seven points. The
   volume is the same in one file, set by the retained planes (about 8 MB per field on 128
   radial and 4096 spectral samples); the count is what differs, and it matters on file
   systems with a quota of files. A tool that packs a finished cell into one file would be
   follow-up work. The AI review supports the directory with a versioned record, and asks
   that checkpoints be counted (R7): there are none beyond these files, the checkpoint of
   a pass being the last plane of its last run (section 5.2), and the record and its lock
   add two files per cell. The human review is still to weigh the count.
3. **Q3 — Scans of cells:** resolved by the author on 2026-11-12: in scope, phase 5
   (section 5.5).
4. **Q4 — The homogeneity measure** for phase 4. [12] and [14] both report the spectral
   overlap measure of Weitenberg et al.; it is the proposed default, defined and cited from
   its primary source in phase 4. Deferred to phase 4.
5. **Q5 — A solid-state reference:** answered by the author with [2, 12, 13, 14, 15]. Which
   one becomes the worked example, and how the inputs that [12] does not state (its $n_2$
   and its coating; its plate loss is bounded by the two transmissions it gives, section
   7) are fixed, is decided in phase 3. The conditions of [13] and [14] are published
   (sections 4.4 and 7). The second cell of [15] has a complementary pair of
   chirped mirrors, which the first version admits, and plates at Brewster's angle, which
   it does not (section 2); its first cell, at normal incidence, is within it.
6. **Q6 — Coating data:** whether `TaylorPhase`, `ChirpedMirror` data and a callable with
   its calibration are enough to describe the coatings in use. Deferred to phase 2, where
   the coatings are first applied (the author, 2026-11-13). A `ChirpedMirror` runs with
   an uncalibrated reference until its constant phase and delay are given (section 4.5).
7. **Q7 — The Kerr lens of a plate in the design:** the moment lens (proposed default,
   consistent with the $\sigma$ of a fill and exact for a Gaussian at the default
   constant) or the on-axis lens, $2\alpha$ times stronger (section 4.4). Both stay
   selectable. Deferred to phase 3, settled by the closure comparison of section 7 on
   propagated cells. The designs of [13] and [15] are reproduced by the moment lens and
   not by the on-axis one (section 4.4); that says which lens those designs use, not
   which is right.
8. **Q8 — Delivery:** SEP 0 asks for one branch and one pull request, and this
   implementation is about twice the size budget of one pull request (section 6.5).
   Whether it merges as one pull request with the `size-ok` label, or by the delivery in
   phases that SEP 47 was granted, is the maintainers' decision, raised by the AI review
   (R18). Until it is made, the plan is one pull request.

### 13. References

DOIs, titles and authors checked against Crossref on 2026-11-12, and for [15] on
2026-11-13.

1. D. Herriott, H. Kogelnik and R. Kompfner, "Off-axis paths in spherical mirror
   interferometers," *Applied Optics* **3**, 523 (1964). doi:10.1364/AO.3.000523
2. J. Schulte, T. Sartorius, J. Weitenberg, A. Vernaleken and P. Russbueldt, "Nonlinear pulse
   compression in a multi-pass cell," *Optics Letters* **41**, 4511 (2016).
   doi:10.1364/OL.41.004511
3. M. Hanna, X. Délen, L. Lavenu, F. Guichard, Y. Zaouter, F. Druon and P. Georges,
   "Nonlinear temporal compression in multipass cells: theory," *Journal of the Optical
   Society of America B* **34**, 1340 (2017). doi:10.1364/JOSAB.34.001340
4. M. Hanna, L. Daniault, F. Guichard, N. Daher, X. Délen, R. Lopez-Martens and P. Georges,
   "Nonlinear beam matching to gas-filled multipass cells," *OSA Continuum* **4**, 732
   (2021). doi:10.1364/OSAC.418247
5. P.-A. Bélanger and C. Paré, "Self-focusing of Gaussian beams: an alternate derivation,"
   *Applied Optics* **22**, 1293 (1983). doi:10.1364/AO.22.001293
6. G. Fibich and A. L. Gaeta, "Critical power for self-focusing in bulk media and in hollow
   waveguides," *Optics Letters* **25**, 335 (2000). doi:10.1364/OL.25.000335
7. A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and M. Kolesik,
   "Practitioner's guide to laser pulse propagation models and simulation," *The European
   Physical Journal Special Topics* **199**, 5 (2011). doi:10.1140/epjst/e2011-01503-3
8. S. N. Vlasov, V. A. Petrishchev and V. I. Talanov, "Averaged description of wave beams in
   linear and nonlinear media (the method of moments)," *Radiophysics and Quantum
   Electronics* **14**, 1062 (1971). doi:10.1007/BF01029467
9. A.-L. Viotti, M. Seidel, E. Escoto, S. Rajhans, W. P. Leemans, I. Hartl and C. M. Heyl,
   "Multi-pass cells for post-compression of ultrashort laser pulses," *Optica* **9**, 197
   (2022). doi:10.1364/OPTICA.449225
10. J.-F. Hergott, F. Réau, O. Tcherbakoff, F. Lepetit, N. Lericheux, M. Guer, H. Marroux,
    R. Géneaux, A. Tyagi, R. Cazali, P. Nonnon, D. Bresteau, T. Ruchon, P. Salières and
    T. Auguste, "Experimental and numerical study of direct sub-20fs multi-pass-cell
    post-compression of up to 4mJ, 80W Yb-based lasers," *Applied Physics B* **131**, 56
    (2025). doi:10.1007/s00340-025-08415-0
11. A. E. Siegman, *Lasers* (University Science Books, 1986).
12. L. Shen, J. Song, Y. Peng, G. Luo, Y. Liu, J. Sun and Y. Leng, "Post-compression of the
    GW-level femtosecond pulse in a solid-state multi-pass cell," *High Power Laser Science
    and Engineering* **12**, e80 (2024). doi:10.1017/hpl.2024.67
13. Z. Gao, J. Guo, Y. Huang, J. Li, L. Guo, Z. Tu, J. Xu and X. Liang, "High energy nonlinear
    pulse compression to sub-50 fs by a hybrid bulk-air multi-pass cell," *Optics Express*
    **33**, 13613 (2025). doi:10.1364/OE.549177
14. J. Manz, G. Barbiero, S. Klingebiel, C. Y. Teisset, H. Wang, M. Rampp, J. Brons,
    C. J. Saraceno and T. Metzger, "Bulk-based nonlinear pulse compression of 30 mJ and 150 W
    pulses of an ultrafast thin-disk amplifier," *Optics Express* **33**, 54907 (2025).
    doi:10.1364/OE.580757
15. A.-L. Viotti, C. Li, G. Arisholm, L. Winkelmann, I. Hartl, C. M. Heyl and M. Seidel,
    "Few-cycle pulse generation by double-stage hybrid multi-pass multi-plate nonlinear
    pulse compression," *Optics Letters* **48**, 984 (2023). doi:10.1364/OL.478790

## Part II — Implementation record

## Part III — History

- 2026-11-12: Drafted.
- 2026-11-12: Revised with the author's answers: scans of cells in scope (section 5.5,
  phase 5, Q3); solid-state references [12, 13, 14] (Q5); section 4.4 on what survives of
  the Kerr correction in cells with plates, and Q7; the four cells of [10] in the design
  checks and the galleries.
- 2026-11-13: [2] described as published, with the mirror substrates as the medium, and
  its linear radius in the design checks; [15] added, with the dependence of the Kerr
  correction on the position of the plates (section 4.4) and its check; the cells of
  Table 1 of [9] noted.
- 2026-11-13: Revised with the author's answers: Q1 resolved (the exact $P_{\rm eff}$; [10]
  used $E/(\sqrt2\,\tau)$ and has MPC3 at 2 bar; `power` reproduces a publication); the
  nonlinear index from optmat, its values for argon and silica against the literature's
  and a sensitivity run beside each reference comparison (sections 4.3 and 7); two
  coatings on the two mirrors of a cell (sections 2, 4.1 and 5.1); the context updated for
  the split of `simulation.py` and SEP 49, both merged.
- 2026-11-13: Q6 deferred to phase 2; the default constant of $P_c$ and the form of the
  output (Q2, with the number of files it makes) left to the review (the author).
- 2026-11-13: Tracking issue [#793](https://github.com/LupoLab/solprop/issues/793).
- 2026-11-13: GPT-6 completed the AI review below, incorporating the seven Codex
  findings on PR #794. Part I is unchanged; the findings await the author's response
  and the maintainer's decisions.
- 2026-11-13: Revised in response to the AI review, with the author's decisions (Claude
  Fable 5.2 as drafting agent). The default constant of $P_c$ is the variance one (Q1);
  the beam quality enters the design as $M^4$; plates carry their responses and are
  crossed in reverse on the return pass; the stability of a cell with plates, the output
  plane, the transaction of the resume, the noise identity, the three durations and the
  intensity integral are defined; the statement on Table 2 of [10] is corrected; scans of
  cells exclude `Batch`; the implementation is one branch and one pull request, its
  delivery left to the maintainers (Q8). The response to each finding follows the
  review.
- 2026-11-14: [4], [13], [14] and [15] read in full and Part I checked against them. The
  comparison with [13] is redone in its own conventions and extended to the two cells of
  [15] (section 4.4); the geometry of [14] is corrected, its stability parameter being
  that of the round trip (sections 4.4 and 7). The record of what was checked is in the
  probes section below.

### AI review — 2026-11-13

Reviewer: **GPT-6 (AI)**. Reviewed all of Part I in
[PR #794](https://github.com/LupoLab/solprop/pull/794) at
`d1419335614a90148b86f4572c7df43a329e8036`, against the architecture, conventions,
SEP 0, and the existing coupling, runner, spatial-statistics, compression and scan
contracts. These findings are part of the work tracked by
[#793](https://github.com/LupoLab/solprop/issues/793). They are review comments and
questions, not revisions to the proposed model or approval to implement it.

**Assessment.** Composing ordinary free-space runs and existing optical elements is
a sound direction. The separation of a sizing model from the propagated result, and
the explicit comparison of nonlinear-index choices, are useful. Before acceptance,
the proposal needs firmer contracts for the plate approximation, the physical output
plane, restart, spatial compression and scan execution. Several advertised cases
cannot be expressed by the proposed interface. **P1** below means a correctness or
contract issue to settle before accepting the affected scope; **P2** means a necessary
clarification, qualification change or maintainer decision.

**Evidence and limits.** Code paths were read at the commit above. Short Python 3.14.5
probes independently integrated a Gaussian plate's phase kick, checked a cavity
matrix, compared an integrated-spectrum transform limit with two orthogonal spatial
channels, and checked phase-only chirping and the existing medium guard. Their inputs
and results are given below. With optmat 0.9.1, `physdata.n2` gives
`1.0001440357542289e-23` m²/W for argon at 1030 nm, 1 bar and 293.15 K, consistent with
the rounded value in §4.3. The publisher's full texts of
[Hergott et al. [10]](https://link.springer.com/article/10.1007/s00340-025-08415-0)
and [Shen et al. [12]](https://doi.org/10.1017/hpl.2024.67) were checked for the
comparisons discussed in R17. Full text of [4] and [13] was not accessible through
the sources attempted; their detailed numerical claims, the other experimental
reproductions, and the author's corrections to [10] have not been independently
verified. No cell implementation or full-length propagation was run.

**Recommendations on the questions left to review.** For **Q1**, I recommend
`critical="variance"` as the default of the moment design, with the published
constant an explicit reproduction option. The Gaussian phase-kick check is then
consistent with the design's coefficient. Keep the Townes default of the existing
`critical_power` unchanged, distinguish all three powers in the record, and do not
describe the variance scale as a physical no-collapse guarantee (R1–R2). This is a
recommendation for a maintainer decision, not a change to Part I. For **Q2**, I support
ordinary run files plus a versioned cell manifest, provided R7 defines the saved exit
plane and commit/restart contract. A later packing tool is preferable to changing
every run reader for this feature. Count the checkpoint files as well as segment
files when reporting the storage cost.

1. **R1 — P2: Finish the plate-lens derivation in the stated medium
   (§§4.3–4.4, 5.1, Q1, Q7).** The thin-plate formula says that its focal length is
   measured in the ambient medium but omits its index, $n_a$. Combining the proposed
   moment law with the exit rule $U_a=(n_p/n_a)U_p$ gives the nonlinear kick
   $$
   \frac{1}{f_{K,a}}=
   \frac{x_p\lambda_0^2d}{\pi^2 n_a n_p w^4}
   =\frac{4}{\alpha}\frac{n_2P_{\rm eff}d}{\pi n_a w^4}.
   $$
   Here $P_{\rm eff}$ is the power in the plate. The on-axis lens likewise carries
   $1/n_a$. For a continuous Gaussian with
   $w=0.7$ mm, $P=10^8$ W, $n_2=2.53\times10^{-20}$ m²/W, $d=1$ mm and
   $n_a=1.5$, direct radial integration of the phase kick gives
   $1/f=0.00447215889$ m⁻¹ for $\alpha=2$, whereas the printed expression gives
   $0.00670823833$ m⁻¹. This deliberately non-unity index checks the algebra; it is
   not an added cell use case. Either retain $n_a$ or state and bound the
   $n_a\simeq1$ approximation for the admitted gases. Also make clear that substituting
   $\alpha=3.77/2$ rescales this exact Gaussian moment kick: it is a calibrated
   approximation, not the same variance identity. Please retain the independent
   phase-imprint check for $\alpha=2$ rather than validating two uses of the same
   moment formula against each other.

2. **R2 — P1: Specify stability and distinguish moment positivity from physical
   collapse (§§4.2–4.4, 5.1, 7).** $0<L<2R$ certifies the empty symmetric cell,
   not a cell with focusing elements. For example, $R=0.3$ m, $L=0.55$ m and a
   centred thin lens with $f=0.05$ m give half-trip trace/2 $=-1.2916667$ although
   $L<2R$. Define how the physical round-trip map selects a branch, establishes
   linear stability, and tests the Jacobian of the nonlinear $(W,U)$ map; convergence
   of `scipy.optimize` alone is not stability. Require positive $W$ throughout every
   segment, including an interior minimum, and specify the refusal for no stable
   solution or an ambiguous branch. Separately, $W>0$ is not proof that the actual
   field cannot collapse: the proposal's own Gaussian collapse threshold is below
   the zero-Hamiltonian threshold. Nor does $P_{\rm eff}<P_c$ imply
   $P_{\rm peak}<P_c$; a Gaussian at $P_{\rm eff}=0.8P_c$ has
   $P_{\rm peak}=1.13137P_c$. What does the promised peak-slice design report when
   that slice has no stationary fill solution? State the diagnostic's validity and
   failure semantics without making the full propagator's admission depend on an
   unproved collapse criterion. Include the exactly linear/vacuum limit explicitly,
   since `critical_power` refuses a zero susceptibility.

3. **R3 — P1: Define the physical plate layout and its reverse traversal
   (§§4.2, 4.4, 5.1–5.2, 6.1).** Incorporates
   [Codex's plate-order finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018681).
   Record a plate once as the interval $[p,p+d]$ measured from the first mirror.
   A return pass must traverse the plates in reverse order, with entrance
   $L-p-d$, the corresponding reversed gaps, and the appropriate incident/transmitted
   interface media. Use the same sequence in the design and the executor. A pair
   of unequal plates at asymmetric positions is necessary evidence; a centred plate
   or a symmetric pair hides the error. How are a plate touching a mirror and its
   two crossings around that reflection represented without a negative or duplicate
   gap, especially for the substrate geometry mentioned in §2?

4. **R4 — P1: Separate a beam-quality design parameter from a realizable launch
   (§§4.2–4.4, 5.1, 7).** Incorporates
   [Codex's beam-quality finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018691).
   The specified Gaussian has $M^2=1$, whereas the acceptance numbers use 1.05 and
   approximately 1.26. Define the measured radius and beam-quality convention and
   derive its place in the second-moment invariant and nonlinear closure. Merely
   enlarging a coherent Gaussian waist does not produce $M^2>1$: it changes its
   Rayleigh range while keeping its beam quality at one. An $M^2$ value alone also
   does not determine a unique coherent field or its Kerr response. Please choose
   between a qualified field family/measurement for such launches and an explicitly
   approximate design-only parameter whose output does not claim to realize it.
   Keep ideal-Gaussian numerical checks separate from experimental comparisons with
   measured beam quality, and test the reported quality of any generated launch.

5. **R5 — P1: Bind nonlinear responses to each segment and distinguish electronic
   and delayed indices (§§4.3–4.4, 5.1–5.2, 7).** A cell admits multiple bulk media
   and a design `n2` for each plate, but the executor has one `responses` sequence
   for all plates and one for the ambient. A `SpatialMedium` supplies dispersion and
   density; it does not choose the plate's nonlinear response. Two different glasses
   therefore cannot be assigned their respective Kerr/Raman responses through this
   API. Specify a per-segment binding, or narrow the admitted plate family explicitly.
   The quoted silica index includes a delayed Raman share, while `KerrResponse` is
   instantaneous: setting its coefficient from the total index and then adding Raman
   counts that share twice; using only the electronic share gives a different design.
   Define the effective index used for a finite pulse, the polarization convention,
   and what scaling a literature index changes. The instantaneous circular factor
   $2/3$ must not silently scale a delayed tensor response. Require the recorded
   design/run comparison to identify coefficients and response models per segment,
   and demonstrate one glass example with consistent units and normalization.

6. **R6 — P2: Bound the fixed-pulse approximation when coatings or plates change
   the pulse (§§4.1, 4.3–4.4, 5.1).** The recurrence
   $P_{{\rm eff},j}=R_c^jP_{{\rm eff},0}$ assumes an unchanged temporal shape and
   identical, frequency-independent losses. Different coatings, spectral filtering,
   plate dispersion and chirped mirrors are all admitted. A phase-only coating can
   preserve energy while changing $P_{\rm eff}$: an initially transform-limited
   Gaussian of 300 fs intensity FWHM with $10^5$ fs² GDD has an effective-power
   ratio of approximately 0.308747 at unchanged energy, as checked by Fourier
   transformation. Please define whether the design
   propagates a temporal surrogate through these known linear elements or reports
   a fixed-pulse estimate with an explicit limitation. In particular, distinguish
   instantaneous matched radii at reduced power from the actual drift of one launch
   propagated through a changing map; substituting a new $\sigma_j$ into the
   stationary formula does not compute that drift. Give an unequal-coating example
   and state which result is compared with the propagation.

7. **R7 — P1: Define the cell's output plane and make restart a transaction
   (§§4.1, 5.2–5.3, 5.5, 6.4, Q2).** A gas pass has one run output; a plate pass
   has several, and its last run may end before a linear gap and the final mirror.
   Consequently `pass_output(j)`, the scan's retained last-pass field, and the next
   launch are not necessarily the same plane. State whether an $N$-pass result is
   before or after the $N$th reflection, how many coating factors it contains, and
   how callers access both sides and each segment. Retain or reproducibly reconstruct
   the actual terminal `FieldPlane`, including its medium, geometry, calibrated
   reference and transport reports; an ordinary final-segment output alone is
   insufficient. Also define the terminal result when no segment has nonlinear
   responses.
   Define a manifest commit point after the required files are durable, the lock
   covering a cell directory, and the identity covering launch, seed, ordered
   elements, media/responses, settings and calibration. Specify conflict handling
   and removal/replacement of only an incomplete attempt. Test interruption after
   segment completion, after applying the mirror, and before publishing the pass
   record: restart must apply each element exactly once and preserve the ledger.
   Mandatory handoff/checkpoint retention must survive `fields="none"` in a scan,
   and boundary stops must not be marked as completed passes.

8. **R8 — P2: Expose noise selection and its single-draw identity
   (§§4.5, 5.2–5.5).** Incorporates
   [Codex's noise finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018703).
   Add the public choice of supported input noise or no noise, an explicit seed,
   and the scan reseeding contract. A `FieldPlane` contains samples, not the
   metadata needed to decide whether to add a fresh draw. State how a caller supplies
   an already noisy plane without adding another floor, when an unseeded draw is
   persisted, and how restart before the first completed pass recovers it. The
   existing scan derives a seed from its name and index when `ScanPoint.seed` is
   absent; preserve that behavior as well as explicit-seed checks. Test noiseless,
   seeded, initially unseeded, supplied-realization and interrupted-first-pass cases.
   Every reused segment runner should continue to carry the proposed no-noise model.

9. **R9 — P1: Admit every simultaneously live runner and account for handoff
   allocations (§§5.5, 6.3–6.4).** Incorporates
   [Codex's memory finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018698).
   Keeping one backend for every nonlinear segment makes the workspace cost their
   sum, not their maximum. Add the simultaneously held translated tables, retained
   planes, transport/coupling workspaces and any required host/device copies, avoiding
   double counting genuinely shared storage. Otherwise the scan can admit more cells
   than memory can hold. Either adopt that lifecycle explicitly or document a
   bounded-lifetime alternative and its planning cost. All retained runners need
   deterministic cleanup on construction failure, run failure and cancellation.
   A solid-state cell also pays a handoff at every segment boundary, not only once
   per pass as §6.4 suggests. Benchmark a short gas case and a short multi-segment
   case, reporting those actual allocations and transfers.

10. **R10 — P1: Rebuild a pressure-scan launch in its point's medium
    (§5.5).** Incorporates
    [Codex's scan-launch finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018694).
    The example reuses a nominal `launch` after changing `SpatialMedium`. The
    `same_basis` path calls `coupling.port.check_same_medium`, which rejects a
    different admittance; a direct call with otherwise identical metrics 1.0003
    and 1.0006 confirms that refusal. For a matched study, construct the design and
    plane in each point. For a mismatch study, keep the intended nominal beam
    parameters but sample and normalize the plane in that point's medium. State
    what is held fixed: energy/peak axial power, field amplitude, radius and
    curvature are not interchangeable constraints. Do not relabel the old metric
    or insert an artificial interface simply to satisfy the guard. Test both kinds
    of pressure scan through the actual public builder.

11. **R11 — P1: Specify the scan execution contract, especially `Batch`
    (§§3, 5.5, 6.2, 6.5, 11).** This requires more than widening the builder's
    type check. `_scan_batch.py` constructs one backend per `Simulation`, starts
    backend group sessions, and consumes their observations. It is not an arbitrary
    whole-task dispatcher, and a `MultipassSimulation` has no single kernel problem
    to submit there. Define how cell points integrate with it without violating
    the proposed absence of seam/backend changes, or explicitly defer/refuse that
    executor. `SimulationRunner.run` also has no public `stop` argument, whereas
    Local/Pool scans promise interruption of a running point within a step. State
    how cancellation reaches an active segment and how reporters, failures, leases,
    reduction schemas and retained-field choices work for each supported executor.
    Energy/chirp scans must rerun the existing transverse checks and preserve
    material/rate admission when they reuse a runner. Include short tests of every
    admitted execution path and update the cost estimate for the chosen design.
    Finally, settle the public import: §5.2 imports `MultipassSimulation` from
    `solprop.multipass`, but §6.2 assigns that module to P and the class to D.
    Use a driver import or a deliberate public facade without introducing a P-to-D
    dependency.

12. **R12 — P2: Materialize center observations and distinguish center from focus
    (§§2, 4.4, 5.2, 6.4, 7).** Incorporates
    [Codex's center-plane finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018708).
    A `FreeSpace` element has no internal run save planes. If $L/2$ lies in a
    linear ambient gap, split the transport there and retain or measure that
    intermediate plane. If it lies inside a nonlinear segment, translate the
    cell coordinate to that segment's local save position. Specify the observation
    side if it is exactly an interface. Also, §4.4 itself describes two waists
    around a central plate, and asymmetric layouts need not focus at $L/2$.
    Call the fixed observation the center; a claimed focus observable needs a
    defined search and sampling error. Define whether `mirror_radius` uses the
    existing fluence second-moment radius or the $1/e^2$ radius, which diverge for
    an aberrated beam.

13. **R13 — P1: Define physical spatial compression before promising compressed
    durations (§§3, 4.6, 7–8, phase 4).** The integrated spectral energy
    $S(\omega)$ determines a conventional spectral transform limit obtained from
    $\sqrt{S}$, but not the pulse a common compressor can produce when the beam
    has spatially varying spectra or phase. A Fourier probe with two orthogonal
    spatial channels on 4096 FFT-ordered bins, with amplitudes
    $\exp[-(k\mp70)^2/(2\times20^2)]$, gives a synthetic $\sqrt{S}$ peak
    1.99922 times the summed physical peak at the same energy.
    `processing.compression._require_modal` explicitly
    rejects spatial outputs; its array route merely sums squared amplitudes and
    does not supply the missing physical-flux objective. `apply_transfer` takes
    an output, not an arbitrary spectrum as §3 states. Please distinguish the
    integrated-spectrum diagnostic, any on-axis/aperture diagnostic, and compression
    by one common spectral transfer applied to the full field. Define the objective
    and duration from the spatial axial flux, including its frequency-dependent
    admittance, and say which measurement each literature comparison uses. Test a
    separable beam and a beam with spatially varying chirp; the latter must expose
    the difference between the spectral limit and attainable compression.

14. **R14 — P2: Measure the named B-integral and define its sampling error
    (§§4.3, 4.6, 7).** Spatial `peak_intensity` is the maximum over transverse
    nodes and time (`backends/spatial.py`), not the maximum in time on the axis.
    Those disagree for a ring or an aberrated beam. Use `on_axis_intensity` for
    the proposed axis diagnostic and name $\int \max_t I(0,t,z)\,dz$ precisely:
    it is not generally the nonlinear phase of one temporal slice. State the
    effective nonlinear coefficient for vector polarization and delayed responses,
    or restrict this diagnostic to a specified instantaneous scalar model. Join
    sample intervals within each medium, preserving interface sides and avoiding
    trapezoids across a coefficient jump. The center and end alone do not resolve
    a narrow focus, and an adaptive step history alone supplies no quadrature error
    estimate. Specify the sampling/refinement rule and verify its convergence on
    the analytic Gaussian integral before using it to compare matching models.

15. **R15 — P1: Preserve unknown coating phase/delay and test the actual mirror
    map (§§4.1, 4.5, 6.2, 7, Q6).** `ChirpedMirror` data omit constant phase and
    delay; `coupling.transport._reference` marks these terms uncalibrated, and
    `FieldPlane.absolute()` then raises. Thus the unconditional promise that
    `absolute()` is exact over every admitted cell is incompatible with an existing
    supported coating. Specify the required calibration, or retain the unknown
    terms and limit absolute-phase assertions to calibrated models. Restart must
    preserve that status. Nor is the field continuous across a nontrivial mirror
    screen: the coherent check should assert the stated screen/Jones map between
    the two sides, with consistent references. A single flat vector/Cartesian
    mirror still changes handedness and the transverse image; the identity
    handoff test must restrict itself to the scalar symmetric case or undo that
    known map before comparison. Use two reflections and an asymmetric test field
    to check the full convention, and require scalar/vector agreement for a
    linearly polarized input, including the coating's constant phase convention.

16. **R16 — P2: Make the qualification claims match the numerical policy
    (§7).** Restrict the per-frequency energy equality to a linear, lossless
    propagation case with only the specified coating losses: Kerr broadening
    redistributes energy between frequency bins, and absorbers/interfaces need
    their own ledger terms. Across separately integrated runs use an accuracy
    envelope plus transform/transport error bounds, not an unconditional rounding
    budget; use rounding bounds only for matched algebraic operations. Replace
    the scan's promise of identical numerical results by SEP 25's appropriate
    numerical comparison. Add narrowband field/envelope parity through a mirror
    and a plate, and short deliberately mismatched/aberrated launches as well as
    matched Gaussians. Name the tolerance construction for each test class before
    its reference result is known. Pass-by-pass ledgers should distinguish actual
    joule losses from approximation bounds and specify how bounds and cumulative
    boundary removals are combined over the whole cell, rather than resetting
    the effective budget on every relaunch.

17. **R17 — P2: Separate published data, author corrections and model uncertainty
    (§§4.3–4.4, 7, 13).** The public text of [10] uses peak-power wording, places
    Fig. 2 at 2 bar, and separately reports MPC3 compression at 1.2 bar in §2.4.
    The co-author's correction may be right, but the draft's statement about Table 2
    is not independently established by Fig. 2. Please publish a small input table
    distinguishing the published conditions, the explicitly attributed corrections,
    and the conditions selected for each numerical reproduction, with the author's
    confirmation linked in the review response. A discrepancy of 5 % is not a
    general consequence of printing two significant figures. Derive experimental
    envelopes from input/measurement uncertainty separately from numerical error;
    [10] itself quotes an uncertainty of $1.2/9.7$ on its argon nonlinear index.
    A small fit residual in optmat is not automatically the total experimental
    uncertainty. [12] reports transmission with and without its plates and beam
    quality, and the authors have a
    [public dataset](https://doi.org/10.57760/sciencedb.09731); inspect whether these
    constrain the missing inputs before fitting an arbitrary loss or pulse. Choose
    the comparison metric and tolerance before adjusting those inputs, and label
    a tuned reproduction separately from an independent validation. Missing
    information in [13] must not let agreement of one waist ratio alone choose
    the plate closure for Q7.

18. **R18 — P1: Keep the implementation in one PR and budget its total cost
    (§§6.5, 11).** Incorporates
    [Codex's implementation-process finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018675).
    SEP 0 requires one feature branch and one draft implementation PR, with phase
    reports accumulated there until `Final`. Make the five phases qualified
    milestones on that branch unless a maintainer explicitly grants an exception;
    splitting them solely to stay below the size budget is not the established
    process. `.github/workflows/size.yml` measures the net Python-file line growth
    against the merge base, including docstrings and other lines, whereas the
    proposal quotes code lines separately. Its 1550 code lines plus 520 docstring
    lines already exceed that PR budget before comments/blanks. Re-estimate the
    whole implementation after the response, scan and output contracts are fixed,
    identify reuse/removals, and leave any `size-ok` decision to the maintainer.

The seven linked Codex threads are incorporated above as R3, R4, R8–R10, R12 and
R18. Resolving those GitHub threads records their transfer into this AI review;
it does not mean that the proposal has already addressed the findings. Please
answer each R-number in the separate response-to-review section required by SEP 0,
with the resulting Part I changes or the reason for a different decision.

### Response to review — 2026-11-13

Drafting agent: **Claude Fable 5.2**, with the author's decisions of 2026-11-13 where a
point was the author's to settle. Each finding is answered with what changed and where.
Every recommendation is taken; R7 and R11 are taken at the level of the contract, with
the reason given, and the decision of R18 is left to the maintainers.

- **R1 — taken.** Section 4.4 keeps $n_a$ in the thin-plate lens and in the on-axis
  lens, derived with the exit rule, and says that the published constant rescales the
  exact Gaussian kick by 6 %. Section 7 keeps the independent phase-kick integration as
  the check. With Q1 resolved to the variance constant, the default lens is the exact
  one.
- **R2 — taken.** Section 4.2 requires $|A+D|/2<1$ of the round trip with the plates in
  place. Section 4.4 defines the branch (continuation in power from the linear
  eigenmode), the stability test (a Jacobian of unit determinant with
  $|\operatorname{tr}J|/2\le1$) and the refusals, requires $W>0$ at the interior minimum
  too, and separates positivity of the model from collapse of the field. No run is
  refused on a criterion of the design (section 5.4). Section 4.3 notes that
  $P_{\rm eff}<P_c$ does not bound the peak, and section 4.4 says what the peak-slice
  report is when that slice has no stationary beam. A segment without a Kerr index is
  $x=0$, without a call to `critical_power`. Section 7 adds the tests, with the fixed
  lens of the review and a case found while answering it: an asymmetric pair of plates
  whose stationary beam exists but is unstable above $x_p\approx102$ (section 4.4).
- **R3 — taken.** Section 4.1 declares a plate once, by its interval, and gives the
  reversed order, positions, gaps and interfaces of an odd pass. Sections 4.2 and 4.4
  take the round trip of two passes as the period, and section 5.1 defines `position`.
  A plate may touch a mirror with a gap of zero length, which is how the substrate
  geometry of [2] is declared. Section 7 adds the unequal, asymmetric pair.
- **R4 — taken, with the author's choice of convention.** Section 4.3 defines the beam
  quality and derives its place in the second-moment law: $M^4$ on the diffraction term,
  the Kerr term of a Gaussian intensity profile unchanged, $\sigma=1-P/(M^4P_c)$. It is a
  parameter of the design only (`beam_quality`, section 5.1). `matched_plane` refuses a
  design whose beam quality is not one, and a launch of another quality is the caller's
  plane (section 2). [10] applies its $M^2$ to the beam size and not to $\sigma$; it is
  reproduced by declaring that (sections 4.3 and 7). Checks on ideal Gaussians and
  comparisons with a measured beam quality are separate items of section 7.
- **R5 — taken.** A `Plate` carries its `responses` (section 5.1), and the executor's are
  those of the cell's medium (section 5.2). Section 4.3 defines the design's index as
  the long-pulse total, with its electronic and delayed shares, its validity for a
  finite pulse, what an override changes and the scaling that avoids the double count.
  Section 4.4 gives a polarisation factor to each share: 2/3 for the electronic one,
  $\bar a+\bar b/2$ for a delayed one, 0.718 in all for silica's tensor shape where the
  first draft applied 0.667 to the total. A circular launch on a copolarised-only shape
  is refused. The record lists the design and the run side by side for each segment
  (section 5.3), and section 7 adds the silica example and two glasses in one cell.
- **R6 — taken, as a stated limit.** Section 4.3 ("From pass to pass") makes the design
  a fixed-pulse estimate, with the energy transmissions of the actual elements, unequal
  and spectrally varying coatings included. It names the matched radius and the drift
  apart, defines the drift as the launch carried through the map of each pass, and says
  that the drift is what is compared with the propagation. No temporal surrogate is
  propagated: the propagated run exists for that.
- **R7 — the output plane taken; the transaction taken as a contract.** Sections 4.1 and
  5.2 define the planes of a pass, the $N-1$ reflections of $N$ passes, `output_plane`,
  `plane` and `pass_outputs`, and the terminal plane as distinct from the last plane of
  the last run. Section 5.3 defines the identity, the conflict, the commit point, the
  lock, the removal of an uncommitted pass only, and what is not a committed pass;
  section 7 adds the three interruption points. The checkpoint is the last plane of the
  last run of a pass, always retained, and the exit plane is recomputed from it, so the
  file counts of Q2 stand. The format of the record and the lock primitive are not fixed
  here: the implementer takes them from the scan's and reports them in phase 2, the plan
  line by line being the implementer's (SEP 0 section 7).
- **R8 — taken.** Sections 4.5 and 5.2 give `MultipassSimulation` the `noise` and `seed`
  of a free-space constructor, and say how a plane that already carries noise is passed,
  when the seed of an unseeded draw is persisted and how a resume before the first
  committed pass recovers it. Section 5.5 keeps the scan's seed rules, explicit and
  derived, and section 7 lists the five cases.
- **R9 — taken.** Section 5.5 admits the sum of the live runners, with their tables, the
  retained planes, a transport workspace and the copy of a handoff. Section 5.2 makes
  the cleanup deterministic on failure and on stop. Section 6.4 counts a handoff at every
  segment boundary and asks for two benchmark cases, and section 10 records the
  alternative of one live backend and its cost.
- **R10 — taken.** The example of section 5.5 builds the design and the launch at each
  point. The section says what a mismatch study holds (radius, curvature, energy) and
  what it does not, and `matched_plane` takes the medium to sample in (section 5.1).
  Section 7 tests both scans through the public builder, and the refusal of a launch
  kept from another pressure.
- **R11 — taken, with `Batch` refused rather than integrated.** Section 3 corrects what
  `Batch` is. Section 5.5 refuses a cell on it by name and says how the stop, the
  reporters, failures, leases, reductions and retained fields work on `Local`, `Pool`,
  `Queue` and `Slurm`. Section 6.2 gives `SimulationRunner.run` a public `stop` and
  lists what the scan worker needs beyond its type check. The import is `solprop.driver`
  (sections 5.2 and 6.2), and the physics module does not import the driver. Section 6.5
  raises the scan work from 200 to 300 lines. Cells on `Batch` need a backend change and
  go to an issue (section 11).
- **R12 — taken.** Section 5.2 materialises the centre: a save plane inside a segment, a
  split `FreeSpace` in a linear gap, the ambient side at a face of a plate. Section 4.6
  calls it the centre, offers no focus observable, and fixes the radius as the
  second-moment one, with the $1/e^2$ radius beside it.
- **R13 — taken.** Section 3 is corrected: `apply_transfer` takes an output, and nothing
  compresses a spatial field today. Section 4.6 defines the spectral transform limit,
  the compressed pulse (one transfer on the whole field, a duration from the axial power
  with its admittance) and the on-axis pulse. Section 6.2 puts the flux objective in the
  existing compressors. Section 7 says which quantities are reported against [10], and
  adds the separable and the spatially chirped beam.
- **R14 — taken.** Section 4.6 reads `on_axis_intensity`, names the quantity as the
  integral of the largest on-axis intensity in time, restricts it to the polarisations
  and the limit in which the design's index is defined, integrates within each medium,
  and gives the sampling with its error estimate, verified on the analytic integral
  (section 7).
- **R15 — taken.** Section 4.5 conditions `absolute()` on calibrated coatings and keeps
  the status through a resume; `CellMirror` takes a `calibration` (section 5.1). Section
  7 asserts the stated map across a mirror in place of continuity, restricts the
  identity handoff to the scalar radial case, and adds two reflections of a field
  without symmetry and the agreement of the scalar and vector cells, constant phase of
  the coating included.
- **R16 — taken.** Section 7 opens with the classes of testing.md and the rule that a
  tolerance is constructed first. The per-bin energy equality is restricted to a linear
  cell; comparisons across separately integrated runs are accuracy envelopes; the scan
  comparison is class R on a prescribed schedule and class T otherwise; a test of field
  against envelope and mismatched and aberrated launches are added. Section 5.3 keeps
  losses apart from bounds and accumulates bounds and boundary removals over the cell.
- **R17 — taken.** Section 4.3 replaces the paragraph on [10] by a table of what is
  published, what the author states and what is inferred. The statement that the
  1.2 bar of Table 2 is an error is withdrawn: 1.2 bar is the pressure of the 20 fs
  measurement, as section 2.4 of [10] says, and the author holds that the fluence of
  that row was in all likelihood computed at another pressure; that 2 bar reproduces it
  is an inference of this SEP. Section 7 replaces the 5 % tolerance by a recorded difference, keeps
  numerical error, input uncertainty and measurement uncertainty apart, quotes the 12 %
  of [10] on $n_2$, uses the transmissions and the beam quality that [12] states, defers
  the inspection of its dataset to phase 3 and labels tuned reproductions. For [13],
  section 4.4 no longer rests on one waist ratio: read in full on 2026-11-14, [13] gives
  its conventions and a second number, the position of the focus to inject, and both,
  like the two cells of [15], are reproduced by the moment lens and not by the on-axis
  one. That is recorded as what those designs use; Q7 stays with the propagated
  comparison. The review asks for a linked
  confirmation of the two statements attributed to the author. The author confirmed both
  on 2026-11-14, on reading this revision, and posts the confirmation on PR #794.
- **R18 — taken for the plan; the decision is the maintainers'.** Section 11 makes the
  phases milestones on one branch and one pull request. Section 6.5 re-estimates the
  whole, about 2100 lines of code, 650 of docstrings and 1900 of tests, about 3000 net
  lines of `src/solprop` against a budget of 1500, and finds no removal to offset it. Q8
  asks the maintainers for the `size-ok` label or a delivery in phases.

**The questions left to the review.** Q1: the variance constant is the default of the
design, the published one an option for reproduction, the Townes default of
`critical_power` unchanged, and the three powers are named apart in the record; no power
is described as a guarantee against collapse (sections 4.3 and 4.4). Q2: the directory
with a versioned record stays the proposal, with its files counted; the human review
weighs the count.

### Probes and sources of the revision — 2026-11-13

The design numbers of this revision were recomputed with short standalone scripts, which
are not part of the change. They first reproduce the numbers of the first draft in its
conventions: $\sigma=0.595$ and 4.26 rad for [4]; 110.9 and 170.1 mJ/cm² for Fig. 2 of
[10]; 70, 133, 156 and 145 mJ/cm² for its Table 2; 0.926 and 0.80 for [13]. For the
geometry of [15], thick plates under the segment law give −23 % and +22 %, +2 % and
+16 %, where the thin lenses of the first draft gave −24 % and +24 %, +2 % and +16 %. The
scripts then give the numbers of the revised conventions quoted in Part I.

The four probes of the review were repeated and agree: $4.472\times10^{-3}$ m⁻¹ by
radial integration of the phase kick (R1), a half-trace of −1.2917 (R2), a ratio of
0.3087 (R6) and 1.9992 (R13). For R2, the Jacobian of the round-trip moment map has unit
determinant, to five digits, at every fixed point computed, and its half-trace at zero
power is $\cos2\Psi$ (−0.7029 for the cell of [13]). In the symmetric cells tried (the
geometry of [15] and a centred plate to $x_p=10^4$, the geometry of [14] to
$4.6\times10^4$) the stationary beam of the moment model persists and stays stable. The
asymmetric pair of section 7 does not: its half-trace reaches −1 at $x_p\approx102$ and
−1.57 at 200, where a perturbation of 0.1 % iterated through the map grows about
threefold at each round trip and changes sign, while at $x_p=50$ it stays within 0.3 %
over twelve round trips. These are observations on a few cells. For R3, with that pair
at $x_p=50$, the naive order is 6 % and 5 % off on the two mirrors. For R5, the
polarisation factor was evaluated on solprop's silica bundle: $f_R=0.245$, integrals
0.750 and 0.250, a total of 0.7177.

The code paths the review names were read at 23959bc1 and agree with it:
`check_same_medium` and its tolerance, the `Batch` executor and the scan's batch
driver, `SimulationRunner.run` and its private stop path, the refusal of spatial outputs
by the compressors, `apply_transfer`, the spatial `peak_intensity` and
`on_axis_intensity`, the calibration of `ChirpedMirror` and the refusal of
`FieldPlane.absolute()`, `critical_power`, and the size workflow.

[10] was read in full, with its Tables 1 and 2 and its Fig. 2; the quotations of section
4.3 are from it. For [12], the transmissions and the beam quality were read from the
publisher's page; the landing page of its dataset shows the title of the article, and
its content was not inspected. The cost of section 6.5 is an estimate scaled from the
first one.

[4], [13], [14] and [15] were read in full on 2026-11-14, and what Part I takes from them
was checked against them.

- [4]: its constant of $P_c$, eqs. (3), (4), (6) and (8), the parameters and results of
  its section 3 (131 and 115 µm, 2.5 → 1.9 mm, 2.18 ± 0.02 mm, 117 ± 1 µm, 11.2, 5.6, 4.3
  and 3.3 rad, the 16 passes of its Fig. 2) and its remark on plates agree with Part I.
  Its mirrors reflect 99.5 %, which section 7 now states.
- [13]: its indices are the two that the first draft took ($2.74\times10^{-20}$ and
  $2.6\times10^{-23}$ m²/W). Three statements of this revision were wrong or too
  cautious and are corrected in section 4.4: its 344 µm is the waist at the centre of
  the cell, not the focus of the beam to inject, which it gives separately; its beam
  quality enters as section 4.3 puts it; and its matching leaves the air out. Its
  B-integral in air is reproduced with a peak power of $E/\tau$ (0.52 rad against the
  0.5 it quotes).
- [14]: its stability parameter is that of the round trip, not of one pass as section 7
  read it. The geometry is corrected, with the mirror spot at almost 30 times the focus
  instead of "more than 15", and the stability probe was repeated on it.
- [15]: the 30 % and 15 % and the layouts agree with Part I. The powers of its two cells
  are published, so section 4.4 now compares magnitudes and not only signs. That its
  cells stand in air is implied by the paper, not stated; it gives no index for silica,
  and the positions of the five plates of its first cell are not published.

The earlier work from which [13] and [15] take their nonlinear matching (Seidel et al.,
2022) was not read, so the lens they use is inferred from their numbers, not read.

Checks of this documentation-only revision: `git diff --check` and codespell on the two
changed files, clean; `pytest tests/test_seps.py`, 363 passed and 3 skipped;
`sphinx-build -b html -W -n --keep-going`, the command behind `poe docs`, succeeded on a
clean export of the branch with the revised pages. The tests and the build ran from an
existing environment with this branch's sources on the path, not from a fresh `uv sync`.
