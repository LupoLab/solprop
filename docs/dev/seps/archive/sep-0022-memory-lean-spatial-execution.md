# SEP 22 — Memory-lean spatial execution: statistics-only planes, construct-once runs and operator settings

| | |
|---|---|
| **SEP** | 22 |
| **Title** | Memory-lean spatial execution: statistics-only planes, construct-once runs and operator settings |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-16) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-10-16 |
| **Resolution** | Accepted 2026-10-18 by John Travers after his final review, on the revisions of 2026-10-18 recorded in Part III; Part I is frozen from here |
| **Requires** | — (related: [SEP 21](../sep-0021-absorbing-boundaries.md), the temporal collar, whose Q10 names this SEP as its transverse follow-up) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | phase 1: Opus 5.5 (AI) for John Travers, branch `feature/sep-0022-phase-1`; phase 2: Opus 5.5 (AI) for John Travers, branch `feature/sep-0022-phase-2`; phase 3: Opus 5.5 (AI) for John Travers, branch `feature/sep-0022-phase-3` |
| **Manual pages** | [output](../../../guide/results/output.md), [resuming](../../../guide/results/resuming.md), [running](../../../guide/setup/running.md), [scans](../../../guide/workflows/scans.md), [boundaries](../../../guide/numerics/boundaries.md); the four-wave-mixing delay-scan tutorial (`docs/tutorials/free_space/plot_06_four_wave_mixing_scan.py`) |

## Part I — Proposal

### 1. Abstract

Make large Cartesian free-space runs practical when their state is gigabytes and they
are repeated for hundreds of inputs. Five changes, none of which alters a default or the
result of a run that completes today: (1) **statistics-only save planes**, which store
and transfer no field; (2) **construct once, run many**, a runner that keeps one
translated problem and one backend and runs many initial fields; (3)
**separable-operator settings**: a user-set evanescent attenuation, a transverse-spectral
(k⊥) safety collar, and a propagator exponent budget that refuses a replayed step whose
loss exponent overflows; (4) a **custom spectral absorber** built on a collar passed by
value, with the transverse collar's strength decoupled from the spectral rate's ceiling,
and a spectral-only collar that compiles no temporal pass; (5) **configuration records that stay stable** when these records gain fields. Success is
judged by exact structural identities where an operation is unchanged by construction:
every new setting at its default takes today's code path, a reused backend restores a
fresh one's state, and configuration digests stay stable. Results of separate runs (fields
with and without statistics-only planes, runner results against separate runs) agree
within a rounding bound fixed before measurement on a replayed schedule and within the
case's accuracy envelope when adaptive
([SEP 25](sep-0025-rounding-level-reproducibility.md)). It is also judged by measured
memory and setup time at the gigabyte scale.

### 2. Motivation and scope

**The use case.** A thin nonlinear slab or a bulk sample is propagated on a transverse
grid of several hundred points a side and several hundred frequency bins. At 768 × 768
columns and 600 bins one state is $16\,n_xn_yn_\omega$ bytes = 5.27 GiB. Typical
experiments of this kind are a four-wave-mixing or pump–probe measurement, in which beams
cross in the sample and the diffracted signal is recorded spectrally [5], or a sweep of
the input of a bulk filament. Every saved plane is wanted as spectra and diagnostics,
never as a stored field. The same translated problem is run for hundreds of initial
fields: a delay scan of one beam, or a sweep of the input alone. The spectral absorber,
the k⊥ absorber and the evanescent attenuation are set by the physics of the
measurement, not by a default collar. Examples are a spectral band the instrument cannot
record, and a thin sample whose non-propagating components must decay as they
physically do.

**What it costs today.**

- *Each saved plane makes six full-volume passes on the CPU backends.* `StepDriver.observe`
  (`solprop/backends/driver.py`) hands over the state through `_observed_field`. That is
  the identity on `numpy_ref` and `numba_cpu`, and a device-to-host copy (`field_to_host`)
  only on Kokkos. `Observation.__post_init__` (`backends/protocol.py`) then scans the
  field for finiteness. `run_problem` (`driver/run.py`) builds a `SavedPlane` and then
  replaces its statistics; `SavedPlane.__post_init__` copies the field on each
  construction, so that is two copies. The handler converts to the optics convention and
  stores the result. On a Cartesian grid the collar diagnostic also inverse-transforms
  the whole volume (`backends/spatial.py`, `_free_physical`'s `numpy.fft.ifft2`). That
  transient is outside the buffer ledger and the memory admission: it is the roadmap's
  known issue *The collar diagnostic on Cartesian saved planes*. With a noise model,
  `_noise_phase` adds a float64 volume per plane.
- *Storage scales with planes × volume.* Since
  [SEP 25](sep-0025-rounding-level-reproducibility.md) `MemoryOutput` allocates
  its field block for the planes the run reserved, the entrance and every save plane,
  and memory admission charges that store at its planned capacity as the group `output`
  (`driver/memory.py`): 17 planes of the size above are 90 GiB, charged beside the
  march's own 79 GiB. `HDF5Output` preallocates its datasets at the planned count and
  writes one volume per plane: 17 planes are 90 GiB on disk per input.
- *Each input re-validates, re-allocates and re-plans.* `Simulation.with_input`
  (`driver/simulation.py`) shares the translated tables by identity. It still scans the
  new field for finiteness, then runs `KernelProblem.validate()`, which scans it and every
  table again. `Simulation.run` admits memory, constructs a backend (about 15 volumes and
  the FFT plans) and closes it on every call. An array input is also written in full into
  the run's configuration record (`output/schema.py`, `_serialise_numpy` via `tolist`),
  and so into the hash and the file's metadata. At 353 million complex samples that is
  tens of gigabytes of JSON for each input.
- *The operator cannot be set as the physics requires.* The evanescent cap is fixed at
  `solprop.spatial.EVANESCENT_RATE` = 200 m⁻¹, which over a 40 µm slab attenuates a
  non-propagating component by 0.8 %. A custom spectral rate needs a hand-built
  `Absorber`, whose one length ties the rate's ceiling to the transverse collar's
  strength (§4.6). There is no k⊥ collar at all.

**In scope.** The five items of §1, on the Cartesian kind (and the line, `LineGrid`). They
also apply to the radial and planar kinds wherever an item is not specific to periodic
grids (§5). This is the follow-up SEP that [SEP 21](../sep-0021-absorbing-boundaries.md)'s
Q10 names for the transverse collars, in two respects: the Cartesian routes' k⊥ collar,
and decoupling the transverse collar's length from the spectral rate's ceiling. The
real-space transverse collar's geometry and strength stay as SEP 21's phase-0 finding R4
leaves them; this SEP changes neither.

**Out of scope.**

- **The temporal collar**, owned by SEP 21: its geometry, strength, integration and
  diagnostics, `absorber=False` and `absorber_scheme=`. Compiling a zero temporal rate to
  no pass, and the keyword that builds a spectral-only collar, are this SEP's: they were
  handed to SEP 21 on 2026-10-16 and taken back on 2026-10-18, when SEP 21 was deferred,
  because they decide the cost of every spectral-only run (§3, §5.4).
- **New catalogue statistics.** [SEP 23](sep-0023-far-field-aperture-spectra.md) proposes
  far-field and angular-region spectra.
- **A lower-storage integrator**, which would shrink the march itself
  ([SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md)).
- **Lockstep batches.** `Batch` requires fibre geometry (`driver/batch.py`), so
  `observe_members` keeps observing fields (Q9).
- **A native collar-fraction reduction** on Kokkos (Q2).

### 3. Context

**The outer loop and its outputs.** `run_problem` is the Python loop around the backend
([architecture §5](../../architecture.md), ADR-012): propagate, observe, collect statistics,
save. Backends compute statistics from their live state without a copy; Kokkos computes
them from its native state, and only `observe` moves the field. Scans already have the
retention vocabulary, `run_scan(fields="none"|"planes"|"all", field_planes=…)`
(`driver/scan.py`; `_validate_options` in `driver/_scan_run.py`). The worker's
`simulation.run(capture)` still materialises every field before `_Capture.save` discards
it. A custom reduction spools the whole run, fields included, to a temporary
`HDF5Output` ([scans](../../../guide/workflows/scans.md#retention-reductions-and-reading)).

**Reuse of a translated problem.** The scan guide recommends one base simulation per
worker, with each point derived by `with_input`
([reuse a translated simulation](../../../guide/workflows/scans.md#reuse-a-translated-simulation)).
ADR-014 describes `Simulation` as re-runnable, with `reset()` and `propagate_to(z)`.
Neither method exists, and every `run` constructs a backend.
`Backend.set_initial(field, noise, *, z=0.0)` (`backends/protocol.py`;
`StepDriver.set_initial`) already resets almost the whole march, including each
backend's Adams history and recycled stages. §4.7 lists what it misses.

**The separable operator and the collars.** Cartesian propagation keeps constant
material coefficients as a `SeparableOperator` leaf (`translate/spatial.py`;
[kernel contract §3](../../kernel-contract.md#3-linear-operator);
[Cartesian free space](../../../guide/free-space/cartesian.md)). It is evaluated per element
by `separable_values` on the host and by `bodies.longitudinal_rate` in `numba_cpu`. In
solprop-kokkos it is `SpatialCoefficients::value` (`src/general.cpp`). The boundaries are
rates (ADR-007; [absorbing boundaries](../../../guide/numerics/boundaries.md)). The spectral
rate is folded into $D$. The spatial collar is the in-stage term
$\mathcal F_\perp\operatorname{diag}(-\alpha_r/2)\mathcal F_\perp^{-1}E$ with
$\alpha_r = 12.6\,(1-W)/\ell$
([spatial collar](../../../guide/free-space/model.md#spatial-collar-and-statistics)). Several
places read $\ell$ from `Absorber.reference_length` for an in-stage collar's strength:
`translate/free.py`, `translate/radial.py`, `translate/planar.py`, the `warn_walkoff` call
and the TDSE collar refinement (`_refined_tdse_collar`), both in `driver/simulation.py`.

**SEP 21.** SEP 21 owns the temporal collar. Its phase 2 may parameterise the collar's
strength by a protected walk-off (`absorber(grid, *, …, walkoff=None, profile=…)`),
moves the collar outside the requested window (its Q4) and adds the keywords and the
diagnostics, with an output schema bump and, if its committed-endpoint hook changes the
protected driver surface, a kernel bump; its phase 3 (the escaped record) and phase 4
(the implicit collar) bump the kernel schema. It leaves out the spectral collar and the
transverse collars, and its Q16 records this SEP's hand-offs. SEP 21's cumulative
`absorbed_energy` statistics (J) cover the temporal and spectral collars. SEP 23's
`absorbed_power` reports the spectral collar's rate (J/m) and the power of this SEP's
k⊥ rate.

**The zero-rate pass and the spectral-only collar (phase 1).** *A spectral-only
absorber (zero temporal rates) compiles no temporal pass under either scheme, and a user
builds one by keyword while the spectral rate stays.* Today `translate/masks.py`'s
`temporal_rates` compiles any absorber's temporal rates, zero or not. The free kinds'
compiled stages then evaluate the transform pipeline a second time per stage for the
absorber term under the in-stage scheme, or apply the zero rate once per step with a
transverse transform pair and without first-same-as-last recycling under the split. The
cost was measured on 2026-10-18 on a Cartesian envelope grid of 192 × 192 × 300 (Kerr
only, no spatial collar, replayed 0.5 µm steps, 20 steps, four threads, the same process
interleaved over two rounds, wall time per step including about a second of
construction): a spectral-only `Absorber` as translated today, against the same collar
handed to the kernels as no temporal rates.

| backend, scheme | today (ms per step) | no temporal pass | today ÷ without |
|---|---:|---:|---:|
| `numba_cpu`, `split_step` | 1692, 1419 | 1102, 1127 | 1.26–1.54 |
| `numba_cpu`, `nonlinear` (the default) | 2430, 2405 | 1077, 1189 | 2.02–2.26 |
| Kokkos CPU, `split_step` | 1341, 1382 | 1161, 1186 | 1.16–1.17 |
| Kokkos CPU, `nonlinear` | 2040, 2107 | 1272, 1290 | 1.60–1.63 |

Under the in-stage scheme the two fields are identical (exact zeros were added); under
the split they differ by $3\times10^{-15}$ of the peak, the removed transform round
trip. Both items were handed to SEP 21 on 2026-10-16 and taken back on 2026-10-18, when
SEP 21 was deferred: they decide the cost of every spectral-only run, including under
[SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md)'s march, which would
otherwise pay the second evaluation in every pass. Phase 1 delivers both (§5.4); SEP 21
adopts the keyword when it lands (its Q16).

**Roadmap.** This SEP closes *The collar diagnostic on Cartesian saved planes*.

### 4. Theory

Conventions are those of [conventions.md](../../conventions.md): the internal FFT
convention, and $D = -i(\beta-\beta_0-\beta_1(\omega-\omega_0)) - \alpha/2$. Every rate
$\alpha$ is a power coefficient, so the field falls as $e^{-\alpha z/2}$.

#### 4.1 The separable operator as it stands

For a column $c$ with transverse square $k_\perp^2 = k_x^2+k_y^2$ and a frequency
$\omega$, let $q = k^2(\omega)-k_\perp^2$, formed from the high parts and residuals in
that order. Let $s=\sqrt q$ be the principal root, the exact longitudinal wavenumber of
the unidirectional propagation equation [3, 4]. Then

$$
a(c,\omega) =
\begin{cases}
\operatorname{Im}s, & \operatorname{Re}q>0,\\
\min\bigl(\lvert\operatorname{Im}s\rvert,\ r_{\mathrm{ev}}\bigr), & \operatorname{Re}q\le0,
\end{cases}
\qquad
D = \bigl[-a - i(\operatorname{Re}s - f(\omega))\bigr]\chi(\omega) - \tfrac12\alpha_\omega(\omega),
$$

with $f$ the frame subtraction, $\chi$ the active mask, $r_{\mathrm{ev}}$ the
evanescent cap and $\alpha_\omega$ the spectral rate. The medium is constant, so the
cumulative operator is $\Phi(z)=Dz$. Every backend advances by $e^{D\Delta z}$ and
returns to the interaction picture with $e^{-D\Delta z}$. The noise phase is
$\operatorname{Im}\Phi$ alone (ADR-008).

#### 4.2 Diagonal rates are exact under the cumulative flow

A real, nonnegative rate $\kappa(c,\omega)$ added as $D \to D-\kappa/2$ commutes with the
rest of $D$. The linear flow over any sub-interval gains the factor
$e^{-\kappa\Delta z/2}$ exactly, so the attenuation over a distance $L$ is
$e^{-\kappa L/2}$ however many steps are taken. That is ADR-007's argument for the
spectral collar, and it holds for any rate diagonal in $(c,\omega)$. Such a rate costs
one load and one add per element and needs no transform. It is unconditionally stable
and leaves the noise phase unchanged. The spatial collar is diagonal in $(x,y)$ instead,
so it rides the source as an explicit term. An explicit method with real stability
interval $R$ must keep $h\max\alpha_r/2 \le R$. `boundaries.stable_step` gives
$2R\ell/12.6 \approx 0.52\,\ell$ for DOPRI5 ($R = 3.3$), and far less for the Adams
method's higher orders.

#### 4.3 Aliasing: grid sizing, and the k⊥ collar as a safety ramp

On a periodic grid of spacing $\Delta x$ the sampled transverse wavenumbers satisfy
$\lvert k_x\rvert \le k_{x,\max} \approx \pi/\Delta x$. A **polynomial cubic source**
(Kerr, and Raman, which is delayed in time but local in $(x,y)$) is a product of three
field factors evaluated pointwise in $(x,y)$, so its transverse spectrum is a double
convolution. The envelope (THG-free) Kerr product $EEE^*$ of components at
$k_a,k_b,k_c$ lands at $k_a+k_b-k_c$; a field grid's third-harmonic terms also give
$k_a+k_b+k_c$. Content within $\lvert k_x\rvert\le K$ therefore produces products within
$3K$. The discrete product is computed on the grid, so a product beyond $k_{x,\max}$ is
never generated and then removed: it is generated **already folded**, at
$k - 2k_{x,\max}$. For instance, envelope Kerr acting on a beam at $+0.95\,k_{x,\max}$ and
one at $0$ produces $2(0.95)-0 = 1.9\,k_{x,\max}$, which appears at $-0.1\,k_{x,\max}$.
Beams at $+0.4$ and $+0.6\,k_{x,\max}$ produce $0.2$ and $0.8\,k_{x,\max}$, both in band.
Folded products miss the energetic band $[-K,K]$ only while $3K-2k_{x,\max} < -K$, that is
$K < k_{x,\max}/2$. This is the one-half rule, the cubic analogue of Orszag's two-thirds
rule for quadratic products [1].

**Grid sizing is the anti-aliasing.** The grid must be fine enough that the content
carrying energy satisfies the one-half rule, and refinement checks it, as the
crossing-beam lesson and the Cartesian page do. The rule holds for polynomial cubic
terms. Plasma and ionisation sources are not polynomial in the field: a rate with a
threshold or an exponential in $\lvert E\rvert$ has unbounded transverse support, no
finite multiple of $K$ bounds its products, and only refinement judges the grid.

**The k⊥ collar is a safety ramp**, like the temporal collar. For each axis let
$\rho_x = \lvert k_x\rvert/k_{x,\max}$, with $k_{x,\max}$ the largest $\lvert k_x\rvert$
the grid holds, likewise $\rho_y$, and let $\rho = \max(\rho_x,\rho_y)$. With a fraction
$f\in(0,1)$ and a strength $S \ge 0$ in m⁻¹,

$$
\alpha_\perp(c) = S\,P(\rho),\qquad
P(\rho)=
\begin{cases}
0, & \rho\le 1-f,\\
\tfrac12\bigl[1-\cos\bigl(\pi(\rho-1+f)/f\bigr)\bigr], & \rho>1-f.
\end{cases}
$$

Content at $\rho$ decays in field at $\alpha_\perp/2 = S\,P(\rho)/2$; that is $S/2$ only at
$\rho = 1$. The collar does three things.

- It damps content that reaches the outer band, from cascades or from content drifting
  towards the edge. The products of that content, which would fold into the band, are
  weakened with it: a product quadratic in the damped component loses twice its field
  exponent.
- It keeps rounding and noise from accumulating at the Nyquist edge.
- It **cannot** remove an alias already generated by in-band content. The fold happens
  inside the product's evaluation, and the folded product lands where $P = 0$.

The max-norm makes the protected region the rectangle the grid defines, since aliasing
in $x$ is decided by $k_x$ alone (Q5). The default $f = 0.1$ is deliberately narrow.
On a grid sized by the one-half rule the outer band is empty of energetic content. Products
between $K$ and $k_{x,\max}$ are physical and must not be damped: a four-wave-mixing signal
leaving at a larger angle, say. A ten per cent ramp, the width of the spatial grids'
default `window_factor`, touches only content that should not be there. A wider ramp
would damp physical products and still not remove in-band aliases. The profile is exactly
zero inside and rises with a continuous slope, which keeps its real-space kernel short. A
strength with $S L_{\mathrm{int}}/2 \ge \ln 10^3$ removes 60 dB of field at the edge over
an interaction length $L_{\mathrm{int}}$; its upper bound is §4.5's budget.

**Why the real-space collar does not do this.** $\alpha_r(x,y)$ removes light that has
*walked* to the edge of the box. A component near the Nyquist edge is a plane wave spread
over the whole box. The real-space collar touches it only over the collar's share of the
area, at the same rate as low-angle light there, and selects nothing by angle. Conversely
the k⊥ collar does nothing for a beam that walks out of the box. The two are
complementary and configured independently.

#### 4.4 The evanescent model and its physical rate

For $\operatorname{Re}q \le 0$ a column does not propagate. Physically it decays as
$e^{-\lvert\operatorname{Im}s\rvert z}$, with
$\lvert\operatorname{Im}s\rvert \approx (k_\perp^2-\operatorname{Re}k^2)^{1/2}$, of order
$10^6$–$10^7$ m⁻¹ on a sub-wavelength grid [4]. The declared model caps this at
$r_{\mathrm{ev}}$, a numerical choice that keeps the operator bounded
([exact longitudinal operator](../../../guide/free-space/model.md#exact-longitudinal-operator)).
For a thin sample the 200 m⁻¹ cap leaves non-propagating content effectively undamped.

The largest physical rate is cheap to find. For $\operatorname{Im}z\ge0$,
$\operatorname{Im}\sqrt z = \sqrt{(\lvert z\rvert-\operatorname{Re}z)/2}$, which increases
as $\operatorname{Re}z$ decreases at fixed $\operatorname{Im}z$. At each frequency
$\operatorname{Im}\sqrt{k^2-k_\perp^2}$ is therefore nondecreasing in $k_\perp^2$, and
nonincreasing for gain, $\operatorname{Im}k^2<0$. The extreme over all columns lies at
the grid's largest $k_\perp^2$, and over propagating columns at the largest $k_\perp^2$
below $\operatorname{Re}k^2$. Both cost $O(n_\omega\log n_{\mathrm{modes}})$ over the sorted
columns that `translate/free.py`'s `_warn_grazing` already searches. Take glass of
$n \approx 1.45$ on a grid of $\Delta x = 0.4$ µm: at 1.1 µm the grid corner reaches
$\lvert\operatorname{Im}s\rvert \approx 7.4\times10^{6}$ m⁻¹.

#### 4.5 The propagator exponent budget

**Where the factor grows.** `numpy_ref` evaluates the inverse stage propagator as
`np.exp(-exponent)`. `backends/bodies.py`'s `propagator_pair` forms `1/exp(real)`, and
the Kokkos backend its `exponential` of the negated exponent. For a decaying column the
factor $e^{\lambda_-\Delta z}$, with $\lambda_- = \max(-\operatorname{Re}D)$, is infinite
on every backend once $\lambda_-\Delta z > \ln(\mathrm{DBL\_MAX}) \approx 709.78$ [6]. A
stage then yields $\infty\cdot0$ or $\infty$, which the next transverse transform spreads
to every column. For loss this factor never spans more than one step:

- the Runge–Kutta stages use offsets $c_ih \le h$;
- the interaction-picture Adams history is transported by the forward, decaying
  propagator (`backends/multistep.py`, `store[slot] *= propagator`);
- the exponential formulation carries samples by $e^{(1-\theta_j)x}$ with
  $\theta_j \le 0$, also decaying;
- its dense inverse factor $e^{(\xi-\theta_j)x}$ spans one step and is masked where
  $\operatorname{Re}x < -1$.

For gain, $\lambda_+ = \max\operatorname{Re}D > 0$, the forward factor grows. The
exponential carry spans up to $q+1$ steps, with $q$ the method's largest order. Write
$\sigma = 1$ for `RungeKutta()` and $\sigma = q+1$ for `Multistep()`.

**What is refused, and what is not.** An adaptive march already rejects a trial with a
non-finite error norm (`PIController.accepts`, `stepper/controller.py`) and shrinks by
`min_factor`. It therefore finds its own limit at the cost of rejected trials, as it does
for the in-stage collar's stability limit. Adaptive runs are never refused. A
lossy medium with $\operatorname{Im}n = 10^{-3}$ at 800 nm and `dz_max` = 0.1 m gives
$\lambda_-h \approx 785$ and runs as today. The refusal applies where the step cannot
shrink. Those are a replayed schedule (its largest step) and the floor `dz_min`, as
`KernelProblem._check_collar_stability` treats the collar's limit. The condition is

$$
h\,\max(\lambda_-,\ \sigma\lambda_+) > \ln(\mathrm{DBL\_MAX}),
$$

with sufficient bounds from §4.4's search:

$$
\lambda_- \le \max\bigl(a_{\mathrm{prop}}^{+},\ \min(a_{\mathrm{ev}}, r_{\mathrm{ev}})\bigr) + \tfrac12\max\alpha_\omega + \tfrac12\max\alpha_\perp,
\qquad \lambda_+ \le \max\bigl(0,\ -\min a_{\mathrm{prop}}\bigr).
$$

Here $a_{\mathrm{prop}}^{+}$ is the largest propagating attenuation and $a_{\mathrm{ev}}$
the largest evanescent one. A run past this condition forms an infinite factor on its
first such step, so it cannot complete today; refusing it refuses no run that completes.
Between $B = 0.9\ln(\mathrm{DBL\_MAX}) \approx 638.8$ and the hard limit, a replayed run
completes only while field and drive magnitudes stay below $e^{709.78-\lambda h}$. There
translation warns, naming the admissible step, and does not refuse. The present spectral
ceiling keeps the spectral term inside $B$ while $h < 21\,\ell$, which every default
satisfies ($\ell = L/20$, $h \le L$), because $\alpha_\omega \le 60/\ell$ gives
$\alpha_\omega h/2 \le 30h/\ell$. §4.4's physical evanescent rate admits replayed steps up
to $709.78/7.4\times10^6 \approx 96$ µm, and warns above 86 µm.

#### 4.6 Decoupling the transverse collar's length

Today one length, $\ell$ = `reference_length`, sets three things:

- the spectral collar's strength;
- the ceiling $60/\ell$ that `Absorber.__post_init__` validates every rate against;
- the in-stage collars' strength, temporal ($12.6(1-W(t))/\ell$) and spatial
  ($12.6(1-W(x,y))/\ell$).

Take a 1 mm sample whose measurement calls for a user spectral power rate of $10^7$ m⁻¹.
The default $\ell = 50$ µm caps rates at $1.2\times10^6$ m⁻¹, so $\ell \le 6$ µm is
needed. The spatial collar's peak would then rise from $2.5\times10^5$ to
$2.1\times10^6$ m⁻¹, and its DOPRI5 stable step fall from 26 µm to 3.1 µm. A replayed
schedule at the step the physics needs would be refused, and an adaptive run would pay in
steps. The spectral rate is exact and free; the in-stage collars are explicit (§4.2).
Their strengths therefore need separate lengths. `reference_length` keeps the spectral
rate and its ceiling. A new `collar_length` is whatever sets the temporal collar's
strength (`boundary_N` or `boundary_length` today, possibly SEP 21's walk-off), and the
spatial collar follows it. That makes SEP 21's phase-0 question R4, whether its geometry
findings transfer to the spatial collars, a question about this one length: should R4
find that the spatial collar must not follow a walk-off-derived temporal strength,
`collar_length` splits into a temporal and a spatial length at that point, each
defaulting to `reference_length` as before.

#### 4.7 Why neither change alters a result

**Statistics-only planes.** `observe` reads the state and never writes it ("it never
touches the state between planes", `driver/run.py`). Statistics are computed from the
live state either way, so skipping the field changes no operation of the march. An
interpolated plane positions the state for its statistics exactly as before.

**A reused backend.** A run on a reused backend repeats a fresh backend's run, to the
rounding that separates any two runs, if `set_initial` restores everything the march
reads. The restored state is exact. Four pieces are not restored today:

- `StepDriver._dense_ahead`, a dense observation's look-ahead. A failed dense run can
  leave it set, and the next `propagate_to` would restore that stale state over the new
  one.
- The quadrature alarm (`_alarm`, `_alarm_reported` in all three backends). It is cleared
  at the start of each march, but the entrance plane's `quadrature_error` is read before
  any march, so it would report the previous run's value.
- The Adams march's cumulative call ledger.
- Its `_last_rejected` flag (`backends/multistep.py`).

A resume restores the ledger after `set_initial` (`Simulation._prime`), so resetting it
there is compatible. FFT plans are reused with identical results. Separate constructions
may choose different algorithms under FFTW's measuring planner
([tuning](../../../guide/performance/tuning.md)), and then agree only to rounding, which the
runner's rounding bound covers.

### 5. Proposed interface

#### 5.1 Statistics-only planes

- `SimulationSettings.fields: Literal["none", "planes", "all"] = "all"` and
  `field_planes: Sequence[int] = ()`, with `run_scan`'s vocabulary and one shared
  validator. Indices count saved planes, the entrance as 0, negative from the end.
  `run_problem` gains the same keywords. Inside a scan `run_scan`'s arguments decide,
  and a builder whose settings set `fields` is refused by name.
- `SavedPlane.field` becomes `… | None`. A run is **homogeneous**: it keeps fields on
  every plane, on the named planes, or on none. `PlaneLayout` records the rule at the
  first save, and `Output._check_layout` refuses a plane that breaks it.
  `fields="planes"` beside `step_planes > 0` is refused, since the plane count is then
  unknown in advance.
- For `"none"`, `MemoryOutput` allocates no field block, `HDF5Output` writes no
  `field` or `noise_phase` dataset, and the admission's `output` group charges the
  statistics alone. For `"planes"`, `MemoryOutput` allocates `len(field_planes)` planes,
  `HDF5Output` adds a `field_planes` index beside the datasets, and the group charges
  those planes: `PlaneStore.held_bytes` takes the retention rule. Both handlers count
  planes by SEP 25's `n_saved` attribute, written after each plane; the fallback for a
  file without it, today `field`'s extent (`output/hdf5.py`), becomes `z`'s.
- Accessors that need a field (`output["field"]`, `fields()`, processing and plotting)
  raise `OutputError` naming the run's `fields`; `planes()` yields `field=None`.
- **Resume** continues only from a plane that carries a field. A file whose last
  regular plane is field-free is refused by name. A terminal checkpoint carries the field
  even in a field-free run (Q1). That costs one transfer when the run stops, and keeps a
  run stopped by a signal or a budget resumable (ADR-047;
  [resuming](../../../guide/results/resuming.md)).
- `fields` is outside the configuration hash, like `resume_history`: it changes what is
  kept, not what is computed.
- **Seam (phase 2).** `StepDriver.observe(*, field: bool = True)`. With `field=False` it
  calls neither `_observed_field` nor `_noise_phase`, and `Observation` carries
  `field = None` and `noise_phase = None`; its shape check applies only when a field is
  present. `run_problem` asks for the field exactly on the planes it keeps. The scan
  worker also asks on each built-in reduction's target plane, and on every plane while a
  custom reduction spools the run (Q3). The Cartesian collar diagnostic runs only where a
  field is observed.

#### 5.2 Construct once, run many

```text
runner = simulation.runner(memory_budget="60 GiB")   # translate, admit, construct once
with runner:
    for delay in delays:
        out = runner.run(build_input(delay),                 # array or SpatialPulse
                         output=None,                        # MemoryOutput, a path, or a handler
                         configuration={"delay": delay},     # recorded in this run's hash
                         noise="keep",
                         progress="none")
        spectra.append(out["stats/on_axis_spectrum"])
# runner.close() on exit; a closed runner refuses run()
```

- `Simulation.runner(*, memory_budget=None) -> SimulationRunner` translates or reuses the
  cached translation, admits memory once and constructs one backend.
- `SimulationRunner.run(initial, *, output=None, configuration=None, noise="keep",
  progress=None, progress_period=…) -> Output` derives the input with `with_input`, calls
  `backend.set_initial` and runs `run_problem` with that input's own metadata. Each call
  returns its own handler, and the backend persists.
- The runner checks that the derived problem is the constructed one but for its input.
  The leaves `with_input` shares (`grids`, `linear`, `transform`, `responses`,
  `stats_spec`, `tdse`, `current_numerator`) are checked by identity. `stepper` and
  `flags`, which `with_input` rebuilds, are compared field by field, with the save planes
  compared by `numpy.array_equal`.
- `Simulation.with_input(…, configuration=None)` merges caller keys into the derived
  run's configuration record. Without `initial_z` it validates only the replaced
  `Initial` (shape, dtype and one finiteness pass) and the shared leaves by identity,
  instead of re-running `KernelProblem.validate()`. With `initial_z`, which changes the
  save planes, it keeps `validate()`'s stepper checks: the launch before the first plane,
  and the schedule containing every plane.
- An array `initial` above a size threshold is recorded by a content digest (SHA-256 of
  its bytes, with dtype and shape) rather than inline (§6.3, Q4).
- **Refused by name:**
  - a different `initial_z`, because the save planes and the schedule cursor belong to
    the constructed problem;
  - a different noise model, since `StepDriver._noise_field` returns `None` whenever the
    constructed model is `"none"` and would silently drop a replacement;
  - TDSE responses, whose atomic binding has its own lifecycle;
  - concurrent use.

  A run that raises leaves the runner open, since the next `set_initial` resets
  everything. A backend that closed itself closes the runner.

#### 5.3 Separable-operator settings (phase 2)

- `SimulationSettings.evanescent_rate: float | None = None`. `None` keeps
  `spatial.EVANESCENT_RATE`; a value must be positive and finite. It is threaded through
  `assemble` to `SpatialDispersion(medium, grid, frame, evanescent_rate=…)`, which already
  accepts it. A non-spatial run is refused. A `SpatialPulse` launched with `propz`
  (`pulses.py`) keeps the default 200 m⁻¹ for its pre-propagation. A physical rate would
  overflow a backward launch: at $7.4\times10^6$ m⁻¹ a 100 µm backward `propz` is
  $e^{740}$, which `SpatialPulse.build` refuses.
- `spatial.physical_evanescent_rate(space, medium, grid, *, z=0.0) -> float` returns the
  largest $\lvert\operatorname{Im}k_z\rvert$ over evanescent pairs (§4.4). A grid with no
  evanescent pair returns the default, since the cap is then inert.
- `boundaries.TransverseCollar(strength, fraction=0.1)` is a frozen record, with the
  strength $S$ in m⁻¹ and $f \in (0,1)$. It goes in
  `SimulationSettings.transverse_absorber: TransverseCollar | None = None`.
  `boundaries.transverse_collar_rate(space, collar)` evaluates §4.3's $\alpha_\perp$ at
  translation, a nonnegative `(n_x*n_y,)` power rate in column order; on a line it acts
  on $k_x$ alone. It is refused on the radial kind (not periodic), on modal runs, and on
  the planar kind in this SEP (Q5). The manual presents it as a safety ramp, not as
  anti-aliasing (§4.3).
- **The exponent budget** is checked at translation for replayed schedules and the
  `dz_min` floor (§4.5). The refusal names the binding term, its value and the admissible
  step and rate. For example: "replayed step 1.0e-4 m with evanescent_rate 7.4e6 1/m forms
  a propagator factor exp(740) that overflows; take steps of at most 9.6e-5 m or
  evanescent_rate at most 7.1e6 1/m".

#### 5.4 Custom spectral absorber and the collar length (phase 1)

- `Absorber.collar_length: float | None = None` is the length $\ell_c$ that sets the
  temporal collar's strength, so its largest rate is $12.6/\ell_c$, however that is
  parameterised. `None` means `reference_length`, as for every absorber built today.
  Every reader of an in-stage collar's strength listed in §3 takes $\ell_c$ instead of
  `reference_length`: the spatial collar $12.6(1-W)/\ell_c$ in the three transforms, the
  `warn_walkoff` call, and the TDSE refinement, which rebuilds each rate from its own
  length. `Absorber.__post_init__` validates `spectral_rate` against
  $60/$`reference_length` and the temporal rates against $60/\ell_c$, which is today's
  rule when `collar_length` is `None`. Whichever of SEP 21's phase 2 and this SEP's
  phase 1 lands second adapts the field to the other's parameterisation, so that
  `absorber(…, walkoff=…)` fills it.
- `absorber(grid, *, …, temporal: bool = True)`. With `temporal=False` the collar's
  temporal rates are zero arrays on both time axes, and the band mask and the spectral
  rate are as before; the record says so. The keyword interfaces keep `absorber=False`
  (SEP 21) for both soft collars off. Translation (`translate/masks.py`,
  `temporal_rates`) returns no temporal rates for a collar whose rates are all zero,
  however it was built, so the kernel problem carries no temporal absorber under either
  scheme: no in-stage term, no split pass, and the first-same-as-last stage recycled.
  `absorber_scheme` is then recorded and applies to nothing, and its refusal of
  interpolated planes does not fire. The grid is unchanged: a collar that absorbs nothing
  adds no width, which SEP 21's placement of the collar outside the requested window
  must keep (its Q16).
- `boundaries.custom_absorber(grid, spectral_rate, *, collar: Absorber | None = None)
  -> Absorber` takes the collar by value: `None` means `absorber(grid)`, and any collar
  SEP 21's keywords build is handed in as it is, among them a spectral-only collar from
  `temporal=False`. The function adds the user's power rate
  to the collar's spectral rate. That rate may be a scalar, an `(n_omega,)` array or a
  callable of $\omega$ in rad/s; it is zeroed outside the band mask, and must be finite
  and nonnegative. The temporal rates and the band mask are the collar's. `collar_length`
  is the collar's effective $\ell_c$. `reference_length` stays the collar's when the
  summed spectral rate fits under its ceiling; otherwise it becomes $60/\max$ of the sum,
  the longest admitting length. The in-stage collars therefore never move (§4.6).
  `collar_length` is recorded only where it differs from `reference_length`, so a zero
  user rate returns a record equal to the collar, field for field.

#### 5.5 Configuration records

A field added to a recorded dataclass after its record existed declares
`metadata={"omit_default": True}`, and `output/schema.py`'s `_serialise_dataclass` leaves
it out while it holds its default. This generalises that function's two special cases,
`TDSEGasResponse.source_filter == "none"` and `PlasmaResponse.integration ==
"trapezoid"`, and `metadata_for`'s rule for `transverse_measure` and
`plasma_integration`. It covers `Absorber.collar_length`, later `StatsSpec` fields, and
this SEP's configuration keys. Those are `evanescent_rate`, and `transverse_absorber`,
recorded as its `TransverseCollar` (two numbers), never as the `(n_x*n_y,)` array.

### 6. Implementation strategies

#### 6.1 Memory arithmetic

One volume is $F = 16\,n_xn_yn_{\mathrm{pol}}n_\omega$ bytes: 5.27 GiB at 768 × 768 × 600.
The Runge–Kutta march's `memory_ledger` was measured while drafting on a 64 × 64 × 256
Cartesian Kerr problem:

- `numpy_ref` holds 13 state-sized buffers: seven stages, two accumulators, two
  propagators, the state and the transverse scratch;
- `numba_cpu` holds 16;
- the Kokkos CPU build holds 15.

`estimate_memory` and the manual's "about 14 copies"
([choosing a stepper](../../../guide/numerics/choosing-a-stepper.md#large-problems-memory))
are the reference's count; the compiled backends' extras are declared not counted.

| item | volumes | at 768 × 768 × 600 |
|---|---:|---:|
| Runge–Kutta march (Kokkos / `numba_cpu`) | 15 / 16 | 79 / 84 GiB |
| per saved plane today: finiteness scan, collar `ifft2`, two record copies, optics conversion, store; plus a device copy on Kokkos | 6 passes (7), up to 3 alive | 32 (37) GiB moved |
| storage per plane | 1 | 5.27 GiB |
| `MemoryOutput` at its planned capacity, charged by admission (SEP 25) | 17 | 90 GiB |
| statistics-only plane | 0 | statistics only |

Phase 1 removes the two record copies, the conversion and the storage on every backend.
Phase 2's `observe(field=False)` removes the device copy on Kokkos, and on the CPU
backends the finiteness scan, the collar diagnostic and the noise-phase volume. Planes
interpolated inside a step (`interpolate_planes`) are outside these figures. The dense
observation holds about 13 host volumes per crossing step: the start state, seven stages,
five extension coefficients and `_dense_ahead` (`StepDriver._observe_inside`). Field-free
planes do not remove them, so large grids land on their planes.

Backend construction allocates and first-touches about 15 volumes per input (79 GiB;
roughly 8 s at 10 GB/s, an estimate to be measured) and plans the FFTs. Planning a large
problem can take seconds ([tuning](../../../guide/performance/tuning.md)). Over hundreds of
inputs that is minutes to hours of setup.

**The node budget, for context.** `SimulationSettings.node_budget` bounds the source's
chunk buffers and is unchanged here. It was measured while drafting, before SEP 25's
stage-kernel work, on §3's 192 × 192 × 300 grid:

| node budget | 64 MiB | 256 MiB | 1 GiB | one chunk (4 GiB) |
|---|---:|---:|---:|---:|
| Kokkos CPU, 18 threads (ms per step) | 742 | 597 | 559 | 630 |
| `numba_cpu` (ms per step) | 885 | 697 | 700 | 827 |

The one-chunk Kokkos figure is within noise of the 1 GiB one. The budget is a modest CPU
lever, worth up to about a quarter of a step. A device-sized default is the roadmap item
*A device-sized node chunk*.

#### 6.2 Layers and the seam

Items 1, 2, 4 and 5 are driver, output and physics-layer work and change no kernel
field. Item 1's seam half and item 3 change the kernel contract. `observe`'s signature
is in the frozen `DRIVER_SURFACE` literal of `tests/backends/test_driver.py`, and
`kperp_rate` extends its `SEPARABLE_OPERATOR_FIELDS`. Both land in **one** bump of
`translate/problem.py`'s `SCHEMA_VERSION`, to the next free version at landing (29 at
this writing, after SEP 25's 28, unless SEP 21 or SEP 24 lands a bump first). The bump brings the rewritten literals,
the kernel contract §3 and §9 rows and a changelog line. In solprop-kokkos it moves
`build_record()`'s `schema_version` (`src/bindings.cpp`) and `check_build`, and extends
`SEPARABLE_OPERATOR_FIELDS` (`src/solprop_kokkos/__init__.py`). The output
`SCHEMA_VERSION` moves in phase 1 to the next free version at landing (19 at this
writing, unless SEP 21 or SEP 24 lands a bump first).

#### 6.3 Strategies, item by item

**Field-free planes.** Phase 1 drops the field in the driver: `run_problem` still calls
`observe()` but keeps no field. That removes the storage, the preallocation and the
record copies, and is measured on its own. Phase 2's `observe(field=False)` removes the
rest of §6.1's per-plane cost. A field-free observation then relies on the step loop's
rule, which accepts only finite norms and makes a replay raise, and on the statistics.
That reliance is tested with an injected NaN.

**The zero-rate pass.** A translation change only: `temporal_rates` tests the two arrays
once and returns `None` for both when every rate is zero, and the backends already take
`None`. The field change is exact under the in-stage scheme and at rounding under the
split (§3), so the existing zero-rate fixtures pass unchanged under the first and within
a rounding bound under the second.

**The collar diagnostic.** It moves from today's whole-volume `ifft2` to a transform by
frequency block, which holds one block bounded by `node_budget` and needs host code only.
That is recommended for phase 1. Reusing a compiled backend's transverse scratch needs a
protected hook per backend and is a later optimisation. A native reduction would cover
field-free planes on Kokkos (Q2). The fraction's numerator and denominator are sums over
frequency, so blocks accumulate them exactly, up to summation order. On field-free planes
the diagnostic is not evaluated; the run records that and logs it once.

**The runner.** It is a separate context manager rather than a backend cached inside
`Simulation`, which would hold 15 volumes past `run()` unasked and own a resource with no
`close()`. §4.7's four resets go into `StepDriver.set_initial` and the backends'
overrides, so a fresh `run()` gains them too. A gigabyte array cannot be recorded inline,
so above Q4's threshold the record carries its digest. That costs seconds of hashing per
input, against the serialisation it replaces; smaller arrays keep their inline records.

**Operator settings.** `kperp_rate` is appended last. `None` is normalised to zeros in
`SeparableOperator.__post_init__`, so backends never branch, and values are validated
finite, nonnegative and of length `n_modes`. Each backend forms
$D - \tfrac12(\alpha_\omega + \alpha_\perp)$, adding the rates first, so a zero rate
leaves every element as it is ($s + 0 = s$ for nonnegative $s$). Like $\alpha_\omega$, it
enters outside the `active` factor; inactive bins hold exact zeros, so it acts on nothing
there (Q6). It is evaluated in `separable_values`, in `bodies.longitudinal_rate` (shared
by `spatial_element` and `slab_element`) and in `SpatialCoefficients::value`. Being real,
it never reaches $\operatorname{Im}\Phi$. The budget uses §4.4's search on the leaf.

**Backends and cost.** `numpy_ref` first, then `numba_cpu`, then the Kokkos CPU build.
On Kokkos, `observe(field=False)` needs no native change, because `StepDriver.observe`
does not call the `_observed_field` override; `SpatialCoefficients` gains one `RealView`.
The k⊥ rate costs one load and one add per element, measured on a ten-step window. GPU
builds gain most from the runner (no repeated device allocation or planning) and from
field-free planes (no transfer). They are qualified later under the standing policy
([backends](../../backends.md)). No new dependencies.

### 7. Verification and validation

Tolerances follow [testing §3](../../testing.md#3-tolerance-policy). Exact equality
(`numpy.array_equal`) is required only for structural contracts: copies, configuration
records and digests, integer ledgers, restored state, and defaults or exact zeros that
leave today's operations unchanged by construction. Separate runs are compared within a
rounding bound fixed before measurement on a replayed schedule, and within the case's
accuracy envelope against a tighter or stored reference when adaptive, with adaptive step
counts in a loose band ([SEP 25](sep-0025-rounding-level-reproducibility.md)).
Backend parity uses the peak-normalised deviation against the summation-length bound, on
nontrivial fields. Below, $\varepsilon = 2^{-53}$, $\gamma_m = m\varepsilon/(1-m\varepsilon)$,
and $n$ is the number of steps of the window.

- **Field-free planes leave the march unchanged.** Skipping a plane's field removes a
  copy and no march operation (§4.7). The same run is made with
  `fields="all"`, `"planes"` (`[0, -1]`) and `"none"` on all three backends: ten steps on
  a small Cartesian Kerr grid, adaptive and replayed, with and without interpolated
  planes and noise. On the replayed schedule the step histories must be equal and the
  statistics and kept fields within a rounding bound fixed before measurement; adaptive
  runs meet the case's accuracy envelope against a tighter-tolerance reference. So must
  the scan worker's results against today's `run_scan`.
- **Layout and refusals.** Inhomogeneous layouts and field reads on field-free planes are
  refused by name. `n_saved` counts from `z`, a field-free file resumes from its terminal
  checkpoint only, and fixtures at the previous output schema still read and verify.
- **Collar diagnostic.** The blockwise fraction agrees with the whole-volume one within
  $\gamma_m$ of each fluence sum ($m$ terms). `tests/backends/test_free_absorber.py` and
  `test_radial_absorber.py` pass, and the measured transient is at most one block.
- **Runner equals separate runs.** Three inputs in sequence through one runner, also
  after a deliberately failed run and after a dense run, agree with three fresh
  `Simulation.run` calls within a rounding bound fixed before measurement on a replayed
  schedule, including the entrance plane's `quadrature_error`, and the Adams call ledger
  is equal there. After each run, §4.7's four restored pieces equal a fresh backend's
  exactly. Hashes differ per input.
- **with_input validation.** Non-finite or misshaped inputs, a replaced shared leaf and a
  changed `initial_z` in the runner are refused. A large array's digest changes with one
  changed sample.
- **A zero k⊥ rate leaves today's operator.** Its elements equal today's exactly
  ($s+0=s$, §6.3) on all three backends, and a run with it agrees with today's within a
  rounding bound fixed before measurement.
- **The k⊥ collar on an aliased product.** A narrowband envelope run (Kerr, THG off)
  carries a weak beam at $0.95\,k_{x,\max}$, inside the ramp where $P = 1/2$, and a beam
  at $k_x = 0$. Their product at $1.9\,k_{x,\max}$ folds to the target region near
  $-0.1\,k_{x,\max}$, where nothing else lands. With the collar, the product's source
  decays at $\kappa = S\,P(0.95) = S/2$. The weak-conversion prediction for the target
  region's power at length $Z$, relative to the same run without the collar, is

  $$
  R=\Bigl\lvert\int_0^Z e^{(-\kappa+i\Delta k)z}\,dz\Bigr\rvert^2\Big/\Bigl\lvert\int_0^Z e^{i\Delta k z}\,dz\Bigr\rvert^2 ,
  $$

  with $\Delta k$ the phase mismatch from the operator's own $k_z$. The measured ratio
  must lie within 10 % of $R$, which bounds the neglected depletion and cascade terms at
  a conversion below $10^{-3}$. A unit test checks the profile's zeros, its value $S$ at
  $\rho = 1$, its slope and the column order.
- **Exactness of the rate.** A linear run in vacuum with $\alpha_\perp$ alone attenuates
  each column's magnitude by $e^{-\alpha_\perp L/2}$. The bound is
  $\gamma_{3n} + \tfrac12\alpha_\perp L\,\varepsilon$ relative: one exponential, one product
  and one rounded exponent per step. Adaptive steps and two replayed schedules each meet
  it, and the noise phase is unchanged.
- **Evanescent setting and helper.** `evanescent_rate=None` takes the default's value and
  record exactly. The
  helper equals a brute-force maximum over all columns on small grids, for loss and gain.
  The `SpatialPulse` launch is unchanged by the setting.
- **Exponent budget.**
  - A replayed schedule just inside $\ln(\mathrm{DBL\_MAX})/\lambda$ completes finite.
  - One just outside it is refused at translation, with the admissible values.
  - One in the warning band warns.
  - An adaptive run with $\lambda h_{\max} = 785$ completes as today.
  - A direct backend call without the refusal reproduces the non-finite stage.
  - No existing test or gallery case is refused.
- **Collar length and custom absorber.** At its default, `collar_length` gives today's
  rates and records exactly on the radial, Cartesian and planar routes. `custom_absorber(grid, 0.0,
  collar=c)` equals `c` field for field, and so gives `c`'s results and hash. With a
  nonzero rate only the spectral rate and, when needed, `reference_length` differ; the
  temporal and spatial rates are exactly `c`'s. A collar with $\ell_c$ far below
  `reference_length` validates.
- **The zero-rate pass.** `absorber(grid, temporal=False)` and a hand-built collar with
  zero rates translate to no temporal absorber under both schemes, and the record names
  the keyword. A ten-step Cartesian Kerr run equals today's exactly under the in-stage
  scheme and within a rounding bound fixed before measurement under the split; the step
  ledger shows the recycled stage; the per-step cost goes into the benchmark ledger
  beside §3's table.
- **Configuration-hash stability.** A stored-digest test pins the new output schema's digests of a
  modal record, a radial record, two Cartesian records (with and without an absorber) and
  `custom_absorber` at defaults. Later phases and proposals must keep them; the precedents
  are `tests/output/test_schema_lineages.py` and
  `tests/output/test_checkpoints.py::test_old_non_tdse_metadata_keeps_its_hash_and_readability`.
- **Cross-check with SEP 23.** Once SEP 23's `absorbed_power` lands, its transverse
  channel equals $\alpha_\perp$ times each column's power.
- **Parity (phase 2).** Each new kernel quantity, and a ten-step run with `kperp_rate`,
  agrees across `numpy_ref`, `numba_cpu` and Kokkos CPU within the peak-normalised
  summation-length bound of the existing Cartesian parity tests.
- **Benchmarks** go into the ledger ([benchmarks](../../benchmarks.md)), on ten-step
  windows: per-plane observation time and host peak with and without fields; the runner
  against repeated `run()`; and the separable element with the k⊥ term.

### 8. Documentation

- [Output](../../../guide/results/output.md), [resuming](../../../guide/results/resuming.md),
  [running](../../../guide/setup/running.md) and [scans](../../../guide/workflows/scans.md):
  field retention, the new output schema, resume from checkpoints and the runner.
- [Free-space model](../../../guide/free-space/model.md) and
  [Cartesian](../../../guide/free-space/cartesian.md): grid sizing by the one-half rule and
  the k⊥ safety ramp (§4.3), the evanescent setting (§4.4), the budget (§4.5) and the
  memory table. [Absorbing boundaries](../../../guide/numerics/boundaries.md):
  `custom_absorber`, `collar_length` and §4.6, cross-linked with SEP 21.
- [Kernel contract](../../kernel-contract.md) §3 and §9, [architecture
  §13](../../architecture.md#13-memory-discipline) and [backends](../../backends.md).
- A focused tutorial: a small Cartesian four-wave-mixing delay scan through a runner,
  with statistics-only planes, a custom spectral absorber and a k⊥ collar. It includes
  §7's aliased-product case with and without the collar, and a refinement showing the
  grid sizing.

### 9. Compatibility and impact

- **No default changes.** §7's default tests hold every new setting at its default to
  today's operator values and records exactly, and its runs to today's within a rounding
  bound; the budget refuses only runs that cannot complete.
- **Output schema** (the next free version at landing, 19 at this writing). `field` becomes
  optional, and older files read as before. `RunMetadata.hashed()` includes
  `schema_version` and `output.hdf5.check_cache` compares digests. Files written before
  the bump therefore do not resume after it, as with every output schema bump. §5.5 keeps
  digests stable from then on.
- **Kernel schema** (the next free version at landing, 29 at this writing) in phase 2.
  solprop-kokkos is rebuilt, and an older build refuses the problem by its schema.
- **Breaking:** none at the interface. `SavedPlane.field` widens to admit `None`.

### 10. Alternatives considered

| alternative | why not chosen |
|---|---|
| A sink `Output` that discards fields after `observe` | Saves storage, but not the Kokkos transfer, the scan or the record copies; it is phase 1's interim, not the end state |
| Dealiasing by a doubled grid in every nonlinear evaluation | Exact for polynomial sources, but four times the memory and transforms per evaluation at the scale this SEP exists for; grid sizing achieves the same for the content that carries energy |
| A wide k⊥ ramp ($f \ge 1/2$) as anti-aliasing | Cannot remove aliases generated in band (§4.3) and would damp physical products |
| A k⊥ collar applied in-stage, like the real-space collar | Conditionally stable and step-dependent; the diagonal rate is exact and free (§4.2) |
| An elliptical (isotropic) k⊥ collar | Leaves the rectangle's edge bands partly unprotected; kept as Q5 |
| Clamping the propagator exponent instead of refusing | Silently changes the operator; a refusal names the remedy |
| Refusing adaptive runs past the budget | Refuses runs that complete today; the controller already rejects non-finite trials |
| Caching the backend inside `Simulation` | Hidden 15-volume lifetime and no `close()`; the explicit runner is the one obvious way |
| `custom_absorber` mirroring `absorber`'s keywords | Would need to follow every SEP 21 parameterisation; taking the collar by value does not |

### 11. Phases

Each phase is on its own `feature/sep-0022-…` branch, `numpy_ref` first, and appends its
report to Part II.

- **Phase 1 — driver, output and absorber; no kernel bump.** It covers:
  - statistics-only planes in the driver, the handlers and the scan worker (the output
    schema bump, homogeneity, readers, resume from checkpoints);
  - the blockwise collar diagnostic;
  - the runner, with `with_input(configuration=)`, Initial-only validation, digest
    records and the four resets;
  - `collar_length` with its readers and ceiling, and `custom_absorber`;
  - the spectral-only collar (`temporal=False`) and the zero-rate pass compiled away;
  - the omit-default rule and its stored-digest test.

  It is verified by §7's tests for these items, with per-plane and per-input costs
  recorded, and it writes their manual sections.
- **Phase 2 — the seam; one kernel schema bump.** It adds `observe(field=False)` (field
  and noise phase), the collar diagnostic inside the guard, `kperp_rate` with
  `TransverseCollar`, `evanescent_rate` with `physical_evanescent_rate`, and the exponent
  budget. These go on `numpy_ref`, `numba_cpu` and then Kokkos CPU, with the frozen
  literals, `check_build`, ledger entries and parity. It is verified by the
  aliased-product, exactness, budget and default tests.
- **Phase 3 — documentation and scale.** The tutorial, the design pages, the roadmap
  item's removal and benchmark entries at a large shape. Then the closing gate and the
  slow, coverage and Kokkos lanes, once. Device qualification follows separately.

### 12. Open questions

- **Q1 — Checkpoint fields in field-free runs (phase 1).** Should a terminal checkpoint
  always carry the field, for one transfer at the stop? Recommended: yes.
- **Q2 — The collar diagnostic on field-free planes (phase 2).** Skip and record it
  (recommended), or add backend statistics? A k-space fraction needs no transform; a
  real-space one needs a native Kokkos reduction, a catalogue addition beyond this SEP.
- **Q3 — Custom reductions in scans (phase 1).** May a reduction declare that it reads no
  field, so that its spooled run can be field-free?
- **Q4 — The digest threshold (phase 1).** Proposed: above $2^{20}$ elements. May a
  caller-declared identity (`configuration=`) replace the digest?
- **Q5 — k⊥ collar shape (phase 2).** Max-norm (recommended) or elliptical? Should the
  hybrid planar kind admit a $k_y$-only collar?
- **Q6 — `kperp_rate` outside `active` (phase 2).** Mirror $\alpha_\omega$, as drafted, or
  multiply by `active`, which keeps inactive bins' operator zero and tightens the budget?
- **Q7 — The ceiling for user rates (phase 1).** Keep the derived `reference_length` of
  §5.4, or validate custom spectral rates against the exponent budget, where the step is
  known?
- **Q8 — Zero temporal rates and the spectral-only collar.** Taken back from SEP 21 on
  2026-10-18 (§3, §5.4): phase 1 delivers both. `absorber=False` on the keyword
  interfaces stays SEP 21's; its Q16 records the hand-back and the grid requirement.
- **Q9 — Follow-ups (roadmap).** Field-free `observe_members`; runner reuse inside
  `run_scan` workers; counting the compiled backends' extra base volumes in
  `estimate_memory` on the Cartesian kind, where each is gigabytes.
- **Q10 — The budget's span.** Closed from the code in §4.5: one step for loss, and
  $q+1$ steps for gain under `Multistep()`.
- **Q11 — The user rate apart from the collar's (phase 1).** `custom_absorber` sums the
  user's spectral rate into `spectral_rate`, so SEP 21's spectral `absorbed_energy` and
  SEP 23's `absorbed_power` spectral channel report collar and user rate together.
  Should `Absorber` keep the user rate as its own field, summed at translation, so that
  each can be attributed? Raised by SEP 21's review (its Q16).

### 13. References

The DOIs were resolved on 2026-10-16, and the titles, authors and venues were checked
against the registration records.

1. S. A. Orszag, "On the elimination of aliasing in finite-difference schemes by filtering
   high-wavenumber components," *Journal of the Atmospheric Sciences* **28**, 1074 (1971).
   doi:10.1175/1520-0469(1971)028<1074:OTEOAI>2.0.CO;2
2. T. Y. Hou and R. Li, "Computing nearly singular solutions using pseudo-spectral
   methods," *Journal of Computational Physics* **226**, 379–397 (2007).
   doi:10.1016/j.jcp.2007.04.014
3. M. Kolesik and J. V. Moloney, "Nonlinear optical pulse propagation simulation: From
   Maxwell's to unidirectional equations," *Physical Review E* **70**, 036604 (2004).
   doi:10.1103/PhysRevE.70.036604
4. A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and M. Kolesik,
   "Practitioner's guide to laser pulse propagation models and simulation," *The European
   Physical Journal Special Topics* **199**, 5–76 (2011). doi:10.1140/epjst/e2011-01503-3
5. R. Trebino, K. W. DeLong, D. N. Fittinghoff, J. N. Sweetser, M. A. Krumbügel,
   B. A. Richman and D. J. Kane, "Measuring ultrashort laser pulses in the time-frequency
   domain using frequency-resolved optical gating," *Review of Scientific Instruments*
   **68**, 3277–3295 (1997). doi:10.1063/1.1148286
6. *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (2019).
   doi:10.1109/IEEESTD.2019.8766229

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8),
following the skeleton in [the template](../template.md).*

### Phase 1 report — driver, output and absorber

- **Scope:** phase 1 of §11 (driver, output and absorber; no kernel schema bump),
  2026-10-18, Opus 5.5 (AI) for John Travers, branch `feature/sep-0022-phase-1` from
  `main` 62b0250 and merged with `main` at 00ae962 (SEP 24's phase 1 and the fast haCC
  presets), then with `main` at e1a8274 (SEP 26 accepted); merged into `main` on
  2026-10-18 by the merge commit that follows the commit carrying this report. solprop-kokkos is unchanged. The
  output `SCHEMA_VERSION` moves to 20: [SEP 24](sep-0024-low-storage-interaction-picture-runge-kutta.md)'s
  phase 1, merged first, took 19 (and kernel schema 29).
- **Delivered:**
  - *Statistics-only planes* (§5.1, §6.3): `output/schema.py` — `FieldRetention`, the
    one validator `field_retention_refusal`, `SavedPlane.field | None`,
    `TerminalCheckpoint` requiring a field, `PlaneLayout.retention`,
    `PlaneStore.retain` / `retention` / `field_shape` / `plane_field`, the
    `"field_planes"` key, the homogeneous-layout refusal in `_check_layout`, and
    `held_bytes(..., kept=, field_bytes=)`; `output/memory.py` (field block sized by the
    rule, none for `"none"`; `planes()` yields `field=None`); `output/hdf5.py` (root
    attributes `fields` and `field_shape`, the `field_planes` dataset, no `field` or
    `noise_phase` dataset for `"none"`, the `n_saved` fallback from `z`'s extent,
    `resume` returning a field-free last plane, and `_field_room`, which grows the
    named planes' field rows back after a close trimmed them); `driver/run.py` (`run_problem(...,
    fields=, field_planes=)`, `field_retention`, the field dropped before any copy or
    conversion, the terminal checkpoint observed again for its field);
    `driver/simulation.py` (`SimulationSettings.fields` / `field_planes`, both execution
    settings; `Simulation.run` passing them and the admission's retention; the
    field-free resume refusal and `_completed`); `driver/memory.py` (`admit_memory(...,
    retention=)`, `field_bytes`); `driver/_scan_run.py` (`_retained`, the capture's
    `retain`, the builder refusal); `processing/scans.py` (`custom(...,
    reads_fields=)`, `_Builtin.reads_fields`); `processing/axes.py` (one plane read
    through `plane_field`); `plotting/style.py` (`field_modes`); `stats.py` (refusal on a
    field-free plane). `fields` is outside the configuration hash. Every rule but
    `"all"` is refused beside `LowStorageRungeKutta()` (`driver/run.py`'s
    `low_storage_retention_refusal`, called by the settings and by the driver's door;
    the scan worker keeps every field under that march).
  - *The blockwise collar diagnostic* (§6.3): `backends/spatial.py` (`collar_block`,
    `_blockwise_fluence`; the Cartesian and planar kinds), `fourier.py`
    (`energy_weights`). The roadmap item *The collar diagnostic on Cartesian saved
    planes* is removed.
  - *The runner* (§5.2, §4.7): `driver/simulation.py` (`Simulation.runner`,
    `SimulationRunner`, `check_derived_problem`, `with_input(..., configuration=)` with
    Initial-only validation, `recorded_input`); `translate/problem.py`
    (`KernelProblem.validate_launch`); the resets in `backends/driver.py`
    (`StepDriver.set_initial`), `backends/multistep.py` (`AdamsMarch.begin_run`),
    `backends/numpy_ref/backend.py` and `backends/numba_cpu/backend.py` (the quadrature
    alarm), and, after the merge with SEP 24, the low-storage march's
    `source_evaluations` count, zeroed on load; `output/schema.py` (`ArrayDigest`, `INLINE_ARRAY_LIMIT`).
  - *Collar length and custom absorber* (§4.6, §5.4): `boundaries.py`
    (`Absorber.collar_length`, `Absorber.temporal`, `in_stage_length`, the two
    ceilings, `absorber(..., temporal=)`, `custom_absorber`); the readers in
    `translate/free.py`, `translate/radial.py`, `translate/planar.py`,
    `_warn_capillary_walkoff` and `_refined_tdse_collar`.
  - *The zero-rate pass* (§3, §5.4, §6.3): `translate/masks.py` (`temporal_rates`).
  - *The omit-default rule* (§5.5): `output/schema.py` (`OMIT_DEFAULT`,
    `RECORDED_DEFAULT`, `_holds_default`), replacing the special cases of
    `nonlinear/tdse.py`'s source filter, model and atomic window and
    `nonlinear/responses.py`'s plasma integration rule.
  - *Tests:* `tests/output/test_field_retention.py`, `tests/output/test_record_stability.py`
    (with its stored numbers), `tests/driver/test_field_free_planes.py`,
    `tests/driver/test_scan_field_retention.py`, `tests/driver/test_runner.py`,
    `tests/driver/test_zero_rate_pass.py`, `tests/test_collar_length.py`,
    `tests/backends/test_collar_diagnostic.py`, and plane reads through the processing
    axes in `tests/processing/test_axes.py`; current-schema literals moved to 20, and
    three tests that handed a zero-rate collar's fine rate back to the seam now give
    zeros on the coarse axis beside it.
  - *Pages:* [output](../../../guide/results/output.md), [resuming](../../../guide/results/resuming.md),
    [running](../../../guide/setup/running.md), [scans](../../../guide/workflows/scans.md),
    [absorbing boundaries](../../../guide/numerics/boundaries.md), the
    [free-space model](../../../guide/free-space/model.md)'s collar formula,
    [architecture §13](../../architecture.md#13-memory-discipline), the
    [roadmap](../../roadmap.md), the [benchmark ledger](../../benchmarks.md) and `CHANGELOG.md`.
  - *Benchmark:* `benchmarks/sep22/phase1_costs.py`. The SEP 12 parent reconstruction
    (`benchmarks/sep12/parent.py`, `tests/test_sep12_parent.py`) admits and converts a
    schema-19 parent onto schema 20, which moves nothing of a pulse-launched parent's
    record.
- **Deviations from the proposal:**
  - *The collar diagnostic on field-free planes.* §6.3 says the diagnostic is not
    evaluated on field-free planes, the run recording that and logging it once. In
    phase 1 `observe()` is unchanged, so it still observes the field at every plane and
    the diagnostic still runs there, blockwise; §6.1 assigns its removal on field-free
    planes to phase 2's `observe(field=False)`, and §11 puts "the collar diagnostic
    inside the guard" in phase 2. The record and the single log line go there with it.
  - *The stored-digest test* (§7, "Configuration-hash stability") pins each record as
    SEP 25 pins a record (`tests/record_numbers.py`): its structure by an exact SHA-256
    and its numbers, stored beside the test, to `RECORD_RTOL`, with an optmat release's
    text neutralised. One SHA-256 of the whole record would also cover the last bits of
    derived numbers (an absorber's logarithmic spectral rate among them), which SEP 25
    found another platform's libm moves.
  - *The Kokkos quadrature alarm.* §4.7 names `_alarm` and `_alarm_reported` in all three
    backends. The in-tree backends reset them on load; Kokkos is unchanged in this phase,
    and needs no change: its `quadrature_error` statistic is read from its native state,
    which a load resets, and its `_alarm` is never read before a march. The restore test
    compares the alarm by that observable on every backend.
- **Decisions:**
  - *Q1 — checkpoint fields in field-free runs:* adopted as recommended. A terminal
    checkpoint always carries the field; a field-free pending plane is observed again
    for it when the run stops. A file whose last regular plane kept no field and which
    holds no checkpoint is refused by name; a finished field-free file resumes as
    finished.
  - *Q3 — custom reductions in scans:* yes. `scans.custom(..., reads_fields=False)`
    declares a reduction that reads no field; its spooled run keeps the fields the scan
    keeps anyway (its retained planes and its built-in reductions' target planes) and no
    other. The default stays `True`, the safe one. The declaration enters the
    reduction's provenance only when it is `False`, so every earlier scan's settings are
    unchanged.
  - *Q4 — the digest threshold:* 2²⁰ elements, as proposed, applied to the run's
    `initial` entry (not to every array the record serialises: the noise record and the
    tables keep their forms). A caller-declared identity does **not** replace the digest:
    a label could claim two different arrays are one run, while the digest proves the
    content; a caller's identity is recorded beside it through `configuration=`.
  - *Q7 — the ceiling for user rates:* the derived `reference_length` of §5.4, as
    recommended. Because the default collar's spectral rate reaches its ceiling at the
    band edges, any user rate there moves the reference length; the in-stage collars
    never move.
  - *Q11 — the user rate apart from the collar's:* not kept apart in this phase.
    `custom_absorber` sums it into `spectral_rate`, which is what the seam carries; a
    separate field would need either a second spectral leaf or a translation-time sum
    that SEP 21's and SEP 23's diagnostics could not see. Should attribution be needed
    when SEP 23's `absorbed_power` lands, a `user_rate` field added with the omit-default
    marker keeps every record; it moves to Q12.
  - *The retention rule reaches the output by `PlaneStore.retain`*, called by the run
    loop after the output opens (so a continued file compares it with its own) and before
    the first plane; an output without `retain` is refused for a field-free run before it
    is opened. The shared validator returns a message and each door raises its own
    layer's error (`DriverError` in the driver and the scan, `OutputError` in the output).
  - *Whole-run field reads* (`output["field"]`, `output["noise_phase"]`, `fields()`) are
    refused on any run that did not keep every field, naming the rule, rather than
    returning the kept rows: processing indexes them by plane, and a shorter array would
    be read at the wrong planes. `plane_field(index)` and `output["field_planes"]` are
    the readers for the others.
  - *The scan worker* passes its retention through a shallow copy of the builder's
    simulation with the two execution settings replaced, so a builder that returns one
    object for every point is left as it was.
  - *`collar_length` equal to `reference_length` is stored as `None`*, so the record
    carries it only where the two differ whichever way the collar was built.
    `Absorber.temporal` (omit-default) records `temporal=False`; a collar declared
    spectral-only with a non-zero temporal rate is refused.
  - *The runner's admission:* the backend is admitted once, before it is constructed;
    each run admits its in-memory output beside it (a file output adds nothing), so a
    runner writing files is not charged a memory store it never holds.
  - *The four resets* sit in `StepDriver.set_initial` (the dense look-ahead, and the
    Adams ledger and last-rejection flag through `AdamsMarch.begin_run`) and in the
    in-tree backends' overrides (the quadrature alarm).
  - *The collar block* holds at most one node chunk's fine temporal column of samples
    with its working copies (three block volumes), a twentieth of what the node budget
    allows the response pass; a small problem takes its band in about three blocks. The
    planar kind uses the same blockwise path.
  - *The omit-default marker* takes an optional `recorded_default` beside it, for the
    fields whose historical records left out a value other than their declared default,
    and so replaces the TDSE and plasma special cases; the scalar-Raman case, which
    records a different set of fields, stays. This phase adds no configuration key:
    `metadata_for`'s two conditional keys, which are dictionary entries rather than
    dataclass fields, are as they were, and phase 2's keys can use the marker on their
    settings or follow the same pattern.
  - *Field-free planes beside the low-storage march are refused* (from the phase's
    review). That march writes no terminal checkpoint for a step abandoned in flight
    (SEP 24 §6.2), so a run stopped after a field-free plane would leave nothing to
    resume from. `fields` other than `"all"` is refused beside it by name, in the
    settings and at the driver's door before the output is opened, naming the
    remedies; the abandoned-step note now says the file resumes from its last plane
    when that plane kept its field. Whether to hold the last plane's field instead is
    Q14.
  - *`SimulationRunner.close()` takes the runner's lock*, so a close from another
    thread waits for a run in progress, and `run()` checks the closed flag under it.
  - *The merge with SEP 24* keeps both version notes in `SCHEMA_VERSION`'s docstring
    and corrects SEP 24's, which named its version 20, to the 19 it took.
  - No ADR: the SEP asks for none, and every decision above is local to this SEP's
    interfaces.
- **Verification:** on the branch, Apple silicon (18 cores, a busy machine), with the
  Kokkos CPU build installed; targeted runs only, as SEP 0 and `AGENTS.md` ask of a
  phase: the full gate, the slow lane, the coverage lane and the Kokkos-selected slow
  lane are the closing phase's.
  - *Tests:* the new modules and every module that exercises the changed code —
    `tests/output`, `tests/processing`, `tests/plotting`, the driver's run, resume,
    memory, scan, simulation, entry-point and checkpoint modules, the backends' driver,
    absorber and collar modules, `tests/test_display.py`, `tests/test_stats.py`,
    `tests/test_boundaries.py` and `tests/translate/test_tdse_admission.py` — pass
    (2 306 tests in the last full targeted run, and 1 111 after the move to schema 20);
    an earlier run of the whole of `tests/driver`, `tests/backends`, `tests/translate`,
    `tests/nonlinear`, `tests/processing` and `tests/plotting` (8 924 tests) found the
    eighteen cases of `test_spatial_absorber_schemes.py` that handed a zero-rate
    collar's fine rate back to the seam, since given zeros on the coarse axis beside it.
    The doctests of every changed module pass (169). After the merge with `main` at
    00ae962 and a rebuild of the Kokkos CPU wheel at kernel schema 29, the same
    targeted modules with SEP 24's low-storage modules, the SEP 12 parent test and the
    structure figures pass (2 074 tests), as do the doctests of every changed module
    with the Kokkos package imported first, as the suite imports it (181); the record
    skeletons are unchanged by the merge. The review's fix pass (the HDF5 field rows
    grown back on resume, field-free planes refused beside the low-storage march, its
    evaluation count zeroed on load, the runner's close under its lock, a per-plane
    noise-phase check and the processing axes' plane reads) adds its tests -- the
    HDF5 resume and the evaluation count were seen to fail without their fixes -- and
    the modules it touches pass
    (975 tests, with the doctests of the changed modules, and 66 in
    `tests/processing/test_axes.py`); a coverage run of `tests/processing` covers every
    line of `_plane_field`.
  - *Bounds, fixed before the runs:* a replayed ten-step Cartesian window is compared
    across retention rules within `10 (γ₁₆ + 7 B_fft)` of the field (twice that for a
    statistic, and that over `rtol` for the error norms), the runner against fresh runs
    within `10 Σ|w| (B_fft(n_to) + B_fft(n_nodes))` with the Adams call ledger and the
    step history exact, the blockwise collar fraction against the whole-volume one
    within `2 γ_m + 4 B_fft(n_nodes)`; adaptive runs meet an envelope of four times the
    full-field run's own error against an `rtol = 1e-10` reference. Structural
    contracts are exact: prescribed step histories, stored fields and indices, the four
    restored pieces of state (the alarm compared by the entrance's `quadrature_error`,
    on all three backends), `custom_absorber`'s copied arrays, a zero rate's record, and
    the reference backend's in-stage run with and without the zero-rate pass. The
    record skeletons are pinned at output schema 20.
  - *Lint:* `ruff check` (the whole tree), `ruff format` (`poe fmt` leaves every
    file unchanged), `basedpyright` (the whole project before the merge, the changed
    modules after it; 0 errors), the numpydoc lint of every changed module and `codespell` of every changed
    file are clean; the public-tree scanner finds nothing on any line this branch adds.
    A targeted coverage run of the changed modules covers every line and branch the
    branch adds (one changed line of `stats.py` is covered by
    `tests/test_spatial_statistics.py`).
  - *Docs:* `poe docs` (`-W -n --keep-going`) builds clean on the merged tree.
  - *Benchmarks* ([ledger](../../benchmarks.md), ten-step windows, a load average of 23
    to 28): the zero-rate pass removes 1.33 and 2.38 times a step on `numba_cpu`
    (split, in-stage) and 1.23 and 1.86 on Kokkos CPU, in line with §3; field-free
    planes cut the save-plane loop's host peak from 962 to 130 MiB (`numba_cpu`) and
    898 to 6 MiB (Kokkos) at eleven 64 MiB planes, the time inside the spread; the
    runner's saving at this size (a backend constructed in 0.05–0.09 s) is inside the
    noise of a three-second window, and is measured at a large shape in phase 3.
  - *Not run:* the full gate, the slow, coverage and Kokkos lanes, GPU builds, and any
    large-shape measurement (phase 3).
- **Sources checked:** no physical constant or coefficient was transcribed. The
  numbers of §4.6 quoted on the boundaries page (the `1.2 × 10⁶ m⁻¹` cap at 50 µm, the
  spatial collar's `2.5 × 10⁵` and `2.1 × 10⁶ m⁻¹` peaks and its 26 µm and 3.1 µm DOPRI5
  stable steps) were rederived from `MAX_ALPHA_LENGTH`, `TEMPORAL_ALPHA_LENGTH` and
  `STABILITY_RADIUS`; the 1.3–2.3 and 1.2–1.6 cost ratios are §3's measurement, quoted
  beside this phase's re-measurement.
- **Limitations:**
  - On every backend the field is still observed at a field-free plane and dropped by
    the driver: the device transfer on Kokkos, the finiteness scan, the collar
    diagnostic and the noise-phase volume remain until phase 2.
  - The keyword interfaces (`prop_*`) take no `fields=`; set it on the simulation's
    settings, or through `run_scan`.
  - A `"planes"` run's noise phase is read through `MemoryOutput.planes()` or a resume;
    there is no per-plane noise-phase accessor beside `plane_field`.
  - Lockstep batches (`observe_members`) keep their fields (Q9), and scan workers
    construct a backend per point rather than a runner (Q9).
  - `LowStorageRungeKutta()` runs keep every field (Q14).
- **Open questions:**
  - Q12 — Attributing a custom spectral rate apart from the collar's (from Q11): add an
    omit-default `user_rate` to `Absorber` if SEP 23's `absorbed_power` or SEP 21's
    spectral `absorbed_energy` needs the split. Destination: SEP 23's integration, or
    the roadmap.
  - Q13 — Whether the keyword interfaces should offer `fields=` beside `memory_budget`
    (the roadmap's *A memory budget keyword on the keyword interfaces*). Destination:
    phase 3's review of the manual, or the roadmap.
  - Q14 — Field-free planes under the low-storage march (from the phase's review):
    hold the last saved plane's field, one volume, so that a step abandoned in flight
    still leaves a resumable state, or keep this phase's refusal. Destination: phase 2,
    beside `observe(field=False)`.
- **Handoff:** phase 2's `observe(field=False)` has one caller to change: `_observed` in
  `driver/run.py`, which already knows whether the plane keeps its field (`keep`), and
  `_persist_failure`, which already observes again for a checkpoint's field. The
  collar diagnostic is blockwise and host-only (`spatial_collar_fraction`), ready to move
  inside the guard with its once-per-run record and log line. The output schema is 20;
  SEP 24's phase 1 took 19 and kernel schema 29, so phase 2's kernel bump takes the next
  free version at landing (30 at this writing). `Absorber.collar_length` follows
  `boundary_N` or `boundary_length` today; whichever of SEP 21's phase 2 and this SEP
  lands second makes `absorber(..., walkoff=...)` fill it (§5.4). The schema-20 record
  skeletons of `tests/output/test_record_stability.py` must be kept by `kperp_rate`,
  `evanescent_rate` and `transverse_absorber`: give each the omit-default marker, or
  leave it out of the record at its default.

### Phase 2 report — the seam (kernel schema 31)

- **Scope:** phase 2 of §11 (the seam; one kernel schema bump), 2026-10-18, Opus 5.5
  (AI) for John Travers, branch `feature/sep-0022-phase-2` from `main` efb7501 (SEP 22
  phase 1, SEP 24 phase 1, SEP 23 phase 1 and SEP 26 phase 1 merged), and solprop-kokkos
  branch `feature/sep-0022-phase-2` from its `main` f40728d; merged with `main` at
  4efb87b (SEP 26 phase 2, no conflict) and a4f3a38 (SEP 26 phase 3 and Final; the
  kernel contract's version-30 row, taken from `main`) and merged into `main` on 2026-10-19 by the
  merge commits that follow the commit carrying this report, solprop-kokkos alongside. The kernel
  `SCHEMA_VERSION` moves to **31**: SEP 26's phase 1 took 30, which the phase 1 handoff
  named. The output schema stays 20: no stored record changes (the new configuration
  keys are written only when set).
- **Delivered:**
  - *The seam* (§5.1): `backends/driver.py` (`StepDriver.observe(*, field=True)`, which
    with `field=False` calls neither `_observed_field` nor `_noise_phase`; the skipped
    collar diagnostic's count and once-per-load log, `collar_diagnostic_skipped` in
    `metadata()`; `held_observation()` and the held volume of Q14, reset by
    `set_initial`); `backends/protocol.py` (`Observation.field | None`, a noise phase
    refused without a field, `Observation.require_field()`, the `Backend.observe`
    signature); `backends/batch.py` (the lockstep runner's refusal takes the keyword);
    `driver/run.py` (`_observed` asks for the field exactly on kept planes; a
    low-storage step in flight checkpoints the held plane; `low_storage_retention_refusal`
    removed); `driver/simulation.py` and `driver/_scan_run.py` (the refusal removed,
    the scan worker keeps only the fields the scan reads under every march);
    `driver/memory.py` (the `held_plane` admission group); `driver/batch.py` (a
    field-free batch observation refused by name); the in-tree ledgers' `held_plane`
    row (`backends/numpy_ref/backend.py`, `backends/numba_cpu/backend.py`).
  - *Operator settings* (§5.3): `boundaries.py` (`TransverseCollar`,
    `transverse_collar_rate`); `spatial.py` (`attenuation_extremes`,
    `physical_evanescent_rate`); `translate/spatial.py`
    (`SeparableOperator.kperp_rate`, its rate in `separable_values` and in
    `axial_factors`' transverse channel, `RateBounds`, `rate_bounds`);
    `translate/free.py` (`free_operators(..., transverse_rate=)`);
    `translate/problem.py` (`assemble(..., evanescent_rate=, transverse_absorber=)`,
    their refusals, `KernelProblem._check_exponent_budget`, `LOG_DBL_MAX`,
    `EXPONENT_BUDGET`, schema 31); `exceptions.py` (`ExponentBudgetWarning`);
    `driver/simulation.py` (`SimulationSettings.evanescent_rate` and
    `transverse_absorber`, recorded only when set); `backends/bodies.py` and
    `backends/numba_cpu/spatial.py` (the element's transverse rate).
  - *solprop-kokkos*: `SpatialCoefficients::value` adds the rate on active bins
    (`src/general.cpp`, one `RealView` per column); `Problem::spatial_kperp_rate`
    (`include/solprop_kokkos/core.hpp`, `src/bindings.cpp`) with its validation and
    ledger row; the adapter packs it (`backend.py`); `SEPARABLE_OPERATOR_FIELDS`, the
    build record and the tests at schema 31; `CHANGELOG.md` and `README.md`. The
    evanescent rate was already a problem field, and `observe(field=False)` needed no
    native change: the inherited observation never calls `_observed_field`.
  - *Tests:* `tests/test_transverse_collar.py`, `tests/translate/test_operator_settings.py`,
    `tests/backends/test_kperp_collar.py`, the seam tests appended to
    `tests/driver/test_field_free_planes.py`, the held-plane tests in
    `tests/backends/test_low_storage.py`, the transverse channel in
    `tests/backends/test_far_field_statistics.py`, the phase 1 follow-ups (the restore
    test sets `_last_rejected`, `tests/driver/test_runner.py`; field retention leaves
    the configuration hash unchanged, `test_operator_settings.py`), and the frozen
    literals at schema 31. Every test that read `backend.observe().field` reads
    `require_field()` (solprop and solprop-kokkos).
  - *Pages:* [Cartesian](../../../guide/free-space/cartesian.md) (grid sizing by the
    one-half rule and the safety ramp), the [free-space model](../../../guide/free-space/model.md)
    (the evanescent setting, the exponent budget, the skipped diagnostic),
    [absorbing boundaries](../../../guide/numerics/boundaries.md),
    [output](../../../guide/results/output.md), the stepper pages' in-flight notes, the
    [kernel contract](../../kernel-contract.md) (§3, §9, the budget, version 31), the
    contract figure regenerated, the [benchmark ledger](../../benchmarks.md) and
    `CHANGELOG.md`.
  - *Benchmark:* `benchmarks/sep22/phase2_costs.py`.
- **Deviations from the proposal:**
  - *The transverse rate sits inside `active`* (Q6, below), where §5.3 and §6.3
    drafted it beside the spectral rate, outside: `D = (-(a + α⊥/2) - i(Re s - f))χ -
    α_ω/2`, formed as `-a - α⊥/2` so that a zero rate changes no bit, signed zeros
    included. "Adding the rates first" therefore adds it to the attenuation, not to the
    spectral rate; the default identity of §7 holds exactly either way.
  - *The budget's loss bound is taken per frequency*: the larger of the propagating
    and the capped evanescent attenuation, plus half the largest transverse rate on
    active bins, plus half that bin's spectral rate, maximised over the frequencies.
    It is still sufficient and is tighter than §4.5's sum of separate maxima, which
    would charge an inactive bin's large spectral rate to an active bin's evanescent
    one. The search runs per slab channel, so the hybrid planar kind is bounded too.
  - *`Observation.require_field()`* is new: with `field` optional, the 125 places in
    the two test suites that read `backend.observe().field` would each need a
    narrowing; one accessor that raises by name on a field-free observation replaces
    them.
  - *`StepDriver.held_observation()`* joins the driver surface for Q14 (frozen in
    `tests/backends/test_driver.py`); solprop-kokkos does not use it, since it refuses
    the low-storage march.
  - *The skipped diagnostic's record* is the execution record: the backend counts the
    field-free planes in `metadata()` (`collar_diagnostic_skipped`, outside the
    configuration hash, written with every plane's outcome) and logs the first at
    INFO, once per load. No statistic was added (Q2).
  - *The new settings' records* are configuration keys written only when set, as the
    phase 1 handoff allowed, rather than omit-default markers: `SimulationSettings` is
    not serialised as a dataclass. The collar is recorded as its two numbers.
  - *Refusals*: `transverse_collar_rate` refuses a radial grid itself; the hybrid
    planar kind is refused by translation (and by `SeparableOperator` for a nonzero
    rate beside `kappa_sq`), since the function's annotation takes Cartesian and radial
    grids only. `physical_evanescent_rate` takes `FreeGrid`, `LineGrid` and
    `RadialGrid` and refuses a `PlanarGrid`, whose columns are slab channels.
  - *The aliased-product test* uses two equal plane waves (columns 0 and 19 of 40 on a
    line, ρ = 0.95) rather than a weak and a strong beam: with a strong beam a hundred
    times the weak one the idler at −0.95 k_max, itself driven by the strong beam,
    feeds the target column through a cascade comparable to the direct product (the
    measured ratio fell to 0.78 against R = 1.50). With equal beams at a conversion of
    7 × 10⁻¹¹ the cascade is second order and the ratio is within 3 × 10⁻⁴ of R.
  - *The parity test* uses the free-space ladder's `amplitude_bound` and writes no
    row into the stored equivalence records.
- **Decisions:**
  - *Q2 — the collar diagnostic on field-free planes:* skipped and recorded, as the
    maintainer decided: the count in the execution record and one log line per load;
    kept planes still evaluate it.
  - *Q5 — the collar's shape:* the max-norm, as the maintainer decided; the hybrid
    planar kind's `k_y`-only collar is refused by name.
  - *Q6 — `kperp_rate` inside `active`:* multiplied by `active`. The budget argument
    decides it: the rate is a function of the column alone, so outside `active` it
    would add `α⊥/2` to every inactive bin's exponent, where the state holds exact
    zeros and a strong collar could form `e^{α⊥h/2}·0`, a NaN in the next transform,
    on bins that carry nothing; inside, an inactive bin's operator is today's whatever
    the collar, the budget charges the collar to active bins only (the per-frequency
    bound above), and SEP 23's transverse channel is `α⊥` on the active band, still
    twice the negated real part of the operator. The cost is one multiply already
    there. Recorded here; no ADR (the choice is local to this leaf and stated in the
    kernel contract §3).
  - *Q14 — field-free planes under `LowStorageRungeKutta`:* admitted, as the
    maintainer decided. `observe(field=False)` under that march copies the state into
    a held buffer (allocated at the first such plane, listed as `held_plane` by the
    in-tree ledgers, charged at admission as the `held_plane` group when the retention
    keeps fewer than every field); a kept plane releases it and a load clears it. When
    a step is abandoned in flight with a held plane, the driver writes the terminal
    checkpoint from it (an empty history, the plane's own having been saved, and its
    noise phase), and the run resumes there; otherwise it records the abandoned step
    as before. `low_storage_retention_refusal` is removed, and the scan worker no
    longer keeps every field under that march. solprop-kokkos still refuses the march.
  - *The gain span*: `σ = q + 1` for every `Multistep()`, both formulations, as §4.5
    states; 1 for the Runge-Kutta and low-storage marches.
  - *The warning band* names the admissible step at nine tenths of the limit; the
    refusal names it at the limit, and the admissible rate of the binding setting
    (`evanescent_rate`, the collar's strength or the spectral rate) when there is one.
- **Verification:** on the branch, Apple silicon (18 cores, a busy machine), with the
  Kokkos CPU wheel rebuilt at schema 31 and installed in the worktree; targeted runs
  only.
  - *Tests:* the new and changed modules and every module that exercises the seam --
    `tests/translate`, `tests/driver`, `tests/output`, the backends' driver, protocol,
    memory, low-storage, far-field, free-space, line, planar, spatial-execution, kernel,
    batch and collar modules, `tests/test_boundaries.py`, `tests/test_collar_length.py`
    and `tests/test_far_field_statistics.py` -- pass (4 436 in one run under xdist, with
    the Kokkos CPU build; the one failure there, the spatial contract's frozen field
    list, was then rewritten for schema 31, and the 15 collection errors of
    `test_radial_absorber.py` in that combined invocation do not reproduce when the
    module runs, 113 passing with the contract test). The doctests of every changed
    module pass (76), with the Kokkos package imported first. A targeted coverage run
    (`--cov-config=coverage-cpu.ini`, the compiled suite above plus the kernel bodies
    interpreted) covers every line and branch the branch adds under `src/`.
  - *Bounds, fixed before the runs:* a zero rate's element equals today's exactly on
    the reference formula, `PLAIN` and the compiled `HOST` bodies; the separable values
    with a zero collar equal the default's exactly; the noise phase with and without
    the collar is equal exactly (the rate is real); each column's attenuation over a
    linear vacuum run is within `γ_{3n} + ½ max α⊥ L ε` of `e^{-α⊥L/2}` (adaptive and
    two replayed schedules); the three backends agree on ten replayed steps carrying
    the rate within the Cartesian ladder's `amplitude_bound`; the transverse channel
    is within `2γ_n` of each column's `α⊥` power and the channel sum within its bound
    of `-dU/dz`; the aliased product within 10 % of `R` (measured 3 × 10⁻⁴); the
    held-plane resume within §4.5's bound of SEP 24 (as its kept-plane resume); the
    search's extremes equal the brute force's exactly, and the physical rate equals
    the brute-force maximum exactly.
  - *Lint:* `ruff check` and `ruff format` (`poe fmt` leaves every file unchanged),
    `basedpyright` of `src` and `tests` (0 errors), the numpydoc lint and `codespell` of
    every changed file are clean; the public-tree scanner finds nothing on any line
    either branch adds.
  - *solprop-kokkos:* `scripts/check.sh` passes (ruff, reuse, pyright, clang-format, the tidy and sanitizer
    builds with their native tests, 323 Python tests, the distribution wheel and its
    isolated smoke test); the CPU wheel built from the branch is installed in the
    solprop worktree (the later commits there are formatting and tests).
  - *Docs:* `poe docs` (`-W -n --keep-going`) builds clean.
  - *Benchmarks* ([ledger](../../benchmarks.md), ten-step windows): the collar's term costs 1.014 and 0.994 times a step on `numba_cpu`
    and Kokkos CPU (A/A 0.995 and 0.998), within the spread; a field-free observation
    takes 0.011 ms against 64 ms (`numba_cpu`) and 103 ms (Kokkos, whose host peak of
    81 MiB includes the device copy) with the field, on 128 × 128 × 256 columns (one
    state 64 MiB) at a load average of about 7.
  - *At review, on Linux:* `poe test-slow` (147 passed, 1 skipped) with the
    schema-31 wheel built from the solprop-kokkos branch; `poe cov` at 100 %;
    `poe check` and the Kokkos-selected default and slow lanes (148 slow passed). The
    lanes' failures were fixed at review and the failing modules re-run green: a
    refusal test that still wrote the previous contract version, and an order
    dependence in which the `configure_logging` example left the package logger at
    WARNING for later tests in the same worker. The import-time budget and the benchmark
    suite's collection timed out once on a heavily loaded node and passed on re-run.
    The review also kept the original exception when the held plane cannot be
    observed (with a test) and recorded the radial budget gap (Q16).
  - *Not run:* GPU builds, large-shape measurements (phase 3).
- **Sources checked:** `ln(DBL_MAX) = 709.78` from IEEE 754-2019 binary64 (§4.5's [6]),
  as `numpy.finfo(float).max`; Orszag's two-thirds rule cited on the Cartesian page
  ([1], whose DOI §13 resolved). The `Im √z` monotonicity of §4.4 was rederived; the
  worked numbers on the manual pages (7 × 10⁶ m⁻¹ at 1.1 µm on a 0.4 µm glass grid,
  `e^{740}` and 96 µm) are §4.4's and §4.5's, recomputed from those formulas.
- **Limitations:**
  - The spatial collar diagnostic is not evaluated on field-free planes; a native
    reduction would be a catalogue addition (Q2).
  - The exponent budget is checked only where the operator is separable (Cartesian and
    line). A radial run accepts `evanescent_rate` without the check; at review a radial
    replay at an exponent of about 3992 stayed finite over one `numpy_ref` step, so the
    radial path may not form the overflowing factor, but that is not shown (Q16).
  - At review the failure path was made to keep the original exception when the held
    plane cannot be observed: the step is then recorded as abandoned, without a
    checkpoint, and a test covers it.
  - The held plane is in-tree only: solprop-kokkos refuses the low-storage march until
    SEP 24's phase 3, which must hold the plane on the device or copy it.
  - Lockstep batches still observe every member's field (Q9).
  - `rate_bounds` bounds the loss by a per-frequency sum of maxima over the columns;
    a run whose largest attenuation and largest collar rate sit on different columns
    is refused slightly before its true limit (by at most the smaller of the two).
- **Open questions:**
  - Q15 — Whether SEP 24's phase 3 holds the low-storage march's field-free plane on
    the device (one more device volume) or copies it to the host at each field-free
    plane (one transfer, which phase 2 removed elsewhere). Destination: SEP 24 phase 3.
  - Q16 — Whether the radial kind needs the exponent budget: show that its path never
    forms an overflowing inverse factor, or bound its operator as the separable one is.
    Destination: phase 3.
- **Handoff:** phase 3 writes the tutorial (a Cartesian four-wave-mixing delay scan
  through a runner, statistics-only planes, a custom spectral absorber and a transverse
  collar, with the aliased-product case and a refinement), the large-shape benchmarks
  (per-plane observation with and without fields, the runner, the separable element),
  the design-page review and the closing gate with the slow, coverage and Kokkos
  lanes. The kernel schema is 31 and the output schema 20; solprop-kokkos `main` needs
  this phase's branch merged with solprop's. Q13 (`fields=` on the keyword
  interfaces) stays open for phase 3.

### Phase 3 report — documentation and scale

- **Scope:** phase 3 of §11 (documentation and scale), 2026-10-19, Opus 5.5 (AI) for
  John Travers, branch `feature/sep-0022-phase-3` from `main` 8674f98 (kernel schema 32,
  output schema 20). No schema moves; solprop-kokkos is unchanged. Merged with `main` at
  3d2d1f7 (SEP 23 phase 2, no conflict) and into `main` on 2026-10-19 by the merge commit
  that follows the commit carrying this report.
- **Delivered:**
  - *Q16, the radial exponent budget:* `translate/radial.py`
    (`radial_rate_bounds`), `translate/problem.py` (`_check_exponent_budget` bounds the
    radial kind; the budget messages name the operator's own attenuation per kind);
    tests in `tests/translate/test_operator_settings.py` (the constant table's exact
    extremes, refusal, warning band and admission, a finite run inside the budget, the
    non-finite stage a backend forms past it, and the bounds of a varying medium as an
    expansion and as a z table against a sampled rate).
  - *The tutorial:* `docs/tutorials/free_space/plot_06_four_wave_mixing_scan.py` and
    the gallery's `README.rst`: a Cartesian (150 × 16 columns) degenerate
    four-wave-mixing delay scan through `Simulation.runner()` with `fields="none"`,
    `region_spectrum` boxes for the beams and signals, `custom_absorber` on a
    spectral-only collar and a `TransverseCollar`; the correlation width against the
    closed form `T sqrt(3/2)`; the aliased product on a 7.4 µm grid with and without the
    collar beside the 2 µm grid; a three-grid refinement; and the absorber's exact
    transmission on a weak input. About 10 s at a load of 12.
  - *Design and manual pages:* [architecture §13](../../architecture.md#13-memory-discipline)
    (field-free observation, the held plane, the runner's admission and the digest
    record), [kernel contract](../../kernel-contract.md) §3's budget paragraph (the radial
    bounds), [backends](../../backends.md) §8 (the `output` and `held_plane` admission
    groups, the runner) and §9 (a row for this SEP's features), the
    [free-space model](../../../guide/free-space/model.md)'s budget paragraph, the
    [roadmap](../../roadmap.md) (this SEP's entry; new items for runner reuse and
    field-free members in scans, the compiled backends' extra volumes in the estimate,
    a custom rate apart from the collar's; the keyword-interface item widened to the
    retention keywords), the [benchmark ledger](../../benchmarks.md) and `CHANGELOG.md`.
  - *Benchmark:* `benchmarks/sep22/phase3_costs.py`.
- **Deviations from the proposal:**
  - *The tutorial's shape.* §8 asks for "a small Cartesian" scan; it is 150 × 16 columns
    and 128 bins, the smallest grid on which the signal itself meets the one-half rule
    and the refinement settles, and it sets `backend="numba_cpu"`, because Kokkos does
    not yet form `region_spectrum` (SEP 23's phase 3).
  - *The roadmap item.* *The collar diagnostic on Cartesian saved planes* was already
    removed by phase 1; this phase rewrites the SEP's roadmap entry and moves its
    remaining follow-ups there.
- **Decisions:**
  - *Q16 — the radial budget:* the radial kind **can** form the overflowing inverse
    factor. Every backend forms `exp(-ΔΦ)` from the dense radial operator exactly as on
    the separable kinds: a replayed radial step of 0.1 mm with `evanescent_rate=1e8`
    (physical rate 3.1 × 10⁷ m⁻¹ on a 4 µm, 40-point grid) gave a non-finite error norm
    on `numpy_ref` and Kokkos and a division by zero on `numba_cpu` before this phase.
    Why the phase 2 review's case stayed finite was not reconstructed. The check is
    added: the bounds are read off the operator, exact for a constant table and
    sufficient for a varying medium (each interval's Chebyshev series bounded by
    `|T_j| ≤ 1`, for a z table and for an expansion's weights against the sign of each
    term's real part). The measured varying-medium bound is within 1 % of the sampled
    extreme. A radial refusal names the radial operator's attenuation as a whole (its
    cap, spectral rate and medium loss are not separable in `D`) and the admissible step.
    The modal kinds stay unchecked. Closed.
  - *Q13 — `fields=` on the keyword interfaces:* moved to the roadmap, folded into the
    memory-budget keyword item (*Memory keywords on the keyword interfaces*).
  - *Q15 — the held plane on the device:* answered by SEP 24's phase 3, which holds it
    in Kokkos's memory space (`_hold`/`_held_field`); closed.
  - *Q9 and Q12:* moved to the roadmap as listed above.
- **Verification:** on the branch, Apple silicon (18 cores, a shared machine), the
  Kokkos CPU wheel at schema 32 installed.
  - *Tests:* `tests/translate/test_operator_settings.py` (19 passed, five new); the
    radial, Hankel and budget selection of `tests/translate`, `tests/backends` and
    `tests/test_spatial_statistics.py` (516 passed under xdist, so no existing radial
    case is refused or warned); `tests/test_seps.py` (211 passed, 1 skipped); the
    doctests of the changed modules (13). A targeted coverage run covers every new line
    of `radial.py` and the budget's new branches but one, the modal `return`, which
    any modal run reaches in the full lane.
  - *Lint:* `ruff check`, `ruff format`, `basedpyright` (0 errors) on the changed
    modules, the tutorial and the benchmark; `codespell` and `git diff --check` clean;
    the public-tree scanner finds nothing on any line this branch adds.
  - *Docs:* `poe docs` (`-W -n --keep-going`) builds clean, executing the tutorial.
  - *Benchmarks* ([ledger](../../benchmarks.md), 384 × 384 × 128, ten replayed steps, load
    5 rising to 17): a kept plane costs about 60 ms beyond its statistics (3 to 6 % of a
    step) and the loop's host peak falls from 4.3 GiB to 0.6 GiB (`numba_cpu`) and from
    4.0 GiB to 33 MiB (Kokkos); a bare observation 6.2 and 20.7 ms with the field,
    0.01 ms without; the runner's per-input saving is inside the spread (−4.6 % and
    +1.9 %, construction 0.04 to 0.35 s against 11 to 18 s of stepping); the k⊥ term
    1.019 and 0.988 times a step (A/A 0.995 and 0.985).
  - *Closing lanes, on Linux with the schema-32 wheel, before the merge with main:*
    `poe check` (16 061 passed, 29 skipped), `poe cov` at 100 %, `poe test-slow` (147
    passed, 1 skipped) and the Kokkos-selected default (16 061 passed) and slow (148
    passed) lanes, all green; the related modules (650 tests) pass on the merged branch.
  - *Not run:* GPU builds.
- **Sources checked:** `ln(DBL_MAX)` as before (§4.5's [6]). The tutorial's nonlinear
  index (3 × 10⁻²⁰ m²/W) and index (1.45) are illustrative values of the order of a
  silicate glass's, not transcribed data; the correlation width `T sqrt(3/2)` of a
  Gaussian `∫ I₁² I₂` was rederived.
- **Limitations:**
  - The radial budget's varying-medium bound is sufficient, not tight: a strongly
    varying table may be refused somewhat before its true limit.
  - The modal kinds have no exponent budget; their losses have not reached it.
  - The tutorial runs on `numba_cpu` until Kokkos forms `region_spectrum`.
  - The runner's saving is not resolved on the CPU at this shape; device builds,
    where construction and planning cost more, are not measured.
- **Open questions:** none new. Q9, Q12 and Q13 are on the roadmap; Q15 and Q16 are
  closed.
- **Handoff:** the closing gate, the slow, coverage and Kokkos-selected lanes, and the
  move to `Final`. When SEP 23's Kokkos far-field statistics land, the tutorial's
  `backend="numba_cpu"` can be dropped. Device qualification follows separately.

## Part III — History

- 2026-10-16: Drafted.
- 2026-10-16: Reviewed by Opus 5.5 (AI); two blockers (the k⊥ collar's claim and the budget's scope) and eleven should-fix items resolved in the text.
- 2026-10-16: Coordinated with SEP 21. This SEP is SEP 21's transverse follow-up (its
  Q10). `custom_absorber` takes the collar by value, and `collar_length` is defined against
  whatever sets the temporal collar's strength. Compiling zero temporal rates away and
  `absorber=False` are handed to SEP 21, and schema versions are left to landing order.
- 2026-10-17: Restated for [SEP 25](sep-0025-rounding-level-reproducibility.md).
  Comparisons of separate runs become bounds: field-free planes against kept fields, the
  scan worker against `run_scan` and the runner against fresh runs (§1, §4.7, §7) are
  held to a rounding bound fixed before measurement on a replayed schedule and to the
  case's accuracy envelope against a tighter-tolerance reference when adaptive, and a
  zero k⊥ rate's run to such a bound. The estimating-planner restriction on the runner
  test is dropped. Structural contracts stay exact: replayed step histories and the
  Adams call ledger, the four restored pieces of backend state, a zero k⊥ rate's
  operator elements, the default `evanescent_rate` and `collar_length`,
  `custom_absorber`'s copied rates, configuration hashes and the stored-digest test.
- 2026-10-18: Revised against SEP 25 Final and SEP 21 as reviewed. SEP 25's output
  preallocation and planned-capacity admission (§2, §5.1, §6.1: a field-free run charges
  no field block, and plane counts come from `n_saved`), its kernel schema 28 (§6.2, §9:
  the next free versions are 29 and 19 at this writing) and its kernel work (the drafting
  timings of §3 and §6.1 predate it). SEP 21's phases and schema bumps restated (§3); the
  spectral-only collar's keyword left to SEP 21's Q16 (§3, §5.4, Q8); the `collar_length`
  coupling to SEP 21's R4 made explicit (§4.6). The reference's buffer count corrected
  to 13 (§6.1), re-measured on the current tree. Q11 (attributing the user's spectral
  rate) opened from SEP 21's review.
- 2026-10-18 (later): SEP 21 deferred; the zero-rate pass and the spectral-only keyword
  return to phase 1 with their measured cost (§1, §2, §3, §5.4, §6.3, §7, §11, Q8).
- 2026-10-18: Accepted (John Travers) after his final review; Part I frozen.
- 2026-10-18: In progress. Phase 1 (driver, output and absorber) implemented on
  `feature/sep-0022-phase-1` at output schema 20; its report is in Part II.
- 2026-10-18: Phase 2 (the seam, kernel schema 31) implemented on
  `feature/sep-0022-phase-2`; its report is in Part II.
- 2026-10-19: Phase 3 (documentation and scale, the radial exponent budget of Q16)
  implemented on `feature/sep-0022-phase-3`; its report is in Part II.
- 2026-10-19: Final (John Travers). Phases 1–3 merged with their closing lanes green; open questions moved to the roadmap in the phase 3 report; moved to the archive.
