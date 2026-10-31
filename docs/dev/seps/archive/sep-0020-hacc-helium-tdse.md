# SEP 20 — haCC helium TDSE target

| | |
|---|---|
| **SEP** | 20 |
| **Title** | haCC helium TDSE target |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-16; revised 2026-10-17); GPT-6 (review revisions, 2026-10-17); Opus 5.5 (re-review against the provider code, 2026-10-17) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-10-16 |
| **Resolution** | Accepted 2026-10-17 by the maintainer, after the human, AI and master re-reviews (branch `docs/sep-0020-accept`) |
| **Requires** | [SEP 12](sep-0012-tdse-nonlinear-response.md) (the TDSE response route and its `tdse_native` policy); [SEP 19](sep-0019-field-scale-quadrature.md) (field-scale radial rules) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers; Phases 1, 2 and 3 merged into main (`9017661`, `4144927`, Phase 3 from `feature/sep-0020-hacc-witness`; L1–L4 pass, 6 nodes selected; L5 fails and the 60 fs window is kept with a recorded exception, Q30); Phase 4 merged into main from `feature/sep-0020-hacc-segment` (L6: every gate passes for the rate, SAE, haCC(1) and haCC(5) segments; US$16.55 on one MI325X; SEP 20 total US$22.82) |
| **Manual pages** | [haCC helium response](../../../guide/media/tdse.md); [raw TDSE response archive](../../../guide/media/tdse.md); haCC sections of [TDSE configuration](../../../guide/media/tdse.md) and [TDSE in capillaries](../../../guide/media/tdse.md) |

## Part I — Proposal

### 1. Abstract

Add a second atomic model to SEP 12's helium TDSE response: the hybrid
anti-symmetrised coupled-channel (haCC) model of the whole two-electron atom
[1–3], in its two helium forms, haCC(1) (frozen He⁺ 1s channel) and haCC(5)
(frozen He⁺ n ≤ 2 channels). A model comes either from one of the provider's two
qualified presets, which carry a stable identity and a stated accuracy, or from a
custom basis for research, which is identified by its full definition and recorded
as unqualified. Both forms of haCC are selected through one record. haCC(1) is
expected to be the main target, being about eight times cheaper; haCC(5) is its
ionic-channel convergence partner, and its production use depends on how much its
runs cost after optimisation. The source policy is SEP 12's
`kerr_model="tdse_native"`, $J=s\,n_0(j_{\mathrm{tot}}-j_1)$ with shadow order one
only, but with $s=1$ fixed: haCC returns the closed current of both electrons of one
atom, so the SAE route's empirical electron-count multiplier has no meaning here and
is refused. Because nothing is rescaled, every haCC configuration is checked before
propagation against optmat's weak-field linear response of helium: a deviation above
10% warns, one above 30% (the range of gross errors) refuses unless explicitly
overridden, and the result is cached per configuration and provider revision, so a
preset pays for it once. The Kerr response is measured once per preset and reported,
not gated.
To save atomic steps, the atom runs by default only over the requested 60 fs, with a
matching source gate; an explicit duration overrides this for validation and
research. The response
runs on the field-scale radial rules of SEP 19, with a node count selected by a
small haCC(1) convergence check. A `tdse=` keyword on `Simulation.from_capillary`
and `prop_capillary` makes the maintainer's `run_short.py` capillary example (a
4.05 m helium capillary with a steady
fill peaking at 150 mbar and a measured 1.88 mJ, 800 nm
pump) runnable with either haCC model in place of its Kerr and plasma terms. Success
is judged by fake-provider tests of every record, refusal, check and seam path, then
a validation ladder on the real provider that ends with short segment runs near the
compression plane, on a rented GPU within a US$100 cap, comparing rate, SAE,
haCC(1) and haCC(5). The segment shows that the route works mechanically and gives
sensible optics; full-length runs are left to the maintainer, outside this SEP. The
provider contract solprop binds to is stated in §5.7. TDSE convenience runs save every
accepted optical step by default, with only the final position requested when no
save positions are supplied. An optional HDF5 sidecar preserves every atomic input
and returned response, including unsuccessful optical trials (§5.8).

### 2. Motivation and scope

**Use case.** The maintainer's capillary example, `run_short.py`, is today a
rate-based Kerr-plus-plasma propagation:

```text
fill = steady_fill("He", [0.0, L / 2, L], [5e-3, 150e-3, 5e-3], radius=a)
pulse = DataPulse.from_spectral_density(2 * np.pi * nu, P, -phi, energy=1.88e-3)
orun = prop_capillary(
    a, L, "He", fill, lambda0=800e-9, pulses=[pulse], modes=1,
    trange=60e-15, wavelength_limits=(9e-8, 1.6e-6), rtol=1e-5, n_save=501,
)
```

with $a = 265$ µm (a 530 µm bore), $L = 4.05$ m, a steady fill of 5 mbar at both
ends and 150 mbar at the centre port (pressures in bar at this interface), and a
measured pump spectrum with its phase. The maintainer wants the same physical
problem with the helium electronic response computed by haCC instead of optmat's
Kerr coefficient and a PPT-type rate. The SAE `tdse_native` route of SEP 12 already
runs such a propagation, but its one-electron current has to be multiplied by an
empirical factor (1.537 in the selected research configuration) to stand in for
the atom. haCC removes that factor: it is a correlated two-electron model with
exact antisymmetry, ionisation into frozen ionic channels and no fitted parameter.

A rate-based parent of this case on `numpy_ref` with `Multistep()` at
`rtol=1e-2` and 51 saved planes reaches 4.05 m in 142 accepted Adams steps (order
one throughout, no rejections, 144 logical response calls), two of them over the
last 81 mm. Its peak power grows from 0.26 TW at the input to 0.59 TW at the last
plane, where the on-axis peak intensity is $8.0\times10^{14}$ W/cm² and the peak
ionised fraction about $5.6\times10^{-3}$. The largest 90–200 nm energy, 5.0 µJ, is
at 3.807 m. These facts set the field range the atomic model must cover and the
segment's expected call count (§6.7). The provider's presets were qualified on this
same parent's waveforms.

**In scope.**

- A haCC helium basis record and a model discriminator on `TDSEGasResponse`, with
  stable identities that distinguish haCC(1) from haCC(5), preset from custom, and
  both from SAE.
- A haCC preparation path in the driver's provider binding behind the existing
  `TDSEBinding` protocol, for presets and custom bases; source assembly is
  unchanged.
- A reduced atomic time window with a matching source gate (§4.6, §5.3), available
  to SAE and haCC alike.
- A cached weak-field linear consistency check of every haCC configuration before
  propagation, which warns, refuses only gross errors and can be skipped (§4.9, §5.4),
  and a once-per-preset Kerr measurement reported as evidence (§4.8, §7.2 L3).
- Refusal of every policy and scale that has no meaning for a whole-atom current; a
  run-time cost warning for many-node rules; domain flags for runs beyond a preset's
  validity domain; and an unqualified-model warning for custom bases, including
  research controls without the escaped-electron closure (§4.3).
- Save every accepted optical step by default through the new TDSE capillary
  entry point, using the existing natural-step output mechanism (§5.8).
- An optional, append-only archive of every atomic evaluation's input and complete
  provider response, for SAE and haCC, linked to the optical output (§5.8).
- Descriptor and output schema versions for the new records.
- The `tdse=` and `tdse_execution=` keywords on the capillary entry points, for
  any `TDSEGasResponse` (SAE and haCC alike).
- A documented `run_short` variant script for the maintainer's later full runs,
  exercised in this SEP only over the segment.
- The validation ladder of §7.2 under `benchmarks/sep20/`, ending with the
  segment, with the haCC segment runs, and compute-heavy atomic checks where a GPU is
  much faster, on rented GPUs within the US$100 cap (§6.7).

**Out of scope.**

- **Full-length runs**, their cost, hardware and comparison. The maintainer runs
  them later, outside this SEP, with the entry point and script it delivers.
- The provider's model, closure and presets themselves (§3.3). solprop states what
  it requires; it does not implement or qualify atomic physics, and it does not
  qualify custom bases.
- `optmat_cubic` and `tdse_shadow` for haCC (they need the third order of the
  whole-atom current, which the provider refuses, §4.3).
- A haCC(5) radial-node study: haCC(5) uses the rule and node count selected for
  haCC(1) (§7.2 L4).
- An optical-tolerance study (`rtol` ladder) for haCC: it belongs with the full
  runs.
- A run-time consistency check for SAE responses, whose deviations from optmat are
  known to be 50–80% (SEP 12 §4.4) and are handled by its explicit scale.
- Argon or other haCC/TDCIS targets, mixtures, preionised gas and double
  ionisation.
- `numba_cpu` and Kokkos execution of TDSE responses; they keep refusing by name
  (§6.6).
- Multimode or tapered capillaries, and any change to the SAE route's science.
- Wavefunction trajectories or checkpoints at every internal atomic RK4 substep:
  these are not returned by the response-plan API and need a separate provider design.

### 3. Context

#### 3.1 The TDSE route today

SEP 12 delivered a scalar, one-mode helium TDSE response on `numpy_ref` with
`Multistep()` (manual: [TDSE response](../../../guide/media/tdse.md),
[declaring a TDSE response](../../../guide/media/tdse.md)). The pieces
this SEP extends are:

- `solprop.nonlinear.tdse`: the provider-free records `TDSEBasis`,
  `TDSECalibration`, `TDSESampling`, `TDSEExecution` and `TDSEGasResponse`. The
  basis describes an SAE atom (finite-element edges and orders, `lmax`, potential
  identity, optional complex scaling); its `identity` is the SHA-256 of its
  canonical JSON.
- `solprop.translate.tdse`: `TDSEDescriptor` (canonical configuration JSON,
  schemas `solprop.tdse-descriptor.v1` and `v2`, the latter for
  `source_filter="time_collar"`), `TDSEResult` (owning, read-only per-atom
  currents), the call budget and the `TDSEBinding` protocol. The atom runs over the
  whole refined fine lattice, from its first to its last sample.
- `solprop.translate.tdse_source.admit_tdse`: admits the scalar HE₁₁ helium route
  and decides which ordinary responses remain (none for `tdse_native`).
- `solprop.driver.tdse`: the only module that imports the optional provider. It
  checks the provider's build revision against a pin, resolves SAE presets
  (`basis_from_preset`), and `_ProviderBinding` prepares the atomic model, one
  response plan per distinct chunk size, and evaluates chunks with transactional
  call accounting: each chunk is charged before invocation, and failures are never
  refunded. Its diagnostic checks read the SAE full record's bound from
  `raw.plasma` and require float64 monitors apart from an int64 step index.
- `solprop.backends.numpy_ref.tdse.atomic_sources`: forms
  `remainder = total − first`, applies the optional time collar and then
  `density * remainder_scale * remainder`, with zero polarisation for
  `tdse_native`. It reads only `TDSEResult` arrays and descriptor scalars.
- `solprop.driver.simulation`: a TDSE run is built today in two steps, a
  `Simulation.from_capillary` parent followed by `Simulation(grid, modes,
  [response], pulse, settings=replace(..., tdse_execution=...))`. The helpers
  `_resolve_tdse_grid` and `_check_tdse_execution` refine the response lattice and
  refuse unsupported settings (non-NumPy backend, non-Adams integrator, noise,
  fixed schedules, interpolated planes, other absorber schemes, spatial or general
  routes).
- `solprop.physdata.species`: optmat helium's per-particle linear susceptibility
  (m³) over its declared support and its per-particle Kerr coefficient
  $\gamma_3$ (m⁵/V²) at a carrier wavelength, the references of the consistency
  check.

For run_short's grid, `FieldGrid.from_limits` with `trange=60e-15` gives 1024
coarse samples of 100.07 as and 2048 fine samples of 50.03 as: a 102.5 fs lattice
from −51.2 to +51.2 fs. Its temporal collar is one over the requested ±30 fs and
tapers to zero over the remaining 21.2 fs at each end. The atom therefore runs over
102.5 fs today, although only 60 fs were requested.

The selected SAE research configuration is `kerr_model="tdse_native"`,
`remainder_scale=1.537`, `source_filter="time_collar"`, a 400-hartree cut and
`lmax=29`; its evidence and open window, filter and tolerance questions are in the
native-current qualification (recorded in the development archive) and on the
[roadmap](../../roadmap.md).

#### 3.2 Field-scale radial rules

SEP 19 added `radial_rule="field_scale"` (Gauss in $\eta=J_0(u_{01}r/a)$, $n_r\ge1$)
and `"field_scale_radau"` (with an axis node, $n_r\ge4$) for one Marcatili HE₁₁ mode
([radial nodes for expensive responses](../../../guide/waveguides/radial-quadrature.md)).
Its dense SAE scan found the TDSE source finely structured in η, with 0.25–1.5% source
errors for 5–8 nodes depending on node placement, and sub-percent errors at 16. It
states that all-electron responses need their own node-convergence study; §7.2 L4 is
a minimal one for haCC(1).

#### 3.3 Provider deliverables

The atomic provider is the optional `tdse` extra (Pyroraptor). The three
deliverables this SEP depends on are integrated in the provider at revision
`b8fc0c16` and later:

- **(a) The qualified whole-atom `total_current`.** For helium with a mixed gauge,
  the provider's `total_current` is an escaped-electron closure: it counts each
  electron of the atom once, including the momentum carried by electrons that leave
  the unscaled region, and `info.total_current_qualified` is true. For argon and the
  velocity-gauge control it is false. The provider reports a
  `neglected_current_bound`, which is a loose, partly empirical estimate (§4.4),
  not an accuracy bound.
- **(b) The order-one shadow of that current in the compact output.**
  `ResponsePlan(shadow_order=1, outputs="total_current")` returns
  `shadow_orders == (1,)` and `shadow_total_current[:, 0]`, the in-box
  first-order current. Its use as the closed current's first order has the
  finite-tail qualification of §4.3 and Q15; order three is refused.
- **(c) Two qualified presets**, `hacc.helium_preset(1)` (haCC(1)) and
  `hacc.helium_preset(2)` (haCC(5)), with stable identities, stated accuracy and a
  validity domain, qualified on the run_short field bundle (§4.4). Any other model
  is built with `hacc.prepare(atom, basis, gauge=..., cut_energy=...)`.

§5.7 states the contract in solprop's terms. Phase 1 moves the solprop pin to a
revision containing all three and records it.

### 4. Theory

#### 4.1 The haCC helium model in brief

Within the electric-dipole, nonrelativistic, fixed-nucleus approximation, one
helium atom driven by the local real field $E(t)$ along the polarisation axis obeys

$$
i\hbar\partial_t\Psi(\boldsymbol x_1,\boldsymbol x_2,t)=\Bigl[\sum_{i=1}^{2}
\Bigl(\frac{\boldsymbol p_i^2}{2m_e}-\frac{2e^2}{4\pi\epsilon_0|\boldsymbol x_i|}
\Bigr)+\frac{e^2}{4\pi\epsilon_0|\boldsymbol x_1-\boldsymbol x_2|}
+H_{\mathrm{int}}(t)\Bigr]\Psi,
$$

with $\Psi$ antisymmetric under exchange and normalised to one atom. haCC [1]
expands $\Psi$ as a correlated two-electron neutral, confined to a finite region,
plus antisymmetrised products of frozen He⁺ eigenstates with an active-electron
function on a radial finite-element axis. The ionic content is the model choice:

| Model | Frozen He⁺ channels | Ionic states |
|---|---|---|
| haCC(1) | 1s | 1 |
| haCC(5) | n ≤ 2: 1s, 2s, 2p₀, 2p₊₁, 2p₋₁ | 5 |

haCC(5) adds ionic excitation and part of the core polarisation. The difference between
the two measures sensitivity to adding these four channels, not the full ionic-channel
truncation error: both omit higher bound channels and the ionic continuum [1, §4.1].
Agreement cannot by itself certify either model. The segment of §7.2 L6 measures this
sensitivity on the observables that matter here.

The interaction uses a mixed gauge [2]: length gauge inside a gauge radius
$R_g$ and velocity gauge outside, because frozen ionic functions in a pure velocity
gauge give wrong ionisation [2]. The active electron is absorbed by
infinite-range exterior complex scaling (irECS) beyond a radius $R_0$ [3]. A
spectral cut removes field-free roots above a cut energy. Gauge, cut, $R_0$,
$R_g$, radial lattice, angular truncation and the neutral's correlation content
are basis choices. Both presets share them: radial edges
$[0,0.5,1,2,4,8,\ldots,80]$ bohr with element order 10 (208 inner radial nodes),
$R_0=80$ bohr, irECS angle 0.15 rad, infinite-element order 20 and decay $0.5/a_0$,
particle $l_{\max}=50$,
neutral square 6, $R_g=20$ bohr, a 200-hartree cut and an RK4 step of at most
0.0088 au; only the ion count differs. A custom basis may choose any of them within
the provider's admitted ranges. All of them enter solprop's record and its identity
(§5.1).

The in-box observables use the conjugated pairing in the unscaled region $|\boldsymbol
x|<R_0$. The closed current also carries the history-dependent exterior momentum
contribution (§4.4); it is not just an in-box integral. Each Adams source evaluation
restarts every atom from the prepared ground state, as in SEP 12 §4.1: each $z$
contains different atoms.

#### 4.2 The whole-atom current, and why $s=1$

The macroscopic current density of a gas of $n_0(z)$ atoms per m³ is
$J=n_0\,j$, where $j$ in C m/s (A m) is the current of **one atom**: the
expectation value of $-e\sum_{i=1}^{2}\dot z_i$ in the atom's state. For SAE this
sum has one term, computed for one active electron in an effective potential; the
second electron is absent and the potential is a fit. SEP 12 therefore carried an
explicit `remainder_scale`, with the hypothesis 2, the rate-fit factor 1.47 and the
selected weak-current factor 1.537 as candidates. None is an electron count; each is
an empirical multiplier that makes a one-electron current stand in for an atom, and
absorbs the SAE potential's error in the same number.

haCC's state is the two-electron state of one atom, and its current observable is
already the sum over both electrons. The exact, complete two-electron theory satisfies
the Thomas–Reiche–Kuhn sum rule with oscillator-strength sum two [7]. A finite haCC
channel space and spectral cut need not satisfy it exactly, so the sum rule is a
convergence diagnostic, not a proof of this model's linear accuracy. Ionisation removes
one of two indistinguishable electrons into an ionic channel; its probability is in the
amplitude, not in a count. Consequently:

- there is an active-electron basis in the ansatz, but the current already sums both
  indistinguishable electrons; applying an electron-count multiplier to that observable
  is unjustified ($s=2$ doubles the whole atomic response);
- the residual model error of haCC (ionic truncation, correlation content, basis)
  is not an electron count and must not be absorbed into $s$; it is measured
  (§4.8, §4.9, §7.2) and reported;
- the number density $n_0$ is the atom density from the existing equation of state,
  as on the SAE route. No neutral fraction or electron density multiplies a current
  that already contains depletion.

So for haCC, `remainder_scale` is fixed at exactly `1.0`, validated, and recorded;
it is not a tunable parameter, for presets and custom bases alike. Keeping the field
(rather than removing it for haCC) keeps one source formula for both models and
makes the value visible in every record.

#### 4.3 The source: first-order subtraction only

The source is SEP 12's native policy with $s=1$:

$$
J_{\mathrm{NL}}(r_q,t)=n_0(z)\,g(t)\,\bigl(j_{\mathrm{tot}}[gE_q](t)-j_1[gE_q](t)\bigr),
\qquad P_{\mathrm{NL}}=0,
$$

where $j_{\mathrm{tot}}$ is the provider's compact `total_current`, $j_1$ is
`shadow_total_current[:, 0]`, and $g$ is the source gate of §4.6. For a windowed
response both currents are evaluated on the **same tapered drive** $gE_q$.
The density and output gate are applied only after subtraction. Linear dispersion stays in optmat's
$D_m(\omega,z)$, so any surviving order-one term competes with the much smaller
nonlinear source (SEP 12 §4.3).

**Finite-basis qualification of the shadow.** The provider returns the in-box
first-order shadow. The closure includes momentum transport through the **closure
surface** $R_c$ (16 bohr for the presets) as well as field acceleration of the exterior
population, so the shadow equals the closed current's first order only up to terms that
need the states' amplitudes at $R_c$; inversion symmetry alone does not remove them. The
provider's evidence (Pyroraptor report 0126, §B.2 and §B.3; Q15, resolved) is:

- **Derivation.** Write the state as $x_0+\lambda x_1+\cdots$ with $x_0$ the field-free
  root. Every first-order term of the closure beyond $j_1$ is bilinear in $(x_0,x_1)$ and
  involves either their values at the nodes adjacent to $R_c$ or their parts beyond
  $R_c$; $N_0-N$ has no zeroth- or first-order part. Hence
  $J^{(1)}=j_1+O(\|(1-\Theta)x_0\|\,\|(1-\Theta)x_1\|)+O(|x_0(R_c)|\,|x_1(R_c)|)$.
  The ground root decays as $e^{-\kappa r}$ with $\kappa=\sqrt{2I_p}\approx1.345/a_0$,
  so its amplitude at $R_c=16$ bohr is about $e^{-21.5}\approx5\times10^{-10}$. (Report
  0126 quotes $2\times10^{-12}$, the value at its prototype's 20 bohr surface.)
- **Measurement.** On the run_short peak-power field scaled by 0.01, with haCC(1) at
  preset-like numerics, $|J_{\rm closure}-j_{\rm in}|$ was $5.7\times10^{-12}$ of
  $\max|j_{\rm in}|$, while the third-order remainder was $2.8\times10^{-5}$: the linear
  leakage is below $2\times10^{-7}$ of the nonlinear source at that amplitude, far weaker
  than any retained node.
- **Regression test.** The provider's `test_order_one_is_the_in_box_shadow` checks the
  odd part against the shadow for haCC(1) and haCC(5) with $R_c=16$ bohr on a small
  basis.

What remains is a preset-scale haCC(5) figure. L3's amplitude ladder supplies it without
an extra run: the odd part's linear term of $j_{\rm tot}-j_1$ across $\lambda\in\{1,2,4\}$
is reported beside its cubic term (§7.2 L3). A custom basis whose frozen ionic functions
or ground root have appreciable support at $R_c$ (a small $R_c$, or more diffuse ionic
channels) is not covered by this evidence; `total_current_qualified=True` only identifies
an available closure, and such a basis is unqualified anyway (§5.6).

The third order is different: the escaped term can contribute at third order
where one-photon loss is possible, so $j_{\mathrm{tot}}-j_1-j_3$ would need the
third order of the closed current, which the provider refuses. The `optmat_cubic`
and `tdse_shadow` policies, which subtract $j_3$, are therefore refused for haCC.
They are retired research options on the SAE route too, and nothing in the
run_short use case needs them. The haCC source needs only the provider's compact
products; full diagnostic archiving uses the same currents from the full record,
which has no separate plasma record (§5.8).

The closure exists only for helium with a mixed gauge whose $R_g$ has an element
edge one element inside it; the provider reports it through
`info.total_current_qualified`. Both presets have it. A custom model without it (the
velocity-gauge control, or a gauge radius that does not admit the closure) has no
qualified whole-atom current, but it is a legitimate research control, so it is
admitted with a `TDSEUnqualifiedModelWarning` saying that its total current lacks the
escaped-electron closure, and recorded with `closure: false`. Its currents come from
the provider's full output, because the compact `total_current` mode with
`shadow_order=1` is refused without the closure (§5.7 C3):

- $j_{\mathrm{tot}}$ is the full record's `total_current`: the in-box current plus the
  field impulse on the escaped norm, $\int E\,(N_0-N)\,dt$. It omits the momentum
  carried out through $R_0$, which is the reason the closure exists (§4.4);
- $j_1$ is the in-box first-order current `shadow.component("current", 1)`; the
  full record has no total-current order without the closure. Because the field-impulse
  term is at least third order, $j_1$ is also the first order of this $j_{\mathrm{tot}}$;
- the record has no `neglected_current_bound` (the provider returns `None`); solprop
  records it as absent rather than as zero.

The same source formula, $s=1$ and all other rules apply; the escaped-electron current
of such a run is known to be incomplete, and the record says so.

#### 4.4 What the closure includes, its accuracy, and what it omits

Included: bound linear and nonlinear response of the correlated atom (the cubic
and all higher orders), excitation, single ionisation into the declared ionic
channels, the current of escaped electrons through the closure (including their
acceleration by the field), and depletion of the neutral. No separate Kerr, plasma,
Brunel or ionisation-loss term is added (SEP 12 §4.3).

**Accuracy of the presets.** The presets state it from three-level convergence
ladders on the run_short peak-power waveform (8.0×10¹⁴ W/cm², node 0), as the
largest relative change of the closed current, $\max_t|\Delta J|/\max_t|J|$, up to
8 fs after the pulse centre:

- `accuracy` = 1.22% for both presets (lmax, finite-element order, cut and step;
  the finite-element order rung is the largest);
- `box_sensitivity` = 0.95% (haCC(1)) and 0.93% (haCC(5)), across $R_0$ = 80, 120
  and 160 bohr: the closure's recrossing residual, about 1% of the peak per escape
  burst;
- `gauge_sensitivity` = 0.11%, across $R_g$ = 12, 20 and 28 bohr.

For haCC(1), the source $j_{\mathrm{tot}}-j_1$ changes by at most 1.02% over
the same ladders. These are measured sensitivities of the sampled preset configuration, not error bounds on the
exact helium response or on a propagated UV spectrum. The source and shadow sensitivity
evidence is complete for haCC(1); haCC(5)'s non-base ladders did not propagate the
shadow. The provider also reports an haCC(1) irECS-angle ladder (0.10, 0.15, 0.20 rad)
with total-current changes of 1.9% and 0.58%; no corresponding haCC(5) angle ladder was
run. Keep these qualifications beside the provider's `accuracy` field. A
custom basis has no stated accuracy: solprop records it as unqualified and warns
(§5.6).

**The neglected-current bound is a diagnostic.** The closure transports momentum through
$R_c$, includes the outer-electron Coulomb force from $R_c$ to $R_0$, and integrates the
field force on electrons beyond $R_c$. It omits some nonlocal transport from finite
frozen-state tails; after escape beyond $R_0$ it also drops the residual ion's current
for haCC(5). The provider's
`neglected_current_bound` estimates the exterior Coulomb force, the recrossing of
$R_0$ (with an empirical coefficient) and, for haCC(5), the ion's own current after
the electron leaves. On the qualifying fields it is 15–55 times the observed box
difference and reaches about 0.21 (haCC(1)) and 0.86 (haCC(5)) times the peak
current by 8 fs, and up to 1.6 times in the provider's other closure runs. It also
omits the change of model with $R_0$ before escape (1–5×10⁻⁵ of the peak). solprop
therefore records its absolute maximum and its ratio to the peak current per row, and
never refuses on its magnitude. For a zero-current row the ratio is `null` with an
explicit `zero_reference` flag; neither division by zero nor an infinite JSON value is
admitted.

Omitted, and stated as limitations:

- **double ionisation and ionic channels beyond n = 1 (haCC(1)) or n = 2 (haCC(5));**
- **macroscopic effects** outside the scalar one-mode model: ion motion,
  collisions, recombination, space charge, and transverse modes other than HE₁₁;
- **the field outside the atomic window and in its tapers** (§4.6);
- **the presets' validity domain.** The presets are qualified for helium driven by
  an 800 nm linearly polarised field whose spectrum stays below the helium
  threshold (photon energies up to about 20 eV; run_short's 90 nm grid edge is
  13.8 eV), at peak intensities up to $8\times10^{14}$ W/cm² (`max_peak_intensity`),
  over an atomic-window duration up to the preset's `window`. Today the ladders run
  from the start of the 102.5 fs bundle, 51.2 fs before its centre, to 8 fs after it,
  a 59.2 fs `window`. Pyroraptor's haCC timing study Part D re-runs the haCC(1) ladders over
  ±30 fs and updates `window` to the tested duration, about 60 fs; haCC(5) carries it
  over without its own ladder, which the provider states. Each further escape burst
  beyond the tested time adds about 1% of the peak through recrossing. The TDSE run's own compression can exceed the parent's
  $8\times10^{14}$ W/cm². Runs beyond the domain are flagged and warned, not refused
  (§5.6).

#### 4.5 Radial integration on field-scale rules

For one HE₁₁ mode in a gas uniform across the core, every node's field is the
on-axis field scaled by $\eta_q=J_0(u_{01}r_q/a)$ (SEP 19 §4.1). The projected source
is

$$
N_1(\omega)\propto\sum_q w_q\,\eta_q\,\widehat{J}_{\mathrm{NL}}[\eta_qE_{\mathrm{axis}}](\omega),
$$

where $w_q$ denotes the physical area quadrature weight. For the Gauss weights $W_q$ of
SEP 19's measure $d\mu=\eta\xi\,d\xi$, with $\xi=r/a$, the same expression is
proportional to $\sum_q W_q\widehat J[\eta_qE_{\rm axis}]$: the projection factor η is
already in that measure and must not be applied twice. This is an integral over η of an atomic response evaluated at $n_r$ amplitudes. Each node is
one atom per call, so the node count is the main cost lever (§6.7). The haCC current
is expected to have the same fine structure in η as SAE's (it comes from
interference of ionisation pathways, which haCC also contains), so the node count is
chosen by measurement (§7.2 L4), not assumed. The node-convergence check needs no
propagation: nodal fields for any rule are $\eta_q E_{\mathrm{axis}}(t)$ from a saved
on-axis waveform. The η structure comes from the active electron's ionisation
dynamics, which haCC(1) and haCC(5) share; haCC(5) therefore uses the count selected
for haCC(1) without its own study, and the segment's haCC(1)–haCC(5) comparison is
read with that in mind.

#### 4.6 The atomic window and its taper

The maintainer requests `trange=60e-15`; the resulting FFT lattice is 102.5 fs long. In
this case the temporal collar is one over the requested interval $[t_-,t_+]$ (±30 fs for
run_short). In general its flat region is narrower: `grid._temporal_collar` uses $t_{\rm
flat}=\min(\mathrm{trange}/2,(T/2)(1-2f_{\rm collar}))$. The grid does not retain the
original request as a separate `trange` field. Thus neither "padding" nor the collar's
flat edges identify a physically empty region, and recovering an arbitrary requested
duration from those edges is ambiguous. The resolved duration therefore comes
from the explicit record or the capillary request (§5.3; Q20). The field in the run_short bundle is concentrated well inside the request. Its
support above 1% of the peak is −18.7 to +18.2 fs at the input, −20.5 to +22.0 fs at
the maximum-peak-power plane and −19.8 to +21.8 fs at the maximum-UV plane; at 0.1%
the peak-power field reaches +36 fs, and the maximum-UV plane's weak dispersive and
UV tail fills the whole grid.

The proposed haCC default drives the atom only over the requested window. This is a
finite-memory approximation, subject to L5, not a consequence of FFT padding being
physically irrelevant. With a taper width $w$ (default 5 fs), define the window function

$$
g(t)=\begin{cases}
0, & t<t_- \ \text{or}\ t>t_+,\\
P\!\left(\frac{t-t_-}{w}\right), & t_-\le t<t_-+w,\\
1, & t_-+w\le t\le t_+-w,\\
P\!\left(\frac{t_+-t}{w}\right), & t_+-w<t\le t_+,
\end{cases}
$$

with $P$ the same Planck taper the grid's collar uses. The atom runs over
$[t_s,t_e]=[t_-,t_+]$ from its ground state, driven by $g(t)E(t)$, and its
nonlinear current is gated by the same $g$ before the source transform:

- **What is approximated.** The field outside the window is dropped and the field
  in the tapers is reduced. With $w=5$ fs the tapers lie at $|t|$ from 25 to 30 fs,
  outside the 1% support of every plane above, so the drive is changed only where
  the field is below 1% of its peak. The maximum-UV plane's far tail beyond ±30 fs is
  truncated; §7.2 L5 and the segment (L6) show whether that matters. The atom's
  vector potential is integrated from zero at $t_s$.
- **Causal stopping.** For a fixed drive and identical atomic state at $t_s$, the
  current up to time $t$ is independent of later drive. Since $g=0$ after $t_e$,
  stopping there adds no error beyond the chosen output gate. Starting at $t_s$ in the
  ground state instead of propagating the earlier field is a separate approximation. A
  changed RK4 time origin or substep lattice can also change the discrete result, so
  causality alone is not a bitwise provider-parity guarantee. Record the tapered field's
  integral and vector-potential endpoints: multiplying by $g$ need not preserve a
  zero-area drive; do not silently remove a DC component.
- **What it replaces.** For a windowed response the gate $g$ is the source filter,
  the `time_collar` filter narrowed to the requested window; the drive taper is new
  and explicit. The returned raw histories are retained before source gating, but they
  belong to the tapered drive and can differ from the full-drive histories.

The nominal atomic duration is 60 fs against today's 102.5 fs: approximately 60/102.5 of
the steps at a fixed realised RK step. The exact duration is the span of the retained
sample endpoints, not the nominal width, and the measured saving also includes slicing,
zero padding and transforms.

**Coverage against the presets' ladders.** The presets' 59.2 fs ladder window ran
from −51.2 fs (the start of the bundle) to +8 fs after the centre. On a ±30 fs
atomic window, the part from −30 to +8 fs overlaps the sampled interval (with a different startup), but the post-peak part from +8
to +30 fs lies outside the ladders, and that is where escape bursts and their
recrossing residual (about 1% of the peak each) accumulate. Such runs are warned and
recorded (§5.6). The provider has scheduled the ±30 fs re-check, haCC(1) only, on free
CPU nodes, as Part D of its haCC timing study, and updates the presets' `window` to the tested
duration. The domain check compares durations, not intervals (§5.6, Q17): until 0132
lands, a default 60 fs run is flagged against today's 59.2 fs `window`; afterwards it is
not.

**Arbitrary grids and windows.** Any time grid, `trange`, atomic-window duration and
taper is admitted, provided the window lies within the optical lattice. The preset
domain flags are warnings and per-call records, never refusals, and a custom basis gets
none. The presets' error accumulates after the pulse peak (recrossing of escaped
electrons), so the meaningful coordinate is time relative to the pulse; an interval on
the qualifying grid would mean nothing for other grids, pulse positions and shapes.
Duration and peak intensity are honest, simple proxies. They do not establish accuracy
for pulses that differ in shape, timing or carrier.

#### 4.7 Response sampling

The provider interpolates the sampled field cubically to its Runge–Kutta stage
times (SEP 12 §4.8). For $E=\sin\omega t$ on spacing $h$, the interior cubic Lagrange
error is bounded by

$$
|E-\Pi_3E|\le\frac{(\omega h)^4}{4!}\max_{x\in[1,2]}|x(x-1)(x-2)(x-3)|
=\frac{9}{16}\frac{(\omega h)^4}{24}\approx0.023\,(\omega h)^4 .
$$

At the 90 nm band edge on run_short's fine lattice ($h=50.03$ as, $\omega
h\approx1.05$) this is about 2.8% of that component's amplitude. Refinement by
$q$ divides $\omega h$ by $q$: $q=8$ gives $7\times10^{-6}$ and $q=32$ gives
$3\times10^{-8}$. The bound concerns interpolation of one sinusoid, not the nonlinear
source's error. Refinement also changes the integer RK4 subdivision. The provider uses
$n_{\rm sub}=\lceil h/(q h_{\max})\rceil$ (`ResponsePlan`, at revision `b8fc0c16`), so
its realised step is $h/(q n_{\rm sub})$, and the step count is not monotone in $q$:

| $q$ | spacing $h/q$ | substeps | realised step | steps per 60 fs |
|---|---|---|---|---|
| 8 | 6.254 as = 0.2585 au | 30 | 0.008617 au | about 288,000 |
| 32 | 1.563 as = 0.0646 au | 8 | 0.008079 au | about 307,000 |

This SEP therefore prescribes `TDSESampling(8)` for haCC: about 6% fewer atomic steps
than $q=32$ and a quarter of the samples (9,600 per 60 fs instead of 38,400), at an
interior interpolation bound of $7\times10^{-6}$ at the 90 nm edge. The presets were
qualified at $q=32$, with a realised step of 0.00808 au; their step ladder covers the
change to 0.00862 au, which lies between that base and the 0.0088 au ceiling, so it does
not leave the qualified numerics. Record `info.substeps`, `step`, `steps` and timing
rather than assuming the atomic cost. Cubic interpolation's first and last stencils need
endpoint tests; the interior bound alone does not cover them.

#### 4.8 Weak-field consistency

Because $s=1$ is fixed by §4.2, haCC's weak-field responses are a check on the
model, not a calibration. Two dimensionless ratios compare them with optmat's
helium data. Only the linear one gates a run; the cubic one is measured once per preset
as evidence (2026-10-17 amendment of Q8):

- **linear (the gate):** with $\hat{\ }$ the internal Fourier transform, in which
  $\partial_t$ becomes $+i\omega$, the per-atom first-order dipole is
  $\hat p_1=\hat j_1/(i\omega)$ and the haCC per-atom susceptibility is
  $\alpha_{\mathrm{haCC}}(\omega)=\hat p_1/(\epsilon_0\hat E)$ in m³. Then
  $\kappa_1(\omega)=\alpha_{\mathrm{optmat}}(\omega)/\alpha_{\mathrm{haCC}}(\omega)$,
  evaluated where $|\hat E|$ exceeds $10^{-3}$ of its maximum and inside optmat's
  declared support. Its summary is the largest $|\kappa_1-1|$ over that band. Compare
  like Fourier conventions: a complex optmat susceptibility in the optics convention is
  conjugated for this internal-DFT comparison. Do not replace a complex ratio by its
  real part and conceal a phase or absorptive discrepancy. $j_1$ is the order-one
  shadow itself, so one row at one amplitude suffices.
- **cubic (evidence only, L3):** with an amplitude ladder $\lambda\in\{1,2,4\}$ of the
  same weak field and the remainders $r_\lambda=j_{\mathrm{tot}}-j_1$, write
  $r_a=a\,c_1+a^3c_3+a^5c_5+a^7c_7+\cdots$ for the fixed base field, where $c_1$ is any
  first-order leakage of the shadow (§4.3; expected to be negligible, since report 0126
  B.3 bounds it below $2\times10^{-7}$ of the nonlinear remainder at 0.01 of the
  peak-power field, for haCC(1)). Neglecting $c_1$, $\widehat c_3=(32r_1-r_2)/24=c_3-4c_7+\cdots$ and
  $\widehat c_5=(r_2-8r_1)/24$; the λ = 4 row checks that model, and the same three rows
  give the fit with $(c_1,c_3,c_5)$ instead, whose $c_1$ is the preset-scale leakage
  figure of Q15. The remaining bias is seventh order in the **base field amplitude**.
  The comparison is made in the frequency domain over the **fundamental band** only,
  the Kerr (self-phase-modulation) component that matters for propagation:
  $\kappa_3$ is the real least-squares scale with
  $\mathcal F[j_{3,\mathrm{ref}}]\approx\kappa_3\,\mathcal F[\widehat c_3]$ there, with
  $j_{3,\mathrm{ref}}=\epsilon_0\gamma_3\partial_t(E^3)$ and $\gamma_3$ optmat's
  per-atom Kerr coefficient at the field's carrier; it is reported with its relative
  residual.

**Why the cubic ratio is not a gate.** The synthetic field is a 4 fs, 1.5-cycle pulse,
and the instantaneous reference applies one carrier coefficient to the whole broadband
cubic response, including third-harmonic generation, for which helium's
$\gamma(-3\omega;\omega,\omega,\omega)$ exceeds $\gamma(-\omega;\omega,\omega,-\omega)$
by a few per cent at 800 nm. A time-domain fit would therefore carry an extraction bias
of a few per cent; with haCC(1)'s Kerr response about 5% below haCC(5)'s, the main model
could approach a 10% threshold for a reason that is not a model error. The cubic
response is already validated by the provider: haCC(5)'s
$\gamma(-\omega;\omega,\omega,-\omega)$ is within +0.19 to −0.01% and its
$\gamma(-3\omega;\omega,\omega,\omega)$ within +0.17 to −0.60% of Bishop and Pipin
(1989) for ω = 0.05–0.20 au, and haCC(1) is about 5% lower in the Kerr response. The
gross errors the run-time check exists to catch (a doubled or missing electron count, a
sign, a unit error) scale the whole current and appear in $\kappa_1$ alone; a density
error is outside any per-atom check. Restricting $\kappa_3$ to the fundamental band
removes the third-harmonic bias from the reported evidence.

For SAE these ratios were 1.764 and 1.537 (SEP 12 §4.4). For an ab initio
two-electron model they should be near one. Their deviation measures agreement with the
chosen reference and extraction protocol; it combines atomic truncation, numerical
extraction and reference-model differences. The ratios are reported and never applied
as scales.

**Thresholds.** A valid diagnostic (§5.4) with $|\kappa_1-1|\le0.1$ passes; between
0.1 and 0.3 the run proceeds with a warning; above 0.3 it is refused by default. The
10% warning does not trip on known, legitimate differences, and the 30% refusal is
reserved for gross errors, so a research basis that is merely less accurate is not
stopped:

- haCC(5)'s α(ω) is −0.94 to −0.97% from Liu (1997) for ω = 0.05–0.20 au in the
  provider's validity table, so the model alone uses little of the margin;
- the selected optmat model, its frequency dependence and its documented uncertainty
  must be recorded. A generic few-percent uncertainty must not be assigned to helium's
  linear polarizability: primary calculations can be much more accurate [8];
- the errors the refusal is meant to catch are large: a factor of two (an electron
  count applied twice, or a missing one, $|\kappa_1-1|\ge0.5$), a sign
  ($|\kappa_1-1|\approx2$), or a unit error. A 30% bound catches all of them and leaves
  room for exploratory bases.

#### 4.9 The run-time consistency check

The linear check runs before every haCC propagation, preset or custom, on a fixed
synthetic weak field that does not depend on the run's grid, so that its result
depends only on the atomic configuration and can be cached:

- **Field.** A Gaussian pulse at the run's reference wavelength $\lambda_0$, 4 fs
  intensity FWHM (the tested band is set by the Fourier-amplitude rule and the material
  support, not an assumed wavelength interval), peak intensity $10^{11}$ W/cm². It is
  sampled on a 40 fs window at a fixed protocol spacing, with the model's maximum atomic
  step.
- **Atoms.** One row, `shadow_order=1` and compact output (full when requested by the
  archive), built and closed by the check, within the execution limits (§6.5).
- **Cost.** One one-row call over 40 fs, about 188,000 RK4 steps at the presets' step.
  At the measured single-row CPU costs with the shadow (2.4 ms per step for haCC(1),
  19 ms for haCC(5), on 32 CPU threads) that is about 0.13 h for haCC(1) and 1.0 h for
  haCC(5) on CPU, plus model setup (0.03 h and 0.11 h), about 0.16 h and 1.1 h in all; on
  a GPU it is expected to be minutes. It is paid once per configuration and provider
  revision, on whichever machine first runs it: the cache key excludes the build (§6.5),
  so each preset pays once per provider revision.

The protocol (pulse including carrier phase, sample endpoints, window, spacing,
amplitude, derivative and Fourier conventions, band and support rule, norms, floors and
threshold arithmetic) has a version number. Phase 1 freezes these values with a small
independent extraction witness before expensive checks run (Q16). Exclude DC from
division by $i\omega$; require a nonempty supported band, finite complex ratios and
controlled endpoint/window leakage. When the check runs, failure of these validity
conditions is an invalid diagnostic and refuses the run; it is not a mismatch that a
wider `refuse` bound can admit, though `enforce=False` skips the check altogether
(§5.4). Changing any part of the protocol bumps
the version, which invalidates every cached result (§6.5). The L3 cubic evidence uses
the same field and conventions at λ ∈ {1, 2, 4}, with its own fundamental-band rule and
fit residual limits; it is not part of the cached gate.

### 5. Proposed interface

#### 5.1 Records

New in `solprop.nonlinear.tdse`, exported from `solprop.sol`:

```text
@dataclass(frozen=True)
class HaccBasis:
    ions: Literal[1, 2]             # 1: haCC(1); 2: haCC(5) (He+ shells n <= ions)
    neutral_square: int             # largest l of the neutral's (l, l, 0) squares, 0..7
    radial_edges: tuple[float, ...] # m, from 0, strictly increasing
    radial_orders: tuple[int, ...]  # points per element, 2..64
    angular_limit: Literal["particle", "total"]
    angular_order: int              # particle l (1..120) or total L (1..120)
    scaling_radius: float           # R0 in m, exactly a radial edge (irECS start)
    scaling_angle: float            # rad, in (0, pi/2)
    infinite_order: int             # 1..64
    decay: float                    # 1/m, > 0
    gauge_radius: float | None      # Rg in m, a radial edge strictly inside R0;
                                    # None: velocity gauge (custom control only)
    cut_energy: float               # J, > 0
    max_step_au: float              # largest RK4 step, atomic units, > 0
    _: KW_ONLY
    source: str                     # provenance: the preset's evidence, or the user's
    preset: str | None = None       # provider preset identity, or None for a custom basis

    @property
    def identity(self) -> str: ...  # SHA-256 of canonical JSON with kind="hacc_helium"
    @property
    def label(self) -> str: ...     # "haCC(1)" or "haCC(5)", with " (custom)" when preset is None
    @property
    def qualified(self) -> bool: ...  # preset claim; verified against provider at preparation
```

A `HaccBasis` is either a frozen copy of a provider preset, made by `hacc_preset`,
or a custom definition, written directly or made from provider objects by
`hacc_basis(atom, basis, gauge, cut_energy, max_step_au)` (below). Its identity is
the SHA-256 of canonical JSON of every field, including `preset` and a
`"kind": "hacc_helium"` tag, so it is derived from the full definition in both
cases: a haCC identity cannot collide with an SAE `TDSEBasis` identity, haCC(1) and
haCC(5) always differ, and a custom basis equal to a preset in every number differs
from it only through `preset`. Validation mirrors `TDSEBasis` (finite positive SI
values, edges and orders, immutable tuples) and adds: `ions` in {1, 2};
`scaling_radius` an edge; `gauge_radius` an edge with $0<R_g<R_0$; all irECS fields
given; a preset string of the provider's form
`hacc-helium-<1|5>-v<version>-<16 hex digits>` whose model number agrees with
`ions`; and a preset record must have a gauge radius. `gauge_radius=None` is the
provider's `hacc.VelocityGauge()` control, representable for custom bases only. Argon
is not representable. A custom model without the escaped-electron closure (the
velocity gauge, or a mixed gauge whose radius does not admit it) is admitted at
preparation as an unqualified research control (§4.3, §5.6).

For a preset basis the binding checks at preparation that the installed provider
reproduces the preset's identity and fields (§6.3). A custom basis is prepared
through the provider's general `hacc.prepare`; it is admitted with a
`TDSEUnqualifiedModelWarning` and recorded as unqualified (§5.6).

`TDSEGasResponse` gains a model discriminator, an atomic window and the consistency
policy, and its source filter becomes model-dependent by default:

```text
model: Literal["sae", "hacc"] = "sae"
basis: TDSEBasis | HaccBasis
source_filter: Literal["none", "time_collar"] | None = None   # None: the model's default
atomic_window: TDSEAtomicWindow | Literal["auto"] | None = "auto" # resolved in __post_init__
consistency: TDSEConsistency = TDSEConsistency()               # haCC only, §5.4
```

`atomic_window="auto"` resolves in `__post_init__` to `None` for SAE and to
`TDSEAtomicWindow()` for haCC. An explicit `None` is distinguishable from omission and
selects the full lattice for haCC too, without a warning, as a research choice; the
stored record contains only the resolved value.

`source_filter=None` resolves in `__post_init__` to `"none"` for SAE (today's
default, unchanged) and to `"time_collar"` for haCC; the record stores the resolved
value, so every identity names the filter actually used. An explicit value of either
kind is admitted, for SAE or haCC, when the response has no atomic window; every window
requires `"time_collar"`, so a windowed response refuses explicit `"none"`. An
unwindowed haCC response still defaults to `"time_collar"`. `max_step_au` is an explicitly
named provider-metadata exception to the SI record convention: it preserves the
provider's exact binary step ceiling; `max_atomic_step` remains the physical value in
seconds. For haCC, `max_atomic_step` must equal `basis.max_step_au`
converted to seconds within two units in the last place of the conversion; the
binding passes `basis.max_step_au` to the provider, so no round trip changes the
step. The rules are:

| Condition | Result |
|---|---|
| `model="sae"` with a `HaccBasis`, or `model="hacc"` with a `TDSEBasis` | `MaterialError` naming the mismatch |
| `model="hacc"` and `kerr_model` is not `"tdse_native"` | `MaterialError`: the provider refuses the third order of the closed current (§4.3) |
| `model="hacc"` and `remainder_scale != 1.0` (exactly) | `MaterialError`: haCC's current is the whole atom's (§4.2) |
| `model="hacc"` and `cubic_scale is not None` | `MaterialError` (already refused for `tdse_native`) |
| `model="hacc"` and `max_atomic_step` not the basis's step | `MaterialError` |
| `model="hacc"`, `source_filter` omitted | resolves to `"time_collar"` |
| `atomic_window` with `source_filter="none"` | `MaterialError`: the window's gate is the source filter |
| `model="sae"` with a non-default `consistency` | `MaterialError`: the check applies to haCC only |
| `calibration.basis_identity != basis.identity` | `MaterialError` (unchanged rule) |

For a preset, the `TDSECalibration` record documents the preset's evidence: its
identity; a `wavelength_range` of 62 nm (20 eV, the preset's below-threshold bound)
to 1.6 µm, the spectral band the 800 nm qualifying drive and run_short's grid
occupy; the source; `uncertainty=None`, because the reported ladder sensitivities are not certified
uncertainties; and limitations naming the window, the intensity bound, finite-tail
shadow qualification and the diagnostic-only bound. Record `accuracy`, `box_sensitivity`
and `gauge_sensitivity` separately with their metric and tested waveform. Adding two
ladder maxima would not bound omitted model, absorber, time-window or radial errors. The
wavelength range is a proposed admission range, not independent qualification across all
its wavelengths. For a custom basis the user supplies the record, with `status="research"`, no
uncertainty and the limitation that the basis is unqualified;
constructing it is documented beside `hacc_basis`, which itself returns only a basis.
Both have status `"research"`: the preset qualifies the provider's
current on one waveform, not solprop's propagated source. Neither carries a scale.

The SAE default keeps today's records byte for byte: `model="sae"`, an absent
`atomic_window` and `source_filter="none"` are omitted from the serialised response
record, so SAE descriptor identities, output records and resume keys do not change.

Two resolvers join `basis_from_preset` in `solprop.driver.tdse` (they need the
provider, so they live in the driver layer):

```text
@dataclass(frozen=True)
class HaccPreset:
    basis: HaccBasis
    max_atomic_step: float                      # s, from the preset's max_step_au
    accuracy: float                             # relative, of the peak current
    box_sensitivity: float
    gauge_sensitivity: float
    max_peak_intensity: float                   # W/m^2 (8e18)
    wavelength: float                           # m (800e-9), the qualifying carrier
    window: float                               # s, tested atomic duration (59.2e-15 today)
    device_admissible: bool                     # the device route admits the lattice
    identity: str                               # the provider's preset identity
    source: str

    def calibration(self) -> TDSECalibration: ...

def hacc_preset(ions: Literal[1, 2]) -> HaccPreset
def hacc_basis(atom, basis, gauge, cut_energy, max_step_au, *, source) -> HaccBasis
```

`hacc_preset` calls the provider's `hacc.helium_preset(ions)`, converts its fields to
solprop's record, and verifies the build pin; it does not prepare a model. `window`
is the duration the preset's accuracy ladders drove: 59.2 fs today, and the duration
re-tested by Pyroraptor's haCC timing study Part D (about 60 fs) once that is integrated.
`hacc_basis` converts provider `hacc.Helium`, `hacc.Basis`, `hacc.MixedGauge` or
`hacc.VelocityGauge` objects and a cut and step into a custom `HaccBasis` with
`preset=None`; it is the "accept any provider model" path, and it refuses argon and any
field the record cannot represent by name.

#### 5.2 Entry point

`Simulation.from_capillary` and `prop_capillary` gain these response keywords:

```text
tdse: TDSEGasResponse | None = None
tdse_execution: TDSEExecution | None = None
```

With `tdse` given, the builder uses `[tdse]` as the complete electronic response
for the helium fill instead of the Kerr, plasma and Raman responses it would
otherwise build, sets `settings.tdse_execution`, and constructs the simulation in one
step (it reuses `_resolve_tdse_grid` and `_check_tdse_execution`). The keyword works
for SAE and haCC responses alike; the science stays in the record and the resources
stay in `TDSEExecution`, as SEP 12 separated them. Two keywords rather than one keep
that separation; folding resources into the response would make a device choice part
of the scientific identity.

Defaults interact as follows:

- `backend="auto"` resolves to `"numpy_ref"` when `tdse` is given, because it is the
  only backend that executes the response; the resolution is logged and recorded in
  the run metadata. An explicit other backend is refused by the existing
  `BackendError` naming `numpy_ref`.
- `integrator=None` already resolves to `Multistep()`; `absorber_scheme=None` to
  `"nonlinear"`; `noise=None` and `interpolate_planes=False` are already the
  defaults. Explicit incompatible values are refused by the existing checks.
- `stats=None` resolves to the default statistics minus
  `TDSE_UNAVAILABLE_STATISTICS`; an explicit request for an unavailable name is
  refused as today.
- Natural-step output is the TDSE default (§5.8): omitted `step_planes` resolves to
  1; if both `save_planes` and `n_save` are omitted, request only the final position
  (`n_save=2`, including the entrance). Explicit settings retain their meaning.
- `n_r`, `radial_rule` and `rtol` keep their meaning and defaults. The
  default 33-node Gauss–Kronrod rule is admitted with a TDSE response and
  documented as expensive. When the run starts, a TDSE response on a rule that is
  not a field-scale rule raises a `TDSECostWarning` (a `SolpropWarning`) once,
  naming the node count, that each node is one atom per call (so the atomic cost
  is proportional to it), and the field-scale alternative.

Without `tdse`, both entry points behave exactly as before; every existing test
must pass unchanged. `prop_capillary` also forwards `tdse_archive` to
`Simulation.run`, where it belongs with output handling rather than the physical
response (§5.8).

#### 5.3 The atomic window

```text
@dataclass(frozen=True)
class TDSEAtomicWindow:
    duration: float | None = None  # s, finite, > 0; None resolves from the capillary trange
    taper: float = 5e-15           # s, finite, > 0; width of each inner Planck taper
```

An explicit `duration` selects the atomic interval independently of the optical
grid and collar. With `duration=None`, the capillary entry point resolves it from
the requested `trange` (60 fs in the motivating case), using a replaced immutable
response record. A lower-level `Simulation` constructed from an existing grid
must supply the duration explicitly when it cannot recover the original request;
it refuses an unresolved duration before allocating a provider. It never infers
the requested duration from the collar's flat edges. Resolved default and explicit
records with the same duration have the same scientific identity.

For duration $T_a$, the nominal bounds are $t_-=-T_a/2$, $t_+=T_a/2$.
Require $0<T_a\le T$, where $T$ is the optical lattice period, and
$0<w<T_a/2$. On the refined lattice resolve
$t_s=\min\{t_n:t_n\ge t_-\}$ and $t_e=\max\{t_n:t_n\le t_+\}$.
The binding passes the retained samples multiplied by the §4.6 gate, evaluated
at their original absolute times. It records the provider's time zero as $t_s$
and duration as $t_e-t_s$, and pads returned currents with zeros on the full
lattice. `TDSEResult` retains its full-lattice shape; source assembly uses this
window's gate in place of the grid collar.

The resolved duration, nominal bounds, sample indices and taper breakpoints enter
the descriptor JSON and identity. Never stretch the gate to the rounded sample
endpoints. Require the provider's minimum sample count and test non-grid-aligned
bounds and endpoint interpolation. Changing only the atomic duration leaves the
optical time/frequency lattice and its absorber unchanged. In particular,
`TDSEAtomicWindow(duration=60e-15)` and `TDSEAtomicWindow(duration=100e-15)`
provide the candidate and wider reference for L5 on the same 102.5 fs lattice.

For haCC, the automatic default is `TDSEAtomicWindow()` with a 5 fs taper and
duration resolved as above; explicit `atomic_window=None` runs the atom over the full
lattice, as SAE does, and is admitted without a warning. Wider explicit windows, and
the full lattice, are admitted for validation and research; with a preset they receive
the duration flag of §5.6. SAE retains its resolved default `None` (full-lattice behaviour),
and may opt into the same explicit window record; L6 uses that control.

#### 5.4 The consistency check and its bounds

```text
@dataclass(frozen=True)
class TDSEConsistency:
    warn: float = 0.1               # |kappa_1 - 1| above this warns; finite, > 0
    refuse: float = 0.3             # |kappa_1 - 1| above this refuses; finite, >= warn
    enforce: bool = True            # False: skip the check (no atomic call)
```

Validation requires finite values with $0<\texttt{warn}\le\texttt{refuse}$ and refuses
booleans. With `enforce=True`, before the first propagation call of a haCC run, the
binding obtains the §4.9 result for its configuration, from the cache or by running it.
The outcome follows these rules, in order:

1. **Invalid:** malformed or non-finite output, an empty comparison band or another
   failed validity condition of §4.9 refuses the run with a `TDSEConsistencyError`,
   whatever the bounds, and records `invalid`. Native failures retain their normal
   charged failure path.
2. **Passed:** $|\kappa_1-1|\le\texttt{warn}$. The run proceeds and records `passed`.
3. **Warned:** $\texttt{warn}<|\kappa_1-1|\le\texttt{refuse}$. The run proceeds with a
   `TDSEConsistencyWarning` naming the model, $\kappa_1$'s worst wavelength and value
   and the bounds, and records `warned`.
4. **Refused:** $|\kappa_1-1|>\texttt{refuse}$. The run is refused with a
   `TDSEConsistencyError` (a `MaterialError`) naming the same values and how to widen
   `refuse` or skip the check, and records `refused`.

With `enforce=False` the check is **skipped**: no atomic call is made, the cache is not
read, a `TDSEConsistencyWarning` is raised once per run, and the record says `skipped`.
The defaults (0.1 and 0.3) follow §4.8: a merely less accurate research basis warns, a
gross error refuses.

`warn`, `refuse` and `enforce` are policy, not physics: `consistency` is excluded from
the descriptor JSON, the response identity and the resume key (dataclass metadata
`record=False`, as the output schema already supports), so changing them never changes a
run's scientific identity or prevents a resume. Every haCC run's provenance records,
under `tdse.consistency`: $\kappa_1$ (worst value and its wavelength, and the
per-frequency values), the protocol version, the cache key, whether the result came
from the cache and on which execution space it was computed, the bounds, `enforce`, and
the outcome (`passed`, `warned`, `refused`, `invalid` or `skipped`). For a preset it
also references the L3 cubic evidence (κ₃, its residual and the leakage term) by the
preset identity, without gating on it. A refused run writes the same record to its
requested output before raising, without creating a false propagation checkpoint; also
attach the structured result to the exception when no output handler exists. Preserve
the underlying error if output writing fails.

The check also runs on its own, without a propagation, through
`solprop.driver.tdse.check_consistency(response, execution, *, reference_wavelength,
refresh=False, tdse_archive=None) -> ConsistencyResult`, with the wavelength in
metres and the optional archive configuration of §5.8, which uses the same
cache. Define and document `ConsistencyResult` as a frozen record, including the
validity outcome and call ledger, rather than leaving its public return type
unspecified.

#### 5.5 The run_short variant

The entry point selects the response and its execution resources. A full run also
needs a deliberate radial rule and tolerance, so the SEP delivers a documented
`benchmarks/sep20/run_short_hacc.py` for the maintainer's later full runs, rather
than claiming a one-argument change:

```text
preset = hacc_preset(ions=1)                     # or ions=2 for haCC(5)
response = TDSEGasResponse(
    "helium", preset.basis, preset.max_atomic_step, "tdse_native", 1.0,
    preset.calibration(), TDSESampling(8), model="hacc",
    # atomic_window defaults to TDSEAtomicWindow(taper=5e-15): the requested 60 fs
)                                                # source_filter: time_collar
run = prop_capillary(
    a, L, "He", fill, lambda0=800e-9, pulses=[pulse], modes=1,
    trange=60e-15, wavelength_limits=(9e-8, 1.6e-6),
    tdse=response,
    tdse_execution=TDSEExecution(device="cuda", max_batch_rows=6,
                                 max_atomic_calls=400, memory_limit=...),
    radial_rule="field_scale", n_r=selected_n_r, # chosen by §7.2 L4
    rtol=1e-2,                                   # not 1e-5
    n_save=2, step_planes=1,                     # TDSE defaults, shown explicitly
    output="run_short_hacc1.h5",                 # every accepted optical step
    tdse_archive=TDSEArchive("run_short_hacc1.tdse.h5"),  # full raw diagnostics
)
```

- **`rtol=1e-2`** is SEP 12's maintainer-approved tolerance for this route; `1e-5`
  would multiply the call count by a large, unmeasured factor. Whether `1e-2`
  suffices for haCC is a question for the full runs.
- **`n_save=2, step_planes=1`.** Save every accepted endpoint while letting the
  stepper select its steps subject to tolerance, physical kinks and step limits.
  TDSE still refuses interpolated save planes. A dense requested plane list would
  force extra landings; saving natural endpoints does not. The quoted parent's 142
  accepted steps were measured with 51 requested planes, so they are not a prediction
  for this new itinerary.
- **`TDSESampling(8)`.** Fewer atomic steps than $q=32$ at negligible interpolation
  error (§4.7).
- **`TDSEArchive(...)`.** Save all atomic inputs and full observable outputs to a
  separate, append-only file. This is explicitly enabled in the research script; its
  cost and storage are measured, and its contents are specified in §5.8.
- **An HDF5 output** gives terminal checkpoints and cumulative call ledgers, so a
  long run can stop at its call cap and resume, including on another rented
  machine.

The script writes the provenance the AGENTS evidence rules require (input hashes,
provider build, threads, timings, ledgers, the consistency record) beside the
output. It takes a `--start-plane` option that launches from a saved parent plane;
with it the same script runs the segment of §7.2 L6, which is its only execution in
this SEP.

#### 5.6 Refusals, warnings and flags

| Case | Where | Result |
|---|---|---|
| haCC with `optmat_cubic` or `tdse_shadow` | `TDSEGasResponse` | `MaterialError` (§4.3) |
| haCC with `remainder_scale` other than exactly 1.0 | `TDSEGasResponse` | `MaterialError` (§4.2) |
| model/basis type mismatch | `TDSEGasResponse` | `MaterialError` |
| SAE with a non-default `consistency` | `TDSEGasResponse` | `MaterialError` |
| atomic window without `time_collar`, unresolved/non-positive duration, duration beyond the optical lattice, or taper of half the duration or more | response / entry point / descriptor | `MaterialError`, `DriverError` or `TranslationError` before provider allocation |
| haCC with explicit `atomic_window=None` | `TDSEGasResponse` | admitted: the full lattice, without a warning (§5.3) |
| a preset `HaccBasis` the installed provider does not reproduce | binding preparation | `BackendError` naming the expected identity |
| a custom `HaccBasis` (`preset=None`) | binding preparation | admitted; `TDSEUnqualifiedModelWarning` once per run, recorded as unqualified with no preset accuracy |
| a custom basis whose provider `info.total_current_qualified` is false (velocity gauge, a gauge radius without the closure) | binding preparation | admitted as a research control; the same warning states that the total current lacks the escaped-electron closure; full output with j1 from the in-box order 1; recorded `closure: false`, bound absent (§4.3) |
| a preset whose provider `info.total_current_qualified` is false | binding preparation | `BackendError`: a provider contract violation (presets always have the closure) |
| haCC with `TDSEExecution(complement="vendor_blas")` | binding preparation | `BackendError`: haCC plans have no complement option |
| a preset on a device route it does not admit | binding preparation | `BackendError` |
| provider without the haCC consumer surface, or an unpinned revision | binding preparation | `BackendError` naming the required pin |
| `ValueError` or `TypeError` from the provider | binding | `BackendError`, classified `interface`; preparation has no evaluation charge, but any attempted `evaluate` remains charged and records a failure |
| `DrivenError` from `evaluate` (a native failure or non-finite row) | binding evaluation | a failed, charged call, as today, raised as `BackendError` with its failure record |
| a compact/full result without the required order-one shadow, or with a malformed bound array (a missing bound is admitted only for `closure: false`) | binding evaluation | `BackendError`: contract violation (the bound's size is never a refusal) |
| haCC weak-field linear ratio with $\lvert\kappa_1-1\rvert$ above `refuse` (default 0.3; §5.4 rule 4) | before propagation | `TDSEConsistencyError`, recorded `refused` |
| an invalid consistency diagnostic (§5.4 rule 1) | before propagation | `TDSEConsistencyError` whatever the bounds, recorded `invalid` |
| $\lvert\kappa_1-1\rvert$ above `warn` (default 0.1) and at most `refuse` (§5.4 rule 3) | before propagation | `TDSEConsistencyWarning`, recorded `warned` |
| `TDSEConsistency(enforce=False)` | before propagation | check skipped, no atomic call; `TDSEConsistencyWarning` once per run, recorded `skipped` |
| `tdse` with explicit `plasma`, `raman`, `preionisation`, `plasma_integration`, `ionisation_field_range` or `thg`, or `kerr=False` | entry point | `DriverError`: the TDSE response owns the electronic response |
| `tdse_execution` without `tdse` | entry point | `DriverError` (existing rule) |
| `tdse` with a mixture, a non-helium gas, preionisation or a taper | admission | existing `TranslationError`s |
| explicit non-NumPy backend, Runge–Kutta, fixed schedule, noise, interpolated planes | existing checks | existing errors |
| unavailable statistics | existing checks | existing errors |
| TDSE response on a non-field-scale rule (for example the 33-node default) | run start | `TDSECostWarning`, once per run (§5.2) |
| a preset call whose peak input intensity $\epsilon_0cE_{\max}^2/2$ exceeds `max_peak_intensity` | binding evaluation | admitted; `outside_preset_domain` flag on that call's record and a `TDSEDomainWarning` once per run |
| a preset run whose resolved atomic-window duration $t_e-t_s$ exceeds the preset's `window`, or whose reference wavelength differs from the preset's carrier | run start | admitted; record separate reasons and a `TDSEDomainWarning` once per run (§4.6, §12 Q17); a default 60 fs run is flagged until the pin includes Pyroraptor's haCC timing study Part D's updated `window` |

Domain cases are flagged rather than refused: a single call slightly above the
bound should not destroy a long run, and the per-call record lets the analysis
exclude or qualify such planes. A custom basis has no domain, so it gets no domain
flags, only the unqualified-model warning. The neglected-current bound is never a
refusal; it is recorded with every call (§4.4).

**Attempt accounting (Q21, resolved by the maintainer).** Validate inputs in
the binding before invocation. Once a provider `evaluate` is attempted, its charge
is irrevocable, including for `ValueError` and `TypeError`: exception type cannot
prove that native work did not begin. Keep interface errors separately classified
in the failure record, without erasing the attempted or failed-call counts.
Construction and binding-side validation failures have no evaluation charge.
Release only unused reservations for chunks that were never attempted. This
preserves SEP 12's conservative ledger for SAE and haCC alike.

#### 5.7 The provider contract solprop requires

This is the contract the haCC preparation path binds to, stated in solprop's terms.
It is the provider's documented haCC consumer contract at revision `b8fc0c16`.

- **C1 Models.**
  - *Presets.* `pyroraptor.hacc.helium_preset(ions)` with `ions` a plain int, 1
    (haCC(1)) or 2 (haCC(5)), returns a frozen, hashable `HeliumPreset` with
    `name`, `identity`, `atom`, `basis`, `gauge`, `cut_energy` (J), `max_step_au`,
    `accuracy`, `box_sensitivity`, `gauge_sensitivity`, `max_peak_intensity`
    (W/m²), `wavelength` (m), `window` (s, the tested atomic duration; Pyroraptor's haCC timing study
    Part D updates it), `inner_nodes`, `device_admissible` and `source`. `preset.prepare()` returns the prepared model.
  - *Custom.* `pyroraptor.hacc.prepare(hacc.Helium(ions, neutral_square),
    hacc.Basis(radial, scaling, angular), gauge=hacc.MixedGauge(Rg)` (or
    `hacc.VelocityGauge()`), `cut_energy=...)` returns the same kind of model for any
    admitted basis.
  - Either model is a context manager that must stay open for the lifetime of every
    plan built on it.
- **C2 Identities.** `preset.identity` is
  `hacc-helium-<1|5>-v<version>-<16 hex digits>`, the SHA-256 of the preset's
  canonical definition, the same in every process; at version 1 they are
  `hacc-helium-1-v1-28f0eae146b1d665` and `hacc-helium-5-v1-5853d5d50c187aa9`.
  solprop stores it in `HaccBasis.preset`, its identity and the descriptor. A
  changed preset definition gives a new identity, which solprop refuses for an
  existing record until it is regenerated. A custom model has no provider identity;
  `asdict(plan.info)` gives its atom, basis, gauge and cut, which solprop checks
  against the `HaccBasis` it derived its identity from.
- **C3 Response plan.** `pyroraptor.hacc.ResponsePlan(model, *, rows, spacing,
  samples, max_step_au, field="electric_field", shadow_order, outputs,
  norm_growth_limit)`, one plan per distinct row count. `rows` and `samples` are
  plain ints; `spacing` (s) and `max_step_au` are plain positive Python floats, so
  the binding converts NumPy scalars. solprop passes `max_step_au=basis.max_step_au`,
  `shadow_order=1`, `outputs="total_current"` (or `"full"` for the archive of §5.8)
  and `norm_growth_limit=2.0`; the
  descriptor's haCC setup record pins the limit. The plan is a context manager;
  exiting closes it. It exposes `endpoints` and `info`, a frozen
  `pyroraptor.hacc.ResponsePlanInfo` with `rows`, `samples`, `spacing` (s), `route`,
  `substeps`, `step` (s), `step_au`, `steps`, `stepper` (`"rk4"`),
  `monitor_interval`, `norm_growth_limit`, `radius` ($R_0$, m), `gauge_radius`
  ($R_g$, m), `closure_radius` ($R_c$, m), `cut_energy` (J), `atom`, `basis`,
  `gauge`, `dimension`, `total_current_qualified`, `build`, `shadow_order` and
  `outputs`. The binding checks that the atom, basis, gauge and cut equal the
  `HaccBasis`, and requires `total_current_qualified` for a preset. Without the closure
  the provider refuses `outputs="total_current"` with `shadow_order=1` (`ValueError`,
  "needs the escaped-electron closure"), so the binding prepares a custom unqualified
  model's plans with `outputs="full"` (§4.3). `memory_ledger().entries` have
  `name`, `bytes`, `memory_space` and `shared`; the model's entries (`model.*`) are
  shared and counted once across plans, and the plan's own entries are not shared.
- **C4 Evaluation.** `evaluate(fields)` takes a NumPy float64 `(rows, samples)`
  real E in V/m at $t_m=m\cdot$`spacing` and restarts every row from the
  field-free ground root. With `outputs="total_current"` it returns a
  `pyroraptor.hacc.TotalCurrentResponse` with:
  - `time`, float64 `(samples,)` in s, and `radius`, $R_0$ in m;
  - `total_current`, float64 `(rows, samples)`, the closed current of one whole
    atom in C m/s;
  - `shadow_total_current`, float64 `(rows, 1, samples)` in C m/s, and
    `shadow_orders == (1,)`: the in-box first-order shadow, used as the closed
    current's first order with the finite-tail qualification of §4.3 (Q15, resolved:
    leakage below $2\times10^{-7}$ of the nonlinear source in the provider's weak-field
    run; L3 adds the haCC(5) figure);
  - `neglected_current_bound`, nonnegative float64 `(rows, samples)` in C m/s, a
    diagnostic (§4.4);
  - `valid`, bool `(rows, samples)`, false at and after a row's first failure;
  - `outcomes`, one per row, each with `ok`, `failure` (`"none"`, `"non_finite"` or
    `"norm_growth"`) and `failure_step`;
  - `monitors`, a `pyroraptor.hacc.RowMonitors` of `(rows,)` arrays:
    `initial_norm`, `norm`, `peak_norm` (float64), `monitored_step` (int64) and
    `non_finite` (bool). solprop's monitor dtype check, written for float64 SAE
    monitors with one int64 index, must admit the bool `non_finite`;
  - `info`, as C3.

  All arrays are owning and read-only. A failed row does not affect the other
  rows. For a full archive, `pyroraptor.hacc.GridResponse` additionally carries
  `vector_potential`, `electric_field`, `charge_dipole`, `current`, `outside` and
  a shadow with its order labels, dipole, in-box current and available total-current
  arrays. It retains `time`, `total_current`, the bound, validity, monitors, outcomes
  and `info`. There is no separate plasma record. The binding extracts the same
  total and first-order currents through the model-specific full adapter; it does
  not apply SAE's `raw.plasma` access to haCC. Full/compact equality must be verified
  for the consumed currents and their nonlinear remainder (§7.1, §7.2).

  For a model without the closure (`total_current_qualified` false, custom only), the
  full record's `total_current` is the in-box current plus the field-impulse term
  $\int E(N_0-N)\,dt$ (`hacc.solve.escaped_current`; it omits the momentum carried out
  through $R_0$), `shadow.total_current` is `None` (the provider gives the total no
  orders without the closure), `neglected_current_bound` is `None`, and $j_1$ is taken
  from `shadow.component("current", 1)`, the in-box order 1.
- **C5 Errors.** `pyroraptor.DrivenError` for a closed plan or model (from
  `evaluate` and `memory_ledger`) and for a native failure, including one during
  plan construction (the plan releases what it built). `ValueError` for a
  non-finite field, an unsupported order (order three of the closed current) or an
  out-of-range argument; `TypeError` for wrong argument types. solprop maps them as
  §5.6 says: `DrivenError` from `evaluate` is a failed, charged atomic call;
  `DrivenError` during preparation is a preparation failure; `ValueError` and
  `TypeError` anywhere are interface errors.
- **C6 Build identity.** The existing build record (source revision, execution
  space, requested threads) and unit scales, the same as for SAE, so the pin check
  and device check work unchanged; `info.build` repeats it per plan.
- **C7 Memory admission.** The provider offers no estimate before preparation.
  solprop admits in two steps: the model's measured ledger after preparation (setup
  0.03 h for haCC(1) and 0.11 h for haCC(5) on CPU at the presets), then each plan's
  ledger after construction, both before the first evaluation and against
  `TDSEExecution.memory_limit`. The manual records the presets' measured memory from
  Pyroraptor's haCC timing study and the L2 witness, so users can size `memory_limit` beforehand; for a custom basis the
  user sizes it. This is post-allocation admission, not a bound on setup peak memory:
  allocation can fail before a ledger exists, and that failure must close all partially
  constructed resources and be reported as preparation failure.
- **C8 Repeatability.** Identical atomic inputs on the same build and thread count must
  give bitwise identical atomic outputs; L2 tests this for both models. It does not
  imply identical optical trajectories after an Adams restart or bitwise equality
  between different devices.

#### 5.8 Optical steps and atomic-response archives

**Optical output.** SEP 14 already supplies `step_planes=1`: the march returns at
each accepted step, the driver saves the endpoint, and the next call continues
with the same Adams history. Saving does not evaluate the atomic response, restart
the method or cap a step at an artificial output position. The entrance and final
endpoint are saved once; an endpoint coinciding with an explicitly requested plane
is not duplicated. Physical discontinuities and the normal step limits still apply.

For `tdse=` on either capillary constructor, an omitted `step_planes` resolves to
1; omitted `n_save` and `save_planes` resolve to entrance plus final position. Use
`step_planes: int | None = None` at these convenience boundaries so an explicit
`0` still disables natural-step saves. Without `tdse`, `None` resolves to the
existing 0 and the existing requested-plane count remains 201. An explicit
`n_save` or `save_planes` is honoured and still forces landings alongside natural
saves. The resolved settings are recorded. Lower-level `SimulationSettings` and
existing SAE construction keep their defaults; their documented equivalent is
`step_planes=1` with only the final requested plane. Resuming a saved endpoint has
the existing restart semantics; writing it did not itself restart the live march.
The output coordinate is now generally nonuniform, and tools requiring a common
plane list retain their existing refusals (including scans).

**Archive interface.** Add a frozen output configuration, exported from
`solprop.output` and `solprop.sol`:

```text
TDSEArchive(path: Path | str, *, outputs: Literal["full", "required"] = "full")
Simulation.run(..., tdse_archive: TDSEArchive | None = None)
prop_capillary(..., tdse_archive: TDSEArchive | None = None)
```

`None` leaves raw archiving disabled. `"required"` stores every field of the
provider result already required by the source policy; `"full"` asks the same
evaluation for the provider's full observable record and stores it all. It does not
add shadow orders, change the atomic state or perform a second evaluation. This
works for SAE and haCC; refuse it without a TDSE response. The explicit path may
accompany an in-memory optical output. It must differ from the main HDF5 file;
existing unrelated files are refused, never overwritten. The separate file has
its own versioned schema, initially `solprop.tdse-archive.v1`, and a streaming
reader that enumerates calls and loads selected arrays without loading the run.

**What is saved.** The sidecar is append-only HDF5. Each provider evaluation is one
group, written as the evaluation completes, holding:

- the exact float64 input passed to the provider chunk after refinement, slicing and
  gating, in V/m, and the full pre-window nodal field of its logical evaluation;
- the raw provider result, captured immediately after `evaluate` returns and before
  validation, subtraction, scaling, padding or reduction: every returned array with its
  dtype, shape and values (including invalid samples, NaN/Inf, validity masks, bounds,
  all available shadow orders and monitors) and the scalar/structured diagnostics and
  `info`, through explicit versioned serializers for the SAE and haCC full/compact
  records, with units, order labels and absent fields marked; no pickle, no decimation,
  no float32 conversion;
- labels: a monotonically increasing evaluation ID, the optical trial ID and role
  (`startup`, `predictor`, `corrector`, `diagnostic`, or `rejected` once the trial is
  rejected), `z`, the field and response identities, provider plan/build, execution
  configuration, density, the gate and the optical time map (origin, refined spacing,
  window indices) relating provider-local `time=0` to the optical coordinate, and the
  row ranges, radial coordinates and quadrature weights.

An evaluation that raises still writes its input and the exception record. The full
SAE record includes its plasma decomposition and acceleration products; the haCC record
includes the dipole, in-box and total currents and lost norm of C4. Lost norm is
recorded as the provider defines it, not relabelled as an ionisation probability. A
predictor's response is labelled as such; it is not the response to the corrected
endpoint. Consistency-check evaluations made by the run, or by the standalone
`check_consistency` with an archive, are stored with the `diagnostic` role; a cache hit
writes nothing. Shared immutable grids and configuration records may be stored once and
referenced, preserving their values exactly.

**Sessions and failure.** Each run, and each resume, appends a new session group; the
run UUID is shared with the optical file, which records the archive schema and the
sidecar's relative path and remains readable without it. The archive does no charge
accounting: the optical ledger remains authoritative for calls, budgets and resume.
There are no transaction markers or cross-file reconciliation. A crash leaves the
evaluations written so far readable; the last group may be incomplete, and the reader
flags it rather than failing. Use one synchronous writer with bounded memory, streaming
each provider chunk before releasing it, and include its buffers in memory admission.
Open and validate the sidecar before paid atomic work; an existing unrelated file is
refused, never overwritten. A write failure (disk full, serialization) stops the run
before the next provider call, preserving the last optical checkpoint; if writing a
provider failure's record also fails, the provider failure stays the primary cause.
Archive paths and output selection are execution provenance, not physical identities.
The sidecar is a history of actual evaluations, not a restart store or permission to
reuse rejected data. Report bytes written and write time.

**Cost and scope.** For `R` rows, `N` samples and `K` float64 trace arrays, array
storage is approximately `8 R N K` bytes per evaluation, plus validity, metadata,
shared time axes and the pre-window field. Here 60 fs at `50.03 as / 8` is about
9,600 samples: one six-row scalar trace is about 0.46 MB (decimal). A collection
of ten such traces is about 4.6 MB per evaluation, or 0.46 GB per 100 evaluations,
before the extra records; this is a sizing example, not a measured run total or
the exact field count of either result type. Count the actual arrays in the
manifest. Full observations can add provider work and device transfers even when
writing is cheap relative to propagation; measure both modes and check consumed
current/remainder parity. These are sampled observable histories. Saving the
many-electron wavefunction at every internal atomic step is a different, much
larger provider feature, outside this SEP.

### 6. Implementation strategies

#### 6.1 Layers and files

| Layer | File | Change |
|---|---|---|
| Physics records | `src/solprop/nonlinear/tdse.py` | `HaccBasis`, `TDSEAtomicWindow`, `TDSEConsistency`; `TDSEGasResponse.model`, `atomic_window`, `consistency`, the model-dependent `source_filter` default and the §5.1 rules |
| Exceptions | `src/solprop/exceptions.py` | `TDSEConsistencyError` (`MaterialError`); `TDSECostWarning`, `TDSEDomainWarning`, `TDSEConsistencyWarning`, `TDSEUnqualifiedModelWarning` (`SolpropWarning`) |
| Public names | `src/solprop/sol.py`, `src/solprop/nonlinear/__init__.py`, `src/solprop/driver/__init__.py` | physics records in `nonlinear`; provider resolvers and `ConsistencyResult` in `driver`; `TDSEArchive` in `output`; convenience exports in `sol`, with no physics-to-driver import |
| Seam | `src/solprop/translate/tdse.py` | descriptor schema `v3` for haCC and for any atomic window; reconstruction of either basis type in `_response_from_json`; a haCC `_setup` record; the resolved atomic window and gate; a derived `TDSEDescriptor.model` property; binding trial/evaluation linkage hooks for the archive |
| Admission | `src/solprop/translate/tdse_source.py` | unchanged logic; the error text of the `optmat_cubic` path gains the model |
| Runtime | `src/solprop/driver/tdse.py` | `_Provider` gains the haCC namespace; `_ProviderBinding` dispatches to `_prepare_sae` or `_prepare_hacc` (preset or custom); window slicing and padding; model-aware bound and monitor checks; qualification gate; bound recorded as a diagnostic; domain flags; the §5.6 error mapping; `hacc_preset`, `hacc_basis`; pin bump |
| Source | `src/solprop/backends/numpy_ref/tdse.py` | the source gate $g$ read from the descriptor in place of the grid collar when an atomic window is set; source arithmetic otherwise unchanged |
| Trial events | `src/solprop/backends/numpy_ref/backend.py`, `src/solprop/backends/multistep.py` | pass trial IDs and roles (including rejection) through the host binding hooks for the archive labels, without changing numerical stepping or adding provider calls |
| Consistency | new `src/solprop/driver/tdse_consistency.py` | the §4.9 protocol: synthetic field, the κ₁ reduction and gate against optmat helium, the cache, `check_consistency`; the L3 κ₃ evidence extraction as a separate function, not part of the gate |
| Driver | `src/solprop/driver/simulation.py`, `src/solprop/driver/interface.py` | `tdse=`, `tdse_execution=` on `from_capillary` and `prop_capillary`; `auto` backend resolution; statistics and natural-step output defaults; archive lifecycle and trial/checkpoint linkage; cost, domain and unqualified warnings; consistency gate before the first call and its provenance |
| Output | `src/solprop/output/schema.py`, new `src/solprop/output/tdse_archive.py` | `model`, an absent window and `source_filter="none"` omitted from serialised SAE responses; `consistency` never serialised into identities; `TDSEArchive`, the append-only sidecar writer, versioned raw serializers and reader, and the run UUID and relative-path reference in the optical file; `SCHEMA_VERSION` 17 → 18 |
| Packaging | `pyproject.toml`, `uv.lock`, `docs/dev/dependencies.md` | provider pin to a revision at or after `b8fc0c16` |
| Evidence | `benchmarks/sep20/` | `parent.py`, `witness.py`, `nodes.py`, `window.py`, `segment.py`, `run_short_hacc.py`, `gpu_timing.sh`, `README.md`, `evidence/` |

#### 6.2 Records and schemas

**Descriptor.** A haCC response, or any response with an atomic window, serialises
under `"schema": "solprop.tdse-descriptor.v3"`, with `response.model`,
`response.basis.kind="hacc_helium"` and `response.basis.preset` (a provider identity
or `null`) for haCC, `lattice.atomic_window` (the resolved duration, taper, nominal bounds, first and last
samples, $t_s$, $t_e$ and the gate's breakpoints) when set, and for haCC a setup
record
`{"initial": "ground_root", "restart": "every_evaluation", "growing": "refuse_below_cut", "norm_growth_limit": 2.0}`.
The v1/v2 schemas stay SAE-only, without a window, and byte-identical.
`TDSEDescriptor.__post_init__` validates the v3 setup, the window and, for haCC,
`shadow_order == 1` and `outputs == "total_current"`. The model is a derived
property read from `configuration_json`. Nevertheless the descriptor and gate
change what the seam accepts and what the backend must interpret. Phase 1 must bump the
kernel `SCHEMA_VERSION` from the version current at integration, update its frozen
literal and compatibility documentation together, and refuse new descriptors in older
consumers. An unchanged dataclass field list does not make a new descriptor semantics
compatible. `consistency` never enters the descriptor JSON. The descriptor's
`outputs` continues to identify the source policy's required products. A full
archive may request the provider's superset without changing that descriptor;
record and validate the actual native selection separately, with explicit
model-specific adapters. Require full/required parity before treating this as
output-only provenance. Update the binding protocol and its frozen tests for
trial/evaluation events; archive I/O remains in the host driver/output layers.

**Output.** `RunMetadata.tdse` and the launch identity can now carry a haCC
response, its preset identity or custom flag, its atomic window, its consistency
record and the per-call bound diagnostics. `SCHEMA_VERSION` goes from 17 to 18 with
a docstring line saying that version 18 admits `model="hacc"` TDSE records, atomic
windows, `consistency` provenance and optional archive references. SAE records
without a window or archive keep their keys, so existing resume keys and identities
are unchanged. Tests in
`tests/output/test_schema_lineages.py` cover both.

**Identities.** `HaccBasis.identity` covers every field, including the preset
identity or its absence; `TDSEGasResponse` is identified through the descriptor
JSON as today. For a preset, both solprop's identity and the provider's preset
identity are recorded; solprop's is authoritative for resume, and the binding
refuses a preset identity that the installed provider does not reproduce. A custom
basis's identity is solprop's alone.

**Breaking change.** `TDSEGasResponse.source_filter` changes its annotation from
`Literal["none", "time_collar"] = "none"` to an optional value resolved per model.
SAE behaviour, records and identities are unchanged; code that reads the field
after construction still sees a resolved string. Code that passes `None` explicitly
for SAE now gets `"none"` instead of a validation error. The changelog records the
change.

#### 6.3 The haCC preparation path

`_ProviderBinding.__init__` keeps its structure: estimate, admit, load provider,
check build, prepare inside one `ExitStack`, admit measured memory. `_prepare`
dispatches on the descriptor's model:

1. for a preset basis, call `hacc.helium_preset(basis.ions)` and refuse it unless
   its identity equals `basis.preset` and its fields, converted to SI, equal the
   `HaccBasis` fields exactly; refuse a device route the preset does not admit. For a
   custom basis, build the provider's `Helium`, `Basis` and `MixedGauge` (or
   `VelocityGauge` for `gauge_radius=None`) from the fields and warn once that it is
   unqualified. Refuse `vendor_blas`;
2. enter the model (`preset.prepare()` or `hacc.prepare(...)`) in the stack first,
   so it closes after the plans, and admit its ledger against the memory limit;
3. for each distinct chunk size, enter a response plan with `rows`, the response
   `spacing` (a Python float), the atomic window's `samples`,
   `max_step_au=basis.max_step_au`, `field="electric_field"`, `shadow_order=1`,
   `outputs="total_current"` (or `"full"` for full archiving, and always `"full"` for a
   custom model without the closure, §4.3) and `norm_growth_limit=2.0`, and admit its
   non-shared ledger entries;
4. check each plan's `info` against the descriptor (rows, samples, spacing, order,
   atom, basis, gauge, cut) and `info.outputs` against the selected native output mode.
   `total_current_qualified=False` is a contract violation for a preset; for a custom
   basis it is recorded as `closure: false`, and the unqualified-model warning says
   that the total current lacks the escaped-electron closure;
5. record the shared `model.*` ledger entries once and each plan's non-shared
   entries under `plan.<rows>.*`, the host reserve as for SAE, and the `info`
   records (including `closure_radius`) for metadata.

`evaluate`, budgeting, failure records, caching and `_combine` are shared. The
binding validates the field (dtype, contiguity, finiteness) before charging, slices
it to the atomic window, and after the call places the currents back on the full
lattice with zeros outside. Exceptions map as §5.6 says: a `DrivenError` is a
failed, charged call; a `ValueError` or `TypeError` from `evaluate` is a
separately classified interface failure that also retains the charge. The archive
captures any returned object before validation. `_normalize` checks the selected
native type, `info`, the order-one labels and shadow extents, then builds the same
`TDSEResult` from compact arrays or the full haCC record of C4; without the closure
$j_1$ is the full record's in-box `shadow.component("current", 1)`. The diagnostic
helpers read the bound from `raw.plasma` only for the SAE full record and directly
from the result otherwise; a `None` bound is admitted only for `closure: false` and
recorded as absent. For haCC they validate its shape, dtype, finiteness and sign, and
record its per-row maximum relative to the peak current, but never compare it with
a threshold. The monitor check admits bool `non_finite` beside float64 and int64
vectors. The domain flag compares each preset call's recorded peak input intensity
with
`max_peak_intensity`, recording peaks before and after the drive taper so the gate cannot
conceal a field above the bound; the duration and carrier reasons of §5.6 are recorded
at run start. The SAE path is not edited beyond being moved into
`_prepare_sae`, sharing the generalised diagnostic checks, window option and
optional full-output adapter/archive.

`TDSEExecution.complement` has no haCC meaning: `"deterministic"` is recorded as not
applicable, `"vendor_blas"` is refused (§5.6).

#### 6.4 Entry-point options

| Option | Effect | Assessment |
|---|---|---|
| A. `tdse=`/`tdse_execution=` on `from_capillary` and `prop_capillary` | one-step construction; `run_short.py` changes by keywords | **Recommended.** Both entry points share one builder; the change is local to response selection and settings; SAE gains it too |
| B. Only on `from_capillary` | users keep the two-step pattern for `prop_capillary` | Inconsistent: `prop_capillary` is documented as `from_capillary` plus `run` with the same keywords |
| C. No entry-point change; a script builds the parent and then `Simulation(...)` | as SEP 12 benchmarks do today | Works, but every user repeats a fragile private pattern (grid refinement, collar resampling, settings replacement) |

Option A is clean because the builder already chooses responses in one place and
the TDSE checks already exist. It is a small public-interface addition; this SEP
covers it.

#### 6.5 The consistency cache

The check's result depends only on the atomic configuration and the protocol, so it
is cached on disk and reused across runs, processes, jobs and machines:

- **Key:** SHA-256 of canonical JSON holding `HaccBasis.identity` (which contains the
  preset identity, or the full custom definition), the provider's source revision,
  the optmat version and helium material identity, the reference wavelength, and the
  protocol version. The execution space, provider build fingerprint and thread count
  are recorded in the entry but are not in the key: a CPU-computed entry is reused on a
  GPU and vice versa. κ₁ is a 10%-level diagnostic; device rounding changes it by many
  orders of magnitude less (the provider's CPU/device current agreement is of order
  $10^{-13}$ for haCC(1)), and the L2 witness reports κ₁ parity for both presets as
  evidence. Each preset therefore pays for its check once per provider revision, on
  whichever machine first runs it. `warn`, `refuse` and `enforce` are not in the key:
  the cache holds measured values, and the decision is made afresh from them on every
  use.
- **Location:** `$SOLPROP_CACHE_DIR/tdse-consistency/<key>.json` when that variable
  is set; otherwise `$XDG_CACHE_HOME/solprop/tdse-consistency/`, falling back to
  `.cache/solprop/tdse-consistency/` in the user's home directory. Tests set the
  variable to a `tmp_path`. Cluster jobs can point it at shared storage, and an entry
  can be copied to a rented machine.
- **Content:** the key's inputs, the execution space, build fingerprint and threads,
  the per-frequency κ₁ values, the validity outcome, the solprop and provider versions,
  wall and CPU time, and the creation time. κ₃ is not cached and does not enter the decision; it is L3 evidence
  (§7.2).
- **Writing:** to a temporary file in the same directory, then an atomic rename, so
  concurrent jobs never read a partial entry; two jobs that miss together both
  compute and the later rename wins with an equivalent result.
- **Invalidation:** any change to a key input gives a new key, so stale entries are
  never reused. Bumping the protocol version invalidates every entry. An entry whose
  stored inputs do not match its key, does not parse, has non-finite or malformed
  values, or fails the diagnostic validity checks, is ignored with a
  logged warning and recomputed; it never counts as a pass. `check_consistency(...,
  refresh=True)` recomputes and overwrites. Old entries are small (kilobytes) and are
  not pruned automatically; deleting the directory is always safe.
- **Accounting:** All attempted check chunks are charged to the run's existing budget
  with the `"diagnostic"` purpose, reserved before evaluation, and cost nothing on a
  cache hit or when the check is skipped (`enforce=False`). The single row obeys `max_batch_rows` and the same measured memory limit
  as propagation. The
  diagnostic plan must not overlap propagation-plan allocations unless their combined
  peak fits the limit. The standalone check uses an explicit budget and returns the same
  ledger. The run
  provenance says which happened.

#### 6.6 Backends

The TDSE route runs on `numpy_ref` only, by SEP 12's recorded exception: the atomic
call dominates the cost, and the optical step is unchanged. `numba_cpu` and Kokkos
keep refusing TDSE problems by name. This SEP proposes the same exception for haCC;
no backend parity work is planned, and the Final status needs only the exception
recorded here and the roadmap item SEP 12 already holds. GPU execution concerns the
atom, through `TDSEExecution(device="cuda" | "hip")`, and is qualified only on real
hardware.

#### 6.7 Cost and hardware

**The GPU is the primary target.** Propagation of haCC responses targets rented
NVIDIA H200 or AMD MI300-class GPUs, which are expected to be far faster than any
CPU allocation. The provider's measurement that haCC(5) at the presets' size runs
slower on an A40 than on 32 CPU threads (8.6 against 4.4 ms per step for the closure
without the shadow) is not representative of them: the A40's double-precision rate
is 1/64 of its single-precision rate, while the H200 and MI300 class have much greater double-precision
throughput and far higher memory bandwidth. Their timing of the closed, shadowed
presets comes from Pyroraptor's haCC timing study (haCC device performance), which measures both
presets at 5, 6, 8 and 33 rows on a rented H200 and MI325X (per step, per 60 fs call and
setup, after its optimisation series) under its own US$60 budget, outside this SEP's
cap. SEP 20 does not rent for timing.

**Measured CPU cost, for reference.** The provider measured the presets on a shared CPU
system, 32
threads, shared nodes, one row, with the order-one shadow on:

| Model | Per step | March to 8 fs (303k steps) | 102.5 fs call (524k steps) | Setup |
|---|---|---|---|---|
| haCC(1) | 2.4 ms | 0.20 h | about 0.35 h | 0.03 h |
| haCC(5) | 19 ms | 1.60 h | about 2.8 h | 0.11 h |

The shadow about triples a haCC(5) step (19 against 5.8 ms without it); haCC(5)
costs about eight times haCC(1) per row on CPU. The 60 fs atomic window of §4.6 at
$q=8$ needs about 288k steps (§4.7): about 0.19 h per haCC(1) row and 1.5 h per haCC(5)
row on CPU. These CPU figures size the atomic-only studies when they run on free CPU
nodes (below); they are not the plan for the segment.

**GPU planning bracket.** Extrapolating the provider's earlier unshadowed H200 and
MI325X haCC timings by the shadow's ×3 and the closure's measured +59–86% device
overhead gives, for a 6-row call over the 60 fs window: haCC(1) about 0.5–0.95 h and
haCC(5) about 1.0–1.7 h. This is an unmeasured bracket for budgeting only, kept until
Pyroraptor's haCC timing study reports; its Part C table then replaces it.

**What is rented** (Q10, amended 2026-10-17). Rented GPUs run the haCC segments, the
short witness step that opens them, and compute-heavy one-off atomic checks where a GPU
is much faster: the maintainer would rather not wait six hours on CPU for a task a GPU
finishes in about thirty minutes. Long development cycles stay on CPU, and no rented GPU
is left idle:

- **On rented GPUs, where 0132's measured costs show a large wall-time gain:** the L3
  consistency evidence, the L4 node study and the L5 window check, batched into one
  rental session together with the L2 witness, so the machine runs back-to-back
  scripted jobs rather than waiting on development. Before renting, project the
  session's cost from 0132's measured per-step times at the rows the jobs use (several
  study rows can share one call) and record the choice and projection. If the gain is
  small, or the projection does not fit the cap, the study runs on free CPU nodes.
- **On free CPU nodes:** development and fake-provider work, the Phase 1 and Phase 2
  tests, the rate parent (L1) and the rate and SAE segment controls, and any study for
  which a GPU gives little gain. The A40 may take haCC(1) jobs.

The preset re-check of Pyroraptor's haCC timing study Part D is provider work on free CPU nodes. All rented
work counts against this SEP's US$100 cap, and spending beyond it needs a new
maintainer approval.

**Segment plan, within a US$100 cap** (the maintainer's starting cap, 2026-10-17):

1. **Projection, before renting.** From Pyroraptor's haCC timing study's measured per-call times at the
   selected node count and hardware, project each segment run (3–10 calls over the
   60 fs atomic window) and record the projection.
2. **L2 witness, the first rented step.** At the start of the first segment rental, the
   short checks of §7.2 L2 for both models on the target build (bitwise repeat, CPU
   parity, compact/full parity, and κ₁ parity reported as evidence), plus model setup
   and the plan ledgers, followed in the same session by any atomic studies chosen for
   the GPU (above). It is charged to the cap; the witness alone should take well under
   one GPU-hour.
3. **haCC(1) segment**, the main target: projected 1.5–9.5 GPU-hours on the
   bracket, about US$6–36, until 0132's measured times replace it.
4. **haCC(5) segment**, if the remaining cap covers its measured projection with a
   20% margin: projected 3–17 GPU-hours, about US$11–65. If it does not fit, it is
   shortened (fewer calls, a shorter interval) or deferred, and the projection goes
   to the maintainer.

At US$3.80 per GPU-hour (SEP 12's MI325X quote), the bracket totals US$21–109, so
neither segment is guaranteed to fit. Use live prices and measured costs for the
selected node count, including counts above eight, setup, diagnostics, repeats, idle
time and retained storage; checkpoint-resume is validated on CPU (§7.2 L6), not on the
rental. Reserve teardown margin and enforce a
wall-time/spend cutoff as well as `max_atomic_calls`; a call cap alone does not cap
money. Shorten a segment only through a recorded validation revision. Spending beyond
US$100 needs a new maintainer approval. The rate and SAE segments run on free CPU nodes
at no rental cost: the SAE control twice, with and without the atomic window (§7.2 L6),
each 3–10 calls of about 1–2.2 h at 6 nodes. Every GPU run sets `max_atomic_calls` and
writes resumable output, so a cap,
a pre-emption or the end of a rental leaves a continuable record, never a silent
partial result; the pod's spend is recorded with the evidence.

**Atomic-only studies.** Their CPU size, one row per job at 32 threads, at $q=8$, is
the baseline against which a GPU session is chosen (above). No GPU figure is given here:
once Pyroraptor's haCC timing study reports, the GPU cost of each study is projected from its per-step
times.

| Study | Rows | CPU core-hours |
|---|---|---|
| L3 consistency gate (one row) and cubic evidence (λ = 2 and 4; λ = 1 is the gate's row), haCC(1) and haCC(5) presets | 3 per model | about 120 |
| L5 window: haCC(1), on-axis row, two planes, the ±30 fs window against a ±50 fs one | 4 | about 33 |
| L4 nodes: haCC(1), Gauss rule, 5, 6 and 8 nodes against 16 at the peak-power plane, and the selected count against 16 at the maximum-UV plane, on the 60 fs window | at most 59 | about 365 |
| L4 extension (Q9), if 5–8 fail: 10 and 12 against 24 at both planes | about 90 more | about 565 more |

The total is about 520 core-hours without the extension (L3 at its 40 fs protocol
window; L4 and L5 at 288k and, for the ±50 fs reference, 480k steps per row).

#### 6.8 Dependencies

No new third-party dependency; the cache uses the standard library. The `tdse`
extra's pin moves to a provider revision at or after `b8fc0c16`;
`_PROVIDER_REVISION`, `uv.lock` and `docs/dev/dependencies.md` move together. The
SAE fixture and binding tests rerun on the new pin, because the pin covers both
models and the provider's breaking changes since the old pin (closed-object errors,
argument strictness) also touch the SAE path. GPU builds of the provider on the
rented machine follow its installation guide; their build record is checked by the
existing pin and device checks.

### 7. Verification and validation

#### 7.1 Fake-provider tests (Phases 1–2)

All run in the default lane without the provider, through doubles of the C1–C8
surface in the style of `tests/driver/test_tdse_binding.py`. Tolerances are exact
where the operation is a copy or a bitwise-defined formula, and are justified in the
test otherwise.

1. **`HaccBasis` validation** (`tests/nonlinear/test_tdse_spec.py`): each field's
   valid range and type, including booleans, non-finite values, mutable sequences,
   a malformed preset identity or one whose model number disagrees with `ions`,
   $R_0$ not an edge, $R_g\ge R_0$ or not an edge, `ions` outside {1, 2}, partial
   irECS; `gauge_radius=None` admitted for a custom record and refused for a preset;
   `qualified` and `label` for preset and custom records.
2. **Identities:** a fixed record's identity equals a stored literal for a preset
   and for a custom basis; changing any field (including `ions`, `preset` and
   `max_step_au`) changes it; haCC(1) and haCC(5) records differ; a custom basis
   numerically equal to a preset differs from it; no `HaccBasis` identity equals the
   `TDSEBasis` identity of a record with the same numbers.
3. **Response rules:** every row of the §5.1 table, including `remainder_scale`
   1.537, 2.0 and `nextafter(1.0, 2.0)` refused for haCC, and a `max_atomic_step`
   off the basis's step; `source_filter` omitted resolves to `"time_collar"` for
   haCC and `"none"` for SAE; both explicit values admitted for any unwindowed
   response, SAE or haCC, and `"none"` refused for any windowed response; explicit
   `atomic_window=None` admitted for haCC; `TDSEAtomicWindow` and `TDSEConsistency`
   validation (non-positive, non-finite or boolean values, `warn > refuse`) and the
   consistency refusal for SAE; the calibration identity rule.
4. **SAE unchanged:** stored v1 and v2 descriptor JSON and identities, the
   serialised SAE response record and its resume key are byte-identical to the
   pre-change literals.
5. **Descriptor v3** (`tests/translate/test_tdse_contract.py`): round trip through
   `_response_from_json` for a preset, a custom basis and an SAE response with a
   window; tampering with any field, the preset identity, the window record, the
   setup record (including `norm_growth_limit`), the order or outputs is refused;
   `model` property; two responses differing only in `consistency` have the same
   descriptor identity and resume key.
6. **Atomic window** (`tests/translate/test_tdse_window.py`): on run_short's grid,
   $t_s$, $t_e$, the sample range and $g$ for tapers of 2, 5 and 10 fs equal values
   computed independently from the collar formula; $g$ is one between the tapers,
   zero outside the requested window and continuous at the joins; a taper of half
   the window is refused; the omitted/`"auto"` haCC default is a 5 fs taper, an
   explicit haCC `None` drives the fake over the full lattice without a warning, and
   the SAE default is none; the binding passes the fake exactly the
   window's samples multiplied by $g$, records time zero at $t_s$ and pads zeros
   outside; for a fake whose current depends causally on its drive, the gated source
   equals, bitwise, the gated source computed from a full-lattice run driven by the
   same tapered field (causal stopping for this manufactured map, §4.6). Test
   default duration resolution from the capillary request, an explicit equal
   duration's identical identity, refusal of unresolved duration at the direct
   boundary, wider duration on an unchanged optical grid, non-grid-aligned bounds,
   insufficient samples and duration greater than the lattice period.
7. **Binding preparation** (`tests/driver/test_tdse_binding.py`): the fake preset,
   custom `prepare` and plan record the exact arguments (`helium_preset(ions)` or
   `prepare(atom, basis, gauge=..., cut_energy=...)`, rows per chunk, Python-float
   `spacing`, `max_step_au=basis.max_step_au`, `shadow_order=1`,
   `outputs="total_current"` by default or `"full"` for full archiving or a model
   without the closure, `norm_growth_limit=2.0`); the model is entered before
   and closed after the plans, also when a plan constructor raises `DrivenError`;
   shared `model.*` entries are counted once across two plan sizes; a memory limit
   below the model's or a plan's measured ledger is refused before any evaluation; a
   custom basis is admitted with one `TDSEUnqualifiedModelWarning` and recorded as
   unqualified; a custom basis whose fake reports `total_current_qualified=False`
   (mixed gauge without the closure, and `gauge_radius=None`) is admitted with the
   same single warning naming the missing closure, prepared with `outputs="full"`,
   takes $j_1$ from the in-box `component("current", 1)`, admits a `None` bound and
   records `closure: false`; a preset identity or field the provider does not
   reproduce, an `info` whose atom, basis, gauge or cut differs,
   `total_current_qualified=False` for a preset, `vendor_blas`, a device route the
   preset does not admit and a provider without the haCC surface are each refused by
   name; a stale revision and a device-space mismatch are refused as for SAE.
8. **Binding evaluation:** a valid call gives `TDSEResult` arrays equal to the
   fake's (compact `shadow_total_current[:, 0]` or the full order-one current), read-only and
   owning; monitors with a bool `non_finite` are admitted and recorded; a bound of
   any size, including ten times the peak current, is admitted and recorded as a
   per-row relative maximum; a wrong result type, `info` mismatch,
   `shadow_orders != (1,)`, a `None` bound for a model with the closure, a negative,
   non-finite or mis-shaped bound,
   wrong extents, a row with `ok=False`, a `valid=False` sample and malformed
   monitors (wrong dtype for each field, wrong shape, non-finite float values) each
   raise `BackendError`, record a failure and keep the charge; chunks recombine in
   row order; the diagnostic cache works as for SAE; the SAE full-record path still
   reads its bound from the plasma record.
9. **Error mapping:** a `DrivenError` from `evaluate` is a failed, charged call with
   a failure record; a `DrivenError` from plan construction is a preparation
   failure; a `ValueError` or `TypeError` raises `BackendError` classified
   `interface`. Construction has no evaluation charge; after `evaluate` was
   invoked, the attempt and failure remain in the ledger, including when a fake
   performs work and then raises. Only unused chunk reservations are released; the binding's own input validation refuses a non-finite or non-float64
   field before any charge.
10. **Domain flags:** a preset call whose peak input intensity exceeds
    `max_peak_intensity` is evaluated, its record carries `outside_preset_domain`,
    and a `TDSEDomainWarning` is raised once per run (asserted with `pytest.warns`);
    a call within all domain conditions has no flag; separately test a resolved
    atomic-window duration above, equal to and below `window` (equal is not flagged),
    and a carrier mismatch, each with its own recorded reason; a custom basis raises no
    domain flag.
11. **Consistency reduction** (new `tests/driver/test_tdse_consistency.py`): a fake
    provider whose first-order current is built from a manufactured per-atom
    susceptibility $\alpha_{\mathrm{fake}}=\alpha_{\mathrm{optmat}}/k_1$ gives
    $\kappa_1=k_1$ to a tolerance derived from the transform's rounding; the band rule
    excludes frequencies where the field is below $10^{-3}$ of its peak and outside
    optmat's support. Test empty bands, DC exclusion, endpoint leakage, phase mismatches
    and non-finite ratios; an invalid diagnostic refuses whatever `warn` and `refuse`
    say. The L3
    evidence extraction is tested separately: a fake with a cubic coefficient
    $\gamma_{\mathrm{fake}}=\gamma_3/k_3$, a λ⁵ term and a small linear leakage term
    gives $\kappa_3=k_3$ over the fundamental band, the leakage coefficient and a λ = 4
    residual that trips at the stated bound when a resolvable seventh-order term is
    added; a third-harmonic component with a different coefficient leaves the
    fundamental-band $\kappa_3$ unchanged.
12. **Consistency gate:** with the defaults, $k_1\in\{1.05,\,0.95\}$ proceeds and
    records `passed`; $k_1\in\{1.2,\,0.8\}$ proceeds with one `TDSEConsistencyWarning`
    and records `warned`; $k_1=0.5$ (a doubled current), $k_1=2$ (a halved current) and
    $k_1=-1$ (a sign error) are refused with `TDSEConsistencyError` before any
    propagation call, and the output holds the `refused` record; the boundaries
    $|\kappa_1-1|=0.1$ (passed) and $0.3$ (warned) are exact; `refuse=0.6` admits
    $k_1=0.5$ as `warned`; `warn=0.03` warns at $k_1=1.05$; `enforce=False` makes no
    atomic call, raises one `TDSEConsistencyWarning` and records `skipped`; a custom
    basis is checked the same way; no κ₃ value affects the outcome.
13. **Consistency cache:** the first check makes one one-row atomic call and writes one
    entry under `SOLPROP_CACHE_DIR` in `tmp_path`, recording its execution space, build
    and threads; a second run with the same configuration records a cache hit, also
    from a different execution space, build or thread count; changing the preset
    identity, any custom field, `ions`, the provider revision, the optmat version or
    the protocol version misses; changing only `warn` or `refuse` hits and reapplies the
    new decision; `enforce=False` neither reads nor writes the cache; a truncated file, invalid JSON or an entry whose stored inputs
    disagree with its key is ignored with a logged warning and recomputed;
    `refresh=True` recomputes; a write is atomic (a reader never sees a partial
    file, checked by interrupting the write in the test double); the location falls
    back from `SOLPROP_CACHE_DIR` to `XDG_CACHE_HOME` as documented.
14. **Source assembly:** for identical arrays, a haCC descriptor gives
    `atomic_sources` output bitwise equal to the SAE `tdse_native` path at scale 1.0
    with the same filter and window (same formula,
    `tests/backends/test_tdse_sources.py`).
15. **Entry point** (new `tests/driver/test_tdse_entry.py`): `from_capillary` and
    `prop_capillary` with `tdse=` give `responses == (tdse,)` and the execution
    settings; `auto` resolves to `numpy_ref` and is recorded; each §5.6 entry-point
    refusal; `tdse_execution` alone refused; statistics default excludes the
    unavailable set; without `tdse`, a representative call produces settings and
    responses identical to before. Check TDSE defaults of final-only requested
    output plus every accepted step, explicit `step_planes=0`, explicit requested
    planes, and unchanged lower-level settings.
16. **Cost warning:** a TDSE run on the default Gauss–Kronrod rule raises one
    `TDSECostWarning` whose message names the node count and the atoms-per-call
    reason; a field-scale rule raises none.
17. **End to end, medium:** a two-step propagation on a tiny grid with a fake
    provider whose current is a manufactured polynomial in $E$ with an exact
    order-one part (so $j_{\mathrm{tot}}-j_1$ is known in closed form), with and
    without an atomic window, checking the consistency record (`passed`, `warned` and
    `skipped`), the ledger, the bound diagnostics, the metadata (`model`, preset
    identity or custom flag, `closure`, window, schema 18), checkpoint and resume with
    identical identities, fresh startup charges and no repeated consistency call, and
    refusal of a resume with a different basis. This is the haCC checkpoint-resume test;
    the real haCC segments are not deliberately interrupted (§7.2 L6).
18. **Output lineage:** schema 18 round trip with a preset haCC record (including its
    consistency provenance), a custom one and an SAE record; reading a schema-17 SAE
    file unchanged.
19. **Resolvers:** with a fake provider, `hacc_preset` returns the declared SI
    fields, identity, accuracy and domain, its `calibration()` keeps uncertainty unset and
    records the separate measured sensitivities, and it refuses `ions` other than 1 or 2; `hacc_basis` converts
    provider objects into a custom record equal to a hand-written one, maps a
    `VelocityGauge` to `gauge_radius=None`, and refuses argon by name.

20. **Natural TDSE output:** a manufactured adaptive response with a rejection
    gives identical attempted steps, accepted fields, Adams orders and atomic-call
    ledger with `step_planes=0` and 1 on the same final-only itinerary. Every accepted
    endpoint is written once; rejected trials are absent from optical output. Include
    hard-kink coincidence, callback interruption and endpoint resume. No output or
    statistic collection may add a provider call.
21. **Raw archive round trip** (Phase 2): SAE and haCC, required and full modes,
    multiple chunks, all shadow labels and actual dtypes, pre-window and exact passed
    inputs, gates and time maps, monitors, and raw invalid/non-finite samples that the
    source subsequently refuses, read back exactly through the streaming reader;
    full/required source arrays and their nonlinear remainder agree bitwise on the same
    build; archiving leaves the optical march and ledger unchanged.
22. **Archive labels and failures** (Phase 2): rejected and predictor evaluations carry
    their roles; a resume appends a new session group under the same run UUID; an
    evaluation that raises records its input and exception; a write failure (disk
    full, serialization) stops the run before the next provider call with the provider
    error kept as primary; an existing unrelated file is refused; a truncated last
    group is flagged by the reader; the optical file reads without the sidecar; writer
    memory stays bounded.

Coverage of the touched production code is complete in the default and coverage
lanes, as AGENTS requires. Phase 1 also includes one optional installed-provider
test, skipped without the provider: it resolves both presets through `hacc_preset`
and checks their identities and SI conversion, and runs one call of a tiny custom
haCC(1) basis through the binding (the provider's own consumer example basis), which
exercises the real C3–C5 surface in seconds.

#### 7.2 Validation ladder on the real provider (Phases 3–4)

Each level names its question, its inputs and what sets its tolerance before it is
run. haCC(1) is the main target; haCC(5) runs the consistency check and cubic evidence
(L3), the GPU witness (L2) and the segment (L6), and otherwise follows haCC(1). Timing
comes from Pyroraptor's haCC timing study. All atomic work goes
through solprop's binding, not around it, so the ladder also exercises the seam.
Evidence is compact JSON plus hashes under `benchmarks/sep20/evidence/`; raw arrays
stay outside the repository.

- **L1 Parent and fields** (`benchmarks/sep20/parent.py`, import/verification). Reuse
  the existing rate-based parent output on `numpy_ref`, `Multistep()`, `rtol=1e-2`,
  `n_save=51`, and export the on-axis and nodal $E(t)$ at the input, maximum-peak-power
  (4.05 m), maximum-UV (3.807 m) and segment-launch (3.969 m) planes with a manifest
  of hashes, solprop and optmat versions, threads and timings, following
  `benchmarks/sep12/parent.py`. *Check:* the export reproduces the plane statistics
  recorded in §2 with the saved precision and recorded source revision. No full 4.05 m
  propagation runs as a test, benchmark or this SEP's validation. The manifest records
  the input hashes (the pump file, the parent output, the exported planes), the solprop
  and provider revisions and regeneration instructions; redistributable inputs are not
  required, and the maintainer's measured pump stays private (Q19). A separately
  requested manual worked example may regenerate the parent.
- **L3 Weak-field consistency and cubic evidence** (`witness.py weak`, on the rented
  GPU session or free CPU nodes, §6.7). The §4.9 run-time check through
  `check_consistency` for each preset, which seeds the cache for every build of that
  provider revision (§6.5). *Gate:* $|\kappa_1-1|\le0.1$ for both presets (the run-time
  `warn` bound); a value above it stops the ladder for review, without an override. *Evidence, reported and not gated:* the λ ∈ {1, 2, 4} ladder for
  each preset, giving κ₃ over the fundamental band with its fit and λ = 4 residuals
  (§4.8), and the odd part's linear term, the preset-scale shadow-leakage figure of Q15
  (expected to be negligible; §4.3 gives the haCC(1) bound). A leakage term that is not negligible
  against the weakest retained node's nonlinear source stops the ladder for review;
  κ₃ is read beside the provider's validated γ values.
- **L5 Atomic window** (`benchmarks/sep20/window.py`, haCC(1), on the rented GPU
  session or free CPU nodes, §6.7). On the on-axis row ($\eta=1$, the strongest
  nonlinearity) at the maximum-peak-power and maximum-UV planes, with $q=8$: the source
  from the ±30 fs window with its 5 fs
  taper, against the same saved field on a ±50 fs atomic window, on the same full optical lattice, using the explicit durations of §5.3. The reference must retain the earlier
  drive history and its own wider output gate. Metric: the norm of the difference
  between the two **full-lattice gated source spectra**, over the active band and over 90–200 nm, with
  SEP 12's absolute cancellation floor. Do not crop both currents to ±30
  fs before transforming: that would hide the tail discarded by the candidate. Report a
  common-window comparison separately if useful to distinguish startup/drive error from
  missing late emission. *Check:* at most the L4 tolerances (1% over
  the band, 2% over 90–200 nm). The difference measures the dropped field, including
  the maximum-UV plane's far tail, and the taper together. The value is recorded;
  if it exceeds the tolerances the ladder stops for review rather than widening the
  window silently.
- **L4 Node convergence** (`benchmarks/sep20/nodes.py`, haCC(1) only, on the rented GPU
  session or free CPU nodes, §6.7), on the 60 fs atomic window and $q=8$. The rows of
  each plane's rules may share calls. At the maximum-peak-power plane (the
  strongest field, where the η structure is richest), nodal fields
  $\eta_qE_{\mathrm{axis}}$ for the Gauss field-scale rule at $n_r\in\{5,6,8\}$
  against 16; at the maximum-UV plane, the selected count against 16. Metric:
  relative norm of the projected source spectrum (§4.5) after the gate, over the
  active band and over 90–200 nm, with SEP 12's absolute cancellation floor.
  *Tolerances, set from SEP 19's measured transfer (a 1–2% source error gives a
  comparable UV error and about 10× less in field and peak power):* at most 1% over
  the band and 2% over 90–200 nm. These are
  study acceptance targets motivated by SEP 19, not error bounds transferred from SAE or
  from the presets' total-current metric. Passing against 16 nodes establishes agreement
  with that finite rule; it does not bound the reference's own error. *Selection:* the
  smallest count meeting both at the peak-power plane and confirmed at the
  maximum-UV plane. The pass is reported as agreement with the finite 16-node rule,
  not a certified error (Q18). *Extension (Q9):* if none of 5–8 does, 10 and 12 run against 24
  at both planes, and the smallest passing count is taken despite its higher cost
  per call; if 12 does not pass, 16 is used against 24, and if that fails, stop and
  report. *Not measured in this minimal set:* the Radau variant (SEP 19 found
  neither rule consistently ahead below about 15 nodes), the 16-node reference's own
  error (SEP 19 found 16 sub-percent for SAE), and haCC(5)'s own node error.
- **L2 GPU witness** (`benchmarks/sep20/witness.py gpu`, the first step of the first
  segment rental, §6.7). Timing is not measured here: the per-call costs for both
  presets at 5, 6, 8 and 33 rows on the H200 and MI325X come from Pyroraptor's haCC timing study
  Part C and fix the segment projection before renting. On the rented GPU, for haCC(1)
  and haCC(5) at the presets' numerics with the closure and order-one shadow, over a
  short window of the maximum-peak-power field at the selected node count: model setup
  time and ledger. *Checks:* all rows valid; `total_current_qualified` true; a repeated
  call for each model bitwise equal (C8); the first-order current scales linearly to
  rounding between the row amplitudes; the GPU's currents agree with a CPU call on the
  same short window to declared rounding-based tolerances, separately for total, shadow
  and remainder; compact and full outputs agree bitwise for the consumed currents and
  remainder (§7.3). *Reported as evidence:* the §4.9 κ₁ computed on the target build
  against the cached CPU value (κ₁ parity); the cache entry is reused across builds
  regardless (§6.5). The roughly $10^{-13}$ total-current agreement 0132 and earlier
  provider work report is context, not a tolerance proved for every quantity. The bound
  is reported only. Any atomic studies chosen for the GPU (L3–L5) follow in the same
  session (§6.7).
- **L6 Segment near compression** (`benchmarks/sep20/segment.py`, driving
  `run_short_hacc.py --start-plane`). From the saved field at 3.969 m (the plane
  before maximum peak power) to 4.05 m, with the absolute fill, frame and launch
  preserved (as `benchmarks/sep12/segment.py`), run from the identical launch: the
  rate model and SAE `tdse_native` with the selected research record (scale 1.537,
  time collar), the SAE run twice, on the full lattice and with the ±30 fs atomic
  window, on free CPU nodes; haCC(1) and then haCC(5), on the rented GPU within the cap
  (§6.7). All haCC runs use the L4 rule and count, the 60 fs window and $q=8$ at
  `rtol=1e-2`, and are built through the new `tdse=` entry point, so the haCC runs
  pass the run-time consistency gate from the L3 cache entry. They write resumable
  output but are not deliberately interrupted.
  TDSE segment runs use final-only requested output, save every accepted endpoint
  and enable the full archive; compare runs at common endpoints or explicitly
  documented post-processing positions, not by matching their plane indices.

  *Mechanical gates:* every run completes with no failed calls, a consistent atomic
  and logical ledger and finite fields; the haCC runs record the consistency result and
  cache use. Checkpoint-resume is validated on free CPU nodes with the full-lattice SAE
  segment run: interrupted once and resumed, it preserves the saved field and
  identities exactly, charges fresh startup, and agrees with the uninterrupted endpoint
  within SEP 12's interior-restart agreement (Q18). Adams history is not persisted
  (SEP 12), so the resumed trajectory and endpoint need not be bitwise identical. The
  haCC resume path is covered by the fake-provider end-to-end test (§7.1 item 17); the
  binding's resume logic is model-independent. The rate continuation agrees with the
  parent endpoint within SEP 12's interior-restart agreement.

  *Sensible-optics gates* (sanity bounds, not accuracy claims; a failure stops the
  SEP for review): the endpoint full-band energy of each TDSE run does not exceed
  its launch energy by more than 1% (Q18). This is a sanity allowance: `rtol=1e-2`
  controls a local field error estimate, not a global energy error. The source
  gate and approximate closure also need not preserve the exact passive atom's energy
  balance; it lies
  within 5% of the rate endpoint, the size of the differences SEP 12's SAE segment
  showed; and the 90–200 nm energy is within a factor of two of the rate and SAE
  values.

  *Reported, not gated:* endpoint peak power, FWHM and RMS duration, 90–200 nm
  energy and spectrum, complex spectral differences between every pair, in
  particular **haCC(1) against haCC(5)** (what the extra ionic channels change),
  haCC against SAE (what correlation and the fixed scale change), and the two SAE
  runs against each other (what the 60 fs atomic window and the truncated far tail
  change in propagation, which should be negligible); calls, wall,
  memory and GPU spend; the neglected-current bound diagnostics; any domain flags.
  Each difference is reported beside the preset
  sensitivities, the window and node checks, and the unmeasured haCC(5) quadrature and
  ionic-continuum errors, which remain stated limitations rather than inferred
  successes (Q18). The 1.22% total-current sensitivity is not an error bar on these
  differences. If the haCC(5) segment does not fit the cap, the
  comparison is reported for what ran, and the projection goes to the maintainer.
  This comparison is the end of the SEP's validation.

#### 7.3 Benchmarks and evidence

Each level writes a manifest with input hashes, solprop and provider revisions,
preset identities, build and execution space, GPU model and driver for rented runs,
thread settings, host load and free memory before and after, wall, user and system
CPU, peak RSS and device memory, call ledgers, outputs and, for rentals, the spend
and the confirmation that the machine was released. Timings follow
`docs/dev/benchmarks.md` §5: a measured window is reported as measured, and no
segment's cost is extrapolated to a full-run cost in this SEP. The new drive slicing, source gate and provider path do
change work done at each source evaluation. Record their measured windows and costs in
the benchmark ledger; do not infer a full-run cost or an unmeasured speedup.
The L2 witness also compares required versus full output for both haCC
models on identical fields: consumed currents, first-order shadows and nonlinear
remainders must agree bitwise on the same build. Record provider time, archive
write time, actual array bytes and peak host/device memory separately. Enable full
archiving for the L6 segment evidence; archive existing calls rather than adding
diagnostic evaluations. The optional installed-provider test covers full/compact
adapters on tiny SAE and haCC cases before the expensive ladder.

### 8. Documentation

- A new manual page, `docs/guide/media/tdse-hacc.md` (helium haCC response):
  §4's theory in its current form, the two models and their difference, why the
  scale is one, presets against custom bases, the presets' accuracy and validity
  domain, why the neglected-current bound is a diagnostic, the atomic window and its
  gate, the consistency check (warn and refuse bounds and their reasons, skipping it,
  cost, cache location and invalidation, provenance), research controls without the
  closure and what their current omits, the provider contract in user terms, the
  run_short variant and how to start it from a saved plane, measured per-call costs
  and memory on CPU and GPU, the cost warning, the hardware advice (a GPU selected by measured FP64 workload
  performance), the segment evidence, and accuracy and limitations (window,
  domain, node error, no full-run evidence).
- `docs/guide/media/tdse-configuration.md`: the `model` field, the model-dependent
  `source_filter` default, `HaccBasis`, `hacc_preset`, `hacc_basis`,
  `TDSEAtomicWindow`, `TDSEConsistency`, the refusals, warnings and flags, the error
  mapping, and the new entry-point keywords (for SAE too), including the cost of the
  33-node default; natural-step TDSE output and explicit overrides.
- `docs/guide/media/tdse-capillary.md` and `tdse-installation.md`: the provider
  revision with haCC and its CPU and GPU build requirements; `SOLPROP_CACHE_DIR`.
- The TDSE manual and output guide document `TDSEArchive`, full versus required
  products, the streaming reader, observable units and model-specific meaning,
  roles and sessions, storage estimates, write failures and a truncated last group.
  Explain that a predictor response is not necessarily the response to a saved
  endpoint, that the optical ledger, not the archive, accounts for calls, and that
  atomic wavefunction histories are absent.
- `benchmarks/sep20/README.md`: how to regenerate the parent, run each ladder level on
  CPU or in a scripted rented-GPU session, run the L2 witness on a rented machine, and
  run `run_short_hacc.py` at full length.
- Docstrings on every new public name; `CHANGELOG.md` under Unreleased citing this
  SEP and the `source_filter` change; `docs/dev/dependencies.md` for the pin.
- No gallery example: even the segment takes hours.

### 9. Compatibility and impact

- Existing lower-level SAE construction keeps its defaults and, without new
  options, its records, descriptors, identities and resume keys. The new `tdse=`
  convenience path uses natural-step output for SAE and haCC (§5.8); record its
  resolved itinerary because save landings can change the adaptive trajectory.
- `TDSEGasResponse.source_filter` becomes model-dependent by default (§6.2), a small
  breaking change in its annotation only.
- The atomic window is new, on by default for haCC and off by default for SAE.
- Output `SCHEMA_VERSION` 17 → 18 and a kernel schema bump (§6.2), using the next free
  versions at integration if concurrent work has already advanced either.
- `prop_capillary` and `from_capillary` gain the two response keywords and an
  omission-sensitive `step_planes` default; `Simulation.run`, `prop_capillary` and
  `check_consistency` gain `tdse_archive`. Existing TDSE runs on non-field-scale
  rules now emit a `TDSECostWarning`.
- Raw archives are opt-in, append-only, separately versioned and excluded from
  scientific identities. Their full provider output can increase work and memory;
  qualification establishes that consumed responses are unchanged.
- haCC runs write a small cache under the user's cache directory (§6.5), shared by
  every build of one provider revision.
- The provider pin bump brings the provider's own breaking changes (closed-object
  errors, plain-float arguments); the binding adapts and solprop's rerun SAE tests
  cover them. The §5.6 error mapping applies to SAE plans too.
- Luna.jl has no TDSE response, so there is no new difference to record.

### 10. Alternatives considered

- **A dense fixed save grid for expensive TDSE runs.** The existing natural-step
  output retains every accepted endpoint without imposing extra landings. Explicit
  fixed planes remain available when scientifically required.
- **Archive only accepted endpoints, or rerun the atom when saving.** This loses
  expensive rejected/predictor results or adds expensive evaluations. Capture the
  actual provider call and label its optical role instead.
- **Put all raw arrays in the normal optical-plane datasets.** Calls and accepted
  planes have different counts and meanings; a linked sidecar keeps ordinary
  optical processing independent and streams results before a plane is accepted.
- **Always request full atomic output.** Full diagnostics are valuable but can
  add provider work. Full archiving is an explicit research option, enabled in the
  supplied script; required-only archiving and disabled archiving remain available.
- **A transactional archive with charge reconciliation.** The review's first design
  flushed pending and completion markers around each charge, reconciled an archive
  high-water mark with optical checkpoints at resume and recorded coverage
  discontinuities. Simplified at the maintainer's direction (2026-10-17): the archive
  is a diagnostic history, the optical ledger already accounts for every call, and an
  append-only file whose reader flags a truncated last group keeps everything the
  research use needs at a fraction of the implementation and test cost.
- **Keep `remainder_scale` free for haCC.** Rejected: §4.2. A tunable scale would
  invite calibrating away model error and double counting.
- **Gate on the neglected-current bound.** Rejected: the bound is loose and partly
  empirical (§4.4), and would refuse well-converged runs; the presets' ladders give
  measured sensitivities, with the limitations in §4.4.
- **Admit presets only.** Rejected by the maintainer: solprop is a research code, and
  a user must be able to try a different box, lattice or channel content. Custom
  bases are warned and recorded as unqualified instead.
- **Refuse custom models without the escaped-electron closure.** Withdrawn by the
  maintainer (2026-10-17): the velocity-gauge control and other closure-less bases are
  legitimate research comparisons. They are admitted with a warning that names the
  missing closure, run through the full output with the in-box first order, and are
  recorded with `closure: false` (§4.3).
- **Refuse an explicit full-lattice haCC run (`atomic_window=None`).** Withdrawn
  (2026-10-17): the full lattice is a legitimate research choice and the
  wider-reference control for L5; only the default is windowed.
- **Use the full lattice as haCC's default.** The motivating case uses the
  shorter requested duration to reduce cost, subject to L5. Explicit wider
  durations are supported for validation and research; padding is not assumed
  physically empty (§4.6).
- **Taper outside the requested window instead of inside it.** It keeps more of the
  far tail but lengthens the atomic window beyond the request; the field in the
  inner tapers is already below 1% of its peak.
- **A shorter `trange`.** It would shrink the optical window the maintainer works
  with; the atomic window reduces the atomic cost without changing the optics' grid.
- **A sampling study.** Dropped: the interior interpolation bound is already
  $7\times10^{-6}$ at $q=8$, and the realised step at $q=8$ lies inside the presets'
  step ladder. The realised RK4 subdivision and endpoint interpolation are still
  recorded and tested (§4.7).
- **Keep the presets' $q=32$ sampling.** Replaced (2026-10-17) by $q=8$: the provider's
  integer subdivision makes $q=32$ take about 6% more atomic steps and four times the
  samples for no accuracy benefit (§4.7).
- **A haCC(5) node study.** Dropped by the maintainer: haCC(5) follows haCC(1).
- **No run-time consistency check; rely on the validation ladder.** Rejected by the
  maintainer: a custom basis or a future preset would not have been through the
  ladder, and the errors the check catches are cheap to detect and expensive to
  propagate.
- **Refuse every run with $|\kappa_1-1|>0.1$, and key the cache by build.** Revised by
  the maintainer (2026-10-17): a 10% refusal would stop exploratory bases that are
  merely less accurate, and a per-build key would repeat the check on every rented pod.
  The gate now warns above 10%, refuses above 30% (gross errors), can be skipped, and is
  cached per configuration and provider revision (§5.4, §6.5).
- **Gate on the cubic ratio κ₃ as well.** Withdrawn by the maintainer (2026-10-17):
  the 4 fs pulse and the instantaneous carrier-γ reference bias a broadband fit by a
  few per cent, which with haCC(1)'s −5% Kerr could refuse the main model for a
  non-error; the gross errors the gate exists for appear in κ₁, and the provider has
  already validated γ. κ₃ is measured once per preset as evidence (L3).
- **A separate L2 timing rental.** Replaced (2026-10-17) by Pyroraptor's haCC timing study, which
  measures the same per-call costs on the same hardware classes under its own budget.
- **An uncached check on every run.** Rejected: it would add hours of haCC(5) CPU to
  every run of an unchanged configuration.
- **Put the override in the scientific identity.** Rejected: a policy switch would
  then block resumes and split identical physics into different records.
- **Refund every `ValueError` or `TypeError`.** Rejected at review (Q21):
  exception type does not prove that native work never began. Every attempted
  evaluation remains charged; interface errors retain a separate classification.
- **A separate `HaccGasResponse` class.** It would duplicate admission, descriptor,
  output and source logic for one changed atomic preparation; the discriminator keeps
  one route and lets the existing refusals apply to both models.
- **Infer the model from the basis type, with no `model` field.** Explicit beats
  implicit, and the field makes the choice visible in every serialised record.
- **Offer `optmat_cubic`/`tdse_shadow` for haCC with the in-box third order.**
  Rejected: it would subtract the wrong cubic term (§4.3).
- **Refuse many-node rules with a TDSE response.** Rejected: a dense rule is a
  legitimate reference; the warning states the cost instead.
- **Refuse runs beyond a preset's domain.** Rejected: one marginal call would end
  a long run; the flag keeps the evidence.
- **An exact pulse- or grid-relative tested interval for the presets.** Proposed and
  withdrawn on 2026-10-17: the presets' error accumulates relative to the pulse peak,
  so an interval on the qualifying grid means nothing for arbitrary grids, pulse
  positions and shapes, and a pulse-relative one needs a definition of the pulse's
  timing that a propagated field does not supply. Duration and peak intensity are
  the simpler, honest flags (§4.6).
- **Mode-averaged one-node rule.** Cheapest, but SEP 19 measured it 85% wrong in the
  source for this kind of response.
- **Run the haCC segments on free CPU nodes.** Rejected by the maintainer: the GPU is the
  primary target and is expected to be far faster; the A40 result does not transfer
  to H200 or MI300.
- **Keep every one-off atomic study on CPU.** Revised by the maintainer (2026-10-17):
  waiting hours on CPU for a check a GPU finishes in tens of minutes slows research for
  little saving. Compute-heavy studies may join a scripted GPU session when 0132's costs
  show a large gain; development stays on CPU so that no rented GPU sits idle (§6.7).
- **Test checkpoint-resume on a rented haCC segment.** Replaced (2026-10-17): it pays
  fresh startup calls on rental to test model-independent plumbing. The SAE segment on
  CPU and the fake-provider end-to-end test cover it (§7.2 L6).
- **Include full runs in this SEP.** Rejected by the maintainer: the segment shows
  the route works; full runs, their tolerance study and their cost are the
  maintainer's later work.

### 11. Phases

The provider deliverables are available, so there is no separate provider phase.

**Phase 1 — Records, binding, window, consistency check and entry point
(`feature/sep-0020-hacc-binding`).** The provider pin bump; the records, seam, binding,
atomic window, κ₁ consistency gate and cache, entry-point keywords and natural-step
TDSE output of §6.1, with §7.1's tests 1–20, the manual pages and the changelog, against
fake providers, plus the optional installed-provider test. The pin moves to a provider
revision at or after `b8fc0c16`, and to one containing Pyroraptor's haCC timing study (its updated preset
`window` and timing) once that is integrated; if Phase 1 lands first, a later pin bump adopts
it. Slice checks per AGENTS during development. Resolve the Phase 1 review questions
below before declaring the consumer contract ready. The full gate, slow lane, coverage
lane and Kokkos-selected slow lane run once at the end of the SEP, as AGENTS requires;
after a failure rerun the affected modules.

**Phase 2 — Raw TDSE archive (`feature/sep-0020-tdse-archive`).** `TDSEArchive`, the
append-only sidecar writer, its versioned serializers and streaming reader, trial roles
through the backend and multistep hooks, and the run UUID and relative-path reference in
the optical file (§5.8), with §7.1's tests 21–22 and the output-guide pages. It runs
alongside Phase 3 and must land before Phase 4, whose segment enables the full archive.

**Phase 3 — Atomic-only studies (`feature/sep-0020-hacc-witness`).** L1 on free CPU
nodes. L3, L5 and L4 on free CPU nodes (about 520 core-hours without the Q9
extension), or, where Pyroraptor's haCC timing study's measured costs show a large wall-time gain, as one
scripted rented-GPU session together with the L2 witness, within the US$100 cap and
after a recorded projection (§6.7). The κ₁ values of both presets recorded and cached,
the κ₃ and leakage evidence reported; the window check, rule and node count recorded in
the phase report; the CPU or GPU choice and its cost recorded.

**Phase 4 — GPU witness, segment and run_short variant
(`feature/sep-0020-hacc-segment`).** The recorded segment projection from Pyroraptor's haCC timing study's
timing; L2 at the start of the first rental on an H200 or MI300-class GPU, unless the
Phase 3 GPU session already ran it on the same build;
`run_short_hacc.py` and `segment.py`; L6, with haCC(1) and then haCC(5) on the GPU within
the US$100 cap and rate and SAE on free CPU nodes; the comparison report and the manual
page's evidence section. The SEP becomes Final after this phase, the final qualification
lanes and disposition of every open review question, with the recorded backend
exception. If haCC(5)'s segment is deferred for budget, record that explicit validation
exception and its follow-up on the roadmap before Final; do not report an unperformed
comparison as passed. Full runs follow outside it.

### 12. Open questions

**Resolved (2026-10-16, by the maintainer):**

- **Q4.** `backend="auto"` resolves to `numpy_ref` for a TDSE response, logged and
  recorded (§5.2).
- **Q5.** The default 33-node rule is admitted with a TDSE response, documented as
  expensive, and warned about at run time with the row-count cost reason (§5.2,
  §5.6).
- **Q6.** A field above a preset's intensity bound is flagged and warned, not
  refused (§5.6); the bound is the preset's `max_peak_intensity`.
- **Q8.** Stop the ladder for review if $\kappa_1$ or $\kappa_3$ differs from 1 by
  more than 10% for either model, and always report the values (§4.8, §7.2 L3). The
  reasons: haCC(1)'s Kerr response is about 5% below haCC(5)'s, reference uncertainty is model-
  and quantity-specific (§4.8), and real errors tend to be large (a factor of
  two, a sign, the density). The same check, with the same threshold, runs before
  every haCC propagation, cached per configuration, refusing unless overridden and
  recording its values and the override in the provenance (§4.9, §5.4, §6.5).
  *Amended 2026-10-17 by the maintainer:* the run-time gate uses κ₁ only (one row);
  κ₃ is measured once per preset in L3 over the fundamental band and reported as
  evidence, never gated (§4.8). *Amended again 2026-10-17 by the maintainer:* at run
  time $|\kappa_1-1|$ above 0.1 warns and above 0.3 refuses by default
  (`TDSEConsistency(warn, refuse)`), `enforce=False` skips the check, and the cache key
  excludes the build, so each preset pays once per provider revision. L3 still stops
  the ladder for review above 0.1.
- **Q9.** If no node count from the base set meets the L4 tolerances, go up to
  10–16 nodes and take the smallest count that meets them, despite the higher cost
  per call (§6.7, §7.2 L4).
- **Q11.** `source_filter="time_collar"` is the default for haCC (§5.1).
- **Q12.** No SAE full run; the segment comparison is enough. More broadly, this SEP
  runs no full runs at all; its validation ends with the segment (§2, §7.2).

**Resolved (2026-10-17, against the provider's implemented contract at
`b8fc0c16`):**

- **Q1.** The closure is qualified for helium with a mixed gauge:
  `info.total_current_qualified` is true and solprop gates on it. The provider returns
  the in-box first-order shadow; the finite-tail qualification of its equality to the
  closed-current derivative is retained in §4.3 and Q15. The `neglected_current_bound` exists but is loose and partly
  empirical, so it is a recorded diagnostic, never a refusal; accuracy comes from
  the presets' measured sensitivities, `accuracy` 1.22% and
  `box_sensitivity` about 0.9%, with the limitations in §4.4.
- **Q2.** The presets' domain is an 800 nm drive with its spectrum below the helium
  threshold (run_short's 90 nm edge lies inside it), peak intensity up to
  $8\times10^{14}$ W/cm², over a 59.2 fs ladder window. Both presets fit the device
  route (208 of at most 256 inner nodes).
- **Q3.** Names, identities, errors and pin: `hacc.helium_preset(1 | 2)` returning
  `HeliumPreset`, identities `hacc-helium-1-v1-28f0eae146b1d665` and
  `hacc-helium-5-v1-5853d5d50c187aa9`, `ResponsePlanInfo`, `RowMonitors`,
  `TotalCurrentResponse`, and `DrivenError` for closed objects and native failures;
  the pin moves to a revision at or after `b8fc0c16` (§5.7).
- **Q7.** The provider gives no size estimate before preparation. solprop admits the
  model's measured ledger after preparation and each plan's after construction,
  before any evaluation, and documents the presets' measured memory so users can
  size the limit beforehand (C7).

**Maintainer decisions, 2026-10-17:**

- **Q10, hardware.** Rent a GPU (H200 or AMD MI300-class) for the haCC segment runs
  only, with a starting cap of US$100 in total; one-off checks and studies use free
  resources (free CPU nodes and the A40); measure first, then decide how far to go
  within the cap; spending beyond US$100 needs a new approval. The first rented step
  is a short per-call timing of haCC(1) and haCC(5) (closure and order-one shadow,
  the presets' numerics, 5–8 rows), then the segment. The A40 result is not
  representative of H200 or MI300. The earlier "CPU first, GPU only past seven days"
  rule is withdrawn (§6.7, §7.2 L2, L6). *Amended 2026-10-17 by the maintainer:* the
  per-call timing comes from Pyroraptor's haCC timing study (5, 6, 8 and 33 rows on an H200 and
  an MI325X, under its own US$60 budget); SEP 20 rents only for the segments, opening
  the first rental with the short L2 witness inside the US$100 cap. *Amended again
  2026-10-17 by the maintainer:* rented GPUs may also run compute-heavy one-off checks
  when they are much faster (rather than waiting six hours on CPU for a 30-minute GPU
  task): L3, L4 and L5 may form one scripted GPU session with the L2 witness where
  0132's measured costs show a large gain, with a recorded projection, inside the
  US$100 cap. Long development cycles stay on CPU, and no rented GPU is left idle.
- **Q13, the atomic window.** Reduce the atomic window as far as is feasible while
  giving reasonable results; the maintainer typically runs with 60 fs. The 102.5 fs
  is the padded FFT grid, not the physics window. The proposed default drives the atom over the
  requested `trange` (±30 fs), with a 5 fs Planck taper of field and source at its
  edges: about 60/102.5 of the steps by default. Q20 subsequently permits explicit
  wider durations for validation and research. The field in the
  run_short bundle is above 1% of its peak only within about ±22 fs; the maximum-UV
  plane's weak far tail is truncated, and L5 and the SAE control in L6 show whether
  that matters. Coverage gap: the presets' ladders ran from −51 to +8 fs, so the
  post-peak part from +8 to +30 fs, where escape bursts accumulate, is outside them.
  Warn-and-record covers it until the provider's re-check lands: the ±30 fs preset
  re-check, haCC(1) only, on free CPU nodes, is scheduled as Part D of Pyroraptor's haCC timing study, which updates the presets' `window` to the tested duration (Q17; §4.6, §5.3,
  §7.2 L5).
- **Q14, the node study.** Minimise the runs: no radial-node study with haCC(5),
  which follows haCC(1); a minimal haCC(1) study (Gauss rule, 5, 6 and 8 nodes
  against 16 at the peak-power plane, confirmed at the maximum-UV plane), about 390 CPU
  core-hours for the base node study (about 365 at the later $q=8$ sampling, §6.7). haCC(1) is probably the main target; whether haCC(5) is used in
  production depends on how much longer its runs are after optimisation (§6.7,
  §7.2 L4).
- **Custom models.** Accept any haCC helium model and basis, not only presets. A
  preset carries its stable identity and stated accuracy; a custom basis gets an
  identity derived from its full definition, is marked unqualified with no preset
  accuracy, and is warned and recorded rather than refused. A model whose
  `total_current_qualified` is false (argon, the velocity gauge) is still refused, as
  a physics refusal (§4.3, §5.1, §5.6). *Amended later on 2026-10-17 by the
  maintainer:* a custom helium model without the closure (the velocity-gauge control,
  or a gauge radius that does not admit it) is admitted with a warning naming the
  missing closure and recorded with `closure: false`; argon remains unrepresentable,
  and a preset always has the closure.

**Resolved during review (2026-10-17, by the maintainer):**

- **Q20 — Atomic-window representation.** Add an explicit duration for validation
  and research. Its capillary default is the requested `trange` (60 fs here),
  resolved into the record; direct construction must supply a duration if the
  original request is unavailable. Wider references use the same optical grid
  and their own explicit duration (§5.3).
- **Q21 — Attempt accounting.** Validate before invocation, then charge every
  attempted provider evaluation without refunds. Classify interface errors
  separately, including `ValueError` and `TypeError`; preparation and local
  validation before an evaluation remain uncharged (§5.6).

**Resolved output extension (2026-10-17, maintainer request):**

- **Q22 — Every accepted optical step.** Reuse `step_planes=1` with final-only
  requested output as the new TDSE convenience default; preserve explicit overrides
  and existing lower-level defaults. Saving must not add atomic calls (§5.8).
- **Q23 — Raw atomic diagnostics.** Add opt-in full/required HDF5 sidecar archives
  for SAE and haCC, including every actual input/result and rejected attempts,
  with bounded storage buffers, provenance and recovery tests (§5.8). Full means
  the provider's complete observable record, not wavefunction trajectories. The
  AI review specifies the interface and recovery details; qualification remains
  Phase 1 plus the L2/L6 evidence, not a claim of existing implementation.
  *Amended later on 2026-10-17 by the maintainer:* the archive is simplified to an
  append-only file with one group per evaluation and a session group per run or
  resume; it does no charge accounting and has no transaction markers or cross-file
  reconciliation, and its reader flags a truncated last group (§5.8, Phase 2).

**Resolved at the coordinating agent's re-review (2026-10-17, maintainer decisions):**

- **Q15 — Shadow leakage.** Resolved by provider evidence (§4.3): Pyroraptor report 0126
  §B.2 derives the first-order identity up to terms bilinear in the states' tails at
  $R_c$, about $5\times10^{-10}$ in amplitude for the ground root at 16 bohr; §B.3
  measured the leakage at $5.7\times10^{-12}$ of the peak against a $2.8\times10^{-5}$
  third-order remainder for haCC(1) on the run_short peak-power field scaled by 0.01;
  `test_order_one_is_the_in_box_shadow` covers both models on a small basis. Residual,
  owned by L3: the preset-scale haCC(5) leakage term from the λ ladder, which stops the
  ladder for review if not negligible against the weakest retained node's source. A
  custom basis with appreciable support at $R_c$ is outside this evidence and is
  unqualified anyway.
- **Q17 — Domain evidence.** Flag a preset run only when its atomic-window duration
  exceeds the preset's `window`, a call's peak input intensity exceeds
  `max_peak_intensity`, or the carrier differs; never refuse (§4.6, §5.6). A
  grid-relative tested interval (an earlier proposal of the same day) is withdrawn: the
  error accumulates relative to the pulse, so such an interval means nothing for
  arbitrary grids and pulses (§10). Pyroraptor's haCC timing study Part D re-runs the haCC(1) ladders over
  ±30 fs and updates `window` to about 60 fs; haCC(5) carries it over without its own
  ladder, as the provider states. Until then a default 60 fs run is flagged. The flags
  do not establish accuracy for other pulse shapes, timings or carriers.
- **Q16 — Valid diagnostic protocol.** Now an implementation obligation, not an open
  decision. Phase 1 freezes the κ₁ protocol (spacing, carrier phase, endpoint handling,
  supported-band rule, norms and absolute floors) with a small independent extraction
  witness before expensive checks run; the L3 κ₃ extraction protocol (fundamental-band
  rule, λ-ladder and fit residual limits) is frozen before L3 runs. When the check runs,
  an invalid extraction refuses whatever the bounds; only a valid model/reference
  mismatch is subject to `warn` and `refuse`, and `enforce=False` skips the check
  entirely.
- **κ₁-only gate (Q8 amended)** and **L2 from Pyroraptor's haCC timing study (Q10 amended)**, as recorded
  under those questions; the raw archive becomes its own Phase 2 (§11).

**Resolved by the maintainer (2026-10-17), after the high-level review:**

- **Q18 — Numerical comparison limits.** The segment's full-band endpoint energy may not
  exceed its launch energy by more than 1%; the SAE checkpoint-resume endpoint must
  agree with the uninterrupted run within SEP 12's interior-restart agreement (§7.2
  L6). L5 uses full retained-band source differences; L4 is reported as agreement with
  the finite 16-node reference, not a certified error; the omitted haCC(5) node and
  reference-convergence studies remain stated limitations rather than inferred
  successes.
- **Q19 — Reproducible evidence.** The evidence records input hashes (the pump file, the
  parent output, the exported planes), the solprop and provider revisions and
  regeneration instructions, with the provider's evidence cited by report and revision.
  Redistributable inputs are not required: the maintainer's measured pump stays private
  (§7.2 L1).
- **Lighter consistency gate, simplified archive, $q=8$ sampling, resume testing on CPU,
  research controls without the closure, full-lattice haCC runs and GPU sessions for
  heavy studies**, as recorded under Q8, Q23, Q10 and the custom-models decision and in
  §4.3, §4.7, §5.4, §5.8, §6.7, §7.2 and §10.

**Open:** none. Reviewed records completion of the review, not acceptance or numerical
qualification; decisions changing Part I after acceptance follow SEP 0.

The haCC(5) segment's fit within the US$100 cap is decided from Pyroraptor's haCC timing study's timing. A projection beyond
the cap goes back to the maintainer. This review does not launch compute or spend that
budget.

### 13. References

1. V. P. Majety, A. Zielinski and A. Scrinzi, "Photoionization of few electron
   systems: a hybrid coupled channels approach," *New Journal of Physics* **17**,
   063002 (2015). doi:10.1088/1367-2630/17/6/063002. The haCC ansatz.
2. V. P. Majety, A. Zielinski and A. Scrinzi, "Mixed gauge in strong laser-matter
   interaction," *Journal of Physics B* **48**, 025601 (2015).
   doi:10.1088/0953-4075/48/2/025601. Mixed gauge and the failure of velocity gauge
   with frozen ionic functions.
3. A. Scrinzi, "Infinite-range exterior complex scaling as a perfect absorber in
   time-dependent problems," *Physical Review A* **81**, 053845 (2010).
   doi:10.1103/PhysRevA.81.053845.
4. J. C. Travers, T. F. Grigorova, C. Brahms and F. Belli, "High-energy pulse
   self-compression and ultraviolet generation through soliton dynamics in hollow
   capillary fibres," *Nature Photonics* **13**, 547–554 (2019).
   doi:10.1038/s41566-019-0416-4. Application context.
5. M. Kolesik and J. V. Moloney, "Nonlinear optical pulse propagation simulation:
   From Maxwell's to unidirectional equations," *Physical Review E* **70**, 036604
   (2004). doi:10.1103/PhysRevE.70.036604. The current source, as re-derived in
   SEP 12 §4.6.
6. A. Scrinzi, "tRecX — an environment for solving time-dependent Schrödinger-like
   problems," *Computer Physics Communications* **270**, 108146 (2022).
   doi:10.1016/j.cpc.2021.108146. Independent-reference method context for haCC.
7. H. A. Bethe and E. E. Salpeter, *Quantum Mechanics of One- and Two-Electron
   Atoms* (Springer, 1957). doi:10.1007/978-3-662-12869-5. The dipole sum rule for
   a two-electron atom; the complete-spectrum statement does not guarantee it in a
   truncated haCC space.
8. M. Puchalski, K. Piszczatowski, J. Komasa, B. Jeziorski and K. Szalewicz,
   "Theoretical determination of the polarizability dispersion and the refractive index
   of helium," *Physical Review A* **93**, 032515 (2016).
   doi:10.1103/PhysRevA.93.032515. Accuracy of primary linear-polarizability data; not a
   replacement for the actual optmat model's provenance.

## Part II — Implementation record

### Phase 1 — records, binding, window, consistency check, entry point (2026-10-17)

**Scope.** Phase 1 as section 11 defines it, without the archive (Phase 2) and the
`benchmarks/sep20/` evidence scripts (Phases 3 and 4). Implementer: Opus 5.5 (AI) for John Travers. Branch
`feature/sep-0020-hacc-binding`, based on `8d8e7a4`; not yet merged (the merge commit
is recorded at integration).

**Delivered.**

- Records (`src/solprop/nonlinear/tdse.py`): `HaccBasis` with its `kind`-tagged
  identity, `label` and `qualified`; `TDSEAtomicWindow`; `TDSEConsistency`;
  `TDSEGasResponse.model`, `atomic_window` (`"auto"`), `consistency`
  (`record=False`) and the per-model `source_filter` default; the rules of the
  section 5.1 table. `TDSEBasis` now shares the radial-axis validator.
- Exceptions (`src/solprop/exceptions.py`): `TDSEConsistencyError` (a
  `MaterialError` carrying `result`), `TDSECostWarning`, `TDSEDomainWarning`,
  `TDSEConsistencyWarning`, `TDSEUnqualifiedModelWarning`.
- Seam (`src/solprop/translate/tdse.py`): `solprop.tdse-descriptor.v3` for haCC and
  for any window, `AtomicWindowRecord`/`resolve_atomic_window`, `lattice_times`, the
  pinned haCC setup record, `TDSEDescriptor.model`, `.response`, `.atomic_window`
  and `.source_gate()`, and an explicit `calls` count on `AtomicCallBudget.reserve`
  and `begin_evaluation` for a one-call diagnostic. Kernel `SCHEMA_VERSION` 26 → 27
  (`translate/problem.py`, `docs/dev/kernel-contract.md`, the frozen literal in
  `tests/backends/test_driver.py`).
- Source (`src/solprop/backends/numpy_ref/tdse.py`): the window's gate replaces
  `grids.t_window_fine` as the `time_collar` gate, also in the rounding floors.
- Binding (`src/solprop/driver/tdse.py`): pin `c2833fa…` →
  `272ef730a7dfa5caab8c0ad75235107b976408c1` (Pyroraptor main at dispatch, which
  contains `b8fc0c16`); `_HaccModel` (preset verification of identity, every field
  and the device route; custom `hacc.prepare`; `vendor_blas` refused; the haCC
  surface checked by name); one plan per chunk size with exact plain-type arguments;
  `info` checked against atom, basis, gauge, cut, extents, order, output mode and the
  closure flag; measured model then plan ledgers admitted before any call; window
  slicing, gating and zero padding; model-aware bound and monitor checks (bool
  `non_finite`); per-row bound maxima relative to the peak with `zero_reference`;
  preset domain flags (duration, carrier at preparation; intensity per call, before
  and after the taper); `native`/`interface`/`contract` failure classification on
  every charged attempt (SAE too); `diagnostic_currents` (a temporary one-row plan,
  charged as `diagnostic`); `record_consistency`; `DiagnosticSession` for the
  standalone check; `hacc_preset`, `HaccPreset` (with `calibration()`), `hacc_basis`,
  `pinned_revision`.
- Consistency (new `src/solprop/driver/tdse_consistency.py`): frozen protocol v1
  (`protocol_field`), `reduce_linear` and its validity rules, the build-independent
  cache (`cache_directory`, atomic writes, validated reads), `evaluate_consistency`,
  `gate_run`, `apply_outcome`, `check_consistency`, `ConsistencyResult`, and the
  separate L3 extraction `cubic_evidence`/`CubicEvidence`.
- Driver (`src/solprop/driver/simulation.py`, `interface.py`, `run.py`): `tdse=` and
  `tdse_execution=` on `from_capillary` and `prop_capillary`, the electronic-keyword
  refusals, `auto` → `numpy_ref`, TDSE default statistics, `step_planes: int | None`
  with the final-only natural-step default, the window duration from `trange`, the
  `tdse_entry` configuration record, the run-start `TDSECostWarning`, and the
  consistency gate after the binding and ledger restore and before `run_problem`,
  writing a refused run's record with `open_run_record` (no plane, no checkpoint).
- Output (`src/solprop/output/schema.py`): `SCHEMA_VERSION` 17 → 18; SAE responses
  serialise without `model` and an absent window. Current-version literals in 16
  existing tests move to 18.
- Exports: `solprop.nonlinear`, `solprop.driver` and `solprop.sol`.
- Consequences of the bumps: the SEP 12 harness (`benchmarks/sep12/parent.py` admits
  schema-18 rate parents and maps 17 → 18; `segment.py` gains `response_record`, used
  by `native_campaign.py`, so SEP 12's stored response records keep their SEP 12
  form, and `response_from_record` accepts the new fields); the layers and one-run
  structure figures regenerated with `poe figures`; doctest and version literals in
  `translate/problem.py`, `tests/test_display.py`, `tests/backends/test_numpy_ref_backend.py`
  and `tests/test_sep12_parent.py`. The existing SEP 12 fixtures that run a TDSE
  response on a multi-node rule filter the new `TDSECostWarning` by name.
- Tests: `tests/hacc_cases.py` (records and a fake provider implementing C1–C8);
  `tests/nonlinear/test_tdse_spec.py` (items 1–3), `tests/translate/test_tdse_contract.py`
  (4, 5), `tests/translate/test_tdse_window.py` and the window part of
  `tests/driver/test_tdse_hacc_binding.py` (6), `test_tdse_hacc_binding.py` (7–10,
  19), `tests/driver/test_tdse_consistency.py` (11–13),
  `tests/backends/test_tdse_sources.py` (14), `tests/driver/test_tdse_entry.py` (15,
  16), `tests/driver/test_tdse_hacc_run.py` (17 and the run-level parts of 12),
  `tests/output/test_schema_lineages.py` (18), `tests/driver/test_tdse_natural_output.py`
  (20), and the optional `tests/driver/test_tdse_hacc_installed.py`.
- Manual: new `docs/guide/media/tdse-hacc.md`; `tdse-configuration.md`,
  `tdse-capillary.md`, `tdse-installation.md`, `docs/api/index.md`,
  `docs/dev/architecture.md`, `docs/dev/dependencies.md`; `CHANGELOG.md`.

**Deviations from the proposal.**

1. `tdse_archive` is absent from `check_consistency`, `Simulation.run` and
   `prop_capillary`: it belongs to Phase 2 with the archive itself.
2. A windowed SAE response with an omitted `source_filter` resolves to
   `"time_collar"`, not `"none"`: section 5.1 makes every window require the gate and
   refuses only an *explicit* `"none"`. Unwindowed SAE still resolves to `"none"`.
3. A custom mixed-gauge basis is first planned with the compact output; the
   provider's `ValueError` naming the closure selects the full record (C3 gives no
   other way to know beforehand). The velocity gauge goes to full directly. Any other
   `ValueError` is a preparation failure.
4. A haCC run prepares its binding with `defer_plans=True`: the model is prepared
   and admitted first, the consistency check's one-row plan is built and closed,
   and only then does `build_plans()` build and admit the propagation plans, so the
   run's peak memory is the larger of the two, not their sum (revised at the coordinating agent's review; first implemented with overlapping plans admitted together).
   Propagation calls on a deferred binding are refused until its plans exist. The
   standalone check prepares the model alone.
5. `tdse_execution=None` beside `tdse=` takes `TDSEExecution()` instead of a refusal.
6. Invalid consistency results are never cached (section 6.5 is silent); the next run
   recomputes.
7. `TDSEDomainWarning` is raised once per reason (duration, carrier, intensity) per
   run, not once in all; every flagged call is recorded.
8. The carrier reason compares with `math.isclose(rel_tol=1e-12)`.
9. `check_consistency` returns the decided result without warning or raising; only
   a run's gate warns or refuses.
10. Section 6.8 expected the pin bump to carry provider breaking changes in the SAE
    path. `git diff c2833fa..272ef730` over the SAE modules is empty; only the fake's
    revision, a `DrivenError` stand-in and the error classification changed.
11. Item 17's "fresh startup charges" are tested with `resume_history=False`; with a
    kept Adams history the resume does not repeat the startup.
12. Closure detection keys on the provider's refusal text ("needs the escaped-electron
    closure"); the installed-provider test pins that message and checks that a
    mixed-gauge basis without the closure (R_g = 1 bohr on the tiny basis) takes the
    full-output route against the real provider.
13. The two SEP 12 fixture modules that run TDSE responses on multi-node rules keep a
    filter for `TDSECostWarning` by class alone, because most of their run sites
    refuse before the warning is raised, and each asserts the warning once with
    `pytest.warns` in a dedicated test.

**Decisions.**

- Protocol v1 values (Q16), frozen in code and the manual: 4 fs intensity-FWHM
  Gaussian at λ₀, cosine phase at the centre of a 40 fs window, 10¹¹ W/cm², 25 as
  spacing (1601 samples); band: ω > 0, |Ê| > 10⁻³ max, optmat support; endpoint
  leakage ≤ 10⁻⁶ of the peak over the first and last 2% of samples for field and j₁;
  every band bin of ĵ₁ above 64 ε ‖j₁‖₁; finite ratios. Independent witness: a
  damped and an undamped Lorentz oscillator integrated by SciPy DOP853 give
  |κ₁ − 1| < 10⁻⁶, which fixes the sign of iω, ε₀ and the optics-convention
  conjugation.
- L3 extraction protocol, **confirmed by the maintainer on 2026-10-17** (Q25): fundamental band |Ê| ≥ 10⁻¹ max; λ = 4 ladder residual limit 10⁻⁴
  (tightened from 10⁻² at the coordinating agent's review: at 10¹¹ W/cm² successive orders differ
  by about 10⁻⁴ and the c₇ residual of the λ = 4 row is about 10⁻⁶, so 10⁻² could not
  fail); leakage = ‖c₁‖/‖c₃‖ from the three-term fit, reported beside its
  seventh-order floor 64(‖ĉ₅‖/‖ĉ₃‖)² (a c₇ enters the fitted c₁ as exactly 64 c₇),
  with `leakage_resolved` when the leakage exceeds it.
- Cached κ₁ entries are validated by explicit checks, never `assert`, so `python -O`
  cannot admit a malformed or non-finite entry; the remaining guards of the gate are
  explicit too.
- The κ₃ reference recorded with each preset's result is a placeholder naming L3;
  values come in Phase 3.
- No ADR is added: the lasting choices (descriptor v3, the gate's place in the
  driver, the cache key) are in this SEP and the design pages.

**Verification** (Apple Silicon, macOS 27.0.1, CPython 3.14.5, `uv` environment of the
worktree; tests with `PYTEST_XDIST_AUTO_NUM_WORKERS=6`, one thread per worker).

- Base-commit literals: the v1 and v2 SAE descriptor identities, the serialised
  response record and the schema-17 resume key, and a representative rate-based
  `from_capillary` record, were computed from a `git archive` of `8d8e7a4` and are
  reproduced bitwise (items 4 and 15).
- Focused suites during development: the TDSE spec, contract, window, source,
  projection, march, binding, consistency, entry, run, natural-output and lineage
  modules (about 810 tests), all passing.
- Touched-code coverage (pytest-cov, branch, `coverage-cpu.ini`): 100% of statements
  and branches of `nonlinear/tdse.py`, `translate/tdse.py`, `driver/tdse.py`,
  `driver/tdse_consistency.py`, `backends/numpy_ref/tdse.py`, and of every changed line
  and branch of `driver/simulation.py`, `run.py`, `interface.py` and `output/schema.py`.
- Review fixes (after `5e4be50`): `0f94493`, `e45dbda`, `e64bca2`, `5468998`,
  `dca40c3`, `db190ab`, `c36b54a`, with their tests (malformed cache entries also
  under `python -O`; the ladder limit and floor; the deferred plans and their event
  order and memory; the pinned closure message against the real provider).
  Touched-module coverage stays at 100% (`driver/tdse.py`, `driver/tdse_consistency.py`
  and every changed line and branch of `driver/simulation.py`). After them, on `c36b54a`:
  `ruff check` passed; `ruff format --check` reports only the frozen Part I block of
  Q27; `basedpyright` 0 errors; `numpydoc lint`, `codespell` and `reuse lint` passed;
  `poe test` 14,306 passed, 173 skipped (14 min, the installed-provider test
  included); the strict Sphinx build succeeded.
- Gate, on the tree as first reported (`ba37497` plus this report): `ruff check --fix`,
  `ruff format`, `ruff check`, `basedpyright` (0 errors), `numpydoc lint`,
  `codespell`, `reuse lint` and `poe test` (14,273 passed, 173 skipped, 12 min,
  `-n auto` with `PYTEST_XDIST_AUTO_NUM_WORKERS=6`, one library thread per worker,
  including the optional installed-provider test) all passed, and the strict Sphinx
  build (`poe docs`, `-W -n --keep-going`) succeeded. They were run as `poe check`'s
  steps; the combined `poe check` invocation itself is blocked by Q27.
- Installed provider: Pyroraptor `272ef730` built locally (Homebrew LLVM/libomp 23.1.3,
  FFTW 3.3.11, Release, LOCAL, OpenMP, Kokkos 5.2.1, linked to Homebrew `libomp`). The
  optional test passed: both presets resolve with the version-1 identities and equal
  the test copies field for field; the provider's tiny custom haCC(1) basis prepares,
  evaluates three rows through the binding (two plans, two charged calls, zero outside
  the window). By hand, `check_consistency` on that tiny basis gave
  κ₁ = 1.0452 + 3.4×10⁻¹⁰ i at its worst wavelength (19 band bins, `passed`, 27 s on
  2 threads): a contract check of the real path, not an accuracy statement.

**Not run.** The slow lane, the full coverage lane and the Kokkos-selected slow lane
(once, at the end of the SEP, per section 11); any preset-scale atomic call; GPU
builds; Windows.

**Sources checked.** The provider's `hacc/{presets,basis,model,response,_plan,shadow}.py`
and its consumer guide at `272ef730` against C1–C8 (no contradiction found); optmat
helium's susceptibility record (`fit/He`, support 60 nm to ∞) and Kerr coefficient;
the Planck taper of `solprop.grid`. The papers [1–3, 8] were not re-read; the
provider's ladders and costs are quoted, not reproduced.

**Limitations.** No haCC propagation is qualified; the presets' tested `window` at
this pin is 59.2 fs, so a default 60 fs run is flagged; the record's
`angular_order` admits 1–120 for a total-L limit too (the provider admits 0);
`numba_cpu` and Kokkos refuse TDSE as before.

**Open questions.**

- Q24, resolved (maintainer, 2026-10-17): solprop-kokkos follows kernel schema 27
  in a coordinated change integrated with this phase.
- Q25, resolved (maintainer, 2026-10-17): the κ₃ extraction values above (band floor
  10⁻¹, ladder limit 10⁻⁴, the floor's estimate) are confirmed.
- Q27, resolved (master, 2026-10-17): the `run_short.py` excerpt in section 2 is
  fenced as `text` on solprop main (`a8a8bbc`), so `ruff format` leaves it verbatim.
- Q26, resolved (master, 2026-10-17): the pin moves to Pyroraptor `ea2bfcaa`
  (Pyroraptor's haCC timing study: preset `window` 60 fs, device route to 512 inner nodes); the manual's
  haCC page gives 0132's measured GPU costs and memory.

**Handoff to Phase 2 (archive).** The binding has one place per evaluation where the
raw provider object exists before validation (`_ProviderBinding.evaluate`, per chunk,
and `_diagnostic`); capture there, before `_validate_diagnostics`. The exact drive is
`self._drive(fields, chunk)`, the pre-window field is `fields`; the window record
(`descriptor.atomic_window`) and gate give the time map (provider time zero is
`t_s`). Native output selection is `_HaccModel.outputs`; a full archive must request
`"full"` without changing the descriptor's `outputs`, and the haCC full adapter
(`_hacc_currents`) already reads the total and order-one currents from a
`GridResponse`. Failure records carry `classification`. The run's consistency call
is a `diagnostic` evaluation made through `diagnostic_currents`, which an archive
should label. Trial roles need the backend and multistep hooks named in section 6.1;
none were added here.

### Phase 2 — raw TDSE response archive (2026-10-17)

**Scope.** Phase 2 as section 11 defines it: `TDSEArchive`, the append-only sidecar
writer, its versioned serializers and streaming reader, trial roles, the run UUID
and relative-path reference in the optical file (section 5.8), tests 21 and 22 of
section 7.1, and the archive's manual pages. Implementer: Opus 5.5 (AI) for John Travers. Branch
`feature/sep-0020-tdse-archive`, based on solprop main `9017661` (Phase 1 merged);
not yet merged (the merge commit is recorded at integration). solprop-kokkos is
not changed.

**Delivered.**

- Archive (new `src/solprop/output/tdse_archive.py`): `TDSEArchive(path, *,
  outputs="full" | "required")`; schema `solprop.tdse-archive.v1`; the explicit
  serializers `sae.grid_response.v1`, `sae.total_current_response.v1`,
  `hacc.grid_response.v1` and `hacc.total_current_response.v1` (with nested
  plasma, shadow and row-monitor serializers), each field with its SI unit and axes
  and absent fields listed, no pickle; `TDSEArchiveWriter` (one session per run,
  resume or standalone check; one group per evaluation with `prewindow_field`, per
  chunk `drive` and `raw`, an `exception` group, the `outcome` written last; trial
  decisions under `trials`; content-addressed shared values hard-linked; a capped
  HDF5 metadata cache and `writer_memory` for admission; bytes written, array bytes
  and write time); `open_tdse_archive`, `TDSEArchiveReader`, `ArchivedSession`,
  `ArchivedEvaluation`, `ArchivedChunk`. `TDSEArchiveError` (`OutputError`) in
  `src/solprop/exceptions.py`. Exports in `solprop.output` and `solprop.sol`.
- Binding protocol (`src/solprop/translate/tdse.py`): `TDSEBinding.evaluate` takes an
  optional `density` label; `reserve` opens a trial; new `record_trial(accepted=)`.
- Reference backend (`src/solprop/backends/numpy_ref/backend.py`): passes the stage
  density and overrides `_record` to report each attempt's outcome to the binding.
- Binding (`src/solprop/driver/tdse.py`): the archive writer in `prepare_binding`;
  trial numbering; input written before the first provider call and each raw result
  before validation; the full selection for a full archive (`native_outputs`) without
  changing the descriptor, with the haCC closure decoupled from the output mode and
  taken from a custom basis's first plan; failure records; archive buffers in the
  memory ledger and in `estimate_binding_memory(..., archive=)`; the archive
  reference in `metadata()`; consistency diagnostics and `DiagnosticSession` archived
  as `diagnostic`.
- Driver: `Simulation.run(..., tdse_archive=)` (refusals, the session opened before
  binding preparation, the run UUID kept across resumes, the close order and error
  priority), `prop_capillary(..., tdse_archive=)`, `check_consistency(...,
  tdse_archive=)`.
- Test doubles of the binding (`tests/backends/test_tdse_march.py`,
  `tests/driver/test_sep12_sep14_integration.py`, `tests/test_sep12_qualification.py`,
  `tests/test_sep12_segment.py`) and the SEP 12 harness wrappers
  (`benchmarks/sep12/fixture.py`, `segment.py`) follow the protocol; the haCC fake's
  full record gains `vector_potential`, `electric_field` and `outside`.
- Tests: `tests/tdse_archive_cases.py`, `tests/output/test_tdse_archive.py` (writer
  and reader), `tests/driver/test_tdse_archive_binding.py` (binding, diagnostics,
  standalone check), `tests/driver/test_tdse_archive_run.py` (runs).
- Manual: new `docs/guide/media/tdse-archive.md`; `docs/guide/results/output.md`,
  `tdse-configuration.md`, `tdse-hacc.md`, `docs/api/index.md`,
  `docs/dev/architecture.md`, `docs/dev/kernel-contract.md`, the schema-18 docstring
  in `src/solprop/output/schema.py`; `CHANGELOG.md`.

**Acceptance against section 7.1.**

- Item 21 (round trip). All four records read back exactly (dtype, shape, values,
  NaN and infinities, invalid samples, absent fields, every shadow label, int64 and
  bool monitors) over two chunks; SAE (`tdse_native`, `optmat_cubic`, `tdse_shadow`)
  and haCC (preset, custom mixed gauge, velocity gauge) through the binding in both
  selections over several chunks, with the pre-window field, the exact gated drives
  (equal to the provider's received arrays), the gate, the time map and the row
  ranges; a raw non-finite result the binding refuses is stored as returned; the
  source arrays and their remainder are bitwise those of an unarchived binding in
  both selections; and a run's saved fields, plane positions, provider inputs and
  both ledgers are bitwise unchanged with no, required and full archives. Covered.
- Item 22 (labels and failures). Rejected and predictor roles from a run that
  rejects, and the rejected/accepted trial counts, against the logical ledger; the
  corrector role at the binding; a resume appends a `resume` session under the same
  UUID with continuing evaluation identifiers; a raising evaluation (native,
  interface, contract) keeps its input and exception; a full disk or a serialization
  failure stops the binding before the next provider call and a run at its last
  checkpoint (`failed` outcome, resumable), with the provider or contract error kept
  primary when its record also fails; an existing text file, a foreign HDF5 file or
  another run's archive is refused and left unchanged before any provider call; a
  truncated last group (also after a real `os._exit` in a subprocess) is flagged by
  the reader; the optical file reads without the sidecar; the writer keeps no array
  and its traced memory does not grow over repeated evaluations. Covered.

**Section 5.8 requirements.** Interface and exports, refusal without a TDSE
response and at the optical path, in-memory optical output: done. Exact input and
pre-window field; raw result before validation, dtype-preserving, through explicit
versioned serializers; labels (evaluation ID, trial ID and role, z, identities,
plan/build and execution through the shared `binding` record, density, gate, time
map, row ranges; radial nodes and weights in the session context): done. Raised
evaluations, diagnostic role, cache hit writing no evaluation, shared values stored
once: done. Sessions, run UUID and relative path in the optical file, readable
without the sidecar, no charge accounting, flagged truncation, one synchronous
writer with admitted bounded buffers, opened before paid work, write-failure
semantics, bytes and write time: done. Not done here: measuring both modes on the
real provider (Phase 4, Q28).

**Deviations from the proposal.**

1. `src/solprop/backends/multistep.py` is unchanged. The trial outcome reaches the
   binding through the reference backend's override of `StepDriver._record`, which
   every Adams attempt reaches (adaptive, replayed and rejected-for-free), and each
   reservation opens a trial. A new protected hook would have changed the frozen
   `StepDriver` surface and the kernel schema, which the dispatch asked to avoid.
2. The host binding protocol gains `record_trial` and an optional `density` on
   `evaluate`, as section 6.2 anticipates. The kernel `SCHEMA_VERSION` stays 27: the
   protocol is not part of the protected surface, the kernel problem or the stage
   bundles, and solprop-kokkos never holds a binding (it names only
   `_reserve_source_calls`, unchanged). Recorded as Q29 for the coordinating agent.
3. The optical reference lives in the binding's execution record
   (`backend.settings["tdse"]["archive"]` and each outcome's `execution`), not in a
   new `RunMetadata` field, so output schema 18 and every identity are unchanged; the
   schema docstring records it as the optional archive reference section 6.2 names.
4. Completeness uses the evaluation's own `outcome` attribute and the expected chunk
   datasets; the session's `closed` totals are written on a normal close. Neither is
   a cross-file transaction marker.
5. A resume whose recorded archive file is missing starts a new file under the run's
   UUID; an archive first requested by a resume gets a new UUID, recorded in that
   execution's outcome.
6. A standalone `check_consistency` archive is a new file per call (an existing one
   is refused); with `enforce=False` none is opened; a cache hit leaves a session with
   no evaluation.

**Decisions.**

- `required` (the conservative reading of section 5.8): the provider's selection the
  source policy needs, every field of the returned record stored.
- Startup evaluations keep the `startup` role in a rejected trial (taken at an accepted
  state); only predictor and corrector evaluations read back as `rejected`, beside
  their `evaluated_role`. Trial identifiers restart per session; evaluation
  identifiers increase through the file.
- A serialization failure of a result that the binding then refuses is a note on the
  binding's error; otherwise it is raised after validation, before any further call.
  A misuse of the writer (an evaluation not begun) marks it failed like a write.
- Unknown objects inside `info` are stored as `{"type", "repr"}` so a structured
  diagnostic never stops a run; arrays inside `info` become JSON lists.
- An in-memory optical output records the archive's absolute path
  (`relative_to = null`).
- No ADR: the archive's choices are in this SEP, the manual page and the module.

**Verification** (Apple Silicon, macOS 27.0.1, CPython 3.14.5, the worktree's `uv`
environment; `PYTEST_XDIST_AUTO_NUM_WORKERS=6`, one thread per worker; the host was
shared, load average about 20 on 18 cores).

- Focused suites during development: the four archive modules (83 tests) and the
  existing TDSE binding, haCC binding, run, natural-output, consistency, entry,
  march, source, SEP 12/SEP 14 integration, driver-surface, checkpoint and `sol`
  modules, all passing.
- Touched-code coverage (pytest-cov, branch, `coverage-cpu.ini`, on those suites):
  100% of statements and branches of `output/tdse_archive.py`, `driver/tdse.py`,
  `driver/tdse_consistency.py` and `output/__init__.py`; every new line and branch
  of `driver/simulation.py`, `driver/interface.py` and `backends/numpy_ref/backend.py`
  (the remaining misses in those files are pre-existing lines exercised by other
  suites).
- Gate steps on the final tree: `ruff check`, `ruff format --check`, `basedpyright`
  (0 errors), `numpydoc lint` (`poe docstrings`), `codespell`, `reuse lint` and
  `git diff --check` passed; `poe test` 14,415 passed, 174 skipped (11 min 53 s; the
  first full run found the regenerated layers figure missing the new module and two
  driver test doubles without the new protocol method, fixed in `6e0b370` and rerun);
  the strict Sphinx build (`poe docs`, `-W -n --keep-going`) succeeded after two
  cross-reference fixes. Kernel schema 27 and output schema 18 are unchanged.
- A script compared the four serializers' field lists, and their nested records', with
  the provider's dataclasses at the pin: identical.

**Not run.** The slow lane, the full coverage lane and the Kokkos-selected slow lane
(at the end of the SEP, per section 11); the optional installed-provider round trip
(no build of the pinned provider was available in this worktree, and building one
was not cheap); any real preset run; GPU builds; Windows.

**Sources checked.** The provider's public result records at the pinned revision
`272ef730` (`response_plan.py`, `plasma.py`, `shadow.py`, `history.py`,
`hacc/response.py`, `hacc/_plan.py`, `hacc/shadow.py`): the field lists of all four
serializers and their nested records were compared by a script against the
dataclass annotations and agree exactly; units are the provider's documented ones.
The provider's statement that the haCC compact total current is bitwise that of the
full record was read, not measured.

**Limitations.** Parity of the full and required selections, the archive's size
and its write time are established only against fake providers. The `corrector`
role occurs only with a PECE method. HDF5 is not journalled: flushed groups make
a damaged file after a crash unlikely, not impossible. A rerun from scratch needs a
new archive path.

**Open questions.**

- Q28 (Phase 4): measure required/full archives on the real provider (bytes, write
  time, provider cost) and check consumed current and remainder parity on the same
  build, as section 5.8 asks.
- Q29, resolved (master, 2026-10-17): no kernel schema bump. The schema counts the
  kernel-contract field list, the protected `StepDriver` surface and the stage
  records (`tests/backends/test_driver.py` literal, unchanged); the host binding
  protocol is none of these, and compiled backends never hold a binding.

**Handoff.** Phase 4's segment enables the full archive with
`tdse_archive=TDSEArchive(path)`; read it with `open_tdse_archive`. The run's
archive reference is in each outcome's `execution["tdse"]["archive"]`. The fake
records in `tests/tdse_archive_cases.py` carry every provider field; a provider
change of a result record must bump the matching serializer version.

### Phase 3 — atomic-only studies L1–L5 (2026-10-17)

**Scope.** Phase 3 as section 11 defines it: L1 on saved data, and L2 (the GPU
witness, brought forward from Phase 4 as section 11 permits), L3, L4 and L5 in one
scripted rented-GPU session, with CPU references on free nodes. Implementer: Opus 5.5
(AI) for John Travers, who approved the
session and its spend (US$15 within the US$100 cap) on 2026-10-17. Branch
`feature/sep-0020-hacc-witness`, based on solprop main `8933511` (Phases 1 and 2
merged; provider pin Pyroraptor `ea2bfcaa`). **Outcome: L1–L4 pass; L5 fails its
band tolerance, so the ladder stops for review before Phase 4.**

**Delivered** (`benchmarks/sep20/`).

- `parent.py` (L1 export and regeneration), `witness.py` (L2 `witness` and
  `compare`, L3 `weak` and `kappa`), `nodes.py` (L4 `run`, `analyse`), `window.py`
  (L5 `run`, `analyse`), shared `case.py` (the run_short case, plane bundle and the
  never-propagated `tdse=` harness), `calls.py` (every atomic call through
  `prepare_binding`/`evaluate`), `sources.py` (solprop's `atomic_sources` and
  `project_atomic_sources`, with SEP 12's floor per band), `metrics.py` (the
  tolerances and comparisons), `collect.py` (compact evidence), `gpu_session.sh`,
  `README.md`, `PROTOCOL.md` (committed in `38bb2e9`, before any preset-scale call).
- `tests/test_sep20_benchmarks.py`: the band mask, the floor-aware screen, the node
  selection, the device allowance (reproducing Pyroraptor 0132's protocol values),
  parity, linear scaling, bitwise comparison, the L1 statistics check and the
  reconstruction admission (19 tests, default lane).
- `benchmarks/sep20/evidence/`: `l1-planes.json`, `l2-witness.json`, `l3-weak.json`,
  `l4-nodes.json`, `l5-window.json`, `session.json`, `manifest.json` (SHA-256 of every
  raw file kept outside the repository) and `kappa1-cache/` (the two seeded cache
  entries, byte for byte).

**L1 — parent and fields: pass.** The saved rate parent of Pyroraptor's 0126 field
bundle was reused (`parent.h5` SHA-256 `079d37dc…`, equal to its producing
manifest's; pump `d9a512d3…`). Its configuration was rebuilt from the pump: apart
from the schema 17 → 18 layout, the only differences are optmat provenance (0.4.0 at
the parent's run, 0.8.0 now: version and citation strings, and the ionisation-rate
record digests of the rate response), which the export records and admits; the saved
spectra are reused unchanged. The four planes (0, 3.807, 3.969 and 4.05 m; indices 0,
47, 49 and 50, checked against the saved extrema) were exported with their modal and
vacuum spectra, on-axis and 33-node fields; the three planes 0126 also exported are
bitwise equal to its on-axis fields. Every section 2 statistic agrees at its quoted
precision: 0.264 and 0.590 TW, 7.99×10¹⁴ W/cm² on axis at 4.05 m, peak ionised
fraction 5.64×10⁻³, 5.05 µJ in 90–200 nm at 3.807 m, 142 accepted steps (order one),
144 logical calls, two steps over the last 81 mm. The run_short script changed after
the parent ran (a commented-out material selection); both hashes are recorded.

**L2 — GPU witness: pass** (both presets; 6-node Gauss field-scale rows of the
peak-power plane, a 4 fs window with 1 fs tapers, 19,140 RK4 steps; the device input
is the cluster CPU reference's saved field, bitwise).

| | haCC(1) | haCC(5) |
|---|---|---|
| valid rows, `total_current_qualified`, closure, no failure | yes | yes |
| repeated call bitwise (C8) | yes | yes |
| compact = full, bitwise (total, j₁, remainder) | yes | yes |
| j₁/η linear across rows, max deviation (au) | 2.8×10⁻¹⁶ | 3.3×10⁻¹⁶ |
| GPU − CPU, max (au): total, j₁, remainder | 7.2×10⁻¹³, 2.1×10⁻¹³, 7.5×10⁻¹³ | 1.0×10⁻¹², 2.1×10⁻¹², 3.0×10⁻¹² |
| allowance (au): total and j₁ / remainder | 1.7×10⁻⁵ / 3.4×10⁻⁵ | 6.8×10⁻⁵ / 1.4×10⁻⁴ |
| κ₁ GPU − CPU, max over 19 bins | 2.0×10⁻⁹ | 6.4×10⁻¹⁰ |
| model setup, GPU (cluster CPU, 32 threads) | 68 s (124 s) | 188 s (350 s) |
| evaluate, GPU (cluster CPU) | 24.3 s (180 s) | 86.2 s (1,002 s) |
| plan ledgers, required / full | 209 / 231 MB | 1,175 / 1,197 MB |

No difference approaches 0132's 10⁻⁸ au "unexpected" mark. **Q28 (archive cost):**
the full selection cost 0.3% (haCC(1)) and 0.02% (haCC(5)) more evaluate time than
the required one; an evaluation's archive group is 1.00 MB written (0.98 MB of arrays)
in required mode and 1.29 MB (1.26 MB) in full mode, of which the 6×16,384-sample
pre-window field is 0.79 MB, and writing took 4–5 ms; the consumed currents and
remainder are bitwise equal across the modes. These are 4 fs witness calls, not a
propagation's archive.

**L3 — weak field: gate passes; ladder evidence reported.** `check_consistency` with a
fresh cache on the MI325X, each seeding its entry (keys `52b11b66…` haCC(1),
`22158d21…` haCC(5)):

| | haCC(1) | haCC(5) |
|---|---|---|
| \|κ₁ − 1\| (worst at 500 nm; band 500 nm–2.0 µm, 19 bins) | 0.0347 | 0.0189 |
| κ₁ there | 1.0347 + 2.2×10⁻⁷ i | 1.0189 + 2.1×10⁻⁷ i |
| cluster CPU \|κ₁ − 1\| | 0.0347 | 0.0189 |
| κ₃ (fundamental band 600–1,200 nm), fit residual | 1.0523, 0.34% | 1.0039, 0.33% |
| λ = 4 ladder residual (limit 10⁻⁴) | 1.87×10⁻³ (exceeded) | 1.68×10⁻³ (exceeded) |
| leakage ‖c₁‖/‖c₃‖, seventh-order floor | 6.7×10⁻⁴, 2.5×10⁻⁶ (resolved) | 6.0×10⁻⁴, 2.1×10⁻⁶ (resolved) |
| leak at the weakest retained node / its remainder (limit 10⁻³) | 6.1×10⁻⁶ | 5.7×10⁻⁶ |

Both presets pass the gate with a wide margin; κ₃ is near one (κ₃ = γ_optmat/γ_haCC
in the fit, so haCC(1)'s Kerr response is about 5% below optmat's coefficient, as the
provider reports; haCC(5) within 0.4%), read beside the
provider's validated γ, not gated. The λ = 4 ladder residual exceeds its 10⁻⁴ flag for
both presets: as the Q25 note anticipated, that flags a first-order leak above about
4×10⁻⁵ of c₃, and the three-term fit resolves one at 6–7×10⁻⁴ of c₃ (about 1.5×10⁻⁸ of
the first-order current itself). The cluster CPU ladders of both presets reproduce it (κ₃,
residuals and leakage to about 10⁻⁶ relative), so it is the model's, not the device's. Scaled to the
weakest retained node of the selected rule (5.3×10⁹ V/m at the peak-power plane) the
leak is 6×10⁻⁶ of that node's nonlinear remainder, far below the 10⁻³ stop: negligible
for the source, though larger than report 0126's haCC(1) figure at its field. This is
the haCC(5) preset-scale figure Q15 left to L3. Costs: one 188,800-step row took
293 s (haCC(1)) and 404 s (haCC(5)) on the MI325X, against 527 s and 2,857 s for the
cluster gate including setup.

**L4 — nodes: pass; 6 nodes selected.** haCC(1), Gauss field-scale rules on the 60 fs
window at q = 8 (287,760 steps), all of 5, 6, 8 and 16 at both planes in two 36-row
calls (with the L5 rows); relative source difference against 16 nodes:

| Nodes | peak power, band | peak power, 90–200 nm | maximum UV, band | maximum UV, 90–200 nm |
|---|---|---|---|---|
| 5 | 0.45% | 0.82% | **1.06%** (fail) | 1.97% |
| 6 | 0.65% | 1.24% | 0.42% | 0.77% |
| 8 | 0.64% | 1.21% | 0.44% | 0.82% |

Tolerances 1% (band) and 2% (UV); every rounding floor is about 10⁻¹¹ of the reference
norm, so every comparison is resolved. **Selected: 6 nodes** (5 fails the band at the
maximum-UV plane), reported as agreement with the finite 16-node rule, not a certified
error (Q18). The Q9 extension did not run. The peak-power plane's on-axis row reached
8.03×10¹⁸ W/m² on the refined lattice, above the presets' 8×10¹⁸ W/m², and that call is
flagged `outside_preset_domain`, as section 5.6 prescribes.

**L5 — atomic window: fails; the ladder stops for review.** On-axis rows, the 60 fs
window against an explicit 100 fs window (479,640 steps), full-lattice gated source
spectra:

| Plane | band (tolerance 1%) | 90–200 nm (2%) | common window: band, 90–200 nm |
|---|---|---|---|
| peak power | **4.61%** (fail) | 0.40% (pass) | 0.036%, 0.040% |
| maximum UV | **1.43%** (fail) | 0.36% (pass) | 0.046%, 0.033% |

The common-window comparison (the 100 fs currents gated by the 60 fs gate) agrees to
0.05%, so the startup, the dropped early drive and the taper are not the cause: the
difference is emission after +30 fs that the 60 fs gate discards. The reported
decomposition puts 97% of the difference above 900 nm (at the peak-power plane 27% of
the 1.2–1.6 µm source and 7% of the 0.9–1.2 µm source, against 0.7–1.0% from 400 to
900 nm), and the reference's remainder beyond the 60 fs gate reaches 28% (peak power)
and 16% (maximum UV) of its largest value inside it: a slowly varying post-pulse
current (escaped-electron drift and recrossing is the likely origin; not diagnosed
further). The 90–200 nm source passes at both planes. Per the protocol nothing was
widened or re-run.

**Rental and resources.** One DigitalOcean `gpu-mi325x1-256gb` in nyc2 (AMD EPYC 9575F host, 20 CPUs, no cgroup quota, ROCm 7.14, gfx942), built
at the pin (HIP, Release, `PYRORAPTOR_THREADS=20`), created 15:40 and deleted
17:19 UTC: 5,947 s at US$3.80/h, **US$6.28**. Watchdog 210 min; the session ran
98 min (build 1 min, smoke 0.2, L2 14, L3 44, L4 22, L5 14) and the extension was
skipped because a count was selected. After deletion no rented machine of the session remained. Free cluster CPU (GCC 16, x86-64-v3,
OpenMP, 32 bound threads per job): build, the two witness references, the two κ₁
references and the two cross-check ladders, about 140 core-hours (allocated CPUs × elapsed). The scripts were
first run on an Apple M5 Max (Homebrew LLVM 23.1.3, OpenMP) on the provider's
tiny custom basis.

**Deviations.**

1. L2 ran in this phase's session (section 11 allows it) on a 6-node rule fixed in
   advance, since the L4 count was not known when it ran; it used a 4 fs window.
2. All of 5, 6 and 8 nodes also ran at the maximum-UV plane (the coordinating agent accepted it);
   the selection rule is section 7.2's.
3. The L5 candidate rows were part of the L4 calls (same descriptor).
4. The coordinating agent's request to make the λ = 2 and 4 rows one 2-row call was not adopted:
   `DiagnosticSession` evaluates one row per call, and a 2-row protocol call would
   have needed a descriptor on the protocol lattice outside solprop's API. The rows
   ran as two 1-row calls (no protocol change).
5. The device witness reused the cluster reference's saved fields (checked against its
   own synthesis to rounding), because NumPy's FFT need not round identically on
   different CPUs and a parity check needs identical inputs.
6. Device memory was sampled with `rocm-smi` between steps only, so no device peak was
   measured; the plan ledgers are the memory record.
7. The analysis repeated on an Apple M5 Max reproduces the device-host analysis: identical
   statuses and selection, relative differences equal to 2×10⁻¹⁶.

**Checks run.** `ruff check`, `ruff format --check`, `basedpyright` (benchmarks/sep20
and the new test: 0 errors), `reuse lint`, `codespell` on the changed files,
`git diff --check`, `uv run pytest tests/test_sep20_benchmarks.py tests/test_seps.py`,
and the strict documentation build. Not run: the full gate, slow, coverage and
Kokkos lanes (at the end of the SEP).

**Limitations.** L4 is agreement with 16 nodes for haCC(1) at two planes; haCC(5)'s
node error, the Radau rule and the reference's own error are unmeasured. κ₃ uses
optmat's single carrier coefficient. The L5 failure means the default 60 fs window
is not shown adequate for the propagated band above about 900 nm on these planes.

**Questions raised, resolved 2026-10-17.**

- **Q30 — L5 window (resolved by the maintainer).** The 60 fs window misses 1.4–4.6%
  of the band source, almost all above 900 nm, from post-pulse current after +30 fs;
  the 90–200 nm source passes. **Decision: keep the 60 fs default window and record
  an exception.** The exception is that the source above about 900 nm carries a known
  atomic-window error of up to about 5% on these planes. The window is a cost choice:
  the 102.5 fs propagation lattice could carry the late current, and the 60 fs window
  drops the near-infrared emission of the post-pulse current at the source. (An
  earlier wording said a 60 fs grid never carries that current; it does, so the
  wording was corrected at the Phase 4 review.) The rate model's plasma current has
  dJ/dt ∝ n_e E, so it is constant after the pulse and radiates nothing there; the
  dropped emission is physics the rate model lacks, not a matched truncation. Phase 4
  runs at 60 fs and reports this limit beside its near-infrared results; its SAE runs
  on the full lattice and on the 60 fs window differ by 0.32% over the band and 1.35%
  over 1.2–1.6 µm after propagation, with 0.07% over 90–200 nm.
  The alternatives were a 100 fs default (about 1.67× the steps per call), an atomic
  duration longer than the source gate, or a diagnosis with the provider first. Wider
  windows remain available through `TDSEAtomicWindow(duration=...)`.

  *Erratum (2026-10-19).* Q30's exception is restated by
  [SEP 26](sep-0026-edge-free-tdse-source.md), whose edge-free source is now
  haCC's default; see SEP 26's Phase 2 and Phase 3 reports and the haCC manual
  page for its measured size.
- **Q31 — ladder leak (resolved by the coordinating agent).** The λ = 4 ladder residual exceeds
  its 10⁻⁴ flag for both presets. A resolved first-order leak of 6–7×10⁻⁴ of c₃
  explains it, and it is negligible against the sources (6×10⁻⁶ of the weakest node's
  remainder). No change in solprop; the leak's size against report 0126 §B.3 is
  recorded as a provider follow-up in Pyroraptor's status.

**Handoff.** The seeded κ₁ entries in `evidence/kappa1-cache/` can be copied into
`$SOLPROP_CACHE_DIR/tdse-consistency/` on a Phase 4 machine. The selected rule is
`radial_rule="field_scale", n_r=6`. Raw arrays, archives and logs are in the
maintainer's local evidence directory and in the cluster campaign directory, hashed in
`evidence/manifest.json`.

### Phase 4 — run_short variant and the L6 segment (2026-10-17)

**Scope.** Phase 4 as section 11 defines it: the recorded projection, `run_short_hacc.py`
and `segment.py`, and L6 with the rate and SAE runs on free cluster CPU nodes and haCC(1)
then haCC(5) on one rented MI325X. Implementer: Opus 5.5 (AI) for John Travers, who approved up to US$60 for this phase and the
rental plan on 2026-10-17. Branch `feature/sep-0020-hacc-segment`, based on solprop
main `d5159b1` (Phases 1–3 merged; provider pin Pyroraptor `ea2bfcaa`). **Outcome:
every L6 mechanical and sensible-optics gate passes for every run;
haCC(1) and haCC(5) agree to 0.17% over the full band and 1.8% over 90–200 nm.**

**Delivered** (`benchmarks/sep20/`).

- `run_short_hacc.py`: the documented run_short variant of section 5.5.
  - It builds section 2's case from `case.py`; the measured pump is a path argument,
    and only its hash is recorded.
  - The response comes through the `tdse=` entry point: `hacc1`, `hacc5`, `sae` (an
    explicit record, by default SEP 12's selected research record), `rate` or `tiny`.
  - Numerics: the Gauss field-scale rule with 6 nodes, `rtol=1e-2`, interaction-picture
    Adams (the parent's integrator), every accepted step saved and the full raw archive.
  - `--start-plane INDEX --parent PARENT.h5` relaunches from a saved rate-parent plane.
    It keeps the full capillary's absolute fill, frame and mode normalisation, after
    rebuilding the parent's configuration from the pump and comparing it with the
    saved one.
  - `--resume` continues a capped run, and `--resume-history afresh` restarts its
    Adams history.
  - A provenance JSON record is written beside the output.
- `segment.py`:
  - `run` drives `run_short_hacc` for the L6 kinds, and `--interrupt-after N` gives the
    resume test.
  - `analyse` applies the L6 gates as fixed in `PROTOCOL.md` and reports every pair's
    complex spectral difference at 4.05 m.
  - `project` makes the haCC(5) admission decision on the rental.
- `segment_session.sh`: the scripted rental session (Phase 3's build and smoke steps,
  cache seed, haCC(1), projection, haCC(5), analysis).
- `PROTOCOL.md`, Phase 4: the gates, committed in `0298bf4` before any segment run and
  amended in `563867c` before any preset or SAE result (deviation 1).
- `parent.py` admits platform rounding in the rebuilt configuration (deviation 2).
- `tests/test_sep20_segment.py` covers the response selection, the band and difference
  metrics, the mechanical gates, the projection and its review stop.
  `tests/test_sep20_benchmarks.py` gains the rounding admission.
- Evidence: `evidence/l6-segment.json` (the analysis), `evidence/session-p4.json` (the
  rental, build and timeline) and the Phase 4 raw-file hashes in `evidence/manifest.json`.

**Inputs.**

- Launch: saved parent `parent.h5` (SHA-256 `079d37dc…`), plane 49 at 3.969 m, to
  4.05 m. Launch energy 1.3601 mJ, of which 3.854 µJ in 90–200 nm.
- Pump: `d9a512d3…`, private.
- haCC: the presets at q = 8 with the default 60 fs window.
- SAE: `selected-filtered.response.json` (scale 1.537, time collar, q = 32, cut 400 Eh,
  lmax 29), run on the full lattice and on the 60 fs window.
- Every run starts from the identical launch spectrum and ends at the final requested
  plane, 4.05 m, so all comparisons are at one common endpoint.

**Mechanical gates: all pass.**

- Every run completes at 4.05 m with finite fields at every saved plane.
- No TDSE run has a failed call or a binding failure record.
- Every TDSE ledger is consistent: attempted calls equal the logical startup plus
  predictor evaluations (9 for each uninterrupted run, 14 for the fresh-history
  resume), with `reserved = 0` and no diagnostic call.
- Both haCC runs passed the run-time κ₁ gate **from the cache**: from the L3 entries
  `52b11b66…` and `22158d21…` copied onto the rental, with no calls.
- Every uninterrupted TDSE run took 8 accepted steps at order 1 with no rejections. The step doubled
  from 0.41 mm to 25.9 mm, then a 29.6 mm landing. That is 9 provider calls per run, at
  the bottom of the 9–25 projection.
- Neither haCC run raised a domain flag. The segment's peak input intensity stayed
  below the presets' 8×10¹⁸ W/m², unlike the L4 refined-lattice row at 4.05 m.
- **Rate continuation:** 8 accepted steps and 9 logical calls; it differs from the
  parent's plane 50 by **0.21%** full band (gate 1%). The 90–200 nm difference is
  2.3%, reported only.
- **Checkpoint-resume.** Both variants use the full-lattice SAE run, stopped at a
  5-call cap at 3.975 m and resumed.
  - With the kept Adams history (solprop's default): every saved plane is bitwise
    unchanged, the identity is unchanged, there is no second startup, and the endpoint
    is **bitwise equal** to the uninterrupted run's (difference 0).
  - With a fresh history (section 7.2's literal case): a second startup evaluation,
    restarting Adams at order one. That took 14 calls and 12 accepted steps instead of
    9 and 8. The endpoint agrees with the uninterrupted run to **0.073%** full band
    (gate 1%) and 0.82% over 90–200 nm.

**Sensible-optics gates: all pass.** The rate column is the reference; SAE-full is the
full-lattice SAE run and SAE-60 the 60 fs-window one.

| | rate | SAE-full | SAE-60 | haCC(1) | haCC(5) |
|---|---|---|---|---|---|
| endpoint energy (mJ) | 1.3475 | 1.3406 | 1.3406 | 1.3400 | 1.3398 |
| energy / launch (gate ≤ 1.01) | 0.991 | 0.986 | 0.986 | 0.985 | 0.985 |
| energy / rate (gate within 5%) | 1 | 0.9949 | 0.9949 | 0.9945 | 0.9943 |
| 90–200 nm energy (µJ) | 3.936 | 5.005 | 5.005 | 4.880 | 4.963 |
| 90–200 nm / rate (gate 0.5–2) | 1 | 1.271 | 1.271 | 1.240 | 1.261 |
| 90–200 nm / SAE-full, SAE-60 (gate 0.5–2) | | | | 0.975, 0.975 | 0.992, 0.992 |
| peak power (TW) | 0.588 | 0.592 | 0.592 | 0.591 | 0.591 |
| FWHM, RMS duration (fs) | 1.333, 4.307 | 1.333, 4.318 | 1.333, 4.319 | 1.337, 4.321 | 1.335, 4.321 |

**Reported, not gated: complex spectral differences at 4.05 m.** Each entry is
‖a − b‖/‖b‖ for the pair "a against b", with no phase or time alignment.

| a against b | full band | 90–200 nm | 200–400 | 400–600 | 600–700 | 700–900 | 0.9–1.2 µm | 1.2–1.6 µm |
|---|---|---|---|---|---|---|---|---|
| **haCC(1) against haCC(5)** | **0.17%** | **1.79%** | 0.28% | 0.07% | 0.21% | 0.11% | 0.06% | 0.16% |
| SAE-full against haCC(1) | 0.50% | 2.82% | 0.55% | 0.33% | 0.32% | 0.17% | 0.25% | 1.55% |
| SAE-full against haCC(5) | 0.47% | 1.12% | 0.30% | 0.29% | 0.47% | 0.23% | 0.29% | 1.64% |
| SAE-60 against haCC(1) | 0.37% | 2.82% | 0.55% | 0.33% | 0.31% | 0.18% | 0.19% | 0.55% |
| SAE-60 against haCC(5) | 0.33% | 1.12% | 0.30% | 0.29% | 0.46% | 0.24% | 0.24% | 0.68% |
| SAE-full against SAE-60 | 0.32% | 0.07% | 0.01% | 0.02% | 0.04% | 0.03% | 0.15% | 1.35% |
| rate against haCC(1) | 2.56% | 29.1% | 1.85% | 0.98% | 3.56% | 1.87% | 1.07% | 4.49% |
| rate against haCC(5) | 2.71% | 30.6% | 2.12% | 1.00% | 3.77% | 1.98% | 1.13% | 4.60% |
| rate against SAE-full | 2.64% | 31.3% | 2.37% | 0.85% | 3.34% | 1.80% | 0.92% | 3.54% |

**Reading the comparisons.**

- **The extra ionic channels.** Over this 81 mm, haCC(5)'s n ≤ 2 channels change the
  propagated field by 0.17%. They change the 90–200 nm field by 1.8%, and its energy by
  +1.7% (4.963 against 4.880 µJ).
- **Correlation and the fixed scale.** haCC lies within 0.3–0.5% of the scaled SAE run
  over the full band. Over 90–200 nm it differs by 2.8% (haCC(1)) and 1.1% (haCC(5)),
  and haCC gives 0.8–2.5% less UV energy.
- **The atomic window.** The two SAE runs differ by 0.32% over the full band and 0.07%
  over 90–200 nm. The difference sits almost entirely above 900 nm (1.35% in
  1.2–1.6 µm): the near-infrared atomic-window limit of Q30, now seen in a propagated
  field. The haCC runs carry the same 60 fs window, so their results above about 900 nm
  stand beside Q30's recorded exception (a source error of up to about 5% there). Their
  0.06–0.16% mutual difference above 900 nm does not measure that error.
- **The rate model** differs from every atomic model by about 2.6% over the full band
  and 29–31% over 90–200 nm. The atomic models give 24–27% more UV energy.
- **Limits on all of these.** They are model differences over one short segment, not
  errors against an exact solution. Read them beside:
  - the presets' 1.22% total-current sensitivity (not an error bar on these
    differences);
  - the L4 agreement with 16 nodes for haCC(1) only (haCC(5)'s node error is
    unmeasured);
  - L5's window result and Q30;
  - the unmeasured ionic-continuum truncation;
  - the SAE record's own limitations.
- **Neglected-current bound** (diagnostic only), maximum over all calls of the bound
  over the row's peak current:
  - haCC(1): 0.15 (median 1.0×10⁻³);
  - haCC(5): 1.18 (median 1.0×10⁻²);
  - SAE: 0.063 and 0.034.

**Resources.**

| Run | Host | Wall | Per call | Peak RSS | Binding ledger | Archive |
|---|---|---|---|---|---|---|
| haCC(1) | MI325X, 20 host threads | 3,366 s | 366 s | 2.24 GB | 231 MB (plans 114) | 53.6 MB |
| haCC(5) | MI325X | 11,958 s | 1,307 s | 4.52 GB | 1,197 MB (plans 879) | 53.6 MB |
| SAE-full | cluster, 32 bound threads | 10,486 s (93 core-h) | about 1,170 s | 1.14 GB | 648 MB | 599 MB |
| SAE-60 | cluster | 8,479 s (75 core-h) | about 930 s | 0.88 GB | 456 MB | 363 MB |
| SAE resume, kept | cluster | 5,525 + 8,328 s (123 core-h) | | 1.15 GB | | 599 MB |
| SAE resume, fresh | cluster | 7,947 + 14,511 s (200 core-h) | about 1,600 s | 1.15 GB | | 932 MB |
| rate | cluster, 4 CPUs | 10 s | | 0.44 GB | | |

- **Measured against projected.** Per-call times match the projection from Phase 3:
  366 s against 365.8 s for haCC(1), and 1,307 s against 1,296 s for haCC(5).
- **Device memory.** The peak sampled device memory (rocm-smi, every 30 s) was
  2.44 GB.
- **Cluster CPU.** The four SAE runs cost about 490 core-hours (allocated CPUs ×
  elapsed). Builds, the tiny-basis plumbing runs, the rate run and the cache check add
  about 25 core-hours. The fresh-history resume ran on a busier node at about 1,600 s
  per call.
- **Projection, made on the rental.** From haCC(1)'s measured 9 calls, haCC(5) was
  projected at 12,115 s (US$12.79). With US$3.87 spent and the 1.2 margin, the need was
  US$20.17, within the US$55 session budget, so haCC(5) was admitted with a 35-call cap.

**Rental.**

- One DigitalOcean `gpu-mi325x1-256gb` in nyc2.
  - Host: AMD EPYC 9575F, 20 CPUs, ROCm 7.14, gfx942.
  - Built at the pin by Phase 3's recipe: HIP, Release, `PYRORAPTOR_THREADS=20`.
    Phase 3 ran L2 on the same pin, recipe and SKU so L2 was
    not rerun; this phase ran the smoke witness only.
- Times (UTC): created 18:58:58, haCC(1) 19:03:53–20:00:01, haCC(5) 20:00:02–23:19:22,
  deleted 23:20:13 (HTTP 204).
- Duration and spend: 15,676 s at US$3.80/h, **US$16.55**. The SEP 20 total is
  US$22.82 of the US$100 cap.
- Watchdog: 880 min, deleting only the recorded ID; the session never reached it.
- After deletion the droplet's ID returns 404, and no rented machine of the session remains.
- The private pump was staged with mode 600, deleted on the droplet before the pull,
  and appears in no pulled file.

**Deviations.**

1. **Resume.** Section 7.2 expects a resume to charge a fresh startup; since ADR-046
   solprop continues the saved Adams history instead. Both variants ran (protocol
   amendment `563867c`, accepted by the coordinating agent): the kept history (solprop's default,
   bitwise) and the fresh history (the literal case).
2. **Parent rebuild check.** It was relaxed after an exact-match failure. The parent was
   written on Apple Silicon; rebuilding its configuration on x86 cluster differs at
   rounding in regenerated arrays (the pump FFT, fitted plasma-integration taps and
   their certified error, field ranges, the mode area). The largest relative
   difference is 2.25×10⁻¹¹, over 314 numbers. The launch check now admits regenerated
   floating-point numbers within 1×10⁻⁸ relative and records the count and the largest.
   This is a provenance check, not a scientific tolerance; strings, integers, keys and
   shapes stay exact. The rebuild on an Apple M5 Max is exact.
3. **GPU false start.** The first haCC(1) attempt failed at import, because
   `segment_session.sh` did not source the device runtime path. It was fixed in
   `94707de` and restarted at once; the GPU idled for about 1 minute.
4. **SAE record.**
   - It was qualified with 33 Gauss–Kronrod nodes and a 120 nm–1.6 µm band. Here it runs
     on the 6-node field-scale rule (section 6.7's SAE control) on run_short's 90 nm grid.
   - Its calibration band is recorded, not enforced, so the 90–120 nm source lies
     outside it.
   - It keeps q = 32, the record's sampling, rather than haCC's q = 8.
5. **Rate continuation.** It keeps the parent's 33-node rule and 51-plane schedule, so
   that it reproduces the parent; the TDSE runs save every accepted step.
6. **Projection inputs.** The projection used Phase 3's MI325X costs measured through
   solprop (1.27 and 4.5 ms per step at 6 rows), not Pyroraptor's haCC timing study's table; they agree.
7. **The pump on the rental.** `run_short_hacc` rebuilds the case from the pump, so the
   pump was staged on the rental, as the coordinating agent accepted.

**Checks run.**

- `ruff check`, `ruff format --check`, `basedpyright` (the Phase 4 scripts and tests: 0
  errors), `reuse lint`, `codespell` on the changed files and `git diff --check`.
- `uv run pytest tests/test_sep20_segment.py tests/test_sep20_benchmarks.py
  tests/test_seps.py`.
- The strict documentation build.
- Not run: the full gate and the slow, coverage and Kokkos lanes (at the end of the
  SEP).

**Limitations.**

- One 81 mm segment near compression at `rtol=1e-2`; no full-length run and no
  optical-tolerance study.
- The haCC(5) node count, the ionic continuum and the 16-node reference's own error are
  unmeasured.
- The 60 fs window's near-infrared limit (Q30) applies to every windowed run.
- The comparisons measure model differences, not accuracy.

**Handoff.**

- haCC(5) ran in full, so no section 11 validation exception is needed for it.
- The SEP's status change to Final, the final qualification lanes and the open
  disposition of every review question are left to the coordinating agent.
- Raw outputs, archives, spectra and logs are in the maintainer's local evidence
  directory and the cluster campaign directory, hashed in `evidence/manifest.json`.

## Part III — History

- 2026-10-16: Drafted, with the high-level design and the maintainer's direction
  that haCC(1) and haCC(5) are treated equally.
- 2026-10-16: Revised after the maintainer's decisions. Full runs are removed and
  validation ends with the segment. Q4, Q5, Q6, Q8, Q9, Q11 and Q12 are resolved:
  an automatic backend, a cost warning, a domain flag, a cached run-time weak-field
  consistency check with a 10% threshold and an override, a node ladder extended to
  16 when needed, and the time collar as haCC's default. The provider contract is
  aligned with the provider's merged haCC consumer surface, with its pending items
  marked.
- 2026-10-17: Revised to the provider's implemented contract. Q1, Q2, Q3 and Q7 are
  resolved: the model comes from the qualified presets, the qualification flag is
  the gate, and the neglected-current bound is a diagnostic only. Costs now use the
  measured per-call figures.
- 2026-10-17: Revised after the maintainer's decisions of the same day. Q10: the haCC
  segments run on a rented H200 or MI300-class GPU within a US$100 cap, starting
  with a per-call timing; studies use free resources. Q13: the atom is driven only
  over the requested 60 fs, with a 5 fs taper. Q14: the node study is a minimal haCC(1) Gauss set, with none for
  haCC(5). Custom haCC bases are admitted, unqualified and warned. Provider argument
  errors are interface errors. The sampling study is dropped in favour of the
  presets' $q=32$.

- 2026-10-17: Reviewed by GPT-6 (AI) and John Travers (human), with the human review
  recorded at the maintainer's explicit request. Lifecycle status remains Draft. Review
  corrections distinguish finite-tail shadow leakage, channel sensitivity and numerical
  ladders from exactness or certified accuracy; clarify the tapered drive, sample
  bounds, interpolation cost, diagnostic validity, cache portability, resource
  accounting and schema changes; replace the bitwise Adams-resume claim; and make the
  source-window comparison include discarded emission. Q15–Q19 assign the remaining evidence requirements to owners and phases;
  Q20–Q21 record the maintainer's resolved duration and accounting decisions. No implementation or atomic
  qualification was performed. Sources checked: the haCC and mixed-gauge papers [1, 2],
  irECS [3], primary helium polarizability [8], optmat's process-dependent
  nonlinear-response conventions, solprop's grid/seam/resume code and documentation, and
  the provider's consumer/model documentation at revision `b8fc0c16`. Provider numerical
  ladders, performance measurements, the full book [7] and the original measured pulse
  were not independently reproduced; Q19 preserves that evidence limitation. Verification: `git diff --check`, codespell on the changed files,
  `uv run pytest tests/test_seps.py` (162 passed, one expected process-SEP skip),
  and `uv run poe docs` passed. The full code, slow and coverage lanes are not
  required for this documentation-only review.
- 2026-10-17: The maintainer approved explicit atomic-window duration for
  validation and research, defaulting to the capillary request, and conservative
  charging of every attempted provider evaluation. Q20 and Q21 are resolved;
  interface, refusal and verification requirements are reconciled with both decisions.

- 2026-10-17: At the maintainer's request, added every accepted optical step as the
  new TDSE convenience output default, using SEP 14's existing `step_planes`
  mechanism, and an optional full raw-response HDF5 sidecar (§5.8, Q22–Q23).
  Reconciled the example, provider full/compact adapters, output provenance,
  recovery/accounting, tests, qualification costs and documentation. Review status
  remains Reviewed; lifecycle remains Draft and no runtime code is changed.

- 2026-10-17: Re-reviewed by the coordinating agent (Opus 5.5) against the provider code
  and reports at revision `b8fc0c16` (RK4 subdivision, cubic stage interpolation,
  closure surface, preset sensitivities and identities, full/compact cost) and
  solprop's grid, schema and save defaults. Maintainer decisions recorded: the run-time
  consistency gate uses κ₁ only, with κ₃ measured once per preset over the fundamental
  band as evidence (Q8 amended); per-call GPU timing comes from Pyroraptor's haCC timing study
  and L2 becomes a short witness at the start of the first segment rental (Q10
  amended); Q15 resolved by the provider's derivation, measurement and test, with the
  haCC(5) preset figure from L3; Q17 resolved by the provider's new
  `HeliumPreset.coverage` from Pyroraptor's haCC timing study Part D; the raw archive becomes its own
  phase, and the phases are renumbered. Editorial corrections to the abstract, §4.2,
  §5.4 and the resource wording. Review status remains Reviewed; lifecycle remains
  Draft.
- 2026-10-17: At the maintainer's decision, the presets' grid-relative `coverage`
  interval is withdrawn. The domain flags are now duration (against `window`), peak
  intensity and carrier only, with arbitrary grids, windows and tapers admitted and
  never refused (§4.6, §5.6, §10). Pyroraptor's haCC timing study Part D updates `window` to the ±30 fs
  re-checked duration. Q16 becomes a Phase 1 implementation obligation and Q17 is
  resolved; only Q18 and Q19 remain open. Lifecycle remains Draft.
- 2026-10-17: Revised after the coordinating agent's high-level review, at the maintainer's
  decisions. The consistency check warns above 10%, refuses above 30% and can be
  skipped, with a cache key that excludes the build (Q8 amended). The raw archive is
  append-only, without charge accounting or transaction markers (Q23 amended). haCC
  samples at $q=8$, about 6% fewer atomic steps than $q=32$. Checkpoint-resume is
  validated on CPU with SAE and by the fake-provider test, not on a rented haCC
  segment. Q18 (a 1% energy allowance, SEP 12's restart agreement) and Q19 (hashes and
  regeneration instructions, no redistributable inputs) are resolved; no question
  remains open. Custom helium models without the escaped-electron closure and explicit
  full-lattice haCC runs are admitted with a warning or as a research choice. Rented
  GPUs may also run compute-heavy one-off atomic checks in a scripted session where
  Pyroraptor's haCC timing study's costs show a large gain (Q10 amended). Review status remains Reviewed;
  lifecycle remains Draft.
- 2026-10-17: Accepted by the maintainer (John Travers). Part I is frozen; deviations
  found during implementation go in the phase reports.
- 2026-10-17: Picked up for implementation (status In progress): Phase 1 by Opus 5.5
  (AI) on `feature/sep-0020-hacc-binding`.
- 2026-10-17: Phase 1 revised after the coordinating agent's review: explicit cache validation, a
  10⁻⁴ L3 ladder limit with the leakage floor, deferred haCC plans,
  the cost warning asserted in the SEP 12 modules, and the closure message pinned
  against the real provider.
- 2026-10-17: Phase 1 implemented on `feature/sep-0020-hacc-binding` and reported in
  Part II (records, seam, binding, window, consistency check, entry point and
  natural-step output; provider pin `272ef730`); awaiting the coordinating agent's review.
- 2026-10-17: The maintainer confirmed the L3 κ₃ extraction values (Q25) and the
  coordinated solprop-kokkos schema 27 change (Q24); Q27 is fixed on main.
- 2026-10-17: Phase 2 implemented on `feature/sep-0020-tdse-archive` and reported in
  Part II (the raw TDSE response archive `solprop.tdse-archive.v1`, trial roles,
  sessions and the optical reference; kernel schema 27 and output schema 18
  unchanged); awaiting the coordinating agent's review. Q28 and Q29 opened.
- 2026-10-17: Phase 2 reviewed by the coordinating agent and merged; Q29 resolved (no kernel
  schema bump). Q28 moves to Phase 4's L2 witness.
- 2026-10-17: Provider pin moved to Pyroraptor `ea2bfcaa` (Pyroraptor's haCC timing study); Q26 resolved.
- 2026-10-17: Phase 3 implemented on `feature/sep-0020-hacc-witness` and reported in
  Part II: L1 export from the saved rate parent; on one rented MI325X (US$6.28) the L2
  witness, L3 (|κ₁ − 1| 0.035 and 0.019), L4 (6 nodes selected) and L5, which fails its
  band tolerance from post-pulse current beyond +30 fs; the ladder stops for review.
  Q30 and Q31 opened.
- 2026-10-17: Phase 3 reviewed by the coordinating agent and merged. Q30 resolved by the
  maintainer: the 60 fs window stays the default, with a recorded exception for the
  source above about 900 nm. Q31 resolved by the coordinating agent as negligible, with a provider
  follow-up.
- 2026-10-17: Amended by [SEP 25](sep-0025-rounding-level-reproducibility.md), which
  removes bitwise requirements between independent floating-point computations; Part I
  and Part II are unchanged and are read with this note. Restated as rounding bounds
  fixed before measurement, from operation counts with a margin: C8's bitwise identical
  atomic outputs for identical inputs on one build and thread count (§5.7), and so the
  "bitwise repeat" of the L2 witness (§6.7 item 2, §7.2 L2); §7.1 item 6's gated source
  of a causal fake against a full-lattice run; §7.1 item 14's haCC `atomic_sources`
  against the SAE `tdse_native` path, unless both reach the same assembly code on the
  same arrays; and §7.1's preamble, whose exact tolerance for "a bitwise-defined
  formula" now covers only a copy or the same code path on the same data. Restated as
  accuracy envelopes against a tighter or stored reference: the L6 checkpoint-resume
  agreement (§7.2), already within SEP 12's interior-restart agreement and not bitwise.
  Kept exact as structural contracts: the archive's round trip read back exactly and
  archiving leaving the optical march and both ledgers unchanged (§7.1 item 21); the
  compact (required) and full outputs' consumed currents, shadows and remainders where
  the compact record copies the same arrays the full one holds, as the provider states
  for the haCC compact total current (§7.1 item 21, §7.2 L2, §7.3, Part II phase 2),
  with a rounding bound where a selection computes them separately; the resumed run's
  saved field and identities (§7.2 L6); and the base-commit identity, record and
  resume-key literals (Part II phase 1). The runs with and without archives in Part II
  phase 2 keep exact ledgers, plane positions and provider inputs on a prescribed
  schedule; their saved fields are compared within a rounding bound when re-verified.
  §4.6's statement that causality is not a bitwise provider-parity guarantee stands.
- 2026-10-17: Phase 4 implemented on `feature/sep-0020-hacc-segment` and reported in
  Part II: `run_short_hacc.py`, `segment.py` and the scripted session; L6 from 3.969
  to 4.05 m with the rate, SAE (full lattice and 60 fs window), haCC(1) and haCC(5)
  runs, every mechanical and sensible-optics gate passing; haCC(1) against haCC(5)
  0.17% over the band and 1.8% over 90–200 nm; one MI325X, US$16.55. The resume
  check runs with kept and fresh Adams history (ADR-046), and the parent rebuild
  admits platform rounding; both recorded as deviations. Awaiting the coordinating agent's review.
- 2026-10-18: Phase 4 reviewed by the coordinating agent and merged; deviations 1–7 accepted. Q30's
  rationale corrected: the 60 fs window is a cost choice that drops post-pulse
  near-infrared emission, which the 102.5 fs lattice could carry and the rate model
  does not produce. Pyroraptor's late-current study measures the late current at 100 fs as an
  addendum.
- 2026-10-18: Final, approved by the maintainer. The end-of-SEP gate on `ddf7921`
  passed the slow and Kokkos-selected slow lanes, and every SEP 20 test passed in every
  lane; the remaining failures of `poe check`, `poe cov` and the Kokkos default lane
  were SEP 25's; after SEP 25's Final, `0674f84` passes the schema, doctest, docstring,
  spelling and documentation checks that had failed (the coverage lane was not
  rerun here). TDSE responses stay on `numpy_ref`
  (SEP 12's recorded exception). Moved to the roadmap: full-length runs; the provider
  cost reductions (Pyroraptor's linear-response kernel, the faster preset, CPU
  placement) and their pin bump; haCC(5)'s node count; the 60 fs window's
  near-infrared limit (Q30) with Pyroraptor's 100 fs box addendum.
