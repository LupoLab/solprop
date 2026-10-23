# SEP 3 — Materials, Raman responses and ionisation rates from optmat

| | |
|---|---|
| **SEP** | 3 |
| **Title** | Materials, Raman responses and ionisation rates from optmat |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; Opus 5.5 (2026-09-29 revision and split; 2026-10-01 revision against optmat 0.1.0) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-09-27 |
| **Resolution** | 2026-10-01: accepted by John Travers after the human and AI reviews and the re-review against optmat 0.1.0; merged directly to `main`. 2026-10-04: Final, approved by John Travers after the end-of-SEP lanes passed on `main` |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI), for John Travers: Phases 0 to 4c and the close-out on `feature/sep-0003-optmat-materials-*` branches, reported in Part II |
| **Manual pages** | [Materials](../../../guide/media/materials.md), What changed from solprop's old material data (recorded in the development archive), [Strong-field ionisation](../../../guide/media/ionisation.md#rates-from-optmat), [The Raman response](../../../guide/media/raman-data.md#raman-responses-from-optmat), [Output](../../../guide/results/provenance.md#the-material-manifest); the tutorial *Compare material choices* (`docs/tutorials/basics/plot_06_material_choices.py`) |

## Part I — Proposal

### 1. Abstract

Make optmat a required dependency and the sole source of solprop's built-in
material data:

- the gases, with their dispersion, carrier-evaluated electronic nonlinearity,
  equation of state and transport properties;
- the glasses, crystals and metals that solprop ships;
- the gas and fused-silica Raman responses;
- the strong-field ionisation rates of the gases.

solprop keeps the propagation equations, geometry, tabulation at translation,
plasma evolution, response assembly and numerical execution. It deletes three
things:

- its material tables and coefficient sets;
- its own ionisation-rate formulas;
- its built-in Raman parameter sets.

This SEP is deliberately a **data migration with the kernel unchanged**. Every
optmat quantity enters through a representation the kernel contract already has:
index tables, rate tables, and Raman rows with decay-rate paths. Two physics
upgrades that the migration makes possible have their own proposals:

- a low-rank Raman kernel representation for density gradients
  ([SEP 10](../sep-0010-low-rank-raman.md));
- the finite-density Lorentz–Lorenz gas law with mixture optics
  ([SEP 11](../sep-0011-finite-density-gas-index.md)).

Larger optmat design work is done and reviewed upstream before implementation
begins (§11, Phase 0).

This is a change of physical models and data, not a promise of agreement with old
simulations. Qualification keeps three questions apart: adapter correctness,
numerical accuracy, and physical-model change. Material identities enter output and
resume checks. Old environments reproduce old results, and no legacy material
database remains in solprop.

### 2. Motivation and scope

#### Status

**Accepted; implementation not started.** The proposal first audited:

- solprop at `2dc84d005be17196c25410c5da0de9d33ca8e50a`;
- optmat at `d57963b0cdae8f5d38e9058d85551d2b4eca0442`;
- solprop-kokkos at `87ac816190ffb0c3978de7e236b6e30d7c068ab8`.

The GNLSE Raman-fraction rule (§4.5, D4) was re-checked against solprop main at
`62cfd26`, where it changed. On 2026-09-30, D8, D9 and D11 and the sections they touch
were revised against optmat at `ea0dcaf` and its review of OEP 1 [30].

**The upstream preparation is complete.** optmat's OEP 1, the propagation-code interface
[30], and OEP 4, the mixture index and density [31], are Final. Both are in optmat
0.1.0, tag `v0.1.0` at commit `d6945ed8e34fdff848a860a64d9b8597f9471774`, which closes
G1–G10 of §3.2. On 2026-10-01 this proposal was re-reviewed against that tag's public
surface [32], with read-only evaluations of its models, and revised to match it (§3.1,
§3.2, §4.4–§4.6, §5, §11). The implementation pins that tag (§6).

optmat changed after the first audit in ways §3.1 and §9.1 show only in part:

- the O2 and CH4 rate defaults;
- helium's retardation term;
- CO2's bands;
- the molecules' reference temperature;
- the molecular fits' qualified infrared windows (§3.1).

The solprop side of Phase 0 re-measures §9.1 at the pinned tag and reports it in Part II.

The evaluations in §9 were computed at these revisions by calling both packages'
public functions; no propagation was run. Neither those evaluations nor checks of
this document qualify the proposed implementation. Part II remains empty until
implementation phases merge.

Take a gas-filled capillary with a silica wall, or a bulk window, as the use case.
Its dispersion, nonlinear response, Raman kernel, plasma rate and wall index
currently come from separate solprop tables, each with its own calibration,
approximations and reference state. The audit found errors in those tables that
optmat corrects with evidence:

- window-crystal Sellmeier formulas measured relative to air are used as absolute
  indices, $4\times10^{-4}$ too low;
- the default silica index passes through a formula pole near 116 nm;
- an ionisation occupancy of two doubles an already correct helium rate;
- PPT Stark and dipole factors have no traceable source;
- the neon and argon Kerr coefficients are about 29 % and 19 % below the curated fits.

Keeping two copies of material data is itself the duplication this SEP removes.
optmat, meanwhile, has become a validated source for exactly these quantities:

- absolute solid indices with their primary provenance;
- per-line Raman responses checked against measured gains and revivals;
- rate models checked against all-electron static rates, a time-dependent
  calculation and measured yields.

#### Included

- **Gases.** Every existing pure gas: He, Ne, Ar, Kr, Xe, H2, D2, N2, O2, CH4, SF6
  and N2O. Also the dispersion-only `Gas.Air` interface, the explicitly resolved air
  fills, and supplied species and labelled mixtures, keeping their existing
  contracts. For each gas:
  - linear dispersion and available absorption, at the existing linear density
    scaling (§4.2);
  - number density and its inverse;
  - the transport inputs to gas flow;
  - the electronic Kerr coefficient at the carrier.
- **Gas Raman responses.** optmat's per-line responses of N2, O2, H2, D2, CH4 and
  SF6, and of dry air as a named CIPM-2007 composition [29], on the existing Raman
  seam (§4.4). A mixture's response is built only from the collision partners optmat
  has curated. A mixture with any other partner, such as H2 or N2 in helium, is
  refused (D16).
- **Gas ionisation.** optmat's rate models for every existing gas, and the energies
  used by plasma energy loss.
- **Solids.** Fused silica and every other solid solprop ships: CaF2, MgF2 (both
  rays), BaF2, sapphire (both rays), N-BK7, silver and aluminium. Covered wherever
  they are used: capillary, antiresonant, rectangular, slab and step-index walls and
  cores; general modal material regions; free space; bulk optics, prisms and
  compressors; design tools.
- **Silica nonlinear response.** The silica Kerr coefficient, and its scalar and
  tensor Raman responses as a paired bundle of $n_2$, fraction and shape.
- **Cross-cutting.** Provenance and resume identity, dependency packaging,
  documented limitations, qualification of changed results, and deletion of the
  migrated database and of solprop's rate formulas.

"All gases" means complete migration of the existing supported capabilities, **not**
an invented Raman or ionisation model for every name in a registry. optmat also
knows CO, CO2 and O3; they are not added here (D13). A missing
response is distinct from a physically absent one. An atom has no molecular Raman
response, but a molecule without curated Raman data is not assigned a zero response.

#### Settled decisions

These were settled on 2026-09-29:

- **D1 — solprop keeps no ionisation-rate formulas.** Every rate for a built-in or
  custom species comes from an optmat rate model. User-supplied rate objects and
  tables remain supported.
- **D2 — Raman responses use the existing seam.** Each damping group and tensor
  family becomes one existing Raman term: an undamped row and a decay-rate path. The
  low-rank representation, which also carries H2's density-dependent line shift and
  removes the rebuild cost on gradients, is
  [SEP 10](../sep-0010-low-rank-raman.md).
- **D3 — Chirped-mirror curves are device data.** They stay in solprop, outside this
  SEP, with a roadmap item for their provenance.
- **D4 — The silica scalar Raman default is optmat's paired response.** That is
  Hollenbeck–Cantrell's shape with the fraction $f_R=0.219$ of optmat's silica Kerr
  fit. It replaces only the inline 0.18 in solprop's GNLSE fraction rule
  ([GNLSE](../../../guide/waveguides/gnlse.md), "The Raman fraction is not silently taken"), and
  leaves the rule itself unchanged:
  - `raman_fraction=None` with Raman on resolves to the bundle's $f_R$;
  - `raman_fraction=None` with Raman off still resolves to 0, so the whole of
    $\gamma$ goes to Kerr;
  - an explicit `raman_fraction` is always held back from Kerr, with Raman on or
    off, and overrides the bundle;
  - a user-supplied $n_2$ or $\gamma$ sets the total and takes the bundle's
    fraction unless one is given.
- **D5 — Linear density scaling stays for now.** Gas susceptibility remains
  $\chi=N\alpha(\omega)/\varepsilon_0$, with optmat's polarizability and a declared,
  measured error (§4.2). The finite-density law and mixture optics are
  [SEP 11](../sep-0011-finite-density-gas-index.md).
- **D6 — optmat design first.** Public contracts, identities, explicit-parameter
  construction and other larger optmat design decisions are reviewed and implemented
  upstream before Phase 1 (§11, Phase 0). The implementer may still fix small gaps
  found during implementation in optmat, through optmat's own review process.
- **D7 — Quick experiments stay one line.** Every material quantity keeps a
  documented bring-your-own path in solprop, and every optmat model the adapters use
  accepts explicit parameter overrides (§5). Trying a modified width, occupancy or
  Sellmeier coefficient must not require a change to optmat.
- **D8 — Plasma energy per event is the rate's own threshold.** The loss per
  ionisation is the rate-weighted energy of the channels the rate model sums. For the
  molecular defaults, that is the vertical energy the rate uses. The same threshold
  therefore enters the rate's exponent and the loss. For the noble gases the
  rate-weighted energy moves with the field between the two fine-structure channels,
  and solprop's plasma kernel carries one energy per population. The loss therefore
  uses a constant energy per species, the rate-weighted energy at the run's peak field,
  as a named approximation (`energy-loss="peak-weighted"`). Its bound, the largest
  departure of the rate-weighted energy from that constant over the fields the run
  tabulates, is recorded (§4.6). The kernel is unchanged. Decided 2026-09-30.
- **D9 — `Gas.Air` is the named composition's summed polarizabilities.** It is built
  from optmat's dry-air composition `"dry air (N2, O2, Ar)"`, not a separate
  refractometry formula, so it is consistent with the N2/O2/Ar populations solprop
  propagates and with optmat's dry-air Raman response. The standard air formula is a
  validation check on it. Its number density is that of optmat's air model, CoolProp's
  pseudo-pure dry air, which includes the unlike-pair interactions (decided 2026-09-30,
  §4.2).
- **D10 — The weak rate defaults stay defaults, flagged.** Kr/Xe's suggested
  barrier-suppression constant and SF6's molecular prediction remain the recommended
  rates, carrying their uncertainty factors. SF6 is flagged `untested` and solprop
  warns once per run.
- **D11 — The silica tensor default is Lin–Agrawal's own response, holding the
  fitted total $n_2$.** The paired variant first proposed, Lin–Agrawal's shapes scaled
  to the fitted $f_R=0.219$, was checked against the measured absolute gain [17] at
  330 cm⁻¹: it is 17.8 % below it, 1.8 times the measurement's uncertainty, where the
  scalar default is 9.7 % below. The fallback this decision provided for therefore
  applies (2026-09-30; optmat's register [8]). The tensor default is `lin_2006` with its own
  $f_R=0.245$, 1.0 standard deviation from the fitted value, and `kerr/SiO2`'s total
  $n_2$; its gain is 7.9 % below the measurement. Scalar and vector runs share the
  total $n_2$, so they agree for long pulses. They do not share the split: the vector
  run's electronic part is 3.3 % smaller and its Raman part 12 % larger. The paired
  variant is not added.
- **D12 — Rates are presented honestly, and non-linear polarisation stays admitted.**
  - The ionisation manual shows both absolute-yield comparisons as optmat reports
    them: the default matches the 5.5 fs calibration-standard yields within their
    calibration, and is 1.7–8 times above the 42 fs yields [26, 27].
  - It names that disagreement, which is unresolved upstream, and neither rescales
    the rate nor picks a winner; the uncertainty factor is the summary.
  - Elliptical and circular runs remain admitted, as today, on the quasi-static
    average of the instantaneous rate around the polarisation ellipse. The manifest
    records the approximation.
  - A run warns once when it is not linearly polarised and its Keldysh parameter at
    peak field exceeds 1, where elliptic and orbital-resolved theories depart from
    that average.
- **D13 — N2O's provisional Raman is opt-in; no new gases are added here.**
  - N2O Raman stays refused by default. An explicit `raman="provisional"` admits
    optmat's provisional vibrational bands, with a one-time warning, and records it.
  - CO, CO2 and O3 are not added by this SEP. CO2 is the first follow-up candidate
    for a separately qualified built-in gas; CO follows. O3 waits for proper
    handling of absorbing gas indices and non-ideal density.
- **D14 — Follow-up items are created at acceptance.** The solprop roadmap gains the
  missing items when this SEP is accepted, not in Phase 4, and Phase 4 confirms they
  are current. optmat's own scientific gaps stay on optmat's roadmap and are linked,
  not duplicated. These include Q-branch line mixing, N2O rotations, modern H2/D2
  levels and the H2/D2 gain excess.
- **D15 — D1 extends to solids.** optmat owns the solid photoionisation rate models
  (Keldysh family and multiphoton), with their normalisation, averaging declaration
  and validation, and the carrier and avalanche coefficients as data. solprop keeps
  carrier kinetics, the avalanche closure, collisional currents, tabulation and
  propagation. [SEP 5](../sep-0005-solid-ionisation.md) is revised accordingly and
  needs its own upstream optmat work before acceptance.

These were settled on 2026-10-01, in the re-review against optmat 0.1.0:

- **D16 — Raman mixtures need curated partners.** optmat builds a mixture's response
  only from the broadening coefficients it has curated for each pair:
  - N2 and O2 with each other and with Ar;
  - SF6 with any atom, because the width of its band does not depend on density;
  - H2, D2 and CH4 with no other gas.

  solprop builds no response for any other mixture, and refuses it by name, naming the
  pair. Today's per-component, self-broadened mixture responses are therefore removed,
  as an accepted loss of capability. solprop does not substitute self-broadening for a
  missing partner (§4.4). The curation of foreign-gas broadening, such as H2, D2, CH4,
  N2 and O2 by He, Ne, Kr and Xe, is an optmat roadmap item (D14).
- **D17 — Mixture density stays the partial-pressure rule.** optmat's mixture equation
  of state [31] exists now. It is multi-fluid where binary models fit to data exist,
  qualified for He/Ar, Ne/Ar and He/Ne, and provisional for dry air. Adopting it is
  [SEP 11](../sep-0011-finite-density-gas-index.md)'s question Q4, not this SEP's (§4.2).
- **D18 — One backend qualification, no staging refusal.** The kernel arrays are
  unchanged, so an explicit optmat selection runs on every backend from Phase 1. One
  qualification phase, numba_cpu and then a freshly built Kokkos CPU backend, precedes
  the switch of defaults. Phase 4 is split into the switch, the deletion and the
  documentation (§11).

The scope decisions of the first draft also stand: curated defaults, with old
environments used for historical reproduction; and carrier-evaluated electronic
response now, with full dispersion in [SEP 6](sep-0006-dispersive-nonlinearity.md).

#### Explicit follow-up

- **[SEP 10](../sep-0010-low-rank-raman.md)**: low-rank Raman kernels, with H2's shift
  along density gradients and a cheaper rebuild.
- **[SEP 11](../sep-0011-finite-density-gas-index.md)**: the Lorentz–Lorenz law with
  refractivity virials, and mixture optics with cross terms.
- **Device data** (D3): measured mirror GDD and reflectance curves.
- **Solid ionisation and carriers**: [SEP 5](../sep-0005-solid-ionisation.md), with its
  rate models upstream under D15.
- **Further built-in materials**: CO2 first, then CO, gold, copper and further
  crystals, each separately qualified (D13).
- **Full dispersive third-order response**:
  [SEP 6](sep-0006-dispersive-nonlinearity.md).
- **Raman physics optmat has not yet curated**: foreign-gas broadening of the Raman
  lines by noble-gas buffers (D16), Q-branch line mixing at high density,
  N2O's rotational lines, modern H2/D2 level energies, local thermal populations
  along a temperature path, and [SEP 1](../sep-0001-raman-response.md)'s population
  dynamics.
- **Additional materials**: gold and copper are recommended in optmat but not built
  into solprop today, and are admitted only deliberately. The same applies to
  birefringent and quadratic crystals beyond those above
  ([SEP 8](../sep-0008-quadratic-crystal-optics.md)).

No Maxwell–Bloch dynamics, molecular line mixing, new solid ionisation model or
two-photon absorption is included. Universal constants, numerical tolerances,
quadrature choices, geometric parameters and user-supplied measurements are not a
material database, and may remain in solprop. Empirical fits, rate-correction
constants and source validity ranges do not become numerical constants merely
because they currently sit in a solver module.

### 3. Context

The governing design is [architecture §6.3](../../architecture.md), together with:

- the [materials](../../../guide/media/materials.md), [mixtures](../../../guide/media/mixtures.md),
  [Raman](../../../guide/media/raman.md), [ionisation](../../../guide/media/ionisation.md) and
  [compression](../../../guide/workflows/compression.md) manuals;
- the [kernel contract](../../kernel-contract.md);
- [conventions](../../conventions.md).

The table below is the removal and adaptation inventory, including data outside
`physdata`.

| Current owner in solprop | Current data or assumption | Disposition |
|---|---|---|
| `physdata/gases.py`, `sellmeier.py` | Species registry; published and unpublished coefficient sets; state reduction; $\chi=N\gamma(\omega)$ | Registry adapters and public names; optmat's polarizability at the existing linear density scaling (D5) |
| `physdata/equation_of_state.py`, `species.RegistryGas` | CoolProp fluid mapping, number density and inverse pressure | Delegate to optmat's density and its inverse; keep bar-to-Pa conversion at solprop's interface |
| `physdata/transport.py` | Viscosity tables of five gases and CoolProp lookups, their ranges, heat capacities, Wilke mixing [22] | Delegate the material properties to optmat; solprop keeps the flow problem, the kinetic-theory mean free path and its regime checks |
| `physdata/nonlinear_data.py`, `Species.gamma3` | One wavelength-independent scalar per particle | optmat's electronic Kerr coefficient at the declared carrier |
| `physdata/raman_data.py` | Molecular constants, spin weights, derivatives, linewidth laws; silica line and tensor parameters | Delete; optmat response objects; keep supplied-data types |
| `nonlinear/raman.py` built-in constructors | `vibrational`, `rotational_nonrigid`, `IntermediateBroadening(silica_raman())`, `SilicaTensorResponse` defaults | Adapters from optmat responses onto existing terms; supplied `RamanModel` types remain |
| `physdata/ionisation_data.py` | Thresholds, shell numbers, effective charges, polarisabilities, MO-ADK coefficients | Delete; optmat rate models and ionisation parameters |
| `nonlinear/ionisation.py` | `ADK`, `PPT`, `IsotropicMOADK`, barrier-suppression tables, occupancy default | Delete the formulas (D1); keep the rate protocol, conventions, diagnostics and `MAX_RATE` |
| `nonlinear/mixtures.py::_default_rate` | Per-gas choice of MO-ADK, ADK or PPT | optmat's recommended model per species |
| `translate/ionisation.py`, `RateTable` | Log–log spline of a rate callable | Unchanged; tabulates the optmat rate |
| `physdata/mixtures.py` | Hard-coded air recipes; additive per-particle susceptibility and Kerr scalar | optmat named compositions; same additive rule and labelled population paths (D5) |
| `physdata/glasses.py`, `data/silica_franta_2016.csv` | Silica hybrid default; CaF2, MgF2 (o), BaF2, sapphire (o), N-BK7 Sellmeier sets used as absolute indices | optmat recommended models, made absolute; `SellmeierGlass` and `TabulatedGlass` remain as user-data types |
| `physdata/metals.py`, `data/silver_stitched.csv`, `data/aluminium_stitched.csv` | Stitched n, k tables of Ag and Al | optmat table models |
| `tests/data/generate_*` table generators | Recreate the shipped tables | Delete with the tables |
| `nonlinear/gnlse.py`, drivers | `raman_fraction=None` resolves to an inline 0.18 with Raman on and to 0 with Raman off; `raman_susceptibility(n2)` | Bundle's $f_R$ in place of the inline 0.18 (D4); the rule for Raman off and for explicit fractions is unchanged |
| `data/mirrors/`, `optics.ChirpedMirror` | Measured mirror curves | Unchanged: device data (D3) |

Every consumer must be traced, not just convenience propagation:

- **Gas consumers:** `modes/fill.py`; the capillary, antiresonant, rectangular and
  slab modes; `spatial.py`; `modes/material.py` and the general modal providers;
  `profiles.py` and `gasflow.py`.
- **Solid consumers:** every `cladding=None → fused_silica()` default (capillary,
  antiresonant, rectangular, slab and step-index modes); `IndexMaterial` and
  `MaterialKerr`; `optics.Material`, prism and Brewster helpers;
  `processing/compression.py`; the design tools in `tools.py`; `SpatialMedium`; the
  `sol.py` re-exports.
- **Everything else:** response builders, scans, serialization and material-dependent
  tests.

Because D5 keeps linear density scaling, the frequency-only per-particle caches
(`GasFillMixin._sampled_susceptibility`, `Mixture._optical_sum`) and their fast paths
keep their meaning. Only their source changes.

#### 3.1 Availability in optmat

The dependency manuals [1–8] describe the recommended models and their limitations,
and the page for propagation codes [32] lists the public surface solprop builds on.
"Available" describes data and API availability at optmat 0.1.0, not qualification in
solprop.

**Gases: dispersion, Kerr and Raman**

| Species | Linear index and carrier Kerr | Raman response (`raman_response`) |
|---|---|---|
| He, Ne, Ar, Kr, Xe | Recommended fits; separate complex EUV models | Physically absent |
| H2 | Recommended fit | S and resolved Q lines; 3 damping groups; Q-line shift; 0.94–3290 amagat, 77–300 K |
| D2 | Recommended fit | S and Q lines; 12 groups; 0.62–1020 amagat, 291–298 K |
| N2 | Recommended fit | 53 S and 53 Q lines; 43 groups; 0.054–2.09 amagat, 293–300 K |
| O2 | Recommended fit | 32 S and 32 Q lines; 38 groups; 0.035–4.2 amagat, 290–1322 K |
| CH4, SF6 | Recommended fits, with acknowledged source-separation limits in the electronic model | One unresolved vibrational band each; no rotational anisotropy (spherical tops) |
| N2O | Recommended fit; Kerr dispersion partly constrained by priors | Two vibrational bands with provisional widths and depolarisation; no rotational lines |
| Dry air | `air_model("dry air (N2, O2, Ar)")` and `"dry air (N2, O2)"`: the compositions' summed polarizabilities; the refractometric formulas are validators | CIPM-2007 N2/O2/Ar composition, CO2 dropped; 85 S, 85 Q lines; 82 groups |
| CO2 (not built in) | Fit | 61 S lines and the Fermi dyad; Q widths provisional |
| CO, O3 (not built in) | Fits; O3's index is absorbing and its equation of state ideal | No curated Raman data |

Every gas model exposes an SI complex per-particle polarizability,
`complex_polarizability(wavelength)`, and its exact frequency derivative,
`dpolarizability_domega`. A model holds on its qualified `intervals` of vacuum
wavelength and warns outside them (`ModelRangeWarning`), naming the reason of a gap
between two intervals. The atoms, H2, D2, N2 and O2 have one interval, which reaches
the static limit. Their lower ends are 60 nm (He) to 195 nm (O2). The
infrared-active molecules have their windows between absorption bands:

- CH4: 240–1123, 1137–1149, 1177–1187 and 1188–1313 nm, and 11.1–12.1 µm;
- N2O: 320–1516 and 1518–1950 nm, 4.81–4.96 µm and 6.75–7.05 µm;
- SF6: 230 nm–5 µm.

A broadband grid routinely leaves these intervals; §5 states how solprop reports it.

A mixture's Raman response is built only from the pairs whose broadening optmat has
curated. N2 and O2 combine with each other and with Ar, SF6 with any atom, and H2, D2
and CH4 with no other gas (D16).

**Gases: ionisation rates (`ionisation_rate_model`, `molecular_rate_model`)**

| Species | Recommended model | Effective electrons | Stated uncertainty factor |
|---|---|---|---|
| He | Short-range PPT with static Coulomb factor and Tong–Lin correction | 1.47 | 1.3 |
| Ne, Ar | As He, with the two fine-structure channels of the ion | 1 | 1.5 |
| Kr, Xe | As Ne, with Tong–Lin's suggested α = 9 (no fitted value) | 1 | 2 |
| H2, D2 | Molecular model with orientation-averaged structure factors | 2 | 1.5 / 2 |
| N2 | Molecular model | 2 | 1.5 |
| O2, N2O | Molecular model | 2 / 4 | 2 / 3 (O2 was 3 at the first audit) |
| CH4 | Molecular model, Zhao et al. 2011 structure [20] | 6 | 2 (5 at the first audit) |
| SF6 | Molecular model, untested prediction (status `untested`) | 6 | 10 |
| CO, CO2 (not built in) | Molecular model | 2 / 4 | 1.5 / 3 |

All the rates are for **linear polarisation**. They take the field in V/m, return
1/s, and are vectorised. The atoms are served by `ionisation_rate_model` and the
molecules by `molecular_rate_model`; each refuses the other's species. Every rate model
gives `channel_rates`, `energy_loss_rate`, `energy_per_event` and `energy_kind`. The
molecular defaults use the vertical ionisation energy (`energy_kind="vertical"`), and
`ionisation_parameters` records the adiabatic value and its source. An atom's energies
are its curated adiabatic thresholds, which for an atom are also its vertical ones.
Each model declares a maximum field, beyond which it warns: twice the
barrier-suppression field with the Tong–Lin correction (8.6 × 10¹⁰ V/m for Ar at
800 nm), 4.5 times with Zhang's, and 1.5 times for the molecules. It also warns below
350 nm.

Named literature forms are selectable alongside the defaults:

- `ppt_ilkov`, `ppt_larochelle`, `ppt_popruzhenko`;
- `moadk_zhao_2010`, `moadk_tong_2002`, `moadk_zhao_2017`;
- `ppt_talebpour`, `ppt_talebpour_ilkov`;
- `adk_hydrogenic`.

The MO-ADK forms, `ppt_talebpour_ilkov` and `adk_hydrogenic` reproduce solprop's
current molecular rates to $10^{-8}$–$10^{-10}$ when given solprop's energies and
occupancies [5].

**Solids (`optmat.solid.models.registry.default_model`)**

| Material | Recommended model | Range | Absorbing | Reference state |
|---|---|---|---|---|
| Fused silica | `fit/SiO2`: Franta's universal dispersion model refitted to absolute refractometry [9, 10] | 24.8 nm–125 µm | Yes, causal and passive | 20 °C bulk grade |
| CaF2 | `daimon_2002/sellmeier_20C` [11] | 138 nm–2.33 µm | No | 20 °C; measured in N2, made absolute |
| MgF2, o and e | `dodge_1984/sellmeier_o`, `_e` [12] | 0.20–7.04 µm | No | 19 °C; room air, made absolute |
| BaF2 | `malitson_1964/sellmeier` [13] | 0.27–10.3 µm | No | 25 °C; room air, made absolute |
| Sapphire, o and e | `weber_2003/sellmeier_o`, `_e` (Malitson & Dodge) [14] | 0.22–5.0 µm | No | Room temperature; air, made absolute |
| N-BK7 | `schott_2023/sellmeier` [15] | 0.31–2.33 µm | No (transmittance curated separately) | 20 °C catalogue |
| Silver | `table/Ag` | 4.1 nm–12.4 µm | Yes | Tabulated, stitched from primary sets [16] |
| Aluminium | `table/Al` | 0.79 nm–200 µm | Yes | Tabulated, stitched from primary sets [16] |

A model's methods take the vacuum wavelength in metres: `permittivity`,
`dpermittivity_domega`, `refractive_index` and `dindex_domega`. The derivatives are
exact:

- analytic, for the Sellmeier and fitted models;
- the interpolant's own derivative, for the PCHIP metal tables.

A birefringent crystal refuses to resolve without an explicit axis. A metal table
holds its end values outside its range and warns. Only silica has a Kerr model
(`kerr/SiO2`) and Raman responses; the other solids' $n_2$ values are curated
datasets, not models. solprop has no built-in solid Kerr data today, so this is no
regression.

**Silica nonlinear response**

- `kerr_models()["kerr/SiO2"]` gives the long-pulse $n_2(\lambda)$ and its
  electronic part. Its Raman fraction, $f_R=0.219\pm0.026$, is set by the measured
  absolute Raman gain [17] rather than by a prior.
- `silica_raman_response(model)` offers three shapes, each with a unit-integral
  `response(t, component)` and an analytic `spectrum`:
  - `hollenbeck_2002` [18], the default: paired with the Kerr fit and copolarised
    only;
  - `lin_2006` [19]: isotropic and anisotropic parts, $f_R=0.245$, not paired;
  - `blow_1989`: $f_R=0.222$, not paired.
- `silica_bundle(name)` (`optmat.solid.bundle`) pairs the Kerr fit's total $n_2$ with
  a shape and its $f_R$, so that the two cannot be mismatched. `"default"` is
  Hollenbeck's shape, paired. `"tensor"` is the tensor default of D11. Its
  `h_r(t, component)` gives the parallel response or, for `lin_2006`, Hellwarth's $a$
  and $b$. Hollenbeck's shape refuses $a$ and $b$.

#### 3.2 Upstream gaps

Phase 0 closed these in optmat before implementation (D6), through OEP 1 [30], with the
equation-of-state object reused by OEP 4 [31]. The table records each requirement and
what optmat 0.1.0 delivers for it.

| Gap | Required upstream treatment | Delivered in optmat 0.1.0 |
|---|---|---|
| G1. Gas polarizability derivative | A public SI complex `dpolarizability_domega` for every gas model, exact where the model is analytic, with the same range semantics | `Model.dpolarizability_domega` on every registry model and on `AirModel`, analytic branch by branch, O3's measured term included |
| G2. EOS inverse and transport | A consistent inverse of the density law on a declared gas and supercritical domain; sourced viscosity and heat capacities with ranges, and a stated mixture rule; a real-gas sound speed; the vacuum limit handled by solprop | `equation_of_state(gas)`: `number_density`, `pressure` (round trip within 10⁻¹²), `phase_domain`, `saturation_pressure`, `speed_of_sound`, with `StateError` for a condensed state and for zero density or pressure; `transport_model(gas)`: dilute `viscosity`, ideal-gas `heat_capacity`, `molar_mass` and their ranges; Wilke's `mixture_viscosity` |
| G3. Named compositions | The N2/O2-only air beside CIPM-2007 dry air, with its renormalisation stated, so that solprop's two air recipes come from optmat | `composition("dry air (N2, O2, Ar)")`, `"dry air (N2, O2)"` and `"dry air (CIPM-2007)"`; `air_model(name)`, one identity per name; `air_equation_of_state()`, Lemmon et al.'s pseudo-pure air |
| G4. Model identities | One identity record on every model and response: name, kind, content digest of its evaluation definition and data, optmat version and sources | `ModelIdentity` on every model, response, bundle, equation of state and transport model: a SHA-256 digest of its parameters, data, intervals, evaluator versions and bound components; the version beside it, not in it |
| G5. Explicit parameters | A public constructor from explicit, cited parameters for rate models (custom species need it, D1), and overrides on rate, Raman and solid models returning a model whose identity is marked modified (D7) | `IonisationRateModel.from_parameters` and `MolecularRateModel.from_parameters` with a plain species name; `GasRamanResponse.from_lines`; `FormulaModel.sellmeier`, `TabulatedModel.from_arrays`; `with_parameters(**inputs)` on every class with declared `INPUTS`, giving the status `user`, the changes in `modified` and the base's digest in `base` |
| G6. Raman groups and metadata | Stable line identifiers; a documented group-level view giving each damping group's lines, amplitudes, shift and width law in SI; a documented zero-density limit | `line_id`; `groups()` of `RamanGroup` with SI amplitudes, frequencies, shift and `decay_rate(N)` defined for $N\ge0$; `check_densities(N)` for the validated range |
| G7. Rate metadata | Per-channel rates summing to the rate, the energy-loss rate and the rate-weighted energy per event, and the kind of the energies the rate uses | `channel_rates`, `energy_loss_rate`, `energy_per_event` and `energy_kind` on both rate models |
| G8. Silica bundles | A named bundle API holding the Kerr fit's total $n_2$ with a shape and its $f_R$; the tensor default of D11 with its gain check | `silica_bundle("default")` (Hollenbeck, paired, $f_R=0.2188$) and `silica_bundle("tensor")` (`lin_2006`, $f_R=0.245$), with `n2`, `n2_electronic`, `n2_raman`, `h_r(t, component)` and `gain_check` |
| G9. Capability flags | Machine-readable flags on models and responses, so that solprop can warn without parsing text | `status`: `validated`, `provisional`, `untested`, `comparison` or `user`, with `status_reason` |
| G10. Distribution | A tagged, reproducible release containing all data and identities before solprop's own release; a pinned commit is enough during development | Tag `v0.1.0`, whose clean-install test runs the public surface from the built wheel and sdist. It is not yet published, so publication remains required before solprop's release (§6) |

The following remain *limitations* rather than prerequisites, and are documented,
not closed, by this SEP:

- molecular Raman physics optmat has not curated (§2);
- one grade and temperature per solid, a lossless N-BK7, and no OH, dopant or
  temperature absorption model;
- high-density corrections, which are [SEP 11](../sep-0011-finite-density-gas-index.md).

### 4. Theory

#### 4.1 Units, state and derivatives

solprop's interface continues to accept pressure in bar and temperature in K. The
conversion to optmat is exactly $p_{\rm Pa}=10^5p_{\rm bar}$. Defaults pass 294 K
explicitly rather than inheriting optmat's reference state; a source's reference
state is not the simulation's operating state. optmat's methods take vacuum
wavelengths, $\lambda=2\pi c/\omega$, and

$$
\frac{d}{d\omega}=-\frac{\lambda}{\omega}\frac{d}{d\lambda}.
$$

Material derivatives are taken at fixed state. Do not query negative frequencies or
DC as finite vacuum wavelengths; a static limit is an explicit `inf` wavelength
where a model supports it.

optmat's real-gas number density is $N=p/(Zk_BT)$, whereas solprop currently derives
it from mass density and molar mass. The gas constant embedded in an EOS can differ
from present CODATA, so the two are not numerically identical even in the dilute
limit (§9.1: −15 to +6 ppm). Inversion must solve the *same* forward law; keeping
the old inverse would break density/pressure round trips.

At $N=0$, the gas susceptibility, nonlinear polarisation, Raman polarisation and
plasma source vanish, and the material index is exactly one. Do not evaluate an
EOS or a rate model at vacuum and rely on multiplication by zero to repair a NaN.
(A capped Dicke width is finite at vacuum, §4.4.)

**Phase domain.** optmat's density and its inverse are defined for the gas below
saturation and the fluid above the critical temperature, and refuse a liquid, a
compressed liquid or a two-phase state ([30], §4.2). solprop's present density
returns a liquid's past saturation (SF6 at 30 bar and 294 K gives 217 amagat), so a
fill in that state, accepted today, is refused with the gas, state and saturation
pressure named.

#### 4.2 Gas index at linear density scaling

By D5, the gas law keeps its present form, with optmat's SI per-particle
polarizability $\alpha(\omega)$:

$$
\chi(\omega;z)=\sum_s\frac{N_s(z)\,\alpha_s(\omega)}{\varepsilon_0},\qquad
\partial_\omega\chi=\sum_s\frac{N_s\,\partial_\omega\alpha_s}{\varepsilon_0},\qquad
n-1=\frac{\chi}{\sqrt{1+\chi}+1}.
$$

Form $n-1$ this way, including for a weakly absorbing gas, to keep dilute
precision. The physical branch has nonnegative absorption. optmat's *refractive*
polarizability already includes the magnetic contribution, so it must not be added
again. It is not the electronic static polarizability.

This is the dilute limit of optmat's own constitutive law, which applies
Lorentz–Lorenz and a refractivity virial to the same $\alpha$ [1]. The two differ by
a relative amount of order $N\alpha/(3\varepsilon_0)$ plus the virial term. Measured
against optmat's `refractivity` at 800 nm and 294 K, the relative error of dilute
scaling in $n-1$ is:

| Gas | 1 bar | 10 bar | 30 bar | 60 bar |
|---|---:|---:|---:|---:|
| He | −0.002 % | −0.016 % | −0.047 % | −0.093 % |
| H2 | −0.009 % | −0.086 % | −0.26 % | −0.50 % |
| Ar | −0.019 % | −0.19 % | −0.57 % | −1.16 % |
| N2 | −0.019 % | −0.19 % | −0.57 % | −1.13 % |
| Xe | −0.052 % | −0.55 % | −1.9 % | −6.5 % |
| SF6 | −0.055 % | −0.62 % | — | — |

SF6 condenses near 21 bar at this temperature, so its higher entries are omitted.
This error is not new: solprop's present law has the same form. It is, however,
comparable to the data improvements at high pressure (§9.1).

The adapter therefore evaluates the dilute error at each gas's largest density in
the run, by comparing optmat's two laws directly. It warns once, with the value,
when the error exceeds a declared threshold ($10^{-3}$ relative in $n-1$ is proposed),
and records it. [SEP 11](../sep-0011-finite-density-gas-index.md) replaces the law.

**Mixtures.** Mixtures stay additive in per-particle susceptibility. That is exact
in the dilute limit and has the same error order as above. Repeated labelled
populations of one chemical species keep separate labels for plasma histories and
output. Air compositions come from optmat's named records (G3), which state their
omitted constituents and renormalisation. optmat's standard-air refractometry
formula is not a replacement for labelled N2/O2/Ar propagation populations. The
partial-pressure EOS approximation for mixtures remains, separately identified: each
species' density is optmat's pure-gas density at its partial pressure (D17). optmat's
mixture equation of state [31], with its unlike-pair interactions, is the better
physics. Adopting it is [SEP 11](../sep-0011-finite-density-gas-index.md)'s question Q4.

`Gas.Air`, the dispersion-only registry entry, resolves to the named dry-air
composition's summed polarizabilities (D9), `"dry air (N2, O2, Ar)"`. It is supplied
by optmat as a named model with its own identity and derivative, and validated against
the standard air formula. solprop's argon-free recipe, which its interface string
`"air"` uses, resolves to `"dry air (N2, O2)"`, a model with its own identity.

`Gas.Air`'s density comes from the air model's equation of state, CoolProp's
pseudo-pure dry air (D9). `Mixture.air`'s labelled populations keep the
partial-pressure approximation, whose total density is lower by
$1.9\times10^{-4}$ at 1 bar, $1.8\times10^{-3}$ at 10 bar and $6.0\times10^{-3}$ at
50 bar (294 K, N2/O2/Ar). The two routes therefore differ in density, and the materials
manual says so; their per-particle polarizabilities are identical.

#### 4.3 Carrier-evaluated electronic response

The curated hyperpolarizability uses the Taylor convention [2, 21]: a molecular
dipole has the cubic term $p^{(3)}=\gamma_T E^3/6$. In solprop's
$P^{(3)}=\varepsilon_0\chi^{(3)}E^3$ convention this gives

$$
\gamma_3(\omega_0)=\frac{\gamma_{T,\mathrm{Kerr}}(\omega_0)}{6\varepsilon_0},
\qquad \chi^{(3)}(z)=\sum_s N_s(z)\gamma_{3,s}(\omega_0).
$$

Use the public `gamma3("kerr", wavelength)`, which already returns solprop's
per-molecule coefficient, rather than copied atomic-unit factors or fitted
coefficients. No molecular Raman amplitude is multiplied by this electronic
coefficient. The residual vibrational contribution that optmat acknowledges in some
ESHG-based fits is an uncertainty, not permission to subtract an invented
correction.

For a monochromatic field, $I=n_0\varepsilon_0c|E|^2/2$ and the fundamental
component of $E^3$ give

$$
n_2=\frac{3\chi^{(3)}}{4\varepsilon_0c n_0^2}
=\frac{N\gamma_{T,\mathrm{Kerr}}}{8\varepsilon_0^2c n_0^2}.
$$

optmat's gas `n2` helper uses the dilute conversion $n_0\simeq1$. Importing that
value and converting back with solprop's exact $n_0^2$ would introduce an
unintended factor, so import $\gamma_3$ and use the existing field and power
convention. Do not add Lorentz local-field factors or nonlinear pair terms; this
nonlinear model lacks them. The general modal prefactors, the vacuum-normalised
internal amplitudes and the GNLSE calibration remain as
[conventions §4](../../conventions.md) specifies.

A single frozen Kerr coefficient cannot reproduce optmat's distinct THG coefficient.
This SEP keeps the existing single-coefficient cubic operator, including when
`thg=True`, and labels it `carrier-kerr`:

- the carrier is the grid/input reference, not a peak tracked during propagation;
- it is an explicit nondispersive approximation, not correct broadband four-wave
  mixing or process-correct THG;
- the THG/Kerr discrepancy is reported where the upstream THG evaluation is within
  range (§9.1).

[SEP 6](sep-0006-dispersive-nonlinearity.md) supplies frequency dependence.

#### 4.4 Molecular Raman responses on the existing seam

optmat's convention [3] is solprop's. For a medium of total number density $N$,
built for one composition, one temperature $T$ and one pump wavelength
$\lambda_p$:

$$
P_{R,x}(t,z)=\varepsilon_0N(z)E_x(t,z)\int_0^\infty h(\tau;N)E_x^2(t-\tau,z)\,d\tau,
$$

$$
h(\tau;N)=\Theta(\tau)\sum_k A_k\sin(\Omega_k(N)\tau)\,e^{-\Gamma_k(N)\tau},\qquad
\Gamma_k=\pi\,\Delta\nu_k(N).
$$

The ingredients are:

- **Amplitudes.** $A_k$, in m$^5$ V$^{-2}$ s$^{-1}$, already include the constituent's
  mole fraction, so a mixture's response is driven by the total density.
- **Frequencies.** $\Omega_k(N)=\Omega_{k,0}+2\pi s_k n$ in rad/s, with $n$ in
  amagat and $s_k$ the shift in Hz/amagat. Only H2's Q lines shift [28].
- **Widths.** The FWHM in Hz is
  $\Delta\nu_k=c_k+\min(d_k/n,\,\Delta\nu_{D,k})+\beta_kn$: a constant, a
  Dicke-narrowing term capped at the forward-scattering Doppler width, and
  collisional broadening.

The factor $\pi$ arises because the amplitude decay $e^{-\Gamma t}$ has Lorentzian
half width $\Gamma$ in angular frequency, hence full width $\Gamma/\pi$ in Hz. Do not
multiply an already angular line position by $2\pi$, or a per-molecule amplitude by
density twice. Envelope drive factors remain those of the Raman manual.

**Mapping onto terms.** Lines of one damping group share their width and shift at
every density. Group $g$ and tensor family $\kappa$ therefore become one existing
Raman term:

$$
u_{g\kappa}(t)=w(t)\sum_{k\in g}A_{k\kappa}\sin(\Omega_{k}t),\qquad
\gamma_{g}(z)=\pi\,\Delta\nu_g\big(n(z)\big),
$$

where $w$ is the grid's existing causal taper. $u_{g\kappa}$ is the undamped row and
$\gamma_g$ the decay-rate path. This is **exact** for every gas whose lines do not
shift, at any density path. The kernel contract already admits any number of terms;
the Kokkos and Numba backends take the row count from the array shape. N2 gives
about 43 groups and dry air about 82, before the tensor split.

**H2's shift.** With a shift, the row depends on density.

- At constant density, the row is built at that density and the representation is
  exact.
- On a varying density path, the H2 Q-branch terms are **refused by name** unless the
  user selects the explicit approximation `raman_shift="reference-density"`. That
  freezes the shift at the density-weighted reference density, the analogue of the
  reference-temperature rule, and records the phase bound
  $2\pi\,|s|\max_z|n(z)-n_{\rm ref}|\,T_w$ over the window $T_w$.

[SEP 10](../sep-0010-low-rank-raman.md) removes this restriction.

**Cost.** The kernel rebuilds $\sum_g e^{-\gamma_g t}u_g$ and transforms it whenever a
rate changes. With tens of groups this costs several milliseconds per rebuild at
$2^{14}$ samples, against about 0.2 ms per convolution [3]. At constant density it
happens once per run. On a density gradient it is a real, measured cost, recorded
in [benchmarks](../../benchmarks.md) and addressed by SEP 10. Groups are not merged to
save work.

**Tensor form.** Each line has isotropic and anisotropic amplitudes. optmat's
components use Hellwarth's decomposition
$R_{ijkl}=a\,\delta_{ij}\delta_{kl}+\tfrac12 b(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$:

- `parallel` $=a+b=$ iso + aniso;
- $a=$ iso $-\tfrac12$aniso;
- $b=\tfrac32$aniso.

S lines are purely anisotropic, and a Q line's anisotropic share follows its
Placzek–Teller factor. solprop's Raman leaf gives each term trace/traceless weights:
$(1,0)$ for isotropic rows and $(1/4,3/4)$ for rank-two rows. The adapter maps
isotropic amplitudes to isotropic terms and anisotropic amplitudes to rank-two
terms. The mapping must be *derived and verified*, not asserted:

- a copolarised probe sees $a+b$;
- a cross-polarised probe sees $a$, which for a pure S line is $-\tfrac12$aniso.

**Pump dispersion.** optmat disperses vibrational strengths to the pump wavelength
it is given; without one it returns static or source-wavelength strengths. The
adapter always passes the grid reference wavelength, the carrier of §4.3 and §4.6.

**Temperature and composition.** A response object fixes its temperature, which
enters both the populations and the widths' temperature laws. solprop keeps its
reference-temperature rule for a temperature path: the density-weighted reference,
or an explicit override, with its warning. That rule now fixes the widths'
temperature laws too, while the densities follow z.

A response is built for one composition, with each constituent's true collision
partners and upstream substitutions named in provenance. Duplicate labels of one
chemical species merge before the composition is formed. If component density paths
are not proportional, the composition varies along z. That case is refused by name,
rather than handled with a self-broadening shortcut.

A composition is admitted only where optmat has curated the broadening of every
Raman-active constituent by every partner (§3.1). N2 and O2 may be with each other and
with Ar, SF6 with any atom, and H2, D2 and CH4 alone. Any other mixture with Raman,
H2 or N2 in helium for instance, is refused by name, naming the pair optmat lacks
(D16). A fill whose Raman-active constituents are all physically absent, such as a
noble-gas mixture, has no Raman response and is not refused. Since the response is
driven by the total density of the composition, its decay-rate paths are evaluated on
the summed density of the labels that form it, which is still one path per term.

**Provisional data.** N2O's vibrational bands are admitted only with an explicit
`raman="provisional"`, which warns once and is recorded (D13). Otherwise N2O Raman
is refused, as today.

**Range and absence.** optmat warns outside each response's density and temperature
ranges. Among the stated limits:

- N2 and air above 2.09 amagat, where Q lines overlap and line mixing is not
  modelled, and above 300 K;
- H2 below 0.94 amagat and D2 below 0.62 amagat, the Dicke cutoff.

solprop declares the densities a run uses to the response's `check_densities` once
at translation. Those are the non-zero densities of the path, because the
polarisation is exactly zero in vacuum. The response's temperature range is checked
in the same way. These warnings are recorded, and solprop does not suppress them.
Capillaries filled with several bar of N2 or air therefore run outside the
independent-line model's validated range. So does H2 at 1 bar and 294 K, which is
0.92 amagat. A gradient that falls into the Dicke regime warns as well. The manifest
and the Raman manual say so.

**Vacuum.** The polarisation carries the actual $N(z)$ and vanishes with it. optmat's
group view gives each group's decay-rate law in SI, defined for $N\ge0$: every Dicke
term is capped at its Doppler width, so the law has a finite limit at zero density,
$\pi(c+\Delta\nu_D)$ with a Dicke term and $\pi c$ without (zero for N2's and O2's
lines, whose widths are pure broadening). The decay-rate paths are evaluated at the
actual density, vacuum included, with no floor or clamp. The response's validated
density range still warns once at translation for the non-zero densities the run
declares.
The existing cap at `MAX_DECAY_EXPONENT / dt_fine` remains a numerical guard.

The analytic line list stays in the host-side record and the manifest, so an exact
pole-based evaluator such as [SEP 1](../sep-0001-raman-response.md)'s recurrence can
still use it.

#### 4.5 Silica nonlinear response

For a scalar silica response use the unit-integral $h_R$, the total long-pulse
$n_2(\lambda_0)$ and the matching fraction $f_R$ once:

- electronic part $\chi_e^{(3)}=(1-f_R)\chi^{(3)}_{\rm total}$;
- delayed part $f_R\chi^{(3)}_{\rm total}$, convolved with $h_R$.

`n2_effective` predicts a finite-pulse measurement. It is not the material $n_2$ to
feed a propagation that already convolves $h_R$; doing both counts pulse-duration
weighting twice.

By D4, the default scalar response is `hollenbeck_2002` with $f_R=0.219$ from
`kerr/SiO2`. That holds when $n_2$ is also taken from `kerr/SiO2` at the carrier,
and when a user supplies $n_2$ or $\gamma$. In the GNLSE it is the value that
`raman_fraction=None` resolves to with Raman on. The existing rule for Raman off
(0 when left out, held back when given explicitly) is unchanged. The adapter samples optmat's
`response(t)` into the existing single undamped row with a zero decay path, since
the shape is density independent. Hollenbeck's shape is copolarised only, and optmat
refuses its $a$ and $b$ components. A vector response, or a `MaterialRegion`
needing the tensor, uses the tensor default of D11, Lin–Agrawal's response with its
own $f_R$ and its two channels. Its $f_R$ differs from the scalar default's, so the
GNLSE's `raman_fraction=None` resolves to the fraction of the bundle in use.

The tensor bundle gives Hellwarth's $a$ and $b$, `h_r(t, "a")` and `h_r(t, "b")`,
which are normalised so that $\int(a+b)\,dt=1$. With the decomposition of §4.4,
$a=h_0-\tfrac12h_2$ and $b=\tfrac32h_2$. solprop's isotropic row is therefore
$h_0=a+\tfrac13b$ and its rank-two row $h_2=\tfrac23b$, with the weights $(1,0)$ and
$(\tfrac14,\tfrac34)$. The copolarised probe then sees $a+b$ and the cross-polarised
probe $a$; both are tested. A vector run on Hollenbeck's copolarised-only shape is
refused by name.

**Convention.** optmat's `raman_coefficient` and `electronic_susceptibility`
include $n_0^2$ from its silica index:
$K=2n_0^2\varepsilon_0cf_Rn_2$ and $\chi_e=\tfrac43n_0^2\varepsilon_0c(1-f_R)n_2$.
solprop's GNLSE calibration and its vacuum-normalised internal amplitudes define the
coefficients with $n_0=1$ in these formulas ([conventions §4](../../conventions.md)).
Importing $K$ or $\chi_e$ would multiply the nonlinearity by $n_0^2\approx2.1$. The
adapter therefore imports $n_2$, $f_R$ and the normalised $h_R$, and applies
solprop's own conventions. A parity test records the $n_0^2$ ratio.

**Normalisation.** Hollenbeck–Cantrell's Table 1 footnote [18] gives the width
conversions and the relation of the tabulated amplitudes to their equation (9). Its
Gaussian time factor is $\exp(-G_j^2\tau^2/4)$, and both tabulated widths convert
with $\pi c$ when the wavenumber unit is used consistently. optmat normalises the
shape analytically through the Faddeeva function, as solprop does, and the two unit
integrals are compared independently.

#### 4.6 Ionisation rates

By D1, the rate is optmat's model. solprop keeps the rate *protocol*: a vectorised
rate in 1/s of $|E|$ in V/m, exactly zero below a declared threshold field, capped
at `MAX_RATE`, and carrying a hashable parameter tuple. It also keeps the rate's
uses: tabulation, plasma evolution, energy loss, currents and preionisation checks.

**Carrier.** Models are constructed at the grid reference wavelength. optmat warns
below 350 nm; the warning is recorded, not suppressed.

**Field grids.** The adapter tabulates `instantaneous_rate(|E|)` into the existing
log–log `RateTable`, unchanged. Its threshold is the field below which the rate is
under a declared negligible value, recorded with the table.

**Envelope grids.** solprop's envelope plasma averages the instantaneous rate over
the polarisation ellipse by its existing quadrature, so one definition applies at
every ellipticity. At linear polarisation this average is checked against optmat's
`cycle_averaged_rate`. optmat's "consistent" averaging makes the two agree to about
3 % up to the barrier-suppression field [4]. The check is a qualification test, not
a switch between definitions.

**Polarisation.** The upstream rates are defined for linear polarisation.
Evaluating the instantaneous rate along an elliptical trajectory is a quasi-static
approximation. It is admitted as today and recorded, with a one-time warning when
the Keldysh parameter at peak field exceeds 1 (D12).

**Threshold and energy loss.** The plasma energy loss per ionisation event uses the
selected model's own threshold (G7). The molecular defaults use the **vertical**
ionisation energy, so their plasma loss must not use the adiabatic value. The
noble-gas defaults sum two fine-structure channels with different thresholds, so the
loss is the rate-weighted channel energy $\bar I(E)=L(E)/W(E)$, where optmat supplies
the energy-loss rate $L=\sum_c I_cW_c$ and the channel rates (D8, G7). $\bar I$ moves
with the field: for Ar at 800 nm from 15.797 eV at $5\times10^9$ V/m to 15.814 eV at
$5\times10^{10}$ V/m, between channels at 15.760 and 15.937 eV (0.25 %); for Xe, whose
channels are 1.31 eV apart, from 12.14 eV at $5\times10^9$ V/m to 12.42 eV at its
maximum field (2.3 %). Each plasma population of solprop's
kernel carries one energy, used in the field formulation's loss term and outside the
cycle moment of the envelope formulation. Representing $\bar I(E)$ exactly would need
a second table per population, a kernel-contract change. This SEP keeps the kernel and
uses a constant per species, $\bar I$ at the run's peak field, where most ionisation
happens (D8). The peak field is the input's peak field amplitude, the one that sets the
default field range of the rate table, or a field the user gives, and it is clipped to
the table. The post-run diagnostic also reports $\bar I$ at the largest field each
population saw, which self-compression can raise above the input's. The manifest
records the constant and its bound,
$\max|\bar I(E)/\bar I(E_\text{peak})-1|$ over the tabulated fields at which the
rate is not negligible. A one-channel model, every molecular default among them, is
exact. In the tunnelling exponent $-2(2I_p)^{3/2}/(3|F|)$ (atomic units), a small
change of threshold is not a small change of rate. Changing the exponent's threshold
without the loss's, or the reverse, is a model error; the constant energy approximates
only the loss, never the rate.

**Validity.** Each model declares its maximum field. The manifest records it, and a
post-run diagnostic compares it with the largest field each channel saw. The rate
table's default range extends far above the input's peak field, and so routinely
beyond the maximum field, where optmat warns at every evaluation. The adapter therefore
records optmat's maximum-field warning during tabulation instead of emitting it. It
warns at translation only when the input's own peak field exceeds the maximum, and
otherwise leaves the verdict to the post-run diagnostic. `MAX_RATE`
remains a numerical safety cap, and no new silent clamp is added. The uncertainty
factor (1.3 for He up to 10 for SF6) is carried into provenance. It describes the
model; it is not a software tolerance.

**What changes physically.**

- The atomic model uses optmat's effective electron numbers: 1.47 for He from its
  two-electron comparison, 1 for Ne to Xe. Today's occupancy of two doubles an
  already correct low-field He rate [4].
- It omits a Stark shift and a dipole correction, which would double-count what the
  fitted structure already contains. solprop's current PPT applies both by default,
  and their source could not be traced [4].
- CH4, SF6 and N2O move from the potential-only ADK form to optmat's molecular model.
- H2, D2, N2 and O2 move from isotropic MO-ADK with occupancy one to the molecular
  default, with shell electron counts and a barrier-suppression correction.

§9.1 measures the combined effect.

**Named forms.** The literature forms remain selectable by name, for comparison.
Selecting one does not bring back solprop's own implementation. The molecular forms
reproduce solprop's current molecular rates (§3.1), but no upstream form carries
solprop's atomic PPT Stark and dipole factors, so reproducing old atomic results
relies on the old environment. The cautions attached to those forms still apply:

- Talebpour's effective charges belong to a PPT construction fitted at a stated
  threshold; O2's used 12.55 eV [23];
- orbital $l$ and the effective $l^*$ are different quantities;
- MO-ADK structure coefficients are bound to their normalisation.

**Custom species.** A custom species with its own energy, orbital and structure gets
its rate through optmat's constructor from explicit parameters (G5), or through a
user-supplied rate object. No solprop formula survives as a fallback.

Curated data do not make approximate rate theories accurate for every species and
field. optmat reports its own discrepancies:

- helium within about 20 % of all-electron static and laser-field rates [24, 25];
- measured argon, krypton and xenon yields reproduced within the calibration of one
  experiment [26], but 1.7 to 8 times above another's absolute yields [27];
- no reference at all for SF6's rate.

These limits are carried into solprop's manual, which shows both yield comparisons
(D12). They are not removed by a wider regression tolerance.

#### 4.7 Solid dispersion, absorption and propagation sign

Use one complex model $n=n_r+i\kappa$, $\kappa\ge0$, with $\varepsilon=n^2$ and
$\partial_\omega n=(\partial_\omega\varepsilon)/(2n)$. For the interface plane wave
$e^{i(\beta z-\omega t)}$,

$$
I(z)=I(0)e^{-\alpha_{\rm abs}z},\qquad
\alpha_{\rm abs}=\frac{2\omega\kappa}{c}=\frac{4\pi\kappa}{\lambda}.
$$

Inside solprop the linear generator remains
$D=-i(\beta_r-\beta_0-\beta_1(\omega-\omega_0))-\alpha_{\rm abs}/2$. Do not conjugate
optmat's index on entry: it already follows solprop's positive-loss convention. The
named Fourier boundaries are unchanged.

**Absolute index.** Every optmat solid index is relative to vacuum. The published
Sellmeier formulas of CaF2, MgF2, BaF2, sapphire and N-BK7 were measured relative to
air, or to nitrogen for Daimon & Masumura, and optmat multiplies by the reference
gas's index [6]. solprop currently uses those formulas as absolute indices, which
is $3.7$–$5.6\times10^{-4}$ low in the transparent window (§9.1). This is a
correction, not a disagreement between sources.

**Silica.** The recommended model returns the absolute index with coherent real and
imaginary dispersion from 24.8 nm to 125 µm. The old hybrid combines a real
Sellmeier formula with unrelated extinction and inherits the formula's pole near
116 nm, so it is not a valid substitute in the VUV.

**Birefringence.** MgF2 and sapphire are uniaxial. A built-in window or wall of
either names its ray (`axis="o"` or `"e"`), because optmat refuses to guess. The
present o-ray-only functions become explicit, a deliberate breaking change.
Propagation along a general direction in a birefringent crystal is
[SEP 8](../sep-0008-quadratic-crystal-optics.md)'s scope.

**Metals.** Silver and aluminium are complex tables with exact interpolant
derivatives, held at their end values outside their ranges with a warning. Where
their stitched sources differ from solprop's, their values differ; §9.1 gives the
largest changes. A metal wall uses the complex index in the existing wall
calculation.

**N-BK7.** The recommended model is lossless. Its measured internal transmittance is
curated separately, and joining the two would be a new model, so N-BK7 stays lossless
here and says so.

**Grade and temperature.** These are material attributes. The gas temperature beside
a capillary is not automatically the wall temperature. Alternatives remain explicit
selections, such as Malitson's silica formula, Leviton & Frey's
temperature-dependent silica model, or Malitson's 1963 CaF2 formula beyond 2.33 µm.

**Loss accounting.** Material absorption is distinct from waveguide leakage,
scattering, device loss and numerical absorbing collars.

- Pass the complex wall index into the existing wall calculation, and do not add
  bulk attenuation again to a hollow-core mode.
- Bulk propagation uses the bulk attenuation once.
- A lossless-only general modal provider must refuse a complex material it cannot
  solve, or receive an explicit, recorded transparent approximation within a
  justified domain. It must not silently discard optmat's small imaginary part to
  pass its old admission check.
- Species-resolved gas absorption statistics are not an exact decomposition of an
  absorbing mixture's index; label them as an approximation.

### 5. Proposed interface

Familiar names are kept, and their built-in defaults resolve to optmat once per
simulation:

- gases: `Gas.Ar`, `species("Ar")`, `Mixture.air`;
- solids: `fused_silica()`, `calcium_fluoride()`, `magnesium_fluoride(axis=)`,
  `barium_fluoride()`, `sapphire(axis=)`, `bk7()`, `silver()`, `aluminium()`,
  `glass(name)`;
- drivers: `prop_capillary(...)`, `prop_gnlse(...)`.

A small immutable selection record in `physdata` names the linear, electronic, Raman
and ionisation models, the approximation policies and any overrides. The resolved
record carries provenance, capabilities and identities. Exact names are proposed
interface work, not APIs present in this Draft:

```text
selection = MaterialSelection(
    linear="recommended",
    electronic="carrier-kerr",
    raman="recommended",          # or an optmat response name
    ionisation="recommended",     # or "ppt_ilkov", "moadk_zhao_2010", ...
    carrier_wavelength=1030e-9,
)
argon = species("Ar", selection=selection)
fill = Mixture.fill({argon: 1.0}, pressure=3.0, temperature=294.0)

wall = fused_silica()                       # fit/SiO2, absolute, 20 °C grade
window = sapphire(axis="o")                 # weber_2003/sellmeier_o
rate = OptmatRate("Ar", model="recommended", wavelength=800e-9)
```

The adapters accept either a name, resolved through optmat's registries (atoms through
`ionisation_rate_model`, molecules through `molecular_rate_model`), or an optmat model
object, which they take as it is.

**Quick experiments (D7).** Each material quantity has a one-line path that needs
no change to optmat:

```text
# modify a curated model with optmat's own override: status "user", base digest kept
rate = OptmatRate(ionisation_rate_model("Ar", 800e-9).with_parameters(electrons=2))
lines = raman_response("N2", 294.0, wavelength=800e-9).with_parameters(width_scale=0.8)
# a custom species from explicit, cited parameters
xenon_like = OptmatRate(IonisationRateModel.from_parameters(
    "X", 800e-9, ionization_energy=..., orbital_l=1, electrons=1, citation="..."))
# or bring your own, with its own citation
glass = SellmeierGlass(poles=[...], citation="my fit")
rate = TabulatedRate(field, rate_values, citation="TDSE run 12")
```

The solprop names are illustrative; the optmat calls are its public surface [32].
Overrides go through optmat's explicit-parameter interface (G5), so the resulting model
still knows what it is, and solprop keeps no override dictionary of its own. Supplied
objects carry their own citation, or are tagged unreproducible under the existing
policy.

**Carrier wavelength.** The high-level driver takes the carrier from the grid
reference. An explicit setting must agree with it, or be a deliberately named
approximation. The same carrier sets the Kerr coefficient, the pump dispersion of
the Raman strengths and the rate model. `Species` no longer binds one universal
`gamma3`.

**Removals.** `ADK`, `PPT` and `IsotropicMOADK` are removed, together with the
barrier-suppression dictionaries and occupancy defaults. `OptmatRate` is a proposed
name for the adapter. It satisfies the existing rate protocol, carries its optmat
identity, and is what `plasma=True` resolves to. The existing `RateTable`, rate
conventions and user-supplied rate objects remain. Independent reference rates
written for tests, such as the envelope-plasma oracles, are test code rather than
runtime rates, and may remain.

**Solid laws.** Old source names map to an optmat model only where the mapping is
scientifically accurate and documented. For example, `fused_silica(source="malitson")`
becomes `malitson_1965/sellmeier`, now absolute. Any other old name gives an
actionable error. There is no `legacy=True` database.

**GNLSE.** With Raman on, `raman_fraction=None` resolves to the resolved silica
bundle's $f_R$ (D4) in place of the inline 0.18. With Raman off it still resolves
to 0. An explicit `raman_fraction` is held back from Kerr either way, as the
[GNLSE manual](../../../guide/waveguides/gnlse.md) documents, and it is recorded, as is an
explicit Raman response. $n_2$ or
$\gamma$ remain required inputs, because a fibre's $n_2$ depends on its composition;
`n2="recommended"` explicitly takes `kerr/SiO2` at the carrier.

**Supplied data.** A supplied law must never be matched to a built-in gas or solid by
name, and so acquire its corrections. A supplied per-particle susceptibility keeps
its dilute meaning.

**Errors and warnings.** optmat errors are translated into solprop's exception
hierarchy with the species or material, quantity, requested state or band, and model
identity. Range, absorption, provisional-model and approximation warnings keep their
meaning and are emitted once during construction or translation. Validate the band
actually evaluated, **including collars**, not the padded FFT array or the user's
nominal band. Extrapolation needs a recorded policy, and is not silenced by blanket
warning suppression.

The policy for qualified intervals (§3.1) is this:

- a band outside a model's intervals is still evaluated;
- optmat's `ModelRangeWarning` becomes one `MaterialRangeWarning` per material and
  translation, naming the parts of the band that lie outside and the reasons of any
  gaps they fall in;
- the manifest records those parts.

The interactive `solprop.sol` session keeps its existing choice not to display
`MaterialRangeWarning`. That is a display policy only; the manifest records the
same band either way.

**Statuses.** A resolved model whose status is `provisional` or `untested` warns once
per run and is recorded. At optmat 0.1.0 these are:

- SF6's rate;
- N2O's Raman bands, when admitted;
- CH4's viscosity in a gas-flow run.

`comparison` and `user` models, the named literature forms and overrides, are recorded
without a warning, since the user chose them.

**Scans.** Scans resolve selections at each physical parameter point, freeze model
identities for a run and include them in cache keys. Model selection must not depend
on import order, available backends or mutable global defaults.

### 6. Implementation strategies

#### Recommended ownership boundary

Use optmat's **public models, responses and records** through thin adapters in
solprop's physics layer. Do not vendor its coefficient files or copy its formulas.
The dependency direction is `solprop -> optmat`, and optmat never imports solprop.
An adapter converts units and conventions and maps onto existing solprop types. If
it starts to reimplement physics, the missing piece belongs upstream (D6).

Translation samples optmat's models into solprop's existing tables: gas and solid
index tables, `RateTable`s, and Raman rows with decay-rate paths. No optmat object,
file loader, Python callback or device array crosses the kernel seam. NumPy remains
the oracle, and Numba and Kokkos receive the same arrays. Kokkos acquires no optmat
dependency and no species constants.

#### Contract and schema effects

The audited kernel schema is 15 and the output schema is 7.

- **Kernel schema: unchanged.** More Raman terms, new coefficient values and new
  sources of existing arrays are within the current contract. If implementation finds
  a backend that assumes a small fixed term count, that is a bug to fix, not a
  contract change. SEP 10 and SEP 11 carry their own schema decisions.
- **Output schema increment: required.** The run records a material manifest (§9.3),
  and resume and scan identity use it. Package versions currently sit outside
  `RunMetadata.configuration_hash()`, so adding optmat to `package_versions()` alone
  does not protect a resumed run.
- **Rate table schema:** unchanged, unless the table itself must carry the model
  identity.

#### Cost and dependencies

Require `optmat[gas]`. During development and until solprop's own release, a pinned
optmat commit in the lockfile is acceptable. Implementation pins the tag `v0.1.0`
through a uv git source on a local clone of optmat. The tag fixes the commit whatever
state that clone's checkout is in. The source is for development only, and is
replaced by a published one before solprop itself is pushed or released. A released
solprop requires a tagged
optmat release, a minimum version with the agreed contracts, and a tested
compatibility policy. A direct pyfluids dependency may stay while thermophysical
calls migrate, and goes once no direct use remains. No other new dependency is
proposed.

The costs:

- **Translation.** It gains model construction and complex solid dispersion. optmat
  evaluates about $10^5$ rate samples in 0.1–0.4 s, so a rate table costs a fraction
  of a second per channel.
- **Stepping.** Unchanged at constant density. On a density gradient, the Raman
  rebuild scales with the number of damping groups (§4.4).
- **Import.** optmat and its SciPy use add import time; measure it.

Measure translation time, peak memory, Raman term counts and recomputations, import
time and a roughly ten-step propagation.

### 7. Verification and validation

Separate three questions:

- Did the adapter reproduce the selected optmat model?
- Did solprop solve its declared equations accurately?
- How did changing the material model change the result?

Agreement with the old arrays answers none of these by itself.

| Check | Reference and invariant | Tolerance basis |
|---|---|---|
| Units and state | Pa/bar and wavelength/frequency conversions; exact vacuum; explicit temperature; density–pressure round trips | Floating-point propagation plus the EOS inversion stopping error |
| Gas linear adapters | $\chi=N\alpha/\varepsilon_0$ against optmat's `complex_polarizability` for every species; the dilute-error warning against optmat's `refractivity`; range edges; absorbing examples | Roundoff for direct evaluation; separately measured interpolation error for translated tables |
| Solid adapters | Direct optmat index and derivative for every solid and axis at 110, 120, 300, 800 and 1000 nm and 5 µm, where in range, and at range edges; axis refusal; metal end-value warnings | Roundoff; the ω↔λ chain rule exact |
| Derivatives | Upstream SI derivatives against independent multi-step finite differences away from poles, including complex silica and metals | Truncation/roundoff balance across step sizes; one finite difference never certifies a rough derivative |
| Mixtures | One component equals the pure gas; splitting or reordering labels changes no material result; vacuum and dilute limits; named air compositions | Summation bounds |
| Transport | optmat viscosity, heat capacities and mixture rule against the current solprop values where the sources agree; flow regime checks unchanged | Source uncertainty for physics; roundoff for identical sources |
| Electronic coefficient | Taylor factor of six; exact $n_2$/$\chi^{(3)}$ round trip with $n_0$; single-tone cubic polarisation; known nonlinear phase in a short fixed-intensity test | Algebraic roundoff, then step convergence |
| Raman adapter | Summed terms against direct optmat `response` at the run's densities; causality; $\Gamma=\pi\Delta\nu$; density scaling; mixture equals composition response; the group mapping exact for unshifted gases | Summation bounds; separate time-window convergence |
| H2 shift | Constant density exact; varying density refused by default; the reference-density approximation's recorded phase bound against direct evaluation | The stated bound, measured |
| Raman tensor | Isotropic and rank-two terms against Hellwarth $a$ and $b$ for S and Q lines; copolarised and cross-polarised probes | Algebraic identities |
| Silica bundle | Unit integral of each $h_R$; $f_R$ and $n_2$ from the bundle; the $n_0^2$ ratio to optmat's $K$ and $\chi_e$; the long-pulse split | Independent integrals; exact ratio |
| Rate adapter | `RateTable` against direct `instantaneous_rate`; envelope quadrature at linear polarisation against `cycle_averaged_rate`; threshold and energy-loss identity; the constant energy against optmat's `energy_per_event` at the peak field, and its recorded bound against direct evaluation over the tabulated fields; maximum-field diagnostic | Table tolerance; optmat's stated averaging consistency; the recorded bound |
| Pre-deletion parity | Before deletion, the named literature forms against solprop's molecular `ADK`, `PPT` and `IsotropicMOADK` with identical inputs; for atomic `PPT` the difference from `ppt_ilkov` is measured and reported as a physics change | $10^{-8}$–$10^{-10}$ for the molecular forms; a failure there blocks deletion |
| Overrides | An override changes the model identity and the configuration hash; a bring-your-own object for each material type runs end to end | Exact identity behaviour |
| Loss and signs | Bulk Beer–Lambert attenuation, nonnegative absorption, pure linear propagation; complex wall-model derivatives for silica and metal walls | Analytic exponential and roundoff/step bounds; no double counting |
| Output and scans | Same identity resumes; changed identity cannot resume or reuse a scan cache; older files stay readable | Exact identity and hash behaviour |
| Backend parity | One linear application, one nonlinear evaluation, one step, eight steps, each statistic and a complete *short* run, with many Raman terms and a density gradient | Existing operation-count and summation bounds; tolerances fixed before examining failures |

The acceptance matrix covers:

- constant and varying density; mixtures with and without Raman and plasma;
- full and reduced capillaries with silica and metal walls; multimode and supported
  free-space routes;
- bulk windows of every solid and axis, prism pairs and compressors;
- step-index and general-modal material regions;
- scalar and vector polarisation; field and envelope formulations;
- single and batch execution.

It also exercises the refusal paths:

- unsupported species, partner or correction;
- a birefringent solid without an axis;
- varying composition with Raman, a Raman mixture with a partner optmat has not
  curated (D16), and H2 on a gradient without the named approximation;
- a vector run on a copolarised-only silica shape;
- a condensed gas state;
- lossy input to a lossless-only provider;
- an out-of-range carrier or band;
- incompatible source aliases.

Current field/envelope restrictions are preserved, including the refusal of envelope
plasma with THG.

Physical validation uses optmat's primary datasets and its own independent checks:

- gas refractometry and dispersion;
- nonlinear measurements with their calibration conventions;
- Raman gains, O2 revivals and silica's absolute gain;
- all-electron and time-dependent rates and measured yields;
- solid refractometry in the transparent window, and silica's absorption bands.

Parity between optmat and the adapter is not independent evidence for optmat's
physics. The unresolved H2/D2 gain excess and the disagreement between measured
yields are recorded, not removed by a wider regression tolerance.

Before each implementation phase merges, run `uv run poe check`,
`uv run poe test-slow` and `uv run poe cov` on the integrated tree. Phases 1a–1d
develop and test on `numpy_ref`. Phase 2 then qualifies `numba_cpu` and a freshly
built Kokkos CPU backend, with its own gate, before Phase 4 switches the defaults
(D18). GPU qualification is separate and follows the standing device policy; CPU
success is not a GPU result. Keep tests and benchmarks to about ten accepted steps,
or a propagation that finishes in seconds. Only gallery problems run at full length.

### 8. Documentation

Expand the materials manual into the user-facing ownership and selection guide. It
covers:

- recommended models and material identities;
- reference state versus operating state; SI conversions;
- the support matrix of §3.1, including axes and metals;
- uncertainty and provisional flags; the dilute-scaling error;
- the quick-experiment paths of §5.

Rewrite these manuals:

- the ionisation manual, around optmat's rate models: their effective electrons,
  thresholds, maximum fields and uncertainty factors, and the removal of solprop's
  formulas;
- the Raman manual, for optmat's responses, the damping-group mapping, H2's shift
  rule and the range warnings, including N2 and air above 2.09 amagat.

Update these where their behaviour changes: the mixture, GNLSE (the $f_R$ default),
modes, general-modal, free-space, compression and design-tools, linear-operator,
output and scan manuals. The installation guide, `README.md`, `docs/introduction/first-run.md`, the
dependency and design pages and the docstrings must describe the installed
dependency and the public names.

Add a short tutorial that compares resolved material choices, and runs a short
propagation before and after a *declared* material change. Include a table that
separates the changes in dispersion, electronic coefficient, Raman response,
ionisation rate and solid index. Historical results come from archived artifacts or
from the old environment; the tutorial must not rebuild the old database.

Rerun the affected worked examples through `poe docs-examples`, and commit the
generated outputs with their sources and measured runtime. Update the Luna
differences page. Link the completed SEP from the manual and cite it in the final
changelog entry.

### 9. Compatibility and impact

#### 9.1 Changes measured during this audit

These are **material evaluations**, not propagation results or independent accuracy
assessments. They were computed at the first audit's revisions by calling each
package's public functions. The solprop side of Phase 0 re-measures them at optmat
0.1.0 and reports the result in Part II. The figures quoted in §3.1, §4.4 and §4.6 were
evaluated at 0.1.0. Percent changes are $100(\mathrm{optmat}/\mathrm{solprop}-1)$.

**Gases, at 800 nm, 1 bar and 294 K.** The inputs were solprop's default
`refractive_index_minus_one`, `gamma3`, `density` and `ionisation_potential`, against
optmat's `default_model(...).refractivity`,
`HyperpolarizabilityModel.gamma3("kerr" | "thg", ...)`, `number_density` and
`ionisation_parameters(...).ionization_energy`. Each package uses its own EOS. The
coefficient is per particle, so its comparison does not depend on density. The
refractivity column includes optmat's finite-density law; at 1 bar that is below
0.06 % (§4.2).

| Gas | n − 1 (%) | Carrier Kerr γ3 (%) | Number density (ppm) | Adiabatic threshold (meV) | optmat THG/Kerr γ |
|---|---:|---:|---:|---:|---:|
| He | −0.196 | +1.98 | −0.34 | −0.82 | 1.060 |
| Ne | −0.299 | +41.55 | −0.34 | −0.48 | 1.075 |
| Ar | −0.050 | +23.13 | +5.70 | −1.22 | 1.177 |
| Kr | −0.039 | +3.00 | +1.13 | +7.51 | 1.256 |
| Xe | −0.200 | −4.49 | +1.13 | −0.99 | 1.388 |
| H2 | +0.060 | +8.53 | +1.13 | −0.33 | 1.200 |
| D2 | −1.210 | +5.57 | −0.06 | −0.49 | 1.196 |
| N2 | −0.032 | +7.47 | +5.70 | −0.52 | 1.163 |
| O2 | +0.011 | +7.34 | −14.75 | −0.08 | 1.260 |
| CH4 | −0.313 | +7.16 | +5.70 | −5.20 | 1.294 |
| SF6 | +0.195 | +5.80 | +1.13 | +30.00 | 1.124 |
| N2O | +0.418 | −4.87 | +1.13 | −12.20 | 1.213 |

Some old fits warn at 800 nm (CH4, SF6, N2O), so those rows compare extrapolated old
behaviour. Refractivity changes do not predict derivative changes. The THG/Kerr
column shows the remaining single-coefficient limitation of `carrier-kerr`; N2O's
THG evaluation warns and is shown only for completeness. The molecular default
*rates* use the vertical threshold, not the adiabatic one tabulated here.

**Solids, refractive index.** The table gives the real index difference
optmat − solprop between optmat's recommended models and solprop's current defaults;
the extinction coefficient is compared where it matters.

| Material (solprop source) | 300 nm | 1 µm | 5 µm | Notes |
|---|---:|---:|---:|---|
| Fused silica (`malitson+franta`) | +4.48e−4 | +3.91e−4 | −2.28e−3 | At 120 nm: 2.940 → 2.240; at 110 nm: 0.000 → 1.805 (old formula's pole) |
| CaF2 (`daimon`) | +4.23e−4 | +3.93e−4 | +3.83e−4 | 5 µm is beyond both models' recommended range |
| MgF2 o (`dodge-o`) | +4.13e−4 | +3.73e−4 | +3.82e−4 | e-ray newly available: 1.38557 at 1 µm |
| BaF2 (`malitson`) | +4.45e−4 | +3.91e−4 | +3.91e−4 | |
| Sapphire o (`malitson-o`) | +5.59e−4 | +4.78e−4 | +5.16e−4 | Same formulas, now absolute; e-ray newly available: 1.74828 at 1 µm |
| N-BK7 (`schott`) | +4.76e−4 | +4.10e−4 | — | Both lossless; 5 µm outside range |
| Silver (stitched) | +2.95e−3 | −2.2e−5 | +1.99e−3 | k at 120 nm: 0.883 → 0.568 (−36 %) |
| Aluminium (stitched) | +7.06e−2 | +0.226 | −0.212 | k at 800 nm: 7.52 → 8.45 (+12 %); at 1 µm: 8.51 → 9.58 |

For the crystals and N-BK7, the transparent-window offsets are the absolute
correction; for silica they are the refit to absolute refractometry. They are nearly
constant across the window, so they change the phase index and Fresnel coefficients
far more than the GVD; Phase 4 reports the GVD changes. The metal differences come
from different stitched sources and interpolants [16], and matter for metal-walled
guides at those wavelengths.

**Silica nonlinear response.** `kerr/SiO2` gives $n_2=2.59\times10^{-20}$ m²/W at
800 nm, $2.53\times10^{-20}$ at 1030 nm and $2.47\times10^{-20}$ at 1550 nm, with
electronic parts 2.02, 1.97 and $1.93\times10^{-20}$ m²/W. For a GNLSE with a
user-supplied $n_2$, Raman on and `raman_fraction` left out, D4 changes $f_R$ from
0.18 to 0.219:

- the instantaneous electronic share falls by 4.8 %;
- the delayed share rises by 21.6 %;
- the long-pulse total is unchanged by construction.

With Raman off, or with an explicit fraction, nothing changes.

The paired default reproduces silica's measured absolute Raman gain to −10 % [17].
The tensor default of D11 reproduces it to −7.9 %; the Lin–Agrawal shape scaled to the
fitted $f_R$ would have been −17.8 % ([30], §4.7).

**Ionisation rates, at 800 nm, linear polarisation, instantaneous.** The table gives
the ratio of optmat's recommended rate to solprop's current default at the field
amplitude $F=\sqrt{2I/(\varepsilon_0c)}$ of peak intensity $I$:

| Gas | solprop default | $5\times10^{13}$ W/cm² | $10^{14}$ | $3\times10^{14}$ |
|---|---|---:|---:|---:|
| He | PPT | 2.15 | 1.67 | 1.30 |
| Ne | PPT | 0.77 | 0.61 | 0.51 |
| Ar | PPT | 2.54 | 2.45 | 3.54 |
| Kr | PPT | 1.40 | 1.03 | 0.76 |
| Xe | PPT | 1.76 | 1.27 | 0.99 |
| H2 | Isotropic MO-ADK | 25.2 | 3.28 | 0.86 |
| D2 | Isotropic MO-ADK | 27.6 | 3.51 | 0.89 |
| N2 | Isotropic MO-ADK | 26.5 | 2.87 | 0.43 |
| O2 | Isotropic MO-ADK | 2.29 | 0.46 | 0.06* |
| CH4 | ADK | 4.95 | 1.37 | 0.40 |
| SF6 | ADK | 855 | 126 | 30.2 |
| N2O | ADK | 5.51 | 0.97 | 0.16* |

\* Above optmat's declared maximum field for that model. CH4 and Xe at
$3\times10^{14}$ W/cm² are at their limits.

These are large, non-uniform changes. They combine several factors:

- effective electron numbers;
- the non-adiabatic PPT factor at Keldysh parameters near one, which dominates the
  molecular ratios at low intensity;
- barrier-suppression corrections at high intensity;
- vertical molecular thresholds;
- the removed Stark and dipole factors.

Phase 4 attributes them. SF6's default is an untested prediction with uncertainty
factor 10, so its ratio shows the size of the unknown, not an improvement. Plasma
densities, loss, spectra and ionisation-driven dynamics will change substantially in
molecular gases.

**Raman responses.** optmat compares its responses with solprop's at the audited
solprop revision [3], at 295 K and 1 amagat. "Early" is the least-squares scale over
the first 100 fs.

| Gas | Part | Early scale | At 20 ps |
|---|---|---:|---:|
| N2 | rotation | 1.000 | 0.958 |
| N2 | vibration | 1.033 | 5.47 |
| O2 | rotation | 1.000 | 1.02 |
| O2 | vibration | 1.081 | 0.155 |
| H2 | rotation | 1.101 | 1.24 |
| H2 | vibration | 0.897 | 0.688 |
| D2 | rotation | 0.981 | 1.21 |
| D2 | vibration | 0.772 | 0.374 |
| CH4 | vibration | 1.116 | 1.07 |
| SF6 | vibration | 1.124 | 1.14 |

The vibrational early scales are optmat's strength ratios times the anisotropic Q
share, which solprop's single vibrational line omits. Per-line widths dominate the
late-time ratios. For example, N2's resolved Q branch first decays like solprop's
6 ps envelope and then persists at about a tenth for about 100 ps, whereas O2's
resolved widths decay faster. optmat's H2/D2 gains still exceed the measured gains by
11–39 %; that limitation carries forward and is not tuned away. Phase 1c re-measures
this table in solprop.

#### 9.2 Controlled migration of numerical results

1. **Freeze the baseline before changing defaults.** Record the solprop, optmat and
   backend revisions and builds, the lockfile, resolved configurations, seeds, grids
   and windows. Capture short representative witnesses and existing gallery outputs.
2. **Test conversion without changing physics where possible.** With identical
   explicitly supplied data, verify the old and new assembly paths. That means
   identical rates in the rate table, identical Raman lines on old and new terms,
   and identical index laws through the solid adapters. Use synthetic inputs or
   immutable test artifacts, and build no historical-material runtime option.
3. **Attribute changes.** Compare old and new models one family at a time: EOS;
   gas index; solid indices; electronic coefficient; silica bundle; Raman response;
   ionisation rate. Then evaluate their combined change at fixed experimental inputs.
4. **Separate physical and numerical differences.** Use fixed short step schedules
   and refinement to measure numerical errors, then adaptive runs to expose the
   changed trajectory. Do not require identical step histories, or move the input
   pressure or energy to restore a familiar plot.
5. **Review each changed regression.** Keep conservation, convergence and
   backend-parity tolerances independently justified. Replace material-dependent
   golden outputs only with a report of the model change and independent checks.
   Old Luna witness arrays remain historical comparisons, not physical truth.
6. **Switch defaults once qualified, and remove duplicates.** Announce the scientific
   behaviour change, and provide the comparison report and updated examples. Delete
   the migrated tables, coefficient sets, rate formulas and fallback selectors. Future
   optmat changes to defaults or data undergo the same review, rather than arriving
   as an unnoticed lock bump.

#### 9.3 Provenance and resume

Each run records a material manifest:

- the optmat version and commit;
- for every resolved model and response, its optmat identity (G4) and any overrides;
- species, compositions, solid materials and axes;
- the policies in force: dilute scaling and its measured error, `carrier-kerr` and
  its carrier, the Raman reference temperature and any `reference-density` shift, the
  rate threshold kind, and the energy-loss convention with each species' constant
  energy and its bound;
- the GNLSE's held-back Raman fraction and its origin: the bundle, the user, or 0
  with Raman off;
- all user-supplied objects' citations.

The configuration hash covers the manifest's identities and policies, not the string
`recommended`. The optmat version is recorded as its full version string; for an
editable install that string is fixed at installation, so the identities, not the
version, are what distinguish two optmat trees. Automatic resume and scan-cache reuse are refused when any identity,
policy or the optmat version differs. The version test is deliberately conservative:
it can refuse a resume that would have been safe, but never accepts one that is not.
Older output files remain readable. A file without a manifest is refused for
automatic resume with an explanation; reproduce or continue it in its recorded old
environment.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Copy optmat coefficient files into solprop | Recreates the maintenance problem and leaves two authorities; rejected |
| Make optmat optional, with current tables as fallback | Identical scripts would acquire different physics depending on installation; rejected |
| One SEP containing the data migration, low-rank Raman, the finite-density law and full digest provenance | The first revision's plan. It bundled three kernel- or physics-level changes with a data swap and made the whole depend on the hardest part; split into SEPs 3, 10 and 11 |
| Low-rank Raman kernels in this SEP | Removes the H2 restriction and the rebuild cost, but changes the kernel contract in three backends; [SEP 10](../sep-0010-low-rank-raman.md) |
| Finite-density law in this SEP | Changes solprop's linear-density fast paths, mode freezing and expansions; the error it removes is below 0.06 % at 1 bar; [SEP 11](../sep-0011-finite-density-gas-index.md) |
| Keep solprop's rate formulas and take only optmat's parameters | Duplicates a validated engine and keeps occupancy and correction policy in two places; rejected (D1) |
| Keep solprop's formulas for custom species only | Two rate engines with different defaults; rejected in favour of optmat's explicit-parameter constructor (D1, G5) |
| Import optmat's gas $n_2$, or its silica $K$ and $\chi_e$ | Mixes optmat's dilute or $n_0^2$-inclusive conventions with solprop's normalisation; import $\gamma_3$, or $n_2$, $f_R$ and $h_R$; rejected |
| Call optmat from every propagation stage or backend | Breaks the seam, cache discipline and device execution; sample at translation instead |
| Content digests as the only resume identity from the start | Preferable, and adopted once optmat delivers them (G4); until then the optmat version is the conservative identity |
| Freeze a legacy solprop dataset in optmat behind a switch | Possible, but old environments reproduce old results instead |
| Move the mirror curves to optmat now | Device data with no material record; deferred (D3) |

### 11. Phases

Every phase is independently reviewable and mergeable with the prescribed gate. Use
`feature/sep-0003-optmat-materials-<phase>` branches. Each phase report records:

- both repositories' changes where relevant;
- deviations from this proposal;
- sources checked;
- the qualification actually run;
- a handoff under [SEP 0](../sep-0000-process.md).

#### Phase 0 — optmat preparation

In optmat, through its own review process, close G1–G9 and produce a pinned commit
(G10 before solprop's release). The work is specified as an optmat work package
covering:

- public contracts: API stability, a tested import surface, SI derivatives, the EOS
  inverse and transport;
- identities and capability flags;
- explicit-parameter constructors and overrides;
- the Raman group view and zero-density limit;
- rate thresholds and energy-loss channels;
- the silica bundles;
- named compositions and the `Gas.Air` model;
- refreshed solprop parity fixtures.

The optmat side is complete: OEP 1 [30] and OEP 4 [31] are Final at optmat 0.1.0
(§2, §3.2).

In solprop, this phase:

- adds the pinned dependency (§6);
- records the accepted contract and the baseline witnesses: short representative runs
  on the current defaults, with their revisions, grids and seeds, as immutable test
  artifacts (§9.2, step 1);
- re-measures §9.1 at the pinned tag.

This phase changes no solprop default. The design decisions it needed were taken on
2026-09-29 (D8–D11 here, and the rest within the optmat work package) and on
2026-10-01 (D16–D18).

#### Phase 1a — Dependency seams, manifest, gases and solids (NumPy)

- Add `MaterialSelection` and the manifest, with its output schema increment, and the
  private bridge that translates optmat's errors and warnings (§5).
- Route gas linear laws through optmat's polarizability at linear density scaling,
  with the dilute-error warning.
- Delegate the EOS, its inverse and transport.
- Resolve every built-in solid, with explicit axes, and the metal tables.
- Deliver the quick-experiment paths, the tests and the materials manual.

Curated selections are explicit until Phase 4a switches the defaults. Because the
kernel arrays are unchanged, an explicit selection runs on every backend (D18).

Phases 1b, 1c and 1d build on 1a's seams and touch separate modules, so they may run in
parallel. They are merged in the order 1b, 1d, 1c.

#### Phase 1b — Electronic and silica bundle (NumPy)

- Bind the gas Kerr coefficient at the carrier.
- Resolve the silica bundle, and replace the GNLSE's inline 0.18 with the bundle's
  $f_R$ (D4). Test that `None` with Raman off resolves to 0, and that an explicit
  fraction is held back whatever `raman` is.
- Add the $n_0^2$ convention test, and the tensor bundle with its channel map (§4.5).

#### Phase 1c — Gas Raman responses (NumPy)

- Map optmat's responses onto terms by damping group and tensor family, with the
  summed-density decay path of a composition.
- Add the pump wavelength, the reference temperature, the zero-density limit, the range
  warnings, the composition and partner refusals (D16) and the H2 shift rule.
- Record term counts and rebuild cost, and rewrite the Raman manual.

#### Phase 1d — Rate adapters (NumPy)

- Implement `OptmatRate`. It covers the threshold, the table, the envelope quadrature
  check, the energy-loss binding with its reference field and bound, the handling of
  maximum-field warnings, the post-run diagnostic and the manifest entries.
- Measure pre-deletion parity, and rewrite the ionisation manual.

#### Phase 2 — Backend qualification (Numba CPU and Kokkos CPU)

- Qualify the translated problems on `numba_cpu` and on a freshly built Kokkos CPU
  backend against `numpy_ref`. They include many Raman terms on a density gradient,
  mixtures, scalar and vector field and envelope cases, rate tables, batch runs and
  statistics.
- No native change is expected, since the contract is unchanged. Report any backend
  assumption about term counts as a bug, and fix it.
- Record numerical parity, memory and short-window costs.
- Run the full gate, the slow and coverage lanes, and the Kokkos lane.
- GPU qualification remains a separate lane.

The former separate Kokkos phase is merged into this one (D18).

#### Phase 4a — Curated defaults and the changed-results report

- Switch built-in defaults, now that their backend matrix is qualified.
- Complete §9's attributed comparisons against the Phase 0 witnesses, including the
  solids' GVD changes and the attribution of the rate ratios.
- Update expected outputs on scientific grounds.

#### Phase 4b — Deletion

- Delete from solprop:
  - gas and solid coefficients, and the silica and metal tables with their
    generators;
  - `raman_data.py`, `ionisation_data.py` and the transport tables;
  - `ADK`, `PPT` and `IsotropicMOADK`, with their correction tables and occupancy
    defaults;
  - the legacy source names, mapped where accurate (§5);
  - unneeded direct dependencies.
- Audit the whole package, not only `physdata`: driver defaults, GNLSE, rate
  selection and air recipes all count.

#### Phase 4c — Documentation and examples

- Complete the manuals and the tutorial of §8.
- Regenerate the affected galleries and examples, and publish the compatibility notes.

Tests may keep explicitly historical witnesses, and examples may contain
user-declared experimental inputs, but neither may become an alternative built-in
registry.

Final requires:

- the numerical gates, the public manuals and the source report;
- every gas, solid, response and rate capability migrated, or explicitly refused
  with accepted reasons;
- the follow-ups of §2 confirmed current on the roadmap (D14).

### 12. Open questions

None remain. The former questions were settled as D8–D15 on 2026-09-29. The
re-review against optmat 0.1.0 on 2026-10-01 was settled as D16–D18 and by revisions
of §3–§7 and §11. Scientific
questions still open upstream are optmat's roadmap items, and they are stated as
limitations in §4 and §9. They include the disagreement between the two measured
yield sets, line mixing, and the elliptic rate theory.

### 13. References

The optmat references below identify the dependency's material contract and
curation reports. Links pin the tag `v0.1.0` in optmat's declared repository. That
tag was not yet published at acceptance. These versioned reports must be published, or
a durable public archive provided, before the migration is qualified for solprop's
release (G10).

1. optmat, [Conventions](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/conventions.md),
   [Gas models](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/gas_models.md)
   and [Accuracy](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/accuracy.md).
2. optmat, [Third-order nonlinearity](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear.md)
   and [Third-order accuracy](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/nonlinear_accuracy.md).
3. optmat, [Raman response functions](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman_response.md),
   including its comparison with solprop, the cost along a density gradient and the
   measured ranks; and [Raman lines](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/raman.md).
4. optmat, [Ionisation rates](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/ionisation_rates.md)
   and [Ionisation parameters](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/ionisation.md).
5. optmat, [Molecular ionisation rates](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/molecular_ionisation_rates.md),
   including the parity of its literature forms with solprop's rates.
6. optmat, [Solids](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/solid.md),
   [Window crystals](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/crystals.md)
   and [Fused silica](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/silica.md).
7. optmat, [Metals](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/metals.md).
8. optmat, [Review register](https://github.com/LupoLab/optmat/blob/v0.8.0/REGISTER.md)
   and [Roadmap](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/roadmap.md).
9. D. Franta, D. Nečas, I. Ohlídal and A. Giglia, "Optical characterization of SiO2
   thin films using universal dispersion model over wide spectral range,"
   *Proc. SPIE* **9890**, 989014 (2016),
   [doi:10.1117/12.2227580](https://doi.org/10.1117/12.2227580).
10. I. H. Malitson, "Interspecimen comparison of the refractive index of fused
    silica," *J. Opt. Soc. Am.* **55**, 1205 (1965),
    [doi:10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205);
    D. B. Leviton and B. J. Frey, "Temperature-dependent absolute refractive index
    measurements of synthetic fused silica" (2008),
    [doi:10.48550/arXiv.0805.0091](https://doi.org/10.48550/arXiv.0805.0091).
11. M. Daimon and A. Masumura, "High-accuracy measurements of the refractive index
    and its temperature coefficient of calcium fluoride in a wide wavelength range
    from 138 to 2326 nm," *Appl. Opt.* **41**, 5275 (2002),
    [doi:10.1364/AO.41.005275](https://doi.org/10.1364/AO.41.005275).
12. M. J. Dodge, "Refractive properties of magnesium fluoride," *Appl. Opt.* **23**,
    1980 (1984), [doi:10.1364/AO.23.001980](https://doi.org/10.1364/AO.23.001980).
13. I. H. Malitson, "Refractive properties of barium fluoride," *J. Opt. Soc. Am.*
    **54**, 628 (1964),
    [doi:10.1364/JOSA.54.000628](https://doi.org/10.1364/JOSA.54.000628).
14. M. J. Weber, *Handbook of Optical Materials* (CRC Press, 2003), §1.3.4,
    reprinting Malitson and Dodge's (1972) sapphire formulas; checked against
    I. H. Malitson, "Refraction and dispersion of synthetic sapphire,"
    *J. Opt. Soc. Am.* **52**, 1377 (1962),
    [doi:10.1364/JOSA.52.001377](https://doi.org/10.1364/JOSA.52.001377).
15. SCHOTT AG, data sheet N-BK7 517642.251 (December 2023), with Technical
    Information TIE-29, "Refractive index and dispersion" (August 2023).
16. S. Babar and J. H. Weaver, "Optical constants of Cu, Ag, and Au revisited,"
    *Appl. Opt.* **54**, 477 (2015),
    [doi:10.1364/AO.54.000477](https://doi.org/10.1364/AO.54.000477);
    A. D. Rakić, "Algorithm for the determination of intrinsic optical constants of
    metal films: application to aluminum," *Appl. Opt.* **34**, 4755 (1995),
    [doi:10.1364/AO.34.004755](https://doi.org/10.1364/AO.34.004755);
    K. M. McPeak et al., "Plasmonic films can easily be better: rules and recipes,"
    *ACS Photonics* **2**, 326 (2015),
    [doi:10.1021/ph5004237](https://doi.org/10.1021/ph5004237);
    M. A. Ordal et al., "Optical properties of the metals Al, Co, Cu, Au, Fe, Pb, Ni,
    Pd, Pt, Ag, Ti, and W in the infrared and far infrared," *Appl. Opt.* **22**, 1099
    (1983), [doi:10.1364/AO.22.001099](https://doi.org/10.1364/AO.22.001099).
    The full stitching sources are listed in [7].
17. R. H. Stolen and E. P. Ippen, "Raman gain in glass optical waveguides,"
    *Appl. Phys. Lett.* **22**, 276 (1973),
    [doi:10.1063/1.1654637](https://doi.org/10.1063/1.1654637);
    for the nuclear share, R. Hellwarth, J. Cherlow and T.-T. Yang, *Phys. Rev. B*
    **11**, 964 (1975),
    [doi:10.1103/PhysRevB.11.964](https://doi.org/10.1103/PhysRevB.11.964).
18. D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for
    fiber-optic Raman gain spectrum and response function," *J. Opt. Soc. Am. B*
    **19**, 2886 (2002), equation (9) and Table 1 including its footnote,
    [doi:10.1364/JOSAB.19.002886](https://doi.org/10.1364/JOSAB.19.002886).
19. Q. Lin and G. P. Agrawal, "Raman response function for silica fibers,"
    *Opt. Lett.* **31**, 3086 (2006),
    [doi:10.1364/OL.31.003086](https://doi.org/10.1364/OL.31.003086);
    K. J. Blow and D. Wood, "Theoretical description of transient stimulated Raman
    scattering in optical fibers," *IEEE J. Quantum Electron.* **25**, 2665 (1989),
    [doi:10.1109/3.40655](https://doi.org/10.1109/3.40655).
20. X. M. Tong, Z. X. Zhao and C. D. Lin, *Phys. Rev. A* **66**, 033402 (2002),
    [doi:10.1103/PhysRevA.66.033402](https://doi.org/10.1103/PhysRevA.66.033402);
    S. F. Zhao et al., *Phys. Rev. A* **81**, 033423 (2010),
    [doi:10.1103/PhysRevA.81.033423](https://doi.org/10.1103/PhysRevA.81.033423),
    with [erratum](https://doi.org/10.1103/PhysRevA.82.049903);
    S.-F. Zhao et al., *J. Phys. B* **44**, 035601 (2011),
    [doi:10.1088/0953-4075/44/3/035601](https://doi.org/10.1088/0953-4075/44/3/035601);
    S.-F. Zhao et al., *Commun. Theor. Phys.* **67**, 289 (2017),
    [doi:10.1088/0253-6102/67/3/289](https://doi.org/10.1088/0253-6102/67/3/289).
21. D. P. Shelton and J. E. Rice, "Measurements and calculations of the
    hyperpolarizabilities of atoms and small molecules in the gas phase,"
    *Chem. Rev.* **94**, 3 (1994),
    [doi:10.1021/cr00025a001](https://doi.org/10.1021/cr00025a001);
    D. P. Shelton, *Phys. Rev. A* **42**, 2578 (1990),
    [doi:10.1103/PhysRevA.42.2578](https://doi.org/10.1103/PhysRevA.42.2578).
22. M. L. Huber, *Models for Viscosity, Thermal Conductivity, and Surface Tension of
    Selected Pure Fluids as Implemented in REFPROP v10.0*, NISTIR 8209 (2018),
    [doi:10.6028/NIST.IR.8209](https://doi.org/10.6028/NIST.IR.8209);
    C. R. Wilke, "A viscosity equation for gas mixtures," *J. Chem. Phys.* **18**,
    517 (1950), [doi:10.1063/1.1747673](https://doi.org/10.1063/1.1747673).
23. A. Talebpour, J. Yang and S. L. Chin, "Semi-empirical model for the rate of
    tunnel ionization of N2 and O2 molecule in an intense Ti:sapphire laser pulse,"
    *Opt. Commun.* **163**, 29 (1999),
    [doi:10.1016/S0030-4018(99)00113-3](https://doi.org/10.1016/S0030-4018(99)00113-3);
    F. A. Ilkov, J. E. Decker and S. L. Chin, *J. Phys. B* **25**, 4005 (1992),
    [doi:10.1088/0953-4075/25/19/011](https://doi.org/10.1088/0953-4075/25/19/011);
    X. M. Tong and C. D. Lin, *J. Phys. B* **38**, 2593 (2005),
    [doi:10.1088/0953-4075/38/15/001](https://doi.org/10.1088/0953-4075/38/15/001);
    Q. Zhang, P. Lan and P. Lu, *Phys. Rev. A* **90**, 043410 (2014),
    [doi:10.1103/PhysRevA.90.043410](https://doi.org/10.1103/PhysRevA.90.043410).
24. J. S. Parker, K. J. Meharg, G. A. McKenna and K. T. Taylor, "Single-ionization of
    helium at Ti:sapphire wavelengths: rates and scaling laws," *J. Phys. B* **40**,
    1729 (2007),
    [doi:10.1088/0953-4075/40/10/008](https://doi.org/10.1088/0953-4075/40/10/008).
25. T.-C. Jagau, "Investigating tunnel and above-barrier ionization using
    complex-scaled coupled-cluster theory," *J. Chem. Phys.* **145**, 204115 (2016),
    [doi:10.1063/1.4967961](https://doi.org/10.1063/1.4967961);
    V. P. Majety and A. Scrinzi, "Static field ionization rates for multi-electron
    atoms and small molecules," *J. Phys. B* **48**, 245603 (2015),
    [doi:10.1088/0953-4075/48/24/245603](https://doi.org/10.1088/0953-4075/48/24/245603).
26. W. C. Wallace et al., "Precise and accurate measurements of strong-field
    photoionization and a transferable laser intensity calibration standard,"
    *Phys. Rev. Lett.* **117**, 053001 (2016),
    [doi:10.1103/PhysRevLett.117.053001](https://doi.org/10.1103/PhysRevLett.117.053001).
27. J. K. Wahlstrand, S. Zahedpour, A. Bahl, M. Kolesik and H. M. Milchberg,
    "Bound-electron nonlinearity beyond the ionization threshold,"
    *Phys. Rev. Lett.* **120**, 183901 (2018),
    [doi:10.1103/PhysRevLett.120.183901](https://doi.org/10.1103/PhysRevLett.120.183901).
28. W. K. Bischel and M. J. Dyer, "Temperature dependence of the Raman linewidth and
    line shift for the Q(1) and Q(0) transitions in normal and para-H2,"
    *Phys. Rev. A* **33**, 3113 (1986),
    [doi:10.1103/PhysRevA.33.3113](https://doi.org/10.1103/PhysRevA.33.3113);
    G. C. Herring, M. J. Dyer and W. K. Bischel, "Temperature and density dependence
    of the linewidths and line shifts of the rotational Raman lines in N2 and H2,"
    *Phys. Rev. A* **34**, 1944 (1986),
    [doi:10.1103/PhysRevA.34.1944](https://doi.org/10.1103/PhysRevA.34.1944).
29. A. Picard, R. S. Davis, M. Gläser and K. Fujii, "Revised formula for the density
    of moist air (CIPM-2007)," *Metrologia* **45**, 149 (2008),
    [doi:10.1088/0026-1394/45/2/004](https://doi.org/10.1088/0026-1394/45/2/004).
30. optmat, [OEP 1 — The propagation-code interface](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/archive/oep-0001-propagation-interface.md)
    (Final), with its implementation record; and
    [Identity, status and validity](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/identity.md),
    [Transport](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/transport.md).
31. optmat, [OEP 4 — The refractive index of gas mixtures at finite density](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/oeps/archive/oep-0004-mixture-index.md)
    (Final), and [Mixtures](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/mixtures.md).
32. optmat, [Using optmat from a propagation code](https://github.com/LupoLab/optmat/blob/v0.8.0/docs/propagation.md):
    the public surface, units, identities, statuses and validity rules on which the
    adapters are built.

**Source-check boundary of this Draft.** The repository audit checked the
implementation contracts, public APIs, curated records and reports, and the material
evaluations of §4.2 and §9.1. The bibliographic data of [9–29] were taken from
optmat's curated source records and checked against their DOIs. Hollenbeck and
Cantrell's equation (9) and Table 1 footnote were checked in the paper during the
first audit. This is not a fresh primary-source verification of every optmat
measurement, fit, Raman strength, width law or rate validation; those rely on the
cited optmat curation reports. Phase 0 reports the primary sources checked for new
or moved data, including the transport sources.

## Part II — Implementation record

### Phase 0 — optmat preparation (the solprop side)

- **Scope:** Phase 0 of §11 in solprop, 2026-10-01, implemented by Opus 5.5 (AI) for
  John Travers, on `feature/sep-0003-optmat-materials-0`. The optmat side was complete
  before it began: OEP 1 and OEP 4 are Final in optmat 0.1.0 (§2).
- **Delivered:**
  - `pyproject.toml` and `uv.lock`: the dependency `optmat[gas]>=0.1.0`, with a
    development-only uv git source on a local clone at the tag `v0.1.0`. The lock
    resolves it to `d6945ed8e34fdff848a860a64d9b8597f9471774`, and adds no other package.
    CoolProp stays at 8.0.0, the version both packages already used.
  - `tests/data/generate_sep3_baseline.py`, which writes `tests/data/sep3_baseline.npz`
    and `.json`: the frozen witnesses of §9.2, step 1. There are nine cases, each taking
    ten fixed steps on `numpy_ref` on the current defaults:
    - Ar with plasma, field and envelope;
    - N2 Raman at a constant pressure and on a 1 → 0.2 bar gradient;
    - H2 Raman;
    - an He/Ar mixture;
    - a silver-walled capillary;
    - the GNLSE with Raman on and off and the default fraction.

    The record holds each case's parameters, grid record, configuration hash, warnings
    and runtime, with the solprop commit and the package versions.
  - `tests/data/measure_sep3_changes.py`, which writes `tests/data/sep3_changes.json`:
    §9.1 re-measured at optmat 0.1.0, with the identity digest of every optmat model
    it evaluated and the warnings both packages raised.
  - `tests/test_sep3_baseline.py`, which keeps the record and the arrays consistent.
  - `docs/dev/dependencies.md`, `docs/introduction/installation.md`, `AGENTS.md` (the pin)
    and `CHANGELOG.md`.
- **Deviations from the proposal:**
  - The baseline has no bulk-window run. A window's change is its index, which
    `sep3_changes.json` records at every built-in solid's sample wavelengths, and
    which optmat's own parity fixtures record across each range at solprop
    `d577414`, where `physdata` is identical.
  - The capillary witnesses are 0.4 mm long, with a weak in-stage collar (a reference
    length of 1 m). The default collar of so short a guide is stiff: ten fixed steps
    of 0.2 mm or 40 µm diverged beyond its explicit stability limit of 52 µm and
    10 µm. Steps that fit under it would not be ten.
- **Decisions:**
  - The witnesses run on `numpy_ref` only. Backend parity is Phase 2's, and the
    attribution compares models, not backends.
  - solprop imports optmat lazily, inside the functions that resolve a material, never
    at `import solprop`. A warm import of optmat's registries takes 1.3–1.6 s on the
    development Mac, mostly scipy.signal and scipy.stats through optmat's absorption
    and comparison modules, and CoolProp through its equation of state, against 0.05 s
    for `import solprop`. Phase 1a must keep the existing import-time test.
- **Verification:**
  - The full gate on the branch; the baseline integrity test.
  - The witnesses and the re-measurement were generated on `main` at `eae0ed8`, with
    this branch's dependency installed but called by no material path, so they record
    the old defaults exactly.
  - The re-measurement against §9.1. Every gas, solid and silica entry reproduces the
    first audit's values to their printed precision, except where optmat changed after
    the first audit, as §2 lists:
    - CH4's n − 1 change is −0.308 % (it was −0.313 %), O2's +0.010 % (+0.011 %) and
      Ne's −0.298 % (−0.299 %);
    - the rate ratios of O2 are 4.99, 1.38 and 0.447 at 5 × 10¹³, 10¹⁴ and
      3 × 10¹⁴ W/cm² (they were 2.29, 0.46 and 0.06), the last above optmat's maximum
      field;
    - the rate ratios of CH4 are 1.06, 0.376 and 0.141 (they were 4.95, 1.37 and 0.40),
      none of them above it.
  - optmat's statuses, uncertainty factors and energy kinds are as §3.1 states: SF6's
    rate is `untested` with factor 10; the molecules' `energy_kind` is `vertical`, and
    the atoms' `adiabatic`.
  - N-BK7 at 5 µm, which §9.1 leaves out because it is outside both models' ranges,
    differs by +4.6 × 10⁻⁴ when both are extrapolated there.
  - Not run: no propagation used optmat, since nothing does yet.
- **Sources checked:** none new. Every number is an evaluation of the two packages'
  public functions.
- **Limitations:**
  - The witnesses are short and record the old defaults, warnings included (the gas
    fits' and the silica wall's range warnings, and the gradient's Reynolds number of
    2102). They are a historical record, not a physical reference.
  - The envelope-plasma witness takes about 80 s on the development Mac, and the others
    under 1.2 s each. The generator is run by hand and is not part of the test lanes.
- **Open questions:**
  - Q1. optmat's import cost. Lazy importing in solprop contains it. Making optmat's
    registries import scipy.signal and scipy.stats only where a measured absorption or
    a comparison needs them would remove it, but that is an optmat change, which the
    author decides on optmat's side. *Measured again during Phase 1a:* the 1.3–1.6 s is
    the cost of a bare import. Once solprop's driver modules are loaded, which import
    scipy.signal and scipy.stats themselves, the whole optmat surface adds about 0.1 s.
    So it matters for optmat on its own, and little for a solprop run. It was passed to
    optmat as a low-priority change. *Resolved* in optmat 0.1.1 (Phase 1a, first part).
- **Handoff:** Phase 1a starts from this branch's `main`. The development environment
  needs `uv sync`, and the Kokkos wheel is reinstalled by hand after it, as before.
  Phase 4a attributes changes against `sep3_baseline.*` and `sep3_changes.json`. The
  generator must not be re-run after any default has changed.

### Phase 1a, first part — selection, bridge, gases, EOS, transport and solids (NumPy)

- **Scope:** the first part of §11's Phase 1a, 2026-10-01, implemented by Opus 5.5 (AI)
  for John Travers, on `feature/sep-0003-optmat-materials-1a` from `main` at `05172a4`.
  The run manifest, the output schema increment, resume and scan identity and the
  drivers' material keywords are the second part. Everything here is **opt-in**: no
  built-in default changed, and every existing test passes unchanged.
- **Delivered:**
  - `src/solprop/physdata/selection.py`: `MaterialSelection` (the linear, electronic,
    Raman and ionisation choices, the carrier, `energy_loss="peak-weighted"` and
    `raman_shift`), `MaterialRecord` (an optmat `ModelIdentity` in plain values, with
    `to_dict()`) and `DiluteScalingCheck`. It imports nothing from optmat.
  - `src/solprop/physdata/_optmat.py`, the private bridge. It imports optmat inside its
    functions only. It converts bar to Pa (×10⁵ exactly) and ω to vacuum wavelength
    (ω = 0 is the static limit `inf`), and turns identities into records. It applies the
    status policy (`MaterialStatusWarning` once for `provisional` or `untested`), and
    `translated()` turns optmat's `ModelRangeWarning` and `ModelAbsorptionWarning` into
    one `MaterialRangeWarning` per material and evaluation context. That warning names
    the parts of the band outside the qualified intervals and each gap's reason, and
    `OptmatError` becomes `MaterialError` naming the material, quantity, state and model.
    `material_context()` scopes "once".
  - `physdata/sellmeier.py`: `OptmatSusceptibility`, a `GasSusceptibility` with
    γ = α/ε₀ from `complex_polarizability` and the exact `dpolarizability_domega`, real
    for a transparent model, with an `absorbing` property (False on the base class) and
    `dilute_scaling_error`. `Gas.Air` resolves to `air_model("dry air (N2, O2, Ar)")`.
  - `physdata/species.py`: `OptmatGas`, the `EquationOfState` protocol on optmat's
    density and its inverse, with exact vacuum and the saturation pressure named for a
    condensed state, and carrying the transport properties. `species(...,
    selection=)` and `Species.material_records` are added.
  - `physdata/transport.py`: optmat dispatch for `viscosity` (optmat's Wilke rule for a
    mixture, air resolved into its constituents), `heat_capacity_ratio`, `molar_mass`,
    the real-gas `sound_speed` and `temperature_range`. `gasflow.py` reaches densities
    and molar masses through the same dispatch; the mean free path and regime checks
    are unchanged.
  - `physdata/mixtures.py`: `selection=` on `Mixture.fill`, `partial_pressures` and
    `air` (whose fractions come from optmat's named compositions on that path);
    `Mixture.material_records`, `Mixture.dilute_scaling_checks`,
    `DILUTE_SCALING_THRESHOLD` (10⁻³) and the `DiluteScalingWarning`.
    `modes/fill.py` runs the check once per gas when a fill's spectrum is sampled.
  - `physdata/glasses.py` and `metals.py`: `OptmatGlass` (optmat's complex index and its
    exact derivative, never conjugated), and `model=` on `fused_silica`,
    `calcium_fluoride`, `magnesium_fluoride(axis=)`, `barium_fluoride`, `sapphire(axis=)`,
    `bk7`, `glass(name, axis=)`, `silver` and `aluminium`.
  - `modes/stepindex.py`: the lossless eigenproblem refuses an absorbing `OptmatGlass` by
    name; `OptmatGlass.as_transparent()` is the recorded alternative.
  - `exceptions.py`: `MaterialStatusWarning` and `DiluteScalingWarning`.
  - Tests: `tests/physdata/test_optmat_bridge.py`, `test_optmat_gases.py`,
    `test_optmat_eos.py`, `test_optmat_transport.py` and `test_optmat_solids.py`.
  - Pages: the new section of `docs/guide/media/materials.md`, "Materials from optmat"; a
    paragraph in `docs/guide/media/mixtures.md`; `docs/api/species.md`; `CHANGELOG.md`.
- **Deviations from the proposal:**
  - The selection's `electronic`, `raman`, `ionisation`, `carrier_wavelength`,
    `energy_loss` and `raman_shift` are validated and recorded, but not yet resolved.
    A selected species keeps solprop's γ₃, ionisation and Raman data, which Phases 1b,
    1d and 1c replace.
  - The lossless-provider refusal is by name only in `StepIndexMode`, the one provider
    that silently took the real part of any glass. The step-index provider's material
    check, the graded-index provider and the vector-field path already refuse a
    complex permittivity, without naming the material. The legacy `fused_silica()`
    default, whose real part `StepIndexMode` takes as documented, is unchanged.
- **Decisions:**
  - `MaterialSelection.electronic` accepts `"recommended"` and `"carrier-kerr"`,
    synonyms for the one policy §4.3 defines. A model field takes a name or any object
    with an `identity`, checked against its material when it is resolved.
  - A transparent optmat gas model's susceptibility is returned real. Its imaginary
    part is identically zero, and a real array keeps the legacy dtype in the fills'
    fast paths. An absorbing model's stays complex, and the fill's species attribution
    and the saved cross sections count it (`absorbing`).
  - `OptmatSusceptibility.reference_state` is zero pressure at the source's reference
    temperature: a polarizability is a per-particle, dilute quantity, with no density
    to divide out. Its `_chi_and_derivative` refuses.
  - The dilute check runs at a fill's spectral sampling (`_sampled_susceptibility`),
    which already knows the band, each population's density path and the temperature.
    It takes the largest density over the path's breakpoints and 65 probes, at most 64
    frequencies of the band inside the qualified intervals, and warns once per gas. It
    never runs for a registry species. It compares the two laws **at the density
    itself**: optmat's `refractivity` takes the ideal-gas state `p = N k_B T` with
    `ideal=True`, whose density is `N` to one rounding, with the refractivity virial
    still applied; no equation of state or inverse is consulted. At 294 K this
    matches the real-gas state's refractivity to 0.0 relative for N₂ at 1 bar, and to
    4 × 10⁻¹⁶ for Ar at 10 bar. A refusal
    from optmat records the check as not measured, with its reason
    (`DiluteScalingCheck.measured`, `reason`), and never fails the fill. Two `neff`
    calls on one mode each warn once, until the second part's run-level context.
  - `Gas.Air`'s viscosity is the Wilke rule over the constituents of
    `"dry air (N2, O2, Ar)"`, since optmat has no transport model of air. Its density is
    the pseudo-pure air's (D9), and so is its molar mass, 0.0289655 kg/mol, which the
    mass flow, the dilute sound speed and the mean free path take, not the
    constituents' mean, 0.0289597 (2.0 × 10⁻⁴ lower). optmat's equation-of-state object
    does not expose its molar mass, so solprop reads it from the same CoolProp fluid
    (`EquationOfState.fluid`, `"Air"`) through pyfluids; a test pins the fluid name.
  - A transport composition that mixes optmat-selected and registry species is refused,
    so one flow never combines two transport sources.
  - Under optmat, MgF₂ and sapphire need an axis for the recommended model; a named or
    supplied model names its own ray. The legacy models accept `axis="o"` and refuse
    `"e"` by name.
  - `OptmatGlass.largest_dropped_extinction` accumulates over the adapter's
    evaluations. `as_transparent()` returns a fresh adapter, so one run's record is its
    own.
  - The transparent approximation has a declared domain:
    `TRANSPARENT_ABSORPTION_BOUND`, 100 m⁻¹ in the intensity absorption coefficient
    α = 4πκ/λ (an e-folding over a centimetre), adjustable per adapter
    (`as_transparent(absorption_bound=)`). Beyond it the adapter warns once
    (`MaterialRangeWarning`) and records α, κ, the wavelength and the bound
    (`OptmatGlass.transparency`). The bound is set by optmat 0.1.0's `fit/SiO2`:
    - across 200 nm–2.5 µm, α stays below 60 m⁻¹ (1.9 m⁻¹ at 200 nm, 59 m⁻¹ in the
      2.21 µm band, 25 m⁻¹ at 2.5 µm; 6.5 × 10⁻¹¹ κ at 250 nm, zero at 300–500 nm);
    - at the ultraviolet edge it passes 100 m⁻¹ near 176 nm (510 m⁻¹ at 170 nm,
      5.2 × 10³ at 160 nm, 1.0 × 10⁸ at 120 nm, κ = 0.97);
    - in the OH band it reaches 2.8 × 10³ m⁻¹ at 2.73 µm.

    The window passes the bound and both edges fail it.
  - `MaterialRecord` also carries optmat's `sources` and `reference_temperature`, for
    the manifest.
  - "Once per evaluation context": a fill's sampling, a `Mixture.fill` (whose flow
    evaluates the viscosity many times) and each bridge call open one. Outside them,
    each call is its own context.
- **Verification:**
  - `uv run poe check` on the branch: lint, format, types, docstrings and spelling
    clean. The tests: 11 846 passed and 30 skipped. Two scan-queue tests timed out
    under the fully loaded `-n auto` lane, as they do on `main`
    (`test_two_controllers_share_claims_and_one_collector` and
    `test_killed_controller_claim_is_recovered_without_waiting_for_age`); both pass
    alone. The strict docs build succeeds.
  - `uv run poe cov`: 100 % line and branch coverage, without new pragmas, with
    11 848 tests passed (those two included).
  - The new tests compare every adapter with direct calls of optmat 0.1.0's public
    functions:
    - χ of every gas, `Gas.Air` included, against `complex_polarizability`;
    - derivatives against fourth-order central differences at three steps each (gases
      to 10⁻⁸, solids to 5 × 10⁻⁸ of |dn/dω|, complex silica and metals included);
    - every solid and ray at 110, 120, 300, 800 and 1000 nm and 5 µm where in range,
      and at its range edges;
    - interval gaps of CH₄ and N₂O, with their reasons;
    - density–pressure round trips to 2 × 10⁻¹², exact vacuum, and the condensed-state
      and axis refusals;
    - mixtures: one component, split and reordered labels, vacuum, named air;
    - transport against solprop's registry, and short steady-flow fills;
    - overrides with status `user` and their `base`, the lossless-provider refusal,
      and the lazy import;
    - after the physics review: cooled optmat fills of N₂ at 280 K and Ar at 250 K
      that build their modes and measure their dilute error; a refused check
      recorded unmeasured with the fill still built; the inverse-failure message;
      the air molar mass against its equation's fluid and the mass flow; the
      transparent bound passed by silica's window and failed at 120 nm and 2.73 µm,
      once per adapter.
  - After those fixes: `poe check` again clean except
    `test_two_controllers_share_claims_and_one_collector` under `-n auto` (passes
    alone), 11 853 passed; the strict docs build succeeds; `poe cov` 100 % with
    11 854 passed.
  - No propagation was run: nothing here changes a kernel array for the defaults.
  - **The pin moved to optmat 0.1.1** (tag `v0.1.1`, `7acfbbd`), reviewed as §9.2's
    step 6 asks. It contains optmat's OEP 2 and OEP 3 made Final, its lazy imports
    and the density inverse fixed at the melting line. Of the 84 models solprop uses,
    82 keep their identity digests:
    - CH₄'s dispersion model changed its digest, but its polarizability, frequency
      derivative and qualified intervals are bitwise identical at 400 wavelengths
      from 200 nm to 15 µm;
    - `"dry air (CIPM-2007)"`, which solprop does not use, also changed.

    A run resumed across the bump is refused for CH₄, which is the conservative side.
    The import of the whole surface fell from 2.3–2.6 s to 0.97 s in a same-process
    A/B, and scipy.signal, scipy.stats and scipy.ndimage are no longer loaded (Q1). A
    new optmat refusal: a temperature at an equation's lowest temperature, its triple
    point.
- **Sources checked:** none new. Every number is optmat 0.1.0's, and the tests compare
  the adapters with direct calls of its public functions. The dilute-scaling error of
  argon at 10 bar (−0.19 %) and the 1.9 × 10⁻⁴ density difference of the air
  populations reproduce §4.2. The transport values agree with solprop's registry within
  the differences of their sources (viscosity 2.2 × 10⁻³ at most, heat-capacity ratio
  10⁻⁵, molar masses exactly).
- **Limitations:**
  - optmat 0.1.0's density inverse (`EquationOfState.pressure`) of Ar, N₂ and CH₄
    failed below about 255 K, 284 K and 256 K at any density, where each is
    supercritical: its check and root bracket reached the equation's top pressure,
    past the melting line. optmat 0.1.1 ends the domain at the melting line, and this
    phase moves the pin to it (below), so the limitation and its special error
    message are gone. A cooled fill's pressure now round-trips
    (`test_a_cooled_supercritical_fill_inverts_its_density`).
  - No absorbing optmat model exists for a solprop gas at 0.1.0 (only O₃), so the
    absorbing branch is tested with a test double.
  - The air compositions are recorded only through their species' records. The
    composition name is not yet a field of the mixture.
- **Open questions:**
  - Q2. optmat's density inverse below the temperatures above. *Resolved:* fixed in
    optmat 0.1.1, which solprop now requires (`optmat[gas]>=0.1.1`).
  - Q3. Should the step-index provider, the graded-index provider and the vector-field
    path name an absorbing optmat material and offer `as_transparent()` too? Their
    refusals today are generic (next phases). *Decided* by John Travers on
    2026-10-02: every lossless-only route takes the transparent approximation
    automatically and never refuses for absorption ([pre-defaults fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
  - Q4. Should a mixture record the named composition it was built from, for the
    manifest (second part)? *Resolved* in the second part: yes, as `Mixture.composition`.
- **Handoff to the second part:**
  - **Records.** Every optmat-backed object has `material_records`, a tuple of
    `MaterialRecord` with `to_dict()`: `Species`, `Mixture` (each digest once),
    `OptmatSusceptibility`, `OptmatGas` (its equation of state, then each
    constituent's transport model) and `OptmatGlass`. A run's manifest can collect them
    from its mixture, or from the mode's `mixture` and `susceptibility`, and from every
    `OptmatGlass` among the walls, cores and materials it uses. A `GlassIndex` is
    recognised by `isinstance(index, OptmatGlass)`.
  - **Policies.** `MaterialSelection.to_dict()` gives the selection.
    `Mixture.dilute_scaling_checks(omega)` gives each population's measured dilute
    error over the run's band; record it, and do not recompute it per stage.
    `OptmatGlass.transparency` gives the transparent approximation and its bound;
    record a `DiluteScalingCheck` that was not measured with its `reason`.
  - **Contexts.** Wrap the translation in `_optmat.material_context()`, so that each
    range and status warning is raised once per run.
  - **Identity.** `IndexMaterial.identity`, which a general modal guide's key uses,
    already changes with the model, because `OptmatGlass`'s representation carries the
    full digest.

### Phase 1a, second part — manifest, output schema 8, resume and scans, driver keywords (NumPy)

- **Scope:** the second part of §11's Phase 1a, 2026-10-01, implemented by Opus 5.5 (AI)
  for John Travers, on `feature/sep-0003-optmat-materials-1a2` from `main` at `fd6e086`.
  It adds the material manifest of §9.3, the output schema increment of §6, the resume
  and scan identity, and the drivers' `materials=` keyword. Everything stays **opt-in**:
  no default changed, and `materials=None` leaves every material and array as it was.
- **Delivered:**
  - `src/solprop/physdata/selection.py`: `MaterialManifest`, frozen, deduplicated and in
    canonical order, with `hashed()` (the identities and policies), `to_dict()` (with
    the measurements) and `from_dict()`; and `MaterialLog`, what one evaluation context
    has warned about and the bands it found outside a model's intervals.
  - `src/solprop/physdata/_optmat.py`: `material_context(log=)` resumes a run's log, and
    a log resumed inside another context is stacked on it; `open_log()`; a range
    warning records the parts of the band it names in every open log.
  - `src/solprop/physdata/mixtures.py`: `Mixture.composition`, the named optmat
    composition `Mixture.air(selection=)` took its fractions from (Q4).
  - `src/solprop/driver/material_output.py`: `material_manifest()`, which walks a run's
    own objects and collects every optmat-backed one wherever it sits: the mixture, a
    mode's fill and cladding, a planar grid's slab modes, a step-index core, a general
    guide's `IndexMaterial` and `MaterialRegion`, a bulk medium.
  - `src/solprop/output/schema.py`: `SCHEMA_VERSION = 8`; `RunMetadata.materials`,
    which enters `hashed()` from schema 8 and is written whole by `to_json()`;
    `package_versions()` adds optmat's version, outside the key; a manifest below
    schema 8 is refused; the printed record names optmat and the models.
  - `src/solprop/output/hdf5.py`: `check_cache` refuses a file of schema 7 or earlier
    for a schema-8 run with an explanation, and says so when a refused difference is in
    the manifest. `src/solprop/output/scan.py`, `driver/_scan_run.py` and
    `driver/_scan_batch.py`: a refused reuse of a completed point names the fields that
    changed, read from the point's stored record.
  - `src/solprop/driver/simulation.py`, `interface.py`, `mixtures.py`: `materials=` on
    `prop_capillary`, `prop_waveguide`, `prop_gnlse`, `prop_free`, `prop_radial`,
    `prop_planar` and their `Simulation.from_*` constructors, and on `Simulation`;
    `Simulation.material_manifest()`; one material context per run. `solprop.sol` and
    `solprop.physdata` export `MaterialSelection` and `MaterialManifest`.
  - Tests: `tests/physdata/test_material_manifest.py`,
    `tests/driver/test_material_manifest.py` and `tests/output/test_material_resume.py`;
    schema-version assertions moved from 7 to 8, and `tests/test_display.py`'s
    run-record literal gained a schema-8 value beside the unchanged schema-7 one.
  - Pages: `docs/guide/results/output.md` ("The material manifest", the resume key, files of
    earlier schemas), `docs/guide/media/materials.md` ("Selecting a run's materials"),
    `docs/guide/workflows/scans.md`, `docs/guide/setup/running.md`, `docs/guide/workflows/interactive.md`,
    `CHANGELOG.md`. The kernel contract is unchanged (kernel schema 15).
- **Deviations from the proposal:**
  - §9.3 lists the optmat "version and commit". The manifest records the version
    string; optmat exposes no commit, and the digests are what tell two optmat trees
    apart, as §9.3 says.
  - The GNLSE's held-back Raman fraction, the rate threshold kind and each species'
    constant energy and bound are not recorded: they are resolved in 1b and 1d, which
    add them (Handoff).
  - `prop_planar` cannot resolve a registry gas under a selection, because the planar
    transform admits no mixture, so a selection beside a registry fill is refused by
    name rather than left unresolved.
- **Decisions:**
  - **What the key covers.** `MaterialManifest.hashed()` holds the optmat version,
    every record (all its fields), the species, the compositions, each solid's
    identity and transparency policy (its bound), and the policies. The measurements
    stay outside: the dilute-scaling errors, a transparent solid's dropped loss, the
    out-of-interval bands and the citations of other material objects. So does the
    selection as given, since what it resolved to is in the records, which keeps
    `"recommended"` out of the key. The selection's `raman` and `ionisation` choices
    resolve nothing yet and are therefore not keyed; 1c and 1d key them through
    their records.
  - **Keys that name the entry.** Records are keyed `material:kind:name`, species by
    label and solids by `material:axis:model`, numbered `#1`, `#2` when two share a key,
    so a refused resume names `materials.records.Ar:gas_refractive:fit/Ar.digest`
    rather than a list index.
  - **The empty manifest.** A run with no optmat-backed object and no selection keys
    on `materials: null`, and its record taken at schema 7 is the schema-7 key bit for
    bit (`test_the_run_record_hashes_as_it_did_before_the_printing` keeps the old
    literal beside the new one). A schema-8 hash cannot equal a schema-7 one, since the
    schema version is in the key, as at every earlier increment.
  - **Q4: yes.** `Mixture.composition` is set by `Mixture.air(selection=)` when it took
    optmat's named fractions, recorded by `Mixture.serialise()` and the manifest, and
    kept out of the generic dataclass record (`record=False`), so every mixture without
    one keeps its digest. A mixture derived by `with_fill`, `with_densities` or
    `at_parameters` drops it.
  - **The manifest module.** The manifest and the log live in
    `physdata/selection.py`, which imports nothing from optmat; the collection, which
    needs the modes, lives in `driver/material_output.py`. No module was added.
  - **What a driver resolves.** Names and defaults the driver resolves itself (a gas,
    a fraction mapping, `"air"`, `Gas.Air`, a numbered capillary's default cladding)
    come from optmat under a selection; objects the caller built are taken as they
    are. A `SpatialMedium` naming a registry gas with the registry's own
    susceptibility is the exception: it meant the built-in gas, so it becomes the
    one-population optmat `Mixture` at the same pressure and temperature, and a
    supplied susceptibility is kept beside optmat's equation of state, as
    `capillary_fill` does. That mixture medium takes the mixture route's default
    statistics.
  - **One context per run.** Each `Simulation.from_*` opens a material context; the
    simulation adopts its log, and `translate()`, `metadata()`, `material_manifest()`
    and `run()` resume it. A simulation derived by `with_input` or `with_mixture`
    takes a copy of its parent's log: it does not warn again about what the parent
    has, and its own warnings and recorded bands do not reach the parent or a
    sibling, so a scan's points derived from one base record independently of
    their order.
  - **What the walk enters** (after the contract review). Mappings, sequences,
    dataclasses, `functools.partial` (its function, arguments and keywords), bound
    methods (their instance) and any object with a `__dict__`, whoever defined it,
    so a caller's subclass of a solprop mode or a mode of their own is walked. It
    never enters arrays, numbers, strings, functions (a closure's captures are not
    walked), modules, or objects of numpy, SciPy, h5py, optmat (recorded through the
    adapters that hold them) and a few other packages that hold no material. Every
    object is visited once, so a cycle ends. The roots are the run's configuration,
    modes, responses, bulk medium, spatial grid, general route and wall: the planar
    grid was missing at first, and its slab modes' cladding was not collected.
  - **The dilute check once per run.** The manifest measures each optmat
    population's dilute-scaling error once, over the run's whole grid band, collars
    included, at its largest density; the fill keeps warning at its own sampling.
- **Verification:**
  - `uv run poe check` on the branch: lint, format, types, docstrings and spelling
    clean; 11 907 tests passed and 30 skipped (the two scan-queue tests included); the
    strict docs build succeeds.
  - Targeted coverage, the default lane measured with `--cov=src/solprop --cov-branch`:
    100 % line and branch coverage, without new pragmas, of every module changed here
    (`driver/simulation.py`, `interface.py`, `mixtures.py`, `material_output.py`,
    `_scan_run.py`, `_scan_batch.py`, `output/schema.py`, `hdf5.py`, `scan.py`,
    `physdata/selection.py`, `_optmat.py`, `mixtures.py`, `__init__.py` and `sol.py`).
    The package total is 97 %, the compiled kernels being measured only by the
    coverage lane.
  - After the contract review: a planar run's optmat slab cladding is collected,
    with and without a selection; an `OptmatGlass` is found in a caller's dataclass
    subclass of `StepIndexMode`, a caller's own mode class, a `functools.partial` and a
    bound method, and not through a closure or in a raw optmat model; a derived
    simulation's log is a copy, which neither shares the parent's container nor
    writes back to it. On the result, `poe check` is clean with 11 915 tests passed
    and 30 skipped and the strict docs build succeeds, and the changed modules are
    again at 100 % line and branch coverage in the default lane.
  - What the new tests check: canonical order and deduplication, keyed collisions,
    round trips through plain values, JSON and HDF5, and a malformed record refused;
    which fields are hashed and which are not; the run log once across re-entries and
    stacked into an outer context, with the band parts recorded; the composition of
    both air conventions and its absence elsewhere; collection from a capillary (the
    gas, its equation of state and transport, the recommended wall), from air, a
    fraction mapping, a supplied cladding (cited, not recorded as optmat), a
    waveguide's supplied modes, free space and radial media (resolved, a supplied
    susceptibility kept, a glass taken as it is), a step-index core and a material
    region, and a GNLSE run whose record differs only by `materials`; the planar
    refusal; exactly one `MaterialRangeWarning` per run across construction,
    translation and record on a band below argon's interval; manifests independent of
    what was resolved before; resume accepted for an identical manifest and refused,
    by the entry, for an optmat override (`with_parameters`), a changed policy, a
    dropped selection, a monkeypatched optmat version and a schema-7 file; scan reuse
    refused for an override and accepted unchanged; and ten fixed steps of an argon
    capillary on `numpy_ref` through `prop_capillary(..., materials=MaterialSelection())`
    writing the manifest to HDF5.
  - Not run: `poe test-slow` and `poe cov`, which run at the SEP's milestones; no
    backend or kernel changed.
- **Sources checked:** none new. The manifest's values are optmat 0.1.1's identities as
  its public `identity` reports them.
- **Limitations:**
  - The walk does not find an optmat-backed object reached only through a function's
    closure, a property, or an object of a stopped package.
  - The out-of-interval record depends on what the run actually evaluated: a band the
    translation never evaluates (the dilute check's probes, say) is not named.
  - A `SpatialMedium` given `susceptibility=gamma(gas)` explicitly cannot be told from
    the default and is resolved under a selection.
- **Open questions:**
  - Q5. Should the hybrid planar transform admit a one-population mixture, so that
    `prop_planar(materials=)` can resolve a registry fill instead of refusing it?
    *Decided* by John Travers on 2026-10-02: no mixture; the slab's registry gas takes
    optmat's single-species susceptibility ([pre-defaults fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
  - Q6. Should the manifest key the selection's unresolved choices (`raman`,
    `ionisation`) until 1c and 1d resolve them? Today a change to them, which changes
    nothing in a run, does not refuse a resume. *Resolved* 2026-10-02: 1c and 1d key
    them through what they resolve to, verified by test ([pre-defaults fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
- **Handoff to 1b, 1c and 1d:**
  - **Records.** Give every new optmat-backed object (a silica bundle, a Raman response
    adapter, an `OptmatRate`) a `material_records` property returning
    `MaterialRecord`s, and make sure it is reachable from the run's objects (a response
    in `Simulation.responses`, a rate in a response's fields): `material_manifest()`
    then collects it, keyed and hashed, with no other change.
  - **Policies.** Add a resolved policy to `_policies()` in `driver/material_output.py`,
    or a new manifest field in `MaterialManifest` with its place in `hashed()` (a
    policy) or only in `to_dict()` (a measurement). 1b: the GNLSE's held-back Raman
    fraction and its origin (bundle, user, or 0 with Raman off). 1c: replace
    `raman_reference_temperature: None` by the resolved temperature, and record the
    `reference-density` shift and the term counts. 1d: the rate threshold kind, each
    species' constant energy loss and its bound, and the reference field.
  - **Warnings.** Use `_optmat.warn_once` inside the run's context; a new kind of
    out-of-range record goes into `MaterialLog`.
  - **Keys.** A new hashed field changes the schema-8 key of every optmat run, which
    is acceptable before Phase 4a (nothing public), but an increment of the output
    schema is needed if a stored field changes meaning.

### Phase 1b — electronic coefficient and silica bundle (NumPy)

- **Scope:** §11's Phase 1b, 2026-10-01, implemented by Opus 5.5 (AI) for John Travers,
  on `feature/sep-0003-optmat-materials-1b` from `main` at `368ad17`, in parallel with
  1c and 1d. It binds each gas's Kerr coefficient at the carrier (§4.3) and takes fused
  silica's response from optmat's silica bundle (§4.5, D4, D11). Everything is
  **opt-in**: `materials=None` and a species or a GNLSE problem without a selection
  change nothing, configuration records included.
- **Delivered:**
  - `src/solprop/physdata/nonlinear_data.py`: `OptmatKerr`, optmat's
    hyperpolarizability model of a gas through its public `gamma3("kerr", λ)`, which is
    γ_T/(6ε₀), evaluated at a carrier with the range policy of §5, and its
    `at_carrier(λ)` record, `CarrierKerr`: the coefficient used, optmat's
    third-harmonic one at the same fundamental, and their ratio where that evaluation
    is within the fit. `chi3(..., selection=, wavelength=)` and `n2(..., selection=)`
    give the selected values, `n2` with the exact $n_0^2$ of the selected index;
    `carriers_agree` and `selected_carrier`.
  - `src/solprop/physdata/species.py`: `Species.kerr` (kept out of the configuration
    record) and `Species.kerr_coefficient(λ)`; under a selection `species()` gives
    `gamma3=None` and the model (air, which optmat models only through its
    constituents, gets neither), refuses a `gamma3_source`, and the species' records
    include the model's.
  - `src/solprop/physdata/mixtures.py`: `Mixture.gamma3_path(wavelength)`.
  - `src/solprop/nonlinear/mixtures.py`: `responses_for` takes each population's
    coefficient at `lambda0`.
  - `src/solprop/physdata/glasses.py`: `SilicaBundle`, optmat's bundle by selection
    name, bundle name or object, with `n2(λ)`, `raman_fraction`, `h_r(t, component)`
    (a tensor part of a copolarised-only shape refused by name) and the records of the
    bundle, its Kerr fit and its shape.
  - `src/solprop/nonlinear/raman.py`: `SampledShape`, a `RamanModel` whose undamped
    row is the bundle's `"parallel"`, `"isotropic"` ($a + b/3$) or `"rank2"`
    ($\tfrac23 b$) shape with no dephasing; `bundle_response`, the scalar response or
    the two tensor terms; `SilicaTensorResponse(parameters=SilicaBundle)` with
    `hellwarth(t)`; the scalar-only refusal of `translate/responses.py` names the
    models.
  - `src/solprop/nonlinear/gnlse.py`: `problem_from(..., selection=)` and
    `n2="recommended"`; `GnlseProblem.bundle`, `fraction_origin` and `n2_origin`
    (kept out of the configuration record). `src/solprop/driver/simulation.py` and
    `interface.py`: `from_gnlse`/`prop_gnlse` pass the selection; the single-gas
    capillary path takes the optmat coefficient at `lambda0`; a selection's carrier
    or `lambda0` that is not the grid's is refused.
  - `src/solprop/physdata/selection.py` and `driver/material_output.py`: the manifest
    field `electronic` (each optmat population's `CarrierKerr`, a measurement) and the
    hashed policies `electronic_response` (the `carrier-kerr` law and its carrier) and
    `gnlse_raman` (the held-back fraction, its origin, whether a response carries it,
    the bundle and the origin of $n_2$).
  - Tests: `tests/physdata/test_optmat_kerr.py`, `tests/physdata/test_silica_bundle.py`,
    `tests/nonlinear/test_silica_bundle_raman.py`, `tests/nonlinear/test_gnlse_selection.py`
    and `tests/driver/test_material_electronic.py`; 1a's tests that listed a selected
    species' records gained `hyperpolarizability/...`.
  - Pages: `docs/guide/media/materials.md` ("The electronic Kerr coefficient at the carrier",
    "Fused silica's nonlinear response", the driver table, records and limitations),
    `docs/guide/waveguides/gnlse.md` (the fraction rule under a selection), `docs/api/species.md`,
    `CHANGELOG.md`.
- **Deviations from the proposal:**
  - solprop has no vector GNLSE: `prop_gnlse` is scalar. The tensor default, with its
    own $f_R = 0.245$, is taken by `SilicaTensorResponse(parameters=
    SilicaBundle.resolve(tensor=True))` in a material region, whose fraction is its
    response's (a region has no fraction field), and by
    `bundle_response(..., tensor=True)` as a `RamanResponse`. The GNLSE takes the
    tensor bundle only when it is named (`MaterialSelection(raman="tensor")`), and then
    its copolarised $a + b$ with 0.245.
  - The design tools that take a registry name, `gas_ratio` and `pressure_rdw`, have no
    `materials=` (none had one in 1a) and keep the registry; `physdata.chi3` and `n2`
    gained `selection=`. A selected `Mixture` passed to `soliton_params` or
    `capillary_params` takes its coefficients at the tool's wavelength (Q7).
  - A free-space or radial run's `KerrResponse` is the caller's, and is taken as it
    is; its manifest records the coefficient the medium's optmat species has at the
    carrier, which that run need not use (Q9).
  - §7's "known nonlinear phase in a short fixed-intensity test" is covered by the
    single-tone cubic polarisation and the $n_2$ conversion it implies, and by ten
    GNLSE steps that reproduce solprop's own response, rather than by a propagated
    phase.
- **Decisions:**
  - **The model, not a number.** A selected species carries the dispersive model and
    no `gamma3`; every consumer asks `kerr_coefficient(λ)` with its carrier, so no
    universal value is bound and one species serves runs at any carrier. A species
    with both is refused.
  - **One carrier, the grid's.** The run's carrier is its grid's reference wavelength.
    A selection's `carrier_wavelength`, and a capillary's `lambda0` beside a grid of
    one's own, must agree with it to $10^{-12}$ relative, or the run is refused: §5
    allows "a deliberately named approximation", and none is named yet (Q10).
  - **The THG discrepancy.** optmat's third-harmonic coefficient is evaluated for the
    record only; where it leaves optmat's fitted data (N₂O at 800 nm) the record keeps
    the value and the reason, and no ratio.
  - **What is keyed.** The law and the carrier are a hashed policy, and the Kerr
    model's digest a record; the coefficients and ratios, determined by those two, are
    measurements. A run of solprop's own materials keeps the empty manifest.
  - **The GNLSE's bundle.** `selection.raman` names it: `"recommended"` is optmat's
    `"default"`, another name an optmat bundle name, an object is taken.
    `"provisional"`, a gas policy, was first mapped to the default; at the merge with
    Phase 1c it is refused by name, the two paths sharing `RAMAN_POLICIES` (Phase 1c's
    Decisions). It is resolved only when it is used, Raman on or
    `n2="recommended"`, which needs a selection. The origins are `"bundle"`, `"user"`,
    `"raman-off"` and, without a selection, `"default"`.
  - **The sampled row at zero delay** is half the shape's right-hand limit, the
    trapezoid's endpoint, as `SilicaTensorResponse.kernels` already does. The limit is
    $2h(t_{\min}) - h(2t_{\min})$ at the smallest normal delay: a shape that opens with
    a sine doubles exactly between the two and cancels to an exact zero, so the default
    bundle's row equals solprop's own Hollenbeck–Cantrell row to $3.5\times10^{-16}$,
    and the boson peak's jump, $2f_b/\tau_b$, survives.
  - **Legacy records unchanged.** The new fields of `Species` and `GnlseProblem` carry
    `record=False`; `SilicaTensorResponse.parameters` accepts a bundle in the same
    field; its kernels now go through Hellwarth's $a$ and $b$ by the same operations,
    so the declared parameters' kernels are bit for bit what they were.
  - **Caller-built objects** (a region's `SilicaTensorResponse()`, a free run's
    `KerrResponse`) are taken as they are, as in 1a (Q8, Q9).
- **Verification:**
  - The physics review found, and the merge fixed:
    - the GNLSE driver did not compare `lambda0` with the grid's carrier under a
      selection. An `EnvelopeGrid` at 1030 nm with `lambda0=800e-9` was accepted, and
      the bundle's n₂ was taken at 800 nm (2.4 % high). `from_gnlse` now refuses it
      as the capillary driver does;
    - `n2(..., selection=)` fell back silently to 800 nm for the dispersive
      coefficient. It now takes the caller's or the selection's carrier and refuses
      when there is neither, and the registry path keeps 800 nm;
    - the `gnlse_raman` policy now records the carrier at which the bundle's n₂ was
      evaluated.

    The review confirmed χ₃ = γ₃ρ exactly, with n₂ = 1/n₀² of optmat's dilute value.
    It also confirmed the fraction rules, the n₀² ≈ 2.10 ratio that is not imported,
    the sampled row matching the legacy Hollenbeck row to 3.5 × 10⁻¹⁶, and the tensor
    probes to 2 × 10⁻¹⁶.
  - `uv run poe check` on the branch: lint, format, types, docstrings and spelling
    clean; the tests 11 991 passed and 30 skipped, with the two scan-queue tests timing
    out under the loaded `-n auto` lane as on `main`
    (`test_killed_controller_claim_is_recovered_without_waiting_for_age`,
    `test_two_controllers_share_claims_and_one_collector`); both pass alone. The strict
    docs build, aborted by that, was run on its own from clean and succeeds, after two
    reference fixes; the affected tests and doctests passed again after them.
  - Targeted coverage, the whole default lane with `--cov=src/solprop --cov-branch`
    (11 993 passed): 100 % line and branch coverage, without new pragmas, of every
    module changed here (`physdata/nonlinear_data.py`, `species.py`, `mixtures.py`,
    `glasses.py`, `selection.py`, `__init__.py`; `nonlinear/raman.py`, `gnlse.py`,
    `mixtures.py`, `__init__.py`; `driver/simulation.py`, `interface.py`,
    `material_output.py`; `translate/responses.py`; `tools.py`). The package total is
    97 %, the compiled kernels being measured only by the coverage lane.
  - Not run: `poe test-slow` and `poe cov`, which run at the SEP's milestones; no
    backend or kernel changed. The declared tensor kernels were checked bit for bit
    against the arithmetic before the refactor.
  - What the new tests check: the Taylor factor six for every gas at four carriers
    (to $4\times10^{-16}$); the coefficient's dispersion and the refusal without a
    carrier; the THG ratio of argon (1.177, as §9.1) and its absence for N₂O; one
    range warning per context below 400 nm; `chi3` and the exact $n_2 \leftrightarrow
    \chi^{(3)}$ round trip with $n_0$ against $N\gamma_T/(8\varepsilon_0^2cn_0^2)$; the
    single-tone cubic polarisation's $\tfrac34$ fundamental and $\tfrac14$ harmonic and
    the index change $n_2I$; the unit integral of each bundle shape and of $a$ and $b$
    (0.75, 0.25) by quadrature, and Hollenbeck's normalisation against an independent
    Faddeeva integral; the $n_0^2$ ratio of optmat's `raman_coefficient` and
    `electronic_susceptibility` to solprop's (exact to $10^{-14}$, $n_0^2 \approx 2.1$)
    at three wavelengths for both bundles; the long-pulse split (both bundles carry the
    whole $n_2$; the vector electronic part −3.35 %, the Raman part +12.0 %, as D11);
    the scalar row against solprop's Hollenbeck row; the tensor channels $a + b/3$ and
    $\tfrac23 b$, their co- and cross-polarised sums $a + b$ and $a$, the seam weights
    $(1, 0)$ and $(\tfrac14, \tfrac34)$, and a pump–probe evaluation in which a later
    parallel probe sees $a + b$ and a crossed one $a$; `SilicaTensorResponse` from the
    tensor bundle against the declared parameters (to $4\times10^{-16}$); the refusals
    by name; the fraction rule under a selection (bundle, 0, held back with Raman on and
    off) and unchanged without one, with the provenance outside the record;
    `n2="recommended"`; the capillary's Kerr terms (a single gas and a mixture) at the
    carrier and the registry's without a selection; the manifest's law, carrier, THG
    record and keys; the carrier refusals; the GNLSE manifest's fraction and origin;
    and ten fixed GNLSE steps under a selection that reproduce solprop's own response
    given the bundle's fraction to $10^{-12}$.
- **Sources checked:** none new. Every coefficient is optmat 0.1.1's, through its
  public functions; argon's carrier Kerr coefficient at 800 nm, 23.1 % above the
  registry's, and its THG/Kerr ratio reproduce §9.1, and D11's split reproduces.
- **Limitations:**
  - `carrier-kerr` is one coefficient per species and run; the third harmonic is
    driven by it.
  - The single-gas capillary path still builds its rate and Raman response from the
    registry; 1c and 1d replace them.
- **Open questions:**
  - Q7. Should the design tools that take a registry name (`gas_ratio`,
    `pressure_rdw`) take `materials=`, so a design and a run under a selection agree?
    *Decided* by John Travers on 2026-10-02: yes, every such tool ([pre-defaults
    fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
  - Q8. Should a driver resolve a caller's `SilicaTensorResponse()` to the tensor
    bundle under a selection? The two are equal to $4\times10^{-16}$ today, but only
    the bundle carries an identity. *Decided* by John Travers on 2026-10-02: yes, for
    the default parameters ([pre-defaults fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
  - Q9. Should `prop_free` and `prop_radial` under a selection build the Kerr
    response from the medium at the carrier, rather than take the caller's?
    *Decided* by John Travers on 2026-10-02: yes; implemented in [Phase 4a's first
    part](#phase-4a-first-part--the-defaults-switched).
  - Q10. Should a carrier other than the grid's be admitted as a named approximation
    (§5), recorded as a policy, rather than refused?
    *Decided* by John Travers on 2026-10-02: no; the refusal stays.
- **Handoff to 1d, 1c and Phase 2:**
  - **The carrier.** Under a selection the carrier is the grid's reference wavelength,
    and `_check_selection_carrier` (`driver/simulation.py`) has refused any other in
    the selection or in `lambda0`; `responses_for`'s `lambda0` is it, and so is
    `Simulation.material_manifest`'s `carrier=`. A rate or a Raman pump wavelength can
    take the same value.
  - **Shared code touched.** `Species` gained `kerr` as its last field (with
    `record=False`; a new field should do the same). `_policies()` takes
    `carrier=` and `gnlse=` and gained `electronic_response` and `gnlse_raman`; the
    manifest gained `electronic`. In `nonlinear/mixtures.py` only `responses_for`'s
    Kerr lines changed. `nonlinear/raman.py` gained `SampledShape`, `bundle_response`
    and `_right_limit` after `SilicaTensorResponse`.
  - **The single-gas capillary path** (`_capillary_responses`) is still taken under a
    selection; it now takes `kerr_chi3=`, and its rate and Raman response remain the
    registry's until 1d and 1c route them.
  - **Phase 2.** The bundle's rows are ordinary undamped rows with no decay path, and
    the tensor's rank-two row has a non-zero sample at zero delay; no backend change
    is expected.

### Phase 1d — rate adapters (NumPy)

- **Scope:** §11's Phase 1d, 2026-10-01, implemented by Opus 5.5 (AI) for John Travers,
  on `feature/sep-0003-optmat-materials-1d` from `main` at `368ad17`, in parallel with
  1b and 1c. Everything is **opt-in**: without a selection every rate, energy and
  array is as it was, and no existing test changed but `solprop.sol`'s exclusion
  list.
- **Delivered:**
  - `src/solprop/nonlinear/optmat_rates.py`: `OptmatRate`, the rate protocol on an
    optmat model, from a species name (atoms through `ionisation_rate_model`, molecules
    through `molecular_rate_model`, each refusing the other's species and forms) or a
    model object (`with_parameters`, `from_parameters`). It is `min(W(|E|), MAX_RATE)`
    above a threshold and exactly zero below; the threshold is the field at which
    optmat's instantaneous rate reaches `NEGLIGIBLE_RATE` (10⁻³⁰ s⁻¹), by bisection in
    `log E` to 10⁻¹². It carries `convention`, `species_identity`, `status`,
    `maximum_field`, `uncertainty_factor`, `energy_kind`, `material_records`, the
    constant energy `ionisation_potential` (`plasma_energy`) at a `reference_field`,
    clipped, with `energy_bound` over `energy_span`, and `parameters` (digest, species,
    carrier, threshold, negligible rate, ceiling, energy convention). Also
    `selected_rate` (what a selection means for a species), `bind_reference_field`,
    `check_input` (the maximum-field and D12 warnings), `PeakFieldReport` and
    `peak_field_report` (the post-run diagnostic).
  - `nonlinear/ionisation.py`: `rate_energy` and `bind_rate_energy`, the duck-typed
    seam by which a rate that declares its energy per event sets the energy its
    response pays. `nonlinear/responses.py` (`PlasmaResponse`) and
    `nonlinear/envelope_plasma.py` (`EnvelopePlasmaResponse`) apply it at
    construction, so the field loss term, the envelope loss current, the compiled
    `PlasmaCoefficients.ionisation_potential` and the species statistics all pay it, and
    the envelope's consistency check holds by construction.
  - `nonlinear/mixtures.py`: `_default_rate` resolves to `OptmatRate` under a
    selection; `responses_for(..., selection=)`; a bare `OptmatRate` is matched by its
    `species_identity`.
  - `translate/ionisation.py`: `_usable_range` stops a table at a rate's declared
    `maximum_field` (rate-table schema 2 unchanged).
  - `driver/simulation.py`: the capillary and waveguide route binds every unbound
    optmat rate to the input's peak field (`_capillary_input_peak`, the peak
    `default_field_range` takes, now with the field's circular share there) and checks
    the input; `_ionisation_rate` and `_capillary_responses` take the selection;
    `Simulation.run` computes the post-run diagnostic (`_peak_field_reports`), which
    `Simulation.material_manifest()` then carries. `driver/mixtures.py`:
    `ResponseSelection` keeps the selection, the reference field and the table range, so
    `with_mixture` rebinds the same energy.
  - `driver/material_output.py`: `policies["ionisation"]` (hashed), one entry per rate;
    `physdata/selection.py`: `MaterialManifest.peak_fields` (a measurement, in
    `to_dict` and `from_dict` only); `physdata/_optmat.py`: `record_outside`.
  - Tests: `tests/nonlinear/test_optmat_rates.py`, `tests/driver/test_optmat_rates.py`;
    `tests/test_sol.py` excludes `rate_energy` and `bind_rate_energy` from the
    interactive namespace. `docs/dev/figures/build.py` and `layers.svg` draw the new
    module.
  - Pages: `docs/guide/media/ionisation.md` ("Rates from optmat"), `docs/guide/media/materials.md`
    (pointer), `CHANGELOG.md`.
- **Deviations from the proposal:**
  - **Above the maximum field the rate is held** at its value there, and a table stops
    there. §4.6 says no new silent clamp is added, and expected optmat's rate to be
    tabulated above the maximum field with its warnings recorded. At optmat 0.1.1 that
    is not possible: a few per cent above the maximum field (1.04–1.07 times it for
    Ar, Kr and Xe, 1.16–1.19 for N₂ and He) the Tong–Lin-corrected instantaneous rates
    of the atoms and of the molecular defaults turn over, are erratic beyond (the
    "consistent" average divides by a cycle factor that is no longer smooth; argon's
    spikes to 2600 times its value at the maximum field at 1.59 times that field) and
    fall towards zero within a decade (Ar: 2.3 × 10¹⁰ s⁻¹ at 6.9 × 10¹¹ V/m). A
    default table of an argon run reaches 30 times the input's peak and did not
    converge in 2²⁰ knots. Holding is the least bad choice: at the held rate a
    population is depleted within a femtosecond. The hold is not silent: the input's
    peak above the maximum warns at construction (and in `with_input`), and a run whose
    recorded peak rate reaches the held value warns after it; whether a run reached
    the maximum field is that post-run diagnostic's to say. The manifest's `ranges`
    records a direct evaluation above the maximum field, never a table's, whose last
    knot is the maximum field to an ulp (a review found the first version recording an
    entry for every run, from that ulp). The hold leaves a corner in `W(E)`, which the
    backend's transverse-quadrature alarm reports in a run that reaches it (Q12).
  - **The post-run diagnostic is new, not an extension.** No peak-field report existed
    in `driver/simulation.py`. The diagnostic inverts each population's recorded
    `species_peak_ionisation_rate` (at the save planes). A field grid's is inverted
    through the rate; an envelope's, which is cycle-averaged, through the run's own
    cycle integrals of its table at linear polarisation (`cycle_moments`). The first
    version inverted optmat's `cycle_averaged_rate`, which departs from the envelope's
    average above the barrier-suppression field (Ar 1.003 at it, 1.07 at 0.8 and 1.35
    at the maximum field; Xe 1.26, He 1.16, N₂ 1.13 there), so a true peak of 0.95 of
    argon's maximum field read as reaching it. It is
    recorded in the run's manifest *after* the run (`Simulation.material_manifest()`);
    the file's record, written before the run, carries its inputs — the maximum field,
    the reference field, the constant and its bound — and the peak rates, not the
    result (Q13).
  - **Only the capillary and waveguide route binds the reference field**, the one
    whose default table range the SEP names. A free-space route takes its responses
    from the caller: an `OptmatRate` there keeps its own `reference_field`, or takes
    the weak-field limit (the lowest channel's energy) without one (Q14).
  - Beyond `_default_rate`, `nonlinear/mixtures.py`'s `responses_for` gained its
    `selection=` keyword and the bare-rate name check its `OptmatRate` branch;
    `PlasmaResponse`, `EnvelopePlasmaResponse` and `_usable_range` changed as listed.
  - The energy bound's span ends at the maximum field rather than at the table's top,
    since the rate is held above it.
- **Decisions:**
  - `NEGLIGIBLE_RATE` = 10⁻³⁰ s⁻¹, the envelope cycle tables' comparison floor: over a
    picosecond one atom in 10⁴² ionises. The threshold kind is recorded as
    `"negligible-rate"` with its value and field.
  - The energy binding goes through the species: a response given a rate that
    declares `plasma_energy` stores its species with that energy as its potential
    (`bind_rate_energy`), so no consumer of `response.potential` changes and a rate
    replaced by its table keeps the energy. A rate for another species is refused.
  - Without a reference field the constant is the weak-field limit, optmat's
    `energy_per_event(0)`. With one it is clipped to the tabulated fields above the
    threshold, up to the maximum field. A rate given explicitly keeps its own
    reference.
  - A **custom species under a selection** is a name solprop does not ship, `Gas.Air`,
    or a built-in name whose ionisation data differ from the registry's: it is refused
    by name, and takes an explicit rate by label.
  - The carrier is `selection.carrier_wavelength` when set, otherwise the run's
    `lambda0` (as 1c decided).
  - The D12 and maximum-field warnings are `MaterialRangeWarning`s, once per run; no
    warning class was added. A field is linear when its circular share
    `|S3|/S0` at the peak is below 10⁻⁶. A constructor of the capillary route launches
    one scalar pulse in one mode, linear at every node, so the D12 check runs where the
    polarisation can be otherwise: in `with_input`, from the new input's field over the
    axis and the nodes, and on the free-space routes (`_from_spatial`), from the
    pulse's Jones vector, the peak being the physical intensity's maximum (a spectrum
    given as an array is taken as linear). `with_input` keeps the reference field and
    the table range of the run it derives from, and checks the new input's peak
    against each maximum field too.
  - `linear_cycle_average` (`nonlinear/envelope_plasma.py`) exposes the envelope's own
    average around a linear cycle, and the envelope's adaptive rule is seeded with a
    rate's `maximum_field` as with a table's `field_max`.
  - `parameters` carries the energy convention but not the reference field; the table
    key still varies with the energy through `rate_species` (Q15).
  - `policies["ionisation"]` is hashed (a changed reference field is another
    computation); `peak_fields` is a measurement.
- **Verification:**
  - New tests (fast and medium), all against direct calls of the same optmat model:
    the rate is `min(instantaneous_rate, MAX_RATE)` bit for bit above the threshold for
    He, Ar, Xe, N₂, O₂ and CH₄, zero below and even in the field; the threshold
    brackets the negligible rate to 10⁻¹⁰; the hold above the maximum field and its log
    record; a table within twice its tolerance of the direct rate, stopping at the
    maximum field; `I = L/W` from `energy_loss_rate`/`instantaneous_rate` to 10⁻¹² for
    Ne, Ar, Kr and Xe, and one-channel models (He, H₂, N₂, O₂, CH₄, N₂O) exact; the bound
    against 20 001 direct evaluations (Ar, Xe); the envelope's cycle average at linear
    polarisation against `cycle_averaged_rate` within optmat's stated 3 % (Ar and N₂
    from 2 × 10¹⁰ V/m, He from 5 × 10¹⁰ V/m, up to the barrier-suppression field;
    measured within 1.8 %); statuses (SF₆ warns once per context; comparison forms
    silent), carriers below 350 nm (once), an override's digest, `base` and cache key,
    a `from_parameters` species, every refusal; the selection's rate per species and its
    refusals; the capillary's binding to the input's peak on field and envelope grids
    and a given table range; `with_mixture`; the manifest's entries; ten fixed steps of
    an argon capillary on `numpy_ref` under a selection, field and envelope, writing
    HDF5, with the largest field inverted from the stored peak rate; a run past a
    model's maximum field warning once at construction and once after.
  - **Pre-deletion parity** (41 fields from 2 × 10⁹ to 8 × 10¹⁰ V/m at 800 nm, with
    solprop's energies and occupancies, below `MAX_RATE`): `moadk_zhao_2010` and
    `moadk_tong_2002` against `IsotropicMOADK` for H₂, D₂, N₂ and O₂, 7.0 × 10⁻¹²;
    `ppt_talebpour_ilkov` against molecular `PPT` for N₂ and O₂ at occupancies 1 and 2,
    4.5 × 10⁻¹³; `adk_hydrogenic` against potential-only `ADK` for H₂, D₂, CH₄, SF₆ and
    N₂O at occupancies 1 and 2, 6.9 × 10⁻¹² (CH₄ at occupancy 2 reaches `MAX_RATE` by
    8 × 10¹⁰ V/m, so the comparison is of clipped rates). The gate is 10⁻¹⁰, and every
    form passes it: nothing blocks Phase 4b's deletion.
  - **Atomic PPT against `ppt_ilkov`, a physics change.** With solprop's energy and
    PPT's Stark shift, dipole factor, occupancy of two, sum over `m` and
    barrier-suppression correction all off, the two agree to 9.5 × 10⁻¹³ (He to Xe). The
    default PPT differs from `ppt_ilkov` by these factors, at 5 × 10¹³, 10¹⁴ and
    3 × 10¹⁴ W/cm²: `ppt_ilkov`/PPT = 0.60, 0.65, 0.80 (He); 0.69, 0.80, 1.15 (Ne);
    1.47, 2.36, 7.67 (Ar); 0.89, 1.14, 2.14 (Kr); 1.17, 1.68, 4.12 (Xe). The occupancy
    contributes 0.5; the Stark shift 1.04–1.48 (He, Ne, Ar, which have
    polarisabilities); the dipole factor 1.02–3.52 (Ar's the largest); `m = 0` alone
    0.90–0.98; dropping Tong and Lin's correction 1.13–9.18 (Xe's the largest); the
    energy difference 0.988–1.002. optmat's recommended models differ again (§9.1).
  - Costs (loaded Mac, same process): an `OptmatRate` builds in 20–30 ms for an atom
    and 0.1–0.2 s for a molecule, mostly the threshold search; a rate table of argon,
    1025 knots, 12 ms; ten fixed steps of the argon capillary 2.3 s against 0.7 s
    without a selection on the field grid, and 12 s against 7 s on the envelope grid,
    the difference being optmat's resolution and the manifest.
  - `uv run poe check` on the branch: lint, format, types, docstrings and spelling
    clean; 12 008 tests passed and 30 skipped; the strict docs build succeeds. The
    first run failed `test_the_layers_figure_draws_the_whole_package` (the new module
    was not in the structure figure, now drawn) and
    `test_nothing_public_in_a_home_module_is_forgotten` (the two new helpers of `nonlinear/ionisation.py`, now excluded from
    `solprop.sol` with the other rate helpers). The two scan-queue tests timed out once
    under a loaded `-n auto` lane and pass alone.
  - Targeted coverage, the default lane with `--cov=src/solprop --cov-branch`: 100 %
    line and branch coverage of every module changed here (`nonlinear/optmat_rates.py`,
    `ionisation.py`, `mixtures.py`, `responses.py`, `envelope_plasma.py`,
    `translate/ionisation.py`, `driver/simulation.py`, `mixtures.py`,
    `material_output.py`, `physdata/_optmat.py`, `selection.py`), without new pragmas.
    A file-path source (`--cov=src/solprop/nonlinear/optmat_rates.py`) collects no
    data with this pytest-cov; the package source was used and read per module.
  - Not run: `poe test-slow` and `poe cov`, which run at the SEP's milestones; no
    backend or kernel changed.
  - After the physics review (four should-fixes, no blockers): an argon run below the
    maximum field records no "above its maximum field" entry, on field and envelope
    grids; a known envelope peak at 0.90 and 0.95 of the maximum field is recovered to
    10⁻⁸ for Ar, Xe, He and N₂, inverted through the envelope's own average of the
    table, and the ten-step envelope run's report matches the run's cycle integrals to
    10⁻⁹; `with_input` warns for a new input above the maximum field and keeps the
    reference field; a circular free-space pulse at a Keldysh parameter above one
    warns, a linear one and an array do not; the envelope's sources of a raw rate at
    1.01 and 2 times the maximum field (He, Ar) and at 1.0001 times (Ar) converge.
- **Sources checked:** none new. Every number is optmat 0.1.1's, read from its public
  functions; its module docstrings were read for the averaging consistency (3 %)
  and the maximum fields. The yield comparisons are quoted from §4.6 and optmat's
  report [4], not re-derived.
- **Limitations:**
  - The diagnostic sees the save planes only, and above the maximum field reports a
    lower bound.
  - An elliptical input given to a free-space route as an array is taken as linear.
  - The envelope's reference quadrature (2 × 10⁻¹⁰) cannot integrate optmat's raw
    helium rate within a few parts in 10⁴ of its maximum field: optmat's instantaneous
    rate carries about 3 × 10⁻⁹ of rounding noise from its averaging's finite
    differences. A run integrates its smooth table, which converges; only a direct
    `EnvelopePlasmaResponse(..., OptmatRate(...)).sources()` there raises.
  - The first resolution in a process builds optmat's whole rate registry (1.8 s on an
    idle Mac, up to 15 s under load).
- **Open questions:**
  - Q12. What should a rate be above its model's maximum field? The hold is C⁰ and
    leaves a corner a transverse quadrature resolves slowly; a power-law continuation
    with the slope at the maximum field would be C¹ but is invented physics; optmat
    could instead define its rate there. The turn-over of the instantaneous rate a few
    per cent above the maximum field, and the rate's 3 × 10⁻⁹ rounding noise, are
    upstream findings to pass to optmat.
    *Resolved* upstream: optmat 0.2.0 defines the rate's hold above the maximum field
    ([the pin moved to optmat 0.2.0](#the-pin-moved-to-optmat-020-before-phase-4a)).
  - Q13. Should the output layer write a post-run annotation, so that the file carries
    the peak-field diagnostic as well as its inputs? *Decided* by John Travers on
    2026-10-02: yes, outside the configuration hash ([pre-defaults fixes](#pre-defaults-fixes--q3-q5-q6-q7-q8-q13)).
  - Q14. Should the free-space routes find the input's peak field and bind the
    reference field, as the capillary route does?
    *Decided* by John Travers on 2026-10-02: yes; implemented in [Phase 4a's first
    part](#phase-4a-first-part--the-defaults-switched).
  - Q15. The rate-table key includes the constant energy (through `rate_species`), so an
    energy scan rebuilds a table per point (about 10 ms each). Should an `OptmatRate`'s
    table key drop it?
    *Decided* by John Travers on 2026-10-02: left as it is.
  - Q16. Should the D12 check also cover inputs given by `with_input` and the
    free-space routes' elliptical pulses? *Resolved* after the physics review: it does,
    from the field on the mode-set route and from the Jones vector on the free-space
    routes.
- **Handoff:**
  - **Merge.** 1c also adds `ResponseSelection.selection` and a `selection=` keyword on
    `responses_for`, and 1b edits `responses_for`'s Kerr lines, `_policies()` and the
    single-gas capillary path; the keyword and the field are the same in substance.
    This branch adds `policies["ionisation"]` beside `_policies()`, not inside it.
    *Merged with 1b* (main at `8e06f3d`): `_policies(carrier=, gnlse=)` and the
    `ionisation` key side by side, `MaterialManifest.electronic` beside `peak_fields`,
    `material_manifest(peak_fields=, carrier=)`, and `_capillary_responses(selection=,
    kerr_chi3=)`, so the single-gas capillary under a selection takes 1b's Kerr
    coefficient and this phase's rate.
  - **Phase 2.** The kernel arrays are unchanged (one energy per population, a
    rate table of schema 2), so the backends need no change; qualify an argon and an N₂
    run under a selection on `numba_cpu` and Kokkos CPU, field and envelope.
  - **Phase 4a/4b.** The parity above holds at 10⁻¹⁰ for every molecular form and for
    `ppt_ilkov` without PPT's corrections, so `ADK`, `PPT` and `IsotropicMOADK` can be
    deleted once the defaults switch; `_default_rate`'s legacy branch and
    `MOADK_DEFAULT_GASES` go with them. `bind_rate_energy` stays.

### Phase 1c — gas Raman responses (NumPy)

- **Scope:** §11's Phase 1c, 2026-10-01, implemented by Opus 5.5 (AI) for John
  Travers, on `feature/sep-0003-optmat-materials-1c` from `main` at `368ad17`, in
  parallel with 1b and 1d, and merged after them. Everything is **opt-in**: a run, a
  `responses_for` call or a `response_for` call without a selection is unchanged, and
  so is one whose populations are solprop's own species.
- **Delivered:**
  - `src/solprop/nonlinear/optmat_raman.py`: `RamanGroupLines` (a group's undamped row
    $\sum_k A_k\sin(\Omega_k t)$, one tensor family), `GroupDephasing` (a
    `Dephasing` whose decay rate is optmat's own group law, finite at vacuum),
    `OptmatRamanResponse` (a `RamanResponse` with the composition's `labels`, its
    `assembly` and `material_records`), `RamanAssembly`, `RamanShift`,
    `COMPOSITION_TOLERANCE`, and the builders `mixture_raman` (a mixture's whole
    composition) and `gas_raman` (one gas).
  - `nonlinear/mixtures.py`: `responses_for(..., selection=)` takes the composition's
    one response when the populations are optmat-resolved; `raman_temperature` takes a
    composition's labels (weight: their summed density) and `widths=`.
  - `nonlinear/raman.py`: `response_for(gas, selection=, density=)`, the gas path only.
  - `translate/responses.py`: `_raman_density`. A composition's leaf is driven and
    damped by the **summed density of its labels**: one amplitude table and one
    decay-rate path per term, the kernel contract unchanged (kernel schema 15).
  - `driver/simulation.py`, `driver/mixtures.py`: `prop_capillary` and
    `prop_waveguide` under `materials=` take optmat's response for a single gas and for
    a mixture; `ResponseSelection.selection` keeps it across `with_mixture`.
  - `driver/material_output.py`, `physdata/selection.py`: the policies
    `raman_reference_temperature` and `raman_terms` (hashed), and
    `MaterialManifest.raman`, each response's measured assembly (not hashed).
    `physdata/_optmat.py`: `record_outside`, a range record that is not one of
    wavelength.
  - Tests: `tests/nonlinear/test_optmat_raman.py` and
    `tests/driver/test_optmat_raman_runs.py`.
  - Pages: the Raman manual's new part "Raman responses from optmat"; pointers in the
    materials manual; the rebuild cost in `docs/dev/benchmarks.md`; `CHANGELOG.md`.
- **Deviations from the proposal:**
  - The densities are declared to the response's range check, and its temperature
    checked, when the response is assembled, which is inside the run's material
    context, rather than during the translation proper. The densities are the run's
    path at its breakpoints and 65 positions, so the check happens once per run either
    way, and a hand-built response warns where it is built.
  - `MaterialSelection.raman` takes `"recommended"`, `"provisional"` or an optmat
    response object. §5's "or an optmat response name" is refused by name: optmat
    names a response by its gas or composition, which the run already supplies.
  - Under a selection, `raman=True` for a noble-gas single fill raises a
    `MaterialError` naming the atom, where solprop's own route raises a `DriverError`.
- **Decisions:**
  - **One response per mixture.** Every labelled population is in the composition,
    Raman-active or not; the labels select only whose lines are kept. The leaf carries
    the first selected Raman-active label (a label the kernel contract requires), and
    its density table is the sum of all the mixture's labels. A single population's
    response takes the run's one density, as before.
  - **The composition** is the populations' density ratios, the ratio of their sums
    over the run's probes, with duplicate labels merged and an all-vacuum label left
    out; a named air composition enters through its fractions, so the response's
    optmat identity is `N2+O2+Ar`. The partial-pressure rule keeps a gradient fill's
    fractions proportional only to the gases' differing non-ideality (argon's share of
    dry air moves by $1.6\times10^{-3}$ of itself between 10 bar and vacuum, nitrogen's
    by $7\times10^{-5}$), so "proportional" has a declared tolerance,
    `COMPOSITION_TOLERANCE`, 1 % relative in any constituent's fraction; the measured
    spread is recorded.
  - **Partners.** optmat decides which pairs it has curated: solprop asks for the
    composition and, only when optmat refuses it, tries each Raman-active constituent
    against each partner to name the pairs. A population optmat does not resolve (a
    supplied species, `Gas.Air`) is refused beside a Raman-active molecule, since
    optmat has no broadening by it; a mixture of solprop's own species beside a
    selection keeps the per-component route, taken as it was built.
  - **Decay law.** `GroupDephasing` subclasses `Dephasing`, so the translation and the
    backends take it unchanged, and calls optmat's `RamanGroup.decay_rate` through a
    live handle that is not recorded; its recorded fields are the law in Hz and the
    Doppler cap.
  - **Tensor rows.** Isotropic amplitudes go to an isotropic term and anisotropic ones
    to a rank-two term unchanged, verified against Hellwarth's tensor (below). The rows
    are split for every run, scalar ones included, as §4.4 has it; all-zero rows are
    dropped (N₂: 43 groups, 72 rows; dry air: 82 groups, 126 rows).
  - **Shift.** "Constant density" is every population's path constant. The reference
    density is $\int N^2\,dz/\int N\,dz$ over the run by quadrature; the frequency error
    is $\max_g|\sigma_g|$ times the largest departure from it at the probes; the
    manifest multiplies it by the run's window, $2\pi/\Delta\omega$ of its grid.
  - **Range warnings** are one per response and kind (density, temperature) and run,
    the messages of both ends of a range joined, and recorded in the run's log as
    states outside, beside the bands outside. A path that falls to vacuum between two
    probes covers every density between zero and its last positive probe by
    continuity: that interval is declared too (the smallest positive double), and the
    record's lower bound is zero, "to vacuum" (after the physics review: an H₂ or
    dry-air gradient from 10 bar to vacuum had declared only its densities above
    1.1 amagat, and missed the Dicke regime and N₂'s lower limit).
  - **Carrier.** `MaterialSelection.carrier_wavelength`, when set, is the run's
    deliberate choice and is used as it is; otherwise the grid reference.
  - **Options.** `MolecularOptions.rotation`/`vibration` select bands (`"rotation"`
    against the vibrational modes), `True` is refused by species when the band is
    absent, a non-default `min_j`/`max_j` is refused (optmat keeps every level above
    its population floor), and the species of a composition must agree on an explicit
    temperature. A `Component.dephasing` becomes optmat's `width_law`
    $(c, 0, \beta, 0)$ for every band of its species; a law with a Dicke term is
    refused, since optmat caps that term at a Doppler width the legacy law lacks.
  - **N₂O** is refused unless `raman="provisional"`, then warns once. The capillary's
    default `raman=None` keeps solprop's rule, so N₂O stays off by default even then.
  - **One `raman` field, two materials.** `MaterialSelection.raman` names a gas run's
    Raman response here and the silica bundle on a fibre's GNLSE (Phase 1b), and its
    policy names are one tuple, `physdata.selection.RAMAN_POLICIES`, which both
    paths use: `"recommended"` is the curated default in both, and `"provisional"`
    a gas policy, which also admits a gas's provisional bands and which
    `SilicaBundle.resolve` refuses by name (it mapped it silently to the default
    bundle before the merge). A gas run refuses a silica bundle's name (`"tensor"`,
    `"lin_2006"`, ...) or a bundle object, saying it applies to the GNLSE only; a
    silica run refuses a gas response object, which is not a bundle.
  - **The single-gas capillary** under `materials=` builds its Raman response from
    the resolved fill, as a mixture does, so `_capillary_responses` no longer carries
    the registry's response there; its Kerr and rate stay that path's.
- **Verification:**
  - `uv run poe check`: lint, format, types, docstrings and spelling clean; 11 992
    tests passed and 30 skipped, with two failures under the fully loaded `-n auto`
    lane: the structure figure, which did not yet draw the new module (fixed, and the
    figures regenerated), and `test_two_controllers_share_claims_and_one_collector`,
    which timed out as on `main` and passes alone. The strict docs build succeeds.
  - `uv run poe cov`: 100 % line and branch coverage of the whole package, without new
    pragmas, with 11 995 tests passed and 30 skipped (the scan-queue test included).
  - `uv run poe test-slow`: 137 passed and 1 skipped (24 min on the loaded machine).
  - What the new tests check, each against optmat 0.1.1's own evaluation of the same
    response or an independent statement:
    - the summed terms of N₂, O₂, D₂, CH₄ and SF₆ against `response(t, N)` at two
      densities, and the isotropic and rank-two rows against optmat's two shares, to
      a summation bound ($4\,\epsilon$ per line times $\sum|A_k|$);
    - causality of the response, the rows and the sampled kernel;
    - $\Gamma = \pi\Delta\nu$ for every group of five gases (to $10^{-14}$), the
      vacuum limit $\pi(c + \Delta\nu_D)$ of H₂'s groups and zero for N₂'s;
    - the vector polarisation against Hellwarth's, with optmat's $a$ and $b$, by a
      direct causal sum ($10^{-11}$ of its largest value), and a probe across a
      rotational response seeing $-\tfrac12$ of one along it ($10^{-9}$);
    - dry air's response against optmat's for the measured fractions, driven by the
      total density; duplicate labels merged; the curated partners accepted; H₂/He,
      N₂/He, CH₄/Ar and H₂/N₂ refused naming the pair; atoms alone and vacuum building
      none; non-proportional populations refused, and a 10 bar → vacuum air gradient
      admitted with its spread recorded;
    - H₂ exact at constant density, refused on a gradient, and under the
      reference-density approximation the reference $\int N^2/\int N$, the recorded
      frequency error, and the response's departure from optmat's direct evaluation
      along the path within the bound;
    - the density and temperature range warnings, once per run and recorded; N₂O
      refused, then admitted once with its status warning; overrides and supplied
      responses; the term flags, the options' conflicts and the gas/silica meaning of
      `raman`;
    - in the translation, a dry-air gradient's leaf: its amplitude table against the
      summed density and every decay table against the group law at it, to the
      tables' tolerance; a gradient to vacuum, with finite decay paths there;
    - ten fixed steps of an N₂ and of an N₂/O₂/Ar capillary on `numpy_ref` under a
      selection, finite, with their manifests; a hydrogen gradient's recorded phase
      bound over the run's window; `with_mixture` keeping the response; and, without a
      selection or with solprop's own species, responses identical to today's.
  - **Phase 1 milestone**, on the tree merged with Phases 1b and 1d and the physics
    review's fixes (a gradient's vacuum tail declared; one `RAMAN_POLICIES`):
    `uv run poe check` clean, 12 192 tests passed and 30 skipped, the strict docs
    build succeeds; `uv run poe cov` 100 % line and branch coverage of the package,
    12 192 passed; `uv run poe test-slow` 137 passed and 1 skipped.
  - Not run: no backend other than `numpy_ref`, which Phase 2 qualifies; no
    full-length propagation.
- **Sources checked:** none new. Every reference value is optmat 0.1.1's own
  evaluation: `GasRamanResponse.response` (parallel, isotropic, anisotropic, `a`, `b`),
  `fwhm`, `decay_rates`, `groups()`, `check_densities`, `status`, `with_parameters`.
  The ranges quoted in the manual (N₂ 0.054–2.09 amagat and 293–300 K, H₂ above 0.941
  amagat, D₂ above 0.61) are optmat's at 294 K and 800 nm, as its responses report
  them. optmat's physics (the line strengths, widths and their gains) is not checked
  here.
- **Limitations:**
  - On a density gradient the kernel is rebuilt at every stage, at a cost that scales
    with the rows: 6 ms for N₂ and 10 ms for dry air at $2^{14}$ samples (benchmarks).
    SEP 10 removes it.
  - H₂ on a gradient needs the reference-density approximation; its bound is recorded,
    not reduced.
  - The densities and the composition are sampled at the run's breakpoints and 65
    probes: a non-monotone path's extreme between probes is not declared.
  - optmat's response omits the vibrational O and S branches (3 % of the anisotropic
    Q share of H₂ and D₂, 8 % of N₂'s, 19 % of O₂'s), and models no line mixing.
  - Only numpy_ref is qualified here; the arrays are the contract's, and Phase 2
    qualifies the others.
- **Open questions:**
  - Q17. Should the capillary's default `raman=None` turn N₂O's response on under
    `MaterialSelection(raman="provisional")`, where today it stays off with solprop's
    "incomplete data" warning and needs `raman=True`?
    *Decided* by John Travers on 2026-10-02: yes; implemented in [Phase 4a's first
    part](#phase-4a-first-part--the-defaults-switched).
  - Q18. Is 1 % the right composition tolerance? A per-species leaf, each species'
    lines divided by its fraction and driven by its own density, would follow the
    non-ideal drift exactly, at the price of one leaf per Raman-active species and a
    second amplitude rule beside §4.4's total-density one.
    *Decided* by John Travers on 2026-10-02: 1 % stays.
  - Q19. A scalar run does not need the tensor split: merging each group's two rows
    would cut its rebuild by about 40 % (N₂: 72 rows to 43), at the price of
    scalar-only terms. Should the adapter merge them when the run is scalar, or wait
    for SEP 10?
    *Decided* by John Travers on 2026-10-02: accepted as it is; the merge is SEP 10's.
- **Handoff to Phase 2:** backend parity must cover
  - one leaf with many rows (72 for N₂, 126 for dry air), mixed isotropic and rank-two,
    on scalar and vector field and envelope runs;
  - a composition's leaf, whose amplitude table is the sum of several labels' densities
    and whose label is the first of them, beside the per-population Kerr terms;
  - a density gradient, which rebuilds the damped spectrum at every stage, including
    rows whose decay rate is zero at vacuum (N₂, O₂: undamped up to the taper) and
    rows with a finite vacuum rate (H₂'s Dicke cap);
  - H₂'s constant-density rows and its reference-density rows;
  - the memory of `n_terms × 2 n_to` rows (126 × 2¹⁵ doubles is 33 MB) and the
    rebuild cost, measured on an idle machine. Several Raman leaves already run on
    `numba_cpu` and Kokkos (it reads `raman_terms[0]` only for the leaves' uniform
    `thg` flag), so a composition leaf beside other leaves needs no new path either.


### Phase 2 — backend qualification (numba_cpu and Kokkos CPU)

- **Scope:** §11's Phase 2, 2026-10-01, implemented by Opus 5.5 (AI) for John Travers,
  on `feature/sep-0003-optmat-materials-2` from `main` at `43cdef9` (Phase 1
  complete). The translated problems of Phases 1a–1d are qualified on `numba_cpu` and
  on a freshly built Kokkos CPU backend against `numpy_ref`. No source file of the
  package changed; nothing user-visible changed, so `CHANGELOG.md` has no entry.
- **Delivered:**
  - `tests/optmat_parity_cases.py`: fifteen runs under `MaterialSelection()`, built
    through `Simulation.from_capillary` and `Simulation.from_gnlse` (one assembled
    directly), each ten fixed steps: argon with optmat's rate on field and envelope
    grids, and launched above the rate model's maximum field (`ar_hold`); nitrogen
    on a 1.5 → 0.3 bar gradient with plasma and its 72-row leaf, field and envelope;
    dry air's 126-row composition leaf on a pressure gradient to the vacuum floor and
    on density paths to exact vacuum; hydrogen at constant density and to exact vacuum
    under `raman_shift="reference-density"`; a helium/argon Kerr mixture; argon in
    optmat's silver; nitrogen to exact vacuum on two polarisations, field and envelope;
    the GNLSE with the silica bundle; the tensor bundle on two polarisations. Every
    other capillary is walled by optmat's silica, the default cladding under the
    selection.
  - `tests/optmat_parity.py`: the ladder (one linear application, one nonlinear
    evaluation, one step, eight steps, every statistic of the run's spec, and the
    driver's complete run through `Simulation.run` and the output layer, with its
    planes, per-plane statistics and periodic rows) and its bounds.
  - `tests/backends/test_optmat_plasma_parity.py`, `test_optmat_raman_parity.py`,
    `test_optmat_material_parity.py`: the ladder on every installed compiled backend;
    a check that each rung can see the term it qualifies (the plasma's or the Raman
    leaf's share of the first stage, against the run without it, above a thousand
    bounds); that the hold above the maximum field is inside the run; lockstep batches
    equal to their members, bitwise; the leaves' row counts, tensor weights and vacuum
    decay paths; Kokkos's memory ledger for the rows. A Kokkos case skips cleanly
    without the backend, through the existing fixtures.
  - `docs/dev/backends.md` (§9: the optmat paths' status), `docs/dev/benchmarks.md`
    (costs and memory).
  - **Kokkos:** a fresh local-profile build of `solprop-kokkos` `main` at `57dc287`
    (Release, Accelerate BLAS, `HYBRID` GEMM, native architecture, OpenMP, FFTW, kernel
    schema 15, built beside solprop `43cdef9`), wheel
    `solprop_kokkos-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`, SHA-256
    `6f84f56a…6b0813`, installed into the branch's environment only. No native change.
- **Parity measured** (peak-normalised deviation from `numpy_ref`; the bounds were
  fixed before the first measurement, and none was changed after it). Every field rung
  of every case on both compiled backends is at most 2.1 × 10⁻¹⁵ (the first stage of
  the nitrogen gradient on Kokkos) against bounds of 3 × 10⁻¹¹ to 9 × 10⁻⁷; every
  statistic at most 6.1 × 10⁻¹⁴ (`quadrature_error`). The complete runs take the same
  ten steps. Kokkos refuses dry air on independent density paths by name, as it does
  any mixture whose operator took the z table, and runs it on a shared pressure path.
- **Deviations from the proposal:** none in the backends. The cases differ from §11's
  list in three places:
  - **Exact vacuum** is reached by density paths. An endpoint pair keeps a numerical
    floor (1.5 × 10⁻⁴ bar for dry air, where the decay rates fall to 3 × 10⁵–
    1.8 × 10⁶ s⁻¹), and a linear pressure profile to zero is refused by the
    translation, its density table rounding to about −10⁹ m⁻³ at the end (Q23). Dry
    air on three density paths takes the z table, so the exact vacuum of a
    composition leaf is qualified on `numba_cpu` alone; nitrogen's and hydrogen's single
    paths run on all three.
  - **The tensor bundle on a vector route** is assembled directly: no driver takes it
    there (Q8), and the general modal route takes only `SilicaTensorResponse`.
  - **The gradient to the floor** keeps a fixed wall index, 1.45: with optmat's
    silica its translation takes 19 s (Q22).
- **Decisions:**
  - **The bound.** One stage of two backends is within
    `256 eps (n_to + n_nodes + n_rows)`, peak-normalised: the existing bound of
    `tests/backends/test_vector_raman_routes.py` per fine sample and node, plus one
    rounding per Raman row and sample of the kernel's synthesis. A rung of `k` stages
    takes `k` of it, times `exp(Phi)` with `Phi = z ||N(a_0)|| / ||a_0||` measured on
    the reference. A field statistic takes four times its rung's bound, and a plasma
    statistic `2 S` times, with `S = d ln W / d ln F` where the rate is a thousandth
    of the run's peak.
  - **The linear rung** is one step of the launch scaled by 10⁻²⁰, so every pass runs
    and the cubic terms are below 10⁻⁴⁰ of the linear one.
  - **The complete run** is the driver's, on a shallow copy of the simulation whose
    settings name the backend, so the translation is shared.
  - **Three modules**, by response family, so that `--dist loadfile` spreads them; the
    tests are in the medium tier (a real ten-step run each). Batches are compared on
    the compiled backends only, where they are batches.
  - **No native change.** Neither compiled backend assumes a Raman term count: the
    host backends read the problem's rows, and Kokkos sizes `raman_undamped`
    (`n_terms × 2 n_to`) and `stage_raman_rates` (`n_stages × n_terms`) from the
    problem, which a test asserts. Kokkos's slower rebuild at one thread (below) is a
    cost, not a contract fault, and is left to SEP 10 and the Kokkos side (Q24).
- **Verification:** the three modules pass on `numpy_ref`, `numba_cpu` and Kokkos CPU
  (53 tests: 20, 24 and 9), serially and under xdist. The **milestone lanes** on the
  final tree with the fresh wheel installed, each with `SOLPROP_KOKKOS_THREADS=1`
  (and `OMP_PROC_BIND=false` but for the Kokkos-selected lane): `uv run poe check`
  clean (lint, format, types, docstrings, spelling; 12 245 passed and 30 skipped; the
  strict docs build succeeds); `uv run poe test-slow` 137 passed and 1 skipped
  (16 min on the loaded machine); `uv run poe cov` 100 % line and branch coverage of
  the package (393 kernel-body tests, then 12 245 passed and 30 skipped);
  `SOLPROP_TEST_BACKEND=kokkos uv run poe test-slow` 138 passed (16 min). No
  scan-queue timeout occurred.
  Not run: GPU builds (the standing device policy); full-length propagations.
- **Costs and memory** (`docs/dev/benchmarks.md`, same-process A/B at one thread on
  the loaded development machine): over ten steps, `numba_cpu` and Kokkos are
  2.3–3.8 times faster than `numpy_ref` on the plasma, Kerr and GNLSE cases and 23–30
  times on envelope plasma; with 72 or 126 rows on a gradient the stage's Raman
  rebuild dominates, and they are 1.1–1.4 times faster at `n_to = 2048`. At
  $2^{14}$ the host rebuild of dry air takes 8.6 ms and Kokkos's 19–24 ms at one
  thread (7.6 ms at four), so there Kokkos runs ten steps 0.63 times as fast as
  `numpy_ref`. Kokkos holds the rows, 33 MB for dry air at $2^{14}$ (its ledger
  88 MB); the host backends read them from the problem. Translation, in fresh
  processes: an N₂ gradient 1.7 s under the selection (0.12 s with a fixed wall,
  0.07 s without a selection); dry air to the vacuum floor 19.4 s (4.0 s and 0.85 s).
- **Sources checked:** none new; no coefficient was transcribed.
- **Limitations:**
  - Device builds are not qualified on optmat's runs.
  - A composition leaf at exact vacuum is qualified on `numba_cpu` only (above).
  - The cases are short and small (`n_to` 512 to 2048 in the tests; the costs also at
    $2^{14}$); the bounds are those of operation counts and are 10⁴–10⁸ above the
    measurements, which are what bound the difference in practice.
  - The three modules add about 2 minutes of serial time to the default lane, the
    reference's envelope plasma about half of it.
- **Open questions:**
  - Q22. optmat's silica wall is evaluated afresh at every sample of the frame tables
    along z, though the wall index does not depend on z: a gradient's translation is
    14 times as long with it (N₂, 1.7 against 0.12 s) and dry air's to the floor
    takes 19 s. Should the capillary modes evaluate the wall once per frequency grid
    before Phase 4a makes optmat's silica the default wall? optmat's gas
    polarisabilities make a selection's gradient 2–5 times the registry's even with a
    fixed wall.
    *Resolved* in the [Q22
    follow-up](#phase-2-follow-up--q22-an-optmat-material-evaluated-once-per-frequency-array).
  - Q23. A linear pressure profile to exact vacuum is refused by the translation with
    or without a selection: the population's density table rounds to about −10⁹ m⁻³
    (10⁻¹⁶ of its scale) at the end, and the domain check admits no negative value.
    Should the check allow the table's own rounding, or the table be clipped at zero?
    A mixture could then reach exact vacuum on a shared path, which Kokkos runs.
    *Resolved* after Phase 2: `translate.linop.profile_table` tabulates a path's exact
    zero at an interval edge as an exact zero, within the table's tolerance, and the
    domain check is unchanged.
  - Q24. Kokkos rebuilds the Raman kernel with one exponential per row and sample, two
    to three times the host's time at one thread. Should solprop-kokkos vectorise it
    before Phase 4a, or is SEP 10's low-rank representation the fix?
    *Decided* by John Travers on 2026-10-02: accepted; the fix is SEP 10's.
- **Handoff to Phase 4a:**
  - Phase 4a first waits for optmat's ionisation-rate work and the pin bump that
    follows it: digest review, re-run of
    `tests/data/measure_sep3_changes.py`, then the attribution and the switch. The
    parity suite then re-runs unchanged: it compares backends, and hard-codes only the
    leaves' row counts (4, 72, 126), the wall models' names and that 1 mJ exceeds
    argon's maximum field, which a new optmat release could move.
  - The backends need no change for the switch: what a default selection produces is
    what these cases carry. The parity suite is the gate for any further case (add it
    to `tests/optmat_parity_cases.py`).
  - Before the defaults switch, decide Q22: every gradient capillary would otherwise
    pay optmat's wall evaluation at translation.
  - Rebuild the Kokkos wheel when the kernel schema moves; this one is schema 15.

### Phase 2, follow-up — Q22: an optmat material evaluated once per frequency array

- **Scope:** Phase 2's Q22, 2026-10-02, implemented by Opus 5.5 (AI) for John Travers,
  on `feature/sep-0003-optmat-materials-q22` from `main` at `a88dbce` (Phase 2
  merged). No backend and no contract change; the translated problems are bitwise
  unchanged.
- **Found:** the frame tables (`translate/linop.py::_frame_tables`) read the
  reference mode's `neff` and `dneff_domega` at the reference frequency at every
  sample along z, and each read of a `MarcatiliMode` samples its fill and its wall
  afresh (`sampled_terms`). The default modal route of `_modal_linear` ran outside
  the sampling scope that the wall and taper assemblies already use. Each sample cost
  four optmat evaluations of the silica wall (`permittivity` and
  `dpermittivity_domega` each read the index and its derivative), the gas
  polarisabilities and the derivative, and the dilute-scaling check of every
  population. Dry air to the vacuum floor took 3399 samplings, 20 of its 28 s under a
  profiler in the wall; the fills had no per-frequency cache, `_sampled_susceptibility`
  sampled every time.
- **Delivered:**
  - `src/solprop/physdata/_optmat.py`: `EvaluationCache`, an adapter's last
    `EVALUATION_CACHE_SIZE` (8) evaluations, least recently used evicted, keyed on
    the quantity and the frequency array's shape, type and bytes. A miss evaluates
    under the policy of `translated` (refactored into `_captured` and the
    re-emission) and stores read-only copies with the warnings it captured; every use
    returns writeable copies and raises the stored warnings again through `_reemit`.
    A copied or pickled cache is empty.
  - `src/solprop/physdata/glasses.py`: `OptmatGlass` evaluates through its cache.
  - `src/solprop/physdata/sellmeier.py`: `OptmatSusceptibility` likewise, one entry per
    quantity (polarizability, derivative).
  - `src/solprop/translate/problem.py`: the default modal route's `build_phi` runs
    inside `_shared_sampling`, so the frame tables sample the mode once per frequency
    vector.
  - `tests/physdata/test_optmat_evaluation_cache.py`; `docs/guide/media/materials.md` (one
    sentence), `docs/dev/benchmarks.md`, `CHANGELOG.md`; the comment of the
    `air_floor` parity case.
- **Deviations from the proposal:** none; Part I does not cover the cost.
- **Decisions:**
  - **Both levels.** The cache sits in the adapters because the wall's consumers are
    many (antiresonant tubes, step-index and graded-index guides, spatial media, the
    optics), and each reads the same few frequency arrays repeatedly; the scope
    removes, for a capillary, the per-sample fill sampling and dilute-scaling check
    that remain (1.4 s of dry air's 7.5 s with the cache alone).
  - **The key is the bytes**, not the array's identity: the frame tables build a new
    one-element array per sample. A validated frequency array is always `float64`, so
    equal bytes are equal values, and the stored arrays are bitwise a fresh
    evaluation's. Eight entries cover a translation's band, reference and modes'
    samplings, and bound an adapter at eight pairs of spectra.
  - **Warnings are replayed, not suppressed.** An adapter can outlive a run (a mode or
    a species reused by the next), so a stored band raises and records what its first
    evaluation did in every context it is used in: once per run inside a run, at
    every use outside any context, as before. A transparent index's dropped loss is
    recorded at every use as before (a maximum, so unchanged).
  - **Legacy indices** (`SellmeierGlass`, `TabulatedGlass`, `CompositeGlass`) keep no
    cache: they cost microseconds. In a capillary they share the route's scope,
    which is bitwise by construction (the existing wall-sampling tests).
  - **Outside a run's context**, a direct `assemble` of a band outside a model's
    intervals now warns once per frequency vector of the frame tables rather than
    once per sample; inside a run, which every driver opens, nothing changes.
  - **The `air_floor` parity case keeps its fixed wall,** as Phase 2 qualified it; with
    the cache the wall costs it nothing, and Phase 4a re-runs the suite with the
    default wall anyway.
- **Verification:**
  - The translated problems of the N₂ gradient and of dry air to the floor, each with
    optmat's wall and with a fixed one, are bitwise equal with and without the cache
    and the scope (every array and scalar of the `KernelProblem`, 598 and 838
    leaves), and so are the runs' material logs and the warnings raised from
    construction through translation.
  - Tests: the cache's result is bitwise a direct evaluation's, for the index, its
    derivative, the permittivity forms and the gas quantities, which are kept apart;
    one optmat evaluation per band and adapter; bounded at eight, least recently used
    first; a refusal is not stored; a returned array is the caller's own; a copied or
    pickled adapter starts empty; stored bands warn and record as fresh ones do,
    once per context and at every use outside one; a transparent index's dropped
    loss and its once-per-adapter warning. A medium test translates the N₂ gradient
    with and without the change, bitwise, and asserts at most four optmat wall
    evaluations in its translation against over a hundred times as many before: the
    regression guard, a count rather than a time.
  - Timings (`docs/dev/benchmarks.md`, same-process A/B on the loaded development
    machine, median of four alternating rounds): N₂ gradient with optmat's wall
    1.75 → 0.077 s, with a fixed wall 0.10 → 0.068 s; dry air to the floor with
    optmat's wall 19.8 → 2.92 s, with a fixed wall 4.2 → 2.91 s. Without a selection
    the same process takes 0.040 s and 0.64 s.
  - The gate, `uv run poe check`, in the branch's environment, which has no Kokkos
    backend: lint, format, docstrings and spelling clean; 11 684 passed and 147
    skipped; the strict docs build succeeds. Its only errors are that environment's:
    basedpyright cannot resolve `solprop_kokkos` in nine imports of the Kokkos tests
    and probes, and the three Kokkos thread-probe modules fail to collect for the
    same reason. Line and branch coverage of the changed code is complete.
  - Not run: the Kokkos lanes (no backend or contract change), GPU builds.
- **Sources checked:** none; no coefficient was transcribed.
- **Limitations:**
  - The cache is per adapter: two adapters of one model (each `fused_silica()` call
    builds one) each evaluate once.
  - The tabulated and rectangular modes sample their fills outside the scope; the
    adapters' cache makes optmat's part of it cheap, but their dilute-scaling check
    still runs at every sample.
- **Open questions:**
  - Q22 is answered: the wall and the gas are evaluated once per frequency array, and
    a gradient capillary under the selection no longer pays the wall at translation.
  - Q25. Dry air under the selection still translates 4.5 times as long as without one
    (2.9 against 0.64 s): the tabulation of its Raman leaf's 126 decay rates along z
    (`raman_coefficients`, most of the remaining time), which depend on the density.
    Should Phase 4a accept it, or should the decay-rate tabulation be made cheaper
    first?
    *Decided* by John Travers on 2026-10-02: accepted; a cheaper tabulation is SEP 10's.
- **Handoff to Phase 4a:** Q22 no longer blocks the defaults switch. Q25 is the
  remaining translation cost of a many-row Raman leaf on a gradient.


### Pre-defaults fixes — Q3, Q5, Q6, Q7, Q8, Q13

- **Scope:** six open questions of Phases 1a, 1b and 1d that John Travers decided on
  2026-10-02, so that switching the defaults in Phase 4a does not refuse ordinary runs;
  implemented 2026-10-02 by Opus 5.5 (AI) for John Travers, on
  `feature/sep-0003-optmat-materials-4p` from `main` at `bacc49d`. Everything stays
  **opt-in**: `materials=None` changes nothing, except that an optmat index a caller
  builds and gives a lossless-only route is no longer refused (Q3).
- **Delivered:**
  - **Q3, automatic transparency.** `src/solprop/physdata/glasses.py`:
    `lossless_index`, the index a lossless-only route takes, which for an optmat index
    that is not already transparent is its transparent counterpart, one per adapter
    (held outside the dataclass fields, so neither `replace` nor the manifest's walk
    sees it); `OptmatGlass`'s dropped-loss warning is once per run inside a run's
    context and once per adapter outside one. `modes/stepindex.py`: `StepIndexMode`
    holds the counterpart in `core`, `cladding` and `_core`, and `_material` no longer
    refuses (1a-i's refusal by name is removed). `modes/material.py`:
    `IndexMaterial` holds the counterpart, so the step-index provider's material
    check, the graded-index provider (`PowerLawPermittivity(IndexMaterial(...))`) and
    the regions' Kerr conversion see a real law. The vector-field path
    (`modes/_stepindex_vector.py`) reads the mode's own counterpart.
  - **Q5, planar runs.** `driver/simulation.py`: `_selected_planar`, which replaces
    `_refuse_planar_selection`. Under a selection a slab fill of a registry gas with
    the registry's own susceptibility takes `species(gas, selection=).susceptibility`
    (an `OptmatSusceptibility`, shared by every mode), and a slab declared without a
    cladding takes `fused_silica(model="recommended")`; the grid is rebuilt on those
    modes and the medium follows it. `prop_planar`'s docstring.
  - **Q7, design tools.** `tools.py`: `materials=` on `soliton_params`,
    `capillary_params`, `gas_ratio`, `pressure_zdw` and `pressure_rdw`;
    `_resolve_fill(..., materials)` as the drivers' `capillary_fill` (a name, a
    fraction mapping, `"air"`), `_check_carrier` (a selection's other carrier refused,
    as a run refuses it), `_default_cladding` (the capillary driver's rule, copied
    because the tools may not import the driver), `_fill_chi3`/`_fill_n2` (the fill's
    Kerr coefficient at the pump and its exact $n_0$), `_design_capillary` (a
    one-bar optmat fill and the default wall, Marcatili or Vincetti) and
    `_capillary_at` (each density from the selected species' equation of state; a
    selected air at its shared pressure). `_pressure_root` now probes silently and
    re-evaluates its answer outside, and `_probing` silences `DiluteScalingWarning`
    too.
  - **Q8, the tensor bundle.** `driver/simulation.py`: `_selected_guide`, applied by
    `_from_general`: a region's `SilicaTensorResponse()` with the declared default
    parameters becomes `SilicaTensorResponse(parameters=SilicaBundle.resolve(
    selection.raman, tensor=True))`, the provider (step-index or graded-index) and the
    guide rebuilt around the same sampling; `Simulation.guide` keeps the guide so the
    manifest walks its regions. `translate/general.py`: `_raman_identity`, which
    records a bundle in the guide's identity by name, digest and fraction rather than
    by its optmat object's fields.
  - **Q13, the post-run annotation.** `output/schema.py`: `PlaneStore.annotate` and
    `annotations`, the `_write_annotation` hook, `PEAK_FIELDS_ANNOTATION`
    (`"material_peak_fields"`). `output/hdf5.py`: annotations under
    `/meta/annotations/<name>`, written into an open file or into a closed one by
    reopening it, read back by `open_file` and by an appending handler.
    `output/memory.py`: `to_file` copies them. `driver/simulation.py`: `run` writes
    the peak-field report when it is not empty.
  - Tests: `tests/physdata/test_optmat_solids.py` (the refusal test replaced by three:
    the counterpart and its sharing, everything else returned as it is, the warning
    once per run); `tests/driver/test_material_manifest.py` (the planar refusal test
    replaced by three, a general guide on an absorbing optmat core, the tensor
    bundle); `tests/test_tools.py` (six); `tests/output/test_material_resume.py` (Q6,
    three cases); `tests/output/test_handlers.py` (three);
    `tests/driver/test_optmat_rates.py` (the annotation on a file and in memory).
  - Pages: `docs/guide/media/materials.md` (lossless routes, the driver table, the bundle,
    the limitations), `docs/guide/workflows/design-tools.md` ("Designs under a material
    selection"), `docs/guide/results/output.md` ("Post-run annotations", the file layout),
    `docs/guide/media/ionisation.md`, `docs/guide/waveguides/general-modal.md`, `docs/guide/waveguides/modes.md`,
    `docs/guide/setup/spatial-decompositions.md` (planar mixtures still refused);
    `CHANGELOG.md`.
- **Deviations from the proposal:** none from Part I. §4.7's "receive an explicit,
  recorded transparent approximation within a justified domain" is met by an
  automatic one: it is recorded (the manifest's `solids`, hashed with its bound), its
  domain is the declared bound, and beyond it the run warns. It is not "silently"
  discarded, which §4.7 forbids.
- **Decisions:**
  - **Which routes.** The lossless-only routes are the step-index modes (core and
    cladding), the general guide's `IndexMaterial` laws (both providers solve real
    permittivities; `IndexMaterial` exists only for them) and the step-index vector
    fields. Hollow guides' walls, slabs and bulk media take the complex index as they
    did. A caller's own `RadialPermittivity` built on an optmat index directly, not
    through `IndexMaterial`, keeps the graded-index provider's generic refusal.
  - **Every optmat index, not only absorbing ones.** A route cannot tell an absorbing
    model from its band without evaluating it, and a structural rule keeps the hash
    independent of what has been evaluated: every optmat index on such a route is
    held transparent. A model with $\kappa \equiv 0$ (N-BK7) drops nothing, but its
    manifest entry now says `transparent` with the bound; nothing is public, and it
    was never refused.
  - **The route holds the counterpart.** `StepIndexMode.core` and `IndexMaterial.index`
    are replaced by it, so the manifest walks one object and records one solid entry
    with `transparent: true`, never the opaque source beside it. One counterpart per
    adapter keeps two modes of one guide equal (`_same_stepindex_guide` compares by
    identity) and makes their dropped-loss record one.
  - **Non-optmat complex inputs keep today's behaviour.** A complex `GlassIndex`,
    `TabulatedGlass` or the legacy hybrid silica carries no bound and no identity to
    record the approximation by, so the rule does not apply cleanly: the step-index
    modes take its real part as documented, and the general providers and the vector
    fields refuse it as before.
  - **Once per run.** Inside a run's context the warning is keyed by the model's
    digest and bound; outside every context it stays once per adapter, as 1a-i had it.
    The measurement (`transparency`) is the counterpart's maximum over its life, which
    spans every run that shares the source adapter (Q28).
  - **Planar density.** No mixture: the slab fill keeps the registry's density path
    (`DensityPath(Gas)`, pyfluids), with optmat's susceptibility beside it, $5.7\times
    10^{-6}$ above optmat's equation of state for Ar and N₂ at 294 K and $-3.4\times
    10^{-7}$ for He. Only the susceptibility (`fit/Ar`) and the default wall
    (`fit/SiO2`) are recorded; no dilute-scaling check runs, since only a mixture's
    fill measures it (Q27). Gas mixtures in planar guides remain refused, with or
    without a selection.
  - **Planar resolution rule.** A fill whose susceptibility equals the registry's
    default (`gamma(gas)`, given or not) resolves; a named registry `source`, a
    supplied susceptibility and a declared cladding are the caller's. The grid's own
    `n_x` warning is not repeated by the rebuild.
  - **Design tools.** The fill resolves exactly as `capillary_fill` resolves it and
    the default wall as `_default_cladding`; the Kerr coefficient is taken at the
    tool's pump wavelength (`wavelength`), which is the run's carrier, and a
    selection naming another carrier is refused, as a run refuses it. `gas_ratio`
    under a selection takes both gases as one-population fills (air as nitrogen and
    oxygen) for the index and $\chi^{(3)}$. A supplied `Mixture` and a caller's mode
    are taken as they are.
  - **The tensor bundle.** `selection.raman` names it (`"recommended"` the tensor
    bundle, an optmat name or object as given; `"provisional"` refused by name, as for
    the GNLSE). Default detection is by value: `SilicaTensorResponse()` and one built
    with the default `LinAgrawalRaman` both resolve. A provider of another kind is
    taken as it is. The guide's identity records a bundle by `{bundle, digest,
    fraction}`; before, `dataclasses.asdict` copied the optmat object's fields into
    the record.
  - **The manifest finds a guide's regions.** `Simulation` keeps the general guide
    (`guide=`), which enters no translation, and `material_manifest()` walks it: until
    now a general run's regions were reached only through step-index channels.
  - **Annotations.** Plain JSON by name, outside `RunMetadata` and its hash; no output
    schema increment, since no stored field changes meaning and a file without them
    reads as before. A store may be annotated after it is closed (the run closes its
    handler before the diagnostic exists); a read-only handler refuses. Only a
    non-empty report is written, so a run without optmat rates writes nothing.
  - **Q6.** No hole remains: a changed `ionisation` (argon with plasma, `ppt_ilkov`)
    or `raman` (nitrogen with Raman, an optmat override) changes the records and
    policies it resolves to, and so the hash, and a resume is refused naming
    `materials.`; a choice the run does not use (argon without plasma) is the same run,
    as 1a-ii decided.
- **Verification** (run before the process change of 2026-10-02, which defers the full
  gate to the end of Phase 4; recorded as run):
  - `uv run poe check` on the branch: lint, format, types (0 errors), docstrings and
    spelling clean; 12 278 tests passed and 30 skipped, the two scan-queue tests
    included. The strict docs build failed on one warning, a `:func:` reference to the
    private `_selected_planar` in `Simulation.from_planar`'s docstring; after the fix
    `uv run poe docs` alone succeeds, and `tests/test_seps.py` passes (92 passed, 1
    skipped).
  - Targeted coverage, the whole default lane with `--cov=src/solprop --cov-branch`
    (12 278 passed, before that docstring-only fix): 100 % line and branch coverage,
    without new pragmas, of every module changed here (`physdata/glasses.py`,
    `modes/stepindex.py`, `modes/material.py`, `driver/simulation.py`,
    `driver/interface.py`, `tools.py`, `output/schema.py`, `output/hdf5.py`,
    `output/memory.py`, `translate/general.py`); package total 97 %, the compiled
    kernels being measured only by `poe cov`.
  - Not run: `poe test-slow` and `poe cov` (no backend or kernel input changed), and
    after the docstring fix no second full gate.
  - What the new tests check: the transparent counterpart held by `StepIndexMode`
    (core and cladding) and `IndexMaterial`, shared by two modes, real index bit for
    bit optmat's; everything else returned as it is; the warning once per run inside
    a context and not again outside it; a general guide on optmat silica translating,
    its manifest's solid `transparent` with the bound; the planar fill resolved
    (shared `OptmatSusceptibility`, optmat's wall, the density unchanged, the
    translation built) and a supplied fill or cladding kept, the `n_x` warning once;
    the tensor bundle in a step-index and a graded-index guide, the kernels within
    $4\times10^{-16}$ of the declared ones, the guide's identity, a caller's provider
    and other parameters kept; a zero-dispersion pressure designed under the
    selection reproduced by four fixed steps of the run to $10^{-7}$ in wavelength
    (measured $3\times10^{-9}$), the registry's design missing it by $2.7\times
    10^{-3}$; a phase-matching pressure reproduced to $10^{-8}$ (measured
    $5\times10^{-11}$); `capillary_params`, air, mappings, `gas_ratio` and the
    carrier refusal; Q6's resume refusals; annotations through memory, HDF5 (open,
    closed, appended, copied by `to_file`), their absence in older files, their
    refusals, the hash and resume unchanged, and the run's own annotation on file and
    in memory.
- **Sources checked:** none new; no coefficient was transcribed. The numbers quoted
  are optmat 0.1.1's and solprop's own, measured here.
- **Limitations:**
  - A transparent counterpart's dropped-loss record spans every run that shares its
    source adapter.
  - A planar fill under a selection keeps solprop's density and has no dilute check.
  - The design tools' pressure searches probe up to `pressure_max` silently; only the
    answer's warnings are raised.
- **Open questions:**
  - Q26. When Phase 4a makes `fused_silica()` an optmat adapter, two
    `StepIndexMode(core=None)` channels each build their own adapter and counterpart,
    and `ModalGuide.from_step_index` refuses them as not sharing one guide (verified
    here with two `fused_silica(model="recommended")` cores). Should an
    `OptmatGlass` compare by model digest, transparency and bound, or the default be
    one shared adapter per process?
    *Resolved* in [Phase 4a's first part](#phase-4a-first-part--the-defaults-switched):
    value equality.
  - Q27. Should a planar fill under a selection take optmat's equation of state and
    measure its dilute error, which needs the slab fill to carry a species without
    becoming a mixture?
    *Resolved* in the [final fixes](#final-fixes-before-final): the slab's fill is
    carried as one species, with optmat's equation of state and its dilute-scaling check
    recorded.
  - Q28. Should a transparent counterpart's dropped loss be measured per run (in the
    run's `MaterialLog`) rather than over the counterpart's life?
    *Resolved* in the [final fixes](#final-fixes-before-final): per run, in the run's
    `MaterialLog`.
- **Handoff to Phase 4a:** nothing in these six blocks the defaults switch, but Q26
  does: decide it before `fused_silica()` returns an optmat adapter, or the
  step-index and general-route tests with `core=None` fail. With the defaults
  switched, a step-index run records its core as `transparent` with the 100 m⁻¹
  bound and warns only where the band reaches silica's edges.
### The pin moved to optmat 0.2.0 (before Phase 4a)

- **Scope:** 2026-10-02, by Opus 5.5 (AI) for John Travers, on
  `feature/sep-0003-optmat-pin-0.2.0`. optmat's rate work, which Phase 4a waited for, was
  released as 0.2.0 (tag `v0.2.0`, `58f085e`).
- **Review (§9.2, step 6):** of the 84 models solprop uses, the 12 ionisation-rate models
  changed their digests and nothing else did. Gas indices, equations of state, transport,
  Kerr, Raman, solids and the silica bundles are all unchanged. The public surface
  changed by additions only: `circular_rate` and its helpers, and a `correction_range`
  field on the molecular rate model. 0.2.0 holds both the instantaneous and the
  cycle-averaged rate at their maximum-field values above that field, the rule Phase 1d
  adopted. Q12 is therefore resolved upstream, and solprop's hold remains as a guard
  that agrees with it.
- **Re-measured:** `tests/data/sep3_changes.json` at 0.2.0. The rate ratios of §9.1's
  intensities moved by at most a few per cent from 0.1.1.
- **Checks:** the optmat-related tests, 556 passed and one skipped. Under the new rule
  (`AGENTS.md`, *Slices of a phased SEP*), the full lanes run at the end of the
  implementation.

### Phase 4a, first part — the defaults switched

- **Scope:** the first part of §11's Phase 4a, 2026-10-02, implemented by Opus 5.5 (AI)
  for John Travers, on `feature/sep-0003-optmat-materials-4a` from `main` at `020214b`.
  It switches every built-in default to optmat, adds the temporary `"registry"` value
  that Phase 4a's second part needs to attribute the change one family at a time, decides
  Q26 and implements Q9, Q14 and Q17, and re-baselines the tests the switch breaks. The
  attributed changed-results report of §9.2, step 3, is the second part's.
- **Delivered:**
  - `physdata/selection.py`: `REGISTRY` (`"registry"`, accepted by `linear`,
    `electronic`, `raman` and `ionisation`), `FAMILIES`, `REGISTRY_SELECTION` (every family
    on it), `default_selection`, `takes_optmat` and `is_registry`; all six are documented
    as temporary and exported (in `__all__`, so the API reference renders them) until 4b
    removes them.
  - Defaults: `species()` (a `source=` or `gamma3_source=` takes the registry for its own
    family), `Mixture.fill`, `partial_pressures` and `air` (the named composition unless
    `linear="registry"`), `responses_for`, `problem_from` (the bundle's $f_R$ = 0.2188
    with Raman on), every `Simulation.from_*` and `prop_*`, and the design tools take
    `MaterialSelection()` for `None`. The solid factories return optmat's recommended
    model for `model=None`, and `source=` keeps solprop's own: `magnesium_fluoride`,
    `bk7`, `sapphire`, `barium_fluoride`, `silver` and `aluminium` gained it
    (`"dodge-o"`, `"schott"`, `"malitson-o"`, `"malitson"`, `"stitched"`). MgF₂ and
    sapphire without an axis, `glass("MgF2")` included, are refused by name.
  - A gas named on a mode (`GasFillMixin`, every hollow and slab mode) or on a
    `SpatialMedium` without `source=` is `physdata.species.default_species(gas)`: one
    cached optmat species per gas, its polarizability and its equation of state.
    `SpatialMedium` gained a temporary `source=` (kept out of the record). A name given to
    `physdata.transport` and `gasflow` is optmat's gas (`OptmatGas`). `optics.Material`
    resolves a gas by name to its one-population optmat fill; prisms and compressors take
    `glass(name)`.
  - Drivers (`driver/simulation.py`): `_default_cladding`, `_selected_medium` and
    `_selected_planar` follow the linear family (under `linear="registry"` the registry's
    default law, equation of state and silica); `_ionisation_rate`, the single-gas Raman
    and Kerr paths, `_selected_guide` and `_check_selection_carrier` follow theirs.
    `material_output.material_manifest` treats the all-registry selection as none.
  - Q26: `OptmatGlass` compares and hashes by the model's digest, its transparency and
    its bound; `OptmatSusceptibility` by gas and digest; `SilicaBundle` by digest.
  - Q9 and Q14 (`_selected_kerr`, `_spatial_input_peak`): on the free, radial and planar
    routes a `KerrResponse` whose coefficient is the registry's own for the medium's gas
    becomes the optmat coefficient at the carrier, and an unbound `OptmatRate` is bound
    to the input's peak field and the settings' table range.
  - Q17: `_default_raman(provisional=)`: under `raman="provisional"` a capillary's
    `raman=None` turns N₂O's provisional response on, with its one status warning.
  - `physdata.species.built_in_population`: a population optmat can stand for by name
    (optmat-backed, or the registry's own species of a built-in gas).
  - `tools._BulkGas.excess_beta`: the bulk GVD differences $\omega(n-1)/c$ (below).
  - Tests: `tests/test_sep3_defaults.py` (new) and the re-baselined modules (table
    below). Pages: `materials.md` (the default, "The temporary registry handle", solids,
    limitations), `mixtures.md`, `gnlse.md`, `modes.md`, `design-tools.md`, `output.md`,
    `ionisation.md`, `raman.md`, `free-space.md`, `compression.md`, `general-modal.md`,
    `spatial-decompositions.md`, `docs/guide/workflows/luna.md`; `CHANGELOG.md` ("Changed").
- **Deviations from the proposal:**
  - The mode-level `gas=` fill and `SpatialMedium` are switched too, beyond the scope's
    list for the modes: a reference block of `MarcatiliMode(a, "He", 1.0)` otherwise
    disagreed with the run it checks (the coherent wall's true-power tests floored at
    1.6e-8), and a planar slab's fill now carries optmat's equation of state (part of
    Q27).
  - Transport names resolve to optmat: otherwise `steady_fill("He", ...)` beside
    `Mixture.fill({"He": 1.0}, ...)` is refused, their flow weights 3.7e-5 apart.
  - `chi3`, `n2`, `gamma`, `gamma3`, `density`, `refractive_index` and
    `raman.response_for` keep the registry without a selection: they need a carrier or
    are the registry's accessors (Q30).
  - Q9 is applied to `prop_planar` as well as `prop_free` and `prop_radial`.
- **Decisions:**
  - **Q26: value equality**, not a shared instance. Two adapters of one model under one
    approximation compute the same index, so two default cores are one guide's material;
    what each has evaluated and dropped stays its own. Tested: two `StepIndexMode`
    channels built separately make one `ModalGuide`.
  - **The registry handle** maps families as follows. `linear`: the gas law, its equation
    of state and transport, the air composition and the driver's default silica.
    `electronic`: the registry's $\gamma_3$. `raman`: the registry's molecular constants,
    and on a GNLSE solprop's silica line set with 0.18. `ionisation`: `PPT`, `ADK` and
    isotropic MO-ADK. Every family on it reproduces the former run bit for bit and keys no
    manifest. A caller's objects stay the caller's: a supplied mode or medium is
    registry only through `source=`.
  - **The Raman family decides the route** (revises Phase 1c's "a mixture of solprop's
    own species keeps the per-component route"): a built-in gas, registry or optmat, takes
    optmat's response under `raman` from optmat; a supplied species keeps its own.
  - **The carrier check** applies when the electronic, Raman or ionisation family takes
    optmat; a linear-only selection evaluates nothing at the carrier.
  - **A run of the default selection records it**, a vacuum run included, so every
    default run's hash changes with the switch.
- **Re-baselined expectations** (§9.2, step 5), by category. (a) The registry or Luna
  pinned through the registry selection or `source=`, unchanged numbers: the window
  glasses', metals', silica's and transport's own tests; the registry species, air and
  Luna mixture witness; the per-component rates and Raman models
  (`test_mixture_responses`, `test_raman_temperature`, `test_temperature_paths`, the
  rate-policy tests of `test_prop_capillary`); the Luna CtC, antiresonant, mode and tool
  witnesses; and the optmat tests' "without a selection" contrasts. (a′) Numerical
  goldens recorded on the former defaults, pinned: the legacy entry points' digests,
  `rectangular_before_slab`, `fill_before_extraction`, the mixture byte records, the
  version-3 hash and `phi_quadrature` (Q34). (b) Updated with the model change named:

  | Test | Before | After | Model change |
  |---|---|---|---|
  | GNLSE default $f_R$ | 0.18 | 0.2188 | silica bundle (D4) |
  | planar and slab closed-form B | 0.358 | 0.441 | Ar Kerr at 800 nm, +23 % |
  | `gas_ratio` Ar/He $\chi^{(3)}$, $\beta_2$ | 23.5, 21.2 | 28.4, 18.6 | Kerr fits; He GVD |
  | He 0.4 bar, 125 µm: ZDW; $\gamma$; N | 364.8 nm; 4.78e-11; 2.23 | 376.4 nm; 4.87e-11; 2.26 | He index, Kerr |
  | He RDW (125 µm, 1 bar) | 186.8 nm | 199.7 nm | He index |
  | Gebhardt N₂ peak Re | 2539 | 2542 | N₂ viscosity −0.075 % |
  | He transitional length | 6.27 µm | 6.238 µm | He transport |
  | flow pressure at the planes | 0.820933, 0.404941 | 0.820943, 0.404996 | He transport |
  | MaterialKerr 800 nm/2 µm | 2.12958 % | 2.13017 % | silica $n_0$ |
  | wall loss (printed) | 0.9612 /m | 0.9611 /m | silica wall |
  | air fractions | 0.7808/0.9903, … | CIPM "dry air (N2, O2)" | composition (D9) |
  | N₂ transport range | 200–600 K | 63.151–2000 K | optmat range |
  | varying and vector wall digests, 8 flow digests | `fd7e2dfa…`, `5c3cb5ab…`, … | re-recorded | He, silica; the registry still gives each former digest bit for bit |
  | response types, rates, Raman leaf | `RamanResponse`, PPT/MO-ADK, per component | `OptmatRamanResponse`, `OptmatRate`, one per composition | §4.4, §4.6 |

  Some error paths changed with the default (an atom's Raman, N₂O's refusal, a
  silica-range probe), as did thirty docstring examples and four printed blocks.
  (c) No invariant tolerance was changed, and the mode-versus-mixture bitwise reductions
  hold unchanged on optmat. Three things were done for the warnings or schedules, not
  the bounds:
  - module filters for `DiluteScalingWarning` where high-pressure fills are fixtures;
  - registry-range acknowledgements (`pytest.warns(MaterialRangeWarning)`) removed
    where optmat's model is qualified over the band;
  - the planar `field-plasma` parity family takes 20 fixed steps instead of 10. optmat's
    argon rate is 2.5–9 times PPT over its fields, and ten centimetre steps diverged on
    `numpy_ref` and `numba_cpu`; 20, 40 and 80 converge, 40 and 80 to 1e-5.
- **Found and fixed:** `tools._bulk_beta2` differenced the whole $\omega n/c$, whose
  vacuum rounding swamped a gas's GVD at low pressure. The Ar/He $\beta_2$ ratio read
  18.81, 18.51 and 18.56 at 0.01, 0.1 and 1 bar on optmat. Differencing $\omega(n-1)/c$
  gives 18.541, 18.542 and 18.558; the registry's 1 bar ratio moves from 21.2240 to
  21.2197.
- **Verification:** `ruff check --fix`, `ruff format`, `ruff check` and `basedpyright`
  clean; `numpydoc lint` and `codespell` on the changed files clean. The full
  `pytest -n auto`: 12 325 passed, 30 skipped, none failed. The registry selection
  reproduces the witnesses `ar_plasma_field`, `n2_raman_constant`, `he_ar_mixture` and
  `gnlse_raman_on` of `sep3_baseline.npz` bit for bit, and `silver_wall` with
  `silver(source="stitched")`. Each family switches alone from it and runs (N₂, two
  steps). Under the default the witnesses' final planes move by 0.115 (`ar_plasma_field`),
  6.3e-4 and 6.1e-4 (N₂), 1.7e-3 (H₂), 2.2e-3 (He/Ar), 2.6e-3 (silver), 1.4e-2 (GNLSE,
  Raman on) and 0 (Raman off), relative in norm: the second part attributes them. The
  strict docs build (`poe docs`, tutorials re-run on the new defaults) succeeds. Not run:
  `poe check` as a whole, `poe cov`, `test-slow`, the Kokkos lane (end of the SEP).
- **Sources checked:** none new; every number is optmat 0.2.0's or solprop's own.
- **Limitations:**
  - A default run of a dense gas warns `DiluteScalingWarning` (argon from about 5 bar,
    O₂ at 6 bar, H₂ near 10 bar) (Q29).
  - optmat's `fit/SiO2` index carries about 1e-14 of evaluation noise, which limits a
    Richardson derivative check to about 3e-9 (Q32).
  - optmat's D₂ viscosity over 30–600 K is not reproduced to 1e-6 by a Chebyshev series
    of degree 256, so a D₂ temperature path spanning it is refused (Q33).
  - Under the registry selection a numbered mode set's record names the registry silica
    instead of `None`: arrays are bit for bit, hashes are not.
- **Open questions:**
  - Q29. Should the dilute-scaling threshold or its once-per-run warning change, now that
    every dense default run raises it, or wait for SEP 11?
    *Decided* by John Travers on 2026-10-02: the error is recorded for every run and
    warned about above 1 % until SEP 11; implemented in the [final
    fixes](#final-fixes-before-final).
  - Q30. The registry accessors without a selection (above): delete them in 4b, or give
    them a carrier and optmat's models?
    *Answered* in Phase 4b: rerouted to optmat, with the carrier where dispersive.
  - Q31. `linear` bundles the gas law, its equation of state, transport and the default
    silica. §9.2, step 3, attributes them separately: should the second part use explicit
    objects (`source=` claddings, registry species) or finer temporary families?
    *Answered* in Phase 4a's second part: neither in the package; the attribution
    script substitutes the species' parts and passes the wall explicitly.
  - Q32. Pass optmat's silica evaluation noise upstream?
    *Passed to optmat* as a note on 2026-10-02; *resolved* in optmat 0.2.2, whose silica
    page gives the model's $10^{-15}$ evaluation noise.
  - Q33. D₂'s viscosity law: an optmat representation or solprop's degree limit?
    *Passed to optmat* as a note on 2026-10-02; *resolved* in optmat 0.2.2 (D₂'s
    viscosity is Assael et al.'s correlation, which a degree-20 series represents), and
    the strict xfail removed at the [pin to 0.2.2](#the-pin-moved-to-optmat-022).
  - Q34. `phi_quadrature`'s z table on optmat's helium is 1.37e-10 against its 1e-10 bound
    (9.97e-11 on the registry, and its docstring's 1.4e-12 is stale): re-derive the bound
    or the case in 4b.
    *Resolved* in the [final fixes](#final-fixes-before-final): the test runs on
    optmat's helium against a bound derived from the table's build tolerance and the
    oracle's error estimate.
  - Q35. Ten-step qualification cases with strong plasma may need more steps on optmat's
    rates (one planar family did); review at the SEP's final lanes.
    *Handled* at the [final lanes](#final-lanes): no ten-step case failed on optmat's
    rates, so none needed more steps.
- **Handoff to the second part:**
  - **Toggling.** `MaterialSelection(**{f: "registry" for f in FAMILIES if f != family})`
    switches one family from the former defaults, and
    `MaterialSelection(**{family: "registry"})` returns one family to them. Every family on the registry is
    `REGISTRY_SELECTION`. Pass it to `prop_*`/`Simulation.from_*` with `materials=`.
    Supplied solids keep `source=` (`silver(source="stitched")` for the silver witness),
    and a supplied mode or medium is registry by `source=`.
  - **Witnesses** (`tests/data/generate_sep3_baseline.py`, cases of
    `sep3_baseline.npz`): attribute `ar_plasma_field`/`ar_plasma_envelope` (ionisation,
    linear, electronic), the N₂ cases and H₂ (raman, linear), `he_ar_mixture` (linear,
    electronic), `silver_wall` (the wall alone, by `source=`), the GNLSE pair (raman).
    The solid indices and GVD of §9.1 and the rate ratios come from
    `tests/data/sep3_changes.json`.
  - **To decide first:** Q31 (how the linear family splits) and Q29 (the warnings a
    report run will raise).

### Phase 4a, second part — the changed results, attributed

- **Scope:** the second part of §11's Phase 4a, 2026-10-02, implemented by Opus 5.5 (AI)
  for John Travers, on `feature/sep-0003-optmat-materials-4a2` from `main` at `ea1ccc0`
  (optmat 0.2.0 the default). It completes §9.2, steps 3 to 5: the change of every
  Phase 0 witness attributed one material family at a time, the numerical error of that
  comparison, the solids' GVD changes and the decomposition of the rate ratios of §9.1.
  It changes no package code.
- **Delivered:**
  - `tests/data/attribute_sep3_changes.py`, run by hand like the other generators, and
    its record `tests/data/sep3_attribution.json`: the solprop commit (and whether the
    tree was clean), the witnesses' commit, optmat's version, and per run the switched
    families, the observables, their changes, the SHA-256 of the output plane, the
    digest of every optmat model the run recorded, its warnings and runtime.
  - `tests/test_sep3_attribution.py`: the record's form; the all-registry rows are the
    witnesses (each output plane's SHA-256 against `sep3_baseline.npz`); every family is
    run or said not to apply; the default equals the families switched together; every
    resolved change stands above its comparison error; each rate chain closes and is
    the ratio `sep3_changes.json` measured, of the same model; every solid and axis has
    its GVD. No propagation is re-run.
  - The manual page *What changed from solprop's old material data*
    (`docs/dev/reports/material-changes.md`), linked from `materials.md` and the model index;
    `CHANGELOG.md` ("Changed") points to it.
- **Method.**
  - *Runs.* Each witness (same parameters, grid, ten fixed steps, `numpy_ref`) is run
    with every family on `"registry"`, then with each applicable family alone switched
    to optmat, then with all of them through the same mechanism, and with the real
    default (no selection). One at a time from the former defaults, not cumulatively,
    so that each family's row does not depend on an order; the combined run measures
    how far the single changes fail to add up ("interaction").
  - *The linear family split (Q31).* The selection keeps `linear="registry"`, and the
    script substitutes the parts of each species at the driver's one resolution point
    (`solprop.driver.mixtures.species`): optmat's susceptibility for the gas index;
    for the equation of state alone, a script-local `RegistryGas` whose density and
    pressure are optmat's (so the flow keeps the registry's transport); for transport
    alone, a script-local `OptmatGas` whose density is the registry's. The wall is an
    explicit `cladding` (`fused_silica()`, or `silver()` against
    `silver(source="stitched")`). Transport is its own family, which acts only on the
    pressure-gradient witness.
  - *The silica bundle* is split further on the GNLSE: its $f_R$ alone (the registry's
    shape at 0.2188) and its $h_R$ alone (the bundle's shape at 0.18).
  - *Observables*, of the last plane: the spectrum's relative change in norm
    $\lVert E - E_\mathrm{reg}\rVert / \lVert E_\mathrm{reg}\rVert$; the energy; the
    centroid and RMS width of the fundamental band (below twice the carrier; on a field
    grid the third harmonic's share of the spectral energy is recorded apart, since it
    would otherwise dominate the second moment); the peak power and peak intensity; the
    exit density; the electron density and ionised fraction of the plasma witnesses.
  - *Refinement (§9.2, step 4).* Every witness again at twenty steps, on the registry,
    the default and its largest single family. The comparison error is
    $\lVert \Delta_{10} - \Delta_{20} \rVert / \lVert E_\mathrm{reg} \rVert$, with
    $\Delta_N$ the attributed difference at $N$ steps.
  - *GVD.* $\beta_2$ and $\beta_3$ of $\omega n(\omega)/c$ by seven-point central
    stencils at a 1 % frequency step, from the index alone, checked at half the step;
    the analytic first derivative of the optmat solids was too noisy for the stencils
    (a 2 % disagreement in $\beta_3$ between steps when differenced). The capillary's
    change is differenced from the change of $n_\mathrm{eff} - 1$ directly.
  - *Rates.* For each gas a chain of single changes from solprop's former default to
    optmat's recommended rate, at 800 nm and the fields of 5 × 10¹³, 10¹⁴ and
    3 × 10¹⁴ W/cm²: first solprop's own options switched off (Stark shift, ion dipole
    factor, barrier suppression, $m$ sum, occupancy), then optmat's named form of what
    remains (`ppt_ilkov`, `moadk_zhao_2010` or `adk_hydrogenic`, at solprop's
    potential), then optmat's parameters one by one through `with_parameters`. The
    links telescope, so their product is the whole ratio exactly; the record groups them
    into §9.1's factors.
- **Attribution (§9.2, step 3).** The change of each witness's output spectrum,
  $\lVert E - E_\mathrm{reg}\rVert / \lVert E_\mathrm{reg}\rVert$, with one family
  switched; "—" where the family does not act (the record gives the reason: constant
  pressure, no plasma, an atomic gas without Raman, a GNLSE that takes $\gamma$ and the
  $\beta_n$ as numbers). "All" is the default run, which equals the families switched
  together bit for bit in every witness. "Interaction" is how far the single changes
  fail to add up, in the same norm. The last two columns give the change of the
  default without its best constant phase (the carrier-envelope phase, below) and that
  phase.

  | Witness | EOS | transport | gas index | wall | Kerr | silica bundle | Raman | rate | all | interaction | all, no carrier phase | carrier phase (rad) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | `ar_plasma_field` | 4.8e-7 | — | 1.8e-4 | 6.9e-8 | 5.5e-3 | — | — | **0.119** | 0.115 | 9.7e-4 | 7.3e-2 | 8.9e-2 |
  | `ar_plasma_envelope` | 4.3e-7 | — | 1.2e-7 | 6.9e-8 | 3.8e-3 | — | — | **0.122** | 0.119 | 4.4e-4 | 7.6e-2 | 9.2e-2 |
  | `n2_raman_constant` | 5.4e-8 | — | 2.7e-4 | 6.5e-8 | **7.9e-4** | — | 3.1e-5 | — | 6.3e-4 | 4.9e-6 | 5.9e-4 | −2.2e-4 |
  | `n2_raman_gradient` | 5.2e-8 | 2.5e-7 | 2.6e-4 | 6.5e-8 | **7.6e-4** | — | 3.0e-5 | — | 6.1e-4 | 4.5e-6 | 5.7e-4 | −2.1e-4 |
  | `h2_raman_constant` | 1.5e-8 | — | 4.4e-6 | 6.5e-8 | **1.3e-3** | — | 5.4e-4 | — | 1.7e-3 | 2.0e-6 | 1.1e-3 | −1.3e-3 |
  | `he_ar_mixture` | 5.3e-8 | — | 3.7e-4 | 6.5e-8 | **2.5e-3** | — | — | — | 2.2e-3 | 1.4e-6 | 1.9e-3 | −1.3e-3 |
  | `silver_wall` | 5.9e-8 | — | 1.8e-4 | 3.0e-6 | **2.7e-3** | — | — | — | 2.6e-3 | 7.8e-7 | 2.0e-3 | −1.6e-3 |
  | `gnlse_raman_on` | — | — | — | — | — | **1.38e-2** | — | — | 1.38e-2 | 0 | 1.33e-2 | 3.8e-3 |
  | `gnlse_raman_off` | — | — | — | — | — | — | — | — | 0 | 0 | 0 | 0 |

  What the observables add:
  - *Argon with plasma.* The rate alone raises the ionised fraction at the exit by
    127 % (field grid; 2.88 % → 6.53 %) and 129 % (envelope), lowers the output energy
    by 0.57 % and 0.76 % and the peak power by 1.6 % and 2.3 %, and moves the
    fundamental band's centroid up by 0.83 and 0.85 THz (the plasma blue shift); its
    change of the spectrum is 7.4 × 10⁻² and 7.6 × 10⁻² without the carrier phase it
    also adds (9.3 × 10⁻² and 9.6 × 10⁻² rad). The
    Kerr coefficient alone (+23 %) raises the peak power by 1.0 % and the third
    harmonic's energy by 51 % (1.23² = 1.51), and lowers the ionised fraction by 0.6 %.
    Together: ionised fraction +125 %, energy −0.56 %, peak power −0.6 %, centroid
    +0.82 THz. The two interact (9.7 × 10⁻⁴, 0.8 % of the change), the only witness
    where the families do not add to within 5 × 10⁻⁶.
  - *Gases without plasma.* The Kerr coefficient dominates, and its third-harmonic
    energy follows its square: N₂ +15.5 % (1.0747²), H₂ +17.9 % (1.0853²), He/Ar
    +46.9 % (the mixture's coefficient, dominated by argon's). The gas index comes
    second in the plain norm (2 to 4 × 10⁻⁴; H₂'s 4 × 10⁻⁶), but on a field grid its
    change is almost all one constant phase of the whole spectrum, the
    carrier-envelope phase: the full phase of the carrier propagates there, and an
    index change at the carrier shifts it (argon 1.77 × 10⁻⁴ rad). The envelope grid's
    frame removes that phase, and its index change is 1.2 × 10⁻⁷. Without it the index
    changes the field-grid witnesses by 2.7 × 10⁻⁶ (silver) to 4.9 × 10⁻⁵ (N₂), argon's
    1.2 × 10⁻⁵ mostly through the third harmonic. The Kerr change's own constant phase
    has the opposite sign, so the default moves N₂, He/Ar and the silver witness less
    than the Kerr change alone in the plain norm, and as much without the carrier phase.
    Raman comes third for N₂ (3 × 10⁻⁵; 1.2 × 10⁻⁵ without the carrier phase), and is
    H₂'s second (5.4 × 10⁻⁴, a further red shift of the centroid by 2.4 GHz).
  - *The EOS* moves the densities by +6 ppm (Ar, N₂) and +1 ppm (H₂, He/Ar), and the
    spectra by at most 5 × 10⁻⁷. *Transport* moves the gradient's exit density by
    +5.9 × 10⁻⁵ (the flow) and its spectrum by 2.5 × 10⁻⁷. *The wall* is invisible
    through silica (6.5 to 6.9 × 10⁻⁸), and 3.0 × 10⁻⁶ through silver, almost all of it
    a carrier phase (7 × 10⁻⁸ without).
  - *The GNLSE.* The whole change is $f_R$: 0.18 → 0.2188 moves the spectrum by
    1.38 × 10⁻², lowers the peak power by 0.66 % and moves the centroid down by
    65 GHz (a stronger Raman red shift). The bundle's $h_R$ at 0.18 reproduces the
    registry's run to 1.3 × 10⁻¹⁶. With Raman off nothing changes.
- **Numerical against physical (§9.2, step 4).** From ten to twenty steps the registry
  runs move by at most 6.7 × 10⁻¹³ (capillaries) and 7.1 × 10⁻⁸ (GNLSE). The attributed
  difference of the default, and of each witness's largest family, moves by at most
  5.6 × 10⁻¹¹ (capillaries) and 5.2 × 10⁻⁹ (GNLSE), so every reported change stands
  2.2 × 10⁹ (capillaries) and 2.6 × 10⁶ (GNLSE) times above the numerical error of its
  comparison. The witnesses are short (the capillaries 0.4 mm), so this shows the
  comparison is numerically exact; it does not show how the changes grow in a long
  adaptive run (Q36).
- **The solids' GVD (§9.1).** $\beta_2$, new minus former, in fs²/mm (and relative):

  | Solid | 400 nm | 800 nm | 1030 nm | 1550 nm | 2 µm |
  |---|---:|---:|---:|---:|---:|
  | fused silica | +0.214 (+0.22 %) | +0.042 (+0.12 %) | +0.029 (+0.15 %) | +0.075 (−0.27 %) | −1.664 (+1.65 %) |
  | CaF₂ | +0.085 (+0.13 %) | +0.036 (+0.13 %) | +0.027 (+0.15 %) | +0.014 (β₂ ≈ −0.05) | +0.005 (−0.03 %) |
  | MgF₂ o | +0.118 (+0.24 %) | +0.049 (+0.25 %) | +0.038 (+0.33 %) | +0.030 (−0.33 %) | +0.032 (−0.09 %) |
  | BaF₂ | +0.163 (+0.18 %) | +0.064 (+0.17 %) | +0.049 (+0.18 %) | +0.033 (+0.29 %) | +0.029 (−1.2 %, β₂ ≈ −2.4) |
  | sapphire o | +0.246 (+0.16 %) | +0.097 (+0.17 %) | +0.076 (+0.24 %) | +0.066 (−0.20 %) | +0.084 (−0.07 %) |
  | N-BK7 | +0.205 (+0.17 %) | +0.078 (+0.17 %) | +0.061 (+0.24 %) | +0.054 (−0.22 %) | +0.077 (−0.08 %) |

  The crystals' and N-BK7's changes are nearly constant, a few hundredths of fs²/mm in
  the near infrared: the size of the GVD of the air (about 0.02 fs²/mm at 800 nm) that
  their former indices, relative to air, left out. $\beta_3$ changes by 0.01 % to
  0.33 %, except fused silica at 1550 nm (+1.24 %) and 2 µm (−0.82 %). Fused silica at
  2 µm is the one material change of note: optmat's refit to absolute refractometry
  curves differently from Malitson's formula there, and its index has structure the
  stencil check resolves (2.6 × 10⁻³ in $\beta_3$ between steps, against 10⁻⁶
  elsewhere). The extraordinary rays of MgF₂ and sapphire, new, have $\beta_2$ = 20.60
  and 56.72 fs²/mm at 800 nm (ordinary: 19.83 and 58.14). Every value is inside both
  models' ranges; the stencils agree with half their step to 3 × 10⁻⁶ in $\beta_2$
  and 8 × 10⁻⁵ in $\beta_3$ except there.
- **A silica capillary.** The HE₁₁ mode of a 100 µm-radius evacuated capillary at
  800 nm: $\beta_2$ = −13.2817 fs²/m with either wall, changed by −9.1 × 10⁻⁷ fs²/m
  (7 × 10⁻⁸ relative), $\beta_3$ by 3.6 × 10⁻⁶ fs³/m; its loss, 1.2016 dB/m, by
  −0.015 %. Marcatili's wall term is imaginary for a transparent wall, so the wall's
  index moves the loss and leaves the real index unchanged to 5 × 10⁻¹⁵.
- **The rate ratios, attributed (§9.1).** The ratios are those of §9.1 at optmat 0.2.0,
  identical to `sep3_changes.json`'s, and none of the fields is above optmat's maximum
  field. Each chain ends on optmat's default rate to 0 (exactly), and its link from
  solprop's form to optmat's named form of it is 1 to 7.2 × 10⁻¹³ for every gas:
  solprop's PPT without its Stark shift, dipole factor, barrier suppression, $m$ sum
  and occupancy *is* `ppt_ilkov`; its isotropic MO-ADK is `moadk_zhao_2010`; its ADK
  of a molecule is `adk_hydrogenic` times its occupancy 2. The factors at
  10¹⁴ W/cm² (the record has all three intensities):

  | Gas | ratio 5e13 / 1e14 / 3e14 | electrons | non-adiabatic | structure | averaging | barrier suppression | threshold | Stark, dipole | vibration |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | He | 2.15 / 1.67 / 1.30 | 0.735 | 2.30 | 1.10 | 0.820 | 0.999 | 1.001 | 1.10 | — |
  | Ne | 0.77 / 0.61 / 0.51 | 0.462 | 2.27 | 0.573 | 0.831 | 0.997 | 1.001 | 1.22 | — |
  | Ar | 2.54 / 2.45 / 3.53 | 0.443 | 2.18 | 1.17 | 0.855 | 0.993 | 1.002 | 2.54 | — |
  | Kr | 1.40 / 1.03 / 0.76 | 0.383 | 2.15 | 1.48 | 0.862 | 0.990 | 0.990 | 1 | — |
  | Xe | 1.75 / 1.26 / 0.96 | 0.346 | 2.10 | 2.03 | 0.869 | 0.981 | 1.001 | 1 | — |
  | H₂ | 25.2 / 3.28 / 0.85 | 2 | 4.74 | — | 0.856 | 0.652 | 0.403 | — | 1.54 |
  | D₂ | 27.6 / 3.51 / 0.89 | 2 | 4.78 | — | 0.856 | 0.652 | 0.427 | — | 1.54 |
  | N₂ | 26.5 / 2.87 / 0.43 | 2 | 4.92 | — | 0.856 | 0.341 | 1.002 | — | 1 |
  | O₂ | 4.98 / 1.38 / 0.44 | 2 | 2.57 | — | 0.870 | 0.428 | 0.721 | — | 1 |
  | CH₄ | 1.06 / 0.37 / 0.14 | 3 | 2.78 | 3.37 | 0.868 | 0.185 | 0.083 | — | 1 |
  | SF₆ | 855 / 126 / 30.1 | 3 | 4.92 | 49.2 | 0.858 | 0.347 | 0.581 | — | 1 |
  | N₂O | 5.51 / 0.96 / 0.15 | 2 | 2.97 | 1.05 | 0.867 | 0.177 | 1.011 | — | 1 |

  The groups, each a product of links: *electrons* is solprop's $m$ sum and occupancy
  against optmat's $N_\mathrm{eff}$ and, for the atoms, the ion's fine-structure
  channels; *non-adiabatic* is, for the atoms, the removal of Ilkov's
  $(1+\gamma^2)^{-n^*}$ (the static Coulomb factor), and for the molecules the static
  MO-ADK rate against PPT's frequency-dependent rate of the same orbital; *structure*
  is the Hartree coefficient against Tong, Zhao and Lin's (atoms), or a hydrogen-like
  s electron against the orbital's structure factors (CH₄, SF₆, N₂O); *averaging* is
  the instantaneous rate's relation to the cycle average (the tunnelling-limit factor
  against optmat's consistent one); *barrier suppression* is solprop's Tong–Lin factor
  removed and optmat's added (for the atoms they nearly cancel; the molecules had
  none); *threshold* is solprop's ionisation energy against optmat's (vertical for the
  molecules). With intensity, the non-adiabatic factor falls (atoms 3.2–3.4 → 1.36–1.47;
  molecules 8.5–36 → 1.27–1.51 from 5 × 10¹³ to 3 × 10¹⁴ W/cm²), the averaging factor
  rises (0.74–0.81 → 0.91–0.94), and the molecular barrier suppression deepens (to
  0.05–0.48); argon's ratio rises because solprop's Stark shift and dipole factor had
  cut its rate by 1.9, 2.5 and 5.2.

  Where a factor cannot be isolated:
  - The links depend on their order: each is evaluated with the links before it
    applied, so a factor that changes $I_p$ or $n^*$ (the Stark shift, the threshold)
    shifts the others slightly. The order is fixed (solprop's own options first, then
    optmat's in the order of its parameters) and the product is exact.
  - The molecules' non-adiabatic link changes the form (MO-ADK to optmat's `molecular`)
    in one step: it cannot be split into the frequency dependence and the change of
    tunnelling formula. Evaluated in a static field the link is 1 to 2 × 10⁻¹⁵, so it
    is the frequency dependence alone.
  - The potential-only molecules' structure link replaces a hydrogen-like s electron
    by the orbital's structure factors at once; their angular and radial parts are not
    separable through optmat's parameters.
  - The Stark shift and the dipole factor exist only in solprop's PPT, for He, Ne and
    Ar (Kr and Xe had no polarisabilities, so their factor is 1). They are isolated on
    solprop's side; optmat has no counterpart to switch.
  - The molecules' effective electrons against solprop's occupancy are exact factors
    (2, or 6 × ½ and 4 × ½), not fitted; SF₆'s structure factor (44 to 64) is an
    untested prediction with uncertainty factor 10.
- **In plain terms.**
  - Plasma in gases changes most, through the ionisation rate. Every rate moved:
    argon's by 2.5 to 3.5 times, the diatomic molecules' by up to 28 times at
    5 × 10¹³ W/cm² and down to 0.4 at 3 × 10¹⁴, mostly because the former molecular
    rates had no frequency dependence. In the argon witness the ionised fraction more
    than doubles.
  - A GNLSE with Raman on and the default fraction changes through $f_R$ alone (0.18 →
    0.2188); the response shape is the same.
  - Gas runs without plasma change through the Kerr coefficient (argon +23 %, neon
    +42 %, most molecules +5 % to +9 %, N₂O and xenon −5 %), which scales the nonlinear phase and the third
    harmonic's energy with its square. The gas index comes second and, on a field grid,
    acts mostly as a constant carrier-envelope phase.
  - Densities, flows and walls change by parts per million; the solids' GVD by 0.1 %
    to 0.35 % (fused silica 1.7 % at 2 µm), and a capillary's dispersion not at all.
- **Decisions:**
  - **Q31 is answered in the attribution script**, not in the package: no finer public
    family and no new temporary option. The split is an attribution concern, and the
    registry goes in Phase 4b; the substitution at the driver's one resolution point is
    shown sufficient by the default run equalling the four parts switched together bit
    for bit, in every witness.
  - Families are switched one at a time from the former defaults, not cumulatively, and
    their non-additivity is measured, not apportioned.
  - The GNLSE's bundle is split into $f_R$ and $h_R$, since §4.5 changes both.
  - The observables' moments are of the fundamental band; the third harmonic's share is
    recorded apart, on field grids only.
  - The spectrum's change is also recorded without its best constant phase
    ("carrier phase"): on a field grid a change of the phase index at the carrier is a
    carrier-envelope phase, which the plain norm counts in full and the envelope frame
    removes.
  - The rate chains start from solprop's side, so that every link that exists only in
    solprop (Stark, dipole, $m$ sum, occupancy) is a factor of its own.
- **Deviations from the proposal:**
  - No adaptive runs (§9.2, step 4: "then adaptive runs to expose the changed
    trajectory"): this phase was limited to runs of ten and twenty fixed steps, and
    the witnesses are too short to have a trajectory of their own (Q36).
  - Transport is a family of its own beside the EOS, as Q31 asked; it acts only on the
    gradient witness.
  - No bulk-window propagation, as in Phase 0: a window's change is its index and GVD,
    evaluated directly.
- **Verification:**
  - Every witness's all-registry run is its frozen output plane bit for bit (nine of
    nine, by SHA-256, including the four Phase 4a's first part had not listed:
    `ar_plasma_envelope`, `n2_raman_gradient`, `h2_raman_constant` and
    `gnlse_raman_off`); every default run equals the families switched together bit for
    bit.
  - Rate chains: closure 0; solprop-to-optmat named-form parity ≤ 7.2 × 10⁻¹³; ratios
    and digests equal to `sep3_changes.json`'s.
  - GVD stencils: the change at half the step ≤ 3 × 10⁻⁶ ($\beta_2$) and 8 × 10⁻⁵
    ($\beta_3$), and 2.6 × 10⁻³ for optmat's fused silica at 2 µm.
  - `ruff check --fix`, `ruff format`, `ruff check`, `basedpyright` on the new files;
    `pytest tests/test_sep3_attribution.py tests/test_sep3_baseline.py
    tests/test_seps.py`; the strict docs build. Not run: `poe check`, `test-slow`,
    `cov`, the Kokkos lane (end of the SEP).
- **Sources checked:** none new; every number is an evaluation of optmat 0.2.0's or
  solprop's own public functions, or a run of solprop.
- **Limitations:**
  - The witnesses rank the families and show the direction of each change; they do not
    size the change of a long run, whose nonlinear phase and plasma are much larger.
  - The relative norm of the spectrum is one number for a whole spectrum; on a field
    grid it is dominated by a constant phase for the gas index's change.
  - The rate decomposition is order-dependent (above), and is of the instantaneous
    rate at linear polarisation and 800 nm only.
  - The finer families exist only in the script, which runs while the registry exists;
    after Phase 4b the record stands as a historical one, like the witnesses.
- **Open questions:**
  - Q36. The witnesses are 0.4 mm and 1 cm long. Should Phase 4c's regenerated worked
    examples (full size, by hand) also be run once on `REGISTRY_SELECTION` before Phase
    4b deletes it, so that the changes of their headline results (an argon RDW, a
    soliton self-compression, a CtC spectrum) are recorded at full length?
    *Answered* before Phase 4b: no; the committed outputs are the archive.
  - Q37. optmat's fused silica index has structure near 2 µm that changes $\beta_2$ by
    1.7 % from Malitson's formula and resolves in a stencil check (2.6 × 10⁻³ in
    $\beta_3$); confirm upstream that this is the refit to the measurements, not an
    artefact of the fit (with Q32).
    *Passed to optmat* as a note on 2026-10-02; *resolved* in optmat 0.2.2: Lithosil
    Q2's OH bands, which the fit keeps unscaled (+2.1 % of the GVD at 2 µm over
    Malitson's formula); the absolute refractometry neither confirms nor excludes them.
  - Q38. The optmat solids' analytic first derivative (`OptmatGlass.dindex_domega`)
    jitters by about 2 × 10⁻⁸ relative between neighbouring frequencies (CaF₂, an
    alternating sign in its second differences), where solprop's Sellmeier formula is
    smooth: whether the adapter's chain rule or optmat's derivative is the source was
    not found. It does not touch propagation, but spoils a third derivative
    differenced from it.
    *Passed to optmat* as a note on 2026-10-02; *resolved* in optmat 0.2.2: the
    reference gas's refractivity, not its squared index, is now differenced, and the
    derivatives move by up to $2.5\times10^{-8}$.
- **Handoff:** Phase 4b may delete the registries: the attribution no longer needs
  them, and its record and script stay as historical artefacts (the script will not run
  after 4b, as the generator of the witnesses will not). Phase 4c's manual and example
  work starts from *What changed from solprop's old material data* and Q36.

### Phase 4b — the old database and formulas deleted

- **Scope:** §11's Phase 4b, 2026-10-02, implemented by Opus 5.5 (AI) for John Travers,
  on `feature/sep-0003-optmat-materials-4b` from `main` at `dc4ca0d` (optmat 0.2.0 the
  default, Phase 4a complete). It deletes solprop's own material data, its rate formulas
  and the temporary registry handle, maps the solid factories' former `source=` names
  where an optmat model is the same published formula, reroutes the bare accessors to
  optmat (Q30), and reworks the tests and pages that relied on the deleted code. The
  physics of a default run does not change: Phase 4a already took every default from
  optmat.
- **Deleted, mapped and rerouted:**

  | What | Disposition |
  |---|---|
  | `physdata/sellmeier.py`: the gas index models (`_MODELS`, `_DEFAULT_SOURCE`, the reference states and citations), `sources()`, every `source=` | deleted; the supplied formula types (`GasSusceptibility`, `PoleSumSusceptibility`, `RefractivitySusceptibility`, `InversePoleIndex`, `WavenumberPoleIndex`, `ReferenceState`) kept, their reference density now optmat's equation of state |
  | `gamma(gas)`, `chi1`, `dchi1_domega`, `refractive_index`, `refractive_index_minus_one`, `drefractive_index_domega` | rerouted: optmat's law (`model=` names another optmat model) and optmat's equation of state |
  | `physdata/nonlinear_data.py`: the Lehmeier, Shelton and measured-$n_2$ tables, `gamma3_sources`, `N2_REFERENCE_WAVELENGTH` | deleted; `n2_from_chi3`, `chi3_from_n2`, `OptmatKerr`, `carriers_agree`, `selected_carrier` kept |
  | `gamma3(gas, wavelength=)`, `chi3(...)`, `n2(...)` | rerouted: optmat's Kerr coefficient at a carrier, which they now require |
  | `physdata/raman_data.py` (molecular constants, the silica line and tensor sets, `raman_parameters`, `raman_data_status`, `silica_raman`, `silica_tensor_raman`) | deleted; the supplied types (`Linewidth`, `linewidth_from_dephasing_time`, `RotationalRaman`, `VibrationalRaman`, `MolecularRaman`, `IntermediateBroadeningRaman`, `LinAgrawalRaman`, `Dephasing`, `MolecularOptions`) moved to `physdata/raman_types.py` |
  | `physdata/ionisation_data.py` | deleted; `ionisation_potential(gas)` rerouted to optmat's curated first ionisation energy (`physdata.species`); `IonisationData` is the energy and its citation |
  | `physdata/equation_of_state.py`: the pyfluids equation of state | deleted; `density`, `pressure` rerouted to `OptmatGas`; `ideal_gas_density` kept; one pyfluids call left, air's molar mass (Q39) |
  | `physdata/transport.py`: the viscosity table, CoolProp lookups, ranges and Wilke code; `DILUTE_PRESSURE_BAR`, `VISCOSITY_REFERENCE_PRESSURE_BAR` | deleted; `viscosity`, `heat_capacity_ratio`, `sound_speed`, `molar_mass`, `temperature_range` are optmat's through `OptmatGas`; the mean free path and the dilute sound speed stay solprop's |
  | `species.RegistryGas`; `species(source=, gamma3_source=)` | deleted; a built-in species carries optmat's law, `OptmatGas`, `OptmatKerr`, optmat's ionisation energy and no Raman constants |
  | `physdata/glasses.py`: the silica, CaF₂, MgF₂, BaF₂, sapphire and N-BK7 coefficient sets, the table loader, `fused_silica_sources`, `calcium_fluoride_sources`; `data/silica_franta_2016.csv`, `silver_stitched.csv`, `aluminium_stitched.csv` with `tests/data/generate_silica_table.py` and `generate_metal_tables.py` | deleted; `SellmeierGlass`, `TabulatedGlass`, `CompositeGlass` kept; the mirror curves kept (D3) |
  | the solid factories' `source=` | mapped (`glasses.LEGACY_SOURCES`, below), or refused naming the optmat model to use |
  | `physdata.mixtures.AIR`, `AIR_NITROGEN_OXYGEN` (hard-coded NOAA recipes) | deleted; `Mixture.air()` is always optmat's named composition |
  | `nonlinear/ionisation.py`: `ADK`, `PPT`, `IsotropicMOADK`, `MOADK_DEFAULT_GASES`, `BSI_TONG_LIN_ALPHA`, `BSI_ZHANG_COEFFICIENTS`, `BsiCorrection`, the occupancy default and every helper | deleted (D1); the rate protocol, `RateConvention`, `MAX_RATE`, `rate_convention`, `rate_species`, `rate_energy`, `bind_rate_energy`, `keldysh`, `barrier_suppression_field`, `ionisation_fraction` kept |
  | `selection.REGISTRY`, `REGISTRY_SELECTION`, `FAMILIES`, `default_selection`, `takes_optmat`, `is_registry`, and every branch that read them (drivers, tools, mixtures, GNLSE, manifest) | deleted |
  | the GNLSE's inline 0.18 and its `"default"` fraction origin | deleted; `None` with Raman on is the bundle's $f_R$ |
  | `source=` of the hollow, rectangular and slab modes, the mode-collection builders and `SpatialMedium` | deleted |
  | Q9's by-value substitution (`_selected_kerr`) | deleted: it recognised the registry's $\gamma_3$, which no longer exists; a `KerrResponse` is the caller's, and `gamma3("Ar", wavelength=)` is optmat's coefficient |
  | `SilicaTensorResponse()` | default parameters rerouted to optmat's tensor bundle |
  | `response_for(gas)` | rerouted to optmat's response (a carrier is required); a caller's species with its own `MolecularRaman` keeps the supplied route |
  | `solprop.sol`: `ADK`, `PPT`, `IsotropicMOADK` | removed; `OptmatRate` added; `n2`, `density`, `refractive_index` are the rerouted accessors |
  | `tests/data/generate_sep3_baseline.py`, `measure_sep3_changes.py`, `attribute_sep3_changes.py` | deleted; last run at `eae0ed8`, `7038558` and `8d116c8` respectively, the commits their records name; `sep3_baseline.*`, `sep3_changes.json` and `sep3_attribution.json` kept as historical records with their light tests |
  | `tests/data/generate_luna_ionisation.py`, `luna_ionisation.json`, `tests/nonlinear/test_luna_ionisation.py` | deleted: they compared solprop's deleted ADK and PPT with Luna's |

- **The solid source mapping (§5, "Solid laws").** Each mapped name is the optmat model
  of the same published formula, now absolute: the coefficients are identical (checked
  by evaluation: the ratio of the new to the former index is the reference gas's
  refractive index, $2.7\times10^{-4}$ above one, to the precision of the air model
  used for the check):

  | Material | Former `source=` | optmat model |
  |---|---|---|
  | fused silica | `"malitson"` | `malitson_1965/sellmeier` |
  | fused silica | `"franta"` | `franta_2016/udm_lithosil_q2` (the universal dispersion model the former table sampled; the two agree to $2\times10^{-5}$) |
  | fused silica | `"malitson+franta"` | refused: solprop's own hybrid; `fit/SiO2` (the default) or `malitson_1965/sellmeier` |
  | CaF₂ | `"daimon"`, `"malitson"` | `daimon_2002/sellmeier_20C`, `malitson_1963/sellmeier` |
  | MgF₂ | `"dodge-o"` | `dodge_1984/sellmeier_o` |
  | N-BK7 | `"schott"` | `schott_2023/sellmeier` |
  | sapphire | `"malitson-o"` | `weber_2003/sellmeier_o` (Malitson and Dodge's formula as Weber's handbook prints it; the first coefficient carries one more digit) |
  | BaF₂ | `"malitson"` | `malitson_1964/sellmeier` |
  | silver, aluminium | `"stitched"` | refused: other measurements than optmat's tables stitch; `table/Ag`, `table/Al` (the default) |

  An unknown name, and a `source=` beside `model=`, are refused by name.
- **Q30, decided.** Every bare accessor is kept and rerouted to optmat, with the carrier
  where the coefficient is dispersive (`gamma3`, `chi3`, `n2` require one); none only
  made sense for the registry except `sources()`, `gamma3_sources()` and the
  `*_sources()` of the solids, which are deleted. `solprop.sol` re-exports the rerouted
  accessors.
- **Other decisions:**
  - **Gas `source=` is removed, not mapped.** optmat names the literature forms of the
    gases (`borzsonyi_2008/sellmeier_Ar`, ...), but they are reductions at optmat's own
    reference states and equation of state, not the former registry's numbers, so no
    former gas name is claimed as an equivalent; a selection (`linear=`) or
    `gamma(gas, model=)` names them.
  - **Built-in species carry optmat's first ionisation energy** (`ionisation_parameters`,
    adiabatic), so that a caller's rate on a built-in gas has an energy per event; an
    optmat rate pays its own (D8).
  - **A rate table keeps its rate's energy per event** (`RateTable.plasma_energy`): a
    response on a prebuilt table pays the energy the rate declared, as one on the rate
    does. Found by the test rework: before, a table dropped it and the response paid the
    species' potential, which for a molecule is the adiabatic energy, not the vertical
    one the rate uses. Beside it, `responses_for` names a bare rate by `rate_species`,
    which every rate answers, rather than by a `gas` attribute only the deleted rates
    had.
  - **`raman=None` follows optmat's status** (`optmat_raman.raman_status`): on for a gas
    whose response is validated, off with a warning for N₂O's provisional bands unless
    the selection is `raman="provisional"` (Q17).
  - **Test-only rate oracle.** `tests/rate_oracles.py` holds an ADK-form rate with
    explicitly supplied parameters, bit for bit the former `ADK` (checked against
    `dc4ca0d` over five gases and four option sets), with the noble gases' former inputs
    as named historical test inputs. The numerics tests (tables, kernels, parity,
    quadrature, envelope oracles) use it, so their numbers do not move when optmat's
    curated rates do; the physics-default tests use `OptmatRate`. The package never
    reads it (Q40).
  - **Historical witnesses on explicit data.** The Luna mode, CtC, Raman, plasma,
    antiresonant, mixture and tool witnesses keep their bounds with Luna's inputs
    supplied test-locally: `tests/modes/witness_materials.py` and
    `tests/physdata/former_gas_laws.py` (the Börzsönyi, Ermolov, Kingston and Peck
    laws and Malitson's silica as supplied laws), `tests/former_raman.py` (the former
    Raman constants as `MolecularRaman` and line sets, bit for bit `dc4ca0d`'s), the ADK
    oracle, and Luna's own recorded rate where the fixture has it (the PPT plasma
    witness, now exact at its samples: its bound tightens from $2\times10^{-7}$ to
    $2\times10^{-13}$). Each is a caller's object; nothing is looked up by name.
  - **Q36, answered before this phase:** no extra registry runs; the worked examples'
    committed outputs are the archive of the former results.
- **Test disposition** (the ~20 modules Phase 4a pinned to the registry included):
  the suite went from 12 325 passed and 30 skipped (Phase 4a) to 11 850 passed, 29
  skipped and 1 xfailed; 159 test functions fewer net, about 230 deleted and 70 added.
  - *Deleted, testing the deleted data or formulas themselves* (whole modules):
    `nonlinear/test_luna_ionisation.py` (solprop's ADK and PPT against Luna's rates),
    `nonlinear/test_molecular_adk.py` (MO-ADK; its table test moved onto `OptmatRate`),
    `nonlinear/test_raman_source_audit.py` (the registry's D₂ and CH₄ against their
    sources). In part: `test_ionisation.py` (about 57 formula tests; the protocol,
    conventions, `MAX_RATE` and diagnostics kept on the oracle), `test_mixture_responses.py`
    (8), `test_oxygen_raman.py` (3), `test_raman_data.py` (20), `test_ionisation_data.py`
    (14), `test_nonlinear_data.py` (9), `test_glasses.py` (about 15, with the table
    parser), `test_window_glasses.py` (9 transcription checks), `test_metals.py` (2),
    `test_sellmeier.py` (the registry laws, their cross-source agreement and golden),
    `test_transport.py` (the pyfluids and NIST-table tests), and in
    `test_optmat_rates.py` three parities against the deleted formulas.
  - *Kept as historical witnesses on explicit, test-local inputs* (bounds unchanged):
    the Luna mode, antiresonant, step-index, FEM and tabulated witnesses (`modes/`), the
    CtC witness (`test_prop_capillary.py`: Peck's H₂ law, Luna's $\gamma_3$, the former
    Raman constants, the ADK oracle in its potential-only form, which also matches Luna's
    sampled rate to $3.4\times10^{-6}$), the Luna Raman (57 tests) and plasma witnesses,
    the Luna reference indices (to $10^{-11}$), the mixture, optics, compressor and tool
    witnesses, the mixture-connected case (Raman now off: optmat refuses broadening by
    the synthetic absorber, D16), the z-table bound of `test_linop.py` on Ermolov's
    helium (Q34: 1.37e-10 against 1e-10 on optmat's helium; the bound is kept and the
    case's former input supplied), and the polarised-run energy rows.
  - *Registry-handle and `source=` tests replaced* by tests of the deletion
    (`test_sep3_defaults.py`: the names are gone, `"registry"` is no longer a value) and
    of the solid mapping and refusal (`physdata/test_glasses.py`,
    `test_window_glasses.py`, `test_metals.py`, a coverage check over
    `LEGACY_SOURCES`).
  - *The modules Phase 4a pinned to the registry*: each runs on optmat with its expected
    number re-baselined (cause named) or keeps its former number with the former input
    supplied: the window glasses, metals, silica and transport tests; the registry
    species, air and Luna mixture witness; the per-component rates and Raman models; the
    Luna CtC, antiresonant, mode and tool witnesses; the "without a selection"
    contrasts of the optmat tests (removed: there is no other path); the legacy entry
    points' digests, `rectangular_before_slab`, `fill_before_extraction`, the mixture
    byte records, the version-3 hash and `phi_quadrature` (below).
  - *Numerics tests on a fixed rate*: the ADK oracle (bit for bit the former `ADK`) in
    the tables, kernels, stage bundles, parity, envelope and quadrature tests; optmat's
    `ppt_ilkov` where a PPT-form rate was the point.
- **Re-baselined comparisons** (§9.2, step 5; material comparisons, not invariant
  bounds; each with its cause in the test):
  - the reference densities of Luna's gases against optmat's equation of state: Luna's
    CoolProp and optmat's differ by a constant per gas, up to $1.5\times10^{-5}$ (was
    $1.6\times10^{-8}$ against solprop's own CoolProp call); the CtC helium ratio
    $3.551\times10^{-7}$ (bound $4\times10^{-7}$, was $1.6\times10^{-8}$ and $3\times10^{-8}$),
    H₂ $-1.1\times10^{-6}$ (bound $2\times10^{-6}$);
  - the NIST WebBook viscosities against optmat's: $10^{-3}$ at 294 K (optmat's is the
    zero-density limit, 0.075 % below the 0.1 MPa value), $10^{-2}$ over temperature
    (worst 0.65 %);
  - Luna's ionisation potentials against optmat's curated energies: $10^{-3}$ (worst
    $9.5\times10^{-4}$);
  - Luna's metal indices against optmat's tables: Ag 0.149 (bound 0.171), Al 0.244
    (bound 0.280), within optmat's range;
  - the Ar zero-dispersion pressure against Luna's: 0.6 % on optmat's argon;
  - the optmat-law hypothesis test of the gas index over pressure: up to 5 bar (was
    100), since optmat refuses condensed states.
  - Goldens regenerated as self-consistency records, with the model change named:
    `fill_before_extraction.json` (`source` removed; optmat's equation of state moves
    values by $\le 1.2\times10^{-13}$), `rectangular_before_slab.json` (optmat's metals:
    $n_\mathrm{eff}-1$ by $6.5\times10^{-4}$ and $7.3\times10^{-3}$; optmat's mixture:
    $2.2\times10^{-3}$), the M2 entry of `mixture_single_component_baseline.json`, the
    display hash and `SilicaTensorResponse.txt`, the legacy-entry-point digests of the
    coherent capillary on `numpy_ref` and `numba_cpu` (the Kokkos pin still holds the
    former digest and needs re-recording on the reference build), and
    `convergence/ionisation_knots.json` (optmat's neon `ppt_ilkov` in place of solprop's
    PPT; ratios 21.90 and 16.01 inside the unchanged 10–24 gate).
  - Retired: the byte witness of `test_mixture_interface_baseline.py` and its
    `mixture_interface_baseline.json` (its flow needs optmat's equation of state and
    transport, which a caller's species cannot supply, so the former bytes cannot be
    reproduced; its power-weight and log-weight checks remain at their tolerances), and
    the refractivity regression `.npz` of the registry laws.
  - Unchanged: no invariant tolerance (conservation, convergence, parity, bitwise
    reductions). One case is marked `xfail(strict)` rather than widened: optmat 0.2.0's
    D₂ viscosity is not smooth near 86–92 K, so a viscosity law over 60–600 K or wider
    is refused (Q33).
- **Pages:** the manuals (`materials.md`, `ionisation.md`, `raman.md`, `mixtures.md`,
  `modes.md`, `free-space.md`, `gnlse.md`, `envelope-plasma.md`, `running.md`,
  `nonlinear.md`, `general-modal.md`, `material-changes.md` and smaller ones),
  `docs/guide/workflows/luna.md`, `docs/introduction/first-run.md`, `README.md`, the API page of the species, the design
  pages (`architecture.md` §6.3 and §6.6, `accuracy.md`, `algorithms.md`, `testing.md`,
  `roadmap.md`, a note on ADR-010, `dependencies.md`), the layers figure, the
  installation guide; the tutorials that built deleted objects (air mixture, envelope
  plasma, filamentation, the free-space ones, compressor pairs, interactive session,
  vector Raman) and one worked example, `plot_06_planar_compressor.py` (edited, not
  re-run: Phase 4c). `CHANGELOG.md`: "Removed", and "Changed" for the rerouted accessors.
- **Dependency:** pyfluids stays, for one value: the molar mass of CoolProp's
  pseudo-pure dry air, which optmat's equation-of-state object does not expose (Q39).
  §6 allows a direct pyfluids dependency while a direct use remains; no other
  dependency changed, and `uv.lock` is unchanged.
- **Verification:** `ruff check --fix`, `ruff format`, `ruff check` and `basedpyright` (whole
  tree) clean; `codespell` on the changed files clean. The full `pytest -n auto`:
  11 850 passed, 29 skipped, 1 xfailed, none failed (8 min 28 s). The strict docs build
  (`poe docs`) succeeds with `plot_06_planar_compressor.py` held at its committed
  source; with the edit it stops, as it must, naming that worked example (Phase 4c re-runs
  it). The interactive-session and vector-Raman tutorials were run directly. The test
  oracle reproduces the former `ADK` bit for bit (five noble gases, four option sets).
  Not run: `poe check` as a whole, `poe cov`, `test-slow`, the Kokkos lane (end of the
  SEP); the slow CtC and M2 witnesses' bounds were not re-measured on their explicit
  inputs.
- **Sources checked:** the optmat 0.2.0 solid models' coefficients against the former
  sets (identical), and their index against the former by evaluation; optmat's
  ionisation parameters; no new literature.
- **Limitations:**
  - `TabulatedGlass` keeps no public constructor now that its loader is gone; a
    caller's table is built from its spline fields, or through optmat's
    `TabulatedModel.from_arrays` and `OptmatGlass` (Q41).
  - `raman_status` reads a response's status by building it at room temperature and
    800 nm (Q42).
  - Two separately built `species("Ar")` compare unequal (a species compares by
    identity; its adapters by digest).
  - The Linux record of the mixture baseline (`mixture_linux_x86_64_baseline.json`) is
    stale and skipped until re-recorded on Linux.
- **Open questions:**
  - Q39. optmat's `EquationOfState` exposes no molar mass, so `Gas.Air`'s mass flow reads
    CoolProp's pseudo-pure air through pyfluids, the last direct pyfluids use. Ask optmat
    for `EquationOfState.molar_mass` (then drop pyfluids), or take the constituents'
    mean ($2.0\times10^{-4}$ lower)? *Resolved:* optmat 0.2.1 exposes `EquationOfState.molar_mass`;
    solprop takes Gas.Air's molar mass from it and no longer depends on pyfluids
    directly (it remains a transitive dependency of optmat's `gas` extra).
  - Q40. Is the test-only ADK oracle with the noble gases' former inputs
    (`tests/rate_oracles.py`) acceptable as historical test input, or should the
    numerics tests move to `OptmatRate` and re-record what depends on the rate?
    *Accepted* by John Travers on 2026-10-02: the oracle stays as historical test input.
  - Q41. Give `TabulatedGlass` a public constructor from arrays, or leave a caller's
    table to optmat's `TabulatedModel.from_arrays`?
    *Decided* by John Travers on 2026-10-02: left to optmat's
    `TabulatedModel.from_arrays`, which the materials manual now names.
  - Q42. Ask optmat for a status accessor of a gas's Raman response, in place of
    building one to read it?
    *Passed to optmat* as a note on 2026-10-02; *resolved* in optmat 0.2.2
    (`raman_response_status`), which `optmat_raman.raman_status` now reads.
  - Q43. The re-baselined comparisons above widen material-comparison bounds to
    optmat's measured offsets (densities, viscosities, ionisation energies, metals);
    review that none of them is an invariant in disguise. *Decided (John Travers,
    2026-10-02):* the density, viscosity, ionisation-energy and argon zero-dispersion
    comparisons stay as documented data comparisons. Three were changed:
    - the CtC witness's helium fit divides out the reference-density offset by
      computing Luna's reference density with CoolProp, so its bound returns to 3e-8;
    - the metal comparison with Luna's tables is deleted, since optmat validates its
      own metals and Luna's tables are no authority;
    - the hypothesis test of the gas index draws each gas up to 100 bar or 1 % below
      the top of its own gas domain, whichever is lower, in place of 5 bar for all.
  - Q44. The Kokkos digest of the coherent capillary's legacy entry point and the Linux
    mixture record need re-recording at the SEP's final lanes.
    *Handled* at the [final lanes](#final-lanes): the Kokkos digest re-recorded on the
    reference build; the Linux mixture record still needs a Linux machine and stays
    skipped. *Closed* at the [close-out](#close-out-the-long-examples-deferred): the
    record was orphaned (its witness retired in Phase 4b) and is deleted.
- **Handoff to Phase 4c:** re-run `plot_06_planar_compressor.py` (edited) and every
  worked example whose numbers moved with Phase 4a (they still import and translate;
  `plot_04_graded_index_multimode.py`'s `fused_silica(source="malitson")` now maps to the
  absolute `malitson_1965/sellmeier`), complete the manuals from *What changed from
  solprop's old material data*, and run the end-of-SEP lanes (Q35, Q44).

### The pin moved to optmat 0.2.1

- **Scope:** 2026-10-02, by Opus 5.5 (AI) for John Travers, on
  `feature/sep-0003-optmat-pin-0.2.1`.
- **Review (§9.2, step 6):** optmat 0.2.1 (tag `v0.2.1`, `fb43af3`) adds
  `EquationOfState.molar_mass`, which closes Q39. It also adds a frequency-dependent
  Coulomb factor as provisional models that are not the default. None of the 84 models
  solprop uses changed its digest.
- **Change:** `OptmatGas.molar_mass` is the equation of state's own for every gas.
  For Gas.Air that is CoolProp's pseudo-pure value, 0.02896546 kg/mol. solprop's
  direct pyfluids dependency and its last pyfluids call are gone.
- **Checks:** ruff and basedpyright on the changed modules, and the affected tests
  (physdata, gas flow, profiles, manifest, resume, baseline, SEPs, package): 1328
  passed and one skipped.

### Phase 4c — documentation, tutorial and worked examples

- **Scope:** §11's Phase 4c and §8, 2026-10-02, implemented by Opus 5.5 (AI) for John
  Travers, on `feature/sep-0003-optmat-materials-4c` from `main` at `08a770a`, with
  `main` at `d591177` (optmat 0.2.1, pyfluids dropped, Q43) merged in. It completes the
  manuals, adds the tutorial, consolidates the changelog and re-runs the worked
  examples on optmat's materials. It changes no package code.
- **Delivered:**
  - `docs/guide/media/materials.md` is the ownership and selection guide. A new section,
    *Who owns what, and how to choose*, gives the split (optmat owns the data, the
    models, their statuses and uncertainty; solprop the propagation physics built on
    them), three ways to choose (nothing, a `MaterialSelection`, an object of one's
    own), and the support matrix of the built-in gases at optmat 0.2.1: index model
    and its shortest qualified wavelength, equation of state and transport, Kerr,
    Raman status, rate form and uncertainty factor; the solids' table, the source
    mapping, statuses, qualified intervals, the dilute-scaling error and the quick
    experiments were already there and are kept. Wording left from the opt-in phases
    ("under a selection", "in this phase") is gone.
  - `ionisation.md`: the recommended-model table at optmat 0.2.1 with each rate's
    threshold, maximum field and weak-field energy per event at 800 nm (evaluated
    through `OptmatRate`); a custom species' refusal stated without the opt-in wording.
  - Smaller edits: `gnlse.md` (the 0.18 of the photon-number check is a given
    fraction), `output.md` (the example manifest of a default capillary run, the
    policies' `None`, "supplied materials only"), `scans.md`, `modes.md`,
    `compression.md`; `docs/introduction/installation.md` (optmat 0.2.1 with its `gas` extra,
    no pyfluids, the version check prints optmat's), `docs/introduction/first-run.md` (a *Materials*
    item), `README.md` (optmat linked), `docs/guide/workflows/luna.md` (Luna's `PhysData` and
    `Ionisation` calls mapped to the optmat accessors and `OptmatRate`).
  - The tutorial `docs/tutorials/basics/plot_06_material_choices.py` (*Compare material
    choices*, about 5 s): it resolves and prints argon's and silica's optmat records,
    the rate's identity, status, uncertainty factor, threshold and maximum field and
    the Raman statuses; modifies the rate with `with_parameters(electrons=2.0)`
    (status `user`, base digest kept); runs the archived `ar_plasma_field` witness
    (ten fixed steps, `numpy_ref`) on the defaults and under three declared changes
    (`linear="borzsonyi_2008/sellmeier_Ar"`, the doubled rate, a Malitson
    `SellmeierGlass` wall); and tabulates, from the runs' manifests, which of
    dispersion, electronic coefficient, Raman, ionisation rate and solid index each
    change touched, with the change of the exit spectrum, ionised fraction and peak
    power. A second table reads the former-to-optmat attribution from
    `tests/data/sep3_attribution.json` in the same five columns, and the figure adds the
    archived former spectrum from `sep3_baseline.npz`; nothing rebuilds the old data.
    The default run reproduces the archived default run's output plane bit for bit
    (SHA-256), which the tutorial prints.
  - `CHANGELOG.md`: the SEP 3 entries consolidated into Added (materials from optmat,
    rates, Raman, the manifest and schema 8), Changed (every built-in material is
    optmat's; the worked examples re-run), Removed and Fixed (the vacuum fill, a rate
    table's energy per event, `gas_ratio`'s dilute GVD); the pre-SEP GNLSE entry's 0.18
    corrected.
  - The SEP header's **Manual pages** field.
- **Worked examples re-run** (`poe docs-examples`, one at a time, Kokkos CPU with 18
  threads; the earlier outputs used 6, so the times are not comparable):

  | Example | Runtime | Notable change |
  |---|---:|---|
  | `plot_00_vuv_dispersive_wave` | 32 s | VUV 7.44 → 9.86 µJ (2.71 → 3.58 %); range filter removed (the band is inside optmat's helium range) |
  | `plot_01_self_phase_modulation` | 28 s | 72 → 67 % of the transform-limited peak; `DiluteScalingWarning` at 5 bar, now named in the text |
  | `plot_02_soliton_self_compression` | 12 s | 1.73 fs/23.6 GW → 1.83 fs/21.5 GW at 1.80 m |
  | `plot_03_four_wave_mixing` | 21 s | re-tuned from 0.48 to 0.445 bar (below): idler 42.8 % of the pump; at 0.48 bar it is now 9.4 % (was 33.3 %) |
  | `plot_04_cross_phase_modulation` | 7 s | seed broadening 8.3 → 11.9 times (neon's Kerr coefficient) |
  | `plot_05_raman_self_frequency_shift` | 2 min 15 s | metric changed (below): soliton at 1748 nm (was 1737 nm) |
  | `plot_06_ionisation_blue_shift` | 14 s | centroid 682 → 639 nm |
  | `plot_01_spm_supercontinuum` | 4 min 48 s | no printed change |
  | `plot_02_soliton_supercontinuum` | 4 min 18 s | no printed change |
  | `plot_01_gradient_compressor` | 3 min 39 s | N 2.59 → 2.63; exit 2.22 → 2.03 fs; HE₁₂–₁₄ about 1e-3 |
  | `plot_02_tapered_capillary` | 1 min 21 s | higher modes at the exit 15–28 % lower |
  | `plot_03_vector_wall` | 5 min 33 s | mode losses within 2e-4; HE₁₂ at the exit 2.16e-2 → 1.97e-2 |
  | `plot_04_graded_index_multimode` | 5 min 4 s | LP indices +3.9e-4 (Malitson's formula, now absolute); first GPI pair 801 → 802 nm |
  | `plot_05_moving_solid_guide` | 4 min 5 s | none (its indices are declared constants) |
  | `plot_06_planar_compressor` | 12 min 27 s | argon n₂ +23 %: exit B 6.9 → 8.7, compressed 11.0 → 9.2 fs at −76 fs², energy share 72.1 → 69.1 %; the seeded beam now breaks up past B ≈ 17 (B 35 at 45 cm); electrons at the peak 1.4e14 → 5.3e14 cm⁻³ |

  Deferred, as the person among the maintainers reserves them:
  `plot_03_modulational_instability_supercontinuum` (166 min last time) and
  `plot_04_raman_comb_supercontinuum` (56 min). Their committed outputs are kept, and
  a note after each one's first paragraph says that the output was computed with
  solprop's former built-in material data and is to be refreshed. Both scripts import,
  build and translate on optmat's materials (checked with `Simulation.from_capillary`
  and `translate()`, 2.4 s and 0.8 s; each raises one `DiluteScalingWarning`, argon at
  40 bar −7.6e-3 and hydrogen at 40 bar −3.4e-3).
- **Fixes in the examples:**
  - *The planar compressor's cycle average.* Phase 4b's edit called
    `linear_cycle_average(OptmatRate("Ar"), a)` for each of 4001 samples of the pulse;
    the adaptive quadrature calls the rate many times per amplitude, 0.56 s each, so
    the cell alone took about 37 min (the first run was stopped, not hung: it sat in
    LAPACK at 100 % of one core). The example now evaluates the same average,
    $(1/2\pi)\int W(|a\cos\theta|)\,d\theta$, by a fixed 96-point Gauss–Legendre rule
    over a quarter cycle in one vectorised call (about 5 s), and asserts agreement with
    `linear_cycle_average` at the peak amplitude to $10^{-9}$. Checked separately at
    8, 12 and 16 GV/m: the two agree to $2\times10^{-15}$. The electron density at the
    peak is $5.3\times10^{14}$ cm⁻³ (former PPT: $1.4\times10^{14}$), 2.9 % of the Kerr
    index change (was 0.9 %); optmat's argon $n_2$ is 1.562e-23 m²/W, 5 % above
    Arnold's.
  - *The Raman self-frequency shift's metric.* The example tracked the argmax of the
    red side (> 1.08 µm), which at the exit is a non-soliton cluster near 1.33 µm,
    present with both data sets but now taller than the soliton (0.84 of the red
    maximum). An attribution at `dc4ca0d` showed the soliton itself at 1.75 µm
    and the ordering restored by reverting optmat's Raman or Kerr coefficient alone.
    The example now follows the longest-wavelength peak of at least 30 % of the red
    maximum (10 % prominence), and prints both: soliton 1748 nm, tallest 1332 nm.
  - *Four-wave mixing re-tuned* (decided by John Travers). With optmat's helium the
    linear phase matching $2\beta(\omega_p) = \beta(\omega_s) + \beta(\omega_i)$ of the
    75 µm capillary (267 nm pump, 800 nm seed, 160.2 nm idler) lies at 0.457 bar; full
    runs of the example at 0.40–0.48 bar give the exit idler 5.4, 20.5, 31.2, 36.4,
    40.5, 42.8, 42.2, 32.2 and 9.4 % of the pump at 0.40, 0.42, 0.43, 0.435, 0.44,
    0.445, 0.45, 0.46 and 0.48 bar, so the pump's Kerr phase moves the optimum to
    0.445 bar, which the example now uses and states.
  - Range-warning filters and their comments: the helium examples' bands (down to 73
    nm with collars) lie inside optmat's helium range (from 60 nm), so the claims that
    the index is extrapolated were removed from five examples, and the filter with
    them where it was named in the text (`plot_00`, the three waveguide examples).
    The other examples keep a harmless filter (Q49).
- **Decisions:**
  - The deferred examples' note is in the docstring, as asked. A text change makes a
    worked example stale, and re-generating its page without running it would drop
    its outputs, so the generated page, its script copy, notebook, zip and checksum
    were updated by a script exactly as sphinx-gallery would for that text: the
    notebook regenerated through sphinx-gallery's own `jupyter_notebook` (which
    reproduces the committed notebook byte for byte before the edit), the page's
    source-line markers shifted by the seven added lines, the zip rebuilt with
    `zip_files` and the checksum with `get_md5sum`. Code and outputs are untouched
    (Q47).
  - The tutorial derives the touched families from the manifests rather than stating
    them, so it also shows how to read a manifest. Its electronic and Raman columns
    come from the archive: no curated alternative exists for the electronic
    coefficient through the selection, and argon has no Raman response.
  - The tutorial runs in every docs build (about 5 s) and reads the archive from a
    source checkout; downloaded alone it prints that the records are in the
    repository and skips the historical table.
- **Verification:** ruff check and format on the changed Python; codespell and the
  public-text scan on the changed files; `pytest tests/test_seps.py` (92 passed,
  1 skipped); `tests/test_guide_snippets.py`, compiled and the installation page run;
  the tutorial run directly (4.8 s; bit-for-bit default); the deferred scripts
  translated; the strict `poe docs` build succeeds (no worked example stale; the tutorial ran in
  2.6 s and printed the bit-for-bit match). Not run: `poe check`, `test-slow`, `cov`,
  the Kokkos lane (end of the SEP).
- **Sources checked:** none new. Every number is a run of solprop or an evaluation of
  optmat 0.2.1's public models (unchanged digests from 0.2.0), or the archived records.
- **Limitations:**
  - The per-directory `sg_execution_times.rst` files list only the last example run
    in each directory, since every example was run on its own.
  - The two deferred examples still show the former materials' results.
  - The tutorial is a 0.4 mm witness: it ranks material changes; it does not size
    them for a long run.
- **Open questions:**
  - Q45. A capillary run with plasma under `linear=<model>` lists optmat's
    recommended `fit/Ar` in its manifest beside the selected model: the
    plasma response's species is rebuilt from the gas's name with the default
    selection (`bind_rate_energy`, `species(name)`), although only its density and
    energy act. Should the response keep the selected species, so that the manifest
    lists only what acted?
    *Decided* by John Travers on 2026-10-02: yes; implemented in the [final
    fixes](#final-fixes-before-final).
  - Q46. `linear_cycle_average` costs about 0.56 s per amplitude on an `OptmatRate`
    (the adaptive cycle integral evaluates the rate on many small arrays). Vectorise
    it, or document the cost?
    *Decided* by John Travers on 2026-10-02: vectorise; implemented in the [final
    fixes](#final-fixes-before-final). optmat 0.2.2 also makes its rates three to four
    and a half times faster, with the same numbers to the last bit.
  - Q47. Is the scripted text update of the two deferred examples' generated pages
    acceptable until they are re-run, or should the note move to the gallery's
    section text?
    *Accepted* by John Travers on 2026-10-02, until the two examples are re-run.
  - Q48. The four-wave-mixing example's pressure is now optmat's optimum (0.445 bar),
    no longer the 0.48 bar of the original set-up; should the text cite the
    experiment's own pressure?
    *Decided* by John Travers on 2026-10-02: yes, in one sentence; implemented in the
    [final fixes](#final-fixes-before-final).
  - Q49. Several process examples keep a `MaterialRangeWarning` filter that no longer
    catches anything; remove them at the next refresh of those examples?
    *Decided* by John Travers on 2026-10-02: yes; implemented in the [final
    fixes](#final-fixes-before-final).
- **Handoff:** the end-of-SEP lanes (`poe check`, `test-slow`, `cov`, the Kokkos
  lane; Q35, Q44), the two deferred examples' re-runs by the person among the
  maintainers, and the SEP's move to `Final` (open questions to the roadmap or issues).

### Final fixes before Final

- **Scope:** the open questions John Travers decided on 2026-10-02 that change code,
  tests or examples before the SEP goes `Final`, implemented by Opus 5.5 (AI) for John
  Travers on `feature/sep-0003-optmat-materials-final` from `main` at `629237d`
  (optmat 0.2.1). The remaining questions are marked where they were asked.
- **Delivered:**
  - *Q29, the dilute-scaling warning.* `physdata/mixtures.py`:
    `DILUTE_SCALING_THRESHOLD` is $10^{-2}$ (was $10^{-3}$) until SEP 11 replaces the
    law; every run still records its measured error in the manifest. At 800 nm and
    294 K argon and nitrogen pass the bound near 52 bar, krypton near 32 bar, xenon
    near 17 bar and SF₆ near 15 bar. `materials.md`, `output.md`, the changelog and
    the self-phase-modulation example's text (whose 5 bar argon, $-1.1\times10^{-3}$,
    no longer warns) follow; `tests/physdata/test_optmat_gases.py` checks a fill
    below the bound that records without warning and one above it.
  - *Q45, the plasma response's species.* `driver/simulation.py`:
    `_capillary_responses` gives the plasma response the fill's species as the
    selection resolved it, as `responses_for` already did for a mixture, so
    `bind_rate_energy` no longer rebuilds it from the gas's name with the default
    selection. Under `linear="borzsonyi_2008/sellmeier_Ar"` a plasma run's manifest
    no longer lists `fit/Ar`
    (`tests/driver/test_material_manifest.py`).
  - *Q46, the cycle average.* `nonlinear/envelope_plasma.py`: the new
    `linear_cycle_averages` takes an array of amplitudes (`linear_cycle_average`, for
    one, now calls it) and evaluates the rate **once**, on a
    32-point Gauss–Legendre rule over each piece of the quarter cycle between the
    field bounds the rate declares (its threshold or `field_min`, `field_max`,
    `maximum_field`, crossed at $\arccos(E_b/a)$), with an embedded 24-point rule
    from the same call. Where the two agree to the adaptive quadrature's $2\times10^{-10}$
    relative, the 32-point value is returned; elsewhere (a corner the rate does not
    declare, such as the test oracle's ceiling, or a rate table's spline knots) that
    amplitude takes the adaptive quadrature. The orders come from the check, not a
    guess: the 24-point rule alone is within $5\times10^{-13}$ of the adaptive
    average for optmat's argon, helium, xenon and nitrogen rates at 800 nm from
    $1.001\,E_\mathrm{th}$ to $3\,E_\mathrm{max}$, the hold corner included, so the
    check passes for these rates without a fallback (52 amplitudes each, the 32-point
    value within $3.2\times10^{-13}$ of the adaptive one) and the returned rule is
    above it. On `OptmatRate("Ar")` one amplitude costs 1.8 ms (the adaptive integral
    0.57 s) and 4001 amplitudes 2.4 s. The planar-compressor example calls
    `linear_cycle_averages` in place of its own 96-point copy; that cell was run alone
    and prints the same electron density ($5.3\times10^{14}$ cm⁻³, 2.9 % of the Kerr
    index change), and the example was not re-run.
  - *Q34, the Φ acceptance test.* `tests/translate/test_linop.py`: the gradient
    fixture is optmat's helium (no longer Ermolov's three-pole law as a caller's), and
    each interval is asserted against its own bound, derived rather than chosen:
    `build_phi` resolves $D$ to `max(DEFAULT_TOL, ROUNDOFF_SAFETY * floor)` of its
    value scale, so over $[z_1, z_2]$ the table departs from the exact Φ by at most
    that tolerance times $\max|D|\,(z_2 - z_1)$, to which the oracle's own error
    estimate is added. Here the floor is $2.8\times10^{-10}$, the bound
    $2.8\times10^{-8}$ of the value scale, and the worst deviations
    $4.6\times10^{-11}$ (expansion) and $1.37\times10^{-10}$ (z table). The stale
    1.4e-12 comment is gone; `tests/data/convergence/phi_quadrature.json` is
    re-recorded with a `bound` column, which `docs/dev/accuracy_figures.py` draws,
    and `docs/dev/accuracy.md` quotes the new numbers.
  - *Q27, planar fills.* `physdata/species.py`: `fill_species(gas, susceptibility)`
    is the single species a named fill carries: the built-in gas, with a given optmat
    law of the same gas in place of its own. `modes/fill.py` (`GasFillMixin`, so the
    slab) and `spatial.py` (`SpatialMedium`) build their density path on it, so the
    equation of state is optmat's and the species recorded is the one whose index
    acts; `GasFillMixin.fill_species` returns it. `driver/material_output.py` records
    each distinct single-species fill's species and dilute-scaling check, as for a
    mixture's population, once for the modes that share it. A caller's own law of
    another kind leaves the built-in species, and every existing hash, as it was.
  - *Q28, dropped loss per run.* `physdata/selection.py`: `MaterialLog.dropped`
    keeps, per transparent model and bound, the largest extinction and absorption a
    run dropped (`record_dropped`, through `_optmat.record_dropped` in every open
    context); `OptmatGlass.run_transparency(log)` reads it, and the manifest's
    `solids` take it from the run's log. The adapter keeps its lifetime record
    (`transparency`), and the warning inside a run is decided by the run's own
    evaluations (`tests/physdata/test_optmat_solids.py`).
  - *Q49 and Q48, the process examples.* Each process example was run with its
    `MaterialRangeWarning` filter turned into a recorder: the filters of
    `plot_01`, `plot_02`, `plot_04`, `plot_05` and `plot_06` caught nothing and are
    removed; `plot_03`'s catches helium's Kerr fit outside 400 nm and the rate's
    267 nm carrier below its 350 nm comparison and stays. The four-wave-mixing text
    now says that the original set-up used 0.48 bar and that optmat's helium index
    moved the pressure to 0.445 bar. The six were re-run in one `poe docs-examples`
    build (Kokkos CPU at 8 threads): every printed result is unchanged (the
    four-wave-mixing idler is 42.76 % of the pump), the self-phase-modulation run no
    longer prints a `DiluteScalingWarning`, and the runtimes are 18 s, 7 s, 4 s, 4 s,
    2 min 18 s and 8 s for `plot_01` to `plot_06`.
  - `docs/guide/media/materials.md` names `TabulatedModel.from_arrays` as the way to a
    caller's table (Q41).
- **Deviations:** none from Part I. The planar-compressor page's code change was
  carried into its generated page by the same scripted update as Phase 4c's deferred
  examples (the notebook through sphinx-gallery's own writer, which reproduces the
  committed notebook byte for byte before the edit; the page's line markers shifted;
  the zip and checksum rebuilt), since only the code of one cell changed and its
  output was checked equal.
- **Decisions:** the cycle average falls back per amplitude rather than raising when
  its check fails, so a caller's rate with an undeclared corner keeps the adaptive
  accuracy; the fill's species takes a given law only when it is an optmat law of the
  same gas, so a caller's legacy law and the hashes built on it are unchanged.
- **Verification:** ruff check and format, basedpyright (0 errors), the numpydoc
  lint, codespell and the public-text scan on the changed files; the affected modules
  under xdist at eight workers, `tests/physdata`, `tests/modes`, `tests/output`,
  `tests/nonlinear/test_optmat_rates.py`, the envelope-plasma tests, the material
  manifest, slab, free, radial, capillary and rate driver tests,
  `tests/translate/test_linop.py`, the SEP 3 baseline, defaults and attribution
  tests, the accuracy figures and `tests/test_seps.py` (all pass); the cycle average
  against the adaptive one as above; the strict `poe docs` build (succeeds). Not run:
  `poe check`, `test-slow`, `cov`, the Kokkos lane (the SEP's final lanes).
- **Sources checked:** none new; every number is an evaluation of optmat 0.2.1 or a
  run of solprop.
- **Limitations:** the planar-compressor example's outputs predate the change (its
  printed numbers were checked equal); a rate's ceiling on its value is not declared,
  so amplitudes past it take the adaptive integral.
- **Open questions:** none new.
- **Handoff:** the end-of-SEP lanes (`poe check`, `test-slow`, `cov`, the Kokkos lane;
  Q35, Q44), the two deferred examples' re-runs, and the move to `Final`.

### Final lanes

- **Scope:** the four end-of-SEP lanes (`poe check`, `test-slow`, `cov`, the
  Kokkos-selected slow lane), run on `main` at `629237d` (optmat 0.2.1), and the fixes
  of their failures, 2026-10-02, by Opus 5.5 (AI) for John Travers on
  `feature/sep-0003-optmat-materials-gate`, with `main` at `8e00779` (the final fixes)
  merged in before the lanes were re-run.
- **Failures and their fixes.** Each is classed as (a) a witness that had picked up an
  optmat default where its own input must be supplied, (b) an intended record change,
  or (c) a defect. No bound was widened and no invariant moved.
  - *The run-record hash* (`tests/test_display.py`, `check` and `cov`), (b). The pin to
    optmat 0.2.1 put its version string in 7 version fields and 17 citations of the
    hashed record (`040576c2...`; with `0.2.0` written back the record hashes to
    Phase 4b's `32f86dae...` exactly), and the final fixes' dilute-scaling threshold
    ($10^{-3}$ to $10^{-2}$) is the one other changed field. Re-recorded as
    `4119f06c...`, both causes named in the test.
  - *The M2 witnesses* (`tests/driver/test_prop_capillary.py`: the fundamental shape,
    the VUV band, the higher modes, the step-tolerance separation, the quadrature
    ladder, the order study, the backend parity, the statistics-channel case), (a).
    Since Phase 4a the reduced and full M2 cases named `"He"` and so took optmat's
    Kerr coefficient and rate; Phase 4b supplied only Luna's helium index. Measured on
    the reduced case at `dc4ca0d`, one family on the former registry at a time
    (fundamental shape against the stored 0.05685): all registry 0.05685 (the record
    exactly), none 0.0446, the former Kerr coefficient alone 0.0204, the former rate
    alone 0.0823. optmat's helium coefficient at 800 nm is 2.0 % above Luna's. The
    witness now runs on a caller's helium fill (`luna_helium_fill`: the density path
    a named gas gets, Luna's Ermolov law, Luna's hyperpolarisability
    $4\times3.43\times10^{-28}/N_\mathrm{1\,atm,0\,^\circ C}$ from CoolProp, which is
    the former registry's $5.109203705817757\times10^{-53}$ exactly) and on
    `ppt_oracle("He", 800e-9)`, a new test-only `PptOracle` in `tests/rate_oracles.py`
    (Q40's rule extended to the PPT form). It is the former default `PPT` bit for bit,
    checked at `dc4ca0d` over the five noble gases, three wavelengths, four option
    sets and the plain form (4001 fields each): Luna's PPT, with the Stark shift and
    the ion's dipole factor, times Tong and Lin's barrier-suppression factor, which
    Luna lacks (the departure `accuracy.md` section 5 already names). Every M2 record
    and bound then holds unchanged.
  - *The CtC plane-zero state* (`test_the_two_codes_start_from_the_same_state`), (a):
    the independent mode rebuild read optmat's hydrogen rather than the run's
    `luna_hydrogen()`; it is supplied. (The first lane also failed it on the
    $10^{-3}$ dilute-scaling warning, which the final fixes' $10^{-2}$ removed.)
  - *The statistics-channel cost* (44.3 % against the 25 % gate), neither: load. Run
    alone it passes, and measured in one process on optmat's rate and on the oracle it
    costs the same, 14.3 % each.
  - *Range warnings the former registry raised* (`test_air_tutorial.py`,
    `test_mixture_connected.py` twice, `test_general_modal.py`'s collar test), (b):
    each expected a `MaterialRangeWarning` because the registry's air or argon index
    stopped short of the band (argon's at 1 µm); optmat's cover these bands, so the
    expectations are removed. The air tutorial's run
    still warns, for its 6 bar of air above optmat's validated Raman density, and the
    test now expects that one by name.
  - *The step-index strand* (`test_stepindex_gnlse.py`, two slow tests), (b): the
    default core is optmat's `fit/SiO2`, whose collar reaches the 2.73 µm OH band,
    which the transparent approximation drops with a warning (Q3), as the medium test
    already expected; the slow runs now expect it. The user-table test's index error
    was $8.5\times10^{-8}$ against its $10^{-9}$ bound, all of it at 2704 nm, the OH
    band's real-index structure that a 6.4 nm knot spacing does not resolve
    ($7.1\times10^{-10}$ elsewhere, the former measurement). The test is of the
    interface, so it runs on Malitson's silica (Luna's `:SiO2`, the core it was
    measured on: $7.04\times10^{-10}$) and keeps its bounds.
  - *The `phi_quadrature` z table* (`test_linop.py`), (b): fixed by the final fixes'
    Q34 before these were re-run.
  - *Q44, the Kokkos digest* of the coherent capillary's legacy entry point, (b): not
    a lane failure (a local build checks its parity bound), re-recorded on a
    reference build of solprop-kokkos (`SOLPROP_KOKKOS_PROFILE=REFERENCE`, built in a
    scratch directory and put first on the path): `cbf22776...` in place of
    `60e5a39a...`. The same build gives the former digest at `dc4ca0d` under the
    registry selection and the unchanged `prop_gnlse` digest here, so the materials
    are the whole cause.
  - *The scan-queue timeouts* (`test_scan_queue.py`, two, under the loaded `cov`
    lane): known, as before; both pass alone and are unchanged.
- **Coverage.** The six uncovered places: `driver/simulation.py`'s planar branch for a
  caller's optmat law of another kind (now tested: it is kept); `optmat_raman.raman_status`
  for a molecule optmat curates no Raman response of (CO, O₃: `"none"`, now tested);
  `SilicaTensorResponse`'s printing of a caller's Lin–Agrawal set (now tested);
  `species._optmat_ionisation`'s refusal and `tools._capillary_at`'s fill-less mode,
  both dead after Phase 4b (optmat curates an energy for every built-in gas but air,
  which both callers refuse first; the design capillary always carries its fill), so
  deleted, with `_capillary_at`'s unused gas argument.
- **Q35.** No ten-step qualification case failed on optmat's rates in either slow
  lane, so none needed more steps.
- **Other changes:** `docs/dev/accuracy.md` section 3 names the witnesses' oracles.
- **Verification:** ruff check and format, basedpyright, the numpydoc lint and
  codespell on the changed files; the affected tests directly; the reference Kokkos
  build on `test_coherent_capillary.py` (92 passed) and the local one on the digest
  test. The lanes on the merged tree (`main` at `8e00779` with this branch):
  `test-slow` 137 passed and 1 skipped (19 min); the Kokkos-selected slow lane 138
  passed (27 min). `poe cov` and `poe check` on the merged tree were stopped to unload
  the machine and still need to run; the coverage fixes above were checked by the
  affected tests only.
- **Sources checked:** none new; Luna's `PhysData.gamma3_gas` and `IonRatePPT` defaults
  for the witness's inputs.
- **Limitations:** the Linux mixture record (Q44) is still stale and skipped;
  `PptOracle` has the smooth channel sum and Tong and Lin's correction only, the
  former `PPT`'s literal sum and Zhang's correction being unused by any witness.
- **Open questions:** none new.

### The pin moved to optmat 0.2.2

- **Scope:** 2026-10-02, by Opus 5.5 (AI) for John Travers, on
  `feature/sep-0003-optmat-pin-0.2.2`.
- **Review (§9.2, step 6):** optmat 0.2.2 (tag `v0.2.2`, `b6571f2`) resolves Q32, Q33,
  Q37, Q38 and Q42 and speeds up its rates (Q46). Of the models solprop uses, eight
  changed their digests: `transport/D2` (the correlation in place of a spline through
  Assael et al.'s table; values move by up to $5.1\times10^{-4}$, $1.2\times10^{-3}$ at
  90 K) and the six formula solids made absolute (CaF₂, MgF₂ o and e, BaF₂, Al₂O₃ o
  and e, N-BK7; `FormulaModel.EVALUATION_VERSION` 2). Their indices are unchanged
  (0.0 difference) and $\mathrm{d}n/\mathrm{d}\omega$ moves by at most
  $3.6\times10^{-8}$ relative, the former jitter. Every rate's digest and value is
  unchanged.
- **Change:** `optmat_raman.raman_status` reads `raman_response_status` in place of
  building a response at room temperature and 800 nm. The strict xfail of D₂'s
  viscosity series is removed: the series now converges over D₂'s whole range. The
  run-record hash is re-recorded for the version string alone (with `"0.2.1"` written
  back it hashes to the former literal exactly). The manuals name 0.2.2.
- **Checks:** ruff and basedpyright on the changed modules; the temperature-profile,
  display and physdata tests (1370 passed).

### Close-out: the long examples deferred

- **Scope:** 2026-10-02, by Opus 5.5 (AI) for John Travers, on
  `feature/sep-0003-close`.
- **The two long examples.** `plot_03_modulational_instability_supercontinuum` (eight
  shots) and `plot_04_raman_comb_supercontinuum` were started on the Linux workstation,
  where under its load one shot of the ensemble took about 1 h 49 min (185 min for all
  eight before, on an unloaded machine). John Travers decided to defer both re-runs:
  solprop is not released. Both runs were stopped and no output was kept. The two
  scripts and their generated pages keep Phase 4c's note (Q47) that they still show
  the former materials' results, until they are re-run with `poe docs-examples`.
- **Q44.** The orphaned Linux mixture record (`tests/data/mixture_linux_x86_64_baseline.json`)
  is deleted: no test reads it since Phase 4b retired its witness.
- **Still open before `Final`:** the end-of-SEP lanes on `main` (`poe check`, `poe cov`,
  `test-slow`, the Kokkos-selected slow lane) after the pin to optmat 0.2.2, and the two
  examples' re-runs.

### Final

- **Scope:** 2026-10-04, by Opus 5.5 (AI) for John Travers, on `docs/sep-0003-final`.
- **Lanes.** The lanes started on 2026-10-03 (`main` at `09e5a99`) found two docstring
  lines over the line limit and three uncovered paths of the vectorised cycle average
  (a rate with no declared bounds, and its two refusals of a malformed rate); both were
  fixed and merged (`53d1c81`), and the rerun was stopped to free the machine. The full
  lanes (`poe check`, `poe cov`, `poe test-slow` and the Kokkos-selected slow lane) have
  since passed on `main` with this work merged, run by the work merged after it.
- **Completion (SEP 0, §6).** Every phase is merged and reported. The materials run on
  `numpy_ref`, `numba_cpu` and the Kokkos CPU build, qualified in Phase 2 and by the
  slow lanes; the kernel is unchanged (schema 15). The manual pages in the header
  document it; the two it produced (*Materials* and *What changed*) now end with a link
  to this SEP, and the others link to it where they describe optmat's models.
- **Left open, moved to the roadmap:**
  - the development pin of optmat (a local tag), to be replaced by the published
    release before solprop is published (§6, G10);
  - the re-runs of the two longest worked examples (the modulational-instability
    ensemble and the Raman comb), which keep their note that they show the former
    materials' results (Q47);
  - the follow-ups of D14, already on the roadmap: device data, CO₂ then CO, gold and
    copper, foreign-gas Raman broadening (optmat), the Raman physics optmat has not
    curated, the mixture density (SEP 11, Q4) and the `Air` entry of a hand-built
    mixture.
- **Open questions:** none; all 46 are answered, decided, resolved upstream or moved to
  the roadmap above.
- **Status:** Final, approved by John Travers on 2026-10-04; the SEP is archived.

## Part III — History

- 2026-09-27: Drafted from the gas/silica material and consumer audit. Scope
  confirmed as gases and silica, with remaining materials as follow-up; historical
  reproduction uses old environments; full nonlinear dispersion is deferred to a
  separate SEP.
- 2026-09-27: Expanded the Raman change inventory and ionisation adapter mapping
  against the upstream integration assessment and its public curation reports.
- 2026-09-29: Revised and retitled for optmat's recommended solid models, rate models
  and Raman response objects. Re-audited at new revisions, with §9.1 recomputed.
  Recorded that solprop deletes its rate formulas, that mirror curves are device
  data, and that the paired silica Raman fraction is used.
- 2026-09-29: Split. The low-rank Raman representation moved to
  [SEP 10](../sep-0010-low-rank-raman.md), and the finite-density gas law and mixture
  optics to [SEP 11](../sep-0011-finite-density-gas-index.md). This SEP became a data
  migration with the kernel unchanged:
  - Raman responses on the existing seam (D2, revised);
  - linear density scaling with a measured error (D5);
  - optmat design work before implementation (D6);
  - one-line quick experiments (D7).

  Provenance was simplified to identities, policies and the optmat version. The
  upstream gaps were regrouped as Phase 0 work (G1–G10).
- 2026-09-29: Recorded four decisions, D8–D11, which closed the former questions on
  weak rate defaults, the silica tensor bundle and `Gas.Air`:
  - plasma energy per event is the rate's own threshold;
  - `Gas.Air` is the summed polarizabilities of the named composition;
  - the weak rate defaults stay, flagged;
  - the silica tensor default is a paired, gain-checked Lin–Agrawal variant.

  The remaining questions were renumbered Q1–Q4.
- 2026-09-29: After solprop's GNLSE fraction rule changed on main (`raman_fraction`
  defaults to `None`: 0.18 with Raman on, 0 with Raman off; an explicit fraction is
  always held back), D4 was narrowed. The bundle's $f_R$ replaces only the inline
  0.18, and the rule is cited rather than redecided.
- 2026-09-29: Recorded D12–D15, which closed every open question:
  - both yield comparisons are presented, and non-linear polarisation stays admitted
    with a warning when the Keldysh parameter exceeds 1;
  - N2O's provisional Raman is opt-in, and no new gases are added;
  - follow-up items are created at acceptance;
  - solid photoionisation rate models move upstream, with SEP 5 revised.
- 2026-09-29: Recorded the reviews by John Travers (human) and Opus 5.5 (AI) in the
  new review fields; review status `Reviewed`.
- 2026-09-30: Revised against optmat's OEP 1 review (recorded in optmat's register [8]), with
  John Travers's decisions of that day:
  - D8: solprop's kernel carries one energy per population, so the noble gases' loss
    uses the rate-weighted energy at the peak field, with its bound recorded;
  - D9: `Gas.Air` is `"dry air (N2, O2, Ar)"`, with the density of CoolProp's
    pseudo-pure dry air; the `"air"` string resolves to `"dry air (N2, O2)"`;
  - D11: the paired Lin–Agrawal variant failed its gain check (−17.8 %), so the tensor
    default is `lin_2006` with its own $f_R$ and the fitted total $n_2$ (−7.9 %);
  - the Raman width law is evaluated at the actual density, vacuum included, through
    optmat's finite zero-density limit, in place of a density floor;
  - the density and its inverse refuse condensed states;
  - §3.1's rate uncertainty factors of O2 and CH4 updated, and §9.1 marked for
    re-measurement at Phase 0's pinned commit.
- 2026-10-01: Re-reviewed by Opus 5.5 (AI) against optmat 0.1.0 (tag `v0.1.0`), after
  optmat's OEP 1 and OEP 4 became Final. The review checked their records, the page for
  propagation codes and read-only evaluations of the public surface. It found G1–G10
  closed by APIs that match this proposal, and revised:
  - the status, §3.1 and §3.2 to the delivered surface, including the molecular fits'
    qualified infrared windows;
  - the warning policy for qualified intervals and the status policy (§5);
  - the Raman density check, which uses only the non-zero densities, and the
    summed-density decay path of a composition (§4.4);
  - the silica tensor channel map (§4.5);
  - the reference field of the constant energy loss, and the handling of optmat's
    maximum-field warning during tabulation (§4.6);
  - the override sketches, now optmat's own `with_parameters` (§5);
  - the references, pinned to the tag.

  John Travers decided D16 (Raman mixtures need curated partners), D17 (the
  partial-pressure density stays, with SEP 11's Q4 open) and D18 (one backend
  qualification phase, no staging refusal, and Phase 4 in three parts), and chose a
  development-only pin of the tag (§6).
- 2026-10-01: Accepted by John Travers. Part I is frozen.
- 2026-10-01: Implementation started. The solprop side of Phase 0 (the dependency, the
  baseline witnesses and the re-measurement of §9.1) is reported in Part II.
- 2026-10-01: Phase 1a, first part (the selection, the bridge, the gas, equation-of-state,
  transport and solid adapters, all opt-in), reported in Part II.
- 2026-10-01: Phase 1a, second part (the material manifest, output schema 8, resume and
  scan identity, and the drivers' `materials=` keyword, all opt-in), reported in Part II;
  Q4 resolved there, and Q5 and Q6 opened.
- 2026-10-01: Phase 1b (the Kerr coefficient at the carrier and fused silica's bundle,
  opt-in), reported in Part II; Q7–Q10 opened.
- 2026-10-01: Phase 1d (the ionisation-rate adapter, its reference field, energy
  binding, maximum-field handling, manifest entries and post-run diagnostic, all
  opt-in), reported in Part II; pre-deletion parity met; Q12–Q16 opened, Q16
  resolved after the physics review.
- 2026-10-01: Phase 1c (the gas Raman responses from optmat on the existing seam, one
  response per composition driven by its labels' summed density, the D16 refusals, H₂'s
  shift rule and the range warnings, all opt-in), reported in Part II; Q17–Q19 opened.
- 2026-10-01: Phase 2 (backend qualification: the translated optmat problems on
  `numba_cpu` and a fresh Kokkos CPU build against `numpy_ref`, no backend change),
  reported in Part II; Q22–Q24 opened.
- 2026-10-02: Phase 2 follow-up (Q22: an optmat solid and gas evaluated once per
  frequency array, and the capillary route's frame tables sampled once per frequency
  vector; bitwise unchanged), reported in Part II; Q22 answered, Q25 opened.
- 2026-10-02: Pre-defaults fixes (Q3 automatic transparency on lossless routes, Q5
  planar runs under a selection, Q6 verified, Q7 design tools under a selection, Q8
  the tensor bundle, Q13 the post-run annotation; decided by John Travers, all
  opt-in), reported in Part II; Q26–Q28 opened.
- 2026-10-02: The pin moved to optmat 0.2.0, whose rate work Phase 4a waited for; Q12 resolved
  upstream.
- 2026-10-02: Phase 4a, first part (the defaults switched to optmat, a temporary
  `"registry"` value per selection family, Q26 value equality, Q9, Q14 and Q17, the tests
  re-baselined), reported in Part II; Q29–Q35 opened.
- 2026-10-02: Phase 4a, second part (the changed results of the Phase 0 witnesses
  attributed one material family at a time, with step refinement; the solids' GVD
  changes; the rate ratios decomposed into §9.1's factors; Q31 answered in the
  attribution script; the manual page of what changed), reported in Part II; Q36–Q38
  opened.
- 2026-10-02: Phase 4b (solprop's material database, its rate formulas and the temporary
  registry handle deleted; the solid factories' former source names mapped or refused;
  the bare accessors rerouted to optmat, Q30; the tests reworked on explicit historical
  inputs), reported in Part II; Q39–Q44 opened.
- 2026-10-02: Q43 decided by John Travers: the CtC helium fit back to its 3e-8 bound, the
  Luna metal comparison deleted, and the gas-index hypothesis test drawn to each gas's
  own domain.
- 2026-10-02: The pin moved to optmat 0.2.1; Q39 resolved and the direct pyfluids dependency
  removed.
- 2026-10-02: Phase 4c (the materials manual as the ownership and selection guide, the
  ionisation, Raman and other manuals brought to optmat 0.2.1, the material-choices
  tutorial, the changelog consolidated, the worked examples re-run on optmat's
  materials except the two longest, the four-wave-mixing example re-tuned to 0.445 bar
  and the Raman soliton tracked by its own peak), reported in Part II; Q45–Q49 opened.
- 2026-10-02: Final fixes (Q27 planar fills as one species, Q28 dropped loss per run,
  Q29 the warning at 1 %, Q34 the Φ test's derived bound, Q45 the plasma species,
  Q46 the vectorised cycle average, Q48 and Q49 the process examples), reported in
  Part II; the questions John Travers decided that day, and those resolved earlier or
  passed to optmat, marked where they were asked.
- 2026-10-02: Final lanes: the M2 witness on Luna's helium hyperpolarisability and a
  test-only oracle of the former PPT, the CtC plane-zero check on Luna's hydrogen, the
  former registry's range warnings dropped from four slow tests, the step-index table
  test on Malitson's silica, the display hash and the Kokkos legacy digest (Q44)
  re-recorded for their named causes, two dead branches deleted, three tested;
  reported in Part II.
- 2026-10-02: The pin moved to optmat 0.2.2; Q32, Q33, Q37, Q38 and Q42 resolved upstream,
  the D₂ viscosity xfail removed and the run-record hash re-recorded for the version.
- 2026-10-02: Close-out: the two long examples' re-runs deferred by John Travers (their
  notes kept), the orphaned Linux mixture record deleted (Q44 closed).
- 2026-10-04: Final, approved by John Travers; archived.
