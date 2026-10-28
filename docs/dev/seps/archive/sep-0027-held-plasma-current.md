# SEP 27 — An edge-free field-grid plasma source: hold the current at the temporal collar

| | |
|---|---|
| **SEP** | 27 |
| **Title** | An edge-free field-grid plasma source: hold the current at the temporal collar |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-19) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-19 |
| **Resolution** | Accepted by John Travers, 2026-10-19; Final, 2026-10-20, under the recorded closing scope amendment |
| **Requires** | [SEP 18](sep-0018-band-exact-plasma-integrals.md) (the plasma column's running-integral rule); [SEP 26](sep-0026-edge-free-tdse-source.md) (the held-current identity, the analytic collar derivative and the steady-current certificate) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Phases 1–5 complete on `feature/sep-0027-phase-5`; native backend on `feature/sep-0027-phase-3`; locally committed, awaiting integration |
| **Manual pages** | [Field plasma](../../../guide/media/field-plasma.md), [grids](../../../guide/setup/grids.md), [Luna differences](../../../guide/workflows/luna.md), [output](../../../guide/results/output.md) |


## Part I — Proposal

### 1. Abstract

The field-grid plasma response (`PlasmaResponse`, the Geissler-type rate model) multiplies
its polarisation $P=\int J\,dt$ by the fine temporal collar $g$ before the transform.
After the pulse the free-electron current is steady, so $P$ keeps growing and the collar
switches it off. In current variables the source then contains the collar's switch-off,
which no electron emits: a steady current radiates nothing at $\omega\neq0$. SEP 26
removed the same artefact from the TDSE source and measured it on the rate route offline:
3.4–7.4% of the 90 nm–1.6 µm source and 8.4–18% of 0.9–1.6 µm at SEP 20's late L6 planes,
and about $1.5\times10^{-5}$ at 90–200 nm.

This SEP **replaces** the gated polarisation, as the only behaviour, by the transform of
a current *held* at its collar-weighted value. Every time variation of $J$ inside the
window is kept, and only the switch-off is removed. The source is formed directly from
the sampled current in frequency,

$$
A_J=\frac{\mathrm{DFT}[gJ]}{i\omega}-\frac{\mathrm{DFT}[g'J]}{(i\omega)^2}
\quad\text{on the active bins},
$$

which is exact to rounding on resolved collars, under either running-integral rule. The
column's third running integral, the one that formed $P$, is no longer needed. There is
no public option and no compatibility path (§9). The truncating form survives only as a
private, labelled, non-resumable comparison hook for validation.

Success means:

- the analytic and fixed-field tests close within budgets derived in advance;
- SEP 26's fixed-field measurement is reproduced by the production form;
- `numpy_ref`, `numba_cpu` and Kokkos CPU agree within SEP 25 budgets;
- propagated before/after comparisons are reported with their integration-error bounds
  and controls.

### 2. Motivation and scope

**The problem.** Most of the maintainer's propagation runs, and the parents and baselines
of SEP 12, SEP 20 and SEP 26, use the field-grid rate plasma. Its near-infrared source
carries a spurious collar term whose size grows with the plasma's residual drift and with
the delay between ionisation and the trailing taper. On the run_short case (60 fs
requested window, 21 fs taper on a 2048-sample, 50 as lattice) the effect is large. SEP 26
Phase 2 measured it on fixed saved fields of the L6 rate run (reference 5):

| plane | band (90 nm–1.6 µm) | 0.9–1.6 µm | 0.9–1.2 µm | 1.2–1.6 µm | 90–200 nm |
|---|---|---|---|---|---|
| 3.969 m | 3.37% | 8.40% | 3.24% | 18.7% | $1.8\times10^{-5}$ |
| 4.05 m | 7.36% | 18.3% | 7.15% | 38.2% | $1.4\times10^{-5}$ |

These are $\lVert Q_{\rm route}-Q_{\rm held}\rVert_B/\lVert Q_{\rm route}\rVert_B$ in the
assembled source units of §4.4, with $Q_{\rm held}$ built as $A_J$. The running-integral
closure residual was $3$–$10\times10^{-7}$ and the collar's certificate
$1.8\times10^{-14}$. This was a fixed-field source comparison: no propagation was run.

**In scope.**

- The held source on `numpy_ref`, `numba_cpu` and the Kokkos CPU build, on every route
  that already admits the field-grid plasma (§3), for one or several channels, either
  running-integral rule and every admitted mix with Kerr and Raman.
- Its discrete definition, certificate, rounding budget and work identity; the seam and
  saved-output changes.
- Validation: analytic tests, a fixed-field replay of SEP 26 §4.7, propagated before/after
  comparisons, backend parity and the effect on existing witnesses and benchmarks.

**Out of scope.**

- Kerr, Raman and dispersive-Kerr gating, which stay as they are (§4.2).
- The envelope plasma (`EnvelopePlasmaResponse`, §4.2) and the TDSE source.
- A field plasma on the general modal route, which refuses it today and keeps refusing
  it.
- Collisions, recombination, an explicit drift-current model and THz emission below the
  active band (§10).
- The window and collar defaults ([grids](../../../guide/setup/grids.md)) and the field's
  absorbing collar ([SEP 21](sep-0021-absorbing-boundaries.md)).
- General GPU performance optimisation. The implementation includes a CUDA build check
  and bounded HIP device qualification under the maintainer's revised decision below.

### 3. Context

- **Today's column** (`solprop.backends.numpy_ref.passes`). `plasma_scan` forms the
  exposure and the free current $J_{\rm free}=(e^2/m_e)\int fE\,dt$. `plasma_current`
  adds $J_{\rm loss}=I_pW(1-f)E/|E|^2$ and returns $P=\rho\int(J_{\rm free}+J_{\rm loss})\,dt$.
  `plasma_channels_polarisation` does the same for several channels, with the weights
  inside the shared integrals. All three integrals use `responses.plasma_integral`, the
  band rule by default or the trapezoid ([SEP 18](sep-0018-band-exact-plasma-integrals.md),
  [field plasma](../../../guide/media/field-plasma.md)).
- **The collar.** `nonlinear_polarisation` sums the plasma, Kerr and Raman polarisations
  and multiplies the total by `grids.t_window_fine`; the dispersive Kerr applies the
  collar inside, to its products. The total is projected, transformed and cropped, then
  multiplied by the prefactor $p\propto-i\omega/(2\varepsilon_0c\,{\rm Re}\,n_{\rm eff})$
  (crop scale included) and the spectral collar $W$. The fine collar is the grid's
  temporal `Collar`, with its zeros at the lattice ends and each taper at least 5% of the
  realised window. `Collar.window_derivative` (SEP 26) gives its analytic derivative.
- **Routes that admit the field plasma.**
  - *Modal and radial*, through `_over_nodes`: synthesis, column, time-domain projection
    and one transform per modal column.
  - *Cartesian*, through `_nonlinear_free`: one temporal transform per node, then the
    transverse FFT. *Line free space* is the same path with one $y$ sample, and is
    covered with it.
  - *Planar*, through `_nonlinear_slab`: $n_pn_y$ temporal transforms each way.
  - *General modal* refuses the gas and plasma keywords beside a provider. That is
    unchanged.
- **Compiled columns.** `numba_cpu` has a fused single-pass trapezoid column and a
  multipass band column; solprop-kokkos has a host team column, a tiled device column and
  trapezoid columns. Each forms the three running integrals and collars the total with
  Kerr.
- **`PlasmaResponse.evaluate`** returns the physical, ungated $P=I_R[J]$ and serves as the
  physics oracle of the column's currents.
- **Prior work.** SEP 18 §4.8 noted that the collar adds $w'P$ to $\partial_t(wP)$. SEP 26
  §4.7 defined the comparison in current variables. Its Phase 3 moved the question to the
  roadmap item that this SEP answers.

### 4. Theory

Transforms use the internal FFT convention $\tilde f(\omega)=\int f e^{-i\omega t}dt$
([conventions](../../conventions.md) §2), so $\mathcal F[f']=i\omega\mathcal F[f]$. Write
$q=i\omega$. $J$ is the total plasma current density, densities and channel weights
included. $t_a$ is the lattice start, where $J=0$; $t_\pm$ denote the tapers' times. $z$
is reserved for the propagation coordinate.

#### 4.1 What today's source contains, and what holding removes

The propagated source is proportional to $q\,\mathcal F[gP]$. Since $gP$ vanishes at both
lattice ends,

$$
q\,\mathcal F[gP]=\mathcal F[gJ]+\mathcal F[g'P],\qquad \int g'P\,dt=-\int gJ\,dt .
$$

The first term switches the current off with the collar. The second is a current pulse,
the switch-off of the polarisation. Neither is emitted by an electron. The held source is
SEP 26's form applied to $J$:

$$
\boxed{\;S_{\rm held}=\frac{\mathcal F[gJ']}{q}=\mathcal F[gJ]-\frac{\mathcal F[g'J]}{q}\;}
\qquad(\omega\neq0),
$$

with polarisation-equivalent $S_{\rm held}/q$. The term removed from today's source, in
current units, is $\mathcal F[g'P]+\mathcal F[g'J]/q$. $J$ is never differentiated, and
$g'$ is analytic (SEP 26 §4.2).

**The continuation.** Integrating by parts, with $J(t_a)=0$, the held current is

$$
J_{\rm h}(t)=\int_{t_a}^{t}gJ'\,ds=g(t)J(t)-\int_{t_a}^{t}g'J\,ds .
$$

- *General case.* On the plateau it is $J(t)-c_-$, with $c_-=\int_{\rm lead}g'J\,dt$.
  After the window it is held at $H=-\int_{\rm lead+trail}g'J\,dt$.
- *Empty leading taper* ($g'J=0$ there, the usual case): $J_{\rm h}=J$ on the plateau,
  and $H=-\int_{\rm trail}g'J\,dt$, which is $J_\infty$ when $J$ is steady through the
  trailing taper.

The leading taper can be occupied by pre-ionised electrons, or by ionisation in a field
that reaches it; admission does not constrain the field there. $\mathcal F[J_{\rm h}]=S_{\rm held}$
at every $\omega\neq0$, and $\int J_{\rm h}$ grows at slope $H$, the collisionless
plasma's own continuation.

Three cases:

- **A steady drift**: the hold removes both switch-offs. Holding $P$ instead gives
  $\mathcal F[gJ]$, which still switches the current off.
- **No drift but a residual polarisation** ($J\to0$, $\int J\,dt\neq0$): today's form
  radiates the polarisation's switch-off; the held form has no edge.
- **Variation inside the trailing taper** is kept as $gJ'$. Variation after the window is
  dropped: that is the window's finite-memory error.

**Which part dominates.** On the trailing taper, $|g'J|/|\omega g'P|\sim1/(\omega\,\Delta t_c)$,
with $\Delta t_c$ the delay from ionisation to the taper: about 0.02 in run_short's near
infrared. SEP 26's measurement agrees. The polarisation edge alone matches the full
route-minus-held figure to $7\times10^{-6}$ of the band source at 3.969 m.

#### 4.2 Which terms are affected

- **The Drude current** is held. After the pulse it is the Brunel drift
  $J_\infty=(e^2/m_e)\int fE\,dt$ (references 1, 2), generally nonzero.
- **The ionisation-loss current** $\propto W(|E|)$ is exactly zero below the rate table's
  `field_min`. It is held with $J$, contributes to $H$ only where the field reaches a
  taper, and its in-window variation is kept.
- **Kerr** $P_K=\varepsilon_0\rho\chi^{(3)}E^3$ is local. Its collar edge is bounded by
  $|g'|\max_{\rm taper}|E|^3$, of order $(E_{\rm taper}/E_{\rm peak})^3\approx10^{-9}$ for
  run_short's edge field (SEP 18 §4.4). A local response has nothing to hold, and where
  the field reaches the taper the collar is acting as the absorbing boundary. Kerr keeps
  $g$.
- **Raman** $P_R=\varepsilon_0\rho E\,(R*E^2)$ vanishes with $E$, and keeps $g$. The
  **dispersive Kerr** keeps its inner collar.
- **The envelope plasma** enters as the gated drive $gJ'$ times $-1/\omega^2$, the held
  structure. With `shock=True` this is exactly $\mathcal F[gJ']/q^2$. With `shock=False`
  its factors are frozen at $\omega_0$, the carrier approximation of the same structure.
  It is unchanged.
- **TDSE**: `admit_tdse` still refuses a plasma beside a TDSE response.

#### 4.3 Every current exactly once

- $J_{\rm free}$ and $J_{\rm loss}$ of every channel enter once, through one shared $J$.
- The held value adds no current: at $\omega\neq0$ a steady $H$ contributes nothing.
- The removed terms are the collar's switch-offs.
- Kerr and Raman enter once, collared as today. The linear index stays in the operator,
  and pre-ionised electrons stay in $J_{\rm free}$.
- No coexisting response contains a plasma current (the envelope plasma is envelope-grid
  only; TDSE is refused).

#### 4.4 The discrete held source

**Lattice**, as in SEP 26 §4.4: $t_n=(n-N/2)\Delta t$, the raw DFT $x_k=\sum_nx_ne^{-2\pi ikn/N}$,
and $\Delta t\,x_k=(-1)^k\sum_m\mathcal F[x](\omega_k+m\omega_s)$ for smooth $x$ supported
in one period. Let $X=\mathrm{DFT}[gJ]$ and $Z=\mathrm{DFT}[g'J]$, nodal and per
component, with $J$ the column's current from its two remaining running integrals.

**Definition.** On the active bins $A$, the plasma's polarisation spectrum is

$$
\boxed{\;A_J=\frac{X}{q}-\frac{Z}{q^2}\;}
$$

and zero elsewhere. The non-plasma part keeps today's $V_{\rm nl}=\mathrm{DFT}[g(P_K+P_R)]$,
plus the dispersive Kerr. The assembled stage source is

$$
Q=p\,W\,\mathrm{crop}\Big[\textstyle\sum_{\rm nodes}\Pi\,(V_{\rm nl}+A_J)\Big],
$$

with $\Pi$ the route's projection weights. Every bound and relative figure in §7 is
carried through $\Pi$, the crop, $p$ and $W$, and divided by a norm in these same units.

- **Accuracy.** On a resolved collar $A_J$ is the continuous held form, apart from the
  sampled products' alias and rounding. Its third integral is exact division, so the
  route's `integration` rule now governs only the exposure and free-current integrals.
- **Ordering is free.** Projection commutes with the temporal DFT. A route may project
  $g(P_K+P_R)$, $gJ$ and $g'J$ and transform the modal columns, or transform per node,
  combine with the weights $1$, $1/q$ and $-1/q^2$, and project in frequency. The cheaper
  order is chosen per route (§6).
- **A decomposition check, not a definition.** $A_P=V-Y/q-Z/q^2$, with $V=\mathrm{DFT}[gP]$
  and $Y=\mathrm{DFT}[g'P]$, subtracts the edge from today's sampled polarisation. It
  differs from $A_J$ by $C/q$, where $C=qV-X-Y$ is the running-integral closure residual
  that SEP 26 measured at $3$–$10\times10^{-7}$.

**Drafting probe.** The review's table, against the analytic held step (run_short
lattice, $J=\tfrac H2[1+\operatorname{erf}((t+5\,{\rm fs})/1\,{\rm fs})]$; band norm,
relative to the analytic polarisation):

| rule | today's gated | $A_P$ | in-column (§10) | $A_J$ |
|---|---|---|---|---|
| band | 2.56 | $7.0\times10^{-8}$ | $7.0\times10^{-8}$ | $5.3\times10^{-16}$ |
| trapezoid | 2.56 | $1.6\times10^{-4}$ | $1.6\times10^{-4}$ | $5.3\times10^{-16}$ |

#### 4.5 The collar certificate

The steady-current identity on the sampled collar is certified per configuration at
translation, as in SEP 26 §4.4, normalised by the full transform:

$$
\kappa_{\rm ss}=\frac{\big\lVert\mathrm{DFT}[g]-\mathrm{DFT}[g']/q\big\rVert_A}{\lVert\mathrm{DFT}[g]\rVert_2}\le10^{-10}.
$$

It is recorded in the run metadata. The same normalisation is now applied to the TDSE
certificate by a pending bug fix (Q3). Drafting probe:

| grid (band, requested window) | $N$ | taper samples | band-relative ratio | $\kappa_{\rm ss}$ |
|---|---|---|---|---|
| 90 nm–1.6 µm, 60 fs (run_short) | 2048 | 424 | $1.8\times10^{-14}$ | $7.9\times10^{-17}$ |
| 90 nm–1.6 µm, 100 fs | 2048 | 102 | $3.7\times10^{-14}$ | $2.0\times10^{-15}$ |
| 200 nm–2 µm, 40 fs | 512 | 76 | $7.7\times10^{-12}$ | $3.4\times10^{-13}$ |
| 400 nm–2 µm, 1 ps | 8192 | 1848 | $1.4\times10^{-7}$ | $5.9\times10^{-17}$ |
| synthetic tapers of ¼–40 samples, run_short lattice | 2048 | — | 1.0 … $1.2\times10^{-8}$ | $9.3\times10^{-3}$ … $7.9\times10^{-10}$ (refused) |

**What it certifies.** $\kappa_{\rm ss}$ qualifies that one collar identity: a steady
current is held, not switched off. It does not certify aliasing of arbitrary sampled
products $gJ$ or $g'J$. Those carry the column's existing sampling assumptions, as
$\mathrm{DFT}[gP]$ does today, and are budgeted per test (§7).

#### 4.6 Rounding budget

The budget is absolute and operand-based (SEP 25, SEP 26 §4.5). Errors are propagated
through the named operators per active bin:

1. **The current.** $\delta J$ is the column's existing bound on its two running integrals
   (SEP 25 §4.6), unchanged.
2. **The products.**
   - $\delta(gJ)\le|g|\,\delta J+\varepsilon(1+c_g)|gJ|$;
   - $\delta(g'J)\le|g'|\,\delta J+(19+3|\zeta|)\varepsilon|g'J|$ (SEP 26 Phase 1,
     deviation 2).
3. **Projection** with absolute weights adds $n_{\rm nodes}\varepsilon$ of the absolute
   projected operands.
4. **The FFT** gives $\sqrt N\lVert\delta x\rVert_2+c_F\varepsilon\log_2N\sqrt N\lVert x\rVert_2$
   per bin, with $c_F=8$ qualified on the supported smooth sizes as SEP 25 requires.
5. **Division.** $1/q$ and $1/q^2$ scale by $1/\omega_k$ and $1/\omega_k^2$, adding
   $2\varepsilon$ relative; one subtraction adds $\varepsilon(|X/q|+|Z/q^2|)$.
6. **Assembly.** Crop, $p$ and $W$ add a few $\varepsilon$ relative, and scale everything
   by $|p|W$.

This is SEP 26 §4.5's budget with a second division, applied to a current that carries
its own column bound.

#### 4.7 The work identity

For one real nodal column, the band-limited work is

$$
\mathcal W[S]=\frac{\Delta t}{N}\sum_{k\in A}w_k\,\mathrm{Re}\big(\overline{E_k}\,S_k\big),
\qquad w_0=w_{N/2}=1,\ w_k=2\ \text{otherwise}.
$$

By Parseval, $\Delta t\sum_na_nb_n=\frac{\Delta t}{N}\sum_{k=0}^{N/2}w_k\mathrm{Re}(\overline{a_k}b_k)$.
Take the held source in current units, $S=qA_J=X-Z/q$. The periodic antiderivative of
$g'J$ on the plateau is $c_--m-(n/N)\Sigma$, with $\Sigma=\Delta t\sum g'J=-H$ and $m$ its
mean. Then, exactly up to rounding,

$$
\mathcal W[qA_J]=\Delta t\sum_nE_nJ_n
\;+\;\underbrace{\Delta t\sum_nE_n(g_n-1)J_n}_{\text{field on a taper}}
\;-\;\underbrace{\Delta t\sum_nE_n\big(I[g'J]_n-c_-\big)}_{\text{field on the trailing taper}}
\;-\;H M_1\;-\;O_{\rm out},
$$

with these terms:

- $M_1=\Delta t\sum_n(n/N)E_n$, the field's first moment over the period;
- $O_{\rm out}=\frac{\Delta t}{N}\sum_{k\notin A}w_k\mathrm{Re}\big(\overline{E_k}\,\mathrm{DFT}[gJ-\mathcal J[g'J]]_k\big)$,
  the out-of-band part, where $\mathcal J$ is the periodic antiderivative that $1/q$
  represents (DC included, so the term $(m-c_-)\Delta t\sum E_n$ is folded in).

$HM_1$ is the window term of a held current represented on a periodic lattice; it falls
as $1/T$. For a band-limited field confined to the plateau it is small, because
$\sum_n nE_n$ follows $\partial_\omega\widehat E$ at zero frequency. The taper terms are
not small when the field overlaps a taper, and no bound from the field's out-of-band
fraction alone controls them: the review's counterexample is such a case. When the field
does not overlap the tapers, the identity is
$\mathcal W[qA_J]=\Delta t\sum EJ-HM_1-O_{\rm out}$. A drafting probe on run_short (a
plateau field and a stepped current) closed it to $3\times10^{-15}$ of the work; there
$HM_1$ was itself only $5\times10^{-16}$ of it. SEP 18's
population and loss-work check is separate and unchanged.

### 5. Proposed interface

- **No public option.** Every field-grid `PlasmaResponse` uses the held source.
  Signatures are unchanged. Translation refuses a configuration with $\kappa_{\rm ss}>10^{-10}$,
  naming the collar's taper samples and the remedy (a wider window).
- **`PlasmaResponse.evaluate`** keeps its meaning: the physical, ungated $P=I_R[J]$, the
  physics oracle of the column's currents. Its docstring says that propagation assembles
  the held source (§4.4) from the same current. The assembled-source oracle is a separate
  private function of the reference backend, used by the tests.
- **A private comparison hook.** `numpy_ref`'s pass takes a private switch that
  reproduces today's $\mathrm{DFT}[gP]$, for validation only.
  - A run using it carries a private benchmark label in its metadata and in its hashed
    configuration, so it can never share a resume key with a production run.
  - Its output refuses to resume, and is written only by the benchmark scripts, to their
    own directories.
  - It is not public API, and it is not implemented in the compiled backends.
- **Run metadata.** `metadata()["plasma_source"]` records $\kappa_{\rm ss}$, its limit and
  the collar's taper samples, computed once at translation.
- **No in-run edge diagnostic.** Edge fractions and $\rho_-=\max_{\rm lead}|J|/\max|J|$
  are an offline analysis of saved fields (`benchmarks/sep26/rate_plasma.py`, generalised).

```text
output = prop_capillary(..., plasma=True)   # unchanged call; the source is now held
```

### 6. Implementation strategies

**Recommended: $A_J$ in frequency.**

- *Column.* The column outputs the plasma's $gJ$ and $g'J$ beside $g(P_K+P_R)$, instead
  of a collared total, and drops its third running integral.
- *Transforms.* The route transforms the three arrays and combines them with the weights
  $1$, $1/q$ and $-1/q^2$, or projects first, whichever is cheaper.
- *Cost per stage evaluation*, for $n_c$ field components:

| route | today | with $A_J$ | per-node column |
|---|---|---|---|
| single-mode modal | 1 projection, $n_c$ modal transforms | 3 projections, $3n_c$ modal transforms (projections negligible) | one scan fewer |
| radial ($n_m=n_{\rm nodes}$) | 1 projection GEMM, $n_mn_c$ transforms | per-node transforms ($3n_{\rm nodes}n_c$), combined, then one frequency-domain projection on the cropped bins; projecting three arrays would triple the dominant GEMM and is not used | one scan fewer |
| Cartesian and line | $n_{\rm nodes}n_c$ per-node transforms | $3n_{\rm nodes}n_c$ per-node transforms, combined before the transverse FFT | one scan fewer |
| planar | $n_pn_yn_c$ row transforms | $3n_pn_yn_c$, combined before $k_y$ | one scan fewer |

  The net change is $+2$ real transforms and $-1$ filter-and-scan per node and component.
  Indications:
  - In `numpy_ref` on run_short's lattice, one band-rule running integral of 33 rows costs
    about four real FFTs of the same rows (440 against 104 µs), so the change is a net
    saving there.
  - In compiled code a scan costs about 10–15 flops per sample, and a real transform about
    $1.25\log_2N\approx14$ flops per sample at $N=2048$, so the estimate runs from roughly
    neutral (modal) to about $+20\%$ of a nonlinear evaluation (Cartesian, where the
    per-node transforms already weigh most beside the column).

  The option kept for later, B, adds a scan instead: an estimated 10–25% of the plasma
  column. Phase 1 measures both.
- *Memory.* Two more nodal rows per concurrent node, and two more modal or spectral
  arrays where a route projects first. Both are in `memory_ledger()`.

**Kept as a later optimisation (B, the first draft's in-column form).** The held
polarisation can be built in time with the route's running integral, closed periodically.
The review showed what it needs:

- periodic, not point-reflected, filters for the new integrals (R1);
- an additive edge residual $R(1-R)Z/q^2$ (R3);
- support recomputed after closure (R10);
- a longer rounding recurrence (R7).

It is admitted later only if it matches $A_J$ within a derived budget on every supported
grid. The single-extra-transform variant $V-\mathrm{DFT}[g''P+2g'J]/q^2$ is an
$A_P$-type form carrying $C/q$, and is likewise not primary.

**Layers.**

- *Physics:* the `PlasmaResponse` docstring.
- *Translation:* the fine collar derivative and the certificate.
- *Backends:* the column outputs and the route assembly.
- *Driver:* the metadata, the benchmark label and the resume refusal.

**Seam and saved output.**

- `grids` gains `t_window_fine_derivative`, and the column's output contract changes, so
  the kernel `SCHEMA_VERSION` takes its next free number.
- The output `SCHEMA_VERSION` also takes its next number. `RunMetadata.configuration_hash`
  digests it, so every file written before this SEP has a different resume key and its
  resume is refused. That is the concrete rejection boundary for old production files.
- A concurrent cleanup moves the output schema to 21 and the TDSE descriptors to v5. The
  implementer takes the next free numbers at implementation, and tests the refusal.

**Backends.**

- *Consistency.* Phases 1–3 are reviewed separately and integrated into `main` together,
  so the backends never disagree.
- *`numpy_ref`* (Phase 1): `plasma_current` and `plasma_channels_polarisation` return
  $gJ$ and $g'J$; each route assembles $A_J$.
- *`numba_cpu`* (Phase 2): both columns write the two rows and drop the third integral.
  The Raman finish collars only the non-plasma part.
- *solprop-kokkos* (Phase 3): the same in the host team, tiled device and trapezoid
  columns, plus the derivative table on the device and the route assembly. Device builds
  compile the same code; their qualification follows the standing policy.

**Interaction with the stepper.**

- Each stage evaluation is a pure function of the stage field, as today. No state crosses
  stages, rejections or restarts, and no evaluation is added.
- No allocation per stage.
- The error norm sees changed stage derivatives. Step counts are measured, not gated
  (§7.3).

**Dependencies.** None.

### 7. Verification and validation

Tolerances are derived before evaluation and stated beside each assertion.

- **Exact equality** holds only for masks, accounting, lossless copies and the structural
  zero.
- **"Gated"** means the private hook's output.
- **Matched comparisons** use §4.6 plus the named discretisation terms.
- **Relative figures** are in §4.4's assembled units, divided by a norm in the same units.

#### 7.1 Analytic and unit tests (default suite, fast)

Each runs on a given $J$ through the assembled-source oracle, then through the column
from a field.

- **T1 Steady current.** $J\equiv H$: per bin $|A_J|\le|H|\,|\mathrm{DFT}[g]-\mathrm{DFT}[g']/q|_k/\omega_k+$
  budget, with the band norm $\le\kappa_{\rm ss}\lVert\mathrm{DFT}[g]\rVert_2|H|/\omega_{\rm lo}+$
  budget. The gated source is reported as nonzero.
- **T2 Post-pulse drift.** The erf rise of §4.4, then steady. The oracle is
  $(-1)^kHe^{-\omega_k^2\sigma^2/4}e^{-i\omega_kt_c}/(\Delta t\,q_k^2)$. Each per-bin term
  is additive and absolute, so bins with negligible physical signal are covered:
  - the rise's mass outside the plateau, $|H|\operatorname{erfc}(d/\sigma)/(\Delta t\,\omega_k^2)$;
  - the steady tail's collar residual, $|H|\,|\mathrm{DFT}[g]-\mathrm{DFT}[g']/q|_k/\omega_k$;
  - the product aliases $|H|e^{-(\omega_N\sigma)^2/4}/(\Delta t\,\omega_k^2)$;
  - the budget.
  Both rules, broad and narrow rises, asymmetric placement. A band floor may be used for
  relative reporting only.
- **T3 Zero.** $J\equiv0$ gives exactly zero, held and gated.
- **T4 A known ionisation history (free current only).** The fraction is prescribed,
  $f=f_0+\tfrac{\Delta f}{2}[1+\operatorname{erf}((t-t_i)/\sigma_i)]$, with
  $E=-\partial_tA$ and $A=A_0e^{-t^2/\tau^2}\sin(\omega_0t+\varphi)$. Then
  $J_\infty=+(e^2/m_e)\Delta f\,(A*\phi_{\sigma_i})(t_i)$, which the review checked by
  quadrature. The loss current is set to zero, since the held source is linear in $J$;
  the loss current is covered by T1, T6 and the column runs.
  - (a) The removed term equals the analytic switch-off of $J_\infty$ and of the residual
    polarisation on the trailing taper, within the budget and T2's terms.
  - (b) At a zero of $A*\phi_{\sigma_i}$: $J_\infty=0$ but $\int J\,dt\neq0$. The held
    source has no edge, and the gated one has the polarisation's switch-off.
  - (c) Across $\varphi$, the removed term is the sum of the two parts, each linear in
    its own amplitude ($J_\infty$ and the residual polarisation).
- **T5 Kerr and Raman unaffected.**
  - Held minus gated is independent of $\chi^{(3)}$ and the Raman amplitude, within the
    budget.
  - Held and gated agree within the budget only when both $J$ and $\int J\,dt$ vanish
    before the trailing taper.
  - A compact current with $\int J\,dt\neq0$ shows the residual-polarisation edge removed.
- **T6 Decomposition check.** On real column currents (optmat He, a few-cycle ionising
  pulse; one, two and three components where admitted; one and two channels;
  pre-ionisation): $\lVert A_P-A_J-C/q\rVert\le$ budget. $\lVert C/q\rVert$ is reported.
- **T7 Certificate.** Collars of ¼–40 samples are refused; the §4.5 grids are admitted;
  the recorded $\kappa_{\rm ss}$ agrees with an independent recomputation within
  $10\varepsilon\log_2N$ absolute.
- **T8 Work** (§4.7).
  - Gated: with a field confined to the plateau,
    $|\mathcal W[qA_J]-\Delta t\sum EJ+HM_1+O_{\rm out}|\le$ budget.
  - Reported, not gated: one case with a field on the trailing taper, showing both taper
    terms.
- **T9 Leading taper.** With pre-ionisation and a field on the leading taper, the plateau
  offset $c_-$ and the final value $H=-\int_{\rm lead+trail}g'J\,dt$ are recovered from
  the held source, against §4.1, within the budget. $\rho_-$ is recorded.
- **Seam and output.**
  - The previous kernel schema's consumer refuses the problem.
  - A file at the previous output schema refuses to resume.
  - A benchmark-labelled gated output refuses to resume, and never shares a production
    resume key.
  - The memory ledger lists the new rows.
- **Hot-path benchmark** (benchmark lane, bounded). One nonlinear evaluation on
  run_short's lattice, a radial and a Cartesian case: $A_J$, gated and B's prototype, with
  warmup, interleaved repeats and a distribution. A ledger entry, with the hardware in the
  ledger.

#### 7.2 Fixed-field replay (offline, minutes)

`rate_plasma.py` gains the production source. It reads the same hashed L6 planes with the
same rule, nodes, density and crop.

- **(i)** The production $Q$ against SEP 26's $Q_{\rm held}$. That is the same $A_J$
  computation through different code, so the bound is the §4.6 budget in assembled units.
- **(ii)** The route-minus-held figures of §2, including $1.77\times10^{-5}$ at 90–200 nm,
  are reproduced within the same budget relative to $\lVert Q_{\rm route}\rVert_B$.
- **(iii)** The gated hook reproduces $Q_{\rm route}$ within the column budget.
- $\lVert C/q\rVert_B$ is reported per plane.

#### 7.3 Propagated before/after comparisons (CPU execution; no GPU)

**Full-length exception.** The run_short rate parent runs on `numpy_ref` in about a
core-minute. **Maintainer decision (John Travers, 2026-10-19):** the full-length
0–4.05 m campaign below is an explicit exception to SEP 0 §7 and `AGENTS.md`'s
qualification limit, because these runs are fast. Backend qualification and hot-path
timings stay bounded (§7.1, §7.4).

**Runs** (both forms; gated through a benchmark-script flag):

- **(a) The run_short rate case**, 0–4.05 m, 51 planes, on an `rtol` ladder from
  $10^{-2}$, tightened by factors of about 3 until the residual bound below is resolved.
- **(b) The SEP 20 L6 rate segment**, 3.969–4.05 m, launched from the same parent plane,
  and from a held parent's plane.
- **(c) A window control**: (a) at the converged level with a 120 fs requested window.

Every run records `atol`, the maximum step and the save-plane policy: fixed common planes,
identical for both forms.

**Convergence.** For each form, $d_j$ is the difference between successive levels on the
metric below. The tightest level's residual is bounded from the last three levels, by
$d_{\rm last}\,r/(1-r)$ with $r=d_{\rm last}/d_{\rm prev}$, if $r<1$ and the three are
consistent; otherwise tighten further. No order is assumed from the `rtol` ratios. A form
difference below the sum of the two forms' residual bounds is unresolved.

**Metric.** At common planes, $\lVert a-b\rVert_B/\lVert b\rVert_B$ on complex spectra
($b$ gated) over the full band, 90–200 nm, 0.9–1.2, 1.2–1.6 and 0.9–1.6 µm. Also
energy, peak power, 90–200 nm energy, steps accepted and rejected, evaluations and wall
time.

**Matched-input controls.** At every tenth saved plane, both sources are evaluated on the
same saved field (§7.2). This separates the direct change from the propagated one.

**Window control.** Both grids' spectra are converted to the continuous-transform
normalisation ($\Delta t$ included, the $(-1)^k$ origin removed, the interface amplitude
convention). Band energies are compared, with complex spectra only on coinciding bins.
The launch is rebuilt from the same pulse definition on each grid.

**Investigation threshold.** A 90–200 nm difference above 0.5% at the converged level is
investigated with the matched-input controls and the ladder, not failed.

**Hypotheses, not pass conditions.** No sign, size or attenuation is a pass condition.

- The 0.9–1.6 µm spectra change at the percent level at the late planes.
- The UV changes little, although the source record's UV difference is nonzero and
  propagation can amplify it.
- The held form's window sensitivity is lower than the gated form's.

#### 7.4 Backend parity

`numba_cpu` and Kokkos CPU are held to `numpy_ref`, with SEP 25's budgets and §4.6:

- one evaluation, one step, eight steps, the statistics and a ten-step run;
- both rules, every component count each route admits, one and two channels,
  pre-ionisation;
- plasma alone, with Kerr, with Raman, and with both, so that the changed collar
  placement is exercised;
- on the modal, radial, Cartesian, line and planar routes;
- the general modal refusal unchanged.

#### 7.5 Existing witnesses, benchmarks and records

Every field-grid plasma result changes. No test is weakened.

- **Luna plasma witnesses** compare against the gated hook, like with like, at their
  existing tolerances. The held form is reported against them as the deliberate
  difference.
- **Other stored plasma witnesses** are regenerated from the held form, with provenance.
- **The SEP 18 qualification** is replayed. Its third-integral comparison becomes exact
  division and is restated.
- **Benchmark cases** with a plasma are re-measured, with new ledger entries.
- **The TDSE campaigns' rate parents and baselines** remain historical evidence. Phase 4
  reports how the L6 gates move against a held rate baseline.
- **Galleries.** Tutorials re-run on every build. Worked examples with a field-grid plasma
  are re-run before Final.

### 8. Documentation

- **[Field plasma](../../../guide/media/field-plasma.md)**: the identity, the continuation,
  $A_J$, the certificate, the work identity, `evaluate`'s meaning, and a tested worked
  comparison against the truncation.
- **[Grids](../../../guide/setup/grids.md)**: what the collar does to a memory current.
- **[Differences from Luna](../../../guide/workflows/luna.md)**: the held form.
- **[Kernel contract](../../kernel-contract.md)**: the schema row and the column's outputs.
- **[Output](../../../guide/results/output.md)**: the output schema bump.
- Docstrings, the changelog, the roadmap and `benchmarks/sep27/`.

### 9. Compatibility and impact

- **No backward compatibility (maintainer's decision, 2026-10-19).** solprop is
  unreleased, so the held source simply replaces the truncation as the only behaviour,
  with no public option, no dual path and no resume of old files. The output and kernel
  schema bumps make old files refuse to resume (§6).
- **What changes.** Every field-grid plasma result; scripts are unchanged.
- **The maintainer's existing results** are evidence of the truncating form. §7.3's
  hypotheses are not predictions.
- **Luna.** The held form is a deliberate difference.

### 10. Alternatives considered

- **B, in the column** (§6): no extra transform, but boundary, additive-edge, support and
  rounding complications. Kept as a later optimisation, admitted only against $A_J$.
- **$A_P$, subtraction from today's polarisation**: carries $C/q$. Kept as a check.
- **A wider collar.** The in-band edge falls super-algebraically with the taper width. On
  a unit step current in 0.9–1.6 µm it is 1.36 at 21 fs, $7.7\times10^{-2}$ at 40 fs,
  $2.2\times10^{-3}$ at 80 fs and $1.2\times10^{-5}$ at 160 fs, for 4–8 times the
  lattice. It changes every grid and leaves the edge in principle.
- **Subtracting the steady current explicitly** needs an arbitrary turn-on. For a
  collisionless model it gives nothing the hold does not. Reconsider with collisions
  ([SEP 5](../sep-0005-solid-ionisation.md)).
- **Holding the polarisation** still switches the current off.
- **Leaving it, documented**: percent-level near-infrared source errors and a window
  dependence in the most-used routes.
- **A public option** is not needed for unreleased software.

### 11. Phases

1. **Phase 1: `numpy_ref` and the seam** (`feature/sep-0027-phase-1`).
   - The column outputs, the $A_J$ assembly on every admitted route, the certificate and
     metadata.
   - The private hook and its label, and both schema bumps.
   - T1–T9, the seam and output tests, the hot-path benchmark, the manual and the
     changelog.
2. **Phase 2: `numba_cpu`** (stacked). Both columns and every route; §7.4 parity; a
   ledger entry.
3. **Phase 3: Kokkos CPU** (solprop-kokkos branch of the same name). The columns and
   assembly; §7.4 parity; the Kokkos lanes; a CUDA/HIP compile-only check where available;
   a ledger entry. Phases 1–3 integrate into `main` together.
4. **Phase 4: validation**, which may start once Phase 1 is reviewed.
   - The benchmark-script flag for the gated runs.
   - §7.2, then §7.3 on CPU execution, under 20 core-hours, and §7.5.
5. **Phase 5: the final gate and Final.** The worked examples and the Luna page; the full
   gate, the slow, coverage and Kokkos CPU lanes.

**Blocking conditions.**

- *Acceptance:* responses to the reviews.
- *Integration of Phases 1–3:* T1–T9 and §7.4 pass on all three backends.
- *Final:* §7.2 passes, every §7.3 investigation is closed, every phase is reported, the
  lanes pass, and the manual is updated. A finding that the held form is wrong returns
  the SEP to Draft.

**Cost.** Phase 1 is about SEP 26 Phase 1's size. Phases 2 and 3 are each a column-output
change plus route assembly. Compute is minutes for §7.2 and under 20 CPU core-hours for
§7.3.

**Risks and integration responsibility.**

- SEP 24 Phase 3 and SEP 17 touch the same column files. The SEP's implementer integrates
  with any concurrent SEP at merge, rebasing onto whichever lands first and re-running the
  affected parity tests. SEP 17's continuum electrons, if they enter $J$, are held with
  the rest.
- Every field-grid plasma result changes at once.
- The per-node routes pay two extra transforms (§6). If Phase 1's benchmark shows this
  unacceptable, B is pursued under its admission condition.

### 12. Open questions

Resolved by the maintainer on 2026-10-19: Q1 integrate Phases 1–3 together; Q4 every
admitted field-grid route; Q5 re-run the field-grid worked examples before Final; Q6
originally required compile-only device coverage. On acceptance, the maintainer revised
Q6: verify the A40 CUDA build, then run one HIP qualification campaign on an AMD GPU,
with a USD 20 rental cap. CPU qualification and physics validation precede the rental.

- **Q2 Implementation — resolved 2026-10-19: $A_J$** (the maintainer confirmed the
  changed recommendation, and that the `integration` rule then governs only the
  exposure and free-current integrals, §4.4).
  Background: The maintainer accepted B, in the column. The review showed B's
  discrete identity is conditional on a boundary extension the admitted grids do not all
  satisfy (R1, $2.6\times10^{-9}$ on a 512-point grid), and that it needs an additive
  edge term, support recomputation and a longer rounding recurrence (R3, R7, R10). $A_J$
  is exact to $5\times10^{-16}$ under both rules (R2). Its cost is two extra real
  transforms per node, set against one scan saved: a net saving in `numpy_ref`, and an
  estimated range from neutral (modal) to about $+20\%$ of a nonlinear evaluation
  (Cartesian) in compiled code, to be measured in Phase 1. *Recommendation:* $A_J$ as the
  definition and implementation; B only as a later, budget-admitted optimisation.
- **Q3 Certificate normalisation.** Resolved: the full transform. The earlier
  recommendation of a roadmap item for the TDSE certificate is superseded by the bug fix
  itself (cleanup branch `refactor/remove-compat-paths`, commit `f6b76751`, pending
  merge).

### 13. References

1. M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec, "Light
   propagation in field-ionizing media: extreme nonlinear optics," *Phys. Rev. Lett.*
   **83**, 2930 (1999). doi:10.1103/PhysRevLett.83.2930
2. F. Brunel, "Harmonic generation due to plasma effects in a gas undergoing
   multiphoton ionization in the high-intensity limit," *J. Opt. Soc. Am. B* **7**,
   521–526 (1990). doi:10.1364/JOSAB.7.000521
3. D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash, "A tapering window for
   time-domain templates and simulated signals in the detection of gravitational waves
   from coalescing compact binaries," *Class. Quantum Grav.* **27**, 084020 (2010),
   eq. (7). doi:10.1088/0264-9381/27/8/084020
4. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM
   (2002), ch. 24. doi:10.1137/1.9780898718027
5. SEP 26 Phase 2's fixed-field rate-plasma record,
   `benchmarks/sep26/evidence/rate-plasma.json` in the solprop repository, written by
   `benchmarks/sep26/rate_plasma.py`, with input hashes from
   `benchmarks/sep20/evidence/manifest.json` (`phase4.l6_runs`).

## Part II — Implementation record

### Acceptance and execution agreement — 2026-10-19

John Travers accepted the revised direct-current formulation and authorised all five
phases. Part I is frozen with the revised Q6 device qualification above. SEP 28
implementation waits for this physics change and its measured baseline. Phases 1–3
remain an integration unit; subsequent upstream changes are integrated by ordinary
merges, without rewriting branch history.

### Approved T2 correction — 2026-10-19

John Travers approved replacing T2's steady-tail allowance by independent sampling
errors for its two products. The equation $A_J$ and the $10^{-10}$ translation
certificate are unchanged. Part I's original T2 bound is superseded by this correction.

**Counterexample.** On the admitted 90 nm–1.6 µm, 60 fs smooth grid ($N=1200$,
$\kappa_{\rm ss}=5.3291\times10^{-12}$), the erf rise at 3 fs with width 2 fs
exceeds the original allowance by a factor of 33 at one bin. Its largest absolute
source discrepancy is $1.6435\times10^{-24}$ in the raw polarisation-equivalent
units of a unit current. Sampling the same continuous products 8, 16 and 32 times
more finely reduces that discrepancy to $2.98$, $2.70$ and $2.70\times10^{-29}$.
The admitted 512-sample power-of-two grid also violates the old allowance. The
360-sample smooth 200 nm–2 µm grid fails admission ($1.312\times10^{-6}$) and is
now an explicit refusal case, not an analytic-source qualification case.

**Reason.** The two edges' discretisation errors can cancel in
$\operatorname{DFT}[g]-\operatorname{DFT}[g']/q$. A current occupying only the
trailing edge has no corresponding leading-edge error to cancel. The global
steady-current certificate therefore cannot bound that per-bin product error.

**Replacement.** Let $X_c$ and $Z_c$ be the continuous transforms of $gJ$ and
$g'J$, in the same raw-DFT scaling and phase origin as $X$ and $Z$. Compute them
independently by adaptive quadrature, splitting at the analytic collar boundaries
and the rise. With absolute quadrature errors $e_X,e_Z$, use

$$
\frac{|X-X_c|+e_X}{|\omega|}
+\frac{|Z-Z_c|+e_Z}{\omega^2}
+\text{analytic rise-tail allowance}+\text{rounding budget}.
$$

The two sampling terms cannot cancel. They are computed before the production
assembly, without inspecting its result; weak physical bins get the same absolute
error accounting. `tests/backends/test_held_plasma.py` implements the independent
quadrature and retains the original erf oracle and operand-based rounding terms.
The quadrature reports a numerical error estimate, not an interval-arithmetic proof.
The initial 36 analytic/certificate/identity checks pass; these are not the complete
phase or propagated-physics qualification.


### Phase 1 implementation checkpoint — 2026-10-19

**Delivered.** The NumPy source separates collared non-plasma polarisation from
current, projects all three streams on modal and planar routes, and combines
nodal spectra before the single radial projection or transverse FFT. Both running
integral rules stop at current in production. The physical `evaluate` oracle and
its old pass helpers retain ungated polarisation for independent physical tests.
This is a deliberate implementation detail differing from §6's proposed helper
renaming: `nonlinear_sources` owns the new production result instead.

Kernel schema 33 carries the analytic collar derivative; output schema 22 changes
the equation identity. The certificate uses the full fine transform, including DC,
and caches only scalar evidence with a weak reference to its immutable grid.
Its measured residual is execution evidence, outside the configuration hash;
source form, limit and taper counts are hashed. The private gated context is
NumPy-only, labels its configuration and execution, and refuses continuation.
The unported Numba backend refuses field plasma by name in this checkpoint.
Kokkos's old schema is incompatible. Phases 1–3 remain an integration unit.

**Review findings addressed.** The one-channel convenience property is absent on
mixtures, so admission, source allocation, assembly and metadata now test the
channel tuple. A full mixture-source regression checks this independently of the
Kerr-separation test. Existing spatial test fixtures with a 150 fs request had only
30–31 taper samples and residual 9.763e-9; their window is now 200 fs, and the old
fixture remains an explicit admission-refusal test. The threshold is unchanged.

**Verification so far.** Short checks cover analytic drift and zero currents,
independent product-sampling bounds, prescribed ionisation and residual
polarisation, work and decomposition identities, occupied leading tapers, every
NumPy geometry and both rules, scratch reuse, Kerr/Raman separation, one/two
channels and components, and pre-ionisation. The B comparison prototype is checked
against its periodic multiplier including the additive edge residual. Focused
source type-checking reports zero errors; lint and formatting pass. The complete
focused slice, full-project type check and exclusive timing comparison are queued
on the CPU qualification machine; this checkpoint is not a completed phase report.

**Fixed-field replay.** `benchmarks/sep27/rate_replay.py` verifies the three original
L6 input hashes, then calls production assembly and the private gated hook. Both
saved planes reproduce SEP 26's offline figures. At 3.969 m, the full-band norm
ratio is 0.03373296046656 and the 90–200 nm ratio is 1.770797114535e-5. At 4.05 m,
they are 0.07361588320044 and 1.447237841658e-5. These are matched-input source
ratios, not propagated output differences or convergence results. The unchanged
SEP 26 script and historical TDSE evidence remain intact.

**Sources and limitations.** The identities were re-derived from the accepted SEP
and tested against analytic Gaussian/erf currents, adaptive quadrature and the
original fixed-field calculation. No new physical constant or material coefficient
was transcribed; the existing optmat rate tables supply the real-column tests.
The cited papers were not independently re-read for this checkpoint. Quadrature
error estimates are numerical estimates, not interval proofs. Propagated tolerance
ladders, full backend ladders, performance conclusions, examples and the final gates
remain outstanding. No rental has been provisioned or charged.

### Phase 2 implementation checkpoint — 2026-10-19

**Delivered.** Both compiled columns now stop at the physical current. The band
column uses five histories instead of six and omits its outer filter and scan.
The fused trapezoid column writes every current sample separately from the
collared Kerr/Raman polarisation. `HeldWorkspace` owns both products, projected
streams, FFTW plans, embedded quadrature streams and frequency assembly. Modal
and planar routes project before their temporal transforms; radial routes
transform, combine and crop per node before one frequency projection; Cartesian
and line routes assemble before the transverse transform. All persistent arrays
appear in `memory_ledger()`, and node admission charges mixed-response histories
together. No production equation or public call signature changed in this phase.
The old rejected scan/tile prototypes retain their physical-polarisation oracle
in test support, independently of the changed production column.

**Approved rate-bound correction.** John Travers approved replacing the old fixed
16-epsilon rate allowance after the first-sample current exposed its missing
conditioning term. A stress input described as below threshold actually has a
tiny nonzero rate. Its first loss current differed by approximately 7e-14
relative between the array and compiled evaluations on the CPU qualification
machine; its historical integrated polarisation was exactly zero at that sample.
The input and production rate implementation are retained unchanged.

The revised allowance is derived from the operands before evaluating the column.
For the real field norm, gamma(8) bounds relative rounding. Assuming one-epsilon
`log` and `exp`, the logarithm's absolute error is at most
`gamma(8) + eps*abs(log(E))`. Forming the local coordinate adds gamma(3) times
the sum of its logarithm, lower knot and interval-offset magnitudes. With
`r = abs(local) + local_error`, the cubic's error is bounded by its derivative
magnitude at `r` times `local_error`, plus gamma(6) times its absolute Horner
polynomial. `expm1` maps this to relative rate error. If interval selection can
straddle a knot, both stored polynomials and their rounding bounds are retained;
no exact continuity of their rounded coefficients is assumed. The stress set
stays away from the table's discontinuous lower clip. These are explicit libm
accuracy assumptions, not vendor proofs. The current-chain test carries this
rate allowance through exposure, population, free current, loss and density.

**Verification so far.** The first focused compiled slice passed 265 tests. The
expanded slice passed 551 tests, including every admitted geometry under both
rules, complete masked rows, partial chunks, scratch reuse, ledger agreement,
NumPy analytic tests, spatial ladders and the historical prototypes. Its four
remaining failures were the rate-bound cases above; their corrected rerun and
broader propagation/statistics/low-storage ladders are pending. Source lint and
focused type checks pass. The two remaining under-resolved spatial fixtures and
the synthetic mixture fixture now use resolved windows; their former settings
remain covered by certificate refusal. The admission limit remains 1e-10.

**Limitations and handoff.** This is an implementation checkpoint, not completed
backend qualification. Exclusive alternating source timings, the extended ladders,
Kokkos implementation, full physics campaign and final gates remain outstanding.
The unchanged baseline has been captured before candidate measurements. No GPU
rental has started. Phase 3 receives the current-stream contract at kernel schema
33 and retains the established envelope-plasma path.

### Phase 4 preparation — 2026-10-19

The production-assembly replay has passed on the CPU qualification machine;
`benchmarks/sep27/evidence/rate-replay.json` retains input hashes, source
certificate, both saved-plane comparisons and sanitized source provenance.
It independently reproduces the earlier offline ratios within the operand-based
rounding budgets. This is fixed-field evidence, not a propagation claim.

`physics.py` launches fresh labeled runs, saves common-plane continuous-normalized
spectra, energy, peak power, UV energy, work counts and matched-input source
comparisons. A short run exercises both source forms. `campaign.py` holds an
exclusive campaign lock, charges actual child CPU time including failed workers,
and applies a hard CPU limit before each child starts. An interrupted unaccounted
worker prevents further launches. The shared allowance is 72,000 CPU seconds;
exhaustion is recorded as unresolved. No full-length physics run had started at
this checkpoint. `convergence.py` retains three-level differences and their
observed contraction; noncontracting sequences remain explicitly unresolved.
The accounting and residual-formula checks pass without launching a full run.

### Historical witnesses and phase-2 follow-up — 2026-10-19

The historical Luna plasma comparisons now construct and run through the private
NumPy gated hook, keeping their existing spectral bounds and stored arrays.
Non-plasma comparisons and production backend parity retain their original
routes. The SEP 12/20 parent reconstructions explicitly label the historical
equation; their private identity adapter refuses a held-current output. This
adapter does not permit any output to resume under another schema or equation.

A frozen Luna witness supplies its own historical collar. Its derivative does
not satisfy the held-current certificate (residual 6.703e-5), so the private gated
hook records only its equation and non-resumable identity. It makes no claim to
the held-current certificate. Production admission remains unchanged at 1e-10,
and tests assert refusal both before and after leaving the private context.

The corrected rate bound passed its original stress cases. The expanded Numba
slice passed 236 tests and found one further under-resolved low-storage fixture;
its field-plasma window is now 200 fs. The follow-up collar, low-storage and Luna
slice passed all 60 tests. The combined documentation, historical reconstruction,
benchmark-controller and kernel slice passed 626 tests with one expected process-SEP
skip. A full-source type check found one NumPy scalar needing an explicit
conversion to `float` in a test; the focused corrected type check passed. These
are slice checks, not the final full gate, slow lane or coverage qualification.

### Phase 3 implementation checkpoint — 2026-10-19

**Delivered.** Kokkos kernel schema 33 carries the analytic collar derivative and
rejects incompatible source contracts before allocation. The host band column,
tiled device column and trapezoid column write physical current separately from
the collared non-plasma response. Persistent real streams and transform plans
form the two current products. Modal routes project before transforming; planar
routes contract each complete transverse row first; Cartesian and line routes
combine before the transverse transform. Radial routes transform the three nodal
streams, combine and crop, then use one real matrix product on interleaved complex
values. Every inactive bin and every padded final-chunk row is written explicitly.

**Review and accounting.** The obsolete preloaded-polarisation template path was
removed: no compiled production path retains the outer plasma integral. Native
and Python ledgers agree for all new buffers, including embedded projections,
radial packing and the native owning copy of the derivative. Translation now
charges the tiled component histories and reserves complete planar rows. Charging
planar projected streams per physical node would incorrectly reject an otherwise
admissible default-budget problem; they are charged per projected modal column.
Regression tests cover both the minimum-row refusal and rounded chunk admission.

The approved rate bound also revealed that rounding a clipped zero upward to the
smallest subnormal invented loss current at zero field. The bound now preserves
the rate table's exact lower clip; production rate evaluation is unchanged.

**Verification so far.** The native debug suite passes with an independent
eight-sample pre-ionised-current/DFT oracle, complete-row reuse and malformed
derivative/active-frequency refusal cases. Strict native-adapter typing reports
zero errors. The source/ledger slice passed 116 tests; the native Python response
slice passed 107 tests. A further 82-test source, planar and low-storage slice
passed 76 and refused six old 150 fs planar fixtures at the unchanged certificate
limit. Those fixtures now request 200 fs and their rerun is pending. The final
native gate, complete CPU lanes, timing campaign and CUDA/HIP qualification remain
outstanding. No device support claim is inferred from the host build.

**Sources and handoff.** The native oracle re-derives the sampled current from a
cosine field and discrete trapezoid, then evaluates both DFT terms independently.
No physical constants or literature coefficients were transcribed. Phases 1–3
remain one integration unit. The unchanged baseline and candidate use identical
pinned numerical dependencies for exclusive timing. SEP 28 remains deferred.

### Phase 4 physics results and qualification follow-up — 2026-10-19

The full-length campaign completed 120 fresh workers, charging 2,510.49 CPU
seconds (0.6974 core-hours) including controller and worker overhead, well within
the 20 core-hour allowance. The immutable run records, actual work counts,
matched-input comparisons, input/output hashes and complete solver source
manifest are collected in
`physics.json`.
The collection verifies every record against its ledger hash. Its own
postprocessing used a further 2.91 CPU seconds. No historical TDSE output was
rewritten.

All five complex-spectrum bands have bounded last-three-level residuals on
every saved plane, for both parents and both L6 input families. The 60 fs and
120 fs parents finish at `rtol=7.7435e-11`; L6 finishes at `rtol=5.6450e-8`.
At 4.05 m the held-minus-gated UV energy change is -0.1596090% (60 fs),
-0.0672139% (120 fs), -0.0008128% (historical L6 parent), and -0.0008209%
(held L6 parent). Their summed relative residual estimates are respectively
1.91e-10, 3.50e-10, 6.91e-8 and 7.79e-8. These bounds estimate truncation
residuals through the observed geometric contraction; they are not rigorous
bounds on every physical or sampling error.

The endpoint full-band complex-spectrum changes are 0.23565% and 0.01852%
for the two parent windows. On the same 60 fs exit field, the matched-input
plasma source changes by about 7.31% over the full band but only 4.47e-6
relative in the UV. Source differences and propagated differences are distinct
measurements, and neither an expected sign nor an expected magnitude was used
to accept a result.

**Window investigation.** The 120 fs control changes the native UV band-energy
sum by -1.10997% for the gated source and -1.01846% for the held source.
An exact decomposition retains both sums: approximately -0.8843 and -0.8854
percentage points respectively come from the changed frequency quadrature
nodes and edge cells. The changes evaluated on the common coarse lattice are
-0.22566% and -0.13309%. This explains most of the above-0.5% discrepancy
without changing the energy definition, interpolating spectra or applying a
fitted correction. The remaining common-bin change is a genuine window
sensitivity; the finite-window model is not claimed to be window independent.

**Precision review.** At a few intermediate parent planes, scalar differences
at roughly 1e-12 relative no longer contract independently. Every such raw
sequence remains marked unbounded in the report; endpoint observables and
complex-spectrum ladders are resolved. John Travers questioned the unnecessary
precision at this point. Further tightening was stopped because it adds no
useful discrimination to the reported physics changes. A proposed conversion
of spectral residuals into scalar Cauchy–Schwarz bounds was not adopted;
this record does not silently replace those scalar sequences or claim that
all scalar residuals are bounded.

**Backend follow-up.** The corrected planar fixtures pass the complete 82-test
source, planar and low-storage slice. The native Python response slice passes
107 tests, and forcing four band tiles on the CPU passes the 17 current-source
and ledger tests. The interpreted-kernel coverage pass passes 522 tests.
The full default, compiled coverage and slow lanes are still in progress and
have exposed further failures. Native static analysis found nine redundant
host view copies in the new projection path; these were replaced by aliases,
with the device lambdas retaining value capture. Its affected unit and the
remaining native gate are being rerun. CUDA/HIP verification and the exclusive
compiled timing campaign remain pending; the SEP is not Final.

### Qualification review — 2026-10-19

The full default lane exposed old schema expectations, historical witness fixtures
that had not entered the private gated context, and plasma fixtures whose temporal
collars fail the new certificate. These fixtures now use admitted collars; the
certificate limit and the numerical parity bounds are unchanged. The memory
admission helper preserves the original transform when its chunk is unchanged,
restoring mixture-update sharing. Explicit tests cover historical memory accounting,
the driver resume refusal, and partial-chunk accumulation of embedded current streams.

The embedded-current regression retains an ionising analytic-rate input whose
trailing current is small after cancellation. Its first assertion incorrectly
scaled the column error by that tail. Applying the already accepted §4.6 recurrence
instead carries the unchanged full-current allowance through the collar products
and absolute projection weights. The test also observes the current before those
operations and checks its allowance independently. The diagnostic measured current
errors below 0.004 of that allowance and projected errors below 0.00014 of the
propagated allowance; no production equation or tolerance coefficient changed.

The existing circular-plasma loss-work check already meets its independently derived
analytic bound at each of its three temporal resolutions. Removing the third running
integral makes its old ratio of absolute work errors unsuitable for establishing the
trapezoid order: the absolute error also includes fixed-band projection. On review,
the maintainer approved removing only this redundant order assertion. The analytic
work checks, their three grids and bounds, SEP 18 population checks, and independent
running-integral convergence tests remain unchanged. No replacement precision ladder
or relaxed work tolerance is introduced.

The two failed slow-lane records are solprop's own M2 order and quadrature-alarm
studies, not comparisons to Luna's stored fields. Their physics assertions pass;
their old numerical records describe the gated equation. They have been regenerated
with an explicit held-current identity and source provenance; all four study tests
pass. The harness now selects its recorded backend explicitly, including when an
optional backend is installed. The accuracy page quotes the refreshed records. Historical Luna and
TDSE witnesses retain their existing data and bounds.

### Concurrent integration — 2026-10-19

Merged SEP 23 phase 2 through `d59ed0bf` by an ordinary topic-branch merge.
Its Numba far-field statistics leave the held-current source equation and transform
ordering unchanged. The shared memory estimate retains both the backend-specific
held-current buffers and the far-field partials, and the interpreted coverage lane
includes both new kernel test modules. Kernel schema 33 and output schema 22 remain
the integrated versions; SEP 23 phase 2 does not change either contract. The merged
SEP 28 documentation records future work only; SEP 28 implementation stays deferred.
The final CPU and device checks qualify this combined source tree.

### CPU qualification and CUDA compilation review — 2026-10-19

The integrated source pair has completed its CPU qualification. The
qualification record
retains commands, source revisions, results and raw-log hashes. The full gate's
default lane passed 16,251 tests with 27 expected skips. Three documentation links
failed the final build; their focused repair and strict rebuild pass. The coverage
lane passed 548 interpreted-kernel tests and 16,251 compiled tests, reaching 100%
of 42,760 statements and 12,942 branches. The Kokkos-selected slow lane passed all
148 tests. The Numba slow lane's two historical-record failures were repaired and
their four study cases pass; its other 145 tests had passed, with one expected skip.
The separate statistics-cost test passes with both Numba and Kokkos.

**Phase 1 review.** The analytic and physical tests, source decomposition, private
comparison identity, schema refusals and all admitted NumPy routes pass in the
complete suite. The approved T2 correction and separate physical-polarisation
oracle remain as recorded above. Historical Luna and TDSE witnesses are preserved.
The primer and API descriptions now distinguish the two production current
integrals from the physical polarisation's third integral. No new physical
constant or material coefficient was introduced.

**Phase 2 review.** The complete compiled and interpreted suites pass, including
the current-column bound, embedded partial-chunk projection, masks, zero reuse,
memory accounting, statistics and existing stepper paths. The approved logarithmic
rate bound remains explicit about its libm assumptions. A timing-harness error
called the Adams source seam on a Runge–Kutta-only driver; the harness now constructs
and warms the declared source driver separately from the ten-step driver, and
records both memory ledgers. This repair changes no production code or equation.

**Phase 3 review.** The native complete gate passes Python formatting, lint, strict
typing, licence checks, C++ formatting and analysis, debug native tests, ASan/UBSan,
327 Python tests with six expected skips, and the distribution-wheel smoke test.
The complete solprop CPU suite also passes with this backend selected. The
CUDA compile record
records the successful CUDA 12.8.93/GCC 11.5 build for `sm_86` with four jobs.
Compilation took 1,436 seconds; no CUDA device was allocated or executed. One
pre-existing unused-function diagnostic remains visible in its evidence.

The final incoming SEP 23 phase-2 merge adds its completion report; its implementation
was already present in this qualified source pair. The integrated schemas remain
33 and 22. The exclusive performance campaign, regenerated worked examples and
the approved HIP execution campaign remain outstanding at this checkpoint. The
SEP remains In progress and SEP 28 implementation remains deferred.

### HIP verification and incoming radial admission — 2026-10-19

The HIP execution record
qualifies the schema-33 implementation on an AMD Instinct MI325X (`gfx942`), with
HIP 7.14, AMD Clang 23, Kokkos 5.2.1, Kokkos-FFT 2.0.0, rocFFT and rocBLAS.
The native test executable passed on the device. Solprop's GPU lane passed 18
tests, the held-current backend ladders passed 34, and solprop-kokkos's GPU lane
passed 31. The latter includes the tiled band column, trapezoid diagnostics,
repeatability and partial-chunk Raman cases. No numerical allowance was changed
for this device.

Twelve fresh device processes cover mode averaging, multiple modes, radial,
Cartesian, line and planar sources under both integration rules. Each records ten
warmed source evaluations and three warmed ten-step windows, plus separate source
and Runge–Kutta memory ledgers. Their source medians range from 0.227 to 5.351 ms;
ten-step medians range from 9.016 to 200.226 ms. The complete resident inventories,
including host and device storage, are 13.078–104.728 MiB for the Runge–Kutta driver
and 22.382–230.138 MiB for the source driver. These are bounded execution checks,
not a GPU baseline comparison or a cross-machine speedup claim.

The first environment setup selected Python 3.15, which the pinned Numba package
refused. The device runner now selects Python 3.14 by default; the campaign used
3.14.7 with the unchanged lockfile. Both build attempts are retained. Upstream
compiler deprecation and character-conversion diagnostics remain visible in the
record; they did not prevent the successful build or tests.

During qualification, SEP 22 phase 3 became ready and was merged through `ff862f2f`.
Its only conflict was adjacent imports: the merged code retains both the held-current
certificate and the radial exponent bound. The latter is an admission check and
does not change the admitted propagator or current assembly. All 149 affected
operator, radial and held-current checks passed on the same HIP device after this
merge. The CPU closing lanes are rerun on the integrated pair. SEP 23 phase 3 was
still being edited in its own worktrees at this checkpoint and was not imported.

One rental was created. Evidence retrieval verified the archive and individual
file hashes before deletion; a subsequent API inventory confirmed that the recorded
resource was absent. Creation-to-deletion time was 932 seconds at USD 3.80/hour,
an estimated USD 0.9838 against the approved USD 20 cap. The deletion watchdog was
cancelled after confirmation. No campaign rental remains.

### Incoming far-field statistics integration — 2026-10-19

The completed SEP 23 phase-3 branches became available after the HIP rental was
closed. Ordinary merges `bf1d9b03` (solprop) and `a48e6cab` (solprop-kokkos)
retain both implementations; only adjacent changelog additions conflicted.
The current-source kernels, stream layouts and transform ordering are unchanged.
The shared memory model now also counts the native far-field partials, and the
registry checks the installed plugin's declared statistics. The integrated
schemas remain kernel 33 and output 22.

The final CPU lanes and regenerated worked examples use this source pair. The
ongoing exclusive performance sweep retains its frozen pre-merge pair so that
its alternating processes stay comparable. The merge adds statistics allocation
and admission work, outside those timed source and step calls; the final memory
inventory will distinguish that additional storage. The CUDA compile-only check
is repeated for the combined native tree with the same four-job limit. The
completed HIP evidence remains tied to its recorded revisions: it qualifies the
held-current paths, but not the newly added SEP 23 device statistics. No second
rental is started.

### Integrated CUDA build follow-up — 2026-10-19

The combined native tree passes the CUDA compile-only check. GCC 11.5 initially
rejected two spectral-row count sums in the newly merged far-field pass under
`-Werror=sign-conversion`. Commit `a8ba4bb` makes their 0/1 operands explicitly
unsigned; formulas, loop extents and tolerances are unchanged, and no warning is
suppressed. The repaired incremental build completed in 68 seconds with four
jobs, linking the native library, Python extension, test executable and GEMM
benchmark. The earlier failed attempt took 364 seconds and remains in the
CUDA evidence.
The original full build and its unchanged dependency objects are retained.
No CUDA device was allocated or executed.

### Incoming low-storage flow accounting — 2026-10-19

Merged SEP 24's committed closing work through `a22996f8`, including the
`e7826c83` reusable flow scratch and `bb9034c9` z-table tests. The Numba modal
low-storage march now forms its two step flows successively in one counted
scratch volume, without a transient exponent array. The merge retained the
held-current memory rows and the existing low-storage plasma and Raman tests.
The new exact memory assertion accounts separately for the reference and
compiled current layouts, using the constructed backend's current ledger rows;
their layouts need not occupy the same number of bytes.

Focused low-storage tests and the full integrated closing lanes remain due.
The exclusive source and ten-step performance campaign retains its frozen
source pair: its single-source Adams and Runge–Kutta paths do not execute this
low-storage flow code. Final memory records and correctness checks use the
integrated tree. No source equation or numerical tolerance changed in this merge.

### Performance measurement scope — 2026-10-19

The source-cost campaign measures a changed formulation on existing routes.
Each workload has distinct legacy and held identities; the comparison is the
cost of replacing the former production equation, not a same-equation regression.
A separately configured Adams source evaluation and a ten-step Runge–Kutta
window distinguish source cost from other backend work. Full adaptive work counts
remain in the physics record.

The initial sweep expanded to 1,464 planned fresh processes and was interrupted
when the maintainer questioned the elapsed time. Complete comparison groups are
retained unchanged. Incomplete groups are preserved separately with hashes and
rerun in full using five source samples, one warmup Runge–Kutta step and one
measured ten-step window per process. Every remaining group retains three fresh
rounds, alternating order, the unchanged baseline and two held runs for A/A
spread. The report refuses mixed protocols within a group. Completed NumPy
comparisons retain the gated and alternative-B prototypes; remaining groups
concentrate on the production comparison. This reduces repeated work without
changing the equations, propagation windows or the preselected slowdown rule.

The final performance review must report absolute Numba and Kokkos times as well
as each backend's ratio to its own old source. An improvement relative to one
backend's baseline does not imply that it is the fastest backend. The completed
original modal/radial/Cartesian matrix already demonstrates this distinction:
Kokkos's one-thread radial trapezoid window drops from 4,867 to 2,077 ms, while
Numba's rises from 504 to 744 ms. These are short-window measurements, not
predictions of full propagation time; profiling and the remaining matrix are
still due.

The first cross-backend report incorrectly called the requested-18-thread runs
an equal-thread comparison. Execution metadata showed that Kokkos reached 18
threads but Numba reached two. A lightweight runtime probe confirmed the cause:
with `OMP_PLACES=cores`, OpenMP binds the main thread to a core's two hardware
threads, while the existing allocation helper only recognises a one-CPU binding.
Its inferred allocation therefore shrinks to two CPUs. Explicitly listing one
logical CPU per physical core restores the requested cap without changing the
backend sources. The erroneous scaling interpretation was withdrawn. All
18-thread groups are repeated with matched explicit placement and assertions
of actual pool sizes; superseded records are preserved separately with hashes.
One-thread results remain valid. The general affinity-helper correction is a
separate follow-up, not a change to the held-current implementation.

### Closing integration and parity-fixture review — 2026-10-19

Subsequent incoming changes are taken through `main`. Its revisions through
`4ef7bf09` are merged, including SEP 22's archive and SEP 28's acceptance;
SEP 28's implementation still waits for this proposal. The earlier direct
topic-branch integration of SEP 24 through `a22996f8` remains in this tree and
is identified separately from the changes already on `main`. The native
`main` merge through `95b7502` changes ancestry without changing the already
integrated executable code.

The expanded one/eight/ten-step, two-rule, two-pre-ionisation parity matrix
initially built its constant pre-ionisation table only to the last save plane.
The translated tables include an end margin, so validation correctly refused it.
The fixture now uses the common `linear.z_span`. All 85 non-planar NumPy/Numba
cases pass after that correction.

The planar fixture independently exposed an invalid coarse replay: its original
0.1 m problem explicitly requires 80 Adams steps, while the new matrix requested
one, eight or ten. The pre-ionised reference overflowed before comparison.
The maintainer approved a $10^{-7}$ m window for this new planar matrix, retaining
the grid, launch, responses, both pre-ionisation values, both integration rules,
the requested step counts and all tolerances. The length was selected before
testing its result. The fully ionised cold-plasma rate
$\omega_p^2/(2c\,\operatorname{Re}(n_{\rm eff})\,\omega)$, evaluated from the
translated prefactor with its temporal crop scale removed, is at most
$2.76495\times10^5\ {\rm m}^{-1}$ over its active modes and frequencies.
The proposed whole-window phase is therefore below 0.028 rad. This is a
short-window design scale, not a proof of nonlinear stability or a substitute
for the independent planar accuracy and convergence studies, whose lengths
remain unchanged. All 12 corrected planar cases pass on NumPy/Numba (17.01 s),
completing that matrix with the 85 non-planar cases (30.79 s) on macOS,
Python 3.14.5. The integrated Kokkos matrix remains part of the closing CPU
qualification. Lint, formatting, typing, public docstrings, spelling and licence
checks pass on this source tree; SEP lifecycle tests pass (211, one expected skip).

The additional macOS build of the integrated native backend passes all 113 focused
Kokkos checks in 70.07 s: the 96 one/eight/ten-step comparisons and 17 source,
memory, scratch-reuse and benchmark-refusal checks. It uses Python 3.14.5,
Clang 23.1.3, Kokkos 5.2.1, OpenMP, FFTW and Accelerate in the local Release
profile, with one numerical thread per pytest worker. These are correctness
checks on a concurrently used machine, not performance measurements.

### Final test-budget correction — 2026-10-19

The existing 0.1 m, 80-step uniform-planar field-plasma fixture passed its
unchanged field, transverse-leakage and density assertions, but its peak rate
differed by $1.28\times10^{-7}$ relative, above the old empirical $10^{-7}$
allowance. The maintainer approved propagating the existing spectral field
allowance through the actual synthesis and rate table. This changes only the
field-plasma total and species peak-rate assertions, retaining the original
problem and all other checks.

For the permitted coefficient error $B$, including the existing nonzero-$k_y$
leakage allowance summed through the inverse transverse transform, the real
inverse temporal transform and modal
synthesis give $\delta F = B(2N_{\rm active}/N_{\rm to})\max_n\sum_m|S_{nm}|$.
Both reconstructions' temporal and transverse FFT and synthesis rounding are
added against their absolute operands. The maximum absolute nodal field is
Lipschitz in this error. Every log-cubic interval's derivative is checked at
both endpoints and any internal quadratic vertex, establishing monotonicity
for this table. Evaluating $W(F_*\pm\delta F)$ then bounds the peak rate;
the previously approved logarithm, polynomial and exponential evaluation
bound is added. The allowance is computed from the reference before running
the planar comparison. Here it is about $4.1\times10^{-6}$ relative, against
$F_*\simeq2.0145\times10^{10}\ {\rm V/m}$.
The unchanged field comparison remains the sensitive route-parity check.

The full planar module passes (10 tests, 31.90 s). Two independent bookkeeping
fixtures were also repaired: the low-storage memory comparison now includes
the reference's actual embedded-current rows, and a synthetic timing record
supplies the actual thread count required by the new identity guard. No
production equation or implementation changes were needed. The full coverage
run and affected-module repairs reach 100% of 42,836 statements and 12,970
branches. The NumPy/Numba slow functional lane passes 130 tests with 21 expected
skips (705.13 s), and the Kokkos-selected slow functional lane passes 148 tests
(789.24 s). Exclusive statistics-cost checks remain separate from these
concurrently executed correctness checks.

### Published SEP 24 closure — 2026-10-20

The ordinary merges of solprop `main` through `7bfbab51` and native `main`
through `496fbf5` bring in SEP 24's completed archive and native validation
guards. All previously integrated SEP 24 implementation is now also on `main`.
The solprop executable implementation was already present; its incoming source
changes only update documentation links. The native additions refuse an elided
stage register beside an in-stage absorber, more than one separable flow volume,
and reuse of a constant flow table before construction or after reinitialisation.
They do not change the held-current pipeline or the fixed-step timing workloads.

The final CPU builds use these merged guards. The earlier CUDA and HIP records
retain their actual revisions and scope; they are not relabelled as device tests
of this later integration. The worked-example refresh uses the preceding native
build, whose executed propagation paths are unchanged by these guards.

### Worked-example refresh — 2026-10-20

All fourteen worked examples admitting field-grid plasma pass the supported
`poe docs-examples` workflow. Their generated plots, printed values, downloadable
source archives and the README's VUV figure are refreshed. The
example record
retains each source and raw-log hash, 1,974.676 seconds of script execution and
2,564.45 seconds for the complete rendering workflows. They ran with four
numerical threads alongside correctness work, so these runtimes are provenance,
not evidence of a performance change.

The refreshed figures retain the examples' stated physical conclusions. This
work checks the published end-to-end workflows; the analytic tests and dedicated
campaign above supply the physics validation. The maintainer questioned the
value of additional example runs, and no further example campaign is planned.
Historical TDSE and Luna witnesses remain unchanged.

### Closing scope amendment — 2026-10-20

The maintainer requested prompt closure because other work depends on this SEP,
explicitly allowing remaining performance issues to move to the roadmap and
requiring them to be flagged here. The already running matched timing matrix
is retained, including resolved slowdowns and A/A variation. The unstarted
exhaustive compiled profiling pass is cancelled. Existing NumPy profiles remain
evidence; complete compiled attribution and optimisation are not claimed and
are no longer conditions for Final under this execution agreement.

The [roadmap](../../roadmap.md) records the radial and modal cost questions and
the Cartesian, line and planar handoff to SEP 28, together with the separately
discovered OpenMP allocation-cap issue. The production definition remains
$A_J$; no alternate formulation or performance-driven numerical relaxation is
introduced. Required correctness, schema, memory and statistics-cost checks
remain closing requirements.

### Phase 4 completion and performance review — 2026-10-20

The complete CPU performance record
contains 1,380 fresh processes in 144 comparison groups, with three alternating
rounds, held/held A/A, per-case medians, variability, actual thread pools and
memory ledgers. The [benchmark ledger](../../benchmarks.md)
reports both the old/new ratios and absolute Numba/Kokkos costs. All corrected
18-thread compiled groups actually use 18 physical cores; the earlier two-thread
Numba results are not used in this comparison.

Across fourteen plasma geometry/rule cases, the ten-step geometric-mean ratios
are 0.976/0.983 for NumPy at one/eighteen threads, 1.145/1.341 for Numba, and
0.944/1.103 for Kokkos. None of the sixty non-plasma control groups has a
resolved slowdown above 10%. At eighteen threads Kokkos is faster in absolute
time in every measured plasma case, despite its own modal and spatial
regressions. At one thread Numba remains faster for several spatial cases.
These are ten-step windows, not adaptive full-run speedup claims.

The largest changes remain visible: Numba's eighteen-thread radial source rises
67–72% and its window 53–67%, while Kokkos radial source falls 41–42% and its
window 53–54%. Kokkos multimode and planar trapezoid windows rise about 60% and
59%; Numba's corresponding increases are 64% and 61%. Some Numba A/A pairs
are noisy; the flagged increases survive the full observed process ranges.
The completed NumPy profiles explain added transforms and chunk passes.
Compiled attribution is incomplete and explicitly deferred under the closing
scope amendment. No formulation was substituted and no tolerance was relaxed
to improve timing.

The twenty-four final memory inventories preserve every held-current buffer
size from the frozen timing candidate. The incoming Cartesian and line
far-field tables and host copies add 512 and 784 bytes respectively in these
workloads; measured FFT-plan heap variation remains separately recorded.
The private gated and alternative-B prototype comparisons retain their own
identities and all fourteen one-thread NumPy geometry/rule groups. No benchmark output
can resume into the production equation.

The full physics campaign, fixed-field replay, window investigation, source
comparisons and actual work counts are complete as reported above. The raw
intermediate scalar ladders near 1e-12 remain explicitly unbounded; their
unnecessary tightening was stopped at the maintainer's request. The spectral
and endpoint observable residuals resolve the reported changes. No additional
physical constant, material coefficient or literature claim was introduced
by this closing work. Earlier source-verification limits remain unchanged.

### Phase 5 completion and final review — 2026-10-20

The closing CPU record
records the integrated full gate: 16,451 tests passed, 27 expected skips, all
static checks passed, and the strict documentation build passed. Coverage and
its focused repairs reach 100% of 42,836 statements and 12,970 branches.
The Numba slow functional lane passes 130 tests with 21 expected skips;
the Kokkos-selected slow functional lane passes 148. The separate exclusive
statistics-cost checks pass for both compiled backends. There is no remaining
failed required check.

The native closing record
adds the final main integration to the earlier complete gate. The incoming
low-storage construction fixture initially omitted Cartesian transverse axes,
so the existing extent validator fired before the intended new guard. Commit
`5284a6e` supplies axes consistent with the fixture's existing unit transverse
wavenumber; this changes only the construction test. After repair, Debug CTest,
ASan/UBSan, all 336 Python tests (five expected skips), the distribution-wheel
build and its isolated smoke pass. Static checks had already passed and the
changed fixture's formatting was rechecked. No gate stage was suppressed.

The final integrated executable pair is solprop `def695fe` and solprop-kokkos
`5284a6e`, incorporating main through `7bfbab51` and `496fbf5` respectively;
subsequent solprop commits publish this evidence and archive the proposal.
All earlier integrated SEP 24 work is now on main. The CUDA compile-only and
MI325X execution records retain their actual earlier revisions and qualified
scope; they are not relabelled as device tests of later statistics additions,
CPU matrix extensions or construction guards. The single HIP rental was
retrieved and destroyed, with absence confirmed and estimated cost USD 0.9838
of the USD 20 cap. No campaign rental or queued compute job remains.

All fourteen affected worked examples have been regenerated with provenance.
Historical TDSE and Luna witnesses are preserved. Manual pages, docstrings,
kernel/output contracts, current architecture and changelog describe the held
source; kernel schema 33 and output schema 22 are the integrated contracts.
The physical ungated `PlasmaResponse.evaluate` contract, general-modal refusal,
envelope plasma, Kerr, Raman and TDSE behaviour remain as specified.

**Handoff and remaining work.** Phases 1–5 are complete under the recorded
scope amendment. Mark this proposal Final and archive it. The roadmap retains
(1) the OpenMP core-place allocation-cap defect and (2) compiled held-current
cost attribution and optimisation. SEP 28 can use this completed implementation,
the 144-group timing matrix and final memory inventory as its baseline; its
implementation is not started here. The scalar precision limitation and device
revision scope are recorded above, not silently claimed as resolved. Source
changes remain locally committed on the topic branches; publishing or merging
them into main remains a separate maintainer action.

## Part III — History

- 2026-10-20: Phases 1–5 completed and proposal marked Final. Remaining compiled
  performance attribution and optimisation are roadmap follow-ups under the
  maintainer's closing amendment; required correctness checks pass.

- 2026-10-19: Drafted at the maintainer's request, answering SEP 26 Q3 and Q10. The
  drafting probes of §4.4, §4.5 and §10 used `FieldGrid`, `Collar` and `RunningIntegral`
  at solprop `9eef5425`, with no propagation.
- 2026-10-19: Revised before review at the maintainer's instruction (relayed by the coordinating agent). There is no backward compatibility: the held source replaces the truncation
  outright. The public option, opt-in phases, record and resume rules and the promotion
  phase were removed. A private comparison hook serves validation only.
- 2026-10-19: Q1–Q6 resolved by the maintainer (the drafting agent's recommendations).

### AI review — 2026-10-19

**Reviewers and disposition.** GPT-6 (AI); John Travers (human, recorded at his
request). Review status is **Reviewed**; lifecycle status remains **Draft**. This
records completion of the reviews, not acceptance. Part I has not been edited by
the reviewer. The drafting agent should answer each numbered point in a separate
`Response to review` subsection and make any resulting revisions to Part I there.

**Assessment.** The continuous held-current identity, the motivation for removing
the polarisation edge, and retaining the existing non-plasma responses are well
motivated. The committed SEP 26 record supports the quoted fixed-field differences.
However, the proposed discrete spectral identity is conditional on a boundary
extension that the admitted grids do not all satisfy; the two descriptions of the
frequency-domain oracle are not the same discrete computation; and several
mandatory tests assert incorrect or incomplete identities. Resolve the P1 points
before acceptance. These findings do not establish that option B is unusable: they
establish what its specification and qualification still need to show.

**Evidence scope.** Reviewed all of Part I against the architecture, conventions,
SEP 0, the relevant derivations and reports in SEPs 18, 25 and 26, the current
reference plasma passes, `RunningIntegral`, envelope source factors, output
identity, and the Numba and Kokkos column structure. Numerical probes used solprop
`7e7155c8`, NumPy 2.5.2 and SciPy 1.18.1; the Kokkos source inspected was
`391ca7d`. These were synthetic fixed-current calculations and one independent
quadrature check, with no propagation or backend qualification. The hashes and
numbers in `benchmarks/sep26/evidence/rate-plasma.json` were inspected; the saved
L6 fields were not reacquired or replayed.

#### R1 — P1: qualify the boundary extension in the periodic identity (§4.4–4.5, T1–T2, T6)

The Fourier multiplier follows when the filter uses periodic samples. The existing
`RunningIntegral.filter` instead uses point reflection. The draft recognises this
condition but then asserts that all the new rows satisfy it, citing a
$4\times10^{-15}$ collar level from one SEP 18 grid. That is not an admission
guarantee on the grids in §4.5.

Counterexample: construct
`FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 40e-15)` and its band rule.
It has $N=512$, $\Delta t=1.1118803173271735\times10^{-16}$ s and $K=4$.
The maximum collar value in the first or last $K$ samples is
$1.5605\times10^{-8}$, while the proposed certificate is
$3.4286\times10^{-13}$: this grid is explicitly admitted. For
$J=[1+\operatorname{erf}((t+5\,\mathrm{fs})/(1\,\mathrm{fs}))]/2$, directly evaluating
the two closed integrals in §4.4 differs from its claimed multiplier expression by
$2.57\times10^{-9}$ of the analytic active-band polarisation norm. The same
comparison on run_short is $1.21\times10^{-14}$. These are distinct from the
rule's interior multiplier error and the collar certificate.

Please either define periodic extension for the two new periodic integrations, or
derive and budget the actual reflection residual at both integrations. In the
latter case, let $d_y=F_{\rm reflect}y-F_{\rm periodic}y$. Before rounding, the
correct one-integral relation is

$$
\operatorname{DFT}[\mathcal J_Ry]_k
=\frac{R(\theta_k)Y_k+\rho(\theta_k)\operatorname{DFT}[d_y]_k}{i\omega_k}.
$$

Propagate this term through the nested definition. The certificate on $g$ alone
does not bound it for $g'J$ and the outer integrand. Include the admitted 512-point
grid, the supported smooth FFT sizes and currents reaching either taper in the
tests. Specify what happens when the new boundary bound is not met; do not assume
that small error in the run_short probe proves the identity for every grid.

#### R2 — P1: choose one discrete oracle and reconcile the comparison units (§4.4, §6 A, T6, §7.2)

Option A as written subtracts from the old sampled polarisation. SEP 26's
`rate_plasma.py` instead constructs the held source directly from the sampled
current. They agree continuously when $P'=J$, but that is not an exact identity
for $P=I_R[J]$.

Write $q=i\omega$, $V=\operatorname{DFT}[gP]$, $X=\operatorname{DFT}[gJ]$,
$Y=\operatorname{DFT}[g'P]$, $Z=\operatorname{DFT}[g'J]$, and
$C=qV-X-Y$. Then

$$
A_P=V-Y/q-Z/q^2,
\qquad A_J=X/q-Z/q^2,
\qquad A_P-A_J=C/q.
$$

Ignoring R1's boundary residual, $B=RX/q-R^2Z/q^2$. The proposed T6 bound accounts
for $B-A_J$, but it omits $C/q$ if the oracle is §6's $A_P$. The existing fixed-field
record explicitly measures a nonzero $C$; it is not FFT rounding.

The review reproduced the draft's run_short erf probe ($t_c=-5$ fs,
$\sigma=1$ fs). Relative to the analytic active-band polarisation, the errors are:

| rule | B | subtraction from P, $A_P$ | direct current form, $A_J$ |
|---|---|---|---|
| band | $7.010\times10^{-8}$ | $6.995\times10^{-8}$ | $5.32\times10^{-16}$ |
| trapezoid | $1.622\times10^{-4}$ | $1.619\times10^{-4}$ | $5.32\times10^{-16}$ |

Thus the draft's table is consistent with $A_P$, while the SEP 26 replay uses
$A_J$. Please name them separately, use the direct-current form as the independent
held-source reference, and retain $A_P$ as a decomposition check with its measured
closure residual if desired. Also distinguish raw nodal polarisation spectra from
the projected, cropped, prefactor-weighted `Q` in the replay: carry those linear
maps into every bound and divide the bound by the same denominator when reporting
relative errors. As written, T6 and §7.2 mix a source norm with raw-polarisation
operands and call the resulting estimate approximately $5\times10^{-6}$ of the band.

#### R3 — P1: retain the additive edge error when the physical spectrum is small (§4.5, T1–T2)

T2 bounds the rule error by $\varepsilon_R(2+\varepsilon_R)$ times the analytic
erf-rise spectrum. That omits the residual of the steady tail which §4.4 itself
reports. Using R2's direct oracle,

$$
B=R A_J+R(1-R)Z/q^2.
$$

The second term does not vanish when the rise's Gaussian spectrum becomes small.
It needs an additive, operand-based bound; multiplying the oracle alone cannot
bound it. Broader rises make the distinction particularly visible. Please carry
that term, R1's boundary terms, the appropriate product-alias terms and rounding
separately in T2, including bins with negligible physical signal. A band floor may
be used for relative reporting, with absolute bounds retained elsewhere.

Likewise, $\kappa_R\le\varepsilon_R$ is not the general certificate bound. Even
with a periodic filter,

$$
\kappa_R\le
\varepsilon_R\frac{\|\operatorname{DFT}[g]\|_A}{\|\operatorname{DFT}[g]\|_2}
+(1+\varepsilon_R)\kappa_{\rm ss},
$$

before rounding; reflection adds R1's residual. The steady-source bound also
carries the outer factor $R$ on the collar residual. Define $\varepsilon_R$ for
the trapezoid on the actual active bins, since its `certified_error` is not the
band-rule certificate. Explain explicitly that a steady-current certificate
qualifies that particular collar identity, not aliasing of arbitrary unresolved
$gJ$ or $g'J$ products.

#### R4 — P1: correct the analytic drift sign and the zero-drift tests (T4–T5)

With the definitions in T4, $E=-\partial_t A$ and $A(\pm\infty)=0$ give

$$
\int fE\,dt=-[fA]_{-\infty}^{\infty}+\int f'A\,dt,
\qquad
J_\infty=+\frac{e^2}{m_e}\Delta f\,(A*\phi_{\sigma_i})(t_i).
$$

The draft has a minus sign. An independent quadrature check, in dimensionless
variables, with $A_0=1$, $\tau=3$, $\omega_0=2.1$, $\varphi=0.4$,
$t_i=0.7$, $\sigma_i=0.8$, $f_0=0.2$ and $\Delta f=0.3$, gives
$\int fE\,dt=+0.1396845403804866$ and the same positive smoothed-potential value.
Please correct the analytic reference rather than changing the implementation's
sign to satisfy the proposed test. State whether this test isolates the free
current or also constructs an ionisation-loss current consistent with the
prescribed fraction.

T5's second assertion is also false: $J$ vanishing before the taper does not make
held and gated sources agree. A compact current with $\int J\,dt\ne0$ leaves
$P_\infty\ne0$, exactly the counterexample in §4.1 and T4(b). Require both the
current and its residual polarisation to vanish for that equivalence test, and
keep a separate test that removes the residual-polarisation edge. T4(c) should
not make the total removed term depend only on $J_\infty$: it also depends on
the residual polarisation, which T4(b) deliberately varies independently.

#### R5 — P2: state the two-edge continuation when the leading taper is occupied (§4.1, §4.4, T9)

The plateau equality and trailing-only formula for $H$ in §4.1 assume
$g'J=0$ throughout the leading taper. T9 deliberately violates that assumption.
For the stated zero pre-window current, integration by parts gives

$$
J_h(t)=g(t)J(t)-\int_{t_a}^{t}g'(s)J(s)\,ds.
$$

On the plateau this is $J(t)-\int_{\rm lead}g'J\,dt$, and its final held value is
$H=-\int_{\rm lead+trail}g'J\,dt$. This is consistent with the discrete definition
subtracting both edges, but not with an unconditional statement that it equals
$J$ on the plateau or that $H$ depends on the trailing taper alone. Ionisation
in the leading taper can also populate it without pre-ionisation; admission does
not constrain the field there.

Please give both formulas, label the empty-leading-taper limit, and make T9 test
the plateau offset and final value against them. Also correct the stated area
of $g'P$: in general $\int g'P\,dt=-\int gJ\,dt$, not minus the polarisation at
an unspecified end point. Distinguish the time endpoint from the propagation
coordinate $z$ in that statement. These are clarification of the proposed
continuation, not a request to add another source policy.

#### R6 — P1: derive the work test for the held source, including window terms (T8, §7.5)

The held source is deliberately different from the windowed transform of $J$.
Even with exact integration, $qA_J=X-Z/q$. Relative to the raw periodic transform
$\widehat J$, its current-equivalent difference is

$$
qA_J-\widehat J=\operatorname{DFT}[(g-1)J]-Z/q.
$$

There is no general bound on its work against $E$ using only
$\varepsilon_R$ and the field's out-of-band fraction. A field overlapping a taper
can have a finite contribution from either term while occupying only active
bins. With nonzero final drift, comparing a periodic transform of $J$ with the
continued current also requires the specified endpoint convention.

A concrete counterexample uses R1's 512-point grid and a pre-ionised Drude
current, with its constant coefficient scaled out: set
$E_n=\cos(2\pi 9n/N+\pi/4)$ and
$J_n=[\sin(2\pi 9n/N+\pi/4)-\sin(\pi/4)]/\omega_9$.
This is a single in-band field mode with $J'=E$ and $J(t_a)=0$.
Normalising the work sum by $\|\widehat E\|_A\|\widehat J\|_A$, the raw-current
work is $-9.8\times10^{-17}$, whereas the direct held-current work is
$-1.3707\times10^{-3}$. The field's out-of-active-band norm fraction is only
$1.47\times10^{-15}$. The finite window term is therefore present even before
introducing the running-integral error.

Please derive the discrete work identity for the proposed continuation, including
the filter multiplier, both window terms, endpoint/reflection effects and the
projection/crop. Specify the real-field half-spectrum weights and raw-DFT
normalisation. Where the field has negligible overlap with these contributions,
bound the remaining work defect in terms of those actual operands. Test that
limit and a deliberately occupied taper. The old gated source's work difference
from $J$ also includes $(g-1)J$; $\max_{\rm taper}|E|\,\|g'P\|_1$ alone bounds
only its polarisation-edge part. Preserve SEP 18's population/loss-work defect
as a separate check, since changing the final source integral does not remove it.

#### R7 — P2: make the rounding budget implementable through the whole new operator (§4.6)

The new terms have different units until propagated through the outer integral.
For example, a prefix bound for $a=g'J$ uses trapezoid increments
$\Delta t(\tilde a_n+\tilde a_{n-1})/2$, not the unscaled sum of $|\tilde a|$.
The latter is not a current error. The FIR also needs its own operand bound,
including absolute tap weights and point-reflected endpoint samples; cancellation
in $\tilde a$ cannot be used to bound the error in forming it.

Please express the budget as error propagation through named operators: the
existing error in $J$ enters both $gJ$ and $g'J$; the derivative and products feed
the first filter and scan; uncertainty in $S_a$ feeds the inner closure and outer
integral; and uncertainty in the outer total feeds the final closure. Then apply
absolute projection weights, FFT, crop and source factors. The
$4\varepsilon(|S_a|T+|S|)$ expression can cover arithmetic on already exact
totals, but does not by itself cover the computed totals' errors. Retain the
existing protected arithmetic and qualify the FFT constant on supported smooth
sizes, as SEP 25 requires. This can be a concise recurrence, not a catalogue of
every kernel operation.

#### R8 — P1: distinguish comparison outputs and reject incompatible continuation (§5, §6, §9, §11)

The private gated hook changes the differential equation while §5 explicitly
excludes it from the run identity. Phase 4's external provenance flag helps a
reader, but does not stop two HDF5 outputs or resumable checkpoints from having
the same configuration hash under different sources. It also makes the new
`plasma_source` metadata ambiguous on a gated comparison run.

Please specify a private benchmark output identity/label for the source form, or
make gated comparison output explicitly non-resumable and segregated from
production output. This does not require a public option or a compiled legacy
path. For old production files, state the concrete rejection boundary and the
identity/schema field that changes. The kernel schema bump alone is not a stated
saved-output identity rule: `RunMetadata.configuration_hash` has its own schema
and records. No backward compatibility is required, but silently continuing an
old equation with the new source must not be an accidental consequence of leaving
the version decision to the implementer. Test the new rejection/identity contract.

#### R9 — P2: align the route matrix and public response contract with current admission (§2, §5–6, §7.4)

The scope lists general modal plasma, but the current general-modal manual's
admission section permits the regions' cubic and silica tensor Raman responses;
it does not expose a field-plasma material route. The generic gas/plasma keywords
are explicitly refused beside a provider. Its local fields also have three
components, whereas T6 and §7.4 only exercise one and two. Please identify the
existing admitted general-modal plasma path if one is intended, or say that the
change applies to every route that already admits field plasma and preserves
general-modal refusals. Adding that physics route would be additional scope.
List the line free-space case explicitly as either covered by its shared
Cartesian path or outside the admitted matrix. Tests should cover the actual
matrix, including the mixed Kerr/Raman paths whose collar placement changes.

Separately, `PlasmaResponse.evaluate` currently returns the physical ungated
$P=I_R[J]$ and calls itself the backend oracle. It receives neither $g$ nor $g'$,
so it cannot produce §4.4's periodic representative with its unchanged signature.
Specify its intended continuing meaning and how its physical-current/response
check is separated from the new assembled-source oracle. A docstring saying
only that every `PlasmaResponse` is now held would misdescribe this public call.
Also qualify the envelope comparison in §4.2: `shock=False` freezes its inverse
frequency factors at $\omega_0$, so the exact $1/\omega^2$ identity describes
the frequency-dependent envelope policy.

#### R10 — P2: recompute live support after closure, and budget its cost (§6, Phase 3)

The statement that $g'J$ is zero until the trailing taper excludes T9's supported
leading-taper case. More fundamentally, closing its integral subtracts
$nS_a/N$, which is nonzero across the window even when the unclosed edge integral
is confined to the trailing taper. The outer integrand is therefore not supported
only where the original current or edge was live.

Please preserve `band_support` only where its exact-zero precondition holds.
Either determine support from each actual new row, as the current Kokkos code
does, or carry the analytic ramp separately and derive the corresponding scan and
closure totals. Write the final correction over the whole output row, including
samples skipped by the scans. Add empty-row, leading-edge, trailing-edge and
scratch-reuse cases, so a previous node cannot leak through an unwritten region.
Measure the revised support and extra full-row work in the hot-path benchmark;
the existing quiet-window saving cannot simply be assumed unchanged.

#### R11 — P2: specify independent convergence and consistent spectral scaling (§7.3, §9)

The convergence ladder is a good requirement, but the tightest run has zero
difference from itself. It needs a bound on its remaining error from the last
three resolved levels, with further tightening if that evidence is inconclusive;
the five listed tolerances must not become a fixed endpoint that automatically
certifies the reference. Record `atol`, maximum-step and save-plane policy, and
use the resulting residual bound for each form when deciding whether their
difference is resolved. No particular convergence order should be assumed from
the `rtol` ratios alone.

For the window control, common frequency bins are not sufficient to compare raw
DFT coefficients. Convert each spectrum to the same continuous-transform
normalisation (sample spacing included), Fourier origin and amplitude convention,
and state how the identical physical launch is rebuilt on each grid. Retain band
energy comparisons when bins do not coincide.

Finally, §6's assertion that the derivative changes only in the near infrared
and §9's expectation that propagated differences are attenuated are stronger
than the evidence. The committed source record has a nonzero UV difference, and
nonlinear propagation can amplify a perturbation. Keep the numerical expectations
labelled as hypotheses and let the matched-input controls, integration envelopes
and window control determine the result; no expected sign or attenuation should
be a pass condition.

#### R12 — P2: record the approved full-run exception and make the handoff public (§11–12)

§7.3's full 0–4.05 m comparisons conflict with SEP 0 §7 and the standing
qualification limit in `AGENTS.md`. This was clarified with the maintainer during
review.

**Maintainer decision — John Travers, 2026-10-19:** allow the full-length §7.3
validation campaign as an explicit exception because these runs should be fast.
The drafting agent should record that exception in the proposal. The ordinary
backend qualification and hot-path timings remain bounded as specified in §7.4
and the standing rules. This review did not execute or schedule the campaign.

Also replace the private machine name in Phase 4 with its public compute
requirement (CPU execution); put actual hardware in the eventual benchmark
ledger. Replace coordination through an unnamed “master” with a concrete
cross-SEP integration responsibility. Q3's resolved statement says the TDSE
certificate fix is already done, while its recommendation still asks for a future
roadmap item: mark that recommendation as superseded by the recorded resolution
and cite the public change. No additional maintainer decision is needed on the
full-run question.

#### Sources checked and remaining limits

The bibliographic details of Geissler et al. were checked against the
[APS record](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.83.2930),
and Brunel against the
[Optica record](https://opg.optica.org/josab/abstract.cfm?uri=josab-7-4-521).
Their full articles were not available in the accessed publisher views, so their
equations were not independently verified from those articles. The held-source
and drift identities above were re-derived from the equations stated in the SEP.
The Planck taper and equation (7) were checked in the
[McKechan, Robinson and Sathyaprakash paper](https://orca.cardiff.ac.uk/id/eprint/21998/1/tapering_window_for_time-domain.pdf).
The chapter title and radix-two/twiddle assumptions were checked in the
[SIAM chapter 24 excerpt](https://epubs.siam.org/doi/book/10.1137/1.9780898718027);
this does not independently certify $c_F=8$ for every FFT implementation. No new
physical constant or material coefficient was transcribed. The CPU propagation
campaign, production implementation, device builds and empirical cost estimate
remain work for the implementation phases.

- 2026-10-19: GPT-6 completed the AI review, R1–R12; John Travers's human review
  was recorded at his request. Set review status to `Reviewed`, retain lifecycle
  status `Draft`, and leave Part I unchanged. The maintainer approved the full
  §7.3 validation runs as a qualification-scope exception (R12). The drafting
  agent's responses remain due.

### Response to review — 2026-10-19

The drafting agent (Opus 5.5) revised Part I; the review text above is unchanged. The coordinating agent decided each point under the maintainer's delegation. One decision is returned to
the maintainer: Q2, the change of implementation (R1–R3).

**The implementation choice (R1, R2, R3, R7, R10).** Before answering the points one by
one, the agent reconsidered B against the direct-current form $A_J=X/q-Z/q^2$, which R2
showed is exact to $5\times10^{-16}$ under both rules.

- *What $A_J$ costs.* Two extra real temporal transforms per node and component, against
  one filter-and-scan saved, because $P$ is no longer formed. The per-route counts are in
  §6.
- *Indications.* In `numpy_ref` one band running integral costs about four real FFTs on
  run_short's lattice, so $A_J$ is a net saving there. In compiled code the estimate runs
  from neutral (single-mode modal) to about $+20\%$ of a nonlinear evaluation
  (Cartesian), against B's estimated 10–25% of the plasma column.
- *Radial.* Projecting three arrays would triple the dominant GEMM, so the radial route
  transforms per node and projects once in frequency.

The cost is acceptable, so $A_J$ is now the definition and the recommended
implementation (§1, §4.4, §6). B is kept only as a later optimisation, admitted against
$A_J$ within a derived budget and with periodic filters. Q2 is reopened as a changed
recommendation for the maintainer.

- **R1, resolved by the change.** $A_J$ adds no running integral, so the reflection
  residual does not arise. The column's two remaining integrals keep SEP 18's boundary
  budget unchanged. If B is pursued later, it must use periodic extension for its new
  integrals (§6, "Kept as a later optimisation"). The 512-point grid remains in T7, and
  T2 runs on it.
- **R2, accepted.**
  - $A_P$ and $A_J$ are named separately (§4.4). $A_J$ is the definition and the
    independent reference. $A_P$ is T6's decomposition check, asserting
    $A_P-A_J=C/q$ within the budget and reporting $\lVert C/q\rVert$.
  - §4.4 defines the assembled source $Q=pW\,\mathrm{crop}[\sum\Pi(\cdot)]$, and every
    bound and relative figure in §7 is carried through these maps.
  - §7.2 (i) now compares two implementations of the same $A_J$, so its bound is the
    rounding budget, not $5\times10^{-6}$.
  - The probe table is the review's.
- **R3, accepted.**
  - T2's tolerance is additive and absolute per bin: the collar residual of the steady
    tail, the product aliases, the tail mass and rounding. It covers bins with negligible
    signal, and a band floor is used only for relative reporting.
  - $\kappa_R$ is removed, since $A_J$ has no rule multiplier. B's additive term
    $R(1-R)Z/q^2$ is named in §6 for any later use.
  - §4.5 now states that $\kappa_{\rm ss}$ certifies the collar identity only, not
    aliasing of arbitrary sampled products.
- **R4, accepted.**
  - T4's reference is $J_\infty=+(e^2/m_e)\Delta f\,(A*\phi_{\sigma_i})(t_i)$; the
    implementation is not touched. T4 now isolates the free current, with the loss
    current zero; the loss current is covered by T1, T6 and the column runs.
  - T4(c) treats the removed term as the sum of a drift part and a residual-polarisation
    part, each linear in its own amplitude.
  - T5 requires both $J$ and $\int J\,dt$ to vanish for the equivalence, and keeps a
    separate residual-polarisation case.
- **R5, accepted.**
  - §4.1 gives the general continuation (plateau $J-c_-$, final
    $H=-\int_{\rm lead+trail}g'J$) and the labelled empty-leading-taper limit. It notes
    that ionisation can occupy the leading taper without pre-ionisation.
  - §4.1 replaces the area statement by $\int g'P\,dt=-\int gJ\,dt$, uses $t_\pm$ for the
    tapers, and reserves $z$ for propagation.
  - T9 tests $c_-$ and $H$.
- **R6, accepted in a simplified form.**
  - §4.7 derives the discrete work identity for $qA_J$: the half-spectrum weights, the
    raw-DFT Parseval form, both taper terms, the periodic window term $HM_1$ and the
    out-of-band term.
  - T8 gates the field-off-the-tapers limit and reports one occupied-taper case without
    gating it.
  - The claim that $\varepsilon_R$ and the out-of-band fraction bound the work is
    withdrawn. SEP 18's population and loss-work check stays separate.
  - Projection and crop are not carried into the work test, which is nodal; the
    assembled-source parity covers them.
- **R7, accepted.** §4.6 is now a short recurrence through named operators: the
  current's column bound, the products, projection, FFT (with $c_F$ qualified on the
  supported smooth sizes), the two divisions and assembly. With $A_J$ it is SEP 26
  §4.5's budget plus a second division. The closure and FIR terms of B are no longer
  needed.
- **R8, accepted.**
  - §5: gated comparison runs carry a private benchmark label in metadata and in the
    hashed configuration. Their output refuses to resume and is written only to the
    benchmark scripts' own directories. There is no public option.
  - §6: the rejection boundary for old production files is the output `SCHEMA_VERSION`
    bump. `RunMetadata.configuration_hash` digests it, so every old resume key differs.
    The kernel schema also moves.
  - Both refusals are tested (§7.1, "Seam and output"). The concurrent cleanup's output
    schema 21 and TDSE descriptor v5 are noted; numbers are taken at implementation.
- **R9, accepted.**
  - Scope is every route that already admits the field plasma: modal and radial
    (`_over_nodes`), Cartesian (`_nonlinear_free`), line free space (the same path with
    one $y$ sample, covered) and planar (`_nonlinear_slab`). General-modal refusals are
    preserved (§2, §3).
  - §7.4 tests that matrix, with plasma alone, with Kerr, with Raman, and with both.
  - `PlasmaResponse.evaluate` keeps its meaning as the physical ungated $P$, separate
    from the private assembled-source oracle (§5).
  - The envelope comparison is qualified for `shock=False` (§4.2).
- **R10, resolved by the change.** $A_J$ adds no closure ramp. The two remaining scans
  keep their existing support logic, and the new pointwise rows $gJ$ and $g'J$ are
  written in full. The hot-path benchmark measures the column with its third scan
  removed. If B is pursued later, its support must be recomputed per actual row (§6).
- **R11, accepted.**
  - §7.3 replaces the fixed five-level ladder. It tightens until the tightest level's
    residual is bounded from the last three levels by $d_{\rm last}r/(1-r)$, assumes no
    order, and records `atol`, the maximum step and the save-plane policy.
  - The window control uses continuous-transform normalisation, rebuilds the launch from
    the same pulse definition, and compares band energies.
  - The near-infrared-only and attenuation statements are now labelled hypotheses, not
    pass conditions (§6, §7.3, §9).
- **R12, accepted.**
  - §7.3 records the maintainer's full-length exception.
  - Phase 4 says "CPU execution", and hardware goes in the benchmark ledger; the
    execution handoff stays in `benchmarks/sep27/README.md`.
  - The unnamed coordinator is replaced by the implementer's integration responsibility
    at merge (§11).
  - Q3's roadmap recommendation is marked superseded by the TDSE certificate bug fix,
    cited as pending on `refactor/remove-compat-paths` (`f6b76751`).

**Drafting probes for this revision**, at solprop `2f30fb0c`, with no propagation:

- The work identity of §4.7 closed to $3\times10^{-15}$ of the work for a plateau field
  and a stepped current on run_short's lattice.
- 33 rows of $N=2048$ took 104 µs per `numpy.fft.rfft` and 441 µs per band running
  integral.

**Checks.** `git diff --check`, codespell, `uv run pytest tests/test_seps.py` and
`uv run poe docs` (strict, 8 workers).

- 2026-10-19: Part I revised in response to R1–R12. $A_J$ replaces the in-column form as
  the definition, pending the maintainer's confirmation of the reopened Q2. Status stays
  Draft.
- 2026-10-19: Q2 resolved by the maintainer: $A_J$ is the definition and implementation;
  the `integration` rule's narrower meaning accepted.
