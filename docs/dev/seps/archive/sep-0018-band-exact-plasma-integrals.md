# SEP 18 — Band-exact running integrals for the field-grid plasma

| | |
|---|---|
| **SEP** | 18 |
| **Title** | Band-exact running integrals for the field-grid plasma |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-09) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-10-09 |
| **Resolution** | Accepted 2026-10-09 by John Travers after the human and AI reviews and the author's response to them. Q2: warn above the undershoot threshold. Band floor $10^{-6}$ accepted. Q3, Q4, Q5, Q7, Q9 and Q10 as the author recommended. Q8 remains with the device campaign. Final, approved by John Travers, 2026-10-09, with Q16 and the worked examples' re-run left on the roadmap. |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers; Phases 1–4 merged from `feature/sep-0018-band-integrals-3` (which holds `-1` and `-2`) in solprop, Phases 1 and 3 in solprop-kokkos |
| **Manual pages** | [The plasma response](../../../guide/media/field-plasma.md#the-plasma-response); [output](../../../guide/results/output.md) |

## Part I — Proposal

### 1. Abstract

The field-grid plasma response forms the ionised fraction, the free-electron current
and the polarisation with three cumulative trapezoids along $t$. On a Fourier component
the trapezoid's multiplier is $(\theta/2)\cot(\theta/2)$ times the exact one, with
$\theta = \omega\,\delta t_{\rm fine}$. The default grid puts the top of the band at
$\theta \approx 1.15$. There the polarisation is too small by 7–9 %, and its Drude part
by 20–26 %. This SEP proposes to keep the running sums but filter each integrand first
with a short symmetric FIR filter, nine taps on the default grid. The taps are computed
from the grid so that the interior, linear filter-and-trapezoid multiplier differs from
one by at most $10^{-5}$ across the active band. The proposed column keeps one traversal
of $t$ and gains no transforms. This multiplier bound is not a bound on the complete
nonlinear, finite-window response. The revision after review therefore settles the
column's other parts separately:

- the window ends are closed by point reflection (§4.4);
- the exposure is floored at zero instead of clamped, and its undershoot is recorded
  (§4.5);
- the loss current keeps the raw rate, and its work defect is recorded (§4.5);
- the physics evaluators take a resolved integration object from translation (§5).

Today's trapezoid stays available as a legacy policy with an identity tap and unchanged
arithmetic, for the frozen Luna comparisons. Success means three things. A pre-ionised
tone has the Drude susceptibility across the active band in the interior, and the
startup terms are bounded separately. The polarisation of an ionising field, and its
effect over ten propagation steps, agree with an independent reference (§4.8, §7) to
better than $10^{-4}$ in every band that carries a measurable share of the response.
Backends agree with each other within justified summation bounds. Here **band-exact**
means the stated approximate multiplier property, not an exact antiderivative of the
whole column.

### 2. Motivation and scope

**The problem.** [accuracy.md §1](../../accuracy.md) records the multiplier and its
measured effect. On the default `ffac = 6` grid one integral is short by 9.3 % at
$\omega_{\max}$ and 11 % at $1.1\,\omega_{\max}$; two are short by 18 % and 21 %. On an
ionising field the effects combine. The setup was a 1.3 fs, 800 nm pump of 6e10 V/m
with weak seeds near 100 nm, on a 90 nm to 4 µm grid. Compared with the same field at
`ffac = 96`, the plasma polarisation is short by 1.5 % at $0.5$–$0.6\,\omega_{\max}$,
7.4 % at $0.95$–$1.0$ and 9.4 % in the collar, in helium and in argon alike. The Drude
part alone is short by 20–26 % above $0.85\,\omega_{\max}$. Ten fixed steps of a
capillary propagation get the plasma-induced change of the output spectrum wrong by
2–6 % across the helium band, and by up to 8.5 % in argon's collar.

**Evidence status.** The response and propagation measurements in this proposal are
drafting measurements, not committed qualification records. The AI review reproduced
the small FIR design and checked analytic counterexamples, but did not reproduce those
helium/argon studies. After the review they were repeated for the revised rule against
an independent reference: a Richardson extrapolation of today's trapezoid column, which
shares no filter, clamp or spectral antiderivative with the candidate (§4.8). The
complete protocol is in §7. They remain uncommitted until Phase 1 commits them as
convergence records, and the default does not change before that (Q12).

**Who needs it.** The top of the band is exactly where a user puts the content they care
about when they choose a grid tight around it. Examples are a vacuum-ultraviolet resonant
dispersive wave (RDW) on a grid whose short limit sits just below it, the
ionisation-induced blue shift on a grid cut near its edge, and the third harmonic of a
mid-infrared pump. In each case plasma dispersion and ionisation loss act on that content
10–20 % too weakly today, and nothing warns about it. The two-photon current of
[SEP 17](../sep-0017-gas-two-photon-ionisation.md) joins this column's polarisation
integral at VUV frequencies (its §4.9 and Q18), so its accuracy depends on this one.

**In scope.**

- The three running integrals of the field-grid plasma column:
  - the exposure $\int W\,\mathrm{d}t$;
  - the free current $\int fE\,\mathrm{d}t$;
  - the polarisation $\int (J_{\rm free}+J_{\rm loss})\,\mathrm{d}t$.

  This covers the one-channel path and the mixture path, on every route that uses the
  field-grid column: modal, mode-averaged, radial, Cartesian, planar, line free space and
  general modal.
- The physics-layer evaluators that define the same column:
  {py:meth}`solprop.nonlinear.responses.PlasmaResponse.evaluate` and `ionised_fraction`.
- The plasma statistics that read the column's ionised fraction.
- All three backends: `numpy_ref`, `numba_cpu` and Kokkos CPU, with a design for the
  Kokkos device scan.
- A legacy option that reproduces today's trapezoid exactly.

**Out of scope.**

- **The envelope plasma** ([envelope plasma](../../../guide/media/envelope-plasma.md)). It already
  converts its currents with exact inverse-frequency factors after the transform
  ({py:func}`solprop.translate.envelope_plasma.source_factors`).
- **The TDSE current stream.** It is exact on its active bins
  ({py:func}`solprop.translate.tdse_source.current_numerator`).
- **Changing the sampling policy for products** $fE$ and $W(|E|)$, whose content can
  lie above the fine Nyquist frequency. The reported examples suggest small aliasing
  there; this is not a general guarantee at `ffac = 6`. Qualification must distinguish
  sampling error from integration error, and refuse or document unresolved cases.
  One such case was separated after review. The 30 fs argon case of §7 item 4 leaves
  $2\times10^{-4}$ in the collar band at `ffac = 6`, a band that holds $1.7\times10^{-8}$
  of the response. That residual does not change with $K$, the fit range or the
  exposure policy, and it falls to $1.9\times10^{-7}$ at `ffac = 12` under the same
  rule. It is product sampling, not integration error.
- **The fine-grid oversampling factor** and its default.
- **Other cumulative trapezoids in the package**, such as the spectral-phase integral in
  {py:mod}`solprop.optics`. The public helper
  {py:func}`solprop.nonlinear.ionisation.ionisation_fraction` is Q3.

### 3. Context

The column is defined in [nonlinear.md](../../../guide/media/field-plasma.md#the-plasma-response) and
[algorithms.md](../../algorithms.md) ("Ionisation and the plasma current", "Order and
accuracy"). Where it is implemented:

- **`numpy_ref`.** The oracle is {py:func}`solprop.backends.numpy_ref.passes.plasma_scan`
  (exposure and free current), {py:func}`~solprop.backends.numpy_ref.passes.plasma_current`
  (polarisation), and `plasma_channel_fractions` with `plasma_channels_polarisation` for
  mixtures. Each running integral is a call to {py:func}`scipy.integrate.cumulative_trapezoid`.
- **The physics layer.** {py:class}`solprop.nonlinear.responses.PlasmaResponse` repeats the
  same column.
- **`numba_cpu`.** `make_response_column` fuses the whole nonlinear term into one
  traversal of $t$ per node. It carries the three running sums as scalars, through the
  shared increment `trapezoid` in {py:mod}`solprop.backends.bodies`, in the same order
  as SciPy, so it matches the oracle's multipass scan bit for bit.
- **Kokkos.** The CPU build follows that traversal. The CUDA path evaluates the rate in
  parallel and then runs each running sum as a deterministic fixed-tree inclusive scan
  with one team per node ([algorithms.md](../../algorithms.md)).

The kernel contract ([kernel-contract.md](../../kernel-contract.md)) fixes the stage order
`intensity, ionisation_rate, plasma_scan, plasma_current, …` and carries one
`PlasmaChannel` leaf per channel; `SCHEMA_VERSION` is 24.

Current tests:

- `tests/nonlinear/test_plasma.py` holds the second-order ladder on a constant field, the
  energy identity, the exact two-integral multiplier on a pre-ionised tone, and the
  band-top multiplier itself (0.8225 and 0.7866).
- `tests/nonlinear/test_luna_plasma.py` holds the kernel-level agreement with Luna, 5.4e-14
  ([accuracy.md §3](../../accuracy.md)), which uses the same trapezoid.

The M2 RDW and M4 CtC witness comparisons and several convergence records (`m2_order`,
`m2_mode2_along_z`, `backend_equivalence`, `spatial_accuracy`, `plasma_trapezoid`) run
with plasma on.

### 4. Theory

#### 4.1 The trapezoid's multiplier

This subsection uses the internal FFT convention of
[conventions.md §2](../../conventions.md): a positive-frequency tone is $e^{+i\omega t}$.
The cumulative trapezoid $Y_{n+1} = Y_n + (\delta t/2)(y_n + y_{n+1})$, $Y_0 = 0$, with
$y_n = e^{i\theta n}$ and $\theta = \omega\,\delta t$, has the exact solution
$Y_n = A(e^{i\theta n} - 1)$. Substituting it into the recurrence gives

$$
A\,(e^{i\theta}-1) = \frac{\delta t}{2}(1+e^{i\theta})
\quad\Longrightarrow\quad
A = -\,i\,\frac{\delta t}{2}\cot\frac{\theta}{2}.
$$

The exact antiderivative of $e^{i\omega t}$ is $-i\,e^{i\omega t}/\omega$, so relative to
it the trapezoid applies

$$
\rho(\theta) = \frac{\theta}{2}\cot\frac{\theta}{2}
= 1 - \frac{\theta^2}{12} - \frac{\theta^4}{720} - \dots ,
$$

which is real, with no phase error, positive on $[0, \pi)$, and zero at the fine Nyquist
frequency. The sign convention does not matter: $\rho$ is even in $\theta$. The constant
$-A$ is the integration constant that `initial=0` sets, and it carries no oscillating
content.

On the field grid $\delta t_{\rm fine} = 2\pi/(f_{\rm fac}\,\omega_{\max})$, or the user's
`dt` if finer ([grids.md](../../../guide/setup/grids.md)). The kept bins reach the collar top
$\omega^{\rm win}_{\max} = 1.1\,\omega_{\max}$, so the band edge in $\theta$ is

$$
\theta_b = \omega^{\rm win}_{\max}\,\delta t_{\rm fine} \le \frac{2.2\pi}{f_{\rm fac}},
$$

which is 1.152 at `ffac = 6` and 1.728 at `ffac = 4`. [accuracy.md §1](../../accuracy.md)
tabulates $\rho$ and $\rho^2$ there.

#### 4.2 How the three integrals combine

The loss current goes through one integral, so its polarisation carries $\rho$. For a
fixed fraction the Drude term goes through two, so it carries $\rho^2$; this is the
identity the tests hold. On an ionising field the exposure's integral also gives the
sub-cycle steps of $f$ the factor $\rho$, evaluated at their own frequencies. Those steps
reach a band-top output bin by mixing with the field, from $f$ content at
$\omega_{\rm out} \mp \omega_E$. That is why the measured Drude shortfall, 20–26 %,
exceeds $1-\rho^2$. A remedy must therefore treat all three integrals, the exposure
included. Measured on the ionising helium field of §2, exact spectral antiderivatives for
the two current integrals alone leave the Drude part 5–8 % short at the band top. Adding
the exposure brings it to agreement with the `ffac = 96` reference.

#### 4.3 A filtered running sum

Replace each running integral $\int y\,\mathrm{d}t$ by the trapezoid of a filtered
integrand,

$$
\tilde y_n = c_0\,y_n + \sum_{k=1}^{K} c_k\,(y_{n-k} + y_{n+k}),
\qquad
C(\theta) = c_0 + 2\sum_{k=1}^{K} c_k \cos k\theta ,
$$

so that the composite multiplier is $H(\theta) = C(\theta)\,\rho(\theta)$. The
requirements:

- **Symmetric taps.** $C$ is real, so the composite rule adds no phase error to the
  trapezoid's.
- **Unit response at zero frequency.** $C(0) = c_0 + 2\sum c_k = 1$. Then a constant
  integrand passes unchanged, and $\sum_n \tilde y_n = \sum_n y_n$ whenever $y$ vanishes
  within $2K$ samples of the ends. For that **same integrand**, with no clamp, the
  final trapezoidal integral is preserved up to round-off. This does not preserve
  the full column's final exposure after clamping, nor its residual free current:
  correcting the exposure changes $f(t)$ and hence the next integrand $fE$. The
  revised exposure policy (§4.5) has no prefix clamp. Its final exposure is therefore
  the trapezoid's for the same sampled rate whenever the rate vanishes near the ends,
  which it does exactly below the rate table's threshold. Downstream totals, such as
  the residual current, are measured against the reference and not asserted equal.
- **Flat in band.** $|H(\theta) - 1| \le \varepsilon_{\rm design}$ on $[0, \theta_b]$.
- **Bounded above the band.** The proposed bound is $|H| \le 1.05$ from $\theta_b$
  to Nyquist, allowing at most 5 % amplification relative to the exact integral.
  This is distinct from bounding $C$ itself. Out-of-band content can feed the
  products further down the column, even though the final active mask removes it.

**Design.** Fit $C$ to $1/\rho$ by least squares on $[0, \theta_b]$ under the constraint
$C(0) = 1$, by writing $C = 1 + 2\sum_k c_k(\cos k\theta - 1)$. This is a $K$-unknown
linear least-squares problem, solved once per grid at translation. Take the smallest $K$
that meets both the in-band and out-of-band bounds. A least-squares residual alone does
not certify a maximum-norm bound. Specified after review:

- **Nodes and weights.** 4097 uniformly spaced nodes on $[0, \theta_b]$, zero included,
  with unit weights. The objective is $C$ against $1/\rho$, as in every measurement
  here. Fitting $H - 1$ directly, with weights $\rho$, changes the $K = 4$,
  `ffac = 6` maximum from $4.92\times10^{-6}$ to $5.18\times10^{-6}$ and is not
  preferred.
- **Evaluation near zero.** $\rho(\theta) = 1 - \theta^2/12 - \theta^4/720$ for
  $\theta < 10^{-4}$, and $(\theta/2)/\tan(\theta/2)$ above. The two agree there to
  round-off.
- **Conditioning.** The columns $2(\cos k\theta - 1)$ are scaled to unit norm before an
  SVD-based solve (`numpy.linalg.lstsq`). On every admitted design below, the scaled
  condition number is at most $4.3\times10^4$. A design whose condition number exceeds
  $10^{10}$ is refused, which no admitted grid reaches. Choosing the smallest adequate
  $K$ keeps small $\theta_b$ from pairing with a large $K$.
- **Certified maxima.** The maximum of $|H-1|$ on $[0,\theta_b]$ and of $|H|$ on
  $[\theta_b, \pi)$ is found on 8193 independent sample points. Every sampled local
  maximum is then refined by a bounded scalar search (`scipy.optimize.minimize_scalar`,
  `xatol` $10^{-12}$). $H$ is analytic on $[0, \pi)$, so this verified extremum search
  is the certificate. It does not reuse the fitting nodes.
- **$K = 0$.** The plain trapezoid is the band design whenever
  $1 - \rho(\theta_b) \le \varepsilon_{\rm design}$, that is
  $\theta_b \lesssim 0.011$ (`ffac` above about 630). The provenance still records
  `band`.

Certified designs at $\varepsilon_{\rm design} = 10^{-5}$, $K \le 8$:

| `ffac` | $\theta_b$ | $K$ | $\max\lvert H-1\rvert$ in band | $\max\lvert H\rvert$ above band | condition |
|---|---|---|---|---|---|
| 3.5 | 1.975 | refused (best at $K=8$: $1.2\times10^{-5}$) | — | — | — |
| 3.6 | 1.920 | 8 | 6.7e-6 | 1.0000 | 3.2e4 |
| 4 | 1.728 | 7 | 3.6e-6 | 1.0000 | 4.3e4 |
| 5 | 1.382 | 5 | 4.4e-6 | 1.0000 | 1.5e4 |
| 6 | 1.152 | 4 | 4.9e-6 | 1.0000 | 6.5e3 |
| 8 | 0.864 | 3 | 5.3e-6 | 1.0000 | 1.6e3 |
| 12 | 0.576 | 2 | 9.6e-6 | 1.0000 | 1.1e2 |
| 24 | 0.288 | 2 | 1.5e-7 | 1.0000 | 4.5e2 |
| 48 | 0.144 | 1 | 1.9e-6 | 1.0000 | 1 |
| 96 | 0.072 | 1 | 1.2e-7 | 1.0000 | 1 |

$H$ never exceeds one above the band on these designs: the out-of-band bound is met
with room to spare. The fixed-width comparison of the drafting table, $\max|H-1|$ on
$[0,\theta_b]$ for a given $K$:

| $K$ (taps $2K+1$) | `ffac = 6`, least squares | `ffac = 6`, Taylor-matched (order $2K+2$) | `ffac = 4`, least squares |
|---|---|---|---|
| 2 | 6.8e-4 | 6.3e-3 | 9.1e-3 |
| 3 | 5.7e-5 | 1.7e-3 | 1.9e-3 |
| 4 | 4.9e-6 | 4.4e-4 | 3.9e-4 |
| 5 | 4.3e-7 | 1.2e-4 | 8.2e-5 |
| 6 | 3.7e-8 | 3.3e-5 | 1.7e-5 |

The Taylor-matched column chooses the taps so that $C = 1/\rho$ to $O(\theta^{2K+2})$.
That is the classical high-order corrected trapezoid, and it is an order of magnitude
worse in band for the same width, because it spends its freedom at $\theta = 0$, where
the trapezoid is already exact. Every least-squares design in the table keeps $H \le 1.05$
up to Nyquist; $H$ falls to zero there with $\rho$.

The `ffac = 6` design at $K = 4$ has

$$
c = (1.405171,\ -0.259303,\ 0.068837,\ -0.013510,\ 0.001390),
$$

centre first, from the 4097-node design above, rounded for illustration only. The
full-precision coefficients give a certified $\max|H-1| = 4.92\times10^{-6}$ on
$[0, 2.2\pi/6]$, which agrees with the review's independent probe ($4.925\times10^{-6}$).
Do not use these rounded numbers as production taps or as the unit-sum assertion. The
drafting measurements used a 4000-node variant, whose taps differ in the fifth decimal.

**Evidence for fitting the exposure to the band edge (Q1).** §4.2 shows that $f$ can need
accuracy beyond $\theta_b$. Five ionising fields support fitting to $[0, \theta_b]$:
helium and argon, weak, strong and saturating, short pulses with VUV seeds, and a
30 fs argon pulse with many ionisation steps (§7 item 4). On none of them does fitting
to $[0, 1.4\,\theta_b]$ help. At $K = 4$ it is worse by up to fifty times: $5\times10^{-5}$
against $1.8\times10^{-6}$ on helium, and $1.5\times10^{-4}$ against $3\times10^{-6}$
on the saturating argon field. Raising $K$ to 6 at the wider range only recovers the
band-edge fit's accuracy. The $f$ content that mixes into the band comes mainly from its
steps' low harmonics, and a wider fit spends accuracy where it is not needed. This
settles the range for the fields studied. It remains an admission statement backed by
§7's cases, not a theorem for arbitrary fields.

#### 4.4 Endpoints

The filter reaches $K$ samples past each end of the window. Candidate treatments:

- **Zero extension.** This agrees with an actually vanishing integrand near the
  ends. The temporal collar does not guarantee this for every integrand: the free
  current can retain a constant residual after the field vanishes. In the existing
  column the collar multiplies the completed polarisation, not each integrand.
  For an integrand that does not vanish there, such as
  the constant-field unit tests, it puts an $O(1)$ error into the first $K$ filtered
  samples.
- **Pass-through** (the drafting proposal). For $n < K$ or $n \ge n_{to} - K$,
  $\tilde y_n = y_n$. Constants pass exactly. But the seam between filtered and
  unfiltered samples breaks the filter's symmetry, and startup errors in the first
  integral become a secular ramp in the second. That is the review's counterexample,
  reproduced exactly: ramp coefficient 0.05133 for $\cos(n\pi/3)$ from its maximum.
- **Point reflection (proposed after review).** Virtual samples
  $y_{-m} = 2y_0 - y_m$ and $y_{N-1+m} = 2y_{N-1} - y_{N-1-m}$ for $m = 1\dots K$.
  The filter is then applied unchanged at every sample. The extension is exact for
  constant and linear integrands, so the residual free current $J_\infty$ at the
  trailing end and the constant-field ladder's integrands pass exactly. Its weights
  have norm 3, so no content is amplified.
- **Band-limited prediction** of the virtual samples from the first $M$ samples, by
  least squares over the band. It was tried and rejected. It makes a switched-on tone of
  any phase exact (ramp $7\times10^{-7}$), but band-limited extrapolation is badly
  conditioned: the prediction weights' row norms are $2\times10^3$ at $M = 8$ and
  $3\times10^6$ at $M = 16$. Any out-of-band content at the edge, such as a rate spike,
  would be amplified by those factors.

**What no local closure can do.** An integrand that is non-zero at $t_a$ makes the
causal integral from $t_a$ depend on how the signal is read across $t_a$. A tone
switched on abruptly at $t_a$ is not band-limited, and every local rule leaves an
integration constant that grows into a ramp in the second integral. That includes
today's trapezoid. Measured on the review's probe family ($\theta = \pi/3$, 512
samples, two integrals; ramp coefficient per unit-amplitude tone):

| start | trapezoid | pass-through | point reflection | even reflection |
|---|---|---|---|---|
| $\cos$ from its maximum | 0 (but 18 % amplitude error) | 0.0513 | 0.0254 | 0 |
| $\sin$ from zero | 0.0889 | 0.178 | $1.7\times10^{-6}$ | 0.0889 |
| constant | 0 | 0 | 0 | 0 |

Point reflection is never worse than the trapezoid's secular error on these probes,
except for the cosine started at its maximum, where it is 0.025 against the trapezoid's
0. That case is the even-symmetric start, which even reflection gets right. No single
reflection gets both. The accuracy claim is therefore restricted to integrands that
vanish at the window start, with a separate boundary budget for those that do not.

**Production integrands vanish at $t_a$.**

- **The rate.** It is exactly zero below the rate table's `field_min`, so the exposure
  integrand is zero at both ends unless the field there ionises.
- **The free current.** With no pre-ionisation $f(t_a) = 0$, so the integrand $fE$ is
  zero at $t_a$. With pre-ionisation it is $f_0 E(t_a)$.
- **The current.** $J(t_a) = 0$ by construction.
- **The field at the edge.** Measured on the ten-step launches of §7, it is
  $\approx 10^{-3}$ of its peak: well below any ionisation threshold.

The only leading-edge boundary term in production is therefore the pre-ionised Drude
term, proportional to $f_0\,|E(t_a)|$, and today's trapezoid has the same term. On a
band-limited periodic test field (the field grid's own representation) whose envelope
is $\lambda$ of its peak at $t_a$, the whole-window error of the double integral is about
$30\lambda$ for point reflection and pass-through alike, on a 4096-sample window at
$\theta = \pi/3$. For $\lambda = 10^{-8}$ it is at the multiplier level,
$3.8\times10^{-6}$. The coefficient grows with the window length in carrier periods,
because the term is secular. Trailing-end closure errors cannot propagate backwards,
since the sums are causal. They stay within the last $3K$ samples, where the temporal
collar is at most $4\times10^{-15}$ on the default grid.

**Finite-window budget.** The proposed error budget is the interior multiplier tolerance
$\varepsilon_{\rm design}$ plus a boundary term $\kappa\,\lambda$. Here $\lambda$ is the
largest integrand level in the first $2K$ samples relative to its peak. $\kappa$ is
measured by the §7 item 2 test for the default window and stated with it, as about 30
for 4096 samples at the band top. The evaluator records $\lambda$ for the free-current
integrand of a pre-ionised run, the only one that can be non-zero in production, as a
diagnostic beside those of §4.5.

#### 4.5 Positivity and monotonicity of the fraction

$W \ge 0$, but $\tilde W$ need not be: negative taps let the filter undershoot beside a
sharp rate spike. On the helium field of §2 at `ffac = 6`, the smallest exposure
increment is $-4.8\times10^{-6}$ of the largest. Then $f$ is not monotone at that
level, and with no pre-ionisation it can dip a few parts in $10^6$ of a step below
$f_0$. The drafting proposal clamped it with a prefix maximum over a separate raw
exposure, as the review corrected it:

$$
S_0 = X_0 = 0,\qquad
S_n = S_{n-1} + \tfrac{\delta t}{2}(\tilde W_{n-1} + \tilde W_n),\qquad
X_n = \max(X_{n-1}, S_n).
$$

Then $f_n = 1-(1-f_0)\exp(-X_n)$. This restores $f \in [f_0,1]$ and the
non-decreasing contract of `plasma_scan`, but can permanently increase the final
exposure and alter subsequent positive increments. It is **not** the recurrence
$X_n=X_{n-1}+\max(0,S_n-S_{n-1})$, which clips negative increments and accumulates a
different bias. NumPy's `maximum.accumulate` and a device max-scan implement the
prefix maximum, and a serial pipeline must keep both $S$ and $X$.

Neither rule preserves the net exposure in general. For a single isolated sampled
rate spike with $\delta t\,W=1$, the nine-tap design gives raw final exposure 1,
prefix-maximum exposure about 1.073, and clipped-increment exposure about 1.203.
This adversarial case tests the discrete contract; it is not a resolved physical
ionisation pulse. It disproves a universal small-bias or unchanged-total claim.
Only exposure is a candidate for clamping; signed currents are not clamped.

**The prefix maximum is withdrawn (Q2).** Its bias is not only a contract question: it
costs accuracy on resolved fields. On the helium field of §7 item 4, the polarisation's
error over the whole active band is $1.4\times10^{-7}$ with no clamp and
$3.1\times10^{-5}$ with the prefix maximum. Its permanent exposure bias,
$7.3\times10^{-7}$, shifts the residual drift current and with it the low-frequency
response. The review's spike value, 1.072934, is reproduced exactly.

**Proposed exposure policy: floor at zero, record the undershoot.**

$$
S_0 = 0,\qquad
S_n = S_{n-1} + \tfrac{\delta t}{2}(\tilde W_{n-1} + \tilde W_n),\qquad
X_n = \max(S_n, 0),\qquad f_n = 1-(1-f_0)\,e^{-X_n}.
$$

- **Range.** $f \in [f_0, 1)$ always.
- **Totals.** The floor is pointwise, not a prefix operation, so it accumulates nothing.
  $X_{\rm end} = S_{\rm end}$, which is the trapezoid's final exposure for the same
  sampled rate (§4.3), whenever the rate vanishes near the ends. On the review's spike
  the final exposure stays exactly 1.
- **Monotonicity.** $f$ is non-decreasing only to within the filter's undershoot. This
  changes `plasma_scan`'s contract from "non-decreasing" to "in $[f_0, 1)$, and
  non-decreasing to within the recorded undershoot $u$".
- **The undershoot diagnostic.** $u$ is the largest of $-\min_n\Delta S_n/\max_n\Delta S_n$
  over nodes and evaluations. It is recorded as a run statistic. On the five ionising
  fields of §7 item 4 it is at most $2.2\times10^{-4}$, and on the ten-step runs at most
  $5.3\times10^{-5}$. On the review's one-sample spike it is 0.166.
- **Admission.** An undershoot above $10^{-2}$ means the rate varies within a sample
  faster than the fine grid resolves, which is a sampling failure for any integration
  rule. The proposed response is one warning naming the node and suggesting a finer
  `dt`, not a refusal (Q2, decided at acceptance).

**Population and work (Q11).** The loss current keeps the raw rate,
$J_{\rm loss} = I_p W(1-f)E/|E|^2$, so $E\cdot J_{\rm loss} = I_p W(1-f)$ holds sample
by sample, as it does today. The population follows the filtered exposure, so
$\mathrm{d}f = (1-f)\,\mathrm{d}S$, which holds whether or not $S$ is monotone. The
defect between the work paid and the electrons freed is then

$$
D = \int (1-f)(W - \tilde W)\,\mathrm{d}t = \int W\,\big[(F - I)f\big]\,\mathrm{d}t .
$$

The second form uses the symmetric filter's self-adjointness on sequences whose
extended samples vanish, which is true for $W$ near the ends, and $F$'s preservation of
constants. $(F-I)f_n = \sum_k c_k(f_{n-k} - 2f_n + f_{n+k})$ is a second difference of
the fraction at the filter scale. It vanishes where $f$ is linear over $\pm K$ samples,
and it is $-(\delta t^2/12)f'' + O(\delta t^4)$ because $\sum_k c_k k^2 = -1/12$, the
consistency condition $C(\theta) = 1 + \theta^2/12 + \dots$. So $D$ is second order
in $\delta t$, the same order as today's finite-step defect. Measured, with
$D_{\rm rel} = \max_{\rm nodes}|{\rm paid} - {\rm freed}| / \max_{\rm nodes}{\rm freed}$:

| fields | band rule | today's trapezoid |
|---|---|---|
| §7 item 4, five ionising fields (response level) | $\le 4.7\times10^{-8}$ | up to $5.3\times10^{-4}$ (saturating argon) |
| §7 item 5, ten steps, every node and evaluation | $\le 5.2\times10^{-9}$ | — |

$D_{\rm rel}$ is computed exactly at run time from two running sums the column already
forms. It is recorded as a run statistic, with a proposed warning above $10^{-4}$ (Q2,
Q11). Normalising by the largest node's ionisation, rather than taking a per-node
ratio, is necessary. While this revision was prepared, a per-node ratio reached 0.46 at
radial nodes with negligible ionisation, where both sides are rounding noise.

Locally, $\mathrm{d}f > 0$ can occur within $K$ samples of a rate spike where $E = 0$,
so the energy exchange is not simultaneous on the scale of the filter, $K\,\delta t$ or
200 as on the default grid. A band-limited column cannot localise the release more
finely than its band. The trapezoid's sharper localisation is itself the sampled,
$\rho$-attenuated version of the same content. The identity that is kept is global, to
$D$, and it is measured.

**Alternatives rejected for Q11.**

- **A filtered or effective rate in the loss term.** The review's objection stands:
  filtering can create population growth where $E = 0$.
- **An adjoint survival factor** $W\,F(1-f)$ in place of $W(1-f)$. It would make the
  totals match by construction. It was measured and rejected: it changes the physical
  current in time, costing up to $1.6\times10^{-2}$ in the band (already
  $1.3\times10^{-4}$ at 2 % ionisation), and it
  leaves the finite-step defect at today's size ($5.4\times10^{-4}$ on saturating
  argon).

#### 4.6 Causality

$\tilde y_n$ reads $y_{n+K}$. With three filtered integrals in a chain, the polarisation
at $t_n$ can depend on samples up to $t_n+3K\,\delta t_{\rm fine}$, about 600 as for
the 90 nm grid at `ffac = 6` and $K=4$. This is a noncausal discrete approximation
to a causal material response. The existing cumulative trapezoid at $t_n$ uses
only samples through $t_n$; it is not half a sample acausal. The draft's claim
otherwise was wrong and is withdrawn. Band limitation does not remove the need to
quantify precursor response. Measured after review, on two filtered integrals of a
tone switched on abruptly at $t_s$ (the worst case: a jump in value):

| switch-on | precursor of $J$ before $t_s$ | precursor of $P$ before $t_s$ |
|---|---|---|
| $\cos$ at $\theta = \pi/6$ | $2.3\times10^{-2}$ of peak, 4 samples | $2.1\times10^{-4}$, 8 samples |
| $\cos$ at $\theta = \pi/3$ and $1.1\pi/3$ | $4.5\times10^{-2}$, 4 samples | $2.4\times10^{-4}$, 8 samples |
| $\sin$ at the same $\theta$ | $\le 9.9\times10^{-3}$, 4 samples | $\le 6.8\times10^{-5}$, 8 samples |
| smooth Gaussian onset | the field's own level: $8.5\times10^{-7}$ where the field is $10^{-6}$ of peak | |

The precursor is confined to $K$ samples per filtered integral, $3K$ for the column.
For the smooth fields a propagation produces, it sits at the level of the field
itself. Phase 1 adds the onset test (§7 item 2) and the look-ahead convergence under
refinement. It also updates the `memory` domain's documentation, in nonlinear.md and
`ResponseTraits`, which promises no dependence on later samples: under the band rule
the value at $t_n$ depends on samples up to $t_{n+3K}$.

#### 4.7 Relation to exact spectral factors and to SEP 17

The interior multiplier approximates the nonzero-frequency antiderivative to
$\varepsilon_{\rm design}$. Endpoints, secular terms, the nonlinear population map
and the exposure floor are outside that statement. The ten-step study of §7 measures
their combined effect. Direct current assembly in the envelope and TDSE routes
avoids a numerical current-to-polarisation integral; it is not automatically
equivalent to this finite-window column.

SEP 17's two-photon current joins this column's polarisation integral (its §4.9), so it
would inherit this integration rule if that path is retained. It does **not** close
SEP 17's Q18: its adjoint-filter work identity, density placement, projection,
windowing and boundary terms still require qualification with the revised operator.
$C(0)=1$ alone does not preserve these identities or the nonlinear column's net
current. SEP 17's Phase 0 review remains a prerequisite for its Phase 1.

#### 4.8 Finite-window reference and source work

For the internal transform $\widehat P(\omega)=\int_{t_a}^{t_b}
P(t)e^{-i\omega t}\,\mathrm{d}t$ and $J=\mathrm{d}P/\mathrm{d}t$,

$$
\widehat J = [P e^{-i\omega t}]_{t_a}^{t_b}+i\omega\widehat P.
$$

A nonzero net current therefore prevents replacing the causal finite-window
integral by division by $i\omega$ alone. If a collar $w$ multiplies $P$, then
$\mathrm{d}(wP)/\mathrm{d}t=wJ+w'P$: even when the endpoint term vanishes,
the $w'P$ contribution must be included in a direct-current comparison.

An independent spectral antiderivative must specify its interpolant, zero mode,
Nyquist convention and initial condition. For a periodic interpolant
$y(t)=\bar y+\sum_{k\ne0}\hat y_k e^{i\omega_k(t-t_a)}$, its causal integral is

$$
Y(t)=\bar y(t-t_a)+\sum_{k\ne0}\frac{\hat y_k}{i\omega_k}
\left(e^{i\omega_k(t-t_a)}-1\right).
$$

The mean term and integration constant cannot be dropped: the first integral's
constant becomes a ramp in the second, and exposure has a nonzero mean. Subsequent
integrands need not be periodic; their spectral integration requires a documented
boundary treatment and convergence under padding/refinement. A reference sharing
the candidate's clamp tests implementation agreement, not the clamp's physical
error. Q12 requires a separately converged causal reference as well.

**The reference adopted after review (Q12).** It is today's trapezoid column itself,
refined and extrapolated: $R = (4P_{96} - P_{48})/3$ from `ffac = 48` and `ffac = 96`.

- **Same coarse axis.** With `pow2` sizes and $\delta t$ halved $k$ times, the coarse
  frequency axis is unchanged, so spectra compare bin for bin.
- **Same causal start.** The trapezoid's own start, $Y_0 = 0$, applies, with no
  periodicity assumption, no mean or Nyquist convention, no filter and no clamp.
- **Second order.** Its error is second order in $\delta t$ on the resolved column
  (accuracy.md §1), and Richardson removes the leading term.
- **Uncertainty from a third refinement.** $(4P_{48} - P_{24})/3$ agrees with $R$ to
  $8\times10^{-6}$ per band and $\le 6.4\times10^{-9}$ over the active band on the five
  response-level fields. On the ten-step plasma-induced change it agrees to
  $6.3\times10^{-8}$ (He) and $8.4\times10^{-8}$ (Ar).

The exact spectral antiderivative of the drafting study stays a cross-check: it agreed
with the ffac 96 trapezoid to the latter's own $O(\delta t^2)$. It is not the qualifying
reference, because its non-periodic integrands need the boundary treatment above.

**Source work through the transform.** The work the band-limited source does on the
field is $\int E\,\partial_t P$ over the kept bins. Under the band rule
$\partial_t P = H J$ on those bins, to $\varepsilon_{\rm design}$, so the work is
$\int E J$ to that tolerance. Under today's rule it is $\int E\,\rho J$, which is short
by $1-\rho$ at band-top content. The existing energy test hides this because it
evaluates both sides with the same trapezoid. Phase 1 measures the transform-side work
with the collar term $w'P$ of this section included (§7 item 3).

### 5. Proposed interface

Subject to the acceptance blockers and qualification below, the proposed propagation
default changes. One option selects the rule:

```text
PlasmaResponse("Ar", rate, integration="band")        # default: filtered, band-exact
PlasmaResponse("Ar", rate, integration="trapezoid")   # today's rule, bit for bit

prop_capillary(..., plasma=True, plasma_integration="trapezoid")
```

- `integration: Literal["band", "trapezoid"] = "band"` on
  {py:class}`solprop.nonlinear.responses.PlasmaResponse`, and a `plasma_integration`
  keyword on each driver that builds a plasma response, forwarded to it (Q4).
- The design tolerance $\varepsilon_{\rm design} = 10^{-5}$ and the cap $K \le 8$ are
  package constants, not user options (Q1).
- The resolved rule, $\theta_b$, $K$, the taps and the achieved $\max|H-1|$ go into the
  translated problem's record and the run's output metadata, so a saved run says how its
  plasma was integrated (Q9).
- **Refusal.** If no design with $K \le 8$ meets the tolerance, translation refuses the
  `band` rule by name. That happens as $\theta_b \to \pi$: `ffac = 6` needs $K = 4$ and
  `ffac = 4` needs $K = 7$, but below about `ffac = 3.6` no design reaches the tolerance
  (at 3.5 the best gives 1.2e-5). The message names the achieved error and the two ways
  out: a finer `dt`, or `integration="trapezoid"` (Q5).
- **Unchanged physics outside scope.** Envelope plasma and TDSE retain their existing
  rules. Q4 must specify how convenience drivers handle an explicitly supplied
  field-plasma integration option on these routes, rather than silently imply it
  affects them.

**Interface requirements raised by the review (Q13).** `PlasmaResponse.evaluate` and
`ionised_fraction` currently receive `dt` and `kind`, but no active band edge or
grid. They cannot reconstruct this design from their existing arguments. Before
acceptance, specify how a standalone evaluator obtains the same resolved rule as
translation, without importing T into P or fitting to the input pulse's spectrum.
The public `ionisation_fraction` helper needs the same explicit decision. Also
define whether mixed integration rules across plasma channels are refused or
supported: one shared pair of current integrals cannot silently honour different
channel rules. Statistics, saved provenance and resume validation must consume
the same resolved rule, including endpoint and clamp policies.

**Resolved integration contract (proposed after review, Q13).** The layering rules
([architecture.md §3](../../architecture.md)) decide the split. P may not import T, and
hand-rolled numerics live in T or B. The tap design (a least-squares fit, a
conditioning check and a certified extremum search) is hand-rolled numerics, so it
lives in T. P holds only a value type and its application.

- **P: the value type.** A frozen dataclass in {py:mod}`solprop.nonlinear.ionisation`,
  provisionally `RunningIntegral`, carries:
  - `policy`: `"band"` or `"trapezoid"`;
  - `dt`, the sample spacing it was designed for, in s;
  - `band_edge`, $\omega^{\rm win}_{\max}$ in rad/s;
  - `taps`, the full-precision, read-only `(2K+1,)` array;
  - `certified_error`;
  - `boundary = "point-reflection"`;
  - `exposure = "floor-at-zero"`.

  `RunningIntegral.trapezoid()` is the legacy constant. Applying it to an integrand
  uses only NumPy (`numpy.convolve` on the point-reflected array) and
  `scipy.integrate.cumulative_trapezoid`.
- **T: the design.** A translation function, for example in
  {py:mod}`solprop.translate.responses` or a new `translate/plasma_integration.py` beside
  `translate.chebyshev`, takes the grid and the policy and returns a `RunningIntegral`.
  It applies §4.3's rules and refuses as in the bullet above. It is the only producer of
  a `band` instance; tests and users call it with a grid.
- **Standalone evaluators.** `PlasmaResponse.evaluate`, `PlasmaResponse.ionised_fraction`
  and the public `ionisation_fraction` (if Q3 says it follows) take an optional
  `integral: RunningIntegral`. If `integral` is omitted and the response's
  `integration` is `"band"`, they refuse by name. The message points to the translation
  function and to `integration="trapezoid"`. They also refuse an `integral` whose `dt`
  differs from the call's `dt` (relative $10^{-12}$), or whose policy differs from the
  response's. They never fit taps themselves and never look at the field's spectrum.
- **Mixed rules.** Every plasma response in one problem must request the same policy:
  the mixture path's current integrals are shared. Translation refuses a mixture whose
  responses disagree, naming them, and compiles one `RunningIntegral` per problem.
- **Statistics, provenance and resume.** Translation records the `RunningIntegral`
  fields in the problem record (Q9 decides the output form), including the boundary and
  exposure policies and the diagnostics' thresholds. Backend statistics read the taps
  from the compiled problem, never a second design. A resumed run re-translates and
  refuses unless the recorded policy and the taps are equal bit for bit.
- **Diagnostics.** The two run statistics of §4.5 (the undershoot $u$ and the work
  defect $D_{\rm rel}$) and the boundary level $\lambda$ of §4.4 are recorded for the
  band rule, and also for the legacy rule except where they are trivially zero.

### 6. Implementation strategies

**Layer.** The taps are a property of the grid and the band edge. They are computed in
translation ({py:mod}`solprop.translate.responses`, beside `plasma_coefficients`) and
carried as one read-only `(2K+1,)` float64 array per problem. The array is shared by
every channel if Q13 requires a common rule, because the mixture path's current
integrals are shared. The trapezoid is the one-tap case `(1.0,)`, but the legacy
branch must also bypass clamping and preserve today's operation order, initial
fraction rounding, guards and endpoint handling. One identity tap alone does not
establish bitwise compatibility. A `band` design resolving to $K=0$ must remain
distinguishable in provenance from the explicit legacy policy.

**Kernel contract.** A new field on the response tables (or on the plasma leaf; the
implementer chooses after Q13, and kernel-contract.md records it) bumps
`SCHEMA_VERSION` in Phase 1 from the then-current version (24 at review).
The stage order is unchanged, but the temporal dependency and integration policy
of `plasma_scan` and `plasma_current` change and must be documented. Older external
backends must reject the new schema; schema-compatible backends without the new
rule refuse it by name.

**`numpy_ref`.** Extend each integrand by point reflection (§4.4), filter it along $t$
and call `cumulative_trapezoid`. Floor the exposure at zero with `np.maximum(S, 0)`,
which is pointwise (§4.5), and form the undershoot and work-defect statistics from
arrays the column already holds. The order of the filter's sum is fixed: centre first,
then the pairs $k = 1 \dots K$, each pair summed before it is weighted. The virtual
samples are formed as $2y_0 - y_m$ in that order. The compiled backends reproduce that
order exactly.

**`numba_cpu`.** The fused traversal becomes a pipeline with lags.

- At loop index $n$, the kernel evaluates the field and the rate at sample $n$. It
  finalises the exposure and $f$ at $n-K$, the free current and the loss current at
  $n-2K$, and the polarisation at $n-3K$.
- Ring buffers retain rates per channel, delayed fractions and loss inputs, $fE$
  and $J$ per local field component, and enough field samples to form delayed
  products. The complete memory ledger and overwrite lifetimes are verified in
  Phase 2. General modal synthesis has up to three local components despite
  `n_pol=1`; buffer sizes must use `n_components` there.
- The loop runs to $n_{to} + 3K$. The leading virtual samples need $y_1 \dots y_K$,
  which the lag already holds. The trailing ones are formed when the last sample
  arrives.
- The `trapezoid` increment and loss-current guard retain their legacy order. The
  band path keeps one exposure per channel, floored where $f$ is formed, so no second
  exposure is carried. The loss current keeps today's raw-rate formula (§4.5). The
  two diagnostics are two running sums and a running minimum and maximum per node.
- The `reuse_unchanged_fraction` path still applies: it compares rounded exposures,
  whatever produced them.

**Kokkos CPU.** Follows `numba_cpu`'s order.

**Kokkos device (design only).** The device path keeps one team per node. Each fixed-tree
inclusive scan is preceded by a parallel stencil over $t$ with a $K$-sample halo:
filter, then scan, three times. That stays deterministic and adds three parallel maps.
The exposure floor is pointwise and needs no scan; the diagnostics are deterministic
reductions. GPU qualification follows the standing policy, separately.

**Cost.** With symmetric pairs, each filtered scalar uses $K$ pair additions,
$K+1$ multiplications and $K$ accumulator additions. There are $n_{\rm channels}$
exposure filters and two filters per local field component, plus scans, the floor,
the diagnostics and delayed-buffer traffic. Do not fuse multiply-adds or reassociate the reference
sum to obtain a lower operation count. Whether this cost is small beside rate
evaluation must be measured; the RDW step's plasma column must be benchmarked
before and after
(benchmarks.md), and phase 2 reports it. The extra memory is a few ring buffers per node,
or in `numpy_ref` one filtered copy of each integrand. No new dependency is needed.

**Options considered within B.**

| | the three integrals filtered, least squares, fitted to $\theta_b$ (**recommended**) | Taylor-matched taps | the two current integrals only |
|---|---|---|---|
| band error at $K = 4$, `ffac = 6` | 4.9e-6 (certified design); $\le 6.5\times10^{-6}$ measured on $P$ against the independent reference | 4.4e-4 | Drude part still 5–8 % short |
| $f$ monotone | to within the recorded undershoot $u$ (floor at zero) | the same | yes |
| formal order of the interior linear rule on smooth integrands | generally 2 | $2K+2$ | 2 |

The endpoint closure and the exposure floor are not covered by these interior
orders; in particular the Taylor-matched column does not make the whole proposed
finite-window method order $2K+2$.

### 7. Verification and validation

1. **The multiplier itself.** For every admitted grid in a ladder of `ffac`, 3.6 to 96,
   and user `dt` values, the designed $H$ meets $\varepsilon_{\rm design}$ on
   $[0, \theta_b]$ and satisfies $|H|\le1.05$ up to Nyquist. Verify extrema
   independently of the fitting nodes, including small $\theta_b$, $K=0$ and
   refusal near Nyquist. The test reads full-precision taps from translation. The
   §4.3 ladder is the expected result: refusal at `ffac = 3.5`, $K = 8$ at 3.6, $K = 4$
   at 6, $K = 1$ at 48 and 96, and $K = 0$ above about 630.
2. **The Drude limit across the band.** A pre-ionised tone at $\omega/\omega_{\max}$ =
   0.5, 1.0 and just below 1.1 on the default grid gives the **interior oscillatory**
   Drude susceptibility to $2\varepsilon_{\rm design}+\varepsilon_{\rm design}^2$
   plus round-off (the bound on $H^2-1$). Fit constant and secular terms separately.
   For the complete finite-window result, the tone is a band-limited periodic field
   (the field grid's own representation) whose envelope is $\lambda$ of its peak at
   $t_a$. It is compared with the exact causal double integral of that trigonometric
   polynomial.
   - **Target.** At $\lambda \le 10^{-8}$ the whole-window error is at most
     $10^{-5}$; measured $3.8\times10^{-6}$ at $\theta = \pi/3$, against 18 % for the
     trapezoid.
   - **Boundary coefficient.** A ladder $\lambda = 10^{-8} \dots 10^{-2}$ records
     $\kappa$ of §4.4.
   - **Abrupt switch-ons.** The review's probes and the sine are recorded with the
     §4.4 table's values. They are not asserted at $\varepsilon_{\rm design}$, since no
     local rule can meet that for them.
   - **Onset.** The test asserts that the precursor is confined to $3K$ samples and
     records its size (§4.6).

   `test_the_drude_polarisation_is_too_small_at_the_band_top` is rewritten into this
   test, and the exact two-integral identity holds under `integration="trapezoid"`.
3. **Constant-field ladder and energy identity.** Both converge at least at second order,
   and their convergence records are regenerated and committed. Quantify both
   $\int E\cdot J_{\rm loss}\,dt-I_p\Delta f$ and the loss work computed from
   the actual windowed, projected polarisation source. Their tolerances come from
   §4.5's error budget, not from the FIR multiplier alone: $D_{\rm rel} \le 10^{-6}$
   on the §7 item 4 fields, where it measures $\le 4.7\times10^{-8}$. The test also
   asserts the transform-side work against $\int E J$ to $10^{-5}$, with the collar
   term of §4.8. Include depletion, pre-ionisation, repeated spikes, vector fields and
   the zero-field guard.
4. **Response-level study, committed as a convergence record.** The protocol, complete
   so that the study can be reproduced (Q12):
   - **Grid.** `FieldGrid.from_limits(1.0, 800e-9, limits, 400e-15, ffac=ffac)`
     (`pow2`, collar 0.05). The limits are 90 nm to 4 µm for helium and 110 nm to 4 µm
     for argon. `ffac` is 6 for the candidate and 24, 48 and 96 for the reference. The
     coarse axes are asserted identical.
   - **Field.** A coarse spectrum, the sum of transform-limited Gaussians
     $a\,e^{-(\omega-\omega_c)^2\tau^2/(8\ln2)}$ with zero spectral phase, centred at
     $t = 0$ in mid-window and multiplied by the grid's spectral window. It is
     zero-padded onto each fine grid, rescaled by the size ratio and normalised to the
     stated peak on the `ffac = 6` fine grid.
   - **Cases.**
     - Helium at $6\times10^{10}$ and $9\times10^{10}$ V/m: a 1.3 fs, 800 nm pump
       ($a = 1$) with seeds at 120, 100 and 93 nm (4 fs, $a = 0.03$).
     - Argon at $4\times10^{10}$ and $8\times10^{10}$ V/m (the latter saturating,
       final fraction 0.63): the same pump, seeds at 160, 125 and 113 nm.
     - Argon at $2.5\times10^{10}$ V/m: a 30 fs pump with a 125 nm, 10 fs seed.
   - **Medium.** optmat's default rate through
     `OptmatRate(gas, 800e-9)`, tabulated by `plasma_coefficients` over
     $[10^8, 3E_{\rm peak}]$ V/m. Density $2\times10^{25}$ m⁻³, no pre-ionisation, one
     scalar column.
   - **Quantity and norm.** $\tilde P = \mathrm{FFT}(P\,w_{\rm fine})\,\delta t$ on the
     coarse bins. The bands in $\omega/\omega_{\max}$ are $[0.3,0.5)$, $[0.5,0.7)$,
     $[0.7,0.85)$, $[0.85,0.95)$, $[0.95,1.0)$ and $[1.0,1.1]$. The error in each band
     is the relative L2 error of the complex $\tilde P$, which bounds any
     spectral-energy error. It is asserted where the band holds at least $10^{-6}$
     of the active band's $|\tilde P|^2$, and only reported otherwise. The active-band
     error, normalised by the active-band norm, is asserted everywhere.
   - **Reference.** The Richardson reference of §4.8.
   - **Targets and results.** Every asserted band at most $10^{-4}$, and the active band
     at most $10^{-5}$. Measured: asserted bands $\le 6.5\times10^{-6}$ and active band
     $\le 1.8\times10^{-6}$, against $1.2\times10^{-2}$ to $1.6\times10^{-1}$ and
     $4.7\times10^{-4}$ to $6.0\times10^{-3}$ for today's trapezoid. Reference
     uncertainty: $\le 8.2\times10^{-6}$ per band and $\le 6.4\times10^{-9}$ over
     the active band.
5. **Ten-step propagation study, committed.** Same geometry as today's measurement:
   - single-mode 125 µm capillary, ten fixed steps, `numpy_ref`, split-step absorber;
   - helium at 0.8 bar over 5 mm with 150 µJ, argon at 0.4 bar over 2 mm with 30 µJ.

   The protocol:
   - **Run.** `prop_capillary(125e-6, L, gas, p, lambda0=800e-9, grid=…, pulses=…,
     modes=1, schedule=FixedSchedule.uniform(0, L, L/10), noise=None,
     absorber_scheme="split_step", backend="numpy_ref")`, otherwise at defaults. The
     grid is item 4's at `ffac` 6, 24, 48 and 96.
   - **Launch.** A `GaussianPulse` at 800 nm, 1.3 fs, with energy $\mathcal{E}$, plus
     three 4 fs seeds of $2\times10^{-4}\,\mathcal{E}$ each, at item 4's seed
     wavelengths.
   - **Quantity.** The plasma-induced change is the complex coarse exit spectrum with
     the plasma on minus the same with it off. With the plasma off, all `ffac` agree
     to $10^{-12}$, which is asserted.
   - **Reference.** The Richardson value from `ffac` 48 and 96, with its uncertainty
     from 24 and 48.
   - **Norm and targets.** As in item 4, applied to the plasma-induced change.
   - **Results.** Measured: every band $\le 4.2\times10^{-6}$ (He) and
     $\le 2.6\times10^{-6}$ (Ar), and active band $9.1\times10^{-7}$ and
     $7.6\times10^{-7}$. Today's trapezoid: 0.8–8.5 % per band and
     $2.9\times10^{-3}$ and $4.3\times10^{-3}$ over the active band. Reference
     uncertainty: $6.3\times10^{-8}$ and $8.4\times10^{-8}$.
   - **Diagnostics.** The run's $u \le 5.3\times10^{-5}$ and
     $D_{\rm rel} \le 5.2\times10^{-9}$.
6. **Totals.** For a fixed integrand vanishing within $2K$ samples of both ends,
   verify the filtered trapezoid's total against the ordinary trapezoid at a
   summation-error bound. Separately measure the changed final exposure, fraction,
   residual current and low-frequency emission of the complete column against the
   independent reference. Do not assert their equality to the legacy column when the
   population history changes.
7. **Range and undershoot.** $f \in [f_0, 1)$ on every study above and on an isolated
   one-sample rate spike. The undershoot $u$ is recorded:
   - $\le 2.2\times10^{-4}$ on item 4's fields;
   - 0.166 on the spike, which must raise §4.5's warning;
   - also on a sub-sample physical spike resolved by a refined reference, since one
     sampled value does not specify a spike's shape.

   The precursor population before the pulse is bounded by the filter's support
   (§4.6) and recorded. Long-pulse and saturating-fraction cases are included (Q1).
8. **Legacy rule.** `integration="trapezoid"` reproduces today's results bit for bit on
   all three backends. The Luna kernel test and the M2 and M4 witness comparisons
   state it among their frozen inputs and pass unchanged.
9. **Backend parity.** `numba_cpu` and Kokkos CPU match `numpy_ref` under the band rule
   with bounds derived from tap-weighted summation, scans and the population map,
   starting from [accuracy.md §7](../../accuracy.md). Cover scalar/vector fields,
   mixtures, all listed routes, three-component general-modal fields, short arrays
   with overlapping endpoint regions, phase-appropriate refusal, diagnostics,
   metadata/resume and thread-count invariance. Verify the physics evaluator uses
   the same resolved rule. Qualify the device scan separately.
10. **Benchmark.** The RDW step on `numba_cpu`, before and after, on a short window.

### 8. Documentation

- [nonlinear.md](../../../guide/media/nonlinear.md) §"The plasma response": state the rule, the
  interior multiplier tolerance, finite-window and population/work error budgets,
  causality approximation, validity/admission limits, and legacy option. State
  separately what is established about the Drude susceptibility and startup terms.
- [accuracy.md §1](../../accuracy.md): replace the measured-once paragraph with the
  committed records (items 4 and 5 above); keep the trapezoid's table for the legacy
  rule.
- [algorithms.md](../../algorithms.md): replace the caveat in "Order and accuracy" with the
  filtered rule and its cost.
- [kernel-contract.md](../../kernel-contract.md): the new field and schema version.
- [luna.md](../../../guide/workflows/luna.md): the deliberate difference from Luna's column.
- `CHANGELOG.md`: a changed-results entry citing this SEP.
- Docstrings of `PlasmaResponse`, `plasma_scan`, `plasma_current` and the mixture pair.

### 9. Compatibility and impact

- **Changed results.** Every field-grid plasma run changes. Content at a quarter of
  $\omega_{\max}$ or below moves by about 1 % of the plasma's own effect or less; content
  near $\omega_{\max}$ moves by 7–9 % of it, and the Drude part by up to a quarter. The
  pump-band energy budgets, final fraction and residual drift must be requalified;
  $C(0)=1$ does not guarantee they are unchanged (§4.3–§4.5). On §7's fields the final
  fraction agrees with the `ffac = 96` trapezoid's to five significant digits, and the work defect
  falls from up to $5.3\times10^{-4}$ to $\le 4.7\times10^{-8}$.
- **Schema versions.** The kernel `SCHEMA_VERSION` changes. The output schema gains
  metadata (Q9).
- **Tests and records.**
  - The witness comparisons pin `integration="trapezoid"`.
  - Convergence records with plasma on are regenerated: `m2_order`, `m2_mode2_along_z`,
    `backend_equivalence`, `spatial_accuracy`, `plasma_trapezoid`. Their quoted numbers
    are updated.
  - Tests that assert today's numbers at the band top are rewritten (§7 item 2).
- **Worked examples.** Those with plasma are re-run by hand: the VUV dispersive wave, the
  ionisation blue shift and any other plasma example.
- **Luna.** solprop departs deliberately from Luna's cumulative trapezoid. The
  difference goes on [luna.md](../../../guide/workflows/luna.md).
- **Breaking changes.** There is no public release yet, so no deprecation period is
  needed.

### 10. Alternatives considered

- **Exact spectral source streams** (the envelope route's pattern). In the internal
  FFT convention, the field-grid
  plasma would emit the Drude drive $\rho f (e^2/m_e) E$ and the loss current as two
  time-domain streams, transformed and multiplied by $-1/\omega^2$ and $-i/\omega$ on the
  kept bins.
  - Exact for nonzero periodic frequency components; causal boundary terms require
    the treatment in §4.8. The exposure's error remains. Measured, the Drude part
    is still 5–8 % short at the band top unless the exposure is also corrected.
  - It costs two more transforms per column per evaluation and two new kernel-contract
    streams, and it breaks the single fused traversal.
  - It also changes how the net-current and boundary terms enter, which SEP 17 Q18 would
    then have to requalify.
- **Exact spectral antiderivative for all three integrals.** Three transform pairs per
  column per evaluation. A drafting measurement reports $10^{-7}$ agreement over
  ten steps; Q12 must qualify this reference, including mean and boundary terms.
  Reported exposure undershoot is $6\times10^{-5}$, so sharing a clamp with the
  candidate would conceal its error. Cost and accuracy must both be compared;
  the linear FIR tolerance alone cannot establish equal nonlinear accuracy.
- **A finer fine grid.** Raising `ffac` or reducing the user's `dt` reduces the error
  as $\delta t^2$. A quarter of the error costs twice the fine-grid work for the whole
  nonlinear term, not just the plasma. Rejected as a default; it stays a user choice.
- **Correcting the total polarisation after the transform.** Impossible: the plasma
  polarisation is summed with the Kerr and Raman terms and windowed before the
  transform, so its share of a bin is not known.
- **A Taylor-matched high-order rule.** Higher formal order, but worse in band for the
  same width (§4.3); it suits smooth integrands, and the band edge is the problem here.
- **Documenting and leaving it.** Already done ([accuracy.md §1](../../accuracy.md)), but
  10–20 % at the band top is not a tolerable silent default for VUV work.

### 11. Phases

**Acceptance prerequisite.** Resolve Q2, Q6, Q11 and Q13 with a reviewed mathematical
and interface contract, and specify Q12's independent reference and reproducible
study protocol. The author's proposed resolutions are in §4.3–§4.8, §5 and §7, dated
2026-10-09. They were accepted with the SEP on 2026-10-09 (§12, "Decisions at
acceptance"). Q1 must define an admission/error policy for content outside the
fit band; its numerical qualification can follow in Phase 1. Review completion
does not accept the default change. If qualification cannot meet the proposed
error budgets, revise the proposal before activating `band` as the default.

1. **`numpy_ref`, translation and the physics layer.** Deliverables:
   - the tap design and its refusal;
   - `integration` and the driver keyword;
   - the filtered oracle with the reviewed endpoint, exposure and work policies;
   - `PlasmaResponse` following it;
   - the problem record and output metadata;
   - tests §7 items 1–8 on `numpy_ref`, with the witness inputs pinned to `trapezoid`;
   - the manual pages and the Luna difference.

   The compiled backends refuse the `band` rule by name until their phase lands, never
   falling back silently. Verify §7 items 1–8.
2. **`numba_cpu`.** The pipelined traversal, parity with the oracle (§7 item 9) and the
   benchmark (item 10).
3. **Kokkos CPU** in `solprop-kokkos`. Parity (§7 item 9) and support for Phase 1's
   schema; the device stencil-and-scan design written down for the GPU campaign.
   Device builds without the rule refuse it by name until qualified.
4. **Records, examples and close.** Regenerate the convergence records, re-run the plasma
   worked examples, update the quoted numbers, then run the end-of-SEP gate, the slow
   lane and coverage, including the Kokkos-selected slow lane required for final
   qualification. Earlier slices use the scoped checks in `AGENTS.md`.

### 12. Open questions

- **Q1 — Design tolerance, cap and fit range.** Proposed: $\varepsilon_{\rm design} =
  10^{-5}$, $K \le 8$, fit range $[0, \theta_b]$. Is the band edge enough for the
  exposure on harder cases, such as a long pulse with many ionisation steps or a
  saturating $f$? Phase 1 extends the response-level study to such cases before fixing
  the range.
  *Author's response (2026-10-09):* the five fields of §7 item 4 include a 30 fs
  argon pulse with many ionisation steps and a saturating one (final fraction 0.63).
  The band-edge fit holds on all of them. Widening it is worse at $K = 4$ and only
  recovers at $K = 6$ (§4.3). The tolerance, the cap and the fit range are kept, and
  §4.3 now specifies the nodes, conditioning and certification. The one residual above
  $10^{-5}$ (the 30 fs collar) is product sampling, not the fit (§2).
- **Q2 — Clamp or relax (acceptance blocker; author).** Resolve the prefix-maximum
  candidate, an alternative positivity-preserving rule or a changed contract.
  Bound accumulated exposure bias and precursor population, state admission/refusal
  conditions and reconcile the policy with Q11. The corrected recurrence in §4.5
  expresses the prefix maximum; it does not endorse its physical adequacy.
  *Author's response (2026-10-09):* the prefix maximum is withdrawn. On resolved fields
  it costs accuracy (a whole-band error of $3.1\times10^{-5}$ against $1.4\times10^{-7}$
  without it), and its bias is permanent. The proposal is a changed contract instead:
  - the exposure is floored at zero, pointwise, which keeps $f \in [f_0, 1)$ and
    accumulates no bias, so the final exposure is the trapezoid's for the same sampled
    rate;
  - monotonicity holds to within a recorded undershoot $u$ ($\le 2.2\times10^{-4}$ on
    resolved fields, 0.166 on the review's spike);
  - a warning is raised above $u = 10^{-2}$;
  - precursor population is confined to the filter's support (§4.6);
  - Q11's work defect is then the pure discretisation term (§4.5).

  *Decided at acceptance (2026-10-09):* warn, at the proposed thresholds ($u > 10^{-2}$,
  $D_{\rm rel} > 10^{-4}$).
- **Q3 — The public `ionisation_fraction` helper** in
  {py:mod}`solprop.nonlinear.ionisation`. Should it follow the column, so that there is
  one definition of $f$, or stay the plain trapezoid as a diagnostic?
  *Author's recommendation:* follow the column, through the same optional
  `RunningIntegral` argument (§5), so that there is one definition of $f$. Without an
  integral it refuses `band` by name, as the response does.
- **Q4 — Interface names.** `integration=` on the response and `plasma_integration=` on
  the drivers, or one name for both? Should the drivers expose it at all, or only the
  response object?
  *Author's recommendation:* `integration=` on `PlasmaResponse`, and
  `plasma_integration=` on the drivers, which forward it. Drivers that build no
  field-grid plasma (envelope or TDSE routes) refuse an explicit value by name rather
  than ignore it.
- **Q5 — Small `ffac`.** Refuse the `band` rule when the design fails (proposed), or fall
  back to the best $K \le 8$ with a warning that states the achieved error?
  *Author's recommendation:* refuse. The refusal begins at `ffac` $\approx 3.55$
  (§4.3), below the default and below `ffac = 4`, so it affects only deliberately
  undersampled grids. A silent or warned degradation would undo the point of the rule.
- **Q6 — Endpoints (acceptance blocker; author).** Choose and derive the boundary
  rule, including nonzero residual current and secular terms. Pass-through and
  zero extension differ in production as well as in unit tests. Set a finite-window
  error budget separate from the interior multiplier tolerance.
  *Author's response (2026-10-09):* point reflection is proposed (§4.4). It is exact for
  constant and linear integrands, including the residual current at the trailing end,
  and its weights have norm 3. Band-limited prediction is exact for switched tones but
  amplifies edge content by up to $3\times10^6$, and it is rejected. No local closure is
  exact for an integrand that is non-zero at $t_a$; that holds for the trapezoid too, and
  the §4.4 table quantifies it. In production the only non-zero leading-edge integrand
  is the pre-ionised Drude term, $\propto f_0|E(t_a)|$. Trailing errors stay within $3K$
  samples, under a collar of at most $4\times10^{-15}$. The finite-window budget is
  $\varepsilon_{\rm design} + \kappa\lambda$, with $\kappa \approx 30$ for the default
  window, measured in §7 item 2. $\lambda$ is recorded as a diagnostic.
- **Q7 — Ordering with SEP 17.** Should this SEP land before SEP 17's Phase 1, so that
  the two-photon current is qualified on the band-exact column from the start?
  *Author's recommendation:* yes, if both proceed. SEP 17's Q18 compares a polarisation
  path with a direct current; doing that once, on the final column, avoids qualifying
  it twice. SEP 17's own Phase 0 review is unaffected (§4.7).
- **Q8 — Device scan.** Is the stencil-then-scan design adequate for the CUDA and HIP
  paths, or does the halo exchange inside one team need its own trial measurement?
- **Q9 — Output schema.** Should the rule and taps be stored as attributes, with the
  output schema version bumped, or only in the run's problem record?
  *Author's recommendation:* both. The problem record carries the full
  `RunningIntegral`, for resume, and the output carries the policy, $\theta_b$, $K$, the
  taps and the certified error as attributes, with an output schema bump, so that a
  saved file says how its plasma was integrated without re-translation.
- **Q10 — The legacy rule's lifetime.** Keep `integration="trapezoid"` permanently for
  Luna comparisons, or remove it once the witnesses are regenerated? Policy keeps the
  witnesses frozen, which argues for keeping it.
  *Author's recommendation:* keep it permanently, as the frozen witnesses require.
- **Q11 — Population and source-work consistency (acceptance blocker; author).**
  Derive or bound the loss-work defect caused by filtered/clamped exposure with raw
  $W$ in $J_{\rm loss}$. Include field zeros, saturation, density, the outer integral,
  collar and projection. Set physically justified tolerances and refusal conditions;
  coordinate with SEP 17 Q18. Do not replace a failed energy test with a multiplier
  test or assume a pointwise raw-rate identity proves population energy balance.
  *Author's response (2026-10-09):* derived in §4.5. With the floor and no clamp, the
  defect is $D = \int W\,[(F-I)f]\,\mathrm{d}t$: the rate-weighted second difference of
  $f$ at the filter scale, $-(\delta t^2/12)\int W f''$ to leading order. It is second
  order, as today's defect is. Measured: $\le 4.7\times10^{-8}$ on saturating and
  long-pulse fields (today's trapezoid: up to $5.3\times10^{-4}$), and
  $\le 5.2\times10^{-9}$ over every node of the ten-step runs. $D_{\rm rel}$ is computed
  exactly at run time, normalised by the largest node's ionisation, and warned on above
  $10^{-4}$. The transform-side work, with collar and projection, is a Phase 1
  assertion (§4.8, §7 item 3). A filtered or adjoint rate in the loss term was measured
  and rejected (§4.5). Local non-simultaneity within $K\,\delta t$ is stated plainly.
  Coordination with SEP 17 Q18 stands.
- **Q12 — Reproducible independent evidence (author before acceptance; Phase 1
  implementer for qualification).** Specify the causal reference, its DC/Nyquist
  and boundary treatment, convergence and uncertainty; document all rate-model
  settings, pulse/seed amplitudes and phases, widths, bands and error norms. Keep
  clamp error independent of integration error. Commit the response and ten-step
  studies before changing the default; the reported drafting measurements are
  provisional until reproduced.
  *Author's response (2026-10-09):* the reference and the protocol are specified (§4.8,
  §7 items 4–5). The reference is a Richardson extrapolation of today's column, with no
  filter, clamp or spectral antiderivative, and its uncertainty comes from a third
  refinement. The studies were repeated for the revised rule against it. They remain
  uncommitted. Committing them as convergence records is Phase 1's first deliverable,
  and the default does not change before it.
- **Q13 — Resolved integration contract (acceptance blocker; author).** Specify
  how standalone physics evaluators obtain the active band and resolved taps
  without reversing the P/T dependency, and how the public helper, mixed-channel
  settings, statistics, provenance and resume use that same policy (§5).
  *Author's response (2026-10-09):* §5 "Resolved integration contract".
  - The design lives in T, as hand-rolled numerics must.
  - P holds a `RunningIntegral` value type that the evaluators take as an argument.
    They refuse `band` without one, and refuse one designed for another `dt`.
  - Mixed policies within one problem are refused.
  - Statistics read the compiled taps.
  - Provenance records the full object, and resume compares it bit for bit.

Q3, Q4, Q5, Q7, Q9 and Q10 are maintainer decisions before Phase 1; Q3/Q4/Q9 feed
Q13's interface contract. Q8 belongs to the separately qualified device campaign.
Unresolved blockers keep the lifecycle status at `Draft` even when review status
is `Reviewed`.

The author's responses above, dated 2026-10-09, revise Part I after the review. John
Travers accepted them with the SEP on 2026-10-09, without a further review.

**Decisions at acceptance (2026-10-09).**
- **Resolved:** Q1, Q2, Q6, Q11, Q12 (the author's part) and Q13, as the responses
  above propose.
- **Q2:** a warning, not a refusal.
- **Band floor:** the $10^{-6}$ floor of §7 items 4 and 5 is accepted.
- **Q3, Q4, Q5, Q7, Q9 and Q10:** the author's recommendations are adopted.
  - Q3: the public helper follows the column.
  - Q4: `integration=` and `plasma_integration=`.
  - Q5: refuse.
  - Q7: this SEP lands before SEP 17's Phase 1.
  - Q9: problem record and output attributes, with an output schema bump.
  - Q10: the legacy rule is kept permanently.

**Still open after acceptance:**
- Q8, with the device campaign;
- the commitment of Q12's studies as convergence records, as Phase 1's first
  deliverable.

### 13. References

1. M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec, "Light
   propagation in field-ionizing media: extreme nonlinear optics," *Phys. Rev. Lett.*
   **83**, 2930 (1999).
   [doi:10.1103/PhysRevLett.83.2930](https://doi.org/10.1103/PhysRevLett.83.2930)
2. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.
   (Pearson, 2010): the frequency response of the trapezoidal integrator and of
   linear-phase FIR filters.
3. T. W. Parks and J. H. McClellan, "Chebyshev approximation for nonrecursive digital
   filters with linear phase," *IEEE Trans. Circuit Theory* **19**, 189 (1972).
   [doi:10.1109/TCT.1972.1083419](https://doi.org/10.1109/TCT.1972.1083419): the minimax
   alternative to the least-squares tap design.

## Part II — Implementation record

### Phase 1 — numpy_ref, translation and the physics layer

- **Scope:** Phase 1 of §11, 2026-10-09, implemented by Opus 5.5 (AI) for John Travers
  on `feature/sep-0018-band-integrals-1` in solprop and in solprop-kokkos.
  - **What landed.** The band rule as Part I specifies it: designed and certified in
    translation, carried as a value type in P, and executed by `numpy_ref` and the
    physics layer.
  - **Around it.** The driver keyword, the provenance and output schema, the run
    diagnostics with their warning, the §7 items 1–8 tests, three convergence records
    and the documentation.
  - **Compiled backends.** `numba_cpu` and Kokkos refuse the rule and its diagnostics
    by name.
- **Delivered:**
  - `nonlinear/ionisation.py`:
    - `RunningIntegral`: policy, read-only taps, `dt`, `band_edge`, certified error,
      boundary and exposure policies. Its methods are `trapezoid()`, `integrate`,
      `filter` (point reflection, centre first, pairs summed before they are
      weighted), `floor` (`max(S, 0)` under the band rule) and `check`.
    - `IntegrationPolicy` and `DT_RTOL`.
    - `ionisation_fraction(..., integral=)`.
  - `translate/plasma_integration.py`:
    - `design_running_integral(grid, policy)`, the only producer of a band rule,
      cached on `(dt, band_edge)`. It fits on 4097 nodes with column scaling, certifies
      on 8193 samples plus bounded refinement, refuses above a condition number of
      `1e10`, and takes the smallest $K \le 8$ meeting $10^{-5}$ in band and $1.05$
      above it.
    - `integration_record`, `trapezoid_multiplier`, and the constants of §4.3 and §4.5.
  - `nonlinear/responses.py`:
    - `PlasmaResponse.integration`, defaulting to `"trapezoid"` for this phase (see
      Deviations).
    - `evaluate` and `ionised_fraction` take `integral=`. They refuse the band rule
      without one, an integral of the other rule, and another `dt`, each by name.
  - `translate/responses.py`: `CompiledResponses.plasma_integral`, one per problem, with
    mixed rules refused naming the responses. `translate/problem.py`: kernel schema 25.
  - `backends/numpy_ref`:
    - The four plasma passes and the statistics passes apply the rule.
    - `plasma_integration_diagnostics` and the statistics `plasma_undershoot`,
      `plasma_work_defect` and `plasma_boundary_level` (catalogued in `stats.py`).
  - `backends/numba_cpu/backend.py` and solprop-kokkos: refusal by name of the band rule
    and of the three statistics. Kokkos also gets schema 25 in its build record and
    `plasma_integral` in `RESPONSE_FIELDS`.
  - The drivers:
    - `plasma_integration=` on `prop_capillary`, `prop_waveguide`, `from_capillary`,
      `from_waveguide` and `_from_hollow`, forwarded through the single-gas builder,
      `responses_for` and `ResponseSelection`.
    - It is refused with envelope plasma, with the plasma off, and on the general modal
      route.
    - The three statistics are defaulted under the band rule, and the post-run
      `PlasmaIntegrationWarning` is added (`exceptions.py`).
  - Output schema 16:
    - The configuration carries `plasma_integration` (the taps as Python floats) under
      the band rule only, and the HDF5 root attributes repeat it.
    - `serialise` leaves `integration` out of a response's record when it is
      `"trapezoid"`, so every existing resume key is unchanged.
  - Tests:
    - `tests/translate/test_plasma_integration.py` (item 1).
    - `tests/nonlinear/test_plasma_band.py` (items 2, 3, 6 and 7).
    - `tests/nonlinear/test_plasma_band_response.py` (item 4, and item 3's work
      identity).
    - `tests/driver/test_plasma_band_propagation.py` (item 5, the keyword, the hash and
      the refusals).
    - The Luna kernel comparison and the M2/M4 witness builders state
      `integration="trapezoid"` (item 8).
    - Records `plasma_band`, `plasma_band_response` and `plasma_band_ten_step`.
  - Documentation:
    - `docs/guide/media/nonlinear.md`: "How the three integrals are formed".
    - accuracy.md §1, with the records.
    - algorithms.md, architecture.md §3, kernel-contract.md (schema 25 and the leaf),
      output.md (schema 16), statistics.md and luna.md.
    - The changelogs of both repositories.

#### Decisions

- **D1 — the default stays `"trapezoid"` until Phase 3** (John Travers, 2026-10-09).
  Plasma is on by default on every field-grid capillary and waveguide run, and the
  compiled backends refuse the band rule until their phases. A Phase 1 default would
  have refused every default run on `numba_cpu` and Kokkos in the meantime. The flip
  lands at the end of Phase 3, once all three CPU backends execute the rule.
- **D2 — the diagnostics are computed by the statistics pass at save planes** (John
  Travers, 2026-10-09), not accumulated over every right-hand-side evaluation. The
  three statistics are recorded at each plane. The warning reads their largest value
  after the run, as the peak-field report does. This keeps per-node reductions out of
  the hot column on every backend.
- **D3 — the filter's summation order is explicit.** `RunningIntegral.filter` is a
  NumPy loop over the pairs in the order §6 fixes, not `numpy.convolve`. P, `numpy_ref`
  and the compiled backends therefore share one order, and the physics evaluator and
  the oracle stay bit for bit equal.
- **D4 — `ionisation_fraction` without `integral` is the trapezoid** (Q3). The public
  helper takes the column's `RunningIntegral` and then forms the column's fraction. It
  has no `integration` of its own to refuse on, so omitting the integral means the
  plain rule. The docstring says so.
- **D5 — the legacy value is left out of the resume key.** A response's
  `integration="trapezoid"` is dropped from its serialised record, following the TDSE
  `source_filter` precedent, so that no existing digest moves. The band rule's own
  record is written only under that rule.

#### Deviations from Part I

- §5, §11: the default (D1) and the diagnostics' evaluation points (D2) as above. §5
  says `numpy.convolve` (D3).
- §7 item 7: the undershoot of a sub-sample *physical* spike resolved by a refined
  reference is not yet measured. The one-sample spike (0.166, with the final exposure
  exactly one) and the ionising pulses are.
- §7 item 8: the pins are in place, but the slow M2/M4 witness comparisons were not run
  in this phase. Under the trapezoid default the pins change nothing. They are run with
  the default flip in Phase 3.

#### Verification

On the development machine, `numpy_ref`, solprop at the phase branch's head:

- **§7 item 1.** The certified ladder reproduces §4.3 exactly:
  - refused at `ffac = 3.5` (best $1.21\times10^{-5}$);
  - $K = 8, 7, 5, 4, 3, 2, 2, 1, 1$ at 3.6, 4, 5, 6, 8, 12, 24, 48 and 96;
  - $K = 0$ above about 630;
  - $4.92\times10^{-6}$ at `ffac = 6`, with the taps of §4.3 to every quoted digit.

  The certificate holds on 100003 independent points per interval.
- **Item 2.**
  - **Interior Drude susceptibility.** $|c+1|$ is $1.47\times10^{-6}$,
    $3.59\times10^{-6}$ and $9.804\times10^{-6}$ at 0.5, 1.0 and 1.0999
    $\omega_{\max}$, against the bound $9.849\times10^{-6}$. The trapezoid's is 0.045,
    0.178 and 0.213.
  - **Finite window.** $3.75\times10^{-6}$ at $\lambda = 10^{-8}$, and $\kappa = 33$
    and 30 at $10^{-6}$ and $10^{-4}$.
  - **Probes.** The ramps are $-0.02541$, $1.71\times10^{-6}$ and 0.
  - **Precursors.** They start exactly $K$ and $2K$ samples before the switch. Their
    sizes are $4.46\times10^{-2}$ and $1.51\times10^{-4}$ for the cosine,
    $8.0\times10^{-3}$ and $3.4\times10^{-5}$ for the sine.
- **Item 3.**
  - **Closed form.** $8.7\times10^{-11}$, $1.6\times10^{-11}$, $3.3\times10^{-12}$,
    falling by 5.4 and 4.8.
  - **Energy identity.** $4.39\times10^{-8}$, $1.11\times10^{-8}$,
    $2.77\times10^{-9}$, falling by 3.97 and 3.99.
  - **Work defect** on the item 4 fields: $\le 1\times10^{-6}$, asserted.
  - **Transform-side work** against $\int E(wJ + w'P)$: $\le 9\times10^{-7}$ on all
    five fields, against $1.8\times10^{-4}$ to $3.7\times10^{-3}$ for the trapezoid.
- **Item 4.**
  - Asserted bands $\le 6.5\times10^{-6}$, and the active band $\le 1.8\times10^{-6}$.
  - The trapezoid: $1.1\times10^{-2}$ to $1.6\times10^{-1}$ per band.
  - Reference uncertainty $\le 8.2\times10^{-6}$ per band and $\le 6.4\times10^{-9}$
    active.
  - The 30 fs collar, recorded and not asserted: $2.0\times10^{-4}$, at a share of
    $1.7\times10^{-8}$.
- **Item 5.**
  - Per band, $\le 4.2\times10^{-6}$ (He) and $\le 2.6\times10^{-6}$ (Ar). The active
    band is $9.1\times10^{-7}$ and $7.6\times10^{-7}$.
  - The trapezoid: 0.8–8.5 % per band, and $2.9\times10^{-3}$ and $4.3\times10^{-3}$
    active.
  - Reference uncertainty $6.3\times10^{-8}$ and $8.4\times10^{-8}$.
  - Plasma off: `ffac` 6 and 96 agree to $10^{-12}$.
  - The run's undershoot is $\le 10^{-3}$ and its work defect $\le 10^{-6}$; the 2 mm
    argon check run gave $4.4\times10^{-5}$ and $1.3\times10^{-12}$.
- **Items 6 and 7.**
  - Totals agree to a 400-term summation bound.
  - The spike: $u = 0.16622$ and the final exposure is $1 \pm 10^{-15}$.
  - $f \in [f_0, 1)$ on an ionising pulse, non-decreasing to $10^{-3}$ of its largest
    step.
- **Checks run.** Run as AGENTS.md's slice checks:
  - `ruff`, `basedpyright src`, `numpydoc`, codespell and the docstring doctests of the
    touched modules.
  - The directly affected test files: plasma, translation, numpy_ref plasma and
    mixtures, the output and resume tests, the drivers whose schema literal moved, the
    statistics.
  - `poe docs` with `-W`.
- **Two known failures.** In this worktree's environment, two numba_cpu bitwise digests
  in `tests/driver/test_coherent_capillary.py` fail on the pyFFTW wheel's FFTW 3.3.10.
  The digests need the Homebrew FFTW build. They fail identically without this change.
- **Kokkos** (`feature/sep-0018-band-integrals-1`, f48b233). Its editable CPU build
  against this branch, `ruff` and its pytest suite passed: 251 passed and 5 skipped,
  with 4 threads. From that environment, solprop's band-rule propagation tests,
  including the Kokkos refusal, also pass, as do the envelope plasma, vector Raman,
  temperature-path and mixture-channel tests that select Kokkos.
- **Not run in this phase:** the full gate, the slow lane and coverage. They are kept
  for the end of the SEP.

#### The rest of this report

- **Sources checked:** Part I §4.3–§4.8 and §7. The review's study scripts, which the
  production rule reproduces. `numpy_ref`'s passes, `PlasmaResponse`, the translation
  of plasma channels, the statistics catalogue and both backends' refusal paths.
- **Limitations:**
  - The band rule runs on `numpy_ref` only. The compiled backends refuse it.
  - The diagnostics are save-plane values (D2).
  - The sub-sample spike study (item 7) is outstanding.
- **Open questions:**
  - **Q14 — Phase 2's traversal.** §6 specifies a lagged pipeline with ring buffers.
    `numba_cpu` could instead form each node's three integrands in a per-thread scratch
    row of length `n_to` and integrate them in three passes. That is simpler and
    reuses the oracle's order directly, at `3 n_to` doubles per thread. Phase 2
    measures both against the RDW benchmark before choosing.
- **Handoff:**
  - Phase 2 implements the rule in `make_response_column` and `make_stats_column`,
    lifts both refusals, and qualifies parity (§7 item 9) and the benchmark (item 10).
  - Phase 3 does the same for Kokkos CPU, then flips the default (D1) and runs the
    witness comparisons with their pins.
  - The study scripts in the private development archive are superseded by the tests
    above.

### Phase 2 — numba_cpu

- **Scope:** Phase 2 of §11, 2026-10-09, implemented by Opus 5.5 (AI) for John Travers
  on `feature/sep-0018-band-integrals-2`, which branches from phase 1. Everything on
  this branch is in solprop; solprop-kokkos is unchanged.
  - **What landed.** The band rule and its run diagnostics on `numba_cpu`, on every
    route.
  - **Verification.** Parity with the reference (§7 item 9) and the RDW benchmark
    (item 10).
  - **Carried with it.** One correction to Phase 1's diagnostics.
- **Delivered:**
  - `backends/bodies.py`: `ring_filtered`, one filtered sample read from a ring with
    the window's ends closed by point reflection, and `ring_window`, the interior
    sample read from a contiguous window of a doubled ring. Both sum in the reference
    filter's order.
  - `backends/numba_cpu/kernels.py`:
    - `make_band_plasma_column`: the lagged traversal of §6, with per-node doubled
      rings. It skips windows of exact zeros and reuses the fraction on an unchanged
      exposure.
    - `make_band_stats_column`: the first stage, with the six diagnostic parts per
      node.
    - `band_ring_width`, `band_stats_width` and `BAND_DIAGNOSTICS`.
    - `ColumnSpec.band` and `ColumnSpec.preloaded`. The fused column, compiled
      `preloaded` and without the plasma, starts from the band column's polarisation.
  - `backends/numba_cpu/backend.py`:
    - The band column runs before the fused column (`_band_respond`).
    - Every statistics scan goes through `_plasma_stats`, and
      `_integration_diagnostics` adds the three run statistics.
    - The `band_rings` ledger row.
    - The two Phase 1 refusals are lifted.
  - `backends/numpy_ref/passes.py`: `plasma_integration_diagnostics` now returns six
    maxima, and `plasma_integration_statistics` forms the ratios (the correction
    below).
  - Tests:
    - `tests/backends/test_numba_band_kernels.py`: the kernels against the passes and
      the ring body. It is added to the coverage lane's interpreted pass.
    - `tests/backends/test_band_plasma_parity.py`: eight routes and thread invariance.
  - `benchmarks/sep18/band_cost.py` and the benchmarks.md entry.
  - Documentation: backends.md §3.12, algorithms.md's cost, kernel-contract.md,
    statistics.md and the changelog.

#### Decisions

- **D6 — the diagnostics are ratios of maxima over every node** (a correction to
  Phase 1). Phase 1 formed the undershoot and the boundary level per chunk. Made per
  node, a node the field barely ionises reported 0.035 at the helium launch plane:
  its rate is one sampled spike just above threshold, and the ratio is the filter's
  full undershoot on a rate at the level of rounding. Each diagnostic is now the ratio
  of two maxima over every sample, node and population, as the work defect already
  was. The parts combine over chunks by their maxima, so no statistic depends on
  chunking. The fix is on the Phase 1 branch too.
- **D7 — §6's pipeline, not a multipass column (Q14).** The lagged traversal needs a
  few hundred doubles per node and keeps the trapezoid's kernel untouched, so its
  bitwise digests cannot move. The multipass alternative of Q14, with whole rows per
  node and the filters as vectorisable stencils, was not built. Q15 below asks whether
  to try it.
- **D8 — exact shortcuts only.** A window of exact zeros filters to exactly +0.0, and
  an unchanged exposure gives the same fraction. Every shortcut in the band kernels
  therefore keeps the reference's bits.

#### Deviations from Part I

- **§6's single traversal.** The plasma is a lagged traversal of its own, followed by
  the fused column for Kerr and Raman. The Kerr term is not folded into the band
  column's last stage.
- **§6's two diagnostic running sums.** The diagnostics are computed by the
  statistics column at save planes (D2), not carried in the response column.

#### Verification

On the development machine at the phase branch's head, `numba_cpu` against
`numpy_ref`:

- **Kernels.** On one chunk the band column reproduces `plasma_scan` and
  `plasma_current` bit for bit:
  - at $K = 4$ and $K = 0$;
  - for one and two components;
  - for the real and the analytic families;
  - with the stash.

  The statistics column reproduces the fraction and peak rate bit for bit and the
  diagnostic parts to their sums' rounding. Two channels agree to a $64(2K+1)n_{to}\epsilon$
  bound when interpreted. These pass compiled and with `NUMBA_DISABLE_JIT`.
- **Routes (§7 item 9).** Eight cases match the reference within the equivalence
  ladder's bound with the plasma sums widened by $2K+1$: mode-averaged, analytic
  family, multimode, a helium–argon mixture, radial, Cartesian, line and hybrid
  planar. Each is two fixed steps, with two components where the route has them.
  Their electron density, peak rate and three diagnostics agree too.
- **Ten steps.** Of §7 item 5's helium and argon runs, they agree with `numpy_ref` to
  $1.6\times10^{-13}$ and $2.3\times10^{-14}$ in the exit field, with equal
  diagnostics.
- **Threads.** The parallel band column gives the serial one's bits.
- **The trapezoid's path is unchanged.** The existing numba suites pass: the kernels,
  the plasma scans and tiles, the fraction reuse, the mixtures, the cache policy, the
  bodies and the memory ledger.
- **Benchmark (§7 item 10).** Same-process medians on a busy workstation, load average
  15–22. On the RDW window the band rule costs 1.51–1.58 times the trapezoid in the
  default pools and 2.53 at one thread; the A/A spread is 0.99–1.00. The first
  traversal was 2.37; the doubled rings took it to 1.86, the exact shortcuts to 1.51.
- **Not run:** the full gate, the slow lane and coverage, kept for the end of the SEP.
  No figure in this report is a quiet-machine absolute.

#### The rest of this report

- **Sources checked:** §6 and §7 items 9 and 10. `make_response_column`,
  `make_stats_column`, the bodies and the backend's chunk, stash and statistics paths.
  Benchmarks.md's rules for a busy machine.
- **Limitations:**
  - The band rule's cost on the RDW window (above).
  - The mixture's diagnostics are maxima over its channels, not per channel.
  - Kokkos refuses the rule until Phase 3.
- **Open questions:**
  - **Q15 — the band rule's cost on `numba_cpu`.** The rule costs 1.5–1.6 times the
    trapezoid's RDW window, 2.5 times at one thread. The cost is two nine-tap filters
    per sample after the first electron, scalar, in a lagged traversal. Three options:
    - accept it as the price of the accuracy;
    - try a vectorised multipass column, rows per node and the filters as stencils
      between sequential sums;
    - try specialising the kernel on $K$ and fusing the Kerr term into the band
      column's last stage.

    Each further attempt is measured against the trapezoid as here. Read it with
    §10's alternative, a finer grid, which costs about twice the whole nonlinear term
    for a quarter of the error. The answer also bears on Phase 3's default flip: under
    the band default, every plasma run pays this cost.

    *Measured again on a quiet Xeon Gold 6240 (2026-10-09):* 2.35–2.52 times, with a
    serial response column in every pool setting, which matches the Apple-silicon
    one-thread figure. Fusing the Kerr term would save at most about 2 %.

    *Decided (John Travers, 2026-10-09):* proceed to Phase 3 with the rule as it is,
    and put the optimisation on the roadmap ("The band rule's cost on the compiled
    backends").
- **Handoff:** Phase 3 ports the band column and its statistics to Kokkos CPU, removes
  the parity test's Kokkos skip, then flips the default (D1).

### Phase 3 — Kokkos CPU and the default

- **Scope:** Phase 3 of §11, 2026-10-09, for John Travers, on
  `feature/sep-0018-band-integrals-3` in both repositories. Each branches from Phase 2's
  in its own repository; the default flip was prepared on solprop's
  `feature/sep-0018-band-integrals-3-flip` and merged in.
  - **Implementers.** Opus 5.5 (AI): the Kokkos port as a delegated worker, and the
    default flip and the integration in the main session.
  - **What landed.** The band rule and its diagnostics on Kokkos CPU builds; the band
    rule as the default; and the fixes the full default lane found, from Phases 1 and 2.
- **Delivered:**
  - **solprop-kokkos** (`c96532a`, `9e71315`):
    - **Problem fields.** `Problem.plasma_band`, `plasma_taps`, and their validation
      (1–9 finite taps; the trapezoid is one identity tap).
    - **Column and traversal.** `band_plasma_column<Family, Components, Multiple>`,
      numba's lagged design in the OpenMP traversal, with every product its own
      statement so that nothing is fused that the reference rounds separately. A
      `Preloaded` template flag runs the existing traversal without the plasma,
      starting from the band column's polarisation.
    - **Statistics.** `scan_band_statistics<Diagnose>`, and statistics flag
      `1 << 12`, which carries the six diagnostic maxima.
    - **Scratch and ledger.** Per-node ring scratch, with ledger rows
      `plasma_band_rings` and `plasma_integration_diagnostics`.
    - **Device builds.** They refuse the rule and the diagnostics by name. The
      stencil-then-scan design is recorded in `include/solprop_kokkos/plasma.hpp`;
      Q8 stays open.
    - **Docs and tests.** `tests/test_band_plasma.py` and `tests/band_thread_probe.py`;
      the README (CPU only, and schema 25 in the version contract) and the changelog.
  - **solprop:**
    - **The default.** `PlasmaResponse.integration` defaults to `"band"`.
    - **Pinned tests.** These state `integration="trapezoid"`: the trapezoid's own
      identity tests (`test_plasma.py`), the fused trapezoid column's kernel tests,
      and the comparisons with Luna and the recorded baselines (Luna mixture
      propagation, the M2 array record, the display hash literals, SEP 12's approved
      parent and the ten-step study's reference runs).
    - **Kokkos tests.** The parity test now runs Kokkos, and the refusal test is now
      a run test.
  - **Fixed from Phases 1 and 2.** The full default lane found what the slice checks
    had not:
    - SEP 12's parent harness admits output schema 16;
    - two doctests still showed kernel schema 24;
    - `solprop.sol` excludes the new advanced names;
    - the M2 historical array hashes skip the new leaf;
    - the structure figures draw `translate/plasma_integration` and the two
      diagnostic passes;
    - the benchmark script loads the Kokkos package before numba and pyFFTW.

#### Decisions

- **D9 — tests that are not about the plasma column filter the undershoot warning.**
  The default lane's cheap grids, 200 nm to 2 µm on short windows, under-resolve the
  strongest node's ionisation rate. Under the band rule they now report an undershoot
  of 0.016 to 0.148 and warn, as Q2 decided they should.
  - Five modules filter `PlasmaIntegrationWarning`, with a comment saying why: step
    limits, the wall beside a gas, optmat's rates, the flow's adaptive reuse, and
    optmat's plasma parity.
  - Tests of the plasma column itself keep the warning.
  - Q16 below asks whether this frequency is what was intended.
- **D10 — a few tolerances move under the band default, each with a measured reason.**
  - **Planar against modal peak rates.** They agree to 1.3e-8 rather than 1e-8. The
    two routes' fields differ slightly more at the plane, and the tunnelling exponent
    multiplies that tens of times. The electron densities still agree to 1e-8.
  - **The diagnostics across backends.** They are compared to an absolute $10^{-10}$
    in optmat's parity (`INTEGRATION_AGREEMENT`), as in the band parity test. They
    are ratios judged against $10^{-2}$ and $10^{-4}$, and the work defect is
    rounding-level.

#### Deviations from Part I

- **The Kokkos worker's choices.**
  - When the diagnostics are requested, Kokkos rescans the plane even where the stash
    is current.
  - Its rings and diagnostics views are allocated for every field-plasma problem on a
    CPU build, a few doubles per node.
  - A band rule with $K = 0$ matches the trapezoid to rounding, not bitwise, because
    the band path keeps every product unfused.
- **§7 item 8.** The witness comparisons' pins are now load-bearing. They and the
  convergence records with plasma on (`m2_order`, `m2_mode2_along_z`,
  `backend_equivalence`, `spatial_accuracy`) are slow-lane material, run with Phase 4's
  gate. The default lane passes with the pins.

#### Verification

On the development machine at a load of 15 to 58, solprop at the phase 3 branch's
head:

- **Kokkos parity.** Kokkos CPU against `numpy_ref` on the eight routes, two fixed
  steps: $2.4\times10^{-18}$ to $4.9\times10^{-16}$, and $4.3\times10^{-12}$ on the
  hybrid planar kind, inside bounds of $1.7\times10^{-10}$ to $1.4\times10^{-9}$. The
  diagnostics agree to $4.5\times10^{-14}$.
- **Ten-step runs.** Item 5's helium and argon runs agree to $1.2\times10^{-13}$ and
  $1.6\times10^{-14}$.
- **Threads and edges.** One thread against four gives identical digests. Short
  windows, whose two ends' reflections overlap, agree with the reference.
- **The trapezoid on Kokkos.** It is bitwise unchanged. The assembly comparison is in
  [benchmarks.md](../../benchmarks.md#band-rule-plasma-integration).
- **The default lane under the band default.** Without Kokkos: 13818 passed, 170
  skipped. Seven failed, all numba bitwise digests of runs with the plasma off, which
  fail on this environment's FFTW 3.3.10 wheel before and after.
- **The Kokkos suite.** 271 passed and 6 skipped, and then the short-window case
  passes once translated on the trapezoid.
- **The default lane with Kokkos installed.** 14677 passed and 52 skipped; seven
  failed, the same seven FFTW-wheel digests.
- **Benchmark (§7 item 10, Kokkos).** On the RDW window the band rule costs 1.26–1.32
  times the trapezoid in the default pools, 1.94 at four threads and 2.72 at one.
- **Not run:** the slow lane, coverage, the native ctest and Kokkos's `check.sh`. They
  belong to Phase 4's gate.

#### The rest of this report

- **Sources checked:** the Kokkos worker's report, its diff summary, and its tests'
  figures. Every solprop test the flip moved, file by file.
- **Limitations:**
  - Device builds refuse the rule (Q8).
  - The diagnostics are save-plane values (D2).
  - The cost (Q15) is on the roadmap.
- **Open questions:**
  - **Q16 — how often the undershoot warning fires.** Under the band default it fires
    on the test suite's cheap grids, 200 nm to 2 µm with short windows, at
    $u = 0.016$–$0.148$, on any run whose strongest node's rate rises within a
    sample. That is Q2's intended meaning, a sampling failure of the rate whatever
    the rule. But it will meet users who choose such grids for their spectra. Three
    options:
    - keep $10^{-2}$;
    - raise it;
    - word the warning around what to do: a finer `dt`, or `plasma_integration="trapezoid"`.
- **Handoff:** Phase 4.
  - Regenerate the records with plasma on.
  - Run the slow witness comparisons with their pins.
  - Re-run the plasma worked examples.
  - Run the end-of-SEP gate in both repositories.
  - Then archive.

### Phase 4 — Records, examples and close

- **Scope:** Phase 4 of §11, 2026-10-09, for John Travers, on
  `feature/sep-0018-band-integrals-3` in both repositories: the end-of-SEP gate and
  whatever it found.
  - **Implementer.** Opus 5.5 (AI), in the main session.
- **Delivered:**
  - **Coverage.** The new code is covered at 100 %. The lane gained tests:
    - the design's Nyquist and ill-conditioning refusals, and the certificate's
      refinement;
    - the value type's malformed taps and its band check;
    - the diagnostics on problems without a field plasma, on both backends;
    - the band rule refused on the envelope route, and without channels;
    - the band column pre-ionised and not, and the pre-loaded fused column.
  - **Static checks.** Type and docstring fixes in the new tests and the kernels.
  - **The records with plasma on.** `m2_order`, `m2_mode2_along_z`,
    `backend_equivalence` and `spatial_accuracy` pass unchanged in the slow lane under
    the band default. Their plasma comparisons pin the trapezoid where they compare
    with recorded or external runs (Phase 3), so nothing needed regenerating, and the
    numbers quoted on accuracy.md stand.

#### Decisions

- **D11 — the worked examples wait for John Travers's go.** Their committed output was
  made with the trapezoid. Re-running them is by hand and at full length (AGENTS.md),
  so it waits for that, and stands on the roadmap until it is done. The
  quick tutorials ran under the band default in this phase's docs build.
- **D12 — the manual's standalone plot states the trapezoid.** The plot on the nonlinear
  manual page calls `PlasmaResponse.evaluate` on bare samples, with no grid to design the
  band rule from, and the band default refuses that by name (D4). The plot now passes
  `integration="trapezoid"`, as the methods' own examples do; at its 12 as spacing the
  trapezoid's multiplier is within $10^{-4}$ of exact at the carrier.

#### Deviations from Part I

- **§11 item 4.** No record needed regenerating: see Delivered. The worked examples are
  not yet re-run (D11).

#### Verification

On the development machine at a load of about 55, solprop at this branch's head with the
Kokkos CPU wheel of the same branch installed:

- **Coverage lane.** 100.00 % of the CPU source (14720 passed, 28 skipped).
- **Default lane.** The same run. Seven numba bitwise digests failed on the PyPI pyFFTW
  wheel (FFTW 3.3.10). With pyFFTW rebuilt on the FFTW 3.3.11 they are pinned to, all
  pass. They are runs without the plasma.
- **Slow lane.** On `numba_cpu`: 137 passed and one skipped. The statistics channel's
  cost gate measured 33 % against 25 % while the coverage lane ran beside it; its case
  pins the trapezoid, and it passed in the Kokkos-selected lane below.
- **Kokkos-selected slow lane.** 139 passed.
- **Docs.** `poe docs`, warnings as errors: passes after D12's fix; the tutorials ran under the band default.
- **The Kokkos gate.** `scripts/check.sh` against this branch: passes: format, lint, types,
  clang-format, the tidy and sanitizer native suites, the Python suite (272 passed, six
  skipped) and the distribution wheel's smoke test.

#### The rest of this report

- **Sources checked:** every lane's log, and the coverage report line by line.
- **Limitations:** as Phase 3's. The device builds refuse the rule (Q8), and the cost
  (Q15) is on the roadmap.
- **Open questions:** Q16 (Phase 3) stays open for John Travers.
- **Handoff:** John Travers's decisions on Q16, the merge, and the worked examples.

## Part III — History

- 2026-10-09: Drafted, from the band-top measurements recorded in accuracy.md §1.
- 2026-10-09: Reviewed by GPT-6 (AI); John Travers confirmed his human review.
  Set review status to Reviewed; lifecycle status remains Draft. Corrected the
  prefix-maximum recurrence, limited multiplier and total-preservation claims,
  and recorded acceptance blockers for population/work consistency, clamp bias,
  endpoints and the standalone evaluator contract. Added independent-reference,
  causality, mixture/component, provenance and qualification requirements.
- 2026-10-09: Author's revision after review (accepted with the SEP, without a further
  review). Reproduced the review's probes: the pass-through ramp coefficient 0.05133,
  the prefix-maximum spike value 1.072934 and the certified multiplier
  $4.92\times10^{-6}$.
  - Proposed point-reflection endpoints and a finite-window budget (Q6), with
    band-limited prediction rejected for its conditioning.
  - Withdrew the prefix maximum in favour of a floor at zero plus a recorded
    undershoot (Q2).
  - Derived and measured the population/work defect with the raw-rate loss current,
    and rejected an adjoint survival factor (Q11).
  - Specified the tap design's nodes, conditioning and certified maxima (Q1).
  - Adopted a Richardson-extrapolated trapezoid reference with a third refinement,
    and wrote the full study protocol (Q12).
  - Proposed the T-side design with a P-side `RunningIntegral` (Q13).
  - Withdrew the draft's half-sample-acausality claim for the trapezoid and quantified
    the band rule's precursor.
  - Added author's recommendations for Q3, Q4, Q5, Q7, Q9 and Q10.
  - Measured on the revised rule: response-level error $\le 6.5\times10^{-6}$ in
    asserted bands, ten-step plasma-induced change $\le 4.2\times10^{-6}$, against
    1–16 % for today's column.
- 2026-10-09: Accepted by John Travers, with the decisions recorded at the end of §12
  (Q2: warn; band floor $10^{-6}$; Q3, Q4, Q5, Q7, Q9 and Q10 as recommended).
  Part I is frozen.
- 2026-10-09: Phase 1 implemented on `feature/sep-0018-band-integrals-1`: the band rule
  on `numpy_ref` and in the physics layer, kernel schema 25, output schema 16, the
  driver keyword, the run diagnostics and three convergence records (Part II).
- 2026-10-09: Phase 2 implemented on `feature/sep-0018-band-integrals-2`: the band
  rule and its diagnostics on `numba_cpu`, parity on eight routes, and the RDW
  benchmark (1.5–1.6 times the trapezoid's window, Q15). The diagnostics are now
  ratios of maxima over every node (D6).
- 2026-10-09: Phase 3 implemented on `feature/sep-0018-band-integrals-3` (solprop and
  solprop-kokkos): the band rule and its diagnostics on Kokkos CPU builds, and the
  band rule as the default; the default lane's fixes from Phases 1 and 2.
- 2026-10-09: Phase 4 on `feature/sep-0018-band-integrals-3`: the end-of-SEP gate passed
  in both repositories (coverage 100 %, the slow lanes, the docs and the Kokkos gate);
  no record needed regenerating. The worked examples and Q16 wait for John Travers.
- 2026-10-09: Merged to main in both repositories. **Final**, approved by John Travers,
  and archived. Q16 and the worked examples' re-run stay on the roadmap, with Q8 and the
  cost (Q15).

### AI review evidence and disposition (2026-10-09)

The FIR correction is a plausible way to reduce the interior trapezoidal multiplier
error. It is not yet a complete qualified replacement for the causal plasma column.
The revised questions above preserve the physics decisions for the author and
maintainers; the review does not recommend acceptance until the blockers are resolved.

Checked against solprop's conventions, the NumPy passes, `PlasmaResponse`, the public
fraction helper, the fused Numba column, translation schema 24, the existing Drude
and energy tests, SEP 17 Q18, and solprop-kokkos's `plasma.hpp` scan primitive.
No backend implementation was changed or qualified by this documentation review.

Independent NumPy/SciPy probes used 4097 uniformly spaced least-squares nodes on
$[0,2.2\pi/6]$, $K=4$, $c_0=1-2\sum_{k=1}^4c_k$, centre-first paired filtering
with pass-through endpoints, and trapezoidal prefixes starting at zero:

- The sampled multiplier error was $4.925\times10^{-6}$, consistent with the draft's
  rounded value. This probe is not a continuum maximum certificate.
- On 65 samples with only $W_{32}\delta t=1$ nonzero, the final raw prefix was 1,
  its running maximum 1.072934 and the sum of clipped increments 1.202585.
- On 512 samples of $\cos(n\pi/3)$ with dimensionless $\delta t=1$, the proposed
  double integral with pass-through endpoints had a startup-induced interior ramp
  coefficient about 0.05133. Its whole-trace peak error relative to
  $(1-\cos(n\pi/3))/(\pi/3)^2$ was about 14.19 times the analytic peak, despite
  an accurate interior oscillatory coefficient. This separates the multiplier
  claim from the unqualified whole-window Drude claim.

The trapezoid multiplier, prefix-maximum distinction and finite-window integration
identities were re-derived in the review. The bibliographic identities and abstracts
of references 1 and 3 were checked on their APS and IEEE publisher pages; their full
texts and the Oppenheim/Schafer textbook were not checked. No physical constants or
rate coefficients were transcribed. The reported helium/argon response errors,
ten-step errors and performance estimates remain unverified drafting evidence,
with reproduction assigned to Q12 and the implementation phases.
