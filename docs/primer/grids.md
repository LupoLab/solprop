(primer-grids)=
# 11. Putting the field on a grid

The second half of this primer turns from the equation to its solution. Before any
stepping in $z$, the field has to be represented on a computer: sampled in time, with a
finite window and a finite bandwidth, transformed with the FFT, and multiplied pointwise
to form a nonlinear response. Every choice in that representation can produce an error
that no step-size control will ever see, because it is the same at every step. This
chapter explains solprop's two grids, how they are sized, why the nonlinear response is
evaluated on an oversampled grid, what the active band and the collars are, and how
energies and spectra are computed exactly.

## 11.1 Two grids, one protocol

solprop has two time grids, matching the two descriptions of a pulse in §1.6
([choose a field or an envelope](../guide/setup/grids.md#choose-a-field-or-an-envelope)):

- A **field grid** carries the real field $E(t)$. Its spectrum is stored for the non-negative
  frequencies only, $\omega_k=k\,\delta\omega$ from zero to the Nyquist frequency, with
  $n_\omega=n_t/2+1$ bins, as a real FFT produces it.
- An **envelope grid** carries the complex envelope about a carrier $\omega_0$. Its spectrum covers
  $\omega=\omega_0+\nu$ on both sides of the carrier, $n_\omega=n_t$ bins in FFT order.

Both share one energy functional, so a narrowband problem run on either grid gives the same
answer, and the test suite compares them. The field grid is needed when the carrier matters:
harmonics, very broad spectra, and instantaneous responses such as tunnelling ionisation. The envelope
grid is cheaper for narrowband problems and is the natural grid for the GNLSE. Neither the data
type nor the array length identifies which layout an array has, because an unchirped envelope can be
real, so every function that needs to know is told explicitly.

## 11.2 Window and spacing

Two requirements size a grid, and they are independent. The time window $T=n_t\,\delta t$ sets the
spectral resolution, $\delta\omega=2\pi/T$. The time spacing $\delta t$ sets the spectral reach, up to
the Nyquist frequency $\pi/\delta t$. A grid is therefore requested by a wavelength range and a time
range ([choose the window and wavelength range](../guide/setup/grids.md#choose-the-window-and-wavelength-range)).
The wavelength range sets the highest frequency the run must represent, and the time range sets how far
the pulse may walk or spread. Sizes are even and have no prime factor above 5. By default
(`fft_size="auto"`) a model of the transform's cost picks the cheapest such size between the one the
request needs and the next power of two; `"pow2"` and `"smooth"` ask for the next power of two or the
next such size instead. Time axes are centred, $t_n=(n-n_t/2)\,\delta t$.

The window has to contain more than the pulse. It must hold whatever the pulse sheds and whatever
responds after it: dispersive waves that walk off, a Raman coherence that rings for picoseconds, a
trailing plasma. Light that reaches the edge of a periodic window re-enters on the other side, unless
something absorbs it first (§11.5).

## 11.3 Coarse and fine: why the response is oversampled

A nonlinear response creates new frequencies. Cubing a real field whose spectrum reaches $\omega_{\max}$
produces content up to $3\omega_{\max}$. On a grid whose Nyquist frequency is lower than that, the excess
folds back to $2\Omega_N-\omega$ and lands at a wrong frequency, possibly inside the band of interest.
This is aliasing, and it was recognised at the very start of split-step simulations. Fisher and Bischel
(1975, §II.B) record that "Tappert has pointed out that such an algorithm is susceptible to an 'aliasing
instability' if the window chosen for spectral analysis of the pulse is insufficiently wide."

solprop therefore keeps two grids for each run ([coarse and fine
sampling](../guide/setup/grids.md#coarse-and-fine-sampling)). The **coarse grid** carries the propagated
spectrum. The **fine grid** is where the nonlinear response is evaluated. The fine spacing is
$\delta t_{\rm fine}=2\pi/(\mathtt{ffac}\,\omega_{\max})$, so its Nyquist frequency is
$\mathtt{ffac}\,\omega_{\max}/2$ on a field grid (on an envelope grid read $\Omega$, the relative half-span about
the carrier, for $\omega_{\max}$):

- **`ffac=6`**, the field-grid default, puts the fine Nyquist frequency at $3\omega_{\max}$. The whole third
  harmonic of the requested band is represented, and anything beyond folds to frequencies far above the
  propagated band, where the crop discards it.
- **`ffac=4`** suffices for a response without harmonic content, such as the cycle-averaged
  $\tfrac34|E_a|^2E$, whose spectrum stays near the fundamental. A short calculation shows that it would
  not suffice for the full cubic: the third harmonic of the collar, at $3.3\omega_{\max}$, would fold back
  to $0.7\omega_{\max}$, inside the band.
- **`ffac=2`** is the minimum. On a field grid it truncates the upper collar. On an envelope grid `ffac`
  multiplies the relative half-span $\Omega=\max(|1.1\omega_{\max}-\omega_0|,|\omega_0-\omega_{\min}/2|)$ instead of
  $\omega_{\max}$, so the envelope default, `ffac=2` without a harmonic term, puts the Nyquist offset at exactly
  $\Omega$: the active band and its collars are represented, and the coarse and fine grids coincide.

On that last point the protection is different. On an envelope grid with `ffac=2`, the cubic $|A|^2A$ is
evaluated without oversampling. The products of components near the edges of the window can alias, and
only the requirement that the spectrum and its collars stay well inside the window keeps them small. That
is the classic situation Tappert warned about. Check it in an envelope run whose spectrum broadens towards
the edges by enlarging the wavelength range, not by tightening the tolerance.

`ffac` also affects the accuracy of the time integrals in the plasma model (Chapter 10), because the
trapezoid's error depends on $\omega\,\delta t_{\rm fine}$. Changing `ffac` changes the whole grid: the fine
spacing, the rounded size, the realised window and therefore the frequency spacing. Two values give two
different grids that both satisfy the request, so compare them as physical spectra on their common band,
never array to array.

## 11.4 Pad, evaluate, crop

The coarse size is chosen as a divisor of the fine size, so that every coarse time sample coincides with a
fine one: with $r=n_{to}/n_t$, $t_n=t^{\rm fine}_{nr}$ ([matching the two time
axes](../guide/setup/grids.md#matching-the-two-time-axes)). Each evaluation of the nonlinear term then pads
the coarse spectrum with zeros onto the fine grid, transforms to time, evaluates the response there, transforms
back and crops to the coarse band ([oversampling: pad, evaluate,
crop](../guide/setup/grids.md#oversampling-pad-evaluate-crop)). Padding scales the spectrum by $n_{to}/n_t$ and
cropping by the reciprocal, which compensates for the different $1/N$ factors of the inverse transforms. What
the crop discards is exactly the aliased content the fine grid existed to keep out of the coarse one.

In a multimode or spatial run, this is done for every mode or transverse column at once, with batched
transforms. It is also where the synthesis and projection of Chapter 5 sit: synthesise the field at each node
on the fine grid, evaluate, project, crop.

## 11.5 The active band and the collars

A grid has three spectral regions ([the active band and absorbing
collars](../guide/setup/grids.md#the-active-band-and-absorbing-collars)):

1. **the requested band**, $[\omega_{\min},\omega_{\max}]$, set from your wavelength range;
2. **the active band**, $\omega_{\min}/2<\omega<1.1\,\omega_{\max}$, where the linear operator and the nonlinear
   prefactor are evaluated, noise is added, and the error norm is scored;
3. **everything outside**, where the field is held at zero.

Between the requested and the active band lie the **spectral collars**. These are Planck tapers that rise over
$[\omega_{\min}/2,\omega_{\min}]$ and fall over $[\omega_{\max},1.1\,\omega_{\max}]$, the smooth window of McKechan,
Robinson and Sathyaprakash (2010, Eq. 7). Light that spreads into a collar is absorbed smoothly instead of
reaching a hard edge. In time, a **temporal collar** does the same beyond the edges of the window you ask
for. The grid widens the realised window to at least $T_{\rm req}/(1-2f)$, where $T_{\rm req}$ is the
requested time range and `collar_fraction` $f=0.1$ by default, so the whole request stays flat and each
collar takes at least a tenth of the realised window outside it. solprop refuses a collar narrower than four samples. A collar only one sample wide, which Luna produces
when the realised window equals the requested one, is no absorber at all.

The collars are not masks applied between steps. They are absorption rates added to the equation, for the reason
explained in §12.7. An absorber that is doing its job is silent, so every run also watches the edges of its
time window ([watching the window's edges](../guide/numerics/boundaries.md#watching-the-windows-edges)).
From the launch on, it measures the share of the energy in a guard, the outer 5 % of the realised window at
each edge, judged above the shot-noise floor. A guard share whose square root exceeds a field-error budget
of $10^{-3}$ warns, and one above ten budgets stops the run. At the end the monitor reports the fraction of the
energy the collar removed, which warns by the same budget. `warn_walkoff`, given the group-delay span of the band you care about, warns when the temporal collar is too weak to stop a packet before it wraps, and on a spatial grid the energy fraction in the transverse collar is a run statistic. If significant light reaches a collar, the run was absorbing light that the physics may need, and the window or band should be enlarged.

## 11.6 Signs, phases and energies

The internal spectra are raw forward DFTs, with no $1/N$ and no $\delta t$, and their phase origin is the first
sample. Converting to the optics convention at the interface conjugates the spectrum and moves the phase origin to
the window centre, $\tilde E_{{\rm optics},k}=(-1)^k\operatorname{conj}(\tilde E_{{\rm internal},k})$ (Chapter 1). A
user-facing spectral phase is then referenced to $t=0$ and does not depend on the window length. For an unchirped,
transform-limited pulse centred at $t=0$, the spectral phase at the carrier is its carrier-envelope phase; for a
chirped or shaped pulse the two differ in general.

Energies are computed exactly ([energy and spectral
density](../guide/setup/grids.md#energy-and-spectral-density)). For true-power amplitudes $B_k$ on a field grid and
$A_k$ on an envelope grid,

$$
U_{\rm field}=\frac{\delta t}{n_t}\Big[|B_0|^2+4\sum_{k=1}^{n_t/2-1}|B_k|^2+|B_{n_t/2}|^2\Big],\qquad
U_{\rm env}=\frac{\delta t}{n_t}\sum_k|A_k|^2 .
$$

These are discrete Parseval identities. The time and frequency energies agree to round-off, not to a quadrature
order. The factor of 4 is the analytic-signal doubling of §1.6, and the end bins are counted once because the
analytic signal keeps them undoubled. The spectral energy densities are
$S_{\rm field}=(2\delta t^2/\pi)|B|^2$ and $S_{\rm env}=(\delta t^2/2\pi)|A|^2$ in J/(rad/s). A density per
wavelength needs the Jacobian $2\pi c/\lambda^2$. Luna weights its energies with Simpson's rule, whose fourth-order
error makes its two energies disagree slightly with each other. solprop's trapezoid is exact on a periodic,
band-limited grid.

:::{admonition} What to remember
:class: tip

- A field grid stores the real field's one-sided spectrum, and an envelope grid stores the full spectrum about the
  carrier. They share one energy functional.
- The window sets $\delta\omega$, the spacing sets the reach. The window must contain everything the pulse sheds and
  every response that rings after it.
- The response is evaluated on a fine grid with Nyquist frequency $\mathtt{ffac}\,\omega_{\max}/2$ and cropped back.
  `ffac=6` represents the third harmonic, and 4 is enough without it. The envelope default of 2 does not oversample,
  so keep its spectrum away from the edges.
- The active band is $\omega_{\min}/2<\omega<1.1\,\omega_{\max}$. Outside it the operator and the prefactor are zero.
  Planck collars absorb smoothly at both spectral edges and both temporal edges; the temporal collars lie
  outside the requested window, and a monitor warns when light reaches the window's edges.
- Parseval is exact. On a field grid, interior bins count four times.
:::

The precise statements are in [grids and Fourier conventions](../guide/setup/grids.md) and in [the
algorithms](../dev/algorithms.md#5-fourier-conventions-and-grids). [Chapter 12](stepping.md) marches the field along
$z$.
