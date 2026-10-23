# Evaluating a Raman response

A delayed response needs the field history over its memory time. The temporal window, response sampling and causal convolution each affect the result. This chapter derives the field/envelope drive, gain sign and numerical convolution used with the [Raman models](raman-models.md).

## How the convolution is evaluated

The numerical response depends on which quadratic form of the field drives it, why the transform is taken on a
doubled grid, which sign of the kernel spectrum means gain, and how the kernel is
sampled and updated along a pressure gradient.

## The drive, and why it differs between the grids

The coherence is driven by the intensity, and which real quadratic form that is depends
on the grid, exactly as the Kerr cubic does. {func}`solprop.nonlinear.raman.raman_drive`
implements

| grid | `thg` | drive |
|---|---|---|
| field | `True` | $E^2$ |
| field | `False` | $\tfrac12\lvert E_a\rvert^2$ |
| envelope | — | $\tfrac12\lvert A\rvert^2$ |

The factor of one half is the cycle average of $\cos^2$: $E^2$ and
$\tfrac12\lvert E_a\rvert^2$ agree in their slowly varying part and differ only by the
$2\omega_0$ beat. When that beat lies well above the Raman lines, the response suppresses it
strongly. For a 1030 nm pulse in nitrogen the two
field-grid forms give polarisations differing by 2.4 × 10⁻⁴. The `thg` switch therefore
matters far less for Raman than for Kerr, for that narrowband optical example. At longer wavelengths or with broad mixing
bands, compare the beat frequency with the actual Raman spectrum instead.

On both grids the drive is **real**, and so is $h$, so the convolution needs only a
real transform pair, on an envelope grid as on a field grid.

## The doubled grid

The convolution is evaluated as a product of transforms, and a transform product is a
*circular* convolution: without padding, the tail of the response wraps back onto the
head of the drive. Writing the response and the drive into the first half of an array of
twice the length makes the product the linear convolution exactly.

This is not a refinement. Hydrogen's rotational coherence at one bar lives for about
2.9 ns against a time window of tens of picoseconds; the wrapped tail would be almost
the whole response. It costs one real transform of length $2n_{to}$ in place of one of
length $n_{to}$.

The sample spacing that turns the discrete sum into the integral is folded into the
kernel spectrum by {func}`solprop.translate.raman.kernel_spectrum`, so a backend
computes

$$
\tilde P = \tilde H \cdot \mathrm{rfft}(\text{drive padded}),
$$

and the inverse transform's $1/N$ supplies the rest. Folding $\delta t$ into the kernel
applies it once per density change, rather than once per column on the other side of the
product.

## The sign that means gain

In the internal FFT convention ([conventions](../../dev/conventions.md), §2) a Stokes component of an envelope
sits at $A \sim \mathrm{e}^{-\mathrm{i}\Omega t}$. The drive term at that offset is
$\tfrac12 A_p^{*}A_s$, the convolution multiplies it by $\tilde H(-\Omega) =
\tilde H(\Omega)^{*}$, and with
$\partial_z A_s = -\mathrm{i}\,\omega_s/(2\varepsilon_0 c n)\,P_s$ the growth rate is
proportional to $-\mathrm{Im}\,\tilde H(\Omega)$. So

$$
\boxed{\ \mathrm{Im}\,\tilde H(\Omega) < 0 \ \text{is gain on the Stokes side.}\ }
$$

Analytically $\tilde H(\Omega) \to -\mathrm{i}K\tau_2/2$ at resonance for a positive line
weight $K$, which is a minimum of the imaginary part. The test suite asserts it at the vibrational
line of nitrogen and at every strong rotational line of hydrogen.

The kernel spectrum is an internal-convention array and must never be passed through
{func}`solprop.fourier.to_optics`: that conjugation would silently exchange the Stokes
and anti-Stokes sides, turning gain into loss without any error.

## Sampling the kernel, and the density it depends on

{mod}`solprop.translate.raman` turns a response into the arrays a kernel consumes.

**The tail is tapered by the grid's own collar.** A coherence that outlives the window
has to be cut somewhere, and cutting it with a step puts a sinc ringing across the whole
kernel spectrum. {func}`solprop.translate.raman.kernel_taper` gives the falling edge the
width of the grid's temporal [collar](../setup/grids.md#the-active-band-and-absorbing-collars), so the kernel is cut with the same
profile, over the same fraction of the window, as everything else the grid absorbs: one
statement of how much of the window is boundary, not a second one for Raman. A
coherence that outlives the window is noted in the debug log, once per density, naming
the dephasing time and the span that would hold it.

The padding prevents circular wraparound but does not remove this taper's bias: a
delayed probe samples an attenuated kernel if its lag behind the pump falls in the
collar. Extend the window until the pump–probe lags that matter are untapered, and
check convergence in the window separately from convergence in the time step. A
long-lived coherence need not decay inside a short-pulse window when only shorter,
untapered lags contribute to the observable. The size of the bias depends on the medium:
for fused silica on a 9.1 ps window, whose response has decayed long before the taper,
the tapered polarisation equals the untapered convolution bit for bit; for hydrogen on a
228 fs field window, whose rotational coherence outlives it by four orders of magnitude,
it differs by 12 % at its peak.

**The dephasing is applied per term, per density.** The compiled response keeps
separate undamped rows because the dephasing depends on the density, which varies along
a pressure gradient, and a molecular gas has *two* dephasing times: at one bar hydrogen's rotational and vibrational coherences decay
3.5 times apart, and nitrogen's 17 times apart. Summing the models first and damping the
sum once is wrong for every gas with both a rotation and a vibration.

## Backends, and the drive form

Raman responses run on the supported NumPy, Numba and native routes. Check
[backend support](../performance/backends.md) for the geometry and execution space
of your problem. Backend agreement measures numerical implementation of the
same response, not agreement with a material outside its validated range.

On a field grid the Kerr and Raman responses must agree about `thg`; a Raman
response without a Kerr response is refused on that route. On an envelope grid
the Raman drive is $\tfrac12|A|^2$, independent of `thg`, and an explicitly
constructed Raman response can stand alone. Convenience constructors may impose
additional joint-response choices described in the [Raman workflow](raman.md).

Two physical Cartesian components require a tensor-capable response. A
scalar-only declaration or scalar silica model is refused for vector use. See
[vector Raman](raman-vector.md) for the contraction and material assumptions.

The [kernel contract](../../dev/kernel-contract.md#5-responses-the-compiled-nonlinear-term) states the kernel
layout and backend mechanics. The [qualification record](../../dev/accuracy.md#11-capillary-modes-raman-and-the-moving-basis)
retains the gain, soliton-shift and conservation checks and their measured errors.
For a new run, vary the time spacing, window and longitudinal tolerance separately;
a previous comparison does not bound the error of a different pulse or material.
