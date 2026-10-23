# Radial Hankel transform

A radial grid describes a beam whose Cartesian field components depend only on
radius. This reduces a two-dimensional cross-section to one radial coordinate,
which is useful for a centred, circular beam. It cannot represent an off-axis
beam, azimuthal breakup or multiple filaments. Start with the
[radial filament lesson](../../introduction/lessons/plot_03_filament.rst) and
[beam setup](beams.md) before using the transform directly. The energy, fluence and
intensity of a radial field are measured in the medium at the plane, with the axial
admittance of each Hankel column; a radial run through a pressure gradient also carries
the forward amplitude term and its certificate ([energy and flux in a medium](flux.md)).

The quasi-discrete Hankel transform {py:class}`~solprop.hankel.QDHT` connects the
radial field to its transverse spatial frequencies. Choose both the outer radius
and the sample count: a larger domain and finer sampling address different
errors. Its finite transform is not exactly orthogonal. The derivations below
separate that truncation error from floating-point error and from the finite
window needed to represent a physical beam.

The order-zero transform uses angular spatial frequency, with no Fourier factor in
the transform itself:

$$
 H(k)=\int_0^\infty r f(r)J_0(kr)\,dr,\qquad
 f(r)=\int_0^\infty k H(k)J_0(kr)\,dk.
$$

Let $\alpha_i$ be the positive zeros of $J_0$, $S=\alpha_{N+1}$,
$d_i=|J_1(\alpha_i)|$, $r_i=R\alpha_i/S$ and $k_i=\alpha_i/R$.
The raw quasi-discrete Hankel transform (QDHT) is

$$
 T_{ij}=\frac{2J_0(\alpha_i\alpha_j/S)}{S d_i d_j},\qquad
 F=\frac{R^2}{S}\operatorname{diag}(d)T\operatorname{diag}(d^{-1}),\qquad
 B=\frac{S}{R^2}\operatorname{diag}(d)T\operatorname{diag}(d^{-1}).
$$

`forward` applies $F$ and `inverse` applies $B$. A physical field in V/m therefore
has a radial spectrum in V m. Both operations act on the last array axis. The
integration weights, including the full azimuthal integral, are
$w_{r,i}=4\pi R^2/(S^2d_i^2)$ and $w_{k,i}=4\pi/(R^2d_i^2)$.
They give the continuous Parseval relation
$\int 2\pi r|f|^2dr=\int 2\pi k|H|^2dk$ its discrete counterpart.
No orthogonalisation or numerical matrix inversion changes the paper's transform.
See [Guizar-Sicairos and Gutiérrez-Vega (2004)](https://doi.org/10.1364/JOSAA.21.000053).

## Finite-transform error

The finite matrix is **not exactly orthogonal**, although it is often described as
such. At $N=256$, $\max|T^2-I|=3.0619\times10^{-11}$, far above
$N\epsilon=5.6843\times10^{-14}$. Round trips and discrete Parseval consequently carry
a truncation term in addition to floating-point error.

An independent expression for that term follows by closing an upper-right contour
around the positive Bessel zeros. Put $a_i=\alpha_i/S$ and

$$
 g_{ij}(z)=zJ_0(a_i z)J_0(a_j z)\frac{H_0^{(1)}(z)}{J_0(z)}.
$$

The Wronskian gives residue
$-2iJ_0(a_i\alpha_l)J_0(a_j\alpha_l)/(\pi d_l^2)$ at $\alpha_l$.
Indent above those poles. On the imaginary axis the contour integral is purely
imaginary; its real-axis part is the Lommel integral
$\delta_{ij}S^2d_i^2/2$. The horizontal part vanishes because $a_i+a_j<2$.
The endpoint at $S$ is removable: both numerator Bessel functions vanish there.
Taking real parts gives

$$
 (T^2-I)_{ij}=-\frac{2}{S^2d_i d_j}
                 \operatorname{Im}\int_0^\infty g_{ij}(S+iy)\,dy.
$$

In particular, the following uniform entrywise truncation bound is independent of
any matrix-product measurement:

$$
 b_N=\int_0^\infty
 \max_{i,j}\left|\operatorname{Im}
   \frac{2g_{ij}(S+iy)}{S^2d_i d_j}\right|dy,
 \qquad |(T^2-I)_{ij}|\le b_N.
$$

This is a finite-contour application of the Bessel-zero summation method described
by [Saharian](https://arxiv.org/abs/0708.1187), derived here for these endpoints.
The test suite evaluates this scalar integral using scaled Bessel functions, before it
forms `T @ T`. Its adaptive quadrature has a predeclared
absolute error budget of $32\epsilon$; the reported quadrature error must fit this
budget and is added to the result. The solver is asked for one eighth of that budget
with no relative relaxation. Like the other quadrature-based checks in the test suite,
this is a numerical quadrature error estimate, not an interval-arithmetic enclosure.

Matrix multiplication then adds $\gamma_N |T||T|$, with
$\gamma_n=n\epsilon/(1-n\epsilon)$. Bounds for a stored matrix additionally account
for construction error in its Bessel arguments and scalings. These terms must be
declared separately from $b_N$; they are not replacements for truncation.

For any weighted field vector $u_i=f_i/d_i$, the exact-arithmetic round-trip bound
is $|(T^2-I)u|_i\le b_N\|u\|_1$. Its weighted Parseval defect is at most
$b_N\|u\|_1^2$, since $T$ is symmetric. These bounds do not assume the field
vanishes at the outer sample or suppress the highest transverse frequencies.

## Gaussian reference bound

For $f(r)=\exp(-r^2/w^2)$, the infinite transform is
$H(k)=w^2\exp(-k^2w^2/4)/2$. A bound on the discrete transform can be derived
without measuring its Gaussian error. First subtract $e^{-R^2/w^2}$ so the finite
interval function is zero at $R$, and expand it in its continuous Fourier–Bessel
series. With $L=(1/r)\partial_r r\partial_r$, repeated integration by parts gives
for its coefficient $c_j$, for any integer $p\ge2$,

$$
 |c_j|\le \sum_{q=1}^{p-1}
   2\sqrt\pi R^{2q}|L^q f(R)|\alpha_j^{-2q-1/2}
 +2\pi R^{2p-2}C_p\alpha_j^{1-2p},
 \quad C_p=\frac{w^2}{2}\left(\frac4{w^2}\right)^p p!.
$$

Here $L^pf=(-4/w^2)^pp!L_p(r^2/w^2)e^{-r^2/w^2}$; the integral of its
absolute value is bounded by $C_p$ using Laguerre orthogonality and
Cauchy–Schwarz. The conservative Bessel-zero estimates
$d_j^{-2}\le\pi\alpha_j$ and $\alpha_j>\pi(j-1/4)$ imply

$$
 \sum_{j>N}\alpha_j^{-s}\le
 \frac{\pi^{-s}(N-1/4)^{1-s}}{s-1},\qquad s>1.
$$

The first estimate follows from $J_1(\alpha)Y_0(\alpha)=2/(\pi\alpha)$ and
the real-argument Hankel expansion with its first omitted-term bounds; for
$\alpha\ge\alpha_1$, $|Y_0(\alpha)|\le2/\sqrt{\pi\alpha}$.
See [DLMF, real-argument remainder bounds](https://dlmf.nist.gov/10.17.iii)
and [Bessel zeros](https://dlmf.nist.gov/10.21).

Sum the coefficient bound to obtain $\eta_N$. The sample residual is at most
$\eta_N+e^{-R^2/w^2}$. The difference between the continuous Dirichlet transform
and $H(k_i)$ is bounded by
$\delta_i=e^{-R^2/w^2}(w^2/2+R^2d_i/\alpha_i)$.
Writing the finite expansion through $T^2$ then bounds the forward error by

$$
 |(Ff)_i-H(k_i)|\le
 d_i b_N\sum_j\frac{|H(k_j)|+\delta_j}{d_j}
 +\left(\sum_j|F_{ij}|\right)(\eta_N+e^{-R^2/w^2})+\delta_i.
$$

The test chooses $p=8$ before measuring. Roundoff in application and construction
is added separately. This bound exposes both finite-radius and finite-bandwidth
error; a Gaussian that does not fit either window cannot pass by using machine
epsilon alone.

For the complex Gaussian needed after diffraction, set $f(r)=e^{-ar^2}$ with
$\Re a>0$. Expanding $L_p(ar^2)$ into monomials and integrating each absolute
monomial against $e^{-\Re(a)r^2}$ gives

$$\int_0^R r|L^pf(r)|\,dr\le
\frac{(4|a|)^p p!}{2\Re a}\left(1+\frac{|a|}{\Re a}\right)^p.$$

Replacing $C_p$ by this expression proves the same finite-series error bound;
the boundary terms use $|L^qf(R)|$ directly. At the axis, the omitted Fourier–Bessel
coefficients sum without a Bessel factor because $J_0(0)=1$. A separate on-axis check
sums their coefficient bounds and the finite-disk correction, then adds
the rounding budget for the positive analytic spectrum.

Exact longitudinal diffraction differs from the paraxial Gaussian. For transverse
frequency $q\le k/2$, Taylor's theorem gives
$|k_z-k+q^2/(2k)|\le q^4/[8k^3(3/4)^{3/2}]$. The phase difference at distance z
is at most the smaller of z times this quantity and 2. For $q>k/2$, two
non-growing propagators differ by at most 2. Integrating these bounds against the
positive Gaussian spectrum gives the additional field error before propagation
is measured. Widths propagate the field error through intensity and moment ratios;
interpolated widths also include the Gaussian curvature error of linear interpolation.
The Gouy-phase check converts the on-axis field error into a phase bound,
`arcsin(axis_error/abs(axis_amplitude))`.
