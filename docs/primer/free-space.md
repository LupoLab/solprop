(primer-free-space)=
# 6. Beams in free space

Without a waveguide, nothing holds the beam together except its own nonlinearity, and
the transverse dimensions become part of the problem. A beam diffracts, self-focuses,
collapses towards a point, is arrested by the plasma it creates, and may break up into
many filaments. The propagation equation is still the UPPE of Chapter 2. The difference is
that the transverse label is now a continuous wavevector, so the representation of the
transverse plane becomes a numerical method in its own right.

This chapter covers solprop's free-space routes. You will see why solprop keeps the exact
longitudinal wavenumber instead of the paraxial one, and how a radial beam is represented
with a Hankel transform, together with what that costs and how accurate it is. You will see
what Cartesian and line grids add, and what the critical power and the Marburger formula
really say. Most importantly, the chapter is clear about what a calculation that assumes
radial symmetry cannot predict.

## 6.1 The exact longitudinal wavenumber

For a homogeneous medium, the transverse Fourier transform turns the wave equation into one
ordinary differential equation per frequency and transverse wavevector (§1.5). The UPPE then
propagates each column with

$$
k_z(\omega,k_\perp)=\sqrt{\frac{\omega^2n^2(\omega)}{c^2}-k_\perp^2}
$$

(Kolesik and Moloney 2004, Eq. 22; Couairon and Mysyrowicz 2007, Eq. 72; Bergé et al. 2007,
Eq. 18). Couairon and Mysyrowicz note that the main advantage of this formulation over the
nonlinear envelope equation is that "it fully accounts for nonparaxiality and captures the
chromatic dispersion of the medium over an arbitrary wide frequency range". The paraxial
equations used throughout the filamentation literature expand the square root,
$k_z\approx k-k_\perp^2/2k$ (Bergé et al. 2007, Eq. 15). They hold while $k_\perp^2/k^2\ll1$
(their Eq. 9), which is a statement about the angular spectrum of the beam, not about its
duration.

solprop uses the exact root for every column, with no paraxial approximation
([exact longitudinal operator](../guide/free-space/model.md#exact-longitudinal-operator)). Write
$k_z=\kappa+i\eta$, with the principal square root allowing a complex index. In the optics
sign of this primer, the linear operator of a column is $i(\kappa-\beta_1\omega)-\eta$ minus
any absorber rate, with $\beta_1$ the inverse group velocity of the on-axis column. The nonlinear
prefactor is

$$
\frac{i\omega^2}{2\varepsilon_0c^2\kappa},
$$

the UPPE coefficient of §2.3 with the real part of $k_z$ (Chapter 2 explains the choice of
the real part). Two numerical details complete the model. Columns with $k_\perp>k$ are
evanescent. solprop gives them a bounded decay of at most 200 m⁻¹ and a clamped prefactor,
a declared numerical choice, not physics. Near grazing incidence, $k_z\to0$ and the
$1/k_z$ prefactor is singular. solprop evaluates $k_z^2=k^2-k_\perp^2$ with compensated
arithmetic and warns when a column comes close to the singularity. It does not filter or
regularise it away ([grazing incidence](../guide/free-space/cartesian.md)).

The exact $k_z$ costs nothing in a spectral code: it is one complex number per column, and the
linear step is exact whatever it is. What it buys is correctness for tightly focused beams,
for broadband spectra whose short and long wavelengths diffract very differently, and for
collapsing beams whose cores approach the wavelength, as far as diffraction is concerned; at that
scale the omitted vectorial terms matter too. What it does not buy is the full vector
problem. solprop's free-space propagation is scalar per polarisation component, so the
longitudinal field and the vectorial corrections that Fibich and Ilan describe (summarised by
Couairon and Mysyrowicz 2007, §2.2.9) are not included.

## 6.2 Radial beams and the Hankel transform

A cylindrically symmetric beam depends only on $r$, and the transverse Fourier transform becomes
a Hankel transform of order zero:

$$
H(k)=\int_0^\infty rf(r)J_0(kr)\,\mathrm dr,\qquad
f(r)=\int_0^\infty kH(k)J_0(kr)\,\mathrm dk .
$$

On a computer it is discretised as the quasi-discrete Hankel transform (QDHT), which Guizar-Sicairos
and Gutiérrez-Vega (2004) extended to integer orders from an earlier zero-order transform The beam lives in a radius $R$. The radial samples are
placed at scaled zeros of $J_0$, $r_i=R\alpha_i/\alpha_{N+1}$, and the wavenumbers at
$k_i=\alpha_i/R$. With $S=\alpha_{N+1}$ the transform is the symmetric matrix

$$
T_{ij}=\frac{2J_0(\alpha_i\alpha_j/S)}{S\,|J_1(\alpha_i)|\,|J_1(\alpha_j)|}
$$

(Guizar-Sicairos and Gutiérrez-Vega 2004, Eq. 8), applied to vectors scaled by $|J_1(\alpha_i)|$
([the Hankel transform](../guide/free-space/hankel.md)). It satisfies a discrete Parseval
relation, and propagation is the exact angular-spectrum phase of their Eq. 16.

Two caveats are worth knowing. First, the matrix is *not* exactly orthogonal, although it is often
described that way. Guizar-Sicairos and Gutiérrez-Vega themselves say only that it is "very close to
being a unitary matrix". solprop measures $\max|T^2-I|=3.06\times10^{-11}$ at $N=256$. That is far
below any physical tolerance, but it is a floor, not zero. Second, the transform has two separate
truncations. The radius $R$ limits how far the beam may spread, and the number of points limits
the bandwidth to $k_{\max}\approx\pi(N+\tfrac34)/R$. A beam that is not small at $R$, or not
band-limited within $k_{\max}$, has errors that no matrix precision removes. The transform also
costs $N^2$ per application, as a dense matrix, and its samples are densest near the axis, which
suits a collapsing beam.

The boundary of the bare grid is $E=0$ at $R$, a perfectly reflecting wall. Unlike the waveguide
routes, solprop's free-space routes carry **no absorber unless you declare one**
([spatial collar and statistics](../guide/free-space/model.md#spatial-collar-and-statistics)).
With one declared, a Planck-tapered collar in the outer tenth of the radius absorbs as a rate
inside every stage, not as a mask applied between steps (Chapter 12 explains why). At every saved
plane, solprop measures the share of the fluence that sits in the collar and warns when it is
too large. The remedy is a larger radius, not a stronger collar.

## 6.3 Cartesian and line grids

A Cartesian grid represents both transverse coordinates with a two-dimensional FFT
([Cartesian grids](../guide/free-space/cartesian.md)). It can carry any transverse structure: a
tilted or astigmatic beam, crossing beams, a polarisation pattern, a vortex, any number of
filaments. The price is memory: $n_xn_y$ columns for every time sample. The box is periodic, so
without a collar light leaving one side re-enters on the other.

A **line** grid resolves $x$ only, with the field uniform in $y$. Its quantities are per unit width,
in W/m and J/m. Its physics is genuinely (1+1)-dimensional and not that of a round beam. A
Gaussian diffracts with an axial amplitude falling as $(1+z^2/z_R^2)^{-1/4}$ and a Gouy phase of
$\tfrac12\arctan(z/z_R)$. A Kerr medium supports a one-parameter family of stable sech-shaped
spatial solitons. There is no collapse and no critical power, and solprop's critical-power tool
refuses a line grid ([the line, uniform in y](../guide/free-space/model.md#the-line-uniform-in-y)).
A real sheet beam breaks up through modulational instability in $y$, which a line grid excludes by
construction. The planar route of Chapter 4 resolves $y$ with a Fourier grid above slab modes in
$x$, with $k_z=\sqrt{k^2\varepsilon_{\rm co}-\kappa_p^2-k_y^2}$, which is the right tool for that
problem ([resolving y](../guide/waveguides/planar.md#resolving-y)).

## 6.4 Self-focusing and the critical power

In a Kerr medium with $n_2>0$, the centre of a beam sees a higher index than its edges and acts as
a positive lens. If the beam's power exceeds a critical value, this self-focusing overcomes
diffraction, and in a pure Kerr medium the beam collapses in finite distance. Fibich and Gaeta
(2000) put the threshold in a form that holds for any profile,

$$
P_{\rm cr}=\alpha\,\frac{\lambda^2}{4\pi n_0n_2},
$$

with $\lambda$ the vacuum wavelength. The constant depends on the beam's shape. For the Townes
profile, the stationary self-trapped solution, $\alpha=N_{\rm cr}\approx1.86225$, which is a lower
bound for every profile (their Eq. 6). A negative Hamiltonian is sufficient for collapse and gives
an upper bound. For a Gaussian it is $\alpha=2$ (their Eq. 7). Their numerical value for a
collimated Gaussian is $\alpha=1.8962$ (their Table 1), less than 2% above the Townes value. There is
only one threshold: the old distinction between self-trapping and self-focusing powers is, in their
word, "outdated".

The literature uses several of these constants, often under the same name. Couairon and Mysyrowicz
(2007, Eq. 4) and Bergé et al. (2007, Eq. 49) write $3.72\lambda^2/8\pi n_0n_2$, which is the Townes
value. Elsewhere Couairon and Mysyrowicz quote $P_{\rm cr}=3.2$ GW for air, which corresponds to
$\alpha=2$. Sprangle, Peñano and Hafizi (2002) define a "self-focusing power"
$\lambda^2/(2\pi n_0n_2)$, also $\alpha=2$. When a paper says "$P/P_{\rm cr}=3$", find out which
$P_{\rm cr}$ it means. solprop's `critical_power` uses the Townes constant. The guide's
statement of it, in terms of $\chi^{(3)}$, is
$P_{\rm crit}=\alpha n_0\varepsilon_0c\lambda^2/(3\pi\chi^{(3)})$
([critical power and self-focusing](../guide/free-space/model.md#critical-power-and-self-focusing)).
For argon at 1 bar and 800 nm it gives about 9.2 GW.

This is the physical power: solprop's free-space routes measure power, energy and
intensity as the axial Poynting flux in the medium, $n_0\varepsilon_0c|E|^2/2$ for a
paraxial beam ([energy and flux in a medium](../guide/free-space/flux.md)). The same energy
in glass of index 1.5 therefore has a field $\sqrt{1.5}$ lower than in vacuum, and the
design tool `self_focusing_power` returns the same value as `critical_power`.

Where does a beam above the critical power collapse? For a collimated Gaussian in a pure Kerr
medium, the empirical fit usually credited to Marburger (1975) gives

$$
z_c=\frac{0.367\,z_R}{\sqrt{\big[(P/P_{\rm cr})^{1/2}-0.852\big]^2-0.0219}},
\qquad z_R=\frac{kw_0^2}{2},
$$

with $w_0$ the $1/e^2$ intensity radius (Couairon and Mysyrowicz 2007, Eq. 5). For a focused beam,
the collapse position combines with the focal length as $1/z_{c,f}=1/z_c+1/f$ (their Eq. 6). The
version printed as Eq. 50 of Bergé et al. (2007) is misprinted, and their erratum gives the correct
form. The fit fails near 100 critical powers, and with plasma or dispersion present it is only an
ordering estimate. At 800 nm, a beam of 1 mm waist at 5 critical powers collapses after about a metre.

## 6.5 Filamentation, and what radial symmetry cannot show

A real beam does not collapse to a point. As the intensity rises, multiphoton and tunnel ionisation
produce a plasma whose negative index defocuses the core. In a gas the intensity clamps, typically
near $10^{13}$–$10^{14}$ W/cm². The interplay of Kerr focusing, plasma defocusing, dispersion and
diffraction then produces a filament that persists far beyond the Rayleigh length (Couairon and
Mysyrowicz 2007; Bergé et al. 2007). solprop's [radial filament
lesson](../introduction/lessons/plot_03_filament.rst) follows exactly this sequence in argon.
Chapter 10 gives the ionisation and plasma models.

A radial calculation of a filament is efficient, but its symmetry is an assumption that no amount
of resolution can remove ([spatial decompositions](../guide/setup/spatial-decompositions.md)). It
cannot predict:

- **azimuthal modulational instability**, which grows fastest in the dipolar mode and splits a beam
  into two filaments (Bergé et al. 2007, Eq. 69; Couairon and Mysyrowicz 2007, §3.2.8);
- **multiple filamentation**, which noise seeds at around 100 critical powers (Couairon and
  Mysyrowicz 2007, §1.2.13);
- **tilted, astigmatic, elliptical or crossing beams**;
- **vortices**, or their break-up into $2|m|+1$ filaments (Bergé et al. 2007, §3.6);
- **vectorial and non-paraxial symmetry breaking**, which can split a beam of 20 critical powers
  (Couairon and Mysyrowicz 2007, §3.2.9).

A Cartesian grid lifts the symmetry restriction, so it can represent the first four. It does not
add the physics of the last: solprop's free-space model is scalar per polarisation component (§6.1),
so the vectorial mechanism stays excluded whatever the grid. That is a restriction of the equation,
not of the representation. solprop considered, and did not build, a radial grid
that carries azimuthal orders as well. Each azimuthal order of a Hankel transform samples at the
zeros of its own Bessel function, so different orders share no radial nodes. Such a grid could
carry one order exactly, but could still not show the instability that matters above the critical
power. The [crossing beams lesson](../introduction/lessons/plot_04_crossing_beams.rst) shows the
Cartesian grid doing something no radial calculation can.

:::{admonition} What to remember
:class: tip

- solprop propagates every transverse column with the exact $k_z=\sqrt{k^2-k_\perp^2}$. Paraxial
  models are its second-order expansion. Evanescent columns get a bounded numerical decay.
- The radial route uses the quasi-discrete Hankel transform: samples at Bessel zeros, a symmetric
  kernel that is unitary to about $10^{-11}$, cost $N^2$. Radius and point count are separate
  truncations.
- Free-space grids reflect or wrap unless you declare an absorber, and solprop warns when the
  field reaches the collar.
- Cartesian grids carry any transverse structure at the cost of memory. Line grids are genuinely
  (1+1)-D: no collapse, no critical power.
- $P_{\rm cr}=\alpha\lambda^2/(4\pi n_0n_2)$, with $\alpha=1.86225$ (Townes, lower bound), 1.8962
  (collimated Gaussian), or 2 (sufficient, and often used). solprop uses the Townes value and the
  physical power in the medium.
- Marburger's collapse distance uses $z_R=kw_0^2/2$. Use the erratum form of Bergé et al.'s focused
  version.
- A radial filament calculation cannot show azimuthal instability, multiple filaments, vortices or
  any off-axis structure.
:::

The precise statements are in [free-space propagation](../guide/free-space/model.md), [the
Hankel transform](../guide/free-space/hankel.md), [Cartesian grids](../guide/free-space/cartesian.md)
and [beams](../guide/free-space/beams.md). [Chapter 7](linear-media.md) turns from the geometry to
the medium itself.
