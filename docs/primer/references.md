(primer-references)=
# References

The papers cited in the primer, grouped by topic. Each entry notes what it contributes and
which chapters draw on it. Equation and section numbers refer to the published versions.
The user guide cites further sources for specific parameters and validation data, and
optmat's documentation cites the sources of every material datum.

## Propagation equations

- **M. Kolesik, J. V. Moloney and M. Mlejnek**, "Unidirectional optical pulse propagation
  equation", Phys. Rev. Lett. **89**, 283902 (2002),
  [doi:10.1103/PhysRevLett.89.283902](https://doi.org/10.1103/PhysRevLett.89.283902). The
  original UPPE, derived by projecting Maxwell's equations onto forward and backward
  subspaces, and the divergence term that arrests collapse at wavelength-scale waists.
  Chapters 1 and 2.
- **M. Kolesik and J. V. Moloney**, "Nonlinear optical pulse propagation simulation: From
  Maxwell's to unidirectional equations", Phys. Rev. E **70**, 036604 (2004),
  [doi:10.1103/PhysRevE.70.036604](https://doi.org/10.1103/PhysRevE.70.036604). The
  $z$-propagated UPPE solprop solves (Eq. 28), the coupled-mode form for waveguides (Eq. 18),
  the derivation of envelope equations from it (Sec. VI), and the reference-frequency test
  (Sec. VI G). Chapters 1–3, 5 and 13.
- **M. Kolesik, P. Jakobsen and J. V. Moloney**, "Quantifying the limits of unidirectional
  ultrashort optical pulse propagation", Phys. Rev. A **86**, 035801 (2012),
  [doi:10.1103/PhysRevA.86.035801](https://doi.org/10.1103/PhysRevA.86.035801). The first
  correction to the unidirectional approximation and its size in gases and solids. Chapter 2.
- **M. Kolesik, P. Townsend and J. V. Moloney**, "Theory and simulation of ultrafast intense
  pulse propagation in extended media", IEEE J. Sel. Top. Quantum Electron. **18**, 494–506
  (2012), [doi:10.1109/JSTQE.2011.2149499](https://doi.org/10.1109/JSTQE.2011.2149499). Why a
  single directed wave need not satisfy the wave equation, and the failure of Taylor
  dispersion beyond its radius of convergence (Figs. 1–2). Chapters 1 and 2.
- **M. Kolesik and J. V. Moloney**, "Modeling and simulation techniques in extreme nonlinear
  optics of gaseous and condensed media", Rep. Prog. Phys. **77**, 016401 (2014),
  [doi:10.1088/0034-4885/77/1/016401](https://doi.org/10.1088/0034-4885/77/1/016401). A major
  review: the canonical form $\partial_zE=iKE+iQP$, the interaction picture, and models of the
  bound and freed electrons. Chapters 2, 8, 9 and 12.
- **A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and M. Kolesik**,
  "Practitioner's guide to laser pulse propagation models and simulation", Eur. Phys. J. Spec.
  Top. **199**, 5–76 (2011),
  [doi:10.1140/epjst/e2011-01503-3](https://doi.org/10.1140/epjst/e2011-01503-3). A step-by-step
  derivation from Maxwell's equations to the UPPE and envelope equations, and a practical solver
  recipe. Chapters 1, 2, 8, 9 and 12.
- **P. Kinsler, S. B. P. Radnor and G. H. C. New**, "Theory of directional pulse propagation",
  Phys. Rev. A **72**, 063807 (2005),
  [doi:10.1103/PhysRevA.72.063807](https://doi.org/10.1103/PhysRevA.72.063807). Directional fields
  and why a mismatched reference medium forces a backward field. Chapter 2.
- **P. Kinsler**, "Limits of the unidirectional pulse propagation approximation", J. Opt. Soc. Am.
  B **24**, 2363 (2007), [doi:10.1364/JOSAB.24.002363](https://doi.org/10.1364/JOSAB.24.002363). The
  unidirectional error in a carrier shock, about 4% at damage-threshold fields. Chapter 2.
- **P. Kinsler**, "Optical pulse propagation with minimal approximations", Phys. Rev. A **81**,
  013819 (2010), [doi:10.1103/PhysRevA.81.013819](https://doi.org/10.1103/PhysRevA.81.013819).
  Factorisation of the wave equation into exact first-order equations, with term-by-term validity
  criteria (Eqs. 13–26 and App. A). Chapters 1 and 2.
- **A. Ferrando, M. Zacarés, P. Fernández de Córdoba, D. Binosi and Á. Montero**,
  "Forward-backward equations for nonlinear propagation in axially invariant optical systems",
  Phys. Rev. E **71**, 016601 (2005),
  [doi:10.1103/PhysRevE.71.016601](https://doi.org/10.1103/PhysRevE.71.016601). Exact modal
  forward–backward equations for fibres, and a precedent for keeping only the real index in the
  projection. Chapters 1 and 2.
- **J. Andreasen and M. Kolesik**, "Nonlinear propagation of light in structured media: generalized
  unidirectional pulse propagation equations", arXiv:1206.5340 (2012),
  [doi:10.48550/arXiv.1206.5340](https://doi.org/10.48550/arXiv.1206.5340). The UPPE for
  transversely structured media, and the forward field defined from the field and its
  $z$-derivative. Chapters 1 and 2.
- **C. Brée, S. Amiranashvili and U. Bandelow**, "Spatio-temporal pulse propagation in nonlinear
  dispersive optical media", Opt. Quantum Electron. **45**, 727–733 (2012),
  [doi:10.1007/s11082-012-9631-7](https://doi.org/10.1007/s11082-012-9631-7). The UPPE for the analytic
  signal and the positive-frequency part of the cubic. Chapters 1 and 2.

## Envelope equations and the GNLSE

- **T. Brabec and F. Krausz**, "Nonlinear optical pulse propagation in the single-cycle regime",
  Phys. Rev. Lett. **78**, 3282–3285 (1997),
  [doi:10.1103/PhysRevLett.78.3282](https://doi.org/10.1103/PhysRevLett.78.3282). The nonlinear envelope
  equation and the slowly evolving wave approximation (Eq. 6). Chapter 3.
- **T. Brabec and F. Krausz**, "Intense few-cycle laser fields: Frontiers of nonlinear optics", Rev.
  Mod. Phys. **72**, 545–591 (2000),
  [doi:10.1103/RevModPhys.72.545](https://doi.org/10.1103/RevModPhys.72.545). Sec. V: field and envelope
  equations. The sign of the nonlinear term in its Eq. 26 does not match the paper's own transform
  convention; Eq. 29 is consistent. Chapter 3.
- **J. M. Dudley, G. Genty and S. Coen**, "Supercontinuum generation in photonic crystal fiber", Rev.
  Mod. Phys. **78**, 1135–1184 (2006),
  [doi:10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135). The GNLSE (Eq. 2), the
  effective area (Eq. 1), the frequency-dependent shock time (Eq. 3), soliton fission and dispersive
  waves (Sec. V), and the advice to always test by a convergence study (Sec. IV.B). Chapters 3, 12 and 13.
- **G. Genty, P. Kinsler, B. Kibler and J. M. Dudley**, "Nonlinear envelope equation modeling of
  sub-cycle dynamics and harmonic generation in nonlinear waveguides", Opt. Express **15**, 5382 (2007),
  [doi:10.1364/OE.15.005382](https://doi.org/10.1364/OE.15.005382). An envelope equation with an explicit
  third-harmonic term (Eq. 4), validated against Maxwell's equations. Chapters 3 and 8.
- **G. Genty, S. Coen and J. M. Dudley**, "Fiber supercontinuum sources (Invited)", J. Opt. Soc. Am. B
  **24**, 1771 (2007), [doi:10.1364/JOSAB.24.001771](https://doi.org/10.1364/JOSAB.24.001771). The GNLSE
  in practice, characteristic lengths, and fission against modulational instability. Chapter 3.
- **M. Bache**, "The nonlinear analytical envelope equation in quadratic nonlinear crystals",
  arXiv:1603.00188 (2016), [doi:10.48550/arXiv.1603.00188](https://doi.org/10.48550/arXiv.1603.00188). The
  analytic-envelope equation with Kerr and Raman, and the $\tfrac23$ between field-level and envelope
  Raman shares (Eqs. 21–26), and the full analytic cubic with its $3|A|^2A^*e^{+2i\omega_0t}$ term (Eq. 18).
  Chapters 3, 8 and 9.
- **H. Guo, X. Zeng, B. Zhou and M. Bache**, "Nonlinear wave equation in frequency domain: accurate
  modeling of ultrafast interaction in anisotropic nonlinear media", J. Opt. Soc. Am. B **30**, 494
  (2013), [doi:10.1364/JOSAB.30.000494](https://doi.org/10.1364/JOSAB.30.000494). A spectral field
  equation in the FFT sign convention, with the same prefactor as solprop's internal one. Chapter 3.
- **J. Lægsgaard**, "Mode profile dispersion in the generalised nonlinear Schrödinger equation", Opt.
  Express **15**, 16110 (2007), [doi:10.1364/OE.15.016110](https://doi.org/10.1364/OE.15.016110). The
  four-frequency overlap of a frequency-dependent mode, and the photon-number-conserving form (Eqs.
  16–17, 33–35). Chapters 3 and 5.
- **P. V. Mamyshev and S. V. Chernikov**, "Ultrashort-pulse propagation in optical fibers", Opt. Lett.
  **15**, 1076 (1990), [doi:10.1364/OL.15.001076](https://doi.org/10.1364/OL.15.001076). A frequency-domain
  single-mode equation with a frequency-dependent area and photon-number conservation. Chapter 3.
- **S. V. Chernikov and P. V. Mamyshev**, "Femtosecond soliton propagation in fibers with slowly decreasing
  dispersion", J. Opt. Soc. Am. B **8**, 1633 (1991),
  [doi:10.1364/JOSAB.8.001633](https://doi.org/10.1364/JOSAB.8.001633). Why photon number, not energy, is the
  invariant with Raman (Eqs. 2–6). Chapter 3.
- **S. Amiranashvili, A. G. Vladimirov and U. Bandelow**, "A model equation for ultrashort optical pulses
  around the zero dispersion frequency", Eur. Phys. J. D **58**, 219–226 (2010),
  [doi:10.1140/epjd/e2010-00010-3](https://doi.org/10.1140/epjd/e2010-00010-3). The finite radius of
  convergence of Taylor dispersion (Fig. 1). Chapters 1 and 3.
- **Sh. Amiranashvili, U. Bandelow and N. Akhmediev**, "Dispersion of nonlinear group velocity determines
  shortest envelope solitons", Phys. Rev. A **84**, 043834 (2011),
  [doi:10.1103/PhysRevA.84.043834](https://doi.org/10.1103/PhysRevA.84.043834). The frequency dependence of
  the nonlinear coefficient as an operator (Eq. 4). Chapter 3.
- **S. Amiranashvili, U. Bandelow and A. Mielke**, "Calculation of ultrashort pulse propagation based on
  rational approximations for medium dispersion", Opt. Quantum Electron. **44**, 241–246 (2012),
  [doi:10.1007/s11082-011-9537-9](https://doi.org/10.1007/s11082-011-9537-9). Rational approximations of
  dispersion, which keep the correct limit at high frequency. Chapter 3.
- **N. Akhmediev and M. Karlsson**, "Cherenkov radiation emitted by solitons in optical fibers", Phys. Rev. A
  **51**, 2602–2607 (1995), [doi:10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602). The
  soliton wavenumber and the phase-matching condition for resonant dispersive waves (Eqs. 2 and 5a).
  Chapter 3.
- **D. V. Skryabin and A. V. Yulin**, "Theory of generation of new frequencies by mixing of solitons and
  dispersive waves in optical fibers", Phys. Rev. E **72**, 016619 (2005),
  [doi:10.1103/PhysRevE.72.016619](https://doi.org/10.1103/PhysRevE.72.016619). Four-wave-mixing resonances
  between solitons and dispersive waves (Eq. 11). Chapter 3.
- **J. C. Travers, W. Chang, J. Nold, N. Y. Joly and P. St. J. Russell**, "Ultrafast nonlinear optics in
  gas-filled hollow-core photonic crystal fibers [Invited]", J. Opt. Soc. Am. B **28**, A11 (2011),
  [doi:10.1364/JOSAB.28.000A11](https://doi.org/10.1364/JOSAB.28.000A11). Gas-filled fibre modes and
  nonlinearity, the $(\lambda_{\rm ZDW},N)$ scaling, and the phase-matching condition (Eqs. 1, 4 and 8). The
  core sizes it quotes are diameters. Chapters 3 and 4.
- **N. Y. Joly, J. Nold, W. Chang, P. Hölzer, A. Nazarkin, G. K. L. Wong, F. Biancalana and P. St. J. Russell**,
  "Bright spatially coherent wavelength-tunable deep-UV laser source using an Ar-filled photonic crystal
  fiber", Phys. Rev. Lett. **106**, 203901 (2011),
  [doi:10.1103/PhysRevLett.106.203901](https://doi.org/10.1103/PhysRevLett.106.203901). Deep-ultraviolet
  dispersive waves, and the size of the shock term's effect on them. Chapter 3.

## Waveguides

- **E. A. J. Marcatili and R. A. Schmeltzer**, "Hollow metallic and dielectric waveguides for long distance
  optical transmission and lasers", Bell Syst. Tech. J. **43**, 1783–1809 (1964),
  [doi:10.1002/j.1538-7305.1964.tb04108.x](https://doi.org/10.1002/j.1538-7305.1964.tb04108.x). The leaky
  modes of a hollow circular guide (Eqs. 17–21) and the power normalisation (Eq. 42). Its $\alpha$ is a field
  attenuation. Chapters 4 and 5.
- **L. Vincetti**, "Empirical formulas for calculating loss in hollow core tube lattice fibers", Opt. Express
  **24**, 10313 (2016), [doi:10.1364/OE.24.010313](https://doi.org/10.1364/OE.24.010313). The minimum loss of
  a transmission band (Eq. 3) and the normalised frequency $F$. Chapter 4.
- **L. Vincetti and L. Rosa**, "A simple analytical model for confinement loss estimation in hollow-core tube
  lattice fibers", Opt. Express **27**, 5230 (2019),
  [doi:10.1364/OE.27.005230](https://doi.org/10.1364/OE.27.005230). The resonant loss bands as Lorentzians
  about fitted cut-offs (Eqs. 2–7), and the validated range of geometries (Table 1). Chapter 4.
- **L. Rosa, F. Melli and L. Vincetti**, "Analytical formulas for dispersion and effective area in hollow-core
  tube lattice fibers", Fibers **9**, 58 (2021), [doi:10.3390/fib9100058](https://doi.org/10.3390/fib9100058).
  The index, dispersion and area of the antiresonant model. Chapter 4.
- **F. Melli, L. Rosa, K. Vasko, F. Benabid and L. Vincetti**, "Analytical framework for mode-coupling in
  hollow-core inhibited-coupling fibers", J. Lightwave Technol. **43**, 5464–5472 (2025),
  [doi:10.1109/JLT.2025.3548304](https://doi.org/10.1109/JLT.2025.3548304). Weak and strong coupling between
  core and cladding modes near a resonance; not modelled by solprop. Chapter 4.
- **J. Andreasen and M. Kolesik**, "Midinfrared femtosecond laser pulse filamentation in hollow waveguides: A
  comparison of simulation methods", Phys. Rev. E **87**, 053303 (2013),
  [doi:10.1103/PhysRevE.87.053303](https://doi.org/10.1103/PhysRevE.87.053303). Why an incoherent modal loss
  overestimates the loss of reshaped beams, and the semitransparent wall boundary (Eqs. 10–15) on which
  solprop's coherent wall is built. Chapters 4 and 5.
- **J. Andreasen and M. Kolesik**, "Core-confined beam propagation method for guiding and leaky structures", J.
  Lightwave Technol. **31**, 2999–3005 (2013),
  [doi:10.1109/JLT.2013.2277660](https://doi.org/10.1109/JLT.2013.2277660). The core-only boundary conditions in
  vector form (Eqs. 6, 9, 18–19). Chapter 5.
- **K. D. Laakmann and W. H. Steier**, "Waveguides: characteristic modes of hollow rectangular dielectric
  waveguides", Appl. Opt. **15**, 1334 (1976), [doi:10.1364/AO.15.001334](https://doi.org/10.1364/AO.15.001334),
  with an erratum in Appl. Opt. **15**, 2029 (1976). The grazing-incidence model of hollow rectangular and planar
  guides, extended by solprop to a gas-filled core. Chapter 4.
- **A. W. Snyder and J. D. Love**, *Optical Waveguide Theory* (Chapman and Hall, London, 1983). The mode
  families and names that solprop's step-index fibres follow. Chapter 4.
- **R. Pennetta, M. T. Enders, M. H. Frosz, F. Tani and P. St. J. Russell**, "Fabrication and non-destructive
  characterization of tapered single-ring hollow-core photonic crystal fiber", APL Photonics **4**, 056105
  (2019), [doi:10.1063/1.5093474](https://doi.org/10.1063/1.5093474). The pressure profile of laminar gas flow
  through a tapered fibre. Chapter 4.

## Multimode and vector propagation

- **F. Poletti and P. Horak**, "Description of ultrashort pulse propagation in multimode optical fibers", J.
  Opt. Soc. Am. B **25**, 1645 (2008), [doi:10.1364/JOSAB.25.001645](https://doi.org/10.1364/JOSAB.25.001645).
  The multimode GNLSE with its overlap tensors and symmetries (Eqs. 5–8, 14, 16), and its cost. Chapters 5 and 8.
- **L. G. Wright, Z. M. Ziegler, P. M. Lushnikov, Z. Zhu, M. A. Eftekhar, D. N. Christodoulides and F. W. Wise**,
  "Multimode nonlinear fiber optics: Massively parallel numerical solver, tutorial, and outlook", IEEE J. Sel.
  Top. Quantum Electron. **24**, 1–16 (2018),
  [doi:10.1109/JSTQE.2017.2779749](https://doi.org/10.1109/JSTQE.2017.2779749). The scalar overlap form and the
  break-even against full-field methods. Chapter 5.
- **L. G. Wright, D. N. Christodoulides and F. W. Wise**, "Controllable spatiotemporal nonlinear effects in
  multimode fibres", Nat. Photon. **9**, 306–310 (2015),
  [doi:10.1038/nphoton.2015.61](https://doi.org/10.1038/nphoton.2015.61). Multimode nonlinear dynamics in
  graded-index fibres. Chapter 4.
- **L. G. Wright, W. H. Renninger, D. N. Christodoulides and F. W. Wise**, "Nonlinear multimode photonics:
  nonlinear optics with many degrees of freedom", Optica **9**, 824 (2022),
  [doi:10.1364/OPTICA.461981](https://doi.org/10.1364/OPTICA.461981). A review of multimode nonlinear optics and
  its computational cost. Chapter 4.
- **S. Afshar V. and T. M. Monro**, "A full vectorial model for pulse propagation in emerging waveguides with
  subwavelength structures part I: Kerr nonlinearity", Opt. Express **17**, 2298 (2009),
  [doi:10.1364/OE.17.002298](https://doi.org/10.1364/OE.17.002298). Longitudinal fields, non-orthogonal
  transverse fields (Eq. 36) and the vector Kerr coefficient (Eqs. 28, 39). Chapter 5.
- **M. D. Turner, T. M. Monro and S. Afshar V.**, "A full vectorial model for pulse propagation in emerging
  waveguides with subwavelength structures part II: Stimulated Raman scattering", Opt. Express **17**, 11565
  (2009), [doi:10.1364/OE.17.011565](https://doi.org/10.1364/OE.17.011565). The vector Raman counterpart. Chapter 5.
- **J. Lægsgaard**, "Modeling of nonlinear propagation in fiber tapers", J. Opt. Soc. Am. B **29**, 3183 (2012),
  [doi:10.1364/JOSAB.29.003183](https://doi.org/10.1364/JOSAB.29.003183). Full-vector taper equations and the term
  from a $z$-dependent normalisation (Eq. 26). Chapter 5.
- **M. Kolesik, E. M. Wright and J. V. Moloney**, "Simulation of femtosecond pulse propagation in sub-micron
  diameter tapered fibers", Appl. Phys. B **79**, 293–300 (2004),
  [doi:10.1007/s00340-004-1551-1](https://doi.org/10.1007/s00340-004-1551-1). A modal UPPE with frequency-dependent
  modes; the GNLSE as the frozen-profile limit. Chapter 3.

## Free space and filamentation

- **M. Guizar-Sicairos and J. C. Gutiérrez-Vega**, "Computation of quasi-discrete Hankel transforms of integer
  order for propagating optical wave fields", J. Opt. Soc. Am. A **21**, 53 (2004),
  [doi:10.1364/JOSAA.21.000053](https://doi.org/10.1364/JOSAA.21.000053). The quasi-discrete Hankel transform
  (Eqs. 6–8) and its accuracy. Chapter 6.
- **G. Fibich and A. L. Gaeta**, "Critical power for self-focusing in bulk media and in hollow waveguides", Opt.
  Lett. **25**, 335 (2000), [doi:10.1364/OL.25.000335](https://doi.org/10.1364/OL.25.000335). The critical power
  for any profile, its Townes lower bound and sufficient upper bound (Eqs. 1, 6, 7), and the Gaussian value
  (Table 1). Chapter 6.
- **J. H. Marburger**, "Self-focusing: Theory", Prog. Quantum Electron. **4**, 35–110 (1975),
  [doi:10.1016/0079-6727(75)90003-8](https://doi.org/10.1016/0079-6727(75)90003-8). The collapse-distance fit,
  quoted in the primer as given by Couairon and Mysyrowicz. Chapter 6.
- **A. Couairon and A. Mysyrowicz**, "Femtosecond filamentation in transparent media", Phys. Rep. **441**, 47–189
  (2007), [doi:10.1016/j.physrep.2006.12.005](https://doi.org/10.1016/j.physrep.2006.12.005). The standard review
  of filamentation: critical power and Marburger's formula (Eqs. 4–6), plasma (Eqs. 8 and 28), $n_2$ (Eq. 29), and
  symmetry breaking (§3.2). Chapters 6, 8 and 10.
- **L. Bergé, S. Skupin, R. Nuter, J. Kasparian and J.-P. Wolf**, "Ultrashort filaments of light in weakly
  ionized, optically transparent media", Rep. Prog. Phys. **70**, 1633–1713 (2007),
  [doi:10.1088/0034-4885/70/10/R03](https://doi.org/10.1088/0034-4885/70/10/R03). Paraxial and UPPE models (Eqs. 9,
  15, 18), the critical power (Eq. 49), and azimuthal instability (Eq. 69). Its Eq. 50 is corrected by the
  journal's erratum. Chapter 6.
- **P. Sprangle, J. R. Peñano and B. Hafizi**, "Propagation of intense short laser pulses in the atmosphere",
  Phys. Rev. E **66**, 046418 (2002), [doi:10.1103/PhysRevE.66.046418](https://doi.org/10.1103/PhysRevE.66.046418).
  A self-focusing power with $\alpha=2$, and filamentation on a Cartesian grid. Chapter 6.

## Material data and the Kerr effect

- **J. D. Jackson**, *Classical Electrodynamics*, 3rd ed. (Wiley, New York, 1999), §7.10. Causality and the
  Kramers–Kronig relations. Chapter 1.
- **M. Puchalski, K. Piszczatowski, J. Komasa, B. Jeziorski and K. Szalewicz**, "Theoretical determination of the
  polarizability dispersion and the refractive index of helium", Phys. Rev. A **93**, 032515 (2016),
  [doi:10.1103/PhysRevA.93.032515](https://doi.org/10.1103/PhysRevA.93.032515). The *ab initio* basis of optmat's
  helium index. Chapter 7.
- **M. Lesiuk and B. Jeziorski**, "First-principles calculation of the frequency-dependent dipole polarizability of
  argon", Phys. Rev. A **107**, 042805 (2023),
  [doi:10.1103/PhysRevA.107.042805](https://doi.org/10.1103/PhysRevA.107.042805). The *ab initio* basis of optmat's
  argon index. Chapter 7.
- **D. Franta, D. Nečas, I. Ohlídal and A. Giglia**, "Optical characterization of SiO₂ thin films using universal
  dispersion model over wide spectral range", Proc. SPIE **9890**, 989014 (2016),
  [doi:10.1117/12.2227580](https://doi.org/10.1117/12.2227580). The causal dispersion model on which optmat's fused
  silica is built. Chapter 7.
- **D. P. Shelton and J. E. Rice**, "Measurements and calculations of the hyperpolarizabilities of atoms and small
  molecules in the gas phase", Chem. Rev. **94**, 3–29 (1994). The Taylor convention for hyperpolarisabilities and
  the relative form of their dispersion. Chapter 8.
- **D. M. Bishop and D. W. De Kee**, "The frequency dependence of nonlinear optical processes", J. Chem. Phys.
  **104**, 9876–9887 (1996), [doi:10.1063/1.471752](https://doi.org/10.1063/1.471752). The dispersion expansion over
  the four frequencies of a mixing process, on which optmat's fit and solprop's dispersive Kerr response rest.
  Chapter 8.
- **V. Loriot, E. Hertz, O. Faucher and B. Lavorel**, "Measurement of high order Kerr refractive index of major air
  components", Opt. Express **17**, 13429 (2009),
  [doi:10.1364/OE.17.013429](https://doi.org/10.1364/OE.17.013429); **P. Béjot et al.**, "Higher-order Kerr terms
  allow ionization-free filamentation in gases", Phys. Rev. Lett. **104**, 103903 (2010),
  [doi:10.1103/PhysRevLett.104.103903](https://doi.org/10.1103/PhysRevLett.104.103903); and **J. K. Wahlstrand, Y.-H.
  Cheng and H. M. Milchberg**, "High field optical nonlinearity and the Kramers–Kronig relations", Phys. Rev. Lett.
  **109**, 113904 (2012), [doi:10.1103/PhysRevLett.109.113904](https://doi.org/10.1103/PhysRevLett.109.113904). The
  higher-order Kerr debate. Chapter 8.
- **K. Schuh, M. Kolesik, E. M. Wright and J. V. Moloney**, "Simple model for the nonlinear optical response of gases
  in the transparency region", Opt. Lett. **39**, 5086 (2014),
  [doi:10.1364/OL.39.005086](https://doi.org/10.1364/OL.39.005086). A bound-electron model fitted to the TDSE; not
  implemented in solprop. Chapter 8.

## Raman scattering

- **K. J. Blow and D. Wood**, "Theoretical description of transient stimulated Raman scattering in optical fibers",
  IEEE J. Quantum Electron. **25**, 2665–2673 (1989), [doi:10.1109/3.40655](https://doi.org/10.1109/3.40655). The
  damped-oscillator response of silica and the $\tfrac23$ of the envelope Raman term. Its $\alpha$ is the Kerr share.
  Chapter 9.
- **R. H. Stolen, J. P. Gordon, W. J. Tomlinson and H. A. Haus**, "Raman response function of silica-core fibers", J.
  Opt. Soc. Am. B **6**, 1159 (1989), [doi:10.1364/JOSAB.6.001159](https://doi.org/10.1364/JOSAB.6.001159). The response
  derived from the measured gain, and $f_R=0.18$. Chapter 9.
- **R. H. Stolen and E. P. Ippen**, "Raman gain in glass optical waveguides", Appl. Phys. Lett. **22**, 276–278 (1973),
  [doi:10.1063/1.1654637](https://doi.org/10.1063/1.1654637). The measured silica Raman gain. Chapter 9.
- **R. Hellwarth, J. Cherlow and T.-T. Yang**, "Origin and frequency dependence of nonlinear optical susceptibilities of
  glasses", Phys. Rev. B **11**, 964–967 (1975), [doi:10.1103/PhysRevB.11.964](https://doi.org/10.1103/PhysRevB.11.964).
  The electronic and nuclear shares of silica's nonlinearity, and the isotropic and anisotropic kernels (Eq. 2).
  Chapter 9.
- **D. Hollenbeck and C. D. Cantrell**, "Multiple-vibrational-mode model for fiber-optic Raman gain spectrum and response
  function", J. Opt. Soc. Am. B **19**, 2886 (2002),
  [doi:10.1364/JOSAB.19.002886](https://doi.org/10.1364/JOSAB.19.002886). The shape of solprop's default silica response
  (Eq. 9). Chapter 9.
- **Q. Lin and G. P. Agrawal**, "Raman response function for silica fibers", Opt. Lett. **31**, 3086 (2006),
  [doi:10.1364/OL.31.003086](https://doi.org/10.1364/OL.31.003086). The vector silica response with the boson peak (Eqs.
  1–3), and $f_R=0.245$. Chapter 9.

## Ionisation and plasma

- **L. V. Keldysh**, "Ionization in the field of a strong electromagnetic wave", Sov. Phys. JETP **20**, 1307–1314 (1965).
  The Keldysh parameter (Eq. 17). Chapter 10.
- **A. M. Perelomov, V. S. Popov and M. V. Terent'ev**, "Ionization of atoms in an alternating electric field", Sov. Phys.
  JETP **23**, 924–934 (1966). The short-range rate bridging multiphoton and tunnelling ionisation (Eqs. 33, 54, 55).
  Chapter 10.
- **A. M. Perelomov and V. S. Popov**, "Ionization of atoms in an alternating electrical field. III", Sov. Phys. JETP
  **25**, 336–343 (1967). The Coulomb correction, independent of the Keldysh parameter (Eq. 42). Chapter 10.
- **M. V. Ammosov, N. B. Delone and V. P. Krainov**, "Tunnel ionization of complex atoms and of atomic ions in an
  alternating electromagnetic field", Sov. Phys. JETP **64**, 1191–1194 (1986). The cycle-averaged tunnelling rate (Eq. 1).
  Chapter 10.
- **F. A. Ilkov, J. E. Decker and S. L. Chin**, "Ionization of atoms in the tunnelling regime with experimental evidence
  using Hg atoms", J. Phys. B **25**, 4005–4020 (1992),
  [doi:10.1088/0953-4075/25/19/011](https://doi.org/10.1088/0953-4075/25/19/011). The combined form of the PPT rate common in
  propagation codes (Eq. 8). Chapter 10.
- **X. M. Tong and C. D. Lin**, "Empirical formula for static field ionization rates of atoms and molecules by lasers in
  the barrier-suppression regime", J. Phys. B **38**, 2593–2600 (2005),
  [doi:10.1088/0953-4075/38/15/001](https://doi.org/10.1088/0953-4075/38/15/001). The barrier-suppression field (Eq. 1) and
  correction (Eq. 2). Chapter 10.
- **M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz and T. Brabec**, "Light propagation in field-ionizing media:
  Extreme nonlinear optics", Phys. Rev. Lett. **83**, 2930–2933 (1999),
  [doi:10.1103/PhysRevLett.83.2930](https://doi.org/10.1103/PhysRevLett.83.2930). The free-electron and loss currents (Eqs.
  4–6). Chapter 10.

## Noise

- **P. D. Drummond and J. F. Corney**, "Quantum noise in optical fibers. I. Stochastic equations", J. Opt. Soc. Am. B
  **18**, 139 (2001), [doi:10.1364/JOSAB.18.000139](https://doi.org/10.1364/JOSAB.18.000139). The truncated-Wigner vacuum
  (Eq. 6.13) and Raman Langevin noise (Eq. 6.16), against which solprop's semiclassical seed is contrasted. Chapter 10.

## Numerical methods

- **R. H. Hardin and F. D. Tappert**, "Applications of the split-step Fourier method to the numerical solution of nonlinear
  and variable coefficient wave equations", abstract in "SIAM 1972 Fall Meeting", SIAM Rev. **15**, 423 (1973),
  [doi:10.1137/1015060](https://doi.org/10.1137/1015060). The split-step Fourier method. Chapter 12.
- **F. D. Tappert and C. N. Judice**, "Recurrence of nonlinear ion acoustic waves", Phys. Rev. Lett. **29**, 1308–1311
  (1972), [doi:10.1103/PhysRevLett.29.1308](https://doi.org/10.1103/PhysRevLett.29.1308). An early use of the split-step
  method. Chapter 12.
- **R. A. Fisher and W. Bischel**, "The role of linear dispersion in plane-wave self-phase modulation", Appl. Phys. Lett.
  **23**, 661–663 (1973), [doi:10.1063/1.1654782](https://doi.org/10.1063/1.1654782); and **R. A. Fisher and W. K.
  Bischel**, "Numerical studies of the interplay between self-phase modulation and dispersion for intense plane-wave laser
  pulses", J. Appl. Phys. **46**, 4921–4934 (1975), [doi:10.1063/1.321476](https://doi.org/10.1063/1.321476). The
  independent development of the alternating method for optics, its convergence checks, and Tappert's aliasing warning
  (1975, §II.B). Chapters 11 and 12.
- **O. V. Sinkin, R. Holzlöhner, J. Zweck and C. R. Menyuk**, "Optimization of the split-step Fourier method in modeling
  optical-fiber communications systems", J. Lightwave Technol. **21**, 61–68 (2003),
  [doi:10.1109/JLT.2003.808628](https://doi.org/10.1109/JLT.2003.808628). Symmetric splitting and the local-error step
  control. Chapter 12.
- **J. D. Lawson**, "Generalized Runge–Kutta processes for stable systems with large Lipschitz constants", SIAM Journal on
  Numerical Analysis **4**, 372–380 (1967), [doi:10.1137/0704033](https://doi.org/10.1137/0704033). The integrating-factor
  (interaction-picture) Runge–Kutta method (Eqs. 2.6, 3.3). Chapter 12.
- **J. Hult**, "A fourth-order Runge–Kutta in the interaction picture method for simulating supercontinuum generation in
  optical fibers", J. Lightwave Technol. **25**, 3770–3775 (2007),
  [doi:10.1109/JLT.2007.909373](https://doi.org/10.1109/JLT.2007.909373). RK4IP (Eqs. 12a–f) and the limit set by the
  nonlinear integrator. Chapters 12 and 13.
- **S. Balac and F. Mahé**, "Embedded Runge–Kutta scheme for step-size control in the interaction picture method", Computer
  Physics Communications **184**, 1211–1219 (2013), [doi:10.1016/j.cpc.2012.12.020](https://doi.org/10.1016/j.cpc.2012.12.020). An
  embedded 4(3) pair in the interaction picture that keeps RK4IP's node structure, and the call for a 5(4) one. Chapter 12.
- **J. R. Dormand and P. J. Prince**, "A family of embedded Runge–Kutta formulae", J. Comput. Appl. Math. **6**, 19–26
  (1980), [doi:10.1016/0771-050X(80)90013-3](https://doi.org/10.1016/0771-050X(80)90013-3). The 5(4) pair with FSAL (Table
  2); the hatted weights are the fifth-order ones. Chapter 12.
- **J. R. Dormand and P. J. Prince**, "Runge–Kutta triples", Comput. Math. Appl. **12**, 1007–1017 (1986),
  [doi:10.1016/0898-1221(86)90025-8](https://doi.org/10.1016/0898-1221(86)90025-8). Continuous extensions for output
  between steps. Chapter 12.
- **Ch. Tsitouras**, "Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption", Comput.
  Math. Appl. **62**, 770–775 (2011), [doi:10.1016/j.camwa.2011.06.002](https://doi.org/10.1016/j.camwa.2011.06.002). The
  alternative 5(4) pair. Chapter 12.
- **L. F. Shampine and M. W. Reichelt**, "The MATLAB ODE suite", SIAM J. Sci. Comput. **18**, 1–22 (1997),
  [doi:10.1137/S1064827594276424](https://doi.org/10.1137/S1064827594276424). The Dormand–Prince pair in `ode45`, and the
  per-component error test. Chapter 12.
- **G. Söderlind and L. Wang**, "Adaptive time-stepping and computational stability", J. Comput. Appl. Math. **185**,
  225–243 (2006), [doi:10.1016/j.cam.2005.03.008](https://doi.org/10.1016/j.cam.2005.03.008). The PI.4.2 controller (Eq. 4,
  Table 1). Chapter 12.
- **A. M. Heidt**, "Efficient adaptive step size method for the simulation of supercontinuum generation in optical
  fibers", J. Lightwave Technol. **27**, 3984–3991 (2009),
  [doi:10.1109/JLT.2009.2021538](https://doi.org/10.1109/JLT.2009.2021538). Step control by photon-number conservation,
  and its caveat. Chapters 12 and 13.
- **A. A. Rieznik, A. M. Heidt, P. G. König, V. A. Bettachini and D. F. Grosz**, "Optimum integration procedures for
  supercontinuum simulation", IEEE Photonics J. **4**, 552–560 (2012),
  [doi:10.1109/JPHOT.2012.2188281](https://doi.org/10.1109/JPHOT.2012.2188281). The nonlinear term integrated in the
  frequency domain, and the limits of conservation-based control. Chapter 12.
- **S. M. Cox and P. C. Matthews**, "Exponential time differencing for stiff systems", J. Comput. Phys. **176**, 430–455
  (2002), [doi:10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995). Exponential time differencing and the error
  penalty of integrating factors (Eq. 32). Chapter 12.
- **A.-K. Kassam and L. N. Trefethen**, "Fourth-order time-stepping for stiff PDEs", SIAM J. Sci. Comput. **26**,
  1214–1233 (2005), [doi:10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633). Cancellation in the
  exponential coefficients and its cure (Table 2.1, Eqs. 2.6–2.7). Chapter 12.
- **M. Hochbruck and A. Ostermann**, "Exponential integrators", Acta Numerica **19**, 209–286 (2010),
  [doi:10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048). The review: $\varphi$-functions (Eq. 2.10),
  exponential Adams methods (§2.5), Lawson methods (§2.7), and the history, including Certaine (1960) and Nørsett (1969)
  (§6). Chapter 12.
- **M. Hochbruck and A. Ostermann**, "Exponential multistep methods of Adams-type", BIT Numerical Mathematics **51**, 889–908 (2011),
  [doi:10.1007/s10543-011-0332-6](https://doi.org/10.1007/s10543-011-0332-6). The analysis of exponential Adams methods at
  fixed step. Chapter 12.
- **D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash**, "A tapering window for time-domain templates and simulated
  signals in the detection of gravitational waves from coalescing compact binaries", Class. Quantum Grav. **27**, 084020
  (2010), [doi:10.1088/0264-9381/27/8/084020](https://doi.org/10.1088/0264-9381/27/8/084020). The Planck taper used for
  solprop's collars (Eq. 7). Chapter 11.
