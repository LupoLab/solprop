:orphan:

Worked examples
===============

Full-size simulations of experimentally motivated configurations, each run to the length its physics needs and
explained from the set-up to what to look for in the result. Unlike the :doc:`focused
demonstrations <../auto_tutorials/index>`, these are **not** executed when the documentation
is built: they take from seconds to a few hours each, so the pages, figures and
printed output below are the stored result of their last run, and every page ends with
the time that run took and the backend it used. To run one yourself, download the script
from its page; to re-run and refresh them all in a checkout, use
``uv run poe docs-examples`` (see :doc:`/dev/contributing`).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Nonlinear processes in gas-filled hollow-core fibres
----------------------------------------------------

One process per example -- self-phase modulation, soliton self-compression and
dispersive-wave emission, four-wave mixing, cross-phase modulation, Raman and
photoionisation-induced soliton frequency shifts -- each a single deterministic run
with the parameters of the survey figure in J. C. Travers, T. Grigorova, C. Brahms and
F. Belli's review of nonlinear optics in gas-filled hollow-core fibres, headed by the
vacuum-ultraviolet dispersive wave that illustrates this package's README. The last two
compare the electronic responses of :doc:`/guide/media/dispersive-kerr` at full size: the
helium dispersive wave, with one, two and four modes, and deep-ultraviolet four-wave
mixing in argon over a pressure scan.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A few-cycle near-infrared pulse travelling through a hollow capillary filled with a light noble gas at low pressure can enter a higher-order soliton regime when its dispersion, duration and energy are suitable. The anomalous dispersion of the gas-filled waveguide balances self-phase modulation, the pulse compresses to a sub-cycle spike (about 1 fs), and at the point of maximum compression its spectrum extends far enough into the ultraviolet to overlap a phase-matched resonance: the soliton sheds a resonant dispersive wave (RDW) at the wavelength where the waveguide dispersion intersects the soliton&#x27;s effective propagation wavenumber, including its group-delay and nonlinear phase contributions. Because the zero-dispersion wavelength of a helium-filled capillary lies deep in the ultraviolet at sub-bar pressures, that wavelength falls in the vacuum ultraviolet, 100-150 nm here. The process is efficient, tunable with the pressure and produces pulses of a few femtoseconds; it is the basis of tabletop vacuum- and deep-ultraviolet sources.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_00_vuv_dispersive_wave_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_00_vuv_dispersive_wave`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Vacuum-ultraviolet dispersive-wave emission in a gas-filled capillary</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A common use of a gas-filled hollow capillary is post-compression: a femtosecond pulse is spectrally broadened by self-phase modulation (SPM) in the gas, and the broadened, chirped pulse is then compressed with a dispersive delay line -- chirped mirrors, prisms or gratings -- to a duration far below the input&#x27;s.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_01_self_phase_modulation_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_01_self_phase_modulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Self-phase modulation and post-compression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the dispersion of the gas-filled waveguide is anomalous at the pump wavelength, the spectral phase that self-phase modulation writes is compensated as it is written: the blue-shifted trailing edge catches up with the red-shifted leading edge, the pulse gets shorter, its peak power rises, the nonlinearity grows, and the pulse can undergo strong compression. Higher dispersion, loss, ionisation and departure from the ideal NLSE alter that evolution. This is soliton self-compression, the dynamics of a higher-order soliton of order N = \sqrt{L_D/L_{NL}} in the language of the nonlinear Schrödinger equation. In a gas-filled capillary the anomalous dispersion comes from the waveguide and is smooth over a huge bandwidth, so the compression can continue to a single optical cycle and below.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_02_soliton_self_compression_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_02_soliton_self_compression`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Soliton self-compression and ultraviolet dispersive-wave emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two pulses of different colours travelling together through a gas-filled capillary mix through the third-order susceptibility. In degenerate four-wave mixing two pump photons are converted into a seed photon and an idler photon, 2\omega_p = \omega_s + \omega_i. With a deep-ultraviolet pump at 267 nm (the third harmonic of a titanium-sapphire laser) and its 800 nm fundamental as the seed, the idler is at">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_03_four_wave_mixing_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_03_four_wave_mixing`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Four-wave mixing into the vacuum ultraviolet</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A strong pulse modulates the refractive index seen by any other pulse that overlaps it. In the co-polarised, constant-susceptibility approximation used here, the Kerr index change seen by a weak probe is 2 n_2 I_\text{pump}(t), twice the self-phase modulation coefficient, and a weak probe acquires a time-dependent phase that broadens its spectrum -- cross-phase modulation (XPM). Because the two pulses travel at different group velocities they slide through each other, and the broadening a probe acquires depends on where it starts relative to the pump: a probe that is overtaken by the pump sees the whole of the pump&#x27;s intensity profile sweep across it, and its spectrum is broadened, and shifted, most.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_04_cross_phase_modulation_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_04_cross_phase_modulation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cross-phase modulation in the deep ultraviolet</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a Raman-active gas a soliton shifts its own spectrum to longer wavelengths. The blue edge of its spectrum pumps the red edge through stimulated Raman scattering -- here from the rotational transitions of molecular hydrogen, whose coherence persists for tens to hundreds of picoseconds -- so the soliton continuously transfers energy to longer wavelengths as it propagates: the soliton self-frequency shift (SSFS). Because the group velocity changes as the wavelength changes, the red-shifting soliton also decelerates and its trajectory in time bends. In solid-core fibres the same effect comes from the vibrational response of glass; in hydrogen the response is dominated by the rotational Raman lines and can be tuned with the gas pressure.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_05_raman_self_frequency_shift_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_05_raman_self_frequency_shift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Raman soliton self-frequency shift in hydrogen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When a soliton self-compresses to intensities of order 10^{14}\,\text{W/cm}^2 it can ionise the gas; the fraction depends on species, pulse duration and the selected rate model. The free electrons it creates lower the refractive index, and because they accumulate during the pulse the index falls along it: the trailing part of the pulse sees a lower index than the leading part, and the resulting phase shifts its frequency upwards. A soliton that is ionising therefore shifts its own spectrum to the blue -- the opposite of the Raman self-frequency shift -- and accelerates, so that its trajectory in time bends towards earlier arrival. The shift is self-limiting: the blue-shifted soliton sees a different dispersion and its intensity, and hence the ionisation, falls. The effect was a signature of the first experiments on plasma-soliton dynamics in gas-filled hollow-core photonic crystal fibres, and it is readily reproduced in small-core capillaries.">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_06_ionisation_blue_shift_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_06_ionisation_blue_shift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photoionisation-induced soliton blue-shift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The dispersive-wave example &lt;plot_00_vuv_dispersive_wave&gt; runs the canonical helium case -- a 7.5 fs, 275 uJ pulse at 800 nm in 1.5 m of 125 um-radius capillary, helium falling from 0.8 bar to vacuum, four HE_1m modes, plasma on -- with the default electronic response, which freezes helium&#x27;s Kerr coefficient at the 800 nm carrier and uses it for every quartet the cubic couples. This example repeats that run, unchanged but for the electronic policy, under the two frequency-dependent responses of /guide/media/dispersive-kerr:">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_07_rdw_dispersive_kerr_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_07_rdw_dispersive_kerr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The vacuum-ultraviolet dispersive wave under a frequency-dependent Kerr response</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 400 nm pump and an 800 nm seed in an argon-filled capillary generate a deep-ultraviolet signal at 266.7 nm by degenerate four-wave mixing, \omega_s = 2\omega_p - \omega_i, the phase-matched source of Durfee et al., Optics Letters 24, 697 (1999), doi:10.1364/OL.24.000697. The waveguide&#x27;s anomalous contribution to the dispersion cancels the gas&#x27;s normal one at one pressure, where 2\beta(\omega_p) = \beta(\omega_i) + \beta(\omega_s).">

.. only:: html

  .. image:: /auto_examples/processes/images/thumb/sphx_glr_plot_08_duv_fwm_dispersive_kerr_thumb.png
    :alt:

  :doc:`/auto_examples/processes/plot_08_duv_fwm_dispersive_kerr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Deep-ultraviolet four-wave mixing in argon under a frequency-dependent Kerr response</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Supercontinuum generation
-------------------------

The four mechanisms of supercontinuum generation in gas-filled hollow-core fibres --
self-phase modulation, soliton self-compression and dispersive-wave emission,
modulational instability, and Raman comb-to-continuum -- with ensembles of noisy shots
for the averaged spectrum, its relative intensity noise and its first-order coherence.
The MI ensemble uses eight seeded shots over 0.4 m; the Raman comb-to-continuum
example uses one seeded shot over 0.4 m and does not estimate ensemble coherence.
The SPM and soliton examples use twenty shots. Measured runtimes and backends are
reported on the individual pages; they depend on the problem and machine.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A supercontinuum is a spectrum broadened by nonlinear propagation over an octave or more. How it is broadened decides its noise and coherence: whether every laser shot produces the same spectrum with the same spectral phase, which is what a pulse compressor or a frequency comb needs, or a spectrum that fluctuates from shot to shot because it grew from quantum noise.">

.. only:: html

  .. image:: /auto_examples/supercontinuum/images/thumb/sphx_glr_plot_01_spm_supercontinuum_thumb.png
    :alt:

  :doc:`/auto_examples/supercontinuum/plot_01_spm_supercontinuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supercontinuum by self-phase modulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In anomalous dispersion, a sufficiently nonlinear short pulse can self-compress and emit a resonant dispersive wave. Its soliton order depends on power, duration, dispersion and nonlinearity, rather than on energy alone. Here compression produces a broad spectrum, and a seeded ensemble measures how repeatable its amplitude and phase are under the chosen input noise.">

.. only:: html

  .. image:: /auto_examples/supercontinuum/images/thumb/sphx_glr_plot_02_soliton_supercontinuum_thumb.png
    :alt:

  :doc:`/auto_examples/supercontinuum/plot_02_soliton_supercontinuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supercontinuum by soliton self-compression, reaching the vacuum ultraviolet</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Modulational instability (MI) amplifies small perturbations on a pulse through the combined action of Kerr nonlinearity and anomalous dispersion. A long pulse can break into shorter structures whose spectra broaden differently in each shot. An average spectrum can therefore look smooth even when individual spectra vary strongly. This example separates these two observations: spectral broadening and shot-to-shot reproducibility.">

.. only:: html

  .. image:: /auto_examples/supercontinuum/images/thumb/sphx_glr_plot_03_modulational_instability_supercontinuum_thumb.png
    :alt:

  :doc:`/auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supercontinuum by modulational instability</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A molecular Raman response transfers energy between optical frequencies separated by rotational or vibrational transition frequencies. Repeated scattering can produce a set of Stokes and anti-Stokes sidebands: a Raman frequency comb. As these components broaden and interact, the gaps can fill to form a continuum. We call this comb-to-continuum (CtC) broadening.">

.. only:: html

  .. image:: /auto_examples/supercontinuum/images/thumb/sphx_glr_plot_04_raman_comb_supercontinuum_thumb.png
    :alt:

  :doc:`/auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comb-to-continuum broadening in hydrogen</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Waveguide physics
-----------------

Pressure-gradient compressors with a coherent capillary wall, mode conversion in
tapers, the vector wall, multimode graded-index fibres, a moving solid guide and a
planar compressor.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A hollow capillary filled with a noble gas is the standard way to take a ten-femtosecond pulse at the few-hundred-microjoule level down to a single optical cycle. The pulse is launched in the anomalous-dispersion regime of the gas-filled guide, where self-phase modulation and anomalous dispersion together make it a higher-order soliton; it then self-compresses, reaching its shortest and most intense point at roughly the fission length, and at that point sheds a resonant dispersive wave in the deep ultraviolet.">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_01_gradient_compressor_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_01_gradient_compressor`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A pressure-gradient soliton compressor</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A capillary whose bore narrows along its length is a second handle on soliton dynamics beside the fill. Narrowing the core raises the intensity for the same energy, which accelerates self-compression, and strengthens the waveguide&#x27;s anomalous dispersion, which moves the zero-dispersion wavelength -- and with it the resonant dispersive wave -- towards shorter wavelengths.">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_02_tapered_capillary_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_02_tapered_capillary`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A tapered capillary: compression, a dispersive wave and mode conversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The launches here are pure capillary modes: linearly polarised HE_11, radially polarised TM_01 and azimuthally polarised TE_01. The latter two have cylindrical components independent of azimuth, but their Cartesian field directions rotate around the axis. A laboratory polarisation converter need not launch a pure radial mode. The two selected transverse families have the same core field shape, the same propagation constant to a part in 10^9, and the same nonlinearity -- and they do not see the same wall. At grazing incidence the glass imposes one condition on the field tangential to it (``TE``) and another on the field normal to it (``TM``), and the TM field loses \varepsilon = n_{\rm glass}^2 times faster. The scalar coherent wall gives the glass one averaged impedance; CoherentWall(impedance=&quot;te_tm&quot;) gives each field its own (/guide/waveguides/vector-wall).">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_03_vector_wall_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_03_vector_wall`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radially and azimuthally polarised pulses: the vector wall over a metre</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A parabolic graded-index fibre provides a useful setting for studying periodic transverse motion and nonlinear spectral broadening. Its mode groups are almost equally spaced in propagation constant, so a beam launched into several of them self-images: the transverse pattern contracts and expands with a period of a fraction of a millimetre, and the intensity on the axis oscillates with it. At high power that periodic oscillation is a longitudinal grating written into the Kerr effect, and it phase-matches geometric parametric instability (GPI): sidebands far from the pump, with the leading-order, weak-nonlinearity estimate for frequency offsets \Omega_h = \sqrt{2\pi h/(z_p\beta_2)}, where z_p is the self-imaging period and \beta_2 the group-velocity dispersion (Krupa et al., Phys. Rev. Lett. 116, 183901 (2016), doi:10.1103/PhysRevLett.116.183901; Wright et al., Nat. Photon. 9, 306 (2015)).">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_04_graded_index_multimode_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_04_graded_index_multimode`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spatiotemporal propagation in a graded-index multimode fibre</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A fibre taper changes the shape of the modes it guides. In a parabolic graded-index core the fundamental mode is, to a very good approximation, a Gaussian whose width squared is proportional to the core radius, w^2 = 2a/(k n_1 \sqrt{2\Delta}); narrow the core from 25 um to 15 um and the mode narrows by \sqrt{0.6}. If the core narrows slowly, the light follows the local mode and stays in it -- the taper is adiabatic. If it narrows quickly, the light keeps the shape it had, which is no longer a mode of the narrower core, and part of it is left in the next mode that shares its symmetry, LP_02. The rule of thumb that separates the two is Love&#x27;s: a taper is adiabatic where its local length scale z_t = a/|da/dz| is longer than the beat length z_b = 2\pi/(\beta_1 - \beta_2) between the two modes (J. D. Love et al., IEE Proc. J 138, 343 (1991); T. A. Birks and Y. W. Li, J. Lightwave Technol. 10, 432 (1992)).">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_05_moving_solid_guide_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_05_moving_solid_guide`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A tapered graded-index fibre: adiabatic and abrupt mode conversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a hollow capillary the bore fixes the beam&#x27;s area. The energy it can compress is therefore bounded by the intensity at which the gas ionises or the entrance is damaged, and a larger bore raises the bound only at the cost of a longer guide. A planar hollow waveguide removes the bound in one direction. Two polished glass plates, a fraction of a millimetre apart, guide the beam in one direction only (x, across the gap). In the other direction (y, along the plates) the beam is free, so it can be made as wide as the energy needs to keep the intensity where self-phase modulation broadens the spectrum and the gas barely ionises. Akturk et al. (Opt. Lett. 34, 1462 (2009)) compressed 12.5 mJ pulses from 45 fs to 13.6 fs this way, with 8.1 mJ leaving the guide, and Arnold et al. (New J. Phys. 12, 073015 (2010)) reached 10.6 mJ at 10.1 fs.">

.. only:: html

  .. image:: /auto_examples/waveguides/images/thumb/sphx_glr_plot_06_planar_compressor_thumb.png
    :alt:

  :doc:`/auto_examples/waveguides/plot_06_planar_compressor`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A planar hollow-waveguide compressor</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/processes/index.rst
   /auto_examples/supercontinuum/index.rst
   /auto_examples/waveguides/index.rst

