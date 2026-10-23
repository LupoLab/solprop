

.. _sphx_glr_auto_examples_processes:

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


.. toctree::
   :hidden:

   /auto_examples/processes/plot_00_vuv_dispersive_wave
   /auto_examples/processes/plot_01_self_phase_modulation
   /auto_examples/processes/plot_02_soliton_self_compression
   /auto_examples/processes/plot_03_four_wave_mixing
   /auto_examples/processes/plot_04_cross_phase_modulation
   /auto_examples/processes/plot_05_raman_self_frequency_shift
   /auto_examples/processes/plot_06_ionisation_blue_shift
   /auto_examples/processes/plot_07_rdw_dispersive_kerr
   /auto_examples/processes/plot_08_duv_fwm_dispersive_kerr

