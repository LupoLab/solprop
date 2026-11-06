

.. _sphx_glr_auto_examples_supercontinuum:

Supercontinuum generation
-------------------------

The four mechanisms of supercontinuum generation in gas-filled hollow-core fibres --
self-phase modulation, soliton self-compression and dispersive-wave emission,
modulational instability, and Raman comb-to-continuum -- with ensembles of noisy shots
for the averaged spectrum, its relative intensity noise and its first-order coherence.
The MI ensemble uses eight seeded shots over 0.4 m; the Raman comb-to-continuum
example uses one seeded shot over 0.4 m and does not estimate ensemble coherence.
The SPM and soliton examples use twenty shots. The last example is the canonical
solid-core continuum, in a photonic crystal fibre, from one seeded shot. Measured
runtimes and backends are reported on the individual pages; they depend on the problem
and machine.


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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The canonical solid-core supercontinuum: a 50 fs, 10 kW sech pulse at 835 nm in 15 cm of photonic crystal fibre, pumped in anomalous dispersion close to the zero-dispersion wavelength. The parameters are those of Fig. 3 of the review by Dudley, Genty and Coen, the standard test of a generalised nonlinear Schrödinger equation solver: the fibre&#x27;s dispersion as ten Taylor coefficients about the pump (beta_2 to beta_10), gamma = 0.11 /W/m, self-steepening, and silica&#x27;s Raman response.">

.. only:: html

  .. image:: /auto_examples/supercontinuum/images/thumb/sphx_glr_plot_05_fibre_supercontinuum_thumb.png
    :alt:

  :doc:`/auto_examples/supercontinuum/plot_05_fibre_supercontinuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Supercontinuum in a photonic crystal fibre</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/supercontinuum/plot_01_spm_supercontinuum
   /auto_examples/supercontinuum/plot_02_soliton_supercontinuum
   /auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum
   /auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum
   /auto_examples/supercontinuum/plot_05_fibre_supercontinuum

