

.. _sphx_glr_auto_tutorials_waveguides:

Waveguide models
----------------

Antiresonant fibres, tabulated modes, step-index, graded-index and rectangular guides,
the general modal route for guides that supply their own vector fields, a declared basis
over a split pair, and hollow planar guides per unit width.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare an evacuated 30 um diameter core with a capillary of the same radius. The tube lattice has six tubes of 9 um outer radius and 180 nm wall thickness. The 350-400 nm shaded interval marks the nominal first wall-resonance band; it is a visual guide, not a fitted validity boundary. The plots span 250-1200 nm.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_01_antiresonant_fibre_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_01_antiresonant_fibre`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Propagate through a single-ring antiresonant fibre</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="These rows are one converged JCMsuite finite-element solution of one fibre: 15 um core radius, 180 nm walls, six tubes of 10.38 um outer radius. The CSVs are exact copies of the vacuum and 10 bar argon rows of the reference data set solprop&#x27;s tests use (``tests/data/jcmwave_vincetti.json``), originally read from pyJCMwave&#x27;s library. Their headers identify the entries and source hash. They are not a universal accuracy reference. The gas comparison uses Kingston argon at 293 K. For a downloaded script, place the vacuum CSV &lt;../../tutorials/waveguides/data/fem-vacuum.csv&gt; and argon CSV &lt;../../tutorials/waveguides/data/fem-argon.csv&gt; in a data directory beside it.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_02_tabulated_modes_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_02_tabulated_modes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use tabulated hollow and solid-core modes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The first chart resolves the six lowest vector branches of a weakly guiding fibre: a 2 um core radius, core index 1.47 in a 1.45 cladding. Hybrid cutoffs use the exact equation; 2.405 and 3.832 are weak-guidance limits. The silica strand has a 1.25 um radius and its transverse profile is frozen at 835 nm. A 568 pJ, 50 fs sech pulse propagates for 15 cm with Kerr and silica Raman. The time window is reduced to 1 ps, and no shot noise is added. This is an example of mode-averaged propagation, not a converged supercontinuum prediction.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_03_stepindex_rectangular_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_03_stepindex_rectangular`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Explore exact step-index fibres and hollow rectangular guides</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare a bound hybrid strand mode with a radial TE graded-index mode. The providers return full E/H fields normalised by Poynting flux. This example inspects mode construction; to propagate on such a guide, pass it to prop_waveguide(provider=...) (the general modal route: see sphx_glr_auto_tutorials_waveguides_plot_05_general_modal.py, sphx_glr_auto_tutorials_waveguides_plot_06_graded_index.py and the worked examples).">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_04_vector_mode_providers_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_04_vector_mode_providers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Inspect step-index and graded-index electric fields</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The general modal route propagates a guide&#x27;s own three-component E and H (/guide/waveguides/general-modal). This example builds a weakly guiding step-index fibre, propagates twenty microns of it with prop_waveguide(provider=...), and compares the self-phase modulation it accumulates with the calibrated GNLSE route at the matched Poynting coefficient.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_05_general_modal_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_05_general_modal`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Propagate a vector fibre mode, and check it against the GNLSE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The general modal route (/guide/waveguides/general-modal) takes any provider that solves its own three-component E and H, not only a step-index strand. This example builds the parabolic graded-index sector, checks its two radial channels against the analytic weak-guidance closed form, and propagates the beat between them to the length at which the transverse pattern repeats -- self-imaging.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_06_graded_index_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_06_graded_index`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Self-imaging in a graded-index guide, and a launch declared as a field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A guide&#x27;s channels arrive in whatever basis its provider solved them in. Any unitary combination of a block is as good a basis, but U^H diag(beta) U is diagonal only where the block&#x27;s propagation constants are equal, and every &quot;degenerate&quot; group of a real guide is split by something. Declare a basis over a split pair and the rotated channel amplitudes beat. The general modal route propagates the full coupled linear operator in that basis; see /guide/waveguides/modal-bases.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_07_split_block_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_07_split_block`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A declared basis of a split pair: two bases, one answer</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two silica plates 127 um apart, filled with argon: the planar compressor of Akturk et al. (2009). The guide is modelled uniform in y -- a plane wave in the free direction -- so every power is per metre of y (W/m) and every energy J/m; the intensity is still W/m². The first chart is the modal loss and dispersion of the TE and TM families; the second launches a 0.5 J/m, 100 fs pulse projected from a Gaussian of waist 0.735a onto TE1, TE3 and TE5 and propagates ten fixed steps through 10 cm, to show the per-unit-width labels and the coupled energy.">

.. only:: html

  .. image:: /auto_tutorials/waveguides/images/thumb/sphx_glr_plot_08_planar_thumb.png
    :alt:

  :doc:`/auto_tutorials/waveguides/plot_08_planar`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A hollow planar waveguide, per unit width</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/waveguides/plot_01_antiresonant_fibre
   /auto_tutorials/waveguides/plot_02_tabulated_modes
   /auto_tutorials/waveguides/plot_03_stepindex_rectangular
   /auto_tutorials/waveguides/plot_04_vector_mode_providers
   /auto_tutorials/waveguides/plot_05_general_modal
   /auto_tutorials/waveguides/plot_06_graded_index
   /auto_tutorials/waveguides/plot_07_split_block
   /auto_tutorials/waveguides/plot_08_planar

