

.. _sphx_glr_auto_tutorials_basics:

Getting started
---------------

A first run, the pulses a run starts from, reading and plotting a saved run, the
interactive namespace, scans over parameters, how to inspect and compare the
material models a run takes from optmat, and how to tell whether a time window is big
enough.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Check the installation, see which backend a run will use, and propagate a pulse: a two-centimetre run through a helium-filled capillary, read back through the same accessors every later example uses. The tutorials that follow take each part in turn -- pulses, analysing a run, the interactive namespace and scans; then gas-filled capillaries, other waveguides, free space and the compression tools -- and the ../../guide/index says which model to use for what.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_01_package_overview_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_01_package_overview`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Package overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same 30 fs, 1 uJ pulse at 800 nm, built on a field grid and on an envelope grid. The two representations store different things -- a real electric field and a complex envelope about a carrier -- but they describe one pulse, and solprop&#x27;s energy functional normalises both launches to the declared energy. Agreement of sampled curves still depends on bandwidth and temporal resolution.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_02_pulses_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_02_pulses`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A pulse on both grids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Everything a propagation leaves behind is read through one protocol, and everything drawn from it comes from one set of accessors. This example makes a run without a solver -- a Gaussian carried along z by the closed-form phase of pure group-velocity dispersion, which supplies an analytic reference for this example -- saves it, reads it back, and draws the standard figures. Swap the synthetic planes for a real output and nothing else changes.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_03_analysing_a_run_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_03_analysing_a_run`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Analysing a saved run</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This page is written as you would type it at a terminal or in a notebook: one import, then everything is an attribute of sol. The case is the everyday one of a hollow-fibre compressor. A 30 fs, 100 uJ pulse at 800 nm is spectrally broadened in a metre of 125 um capillary filled with argon, and the positively chirped pulse that comes out is compressed with a spectral phase, with glass, and with chirped mirrors and wedges.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_04_interactive_session_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_04_interactive_session`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">An interactive session: design, run, compress</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A reduced version of a resonant-dispersive-wave (RDW) set-up shows how to vary pressure and energy together and collect a compact result from each run. Point indices make the scan reproducible, while shared setup and per-worker reductions control its execution cost. The 125 um capillary and the 7.5 fs, 800 nm pulse are those of the helium RDW case of sphx_glr_introduction_lessons_plot_01_rdw.py, but this scan uses only 2 mm of propagation, one mode and a 200 nm lower wavelength limit to keep its execution short. It is not a converged RDW prediction. The padded grid extends beyond the fitted helium and silica ranges, so MaterialRangeWarning notices are expected. Helium has no Raman response; plasma and third-harmonic generation are explicitly disabled here.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_05_pressure_energy_scan_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_05_pressure_energy_scan`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Scan helium pressure and pulse energy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The default material data for gases, solids, Raman responses and ionisation rates come from the curated library optmat (SEP 3): a run with no materials= takes MaterialSelection(), optmat&#x27;s recommended model of each family. This tutorial shows how to see what a run resolved, how to declare a different material, and how to tell which part of the physics a change touches.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_06_material_choices_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_06_material_choices`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compare material choices</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The time window is periodic: light that walks past one edge comes back at the other. Every run therefore carries a soft absorbing collar just outside the window you ask for (``trange``), and a boundary monitor that watches the outer 5 % of the window at each edge, the guards, and the energy the collar removes. This tutorial uses both to size a window, on a short fibre where a Raman soliton decelerates towards the window&#x27;s late edge.">

.. only:: html

  .. image:: /auto_tutorials/basics/images/thumb/sphx_glr_plot_07_time_window_thumb.png
    :alt:

  :doc:`/auto_tutorials/basics/plot_07_time_window`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Is my time window big enough?</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/basics/plot_01_package_overview
   /auto_tutorials/basics/plot_02_pulses
   /auto_tutorials/basics/plot_03_analysing_a_run
   /auto_tutorials/basics/plot_04_interactive_session
   /auto_tutorials/basics/plot_05_pressure_energy_scan
   /auto_tutorials/basics/plot_06_material_choices
   /auto_tutorials/basics/plot_07_time_window

