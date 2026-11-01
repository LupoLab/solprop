

.. _sphx_glr_auto_tutorials_free_space:

Free space
----------

Beams in radial and Cartesian geometries, a self-focusing diagnostic, a line beam uniform in
one direction, a planar guide resolved across its width, a weak band calibrated under
the low-storage march, a four-wave-mixing delay scan run through one backend with
statistics-only planes, far-field spectra collected through apertures in a mask
plane, a beam prepared by Kerr self-focusing in a gas gap and coupled into a
capillary, and a transverse window sized with the boundary monitor.

For the full filament and crossing-beam lessons, begin with the
:doc:`/introduction/index`.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The first beam is a weak one: argon at 1.2 bar, 800 nm, 20 fs, 2 nJ in the argon, and a 40 um waist. This reduced calculation uses 128 radial samples within 0.6 mm and propagates 6 mm from a launch focused as in vacuum, 3 mm before the waist. A 2 nJ pulse is far below the critical power for self-focusing.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_01_radial_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_01_radial`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial diffraction and a separate self-focusing diagnostic</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A weak pulsed beam on a Cartesian transverse grid: 800 nm, 30 fs, 1.5 nJ, a 2 mm waist and argon at 4 bar. The energy is the physical energy crossing the entrance plane in the argon (/guide/free-space/flux). The calculation is reduced to a 32 by 40 grid and a 20 mm distance to keep it short.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_02_cartesian_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_02_cartesian`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A Cartesian beam in argon</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A LineGrid propagates a beam that is uniform in y, per metre of y: free space resolved in x alone. Its energies are in J/m and its powers in W/m. With one free transverse dimension Kerr self-focusing does not collapse; the stationary beam is the soliton A0 sech(x/x0), whose peak field obeys A02 = 8 / (3 k02 chi3 x02) whatever its width. Its physical power per metre of y is then 8 n0 eps0 c / (3 k02 chi3 x0) (/guide/free-space/model). Here a long pulse carries that field at its peak through three diffraction lengths in vacuum with an absolute Kerr susceptibility, and at half the power, a field lower by sqrt(2), the same beam diffracts. Ten fixed steps each; the run takes seconds.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_03_line_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_03_line`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A line beam: diffraction and the (1+1)-D Kerr soliton</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The hybrid planar transform: the slab modes of a 127 um argon-filled gap (Akturk et al. 2009) in x, and a periodic, collared y axis. Unlike the slab uniform in y of sphx_glr_auto_tutorials_waveguides_plot_08_planar.py, the beam here has a finite height, so the run integrates over y and reports energy in J, and the model can describe diffraction, whole-beam self-focusing and modulational instability in y, subject to the retained modes, grid, boundary and response assumptions. A 1 mJ, 100 fs pulse (the incident beam&#x27;s physical energy in the argon fill), Gaussian in both directions (``w_x = 0.735a`` projected onto TE1, TE3 and TE5; w_y = 0.5 mm), propagates 10 cm with the Kerr effect under the step controller; the run takes seconds on numpy_ref, the reference backend. numba_cpu and the native backend execute the geometry too.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_04_planar_hybrid_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_04_planar_hybrid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A planar guide resolved in y</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="LowStorageRungeKutta replays a fixed schedule with classical RK4 in the interaction picture, holding three copies of the state where the adaptive marches hold fifteen or more. It has no error estimate, so its accuracy is measured from outside, on the quantity that matters, by three-level Richardson calibration (/guide/numerics/fixed-schedules). Here a 30 fs line beam whose peak field is sqrt(2) times the soliton&#x27;s, twice the soliton power, self-phase modulates over one diffraction length, and the weak band is the light it generates below 690 nm: three parts in ten thousand of the energy. The whole field converges twenty times faster than that band, so a norm scored on the field would call a schedule converged while the band is still wrong. A few seconds on numpy_ref.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_05_low_storage_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_05_low_storage`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calibrating a weak band with the low-storage march</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two equal pulses cross at a small angle in a thin Kerr sample. Where they overlap in space and time the cubic response writes a transient grating, and each beam diffracts from it into a new direction: the self-diffraction signals of degenerate four-wave mixing. Scanning the delay of one beam and recording the signal&#x27;s energy traces their correlation, which is how such measurements characterise pulses.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_06_four_wave_mixing_scan_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_06_four_wave_mixing_scan`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A four-wave-mixing delay scan through one backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A spectrometer behind a nonlinear sample rarely sees the whole beam: the light crosses a distance, or a lens, and part of it enters through a hole. This tutorial records what such detectors measure, at every saved plane of a free-space run, without storing the field there (/guide/results/statistics, Far-field apertures and angular regions):">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_07_far_field_statistics_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_07_far_field_statistics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Far-field spectra through apertures</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A beam focused onto a gas-filled capillary often crosses some centimetres of the same gas first. When it is strong, the Kerr effect along that gap changes the beam before it reaches the entrance. This tutorial asks how much that changes the modes the capillary takes up. It propagates one physically focused beam through a 20 mm argon gap twice, once linearly and once with the Kerr effect. It takes each complete complex radial field at the entrance with field_plane, couples it into the same capillary mode set with couple under the scalar_paraxial model, and launches both through the same 5 mm Kerr-only capillary (/guide/workflows/coupling).">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_08_radial_capillary_coupling_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_08_radial_capillary_coupling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear beam preparation and coupling into a capillary</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A free-space grid ends at a boundary: a radial grid has a hard wall at its outer radius, which reflects, and a Cartesian or line grid is periodic, so light that leaves one side comes back at the other. By default solprop puts no absorber there. It folds a safety ramp into the operator, which removes light only at the grid&#x27;s largest transverse wavenumbers, and its boundary monitor reads the outer 5 % of the grid, the transverse edge. A run whose light reaches the edge warns, and stops at ten times the budget, naming a size to try. This tutorial sizes a radial window with the monitor, checks the result against a much wider window, and then asks for a designed real-space collar instead. It is the spatial companion of /auto_tutorials/basics/plot_07_time_window.">

.. only:: html

  .. image:: /auto_tutorials/free_space/images/thumb/sphx_glr_plot_09_transverse_window_thumb.png
    :alt:

  :doc:`/auto_tutorials/free_space/plot_09_transverse_window`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Is my transverse window big enough?</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/free_space/plot_01_radial
   /auto_tutorials/free_space/plot_02_cartesian
   /auto_tutorials/free_space/plot_03_line
   /auto_tutorials/free_space/plot_04_planar_hybrid
   /auto_tutorials/free_space/plot_05_low_storage
   /auto_tutorials/free_space/plot_06_four_wave_mixing_scan
   /auto_tutorials/free_space/plot_07_far_field_statistics
   /auto_tutorials/free_space/plot_08_radial_capillary_coupling
   /auto_tutorials/free_space/plot_09_transverse_window

