:orphan:

Focused demonstrations
======================

These runnable examples isolate a feature or modelling choice. For a paced first
calculation, start with the :doc:`introduction <../introduction/index>`; for full
comparisons and ensembles, use the :doc:`worked examples <../auto_examples/index>`.
The :doc:`user guide <../guide/index>` explains the models and their limitations.

Each page includes a downloadable Python script and notebook. Demonstrations
execute during documentation builds, with unchanged results reused from the
Sphinx-Gallery cache. Their size and runtime follow the physical question; a
reduced calculation should not be treated as a convergence study.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

Gas-filled capillaries
----------------------

Gas flow and mixtures in hollow capillaries, envelope plasma and vector Raman responses,
the coherent wall that couples a capillary's modes (on a straight bore, on a taper, and
with the vector (TE/TM) wall), the low-call integrators and a frequency-dependent Kerr
response.

For the soliton and dispersive-wave lessons, begin with the :doc:`/introduction/index`.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A gradient fill is a steady flow between two gas cells. For laminar flow of an isothermal ideal gas, the mass flow a^4 p dp/dz is the same everywhere, so p^2 is linear in the hydraulic resistance int a^-4 dz (Pennetta et al., APL Photonics 4, 056105 (2019), eq. 6): a narrow section carries most of the pressure drop. An end pumped to vacuum cannot pass more gas than leaves it at the speed of sound, so it chokes at a pressure well above zero. And the laminar law has limits: above a Reynolds number of about 2000 the flow is no longer laminar, and Gebhardt et al. (Opt. Express 29, 6957 (2021)) measured output-beam instabilities from 2040.">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_03_gas_flow_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_03_gas_flow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas flow in a hollow fibre: tapers, choked ends and the flow regime</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 15 um radius, 0.225 m capillary contains dry air at 6 bar and 300 K. The 800 nm input has 30 fs intensity FWHM and 2.2 uJ energy. Nitrogen and oxygen each take optmat&#x27;s recommended molecular rate (``plasma`` by label), and deplete independently under their shared electric field. Argon contributes dispersion and Kerr response but has no ionisation channel in this example.">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_04_air_mixture_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_04_air_mixture`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Separate nitrogen and oxygen ionisation in an air-filled capillary</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Both runs use the same instantaneous argon rate, optmat&#x27;s recommended model (~solprop.nonlinear.optmat_rates.OptmatRate), 800 nm, 60 fs intensity FWHM, 1 mJ pulse energy and a 125 um capillary radius. The 8 um propagation distance keeps this deterministic illustration short. Kerr, Raman and third harmonics are absent. This compares two propagation models using the same atomic rate; it does not establish the rate&#x27;s accuracy for an experiment.">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_05_envelope_plasma_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_05_envelope_plasma`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compare a carrier-resolved pulse with envelope plasma</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare equal-amplitude linear and circular envelopes using hydrogen&#x27;s rotational response. A fixed circular pump has one quarter of the rotational source along its Jones direction. This is a local polarisation comparison, not a pump-Stokes gain ratio or a propagated energy comparison.">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_06_vector_raman_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_06_vector_raman`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Follow vector Raman coherence with a delayed probe</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Marcatili&#x27;s loss is incoherent: every HE_1m channel of a capillary attenuates on its own and the wall never moves power between them. The coherent wall, prop_capillary(..., wall=CoherentWall()), condenses the outgoing Robin boundary condition of the exact radial problem onto the retained channels instead, so the wall becomes one rank-one term that couples every channel to every other with a definite phase (/guide/waveguides/coherent-wall).">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_07_coherent_wall_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_07_coherent_wall`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The coherent wall of a capillary, against the incoherent one</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A capillary whose bore varies along z is written in a basis attached to the local radius, and a moving basis costs a term: the connection">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_08_coherent_wall_taper_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_08_coherent_wall_taper`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mode conversion and loss in a taper with a coherent wall</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scalar coherent wall gives the glass one impedance, the average a linearly polarised HE_1m field sees around the bore. The wall imposes two conditions, not one: at grazing incidence the field tangential to the glass sees d_r E = i k_o E and the field normal to it sees d_r E = i (k_o/n_r^2) E. CoherentWall(impedance=&quot;te_tm&quot;) condenses both (/guide/waveguides/vector-wall). Because the wall is rotationally symmetric the total azimuthal order nu is conserved, and an order has at most two radial families, so the block is one order&#x27;s two families: TE_0m and TM_0m for nu = 0, HE_nu,m and EH_nu,m for nu &gt;= 1.">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_09_vector_wall_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_09_vector_wall`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The vector wall: two impedances, two families, one conversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the nonlinear response is expensive -- a time-dependent Schrödinger equation at every radial node, say -- response evaluations can dominate the run time. The default integrator, Multistep(), evaluates it once a step, reusing the sources of earlier steps (/guide/numerics/adams); RungeKutta(), the interaction-picture DOPRI5 march, evaluates it seven times a step on the reference backend (/guide/numerics/choosing-a-stepper).">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_10_low_call_stepping_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_10_low_call_stepping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fewer source calls: the two integrators</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A gas&#x27;s electronic third-order response is not one number. Its hyperpolarisability depends on all four frequencies of each mixing quartet, and it rises towards the ultraviolet: argon&#x27;s Kerr coefficient at 267 nm is twice its value at 800 nm. The default carrier-kerr policy freezes it at the run&#x27;s carrier and uses that value for every process the cubic drives. Two opt-in policies of SEP 6 evaluate optmat&#x27;s whole fit instead, at the absolute frequencies of every quartet, in one broadband polarisation (/guide/media/dispersive-kerr):">

.. only:: html

  .. image:: /auto_tutorials/capillaries/images/thumb/sphx_glr_plot_11_dispersive_kerr_thumb.png
    :alt:

  :doc:`/auto_tutorials/capillaries/plot_11_dispersive_kerr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A frequency-dependent Kerr response</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

Free space
----------

Beams in radial and Cartesian geometries, a self-focusing diagnostic, a line beam uniform in
one direction, a planar guide resolved across its width, a weak band calibrated under
the low-storage march, a four-wave-mixing delay scan run through one backend with
statistics-only planes, far-field spectra collected through apertures in a mask
plane, and a beam prepared by Kerr self-focusing in a gas gap and coupled into a
capillary.

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


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Design and compression tools
----------------------------

Grating and prism compressors: recovering the stage setting that compresses a chirped
pulse.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Recover known compressor settings from synthetic chirped pulses. No propagation is needed: an inverse pair phase makes a test input and the bounded compressor finds the physical stage setting that removes it. Independent ray and dispersion references, rather than this round trip, establish the phase model in the test suite.">

.. only:: html

  .. image:: /auto_tutorials/tools/images/thumb/sphx_glr_plot_01_compressor_pairs_thumb.png
    :alt:

  :doc:`/auto_tutorials/tools/plot_01_compressor_pairs`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Grating spacing and prism insertion</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_tutorials/basics/index.rst
   /auto_tutorials/capillaries/index.rst
   /auto_tutorials/waveguides/index.rst
   /auto_tutorials/free_space/index.rst
   /auto_tutorials/tools/index.rst

