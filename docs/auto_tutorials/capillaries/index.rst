

.. _sphx_glr_auto_tutorials_capillaries:

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


.. toctree::
   :hidden:

   /auto_tutorials/capillaries/plot_03_gas_flow
   /auto_tutorials/capillaries/plot_04_air_mixture
   /auto_tutorials/capillaries/plot_05_envelope_plasma
   /auto_tutorials/capillaries/plot_06_vector_raman
   /auto_tutorials/capillaries/plot_07_coherent_wall
   /auto_tutorials/capillaries/plot_08_coherent_wall_taper
   /auto_tutorials/capillaries/plot_09_vector_wall
   /auto_tutorials/capillaries/plot_10_low_call_stepping
   /auto_tutorials/capillaries/plot_11_dispersive_kerr

