

.. _sphx_glr_auto_examples_performance:

Performance
-----------

Running the same physics faster or in less memory: single precision on an Apple GPU,
compared with binary64.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A run can hold its field and evaluate its nonlinear response in binary32 with precision=&quot;single&quot;, which halves the memory of every field-sized buffer and is what an Apple-silicon GPU computes in. Whether a problem keeps its accuracy in binary32 is a property of the problem, so this example runs two of them both ways and compares: the photoionisation blue shift of /auto_examples/processes/plot_06_ionisation_blue_shift, here on an envelope grid with the envelope-grid plasma, and a short radial argon filament on a field grid. Each runs in binary64 on backend=&quot;auto&quot; and in binary32, where &quot;auto&quot; chooses the mlx backend on a Mac with a Metal GPU.">

.. only:: html

  .. image:: /auto_examples/performance/images/thumb/sphx_glr_plot_01_single_precision_thumb.png
    :alt:

  :doc:`/auto_examples/performance/plot_01_single_precision`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Single precision on an Apple GPU</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/performance/plot_01_single_precision

