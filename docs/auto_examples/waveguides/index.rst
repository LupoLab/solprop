

.. _sphx_glr_auto_examples_waveguides:

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

   /auto_examples/waveguides/plot_01_gradient_compressor
   /auto_examples/waveguides/plot_02_tapered_capillary
   /auto_examples/waveguides/plot_03_vector_wall
   /auto_examples/waveguides/plot_04_graded_index_multimode
   /auto_examples/waveguides/plot_05_moving_solid_guide
   /auto_examples/waveguides/plot_06_planar_compressor

