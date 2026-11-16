# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The passes of a stage, one named function each.

``docs/dev/architecture.md`` section 8 lists the anatomy of a stage as seven numbered
passes over the data, and ``docs/dev/backends.md`` section 2 says the reference backend
writes each of them as a named function with its formula in the docstring. These
modules are those functions. Nothing here is fused, nothing is written in place and
nothing allocates a buffer it keeps: every function takes arrays and returns a new one,
so a reader can follow one pass without holding the rest of the step in mind. The
native backend fuses these passes and is validated against the numbers they produce.

- :mod:`~solprop.backends.numpy_ref.passes.transforms`: pad and inverse transform,
  synthesis at the nodes, projection onto the modes, forward transform and crop
  (passes 2, 3, 5 and 6);
- :mod:`~solprop.backends.numpy_ref.passes.responses`: the response pass (pass 4),
  its Kerr, dispersive Kerr and Raman stages;
- :mod:`~solprop.backends.numpy_ref.passes.plasma`: its plasma stages, on the field
  grid and cycle-averaged on the envelope grid;
- :mod:`~solprop.backends.numpy_ref.passes.stepping`: the step around the stage --
  the interaction-picture Runge-Kutta trial's stage input, accumulation, acceptance and
  error norm, the stage output every march uses, and the boundary treatments.

The transform and response passes serve every march: the Runge-Kutta trial evaluates
them at the interaction picture's stage fields, and the Adams, exponential and
low-storage engines at lab-frame fields with a unit frame factor
(:meth:`~solprop.backends.numpy_ref.workspace.NumpyWorkspace.lab_source`).

**Scales.** The oversampling **pad** scale is already inside ``transform.synthesis`` and
the **crop** scale is already inside ``responses.prefactor``
(``docs/dev/kernel-contract.md`` sections 4 and 5), so
:func:`~solprop.backends.numpy_ref.passes.transforms.to_time` and
:func:`~solprop.backends.numpy_ref.passes.transforms.to_frequency` scatter and gather
through ``grids.crop_index`` and apply no scale of their own. Using
:func:`solprop.fourier.oversample` and :func:`solprop.fourier.crop` instead would count
both factors twice.

**Convention.** Every complex array is in the FFT convention of
``docs/dev/conventions.md`` section 2; nothing here conjugates. The Raman kernel
spectrum is an internal-convention array whose imaginary part is what distinguishes
gain from loss, so it is never passed through :func:`solprop.fourier.to_optics`
(``docs/guide/media/raman-numerics.md``, "The sign that means gain").

**The one physical constant.** :data:`solprop.constants.EPS0` is read by the Raman pass
alone. Every other constant reaches a backend through a compiled coefficient --
``eps0 chi3`` for the Kerr term, ``e^2/m_e`` for the plasma -- but the Raman response
writes ``eps0`` *outside* the convolution integral, so folding it into a row would
change the stated units of ``raman_kernel_undamped`` and cost the bitwise agreement with
:meth:`solprop.nonlinear.raman.RamanResponse.evaluate` that pins that pass.
"""
