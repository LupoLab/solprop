The desktop front end, `solprop-gui`, **experimental** in this release: its window, its
workflow and its files may change without notice before 1.0. Install it with
`pip install "solprop[gui]"` and start it with `solprop-gui`. It sets up, runs and reads
simulations without a script, and has a chapter of its own in the user guide:
- Home offers a problem class for each common geometry (capillary, antiresonant fibre,
  radial filament, Cartesian and line beams, GNLSE, step-index and graded-index fibres),
  every other route on a form built from solprop's parameter metadata, and a waveguide
  designer that needs no pulse;
- each class has one setup sheet, checked as it is edited, with the numbers solprop
  computes for it and a preview plot; a number such as the soliton order can be typed as
  a target, which writes the recipe's `[pins]`;
- runs go in separate processes with live views, stop and resume, or in a service
  process that keeps them going when the window closes;
- results are read on pages of the class, all at one plane, with plot tools (zoom,
  readouts, widths at half maximum, peaks) and publication figures drawn by
  `solprop.plotting`;
- scans run a setup over a grid of its parameters, record bands, spectra and pulses,
  and open any point as a setup;
- every setup saves as a recipe or shows as a Python script.
