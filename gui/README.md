# solprop-gui

`solprop-gui` is a desktop front end for [solprop](https://solprop.readthedocs.io/en/stable/),
a Python package for ultrafast nonlinear pulse propagation. It sets up, runs and reads
solprop simulations without a script. Setups are described in physics terms: a capillary's
core radius, a gas and its pressure, a pulse's wavelength, duration and energy.

**It is experimental.** Its window, its workflow and the files it writes for itself may
change between releases before 1.0. The recipes and results files it makes are solprop's
own, and keep solprop's guarantees.

## Installing

It installs with solprop's `gui` extra, and starts with one command:

```sh
pip install "solprop[gui]"
solprop-gui
```

It is released at solprop's own version, and requires exactly that version of solprop.
On Linux, Qt needs a few system libraries; on Debian or Ubuntu:

```sh
sudo apt install libegl1 libxkbcommon0 libxcb-cursor0
```

## What it does

- **Home** offers a problem class for each common geometry: gas-filled capillaries,
  antiresonant fibres, radial filaments, Cartesian and line beams, the GNLSE, and
  step-index and graded-index fibres. A recipe or results file of any other route,
  made by a script, opens on a form built from solprop's parameter metadata.
- **Setup** is one sheet per class, checked as you type, with the numbers solprop computes
  for it (a soliton order, a zero-dispersion wavelength) and a preview plot. A number can
  be typed as a target, and solprop solves for the input that gives it.
- **Run** shows the spectrum and the pulse growing along the waveguide, and can stop and
  resume. A service process can keep runs going after the window closes.
- **Analyse** reads the results on pages of the class, all at one plane, with zoom,
  readouts, widths at half maximum and publication figures drawn by `solprop.plotting`.
- **Scans** run a setup over a grid of its parameters, and any point opens as a setup.
- **Design a waveguide** explores a guide's dispersion, loss and phase matching, with no
  pulse and no run.

Every setup saves as a solprop recipe, or shows as the Python script that builds it, and
every solprop results file opens in the window. solprop does all the physics: the window
runs it in separate processes and shows what it returns. It is drawn with Qt for Python
([PySide6](https://doc.qt.io/qtforpython-6/)) and [pyqtgraph](https://www.pyqtgraph.org/).

## Documentation

- [The user guide's chapter on the front end](https://solprop.readthedocs.io/en/stable/guide/front-end/index.html),
  which also opens offline from the window's **Help** menu
- [The solprop manual](https://solprop.readthedocs.io/en/stable/)
- [The front end's design](https://solprop.readthedocs.io/en/latest/dev/gui/design.html)

The source is the
[`gui/` directory of the solprop repository](https://github.com/LupoLab/solprop/tree/main/gui);
issues go to [its tracker](https://github.com/LupoLab/solprop/issues).

## Licence

The program is GPL-3.0-or-later, as solprop is. The user guide's pages and figures, and
the application's icon, are CC-BY-4.0. The window's font, IBM Plex Sans, is under the SIL
Open Font License 1.1. The package carries each licence's text.
