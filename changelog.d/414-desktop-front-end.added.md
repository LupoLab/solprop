The desktop front end, `solprop-gui`, beyond its skeleton:
- setups on every route are made from solprop's parameter metadata and examples, with
  registered types edited in their own fields, design targets pinned, and scans;
- each setup is checked as it is edited;
- runs are run, stopped and resumed in separate processes with live views, or in a
  service process that keeps them going when the window closes;
- results are opened, browsed plane by plane, compressed and compared, with figures
  drawn by `solprop.plotting`;
- every setup saves as a recipe or exports as a Python script.

Its design is `docs/dev/gui/design.md`.
