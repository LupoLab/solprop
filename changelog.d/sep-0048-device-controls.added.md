Add per-run `SimulationSettings.device` execution selection and forward it through
recipes, runners and spawned scan builders. Native choices must match the installed
build; NumPy and the currently qualified Julia backend accept CPU only. Keep device
and backend options outside scientific and restart identities, preserve recipe
execution options in Python export, and rebuild reused runners when execution choices
change (SEP 48; #703, #705).
