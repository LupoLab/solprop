`stop=` now stops a native `Multistep()` run during its march instead of at its next save
plane: within a native call's bound, 256 attempts or 0.1 s, and the attempt in flight. With
one plane the flag was ignored and the run finished as `complete`. `solprop run` and
`solprop resume`, whose first SIGTERM or Ctrl-C reaches a run through that flag, stop the
same way; they used to report a stop that the file showed had not happened (#519). A signal that comes too late to stop the run now exits with status zero and says the file is complete (#521).
