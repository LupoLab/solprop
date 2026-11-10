# Julia setup and status

The optional backend's [setup manual](../guide/performance/julia.md) describes installation,
published identities and runtime limitations. These Python functions expose the same
explicit setup and filesystem-only inspection as `solprop julia setup` and
`solprop julia status`. Importing this interface does not initialize Julia.

```{eval-rst}
.. autofunction:: solprop.backends.julia.setup

.. autofunction:: solprop.backends.julia.status

.. autofunction:: solprop.backends.julia.select_setup

.. autoclass:: solprop.backends.julia.SetupRecord

.. autoclass:: solprop.backends.julia.SetupStatus
```
