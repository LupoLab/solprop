# Transport between planes

{func}`solprop.coupling.transport` carries a radial or Cartesian
{class}`~solprop.coupling.plane.FieldPlane` through an ordered sequence of the linear
elements of {mod}`solprop.coupling.elements` and returns the transported plane with its
{class}`~solprop.coupling.transport.TransportReport`, the sequential ledger in joules
([transporting a plane](../guide/workflows/coupling.md#transporting-a-plane-between-planes)).

Because the function has the same name as its module, `solprop.coupling.transport`
refers to the function, not the module; the module's report class is documented on this
page. Both names are importable from {mod}`solprop.coupling`.

```{eval-rst}
.. autoclass:: solprop.coupling.transport.TransportReport
   :members:
```
