# Species data

A gas species is described by a {class}`~solprop.physdata.species.Species`: its linear
susceptibility, its Kerr hyperpolarisability, its ionisation data, its Raman response if
it has one, and the equation of state that turns a partial pressure and a temperature
into a number density. Species are the members of a gas mixture
([gas mixtures](../guide/media/mixtures.md)).

The factory `solprop.physdata.species("Ar")` builds a built-in gas from optmat: its
linear susceptibility, with optmat's equation of state and transport properties
({class}`~solprop.physdata.species.OptmatGas`), its Kerr model, evaluated at a run's
carrier by {meth}`~solprop.physdata.species.Species.kerr_coefficient`
({class}`~solprop.physdata.nonlinear_data.OptmatKerr`), and its first ionisation energy
({func}`~solprop.physdata.species.ionisation_potential`); its rate and Raman response
are optmat's, resolved by the drivers. `selection=` (a
{class}`~solprop.physdata.selection.MaterialSelection`) names another of optmat's
models (see [material selection](../guide/media/material-selection.md)).
A `Species` passed to it is returned
unchanged, so any function that takes a species also takes a custom one built from the
classes below: for example a {class}`~solprop.physdata.species.TabulatedSusceptibility`
of measured refractivity and absorption, with the {class}`~solprop.physdata.species.IdealGas`
equation of state, and its own Raman constants as a
{class}`~solprop.physdata.raman_types.MolecularRaman`.

Because the factory has the same name as its module, `solprop.physdata.species` refers
to the function, not the module; the module's classes and its other public functions
are documented on this page. The classes and `ionisation_potential` are importable from
{mod}`solprop.physdata`.

```{eval-rst}
.. autoclass:: solprop.physdata.species.Species
   :members:

.. autoclass:: solprop.physdata.species.EquationOfState
   :members:

.. autoclass:: solprop.physdata.species.IdealGas
   :members:

.. autoclass:: solprop.physdata.species.OptmatGas
   :members:

.. autoclass:: solprop.physdata.species.IonisationData
   :members:

.. autoclass:: solprop.physdata.species.TabulatedSusceptibility
   :members:

.. autofunction:: solprop.physdata.species.ionisation_potential
```
