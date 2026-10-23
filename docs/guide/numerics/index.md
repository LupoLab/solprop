# Numerical methods and accuracy

A calculation must resolve propagation in z as well as the temporal and spatial
structure of the field. This section explains the linear operator, integration
methods, error control, and absorbing boundaries. Read
[grids and Fourier conventions](../setup/grids.md) alongside it: reducing the
step size cannot repair a time window, bandwidth, or spatial basis that is too
small. For an application of the soliton model, see the
[recurrence tutorial](../../introduction/lessons/plot_02_soliton.rst).

```{toctree}
:maxdepth: 1

accuracy
linear-operator
linear-tables
solver
choosing-a-stepper
adams
fixed-schedules
reproducibility
boundaries
```
