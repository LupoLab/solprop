A `Local` or `Pool` scan that ends with an exception (an interrupt, a refused worker
admission, a point that conflicts with the file) gives back the points it had claimed,
and a rerun computes any point an earlier scan left `running`, rather than skipping it
for ten minutes and reporting it as running
([#168](https://github.com/LupoLab/solprop/issues/168)).
