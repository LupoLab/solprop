# Selected execution at the PythonCall boundary

`prepare_modal` accepts an optional fourth execution name. CPU remains the default;
CUDA uses the already loaded vendor extension and its first device. The public
backend admission remains CPU-only until the complete CUDA catalogue is qualified.
Spatial device preparation still refuses explicitly. Neither bridge selection nor
device loading adds fields to the scientific descriptor or resume identity.

The Python preparation transaction forwards its selected target to runtime setup
selection and to the Julia factory. CPU preparation retains its shared BLAS lease
and FFTW planning cache. Device preparation acquires neither, and a device start
cannot attach CPU FFTW planning. Constructor failure releases unpublished owners;
deferred interrupts still observe a fully published preparation.

Scalar material, density, taper and decay profiles are evaluated by Julia host
control and retain host tables. Modal prefactors and power weights use execution-aware
constructors to upload their separately owned numerical tables. Linear-operator
tables already follow their own selected execution. Expansion flow scratch is now
allocated through that execution as well. No Python loop evaluates a source or step.

An uploaded starting field is checked for finiteness in its owned stream using one
integer status per column. This avoids an ambient vendor reduction or full-field
download. The common session constructor still copies and hard-masks its owned
initial state. Host input and CPU advancement retain their original numerical graph.
Every session supplies its actual field shape to transactional device-export storage,
including spatial preparations without modal power weights.

Focused Python transaction/BLAS/FFT/group checks pass 84 tests, and changed-file
typing reports no errors or warnings. The launch/source/march engine tests pass
272 bounds-enabled assertions, including 16 checks with a model that refuses host
indexing of its device field. Six real managed Julia CPU cases pass: bounded eight-step
replay, represented operator/prefactor tables and expansions, and exact destination
recipes. Their reference is NumPy; no native-backend qualification is claimed by
this isolated lightweight environment. CPU Aqua/JET passes 16 checks, with all
2,292 definitions analyzed and none cached. Actual CUDA bridge execution remains
pending; the Raman/plasma execution-aware factories and the device service catalogue
are separate integration work.
