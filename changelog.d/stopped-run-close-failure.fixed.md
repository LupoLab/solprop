A run stopped by its own `stop=` flag now raises a failure to close its output, backend
or TDSE archive, instead of returning as a clean stop with the error lost.
