`solprop.output.RetainingOutput`, the documented protocol for a handler of your own that
takes field-free planes: its one method, `retain(retention, field_shape)`, is called once
per run, after `open` and before any `save`, with the rule resolved against the run's
planes. A run that keeps fewer than every field used to look the method up without
declaring it anywhere (#450).
