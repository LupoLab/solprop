`StepperSettings` and `PropagatorMode` moved from `solprop.translate.problem` to
`solprop.stepper.settings`, the run-specification layer the run request's stepper
section belongs to: import them from there, or `StepperSettings` from `solprop.stepper`.
`solprop.translate` no longer re-exports `StepperSettings`.
