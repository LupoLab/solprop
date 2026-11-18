`solprop.exceptions.HostMarchWarning` is removed: the native backend steps every run it
admits in its own session and refuses the rest by name, so no configuration issued it.
Delete any import of it, or warning filter that names it, such as
`ignore::solprop.exceptions.HostMarchWarning`.
