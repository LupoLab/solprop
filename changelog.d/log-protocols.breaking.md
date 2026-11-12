`solprop.log` has two reporter protocols instead of six. `ProgressReporter` now declares all
six calls (`summary`, `started`, `step`, `plane`, `finished`, `interrupted`), each with a
no-op default that a subclass inherits, and `ScanReporter` declares `point_reporter`
(default `None`). `InterruptibleReporter`, `SummarisingReporter`, `PlaneObserver` and
`WatchingScanReporter` are removed: an `isinstance` test against them becomes one against
`ProgressReporter` or `ScanReporter`, and `observe_plane` stays an optional method that a
reporter may define. A reporter that subclasses `NullReporter` or `NullScanReporter` needs
no change. One written from scratch to the four basic calls must now also define `summary`
and `interrupted`, or subclass `NullReporter` (or `ProgressReporter`) to inherit them; a
run refuses it with a `DriverError` before it starts. Likewise a scan reporter must define
`point_reporter` or subclass `NullScanReporter` (or `ScanReporter`). What scripts, the GUI,
the service, scans and the CLI show is unchanged.
