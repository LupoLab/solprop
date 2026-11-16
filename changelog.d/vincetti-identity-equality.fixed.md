`VincettiMode` now compares by identity, like every other mode, instead of by value: its
hash no longer changes when the mode logs its one-time gain warning, so a mode kept in
a set or used as a dictionary key is found again afterwards.
