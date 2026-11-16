The tube-lattice geometry helpers take every argument after the first by keyword, so
two lengths can no longer be swapped silently: `tube_gap(core_radius, *, tube_radius,
n_tubes)`, `core_radius_from_gap(tube_radius, *, n_tubes, gap)`,
`tube_radius_from_gap(core_radius, *, n_tubes, gap)`,
`TubeLattice.from_gap(core_radius, *, gap, n_tubes, wall_thickness)` and
`TubeLattice.from_inner_diameter_ratio(core_radius, *, wall_thickness,
inner_diameter_ratio, n_tubes)`. Name the arguments, for example
`TubeLattice.from_inner_diameter_ratio(20e-6, wall_thickness=300e-9,
inner_diameter_ratio=0.6, n_tubes=6)`.
