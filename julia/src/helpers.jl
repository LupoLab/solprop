# The shared per-stage and per-step evaluations (`solprop._helpers`), and an exponential
# Adams trial's coefficient tables. Each is a transcription of its Python form with the
# same IEEE operations in the same order (Julia contracts nothing without being asked),
# so the values agree with numpy_ref's and native's bit for bit.

"""`chebyshev._clenshaw_points`: a table's rows at several positions, written to `out`.

`c` is (n_intervals, n_terms, n_values) and `out` (n_points, n_values), NumPy arrays
seen through PythonCall. Returns the first position outside `[lower, upper]`, zero
based, or -1.
"""
function clenshaw_points!(z, edges, lower, upper, c, out)
    n, m = size(c, 2), size(c, 3)
    last = length(edges) - 2  # the last interval, zero based
    @inbounds for p in eachindex(z)
        zp = z[p]
        (zp < lower || zp > upper) && return p - 1
        if zp != zp  # NaN sorts last, as in `searchsorted`
            k = last
        else
            # `searchsorted(edges, zp, side="right") - 1`, clamped to an interval.
            low, high = 0, length(edges)
            while low < high
                middle = (low + high) ÷ 2
                if edges[middle+1] <= zp
                    low = middle + 1
                else
                    high = middle
                end
            end
            k = min(max(low - 1, 0), last)
        end
        lo, hi = edges[k+1], edges[k+2]
        xp = (2.0 * zp - lo - hi) / (hi - lo)
        if n == 1
            for j in 1:m
                out[p, j] = c[k+1, 1, j] + 0.0 * xp
            end
        elseif n == 2
            for j in 1:m
                out[p, j] = c[k+1, 1, j] + c[k+1, 2, j] * xp
            end
        else
            x2 = 2 * xp
            for j in 1:m
                c0, c1 = c[k+1, n-1, j], c[k+1, n, j]
                for i in 3:n
                    tmp = c0
                    c0 = c[k+1, n-i+1, j] - c1
                    c1 = tmp + c1 * x2
                end
                out[p, j] = c0 + c1 * xp
            end
        end
    end
    -1
end

"""`multistep._basis_rows`: the Lagrange basis on `points`, written to `rows` (its
coefficients, unless integrating) or `weights` (its integrals from 0 to `upper`)."""
function basis_rows!(points, upper, integrate, rows, weights)
    n = length(points)
    coefficients, product = zeros(n), zeros(n)
    @inbounds for index in 1:n
        node = points[index]
        filled = 1
        coefficients[1] = 1.0
        denominator = 1.0
        for other in 1:n
            other == index && continue
            root = points[other]
            product[1] = 0.0
            for power in 1:filled
                product[power+1] = coefficients[power]
            end
            for power in 1:filled
                product[power] -= root * coefficients[power]
            end
            filled += 1
            for power in 1:filled
                coefficients[power] = product[power]
            end
            denominator *= node - root
        end
        if integrate
            integral = 0.0
            for power in filled:-1:1
                integral = (integral + coefficients[power] / power) * upper
            end
            weights[index] = integral / denominator
        else
            for power in 1:filled
                rows[index, power] = coefficients[power] / denominator
            end
        end
    end
    nothing
end

"""`stepper.multistep.lagrange_coefficients`, unchecked: (n, n)."""
function lagrange(points)
    rows = zeros(length(points), length(points))
    basis_rows!(points, 1.0, false, rows, nothing)
    rows
end

"""`stepper.multistep.integrated_weights` on `[0, 1]`, unchecked: (n,)."""
function integrated(points)
    weights = zeros(length(points))
    basis_rows!(points, 1.0, true, nothing, weights)
    weights
end

"""`backends.multistep.factored_defect_rows`, into `rows` ((count, 2 width), zeroed)."""
function factored_defect_rows!(rows, nodes, orders, width)
    for (index, order) in enumerate(orders)
        points = [1.0; nodes[1:order]]
        for (t, node) in enumerate(points)
            denominator = 1.0
            for (s, other) in enumerate(points)
                s != t && (denominator *= node - other)
            end
            rows[index, t] = 1.0 / denominator
        end
        coefficients = [1.0]
        for root in nodes[1:order]
            product = [0.0; coefficients]
            for (power, value) in enumerate(coefficients)
                product[power] -= root * value
            end
            coefficients = product
        end
        rows[index, width+1:width+order+1] .= coefficients
    end
    rows
end

"""An exponential trial's tables (`backends.multistep.trial_tables`) written straight
into the engine's host tables, then copied to the device: `nodes` newest first, the
step's `order` and the `orders` scored, the step's own first; `slots` the history's
zero-based slots and `rate` the operator's rate weights (compensated only)."""
function configure_adams_nodes!(engine, nodes, order, orders, slots, rate)
    a = engine.work.adams
    h = a.host
    nodes, orders = Vector{Float64}(nodes), Vector{Int}(orders)
    width, count = 1 + maximum(orders), length(orders)
    a.sizes .= (order, width, count)
    h.predictor[1:order, 1:order] .= lagrange(nodes[1:order])
    h.corrector[1:width, 1:width] .= 0.0
    h.corrector[1:order+1, 1:order+1] .= lagrange([1.0; nodes[1:order]])
    h.defects[1:count, 1:2width] .= 0.0
    factored_defect_rows!(view(h.defects, 1:count, 1:2width), nodes, orders, width)
    incoming = Vector{Int}(slots)
    h.slots[1:width-1] .= view(incoming, 1:width-1) .+ 1
    if a.compensated
        h.spans[1:width-1] .= 1.0 .- view(nodes, 1:width-1)
        # Each count's pair once: the step's own is also its defect's.
        weights = Dict(n => (integrated(nodes[1:n]), integrated([1.0; nodes[1:n]]))
                       for n in unique([order; orders]))
        h.cdefects[1:count, 1:width] .= 0.0
        for (index, scored) in enumerate(orders)
            predicted, through = weights[scored]
            h.cdefects[index, 1:scored+1] .= through
            h.cdefects[index, 2:scored+1] .-= predicted
        end
        predictor, corrector = weights[order]
        h.cpredictor[1:order] .= predictor
        h.ccorrector[1:order+1] .= corrector
        h.rate .= Vector{Float64}(rate)
    end
    oncpu(engine.config) || tally!(:upload, length(a.device))
    foreach(copyto!, a.device, a.host)
    nothing
end

const clenshaw_points_b = clenshaw_points!
const basis_rows_b = basis_rows!
const configure_adams_nodes_b = configure_adams_nodes!
