using ValueHistories

function ValueHistories.History(iters::Vector, vals::Vector)
    hist = History(eltype(vals), eltype(iters))
    hist.iterations = copy(iters)
    hist.values = copy(vals)
    hist.lastiter = last(iters)
    return hist
end

function Base.real(hist::History)
    it, val = get(hist)
    return History(it, real(val))
end

function Base.imag(hist::History)
    it, val = get(hist)
    return History(it, imag(val))
end
