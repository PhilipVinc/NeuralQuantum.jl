"""
    states(hilb) -> iterator

Returns an iterator to iterate all states in the hilbert space
"""
states(T::Type{<:Number}, h::AbstractHilbert) = StateIterator{T, typeof(h)}(h)
states(h::AbstractHilbert) = states(STD_REAL_PREC, h)

Base.length(iter::StateIterator) = spacedimension(iter.basis)
Base.eltype(iter::StateIterator) = typeof(state(iter.basis))
function Base.getindex(iter::StateIterator{T}, i::Int) where T
    @assert i > 0 && i <= length(iter)
    state_i(T, iter.basis, i)
end

function Base.iterate(iter::StateIterator{T}, idx = 1) where T
    if idx > spacedimension(iter.basis)
        return nothing
    end

    σ = state_i(T, iter.basis, idx)
    return (σ, idx + 1)
end
