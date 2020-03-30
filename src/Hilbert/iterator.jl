"""
    StateIterator{H,S}

An iterator for enumerating all the states in a basis
"""
struct StateIterator{T,H}
    basis::H
end

"""
    states(hilb) -> iterator

Returns an iterator to iterate all states in the hilbert space
"""
states(T::Type{<:Number}, h::AbstractHilbert) = StateIterator{T, typeof(h)}(h)
states(h::AbstractHilbert) = states(STD_REAL_PREC, h)

Base.length(iter::StateIterator) = spacedimension(iter.basis)
Base.eltype(iter::StateIterator) = typeof(state(iter.basis))
Base.getindex(iter::StateIterator{T}, i::Int) where T =
    state_i(T, iter.basis, i)

function Base.iterate(iter::StateIterator{T}, idx = 1) where T
    idx > spacedimension(iter.basis) && return nothing

    @inbounds σ = state_i(T, iter.basis, idx)
    return (σ, idx + 1)
end
