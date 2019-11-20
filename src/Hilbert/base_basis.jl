function shape(h::AbstractBasis)
    h.shape
end

function state(h::AbstractBasis, i::Int)
    σ = state(h)
    return state!(σ, h, i)
end

struct StateIterator{H,S}
    basis::H
    state::S
end

states(h::AbstractBasis) = StateIterator(h, state(h))

Base.length(iter::StateIterator) = spacedimension(iter.basis)
Base.eltype(iter::StateIterator) = eltype(iter.state)
function Base.getindex(iter::StateIterator, i::Int)
    @assert i > 0 && i <= length(iter)
    state!(iter.state, iter.basis, i)
end

function Base.iterate(iter::StateIterator)
    σ = zero!(iter.state)
    return (σ, 1)
end

function Base.iterate(iter::StateIterator, idx)
    if idx >= spacedimension(iter.basis)
        return nothing
    end

    σ = iter.state
    state!(σ, iter.basis, idx + 1)
    return (σ, idx + 1)
end
