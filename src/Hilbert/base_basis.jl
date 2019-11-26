export local_dimension, spacedimension
export nsites, toint, index

# Global somewhat slow methods
"""
    local_dim(hilbert, i) -> Int

Returns the local dimension of the `hilbert` space on site `i`
"""
@inline local_dim(h::AbstractHilbert, i) = shape(h)[i]

"""
    index(hilb, state)

Converts to an int the `state` of the hilbert space `hilb`
assuming it's indexable.
"""
@inline index(h::AbstractHilbert, s) = toint(s, h)

"""
    state([arrT=Vector], [T=STD_REAL_PREC], hilbert)

Constructs a state for the `hilbert` space with precision `T` and array type
`arrT`.
"""
state(h::AbstractHilbert) = state(STD_REAL_PREC, h)
state(T::Type{<:Number}, h::AbstractHilbert) = state(zeros(1), STD_REAL_PREC, h)
state(arrT::AbstractArray, h::AbstractHilbert) = state(arrT, STD_REAL_PREC, h)


function state(h::AbstractHilbert, i::Int)
    σ = state(h)
    return set!(σ, h, i)
end

function apply!(σ::ADoubleState, h::AbstractHilbert, cngs_l, cngs_r) 
    apply!(row(σ), h, cngs_l)
    apply!(col(σ), h, cngs_r)
    return σ
end

function apply!(σ::ADoubleState, cngs_l::Union{StateChanges,Nothing}) 
    apply!(row(σ), cngs_l)
    return σ
end

function apply!(σ::ADoubleState, (cngs_l, cngs_r)) 
    apply!(row(σ), cngs_l)
    apply!(col(σ), cngs_r)
    return σ
end

function apply!(σ::ADoubleState, cngs_l, cngs_r) 
    apply!(row(σ), cngs_l)
    apply!(col(σ), cngs_r)
    return σ
end

function apply!(σ::AbstractVector, h::AbstractHilbert, cngs) 
    for (site, val)=cngs
        σ[site] = val
    end
    return σ
end

@inline apply!(σ::AbstractVector, cngs::Nothing) = σ
function apply!(σ::AbstractVector, cngs) 
    for (site, val)=cngs
        σ[site] = val
    end
    return σ
end


@inline flipat!(σ, hilb::AbstractHilbert, i::Int) = flipat!(GLOBAL_RNG, σ, hilb, i)

@inline Random.rand!(σ, h::AbstractHilbert) = rand!(GLOBAL_RNG, σ, h)
@inline Random.rand(h::AbstractHilbert) = rand!(GLOBAL_RNG, state(h), h)

"""
    StateIterator{H,S}

An iterator for enumerating all the states in a basis
"""
struct StateIterator{H,S}
    basis::H
    state::S
end

"""
    states(hilb) -> iterator

Returns an iterator to iterate all states in the hilbert space
"""
states(h::AbstractHilbert) = StateIterator(h, state(h))

Base.length(iter::StateIterator) = spacedimension(iter.basis)
Base.eltype(iter::StateIterator) = eltype(iter.state)
function Base.getindex(iter::StateIterator, i::Int)
    @assert i > 0 && i <= length(iter)
    state!(iter.state, iter.basis, i)
end

function Base.iterate(iter::StateIterator)
    σ = set!(iter.state, iter.basis, 1)
    return (σ, 1)
end

function Base.iterate(iter::StateIterator, idx)
    if idx >= spacedimension(iter.basis)
        return nothing
    end

    σ = iter.state
    set!(σ, iter.basis, idx + 1)
    return (σ, idx + 1)
end
