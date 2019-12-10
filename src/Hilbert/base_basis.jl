export local_dimension, spacedimension
export nsites, toint, index, shape

"""
    nsites(hilbert) -> Int

Returns the number of lattice sites in the Hilbert space.
"""
function nsites end

"""
    shape(hilbert) -> Vector{Int}

Returns a vector containing the local hilbert space dimensions of every
mode in the Hilbert space.

In the case of homogeneous spaces, it is usually more efficient to call
`local_dim`.
"""
function shape end

"""
    spacedimension(hilbert) = prod(shape(hilbert)) -> Int

Returns the total dimension of the vector space `hilbert`.
This is only valid if `indexable(hilbert) == true`, otherwise it's 0.
"""
function spacedimension end

"""
    indexable(hilbert) -> Bool

Returns true if the space is can be indexed with an Int64, which means that htis
tests true when

```
    spacedimension(hilbert) <= typemax(Int64)
```
"""
function indexable end

"""
    is_homogeneous(hilbert) -> Bool

Returns true if the space is homogeneous, that is, if all the modes have the
same local hilbert space dimension.
"""
function is_homogeneous end


# Global somewhat slow methods
"""
    local_dim(hilbert [, i]) -> Int

Returns the local dimension of the `hilbert` space on site `i`.
If the space is homogeneous then specifying `i` is not needed.
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

"""
    apply!(state, changes, [changes_r] )

Apply the changes `changes` to the `state`. If state is a double
state and changes is a tuple, then the first element of the tuple
is the row-changes and the second element is the column-changes.
Optionally the two elements of the tuple can be passed separately.

If the state is double but there is only 1 element of changes,
it's applied to the rows.
"""
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

"""
    flipat!(state, hilb, i) -> (old_val, new_val)

Randomly flips `state[i]` to another available state. Returns the
old value of `state[i]` and the new value. state is changed in-place.
"""
@inline flipat!(σ, hilb::AbstractHilbert, i::Int) = flipat!(GLOBAL_RNG, σ, hilb, i)

"""
    rand!([rng=GLOBAL_RNG], state, hilb)

Generates a random state of hilbert space and stores on the preallocated
state `state`. It can also be a batch of states.
Optionally you can pass the rng.
"""
@inline Random.rand!(σ::AbstractArray, h::AbstractHilbert) = rand!(GLOBAL_RNG, σ, h)
@inline Random.rand!(σ::NTuple{2,<:AbstractArray}, h::AbstractHilbert) = rand!(GLOBAL_RNG, σ, h)

"""
    rand([rng=GLOBAL_RNG], hilb)

Generates a random state of hilbert space on the state.
Optionally you can pass the rng.
"""
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
