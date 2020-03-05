export row, col
export add!, zero!, apply!, apply
export setat!, set!, set_index!, rand!

const AState{T}         = AbstractVector{T}  where T
const AStateBatch{T}    = AbstractMatrix{T}  where T
const AStateBatchVec{T} = AbstractArray{T,3} where T

const AbstractDoubled{T,N}    = NTuple{2,AbstractArray{T,N}} where {T,N}
const ADoubleState{T}         = AbstractDoubled{T,1}         where T
const ADoubleStateBatch{T}    = AbstractDoubled{T,2}         where T
const ADoubleStateBatchVec{T} = AbstractDoubled{T,3}         where T

const AbstractState{T}         = Union{AState{T}, ADoubleState{T}}                 where T
const AbstractStateBatch{T}    = Union{AStateBatch{T}, ADoubleStateBatch{T}}       where T
const AbstractStateBatchVec{T} = Union{AStateBatchVec{T}, ADoubleStateBatchVec{T}} where T

const AStateOrBatch{T}       = Union{AState{T},AStateBatch{T}} where T
const ADoubleStateOrBatch{T} = Union{ADoubleState{T},ADoubleStateBatch{T}} where T

const AStateOrBatchOrVec{T} = Union{AState{T},AStateBatch{T},AStateBatchVec{T}} where T
const ADoubleStateOrBatchOrVec{T} = Union{ADoubleState{T},ADoubleStateBatch{T},ADoubleStateBatchVec{T}} where T

"""
    row(v::ADoubleState(orBatch,orVec)) = first(v)

Returns the row indices of the state `v`.
A doubled state is implemented as a tuple, so this effectively returns the
first (subject to change) element of the tuple.
"""
@inline row(v::ADoubleStateOrBatchOrVec) = first(v)

"""
    col(v::ADoubleState(orBatch,orVec)) = last(v)

Returns the column indices of the state `v`.
A doubled state is implemented as a tuple, so this effectively returns the
last (subject to change) element of the tuple.
"""
@inline col(v::ADoubleStateOrBatchOrVec) = last(v)

## utilities similar to base size

"""
    batch_size(σ::BatchedStateOrVec) -> Int

Returns the batch size of an arbitrary state.
This is the second dimension of the array (but works also on doubled states).
"""
batch_size(σ::Union{AStateBatch,AStateBatchVec}) = size(σ, 2)
batch_size(σ::Union{ADoubleStateBatch,ADoubleStateBatchVec}) = batch_size(row(σ))

"""
    chain_length(σ::BatchStateVec)

Returns the length of a chain of states.
This is the third dimension of the array (but works also on doubled states).
"""
chain_length(σ::Union{AStateBatchVec}) = size(σ, 3)
chain_length(σ::Union{ADoubleStateBatchVec}) = chain_length(row(σ))

"""
    state_size(σ) -> tuple

Works as `size(σ)` returning the size of an array of states, with the two
following differences:
    - It drops the first dimension (as it is the dimension of the hilbert
    space and is not relevant);
    - It also works on tuples (Doubled States).
"""
state_size(σ::AbstractState) = tuple()
state_size(σ::AbstractStateBatch) = (batch_size(σ),)
state_size(σ::AbstractStateBatchVec) = (batch_size(σ),chain_length(σ))

# statesimilar
"""
    state_similar(σ, [dims...])

Works as `similar(σ, dims...)` but
    - does not change the first dimension;
    - Works with tuples (Doubled states).
"""
state_similar(σ::AbstractArray, dims::Int...) = similar(σ, size(σ,1), dims...)
state_similar(σ::AbstractArray, dims::Dims)   = similar(σ, size(σ,1), dims...)
state_similar(σ::AbstractArray)               = similar(σ)

# Special case for doubled states
state_similar(σ::AbstractDoubled, dims::Vararg{T,N}) where {T,N} =
    (state_similar(row(σ), dims...), state_similar(col(σ), dims...))

# Allocating
statecopy(σ) = statecopy!(state_similar(σ), σ)
statecopy(σ, σp, mask) = statecopy!(statecopy(σ), σp, mask)
statecopy_invertmask(σ, σp, mask) = statecopy_invertmask(statecopy(σ), σp, mask)

"""
    statecopy!(σp, σ, [mask=nothing])

Copies the state `σ` onto `σp`.
Equivalent to `σp .= σ` in most cases, but if the state is a tuple (double state)
maps the operation on every element of the tuple.

If the BitMask `mask` is passed, only the elements where mask has 1s are copied
over, while the others are left unchanged.
"""
@inline statecopy!(σp::S, σ::Sp) where {T,N,S<:AbstractArray{T,N}, Sp<:AbstractArray{T,N}}= copyto!(σp, σ)
@inline statecopy!(σp::S2, σ::S) where {T,N,S<:AbstractArray{T,N},S2<:NTuple{2,AbstractArray{T,N}}} =
    statecopy!(σp, (σ,σ))
@inline function statecopy!(σp::S, σ::S2) where {T,N,N2,S<:NTuple{2,AbstractArray{T,N}},S2<:NTuple{2,AbstractArray{T,N2}}}
    statecopy!(row(σp), row(σ))
    statecopy!(col(σp), col(σ))
    return σp
end

@inline statecopy!(σp::AStateBatch, σ::AState) = σp .= σ
@inline statecopy!(σp::AStateBatchVec, σ::AStateBatch) = σp .= σ
@inline statecopy!(σp::ADoubleStateBatch{T}, σ::AState{T}) where T =
    statecopy!(σp, (σ, σ))
@inline statecopy!(σp::ADoubleStateBatchVec{T}, σ::AStateBatch{T}) where T =
    statecopy!(σp, (σ, σ))
@inline function statecopy!(σp::Union{AState, AStateBatch, AStateBatchVec}, σ::Union{AState, AStateBatch, AStateBatchVec}, mask)
    σp .= σ .* mask .+ σp .* .! mask
    return σp
end
function statecopy!(σp::ADoubleStateOrBatchOrVec,
           σ::ADoubleStateOrBatchOrVec, mask)
    statecopy!(row(σp), row(σ), mask)
    statecopy!(col(σp), col(σ), mask)
    return σp
end


"""
    statecopy_invertmask!(σp, σ, mask)

Copies the state `σ` onto `σp`, but only the elements where `mask` has 0s.
Equivalent to `statecopy!(σp, σ, !mask)`.
"""
@inline function statecopy_invertmask!(σp::Union{AState, AStateBatch, AStateBatchVec}, σ::Union{AState, AStateBatch},
                   mask)
    σp .= σ .* .! mask .+ σp .* mask
    return σp
end

function statecopy_invertmask!(σp::Union{ADoubleState, ADoubleStateBatch, ADoubleStateBatchVec},
           σ::Union{ADoubleState, ADoubleStateBatch, ADoubleStateBatchVec}, mask)
    statecopy_invertmask!(row(σp), row(σ), mask)
    statecopy_invertmask!(col(σp), col(σ), mask)
    return σp
end

# TODO add @inbounds
"""
    unsafe_get_el(state, [batch], el)

Given a vector of batches of states with size `size(state) = [:, batches, els]`,
take the batch group `el`, and if specified also selects one single batch.

It's somewhat equivalent to a `view`, but handles tuples of states for density
matrices correctly and uses unsafe views to prevent allocation on CPU.
"""
@inline unsafe_get_el(σ::AStateBatch, i)    = uview(σ, :, i)
@inline unsafe_get_el(σ::AStateBatchVec, i) = uview(σ, :, :, i)
@inline unsafe_get_el(σ::AStateBatchVec, batch, el) = uview(σ, :, batch, el)

@inline unsafe_get_el(σ::AbstractDoubled, i::Vararg{T,N}) where {T,N} =
    (unsafe_get_el(row(σ), i...), unsafe_get_el(col(σ), i...))
