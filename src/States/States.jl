using Base: @propagate_inbounds

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
@inline row(v::ADoubleStateOrBatchOrVec) = @inbounds first(v)

"""
    col(v::ADoubleState(orBatch,orVec)) = last(v)

Returns the column indices of the state `v`.
A doubled state is implemented as a tuple, so this effectively returns the
last (subject to change) element of the tuple.
"""
@inline col(v::ADoubleStateOrBatchOrVec) = @inbounds last(v)

## utilities similar to base size

"""
    batch_size(σ::BatchedStateOrVec) -> Int

Returns the batch size of an arbitrary state.
This is the second dimension of the array (but works also on doubled states).
"""
@inline batch_size(σ::Union{AStateBatch,AStateBatchVec}) = @inbounds size(σ, 2)
@inline batch_size(σ::Union{ADoubleStateBatch,ADoubleStateBatchVec}) = batch_size(row(σ))

"""
    chain_length(σ::BatchStateVec)

Returns the length of a chain of states.
This is the third dimension of the array (but works also on doubled states).
"""
@inline chain_length(σ::AStateBatchVec) = @inbounds size(σ, 3)
@inline chain_length(σ::ADoubleStateBatchVec) = chain_length(row(σ))

"""
    state_size(σ) -> tuple

Works as `size(σ)` returning the size of an array of states, with the two
following differences:
    - It drops the first dimension (as it is the dimension of the hilbert
    space and is not relevant);
    - It also works on tuples (Doubled States).
"""
@inline state_size(σ::AbstractState) = tuple()
@inline state_size(σ::AbstractStateBatch) = (batch_size(σ),)
@inline state_size(σ::AbstractStateBatchVec) = (batch_size(σ),chain_length(σ))

@inline state_length(σ::AbstractState) = 1
@inline state_length(σ) = prod(state_size(σ))

@inline state_eltype(σ::AbstractState) = typeof(σ)
@inline state_eltype(σ::AStateBatch) = @inbounds typeof(σ[:,1])
@inline state_eltype(σ::AStateBatchVec) = @inbounds typeof(σ[:,1,1])

@inline batch_eltype(σ::AStateBatch) = typeof(σ)
@inline batch_eltype(σ::AStateBatchVec) = @inbounds typeof(σ[:,:,1])

@inline state_eltype(σ::AbstractDoubled) = Tuple{state_eltype(row(σ)), state_eltype(col(σ))}
@inline batch_eltype(σ::AbstractDoubled) = Tuple{batch_eltype(row(σ)), batch_eltype(col(σ))}

# statesimilar
"""
    state_similar(σ, [dims...])

Works as `similar(σ, dims...)` but
    - does not change the first dimension;
    - Works with tuples (Doubled states).
"""
state_similar(σ::AbstractArray, dims::Int...) = @inbounds similar(σ, size(σ,1), dims...)
state_similar(σ::AbstractArray, dims::Dims)   = @inbounds similar(σ, size(σ,1), dims...)
state_similar(σ::AbstractArray)               = similar(σ)

# Special case for doubled states
state_similar(σ::AbstractDoubled, dims::Vararg{T,N}) where {T,N} =
    (state_similar(row(σ), dims...), state_similar(col(σ), dims...))

# Allocating
state_copy(σ) = state_copy!(state_similar(σ), σ)
state_copy(σ, σp, mask) = state_copy!(state_copy(σ), σp, mask)
state_copy_invertmask(σ, σp, mask) = state_copy_invertmask(state_copy(σ), σp, mask)

"""
    state_copy!(σp, σ, [mask=nothing])

Copies the state `σ` onto `σp`.
Equivalent to `σp .= σ` in most cases, but if the state is a tuple (double state)
maps the operation on every element of the tuple.

If the BitMask `mask` is passed, only the elements where mask has 1s are copied
over, while the others are left unchanged.
"""
@inline state_copy!(σp::S, σ::Sp) where {T,N,S<:AbstractArray{T,N}, Sp<:AbstractArray{T,N}} =
    copyto!(σp, σ)
@inline state_copy!(σp::AbstractDoubled{T}, σ::AbstractArray{T}) where T =
    state_copy!(σp, (σ,σ))
@inline function state_copy!(σp::S, σ::S2) where {T,N,N2,S<:AbstractDoubled{T,N},S2<:AbstractDoubled{T,N2}}
    state_copy!(row(σp), row(σ))
    state_copy!(col(σp), col(σ))
    return σp
end

@inline state_copy!(σp::AStateBatch, σ::AState) = σp .= σ
@inline state_copy!(σp::AStateBatchVec, σ::AStateBatch) = σp .= σ
@inline function state_copy!(σp::Union{AState, AStateBatch, AStateBatchVec}, σ::Union{AState, AStateBatch, AStateBatchVec}, mask)
    σp .= σ .* mask .+ σp .* .! mask
    return σp
end
function state_copy!(σp::ADoubleStateOrBatchOrVec,
           σ::ADoubleStateOrBatchOrVec, mask)
    state_copy!(row(σp), row(σ), mask)
    state_copy!(col(σp), col(σ), mask)
    return σp
end


"""
    state_copy_invertmask!(σp, σ, mask)

Copies the state `σ` onto `σp`, but only the elements where `mask` has 0s.
Equivalent to `state_copy!(σp, σ, !mask)`.
"""
@inline function state_copy_invertmask!(σp::Union{AState, AStateBatch, AStateBatchVec}, σ::Union{AState, AStateBatch},
                   mask)
    σp .= σ .* .! mask .+ σp .* mask
    return σp
end

function state_copy_invertmask!(σp::Union{ADoubleState, ADoubleStateBatch, ADoubleStateBatchVec},
           σ::Union{ADoubleState, ADoubleStateBatch, ADoubleStateBatchVec}, mask)
    state_copy_invertmask!(row(σp), row(σ), mask)
    state_copy_invertmask!(col(σp), col(σ), mask)
    return σp
end

# TODO add @inbounds
"""
    state_uview(state, [batch], el)

Given a vector of batches of states with size `size(state) = [:, batches, els]`,
take the batch group `el`, and if specified also selects one single batch.

Returns an UnsafeArrays.UnsafeView object to avoid allocating.
"""
@propagate_inbounds state_uview(σ::AStateBatch, i)    = uview(σ, :, i)
@propagate_inbounds state_uview(σ::AStateBatchVec, i) = uview(σ, :, :, i)
@propagate_inbounds state_uview(σ::AStateBatchVec, batch, el) = uview(σ, :, batch, el)

@propagate_inbounds state_uview(σ::AbstractDoubled, i::Vararg{T,N}) where {T,N} =
    (state_uview(row(σ), i...), state_uview(col(σ), i...))
@propagate_inbounds state_uview(σ::AbstractDoubled, j::AbstractRange, i::Vararg{T,N}) where {T,N} =
    (state_uview(row(σ), j, i...), state_uview(col(σ), j, i...))

"""
    state_view(state, [batch], el)

Given a vector of batches of states with size `size(state) = [:, batches, els]`,
take the batch group `el`, and if specified also selects one single batch.
"""
@propagate_inbounds state_view(σ::AStateBatch, i)    = view(σ, :, i)
@propagate_inbounds state_view(σ::AStateBatchVec, i) = view(σ, :, :, i)
@propagate_inbounds state_view(σ::AStateBatchVec, batch, el) = view(σ, :, batch, el)

@propagate_inbounds state_view(σ::AbstractDoubled, i::Vararg{T,N}) where {T,N} =
    (state_view(row(σ), i...), state_view(col(σ), i...))
@propagate_inbounds state_view(σ::AbstractDoubled, j::AbstractRange, i::Vararg{T,N}) where {T,N} =
    (state_view(row(σ), j, i...), state_view(col(σ), j, i...))
