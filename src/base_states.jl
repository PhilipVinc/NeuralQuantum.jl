export index_to_int, flipped, row, col
export add!, zero!, apply!, apply
export setat!, set!, set_index!, rand!

const AState = AbstractVector
const AStateBatch = AbstractMatrix
const AStateBatchVec{T} = AbstractArray{T,3}

const ADoubleState{T} = NTuple{2,AState{T}} where T
const ADoubleStateBatch{T} = NTuple{2,AStateBatch{T}} where T
const ADoubleStateBatchVec{T} = NTuple{2,AStateBatchVec{T}} where T

const AStateOrBatchOrVec{T} = Union{AState{T},AStateBatch{T},AStateBatchVec{T}} where T
const ADoubleStateOrBatchOrVec{T} = Union{ADoubleState{T},ADoubleStateBatch{T},ADoubleStateBatchVec{T}} where T

@inline row(v::Union{ADoubleState,ADoubleStateBatch,ADoubleStateBatchVec}) = first(v)
@inline col(v::Union{ADoubleState,ADoubleStateBatch,ADoubleStateBatchVec}) = last(v)

"""
    apply!(σ, changes)

Applies the changes `changes` to the `σ`.

If `state isa DoubleState` then single-value changes
are applied to the columns of the state (in order to
compute matrix-operator products). Otherwise it should
be a tuple with changes of row and columns.

If changes is nothing, does nothing.
"""
apply!(σ::AbstractArray, cngs::Nothing) = σ

"""
    apply(state, cngs)

Applies the changes `cngs` to the state `σ`, by allocating a
copy.

See also @ref(apply!)
"""
apply(σ::Union{AState, ADoubleState}, cngs) = apply!(deepcopy(σ), cngs)


# Logic to copy states
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
@inline function statecopy!(σp::Union{AState, AStateBatch}, σ::Union{AState, AStateBatch}, mask)
    σp .= σ .* mask .+ σp .* .! mask
    return σp
end

@inline function statecopy_invertmask!(σp::Union{AState, AStateBatch}, σ::Union{AState, AStateBatch},
                   mask)
    σp .= σ .* .! mask .+ σp .* mask
    return σp
end

function statecopy!(σp::Union{ADoubleState, ADoubleStateBatch},
           σ::Union{ADoubleState, ADoubleStateBatch}, mask)
    statecopy!(row(σp), row(σ), mask)
    statecopy!(col(σp), col(σ), mask)
    return σp
end

function statecopy_invertmask!(σp::Union{ADoubleState, ADoubleStateBatch},
           σ::Union{ADoubleState, ADoubleStateBatch}, mask)
    statecopy_invertmask!(row(σp), row(σ), mask)
    statecopy_invertmask!(col(σp), col(σ), mask)
    return σp
end

# Batches
vec_of_batches(v::AStateBatch, n) = similar(v, size(v)..., n)
vec_of_batches(v::ADoubleStateBatch, n) =
    (similar(row(v), size(row(v))..., n), similar(col(v), size(col(v))..., n))

# TODO add @inbounds
"""
    unsafe_get_el(state, [batch], el)

Given a vector of batches of states with size `size(state) = [:, batches, els]`,
take the batch group `el`, and if specified also selects one single batch.

It's somewhat equivalent to a `view`, but handles tuples of states for density
matrices correctly and uses unsafe views to prevent allocation on CPU.
"""
@inline unsafe_get_el(σ::AStateBatchVec, i) = uview(σ, :, :, i)
@inline unsafe_get_el(σ::AStateBatchVec, batch, el) = uview(σ, :, batch, el)

@inline unsafe_get_el(σ::ADoubleStateBatchVec, i) = (unsafe_get_el(row(σ), i), unsafe_get_el(col(σ), i))
@inline unsafe_get_el(σ::ADoubleStateBatchVec, batch, el) = (unsafe_get_el(row(σ), batch, el), unsafe_get_el(col(σ), batch, el))

batch_size(σ::Union{AStateBatch,AStateBatchVec}) = size(σ, 2)
batch_size(σ::Union{ADoubleStateBatch,ADoubleStateBatchVec}) = batch_size(row(σ))

chain_length(σ::Union{AStateBatchVec}) = size(σ, 3)
chain_length(σ::Union{ADoubleStateBatchVec}) = chain_length(row(σ))
