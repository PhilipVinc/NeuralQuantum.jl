export local_dimension, spacedimension
export nsites, toint, index, shape
export state_i

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

function index(h::AbstractHilbert, s::Union{AStateBatch, AStateBatchVec,ADoubleStateBatch, ADoubleStateBatchVec})
    sz = state_size(s)
    out = zeros(Int, sz...)
    for (i,sᵢ)=enumerate(states(s))
        @inbounds out[i] = index(h, sᵢ)
    end

    return out
end

"""
    state([arrT=Vector], [T=STD_REAL_PREC], hilbert, [i=0])

Constructs a state for the `hilbert` space with precision `T` and array type
`arrT`. By default that's the lowest state, otherwise if the hilbert space it's
indexable you can specify with a second argument.
"""
state(arrT::AbstractArray,  T::Type, h::AbstractHilbert, dims::Integer...) = state(arrT, T, h, dims)
state(arrT::AbstractArray,           h::AbstractHilbert, dims::Integer...) = state(arrT, STD_REAL_PREC, h, dims)
state(                      T::Type, h::AbstractHilbert, dims::Integer...) = state(zeros(0), T, h, dims)
state(                               h::AbstractHilbert, dims::Integer...) = state(STD_REAL_PREC, h, dims)

state(arrT::AbstractArray,  h::AbstractHilbert, dims::Dims) = state(arrT, STD_REAL_PREC, h, dims)
state(   T::Type{<:Number}, h::AbstractHilbert, dims::Dims) = state(zeros(0), T, h, dims)
state(                      h::AbstractHilbert, dims::Dims) = state(STD_REAL_PREC, h, dims)

Base.@propagate_inbounds state_i(h::AbstractHilbert, i::Int) = state_i(STD_REAL_PREC, h, i)
Base.@propagate_inbounds state_i(T::Type{<:Number}, h::AbstractHilbert, i::Int) = set!(state(T, h), h, i)

Base.@propagate_inbounds function apply!(σ::ADoubleState, h::AbstractHilbert, cngs_l, cngs_r) 
    apply!(row(σ), h, cngs_l)
    apply!(col(σ), h, cngs_r)
    return σ
end

function apply!(σ::AbstractVector, h::AbstractHilbert, cngs) 
    @boundscheck checkbounds(σ, site)

    for (site, val)=cngs
        @inbounds σ[site] = val
    end
    return σ
end

"""
    flipat!(state, hilb, i) -> (old_val, new_val)

Randomly flips `state[i]` to another available state. Returns the
old value of `state[i]` and the new value. state is changed in-place.
"""
Base.@propagate_inbounds flipat!(σ, hilb::AbstractHilbert, i::Int) = flipat!(GLOBAL_RNG, σ, hilb, i)

"""
    rand!([rng=GLOBAL_RNG], state, hilb)

Generates a random state of hilbert space and stores on the preallocated
state `state`. It can also be a batch of states.
Optionally you can pass the rng.
"""
@inline Random.rand!(σ::AbstractArray, h::AbstractHilbert) = rand!(GLOBAL_RNG, σ, h)
@inline Random.rand!(σ::NTuple{2,<:AbstractArray}, h::AbstractHilbert) = rand!(GLOBAL_RNG, σ, h)

"""
    rand([rng=GLOBAL_RNG], hilb, [batch_size, chain_length])

Generates a random state of hilbert space on the state.
Optionally you can pass the rng or the batch size/chain length
"""
@inline Random.rand(rng::AbstractRNG, h::AbstractHilbert, dims::Int...) = rand!(rng, state(h, dims...), h)
@inline Random.rand(rng::AbstractRNG, h::AbstractHilbert, dims::Dims)   = rand!(rng, state(h, dims), h)

@inline Random.rand(                  h::AbstractHilbert, dims::Int...) = rand(GLOBAL_RNG, h, dims...)
@inline Random.rand(                  h::AbstractHilbert, dims::Dims)   = rand(GLOBAL_RNG, h, dims)

@inline function checkbounds_hilbert(hilb::AbstractHilbert, i)
    (i >= 1 && i<= spacedimension(hilb)) || Base.throw_boundserror(hilb, i)
    return nothing
end
