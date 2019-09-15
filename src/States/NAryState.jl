"""
    NAryState{T <: Number, BT <:Unsigned} <: DualValuedState
"""
mutable struct NAryState{T <: Number, N} <: FiniteBasisState
    n::Int                  # number of elements
    σ::Vector{T}            # the states as 0/1
    i_σ::Int64              # the number corresponding to σ_binary

    space_dim::Int64        # N^Nsites
end
NAryState{T,N}(n, i_σ=0) where {T, N} = NAryState(T, N, n, i_σ)

NAryState(N::Int, n, args...) = NAryState(Float32, N, n, args...)
function NAryState(T::Type, N::Int, n, i_σ=0)
    @assert N>= 0
    @assert i_σ >= 0

    NAryState{T,N}(n,
              IntegerToState!(Vector{T}(undef, n), N, i_σ),
              i_σ,
              N^n)
end

export NAryState

# custom accessors
@inline config(state::NAryState) = state.σ

## Property Accessors
@inline spacedimension(st::NAryState) = st.space_dim
@inline nsites(state::NAryState) = state.n
@inline local_dimension(state::Type{NAryState{T,N}}) where {T,N} = N
@inline local_dimension(state::NAryState{T,N}) where {T,N} = local_dimension(typeof(state))
@inline eltype(state::NAryState{T,N}) where {T,N} = T

@inline toint(state::NAryState) = state.i_σ
@inline index(state::NAryState) = toint(state)+1
@inline index_to_int(state::NAryState, id) = id - 1

# checks
same_basis(s1::NAryState, s2::NAryState)  =
    nsites(s1)==nsites(s2) && local_dimension(s1) == local_dimension(s2) && eltype(s1) == eltype(s2)

# Operations on tuples
@inline _toint(left::NAryState{T,N}, right::NAryState{T,N}) where {T,N} =
    toint(left) * spacedimension(right) + toint(right)
#@inline index(row::NAryState, col::NAryState) = toint(row, col) + 1
#=function index_to_int(id, σrow::NAryState{T,N}, σcol::NAryState{T,N}) where {T,N}
    i = id - 1
    row = div(i, spacedimension(σrow)) #div(i, N^nsites(σrow))
    col = i - row * spacedimension(σrow)
    return (row, col)
end=#



# Operations
"""
    flipat!(rng, state, site) -> (old_val, new_val)

Flips `state[site]` to a random new state. The old value is returned together
with the new value.
"""
function flipat!(rng::AbstractRNG, state::NAryState{T, N}, i::Int) where {T, N}
    old_val = state.σ[i]

    # Generate new value
    new_val = T(rand(rng, 0:(N-1)))
    while new_val == old_val
        new_val = T(rand(rng, 0:(N-1)))
    end

    # Modify
    state.i_σ += (Int(new_val)-Int(old_val))*N^(i-1)
    state.σ[i] = new_val

    return (old_val, new_val)
end

# For Nary states the fast method is equivalent to the standard one
flipat_fast!(rng::AbstractRNG, state::NAryState, i::Int) =
    flipat!(rng, state, i)

"""
    setat!(state, site, value) -> old_val

Sets `state[site] = value`. Returns the old value of `state[site]`
"""
function setat!(state::NAryState{T, N}, i::Int, val::T) where {T, N}
    old_val = state.σ[i]

    state.i_σ += (Int(val)-Int(old_val))*N^(i-1)
    state.σ[i] = val
    return old_val
end

set_index!(state::NAryState, val::Integer) = set!(state, index_to_int(state, val))
function set!(state::NAryState{T, N}, val::Integer) where {T, N}
    state.i_σ = val
    for i=1:state.n
        val, state.σ[i] = divrem(val, N)
    end
    state
end

function add!(state::NAryState, val::Integer)
    set!(state, val+toint(state))
    state
end

function rand!(rng::AbstractRNG, state::NAryState)
    val = rand(rng, 0:(spacedimension(state)-1))
    set!(state, val)
end

flipped(a::NAryState{T,N}, b::NAryState{T,N}) where {T,N} =
    throw(ErrorException("Flipped for Nary states not implemented"))

IntegerToState(n_sites, loc_dim, val, T::Type{<:Number}) =
    IntegerToState!(Vector{T}(undef, n_sites), loc_dim, val)

@inline function IntegerToState!(arr, loc_dim, val)
    for i=1:length(arr)
        val, arr[i] = divrem(val, loc_dim)
    end
    arr
end

# --- Indexing function for local spaces used in operators
local_index(s::NAryState, i::T) where {T<:Integer} =
    T(s.σ[i])+1

function local_index(s::NAryState{T,Nb}, is::AbstractVector{T2}) where {T,Nb,T2<:Integer}
    idx = 1
    for (i,j)=enumerate(is)
        idx += T2(s.σ[j]) * Nb^(i-1)
    end
    return idx
end
# -- end

function apply!(state::NAryState, changes::StateChanges)
    for (id, val)=changes
        setat!(state, id, val)
    end
    return state
end


# -------------- Base.show extension for nice printing -------------- #
Base.show(io::IO, ::MIME"text/plain", bs::NAryState) = print(io, "NAryState(",bs.n,") : ", String(bs.σ,false),
                                           " = ", bs.i_σ)
Base.show(io::IO, bs::NAryState) = print(io, bs.i_σ, String(bs.σ,false))

function String(bv::Vector, toInt=true)
    str="["
    for el in bv
        str *= Char(Int(el) + '0')
    end
    str *= ']'
    str
end

Base.show(io::IO, ::MIME"text/plain", bs::NAryState{T,2}) where T = print(io, "NAryState(",bs.n,") : ", StringToSpin(bs.σ,false),
                                           " = ", bs.i_σ)
Base.show(io::IO, bs::NAryState{T,2}) where T = print(io, bs.i_σ, StringToSpin(bs.σ,false))
function StringToSpin(bv::Vector, toInt=true)
    str="["
    for el in bv
        str *= el > 0.0 ? Char('↑') : Char('↓')
    end
    str *= ']'
    str
end
