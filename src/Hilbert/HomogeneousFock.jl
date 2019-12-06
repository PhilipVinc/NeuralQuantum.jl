export HomogeneousFock

mutable struct HomogeneousFock{D} <: AbstractHilbert
    n_sites::Int
    shape::Vector{Int}
end

"""
    HomogeneousFock(N, m)

Constructs the Hilbert space of `N` identical modes with `m` levels.
"""
HomogeneousFock(n_sites, hilb_dim) =
    HomogeneousFock{hilb_dim}(n_sites, fill(hilb_dim, n_sites))

@inline nsites(h::HomogeneousFock) = h.n_sites
@inline local_dim(h::HomogeneousFock{D}) where D = D
@inline local_dim(h::HomogeneousFock{D}, i) where D = D
@inline shape(h::HomogeneousFock) = h.shape

@inline spacedimension(h::HomogeneousFock) = local_dim(h)^nsites(h)
@inline indexable(h::HomogeneousFock) = spacedimension(h) != 0
@inline is_homogeneous(h::HomogeneousFock) = true

state(arrT::AbstractArray, T::Type{<:Number}, h::HomogeneousFock) = similar(arrT, T, nsites(h)) .= 0.0

Base.show(io::IO, ::MIME"text/plain", h::HomogeneousFock) =
    print(io, "Hilbert Space with $(nsites(h)) identical sites of dimension $(local_dim(h))")

Base.show(io::IO, h::HomogeneousFock) =
    print(io, "HomogeneousFock($(nsites(h)), $(local_dim(h)))")


## Operations

function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousFock{N}, i) where N
    T = eltype(σ)

    old_val = σ[i]
    #new_val = T(rand(rng, 0:(N-2)))
    new_val =  floor(rand(rng, T)*N)
    σ[i]    = new_val + (new_val >= old_val)
    return old_val, new_val
end

# special case N== 2 to be faster
function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousFock{2}, i)
    T = eltype(σ)

    old_val = σ[i]
    new_val = old_val == 0.0 ? 1.0 : 0.0
    σ[i]    = new_val

    return old_val, new_val
end

function setat!(σ::AState, h::HomogeneousFock, i::Int, val)
    old_val = σ[i]
    σ[i] = val

    return old_val
end

set_index!(σ::AState, h::HomogeneousFock, i) = set!(σ, h, i)
function set!(σ::AState, h::HomogeneousFock{N}, val::Integer) where N
    @assert val > 0 && val <= spacedimension(h)
    val -= 1

    for i=1:nsites(h)
        val, σ[i] = divrem(val, N)
    end
    return σ
end

add!(σ::AState, h::HomogeneousFock, val::Integer) =
    set!(σ, h, val+toint(σ, h))

function Random.rand!(rng::AbstractRNG, σ::Union{AState,AStateBatch}, h::HomogeneousFock{N}) where N
    T = eltype(σ)

    #rand!(rng, σ, 0:(N-1))
    rand!(rng, σ)
    σ .*= N
    σ .= floor.(σ)
    return σ
end

function toint(σ::AState, h::HomogeneousFock{N}) where N
    tot = 0
    for (i,v)=enumerate(σ)
        tot += Int(v)*N^(i-1)
    end
    return tot + 1
end

local_index(σ::AState, h::HomogeneousFock, site)= Int(σ[site])+1

function local_index(σ::AState, h::HomogeneousFock{M}, sites::AbstractVector) where M
    idx = 1
    for (i,j)=enumerate(sites)
        idx += Int(σ[j]) * M^(i-1)
    end
    return idx
end
