export HomogeneousHilbert

mutable struct HomogeneousHilbert{D} <: AbstractHilbert
    n_sites::Int
    shape::Vector{Int}
end

"""
    HomogeneousHilbert(N, m)

Constructs the Hilbert space of `N` identical modes with `m` levels.
"""
HomogeneousHilbert(n_sites, hilb_dim) =
    HomogeneousHilbert{hilb_dim}(n_sites, fill(hilb_dim, n_sites))

@inline nsites(h::HomogeneousHilbert) = h.n_sites
@inline local_dim(h::HomogeneousHilbert{D}) where D = D
@inline local_dim(h::HomogeneousHilbert{D}, i) where D = D
@inline shape(h::HomogeneousHilbert) = h.shape

@inline spacedimension(h::HomogeneousHilbert) = local_dim(h)^nsites(h)
@inline indexable(h::HomogeneousHilbert) = spacedimension(h) != 0
@inline is_homogeneous(h::HomogeneousHilbert) = true

state(arrT::AbstractArray, T::Type{<:Number}, h::HomogeneousHilbert) = similar(arrT, T, nsites(h)) .= 0.0

Base.show(io::IO, ::MIME"text/plain", h::HomogeneousHilbert) =
    print(io, "Hilbert Space with $(nsites(h)) identical sites of dimension $(local_dim(h))")

Base.show(io::IO, h::HomogeneousHilbert) =
    print(io, "HomogeneousHilbert($(nsites(h)), $(local_dim(h)))")


## Operations

function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousHilbert{N}, i) where N
    T = eltype(σ)

    old_val = σ[i]
    #new_val = T(rand(rng, 0:(N-2)))
    new_val =  floor(rand(rng, T)*N)
    σ[i]    = new_val + (new_val >= old_val)
    return old_val, new_val
end

# special case N== 2 to be faster
function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousHilbert{2}, i)
    T = eltype(σ)

    old_val = σ[i]
    new_val = old_val == 0.0 ? 1.0 : 0.0
    σ[i]    = new_val

    return old_val, new_val
end

function setat!(σ::AState, h::HomogeneousHilbert, i::Int, val)
    old_val = σ[i]
    σ[i] = val

    return old_val
end

set_index!(σ::AState, h::HomogeneousHilbert, i) = set!(σ, h, i)
function set!(σ::AState, h::HomogeneousHilbert{N}, val::Integer) where N
    @assert val > 0 && val <= spacedimension(h)
    val -= 1

    for i=1:nsites(h)
        val, σ[i] = divrem(val, N)
    end
    return σ
end

add!(σ::AState, h::HomogeneousHilbert, val::Integer) =
    set!(σ, h, val+toint(σ, h))

function Random.rand!(rng::AbstractRNG, σ::Union{AState,AStateBatch}, h::HomogeneousHilbert{N}) where N
    T = eltype(σ)

    #rand!(rng, σ, 0:(N-1))
    rand!(rng, σ)
    σ .*= N
    σ .= floor.(σ)
    return σ
end

function toint(σ::AState, h::HomogeneousHilbert{N}) where N
    tot = 0
    for (i,v)=enumerate(σ)
        tot += Int(v)*N^(i-1)
    end
    return tot + 1
end

local_index(σ::AState, h::HomogeneousHilbert, site)= Int(σ[site])+1

function local_index(σ::AState, h::HomogeneousHilbert{M}, sites::AbstractVector) where M
    idx = 1
    for (i,j)=enumerate(sites)
        idx += Int(σ[j]) * M^(i-1)
    end
    return idx
end
