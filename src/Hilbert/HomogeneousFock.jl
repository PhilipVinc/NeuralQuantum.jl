export HomogeneousFock

mutable struct HomogeneousFock{D,C} <: AbstractHilbert
    n_sites::Int
    shape::Vector{Int}

    # Constrain on total number of excitations
    n_exc::Int
end

"""
    HomogeneousFock(N, m; excitations=N*m)

Constructs the Hilbert space of `N` identical modes with `m` levels.
If "n_exc" is set, then the total number of excitations is bounded when
generating a state.

!!!note
    If using a sampler that does not respect the symmetries of the system,
    the n_exc will not be respected during sampling.
"""
function HomogeneousFock(n_sites, hilb_dim; excitations = -1)
    n_sites = n_sites isa AbstractGraph ? nv(n_sites) : n_sites

    constrained = excitations >= 0 ? true : false
    if constrained && excitations > n_sites*(hilb_dim-1)
        throw(ErrorException("Constrain is useless: bigger than total number of allowed particles."))
    end
    HomogeneousFock{hilb_dim,constrained}(n_sites, fill(hilb_dim, n_sites),
                                          excitations)
end

Base.similar(hilb::HomogeneousFock, N) =
    HomogeneousFock(N, local_dim(hilb))

@inline nsites(h::HomogeneousFock) = h.n_sites
@inline local_dim(h::HomogeneousFock{D}) where D = D
@inline local_dim(h::HomogeneousFock{D}, i) where D = D
@inline shape(h::HomogeneousFock) = h.shape

@inline spacedimension(h::HomogeneousFock) = local_dim(h)^nsites(h)
@inline indexable(h::HomogeneousFock) = spacedimension(h) != 0
@inline is_homogeneous(h::HomogeneousFock) = true

@inline is_contrained(h::HomogeneousFock{H,C}) where {H,C} = C
@inline constraint_limit(h::HomogeneousFock) = h.n_exc

state(arrT::AbstractArray, T::Type{<:Number}, h::HomogeneousFock, dims::Dims) =
    similar(arrT, T, nsites(h), dims...) .= 0.0

Base.show(io::IO, ::MIME"text/plain", h::HomogeneousFock) =
    print(io, "Hilbert Space with $(nsites(h)) identical sites of dimension $(local_dim(h))")

Base.show(io::IO, h::HomogeneousFock) =
    print(io, "HomogeneousFock($(nsites(h)), $(local_dim(h)))")

## Operations

function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousFock{N}, i) where N
    @boundscheck checkbounds(σ, i)
    T = eltype(σ)

    @inbounds old_val = σ[i]
    new_val =  floor(rand(rng, T)*(N-1))
    new_val = new_val + (new_val >= old_val)
    @inbounds σ[i]    = new_val

    return old_val, new_val
end

# special case N== 2 to be faster
function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousFock{2}, i)
    @boundscheck checkbounds(σ, i)
    T = eltype(σ)

    @inbounds old_val = σ[i]
    new_val = old_val == 0.0 ? 1.0 : 0.0
    @inbounds σ[i]    = new_val

    return old_val, new_val
end

function setat!(σ::AState, h::HomogeneousFock, i::Int, val)
    @boundscheck checkbounds(σ, val)

    @inbounds old_val = σ[i]
    @inbounds σ[i] = val

    return old_val
end

set_index!(σ::AState, h::HomogeneousFock, i) = set!(σ, h, i)
function set!(σ::AState, h::HomogeneousFock{N}, val::Integer) where N
    @boundscheck checkbounds_hilbert(h, val)

    val -= 1
    for i=1:nsites(h)
        @inbounds val, σ[i] = divrem(val, N)
    end
    return σ
end

Base.@propagate_inbounds add!(σ::AState, h::HomogeneousFock, val::Integer) =
    set!(σ, h, val+toint(σ, h))

function Random.rand!(rng::AbstractRNG, σ::AbstractArray, h::HomogeneousFock{N, false}) where N
    T = eltype(σ)

    #rand!(rng, σ, 0:(N-1))
    rand!(rng, σ)
    σ .*= N
    σ .= floor.(σ)
    return σ
end

# Specialized for constrained fock spaces
function Random.rand!(rng::AbstractRNG, σ::AState, h::HomogeneousFock{N, true}) where N
    T = eltype(σ)
    n_max = constraint_limit(h)
    n_sites = nsites(h)

    σ .= zero(eltype(σ))

    # add all constraints one by one
    for i=1:constraint_limit(h)
        # select a (non full) site to which it should be added
        site = rand(rng, 1:n_sites)
        i = 0
        @inbounds while σ[site] == N-1
            i += 1
            site = rand(rng, 1:n_sites)
        end
        @inbounds σ[site] += 1
    end

    return σ
end


function toint(σ::AState, h::HomogeneousFock{N}) where N
    tot = 0
    for (i,v)=enumerate(σ)
        tot += Int(v)*N^(i-1)
    end
    return tot + 1
end

function local_index(σ::AState, h::HomogeneousFock, site)
    @boundscheck checkbounds(σ, site)

    @inbounds id = Int(σ[site])+1
    return id
end

function local_index(σ::AState, h::HomogeneousFock{M}, sites::AbstractVector) where M
    @boundscheck checkbounds(σ, sites)

    idx = 1
    for (i,j)=enumerate(sites)
        @inbounds idx += Int(σ[j]) * M^(i-1)
    end
    return idx
end
