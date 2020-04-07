export HomogeneousSpin

mutable struct HomogeneousSpin{D,S,C} <: AbstractHilbert
    n_sites::Int
    shape::Vector{Int}

    # Constrain on total spin number in units of S
    constraint::Int
end

"""
    HomogeneousSpins(N, S::Rational=1//2)

Constructs the Hilbert space of `N` identical spins-S (by default 1//2).
"""
function HomogeneousSpin(n_sites, S::Rational=1//2; total_sz::Union{Nothing,Rational,Int}=nothing)
    n_sites = n_sites isa AbstractGraph ? nv(n_sites) : n_sites

    @assert S.den == 2 || S.den == 1

    # N is dimension of local space
    if S.den == 2
        N = S.num +1
    elseif S.den == 1
        N = 2*S.num +1
    end

    constrained = isnothing(total_sz) ?  false : true
    if constrained
        try
            constraint = Int(total_sz * inv(S))
        catch err
            throw(ErrorException("total_sz is not in units of S"))
        end

        if constraint > n_sites
            throw(ErrorException("total_sz is too big"))
        end

        if N == 2
            (n_sites + constraint) % N == 0 || error("total_sz not valid")
        end
    else
        constraint = 0
    end

    return HomogeneousSpin{N,S,constrained}(n_sites, fill(N, n_sites),
                                          constraint)
end

Base.similar(hilb::HomogeneousSpin, N::Int) =
    HomogeneousSpin(N, spin(hilb))

@inline nsites(h::HomogeneousSpin) = h.n_sites
@inline local_dim(h::HomogeneousSpin{D}) where D = D
@inline local_dim(h::HomogeneousSpin{D}, i) where D = D
@inline shape(h::HomogeneousSpin) = h.shape
@inline spin(h::HomogeneousSpin{D,S}) where {D,S} = S

@inline spacedimension(h::HomogeneousSpin) = local_dim(h)^nsites(h)
@inline indexable(h::HomogeneousSpin) = spacedimension(h) != 0
@inline is_homogeneous(h::HomogeneousSpin) = true

@inline is_contrained(h::HomogeneousSpin{H,S,C}) where {H,S,C} = C
@inline constraint_limit(h::HomogeneousSpin) = h.constraint
@inline magnetization(h::HomogeneousSpin{H,S,false}) where {H,S} = false
@inline magnetization(h::HomogeneousSpin{H,S,true}) where {H,S} = S*h.constraint

state(arrT::AbstractArray, T::Type{<:Number}, h::HomogeneousSpin{N}, dims::Dims) where {N} =
    similar(arrT, T, nsites(h), dims...) .= -(N-1)

Base.show(io::IO, ::MIME"text/plain", h::HomogeneousSpin{N}) where N = begin
    print(io, "Hilbert Space with $(nsites(h)) identical spins $(N-1)/2 of dimension $(local_dim(h))")
    is_contrained(h) && print(io, " with constraint Sz=$(constraint_limit(h)*spin(h))")
end

Base.show(io::IO, h::HomogeneousSpin) =
    print(io, "HomogeneousSpin($(nsites(h)), $(local_dim(h)))")


## Operations

function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousSpin{N}, i) where N
    @boundscheck checkbounds(σ, i)
    T = eltype(σ)

    @inbounds old_val = σ[i]
    new_val = T(rand(rng))
    new_val = floor(new_val*(N-1))*2 - (N-1)
    @inbounds σ[i]    = new_val + T(2) * (new_val >= old_val)
    return old_val, new_val
end

# special case N== 2 to be faster
function flipat!(rng::AbstractRNG, σ::AState, h::HomogeneousSpin{2}, i)
    @boundscheck checkbounds(σ, i)
    T = eltype(σ)

    @inbounds old_val = σ[i]
    new_val = old_val == one(T) ? -one(T) : one(T)
    @inbounds σ[i]    = new_val

    return old_val, new_val
end

function setat!(σ::AState, h::HomogeneousSpin, i::Int, val)
    @boundscheck checkbounds(σ, i)

    @inbounds old_val = σ[i]
    @inbounds σ[i] = val

    return old_val
end

set_index!(σ::AState, h::HomogeneousSpin, i) = set!(σ, h, i)
function set!(σ::AState, h::HomogeneousSpin{N}, val::Integer) where N
    @boundscheck checkbounds_hilbert(h, val)

    val -= 1
    for i=1:nsites(h)
        val, tmp = divrem(val, N)
        @inbounds σ[i] = tmp * 2 - (N-1)
    end
    return σ
end

Base.@propagate_inbounds add!(σ::AState, h::HomogeneousSpin, val::Integer) =
    set!(σ, h, val+toint(σ, h))

function Random.rand!(rng::AbstractRNG, σ::AbstractArray, h::HomogeneousSpin{N,S,false}) where {N,S}
    T = eltype(σ)
    rand!(rng, σ)
    σ .= floor.(σ.*N).*2 .- (N-1)
    return σ
end

# Specialized for constrained fock spaces
function Random.rand!(rng::AbstractRNG, σ::AState, hilb::HomogeneousSpin{N,S,true}) where {N,S}
    T = eltype(σ)

    if N == 2
        m = constraint_limit(hilb) # magnetization in units of S
        nup   = (nsites(hilb) + m) ÷ 2
        ndown = (nsites(hilb) - m) ÷ 2

        @inbounds uview(σ, 1:nup) .= one(T)
        @inbounds uview(σ, nup+1:ndown+nup) .= -one(T)
        shuffle!(rng, σ)
    else
        throw(ErrorException("not implemented!"))
    end

    return σ
end

function toint(σ::AState, h::HomogeneousSpin{N}) where N
    tot = 0
    for (i,v)=enumerate(σ)
        tot += Int((v + (N-1))/2)*N^(i-1)
    end
    return tot + 1
end

function local_index(σ::AState, h::HomogeneousSpin{N}, site) where N
    @boundscheck checkbounds(σ, site)

    @inbounds id = Int((σ[site] + (N-1))/2)+1
    return id
end

function local_index(σ::AState, h::HomogeneousSpin{M}, sites::AbstractVector) where M
    @boundscheck checkbounds(σ, sites)

    idx = 1
    for (i,j)=enumerate(sites)
        @inbounds idx += Int((σ[j] + (M-1))/2) * M^(i-1)
    end
    return idx
end
