export SuperOpSpace

abstract type AbstractSuperOpBasis <: DoubledBasis end
@inline nsites(h::AbstractSuperOpBasis) = 2*nsites(physical(h))
@inline spacedimension(h::AbstractSuperOpBasis) = spacedimension(physical(h))^2

state(arrT::AbstractArray, T::Type{<:Number}, h::AbstractSuperOpBasis, dims::Dims) =
    (state(arrT, T, physical(h), dims),
     state(arrT, T, physical(h), dims))

mutable struct SuperOpSpace{H<:AbstractHilbert} <: AbstractSuperOpBasis
    basis::H
end

Base.show(io::IO, ::MIME"text/plain", h::SuperOpSpace) =
    print(io, "Hilbert Space for Superoperators with $(physical(h)))")

Base.show(io::IO, h::SuperOpSpace) =
    print(io, "SuperOpSpace($(physical(h))))")

##

function set!(σ::ADoubleState, h::SuperOpSpace, i::Integer)
    @boundscheck checkbounds_hilbert(h, i)

    hp = physical(h)
    np = nsites_physical(h)
    i -= 1

    i_r = div(i, spacedimension(hp))
    i_c = i - i_r*spacedimension(hp)

    @inbounds set!(col(σ), hp, i_r + 1) #i
    @inbounds set!(row(σ), hp, i_c + 1) #j
    return σ
end

Base.@propagate_inbounds function set!(σ::ADoubleState, h::SuperOpSpace, i_r::Integer, i_c::Integer)
    set!(row(σ), physical(h), i_r)
    set!(col(σ), physical(h), i_c)
    return σ
end

@inline function Random.rand!(rng::AbstractRNG, σ::AbstractDoubled, h::SuperOpSpace)
    rand!(rng, row(σ), physical(h))
    rand!(rng, col(σ), physical(h))
    return σ
end

function toint(σ::ADoubleState, h::SuperOpSpace)
    hp = physical(h)

    left = toint(col(σ), hp) - 1
    right = toint(row(σ), hp) - 1
    return left * spacedimension(hp) + right + 1
end
