export SuperOpSpace, physical

abstract type AbstractSuperOpBasis <: AbstractHilbert end

@inline physical(h::AbstractSuperOpBasis) = h.basis

@inline nsites(h::AbstractSuperOpBasis) = 2*nsites(physical(h))
@inline spacedimension(h::AbstractSuperOpBasis) = spacedimension(physical(h))^2
@inline indexable(h::AbstractSuperOpBasis) = spacedimension(h) != 0
@inline is_homogeneous(h::AbstractSuperOpBasis) = is_homogeneous(physical(h))

state(arrT::AbstractArray, T::Type{<:Number}, h::AbstractSuperOpBasis) = (state(arrT, T, physical(h)), state(arrT, T, physical(h)))#DoubleState(state(physical(h)))

@inline nsites_physical(h::AbstractSuperOpBasis)         = nsites(physical(h))
@inline spacedimension_physical(h::AbstractSuperOpBasis) = spacedimension(physical(h))

mutable struct SuperOpSpace{H<:AbstractHilbert} <: AbstractSuperOpBasis
    basis::H
end

Base.show(io::IO, ::MIME"text/plain", h::SuperOpSpace) =
    print(io, "Hilbert Space for Superoperators with $(physical(h)))")

Base.show(io::IO, h::SuperOpSpace) =
    print(io, "SuperOpSpace($(physical(h))))")

##
function flipat!(rng::AbstractRNG, σ::ADoubleState, h::SuperOpSpace, i)
    hp = physical(h)
    np = nsites_physical(h)

    res = i > np ? flipat!(rng, row(σ), hp, i-np) : flipat!(rng, col(σ), hp, i)
    return res
end

function setat!(σ::ADoubleState, h::SuperOpSpace, i::Int, val)
    hp = physical(h)
    np = nsites_physical(h)

    old = i > np ? setat!(row(σ), hp, i-np, val) : setat!(col(σ), hp, i, val)
    return old
end

function set!(σ::ADoubleState, h::SuperOpSpace, i::Integer)
    hp = physical(h)
    np = nsites_physical(h)
    i -= 1

    i_r = div(i, spacedimension(hp))
    i_c = i - i_r*spacedimension(hp)

    set!(col(σ), hp, i_r + 1) #i
    set!(row(σ), hp, i_c + 1) #j
    return σ
end

function set!(σ::ADoubleState, h::SuperOpSpace, i_r::Integer, i_c::Integer)
    set!(row(σ), physical(h), i_r)
    set!(col(σ), physical(h), i_c)
    return σ
end

function Random.rand!(rng::AbstractRNG, σ::Union{ADoubleState,ADoubleStateBatch}, h::SuperOpSpace)
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
