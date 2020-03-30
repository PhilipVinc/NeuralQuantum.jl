export physical
abstract type DoubledBasis <: AbstractHilbert end

@inline physical(h::DoubledBasis)                = h.basis
@inline nsites_physical(h::DoubledBasis)         =
    nsites(physical(h))
@inline spacedimension_physical(h::DoubledBasis) =
    spacedimension(physical(h))
@inline indexable(h::DoubledBasis) = spacedimension(h) != 0
@inline is_homogeneous(h::DoubledBasis) = is_homogeneous(physical(h))

@inline function checkbounds_doubled(σ::ADoubleState, i)
    @inbounds N = length(first(σ)) + length(last(σ))
    i>= 1 && i<= N || Base.throw_boundserror(σ, i)
    return nothing
end

# generalMethods
function flipat!(rng::AbstractRNG, σ::ADoubleState, h::DoubledBasis, i)
    @boundscheck checkbounds_doubled(σ, i)

    hp = physical(h)
    np = nsites_physical(h)

    @inbounds res = i > np ? flipat!(rng, col(σ), hp, i-np) : flipat!(rng, row(σ), hp, i)
    return res
end

function setat!(σ::ADoubleState, h::DoubledBasis, i::Int, val)
    @boundscheck checkbounds_doubled(σ, i)

    hp = physical(h)
    np = nsites_physical(h)

    @inbounds old = i > np ? setat!(row(σ), hp, i-np, val) : setat!(col(σ), hp, i, val)
    return old
end
