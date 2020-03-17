export physical
abstract type DoubledBasis <: AbstractHilbert end

@inline physical(h::DoubledBasis)                = h.basis
@inline nsites_physical(h::DoubledBasis)         =
    nsites(physical(h))
@inline spacedimension_physical(h::DoubledBasis) =
    spacedimension(physical(h))
@inline indexable(h::DoubledBasis) = spacedimension(h) != 0
@inline is_homogeneous(h::DoubledBasis) = is_homogeneous(physical(h))

# generalMethods
function flipat!(rng::AbstractRNG, σ::ADoubleState, h::DoubledBasis, i)
    hp = physical(h)
    np = nsites_physical(h)

    res = i > np ? flipat!(rng, row(σ), hp, i-np) : flipat!(rng, col(σ), hp, i)
    return res
end

function setat!(σ::ADoubleState, h::DoubledBasis, i::Int, val)
    hp = physical(h)
    np = nsites_physical(h)

    old = i > np ? setat!(row(σ), hp, i-np, val) : setat!(col(σ), hp, i, val)
    return old
end
