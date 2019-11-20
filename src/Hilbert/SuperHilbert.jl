abstract type AbstractSuperOpBasis <: AbstractBasis end

@inline physical(h::AbstractSuperOpBasis) = h.basis
@inline nsites(h::AbstractSuperOpBasis) = 2*nsites(physical(h))
@inline spacedimension(h::AbstractSuperOpBasis) = spacedimension(physical(h))^2
@inline indexable(h::AbstractSuperOpBasis) = spacedimension(h) != 0

state(h::AbstractSuperOpBasis) = DoubleState(state(physical(h)))
state!(s::DoubleState, h::AbstractSuperOpBasis, i::Int) = set_index!(s, i)
index(h::AbstractSuperOpBasis, s::DoubleState) = index(s)

mutable struct SuperOpSpace{H<:AbstractBasis} <: AbstractSuperOpBasis
    basis::H
end

Base.show(io::IO, ::MIME"text/plain", h::SuperOpSpace) =
    print(io, "Hilbert Space for Superoperators with $(physical(h)))")

Base.show(io::IO, h::SuperOpSpace) =
    print(io, "SuperOpSpace($(physical(h))))")
