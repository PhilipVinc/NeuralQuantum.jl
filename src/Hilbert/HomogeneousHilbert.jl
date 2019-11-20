export HomogeneousHilbert, HomogeneousSpin

mutable struct HomogeneousHilbert <: AbstractBasis
    n_sites::Int
    hilb_dim::Int
    shape::Vector{Int}
end

HomogeneousHilbert(n_sites, hilb_dim) =
    HomogeneousHilbert(n_sites, hilb_dim, fill(hilb_dim, n_sites))
HomogeneousSpin(n_sites) = HomogeneousHilbert(n_sites, 2)

@inline nsites(h::HomogeneousHilbert) = h.n_sites
@inline local_dim(h::HomogeneousHilbert) = h.hilb_dim

@inline spacedimension(h::HomogeneousHilbert) = local_dim(h)^nsites(h)
@inline indexable(h::HomogeneousHilbert) = spacedimension(h) != 0

state(h::HomogeneousHilbert) = NAryState(local_dim(h), nsites(h))
state!(s::NAryState, h::HomogeneousHilbert, i::Int) = set_index!(s, i)

index(h::HomogeneousHilbert, s::NAryState) = index(s)

Base.show(io::IO, ::MIME"text/plain", h::HomogeneousHilbert) =
    print(io, "Hilbert Space with $(nsites(h)) identical sites of dimension $(local_dim(h))")

Base.show(io::IO, h::HomogeneousHilbert) =
    print(io, "HomogeneousHilbert($(nsites(h)), $(local_dim(h)))")


## Operations
