struct DiscreteHilbert <: AbstractBasis
    shape::Vector{Int}
end

DiscreteHilbert(Nsites, hilb_dim) =
    DiscreteHilbert(fill(hilb_dim, Nsites))

spacedimension(hilb::DiscreteHilbert) = prod(shape)
