struct DiscreteHilbert <: AbstractHilbert
    shape::Vector{Int}
end

DiscreteHilbert(Nsites, hilb_dim) =
    DiscreteHilbert(fill(hilb_dim, Nsites))

@inline shape(h::DiscreteHilbert) = h.shape
@inline nsites(h::DiscreteHilbert) = length(shape(h))

@inline spacedimension(hilb::DiscreteHilbert) = prod(shape)
@inline is_homogeneous(h::DiscreteHilbert) = all(first(h.shape) .== h.shape)
