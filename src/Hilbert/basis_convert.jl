"""
    qo_to_nq_basis(basis)

Converts a QuantumOptics.jl basis to NeuralQuantum basis of the hilbert
space.
"""
function qo_to_nq_basis(b::Basis)
    hilb_shape = b.shape
    if all(first(hilb_shape) .== hilb_shape)
        hilb = HomogeneousFock(length(hilb_shape), first(hilb_shape))
    else
        hilb = DiscreteHilbert(hilb_shape)
    end
    return hilb
end

# No op
qo_to_nq_basis(b::AbstractHilbert) = b
