"""
    qo_to_nq_basis(basis)

Converts a QuantumOptics.jl basis to NeuralQuantum basis of the hilbert
space.
"""
function qo_to_nq_basis(b::Basis)
    hilb_shape = b.shape
    if all(first(hilb_shape) .== hilb_shape)
        if first(b.bases) isa SpinBasis
            hilb = HomogeneousSpin(length(hilb_shape), first(b.bases).spinnumber)
        else
            hilb = HomogeneousFock(length(hilb_shape), first(hilb_shape))
        end
    else
        hilb = DiscreteHilbert(hilb_shape)
    end
    return hilb
end

# No op
qo_to_nq_basis(b::AbstractHilbert) = b

function nq_to_qo_basis(b::HomogeneousFock)
    cutoff = local_dim(b)
    N = nsites(b)
    lb = FockBasis(cutoff-1)
    return lb^N
end

function nq_to_qo_basis(b::HomogeneousSpin)
    cutoff = local_dim(b)
    S = (cutoff-1)//2
    N = nsites(b)
    lb = SpinBasis(S)
    return lb^N
end

function nq_to_qo_basis(b::SuperOpSpace)
    qo_b = nq_to_qo_basis(physical(b))
    return (qo_b, qo_b)
end

Base.convert(::Type{<:CompositeBasis}, b::AbstractHilbert) =
    nq_to_qo_basis(b)
