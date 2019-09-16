export GroundStateProblem

"""
    GroundStateProblem([T=STD_REAL_PREC], ham, operators=true, variance=false)

Returns the problem encoding the targeting of the ground state energy. If
`variance=false` this is done by minimizing the energy, otherwise it's done by
minimizing the variance.

If `operators=true` a memory efficient representation of the hamiltonian is used,
resulting in less memory consuption but higher CPU usage. This is needed for lattices
bigger than a certain threshold.
"""
GroundStateProblem(args...) = GroundStateProblem(STD_REAL_PREC, args...)

function GroundStateProblem(T::Type{<:Number}, hamiltonian; operators=true, variance=false)
    base = basis(hamiltonian)

    if operators
        ham = to_linear_operator(hamiltonian)
    else
        ham = data(SparseOperator(hamiltonian))
    end

    if !variance
        return HamiltonianGSEnergyProblem(basis(hamiltonian), ham, 0.0)
    else
        throw("Variance minimization problem not implemented yet")
    end
end
