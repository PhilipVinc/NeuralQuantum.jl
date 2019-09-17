export GroundStateProblem

"""
    GroundStateProblem([T=STD_REAL_PREC], ham, operators=true, variance=false)

Returns the problem targeting of the ground state energy. If `variance=false`
this is done by minimizing the energy, otherwise it's done by minimizing the
variance.


If `operators=true` a memory efficient representation of the hamiltonian is used,
resulting in less memory consuption but higher CPU usage. This is needed for lattices
bigger than a certain threshold.
To use this feature, the hamiltonian must be provided as a `GraphOperator` object,
from `QuantumLattices` package.
"""
GroundStateProblem(args...; kwargs...) = GroundStateProblem(STD_REAL_PREC, args...; kwargs...)
GroundStateProblem(T::Type{<:Number}, args...; kwargs...) = _build_groundstate_problem(T, args...; kwargs...)

# If Called with QuantumOptics' DataOperators, we can't convert them to
# KLocalOperators, so set `operators=false` and warn the user
function GroundStateProblem(T::Type{<:Number}, hamiltonian::DataOperator; operators=false, kwargs...)
    if operators
        @warn """Converting QuantumOptics' DataOperators to memory-efficient Local Operators is not possible.
        Setting `operators=false`.
        To suppress this warning, don't request `operators=true`.
        """
        operators = false
    end
    return _build_groundstate_problem(T, hamiltonian, operators=operators; kwargs...)
end


# The actual implementation
function _build_groundstate_problem(T::Type{<:Number}, hamiltonian; operators=true, variance=false)
    base = basis(hamiltonian)
    T    = Complex{real(T)}

    if operators
        ham = to_linear_operator(hamiltonian, T)
    else
        ham = convert(AbstractMatrix{T},
                      data(SparseOperator(hamiltonian)))
    end

    if !variance
        return HamiltonianGSEnergyProblem(basis(hamiltonian), ham, 0.0)
    else
        throw("Variance minimization problem not implemented yet")
    end
end
