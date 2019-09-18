export SteadyStateProblem

"""
    SteadyStateProblem([T=STD_REAL_PREC], ℒ, operators=true, variance=true)
    SteadyStateProblem([T=STD_REAL_PREC], H, J, operators=true, variance=true)

Returns the problem targeting the steady-state of the Liouvillian `ℒ` through
a minimisation of the cost function 𝒞 = Tr[ρ'ℒ'ℒ ρ]/Tr[ρ'ρ] .

If `variance=true` the cost function is sampled by evaluating ℒρ, which has
better convergence properies. The sampling is performed on ℒ'ℒ ρ otherwise.

See appendix of https://arxiv.org/abs/1902.10104 for more info.

If `operators=true` a memory efficient representation of the hamiltonian is used,
resulting in less memory consuption but higher CPU usage. This is needed for lattices
bigger than a certain threshold.
To use this feature, the hamiltonian must be provided as a `GraphOperator` object,
from `QuantumLattices` package.
"""
SteadyStateProblem(args...; kwargs...) = SteadyStateProblem(STD_REAL_PREC, args...; kwargs...)

# Dispatched if called with Hamiltonian and Jump operators
function SteadyStateProblem(T::Type{<:Number}, H::DataOperator, J::AbstractVector,
                            operators=false; kwargs...)
    if operators
        @warn """Converting QuantumOptics' DataOperators to memory-efficient Local Operators is not possible.
        Setting `operators=false`.
        To suppress this warning, don't request `operators=true`.
        """
        operators = false
    end
    return _build_steadystate_sparsemat_problem(T, H, J; kwargs...)
end

# Dispatched when called with the object for the whole liouvillian
function SteadyStateProblem(T::Type{<:Number}, ℒ; operators=true, variance=true, kwargs...)
    base = basis(ℒ)

    if operators
        if !variance
            throw("Can't use operators=true and variance=false. Operators are not
                   compatible with non-variance minimization.")
        end

        return LRhoKLocalOpProblem(T, ℒ)
    else # not operators
        H = SparseOperator(hamiltonian(ℒ))
        J = jump_operators(ℒ)
        _build_steadystate_sparsemat_problem(T, H, J; variance=variance, kwargs...)
    end
end

# Builds the problem if not using KLocal operators
function _build_steadystate_sparsemat_problem(T::Type{<:Number}, H, J; variance=false, fullmatrix=false)
    if variance
        if fullmatrix
            return LRhoSparseSuperopProblem(T, H, J)
        else
            return LRhoSparseOpProblem(T, H, J)
        end
    else # not variance
        if fullmatrix
            return LdagLSparseSuperopProblem(T, H, J)
        else
            return LdagLSparseOpProblem(T, H, J)
        end
    end
end
