export SteadyStateProblem

"""

"""
SteadyStateProblem(args...) = SteadyStateProblem(STD_REAL_PREC, args...)

function SteadyStateProblem(T::Type{<:Number}, lind; operators=true, variance=true, fullmatrix=false)
    base = basis(lind)

    if operators
        if !variance
            throw("Can't use operators=true and variance=false. Operators are not
                   compatible with non-variance minimization.")
        end

        return LRhoKLocalOpProblem(T, lind)
    else # not operators
        if variance
            if fullmatrix
                return LdagLSparseSuperopProblem(T, lind)
            else
                return LRhoSparseOpProblem(T, lind)
            end
        else # not variance
            if fullmatrix
                return LRhoSparseSuperopProblem(T, lind, 0.0)
            else
                return LdagLSparseOpProblem(T, lind, 0.0)
            end
        end
    end
end
