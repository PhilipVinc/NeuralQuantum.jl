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

        return LdagL_Lrho_op_prob(T, lind)
    else # not operators
        if variance
            if fullmatrix
                return LdagL_L_prob(T, lind)
            else
                return LdagL_Lrho_prob(T, lind)
            end
        else # not variance
            if fullmatrix
                return LdagL_sop_prob(T, lind, 0.0)
            else
                return LdagL_spmat_prob(T, lind, 0.0)
            end
        end
    end
end
