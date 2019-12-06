"""
    to_matrix(operator)

Converts to a dense matrix the KLocal Operator
"""
function to_matrix(op::AbsLinearOperator)
    hilb = basis(op)
    @assert indexable(hilb)

    N = spacedimension(hilb)
    mat = zeros(ComplexF64, N, N)


    for (i, σ) = enumerate(states(hilb))
        fun = (mel, cngs, σ) -> begin
            σp = apply(σ, cngs)
            j = index(hilb, σp)
            mat[i, j] += mel
        end

        map_connections(fun, op, σ)
    end
    return mat
end

Base.Matrix(op::AbsLinearOperator) = to_matrix(op)

"""
    to_sparse(operator)

Converts to a sparse matrix the KLocal Operator.
"""
function to_sparse(op::AbsLinearOperator)
    hilb = basis(op)
    @assert indexable(hilb)

    i_vals   = Vector{Int}()
    j_vals   = Vector{Int}()
    mel_vals = Vector{ComplexF64}()

    for (i, σ) = enumerate(states(hilb))
        conns = row_valdiff(op, σ)

        for (mel, cngs) = conns
            σp = apply(σ, cngs)
            j = index(hilb, σp)

            push!(i_vals, i)
            push!(j_vals, j)
            push!(mel_vals, mel)
        end
    end

    N = spacedimension(hilb)
    return sparse(i_vals, j_vals, mel_vals, N, N)
end

SparseArrays.sparse(op::AbsLinearOperator) = to_sparse(op)


function to_map(op::AbsLinearOperator)
    hilb = basis(op)
    N = spacedimension(hilb)

    σ = state(hilb)
    function op_v_prod!(y::AbstractVector, x::AbstractVector)
        y .= zero(eltype(y))

        for (i, x_val) = enumerate(x)
            x_val == 0 && continue

            state!(σ, hilb, i)
            conns = row_valdiff(op, σ)

            for (mel, cngs) = conns
                mel == 0 && continue

                σp = apply(σ, cngs)
                j = index(hilb, σp)
                println("$i, $j  => $mel -- $σ - $σp")
                y[j] += x_val*mel
            end
        end
        return y
    end

    return LinearMap{ComplexF64}(op_v_prod!, N, N; ismutating=true)
end
