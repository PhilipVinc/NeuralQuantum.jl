export ObservablesProblem

"""
    ObservablesProblem

The problem representing the estimation of several observables.
"""
struct ObservablesProblem{B,SM} <: OperatorEstimationProblem where {SM}
    HilbSpace::B            # 0
    ObservablesTransposed::Vector{SM}
    Names::Vector{Symbol}
end

"""
    ObservablesProblem([T=STD_REAL_PREC], obs1, obs2, ....)

Constructs an ObservablesProblem from the observables provided.
Userd to compute observables.
"""
ObservablesProblem(args...; kwargs...) = ObservablesProblem(STD_REAL_PREC, args...; kwargs...)

function ObservablesProblem(T::Type{<:Number}, obs::Any...; operator=true)
    if length(obs) == 1
        obs = first(obs)
    end
    if obs isa DataOperator || obs isa AbstractOperator
        obs = [obs]
    end

    names          = Vector{Symbol}()
    matrices_trans = Vector{Any}()

    if operator
        b = nothing
        for (i, el) = enumerate(obs)
            if el isa Tuple
                name = first(el)
                op = last(el)
            else
                name = "obs_$i"
                op = el
            end
            push!(names, Symbol(name))
            push!(matrices_trans, to_linear_operator(op))
            b = basis(op)
        end

        T = typeof(first(matrices_trans))
        if all(T.==typeof.(matrices_trans))
            matrices_trans = Vector{T}(matrices_trans)
        end

        return ObservablesProblem(b, matrices_trans, names)
    else
        b = nothing
        matrices_trans = Vector{SparseMatrixCSC}()
        for (i, el) = enumerate(obs)
            if el isa Tuple
                name = first(el)
                op = last(el)
            else
                name = "obs_$i"
                op = el
            end
            push!(names, Symbol(name))
            if op isa QuantumOptics.DataOperator
                if op isa DenseOperator
                    op = SparseOperator(op)
                end
                push!(matrices_trans, Complex{T}.(op.data))
            elseif isa(op, QuantumLattices.GraphOperator)
                push!(matrices_trans, Complex{T}.(SparseOperator(op).data))
            else
                push!(matrices_trans, Complex{T}.(last(el)))
            end
            b = basis(op)
        end
        T = typeof(first(matrices_trans))
        return ObservablesProblem(b, T.(matrices_trans), names)
    end
end


basis(prob::ObservablesProblem) = prob.HilbSpace

state(T::Type{<:Number}, prob::ObservablesProblem, net::MatrixNet) =
    DiagonalStateWrapper(state(T, basis(prob), net))

#state(T::Type{<:Number}, prob::ObservablesProblem{<:Any, <:AbsLinearOperator}, net) =
#    DiagonalStateWrapper(state_lut(T, basis(prob), net))

Base.show(io::IO, p::ObservablesProblem) = print(io,
    "ObservablesProblem on space : $(basis(p)) for the observables:"*
    prod(["\n\t$obj" for obj=p.Names]))
