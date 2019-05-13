export ObservablesProblem
"""
    ObservablesProblem

The problem representing the estimation of several observables.
"""
struct ObservablesProblem{B,SM} <: OperatorEstimationProblem where {SM<:SparseMatrixCSC}
    basis::B
    ObservablesTransposed::Vector{SM}
    Names::Vector{Symbol}
end

"""
    ObservablesProblem([T=Float64], obs1, obs2, ....)

Constructs an ObservablesProblem from the observables provided.
"""
ObservablesProblem(args...) = ObservablesProblem(Float32, args...)
ObservablesProblem(T::Type{<:Number}, args...) = ObservablesProblem(T, [(Symbol("obs_", i), obs) for (i,obs)=enumerate(args)])
ObservablesProblem(T::Type{<:Number}, args::Tuple...) = ObservablesProblem(T, [args...])
function ObservablesProblem(T::Type{<:Number}, obs::Vector{<:Tuple})
    nobs           = length(obs)
    names          = Vector{Symbol}()
    matrices_trans = Vector{SparseMatrixCSC}()
    for el=obs
        #println("$(first(el))")
        push!(names, first(el))
        if typeof(last(el)) <: QuantumOptics.SparseOperator
            push!(matrices_trans, Complex{T}.(last(el).data))
        elseif isa(last(el), QuantumLattices.GraphOperator)
            push!(matrices_trans, Complex{T}.(SparseOperator(last(el)).data))
        else
            push!(matrices_trans, Complex{T}.(last(el)))
        end
    end
    T = typeof(first(matrices_trans))
    b = basis(last(first(obs)))
    ObservablesProblem(b, T.(matrices_trans), names)
end

basis(prob::ObservablesProblem) = prob.basis

state(T::Type{<:Number}, prob::ObservablesProblem, net) =
    DiagonalStateWrapper(state(T, basis(prob), net))
