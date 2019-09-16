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
    ObservablesProblem([T=Float64], obs1, obs2, ....)

Constructs an ObservablesProblem from the observables provided.
Userd to compute observables.
"""
ObservablesProblem(args...; kwargs...) = ObservablesProblem(Float32, args...; kwargs...)
ObservablesProblem(T::Type{<:Number}, args...; kwargs...) = ObservablesProblem(T, [(Symbol("obs_", i), obs) for (i,obs)=enumerate(args)]; kwargs...)
ObservablesProblem(T::Type{<:Number}, args::Tuple...; kwargs...) = ObservablesProblem(T, [args...]; kwargs...)
#=
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
=#
function ObservablesProblem(T::Type{<:Number}, obs::Vector{<:Tuple}; operator=true)
    names          = Vector{Symbol}()
    matrices_trans = Vector{Any}()

    if operator
        for el=obs
            push!(names, first(el))
            push!(matrices_trans, to_linear_operator(last(el)))
        end
        b = basis(last(first(obs)))

        T = typeof(first(matrices_trans))
        if all(T.==typeof.(matrices_trans))
            matrices_trans = Vector{T}(matrices_trans)
        end

        return ObservablesProblem(b, matrices_trans, names)
    else
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
end


basis(prob::ObservablesProblem) = prob.HilbSpace

state(T::Type{<:Number}, prob::ObservablesProblem, net::MatrixNet) =
    DiagonalStateWrapper(state(T, basis(prob), net))

Base.show(io::IO, p::ObservablesProblem) = print(io,
    "ObservablesProblem on space : $(basis(p)) for the observables:"*
    prod(["\n\t$obj" for obj=p.Names]))
