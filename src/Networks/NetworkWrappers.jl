export PureStateAnsatz

struct PureStateAnsatz{A,IT,OT,Anal} <: KetNeuralNetwork
    __ansatz::A
end

function PureStateAnsatz(ansatz)
    # input type
    in_t = real(eltype(trainable_first(ansatz)))
    # output_type
    out_t = Complex{eltype(trainable_first(ansatz))}
    # is_analytic
    anal = true
    ansatz = (ansatz,)
    return PureStateAnsatz{typeof(ansatz), in_t, out_t, anal}(ansatz)
end

@forward PureStateAnsatz.__m_ansatz Base.getindex, Base.length, Base.first, Base.last,
         Base.iterate, Base.lastindex, cache

ansatz(psa::PureStateAnsatz) = first(getfield(psa, :__ansatz))
@functor PureStateAnsatz

(c::PureStateAnsatz)(x::Vararg{N,V}) where {N,V} = ansatz(c)(config.(x)...)
(c::PureStateAnsatz)(cache::NNCache, σ) = ansatz(c)(cache, config(σ))

logψ_and_∇logψ!(∇logψ, net::PureStateAnsatz, c::NNCache, σ) =
    logψ_and_∇logψ!(∇logψ.__ansatz[1], ansatz(net), c, σ)


input_type(::PureStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = IT
out_type(::PureStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = OT
is_analytic(::PureStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = AN


Base.propertynames(psa::PureStateAnsatz) = propertynames(ansatz(psa))
Base.getproperty(psa::PureStateAnsatz, x::Symbol) = begin
    x === :__ansatz   && return getfield(psa, x)
    x === :__m_ansatz && return return ansatz(psa)
    getfield(ansatz(psa), x)
end
Base.setproperty!(psa::PureStateAnsatz, name::Symbol, x) = begin
    name === :__ansatz   && return setfield!(psa, name, x)
    name === :__m_ansatz && return setfield!(psa, ansatz(psa), x)
    setfield!(ansatz(psa), name, x)
end
