export PureStateAnsatz

struct PureStateAnsatz{A,IT,OT,Anal,S} <: KetNeuralNetwork
    __ansatz::A
    in_size::S
end

function PureStateAnsatz(ansatz, in_size)
    # input type
    in_t = real(eltype(trainable_first(ansatz)))
    # output_type
    out_t = Complex{real(eltype(trainable_first(ansatz)))}
    # is_analytic
    anal = true
    ansatz = (ansatz,)
    S=typeof(in_size)
    return PureStateAnsatz{typeof(ansatz), in_t, out_t, anal,S}(ansatz, in_size)
end

@forward PureStateAnsatz.__m_ansatz Base.getindex, Base.length, Base.first, Base.last,
         Base.iterate, Base.lastindex

ansatz(psa::PureStateAnsatz) = first(getfield(psa, :__ansatz))
functor(psa::PureStateAnsatz) = psa.__ansatz,
        a -> PureStateAnsatz(a, input_size(psa))

cache(psa::PureStateAnsatz) =
    cache(psa.__m_ansatz, input_type(psa), input_size(psa))

cache(psa::PureStateAnsatz, batch_sz::Int) =
    cache(psa.__m_ansatz, input_type(psa), input_size(psa), batch_sz)


(c::PureStateAnsatz)(x::Vararg{N,V}) where {N,V} = ansatz(c)(x...)
(c::PureStateAnsatz)(cache::NNCache, σ) = ansatz(c)(cache, σ)

logψ_and_∇logψ!(∇logψ, net::PureStateAnsatz, c::NNCache, σ) =
    logψ_and_∇logψ!(∇logψ[1], ansatz(net), c, σ)


input_type(::PureStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = IT
input_size(psa::PureStateAnsatz) = getfield(psa, :in_size)
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
