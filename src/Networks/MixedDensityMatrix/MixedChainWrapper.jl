export MixedStateAnsatz, MixedChain

struct MixedStateAnsatz{A,IT,OT,Anal,S} <: MatrixNeuralNetwork
    __ansatz::A
    in_size::S
end

#TODO rewrite
"""
    MixedStateAnsatz(ansatz, input_size)

Wraps an arbitrary function into a structure that can be used as a variational
ansatz for a mixed state, assuming that the width of the input is `input_size`
(corresponding to the number of lattice sites in the input hilbert space).

For this to work, ansatz must be a function respecting the following:
```
ansatz(rand(input_size)) --> scalar
trainable(ansatz) -> NamedTuple
```

To make `trainable` work with your functions, see the docs of [`functor`](@ref)
"""
function MixedStateAnsatz(ansatz, in_size)
    if ansatz isa MixedStateAnsatz
        throw("ansatz is already a network!")
    end
    # input type
    in_t = real(eltype(trainable_first(ansatz)))
    # output_type
    out_t = Complex{real(eltype(trainable_first(ansatz)))}
    # is_analytic
    anal = true
    ansatz = (ansatz,)
    S=typeof(in_size)
    return MixedStateAnsatz{typeof(ansatz), in_t, out_t, anal,S}(ansatz, in_size)
end

"""
    MixedChain(N, args...)

Constructs a FeedForward network (a chain) encoding a mixed-state. `N` must be
the number of input sites, while args... should be a series of layers.

Note: when you want to use sum, to output the total sum of states, don't use it
and use instead `sum_autobatch`, defined in this package. `sum_autobatch` behaves
as sum only for vectors, while for batches of data it sum along batches.

ex:
```
MixedChain(5, Dense(5,3), Dense(3,2), sum_autobatch)
```
"""
function MixedChain(N, args...)
    ch = Chain(args...)
    return MixedStateAnsatz(N, ch)
end

@forward MixedStateAnsatz.__m_ansatz Base.length, Base.first, Base.last,
         Base.iterate, Base.lastindex

ansatz(psa::MixedStateAnsatz) = first(getfield(psa, :__ansatz))
functor(psa::MixedStateAnsatz) = psa.__ansatz,
        a -> MixedStateAnsatz(a, input_size(psa))

cache(psa::MixedStateAnsatz) =
    cache(psa.__m_ansatz, trainable_first(psa), input_type(psa), input_size(psa))

cache(psa::MixedStateAnsatz, batch_sz::Int) =
    cache(psa.__m_ansatz, trainable_first(psa), input_type(psa), input_size(psa), batch_sz)


#(c::MixedStateAnsatz)(x::Vararg{N,V}) where {N,V} = ansatz(c)(x...)
@inline (c::MixedStateAnsatz)(σ::ADoubleStateOrBatch) = ansatz(c)(σ)
@inline (c::MixedStateAnsatz)(σr::AStateOrBatch, σc::AStateOrBatch) = ansatz(c)((σr, σc))
@inline (c::MixedStateAnsatz)(cache::NNCache, σ) = ansatz(c)(cache, σ)
@inline (c::MixedStateAnsatz)(cache::NNCache, σr::AStateOrBatch, σc::AStateOrBatch) = ansatz(c)(cache, (σr, σc))

logψ!(out, net::MixedStateAnsatz, c::NNBatchedCache, σr::AStateBatch, σc::AStateBatch) =
    logψ!(out, ansatz(net), c, (σr, σc))

logψ_and_∇logψ!(∇logψ, out::AbstractMatrix, net::MixedStateAnsatz, c::NNBatchedCache, σr::AStateOrBatch, σc::AStateOrBatch) =
    logψ_and_∇logψ!(∇logψ[1], out, ansatz(net), c, (σr, σc))

logψ_and_∇logψ!(∇logψ, net::MixedStateAnsatz, c::NNCache, σr::AStateOrBatch, σc::AStateOrBatch) =
    logψ_and_∇logψ!(∇logψ[1], ansatz(net), c, (σr, σc))


input_type(::MixedStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = IT
input_size(psa::MixedStateAnsatz) = getfield(psa, :in_size)
out_type(::MixedStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = OT
is_analytic(::MixedStateAnsatz{A,IT,OT,AN}) where {A,IT,OT,AN} = AN


Base.propertynames(psa::MixedStateAnsatz) = propertynames(ansatz(psa))
Base.getproperty(psa::MixedStateAnsatz, x::Symbol) = begin
    x === :__ansatz   && return getfield(psa, x)
    x === :__m_ansatz && return return ansatz(psa)
    getproperty(ansatz(psa), x)
end
Base.setproperty!(psa::MixedStateAnsatz, name::Symbol, x) = begin
    name === :__ansatz   && return setfield!(psa, name, x)
    name === :__m_ansatz && return setfield!(psa, ansatz(psa), x)
    setproperty!(ansatz(psa), name, x)
end

Base.getindex(psa::MixedStateAnsatz, x::Int) = begin
    @assert x==1
    return ansatz(psa)
end
