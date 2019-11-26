state(prob::AbstractProblem, net, args...) = state(input_type(net), prob, net, args...)
state(hilb::AbstractHilbert, net::NeuralNetwork) = state(input_type(net), hilb, net)
state(T::Type{<:Number}, prob::AbstractProblem, net) = state(T, basis(prob), net)

state(T::Type{<:Number}, hilb::AbstractHilbert, net::CachedNet) =
    state(T, hilb, net.net, net.cache)

#ignore cache if standard cache
state(T::Type{<:Number}, hilb::AbstractHilbert, net::NeuralNetwork, cache::NNCache) =
    state(T, hilb, net)

state(T::Type{<:Number}, hilb::AbstractHilbert, net::NeuralNetwork, cache::NNBatchedCache) =
    state_batch(T, hilb, net, batch_size(cache))

state(T::Type{<:Number}, hilb::AbstractHilbert, net::Union{MatrixNeuralNetwork, KetNeuralNetwork}) =
    state(trainable_first(net), T, hilb)

function state_batch(T::Type{<:Number}, hilb::AbstractHilbert, net::KetNet, b_sz)
    v = state(T, hilb, net)
    return similar(v, length(v), b_sz)
end

function state_batch(T::Type{<:Number}, hilb::AbstractSuperOpBasis, net::MatrixNet, b_sz)
    v = state(T, hilb, net)
    vr = row(v)
    vc = col(v)
    return (similar(vr, length(vr), b_sz), similar(vc, length(vc), b_sz))
end

function state_batch(T::Type{<:Number}, hilb::AbstractHilbert, net::MatrixNet, b_sz)
    v = state(T, hilb, net)
    return similar(v, length(v), b_sz)
end
