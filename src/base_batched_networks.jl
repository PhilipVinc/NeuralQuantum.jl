"""
    NNCache{N}

The base abstract type holding a cache for the network type `N<:NeuralNetwork`.
"""
abstract type NNBatchedCache{N} <: NNCache{N}
    #valid::Bool
end

cached(net::NeuralNetwork, batch_sz::Int) =
    CachedNet(net, cache(net, batch_sz))

#
grad_cache(net::NeuralNetwork, batch_sz) = begin
    is_analytic(net) && return RealDerivative(net, batch_sz)
    return WirtingerDerivative(net, batch_sz)
end

function RealDerivative(net::NeuralNetwork, batch_sz::Int)
    pars = trainable(net)

    vec    = similar(trainable_first(pars), out_type(net), _tlen(pars), batch_sz)
    i, fields = batched_weight_tuple(net, vec)
    return RealDerivative(fields, [vec])
end
