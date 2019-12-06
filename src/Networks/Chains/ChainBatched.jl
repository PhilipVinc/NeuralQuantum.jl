struct ChainBatchedCache{T<:Tuple} <: NNBatchedCache{Chain}
    caches::T
    valid::Bool
end

cache(l::Chain, arr_T, in_T, in_sz, batch_sz) = begin
    caches = []
    for layer = l.layers
        c = cache(layer, arr_T, in_T, in_sz, batch_sz)
        in_T, in_sz = layer_out_type_size(layer, in_T, in_sz)
        push!(caches, c)
    end
    ChainBatchedCache(Tuple(caches), false)
end

@inline batch_size(c::ChainBatchedCache) = batch_size(first(c.caches))

logψ!(out, c::Chain, ch::ChainBatchedCache, x) =
    out .= applychain(c.layers, ch.caches, x)

function logψ_and_∇logψ!(∇logψ, out, c::Chain, ch::ChainBatchedCache, x)
    # forward pass
    logψ!(out, c, ch,  x)

    # backward
    backpropchain(∇logψ, c.layers, ch.caches, 1.0)

    out .= applychain(c.layers, ch.caches, x)
    return out
end
