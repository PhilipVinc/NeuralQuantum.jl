
struct BatchedChainCache{T<:Tuple} <: NNCache{Chain}
    caches::T
    valid::Bool
end

cache(l::Chain, T, in_sz) = begin
    caches = []
    for layer = l.layers
        c, T, in_sz = cache(layer, T, in_sz)
        push!(caches, c)
    end
    ChainCache(Tuple(caches), false)
end
