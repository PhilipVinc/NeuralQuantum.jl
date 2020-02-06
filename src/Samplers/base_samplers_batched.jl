abstract type BatchedSamplerCache{T} <: SamplerCache{T} end

# If constructing the cache for a BatchedNet, then build a batched sampler
#cache(s::Sampler, v::State, net::CachedNet{N,C}, par=NotParallel()) where {N,C<:NNBatchedCache} =
#    _batched_sampler_cache(s, v, net, par)

cache(s::Sampler, v, net::CachedNet{N,C}, par_cache=NotParallel()) where {N,C<:NNBatchedCache} =
    _batched_sampler_cache(s, v, net, par_cache)

_batched_sampler_cache(s, v, net, par) =
    _sampler_cache(s, v, net, par)
