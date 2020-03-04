abstract type MonteCarloSampler <: Sampler end
abstract type FullSpaceSampler <: Sampler end

parallel_type(s::Sampler) = true

abstract type SamplerCache{T}  end

"""
    caches(sampler, v, net, [parallel_type=NotParallel()])

Creates a `SamplerCache` for this sampler and state.
"""
cache(s::Sampler, hilb::AbstractHilbert, net, par=NotParallel()) =
    _sampler_cache(s, state(hilb, net), hilb, net, par)

cache(s::Sampler, σ, hilb::AbstractHilbert, net, par=NotParallel()) =
    _sampler_cache(s, σ, hilb, net, par)

"""
    init_sampler!(sampler, net, σ, [c=cache(sampler, σ, net)]) -> c

Initializes the sampler `sampler` and state `σ`. If no `SamplerCache` is
provided, one will be initialized and returned. The state σ is the first in the
list of sampled states.
"""
init_sampler!(s::Sampler, net, hilb::AbstractHilbert, σ, par_cache::ParallelType=NotParallel()) =
    init_sampler!(s, net, σ, cache(s, σ, hilb, net, par_cache))

"""
    chain_length(sampler, sampler_cache) -> Int

Returns the estimated length of the chain.
"""
function chain_length end

"""
    sampling_function(s::SamplerCache) -> function

Returns the sampling function. By default it's 2.0*real().
"""
function sampling_function end

samplenext!(σ, sampl::Sampler, net, sampler_cache) =
    samplenext!(σ, σ, sampl, net, sampler_cache)
