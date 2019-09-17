abstract type MonteCarloSampler <: Sampler end
abstract type FullSpaceSampler <: Sampler end

parallel_type(s::Sampler) = true

abstract type SamplerCache{T}  end

"""
    caches(sampler, problem, net, [parallel_type=NotParallel()])

Creates a `SamplerCache` for this problem and sampler.
"""
cache(s::Sampler, prob::AbstractProblem, net, par=NotParallel()) =
    cache(s, state(prob, net), net, par)

"""
    caches(sampler, v, net, [parallel_type=NotParallel()])

Creates a `SamplerCache` for this sampler and state.
"""
cache(s::Sampler, v::State, net, par=NotParallel()) =
    _sampler_cache(s, v, net, par)


"""
    init_sampler!(sampler, net, σ, [c=cache(sampler, σ, net)]) -> c

Initializes the sampler `sampler` and state `σ`. If no `SamplerCache` is
provided, one will be initialized and returned. The state σ is the first in the
list of sampled states.
"""
init_sampler!(s::Sampler, net, σ) = init_sampler!(s, net, σ, cache(s, σ, net))
