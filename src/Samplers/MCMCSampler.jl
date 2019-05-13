export MCMCSampler

abstract type MCMCRule end
mutable struct MCMCSampler{T<:MCMCRule} <: MonteCarloSampler
    chain_length::Int
    burn_length::Int
    seed::UInt
    rule::T
end

MCMCSampler(rule, n_samples; burn=0, seed=rand(UInt)) =
    MCMCSampler{typeof(rule)}(n_samples, burn, seed, rule)

## Cache
mutable struct MCMCSamplerCache{A<:AbstractRNG} <: SamplerCache{ExactSampler}
    rng::A
    steps_done::Int
    steps_accepted::Int
end

_sampler_cache(s::MCMCSampler, v, net, part) =
    MCMCSamplerCache(MersenneTwister(s.seed),
                      0, 0)

function init_sampler!(s::MCMCSampler, net, σ, c::MCMCSamplerCache)
    c.steps_done = 0
    c.steps_accepted = 0
    set_index!(σ, rand(c.rng, 1:spacedimension(σ)))

    while c.steps_done < s.burn_length
        samplenext!(σ, s, net, c)
    end
    c.steps_done = 0

    return c
end

done(s::MCMCSampler, σ, c) = c.steps_done >= s.chain_length


## Mulithreading
function _mt_recompute_sampler_params!(samplers, s::MCMCSampler)
    nt = Threads.nthreads()
    _chain_length = Int(ceil(s.chain_length / nt))
    rng = MersenneTwister(s.seed)

    for i=1:Threads.nthreads()
        samplers[i] = MCMCSampler(s.rule,
                                   _chain_length,
                                   burn=s.burn_length,
                                   seed=rand(rng, UInt))
    end
end
