export MetropolisSampler

abstract type MCMCRule end
mutable struct MetropolisSampler{T<:MCMCRule} <: MonteCarloSampler
    chain_length::Int
    burn_length::Int
    passes::Int
    seed::UInt
    rule::T
end

"""
    MetropolisSampler(rule, chain_length, passes; burn=0, seed=rand)

Constructs a Metropolis-Hastings sampler which samples Markov chains of length
`chain_length + burn`, ignoring the first `burn` samples. Transition rules are
specified by the rule `rule`. To reduce auto-correlation, `passes` number of
metropolis-hastings steps are performed for each sample returned (minimum 1).

Effectively, this means that the chain is actually `passes * (chain_lenght + burn)`
long, but only 1 every passes elements are stored and used to compute expectation
values.

Initial seed can be set bu specifying `seed`.
"""
function MetropolisSampler(rule, chain_length, passes; burn=0, seed=rand(UInt))
    @assert passes > 0
    @assert chain_length > 0

    if iseven(passes)
        @warn """
            LocalRule sampling with an even number of switches per step
            is known to be non-ergodic, because it can potentially never
            change the configuration.
            Setting the number of switches to $(passes+1).
            """
        passes += 1
    end
    MetropolisSampler{typeof(rule)}(chain_length, burn, passes, seed, rule)
end

## Cache
mutable struct MetropolisSamplerCache{A<:AbstractRNG,RC,H,S,RV,V,M} <: SamplerCache{MetropolisSampler}
    rng::A
    loc_chain_length::Int
    steps_done::Int
    steps_accepted::Int
    rule_cache::RC

    hilb::H
    σ_old::S

    logψ_σ::RV
    logψ_σp::RV
    log_prob_bias::RV

    ψtmp::V
    prob::RV
    mask::M
end

function _sampler_cache(s::MetropolisSampler, v, hilb, net, par_cache)
    loc_chain_length = Int(ceil(s.chain_length/num_workers(par_cache)))
    loc_seed         = worker_local_seed(s.seed, par_cache)

    logψ_σ   = real(out_similar(net))
    logψ_σp  = real(out_similar(net))
    log_bias = real(out_similar(net)) .= 0.0

    out      = out_similar(net)
    prob     = logψ_σp - logψ_σ
    mask     = similar(logψ_σ, Bool)

    rng      = build_rng_generator_T(prob, loc_seed)

    MetropolisSamplerCache(rng, loc_chain_length, 0, 0,
                      RuleSamplerCache(s.rule, s, v, net, par_cache),
                      hilb, deepcopy(v),
                      logψ_σ, logψ_σp, log_bias,
                      out, prob, mask)
end

# Construct the cache of a specific rule.
# By default rules have no cache.
RuleSamplerCache(r::MCMCRule, s, v, net, par_cache) = nothing
rule(s::MetropolisSampler) = s.rule
rule_cache(sc::MetropolisSamplerCache) = sc.rule_cache

function init_sampler!(s::MetropolisSampler, net, σ,
                       c::MetropolisSamplerCache)
    c.steps_done = - s.burn_length
    c.steps_accepted = 0
    rand!(c.rng, σ, c.hilb)
    init_sampler_rule_cache!(rule_cache(c), s, net, σ, c)
    c.log_prob_bias .= 0

    while c.steps_done < 0
        samplenext!(σ, σ, s, net, c)
    end

    return c
end

init_sampler_rule_cache!(rc, s, net, σ, c) =
    nothing

chain_length(s::MetropolisSampler, c::MetropolisSamplerCache) = c.loc_chain_length

done(s::MetropolisSampler, σ, c) = c.steps_done >= chain_length(s, c)-1

function samplenext!(σ_out::T, σ_in::T, s::MetropolisSampler,
                     net::Union{MatrixNet,KetNet}, c) where {T<:Union{AStateBatch, ADoubleStateBatch}}
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    # Caches for batched state output
    logψ_σ  = c.logψ_σ
    logψ_σp = c.logψ_σp

    # Copy the old state in the output and tmp cache
    statecopy!(c.σ_old, σ_in)
    statecopy!(σ_out, c.σ_old) # equivalent to σ_in but evades problem with aliasing
    # when input === ouput

    # Compute the old value
    ψtmp     = log_prob_ψ!(logψ_σ, c.ψtmp, net, σ_in)

    for i=1:s.passes
        # Apply the transition rule
        propose_step!(σ_out, s, net, c, rule_cache(c))

        ψtmp     = log_prob_ψ!(logψ_σp, c.ψtmp, net, σ_out)

        rand!(c.rng, c.prob)
        c.prob .-= exp.(logψ_σp .- logψ_σ .+ c.log_prob_bias)

        # the mask encodes the non-switched states
        # when rand >= prob
        # ex : prob = 0.1 => 0.1..1.0 >= 0.1
        c.mask .= c.prob .>= 0

        # Copy the old configurations for chains that did not swithc
        statecopy!(σ_out, c.σ_old, c.mask)
        # Copy the new log_σ values for states that switched
        statecopy_invertmask!(logψ_σ, logψ_σp, c.mask)
    end
    c.steps_done += 1
    return true
end



"""
    markov_chain_step!(state, sampler, net, sampler_cache)

performs one step of markov chain
"""
function markov_chain_step!(σ, s::MetropolisSampler, net, c)
    throw("markov_chain_step! undefined for this type")
end

## Mulithreading
function _mt_recompute_sampler_params!(samplers, s::MetropolisSampler)
    nt = Threads.nthreads()
    _chain_length = Int(ceil(s.chain_length / nt))
    rng = MersenneTwister(s.seed)

    for i=1:Threads.nthreads()
        samplers[i] = MetropolisSampler(s.rule,
                                   _chain_length,
                                   burn=s.burn_length,
                                   seed=rand(rng, UInt))
    end
end
