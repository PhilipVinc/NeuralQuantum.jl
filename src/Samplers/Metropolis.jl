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

## Show method
function Base.show(io::IO, s::MetropolisSampler)
    burn_time = @sprintf "%2.2f" (s.burn_length/s.chain_length)*100

    println(io, "Metropolis Sampler:")
    println(io, "\t- chain_length     : $(s.chain_length)")
    println(io, "\t- burn_length      : $(s.burn_length) ($burn_time% warmup time)")
    println(io, "\t- passes           : $(s.passes)")
    print(io,   "\t- seed             : $(s.seed)")
end

## Cache
mutable struct MetropolisSamplerCache{A<:AbstractRNG,RC,H,S,RV,V,M} <: SamplerCache{MetropolisSampler}
    rng::A
    loc_chain_length::Int
    steps_done::Int
    passes_done::Int
    passes_accepted::Int
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

    MetropolisSamplerCache(rng, loc_chain_length, 0, 0, 0,
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
    c.steps_done = - s.burn_length + 1
    c.passes_done = 0
    c.passes_accepted = 0
    rand!(c.rng, σ, c.hilb)
    init_sampler_rule_cache!(rule_cache(c), s, net, σ, c)
    c.log_prob_bias .= 0

    while c.steps_done <= 0
        samplenext!(σ, σ, s, net, c)
    end

    return c
end

init_sampler_rule_cache!(rc, s, net, σ, c) =
    nothing

chain_length(s::MetropolisSampler, c::MetropolisSamplerCache) = c.loc_chain_length

done(s::MetropolisSampler, σ, c) = c.steps_done >= chain_length(s, c)

function samplenext!(σ_out::T, σ_in::T, s::MetropolisSampler,
                     net::Union{MatrixNet,KetNet}, c) where {T<:Union{AStateBatch, ADoubleStateBatch}}
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    # Caches for batched state output
    logψ_σ  = c.logψ_σ
    logψ_σp = c.logψ_σp

    # Copy the old state in the output and tmp cache
    # Only do this when when input is not same as ouput
    # to avoid aliasing issues (especially on GPU)
    σ_out === σ_in || statecopy!(σ_out, σ_in)

    # Compute the old value
    ψtmp     = log_prob_ψ!(logψ_σ, c.ψtmp, net, σ_in)

    for i=1:s.passes
        statecopy!(c.σ_old, σ_out)
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

        # Store number of accepted moves
        c.passes_accepted += length(c.prob) - sum(c.mask)
        c.passes_done     += length(c.prob)
    end
    c.steps_done += 1
    return true
end

# General implementation of Rule-step proposal on a batch, serially applying
# the rule on all batches.
@inline function propose_step!(
                       σp::Union{AStateBatch,ADoubleStateBatch},
                       s::MetropolisSampler,
                       net::Union{MatrixNet,KetNet}, c, rc)
    for i=1:num_batches(σp)
        propose_step!(unsafe_get_batch(σp, i), s, net, c, rc)
    end
end


## Show method
function Base.show(io::IO, c::MetropolisSamplerCache)
    println(io, "MetropolisSamplerCache{...}:")
    println(io, "\t- Space                 : $(c.hilb)")
    println(io, "\t- Stored chain length   : $(c.loc_chain_length)")
    println(io, "\t- MCMC steps done       : $(c.steps_done)")
    println(io, "\t- Passes done           : $(c.passes_done)")

    acceptance_ratio = @sprintf "%2.2f" (c.passes_accepted/c.passes_done)*100
    print(io, "\t- Passes accepted       : $(c.passes_accepted) ($acceptance_ratio %)")
end
