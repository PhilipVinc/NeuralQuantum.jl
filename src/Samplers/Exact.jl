export ExactSampler

"""
    ExactSampler(n_samples; seed=rand)

Constructs an exact sampler, which builds the full pdf of the quantum state and
samples it exactly.

This sampler can only be used on `indexable` spaces, and should be used only for
somewhat small systems (Closed N<14, Open N<7), as the computational cost increases
exponentially with the number of sites.

Initial seed can be set bu specifying `seed`.
"""
mutable struct ExactSampler{S} <: MonteCarloSampler
    samples_length::Int
    seed::S
end

ExactSampler(n_samples; seed=rand(UInt)) =
    ExactSampler(n_samples, seed)

## Cache
mutable struct ExactSamplerCache{H,A<:AbstractRNG,T} <: SamplerCache{ExactSampler}
    hilb::H

    rng::A
    steps_done::Int
    pdf::Vector{T}
end

function _sampler_cache(s::ExactSampler, v, hilb, net, part)
    if net isa CachedNet
        if net.cache isa NNBatchedCache
            return _sampler_batched_cache(s, v, hilb, net, part)
        end
    end
    ExactSamplerCache(hilb,
                      MersenneTwister(s.seed),
                      0,
                      zeros(real(eltype(v)), spacedimension(hilb)))
end
##
function init_sampler!(sampler::ExactSampler, net::Union{MatrixNet,KetNet}, σ, c::ExactSamplerCache)
    # Compute the distribution
    for i=1:spacedimension(c.hilb)
        set!(σ, c.hilb, i)
        c.pdf[i] = exp(log_prob_ψ(net, σ))
    end
    c.pdf ./= sum(c.pdf)
    cumsum!(c.pdf, c.pdf)
    c.steps_done = 0

    samplenext!(σ, sampler, net, c)
    return c
end

chain_length(s::ExactSampler, c::ExactSamplerCache) = s.samples_length

done(s::ExactSampler, σ, c) = c.steps_done >= s.samples_length

function samplenext!(σ, s::ExactSampler, net::Union{MatrixNet,KetNet}, c)
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    # sample and set
    r = rand(c.rng)
    i = searchsortedfirst(c.pdf, r)
    set!(σ, i-1)
    c.steps_done += 1

    return true
end

# return findfirst(probs .> number)
function sample_from_distr(number::Real, probs::Vector)
    @inbounds for (i,p)=enumerate(probs)
        if p > number
            return i
        end
    end
end


######## Multithreaded
function _mt_recompute_sampler_params!(samplers, s::ExactSampler)
    nt = Threads.nthreads()
    _n_samples = Int(ceil(s.samples_length / nt))
    rng = MersenneTwister(s.seed)
    for i=1:Threads.nthreads()
        samplers[i] = ExactSampler(_n_samples, seed=rand(rng, UInt))
    end
end

function _mt_sampler_cache(s::MTSampler{T}, v, net, ::ParallelThreaded) where T<:ExactSampler
    pdf = zeros(real(eltype(v)), spacedimension(v))
    rng = MersenneTwister(s.seed)
    seeds = rand(UInt, Threads.nthreads())
    scs = Vector{ExactSamplerCache{typeof(rng), eltype(pdf)}}()
    for i=1:Threads.nthreads()
        push!(scs, ExactSamplerCache(MersenneTwister(seeds[i]), 0, pdf))
    end
    return MTSamplerCache(s, scs, net, v)
end

function mt_init_sampler(s::MTSampler{T}, net::Union{MatrixNet,KetNet}, σ, c::MTSamplerCache) where T<:ExactSampler
    np     = Threads.nthreads()

    Threads.@threads for i=1:Threads.nthreads()
        np     = Threads.nthreads()
        rank   = i - 1
        _σ = c.σs[i]
        _c = c.caches[i]
        _n = c.nets[i]
        n_min, extra = divrem(spacedimension(_σ), np)
        iter_length  = n_min + (rank < extra ? 1 : 0)
        iter_start   = n_min * rank + min(rank, extra) + 1
        iter_end     = iter_start + iter_length - 1

        for j=iter_start:iter_end
            set!(_σ, _c.hilb, j)
            _c.pdf[j] = exp(log_prob_ψ(_n, _σ))
        end
    end

    c.caches[1].pdf ./= sum(c.caches[1].pdf)
    cumsum!(c.caches[1].pdf, c.caches[1].pdf)

    Threads.@threads for i=1:np
        _σ   = c.σs[i]
        _c   = c.caches[i]
        _net = c.nets[i]
        _c.steps_done = 0

        samplenext!(_σ, sampler_list(s)[i], _net, _c)
    end
    return c
end


## Batched

mutable struct ExactSamplerBatchedCache{H,A<:AbstractRNG,T,V} <: SamplerCache{ExactSampler}
    hilb::H

    rng::A
    steps_done::Int
    pdf::Vector{T}

    logψ::V
end

_sampler_batched_cache(s::ExactSampler, v, hilb, net, part) =
    ExactSamplerBatchedCache(hilb,
                      MersenneTwister(s.seed),
                      0,
                      zeros(real(out_type(net)), spacedimension(hilb)),
                      out_similar(net))

##
function init_sampler!(sampler::ExactSampler,
                       net::Union{MatrixNet,KetNet},
                       σ, c::ExactSamplerBatchedCache)
    batch_sz = batch_size(net)
    # Compute the distribution
    for batch_i=Iterators.partition(1:spacedimension(c.hilb), batch_sz)
        for (i, h_i)=enumerate(batch_i)
            set!(unsafe_get_batch(σ, i), c.hilb, h_i)
        end
        log_prob_ψ!(c.logψ, c.logψ, net, σ)

        for (i, h_i)=enumerate(batch_i)
            c.pdf[h_i] = c.logψ[i]
        end
    end
    #c.pdf ./= sum(c.pdf)
    tot = sum(c.pdf)
    cumsum!(c.pdf, c.pdf)
    c.pdf ./= tot

    c.steps_done = 0

    samplenext!(σ, σ, sampler, net, c)
    return c
end

chain_length(s::ExactSampler, c::ExactSamplerBatchedCache) = s.samples_length

function samplenext!(σ_out::T, σ_in::T,
                     s::ExactSampler,
                     net::Union{MatrixNet,KetNet},
                     c::ExactSamplerBatchedCache) where {T<:Union{AStateBatch, ADoubleStateBatch}}
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    for i=1:batch_size(net)
        r = rand(c.rng)
        hi = searchsortedfirst(c.pdf, r)
        set!(unsafe_get_batch(σ_out, i), c.hilb, hi)
    end

    c.steps_done += 1
    return true
end
