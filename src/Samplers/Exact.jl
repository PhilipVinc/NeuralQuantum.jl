export ExactSampler

mutable struct ExactSampler{S} <: MonteCarloSampler
    samples_length::Int
    seed::S
end

ExactSampler(n_samples; seed=rand(UInt)) =
    ExactSampler(n_samples, seed)

## Cache
mutable struct ExactSamplerCache{A<:AbstractRNG,T} <: SamplerCache{ExactSampler}
    rng::A
    steps_done::Int
    pdf::Vector{T}
end

_sampler_cache(s::ExactSampler, v, net, part) =
    ExactSamplerCache(MersenneTwister(s.seed),
                      0,
                      zeros(real(eltype(v)), spacedimension(v)))

##
function init_sampler!(sampler::ExactSampler, net::Union{MatrixNet,KetNet}, σ, c::ExactSamplerCache)
    # Compute the distribution
    for i=1:spacedimension(σ)
        set!(σ, i-1)
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
            set_index!(_σ, j)
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
