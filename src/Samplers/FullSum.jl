export FullSumSampler

struct FullSumSampler <: FullSpaceSampler end

## Cache
mutable struct FullSumSamplerCache{H,T} <: SamplerCache{FullSumSampler}
    hilb::H

    last_position::Int
    interval::T
end

_sampler_cache(s::FullSumSampler, v, hilb, net, part) =
    FullSumSamplerCache(hilb, 0, 1:spacedimension(hilb))

function init_sampler!(sampler::FullSumSampler, net, σ::FiniteBasisState, c::FullSumSamplerCache)
    c.last_position = 1
    # only initialize if it's really bigger.
    length(c.interval) >0 && set!(σ, c.hilb, c.interval[1])
    return c
end

chain_length(s::FullSumSampler, c::FullSumSamplerCache) = length(c.interval)

done(s::FullSumSampler, σ, c) = c.last_position >= length(c.interval)

function samplenext!(σ, s::FullSumSampler, net, c)
    done(s, σ, c) && return false
    c.last_position += 1
    set!(σ,c.hilb, c.interval[c.last_position])
    return true
end


######## Multithreaded
_mt_recompute_sampler_params!(samplers, s::FullSumSampler) = nothing

function _divide_in_blocks(interval, rank, n_par)
    rank         = rank -1
    n_min, extra = divrem(length(interval), n_par)
    iter_length  = n_min + (rank < extra ? 1 : 0)
    iter_start   = n_min * rank + min(rank, extra) + 1
    iter_end     = iter_start + iter_length - 1

    return iter_start:iter_end
end

_sampler_cache(s::FullSumSampler, v, hilb, net, ::ParallelThreaded, thread_i) =
    FullSumSamplerCache(hilb, 0, _divide_in_blocks(1:spacedimension(hilb), thread_i, Threads.nthreads()))
