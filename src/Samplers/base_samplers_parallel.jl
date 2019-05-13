export get_sampler, sampler_list, multithread

"""
    MTSampler{ST} <: Sampler where ST<:Sampler

A multithreaded wrapper for a sampler. Can be created by calling
`multithread(<:Sampler)`. This structure has two fields, a sampler, and a list
of thread-local samplers. It can be accessed as a standard sampler. Whenever it
is accessed as that, it will recompute the parameters of every thread-local sampler.
"""
mutable struct MTSampler{ST} <: Sampler
    sampler::ST
    samplers::Vector{ST}
end

"""
    multithread(sampler)

Creates a multithreaded version of the sampler, which will use `Threads.nthreads()`
independent processes when sampling.
"""
function multithread(sampler::Sampler)
    mts = MTSampler(sampler, [deepcopy(sampler) for i=1:Threads.nthreads()])
    recompute_params!(mts)
    mts
end
# dont multithread multithreaded code
multithread(sampl::MTSampler) = sampl

# Every time we get a property from the MTSampler, we forward the call to the
# original sampler.
@inline function Base.getproperty(s::MTSampler, val::Symbol)
    return getfield(getfield(s,:sampler), val)
end

function Base.setproperty!(s::MTSampler, sim::Symbol, val)
    Base.setfield!(getfield(s, :sampler), sim, val)
    _mt_recompute_sampler_params!(getfield(s, :samplers), getfield(s, :sampler))
end

Base.propertynames(x::MTSampler, private=false) =
    Base.propertynames(get_sampler(x), private)


recompute_params!(s::MTSampler) = _mt_recompute_sampler_params!(getfield(s, :samplers), getfield(s, :sampler))

get_sampler(s::MTSampler) = getfield(s, :sampler)
sampler_list(s::MTSampler) = getfield(s, :samplers)

##
"""
    MTSamplerCache

A multithreading wrapper for SamplerCache, holding `num_threads` independent
copies of a `SamplerCache` and of the network, states.
"""
mutable struct MTSamplerCache{S,SC<:SamplerCache,V,N} <: SamplerCache{S}
    caches::Vector{SC}
    nets::Vector{N}
    σs::Vector{V}
end

MTSamplerCache(::S, caches, net::N, σ::V, op=copy) where {N,V,S<:Sampler} =
    MTSamplerCache{S,eltype(caches),V,N}(caches,
                                        [op(net) for i=1:length(caches)],
                                        [deepcopy(σ) for i=1:length(caches)])

# When initializing a MTCache the chain of function calls is the following:
# instead of calling directly _sampler_cache, _mt_sampler_cache is called, which
# is responsible for creating the multithreaded cache. You can override this, or
cache(s::MTSampler, v::State, net) =
    _mt_sampler_cache(s, v, net, ParallelThreaded())

# Mt sampler cache normally calls _sampler_cache for every thread. with the thread-local
# sampler. If this is fine enough, good, overwise you can override both _sampler_cache(.... ::ParallelThreaded)
# or the whole function.
function _mt_sampler_cache(s::MTSampler, v, net, ::ParallelThreaded)
    T = typeof(_sampler_cache(sampler_list(s)[1], v, net, ParallelThreaded()))
    scs = Vector{T}(undef, Threads.nthreads())

    Threads.@threads for i=1:Threads.nthreads()
        scs[i] = _sampler_cache(sampler_list(s)[i], v, net, ParallelThreaded())
    end

    return MTSamplerCache(s, scs, net, v)
end


# The call tree for init_sampler is the following: if called with multithreading
# structures, first it updates the weight in the cached networks, then it calls
# mt_init_sampler, which can be overridden by the user. If simple functionality
# is sufficient, then you can just override mt_init_sampler_perthread.
function init_sampler!(s::MTSampler, net, σ, c::MTSamplerCache)
    for cn=c.nets
        update!(Optimisers.Identity(), cn, net, nothing)
    end
    mt_init_sampler(s, net, σ, c)
end

function mt_init_sampler(s::MTSampler, net, σ, c::MTSamplerCache)
    Threads.@threads for i=1:Threads.nthreads()
        mt_init_sampler_perthread(sampler_list(s)[i],
                                  c.nets[i],
                                  c.σs[i],
                                  c.caches[i])
    end
    return c
end

mt_init_sampler_perthread(args...) = init_sampler!(args...)
