export MTIterativeSampler

mutable struct MTIterativeSampler{N,P,IC,MIC,EC,S,SC,V} <: AbstractIterativeSampler
    net::N
    problem::P
    itercache::IC
    mitercache::MIC
    sampled_values::EC

    sampler::S
    sampler_cache::SC
    v::V
end

"""
    MTIterativeSampler(net, sampler, algorithm, prob)

Creates a multithreaded iterative sampler.
"""
function MTIterativeSampler(net,
                            sampl,
                            prob,
                            algo=prob)
    sampl = multithread(sampl)

    itercache      = SamplingCache(algo, prob, net)
    mitercache     = [SamplingCache(algo, prob, net) for i=1:Threads.nthreads()]
    v              = state(prob, net)
    sampler_cache  = init_sampler!(sampl, net, v)

    evaluated_vals = EvaluatedNetwork(algo, net)

    MTIterativeSampler(net, prob, itercache, mitercache, evaluated_vals, sampl, sampler_cache, v)
end

function sample!(is::MTIterativeSampler)
    # Init the samplers on each thread
    init_sampler!(is.sampler, is.net, is.v, is.sampler_cache)

    # Sample on every thread
    Threads.@threads for i=1:Threads.nthreads()
        s = sampler_list(is.sampler)[i]
        n = is.sampler_cache.nets[i]
        v = is.sampler_cache.σs[i]
        _ic = is.mitercache[i]
        zero!(_ic)
        while true
            if !(s isa FullSumSampler)
                sample_network!(_ic, is.problem, n, v)
            else
                sample_network_wholespace!(_ic, is.problem, n, v)
            end
            !samplenext!(v, s, n, is.sampler_cache.caches[i]) && break
        end
    end

    # Add results of all threads
    zero!(is.itercache);
    for ic=is.mitercache
        NeuralQuantum.add!(is.itercache, ic)
    end
    evaluation_post_sampling!(is.sampled_values, is.itercache)
    return is.sampled_values
end

Base.show(io::IO, is::MTIterativeSampler) = print(io,
    "MTIterativeSampler for parallel sampling with $(Threads.nthreads()) threads:"*
    "\n\tnet\t\t: $(is.net)"*
    "\n\tproblem\t: $(is.problem)"*
    "\n\tsampler\t: $(is.sampler)")
