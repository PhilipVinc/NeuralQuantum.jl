export IterativeSampler

mutable struct IterativeSampler{N,P,IC,EC,S,SC,V} <: AbstractIterativeSampler
    net::N
    problem::P
    itercache::IC
    sampled_values::EC

    sampler::S
    sampler_cache::SC
    v::V
end


"""
    IterativeSampler(net, sampler, algorithm, problem)

Create a single-thread iterative sampler for the quantities defined
by algorithm.
"""
function IterativeSampler(net,
                          sampl,
                          prob,
                          algo=prob)

    evaluated_vals = EvaluatedNetwork(algo, net)
    itercache      = SamplingCache(algo, prob, net)
    v              = state(prob, net)
    sampler_cache  = init_sampler!(sampl, net, v)

    IterativeSampler(net, prob, itercache, evaluated_vals, sampl, sampler_cache, v)
end

"""
    sample!(is::IterativeSampler)

Samples the quantities accordingly, returning the sampled values and sampling history.
"""
function sample!(is::IterativeSampler)
    init_sampler!(is.sampler, is.net, is.v, is.sampler_cache)
    zero!(is.itercache);
    while true
        if !(is.sampler isa FullSumSampler)
            sample_network!(is.itercache, is.problem, is.net, is.v)
        else
            sample_network_wholespace!(is.itercache, is.problem, is.net, is.v)
        end
        !samplenext!(is.v, is.sampler, is.net, is.sampler_cache) && break
    end

    evaluation_post_sampling!(is.sampled_values, is.itercache)
    return is.sampled_values
end

Base.show(io::IO, is::IterativeSampler) = print(io,
    "IterativeSampler for :"*
    "\n\tnet\t\t: $(is.net)"*
    "\n\tproblem\t: $(is.problem)"*
    "\n\tsampler\t: $(is.sampler)")
