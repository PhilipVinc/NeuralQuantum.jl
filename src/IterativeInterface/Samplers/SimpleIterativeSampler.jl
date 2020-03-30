export BatchedSampler

mutable struct SimpleIterativeSampler{BN, S, Sc, Sv, H, Pd} <: AbstractIterativeSampler
    bnet::BN

    sampler::S
    sampler_cache::Sc
    samples::Sv

    hilb::H

    parallel_cache::Pd
end

function SimpleIterativeSampler(net,
                                sampl,
                                hilb;
                                batch_sz=2^4,
                                par_type=automatic_parallel_type())
    if net isa CachedNet
        throw("Only takes standard network.")
    end

    if hilb isa AbstractOperator
        hilb = basis(hilb)
    end

    par_cache      = parallel_execution_cache(par_type)

    bnet           = cached(net, batch_sz)
    v              = state(hilb, bnet)
    sampler_cache  = cache(sampl, v, hilb, bnet, par_cache)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = state(hilb, bnet, ch_len)

    sis = SimpleIterativeSampler(bnet,
            sampl, sampler_cache, samples,
            hilb,
            par_cache)

    return sis
end

function sample!(is::SimpleIterativeSampler)
    # Sample the new configurations
    _sample_state!(is)

    return is.samples
end

samples(is::SimpleIterativeSampler) = is.samples
chain_length(is::SimpleIterativeSampler) = chain_length(is.sampler, is.sampler_cache)

Base.show(io::IO, is::SimpleIterativeSampler) = print(io,
    "SimpleIterativeSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tsampler\t: $(is.sampler)")
