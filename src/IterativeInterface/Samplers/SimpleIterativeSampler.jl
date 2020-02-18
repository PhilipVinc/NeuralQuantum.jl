export BatchedSampler

mutable struct SimpleIterativeSampler{BN, S, Sc, Sv, Pd} <: AbstractIterativeSampler
    bnet::BN

    sampler::S
    sampler_cache::Sc
    samples::Sv

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
    sampler_cache  = init_sampler!(sampl, bnet, hilb, v, par_cache)
    sampler_cache  = cache(sampl, hilb, bnet, par_cache)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = NeuralQuantum.vec_of_batches(v, ch_len)

    sis = SimpleIterativeSampler(bnet,
            sampl, sampler_cache, samples,
            par_cache)

    return sis
end

function sample!(is::SimpleIterativeSampler)
    # Sample the new configurations
    _sample_state!(is)

    return is.samples
end


Base.show(io::IO, is::SimpleIterativeSampler) = print(io,
    "SimpleIterativeSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tsampler\t: $(is.sampler)")
