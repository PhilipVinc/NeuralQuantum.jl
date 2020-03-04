export BatchedSampler

mutable struct BatchedObsDMSampler{BN, P, S, Sc, Sv, Pv, LC, Lv, Pd} <: AbstractIterativeSampler
    observables::Dict

    bnet::BN
    hilb_do::P

    sampler::S
    sampler_cache::Sc
    samples::Sv

    ψvals::Pv

    accum::LC

    local_vals::Lv

    results::Dict

    parallel_cache::Pd
end

function BatchedObsDMSampler(bnet,
                        sampl,
                        hilb_doubled;
                        batch_sz=2^4,
                        local_batch_sz=batch_sz,
                        par_type=automatic_parallel_type())
    hilb_ph        = physical(hilb_doubled)

    par_cache      = parallel_execution_cache(par_type)

    v              = state(hilb_ph, bnet)
    sampler_cache  = cache(sampl, hilb_ph, bnet, par_cache)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = state(hilb_ph, bnet, ch_len)
    ψvals          = similar(trainable_first(bnet), out_type(bnet), 1, batch_sz, ch_len)

    local_acc      = AccumulatorObsScalar(bnet, hilb_doubled, v, local_batch_sz)
    Llocal_vals    = similar(ψvals, size(ψvals)[2:end]...)

    nq = BatchedObsDMSampler(Dict(), bnet, hilb_doubled,
            sampl, sampler_cache, samples,
            ψvals,
            local_acc, Llocal_vals,
            Dict(),
            par_cache)

    return nq
end

function add_observable!(is::BatchedObsDMSampler, name::String, obs::AbsLinearOperator)
    is.observables[name] = obs
end

function compute_observables(is::BatchedObsDMSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    # skip if no observables
    isempty(is.observables) && return nothing

    # Sample phase
    _sample_state!(is)

    # Compute logψ and ∇logψ
    logψ!(is.ψvals, is.bnet, is.samples)

    # Compute LdagL
    for (name, Ô) = is.observables
        is.results[name] = _local_obs_eval(is, Ô)
    end

    return is.results
end

function compute_observable(is::BatchedObsDMSampler, Ô)
    # Sample phase
    _sample_state!(is)

    # Compute logψ and ∇logψ
    logψ!(is.ψvals, is.bnet, is.samples)

    return _local_obs_eval(is, Ô)
end

function _local_obs_eval(is::BatchedObsDMSampler, Ô)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    for i=1:ch_len
        for j = 1:batch_sz
            σv = unsafe_get_el(is.samples, j, i)
            init!(is.accum, σv, is.ψvals[1,j,i])
            accumulate_connections!(is.accum, Ô, σv)
            O_loc = NeuralQuantum.finalize!(is.accum)
            is.local_vals[j, i] = O_loc
        end
    end

    return stat_analysis(is.local_vals, is.parallel_cache)
end

Base.show(io::IO, is::BatchedObsDMSampler) = print(io,
    "BatchedObsDMSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tsampler\t: $(is.sampler)")
