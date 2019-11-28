export BatchedSampler

mutable struct BatchedObsDMSampler{BN, P, S, Sc, Sv, Pv, Gv, Gvv, LC, Lv} <: AbstractIterativeSampler
    observables::Dict

    bnet::BN
    hilb_do::P

    sampler::S
    sampler_cache::Sc
    samples::Sv

    ψvals::Pv
    ∇vals::Gv
    ∇vals_vec::Gvv

    accum::LC

    local_vals::Lv

    results::Dict
end

function BatchedObsDMSampler(bnet,
                        sampl,
                        hilb_doubled;
                        batch_sz=2^4,
                        local_batch_sz=batch_sz)
    hilb_ph        = physical(hilb_doubled)

    #bnet           = cached(net, batch_sz)
    v              = state(hilb_ph, bnet)
    sampler_cache  = init_sampler!(sampl, bnet, hilb_ph, v)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = NeuralQuantum.vec_of_batches(v, ch_len)
    ψvals          = similar(trainable_first(bnet), out_type(bnet), 1, batch_sz, ch_len)
    ∇vals, ∇vec    = grad_cache(bnet, batch_sz, ch_len)

    local_acc      = AccumulatorObsScalar(bnet, hilb_doubled, v, local_batch_sz)
    Llocal_vals    = similar(ψvals, size(ψvals)[2:end]...)

    nq = BatchedObsDMSampler(Dict(), bnet, hilb_doubled,
            sampl, sampler_cache, samples,
            ψvals, ∇vals, ∇vec,
            local_acc, Llocal_vals,
            Dict())

    return nq
end

function add_observable(is::BatchedObsDMSampler, name::String, obs::AbsLinearOperator)
    is.observables[name] = obs
end

function compute_observables(is::BatchedObsDMSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    # Sample phase
    init_sampler!(is.sampler, is.bnet, unsafe_get_el(is.samples, 1),
                  is.sampler_cache)
    for i=1:ch_len-1
        !samplenext!(unsafe_get_el(is.samples, i+1), unsafe_get_el(is.samples, i),
                        is.sampler, is.bnet, is.sampler_cache) && break
    end

    # Compute logψ and ∇logψ
    logψ!(is.ψvals, is.bnet, is.samples)

    # Compute LdagL
    for (name, Ô) = is.observables
        for i=1:ch_len
            for j = 1:batch_sz
                σv = unsafe_get_el(is.samples, j, i)
                init!(is.accum, σv, is.ψvals[1,j,i], vec_data(is.∇vals[i])[1][:,j] )
                accumulate_connections!(is.accum, Ô, σv)
                O_loc = NeuralQuantum.finalize!(is.accum)
                is.local_vals[j, i] = O_loc
            end
        end
        is.results[name] = stat_analysis(is.local_vals)
    end

    return is.results
end

Base.show(io::IO, is::BatchedObsDMSampler) = print(io,
    "BatchedObsDMSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tproblem\t: $(is.problem)"*
    "\n\tsampler\t: $(is.sampler)")
