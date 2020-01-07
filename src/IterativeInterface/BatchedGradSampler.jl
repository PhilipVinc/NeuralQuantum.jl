export BatchedSampler

mutable struct BatchedGradSampler{BN, P, S, Sc, Sv, Pv, Gv, Gvv, Gva, LC, Lv, Lgv, Pc} <: AbstractIterativeSampler
    bnet::BN
    Ĉ::P

    sampler::S
    sampler_cache::Sc
    samples::Sv

    ψvals::Pv
    ∇vals::Gv
    ∇vals_vec::Gvv
    ∇vec_avg::Gva

    accum::LC

    local_vals::Lv
    ∇local_vals::Lgv
    precond_cache::Pc

    observables_sampler
end

function BatchedGradSampler(net,
                        sampl,
                        prob,
                        algo=prob;
                        batch_sz=2^4,
                        local_batch_sz=batch_sz)
    if net isa CachedNet
        throw("Only takes standard network.")
    end

    bnet           = cached(net, batch_sz)
    v              = state(prob, bnet)
    sampler_cache  = init_sampler!(sampl, bnet, basis(prob), v)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = NeuralQuantum.vec_of_batches(v, ch_len)
    ψvals          = similar(trainable_first(bnet), out_type(bnet), 1, batch_sz, ch_len)
    ∇vals, ∇vec    = grad_cache(bnet, batch_sz, ch_len)
    ∇vec_avg       = tuple([similar(∇vec_i, size(∇vec_i, 1)) for ∇vec_i=∇vec]...)

    local_acc      = AccumulatorObsGrad(net, basis(prob), v, local_batch_sz)
    Llocal_vals    = collect(similar(ψvals, size(ψvals)[2:end]...)) #ensure it's on cpu #tocheck
    ∇Llocal_vals   = tuple([similar(ψvals, size(∇vec_i, 1), size(ψvals)[2:end]...) for ∇vec_i=∇vec]...)

    precond        = algorithm_cache(algo, prob, net)

    if prob isa KLocalLiouvillian
        obs        = BatchedObsDMSampler(bnet, sampl, basis(prob), batch_sz=batch_sz)
    else
        obs        = BatchedObsKetSampler(samples, ψvals, local_acc)
    end

    nq = BatchedGradSampler(bnet, prob,
            sampl, sampler_cache, samples,
            ψvals, ∇vals, ∇vec, ∇vec_avg,
            local_acc, Llocal_vals, ∇Llocal_vals, precond, obs)

    return nq
end

function sample!(is::BatchedGradSampler; sample=true)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    # Sample phase
    if sample
        init_sampler!(is.sampler, is.bnet, unsafe_get_el(is.samples, 1),
                        is.sampler_cache)
        for i=1:ch_len-1
            !samplenext!(unsafe_get_el(is.samples, i+1), unsafe_get_el(is.samples, i),
                            is.sampler, is.bnet, is.sampler_cache) && break
        end
    end
    # Compute logψ and ∇logψ
    logψ_and_∇logψ!(is.∇vals, is.ψvals, is.bnet, is.samples)

    # Compute LdagL
    for i=1:ch_len
        for j = 1:batch_sz
            σv = unsafe_get_el(is.samples, j, i)
            init!(is.accum, σv, is.ψvals[1,j,i], vec_data(is.∇vals[i])[1][:,j] )
            accumulate_connections!(is.accum, is.Ĉ, σv)
            L_loc, ∇L_loc = NeuralQuantum.finalize!(is.accum)
            is.local_vals[j, i]   = L_loc
            uview(is.∇local_vals[1], :, j, i) .= vec_data(∇L_loc)[1]
        end
    end

    Ĉ2 = abs2.(is.local_vals)

    L_stat = stat_analysis(Ĉ2)

    # Center the gradient so that it has zero-average
    for (∇vals_vec, ∇vec_avg)=zip(is.∇vals_vec, is.∇vec_avg)
        mean!(∇vec_avg, ∇vals_vec)
        ∇vals_vec .-= ∇vec_avg
    end

    # Compute the gradient
    # Flatten the batches and the iterations
    ∇vr = reshape(is.∇vals_vec[1], :, ch_len*batch_sz)
    ∇Ĉr = reshape(is.∇local_vals[1], :, ch_len*batch_sz)
    Ĉr  = reshape(is.local_vals, 1, :)
    #Ĉ2r = reshape(Ĉ2, 1, :)

    # Compute the gradient
    ∇C  = Ĉr*∇Ĉr'
    ∇C ./= (ch_len*batch_sz)
    ∇C .-= L_stat.mean .* is.∇vec_avg[1]'
    ∇C .= conj.(∇C)

    setup_algorithm!(is.precond_cache, reshape(∇C,:), ∇vr)

    return L_stat, is.precond_cache
end

function compute_observables(is::BatchedGradSampler)
    return compute_observables(is.observables_sampler)
end

function add_observable!(is::BatchedGradSampler, name::String, obs)
    return add_observable!(is.observables_sampler, name, obs)
end


Base.show(io::IO, is::BatchedGradSampler) = print(io,
    "BatchedGradSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tproblem\t: $(is.Ĉ)"*
    "\n\tsampler\t: $(is.sampler)")
