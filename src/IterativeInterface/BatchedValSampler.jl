export BatchedSampler

mutable struct BatchedValSampler{BN, P, S, Sc, Sv, Pv, Gv, Gvv, Gva, LC, Lv, Pc} <: AbstractIterativeSampler
    bnet::BN
    Ĉ::P # the operator defining the observable to minimise

    sampler::S
    sampler_cache::Sc
    samples::Sv

    ψvals::Pv
    ∇vals::Gv
    ∇vals_vec::Gvv
    ∇vec_avg::Gva

    accum::LC

    local_vals::Lv
    precond_cache::Pc

    observables_sampler
end

function BatchedValSampler(net,
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
    ∇vec_avg       = similar(∇vec, size(∇vec, 1))

    local_acc      = AccumulatorObsScalar(net, basis(prob), v, local_batch_sz)
    Llocal_vals    = collect(similar(ψvals, size(ψvals)[2:end]...)) #ensure it's on cpu

    precond        = algorithm_cache(algo, prob, net)

    obs            = BatchedObsKetSampler(samples, ψvals, local_acc)

    nq = BatchedValSampler(bnet, prob,
            sampl, sampler_cache, samples,
            ψvals, ∇vals, ∇vec, ∇vec_avg,
            local_acc, Llocal_vals, precond, obs)

    return nq
end

function sample!(is::BatchedValSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    # Monte-Carlo sampling
    σ_old = unsafe_get_el(is.samples, 1)

    init_sampler!(is.sampler, is.bnet, σ_old, is.sampler_cache)
    for i=1:ch_len-1
        σ_next = unsafe_get_el(is.samples, i+1)
        !samplenext!(σ_next, σ_old,
                        is.sampler, is.bnet, is.sampler_cache) && break
        σ_old = σ_next
    end

    # Compute logψ and ∇logψ
    logψ_and_∇logψ!(is.∇vals, is.ψvals, is.bnet, is.samples)

    # If we have gpu, this thing is indexed by value so better
    # to copy it to the cpu first in one go...
    if typeof(is.ψvals) <: GPUArray
        ψvals   = collect(is.ψvals)
        samples = collect(is.samples)
    else
        ψvals   = is.ψvals
        samples = is.samples
    end

    # Compute terms C^{loc} = ⟨σ|Ĉ|ψ⟩/⟨σ|ψ⟩
    for i=1:ch_len
        for j = 1:batch_sz
            σv = unsafe_get_el(samples, j, i)
            init!(is.accum, σv, ψvals[1,j,i])
            accumulate_connections!(is.accum, is.Ĉ, σv)
            L_loc = NeuralQuantum.finalize!(is.accum)
            is.local_vals[j, i] = L_loc
        end
    end

    # Perform some simple analysis to estimate error and autocorrelations
    L_stat = stat_analysis(is.local_vals)

    # Center the gradient so that it has zero-average
    mean!(is.∇vec_avg, is.∇vals_vec)
    is.∇vals_vec .-= is.∇vec_avg

    # Flatten the batches and the iterations
    ∇vr = reshape(is.∇vals_vec, :, ch_len*batch_sz)
    Ĉr  = reshape(is.local_vals, 1, :)

    # Compute the gradient
    ∇C  = Ĉr*∇vr'
    ∇C ./= (ch_len*batch_sz)

    # Setup the algorithm.
    # IF we are doing gradient descent, this does nothing, otherwise initializes
    # the SR/Natural gradient structures.
    setup_algorithm!(is.precond_cache, reshape(∇C,:), ∇vr)

    return L_stat, is.precond_cache
end

function compute_observables(is::BatchedValSampler)
    return compute_observables(is.observables_sampler)
end

function add_observable!(is::BatchedValSampler, name::String, obs)
    return add_observable!(is.observables_sampler, name, obs)
end

Base.show(io::IO, is::BatchedValSampler) = print(io,
    "BatchedValSampler for :"*
    "\n\tnet\t\t: $(is.bnet)"*
    "\n\tproblem\t: $(is.Ĉ)"*
    "\n\tsampler\t: $(is.sampler)")
