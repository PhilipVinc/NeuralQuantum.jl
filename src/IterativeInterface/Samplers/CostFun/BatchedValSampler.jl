export BatchedSampler

mutable struct BatchedValSampler{BN, P, S, Sc, Sv, Pv, Gv, Gvv, Gva, LC, Lv, Pc, Pd} <: AbstractIterativeSampler
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

    parallel_cache::Pd
end

function BatchedValSampler(net,
                        sampl,
                        prob,
                        algo=prob;
                        batch_sz=2^4,
                        local_batch_sz=batch_sz,
                        par_type=automatic_parallel_type())
    if net isa CachedNet
        throw("Only takes standard network.")
    end

    par_cache      = parallel_execution_cache(par_type)

    bnet           = cached(net, batch_sz)
    v              = state(prob, bnet)
    sampler_cache  = cache(sampl, basis(prob), bnet, par_cache)

    ch_len         = chain_length(sampl, sampler_cache)

    samples        = NeuralQuantum.vec_of_batches(v, ch_len)
    ψvals          = similar(trainable_first(bnet), out_type(bnet), 1, batch_sz, ch_len)
    ∇vals, ∇vecs   = grad_cache(bnet, batch_sz, ch_len)
    ∇vec_avg       = tuple([similar(∇vec, size(∇vec, 1)) for ∇vec=∇vecs ]...)

    local_acc      = AccumulatorObsScalar(net, basis(prob), v, local_batch_sz)
    Llocal_vals    = collect(similar(ψvals, size(ψvals)[2:end]...)) #ensure it's on cpu

    precond        = algorithm_cache(algo, prob, net, par_cache)

    obs            = BatchedObsKetSampler(samples, ψvals, local_acc)

    nq = BatchedValSampler(bnet, prob,
            sampl, sampler_cache, samples,
            ψvals, ∇vals, ∇vecs, ∇vec_avg,
            local_acc, Llocal_vals, precond, obs,
            par_cache)

    return nq
end

function _sample_compute_obs!(is::BatchedValSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

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
    L_stat = stat_analysis(is.local_vals, is.parallel_cache)

    return L_stat
end

function _compute_gradient!(is::BatchedValSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = size(is.local_vals, 1)

    # Flatten the batches and the iterations
    ∇vr = reshape(is.∇vals_vec[1], :, ch_len*batch_sz)
    Ĉr  = reshape(is.local_vals, 1, :)

    if typeof(is.ψvals) <: GPUArray
        Ĉr = CuArrays.adapt(CuArray, Ĉr)
    end

    # Compute the gradient
    ∇C  = Ĉr*∇vr'
    ∇C ./= (ch_len*batch_sz) # MPI
    workers_mean!(∇C, is.parallel_cache)

    return ∇C, ∇vr
end

function sample!(is::BatchedValSampler; sample = true)
    # Sample the new configurations
    sample && _sample_state!(is)

    # Compute logψ and ∇logψ
    logψ_and_∇logψ!(is.∇vals, is.ψvals, is.bnet, is.samples)

    # Compute the cost function
    L_stat = _sample_compute_obs!(is)

    # Center the gradient
    _center_gradient!(is)

    # Compute the cost-gradient for optimization
    ∇C, ∇vr = _compute_gradient!(is)

    # Setup the algorithm.
    # IF we are doing gradient descent, this does nothing, otherwise initializes
    # the SR/Natural gradient structures.
    setup_algorithm!(is.precond_cache, reshape(∇C,:), ∇vr, is.parallel_cache)

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
