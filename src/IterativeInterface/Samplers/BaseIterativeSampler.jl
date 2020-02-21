abstract type AbstractIterativeSampler end

export gradient

function _sample_state!(is::AbstractIterativeSampler)
    ch_len         = chain_length(is.sampler, is.sampler_cache)
    batch_sz       = batch_size(is.samples)

    # Monte-Carlo sampling
    σ_old = unsafe_get_el(is.samples, 1)

    init_sampler!(is.sampler, is.bnet, σ_old, is.sampler_cache)
    for i=1:ch_len-1
        σ_next = unsafe_get_el(is.samples, i+1)
        !samplenext!(σ_next, σ_old,
                        is.sampler, is.bnet, is.sampler_cache) && break
        σ_old = σ_next
    end
end

function _center_gradient!(is::AbstractIterativeSampler)
    # Center the gradient so that it has zero-average
    # do it for every block of gradients / cogradients
    for (∇vec_avg, ∇vals_vec)=zip(is.∇vec_avg, is.∇vals_vec)
        workers_mean!(∇vec_avg, ∇vals_vec, is.parallel_cache) # MPI
        ∇vals_vec .-= ∇vec_avg
    end
end

function gradient(is::AbstractIterativeSampler, iter; kwargs...)
    loss, grad_before_precond = sample!(is; kwargs...)
    grad = precondition!(grad_before_precond, is.precond_algo, iter)
    return loss, grad
end
