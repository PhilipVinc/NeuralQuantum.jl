struct NDMSymmBatchedCache{BC,BD} <: NNBatchedCache{NDMSymm}
    bare_cache::BC
    bare_∇logψ::BD
end

cache(net::NDMSymm, batch_sz) =
    NDMSymmBatchedCache(cache(net.bare_net, batch_sz),
                        grad_cache(net.bare_net, batch_sz))

batch_size(c::NDMSymmBatchedCache) = batch_size(c.bare_cache)

logψ!(out::AbstractMatrix, net::NDMSymm, c::NDMSymmBatchedCache, σr_r::AbstractMatrix, σc_r::AbstractMatrix) =
    logψ!(out, net.bare_net, c.bare_cache, σr_r, σc_r)

function logψ_and_∇logψ!(∇logψ, out::AbstractMatrix, net::NDMSymm, c::NDMSymmBatchedCache, σr_r::AbstractMatrix, σc_r::AbstractMatrix)
    out = logψ_and_∇logψ!(c.bare_∇logψ, out, net.bare_net, c.bare_cache, σr_r, σc_r)

    symmetrize_∇logψ_NDM_batched!(∇logψ, c.bare_∇logψ, net)
    return out
end

function symmetrize_∇logψ_NDM_batched!(∇lnψ_symm, ∇lnψ, net)
    mul!(∇lnψ_symm.b_μ, net.∇b_mat, ∇lnψ.b_μ)
    mul!(∇lnψ_symm.b_λ, net.∇b_mat, ∇lnψ.b_λ)
    mul!(∇lnψ_symm.h_μ, net.∇h_mat, ∇lnψ.h_μ)
    mul!(∇lnψ_symm.h_λ, net.∇h_mat, ∇lnψ.h_λ)
    mul!(∇lnψ_symm.d_λ, net.∇d_mat, ∇lnψ.d_λ)
    for i=1:size(∇lnψ_symm.w_μ, 3)
        mul!(_uview_vec_destride(∇lnψ_symm.w_μ, i), net.∇w_mat, _uview_vec_destride(∇lnψ.w_μ, i))
        mul!(_uview_vec_destride(∇lnψ_symm.w_λ, i), net.∇w_mat, _uview_vec_destride(∇lnψ.w_λ, i))
        mul!(_uview_vec_destride(∇lnψ_symm.u_μ, i), net.∇u_mat, _uview_vec_destride(∇lnψ.u_μ, i))
        mul!(_uview_vec_destride(∇lnψ_symm.u_λ, i), net.∇u_mat, _uview_vec_destride(∇lnψ.u_λ, i))
    end

    return ∇lnψ_symm
end
