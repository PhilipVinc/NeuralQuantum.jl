struct DenseSplitBatchedCache{Ta,Tb,Tc,Td,Te}
    σr::Tc
    σc::Tc
    out::Tb

    δℒℒ::Td
    δℒr::Te
    δℒc::Te

    θ::Ta
    θ_tmp::Ta
    valid::Bool
end

function cache(l::DenseSplit{Ta,Tb}, arr_T, in_T ,in_sz, batch_sz) where {Ta,Tb}
    c = DenseSplitBatchedCache(
            similar(l.Wr, size(l.Wr,2), batch_sz),
            similar(l.Wr, size(l.Wr,2), batch_sz),
            similar(l.b, size(l.b,1), batch_sz),

            similar(l.Wr, size(l.Wr,1), batch_sz),
            similar(l.Wr, size(l.Wr,2), batch_sz),
            similar(l.Wr, size(l.Wr,2), batch_sz),

            similar(l.b, size(l.b,1), batch_sz),
            similar(l.b, size(l.b,1), batch_sz),
            false)
    return c
end

batch_size(c::DenseSplitBatchedCache) = size(c.out, 2)

function (l::DenseSplit)(c::DenseSplitBatchedCache, (xr, xc))
    # The preallocated caches
    out  = c.out
    θ = c.θ
    θ₂ = c.θ_tmp

    # Store the input to this layer for the backpropagation
    σr = copyto!(c.σr, xr)
    σc = copyto!(c.σc, xc)

    #θ .= net.b .+ net.W * x
    mul!(θ, l.Wr, σr)
    mul!(θ₂, l.Wc, σc)
    θ .+= l.b .+ θ₂

    # Apply the nonlinear function
    out  .= l.σ.(θ)
    return out
end

function backprop(∇, l::DenseSplit, c::DenseSplitBatchedCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= fwd_der.(l.σ, θ)

    _batched_outer_prod_noconj!(∇.Wr, δℒℒ, c.σr)
    _batched_outer_prod_noconj!(∇.Wc, δℒℒ, c.σc)
    ∇.b .= δℒℒ

    δℒr = mul!(c.δℒr, transpose(l.Wr), δℒℒ)
    δℒc = mul!(c.δℒc, transpose(l.Wc), δℒℒ)
    return (δℒr, δℒc)
end
