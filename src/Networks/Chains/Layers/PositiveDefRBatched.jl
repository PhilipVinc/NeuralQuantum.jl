struct PositiveDefRBatchedCache{Ta,Tb,Tc,Td,Te}
    σr::Tc
    σc::Tc

    ∑σ::Tc
    Δσ::Tc

    out::Tb

    δℒℒ::Td
    δℒr::Te
    δℒc::Te

    θ::Tb
    θr::Ta
    θr2::Ta
    valid::Bool
end

function cache(l::PositiveDefR{Ta,Tb}, arr_T, in_T ,in_sz, batch_sz) where {Ta,Tb}
    c = PositiveDefRBatchedCache(
            similar(l.Wr, size(l.Wr,2), batch_sz),
            similar(l.Wr, size(l.Wr,2), batch_sz),

            similar(l.Wr, size(l.Wr,2), batch_sz),
            similar(l.Wr, size(l.Wr,2), batch_sz),

            similar(l.b, Complex{eltype(l.b)}, size(l.b,1), batch_sz),

            similar(l.Wr, Complex{eltype(l.b)}, size(l.Wr,1), batch_sz),
            similar(l.Wr, Complex{eltype(l.b)}, size(l.Wr,2), batch_sz),
            similar(l.Wr, Complex{eltype(l.b)}, size(l.Wr,2), batch_sz),

            similar(l.b, Complex{eltype(l.b)}, size(l.b,1), batch_sz),
            similar(l.b, size(l.b,1), batch_sz),
            similar(l.b, size(l.b,1), batch_sz),
            false)
    return c
end

batch_size(c::PositiveDefRBatchedCache) = size(c.out, 2)

function (l::PositiveDefR)(c::PositiveDefRBatchedCache, (xr, xc))
    # The preallocated caches
    out  = c.out
    θ    = c.θ
    θᵣ₁  = c.θr
    θᵣ₂  = c.θr2


    # Store the input to this layer for the backpropagation
    σr = copyto!(c.σr, xr)
    σc = copyto!(c.σc, xc)

    c.∑σ .= σr .+ σc
    c.Δσ .= σr .- σc

    #θ .= net.b .+ net.W * x
    mul!(θᵣ₁, l.Wr, c.∑σ)
    mul!(θᵣ₂, l.Wi, c.Δσ)
    θ .= θᵣ₁ .+ im .* θᵣ₂ .+ l.b

    # Apply the nonlinear function
    out  .= l.σ.(θ)
    return out
end

function backprop(∇, l::PositiveDefR, c::PositiveDefRBatchedCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= fwd_der.(l.σ, θ)

    _batched_outer_prod_noconj!(∇.Wr, δℒℒ, c.∑σ)
    _batched_outer_prod_noconj!(∇.Wi, δℒℒ, c.Δσ)
    ∇.Wi .*= im

    ∇.b .= δℒℒ

    δℒr = 0 #mul!(c.δℒr, transpose(l.Wr), δℒℒ)
    δℒc = 0 #mul!(c.δℒc, transpose(l.Wi), δℒℒ)
    return (δℒr, δℒc)
end
