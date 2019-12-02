# Cached version
mutable struct NDMBatchedCache{T,VT,VCT} <: NNBatchedCache{NDM}
    θλ_σ::VT
    θμ_σ::VT
    θλ_σp::VT
    θμ_σp::VT

    θλ_σ_tmp::VT
    θμ_σ_tmp::VT
    θλ_σp_tmp::VT
    θμ_σp_tmp::VT

    σr::VT
    σc::VT
    ∑σ::VT
    Δσ::VT

    ∑logℒ_λ_σ::T
    ∑logℒ_μ_σ::T
    ∑logℒ_λ_σp::T
    ∑logℒ_μ_σp::T
    ∂logℒ_λ_σ::VT
    ∂logℒ_μ_σ::VT
    ∂logℒ_λ_σp::VT
    ∂logℒ_μ_σp::VT

    Γ_λ::T
    Γ_μ::T
    Π::VCT

    _Π::VCT
    _Π2::VCT
    _Π_tmp::VT
    ∂logℒ_Π::VCT

    σ_row_cache::VT
    i_σ_row_cache::Int

    valid::Bool # = false
end

cache(net::NDM, batch_sz) = begin
    n_h = length(net.h_μ)
    n_v = length(net.b_μ)
    n_a = length(net.d_λ)
    CT  = complex(eltype(net.d_λ))

    NDMBatchedCache(similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),

              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),

              similar(net.b_μ, n_v, batch_sz),
              similar(net.b_μ, n_v, batch_sz),
              similar(net.b_μ, n_v, batch_sz),
              similar(net.b_μ, n_v, batch_sz),

              similar(net.b_μ, 1, batch_sz),
              similar(net.b_μ, 1, batch_sz),
              similar(net.b_μ, 1, batch_sz),
              similar(net.b_μ, 1, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),
              similar(net.h_μ, n_h, batch_sz),

              similar(net.b_μ, 1, batch_sz),
              similar(net.b_μ, 1, batch_sz),
              similar(net.b_μ, CT, 1, batch_sz),

              similar(net.d_λ, CT, n_a, batch_sz),
              similar(net.d_λ, CT, n_a, batch_sz),
              similar(net.d_λ,     n_a, batch_sz),
              similar(net.d_λ, CT, n_a, batch_sz),

              similar(net.b_μ, n_v, batch_sz),
              -1,

              false)
end

batch_size(c::NDMBatchedCache) = size(c.σr, 2)

function Base.show(io::IO, m::NDMBatchedCache)
    print(io, "NDMBatchedCache with batch-size = $(batch_size(m))")
end


function logψ!(out::AbstractMatrix, W::NDM, c::NDMBatchedCache, σr_r::AbstractMatrix, σc_r::AbstractMatrix)
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    θλ_σ    = c.θλ_σ
    θμ_σ    = c.θμ_σ
    θλ_σp   = c.θλ_σp
    θμ_σp   = c.θμ_σp
    _Π      = c._Π
    _Π2     = c._Π2
    _Π_tmp  = c._Π_tmp
    T       = eltype(c.θλ_σ)

    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        # θλ_σ .= W.h_λ + W.w_λ*σr
        mul!(θλ_σ, W.w_λ, σr)
        θλ_σ .+= W.h_λ

        # θμ_σ .= W.h_μ + W.w_μ*σr
        mul!(θμ_σ, W.w_μ, σr)
        θμ_σ .+= W.h_μ

        c.θλ_σ_tmp .= W.f.(θλ_σ)
        c.∑logℒ_λ_σ .= 0.0
        Base.mapreducedim!(identity, +, c.∑logℒ_λ_σ, c.θλ_σ_tmp)

        c.θμ_σ_tmp .= W.f.(θμ_σ)
        c.∑logℒ_μ_σ .= 0.0
        Base.mapreducedim!(identity, +, c.∑logℒ_μ_σ, c.θμ_σ_tmp)

        c.∂logℒ_λ_σ .= fwd_der.(W.f, θλ_σ)
        c.∂logℒ_μ_σ .= fwd_der.(W.f, θμ_σ)
    end

    ∑logℒ_λ_σ = c.∑logℒ_λ_σ
    ∑logℒ_μ_σ = c.∑logℒ_μ_σ
    ∂logℒ_λ_σ = c.∂logℒ_λ_σ
    ∂logℒ_μ_σ = c.∂logℒ_μ_σ

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    #θλ_σp .= W.h_λ + W.w_λ*σc
    #θμ_σp .= W.h_μ + W.w_μ*σc
    mul!(θλ_σp, W.w_λ, σc)
    θλ_σp .+= W.h_λ
    mul!(θμ_σp, W.w_μ, σc)
    θμ_σp .+= W.h_μ

    c.θλ_σp_tmp .= W.f.(θλ_σp)
    c.∑logℒ_λ_σp .= 0.0
    Base.mapreducedim!(identity, +, c.∑logℒ_λ_σp, c.θλ_σp_tmp)

    c.θμ_σp_tmp .= W.f.(θμ_σp)
    c.∑logℒ_μ_σp .= 0.0
    Base.mapreducedim!(identity, +, c.∑logℒ_μ_σp, c.θμ_σp_tmp)

    mul!(_Π_tmp, W.u_λ, ∑σ)
    _Π .= 0.5 .* _Π_tmp .+ W.d_λ
    mul!(_Π_tmp, W.u_μ, Δσ)
    _Π .+= T(0.5)im.* _Π_tmp

    Γ_λ = c.Γ_λ .=0.0
    Γ_μ = c.Γ_μ .=0.0
    mul!(Γ_λ, transpose(W.b_λ), ∑σ)
    Γ_λ .= T(0.5) .* (Γ_λ .+ c.∑logℒ_λ_σ .+ c.∑logℒ_λ_σp)
    mul!(Γ_μ, transpose(W.b_μ), Δσ)
    Γ_μ .= T(0.5) .* (Γ_μ .+ c.∑logℒ_μ_σ .- c.∑logℒ_μ_σp)

    _Π .= W.f.(_Π)
    Π = c.Π .=0.0
    Base.mapreducedim!(identity, +, Π, _Π)

    out .= Γ_λ .+ T(1.0)im .* Γ_μ .+ Π
    return out
end

function logψ_and_∇logψ!(∇logψ, out::AbstractMatrix, W::NDM, c::NDMBatchedCache, σr_r::AbstractMatrix, σc_r::AbstractMatrix)
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    θλ_σ    = c.θλ_σ
    θμ_σ    = c.θμ_σ
    θλ_σp   = c.θλ_σp
    θμ_σp   = c.θμ_σp
    _Π      = c._Π
    _Π2     = c._Π2
    _Π_tmp  = c._Π_tmp
    T       = eltype(c.θλ_σ)

    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    if !c.valid || c.σ_row_cache ≠ σr
        c.σ_row_cache .= σr
        c.valid = true

        # θλ_σ .= W.h_λ + W.w_λ*σr
        mul!(θλ_σ, W.w_λ, σr)
        θλ_σ .+= W.h_λ

        # θμ_σ .= W.h_μ + W.w_μ*σr
        mul!(θμ_σ, W.w_μ, σr)
        θμ_σ .+= W.h_μ

        c.θλ_σ_tmp .= W.f.(θλ_σ)
        c.∑logℒ_λ_σ .= 0.0
        Base.mapreducedim!(identity, +, c.∑logℒ_λ_σ, c.θλ_σ_tmp)

        c.θμ_σ_tmp .= W.f.(θμ_σ)
        c.∑logℒ_μ_σ .= 0.0
        Base.mapreducedim!(identity, +, c.∑logℒ_μ_σ, c.θμ_σ_tmp)

        c.∂logℒ_λ_σ .= fwd_der.(W.f, θλ_σ)
        c.∂logℒ_μ_σ .= fwd_der.(W.f, θμ_σ)
    end

    ∑logℒ_λ_σ = c.∑logℒ_λ_σ
    ∑logℒ_μ_σ = c.∑logℒ_μ_σ
    ∂logℒ_λ_σ = c.∂logℒ_λ_σ
    ∂logℒ_μ_σ = c.∂logℒ_μ_σ

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    #θλ_σp .= W.h_λ + W.w_λ*σc
    #θμ_σp .= W.h_μ + W.w_μ*σc
    mul!(θλ_σp, W.w_λ, σc)
    θλ_σp .+= W.h_λ
    mul!(θμ_σp, W.w_μ, σc)
    θμ_σp .+= W.h_μ

    c.θλ_σp_tmp .= W.f.(θλ_σp)
    c.∑logℒ_λ_σp .= 0.0
    Base.mapreducedim!(identity, +, c.∑logℒ_λ_σp, c.θλ_σp_tmp)

    c.θμ_σp_tmp .= W.f.(θμ_σp)
    c.∑logℒ_μ_σp .= 0.0
    Base.mapreducedim!(identity, +, c.∑logℒ_μ_σp, c.θμ_σp_tmp)

    mul!(_Π_tmp, W.u_λ, ∑σ)
    _Π .= 0.5 .* _Π_tmp .+ W.d_λ
    mul!(_Π_tmp, W.u_μ, Δσ)
    _Π .+= T(0.5)im.* _Π_tmp

    Γ_λ = c.Γ_λ .=0.0
    Γ_μ = c.Γ_μ .=0.0
    mul!(Γ_λ, transpose(W.b_λ), ∑σ)
    Γ_λ .= T(0.5) .* (Γ_λ .+ c.∑logℒ_λ_σ .+ c.∑logℒ_λ_σp)
    mul!(Γ_μ, transpose(W.b_μ), Δσ)
    Γ_μ .= T(0.5) .* (Γ_μ .+ c.∑logℒ_μ_σ .- c.∑logℒ_μ_σp)

    _Π2 .= W.f.(_Π)
    Π = c.Π .=0.0
    Base.mapreducedim!(identity, +, Π, _Π2)

    out .= Γ_λ .+ T(1.0)im .* Γ_μ .+ Π
    # --- End common terms with computation of ψ --- #
    ∂logℒ_λ_σp = c.∂logℒ_λ_σp; ∂logℒ_λ_σp .= fwd_der.(W.f, θλ_σp)
    ∂logℒ_μ_σp = c.∂logℒ_μ_σp; ∂logℒ_μ_σp .= fwd_der.(W.f, θμ_σp)
    ∂logℒ_Π    = c.∂logℒ_Π;    ∂logℒ_Π    .= fwd_der.(W.f, _Π)

    # Store the derivatives
    ∇logψ.b_λ .= T(0.5)   .* ∑σ
    ∇logψ.b_μ .= T(0.5)im .* Δσ

    ∇logψ.h_λ .= T(0.5)   .* (∂logℒ_λ_σ .+ ∂logℒ_λ_σp)
    ∇logψ.h_μ .= T(0.5)im .* (∂logℒ_μ_σ .- ∂logℒ_μ_σp)

    #∇logψ.w_λ .= T(0.5)   .* (∂logℒ_λ_σ.*transpose(σr) .+ ∂logℒ_λ_σp.*transpose(σc))
    #∇logψ.w_μ .= T(0.5)im .* (∂logℒ_μ_σ.*transpose(σr) .- ∂logℒ_μ_σp.*transpose(σc))
    _batched_outer_prod_∑!(∇logψ.w_λ, T(0.5), ∂logℒ_λ_σ, σr, ∂logℒ_λ_σp, σc)
    _batched_outer_prod_Δ!(∇logψ.w_μ, T(0.5)im, ∂logℒ_μ_σ, σr, ∂logℒ_μ_σp, σc)

    ∇logψ.d_λ .= ∂logℒ_Π
#    ∇logψ.u_λ .= T(0.5) .* ∂logℒ_Π .* transpose(∑σ)
#    ∇logψ.u_μ .= T(0.5)im .*  ∂logℒ_Π .* transpose(Δσ)
    _batched_outer_prod!(∇logψ.u_λ, T(0.5), ∂logℒ_Π, ∑σ)
    _batched_outer_prod!(∇logψ.u_μ, T(0.5)im, ∂logℒ_Π, Δσ)

    return out
end
