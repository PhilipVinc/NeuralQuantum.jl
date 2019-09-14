mutable struct NDMLUT{VT,VCT} <: NNLookUp
    θλ_σ::VT
    θμ_σ::VT
    θλ_σp::VT
    θμ_σp::VT
    _Π::VCT
end

lookup(net::NDM) = NDMLUT(net)
NDMLUT(net::NDM) = NDMLUT(similar(net.h_μ),
                          similar(net.h_μ),
                          similar(net.h_μ),
                          similar(net.h_μ),
                          similar(net.d_λ, complex(eltype(net.d_λ))))

function set_lookup!(lt::NDMLUT, W::NDM, c::NDMCache, σr, σc)
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    _Π_tmp  = c._Π_tmp
    T       = eltype(c.θλ_σ)

    # θλ_σ .= W.h_λ + W.w_λ*σr
    LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, lt.θλ_σ, 1)
    LinearAlgebra.BLAS.gemv!('N', one(T), W.w_λ, σr, one(T), lt.θλ_σ)

    # θμ_σ .= W.h_μ + W.w_μ*σr
    LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, lt.θμ_σ, 1)
    LinearAlgebra.BLAS.gemv!('N', one(T), W.w_μ, σr, one(T), lt.θμ_σ)

    #θλ_σp .= W.h_λ + W.w_λ*σc
    LinearAlgebra.BLAS.blascopy!(length(W.h_λ), W.h_λ, 1, lt.θλ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_λ, σc, T(1.0), lt.θλ_σp)

    #θμ_σp .= W.h_μ + W.w_μ*σc
    LinearAlgebra.BLAS.blascopy!(length(W.h_μ), W.h_μ, 1, lt.θμ_σp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(1.0), W.w_μ, σc, T(1.0), lt.θμ_σp)

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    #_Π         = (T(0.5)  * W.u_λ*∑σ
    #               + T(0.5)im * W.u_μ*Δσ .+ W.d_λ)
    LinearAlgebra.BLAS.blascopy!(length(W.d_λ), W.d_λ, 1, _Π_tmp, 1)
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_λ, ∑σ, one(T), _Π_tmp)
    lt._Π .= _Π_tmp
    LinearAlgebra.BLAS.gemv!('N', T(0.5), W.u_μ, Δσ, T(0.0), _Π_tmp)
    lt._Π .+= T(1.0)im.* _Π_tmp

    return lt
end

function update_lookup!(lt::NDMLUT, W::NDM, c::NDMCache, σr, σc,
                         changes_r, changes_c)
    θλ_σ    = lt.θλ_σ
    θμ_σ    = lt.θμ_σ
    θλ_σp   = lt.θλ_σp
    θμ_σp   = lt.θμ_σp
    Π       = lt._Π
    T       = eltype(c.θλ_σ)

    for (i, nv) = changes_r
        θλ_σ .+= W.w_λ[:,i] .* (nv - σr[i])
        θμ_σ .+= W.w_μ[:,i] .* (nv - σr[i])
        Π    .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σr[i]) .+
                 T(0.5)im .* W.u_μ[:,i] .* (nv - σr[i])
    end
    for (i, nv) = changes_c
        θλ_σp .+= W.w_λ[:,i] .* (nv - σc[i])
        θμ_σp .+= W.w_μ[:,i] .* (nv - σc[i])
        Π     .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σc[i]) .-
                  T(0.5)im .* W.u_μ[:,i] .* (nv - σc[i])
    end

    return lt
end

function logψ(W::NDM, c::NDMCache, lt::NDMLUT, σr, σc,
              changes_r, changes_c)
    θλ_σ    = c.θλ_σ;   θλ_σ  .= lt.θλ_σ
    θμ_σ    = c.θμ_σ;   θμ_σ  .= lt.θμ_σ
    θλ_σp   = c.θλ_σp;  θλ_σp .= lt.θλ_σp
    θμ_σp   = c.θμ_σp;  θμ_σp .= lt.θμ_σp
    Π       = c._Π;     Π     .= lt._Π
    Π_tmp   = c._Π2
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    T       = eltype(θλ_σ)

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    for (i, nv) = changes_r
        θλ_σ .+= W.w_λ[:,i] .* (nv - σr[i])
        θμ_σ .+= W.w_μ[:,i] .* (nv - σr[i])
        Π    .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σr[i]) .+
                 T(0.5)im .* W.u_μ[:,i] .* (nv - σr[i])
        ∑σ[i] += (nv - σr[i])
        Δσ[i] += (nv - σr[i])
    end
    for (i, nv) = changes_c
        θλ_σp .+= W.w_λ[:,i] .* (nv - σc[i])
        θμ_σp .+= W.w_μ[:,i] .* (nv - σc[i])
        Π     .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σc[i]) .-
                  T(0.5)im .* W.u_μ[:,i] .* (nv - σc[i])
        ∑σ[i] += (nv - σc[i])
        Δσ[i] -= (nv - σc[i])
    end

    c.θλ_σ_tmp .= logℒ.(θλ_σ)
    ∑logℒ_λ_σ = sum(c.θλ_σ_tmp)

    c.θμ_σ_tmp .= logℒ.(θμ_σ)
    ∑logℒ_μ_σ = sum(c.θμ_σ_tmp)

    c.θλ_σp_tmp .= logℒ.(θλ_σp)
    ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)

    c.θμ_σp_tmp .= logℒ.(θμ_σp)
    ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)

    Π_tmp     .= logℒ.(Π)
    ∑Π        = sum(Π_tmp)

    Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp + ∑σ ⋅ W.b_λ)
    Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp + Δσ ⋅ W.b_μ)
    logψ = Γ_λ + T(1.0)im * Γ_μ + ∑Π

    return logψ
end

function logψ_and_∇logψ!(∇logψ, W::NDM, c::NDMCache, lt::NDMLUT,
                         σr_r, σc_r, changes_r, changes_c)
    θλ_σ    = c.θλ_σ;   θλ_σ  .= lt.θλ_σ
    θμ_σ    = c.θμ_σ;   θμ_σ  .= lt.θμ_σ
    θλ_σp   = c.θλ_σp;  θλ_σp .= lt.θλ_σp
    θμ_σp   = c.θμ_σp;  θμ_σp .= lt.θμ_σp
    Π       = c._Π;     Π     .= lt._Π
    Π_tmp   = c._Π2
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    T       = eltype(θλ_σ)

    σr = c.σr; σr .= σr_r
    σc = c.σc; σc .= σc_r
    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    for (i, nv) = changes_r
        θλ_σ .+= W.w_λ[:,i] .* (nv - σr[i])
        θμ_σ .+= W.w_μ[:,i] .* (nv - σr[i])
        Π    .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σr[i]) .+
                 T(0.5)im .* W.u_μ[:,i] .* (nv - σr[i])
        ∑σ[i] += (nv - σr[i])
        Δσ[i] += (nv - σr[i])
        σr[i] = nv
    end
    for (i, nv) = changes_c
        θλ_σp .+= W.w_λ[:,i] .* (nv - σc[i])
        θμ_σp .+= W.w_μ[:,i] .* (nv - σc[i])
        Π     .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σc[i]) .-
                  T(0.5)im .* W.u_μ[:,i] .* (nv - σc[i])
        ∑σ[i] += (nv - σc[i])
        Δσ[i] -= (nv - σc[i])
        σc[i] = nv
    end

    c.θλ_σ_tmp .= logℒ.(θλ_σ)
    ∑logℒ_λ_σ = sum(c.θλ_σ_tmp)

    c.θμ_σ_tmp .= logℒ.(θμ_σ)
    ∑logℒ_μ_σ = sum(c.θμ_σ_tmp)

    c.θλ_σp_tmp .= logℒ.(θλ_σp)
    ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)

    c.θμ_σp_tmp .= logℒ.(θμ_σp)
    ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)

    Π_tmp     .= logℒ.(Π)
    ∑Π        = sum(Π_tmp)

    Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp + ∑σ ⋅ W.b_λ)
    Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp + Δσ ⋅ W.b_μ)
    logψ = Γ_λ + T(1.0)im * Γ_μ + ∑Π

    # --- End common terms with computation of ψ --- #

    # Compute additional terms for derivatives
    ∂logℒ_λ_σ  = c.∂logℒ_λ_σ;  ∂logℒ_λ_σ  .= ∂logℒ.(θλ_σ)
    ∂logℒ_μ_σ  = c.∂logℒ_μ_σ;  ∂logℒ_μ_σ  .= ∂logℒ.(θμ_σ)
    ∂logℒ_λ_σp = c.∂logℒ_λ_σp; ∂logℒ_λ_σp .= ∂logℒ.(θλ_σp)
    ∂logℒ_μ_σp = c.∂logℒ_μ_σp; ∂logℒ_μ_σp .= ∂logℒ.(θμ_σp)
    ∂logℒ_Π    = c.∂logℒ_Π;    ∂logℒ_Π    .= ∂logℒ.(Π)

    # Store the derivatives
    ∇logψ.b_λ .= T(0.5)   .* ∑σ
    ∇logψ.b_μ .= T(0.5)im .* Δσ

    ∇logψ.h_λ .= T(0.5)   .* (∂logℒ_λ_σ .+ ∂logℒ_λ_σp)
    ∇logψ.h_μ .= T(0.5)im .* (∂logℒ_μ_σ .- ∂logℒ_μ_σp)

    ∇logψ.w_λ .= T(0.5)   .* (∂logℒ_λ_σ.*transpose(σr) .+ ∂logℒ_λ_σp.*transpose(σc))
    ∇logψ.w_μ .= T(0.5)im .* (∂logℒ_μ_σ.*transpose(σr) .- ∂logℒ_μ_σp.*transpose(σc))

    ∇logψ.d_λ .= ∂logℒ_Π
    ∇logψ.u_λ .= T(0.5) .* ∂logℒ_Π .* transpose(∑σ)
    ∇logψ.u_μ .= T(0.5)im .*  ∂logℒ_Π .* transpose(Δσ)

    return logψ
end


function Δ_logψ(W::NDM, c::NDMCache, lt::NDMLUT, σr, σc,
                     changes_r, changes_c)
    θλ_σ    = c.θλ_σ;   θλ_σ  .= lt.θλ_σ
    θμ_σ    = c.θμ_σ;   θμ_σ  .= lt.θμ_σ
    θλ_σp   = c.θλ_σp;  θλ_σp .= lt.θλ_σp
    θμ_σp   = c.θμ_σp;  θμ_σp .= lt.θμ_σp
    Π       = c._Π;     Π     .= lt._Π
    Π_tmp   = c._Π2;
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    T       = eltype(θλ_σ)

    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    logvaldiff = zero(eltype(Π))

    if !isempty(changes_r) || !isempty(changes_c)
        c.θλ_σ_tmp  .= logℒ.(θλ_σ)
        c.θμ_σ_tmp  .= logℒ.(θμ_σ)
        c.θλ_σp_tmp .= logℒ.(θλ_σp)
        c.θμ_σp_tmp .= logℒ.(θμ_σp)
        Π_tmp       .= logℒ.(Π)

        ∑logℒ_λ_σ  = sum(c.θλ_σ_tmp)
        ∑logℒ_μ_σ  = sum(c.θμ_σ_tmp)
        ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)
        ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)
        ∑Π_old     = sum(Π_tmp)

        Γ_λ_old = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp )
        Γ_μ_old = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp )
        logvaldiff -=  Γ_λ_old + T(1.0)im * Γ_μ_old + ∑Π_old

        for (i, nv) = changes_r
            θλ_σ .+= W.w_λ[:,i] .* (nv - σr[i])
            θμ_σ .+= W.w_μ[:,i] .* (nv - σr[i])
            Π    .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σr[i]) .+
                     T(0.5)im .* W.u_μ[:,i] .* (nv - σr[i])
            logvaldiff += T(0.5)   .* W.b_λ[i] * (nv - σr[i])
            logvaldiff += T(0.5)im .* W.b_μ[i] * (nv - σr[i])
        end

        for (i, nv) = changes_c
            θλ_σp .+= W.w_λ[:,i] .* (nv - σc[i])
            θμ_σp .+= W.w_μ[:,i] .* (nv - σc[i])
            Π     .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σc[i]) .-
                      T(0.5)im .* W.u_μ[:,i] .* (nv - σc[i])
            logvaldiff += T(0.5)   .* W.b_λ[i] * (nv - σc[i])
            logvaldiff -= T(0.5)im .* W.b_μ[i] * (nv - σc[i])
        end

        c.θλ_σ_tmp  .= logℒ.(θλ_σ)
        c.θμ_σ_tmp  .= logℒ.(θμ_σ)
        c.θλ_σp_tmp .= logℒ.(θλ_σp)
        c.θμ_σp_tmp .= logℒ.(θμ_σp)
        Π_tmp       .= logℒ.(Π)

        ∑logℒ_λ_σ  = sum(c.θλ_σ_tmp)
        ∑logℒ_μ_σ  = sum(c.θμ_σ_tmp)
        ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)
        ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)
        ∑Π         = sum(Π_tmp)

        Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp)
        Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp)
        logvaldiff +=  Γ_λ + T(1.0)im * Γ_μ + ∑Π
    end

    return logvaldiff
end

function Δ_logψ_and_∇logψ!(∇logψ, W::NDM, c::NDMCache, lt::NDMLUT,
                         σr_r, σc_r, changes_r, changes_c)
    θλ_σ    = c.θλ_σ;   θλ_σ  .= lt.θλ_σ
    θμ_σ    = c.θμ_σ;   θμ_σ  .= lt.θμ_σ
    θλ_σp   = c.θλ_σp;  θλ_σp .= lt.θλ_σp
    θμ_σp   = c.θμ_σp;  θμ_σp .= lt.θμ_σp
    Π       = c._Π;     Π     .= lt._Π
    Π_tmp   = c._Π2;
    ∑σ      = c.∑σ
    Δσ      = c.Δσ
    T       = eltype(θλ_σ)

    σr = c.σr; σr .= σr_r
    σc = c.σc; σc .= σc_r
    ∑σ .= σr .+ σc
    Δσ .= σr .- σc

    logvaldiff = zero(eltype(Π))

    if !isempty(changes_r) || !isempty(changes_c)
        c.θλ_σ_tmp  .= logℒ.(θλ_σ)
        c.θμ_σ_tmp  .= logℒ.(θμ_σ)
        c.θλ_σp_tmp .= logℒ.(θλ_σp)
        c.θμ_σp_tmp .= logℒ.(θμ_σp)
        Π_tmp       .= logℒ.(Π)

        ∑logℒ_λ_σ  = sum(c.θλ_σ_tmp)
        ∑logℒ_μ_σ  = sum(c.θμ_σ_tmp)
        ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)
        ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)
        ∑Π_old     = sum(Π_tmp)

        Γ_λ_old = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp )
        Γ_μ_old = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp )
        logvaldiff -=  Γ_λ_old + T(1.0)im * Γ_μ_old + ∑Π_old

        for (i, nv) = changes_r
            θλ_σ .+= W.w_λ[:,i] .* (nv - σr[i])
            θμ_σ .+= W.w_μ[:,i] .* (nv - σr[i])
            Π    .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σr[i]) .+
                     T(0.5)im .* W.u_μ[:,i] .* (nv - σr[i])
            logvaldiff += T(0.5)   .* W.b_λ[i] * (nv - σr[i])
            logvaldiff += T(0.5)im .* W.b_μ[i] * (nv - σr[i])
            ∑σ[i] += (nv - σr[i])
            Δσ[i] += (nv - σr[i])
            σr[i] = nv
        end

        for (i, nv) = changes_c
            θλ_σp .+= W.w_λ[:,i] .* (nv - σc[i])
            θμ_σp .+= W.w_μ[:,i] .* (nv - σc[i])
            Π     .+= T(0.5)   .* W.u_λ[:,i] .* (nv - σc[i]) .-
                      T(0.5)im .* W.u_μ[:,i] .* (nv - σc[i])
            logvaldiff += T(0.5)   .* W.b_λ[i] * (nv - σc[i])
            logvaldiff -= T(0.5)im .* W.b_μ[i] * (nv - σc[i])
            ∑σ[i] += (nv - σc[i])
            Δσ[i] -= (nv - σc[i])
            σc[i] = nv
        end

        c.θλ_σ_tmp  .= logℒ.(θλ_σ)
        c.θμ_σ_tmp  .= logℒ.(θμ_σ)
        c.θλ_σp_tmp .= logℒ.(θλ_σp)
        c.θμ_σp_tmp .= logℒ.(θμ_σp)
        Π_tmp       .= logℒ.(Π)

        ∑logℒ_λ_σ  = sum(c.θλ_σ_tmp)
        ∑logℒ_μ_σ  = sum(c.θμ_σ_tmp)
        ∑logℒ_λ_σp = sum(c.θλ_σp_tmp)
        ∑logℒ_μ_σp = sum(c.θμ_σp_tmp)
        ∑Π         = sum(Π_tmp)

        Γ_λ = T(0.5) * (∑logℒ_λ_σ + ∑logℒ_λ_σp)
        Γ_μ = T(0.5) * (∑logℒ_μ_σ - ∑logℒ_μ_σp)
        logvaldiff +=  Γ_λ + T(1.0)im * Γ_μ + ∑Π
    end

    # Compute additional terms for derivatives
    ∂logℒ_λ_σ  = c.∂logℒ_λ_σ;  ∂logℒ_λ_σ  .= ∂logℒ.(θλ_σ)
    ∂logℒ_μ_σ  = c.∂logℒ_μ_σ;  ∂logℒ_μ_σ  .= ∂logℒ.(θμ_σ)
    ∂logℒ_λ_σp = c.∂logℒ_λ_σp; ∂logℒ_λ_σp .= ∂logℒ.(θλ_σp)
    ∂logℒ_μ_σp = c.∂logℒ_μ_σp; ∂logℒ_μ_σp .= ∂logℒ.(θμ_σp)
    ∂logℒ_Π    = c.∂logℒ_Π;    ∂logℒ_Π    .= ∂logℒ.(Π)

    # Store the derivatives
    ∇logψ.b_λ .= T(0.5)   .* ∑σ
    ∇logψ.b_μ .= T(0.5)im .* Δσ

    ∇logψ.h_λ .= T(0.5)   .* (∂logℒ_λ_σ .+ ∂logℒ_λ_σp)
    ∇logψ.h_μ .= T(0.5)im .* (∂logℒ_μ_σ .- ∂logℒ_μ_σp)

    ∇logψ.w_λ .= T(0.5)   .* (∂logℒ_λ_σ.*transpose(σr) .+ ∂logℒ_λ_σp.*transpose(σc))
    ∇logψ.w_μ .= T(0.5)im .* (∂logℒ_μ_σ.*transpose(σr) .- ∂logℒ_μ_σp.*transpose(σc))

    ∇logψ.d_λ .= ∂logℒ_Π
    ∇logψ.u_λ .= T(0.5) .* ∂logℒ_Π .* transpose(∑σ)
    ∇logψ.u_μ .= T(0.5)im .*  ∂logℒ_Π .* transpose(Δσ)

    return logvaldiff
end
