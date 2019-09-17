mutable struct RBMSplitLUT{VT} <: NNLookUp
    θ::VT
end

lookup(net::RBMSplit) = RBMSplitLUT(net)
RBMSplitLUT(net::RBMSplit) = RBMSplitLUT(similar(net.b))

function set_lookup!(lt::RBMSplitLUT, net::RBMSplit, c::RBMSplitCache, σr_r, σc_r)
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    mul!(lt.θ,    net.Wr, σr)
    mul!(c.θ_tmp, net.Wc, σc)
    lt.θ .+= net.b + c.θ_tmp

    return lt
end

function update_lookup!(lt::RBMSplitLUT, net::RBMSplit, c::RBMSplitCache, σr_r, σc_r,
                         changes_r, changes_c)
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    θ = lt.θ

    for (i, nv) = changes_r
        θ .+= net.Wr[:,i] .* (nv - σr_r[i])
    end
    for (i, nv) = changes_c
        θ .+= net.Wc[:,i] .* (nv - σc_r[i])
    end
    return lt
end

function logψ(net::RBMSplit, c::RBMSplitCache, lt::RBMSplitLUT, σr_r, σc_r,
              changes_r, changes_c)
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)
    θ  = c.θ;
    logℒθ = c.logℒθ
    ∂logℒθ = c.∂logℒθ

    θ .= lt.θ
    for (i, nv) = changes_r
        θ .+= net.Wr[:,i] .* (nv - σr_r[i])
        σr[i] = nv
    end
    for (i, nv) = changes_c
        θ .+= net.Wc[:,i] .* (nv - σc_r[i])
        σc[i] = nv
    end

    logℒθ .= logℒ.(θ)
    logψ = dot(σr, net.ar) + dot(σc, net.ac) + sum(logℒθ)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitCache, lt::RBMSplitLUT,
                         σr_r, σc_r, changes_r, changes_c)
    # copy the states to complex valued states for the computations.
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)

    θ      = c.θ
    logℒθ  = c.logℒθ
    ∂logℒθ = c.∂logℒθ

    θ .= lt.θ
    for (i, nv) = changes_r
        θ .+= net.Wr[:,i] .* (nv - σr_r[i])
        σr[i] = nv
    end
    for (i, nv) = changes_c
        θ .+= net.Wc[:,i] .* (nv - σc_r[i])
        σc[i] = nv
    end

    logℒθ  .= logℒ.(θ)
    ∂logℒθ .= ∂logℒ.(θ)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒθ
    ∇logψ.Wr .= ∂logℒθ .* transpose(σr)
    ∇logψ.Wc .= ∂logℒθ .* transpose(σc)

    logψ = dot(σr,net.ar) + dot(σc,net.ac) + sum(logℒθ)
    return logψ
end

function Δ_logψ(net::RBMSplit, c::RBMSplitCache, lt::RBMSplitLUT, σr_r, σc_r,
                     changes_r, changes_c)
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)
    θ = c.θ
    logℒθ = c.logℒθ
    ∂logℒθ = c.∂logℒθ

    logvaldiff = zero(eltype(θ))

    if !isempty(changes_r) || !isempty(changes_c)
        θ .= lt.θ
        logℒθ .= logℒ.(θ)
        sumlogℒθ_old = sum(logℒθ)

        for (i, nv) = changes_r
            θ .+= net.Wr[:,i] .* (nv - σr_r[i])
            logvaldiff += net.ar[i] * (nv - σr_r[i])
        end
        for (i, nv) = changes_c
            θ .+= net.Wc[:,i] .* (nv - σc_r[i])
            logvaldiff += net.ac[i] * (nv - σc_r[i])
        end

        logℒθ .= logℒ.(θ)
        logvaldiff += sum(logℒθ) - sumlogℒθ_old
    end

    return logvaldiff
end

function Δ_logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitCache, lt::RBMSplitLUT,
                         σr_r, σc_r, changes_r, changes_c)
    σr = c.σr; copy!(σr, σr_r)
    σc = c.σc; copy!(σc, σc_r)
    θ = c.θ;
    logℒθ  = c.logℒθ
    ∂logℒθ = c.∂logℒθ

    θ .= lt.θ

    logvaldiff = zero(eltype(θ))
    if !isempty(changes_r) || !isempty(changes_c)
        logℒθ .= logℒ.(θ)
        sumlogℒθ_old = sum(logℒθ)

        for (i, nv) = changes_r
            θ .+= net.Wr[:,i] .* (nv - σr_r[i])
            logvaldiff += net.ar[i] * (nv - σr_r[i])
            σr[i] = nv
        end
        for (i, nv) = changes_c
            θ .+= net.Wc[:,i] .* (nv - σc_r[i])
            logvaldiff += net.ac[i] * (nv - σc_r[i])
            σc[i] = nv
        end

        logℒθ .= logℒ.(θ)
        logvaldiff += sum(logℒθ) - sumlogℒθ_old
    end

    ∂logℒθ .= ∂logℒ.(θ)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒθ
    ∇logψ.Wr .= ∂logℒθ .* transpose(σr)
    ∇logψ.Wc .= ∂logℒθ .* transpose(σc)

    return logvaldiff
end
