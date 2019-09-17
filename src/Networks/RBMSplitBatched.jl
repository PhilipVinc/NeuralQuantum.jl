mutable struct RBMSplitBatchedCache{VT,VS,VST} <: NNBatchedCache{RBMSplit}
    θ::VT
    θ_tmp::VT
    logℒθ::VT
    ∂logℒθ::VT

    # complex sigmas
    res::VS #batch
    res_tmp::VST #batch

    # states
    σr::VT
    σc::VT

    valid::Bool # = false
end

cache(net::RBMSplit, batch_sz) = begin
    n_h = length(net.b)
    n_v = length(net.ar)
    return RBMSplitBatchedCache(
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, 1, batch_sz),
                  similar(net.b, 1, batch_sz),
                  similar(net.b, n_v, batch_sz),
                  similar(net.b, n_v, batch_sz),
                  false)
end

function (net::RBMSplit)(c::RBMSplitBatchedCache, σr_r, σc_r)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    res = c.res
    res_tmp = c.res_tmp
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copyto!(σr, σr_r)
    σc = c.σc; copyto!(σc, σc_r)

    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp
    logℒθ .= NeuralQuantum.logℒ.(θ)

    #res = σr'*net.ar + σc'*net.ac # + sum(logℒθ, dims=1)
    mul!(res_tmp, net.ar', σr)
    mul!(res, net.ac', σc)
    res .+= res_tmp
    Base.mapreducedim!(identity, +, res, logℒθ)

    return res
end

function logψ_and_∇logψ!(∇logψ, net::RBMSplit, c::RBMSplitBatchedCache, σr_r, σc_r)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    ∂logℒθ = c.∂logℒθ
    res = c.res
    res_tmp = c.res_tmp
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σr = c.σr; copyto!(σr, σr_r)
    σc = c.σc; copyto!(σc, σc_r)

    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp
    logℒθ  .= logℒ.(θ)
    ∂logℒθ .= ∂logℒ.(θ)

    #res = σr'*net.ar + σc'*net.ac # + sum(logℒθ, dims=1)
    mul!(res_tmp, net.ar', σr)
    mul!(res, net.ac', σc)
    res .+= res_tmp

    Base.mapreducedim!(identity, +, res, logℒθ)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒθ
    #∇logψ.Wr .= ∂logℒθ .* transpose(σr)
    #∇logψ.Wc .= ∂logℒθ .* transpose(σc)

    _batched_outer_prod!(∇logψ.Wr, ∂logℒθ, σr)
    _batched_outer_prod!(∇logψ.Wc, ∂logℒθ, σc)

    return res
end
