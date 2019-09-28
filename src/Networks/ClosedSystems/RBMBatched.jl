mutable struct RBMBatchedCache{VT,VS,VST} <: NNBatchedCache{RBMSplit}
    θ::VT
    θ_tmp::VT
    logℒθ::VT
    ∂logℒθ::VT

    # complex sigmas
    res::VS #batch
    res_tmp::VST #batch

    # states
    σ::VT

    valid::Bool # = false
end

cache(net::RBM, batch_sz) = begin
    n_v = length(net.a)
    n_h = length(net.b)
    return RBMBatchedCache(
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, n_h, batch_sz),
                  similar(net.b, 1, batch_sz),
                  similar(net.b, 1, batch_sz),
                  similar(net.b, n_v, batch_sz),
                  false)
end

(net::RBM)(c::RBMBatchedCache, σ::State) = net(c, config(σ))
function (net::RBM)(c::RBMBatchedCache, σ::AbstractArray)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    res = c.res
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    copyto!(c.σ, σ)

    #θ .= net.b .+ net.W * σ
    mul!(θ, net.W, σ)
    θ .+= net.b
    logℒθ .= logℒ.(θ)

    #res = σ'*net.a + sum(logℒθ, dims=1)
    mul!(res, net.a', σ)
    conj!(res)
    Base.mapreducedim!(identity, +, res, logℒθ)

    return res
end

function logψ_and_∇logψ!(∇logψ, net::RBM, c::RBMBatchedCache, σ)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    ∂logℒθ = c.∂logℒθ
    res = c.res
    res_tmp = c.res_tmp
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    copyto!(c.σ, σ)

    #θ .= net.b .+ net.W * σ
    mul!(θ, net.W, σ)
    θ .+= net.b
    logℒθ  .= logℒ.(θ)
    ∂logℒθ .= ∂logℒ.(θ)

    #res = σ'*net.a + sum(logℒθ, dims=1)
    mul!(res, net.a', σ)
    conj!(res)
    Base.mapreducedim!(identity, +, res, logℒθ)

    ∇logψ.a   .= σ
    ∇logψ.b   .= ∂logℒθ
    #∇logψ.W  .= ∂logℒθ .* transpose(σ)

    _batched_outer_prod!(∇logψ.W, ∂logℒθ, σ)

    return res
end
