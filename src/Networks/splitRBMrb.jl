# Cached version
mutable struct rbRBMSplitCache{T} <: NNCache{rRBMSplit{T}}
    θ::T
    logℒθ::T
    valid::Bool # = false
end

cache(net::rRBMSplit{T}, bs::Int) where T =
    rbRBMSplitCache(zeros(T, length(net.b), bs),
                   zeros(T, length(net.b), bs),
                  false)

function (net::rRBMSplit)(c::rbRBMSplitCache, (σr,σc))
    θ = c.θ
    logℒθ = c.logℒθ

    θ .= net.b .+
            net.Wr*σr .+
                net.Wc*σc

    logℒθ .= logℒ.(θ)
    logψ = transpose(net.ar)*σr + transpose(net.ac)*σc + sum(logℒθ, dims=1)
    return logψ
end

function logψ_and_∇logψ!(∇logψ, net::rRBMSplit, c::rbRBMSplitCache, (σr,σc))
    θ = c.θ
    logℒθ = c.logℒθ
    T = eltype(θ)
    θ .= net.b .+
            net.Wr*σr .+
                net.Wc*σc

    logℒθ .= logℒ.(θ)
    logψ = transpose(net.ar)*σr + transpose(net.ac)*σc + sum(logℒθ, dims=1)

    ∇logψ.ar .= σr
    ∇logψ.ac .= σc
    ∇logψ.b  .= ∂logℒ.(θ)
    ∇logψ.Wr .= ∂logℒ.(θ) .* transpose(σr)
    ∇logψ.Wc .= ∂logℒ.(θ) .* transpose(σc)

    return logψ, ∇logψ
end
