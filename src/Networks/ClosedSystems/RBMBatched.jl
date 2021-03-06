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
batch_size(c::RBMBatchedCache) = size(c.θ, 2)

function Base.show(io::IO, m::RBMBatchedCache)
    print(io, "RBMBatchedCache with batch-size = $(batch_size(m))")
end


function logψ!(out::AbstractArray, net::RBM, c::RBMBatchedCache, σ_r::AStateBatch)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σ = c.σ;  σ.=σ_r #σ = copy!(c.σ, σ_r)

    #θ .= net.b .+ net.W * σ
    mul!(θ, net.W, σ)
    θ .+= net.b
    logℒθ .= net.f.(θ)

    #res = σᵗ*net.a + sum(logℒθ, dims=1)
    mul!(out, transpose(net.a), σ)
    Base.mapreducedim!(identity, +, out, logℒθ)

    return out
end

function logψ_and_∇logψ!(∇logψ, out, net::RBM, c::RBMBatchedCache, σ_r::AStateBatch)
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ = c.logℒθ
    ∂logℒθ = c.∂logℒθ
    res = out # c.res
    res_tmp = c.res_tmp
    T = eltype(θ)

    # copy the states to complex valued states for the computations.
    σ = c.σ;  σ.=σ_r #σ = copy!(c.σ, σ_r)

    #θ .= net.b .+ net.W * σ
    mul!(θ, net.W, σ)
    θ .+= net.b
    logℒθ  .= net.f.(θ)
    ∂logℒθ .= fwd_der.(net.f, θ)

    #res = σ'*net.a + sum(logℒθ, dims=1)
    mul!(out, transpose(net.a), σ)
    #sum!(logℒθ, res, init=false)
    Base.mapreducedim!(identity, +, res, logℒθ)

    ∇logψ.a   .= σ
    ∇logψ.b   .= ∂logℒθ
    #∇logψ.W  .= ∂logℒθ .* transpose(σ)

    _batched_outer_prod!(∇logψ.W, ∂logℒθ, σ)

    # TODO make this better
    #copyto!(out, 1, res, 1, length(out))

    return out
end
