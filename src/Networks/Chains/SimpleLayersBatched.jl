struct DenseBatchedCache{Ta,Tb,Tc,Td}
    σ::Tc
    out::Tb
    δℒℒ::Td

    θ::Ta
    out2::Tb
    valid::Bool
end

function cache(l::Dense{Ta,Tb}, in_T ,in_sz, batch_sz) where {Ta,Tb}
    c = DenseBatchedCache(similar(l.W, size(l.W,2), batch_sz),
                          similar(l.b, size(l.b,1), batch_sz),
                          similar(l.W, size(l.W,1), batch_sz),
                          similar(l.b, size(l.b,1), batch_sz),
                          similar(l.b, size(l.b,1), batch_sz),
                          false)
    return c
end

batch_size(c::DenseBatchedCache) = size(c.out, 2)

function (l::Dense)(c::DenseBatchedCache, x)
    # The preallocated caches
    out  = c.out
    θ = c.θ

    # Store the input to this layer for the backpropagation
    σ = copyto!(c.σ, x)

    #θ .= net.b .+ net.W * x
    mul!(θ, l.W, σ)
    θ .+= l.b

    # Apply the nonlinear function
    out  .= l.σ.(θ)
    return out
end

function backprop(∇, l::Dense, c::DenseBatchedCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= fwd_der.(l.σ, θ)

    _batched_outer_prod_noconj!(∇.W, δℒℒ, c.σ)
    ∇.b .= δℒℒ

    return transpose(transpose(δℒℒ)*l.W)
end


mutable struct WSumBatchedCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(l::WSum, in_T, in_sz, batch_sz)  =
    WSumBatchedCache(similar(l.c, Complex{real(eltype(l.c))}, length(l.c),batch_sz),
              similar(l.c, Complex{real(eltype(l.c))}, 1, batch_sz),
              similar(l.c, Complex{real(eltype(l.c))}, 1, length(l.c), batch_sz),
              false)

batch_size(c::DenseSplitCache) = size(c.out, 2)

function (l::WSum)(c::WSumBatchedCache, x)
    # dot product in temp cache
    c.σᵢₙ .= x .* l.c
    c.out .= sum_autobatch(c.σᵢₙ)

    # Store the input for backpropagation
    σ = copyto!(c.σᵢₙ, x)

    return c.out
end

function backprop(∇, l::WSum, c::WSumBatchedCache, δℒ::Number)
    # compute the derivative
    ∇.c .= c.σᵢₙ

    # Backpropagate
    c.δℒ .= δℒ .* transpose(l.c)
    return c.δℒ
end
