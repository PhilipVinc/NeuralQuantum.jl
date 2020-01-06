struct DenseBatchedCache{Ta,Tb,Tc,Td,Te}
    σ::Tc
    out::Tb
    δℒℒ::Td
    δℒ::Te

    θ::Ta
    out2::Tb
    valid::Bool
end

function cache(l::Dense{Ta,Tb}, arr_T, in_T ,in_sz, batch_sz) where {Ta,Tb}
    in_T2 = promote_type(in_T, eltype(l.W))

    c = DenseBatchedCache(similar(l.W, in_T2, size(l.W,2), batch_sz),
                          similar(l.b, in_T2, size(l.b,1), batch_sz),
                          similar(l.W, in_T2, size(l.W,1), batch_sz),
                          similar(l.W, in_T2, size(l.W,2), batch_sz),

                          similar(l.b, in_T2, size(l.b,1), batch_sz),
                          similar(l.b, in_T2, size(l.b,1), batch_sz),
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

    δℒ = mul!(c.δℒ, transpose(l.W), δℒℒ)
    return δℒ
end
