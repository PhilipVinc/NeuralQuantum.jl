mutable struct WSumBatchedCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(l::WSum, arr_T, in_T, in_sz, batch_sz)  =
    SumBatchedCache(similar(l.c, Complex{real(eltype(l.c))}, length(l.c),batch_sz),
              similar(l.c, Complex{real(eltype(l.c))}, 1, batch_sz),
              similar(l.c, Complex{real(eltype(l.c))}, 1, length(l.c), batch_sz),
              false)

batch_size(c::WSumBatchedCache) = size(c.out, 2)

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
