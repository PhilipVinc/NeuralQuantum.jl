mutable struct SumBatchedCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(::typeof(sum_autobatch), arr_T, in_T, in_sz, batch_sz)  =
    SumBatchedCache(similar(arr_T, in_T, prod(in_sz), batch_sz),
              similar(arr_T, in_T, 1, batch_sz),
              similar(arr_T, in_T, 1, prod(in_sz), batch_sz),
              false)

batch_size(c::SumBatchedCache) = size(c.out, 2)

function sum_autobatch(c::SumBatchedCache, x::AbstractMatrix)
    σ = copyto!(c.σᵢₙ, x)

    sum!(c.out, c.σᵢₙ)

    return c.out
end

function backprop(∇, l::typeof(sum_autobatch), c::SumBatchedCache, δℒ::Number)
    c.δℒ .= δℒ

    return c.δℒ
end
