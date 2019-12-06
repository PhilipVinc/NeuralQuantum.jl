mutable struct SumCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(::typeof(sum_autobatch), arr_T, in_T, in_sz) =
    SumCache(similar(arr_T, in_T, in_sz...),
              zero(in_T),
              similar(arr_T, in_T, prod(in_sz)),
              false)

layer_out_type_size(::typeof(sum_autobatch), in_T ,in_sz) = in_T, (1,)

function sum_autobatch(c::SumCache, x::AbstractVector)
    σ = copyto!(c.σᵢₙ, x)

    c.out = sum(x)
    return c.out
end

function backprop(∇, l::typeof(sum_autobatch), c::SumCache, δℒ::Number)
    c.δℒ .= δℒ

    return c.δℒ
end
