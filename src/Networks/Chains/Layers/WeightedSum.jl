struct WSum{T}
    c::T
end
@functor WSum
(l::WSum)(x::AbstractVector) = sum(x.*l.c)
(l::WSum)(x::AbstractArray) = sum_autobatch(x.*l.c)

WSum(in::Integer) = WSum(Complex{STD_REAL_PREC}, in)
function WSum(T::Type, in::Integer;
              initb = glorot_uniform)
  return WSum(initb(T, in))
end


mutable struct WSumCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(l::WSum, arr_T, in_T, in_sz)  =
    WSumCache(similar(l.c, Complex{real(eltype(l.c))}),
              zero(Complex{real(eltype(l.c))}),
              similar(l.c, Complex{real(eltype(l.c))}, 1, length(l.c)),
              false)

function layer_out_type_size(l::WSum, in_T ,in_sz)
    out_T     = promote_type(in_T, eltype(l.c))
    return out_T, (1,)
end


function (l::WSum)(c::WSumCache, x)
    σ = copyto!(c.σᵢₙ, x)

    # TODO the dot product allocates
    c.out = sum(x.*l.c)
    return c.out
end

function backprop(∇, l::WSum, c::WSumCache, δℒ::Number)
    # compute the derivative
    ∇.c .= c.σᵢₙ

    # Backpropagate
    c.δℒ .= δℒ .* transpose(l.c)
    return c.δℒ
end
