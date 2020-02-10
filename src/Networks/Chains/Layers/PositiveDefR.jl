export PositiveDefR

struct PositiveDefR{VT,MT,C}
    Wr::MT
    Wi::MT
    b::VT
    σ::C
end
functor(d::PositiveDefR) = (Wr=d.Wr, Wi=d.Wi, b=d.b), t -> PositiveDefR(t.Wr, t.Wi, t.b, d.σ)

# NOTE This is based upon the fact that σr/c are real so that conj(W*σc) == conj(W)*σc
(l::PositiveDefR)((σr, σc)) = l.σ.(       l.Wr * (σr .+ σc) .+
                                    im .* l.Wi * (σr .- σc) .+ l.b)

PositiveDefR(in::Integer, args...;kwargs...) =
    PositiveDefR(STD_REAL_PREC, in, args...;kwargs...)
function PositiveDefR(T::Type{<:Real}, in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = glorot_uniform)
  return PositiveDefR(initW(T, out, in), initW(T, out, in),
                      initb(T, out), σ)
end

function Base.show(io::IO, l::PositiveDefR)
  print(io, "PositiveDefR{$(eltype(l.Wr))}(", size(l.Wr, 2), ", ", size(l.Wr, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


struct PositiveDefRCache{Ta,Tb,Tc,Td,Te}
    σr::Tc
    σc::Tc
    ∑σ::Tc
    Δσ::Tc

    out::Tb

    δℒℒ::Td
    δℒr::Te
    δℒc::Te

    θ::Tb
    θr::Ta
    θr2::Ta
    valid::Bool
end

cache(l::PositiveDefR{Ta,Tb}, arr_T, in_T, in_sz) where {Ta,Tb} =
    PositiveDefRCache(
               similar(l.Wr, size(l.Wr,2)),
               similar(l.Wr, size(l.Wr,2)),
               similar(l.Wr, size(l.Wr,2)),
               similar(l.Wr, size(l.Wr,2)),

               similar(l.b, Complex{eltype(l.b)}),

               similar(l.Wr, Complex{eltype(l.b)}, size(l.Wr,1)),
               similar(l.Wr, Complex{eltype(l.b)}, 1, size(l.Wr, 2)),
               similar(l.Wr, Complex{eltype(l.b)}, 1, size(l.Wr, 2)),

               similar(l.b, Complex{eltype(l.b)}),
               similar(l.b),
               similar(l.b),
               false)

function layer_out_type_size(l::PositiveDefR, in_T ,in_sz)
    T1     = promote_type(in_T, Complex{eltype(l.Wr)})
    out_T  = promote_type(T1, Complex{eltype(l.b)})
    out_sz = size(l.b)
    return out_T, out_sz
end

function (l::PositiveDefR)(c::PositiveDefRCache, (xr, xc))
    # The preallocated caches
    out  = c.out
    θ    = c.θ
    θᵣ₁  = c.θr
    θᵣ₂  = c.θr2

    # Store the input to this layer for the backpropagation
    σr = copyto!(c.σr, xr)
    σc = copyto!(c.σc, xc)

    c.∑σ .= σr .+ σc
    c.Δσ .= σr .- σc

    #θ .= net.b .+ net.W * x
    mul!(θᵣ₁, l.Wr, c.∑σ)
    mul!(θᵣ₂, l.Wi, c.Δσ)
    # NOTE This is based upon the fact that σr/c are real so that conj(W*σc) == conj(W)*σc
    θ .= θᵣ₁ .+ im .* θᵣ₂ .+ l.b

    # Apply the nonlinear function
    out  .= l.σ.(θ)
    return out
end

function backprop(∇, l::PositiveDefR, c::PositiveDefRCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= fwd_der.(l.σ, θ)

    ∇.Wr .= δℒℒ .*       transpose(c.∑σ)
    ∇.Wi .= δℒℒ .* im .* transpose(c.Δσ)
    ∇.b .= δℒℒ

    # TODO alloc of transpose
    δℒr = 0 #mul!(c.δℒr, transpose(δℒℒ), l.Wr)
    δℒc = 0 #mul!(c.δℒc, transpose(δℒℒ), l.Wc)
    #c.δℒc.*= im

    return (δℒr, δℒc)
end
