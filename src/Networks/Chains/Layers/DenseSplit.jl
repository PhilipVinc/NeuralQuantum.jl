export DenseSplit

struct DenseSplit{VT,MT,C}
    Wr::MT
    Wc::MT
    b::VT
    σ::C
end
functor(d::DenseSplit) = (Wr=d.Wr, Wc=d.Wc, b=d.b), (Wr,Wc,b) -> DenseSplit(Wr,Wc,b,d.σ)
(l::DenseSplit)((σr, σc)) = l.σ.(l.Wr*σr .+ l.Wc*σc .+ l.b)

DenseSplit(in::Integer, args...;kwargs...) =
    DenseSplit(Complex{STD_REAL_PREC}, in, args...;kwargs...)
function DenseSplit(T::Type, in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = glorot_uniform)
  return DenseSplit(initW(T, out, in), initW(T, out, in), initb(T, out), σ)
end

function Base.show(io::IO, l::DenseSplit)
  print(io, "DenseSplit(", size(l.Wr, 2), ", ", size(l.Wr, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


struct DenseSplitCache{Ta,Tb,Tc,Td,Te}
    σr::Tc
    σc::Tc
    out::Tb

    δℒℒ::Td
    δℒr::Te
    δℒc::Te

    θ::Ta
    θ_tmp::Ta
    out2::Tb
    valid::Bool
end

cache(l::DenseSplit{Ta,Tb}, arr_T, in_T, in_sz) where {Ta,Tb} =
    DenseSplitCache(similar(l.Wr, size(l.Wr,2)),
               similar(l.Wr, size(l.Wr,2)),
               similar(l.b),

               similar(l.Wr, size(l.Wr,1)),
               similar(l.Wr, 1, size(l.Wr, 2)),
               similar(l.Wr, 1, size(l.Wr, 2)),

               similar(l.b),
               similar(l.b),
               similar(l.b),
               false)

function layer_out_type_size(l::DenseSplit, in_T ,in_sz)
    T1     = promote_type(in_T, eltype(l.Wr))
    out_T  = promote_type(T1, eltype(l.b))
    out_sz = size(l.b)
    return out_T, out_sz
end

function (l::DenseSplit)(c::DenseSplitCache, (xr, xc))
    # The preallocated caches
    out  = c.out
    θ = c.θ
    θ₂ = c.θ_tmp

    # Store the input to this layer for the backpropagation
    σr = copyto!(c.σr, xr)
    σc = copyto!(c.σc, xc)

    #θ .= net.b .+ net.W * x
    mul!(θ, l.Wr, σr)
    mul!(θ₂, l.Wc, σc)
    θ .+= l.b .+ θ₂

    # Apply the nonlinear function
    out  .= l.σ.(θ)
    return out
end

function backprop(∇, l::DenseSplit, c::DenseSplitCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= fwd_der.(l.σ, θ)

    ∇.Wr .= δℒℒ.*transpose(c.σr)
    ∇.Wc .= δℒℒ.*transpose(c.σc)
    ∇.b .= δℒℒ

    # TODO alloc of transpose
    δℒr = mul!(c.δℒr, transpose(δℒℒ), l.Wr)
    δℒc = mul!(c.δℒc, transpose(δℒℒ), l.Wc)

    return (δℒr, δℒc)
end
