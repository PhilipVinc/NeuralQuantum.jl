struct DenseSplit{VT,MT}
    Wr::MT
    Wc::MT
    b::VT
end
@functor DenseSplit
(l::DenseSplit)((σr, σc)) = logℒ.(l.Wr*σr .+ l.Wc*σc .+ l.b)

function DenseSplit(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = glorot_uniform)
  return Dense(initW(out, in), initW(out, in), initb(out))
end

struct DenseSplitCache{Ta,Tb,Tc,Td}
    σr::Tc
    σc::Tc
    out::Tb
    δℒℒ::Td

    θ::Ta
    θ_tmp::Ta
    out2::Tb
    valid::Bool
end

cache(l::DenseSplit{Ta,Tb}) where {Ta,Tb} =
    DenseSplitCache(similar(l.Wr, size(l.W,2)),
               similar(l.Wr, size(l.W,2)),
               similar(l.b),
               similar(l.Wr, size(l.W,1)),
               similar(l.b),
               similar(l.b),
               similar(l.b),
               false)

function (l::DenseSplit)(c::DenseSplitCache, (xr, xc))
    # The preallocated caches
    θ = c.θ
    θ_tmp = c.θ_tmp
    logℒθ  = c.out

    # Store the input to this layer for the backpropagation
    σr = copyto!(c.σr, xr)
    σc = copyto!(c.σc, xc)

    #θ .= net.b .+ net.W * x
    mul!(θ, net.Wr, σr)
    mul!(θ_tmp, net.Wc, σc)
    θ .+= net.b .+ θ_tmp

    # Apply the nonlinear function
    logℒθ  .= logℒ.(θ)
    return logℒθ
end

function backprop(∇, l::DenseSplit, c::DenseSplitCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= ∂logℒ.(θ)

    ∇.Wr .= δℒℒ.*transpose(c.σr)
    ∇.Wc .= δℒℒ.*transpose(c.σc)
    ∇.b .= δℒℒ

    return (transpose(δℒℒ)*l.Wr, transpose(δℒℒ)*l.Wc)
end
