export Dense, WSum

struct Dense{Ta,Tb}
    W::Ta
    b::Tb
end
@functor Dense
(l::Dense)(x) = logℒ.(l.W*x .+ l.b)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = glorot_uniform)
  return Dense(initW(out, in), initb(out))
end

struct DenseCache{Ta,Tb,Tc,Td}
    σ::Tc
    out::Tb
    δℒℒ::Td

    θ::Ta
    out2::Tb
    valid::Bool
end

function cache(l::Dense{Ta,Tb}, in_T ,in_sz) where {Ta,Tb}
    c = DenseCache(similar(l.W, size(l.W,2)),
               similar(l.b),
               similar(l.W, size(l.W,1)),
               similar(l.b),
               similar(l.b),
               false)
    return c
end

function layer_out_type_size(l::Dense, in_T ,in_sz)
    T1     = promote_type(in_T, eltype(l.W))
    out_T  = promote_type(T1, eltype(l.b))
    out_sz = size(l.b)
    return out_T, out_sz
end

function (l::Dense)(c::DenseCache, x)
    # The preallocated caches
    logℒθ  = c.out
    θ = c.θ

    # Store the input to this layer for the backpropagation
    σ = copyto!(c.σ, x)

    #θ .= net.b .+ net.W * x
    mul!(θ, l.W, σ)
    θ .+= l.b

    # Apply the nonlinear function
    logℒθ  .= logℒ.(θ)
    return logℒθ
end

function backprop(∇, l::Dense, c::DenseCache, δℒ)
    # The preallocated caches
    θ = c.θ
    δℒℒ = c.δℒℒ

    # Compute the actual sensitivity
    copyto!(δℒℒ, δℒ)
    δℒℒ .*= ∂logℒ.(θ)

    ∇.W .= δℒℒ.*transpose(c.σ)
    ∇.b .= δℒℒ

    return transpose(δℒℒ)*l.W
end

struct WSum{T}
    c::T
end
@functor WSum
(l::WSum)(x) = sum(x.*l.c)

function WSum(in::Integer, σ = identity;
              initb = glorot_uniform)
  return WSum(initb(in))
end


mutable struct WSumCache{Ta,Tb,Tc}
    σᵢₙ::Ta
    out::Tc
    δℒ::Tb
    valid::Bool
end

cache(l::WSum, in_T, in_sz)  =
    WSumCache(similar(l.c, Complex{real(eltype(l.c))}),
              zero(Complex{real(eltype(l.c))}),
              similar(l.c, Complex{real(eltype(l.c))}, 1, length(l.c)),
              false)

function layer_out_type_size(l::WSum, in_T ,in_sz)
    out_T     = Complex{real(eltype(l.c))}
    return out_T, (1,)
end


function (l::WSum)(c::WSumCache, x)
    σ = copyto!(c.σᵢₙ, x)

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
