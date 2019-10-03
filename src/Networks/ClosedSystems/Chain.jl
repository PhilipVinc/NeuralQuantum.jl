export Chain

struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

functor(c::Chain) = c.layers, ls -> Chain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(Base.tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)


function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

struct ChainCache{T<:Tuple} <: NNCache{Chain}
    caches::T
    valid::Bool
end

cache(l::Chain) = ChainCache(cache.(l.layers), false)

# Applychain with caches
applychain(::Tuple{}, ::Tuple{}, x) = x
applychain(fs::Tuple, fc::Tuple, x) = applychain(Base.tail(fs),
                                                 Base.tail(fc),
                                                 first(fs)(first(fc), x))
(c::Chain)(ch::ChainCache, x) = applychain(c.layers, ch.caches, x)

backpropchain(∇, ::Tuple{}, ::Tuple{}, δℒ) = δℒ
function backpropchain(∇, fs::Tuple, fc::Tuple, δℒ)
    a = reverse(Base.tail(reverse(∇)))
    b = reverse(Base.tail(reverse(fs)))
    c = reverse(Base.tail(reverse(fc)))

    backpropchain(a,b,c,
        backprop(last(∇), last(fs), last(fc), δℒ))
end

function logψ_and_∇logψ!(∇lnψ, net::Chain, c::ChainCache, σ)
    # forward pass
    lnψ = net(c, σ)

    # backward
    backpropchain(fields(∇lnψ), net.layers, c.caches, 1.0)
    return lnψ, ∇lnψ
end

is_analytic(net::Chain) = true
out_type(net::Chain) = ComplexF64
