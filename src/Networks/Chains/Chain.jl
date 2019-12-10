export Chain

"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

```julia
m = Chain(x -> x^2, x -> x+1)
m(5) == 26
m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))
```

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.

# Note
Chain is simply a Neural Network, but it is not a quantum state. To use it with
NeuralQuantum you should wrap it into a `PureStateAnsatz`.
"""
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

cache(l::Chain, arr_T, in_T, in_sz) = begin
    caches = []
    for layer = l.layers
        c = cache(layer, arr_T, in_T, in_sz)
        in_T, in_sz = layer_out_type_size(layer, in_T, in_sz)
        push!(caches, c)
    end
    ChainCache(Tuple(caches), false)
end

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
    backpropchain(∇lnψ, net.layers, c.caches, 1.0)
    return lnψ
end

is_analytic(net::Chain) = true
out_type(net::Chain) = ComplexF64
