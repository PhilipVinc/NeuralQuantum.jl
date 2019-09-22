# This file contains methods that are dispatched to when using neural networks
# with a preallocated cache (called CachedNetworks). Those are used to greatly
# improve performance, especially in

export cached, vectorize_gradient, weights, grad_cache

"""
    NNCache{N}

The base abstract type holding a cache for the network type `N<:NeuralNetwork`.
"""
abstract type NNCache{N}
    #valid::Bool
end

"""
    invalidate_cache!(net)

Invalidates the cache of a cached Neural Network. Usually called when weights
are changed to signal the fact that lookup tables must be cleared.
"""
@inline invalidate_cache!(net::NNCache) = net.valid = false
@inline invalidate_cache!(net::Nothing) = nothing # if not implemented

"""
    CachedNet{N, NC}

A Cached Version of Neural Network `N` using Cache `NC`. It behaves the same as
the network `N`, but uses a preallocated cache to speed up computation of the
gradient and the function.
"""
struct CachedNet{N, NC} <: NeuralNetwork
    net::N
    cache::NC
end

"""
    cached(net)

Constructs a cached version of the neural network `net`, preallocating several
arrays for intermediate results for improved performance. This object will
behave identically to a standard network `net`.
"""
cached(net::NeuralNetwork) = CachedNet(net, cache(net))
cached(net::CachedNet) = throw("Can't build cached net of cached network!")

# When copying shallow copy the net but deepcopy the der_vec
"""
    copy(cnet::CachedNet)

Copy a cached network, building a shallow copy of the network and a deep-copy
of the cache.
"""
copy(cnet::CachedNet) = CachedNet(cnet.net, deepcopy(cnet.cache))

"""
    cache(net)

Constructs the `NNCache{typeof(net)}` object that holds the cache for this network.
If it has not been implemented returns nothing.
"""
cache(net) = nothing

"""
    weights(net)

Access the underlying structure holding the weights of the network. This is usefull
when dealing with Symmetrized networks or cached networks, as it gives access to
the wrapped data structure.
"""
weights(net) = net
weights(cnet::CachedNet) = cnet.net
grad_cache(net::CachedNet) = grad_cache(net.net)

@inline (cnet::CachedNet)(σ...) = logψ(cnet, σ...)
# When you call logψ on a cached net use the cache to compute the net
@inline logψ(cnet::CachedNet, σ...) = cnet.net(cnet.cache, config(σ)...)

#@inline ∇logψ(n::CachedNet, σ) = logψ_and_∇logψ(n, σ)[2]
function logψ_and_∇logψ(n::CachedNet, σ::Vararg{N,V}) where {N,V}
    #@warn "Inefficient calling logψ_and_∇logψ for cachedNet"
    ∇lnψ   = grad_cache(n)
    lψ, _g = logψ_and_∇logψ!(∇lnψ, n, σ...);
    return (lψ, ∇lnψ)
end

# Declare the two functions, even if config(blabla)=blabla, because of a shitty
# Julia's performance bug #32761
# see https://github.com/JuliaLang/julia/issues/32761
@inline logψ_and_∇logψ!(der, n::CachedNet, σ::State) = logψ_and_∇logψ!(der, n, config(σ))
@inline function logψ_and_∇logψ!(der, n::CachedNet, σ::NTuple{N,AbstractArray}) where N
    lψ = logψ_and_∇logψ!(der, n.net, n.cache, config(σ)...)
    return (lψ, der)
end
@inline function logψ_and_∇logψ!(der, n::CachedNet, σ::Vararg{AbstractArray,N}) where N
    lψ = logψ_and_∇logψ!(der, n.net, n.cache, σ...);
    return (lψ, der)
end

## Optimisation of cachednet
update!(opt, cnet::CachedNet, Δ, state=nothing) = (update!(opt, weights(cnet), weights(Δ), state); invalidate_cache!(cnet.cache))
apply!(opt, val1::Union{NeuralNetwork, CachedNet}, val2::Union{NeuralNetwork, CachedNet}, args...) =
    apply!(weights(val1), weights(val2), args...)

# The common operations are forwarded to the underlying network.
input_type(cnet::CachedNet) = input_type(cnet.net)
out_type(cnet::CachedNet) = out_type(cnet.net)
input_shape(net::CachedNet) = input_shape(cnet.net)
random_input_state(cnet::CachedNet) = random_input_state(cnet.net)
is_analytic(cnet::CachedNet) = is_analytic(cnet.net)
num_params(cnet::CachedNet) = num_params(cnet.net)

# Visualization
function Base.show(io::IO, m::CachedNet)
    print(io, "CachedNet{$(m.net)}")
end

Base.show(io::IO, ::MIME"text/plain", m::CachedNet) = print(io, "CachedNet{$(m.net)}")

## Particoular kinds of networks
# Thing for Matrix Neural Network
# A MatrixNeuralNetwork is a network for a state that has a bi-partite state,
# representing row and column of a density matrix in the same space.
abstract type MatrixNeuralNetwork <: NeuralNetwork end
const MatrixNet   = Union{MatrixNeuralNetwork, CachedNet{<:MatrixNeuralNetwork}}
#random_input_state(net) = sample_input_type(net).([rand(0:1) for i=1:input_length(model)])

# Thing for Pure Neural Network
# A MatrixNeuralNetwork is a network for a state that has a bi-partite state,
# representing row and column of a density matrix in the same space.
abstract type PureNeuralNetwork <: NeuralNetwork end
const PureNet   = Union{PureNeuralNetwork, CachedNet{<:PureNeuralNetwork}}
#random_input_state(net) = sample_input_type(net).([rand(0:1) for i=1:input_length(model)])

# Thing for Pure-state (Closed system) Neural Network
# A MatrixNeuralNetwork is a network for a state that has a bi-partite state,
# representing row and column of a density matrix in the same space.
abstract type KetNeuralNetwork <: NeuralNetwork end
const KetNet   = Union{KetNeuralNetwork, CachedNet{<:KetNeuralNetwork}}
#random_input_state(net) = sample_input_type(net).([rand(0:1) for i=1:input_length(model)])

# This overrides the standard behaviour of net(σ...) because Vector unpacking
# should not happen
@inline logψ(cnet::CachedNet{<:KetNeuralNetwork}, σ) = cnet.net(cnet.cache, config(σ))
function logψ_and_∇logψ(net::KetNeuralNetwork, σ)
    σ = config(σ)
    y, back = forward(net -> net(σ), net)

    # This computes the gradient, which is the conjugate of the derivative
    der = back(Int8(1))[1]

    # We take the conjugate of every element
    for key=keys(der)
        conj!(der[key])
    end
    der = add_vector_field(net, der)

    return y, der
end

function logψ_and_∇logψ(n::CachedNet{<:KetNeuralNetwork}, σ)
    ∇lnψ = grad_cache(n)
    lnψ = logψ_and_∇logψ!(∇lnψ, n.net, n.cache, config(σ));
    return (lnψ, ∇lnψ)
end
