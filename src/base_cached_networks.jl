export cached, vectorize_gradient, weights, grad_cache

abstract type NNLookUp
    #valid
end

"""
    NNCache{N}

The base abstract type holding a cache for the network type `N<:NeuralNetwork`.
"""
abstract type NNCache{N}
    #valid::Bool
end

"""
    grad_cache(net::N) -> tuple

Creates a tuple holding all derivatives of the network N.
"""
grad_cache(net::NeuralNetwork) = derivative_tuple(net)

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
struct CachedNet{N, NC}#=, ND, NDV}=# <: NeuralNetwork
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

# When copying shallow copy the net but deepcopy the der_vec
copy(cnet::CachedNet) = CachedNet(cnet.net, deepcopy(cnet.cache))

"""
    cache(net)

Constructs the `NNCache{typeof(net)}` object that holds the cache for this network.
If it has not been implemented returns nothing.
"""
cache(net) = nothing

"""
    lookup(net)

Constructs the `NNCache{typeof(net)}` object that holds the cache for this network.
If it has not been implemented returns nothing.
"""
lookup(net) = nothing

weights(net) = net
weights(cnet::CachedNet) = cnet.net
grad_cache(net::CachedNet)     = grad_cache(net.net)

@inline (cnet::CachedNet)(σ...) = logψ(cnet, σ...)
@inline logψ(cnet::CachedNet, σ...) = cnet.net(cnet.cache, config(σ)...)

#@inline ∇logψ(n::CachedNet, σ) = logψ_and_∇logψ(n, σ)[2]
function logψ_and_∇logψ(n::CachedNet, σ)
    @warn "Inefficient calling logψ_and_∇logψ for cachedNet"
    ∇lnψ = grad_cache(n)
    lψ, ∇ψ = logψ_and_∇logψ!(∇lnψ, n, σ);
    return (lψ, ∇ψ)
end

function logψ_and_∇logψ!(der, n::CachedNet, σ)
    lψ, ∇ψ = logψ_and_∇logψ!(der, n.net, n.cache, config(σ)...);
    return (lψ, der)
end

## Optimisation of cachednet
update!(opt, cnet::CachedNet, Δ, state=nothing) = (update!(opt, weights(cnet), weights(Δ), state); invalidate_cache!(cnet.cache))
apply!(opt, val1::Union{NeuralNetwork, CachedNet}, val2::Union{NeuralNetwork, CachedNet}, args...) =
    apply!(weights(val1), weights(val2), args...)

# Common Operations on networks that should always be defined
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
    lψ, ∇ψ = logψ_and_∇logψ(n.der, n.cache, n.net, config(σ));
    return (lψ, n.der)
end
