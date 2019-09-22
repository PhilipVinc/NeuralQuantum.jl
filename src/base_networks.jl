# This file contains some aliases and generic methods to use neural networks.

export logψ, ∇logψ, ∇logψ!, logψ_and_∇logψ, logψ_and_∇logψ!, grad_cache

"""
    grad_cache(net::N) -> tuple

Creates a tuple holding all derivatives of the network N.
"""
grad_cache(net::NeuralNetwork) = begin
    is_analytic(net) && return RealDerivative(net)
    return WirtingerDerivative(net)
end

# Define various aliases used a bit everywhere in networks and when testing
"""
    logψ(net, σ) -> Number

Evaluates the neural network `net` for the given input `σ`.
This will return the value ⟨σ|ψ⟩ if the network represents a pure state (a ket)
or ⟨σᵣ|ρ̂|σᵪ⟩, where σ=(σᵣ,σᵪ), if the network encodes a mixed state.

This is equivalent to `net(σ)`.

The numerical type of the output is `out_type(net)`.

If `net isa CachedNet` then the computation will be performed efficiently
with minimal allocations.
"""
@inline logψ(net::NeuralNetwork, σ) = net(σ)
@inline log_prob_ψ(net, σ...)       = 2.0*real(net(σ...))
@inline ∇logψ(args...)              = logψ_and_∇logψ(args...)[2]
@inline ∇logψ!(args...)             = logψ_and_∇logψ!(args...)[2]

"""
    logψ_and_∇logψ(net, σ)

Evaluates the neural network `net` for the given input `σ` and returns both the
value and it's gradient w.r.t. to the nework's parameters.

!!  The gradient will be allocated. Don't use this in a hot loop and prefer the
    in-place version `logψ_and_∇logψ!`. The gradient is always allocated by calling
    `grad_cache(net)`.

!!  If `net isa CachedNet` and the gradient has been hand-coded, the hand-coded
    version will be used, otherwise `Zygote` will be used to generate the
    gradient with AD. This case is much slower and incurs a long precompilation
    time.

See also: `logψ`, `logψ_and_∇logψ!`

"""
function logψ_and_∇logψ(net::NeuralNetwork, σ::Vararg{N,V})where {N,V}
    der = grad_cache(net)
    y, der = logψ_and_∇logψ!(der, net, σ)
    return y, der
end

# Autodiff version of the gradient: if you are hitting this, expect
# bad performance. Maybe you should hand-code the gradient of your model?
function logψ_and_∇logψ!(der, net::NeuralNetwork, σ)
    σ = config(σ)
    # Zygote's autodiff: generate the pullback
    y, back = forward(net -> net(σ...), net)

    # This computes the gradient, which is the conjugate of the derivative
    _der = back(Int8(1))[1]

    for key=keys(_der)
        conj!(_der[key])
        copyto!(der[key], _der[key])
    end
    return y, der
end

# Common Operations on networks that should always be defined
# If you hit this section it probably means that you forgot to define one
# of those functions on the type of your neural network.
"""
    input_type(net) -> Type

Returns the numerical `eltype` of the input for efficiently evaluating in a
type-stable way the network.
"""
input_type(net::NeuralNetwork) = error("Not Implemented")

"""
    out_type(net) -> Type

Returns the numerical `eltype` of the output of the network.
"""
out_type(net::NeuralNetwork) = error("Not Implemented")
"""
    input_shape(net)

This returns the shape of the numerical input that should be provided to the
network, inside a tuple. This is basically the shape of the input layer.

If the network encodes a Matrix, then it will be a tuple with two equal-length
states. If the network encodes a Ket it will be a tuple with a single element
of length equal to the number of input units.
"""
input_shape(net::NeuralNetwork) = error("Not Implemented")

"""
    random_input_state(net)

Returns a random input state that can be fed into the network. This is not of
type `State`, but is rather the underlying numerical values.

This method is used for testing and to construct states. The shape of the state
will be the same of `input_shape`.
"""
random_input_state(net::NeuralNetwork) = error("Not Implemented")

"""
    is_analytic(net) -> Bool

Returns true if the network is an analytic function (almost everywhere) or if it
has real weights.
"""
is_analytic(net::NeuralNetwork) = false

"""
    num_params(net) -> Net

Returns the total number of parameters of the neural network.
"""
num_params(net::NeuralNetwork) = sum([length(getfield(net, f)) for f=fieldnames(typeof(net))])

# TODO does this even make sense?!
# the idea was that a shallow-copy of the weights of the net is not
# even a copy....
copy(net::NeuralNetwork) = net
