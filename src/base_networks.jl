# This file contains some aliases and generic methods to use neural networks.

export logψ, ∇logψ, ∇logψ!, logψ_and_∇logψ, logψ_and_∇logψ!, grad_cache

"""
    grad_cache([T=out_type(net)], net::N) -> tuple

Creates a tuple holding all derivatives of the network N.
If the type is not specified, it defaults to the output type of the network.
"""
grad_cache(net::NeuralNetwork) = grad_cache(out_type(net), net)
grad_cache(T::Type{<:Number}, net::NeuralNetwork) = begin
    is_analytic(net) && return RealDerivative(T, net)
    return WirtingerDerivative(T, net)
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
function logψ end

@specialize_vararg 5 @inline logψ(net::NeuralNetwork, σ...) = net(σ...)
# Standard probability
@inline log_prob_ψ(net, σ)          = 2.0*real(net(σ))

@specialize_vararg 5 @inline ∇logψ(args...)   = logψ_and_∇logψ(args...)[2]
@specialize_vararg 5 @inline ∇logψ!(args...)  = logψ_and_∇logψ!(args...)[2]

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
function logψ_and_∇logψ(net::NeuralNetwork, σ) #::Vararg{N,V})where {N,V}
    der = grad_cache(net)
    y, der = logψ_and_∇logψ!(der, net, σ)
    return y, der
end

# Autodiff version of the gradient: if you are hitting this, expect
# bad performance. Maybe you should hand-code the gradient of your model?
function logψ_and_∇logψ!(der, net::NeuralNetwork, σ)

    # Old implementation. Probably must be resuscitated one day, but not compatible
    # with our implementation of gradient caches and trainable.
    #=
    # If it is a KetNet we should not use σ...
    if isa(net, KetNeuralNetwork)
        y, back = pullback(net -> net(σ), net)
    else
        # Zygote's autodiff: generate the pullback
        y, back = pullback(net -> net(σ...), net)
    end

    # This computes the gradient, which is the conjugate of the derivative
    _der = back(Int8(1))[1]
    =#

    pars = params(net)
    fun = isa(net, KetNeuralNetwork) ? ()->net(σ) : ()->net(σ...)
    y, back = pullback(fun, pars)
    _der = back(Int8(1))

    # Convert the AD derivative to our contiguous type.
    apply_recurse!(fields(der), _der, net, conj!)
    return y, der
end

# Common Operations on networks that should always be defined
# If you hit this section it probably means that you forgot to define one
# of those functions on the type of your neural network.
"""
    input_type(net) -> Type

Returns the numerical `eltype` of the input of the network.
By default it returns `real(eltype(trainable_first(net)))`, but you should
ensure this result is correct (or specialize input_type for your network) so
that the computation is type stable
"""
input_type(net::NeuralNetwork) = real(eltype(trainable_first(net)))

"""
    out_type(net) -> Type

Returns the numerical `eltype` of the output of the network.
"""
out_type(net::NeuralNetwork) = error("Not Implemented")

"""
    out_similar(net) -> zero(T)
    out_similar(net, dims...) -> zeros(T, dims...)

Returns the result of similar for the output of the network.
This is either a scalar or a row vector.
"""
@inline out_similar(net::NeuralNetwork) = zero(out_type(net))
@inline out_similar(net::NeuralNetwork, dims::Vararg{T,N}) where {T<:Integer,N} = zeros(out_type(net), 1, dims...)

"""
    is_analytic(net) -> Bool

Returns true if the network is an R-analytic function (almost everywhere).

Please note that this is a weaker condition than holomorphicity, which is
equivalent to C-analytic.

Ref:
    [1] K. Kreuz-Delgado The Complex Gradient Operator and the CR-Calculus,
        arXiv:0906.4835
"""
is_analytic(net::NeuralNetwork) = false

"""
    num_params(net) -> Net

Returns the total number of parameters of the neural network.
"""
num_params(net::NeuralNetwork) = trainable_length(net)

# TODO does this even make sense?!
# the idea was that a shallow-copy of the weights of the net is not
# even a copy....
Base.copy(net::NeuralNetwork) = net

## For Matrix networks evaluated squared
abstract type MatrixNeuralNetwork <: NeuralNetwork end
abstract type PureNeuralNetwork <: NeuralNetwork end
abstract type KetNeuralNetwork <: NeuralNetwork end

@inline log_prob_ψ(net::MatrixNeuralNetwork,
                        σ::ADoubleStateOrBatchOrVec) = 2.0*real(net(σ))
# For Matrix network evaluated on diagonal
@inline log_prob_ψ(net::MatrixNeuralNetwork,
                        σ::AStateOrBatchOrVec) = real(net(σ))
