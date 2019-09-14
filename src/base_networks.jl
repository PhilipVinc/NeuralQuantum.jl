export logψ, ∇logψ, ∇logψ!, logψ_and_∇logψ, logψ_and_∇logψ!, grad_cache

# abstract type NeuralNetwork end
@inline logψ(net::NeuralNetwork, σ) = net(σ)
@inline log_prob_ψ(net, σ...)       = 2.0*real(net(σ...))
@inline ∇logψ(args...)              = logψ_and_∇logψ(args...)[2]
@inline ∇logψ!(args...)             = logψ_and_∇logψ!(args...)[2]

function logψ_and_∇logψ(net::NeuralNetwork, σ::Vararg{N,V})where {N,V}
    der = grad_cache(net)
    y, der = logψ_and_∇logψ!(der, net, σ)

    return y, der
end

function logψ_and_∇logψ!(der, net::NeuralNetwork, σ)
    σ = config(σ)
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
input_type(net::NeuralNetwork) = error("Not Implemented")
out_type(net::NeuralNetwork) = error("Not Implemented")
input_shape(net::NeuralNetwork) = error("Not Implemented")
random_input_state(net::NeuralNetwork) = error("Not Implemented")
is_analytic(net::NeuralNetwork) = false
num_params(net::NeuralNetwork) = sum([length(getfield(net, f)) for f=fieldnames(typeof(net))])
copy(net::NeuralNetwork) = net
