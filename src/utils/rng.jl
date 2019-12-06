export init_random_pars!

"""
    build_rng_generator_T(T::abstractArray, seed)

builds an RNG generator for the type of T.
    Defaults to MersenneTWister.
"""
function build_rng_generator_T(arrT::Array, seed)
    return MersenneTwister(seed)
end

"""
    init_random_pars!([rng=GLOBAL_RNG], net; sigma=0.01 )

Initializes all weights of the neural network to a random distribution with
variance `sigma`.
"""
init_random_pars!(net::NeuralNetwork, args...; kwargs...) = init_random_pars!(GLOBAL_RNG, net, args...; kwargs...)
function init_random_pars!(rng::AbstractRNG, net; sigma=0.01)
    for f=trainable(net)
        init_random_pars!(rng, f; sigma=sigma)
    end
    return net
end

function init_random_pars!(rng::AbstractRNG, f::AbstractArray; sigma=0.01)
    randn!(rng, f)
    f .*= sqrt(sigma)

    return f
end
