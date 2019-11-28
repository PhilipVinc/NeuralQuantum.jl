"""
    build_rng_generator_T(T::abstractArray, seed)

builds an RNG generator for the type of T.
    Defaults to MersenneTWister.
"""
function build_rng_generator_T(arrT::Array, seed)
    return MersenneTwister(seed)
end
