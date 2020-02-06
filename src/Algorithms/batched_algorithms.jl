"""
    algorithm_cache(algorithm, problem, network)

Builds the cache for the given algorithm
"""
function algorithm_cache end

"""
    setup_algorithm!(alg_cache, ∇C, ∇logψ, par_cache)
"""
setup_algorithm!(g, data, grad, par_cache) = setup_algorithm!(g, data, par_cache)
