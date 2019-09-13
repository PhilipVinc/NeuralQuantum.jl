"""
    NNCache{N}

The base abstract type holding a cache for the network type `N<:NeuralNetwork`.
"""
abstract type NNBatchedCache{N} <: NNCache{N}
    #valid::Bool
end
