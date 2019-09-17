# This file contains basic structures and method dispatched when the network
# is evaluated with respect to a state with an associated lookup table.
# At the moment, those have a bad performance and should not be used

# !!!!! Ignore this file! Lookup tables are not really needed....

abstract type NNLookUp
    #valid
end

"""
    lookup(net)

Constructs the `NNCache{typeof(net)}` object that holds the cache for this network.
If it has not been implemented returns nothing.
"""
lookup(net::NeuralNetwork) = nothing
lookup(net::CachedNet) = lookup(net.net)
