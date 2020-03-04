#=
    generate_state.jl

This file contains several methods to generate the state associated to a
Neural Network and an hilbert space.
While the basic logic is simple, handling batches of states and diagonal
states of the density matrix complicates it.
=#

state(op::AbsLinearOperator, net, args...) =
    state(basis(op), net, args...)
state(hilb::AbstractHilbert, net::NeuralNetwork, args...) =
    state(input_type(net), hilb, net, args...)

state(T::Type{<:Number}, op::AbsLinearOperator, net, args...) =
    state(T, basis(op), net, args...)
state(T::Type{<:Number}, hilb::AbstractHilbert, net::CachedNet, args...) =
    state(T, hilb, net.net, net.cache, args...)

# If it's a cached net enforce the batch_size
state(T::Type{<:Number}, hilb::AbstractHilbert, net::NeuralNetwork, cache::NNCache, dims::Vararg{Int,n}) where n =
    state(T, hilb, net, dims...)
state(T::Type{<:Number}, hilb::AbstractHilbert, net::NeuralNetwork, cache::NNBatchedCache, dims::Vararg{Int,n}) where n =
    state(T, hilb, net, batch_size(cache), dims...)

state(T::Type{<:Number}, hilb::AbstractHilbert,
      net::Union{MatrixNeuralNetwork, KetNeuralNetwork},
      dims::Vararg{Int,n}) where n =
    state(trainable_first(net), T, hilb, dims...)
