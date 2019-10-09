export logψ!

# Definitions for batched evaluation of networks
# When the networrks are not cached and therefore allocate
# the result structure
@inline logψ!(out::AbstractArray, net::NeuralNetwork, σ::State) where N =
    out .= logψ(net, config(σ))
@inline logψ!(out::AbstractArray, net::NeuralNetwork, σ::NTuple{N,<:AbstractArray}) where N =
    out .= logψ(net, σ)
@inline logψ!(out::AbstractArray, net::NeuralNetwork, σ::AbstractArray) =
    out .= logψ(net, σ)
#@inline function logψ_and_∇logψ!(der, out, n::NeuralNetwork, σ...)
#    lnψ, der = logψ_and_∇logψ!(der, n, σ...)
#    out .= lnψ
#¶    return (out, der)
#end

"""
    NNCache{N}

The base abstract type holding a cache for the network type `N<:NeuralNetwork`.
"""
abstract type NNBatchedCache{N} <: NNCache{N}
    #valid::Bool
end

cached(net::NeuralNetwork, batch_sz::Int) =
    CachedNet(net, cache(net, batch_sz))
cached(net::CachedNet, batch_sz::Int) =
    CachedNet(net.net, cache(net.net, batch_sz))

batch_size(cache::NNBatchedCache) = throw("Not Implemented")

# Definition for inplace evaluation of batched cached networks
@inline logψ!(out::AbstractArray, net::CachedNet, σ::NTuple{N,<:AbstractArray}) where N =
    logψ!(out, net.net, net.cache, σ...)
@inline logψ!(out::AbstractArray, cnet::CachedNet, σ::AbstractArray) =
    logψ!(out, cnet.net, cnet.cache, σ)
@inline logψ!(out::AbstractArray, cnet::CachedNet, σ::State) =
    logψ!(out, cnet, config(σ))

# Definition for allocating evaluation of batched cached networks
# Shadowing things at ~80 of base_cached_networks.jl
@inline logψ(net::CachedNet{NN,NC}, σ::NTuple{N,<:AbstractArray}) where {N,NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(first(σ)))
    out = similar(trainable_first(net), out_type(net), 1, b_sz)
    logψ!(out, net.net, net.cache, σ...)
end
@inline logψ(net::CachedNet{NN,NC}, σ::Vararg{N,T}) where {N,T,NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(first(σ)))
    out = similar(trainable_first(net), out_type(net), 1, b_sz)
    logψ!(out, net.net, net.cache, σ...)
end
@inline logψ(cnet::CachedNet{NN,NC}, σ::AbstractArray) where {NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(σ))
    out = similar(trainable_first(cnet), out_type(cnet), 1, b_sz)
    logψ!(out, cnet.net, cnet.cache, σ)
end

# Declare the two functions, even if config(blabla)=blabla, because of a shitty
# Julia's performance bug #32761
# see https://github.com/JuliaLang/julia/issues/32761
@inline logψ_and_∇logψ!(der, n::CachedNet{NN,NC}, σ::AbstractArray) where {NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(σ))
    out = similar(trainable_first(n), out_type(n), 1, b_sz)
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ)
    return out, der
end
@inline logψ_and_∇logψ!(der, n::CachedNet{NN,NC}, σ::NTuple{N,<:AbstractArray}) where {N,NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(first(σ)))
    out = similar(trainable_first(n), out_type(n), 1, b_sz)
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return out, der
end
@inline logψ_and_∇logψ!(der, n::CachedNet{NN,NC}, σ::Vararg{<:AbstractArray,N}) where {N,T,NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(first(σ)))
    out = similar(trainable_first(n), out_type(n), 1, b_sz)
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return out, der
end

@inline function logψ_and_∇logψ!(der, out, n::CachedNet, σ::State) =
    logψ_and_∇logψ!(der, out, n, config(σ))
@inline function logψ_and_∇logψ!(der, out, n::CachedNet, σ::NTuple{N,<:AbstractArray}) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return (out, der)
end
@inline function logψ_and_∇logψ!(der, out, n::CachedNet, σ::Vararg{<:AbstractArray,N}) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return (out, der)
end
@inline function logψ_and_∇logψ!(der, out, n::CachedNet, σ::AbstractArray) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ);
    return (out, der)
end


#
grad_cache(net::NeuralNetwork, batch_sz) =
    grad_cache(out_type(net), net, batch_sz)
grad_cache(T::Type{<:Number}, net::NeuralNetwork, batch_sz) = begin
    is_analytic(net) && return RealDerivative(T, net, batch_sz)
    return WirtingerDerivative(T, net, batch_sz)
end

function RealDerivative(T::Type{<:Number}, net::NeuralNetwork, batch_sz::Int)
    pars = trainable(net)

    vec    = similar(trainable_first(pars), T, _tlen(pars), batch_sz)
    i, fields = batched_weight_tuple(net, vec)
    return RealDerivative(fields, [vec])
end


## Things for batched states
preallocate_state_batch(arrT::Array,
                        T::Type{<:Real},
                        v::NAryState,
                        batch_sz) =
    _std_state_batch(arrT, T, v, batch_sz)

preallocate_state_batch(arrT::Array,
                        T::Type{<:Real},
                        v::DoubleState,
                        batch_sz) =
    _std_state_batch(arrT, T, v, batch_sz)


_std_state_batch(arrT::AbstractArray,
                 T::Type{<:Number},
                 v::NAryState,
                 batch_sz) =
    similar(arrT, T, nsites(v), batch_sz)

_std_state_batch(arrT::AbstractArray,
                 T::Type{<:Number},
                 v::DoubleState,
                 batch_sz) = begin
    vl = similar(arrT, T, nsites(row(v)), batch_sz)
    vr = similar(arrT, T, nsites(col(v)), batch_sz)
    return (vl, vr)
end


@inline store_state!(cache::Array,
             v::AbstractVector,
             i::Integer) = begin
    #@uviews cache v begin
        uview(cache, :, i) .= v
    #end
    return cache
end

@inline store_state!(cache::NTuple{2,<:Matrix},
             (vl, vr)::NTuple{2,<:AbstractVector},
             i::Integer) = begin
    cache_l, cache_r = cache
    #@uviews cache_l cache_r vl vr begin
        uview(cache_l, :,i) .= vl
        uview(cache_r, :,i) .= vr
    #end
    return cache
end
