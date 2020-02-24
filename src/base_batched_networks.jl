export logψ!

# Definitions for batched evaluation of networks
# When the networrks are not cached and therefore allocate
# the result structure
@inline logψ!(out::AbstractMatrix, net::NeuralNetwork, σ::ADoubleStateBatch) where N =
    out .= net(σ...)
@inline logψ!(out::AbstractMatrix, net::NeuralNetwork, σ::AStateBatch) =
    out .= net(σ)

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

batch_size(net::CachedNet) = batch_size(net.cache)
batch_size(cache::NNBatchedCache) = throw("Not Implemented")

out_similar(net::CachedNet{N,C}) where{N,C<:NNBatchedCache} =
    similar(trainable_first(net), out_type(net), 1, batch_size(net))
out_similar(net::CachedNet{N,C}, dims::Vararg{T,I}) where{T<:Integer, I, N,C<:NNBatchedCache} =
    similar(trainable_first(net), out_type(net), 1, dims...)


# Definition for inplace evaluation of batched cached networks
@inline logψ!(out::AbstractMatrix, net::CachedNet, σ::Vararg{T,N}) where {N,T<:AbstractVecOrMat} =
    logψ!(out, net.net, net.cache, σ...)
@inline logψ!(out::AbstractMatrix, net::CachedNet, σ::ADoubleStateBatch) =
    logψ!(out, net.net, net.cache, row(σ), col(σ))
@inline logψ!(out::AbstractMatrix, cnet::CachedNet, σ::AStateBatch) =
    logψ!(out, cnet.net, cnet.cache, σ)

# Definition for allocating evaluation of batched cached networks
# Shadowing things at ~80 of base_cached_networks.jl
@inline logψ(net::CachedNet{NN,NC}, σr::AStateBatch, σc::AStateBatch) where {N,T,NN,NC<:NNBatchedCache} = begin
    logψ(net, (σr, σc))
end
@inline logψ(net::CachedNet{NN,NC}, σ::ADoubleStateBatch) where {N,T,NN,NC<:NNBatchedCache} = begin
    b_sz = last(size(first(σ)))
    out = similar(trainable_first(net), out_type(net), 1, b_sz)
    logψ!(out, net.net, net.cache, σ...)
end
@inline logψ(cnet::CachedNet{NN,NC}, σ::AStateBatch) where {NN,NC<:NNBatchedCache} = begin
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

@inline function logψ_and_∇logψ!(der::AbstractDerivative, out, n::CachedNet, σ::NTuple{N,<:AbstractArray}) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return (out, der)
end
@inline function logψ_and_∇logψ!(der::AbstractDerivative, out, n::CachedNet, σ::Vararg{<:AbstractArray,N}) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ...)
    return (out, der)
end
@inline function logψ_and_∇logψ!(der::AbstractDerivative, out, n::CachedNet, σ::AbstractArray) where N
    logψ_and_∇logψ!(der, out, n.net, n.cache, σ);
    return (out, der)
end


#
grad_cache(net::NeuralNetwork, batch_sz) =
    grad_cache(out_type(net), net, batch_sz)
function grad_cache(T::Type{<:Number}, net::NeuralNetwork, batch_sz)
    is_analytic(net) && return RealDerivative(T, net, batch_sz)
    return WirtingerDerivative(T, net, batch_sz)
end

#
grad_cache(T::Type{<:Number}, net::CachedNet{N,C}) where {N,C<:NNBatchedCache} =
    grad_cache(T, net, batch_size(net))

function RealDerivative(T::Type{<:Number}, net::NeuralNetwork, batch_sz::Int)
    pars = trainable(net)

    vec       = similar(trainable_first(pars), T, _tlen(pars), batch_sz)
    i, fields = batched_weight_tuple(net, vec)
    return RealDerivative(fields, [vec])
end

#
"""
    grad_cache_vec(net, batch_sz, vec_len)

Builds a vector of `grad_cache` batching `batch_sz` evaluations.
All share the same underlying vector.
"""
grad_cache(net::NeuralNetwork, batch_sz::Integer, vec_len::Integer) =
    grad_cache(out_type(net), net, batch_sz, vec_len)

function grad_cache(T::Type{<:Number}, net::NeuralNetwork, batch_sz, vec_len)
    pars  = trainable(net)
    npars = _tlen(pars)

    if is_analytic(net)
        vec       = similar(trainable_first(pars), T, npars, batch_sz, vec_len)
        # compute type
        j, fields = batched_weight_tuple(net, view(vec, :, :, 1))
        der       = RealDerivative(fields, (view(vec, :, :, 1),))
        ders      = Vector{typeof(der)}()
        for i=1:vec_len
            vec_rsp   = view(vec, :, :, i)
            j, fields = batched_weight_tuple(net, vec_rsp)
            der       = RealDerivative(fields, (vec_rsp,))
            push!(ders, der)
        end
        vecs = (vec,)
    else
        throw("To implement")
    end
    return ders, vecs
end


## Things for batched states
preallocate_state_batch(arrT::Array,
                        T::Type{<:Real},
                        v::AState,
                        batch_sz) =
    _std_state_batch(arrT, T, v, batch_sz)

preallocate_state_batch(arrT::Array,
                        T::Type{<:Real},
                        v::ADoubleState,
                        batch_sz) =
    _std_state_batch(arrT, T, v, batch_sz)


_std_state_batch(arrT::AbstractArray,
                 T::Type{<:Number},
                 v::AState,
                 batch_sz) =
    similar(arrT, T, length(v), batch_sz)

_std_state_batch(arrT::AbstractArray,
                 T::Type{<:Number},
                 v::ADoubleState,
                 batch_sz) = begin
    vl = similar(arrT, T, length(row(v)), batch_sz)
    vr = similar(arrT, T, length(col(v)), batch_sz)
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

@inline @inbounds unsafe_get_batch(σ::AStateBatch, i::Integer) = uview(σ, :, i)
@inline @inbounds unsafe_get_batch(σ::ADoubleStateBatch, i::Integer) = (uview(row(σ), :, i), uview(col(σ), :, i))
@inline num_batches(σ::AStateBatch) = size(σ, 2)
@inline num_batches(σ::ADoubleStateBatch) = size(row(σ), 2)

# Automatic evaluation of batches
logψ!(out::AbstractArray{T,3}, cnet::CachedNet, σ::NTuple{2,AbstractArray{T2,3}}) where{T,T2} =
    logψ!(out, cnet, first(σ), last(σ))
function logψ!(out::AbstractArray{T,3}, cnet::Union{NeuralNetwork, CachedNet}, σr::AT, σc::AT) where {T, T2, AT<:AbstractArray{T2, 3}}
    n_batches = size(σr, 3)
    for i=1:n_batches
        σr_i  = unsafe_get_el(σr,  i)
        σc_i  = unsafe_get_el(σc,  i)
        out_i = unsafe_get_el(out, i)
        logψ!(out_i, cnet, σr_i, σc_i)
    end
    return out
end

function logψ!(out::AbstractArray{T,3}, cnet::NeuralNetwork, σ::AT) where {T, T2, AT<:AbstractArray{T2, 3}}
    n_batches = size(σ, 3)
    for i=1:n_batches
        σ_i   = unsafe_get_el(σ,  i)
        out_i = unsafe_get_el(out, i)
        logψ!(out_i, cnet, σ_i)
    end
    return out
end


# Compute batches of those things
function logψ_and_∇logψ!(der::Vector{<:AbstractDerivative}, out::AbstractArray{T,3},
                         net::Union{NeuralNetwork, CachedNet},
                         σ::Union{AStateBatchVec, ADoubleStateBatchVec}) where {T}
    n_batches  = length(der)
    for i=1:n_batches
        σ_i   = unsafe_get_el(σ,  i)
        out_i = unsafe_get_el(out, i)
        logψ_and_∇logψ!(der[i], out_i, net, σ_i)
    end
    return out
end

function logψ!(out::AbstractArray{T,3}, net::Union{NeuralNetwork, CachedNet},
               σ::Union{AStateBatchVec, ADoubleStateBatchVec}) where {T}
    n_batches  = length(der)
    for i=1:n_batches
        σ_i   = unsafe_get_el(σ,  i)
        out_i = unsafe_get_el(out, i)
        logψ!(out_i, net, σ_i)
    end
    return out
end

function logψ_and_∇logψ(net::Union{NeuralNetwork, CachedNet},
                        σ::Union{AStateBatchVec, ADoubleStateBatchVec}) where {T}
    ∇vals, ∇vec    = grad_cache(bnet, 1, batch_size(σ), chain_length(σ))
    out = similar(out_similar(net), batch_size(σ), chain_length(σ))
    logψ_and_∇logψ!(∇vals, out, net, σ)
    return ∇vals, out
end

function logψ(net::Union{NeuralNetwork, CachedNet},
               σ::Union{AStateBatch, ADoubleStateBatch}) where {T}
    out = out_similar(net, batch_size(σ))
    logψ!(out, net, σ)
end

function logψ(net::Union{NeuralNetwork, CachedNet},
               σ::Union{AStateBatchVec, ADoubleStateBatchVec}) where {T}
    out = similar(out_similar(net), 1, batch_size(σ), chain_length(σ))
    logψ!(out, net, σ)
end

# diagonal
logψ!(out::AbstractArray, net::MatrixNeuralNetwork, cache::NNCache, σ::AStateBatch) =
    logψ!(out, net, cache, σ, σ)

log_prob_ψ!(prob::AbstractArray, net::NeuralNetwork, σ) =
    log_prob_ψ!(prob, prob, net, σ)

function log_prob_ψ!(prob::AbstractArray, out_tmp::AbstractArray, net::MatrixNet, σ::AStateOrBatchOrVec)
    logψ!(out_tmp, net, σ, σ)
    prob .= abs.(out_tmp)
    return prob
end

function log_prob_ψ!(prob::AbstractArray, out_tmp::AbstractArray, net::NeuralNetwork, σ)
    logψ!(out_tmp, net, σ)
    prob .= 2 .* real.(out_tmp)
    return prob
end
