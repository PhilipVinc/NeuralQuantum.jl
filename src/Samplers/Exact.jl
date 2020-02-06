export ExactSampler

"""
    ExactSampler(n_samples; seed=rand)

Constructs an exact sampler, which builds the full pdf of the quantum state and
samples it exactly.

This sampler can only be used on `indexable` spaces, and should be used only for
somewhat small systems (Closed N<14, Open N<7), as the computational cost increases
exponentially with the number of sites.

Initial seed can be set bu specifying `seed`.
"""
mutable struct ExactSampler{S} <: MonteCarloSampler
    samples_length::Int
    seed::S
end

ExactSampler(n_samples; seed=rand(UInt)) =
    ExactSampler(n_samples, seed)

## Cache
mutable struct ExactSamplerCache{H,A<:AbstractRNG,T} <: SamplerCache{ExactSampler}
    hilb::H
    loc_samples_length::Int

    rng::A
    steps_done::Int
    pdf::Vector{T}
end

function _sampler_cache(s::ExactSampler, v, hilb, net, par_cache)
    if net isa CachedNet
        if net.cache isa NNBatchedCache
            return _sampler_batched_cache(s, v, hilb, net, par_cache)
        end
    end

    loc_chain_length = Int(ceil(s.samples_length/num_workers(par_cache)))

    loc_seed         = worker_local_seed(s.seed, par_cache)
    rng              = build_rng_generator_T(prob, loc_seed)

    ExactSamplerCache(hilb,
                      loc_chain_length,
                      rng,
                      0,
                      zeros(real(eltype(v)), spacedimension(hilb)))
end
##
function init_sampler!(sampler::ExactSampler, net::Union{MatrixNet,KetNet}, σ, c::ExactSamplerCache)
    # Compute the distribution
    for i=1:spacedimension(c.hilb)
        set!(σ, c.hilb, i)
        c.pdf[i] = exp(log_prob_ψ(net, σ))
    end
    c.pdf ./= sum(c.pdf)
    cumsum!(c.pdf, c.pdf)
    c.steps_done = 0

    samplenext!(σ, sampler, net, c)
    return c
end

chain_length(s::ExactSampler, c::ExactSamplerCache) = s.samples_length

done(s::ExactSampler, σ, c) = c.steps_done >= chain_length(s, c)

function samplenext!(σ, s::ExactSampler, net::Union{MatrixNet,KetNet}, c)
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    # sample and set
    r = rand(c.rng)
    i = searchsortedfirst(c.pdf, r)
    set!(σ, i-1)
    c.steps_done += 1

    return true
end

# return findfirst(probs .> number)
function sample_from_distr(number::Real, probs::Vector)
    @inbounds for (i,p)=enumerate(probs)
        if p > number
            return i
        end
    end
end

## Batched

mutable struct ExactSamplerBatchedCache{H,A<:AbstractRNG,I,Ia,T,V,Pc} <: SamplerCache{ExactSampler}
    hilb::H
    loc_samples_length::Int

    loc_hilb_numbers::I
    all_hilb_numbers::Ia

    rng::A
    steps_done::Int
    pdf::Vector{T}

    logψ::V

    parallel_cache::Pc
end

function _sampler_batched_cache(s::ExactSampler, v, hilb, net, par_cache)
    loc_chain_length = Int(ceil(s.samples_length/num_workers(par_cache)))

    loc_seed         = worker_local_seed(s.seed, par_cache)
    rng              = build_rng_generator_T(out_similar(net), loc_seed)

    # all blocks of iterations per worker
    all_hilb_numbers = [_iterator_blocks(1:spacedimension(hilb),
                        i,
                        num_workers(par_cache)) for i=1:num_workers(par_cache)]
    loc_hilb_iter    = my_block(all_hilb_numbers, par_cache)
    all_hilb_numbers = Int32.(length.(all_hilb_numbers))

    ExactSamplerBatchedCache(hilb,
                             loc_chain_length,
                             loc_hilb_iter,
                             all_hilb_numbers,
                             rng,
                             0,
                             zeros(real(out_type(net)), spacedimension(hilb)),
                             out_similar(net),
                             par_cache)
end

##
function init_sampler!(sampler::ExactSampler,
                       net::Union{MatrixNet,KetNet},
                       σ, c::ExactSamplerBatchedCache)
    batch_sz = batch_size(net)
    # Compute the distribution
    for batch_i=Iterators.partition(c.loc_hilb_numbers, batch_sz)
        for (i, h_i)=enumerate(batch_i)
            set!(unsafe_get_batch(σ, i), c.hilb, h_i)
        end
        log_prob_ψ!(c.logψ, c.logψ, net, σ)

        for (i, h_i)=enumerate(batch_i)
            c.pdf[h_i] = exp.(c.logψ[i])
        end
    end
    worker_allgatherv!(c.pdf, c.all_hilb_numbers, c.parallel_cache)

    #c.pdf ./= sum(c.pdf)
    tot = sum(c.pdf)
    cumsum!(c.pdf, c.pdf)
    c.pdf ./= tot

    c.steps_done = 0

    samplenext!(σ, σ, sampler, net, c)
    return c
end

chain_length(s::ExactSampler, c::ExactSamplerBatchedCache) = c.loc_samples_length

function samplenext!(σ_out::T, σ_in::T,
                     s::ExactSampler,
                     net::Union{MatrixNet,KetNet},
                     c::ExactSamplerBatchedCache) where {T<:Union{AStateBatch, ADoubleStateBatch}}
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    for i=1:batch_size(net)
        r = rand(c.rng)
        hi = searchsortedfirst(c.pdf, r)
        set!(unsafe_get_batch(σ_out, i), c.hilb, hi)
    end

    c.steps_done += 1
    return true
end
