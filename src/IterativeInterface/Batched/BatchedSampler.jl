export BatchedSampler

mutable struct BatchedSampler2{N,BN,P,IC,EC,S,SC,V,Vb,Vi,Pv,Gv,LC} <: AbstractIterativeSampler
    net::N
    bnet::BN
    problem::P
    itercache::IC
    sampled_values::EC

    sampler::S
    sampler_cache::SC

    ψvals::Pv
    ∇vals::Gv
    ψ_batch::Pv
    ∇ψ_batch::Gv

    accum::LC

    v::V
    vc::Vb
    vi_vec::Vi
    batch_sz::Int
end


"""
    IterativeSampler(net, sampler, algorithm, problem)

Create a single-thread iterative sampler for the quantities defined
by algorithm.
"""
function BatchedSampler2(net,
                        sampl,
                        prob,
                        algo=prob;
                        batch_sz=2^3)

    !ispow2(batch_sz) && @warn "Batch size is not a power of 2. Bad performance guaranteed."

    cnet           = cached(net)
    bnet           = cached(net, batch_sz)
    evaluated_vals = EvaluatedNetwork(algo, net)
    itercache      = SamplingCache(algo, prob, net)
    v              = state(prob, bnet)
    sampler_cache  = init_sampler!(sampl, bnet, basis(prob), v)

    # Total length of the markov chain, used to preallocate
    N_tot          = chain_length(sampl, sampler_cache)
    @assert N_tot>0 "Error: chain length not inferred"

    ψvals  = similar(trainable_first(net), out_type(net), 1, N_tot)
    ∇vals  = grad_cache(net, N_tot)
    vi_vec = zeros(Int, N_tot)

    ψ_batch = similar(trainable_first(net), out_type(net), 1, batch_sz)
    ∇ψ_batch = grad_cache(net, batch_sz)

    accum = Accumulator(net, prob, N_tot, batch_sz)

    vc   = preallocate_state_batch(trainable_first(net),
                                   input_type(net),
                                   v,
                                   batch_sz)

    BatchedSampler2(cnet, bnet,
                    prob,
                    itercache,
                    evaluated_vals,
                    sampl,
                    sampler_cache,
                    ψvals,
                    ∇vals,
                    ψ_batch,
                    ∇ψ_batch,
                    accum,
                    v,
                    vc,
                    vi_vec,
                    batch_sz)
end

"""
    sample!(is::IterativeSampler)

Samples the quantities accordingly, returning the sampled values and sampling history.
"""
function sample!(is::BatchedSampler2)
    init_sampler!(is.sampler, is.net, is.v, is.sampler_cache)
    #vc_vec = zeros(0)
    vi_vec = is.vi_vec .= 0
    for i=1:typemax(Int)
        vi_vec[i] = index(is.v)
        !samplenext!(is.v, is.sampler, is.net, is.sampler_cache) && break
    end

    b_sz = is.batch_sz
    Nv   = length(vi_vec)
    vc   = is.vc

    # Those won't work on the cpu
    ψvals_data    = uview(is.ψvals)
    ∇vals_data    = uview(first(vec_data(is.∇vals)))
    ∇ψ_batch_data = uview(first(vec_data(is.∇ψ_batch)))

    for i=1:b_sz:(Nv-b_sz)
        for j=1:b_sz
            set_index!(is.v, vi_vec[i+j-1])
            store_state!(vc, config(is.v), j)
        end
        @views logψ_and_∇logψ!(is.∇ψ_batch, ψvals_data[:,i:i+b_sz-1], is.bnet, vc)
        ∇vals_data[:,i:i+b_sz-1] .= ∇ψ_batch_data
    end

    ip = Nv-b_sz+1
    i = last(1:b_sz:(Nv)); l = Nv-i+1
    for j=1:l
        set_index!(is.v, vi_vec[i+j-1])
        store_state!(vc, config(is.v), j+(l-b_sz))
    end
    @views logψ_and_∇logψ!(is.∇ψ_batch, ψvals_data[:, ip:end], is.bnet, vc)
    @views ∇vals_data[:,ip:end] .= ∇ψ_batch_data

    compute_local_term!(is)

    # Now I have gradients and other stuff

    return is.accum
end

function compute_local_term!(is::BatchedSampler2)
    vi_vec = is.vi_vec
    ψvals_data    = collect(uview(is.ψvals))
    ∇vals_data    = uview(first(vec_data(is.∇vals)))
    ∇ψ_batch_data = uview(first(vec_data(is.∇ψ_batch)))

    if is.problem isa LRhoSquaredProblem
        op = is.problem.L
    elseif is.problem isa HamiltonianGSEnergyProblem
        op = is.problem.H
    end

    ## Ended sampling those things
    accum = is.accum
    init!(accum)
    for (i, σi)=enumerate(vi_vec)
        σ = set_index!(is.v, σi)
        @views push!(accum, is.ψvals[i], ∇vals_data[:,i])
        accumulate_connections!(accum, op, σ)
    end
    finalize!(accum)
end

Base.show(io::IO, is::BatchedSampler2) = print(io,
    "BatchedSampler2 for :"*
    "\n\tnet\t\t: $(is.net)"*
    "\n\tproblem\t: $(is.problem)"*
    "\n\tsampler\t: $(is.sampler)")
