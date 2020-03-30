export ExchangeRule

"""
    ExchangeRule(graph)

Transition rule for Metropolis-Hastings sampling where at every step a random
couple of sites i,j is switched.

Couples of sites are generated from the graph `graph` (or operator), where all
sites that are connected (or coupled by a 2-body term) are considered for switches.
"""
struct ExchangeRule{T<:Vector} <: MCMCRule
    distances::T
end

function ExchangeRule(H::AbsLinearOperator)
    N_sites = nsites(basis(H))

    all_sites = sites(H)
    if !(eltype(all_sites) <: Vector)
        all_sites = [all_sites]
    end

    couplings = Vector{Tuple{Int,Int}}()
    for coupling=all_sites
        length(coupling) == 1 && continue
        if length(coupling) > 2
            @warn "Can't exchange between 3-site couplings. This coupling is ignored"
            continue
        end
        push!(couplings, (coupling[1], coupling[2]))
    end

    return ExchangeRule{typeof(couplings)}(couplings)
end

function propose_step!(σp::Union{AState,ADoubleState}, s::MetropolisSampler{<:ExchangeRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    dist = c.rule.distances

    couple_i = rand(c.rng, 1:length(dist))
    i, j = dist[couple_i]
    if σp isa AState
        @inbounds tmp = σp[i]
        @inbounds σp[i] = σp[j]
        @inbounds σp[j] = tmp
    else
        throw("not implemented")
    end
end

function propose_step!(σp_b::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{<:ExchangeRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    dist = s.rule.distances

    for σp=states(σp_b)

        couple_i = rand(c.rng, 1:length(dist))
        i, j = dist[couple_i]
        if σp isa AState
            @inbounds tmp = σp[i]
            @inbounds σp[i] = σp[j]
            @inbounds σp[j] = tmp
        else
            throw("not implemented")
        end
    end
end
