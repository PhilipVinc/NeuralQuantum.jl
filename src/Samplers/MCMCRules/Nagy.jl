## LocalRule
export NagyRule

struct NagyRule{T<:Vector} <: MCMCRule
    adjacency_list::T
end

function NagyRule(H::AbsLinearOperator)
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

    return NagyRule{typeof(couplings)}(couplings)
end


struct NagyRuleCache
end

function RuleSamplerCache(rule::NagyRule, s, v, net, par_cache)
    return NagyRuleCache()
end


function propose_step!(σp::ADoubleState, s::MetropolisSampler{<:NagyRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    adj_list = s.rule.adjacency_list
    hilb     = physical(c.hilb)

    # determine move type
    move_type = rand(c.rng, 1:8)#1:n_moves)
    sites_to_flip = 1:nsites(hilb)

    # 1- Flip a spin
    if move_type == 1 || move_type == 2                 # hopping in σ
        flipat_r1 = rand(c.rng, sites_to_flip)
        flipat_r2 = rand(c.rng, adj_list[flipat_r1])

        @inbounds flipat!(c.rng, row(σp), hilb, flipat_r1)
        @inbounds flipat!(c.rng, row(σp), hilb, flipat_r2)

    elseif move_type == 3 || move_type == 4             # hopping in η
        flipat_c1 = rand(c.rng, sites_to_flip)
        flipat_c2 = rand(c.rng, adj_list[flipat_c1])

        @inbounds flipat!(c.rng, col(σp), hilb, flipat_c1)
        @inbounds flipat!(c.rng, col(σp), hilb, flipat_c2)

    elseif move_type == 5                               # exctitation in σ
        flipat_r1 = rand(c.rng, sites_to_flip)

        @inbounds flipat!(c.rng, row(σp), hilb, flipat_r1)

    elseif move_type == 6                               # excitations in η
        flipat_r1 = rand(c.rng, sites_to_flip)

        @inbounds flipat!(c.rng, col(σp), hilb, flipat_r1)

    elseif move_type == 7                                # Dissipator
        flipat_r  = rand(c.rng, sites_to_flip)
        old_val_r = row(σp)[flipat_r]

        if old_val_r == 0
            if rand(c.rng, 1:10) == 1 # excite with 10% chance
                @inbounds flipat!(c.rng, row(σp), hilb, flipat_r)
            end
        else # always dissipate
            @inbounds flipat!(c.rng, row(σp), hilb, flipat_r)
        end

        flipat_c  = flipat_r #rand(c.rng, sites_to_flip)
        old_val_c = col(σp)[flipat_c]
        if old_val_c == 0
            if rand(c.rng, 1:10) == 1 # excite with 10% chance
                @inbounds flipat!(c.rng, col(σp), hilb, flipat_c)
            end
        else # always dissipate
            @inbounds flipat!(c.rng, col(σp), hilb, flipat_c)
        end

    elseif move_type == 8                                # jumper
        flipat_r = rand(c.rng, sites_to_flip)
        flipat_c = rand(c.rng, sites_to_flip)

        @inbounds old_val_r, _ =flipat!(c.rng, row(σp), hilb, flipat_r)
        @inbounds old_val_c, _ =flipat!(c.rng, col(σp), hilb, flipat_c)
    end
#    elseif move_type == 9
#        if rand(c.rng, 1:100) == 1
#            zero!(σ)
#        end
        #prob = exp(2*real(net(σ)) - logψ_σ)
        #rv = rand(c.rng)
        #if !(rv < prob)
        #    setat!(row(σ), flipat_r, old_val_r)
        #    setat!(col(σ), flipat_c, old_val_c)
        #else
        #end
#    end

    return true
end
