## MEtropolis
export Nagy
struct Nagy{T<:Vector} <: MCMCRule
    adjacency_list::T
end

function markov_chain_step!(σ, s::MCMCSampler{N}, net::MatrixNet, c) where N<:Nagy
    # Check termination condition, and return if verified
    done(s, σ, c) && return false

    logψ_σ = 2*real(net(σ)) #log_prob_ψ(net, σ)

    adj_list = s.rule.adjacency_list

    n_moves = 5+4 #+ length(first(adjacency_matrix))*2
    # 1- Flip a spin
    move_type = rand(c.rng, 5:9)#1:n_moves)
    if move_type == 1 || move_type == 2                 # hopping in σ
        flipat_r1 = rand(c.rng, 1:nsites(row(σ)))
        flipat_r2 = rand(c.rng, adj_list[flipat_r1])

        old_val_r1, new_val_r1 =flipat!(c.rng, row(σ), flipat_r1)
        old_val_r2, new_val_r2 =flipat!(c.rng, row(σ), flipat_r2)
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(row(σ), flipat_r1, old_val_r1)
            setat!(row(σ), flipat_r2, old_val_r2)
        else c.steps_accepted += 1 end
    elseif move_type == 3 || move_type == 4             # hopping in η
        flipat_c1 = rand(c.rng, 1:nsites(col(σ)))
        flipat_c2 = rand(c.rng, adj_list[flipat_c1])

        old_val_c1, _ =flipat!(c.rng, col(σ), flipat_c1)
        old_val_c2, _ =flipat!(c.rng, col(σ), flipat_c2)
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(col(σ), flipat_c1, old_val_c1)
            setat!(col(σ), flipat_c2, old_val_c2)
        else c.steps_accepted += 1 end
    elseif move_type == 5                               # exctitation in σ
        flipat_r1 = rand(c.rng, 1:nsites(row(σ)))

        old_val_r1, new_val_r1 =flipat!(c.rng, row(σ), flipat_r1)
        logψ_σp = 2*real(net(σ))
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(row(σ), flipat_r1, old_val_r1)
        else c.steps_accepted += 1 end
    elseif move_type == 6                               # excitations in η
        flipat_r1 = rand(c.rng, 1:nsites(col(σ)))

        old_val_r1, new_val_r1 =flipat!(c.rng, col(σ), flipat_r1)
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(col(σ), flipat_r1, old_val_r1)
        else c.steps_accepted += 1 end
    elseif move_type == 7                                # Dissipator
        flipat_r  = rand(c.rng, 1:nsites(row(σ)))
        old_val_r = config(row(σ))[flipat_r]
        if old_val_r == 0
            if rand(c.rng, 1:10) == 1 # excite with 10% chance
                old_val_r, _ = flipat!(c.rng, row(σ), flipat_r)
            end
        else # always dissipate
            old_val_r, _ = flipat!(c.rng, row(σ), flipat_r)
        end

        flipat_c  = rand(c.rng, 1:nsites(col(σ)))
        old_val_c = config(col(σ))[flipat_c]
        if old_val_c == 0
            if rand(c.rng, 1:10) == 1 # excite with 10% chance
                old_val_c, _ = flipat!(c.rng, col(σ), flipat_c)
            end
        else # always dissipate
            old_val_c, _ = flipat!(c.rng, col(σ), flipat_c)
        end

        # check and revert
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(row(σ), flipat_r, old_val_r)
            setat!(col(σ), flipat_c, old_val_c)
        else
            c.steps_accepted += 1
        end
    elseif move_type == 8                                # jumper
        flipat_r = rand(c.rng, 1:nsites(row(σ)))
        flipat_c = rand(c.rng, 1:nsites(col(σ)))

        old_val_r, _ =flipat!(c.rng, row(σ), flipat_r)
        old_val_c, _ =flipat!(c.rng, col(σ), flipat_c)
        prob = exp(2*real(net(σ)) - logψ_σ)
        rv = rand(c.rng)
        if !(rv < prob)
            setat!(row(σ), flipat_r, old_val_r)
            setat!(col(σ), flipat_c, old_val_c)
        else c.steps_accepted += 1 end
    elseif move_type == 9
        if rand(c.rng, 1:100) == 1
            zero!(σ)
        end
        #prob = exp(2*real(net(σ)) - logψ_σ)
        #rv = rand(c.rng)
        #if !(rv < prob)
        #    setat!(row(σ), flipat_r, old_val_r)
        #    setat!(col(σ), flipat_c, old_val_c)
        #else
            c.steps_accepted += 1
        #end
    end

    c.steps_done += 1
    return true
end
