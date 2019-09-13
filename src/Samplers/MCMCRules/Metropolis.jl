## MEtropolis
export Metropolis
struct Metropolis <: MCMCRule end

function markov_chain_step!(σ, s::MCMCSampler{Metropolis}, net::MatrixNet, c)
    # the denominator
    logψ_σ = 2*real(net(σ))

    # 1- Flip a spin
    sites_to_flip = 1:nsites(σ)
    flipat = rand(c.rng, sites_to_flip) #1:nsites(σ))
    old_val, new_val =flipat!(c.rng, σ, flipat)

    # 2- Compute the probability to be in that state
    logψ_σp = 2*real(net(σ)) #log_prob_ψ(net, σ)
    log_ratio = logψ_σp - logψ_σ
    prob = exp(log_ratio)
    rv = rand(c.rng)
    # If the move is rejected, revert it
    if !(rv < prob)
        setat!(σ, flipat, old_val)
    else
        #logψ_σ = 2*real(net(σ)) #log_prob_ψ(net, σ, alg_cache)
        c.steps_accepted += 1
    end
    c.steps_done += 1

    return true
end

function markov_chain_step!(σ::LUState, s::MCMCSampler{Metropolis}, net::MatrixNet, c)
    # 1- Flip a spin
    sites_to_flip = 1:nsites(σ)
    flipat = rand(c.rng, sites_to_flip) #1:nsites(σ))
    old_val, new_val = flipat_fast!(c.rng, σ, flipat)

    # 2- Compute the probability to be in that state
    log_ratio = 2*real(logψ_Δ(net, σ))
    prob = exp(log_ratio)
    rv = rand(c.rng)
    # If the move is rejected, revert it
    if !(rv < prob)
        clear_changes!(σ)
    else
        update_lut!(σ, net)
        c.steps_accepted += 1
    end
    c.steps_done += 1

    return true
end
