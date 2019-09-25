## MEtropolis
export Metropolis
struct Metropolis <: MCMCRule
    num_switches::Int

    # Inner constructor to assert that the number of switches is always odd.
    Metropolis(i) = begin
        if iseven(i)
            @warn """
                Metropolis sampling with an even number of switches per step
                is known to be non-ergodic, because it can potentially never
                change the configuration.
                Setting the number of switches to $(i+1).
                """
            i += 1
        end
        return new(i)
    end
end

"""
    Metropolis([n_switches=1])

Metropolis rule for Markov Chain Monte Carlo sampling where at every step
`n_switches` sites are switched.

The number of switches must be odd, otherwise there might
"""
Metropolis() = Metropolis(1)

function markov_chain_step!(σ, s::MCMCSampler{Metropolis}, net::Union{MatrixNet,KetNet}, c)
    # the denominator
    logψ_σ = 2*real(net(σ))

    for i=1:s.rule.num_switches
        # 1- Flip a spin
        sites_to_flip = 1:nsites(σ)
        flipat = rand(c.rng, sites_to_flip)
        old_val, new_val =flipat!(c.rng, σ, flipat)

        # 2- Compute the probability to be in that state
        logψ_σp = 2*real(net(σ)) #log_prob_ψ(net, σ)
        log_ratio = logψ_σp - logψ_σ
        prob = exp(log_ratio)
        rv = rand(c.rng)

        # If the move is rejected, revert it
        if !(rv < prob)
            setat!(σ, flipat, old_val)
        else # else accept it
            logψ_σ = logψ_σp
            c.steps_accepted += 1
        end
    end
    c.steps_done += 1

    return true
end

function markov_chain_step!(σ::LUState, s::MCMCSampler{Metropolis}, net::Union{MatrixNet,KetNet}, c)
    # Switch num_switches times
    for i=1:s.rule.num_switches
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
    end
    c.steps_done += 1

    return true
end
