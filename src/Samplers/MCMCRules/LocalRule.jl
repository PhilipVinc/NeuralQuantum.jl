## LocalRule
export LocalRule

"""
    LocalRule()

Transition rule for Metropolis-Hastings sampling where at every step a random
site is switched to another random state.
"""
struct LocalRule <: MCMCRule
end

function propose_step!(σp::Union{AState,ADoubleState}, s::MetropolisSampler{LocalRule},
                       net::NeuralNetwork, c, rc)
    flipat = rand(c.rng, 1:nsites(c.hilb))
    @inbounds old_val, new_val = flipat!(c.rng, c.σp, c.hilb, flipat)
end

function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch},
                       s::MetropolisSampler{LocalRule},
                       net::NeuralNetwork, c, rc)
    sites_to_flip = 1:nsites(c.hilb)

    for σ=states(σp)
        flipat = rand(c.rng, sites_to_flip)
        @inbounds old_val, new_val = flipat!(c.rng, σ, c.hilb, flipat)
    end
end
