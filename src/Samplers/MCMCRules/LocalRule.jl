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
                       net::Union{MatrixNet,KetNet}, c, rc)
    flipat = rand(c.rng, 1:nsites(c.hilb))
    old_val, new_val =flipat!(c.rng, c.σp, c.hilb, flipat)
end

function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{LocalRule},
                       net::Union{MatrixNet,KetNet}, c, rc)
    sites_to_flip = 1:nsites(c.hilb)

    for i=1:num_batches(σp)
        flipat = rand(c.rng, sites_to_flip)
        old_val, new_val = flipat!(c.rng, unsafe_get_batch(σp, i), c.hilb, flipat)
    end
end
