## LocalRule
export LocalRule

"""
    LocalRule([n_switches=1])

LocalRule rule for Markov Chain Monte Carlo sampling where at every step
`n_switches` sites are switched.

The number of switches must be odd, otherwise there might
"""
struct LocalRule <: MCMCRule
end

function propose_step!(σp::Union{AState,ADoubleState}, s::MetropolisSampler{LocalRule},
                       net::Union{MatrixNet,KetNet}, c)
    flipat = rand(c.rng, 1:nsites(c.hilb))
    old_val, new_val =flipat!(c.rng, c.σp, c.hilb, flipat)
end

function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{LocalRule},
                       net::Union{MatrixNet,KetNet}, c)
    sites_to_flip = 1:nsites(c.hilb)

    for i=1:num_batches(σp)
        flipat = rand(c.rng, sites_to_flip)
        old_val, new_val = flipat!(c.rng, unsafe_get_batch(σp, i), c.hilb, flipat)
    end
end
