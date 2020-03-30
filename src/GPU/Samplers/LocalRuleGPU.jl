struct LocalRuleGPUCache{R,T}
    rng::R
    σ_cpu::T
end

function RuleSamplerCache(r::LocalRule, s::MetropolisSampler, v::Union{gpuAStateBatch,gpuADoubleStateBatch},
                          net, part)
    σ_cpu = state_collect(v)
    rng = build_rng_generator_T(σ_cpu, s.seed)
    return LocalRuleGPUCache(rng, σ_cpu)
end

function init_sampler_rule_cache!(rc::LocalRuleGPUCache, s, net, σ, c)

end

# Efficient step for Local Metropolis sampling on the gpu
# avoids scalar operation on gpu by copying the state to the cpu,
# modyfing it there, then copying it back to gpu
#TODO Improve by creating a gpu kernel that does not require copies
# @timbesard suggests to pre-generate the random data with
# CuArrays.rand
function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{LocalRule},
                       net::Union{MatrixNet,KetNet}, c, rc::LocalRuleGPUCache)
    σ_cpu = rc.σ_cpu
    copy!(σ_cpu, σp)

    #switch the cpu buffer
    sites_to_flip = 1:nsites(c.hilb)
    for i=1:num_batches(σp)
        flipat = rand(rc.rng, sites_to_flip)
        flipat!(rc.rng, unsafe_get_batch(σ_cpu, i), c.hilb, flipat)
    end

    copy!(σp, σ_cpu)
end
