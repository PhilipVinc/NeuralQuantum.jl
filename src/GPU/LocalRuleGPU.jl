struct LocalRuleGPUCache{T}
    σ_cpu::T
end

function RuleSamplerCache(r::LocalRule, s::MetropolisSampler, v::Union{gpuAStateBatch,gpuADoubleStateBatch},
                          net, part)
    σ_cpu = statecollect(v)
    return LocalRuleGPUCache(σ_cpu)
end

function init_sampler_rule_cache!(rc::LocalRuleGPUCache, s, net, σ, c)

end

function propose_step!(σp::Union{AStateBatch,ADoubleStateBatch}, s::MetropolisSampler{LocalRule},
                       net::Union{MatrixNet,KetNet}, c, rc::LocalRuleGPUCache)
    σ_cpu = rc.σ_cpu
    copy!(σ_cpu, σp)

    #switch the cpu buffer
    propose_step!(σ_cpu, s, net, c, nothing)

    copy!(σp, σ_cpu)
end
