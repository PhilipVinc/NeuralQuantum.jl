function BatchedSampler(net, sampler, prob, algo=prob;
                        kwargs...)
    if net isa CachedNet
        throw("Only takes standard network.")
    end

    if prob isa HamiltonianGSEnergyProblem
        return BatchedValSampler(net, sampler, prob, algo; kwargs...)
    elseif prob isa LRhoKLocalSOpProblem
        return BatchedGradSampler(net, sampler, prob, algo; kwargs...)
    elseif prob isa ObservablesProblem
        return BatchedObsDMSampler(net, sampler, prob, algo; kwargs...)
    else
        throw("not known")
    end
end
