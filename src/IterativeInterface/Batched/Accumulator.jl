
function Accumulator(net, prob, batch_sz)
    if prob isa HamiltonianGSEnergyProblem
        return LocalKetAccumulator(net, state(prob, net), batch_sz)
    #elseif prob isa HamiltonianGSVarianceProblem
    #    return LocalGradAccumulator(net, state(net, prob), batch_sz)
    elseif prob isa LRhoKLocalSOpProblem
        return LocalGradAccumulator(net, state(prob, net), batch_sz)
    else
        throw("problem not handled")
    end
end
