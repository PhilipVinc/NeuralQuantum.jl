function BatchedSampler(net, sampler, prob, algo=Gradient();
                        kwargs...)
    if net isa CachedNet
        throw("Only takes standard network.")
    end

    if prob isa KLocalLiouvillian
        return BatchedGradSampler(net, sampler, prob, algo; kwargs...)
    elseif prob isa AbsLinearOperator
        return BatchedValSampler(net, sampler, prob, algo; kwargs...)
    else
        throw("not known")
    end
end
