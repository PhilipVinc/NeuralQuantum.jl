function QuantumOptics.expect(Obs::Union{GraphOperator, SparseOperator},
                     net::NeuralNetwork, sampler::Sampler=FullSumSampler())

    oprob = ObservablesProblem(NeuralQuantum.input_type(net), Obs)
    ov          = state(oprob, net)
    mosampl     = multithread(sampler);
    oitercache  = SamplingCache(oprob)
    moitercache = [SamplingCache(oprob) for i=1:Threads.nthreads()]
    moc         = init_sampler!(mosampl, net, ov)

    sample_net = sampler isa FullSumSampler ? sample_network_wholespace! : sample_network!
    #println(sample_net)
    init_sampler!(mosampl, net, ov, moc)
    #Threads.@threads
    for i=1:Threads.nthreads()
        s = sampler_list(mosampl)[i]
        n = moc.nets[i]
        v = moc.σs[i]
        _ic = moitercache[i]
        zero!(_ic)
        while true
            sample_net(_ic, oprob, n, v)
            !samplenext!(v, s, n, moc.caches[i]) && break
        end
    end
    zero!(oitercache)
    for ic=moitercache
        add!(oitercache, ic)
    end
    ObsEval = evaluation_post_sampling(oitercache)
    if sampler isa FullSumSampler
        return ObsEval.ObsAve[1], ObsEval.ObsVals[1]
    else
        return ObsEval.ObsAve[1], ObsEval.ObsVals[1]
    end
end
