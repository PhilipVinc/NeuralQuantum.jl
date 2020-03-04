using NeuralQuantum, Test
using NeuralQuantum: set_index!, trainable_first
using NeuralQuantum: out_similar, unsafe_get_batch
num_types = [Float32, Float64]
atol_types  = [1e-5, 1e-8]

machines = Dict()

ma = (T, N) -> RBMSplit(T, N, 2)
machines["RBMSplit"] = ma

ma = (T, N) -> NDM(T, N, 2, 3, NeuralQuantum.logℒ)
machines["NDM_softplus"] = ma

ma = (T, N) -> NDM(T, N, 2, 3, NeuralQuantum.logℒ2)
machines["NDM_cosh"] = ma

ma = (T, N) -> NDMSymm(T, N, 1, 2, translational_symm_table(HyperCube([N], periodic=true)), NeuralQuantum.logℒ2)
machines["NDMSymm_cosh"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ)
machines["RBM_softplus"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ2)
machines["RBM_cosh"] = ma

ma = (T, N) -> begin
    ch = Chain(Dense(T, N, N-1, af_softplus), Dense(T, N-1, N-2, af_softplus), WSum(T, N-2))
    return PureStateAnsatz(ch, N)
end
machines["chain_pure"] = ma

ma = (T, N) -> begin
    ch = Chain(Dense(T, N, 2*N, af_softplus), sum_autobatch)
    return PureStateAnsatz(ch, N)
end
machines["chain_pure_softplus"] = ma

ma = (T, N) -> begin
    ch = Chain(Dense(T, N, 2*N, af_logcosh), sum_autobatch)
    return PureStateAnsatz(ch, N)
end
machines["chain_pure_cosh"] = ma

ma = (T, N) -> begin
    ch = Chain(DenseSplit(T, N, 2*N, af_softplus), sum_autobatch)
    return MixedStateAnsatz(ch, N)
end
machines["chain_mixed_softplus"] = ma

N = 4
T = Float32
b_sz = 3

@testset "test batched dispatch - values: $name" for name=keys(machines)
    name == "NDMSymm_cosh" && continue

    net = machines[name](T,N)
    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end
    cnet = cached(net, b_sz)
    v  = state(T, hilb, net)
    vb = state(T, hilb, net, b_sz)

    if vb isa Tuple
        rand!.(vb)
    else
        rand!(vb)
    end

    @test net(vb) ≈ cnet(vb)
    if (vb isa Tuple)
        @test net(vb...) ≈ cnet(vb...)
    end

    o = rand(Complex{T}, 1, b_sz)
    o2 = similar(o)

    oo  = logψ!(o, net, vb)
    oo2 = logψ!(o2, cnet, vb)
    @test oo === o
    @test oo2 === o2
    @test oo ≈ oo2
end

@testset "test cached dispatch - inplace gradients: $name" for name=keys(machines)
    net = machines[name](T,N)

    snet = cache(net)
    cnet = cached(net, b_sz)
    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end

    v  = state(T, hilb, net)
    vb = state(T, hilb, net, b_sz)

    if vb isa Tuple
        rand!.(vb)
    else
        rand!(vb)
    end

    g1 = grad_cache(net, b_sz)
    g2 = grad_cache(net, b_sz)
    g3 = grad_cache(net, b_sz)

    v1, gg1 = logψ_and_∇logψ!(g1, cnet, vb)
    v3 = similar(v1)
    vv3, gg3 = logψ_and_∇logψ!(g3, v3, cnet, vb)
    @test v1 ≈ v3
    @test gg1 ≈ gg3
    @test g3 === gg3
    @test v3 === vv3

    if vb isa Tuple
        v2, gg2 = logψ_and_∇logψ!(g2, cnet, vb...)
        @test v1 ≈ v2
        @test gg1 ≈ gg2
        @test g1 === gg1 && g2 === gg2

        vv3, gg3 = logψ_and_∇logψ!(g3, v3, cnet, vb...)
        @test v1 ≈ v3
        @test gg1 ≈ gg3
        @test g3 === gg3
        @test v3 === vv3
    end
end


@testset "test batched logψ: $name" for name=keys(machines)
    net = machines[name](T,N)

    cnet = cached(net)
    bnet = cached(net, b_sz)
    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end

    v  = state(T, hilb, net)
    vb = state(T, hilb, net, b_sz)

    vals   = out_similar(bnet)
    vals_2 = out_similar(bnet)
    res    = Bool[]
    for batch_i=Iterators.partition(1:spacedimension(hilb), b_sz)
        for (i, h_i)=enumerate(batch_i)
            set!(unsafe_get_batch(vb, i), hilb, h_i)
        end
        logψ!(vals, bnet, vb)

        for (i, h_i)=enumerate(batch_i)
            set!(v, hilb, h_i)
            vals_2[i] = logψ(cnet, v)
        end
        push!(res, vals_2 ≈ vals)
    end
    @test all(res)

end


@testset "test batched ∇ψ: $name" for name=keys(machines)
    name == "NDMSymm_cosh" && continue

    net = machines[name](T,N)

    cnet = cached(net)
    bnet = cached(net, b_sz)
    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end

    v  = state(T, hilb, net)
    vb = state(T, hilb, net, b_sz)

    vals   = out_similar(bnet)
    vals_2 = out_similar(bnet)

    gb = grad_cache(net, b_sz)
    g2 = grad_cache(net)
    gtmp = grad_cache(net)
    res    = Bool[]
    res_grad = Bool[]
    for batch_i=Iterators.partition(1:spacedimension(hilb), b_sz)
        for (i, h_i)=enumerate(batch_i)
            set!(unsafe_get_batch(vb, i), hilb, h_i)
        end
        logψ_and_∇logψ!(gb, vals, bnet, vb)

        for (i, h_i)=enumerate(batch_i)
            set!(v, hilb, h_i)
            vals_2[i] , _ = logψ_and_∇logψ!(g2, cnet, v)

            for (x,y)=zip(vec_data(gtmp), vec_data(gb))
                x .= y[:,i]
            end
            push!(res_grad, gtmp ≈ g2)
        end
        push!(res, vals_2 ≈ vals)
    end
    @test all(res)
    @test all(res_grad)
end
