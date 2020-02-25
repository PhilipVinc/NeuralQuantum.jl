using NeuralQuantum, Test
using NeuralQuantum: set_index!
num_types = [Float32, Float64]
atol_types  = [1e-5, 1e-8]

machines = Dict()

ma = (T, N) -> RBMSplit(T, N, 2)
machines["RBMSplit"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ)
machines["RBM_softplus"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ2)
machines["RBM_cosh"] = ma

ma = (T, N) -> NDM(T, N, 1, 2, NeuralQuantum.logℒ)
machines["NDM_softplus"] = ma

ma = (T, N) -> NDM(T, N, 1, 2, NeuralQuantum.logℒ2)
machines["NDM_cosh"] = ma

graph = HyperCube([N], periodic=true)
symm  = translational_symm_table(graph)
ma = (T, N) -> NDMSymm(T, N, 1, 2, symm, NeuralQuantum.logℒ2)
machines["NDMSymm_cosh"] = ma

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
    ch = Chain(Dense(T, N, 2*N, af_logcosh), Dense(T, 2*N, N, af_logcosh), sum_autobatch)
    return PureStateAnsatz(ch, N)
end
machines["chain_pure_cosh_2"] = ma

ma = (T, N) -> begin
    ch = Chain(DenseSplit(T, N, 2*N, af_softplus), sum_autobatch)
    return MixedStateAnsatz(ch, N)
end
#machines["chain_mixed_softplus"] = ma



N = 4
T = Float32

@testset "test cached dispatch - values: $name" for name=keys(machines)
    net = machines[name](T,N)
    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end
    cnet = cached(net)
    v = state(T, hilb, net)

    @test net(v) ≈ cnet(v)
    if !(v isa AbstractVector)
        @test net(v...) ≈ cnet(v...)
    end
end

@testset "test cached dispatch - allocating gradients: $name" for name=keys(machines)
    net = machines[name](T,N)

    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end
    cnet = cached(net)
    v = state(T, hilb, net)

    @test ∇logψ(net, v) ≈ ∇logψ(cnet, v)
end

@testset "test cached dispatch - inplace gradients: $name" for name=keys(machines)
    net = machines[name](T,N)

    hilb = HomogeneousFock(N, 2)
    if net isa NeuralQuantum.MatrixNet
        hilb = SuperOpSpace(hilb)
    end
    cnet = cached(net)
    v = state(T, hilb, net)

    g1 = grad_cache(net)
    g2 = grad_cache(net)

    v1, gg1 = logψ_and_∇logψ!(g1, net, v)
    v2, gg2 = logψ_and_∇logψ!(g2, cnet, v)
    @test v1 ≈ v2
    @test gg1 ≈ gg2
    @test g1 === gg1 && g2 === gg2

    v1, gg1 = logψ_and_∇logψ!(g1, net, v)
    v2, gg2 = logψ_and_∇logψ!(g2, cnet, v)
    @test v1 ≈ v2
    @test gg1 ≈ gg2
    @test g1 === gg1 && g2 === gg2
end
