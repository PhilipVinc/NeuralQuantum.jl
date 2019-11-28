using NeuralQuantum, Test
using NeuralQuantum: set_index!
num_types = [Float32, Float64]

machines = Dict()

ma = (T, N) -> RBMSplit(T, N, 2)
machines["RBMSplit"] = ma

ma = (T, N) -> RBM(T, N, 2)
machines["RBM"] = ma

N = 4
T = Float32

@testset "test cached dispatch - values: $name" for name=keys(machines)
    net = machines[name](T,N)

    cnet = cached(net)
    v = state(T, SpinBasis(1//2)^N, net)
    arr_v = config(v)

    @test net(v) ≈ cnet(v)
    @test net(arr_v) ≈ cnet(arr_v)
    if !(arr_v isa AbstractVector)
        @test net(arr_v...) ≈ cnet(arr_v...)
    end
end

@testset "test cached dispatch - allocating gradients: $name" for name=keys(machines)
    net = machines[name](T,N)

    cnet = cached(net)
    v = state(T, SpinBasis(1//2)^N, net)
    arr_v = config(v)

    @test ∇logψ(net, v) ≈ ∇logψ(cnet, v)
    @test ∇logψ(net, arr_v) ≈ ∇logψ(cnet, arr_v)
    if !(arr_v isa AbstractVector)
        #@test ∇logψ(net, arr_v...) ≈ ∇logψ(cnet, arr_v...)
    end
end

@testset "test cached dispatch - inplace gradients: $name" for name=keys(machines)
    net = machines[name](T,N)

    cnet = cached(net)
    v = state(T, SpinBasis(1//2)^N, net)
    arr_v = config(v)
    g1 = grad_cache(net)
    g2 = grad_cache(net)

    v1, gg1 = logψ_and_∇logψ!(g1, net, v)
    v2, gg2 = logψ_and_∇logψ!(g2, cnet, v)
    @test v1 ≈ v2
    @test gg1 ≈ gg2
    @test g1 === gg1 && g2 === gg2

    v1, gg1 = logψ_and_∇logψ!(g1, net, arr_v)
    v2, gg2 = logψ_and_∇logψ!(g2, cnet, arr_v)
    @test v1 ≈ v2
    @test gg1 ≈ gg2
    @test g1 === gg1 && g2 === gg2

    if !(arr_v isa AbstractVector)
        #@test ∇logψ(net, arr_v...) ≈ ∇logψ(cnet, arr_v...)
    end
end
