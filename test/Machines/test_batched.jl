using NeuralQuantum, Test
using NeuralQuantum: set_index!, trainable_first, preallocate_state_batch
num_types = [Float32, Float64]

machines = Dict()

ma = (T, N) -> RBMSplit(T, N, 2)
machines["RBMSplit"] = ma

ma = (T, N) -> NDM(T, N, 2, 3)
machines["NDM"] = ma

ma = (T, N) -> RBM(T, N, 2)
im_machines["RBM_softplus"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ2)
im_machines["RBM_cosh"] = ma

N = 4
T = Float32
b_sz = 3

@testset "test batched dispatch - values: $name" for name=keys(machines)
    net = machines[name](T,N)

    cnet = cached(net, b_sz)
    v = state(T, SpinBasis(1//2)^N, net)
    vb = preallocate_state_batch(trainable_first(net), T,
                             v, b_sz)

    rand!(vb, hilb)

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

    cnet = cached(net, b_sz)
    v = state(T, SpinBasis(1//2)^N, net)
    vb = preallocate_state_batch(trainable_first(net), T,
                             v, b_sz)
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
