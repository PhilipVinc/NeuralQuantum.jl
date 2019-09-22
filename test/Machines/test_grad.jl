using NeuralQuantum, Test
using NeuralQuantum: set_index!
num_types = [Float32, Float64]

re_machines = Dict()
im_machines = Dict()

ma = (T, N) -> NDM(T, N, 2, 3)
re_machines["NDM"] = ma

ma = (T, N) -> RBMSplit(T, N, 2)
im_machines["RBMSplit"] = ma

all_machines = merge(re_machines, im_machines)

N = 4

@testset "Test Properties $name" for name=keys(all_machines)
    for T=num_types
        if name ∈ keys(im_machines)
            T = Complex{T}
        end
        net = all_machines[name](T,N)
        cnet = cached(net)

        @test NeuralQuantum.input_type(net) == real(T)
        @test NeuralQuantum.out_type(net) == Complex{real(T)}
        @test NeuralQuantum.input_shape(net) == (N, N)
        @test NeuralQuantum.is_analytic(net)
    end
end


@testset "Test Cached Value $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        cnet = cached(net)

        v = state(T, SpinBasis(1//2)^N, net)
        # compute exact
        vals = []; cvals = [];
        for i=1:spacedimension(v)
            set_index!(v, i)
            push!(vals, net(v))
            push!(cvals, cnet(v))
        end
        @test vals ≈ cvals
    end
end


@testset "Test Cached Gradient $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        cnet = cached(net)
        cder = grad_cache(net)

        v = state(T, SpinBasis(1//2)^N, net)
        # compute exact
        grads = [];
        for i=1:spacedimension(v)
            set_index!(v, i)
            der_ad = ∇logψ(net,  v)
            ∇logψ!(cder, cnet, v)
            for f=propertynames(der_ad)
                ∇  = getproperty(der_ad,  f)
                c∇ = getproperty(cder, f)
                push!(grads, ∇ ≈ c∇)
            end
        end
        @test all(grads)
    end
end
