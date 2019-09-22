using NeuralQuantum, Test
using NeuralQuantum: random_input_state

num_types = [Float32, Float64]

machines = Dict()

ma = (T, N) -> NDMComplex(T, N, 2, 3)
machines["NDMComplex"] = ma

N = 4

@testset "Test Properties $name" for name=keys(machines)
    for T=num_types
        T = Complex{T}
        net = machines[name](T,N)
        cnet = cached(net)

        @test NeuralQuantum.input_type(net) == real(T)
        @test NeuralQuantum.out_type(net) == Complex{real(T)}
        @test NeuralQuantum.input_shape(net) == (N, N)
        @test NeuralQuantum.is_analytic(net) == false
    end
end


@testset "Test Cached Value $name" for name=keys(machines)
    for T=num_types
        net = machines[name](Complex{T},N)
        cnet = cached(net)

        v = state(real(T), SpinBasis(1//2)^N, net)
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
