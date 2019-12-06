using NeuralQuantum, Test

num_types = [Float32, Float64]

machines = Dict()

ma = (T, N) -> NDMComplex(T, N, 2, 3)
machines["NDMComplex"] = ma

N = 4

@testset "Test Properties $name" for name=keys(machines)
    for T=num_types
        net = machines[name](T,N)
        cnet = cached(net)

        @test NeuralQuantum.out_type(net) == Complex{real(T)}
        @test NeuralQuantum.is_analytic(net) == false
    end
end


@testset "Test Cached Value $name" for name=keys(machines)
    for T=num_types
        net = machines[name](T,N)
        cnet = cached(net)

        hilb = HomogeneousFock(N, 2)
        if net isa NeuralQuantum.MatrixNet
            hilb = SuperOpSpace(hilb)
        end

        v = state(T, hilb, net)
        # compute exact
        vals = []; cvals = [];
        for i=1:spacedimension(hilb)
            set!(v, hilb, i)
            push!(vals, net(v))
            push!(cvals, cnet(v))
        end
        @test vals ≈ cvals
    end
end
