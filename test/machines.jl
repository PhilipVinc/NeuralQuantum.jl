using NeuralQuantum, Test
num_types = [Float32, Float64]

#=
@testset "NDMSymm" begin
    N = 4
    permutations = [[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]
    for T=num_types
        net  = NDMSymm(T,N,2,3, permutations)
        cnet = cached(net)
        v    = DoubleState(NAryState(T, 2, N))

        cder = grad_cache(cnet)
        vals = []; cvals = [];
        grads = [];
        for i=1:spacedimension(v)
            set_index!(v, i)
            push!(vals, net(v))
            push!(cvals, cnet(v))

            der  = ∇logψ(net,  v)
            ∇logψ!(cder, cnet, v)
            for f=fieldnames(typeof(der))
                ∇  = getfield(der,  f)
                c∇ = getfield(cder, f)
                push!(grads, ∇ ≈ c∇)
            end
        end
        @test vals ≈ cvals
        @test all(grads)
        @test NeuralQuantum.input_type(net) == real(T)
        @test NeuralQuantum.out_type(net) == Complex{real(T)}
        @test NeuralQuantum.input_shape(net) == (N, N)
        @test NeuralQuantum.is_analytic(net)
    end

end
=#
