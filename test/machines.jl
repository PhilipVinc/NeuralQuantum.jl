using NeuralQuantumBase, Test
num_types = [Float32, Float64]

@testset "NDM" begin
    N = 4
    for T=num_types
        net  = NDM(T,N,2,3)
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
        @test NeuralQuantumBase.input_type(net) == real(T)
        @test NeuralQuantumBase.out_type(net) == Complex{real(T)}
        @test NeuralQuantumBase.input_shape(net) == (N, N)
        @test NeuralQuantumBase.is_analytic(net)
    end

end


@testset "RBMsplit" begin
    N = 4
    for T=num_types
        T=Complex{T}
        net  = RBMSplit(T,N,2)
        cnet = cached(net)
        cder = grad_cache(cnet)

        v    = DoubleState(NAryState(T, 2, N))

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
        @test NeuralQuantumBase.input_type(net) == real(T)
        @test NeuralQuantumBase.out_type(net) == Complex{real(T)}
        @test NeuralQuantumBase.input_shape(net) == (N, N)
        @test NeuralQuantumBase.is_analytic(net)
    end

end

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
        @test NeuralQuantumBase.input_type(net) == real(T)
        @test NeuralQuantumBase.out_type(net) == Complex{real(T)}
        @test NeuralQuantumBase.input_shape(net) == (N, N)
        @test NeuralQuantumBase.is_analytic(net)
    end

end
=#
