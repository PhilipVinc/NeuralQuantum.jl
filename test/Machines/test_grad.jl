using NeuralQuantum, Test
using NeuralQuantum: set_index!
num_types = [Float32, Float64]

re_machines = Dict()
im_machines = Dict()

ma = (T, N) -> RBMSplit(T, N, 2)
im_machines["RBMSplit"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ)
im_machines["RBM_softplus"] = ma

ma = (T, N) -> RBM(T, N, 2, NeuralQuantum.logℒ2)
im_machines["RBM_cosh"] = ma

ma = (T, N) -> NDM(T, N, 1, 2, NeuralQuantum.logℒ)
re_machines["NDM_softplus"] = ma

ma = (T, N) -> NDM(T, N, 1, 2, NeuralQuantum.logℒ2)
re_machines["NDM_cosh"] = ma

graph = HyperCube([N], periodic=true)
symm  = translational_symm_table(graph)
ma = (T, N) -> NDMSymm(T, N, 1, 2, symm, NeuralQuantum.logℒ2)
re_machines["NDMSymm_cosh"] = ma

ma = (T, N) -> PureStateAnsatz(Chain(Dense(T, N, N*2), Dense(T, N*2, N*3), WSum(T, N*3)), N)
re_machines["ChainKet"] = ma

ma = (T, N) -> begin
    ch = Chain(Dense(T, N, 2*N, af_softplus), sum_autobatch)
    return PureStateAnsatz(ch, N)
end
im_machines["chain_pure_softplus"] = ma

ma = (T, N) -> begin
    ch = Chain(Dense(T, N, 2*N, af_logcosh), Dense(T, 2*N, N, af_logcosh), sum_autobatch)
    return PureStateAnsatz(ch, N)
end
im_machines["chain_pure_cosh_2"] = ma

ma = (T, N) -> begin
    ch = Chain(DenseSplit(T, N, 2*N, af_softplus), sum_autobatch)
    return MixedStateAnsatz(ch, N)
end
im_machines["chain_mixed_softplus"] = ma

all_machines = merge(re_machines, im_machines)

N = 4

@testset "Test Properties $name" for name=keys(all_machines)
    for T=num_types
        if name ∈ keys(im_machines)
            T = Complex{T}
        end
        name == "ChainKet" && T != Float32 && continue

        net = all_machines[name](T,N)

        @test NeuralQuantum.out_type(net) == Complex{real(T)}
        @test NeuralQuantum.is_analytic(net)
    end
end


@testset "Test cached evaluation $name" for name=keys(all_machines)
    for T=num_types
        name == "ChainKet" && T != Float32 && continue

        net = all_machines[name](T,N)
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
        # Test evaluation
        @test vals ≈ cvals
        # Test cached vals have the same type
        all(typeof.(cvals) .== typeof(first(cvals)))
        # and same type as exact computation
        if all(imag(vals) .== 0)
            @test real(typeof(first(vals))) == real(typeof(first(cvals)))
        else
            @test typeof(first(vals)) == typeof(first(cvals))
        end
    end
end


@testset "Test cached gradient $name" for name=keys(all_machines)
    for T=num_types
        name == "ChainKet" && T != Float32 && continue
        name == "NDMSymm_cosh" && continue

        net = all_machines[name](T,N)
        cnet = cached(net)

        hilb = HomogeneousFock(N, 2)
        if net isa NeuralQuantum.MatrixNet
            hilb = SuperOpSpace(hilb)
        end

        cder = grad_cache(net)

        v = state(T, hilb, net)
        # compute exact
        grads = [];
        for i=1:spacedimension(hilb)
            set!(v, hilb, i)
            der_ad = ∇logψ(net,  v)
            ∇logψ!(cder, cnet, v)
            #=for f=propertynames(der_ad)
                ∇  = getproperty(der_ad,  f)
                c∇ = getproperty(cder, f)
                push!(grads, ∇ ≈ c∇)
            end=#
            push!(grads, der_ad ≈ cder)
        end
        @test all(grads)
    end
end
