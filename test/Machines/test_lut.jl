using NeuralQuantum, Test
using NeuralQuantum: lookup, init_lut!, flipat_fast!, update_lut!, logψ_Δ, lut
num_types = [Float32, Float64]

re_machines = Dict()
im_machines = Dict()

ma = (T, N) -> NDM(T, N, 2, 3)
re_machines["NDM"] = ma

ma = (T, N) -> RBMSplit(T, N, 2)
im_machines["RBMSplit"] = ma

all_machines = merge(re_machines, im_machines)

N = 4

@testset "Test Update_LUT $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)
        # compute exact
        results = [];
        for i=1:spacedimension(v)
            set_index!(v, i)
            init_lut!(v, cnet)

            for j=1:nsites(v)
                old, new = flipat_fast!(v, j)
            end
            update_lut!(v, cnet)
            u_lut = deepcopy(lut(v))
            init_lut!(v, cnet)
            i_lut = lut(v)
            good = Bool[]
            for f=fieldnames(typeof(i_lut))
                push!(good, getfield(u_lut, f) ≈ getfield(i_lut, f))
            end
            push!(results, all(good))
        end
        @test all(results)
    end
end

@testset "Test Logψ_Δ $name" for name=keys(all_machines)
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)

        vr = state(T, SpinBasis(1//2)^N, net)
        zero!(vr)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)
        # compute exact
        Δ_vals = []; exact_vals = [];
        eval_vals = []; exact = [];
        for i=1:spacedimension(v)
            set_index!(vr, i)
            set_index!(v, i)
            init_lut!(v, cnet)

            lnψ_old = cnet(vr)
            for j=1:nsites(v)
                old, new = flipat_fast!(v, j)
                setat!(vr, j, new)
            end
            lnψ_new  = cnet(vr)
            lnψ_Δ    = logψ_Δ(cnet, v)
            push!(Δ_vals, lnψ_Δ)
            push!(exact_vals, lnψ_new-lnψ_old)
            push!(eval_vals, logψ(cnet, v))
            push!(exact, lnψ_new)
        end
        @test Δ_vals ≈ exact_vals
        @test eval_vals ≈ exact
    end
end


@testset "Test Cached Gradient $name" for name=keys(merge(re_machines, im_machines))
    for T=num_types
        net = all_machines[name](T,N)
        isnothing(lookup(net)) && continue

        cnet = cached(net)
        cder = grad_cache(net)

        vr = state(T, SpinBasis(1//2)^N, net)
        zero!(vr)

        v = state_lut(T, SpinBasis(1//2)^N, net)
        zero!(v)
        init_lut!(v, cnet)

        # compute exact
        grads = [];
        for i=1:spacedimension(v)
            set_index!(vr, i)
            set_index!(v, i)
            init_lut!(v, cnet)

            for j=1:nsites(v)
                old, new = flipat_fast!(v, j)
                setat!(vr, j, new)
            end

            der_c = ∇logψ(cnet, vr)
            der_l = ∇logψ(cnet, v)
            for f=fieldnames(typeof(der_l))
                ∇  = getfield(der_c,  f)
                c∇ = getfield(der_l, f)
                push!(grads, ∇ ≈ c∇)
            end
        end
        @test all(grads)
    end
end
